"""Freeze and collect six train-only initial states, two scales and three pulses."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from check_s3_actions import ROOT, digest, schedule_actions, write_json
from check_s3_deterministic_archive_import import backend_preflight
from collect_s3_small_action_response import monitor, storage_admission
from preflight_s3_training_timing_capture import sequence_digest
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.controller import OnlineTiltController
from ball_retention.deterministic_archive_import_plan import load as load_archive
from ball_retention.geometry import load_tool, sphere_clearance, support_point
from ball_retention.guard import process_memory
from ball_retention.pair_collection import _initial_key
from ball_retention.prescribed import PrescribedPath
from ball_retention.resources import admission, directory_bytes, snapshot
from ball_retention.robot import Robot
from ball_retention.timed_pair_data import SCHEMA, prepare_timed_pairs
from archive_restore import write
from deterministic_archive_metrics import _full_branch
from native_repeatability import _make_env, _record_nominal
from saved_native_snapshot import restore_and_readback, verify_live_layout
from training_timing_capture import nominal_gate, signal_report


BASE = ROOT/'runs/s3_broad_training_20261118'
TEMPLATE = 'runs/s3_history_coverage_20261114/history_train_yplus004_advance_config.json'
FIT61 = 'configs/s3_world_increment_data61_fit_plan.json'
START = 'configs/s1_online_episode.json'
START_SHA = '25bfeaef2ea3c1c876a5f17c6c25e46295aded0ffd5b0d030e1d3411a227f89d'
HORIZON, SUBSTEPS = 32, 16
TOTAL_STEPS = 54528
LIMITS = {'max_wall_s': 3000, 'max_new_output_mib': 128}
RUNTIME = {'newton': '1.6.0', 'warp': '1.17.0', 'device': 'cuda:0', 'cuda_arch': 89,
           'collision_pipeline_deterministic': True, 'xpbd_deterministic_mode': 'RUN_TO_RUN'}
SCOPE = {'training_updates': 0, 'test_episode_files_read': False, 'control_accepted': False}
AMPLITUDES = {'005': .05, '020': .20}
VECTORS = {'roll': [1., 0.], 'pitch': [0., 1.], 'dual': [1., 1.]}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def read(path, expected):
    return json.loads(_verified_bytes(ROOT, path, expected))


def command_series(controller, actions, start, dt):
    series = []
    for tick, action in enumerate(actions, start):
        steps = controller.plan(tick*SUBSTEPS*dt, action, SUBSTEPS)
        for step in steps:
            np.testing.assert_array_equal(step.clipped_action, action)
            if not np.isfinite(np.r_[step.q, step.qd, step.qdd]).all():
                raise ValueError('nonfinite static joint command')
            controller.commit(step)
        series.append({'q': [step.q.tolist() for step in steps],
                       'qd': [step.qd.tolist() for step in steps]})
    return series, step.qdd.tolist()


def preflight():
    began = time.monotonic()
    BASE.mkdir(exist_ok=False)
    try:
        fit = json.loads((ROOT/FIT61).read_bytes())
        pinned = fit['files_sha256']
        cfg_template = read(TEMPLATE, pinned[TEMPLATE])
        start = read(START, START_SHA)['initial']
        train_sources = [*fit['timed_manifests'], 'runs/s3_pair_refit_20261112/capture_manifest.json',
                         'runs/s3_history_coverage_20261114/capture_manifest.json']
        past_keys = set()
        identity_hashes = {}
        for path in train_sources:
            identity_hashes[path] = pinned[path]
            for entry in read(path, pinned[path])['entries']:
                if entry['split'] not in ('train', 'validation'):
                    raise ValueError('capture metadata includes an unexpected split')
                if 'source_config' in entry:
                    source, sha = entry['source_config'], entry['source_config_sha256']
                else:
                    source = (Path(path).parent/entry['config_path']).as_posix()
                    sha = entry['config_sha256']
                past_keys.add(_initial_key(read(source, sha)))
                identity_hashes[source] = sha
        robot = Robot(ROOT/'assets/ec66_description.urdf')
        mesh = load_tool(ROOT/'assets/shovel.STL')
        pose, _, _ = robot.fk(cfg_template['initial']['joint_q_rad'])
        anchor = support_point(mesh, cfg_template['initial']['anchor_xy_tool_m'])[0]
        archive, _, _ = load_archive(ROOT)
        layout_path = archive['native_reference']['layout']
        backend = backend_preflight({'runtime': RUNTIME})
        if int(backend['device']['native_sm']) != RUNTIME['cuda_arch']:
            raise ValueError('CUDA architecture differs from the accepted device')
        codes = [Path(__file__).resolve(), ROOT/'requirements.lock',
                 *sorted((ROOT/'src/ball_retention').glob('*.py')),
                 *sorted((ROOT/'diagnostics/resource_probe').glob('*.py')),
                 *(ROOT/'scripts'/name for name in ('check_s3_actions.py', 'check_s3_deterministic_archive_import.py',
                    'collect_s3_small_action_response.py', 'preflight_s3_training_timing_capture.py'))]
        source_hashes = {relative(path): digest(path) for path in codes}
        source_hashes.update(identity_hashes)
        source_hashes.update({START: START_SHA, TEMPLATE: pinned[TEMPLATE], FIT61: digest(ROOT/FIT61)})
        asset_manifest = ROOT/'assets/manifest.json'
        source_hashes[relative(asset_manifest)] = digest(asset_manifest)
        for asset in json.loads(asset_manifest.read_bytes())['assets']:
            _verified_bytes(ROOT, asset['local_path'], asset['sha256'])
            source_hashes[asset['local_path']] = asset['sha256']
        path = cfg_template['prescribed']['joint_path']
        source_hashes[path] = digest(ROOT/path)
        output_cases = {}
        for tick, dx in ((72, -.006), (168, .006)):
            for lane, dy, dz in (('left', -.014, .003), ('center', 0., .017), ('right', .014, .031)):
                name = f'broad_train_{tick}_{lane}'
                case = {'split': 'train', 'group_id': name, 'episode_id': name, 'history_id': 'advance'}
                offset = [dx, dy, dz]
                ball = np.asarray(start['ball_center_world_m']) + offset
                cfg = {**cfg_template, 'history_capture': case, 'initial': {**cfg_template['initial'],
                       'ball_center_world_m': ball.tolist(), 'ball_velocity_world_m_s': [0., 0., 0.]}}
                key = _initial_key(cfg)
                local = pose[:3,:3].T@(ball-pose[:3,3])
                clearance = float(sphere_clearance(mesh, local, cfg['ball']['radius_m']))
                if (key in past_keys or clearance <= 0 or local[0] < .22
                        or ball[2]-cfg['ball']['radius_m'] <= cfg['physics']['floor_z_m']):
                    raise ValueError(f'{name} failed static initial identity or geometry, clearance {clearance:.6g} m')
                past_keys.add(key)
                directions = {f'{axis}_{code}': np.asarray([vector]*8+[[0., 0.]]*24)
                              for axis, vector in VECTORS.items() for code in AMPLITUDES}
                scales = {label: AMPLITUDES[label.rsplit('_',1)[1]] for label in directions}
                dt = cfg['physics']['dt_s']
                budget = {**cfg['budget'], 'wall_s': 480, 'max_cases': 1, 'max_new_output_mib': 24}
                trace_cap = (tick+14*HORIZON)*SUBSTEPS
                budget.update(max_physics_steps=trace_cap, max_trace_steps=trace_cap)
                cfg['budget'] = budget
                cfg['action_pairs'] = {**cfg['action_pairs'], 'snapshot_ticks': [tick],
                                       'amplitudes': list(AMPLITUDES.values())}
                path = PrescribedPath(ROOT/cfg['prescribed']['joint_path'], cfg, moving=True)
                controller = OnlineTiltController(robot, cfg, anchor, path, ROOT/'assets')
                timeline = schedule_actions(cfg['action_pairs']['schedule'], 0, tick+HORIZON)
                prefix, qdd = command_series(controller, timeline[:tick], 0, dt)
                control = controller.snapshot()
                nominal = np.asarray(timeline[tick:])
                requests = {'nominal_1': nominal, 'nominal_2': nominal}
                for label, direction in directions.items():
                    for side, sign in (('plus', 1.), ('minus', -1.)):
                        requests[f'{label}_{side}'] = nominal+sign*scales[label]*direction
                branches = {}
                for label, actions in requests.items():
                    if np.max(np.abs(actions)) >= 1.:
                        raise ValueError(f'{name}/{label} hits the action boundary')
                    controller.restore(control)
                    executed, _ = command_series(controller, actions, tick, dt)
                    branches[label] = {'requested': actions.tolist(), 'executed': executed}
                if branches['nominal_1'] != branches['nominal_2']:
                    raise ValueError('static nominal commands disagree')
                config_path = BASE/f'{name}_config.json'
                write_json(config_path, cfg)
                evidence = {'case': {**case, 'source_config': relative(config_path),
                             'source_config_sha256': digest(config_path), 'seed': cfg['seed'],
                             'physical_initial_sha256': key}, 'snapshot_tick': tick, 'offset_world_m': offset,
                            'clearance_m': clearance, 'snapshot_joint_qdd_rad_s2': qdd,
                            'snapshot_control_only': control, 'prefix_actions_sha256': sequence_digest(timeline[:tick]),
                            'prefix_requested': timeline[:tick], 'prefix_executed': prefix,
                            'directions_Hx2': {label: v.tolist() for label,v in directions.items()},
                            'amplitudes': scales, 'branches': branches}
                evidence_path = BASE/f'{name}_commands.json'
                write_json(evidence_path, evidence)
                output_cases[name] = {'commands': relative(evidence_path), 'sha256': digest(evidence_path),
                                      'physics_steps': trace_cap}
                mem = process_memory(os.getpid())
                if (time.monotonic()-began >= 120 or mem['peak_rss_bytes'] > 1024**3
                        or mem['peak_private_commit_bytes'] > 1.5*1024**3 or directory_bytes(BASE) > 32*1024**2):
                    raise RuntimeError('static resource cap reached')
        if sum(x['physics_steps'] for x in output_cases.values()) != TOTAL_STEPS:
            raise ValueError('static plans differ from the approved physical reservation')
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('static phase imported the physical backend')
        write_json(BASE/'preflight.json', {'passed': True, 'cases': output_cases,
                   'backend': backend, 'layout': {'path': layout_path, 'sha256': archive['inputs_sha256'][layout_path]},
                   'source_sha256': source_hashes, 'static_branches': 84, 'physics_steps': 0,
                   'physics_steps_reserved': TOTAL_STEPS, 'worker_starts': 0, 'fit_seed': 20261111,
                   'capture_limits': LIMITS, 'wall_s': time.monotonic()-began, **SCOPE})
    except BaseException as error:
        write_json(BASE/'preflight_stopped.json', {'reason': str(error), 'physics_steps': 0, **SCOPE})
        raise


def load_evidence(plan, name):
    row = plan['cases'][name]
    evidence = read(row['commands'], row['sha256'])
    case = evidence['case']
    cfg = read(case['source_config'], case['source_config_sha256'])
    if (name != case['episode_id'] or case['split'] != 'train' or _initial_key(cfg) != case['physical_initial_sha256']
            or cfg['budget']['max_physics_steps'] != row['physics_steps']):
        raise ValueError('frozen case identity, source or step cap changed')
    return cfg, evidence, row['physics_steps']


def admit(out, budget):
    sample = snapshot(ROOT)
    deadline = time.monotonic()+45
    while sample['available_ram_bytes'] < budget['min_start_ram_gib']*1024**3 and time.monotonic() < deadline:
        time.sleep(1)
        sample = snapshot(ROOT)
    hardware = admission(sample, budget)
    storage = storage_admission(ROOT, BASE, budget)
    env_bytes = directory_bytes(ROOT/'.venv')
    write_json(out/'admission.json', {'sample': sample, 'hardware': hardware, 'storage': storage,
                                     'environment_bytes': env_bytes})
    if not hardware['passed'] or not storage['passed'] or env_bytes > budget['max_environment_gib']*1024**3:
        raise RuntimeError('capture resource admission failed before starting Newton')


def record_case(out, cfg, evidence, layout, cap, state, stage):
    tick, dt = evidence['snapshot_tick'], cfg['physics']['dt_s']
    case, budget = evidence['case'], cfg['budget']
    started = time.monotonic()
    state.update(case=case, completed_branches=[], prefix_substeps=0)
    env = _make_env(ROOT, cfg, state, stage, expected_device_arch=RUNTIME['cuda_arch'])
    env.reset(episode_id=case['episode_id'])
    if (env.substeps, env.dt, env.step_index, env.physics.steps) != (SUBSTEPS, dt, 0, 0):
        raise ValueError('initial physics clock differs from the frozen case')
    write(out, 'native_layout.json', verify_live_layout(env.physics, layout))

    def advance(label, action, command):
        index, before = env.step_index, env.physics.steps
        if state['physics_steps']+SUBSTEPS > cap or time.monotonic()-started >= budget['wall_s']:
            raise RuntimeError('worker physics step or wall cap reached')
        try:
            row = env.step(action)
        finally:
            state['physics_steps'] += env.physics.steps-before
        nxt = row.next_state_before_reset
        if (row.episode_id != case['episode_id'] or row.step_index != index or env.step_index != index+1
                or row.state_t.time_s != before*dt or before != index*SUBSTEPS
                or env.physics.steps-before != SUBSTEPS or nxt.time_s != env.physics.steps*dt
                or row.invalid_reason is not None or row.terminated or row.truncated
                or row.requested_action != action or nxt.previous_action != action
                or {'q': row.executed_joint_q, 'qd': row.executed_joint_qd} != command):
            write(out, 'failed_transition.json', {'branch': label, 'transition': row.to_dict()})
            raise RuntimeError(f'{label} tick {index} failed the clock, command or episode check')
        return row

    stage('prefix_start')
    with (out/'prefix.jsonl').open('x', encoding='utf-8') as stream:
        for index, (action, command) in enumerate(zip(evidence['prefix_requested'], evidence['prefix_executed'], strict=True)):
            row = advance('prefix', action, command)
            state['prefix_substeps'] = state['physics_steps']
            stream.write(json.dumps({'absolute_tick': index, 'state_before_reset': asdict(row.next_state_before_reset),
                                     'task': row.task}, allow_nan=False)+'\n')
            if (index+1) % 12 == 0:
                stream.flush()
                stage(f'prefix_tick_{index+1}')
    native = env.snapshot()
    physics = native['physics']
    if (state['prefix_substeps'] != tick*SUBSTEPS or native['step_index'] != tick or physics['steps'] != tick*SUBSTEPS
            or native['controller'] != evidence['snapshot_control_only']
            or physics['command'][2].tolist() != evidence['snapshot_joint_qdd_rad_s2']):
        raise ValueError('runtime prefix does not reproduce the frozen checkpoint')
    write(out, 'snapshot.json', {**native, 'state': asdict(native['state']), 'physics': {**physics,
          'arrays': {name: values.tolist() for name, values in physics['arrays'].items()},
          'command': [values.tolist() for values in physics['command']]}})
    stage('snapshot_saved')

    traces = {}
    for label, branch in evidence['branches'].items():
        readback = restore_and_readback(env, native, SUBSTEPS)
        write(out, f'{label}_readback.json', readback)
        if readback['both_buffers_exact'] is not True:
            raise ValueError(f'{label} native buffers did not restore exactly')
        stage(f'{label}_start')

        def step(action):
            return advance(label, action, branch['executed'][env.step_index-tick])

        trace = _record_nominal(env, native, branch['requested'], step, SUBSTEPS, SUBSTEPS*dt)
        if [r['physics_step'] for r in trace['contacts']] != list(range(tick*SUBSTEPS+1, (tick+HORIZON)*SUBSTEPS+1)):
            raise ValueError(f'{label} physical contact clock is incomplete')
        write(out, f'{label}.json', trace)
        trace = json.loads((out/f'{label}.json').read_bytes())
        _full_branch(trace, tick)
        traces[label] = trace
        if label == 'nominal_2':
            gate = nominal_gate(traces['nominal_1'], trace, tick)
            write(out, 'nominal_comparison.json', gate)
            if not gate['passed'] or not gate['contacts_match']:
                raise RuntimeError('same-case nominal replay gate failed')
        state['completed_branches'].append(label)
        stage(f'{label}_complete')
    if state['physics_steps'] != cap:
        raise ValueError('physical step count differs from the frozen case')
    write(out, 'signal_report.json', signal_report({'directions': evidence['directions_Hx2']}, traces))
    state.update(status='completed', nominal_passed=True)


def worker(out, plan):
    began = time.monotonic()
    state = {'status': 'starting', 'worker_pid': os.getpid(), 'physics_steps': 0}

    def stage(stage_name):
        memory = process_memory(os.getpid())
        state['stage'] = stage_name
        state['memory'] = memory
        write_json(out/'worker_result.json', state)
        if (time.monotonic()-began >= budget['wall_s'] or directory_bytes(out) > budget['max_new_output_mib']*1024**2
                or memory['peak_rss_bytes'] > budget['max_process_rss_gib']*1024**3
                or memory['peak_private_commit_bytes'] > budget['max_process_commit_gib']*1024**3):
            raise RuntimeError('worker resource cap reached')

    try:
        cfg, evidence, cap = load_evidence(plan, out.parent.name)
        budget = cfg['budget']
        stage('resource_recheck')
        if not admission(snapshot(ROOT), budget)['passed']:
            raise RuntimeError('worker resource recheck failed before physics imports')
        layout = plan['layout']
        record_case(out, cfg, evidence, read(layout['path'], layout['sha256']), cap, state, stage)
        stage('complete')
    except BaseException as error:
        write_json(out/'worker_result.json', {**state, 'status': 'stopped', 'error': str(error)})
        raise


def manifest_entry(folder, cfg, evidence):
    def ref(name):
        path = folder/name
        return {'path': path.relative_to(BASE).as_posix(), 'sha256': digest(path)}

    def branch(label):
        return {'label': label, 'requested_actions_Hx2': evidence['branches'][label]['requested'], **ref(f'{label}.json')}

    pairs = [{'direction_id': label, 'direction_Hx2': direction, 'amplitude': evidence['amplitudes'][label],
              'branches': [{'sign': sign, **branch(f'{label}_{side}')} for side,sign in (('plus', 1), ('minus', -1))]}
             for label, direction in evidence['directions_Hx2'].items()]
    native = ref('snapshot.json')
    tick, dt = evidence['snapshot_tick'], cfg['physics']['dt_s']
    return {**evidence['case'], 'snapshot_tick': tick, 'native_snapshot': native, 'native_snapshot_sha256': native['sha256'],
            'clock': {'physics_dt_s': dt, 'control_substeps': SUBSTEPS, 'snapshot_time_s': tick*SUBSTEPS*dt},
            'prefix_actions_sha256': evidence['prefix_actions_sha256'], 'prefix_record': ref('prefix.jsonl'),
            'nominal_branches': [branch('nominal_1'), branch('nominal_2')], 'pairs': pairs,
            'signal_report': ref('signal_report.json'), 'worker_result': ref('worker_result.json')}


def capture(plan, path, sha):
    began = time.monotonic()
    entries, summaries = [], {}
    active = None
    try:
        if backend_preflight({'runtime': RUNTIME}) != plan['backend']:
            raise ValueError('installed backend differs from the static preflight')
        for name in plan['cases']:
            cfg, evidence, cap = load_evidence(plan, name)
            active = BASE/name
            active.mkdir(exist_ok=False)
            admit(active, cfg['budget'])
            with patch.dict(monitor.__globals__, {'__file__': __file__,
                            'directory_bytes': lambda _: directory_bytes(BASE),
                            'storage_admission': lambda root, folder, budget: storage_admission(root, BASE, budget)}):
                state = monitor(active, path, sha, cfg['budget'], LIMITS, began)
            if (state['case'] != evidence['case'] or state['physics_steps'] != cap
                    or state['completed_branches'] != list(evidence['branches']) or not state['nominal_passed']):
                raise ValueError('worker result differs from its frozen capture')
            entries.append(manifest_entry(active/'worker', cfg, evidence))
            summaries[name] = state
        pending = BASE/'capture_manifest.pending.json'
        write_json(pending, {'schema': SCHEMA, 'horizon': HORIZON, 'design_sha256': sha,
                            'development_holdout_group': 'negative_y_basin', 'entries': entries})
        manifest_sha = digest(pending)
        data = prepare_timed_pairs(ROOT, relative(pending), manifest_sha)
        if len(data.batch.initial) != 36 or len(set(data.batch.group_ids)) != 6:
            raise ValueError('capture reader did not recover 36 pairs and six groups')
        if time.monotonic()-began >= LIMITS['max_wall_s'] or directory_bytes(BASE) > LIMITS['max_new_output_mib']*1024**2:
            raise RuntimeError('aggregate capture resource cap reached')
        pending.rename(BASE/'capture_manifest.json')
        write_json(BASE/'validation.json', {'status': 'completed', 'preflight_sha256': sha,
                   'physics_steps': sum(x['physics_steps'] for x in summaries.values()), 'pairs': 36,
                   'worker_starts': 6, 'cases': list(summaries), 'wall_s': time.monotonic()-began,
                   'artifacts_sha256': {'capture_manifest.json': manifest_sha}, **SCOPE})
    except BaseException as error:
        state_path = active/'worker/worker_result.json' if active else None
        current = json.loads(state_path.read_bytes()) if state_path and state_path.is_file() else {}
        if active and active.name in summaries:
            current = {}
        write_json(BASE/'stopped.json', {'status': 'stopped', 'reason': str(error),
                   'completed_cases': list(summaries), 'active_case': active.name if active else None,
                   'completed_physics_steps': sum(x['physics_steps'] for x in summaries.values()),
                   'active_physics_steps_at_last_event': current.get('physics_steps', 0),
                   'wall_s': time.monotonic()-began, **SCOPE})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'capture', 'worker'))
    parser.add_argument('--plan', type=Path, default=BASE/'preflight.json')
    parser.add_argument('--sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase == 'preflight':
        preflight()
    else:
        path = args.plan.resolve()
        if path != BASE/'preflight.json' or not args.sha256:
            parser.error('the accepted preflight path and hash are required')
        plan = read(relative(path), args.sha256)
        if not plan['passed'] or len(plan['cases']) != 6 or plan['capture_limits'] != LIMITS:
            raise ValueError('six-case preflight scope differs from the authorization')
        for name, sha in plan['source_sha256'].items():
            _verified_bytes(ROOT, name, sha)
        if args.phase == 'capture':
            capture(plan, path, args.sha256)
        else:
            out = args.output.resolve()
            if out.parent.parent != BASE or out.name != 'worker':
                parser.error('invalid worker output path')
            worker(out, plan)
