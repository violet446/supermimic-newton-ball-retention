"""Freeze, capture and verify nested training-only Newton route datasets."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from check_s3_deterministic_archive_import import ROOT, backend_preflight, digest
from collect_s3_small_action_response import monitor, storage_admission
from run_s3_closed_loop_baseline import BROAD, BROAD_SHA, RUNTIME, SCOPE, read, relative
from s3_closed_loop_feedback import preflight as feedback_preflight
from s3_continuous_route_worker import (
    DT, HORIZON, SUBSTEPS, TRACE, WARMUP, action, atomic_json, checkpoint, collect,
)

import numpy as np
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.controller import OnlineTiltController
from ball_retention.dataset import load_episode, read_manifest, write_manifest
from ball_retention.geometry import load_tool, sphere_clearance, support_point
from ball_retention.guard import process_memory
from ball_retention.pair_collection import _initial_key
from ball_retention.prescribed import PrescribedPath
from ball_retention.resources import admission, snapshot
from ball_retention.robot import Robot


BASE = ROOT/'runs/s3_continuous_routes_20261118'
BASELINE = 'runs/s3_closed_loop_baseline_20261118'
BASELINE_SHA = '2dda37233461815fb82037252b9cc7196d67a9df3e3b9ee40ad53acea188aefe'
EDGE = 'runs/s1_episode_20261112/edge/transitions.jsonl'
EDGE_SHA = 'a82bb85a57e7d5b5443c3b5f3711ec239c2eac5a5c3722ac9a549e5e4bb0c13c'
EPISODE_CAP = HORIZON*SUBSTEPS
LIMITS = {'max_wall_s': 2100, 'max_new_output_mib': 128, 'max_episodes': 12,
          'max_physics_steps': 12*EPISODE_CAP, 'sequential_environments': 1}
ROUTES = (
    ('72_right', 'feedback'), ('72_center', 'pulse'), ('168_center', 'nominal'),
    ('72_left', 'nominal'), ('168_left', 'feedback'), ('168_right', 'pulse'),
    ('72_right', 'nominal'), ('72_center', 'feedback'), ('168_center', 'feedback'),
    ('72_left', 'pulse'), ('168_left', 'pulse'), ('168_right', 'nominal'),
)
PULSES = {
    '72_center': [[76, 88, [.12, 0.]], [88, 100, [-.12, 0.]], [154, 166, [0., .08]]],
    '168_right': [[96, 108, [0., -.12]], [108, 120, [0., .12]], [164, 180, [.08, -.08]]],
    '72_left': [[70, 86, [0., .12]], [128, 144, [-.08, .08]], [144, 160, [.08, -.08]]],
    '168_left': [[80, 96, [-.12, 0.]], [144, 160, [.08, .08]], [160, 176, [-.08, -.08]]],
}


def config(case):
    cfg = read(case['source_config'], case['source_config_sha256'])
    cfg['budget'].update(wall_s=LIMITS['max_wall_s'], max_cases=1,
                         max_physics_steps=EPISODE_CAP, max_trace_steps=EPISODE_CAP,
                         max_new_output_mib=LIMITS['max_new_output_mib'])
    return cfg


def check_kinematics(cfg, cases, feedback, robot, mesh):
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    path = PrescribedPath(ROOT/cfg['prescribed']['joint_path'], cfg, moving=True)
    controller = OnlineTiltController(robot, cfg, anchor, path, ROOT/'assets')
    initial = controller.snapshot()
    report = {}
    for case in ({'episode_id': 'nominal', 'policy': 'nominal'},
                 *(case for case in cases if case['policy'] == 'pulse')):
        controller.restore(initial)
        peak = np.zeros(2)
        for tick in range(HORIZON):
            requested = action(case, tick, None, feedback)
            if max(map(abs, requested)) > feedback['correction_cap'] or (tick < WARMUP and any(requested)):
                raise ValueError('schedule violates the frozen action scope')
            commands = controller.plan(tick*SUBSTEPS*DT, requested, SUBSTEPS)
            if len(commands) != SUBSTEPS:
                raise ValueError('unexpected number of commands')
            for command in commands:
                np.testing.assert_array_equal(command.clipped_action, requested)
                controller.commit(command)
                peak = np.maximum(peak, [abs(command.qd).max(), abs(command.qdd).max()])
        np.testing.assert_allclose(controller.time_s, 4., rtol=0., atol=1e-12)
        report[case['episode_id']] = {'control_rows': HORIZON, 'joint_speed_rad_s': float(peak[0]),
                                      'joint_acceleration_rad_s2': float(peak[1])}
    return report


def freeze():
    began = time.monotonic()
    BASE.mkdir(exist_ok=False)
    try:
        sources = {f'{BROAD}/validation.json': BROAD_SHA, f'{BASELINE}/validation.json': BASELINE_SHA, EDGE: EDGE_SHA}
        broad = read(f'{BROAD}/validation.json', BROAD_SHA)
        baseline = read(f'{BASELINE}/validation.json', BASELINE_SHA)
        if any(record['status'] != 'completed' or record['test_episode_files_read'] for record in (broad, baseline)):
            raise ValueError('a source capture lacks training-side acceptance')
        manifest_file = f'{BROAD}/capture_manifest.json'
        sources[manifest_file] = broad['artifacts_sha256']['capture_manifest.json']
        baseline_plan = f'{BASELINE}/preflight.json'
        sources[baseline_plan] = baseline['preflight_sha256']
        precedent = read(baseline_plan, sources[baseline_plan])
        feedback = precedent['feedback']
        source_report = f'{BASELINE}/comparison.json'
        sources[source_report] = baseline['artifacts_sha256']['comparison.json']
        baseline_cases = read(source_report, sources[source_report])['case_directories']
        entries = {entry['group_id']: entry for entry in read(manifest_file, sources[manifest_file])['entries']}
        sources.update({relative(p): digest(p) for p in (
            Path(__file__).resolve(), ROOT/'requirements.lock', ROOT/'assets/manifest.json',
            *(ROOT/'src/ball_retention').glob('*.py'),
            *(ROOT/'scripts'/name for name in ('s3_continuous_route_worker.py', 's3_closed_loop_feedback.py',
                'run_s3_closed_loop_baseline.py', 'check_s3_actions.py', 'check_s3_deterministic_archive_import.py',
                'collect_s3_small_action_response.py')),
            *(ROOT/'diagnostics/resource_probe'/name for name in ('native_repeatability.py', 'runner.py',
                                                                'saved_native_snapshot.py')),
        )})
        for item in read('assets/manifest.json', sources['assets/manifest.json'])['assets']:
            sources[item['local_path']] = item['sha256']
        robot = Robot(ROOT/'assets/ec66_description.urdf')
        mesh = load_tool(ROOT/'assets/shovel.STL')
        cases, geometry, common = [], {}, None
        for group, policy in ROUTES:
            key = f'broad_train_{group}'
            entry = entries[key]
            name = f'continuous_{group}_{policy}'
            path, sha = entry['source_config'], entry['source_config_sha256']
            sources[path] = sha
            cfg = read(path, sha)
            initial = cfg['initial']
            identity = _initial_key(cfg)
            if (entry['split'] != 'train' or cfg['history_capture']['split'] != 'train'
                    or entry['seed'] != cfg['seed'] or identity != entry['physical_initial_sha256']
                    or initial['ball_velocity_world_m_s'] != [0., 0., 0.]):
                raise ValueError('initial state is outside the frozen training groups')
            settings = {k: cfg[k] for k in ('ball', 'action', 'episode', 'physics', 'reference', 'prescribed',
                                          'robot', 'admission')}
            settings['initial'] = {k: v for k, v in initial.items() if k != 'ball_center_world_m'}
            if common is None:
                common = settings
            if settings != common:
                raise ValueError('the physical setup changes between groups')
            pose = robot.fk(initial['joint_q_rad'])[0]
            center = np.asarray(initial['ball_center_world_m'])
            local = pose[:3, :3].T@(center-pose[:3, 3])
            clearance = float(sphere_clearance(mesh, local, cfg['ball']['radius_m']))
            if clearance <= 0. or center[2]-cfg['ball']['radius_m'] <= cfg['physics']['floor_z_m']:
                raise ValueError('training initial state collides before Newton reset')
            geometry[key] = {'initial_sha256': identity, 'seed': cfg['seed'], 'center_tool_m': local.tolist(),
                             'center_world_m': center.tolist(), 'clearance_m': clearance}
            case = {'episode_id': name, 'group_id': key, 'policy': policy, 'source_config': path,
                    'source_config_sha256': sha, 'physical_initial_sha256': identity}
            if policy == 'pulse':
                case['pulses'] = PULSES[group]
            baseline_name = f'{key}_{policy}'
            if baseline_name in baseline_cases:
                archive = f'{baseline_cases[baseline_name]}/worker/{TRACE}'
                if archive.removeprefix(BASELINE+'/') not in baseline['artifacts_sha256']:
                    archive = f'{baseline_cases[baseline_name]}/{TRACE}'
                sources[archive] = baseline['artifacts_sha256'][archive.removeprefix(BASELINE+'/')]
                case['baseline_trace'] = archive
                case['baseline_episode_id'] = baseline_name
            cases.append(case)
        ref = cfg['reference']
        if ((ref['start_s'], ref['duration_s'], ref['hold_end_s'], ref['displacement_world_m'],
             cfg['episode']['horizon_s'], cfg['physics']['control_hz'], cfg['physics']['dt_s'])
            != (1., 2., 1., [.08, 0., 0.], 4., 60, DT)
                or len({v['initial_sha256'] for v in geometry.values()}) != 6):
            raise ValueError('unexpected route, clock or number of training initial states')
        for tier in (cases[:6], cases[6:]):
            if (len({case['group_id'] for case in tier}) != 6
                    or Counter(case['policy'] for case in tier) != {'nominal': 2, 'feedback': 2, 'pulse': 2}):
                raise ValueError('tiers must have identical policy counts and initial groups')
        route = cfg['prescribed']['joint_path']
        sources[route] = digest(ROOT/route)
        for path, sha in sources.items():
            _verified_bytes(ROOT, path, sha)
        edge = load_episode(ROOT/EDGE, DT, SUBSTEPS)[-1]
        ending = check_ending(edge)
        if not edge.terminated or ending['terminal_substeps'] >= SUBSTEPS:
            raise ValueError('departure regression must contain an unrestored shortened terminal transition')
        kinematics = check_kinematics(cfg, cases, feedback, robot, mesh)
        if any(precedent['kinematics'][key]['steps'] != HORIZON for key in
               ('corner_1', 'corner_2', 'corner_3', 'corner_4', 'switching_corners')):
            raise ValueError('prior bounded feedback command checks are missing')
        signs = feedback_preflight(np.asarray(cfg['initial']['tool_rotation_world']), feedback)
        backend = backend_preflight({'runtime': RUNTIME})
        if {'newton', 'warp'}.intersection(sys.modules) or time.monotonic()-began >= 120:
            raise ValueError('static validation unexpectedly entered Newton or exceeded 120 s')
        atomic_json(BASE/'preflight.json', {'status': 'passed', 'input_sha256': sources,
                     'limits': LIMITS, 'cases': cases, 'small': [case['episode_id'] for case in cases[:6]],
                     'geometry': geometry, 'feedback': feedback, 'feedback_signs': signs,
                     'kinematics': kinematics, 'archived_departure': ending, 'backend': backend,
                     'physics_steps': 0, 'wall_s': time.monotonic()-began, **SCOPE})
    except BaseException as error:
        atomic_json(BASE/'preflight_stopped.json', {'reason': str(error), 'physics_steps': 0, **SCOPE})
        raise


def check_ending(last):
    applied = len(last.executed_joint_q)
    task = last.task
    time_s = last.next_state_before_reset.time_s
    if last.terminated:
        if (last.truncated or task['success'] or task['departure_time_s'] != time_s
                or task['termination_reason'] not in ('floor', 'outside_projection', 'below_entire_tool')
                or not 1 <= applied <= SUBSTEPS):
            raise ValueError('invalid physical departure record')
    elif (not last.truncated or applied != SUBSTEPS or last.step_index+1 != HORIZON or time_s != 4.
          or task['departure_time_s'] is not None or not task['motion_complete']):
        raise ValueError('invalid full-route ending')
    return {'terminal_substeps': applied, 'terminated': last.terminated, 'truncated': last.truncated,
            'success': task['success'], 'motion_complete': task['motion_complete'],
            'departure_time_s': task['departure_time_s'], 'termination_reason': task['termination_reason'],
            'next_state_time_s': time_s, 'carrying': task['carrying']}


def plan_at(path, sha):
    if path != BASE/'preflight.json':
        raise ValueError('unexpected frozen plan path')
    plan = read(relative(path), sha)
    if (plan['status'] != 'passed' or plan['limits'] != LIMITS or plan['physics_steps'] != 0
            or any(plan[key] != value for key, value in SCOPE.items())
            or [(case['group_id'].removeprefix('broad_train_'), case['policy']) for case in plan['cases']] != list(ROUTES)
            or plan['small'] != [case['episode_id'] for case in plan['cases'][:6]]):
        raise ValueError('frozen capture scope changed')
    for name, expected in plan['input_sha256'].items():
        _verified_bytes(ROOT, name, expected)
    return plan


def worker(output, plan):
    case_dir = output.parent.parent
    if case_dir.parent != BASE or output.name != 'worker' or not output.parent.name.startswith('attempt_'):
        raise ValueError('unexpected worker output')
    case = next(case for case in plan['cases'] if case['episode_id'] == case_dir.name)
    cfg = config(case)
    reservation = json.loads((output.parent/'reservation.json').read_bytes())
    began = time.monotonic()
    state = {'status': 'starting', 'worker_pid': os.getpid(), 'case': case,
             'reserved_new_steps': 0, 'actual_new_steps': 0}

    def stage(label):
        mem = process_memory(os.getpid())
        state.update(stage=label, worker_wall_s=time.monotonic()-began, worker_memory=mem)
        atomic_json(output/'worker_result.json', state)
        budget = cfg['budget']
        if (mem['peak_rss_bytes'] > budget['max_process_rss_gib']*1024**3
                or mem['peak_private_commit_bytes'] > budget['max_process_commit_gib']*1024**3
                or state['worker_wall_s'] >= reservation['wall_s_remaining']):
            raise RuntimeError('local worker time or memory cap reached')

    try:
        collect(output, cfg, plan, case, stage, state, reservation['max_new_physics_steps'])
        state['status'] = 'completed'
        stage('validated')
    except BaseException as error:
        atomic_json(output/'worker_result.json', {**state, 'status': 'stopped', 'error': str(error)})
        raise


def live_bytes(path):
    total = 0
    for file in path.rglob('*'):
        try:
            if file.is_file():
                total += file.stat().st_size
        except FileNotFoundError:
            # A checkpoint or result can be atomically replaced during sampling.
            continue
    return total


def admit(attempt, budget, deadline):
    start = time.monotonic()
    sample = snapshot(ROOT)
    until = min(start+45., deadline)
    while sample['available_ram_bytes'] < budget['min_start_ram_gib']*1024**3 and time.monotonic() < until:
        time.sleep(1.)
        sample = snapshot(ROOT)
    hardware = admission(sample, budget)
    storage = storage_admission(ROOT, BASE, budget)
    size = live_bytes(ROOT/'.venv')
    atomic_json(attempt/'admission.json', {'sample': sample, 'hardware': hardware, 'storage': storage,
                'environment_bytes': size, 'waiting_wall_s': time.monotonic()-start})
    if (not hardware['passed'] or not storage['passed'] or size > budget['max_environment_gib']*1024**3
            or time.monotonic() >= deadline):
        raise RuntimeError('startup resource admission failed')


def verify_route(folder, case, plan):
    name = case['episode_id']
    saved, _ = checkpoint(folder, name)
    trace = folder/TRACE
    if not saved or not saved['terminal'] or trace.stat().st_size != saved['trace_bytes']:
        raise ValueError('episode does not have a fully committed terminal record')
    rows = load_episode(trace, DT, SUBSTEPS, name)
    expected_position = config(case)['initial']['ball_center_world_m']
    np.testing.assert_allclose(rows[0].state_t.ball.position, expected_position, atol=1e-7, rtol=0.)
    if (any(row.invalid_reason is not None or row.requested_action != action(case, tick, row.state_t, plan['feedback'])
            for tick, row in enumerate(rows)) or sum(len(row.executed_joint_q) for row in rows) != saved['committed_physics_steps']):
        raise ValueError('action or physics count differs from the frozen plan')
    ending = check_ending(rows[-1])
    if 'baseline_trace' in case:
        baseline = load_episode(ROOT/case['baseline_trace'], DT, SUBSTEPS, case['baseline_episode_id'])
        if len(baseline) != len(rows):
            raise ValueError('repeatability archive has different route length')
        for i, (old, new) in enumerate(zip(baseline, rows)):
            if {**old.to_dict(), 'episode_id': name} != new.to_dict():
                raise ValueError(f'native checkpoint route differs from accepted baseline at control {i+1}')
        ending['baseline_rows_exact'] = len(rows)
    entry = {'episode_id': name, 'group_id': case['group_id'], 'policy': case['policy'], 'split': 'train',
             'path': trace.relative_to(BASE).as_posix(), 'sha256': saved['trace_sha256'],
             'physical_initial_sha256': case['physical_initial_sha256'],
             'source_config': case['source_config'], 'source_config_sha256': case['source_config_sha256']}
    result = {'entry': entry, 'ending': ending, 'controls': len(rows), 'physics_steps': saved['committed_physics_steps']}
    atomic_json(folder/'accepted.json', result)
    return result


def publish(rows, sha, tier):
    path = BASE/('small_manifest.json' if tier == 'small' else 'manifest.json')
    entries = [row['entry'] for row in rows]
    if path.exists():
        manifest = json.loads(path.read_bytes())
        if [(e['episode_id'], e['sha256']) for e in manifest['entries']] != [(e['episode_id'], e['sha256']) for e in entries]:
            raise ValueError('published training tier differs from committed episodes')
        return manifest['splits']['train']
    manifest = write_manifest(BASE, entries, DT, SUBSTEPS, {'frozen_plan_sha256': sha, 'tier': tier, **SCOPE})
    if read_manifest(BASE, verify_splits=('train',)) != manifest:
        raise ValueError('dataset manifest does not round-trip')
    if tier == 'small':
        os.replace(BASE/'manifest.json', path)
    return manifest['splits']['train']


def capture(plan, path, sha, resume):
    current = BASE/'batch_progress.json'
    if current.exists() != resume:
        raise ValueError('use capture once and resume only a recorded interrupted batch')
    initial = time.monotonic()
    batch = (json.loads(current.read_bytes()) if resume else
             {'plan_sha256': sha, 'status': 'new', 'attempts': [], 'completed': [],
              'physical_steps_upper_bound': 0, 'wall_s': 0.})
    if batch['plan_sha256'] != sha or (resume and (batch['status'] != 'stopped' or not batch['restartable'])):
        raise ValueError('only a recorded admission or RAM stop can resume this batch')
    previous_wall = batch['wall_s']
    began = initial-previous_wall
    deadline = began+LIMITS['max_wall_s']

    def save(status, **fields):
        batch.update(status=status, wall_s=time.monotonic()-began, **fields)
        atomic_json(current, batch)

    rows = []
    in_progress = None
    parent_reason = None
    try:
        save('running', reason=None, restartable=False)
        if backend_preflight({'runtime': RUNTIME}) != plan['backend'] or {'newton', 'warp'}.intersection(sys.modules):
            raise ValueError('launcher backend differs from static preflight')
        for i, case in enumerate(plan['cases']):
            name = case['episode_id']
            folder = BASE/name
            folder.mkdir(exist_ok=True)
            saved, _ = checkpoint(folder, name)
            if not (saved and saved['terminal']):
                previous_steps = saved['committed_physics_steps'] if saved else 0
                reserve = EPISODE_CAP-previous_steps
                if batch['physical_steps_upper_bound']+reserve > LIMITS['max_physics_steps']:
                    raise RuntimeError('physical step cap prevents another full route reservation')
                budget = config(case)['budget']
                attempt = folder/f'attempt_{1+len(list(folder.glob("attempt_*"))):03d}'
                attempt.mkdir(exist_ok=False)
                admit(attempt, budget, deadline)
                atomic_json(attempt/'reservation.json', {'max_new_physics_steps': reserve,
                                                         'wall_s_remaining': deadline-time.monotonic()})
                in_progress = {'episode': name, 'attempt': attempt.relative_to(BASE).as_posix(),
                               'starting_committed_steps': previous_steps, 'max_reserved_steps': reserve}
                batch['attempts'].append(in_progress)
                save('running')
                try:
                    with (patch.dict(monitor.__globals__, {'__file__': __file__, 'directory_bytes': lambda _: live_bytes(BASE),
                              'storage_admission': lambda root, _, b: storage_admission(root, BASE, b)}),
                          patch.dict(storage_admission.__globals__, {'directory_bytes': live_bytes})):
                        state = monitor(attempt, path, sha, budget, LIMITS, began)
                except BaseException as error:
                    parent_reason = str(error)
                    raise
                charge_attempt(batch, in_progress)
                in_progress = None
                if state['case'] != case:
                    raise ValueError('worker identity differs from frozen case')
            rows.append(verify_route(folder, case, plan))
            batch['completed'] = [r['entry']['episode_id'] for r in rows]
            save('running')
            print(f'{i+1}/{len(plan["cases"])} {name}: {rows[-1]["controls"]} controls, '
                  f'success={rows[-1]["ending"]["success"]}', flush=True)
            if i == 5:
                publish(rows, sha, 'small')
        plan_at(path, sha)
        large = publish(rows, sha, 'large')
        small = json.loads((BASE/'small_manifest.json').read_bytes())
        if small['entries'] != json.loads((BASE/'manifest.json').read_bytes())['entries'][:6]:
            raise ValueError('small tier is not an exact whole-episode prefix of the large tier')
        if time.monotonic() >= deadline or live_bytes(BASE) > LIMITS['max_new_output_mib']*1024**2:
            raise RuntimeError('aggregate time or output cap reached')
        save('completed')
        artifacts = {relative(file): digest(file) for file in BASE.rglob('*') if file.is_file()}
        atomic_json(BASE/'validation.json', {'status': 'completed', 'preflight_sha256': sha,
                    'small': small['splits']['train'], 'large': large, 'routes': rows,
                    'physical_steps_upper_bound': batch['physical_steps_upper_bound'],
                    'wall_s': time.monotonic()-began, 'new_output_bytes': live_bytes(BASE),
                    'artifacts_sha256': artifacts, **SCOPE})
    except BaseException as error:
        if in_progress is not None:
            charge_attempt(batch, in_progress)
        reason = str(error)
        restartable = reason in ('startup resource admission failed', 'system_available_ram_floor',
                                 'system_available_ram_floor_at_control_boundary', 'critical_system_ram',
                                 'worker resource recheck failed before Newton imports')
        save('stopped', reason=reason, restartable=restartable)
        atomic_json(BASE/'stopped.json', {'reason': parent_reason or reason, 'plan_sha256': sha,
                    'completed': batch['completed'], 'physical_steps_upper_bound': batch['physical_steps_upper_bound'],
                    'active_attempt': in_progress, 'wall_s': batch['wall_s'], 'restartable': restartable, **SCOPE})
        raise


def charge_attempt(batch, attempt):
    output = BASE/attempt['attempt']/'worker'
    progress = output/'worker_result.json'
    state = json.loads(progress.read_bytes()) if progress.exists() else {}
    saved, _ = checkpoint(output.parent.parent, attempt['episode'])
    start = attempt['starting_committed_steps']
    committed_delta = saved['committed_physics_steps']-start if saved else 0
    upper = max(committed_delta, state.get('reserved_new_steps', 0))
    if not 0 <= upper <= attempt['max_reserved_steps']:
        raise ValueError('worker progress exceeds its reservation')
    attempt.update(physics_steps_upper_bound=upper, committed_steps=committed_delta,
                   final_stage=state.get('stage'), final_status=state.get('status'),
                   last_committed_tick=saved['control_rows'] if saved else 0)
    batch['physical_steps_upper_bound'] += upper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'capture', 'resume', 'worker'))
    parser.add_argument('--plan', type=Path, default=BASE/'preflight.json')
    parser.add_argument('--sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.phase == 'preflight':
        freeze()
    else:
        if not args.sha256:
            parser.error('frozen preflight SHA-256 is required')
        path = args.plan.resolve()
        plan = plan_at(path, args.sha256)
        if args.phase == 'worker':
            worker(args.output.resolve(), plan)
        else:
            capture(plan, path, args.sha256, resume=args.phase == 'resume')
