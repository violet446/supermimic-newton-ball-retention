"""Run the frozen validation-only multiamplitude Newton replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from check_s3_deterministic_archive_import import ROOT, backend_preflight, digest, write_json
import numpy as np

from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.deterministic_archive_import_plan import MARKER, load as load_archive
from ball_retention.guard import process_memory, stop_reason
from ball_retention.resources import admission, directory_bytes, snapshot
from deterministic_response_metrics import response_metrics
from runner import monitored_pid, stop_owned_tree, storage_admission
from small_action_response import nominal_gate, run as replay


SCOPE = {'training_updates': 0, 'test_episode_files_read': False, 'control_accepted': False}


def load_run(path, expected):
    plan = json.loads(_verified_bytes(ROOT, path.relative_to(ROOT).as_posix(), expected))
    if plan['schema'] != 's3-small-action-response-capture-v1' or plan['scope'] != SCOPE:
        raise ValueError('unexpected capture scope')
    files = plan['files_sha256']
    for name, sha in files.items():
        _verified_bytes(ROOT, name, sha)

    def read(name):
        return json.loads(_verified_bytes(ROOT, name, files[Path(name).as_posix()]))

    accepted = read(plan['preflight_validation'])
    static = read(plan['preflight_plan'])
    if (accepted['status'] != 'completed' or accepted['plan_sha256'] != files[plan['preflight_plan']]
            or (accepted['updates'], accepted['simulation_steps'], accepted['newton_worker_starts'],
                accepted['test_episode_files_read'], accepted['control_accepted']) != (0, 0, 0, False, False)
            or (accepted['development_cases'], accepted['static_branches'], accepted['shared_prefixes_verified']) != (6, 14, 4)):
        raise ValueError('static preflight was not accepted')
    folder = Path(plan['preflight_validation']).parent
    spec, commands, models = (read(folder/name) for name in ('replay_spec.json', 'branch_commands.json', 'model_rollouts.json'))
    for name in ('replay_spec.json', 'branch_commands.json', 'model_rollouts.json'):
        if accepted['artifacts_sha256'][name] != files[(folder/name).as_posix()]:
            raise ValueError(f'changed preflight artifact: {name}')
    archive, cfg, _ = load_archive(ROOT)
    parent = read(plan['original_response_plan'])
    limits = spec['future_capture_limits']
    if (spec['split'] != 'validation' or spec['case'] != static['development_case'] or spec['case'] != parent['case']
            or cfg['seed'] != spec['case']['seed'] or spec['runtime'] != parent['runtime']
            or limits != static['future_capture'] or limits != plan['limits']
            or spec['branch_order'] != static['branch_order'] or list(commands) != spec['branch_order']
            or (spec['horizon'], spec['substeps_per_control'], len(commands), limits['max_physics_substeps']) != (32, 16, 14, 7168)
            or set(models) != {'49_pairs', '61_pairs'} or any(set(v['pairs']) != set(spec['cases']) for v in models.values())
            or limits['max_worker_starts'] != 1 or limits['sequential_environments'] != 1
            or limits['baseline_replays'] != 2 or limits['nominal_prefix_substeps'] != 0):
        raise ValueError('small-action execution scope differs from the static preflight')
    for label, row in commands.items():
        if (np.asarray(row['requested']).shape != (32, 2) or len(row['executed']) != 32
                or any(any(np.asarray(step[k]).shape != (16, 6) for k in ('q', 'qd')) for step in row['executed'])):
            raise ValueError(f'wrong command clock for {label}')
    native = {key: read(archive['native_reference'][key]) for key in ('snapshot', 'layout')}
    if native['layout'] != spec['static_snapshot_checks']['native_layout']:
        raise ValueError('snapshot layout differs from the accepted static layout')
    recorded = [read(key) for key in parent['nominal_replays']]
    cfg.pop(MARKER)
    cfg['budget'].update(max_cases=1, wall_s=limits['max_worker_wall_s'], max_physics_steps=7168, max_trace_steps=7168,
        max_process_rss_gib=limits['max_process_rss_gib'], max_process_commit_gib=limits['max_process_commit_gib'],
        max_gpu_used_gib=limits['max_total_gpu_used_gib'], min_start_ram_gib=limits['min_start_ram_gib'],
        min_start_gpu_free_gib=limits['min_start_gpu_free_gib'], min_disk_free_gib=limits['min_disk_free_gib'],
        max_new_output_mib=limits['max_new_output_mib'], max_output_cache_gib=limits['worker_output_plus_shared_cache_and_temp_gib'])
    return plan, cfg, spec, native, commands, recorded, models, read


def score_case(scoring, baseline, plus, minus, model_nominal, model_pair, amplitude):
    return response_metrics(
        baseline, plus, minus, model_nominal, model_pair['plus'], model_pair['minus'], model_pair['directional_jvp'],
        amplitude, leads=scoring['diagnostic_leads'], floors=scoring['absolute_floors_si'],
        **{k: scoring[k] for k in ('noise_multiplier', 'strong_multiplier', 'min_cosine', 'max_relative_error')})


def preflight(plan, spec, recorded, read):
    if backend_preflight(spec) != spec['static_backend']:
        raise ValueError('installed backend differs from the accepted static preflight')
    scoring = spec['scoring']
    gate = nominal_gate(recorded, recorded, scoring['summary_leads'])
    if not gate['passed'] or gate['all_branch_max_half_spread_by_lead'] != spec['static_snapshot_checks'][
            'archived_nominal_gate']['all_branch_max_half_spread_by_lead']:
        raise ValueError('archived nominal gate differs from static preflight')
    root = Path(plan['archived_response_validation']).parent
    archived = read(plan['archived_response_validation'])
    baseline = [read(root/'worker'/f'{name}.json')['states_raw'] for name in ('nominal_1', 'nominal_2')]
    amplitude = scoring['archived_full_amplitude']
    for tag, folder in plan['fixed_endpoint_sources'].items():
        rollout = read(Path(folder)/'timed_rollouts.json')
        old = read(Path(folder)/'comparison.json')['held_timing']
        for case, half in scoring['archived_full_half_states_tool_si'].items():
            a, b = (root/'worker'/f'{case}_{sign}.json' for sign in ('plus', 'minus'))
            for file in (a, b):
                if plan['files_sha256'][file.as_posix()] != archived['artifacts_sha256'][file.relative_to(root).as_posix()]:
                    raise ValueError('archived full-amplitude reference changed')
            pair = rollout['pairs'][case]
            plus, minus = (read(item)['states_raw'] for item in (a, b))
            np.testing.assert_array_equal(.5*(np.asarray(plus)-np.asarray(minus)), half)
            result = score_case(scoring, baseline, plus, minus, rollout['nominal'],
                                {**pair, 'directional_jvp': pair['jvp']}, amplitude)
            for lead, row in old[case]['by_lead'].items():
                if result['by_lead'][lead] != row:
                    raise ValueError(f'archived scorer differs for {tag}/{case}/H{lead}')


def worker(out, path, expected):
    started = time.monotonic()
    state = {'status': 'starting', 'worker_pid': os.getpid(), 'physics_steps': 0,
             'completed_branches': [], 'accepted_prefixes': []}

    def stage(name):
        state.setdefault('phase_events', []).append({'stage': name, 'elapsed_s': time.monotonic()-started,
                                                     **process_memory(os.getpid())})
        write_json(out/'worker_result.json', state)

    stage('starting')
    try:
        _, cfg, spec, native, commands, recorded, _, _ = load_run(path, expected)
        if not admission(snapshot(ROOT), cfg['budget'])['passed']:
            raise RuntimeError('worker resource recheck failed')
        replay(ROOT, cfg, spec, native, commands, recorded, out, state, stage)
    except BaseException as error:
        state.update(status='stopped', error=str(error))
        stage('stopped')
        raise


def monitor(out, path, expected, budget, limits, began):
    worker_dir = out/'worker'
    worker_dir.mkdir()
    env = os.environ.copy()
    for name, rel in {'TEMP': '.tmp', 'TMP': '.tmp', 'TMPDIR': '.tmp', 'WARP_CACHE_PATH': '.cache/warp',
                      'NEWTON_CACHE_PATH': '.cache/newton'}.items():
        folder = ROOT/rel
        folder.mkdir(parents=True, exist_ok=True)
        env[name] = str(folder)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--plan', str(path),
               '--sha256', expected, '--output', str(worker_dir)]
    if time.monotonic()-began >= limits['max_wall_s']:
        raise RuntimeError('end-to-end wall cap before launch')
    with (worker_dir/'worker.log').open('x', encoding='utf-8') as log, (out/'resources.jsonl').open('x', encoding='utf-8') as trace:
        started = time.monotonic()
        low_since = None
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while proc.poll() is None:
                sample = snapshot(ROOT)
                now = time.monotonic()
                if sample['available_ram_bytes'] < budget['min_running_ram_gib']*1024**3:
                    low_since = now if low_since is None else low_since
                else:
                    low_since = None
                pid = monitored_pid(worker_dir/'worker_result.json', proc.pid)
                try:
                    memory = process_memory(pid)
                except OSError:
                    if proc.poll() is not None:
                        break
                    raise
                storage = storage_admission(ROOT, worker_dir, budget)
                total = directory_bytes(out)
                stopped = stop_reason(sample, budget, now-started, memory['peak_rss_bytes'], storage['total_bytes'],
                                      0. if low_since is None else now-low_since, memory['peak_private_commit_bytes'])
                if now-began >= limits['max_wall_s']:
                    stopped = 'end_to_end_wall_budget'
                if total > limits['max_new_output_mib']*1024**2:
                    stopped = 'new_output_budget'
                trace.write(json.dumps({'wall_s': now-started, 'worker_pid': pid, 'stop_reason': stopped,
                                        'output_bytes': total, **memory, **sample})+'\n')
                trace.flush()
                if stopped:
                    raise RuntimeError(stopped)
                time.sleep(1)
        finally:
            if proc.poll() is None:
                stop_owned_tree(proc)
            proc.wait(timeout=15)
    state = json.loads((worker_dir/'worker_result.json').read_bytes())
    if proc.returncode or state['status'] != 'completed':
        raise RuntimeError(state.get('error') or state['status'])
    return state


def score(spec, models, traces):
    policy = spec['scoring']
    nominal = [traces[key]['states_raw'] for key in spec['branch_order'][:2]]
    results = {}
    for tag, model in models.items():
        results[tag] = {}
        for case, item in spec['cases'].items():
            plus, minus = (traces[f'{case}_{sign}']['states_raw'] for sign in ('plus', 'minus'))
            results[tag][case] = score_case(policy, nominal, plus, minus, model['nominal'], model['pairs'][case],
                                             item['amplitude'])
    return {'models': results, 'case': spec['case'], 'amplitudes': {k: v['amplitude'] for k,v in spec['cases'].items()},
            'scope': 'Development-only, measured Newton secants compared at the same amplitude; local JVPs remain model-only.'}


def capture(out, path, expected):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    try:
        plan, cfg, spec, _, commands, recorded, models, read = load_run(path, expected)
        preflight(plan, spec, recorded, read)
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('launcher imported the physical backend')
        write_json(out/'execution_plan.json', {'plan': plan, 'plan_sha256': expected, 'worker_config': cfg})
        budget = cfg['budget']
        limits = plan['limits']
        sample = snapshot(ROOT)
        hardware = admission(sample, budget)
        storage = storage_admission(ROOT, out/'worker', budget)
        environment = directory_bytes(ROOT/'.venv')
        write_json(out/'admission.json', {'hardware': hardware, 'sample': sample, 'storage': storage,
                                         'environment_bytes': environment})
        if not hardware['passed'] or not storage['passed'] or environment > budget['max_environment_gib']*1024**3:
            raise RuntimeError('small-action capture resource admission failed')
        state = monitor(out, path, expected, budget, limits, began)
        if (state['completed_branches'] != spec['branch_order'] or state['physics_steps'] != limits['max_physics_substeps']
                or state['nominal_prefix_substeps'] != 0 or state['nominal_gate_passed'] is not True
                or state['accepted_prefixes'] != [b for _, b, _ in spec['same_sign_prefixes']]):
            raise ValueError('worker finished outside its frozen scope')
        for name, sha in plan['files_sha256'].items():
            _verified_bytes(ROOT, name, sha)
        traces = {label: json.loads((out/'worker'/f'{label}.json').read_bytes()) for label in state['completed_branches']}
        if any(branch['actions'] != commands[label]['requested'] or branch['commands'] != commands[label]['executed']
               for label, branch in traces.items()):
            raise ValueError('recorded branch commands differ from the accepted static commands')
        write_json(out/'same_amplitude_scores.json', score(spec, models, traces))
        if time.monotonic()-began >= limits['max_wall_s'] or directory_bytes(out) > limits['max_new_output_mib']*1024**2:
            raise RuntimeError('end-to-end time or output cap reached')
        write_json(out/'validation.json', {'status': 'completed', 'plan_sha256': expected, 'case': spec['case'],
                   'physics_steps': state['physics_steps'], 'worker_starts': 1, 'nominal_prefix_substeps': 0,
                   'branches': state['completed_branches'], 'nominal_gate_passed': True,
                   'accepted_prefixes': state['accepted_prefixes'], 'wall_s': time.monotonic()-began, **SCOPE,
                   'artifacts_sha256': {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    except BaseException as error:
        record = out/'worker/worker_result.json'
        state = json.loads(record.read_bytes()) if record.exists() else {}
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error), **SCOPE,
                   'physics_steps_at_last_event': state.get('physics_steps', 0), 'branches': state.get('completed_branches', []),
                   'wall_s': time.monotonic()-began})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'capture', 'worker'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    path = args.plan.resolve()
    if args.phase == 'preflight':
        plan, _, spec, _, _, recorded, _, read = load_run(path, args.sha256)
        preflight(plan, spec, recorded, read)
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('preflight imported the physical backend')
        print('Capture sources, archived score reproduction, and deterministic backend verified.')
    else:
        out = args.output.resolve()
        frozen = json.loads(_verified_bytes(ROOT, path.relative_to(ROOT).as_posix(), args.sha256))
        expected_out = ROOT/frozen['output']
        if out != (expected_out/'worker' if args.phase == 'worker' else expected_out):
            parser.error('output differs from frozen capture plan')
        (capture if args.phase == 'capture' else worker)(out, path, args.sha256)
