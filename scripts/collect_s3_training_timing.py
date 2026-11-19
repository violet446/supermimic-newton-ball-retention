"""Collect the two frozen timing cases sequentially and publish accepted pairs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from check_s3_deterministic_archive_import import ROOT, digest, stop_owned_tree, storage_admission, write_json
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.resources import admission, directory_bytes, snapshot
from ball_retention.training_timing_capture_plan import DESIGN, DESIGN_SHA256, MODE, PREFLIGHT, PREFLIGHT_SHA256, load


OUT = ROOT/'runs/s3_training_timing_capture_20261115'


def case_entry(out, folder, plan, cfg, evidence):
    """Bind each declared pair to a full, validated before-reset trace."""
    read = lambda name: json.loads((folder/name).read_bytes())
    reference = lambda name: {'path': (folder/name).relative_to(out).as_posix(), 'sha256': digest(folder/name)}
    case, tick = evidence['case'], plan['snapshot_tick']
    horizon, substeps = plan['horizon'], plan['control_substeps']
    saved = read('snapshot.json')
    dt = cfg['physics']['dt_s']
    if (saved['episode_id'] != case['episode_id'] or saved['step_index'] != tick
            or saved['physics']['steps'] != tick*substeps or saved['state']['time_s'] != tick*substeps*dt
            or saved['controller'] != evidence['snapshot_control_only'] or saved['done'] or saved['invalid']):
        raise ValueError('saved training snapshot has the wrong identity or clock')
    branches = {}
    for label in plan['branch_order']:
        filename = f'{label}.json'
        trace = read(filename)
        expected = evidence['branches'][label]
        if (trace['actions'] != expected['requested'] or trace['commands'] != expected['executed']
                or trace['substeps'] != [substeps]*horizon or trace['terminated'] or trace['truncated']
                or trace['termination_reason'] is not None or len(trace['states_raw']) != horizon
                or len(trace['retention']) != horizon
                or [r['time_s'] for r in trace['states']] != [(tick+i)*substeps*dt for i in range(1, horizon+1)]
                or [r['physics_step'] for r in trace['contacts']] != list(range(tick*substeps+1, (tick+horizon)*substeps+1))
                or read(f'{label}_readback.json')['both_buffers_exact'] is not True):
            raise ValueError(f'training trace failed publication checks: {label}')
        branches[label] = {'label': label, 'requested_actions_Hx2': trace['actions'], **reference(filename)}
    for name in ('nominal_comparison.json', 'shared_prefix_plus.json', 'shared_prefix_minus.json'):
        if read(name)['passed'] is not True:
            raise ValueError(f'training trace failed {name}')
    directions = [{'direction_id': name, 'direction_Hx2': evidence['directions_Hx2'][name],
                   'amplitude': plan['amplitude'], 'branches': [
                       {'sign': sign, **branches[f'{name}_{side}']} for side, sign in (('plus', 1), ('minus', -1))]}
                  for name in plan['directions']]
    native = reference('snapshot.json')
    return {**case, 'snapshot_tick': tick, 'native_snapshot_sha256': native['sha256'], 'native_snapshot': native,
            'clock': {'physics_dt_s': dt, 'control_substeps': substeps, 'snapshot_time_s': tick*substeps*dt},
            'prefix_actions_sha256': evidence['prefix_actions_sha256'], 'prefix_record': reference('prefix.jsonl'),
            'nominal_branches': [branches[label] for label in plan['branch_order'][:2]], 'pairs': directions,
            'signal_report': reference('signal_report.json'), 'worker_result': reference('result.json')}


def run(out):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    completed, attempts = [], []
    active = None

    def guard():
        if time.monotonic()-began >= limits['max_wall_s'] or directory_bytes(out) > limits['max_new_output_mib']*1024**2:
            raise RuntimeError('aggregate training timing wall or output cap reached')

    try:
        plan, cases, _ = load(ROOT)
        limits = plan['budget']
        static = out/'static'
        subprocess.run([sys.executable, '-B', str(ROOT/'scripts/preflight_s3_training_timing_capture.py'),
                        '--output', str(static)], cwd=ROOT, check=True, timeout=plan['static_budget']['max_wall_s'],
                       creationflags=subprocess.CREATE_NO_WINDOW)
        report = json.loads((static/'preflight.json').read_bytes())
        frozen = json.loads(_verified_bytes(ROOT, PREFLIGHT, PREFLIGHT_SHA256))
        if (report['design_sha256'] != DESIGN_SHA256 or report['passed'] is not True
                or report['artifacts_sha256'] != frozen['artifacts_sha256']
                or any(digest(static/name) != sha for name, sha in report['artifacts_sha256'].items())):
            raise ValueError('recomputed static commands differ from the frozen preflight')
        current_path = Path(__file__).resolve()
        sources = {**report['source_sha256'], current_path.relative_to(ROOT).as_posix(): digest(current_path),
                   **{cfg['prescribed']['joint_path']: digest(ROOT/cfg['prescribed']['joint_path']) for cfg, _ in cases.values()}}

        def verify_sources():
            if any(digest(ROOT/name) != value for name, value in sources.items()):
                raise ValueError('training timing execution inputs changed')

        write_json(out/'execution_plan.json', {'design': plan, 'source_sha256': sources,
                   'frozen_preflight_sha256': PREFLIGHT_SHA256, 'fresh_preflight_sha256': digest(static/'preflight.json')})
        guard()
        steps = limits['max_physics_substeps_per_worker']
        for name, (cfg, evidence) in cases.items():
            verify_sources()
            guard()
            resources = snapshot(ROOT)
            folder = out/name
            hardware = admission(resources, cfg['budget'])
            storage = storage_admission(ROOT, folder, cfg['budget'])
            write_json(out/f'{name}_admission.json', {'resources': resources, 'hardware': hardware, 'storage': storage})
            if not hardware['passed'] or not storage['passed']:
                raise RuntimeError(f'resource admission failed for {name}')
            config = out/f'{name}_config.json'
            write_json(config, cfg)
            config_sha = digest(config)
            command = [sys.executable, '-B', str(ROOT/'diagnostics/resource_probe/runner.py'), '--mode', MODE,
                       '--config', str(config), '--output', str(folder), '--steps', str(steps)]
            active = name
            print(f'Launching {name}, capped at {steps} physics substeps.', flush=True)
            with (out/f'{name}_launcher.log').open('x', encoding='utf-8') as stream:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
                attempts.append(name)
                try:
                    while proc.poll() is None:
                        guard()
                        time.sleep(1)
                finally:
                    if proc.poll() is None:
                        stop_owned_tree(proc)
                        proc.wait(timeout=15)
            result = json.loads((folder/'result.json').read_bytes())
            worker = result.get('worker') or {}
            if (proc.returncode or result['status'] != 'completed' or not result['worker_started']
                    or worker.get('status') != 'training_timing_capture_completed'):
                raise RuntimeError(result.get('stop_reason') or worker.get('error') or result.get('error') or result['status'])
            if (result['config_sha256'] != config_sha or digest(config) != config_sha
                    or any(sources.get(path) != value for path, value in result['source_sha256'].items())
                    or worker['physics_steps'] != steps or worker['prefix_substeps'] != limits['prefix_substeps_per_case']
                    or worker['completed_branches'] != plan['branch_order'] or worker['case'] != evidence['case']):
                raise ValueError('completed worker scope or source evidence differs from the frozen plan')
            completed.append(case_entry(out, folder, plan, cfg, evidence))
            active = None
            write_json(out/'progress.json', {'completed': [entry['episode_id'] for entry in completed],
                       'worker_starts': len(attempts), 'physics_steps': len(completed)*steps})

        verify_sources()
        load(ROOT)
        guard()
        if (len(completed) != limits['max_worker_starts'] or len(completed)*steps != limits['max_physics_substeps']
                or len({entry['physical_initial_sha256'] for entry in completed}) != len(completed)):
            raise ValueError('all distinct training cases are required for publication')
        manifest = {'schema': plan['data_contract']['schema'], 'horizon': plan['horizon'],
                    'development_holdout_group': plan['development_holdout']['group'], 'design_sha256': DESIGN_SHA256,
                    'trace_fields': {'requested_actions_Hx2': 'actions', 'executed_joint_q': 'commands[].q',
                                     'executed_joint_qd': 'commands[].qd', 'before_reset_public_states': 'states',
                                     'tool_frame_p_v_spin_si': 'states_raw', 'contact_candidates_and_shape_pairs': 'contacts',
                                     'retention_and_ending': ['retention', 'terminated', 'truncated', 'termination_reason']},
                    'entries': completed}
        write_json(out/'capture_manifest.json', manifest)
        write_json(out/'validation.json', {'status': 'completed', 'worker_starts': len(attempts),
                   'snapshots': len(completed), 'branches': len(completed)*len(plan['branch_order']),
                   'physics_steps': len(completed)*steps, 'case_prefix_substeps': limits['prefix_substeps_per_case'],
                   'training_updates': 0, 'test_episode_files_read': False, 'control_accepted': False,
                   'wall_s': time.monotonic()-began, 'artifacts_sha256': {p.relative_to(out).as_posix(): digest(p)
                       for p in sorted(out.rglob('*')) if p.is_file()}})
    except BaseException as error:
        last = {}
        if active is not None:
            try:
                last = json.loads((out/active/'worker_result.json').read_bytes())
            except (OSError, ValueError):
                pass
        write_json(out/'stopped.json', {'reason': str(error), 'runner_attempts': attempts, 'active_episode': active,
                   'completed_episodes': [entry['episode_id'] for entry in completed],
                   'last_active_physics_steps': last.get('physics_steps'), 'wall_s': time.monotonic()-began,
                   'training_updates': 0, 'test_episode_files_read': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    output = parser.parse_args().output.resolve()
    if not output.relative_to(ROOT/'runs').parts:
        raise ValueError('choose a fresh run subdirectory')
    run(output)
