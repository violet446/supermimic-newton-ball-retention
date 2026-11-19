"""Preflight and execute the fixed four-snapshot Newton offset probe."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

from check_s3_actions import ROOT, digest, write_json
from collect_s3_refit import preflight
from ball_retention.geometry import load_tool, sphere_clearance
from ball_retention.guard import process_memory
from ball_retention.resources import admission, directory_bytes, snapshot
from ball_retention.robot import Robot
from ball_retention.small_offset_probe import DESIGN, archived_record, build_case, verify_design

sys.path.insert(0, str(ROOT))
from diagnostics.resource_probe.action_pairs import _plan
from diagnostics.resource_probe.runner import stop_owned_tree, storage_admission
from evaluate_s3_small_offset import evaluate


OUT = ROOT/'runs/s3_small_offset_20261114'


def checked_inputs(configs):
    files = list((ROOT/'src/ball_retention').glob('*.py'))
    files.extend((ROOT/'diagnostics/resource_probe').glob('*.py'))
    files += [ROOT/'scripts'/name for name in ('collect_s3_small_offset.py', 'evaluate_s3_small_offset.py',
              'collect_s3_refit.py', 'check_s3_actions.py', 'diagnose_s3_action_response.py',
              'check_s3_group_generalization.py', 'summarize_s3_action_response.py', 'evaluate_s3_coverage.py')]
    files.extend(ROOT/cfg['prescribed']['joint_path'] for cfg in configs)
    files += [ROOT/DESIGN, ROOT/'assets/manifest.json', ROOT/'requirements.lock']
    assets = json.loads((ROOT/'assets/manifest.json').read_bytes())['assets']
    for entry in assets:
        path = ROOT/entry['local_path']
        if digest(path) != entry['sha256']:
            raise ValueError(f'asset changed: {path}')
        files.append(path)
    resolved = {path.resolve() for path in files}
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(resolved)}


def verify_inputs(hashes):
    for path, expected in hashes.items():
        if digest(ROOT/path) != expected:
            raise ValueError(f'frozen execution input changed: {path}')


def run(out):
    started = time.monotonic()
    design = verify_design(ROOT)
    budget = design['budget']
    out.mkdir(exist_ok=False)
    entries, attempts, checks = [], [], []
    steps = 0
    stage = 'prepare'
    active_case = None

    def elapsed():
        return time.monotonic()-started

    def guard():
        if elapsed() >= budget['max_wall_s'] or directory_bytes(out) > budget['max_new_output_mib']*1024**2:
            raise RuntimeError('round wall or new output cap reached')

    def resource_admission(cfg):
        observed = snapshot(ROOT)
        hardware = admission(observed, cfg['budget'])
        storage = storage_admission(ROOT, out, cfg['budget'])
        if not hardware['passed'] or not storage['passed']:
            raise RuntimeError(f'resource admission failed: {hardware}, {storage}, {observed}')
        return {'hardware': hardware, 'storage': storage, 'observed': observed}

    def progress(status, **extra):
        write_json(out/'progress.json', {'status': status, 'stage': stage, 'attempts': attempts,
                   'physics_steps': steps, 'wall_s': elapsed(), **extra})

    try:
        cases = design['cases']
        if (len(cases) != budget['max_worker_starts'] or budget['sequential_environments'] != 1
                or sum(case['reserved_substeps'] for case in cases) != budget['max_physics_substeps']):
            raise ValueError('inconsistent round budget')
        configs = [build_case(ROOT, design, case) for case in cases]
        hashes = checked_inputs(configs)
        config_hashes = {}
        mesh = load_tool(ROOT/'assets/shovel.STL')
        robot = Robot(ROOT/'assets/ec66_description.urdf')
        for case, cfg in zip(cases, configs):
            name = case['episode']
            stage = f'preflight_{name}'
            guard()
            admitted = resource_admission(cfg)
            path = out/f'{name}_config.json'
            write_json(path, cfg)
            config_hashes[path.name] = digest(path)
            _, phys_dt, substeps, ticks, horizon, _, actions, limit = _plan(cfg, case['reserved_substeps'])
            if limit != case['reserved_substeps'] or substeps != design['control_substeps'] or ticks != [case['tick']]:
                raise ValueError(f'case bounds differ from the design: {name}')
            reference = archived_record(ROOT, case)
            np.testing.assert_array_equal(actions[ticks[0]:ticks[0]+horizon], reference['nominal_actions'])
            tool, _, _ = robot.fk(cfg['initial']['joint_q_rad'])
            center = np.array(cfg['initial']['ball_center_world_m'])
            local = tool[:3, :3].T@(center-tool[:3, 3])
            radius = cfg['ball']['radius_m']
            clearance = sphere_clearance(mesh, local, radius)
            if (clearance < -cfg['admission']['max_initial_penetration_m'] or local[0] < .22
                    or center[2]-radius <= cfg['physics']['floor_z_m']):
                raise ValueError(f'initial pose/clearance rejected: {name}')
            motion = preflight(cfg, substeps*phys_dt, substeps)
            if motion[0]['branches_checked'] != design['branches_per_snapshot']-1:
                raise ValueError('static preflight did not cover the nominal and all distinct perturbed requests')
            memory = process_memory(os.getpid())
            if memory['peak_rss_bytes'] > budget['max_process_rss_gib']*1024**3 or memory['peak_private_commit_bytes'] > budget['max_process_commit_gib']*1024**3:
                raise RuntimeError('preflight process memory cap reached')
            checks.append({'episode': name, 'motion': motion, 'nominal_requests_equal': True,
                           'initial_mesh_clearance_m': clearance, 'resources': admitted, 'memory': memory})
            write_json(out/'preflight_progress.json', checks)
            print(f'{name}: static preflight passed', flush=True)
        write_json(out/'execution_plan.json', {'design': design, 'source_sha256': hashes,
                   'case_config_sha256': config_hashes, 'preflight_wall_s': elapsed(), 'simulation_started': False,
                   'packages': {name: version(name) for name in ('newton', 'warp-lang', 'numpy', 'scipy', 'trimesh')}})
        write_json(out/'preflight.json', {'passed': True, 'cases': checks})

        for case, cfg in zip(cases, configs):
            name = case['episode']
            path = out/f'{name}_config.json'
            directory = out/name
            stage = f'capture_{name}'
            guard()
            verify_inputs(hashes)
            if digest(path) != config_hashes[path.name]:
                raise ValueError(f'generated config changed: {name}')
            resource_admission(cfg)
            active_case = name
            progress('running', active_episode=name)
            print(f'{name}: Newton capture started, bound {case["reserved_substeps"]} substeps', flush=True)
            command = [sys.executable, '-B', str(ROOT/'diagnostics/resource_probe/runner.py'),
                       '--mode', 's3-pairs', '--output', str(directory), '--config', str(path),
                       '--steps', str(case['reserved_substeps'])]
            with (out/f'{name}_launcher.log').open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    while process.poll() is None:
                        guard()
                        time.sleep(1)
                finally:
                    if process.poll() is None:
                        stop_owned_tree(process)
                        process.wait(timeout=15)
            result_file = directory/'result.json'
            if not result_file.exists():
                raise RuntimeError(f'worker launch produced no result: {name}')
            result = json.loads(result_file.read_bytes())
            worker = result.get('worker') or {}
            executed = worker.get('physics_steps', 0)
            steps += executed
            attempts.append({'episode': name, 'status': result['status'], 'physics_steps': executed,
                             'worker_started': result.get('worker_started', False), 'wall_s': result.get('wall_s'),
                             'reason': worker.get('error') or result.get('stop_reason')})
            active_case = None
            progress('running')
            if process.returncode or result['status'] != 'completed' or worker.get('status') != 'action_pairs_completed':
                raise RuntimeError(f'capture failed for {name}: {attempts[-1]}')
            if executed != case['reserved_substeps'] or result['config_sha256'] != config_hashes[path.name]:
                raise ValueError(f'worker scope differed from the plan: {name}')
            trace = directory/'action_pairs.jsonl'
            record = json.loads(trace.read_bytes())
            if record['snapshot_tick'] != case['tick'] or len(record['branches']) != design['branches_per_snapshot']:
                raise ValueError(f'incomplete snapshot: {name}')
            for branch in record['branches']:
                if branch['terminated'] or branch['truncated'] or branch['substeps'] != [design['control_substeps']]*design['horizon']:
                    raise ValueError(f'incomplete branch: {name}')
            entries.append({**case, 'trace': trace.relative_to(out).as_posix(), 'sha256': digest(trace)})
            print(f'{name}: complete, {steps} cumulative substeps', flush=True)

        stage = 'evaluate'
        guard()
        verify_inputs(hashes)
        write_json(out/'capture_manifest.json', {'cases': entries, 'physics_steps': steps,
                   'execution_plan_sha256': digest(out/'execution_plan.json')})
        evaluation = evaluate(ROOT, out, design, guard=lambda label: guard())
        guard()
        write_json(out/'validation.json', {'status': 'completed', 'branches': design['branches_total'],
                   'snapshots': len(entries), 'physics_steps': steps, 'wall_s': elapsed(),
                   'training_updates': 0, 'test_episode_files_read': False, 'control_accepted': False, 'evaluation': evaluation,
                   'artifacts_sha256': {path.relative_to(out).as_posix(): digest(path) for path in out.rglob('*')
                                        if path.is_file() and path.name != 'progress.json'}})
        progress('completed')
    except BaseException as error:
        reported = None
        if active_case is not None:
            worker_result = out/active_case/'worker_result.json'
            try:
                reported = json.loads(worker_result.read_bytes()).get('physics_steps')
            except (OSError, ValueError):
                pass
        progress('stopped', error=str(error), active_episode=active_case,
                 active_worker_last_reported_physics_steps=reported)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ROOT/'runs')
    run(out)
