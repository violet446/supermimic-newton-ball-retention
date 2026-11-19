"""Run the predeclared S2 pilot through the headless resource supervisor."""
import copy
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.dataset import write_manifest


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    run_dir = ROOT/'runs/s2_pilot_20261112'
    dataset_dir = ROOT/'data/s2_pilot_20261112'
    cases_file = run_dir/'cases.json'
    base_file = ROOT/'configs/s1_online_episode.json'
    cases = json.loads(cases_file.read_text(encoding='utf-8'))['cases']
    base = json.loads(base_file.read_text(encoding='utf-8'))
    preflight_path = run_dir/'preflight.json'
    preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
    if not preflight['passed'] or any(digest(ROOT/name) != sha for name, sha in preflight['source_sha256'].items()):
        raise ValueError('pilot inputs differ from their CPU preflight')
    dt, hz = base['physics']['dt_s'], base['physics']['control_hz']
    substeps = round(1/(dt*hz))
    max_steps, wall_s = 27000, 1200
    master_seed = 20261111
    expected = {('train', 'route'):4, ('train', 'edge'):1, ('validation', 'route'):1,
                ('validation', 'edge'):1, ('test', 'route'):1, ('test', 'edge'):1}
    if Counter((case['split'], case['kind']) for case in cases) != expected:
        raise ValueError('scenario set differs from the declared pilot scope')
    group_splits, initial_splits = {}, {}
    if len({case['id'] for case in cases}) != 9 or len({case['seed'] for case in cases}) != 9:
        raise ValueError('episode IDs and seeds must be unique')
    for case in cases:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', case['id']):
            raise ValueError('invalid episode ID')
        if case['horizon_s'] != (4. if case['kind'] == 'route' else 1.):
            raise ValueError('invalid pilot horizon')
        if set(case['initial'])-{'ball_center_world_m', 'ball_velocity_world_m_s'}:
            raise ValueError('pilot only varies ball position and velocity')
        state_key = json.dumps(base['initial'] | case['initial'], sort_keys=True)
        if group_splits.setdefault(case['group_id'], case['split']) != case['split']:
            raise ValueError('initial-state group crosses splits')
        if initial_splits.setdefault(state_key, case['split']) != case['split']:
            raise ValueError('identical initial state crosses splits')
    reserved = sum(round(case['horizon_s']*hz)*substeps+32 for case in cases)
    if reserved > max_steps:
        raise ValueError('pilot substep reservation exceeds budget')

    dataset_dir.mkdir(parents=True, exist_ok=False)
    config_dir = run_dir/'configs'
    config_dir.mkdir(exist_ok=False)
    dataset_relative = dataset_dir.relative_to(ROOT).as_posix()
    plan = {'schema': 's2-pilot-plan-v1', 'master_seed': master_seed, 'environments': 1,
            'max_episodes': 9, 'max_physics_steps': max_steps, 'max_wall_s': wall_s,
            'reserved_physics_steps_including_replay': reserved,
            'dataset': dataset_relative, 'cases_sha256': digest(cases_file), 'base_config_sha256': digest(base_file),
            'base_config': base_file.relative_to(ROOT).as_posix(), 'collector_sha256': digest(Path(__file__)),
            'preflight_sha256':digest(preflight_path),
            'max_abs_request': .4, 'resource_limits': base['budget'] | {'wall_s':wall_s, 'max_cases':9, 'max_physics_steps':max_steps},
            'failure_policy': 'log pre-physics planner rejection and proceed to the next fixed case; halt on physics, data or resource failure',
            'admission_checks': ['all nine cases valid', 'both ending types in each split',
                                 'positive and negative requests for both axes', 'at least one route with >=5mm tool-frame ball travel']}
    write_json(run_dir/'plan.json', plan)
    started = time.monotonic()
    attempts, entries = [], []
    physics_steps = 0

    def save(status, **extra):
        write_json(run_dir/'summary.json', {'status':status, 'dataset':dataset_relative, 'attempts':attempts,
                  'physics_steps':physics_steps, 'wall_s':time.monotonic()-started, **extra})

    save('collecting')
    for case in cases:
        remaining = wall_s-(time.monotonic()-started)
        trace_steps = round(case['horizon_s']*hz)*substeps
        if remaining <= 0 or physics_steps+trace_steps+32 > max_steps:
            save('stopped_budget')
            return 1
        config = copy.deepcopy(base)
        config['initial'].update(case['initial'])
        config['seed'] = case['seed']
        config['episode']['horizon_s'] = case['horizon_s']
        config['budget'].update(wall_s=remaining, max_cases=9, max_physics_steps=max_steps, max_trace_steps=trace_steps)
        config['collection'] = {'master_seed':master_seed, 'dataset_dir':dataset_relative,
                                'case':case, 'max_abs_request':plan['max_abs_request']}
        cfg_path = config_dir/(case['id']+'.json')
        write_json(cfg_path, config)
        episode_dir = dataset_dir/'episodes'/case['id']
        command = [sys.executable, '-B', str(ROOT/'diagnostics/resource_probe/runner.py'), '--mode', 's2-collect',
                   '--config', str(cfg_path), '--output', str(episode_dir), '--steps', str(trace_steps)]
        print(f"Collecting {case['id']} ({case['split']}, {case['kind']})", flush=True)
        with (run_dir/(case['id']+'.launcher.log')).open('x', encoding='utf-8') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result_path = episode_dir/'result.json'
        if not result_path.exists():
            save('stopped_supervisor', failed_episode=case['id'], exit_code=process.returncode)
            return 1
        result = json.loads(result_path.read_text(encoding='utf-8'))
        worker = result.get('worker') or {}
        physics_steps += worker.get('physics_steps', 0)
        metrics = worker.get('episode')
        attempt = {'id':case['id'], 'split':case['split'], 'kind':case['kind'], 'status':result['status'],
                   'physics_steps':worker.get('physics_steps', 0), 'wall_s':result.get('wall_s'),
                   'planning_rejections':worker.get('planning_rejections', 0),
                   'replay_max_error':worker.get('replay_max_error'), 'episode':metrics,
                   'error':worker.get('error'), 'stop_reason':result.get('stop_reason'),
                   'evidence':result_path.relative_to(ROOT).as_posix()}
        attempts.append(attempt)
        save('collecting')
        if process.returncode != 0 or result['status'] != 'completed':
            if attempt['planning_rejections'] and result['status'] == 'failed':
                print(f"  Planning rejected: {attempt['error']}", flush=True)
                continue
            save('stopped_case_failure', failed_episode=case['id'])
            return 1
        if worker.get('status') != 'collection_completed' or metrics is None:
            save('stopped_validation', failed_episode=case['id'])
            return 1
        entries.append({'episode_id':case['id'], 'group_id':case['group_id'], 'split':case['split'],
                        'path':(episode_dir/worker['data_file']).relative_to(dataset_dir).as_posix(),
                        'sha256':metrics['sha256'], 'provenance':{'seed':case['seed'], 'kind':case['kind'],
                        'source_config_sha256':result['config_sha256'],
                        'config_sha256':digest(episode_dir/'effective_config.json'),
                        'config':(episode_dir/'effective_config.json').relative_to(dataset_dir).as_posix(),
                        'replay_max_error':worker['replay_max_error']}})
        print(f"  {metrics['transitions']} transitions, terminated={metrics['terminated']}", flush=True)

    completed = [attempt for attempt in attempts if attempt['episode'] is not None]
    metric_rows = [attempt['episode'] for attempt in completed]
    split_endings = {split: {label:sum(attempt['episode'][label] for attempt in completed if attempt['split']==split)
                             for label in ('terminated', 'truncated')} for split in ('train','validation','test')}
    action_min = [min(row['requested_action_min'][i] for row in metric_rows) for i in range(2)] if metric_rows else []
    action_max = [max(row['requested_action_max'][i] for row in metric_rows) for i in range(2)] if metric_rows else []
    ball_travel = max((attempt['episode']['max_ball_tool_displacement_m'] for attempt in completed if attempt['kind']=='route'), default=0.)
    passed = (len(entries)==9 and all(all(counts.values()) for counts in split_endings.values())
              and len(action_min)==2 and all(lo<0<hi for lo, hi in zip(action_min,action_max)) and ball_travel>=.005)
    details = {'episode_count':len(entries), 'transitions':sum(row['transitions'] for row in metric_rows),
               'split_endings':split_endings, 'requested_action_min':action_min, 'requested_action_max':action_max,
               'max_route_ball_tool_displacement_m':ball_travel, 'pilot_checks_passed':passed}
    if not passed:
        save('pilot_checks_failed', **details)
        return 1
    try:
        write_manifest(dataset_dir, entries, dt, substeps, provenance={
            'plan':(run_dir/'plan.json').relative_to(ROOT).as_posix(), 'plan_sha256':digest(run_dir/'plan.json'),
            'cases_sha256':plan['cases_sha256'], 'master_seed':master_seed, 'stage':'S2 pilot', 'pilot_checks_passed':True})
    except (OSError, ValueError) as exc:
        save('stopped_manifest_failure', error=str(exc), **details)
        raise
    save('completed', manifest=f'{dataset_relative}/manifest.json', **details)
    return 0


if __name__ == '__main__':
    sys.exit(run())
