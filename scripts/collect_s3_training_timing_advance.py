"""Preflight and collect the one frozen advance-history training snapshot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

from check_s3_deterministic_archive_import import ROOT, backend_preflight, digest, write_json
from collect_s3_training_timing import case_entry
from preflight_s3_training_timing_capture import load_cases, verify_case
from runner import monitored_pid, stop_owned_tree, storage_admission

from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.deterministic_archive_import_plan import load as load_archive
from ball_retention.guard import process_memory, stop_reason
from ball_retention.resources import admission, directory_bytes, snapshot
from ball_retention.timed_pair_data import SCHEMA, prepare_timed_pairs


DESIGN = 'configs/s3_training_timing_advance_plan.json'
DESIGN_SHA = '805e137c8bd3866ea145a9edc3290084fa767ea6ad59db3d6ebbce971f5ad1b5'
CONTROLLER_PROBE = 'runs/s3_velocity_training_coverage_20261115/advance_controller_feasibility.json'
CONTROLLER_PROBE_SHA = 'b40b40cef5f9c77f34b22ac89f4174c2793ef47d80232f38ce347f0f7ca9ecda'
SCOPE = {'training_updates': 0, 'test_episode_files_read': False, 'control_accepted': False}


def read(path, expected):
    return json.loads(_verified_bytes(ROOT, path, expected))


def design_inputs():
    plan = read(DESIGN, DESIGN_SHA)
    hashes = plan['inputs_sha256']
    original = read(plan['original_timing_design'], hashes[plan['original_timing_design']])
    ref = read(plan['reference_design'], hashes[plan['reference_design']])
    case = plan['case']
    limits = plan['budget']
    steps = plan['control_substeps']*(plan['snapshot_tick']+plan['horizon']*len(plan['branch_order']))
    if (plan['schema'] != 's3-training-timing-advance-supplement-v1' or case['split'] != 'train'
            or plan['branch_order'] != ref['branch_order'] or plan['branch_order'][:2] != ['nominal_1', 'nominal_2']
            or any(plan[k] != original[k] for k in ('runtime', 'snapshot_tick', 'control_substeps', 'horizon', 'amplitude'))
            or plan['development_holdout_group'] != original['development_holdout']['group']
            or plan['runtime'] != ref['runtime'] or limits['max_worker_starts'] != 1
            or limits['sequential_environments'] != 1 or steps != limits['max_physics_substeps']
            or steps != limits['max_physics_substeps_per_worker']
            or limits['prefix_substeps_per_case'] != plan['snapshot_tick']*plan['control_substeps']):
        raise ValueError('advance plan does not match the frozen single-case scope')
    original_manifest = read(plan['existing_timed_manifest'], hashes[plan['existing_timed_manifest']])
    existing = original_manifest['entries']
    if (len(existing) != 2 or any(e['episode_id'] == case['episode_id'] or e['history_id'] != 'reverse' for e in existing)
            or sum(e['group_id'] == case['group_id'] for e in existing) != 1):
        raise ValueError('advance case must add a distinct history within an existing training group')
    # The original loader binds provenance, assets, controls and split. Reverse
    # source traces are unnecessary here, since the only case is advance.
    inherited = {k: v for k, v in original['inputs_sha256'].items()
                 if not any(row['episode_id'] in k for row in original['cases'])}
    adapted = {**original, **{k: plan[k] for k in ('budget', 'static_budget', 'report_leads')},
               'cases': [{k: case[k] for k in ('episode_id', 'group_id', 'split', 'history_id', 'seed')}],
               'inputs_sha256': {**inherited, **hashes}}
    return plan, adapted, existing


def build_config(cfg, plan):
    cfg.pop('history_capture')
    cfg['training_timing_advance'] = {'design_sha256': DESIGN_SHA, 'episode_id': plan['case']['episode_id']}
    limits = plan['budget']
    cfg['budget'].update(wall_s=limits['max_worker_wall_s'], max_cases=1,
        max_physics_steps=limits['max_physics_substeps_per_worker'], max_trace_steps=limits['max_physics_substeps_per_worker'],
        max_process_rss_gib=limits['max_process_rss_gib'], max_process_commit_gib=limits['max_process_commit_gib'],
        max_gpu_used_gib=limits['max_total_gpu_used_gib'], min_start_ram_gib=limits['min_start_ram_gib'],
        min_start_gpu_free_gib=limits['min_start_gpu_free_gib'], min_disk_free_gib=limits['min_disk_free_gib'],
        max_new_output_mib=limits['max_worker_output_mib'],
        max_output_cache_gib=limits['worker_output_plus_shared_cache_and_temp_gib'])
    return cfg


def preflight(out):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    try:
        plan, adapted, existing = design_inputs()
        limits = plan['static_budget']

        def guard():
            memory = process_memory(os.getpid())
            if (time.monotonic()-began >= limits['max_wall_s']
                    or memory['peak_rss_bytes'] > limits['max_process_rss_gib']*1024**3
                    or memory['peak_private_commit_bytes'] > limits['max_process_commit_gib']*1024**3
                    or directory_bytes(out) > limits['max_new_output_mib']*1024**2):
                raise RuntimeError('advance static preflight resource cap')
            return memory

        guard()
        backend = backend_preflight(plan)
        if int(backend['device']['native_sm']) != plan['runtime']['cuda_arch']:
            raise ValueError('unexpected CUDA architecture')
        archive, _, _ = load_archive(ROOT)
        layout = archive['native_reference']['layout']
        coverage = read(plan['coverage_validation'], plan['inputs_sha256'][plan['coverage_validation']])
        selected = coverage['selected_training_option']
        if (coverage['status'] != 'completed'
                or coverage['artifact_sha256'][Path(plan['coverage_report']).name] != plan['inputs_sha256'][plan['coverage_report']]
                or (selected['episode'], selected['group'], selected['history'], selected['tick']) !=
                   (plan['case']['episode_id'], plan['case']['group_id'], plan['case']['history_id'], plan['snapshot_tick'])):
            raise ValueError('coverage audit provenance differs')
        design, cases = load_cases(ROOT, adapted)
        (case, cfg, rows, gap), = cases
        same_group, = [entry for entry in existing if entry['group_id'] == case['group_id']]
        if case['physical_initial_sha256'] != same_group['physical_initial_sha256']:
            raise ValueError('advance and reverse do not preserve the initial-group geometry')
        payload, summary = verify_case(ROOT, design, case, cfg, rows, gap, guard)
        old = read(CONTROLLER_PROBE, CONTROLLER_PROBE_SHA)
        if (old['status'] != 'passed' or old['design_sha256'] != DESIGN_SHA or old['summary'] != summary
                or old['prefix_actions_sha256'] != payload['prefix_actions_sha256']
                or (old['physics_steps'], old['training_updates'], old['test_episode_files_read']) != (0, 0, False)):
            raise ValueError('analytic commands did not reproduce the archived advance feasibility check')
        write_json(out/'commands.json', payload)
        write_json(out/'worker_config.json', build_config(cfg, plan))
        source_paths = [Path(__file__).resolve(), ROOT/'requirements.lock',
                        *sorted((ROOT/'src/ball_retention').glob('*.py')),
                        *sorted((ROOT/'diagnostics/resource_probe').glob('*.py')),
                        *(ROOT/'scripts'/name for name in ('check_s3_actions.py', 'check_s3_deterministic_archive_import.py',
                           'collect_s3_training_timing.py', 'preflight_s3_training_timing_capture.py',
                           'preflight_s3_deterministic_action_response.py'))]
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('static preflight imported the simulator')
        write_json(out/'preflight.json', {'passed': True, 'design_sha256': DESIGN_SHA, 'backend': backend,
            'case': case, 'command_checks': summary, 'worker_budget': plan['budget'], 'static_budget': limits,
            'native_layout': {'path': layout, 'sha256': archive['inputs_sha256'][layout]},
            'source_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths},
            'artifacts_sha256': {name: digest(out/name) for name in ('commands.json', 'worker_config.json')},
            'physical_substeps': 0, 'worker_starts': 0, **SCOPE, 'wall_s': time.monotonic()-began, **guard()})
        guard()
    except BaseException as error:
        write_json(out/'stopped.json', {'reason': str(error), 'physical_substeps': 0, 'worker_starts': 0, **SCOPE})
        raise


def load_ready(preflight_path, expected_sha):
    plan, adapted, _ = design_inputs()
    path = preflight_path.relative_to(ROOT).as_posix()
    accepted = read(path, expected_sha)
    if (accepted['passed'] is not True or accepted['design_sha256'] != DESIGN_SHA
            or accepted['worker_budget'] != plan['budget'] or accepted['static_budget'] != plan['static_budget']
            or any(accepted[key] != value for key, value in SCOPE.items())
            or (accepted['worker_starts'], accepted['physical_substeps']) != (0, 0)):
        raise ValueError('advance preflight did not pass within the frozen scope')
    for path, sha in accepted['source_sha256'].items():
        _verified_bytes(ROOT, path, sha)
    for path, sha in adapted['inputs_sha256'].items():
        _verified_bytes(ROOT, path, sha)
    folder = preflight_path.parent.relative_to(ROOT)
    cfg, evidence = (read(folder/name, accepted['artifacts_sha256'][name])
                     for name in ('worker_config.json', 'commands.json'))
    source_cfg = read(plan['case']['source_config'], plan['inputs_sha256'][plan['case']['source_config']])
    if (cfg != build_config(source_cfg, plan) or evidence['case'] != accepted['case']
            or any(evidence['case'][k] != value for k, value in plan['case'].items())
            or evidence['snapshot_tick'] != plan['snapshot_tick'] or list(evidence['branches']) != plan['branch_order']
            or len(evidence['prefix_requested']) != plan['snapshot_tick']
            or len(evidence['prefix_executed']) != plan['snapshot_tick']):
        raise ValueError('worker config or controller evidence changed')
    layout = accepted['native_layout']
    plan = {**plan, 'directions': read(plan['reference_design'], plan['inputs_sha256'][plan['reference_design']])['directions']}
    return plan, cfg, evidence, read(layout['path'], layout['sha256']), accepted


def worker(out, preflight_path, expected_sha):
    began = time.monotonic()
    state = {'status': 'starting', 'worker_pid': os.getpid(), 'physics_steps': 0}

    def save():
        write_json(out/'worker_result.json', state)

    def stage(name):
        state.setdefault('phase_events', []).append({'stage': name, 'elapsed_s': time.monotonic()-began,
                                                    **process_memory(os.getpid())})
        save()

    save()
    try:
        plan, cfg, evidence, layout, _ = load_ready(preflight_path, expected_sha)
        cap = plan['budget']['max_physics_substeps_per_worker']
        if not admission(snapshot(ROOT), cfg['budget'])['passed']:
            raise RuntimeError('advance worker resource recheck failed')
        import training_timing_capture

        def validate(root, candidate, steps):
            if root != ROOT or candidate != cfg or type(steps) is not int or steps != cap:
                raise ValueError('worker arguments differ from the accepted advance case')
            return plan, evidence, layout

        # This process runs one case. Reuse the existing physics and gates with
        # this case's preflight binding, keeping the two-case entry point intact.
        with patch.object(training_timing_capture, 'validate_case', validate):
            training_timing_capture.run(ROOT, cfg, out, cap, state, stage)
        state['final_memory'] = process_memory(os.getpid())
        save()
    except BaseException as error:
        state.update(status='stopped', error=str(error))
        save()
        raise


def capture(out, preflight_path, expected_sha):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    proc = None
    reason = None
    worker_folder = out/'worker'
    try:
        plan, cfg, evidence, _, static = load_ready(preflight_path, expected_sha)
        if backend_preflight(plan) != static['backend']:
            raise ValueError('installed backend or device differs from advance preflight')
        limits = plan['budget']
        budget = cfg['budget']
        write_json(out/'execution_plan.json', {'design': plan, 'design_sha256': DESIGN_SHA,
            'preflight_sha256': expected_sha, 'preflight_path': preflight_path.relative_to(ROOT).as_posix(),
            'source_sha256': static['source_sha256'], 'worker_config': cfg})
        sample = snapshot(ROOT)
        hardware = admission(sample, budget)
        storage = storage_admission(ROOT, worker_folder, budget)
        env_bytes = directory_bytes(ROOT/'.venv')
        write_json(out/'admission.json', {'sample': sample, 'hardware': hardware, 'storage': storage,
                                         'environment_bytes': env_bytes})
        if not hardware['passed'] or not storage['passed'] or env_bytes > budget['max_environment_gib']*1024**3:
            raise RuntimeError('advance capture resource admission failed')
        worker_folder.mkdir()
        env = os.environ.copy()
        for key, value in {'TEMP': '.tmp', 'TMP': '.tmp', 'TMPDIR': '.tmp',
                           'WARP_CACHE_PATH': '.cache/warp', 'NEWTON_CACHE_PATH': '.cache/newton'}.items():
            folder = ROOT/value
            folder.mkdir(parents=True, exist_ok=True)
            env[key] = str(folder)
        command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--output', str(worker_folder),
                   '--preflight', str(preflight_path), '--preflight-sha256', expected_sha]
        if time.monotonic()-began >= limits['max_wall_s']:
            raise RuntimeError('advance capture wall cap before worker launch')
        started = time.monotonic()
        low_since = None
        with (worker_folder/'worker.log').open('x', encoding='utf-8') as log, (out/'resources.jsonl').open('x', encoding='utf-8') as trace:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            while proc.poll() is None:
                now = time.monotonic()
                sample = snapshot(ROOT)
                if sample['available_ram_bytes'] < budget['min_running_ram_gib']*1024**3:
                    low_since = now if low_since is None else low_since
                else:
                    low_since = None
                if proc.poll() is not None:
                    break
                pid = monitored_pid(worker_folder/'worker_result.json', proc.pid)
                try:
                    mem = process_memory(pid)
                except OSError:
                    if proc.poll() is not None:
                        break
                    raise
                worker_storage = storage_admission(ROOT, worker_folder, budget)
                total = directory_bytes(out)
                reason = stop_reason(sample, budget, now-started, mem['peak_rss_bytes'],
                                     worker_storage['total_bytes'], 0 if low_since is None else now-low_since,
                                     mem['peak_private_commit_bytes'])
                if now-began >= limits['max_wall_s']:
                    reason = 'end_to_end_wall_budget'
                if total > limits['max_new_output_mib']*1024**2 or worker_storage['output_bytes'] > budget['max_new_output_mib']*1024**2:
                    reason = 'new_output_budget'
                trace.write(json.dumps({'elapsed_s': now-started, 'worker_pid': pid, 'stop_reason': reason,
                                        'output_bytes': total, **mem, **sample})+'\n')
                trace.flush()
                if reason:
                    stop_owned_tree(proc)
                    break
                time.sleep(1)
            proc.wait(timeout=15)
        result = json.loads((worker_folder/'worker_result.json').read_bytes())
        if (reason or proc.returncode or result['status'] != 'training_timing_capture_completed'):
            raise RuntimeError(reason or result.get('error') or result['status'])
        cap = limits['max_physics_substeps']
        if (result['case'] != evidence['case'] or result['physics_steps'] != cap
                or result['prefix_substeps'] != limits['prefix_substeps_per_case']
                or result['completed_branches'] != plan['branch_order']
                or result['nominal_passed'] is not True or result['shared_prefix_passed'] is not True
                or result['final_memory']['peak_rss_bytes'] > budget['max_process_rss_gib']*1024**3
                or result['final_memory']['peak_private_commit_bytes'] > budget['max_process_commit_gib']*1024**3):
            raise ValueError('worker result differs from the frozen capture scope')
        # Reuse publication checks and bind the same result path as the original schema.
        write_json(worker_folder/'result.json', result)
        entry = case_entry(out, worker_folder, plan, cfg, evidence)
        pending = out/'capture_manifest.pending.json'
        write_json(pending, {'schema': SCHEMA, 'horizon': plan['horizon'], 'design_sha256': DESIGN_SHA,
                             'development_holdout_group': plan['development_holdout_group'], 'entries': [entry]})
        loaded = prepare_timed_pairs(ROOT, pending.relative_to(ROOT), digest(pending))
        if loaded.directions != tuple(plan['directions']) or len(loaded.batch.initial) != 3:
            raise ValueError('supplement reader did not recover all three training pairs')
        load_ready(preflight_path, expected_sha)
        if time.monotonic()-began >= limits['max_wall_s'] or directory_bytes(out) > limits['max_new_output_mib']*1024**2:
            raise RuntimeError('advance capture exceeded the aggregate budget')
        hashes = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file() and p != pending}
        hashes['capture_manifest.json'] = digest(pending)
        write_json(out/'validation.json', {'status': 'completed', 'worker_starts': 1, 'snapshots': 1,
            'branches': len(plan['branch_order']), 'physics_steps': cap, 'pairs': 3, **SCOPE,
            'case': evidence['case'], 'wall_s': time.monotonic()-began, 'artifacts_sha256': hashes})
        pending.rename(out/'capture_manifest.json')
    except BaseException as error:
        record = worker_folder/'worker_result.json'
        try:
            last = json.loads(record.read_bytes())
        except (OSError, ValueError):
            last = {}
        write_json(out/'stopped.json', {'reason': str(error), 'worker_starts': int(proc is not None),
            'last_reported_physics_substeps': last.get('physics_steps', 0),
            'wall_s': time.monotonic()-began, **SCOPE})
        raise
    finally:
        if proc is not None and proc.poll() is None:
            stop_owned_tree(proc)
            proc.wait(timeout=15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'capture', 'worker'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--preflight-sha256')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.relative_to(ROOT/'runs').parts:
        parser.error('select an output below runs')
    if args.phase == 'preflight':
        preflight(output)
    elif args.preflight is None or args.preflight_sha256 is None:
        parser.error('capture and worker require the accepted preflight path and hash')
    elif args.phase == 'capture':
        capture(output, args.preflight.resolve(), args.preflight_sha256)
    else:
        worker(output, args.preflight.resolve(), args.preflight_sha256)
