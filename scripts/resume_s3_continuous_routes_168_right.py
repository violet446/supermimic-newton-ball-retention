"""Validate the Windows sharing stop and resume its one reserved route tail."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import run_s3_continuous_routes as capture
from s3_continuous_route_worker import CHECKPOINT, TRACE, atomic_json, checkpoint


BASE = capture.BASE
NAME = 'continuous_168_right_nominal'
EXPECTED_STOP = '[Errno 13] Permission denied:'
OLD_CAP = 46080
NEW_CAP = 46096


def retry_pid(lookup, path, fallback):
    """Keep polling the original guard after a transient sharing violation."""
    for attempt in range(6):
        try:
            return lookup(path, fallback)
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.01)


def inspect_stop():
    stop_file = BASE/'stopped.json'
    progress_file = BASE/'batch_progress.json'
    stopped = json.loads(stop_file.read_bytes())
    progress = json.loads(progress_file.read_bytes())
    path = BASE/'preflight.json'
    sha = capture.digest(path)
    plan = capture.plan_at(path, sha)
    expected_done = [case['episode_id'] for case in plan['cases'][:-1]]
    attempt = stopped['active_attempt']
    folder = BASE/NAME
    saved, prefix = checkpoint(folder, NAME)
    tail = capture.EPISODE_CAP-saved['committed_physics_steps']
    if (stopped['plan_sha256'] != sha or progress['plan_sha256'] != sha
            or not stopped['reason'].startswith(EXPECTED_STOP) or not stopped['reason'].endswith("worker_result.json'")
            or plan['cases'][-1]['episode_id'] != NAME or stopped['completed'] != expected_done
            or progress['completed'] != expected_done or len(progress['attempts']) != 12
            or progress['status'] != 'stopped' or progress['restartable'] is not False
            or progress['attempts'][-1] != attempt or attempt['episode'] != NAME
            or (saved['control_rows'], saved['terminal'], attempt['final_stage'], attempt['committed_steps'],
                attempt['physics_steps_upper_bound']) != (146, False, 'control_in_flight', 2336, 2352)
            or len(prefix) != (folder/TRACE).stat().st_size
            or progress['physical_steps_upper_bound'] != stopped['physical_steps_upper_bound']
            or stopped['physical_steps_upper_bound']+tail != NEW_CAP
            or capture.LIMITS['max_physics_steps'] != OLD_CAP):
        raise ValueError('the stopped batch differs from the one recoverable sharing violation')
    # Keep the monitored worker PID unchanged across the transient read failure.
    calls = iter((PermissionError('synthetic sharing violation'), 12345))

    def fake_lookup(_, fallback):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    if retry_pid(fake_lookup, Path('fake'), 100) != 12345:
        raise ValueError('PID reader did not recover after a sharing violation')
    sources = [Path(__file__).resolve(), path, stop_file, progress_file, folder/CHECKPOINT, folder/TRACE,
               folder/'attempt_001/worker/worker_result.json']
    return path, sha, plan, progress, {'case': NAME, 'restorable_after_control': saved['control_rows'],
            'committed_physics_steps': saved['committed_physics_steps'], 'tail_max_physics_steps': tail,
            'original_batch_physics_cap': OLD_CAP, 'max_physics_steps_if_approved': NEW_CAP,
            'input_sha256': {capture.relative(p): capture.digest(p) for p in sources},
            'pid_retry_static_test': 'passed', 'new_physics_steps': 0, **capture.SCOPE}


def resume(approved_cap):
    path, sha, plan, progress, evidence = inspect_stop()
    if approved_cap != NEW_CAP:
        raise ValueError('explicitly authorized revised cap required')
    retained = {BASE/'stopped.json': BASE/'original_stopped.json',
                BASE/'batch_progress.json': BASE/'original_batch_progress.json',
                BASE/NAME/CHECKPOINT: BASE/'original_tail_checkpoint.json'}
    for original, archived in retained.items():
        with archived.open('xb') as archive:
            archive.write(original.read_bytes())
    atomic_json(BASE/'tail_resume_plan.json', {**evidence, 'approved_physics_cap': approved_cap,
                'retained_originals': {capture.relative(p): capture.relative(q) for p, q in retained.items()}})
    atomic_json(BASE/'batch_progress.json', {**progress, 'restartable': True,
                'approved_physics_cap': approved_cap, 'original_stop_evidence': 'original_stopped.json'})

    limits = capture.LIMITS
    original_plan_at = capture.plan_at
    original_lookup = capture.monitor.__globals__['monitored_pid']

    def verify_frozen_plan(plan_path, plan_sha):
        with patch.object(capture, 'LIMITS', limits):
            return original_plan_at(plan_path, plan_sha)

    def monitored_pid(worker_path, launcher_pid):
        return retry_pid(original_lookup, worker_path, launcher_pid)

    with (patch.object(capture, 'LIMITS', {**limits, 'max_physics_steps': approved_cap}),
          patch.object(capture, 'plan_at', verify_frozen_plan),
          patch.dict(capture.monitor.__globals__, {'monitored_pid': monitored_pid})):
        capture.capture(plan, path, sha, resume=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-physical-step-cap', type=int)
    args = parser.parse_args()
    if args.approved_physical_step_cap is None:
        *_, check = inspect_stop()
        atomic_json(BASE/'tail_recovery_preflight.json', check)
        print(json.dumps(check, indent=2))
    else:
        resume(args.approved_physical_step_cap)
