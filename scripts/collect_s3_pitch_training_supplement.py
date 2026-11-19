"""Bind the negative-y advance evidence to the existing one-snapshot capture."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import collect_s3_training_timing_advance as capture
from ball_retention.archive_restore_plan import _verified_bytes


ROOT = capture.ROOT
EXECUTION = 'configs/s3_pitch_training_supplement_capture_plan.json'


def readiness(path, expected_sha):
    execution_path = path.relative_to(ROOT).as_posix()
    if execution_path != EXECUTION:
        raise ValueError('expected the frozen negative-y advance execution plan')
    proof = json.loads(_verified_bytes(ROOT, execution_path, expected_sha))
    if proof['schema'] != 's3-pitch-training-supplement-capture-v1':
        raise ValueError('unexpected supplement capture scope')
    files = proof['files_sha256']
    for name, sha in files.items():
        _verified_bytes(ROOT, name, sha)

    def read(p):
        return json.loads(_verified_bytes(ROOT, p, files[p]))

    accepted = read(proof['static_validation'])
    if (accepted['status'] != 'completed' or accepted['plan_sha256'] != files[proof['static_plan']]
            or not accepted['deterministic_controller_preflight_passed']
            or accepted['simulator_worker_started'] or accepted['simulation_steps'] != 0
            or accepted['updates'] != 0 or accepted['test_episode_files_read'] is not False):
        raise ValueError('static controller preflight not accepted')
    for name in ('commands', 'capture_preflight'):
        artifact = proof[name]
        if files[artifact] != accepted['artifacts_sha256'][Path(artifact).name]:
            raise ValueError(f'static preflight artifact changed: {artifact}')
    timing = read(proof['static_plan'])['timing_design']
    ref = read(timing['reference_design'])
    evidence = read(proof['commands'])
    report = read(proof['capture_preflight'])
    case = evidence['case']
    limits = timing['budget']
    horizon, substeps, tick = (timing[k] for k in ('horizon', 'control_substeps', 'snapshot_tick'))
    count = substeps*(tick+horizon*len(ref['branch_order']))
    if (case['episode_id'] != accepted['training_case'] or case != report['candidate']
            or case['split'] != 'train' or case['history_id'] != 'advance' or case['group_id'] == ref['case']['group']
            or not all(case[k] == v for k, v in timing['cases'][0].items())
            or evidence['snapshot_tick'] != tick or list(evidence['branches']) != ref['branch_order']
            or len(evidence['prefix_requested']) != tick or len(evidence['prefix_executed']) != tick
            or count != limits['max_physics_substeps'] or count != limits['max_physics_substeps_per_worker']
            or limits['max_worker_starts'] != 1 or limits['sequential_environments'] != 1
            or not report['shared_first_eight_commands_exact']):
        raise ValueError('static case or physical reservation changed')
    worker_plan = {**timing, 'case': case, 'branch_order': ref['branch_order'], 'directions': ref['directions'],
                   'development_holdout_group': ref['case']['group']}
    config = capture.build_config(read(case['source_config']), worker_plan)
    old_backend = read(proof['backend_acceptance'])
    if not old_backend['passed'] or int(old_backend['backend']['device']['native_sm']) != timing['runtime']['cuda_arch']:
        raise ValueError('accepted deterministic backend or device changed')
    return worker_plan, config, evidence, read(proof['layout']), {
        'backend': old_backend['backend'], 'source_sha256': files,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('capture', 'worker'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preflight', required=True, type=Path)
    parser.add_argument('--preflight-sha256', required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.relative_to(ROOT/'runs').parts:
        parser.error('select an output below runs')
    # Use the existing capture monitor, worker and complete-manifest publication.
    # The spawned child runs this same entry point, so both bind identical evidence.
    with patch.multiple(capture, load_ready=readiness, DESIGN_SHA=args.preflight_sha256, __file__=__file__):
        entry = capture.capture if args.phase == 'capture' else capture.worker
        entry(out, args.preflight.resolve(), args.preflight_sha256)
