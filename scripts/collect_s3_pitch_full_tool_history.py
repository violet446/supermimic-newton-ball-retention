"""Run the hash-pinned one-snapshot capture for the matched full tool history."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import collect_s3_training_timing_advance as collector
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.pair_collection import _initial_key


ROOT = collector.ROOT


def readiness(path, expected):
    plan = json.loads(_verified_bytes(ROOT, path.relative_to(ROOT).as_posix(), expected))
    if plan['schema'] != 's3-pitch-full-tool-capture-v1':
        raise ValueError('unexpected capture plan')
    files = plan['files_sha256']
    for key, digest in files.items():
        _verified_bytes(ROOT, key, digest)

    def read(key):
        return json.loads(_verified_bytes(ROOT, key, files[key]))

    static = read(plan['static_validation'])
    if (static['status'] != 'completed' or static['plan_sha256'] != files[plan['static_plan']]
            or static['updates'] != 0 or static['simulation_steps'] != 0 or static['static_branches'] != 8
            or not static['exact_development_tool_commands'] or static['test_episode_files_read'] is not False):
        raise ValueError('static full tool history not accepted')
    for name in ('static_config','commands'):
        source = plan[name]
        if static['artifacts_sha256'][Path(source).name] != files[source]:
            raise ValueError('frozen static artifacts changed')
    config = read(plan['training_config'])
    evidence = read(plan['commands'])
    case = evidence['case']
    identity = {k:case[k] for k in ('split','group_id','episode_id','history_id')}
    if (config != {**read(plan['static_config']), 'history_capture':identity}
            or config['seed'] != case['seed'] or case['split'] != 'train'
            or _initial_key(config) != case['physical_initial_sha256']):
        raise ValueError('worker would change the training config or group')
    case.update(source_config=plan['training_config'],source_config_sha256=files[plan['training_config']])
    ref = read(plan['reference_design'])
    template = read(plan['worker_budget_template'])
    limits = template['budget']
    tick, horizon, n = (template[k] for k in ('snapshot_tick','horizon','control_substeps'))
    if (case['group_id'] == ref['case']['group'] or list(evidence['branches']) != ref['branch_order']
            or len(evidence['prefix_requested']) != tick or len(evidence['prefix_executed']) != tick
            or evidence['snapshot_tick'] != tick or n*(tick+horizon*len(ref['branch_order'])) != 6784
            or limits['max_physics_substeps'] != 6784 or limits['max_worker_starts'] != 1
            or template['runtime'] != ref['runtime']):
        raise ValueError('physical scope or holdout identity changed')
    worker = {**template,'case':case,'directions':ref['directions'],'branch_order':ref['branch_order'],
              'development_holdout_group':ref['case']['group']}
    accepted = read(plan['backend_acceptance'])
    if not accepted['passed']:
        raise ValueError('backend acceptance missing')
    return worker, collector.build_config(config,worker), evidence, read(plan['layout']), {
        'backend':accepted['backend'],'source_sha256':files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('capture','worker'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--preflight-sha256',required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    if not out.relative_to(ROOT/'runs').parts:
        parser.error('select an output below runs')
    snapshot = collector.snapshot
    pending = args.phase == 'capture'

    def admission_snapshot(root):
        nonlocal pending
        sample = snapshot(root)
        if pending:
            pending = False
            deadline = time.monotonic()+45
            while sample['available_ram_bytes'] < 2*1024**3 and time.monotonic() < deadline:
                time.sleep(1)
                sample = snapshot(root)
        return sample

    with patch.multiple(collector,load_ready=readiness,DESIGN_SHA=args.preflight_sha256,__file__=__file__,
                        snapshot=admission_snapshot):
        run = collector.capture if args.phase == 'capture' else collector.worker
        run(out,args.preflight.resolve(),args.preflight_sha256)


if __name__ == '__main__':
    main()
