"""Complete the pending second case and score the frozen two-case comparison."""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

from collect_s3_gradient_action import (
    CASE_STEPS, IDENTITIES, LABELS, ROOT, _verified_bytes, collector, directory_bytes,
    load_case, load_plan, nominal_gate, score_case, small_action_response, storage_admission,
)


def load_continuation(path, expected):
    plan = json.loads(_verified_bytes(ROOT,path.relative_to(ROOT),expected))
    if (plan['schema']!='s3-gradient-action-second-case-v1' or plan['scope']!=collector.SCOPE
            or plan['episode']!=IDENTITIES[1][0]
            or collector.digest(Path(__file__))!=plan['runner_sha256']):
        raise ValueError('unexpected second-case continuation scope')
    first = Path(plan['partial_validation'])
    partial = json.loads(_verified_bytes(ROOT,first,plan['partial_validation_sha256']))
    if (partial['status'],partial['completed_case'],partial['pending_case'],
            partial['worker_starts'],partial['physics_steps'],partial['second_worker_started'],
            partial['training_updates'],partial['test_episode_files_read'],partial['control_accepted'])!=(
                'stopped_before_second_worker',IDENTITIES[0][0],plan['episode'],1,CASE_STEPS,False,0,False,False):
        raise ValueError('original first-case evidence does not permit this continuation')
    for key,digest in partial['artifacts_sha256'].items():
        _verified_bytes(ROOT,first.parent/key,digest)
    frozen = load_plan(ROOT/plan['frozen_plan'],partial['plan_sha256'])
    second = load_case(frozen,1)
    if plan['limits']!=second[1]['future_capture_limits']:
        raise ValueError('second-case physical or resource budget differs from the frozen one-worker limit')
    return plan,frozen,ROOT/first.parent,second


def verify_first(frozen, prior):
    _,spec,native,commands,recorded,prediction = load_case(frozen,0)
    first = prior/IDENTITIES[0][0]
    traces = {name:json.loads((first/'worker'/f'{name}.json').read_bytes()) for name in LABELS}
    value = score_case(spec,native,commands,traces,prediction)
    if (value!=json.loads((first/'score.json').read_bytes())
            or not nominal_gate([traces[name] for name in LABELS[:2]],recorded,spec['scoring']['summary_leads'])['passed']):
        raise ValueError('first-case score or nominal evidence changed')
    return value


def capture(out, path, expected):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    try:
        plan,frozen,prior,second = load_continuation(path,expected)
        cfg,spec,native,commands,recorded,prediction = second
        first_score = verify_first(frozen,prior)
        backend = collector.backend_preflight(spec)
        if (backend!=spec['static_backend'] or {'newton','warp'}.intersection(sys.modules)
                or not nominal_gate(recorded,recorded,spec['scoring']['summary_leads'])['passed']):
            raise ValueError('second-case backend or frozen nominal baseline did not reproduce')
        limits,budget = plan['limits'],cfg['budget']
        collector.write_json(out/'execution_plan.json',{'plan':plan,'plan_sha256':expected,'backend':backend,
                                                        'worker_config':cfg})
        sample = collector.snapshot(ROOT)
        hardware = collector.admission(sample,budget)
        storage = storage_admission(ROOT,out,budget)
        environment = directory_bytes(ROOT/'.venv')
        collector.write_json(out/'admission.json',{'hardware':hardware,'sample':sample,'storage':storage,
                                                   'environment_bytes':environment})
        if not hardware['passed'] or not storage['passed'] or environment>budget['max_environment_gib']*1024**3:
            raise RuntimeError('second-case resource admission failed')
        with patch.object(collector,'__file__',__file__):
            state = collector.monitor(out,path,expected,budget,limits,began)
        if (state['physics_steps'],state['nominal_prefix_substeps'],state['completed_branches'],
                state['nominal_gate_passed'])!=(CASE_STEPS,0,LABELS,True):
            raise ValueError('second worker finished outside its frozen scope')
        load_continuation(path,expected)
        traces = {name:json.loads((out/'worker'/f'{name}.json').read_bytes()) for name in LABELS}
        second_score = score_case(spec,native,commands,traces,prediction)
        count = sum(row['passed'] for row in (first_score,second_score))
        collector.write_json(out/'comparison.json',{'cases':{IDENTITIES[0][0]:first_score,IDENTITIES[1][0]:second_score},
            'cases_passed':count,'cases_scored':2,'prior_case_evidence':plan['partial_validation']})
        if time.monotonic()-began>=limits['max_wall_s'] or directory_bytes(out)>limits['max_new_output_mib']*1024**2:
            raise RuntimeError('second-case end-to-end time or output cap reached')
        collector.write_json(out/'validation.json',{'status':'completed','plan_sha256':expected,
            'case':spec['case'],'worker_starts':1,'physics_steps':CASE_STEPS,'nominal_prefix_substeps':0,
            'nominal_gate_passed':True,'cases_scored':2,'cases_passed':count,'combined_physics_steps':2*CASE_STEPS,
            'wall_s':time.monotonic()-began,**collector.SCOPE,
            'artifacts_sha256':{p.relative_to(out).as_posix():collector.digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    except BaseException as error:
        record = out/'worker/worker_result.json'
        state = json.loads(record.read_bytes()) if record.exists() else {}
        collector.write_json(out/'stopped.json',{'status':'stopped','reason':str(error),
            'physics_steps_at_last_event':state.get('physics_steps',0),'branches':state.get('completed_branches',[]),
            'wall_s':time.monotonic()-began,**collector.SCOPE})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('capture','worker'))
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    path,out=args.plan.resolve(),args.output.resolve()
    plan=json.loads(_verified_bytes(ROOT,path.relative_to(ROOT),args.sha256))
    target=(ROOT/plan['output']).resolve()
    if len(target.relative_to(ROOT/'runs').parts)!=1 or out!=(target/'worker' if args.phase=='worker' else target):
        parser.error('output differs from frozen second-case plan')
    if args.phase=='capture':
        capture(out,path,args.sha256)
    else:
        _,_,_,(cfg,spec,native,commands,recorded,_)=load_continuation(path,args.sha256)
        bundle = (plan,cfg,spec,native,commands,recorded,None,None)
        with (
            patch.object(collector,'load_run',new=lambda *_: bundle),
            patch.object(small_action_response,'nominal_gate',new=nominal_gate),
        ):
            collector.worker(out,path,args.sha256)
