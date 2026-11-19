"""Replay frozen gradient and reverse actions on two accepted development snapshots."""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import collect_s3_small_action_response as collector
import numpy as np

from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.model_data import _ball_state
from ball_retention.pair_collection import _initial_key
from ball_retention.resources import directory_bytes
from deterministic_archive_metrics import _full_branch
from response_contact_diagnosis import _allowed_pairs, _decode_trace
from runner import storage_admission
from saved_native_snapshot import decode_snapshot
import small_action_response


ROOT = collector.ROOT
IDENTITIES = [('validation_lateral','negative_y_basin',536873541),
              ('history_validation_yplus006_reverse','history_validation_yplus006',3392718248)]
LABELS = ['nominal_1','nominal_2','gradient','reverse']
TICK, HORIZON, SUBSTEPS = 168,32,16
CASE_STEPS = len(LABELS)*HORIZON*SUBSTEPS


def load_plan(path, expected):
    plan = json.loads(_verified_bytes(ROOT,path.relative_to(ROOT),expected))
    if plan['schema']!='s3-gradient-action-capture-v1' or plan['scope']!=collector.SCOPE:
        raise ValueError('unexpected gradient capture scope')
    for name,digest in plan['files_sha256'].items():
        _verified_bytes(ROOT,name,digest)
    return plan


def read(plan, path):
    return json.loads(_verified_bytes(ROOT,path,plan['files_sha256'][Path(path).as_posix()]))


def artifact(plan, validation, name):
    path = Path(validation).parent/name
    if plan['files_sha256'][path.as_posix()]!=read(plan,validation)['artifacts_sha256'][name]:
        raise ValueError(f'accepted artifact changed: {path}')
    return read(plan,path)


def load_case(plan, index):
    name,group,seed = IDENTITIES[index]
    source = plan['cases'][index]
    static_path = plan['static_validation']
    static = read(plan,static_path)
    if (static['status'],static['updates'],static['simulation_steps'],static['test_episode_files_read'],
            static['rank_order_cases'],static['plan_sha256'])!=(
                'completed',0,0,False,2,plan['files_sha256'][plan['static_plan']]):
        raise ValueError('frozen two-case static preflight not accepted')
    frozen = artifact(plan,static_path,'candidate_commands.json')
    choice = frozen['cases'][name]
    case,score,policy = choice['identity'],choice['score'],frozen['policy']
    if source['episode']!=name or (case['episode'],case['group'],case['seed'],case['tick'])!=(name,group,seed,TICK):
        raise ValueError('case order, split group, observation clock or seed changed')
    if (policy['pulse_steps'],policy['amplitude_l2'],policy['position_margin_floor_m'])!=(8,.05,1e-5):
        raise ValueError('frozen action or scoring policy changed')
    commands = choice['commands']
    if list(commands)!=LABELS or commands[LABELS[0]]!=commands[LABELS[1]] or not score['gradient_nominal_reverse_order_has_margin']:
        raise ValueError('frozen branch scope or model ranking changed')
    nominal,gradient,reverse = (np.asarray(commands[k]['requested']) for k in ('nominal_1','gradient','reverse'))
    delta = np.broadcast_to(score['gradient_delta_roll_pitch'],(8,2))
    np.testing.assert_allclose(np.linalg.norm(delta,axis=1),.05,rtol=0.,atol=1e-15)
    np.testing.assert_array_equal(gradient[:8],nominal[:8]+delta)
    np.testing.assert_array_equal(reverse[:8],nominal[:8]-delta)
    for label,row in commands.items():
        request = np.asarray(row['requested'])
        blocks = row['executed']
        if (request.shape!=(HORIZON,2) or not np.all(np.abs(request)<1.)
                or len(blocks)!=HORIZON or any(set(block)!= {'q','qd'} for block in blocks)
                or any(np.asarray(block[k]).shape!=(SUBSTEPS,6) or not np.isfinite(block[k]).all()
                       for block in blocks for k in ('q','qd'))):
            raise ValueError(f'invalid frozen action or command: {label}')
        np.testing.assert_array_equal(request[8:],nominal[8:])

    replay_source = source['nominal_validation']
    accepted = read(plan,replay_source)
    if (accepted['status']!='completed' or accepted['test_episode_files_read'] or accepted['control_accepted']
            or not accepted['nominal_gate_passed'] or accepted['case']!=case or accepted['nominal_prefix_substeps']!=0):
        raise ValueError('deterministic nominal baseline not accepted')
    cfg = artifact(plan,replay_source,'execution_plan.json')['worker_config']
    if cfg['seed']!=seed or _initial_key(cfg)!=choice['physical_initial_sha256']:
        raise ValueError('worker configuration changed the independent initial group')
    if index==0:
        static_restore = choice['restoration']
        if (not static_restore['physical_restore_validated'] or replay_source!=static_restore['nominal_validation']
                or source['native_validation']!=static_restore['source_validation']):
            raise ValueError('native case does not match prior restoration evidence')
        location = Path(source['native_validation']).parent
        native = {k:artifact(plan,source['native_validation'],Path(static_restore[k]).relative_to(location).as_posix())
                  for k in ('snapshot','layout')}
    else:
        if source['native_validation']!=replay_source or not accepted['public_readback_passed'] or accepted['physics_steps']!=1024:
            raise ValueError('second development case is missing its accepted restored snapshot')
        native = {k:artifact(plan,replay_source,f'worker/{k if k=="snapshot" else "native_layout"}.json')
                  for k in ('snapshot','layout')}
    state = decode_snapshot(native['snapshot'],native['layout'],cfg,TICK,SUBSTEPS)
    if state['controller']!=choice['restoration']['controller']:
        raise ValueError('restored controller differs from the static test')
    np.testing.assert_allclose(_ball_state(state['state'])[1],score['target_y_tool_m'],rtol=0.,atol=1e-12)
    recorded = [artifact(plan,replay_source,f'worker/{label}.json') for label in LABELS[:2]]
    for trace in recorded:
        _full_branch(trace,TICK)
        if commands['nominal_1']!={'requested':trace['actions'],'executed':trace['commands']}:
            raise ValueError('static nominal execution differs from same-configuration physical baseline')
    if recorded[0]!=recorded[1]:
        raise ValueError('recorded nominal pair is not identical')

    small = read(plan,plan['small_capture_plan'])
    prior_spec = artifact(plan,small['preflight_validation'],'replay_spec.json')
    if (native['layout']!=prior_spec['static_snapshot_checks']['native_layout']
            or prior_spec['horizon']!=HORIZON or prior_spec['substeps_per_control']!=SUBSTEPS):
        raise ValueError('snapshot layout or physical command clock differs from the deterministic replay')
    limits = {**small['limits'],'max_physics_substeps':CASE_STEPS*len(IDENTITIES),'max_worker_starts':len(IDENTITIES)}
    if (plan['limits']!=limits or
            (limits['sequential_environments'],limits['nominal_prefix_substeps'],limits['baseline_replays'])!=(1,0,2)):
        raise ValueError('frozen two-case resource envelope changed')
    cfg['budget'].update(max_physics_steps=CASE_STEPS,max_trace_steps=CASE_STEPS)
    spec = {**prior_spec,'case':case,'branch_order':LABELS,'same_sign_prefixes':[],
            'future_capture_limits':{**limits,'max_physics_substeps':CASE_STEPS,'max_worker_starts':1}}
    return cfg,spec,native,commands,recorded,score


def nominal_gate(new, archived, leads):
    result = collector.nominal_gate(new,archived,leads)
    result['passed'] = result['passed'] and result['all_nominal_json_equal']
    return result


def score_case(spec, native, commands, traces, prediction):
    y0 = prediction['target_y_tool_m']
    offsets = {}
    for name,trace in traces.items():
        _full_branch(trace,TICK)
        _decode_trace(trace,name,HORIZON,SUBSTEPS,TICK*SUBSTEPS,1/960,_allowed_pairs(native['layout']))
        if commands[name]!={'requested':trace['actions'],'executed':trace['commands']}:
            raise ValueError(f'actual transition requests or commands changed: {name}')
        offsets[name] = [point[1]-y0 for point in trace['states_raw']]
    actual = {key:abs(values[-1]) for key,values in offsets.items()}
    noise = .5*abs(actual['nominal_1']-actual['nominal_2'])
    nominal = .5*(actual['nominal_1']+actual['nominal_2'])
    margin = max(1e-5,5*noise)
    improvement = {'vs_nominal_m':nominal-actual['gradient'],'vs_reverse_m':actual['reverse']-actual['gradient']}
    model = {key:row['h32_abs_lateral_offset_m'] for key,row in prediction['predictions'].items()}
    actual_order = sorted(('nominal','gradient','reverse'),key={
        'nominal':nominal,'gradient':actual['gradient'],'reverse':actual['reverse']}.get)
    return {'case':spec['case'],'target_y_tool_m':y0,'actual_h32_abs_offset_m':actual,
            'model_h32_abs_offset_m':model,'actual_nominal_mean_m':nominal,'absolute_offset_half_spread_m':noise,
            'margin_m':margin,'improvement':improvement,'passed':all(value>margin for value in improvement.values()),
            'predicted_rank':['gradient','nominal','reverse'],'actual_rank':actual_order,
            'actual_signed_lateral_offset_by_lead_m':offsets}


def capture(path, expected, out):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    scores = {}
    try:
        plan = load_plan(path,expected)
        limits = plan['limits']
        if len(plan['cases'])!=len(IDENTITIES):
            raise ValueError('unexpected two-case inventory')

        def guard():
            if (time.monotonic()-began>=limits['max_wall_s']
                    or directory_bytes(out)>limits['max_new_output_mib']*1024**2):
                raise RuntimeError('end-to-end gradient-test wall or output cap reached')

        bundles = [load_case(plan,index) for index in range(len(IDENTITIES))]
        backend = collector.backend_preflight(bundles[0][1])
        if any(spec['static_backend']!=backend or not nominal_gate(recorded,recorded,spec['scoring']['summary_leads'])['passed']
               for _,spec,_,_,recorded,_ in bundles) or {'newton','warp'}.intersection(sys.modules):
            raise ValueError('deterministic backend or two-case static nominal comparison failed')
        collector.write_json(out/'execution_plan.json',{'plan':plan,'plan_sha256':expected,'backend':backend})
        environment = directory_bytes(ROOT/'.venv')
        for cfg,spec,native,commands,_,prediction in bundles:
            guard()
            name = spec['case']['episode']
            case_out = out/name
            case_out.mkdir(exist_ok=False)
            budget = cfg['budget']
            sample = collector.snapshot(ROOT)
            hardware = collector.admission(sample,budget)
            storage = storage_admission(ROOT,out,budget)
            collector.write_json(case_out/'admission.json',{'hardware':hardware,'sample':sample,'storage':storage,
                                 'environment_bytes':environment})
            if not hardware['passed'] or not storage['passed'] or environment>budget['max_environment_gib']*1024**3:
                raise RuntimeError(f'{name}: resource admission failed')
            # The existing monitor sees the sum of both case directories and shared cache.
            with patch.multiple(collector,__file__=__file__,directory_bytes=lambda _:directory_bytes(out),
                                storage_admission=lambda root,_,budget:storage_admission(root,out,budget)):
                state = collector.monitor(case_out,path,expected,budget,limits,began)
            if (state['physics_steps'],state['nominal_prefix_substeps'],state['completed_branches'],
                    state['nominal_gate_passed'])!=(CASE_STEPS,0,LABELS,True):
                raise ValueError(f'{name}: completed worker did not meet its frozen scope')
            traces = {label:json.loads((case_out/'worker'/f'{label}.json').read_bytes()) for label in LABELS}
            scores[name] = score_case(spec,native,commands,traces,prediction)
            collector.write_json(case_out/'score.json',scores[name])
            print(f'{name}: 2048 substeps; gradient test {"passed" if scores[name]["passed"] else "did not pass"}.',flush=True)
        for name,digest in plan['files_sha256'].items():
            _verified_bytes(ROOT,name,digest)
        count = sum(row['passed'] for row in scores.values())
        collector.write_json(out/'comparison.json',{'cases':scores,'cases_passed':count,'cases_scored':len(scores)})
        guard()
        collector.write_json(out/'validation.json',{'status':'completed','plan_sha256':expected,'worker_starts':len(scores),
            'physics_steps':CASE_STEPS*len(scores),'nominal_prefix_substeps':0,'cases_scored':len(scores),'cases_passed':count,
            'wall_s':time.monotonic()-began,**collector.SCOPE,
            'artifacts_sha256':{p.relative_to(out).as_posix():collector.digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    except BaseException as error:
        last = {}
        for record in out.glob('*/worker/worker_result.json'):
            try:
                state = json.loads(record.read_bytes())
            except (OSError,ValueError):
                continue
            last[record.parent.parent.name] = {k:state[k] for k in ('status','physics_steps','completed_branches')}
        collector.write_json(out/'stopped.json',{'status':'stopped','reason':str(error),'scored_cases':list(scores),
                             'last_worker_events':last,'wall_s':time.monotonic()-began,**collector.SCOPE})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('capture','worker'))
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    path,out = args.plan.resolve(),args.output.resolve()
    plan = load_plan(path,args.sha256)
    target = (ROOT/plan['output']).resolve()
    if len(target.relative_to(ROOT/'runs').parts)!=1:
        parser.error('run outside its frozen output path')
    if args.phase=='capture' and out==target:
        capture(path,args.sha256,out)
    elif args.phase=='worker' and out.parent.parent==target and out.name=='worker':
        index, = [i for i,(episode,_,_) in enumerate(IDENTITIES) if episode==out.parent.name]
        cfg,spec,native,commands,recorded,_ = load_case(plan,index)
        with patch.object(collector,'load_run',new=lambda *_:(plan,cfg,spec,native,commands,recorded,None,None)), \
                patch.object(small_action_response,'nominal_gate',new=nominal_gate):
            collector.worker(out,path,args.sha256)
    else:
        parser.error('output does not match the selected phase and its frozen case')
