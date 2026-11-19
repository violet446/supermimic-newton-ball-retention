"""Locate the 61/97-pair endpoints' pitch and nominal error without training."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

from s3_fitability_common import BUDGET, ROOT, SCOPE, run_phase, write_json
import numpy as np

from audit_s3_ball_velocity_components import frame_series
from audit_s3_gradient_action_feedback import prior
from audit_s3_world_increment_drift import WINDOWS, branch_world_terms, require_close
from audit_s3_world_increment_generation import propagate, world_head
from audit_s3_world_increment_newton_reference import propagate_discrete_position, required_residual, tool_transition
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import KNOWN_NAMES, _ball_state, known_rollout
from ball_retention.pair_collection import prepare_collection
from ball_retention.timed_pair_data import prepare_timed_pairs
from ball_retention.world_increment_model import WorldIncrementMLP
from check_s3_actions import digest
from preflight_s3_world_increment_supervision import add, dot, mul, norm
import train_s3_broad_data97 as fit
from train_s3_world_increment import score_candidate


PHASE = 'world_increment_data97_audit'
PLAN = Path('configs/s3_world_increment_data97_audit_plan.json')
OUTPUT = 'runs/s3_world_increment_data97_audit_20261118'
PITCH = ('pitch_pulse', 'pitch_reversal')
BRANCHES = ('nominal_1', 'nominal_2', *(f'{name}_{side}' for name in PITCH for side in ('plus','minus')))
TRACKS = {
    'nominal': [.5,.5,0.,0.,0.,0.],
    'pitch_pulse': [0.,0.,.5,-.5,0.,0.],
    'pitch_reversal': [0.,0.,0.,0.,.5,-.5],
    'pitch_pulse_common': [-.5,-.5,.5,.5,0.,0.],
    'pitch_reversal_common': [-.5,-.5,0.,0.,.5,.5],
}
REPORT_WINDOWS = (*WINDOWS, (1,16), (17,32))
SHA61 = fit.PREVIOUS_SHA
SHA97 = '188a1e38b5c14a823eb917a34dc26a2a38075368378fcfc0e4fd1344d4f2a0c0'


def freeze():
    recorded = json.loads((ROOT/fit.FIT/'execution_plan.json').read_bytes())
    original = json.loads(_verified_bytes(ROOT, fit.PLAN, SHA97))
    if recorded != {'plan_sha256':SHA97,'plan':original}:
        raise ValueError('97-pair execution plan changed')
    files = {**original['files_sha256'], fit.PLAN.as_posix():SHA97}
    for run in (fit.PREVIOUS_RUN, fit.FIT):
        validation = run/'validation.json'
        files[validation.as_posix()] = digest(ROOT/validation)
        accepted = json.loads(_verified_bytes(ROOT, validation, files[validation.as_posix()]))
        for name in (fit.ENDPOINT, 'comparison.json', 'timed_rollouts.json', 'execution_plan.json', 'fit_preflight.json'):
            files[(run/name).as_posix()] = accepted['artifacts_sha256'][name]
    for relative in ('scripts/audit_s3_broad_data97.py', 'scripts/s3_broad_training_gradient.py',
                     'scripts/audit_s3_world_increment_newton_reference.py', 'scripts/audit_s3_gradient_action_feedback.py'):
        files[relative] = digest(ROOT/relative)
    with (ROOT/PLAN).open('x',encoding='utf-8') as stream:
        json.dump({'phase':PHASE,'output':OUTPUT,'budget':BUDGET,'scope':SCOPE,'seed':20261111,'updates':0,
                   'endpoint_updates':900,'held_branches':BRANCHES,'files_sha256':files},stream,indent=2,allow_nan=False)
        stream.write('\n')


def endpoint(plan, folder, count, plan_sha, fold, draws):
    path = folder/'validation.json'
    status = fit.read(plan,path)
    if ((status['status'],status['updates'],status['training_pairs'],status['plan_sha256'],status['sampler_sha256'])
            != ('completed',900,count,plan_sha,draws) or any(status[k] != v for k,v in SCOPE.items())):
        raise ValueError(f'unexpected frozen endpoint {folder}')
    file = fit.accepted_artifact(plan,path,status,fit.ENDPOINT)
    with np.load(BytesIO(_verified_bytes(ROOT,file,status['artifacts_sha256'][fit.ENDPOINT])),allow_pickle=False) as stored:
        if stored['updates'].item() != 900:
            raise ValueError('checkpoint misses its fixed endpoint')
        for name in ('state_mean','state_scale','known_mean','known_scale'):
            np.testing.assert_array_equal(stored[name],getattr(fold,name))
        np.testing.assert_array_equal(stored['known_names'],KNOWN_NAMES)
        model = WorldIncrementMLP.from_state_dict(stored)
    return model, fit.accepted(plan,folder,'validation.json','comparison.json'), fit.accepted(
        plan,folder,'validation.json','timed_rollouts.json')


def windows(values, reduce):
    return {f'{a}-{b}':reduce(values[a-1:b]) for a,b in REPORT_WINDOWS}


def rms(values):
    return float(np.sqrt(np.mean(np.sum(values**2,axis=-1))))


def diagnose(model, fold, held, anchor, archived, forecasts, guard):
    capture,cfg,fields,traces,_ = held
    scores,reproduced = score_candidate(model,held,fold,anchor,guard)
    if scores != archived or forecasts != reproduced:
        raise ValueError('frozen model scoring or trajectories changed')
    state = _parse_state(fields)
    dt = model.dt_s
    raw = known_rollout(state.time_s,state.tilt_rad,state.tilt_rate_rad_s,
                        np.asarray([traces[k]['actions'] for k in BRANCHES]),cfg,dt)
    features = model.features(raw,cfg,anchor,fold.known_mean,fold.known_scale)
    start = np.broadcast_to(_ball_state(state),(len(BRANCHES),9))
    predictions = model.denormalize(model.rollout(model.normalize(start),features))
    expected = [*([forecasts['nominal']]*2),*(forecasts['pairs'][k][s] for k in PITCH for s in ('plus','minus'))]
    closure = require_close(predictions, np.asarray(expected))
    truth = np.stack([traces[k]['raw'] for k in BRANCHES])
    truth_before,pred_before = (prior(start,path) for path in (truth,predictions))
    single = model.denormalize(model.step(model.normalize(truth_before),features))
    guard('compare_frozen_rollouts')
    gravity = np.array([0.,0.,-cfg['physics']['gravity_m_s2']*dt])
    world_errors, world_truth = [], []
    for k in BRANCHES:
        i = BRANCHES.index(k)
        frame = frame_series(state,traces[k],dt)
        _, errors, checked = branch_world_terms(frame,truth[i],single[i],predictions[i],gravity)
        closure = max(closure,checked)
        world_errors.append(errors)
        world_truth.append(np.diff(frame[3],axis=0)-gravity)

    # The archived tool poses are used above for evaluation only. The exact
    # source decomposition below uses the model's own deterministic transport.
    transfer,world,shift = tool_transition(features)
    target,checked = required_residual(truth_before,truth,transfer,world,shift)
    local_head = world_head(model,features,truth_before)
    rolled_head = world_head(model,features,pred_before)
    defect = truth_before[...,:3]+.5*dt*(truth_before[...,3:6]+truth[...,3:6])-truth[...,:3]
    sources = {'independent_increment':propagate(local_head-target,transfer,world,dt),
               'feedback_increment':propagate(rolled_head-local_head,transfer,world,dt),
               'position_rule':propagate_discrete_position(defect,transfer,dt)}
    closure = max(closure,checked,require_close(sum(sources.values()).sum(axis=0),predictions[...,:6]-truth[...,:6]))
    require_close((world@(local_head-target)[...,None])[...,0],single[...,3:6]-truth[...,3:6])
    return SimpleNamespace(truth=truth,predictions=predictions,sources=sources,single=single,
                           world={k:np.stack([v[k] for v in world_errors]) for k in ('local','free')},
                           physical_increments=np.stack(world_truth),
                           measured_pre=truth_before, known=raw, closure=closure)


def track_report(old, new):
    result = {}
    for name,weights in TRACKS.items():
        weighted = lambda x: np.einsum('b,btv->tv',weights,x)
        truth = weighted(old.truth)
        paths = [weighted(x.predictions) for x in (old,new)]
        e61,e97 = (path[-1,:3]-truth[-1,:3] for path in paths)
        change = e97-e61
        sum_of_norms = np.linalg.norm(e97)+np.linalg.norm(e61)
        if sum_of_norms == 0.:
            raise ValueError('expected a nonzero frozen H32 position error')
        norm_axis = (e97+e61)/sum_of_norms
        errors = {}
        contributions = {}
        for label, model in (('61',old),('97',new)):
            errors[label] = {
                'free_position_error_m':weighted(model.predictions-model.truth)[:,:3].tolist(),
                'true_final_position_m':truth[-1,:3].tolist(),
                'predicted_final_position_m':weighted(model.predictions)[-1,:3].tolist(),
                'one_step_position_rms_m':windows(weighted(model.single-model.truth)[:,:3],rms),
                'world_velocity_increment_error_m_s':{k:weighted(v).tolist() for k,v in model.world.items()},
                'world_velocity_increment_rms_m_s':{k:windows(weighted(v),rms) for k,v in model.world.items()},
                'h32_error_norm_m':float(np.linalg.norm(weighted(model.predictions-model.truth)[-1,:3])),
            }
        vectors = {}
        for key in old.sources:
            source = np.einsum('b,sbtv->stv',weights,new.sources[key]-old.sources[key])[:,-1,:3]
            vectors[key] = source
            contributions[key] = {'h32_vector_m_by_source_step':source.tolist(),
                                 'h32_error_growth_m_by_window':windows(source@norm_axis,lambda x:float(x.sum()))}
        closure = require_close(sum(vectors.values()).sum(axis=0), change)
        growth = float(np.linalg.norm(e97)-np.linalg.norm(e61))
        closure = max(closure, require_close(sum(vectors.values()).sum(axis=0)@norm_axis,growth))
        result[name] = {'norm_growth_projection_vector':norm_axis.tolist(),'h32_error_growth_m':growth,
                        'h32_difference_norm_m':float(np.linalg.norm(change)),
                        'endpoints':errors, 'h32_difference_sources':contributions,'closure_m':closure}
    return result


def manifest_metadata(timed):
    return (np.concatenate([x.amplitudes for x in timed]),
            [(entry['group_id'],entry['snapshot_tick']) for blob in timed for entry in blob.entries for _ in entry['nominal_branches']])


def coverage(old_aux, aux, old_fold, fold, held, dev, timed):
    """Count same-lead aligned increment labels separately from pre-state distance."""
    _,_,_,traces,_ = held
    states,_,p_targets,weights,_ = aux['half_difference']
    if len(weights) != len(fold.pairs.initial)-40:
        raise ValueError('auxiliary support rows changed')
    labels = .5*(p_targets[0]-p_targets[1])
    offsets = np.concatenate([x.branch_actions for x in timed],axis=1)
    offsets = .5*(offsets[0]-offsets[1])
    if offsets.shape[:2] != weights.shape:
        raise ValueError('coverage command rows changed')
    reference_pair_count = len(old_aux['half_difference'][3])
    cohorts = {'61':np.arange(reference_pair_count), 'new36':np.arange(reference_pair_count,len(states[0]))}
    report = {}
    for name in PITCH:
        index = [BRANCHES.index(f'{name}_{side}') for side in ('plus','minus')]
        dev_actions = np.stack([traces[BRANCHES[i]]['actions'] for i in index])
        delta = .5*(dev_actions[0]-dev_actions[1])
        exact = np.all(np.isclose(offsets,delta[None,:,:],rtol=0.,atol=1e-12),axis=(1,2))
        physical = .5*(dev.physical_increments[index[0]]-dev.physical_increments[index[1]])
        length = np.linalg.norm(physical,axis=-1)
        magnitude = np.linalg.norm(labels,axis=-1)
        denom = magnitude*length
        cosine = np.divide(np.einsum('ntv,tv->nt',labels,physical),denom,out=np.zeros_like(denom),where=denom>0)
        ratio = np.divide(magnitude,length,out=np.zeros_like(magnitude),where=length>0)
        eligible = (weights>0)&(magnitude>=1e-4)&(cosine>=.8)&(ratio>=.5)&(ratio<=2.)
        distance = np.sqrt(np.mean(((states-dev.measured_pre[index,None,:,:])/old_fold.state_scale)**2,axis=(0,3)))
        both = {}
        for label,rows in cohorts.items():
            matched = rows[exact[rows]]
            nearest = distance[rows].min(axis=0)
            both[label] = {'pair_count':len(rows),'exact_request_offset_pairs':len(matched),
                           'exact_request_offset_pairs_at_tick_168':int(np.sum(fold.pairs.snapshot_ticks[40+matched]==168)),
                           'nearest_prestates_standardized_by_lead':nearest.tolist(),
                           'strong_aligned_label_by_lead':np.any(eligible[rows],axis=0).tolist(),
                           'same_request_strong_aligned_label_by_lead':np.any(eligible[matched],axis=0).tolist()}
        report[name] = {'development_strong_by_lead':(length>=1e-4).tolist(),'training':both}
    return report


def run(plan,out,guard):
    from s3_broad_training_gradient import attribute

    if plan['updates'] != 0 or plan['endpoint_updates'] != 900 or tuple(plan['held_branches']) != BRANCHES:
        raise ValueError('unexpected audit scope')
    previous = fit.read(plan,fit.PREVIOUS)
    _,cfg,scales,fold49,held,draws = fit.reference_data(previous,guard)
    old_timed = [prepare_timed_pairs(ROOT,p,previous['files_sha256'][p]) for p in previous['timed_manifests']]
    old_fold, groups = fit.extend_fold(fold49,old_timed[2:])
    old_aux, labels = fit.supervision(previous,old_fold,previous['timed_manifests'],guard)
    old_pre = fit.accepted(plan,fit.PREVIOUS_RUN,'validation.json','data_preflight.json')
    np.testing.assert_array_equal(old_fold.pair_weights,old_pre['expanded_weights'])
    if groups != old_pre['group_weights'] or labels != old_pre['label_inventory'] or draws != fit.DRAW_SHA:
        raise ValueError('61-pair dataset, labels or S2 draws changed')
    _,manifest = fit.latest_capture(plan)
    latest = prepare_timed_pairs(ROOT,manifest,plan['files_sha256'][manifest.as_posix()])
    fold,masses = fit.widen(old_fold,latest)
    aux,inventory = fit.new_auxiliary(plan,old_aux,latest,fold,guard)
    pre97 = fit.accepted(plan,fit.FIT,'validation.json','fit_preflight.json')
    coef = old_pre['coefficients']
    if (inventory != pre97['added_label_inventory'] or masses != pre97['group_weights']
            or coef != pre97['auxiliary_coefficients']):
        raise ValueError('97-pair data or supervision differed from its accepted fit')
    old,old_score,old_forecast = endpoint(plan,fit.PREVIOUS_RUN,61,SHA61,old_fold,draws)
    new,new_score,new_forecast = endpoint(plan,fit.FIT,97,SHA97,fold,draws)
    model_cfg = held[1]
    if old.dt_s != new.dt_s or not np.array_equal(old.inc_center,new.inc_center):
        raise ValueError('endpoint dynamics clocks or coordinates differ')
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'),model_cfg['initial']['anchor_xy_tool_m'])[0]
    dev_old = diagnose(old,old_fold,held,anchor,old_score['held_timing'],old_forecast,guard)
    dev_new = diagnose(new,fold,held,anchor,new_score['full']['97_pairs'],new_forecast,guard)
    reports = track_report(dev_old,dev_new)
    rows = coverage(old_aux,aux,old_fold,fold,held,dev_old,[*old_timed,latest])
    write_json(out/'drift_audit.json',{'case':held[0]['case'],'tracks':reports,'pitch_training_coverage':rows,
                                     'max_trajectory_closure_si':max(dev_old.closure,dev_new.closure)})
    guard('trajectory_audit_complete',log=True)
    held_plan = fit.read(plan,previous['held_design'])
    legacy = prepare_collection(ROOT,ROOT/held_plan['base_collection']).train
    for key in ('episode_ids','group_ids','snapshot_ticks','axes','initial','plus_known','minus_known'):
        expected = np.concatenate([getattr(legacy.batch,key),*(getattr(x.batch,key) for x in old_timed)])
        np.testing.assert_array_equal(getattr(old_fold.pairs,key),expected)
    old_amplitude,old_nominal = manifest_metadata(old_timed)
    new_amplitude,new_nominal = manifest_metadata([latest])
    pair_amplitudes = np.r_[legacy.amplitudes,old_amplitude]
    leads = fit.lead_vectors(fit.read(plan,fit.read(plan,previous['coverage_design'])['reference_design']),32)
    datasets = {'61':(old_fold,old_aux,pair_amplitudes,old_nominal),
                '97':(fold,aux,np.r_[pair_amplitudes,new_amplitude],[*old_nominal,*new_nominal])}
    gradient_report = {}
    for name,model in (('61_endpoint',old),('97_endpoint',new)):
        parts = {}
        for label,(data,targets,amps,nominal_groups) in datasets.items():
            guard(f'gradient_{name}_{label}',log=True)
            summary,gradient = attribute(model,data,targets,coef,cfg,scales,leads,model_cfg,anchor,amps,nominal_groups,guard)
            parts[label] = (summary,gradient)
        a,b = (parts[k][1] for k in ('61','97'))
        difference = add(b,mul(a,-1.))
        gradient_report[name] = {'datasets':{k:v[0] for k,v in parts.items()},
            'pair_objective_gradient_cosine':dot(a,b)/(norm(a)*norm(b)),
            'pair_gradient_change_norm_over_old':norm(difference)/norm(a),
            'new_descent_within_old_objective_dot':-dot(a,b)}
    write_json(out/'training_gradient_audit.json',
               {'parameter_states_are_fixed':True,'s2_anchor_gradient_excluded':True,
                'scores_and_coverage_do_not_contribute_gradients':True,'results':gradient_report})
    return {'fixed_endpoints':2,'held_branch_count':len(BRANCHES),'frozen_scores_reproduced':True,
            'max_trajectory_closure_si':max(dev_old.closure,dev_new.closure),
            'gradient_endpoint_dataset_combinations':4,'training_pairs':[61,97]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','audit'))
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.phase == 'freeze':
        freeze()
    elif args.sha256:
        run_phase(ROOT/PLAN,args.sha256,PHASE,run)
    else:
        parser.error('audit requires the frozen plan SHA256')
