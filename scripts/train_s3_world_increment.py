"""Fit and evaluate one matched world-frame increment candidate."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from s3_fitability_common import ROOT, forecast, run_phase, write_json

import numpy as np

from audit_s3_ball_velocity_components import branch_terms, frame_series, half, rms
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import KNOWN_NAMES, _ball_state, known_rollout, prepare_dataset
from ball_retention.pair_collection import prepare_collection
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP
from ball_retention.surface_diagnostic import SurfaceDiagnostic
from ball_retention.velocity_response_loss import paired_velocity_response_loss
from ball_retention.world_increment_model import PACKED, WorldIncrementMLP, world_action_jvp
from ball_retention.world_model import Adam
from deterministic_archive_metrics import _full_branch
from deterministic_response_metrics import response_metrics
from preflight_s3_velocity_response_leads import dormant_check, lead_vectors
from preflight_s3_world_increment import predict, reference_data
from train_s3_local_velocity_increment import bound, incumbent
from train_s3_position_integral import CONTROL, REFERENCE, LEADS, action_sequences, primary, regression, report
from train_s3_timing_lead_weights import checks_for


PHASE = 'world_increment_fit'
WINDOW = slice(7,24)


def packed(model, known, cfg, anchor, fold):
    if isinstance(model, WorldIncrementMLP):
        return model.features(known,cfg,anchor,fold.known_mean,fold.known_scale)
    return (known-fold.known_mean)/fold.known_scale


def objectives(model, fold, cfg, scales, leads, model_cfg, anchor):
    batch = fold.pairs

    def features(x):
        result = packed(model,x,model_cfg,anchor,fold)
        np.testing.assert_array_equal(result[...,:len(KNOWN_NAMES)],(x-fold.known_mean)/fold.known_scale)
        return result

    initial = model.normalize(batch.initial)
    anchors = {h:(model.normalize(w.initial),features(w.known),model.normalize(w.target))
               for h,w in fold.anchors.items()}
    known = features(np.stack((batch.plus_known,batch.minus_known)))
    target = np.stack((batch.plus_target,batch.minus_target))
    mask = np.stack((batch.plus_mask,batch.minus_mask))
    components, units = np.asarray(cfg['components']), np.asarray(scales['state_error_scale_si'])
    response_units = np.asarray(cfg['response_scales_si'])
    base, velocity = leads
    weights = cfg['pairs']

    def evaluate(h, take):
        x, feat, y = anchors[h]
        s2, grad = model.loss_and_grad(x[take],feat[take],y[take],state_weights=components,state_error_scale_si=units)
        pair, pair_grad = paired_velocity_response_loss(model,initial,known,target,mask,batch.response_mask,
            batch.noise_mask,batch.response_floor,base,components,response_units,
            weights['absolute_weight'],weights['response_weight'],velocity_lead_weights=velocity,
            sample_weights=fold.pair_weights,absolute_scale_si=units)
        return s2,pair,{key:grad[key]+pair_grad[key] for key in grad}

    return evaluate, initial, known


def score_candidate(model, held, fold, anchor, guard):
    capture,cfg,fields,traces,_ = held
    state = _parse_state(fields)
    nominal = np.asarray(traces['nominal_1']['actions'])
    directions, requests = action_sequences(capture,nominal)
    run = lambda action:predict(model,state,action,cfg,anchor,fold.known_mean,fold.known_scale)
    baselines = [traces[k]['raw'] for k in ('nominal_1','nominal_2')]
    gates = capture['gates']
    floors = tuple(gates[key] for key in ('position_floor_m','velocity_floor_m_s','spin_floor_rad_s'))
    thresholds = {key:gates[key] for key in ('noise_multiplier','strong_multiplier','min_cosine','max_relative_error')}
    fd = capture['model']
    prediction = run(nominal)
    scores, forecasts = {}, {}
    for label,direction in directions.items():
        y,jvp = world_action_jvp(model,_ball_state(state),state.time_s,state.tilt_rad,state.tilt_rate_rad_s,
                                nominal,direction,cfg,anchor,fold.known_mean,fold.known_scale)
        np.testing.assert_array_equal(y,prediction)
        eps = fd['fd_step']*.25
        slope = (run(nominal+eps*direction)-run(nominal-eps*direction))/(2*eps)
        np.testing.assert_allclose(jvp,slope,atol=fd['fd_atol_si'],rtol=fd['fd_rtol'])
        keys = [f'{label}_{side}' for side in ('plus','minus')]
        for k in keys:
            np.testing.assert_array_equal(requests[k],traces[k]['actions'])
        plus,minus = (run(requests[k]) for k in keys)
        scores[label] = response_metrics(baselines,*(traces[k]['raw'] for k in keys),prediction,plus,minus,jvp,
                                         capture['amplitude'],leads=LEADS,floors=floors,**thresholds)
        forecasts[label] = {'plus':plus.tolist(),'minus':minus.tolist(),'jvp':jvp.tolist(),
                            'max_jvp_fd_diff':float(np.max(abs(jvp-slope)))}
        guard('candidate_response')
    return scores, {'nominal':prediction.tolist(),'pairs':forecasts}


def score_exact(models, state, traces, names, cfg, mesh, anchor, fold, guard):
    dt = models['reference'].dt_s
    start = _ball_state(state)
    gravity = np.array([0.,0.,-cfg['physics']['gravity_m_s2']*dt])
    surface = SurfaceDiagnostic(mesh,cfg['ball']['radius_m'],cfg['initial']['tool_rotation_world'])
    rows = {}
    for label in names:
        branches = [traces[f'{label}_{side}'] for side in ('plus','minus')]
        truth = np.stack([v['raw'] for v in branches])
        frames = [frame_series(state,v,dt) for v in branches]
        targets = [np.vstack((start,x)) for x in truth]
        rotations = [x[0] for x in frames]
        n = np.stack([r[:-1].apply([surface.measure(p)['normal_tool'] for p in target[:-1,:3]])
                      for r,target in zip(rotations,targets)])
        target_net = np.stack([r[1:].apply(branch_terms(frame,target,gravity)[0]['non_gravity'])
                               for r,frame,target in zip(rotations,frames,targets)])
        requests = np.stack([np.asarray(v['actions']) for v in branches])
        inputs = known_rollout(state.time_s,state.tilt_rad,state.tilt_rate_rad_s,requests,cfg,dt)
        initial = np.broadcast_to(start,(2,9))
        rows[label] = {}
        for key,model in models.items():
            features = packed(model,inputs,cfg,anchor,fold)
            rows[label][key] = {}
            for mode in ('local','free'):
                pred = forecast(model,initial,features,truth,local=mode=='local')
                terms = [branch_terms(frame,target,gravity,endpoint=x,independent=mode=='local')[0]['non_gravity']
                         for frame,target,x in zip(frames,targets,pred)]
                net = np.stack([r[1:].apply(t) for r,t in zip(rotations,terms)])
                error = net-target_net
                tangent = error-n*np.sum(error*n,axis=-1,keepdims=True)
                vel = half(pred-truth)[:,3:6]
                rows[label][key][mode] = {'net_world_half_error_rms_m_s':rms(half(error)[WINDOW]),
                    'tangent_world_half_error_rms_m_s':rms(half(tangent)[WINDOW]),
                    'velocity_half_error_rms_m_s':rms(vel[WINDOW]),
                    'h32_velocity_half_error_m_s':float(np.linalg.norm(vel[-1]))}
        guard('exact_labels')
    return rows


class LegacyEvaluation:
    """Adapt the protection scorer's standardized 20-feature input."""

    def __init__(self, model, cfg, anchor, fold):
        self.model,self.cfg,self.anchor,self.fold = model,cfg,anchor,fold
        self.normalize,self.denormalize = model.normalize,model.denormalize

    def rollout(self, initial, features):
        known = features*self.fold.known_scale+self.fold.known_mean
        return self.model.rollout(initial,packed(self.model,known,self.cfg,self.anchor,self.fold))


def run(plan,out,guard):
    def read(p):
        return json.loads(bound(plan,p))

    preflight = read(plan['accepted_preflight'])
    if preflight['status'] != 'completed' or not preflight['completed_preflight'] or preflight['updates'] != 0:
        raise ValueError('candidate preflight is not accepted')
    protocol,cfg,scales,fold,held,draws = reference_data(plan,guard)
    if plan['updates'] != protocol['fixed_updates'] or plan['budget'] != protocol['budget']:
        raise ValueError('unexpected match scope')
    dense = read(read(plan['coverage_design'])['reference_design'])
    parent = read(plan['held_design'])
    model_cfg = held[1]
    mesh = load_tool(ROOT/'assets/shovel.STL')
    anchor = support_point(mesh,model_cfg['initial']['anchor_xy_tool_m'])[0]
    baseline,ref = incumbent({**plan,'anchor_draws_sha256':draws},fold)
    model = WorldIncrementMLP(PACKED,fold.state_mean,fold.state_scale,baseline.dt_s,plan['seed'],
                              inc_center=protocol['world_increment_center_m_s'],hidden=baseline.hidden,
                              residual_gain=baseline.residual_gain)
    fresh = TrapezoidPositionResidualMLP(len(KNOWN_NAMES),fold.state_mean,fold.state_scale,baseline.dt_s,
                                        plan['seed'],hidden=baseline.hidden,residual_gain=baseline.residual_gain)
    for key in model.params:
        np.testing.assert_array_equal(model.params[key],fresh.params[key])
    leads = lead_vectors(dense,fold.pairs.plus_target.shape[1])
    evaluate,initial,features = objectives(model,fold,cfg,scales,leads,model_cfg,anchor)
    horizons = cfg['anchors']['horizons']
    size = cfg['anchors']['batch_size']
    rng = np.random.default_rng(plan['seed'])
    sampler = []
    for i in range(plan['updates']):
        h = horizons[i%len(horizons)]
        sampler.append((h,rng.integers(len(fold.anchors[h].initial),size=size)))
    draw_digest = sha256(b''.join(np.r_[h,take].astype('<i8').tobytes() for h,take in sampler)).hexdigest()
    if draws != draw_digest:
        raise ValueError('matched sampler changed')
    h,take = sampler[0]
    s2,pair,grad = evaluate(h,take)
    if not all(np.isfinite(g).all() for g in grad.values()):
        raise FloatingPointError('nonfinite initial loss gradient')
    eps = 1e-6
    prior = model.params['b3'].copy()
    direction = np.zeros(9)
    direction[3:6] = [.3,-.2,.1]
    losses = []
    try:
        for sign in (1.,-1.):
            model.params['b3'][...] = prior+sign*eps*direction
            a,b,_ = evaluate(h,take)
            losses.append(a+b['loss'])
    finally:
        model.params['b3'][...] = prior
    fd = (losses[0]-losses[1])/(2*eps)
    np.testing.assert_allclose(fd,np.dot(grad['b3'],direction),rtol=2e-5,atol=2e-6)
    score_candidate(model,held,fold,anchor,guard)
    legacy = prepare_collection(ROOT,ROOT/parent['base_collection'])
    s2_data = prepare_dataset((ROOT/parent['s2_manifest']).parent,horizons=tuple(horizons),verify_test=False)
    histories = {row['episode_id']:row['history_id'] for row in read(cfg['history_plan'])['episodes']}
    original = read(plan['original_scores'])
    old46 = read(plan['reference_46_scores'])
    gates = (('40_pairs',original['legacy_development'][CONTROL]),
             ('46_pairs',old46['legacy_validation']),('49_pairs',ref['legacy_validation']))
    initial_legacy = report(LegacyEvaluation(model,model_cfg,anchor,fold),fold.known_mean,fold.known_scale,legacy,s2_data,histories)
    if any(len(regression(value,initial_legacy)['checks'])!=22 for _,value in gates):
        raise ValueError('protection scorer did not produce 22 checks')
    write_json(out/'fit_preflight.json',{'sampler_sha256':draw_digest,'training_pairs':len(fold.pairs.initial),
        'initial_s2_loss':s2,'initial_pair_loss':pair,'combined_loss_direction_fd_error':abs(fd-np.dot(grad['b3'],direction)),
        'same_targets_and_weights':True,'initial_candidate_scoring_verified':True})
    adam = cfg['optimizer']
    if adam['name']!='Adam':
        raise ValueError('unexpected optimizer')
    optimizer = Adam(model.params,learning_rate=adam['learning_rate'],max_grad_norm=adam['max_grad_norm'])
    guard('fit',0,log=True)
    with (out/'training.jsonl').open('x',encoding='utf-8') as stream:
        for update,(h,take) in enumerate(sampler,1):
            s2,pair,grad = evaluate(h,take)
            norm = optimizer.step(grad)
            if not all(np.isfinite(x).all() for x in model.params.values()):
                raise FloatingPointError(f'nonfinite update {update}')
            log = update%30 == 0
            guard('fit',update,log=log)
            if log:
                print(json.dumps({'update':update,'anchor_loss':s2,'pair':pair,'gradient_norm':norm},allow_nan=False),
                      file=stream,flush=True)
    dormant_check(model.params)
    ck = out/'world_increment_900.npz'
    np.savez_compressed(ck,**model.state_dict(),known_names=KNOWN_NAMES,known_mean=fold.known_mean,known_scale=fold.known_scale,
                        updates=optimizer.iteration)
    with np.load(ck,allow_pickle=False) as saved:
        fitted = WorldIncrementMLP.from_state_dict(saved)
    np.testing.assert_array_equal(fitted.rollout(initial,features[0]),model.rollout(initial,features[0]))
    timed,forecasts = score_candidate(fitted,held,fold,anchor,guard)
    after = report(LegacyEvaluation(fitted,model_cfg,anchor,fold),fold.known_mean,fold.known_scale,legacy,s2_data,histories)
    after['pairs'] = [x for x in after['pairs'] if x['split']=='validation']
    protections = {k:regression(value,after) for k,value in gates}
    positions = primary(timed)
    nominal = positions['pitch_pulse']['absolute_model_nominal_error_norm']
    checks = checks_for(positions,nominal/original['nominal_h32_position_m'][REFERENCE],{CONTROL:protections['40_pairs']})
    checks.update({f'protections_vs_{key}':value['passed'] for key,value in protections.items() if key!='40_pairs'})
    models = {'reference':baseline,'world_increment':fitted}
    capture,cfg_dev,state,traces,_ = held
    exact = {'development':score_exact(models,_parse_state(state),traces,capture['directions'],cfg_dev,mesh,anchor,fold,guard)}
    for manifest in plan['timed_manifests']:
        for entry in read(manifest)['entries']:
            if entry['split'] != 'train':
                raise ValueError('unexpected exact-label split')
            root = Path(manifest).parent
            from_ref = lambda e: read(root/e['path'])
            snapshot = _parse_state(from_ref(entry['native_snapshot'])['state'])
            rows = {b['label']:_full_branch(from_ref(b),entry['snapshot_tick'])
                    for pair in entry['pairs'] for b in pair['branches']}
            conf = read(entry['source_config'])
            exact[entry['episode_id']] = score_exact(models,snapshot,rows,[p['direction_id'] for p in entry['pairs']],
                                                     conf,mesh,anchor,fold,guard)
    write_json(out/'comparison.json',{'held_timing':timed,'held_h32_position':positions,
        'nominal_h32_position_m':nominal,'legacy_protections':protections,'legacy_validation':after,
        'gates':checks,'exact_label_comparison':exact})
    write_json(out/'timed_rollouts.json',forecasts)
    return {'training_pairs':49,'sampler_sha256':draw_digest,'replacement_gate_passed':all(checks.values()),
            'main_model_replaced':False,'checkpoint_selection':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
