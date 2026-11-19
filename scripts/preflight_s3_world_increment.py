"""Verify the frozen world-frame increment candidate without optimizer updates."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np
from scipy.spatial.transform import Rotation

from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import KNOWN_NAMES, _ball_state, known_rollout
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP
from ball_retention.world_increment_model import PACKED, WorldIncrementMLP, _endpoints, world_action_jvp
from preflight_s3_velocity_response_leads import prepare_comparison
from train_s3_advance_coverage import prepare_candidate
from train_s3_local_velocity_increment import bound, incumbent
from train_s3_position_integral import action_sequences, read_held, report, score_held


PHASE = 'world_increment_preflight'
CHECKS = {'endpoint_atol_si': 2e-6, 'identity_atol_si': 2e-12,
          'probe_param_sigma': .03, 'scalar_fd_step': 1e-6, 'scalar_fd_atol': 2e-6,
          'max_initial_standardized_value': 50., 'horizons': [1, 8, 32],
          'action_fd_step_fractions': [1., .5, .25]}


def reference_data(plan, guard):
    protocol = json.loads(bound(plan, plan['comparison_protocol']))
    coverage = json.loads(bound(plan, plan['coverage_design']))
    design = json.loads(bound(plan, coverage['reference_design']))
    parent, cfg, scales, fold46, legacy, anchors, _, draws, _ = prepare_comparison(design, guard)
    fold, _ = prepare_candidate(coverage, fold46, guard)
    if (len(fold.pairs.initial) != protocol['training_pairs'] or draws != protocol['s2_draws_sha256']
            or set(fold.pairs.group_ids) & set(legacy.validation.batch.group_ids)):
        raise ValueError('49-pair or S2 sampler identity changed')
    baseline, reference = incumbent({**plan, 'anchor_draws_sha256':draws}, fold)
    held = read_held(parent)
    timed, forecasts = score_held(baseline, held, fold, lambda: guard('reference'))
    history = json.loads(bound(plan, cfg['history_plan']))
    histories = {row['episode_id']:row['history_id'] for row in history['episodes']}
    legacy_result = report(baseline,fold.known_mean,fold.known_scale,legacy,anchors,histories)
    legacy_result['pairs'] = [row for row in legacy_result['pairs'] if row['split']=='validation']
    saved_rollouts = json.loads(bound(plan, Path(protocol['reference_checkpoint']).with_name('timed_rollouts.json')))
    if (timed != reference['held_timing'] or legacy_result != reference['legacy_validation']
            or forecasts != {key:saved_rollouts[key] for key in forecasts}):
        raise ValueError('49-pair reference scores or held trajectories did not reproduce')
    return protocol, cfg, scales, fold, held, draws


def predict(model, state, actions, cfg, anchor, mean, scale):
    known = known_rollout(state.time_s,state.tilt_rad,state.tilt_rate_rad_s,actions,cfg,model.dt_s)
    packed = model.features(known,cfg,anchor,mean,scale)
    initial = model.normalize(_ball_state(state))
    return model.denormalize(model.rollout(initial[None],packed[None])[0])


def rollout_identity(model, state, actions, cfg, anchor, mean, scale):
    initial = _ball_state(state)
    raw = known_rollout(state.time_s,state.tilt_rad,state.tilt_rate_rad_s,actions,cfg,model.dt_s)
    packed = model.features(raw,cfg,anchor,mean,scale)
    rollout = np.vstack((initial, predict(model,state,actions,cfg,anchor,mean,scale)))
    first, second = rollout[:-1], rollout[1:]
    (r0,w0,v0), _ = _endpoints(raw,cfg,model.dt_s,anchor,-1)
    (r1,w1,v1), _ = _endpoints(raw,cfg,model.dt_s,anchor,1)
    g = np.array([0.,0.,-cfg['physics']['gravity_m_s2']*model.dt_s])
    this = model.normalize(first)
    untied = TrapezoidPositionResidualMLP(len(KNOWN_NAMES),model.state_mean,model.state_scale,model.dt_s,0,
                                        hidden=model.hidden,residual_gain=model.residual_gain)
    for key in untied.params:
        untied.params[key][...] = model.params[key]
    raw_neural = untied.denormalize(untied.step(this,packed[:, :len(KNOWN_NAMES)]))
    beginning = v0+(r0@(first[:,3:6]+np.cross(w0,first[:,:3]))[...,None])[...,0]
    finish = v1+(r1@(second[:,3:6]+np.cross(w1,second[:,:3]))[...,None])[...,0]
    diff = {'position':second[:,:3]-first[:,:3]-.5*model.dt_s*(first[:,3:6]+second[:,3:6]),
            'world_velocity':finish-beginning-g-model.inc_center-(raw_neural[:,3:6]-first[:,3:6]),
            'spin':second[:,6:]-raw_neural[:,6:]}
    for value in diff.values():
        np.testing.assert_allclose(value,0.,atol=CHECKS['identity_atol_si'],rtol=0.)
    return {k:float(np.max(abs(v))) for k,v in diff.items()}


def endpoint_check(state, states, actions, cfg, anchor, dt):
    raw = known_rollout(state.time_s,state.tilt_rad,state.tilt_rate_rad_s,actions,cfg,dt)
    expected = [state,*states]
    errors = {}
    for label, rows, sign in (('start',expected[:-1],-1),('end',expected[1:],1)):
        tools = [s.tool for s in rows]
        r = Rotation.from_quat([t.quaternion_xyzw for t in tools]).as_matrix()
        w = np.einsum('nji,nj->ni',r,[t.angular_velocity for t in tools])
        v = np.asarray([t.linear_velocity_origin for t in tools])
        actual,_ = _endpoints(raw,cfg,dt,anchor,sign)
        errors[label] = {key:float(np.max(abs(a-b))) for key,a,b in zip(('rotation','local_omega','origin_velocity'),actual,(r,w,v))}
        if max(errors[label].values()) > CHECKS['endpoint_atol_si']:
            raise ValueError(f'analytic endpoints disagree with measured tool state: {errors[label]}')
        ball = np.asarray([s.ball.linear_velocity_origin for s in rows])
        x = np.stack([_ball_state(s) for s in rows])
        rebuilt = v+np.einsum('nij,nj->ni',r,x[:,3:6]+np.cross(w,x[:,:3]))
        np.testing.assert_allclose(rebuilt,ball,atol=CHECKS['identity_atol_si'],rtol=0.)
    return errors


def fd_vjp(model, initial, known, rng):
    rng_dir = lambda x: rng.standard_normal(x.shape)/np.sqrt(x.size)
    probe = rng_dir(np.broadcast_to(initial[:,None],(*known.shape[:2],9)))
    ds, dk, params = model.rollout_vjp(initial,known,probe)
    dx, df = rng_dir(initial), rng_dir(known)
    dp = {key:rng_dir(value) for key,value in model.params.items()}
    eps = CHECKS['scalar_fd_step']
    f = lambda x,y:float(np.sum(model.rollout(x,y)*probe))
    ds_fd = (f(initial+eps*dx,known)-f(initial-eps*dx,known))/(2*eps)
    dk_fd = (f(initial,known+eps*df)-f(initial,known-eps*df))/(2*eps)
    backups = {key:value.copy() for key,value in model.params.items()}
    values = []
    try:
        for sign in (1.,-1.):
            for key,value in model.params.items():
                value[...] = backups[key]+sign*eps*dp[key]
            values.append(f(initial,known))
    finally:
        for key,value in model.params.items():
            value[...] = backups[key]
    expected = [np.sum(ds*dx),np.sum(dk*df),sum(np.sum(params[k]*dp[k]) for k in dp)]
    observed = [ds_fd,dk_fd,(values[0]-values[1])/(2.*eps)]
    np.testing.assert_allclose(expected,observed,atol=CHECKS['scalar_fd_atol'],rtol=1e-5)
    return {key:float(abs(a-b)) for key,a,b in zip(('initial_state','packed_features','parameters'),expected,observed)}


def run(plan, out, guard):
    if plan['checks'] != CHECKS or plan['updates'] != 0:
        raise ValueError('unexpected preflight limits')
    protocol, train_cfg, scales, fold, held, draws = reference_data(plan,guard)
    capture, cfg, fields, traces, _ = held
    state = _parse_state(fields)
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'),cfg['initial']['anchor_xy_tool_m'])[0]
    mean, scale = fold.known_mean, fold.known_scale
    model = WorldIncrementMLP(PACKED,fold.state_mean,fold.state_scale,1/cfg['physics']['control_hz'],plan['seed'],
                              inc_center=protocol['world_increment_center_m_s'],hidden=train_cfg['network'][1],
                              residual_gain=np.asarray(scales['residual_gain_si'])/fold.state_scale)
    reference = TrapezoidPositionResidualMLP(len(KNOWN_NAMES),model.state_mean,model.state_scale,model.dt_s,
                                           plan['seed'],hidden=model.hidden,residual_gain=model.residual_gain)
    for key,value in model.params.items():
        np.testing.assert_array_equal(value,reference.params[key])
    directions, actions = action_sequences(capture,np.asarray(traces['nominal_1']['actions']))
    nominal = actions['nominal_1']
    zero = predict(model,state,nominal,cfg,anchor,mean,scale)
    peak = float(np.max(abs(model.normalize(zero))))
    if peak > CHECKS['max_initial_standardized_value']:
        raise ValueError(f'zero-output H32 preflight exceeded bound: {peak}')
    rng = np.random.default_rng(plan['seed'])
    for key in ('W3','b3'):
        model.params[key][...] = rng.normal(0.,CHECKS['probe_param_sigma'],model.params[key].shape)
    reloaded = WorldIncrementMLP.from_state_dict(model.state_dict())
    np.testing.assert_array_equal(predict(model,state,nominal,cfg,anchor,mean,scale),
                                  predict(reloaded,state,nominal,cfg,anchor,mean,scale))
    endpoints, identity, derivatives = {}, {}, {}
    for label, sequence in actions.items():
        endpoints[label] = endpoint_check(state,traces[label]['states'],sequence,cfg,anchor,model.dt_s)
        identity[label] = rollout_identity(model,state,sequence,cfg,anchor,mean,scale)
        guard(label)
    fd = capture['model']
    for label,direction in directions.items():
        y,jvp = world_action_jvp(model,_ball_state(state),state.time_s,state.tilt_rad,state.tilt_rate_rad_s,
                                 nominal,direction,cfg,anchor,mean,scale)
        np.testing.assert_array_equal(y,predict(model,state,nominal,cfg,anchor,mean,scale))
        derivatives[label] = {}
        coarse = None
        for factor in CHECKS['action_fd_step_fractions']:
            eps = fd['fd_step']*factor
            a,b = (predict(model,state,nominal+s*eps*direction,cfg,anchor,mean,scale) for s in (1.,-1.))
            central = (a-b)/(2.*eps)
            error = abs(jvp-central)
            derivatives[label][str(eps)] = np.max(error,axis=0).tolist()
            if factor == 1.:
                mismatch = error > fd['fd_atol_si']+fd['fd_rtol']*abs(central)
                coarse = error
            else:
                np.testing.assert_allclose(jvp,central,atol=fd['fd_atol_si'],rtol=fd['fd_rtol'])
                if np.any(error[mismatch] > 1.5*factor**2*coarse[mismatch]):
                    raise ValueError(f'coarse derivative discrepancy does not converge under step refinement: {label}')
    initial = model.normalize(np.asarray(fold.pairs.initial[-2:]))
    features = model.features(fold.pairs.plus_known[-2:],cfg,anchor,mean,scale)
    scalar = {h:fd_vjp(model,initial,features[:,:h],rng) for h in CHECKS['horizons']}
    full = model.rollout(initial,features)
    for h in CHECKS['horizons']:
        np.testing.assert_array_equal(full[:,:h],model.rollout(initial,features[:,:h]))
    write_json(out/'preflight.json', {'reference_exact':True,'s2_draws_sha256':draws,
               'endpoint_errors':endpoints,'rollout_identities':identity,'action_fd_max_abs_si':derivatives,
               'vjp_fd_abs_errors': {str(k):v for k,v in scalar.items()},'zero_output_nominal_peak_standardized':peak,
               'synthetic_probe_params':True,'horizon':32,'checkpoint_roundtrip_exact':True})
    return {'reference_reproduced':True,'completed_preflight':True,'training_pairs':49}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
