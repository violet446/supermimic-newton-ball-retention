"""Trace nominal drift and the dual response through two archived world models."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, forecast, run_phase, write_json

import numpy as np

from audit_s3_ball_velocity_components import frame_series, half, rms
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, known_rollout
from ball_retention.position_integral_model import load_position_model
from ball_retention.world_increment_model import WorldIncrementMLP
from diagnose_s3_advance_feedback import diagnose_pair
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'world_increment_drift'
BASE = Path('runs/s3_fitability_p3_20261116')
CANDIDATE = Path('runs/s3_world_increment_fit_20261117')
WINDOWS = ((1,8),(9,16),(17,24),(25,32))
ATOL = 2e-12


def require_close(a,b):
    np.testing.assert_allclose(a,b,atol=ATOL,rtol=0.)
    return float(np.max(abs(np.asarray(a)-b)))


def integral(v,dt,first=None):
    first = np.zeros(v.shape[1:]) if first is None else first
    return .5*dt*(np.concatenate((first[None],v[:-1]),axis=0)+v)


def describe(increments, axis):
    total = increments.sum(axis=0)
    return {'vector_m':total.tolist(),'norm_m':float(np.linalg.norm(total)),
            'projection_m':float(total@axis),
            'window_projection_m':{f'{a}-{b}':float(increments[a-1:b].sum(axis=0)@axis) for a,b in WINDOWS}}


def branch_world_terms(frame, actual, local, free, g):
    rotation, shift, omega, world = frame
    r = rotation[1:]
    w_local = omega[1:]
    def recover(values):
        return shift[1:]+r.apply(values[:,3:6]+np.cross(w_local,values[:,:3]))

    local_world, free_world = (recover(values) for values in (local,free))
    target_increment = np.diff(world,axis=0)-g
    local_error = (local_world-world[:-1]-g)-target_increment
    free_error = (np.diff(np.vstack((world[0],free_world)),axis=0)-g)-target_increment
    accumulated = {'local_increment':r.apply(np.cumsum(local_error,axis=0),inverse=True),
                   'feedback_increment':r.apply(np.cumsum(free_error-local_error,axis=0),inverse=True),
                   'position_lever':-np.cross(w_local,free[:,:3]-actual[:,:3])}
    err = require_close(free[:,3:6]-actual[:,3:6],sum(accumulated.values()))
    return accumulated, {'local':local_error,'free':free_error}, err


def audit(model,features,state,traces,keys,axis,settings):
    initial = _ball_state(state)
    is_pair = len(keys)==2
    take = half if is_pair else lambda x:x[0]
    start = np.broadcast_to(initial,(len(keys),9))
    truth = np.stack([traces[k]['raw'] for k in keys])
    free = forecast(model,start,features,truth)
    local = forecast(model,start,features,truth,local=True)
    first = take(start)
    a,l,f = (take(x) for x in (truth,local,free))
    dt = model.dt_s
    vel = {'local':l[:,3:6]-a[:,3:6], 'feedback':f[:,3:6]-l[:,3:6]}
    if is_pair:
        nominal = np.stack([traces[k]['raw'] for k in ('nominal_1','nominal_2')])
        analysis = diagnose_pair(model,initial,features,truth,nominal,settings)['velocity_m_s']
        vel['midpoint'] = np.asarray(analysis['midpoint_feedback'])
        vel['pair_half_difference'] = np.asarray(analysis['half_difference_feedback'])
        require_close(vel.pop('feedback'),vel['midpoint']+vel['pair_half_difference'])
    newton_remainder = integral(a[:,3:6],dt,first[3:6])-np.diff(np.vstack((first[:3],a[:,:3])),axis=0)
    increments = {k:integral(v,dt) for k,v in vel.items()}
    increments['newton_discrete_remainder'] = newton_remainder
    position_error = f[:,:3]-a[:,:3]
    err_velocity = require_close(sum(vel.values()),f[:,3:6]-a[:,3:6])
    err_position = require_close(np.cumsum(sum(increments.values()),axis=0),position_error)
    ground = []
    raw_world = []
    closure = 0.
    for i,k in enumerate(keys):
        frame = frame_series(state,traces[k],dt)
        terms, raw, e = branch_world_terms(frame,truth[i],local[i],free[i],settings['gravity_step'])
        ground.append(terms)
        raw_world.append(raw)
        closure = max(closure,e)
    world_vel = {k:take(np.stack([r[k] for r in ground])) for k in ground[0]}
    world_increments = {k:integral(v,dt) for k,v in world_vel.items()}
    world_increments['newton_discrete_remainder'] = newton_remainder
    err_velocity = max(err_velocity,require_close(sum(world_vel.values()),f[:,3:6]-a[:,3:6]),closure)
    err_position = max(err_position,require_close(np.cumsum(sum(world_increments.values()),axis=0),position_error))
    window = slice(7,24)
    return {
        'predicted_final_position_m':f[-1,:3].tolist(),'actual_final_position_m':a[-1,:3].tolist(),
        'position_error':describe(np.diff(np.vstack((np.zeros(3),position_error)),axis=0),axis),
        'velocity_error_rms_h8_h24_m_s':{k:rms(v[window]) for k,v in {**vel,'free':f[:,3:6]-a[:,3:6]}.items()},
        'world_increment_error_rms_h8_h24_m_s':{
            k:rms(take(np.stack([x[k] for x in raw_world]))[window]) for k in raw_world[0]},
        'integrated_state_errors':{k:describe(v,axis) for k,v in increments.items()},
        'integrated_world_errors':{k:describe(v,axis) for k,v in world_increments.items()},
        'velocity_closure_m_s':err_velocity,'position_closure_m':err_position,
        'free_branch_states':free.tolist(),'local_branch_states':local.tolist(),
    }


def run(plan,out,guard):
    def read(path):
        return json.loads(bound(plan,path))
    if plan['updates'] != 0 or plan['cases'] != ['nominal','dual_axis_pulse']:
        raise ValueError('unexpected diagnostic scope')
    recipe = read(plan['held_design'])
    capture,cfg,fields,traces,_ = read_held(recipe)
    state = _parse_state(fields)
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'),cfg['initial']['anchor_xy_tool_m'])[0]
    archive_paths = {'reference':BASE,'candidate':CANDIDATE}
    raw_models, rollouts = {}, {}
    for key,root in archive_paths.items():
        status = read(root/'validation.json')
        name = 'world_increment_900.npz' if key=='candidate' else 'velocity_response_leads_900.npz'
        for fname in (name,'timed_rollouts.json','comparison.json'):
            file = root/fname
            if plan['files_sha256'][file.as_posix()] != status['artifacts_sha256'][fname]:
                raise ValueError(f'archived endpoint changed: {file}')
        if (status['updates'],status['training_pairs'],status['test_episode_files_read']) != (900,49,False):
            raise ValueError('unexpected endpoint')
        with np.load(ROOT/root/name,allow_pickle=False) as npz:
            model = (WorldIncrementMLP.from_state_dict if key=='candidate' else load_position_model)(npz)
            raw_models[key] = model,npz['known_mean'],npz['known_scale']
        rollouts[key] = read(root/'timed_rollouts.json')
    dt = raw_models['reference'][0].dt_s
    if dt != raw_models['candidate'][0].dt_s:
        raise ValueError('endpoint clock mismatch')
    settings = {'floors_si':{'position':capture['gates']['position_floor_m'],'velocity':capture['gates']['velocity_floor_m_s']},
                'consecutive_leads':3,'relative_error_limit':.5,
                'gravity_step':np.array([0.,0.,-cfg['physics']['gravity_m_s2']*dt])}
    nominal_pos = traces['nominal_1']['raw'][-1,:3]
    drift = np.asarray(rollouts['candidate']['nominal'][-1][:3])-nominal_pos
    duo = ('dual_axis_pulse_plus','dual_axis_pulse_minus')
    dual = half(np.stack([traces[k]['raw'][-1,:3] for k in duo]))
    scores = {}
    for label,keys,direction in (('nominal',('nominal_1',),drift),('dual_axis_pulse',duo,dual)):
        axis = direction/np.linalg.norm(direction)
        requests = np.stack([np.asarray(traces[k]['actions']) for k in keys])
        known = known_rollout(state.time_s,state.tilt_rad,state.tilt_rate_rad_s,requests,cfg,dt)
        scores[label] = {'projection_axis_tool':axis.tolist(),'models':{}}
        for key,(model,mean,scale) in raw_models.items():
            features = model.features(known,cfg,anchor,mean,scale) if key=='candidate' else (known-mean)/scale
            result = audit(model,features,state,traces,keys,axis,settings)
            recorded = rollouts[key]
            expected = [recorded['nominal']] if label=='nominal' else [recorded['pairs'][label][side] for side in ('plus','minus')]
            require_close(result['free_branch_states'],np.asarray(expected))
            scores[label]['models'][key] = result
            guard(label,log=True)
    write_json(out/'drift_decomposition.json',{'cases':scores,
        'convention':'Nominal projections use candidate final nominal error direction. Dual projections use Newton final position half-difference direction. Measured tool twist and pose are used after prediction only.',
        'position_error_horizons':WINDOWS})
    return {'fixed_endpoints':2,'development_snapshots':1,'physical_branch_labels':3,
            'world_and_tool_position_closure':True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
