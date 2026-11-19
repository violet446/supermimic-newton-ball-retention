"""Audit moving-frame derivatives and measured tool endpoint agreement on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ball_retention.action_eval import evaluate_pair
from ball_retention.dataset import load_episode, read_manifest
from ball_retention.frame_grad import frame_action_sensitivity
from ball_retention.frame_model import FrameResidualMLP, pack_features
from ball_retention.model_data import _rotation, known_rollout
from ball_retention.pair_data import prepare_pairs
from ball_retention.task import reference


def main():
    out = ROOT/'runs/s3_frame_refit_20261113'
    pairs = prepare_pairs(ROOT/'runs/s3_pair_refit_20261112/capture_manifest.json')
    result = {'models':{},'method':'exact actuator, coordinate transform and network chain rule',
              'finite_difference_epsilon':1e-5,'test_used':False,'simulation_steps':0}
    for name in ('frame_fixed','frame_measured'):
        with np.load(out/(name+'.npz'),allow_pickle=False) as saved:
            model = FrameResidualMLP.from_state_dict(saved)
            mean,scale,anchor = saved['known_mean'],saved['known_scale'],saved['anchor_local']
            cfg = json.loads(str(saved['model_config_json']))
        rows = []
        for split in ('train','validation'):
            batch = getattr(pairs,split)
            for i,state in enumerate(batch.initial_state):
                actions,axis = batch.nominal_actions[i],int(batch.axes[i])
                derivative = frame_action_sensitivity(model,batch.initial[i],state['time_s'],state['tilt_rad'],
                    state['tilt_rate_rad_s'],actions,cfg,mean,scale,anchor)
                predictions = []
                eps = 1e-5
                for sign in (1,-1):
                    known = known_rollout(state['time_s'],state['tilt_rad'],state['tilt_rate_rad_s'],
                                         actions+sign*eps*np.eye(2)[axis],cfg,model.dt_s)
                    packed = pack_features(known,cfg,model.dt_s,anchor,model.state_mean,model.state_scale,mean,scale)
                    predictions.append(model.denormalize(model.rollout(model.normalize(batch.initial[i])[None],packed[None]))[0])
                numerical = (predictions[0]-predictions[1])/(2*eps)
                analytic = derivative['jacobian'][:,:,axis]
                errors = {}
                for start,quantity in ((0,'position'),(3,'velocity'),(6,'spin')):
                    error = numerical[:,start:start+3]-analytic[:,start:start+3]
                    errors[quantity] = {'max_abs':float(np.max(np.abs(error))),
                        'relative_l2':float(np.linalg.norm(error)/max(np.linalg.norm(numerical[:,start:start+3]),1e-12))}
                    np.testing.assert_allclose(numerical[:,start:start+3],analytic[:,start:start+3],atol=1e-8,rtol=2e-4)
                half = pairs.amplitude*analytic
                center = derivative['prediction']
                response = evaluate_pair(center+half,center-half,batch.plus_target[i],batch.minus_target[i],
                    pairs.amplitude,anchor[:2],replay_first=batch.baseline_first[i],replay_second=batch.baseline_second[i],leads=(8,16,32))
                if not (batch.plus_mask[i].all() and batch.minus_mask[i].all() and batch.baseline_mask[i].all()):
                    raise ValueError('this diagnostic expects full measured branch/replay leads')
                rows.append({'split':split,'episode':str(batch.episode_ids[i]),'tick':int(batch.snapshot_ticks[i]),
                             'axis':axis,'numerical_errors':errors,'linearized_response':response})
        result['models'][name] = rows

    # Future measured tool states are used here solely for equivalence diagnostics.
    data = ROOT/'data/s2_pilot_20261112'
    manifest = read_manifest(data)
    maxima = {'position_m':0.,'rotation_rad':0.,'origin_velocity_m_s':0.,'omega_rad_s':0.}
    checked = 0
    for entry in manifest['entries']:
        if entry['split'] not in ('train','validation'):
            continue
        for row in load_episode(data/entry['path'],manifest['dt_s'],manifest['control_substeps'],entry['episode_id']):
            state = row.next_state_before_reset
            r,roll_axis = _rotation(np.array(state.tilt_rad),np.array(cfg['initial']['tool_rotation_world']))
            omega = roll_axis*state.tilt_rate_rad_s[0]+np.array([0.,state.tilt_rate_rad_s[1],0.])
            ref = reference(state.time_s,cfg)
            position = ref['anchor_position']-r@anchor
            velocity = ref['anchor_velocity']-np.cross(omega,r@anchor)
            errors = {'position_m':np.linalg.norm(position-state.tool.position),
                      'rotation_rad':(Rotation.from_matrix(r).inv()*Rotation.from_quat(state.tool.quaternion_xyzw)).magnitude(),
                      'origin_velocity_m_s':np.linalg.norm(velocity-state.tool.linear_velocity_origin),
                      'omega_rad_s':np.linalg.norm(omega-state.tool.angular_velocity)}
            for key,value in errors.items():
                maxima[key] = max(maxima[key],float(value))
            checked += 1
    result['tool_endpoint_diagnostic'] = {'states_checked':checked,'maxima':maxima,
        'purpose':'measured states only for diagnostics; no future states used by model prediction'}
    paths = [Path(__file__),ROOT/'src/ball_retention/frame_grad.py',ROOT/'tests/test_frame_model.py']
    result['source_sha256'] = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/'gradients.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'tool_endpoints':result['tool_endpoint_diagnostic'],
        'gradient_max_absolute_error':max(v['max_abs'] for rows in result['models'].values() for r in rows for v in r['numerical_errors'].values())},indent=2))


if __name__ == '__main__':
    main()
