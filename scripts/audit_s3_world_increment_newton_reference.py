"""Trace errors against Newton with the model's own deterministic tool transform."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np

from audit_s3_ball_velocity_components import frame_series
from audit_s3_world_increment_drift import half, require_close
from audit_s3_world_increment_generation import CASES, prior, propagate, summarize, world_head
from audit_s3_world_increment_pitch_drift import load_extended
from audit_s3_world_increment_supervision_drift import load_endpoint
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, known_rollout
from ball_retention.world_increment_model import FEATURES
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'world_increment_newton_reference'


def tool_transition(features):
    axes = features.shape[:-1]
    transform = features[..., FEATURES:FEATURES+18].reshape(*axes, 3, 6)
    world = features[..., -9:].reshape(*axes, 3, 3)
    shift = features[..., FEATURES+18:FEATURES+21]
    return transform, world, shift


def required_residual(pre, truth, transform, world, shift):
    transported = (transform@pre[..., :6, None])[..., 0]+shift
    delta = truth[..., 3:6]-transported
    answer = np.linalg.solve(world, delta[..., None])[..., 0]
    error = require_close(transported+(world@answer[..., None])[..., 0], truth[..., 3:6])
    return answer, error


def propagate_discrete_position(position_source, transform, dt):
    """Propagate missing Newton position updates when the model uses trapezoids."""
    branches, horizon, _ = position_source.shape
    pos = np.zeros((horizon, branches, 3))
    vel = np.zeros_like(pos)
    trace = np.zeros((horizon, branches, horizon, 6))
    for t in range(horizon):
        pre = np.concatenate((pos, vel), axis=-1)
        next_vel = (transform[:, t]@pre[..., None])[..., 0]
        pos += (.5*dt)*(vel+next_vel)
        pos[t] += position_source[:, t]
        vel = next_vel
        trace[:, :, t, :3] = pos
        trace[:, :, t, 3:] = vel
    return trace


def compare_pair(label, endpoints, traces, state, cfg, anchor, axes, earlier):
    keys = tuple(f'{label}_{side}' for side in ('plus', 'minus'))
    truth = np.stack([traces[k]['raw'] for k in keys])
    initial = _ball_state(state)
    dt = endpoints['supervised'][0].dt_s
    known = known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                          np.stack([traces[k]['actions'] for k in keys]), cfg, dt)
    features = {k: m.features(known, cfg, anchor, f.known_mean, f.known_scale)
                for k, (m, f, *_) in endpoints.items()}
    np.testing.assert_array_equal(features['supervised'], features['extended'])
    transform, world, shift = tool_transition(features['supervised'])
    measured_pre = prior(initial, truth)
    baseline, newton_closure = required_residual(measured_pre, truth, transform, world, shift)
    position_source = measured_pre[..., :3]+.5*dt*(measured_pre[..., 3:6]+truth[..., 3:6])-truth[..., :3]
    discrete = propagate_discrete_position(position_source, transform, dt)
    gravity = np.array([0., 0., -cfg['physics']['gravity_m_s2']*dt])
    measured_world = [frame_series(state, traces[k], dt)[3] for k in keys]
    measured_increment = np.diff(np.stack(measured_world), axis=1)-gravity
    center = endpoints['supervised'][0].inc_center
    axis = np.asarray(axes)
    models = {}
    net = {}
    max_closure = 0.
    for tag, (model, _, _, rollout, _) in endpoints.items():
        prediction = np.asarray([rollout['pairs'][label][s] for s in ('plus', 'minus')])
        free_head = world_head(model, features[tag], prior(initial, prediction))
        local_head = world_head(model, features[tag], measured_pre)
        sources = {'on_measured_pre_state': local_head-baseline, 'change_due_to_rolled_input': free_head-local_head}
        propagated = {k: propagate(v, transform, world, dt) for k, v in sources.items()}
        net[tag] = propagate(free_head-baseline, transform, world, dt)
        max_closure = max(max_closure, require_close(net[tag], sum(propagated.values())))
        absolute = net[tag]+discrete
        max_closure = max(max_closure, require_close(absolute.sum(axis=0), prediction[..., :6]-truth[..., :6]))
        models[tag] = {'learned_increment_error': summarize(net[tag], axis),
                       'total_error': summarize(absolute, axis),
                       'split_learned_error': {k: summarize(v, axis) for k, v in propagated.items()},
                       'absolute_position_half_error_norm_m': np.linalg.norm(half(prediction[..., :3]-truth[..., :3]),
                                                                          axis=-1).tolist()}
    prior_matrix = earlier['attribution']['total']['source_projection_by_evaluation_lead_m']
    max_closure = max(max_closure, require_close(
        summarize(net['extended']-net['supervised'], axis)['source_projection_by_evaluation_lead_m'], prior_matrix))
    return {'axis': axes, 'analytic_reference_world_half_increment_m_s': half(baseline+center).tolist(),
            'measured_world_half_increment_m_s': half(measured_increment).tolist(),
            'discrete_position_error': summarize(discrete, axis), 'models': models}, newton_closure, max_closure


def run(plan, out, guard):
    if plan['updates'] != 0 or tuple(plan['cases']) != CASES:
        raise ValueError('unexpected Newton-reference scope')
    held = read_held(json.loads(bound(plan, plan['held_design'])))
    capture, cfg, fields, traces, _ = held
    generation = json.loads(bound(plan, plan['accepted_generation']))
    if generation['development_case'] != capture['case']:
        raise ValueError('development case changed')
    state = _parse_state(fields)
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), cfg['initial']['anchor_xy_tool_m'])[0]
    endpoints = {'supervised': load_endpoint(plan, 'supervised', held, anchor, guard),
                 'extended': load_extended(plan, held, anchor, guard)}
    a, b = endpoints.values()
    if a[-1] != b[-1] or a[0].dt_s != b[0].dt_s or a[0].dt_s != generation['dt_s']:
        raise ValueError('archive sampler or clock changed')
    result = {}
    newton_error, propagation_error = 0., 0.
    for label, source in generation['results'].items():
        result[label], nerr, perr = compare_pair(label, endpoints, traces, state, cfg, anchor,
                                                 source['projection_axis_tool'], source)
        newton_error, propagation_error = max(newton_error, nerr), max(propagation_error, perr)
        guard(label, log=True)
    write_json(out/'newton_reference.json', {'development_case': capture['case'], 'results': result,
        'convention': 'Reference residual reproduces the measured next tool-relative velocity with the analytic tool map. The model-minus-Newton position defect is carried separately.'})
    return {'fixed_endpoints': len(endpoints), 'development_pairs': len(result), 'archived_scores_reproduced': True,
            'newton_velocity_step_closure_m_s': newton_error, 'trajectory_error_closure_max_si': propagation_error}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
