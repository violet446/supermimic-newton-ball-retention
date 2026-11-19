"""Trace frozen world-increment differences by the lead that produced them."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np

from audit_s3_ball_velocity_components import frame_series
from audit_s3_world_increment_drift import WINDOWS, describe, half, require_close
from audit_s3_world_increment_pitch_drift import LABELS, load_extended
from audit_s3_world_increment_supervision_drift import load_endpoint
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, known_rollout
from ball_retention.world_increment_model import FEATURES
from ball_retention.world_model import ResidualMLP
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'world_increment_generation'
CASES = LABELS[:2]


def prior(start, states):
    return np.concatenate((np.broadcast_to(start, (2, 1, 9)), states[:, :-1]), axis=1)


def world_head(model, features, state):
    normalized = model.normalize(state)
    neural, _ = ResidualMLP._forward(model, normalized, features[..., :FEATURES])
    return (neural[..., 3:6]-normalized[..., 3:6])*model.state_scale[3:6]


def propagate(increments, transform, world, dt):
    """Propagate every source lead independently with frozen linear tool kinematics."""
    branches, horizon, _ = increments.shape
    position = np.zeros((horizon, branches, 3))
    velocity = np.zeros_like(position)
    trace = np.zeros((horizon, branches, horizon, 6))
    for t in range(horizon):
        pre = np.concatenate((position, velocity), axis=-1)
        next_velocity = (transform[:, t]@pre[..., None])[..., 0]
        next_velocity[t] += (world[:, t]@increments[:, t, :, None])[..., 0]
        position += (.5*dt)*(velocity+next_velocity)
        velocity = next_velocity
        trace[:, :, t, :3] = position
        trace[:, :, t, 3:] = velocity
    return trace


def summarize(trace, axis):
    paired = .5*(trace[:, 0]-trace[:, 1])
    pos = paired[..., :3]
    # Each row is a generating lead. Column t is its contribution at evaluation lead t.
    position = pos @ axis
    report = describe(pos[:, -1], axis)
    report['source_projection_at_h32_m'] = position[:, -1].tolist()
    report['source_projection_by_evaluation_lead_m'] = position.tolist()
    report['total_projection_over_time_m'] = position.sum(axis=0).tolist()
    report['by_source_window_at_evaluation_lead_m'] = {
        f'{first}-{last}': position[first-1:last].sum(axis=0).tolist() for first, last in WINDOWS}
    return report


def run(plan, out, guard):
    if plan['updates'] != 0 or tuple(plan['cases']) != CASES:
        raise ValueError('unexpected increment-generation diagnostic scope')
    held = read_held(json.loads(bound(plan, plan['held_design'])))
    capture, cfg, fields, traces, _ = held
    archived = json.loads(bound(plan, plan['previous_pitch_decomposition']))
    if capture['case'] != archived['development_case']:
        raise ValueError('unexpected development capture')
    state = _parse_state(fields)
    initial = _ball_state(state)
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), cfg['initial']['anchor_xy_tool_m'])[0]
    endpoints = {'supervised': load_endpoint(plan, 'supervised', held, anchor, guard),
                 'extended': load_extended(plan, held, anchor, guard)}
    previous, current = endpoints.values()
    dt = previous[0].dt_s
    if (previous[-1] != current[-1] or current[0].dt_s != dt or dt != archived['dt_s']
            or not np.array_equal(previous[0].inc_center, current[0].inc_center)):
        raise ValueError('unmatched tool, sampler or output center')
    gravity = np.array([0., 0., -cfg['physics']['gravity_m_s2']*dt])
    cases = {}
    max_step, max_propagation = 0., 0.
    for label in CASES:
        keys = tuple(f'{label}_{side}' for side in ('plus', 'minus'))
        truth = np.stack([traces[k]['raw'] for k in keys])
        known = known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                              np.stack([traces[k]['actions'] for k in keys]), cfg, dt)
        states, predictions, inputs, local = {}, {}, {}, {}
        for key, (model, fold, _, rollout, _) in endpoints.items():
            features = model.features(known, cfg, anchor, fold.known_mean, fold.known_scale)
            states[key] = np.asarray([rollout['pairs'][label][side] for side in ('plus', 'minus')])
            inputs[key] = features
            rolled_prior = prior(initial, states[key])
            predictions[key] = world_head(model, features, rolled_prior)
            local[key] = world_head(model, features, prior(initial, truth))
            transform = features[..., FEATURES:FEATURES+18].reshape(*features.shape[:-1], 3, 6)
            world = features[..., -9:].reshape(*features.shape[:-1], 3, 3)
            shift = features[..., FEATURES+18:FEATURES+21]
            rebuilt = ((transform@rolled_prior[..., :6, None])[..., 0]+shift
                       +(world@predictions[key][..., None])[..., 0])
            max_step = max(max_step, require_close(rebuilt, states[key][..., 3:6]))
        np.testing.assert_array_equal(inputs['supervised'], inputs['extended'])
        actual_world = [frame_series(state, traces[k], dt)[3] for k in keys]
        measured = np.diff(np.stack(actual_world), axis=1)-gravity
        total = predictions['extended']-predictions['supervised']
        independent = local['extended']-local['supervised']
        midpoint = np.broadcast_to(total.mean(axis=0), total.shape)
        source_groups = {'total': total, 'on_measured_pre_state': independent, 'change_due_to_rolled_input': total-independent,
                         'branch_mean': midpoint, 'branch_half_difference': total-midpoint}
        traces_by_source = {name: propagate(change, transform, world, dt) for name, change in source_groups.items()}
        total_trace = traces_by_source['total']
        delta = states['extended'][..., :6]-states['supervised'][..., :6]
        max_propagation = max(max_propagation, require_close(total_trace.sum(axis=0), delta))
        for groups in (('on_measured_pre_state', 'change_due_to_rolled_input'), ('branch_mean', 'branch_half_difference')):
            max_propagation = max(max_propagation, require_close(sum(traces_by_source[k] for k in groups), total_trace))
        axis = np.asarray(archived['cases'][label]['axis'])
        attributed = {name: summarize(trace, axis) for name, trace in traces_by_source.items()}
        expected = archived['cases'][label]['results']['extended_minus_supervised']['position_error']
        max_propagation = max(max_propagation, require_close(attributed['total']['vector_m'], np.asarray(expected['vector_m'])))
        cases[label] = {'projection_axis_tool': axis.tolist(), 'attribution': attributed,
            'measured_world_half_increment_m_s': half(measured).tolist(),
            'model_world_half_increment_free_m_s': {key: half(v+previous[0].inc_center).tolist() for key, v in predictions.items()},
            'model_world_half_increment_local_m_s': {key: half(v+previous[0].inc_center).tolist() for key, v in local.items()}}
        guard(label, log=True)
    write_json(out/'increment_generation.json', {'development_case': capture['case'], 'dt_s': dt,
        'source_windows': WINDOWS, 'results': cases,
        'convention': 'Exact accounting for the increments recorded on two fixed rollouts. Deterministic tool transitions reuse the same requested actions. A contribution holds downstream learned increments to the recorded values.'})
    return {'fixed_endpoints': 2, 'development_pairs': len(cases), 'archived_scores_reproduced': True,
            'max_velocity_head_closure_m_s': max_step, 'max_propagation_closure_si': max_propagation}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
