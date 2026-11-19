"""Attribute paired rollout feedback to three components of the prior ball state."""
import argparse
import json
from itertools import combinations, permutations
from pathlib import Path

from s3_fitability_common import ROOT, forecast, run_phase, write_json

import numpy as np

from audit_s3_world_increment_drift import describe, half, integral, require_close
from audit_s3_world_increment_pitch_drift import LABELS, difference_summary, load_extended
from audit_s3_world_increment_supervision_drift import load_endpoint
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, known_rollout
from ball_retention.world_increment_model import FEATURES
from diagnose_s3_advance_feedback import diagnose_pair
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'world_increment_state_feedback'
BLOCKS = {'position': slice(0, 3), 'linear_velocity': slice(3, 6), 'spin': slice(6, 9)}
OUTPUTS = ('next_velocity', 'carried_linear_velocity', 'tool_frame_transform', 'learned_increment')
CASES = LABELS[:2]


def evaluate_mixed(model, features, midpoint, response):
    state = np.stack((midpoint+response, midpoint-response))
    velocity = model.denormalize(model.step(model.normalize(state), features))[..., 3:6]
    matrix = features[..., FEATURES:FEATURES+18].reshape(*state.shape[:-1], 3, 6)
    affine = (matrix@state[..., :6, None])[..., 0]+features[..., FEATURES+18:FEATURES+21]
    carried = state[..., 3:6]
    return np.stack([half(x) for x in (velocity, carried, affine-carried, velocity-affine)])


def per_midpoint(model, features, midpoint, measured, rolled):
    evaluated = {}
    for count in range(len(BLOCKS)+1):
        for subset in combinations(BLOCKS, count):
            response = measured.copy()
            for block in subset:
                response[:, BLOCKS[block]] = rolled[:, BLOCKS[block]]
            evaluated[frozenset(subset)] = evaluate_mixed(model, features, midpoint, response)
    outcomes = {block: [] for block in BLOCKS}
    for order in permutations(BLOCKS):
        current = frozenset()
        for block in order:
            after = current | {block}
            outcomes[block].append(evaluated[after]-evaluated[current])
            current = after
    return {block: np.stack(rows) for block, rows in outcomes.items()}, evaluated


def decompose(model, features, initial, truth, nominal, axis, settings):
    start = np.broadcast_to(initial, (2, 9))
    free = forecast(model, start, features, truth)
    measured_prior, rollout_prior = (np.concatenate((start[:, None], x[:, :-1]), axis=1) for x in (truth, free))
    d0, d1 = half(measured_prior), half(rollout_prior)
    evaluated = [per_midpoint(model, features, prior.mean(axis=0), d0, d1) for prior in (measured_prior, rollout_prior)]
    corners = [corners for _, corners in evaluated]
    empty, full = frozenset(), frozenset(BLOCKS)
    contributions = {k: np.mean([result[k] for result, _ in evaluated], axis=0) for k in BLOCKS}
    mean = {k: np.mean(v, axis=0) for k, v in contributions.items()}
    prior = diagnose_pair(model, initial, features, truth, nominal, settings)['velocity_m_s']
    expected = np.asarray(prior['half_difference_feedback'])
    by_order = sum(contributions.values())
    velocity_closure = require_close(by_order[:, 0], np.broadcast_to(expected, by_order[:, 0].shape))
    velocity_closure = max(velocity_closure, require_close(sum(v[0] for v in mean.values()), expected))
    midpoint = .5*((corners[1][empty][0]-corners[0][empty][0])+(corners[1][full][0]-corners[0][full][0]))
    velocity_closure = max(velocity_closure, require_close(midpoint, np.asarray(prior['midpoint_feedback'])))
    velocity_closure = max(velocity_closure, require_close(half(free)[:, 3:6], np.asarray(prior['free'])))
    for block, values in mean.items():
        # Each state contribution closes into the literal velocity carry, coordinate correction and learned part.
        velocity_closure = max(velocity_closure, require_close(values[0], values[1:].sum(axis=0)))
        carried = (d1-d0)[:, BLOCKS['linear_velocity']] if block == 'linear_velocity' else np.zeros_like(values[1])
        velocity_closure = max(velocity_closure, require_close(values[1], carried))
    velocity_closure = max(velocity_closure, require_close(mean['spin'][2], np.zeros_like(mean['spin'][2])))
    result = {}
    for block, values in mean.items():
        integrated = {part: describe(integral(row, model.dt_s), axis) for part, row in zip(OUTPUTS, values)}
        totals = [float(integral(v[0], model.dt_s).sum(axis=0)@axis) for v in contributions[block]]
        result[block] = {'integrated': integrated, 'per_lead_velocity_m_s': dict(zip(OUTPUTS, values.tolist())),
                         'per_order_h32_projection_range_m': [min(totals), max(totals)]}
    expected_position = describe(integral(expected, model.dt_s), axis)
    integrated_total = sum(np.asarray(v['integrated']['next_velocity']['vector_m']) for v in result.values())
    position_closure = require_close(integrated_total, np.asarray(expected_position['vector_m']))
    return result, expected_position, velocity_closure, position_closure


def run(plan, out, guard):
    if plan['updates'] != 0 or tuple(plan['cases']) != CASES or tuple(plan['state_blocks']) != tuple(BLOCKS):
        raise ValueError('unexpected state-feedback diagnostic scope')
    held = read_held(json.loads(bound(plan, plan['held_design'])))
    capture, cfg, fields, traces, _ = held
    state = _parse_state(fields)
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), cfg['initial']['anchor_xy_tool_m'])[0]
    endpoints = {'supervised': load_endpoint(plan, 'supervised', held, anchor, guard),
                 'extended': load_extended(plan, held, anchor, guard)}
    recorded = json.loads(bound(plan, plan['previous_pitch_decomposition']))
    if (capture['case'] != recorded['development_case']
            or endpoints['supervised'][-1] != endpoints['extended'][-1]
            or any(x[0].dt_s != recorded['dt_s'] for x in endpoints.values())):
        raise ValueError('unexpected archive, sampler or time step')
    dt = recorded['dt_s']
    initial = _ball_state(state)
    nominal = np.stack([traces[k]['raw'] for k in ('nominal_1', 'nominal_2')])
    settings = {'floors_si': {'position': capture['gates']['position_floor_m'],
                              'velocity': capture['gates']['velocity_floor_m_s']},
                'consecutive_leads': 3, 'relative_error_limit': .5}
    results = {}
    velocity_closure, position_closure = 0., 0.
    for label in CASES:
        keys = [f'{label}_{side}' for side in ('plus', 'minus')]
        truth = np.stack([traces[k]['raw'] for k in keys])
        axis = np.asarray(recorded['cases'][label]['axis'])
        known = known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                              np.stack([traces[k]['actions'] for k in keys]), cfg, dt)
        models = {}
        for tag, (model, fold, _, _, _) in endpoints.items():
            features = model.features(known, cfg, anchor, fold.known_mean, fold.known_scale)
            terms, total, v, p = decompose(model, features, initial, truth, nominal, axis, settings)
            archived = recorded['cases'][label]['results'][tag]['integrated_state_errors']['pair_half_difference']
            if archived != total:
                raise ValueError('previous half-difference feedback did not reproduce')
            velocity_closure, position_closure = max(velocity_closure, v), max(position_closure, p)
            models[tag] = {'by_state': terms, 'half_difference_feedback': total}
            guard(label, log=True)
        extended, supervised = (models[k]['by_state'] for k in ('extended', 'supervised'))
        delta = {k: {part: difference_summary(extended[k]['integrated'][part], supervised[k]['integrated'][part])
                    for part in OUTPUTS} for k in BLOCKS}
        results[label] = {'projection_axis': axis.tolist(), 'models': models, 'extended_minus_supervised': delta}
    write_json(out/'state_feedback.json', {'development_case': capture['case'], 'results': results,
        'method': 'Use all six state block replacement orders at both the measured and rolled branch midpoint. Replace only the prior half-difference of the specified blocks in standalone one-step evaluations.'})
    return {'fixed_endpoints': 2, 'development_pairs': len(results), 'replacement_orders_per_midpoint': 6,
            'archived_scores_and_feedback_reproduced': True, 'velocity_closure_m_s': velocity_closure,
            'position_closure_m': position_closure}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
