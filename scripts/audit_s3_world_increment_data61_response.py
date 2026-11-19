"""Diagnose fixed full-window checkpoints using only archived responses."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from audit_s3_world_increment_drift import WINDOWS, require_close
from audit_s3_world_increment_generation import prior, propagate, summarize, world_head
from audit_s3_world_increment_newton_reference import propagate_discrete_position, required_residual, tool_transition
from audit_s3_world_increment_pitch_drift import EXTENDED, load_extended
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, known_rollout
from ball_retention.pair_collection import prepare_collection
from ball_retention.world_increment_model import WorldIncrementMLP
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held
from train_s3_world_increment import action_sequences, predict, score_candidate


PHASE = 'world_increment_data61_response'
EXPANDED = Path('runs/s3_world_increment_data61_fit_20261117')
RATIOS = (0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.)
HISTORIES = ('history_validation_yminus006', 'history_validation_yplus006')


def read(plan, path):
    return json.loads(bound(plan, path))


def norm(values):
    return np.linalg.norm(values, axis=-1)


def unit(vector):
    return vector / norm(vector)


def half(pair):
    return .5 * (pair[0] - pair[1])


def window_changes(values):
    return {f'{a}-{b}': (values[b-1] - (values[a-2] if a > 1 else 0.)).tolist()
            for a, b in WINDOWS}


def cosine(estimate, reference):
    return float(unit(estimate) @ unit(reference))


def fixed_models(plan, held, anchor, guard):
    old, fold49, score49, rollout49, sampler = load_extended(plan, held, anchor, guard)
    validation = read(plan, EXPANDED/'validation.json')
    comparison = read(plan, EXPANDED/'comparison.json')
    checkpoint = EXPANDED/'world_increment_900.npz'
    for path in (checkpoint, EXPANDED/'comparison.json', EXPANDED/'timed_rollouts.json'):
        if plan['files_sha256'][path.as_posix()] != validation['artifacts_sha256'][path.name]:
            raise ValueError('61-pair fit provenance changed')
    if (validation['status'], validation['updates'], validation['training_pairs'],
            validation['sampler_sha256']) != ('completed', 900, 61, sampler):
        raise ValueError('wrong fixed fit')
    with np.load(ROOT/checkpoint, allow_pickle=False) as saved:
        new = WorldIncrementMLP.from_state_dict(saved)
        fold61 = SimpleNamespace(known_mean=saved['known_mean'], known_scale=saved['known_scale'])
        if saved['updates'].item() != 900:
            raise ValueError('wrong checkpoint update')
    for name in ('known_mean', 'known_scale'):
        np.testing.assert_array_equal(getattr(fold49, name), getattr(fold61, name))
    for name in ('dt_s', 'state_mean', 'state_scale', 'inc_center'):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    score61, rollout61 = score_candidate(new, held, fold61, anchor, guard)
    if score61 != comparison['held_timing'] or rollout61 != read(plan, EXPANDED/'timed_rollouts.json'):
        raise ValueError('61-pair score or trajectory did not reproduce')
    return {'49_pairs': (old, fold49, score49, rollout49), '61_pairs': (new, fold61, score61, rollout61)}


def amplitude_scan(model, fold, state, cfg, anchor, nominal, direction, amplitude, branch, truth):
    linearized = amplitude * np.asarray(branch['jvp'])[:, :3]
    full = half(np.asarray([branch['plus'], branch['minus']]))[:, :3]
    reference = truth[-1]
    rows = []
    for fraction in RATIOS:
        forecasts = [predict(model, state, nominal + s*fraction*amplitude*direction, cfg, anchor,
                             fold.known_mean, fold.known_scale) for s in (1, -1)]
        secant = half(forecasts)[:, :3] / fraction
        if fraction == 1.:
            require_close(secant, full)
        rows.append({'amplitude_fraction': fraction, 'amplitude': fraction*amplitude,
                     'rescaled_half_response_by_lead_m': secant.tolist(),
                     'h32_distance_to_model_linearization_m': float(norm(secant[-1]-linearized[-1])),
                     'h32_distance_to_model_full_m': float(norm(secant[-1]-full[-1])),
                     'h32_cosine_vs_full_amplitude_newton': cosine(secant[-1], reference),
                     'h32_relative_error_vs_full_amplitude_newton': float(norm(secant[-1]-reference)/norm(reference))})
    gap = full-linearized
    return {'rows': rows, 'linearized_by_lead_m': linearized.tolist(),
            'full_minus_linearized_by_lead_m': gap.tolist(), 'gap_change_by_evaluation_window_m': window_changes(gap),
            'h32_full_minus_linearized_m': float(norm(gap[-1])),
            'linearized_h32_cosine': cosine(linearized[-1], reference),
            'linearized_h32_relative_error': float(norm(linearized[-1]-reference)/norm(reference))}


def explain(model, features, start, truth, prediction, named_axes):
    transform, world, shift = tool_transition(features)
    measured_pre = prior(start, truth)
    required, closure = required_residual(measured_pre, truth, transform, world, shift)
    local = world_head(model, features, measured_pre)
    rolled = world_head(model, features, prior(start, prediction))
    dt = model.dt_s
    position_source = measured_pre[..., :3]+(.5*dt)*(measured_pre[..., 3:6]+truth[..., 3:6])-truth[..., :3]
    sources = {'on_measured_pre_state': propagate(local-required, transform, world, dt),
               'change_due_to_rolled_input': propagate(rolled-local, transform, world, dt),
               'newton_vs_trapezoid': propagate_discrete_position(position_source, transform, dt)}
    total = sum(sources.values())
    closure = max(closure, require_close(total.sum(axis=0), prediction[..., :6]-truth[..., :6]))
    return {k: {name: summarize(value, axis) for name, value in {**sources, 'total': total}.items()}
            for k, axis in named_axes.items()}, closure


def development(held, endpoints, anchor, guard):
    capture, cfg, fields, traces, _ = held
    state = _parse_state(fields)
    start = _ball_state(state)
    nominal = np.asarray(traces['nominal_1']['actions'])
    directions, actions = action_sequences(capture, nominal)
    results, max_closure = {}, 0.
    for case, direction in directions.items():
        keys = [f'{case}_{s}' for s in ('plus', 'minus')]
        truth = np.stack([traces[k]['raw'] for k in keys])
        response = half(truth)[:, :3]
        new_response = half([np.asarray(endpoints['61_pairs'][3]['pairs'][case][s]) for s in ('plus', 'minus')])
        axes = {'newton_h32': unit(response[-1]), '61_prediction_h32': unit(new_response[-1, :3])}
        rows = {}
        for label, (model, fold, score, rollout) in endpoints.items():
            branch = rollout['pairs'][case]
            prediction = np.asarray([branch['plus'], branch['minus']])
            known = known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                                  np.stack([actions[k] for k in keys]), cfg, model.dt_s)
            features = model.features(known, cfg, anchor, fold.known_mean, fold.known_scale)
            contributions, closure = explain(model, features, start, truth, prediction, axes)
            max_closure = max(closure, max_closure)
            rows[label] = {'sweep': amplitude_scan(model, fold, state, cfg, anchor, nominal, direction,
                                                  capture['amplitude'], branch, response),
                           'full_prediction_half_position_by_lead_m': half(prediction)[:, :3].tolist(),
                           'score': score[case]['by_lead']['32']['position'], 'attribution': contributions}
        results[case] = {'newton_half_position_by_lead_m': response.tolist(),
                         'projection_axes': {k: v.tolist() for k, v in axes.items()}, 'models': rows}
        guard(case, log=True)
    return results, max_closure


def legacy(plan, endpoints, mesh, guard):
    collection_path = read(plan, plan['held_design'])['base_collection']
    bound(plan, collection_path)
    collection = prepare_collection(ROOT, ROOT/collection_path)
    batch = collection.validation.batch
    cfg = collection.model_config
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    selected = np.flatnonzero(np.isin(batch.group_ids, HISTORIES) & (collection.validation.amplitudes == .2))
    if len(selected) != 16 or not np.all(batch.plus_mask[selected] & batch.minus_mask[selected]):
        raise ValueError('unexpected legacy protection rows')
    archived = {}
    for tag, folder in (('49_pairs', EXTENDED), ('61_pairs', EXPANDED)):
        summary = read(plan, folder/'comparison.json')['legacy_validation']
        archived[tag] = {(r['episode'], r['tick'], r['axis']): r for r in summary['pairs']
                         if r['group'] in HISTORIES and r['amplitude'] == .2}
        if len(archived[tag]) != len(selected):
            raise ValueError('legacy comparison no longer has the same pair identities')
    results, max_closure = [], 0.
    for i in selected:
        episode, tick, axis = str(batch.episode_ids[i]), int(batch.snapshot_ticks[i]), int(batch.axes[i])
        truth = np.stack((batch.plus_target[i], batch.minus_target[i]))
        known = np.stack((batch.plus_known[i], batch.minus_known[i]))
        full_truth = half(truth)[:, :3]
        start = batch.initial[i]
        old_position = archived['49_pairs'][episode, tick, axis]['response']['by_lead']['32']['position']
        strong = old_position['newton_half_spread'] >= 10 * old_position['signal_threshold']
        rows = {}
        for tag, (model, fold, _, _) in endpoints.items():
            features = model.features(known, cfg, anchor, fold.known_mean, fold.known_scale)
            prediction = model.denormalize(model.rollout(model.normalize(np.broadcast_to(start, (2, 9))), features))
            half_prediction = half(prediction)[:, :3]
            absolute_error = prediction[:, :, :3]-truth[:, :, :3]
            model_archive = archived[tag][episode, tick, axis]
            for key in ('8', '16', '32'):
                t = int(key)-1
                expected = model_archive['response']['by_lead'][key]['position']
                require_close(half_prediction[t], .2*np.asarray(expected['model_derivative']))
                require_close(full_truth[t], .2*np.asarray(expected['newton_derivative']))
                require_close(np.sum(absolute_error[:, t]**2), model_archive['absolute'][key]['squared_p_v_w_si'][0])
            sources, closure = explain(model, features, start, truth, prediction, {'newton_h32': unit(full_truth[-1])})
            max_closure = max(closure, max_closure)
            midpoint_error = absolute_error.mean(axis=0)
            pair_error = half(absolute_error)
            require_close(np.mean(norm(absolute_error)**2, axis=0), norm(midpoint_error)**2+norm(pair_error)**2)
            rows[tag] = {'attribution': sources['newton_h32'],
                         'half_position_by_lead_m': half_prediction.tolist(),
                         'direction_cosine_by_saved_lead': {k: model_archive['response']['by_lead'][k]['position']['cosine']
                                                            for k in ('8', '16', '32')},
                         'absolute_mse_by_lead_m2': np.mean(norm(absolute_error)**2, axis=0).tolist(),
                         'midpoint_mse_by_lead_m2': (norm(midpoint_error)**2).tolist(),
                         'half_response_mse_by_lead_m2': (norm(pair_error)**2).tolist()}
        results.append({'episode': episode, 'tick': tick, 'axis': axis, 'group': str(batch.group_ids[i]),
                        'newton_h32_strong': strong,
                        'newton_half_position_by_lead_m': full_truth.tolist(), 'models': rows})
        guard('legacy_history')
    if any(sum(r['newton_h32_strong'] for r in results if r['group'] == g) != 6 for g in HISTORIES):
        raise ValueError('unexpected legacy strong-response inventory')
    return results, max_closure


def run(plan, out, guard):
    if plan['updates'] or plan['amplitude_fractions'] != list(RATIOS):
        raise ValueError('unexpected offline audit scope')
    held = read_held(read(plan, plan['held_design']))
    mesh = load_tool(ROOT/'assets/shovel.STL')
    anchor = support_point(mesh, held[1]['initial']['anchor_xy_tool_m'])[0]
    endpoints = fixed_models(plan, held, anchor, guard)
    dev, a = development(held, endpoints, anchor, guard)
    histories, b = legacy(plan, endpoints, mesh, guard)
    write_json(out/'response_attribution.json', {'development_case': held[0]['case'],
        'requested_amplitude': held[0]['amplitude'], 'windows': WINDOWS, 'development': dev, 'legacy': histories,
        'amplitude_convention': 'Model secants rescaled to full amplitude. Archived Newton response exists only at full amplitude.',
        'attribution_convention': 'Generation lead holds downstream learned increments fixed; prior true state is used only in the independent one-step term.'})
    return {'fixed_checkpoints': len(endpoints), 'development_pairs': len(dev), 'legacy_pairs': len(histories),
            'max_development_closure_si': a, 'max_legacy_closure_si': b, 'archived_scores_reproduced': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
