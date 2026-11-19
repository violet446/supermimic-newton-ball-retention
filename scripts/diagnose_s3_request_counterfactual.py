"""Separate the direct request input from the causal tool motion features."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from s3_fitability_common import ROOT, SCOPE, run_phase, write_json

import numpy as np

from ball_retention.action_grad import _inputs
from ball_retention.archive_state_import import _parse_state
from ball_retention.model_data import KNOWN_NAMES, _ball_state
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from deterministic_response_metrics import response_metrics
from diagnose_s3_action_input_gradients import GROUPS, decompose
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held, score_held
from train_s3_timing_coverage import action_sequences


REQUEST = np.array([KNOWN_NAMES.index(name) for name in GROUPS['request']])
OTHER = np.array([i for i in range(len(KNOWN_NAMES)) if i not in REQUEST])


def pin_requests(features, nominal):
    out = features.copy()
    out[..., REQUEST] = nominal[..., REQUEST]
    np.testing.assert_array_equal(out[..., OTHER], features[..., OTHER])
    return out


def archive(plan, path):
    status = json.loads(bound(plan, path))
    report_names = ('comparison.json', 'timed_rollouts.json', 'velocity_response_leads_900.npz')
    if (status['status'] != 'completed' or status['updates'] != 900 or status['training_pairs'] != 49
            or status['local_scale_m_s'] != .0025 or status['checkpoint_selection'] is not False
            or any(status[k] != value for k, value in SCOPE.items())
            or status['checkpoint_sha256'] != status['artifacts_sha256'][report_names[-1]]
            or any(plan['files_sha256'][path.with_name(name).as_posix()] != status['artifacts_sha256'][name]
                   for name in report_names)):
        raise ValueError('unexpected frozen model identity')
    comparison, rollouts = (json.loads(bound(plan, path.with_name(name))) for name in report_names[:2])
    if rollouts['checkpoint_sha256'] != status['checkpoint_sha256']:
        raise ValueError('forecast archive is bound to another parameter endpoint')
    with np.load(ROOT/path.with_name(report_names[-1]), allow_pickle=False) as saved:
        model = load_position_model(saved)
        if type(model) is not TrapezoidPositionResidualMLP or int(saved['updates']) != 900 or tuple(saved['known_names']) != KNOWN_NAMES:
            raise ValueError('unexpected transition or feature order')
        norms = SimpleNamespace(known_mean=saved['known_mean'], known_scale=saved['known_scale'])
    return model, norms, comparison, rollouts


def diagnose(model, norms, reference, original_rollouts, held, guard):
    case, cfg, fields, traces, _ = held
    state = _parse_state(fields)
    start = (model, _ball_state(state), state.time_s, state.tilt_rad, state.tilt_rate_rad_s)
    tools = (cfg, norms.known_mean, norms.known_scale)
    nominal = np.asarray(traces['nominal_1']['actions'])
    directions, actions = action_sequences(case, nominal)
    for label, requested in actions.items():
        np.testing.assert_array_equal(traces[label]['actions'], requested)
    spec = case['model']
    horizon, amplitude, h = case['horizon'], case['amplitude'], spec['fd_step']
    if not 0. < h < amplitude or np.any(abs(nominal)+amplitude*abs(np.stack(list(directions.values()))) >= 1.):
        raise ValueError('the signed probe leaves the differentiable action regime')
    initial, raw_nominal, normalized_nominal = _inputs(*start, nominal, *tools)

    def features(requested):
        _, raw, values = _inputs(*start, requested, *tools)
        np.testing.assert_array_equal(raw[..., :2], raw_nominal[..., :2])
        np.testing.assert_array_equal(raw[..., REQUEST], requested)
        first_change = np.flatnonzero(np.any(requested != nominal, axis=1))
        prefix = int(first_change[0]) if len(first_change) else horizon
        np.testing.assert_array_equal(raw[:prefix], raw_nominal[:prefix])
        return pin_requests(values, normalized_nominal)

    def rollout(known):
        return model.denormalize(model.rollout(initial[None], known[None])[0])

    def finite(coefficient, direction):
        return rollout(features(nominal+coefficient*direction))

    expected_time = state.time_s+(np.arange(horizon)+.5)*model.dt_s
    np.testing.assert_array_equal(raw_nominal[:, 0], expected_time)
    np.testing.assert_array_equal(features(nominal), normalized_nominal)
    prediction = rollout(normalized_nominal)
    np.testing.assert_array_equal(prediction, original_rollouts['nominal'])
    plain, forecasts = score_held(model, held, norms, guard)
    if plain != reference['held_timing'] or forecasts != {key: original_rollouts[key] for key in forecasts}:
        raise ValueError('original frozen scores or forecasts failed to reproduce')
    full, grouped, _, _, bands, closure = decompose(
        model, norms.known_mean, norms.known_scale, fields, cfg, nominal, directions, original_rollouts)
    request_index = tuple(GROUPS).index('request')
    request = grouped[:, :, request_index]
    without_request = grouped[:, :, np.arange(len(GROUPS)) != request_index].sum(axis=2)
    np.testing.assert_allclose(request+without_request, full, rtol=1e-12, atol=5e-15)
    floor_keys = ('position_floor_m', 'velocity_floor_m_s', 'spin_floor_rad_s')
    compare_keys = ('noise_multiplier', 'strong_multiplier', 'min_cosine', 'max_relative_error')
    options = {key: case['gates'][key] for key in compare_keys}
    options['floors'] = tuple(case['gates'][key] for key in floor_keys)
    options['leads'] = tuple(range(1, horizon+1))
    base = [traces[label]['raw'] for label in ('nominal_1', 'nominal_2')]
    results, prefixes = {}, {}
    for i, (name, direction) in enumerate(directions.items()):
        candidate = {}
        for step in (h, h/2):
            fd = (finite(step, direction)-finite(-step, direction))/(2*step)
            np.testing.assert_allclose(without_request[i], fd, rtol=spec['fd_rtol'], atol=spec['fd_atol_si'],
                                       err_msg=f'{name}: request-fixed local JVP at step {step}')
            candidate[str(step)] = np.max(abs(fd-without_request[i]), axis=0).tolist()
        plus, minus = (finite(sign*amplitude, direction) for sign in (1., -1.))
        prefixes[name] = (plus, minus)
        physics = [traces[f'{name}_{sign}']['raw'] for sign in ('plus', 'minus')]
        effect = response_metrics(base, *physics, prediction, plus, minus, without_request[i], amplitude, **options)
        frozen = original_rollouts['pairs'][name]
        baseline = response_metrics(base, *physics, prediction, frozen['plus'], frozen['minus'], full[i],
                                    amplitude, **options)
        original_half = .5*(np.asarray(frozen['plus'])-np.asarray(frozen['minus']))
        fixed_half = .5*(plus-minus)
        results[name] = {'full': baseline, 'request_fixed': effect,
                         'direct_request_jvp_per_unit_si': request[i].tolist(),
                         'tool_feature_jvp_per_unit_si': without_request[i].tolist(),
                         'jvp_by_feature_and_injection_band_per_unit_si': bands[i].tolist(),
                         'full_minus_request_fixed_finite_half_difference_si': (original_half-fixed_half).tolist(),
                         'request_fixed_fd_max_component_error': candidate}
        guard()
    for pulse, reversal in zip(prefixes['pitch_pulse'], prefixes['pitch_reversal'], strict=True):
        np.testing.assert_array_equal(pulse[:8], reversal[:8])
    return {'directions': results, 'max_analytic_jvp_closure_error_si': closure}


def run(plan, out, guard):
    if plan['updates'] != 0 or len(plan['endpoints']) != 2 or plan['request_columns'] != list(GROUPS['request']):
        raise ValueError('unexpected request counterfactual scope')
    parent = json.loads(bound(plan, plan['held_design']))
    held = read_held(parent)
    if held[0]['case'] != plan['held_case']:
        raise ValueError('frozen development case changed')
    models = {}
    for label, status in plan['endpoints'].items():
        model, norms, scores, forecasts = archive(plan, Path(status))
        models[label] = diagnose(model, norms, scores, forecasts, held, lambda: guard(label))
        guard(label, log=True)
    write_json(out/'request_counterfactual.json', {'models': models, 'case': plan['held_case'],
        'input_groups': GROUPS, 'injection_bands': [[1, 8], [9, 16], [17, 24], [25, 32]],
        'method': 'Freeze only the direct network request columns to the causal nominal features. '
                  'The analytic tool features come from the signed actions. Both models roll out their own states. '
                  'Newton stored finite-amplitude response is the physical comparison for both model paths.'})
    return {'parameter_points': len(models), 'directions_per_point': 3, 'frozen_nominal_and_scores_reproduced': True,
            'request_fixed_centered_fd_verified': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'request_counterfactual', run)
