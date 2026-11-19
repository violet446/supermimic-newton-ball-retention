"""Fit the matched 49-pair objective with direct network requests masked."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from s3_fitability_common import ROOT, SCOPE, run_phase, write_json

import numpy as np
import train_s3_advance_coverage as coverage
import train_s3_local_increment_scale as scale
import train_s3_local_velocity_increment as increment
import train_s3_velocity_response_leads as dense
from ball_retention.action_grad import _inputs, predict_ball
from ball_retention.archive_state_import import _parse_state
from ball_retention.model_data import KNOWN_NAMES, _ball_state
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.request_masked_model import REQUEST_COLUMNS, RequestMaskedTrapezoidMLP, load_request_model
from ball_retention.world_model import STATE_DIM
from train_s3_position_integral import read_held, regression, score_held
from train_s3_timing_coverage import action_sequences
from train_s3_local_increment_window import scaled_reference


REQUEST_ROWS = STATE_DIM+np.array(REQUEST_COLUMNS)


def clone_masked(source):
    state = source.state_dict()
    state['model_type'] = np.asarray(RequestMaskedTrapezoidMLP.MODEL_TYPE)
    model = load_request_model(state)
    for key in source.params:
        np.testing.assert_array_equal(model.params[key], source.params[key])
    return model


def matrix_checks(model, source, held, norms):
    capture, cfg, fields, traces, _ = held
    saved = _parse_state(fields)
    start = (model, _ball_state(saved), saved.time_s, saved.tilt_rad, saved.tilt_rate_rad_s)
    tools = (cfg, norms.known_mean, norms.known_scale)
    nominal = np.asarray(traces['nominal_1']['actions'])
    directions, branches = action_sequences(capture, nominal)
    expected_clock = saved.time_s+(np.arange(capture['horizon'])+.5)*model.dt_s
    features, forecasts = {}, {}
    for name, requests in branches.items():
        np.testing.assert_array_equal(requests, traces[name]['actions'])
        initial, known, normalized = _inputs(*start, requests, *tools)
        np.testing.assert_array_equal(known[:, 0], expected_clock)
        np.testing.assert_array_equal(known[:, REQUEST_COLUMNS], requests)
        _, cache = model._forward(initial, normalized[0])
        independent = normalized.copy()
        independent[:, REQUEST_COLUMNS] = 0.
        np.testing.assert_array_equal(cache[0][STATE_DIM:], independent[0])
        y = model.rollout(initial[None], normalized[None])
        np.testing.assert_array_equal(y, source.rollout(initial[None], independent[None]))
        features[name], forecasts[name] = known, y[0]
    other = [i for i in range(len(KNOWN_NAMES)) if i not in REQUEST_COLUMNS and i >= 2]
    if not np.any(features['dual_axis_pulse_plus'][:, other] != features['dual_axis_pulse_minus'][:, other]):
        raise ValueError('analytic tool features are insensitive to the signed commands')
    for sign in ('plus', 'minus'):
        pulse, reversal = f'pitch_pulse_{sign}', f'pitch_reversal_{sign}'
        np.testing.assert_array_equal(features[pulse][:8], features[reversal][:8])
        np.testing.assert_array_equal(forecasts[pulse][:8], forecasts[reversal][:8])
    eye = np.eye(STATE_DIM)
    state = np.broadcast_to(initial, eye.shape)
    _, _, normalized = _inputs(*start, nominal, *tools)
    ds, dk, dp = model.step_vjp(state, np.broadcast_to(normalized[0], (STATE_DIM, model.known_dim)), eye)
    np.testing.assert_array_equal(dk[:, REQUEST_COLUMNS], 0.)
    np.testing.assert_array_equal(dp['W1'][REQUEST_ROWS], 0.)
    if not np.isfinite(ds).all() or not np.any(dk[:, other]):
        raise ValueError('analytic feature derivative is non-finite or vanishes')
    _, full_dk, grads = model.rollout_vjp(initial[None], normalized[None], np.ones((1, len(nominal), STATE_DIM)))
    np.testing.assert_array_equal(full_dk[..., REQUEST_COLUMNS], 0.)
    np.testing.assert_array_equal(grads['W1'][REQUEST_ROWS], 0.)
    independent = normalized.copy()
    independent[:, REQUEST_COLUMNS] = 0.
    target = model.normalize(traces['nominal_1']['raw'])[None]
    direct_loss, direct_grads = source.loss_and_grad(initial[None], independent[None], target)
    masked_loss, masked_grads = model.loss_and_grad(initial[None], normalized[None], target)
    np.testing.assert_equal(masked_loss, direct_loss)
    for key in direct_grads:
        np.testing.assert_array_equal(masked_grads[key], direct_grads[key])

    snapshot = model.state_dict()
    loaded = load_request_model(snapshot)
    if type(loaded) is not RequestMaskedTrapezoidMLP:
        raise ValueError('masked checkpoint lost its input contract')
    np.testing.assert_array_equal(loaded.rollout(initial[None], normalized[None]),
                                  model.rollout(initial[None], normalized[None]))
    try:
        load_position_model(snapshot)
    except ValueError:
        pass
    else:
        raise ValueError('old checkpoint loader silently ignored masking')
    return start, tools, nominal, directions


def preflight(plan, out, guard):
    if (plan['updates'] != 0 or plan['masked_known_names'] != ['roll_request', 'pitch_request']
            or plan['local_scale_m_s'] != scale.LOCAL_SCALE or plan['training_pairs'] != 49):
        raise ValueError('unexpected request-ablation preflight scope')
    source, norms, reference = scaled_reference(plan)
    model = clone_masked(source)
    held = read_held(json.loads(increment.bound(plan, plan['held_design'])))
    original, _ = score_held(source, held, norms, lambda: guard('original'))
    if original != reference['held_timing']:
        raise ValueError('matching endpoint development scores did not reproduce')
    start, tools, nominal, directions = matrix_checks(model, source, held, norms)
    _, forecasts = score_held(model, held, norms, lambda: guard('masked_jvp'))
    pairs = {}
    for name, sequence in directions.items():
        item = forecasts['pairs'][name]
        spec = held[0]['model']
        step = spec['fd_step']/2.
        vals = [predict_ball(*start, nominal+sign*step*sequence, *tools) for sign in (1., -1.)]
        fd = (vals[0]-vals[1])/(2.*step)
        analytic = np.asarray(item['directional_jvp'])
        np.testing.assert_allclose(fd, analytic, rtol=spec['fd_rtol'], atol=spec['fd_atol_si'])
        pairs[name] = [max(item['fd_max_abs_error_si_by_component']), float(abs(fd-analytic).max())]
    write_json(out/'request_preflight.json', {'forward_matches_explicit_zero_inputs': True,
        'causal_clock_and_shared_prefix_exact': True, 'request_input_and_parameter_gradients_exact_zero': True,
        'training_gradients_match_explicit_zero_inputs': True,
        'checkpoint_loader_verified': True, 'model_scope': 'Numerical checks using existing parameters only.',
        'jvp_fd_max_error_by_step': pairs, 'fd_steps': [held[0]['model']['fd_step'], step]})
    return {'preflight_passed': True, 'training_pairs': 49, 'masked_known_names': plan['masked_known_names']}


def run(plan, out, guard):
    path = Path(plan['request_preflight'])
    report = json.loads(increment.bound(plan, path))
    checks = path.with_name('request_preflight.json')
    if (plan['updates'] != 900 or plan['local_scale_m_s'] != scale.LOCAL_SCALE or report['updates'] != 0
            or report['preflight_passed'] is not True or report['status'] != 'completed'
            or report['masked_known_names'] != ['roll_request', 'pitch_request']
            or report['masked_known_names'] != plan['masked_known_names']
            or any(report[k] != v for k, v in SCOPE.items())
            or report['artifacts_sha256'][checks.name] != plan['files_sha256'][checks.as_posix()]):
        raise ValueError('request-ablation preflight is outside the frozen scope')
    source, norms, prior = scaled_reference(plan)
    held = read_held(json.loads(increment.bound(plan, plan['held_design'])))
    score, _ = score_held(source, held, norms, lambda: guard('reference'))
    if score != prior['held_timing']:
        raise ValueError('matched development scores did not reproduce')

    original_fit = dense.fit

    def fit(initial, fold, cfg, scales, design, leads, output, track):
        candidate = clone_masked(initial)
        request_rows = candidate.params['W1'][REQUEST_ROWS].copy()

        def track_masked(stage, update, *, record=True):
            track(stage, update, record=record)
            np.testing.assert_array_equal(candidate.params['W1'][REQUEST_ROWS], request_rows)
            if not all(np.isfinite(value).all() for value in candidate.params.values()):
                raise FloatingPointError(f'nonfinite request-masked parameter at update {update}')

        with patch.object(dense, 'load_position_model', load_request_model):
            ck, trained, draws = original_fit(candidate, fold, cfg, scales, design, leads, output, track_masked)
        if type(trained) is not RequestMaskedTrapezoidMLP:
            raise ValueError('training reload lost the request-masked input contract')
        return ck, trained, draws

    with patch.object(dense, 'fit', fit):
        result = scale.run({**plan, 'preflight': plan['scale_preflight']}, out, guard)
    current = json.loads((out/'comparison.json').read_bytes())
    protection = regression(prior['legacy_validation'], current['legacy_validation'])
    if len(protection['checks']) != 22:
        raise ValueError('expected 22 protections against the matched 2.5 mm/s endpoint')
    current['legacy_protections']['matched_request_channels'] = protection
    current['fixed_checks']['all_legacy_protections_vs_matched_request_channels'] = protection['passed']
    current['reference_held_timing_matched_request_channels'] = prior['held_timing']
    current['velocity_half_difference_errors_vs_matched_request_channels'] = dense.velocity_errors(
        current['held_timing'], prior['held_timing'])
    current['training_timed']['reference_matched_request_channels'] = prior['training_timed']['candidate_49']
    for key in ('training', 'development'):
        current['independent_one_step_increment'][key]['reference_matched_request_channels'] = (
            prior['independent_one_step_increment'][key]['candidate_49'])
    write_json(out/'comparison.json', current)
    return {**result, 'matched_request_channel_reference_reproduced': True, 'masked_known_names': plan['masked_known_names'],
            'model_type': RequestMaskedTrapezoidMLP.MODEL_TYPE, 'fixed_checks': current['fixed_checks'],
            'replacement_gate_passed': all(current['fixed_checks'].values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'fit'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'request_masked', preflight if args.phase == 'preflight' else run)
