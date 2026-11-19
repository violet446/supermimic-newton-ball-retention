"""Check the fixed trapezoid transition and fitting recipe without updating parameters."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

for thread_var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[thread_var] = '1'

import numpy as np

from train_s3_timing_coverage import ROOT, action_sequences, digest, read_held, write_json
from train_s3_timing_lead_weights import anchor_draws
from ball_retention.action_grad import _inputs, predict_ball
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.deterministic_response_model import _directional_rollout
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES, _ball_state
from ball_retention.pair_loss import paired_loss
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.resources import directory_bytes
from ball_retention.timing_coverage_fit import DESIGN as PARENT_DESIGN, prepare
from ball_retention.world_model import ResidualMLP


DESIGN = 'configs/s3_position_integral_fit_plan.json'
DESIGN_SHA256 = 'c62cbe84bdbd53bba28c318ee0a03ff41d33d206881c981d276b47eb7a232c7f'
OUT = ROOT/'runs/s3_position_integral_preflight_20261115'
REFERENCE = 'held_offsets_plus_timing'
CONTROL = 'held_offsets'


def prepare_recipe(plan, guard):
    inputs = {name: _verified_bytes(ROOT, name, sha) for name, sha in plan['inputs_sha256'].items()}
    accepted = json.loads(inputs[plan['parent_run']])
    parent, cfg, scales, folds, _, legacy, anchors, _ = prepare(ROOT, guard)
    fold = folds[REFERENCE]
    draws = anchor_draws(cfg, fold, plan['seed'], plan['updates'])
    if (plan['parent_design'] != PARENT_DESIGN or accepted['status'] != 'completed'
            or not accepted['control_reproduced_exactly'] or accepted['updates'] != {CONTROL: 900, REFERENCE: 900}
            or accepted['checkpoint_sha256'][REFERENCE] != plan['inputs_sha256'][plan['matched_reference']]
            or accepted['checkpoint_sha256'][CONTROL] != plan['inputs_sha256'][plan['legacy_protection_reference']]
            or accepted['simulation_steps'] != 0 or accepted['test_episode_files_read'] is not False
            or plan['model_type'] != TrapezoidPositionResidualMLP.MODEL_TYPE
            or len(fold.pairs.initial) != 46 or parent['seed'] != plan['seed']
            or parent['updates_per_arm'] != plan['updates']
            or cfg['pairs']['lead_steps'] != plan['paired_leads']
            or cfg['pairs']['lead_weights'] != plan['paired_lead_weights']
            or draws != plan['anchor_draws_sha256'] or draws != accepted['identical_anchor_draws_sha256']):
        raise ValueError('frozen data, recipe, or reference identity changed')
    return inputs, parent, cfg, scales, fold, legacy, anchors, draws


def position_check(model, initial, known):
    predictions = model.denormalize(model.rollout(initial, known))
    before = np.concatenate((model.denormalize(initial)[:, None], predictions[:, :-1]), axis=1)
    expected = before[..., :3]+model.dt_s*.5*(before[..., 3:6]+predictions[..., 3:6])
    np.testing.assert_allclose(predictions[..., :3], expected, rtol=0., atol=1e-14)
    return float(np.max(abs(predictions[..., :3]-expected)))


def checkpoint_check(plan, inputs, baseline, fresh, fold, initial, known):
    for key in baseline.params:
        np.testing.assert_array_equal(baseline.params[key], fresh.params[key])
    np.testing.assert_array_equal(baseline.step(initial, known[:, 0]), fresh.step(initial, known[:, 0]))
    raw = BytesIO(inputs[plan['matched_reference']])
    with np.load(raw, allow_pickle=False) as saved:
        legacy = load_position_model(saved)
        if type(legacy) is not ResidualMLP or int(saved['updates']) != plan['updates']:
            raise ValueError('expected the original residual checkpoint format and endpoint')
        for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold, key))
        np.testing.assert_array_equal(saved['known_names'], KNOWN_NAMES)
        np.testing.assert_array_equal(legacy.residual_gain, fresh.residual_gain)
        np.testing.assert_equal(legacy.dt_s, fresh.dt_s)
        np.testing.assert_equal(legacy.hidden, fresh.hidden)
    # Use learned velocity and spin parameters to exercise nontrivial gradients.
    for key in legacy.params:
        fresh.params[key][...] = legacy.params[key]
    np.testing.assert_array_equal(legacy.step(initial, known[:, 0])[..., 3:],
                                  fresh.step(initial, known[:, 0])[..., 3:])
    stream = BytesIO()
    np.savez_compressed(stream, **fresh.state_dict())
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as saved:
        model = load_position_model(saved)
        # NumPy versions differ in the KeyError text for a missing archive member.
        if 'residual_gain' in saved or 'trapezoid_residual_gain' not in saved:
            raise ValueError('tagged checkpoint has an unexpected gain key')
        try:
            ResidualMLP.from_state_dict(saved)
        except KeyError:
            pass
        else:
            raise ValueError('the legacy loader silently accepted an incompatible transition')
    np.testing.assert_array_equal(fresh.rollout(initial, known), model.rollout(initial, known))
    return model


def dormant_check(grads):
    np.testing.assert_array_equal(grads['W3'][:, :3], 0.)
    np.testing.assert_array_equal(grads['b3'][:3], 0.)


def gradient_check(model, initial, known, checks, rng, horizon):
    step = checks['vjp_centered_step']
    if horizon == 1:
        known = known[:, 0]
        forward, vjp = model.step, model.step_vjp
    else:
        forward, vjp = model.rollout, model.rollout_vjp

    def unit(shape):
        direction = rng.standard_normal(shape)
        return direction/np.linalg.norm(direction)

    cotangent = unit(forward(initial, known).shape)

    def objective(x, k):
        return float(np.sum(cotangent*forward(x, k)))

    state_grad, known_grad, grads = vjp(initial, known, cotangent)
    dormant_check(grads)
    dx, dk = unit(initial.shape), unit(known.shape)
    derivatives = {
        'state': (np.sum(state_grad*dx), (objective(initial+step*dx, known)-objective(initial-step*dx, known))/(2*step)),
        'known': (np.sum(known_grad*dk), (objective(initial, known+step*dk)-objective(initial, known-step*dk))/(2*step)),
    }
    # Isolate each parameter block so large gradients cannot hide a small one.
    for name, parameter in model.params.items():
        value, direction = parameter.copy(), unit(parameter.shape)
        try:
            parameter[...] = value+step*direction
            plus = objective(initial, known)
            parameter[...] = value-step*direction
            minus = objective(initial, known)
        finally:
            parameter[...] = value
        derivatives[name] = (np.sum(grads[name]*direction), (plus-minus)/(2*step))
    for name, (analytic, centered) in derivatives.items():
        np.testing.assert_allclose(analytic, centered, rtol=checks['vjp_rtol'], atol=checks['vjp_atol'],
                                   err_msg=f'H{horizon} derivative for {name}')
    return {name: {'analytic': float(a), 'centered': float(b), 'abs_error': float(abs(a-b))}
            for name, (a, b) in derivatives.items()}


def action_check(model, held, fold, checks):
    capture, config, snapshot, traces, _ = held
    state = _parse_state(snapshot)
    nominal = traces['nominal_1']['actions']
    directions, actions = action_sequences(capture, nominal)
    for name, requests in actions.items():
        np.testing.assert_array_equal(requests, traces[name]['actions'])
    args = (model, _ball_state(state), state.time_s, state.tilt_rad, state.tilt_rate_rad_s)
    tail = (config, fold.known_mean, fold.known_scale)
    initial, raw, features = _inputs(*args, nominal, *tail)
    rows = np.asarray(list(directions.values()))
    step = checks['action_centered_step']
    if np.any(abs(nominal)+step*abs(rows) >= 1.):
        raise ValueError('held directions touch request clipping')
    prediction, slopes = _directional_rollout(model, initial, raw, features, nominal, rows, config, fold.known_scale)
    np.testing.assert_array_equal(prediction, predict_ball(*args, nominal, *tail))
    errors = {}
    for name, direction, slope in zip(directions, rows, slopes, strict=True):
        plus, minus = (predict_ball(*args, nominal+sign*step*direction, *tail) for sign in (1., -1.))
        centered = (plus-minus)/(2*step)
        np.testing.assert_allclose(slope, centered, rtol=checks['action_rtol'], atol=checks['action_atol_si'], err_msg=name)
        errors[name] = abs(slope-centered).max(axis=0).tolist()
    return {'case': capture['case'], 'horizon': len(nominal), 'max_abs_error_si_by_component': errors}


def loss_check(model, fold, cfg, scales, initial, known):
    batch = fold.pairs
    recipe = cfg['pairs']
    leads = np.zeros(known.shape[2])
    leads[np.asarray(recipe['lead_steps'])-1] = recipe['lead_weights']
    units = np.asarray(scales['state_error_scale_si'])
    components = np.asarray(cfg['components'])
    pair, pair_grads = paired_loss(model, initial, known, np.stack((batch.plus_target, batch.minus_target)),
                                   np.stack((batch.plus_mask, batch.minus_mask)), batch.response_mask,
                                   batch.noise_mask, batch.response_floor, leads, components,
                                   np.asarray(cfg['response_scales_si']), recipe['absolute_weight'],
                                   recipe['response_weight'], sample_weights=fold.pair_weights, absolute_scale_si=units)
    window = fold.anchors[max(cfg['anchors']['horizons'])]
    anchor, anchor_grads = model.loss_and_grad(
        model.normalize(window.initial[:2]), (window.known[:2]-fold.known_mean)/fold.known_scale,
        model.normalize(window.target[:2]), state_weights=components, state_error_scale_si=units)
    for grads in (pair_grads, anchor_grads):
        dormant_check(grads)
        if not all(np.isfinite(g).all() for g in grads.values()):
            raise FloatingPointError('static loss check found a nonfinite gradient')
    return {'anchor_loss': anchor, 'paired_loss': pair}


def run(out):
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    scope = {'simulation_steps': 0, 'training_updates': 0, 'test_episode_files_read': False}
    try:
        plan = json.loads(_verified_bytes(ROOT, DESIGN, DESIGN_SHA256))
        budget = plan['static_budget']

        def guard():
            memory = process_memory(os.getpid())
            if (time.monotonic()-started >= budget['max_wall_s']
                    or memory['peak_rss_bytes'] > budget['max_process_rss_gib']*1024**3
                    or memory['peak_private_commit_bytes'] > budget['max_process_commit_gib']*1024**3
                    or directory_bytes(out) > budget['max_new_output_mib']*1024**2):
                raise RuntimeError('position integral preflight exceeded a resource cap')
            return memory

        guard()
        inputs, parent, cfg, scales, fold, legacy, anchors, draws = prepare_recipe(plan, guard)
        args = (len(KNOWN_NAMES), fold.state_mean, fold.state_scale, anchors.dt_s, plan['seed'])
        kwargs = {'hidden': cfg['network'][1], 'residual_gain': np.array(scales['residual_gain_si'])/fold.state_scale}
        baseline = ResidualMLP(*args, **kwargs)
        fresh = TrapezoidPositionResidualMLP(*args, **kwargs)
        initial = fresh.normalize(fold.pairs.initial)
        known = (np.stack((fold.pairs.plus_known, fold.pairs.minus_known))-fold.known_mean)/fold.known_scale
        np.testing.assert_allclose(fresh.denormalize(initial), fold.pairs.initial, rtol=0., atol=1e-14)
        fresh_losses = loss_check(fresh, fold, cfg, scales, initial, known)
        model = checkpoint_check(plan, inputs, baseline, fresh, fold, initial, known[0])
        position_error = max(position_check(model, initial, branch) for branch in known)
        checks = plan['static_checks']
        rng = np.random.default_rng(plan['seed'])
        vjps = {str(h): gradient_check(model, initial[:3], known[0, :3, :h], checks, rng, h) for h in (1, 32)}
        guard()
        actions = action_check(model, read_held(parent), fold, checks)
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('the static preflight loaded a physics module')
        paths = {Path(__file__).resolve(), ROOT/DESIGN, ROOT/'scripts/check_s3_actions.py',
                 ROOT/'scripts/train_s3_timing_coverage.py', ROOT/'scripts/train_s3_timing_lead_weights.py',
                 *(ROOT/'src/ball_retention').glob('*.py')}
        write_json(out/'preflight.json', {
            'status': 'passed', 'design_sha256': DESIGN_SHA256, 'limits': budget, 'tolerances': checks,
            'inputs_sha256': plan['inputs_sha256'],
            'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)},
            'pair_count': len(initial), 'validation_counts': legacy.report['splits']['validation'],
            'anchor_draws_sha256': draws, 'matched_normalization': True, 'fresh_parameter_arrays_equal': True,
            'fresh_step_matches_constant_velocity': True, 'velocity_and_spin_same_input_exact': True,
            'rollout_reload_exact': True, 'legacy_loader_rejects_tagged_checkpoint': True,
            'trapezoid_position_max_abs_error_m': position_error, 'vjp': vjps, 'held_action_jvp': actions,
            'fresh_losses': fresh_losses, 'position_head_gradients_zero': True,
            **scope, 'wall_s': time.monotonic()-started, **guard(),
        })
        guard()
    except BaseException as error:
        write_json(out/'stopped.json', {'reason': str(error), 'wall_s': time.monotonic()-started, **scope})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    target = parser.parse_args().output.resolve()
    if len(target.relative_to(ROOT/'runs').parts) != 1:
        raise ValueError('choose a new run directory immediately under runs')
    run(target)
