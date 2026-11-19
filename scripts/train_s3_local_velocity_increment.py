"""Preflight then fit one matched 49-pair model with mixed velocity supervision."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from s3_fitability_common import ROOT, SCOPE, branch_arrays, run_phase, write_json

import numpy as np
import train_s3_advance_coverage as coverage
import train_s3_velocity_response_leads as dense
from ball_retention.action_grad import _inputs
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.local_velocity_response_loss import LocalVelocityResponse, half
from ball_retention.model_data import _ball_state
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.velocity_response_loss import VELOCITY, paired_velocity_response_loss, response_components
from preflight_s3_velocity_response_leads import parameter_check
from train_s3_position_integral import read_held, regression, score_held


REFERENCE = Path('runs/s3_fitability_p3_20261116/validation.json')
LEADS = (1, 8, 16, 24, 32)


def bound(plan, path):
    return _verified_bytes(ROOT, path, plan['files_sha256'][Path(path).as_posix()])


def incumbent(plan, fold):
    accepted = json.loads(bound(plan, REFERENCE))
    ck = REFERENCE.with_name('velocity_response_leads_900.npz')
    comparison = REFERENCE.with_name('comparison.json')
    if (accepted['status'] != 'completed' or accepted['updates'] != 900 or accepted['training_pairs'] != 49
            or any(accepted[key] != value for key, value in SCOPE.items())
            or accepted['anchor_draws_sha256'] != plan['anchor_draws_sha256']
            or any(plan['files_sha256'][p.as_posix()] != accepted['artifacts_sha256'][p.name]
                   for p in (ck, comparison))):
        raise ValueError('unexpected 49-pair matched reference')
    _verified_bytes(ROOT, ck, accepted['checkpoint_sha256'])
    with np.load(ROOT/ck, allow_pickle=False) as saved:
        model = load_position_model(saved)
        if type(model) is not TrapezoidPositionResidualMLP or int(saved['updates']) != 900:
            raise ValueError('unexpected reference architecture or endpoint')
        for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold, key))
    return model, json.loads(bound(plan, comparison))


def prepare(plan, fold, cfg, scales, leads, model, guard):
    if (plan['training_pairs'] != 49 or len(fold.pairs.initial) != 49
            or plan['transfer_fraction'] != .25 or plan['increment_floor_m_s'] != 1e-5):
        raise ValueError('local supervision plan does not match frozen scope')
    base, velocity = leads
    batch = fold.pairs
    components = np.asarray(cfg['components'])
    units = np.asarray(cfg['response_scales_si'])
    mass = response_components(base, velocity, components, batch.noise_mask, fold.pair_weights)
    args = (batch, mass, fold.pair_weights)
    setup = LocalVelocityResponse(*args, plan['transfer_fraction'], plan['increment_floor_m_s'])
    zero = LocalVelocityResponse(*args, 0., plan['increment_floor_m_s'])
    initial, known, truth, mask = branch_arrays(fold)
    branches = (model, model.normalize(initial), known, truth, mask, batch.response_mask,
                batch.noise_mask, batch.response_floor, base, components, units)
    a, r = (cfg['pairs'][key] for key in ('absolute_weight', 'response_weight'))
    options = {'velocity_lead_weights': velocity, 'sample_weights': fold.pair_weights,
               'absolute_scale_si': np.asarray(scales['state_error_scale_si'])}
    original, original_grad = paired_velocity_response_loss(*branches, a, r, **options)
    control, control_grad = zero.loss(*branches, a, r, **options)
    if original != {k: control[k] for k in original}:
        raise ValueError('zero transfer did not reproduce the original loss')
    for name in original_grad:
        np.testing.assert_array_equal(control_grad[name], original_grad[name])

    metrics, grads = setup.loss(*branches, a, r, **options)
    if metrics['absolute'] != original['absolute']:
        raise ValueError('the absolute branch term changed')
    original_absolute, old_absolute_grad = paired_velocity_response_loss(*branches, a, 0., **options)
    blended_absolute, new_absolute_grad = setup.loss(*branches, a, 0., **options)
    if original_absolute['loss'] != blended_absolute['loss']:
        raise ValueError('the absolute weighted loss changed')
    for name in old_absolute_grad:
        np.testing.assert_array_equal(old_absolute_grad[name], new_absolute_grad[name])
    directions = parameter_check(model, lambda **kw: setup.loss(*branches, a, r, **options, **kw), grads, plan, guard)
    local_loss, local_grad = setup.local_loss(model, known, units, gradients=True)
    if setup.audit['moved_response_mass'] <= 0. or not np.isfinite(local_loss):
        raise ValueError('local supervision mass or loss is invalid')
    norm = lambda values: float(np.linalg.norm(np.concatenate([value.ravel() for value in values])))
    delta_norm = norm(grads[k]-original_grad[k] for k in grads)
    local_norm = norm(r*value for value in local_grad.values())
    if not (np.isfinite(local_norm) and local_norm > 0. and np.isfinite(delta_norm)):
        raise ValueError('local increment parameter gradient is unusable')
    return setup, {'mass_and_noise': setup.audit, 'zero_transfer_exact': True, 'absolute_gradient_unchanged': True,
                   'original_loss': original, 'modified_loss': metrics, 'response_adjustment_gradient_norm': delta_norm,
                   'weighted_local_gradient_norm': local_norm, 'parameter_finite_differences': directions,
                   'local_target_rms_m_s': float(np.sqrt(np.mean(setup.target[setup.eligible]**2)))}


def training_local(model, fold, setup):
    known = branch_arrays(fold)[1]
    error = setup.estimate(model, known)[2]-setup.target
    result = {}
    for history, rows in (('all', slice(None)), ('reverse', slice(40, 46)), ('advance', slice(46, 49))):
        squared = np.sum(error[rows]**2, axis=-1)
        eligible = setup.eligible[rows]
        if not eligible.any() or any(not eligible[:, h-1].any() for h in LEADS):
            raise ValueError(f'no complete local diagnostic samples for {history}')
        result[history] = {'all_leads_rmse_m_s': float(np.sqrt(np.mean(squared[eligible]))),
                           'by_lead_rmse_m_s': {str(h): float(np.sqrt(np.mean(squared[eligible[:, h-1], h-1])))
                                                for h in LEADS}}
    return result


def development_local(model, fold, held):
    plan, cfg, state, traces, _ = held
    start = _parse_state(state)
    raw = _ball_state(start)
    result = {}
    for name in plan['directions']:
        signed = [traces[f'{name}_{sign}'] for sign in ('plus', 'minus')]
        truth = np.stack([trace['raw'] for trace in signed])
        features = np.stack([_inputs(model, raw, start.time_s, start.tilt_rad, start.tilt_rate_rad_s, trace['actions'],
                                     cfg, fold.known_mean, fold.known_scale)[2] for trace in signed])
        previous = np.concatenate((np.broadcast_to(raw, (2, 1, 9)), truth[:, :-1]), axis=1)
        local = model.denormalize(model.step(model.normalize(previous), features))
        errors = np.linalg.norm(half(local-truth)[..., VELOCITY], axis=-1)
        result[name] = {'all_leads_rmse_m_s': float(np.sqrt(np.mean(errors**2))),
                        'by_lead_error_m_s': {str(h): float(errors[h-1]) for h in LEADS}}
    return result


def preflight(plan, out, guard):
    design = coverage.read(coverage.DESIGN, coverage.DESIGN_SHA256)
    _, cfg, scales, ref, _, _, _, draws, _ = coverage.prepare_comparison(design, guard)
    if draws != plan['anchor_draws_sha256']:
        raise ValueError('unexpected S2 anchor stream')
    protocol = coverage.read(coverage.COVERAGE_DESIGN, coverage.COVERAGE_DESIGN_SHA)
    fold, _ = coverage.prepare_candidate(protocol, ref, guard)
    baseline, _ = incumbent(plan, fold)
    leads = coverage.lead_vectors(design, 32)
    setup, report = prepare(plan, fold, cfg, scales, leads, baseline, guard)
    report['reference_one_step_increment'] = training_local(baseline, fold, setup)
    write_json(out/'increment_preflight.json', report)
    return {'preflight_passed': True, 'training_pairs': 49}


def run(plan, out, guard):
    preflight_path = Path(plan['preflight'])
    accepted = json.loads(bound(plan, preflight_path))
    metrics_path = preflight_path.with_name('increment_preflight.json')
    if (accepted['status'] != 'completed' or accepted['updates'] != 0 or accepted['preflight_passed'] is not True
            or any(accepted[key] != value for key, value in SCOPE.items())
            or plan['files_sha256'][metrics_path.as_posix()] != accepted['artifacts_sha256'][metrics_path.name]):
        raise ValueError('local increment preflight did not pass')
    expected = json.loads(bound(plan, metrics_path))
    evidence = {}

    def fit(model, fold, cfg, scales, design, leads, output, track):
        baseline, reference = incumbent(plan, fold)
        setup, checks = prepare(plan, fold, cfg, scales, leads, baseline, guard)
        checks['reference_one_step_increment'] = training_local(baseline, fold, setup)
        if checks != expected:
            raise ValueError('increment loss and gradient preflight did not reproduce')
        held = read_held(json.loads(bound(plan, plan['held_design'])))
        held_ref, _ = score_held(baseline, held, fold, lambda: guard('reference_49'))
        if held_ref != reference['held_timing']:
            raise ValueError('49-pair development response did not reproduce')
        write_json(out/'increment_preflight.json', checks)
        applied = 0

        def mixed_loss(*args, **options):
            nonlocal applied
            applied += 1
            return setup.loss(*args, **options)

        # Only this worker uses the modified loss. The pinned fitter and its
        # Adam, S2 draw stream, update count and save/reload code stay intact.
        with patch.object(dense, 'paired_velocity_response_loss', mixed_loss):
            checkpoint, fitted, draws = dense.fit(model, fold, cfg, scales, design, leads, output, track)
        if applied != plan['updates'] or draws != plan['anchor_draws_sha256']:
            raise ValueError('unexpected increment loss call count or S2 draw stream')
        evidence.update(reference=reference, applied=applied,
            training={'reference_49': checks['reference_one_step_increment'], 'candidate_49': training_local(fitted, fold, setup)},
            development={'reference_49': development_local(baseline, fold, held),
                         'candidate_49': development_local(fitted, fold, held)})
        return checkpoint, fitted, draws

    with patch.object(coverage, 'fit', fit):
        result = coverage.run(plan, out, guard)
    comparison = json.loads((out/'comparison.json').read_bytes())
    reference = evidence['reference']
    matched = regression(reference['legacy_validation'], comparison['legacy_validation'])
    if len(matched['checks']) != 22:
        raise ValueError('expected all 22 protections against original 49-pair endpoint')
    comparison['legacy_protections']['dense_49'] = matched
    comparison['fixed_checks']['all_legacy_protections_vs_dense_49'] = matched['passed']
    comparison['velocity_half_difference_errors_vs_dense_49'] = dense.velocity_errors(
        comparison['held_timing'], reference['held_timing'])
    comparison['reference_held_timing_dense_49'] = reference['held_timing']
    comparison['training_timed']['reference_49'] = reference['training_timed']['candidate_49']
    comparison['independent_one_step_increment'] = {key: evidence[key] for key in ('training', 'development')}
    write_json(out/'comparison.json', comparison)
    return {**result, 'historical_reference_pairs': result['reference_pairs'], 'reference_pairs': 49,
            'increment_loss_updates': evidence['applied'], 'matched_49_development_reproduced': True,
            'fixed_checks': comparison['fixed_checks'], 'replacement_gate_passed': all(comparison['fixed_checks'].values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'fit'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'increment', preflight if args.phase == 'preflight' else run)
