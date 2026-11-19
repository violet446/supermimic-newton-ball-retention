"""Audit and isolate the local velocity-increment loss scale."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from s3_fitability_common import ROOT, SCOPE, branch_arrays, run_phase, write_json

import numpy as np
import train_s3_advance_coverage as coverage
import train_s3_local_velocity_increment as increment
from ball_retention.pair_loss import paired_loss
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP
from ball_retention.velocity_response_loss import VELOCITY, paired_velocity_response_loss, response_components
from train_s3_position_integral import read_held, regression, score_held
from train_s3_velocity_response_leads import velocity_errors


UNSCALED = Path('runs/s3_local_increment_fit_20261116/validation.json')
REFERENCE_SCALE = .025
LOCAL_SCALE = .0025


class RescaledLocalVelocityResponse(increment.LocalVelocityResponse):
    def local_loss(self, model, known, response_units, *, gradients):
        units = np.array(response_units, copy=True)
        units[VELOCITY] = LOCAL_SCALE
        return super().local_loss(model, known, units, gradients=gradients)


def flat(gradient):
    return np.concatenate([value.ravel() for value in gradient.values()])


def relation(vector, removed):
    norm, reference = np.linalg.norm(vector), np.linalg.norm(removed)
    if not (np.isfinite(norm) and np.isfinite(reference) and norm > 0. and reference > 0.):
        raise FloatingPointError('expected finite nonzero gradient norms')
    return {'norm': float(norm), 'ratio_to_removed_free': float(norm/reference),
            'cosine_to_removed_free': float(np.clip((vector@removed)/(norm*reference), -1., 1.))}


def gradient_audit(plan, model, fold, cfg, scales, leads, guard):
    plain, before = increment.prepare(plan, fold, cfg, scales, leads, model, guard)
    with patch.object(increment, 'LocalVelocityResponse', RescaledLocalVelocityResponse):
        scaled, after = increment.prepare(plan, fold, cfg, scales, leads, model, guard)
    before_mass = before['mass_and_noise']
    if before_mass != after['mass_and_noise'] or not before['absolute_gradient_unchanged'] or not after['absolute_gradient_unchanged']:
        raise ValueError('weight, mask or absolute term changed under local-only scaling')
    for key in ('local_weights', 'floor', 'resolved', 'prior', 'target'):
        np.testing.assert_array_equal(getattr(plain, key), getattr(scaled, key))
    base, velocity = leads
    initial, known, truth, masks = branch_arrays(fold)
    batch = fold.pairs
    units = np.asarray(cfg['response_scales_si'])
    np.testing.assert_array_equal(units[VELOCITY], REFERENCE_SCALE)
    components = np.asarray(cfg['components'])
    weight = cfg['pairs']['response_weight']
    absolute = cfg['pairs']['absolute_weight']
    branches = (model, model.normalize(initial), known, truth, masks, batch.response_mask,
                batch.noise_mask, batch.response_floor)
    options = {'sample_weights': fold.pair_weights, 'absolute_scale_si': np.asarray(scales['state_error_scale_si'])}
    original_weights = response_components(base, velocity, components, batch.noise_mask, fold.pair_weights)
    removed_weights = original_weights-plain.free_components
    _, removed = paired_loss(*branches, np.ones(len(base)), removed_weights, units, 0.,
        weight*before_mass['moved_response_mass']/before_mass['response_mass'], **options)
    removed = flat(removed)
    _, original = paired_velocity_response_loss(*branches, base, components, units, absolute, weight,
                                                 velocity_lead_weights=velocity, **options)
    records = {}
    for label, setup, summary in (('25_mm_s', plain, before), ('2_5_mm_s', scaled, after)):
        _, local_grad = setup.local_loss(model, known, units, gradients=True)
        _, total = setup.loss(*branches, base, components, units, absolute, weight, **options)
        vector = weight*flat(local_grad)
        np.testing.assert_allclose(flat(total)-flat(original), vector-removed, rtol=0., atol=5e-13)
        records[label] = {**relation(vector, removed),
            'maximum_parameter_fd_error': max(row['abs_error'] for row in summary['parameter_finite_differences'].values())}
    after['reference_one_step_increment'] = increment.training_local(model, fold, scaled)
    return {'removed_free_gradient_norm': float(np.linalg.norm(removed)), 'local_gradient': records}, after


def preflight(plan, out, guard):
    if (plan['reference_local_scale_m_s'] != REFERENCE_SCALE or plan['local_scale_m_s'] != LOCAL_SCALE
            or plan['comparability_ratio'] != [.25, 4.]):
        raise ValueError('unexpected frozen scale comparison')
    design = coverage.read(coverage.DESIGN, coverage.DESIGN_SHA256)
    _, cfg, scales, ref, _, anchors, _, draws, _ = coverage.prepare_comparison(design, guard)
    if draws != plan['anchor_draws_sha256'] or plan['seed'] != design['seed']:
        raise ValueError('training stream identity changed')
    protocol = coverage.read(coverage.COVERAGE_DESIGN, coverage.COVERAGE_DESIGN_SHA)
    fold, _ = coverage.prepare_candidate(protocol, ref, guard)
    baseline, _ = increment.incumbent(plan, fold)
    fresh = TrapezoidPositionResidualMLP(baseline.known_dim, fold.state_mean, fold.state_scale, anchors.dt_s,
        plan['seed'], hidden=cfg['network'][1], residual_gain=np.asarray(scales['residual_gain_si'])/fold.state_scale)
    leads = coverage.lead_vectors(design, 32)
    endpoint, checks = gradient_audit(plan, baseline, fold, cfg, scales, leads, guard)
    start, _ = gradient_audit(plan, fresh, fold, cfg, scales, leads, guard)
    low, high = plan['comparability_ratio']
    ready = low <= endpoint['local_gradient']['2_5_mm_s']['ratio_to_removed_free'] <= high
    write_json(out/'scale_audit.json', {'reference_endpoint': endpoint, 'fresh_initialization': start,
        'reference_endpoint_fit_gate': ready, 'reference_ratio_interval': [low, high], 'scopes': '49 train pairs only'})
    write_json(out/'increment_preflight.json', checks)
    return {'preflight_passed': ready, 'training_pairs': 49, 'local_scale_m_s': LOCAL_SCALE}


def run(plan, out, guard):
    audit_file = Path(plan['preflight']).with_name('scale_audit.json')
    audit = json.loads(increment.bound(plan, audit_file))
    preflight_result = json.loads(increment.bound(plan, plan['preflight']))
    if (plan['local_scale_m_s'] != LOCAL_SCALE or audit['reference_endpoint_fit_gate'] is not True
            or preflight_result['local_scale_m_s'] != LOCAL_SCALE
            or preflight_result['artifacts_sha256'][audit_file.name] != plan['files_sha256'][audit_file.as_posix()]):
        raise ValueError('rescaled local gradient preflight has not qualified')
    old = json.loads(increment.bound(plan, UNSCALED))
    path = UNSCALED.with_name('comparison.json')
    checkpoint = UNSCALED.with_name('velocity_response_leads_900.npz')
    if (old['status'] != 'completed' or old['updates'] != plan['updates'] or old['training_pairs'] != 49
            or old['anchor_draws_sha256'] != plan['anchor_draws_sha256']
            or any(old[key] != value for key, value in SCOPE.items())
            or any(old['artifacts_sha256'][p.name] != plan['files_sha256'][p.as_posix()] for p in (path, checkpoint))):
        raise ValueError('unscaled mixed reference is mismatched')
    before = json.loads(increment.bound(plan, path))
    increment.bound(plan, checkpoint)
    with np.load(ROOT/checkpoint, allow_pickle=False) as saved:
        baseline = increment.load_position_model(saved)
        norms = SimpleNamespace(known_mean=saved['known_mean'], known_scale=saved['known_scale'])
    held = read_held(json.loads(increment.bound(plan, plan['held_design'])))
    original_scores, _ = score_held(baseline, held, norms, lambda: guard('reference_25_mm_s'))
    if original_scores != before['held_timing']:
        raise ValueError('unscaled development responses did not reproduce')

    with patch.object(increment, 'LocalVelocityResponse', RescaledLocalVelocityResponse):
        result = increment.run(plan, out, guard)
    candidate = json.loads((out/'comparison.json').read_bytes())
    protection = regression(before['legacy_validation'], candidate['legacy_validation'])
    if len(protection['checks']) != 22:
        raise ValueError('expected 22 protections against the unscaled blended reference')
    candidate['legacy_protections']['local_25_mm_s'] = protection
    candidate['fixed_checks']['all_legacy_protections_vs_local_25_mm_s'] = protection['passed']
    candidate['reference_held_timing_local_25_mm_s'] = before['held_timing']
    candidate['velocity_half_difference_errors_vs_local_25_mm_s'] = velocity_errors(
        candidate['held_timing'], before['held_timing'])
    candidate['training_timed']['reference_local_25_mm_s'] = before['training_timed']['candidate_49']
    for key in ('training', 'development'):
        candidate['independent_one_step_increment'][key]['reference_local_25_mm_s'] = (
            before['independent_one_step_increment'][key]['candidate_49'])
    write_json(out/'comparison.json', candidate)
    return {**result, 'local_scale_m_s': LOCAL_SCALE, 'matched_25_mm_s_reference_reproduced': True,
            'fixed_checks': candidate['fixed_checks'], 'replacement_gate_passed': all(candidate['fixed_checks'].values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'fit'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'increment_scale', preflight if args.phase == 'preflight' else run)
