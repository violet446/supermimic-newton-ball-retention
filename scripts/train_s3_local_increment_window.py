"""Move a fixed share of the local-only H32 mass to H8–H24."""
import argparse
from copy import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from s3_fitability_common import ROOT, SCOPE, branch_arrays, run_phase, write_json

import numpy as np
import train_s3_advance_coverage as coverage
import train_s3_local_velocity_increment as increment
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP
from ball_retention.velocity_response_loss import response_components
from diagnose_s3_timed_fitability import cosine, flatten
from train_s3_local_increment_scale import LOCAL_SCALE, RescaledLocalVelocityResponse
from train_s3_position_integral import read_held, regression, score_held
from train_s3_velocity_response_leads import velocity_errors


REFERENCE = Path('runs/s3_local_increment_scale_fit_20261116/validation.json')
WINDOW = slice(7, 24)
LAST = 31
MASS_ATOL = 1e-14
GRADIENT_ATOL = 5e-13


class WindowedLocalVelocityResponse(RescaledLocalVelocityResponse):
    def __init__(self, batch, original_weights, samples, fraction, floor_m_s):
        super().__init__(batch, original_weights, samples, fraction, floor_m_s)
        if self.local_weights.shape[1] != LAST+1:
            raise ValueError('local time redistribution is defined for H32')
        if not fraction:
            return
        original = self.local_weights.copy()
        removed = .5*original[:, LAST]
        inside = original[:, WINDOW]
        totals = inside.sum(axis=0)
        if np.any(totals <= 0.):
            raise ValueError('a requested local step or component has zero eligible mass')
        per_lead = removed.sum(axis=0)/inside.shape[1]
        self.local_weights[:, WINDOW] += (inside/totals[None])*per_lead[None, None]
        self.local_weights[:, LAST] -= removed
        np.testing.assert_allclose(self.local_weights.sum(axis=1), original.sum(axis=1), rtol=0., atol=MASS_ATOL)
        total = self.local_weights.sum()
        np.testing.assert_allclose(self.local_weights[:, LAST].sum()/total, .25, rtol=0., atol=MASS_ATOL)
        lead_mass = self.local_weights.sum(axis=(0, 2))/total
        self.audit['max_lead_component_mass_delta_before_redistribution'] = self.audit.pop('max_lead_component_mass_delta')
        self.audit.update(h32_local_weight_fraction=float(lead_mass[LAST]), local_weight_fraction_by_lead=lead_mass.tolist(),
                          resolved_local_weight_fraction=float(self.local_weights[self.resolved].sum()/total),
                          max_pair_component_mass_delta=float(abs(self.local_weights.sum(axis=1)-original.sum(axis=1)).max()))


def scaled_reference(plan, fold=None):
    accepted = json.loads(increment.bound(plan, REFERENCE))
    ck = REFERENCE.with_name('velocity_response_leads_900.npz')
    result = REFERENCE.with_name('comparison.json')
    if (accepted['status'] != 'completed' or accepted['updates'] != 900 or accepted['training_pairs'] != 49
            or accepted['local_scale_m_s'] != LOCAL_SCALE
            or any(accepted[key] != value for key, value in SCOPE.items())
            or accepted['anchor_draws_sha256'] != plan['anchor_draws_sha256']
            or any(plan['files_sha256'][p.as_posix()] != accepted['artifacts_sha256'][p.name] for p in (ck, result))
            or accepted['checkpoint_sha256'] != plan['files_sha256'][ck.as_posix()]):
        raise ValueError('unexpected 2.5 mm/s endpoint')
    increment.bound(plan, ck)
    with np.load(ROOT/ck, allow_pickle=False) as saved:
        model = increment.load_position_model(saved)
        if type(model) is not TrapezoidPositionResidualMLP or int(saved['updates']) != 900:
            raise ValueError('unexpected reference architecture or update')
        if fold is not None:
            for key in ('known_mean', 'known_scale', 'state_mean', 'state_scale'):
                np.testing.assert_array_equal(saved[key], getattr(fold, key))
        norms = SimpleNamespace(known_mean=saved['known_mean'], known_scale=saved['known_scale'])
    return model, norms, json.loads(increment.bound(plan, result))


def check_window(plan, fold, cfg, scales, leads, model, guard):
    with patch.object(increment, 'LocalVelocityResponse', WindowedLocalVelocityResponse):
        after, checks = increment.prepare(plan, fold, cfg, scales, leads, model, guard)
    base, velocity = leads
    batch = fold.pairs
    units = np.asarray(cfg['response_scales_si'])
    components = np.asarray(cfg['components'])
    original = response_components(base, velocity, components, batch.noise_mask, fold.pair_weights)
    before = RescaledLocalVelocityResponse(batch, original, fold.pair_weights, plan['transfer_fraction'],
                                          plan['increment_floor_m_s'])
    a, r = (cfg['pairs'][key] for key in ('absolute_weight', 'response_weight'))
    for key in ('free_components', 'free_fraction', 'floor', 'eligible', 'resolved', 'prior', 'target'):
        np.testing.assert_array_equal(getattr(before, key), getattr(after, key))
    np.testing.assert_array_equal(after.local_weights[:, LAST], before.local_weights[:, LAST]*.5)
    unaffected = [*range(WINDOW.start), *range(WINDOW.stop, LAST)]
    np.testing.assert_array_equal(after.local_weights[:, unaffected], before.local_weights[:, unaffected])
    old_mass = before.local_weights.sum(axis=(0, 2))
    new_mass = after.local_weights.sum(axis=(0, 2))
    np.testing.assert_allclose(new_mass[WINDOW]-old_mass[WINDOW], .25*old_mass.sum()/17, rtol=0., atol=MASS_ATOL)
    initial, known, target, masks = branch_arrays(fold)

    def local_grad(loss, leads=slice(None)):
        subset = copy(loss)
        subset.local_weights = np.zeros_like(loss.local_weights)
        subset.local_weights[:, leads] = loss.local_weights[:, leads]
        return r*flatten(subset.local_loss(model, known, units, gradients=True)[1])

    old_local, new_local = map(local_grad, (before, after))
    old_window, new_window = (local_grad(loss, WINDOW) for loss in (before, after))
    old_tail, new_tail = (local_grad(loss, slice(LAST, LAST+1)) for loss in (before, after))
    np.testing.assert_allclose(new_tail, .5*old_tail, rtol=0., atol=GRADIENT_ATOL)
    # The experiment uses a complete 49-pair local window with equal original
    # effective mass at H8–H24, so each gradient there has the same multiplier.
    factor = 1.+.25*old_mass.sum()/(17*old_mass[WINDOW.start])
    np.testing.assert_allclose(new_window, factor*old_window, rtol=0., atol=GRADIENT_ATOL)
    np.testing.assert_allclose(new_local-old_local, (new_window-old_window)+(new_tail-old_tail),
                               rtol=0., atol=GRADIENT_ATOL)
    args = (model, model.normalize(initial), known, target, masks, batch.response_mask, batch.noise_mask,
            batch.response_floor, base, components, units, a, r)
    opts = {'sample_weights': fold.pair_weights, 'absolute_scale_si': np.asarray(scales['state_error_scale_si'])}
    old_full, new_full = (flatten(loss.loss(*args, **opts)[1]) for loss in (before, after))
    np.testing.assert_allclose(new_full-old_full, new_local-old_local, rtol=0., atol=GRADIENT_ATOL)
    n = lambda value: float(np.linalg.norm(value))
    checks['reference_one_step_increment'] = increment.training_local(model, fold, after)
    return checks, {'reference_h32_weight': float(old_mass[-1]/old_mass.sum()),
        'candidate_h32_weight': float(new_mass[-1]/new_mass.sum()),
        'reference_window_weight': float(old_mass[WINDOW].sum()/old_mass.sum()),
        'candidate_window_weight': float(new_mass[WINDOW].sum()/new_mass.sum()),
        'window_gradient_norm': [n(old_window), n(new_window)], 'window_gradient_factor': factor,
        'h32_gradient_norm': [n(old_tail), n(new_tail)],
        'window_cosine_to_local': [cosine(old_window, old_local), cosine(new_window, new_local)],
        'window_signed_share_of_local': [float(g@full/(full@full))
                                         for g, full in ((old_window, old_local), (new_window, new_local))],
        'local_gradient_norm': [n(old_local), n(new_local)], 'pair_gradient_delta_norm': n(new_full-old_full),
        'loss_gradient_closure_max_abs': float(abs(new_full-old_full-new_local+old_local).max())}


def preflight(plan, out, guard):
    if (plan['window'] != [8, 24] or plan['h32_share_before_after'] != [.5, .25] or plan['local_scale_m_s'] != LOCAL_SCALE
            or plan['updates'] != 0 or plan['training_pairs'] != 49):
        raise ValueError('unexpected local window scope')
    design = coverage.read(coverage.DESIGN, coverage.DESIGN_SHA256)
    _, cfg, scales, ref, _, _, _, draws, _ = coverage.prepare_comparison(design, guard)
    if draws != plan['anchor_draws_sha256']:
        raise ValueError('unexpected S2 draw stream')
    protocol = coverage.read(coverage.COVERAGE_DESIGN, coverage.COVERAGE_DESIGN_SHA)
    fold, _ = coverage.prepare_candidate(protocol, ref, guard)
    dense, _ = increment.incumbent(plan, fold)
    scaled, _, _ = scaled_reference(plan, fold)
    leads = coverage.lead_vectors(design, 32)
    checks, dense_change = check_window(plan, fold, cfg, scales, leads, dense, guard)
    _, scaled_change = check_window(plan, fold, cfg, scales, leads, scaled, guard)
    ready = all(0. < row['window_gradient_norm'][0] < row['window_gradient_norm'][1]
                for row in (dense_change, scaled_change))
    write_json(out/'window_audit.json', {'dense_49_endpoint': dense_change, 'scaled_49_endpoint': scaled_change,
                                        'window_gate': ready})
    write_json(out/'increment_preflight.json', checks)
    return {'preflight_passed': ready, 'training_pairs': 49, 'local_scale_m_s': LOCAL_SCALE}


def run(plan, out, guard):
    preflight_path = Path(plan['preflight'])
    report = json.loads(increment.bound(plan, preflight_path))
    audit_path = preflight_path.with_name('window_audit.json')
    audit = json.loads(increment.bound(plan, audit_path))
    if (plan['updates'] != 900 or plan['window'] != [8, 24] or plan['h32_share_before_after'] != [.5, .25]
            or plan['local_scale_m_s'] != LOCAL_SCALE or audit['window_gate'] is not True or report['preflight_passed'] is not True
            or report['artifacts_sha256'][audit_path.name] != plan['files_sha256'][audit_path.as_posix()]):
        raise ValueError('window preflight is outside the frozen fit scope')
    baseline, norms, prior = scaled_reference(plan)
    held = read_held(json.loads(increment.bound(plan, plan['held_design'])))
    score, _ = score_held(baseline, held, norms, lambda: guard('reference_2_5_mm_s'))
    if score != prior['held_timing']:
        raise ValueError('matched time-weight reference scores did not reproduce')
    with patch.object(increment, 'LocalVelocityResponse', WindowedLocalVelocityResponse):
        result = increment.run(plan, out, guard)
    candidate = json.loads((out/'comparison.json').read_bytes())
    protections = regression(prior['legacy_validation'], candidate['legacy_validation'])
    if len(protections['checks']) != 22:
        raise ValueError('expected 22 matched protections')
    candidate['legacy_protections']['matched_time_2_5_mm_s'] = protections
    candidate['fixed_checks']['all_legacy_protections_vs_matched_time_2_5_mm_s'] = protections['passed']
    candidate['reference_held_timing_matched_time_2_5_mm_s'] = prior['held_timing']
    candidate['velocity_half_difference_errors_vs_matched_time_2_5_mm_s'] = velocity_errors(
        candidate['held_timing'], prior['held_timing'])
    candidate['training_timed']['reference_matched_time_2_5_mm_s'] = prior['training_timed']['candidate_49']
    for key in ('training', 'development'):
        candidate['independent_one_step_increment'][key]['reference_matched_time_2_5_mm_s'] = (
            prior['independent_one_step_increment'][key]['candidate_49'])
    write_json(out/'comparison.json', candidate)
    return {**result, 'matched_time_2_5_mm_s_reference_reproduced': True, 'local_scale_m_s': LOCAL_SCALE,
            'fixed_checks': candidate['fixed_checks'], 'replacement_gate_passed': all(candidate['fixed_checks'].values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'fit'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'increment_window', preflight if args.phase == 'preflight' else run)
