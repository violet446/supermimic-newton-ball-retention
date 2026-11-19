"""Verify the fixed velocity response weighting without optimizer updates."""
import argparse
from io import BytesIO
import json
from math import fsum
import os
from pathlib import Path
import sys
import time

from preflight_s3_position_integral import ROOT, digest, dormant_check, prepare_recipe, write_json

import numpy as np

from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES
from ball_retention.pair_loss import paired_loss
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.resources import directory_bytes
from ball_retention.velocity_response_loss import VELOCITY, paired_velocity_response_loss, response_components


DESIGN = 'configs/s3_velocity_response_lead_fit_plan.json'
DESIGN_SHA256 = 'd32dae1e74e5479cd9579b5e2f91184790f54f103f7bfa08a33d2f0e4dc42470'
OUT = ROOT/'runs/s3_velocity_response_leads_preflight_20261115'
SCOPE = {'training_updates': 0, 'simulation_steps': 0, 'test_episode_files_read': False}


def prepare_comparison(plan, guard):
    inputs = {name: _verified_bytes(ROOT, name, sha) for name, sha in plan['inputs_sha256'].items()}
    design = json.loads(inputs[plan['parent_design']])
    _, parent, cfg, scales, fold, legacy, anchors, draws = prepare_recipe(design, guard)
    source = json.loads(inputs[plan['parent_run']])
    diagnosis = json.loads(inputs[plan['diagnosis']])
    fixture = Path(plan['matched_reference'])
    if (source['status'] != 'completed' or source['updates'] != plan['updates']
            or source['test_episode_files_read'] is not False or source['simulation_steps'] != 0
            or source['anchor_draws_sha256'] != draws or draws != plan['anchor_draws_sha256']
            or source['checkpoint_sha256'] != plan['inputs_sha256'][fixture.as_posix()]
            or source['artifacts_sha256'][fixture.name] != source['checkpoint_sha256']
            or diagnosis['status'] != 'completed' or any(diagnosis[key] != value for key, value in SCOPE.items())
            or plan['seed'] != design['seed'] or plan['updates'] != design['updates']
            or plan['model_type'] != design['model_type'] or len(fold.pairs.initial) != plan['pair_count']
            or plan['nominal_gate_reference'] != design['matched_reference']
            or plan['legacy_protection_reference'] != design['legacy_protection_reference']
            or cfg['pairs']['lead_steps'] != plan['base_leads'] or cfg['pairs']['lead_weights'] != plan['base_lead_weights']):
        raise ValueError('the matched checkpoint, training data or weights differ from the frozen design')
    if any(plan['inputs_sha256'][(Path(plan['diagnosis']).parent/name).as_posix()] != sha
           for name, sha in diagnosis['artifacts_sha256'].items()):
        raise ValueError('pair-state diagnostic artifacts differ from their acceptance')
    for name, sha in diagnosis['source_sha256'].items():
        _verified_bytes(ROOT, name, sha)

    with np.load(BytesIO(inputs[fixture.as_posix()]), allow_pickle=False) as saved:
        model = load_position_model(saved)
        if (type(model) is not TrapezoidPositionResidualMLP or int(saved['updates']) != plan['updates']
                or tuple(saved['known_names']) != KNOWN_NAMES or model.dt_s != anchors.dt_s):
            raise ValueError('matched checkpoint does not represent the frozen trapezoid recipe')
        for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold, key))
        for key in ('residual_gain_si', 'state_error_scale_si'):
            np.testing.assert_array_equal(saved[key], scales[key])
        np.testing.assert_array_equal(model.residual_gain, np.asarray(scales['residual_gain_si'])/fold.state_scale)
    return parent, cfg, scales, fold, legacy, anchors, model, draws, diagnosis['source_sha256']


def lead_vectors(plan, horizon):
    base, velocity = np.zeros(horizon), np.zeros(horizon)
    base[np.asarray(plan['base_leads'])-1] = plan['base_lead_weights']
    for row in plan['velocity_response_leads']:
        lo, hi = row['first'], row['last']
        if not 1 <= lo <= hi <= horizon or np.any(velocity[lo-1:hi]):
            raise ValueError('velocity lead ranges overlap or exceed the horizon')
        velocity[lo-1:hi] = row['total']/(hi-lo+1)
    if horizon != 32:
        raise ValueError('velocity weighting is frozen to H32')
    np.testing.assert_array_equal(velocity[:31], .5/31)
    np.testing.assert_equal(velocity[31], .5)
    return base, velocity


def weight_audit(base, velocity, components, seen, samples, atol):
    original = base[:, None]*components
    updated = response_components(base, velocity, components, seen, samples)
    eligible = seen[..., None]*samples[:, None, None]
    old, new = eligible*original, eligible*updated
    np.testing.assert_array_equal(new[~seen], 0.)
    for sl in (slice(0, 3), slice(6, 9)):
        np.testing.assert_array_equal(new[..., sl], old[..., sl])
    # Sum each differently distributed component accurately before comparing mass.
    def component_mass(tensor):
        return [fsum(column.flat) for column in np.moveaxis(tensor, -1, 0)]

    np.testing.assert_allclose(component_mass(new), component_mass(old), rtol=0., atol=atol)
    np.testing.assert_allclose(new.sum(), old.sum(), rtol=0., atol=atol)
    new_velocity = new[..., VELOCITY].sum(axis=(0, 2))
    old_velocity = old[..., VELOCITY].sum(axis=(0, 2))
    candidate_fraction = new_velocity/new_velocity.sum()
    expected = velocity*np.sum(seen*samples[:, None], axis=0)
    np.testing.assert_allclose(candidate_fraction, expected/expected.sum(), rtol=0., atol=atol)
    return updated, {'baseline_velocity_mass': float(old_velocity.sum()), 'candidate_velocity_mass': float(new_velocity.sum()),
                     'baseline_response_mass': float(old.sum()), 'candidate_response_mass': float(new.sum()),
                     'baseline_h32_velocity_weight_fraction': float(old_velocity[-1]/old_velocity.sum()),
                     'candidate_h32_velocity_weight_fraction': float(candidate_fraction[-1]),
                     'candidate_velocity_weight_fraction_by_lead': candidate_fraction.tolist(),
                     'eligible_pairs_by_lead': seen.sum(axis=0).tolist(),
                     'candidate_nonzero_velocity_leads': (np.flatnonzero(new_velocity > 0)+1).tolist()}


def parameter_check(model, evaluate, gradients, plan, guard):
    checks = plan['static_checks']
    rng = np.random.default_rng(plan['seed'])
    step = checks['gradient_step']
    results = {}
    for name, parameter in model.params.items():
        original, direction = parameter.copy(), rng.normal(size=parameter.shape)
        direction /= np.linalg.norm(direction)
        try:
            parameter[...] = original+step*direction
            plus = evaluate(gradients=False)[0]['loss']
            parameter[...] = original-step*direction
            minus = evaluate(gradients=False)[0]['loss']
        finally:
            parameter[...] = original
        analytic = float(np.sum(gradients[name]*direction))
        centered = (plus-minus)/(2*step)
        np.testing.assert_allclose(analytic, centered, rtol=checks['gradient_rtol'], atol=checks['gradient_atol'], err_msg=name)
        results[name] = {'analytic': analytic, 'centered': centered, 'abs_error': abs(analytic-centered)}
        guard()
    return results


def check_loss(plan, cfg, scales, fold, model, guard):
    batch = fold.pairs
    known = (np.stack((batch.plus_known, batch.minus_known))-fold.known_mean)/fold.known_scale
    initial = model.normalize(batch.initial)
    targets = np.stack((batch.plus_target, batch.minus_target))
    masks = np.stack((batch.plus_mask, batch.minus_mask))
    base, velocity = lead_vectors(plan, known.shape[2])
    components = np.asarray(cfg['components'])
    units = np.asarray(cfg['response_scales_si'])
    atol = plan['static_checks']['weight_atol']
    weights, audit = weight_audit(base, velocity, components, batch.noise_mask, fold.pair_weights, atol)
    if audit['candidate_nonzero_velocity_leads'] != list(range(1, 33)):
        raise ValueError('the frozen fold has no eligible velocity supervision at a requested lead')
    partial = batch.noise_mask.copy()
    partial[::2, -1] = False
    partial[1::3, 0] = False
    _, partial_audit = weight_audit(base, velocity, components, partial, fold.pair_weights, atol)
    if partial_audit['candidate_h32_velocity_weight_fraction'] >= audit['candidate_h32_velocity_weight_fraction']:
        raise ValueError('partial-mask check did not reduce H32 mass as expected')

    args = (model, initial, known, targets, masks, batch.response_mask, batch.noise_mask, batch.response_floor,
            base, components, units, cfg['pairs']['absolute_weight'], cfg['pairs']['response_weight'])
    options = {'sample_weights': fold.pair_weights, 'absolute_scale_si': np.asarray(scales['state_error_scale_si'])}
    original, grads = paired_loss(*args, **options)
    for equal in (None, base.copy()):
        metrics, equivalent = paired_velocity_response_loss(*args, velocity_lead_weights=equal, **options)
        if metrics != original:
            raise ValueError('default or explicit original velocity weights changed legacy metrics')
        for name in grads:
            np.testing.assert_array_equal(equivalent[name], grads[name])

    def evaluate(*, gradients=True):
        return paired_velocity_response_loss(*args, velocity_lead_weights=velocity, gradients=gradients, **options)

    metrics, gradient = evaluate()
    np.testing.assert_equal(metrics['absolute'], original['absolute'])
    dormant_check(gradient)
    if not all(np.isfinite(value).all() for value in gradient.values()):
        raise FloatingPointError('non-finite velocity-weighted gradient')
    # Independently construct the robust response score from SI signed spreads.
    forecast = model.denormalize(model.rollout(np.tile(initial, (2, 1)), known.reshape(-1, *known.shape[2:]))).reshape(targets.shape)
    spread = .5*(forecast[0]-forecast[1])
    error = np.where(batch.response_mask, spread-.5*(targets[0]-targets[1]),
                     np.sign(spread)*np.maximum(0., abs(spread)-batch.response_floor))
    score = 2*(np.hypot(1., error/units)-1.)
    effective = batch.noise_mask[..., None]*fold.pair_weights[:, None, None]*weights
    np.testing.assert_allclose(metrics['response'], np.sum(effective*score)/effective.sum(), rtol=0., atol=atol)
    guard()
    differences = parameter_check(model, evaluate, gradient, plan, guard)
    return {'default_and_equal_weights_exact': True, 'absolute_state_loss_exact': True,
            'independent_response_score_matches': True, 'h32_weights': audit,
            'partial_mask_control_h32_fraction': partial_audit['candidate_h32_velocity_weight_fraction'],
            'matched_checkpoint_losses': {'original': original, 'candidate_weights': metrics},
            'gradient_by_parameter_block': differences}


def run(out):
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    try:
        plan = json.loads(_verified_bytes(ROOT, DESIGN, DESIGN_SHA256))
        budget = plan['static_budget']

        def guard():
            memory = process_memory(os.getpid())
            if (time.monotonic()-started >= budget['max_wall_s']
                    or memory['peak_rss_bytes'] > budget['max_process_rss_gib']*1024**3
                    or memory['peak_private_commit_bytes'] > budget['max_process_commit_gib']*1024**3
                    or directory_bytes(out) > budget['max_new_output_mib']*1024**2):
                raise RuntimeError('velocity response preflight exceeded a resource cap')
            return memory

        guard()
        _, cfg, scales, fold, _, _, model, draws, sources = prepare_comparison(plan, guard)
        checks = check_loss(plan, cfg, scales, fold, model, guard)
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline loss check imported physics')
        local = (Path(__file__).resolve(), ROOT/'src/ball_retention/velocity_response_loss.py',
                 ROOT/'scripts/preflight_s3_position_integral.py')
        write_json(out/'preflight.json', {'status': 'passed', 'design_sha256': DESIGN_SHA256, 'limits': budget,
                   'inputs_sha256': plan['inputs_sha256'], 'source_sha256': {**sources, **{
                       p.relative_to(ROOT).as_posix(): digest(p) for p in local}},
                   'pair_count': len(fold.pairs.initial), 'anchor_draws_sha256': draws, 'checks': checks,
                   'gradient_fixture': plan['matched_reference'], **SCOPE,
                   'wall_s': time.monotonic()-started, **guard()})
        guard()
    except BaseException as error:
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error), **SCOPE,
                   'wall_s': time.monotonic()-started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    target = parser.parse_args().output.resolve()
    if len(target.relative_to(ROOT/'runs').parts) != 1:
        raise ValueError('choose a new run directory immediately under runs')
    run(target)
