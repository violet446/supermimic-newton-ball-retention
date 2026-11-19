"""Decompose the frozen local-increment objective at three parameter points."""
import argparse
from copy import copy
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

from s3_fitability_common import ROOT, SCOPE, branch_arrays, run_phase, write_json

import numpy as np
import train_s3_advance_coverage as coverage
from ball_retention.pair_loss import paired_loss
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.velocity_response_loss import VELOCITY, response_components
from diagnose_s3_timed_fitability import cosine, flatten
from train_s3_local_increment_scale import LOCAL_SCALE, RescaledLocalVelocityResponse
from train_s3_local_velocity_increment import bound, incumbent


SCALED = Path('runs/s3_local_increment_scale_fit_20261116/validation.json')
BANDS = ((1, 7), (8, 16), (17, 24), (25, 31), (32, 32))


def squared_norm(value):
    return float(value@value)


def summarize(vector, local, full):
    size = np.linalg.norm(vector)
    full_norm = np.linalg.norm(full)
    if not (np.isfinite(size) and np.isfinite(full_norm) and full_norm > 0. and squared_norm(local) > 0.):
        raise FloatingPointError('invalid gradient norm')
    return {'norm': float(size), 'cosine_to_local': cosine(vector, local),
            'cosine_to_full': cosine(vector, full), 'norm_ratio_to_full': float(size/full_norm),
            'signed_projection_share_of_local': float(vector@local/squared_norm(local)),
            'signed_projection_share_of_full': float(vector@full/squared_norm(full))}


def anchor_gradient(model, fold, cfg, units, components, plan, guard):
    windows = {h: (model.normalize(row.initial), (row.known-fold.known_mean)/fold.known_scale, model.normalize(row.target))
               for h, row in fold.anchors.items()}
    horizons = cfg['anchors']['horizons']
    rng = np.random.default_rng(plan['seed'])
    log_hash = sha256()
    gradients = {h: [] for h in horizons}
    for i in range(plan['s2_draw_count']):
        h = horizons[i % len(horizons)]
        initial, known, target = windows[h]
        take = rng.integers(len(initial), size=cfg['anchors']['batch_size'])
        log_hash.update(np.r_[h, take].astype('<i8').tobytes())
        _, grad = model.loss_and_grad(initial[take], known[take], target[take],
                                     state_weights=components, state_error_scale_si=units)
        gradients[h].append(flatten(grad))
        if (i+1) % 60 == 0:
            guard('S2')
    if log_hash.hexdigest() != plan['anchor_draws_sha256']:
        raise ValueError('S2 sampling stream differs from the fixed training plan')
    return {h: np.stack(items) for h, items in gradients.items()}


def local_partitions(model, known, setup, units, weight, groups, local, full):
    _, _, prediction = setup.estimate(model, known)
    residual = np.where(setup.resolved, prediction-setup.target,
                        np.sign(prediction)*np.maximum(abs(prediction)-setup.floor, 0.))/LOCAL_SCALE
    parts, data = {}, {}
    for label, rows in groups.items():
        for first, last in BANDS:
            partition = copy(setup)
            mask = np.zeros_like(setup.eligible)
            mask[rows, first-1:last] = True
            partition.local_weights = setup.local_weights*mask[..., None]
            weights = partition.local_weights
            part_mass = weights.sum()
            if part_mass <= 0:
                raise ValueError('local partition lacks positive effective weights')
            _, grad = partition.local_loss(model, known, units, gradients=True)
            vec = weight*flatten(grad)
            key = (label, first, last)
            parts[key] = vec
            data[key] = {'weight_fraction': float(part_mass/setup.local_weights.sum()),
                         'resolved_fraction': float(weights[setup.resolved].sum()/part_mass),
                         'robust_tail_fraction': float(weights[abs(residual) > 1.].sum()/part_mass)}
    total = sum(parts.values())
    np.testing.assert_allclose(total, local, rtol=0., atol=5e-13)

    def reduce_part(keys):
        vec = sum(parts[key] for key in keys)
        mass = sum(data[key]['weight_fraction'] for key in keys)
        return {**summarize(vec, local, full), 'weight_fraction': mass,
                **{field: sum(data[key]['weight_fraction']*data[key][field] for key in keys)/mass
                   for field in ('resolved_fraction', 'robust_tail_fraction')}}

    return {'by_history': {group: reduce_part([key for key in parts if key[0] == group]) for group in groups},
            'by_lead_band': {f'{first}-{last}': reduce_part([key for key in parts if key[1:] == (first, last)])
                             for first, last in BANDS},
            'by_history_and_band': {group: {f'{first}-{last}': reduce_part([(group, first, last)])
                                           for first, last in BANDS} for group in groups},
            'max_gradient_closure': float(abs(total-local).max())}


def objective_audit(model, fold, cfg, scales, leads, groups, plan, guard):
    initial, known, truth, mask = branch_arrays(fold)
    batch = fold.pairs
    base, velocity = leads
    absolute_units = np.asarray(scales['state_error_scale_si'])
    units = np.asarray(cfg['response_scales_si'])
    components = np.asarray(cfg['components'])
    absolute_weight, response_weight = (cfg['pairs'][k] for k in ('absolute_weight', 'response_weight'))
    original = response_components(base, velocity, components, batch.noise_mask, fold.pair_weights)
    setup = RescaledLocalVelocityResponse(batch, original, fold.pair_weights, plan['transfer_fraction'],
                                        plan['increment_floor_m_s'])
    args = (model, model.normalize(initial), known, truth, mask, batch.response_mask, batch.noise_mask, batch.response_floor)
    options = {'sample_weights': fold.pair_weights, 'absolute_scale_si': absolute_units}
    _, state = paired_loss(*args, base, components, units, absolute_weight, 0., **options)
    _, free = paired_loss(*args, np.ones(len(base)), setup.free_components, units, 0.,
                         response_weight*setup.free_fraction, **options)
    _, local = setup.local_loss(model, known, units, gradients=True)
    _, paired = setup.loss(*args, base, components, units, absolute_weight, response_weight, **options)
    gradients = {'absolute': flatten(state), 'free_response': flatten(free), 'local': response_weight*flatten(local)}
    recomposed = sum(gradients.values())
    np.testing.assert_allclose(recomposed, flatten(paired), rtol=0., atol=5e-13)
    anchors = anchor_gradient(model, fold, cfg, absolute_units, components, plan, guard)
    s2 = np.concatenate(list(anchors.values()), axis=0)
    gradients['S2'] = s2.mean(axis=0)
    all_terms = sum(gradients.values())
    local = gradients['local']
    parts = local_partitions(model, known, setup, units, response_weight, groups, local, all_terms)
    return {'weighted_term_gradients': {key: summarize(value, local, all_terms)
                                        for key, value in gradients.items()},
            'full_gradient_norm': float(np.linalg.norm(all_terms)),
            'cosines': {f'{a}/{b}': cosine(gradients[a], gradients[b]) for a, b in combinations(gradients, 2)},
            'S2_by_horizon': {h: {'norm_of_mean': float(np.linalg.norm(items.mean(axis=0))),
                                  'cosine_of_mean_to_local': cosine(items.mean(axis=0), local),
                                  'batch_cosine_to_local_quantiles': np.quantile([cosine(g, local) for g in items],
                                                                                [0., .5, 1.]).tolist()}
                              for h, items in anchors.items()},
            'local_partitions': parts, 'max_paired_gradient_closure': float(abs(recomposed-flatten(paired)).max())}


def run(plan, out, guard):
    if (plan['updates'] != 0 or plan['s2_draw_count'] != 900 or plan['training_pairs'] != 49
            or plan['local_scale_m_s'] != LOCAL_SCALE or plan['transfer_fraction'] != .25
            or tuple(map(tuple, plan['bands'])) != BANDS):
        raise ValueError('unexpected gradient audit scope')
    design = coverage.read(coverage.DESIGN, coverage.DESIGN_SHA256)
    _, cfg, scales, ref, _, anchors, _, draws, _ = coverage.prepare_comparison(design, guard)
    if draws != plan['anchor_draws_sha256']:
        raise ValueError('frozen draws changed')
    protocol = coverage.read(coverage.COVERAGE_DESIGN, coverage.COVERAGE_DESIGN_SHA)
    fold, _ = coverage.prepare_candidate(protocol, ref, guard)
    before, _ = incumbent(plan, fold)
    result = json.loads(bound(plan, SCALED))
    checkpoint = SCALED.with_name('velocity_response_leads_900.npz')
    sha = plan['files_sha256'][checkpoint.as_posix()]
    if (result['status'] != 'completed' or result['updates'] != 900 or result['local_scale_m_s'] != LOCAL_SCALE
            or result['training_pairs'] != 49 or result['anchor_draws_sha256'] != draws
            or any(result[key] != value for key, value in SCOPE.items())
            or result['checkpoint_sha256'] != sha or result['artifacts_sha256'][checkpoint.name] != sha):
        raise ValueError('scaled model is outside the frozen scope')
    with np.load(ROOT/checkpoint, allow_pickle=False) as saved:
        after = load_position_model(saved)
        if type(after) is not TrapezoidPositionResidualMLP or int(saved['updates']) != 900:
            raise ValueError('unexpected candidate model type or checkpoint')
        for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold, key))
    fresh = TrapezoidPositionResidualMLP(before.known_dim, fold.state_mean, fold.state_scale, anchors.dt_s,
        plan['seed'], hidden=cfg['network'][1], residual_gain=np.asarray(scales['residual_gain_si'])/fold.state_scale)
    episode = fold.pairs.episode_ids
    np.testing.assert_array_equal(np.flatnonzero(fold.pairs.axes < 0), np.arange(40, 49))
    if not all(e.endswith('_reverse') for e in episode[40:46]) or not all(e.endswith('_advance') for e in episode[46:49]):
        raise ValueError('unexpected timed history assignment')
    groups = {'held': slice(0, 40), 'reverse': slice(40, 46), 'advance': slice(46, 49)}
    leads = coverage.lead_vectors(design, 32)
    reports = {}
    for label, model in (('fresh', fresh), ('original_49', before), ('scaled_49', after)):
        reports[label] = objective_audit(model, fold, cfg, scales, leads, groups, plan, guard)
        guard(label, log=True)
    feedback = json.loads(bound(plan, plan['development_timeline']))
    timing = {name: {model: {mode: {key: row['onset'][mode][key]
                                  for key in ('first_sustained_large_relative_error', 'first_sustained_opposition')}
                            for mode in ('free', 'local')} for model, row in models.items()}
              for name, models in feedback['cases']['development'].items()}
    write_json(out/'gradients.json', {'parameter_points': reports, 'bands': BANDS, 'pair_groups': [40, 6, 3],
        'archived_development_onset_46_49': timing, 'S2_explanation': 'Mean of all 900 frozen draws at each fixed point.'})
    return {'training_pairs': 49, 'parameter_points': len(reports), 's2_draws_per_point': plan['s2_draw_count']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'increment_gradient_audit', run)
