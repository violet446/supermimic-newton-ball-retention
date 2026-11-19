"""Fit the 46-pair recipe with a redistribution of response weight only."""
import argparse
from hashlib import sha256
import json

from s3_fitability_common import FIT, FIT_SHA, ROOT, branch_arrays, run_phase, write_json

import numpy as np

from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.model_data import KNOWN_NAMES
from ball_retention.pair_loss import paired_loss
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.velocity_response_loss import paired_velocity_response_loss, response_components
from ball_retention.world_model import Adam
from fit_s3_timed_fitability import pair_scores
from preflight_s3_velocity_response_leads import (
    DESIGN, DESIGN_SHA256, digest, dormant_check, lead_vectors, parameter_check, prepare_comparison,
)
from train_s3_position_integral import read_held, regression, report, score_held
from train_s3_timing_coverage import primary
from train_s3_timing_lead_weights import checks_for
from train_s3_velocity_response_leads import velocity_errors


def response_weights(leads, components, noise_mask, samples, timed, share):
    base, velocity = leads
    component_weights = response_components(base, velocity, components, noise_mask, samples)
    masked = noise_mask[..., None]*component_weights
    before = masked*samples[:, None, None]
    total, timed_mass = before.sum(), before[timed].sum()
    if not (0 < timed_mass < total and timed_mass/total < share < 1):
        raise ValueError('expected a positive shift of response mass to the timed group')
    factors = np.where(timed, share*total/timed_mass, (1-share)*total/(total-timed_mass))
    reweighted = samples*factors
    after = masked*reweighted[:, None, None]
    np.testing.assert_allclose(after.sum(), total, rtol=0., atol=1e-14)
    np.testing.assert_allclose(after[timed].sum()/total, share, rtol=0., atol=1e-14)
    # The frozen cohort has the same eligibility by lead in both groups.
    np.testing.assert_allclose(after.sum(axis=0), before.sum(axis=0), rtol=0., atol=1e-14)
    audit = {'total_before': float(total), 'total_after': float(after.sum()),
             'timed_share_before': float(timed_mass/total), 'timed_share_after': float(after[timed].sum()/total),
             'held_factor': float(factors[~timed][0]), 'timed_factor': float(factors[timed][0]),
             'h32_velocity_timed_share_before': float(before[timed, -1, 3:6].sum()/before[:, -1, 3:6].sum()),
             'h32_velocity_timed_share_after': float(after[timed, -1, 3:6].sum()/after[:, -1, 3:6].sum()),
             'max_lead_component_mass_delta': float(abs(after.sum(axis=0)-before.sum(axis=0)).max())}
    return reweighted, component_weights, audit


def evaluate_pairs(model, fold, cfg, scales, arrays, base, response_components_by_lead, response_samples,
                   *, gradients=True, absolute_weight=None, response_weight=None):
    initial, features, truth, masks = arrays
    batch = fold.pairs
    branches = (model, model.normalize(initial), features, truth, masks,
                batch.response_mask, batch.noise_mask, batch.response_floor)
    units = np.asarray(cfg['response_scales_si'])
    absolute_units = np.asarray(scales['state_error_scale_si'])
    absolute_coeff = cfg['pairs']['absolute_weight'] if absolute_weight is None else absolute_weight
    response_coeff = cfg['pairs']['response_weight'] if response_weight is None else response_weight
    state, state_grad = paired_loss(*branches, base, np.asarray(cfg['components']), units, absolute_coeff, 0.,
        gradients=gradients, sample_weights=fold.pair_weights, absolute_scale_si=absolute_units)
    response, response_grad = paired_loss(*branches, np.ones(len(base)), response_components_by_lead, units,
        0., response_coeff, gradients=gradients, sample_weights=response_samples, absolute_scale_si=absolute_units)
    metrics = {'absolute': state['absolute'], 'response': response['response'], 'loss': state['loss']+response['loss']}
    if not np.isfinite(metrics['loss']):
        raise FloatingPointError('non-finite reweighted pair loss')
    return metrics, ({key: grad+response_grad[key] for key, grad in state_grad.items()} if gradients else None)


def check_gradients(model, fold, cfg, scales, arrays, leads, components_by_lead, adjusted, design, guard):
    initial, features, truth, masks = arrays
    batch = fold.pairs
    base, velocity = leads
    a, r = (cfg['pairs'][key] for key in ('absolute_weight', 'response_weight'))
    original, original_grad = paired_velocity_response_loss(
        model, model.normalize(initial), features, truth, masks, batch.response_mask, batch.noise_mask,
        batch.response_floor, base, np.asarray(cfg['components']), np.asarray(cfg['response_scales_si']), a, r,
        velocity_lead_weights=velocity, sample_weights=fold.pair_weights,
        absolute_scale_si=np.asarray(scales['state_error_scale_si']))

    def evaluate(samples=adjusted, **options):
        return evaluate_pairs(model, fold, cfg, scales, arrays, base, components_by_lead, samples, **options)

    matched, matched_grad = evaluate(fold.pair_weights)
    if matched != original:
        raise ValueError('unchanged pair loss did not exactly reproduce original')
    for key in original_grad:
        np.testing.assert_array_equal(matched_grad[key], original_grad[key])
    changed, changed_grad = evaluate()
    state, state_grad = evaluate(response_weight=0.)
    control_state, control_state_grad = evaluate(fold.pair_weights, response_weight=0.)
    if changed['absolute'] != original['absolute'] or state['loss'] != control_state['loss']:
        raise ValueError('absolute state loss changed under reweighting')
    for key in state_grad:
        np.testing.assert_array_equal(state_grad[key], control_state_grad[key])
    difference = float(np.linalg.norm(np.concatenate([(changed_grad[k]-original_grad[k]).ravel() for k in changed_grad])))
    if not np.isfinite(difference) or difference <= 0:
        raise ValueError('expected finite nonzero response gradient change')
    dormant_check(changed_grad)
    derivatives = parameter_check(model, lambda **options: evaluate(**options), changed_grad, design, guard)
    return {'original_loss_and_gradient_exact': True, 'absolute_loss_and_gradient_exact': True,
            'original_pair_loss': original, 'changed_pair_loss': changed, 'gradient_delta_norm': difference,
            'parameter_finite_differences': derivatives}


def fit(model, fold, cfg, scales, plan, arrays, base, components_by_lead, response_samples, draws, out, guard):
    anchors = {h: (model.normalize(w.initial), (w.known-fold.known_mean)/fold.known_scale, model.normalize(w.target))
               for h, w in fold.anchors.items()}
    units = np.asarray(scales['state_error_scale_si'])
    components = np.asarray(cfg['components'])
    adam = cfg['optimizer']
    if adam['name'] != 'Adam':
        raise ValueError('expected Adam')
    optimizer = Adam(model.params, learning_rate=adam['learning_rate'], max_grad_norm=adam['max_grad_norm'])
    rng = np.random.default_rng(plan['seed'])
    sample_hash = sha256()
    horizons = cfg['anchors']['horizons']
    guard('fit', 0, log=True)
    with (out/'training.jsonl').open('x', encoding='utf-8') as log:
        for step in range(plan['updates']):
            horizon = horizons[step % len(horizons)]
            x, features, target = anchors[horizon]
            take = rng.integers(len(x), size=cfg['anchors']['batch_size'])
            sample_hash.update(np.r_[horizon, take].astype('<i8').tobytes())
            anchor_loss, grads = model.loss_and_grad(x[take], features[take], target[take],
                                                    state_weights=components, state_error_scale_si=units)
            pair, pair_grads = evaluate_pairs(model, fold, cfg, scales, arrays, base, components_by_lead, response_samples)
            norm = optimizer.step({key: grads[key]+pair_grads[key] for key in grads})
            if not all(np.isfinite(value).all() for value in model.params.values()):
                raise FloatingPointError(f'nonfinite parameter at update {optimizer.iteration}')
            periodic = optimizer.iteration % 30 == 0
            guard('fit', optimizer.iteration, log=periodic)
            if periodic:
                log.write(json.dumps({'update': optimizer.iteration, 'anchor_loss': anchor_loss, 'pair': pair,
                                      'gradient_norm': norm}, allow_nan=False)+'\n')
    if optimizer.iteration != plan['updates'] or sample_hash.hexdigest() != draws:
        raise ValueError('fixed endpoint or S2 anchor draw stream changed')
    dormant_check(model.params)
    path = out/'timed_response_weight_900.npz'
    np.savez_compressed(path, **model.state_dict(), known_mean=fold.known_mean, known_scale=fold.known_scale,
                        known_names=np.asarray(KNOWN_NAMES), state_error_scale_si=units,
                        residual_gain_si=np.asarray(scales['residual_gain_si']), updates=np.asarray(optimizer.iteration))
    with np.load(path, allow_pickle=False) as saved:
        restored = load_position_model(saved)
    initial, known, _, _ = arrays
    np.testing.assert_array_equal(restored.rollout(restored.normalize(initial), known[0]),
                                  model.rollout(model.normalize(initial), known[0]))
    return path, restored


def training_summary(model, arrays, rows, selected, gate):
    initial, known, truth, mask = arrays
    inputs, features, target = initial[selected], known[:, selected], truth[:, selected]
    if not mask[:, selected].all():
        raise ValueError('incomplete timed training branch')
    packed = features.reshape(-1, *features.shape[2:])
    predicted = model.denormalize(model.rollout(model.normalize(np.tile(inputs, (2, 1))), packed)).reshape(target.shape)
    return pair_scores(predicted, target, inputs, features, model, [rows[i] for i in selected], [1, 8, 16, 24, 32], gate)


def run(plan, out, guard):
    design = json.loads(_verified_bytes(ROOT, DESIGN, DESIGN_SHA256))
    parent, cfg, scales, fold, legacy, anchors, _, draws, _ = prepare_comparison(design, guard)
    accepted = json.loads(_verified_bytes(ROOT, FIT, FIT_SHA))
    reference_file = FIT.parent/'comparison.json'
    reference = json.loads(_verified_bytes(ROOT, reference_file, accepted['artifacts_sha256'][reference_file.name]))
    checkpoint_file = FIT.parent/'velocity_response_leads_900.npz'
    with np.load(ROOT/checkpoint_file, allow_pickle=False) as saved:
        reference_model = load_position_model(saved)
        for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold, key))
        np.testing.assert_array_equal(saved['known_names'], KNOWN_NAMES)
        if (int(saved['updates']) != 900 or reference_model.dt_s != anchors.dt_s
                or type(reference_model) is not TrapezoidPositionResidualMLP
                or digest(ROOT/checkpoint_file) != accepted['checkpoint_sha256']):
            raise ValueError('unexpected incumbent checkpoint')
    if not (plan['updates'] == design['updates'] == 900 and draws == design['anchor_draws_sha256']
            == accepted['anchor_draws_sha256'] and plan['seed'] == design['seed']):
        raise ValueError('training endpoint or S2 draws differ from incumbent')
    p0 = ROOT/plan['pair_identity_source']
    rows = json.loads(p0.read_text(encoding='utf-8'))['pair_identity']
    selected = [row['row'] for row in rows if row['timed']]
    if selected != plan['timed_rows'] or len(rows) != 46 or len(fold.pairs.initial) != 46:
        raise ValueError('expected frozen six timed rows and 46 total')
    for key, attribute in (('episode', 'episode_ids'), ('group', 'group_ids'), ('tick', 'snapshot_ticks')):
        np.testing.assert_array_equal([row[key] for row in rows], getattr(fold.pairs, attribute))
    np.testing.assert_array_equal([row['pair_weight'] for row in rows], fold.pair_weights)
    timed_mask = np.isin(np.arange(len(rows)), selected)
    arrays = branch_arrays(fold)
    leads = lead_vectors(design, 32)
    samples, components_by_lead, mass = response_weights(leads, np.asarray(cfg['components']),
        fold.pairs.noise_mask, fold.pair_weights, timed_mask, plan['timed_response_share'])
    np.testing.assert_allclose(mass['timed_share_before'], plan['baseline_timed_response_share'], rtol=0., atol=1e-14)
    guard('gradient_preflight', log=True)
    checks = check_gradients(reference_model, fold, cfg, scales, arrays, leads, components_by_lead, samples, design, guard)
    write_json(out/'weight_preflight.json', {'response_mass': mass, **checks})

    history = cfg['history_plan']
    histories = {row['episode_id']: row['history_id'] for row in
                 json.loads(_verified_bytes(ROOT, history, cfg['inputs_sha256'][history]))['episodes']}
    held = read_held(parent)

    def score(model):
        timed, _ = score_held(model, held, fold, lambda: guard('development'))
        legacy_score = report(model, fold.known_mean, fold.known_scale, legacy, anchors, histories)
        legacy_score['pairs'] = [row for row in legacy_score['pairs'] if row['split'] == 'validation']
        return timed, legacy_score

    reference_timed, reference_legacy = score(reference_model)
    if reference_timed != reference['held_timing'] or reference_legacy != reference['legacy_validation']:
        raise ValueError('incumbent development scores did not exactly reproduce')
    before = training_summary(reference_model, arrays, rows, selected, plan['training_gate'])
    np.testing.assert_allclose(before['half_difference_rmse_h32_si']['velocity'], plan['baseline_timed_velocity_rmse_m_s'],
                               rtol=0., atol=1e-13)
    if before['h32_velocity_agrees_count'] != 0:
        raise ValueError('original training response gate did not reproduce')

    fresh = TrapezoidPositionResidualMLP(len(KNOWN_NAMES), fold.state_mean, fold.state_scale, anchors.dt_s,
        plan['seed'], hidden=cfg['network'][1], residual_gain=np.asarray(scales['residual_gain_si'])/fold.state_scale)
    path, model = fit(fresh, fold, cfg, scales, plan, arrays, leads[0], components_by_lead, samples, draws, out, guard)
    after = training_summary(model, arrays, rows, selected, plan['training_gate'])
    timed, legacy_score = score(model)
    protections = {'dense_velocity': regression(reference_legacy, legacy_score)}
    original_reference = plan['legacy_source']
    original = json.loads(_verified_bytes(ROOT, original_reference, plan['files_sha256'][original_reference]))
    for name in ('held_offsets', 'held_offsets_plus_timing'):
        protections[name] = regression(original['legacy_development'][name], legacy_score)
    trapezoid_reference = plan['trapezoid_source']
    trapezoid = json.loads(_verified_bytes(ROOT, trapezoid_reference, plan['files_sha256'][trapezoid_reference]))
    protections['position_integral'] = regression(trapezoid['legacy_validation'], legacy_score)
    if any(len(protection['checks']) != 22 for protection in protections.values()):
        raise ValueError('expected 22 protections for each reference')
    positions = primary(timed)
    nominal_ratio = (positions['pitch_pulse']['absolute_model_nominal_error_norm']/
                     original['nominal_h32_position_m']['held_offsets_plus_timing'])
    replacement_checks = checks_for(positions, nominal_ratio, protections)
    replacement_checks['all_legacy_protections_vs_dense_velocity'] = protections['dense_velocity']['passed']
    errors = velocity_errors(timed, reference_timed)
    write_json(out/'comparison.json', {'training': {'initial_reference': before, 'candidate': after},
        'development': {'reference': reference_timed, 'candidate': timed, 'velocity_half_difference_errors': errors,
                        'nominal_ratio_vs_original_46': nominal_ratio, 'legacy_validation': legacy_score},
        'protections': protections, 'replacement_checks': replacement_checks})
    return {'anchor_draws_sha256': draws, 'reference_development_reproduced': True,
            'checkpoint_sha256': digest(path), 'training_pairs': 46, 'timed_pairs': len(selected),
            'timed_h32_agrees_count': after['h32_velocity_agrees_count'],
            'replacement_checks': replacement_checks, 'replacement_gate_passed': all(replacement_checks.values()),
            'main_model_replaced': False, 'checkpoint_selection': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(ROOT/'configs/s3_timed_response_weight_plan.json', args.plan_sha256, 'P2', run)
