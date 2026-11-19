"""Audit early and late signed increment gradients on frozen parameters."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, SCOPE, run_phase, write_json

import numpy as np

from audit_s3_world_increment_supervision_drift import CHECKPOINT, ENDPOINTS
from ball_retention.geometry import load_tool, support_point
from ball_retention.world_increment_loss import world_increment_loss
from ball_retention.world_increment_model import PACKED, WorldIncrementMLP
from preflight_s3_velocity_response_leads import lead_vectors
from preflight_s3_world_increment import reference_data
from preflight_s3_world_increment_supervision import (
    SETTINGS, add, dot, examples, finite_difference, mul, norm, original_gradient,
)
from train_s3_local_velocity_increment import bound


PHASE = 'world_increment_late_gradient'
EARLY, LATE = slice(0, 16), slice(16, 32)


def cosine(a, b):
    return float(dot(a, b)/(norm(a)*norm(b))) if norm(a)*norm(b) > 0. else None


def normalized_weights(raw):
    total = float(raw.sum())
    if total <= 0.:
        raise ValueError('empty supervision window')
    return raw/total


def rms(error, weights):
    return float(np.sqrt(np.sum(weights*np.sum(error**2, axis=-1))))


def masks_and_pairs(paired, fold, coverage, audited_training):
    states, known, target, early_weights, scale = paired
    records = []
    resolved = []
    for i, row in enumerate(coverage):
        episode, action = row['episode'], row['direction']
        audited = audited_training[episode][action]
        observed = (target[0, i]-target[1, i])*.5
        np.testing.assert_array_equal(observed, audited['target_m_s'])
        valid = np.linalg.norm(observed, axis=-1) >= np.asarray(audited['floor_m_s'])
        np.testing.assert_array_equal(valid, audited['resolved'])
        resolved.append(valid)
        records.append((episode, action))
    raw = np.asarray(resolved)*fold.pair_weights[40:, None]
    early = raw.copy()
    early[:, LATE] = 0.
    np.testing.assert_array_equal(normalized_weights(early), early_weights)
    late = raw.copy()
    late[:, EARLY] = 0.
    if not np.array_equal(early.sum(axis=1), late.sum(axis=1)):
        raise ValueError('early and late windows no longer have equal per-pair mass')
    return {key: (states, known, target, w, scale) for key, w in
            (('early', early_weights), ('late', normalized_weights(late)), ('extended', normalized_weights(raw)))}, records


def attribution(model, args, records, full_loss, full):
    states, known, target, weights, scale = args
    parts = []
    for i in range(len(records)):
        weight = weights[i]
        mass = float(weight.sum())
        value, gradient, _ = world_increment_loss(model, states[:, i], known[:, i], target[:, i],
                                                   weight/mass, scale, paired=True)
        parts.append((mass, mass*value, mul(gradient, mass)))
    np.testing.assert_allclose(sum(v for _, v, _ in parts), full_loss, atol=1e-13, rtol=1e-13)
    total = add(*(g for _, _, g in parts))
    for k in full:
        np.testing.assert_allclose(total[k], full[k], atol=1e-13, rtol=1e-13)
    parts_norm = sum(norm(g) for _, _, g in parts)
    full_norm = norm(full)

    def row(indices):
        subset = [parts[i] for i in indices]
        gradient = add(*(g for _, _, g in subset))
        remainder = add(full, mul(gradient, -1.))
        return {'sample_weight_mass': sum(w for w, _, _ in subset),
                'loss_fraction': sum(v for _, v, _ in subset)/full_loss,
                'share_of_pair_gradient_norms': sum(norm(g) for _, _, g in subset)/parts_norm,
                'projection_fraction_of_total_gradient': dot(gradient, full)/full_norm**2,
                'gradient_norm_over_full': norm(gradient)/full_norm,
                'full_vs_remaining_cosine': cosine(full, remainder)}

    return {
        'pairs': {f'{ep}/{action}': row([i]) for i, (ep, action) in enumerate(records)},
        'snapshots': {ep: row([i for i, x in enumerate(records) if x[0] == ep])
                      for ep in dict.fromkeys(x[0] for x in records)},
        'actions': {action: row([i for i, x in enumerate(records) if x[1] == action])
                    for action in dict.fromkeys(x[1] for x in records)},
    }


def trial(gradients, coefficients):
    original, nominal = gradients['original'], gradients['nominal']
    report = {}
    for window in ('early', 'late', 'extended'):
        pair = gradients[window]
        extra = add(mul(nominal, coefficients['nominal']), mul(pair, coefficients[window]))
        full = add(original, extra)
        report[window] = {'auxiliary_norm_fraction': norm(extra)/norm(original),
                          'combined_direction_cosine': {k: cosine(full, g) for k, g in gradients.items()}}
    return report


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    if plan['updates'] != 0 or plan['settings'] != SETTINGS:
        raise ValueError('unexpected probe scope')
    protocol, cfg, scales, fold, held, draws = reference_data(plan, guard)
    dense = read(read(plan['coverage_design'])['reference_design'])
    leads = lead_vectors(dense, 32)
    model_cfg = held[1]
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), model_cfg['initial']['anchor_xy_tool_m'])[0]
    frozen = read(plan['accepted_supervision_preflight'])
    scored = read(plan['accepted_supervision_result'])
    old_coverage = read(plan['accepted_late_coverage'])
    dataset, coverage, counts = examples(plan, fold, guard)
    if (coverage != frozen['coverage'] or counts != frozen['coverage_by_snapshot']
            or dataset['nominal'][3].tolist() != frozen['nominal_weights']
            or dataset['half_difference'][3].tolist() != frozen['half_difference_weights']):
        raise ValueError('frozen auxiliary data or effective weights changed')
    pairs, records = masks_and_pairs(dataset['half_difference'], fold, coverage, old_coverage['training'])
    models = {}
    for key, path in ENDPOINTS.items():
        status = read(path/'validation.json')
        ck = path/CHECKPOINT
        if (status['status'] != 'completed' or status['updates'] != 900 or status['training_pairs'] != 49
                or status['sampler_sha256'] != draws or any(status[k] != v for k, v in SCOPE.items())
                or status['artifacts_sha256'][ck.name] != plan['files_sha256'][ck.as_posix()]):
            raise ValueError(f'unexpected endpoint: {key}')
        with np.load(ROOT/ck, allow_pickle=False) as saved:
            for k in ('known_mean', 'known_scale', 'state_mean', 'state_scale'):
                np.testing.assert_array_equal(saved[k], getattr(fold, k))
            models[key] = WorldIncrementMLP.from_state_dict(saved)
    reference = models['reference']
    models['initialization'] = WorldIncrementMLP(
        PACKED, reference.state_mean, reference.state_scale, reference.dt_s, plan['seed'],
        inc_center=protocol['world_increment_center_m_s'], hidden=reference.hidden, residual_gain=reference.residual_gain)
    checkpoints, gradients = {}, {}
    rng = np.random.default_rng(plan['seed'])
    for key, model in models.items():
        base, base_losses = original_gradient(model, fold, cfg, scales, leads, model_cfg, anchor, guard, draws)
        terms = {'original': add(*base.values())}
        metrics = {}
        for name, args in {'nominal': dataset['nominal'], **pairs}.items():
            paired = name != 'nominal'
            value, gradient, error = world_increment_loss(model, *args, paired=paired)
            terms[name] = gradient
            metrics[name] = {'loss': value, 'rms_error_m_s': rms(error, args[3]), 'gradient_norm': norm(gradient),
                'fd_error': finite_difference(model, 'half_difference' if paired else 'nominal', args, gradient, rng),
                'cosine_to_original_parts': {part: cosine(gradient, other) for part, other in base.items()}}
            if paired:
                metrics[name]['attribution'] = attribution(model, args, records, value, gradient)
        half_gradient = mul(add(terms['early'], terms['late']), .5)
        for name, array in half_gradient.items():
            np.testing.assert_allclose(terms['extended'][name], array, atol=1e-13, rtol=1e-13)
        if key in ('reference', 'initialization'):
            old = frozen['checkpoints']['endpoint' if key == 'reference' else key]
            for a, b in (('nominal', 'nominal'), ('early', 'half_difference')):
                np.testing.assert_allclose(metrics[a]['loss'], old['auxiliary'][b]['loss'], atol=1e-14, rtol=0.)
                np.testing.assert_allclose(norm(terms[a]), old['gradient_norms'][b], atol=1e-13, rtol=0.)
            np.testing.assert_allclose(norm(terms['original']), old['gradient_norms']['baseline'], atol=1e-13, rtol=0.)
        else:
            np.testing.assert_allclose(metrics['early']['rms_error_m_s'],
                                       scored['new_train_loss']['half_difference']['rms_error_m_s'], atol=1e-14, rtol=0.)
        gradients[key] = terms
        checkpoints[key] = {'original_losses': base_losses, 'original_gradient_norm': norm(terms['original']),
            'terms': metrics, 'direction_cosines': {a: {b: cosine(terms[a], terms[b]) for b in terms if a != b}
                                                    for a in terms}}
        guard(key, log=True)

    anchor_terms = gradients['reference']
    target = SETTINGS['endpoint_aux_norm_fraction']*norm(anchor_terms['original'])
    coefficients = {'nominal': frozen['coefficients']['nominal'], 'early': frozen['coefficients']['half_difference'],
                    'late': target/norm(anchor_terms['late']), 'extended': target/norm(anchor_terms['extended'])}
    np.testing.assert_allclose(coefficients['early'], target/norm(anchor_terms['early']), atol=1e-13, rtol=0.)
    trials = {key: trial(value, coefficients) for key, value in gradients.items()}
    literal = {key: trial(value, {**coefficients, 'extended': coefficients['early']})['extended']
               for key, value in gradients.items()}
    decisions = {key: {
        'initial_auxiliary_fraction_within_original_cap': trials['initialization'][key]['auxiliary_norm_fraction']
                                                         <= SETTINGS['max_init_aux_norm_fraction'],
        'reference_joint_descent': all(trials['reference'][key]['combined_direction_cosine'][k] > 0.
                                       for k in ('original', 'nominal', key)),
    } for key in ('early', 'late', 'extended')}
    write_json(out/'late_gradient.json', {'checkpoints': checkpoints, 'fixed_coefficients': coefficients,
        'matched_endpoint_norm_trials': trials, 'extended_at_existing_coefficient': literal,
        'inherited_static_checks': decisions, 'sampler_sha256': draws})
    return {'fixed_parameter_sets': len(models), 'exact_training_pairs': len(records),
            'old_loss_and_sampler_reproduced': True, 'gradient_decomposition_and_fd_verified': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
