"""Fit one matched world-increment candidate with both auxiliary windows."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np

import train_s3_world_increment as matched
from audit_s3_world_increment_late_coverage import evaluate_snapshot
from audit_s3_world_increment_supervision_drift import CHECKPOINT, ENDPOINTS, load_endpoint
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool
from ball_retention.model_data import prepare_dataset
from ball_retention.pair_collection import prepare_collection
from ball_retention.world_increment_loss import world_increment_loss
from ball_retention.world_increment_model import WorldIncrementMLP
from preflight_s3_world_increment_late_gradient import masks_and_pairs, rms
from preflight_s3_world_increment_supervision import SETTINGS, add, examples, mul, norm
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held, regression, report
from train_s3_world_increment_supervision import losses, nominal_score


PHASE = 'world_increment_extended_window_fit'


def window_metrics(model, windows, records):
    summary = {}
    for key in ('early', 'late'):
        args = windows[key]
        _, _, err = world_increment_loss(model, *args, paired=True)
        w = args[3]
        summary[key] = {'weighted_head_rmse_m_s': rms(err, w),
            'pairs': {f'{episode}/{action}': rms(err[i], w[i]/w[i].sum())
                      for i, (episode, action) in enumerate(records)}}
    return summary


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    protocol = read(plan['window_protocol'])
    preflight = read(protocol['accepted_window_gradient_preflight'])
    gradients_path = Path(protocol['accepted_window_gradient_detail'])
    gradients = read(gradients_path)
    params = protocol['auxiliary']
    coefficients = params['coefficients']
    for key, match_key in (('budget', 'budget'), ('scope', 'scope'), ('seed', 'seed'), ('updates', 'fixed_updates')):
        if plan[key] != protocol[match_key]:
            raise ValueError(f'frozen match changed: {key}')
    if (preflight['status'] != 'completed' or preflight['updates'] != 0
            or not preflight['gradient_decomposition_and_fd_verified']
            or preflight['artifacts_sha256'][gradients_path.name] != plan['files_sha256'][gradients_path.as_posix()]
            or not all(gradients['inherited_static_checks']['extended'].values())
            or gradients['sampler_sha256'] != protocol['sampler_sha256']
            or coefficients != {'nominal': gradients['fixed_coefficients']['nominal'],
                                 'half_difference': gradients['fixed_coefficients']['extended']}
            or params['window_weights'] != [{'first_lead': 1, 'last_lead': 16, 'mass': .5},
                                            {'first_lead': 17, 'last_lead': 32, 'mass': .5}]
            or (params['nominal_scale_m_s'], params['half_difference_scale_m_s'], params['noise_floor_m_s'],
                params['noise_multiplier']) != (SETTINGS['nominal_scale_m_s'], SETTINGS['pair_scale_m_s'],
                                                 SETTINGS['floor_m_s'], SETTINGS['noise_multiplier'])):
        raise ValueError('unexpected extended-window recipe or preflight')
    base = read(protocol['reference_early_fit_plan'])
    for key in ('budget', 'scope', 'seed', 'updates', 'held_design', 'coverage_design', 'accepted_preflight',
                'timed_manifests', 'comparison_protocol', 'original_scores', 'reference_46_scores'):
        if plan[key] != base[key]:
            raise ValueError(f'matched training input changed: {key}')
    for key, field in (('reference', 'reference_unsupervised_checkpoint'), ('supervised', 'reference_early_checkpoint')):
        if protocol[field] != (ENDPOINTS[key]/CHECKPOINT).as_posix():
            raise ValueError(f'unexpected comparison endpoint: {key}')
    audit = read(protocol['accepted_window_label_audit'])
    early_preflight = Path(read(base['supervision_protocol'])['accepted_preflight'])
    archived = read(early_preflight.with_name('supervision.json'))
    held_design = read(plan['held_design'])
    held = read_held(held_design)
    scores = {k: read(path/'comparison.json') for k, path in ENDPOINTS.items()}
    original = matched.objectives
    context = {}

    def objectives(model, fold, cfg, scales, leads, model_cfg, anchor):
        evaluate, initial, features = original(model, fold, cfg, scales, leads, model_cfg, anchor)
        sampled, coverage, counts = examples(plan, fold, guard)
        if (coverage != archived['coverage'] or counts != archived['coverage_by_snapshot']
                or sampled['nominal'][3].tolist() != archived['nominal_weights']
                or sampled['half_difference'][3].tolist() != archived['half_difference_weights']):
            raise ValueError('reference supervision binding changed')
        windows, records = masks_and_pairs(sampled['half_difference'], fold, coverage, audit['training'])
        np.testing.assert_allclose(windows['extended'][3], .5*windows['early'][3]+.5*windows['late'][3],
                                   atol=1e-15, rtol=0.)
        for name in ('early', 'late'):
            if np.count_nonzero(windows[name][3]) != 144:
                raise ValueError(f'{name} resolved label count changed')
        data = {'nominal': sampled['nominal'], 'half_difference': windows['extended']}
        endpoints = {key: load_endpoint(plan, key, held, anchor, guard) for key in ENDPOINTS}
        legacy = prepare_collection(ROOT, ROOT/held_design['base_collection'])
        s2 = prepare_dataset((ROOT/held_design['s2_manifest']).parent, horizons=tuple(cfg['anchors']['horizons']),
                             verify_test=False)
        histories = {r['episode_id']: r['history_id'] for r in read(cfg['history_plan'])['episodes']}
        for key, (before, _, _, _, sampler) in endpoints.items():
            evaluation = report(matched.LegacyEvaluation(before, model_cfg, anchor, fold),
                                fold.known_mean, fold.known_scale, legacy, s2, histories)
            evaluation['pairs'] = [r for r in evaluation['pairs'] if r['split'] == 'validation']
            if sampler != protocol['sampler_sha256'] or evaluation != scores[key]['legacy_validation']:
                raise ValueError(f'frozen legacy scoring or sampler changed: {key}')
        for key, fixed in {'initialization': model, **{k: v[0] for k, v in endpoints.items()}}.items():
            values = losses(fixed, data)
            for term, expected_name in (('nominal', 'nominal'), ('half_difference', 'extended')):
                expected = gradients['checkpoints'][key]['terms'][expected_name]
                loss, grad, _ = values[term]
                np.testing.assert_allclose((loss, norm(grad)), (expected['loss'], expected['gradient_norm']),
                                           atol=1e-13, rtol=0.)
        context.update(windows=windows, records=records, endpoints=endpoints, fold=fold, anchor=anchor)
        write_json(out/'window_binding.json', {'signed_archive_labels_verified': True,
            'window_masses': {key: float(windows[key][3].sum()*.5) for key in ('early', 'late')},
            'window_resolved_counts': {key: int(np.count_nonzero(windows[key][3])) for key in ('early', 'late')},
            'coefficients': coefficients, 'nominal_weights_exact': True, 'references_reproduced': True,
            'initialization_and_both_endpoint_gradients_reproduced': True})

        def combined(h, take):
            total_anchor, pair, original_grad = evaluate(h, take)
            auxiliary = losses(model, data)
            pair_total = pair['loss']+sum(coefficients[k]*v[0] for k, v in auxiliary.items())
            gradient = add(original_grad, *(mul(v[1], coefficients[k]) for k, v in auxiliary.items()))
            return total_anchor, {**pair, 'loss': pair_total, 'original_pair_loss': pair['loss'],
                                  'auxiliary': {k: v[0] for k, v in auxiliary.items()}}, gradient

        return combined, initial, features

    with patch.object(matched, 'objectives', objectives):
        result = matched.run(plan, out, guard)
    if result['sampler_sha256'] != protocol['sampler_sha256']:
        raise ValueError('trained sampler changed')
    with np.load(out/CHECKPOINT, allow_pickle=False) as saved:
        fitted = WorldIncrementMLP.from_state_dict(saved)
    scores['extended'] = json.loads((out/'comparison.json').read_bytes())
    protection = {tag: regression(scores[tag]['legacy_validation'], scores['extended']['legacy_validation'])
                  for tag in ENDPOINTS}
    if any(len(item['checks']) != 22 for item in protection.values()):
        raise ValueError('expected 22 protections per matched comparison')
    endpoints = {**context['endpoints'], 'extended': (fitted, context['fold'])}
    train = {key: window_metrics(x[0], context['windows'], context['records']) for key, x in endpoints.items()}
    for key, name in (('early', 'H1-H16'), ('late', 'H17-H32')):
        for ep, action in context['records']:
            for ref in ENDPOINTS:
                observed = train[ref][key]['pairs'][f'{ep}/{action}']
                expected = audit['training'][ep][action]['models'][ref][name]['head_rmse_m_s']
                np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=0.)
    capture, cfg, fields, traces, _ = held
    dev_windows = evaluate_snapshot('development', _parse_state(fields), cfg, traces, capture['directions'],
                                    endpoints, scores, load_tool(ROOT/'assets/shovel.STL'), capture['gates'])
    nominal = {tag: nominal_score(x[0], held, context['fold'], context['anchor']) for tag, x in endpoints.items()}
    write_json(out/'window_comparison.json', {'reference_keys': {'reference': 'no auxiliary', 'supervised': 'H1-H16'},
        'training_independent_windows': train, 'development_nominal': nominal,
        'development_independent_windows': {action: row['models'] for action, row in dev_windows.items()},
        'legacy_protections_matched': protection})
    return {**result, 'extended_window_preflight_reproduced': True,
            'replacement_gate_passed': result['replacement_gate_passed'] and all(x['passed'] for x in protection.values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
