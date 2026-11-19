"""Run the original dense-velocity recipe on the accepted 49 training pairs."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import FIT, FIT_SHA, ROOT, SCOPE, branch_arrays, run_phase, write_json

import numpy as np

from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.model_data import KNOWN_NAMES
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.timed_pair_data import prepare_timed_pairs
from ball_retention.timing_coverage_fit import _add_pairs
from fit_s3_timed_fitability import pair_scores
from preflight_s3_velocity_response_leads import DESIGN, DESIGN_SHA256, digest, lead_vectors, prepare_comparison, weight_audit
from train_s3_position_integral import read_held, regression, report, score_held
from train_s3_velocity_response_leads import compare_endpoints, fit, velocity_errors


COVERAGE_DESIGN = 'configs/s3_advance_coverage_fit_plan.json'
COVERAGE_DESIGN_SHA = 'c97a194139c2931e79bb7418cd29e3eb68f06a44ce3bd68439bcac14cb0c9164'
LEADS = (1, 8, 16, 24, 32)
TRAIN_GATE = {'min_cosine': .8, 'max_relative_error': .5}


def read(path, sha):
    return json.loads(_verified_bytes(ROOT, path, sha))


def prepare_candidate(protocol, ref, guard):
    for name, sha in protocol['inputs_sha256'].items():
        _verified_bytes(ROOT, name, sha)
        guard()
    hashes = protocol['inputs_sha256']
    capture = protocol['supplement_acceptance']
    manifest = protocol['supplement_manifest']
    accepted = read(capture, hashes[capture])
    if (accepted['status'] != 'completed'
            or (accepted['worker_starts'], accepted['snapshots'], accepted['branches'],
                accepted['physics_steps'], accepted['pairs'], accepted['training_updates']) != (1, 1, 8, 6784, 3, 0)
            or accepted['test_episode_files_read'] is not False or accepted['control_accepted'] is not False
            or accepted['artifacts_sha256'][Path(manifest).name] != hashes[manifest]):
        raise ValueError('supplement is not the accepted advance capture')
    extra = prepare_timed_pairs(ROOT, manifest, hashes[manifest])
    entry, = extra.entries
    identity = {'group': entry['group_id'], 'episode': entry['episode_id'], 'tick': entry['snapshot_tick'],
                'directions': list(extra.directions)}
    if identity != protocol['additional_pair_ids'] or entry['history_id'] != 'advance' or entry['split'] != 'train':
        raise ValueError('supplement identity changed')
    aug, weights = _add_pairs(ref, extra)
    audit = read(protocol['static_data_audit'], hashes[protocol['static_data_audit']])
    changed = np.flatnonzero(aug.pair_weights[:46] != ref.pair_weights).tolist()
    if ({'reference': len(ref.pairs.initial), 'candidate': len(aug.pairs.initial)} != protocol['training_pair_counts']
            or any(weights[k] != audit[k] for k in weights) or changed != audit['changed_reference_rows']
            or audit['added_rows'] != [46, 47, 48] or audit['anchor_draws_sha256'] != protocol['anchor_draws_sha256']):
        raise ValueError('49-row fold differs from static data audit')
    np.testing.assert_array_equal(aug.pair_weights, audit['candidate_pair_weights'])
    for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
        np.testing.assert_array_equal(getattr(aug, key), getattr(ref, key))
    return aug, weights


def score_training(model, fold, rows, selected):
    initial, features, targets, masks = branch_arrays(fold)
    initial, features, targets = initial[selected], features[:, selected], targets[:, selected]
    if not masks[:, selected].all():
        raise ValueError('incomplete timed training branch')
    packed = features.reshape(-1, *features.shape[2:])
    prediction = model.denormalize(model.rollout(model.normalize(np.tile(initial, (2, 1))), packed)).reshape(targets.shape)
    return pair_scores(prediction, targets, initial, features, model, [rows[i] for i in selected], LEADS, TRAIN_GATE)


def run(plan, out, guard):
    protocol = read(COVERAGE_DESIGN, COVERAGE_DESIGN_SHA)
    design = read(DESIGN, DESIGN_SHA256)
    baseline_run = read(FIT, FIT_SHA)
    if (protocol['reference_design'] != DESIGN or protocol['reference_run'] != FIT.as_posix()
            or plan['updates'] != protocol['candidate_updates'] or plan['seed'] != protocol['seed']
            or plan['budget'] != protocol['budget'] or plan['scope'] != {key: protocol[key] for key in SCOPE}
            or plan['seed'] != design['seed'] or plan['updates'] != design['updates']
            or baseline_run['status'] != 'completed' or baseline_run['updates'] != plan['updates']):
        raise ValueError('advance execution differs from the pinned 46/49 recipe')
    parent, cfg, scales, reference, legacy, anchors, _, draws, _ = prepare_comparison(design, guard)
    if draws != protocol['anchor_draws_sha256'] or draws != baseline_run['anchor_draws_sha256']:
        raise ValueError('archived S2 draw stream mismatch')
    fold, weights = prepare_candidate(protocol, reference, guard)
    if set(fold.pairs.group_ids) & set(legacy.validation.batch.group_ids):
        raise ValueError('development group leaked into the training data')
    ck = Path(protocol['reference_checkpoint'])
    _verified_bytes(ROOT, ck, baseline_run['checkpoint_sha256'])
    with np.load(ROOT/ck, allow_pickle=False) as saved:
        baseline = load_position_model(saved)
        if type(baseline) is not TrapezoidPositionResidualMLP or int(saved['updates']) != 900 or baseline.dt_s != anchors.dt_s:
            raise ValueError('unexpected 46-pair incumbent model')
        for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold, key))
        np.testing.assert_array_equal(saved['known_names'], KNOWN_NAMES)

    def read_comparison(source, accepted):
        path = Path(source).with_name('comparison.json')
        return read(path, accepted['artifacts_sha256'][path.name])

    frozen = read_comparison(FIT, baseline_run)
    trapezoid = read_comparison(design['parent_run'],
                                read(design['parent_run'], design['inputs_sha256'][design['parent_run']]))
    integral = read(design['parent_design'], design['inputs_sha256'][design['parent_design']])
    original_path = integral['parent_run']
    original = read_comparison(original_path, read(original_path, integral['inputs_sha256'][original_path]))
    history = cfg['history_plan']
    histories = {row['episode_id']: row['history_id'] for row in read(history, cfg['inputs_sha256'][history])['episodes']}
    held = read_held(parent)

    def evaluate(model):
        timed, forecasts = score_held(model, held, fold, lambda: guard('development'))
        legacy_score = report(model, fold.known_mean, fold.known_scale, legacy, anchors, histories)
        legacy_score['pairs'] = [row for row in legacy_score['pairs'] if row['split'] == 'validation']
        return timed, legacy_score, forecasts

    reference_timed, reference_legacy, _ = evaluate(baseline)
    if reference_timed != frozen['held_timing'] or reference_legacy != frozen['legacy_validation']:
        raise ValueError('46-pair development scores did not exactly reproduce')
    leads = lead_vectors(design, fold.pairs.plus_target.shape[1])
    components = np.asarray(cfg['components'])
    tolerance = design['static_checks']['weight_atol']
    _, before_weights = weight_audit(*leads, components, reference.pairs.noise_mask, reference.pair_weights, tolerance)
    _, after_weights = weight_audit(*leads, components, fold.pairs.noise_mask, fold.pair_weights, tolerance)
    for key in ('candidate_response_mass', 'candidate_velocity_mass', 'candidate_velocity_weight_fraction_by_lead'):
        np.testing.assert_allclose(before_weights[key], after_weights[key], rtol=0., atol=tolerance)
    write_json(out/'data_preflight.json', {'weight_binding': weights, 'reference_lead_weights': before_weights,
        'candidate_lead_weights': after_weights, 'reference_development_scores_exact': True, 'training_updates': 0})

    manifest = parent['timed_manifest']
    reverse_entries = read(manifest, parent['inputs_sha256'][manifest])['entries']
    directions = [(entry['episode_id'], pair['direction_id']) for entry in reverse_entries for pair in entry['pairs']]
    advance = protocol['additional_pair_ids']
    directions.extend((advance['episode'], name) for name in advance['directions'])
    np.testing.assert_array_equal([episode for episode, _ in directions], fold.pairs.episode_ids[40:])
    rows = {i: {'row': i, 'episode': episode, 'group': str(fold.pairs.group_ids[i]),
                'tick': int(fold.pairs.snapshot_ticks[i]), 'history': histories[episode], 'offset_pattern': name}
            for i, (episode, name) in enumerate(directions, start=40)}
    cohorts = {'reverse': list(range(40, 46)), 'advance': list(range(46, 49))}
    if any(rows[i]['history'] != history for history, indices in cohorts.items() for i in indices):
        raise ValueError('timed rows do not have their expected train history')
    before = {key: score_training(baseline, fold, rows, indices) for key, indices in cohorts.items()}
    if before['reverse']['h32_velocity_agrees_count'] != 0:
        raise ValueError('incumbent training response gate did not reproduce')
    fresh = TrapezoidPositionResidualMLP(len(KNOWN_NAMES), fold.state_mean, fold.state_scale, anchors.dt_s,
        plan['seed'], hidden=cfg['network'][1], residual_gain=np.asarray(scales['residual_gain_si'])/fold.state_scale)

    def track(name, update, *, record=True):
        guard(name, update, log=record)
        if not all(np.isfinite(value).all() for value in fresh.params.values()):
            raise FloatingPointError(f'nonfinite parameter after update {update}')

    path, model, actual_draws = fit(fresh, fold, cfg, scales, design, leads, out, track)
    timed, legacy_score, forecasts = evaluate(model)
    comparison = compare_endpoints(timed, trapezoid['held_timing'], legacy_score, original, trapezoid)
    comparison['velocity_half_difference_errors_vs_dense_46'] = velocity_errors(timed, reference_timed)
    comparison['reference_held_timing_dense_46'] = reference_timed
    guard('protections', log=True)
    matched = regression(reference_legacy, legacy_score)
    if len(matched['checks']) != 22:
        raise ValueError('expected 22 matched protections')
    comparison['legacy_protections']['dense_46'] = matched
    comparison['fixed_checks']['all_legacy_protections_vs_dense_46'] = matched['passed']
    comparison['training_timed'] = {'reference_46': before,
        'candidate_49': {key: score_training(model, fold, rows, indices) for key, indices in cohorts.items()}}
    write_json(out/'comparison.json', comparison)
    write_json(out/'timed_rollouts.json', {**forecasts, 'checkpoint_sha256': digest(path)})
    return {'anchor_draws_sha256': actual_draws, 'training_pairs': len(fold.pairs.initial), 'reference_pairs': 46,
            'checkpoint_sha256': digest(path), 'reference_reproduced': True, 'checkpoint_selection': False,
            'fixed_checks': comparison['fixed_checks'], 'replacement_gate_passed': all(comparison['fixed_checks'].values()),
            'main_model_replaced': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'P3', run)
