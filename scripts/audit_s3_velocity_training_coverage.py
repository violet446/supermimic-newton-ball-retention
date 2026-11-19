"""Audit the frozen training-pair responses and cross-group coverage offline."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

from preflight_s3_velocity_response_leads import DESIGN, DESIGN_SHA256, ROOT, lead_vectors, prepare_comparison, write_json

import numpy as np

from check_s3_actions import digest
from ball_retention.action_eval import _direction
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES
from ball_retention.pair_data import DEFAULT_RESPONSE_FLOOR
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.resources import directory_bytes
from ball_retention.timed_pair_data import prepare_timed_pairs
from ball_retention.velocity_response_loss import VELOCITY, response_components


OUT = ROOT/'runs/s3_velocity_training_coverage_20261115'
FIT = Path('runs/s3_velocity_response_leads_fit_20261115/validation.json')
FIT_SHA = 'eb21441489c9a122093c8c6a32b35d66a783f2acf2ee2eee1771a5d5a8a66983'
LEADS = (8, 16, 24, 32)
LIMITS = {'wall_s': 120, 'rss_gib': 1, 'private_commit_gib': 1.5, 'output_mib': 16, 'cpu_threads': 1}
SCOPE = {'training_updates': 0, 'simulation_steps': 0, 'test_episode_files_read': False}


def metadata(fold, legacy, timed, history):
    batch = fold.pairs
    old = legacy.train.batch
    boundary = len(old.initial)
    timed_group = {entry['episode_id']: entry['history_id'] for entry in timed.entries}
    histories = {entry['episode_id']: entry['history_id'] for entry in history['episodes'] if entry['split'] == 'train'}
    if any(histories.get(episode) != name for episode, name in timed_group.items()):
        raise ValueError('timed history differs from the frozen training-only provenance')
    for key in ('initial', 'episode_ids', 'group_ids', 'snapshot_ticks', 'plus_known', 'minus_known', 'axes'):
        np.testing.assert_array_equal(getattr(batch, key), np.concatenate((getattr(old, key), getattr(timed.batch, key))))
    requests = [KNOWN_NAMES.index(name) for name in ('roll_request', 'pitch_request')]
    offsets = (batch.plus_known[..., requests]-batch.minus_known[..., requests])*.5
    np.testing.assert_allclose(offsets[boundary:], timed.amplitudes[:, None, None]*timed.direction_actions,
                               atol=1e-15, rtol=0.)
    amplitudes = np.r_[legacy.train.amplitudes, timed.amplitudes]
    rows = []
    for i, (episode, group, tick, axis, amplitude) in enumerate(zip(
            batch.episode_ids, batch.group_ids, batch.snapshot_ticks, batch.axes, amplitudes, strict=True)):
        label = ('held_roll', 'held_pitch')[axis] if i < boundary else timed.directions[i-boundary]
        rows.append({'row': i, 'episode': str(episode), 'group': str(group), 'tick': int(tick),
                     'history': histories.get(str(episode), 'original'), 'offset_pattern': label,
                     'amplitude': float(amplitude), 'timed': i >= boundary, 'pair_weight': float(fold.pair_weights[i])})
    return rows, offsets


def velocity_report(fold, models, cfg, design, rows):
    batch = fold.pairs
    normalized = (np.concatenate((batch.plus_known, batch.minus_known))-fold.known_mean)/fold.known_scale
    truth = .5*(batch.plus_target[..., VELOCITY]-batch.minus_target[..., VELOCITY])
    replay = .5*(batch.baseline_first[..., VELOCITY]-batch.baseline_second[..., VELOCITY])
    base, dense = lead_vectors(design, truth.shape[1])
    components = np.asarray(cfg['components'])
    response_weights = {}
    fitted = {}
    initial = np.concatenate((batch.initial, batch.initial))
    for (name, model), velocity in zip(models.items(), (base, dense), strict=True):
        predicted = model.denormalize(model.rollout(model.normalize(initial), normalized))
        plus, minus = np.split(predicted[..., VELOCITY], 2)
        fit = (plus-minus)*.5
        if not np.isfinite(fit).all():
            raise FloatingPointError('non-finite training rollout')
        fitted[name] = fit
        weight = response_components(base, velocity, components, batch.noise_mask, fold.pair_weights)
        effective = batch.noise_mask[..., None]*weight*fold.pair_weights[:, None, None]
        response_weights[name] = effective[..., VELOCITY]/effective.sum()

    for i, row in enumerate(rows):
        metrics = {}
        for lead in LEADS:
            j = lead-1
            if not batch.noise_mask[i, j]:
                metrics[str(lead)] = {'eligible': False}
                continue
            target = truth[i, j]
            norm = float(np.linalg.norm(target))
            threshold = max(DEFAULT_RESPONSE_FLOOR[1], 5.*np.linalg.norm(replay[i, j]),
                            32*np.finfo(float).eps*max(1., norm))
            strong = bool(norm >= 10.*threshold)
            scores = {}
            for name, fit in fitted.items():
                prediction = fit[i, j]
                cosine, direction = _direction(prediction, target, np.linalg.norm(prediction) >= threshold,
                                             norm >= threshold, ('model', 'newton'))
                error = float(np.linalg.norm(prediction-target))
                relative = error/norm if norm >= threshold else None
                weights = response_weights[name][i, j]
                resolved = batch.response_mask[i, j, VELOCITY]
                scores[name] = {'error_m_s': error, 'model_half_difference_m_s': prediction.tolist(),
                                'cosine': cosine, 'direction': direction, 'relative_error': relative,
                                'diagnostic_agreement': (cosine is not None and cosine >= .8 and relative <= .5) if strong else None,
                                'velocity_loss_mass': float(weights.sum()), 'resolved_target_loss_mass': float(weights[resolved].sum())}
            metrics[str(lead)] = {'eligible': True, 'newton_half_difference_m_s': target.tolist(),
                                  'newton_norm_m_s': norm, 'newton_strong': strong, 'floor_m_s': float(threshold), 'models': scores}
        row['by_lead'] = metrics

    sections = {'all': rows, 'timed': [row for row in rows if row['timed']]}
    for field in ('group', 'history', 'offset_pattern'):
        for value in sorted({row[field] for row in rows}):
            sections[f'{field}/{value}'] = [row for row in rows if row[field] == value]
    summary = {}
    for label, entries in sections.items():
        summary[label] = {}
        for lead in map(str, LEADS):
            valid = [(row['pair_weight'], row['by_lead'][lead]) for row in entries if row['by_lead'][lead]['eligible']]
            strong = [row for _, row in valid if row['newton_strong']]
            total = sum(weight for weight, _ in valid)
            data = {'eligible': len(valid), 'strong': len(strong), 'pair_weight': total, 'models': {}}
            for name in models:
                assessments = [row['models'][name] for row in strong]
                data['models'][name] = {'positive_cosine_count': sum(s['direction'] == 'aligned' for s in assessments),
                                        'diagnostic_agreement_count': sum(s['diagnostic_agreement'] is True for s in assessments),
                                        'weighted_velocity_rmse_m_s': np.sqrt(sum(weight*row['models'][name]['error_m_s']**2
                                            for weight, row in valid)/total).item() if total else None,
                                        'velocity_loss_mass': sum(row['models'][name]['velocity_loss_mass'] for _, row in valid),
                                        'resolved_target_loss_mass': sum(row['models'][name]['resolved_target_loss_mass'] for _, row in valid)}
            summary[label][lead] = data
    return summary


def coverage(rows, batch, offsets, scales):
    """Compute separate state and action distances across training groups."""
    scaled = batch.initial/np.asarray(scales['state_error_scale_si'])
    state = np.sqrt(np.mean((scaled[:, None]-scaled[None, :])**2, axis=-1))
    action = np.sqrt(np.mean((offsets[:, None]-offsets[None, :])**2, axis=(-2, -1)))
    nominal = np.sqrt(np.mean((batch.nominal_actions[:, None]-batch.nominal_actions[None, :])**2, axis=(-2, -1)))

    def neighbor(i, j):
        return {'row': j, 'state_rmse_fixed_units': float(state[i, j]),
                'state_position_distance_m': float(np.linalg.norm(batch.initial[i, :3]-batch.initial[j, :3])),
                'state_velocity_distance_m_s': float(np.linalg.norm(batch.initial[i, VELOCITY]-batch.initial[j, VELOCITY])),
                'state_spin_distance_rad_s': float(np.linalg.norm(batch.initial[i, 6:]-batch.initial[j, 6:])),
                'offset_sequence_rmse': float(action[i, j]), 'nominal_sequence_rmse': float(nominal[i, j])}

    for i, row in enumerate(rows):
        others = [j for j, other in enumerate(rows) if other['group'] != row['group']]
        nearest = min(others, key=lambda j: (state[i, j], action[i, j], j))
        similar = min(others, key=lambda j: (action[i, j], state[i, j], j))
        exact = [j for j in others if action[i, j] <= 1e-12]
        row['cross_group'] = {'nearest_state': neighbor(i, nearest), 'nearest_signed_offset': neighbor(i, similar),
                              'exact_signed_offset': [neighbor(i, j) for j in exact],
                              'exact_signed_offset_other_history_count': sum(rows[j]['history'] != row['history'] for j in exact)}
    return state, nominal


def candidate_options(rows, batch, state, nominal):
    """Rank missing tick-168 advance histories only by training-side evidence."""
    existing = {(r['episode'], r['tick']) for r in rows if r['timed']}
    timed = [r['row'] for r in rows if r['timed']]
    results = []
    groups = sorted({rows[i]['group'] for i in timed})
    for group in groups:
        candidates = [r['row'] for r in rows if r['group'] == group and r['history'] == 'advance' and r['tick'] == 168]
        first, second = candidates
        if {int(batch.axes[i]) for i in candidates} != {0, 1}:
            raise ValueError('expected both prior held-axis response traces at the candidate state')
        np.testing.assert_array_equal(batch.initial[first], batch.initial[second])
        np.testing.assert_array_equal(batch.nominal_actions[first], batch.nominal_actions[second])
        diff = batch.baseline_first[first]-batch.baseline_second[first]
        max_diff = np.max(abs(diff), axis=0).reshape(3, 3).max(axis=1)
        half_spread = .5*np.linalg.norm(diff[np.asarray(LEADS)-1, :3], axis=1)
        same_group = min((i for i in timed if rows[i]['group'] == group), key=lambda i: (state[first, i], i))
        closest = min(timed, key=lambda i: (state[first, i], i))
        row = rows[first]
        eligible = ((row['episode'], row['tick']) not in existing and np.all(batch.noise_mask[candidates])
                    and np.all(max_diff <= (1e-5, 1e-3, 2e-2)) and np.all(half_spread <= 1e-6))
        results.append({'episode': row['episode'], 'group': group, 'history': row['history'], 'tick': row['tick'],
                        'eligible_from_archives': bool(eligible), 'nearest_timed_row': closest,
                        'nearest_timed_state_distance': float(state[first, closest]),
                        'same_group_reverse_row': same_group, 'same_group_nominal_sequence_rmse': float(nominal[first, same_group]),
                        'nominal_max_component_diff_si_p_v_spin': max_diff.tolist(),
                        'nominal_position_half_spread_m_at_report_leads': half_spread.tolist(),
                        'training_row': first})
    eligible = sorted((entry for entry in results if entry['eligible_from_archives']),
                      key=lambda row: (-row['nearest_timed_state_distance'], row['episode']))
    return {'options': results, 'selected': eligible[0] if eligible else None,
            'selection_rule': 'One unrepresented existing training advance history at tick 168. Require complete '
                              'held-axis and nominal archival comparisons. Prefer larger nearest distance to existing timed '
                              'states in fixed SI units. Break ties by episode ID. No validation results in selection.'}


def run(out):
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    bound = {}

    def verified(path, expected):
        key = Path(path).as_posix()
        bound[key] = expected
        return _verified_bytes(ROOT, key, expected)

    def guard():
        memory = process_memory(os.getpid())
        if (time.monotonic()-started >= LIMITS['wall_s'] or memory['peak_rss_bytes'] > LIMITS['rss_gib']*1024**3
                or memory['peak_private_commit_bytes'] > LIMITS['private_commit_gib']*1024**3
                or directory_bytes(out) > LIMITS['output_mib']*1024**2):
            raise RuntimeError('training coverage audit exceeded resource caps')
        return memory

    try:
        guard()
        design = json.loads(verified(DESIGN, DESIGN_SHA256))
        for name, sha in design['inputs_sha256'].items():
            verified(name, sha)
        parent, cfg, scales, fold, legacy, _, matched, draws, sources = prepare_comparison(design, guard)
        acceptance = json.loads(verified(FIT, FIT_SHA))
        if (acceptance['status'] != 'completed' or acceptance['updates'] != design['updates']
                or acceptance['anchor_draws_sha256'] != draws or acceptance['test_episode_files_read'] is not False
                or acceptance['simulation_steps'] != 0):
            raise ValueError('unexpected dense velocity checkpoint acceptance')
        execution = json.loads(verified(FIT.parent/'execution_plan.json', acceptance['artifacts_sha256']['execution_plan.json']))
        if execution['design'] != design:
            raise ValueError('fit execution used a different design')
        for name, sha in execution['source_sha256'].items():
            _verified_bytes(ROOT, name, sha)
        sources = {**sources, **execution['source_sha256']}
        ck = FIT.parent/'velocity_response_leads_900.npz'
        if acceptance['checkpoint_sha256'] != acceptance['artifacts_sha256'][ck.name]:
            raise ValueError('dense checkpoint identity mismatch')
        with np.load(BytesIO(verified(ck, acceptance['checkpoint_sha256'])), allow_pickle=False) as saved:
            candidate = load_position_model(saved)
            if type(candidate) is not TrapezoidPositionResidualMLP or int(saved['updates']) != design['updates']:
                raise ValueError('unexpected model architecture or update count')
            for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
                np.testing.assert_array_equal(saved[key], getattr(fold, key))
            np.testing.assert_array_equal(saved['known_names'], KNOWN_NAMES)
            np.testing.assert_equal(candidate.dt_s, matched.dt_s)
        manifest = parent['timed_manifest']
        manifest_hash = parent['inputs_sha256'][manifest]
        verified(manifest, manifest_hash)
        timed = prepare_timed_pairs(ROOT, manifest, manifest_hash)
        history = cfg['history_plan']
        histories = json.loads(verified(history, cfg['inputs_sha256'][history]))
        if set(fold.pairs.group_ids) & set(legacy.validation.batch.group_ids):
            raise ValueError('training audit contains a development group')
        rows, offsets = metadata(fold, legacy, timed, histories)
        models = {'trapezoid': matched, 'dense_velocity': candidate}
        summary = velocity_report(fold, models, cfg, design, rows)
        state, nominal = coverage(rows, fold.pairs, offsets, scales)
        options = candidate_options(rows, fold.pairs, state, nominal)
        report = out/'training_coverage.json'
        write_json(report, {'scope': 'Fitted training rows. Cross-group neighbor exclusions are geometric queries.',
                   'leads': LEADS, 'rows': rows, 'summary': summary, 'advance_sampling_candidates': options,
                   'distance': 'State RMS in the fixed 9 SI scales. Offset and nominal RMS are separate normalized request distances.',
                   'velocity_weights': 'Velocity response mass after replay mask and group/row weights, normalized by the full response mass.'})
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline training coverage audit imported physics')
        source = Path(__file__).resolve()
        write_json(out/'validation.json', {'status': 'completed', **SCOPE, 'limits': LIMITS,
                   'wall_s': time.monotonic()-started, **guard(), 'pair_count': len(rows),
                   'timed_pair_count': len(timed.directions), 'selected_training_option': options['selected'],
                   'artifact_sha256': {report.name: digest(report)}, 'inputs_sha256': bound,
                   'source_sha256': {**sources, source.relative_to(ROOT).as_posix(): digest(source)}})
        guard()
    except BaseException as error:
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error), 'wall_s': time.monotonic()-started, **SCOPE})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    target = parser.parse_args().output.resolve()
    if len(target.relative_to(ROOT/'runs').parts) != 1:
        raise ValueError('choose a new run directory immediately under runs')
    run(target)
