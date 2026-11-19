"""Check and fit the frozen 6/12-episode ordinary-rollout comparison."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from s3_fitability_common import BUDGET, ROOT, SCOPE, run_phase, write_json
import numpy as np

from audit_s3_small_action_response import read
from ball_retention.dataset import load_episode
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import KNOWN_NAMES, _ball_state, known_rollout, prepare_dataset
from ball_retention.pair_collection import _initial_key, prepare_collection
from ball_retention.timed_pair_data import prepare_timed_pairs
from ball_retention.world_increment_model import PACKED, WorldIncrementMLP
from ball_retention.world_model import Adam
from preflight_s3_velocity_response_leads import dormant_check, lead_vectors
from preflight_s3_world_increment_supervision import add, dot, mul, norm
from train_s3_local_velocity_increment import REFERENCE as LEGACY_49_VALIDATION, bound
from train_s3_position_integral import CONTROL, REFERENCE, checks_for, primary, regression, report
from train_s3_spin_supervision_window import accepted_artifact, score_small, small_inputs
from train_s3_world_increment import LegacyEvaluation, objectives, reference_data, score_candidate
from train_s3_world_increment_data61 import combine, extend_fold, probe, supervision


REFERENCE_PLAN = 'configs/s3_world_increment_data61_fit_plan.json'
REFERENCE_RUN = 'runs/s3_world_increment_data61_fit_20261117'
CAPTURE = 'runs/s3_continuous_routes_20261118'
PHASE = 'continuous_route_fit'
ENDPOINT = 'world_increment_900.npz'
UPDATES = 900
HORIZON = 32
CALIBRATION_DRAWS = 12
RATIO = .20
SCRIPT = 'scripts/train_s3_continuous_routes.py'


def digest_arrays(arrays, dtype):
    state = sha256()
    for array in arrays:
        state.update(np.asarray(array, dtype=dtype).tobytes())
    return state.hexdigest()


def reference(plan, guard):
    previous = read(plan, REFERENCE_PLAN)
    old_path = f'{REFERENCE_RUN}/validation.json'
    accepted = read(plan, old_path)
    if (accepted['status'] != 'completed' or accepted['updates'] != UPDATES or accepted['training_pairs'] != 61
            or accepted['plan_sha256'] != plan['files_sha256'][REFERENCE_PLAN]
            or any(accepted[k] != v for k, v in SCOPE.items())
            or any(previous[k] != plan[k] for k in ('budget', 'scope', 'seed'))
            or any(plan['files_sha256'][k] != v for k, v in previous['files_sha256'].items())):
        raise ValueError('61-pair reference differs from its acceptance')

    def artifact(name):
        return accepted_artifact(plan, old_path, accepted, name)

    opening = read(plan, artifact('fit_preflight.json'))
    pair_pre = read(plan, artifact('data_preflight.json'))
    reference_scores = read(plan, artifact('comparison.json'))
    protocol, cfg, scales, base, held, sampler_sha = reference_data(previous, guard)
    timed = [prepare_timed_pairs(ROOT, p, previous['files_sha256'][p]) for p in previous['new_timed_manifests']]
    fold, groups = extend_fold(base, timed)
    auxiliary, labels = supervision(previous, fold, previous['timed_manifests'], guard)
    parent = read(plan, previous['matched_fit_plan'])
    coefficients = read(plan, parent['window_protocol'])['auxiliary']['coefficients']
    np.testing.assert_array_equal(fold.pair_weights, pair_pre['expanded_weights'])
    if (len(fold.pairs.initial) != 61 or groups != pair_pre['group_weights']
            or labels != pair_pre['label_inventory'] or coefficients != pair_pre['coefficients']):
        raise ValueError('61-pair weights or label identities changed')
    with np.load(ROOT/artifact(ENDPOINT), allow_pickle=False) as saved:
        endpoint = WorldIncrementMLP.from_state_dict(saved)
        if int(saved['updates']) != UPDATES:
            raise ValueError('61-pair checkpoint is outside the fixed endpoint')
        for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold, key))
        np.testing.assert_array_equal(saved['known_names'], KNOWN_NAMES)
    np.testing.assert_array_equal(endpoint.inc_center, protocol['world_increment_center_m_s'])
    model = WorldIncrementMLP(PACKED, fold.state_mean, fold.state_scale, endpoint.dt_s, plan['seed'],
            inc_center=endpoint.inc_center, hidden=endpoint.hidden, residual_gain=endpoint.residual_gain)
    horizons = cfg['anchors']['horizons']
    rng = np.random.default_rng(plan['seed'])
    stream = [(h, rng.integers(len(fold.anchors[h].initial), size=cfg['anchors']['batch_size']))
              for h in (horizons[i % len(horizons)] for i in range(UPDATES))]
    actual_sha = digest_arrays((np.r_[h, take] for h, take in stream), '<i8')
    if any(actual_sha != key for key in
           (sampler_sha, accepted['sampler_sha256'], pair_pre['sampler_sha256'], opening['sampler_sha256'])):
        raise ValueError('old S2 draw stream changed')
    dense = read(plan, read(plan, previous['coverage_design'])['reference_design'])
    leads = lead_vectors(dense, HORIZON)
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), held[1]['initial']['anchor_xy_tool_m'])[0]
    evaluate, initial, features = objectives(model, fold, cfg, scales, leads, held[1], anchor)
    old = combine(model, evaluate, auxiliary, coefficients)
    measured, _ = probe(model, old, *stream[0], plan['seed'])
    if measured['pair'] != opening['initial_pair_loss'] or measured['s2_loss'] != opening['initial_s2_loss']:
        raise ValueError('original opening losses did not reproduce')
    np.testing.assert_allclose(measured['gradient_norm'],
        pair_pre['finite_differences']['initialization']['61_pairs']['gradient_norm'], atol=2e-13, rtol=0.)
    return (model, endpoint, old, initial, features, stream, sampler_sha, fold, cfg, scales, held, anchor,
            measured, reference_scores, previous, parent)


def route_windows(plan, model, fold, model_cfg, anchor, guard):
    audit = read(plan, f'{CAPTURE}/archive_audit.json')
    accepted = read(plan, f'{CAPTURE}/validation.json')
    small, large = (read(plan, f'{CAPTURE}/{name}') for name in ('small_manifest.json', 'manifest.json'))
    hashes = plan['files_sha256']
    if (accepted['status'] != 'completed' or accepted['training_updates'] != 0 or accepted['test_episode_files_read']
            or accepted['physical_steps_upper_bound'] != 46096
            or audit['validation_sha256'] != hashes[f'{CAPTURE}/validation.json']
            or audit['small_manifest_sha256'] != hashes[f'{CAPTURE}/small_manifest.json']
            or audit['large_manifest_sha256'] != hashes[f'{CAPTURE}/manifest.json']
            or small['entries'] != large['entries'][:6] or len(large['entries']) != 12
            or any(large['splits'][split]['episodes'] for split in ('validation', 'test'))):
        raise ValueError('continuous route capture or nested split changed')
    dt, substeps = large['dt_s'], large['control_substeps']
    if (dt, substeps) != (1/960, 16) or substeps*dt != model.dt_s:
        raise ValueError('continuous and original model clocks differ')
    entries = large['entries']
    groups = [e['group_id'] for e in entries[:6]]
    if (len(set(groups)) != 6 or set(groups) != {e['group_id'] for e in entries[6:]}
            or any(Counter(e['policy'] for e in tier) != {'nominal': 2, 'feedback': 2, 'pulse': 2}
                   for tier in (entries[:6], entries[6:]))):
        raise ValueError('the two tiers lost their matched group and policy counts')
    by_group = {e['group_id']: (i, e) for i, e in enumerate(entries[6:], start=6)}
    second = []
    for first in entries[:6]:
        index, other = by_group[first['group_id']]
        if ((first['physical_initial_sha256'], first['source_config_sha256']) !=
                (other['physical_initial_sha256'], other['source_config_sha256']) or first['policy'] == other['policy']):
            raise ValueError('matched initial states or distinct behaviors changed')
        second.append(index)

    window_count = 240-HORIZON+1
    starts = np.arange(window_count)
    ticks = starts[:, None]+np.arange(HORIZON)
    windows = []
    for entry in entries:
        path = f"{CAPTURE}/{entry['path']}"
        if (entry['split'] != 'train' or entry['sha256'] != hashes[path]
                or accepted['artifacts_sha256'][path] != hashes[path]
                or entry['source_config_sha256'] != hashes[entry['source_config']]):
            raise ValueError('unexpected route split, hash or initial state')
        bound(plan, path)
        cfg = read(plan, entry['source_config'])
        rows = load_episode(ROOT/path, dt, substeps, entry['episode_id'])
        if (_initial_key(cfg) != entry['physical_initial_sha256'] or cfg['history_capture']['split'] != 'train'
                or len(rows) != 240 or any(row.step_index != i or len(row.executed_joint_q) != substeps
                                          or row.invalid_reason is not None for i, row in enumerate(rows))
                or not rows[-1].task['success'] or any(cfg[k] != model_cfg[k] for k in ('physics','action','reference','ball'))
                or any(cfg['initial'][k] != model_cfg['initial'][k] for k in
                       ('tool_rotation_world','anchor_world_m','anchor_xy_tool_m'))):
            raise ValueError('episode configuration or contiguous full steps changed')
        states = [row.state_t for row in rows]+[rows[-1].next_state_before_reset]
        ball = np.stack([_ball_state(x) for x in states])
        actions = np.asarray([row.requested_action for row in rows])
        # Every window uses one initial state and the recorded action sequence.
        # Future physical states provide targets only.
        known = known_rollout([x.time_s for x in states[:window_count]],
            [x.tilt_rad for x in states[:window_count]], [x.tilt_rate_rad_s for x in states[:window_count]],
            actions[ticks], cfg, model.dt_s)
        windows.append((model.normalize(ball[starts]), model.features(known, cfg, anchor, fold.known_mean, fold.known_scale),
                        model.normalize(ball[ticks+1])))
        guard('prepare_continuous_windows')
    arrays = tuple(np.concatenate(field) for field in zip(*windows))
    if (audit['fit_windows_by_tier']['small']['32'], audit['fit_windows_by_tier']['large']['32']) != (
            6*window_count, len(arrays[0])) or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('ordinary windows differ from the audited inventory')
    rng = np.random.default_rng([plan['seed'], HORIZON])
    start_ticks = rng.integers(window_count, size=(UPDATES, 6, 2))
    common = np.arange(6)
    streams = {str(size): (start_ticks+window_count*np.stack((common, ids), axis=-1)).reshape(UPDATES, 12)
               for size, ids in ((6, common), (12, np.asarray(second)))}
    np.testing.assert_array_equal(streams['6'][:, ::2], streams['12'][:, ::2])
    np.testing.assert_array_equal(streams['6'] % window_count, streams['12'] % window_count)
    return arrays, streams, {'windows_per_episode': window_count, 'group_order': groups,
            'shared_start_ticks_sha256': digest_arrays([start_ticks], '<i8'),
            'sampler_sha256_by_tier': {k: digest_arrays([v], '<i8') for k, v in streams.items()},
            'causal_input_and_target_sha256': digest_arrays(arrays, '<f8'),
            'window_counts': {'6': 6*window_count, '12': len(arrays[0])}}


def route_objective(model, arrays, cfg, scales):
    components = np.asarray(cfg['components'])
    units = np.asarray(scales['state_error_scale_si'])

    def evaluate(indices):
        return model.loss_and_grad(*(array[indices] for array in arrays), state_weights=components,
                                   state_error_scale_si=units)

    return evaluate


def objective(old, ordinary, indices, coefficient):
    def evaluate(h, take):
        s2, pair, gradient = old(h, take)
        loss, added = ordinary(indices)
        combined = add(gradient, mul(added, coefficient))
        total = pair['loss']+coefficient*loss
        if not np.isfinite(total+s2) or any(not np.isfinite(x).all() for x in combined.values()):
            raise FloatingPointError('nonfinite ordinary-augmented loss or gradient')
        return s2, {'loss': total, 'original_61_pair_terms': pair, 'ordinary_loss': loss}, combined

    return evaluate


def check_opening(model, old, ordinary, stream, streams, seed):
    old_norms, new_norms = [], {key: [] for key in streams}
    for i, (h, take) in enumerate(stream[:CALIBRATION_DRAWS]):
        old_norms.append(norm(old(h, take)[2]))
        for key, indices in streams.items():
            new_norms[key].append(norm(ordinary(indices[i])[1]))
    largest = max(np.mean(values) for values in new_norms.values())
    coef = float(RATIO*np.mean(old_norms)/largest)
    if not np.isfinite(coef) or coef <= 0.:
        raise FloatingPointError('nonfinite or zero ordinary scaling coefficient')
    h, take = stream[0]
    s2, pair, grad = old(h, take)
    arms = {}
    for key, indices in streams.items():
        loss, add_grad = ordinary(indices[0])
        added = mul(add_grad, coef)
        candidate = objective(old, ordinary, indices[0], coef)
        combined, g = probe(model, candidate, h, take, seed)
        second_probe, _ = probe(model, candidate, h, take, seed+1)
        if (combined['s2_loss'] != s2 or combined['pair']['original_61_pair_terms'] != pair
                or combined['pair']['ordinary_loss'] != loss):
            raise ValueError('opening objective changes original terms')
        np.testing.assert_allclose(combined['pair']['loss']-pair['loss'], coef*loss, rtol=1e-12, atol=1e-14)
        for name in g:
            np.testing.assert_allclose(g[name]-grad[name], added[name], rtol=1e-12, atol=2e-14)
        arms[key] = {**combined, 'second_fd_error': second_probe['fd_error'],
                    'added_fraction_of_original_gradient': norm(added)/norm(grad),
                    'added_vs_original_gradient_cosine': dot(added, grad)/(norm(added)*norm(grad)),
                    'combined_gradient_sha256': digest_arrays(g.values(), '<f8')}
    return {'coefficient': coef, 'calibration_train_only': True, 'calibration_draws': CALIBRATION_DRAWS,
            'calibration_original_gradient_norms': old_norms, 'calibration_unscaled_route_gradient_norms': new_norms,
            'initial_parameter_sha256': digest_arrays(model.params.values(), '<f8'), 'candidate_opening': arms}


def fit(model, old, ordinary, stream, indices, coefficient, optimizer_cfg, out, fold, guard):
    if optimizer_cfg['name'] != 'Adam':
        raise ValueError('original optimizer changed')
    optimizer = Adam(model.params, learning_rate=optimizer_cfg['learning_rate'],
                     max_grad_norm=optimizer_cfg['max_grad_norm'])
    guard('fit', 0, log=True)
    with (out/'training.jsonl').open('x', encoding='utf-8') as log:
        for update, ((h, take), batch) in enumerate(zip(stream, indices, strict=True), 1):
            guard('fit', optimizer.iteration)
            s2, losses, grad = objective(old, ordinary, batch, coefficient)(h, take)
            size = optimizer.step(grad)
            if any(not np.isfinite(x).all() for x in model.params.values()):
                raise FloatingPointError(f'nonfinite parameters after update {update}')
            guard('fit', update, log=update % 30 == 0)
            if update % 30 == 0:
                print(json.dumps({'update': update, 'anchor_loss': s2, 'objective': losses,
                                  'gradient_norm': size}, allow_nan=False), file=log, flush=True)
    dormant_check(model.params)
    if optimizer.iteration != UPDATES:
        raise ValueError('optimizer did not reach the sole fixed endpoint')
    path = out/ENDPOINT
    np.savez_compressed(path, **model.state_dict(), known_names=KNOWN_NAMES,
                        known_mean=fold.known_mean, known_scale=fold.known_scale, updates=optimizer.iteration)
    with np.load(path, allow_pickle=False) as saved:
        return WorldIncrementMLP.from_state_dict(saved)


def route_scores(model, arrays, per_tier):
    initial, features, truth = arrays
    scores = {}
    for name, selection in (('shared_six', slice(0, per_tier)), ('additional_six', slice(per_tier, None))):
        lead = model.rollout(initial[selection], features[selection])[:, -1]-truth[selection, -1]
        if not np.isfinite(lead).all():
            raise FloatingPointError('nonfinite fixed-endpoint route prediction')
        error = lead*model.state_scale
        scores[name] = {key: float(np.sqrt(np.mean(np.sum(error[:, part]**2, axis=1))))
            for key, part in (('position_m', slice(0, 3)), ('velocity_m_s', slice(3, 6)), ('spin_rad_s', slice(6, 9)))}
    return scores


def legacy_evaluator(plan, previous, cfg, held, fold, anchor):
    design = read(plan, previous['held_design'])
    legacy = prepare_collection(ROOT, ROOT/design['base_collection'])
    s2 = prepare_dataset((ROOT/design['s2_manifest']).parent, horizons=tuple(cfg['anchors']['horizons']), verify_test=False)
    histories = {row['episode_id']: row['history_id'] for row in read(plan, cfg['history_plan'])['episodes']}

    def evaluate(model):
        scores = report(LegacyEvaluation(model, held[1], anchor, fold), fold.known_mean, fold.known_scale,
                        legacy, s2, histories)
        scores['pairs'] = [row for row in scores['pairs'] if row['split'] == 'validation']
        return scores

    return evaluate


def recover_endpoint(plan, fold, evidence):
    source = Path(plan['score_interrupted_fit'])
    original_receipt = read(plan, source/'execution_plan.json')
    original = original_receipt['plan']
    stopped = read(plan, source/'stopped.json')
    log = [json.loads(line) for line in bound(plan, source/'training.jsonl').splitlines()]
    if (original_receipt['plan_sha256'] != plan['files_sha256'][plan['interrupted_fit_plan']]
            or original != read(plan, plan['interrupted_fit_plan']) or original['output'] != source.as_posix()
            or original['tier'] != '6' or original['updates'] != UPDATES
            or plan['files_sha256'][(source/'source_before_scoring_fix.py').as_posix()] != original['files_sha256'][SCRIPT]
            or any(plan['files_sha256'][p] != digest for p, digest in original['files_sha256'].items() if p != SCRIPT)
            or stopped['status'] != 'stopped' or stopped['stage'] != 'candidate_response'
            or stopped['reason'] != repr(CONTROL) or stopped['updates'] != UPDATES
            or any(stopped[k] != value for k, value in SCOPE.items())
            or [row['update'] for row in log] != list(range(30, UPDATES+1, 30))
            or evidence != read(plan, source/'opening.json')):
        raise ValueError('interrupted run is not the single completed 6-episode fit')
    path = source/ENDPOINT
    bound(plan, path)
    with np.load(ROOT/path, allow_pickle=False) as saved:
        if int(saved['updates']) != UPDATES:
            raise ValueError('interrupted fit has no fixed endpoint')
        for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold, key))
        np.testing.assert_array_equal(saved['known_names'], KNOWN_NAMES)
        fitted = WorldIncrementMLP.from_state_dict(saved)
    dormant_check(fitted.params)
    return fitted


def run(plan, out, guard):
    tier = plan['tier']
    score_only = 'score_interrupted_fit' in plan
    expected_updates = 0 if tier == 'check' or score_only else UPDATES
    if (plan['schema'] != 's3-continuous-route-fit-v1'
            or tier not in ('check', '6', '12') or plan['updates'] != expected_updates or (score_only and tier != '6')
            or plan['ordinary'] != {'horizon': HORIZON, 'windows_per_update': 12, 'initialization_norm_ratio': RATIO,
                                     'calibration_draws': CALIBRATION_DRAWS}):
        raise ValueError('ordinary fit differs from the frozen scope')
    (model, endpoint, old, initial, features, stream, sampler_sha, fold, cfg, scales, held, anchor,
        measured, previous_scores, previous, parent) = reference(plan, guard)
    arrays, streams, inventory = route_windows(plan, model, fold, held[1], anchor, guard)
    ordinary = route_objective(model, arrays, cfg, scales)
    opening = check_opening(model, old, ordinary, stream, streams, plan['seed'])
    evidence = {'original': measured, 'ordinary': inventory, 'opening': opening, 'optimizer': cfg['optimizer'],
                'original_s2_sampler_sha256': sampler_sha}
    write_json(out/'opening.json', evidence)
    guard('reference_scores', 0)
    small = small_inputs(plan, held)
    full_ref, _ = score_candidate(endpoint, held, fold, anchor, guard)
    small_ref = score_small(endpoint, held, fold, anchor, small, guard)
    legacy = legacy_evaluator(plan, previous, cfg, held, fold, anchor)
    if (full_ref != previous_scores['held_timing'] or small_ref != small[2]
            or legacy(endpoint) != previous_scores['legacy_validation']):
        raise ValueError('original 61-pair endpoint scores did not reproduce')
    summary = {'training_pairs': 61, 'sampler_sha256': sampler_sha, 'opening_gradient_verified': True,
               'original_development_scores_reproduced': True, 'main_model_replaced': False,
               'checkpoint_selection': False, 'tier': tier}
    if tier == 'check':
        return summary

    pre_path = plan['preflight_validation']
    pre = read(plan, pre_path)
    if (pre['status'] != 'completed' or pre['tier'] != 'check' or pre['updates'] != 0
            or not pre['opening_gradient_verified'] or not pre['original_development_scores_reproduced']
            or pre['plan_sha256'] != plan['files_sha256'][plan['preflight_plan']]
            or any(pre[key] != value for key, value in SCOPE.items())
            or evidence != read(plan, accepted_artifact(plan, pre_path, pre, 'opening.json'))):
        raise ValueError('executable opening differs from accepted zero-update preflight')
    if score_only:
        fitted = recover_endpoint(plan, fold, evidence)
    else:
        fitted = fit(model, old, ordinary, stream, streams[tier], opening['coefficient'], cfg['optimizer'], out, fold, guard)
        np.testing.assert_array_equal(fitted.rollout(initial, features[0]), model.rollout(initial, features[0]))
    full_new, forecasts = score_candidate(fitted, held, fold, anchor, guard)
    small_new = score_small(fitted, held, fold, anchor, small, guard)
    legacy_new = legacy(fitted)
    original = read(plan, previous['original_scores'])
    history = {'40_pairs': original['legacy_development'][CONTROL],
               '46_pairs': read(plan, previous['reference_46_scores'])['legacy_validation'],
               '49_pairs': read(plan, LEGACY_49_VALIDATION.with_name('comparison.json'))['legacy_validation'],
               '49_pairs_full_window': read(plan, Path(parent['output'])/'comparison.json')['legacy_validation'],
               '61_pairs': previous_scores['legacy_validation']}
    protections = {key: regression(value, legacy_new) for key, value in history.items()}
    if any(len(value['checks']) != 22 for value in protections.values()):
        raise ValueError('legacy protections no longer have 22 checks')
    positions = primary(full_new)
    nominal = positions['pitch_pulse']['absolute_model_nominal_error_norm']
    gates = checks_for(positions, nominal/original['nominal_h32_position_m'][REFERENCE], {CONTROL: protections['40_pairs']})
    gates.update({f'protections_vs_{key}': value['passed'] for key, value in protections.items()})
    scores = {key: route_scores(m, arrays, inventory['window_counts']['6'])
              for key, m in (('61_pairs', endpoint), ('candidate', fitted))}
    write_json(out/'comparison.json', {'full': {'61_pairs': full_ref, 'candidate': full_new},
        'small': {'61_pairs': small_ref, 'candidate': small_new}, 'legacy_protections': protections,
        'legacy_validation': legacy_new, 'nominal_h32_position_m': {'61_pairs': previous_scores['nominal_h32_position_m'],
            'candidate': nominal}, 'gates': gates, 'ordinary_route_h32_rms_si': scores})
    write_json(out/'timed_rollouts.json', forecasts)
    return {**summary, 'replacement_gate_passed': all(gates.values()), 'checkpoint_updates': UPDATES,
            'score_only': score_only}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
