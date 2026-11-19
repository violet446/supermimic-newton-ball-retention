"""Audit training spin labels and re-roll fixed models with predicted spin controls."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from audit_s3_small_action_response import read
from audit_s3_world_increment_drift import require_close
from ball_retention.action_eval import evaluate_pair
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, known_rollout
from ball_retention.pair_collection import prepare_collection
from ball_retention.timed_pair_data import prepare_timed_pairs
from ball_retention.world_increment_model import WorldIncrementMLP, _endpoints
from collect_s3_small_action_response import load_run, score_case
from evaluate_s3_history import aggregate
from preflight_s3_deterministic_action_response import action_sequences
from preflight_s3_small_action_response import build_branches
from preflight_s3_velocity_response_leads import lead_vectors
from train_s3_history import regression
from train_s3_world_increment import reference_data
from train_s3_world_increment_data61 import extend_fold


PHASE = 'roll_spin_control'
MODES = ('free', 'nominal_mean', 'zero_half', 'nominal_both', 'nominal_both_h9_h16')
LEADS = (8, 16, 32)


def known_for(state, actions, cfg, dt):
    return known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s, actions, cfg, dt)


def rotations(known, cfg, dt, anchor, sign):
    shape = known.shape[:-1]
    return _endpoints(known.reshape(-1, known.shape[-1]), cfg, dt, anchor, sign)[0][0].reshape(*shape, 3, 3)


def matvec(matrix, vector):
    return (matrix @ vector[..., None])[..., 0]


def pair_roll(model, cfg, anchor, mean, scale, start, known, nominal_known, mode):
    features = model.features(known, cfg, anchor, mean, scale)
    initial = model.normalize(start)
    if mode == 'free':
        return model.denormalize(model.rollout(np.broadcast_to(initial, (2, 9)), features))
    nominal = model.denormalize(model.rollout(initial[None], model.features(nominal_known, cfg, anchor, mean, scale)[None])[0])
    nominal_pre = np.vstack((start, nominal[:-1]))
    nominal_world = matvec(rotations(nominal_known, cfg, model.dt_s, anchor, -1), nominal_pre[:, 6:])
    branch_rotation = rotations(known, cfg, model.dt_s, anchor, -1)
    previous = np.broadcast_to(start, (2, 9)).copy()
    results = []
    for step in range(known.shape[1]):
        changed = previous.copy()
        if step > 0 and (mode != 'nominal_both_h9_h16' or 8 <= step < 16):
            rotation = branch_rotation[:, step]
            world = matvec(rotation, previous[:, 6:])
            common = world.mean(axis=0) if mode == 'zero_half' else nominal_world[step]
            half = .5*(world[0]-world[1]) if mode == 'nominal_mean' else np.zeros(3)
            changed[:, 6:] = matvec(rotation.swapaxes(-1, -2), np.stack((common+half, common-half)))
        previous = model.denormalize(model.step(model.normalize(changed), features[:, step]))
        results.append(previous)
    return np.stack(results, axis=1)


def coverage(plan, cfg, anchor, dev, guard):
    fit_path = plan['data61_fit_plan']
    fit = read(plan, fit_path)
    _, training_cfg, _, fold, _, _ = reference_data(fit, guard)
    manifests = fit['timed_manifests']
    timed = {p: prepare_timed_pairs(ROOT, p, fit['files_sha256'][p]) for p in manifests}
    new = [timed[p] for p in fit['new_timed_manifests']]
    expanded, _ = extend_fold(fold, new)
    batch = expanded.pairs
    labels = [d for p in manifests for d in timed[p].directions]
    categories = np.asarray([*('sustained_roll' if n == 0 else 'sustained_pitch' for n in batch.axes[:40]), *labels])
    if len(categories) != 61 or len(set(batch.group_ids)) != 6:
        raise ValueError('training inventory differs from matched fit')
    dt = 1/cfg['physics']['control_hz']
    known = np.stack((batch.plus_known, batch.minus_known))
    rotations_world = rotations(known, cfg, dt, anchor, 1)
    spin_world = matvec(rotations_world, np.stack((batch.plus_target, batch.minus_target))[..., 6:])
    half = .5*(spin_world[0]-spin_world[1])
    norms = np.linalg.norm(half, axis=-1)
    noise = .5*np.linalg.norm(batch.baseline_first[..., 6:]-batch.baseline_second[..., 6:], axis=-1)
    floor = np.maximum(.0001, 5*noise)
    strong = batch.pair_mask & batch.baseline_mask & (norms >= 10*floor)
    dev_norm = np.linalg.norm(dev, axis=-1)
    cosine = np.sum(half*dev, axis=-1)/np.maximum(norms*dev_norm, 1e-30)
    ratios = norms/np.maximum(dev_norm, 1e-30)
    comparable = strong & (cosine >= .8) & (ratios >= .5) & (ratios <= 2)
    dense_path = read(plan, fit['coverage_design'])['reference_design']
    leads, _ = lead_vectors(read(plan, dense_path), 32)
    rows = {}
    for tag in sorted(set(categories)):
        keep = categories == tag
        by_lead = comparable[keep].sum(axis=0)
        rows[tag] = {'pair_rows': int(keep.sum()), 'physical_groups': len(set(batch.group_ids[keep])),
                     'pair_steps_with_full_targets_h9_h16': int(batch.pair_mask[keep, 8:16].sum()),
                     'strong_spin_steps_h9_h16': int(strong[keep, 8:16].sum()),
                     'dev_matching_counts_by_lead': by_lead.tolist(),
                     'dev_leads_covered_h9_h16': int(np.sum(by_lead[8:16] > 0)),
                     'dev_leads_covered_h9_h16_at_optimized_spin_leads': int(np.sum((leads[8:16] > 0) & (by_lead[8:16] > 0)))}
    seen = set()
    nominal = []
    for i, (episode, tick) in enumerate(zip(batch.episode_ids, batch.snapshot_ticks)):
        key = str(episode), int(tick)
        if key in seen:
            continue
        seen.add(key)
        observed = batch.baseline_first[i, :, 6:]
        mask = batch.baseline_mask[i]
        step = np.diff(np.vstack((batch.initial[i, 6:], observed)), axis=0)
        repeat_spread = .5*np.linalg.norm(observed-batch.baseline_second[i, :, 6:], axis=-1)
        nominal.append({'episode': key[0], 'tick': key[1], 'timed': i >= 40,
                        'h9_h16_full_labels': int(mask[8:16].sum()),
                        'max_repeat_spread_rad_s': float(np.max(repeat_spread[mask])),
                        'h9_h16_spin_change_rms_rad_s': float(np.sqrt(np.mean(np.sum(step[8:16]**2, axis=-1))))})
    return {'categories': rows, 'nominal_snapshots': nominal, 'spin_supervised_leads': (np.flatnonzero(leads)+1).tolist(),
            'paired_spin_component_weights': training_cfg['components'][6:9],
            'response_spin_scales_rad_s': training_cfg['response_scales_si'][6:9], 'dev_norm_by_lead_rad_s': dev_norm.tolist()}


def development(model, cfg, anchor, mean, scale, initial, parent, spec, traces, mode):
    nominal = np.asarray(traces['nominal_1']['actions'])
    directions, full_requests = action_sequences(parent, nominal)
    cases, small_dirs, small_requests = build_branches(parent, nominal)
    directions.update(small_dirs)
    requests = {**full_requests, **small_requests}
    start = _ball_state(initial)
    nk = known_for(initial, nominal, cfg, model.dt_s)
    baseline = pair_roll(model, cfg, anchor, mean, scale, start, np.stack((nk, nk)), nk, 'free')[0]
    neutral = pair_roll(model, cfg, anchor, mean, scale, start, np.stack((nk, nk)), nk, mode)
    require_close(neutral, np.broadcast_to(baseline, neutral.shape))
    replay = [traces[x]['states_raw'] for x in ('nominal_1', 'nominal_2')]
    results = {}
    for name, direction in directions.items():
        def predict(actions):
            known = known_for(initial, actions, cfg, model.dt_s)
            return pair_roll(model, cfg, anchor, mean, scale, start, known, nk, mode)

        labels = [f'{name}_{sign}' for sign in ('plus', 'minus')]
        plus, minus = predict(np.stack([requests[l] for l in labels]))
        eps = parent['model']['fd_step']*.25
        def derivative(step):
            # The intervention depends on the signed pair, evaluated around the same nominal request.
            a, b = predict(nominal + np.array([1., -1.])[:, None, None]*step*direction)
            return (a-b)/(2*step)
        local = derivative(eps)
        np.testing.assert_allclose(local, derivative(eps*.5), rtol=3e-5, atol=1e-7)
        amplitude = cases[name]['amplitude'] if name in cases else parent['amplitude']
        scores = score_case(spec['scoring'], replay, *(traces[l]['states_raw'] for l in labels), baseline,
                            {'plus': plus, 'minus': minus, 'directional_jvp': local}, amplitude)
        results[name] = {str(h): scores['by_lead'][str(h)]['position'] for h in (8, 16, 24, 32)}
    return results


def legacy_scores(model, collection, archive, anchor, mean, scale, mode):
    mixed = collection.validation
    b, cfg = mixed.batch, collection.model_config
    old = {tuple(r[k] for k in ('episode','tick','axis','amplitude')): r for r in archive['pairs']}
    rows = []
    for i in range(len(b.initial)):
        identity = (str(b.episode_ids[i]), int(b.snapshot_ticks[i]), int(b.axes[i]), float(mixed.amplitudes[i]))
        info = old[identity]
        nk = known_for(_parse_state(b.initial_state[i]), b.nominal_actions[i], cfg, model.dt_s)
        pred = pair_roll(model, cfg, anchor, mean, scale, b.initial[i], np.stack((b.plus_known[i], b.minus_known[i])),
                         nk, mode)
        def masked(target, mask):
            return [v if a else None for v, a in zip(target, mask)]
        resp = evaluate_pair(*pred, masked(b.plus_target[i], b.plus_mask[i]), masked(b.minus_target[i], b.minus_mask[i]),
                             identity[-1], cfg['initial']['anchor_xy_tool_m'],
                             replay_first=masked(b.baseline_first[i], b.baseline_mask[i]),
                             replay_second=masked(b.baseline_second[i], b.baseline_mask[i]), leads=LEADS)
        absolute = {}
        for h in LEADS:
            errors = [p[h-1]-values[i,h-1] for p, values, mask in zip(
                pred, (b.plus_target,b.minus_target), (b.plus_mask,b.minus_mask)) if mask[i,h-1]]
            absolute[str(h)] = {'branches': len(errors),
                                'squared_p_v_w_si': np.sum(np.asarray(errors).reshape(-1, 3, 3)**2, axis=(0,2)).tolist()}
            if mode == 'free':
                require_close(absolute[str(h)]['squared_p_v_w_si'], info['absolute'][str(h)]['squared_p_v_w_si'])
        rows.append({**info, 'response': resp, 'absolute': absolute})
    summary = {}
    for group, amplitude in sorted({(x['group'], x['amplitude']) for x in rows}):
        key = f'group/validation/{group}/{amplitude}'
        summary[key] = aggregate([x for x in rows if (x['group'], x['amplitude']) == (group, amplitude)], LEADS, .5, 10)
    result = {**archive, 'summary': summary}
    return regression(archive, result), summary


def run(plan, out, guard):
    if plan['updates']:
        raise ValueError('spin audit is offline only')
    cap = Path(plan['capture_plan'])
    launch, cfg, spec, native, _, _, _, read_bound = load_run(ROOT/cap, plan['files_sha256'][cap.as_posix()])
    status = read(plan, plan['capture_validation'])
    if (status['status'], status['physics_steps'], status['training_updates'], status['test_episode_files_read'],
            len(status['accepted_prefixes'])) != ('completed',7168,0,False,4):
        raise ValueError('missing accepted small-action capture')
    folder = Path(plan['capture_validation']).parent/'worker'
    traces = {name: read(plan, folder/f'{name}.json') for name in status['branches']}
    parent = read_bound(launch['original_response_plan'])
    old_dir = Path(launch['archived_response_validation']).parent/'worker'
    for name in parent['branch_order'][2:]:
        traces[name] = read_bound(old_dir/f'{name}.json')
    start = _parse_state(native['snapshot']['state'])
    mesh = load_tool(ROOT/'assets/shovel.STL')
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    world_spin = [np.asarray([row['ball']['angular_velocity'] for row in traces[f'roll_component_{side}']['states']])
                  for side in ('plus', 'minus')]
    measured = .5*(world_spin[0]-world_spin[1])
    train = coverage(plan, cfg, anchor, measured, guard)
    collection = prepare_collection(ROOT, ROOT/plan['legacy_collection'])
    small_scores = read(plan, Path(plan['capture_validation']).with_name('same_amplitude_scores.json'))['models']
    output = {}
    for tag, source in launch['fixed_endpoint_sources'].items():
        path = Path(source)/'world_increment_900.npz'
        archive = read_bound(Path(source)/'comparison.json')['legacy_validation']
        with np.load(ROOT/path, allow_pickle=False) as saved:
            model = WorldIncrementMLP.from_state_dict(saved)
            mean, scale = saved['known_mean'], saved['known_scale']
        results = {}
        for mode in MODES:
            dev = development(model, cfg, anchor, mean, scale, start, parent, spec, traces, mode)
            protection, legacy = legacy_scores(model, collection, archive, anchor, mean, scale, mode)
            if mode == 'free':
                if not protection['passed']:
                    raise ValueError('unmodified legacy protection failed to reproduce')
                for case, row in dev.items():
                    base = small_scores[tag][case] if case in spec['cases'] else read_bound(
                        Path(source)/'comparison.json')['held_timing'][case]
                    for field in ('model_fd_half_difference', 'newton_half_difference', 'model_jvp_linearized_displacement'):
                        np.testing.assert_allclose(row['32'][field], base['by_lead']['32']['position'][field], rtol=0., atol=1e-8)
            results[mode] = {'development': dev, 'legacy_regression': protection, 'legacy_summary': legacy}
            guard(f'{tag}/{mode}')
        output[tag] = results
    write_json(out/'roll_spin_control.json', {'case': spec['case'], 'training': train, 'models': output,
               'scope': 'Every controlled rollout uses only predicted states. The nominal world spin is rotated into each signed branch tool frame.'})
    return {'training_pairs_audited': 61, 'fixed_checkpoints': len(output), 'rollout_modes': list(MODES)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
