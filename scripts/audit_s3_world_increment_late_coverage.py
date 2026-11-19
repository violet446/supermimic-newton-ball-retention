"""Compare exact signed increment labels and one-step error in two frozen windows."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, forecast, run_phase, write_json

import numpy as np

from audit_s3_ball_velocity_components import frame_series, half, rms
from audit_s3_world_increment_drift import branch_world_terms, integral, require_close
from audit_s3_world_increment_supervision_drift import ENDPOINTS, load_endpoint
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, known_rollout
from ball_retention.world_increment_loss import world_increment_loss
from deterministic_archive_metrics import _full_branch
from preflight_s3_world_increment_supervision import SETTINGS
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'world_increment_late_coverage'
WINDOWS = {'H1-H16': slice(0, 16), 'H17-H32': slice(16, 32)}
HORIZON = 32


def trace_inputs(traces, keys, state, cfg, dt, gravity):
    branches = [traces[k] for k in keys]
    frames = [frame_series(state, branch, dt) for branch in branches]
    truth = np.stack([branch['raw'] for branch in branches])
    actions = np.stack([branch['actions'] for branch in branches])
    known = known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s, actions, cfg, dt)
    targets = np.stack([np.diff(frame[3], axis=0)-gravity for frame in frames])
    return frames, truth, known, targets


def one_step(model, initial, fold, cfg, anchor, known, targets, truth, frames, gravity, axis):
    states = np.concatenate((initial[:, None], truth[:, :-1]), axis=1)
    scaled = (known-fold.known_mean)/fold.known_scale
    _, _, head_error = world_increment_loss(model, states, scaled, targets,
        np.full(HORIZON, 1/HORIZON), SETTINGS['pair_scale_m_s'], paired=True)
    features = model.features(known, cfg, anchor, fold.known_mean, fold.known_scale)
    local = forecast(model, initial, features, truth, local=True)
    # Recover model endpoints using archived tool poses, as in the prior development scores.
    free = forecast(model, initial, features, truth)
    terms = [branch_world_terms(frame, y, l, f, gravity)
             for frame, y, l, f in zip(frames, truth, local, free)]
    scored_error = half(np.stack([raw['local'] for _, raw, _ in terms]))
    local_position = None
    if axis is not None:
        velocity = half(np.stack([terms_i['local_increment'] for terms_i, _, _ in terms]))
        local_position = integral(velocity, model.dt_s) @ axis
    return head_error, scored_error, local_position


def evaluate_snapshot(key, state, cfg, traces, names, endpoints, scores, mesh, gates):
    dt = next(iter(endpoints.values()))[0].dt_s
    gravity = np.array([0., 0., -cfg['physics']['gravity_m_s2']*dt])
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    _, nominal, _, targets = trace_inputs(traces, ('nominal_1', 'nominal_2'), state, cfg, dt, gravity)
    noise = np.linalg.norm(half(targets), axis=-1)
    floor = np.maximum(SETTINGS['floor_m_s'], SETTINGS['noise_multiplier']*noise)
    rows = {}
    for name in names:
        keys = tuple(f'{name}_{side}' for side in ('plus', 'minus'))
        frames, truth, known, targets = trace_inputs(traces, keys, state, cfg, dt, gravity)
        target = half(targets)
        size = np.linalg.norm(target, axis=-1)
        resolved, strong = size >= floor, size >= gates['strong_multiplier']*floor
        final = half(truth)[-1, :3]
        pos_floor = max(gates['position_floor_m'], gates['noise_multiplier']*np.linalg.norm(half(nominal)[-1, :3]))
        axis = final/np.linalg.norm(final) if np.linalg.norm(final) >= gates['strong_multiplier']*pos_floor else None
        initial = np.broadcast_to(_ball_state(state), (2, 9))
        metrics, per_step = {}, {}
        for tag, (model, fold, *_) in endpoints.items():
            head, measured, position = one_step(model, initial, fold, cfg, anchor, known, targets, truth, frames, gravity, axis)
            recorded = scores[tag]['exact_label_comparison'][key][name]['world_increment']['local']
            require_close(rms(measured[7:24]), recorded['net_world_half_error_rms_m_s'])
            per_step[tag] = {'head_error_m_s': head.tolist(), 'scored_error_m_s': measured.tolist()}
            metrics[tag] = {label: {
                'head_rmse_m_s': rms(head[w]),
                'scored_rmse_m_s': rms(measured[w]),
                'scored_relative_rms': rms(measured[w])/rms(target[w]),
                'local_increment_position_projection_m': None if position is None else float(position[w].sum()),
            } for label, w in WINDOWS.items()}
        rows[name] = {'target_m_s': target.tolist(), 'noise_m_s': noise.tolist(),
            'floor_m_s': floor.tolist(), 'resolved': resolved.tolist(), 'strong': strong.tolist(),
            'newton_h32_position_response_m': final.tolist(), 'h32_position_strong': axis is not None,
            'by_window': {label: {'resolved': int(resolved[w].sum()), 'strong': int(strong[w].sum()),
                                  'target_rmse_m_s': rms(target[w])} for label, w in WINDOWS.items()},
            'models': metrics, 'per_step_errors': per_step}
    return rows


def direction_coverage(training, development):
    """Compare signed Newton labels for the same action and lead across snapshots."""
    a = np.asarray(development['target_m_s'])
    b = np.stack([v['target_m_s'] for v in training.values()])
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=2)
    dot = np.einsum('ij,kij->ki', a, b)
    norm_product = b_norm*a_norm
    cosine = np.divide(dot, norm_product, out=np.zeros_like(dot), where=norm_product > 0.)
    valid = np.asarray([v['resolved'] for v in training.values()]) & np.asarray(development['strong'])
    result = {}
    for label, w in WINDOWS.items():
        qualified = valid[:, w]
        aligned = (cosine[:, w] >= .8) & qualified
        per_step = []
        first = w.start
        for i in np.flatnonzero(np.asarray(development['strong'])[w]):
            j = first+i
            per_step.append({'lead': int(j+1),
                'training_cosines': {key: float(cosine[k, j]) for k, key in enumerate(training) if valid[k, j]},
                'training_to_development_magnitude': {key: float(b_norm[k, j]/a_norm[j])
                                                       for k, key in enumerate(training) if valid[k, j]}})
        result[label] = {'development_strong_leads': int(np.sum(np.asarray(development['strong'])[w])),
            'leads_with_resolved_training_label': int(np.any(qualified, axis=0).sum()),
            'leads_with_cosine_ge_0_8': int(np.any(aligned, axis=0).sum()),
            'leads_with_cosine_ge_0_8_in_two_snapshots': int(np.sum(aligned.sum(axis=0) >= 2)),
            'by_snapshot': {key: {'valid': int(qualified[k].sum()), 'aligned': int(aligned[k].sum())}
                            for k, key in enumerate(training)}, 'by_lead': per_step}
    return result


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    if plan['updates'] != 0 or plan['windows'] != [[1, 16], [17, 32]]:
        raise ValueError('unexpected diagnostic scope')
    capture, cfg, fields, traces, held_case = held = read_held(read(plan['held_design']))
    mesh = load_tool(ROOT/'assets/shovel.STL')
    dev_anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    endpoints = {key: load_endpoint(plan, key, held, dev_anchor, guard) for key in ENDPOINTS}
    scores = {key: read(path/'comparison.json') for key, path in ENDPOINTS.items()}
    if len({x[-1] for x in endpoints.values()}) != 1 or len({x[0].dt_s for x in endpoints.values()}) != 1:
        raise ValueError('mismatched endpoint sampler or clock')
    dev = evaluate_snapshot('development', _parse_state(fields), cfg, traces, capture['directions'],
                            endpoints, scores, mesh, capture['gates'])
    training, metadata = {}, {}
    for path in plan['timed_manifests']:
        for entry in read(path)['entries']:
            name = entry['episode_id']
            if entry['split'] != 'train' or name in training or entry['group_id'] == held_case['case']['group']:
                raise ValueError('unexpected training episode')
            config_path = entry['source_config']
            if plan['files_sha256'][config_path] != entry['source_config_sha256']:
                raise ValueError('training config digest changed')

            def load_ref(ref):
                file = Path(path).parent/ref['path']
                if plan['files_sha256'][file.as_posix()] != ref['sha256']:
                    raise ValueError(f'training branch digest changed: {file}')
                return read(file)

            snapshot = _parse_state(load_ref(entry['native_snapshot'])['state'])
            names = [pair['direction_id'] for pair in entry['pairs']]
            if names != list(capture['directions']):
                raise ValueError('training direction order changed')
            refs = [*entry['nominal_branches'], *(b for pair in entry['pairs'] for b in pair['branches'])]
            rows = {b['label']: _full_branch(load_ref(b), entry['snapshot_tick']) for b in refs}
            training[name] = evaluate_snapshot(name, snapshot, read(config_path), rows, names,
                                               endpoints, scores, mesh, capture['gates'])
            metadata[name] = {k: entry[k] for k in ('group_id', 'history_id', 'snapshot_tick')}
            guard(name, log=True)
    if len(training) != 3:
        raise ValueError('expected nine training pairs from three snapshots')
    preflight = read(plan['accepted_supervision_preflight'])
    binding = [(ep, name, row) for ep, cases in training.items() for name, row in cases.items()]
    if [(ep, name, row['by_window']['H1-H16']['resolved'], row['by_window']['H1-H16']['strong'])
        for ep, name, row in binding] != [
            (x['episode'], x['direction'], x['h1_h16_resolved'], x['h1_h16_strong']) for x in preflight['coverage']]:
        raise ValueError('frozen early label mask did not reproduce')
    weights = np.asarray(preflight['half_difference_weights'])
    for tag in endpoints:
        errors = np.stack([row['per_step_errors'][tag]['head_error_m_s'] for _, _, row in binding])
        expected = read(plan['accepted_supervision_result'])[{'reference': 'reference_train_loss',
                         'supervised': 'new_train_loss'}[tag]]['half_difference']['rms_error_m_s']
        require_close(np.sqrt(np.sum(weights*np.sum(errors**2, axis=-1))), expected)
    coverage = {name: direction_coverage({k: v[name] for k, v in training.items()}, row) for name, row in dev.items()}
    write_json(out/'late_coverage.json', {'windows': plan['windows'], 'training_metadata': metadata,
        'development_case': capture['case'], 'training': training, 'development': dev, 'direction_coverage': coverage})
    return {'training_pairs': len(binding), 'development_pairs': len(dev),
            'early_mask_and_weighted_head_rmse_reproduced': True, 'archived_development_scores_reproduced': True,
            'archived_one_step_scores_reproduced': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
