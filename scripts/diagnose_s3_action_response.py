"""Compare frozen action Jacobians with matched model and Newton secants."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from ball_retention.action_eval import COMPONENTS, evaluate_pair
from ball_retention.action_grad import action_sensitivity, predict_ball
from ball_retention.guard import process_memory
from ball_retention.local_surface import SurfaceField
from ball_retention.local_surface_model import LocalSurfaceResidualMLP
from ball_retention.pair_collection import prepare_collection
from ball_retention.resources import directory_bytes, snapshot
from ball_retention.world_model import ResidualMLP
from check_s3_group_generalization import sha, write
from summarize_s3_action_response import summarize


PLAN = ROOT/'configs/s3_action_response_diagnosis_plan.json'
OUT = ROOT/'runs/s3_action_response_diagnosis_20261114'
PAIR_ID = ('group', 'episode', 'tick', 'axis', 'amplitude')


def compare(value, reference, value_resolved, reference_resolved):
    value_norm = np.linalg.norm(value)
    reference_norm = np.linalg.norm(reference)
    cosine = (float(np.clip(np.dot(value/value_norm, reference/reference_norm), -1., 1.))
              if value_resolved and reference_resolved else None)
    return {'cosine': cosine,
            'amplitude_ratio': float(value_norm/reference_norm) if reference_resolved else None,
            'relative_error': float(np.linalg.norm(value-reference)/reference_norm) if reference_resolved else None}


def component(metric, jacobian, midpoint_pred, midpoint_newton, amplitude, strong):
    model = np.array(metric['model_derivative'])
    newton = np.array(metric['newton_derivative'])
    threshold = metric['signal_threshold']
    jacobian_resolved = amplitude*np.linalg.norm(jacobian) >= threshold
    physical_resolved = metric['newton_resolved']
    norm = metric['newton_norm']
    signal = 'unresolved'
    if physical_resolved:
        signal = 'strong' if metric['newton_half_spread'] >= strong*threshold else 'near'
    return {'signal': signal, 'half_spread_threshold_si': threshold,
            'replay_half_spread_si': metric['replay_half_spread'],
            'jacobian_si': jacobian.tolist(), 'model_secant_si': model.tolist(), 'newton_secant_si': newton.tolist(),
            'jacobian_resolved': bool(jacobian_resolved), 'model_secant_resolved': metric['model_resolved'],
            'jacobian_vs_newton': compare(jacobian, newton, jacobian_resolved, physical_resolved),
            'secant_vs_newton': compare(model, newton, metric['model_resolved'], physical_resolved),
            'jacobian_vs_secant': compare(jacobian, model, jacobian_resolved, metric['model_resolved']),
            'curvature_to_newton_norm_ratio': float(np.linalg.norm(jacobian-model)/norm) if physical_resolved else None,
            'model_pair_mean_minus_nominal_si': midpoint_pred.tolist(),
            'newton_pair_mean_minus_replay_mean_si': midpoint_newton.tolist()}


def snapshot_jacobians(model, km, ks, collection, batch, cfg, guard):
    snapshots = {}
    maximum_error = 0.
    for i, (episode, tick) in enumerate(zip(batch.episode_ids, batch.snapshot_ticks)):
        key = (str(episode), int(tick))
        state = batch.initial_state[i]
        actions = batch.nominal_actions[i]
        if key in snapshots:
            np.testing.assert_array_equal(actions, snapshots[key]['actions'])
            np.testing.assert_array_equal(batch.initial[i], snapshots[key]['initial'])
            continue
        args = (model, batch.initial[i], state['time_s'], state['tilt_rad'], state['tilt_rate_rad_s'])
        tail = (collection.model_config, km, ks)
        step = cfg['fd_step']
        if (abs(actions)+step >= 1).any():
            raise ValueError(f'nominal numerical derivative crosses a clip boundary: {key}')
        result = action_sensitivity(*args, actions, *tail)
        np.testing.assert_array_equal(result['prediction'], predict_ball(*args, actions, *tail))
        for axis, offset in enumerate(step*np.eye(2)):
            plus = predict_ball(*args, actions+offset, *tail)
            minus = predict_ball(*args, actions-offset, *tail)
            fd = (plus-minus)/(2*step)
            exact = result['jacobian'][..., axis]
            np.testing.assert_allclose(exact, fd, rtol=cfg['fd_rtol'], atol=cfg['fd_atol_si'])
            maximum_error = max(maximum_error, float(np.max(np.abs(exact-fd))))
        snapshots[key] = {**result, 'actions': actions, 'initial': batch.initial[i]}
        guard('jacobian', len(snapshots))
    return snapshots, maximum_error


def cases(name, model, km, ks, collection, histories, reference, cfg, guard):
    mixed = collection.validation
    b = mixed.batch
    snapshots, fd_error = snapshot_jacobians(model, km, ks, collection, b, cfg, guard)
    known = (np.concatenate((b.plus_known, b.minus_known))-km)/ks
    initial = model.normalize(np.concatenate((b.initial, b.initial)))
    plus, minus = np.split(model.denormalize(model.rollout(initial, known)), 2)
    rows = []
    for i, expected in enumerate(reference):
        key = str(b.episode_ids[i]), int(b.snapshot_ticks[i])
        amplitude = float(mixed.amplitudes[i])
        axis = int(b.axes[i])
        identity = {'group': str(b.group_ids[i]), 'episode': key[0], 'tick': key[1], 'axis': axis,
                    'amplitude': amplitude, 'cohort': str(mixed.cohorts[i]),
                    'history': histories.get(key[0], 'original')}
        if any(identity[k] != expected[k] for k in (*PAIR_ID, 'cohort', 'history')):
            raise ValueError(f'frozen pair identity mismatch: {name}, {identity}')
        metrics = evaluate_pair(plus[i], minus[i], b.plus_target[i], b.minus_target[i], amplitude,
                                collection.model_config['initial']['anchor_xy_tool_m'],
                                replay_first=b.baseline_first[i], replay_second=b.baseline_second[i], leads=cfg['leads'])
        nominal = snapshots[key]
        for h in cfg['leads']:
            lead = metrics['by_lead'][str(h)]
            prior = expected['response']['by_lead'][str(h)]
            j = h-1
            row = {**identity, 'model': name, 'lead': h}
            model_even = (plus[i, j]+minus[i, j])*.5-nominal['prediction'][j]
            newton_even = (b.plus_target[i, j]+b.minus_target[i, j]-b.baseline_first[i, j]-b.baseline_second[i, j])*.5
            for label, region in COMPONENTS.items():
                for field in ('model_derivative', 'newton_derivative', 'signal_threshold'):
                    np.testing.assert_allclose(lead[label][field], prior[label][field], rtol=1e-12, atol=1e-15)
                row[label] = component(lead[label], nominal['jacobian'][j, region, axis],
                                       model_even[region], newton_even[region], amplitude, cfg['strong_multiple'])
            rows.append(row)
    return rows, {'snapshots': len(snapshots), 'max_fd_abs_error_si': fd_error}


def dual_amplitudes(rows, cfg):
    buckets = {}
    for row in rows:
        identity = tuple(row[k] for k in ('model', 'lead', 'group', 'episode', 'tick', 'axis', 'cohort', 'history'))
        buckets.setdefault(identity, []).append(row)
    result = []
    for group in buckets.values():
        if len(group) == 1:
            continue
        small, large = sorted(group, key=lambda row: row['amplitude'])
        if [small['amplitude'], large['amplitude']] != cfg['dual_amplitudes']:
            raise ValueError('unexpected dual amplitude identity')
        dual = {k: small[k] for k in ('model', 'lead', 'group', 'episode', 'tick', 'axis', 'cohort', 'history')}
        a, b = small['position'], large['position']
        np.testing.assert_array_equal(a['jacobian_si'], b['jacobian_si'])
        for label, key, resolved in (('newton', 'newton_secant_si', lambda c: c['signal'] != 'unresolved'),
                                      ('model_secant', 'model_secant_si', lambda c: c['model_secant_resolved'])):
            metric = compare(np.array(b[key]), np.array(a[key]), resolved(b), resolved(a))
            metric['stable'] = (metric['cosine'] >= cfg['dual_min_cosine']
                                and metric['relative_error'] <= cfg['dual_max_relative_error']
                                if metric['cosine'] is not None else None)
            dual[label] = metric
        dual['both_newton_strong'] = a['signal'] == b['signal'] == 'strong'
        result.append(dual)
    return result


def main(out):
    started = time.monotonic()
    plan_bytes = PLAN.read_bytes()
    cfg = json.loads(plan_bytes)
    budget = cfg['budget']
    out.mkdir(exist_ok=False)
    stage = {}
    peaks = {'peak_rss_bytes': 0, 'peak_private_commit_bytes': 0}

    def guard(label, completed=None):
        stage.update(stage=label, completed=completed)
        memory = process_memory(os.getpid())
        machine = snapshot(ROOT)
        for key in peaks:
            peaks[key] = max(peaks[key], memory[key])
        wall = time.monotonic()-started
        size = directory_bytes(out)
        with (out/'resources.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps({**stage, 'wall_s': wall, 'output_bytes': size, **memory, **machine})+'\n')
        if (wall >= budget['wall_s'] or peaks['peak_rss_bytes'] > budget['rss_gib']*1024**3
                or peaks['peak_private_commit_bytes'] > budget['commit_gib']*1024**3
                or machine['available_ram_bytes'] < budget['minimum_available_ram_gib']*1024**3
                or machine['disk_free_bytes'] < budget['output_mib']*1024**2 or size > budget['output_mib']*1024**2):
            raise RuntimeError(f'offline budget exceeded at {label}')

    def verify_hashes():
        for name, digest in cfg['sha256'].items():
            if sha(ROOT/name) != digest:
                raise ValueError(f'frozen hash mismatch: {name}')

    try:
        guard('start')
        verify_hashes()
        write(out/'plan.json', {'design_sha256': sha(PLAN), 'config': cfg})
        c = prepare_collection(ROOT, ROOT/cfg['collection'])
        batch = c.validation.batch
        split = c.report['splits']['validation']
        if (c.horizon != 32 or any(split[k] != v for k, v in cfg['expected_validation'].items())
                or not (batch.pair_mask & batch.baseline_mask).all()):
            raise ValueError('expected matched 32-step development and replay leads')
        histories = {e['episode_id']: e['history_id'] for e in
                     json.loads((ROOT/cfg['history_plan']).read_bytes())['episodes']}
        by_model = json.loads((ROOT/cfg['comparison']).read_bytes())['models']
        with np.load(ROOT/cfg['field'], allow_pickle=False) as saved:
            field = SurfaceField(saved['low'], float(saved['spacing']), saved['coefficients'], float(saved['radius']))
        rows, errors = [], {}
        for name, path in cfg['checkpoints'].items():
            guard(name)
            with np.load(ROOT/path, allow_pickle=False) as saved:
                if name == 'candidate_1800':
                    if str(saved['field_sha256']) != cfg['sha256'][cfg['field']]:
                        raise ValueError('local model bound to a different field')
                    model = LocalSurfaceResidualMLP.from_checkpoint(saved, field, c.model_config)
                else:
                    model = ResidualMLP.from_state_dict(saved)
                km, ks = saved['known_mean'], saved['known_scale']
            reference = [row for row in by_model[name]['pairs'] if row['split'] == 'validation']
            if len(reference) != len(batch.initial):
                raise ValueError(f'wrong reference pair count: {name}')
            model_rows, errors[name] = cases(name, model, km, ks, c, histories, reference, cfg, guard)
            rows.extend(model_rows)
            print(f'{name}: verified {len(model_rows)} pair leads', flush=True)
        dual = dual_amplitudes(rows, cfg)
        if len(dual) != cfg['dual_axes_per_model_lead']*len(cfg['checkpoints'])*len(cfg['leads']):
            raise ValueError('wrong number of matched dual amplitude cases')
        write(out/'cases.json', {'pairs': rows, 'dual_amplitude': dual})
        findings = summarize(rows, dual)
        write(out/'findings.json', findings)
        verify_hashes()
        if PLAN.read_bytes() != plan_bytes:
            raise ValueError('diagnostic design changed during execution')
        guard('complete')
        write(out/'summary.json', {'status': 'completed', 'wall_s': time.monotonic()-started, **peaks,
              'validation_pairs_per_model': len(batch.initial), 'pair_leads': len(rows), 'dual_pair_leads': len(dual),
              'jacobian_finite_difference': errors, 'saved_secants_reproduced': True,
              'training_updates': 0, 'simulation_steps': 0, 'test_episode_files_read': False,
              'control_accepted': False, 'artifacts_sha256': {p.name: sha(p) for p in sorted(out.iterdir())}})
    except Exception as error:
        write(out/'failure.json', {'status': 'stopped', **stage, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ROOT/'runs')
    main(out)
