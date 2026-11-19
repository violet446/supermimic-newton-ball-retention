"""Fit the frozen local-surface representation and compare fixed endpoints."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from ball_retention.geometry import load_tool
from ball_retention.group_fold import make_group_fold
from ball_retention.guard import process_memory
from ball_retention.local_surface import SurfaceField
from ball_retention.local_surface_model import LocalSurfaceResidualMLP
from ball_retention.model_data import KNOWN_NAMES, prepare_dataset
from ball_retention.pair_collection import prepare_collection
from ball_retention.pair_loss import paired_loss
from ball_retention.resources import directory_bytes, snapshot
from ball_retention.surface_diagnostic import SurfaceDiagnostic
from ball_retention.world_model import Adam, ResidualMLP
from check_s3_group_generalization import sha, write
from diagnose_s3_position_drift import geometry, project_tangent, tool_omega
from preflight_s3_local_surface import check_action_gradient, run_preflight
from train_s3_history import regression, report
from train_s3_history_duration import development


PLAN = ROOT/'configs/s3_local_surface_refit_plan.json'
OUT = ROOT/'runs/s3_local_surface_refit_20261114'


def fit(model, fold, collection, cfg, units, updates, out, guard, deadline):
    windows = {h: (model.normalize(w.initial), (w.known-model.known_mean)/model.known_scale,
                   model.normalize(w.target)) for h, w in fold.anchors.items()}
    b = fold.pairs
    initial = model.normalize(b.initial)
    known = (np.stack((b.plus_known, b.minus_known))-model.known_mean)/model.known_scale
    target = np.stack((b.plus_target, b.minus_target))
    mask = np.stack((b.plus_mask, b.minus_mask))
    horizons = cfg['anchors']['horizons']
    settings = cfg['pairs']
    leads = np.zeros(collection.horizon)
    leads[np.array(settings['lead_steps'])-1] = settings['lead_weights']
    components = np.array(cfg['components'])
    response_units = np.array(cfg['response_scales_si'])
    rng = np.random.default_rng(cfg['seed'])
    optimizer = Adam(model.params, learning_rate=cfg['optimizer']['learning_rate'],
                     max_grad_norm=cfg['optimizer']['max_grad_norm'])
    with (out/'training.jsonl').open('x', encoding='utf-8') as log:
        for update in range(1, updates+1):
            if time.monotonic() >= deadline:
                raise RuntimeError('training wall cap')
            x, k, y = windows[horizons[(update-1) % len(horizons)]]
            take = rng.integers(len(x), size=cfg['anchors']['batch_size'])
            loss, grads = model.loss_and_grad(x[take], k[take], y[take], state_weights=components,
                                             state_error_scale_si=units)
            pair, pair_grads = paired_loss(
                model, initial, known, target, mask, b.response_mask, b.noise_mask, b.response_floor,
                leads, components, response_units, settings['absolute_weight'], settings['response_weight'],
                sample_weights=fold.pair_weights, absolute_scale_si=units)
            norm = optimizer.step({key: grad+pair_grads[key] for key, grad in grads.items()})
            if not all(np.isfinite(array).all() for array in model.params.values()):
                raise FloatingPointError('nonfinite parameter')
            if update % 30 == 0:
                log.write(json.dumps({'update': update, 'anchor_loss': loss, 'pair': pair,
                                      'gradient_norm': norm}, allow_nan=False)+'\n')
                log.flush()
                guard('fit', update)
    return optimizer.iteration


def drift_report(model, km, ks, collection, mesh, histories):
    mixed = collection.validation
    b = mixed.batch
    cfg, dt = collection.model_config, collection.dt_s
    if not b.pair_mask.all():
        raise ValueError('drift report requires complete paired branches')
    diagnostic = SurfaceDiagnostic(mesh, cfg['ball']['radius_m'], cfg['initial']['tool_rotation_world'])
    initial = np.concatenate((b.initial, b.initial))
    known = np.concatenate((b.plus_known, b.minus_known))
    target = np.concatenate((b.plus_target, b.minus_target))
    free = model.denormalize(model.rollout(model.normalize(initial), (known-km)/ks))
    truth = np.concatenate((initial[:, None], target), axis=1)
    prediction = np.concatenate((initial[:, None], free), axis=1)
    offsets = np.eye(2)[b.axes]*mixed.amplitudes[:, None]
    controls = np.concatenate((b.nominal_actions+offsets[:, None], b.nominal_actions-offsets[:, None]))
    cache = {}
    examples = []
    for i, (group, episode, tick, axis, amplitude, cohort) in enumerate(zip(
            b.group_ids, b.episode_ids, b.snapshot_ticks, b.axes, mixed.amplitudes, mixed.cohorts)):
        sides = []
        for j in (i, i+len(b.initial)):
            omega = tool_omega(b.initial_state[i], controls[j], cfg, dt)
            normals, actual = geometry(truth[j], omega, diagnostic, cache)
            _, predicted = geometry(prediction[j], omega, diagnostic, cache)
            velocity_error = prediction[j, :-1, 3:6]-truth[j, :-1, 3:6]
            tangent_term = dt*project_tangent(velocity_error, normals[:-1]).sum(axis=0)
            sides.append({'position_error_m': (free[j, -1, :3]-target[j, -1, :3]).tolist(),
                          'tangent_velocity_term_m': tangent_term.tolist(),
                          'predicted_gap_m': predicted['gap_m'][-1], 'reference_gap_m': actual['gap_m'][-1],
                          'normal_cosine': float(np.dot(predicted['normal_tool'][-1], actual['normal_tool'][-1])),
                          'predicted_slip_proxy_m_s': float(np.linalg.norm(predicted['tangential_slip_proxy_m_s'][-1])),
                          'reference_slip_proxy_m_s': float(np.linalg.norm(actual['tangential_slip_proxy_m_s'][-1]))})
        examples.append({'group': str(group), 'episode': str(episode), 'tick': int(tick), 'axis': int(axis),
                         'amplitude': float(amplitude), 'history': histories.get(str(episode), 'original'),
                         'cohort': 'new' if cohort == 'history' else 'old', 'branches': sides})

    buckets = {'all': examples}
    for field in ('cohort', 'group', 'history'):
        for label in sorted({r[field] for r in examples}):
            buckets[f'{field}/{label}'] = [r for r in examples if r[field] == label]
    summary = {}
    for label, rows in buckets.items():
        branches = [side for row in rows for side in row['branches']]
        def pair_mean_rmse(key):
            vector = np.array([[side[key] for side in row['branches']] for row in rows]).mean(axis=1)
            return float(np.sqrt(np.mean(np.sum(vector**2, axis=1))))
        summary[label] = {'pairs': len(rows), 'snapshots': len({(r['episode'], r['tick']) for r in rows}),
                          'pair_mean_position_rmse_m': pair_mean_rmse('position_error_m'),
                          'cumulative_tangent_velocity_term_rmse_m': pair_mean_rmse('tangent_velocity_term_m')}
        for key in ('predicted_gap_m', 'reference_gap_m', 'normal_cosine',
                    'predicted_slip_proxy_m_s', 'reference_slip_proxy_m_s'):
            values = [side[key] for side in branches]
            summary[label][key] = {'min': min(values), 'median': float(np.median(values)), 'max': max(values)}
    return {'horizon': 32, 'summary': summary, 'pairs': examples,
            'geometry_contact_labels': False, 'causal_rollout': True}


def run(out):
    started = time.monotonic()
    design = json.loads(PLAN.read_bytes())
    budget = design['budget']
    deadline = started+budget['wall_s']
    out.mkdir(exist_ok=False)
    peaks = {'peak_rss_bytes': 0, 'peak_private_commit_bytes': 0}
    last = {}

    def guard(stage, update=None):
        last.update(stage=stage, update=update)
        memory = process_memory(os.getpid())
        machine = snapshot(ROOT)
        elapsed = time.monotonic()-started
        for key in peaks:
            peaks[key] = max(peaks[key], memory[key])
        size = directory_bytes(out)
        with (out/'resources.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps({**last, 'wall_s': elapsed, 'output_bytes': size, **memory, **machine})+'\n')
        if (elapsed >= budget['wall_s'] or peaks['peak_rss_bytes'] > budget['rss_gib']*1024**3
                or peaks['peak_private_commit_bytes'] > budget['commit_gib']*1024**3
                or machine['available_ram_bytes'] < budget['minimum_available_ram_gib']*1024**3
                or machine['disk_free_bytes'] < budget['output_mib']*1024**2
                or directory_bytes(out) > budget['output_mib']*1024**2):
            raise RuntimeError(f'resource cap at {stage}, update {update}')

    def verify_hashes(hashes):
        for path, digest in hashes.items():
            if sha(ROOT/path) != digest:
                raise ValueError(f'plan hash mismatch: {path}')

    try:
        guard('prepare')
        verify_hashes(design['sha256'])
        candidate = design['candidate']
        cfg = json.loads((ROOT/candidate['base_fit_plan']).read_bytes())
        verify_hashes(cfg['inputs_sha256'])
        if cfg['seed'] != candidate['seed'] or cfg['updates_per_fit'] != candidate['updates']:
            raise ValueError('experiment seed or endpoint mismatch')
        fit_spec = cfg['fits'][candidate['fit_split']]
        manifest_path = ROOT/cfg['s2_manifest']
        data = prepare_dataset(manifest_path.parent, horizons=cfg['anchors']['horizons'], verify_test=False)
        collection = prepare_collection(ROOT, ROOT/fit_spec['collection'])
        if collection.model_config != data.model_config or collection.dt_s != data.dt_s:
            raise ValueError('physical configuration mismatch')
        for split, expected in (('train', fit_spec['expected_train']), ('validation', cfg['expected_validation'])):
            if any(collection.report['splits'][split][key] != value for key, value in expected.items()):
                raise ValueError(f'unexpected {split} split')
        manifest = json.loads(manifest_path.read_bytes())
        mapping = {e['episode_id']: e['group_id'] for e in manifest['entries'] if e['split'] == 'train'}
        fold = make_group_fold(data.train, collection.train.batch, mapping, normalization=cfg['normalization'])
        np.testing.assert_allclose(fold.pair_weights, collection.train.sample_weights, rtol=1e-14, atol=0.)
        physical = json.loads((ROOT/cfg['physical_scales']).read_bytes())
        units = np.array(physical['state_error_scale_si'])
        gain = np.array(physical['residual_gain_si'])
        mesh = load_tool(ROOT/design['surface_field']['mesh'])
        model_cfg = collection.model_config
        field = SurfaceField.from_mesh(mesh, model_cfg['ball']['radius_m'],
                                       model_cfg['initial']['tool_rotation_world'], design['surface_field'],
                                       guard=lambda stage: guard(f'build_field_{stage}'))
        field_path = out/'field.npz'
        np.savez_compressed(field_path, low=field.low, spacing=field.spacing,
                            coefficients=field.coefficients, radius=field.radius)
        field_digest = sha(field_path)
        with np.load(field_path, allow_pickle=False) as saved:
            field = SurfaceField(saved['low'], float(saved['spacing']), saved['coefficients'], float(saved['radius']))
        sources = list((ROOT/'src/ball_retention').glob('*.py'))
        sources += [ROOT/'scripts'/name for name in ('train_s3_local_surface.py', 'preflight_s3_local_surface.py',
                    'check_s3_group_generalization.py', 'train_s3_history.py', 'train_s3_history_duration.py',
                    'diagnose_s3_position_drift.py', 'evaluate_s3_history.py', 'evaluate_s3_coverage.py')]
        source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
        design_digest = sha(PLAN)
        write(out/'plan.json', {'design_sha256': design_digest, 'config': design, 'source_sha256': source_hashes,
              'python': sys.version, 'packages': {name: importlib.metadata.version(name)
              for name in ('numpy', 'scipy', 'trimesh')}, 'field_sha256': field_digest,
              'field_coefficients_shape': list(field.coefficients.shape),
              'field_coefficients_bytes': field.coefficients.nbytes})
        write(out/'training_data.json', {'fold': fold.report, 'collection': collection.report})
        guard('preflight')
        model = LocalSurfaceResidualMLP(field, model_cfg, fold.known_mean, fold.known_scale,
                                       fold.state_mean, fold.state_scale, data.dt_s, candidate['seed'],
                                       hidden=candidate['network'][1], residual_gain=gain/fold.state_scale)
        precheck = run_preflight(model, collection, mesh, candidate['seed'], guard=guard)
        write(out/'preflight.json', precheck)
        guard('before_fit', 0)
        verify_hashes(source_hashes)
        print('Preflight passed. Starting the single fixed fit.', flush=True)
        updates = fit(model, fold, collection, cfg, units, candidate['updates'], out, guard, deadline)
        guard('save', updates)
        checkpoint = out/'candidate_1800.npz'
        arrays = model.state_dict()
        arrays.update(known_mean=fold.known_mean, known_scale=fold.known_scale, known_names=np.array(KNOWN_NAMES),
                      state_error_scale_si=units, residual_gain_si=gain, updates=np.array(updates),
                      field_sha256=np.array(field_digest))
        np.savez_compressed(checkpoint, **arrays)
        if sha(field_path) != field_digest:
            raise ValueError('field hash changed')
        with np.load(checkpoint, allow_pickle=False) as saved:
            restored = LocalSurfaceResidualMLP.from_checkpoint(saved, field, model_cfg)
        b = collection.train.batch
        initial = model.normalize(b.initial)
        known = (b.plus_known-fold.known_mean)/fold.known_scale
        np.testing.assert_array_equal(restored.rollout(initial, known), model.rollout(initial, known))
        write(out/'trained_gradient_check.json', {'model_only': True, 'checks':
              check_action_gradient(restored, collection, candidate['seed'])})
        print('Fixed endpoint complete. Evaluating the three checkpoints.', flush=True)

        histories = {e['episode_id']: e['history_id'] for e in
                     json.loads((ROOT/cfg['history_plan']).read_bytes())['episodes']}
        reference = json.loads((ROOT/'runs/s3_history_duration_20261114/comparison.json').read_bytes())['models']
        drift_reference = json.loads((ROOT/'runs/s3_position_drift_20261114/findings.json').read_bytes())['summary']
        comparison, drifts = {}, {}
        baseline_paths = {'old_900': design['control']['original_guard_reference'],
                          'expanded_1800': design['control']['expanded_same_updates']}
        checkpoints = {**baseline_paths, 'candidate_1800': checkpoint.relative_to(ROOT).as_posix()}
        for name, path in checkpoints.items():
            guard(f'evaluate_{name}', updates)
            with np.load(ROOT/path, allow_pickle=False) as saved:
                evaluated = restored if name == 'candidate_1800' else ResidualMLP.from_state_dict(saved)
                km, ks = saved['known_mean'], saved['known_scale']
            results = report(evaluated, km, ks, collection, data, histories)
            results['development'] = development(results)
            if name in reference and results != reference[name]:
                raise ValueError(f'{name} does not exactly reproduce prior development metrics')
            comparison[name] = results
            drift = drift_report(evaluated, km, ks, collection, mesh, histories)
            if name in drift_reference:
                for cohort in ('old', 'new'):
                    previous = drift_reference[name][cohort]
                    actual = drift['summary'][f'cohort/{cohort}']
                    np.testing.assert_allclose(actual['pair_mean_position_rmse_m'],
                                               previous['free_position_rmse_m']['32'], rtol=1e-12)
                    np.testing.assert_allclose(actual['cumulative_tangent_velocity_term_rmse_m'],
                        previous['H32_account']['tangent_velocity_cumsum_m']['vector_rmse_m'], rtol=1e-12)
            drifts[name] = drift
        gates = {name: regression(comparison[name], comparison['candidate_1800']) for name in baseline_paths}
        if any(len(gate['checks']) != 22 for gate in gates.values()):
            raise ValueError('expected 22 protections per baseline')
        write(out/'comparison.json', {'models': comparison, 'regression': gates})
        write(out/'drift.json', drifts)
        verify_hashes({**design['sha256'], **cfg['inputs_sha256'], **source_hashes})
        if sha(PLAN) != design_digest or sha(field_path) != field_digest:
            raise ValueError('design or field changed during execution')
        guard('complete', updates)
        write(out/'summary.json', {'status': 'completed', 'updates': updates, 'wall_s': time.monotonic()-started,
              **peaks, 'regression': gates, 'preflight_passed': True, 'checkpoint_reload_equal': True,
              'baseline_metrics_reproduced': True, 'checkpoint_selection': False, 'test_episode_files_read': False,
              'simulation_steps': 0, 'control_accepted': False,
              'artifacts_sha256': {p.name: sha(p) for p in sorted(out.iterdir())}})
    except Exception as error:
        write(out/'failure.json', {'status': 'stopped', **last, **peaks,
              'wall_s': time.monotonic()-started, 'error': str(error)})
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT/'runs')
    run(output)
