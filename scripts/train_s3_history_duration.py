"""Compare two fixed training durations with old and expanded pair coverage."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from ball_retention.group_fold import make_group_fold
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES, prepare_dataset
from ball_retention.pair_collection import prepare_collection
from ball_retention.pair_loss import paired_loss
from ball_retention.resources import directory_bytes, snapshot
from ball_retention.world_model import Adam, ResidualMLP
from check_s3_group_generalization import sha, write
from evaluate_s3_history import aggregate
from train_s3_history import regression, report


PLAN = ROOT/'configs/s3_history_duration_plan.json'


def fit(out, name, spec, cfg, collection, data, mapping, physical, guard, deadline):
    fold = make_group_fold(data.train, collection.train.batch, mapping,
                           normalization=cfg['normalization'])
    if any(collection.report['splits']['train'][key] != value
           for key, value in spec['expected_train'].items()):
        raise ValueError(f'{name}: unexpected training split')
    np.testing.assert_allclose(fold.pair_weights, collection.train.sample_weights, rtol=1e-14, atol=0.)
    write(out/f'{name}_training_data.json', {'fold': fold.report, 'collection': collection.report})

    units = np.array(physical['state_error_scale_si'])
    gain = np.array(physical['residual_gain_si'])
    model = ResidualMLP(len(KNOWN_NAMES), fold.state_mean, fold.state_scale, data.dt_s,
                        cfg['seed'], hidden=cfg['network'][1], residual_gain=gain/fold.state_scale)

    def known(values):
        return (values-fold.known_mean)/fold.known_scale

    windows = {h: (model.normalize(w.initial), known(w.known), model.normalize(w.target))
               for h,w in fold.anchors.items()}
    b = fold.pairs
    initial = model.normalize(b.initial)
    pair_known = known(np.stack((b.plus_known, b.minus_known)))
    target = np.stack((b.plus_target, b.minus_target))
    mask = np.stack((b.plus_mask, b.minus_mask))
    horizons = cfg['anchors']['horizons']
    pairs = cfg['pairs']
    leads = np.zeros(collection.horizon)
    leads[np.array(pairs['lead_steps'])-1] = pairs['lead_weights']
    components = np.array(cfg['components'])
    response_units = np.array(cfg['response_scales_si'])
    rng = np.random.default_rng(cfg['seed'])
    optimizer = Adam(model.params, learning_rate=cfg['optimizer']['learning_rate'],
                     max_grad_norm=cfg['optimizer']['max_grad_norm'])

    def checkpoint_arrays(update):
        return {**model.state_dict(), 'known_mean': fold.known_mean, 'known_scale': fold.known_scale,
                'known_names': np.array(KNOWN_NAMES), 'state_error_scale_si': units,
                'residual_gain_si': gain, 'updates': np.array(update)}

    fit_started = time.monotonic()
    with (out/f'{name}_training.jsonl').open('x', encoding='utf-8') as log:
        for update in range(1, cfg['updates_per_fit']+1):
            if time.monotonic() >= deadline:
                raise RuntimeError('wall cap')
            x, k, y = windows[horizons[(update-1)%len(horizons)]]
            take = rng.integers(len(x), size=cfg['anchors']['batch_size'])
            loss, grads = model.loss_and_grad(x[take], k[take], y[take], state_weights=components,
                                             state_error_scale_si=units)
            pair, pair_grads = paired_loss(
                model, initial, pair_known, target, mask, b.response_mask, b.noise_mask, b.response_floor,
                leads, components, response_units, pairs['absolute_weight'], pairs['response_weight'],
                sample_weights=fold.pair_weights, absolute_scale_si=units)
            norm = optimizer.step({key: grad+pair_grads[key] for key,grad in grads.items()})
            if update % 30 == 0:
                log.write(json.dumps({'update': update, 'anchor_loss': loss, 'pair': pair,
                                      'gradient_norm': norm}, allow_nan=False)+'\n')
                log.flush()
                guard(name, update)
            if update == cfg['verification_update']:
                arrays = checkpoint_arrays(update)
                with np.load(ROOT/spec['reference_900'], allow_pickle=False) as saved:
                    for key, array in arrays.items():
                        np.testing.assert_array_equal(array, saved[key], err_msg=f'{name}: update {update} {key}')
                write(out/f'{name}_reproduction.json', {'update': update, 'array_keys': sorted(arrays),
                    'reference_sha256': sha(ROOT/spec['reference_900']), 'array_equal': True})
                print(f'{name}: exact match at update {update}', flush=True)

    checkpoint = out/f'{name}_1800.npz'
    np.savez_compressed(checkpoint, **checkpoint_arrays(optimizer.iteration))
    with np.load(checkpoint, allow_pickle=False) as saved:
        restored = ResidualMLP.from_state_dict(saved)
        np.testing.assert_array_equal(model.rollout(initial, pair_known[0]), restored.rollout(initial, pair_known[0]))
    guard(name, optimizer.iteration)
    print(f'{name}: completed {optimizer.iteration} updates', flush=True)
    return checkpoint, time.monotonic()-fit_started


def development(result):
    rows = [row for row in result['pairs'] if row['split'] == 'validation']
    buckets = {
        'all': rows,
        'old': [row for row in rows if row['cohort'] != 'history'],
        'new': [row for row in rows if row['cohort'] == 'history'],
    }
    return {label: aggregate(take, (8, 16, 32), .5, 10) for label,take in buckets.items()}


def run(out):
    started = time.monotonic()
    cfg = json.loads(PLAN.read_bytes())
    out.mkdir(exist_ok=False)
    budget = cfg['budget']
    deadline = started+budget['wall_s']
    latest = {}
    peaks = {'peak_rss_bytes': 0, 'peak_private_commit_bytes': 0}

    def guard(stage, update=None):
        latest.update(stage=stage, update=update)
        memory = process_memory(os.getpid())
        machine = snapshot(ROOT)
        wall = time.monotonic()-started
        for key in peaks:
            peaks[key] = max(peaks[key], memory[key])
        size = directory_bytes(out)
        with (out/'resources.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps({**latest, 'wall_s': wall, 'output_bytes': size, **memory, **machine})+'\n')
        if (wall >= budget['wall_s'] or peaks['peak_rss_bytes'] > budget['rss_gib']*1024**3
                or peaks['peak_private_commit_bytes'] > budget['commit_gib']*1024**3
                or machine['available_ram_bytes'] < budget['min_available_ram_gib']*1024**3
                or machine['disk_free_bytes'] < budget['output_mib']*1024**2
                or directory_bytes(out) > budget['output_mib']*1024**2):
            raise RuntimeError(f'resource cap at {stage}, update {update}')

    def verify_hashes():
        for path, digest in {**cfg['inputs_sha256'], **cfg['source_sha256']}.items():
            if sha(ROOT/path) != digest:
                raise ValueError(f'plan hash mismatch: {path}')

    try:
        guard('prepare')
        verify_hashes()
        plan_digest = sha(PLAN)
        write(out/'plan.json', {'design_sha256': plan_digest, 'config': cfg})
        physical = json.loads((ROOT/cfg['physical_scales']).read_bytes())
        manifest_path = ROOT/cfg['s2_manifest']
        data = prepare_dataset(manifest_path.parent, horizons=cfg['anchors']['horizons'], verify_test=False)
        manifest = json.loads(manifest_path.read_bytes())
        mapping = {e['episode_id']: e['group_id'] for e in manifest['entries'] if e['split'] == 'train'}
        collections = {name: prepare_collection(ROOT, ROOT/spec['collection'])
                       for name,spec in cfg['fits'].items()}
        for name, collection in collections.items():
            if collection.model_config != data.model_config or collection.dt_s != data.dt_s:
                raise ValueError(f'{name}: physical configuration mismatch')
        common = collections[cfg['evaluation_collection']]
        if any(common.report['splits']['validation'][key] != value
               for key, value in cfg['expected_validation'].items()):
            raise ValueError('unexpected development validation split')
        guard('prepared')

        checkpoints, fit_seconds = {}, {}
        for name,spec in cfg['fits'].items():
            checkpoint, fit_seconds[name] = fit(out, name, spec, cfg, collections[name], data, mapping,
                                                physical, guard, deadline)
            checkpoints[f'{name}_900'] = ROOT/spec['reference_900']
            checkpoints[f'{name}_1800'] = checkpoint

        histories = {e['episode_id']: e['history_id'] for e in
                     json.loads((ROOT/cfg['history_plan']).read_bytes())['episodes']}
        results = {}
        for name, checkpoint in checkpoints.items():
            guard(f'evaluate_{name}')
            with np.load(checkpoint, allow_pickle=False) as saved:
                result = report(ResidualMLP.from_state_dict(saved), saved['known_mean'], saved['known_scale'],
                                common, data, histories)
            result['development'] = development(result)
            results[name] = result

        comparisons = {f'{before}_to_{after}': regression(results[before], results[after])
                       for before,after in cfg['comparisons']}
        write(out/'comparison.json', {'models': results, 'regression': comparisons})
        verify_hashes()
        if sha(PLAN) != plan_digest:
            raise ValueError('experiment plan changed during execution')
        guard('complete')
        write(out/'summary.json', {'status': 'completed', 'total_updates': len(fit_seconds)*cfg['updates_per_fit'],
              'fit_seconds': fit_seconds, 'wall_s': time.monotonic()-started, **peaks,
              'checkpoint_sha256': {name: sha(path) for name,path in checkpoints.items()},
              'regression': comparisons, 'reproduction_900_array_equal': True, 'reload_1800_equal': True,
              'checkpoint_selection': False, 'test_episode_files_read': False, 'test_metrics_used': False,
              'simulation_steps': 0, 'control_accepted': False})
    except Exception as error:
        write(out/'failure.json', {'status': 'stopped', **latest, **peaks,
              'wall_s': time.monotonic()-started, 'error': str(error)})
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/s3_history_duration_20261114')
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT/'runs')
    run(output)
