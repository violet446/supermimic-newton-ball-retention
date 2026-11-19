"""Prepare the fixed training data intervention and shared normalization."""
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from .archive_restore_plan import _verified_bytes
from .group_fold import make_group_fold
from .model_data import KNOWN_NAMES, prepare_dataset
from .pair_collection import prepare_collection
from .timed_pair_data import concatenate_batches, prepare_timed_pairs


DESIGN = 'configs/s3_timing_coverage_fit_plan.json'
DESIGN_SHA256 = 'eb38c904f8dfba58e922f15ba3bc634e48c7a04645ff0959ea42505887894094'


def pair_cells(batch):
    cells = {}
    for i, key in enumerate(zip(batch.group_ids.tolist(), batch.episode_ids.tolist(), batch.snapshot_ticks.tolist())):
        cells.setdefault(key, []).append(i)
    return cells


def _add_pairs(control, timed):
    base = control.pairs
    old_cells, additions = pair_cells(base), pair_cells(timed.batch)
    if not additions.keys() <= old_cells.keys():
        raise ValueError('timed coverage must stay in pre-existing training cells')
    old_weights = control.pair_weights.copy()
    new_weights = np.empty(len(timed.batch.initial))
    cell_mass = {}
    for key, new_rows in additions.items():
        rows = old_cells[key]
        mass = float(old_weights[rows].sum())
        shared = mass/(len(rows)+len(new_rows))
        old_weights[rows] = shared
        new_weights[new_rows] = shared
        cell_mass['/'.join(map(str, key))] = {'held_rows': len(rows), 'timed_rows': len(new_rows),
                                             'before': mass, 'after': float(old_weights[rows].sum()+new_weights[new_rows].sum())}
    weights = np.concatenate((old_weights, new_weights))
    batch = concatenate_batches(base, timed.batch)
    np.testing.assert_allclose(weights.sum(), 1., rtol=0., atol=2e-15)
    groups = {}
    for group in np.unique(base.group_ids):
        initial = float(control.pair_weights[base.group_ids == group].sum())
        augmented = float(weights[batch.group_ids == group].sum())
        np.testing.assert_allclose(augmented, initial, rtol=0., atol=2e-15)
        groups[str(group)] = {'held_offsets': initial, 'held_offsets_plus_timing': augmented}
    return replace(control, pairs=batch, pair_weights=weights, retained_indices=np.arange(len(weights)),
                   report={**control.report, 'retained_axis_amplitude_pairs': len(base.initial),
                           'retained_timed_pairs': len(timed.batch.initial), 'normalization_frozen_from_control': True}), {
                               'touched_cells': cell_mass, 'group_weight_mass': groups}


def prepare(root, guard):
    """Verify source scope and prepare matched folds without fitting or opening test trajectories."""
    root = Path(root).resolve()
    plan = json.loads(_verified_bytes(root, DESIGN, DESIGN_SHA256))
    inputs = {name: _verified_bytes(root, name, digest) for name, digest in plan['inputs_sha256'].items()}
    cfg = json.loads(inputs[plan['hyperparameter_source']])
    if (cfg['seed'] != plan['seed'] or cfg['network'] != [29, 32, 32, 9]
            or cfg['normalization'] != 'retained_training_v2' or cfg['verification_update'] != plan['updates_per_arm']
            or plan['total_updates'] != 2*plan['updates_per_arm'] or cfg['physical_scales'] != plan['physical_scales']
            or cfg['s2_manifest'] != plan['s2_manifest']):
        raise ValueError('ablation recipe differs from the pinned training protocol')
    accepted = json.loads(inputs[plan['timed_validation']])
    timing_path = Path(plan['timed_manifest'])
    if (accepted['status'] != 'completed' or accepted['branches'] != 16 or accepted['training_updates'] != 0
            or accepted['artifacts_sha256'][timing_path.name] != plan['inputs_sha256'][plan['timed_manifest']]):
        raise ValueError('timed training manifest lacks matching capture acceptance')
    timed = prepare_timed_pairs(root, plan['timed_manifest'], plan['inputs_sha256'][plan['timed_manifest']])
    legacy = prepare_collection(root, root/plan['base_collection'])
    guard()
    s2_spec = json.loads(inputs[plan['s2_manifest']])
    anchors = prepare_dataset((root/plan['s2_manifest']).parent, horizons=tuple(cfg['anchors']['horizons']), verify_test=False)
    episode_groups = {e['episode_id']: e['group_id'] for e in s2_spec['entries'] if e['split'] == 'train'}
    if (legacy.model_config != anchors.model_config or timed.model_config != legacy.model_config
            or legacy.dt_s != anchors.dt_s or timed.dt_s != legacy.dt_s
            or timed.batch.plus_target.shape[1] != legacy.horizon or set(timed.batch.group_ids) != set(plan['timed_train_groups'])
            or set(timed.batch.group_ids) & set(legacy.validation.batch.group_ids)
            or plan['held_group'] in (set(legacy.train.batch.group_ids) | set(episode_groups.values()))
            or plan['held_group'] not in set(legacy.validation.batch.group_ids)):
        raise ValueError('training scope or physical configuration changed')
    base_spec = json.loads(inputs[plan['base_collection']])
    target, = [source for source in base_spec['sources'] if source['cohort'] == 'history']
    history_path = Path(target['manifest'])
    history = json.loads(_verified_bytes(root, target['manifest'], target['sha256']))
    history_entries = {entry['episode_id']: entry for entry in history['entries']}
    for entry in timed.entries:
        parent = history_entries[entry['episode_id']]
        if (parent['split'] != 'train' or parent['group_id'] != entry['group_id']
                or parent['config_sha256'] != entry['source_config_sha256'] or parent['sha256'] != entry['source_trace_sha256']
                or (history_path.parent/parent['config_path']).as_posix() != entry['source_config']
                or entry['snapshot_tick'] not in parent['snapshot_ticks']):
            raise ValueError('timed identity disagrees with the existing train history')

    baseline = make_group_fold(anchors.train, legacy.train.batch, episode_groups, normalization=cfg['normalization'])
    np.testing.assert_allclose(baseline.pair_weights, legacy.train.sample_weights, rtol=1e-14, atol=0.)
    with np.load(BytesIO(inputs[plan['control_checkpoint']]), allow_pickle=False) as saved:
        if int(saved['updates']) != plan['updates_per_arm'] or tuple(saved['known_names']) != tuple(KNOWN_NAMES):
            raise ValueError('control checkpoint endpoint or feature order differs from the plan')
        for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(getattr(baseline, key), saved[key], err_msg=f'control normalization {key}')
    augmented, weights = _add_pairs(baseline, timed)
    folds = {'held_offsets': baseline, 'held_offsets_plus_timing': augmented}
    if ({name: len(fold.pairs.initial) for name, fold in folds.items()} != plan['arms']
            or len(pair_cells(baseline.pairs)) != plan['training_cells']
            or {key: legacy.report['splits']['validation'][key] for key in cfg['expected_validation']} != cfg['expected_validation']):
        raise ValueError('ablation sizes disagree with the existing training or validation contract')

    held = json.loads(inputs[plan['held_timing_run']])
    held_path = Path(plan['held_timing_run']).parent
    if held['status'] != 'completed' or held['case']['group'] != plan['held_group'] or held['case']['tick'] != 168:
        raise ValueError('frozen timed validation case changed')
    for label in held['branches']:
        name = f'worker/{label}.json'
        _verified_bytes(root, held_path/name, held['artifacts_sha256'][name])
    guard()
    return plan, cfg, json.loads(inputs[plan['physical_scales']]), folds, timed, legacy, anchors, weights
