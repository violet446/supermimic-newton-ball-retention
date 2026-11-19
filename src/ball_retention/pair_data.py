"""Read matched Newton action pairs without using future states as inputs."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

from .contracts import RigidState, State
from .model_data import KNOWN_NAMES, known_rollout
from .task import ball_in_tool


SCHEMA = 's3-long-pairs-v1'
DEFAULT_RESPONSE_FLOOR = (1e-6, 1e-5, 1e-4)  # m, m/s, rad/s


@dataclass(frozen=True)
class PairBatch:
    """One row per snapshot and perturbed axis, with a common lead grid.

    Targets retain an observed shortened last row for inspection. A full-step
    mask is false at that row and at every unexecuted lead. Missing target
    values are zero-filled and must be used with their corresponding mask.
    """
    initial: np.ndarray
    plus_known: np.ndarray
    minus_known: np.ndarray
    plus_target: np.ndarray
    minus_target: np.ndarray
    plus_mask: np.ndarray
    minus_mask: np.ndarray
    pair_mask: np.ndarray
    noise_mask: np.ndarray
    response_mask: np.ndarray
    replay_noise: np.ndarray
    response_floor: np.ndarray
    baseline_first: np.ndarray
    baseline_second: np.ndarray
    baseline_mask: np.ndarray
    plus_substeps: np.ndarray
    minus_substeps: np.ndarray
    plus_terminated: np.ndarray
    minus_terminated: np.ndarray
    plus_truncated: np.ndarray
    minus_truncated: np.ndarray
    episode_ids: np.ndarray
    group_ids: np.ndarray
    snapshot_ticks: np.ndarray
    axes: np.ndarray
    nominal_actions: np.ndarray
    initial_state: tuple[dict, ...]


@dataclass(frozen=True)
class PairedDataset:
    dt_s: float  # Full control period, not the Newton substep.
    horizon: int
    amplitude: float
    train: PairBatch
    validation: PairBatch
    model_config: dict
    report: dict


def _verified_bytes(root, name, expected_hash):
    if not isinstance(name, str) or '\\' in name or ':' in name:
        raise ValueError('capture paths must be relative POSIX paths')
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('capture paths must stay inside the manifest directory')
    path = (root / relative).resolve()
    path.relative_to(root)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f'capture hash mismatch: {relative}')
    return data


def _array(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{label} must be finite with shape {shape}')
    return array


def _branch(branch, requested, substep_dt, substeps):
    horizon = len(requested)
    raw_steps = branch['substeps']
    length = len(raw_steps)
    if not 1 <= length <= horizon or any(type(n) is not int or not 1 <= n <= substeps for n in raw_steps):
        raise ValueError('invalid branch substep counts')
    if any(n != substeps for n in raw_steps[:-1]):
        raise ValueError('only the final control step may be shortened')
    terminated, truncated = branch['terminated'], branch['truncated']
    if type(terminated) is not bool or type(truncated) is not bool or (terminated and truncated):
        raise ValueError('invalid branch ending flags')
    if (length < horizon and not (terminated or truncated)) or (raw_steps[-1] < substeps and not terminated):
        raise ValueError('incomplete branch has no matching ending')
    reason = branch['termination_reason']
    if (terminated and (not isinstance(reason, str) or not reason)) or (not terminated and reason is not None):
        raise ValueError('invalid physical termination reason')
    elapsed = _array(branch['elapsed_s'], (length,), 'branch elapsed time')
    if not np.allclose(elapsed, np.cumsum(raw_steps)*substep_dt, rtol=0., atol=1e-9):
        raise ValueError('branch elapsed time disagrees with executed substeps')
    actions = _array(branch['actions'], (length, 2), 'branch actions')
    if not np.allclose(actions, requested[:length], rtol=0., atol=1e-12):
        raise ValueError('branch actions disagree with nominal schedule and offset')
    target = np.zeros((horizon, 9))
    target[:length] = _array(branch['states_raw'], (length, 9), 'branch states')
    counts = np.zeros(horizon, dtype=int)
    counts[:length] = raw_steps
    return target, counts, (terminated, truncated, reason)


def _snapshot_rows(record, entry, cfg, horizon, amplitude, substep_dt, substeps, floor, noise_multiplier,
                   record_amplitudes):
    if (record['scenario'] != 'route' or record['amplitudes'] != record_amplitudes
            or record['physics_dt_s'] != substep_dt or record['control_substeps'] != substeps):
        raise ValueError('snapshot metadata does not match the pair plan')
    raw_state = record['initial_state']
    state_fields = {**raw_state, 'ball': RigidState(**raw_state['ball']), 'tool': RigidState(**raw_state['tool'])}
    state = State(**state_fields)
    state.validate()
    tick = record['snapshot_tick']
    dt_s = substep_dt*substeps
    if not math.isclose(state.time_s, tick*dt_s, rel_tol=0., abs_tol=1e-9):
        raise ValueError('snapshot state time does not match capture tick')
    initial = _array(record['initial_raw'], (9,), 'initial ball state')
    terms = ball_in_tool(state.ball, state.tool)
    if not np.allclose(initial, np.r_[terms['position'], terms['velocity'], terms['ball_omega']], rtol=0., atol=1e-10):
        raise ValueError('initial ball state disagrees with snapshot coordinates')
    nominal = _array(record['nominal_actions'], (horizon, 2), 'nominal actions')
    if np.max(np.abs(nominal)) + max(record_amplitudes) >= 1.:
        raise ValueError('paired perturbation would be clipped')
    branches = {}
    for branch in record['branches']:
        axis, offset, repeat = branch['axis'], branch['offset'], branch['repeat']
        if (axis is not None and type(axis) is not int) or type(repeat) is not int:
            raise ValueError('invalid branch identity')
        key = axis, offset, repeat
        if key in branches:
            raise ValueError('duplicate paired branch')
        branches[key] = branch
    expected = {(None, 0., 0), (None, 0., 1)} | {(axis, sign*size, 0) for axis in (0, 1)
                                                               for size in record_amplitudes for sign in (-1, 1)}
    if branches.keys() != expected:
        raise ValueError('missing or unexpected paired branch')
    # An amplitude view shares the original trace; validate unselected branches
    # as well, and never duplicate their data or split snapshots by amplitude.
    for axis, offset, repeat in expected:
        if axis is not None and abs(offset) != amplitude:
            _branch(branches[axis, offset, repeat], nominal+np.eye(2)[axis]*offset, substep_dt, substeps)
    base_a, base_a_steps, base_a_end = _branch(branches[None, 0., 0], nominal, substep_dt, substeps)
    base_b, base_b_steps, base_b_end = _branch(branches[None, 0., 1], nominal, substep_dt, substeps)
    if not np.array_equal(base_a_steps, base_b_steps) or base_a_end != base_b_end:
        raise ValueError('zero perturbation replays disagree on timing or ending')
    baseline_mask = base_a_steps == substeps
    noise = np.where(baseline_mask[:, None], np.abs(base_a-base_b)/2., 0.)
    response_floor = np.maximum(floor, noise_multiplier*noise)

    for axis in (0, 1):
        offset = np.eye(2)[axis]*amplitude
        plus_actions, minus_actions = nominal+offset, nominal-offset
        plus, plus_steps, plus_end = _branch(branches[axis, amplitude, 0], plus_actions, substep_dt, substeps)
        minus, minus_steps, minus_end = _branch(branches[axis, -amplitude, 0], minus_actions, substep_dt, substeps)
        plus_mask, minus_mask = plus_steps == substeps, minus_steps == substeps
        pair_mask = plus_mask & minus_mask
        noise_mask = pair_mask & baseline_mask
        response_mask = noise_mask[:, None] & (np.abs(plus-minus)/2. >= response_floor)
        yield {'initial': initial, 'plus_known': known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                                                              plus_actions, cfg, dt_s),
               'minus_known': known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                                            minus_actions, cfg, dt_s),
               'plus_target': plus, 'minus_target': minus, 'plus_mask': plus_mask, 'minus_mask': minus_mask,
               'pair_mask': pair_mask, 'noise_mask': noise_mask, 'response_mask': response_mask,
               'replay_noise': noise, 'response_floor': response_floor,
               'baseline_first': base_a, 'baseline_second': base_b, 'baseline_mask': baseline_mask,
               'plus_substeps': plus_steps, 'minus_substeps': minus_steps,
               'plus_terminated': plus_end[0], 'minus_terminated': minus_end[0],
               'plus_truncated': plus_end[1], 'minus_truncated': minus_end[1],
               'episode_ids': entry['episode_id'], 'group_ids': entry['group_id'],
               'snapshot_ticks': tick, 'axes': axis, 'nominal_actions': nominal, 'initial_state': raw_state}


def prepare_pairs(manifest_path, *, noise_multiplier=5., response_floor=DEFAULT_RESPONSE_FLOOR):
    """Verify the capture manifest, align leads and derive causal tool inputs.

    Initial and target ball states and known features are in raw SI units.
    Callers apply the fixed training normalization to these arrays. Component
    response masks compare the measured half spread to per-unit replay noise.
    """
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_bytes())
    horizon, amplitude = manifest['horizon'], float(manifest['amplitude'])
    record_amplitudes = manifest.get('record_amplitudes', [amplitude])
    if (not isinstance(record_amplitudes, list) or not record_amplitudes
            or any(not isinstance(a, (int, float)) or not math.isfinite(a) or not 0 < a < 1 for a in record_amplitudes)
            or len(set(record_amplitudes)) != len(record_amplitudes) or amplitude not in record_amplitudes):
        raise ValueError('invalid recorded amplitude set')
    substep_dt, substeps = float(manifest['dt_s']), manifest['control_substeps']
    if (manifest['schema'] != SCHEMA or type(horizon) is not int or horizon < 1
            or not 0 < amplitude < 1 or not math.isfinite(substep_dt) or substep_dt <= 0
            or type(substeps) is not int or substeps < 1):
        raise ValueError('invalid paired capture manifest or clock')
    floor = np.repeat(_array(response_floor, (3,), 'response floors'), 3)
    if np.any(floor <= 0) or not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError('noise floors and multiplier must be positive')

    parts = {'train': [], 'validation': []}
    groups, episodes, trace_splits = {}, set(), {}
    model_config = None
    for entry in manifest['entries']:
        split, group, episode = (entry[key] for key in ('split', 'group_id', 'episode_id'))
        if split not in parts or not isinstance(group, str) or not group or not isinstance(episode, str) or not episode:
            raise ValueError('invalid split, group or episode identity')
        if groups.setdefault(group, split) != split:
            raise ValueError(f'capture group crosses splits: {group}')
        if episode in episodes:
            raise ValueError(f'duplicate capture episode: {episode}')
        episodes.add(episode)
        trace_hash = entry['sha256']
        if trace_splits.setdefault(trace_hash, split) != split:
            raise ValueError('identical capture trace crosses splits')
        cfg = json.loads(_verified_bytes(path.parent, entry['config_path'], entry['config_sha256']))
        if cfg['physics']['dt_s'] != substep_dt or not math.isclose(
                cfg['physics']['control_hz']*substep_dt*substeps, 1., rel_tol=0., abs_tol=1e-10):
            raise ValueError('capture and configuration clocks disagree')
        model_cfg = {key: cfg[key] for key in ('physics', 'ball', 'action', 'reference')}
        model_cfg['initial'] = {key: cfg['initial'][key] for key in
                                ('tool_rotation_world', 'anchor_world_m', 'anchor_xy_tool_m')}
        if model_config is None:
            model_config = model_cfg
        elif model_cfg != model_config:
            raise ValueError('paired capture physical or route configuration changed')
        records = [json.loads(line) for line in _verified_bytes(path.parent, entry['path'], trace_hash).splitlines()]
        ticks = [row['snapshot_tick'] for row in records]
        if (not ticks or any(type(t) is not int or t < 0 for t in ticks)
                or ticks != sorted(set(ticks)) or ticks != entry['snapshot_ticks']):
            raise ValueError('snapshot ticks do not match the capture manifest')
        for record in records:
            parts[split].extend(_snapshot_rows(record, entry, model_cfg, horizon, amplitude, substep_dt,
                                               substeps, floor, noise_multiplier, record_amplitudes))
    if any(not rows for rows in parts.values()):
        raise ValueError('paired capture requires nonempty train and validation splits')
    batches = {split: PairBatch(**{name: tuple(row[name] for row in rows) if name == 'initial_state' else
                                  np.stack([row[name] for row in rows]) for name in rows[0]})
               for split, rows in parts.items()}
    report = {'known_names': KNOWN_NAMES, 'noise_multiplier': float(noise_multiplier),
              'response_floor_p_v_w_si': floor[::3].tolist(), 'splits': {}}
    for split, batch in batches.items():
        report['splits'][split] = {'snapshots':len(set(zip(batch.episode_ids, batch.snapshot_ticks))),
                                   'groups':len(set(batch.group_ids)), 'axis_pairs':len(batch.initial),
                                   'full_matched_leads':int(batch.pair_mask.sum()),
                                   'noise_measured_leads':int(batch.noise_mask.sum()),
                                   'unavailable_leads':int(batch.pair_mask.size-batch.pair_mask.sum()),
                                   'shortened_plus':int(np.count_nonzero((batch.plus_substeps>0) & ~batch.plus_mask)),
                                   'shortened_minus':int(np.count_nonzero((batch.minus_substeps>0) & ~batch.minus_mask)),
                                   'resolved_components':batch.response_mask.sum(axis=(0, 1)).tolist(),
                                   'unresolved_components_with_replay':(batch.noise_mask[:, :, None]
                                                                        & ~batch.response_mask).sum(axis=(0, 1)).tolist(),
                                   'max_replay_noise_by_component_si':batch.replay_noise.max(axis=(0, 1)).tolist()}
    return PairedDataset(substep_dt*substeps, horizon, amplitude, batches['train'], batches['validation'],
                         model_config, report)
