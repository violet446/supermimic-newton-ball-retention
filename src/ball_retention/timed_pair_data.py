"""Read train-only timed action pairs using the existing paired loss tensors."""
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path

import numpy as np

from .archive_state_import import _parse_state
from .model_data import _ball_state, known_rollout
from .pair_collection import _initial_key
from .pair_data import DEFAULT_RESPONSE_FLOOR, PairBatch, _array, _branch, _verified_bytes


SCHEMA = 's3-timed-action-pairs-v1'
TIMED_AXIS = -1


@dataclass(frozen=True)
class TimedPairs:
    batch: PairBatch
    directions: tuple[str, ...]
    direction_actions: np.ndarray
    branch_actions: np.ndarray  # [plus/minus, sample, lead, roll/pitch]
    amplitudes: np.ndarray
    dt_s: float
    model_config: dict
    entries: tuple[dict, ...]


def _json_ref(folder, ref):
    return json.loads(_verified_bytes(folder, ref['path'], ref['sha256']))


def _read_branch(folder, ref, requests, tick, dt, substeps):
    if not np.array_equal(_array(ref['requested_actions_Hx2'], requests.shape, 'manifest requests'), requests):
        raise ValueError('manifest request sequence disagrees with signed timing direction')
    trace = _json_ref(folder, ref)
    target, counts, ending = _branch(trace, requests, dt, substeps)
    horizon = len(requests)
    if (not np.array_equal(trace['actions'], requests) or not np.all(counts == substeps)
            or ending != (False, False, None)
            or any(len(trace[key]) != horizon for key in ('states', 'commands', 'retention'))
            or [row['physics_step'] for row in trace['contacts']] != list(range(tick*substeps+1, (tick+horizon)*substeps+1))):
        raise ValueError('timed capture requires a complete, ordered before-reset trace')
    for lead, (fields_, control, status) in enumerate(zip(trace['states'], trace['commands'], trace['retention']), 1):
        state = _parse_state(fields_)
        if (state.time_s != (tick+lead)*substeps*dt or not np.array_equal(state.previous_action, requests[lead-1])
                or status['terminated'] is not False or status['truncated'] is not False or status['reason'] is not None
                or type(status['carrying_candidate']) is not bool or status['outside_grace_active'] is not (state.departure_outside_s > 0)):
            raise ValueError('timed public state clock, control or retention disagrees with replay')
        for key, values in (('q', state.joint_q), ('qd', state.joint_qd)):
            if not np.array_equal(_array(control[key], (substeps, 6), key)[-1], values):
                raise ValueError('timed joint state disagrees with executed substep commands')
        np.testing.assert_allclose(target[lead-1], _ball_state(state), rtol=0., atol=1e-12)
    return target, counts


def concatenate_batches(*batches):
    return PairBatch(**{f.name: (sum((b.initial_state for b in batches), ()) if f.name == 'initial_state'
                                else np.concatenate([getattr(b, f.name) for b in batches])) for f in fields(PairBatch)})


def prepare_timed_pairs(root, manifest_path, expected_sha256):
    """Validate immutable train-only traces and derive features from the shared initial State."""
    root = Path(root).resolve()
    path = Path(manifest_path)
    manifest = json.loads(_verified_bytes(root, path.as_posix(), expected_sha256))
    folder = (root/path).resolve().parent
    horizon = manifest['horizon']
    if manifest['schema'] != SCHEMA or type(horizon) is not int or horizon < 1 or not manifest['entries']:
        raise ValueError('invalid timed action pair schema or horizon')
    entries = tuple(manifest['entries'])
    if any(e['split'] != 'train' or e['group_id'] == manifest['development_holdout_group'] for e in entries):
        raise ValueError('timed training capture contains a held-out entry')
    rows, labels, directions, actions, amplitudes = [], [], [], [], []
    snapshot_keys, physical_groups, group_initials, seen_paths = set(), {}, {}, set()
    model_config = clock = None
    for entry in entries:
        group, episode, tick = (entry[k] for k in ('group_id', 'episode_id', 'snapshot_tick'))
        if not all(isinstance(key, str) and key for key in (group, episode)) or type(tick) is not int or tick < 0:
            raise ValueError('invalid timed capture identity')
        cfg = json.loads(_verified_bytes(root, entry['source_config'], entry['source_config_sha256']))
        initial_key = _initial_key(cfg)
        if (initial_key != entry['physical_initial_sha256'] or cfg['seed'] != entry['seed']
                or any(cfg['history_capture'][key] != entry[key] for key in ('split', 'group_id', 'episode_id', 'history_id'))
                or physical_groups.setdefault(initial_key, group) != group or group_initials.setdefault(group, initial_key) != initial_key
                or (episode, tick) in snapshot_keys):
            raise ValueError('timed capture changes source identity or initial-state grouping')
        snapshot_keys.add((episode, tick))
        current_clock = entry['clock']
        substeps, dt = current_clock['control_substeps'], current_clock['physics_dt_s']
        if (type(substeps) is not int or substeps < 1 or not math.isfinite(dt) or dt <= 0 or cfg['physics']['dt_s'] != dt
                or cfg['physics']['control_hz']*substeps*dt != 1. or current_clock['snapshot_time_s'] != tick*substeps*dt):
            raise ValueError('timed snapshot clock disagrees with its source configuration')
        model_cfg = {key: cfg[key] for key in ('physics', 'ball', 'action', 'reference')}
        model_cfg['initial'] = {key: cfg['initial'][key] for key in ('tool_rotation_world', 'anchor_world_m', 'anchor_xy_tool_m')}
        if model_config is None:
            model_config, clock = model_cfg, (dt, substeps)
        elif model_config != model_cfg or clock != (dt, substeps):
            raise ValueError('timed sources disagree on model configuration or clocks')

        native = _json_ref(folder, entry['native_snapshot'])
        state = _parse_state(native['state'])
        if (entry['native_snapshot']['sha256'] != entry['native_snapshot_sha256'] or native['episode_id'] != episode
                or native['step_index'] != tick or native['physics']['steps'] != tick*substeps
                or state.time_s != current_clock['snapshot_time_s'] or native['done'] is not False or native['invalid'] is not False):
            raise ValueError('timed native snapshot does not match manifest identity')

        def take(ref, requests):
            if ref['path'] in seen_paths:
                raise ValueError('timed trace path is reused')
            seen_paths.add(ref['path'])
            return _read_branch(folder, ref, requests, tick, dt, substeps)

        first, second = entry['nominal_branches']
        nominal = _array(first['requested_actions_Hx2'], (horizon, 2), 'nominal actions')
        base_a, _ = take(first, nominal)
        base_b, _ = take(second, nominal)
        noise = np.abs(base_a-base_b)/2.
        floor = np.maximum(np.repeat(DEFAULT_RESPONSE_FLOOR, 3), 5.*noise)
        full = np.ones(horizon, dtype=bool)
        initial = _ball_state(state)
        seen_directions = set()
        if not entry['pairs']:
            raise ValueError('timed snapshot has no action pairs')
        for pair in entry['pairs']:
            name, amplitude = pair['direction_id'], pair['amplitude']
            direction = _array(pair['direction_Hx2'], (horizon, 2), 'timed direction')
            if (not isinstance(name, str) or not name or name in seen_directions or type(amplitude) not in (float, int)
                    or not 0 < amplitude < 1 or not np.any(direction)):
                raise ValueError('invalid or duplicate timed action direction')
            seen_directions.add(name)
            ordered = sorted(pair['branches'], key=lambda ref: ref['sign'], reverse=True)
            if [ref['sign'] for ref in ordered] != [1, -1] or any(type(ref['sign']) is not int for ref in ordered):
                raise ValueError('timed pair requires one positive and one negative branch')
            signed = nominal+np.array([1., -1.])[:, None, None]*amplitude*direction
            if np.any(np.abs(signed) >= 1.):
                raise ValueError('timed action pair reaches clipping')
            (plus, plus_steps), (minus, minus_steps) = (take(ref, requested) for ref, requested in zip(ordered, signed))
            kp, km = [known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s, request,
                                    model_config, dt*substeps) for request in signed]
            rows.append(dict(initial=initial, plus_known=kp, minus_known=km, plus_target=plus, minus_target=minus,
                             plus_mask=full, minus_mask=full, pair_mask=full, noise_mask=full,
                             response_mask=np.abs(plus-minus)/2. >= floor, replay_noise=noise, response_floor=floor,
                             baseline_first=base_a, baseline_second=base_b, baseline_mask=full,
                             plus_substeps=plus_steps, minus_substeps=minus_steps, plus_terminated=False,
                             minus_terminated=False, plus_truncated=False, minus_truncated=False,
                             episode_ids=episode, group_ids=group, snapshot_ticks=tick, axes=TIMED_AXIS,
                             nominal_actions=nominal, initial_state=native['state']))
            labels.append(name)
            directions.append(direction)
            actions.append(signed)
            amplitudes.append(amplitude)
    batch = PairBatch(**{key: tuple(row[key] for row in rows) if key == 'initial_state' else np.stack([row[key] for row in rows])
                         for key in rows[0]})
    return TimedPairs(batch, tuple(labels), np.stack(directions), np.stack(actions, axis=1), np.asarray(amplitudes),
                      clock[0]*clock[1], model_config, entries)
