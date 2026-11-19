"""Prepare causal, fixed-control-rate ball-model windows from complete episodes."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .dataset import load_episode, read_manifest
from .task import ball_in_tool, reference


STATE_NAMES = ('position_x_m', 'position_y_m', 'position_z_m',
               'relative_velocity_x_m_s', 'relative_velocity_y_m_s', 'relative_velocity_z_m_s',
               'ball_omega_x_rad_s', 'ball_omega_y_rad_s', 'ball_omega_z_rad_s')
KNOWN_NAMES = ('midpoint_time_s', 'midpoint_reference_phase', 'roll_request', 'pitch_request',
               'midpoint_roll_rad', 'midpoint_pitch_rad', 'midpoint_roll_rate_rad_s',
               'midpoint_pitch_rate_rad_s',
               *(f'midpoint_gravity_tool_{axis}_m_s2' for axis in 'xyz'),
               *(f'midpoint_tool_omega_{axis}_rad_s' for axis in 'xyz'),
               *(f'midpoint_tool_alpha_{axis}_rad_s2' for axis in 'xyz'),
               *(f'midpoint_anchor_accel_tool_{axis}_m_s2' for axis in 'xyz'))


@dataclass(frozen=True)
class Windows:
    initial: np.ndarray
    known: np.ndarray
    target: np.ndarray
    episode_ids: np.ndarray
    ticks: np.ndarray


@dataclass(frozen=True)
class Endings:
    departure_outside_s: np.ndarray
    terminated: np.ndarray
    elapsed_s: np.ndarray  # Actual step duration, diagnostic metadata only.
    episode_ids: np.ndarray
    ticks: np.ndarray
    substeps: np.ndarray


@dataclass(frozen=True)
class PreparedDataset:
    dt_s: float  # Control period.
    train: dict[int, Windows]
    validation: dict[int, Windows]
    train_endings: Endings
    validation_endings: Endings
    state_mean: np.ndarray
    state_scale: np.ndarray
    known_mean: np.ndarray
    known_scale: np.ndarray
    known_names: tuple[str, ...]
    model_config: dict
    report: dict


def filter_advance(tilt, tilt_rate, target, tau_s, elapsed_s):
    """Exact held-target evolution of the same two-pole filter as the executor."""
    u = elapsed_s/tau_s
    decay = np.exp(-u)
    first_stage = tilt + tau_s*tilt_rate
    first_next = target + (first_stage-target)*decay
    next_tilt = target + (tilt-target+u*(first_stage-target))*decay
    next_rate = (first_next-next_tilt)/tau_s
    next_acceleration = (target-2*first_next+next_tilt)/tau_s**2
    return next_tilt, next_rate, next_acceleration


def _rotation(tilt, base_rotation):
    """World Y pitch after world X roll, with rotations in the last axes."""
    roll, pitch = np.moveaxis(tilt, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    pitch_roll = np.stack((cp, sp*sr, sp*cr, np.zeros_like(cp), cr, -sr,
                           -sp, cp*sr, cp*cr), axis=-1).reshape(*tilt.shape[:-1], 3, 3)
    return pitch_roll@base_rotation, np.stack((cp, np.zeros_like(cp), -sp), axis=-1)


def known_rollout(time_s, tilt_rad, tilt_rate_rad_s, actions, config, dt_s):
    """Derive midpoint tool features from the window start and requested controls.

    Accepts one (H, 2) sequence or a batch (N, H, 2). No measured trajectory
    is needed after the window start. The rotation is the analytic tool target.
    """
    actions = np.asarray(actions, dtype=float)
    if actions.ndim not in (2, 3) or actions.shape[-1] != 2 or actions.shape[-2] < 1:
        raise ValueError('expected actions shaped (H,2) or (N,H,2)')
    single = actions.ndim == 2
    if single:
        actions = actions[None, ...]
    size, horizon = actions.shape[:2]
    times = np.broadcast_to(np.asarray(time_s, dtype=float), (size,))
    tilt = np.broadcast_to(np.asarray(tilt_rad, dtype=float), (size, 2))
    rate = np.broadcast_to(np.asarray(tilt_rate_rad_s, dtype=float), (size, 2))
    if not all(np.isfinite(x).all() for x in (actions, times, tilt, rate)) or (times < 0).any():
        raise ValueError('non-finite rollout input or negative time')
    tau = float(config['action']['tilt_filter_time_constant_s'])
    limit = np.deg2rad(config['action']['tilt_limit_deg'])
    if not np.isfinite([dt_s, tau, limit]).all() or min(dt_s, tau, limit) <= 0:
        raise ValueError('invalid action filter or control period')
    base_rotation = np.asarray(config['initial']['tool_rotation_world'], dtype=float)
    gravity = np.array([0., 0., -config['physics']['gravity_m_s2']])
    pitch_axis = np.array([0., 1., 0.])
    clipped = np.clip(actions, -1., 1.)
    known = np.empty((size, horizon, len(KNOWN_NAMES)))
    for k in range(horizon):
        midpoint_times = times+(k+.5)*dt_s
        target = limit*clipped[:, k]
        mid_tilt, mid_rate, mid_accel = filter_advance(tilt, rate, target, tau, .5*dt_s)
        rotation, roll_axis = _rotation(mid_tilt, base_rotation)
        into_tool = rotation.swapaxes(-1, -2)
        roll_velocity = roll_axis*mid_rate[:, 0:1]
        pitch_velocity = pitch_axis*mid_rate[:, 1:2]
        angular_velocity = roll_velocity+pitch_velocity
        angular_acceleration = (roll_axis*mid_accel[:, 0:1] + pitch_axis*mid_accel[:, 1:2]
                                + np.cross(pitch_velocity, roll_velocity))
        ref = [reference(t, config) for t in midpoint_times]
        phase = np.array([row['phase'] for row in ref])
        anchor_acceleration = np.stack([row['anchor_acceleration'] for row in ref])
        known[:, k] = np.column_stack((midpoint_times, phase, clipped[:, k], mid_tilt, mid_rate,
                                      into_tool@gravity,
                                      (into_tool@angular_velocity[..., None])[..., 0],
                                      (into_tool@angular_acceleration[..., None])[..., 0],
                                      (into_tool@anchor_acceleration[..., None])[..., 0]))
        tilt, rate, _ = filter_advance(tilt, rate, target, tau, dt_s)
    return known[0] if single else known


def _ball_state(state):
    terms = ball_in_tool(state.ball, state.tool)
    return np.r_[terms['position'], terms['velocity'], terms['ball_omega']]


def _episode_config(root, entry, context, dt, substeps):
    provenance = entry['provenance']
    path = (root/provenance['config']).resolve()
    path.relative_to(root)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != provenance['config_sha256']:
        raise ValueError(f'configuration digest mismatch: {entry["episode_id"]}')
    cfg = json.loads(data)
    if any(cfg[key] != context[key] for key in ('physics', 'ball', 'action', 'reference')):
        raise ValueError(f'configuration disagrees with recorded context: {entry["episode_id"]}')
    if cfg['physics']['dt_s'] != dt or not np.isclose(substeps*dt*cfg['physics']['control_hz'], 1.):
        raise ValueError('dataset and configuration use different time steps')
    model_cfg = {key: cfg[key] for key in ('physics', 'ball', 'action', 'reference')}
    model_cfg['initial'] = {key: cfg['initial'][key] for key in
                            ('tool_rotation_world', 'anchor_world_m', 'anchor_xy_tool_m')}
    return model_cfg


def prepare_dataset(dataset_root, horizons=(1, 4, 8), *, verify_test=True):
    """Build training and validation windows and fit scales using training only."""
    horizons = tuple(horizons)
    if 1 not in horizons or any(type(h) is not int or h < 1 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError('expected distinct positive horizons including 1')
    root = Path(dataset_root).resolve()
    splits = ('train', 'validation')
    manifest = read_manifest(root) if verify_test else read_manifest(root, verify_splits=splits)
    dt, substeps = manifest['dt_s'], manifest['control_substeps']
    control_period = dt*substeps
    window_parts = {split: {h: [] for h in horizons} for split in splits}
    ending_parts = {split: [] for split in splits}
    equivalence = {'max_endpoint_tilt_error_rad': 0., 'max_endpoint_tilt_rate_error_rad_s': 0.,
                   'max_endpoint_tool_rotation_error_rad': 0., 'full_steps_checked': 0}
    model_cfg = None
    for entry in manifest['entries']:
        split = entry['split']
        if split not in splits:
            continue
        episode_id = entry['episode_id']
        rows = load_episode(root/entry['path'], dt, substeps, episode_id)
        config = _episode_config(root, entry, rows[0].context, dt, substeps)
        if model_cfg is None:
            model_cfg = config
        elif config != model_cfg:
            raise ValueError(f'pilot physical or route configuration changed: {episode_id}')
        states = [rows[0].state_t]+[row.next_state_before_reset for row in rows]
        ticks = np.array([row.step_index for row in rows])
        times = np.array([s.time_s for s in states])
        counts = np.array([len(row.executed_joint_q) for row in rows])
        ids = np.full(len(rows), episode_id)
        ending_parts[split].append((np.array([row.state_t.departure_outside_s for row in rows]),
                                    np.array([row.terminated for row in rows]), counts*dt, ids, ticks, counts))
        balls = np.stack([_ball_state(s) for s in states])
        tilts = np.array([s.tilt_rad for s in states])
        rates = np.array([s.tilt_rate_rad_s for s in states])
        actions = np.array([row.requested_action for row in rows])
        full = counts == substeps
        _check_endpoint_equivalence(config, control_period, states, tilts, rates, actions, full, equivalence)
        for h in horizons:
            indices = np.arange(max(0, len(rows)-h+1))[:, None]+np.arange(h)
            indices = indices[full[indices].all(axis=1)]
            if not len(indices):
                continue
            starts = indices[:, 0]
            known = known_rollout(times[starts], tilts[starts], rates[starts], actions[indices], config, control_period)
            window_parts[split][h].append((balls[starts], known, balls[indices+1], ids[starts], ticks[starts]))

    if any(not window_parts[split][h] for split in splits for h in horizons):
        raise ValueError('a split has no complete windows for the requested horizon')
    windows = {split: {h: Windows(*(np.concatenate(xs) for xs in zip(*parts)))
                       for h, parts in hs.items()} for split, hs in window_parts.items()}
    endings = {split: Endings(*(np.concatenate(xs) for xs in zip(*parts)))
               for split, parts in ending_parts.items()}
    first_steps = windows['train'][1]
    train_states = np.concatenate((first_steps.initial, first_steps.target[:, 0]))
    train_known = first_steps.known[:, 0]
    # Keep dimensions in SI; small floors avoid amplifying roundoff in near-constant axes.
    state_floor = np.repeat([1e-4, 1e-3, 1e-2], 3)
    known_floor = np.full(len(KNOWN_NAMES), 1e-4)
    report = {'window_counts': {split: {h: len(w.initial) for h, w in hs.items()} for split, hs in windows.items()},
              'ending_counts': {split: {'rows': len(e.ticks), 'positive': int(e.terminated.sum()),
                                        'shortened': int(np.count_nonzero(e.substeps != substeps))}
                                for split, e in endings.items()},
              'normalization_source': 'complete train 1-step windows, initial/target state and midpoint features',
              'state_names': STATE_NAMES, 'state_scale_floor': state_floor.tolist(),
              'known_scale_floor': known_floor.tolist(), 'tool_prediction_equivalence': equivalence}
    return PreparedDataset(control_period, windows['train'], windows['validation'], endings['train'], endings['validation'],
                           train_states.mean(axis=0), np.maximum(train_states.std(axis=0), state_floor),
                           train_known.mean(axis=0), np.maximum(train_known.std(axis=0), known_floor), KNOWN_NAMES,
                           model_cfg, report)


def _check_endpoint_equivalence(config, dt, states, tilts, rates, actions, full, evidence):
    """Measure tool prediction error using true endpoints only as diagnostics."""
    if not np.any(full):
        return
    action = config['action']
    target = np.deg2rad(action['tilt_limit_deg'])*np.clip(actions[full], -1., 1.)
    tilt, rate, _ = filter_advance(tilts[:-1][full], rates[:-1][full], target,
                                   action['tilt_filter_time_constant_s'], dt)
    rotation, _ = _rotation(tilt, np.asarray(config['initial']['tool_rotation_world']))
    quaternions = np.array([s.tool.quaternion_xyzw for s, keep in zip(states[1:], full) if keep])
    measured = Rotation.from_quat(quaternions)
    errors = {'max_endpoint_tilt_error_rad': np.max(abs(tilt-tilts[1:][full])),
              'max_endpoint_tilt_rate_error_rad_s': np.max(abs(rate-rates[1:][full])),
              'max_endpoint_tool_rotation_error_rad': np.max((measured.inv()*Rotation.from_matrix(rotation)).magnitude())}
    for key, value in errors.items():
        evidence[key] = max(evidence[key], float(value))
    evidence['full_steps_checked'] += int(np.count_nonzero(full))
