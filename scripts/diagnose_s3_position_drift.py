"""Audit fixed paired rollouts and account for cumulative position error in SI."""
import json
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from ball_retention.geometry import load_tool
from ball_retention.guard import process_memory
from ball_retention.model_data import _rotation, filter_advance
from ball_retention.pair_collection import prepare_collection
from ball_retention.resources import directory_bytes, snapshot
from ball_retention.surface_diagnostic import SurfaceDiagnostic, sustained_crossing
from ball_retention.world_model import ResidualMLP
from check_s3_group_generalization import sha, write

PLAN = ROOT/'configs/s3_position_drift_plan.json'
OUT = ROOT/'runs/s3_position_drift_20261114'


def tool_omega(state, actions, config, dt):
    """Boundary angular velocity from the causal target filter and rotation."""
    tilt, rate = np.array(state['tilt_rad']), np.array(state['tilt_rate_rad_s'])
    tau = config['action']['tilt_filter_time_constant_s']
    limit = np.deg2rad(config['action']['tilt_limit_deg'])
    base = config['initial']['tool_rotation_world']
    velocities = []
    for action in [None, *actions]:
        if action is not None:
            tilt, rate, _ = filter_advance(tilt, rate, limit*np.clip(action, -1., 1.), tau, dt)
        rotation, roll_axis = _rotation(tilt, base)
        velocities.append(rotation.T @ (roll_axis*rate[0] + np.array([0., rate[1], 0.])))
    return np.array(velocities)


def project_tangent(v, normals):
    return v - np.sum(v*normals, axis=-1, keepdims=True)*normals


def geometry(states, omega, surface, cache):
    """Use each branch's nearest mesh face and target tool spin for slip proxy."""
    faces = []
    for state in states:
        key = tuple(state[:3])
        if key not in cache:
            cache[key] = surface.measure(state[:3])
        faces.append(cache[key])
    normals = np.array([face['normal_tool'] for face in faces])
    # Velocity of the sphere point at -R*n relative to the rotating tool.
    slip = project_tangent(states[:, 3:6] - surface.radius*np.cross(states[:, 6:9]-omega, normals), normals)
    return normals, {
        'face': [face['triangle'] for face in faces],
        'normal_tool': normals.tolist(),
        'gap_m': [face['sphere_distance_gap_m'] for face in faces],
        'normal_gap_m': [face['normal_plane_gap_m'] for face in faces],
        'plane_projection_on_face': [face['plane_projection_on_triangle'] for face in faces],
        'outward_side': [face['outward_side'] for face in faces],
        'upward_face': [face['upward_bowl_face'] for face in faces],
        'tangential_slip_proxy_m_s': slip.tolist(),
    }


def branch_account(prediction, truth, local, dt, true_normals):
    error = prediction-truth
    # The baseline uses p[t+1] = p[t] + dt*v[t]. The balance uses matched
    # reference v[t]; its remainder includes the model's position residual
    # and the observed motion missed by left-endpoint quadrature.
    velocity_term = dt*error[:-1, 3:6]
    from_normal_velocity = velocity_term-project_tangent(velocity_term, true_normals[:-1])
    position_remainder = np.diff(error[:, :3], axis=0)-velocity_term
    rebuilt = np.cumsum(velocity_term+position_remainder, axis=0)
    np.testing.assert_allclose(rebuilt, error[1:, :3], atol=2e-15, rtol=0.)
    # Reference consistency at the control period for an alternative scheme.
    truth_trapezoid = dt*(truth[:-1, 3:6]+truth[1:, 3:6])*.5
    pred_trapezoid = dt*(prediction[:-1, 3:6]+prediction[1:, 3:6])*.5
    return {
        'error_p_v_w_si': error[1:],
        'local_truth_input_error_p_v_w_si': local-truth[1:],
        'velocity_cumsum_m': np.cumsum(velocity_term, axis=0),
        'normal_velocity_cumsum_m': np.cumsum(from_normal_velocity, axis=0),
        'tangent_velocity_cumsum_m': np.cumsum(velocity_term-from_normal_velocity, axis=0),
        'position_remainder_cumsum_m': np.cumsum(position_remainder, axis=0),
        'reference_left_step_remainder_m': np.diff(truth[:, :3], axis=0)-dt*truth[:-1, 3:6],
        'reference_trapezoid_step_remainder_m': np.diff(truth[:, :3], axis=0)-truth_trapezoid,
        'prediction_trapezoid_step_remainder_m': np.diff(prediction[:, :3], axis=0)-pred_trapezoid,
    }


def main():
    started = time.monotonic()
    raw_plan = json.loads(PLAN.read_bytes())
    budget = raw_plan['budget']
    OUT.mkdir(exist_ok=False)
    stage = 'start'
    peaks = {'peak_rss_bytes': 0, 'peak_private_commit_bytes': 0}

    def guard(label):
        nonlocal stage
        stage = label
        memory = process_memory(os.getpid())
        machine = snapshot(ROOT)
        elapsed = time.monotonic()-started
        for key in peaks:
            peaks[key] = max(peaks[key], memory[key])
        with (OUT/'resources.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps({'stage': stage, 'wall_s': elapsed, **memory, **machine})+'\n')
        if (elapsed >= budget['wall_s'] or peaks['peak_rss_bytes'] > budget['rss_gib']*1024**3
                or peaks['peak_private_commit_bytes'] > budget['commit_gib']*1024**3
                or machine['available_ram_bytes'] < budget['available_ram_floor_gib']*1024**3
                or machine['disk_free_bytes'] < budget['output_mib']*1024**2
                or directory_bytes(OUT) > budget['output_mib']*1024**2):
            raise RuntimeError(f'offline resource cap at {stage}')

    def verify_hashes():
        for path, digest in raw_plan['sha256'].items():
            if sha(ROOT/path) != digest:
                raise ValueError(f'plan hash mismatch: {path}')

    try:
        guard('start')
        verify_hashes()
        write(OUT/'plan.json', {'config': raw_plan, 'design_sha256': sha(PLAN)})
        collection = prepare_collection(ROOT, ROOT/raw_plan['collection'])
        mixed = collection.validation
        b = mixed.batch
        if len(b.initial) != 36 or not b.pair_mask.all() or collection.horizon != 32:
            raise ValueError('expected 36 complete 32-step development pairs')
        cfg, dt = collection.model_config, collection.dt_s
        surface = SurfaceDiagnostic(load_tool(ROOT/raw_plan['mesh']), cfg['ball']['radius_m'],
                                    cfg['initial']['tool_rotation_world'])
        histories = {e['episode_id']: e['history_id'] for e in
                     json.loads((ROOT/raw_plan['histories']).read_bytes())['episodes']}
        reference = json.loads((ROOT/raw_plan['reference']).read_bytes())['models']
        guard('prepared')

        known = np.concatenate((b.plus_known, b.minus_known))
        initial = np.concatenate((b.initial, b.initial))
        targets = np.concatenate((b.plus_target, b.minus_target))
        truth = np.concatenate((initial[:, None], targets), axis=1)
        offsets = np.eye(2)[b.axes]*mixed.amplitudes[:, None]
        controls = np.concatenate((b.nominal_actions+offsets[:, None], b.nominal_actions-offsets[:, None]))
        omega = [tool_omega(state, action, cfg, dt) for state, action in zip(b.initial_state*2, controls)]
        cached = {}
        true_geometry = [geometry(states, angular, surface, cached) for states, angular in zip(truth, omega)]
        guard('truth_geometry')
        results = {}
        for name, path in raw_plan['checkpoints'].items():
            guard(name)
            with np.load(ROOT/path, allow_pickle=False) as saved:
                model = ResidualMLP.from_state_dict(saved)
                features = (known-saved['known_mean'])/saved['known_scale']
            free = model.denormalize(model.rollout(model.normalize(initial), features))
            local = model.denormalize(model.step(model.normalize(truth[:, :-1]), features))
            prediction = np.concatenate((initial[:, None], free), axis=1)
            saved_rows = [row for row in reference[name]['pairs'] if row['split'] == 'validation']
            rows = []
            for i, (group, episode, tick, axis, amplitude) in enumerate(zip(
                    b.group_ids, b.episode_ids, b.snapshot_ticks, b.axes, mixed.amplitudes)):
                row = {'group': str(group), 'episode': str(episode), 'tick': int(tick),
                       'axis': int(axis), 'amplitude': float(amplitude), 'cohort': str(mixed.cohorts[i]),
                       'history': histories.get(str(episode), 'original')}
                saved_row = saved_rows[i]
                for key in ('group', 'episode', 'tick', 'axis', 'amplitude', 'cohort', 'history'):
                    if row[key] != saved_row[key]:
                        raise ValueError(f'validation pair identity mismatch: {name} {i} {key}')
                accounts = []
                for sign, j in (('plus', i), ('minus', i+len(b.initial))):
                    normals, actual_geom = true_geometry[j]
                    _, predicted_geom = geometry(prediction[j], omega[j], surface, cached)
                    account = branch_account(prediction[j], truth[j], local[j], dt, normals)
                    accounts.append(account)
                    row[sign] = {'reference_geometry': actual_geom, 'predicted_geometry': predicted_geom,
                                 'reference_p_v_w_si': truth[j].tolist(), 'predicted_p_v_w_si': prediction[j].tolist()}
                # Both signs are equally weighted within the pair.
                mean = {key: (accounts[0][key]+accounts[1][key])*.5 for key in accounts[0]}
                for h in (8, 16, 32):
                    error = np.stack([a['error_p_v_w_si'][h-1] for a in accounts]).reshape(2, 3, 3)
                    np.testing.assert_allclose(np.sum(error**2, axis=(0, 2)),
                                               saved_row['absolute'][str(h)]['squared_p_v_w_si'], rtol=1e-12, atol=1e-18)
                p = np.linalg.norm(mean['error_p_v_w_si'][:, :3], axis=1)
                v = np.abs(mean['error_p_v_w_si'][:, 3:6])
                count = raw_plan['consecutive_steps']
                row['onset'] = {
                    'position_lead': {str(t): sustained_crossing(p, t, count) for t in raw_plan['position_thresholds_m']},
                    'velocity_axis_lead': {axis: sustained_crossing(v[:, j], raw_plan['velocity_threshold_m_s'], count)
                                           for j, axis in enumerate('xyz')},
                }
                row['pair_mean'] = {key: value.tolist() for key, value in mean.items()}
                rows.append(row)
                if (i+1) % 6 == 0:
                    guard(f'{name}_{i+1}')
            results[name] = rows
        write(OUT/'audit.json', {'dt_s': dt, 'models': results})
        verify_hashes()
        guard('complete')
        write(OUT/'summary.json', {'status': 'completed', 'models': list(results), 'pairs_per_model': 36,
              'snapshots': len(set(zip(b.episode_ids, b.snapshot_ticks))), 'wall_s': time.monotonic()-started,
              **peaks, 'prior_H8_H16_H32_errors_reproduced': True, 'position_error_balance_verified': True,
              'training_updates': 0, 'simulation_steps': 0, 'test_episode_files_read': False,
              'artifacts_sha256': {p.name: sha(p) for p in sorted(OUT.iterdir())}})
    except Exception as error:
        write(OUT/'failure.json', {'status': 'stopped', 'stage': stage, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
