"""Compare causal midpoint features to archived substep tool kinematics."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np
from scipy.spatial.transform import Rotation

from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _rotation, filter_advance, known_rollout
from ball_retention.robot import Robot
from ball_retention.task import reference
from deterministic_archive_metrics import _full_branch
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


ADVANCE = Path('runs/s3_training_timing_advance_20261116/validation.json')
URDF = 'assets/ec66_description.urdf'
TOOL = 'assets/shovel.STL'
LEADS = (1, 8, 16, 24, 32)


def norm(value):
    return np.linalg.norm(value, axis=-1)


def world_to_tool(rotation, vector):
    return np.einsum('...ji,...j->...i', rotation, vector)


def rotvec(first, zero):
    difference = first@np.swapaxes(zero, -1, -2)
    return Rotation.from_matrix(difference.reshape(-1, 3, 3)).as_rotvec().reshape(*difference.shape[:-2], 3)


def analytic_grid(state, actions, cfg, dt, substeps):
    tau = cfg['action']['tilt_filter_time_constant_s']
    elapsed = np.arange(substeps+1)[:, None]*dt
    target = np.deg2rad(cfg['action']['tilt_limit_deg'])*np.asarray(actions)
    t = state.time_s+(np.arange(len(actions))[:, None]*substeps+np.arange(substeps+1))*dt
    angle, rate = np.asarray(state.tilt_rad), np.asarray(state.tilt_rate_rad_s)
    tilt, rates, accelerations = [], [], []
    for row in target:
        a, w, derivative = filter_advance(angle, rate, row, tau, elapsed)
        tilt.append(a)
        rates.append(w)
        accelerations.append(derivative)
        angle, rate = a[-1], w[-1]
    tilt = np.stack(tilt)
    rotation, roll_axis = _rotation(tilt, np.asarray(cfg['initial']['tool_rotation_world']))
    pitch_axis = np.array([0., 1., 0.])
    rates, accelerations = np.stack(rates), np.stack(accelerations)
    omega = roll_axis*rates[..., :1]+pitch_axis*rates[..., 1:]
    alpha = (roll_axis*accelerations[..., :1]+pitch_axis*accelerations[..., 1:]
             +np.cross(pitch_axis*rates[..., 1:], roll_axis*rates[..., :1]))
    ref = [reference(time, cfg) for time in t.ravel()]
    shape = (*t.shape, 3)
    return {'rotation': rotation, 'tilt': tilt, 'rate': rates, 'omega': omega, 'alpha': alpha,
        'phase': np.asarray([row['phase'] for row in ref]).reshape(t.shape),
        **{k: np.asarray([row[k] for row in ref]).reshape(shape)
           for k in ('anchor_position', 'anchor_velocity', 'anchor_acceleration')}}


def command_grid(robot, anchor, state, commands, substeps, guard):
    q = np.asarray([row['q'] for row in commands])
    qd = np.asarray([row['qd'] for row in commands])
    if q.shape != (len(commands), substeps, 6) or qd.shape != q.shape:
        raise ValueError('unexpected substep joint-command layout')
    streams = [np.vstack((initial, values.reshape(-1, 6))) for initial, values in ((state.joint_q, q), (state.joint_qd, qd))]
    rotations, positions, twists = [], [], []
    for i, (joint, velocity) in enumerate(zip(*streams, strict=True)):
        pose, jacobian, _ = robot.fk(joint)
        rotation = pose[:3, :3]
        rotations.append(rotation)
        positions.append(pose[:3, 3])
        twists.append(jacobian@velocity)
        if i % 128 == 0:
            guard()
    index = np.arange(len(commands))[:, None]*substeps+np.arange(substeps+1)
    R = np.stack(rotations)[index]
    p = np.stack(positions)[index]
    twist = np.stack(twists)[index]
    arm = R@anchor
    return {'rotation': R, 'origin': p, 'omega': twist[..., 3:], 'origin_velocity': twist[..., :3],
            'anchor_position': p+arm, 'anchor_velocity': twist[..., :3]+np.cross(twist[..., 3:], arm)}


def derivative_at_mid(values, mid, dt):
    return (values[:, mid+1]-values[:, mid-1])/(2.*dt)


def branch_audit(robot, anchor, state, trace, cfg, dt, substeps, tolerance, guard):
    actions = np.asarray(trace['actions'])
    if not np.isfinite(actions).all() or np.any(abs(actions) >= 1.):
        raise ValueError('branch reaches action clipping')
    analytic = analytic_grid(state, actions, cfg, dt, substeps)
    executed = command_grid(robot, anchor, state, trace['commands'], substeps, guard)
    public = [state, *trace['states']]
    end = {k: np.concatenate((v[:1, 0], v[:, -1]), axis=0) for k, v in executed.items()}
    record = [x.tool for x in public]
    public_rotation = Rotation.from_quat([x.quaternion_xyzw for x in record]).as_matrix()
    fk_errors = {'rotation_rad': norm(rotvec(end['rotation'], public_rotation)),
                 'origin_m': norm(end['origin']-np.asarray([x.position for x in record])),
                 'omega_rad_s': norm(end['omega']-np.asarray([x.angular_velocity for x in record])),
                 'origin_velocity_m_s': norm(end['origin_velocity']-np.asarray([x.linear_velocity_origin for x in record]))}
    if any(np.max(value) > tolerance for value in fk_errors.values()):
        raise ValueError(f'q/qd FK fails the public pose or twist check: {fk_errors}')
    mid = substeps//2
    R_a, R_e = (d['rotation'][:, mid] for d in (analytic, executed))
    known = known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s, actions, cfg, dt*substeps)
    expected_time = state.time_s+(np.arange(len(actions))+.5)*(dt*substeps)
    np.testing.assert_array_equal(known[:, 0], expected_time)
    np.testing.assert_allclose(known[:, 1], analytic['phase'][:, mid], rtol=0., atol=1e-14)
    np.testing.assert_array_equal(known[:, 2:4], actions)
    np.testing.assert_allclose(known[:, 4:8], np.column_stack((analytic['tilt'][:, mid], analytic['rate'][:, mid])),
                               rtol=0., atol=2e-12)
    g = np.array([0., 0., -cfg['physics']['gravity_m_s2']])
    raw = np.column_stack((world_to_tool(R_a, g), world_to_tool(R_a, analytic['omega'][:, mid]),
                           world_to_tool(R_a, analytic['alpha'][:, mid]),
                           world_to_tool(R_a, analytic['anchor_acceleration'][:, mid])))
    np.testing.assert_allclose(raw, known[:, 8:], rtol=0., atol=2e-12)

    alpha_e = derivative_at_mid(executed['omega'], mid, dt)
    anchor_e = derivative_at_mid(executed['anchor_velocity'], mid, dt)
    magnitude = {key: norm(executed[key][:, 1:]-executed[key][:, mid:mid+1])
                 for key in ('omega', 'anchor_position', 'anchor_velocity')}
    magnitude['rotation'] = norm(rotvec(executed['rotation'][:, 1:], R_e[:, None]))
    feature_checks = {'gravity_m_s2': world_to_tool(R_e, g)-known[:, 8:11],
                      'omega_rad_s': world_to_tool(R_e, executed['omega'][:, mid])-known[:, 11:14],
                      'alpha_rad_s2': world_to_tool(R_e, alpha_e)-known[:, 14:17],
                      'anchor_accel_m_s2': world_to_tool(R_e, anchor_e)-known[:, 17:20]}
    interval_e = (executed['omega'][:, -1]-executed['omega'][:, 0])/(dt*substeps)
    interval_a = (analytic['omega'][:, -1]-analytic['omega'][:, 0])/(dt*substeps)
    stats = {'public_clock_endpoints': {k: float(v.max()) for k, v in fk_errors.items()},
        'time_s': expected_time.tolist(),
        'midpoint_tracking_norm': {'rotation_rad': norm(rotvec(R_e, R_a)).tolist(),
            'anchor_m': norm(executed['anchor_position'][:, mid]-analytic['anchor_position'][:, mid]).tolist(),
            'anchor_velocity_m_s': norm(executed['anchor_velocity'][:, mid]-analytic['anchor_velocity'][:, mid]).tolist(),
            'omega_rad_s': norm(executed['omega'][:, mid]-analytic['omega'][:, mid]).tolist(),
            'alpha_centered_difference_rad_s2': norm(alpha_e-derivative_at_mid(analytic['omega'], mid, dt)).tolist(),
            'anchor_accel_centered_difference_m_s2': norm(anchor_e-derivative_at_mid(analytic['anchor_velocity'], mid, dt)).tolist()},
        'midpoint_feature_error_norm': {key: norm(value).tolist() for key, value in feature_checks.items()},
        'substep_excursion_from_midpoint_max': {key: value.max(axis=1).tolist() for key, value in magnitude.items()},
        'interval_minus_midpoint': {'mean_substep_omega_rad_s': norm(executed['omega'][:, 1:].mean(axis=1)-executed['omega'][:, mid]).tolist(),
            'interval_alpha_rad_s2': norm(interval_e-analytic['alpha'][:, mid]).tolist(),
            'interval_alpha_execution_difference_rad_s2': norm(interval_e-interval_a).tolist()}}
    pairs = {'analytic_mid_omega': analytic['omega'][:, mid], 'executed_mid_omega': executed['omega'][:, mid],
             'executed_mean_omega': executed['omega'][:, 1:].mean(axis=1),
             'analytic_mid_alpha': analytic['alpha'][:, mid], 'executed_interval_alpha': interval_e,
             'analytic_interval_alpha': interval_a, 'analytic_mid_rotation': R_a,
             'executed_mid_rotation': R_e, 'executed_end_rotation': executed['rotation'][:, -1]}
    return stats, pairs


def response_comparison(name, branches, edge, gates):
    plus, minus = f'{name}_plus', f'{name}_minus'

    def half(value):
        return (edge[plus][value]-edge[minus][value])*.5

    orientation_half = {k: .5*rotvec(edge[plus][k], edge[minus][k])
                        for k in ('analytic_mid_rotation', 'executed_mid_rotation', 'executed_end_rotation')}
    physics = .5*(branches[plus]['raw']-branches[minus]['raw'])
    noise = .5*(branches['nominal_1']['raw']-branches['nominal_2']['raw'])
    physical_norms = norm(physics.reshape(-1, 3, 3))
    strong = physical_norms >= gates['strong_multiplier']*np.maximum(
        np.array([gates['position_floor_m'], gates['velocity_floor_m_s'], gates['spin_floor_rad_s']]),
        gates['noise_multiplier']*norm(noise.reshape(-1, 3, 3)))
    values = {k: half(k) for k in edge[plus] if k not in orientation_half}
    signals = {'executed_midpoint_omega_rad_s': values['executed_mid_omega'],
               'analytic_minus_executed_midpoint_omega_rad_s': values['analytic_mid_omega']-values['executed_mid_omega'],
               'endpoint_mean_minus_executed_midpoint_omega_rad_s': values['executed_mean_omega']-values['executed_mid_omega'],
               'executed_interval_alpha_rad_s2': values['executed_interval_alpha'],
               'interval_minus_model_midpoint_alpha_rad_s2': values['executed_interval_alpha']-values['analytic_mid_alpha'],
               'executed_minus_analytic_interval_alpha_rad_s2': values['executed_interval_alpha']-values['analytic_interval_alpha'],
               'executed_midpoint_orientation_rad': orientation_half['executed_mid_rotation'],
               'analytic_minus_executed_midpoint_orientation_rad': orientation_half['analytic_mid_rotation']-orientation_half['executed_mid_rotation'],
               'end_minus_executed_midpoint_orientation_rad': orientation_half['executed_end_rotation']-orientation_half['executed_mid_rotation']}
    return {'by_lead_norm': {key: norm(value).tolist() for key, value in signals.items()},
        'by_lead_vector': {key: value.tolist() for key, value in signals.items()},
        'newton_ball_half_difference_norm': physical_norms.tolist(), 'newton_ball_strong': strong.tolist(),
        'selected_leads': {str(h): {key: float(norm(value[h-1])) for key, value in signals.items()} for h in LEADS}}


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    def inputs(case):
        if case == 'development':
            c, cfg, fields, traces, _ = read_held(read(plan['held_design']))
            return cfg, _parse_state(fields), traces, c
        accepted = read(ADVANCE)
        manifest = ADVANCE.with_name('capture_manifest.json')
        data = read(manifest)
        entry, = data['entries']
        if (accepted['status'] != 'completed' or accepted['pairs'] != 3 or accepted['test_episode_files_read'] is not False
                or entry['split'] != 'train' or entry['history_id'] != 'advance' or entry['snapshot_tick'] != 168):
            raise ValueError('unexpected advance data scope')
        names = [*entry['nominal_branches'], *(b for item in entry['pairs'] for b in item['branches'])]
        def archive(ref):
            file = ADVANCE.parent/ref['path']
            if ref['sha256'] != plan['files_sha256'][file.as_posix()] or ref['sha256'] != accepted['artifacts_sha256'][ref['path']]:
                raise ValueError('advance archive hash changed')
            return read(file)
        snapshot = archive(entry['native_snapshot'])
        traces = {ref['label']: _full_branch(archive(ref), entry['snapshot_tick']) for ref in names}
        cfg = read(entry['source_config'])
        return cfg, _parse_state(snapshot['state']), traces, read(plan['response_design'])

    if plan['updates'] != 0 or plan['snapshots'] != ['development', 'advance'] or plan['fk_tolerance_si'] != 2e-5:
        raise ValueError('unexpected tool feature audit scope')
    robot = Robot(ROOT/URDF)
    mesh = load_tool(ROOT/TOOL)
    results = {}
    for name in plan['snapshots']:
        cfg, state, traces, capture = inputs(name)
        dt, substeps = cfg['physics']['dt_s'], capture['control_substeps']
        if substeps != 16 or abs(substeps*dt*cfg['physics']['control_hz']-1.) > 1e-14:
            raise ValueError('frozen control clock changed')
        anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
        branches, fields = {}, {}
        for label in capture['branch_order']:
            branches[label], fields[label] = branch_audit(robot, anchor, state, traces[label], cfg, dt, substeps,
                plan['fk_tolerance_si'], lambda: guard(name))
        results[name] = {'branches': branches, 'directions': {label: response_comparison(label, traces, fields, capture['gates'])
                                                             for label in capture['directions']}}
        guard(name, log=True)
    write_json(out/'tool_clock_audit.json', {'cases': results, 'substep_convention': 'The archive commands apply at right endpoints. '
               'Interval means use those 16 samples; accelerations use either matched centered differences at midpoint or endpoint velocities over the full interval.'})
    return {'snapshots': 2, 'branches_per_snapshot': 8, 'physics_substeps_reconstructed_per_branch': 512}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'tool_clock', run)
