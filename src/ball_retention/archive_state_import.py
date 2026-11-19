"""Reconstruct an interior single-ball snapshot for the S3 restore pilot."""

from copy import deepcopy
from dataclasses import asdict, fields as dataclass_fields
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import RigidState, State, vector
from .geometry import sphere_clearance
from .task import ball_in_tool, reference


def _parse_state(fields):
    if set(fields) != {item.name for item in dataclass_fields(State)}:
        raise ValueError('archive does not contain the full public State')
    state = State(**{**fields, 'ball': RigidState(**fields['ball']), 'tool': RigidState(**fields['tool'])})
    state.validate()
    return state


def _check_layout(physics):
    model = physics.model
    count = physics.arm_body_count + 1
    joint_types = model.joint_type.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    flags = model.body_flags.numpy()
    kinematic = int(physics.newton.BodyFlags.KINEMATIC)
    dynamic = int(physics.newton.BodyFlags.DYNAMIC)
    free = int(physics.newton.JointType.FREE)
    control = model.control(clone_variables=False)
    joint_effort = control.joint_f.numpy()
    if (model.particle_count or model.body_count != count or model.joint_count != count
            or physics.ball_id != count-1 or not 0 <= physics.tool_id < physics.ball_id
            or list(model.body_label) != [*physics.arm_names, 'ball']
            or not np.array_equal(flags, [kinematic]*(count-1)+[dynamic])
            or np.any(joint_types != free) or np.any(joint_parent != -1)
            or not np.array_equal(joint_child, np.arange(count)) or np.any(joint_effort != 0.)
            or physics.solver.enable_restitution):
        raise ValueError('archive import requires separate prescribed bodies, one free sphere and zero default effort')

    template = physics.initial['arrays']
    shapes = {'body_q': (count, 7), 'body_qd': (count, 6), 'body_f': (count, 6),
              'joint_q': (count*7,), 'joint_qd': (count*6,)}
    for state_buffer in (physics.a, physics.b):
        names = {name for name, value in vars(state_buffer).items() if isinstance(value, physics.wp.array)}
        if names != shapes.keys() or names != template.keys() or state_buffer.requires_grad:
            raise ValueError(f'unsupported native archive state arrays: {sorted(names)}')
        for name, shape in shapes.items():
            native = getattr(state_buffer, name).numpy()
            if (native.shape != shape or template[name].shape != shape or native.dtype != np.float32
                    or template[name].dtype != native.dtype or not np.isfinite(template[name]).all()):
                raise ValueError(f'inconsistent native layout for {name}')
    if np.any(template['body_f'] != 0.):
        raise ValueError('initial force template is not zero')
    return {
        'body_labels': list(model.body_label), 'body_flags': flags.tolist(), 'free_joint_count': count,
        'joint_parent': joint_parent.tolist(), 'joint_child': joint_child.tolist(),
        'particle_count': model.particle_count, 'zero_internal_default_effort': True,
        'restitution_enabled': False, 'shape_labels': list(physics.shape_names),
        'expected_collision_shapes': {'ball': physics.ball_shape, 'shovel': physics.shovel_shape,
                                      'floor': physics.floor_shape},
        'state_arrays': {name: {'shape': list(shape), 'dtype': str(template[name].dtype)}
                         for name, shape in shapes.items()},
    }


def _interior(env, state, gap, floor_reached):
    relative = ball_in_tool(state.ball, state.tool)
    geometry = env.departure.envelope.measure(relative['position'])
    # The runtime resets deep_count after every clean substep and clears the
    # outside timer whenever the sphere is inside the envelope.
    if (not math.isfinite(gap) or gap < -.002 or floor_reached or state.departure_outside_s != 0.
            or not geometry['center_inside_projection'] or geometry['outside_with_margin']
            or geometry['below_entire_tool']):
        raise ValueError('archive state is outside the verified clean interior scope')
    return geometry


def _readback_errors(observed, expected):
    errors = {}
    limits = {'position_m': 1e-6, 'rotation_rad': 1e-5, 'linear_velocity_m_s': 1e-5,
              'angular_velocity_rad_s': 1e-5}
    for label in ('ball', 'tool'):
        actual_rigid, expected_rigid = getattr(observed, label), getattr(expected, label)
        rotation = Rotation.from_quat(actual_rigid.quaternion_xyzw)
        rotation_ref = Rotation.from_quat(expected_rigid.quaternion_xyzw)
        error = {
            'position_m': float(np.linalg.norm(np.asarray(actual_rigid.position)-expected_rigid.position)),
            'rotation_rad': float((rotation*rotation_ref.inv()).magnitude()),
            'linear_velocity_m_s': float(np.linalg.norm(np.asarray(actual_rigid.linear_velocity_origin)
                                                       -expected_rigid.linear_velocity_origin)),
            'angular_velocity_rad_s': float(np.linalg.norm(np.asarray(actual_rigid.angular_velocity)
                                                         -expected_rigid.angular_velocity)),
        }
        if any(not math.isfinite(value) or value > limits[key] for key, value in error.items()):
            raise ValueError(f'archive {label} readback exceeds tolerance: {error}')
        errors[label] = error
    for name in ('time_s', 'joint_q', 'joint_qd', 'tilt_rad', 'tilt_rate_rad_s', 'previous_action',
                 'reference_phase', 'departure_outside_s'):
        if not np.array_equal(getattr(observed, name), getattr(expected, name)):
            raise ValueError(f'archive readback differs in {name}')
    return errors


def reconstruct(env, fields, tick, command):
    """Restore an archived public State using the nominal prefix's last command.

    Physical arrays come from the scene initialization template and the public
    archive. No state or contact buffers from the live continuation are used.
    """
    if not env.episode_id or type(tick) is not int or not 0 < tick < env.horizon_steps:
        raise ValueError('reconstruction requires a reset scene and an interior episode tick')
    state = _parse_state(fields)
    phys_tick = tick*env.substeps
    time_s = phys_tick*env.dt
    if (state.time_s != time_s or state.reference_phase != reference(time_s, env.cfg)['phase']
            or not math.isclose(command.time_s, time_s, abs_tol=1e-12, rel_tol=0.)):
        raise ValueError('archive/controller phase does not match the physical clock')
    for name, observed in (('joint_q', command.q), ('joint_qd', command.qd), ('tilt_rad', command.tilt),
                           ('tilt_rate_rad_s', command.tilt_rate), ('previous_action', command.clipped_action)):
        if not np.array_equal(getattr(state, name), observed):
            raise ValueError(f'archive differs from nominal controller in {name}')
    filtered = vector(command.filter_input_state, 2, 'nominal filter state')
    expected_filtered = np.array(state.tilt_rad)+env.controller.tau*np.array(state.tilt_rate_rad_s)
    filter_error = float(np.max(abs(filtered-expected_filtered)))
    if filter_error > 1e-15:
        raise ValueError('archive tilt rate differs from nominal filter state')

    physics = env.physics
    local = ball_in_tool(state.ball, state.tool)['position']
    radius = env.cfg['ball']['radius_m']
    gap = sphere_clearance(physics.mesh, local, radius)
    floor_reached = state.ball.position[2] <= env.cfg['physics']['floor_z_m']+radius+.001
    geometry = _interior(env, state, gap, floor_reached)
    layout = _check_layout(physics)
    arrays = {name: array.copy() for name, array in physics.initial['arrays'].items()}
    ball = state.ball
    ball_rotation = Rotation.from_quat(ball.quaternion_xyzw)
    ball_id = physics.ball_id
    com = ball_rotation.apply(physics.com_local[ball_id])
    angular = np.array(ball.angular_velocity)
    linear_com = np.asarray(ball.linear_velocity_origin)+np.cross(angular, com)
    arrays['body_q'][ball_id] = [*ball.position, *ball.quaternion_xyzw]
    arrays['body_qd'][ball_id] = [*linear_com, *angular]

    # Apply the standard FK path to the initialized arrays before freezing
    # the rebuilt native snapshot. It overwrites only prescribed arm bodies.
    for name, value in arrays.items():
        getattr(physics.a, name).assign(value)
    joint_command = (command.q.copy(), command.qd.copy(), command.qdd.copy())
    physics._prescribe(time_s, joint_command)
    arrays['body_q'] = physics.a.body_q.numpy().copy()
    arrays['body_qd'] = physics.a.body_qd.numpy().copy()
    saved = {
        'schema': 'ball-retention-episode-snapshot-v1', 'episode_id': env.episode_id,
        'step_index': tick, 'state': state, 'done': False, 'invalid': False, 'deep_count': 0,
        'physics': {'schema': 'prescribed-arm-v1', 'steps': phys_tick, 'arrays': arrays, 'command': joint_command},
        'controller': {'time_s': command.time_s, 'q': command.q.tolist(), 'qd': command.qd.tolist(),
                       'tilt': command.tilt.tolist(), 'filtered': filtered.tolist()},
        'monitor': {'last_time_s': time_s, 'outside_since_s': None, 'terminated': False, 'reason': None},
    }
    env.restore(saved)
    for state_buffer in (physics.a, physics.b):
        for name, value in arrays.items():
            if not np.array_equal(getattr(state_buffer, name).numpy(), value):
                raise ValueError(f'restored native buffers disagreed with reconstruction in {name}')
    row = physics.read(command.rotation)
    if (row['step'] != phys_tick or row['fk_position_error_m'] > 2e-5 or row['fk_velocity_error'] > 2e-5
            or row['anchor_error_m'] > env.cfg['admission']['motion_position_tolerance_m']
            or row['tool_rotation_error_rad'] > np.deg2rad(env.cfg['admission']['motion_angle_tolerance_deg'])):
        raise ValueError('rebuilt native tool disagrees with FK or motion reference')
    observed = env._state(row, command.tilt, command.tilt_rate, command.clipped_action,
                          {'outside_duration_s': 0.})
    observed.validate()
    errors = _readback_errors(observed, state)
    _interior(env, observed, row['surface_clearance_m'], row['floor_reached'])
    saved['state'] = deepcopy(observed)
    env.state = observed
    evidence = {
        'tick': tick, 'physics_tick': phys_tick, 'source_state': asdict(state), 'restored_state': asdict(observed),
        'readback_errors': errors, 'native_layout': layout, 'both_buffers_equal': True,
        'joint_qdd_rad_s2': command.qdd.tolist(), 'filter_first_stage_max_abs_error_rad': filter_error,
        'ball_com_local_m': physics.com_local[ball_id].tolist(), 'ball_velocity_com_world_m_s': linear_com.tolist(),
        'source_surface_gap_m': float(gap), 'restored_surface_gap_m': row['surface_clearance_m'],
        'interior_geometry': geometry, 'restored_departure_monitor': env.departure.snapshot(),
        'restored_deep_count': env.deep_count, 'independent_fk_position_error_m': row['fk_position_error_m'],
        'independent_fk_velocity_error_si': row['fk_velocity_error'], 'anchor_error_m': row['anchor_error_m'],
        'tool_rotation_error_rad': row['tool_rotation_error_rad'],
        'restored_arrays': {'body_q': arrays['body_q'].tolist(), 'body_qd': arrays['body_qd'].tolist(),
                            'body_f': 'initial zeros, cleared by step before integration',
                            'joint_q_joint_qd': 'scene template; solver uses world-free joint metadata and body arrays'},
    }
    return saved, evidence
