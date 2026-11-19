"""Run one deterministic S3 route and write ordinary control-rate transitions."""
from dataclasses import asdict
import json
import math
import time

import numpy as np

from ball_retention.contracts import vector
from ball_retention.dataset import validate_episode
from ball_retention.task import ball_in_tool

from native_repeatability import _make_env


HORIZON, SUBSTEPS = 240, 16
PHYSICS_DT_S = 1 / 960
PHYSICS_CAP = HORIZON * SUBSTEPS
TRACE = 'transitions.jsonl'


def _observe(physical_state):
    values = ball_in_tool(physical_state.ball, physical_state.tool)
    return values['position'], values['velocity']


def run_episode(root, cfg, episode_id, action, out, state, stage):
    """Execute one route from reset, recording failures and task endings distinctly.

    The caller provides the resource watchdog and persists each stage. The
    callback receives the current validated control tick and physical state.
    Simulation or serialization errors are propagated to that same caller.
    """
    if state['physics_steps'] != 0:
        raise ValueError('one route worker must begin with zero physical steps')
    started = time.monotonic()
    state.update(episode_id=episode_id, control_rows=0, trajectory_file=TRACE)
    env = _make_env(root, cfg, state, stage, expected_device_arch=89)
    first = env.reset(episode_id=episode_id)
    first.validate()
    if ((env.substeps, env.horizon_steps, env.dt, env.step_index, env.physics.steps)
            != (SUBSTEPS, HORIZON, PHYSICS_DT_S, 0, 0)
            or first.time_s != 0. or env.episode_id != episode_id or env.done or env.invalid
            or not math.isclose(env.route_end, 4., abs_tol=1e-12, rel_tol=0.)):
        raise ValueError('worker differs from the approved 4-second episode clock')
    initial_read = env.physics.read()
    state['initial_state'] = asdict(first)
    state['initial_surface_clearance_m'] = initial_read['surface_clearance_m']
    state['setup_wall_s'] = time.monotonic()-started
    stage('episode_started')

    step_s = action_s = encode_s = write_s = 0.
    last = None
    trace = out/TRACE
    positions, velocities = [], []
    p, v = _observe(first)
    positions.append(p)
    velocities.append(v)
    max_anchor_error_m = 0.
    max_rotation_error_rad = 0.
    loop_start = time.monotonic()
    with trace.open('xb') as stream:
        for tick in range(HORIZON):
            before = env.physics.steps
            if before != tick*SUBSTEPS or before+SUBSTEPS > PHYSICS_CAP:
                raise RuntimeError('episode would exceed its frozen physical reservation')
            previous = env.state
            action_start = time.monotonic()
            request = vector(action(tick, previous), 2, 'closed-loop action').tolist()
            action_s += time.monotonic()-action_start
            if any(abs(value) > 1. for value in request):
                raise ValueError('episode request exceeds the normalized action bound')
            step_start = time.monotonic()
            try:
                row = env.step(request)
            finally:
                step_s += time.monotonic()-step_start
                state['physics_steps'] += env.physics.steps-before
            applied = len(row.executed_joint_q)
            if (row.episode_id != episode_id or row.step_index != tick or env.step_index != tick+1
                    or row.state_t != previous or row.requested_action != request
                    or row.next_state_before_reset.previous_action != request
                    or row.invalid_reason is not None or env.invalid
                    or applied != env.physics.steps-before or env.physics.steps != state['physics_steps']
                    or env.done != (row.terminated or row.truncated)
                    or row.truncated != (tick+1 == HORIZON and not row.terminated)
                    or not math.isclose(row.next_state_before_reset.time_s, env.physics.steps*env.dt,
                                        abs_tol=1e-12, rel_tol=0.)):
                raise ValueError(f'episode transition or physical clock changed at tick {tick}')

            io_start = time.monotonic()
            encoded = (json.dumps(row.to_dict(), separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')
            encode_s += time.monotonic()-io_start
            io_start = time.monotonic()
            stream.write(encoded)
            stream.flush()
            write_s += time.monotonic()-io_start

            last = row
            state['control_rows'] += 1
            state['physical_time_s'] = row.next_state_before_reset.time_s
            max_anchor_error_m = max(max_anchor_error_m, row.task['max_anchor_error_m'])
            max_rotation_error_rad = max(max_rotation_error_rad, row.task['max_tool_rotation_error_rad'])
            p, v = _observe(row.next_state_before_reset)
            positions.append(p)
            velocities.append(v)
            stage(f'control_tick_{tick+1:03d}')
            if env.done:
                break
    loop_s = time.monotonic()-loop_start
    if last is None or not env.done or state['physics_steps'] <= 0:
        raise ValueError('route did not record a valid task ending')
    if not last.terminated and (state['physics_steps'], state['control_rows'], state['physical_time_s']) != (
            PHYSICS_CAP, HORIZON, 4.):
        raise ValueError('a nonterminal route did not finish its full 4-second clock')

    summary = validate_episode(trace, env.dt, env.substeps, expected_id=episode_id)
    if summary['physics_steps'] != state['physics_steps'] or summary['transitions'] != state['control_rows']:
        raise ValueError('ordinary trajectory readback disagrees with physical counters')
    final_read = env.physics.read()
    initial_anchor = np.asarray(initial_read['anchor_position_world_m'])
    final_anchor = np.asarray(final_read['anchor_position_world_m'])
    position, velocity = np.asarray(positions), np.asarray(velocities)
    output_bytes = trace.stat().st_size
    state['episode'] = {
        **summary,
        'motion_complete': last.task['motion_complete'],
        'departure_time_s': last.task['departure_time_s'],
        'final_carrying': last.task['carrying'],
        'final_surface_clearance_m': final_read['surface_clearance_m'],
        'initial_anchor_world_m': initial_anchor.tolist(),
        'final_anchor_world_m': final_anchor.tolist(),
        'anchor_displacement_world_m': (final_anchor-initial_anchor).tolist(),
        'final_anchor_reference_error_m': final_read['anchor_error_m'],
        'max_anchor_reference_error_m': max_anchor_error_m,
        'max_tool_rotation_tracking_error_rad': max_rotation_error_rad,
        'initial_ball_xy_tool_m': position[0, :2].tolist(),
        'final_ball_xy_tool_m': position[-1, :2].tolist(),
        'final_ball_velocity_tool_m_s': velocity[-1].tolist(),
        'max_ball_xy_speed_tool_m_s': float(np.linalg.norm(velocity[:, :2], axis=1).max()),
        'max_ball_xy_displacement_tool_m': float(np.linalg.norm(position[:, :2]-position[0, :2], axis=1).max()),
        'throughput': {
            'setup_wall_s': state['setup_wall_s'],
            'loop_wall_s': loop_s,
            'episode_step_wall_s': step_s,
            'action_evaluation_wall_s': action_s,
            'json_encoding_wall_s': encode_s,
            'trajectory_write_flush_wall_s': write_s,
            'trajectory_bytes': output_bytes,
            'control_rows_per_s': state['control_rows']/loop_s,
            'physics_steps_per_s': state['physics_steps']/loop_s,
            'trajectory_bytes_per_row': output_bytes/state['control_rows'],
            'worker_elapsed_s_including_validation': time.monotonic()-started,
        },
    }
    state['status'] = 'completed'
    stage('episode_validated')
