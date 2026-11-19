"""Resume one archived closed-loop route after an exact public-state check."""
from dataclasses import asdict
import json
import math
import time

import numpy as np

from ball_retention.archive_state_import import reconstruct
from ball_retention.contracts import validate_window, vector
from ball_retention.controller import OnlineTiltController
from ball_retention.dataset import decode_transition, validate_episode
from ball_retention.geometry import load_tool, support_point
from ball_retention.prescribed import PrescribedPath
from ball_retention.robot import Robot
from native_repeatability import _make_env
from s3_closed_loop_episode import HORIZON, PHYSICS_CAP, PHYSICS_DT_S, SUBSTEPS, TRACE, _observe


RESTORE_TICK = 75
ARCHIVE_ROWS = RESTORE_TICK+1


def _encode(value):
    return (json.dumps(value,separators=(',', ':'),allow_nan=False)+'\n').encode('utf-8')


def _read_prefix(archive_path, episode_id):
    raw = archive_path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    rows = [decode_transition(record) for record in records]
    validate_window(rows,PHYSICS_DT_S,SUBSTEPS)
    if (not raw.endswith(b'\n') or len(rows) != ARCHIVE_ROWS
            or any(row.episode_id != episode_id or row.step_index != tick
                   or row.state_t.time_s != tick*SUBSTEPS*PHYSICS_DT_S
                   or row.next_state_before_reset.time_s != (tick+1)*SUBSTEPS*PHYSICS_DT_S
                   or row.terminated or row.truncated
                   for tick,row in enumerate(rows))):
        raise ValueError('expected exactly 76 contiguous complete archived control rows')
    return raw,rows


def _controller_checkpoint(controller, rows, action):
    """Replay command history on CPU without advancing the rigid-body scene."""
    checkpoint = None
    for tick,row in enumerate(rows):
        request = vector(action(tick,row.state_t),2,'archived policy request').tolist()
        if request != row.requested_action or any(abs(value)>1. for value in request):
            raise ValueError(f'archived policy request changed at tick {tick}')
        commands = controller.plan(row.state_t.time_s,request,SUBSTEPS)
        if ([command.q.tolist() for command in commands] != row.executed_joint_q
                or [command.qd.tolist() for command in commands] != row.executed_joint_qd):
            raise ValueError(f'archived executed controller commands changed at tick {tick}')
        for command in commands:
            controller.commit(command)
        if tick+1 == RESTORE_TICK:
            checkpoint = commands[-1]
    return checkpoint


def audit_prefix(root, cfg, episode_id, action, archive_path):
    """Inspect the full retained prefix and commands without importing Newton."""
    _,rows = _read_prefix(archive_path,episode_id)
    mesh = load_tool(root/'assets/shovel.STL')
    anchor = support_point(mesh,cfg['initial']['anchor_xy_tool_m'])[0]
    controller = OnlineTiltController(Robot(root/'assets/ec66_description.urdf'),cfg,anchor,
                                      PrescribedPath(root/cfg['prescribed']['joint_path'],cfg,moving=True),root/'assets')
    return _controller_checkpoint(controller,rows,action)


def run_tail(root, cfg, episode_id, action, out, state, stage, archive_path):
    """Verify archived row 76 exactly, then append only genuinely new tail rows."""
    if state['physics_steps'] != 0:
        raise ValueError('tail worker starts with zero newly executed physical steps')
    started = time.monotonic()
    raw,rows = _read_prefix(archive_path,episode_id)
    checkpoint = audit_prefix(root,cfg,episode_id,action,archive_path)
    state.update(episode_id=episode_id,control_rows=0,trajectory_file=TRACE,archived_control_rows=ARCHIVE_ROWS,
                 archived_physics_steps=ARCHIVE_ROWS*SUBSTEPS,restore_tick=RESTORE_TICK,
                 exact_verification_transition=False)
    stage('archived_controller_rebuilt')
    env = _make_env(root,cfg,state,stage,expected_device_arch=89)
    first = env.reset(episode_id=episode_id)
    first.validate()
    if ((env.substeps,env.horizon_steps,env.dt,env.step_index,env.physics.steps)
            != (SUBSTEPS,HORIZON,PHYSICS_DT_S,0,0)
            or first != rows[0].state_t or env.episode_id != episode_id or env.done or env.invalid
            or not math.isclose(env.route_end,4.,abs_tol=1e-12,rel_tol=0.)):
        raise ValueError('fresh route does not match the archived initial state or clock')
    initial_read = env.physics.read()
    state.update(initial_state=asdict(first),initial_surface_clearance_m=initial_read['surface_clearance_m'])

    guard_row = rows[RESTORE_TICK]
    saved,evidence = reconstruct(env,asdict(guard_row.state_t),RESTORE_TICK,checkpoint)
    (out/'public_restore.json').write_bytes(_encode(evidence))
    if (saved['state'] != guard_row.state_t or env.state != guard_row.state_t
            or not evidence['both_buffers_equal'] or env.step_index != RESTORE_TICK
            or env.done or env.invalid or env.physics.steps != RESTORE_TICK*SUBSTEPS):
        raise ValueError('public restore did not read back the exact archived comparison state')
    stage('public_state_readback_exact')

    def advance(tick, request):
        before = env.physics.steps
        if (env.step_index != tick or before != tick*SUBSTEPS or before+SUBSTEPS > PHYSICS_CAP
                or state['physics_steps']+SUBSTEPS > PHYSICS_CAP-RESTORE_TICK*SUBSTEPS):
            raise RuntimeError('tail would exceed its frozen physical budget or clock')
        previous = env.state
        try:
            row = env.step(request)
        finally:
            state['physics_steps'] += env.physics.steps-before
            state['physical_time_s'] = env.physics.steps*env.dt
        if (row.episode_id != episode_id or row.step_index != tick or env.step_index != tick+1
                or row.state_t != previous or row.requested_action != request
                or row.next_state_before_reset.previous_action != request
                or row.invalid_reason is not None or env.invalid or len(row.executed_joint_q) != env.physics.steps-before
                or state['physics_steps'] != env.physics.steps-RESTORE_TICK*SUBSTEPS
                or env.done != (row.terminated or row.truncated)
                or row.truncated != (tick+1 == HORIZON and not row.terminated)
                or not math.isclose(row.next_state_before_reset.time_s,env.physics.steps*env.dt,abs_tol=1e-12,rel_tol=0.)):
            raise ValueError(f'tail transition or physical clock changed at tick {tick}')
        return row

    state['setup_wall_s'] = time.monotonic()-started
    stage('verification_cycle_076_start')
    check_start = time.monotonic()
    verification = advance(RESTORE_TICK,guard_row.requested_action)
    state['verification_cycle_wall_s'] = time.monotonic()-check_start
    expected,actual = guard_row.to_dict(),verification.to_dict()
    mismatches = [key for key in expected if expected[key] != actual[key]]
    (out/'verification_cycle.json').write_bytes(_encode({'tick':RESTORE_TICK+1,'exact':not mismatches,
          'mismatched_fields':mismatches,'archived':expected,'verified':actual,'new_physics_steps':state['physics_steps']}))
    if mismatches or state['physics_steps'] != SUBSTEPS:
        raise ValueError(f'period 76 did not exactly reproduce the archived transition: {mismatches}')
    state['exact_verification_transition'] = True
    state['control_rows'] = ARCHIVE_ROWS
    stage('verification_cycle_076_exact')

    action_s = step_s = encode_s = write_s = 0.
    last = verification
    trace = out/TRACE
    positions,velocities = [],[]
    for observation in [first,*(row.next_state_before_reset for row in rows)]:
        position,velocity = _observe(observation)
        positions.append(position)
        velocities.append(velocity)
    max_anchor_error_m = max(row.task['max_anchor_error_m'] for row in rows)
    max_rotation_error_rad = max(row.task['max_tool_rotation_error_rad'] for row in rows)
    with trace.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        loop_start = time.monotonic()
        for tick in range(ARCHIVE_ROWS,HORIZON):
            action_start = time.monotonic()
            request = vector(action(tick,env.state),2,'closed-loop tail action').tolist()
            action_s += time.monotonic()-action_start
            if any(abs(value)>1. for value in request):
                raise ValueError('tail action exceeds the normalized control bound')
            step_start = time.monotonic()
            row = advance(tick,request)
            step_s += time.monotonic()-step_start
            io_start = time.monotonic()
            encoded = _encode(row.to_dict())
            encode_s += time.monotonic()-io_start
            io_start = time.monotonic()
            stream.write(encoded)
            stream.flush()
            write_s += time.monotonic()-io_start
            last = row
            state['control_rows'] += 1
            max_anchor_error_m = max(max_anchor_error_m,row.task['max_anchor_error_m'])
            max_rotation_error_rad = max(max_rotation_error_rad,row.task['max_tool_rotation_error_rad'])
            p,v = _observe(row.next_state_before_reset)
            positions.append(p)
            velocities.append(v)
            stage(f'control_tick_{tick+1:03d}')
            if env.done:
                break
    loop_s = time.monotonic()-loop_start
    if not env.done or (not last.terminated and (env.physics.steps,state['control_rows'],state['physical_time_s'])
                        != (PHYSICS_CAP,HORIZON,4.)):
        raise ValueError('tail did not record a valid complete task ending')

    summary = validate_episode(trace,env.dt,env.substeps,expected_id=episode_id)
    if (summary['physics_steps'] != env.physics.steps or summary['transitions'] != state['control_rows']
            or summary['physics_steps'] != state['physics_steps']+RESTORE_TICK*SUBSTEPS):
        raise ValueError('combined route readback disagrees with physical counters')
    final_read = env.physics.read()
    initial_anchor = np.asarray(initial_read['anchor_position_world_m'])
    final_anchor = np.asarray(final_read['anchor_position_world_m'])
    position,velocity = np.asarray(positions),np.asarray(velocities)
    new_rows = state['control_rows']-ARCHIVE_ROWS
    new_bytes = trace.stat().st_size-len(raw)
    state['episode'] = {
        **summary,
        'motion_complete':last.task['motion_complete'],
        'departure_time_s':last.task['departure_time_s'],
        'final_carrying':last.task['carrying'],
        'final_surface_clearance_m':final_read['surface_clearance_m'],
        'initial_anchor_world_m':initial_anchor.tolist(),
        'final_anchor_world_m':final_anchor.tolist(),
        'anchor_displacement_world_m':(final_anchor-initial_anchor).tolist(),
        'final_anchor_reference_error_m':final_read['anchor_error_m'],
        'max_anchor_reference_error_m':max_anchor_error_m,
        'max_tool_rotation_tracking_error_rad':max_rotation_error_rad,
        'initial_ball_xy_tool_m':position[0,:2].tolist(),
        'final_ball_xy_tool_m':position[-1,:2].tolist(),
        'final_ball_velocity_tool_m_s':velocity[-1].tolist(),
        'max_ball_xy_speed_tool_m_s':float(np.linalg.norm(velocity[:,:2],axis=1).max()),
        'max_ball_xy_displacement_tool_m':float(np.linalg.norm(position[:,:2]-position[0,:2],axis=1).max()),
        'throughput': {
            'scope':'new tail rows only, excludes public restore, period 76 verification and copied prefix',
            'setup_wall_s':state['setup_wall_s'],
            'verification_cycle_wall_s':state['verification_cycle_wall_s'],
            'loop_wall_s':loop_s,
            'episode_step_wall_s':step_s,
            'action_evaluation_wall_s':action_s,
            'json_encoding_wall_s':encode_s,
            'trajectory_write_flush_wall_s':write_s,
            'trajectory_bytes':new_bytes,
            'merged_trajectory_bytes':trace.stat().st_size,
            'recorded_control_rows':new_rows,
            'recorded_physics_steps':state['physics_steps']-SUBSTEPS,
            'control_rows_per_s':new_rows/loop_s,
            'physics_steps_per_s':(state['physics_steps']-SUBSTEPS)/loop_s,
            'trajectory_bytes_per_row':new_bytes/new_rows,
            'worker_elapsed_s_including_validation':time.monotonic()-started,
        },
    }
    state['status'] = 'completed'
    stage('tail_episode_validated')
