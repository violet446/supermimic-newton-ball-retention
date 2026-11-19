"""Bounded end-to-end episode checks for the headless S1 worker."""
from dataclasses import asdict
import json
import math
from numbers import Real

import numpy as np

from ball_retention.contracts import (
    EPISODE_SCHEMA, RigidState, State, Transition, validate_window, vector,
)


def _decode_state(record):
    values = dict(record)
    for name in ('ball', 'tool'):
        values[name] = RigidState(**values[name])
    return State(**values)


def _decode_transition(record):
    values = dict(record)
    for name in ('state_t', 'next_state_before_reset'):
        values[name] = _decode_state(values[name])
    return Transition(**values)


def _max_numeric_error(first, second):
    """Compare numeric leaves with a tolerance and discrete leaves exactly."""
    if type(first) is not type(second):
        raise ValueError('replay changed a field type')
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError('replay changed the fields')
        return max((_max_numeric_error(first[k], second[k]) for k in first), default=0.)
    if isinstance(first, list):
        if len(first) != len(second):
            raise ValueError('replay changed a sequence length')
        return max((_max_numeric_error(a, b) for a, b in zip(first, second)), default=0.)
    if isinstance(first, Real) and not isinstance(first, bool):
        if not math.isfinite(first) or not math.isfinite(second):
            raise ValueError('non-finite replay field')
        return abs(first-second)
    if first != second:
        raise ValueError('replay changed a discrete field')
    return 0.


def run(root, cfg, out, max_steps, summary, stage):
    """Run one deterministic schedule and write only pre-reset transitions."""
    from ball_retention.episode import EpisodeEnv

    smoke = cfg['smoke']
    expected_end = smoke.get('expected_end', 'truncated')
    if expected_end not in ('terminated', 'truncated'):
        raise ValueError('smoke.expected_end must be terminated or truncated')
    expected_success = smoke.get('expected_success')
    if expected_success is not None and type(expected_success) is not bool:
        raise ValueError('smoke.expected_success must be boolean')
    schedule = {}
    for event in smoke.get('actions', []):
        start = event['start_step']
        if type(start) is not int or start < 0 or start in schedule:
            raise ValueError('action start steps must be distinct nonnegative integers')
        schedule[start] = vector(event['action'], 2, 'scheduled action').tolist()

    env = EpisodeEnv(root, cfg, device='cuda:0', report_stage=stage)
    substeps = env.substeps
    dt = cfg['physics']['dt_s']
    control_limit = max_steps//substeps
    replay_step = smoke.get('replay_step', 0)
    if type(replay_step) is not int or not 0 <= replay_step < control_limit:
        raise ValueError('replay_step must fit the trace budget')
    if any(step >= control_limit for step in schedule):
        raise ValueError('action schedule exceeds the trace budget')

    episode_id = f's1-smoke:{out.resolve().relative_to(root.resolve()).as_posix()}'
    initial = env.reset(episode_id=episode_id)
    initial.validate()
    initial_physics = env.physics.read()
    summary.update(
        initial_state=asdict(initial), assembly=env.physics.assembly,
        episode_id=episode_id, expected_end=expected_end, expected_success=expected_success,
        transition_schema=EPISODE_SCHEMA, control_substeps=substeps,
        transition_file='transitions.jsonl', replay_control_step=replay_step,
    )
    stage('episode_initial_state')

    def advance(action):
        before = env.physics.steps
        try:
            row = env.step(action)
        finally:
            summary['physics_steps'] += env.physics.steps-before
        if env.physics.steps-before != len(row.executed_joint_q):
            raise ValueError('logged commands disagree with physics step count')
        if summary['physics_steps'] > max_steps+substeps:
            raise ValueError('smoke run exceeded its physics step budget')
        return row

    action = [0., 0.]
    replay_error = None
    last_row = None
    trace = out/'transitions.jsonl'
    with trace.open('w', encoding='utf-8') as stream:
        for index in range(control_limit):
            action = schedule.get(index, action)
            if index == replay_step:
                saved = env.snapshot()
                first = advance(action)
                env.restore(saved)
                last_row = advance(action)
                replay_error = _max_numeric_error(first.to_dict(), last_row.to_dict())
                if replay_error > 1e-5:
                    raise ValueError(f'episode replay error {replay_error:g} exceeds tolerance')
                summary['replay_max_error'] = replay_error
                stage('episode_replay_complete')
            else:
                last_row = advance(action)
            last_row.validate(dt, substeps)
            if (last_row.schema_version != EPISODE_SCHEMA or last_row.step_index != index
                    or last_row.episode_id != episode_id):
                raise ValueError('unexpected transition schema or identity')
            if last_row.requested_action != action:
                raise ValueError('request was changed in the transition record')
            stream.write(json.dumps(last_row.to_dict(), allow_nan=False)+'\n')
            stream.flush()
            summary['control_steps'] = index+1
            if last_row.terminated or last_row.truncated:
                break

    if last_row is None or not getattr(last_row, expected_end):
        raise ValueError(f'episode did not end as {expected_end} within the trace budget')
    summary.update(
        terminated=last_row.terminated, truncated=last_row.truncated,
        final_state=asdict(last_row.next_state_before_reset), final_task=last_row.task,
        terminal_substeps=len(last_row.executed_joint_q), trace_physics_steps=env.physics.steps,
    )
    if expected_success is not None and last_row.task.get('success') is not expected_success:
        raise ValueError(f'episode success did not match expected_success={expected_success}')
    if replay_error is None:
        raise ValueError('episode ended before its replay check')

    with trace.open(encoding='utf-8') as stream:
        rows = [_decode_transition(json.loads(line)) for line in stream]
    validate_window(rows, dt, substeps)
    if rows[0].state_t != initial or rows[-1].to_dict() != last_row.to_dict():
        raise ValueError('decoded trace changed the recorded episode boundary')

    final_physics_steps = env.physics.steps
    try:
        env.step(action)
    except (RuntimeError, ValueError):
        if env.physics.steps != final_physics_steps:
            raise ValueError('rejected post-terminal action advanced physics')
    else:
        raise ValueError('episode accepted a step after termination or truncation')
    finally:
        summary['physics_steps'] += env.physics.steps-final_physics_steps

    reset = env.reset(episode_id=f'{episode_id}:after-reset')
    reset_physics = env.physics.read()
    reset_error = max(
        _max_numeric_error(asdict(initial), asdict(reset)),
        _max_numeric_error(initial_physics['body_q'], reset_physics['body_q']),
        _max_numeric_error(initial_physics['body_qd_com_world'], reset_physics['body_qd_com_world']),
    )
    if env.physics.steps != 0 or reset_error > 1e-6:
        raise ValueError(f'physical reset error {reset_error:g} or step clock mismatch')

    states = [row.next_state_before_reset for row in rows]
    summary.update(
        status='diagnostic_completed',
        reset_max_error=reset_error, trace_roundtrip_validated=True,
        post_terminal_step_rejected=True,
        max_abs_tilt_rad=float(np.max(np.abs([state.tilt_rad for state in states]))),
        max_abs_tilt_rate_rad_s=float(np.max(np.abs([state.tilt_rate_rad_s for state in states]))),
        max_abs_joint_rate_rad_s=float(max(np.max(np.abs(row.executed_joint_qd)) for row in rows)),
        max_anchor_error_m=float(max(row.task['max_anchor_error_m'] for row in rows)),
        max_tool_rotation_error_rad=float(max(row.task['max_tool_rotation_error_rad'] for row in rows)),
    )
    stage('episode_verified')
