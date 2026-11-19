"""Collect one bounded Newton episode as v0.2 JSONL."""
import json
import math
from numbers import Real

from .contracts import EPISODE_SCHEMA, vector


def replay_error(first, second):
    """Compare numeric leaves and require identical discrete fields."""
    if type(first) is not type(second):
        raise ValueError('replay changed a field type')
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError('replay changed the fields')
        return max((replay_error(first[k], second[k]) for k in first), default=0.)
    if isinstance(first, list):
        if len(first) != len(second):
            raise ValueError('replay changed a sequence length')
        return max((replay_error(a, b) for a, b in zip(first, second)), default=0.)
    if isinstance(first, Real) and not isinstance(first, bool):
        if not math.isfinite(first) or not math.isfinite(second):
            raise ValueError('non-finite replay field')
        return abs(first-second)
    if first != second:
        raise ValueError('replay changed a discrete field')
    return 0.


def collect_episode(root, cfg, out, max_steps, summary, stage):
    from .dataset import validate_episode
    from .episode import EpisodeEnv

    case = cfg['collection']['case']
    dt = cfg['physics']['dt_s']
    substeps = round(1/(cfg['physics']['control_hz']*dt))
    ticks = round(cfg['episode']['horizon_s']*cfg['physics']['control_hz'])
    if ticks*substeps != max_steps:
        raise ValueError('trace budget must equal the configured horizon')
    replay_tick = case['replay_step']
    if type(replay_tick) is not int or not 0 <= replay_tick < ticks:
        raise ValueError('replay tick outside horizon')
    schedule = {}
    for event in case['actions']:
        tick = event['start_step']
        action = vector(event['action'], 2, 'scheduled action')
        if type(tick) is not int or not 0 <= tick < ticks or tick in schedule:
            raise ValueError('invalid action schedule tick')
        if max(abs(action)) > cfg['collection']['max_abs_request']:
            raise ValueError('action outside the pilot scope')
        schedule[tick] = action.tolist()
    if 0 not in schedule:
        raise ValueError('action schedule must start at tick 0')

    env = EpisodeEnv(root, cfg, report_stage=stage)
    initial = env.reset(episode_id=case['id'])
    initial.validate()
    summary.update(episode_id=case['id'], group_id=case['group_id'], split=case['split'],
                   planning_rejections=0, transition_schema=EPISODE_SCHEMA)
    stage('collection_initial_state')

    def advance(action):
        before = env.physics.steps
        try:
            row = env.step(action)
        except (ValueError, ArithmeticError) as exc:
            # Only a failure before simulator state changed is a planning rejection.
            if env.physics.steps == before and not env.invalid:
                summary.update(planning_rejections=1, rejected_step=env.step_index,
                               rejected_action=action, rejection_reason=str(exc))
            raise
        finally:
            summary['physics_steps'] += env.physics.steps-before
        if len(row.executed_joint_q) != env.physics.steps-before:
            raise RuntimeError('executed commands disagree with actual physics steps')
        return row

    action = schedule[0]
    max_replay_error = None
    initial_snapshot = env.snapshot()
    partial = out/'transitions.partial.jsonl'
    with partial.open('x', encoding='utf-8') as stream:
        for tick in range(ticks):
            action = schedule.get(tick, action)
            saved = env.snapshot() if tick == replay_tick else None
            row = advance(action)
            if tick == 0:
                first_row = row.to_dict()
            if saved is not None:
                env.restore(saved)
                replay = advance(action)
                max_replay_error = replay_error(row.to_dict(), replay.to_dict())
                if max_replay_error > 1e-5:
                    raise RuntimeError('episode replay exceeded numerical tolerance')
                row = replay
                summary['replay_max_error'] = max_replay_error
                stage('collection_replay_checked')
            stream.write(json.dumps(row.to_dict(), allow_nan=False)+'\n')
            stream.flush()
            if summary['physics_steps'] > max_steps+substeps:
                raise RuntimeError('episode physics budget exceeded')
            if row.terminated or row.truncated:
                break
    if max_replay_error is None:
        # A valid early departure can precede the scheduled replay tick.
        env.restore(initial_snapshot)
        max_replay_error = replay_error(first_row, advance(schedule[0]).to_dict())
        if max_replay_error > 1e-5:
            raise RuntimeError('episode replay exceeded numerical tolerance')
        summary.update(replay_max_error=max_replay_error, replay_used_initial_tick=True)
        stage('collection_replay_checked')
    metrics = validate_episode(partial, dt, substeps, expected_id=case['id'])
    partial.replace(out/'transitions.jsonl')
    summary.update(status='collection_completed', data_file='transitions.jsonl', episode=metrics)
    stage('collection_verified')
