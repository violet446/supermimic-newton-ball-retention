"""Bounded Newton branch captures for S3 action sensitivity comparisons."""
from dataclasses import asdict
import json
import math

import numpy as np

from ball_retention.collection import replay_error
from ball_retention.contracts import vector
from ball_retention.task import ball_in_tool


def _raw(state):
    terms = ball_in_tool(state.ball, state.tool)
    return np.concatenate((terms['position'], terms['velocity'], terms['ball_omega'])).tolist()


def _plan(cfg, max_steps):
    spec = cfg['action_pairs']
    scenario = spec['scenario']
    if scenario not in ('route', 'edge'):
        raise ValueError('expected route or edge action-pair scenario')
    dt = cfg['physics']['dt_s']
    hz = cfg['physics']['control_hz']
    ticks = round(cfg['episode']['horizon_s']*hz)
    substeps = round(1/(dt*hz))
    if substeps < 1 or not math.isclose(dt*hz*substeps, 1., rel_tol=0., abs_tol=1e-10):
        raise ValueError('control period must contain whole physics substeps')
    horizon = spec['horizon']
    if type(horizon) is not int or horizon < 1:
        raise ValueError('positive integer horizon required')
    snapshots = spec['snapshot_ticks']
    if (not snapshots or any(type(tick) is not int or not 0 <= tick <= ticks-horizon for tick in snapshots)
            or snapshots != sorted(set(snapshots))):
        raise ValueError('snapshot ticks must be unique, increasing and leave room for the horizon')
    amplitudes = list(map(float, spec['amplitudes']))
    if not amplitudes or not all(math.isfinite(a) and 0 < a < 1 for a in amplitudes) or len(set(amplitudes)) != len(amplitudes):
        raise ValueError('amplitudes must be distinct positive normalized perturbations')
    single_scale_ticks = set(spec.get('single_scale_ticks', []))
    if single_scale_ticks-set(snapshots):
        raise ValueError('single-scale ticks must be requested snapshots')
    amplitudes_by_tick = {tick: amplitudes[:1] if tick in single_scale_ticks else amplitudes for tick in snapshots}
    events = {}
    for event in spec['schedule']:
        tick = event['start_step']
        if type(tick) is not int or not 0 <= tick < ticks or tick in events:
            raise ValueError('invalid nominal action schedule tick')
        events[tick] = vector(event['action'], 2, 'nominal request')
    if 0 not in events:
        raise ValueError('nominal action schedule must start at tick 0')
    actions = np.empty((snapshots[-1]+horizon, 2))
    action = events[0]
    for tick in range(len(actions)):
        action = events.get(tick, action)
        actions[tick] = action
    if np.max(np.abs(actions)) > 1.:
        raise ValueError('nominal schedule would be clipped')
    for tick in snapshots:
        if np.max(np.abs(actions[tick:tick+horizon]))+max(amplitudes) >= 1.:
            raise ValueError('paired perturbations must remain inside the unclipped action range')
    bound = (snapshots[-1]+sum(2+4*len(values) for values in amplitudes_by_tick.values())*horizon)*substeps
    if bound > max_steps:
        raise ValueError(f'action pair plan needs up to {bound} physics substeps')
    return scenario, dt, substeps, snapshots, horizon, amplitudes_by_tick, actions, bound


def run(root, cfg, out, max_steps, summary, stage):
    """Replay one nominal prefix and fork each branch from a saved full snapshot."""
    scenario, dt, substeps, snapshots, horizon, amplitudes_by_tick, actions, bound = _plan(cfg, max_steps)
    archive_by_tick = {}
    if 'small_offset_probe' in cfg:
        from ball_retention.small_offset_probe import archived_prefix_records, validate_small_offset_case
        case = validate_small_offset_case(root, cfg)
        if max_steps != bound or bound != case['reserved_substeps']:
            raise ValueError('small-offset worker requires the exact frozen substep count')
        archive_by_tick = archived_prefix_records(root, case)
        tick = case['tick']
        if not np.array_equal(actions[tick:tick+horizon], archive_by_tick[tick]['nominal_actions']):
            raise ValueError('small-offset nominal requests differ from archive')
    from ball_retention.episode import EpisodeEnv

    env = EpisodeEnv(root, cfg, report_stage=stage)
    initial = env.reset(episode_id=f's3-{scenario}')
    initial.validate()
    if env.substeps != substeps or env.dt != dt:
        raise ValueError('episode clock differs from the action pair plan')
    summary.update(scenario=scenario, planned_max_physics_steps=bound,
                   snapshot_ticks=snapshots, completed_snapshots=0, horizon=horizon,
                   amplitudes_by_tick=amplitudes_by_tick, physics_dt_s=dt, control_substeps=substeps)
    stage('action_pairs_initial_state')

    def advance(action):
        if summary['physics_steps']+substeps > max_steps:
            raise RuntimeError('action pair substep budget exhausted')
        before = env.physics.steps
        try:
            row = env.step(action)
        finally:
            applied = env.physics.steps-before
            summary['physics_steps'] += applied
        if len(row.executed_joint_q) != applied:
            raise RuntimeError('recorded control commands disagree with executed Newton substeps')
        if archive_by_tick and (row.invalid_reason is not None or row.terminated or row.truncated
                                    or applied != substeps or not np.array_equal(row.requested_action, action)
                                    or not np.array_equal(row.next_state_before_reset.previous_action, action)):
            raise RuntimeError('small-offset control step rejected, shortened, terminated or clipped')
        return row

    def compare_archived_snapshot(tick, snapshot_state, initial_raw):
        archived = archive_by_tick[tick]
        delta = {
            'archived_snapshot_max_abs_error': replay_error(snapshot_state, archived['initial_state']),
            'archived_initial_raw_max_abs_error': replay_error(initial_raw, archived['initial_raw']),
            'archived_anchor_local_max_abs_error_m': replay_error(env.physics.anchor_local.tolist(),
                                                                  archived['anchor_local_m']),
        }
        summary.setdefault('archive_comparisons_by_tick', {})[str(tick)] = delta
        stage('action_pairs_archived_snapshot_compared')
        if max(delta.values()) > 1e-5:
            raise RuntimeError(f'rebuilt snapshot differs from archive at tick {tick}: {delta}')
        return delta

    def branch(saved, snapshot_state, requested, axis, offset, repeat):
        env.restore(saved)
        if replay_error(asdict(env.state), snapshot_state) != 0.:
            raise RuntimeError('restored branch changed the initial state')
        summary['active_branch'] = {'snapshot_tick':env.step_index, 'axis':axis, 'offset':offset, 'repeat':repeat}
        controls = requested.copy()
        if axis is not None:
            controls[:, axis] += offset
        states_raw, elapsed, counts, executed_requests = [], [], [], []
        for action in controls:
            row = advance(action.tolist())
            next_state = row.next_state_before_reset
            states_raw.append(_raw(next_state))
            elapsed.append(next_state.time_s-snapshot_state['time_s'])
            counts.append(len(row.executed_joint_q))
            executed_requests.append(row.requested_action)
            if row.terminated or row.truncated:
                break
        np.testing.assert_allclose(elapsed, np.cumsum(counts)*dt, rtol=0., atol=1e-9)
        return {'axis':axis, 'offset':offset, 'repeat':repeat, 'actions':executed_requests,
                'states_raw':states_raw, 'substeps':counts, 'elapsed_s':elapsed,
                'terminated':row.terminated, 'truncated':row.truncated,
                'termination_reason':row.task['termination_reason']}

    partial = out/'action_pairs.partial.jsonl'
    report = []
    with partial.open('x', encoding='utf-8') as stream:
        for tick in snapshots:
            summary['active_branch'] = {'snapshot_tick':tick, 'phase':'nominal_prefix'}
            while env.step_index < tick:
                if env.step_index in archive_by_tick:
                    previous_tick = env.step_index
                    state_before_restore = asdict(env.state)
                    compare_archived_snapshot(previous_tick, state_before_restore, _raw(env.state))
                    # Original fork capture restored this earlier snapshot before
                    # advancing the prefix. Restore also rebuilds the solver.
                    env.restore(env.snapshot())
                    if replay_error(asdict(env.state), state_before_restore) != 0.:
                        raise RuntimeError('earlier archive-boundary restore changed the state')
                    summary.setdefault('prefix_restore_ticks', []).append(previous_tick)
                nominal = advance(actions[env.step_index].tolist())
                if nominal.terminated or nominal.truncated:
                    raise RuntimeError(f'nominal rollout ended before snapshot {tick}')
            if env.done or env.invalid or env.step_index != tick:
                raise RuntimeError(f'cannot branch at nominal snapshot {tick}')
            saved = env.snapshot()
            snapshot_state = asdict(env.state)
            initial_raw = _raw(env.state)
            nominal_actions = actions[tick:tick+horizon]
            archive_comparison = {}
            if archive_by_tick:
                archive_comparison = compare_archived_snapshot(tick, snapshot_state, initial_raw)
                summary['archive_comparison'] = archive_comparison
            branches = [branch(saved, snapshot_state, nominal_actions, None, 0., repeat) for repeat in (0, 1)]
            baseline_error = replay_error({k:v for k,v in branches[0].items() if k != 'repeat'},
                                          {k:v for k,v in branches[1].items() if k != 'repeat'})
            noise = np.max(np.abs(np.asarray(branches[0]['states_raw'])-branches[1]['states_raw']), axis=0)
            # In S3, numerical drift is measured per lead and unit. Discrete
            # replay differences still fail above in replay_error.
            if baseline_error > 1e-5 and not cfg['action_pairs'].get('replay_as_noise', False):
                raise RuntimeError(f'baseline restore mismatch at snapshot {tick}: {baseline_error:g}')
            for axis in (0, 1):
                for amplitude in amplitudes_by_tick[tick]:
                    for offset in (-amplitude, amplitude):
                        branches.append(branch(saved, snapshot_state, nominal_actions, axis, offset, 0))
            record = {'scenario':scenario, 'snapshot_tick':tick, 'initial_state':snapshot_state,
                      'initial_raw':initial_raw, 'anchor_local_m':env.physics.anchor_local.tolist(),
                      'nominal_actions':nominal_actions.tolist(), 'branches':branches,
                      'physics_dt_s':dt, 'control_substeps':substeps,
                      'amplitudes':amplitudes_by_tick[tick],
                      'baseline_replay_error':baseline_error, 'baseline_noise_raw_max':noise.tolist(),
                      **archive_comparison}
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            stream.flush()
            env.restore(saved)
            report.append({'snapshot_tick':tick, 'baseline_replay_error':baseline_error,
                           'baseline_noise_raw_max':noise.tolist(), 'branches':len(branches),
                           'early_branches':sum(sum(item['substeps']) < horizon*substeps for item in branches)})
            summary.update(completed_snapshots=len(report), snapshot_report=report)
            stage('action_pairs_snapshot_complete')
    partial.replace(out/'action_pairs.jsonl')
    summary.update(status='action_pairs_completed', data_file='action_pairs.jsonl')
