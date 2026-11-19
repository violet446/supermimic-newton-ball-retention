"""Bounded live and historical nominal replay after public-state import."""
from dataclasses import asdict
import json
import os
import time

import numpy as np

from ball_retention.archive_restore_plan import validate_case
from ball_retention.archive_state_import import reconstruct
from ball_retention.collection import replay_error
from ball_retention.guard import process_memory
from ball_retention.resources import directory_bytes
from ball_retention.task import ball_in_tool

from archive_restore_metrics import historical_metrics, live_metrics


def write(out, name, data):
    with (out/name).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(root, cfg, out, max_steps, summary, stage):
    started = time.monotonic()
    plan, archive = validate_case(root, cfg, max_steps)
    case, = plan['cases']
    tick = case['tick']
    target = archive[tick]
    max_wall = plan['budget']['max_worker_wall_s']
    substeps = round(1/(cfg['physics']['control_hz']*cfg['physics']['dt_s']))
    nominal = target['nominal_actions']
    if len(nominal) != 32 or substeps != 16:
        raise ValueError('invalid archived nominal sequence or control clock')
    events = {event['start_step']: event['action'] for event in cfg['action_pairs']['schedule']}
    action = events[0]
    actions = []
    for i in range(tick+len(nominal)):
        if i in events:
            action = events[i]
        actions.append(action)
    np.testing.assert_array_equal(actions[tick:], nominal)

    def guard():
        memory = process_memory(os.getpid())
        if (time.monotonic()-started >= max_wall
                or memory['peak_rss_bytes'] > plan['budget']['max_process_rss_gib']*1024**3
                or memory['peak_private_commit_bytes'] > plan['budget']['max_process_commit_gib']*1024**3
                or directory_bytes(out) > plan['budget']['max_new_output_mib']*1024**2):
            raise RuntimeError('archive restore worker budget reached')

    # This command is derived without physics using the pinned nominal path.
    from ball_retention.controller import OnlineTiltController
    from ball_retention.geometry import load_tool, support_point
    from ball_retention.prescribed import PrescribedPath
    from ball_retention.robot import Robot

    mesh = load_tool(root/'assets/shovel.STL')
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    np.testing.assert_array_equal(anchor, target['anchor_local_m'])
    path = PrescribedPath(root/cfg['prescribed']['joint_path'], cfg, moving=True)
    controller = OnlineTiltController(Robot(root/'assets/ec66_description.urdf'), cfg, anchor, path, root/'assets')
    control_dt = substeps*cfg['physics']['dt_s']
    for i, action in enumerate(actions[:tick]):
        for command in controller.plan(i*control_dt, action, substeps):
            controller.commit(command)
        if i % 24 == 0:
            guard()
    for value, key in ((command.q, 'joint_q'), (command.qd, 'joint_qd'), (command.tilt, 'tilt_rad'),
                       (command.tilt_rate, 'tilt_rate_rad_s'), (command.clipped_action, 'previous_action')):
        np.testing.assert_array_equal(value, target['initial_state'][key])
    stage('archive_nominal_controller_verified')

    from ball_retention.episode import EpisodeEnv

    env = EpisodeEnv(root, cfg, report_stage=stage)
    env.reset(episode_id='s3-archive-restore-pilot')
    if env.substeps != substeps:
        raise ValueError('physical clock differs from archived clock')

    def advance(action):
        if summary['physics_steps']+substeps > max_steps or time.monotonic()-started >= max_wall:
            raise RuntimeError('archive restore substep or wall cap')
        before = env.physics.steps
        try:
            row = env.step(action)
        finally:
            summary['physics_steps'] += env.physics.steps-before
        if (len(row.executed_joint_q) != substeps or env.physics.steps-before != substeps or row.invalid_reason is not None
                or row.terminated or row.truncated or row.requested_action != action
                or row.next_state_before_reset.previous_action != action):
            raise RuntimeError('nominal transition invalid, constrained or incomplete')
        return row

    def branch(saved, label):
        env.restore(saved)
        if replay_error(asdict(env.state), asdict(saved['state'])):
            raise RuntimeError('native restore changed the saved public state')
        if env.step_index != tick or env.physics.steps != tick*substeps:
            raise ValueError('branch restore clock mismatch')
        stage(f'{label}_start')
        states, raw, commands, retention, elapsed = [], [], [], [], []
        for action in nominal:
            row = advance(action)
            state = row.next_state_before_reset
            states.append(asdict(state))
            terms = ball_in_tool(state.ball, state.tool)
            raw.append(np.r_[terms['position'], terms['velocity'], terms['ball_omega']].tolist())
            commands.append({'q': row.executed_joint_q, 'qd': row.executed_joint_qd})
            carrying = row.task['carrying']
            retention.append({'terminated': row.terminated, 'truncated': row.truncated,
                              'reason': row.task['termination_reason'],
                              'carrying_candidate': carrying['carrying_candidate'],
                              'outside_grace_active': state.departure_outside_s > 0})
            elapsed.append(state.time_s-saved['state'].time_s)
        np.testing.assert_allclose(elapsed, np.arange(1,len(nominal)+1)*control_dt, atol=1e-9, rtol=0.)
        result = {'actions': nominal, 'substeps': [substeps]*len(nominal), 'elapsed_s': elapsed,
                  'terminated': False, 'truncated': False, 'termination_reason': None,
                  'states': states, 'states_raw': raw, 'commands': commands, 'retention': retention}
        write(out, f'{label}.json', result)
        summary.setdefault('completed_branches', []).append(label)
        guard()
        stage(f'{label}_complete')
        return result

    for i, action in enumerate(actions[:tick]):
        if i in archive:
            earlier = env.snapshot()
            env.restore(earlier)
            if replay_error(asdict(env.state), asdict(earlier['state'])):
                raise RuntimeError('prefix boundary native restore changed public state')
            summary.setdefault('prefix_restore_ticks', []).append(i)
            stage(f'prefix_restore_{i}')
        advance(action)
        if i % 24 == 0:
            guard()
    if summary['physics_steps'] != plan['budget']['nominal_prefix_substeps']:
        raise ValueError('physical prefix count differs from design')
    native = env.snapshot()
    fields = json.loads(json.dumps(asdict(native['state']), allow_nan=False))
    write(out, 'live_public_state.json', fields)
    imported, evidence = reconstruct(env, fields, tick, command)
    if (native['monitor'] != imported['monitor'] or native['deep_count'] != imported['deep_count']
            or replay_error(native['controller'], imported['controller']) > 1e-12):
        raise ValueError('live monitor, penetration counter or controller differs from reconstruction')
    evidence['live_counters_and_controller_verified'] = True
    write(out, 'live_import.json', evidence)
    stage('live_import_verified')
    live = live_metrics([branch(native, f'live_native_{i}') for i in (1,2)],
                        [branch(imported, f'live_imported_{i}') for i in (1,2)])
    write(out, 'live_comparison.json', live)
    summary['live_equivalent'] = bool(live['numerical_equivalent'])
    summary['live_position_resolution_passed'] = bool(live['position_resolution_passed'])
    stage('live_comparison')
    if not (summary['live_equivalent'] and summary['live_position_resolution_passed']):
        raise RuntimeError('live native-to-import nominal replay gate failed')

    historical, evidence = reconstruct(env, target['initial_state'], tick, command)
    write(out, 'historical_import.json', evidence)
    stage('historical_import_verified')
    replays = [branch(historical, f'historical_imported_{i}') for i in (1,2)]
    old = sorted((row for row in target['branches'] if row['axis'] is None and row['offset'] == 0.),
                 key=lambda row: row['repeat'])
    history = historical_metrics(replays, old)
    write(out, 'historical_comparison.json', history)
    if summary['physics_steps'] != max_steps:
        raise ValueError('completed branch count does not exhaust the frozen step count')
    guard()
    summary.update(status='archive_restore_completed', historical_comparison=history, control_accepted=False)
