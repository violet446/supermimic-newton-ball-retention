"""Check signed timing commands and execution coverage for the two same-clock groups."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np

from audit_s3_pitch_new_initial_coverage import TICK
from audit_s3_pitch_state_coverage import BRANCHES, archived_tool
from audit_s3_tool_feature_clock import URDF, command_grid, world_to_tool
from audit_s3_world_increment_generation import CASES
from ball_retention.archive_state_import import _parse_state
from ball_retention.controller import OnlineTiltController
from ball_retention.geometry import load_tool, sphere_clearance, support_point
from ball_retention.prescribed import PrescribedPath
from ball_retention.robot import Robot
from check_s3_actions import schedule_actions
from deterministic_archive_metrics import _full_branch
from preflight_s3_deterministic_action_response import action_sequences, verify_commands
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'pitch_new_group_commands'


def static_commands(plan, cfg, row, robot, mesh, guard):
    state = _parse_state(row['initial_state'])
    tick, substeps, horizon = TICK, plan['control_substeps'], plan['horizon']
    dt = cfg['physics']['dt_s']
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    np.testing.assert_array_equal(anchor, row['anchor_local_m'])
    tool, _, _ = robot.fk(cfg['initial']['joint_q_rad'])
    ball = np.asarray(cfg['initial']['ball_center_world_m'])
    local = tool[:3, :3].T@(ball-tool[:3, 3])
    clearance = sphere_clearance(mesh, local, cfg['ball']['radius_m'])
    if clearance <= 0 or local[0] < .22:
        raise ValueError('unsafe archived ball initialization')
    path = PrescribedPath(ROOT/cfg['prescribed']['joint_path'], cfg, moving=True)
    controller = OnlineTiltController(robot, cfg, anchor, path, ROOT/'assets')
    timeline = schedule_actions(cfg['action_pairs']['schedule'], 0, tick+horizon)
    if row['nominal_actions'] != timeline[tick:]:
        raise ValueError('archived nominal requests differ from the pinned controller schedule')
    nominal = []
    for i, request in enumerate(timeline):
        commands = controller.plan(i*substeps*dt, request, substeps)
        for command in commands:
            controller.commit(command)
            np.testing.assert_array_equal(command.clipped_action, request)
        if i+1 == tick:
            for measured, reconstructed in ((state.joint_q, command.q), (state.joint_qd, command.qd),
                    (state.tilt_rad, command.tilt), (state.tilt_rate_rad_s, command.tilt_rate),
                    (state.previous_action, command.clipped_action)):
                np.testing.assert_array_equal(measured, reconstructed)
            saved = {'controller': controller.snapshot(), 'physics': {'steps': tick*substeps}}
        if i >= tick:
            nominal.append({'q': [x.q.tolist() for x in commands], 'qd': [x.qd.tolist() for x in commands]})
        if i % 24 == 0:
            guard()
    directions, requests = action_sequences(plan, timeline[tick:])
    if any(np.any(v[16:]) for v in directions.values()):
        raise ValueError('signed action extends beyond the established perturbation window')
    commands, peaks = verify_commands(ROOT, plan, cfg, saved, requests, nominal, guard)
    return state, anchor, float(clearance), commands, peaks, requests


def execute_window(commands, state, cfg, anchor, robot, guard):
    dt = cfg['physics']['dt_s']
    ran = command_grid(robot, anchor, state, commands[:16], 16, guard)
    grid = {k: v[8:16] for k, v in ran.items()}
    rot = grid['rotation'][:, 8]
    gravity = np.broadcast_to([0., 0., -cfg['physics']['gravity_m_s2']], (8, 3))
    alpha = (grid['omega'][:, -1]-grid['omega'][:, 0])/(16*dt)
    accel = (grid['anchor_velocity'][:, -1]-grid['anchor_velocity'][:, 0])/(16*dt)
    return np.concatenate([world_to_tool(rot, v) for v in (gravity, grid['omega'][:, 8], alpha, accel)], axis=-1)


def distance(a, b, scale):
    return np.sqrt(np.mean(((a-b)/scale)**2, axis=(0, 2)))


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    if plan['updates'] != 0:
        raise ValueError('unexpected static command scope')
    held = read_held(read(plan['held_design']))
    capture, cfg, _, traces, _ = held
    if capture['case']['tick'] != TICK:
        raise ValueError('unexpected development tick')
    mesh, robot = load_tool(ROOT/'assets/shovel.STL'), Robot(ROOT/URDF)
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    dev = {k: archived_tool(traces[k], robot, anchor, cfg, guard) for k in BRANCHES[2:]}
    scale = np.asarray(read(plan['archive_context'])['scales']['tool'])
    baseline = {}
    for path in plan['native_training']:
        parent = Path(path).parent
        for item in read(path)['entries']:
            if item['split'] != 'train' or item['snapshot_tick'] != TICK:
                raise ValueError('unexpected native training case')
            config = read(item['source_config'])
            a = support_point(mesh, config['initial']['anchor_xy_tool_m'])[0]
            labels = {x['label']: x for pair in item['pairs'] for x in pair['branches']}
            baseline[item['episode_id']] = {k: archived_tool(_full_branch(read(parent/labels[k]['path']), TICK),
                                                           robot, a, config, guard) for k in BRANCHES[2:]}
    screening = read(plan['accepted_screening'])
    candidates = [x for x in screening['candidates'] if x['clock_matches'] and x['nominal_repeatability_passed']]
    if len(candidates) != 2:
        raise ValueError('unexpected same-clock candidate inventory')
    results = {}
    for case in candidates:
        config = read(case['config'])
        row, = [x for x in (json.loads(line) for line in bound(plan, case['trace']).splitlines())
                if x['snapshot_tick'] == TICK]
        state, anchor, clearance, commands, peaks, requests = static_commands(capture, config, row, robot, mesh, guard)
        projected = {k: execute_window(commands[k], state, config, anchor, robot, guard) for k in BRANCHES[2:]}
        effects = {}
        for name in CASES:
            keys = [f'{name}_{sign}' for sign in ('plus', 'minus')]
            actual = np.stack([dev[k] for k in keys])
            old = np.stack([distance(np.stack([x[k] for k in keys]), actual, scale) for x in baseline.values()]).min(axis=0)
            new = distance(np.stack([projected[k] for k in keys]), actual, scale)
            effects[name] = {'existing_nearest_by_lead': old.tolist(), 'candidate_by_lead': new.tolist(),
                             'number_of_strict_improvements': int(np.count_nonzero(new < old))}
        scheduled_release = [x['start_step'] for x in config['action_pairs']['schedule']
                             if x['start_step'] > TICK and x['action'] == [0., 0.]]
        results[case['episode']] = {'group': case['group'], 'seed': case['seed'], 'tick': TICK,
            'initial_clearance_m': clearance, 'branch_peaks': peaks, 'nominal_release_ticks': scheduled_release,
            'common_pitch_prefix_exact': True, 'same_archive_controller_state_at_tick168': True,
            'nominal_requests_match_archive': requests['nominal_1'].tolist() == row['nominal_actions'],
            'executed_tool_distance': effects}
        guard(case['episode'], log=True)

    write_json(out/'candidate_commands.json', {'results': results,
        'convention': 'Executed joint commands are reconstructed analytically for candidate histories. No new ball or contact trajectory is measured.'})
    return {'candidate_groups': len(results), 'complete_static_branches': sum(len(x['branch_peaks']) for x in results.values()),
            'simulator_workers': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
