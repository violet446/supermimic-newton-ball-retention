"""Preflight one training initial condition with a matched late tool command."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np

from audit_s3_pitch_new_initial_coverage import TICK
from audit_s3_pitch_state_coverage import BRANCHES, archived_tool
from audit_s3_world_increment_generation import CASES
from ball_retention.controller import OnlineTiltController
from ball_retention.geometry import load_tool, support_point
from ball_retention.pair_collection import _initial_key
from ball_retention.prescribed import PrescribedPath
from ball_retention.robot import Robot
from check_s3_actions import schedule_actions
from preflight_s3_deterministic_action_response import action_sequences, verify_commands
from preflight_s3_pitch_new_group_commands import URDF, distance, execute_window
from preflight_s3_training_timing_capture import sequence_digest
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'pitch_matched_tool_history'


def run(plan, out, guard):
    def read(p):
        return json.loads(bound(plan, p))

    if plan['updates'] != 0:
        raise ValueError('unexpected matched-tool static scope')
    prior = read(plan['accepted_screening'])['candidates']
    original_commands = read(plan['accepted_old_commands'])['results']
    eligible = [row for row in prior if row['clock_matches'] and row['nominal_repeatability_passed']]
    selected = min(eligible, key=lambda row: row['distance']['combined']['median'])
    if selected['episode'] != plan['base_episode'] or selected['episode'] not in original_commands:
        raise ValueError('source training candidate changed')
    original = read(selected['config'])
    archive = read(selected['manifest'])
    entry, = [r for r in archive['entries'] if r['episode_id'] == selected['episode']]
    if (entry['split'] != 'train' or _initial_key(original) != selected['physical_initial_key']
            or not selected['new_physical_initial_group']):
        raise ValueError('candidate training group identity changed')
    modified = read(selected['config'])
    previous = modified['action_pairs']['schedule']
    new_tail = plan['late_schedule']
    if new_tail != [{'start_step': 142, 'action': [.16, 0.]}, {'start_step': 185, 'action': [0., 0.]}]:
        raise ValueError('unexpected matched tool command')
    modified['action_pairs']['schedule'] = [r for r in previous if r['start_step'] < 142]+new_tail
    identity = {'episode_id': plan['new_episode'], 'group_id': entry['group_id'], 'split': 'train',
        'seed': entry['seed'], 'physical_initial_sha256': selected['physical_initial_key'], 'history_id': 'matched_late_tool'}

    capture, cfg_dev, _, traces, _ = read_held(read(plan['held_design']))
    if capture['case']['tick'] != TICK or cfg_dev['action_pairs']['schedule'][-2:] != new_tail:
        raise ValueError('command operating point changed')
    robot = Robot(ROOT/URDF)
    mesh = load_tool(ROOT/'assets/shovel.STL')
    anchor = support_point(mesh, modified['initial']['anchor_xy_tool_m'])[0]
    path = PrescribedPath(ROOT/modified['prescribed']['joint_path'], modified, moving=True)
    controller = OnlineTiltController(robot, modified, anchor, path, ROOT/'assets')
    substeps, horizon = capture['control_substeps'], capture['horizon']
    dt = modified['physics']['dt_s']
    first_archive = json.loads(bound(plan, selected['trace']).splitlines()[0])
    old_timeline = schedule_actions(previous, 0, TICK+horizon)
    timeline = schedule_actions(modified['action_pairs']['schedule'], 0, TICK+horizon)
    np.testing.assert_array_equal(timeline[:142], old_timeline[:142])
    prefix_and_nominal = []
    for k, request in enumerate(timeline):
        steps = controller.plan(k*substeps*dt, request, substeps)
        for command in steps:
            controller.commit(command)
            np.testing.assert_array_equal(request, command.clipped_action)
        prefix_and_nominal.append({'q': [s.q.tolist() for s in steps], 'qd': [s.qd.tolist() for s in steps]})
        if k+1 == first_archive['snapshot_tick']:
            state = first_archive['initial_state']
            for field, expected in (('joint_q', command.q), ('joint_qd', command.qd), ('tilt_rad', command.tilt),
                                    ('tilt_rate_rad_s', command.tilt_rate), ('previous_action', command.clipped_action)):
                np.testing.assert_array_equal(state[field], expected)
        if k+1 == TICK:
            saved = {'controller': controller.snapshot(), 'physics': {'steps': TICK*substeps}}
            start = SimpleNamespace(joint_q=command.q, joint_qd=command.qd)
            qdd = command.qdd.tolist()
        if k % 24 == 0:
            guard()
    directions, requests = action_sequences(capture, timeline[TICK:])
    commands, peaks = verify_commands(ROOT, capture, modified, saved, requests, prefix_and_nominal[TICK:], guard)
    if not all(np.array_equal(np.asarray(commands[k][0]['q']).shape, (substeps, 6)) for k in BRANCHES):
        raise ValueError('unexpected command layout')
    scale = np.asarray(read(plan['archive_context'])['scales']['tool'])
    a_dev = support_point(mesh, cfg_dev['initial']['anchor_xy_tool_m'])[0]
    effects = {}
    for case in CASES:
        keys = [f'{case}_{side}' for side in ('plus', 'minus')]
        actual = np.stack([archived_tool(traces[k], robot, a_dev, cfg_dev, guard) for k in keys])
        projected = np.stack([execute_window(commands[k], start, modified, anchor, robot, guard) for k in keys])
        d = distance(projected, actual, scale)
        baseline = np.asarray(original_commands[selected['episode']]['executed_tool_distance'][case]['existing_nearest_by_lead'])
        effects[case] = {'existing_nearest': baseline.tolist(), 'matched_tool_candidate': d.tolist(),
                         'num_improved_leads': int(np.sum(d < baseline))}
    write_json(out/'candidate_config.json', modified)
    write_json(out/'commands.json', {'case': identity, 'snapshot_tick': TICK,
        'snapshot_joint_qdd_rad_s2': qdd,
        'prefix_actions_sha256': sequence_digest(timeline[:TICK]), 'prefix_requested': timeline[:TICK],
        'prefix_executed': prefix_and_nominal[:TICK], 'snapshot_control_only': saved['controller'],
        'directions_Hx2': {k: v.tolist() for k, v in directions.items()},
        'branches': {k: {'requested': requests[k].tolist(), 'executed': v} for k, v in commands.items()}})
    write_json(out/'matched_tool.json', {'identity': identity, 'earliest_changed_tick': 142,
        'archived_nominal_checkpoint_verified': first_archive['snapshot_tick'], 'branch_peaks': peaks,
        'shared_pitch_prefix_exact': True, 'distance': effects})
    return {'static_branches': len(peaks), 'training_initial_group': identity['group_id'],
            'new_ball_trajectories': 0, 'deployment_or_control_accepted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
