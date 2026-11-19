"""Verify a common controller history on an independent training ball initial state."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np

from ball_retention.archive_state_import import _parse_state
from ball_retention.controller import OnlineTiltController
from ball_retention.geometry import load_tool, sphere_clearance, support_point
from ball_retention.pair_collection import _initial_key
from ball_retention.prescribed import PrescribedPath
from ball_retention.robot import Robot
from check_s3_actions import schedule_actions
from preflight_s3_deterministic_action_response import action_sequences, verify_commands
from preflight_s3_training_timing_capture import sequence_digest
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'pitch_full_tool_history'


def run(plan, out, guard):
    def read(p):
        return json.loads(bound(plan, p))

    if plan['updates'] != 0:
        raise ValueError('unexpected preflight scope')
    capture, dev, fields, traces, _ = read_held(read(plan['held_design']))
    status = read(plan['accepted_preflight'])
    evidence = read(plan['candidate_identity'])
    cfg = read(plan['candidate_config'])
    case = {**evidence['identity'], 'episode_id': plan['new_episode'], 'history_id': 'matched_full_tool',
            'source_config': (out/'candidate_config.json').relative_to(ROOT).as_posix()}
    if (status['status'] != 'completed' or status['static_branches'] != 8 or status['simulation_steps'] != 0
            or not evidence['shared_pitch_prefix_exact'] or case['split'] != 'train'
            or case['group_id'] == capture['case']['group'] or _initial_key(cfg) != case['physical_initial_sha256']
            or _initial_key(dev) == _initial_key(cfg)
            or any(dev[k] != cfg[k] for k in ('physics','action','robot','reference','prescribed'))
            or any(dev['initial'][k] != cfg['initial'][k] for k in ('joint_q_rad','tool_rotation_world','anchor_xy_tool_m'))):
        raise ValueError('unexpected training identity or unequal controller settings')
    cfg['action_pairs']['schedule'] = dev['action_pairs']['schedule']
    mesh = load_tool(ROOT/'assets/shovel.STL')
    robot = Robot(ROOT/'assets/ec66_description.urdf')
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    pose, _, _ = robot.fk(cfg['initial']['joint_q_rad'])
    ball = np.asarray(cfg['initial']['ball_center_world_m'])
    local = pose[:3,:3].T@(ball-pose[:3,3])
    radius = cfg['ball']['radius_m']
    clearance = sphere_clearance(mesh, local, radius)
    if not (clearance > 0 and local[0] >= .22 and ball[2]-radius > cfg['physics']['floor_z_m']):
        raise ValueError('training ball violates initial geometry checks')
    path = PrescribedPath(ROOT/cfg['prescribed']['joint_path'], cfg, moving=True)
    controller = OnlineTiltController(robot, cfg, anchor, path, ROOT/'assets')
    tick, n, horizon = capture['case']['tick'], capture['control_substeps'], capture['horizon']
    if (tick,n,horizon) != (168,16,32):
        raise ValueError('unexpected capture clock')
    dt = cfg['physics']['dt_s']
    timeline = schedule_actions(cfg['action_pairs']['schedule'], 0, tick+horizon)
    dev_state = _parse_state(fields)
    nominal = []
    for i, action in enumerate(timeline):
        step = controller.plan(i*n*dt, action, n)
        for x in step:
            controller.commit(x)
            np.testing.assert_array_equal(x.clipped_action, action)
        nominal.append({'q':[x.q.tolist() for x in step], 'qd':[x.qd.tolist() for x in step]})
        if i+1 == tick:
            for field, actual in (('joint_q',x.q),('joint_qd',x.qd),('tilt_rad',x.tilt),
                                  ('tilt_rate_rad_s',x.tilt_rate),('previous_action',x.clipped_action)):
                np.testing.assert_array_equal(getattr(dev_state,field), actual)
            saved = {'controller':controller.snapshot(),'physics':{'steps':tick*n}}
            qdd = x.qdd.tolist()
        guard()
    directions, requests = action_sequences(capture, timeline[tick:])
    branches, peaks = verify_commands(ROOT,capture,cfg,saved,requests,nominal[tick:],guard)
    for name in capture['branch_order']:
        if not np.array_equal(requests[name], traces[name]['actions']) or branches[name] != traces[name]['commands']:
            raise ValueError(f'executed tool commands differ from frozen development branch {name}')
    saved_cfg = out/'candidate_config.json'
    write_json(saved_cfg,cfg)
    case['source_config_sha256'] = sha256(saved_cfg.read_bytes()).hexdigest()
    write_json(out/'commands.json', {'case':case,'snapshot_tick':tick,'snapshot_joint_qdd_rad_s2':qdd,
        'snapshot_control_only':saved['controller'],'prefix_actions_sha256':sequence_digest(timeline[:tick]),
        'prefix_requested':timeline[:tick],'prefix_executed':nominal[:tick],
        'directions_Hx2':{k:v.tolist() for k,v in directions.items()},
        'branches':{k:{'requested':requests[k].tolist(),'executed':v} for k,v in branches.items()}})
    write_json(out/'static.json',{'clearance_m':float(clearance),'branch_peaks':peaks,
        'distinct_training_ball_initial':True,'development_commands_exact_all_8_branches':True,
        'compared_control_intervals':len(branches)*horizon,'shared_pitch_prefix_exact':True})
    return {'static_branches':len(branches),'exact_development_tool_commands':True,'new_ball_trajectories':0}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
