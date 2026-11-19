"""Static audit of state recorded in the failed small-offset capture sources."""
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from scipy.spatial.transform import Rotation

from check_s3_actions import ROOT, digest, schedule_actions, write_json
from ball_retention.contracts import RigidState, State
from ball_retention.controller import OnlineTiltController
from ball_retention.geometry import load_tool, sphere_clearance, support_point
from ball_retention.guard import process_memory
from ball_retention.prescribed import PrescribedPath
from ball_retention.resources import directory_bytes
from ball_retention.retention import ToolEnvelope
from ball_retention.robot import Robot
from ball_retention.small_offset_probe import DESIGN, archived_record, build_case, verify_design
from ball_retention.task import ball_in_tool, reference


OUT = ROOT/'runs/s3_archived_snapshot_audit_20261114'
LIMITS = {'wall_s': 120, 'rss_bytes': 1024**3, 'commit_bytes': int(1.5*1024**3), 'output_bytes': 8*1024**2}


def audit(root, cfg, case, record, substeps, mesh, robot, guard):
    fields = record['initial_state']
    state = State(**{**fields, 'ball': RigidState(**fields['ball']), 'tool': RigidState(**fields['tool'])})
    state.validate()
    dt = cfg['physics']['dt_s']*substeps
    if not np.isclose(state.time_s, case['tick']*dt, rtol=0., atol=1e-12):
        raise ValueError('archive clock differs from snapshot tick')
    if state.reference_phase != reference(state.time_s, cfg)['phase']:
        raise ValueError('archive route phase does not match the pinned reference')
    raw = ball_in_tool(state.ball, state.tool)
    np.testing.assert_array_equal(np.r_[raw['position'], raw['velocity'], raw['ball_omega']], record['initial_raw'])
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    np.testing.assert_array_equal(anchor, record['anchor_local_m'])
    path = PrescribedPath(root/cfg['prescribed']['joint_path'], cfg, moving=True)
    controller = OnlineTiltController(robot, cfg, anchor, path, root/'assets')
    for tick, action in enumerate(schedule_actions(cfg['action_pairs']['schedule'], 0, case['tick'])):
        for command in controller.plan(tick*dt, action, substeps):
            controller.commit(command)
        if tick % 24 == 0:
            guard()
    for actual, expected in ((command.q, state.joint_q), (command.qd, state.joint_qd),
                             (command.tilt, state.tilt_rad), (command.tilt_rate, state.tilt_rate_rad_s),
                             (command.clipped_action, state.previous_action)):
        np.testing.assert_array_equal(actual, expected)
    filtered = np.array(state.tilt_rad)+controller.tau*np.array(state.tilt_rate_rad_s)
    np.testing.assert_allclose(command.filter_input_state, filtered, rtol=0., atol=1e-15)
    tool_pose, jacobian, _ = robot.fk(state.joint_q)
    measured = Rotation.from_quat(state.tool.quaternion_xyzw).as_matrix()
    origin_error = np.linalg.norm(tool_pose[:3, 3]-state.tool.position)
    rotation_error = Rotation.from_matrix(tool_pose[:3, :3]@measured.T).magnitude()
    twist_error = np.max(abs(jacobian@state.joint_qd-np.r_[state.tool.linear_velocity_origin,
                                                         state.tool.angular_velocity]))
    if max(origin_error, rotation_error, twist_error) > 2e-5:
        raise ValueError('archive tool state disagrees with independent FK')
    radius = cfg['ball']['radius_m']
    envelope = ToolEnvelope(mesh, cfg['initial']['tool_rotation_world'], radius, cfg['episode']['departure_margin_m'])
    geometry = envelope.measure(raw['position'])
    gap = sphere_clearance(mesh, raw['position'], radius)
    if (state.departure_outside_s != 0 or geometry['outside_with_margin'] or geometry['below_entire_tool']
            or gap < -.002 or state.ball.position[2] <= cfg['physics']['floor_z_m']+radius+.001):
        raise ValueError('archive fails the interior, zero-counter reconstruction scope')
    return {'episode': case['episode'], 'tick': case['tick'], 'seed': case['seed'],
            'nominal_controller_state_array_equal': True, 'raw_ball_state_array_equal': True,
            'physics_steps_at_snapshot': case['tick']*substeps,
            'joint_qdd_from_nominal_prefix_rad_s2': command.qdd.tolist(),
            'filter_input_state_max_abs_error_rad': float(np.max(abs(command.filter_input_state-filtered))),
            'tool_fk_origin_error_m': float(origin_error), 'tool_fk_angle_error_rad': float(rotation_error),
            'tool_fk_twist_max_error_si': float(twist_error), 'surface_gap_m': float(gap),
            'departure_outside_s': state.departure_outside_s, 'interior_geometry': geometry}


def run():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)

    def guard():
        memory = process_memory(os.getpid())
        if (time.monotonic()-started >= LIMITS['wall_s'] or memory['peak_rss_bytes'] > LIMITS['rss_bytes']
                or memory['peak_private_commit_bytes'] > LIMITS['commit_bytes']
                or directory_bytes(OUT) > LIMITS['output_bytes']):
            raise RuntimeError('static archive audit budget exceeded')
        return memory

    try:
        design = verify_design(ROOT)
        mesh = load_tool(ROOT/'assets/shovel.STL')
        robot = Robot(ROOT/'assets/ec66_description.urdf')
        cases = []
        for case in design['cases']:
            guard()
            cases.append(audit(ROOT, build_case(ROOT, design, case), case, archived_record(ROOT, case),
                               design['control_substeps'], mesh, robot, guard))
        if 'newton' in sys.modules or 'warp' in sys.modules:
            raise RuntimeError('static audit imported the physics backend')
        paths = [Path(__file__), ROOT/'scripts/check_s3_actions.py', ROOT/DESIGN,
                 *sorted((ROOT/'src/ball_retention').glob('*.py'))]
        write_json(OUT/'audit.json', {'status': 'passed', 'cases': cases, 'cpu_threads': 1,
                   'wall_s': time.monotonic()-started, **guard(), 'limits': LIMITS,
                   'seed': None, 'physics_substeps': 0, 'test_episode_files_read': False,
                   'sources_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}})
        guard()
    except Exception as error:
        write_json(OUT/'failure.json', {'status': 'stopped', 'error': str(error), 'physics_substeps': 0})
        raise


if __name__ == '__main__':
    run()
