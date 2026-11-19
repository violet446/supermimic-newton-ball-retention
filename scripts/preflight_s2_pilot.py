"""Check the fixed S2 pilot's initial geometry and online commands on CPU."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ball_retention.controller import OnlineTiltController
from ball_retention.geometry import load_tool, sphere_clearance, support_point
from ball_retention.prescribed import PrescribedPath
from ball_retention.retention import DepartureMonitor, ToolEnvelope
from ball_retention.robot import Robot
from ball_retention.support import CarryingRegion
from ball_retention.task import reference


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_cases():
    started = time.perf_counter()
    out = ROOT / 'runs/s2_pilot_20261112'
    cases_path = out / 'cases.json'
    plan = json.loads(cases_path.read_text(encoding='utf-8'))
    config_path = ROOT / plan['base_config']
    baseline = json.loads(config_path.read_text(encoding='utf-8'))
    dt = baseline['physics']['dt_s']
    hz = baseline['physics']['control_hz']
    substeps = round(1 / (dt * hz))
    asset_root = ROOT / 'assets'
    tool_path = asset_root / 'shovel.STL'
    mesh = load_tool(tool_path)
    robot = Robot(asset_root / 'ec66_description.urdf')
    path_file = ROOT / baseline['prescribed']['joint_path']
    path = PrescribedPath(path_file, baseline, moving=True)
    initial_tool = robot.fk(path.command(0.)[0])[0]
    anchor = support_point(mesh, baseline['initial']['anchor_xy_tool_m'])[0]
    rotation = baseline['initial']['tool_rotation_world']
    radius = baseline['ball']['radius_m']
    carrying = CarryingRegion(mesh, rotation, radius)
    envelope = ToolEnvelope(mesh, rotation, radius, baseline['episode']['departure_margin_m'])
    results = []
    for case in plan['cases']:
        cfg = copy.deepcopy(baseline)
        cfg['initial'].update(case['initial'])
        cfg['episode']['horizon_s'] = case['horizon_s']
        centre = np.asarray(cfg['initial']['ball_center_world_m'])
        local = initial_tool[:3, :3].T @ (centre - initial_tool[:3, 3])
        gap = sphere_clearance(mesh, local, radius)
        support = carrying.measure(local)
        floor = centre[2] - radius <= cfg['physics']['floor_z_m']
        departure = DepartureMonitor(envelope, cfg['episode']['outside_grace_s']).update(0., local, floor)
        if gap < 0 or not support['within_bowl'] or departure['terminated'] or departure['outside_with_margin']:
            raise ValueError(f'{case["id"]}: inadmissible initial ball geometry')

        controller = OnlineTiltController(robot, cfg, anchor, path, asset_root)
        schedule = {item['start_step']: item['action'] for item in case['actions']}
        ticks = round(hz * case['horizon_s'])
        if 0 not in schedule or len(schedule) != len(case['actions']) or max(schedule) >= ticks or min(schedule) < 0:
            raise ValueError(f'{case["id"]}: invalid action schedule')
        if not 0 <= case['replay_step'] < ticks:
            raise ValueError(f'{case["id"]}: replay outside horizon')
        max_tilt = np.zeros(2)
        max_tilt_rate = np.zeros(2)
        max_anchor_error = max_rotation_error = max_speed = max_acceleration = 0.
        action = schedule[0]
        for tick in range(ticks):
            action = schedule.get(tick, action)
            if np.max(np.abs(action)) > .4:
                raise ValueError(f'{case["id"]}: request outside pilot action range')
            commands = controller.plan(tick * substeps * dt, action, substeps)
            for command in commands:
                tool = robot.fk(command.q)[0]
                ref = reference(command.time_s, cfg)
                anchor_error = np.linalg.norm(tool[:3, :3] @ anchor + tool[:3, 3] - ref['anchor_position'])
                rotation_error = Rotation.from_matrix(tool[:3, :3].T @ command.rotation).magnitude()
                max_anchor_error = max(max_anchor_error, float(anchor_error))
                max_rotation_error = max(max_rotation_error, float(rotation_error))
                max_speed = max(max_speed, float(np.max(np.abs(command.qd))))
                max_acceleration = max(max_acceleration, float(np.max(np.abs(command.qdd))))
                max_tilt = np.maximum(max_tilt, np.abs(command.tilt))
                max_tilt_rate = np.maximum(max_tilt_rate, np.abs(command.tilt_rate))
                controller.commit(command)
        position_tolerance = cfg['admission']['motion_position_tolerance_m']
        angle_tolerance = np.deg2rad(cfg['admission']['motion_angle_tolerance_deg'])
        if max_anchor_error > position_tolerance or max_rotation_error > angle_tolerance:
            raise ValueError(f'{case["id"]}: reference tracking tolerance exceeded')
        row = {
            'id': case['id'],
            'controls_checked': ticks,
            'substep_commands_checked': ticks * substeps,
            'initial_ball_local_m': local.tolist(),
            'initial_mesh_gap_m': gap,
            'initial_carrying': support,
            'max_abs_tilt_deg': np.rad2deg(max_tilt).tolist(),
            'max_abs_tilt_rate_deg_s': np.rad2deg(max_tilt_rate).tolist(),
            'max_abs_joint_rate_rad_s': max_speed,
            'max_abs_joint_acceleration_rad_s2': max_acceleration,
            'max_anchor_error_m': max_anchor_error,
            'max_rotation_error_rad': max_rotation_error,
        }
        results.append(row)
        print(f'{case["id"]}: {ticks} controls, initial mesh gap {gap * 1e3:.3f} mm', flush=True)

    sources = [cases_path, config_path, path_file, tool_path, Path(__file__),
               ROOT / 'src/ball_retention/controller.py', ROOT / 'src/ball_retention/clearance.py']
    report = {
        'scope': 'CPU initial geometry and complete prescribed action schedules. Controller planner checks joint constraints and swept clearance. Physical endings, contacts and replay require Newton.',
        'passed': True,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
        'cases': results,
        'wall_s': time.perf_counter() - started,
    }
    (out / 'preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    check_cases()
