"""Evaluate recorded full-arm runs without importing or launching Newton."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.robot import Robot


def evaluate(out):
    plan = json.loads((out/'plan.json').read_text())
    robot = Robot(ROOT/'assets/ec66_description.urdf')
    cases, traces = [], {}
    for name, steps in [('static2s_attempt3', 1920), ('move4s', 3840), ('move_half_dt2s', 3840)]:
        folder = out/name
        result = json.loads((folder/'result.json').read_text())
        worker = result['worker']
        if result['status'] != 'completed' or worker['status'] != 'diagnostic_completed':
            raise ValueError(f'{name} incomplete; do not omit failed runs')
        rows = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
        cfg = json.loads((folder/'effective_config.json').read_text())
        hz = round(1/cfg['physics']['dt_s'])
        if len(rows) != steps or [r['step'] for r in rows] != list(range(1, steps+1)):
            raise ValueError('incomplete or discontinuous trace')
        np.testing.assert_allclose([r['time_s'] for r in rows], np.arange(1, steps+1)/hz, atol=1e-12, rtol=0)
        bid = worker['assembly']['ball_id']
        p = np.array([r['body_q'][bid][:3] for r in rows])
        v = np.array([r['body_qd_com_world'][bid][:3] for r in rows])
        omega = np.array([r['body_qd_com_world'][bid][3:] for r in rows])
        traces[name] = (hz, p, v, omega)
        body_labels = worker['assembly']['body_labels']
        arm_pos_error, arm_rot_error = 0., 0.
        # Check every material link, not only the flange, on 60Hz boundaries.
        for row in rows[hz//60-1::hz//60]:
            _, _, poses = robot.fk(row['command_q_rad'])
            for i, label in enumerate(body_labels):
                if i == bid:
                    continue
                actual = np.asarray(row['body_q'][i])
                expected = poses[label]
                arm_pos_error = max(arm_pos_error, float(np.linalg.norm(actual[:3]-expected[:3, 3])))
                arm_rot_error = max(arm_rot_error, float(Rotation.from_matrix(Rotation.from_quat(actual[3:]).as_matrix()@expected[:3, :3].T).magnitude()))
        if arm_pos_error > 2e-5 or arm_rot_error > 2e-4:
            raise ValueError(f'{name} full-link FK mismatch')
        resources = [json.loads(line) for line in (folder/'resources.jsonl').read_text().splitlines()]
        peaks = resources+worker['phase_events']
        initial_anchor = np.array(worker['initial_state']['anchor_position_world_m'])
        initial_ball = np.array(worker['initial_state']['body_q'][bid][:3])
        cases.append({'name': name, 'hz': hz, 'trace_seconds': rows[-1]['time_s'],
            'physics_steps_including_replay': worker['physics_steps'], 'wall_s': result['wall_s'],
            'max_anchor_error_m': max(r['anchor_error_m'] for r in rows),
            'max_anchor_velocity_error_m_s': max(r['anchor_velocity_error_m_s'] for r in rows),
            'max_tool_rotation_error_rad': max(r['tool_rotation_error_rad'] for r in rows),
            'max_all_arm_link_position_error_m_at_60hz': arm_pos_error,
            'max_all_arm_link_rotation_error_rad_at_60hz': arm_rot_error,
            'fk_ball_max_change': max(r['fk_ball_max_change'] for r in rows),
            'max_penetration_m': max(0., -min(r['surface_clearance_m'] for r in rows)),
            'max_surface_clearance_m': max(r['surface_clearance_m'] for r in rows),
            'floor_reached': any(r['floor_reached'] for r in rows),
            'anchor_displacement_world_m': (np.array(rows[-1]['anchor_position_world_m'])-initial_anchor).tolist(),
            'ball_displacement_world_m': (p[-1]-initial_ball).tolist(),
            'ball_max_world_excursion_after_0p1s_m': float(np.linalg.norm(p[hz//10-1:]-p[hz//10-1], axis=1).max()),
            'ball_final_linear_speed_m_s': float(np.linalg.norm(v[-1])),
            'replay_start_time_s': worker['replay_start_step']/hz,
            **{k: worker[k] for k in ('replay_position_error_m', 'replay_quaternion_component_error', 'replay_velocity_error', 'reset_max_error')},
            'peak_rss_gib': max(r['peak_rss_bytes'] for r in peaks)/1024**3,
            'peak_commit_gib': max(r['peak_private_commit_bytes'] for r in peaks)/1024**3,
            'max_sampled_gpu_used_gib': max(r['gpu_used_bytes'] for r in resources)/1024**3})
    criteria = plan['dynamic_comparison_criteria']
    compare_times = np.arange(6, 121)/60
    aligned = []
    for name in ('move4s', 'move_half_dt2s'):
        hz, *arrays = traces[name]
        indices = np.rint(compare_times*hz).astype(int)-1
        aligned.append([a[indices] for a in arrays])
    errors = [float(np.linalg.norm(a-b, axis=1).max()) for a, b in zip(*aligned)]
    comparison = {'interval_s': [0.1, 2.0], 'sample_hz': 60, 'sample_count': len(compare_times),
        'interpolation': False, 'max_ball_position_difference_m': errors[0],
        'max_ball_linear_velocity_difference_m_s': errors[1],
        'max_ball_angular_velocity_difference_rad_s': errors[2],
        'preregistered_thresholds': criteria,
        'passed': errors[0] <= criteria['max_ball_position_difference_m'] and errors[1] <= criteria['max_ball_linear_velocity_difference_m_s']}
    summary = {'cases': cases, 'dynamic_comparison': comparison,
        'physics_steps': sum(c['physics_steps_including_replay'] for c in cases),
        'wall_s': sum(c['wall_s'] for c in cases), 's1_complete': False, 's2_started': False,
        'scope': 'one initial state, zero tilt, prescribed kinematic arm; dynamic comparison covers only first 2s, not deceleration or online tilt; no validated edge-drop oracle or task success rate'}
    (out/'physical_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='runs/s1_arm_20261112')
    args = parser.parse_args()
    out = (ROOT/args.run).resolve()
    out.relative_to(ROOT)
    print(json.dumps(evaluate(out), indent=2))
