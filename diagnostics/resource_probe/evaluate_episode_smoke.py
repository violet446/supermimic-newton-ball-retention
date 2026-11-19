"""Check matched S1 episodes and summarize the control-rate records."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]


def evaluate(out):
    plan = json.loads((out/'plan.json').read_text())
    runs = {}
    for name in ('zero', 'tilt', 'edge'):
        directory = out/name
        result = json.loads((directory/'result.json').read_text())
        if result['status'] != 'completed':
            raise ValueError(f'episode validation failed: {name}')
        cfg = json.loads((directory/'effective_config.json').read_text())
        rows = [json.loads(line) for line in (directory/'transitions.jsonl').read_text().splitlines()]
        runs[name] = cfg, rows, result

    zero_cfg, zero, _ = runs['zero']
    tilt_cfg, tilt, _ = runs['tilt']
    for field in ('initial', 'reference', 'physics', 'ball', 'action', 'episode', 'seed'):
        if zero_cfg[field] != tilt_cfg[field]:
            raise ValueError(f'paired episodes differ in {field}')
    if zero[0]['state_t'] != tilt[0]['state_t'] or len(zero) != len(tilt):
        raise ValueError('paired episodes have mismatched initial state or length')
    states = [[rows[0]['state_t']]+[row['next_state_before_reset'] for row in rows] for rows in (zero, tilt)]
    local_positions, rotations = [], []
    for trajectory in states:
        rotation = Rotation.from_quat([s['tool']['quaternion_xyzw'] for s in trajectory])
        ball = np.asarray([s['ball']['position'] for s in trajectory])
        tool = np.asarray([s['tool']['position'] for s in trajectory])
        rotations.append(rotation)
        local_positions.append(rotation.inv().apply(ball-tool))
    ball_difference = float(np.linalg.norm(local_positions[1]-local_positions[0], axis=1).max())
    max_actual_tilt = float((rotations[1]*rotations[0].inv()).magnitude().max())
    max_roll = max(abs(s['tilt_rad'][0]) for s in states[1])
    criteria = plan['pair_criteria']
    pair_passed = (ball_difference >= criteria['min_ball_tool_frame_position_difference_m']
                   and max_roll >= criteria['min_peak_roll_rad'])
    summaries = {name: result['worker'] for name, (_, _, result) in runs.items()}
    terminal = runs['edge'][1][-1]
    terminal_passed = (terminal['terminated'] and not terminal['truncated']
                       and terminal['task']['termination_reason'] == 'outside_projection'
                       and len(terminal['executed_joint_q']) < summaries['edge']['control_substeps'])
    physics_steps = sum(w['physics_steps'] for w in summaries.values())
    wall_s = sum(result['wall_s'] for _, _, result in runs.values())
    if physics_steps > plan['total_max_substeps'] or wall_s > plan['wall_s']:
        raise ValueError('round budget exceeded')
    report = {'pair_passed': pair_passed, 'terminal_passed': terminal_passed,
              'all_checks_passed': pair_passed and terminal_passed,
              'max_ball_tool_frame_position_difference_m': ball_difference,
              'max_measured_rotation_difference_rad': max_actual_tilt,
              'max_filtered_roll_rad': max_roll, 'pair_criteria': criteria,
              'physics_steps': physics_steps, 'wall_s': wall_s,
              'cases': {name: {k: w[k] for k in ('control_steps', 'physics_steps', 'replay_max_error',
                        'reset_max_error', 'max_anchor_error_m', 'max_tool_rotation_error_rad',
                        'trace_roundtrip_validated', 'terminal_substeps', 'terminated', 'truncated', 'final_task')}
                        for name, w in summaries.items()},
              'scope': 'S1 matched online-action and episode-boundary checks; S2 collection has not started'}
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(evaluate(ROOT/'runs/s1_episode_20261112'), indent=2))
