"""Reproduce S1 full-path comparison and departure diagnostic evidence."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.geometry import load_tool
from ball_retention.retention import ToolEnvelope, DepartureMonitor


def evaluate():
    out = ROOT/'runs/s1_retention_20261112'
    plan = json.loads((out/'plan.json').read_text())
    cases, aligned = [], []
    mesh = load_tool(ROOT/'assets/shovel.STL')
    baseline = ROOT/plan['comparison']['baseline']
    folders = [baseline, out/'move_half_dt4s', out/'airborne1s', out/'edge_exit2s']
    for folder in folders:
        result = json.loads((folder/'result.json').read_text())
        if result['status'] != 'completed':
            raise ValueError(f'case failed: {folder}')
        worker = result['worker']
        cfg = json.loads((folder/'effective_config.json').read_text())
        rows = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
        hz = round(1/cfg['physics']['dt_s'])
        bid = worker['assembly']['ball_id']
        np.testing.assert_allclose([r['time_s'] for r in rows], np.arange(1, len(rows)+1)/hz, atol=1e-12, rtol=0)
        monitor = DepartureMonitor(ToolEnvelope(mesh, cfg['initial']['tool_rotation_world'], cfg['ball']['radius_m']))
        initial = worker['initial_state']
        monitor.update(0, initial['ball_local_m'])
        first_departure = None
        for row in rows:
            measured = monitor.update(row['time_s'], row['ball_local_m'], row['floor_reached'])
            if 'departure' in row and measured != row['departure']:
                raise ValueError('recorded monitor labels cannot be reproduced')
            if measured['terminated'] and first_departure is None:
                first_departure = {'time_s': row['time_s'], **measured}
        resources = [json.loads(line) for line in (folder/'resources.jsonl').read_text().splitlines()]
        peaks = resources+worker['phase_events']
        case = {'name': folder.name, 'trace_steps': len(rows), 'physics_steps': worker['physics_steps'],
            'wall_s': result['wall_s'], 'departure': first_departure,
            'no_contact_steps': sum(r['contacts_before_step'] == 0 for r in rows),
            'contact_steps': sum(r['contacts_before_step'] > 0 for r in rows),
            'final_surface_clearance_m': rows[-1]['surface_clearance_m'],
            'floor_reached': any(r['floor_reached'] for r in rows),
            'peak_rss_gib': max(r['peak_rss_bytes'] for r in peaks)/1024**3,
            'peak_commit_gib': max(r['peak_private_commit_bytes'] for r in peaks)/1024**3,
            'max_sampled_gpu_gib': max(r['gpu_used_bytes'] for r in resources)/1024**3,
            **{k: worker[k] for k in ('replay_position_error_m', 'replay_quaternion_component_error', 'replay_velocity_error', 'reset_max_error')}}
        cases.append(case)
        if len(aligned) < 2:
            if len(rows) != 4*hz:
                raise ValueError('full 4s trace required')
            idx = np.arange(6, 241)*(hz//60)-1
            aligned.append([np.array([row['body_q'][bid][:3] for row in rows])[idx],
                np.array([row['body_qd_com_world'][bid][:3] for row in rows])[idx],
                np.array([row['body_qd_com_world'][bid][3:] for row in rows])[idx]])
    criteria = plan['comparison']
    keys = ['max_position_difference_m', 'max_linear_velocity_difference_m_s', 'max_angular_velocity_difference_rad_s']
    errors = [float(np.linalg.norm(a-b, axis=1).max()) for a, b in zip(*aligned)]
    comparison = {**dict(zip(keys, errors)), 'criteria': criteria,
        'passed': all(e <= criteria[k] for e, k in zip(errors, keys)), 'aligned_samples': 235, 'interpolation': False}
    air, edge = cases[2:]
    air_passed = air['departure'] is None and air['no_contact_steps'] > 0 and abs(air['final_surface_clearance_m']) <= .001
    edge_passed = edge['departure'] is not None and edge['contact_steps'] > 0 and not edge['floor_reached']
    reports = {'cases': cases, 'comparison': comparison, 'airborne_return_passed': air_passed,
        'edge_departure_passed': edge_passed,
        'new_run_count': 3, 'new_physics_steps': sum(c['physics_steps'] for c in cases[1:]),
        'new_wall_s': sum(c['wall_s'] for c in cases[1:]),
        'all_checks_passed': comparison['passed'] and air_passed and edge_passed,
        's1_complete': False, 's2_started': False, 'support_region_validated': False,
        'limitations': ['one zero-tilt path only', 'whole-tool projection is a conservative departure envelope, not carrying-face certification',
            'monitor labels not yet integrated with environment episode termination and snapshot/reset', 'no task success rate or training samples'],
        'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'src/ball_retention/retention.py', ROOT/'src/ball_retention/env.py', ROOT/'diagnostics/resource_probe/worker.py', Path(__file__)]}}
    if reports['new_physics_steps'] > plan['max_total_physics_steps']:
        raise ValueError('round physics budget exceeded')
    (out/'summary.json').write_text(json.dumps(reports, indent=2)+'\n', encoding='utf-8')
    return reports


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
