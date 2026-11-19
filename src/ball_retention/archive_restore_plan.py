"""Bind a single archive restoration worker to its frozen pilot inputs."""
import hashlib
import json
from pathlib import Path


DESIGN = 'configs/s3_archive_restore_pilot_plan.json'
DESIGN_SHA256 = '64342978c153534038bea439b920c01e8b299c38c31709f763beb2ec558fb738'


def _verified_bytes(root, relative_path, expected_digest):
    path = (root/relative_path).resolve()
    path.relative_to(root)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError(f'archive restoration input changed: {relative_path}')
    return raw


def load(root):
    """Verify pilot inputs and derive the exact worker configuration."""
    root = Path(root).resolve()
    plan = json.loads(_verified_bytes(root, DESIGN, DESIGN_SHA256))
    inputs = {name: _verified_bytes(root, name, digest) for name, digest in plan['inputs_sha256'].items()}
    for asset in json.loads(inputs['assets/manifest.json'])['assets']:
        _verified_bytes(root, asset['local_path'], asset['sha256'])

    case, = plan['cases']
    cfg = json.loads(_verified_bytes(root, case['archive_config'], case['archive_config_sha256']))
    trace = _verified_bytes(root, case['archive_trace'], case['archive_trace_sha256'])
    rows = [json.loads(line) for line in trace.splitlines()]
    spec = cfg['action_pairs']
    if (cfg['seed'] != case['seed'] or spec['scenario'] != 'route'
            or cfg['prescribed']['joint_path'] not in inputs
            or [row['snapshot_tick'] for row in rows] != spec['snapshot_ticks']
            or spec['snapshot_ticks'][-1] != case['tick']):
        raise ValueError('archive restoration case disagrees with the frozen source')

    dt = cfg['physics']['dt_s']
    substeps = round(1./(dt*cfg['physics']['control_hz']))
    limits = plan['budget']
    branches = sum(limits[key] for key in ('live_native_branches', 'live_reconstructed_branches',
                                          'historical_reconstructed_branches'))
    prefix_steps = case['tick']*substeps
    steps = prefix_steps+branches*spec['horizon']*substeps
    if (steps != limits['max_physics_substeps'] or prefix_steps != limits['nominal_prefix_substeps']
            or limits['sequential_environments'] != 1 or limits['max_worker_starts'] != 1
            or any(row['physics_dt_s'] != dt or row['control_substeps'] != substeps
                   or len(row['nominal_actions']) != spec['horizon'] for row in rows)):
        raise ValueError('archive restoration clock or substep bound disagrees with the frozen plan')

    cfg['budget'].update(
        wall_s=limits['max_worker_wall_s'], max_cases=1, max_physics_steps=steps, max_trace_steps=steps,
        min_start_ram_gib=limits['min_start_ram_gib'], min_start_gpu_free_gib=limits['min_start_gpu_free_gib'],
        min_disk_free_gib=limits['min_disk_free_gib'], max_process_rss_gib=limits['max_process_rss_gib'],
        max_process_commit_gib=limits['max_process_commit_gib'], max_gpu_used_gib=limits['max_total_gpu_used_gib'],
        max_new_output_mib=limits['max_new_output_mib'],
        max_output_cache_gib=limits['worker_output_plus_shared_cache_and_temp_gib'],
    )
    cfg['archive_restore_pilot'] = {'design_sha256': DESIGN_SHA256, 'episode_id': case['episode'], 'tick': case['tick']}
    return plan, cfg, {row['snapshot_tick']: row for row in rows}


def validate_case(root, cfg, steps):
    """Require the exact derived config and total substep reservation."""
    plan, expected_cfg, rows = load(root)
    if cfg != expected_cfg or steps != plan['budget']['max_physics_substeps']:
        raise ValueError('archive restoration worker must match the frozen config and substep bound')
    return plan, rows
