"""Bind a small-offset physics worker to the frozen development probe."""
import hashlib
import json
from pathlib import Path


DESIGN = 'configs/s3_small_offset_probe_plan.json'
DESIGN_SHA256 = '464aa0f1babebc1fdfb5bbe7ac9fe9b6037854628401c7ebb99a0b421461278b'
SUPPORT_SHA256 = {
    'assets/manifest.json': '0dd7a790ff229702abba0aa5d2b2d449d4fd24021c8dfa19b7aaf315901009f3',
    'runs/s1_reachability_20261111/joint_paths.npz':
        'c1dafea9b961ba0562652f7b7573d1ba5574965549a2e56a10a17cb0902d70d7',
}


def _verified_bytes(root, path, expected):
    root = Path(root).resolve()
    target = (root/path).resolve()
    target.relative_to(root)
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'small-offset input changed: {path}')
    return raw


def verify_design(root):
    """Verify the immutable plan, captured references and mechanical inputs."""
    design = json.loads(_verified_bytes(root, DESIGN, DESIGN_SHA256))
    for path, expected in design['inputs_sha256'].items():
        _verified_bytes(root, path, expected)
    for path, expected in SUPPORT_SHA256.items():
        _verified_bytes(root, path, expected)
    manifest = json.loads((Path(root)/'assets/manifest.json').read_bytes())
    for asset in manifest['assets']:
        _verified_bytes(root, asset['local_path'], asset['sha256'])
    for case in design['cases']:
        _verified_bytes(root, case['source_config'], case['source_config_sha256'])
        _verified_bytes(root, case['source_trace'], case['source_trace_sha256'])
    return design


def archived_prefix_records(root, case):
    """Read earlier solver restore boundaries as well as the target snapshot."""
    trace = _verified_bytes(root, case['source_trace'], case['source_trace_sha256'])
    rows = [json.loads(line) for line in trace.splitlines()]
    ticks = [row['snapshot_tick'] for row in rows]
    if ticks != sorted(set(ticks)) or case['tick'] not in ticks:
        raise ValueError(f'archived snapshot identity mismatch: {case["episode"]}')
    return {row['snapshot_tick']: row for row in rows if row['snapshot_tick'] <= case['tick']}


def archived_record(root, case):
    return archived_prefix_records(root, case)[case['tick']]


def build_case(root, design, case):
    """Derive a worker config without changing the source simulation or schedule."""
    cfg = json.loads(_verified_bytes(root, case['source_config'], case['source_config_sha256']))
    spec = cfg['action_pairs']
    if (cfg['seed'] != case['seed'] or spec['master_seed'] != case['source_master_seed']
            or spec['horizon'] != design['horizon'] or case['tick'] not in spec['snapshot_ticks']
            or cfg['prescribed']['joint_path'] not in SUPPORT_SHA256 or not spec['replay_as_noise']):
        raise ValueError(f'archived config identity mismatch: {case["episode"]}')
    spec.update(snapshot_ticks=[case['tick']], amplitudes=design['amplitudes'][:])
    spec.pop('single_scale_ticks', None)
    limits, steps = design['budget'], case['reserved_substeps']
    cfg['budget'].update(
        wall_s=limits['max_worker_wall_s'], max_cases=1, max_physics_steps=steps,
        max_trace_steps=steps, min_start_ram_gib=limits['min_start_ram_gib'],
        min_start_gpu_free_gib=limits['min_start_gpu_free_gib'], min_disk_free_gib=limits['min_disk_free_gib'],
        max_process_rss_gib=limits['max_process_rss_gib'], max_process_commit_gib=limits['max_process_commit_gib'],
        max_gpu_used_gib=limits['max_total_gpu_used_gib'],
        max_output_cache_gib=limits['worker_output_plus_shared_cache_and_temp_gib'],
    )
    cfg['small_offset_probe'] = {'design_sha256': DESIGN_SHA256, 'episode_id': case['episode']}
    return cfg


def validate_small_offset_case(root, cfg):
    design = verify_design(root)
    name = cfg['small_offset_probe'].get('episode_id')
    case = next((entry for entry in design['cases'] if entry['episode'] == name), None)
    if case is None or cfg != build_case(root, design, case):
        raise ValueError(f'small-offset worker config differs from frozen case: {name}')
    return case
