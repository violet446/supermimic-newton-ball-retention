"""Bind the deterministic native-replay worker to the frozen S3 case."""
import json
from pathlib import Path

from .archive_restore_plan import DESIGN as ARCHIVE_DESIGN
from .archive_restore_plan import DESIGN_SHA256 as ARCHIVE_DESIGN_SHA256
from .archive_restore_plan import _verified_bytes, load as load_archive_case


DESIGN = 'configs/s3_native_repeatability_plan.json'
DESIGN_SHA256 = '235f9e6675c17fb1fd31fc49aa9b2ad745332470615caaca68843bbb77bf98c7'
MODE = 's3-native-repeatability'
MARKER = 'native_repeatability'


def load(root):
    """Verify the plan and derive the exact worker inputs for native replays."""
    root = Path(root).resolve()
    plan = json.loads(_verified_bytes(root, DESIGN, DESIGN_SHA256))
    inputs = plan['inputs_sha256']
    for name, digest in inputs.items():
        _verified_bytes(root, name, digest)
    if inputs[ARCHIVE_DESIGN] != ARCHIVE_DESIGN_SHA256:
        raise ValueError('native repeatability case references another archive design')

    archive_plan, cfg, rows = load_archive_case(root)
    case = plan['case']
    if archive_plan['cases'] != [case] or list(rows) != [72, 168]:
        raise ValueError('native repeatability case differs from the frozen scene or restore boundary')

    limits = plan['budget']
    clock = rows[case['tick']]
    substeps = clock['control_substeps']
    horizon = cfg['action_pairs']['horizon']
    prefix = case['tick']*substeps
    steps = prefix+limits['nominal_replays']*horizon*substeps
    if (limits['sequential_environments'] != 1 or limits['max_worker_starts'] != 1
            or limits['nominal_replays'] != 2 or horizon != 32 or substeps != 16
            or prefix != limits['nominal_prefix_substeps'] or steps != limits['max_physics_substeps']
            or steps != 3712):
        raise ValueError('native repeatability budget differs from the frozen episode')

    cfg['budget'].update(
        wall_s=limits['max_worker_wall_s'], max_cases=1, max_physics_steps=steps, max_trace_steps=steps,
        min_start_ram_gib=limits['min_start_ram_gib'], min_start_gpu_free_gib=limits['min_start_gpu_free_gib'],
        min_disk_free_gib=limits['min_disk_free_gib'], max_process_rss_gib=limits['max_process_rss_gib'],
        max_process_commit_gib=limits['max_process_commit_gib'], max_gpu_used_gib=limits['max_total_gpu_used_gib'],
        max_new_output_mib=limits['max_new_output_mib'],
        max_output_cache_gib=limits['worker_output_plus_shared_cache_and_temp_gib'],
    )
    del cfg['archive_restore_pilot']
    cfg[MARKER] = {'design_sha256': DESIGN_SHA256, 'episode_id': case['episode'], 'tick': case['tick']}
    return plan, cfg, rows


def validate_case(root, cfg, steps):
    """Reject config or step drift before the worker can import physics."""
    plan, expected, rows = load(root)
    if cfg != expected or type(steps) is not int or steps != plan['budget']['max_physics_substeps']:
        raise ValueError('native repeatability worker must match the frozen config and step bound')
    return plan, rows
