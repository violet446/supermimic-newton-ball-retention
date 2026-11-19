"""Bind training-timing workers to accepted static commands and source histories."""
import json
from pathlib import Path

from .archive_restore_plan import _verified_bytes
from .deterministic_archive_import_plan import load as load_archive


DESIGN = 'configs/s3_training_timing_capture_plan.json'
DESIGN_SHA256 = '1b523dbe4de150a8df2595b0e97339d25523f326c9df21c839c8f5acac771fbf'
PREFLIGHT = 'runs/s3_training_timing_capture_preflight_20261115/preflight.json'
PREFLIGHT_SHA256 = '46a34b2bd3d3df22bf9bceab8205c0fae5a3bbcfe2330551478d89b0a482aaac'
MODE = 's3-training-timing-capture'
MARKER = 'training_timing_capture'


def load(root):
    root = Path(root).resolve()
    plan = json.loads(_verified_bytes(root, DESIGN, DESIGN_SHA256))
    inputs = {name: _verified_bytes(root, name, value) for name, value in plan['inputs_sha256'].items()}
    report = json.loads(_verified_bytes(root, PREFLIGHT, PREFLIGHT_SHA256))
    reference = json.loads(inputs[plan['reference_design']])
    plan = {**plan, **{key: reference[key] for key in ('branch_order', 'directions')}}
    limits = plan['budget']
    steps = limits['prefix_substeps_per_case']+len(plan['branch_order'])*plan['horizon']*plan['control_substeps']
    if (report['passed'] is not True or report['design_sha256'] != DESIGN_SHA256
            or report['branch_order'] != plan['branch_order'] or report['planned_budget'] != limits
            or (report['worker_starts'], report['physical_substeps'], report['training_updates']) != (0, 0, 0)
            or report['test_episode_files_read'] is not False or len(report['cases']) != len(plan['cases'])
            or limits['max_worker_starts'] != 2 or limits['sequential_environments'] != 1
            or steps != limits['max_physics_substeps_per_worker'] or 2*steps != limits['max_physics_substeps']):
        raise ValueError('timing capture preflight or reservation differs from the frozen plan')
    archive, _, _ = load_archive(root)
    layout_path = archive['native_reference']['layout']
    layout = json.loads(_verified_bytes(root, layout_path, archive['inputs_sha256'][layout_path]))

    cases = {}
    for identity, entry in zip(plan['cases'], report['cases'], strict=True):
        name = identity['episode_id']
        evidence_name = entry['commands_artifact']
        evidence = json.loads(_verified_bytes(root, Path(PREFLIGHT).with_name(evidence_name),
                                               report['artifacts_sha256'][evidence_name]))
        case = evidence['case']
        if (not all(case[key] == value == entry[key] for key, value in identity.items())
                or evidence['snapshot_tick'] != plan['snapshot_tick'] or list(evidence['branches']) != plan['branch_order']
                or case['source_config_sha256'] != plan['inputs_sha256'][case['source_config']]):
            raise ValueError('timing capture command evidence differs from the frozen case')
        cfg = json.loads(inputs[case['source_config']])
        cfg.pop('history_capture')
        cfg[MARKER] = {'episode_id': name, 'design_sha256': DESIGN_SHA256, 'preflight_sha256': PREFLIGHT_SHA256}
        cfg['budget'].update(
            wall_s=limits['max_worker_wall_s'], max_cases=1, max_physics_steps=steps, max_trace_steps=steps,
            max_process_rss_gib=limits['max_process_rss_gib'], max_process_commit_gib=limits['max_process_commit_gib'],
            max_gpu_used_gib=limits['max_total_gpu_used_gib'], min_start_ram_gib=limits['min_start_ram_gib'],
            min_start_gpu_free_gib=limits['min_start_gpu_free_gib'], min_disk_free_gib=limits['min_disk_free_gib'],
            max_new_output_mib=limits['max_worker_output_mib'],
            max_output_cache_gib=limits['worker_output_plus_shared_cache_and_temp_gib'],
        )
        cases[name] = (cfg, evidence)
    return plan, cases, layout


def validate_case(root, cfg, steps):
    plan, cases, layout = load(root)
    entry = cases.get(cfg.get(MARKER, {}).get('episode_id'))
    if (entry is None or cfg != entry[0] or type(steps) is not int
            or steps != plan['budget']['max_physics_substeps_per_worker']):
        raise ValueError('worker config or step cap differs from the frozen training case')
    return plan, entry[1], layout
