"""Bind the deterministic archive import to the saved native run and old case."""
from copy import deepcopy
import json
from pathlib import Path

from .archive_restore_plan import _verified_bytes
from .native_repeatability_plan import MARKER as NATIVE_MARKER


DESIGN = 'configs/s3_deterministic_archive_import_plan.json'
DESIGN_SHA256 = '818022cda3948577c3c0869b5c99277cea90d1ba4bd15ea3d137d5b3062027c1'
MODE = 's3-deterministic-archive-import'
MARKER = 'deterministic_archive_import'
NATIVE_RUN = 'runs/s3_native_repeatability_20261114_retry'


def load(root):
    root = Path(root).resolve()
    plan = json.loads(_verified_bytes(root, DESIGN, DESIGN_SHA256))
    hashes = plan['inputs_sha256']
    inputs = {name: _verified_bytes(root, name, digest) for name, digest in hashes.items()}
    for asset in json.loads(inputs['assets/manifest.json'])['assets']:
        _verified_bytes(root, asset['local_path'], asset['sha256'])

    case = plan['case']
    native_plan = json.loads(inputs['configs/s3_native_repeatability_plan.json'])
    prior_plan = json.loads(inputs['configs/s3_archive_restore_pilot_plan.json'])
    if native_plan['case'] != case or prior_plan['cases'] != [case]:
        raise ValueError('native and historical data do not identify the same frozen case')
    for key in ('archive_config', 'archive_trace'):
        if hashes[case[key]] != case[f'{key}_sha256']:
            raise ValueError(f'inconsistent frozen digest for {key}')

    native = plan['native_reference']
    validation = json.loads(inputs[f'{NATIVE_RUN}/validation.json'])
    references = [native['snapshot'], native['layout'], *native['replays']]
    if (validation['status'] != 'completed' or validation['passed'] is not True
            or validation['physics_steps'] != native_plan['budget']['max_physics_substeps']
            or len(native['replays']) != 2 or hashes[native['replays'][0]] != hashes[native['replays'][1]]
            or any(validation['artifacts_sha256'][name.removeprefix(f'{NATIVE_RUN}/')] != hashes[name]
                   for name in [*references, f'{NATIVE_RUN}/config.json'])):
        raise ValueError('saved native inputs do not match the previously accepted native run')
    old = json.loads(inputs[case['archive_config']])
    cfg = json.loads(inputs[f'{NATIVE_RUN}/config.json'])
    expected_marker = {'design_sha256': hashes['configs/s3_native_repeatability_plan.json'],
                       'episode_id': case['episode'], 'tick': case['tick']}
    if (cfg.pop(NATIVE_MARKER) != expected_marker or cfg['seed'] != case['seed']
            or {k: v for k, v in cfg.items() if k != 'budget'} != {k: v for k, v in old.items() if k != 'budget'}
            or cfg['prescribed']['joint_path'] not in inputs):
        raise ValueError('stored native case configuration differs from the frozen scenario')

    rows = [json.loads(line) for line in inputs[case['archive_trace']].splitlines()]
    spec = cfg['action_pairs']
    dt = cfg['physics']['dt_s']
    substeps = round(1/(dt*cfg['physics']['control_hz']))
    horizon = spec['horizon']
    limits = plan['budget']
    counts = [limits[key] for key in ('native_reference_replays', 'same_configuration_import_replays',
                                       'historical_import_replays')]
    steps = sum(counts)*horizon*substeps
    if (spec['scenario'] != 'route' or [row['snapshot_tick'] for row in rows] != spec['snapshot_ticks']
            or spec['snapshot_ticks'] != [72, case['tick']] or any(
                row['physics_dt_s'] != dt or row['control_substeps'] != substeps
                or len(row['nominal_actions']) != horizon for row in rows)
            or (horizon, substeps, counts, steps) != (32, 16, [2, 2, 2], 3072)
            or steps != limits['max_physics_substeps'] or limits['nominal_prefix_substeps'] != 0
            or limits['sequential_environments'] != 1 or limits['max_worker_starts'] != 1):
        raise ValueError('nominal replay clock or substep bound differs from the import plan')

    # Keep the native admission floors and time limits. Only this round's substep reservation changes.
    expected_budget = deepcopy(cfg['budget'])
    expected_budget.update(
        wall_s=limits['max_worker_wall_s'], max_cases=1, max_physics_steps=steps, max_trace_steps=steps,
        min_start_ram_gib=limits['min_start_ram_gib'], min_start_gpu_free_gib=limits['min_start_gpu_free_gib'],
        min_disk_free_gib=limits['min_disk_free_gib'], max_process_rss_gib=limits['max_process_rss_gib'],
        max_process_commit_gib=limits['max_process_commit_gib'], max_gpu_used_gib=limits['max_total_gpu_used_gib'],
        max_new_output_mib=limits['max_new_output_mib'],
        max_output_cache_gib=limits['worker_output_plus_shared_cache_and_temp_gib'],
    )
    cfg['budget'] = expected_budget
    cfg[MARKER] = {**expected_marker, 'design_sha256': DESIGN_SHA256}
    return plan, cfg, {row['snapshot_tick']: row for row in rows}


def validate_case(root, cfg, steps):
    plan, expected, rows = load(root)
    if cfg != expected or type(steps) is not int or steps != plan['budget']['max_physics_substeps']:
        raise ValueError('deterministic archive worker requires the exact frozen config and substep cap')
    return plan, rows
