"""Bind the one-worker response pilot to its frozen inputs and preflight artifacts."""
import json
from pathlib import Path

from .archive_restore_plan import _verified_bytes
from .deterministic_archive_import_plan import MARKER as IMPORT_MARKER, load as load_archive


DESIGN = 'configs/s3_deterministic_action_response_plan.json'
DESIGN_SHA256 = 'f1ea393adce1fadf3f80a749634d1eae162031537fe94bdbd33d8179bbd39a66'
PREFLIGHT = 'runs/s3_deterministic_action_response_preflight_20261115/preflight.json'
PREFLIGHT_SHA256 = '43ffcaf5cce8aaeaffd31977ebfd0452090c01e4a0a7112ed7a49619fdfb97b7'
MODE = 's3-deterministic-action-response'
MARKER = 'deterministic_action_response'


def load(root):
    root = Path(root).resolve()
    plan = json.loads(_verified_bytes(root, DESIGN, DESIGN_SHA256))
    inputs = {name: _verified_bytes(root, name, expected) for name, expected in plan['inputs_sha256'].items()}
    archive, cfg, _ = load_archive(root)
    preflight = json.loads(_verified_bytes(root, PREFLIGHT, PREFLIGHT_SHA256))
    limits = plan['budget']
    case = plan['case']
    labels = plan['branch_order']
    branches = 2+2*len(plan['directions'])
    steps = branches*plan['horizon']*plan['control_substeps']
    if (preflight['passed'] is not True or preflight['design_sha256'] != DESIGN_SHA256
            or preflight['case'] != case or preflight['branch_order'] != labels
            or preflight['model_id'] != plan['model']['id']
            or preflight['physical_substeps'] != 0 or preflight['worker_starts'] != 0
            or preflight['test_episode_files_read'] is not False
            or (steps, branches, len(set(labels))) != (4096, 8, 8)
            or steps != limits['max_physics_substeps'] or limits['baseline_replays'] != 2
            or limits['nominal_prefix_substeps'] != 0 or limits['max_worker_starts'] != 1
            or limits['sequential_environments'] != 1 or cfg['seed'] != case['seed']):
        raise ValueError('response preflight or worker budget differs from the frozen plan')

    evidence = {}
    for key in ('branch_commands', 'model_rollouts'):
        name = f'{key}.json'
        evidence[key] = json.loads(_verified_bytes(root, Path(PREFLIGHT).with_name(name),
                                                   preflight['artifacts_sha256'][name]))
    for key in ('snapshot', 'layout'):
        path = archive['native_reference'][key]
        evidence[key] = json.loads(_verified_bytes(root, path, archive['inputs_sha256'][path]))
    evidence['recorded'] = [json.loads(inputs[path]) for path in plan['nominal_replays']]
    cfg.pop(IMPORT_MARKER)
    cfg[MARKER] = {'design_sha256': DESIGN_SHA256, 'preflight_sha256': PREFLIGHT_SHA256,
                   'episode_id': case['episode'], 'tick': case['tick']}
    cfg['budget'].update(
        max_physics_steps=steps, max_trace_steps=steps, wall_s=limits['max_worker_wall_s'],
        max_process_rss_gib=limits['max_process_rss_gib'], max_process_commit_gib=limits['max_process_commit_gib'],
        max_gpu_used_gib=limits['max_total_gpu_used_gib'], min_start_ram_gib=limits['min_start_ram_gib'],
        min_start_gpu_free_gib=limits['min_start_gpu_free_gib'], min_disk_free_gib=limits['min_disk_free_gib'],
        max_new_output_mib=limits['max_new_output_mib'],
        max_output_cache_gib=limits['worker_output_plus_shared_cache_and_temp_gib'],
    )
    return plan, cfg, evidence


def validate_case(root, cfg, steps):
    plan, expected, evidence = load(root)
    if cfg != expected or type(steps) is not int or steps != plan['budget']['max_physics_substeps']:
        raise ValueError('response worker requires the exact frozen case and step cap')
    return plan, evidence
