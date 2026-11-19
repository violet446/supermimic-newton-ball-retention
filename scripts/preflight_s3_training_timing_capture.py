"""Freeze and check two training-side timing captures without Newton stepping."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import os
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

from check_s3_actions import ROOT, digest, schedule_actions, write_json
from check_s3_deterministic_archive_import import backend_preflight
from preflight_s3_deterministic_action_response import action_sequences, verify_commands
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.controller import OnlineTiltController
from ball_retention.geometry import load_tool, sphere_clearance, support_point
from ball_retention.guard import process_memory
from ball_retention.history_capture import validate_history_case
from ball_retention.pair_collection import _initial_key
from ball_retention.prescribed import PrescribedPath
from ball_retention.resources import directory_bytes
from ball_retention.robot import Robot


DESIGN = 'configs/s3_training_timing_capture_plan.json'
DESIGN_SHA256 = '1b523dbe4de150a8df2595b0e97339d25523f326c9df21c839c8f5acac771fbf'
OUT = ROOT/'runs/s3_training_timing_capture_preflight_20261115'


def sequence_digest(sequence):
    return sha256(json.dumps(sequence, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def load_cases(root, plan):
    hashes = plan['inputs_sha256']
    inputs = {path: _verified_bytes(root, path, expected) for path, expected in hashes.items()}
    for asset in json.loads(inputs['assets/manifest.json'])['assets']:
        _verified_bytes(root, asset['local_path'], asset['sha256'])
    reference = json.loads(inputs[plan['reference_design']])
    diagnosis = json.loads(inputs[plan['diagnosis_validation']])
    holdout = plan['development_holdout']
    if (plan['runtime'] != reference['runtime'] or reference['case']['group'] != holdout['group']
            or diagnosis['case'] != reference['case'] or diagnosis['status'] != 'completed'
            or diagnosis['test_episode_files_read'] is not False):
        raise ValueError('response reference or development holdout changed')
    if any(plan[key] != reference[key] for key in ('horizon', 'control_substeps', 'amplitude')):
        raise ValueError('training timing sequence clock or amplitude differs from the reference')
    design = {**plan, **{key: reference[key] for key in ('branch_order', 'directions')}}

    source = Path(plan['source_run'])
    read = lambda name: json.loads(inputs[(source/name).as_posix()])
    capture, validation = read('capture_manifest.json'), read('capture_validation.json')
    entries = {row['episode_id']: row for row in capture['entries']}
    limits = plan['budget']
    controls = plan['control_substeps']
    prefix = plan['snapshot_tick']*controls
    replays = len(design['branch_order'])*plan['horizon']*controls
    if (len(entries) != len(capture['entries']) or capture['control_substeps'] != controls
            or capture['horizon'] != plan['horizon'] or capture['amplitude'] != plan['amplitude']
            or validation['status'] != 'completed_and_evaluated'
            or validation['source_config_and_data_hashes_verified'] is not True
            or validation['artifacts_sha256']['capture_manifest.json'] != hashes[(source/'capture_manifest.json').as_posix()]
            or prefix != limits['prefix_substeps_per_case'] or prefix+replays != limits['max_physics_substeps_per_worker']
            or len(plan['cases'])*(prefix+replays) != limits['max_physics_substeps']
            or len(plan['cases']) != limits['max_worker_starts'] or limits['sequential_environments'] != 1
            or len(design['branch_order']) != limits['branches_per_case'] or limits['nominal_replays_per_case'] != 2):
        raise ValueError('capture clock, source acceptance or reservation differs from the frozen scope')

    cases, identities = [], {_initial_key(json.loads(inputs[holdout['config']]))}
    train_groups = validation['merged_collection']['train']['group_weight_mass']
    validation_groups = validation['merged_collection']['validation']['group_weight_mass']
    if holdout['group'] not in validation_groups or holdout['group'] in train_groups:
        raise ValueError('development holdout split changed')
    for case in plan['cases']:
        entry = entries[case['episode_id']]
        config_path = (source/entry['config_path']).as_posix()
        trace_path = (source/entry['path']).as_posix()
        if (any(entry[key] != value for key, value in case.items()) or case['split'] != 'train'
                or case['group_id'] not in train_groups or case['group_id'] in validation_groups
                or hashes[config_path] != entry['config_sha256'] or hashes[trace_path] != entry['sha256']
                or validation['artifacts_sha256'][entry['config_path']] != entry['config_sha256']):
            raise ValueError('training provenance or group split changed')
        cfg = json.loads(inputs[config_path])
        if validate_history_case(root, cfg) != case or cfg['prescribed']['joint_path'] not in inputs:
            raise ValueError('training case is inconsistent with the original frozen history')
        physical_key = _initial_key(cfg)
        if physical_key in identities:
            raise ValueError('training cases must be physically distinct from each other and the development holdout')
        identities.add(physical_key)
        rows = [json.loads(line) for line in inputs[trace_path].splitlines()]
        if ([row['snapshot_tick'] for row in rows] != entry['snapshot_ticks']
                or rows[-1]['snapshot_tick'] != plan['snapshot_tick']
                or any(row['control_substeps'] != controls or row['physics_dt_s'] != cfg['physics']['dt_s']
                       for row in rows)):
            raise ValueError('original training snapshot or clock changed')
        cases.append(({**case, 'source_config': config_path, 'source_config_sha256': entry['config_sha256'],
                       'source_trace': trace_path, 'source_trace_sha256': entry['sha256'],
                       'physical_initial_sha256': physical_key}, cfg, rows, entry['initial_clearance_m']))
    return design, cases


def verify_case(root, design, case, cfg, rows, recorded_clearance, guard):
    robot = Robot(root/'assets/ec66_description.urdf')
    mesh = load_tool(root/'assets/shovel.STL')
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    tool, _, _ = robot.fk(cfg['initial']['joint_q_rad'])
    center = np.asarray(cfg['initial']['ball_center_world_m'])
    radius = cfg['ball']['radius_m']
    local = tool[:3, :3].T @ (center-tool[:3, 3])
    gap = sphere_clearance(mesh, local, radius)
    if (not np.isclose(gap, recorded_clearance, rtol=0., atol=1e-12) or gap < .0001
            or local[0] < .22 or center[2]-radius <= cfg['physics']['floor_z_m']):
        raise ValueError(f'restored training initial geometry differs from recorded case {case["episode_id"]}')
    path = PrescribedPath(root/cfg['prescribed']['joint_path'], cfg, moving=True)
    controller = OnlineTiltController(robot, cfg, anchor, path, root/'assets')
    tick, substeps, horizon = (design[key] for key in ('snapshot_tick', 'control_substeps', 'horizon'))
    dt = cfg['physics']['dt_s']
    if not np.isclose(dt*substeps*cfg['physics']['control_hz'], 1., rtol=0., atol=1e-12):
        raise ValueError('training clock is off the frozen substep ratio')
    timeline = schedule_actions(cfg['action_pairs']['schedule'], 0, tick+horizon)
    archives = {row['snapshot_tick']: row for row in rows}
    commands = []
    for index, action in enumerate(timeline):
        step = controller.plan(index*substeps*dt, action, substeps)
        for command in step:
            controller.commit(command)
            np.testing.assert_array_equal(command.clipped_action, action)
        commands.append({'q': [item.q.tolist() for item in step], 'qd': [item.qd.tolist() for item in step]})
        lead = index+1
        if lead in archives:
            row = archives[lead]
            expected = row['initial_state']
            np.testing.assert_array_equal(row['anchor_local_m'], anchor)
            if row['nominal_actions'] != timeline[lead:lead+horizon] or expected['time_s'] != lead*substeps*dt:
                raise ValueError('archived nominal action history or lead clock changed')
            for name, value in (('joint_q', command.q), ('joint_qd', command.qd), ('tilt_rad', command.tilt),
                                ('tilt_rate_rad_s', command.tilt_rate), ('previous_action', command.clipped_action)):
                np.testing.assert_array_equal(value, expected[name])
        if lead == tick:
            checkpoint = {'controller': controller.snapshot(), 'physics': {'steps': tick*substeps}}
            last_qdd = command.qdd.tolist()
        if lead % 24 == 0:
            guard()

    nominal = timeline[tick:]
    directions, requests = action_sequences(design, nominal)
    all_commands, limits = verify_commands(root, design, cfg, checkpoint, requests, commands[tick:], guard)
    release = 184-tick
    for name, direction in directions.items():
        if np.any(direction[release:]) or np.any(requests[f'{name}_plus'][release:]) or np.any(requests[f'{name}_minus'][release:]):
            raise ValueError('signed sequence must preserve the nominal release and the entire zero-action tail')
    payload = {'case': case, 'snapshot_tick': tick, 'snapshot_control_only': checkpoint['controller'],
               'snapshot_joint_qdd_rad_s2': last_qdd, 'prefix_actions_sha256': sequence_digest(timeline[:tick]),
               'prefix_requested': timeline[:tick], 'prefix_executed': commands[:tick],
               'directions_Hx2': {name: values.tolist() for name, values in directions.items()},
               'branches': {label: {'requested': requests[label].tolist(), 'executed': all_commands[label]}
                            for label in design['branch_order']}}
    return payload, {'initial_clearance_m': float(gap), 'source_controller_ticks_verified': sorted(archives),
                     'branch_peaks': limits, 'zero_tail_ticks': [184, tick+horizon],
                     'shared_first_eight_commands_exact': True}


def run(out):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    try:
        plan = json.loads(_verified_bytes(ROOT, DESIGN, DESIGN_SHA256))
        limits = plan['static_budget']

        def guard():
            memory = process_memory(os.getpid())
            if (time.monotonic()-began >= limits['max_wall_s']
                    or memory['peak_rss_bytes'] > limits['max_process_rss_gib']*1024**3
                    or memory['peak_private_commit_bytes'] > limits['max_process_commit_gib']*1024**3
                    or directory_bytes(out) > limits['max_new_output_mib']*1024**2):
                raise RuntimeError('training timing capture static resource cap reached')
            return memory

        guard()
        backend = backend_preflight(plan)
        if int(backend['device']['native_sm']) != plan['runtime']['cuda_arch']:
            raise ValueError('GPU architecture differs from the frozen deterministic backend')
        design, cases = load_cases(ROOT, plan)
        reports = []
        for case, cfg, rows, clearance in cases:
            payload, summary = verify_case(ROOT, design, case, cfg, rows, clearance, guard)
            target = out/f'{case["episode_id"]}_commands.json'
            write_json(target, payload)
            reports.append({**case, **summary, 'commands_artifact': target.name})
            guard()
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('static capture preflight imported physics')
        source_paths = {Path(__file__).resolve(), ROOT/DESIGN,
                        *(ROOT/'src/ball_retention').glob('*.py'),
                        *(ROOT/'diagnostics/resource_probe').glob('*.py'),
                        *(ROOT/'scripts').glob('*s3_deterministic*.py'), ROOT/'scripts/check_s3_actions.py'}
        memory = guard()
        write_json(out/'preflight.json', {'passed': True, 'design_sha256': DESIGN_SHA256,
                   'backend': backend, 'cases': reports, 'case_count': len(reports),
                   'branch_order': design['branch_order'], 'planned_budget': plan['budget'],
                   'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(source_paths)},
                   'artifacts_sha256': {p.name: digest(p) for p in sorted(out.iterdir())},
                   'worker_starts': 0, 'physical_substeps': 0, 'training_updates': 0,
                   'test_episode_files_read': False, 'control_accepted': False,
                   'wall_s': time.monotonic()-began, **memory})
    except BaseException as error:
        write_json(out/'stopped.json', {'reason': str(error), 'wall_s': time.monotonic()-began,
                   'worker_starts': 0, 'physical_substeps': 0, 'training_updates': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    output = parser.parse_args().output.resolve()
    if not output.relative_to(ROOT/'runs').parts:
        raise ValueError('choose a fresh run subdirectory')
    run(output)
