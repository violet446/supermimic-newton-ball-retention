"""Run the frozen nominal comparison in one deterministic Newton environment."""

from dataclasses import asdict
from hashlib import sha256
import json
import os
import time

import numpy as np

from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import reconstruct
from ball_retention.collection import replay_error
from ball_retention.deterministic_archive_import_plan import validate_case
from ball_retention.guard import process_memory
from ball_retention.resources import directory_bytes

from archive_restore import write
from deterministic_archive_metrics import historical_import_metrics, native_reference_metrics, same_configuration_metrics
from native_repeatability import _make_env, _nominal_schedule, _record_nominal
from saved_native_snapshot import decode_snapshot, restore_and_readback, verify_live_layout


def run(root, cfg, out, max_steps, summary, stage):
    started = time.monotonic()
    plan, archive = validate_case(root, cfg, max_steps)
    limits = plan['budget']
    tick = plan['case']['tick']
    target = archive[tick]
    dt = cfg['physics']['dt_s']
    substeps = target['control_substeps']
    control_dt = substeps*dt

    def guard():
        memory = process_memory(os.getpid())
        if (time.monotonic()-started >= limits['max_worker_wall_s']
                or memory['peak_rss_bytes'] > limits['max_process_rss_gib']*1024**3
                or memory['peak_private_commit_bytes'] > limits['max_process_commit_gib']*1024**3
                or directory_bytes(out) > limits['max_new_output_mib']*1024**2):
            raise RuntimeError('deterministic archive import worker budget reached')

    def read_frozen(key):
        return _verified_bytes(root, key, plan['inputs_sha256'][key])

    guard()
    _, nominal, command, expected_commands = _nominal_schedule(root, cfg, archive, tick, substeps, guard)
    files = plan['native_reference']
    layout = json.loads(read_frozen(files['layout']))
    native_document = json.loads(read_frozen(files['snapshot']))
    native = decode_snapshot(native_document, layout, cfg, tick, substeps, command)
    recorded_bytes = [read_frozen(name) for name in files['replays']]
    recorded = [json.loads(raw) for raw in recorded_bytes]
    for replay in recorded:
        if replay['commands'] != expected_commands or replay['actions'] != nominal:
            raise ValueError('recorded native actions or executed commands differ from the static schedule')
    summary['completed_branches'] = []
    stage('saved_snapshot_and_nominal_controller_verified')

    env = _make_env(root, cfg, summary, stage, expected_device_arch=89)
    env.reset(episode_id='s3-deterministic-archive-import')
    if env.substeps != substeps or env.dt != dt or env.physics.steps != 0 or env.step_index != 0:
        raise ValueError('scene did not initialize at the frozen zero-prefix clock')
    summary['nominal_prefix_substeps'] = 0
    write(out, 'native_layout.json', verify_live_layout(env.physics, layout))
    guard()
    stage('fresh_scene_layout_verified')

    def advance(action):
        if summary['physics_steps']+substeps > max_steps or time.monotonic()-started >= limits['max_worker_wall_s']:
            raise RuntimeError('deterministic archive import substep or worker wall cap')
        before = env.physics.steps
        try:
            row = env.step(action)
        finally:
            summary['physics_steps'] += env.physics.steps-before
        lead = env.step_index-tick
        state = row.next_state_before_reset
        if (not 1 <= lead <= len(nominal) or row.step_index != env.step_index-1
                or env.physics.steps != env.step_index*substeps or env.physics.steps-before != substeps
                or state.time_s != env.physics.steps*dt
                or row.invalid_reason is not None or row.terminated or row.truncated
                or row.requested_action != action or state.previous_action != action
                or {'q': row.executed_joint_q, 'qd': row.executed_joint_qd} != expected_commands[lead-1]):
            raise RuntimeError('nominal transition disagrees with the frozen clock, commands or episode scope')
        return row

    def branch(saved, label):
        readback = restore_and_readback(env, saved, substeps)
        write(out, f'{label}_readback.json', readback)
        guard()
        stage(f'{label}_start')
        trace = _record_nominal(env, saved, nominal, advance, substeps, control_dt)
        for offset, contact in enumerate(trace['contacts'], 1):
            if contact['physics_step'] != tick*substeps+offset:
                raise ValueError('nominal contact log is missing an ordered physical substep')
        filename = f'{label}.json'
        write(out, filename, trace)
        raw = (out/filename).read_bytes()
        summary['completed_branches'].append(label)
        guard()
        stage(f'{label}_complete')
        return raw

    def gate(name, comparison):
        summary[name] = comparison
        write(out, f'{name}.json', comparison)
        guard()
        stage(name)
        if not comparison['passed']:
            raise RuntimeError(f'{name} failed its frozen acceptance gate')

    native_bytes = [branch(native, f'native_replay_{i}') for i in (1, 2)]
    new_native = [json.loads(raw) for raw in native_bytes]
    native_reference = native_reference_metrics(new_native, recorded)
    native_reference.update(
        new_sha256=[sha256(raw).hexdigest() for raw in native_bytes],
        recorded_sha256=[sha256(raw).hexdigest() for raw in recorded_bytes],
        all_native_files_byte_identical=all(raw == recorded_bytes[0] for raw in native_bytes+recorded_bytes),
    )
    gate('native_reference_comparison', native_reference)

    rebuilt, evidence = reconstruct(env, native_document['state'], tick, command)
    if (rebuilt['monitor'] != native['monitor'] or rebuilt['deep_count'] != native['deep_count']
            or replay_error(rebuilt['controller'], native['controller']) > 1e-12):
        raise ValueError('public State import did not preserve the native controller and counters')
    evidence['native_controller_monitor_counters_match'] = True
    evidence['reconstructed_array_max_abs_difference'] = {
        name: float(np.max(np.abs(array-native['physics']['arrays'][name])))
        for name, array in rebuilt['physics']['arrays'].items()
    }
    write(out, 'same_configuration_import.json', evidence)
    stage('same_configuration_import_verified')
    imported = [json.loads(branch(rebuilt, f'same_configuration_replay_{i}')) for i in (1, 2)]
    gate('same_configuration_comparison', same_configuration_metrics(new_native, imported))

    historical, evidence = reconstruct(env, target['initial_state'], tick, command)
    write(out, 'historical_import.json', evidence)
    stage('historical_import_verified')
    old_import = [json.loads(branch(historical, f'historical_replay_{i}')) for i in (1, 2)]
    old_recorded = sorted((row for row in target['branches'] if row['axis'] is None and row['offset'] == 0.),
                          key=lambda row: row['repeat'])
    history = historical_import_metrics(old_import, old_recorded)
    cross = history.pop('cross_configuration')
    summary['historical_cross_configuration'] = cross
    write(out, 'historical_cross_configuration.json', cross)
    gate('historical_repeatability', history)

    if summary['physics_steps'] != max_steps:
        raise ValueError('completed replay count differs from the frozen substep reservation')
    guard()
    summary.update(status='deterministic_archive_import_completed', control_accepted=False)
