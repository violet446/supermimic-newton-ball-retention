"""Replay frozen development branches at their individual action amplitudes."""
from itertools import combinations
import json
import os
import time

import numpy as np

from ball_retention.guard import process_memory
from ball_retention.resources import directory_bytes
from archive_restore import write
from archive_restore_metrics import _half_spreads
from deterministic_action_response import _prefix_gate
from deterministic_archive_metrics import _full_branch, native_reference_metrics
from native_repeatability import _make_env, _record_nominal
from saved_native_snapshot import decode_snapshot, restore_and_readback, verify_live_layout


def nominal_gate(new, archived, summary_leads):
    metrics = native_reference_metrics(new, archived)
    branches = [{'raw': np.asarray(row['states_raw'])} for row in (*new, *archived)]
    leads = range(1, len(branches[0]['raw']) + 1)
    spread = _half_spreads(branches, list(combinations(range(len(branches)), 2)), leads)
    valid = all(spread[str(lead)]['position_m'] <= metrics['position_resolution_m'] for lead in summary_leads)
    return {**metrics, 'passed': metrics['numerical_equivalent'] and valid,
            'all_branch_max_half_spread_by_lead': spread, 'position_resolution_passed': valid,
            'position_resolution_gate_leads': summary_leads,
            'all_nominal_json_equal': all(trace == archived[0] for trace in (*new, *archived))}


def run(root, cfg, spec, native, commands, recorded, out, summary, stage):
    start = time.monotonic()
    limits = spec['future_capture_limits']
    cap = limits['max_physics_substeps']
    tick, substeps, horizon = spec['case']['tick'], spec['substeps_per_control'], spec['horizon']
    step_dt = cfg['physics']['dt_s']
    saved = decode_snapshot(native['snapshot'], native['layout'], cfg, tick, substeps)

    def guard():
        memory = process_memory(os.getpid())
        if (time.monotonic()-start >= limits['max_worker_wall_s']
                or memory['peak_rss_bytes'] > limits['max_process_rss_gib']*1024**3
                or memory['peak_private_commit_bytes'] > limits['max_process_commit_gib']*1024**3
                or directory_bytes(out) > limits['max_new_output_mib']*1024**2):
            raise RuntimeError('small-action response worker budget reached')

    guard()
    stage('frozen_development_evidence_verified')
    env = _make_env(root, cfg, summary, stage, expected_device_arch=spec['runtime']['cuda_arch'])
    env.reset(episode_id='s3-small-action-response')
    if env.substeps != substeps or env.dt != step_dt or env.physics.steps != 0 or env.step_index != 0:
        raise ValueError('scene did not initialize at the frozen zero-prefix clock')
    summary['nominal_prefix_substeps'] = 0
    write(out, 'native_layout.json', verify_live_layout(env.physics, native['layout']))
    stage('live_layout_verified')
    prefixes = {b: (a, n) for a, b, n in spec['same_sign_prefixes']}
    traces = {}
    for label in spec['branch_order']:
        expected = commands[label]['executed']
        write(out, f'{label}_readback.json', restore_and_readback(env, saved, substeps))
        guard()
        stage(f'{label}_start')

        def advance(action):
            if summary['physics_steps'] + substeps > cap or time.monotonic()-start >= limits['max_worker_wall_s']:
                raise RuntimeError('small-action response substep or worker wall cap')
            before = env.physics.steps
            previous = env.state
            try:
                row = env.step(action)
            finally:
                summary['physics_steps'] += env.physics.steps-before
            lead = env.step_index-tick
            state = row.next_state_before_reset
            if (not 1 <= lead <= horizon or row.episode_id != env.episode_id or row.step_index != env.step_index-1
                    or row.state_t != previous or row.state_t.time_s != before*step_dt
                    or env.physics.steps != env.step_index*substeps or env.physics.steps-before != substeps
                    or state.time_s != env.physics.steps*step_dt or row.invalid_reason is not None
                    or row.terminated or row.truncated or row.requested_action != action or state.previous_action != action
                    or {'q': row.executed_joint_q, 'qd': row.executed_joint_qd} != expected[lead-1]):
                raise RuntimeError(f'{label} transition disagrees with the frozen command or control clock')
            return row

        trace = _record_nominal(env, saved, commands[label]['requested'], advance, substeps, substeps*step_dt)
        if len(trace['contacts']) != horizon*substeps or any(
                row['physics_step'] != tick*substeps+i for i, row in enumerate(trace['contacts'], 1)):
            raise ValueError(f'{label} does not have one ordered contact record per physical substep')
        _full_branch(trace, tick)
        filename = f'{label}.json'
        write(out, filename, trace)
        trace = json.loads((out/filename).read_bytes())
        traces[label] = trace
        summary['completed_branches'].append(label)
        guard()
        stage(f'{label}_complete')

        if label == 'nominal_2':
            result = nominal_gate([traces['nominal_1'], trace], recorded, spec['scoring']['summary_leads'])
            write(out, 'nominal_comparison.json', result)
            if not result['passed']:
                raise RuntimeError('live nominal replay failed its frozen gate')
            summary['nominal_gate_passed'] = True
            stage('nominal_gate_passed')
        if label in prefixes:
            first, leads = prefixes[label]
            result = _prefix_gate(traces[first], trace, tick, leads, substeps)
            write(out, f'{label}_prefix.json', result)
            if not (result['passed'] and result['contacts_match']):
                raise RuntimeError(f'{label} diverged before the H9 reversal')
            summary['accepted_prefixes'].append(label)
            stage(f'{label}_prefix_passed')
    if (summary['physics_steps'] != cap or set(summary['accepted_prefixes']) != set(prefixes)
            or not summary.get('nominal_gate_passed')):
        raise ValueError('physical scope differs from the frozen replay specification')
    summary['status'] = 'completed'
    stage('physical_capture_completed')
