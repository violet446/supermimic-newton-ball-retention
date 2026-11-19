"""Rebuild one training prefix and capture the fixed timed-action pairs."""
from dataclasses import asdict
import json
import os
import time

import numpy as np

from ball_retention.guard import process_memory
from ball_retention.pair_data import DEFAULT_RESPONSE_FLOOR
from ball_retention.resources import directory_bytes
from ball_retention.training_timing_capture_plan import validate_case

from archive_restore import write
from archive_restore_metrics import POSITION_RESOLUTION_M
from deterministic_action_response import _prefix_gate
from deterministic_archive_metrics import _full_branch, _live_pair
from native_repeatability import _make_env, _record_nominal
from saved_native_snapshot import restore_and_readback, verify_live_layout


def nominal_gate(first, second, tick):
    a, b = (_full_branch(trace, tick) for trace in (first, second))
    comparison = _live_pair(a, b)
    spreads = np.linalg.norm((a['raw']-b['raw']).reshape(-1, 3, 3)/2., axis=-1)
    leads = (8, 16, 24, 32)
    return {'passed': comparison['numerical_equivalent'] and all(spreads[h-1, 0] <= POSITION_RESOLUTION_M for h in leads),
            'by_lead': {str(h): dict(zip(('position_m', 'velocity_m_s', 'spin_rad_s'), spreads[h-1].tolist())) for h in leads},
            'comparison': comparison, 'contacts_match': first['contacts'] == second['contacts']}


def signal_report(plan, traces):
    first, second = (np.asarray(traces[label]['states_raw']).reshape(-1, 3, 3) for label in ('nominal_1', 'nominal_2'))
    noise = np.linalg.norm((first-second)/2., axis=-1)
    report = {}
    for name in plan['directions']:
        plus, minus = (np.asarray(traces[f'{name}_{sign}']['states_raw']).reshape(-1, 3, 3) for sign in ('plus', 'minus'))
        combined = np.stack((first, second, plus, minus))
        numerical = 32*np.finfo(float).eps*np.maximum(1., np.max(np.linalg.norm(combined, axis=-1), axis=0))
        floor = np.maximum(np.maximum(DEFAULT_RESPONSE_FLOOR, 5.*noise), numerical)
        half = (plus-minus)/2.
        norm = np.linalg.norm(half, axis=-1)
        report[name] = {'half_difference_si': half.reshape(-1, 9).tolist(), 'half_difference_norm_si': norm.tolist(),
                        'nominal_half_spread_si': noise.tolist(), 'signal_floor_si': floor.tolist(),
                        'resolved': (norm >= floor).tolist(), 'strong': (norm >= 10.*floor).tolist()}
    return {'family_order': ['position_m', 'velocity_m_s', 'spin_rad_s'], 'directions': report}


def run(root, cfg, out, max_steps, summary, stage):
    started = time.monotonic()
    plan, evidence, layout = validate_case(root, cfg, max_steps)
    limits = plan['budget']
    tick, substeps = plan['snapshot_tick'], plan['control_substeps']
    dt = cfg['physics']['dt_s']
    case = evidence['case']

    def guard():
        mem = process_memory(os.getpid())
        if (time.monotonic()-started >= limits['max_worker_wall_s']
                or mem['peak_rss_bytes'] > limits['max_process_rss_gib']*1024**3
                or mem['peak_private_commit_bytes'] > limits['max_process_commit_gib']*1024**3
                or directory_bytes(out) > limits['max_worker_output_mib']*1024**2):
            raise RuntimeError('training timing capture worker resource limit reached')

    guard()
    summary.update(case=case, completed_branches=[], prefix_substeps=0)
    env = _make_env(root, cfg, summary, stage, expected_device_arch=plan['runtime']['cuda_arch'])
    env.reset(episode_id=case['episode_id'])
    if env.substeps != substeps or env.dt != dt or env.step_index != 0 or env.physics.steps != 0:
        raise ValueError('timing capture initialization clock differs from the frozen case')
    write(out, 'native_layout.json', verify_live_layout(env.physics, layout))
    stage('training_prefix_start')

    def advance(label, action, expected, expected_index):
        if summary['physics_steps']+substeps > max_steps or time.monotonic()-started >= limits['max_worker_wall_s']:
            raise RuntimeError('training timing capture substep or worker wall cap')
        previous, before = env.state, env.physics.steps
        try:
            row = env.step(action)
        finally:
            summary['physics_steps'] += env.physics.steps-before
            if label == 'prefix':
                summary['prefix_substeps'] += env.physics.steps-before
        state = row.next_state_before_reset
        if (row.episode_id != case['episode_id'] or row.step_index != expected_index or env.step_index != expected_index+1
                or row.state_t != previous or row.state_t.time_s != before*dt or before != expected_index*substeps
                or env.physics.steps-before != substeps or state.time_s != env.physics.steps*dt
                or row.invalid_reason is not None or row.terminated or row.truncated
                or row.requested_action != action or state.previous_action != action
                or {'q': row.executed_joint_q, 'qd': row.executed_joint_qd} != expected):
            write(out, 'failed_transition.json', {'branch': label, 'transition': row.to_dict()})
            raise RuntimeError(f'{label} transition failed the frozen clock, command or episode check at tick {expected_index}')
        return row

    with (out/'prefix.jsonl').open('x', encoding='utf-8') as stream:
        for index, (action, command) in enumerate(zip(evidence['prefix_requested'], evidence['prefix_executed'], strict=True)):
            row = advance('prefix', action, command, index)
            stream.write(json.dumps({'absolute_tick': index, 'state_before_reset': asdict(row.next_state_before_reset),
                                     'task': row.task}, allow_nan=False)+'\n')
            if (index+1) % 24 == 0:
                stream.flush()
                guard()
                stage(f'prefix_tick_{index+1}')

    native = env.snapshot()
    physics = native['physics']
    if (summary['prefix_substeps'] != limits['prefix_substeps_per_case'] or native['step_index'] != tick
            or physics['steps'] != tick*substeps or native['controller'] != evidence['snapshot_control_only']
            or physics['command'][2].tolist() != evidence['snapshot_joint_qdd_rad_s2']):
        raise ValueError('recomputed prefix did not reach its frozen controller checkpoint')
    write(out, 'snapshot.json', {**native, 'state': asdict(native['state']), 'physics': {**physics,
                                'arrays': {name: values.tolist() for name, values in physics['arrays'].items()},
                                'command': [values.tolist() for values in physics['command']]}})
    stage('native_snapshot_saved')

    def replay(label):
        write(out, f'{label}_readback.json', restore_and_readback(env, native, substeps))
        guard()
        stage(f'{label}_start')
        branch = evidence['branches'][label]

        def step(action):
            return advance(label, action, branch['executed'][env.step_index-tick], env.step_index)

        trace = _record_nominal(env, native, branch['requested'], step, substeps, substeps*dt)
        if any(row['physics_step'] != tick*substeps+index for index, row in enumerate(trace['contacts'], 1)):
            raise ValueError(f'{label} physical contact clock is incomplete')
        filename = f'{label}.json'
        write(out, filename, trace)
        trace = json.loads((out/filename).read_bytes())
        _full_branch(trace, tick)
        summary['completed_branches'].append(label)
        guard()
        stage(f'{label}_complete')
        return trace

    labels = plan['branch_order']
    traces = {label: replay(label) for label in labels[:2]}
    nominal = nominal_gate(traces[labels[0]], traces[labels[1]], tick)
    summary['nominal_passed'] = bool(nominal['passed'])
    write(out, 'nominal_comparison.json', nominal)
    stage('nominal_comparison')
    if not nominal['passed']:
        raise RuntimeError('same-case nominal replay gate failed')

    shared = {}
    for label in labels[2:]:
        traces[label] = replay(label)
        if label.startswith('pitch_reversal_'):
            sign = label.removeprefix('pitch_reversal_')
            comparison = _prefix_gate(traces[f'pitch_pulse_{sign}'], traces[label], tick, 8, substeps)
            shared[sign] = comparison['passed']
            write(out, f'shared_prefix_{sign}.json', comparison)
            stage(f'shared_prefix_{sign}')
            if not comparison['passed']:
                raise RuntimeError(f'shared action prefix gate failed for {sign}')

    if summary['physics_steps'] != max_steps or shared != {'plus': True, 'minus': True}:
        raise ValueError('training capture did not exhaust the frozen branches and step count')
    write(out, 'signal_report.json', signal_report(plan, traces))
    guard()
    summary.update(status='training_timing_capture_completed', shared_prefix_passed=True, control_accepted=False)
