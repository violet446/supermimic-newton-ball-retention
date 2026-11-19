"""Run the frozen paired action sequences after deterministic nominal admission."""

from hashlib import sha256
import json
import os
import time

import numpy as np

from ball_retention.deterministic_response_plan import validate_case
from ball_retention.guard import process_memory
from ball_retention.resources import directory_bytes

from archive_restore import write
from archive_restore_metrics import POSITION_RESOLUTION_M
from deterministic_archive_metrics import _full_branch, _live_pair, native_reference_metrics
from deterministic_response_metrics import response_metrics
from native_repeatability import _make_env, _record_nominal
from saved_native_snapshot import decode_snapshot, restore_and_readback, verify_live_layout


def _nominal_gate(new, recorded, leads):
    metrics = native_reference_metrics(new, recorded)
    spread = metrics['all_branch_max_half_spread_by_lead']
    resolved = all(spread[str(lead)]['position_m'] <= POSITION_RESOLUTION_M for lead in leads)
    metrics.update(passed=metrics['numerical_equivalent'] and resolved, position_resolution_passed=resolved,
                   position_resolution_gate_leads=list(leads),
                   all_nominal_json_equal=all(trace == recorded[0] for trace in (*new, *recorded)))
    return metrics


def _prefix_gate(pulse, reversal, tick, prefix, substeps):
    """Apply nominal replay tolerances to every shared causal lead."""
    first, second = ({name: values[:prefix] for name, values in _full_branch(trace, tick).items()}
                     for trace in (pulse, reversal))
    result = _live_pair(first, second)
    position_delta = first['raw'][:, :3]-second['raw'][:, :3]
    spread = (np.linalg.norm(position_delta, axis=1)/2).tolist()
    result.update(
        passed=result['numerical_equivalent'] and max(spread) <= POSITION_RESOLUTION_M,
        shared_leads=prefix, position_half_spread_m_by_lead=spread, position_resolution_m=POSITION_RESOLUTION_M,
        contacts_match=pulse['contacts'][:prefix*substeps] == reversal['contacts'][:prefix*substeps],
    )
    return result


def _response(plan, traces, model):
    gates = plan['gates']
    baseline = [traces[label]['states_raw'] for label in plan['branch_order'][:2]]
    directions, primary = {}, {}
    lead = plan['primary_lead']
    for name in plan['directions']:
        model_pair = model['pairs'][name]
        result = response_metrics(
            baseline, traces[f'{name}_plus']['states_raw'], traces[f'{name}_minus']['states_raw'],
            model['nominal'], model_pair['plus'], model_pair['minus'], model_pair['directional_jvp'],
            plan['amplitude'], leads=plan['leads'],
            floors=(gates['position_floor_m'], gates['velocity_floor_m_s'], gates['spin_floor_rad_s']),
            **{key: gates[key] for key in ('noise_multiplier', 'strong_multiplier', 'min_cosine', 'max_relative_error')},
        )
        directions[name] = result
        position = result['by_lead'][str(lead)]['position']
        jvp = position['model_jvp_vs_newton']
        secant = position['model_fd_vs_newton']
        strong = position['newton_strong']
        signal = 'strong' if strong else ('resolved_below_strong_floor' if position['newton_resolved'] else 'unresolved')
        primary[name] = {
            'lead': lead, 'signal': signal, 'newton_half_difference_norm_m': position['newton_half_difference_norm'],
            'strong_signal_floor_m': position['strong_signal_floor'],
            'nominal_repeat_half_spread_m': position['nominal_repeat_half_spread'],
            'absolute_model_nominal_error_m': position['absolute_model_nominal_error_norm'],
            'model_jvp_vs_newton': jvp, 'model_secant_vs_newton': secant,
            'both_agree': (jvp['agrees'] and secant['agrees']) if strong else None,
        }
    return {'primary': primary, 'directions': directions,
            'scope': 'One development snapshot, one finite amplitude and one frozen model.'}


def run(root, cfg, out, max_steps, summary, stage):
    started = time.monotonic()
    plan, evidence = validate_case(root, cfg, max_steps)
    limits = plan['budget']
    tick, substeps = plan['case']['tick'], plan['control_substeps']
    dt = cfg['physics']['dt_s']
    labels = plan['branch_order']
    native = decode_snapshot(evidence['snapshot'], evidence['layout'], cfg, tick, substeps)

    def guard():
        memory = process_memory(os.getpid())
        if (time.monotonic()-started >= limits['max_worker_wall_s']
                or memory['peak_rss_bytes'] > limits['max_process_rss_gib']*1024**3
                or memory['peak_private_commit_bytes'] > limits['max_process_commit_gib']*1024**3
                or directory_bytes(out) > limits['max_new_output_mib']*1024**2):
            raise RuntimeError('deterministic action response worker budget reached')

    guard()
    summary['completed_branches'] = []
    stage('frozen_response_evidence_verified')
    env = _make_env(root, cfg, summary, stage, expected_device_arch=plan['runtime']['cuda_arch'])
    env.reset(episode_id='s3-deterministic-action-response')
    if env.substeps != substeps or env.dt != dt or env.physics.steps != 0 or env.step_index != 0:
        raise ValueError('scene did not initialize at the frozen zero-prefix clock')
    summary['nominal_prefix_substeps'] = 0
    write(out, 'native_layout.json', verify_live_layout(env.physics, evidence['layout']))
    guard()
    stage('fresh_scene_layout_verified')

    def replay(label):
        branch = evidence['branch_commands'][label]
        expected = branch['executed']
        write(out, f'{label}_readback.json', restore_and_readback(env, native, substeps))
        guard()
        stage(f'{label}_start')

        def advance(action):
            if (summary['physics_steps']+substeps > max_steps
                    or time.monotonic()-started >= limits['max_worker_wall_s']):
                raise RuntimeError('deterministic action response substep or worker wall cap')
            before = env.physics.steps
            previous = env.state
            try:
                row = env.step(action)
            finally:
                summary['physics_steps'] += env.physics.steps-before
            lead = env.step_index-tick
            state = row.next_state_before_reset
            if (not 1 <= lead <= plan['horizon'] or row.episode_id != env.episode_id
                    or row.step_index != env.step_index-1 or row.state_t != previous or row.state_t.time_s != before*dt
                    or env.physics.steps != env.step_index*substeps or env.physics.steps-before != substeps
                    or state.time_s != env.physics.steps*dt
                    or row.invalid_reason is not None or row.terminated or row.truncated
                    or row.requested_action != action or state.previous_action != action
                    or {'q': row.executed_joint_q, 'qd': row.executed_joint_qd} != expected[lead-1]):
                raise RuntimeError(f'{label} transition disagrees with the frozen clock, commands or episode scope')
            return row

        trace = _record_nominal(env, native, branch['requested'], advance, substeps, substeps*dt)
        for offset, contact in enumerate(trace['contacts'], 1):
            if contact['physics_step'] != tick*substeps+offset:
                raise ValueError(f'{label} contact log is missing an ordered physical substep')
        _full_branch(trace, tick)
        filename = f'{label}.json'
        write(out, filename, trace)
        summary['completed_branches'].append(label)
        guard()
        stage(f'{label}_complete')
        return json.loads((out/filename).read_bytes())

    traces = {label: replay(label) for label in labels[:limits['baseline_replays']]}
    nominal = _nominal_gate(list(traces.values()), evidence['recorded'], plan['leads'])
    nominal_digests = {label: sha256((out/f'{label}.json').read_bytes()).hexdigest() for label in traces}
    recorded_digests = [plan['inputs_sha256'][path] for path in plan['nominal_replays']]
    nominal.update(new_nominal_sha256=nominal_digests, recorded_nominal_sha256=recorded_digests,
                   all_nominal_files_byte_identical=len(set([*nominal_digests.values(), *recorded_digests])) == 1)
    summary['nominal_comparison'] = nominal
    write(out, 'nominal_comparison.json', nominal)
    stage('nominal_comparison')
    if not nominal['passed']:
        raise RuntimeError('nominal comparison failed its frozen gate')

    by_sign = {}
    for label in labels[limits['baseline_replays']:]:
        traces[label] = replay(label)
        if label.startswith('pitch_reversal_'):
            sign = label.removeprefix('pitch_reversal_')
            shared = _prefix_gate(traces[f'pitch_pulse_{sign}'], traces[label], tick, 8, substeps)
            by_sign[sign] = shared
            summary['shared_prefix_checks'] = {'passed': len(by_sign) == 2 and all(row['passed'] for row in by_sign.values()),
                                               'by_sign': by_sign}
            write(out, f'shared_prefix_{sign}.json', shared)
            stage(f'shared_prefix_{sign}')
            if not shared['passed']:
                raise RuntimeError(f'shared action prefix for {sign} failed its frozen gate')

    if summary['physics_steps'] != max_steps or not summary['shared_prefix_checks']['passed']:
        raise ValueError('completed replay count or shared prefix checks differ from the frozen plan')
    comparison = _response(plan, traces, evidence['model_rollouts'])
    write(out, 'response_comparison.json', comparison)
    guard()
    summary.update(status='deterministic_action_response_completed', primary=comparison['primary'],
                   control_accepted=False)
