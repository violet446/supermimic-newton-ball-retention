"""Gates for the frozen deterministic nominal replay batches."""
from itertools import combinations, product

import numpy as np

from ball_retention.task import ball_in_tool

if __package__:
    from . import archive_restore_metrics as metrics
else:
    import archive_restore_metrics as metrics


LEADS = (8, 16, 32)
TICK = 168
PHYSICS_DT = 1 / 960
CONTROL_FIELDS = ('joint_q', 'joint_qd', 'tilt_rad', 'tilt_rate_rad_s', 'previous_action')


def _full_branch(branch, tick):
    trace = metrics._branch(branch, require_full_state=True)
    clock = (tick + np.arange(1, metrics.HORIZON + 1)) * metrics.SUBSTEPS * PHYSICS_DT
    elapsed = clock - tick * metrics.SUBSTEPS * PHYSICS_DT
    if ([state.time_s for state in trace['states']] != clock.tolist()
            or not np.array_equal(trace['elapsed'], elapsed)):
        raise ValueError('replay clocks disagree with the fixed snapshot tick and physics clock')

    derived_raw = []
    for state, action, command, retention in zip(trace['states'], trace['actions'], trace['commands'],
                                                  trace['retention']):
        if command.keys() != {'q', 'qd'}:
            raise ValueError('replay must contain both executed command arrays')
        for key, expected in (('q', state.joint_q), ('qd', state.joint_qd)):
            values = metrics._finite_array(command[key], (metrics.SUBSTEPS, 6), f'command {key}')
            if not np.array_equal(values[-1], expected):
                raise ValueError(f'public joint state disagrees with final executed {key}')
        if not np.array_equal(state.previous_action, action):
            raise ValueError('replay public state disagrees with nominal action')
        if (retention.keys() != {'terminated', 'truncated', 'reason', 'carrying_candidate', 'outside_grace_active'}
                or retention['terminated'] is not False or retention['truncated'] is not False
                or retention['reason'] is not None or type(retention['carrying_candidate']) is not bool
                or retention['outside_grace_active'] is not (state.departure_outside_s > 0)):
            raise ValueError('replay retention classification inconsistent with the public state and nominal scope')
        terms = ball_in_tool(state.ball, state.tool)
        derived_raw.append(np.r_[terms['position'], terms['velocity'], terms['ball_omega']])
    if not np.allclose(trace['raw'], derived_raw, rtol=0., atol=1e-12):
        raise ValueError('recorded tool-frame values disagree with the public states')
    return trace


def validate_recorded_replays(native_replays, old_replays, nominal_actions, tick=TICK):
    """Validate the reference traces and their common nominal action and clock."""
    native = [_full_branch(branch, tick) for branch in metrics._two_branches(native_replays, 'recorded native')]
    historical = [metrics._branch(branch, require_full_state=False)
                  for branch in metrics._two_branches(old_replays, 'recorded historical')]
    for trace in (*native, *historical):
        if (not np.array_equal(trace['actions'], nominal_actions)
                or not np.array_equal(trace['elapsed'], native[0]['elapsed'])):
            raise ValueError('frozen nominal actions or lead clocks disagree across recorded traces')


def _live_pair(first, second):
    result = metrics._live_pair(first, second)
    same_clock = ([state.time_s for state in first['states']] == [state.time_s for state in second['states']]
                  and np.array_equal(first['elapsed'], second['elapsed']))
    control_state_match = all(getattr(a, key) == getattr(b, key)
                              for a, b in zip(first['states'], second['states']) for key in CONTROL_FIELDS)
    reference_phase_match = ([state.reference_phase for state in first['states']]
                             == [state.reference_phase for state in second['states']])
    result['actions_and_timing_match'] &= bool(same_clock)
    result['public_control_state_match'] = control_state_match
    result['public_reference_phase_match'] = reference_phase_match
    result['numerical_equivalent'] &= bool(same_clock and control_state_match and reference_phase_match)
    return result


def _compare(groups, required_leads, *, across_sources=False):
    traces = []
    labels = []
    for name, branches in groups.items():
        traces.extend(_full_branch(branch, TICK) for branch in metrics._two_branches(branches, name))
        labels.extend(f'{name}_{i}' for i in (1, 2))
    pairs = list(combinations(range(len(traces)), 2))
    comparisons = {f'{labels[i]}_vs_{labels[j]}': _live_pair(traces[i], traces[j]) for i, j in pairs}
    spread = metrics._half_spreads(traces, pairs, LEADS)
    cross = metrics._half_spreads(traces, list(product((0, 1), (2, 3))), LEADS) if len(groups) == 2 else None
    numerical = all(row['numerical_equivalent'] for row in comparisons.values())
    gate_spread = cross if across_sources else spread
    resolved = all(gate_spread[str(lead)]['position_m'] <= metrics.POSITION_RESOLUTION_M for lead in required_leads)
    result = {'passed': numerical and resolved, 'numerical_equivalent': numerical,
              'position_resolution_passed': resolved, 'position_resolution_m': metrics.POSITION_RESOLUTION_M,
              'position_resolution_gate_leads': list(required_leads), 'tool_relative_component_limits': metrics.LIMITS,
              'pairwise': comparisons, 'all_branch_max_half_spread_by_lead': spread}
    if len(groups) == 2:
        result['repeat_half_spread_by_source'] = {
            name: metrics._half_spreads(traces, [(i, i + 1)], LEADS)
            for i, name in zip((0, 2), groups)
        }
        result['cross_source_max_half_spread_by_lead'] = cross
    return result


def native_reference_metrics(new_native, recorded_native):
    """Compare the fresh native branches with both frozen native branches."""
    return _compare({'native': new_native, 'recorded': recorded_native}, (32,))


def same_configuration_metrics(new_native, reconstructed):
    """Check all four native and same-configuration reconstructed trajectories."""
    return _compare({'native': new_native, 'reconstructed': reconstructed}, (32,), across_sources=True)


def historical_import_metrics(imported_old, recorded_old):
    """Gate repeatability in the new configuration and measure old-config divergence."""
    result = _compare({'historical_import': imported_old}, LEADS)
    context = metrics.historical_metrics(imported_old, recorded_old, LEADS)
    old = [np.asarray(branch['states_raw']) for branch in recorded_old]
    new = [np.asarray(branch['states_raw']) for branch in imported_old]
    cross_component = np.max([np.max(np.abs(a - b), axis=0) for a, b in product(new, old)], axis=0)
    result['cross_configuration'] = {
        'original_repeat_half_spread_by_lead': context['archived_replay_half_spread_by_lead'],
        'cross_source_max_half_spread_by_lead': context['cross_source_max_half_spread_by_lead'],
        'cross_source_max_component_difference': {
            name: float(np.max(cross_component[i:i + 3])) for i, name in zip((0, 3, 6), metrics.LIMITS)
        },
    }
    return result
