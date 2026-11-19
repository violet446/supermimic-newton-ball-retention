"""Compare the complete nominal rollouts in the archived-state restore pilot."""
from itertools import combinations, product

import numpy as np
from scipy.spatial.transform import Rotation

from ball_retention.collection import replay_error
from ball_retention.contracts import RigidState, State


HORIZON = 32
SUBSTEPS = 16
CONTROL_DT = SUBSTEPS / 960
POSITION_RESOLUTION_M = 1e-6
LIMITS = {'position_m': 1e-5, 'velocity_m_s': 1e-3, 'spin_rad_s': 2e-2}


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite values with shape {shape}')
    return array


def _two_branches(branches, name):
    if len(branches) != 2:
        raise ValueError(f'expected two {name} replays')
    return branches


def _branch(branch, require_full_state):
    substeps = branch['substeps']
    if len(substeps) != HORIZON or any(type(n) is not int or n != SUBSTEPS for n in substeps):
        raise ValueError('replay must execute all 16 substeps at all 32 leads')
    if branch['terminated'] is not False or branch['truncated'] is not False or branch['termination_reason'] is not None:
        raise ValueError('nominal replay ended early or has inconsistent ending flags')
    actions = _finite_array(branch['actions'], (HORIZON, 2), 'actions')
    if (np.abs(actions) >= 1.).any():
        raise ValueError('nominal replay reached a clipping boundary')
    elapsed = _finite_array(branch['elapsed_s'], (HORIZON,), 'elapsed_s')
    if not np.allclose(elapsed, CONTROL_DT*np.arange(1, HORIZON+1), atol=1e-9, rtol=0.):
        raise ValueError('replay lead clock disagrees with executed substeps')
    raw = _finite_array(branch['states_raw'], (HORIZON, 9), 'states_raw')
    if not require_full_state:
        return {'actions': actions, 'elapsed': elapsed, 'raw': raw}

    for key in ('states', 'commands', 'retention'):
        if len(branch[key]) != HORIZON:
            raise ValueError(f'{key} must describe every lead')
    states = []
    for fields in branch['states']:
        values = {name: RigidState(**fields[name]) for name in ('ball', 'tool')}
        state = State(**{**fields, **values})
        state.validate()
        states.append(state)
    times = np.array([state.time_s for state in states])
    if not np.allclose(times-times[0], elapsed-elapsed[0], atol=1e-9, rtol=0.):
        raise ValueError('absolute state clocks disagree with replay leads')
    return {'actions': actions, 'elapsed': elapsed, 'raw': raw, 'states': states,
            'commands': branch['commands'], 'retention': branch['retention']}


def _absolute(x, y):
    return float(np.max(np.abs(np.asarray(x)-np.asarray(y))))


def _kinematic_metrics(a, b):
    result = {}
    for body in ('ball', 'tool'):
        first = [getattr(state, body) for state in a]
        second = [getattr(state, body) for state in b]
        rotations = [Rotation.from_quat([s.quaternion_xyzw for s in group]) for group in (first, second)]
        result[body] = {
            'max_component_position_m': _absolute([s.position for s in first], [s.position for s in second]),
            'max_rotation_geodesic_rad': float(np.max((rotations[0].inv()*rotations[1]).magnitude())),
            'max_component_origin_velocity_m_s': _absolute([s.linear_velocity_origin for s in first],
                                                            [s.linear_velocity_origin for s in second]),
            'max_component_angular_velocity_rad_s': _absolute([s.angular_velocity for s in first],
                                                             [s.angular_velocity for s in second]),
        }
    return result


def _same_leads(a, b):
    return (np.array_equal(a['actions'], b['actions'])
            and np.allclose(a['elapsed'], b['elapsed'], atol=1e-9, rtol=0.))


def _live_pair(a, b):
    numerical = {key: _absolute(a['raw'][:, i:i+3], b['raw'][:, i:i+3])
                 for i, key in zip((0, 3, 6), LIMITS)}
    same_controls = replay_error(a['commands'], b['commands']) == 0.
    same_retention = a['retention'] == b['retention']
    times_a, times_b = ([s.time_s for s in group] for group in (a['states'], b['states']))
    same_clock = np.allclose(times_a, times_b, atol=1e-9, rtol=0.)
    return {'actions_and_timing_match': bool(_same_leads(a, b) and same_clock),
            'executed_commands_match': same_controls, 'retention_classification_match': same_retention,
            'tool_relative_max_component_errors': numerical,
            'body_world_errors': _kinematic_metrics(a['states'], b['states']),
            'numerical_equivalent': bool(_same_leads(a, b) and same_clock and same_controls and same_retention
                                         and all(numerical[key] <= cap for key, cap in LIMITS.items()))}


def _lead_list(leads):
    leads = tuple(leads)
    if not leads or len(set(leads)) != len(leads) or any(type(h) is not int or not 0 < h <= HORIZON for h in leads):
        raise ValueError('report leads must be distinct integers between 1 and 32')
    return leads


def _half_spreads(trajectories, pairs, leads):
    indices = np.array(leads)-1
    estimates = np.stack([np.linalg.norm((trajectories[i]['raw'][indices]-trajectories[j]['raw'][indices])
                                        .reshape(len(leads), 3, 3), axis=-1)/2 for i, j in pairs])
    names = ('position_m', 'velocity_m_s', 'spin_rad_s')
    return {str(lead): dict(zip(names, values.tolist())) for lead, values in zip(leads, estimates.max(axis=0))}


def live_metrics(native, imported, leads=(8, 16, 32)):
    """Compare all six native/imported pairings from one public State export."""
    leads = _lead_list(leads)
    branches = [_branch(branch, require_full_state=True)
                for group, label in ((native, 'native'), (imported, 'imported'))
                for branch in _two_branches(group, label)]
    all_pairs = list(combinations(range(4), 2))
    cross_pairs = list(product((0, 1), (2, 3)))
    cross_leads = tuple(dict.fromkeys((*leads, HORIZON)))
    cross = _half_spreads(branches, cross_pairs, cross_leads)
    names = ('native_0', 'native_1', 'imported_0', 'imported_1')
    comparisons = {f'{names[i]}_vs_{names[j]}': _live_pair(branches[i], branches[j]) for i, j in all_pairs}
    return {'numerical_equivalent': all(value['numerical_equivalent'] for value in comparisons.values()),
            'position_resolution_passed': cross[str(HORIZON)]['position_m'] <= POSITION_RESOLUTION_M,
            'tool_relative_component_limits': LIMITS, 'position_resolution_m': POSITION_RESOLUTION_M,
            'pairwise': comparisons,
            'native_repeat_half_spread_by_lead': _half_spreads(branches, [(0, 1)], leads),
            'imported_repeat_half_spread_by_lead': _half_spreads(branches, [(2, 3)], leads),
            'all_branch_max_half_spread_by_lead': _half_spreads(branches, all_pairs, leads),
            'cross_source_max_half_spread_by_lead': cross}


def native_metrics(branches, leads=(8, 16, 32)):
    """Check two independently restored continuations against the same ceilings."""
    leads = tuple(dict.fromkeys((*_lead_list(leads), HORIZON)))
    trajectories = [_branch(branch, require_full_state=True) for branch in _two_branches(branches, 'native')]
    comparison = _live_pair(*trajectories)
    half_spread = _half_spreads(trajectories, [(0, 1)], leads)
    return {'numerical_equivalent': comparison['numerical_equivalent'],
            'position_resolution_passed': half_spread[str(HORIZON)]['position_m'] <= POSITION_RESOLUTION_M,
            'tool_relative_component_limits': LIMITS, 'position_resolution_m': POSITION_RESOLUTION_M,
            'pairwise': comparison, 'half_spread_by_lead': half_spread}


def historical_metrics(replays, archived_replays, leads=(8, 16, 32)):
    """Report newly imported replays and the historical replays separately."""
    leads = _lead_list(leads)
    branches = [_branch(branch, require_full_state=False)
                for group, label in ((replays, 'historical import'), (archived_replays, 'recorded archive'))
                for branch in _two_branches(group, label)]
    if any(not _same_leads(branches[0], branch) for branch in branches[1:]):
        raise ValueError('historical validation requires the same nominal actions, substeps and lead timing')
    observed = _half_spreads(branches, [(0, 1)], leads)
    archived = _half_spreads(branches, [(2, 3)], leads)
    cross = _half_spreads(branches, list(product((0, 1), (2, 3))), leads)
    return {'position_resolution_m': POSITION_RESOLUTION_M,
            'position_resolution_passed': all(row['position_m'] <= POSITION_RESOLUTION_M for row in observed.values()),
            'archive_position_agrees_at_resolution': all(row['position_m'] <= POSITION_RESOLUTION_M for row in cross.values()),
            'new_replay_half_spread_by_lead': observed, 'archived_replay_half_spread_by_lead': archived,
            'cross_source_max_half_spread_by_lead': cross}
