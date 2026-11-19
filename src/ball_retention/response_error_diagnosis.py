"""Locate response disagreement in saved paired trajectories using SI quantities."""

import numpy as np

from .action_eval import _direction


FAMILIES = {
    'position': (slice(0, 3), 'm', 'position_floor_m'),
    'velocity': (slice(3, 6), 'm/s', 'velocity_floor_m_s'),
    'spin': (slice(6, 9), 'rad/s', 'spin_floor_rad_s'),
}


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} requires finite values of shape {shape}')
    return result


def _norm(vector):
    return float(np.linalg.norm(vector))


def _selected(rows, leads):
    return {str(lead): rows[str(lead)] for lead in leads}


def _branch(truth, free, local, report_leads):
    error = {'free': free-truth, 'local': local-truth, 'propagated': free-local}
    np.testing.assert_allclose(error['free'], error['local']+error['propagated'], rtol=0., atol=1e-12)
    rows = {str(i+1): {} for i in range(len(truth))}
    whole = {}
    for name, (components, units, _) in FAMILIES.items():
        norms = {key: np.linalg.norm(value[:, components], axis=-1) for key, value in error.items()}
        whole[name] = {'units': units, **{f'{key}_rmse': float(np.sqrt(np.mean(value**2)))
                                         for key, value in norms.items()}}
        for i, row in enumerate(rows.values()):
            row[name] = {
                'units': units,
                **{f'{key}_state': value[i, components].tolist() for key, value in
                   (('newton', truth), ('free', free), ('local', local))},
                **{f'{key}_error': value[i, components].tolist() for key, value in error.items()},
                **{f'{key}_error_norm': float(value[i]) for key, value in norms.items()},
            }
    return {'by_lead': rows, 'report_leads': _selected(rows, report_leads), 'whole_trajectory': whole}


def _compare(estimate, reference, threshold, gates):
    estimate_norm, reference_norm = _norm(estimate), _norm(reference)
    estimated_resolved, reference_resolved = estimate_norm >= threshold, reference_norm >= threshold
    strong = reference_norm >= gates['strong_multiplier']*threshold
    cosine, direction = _direction(estimate, reference, estimated_resolved, reference_resolved, ('model', 'newton'))
    relative = _norm(estimate-reference)/reference_norm if reference_resolved else None
    agrees = (cosine is not None and cosine >= gates['min_cosine']
              and relative <= gates['max_relative_error']) if strong else None
    return {'cosine': cosine, 'direction': direction, 'relative_error': relative, 'agrees': agrees,
            'magnitude_ratio': estimate_norm/reference_norm if reference_resolved else None,
            'newton_resolved': reference_resolved, 'newton_strong': strong, 'model_resolved': estimated_resolved}


def _onset(rows, method, consecutive):
    comparisons = [row['position']['comparisons'][method] for row in rows.values()]
    opposed = [value['newton_strong'] and value['direction'] == 'opposed' for value in comparisons]
    return {
        'first_strong_lead': next((i+1 for i, value in enumerate(comparisons) if value['newton_strong']), None),
        'first_strong_opposed_lead': next((i+1 for i, value in enumerate(opposed) if value), None),
        'first_sustained_strong_opposed_lead': next((i+1 for i in range(len(opposed)-consecutive+1)
                                                    if all(opposed[i:i+consecutive])), None),
        'strong_opposed_leads': [i+1 for i, value in enumerate(opposed) if value],
    }


def _pair(truth, free, local, jvp_effect, nominal, initial, gates, report_leads, consecutive):
    # The local prediction starts from each branch's measured prior state. Its
    # signed output still includes the separation already present at that lead.
    newton, finite, one_step = ((plus-minus)/2 for plus, minus in (truth, free, local))
    inherited = np.vstack((np.zeros_like(initial), newton[:-1]))
    local_increments = one_step-inherited
    newton_increments = newton-inherited
    nominal_increments = [np.diff(np.vstack((initial, value)), axis=0) for value in nominal]
    rows = {str(i+1): {} for i in range(len(newton))}
    for name, (components, units, floor_key) in FAMILIES.items():
        floor = gates[floor_key]
        for i, row in enumerate(rows.values()):
            p, f, l, j, inherit, p_increment, l_increment = (
                value[i, components] for value in (newton, finite, one_step, jvp_effect, inherited,
                                                   newton_increments, local_increments))
            noise, increment_noise = (_norm((value[0][i, components]-value[1][i, components])/2)
                                      for value in (nominal, nominal_increments))
            numerical_floor = 32*np.finfo(float).eps * max(1., *(
                _norm(state[i, components]) for state in (*truth, *free, *nominal)))
            threshold = float(max(floor, gates['noise_multiplier']*noise, numerical_floor))
            increment_threshold = float(max(floor, gates['noise_multiplier']*increment_noise, numerical_floor))
            vectors = {'newton': p, 'free': f, 'local': l, 'jvp_effect': j, 'inherited_newton': inherit,
                       'newton_increment': p_increment, 'local_increment': l_increment,
                       'free_error': f-p, 'local_error': l-p, 'propagated_error': f-l}
            row[name] = {
                'units': units,
                'vectors': {key: value.tolist() for key, value in vectors.items()},
                'norms': {key: _norm(value) for key, value in vectors.items()},
                'same_session_nominal_half_spread': noise,
                'same_session_nominal_increment_half_spread': increment_noise,
                'signal_floor': threshold,
                'increment_signal_floor': increment_threshold,
                'strong_signal_floor': gates['strong_multiplier']*threshold,
                'comparisons': {'free': _compare(f, p, threshold, gates), 'jvp': _compare(j, p, threshold, gates),
                                'local': _compare(l, p, threshold, gates),
                                'local_increment': _compare(l_increment, p_increment, increment_threshold, gates)},
            }
    return {'by_lead': rows, 'report_leads': _selected(rows, report_leads),
            'position_onsets': {key: _onset(rows, key, consecutive)
                                for key in ('free', 'jvp', 'local', 'local_increment')}}


def diagnose_responses(plan, truths, free, local, frozen_model, initial):
    """Compare lead-aligned responses and split errors from offline predictions.

    All trajectories contain post-step tool-frame [p, v, spin] in SI.
    Local predictions condition on each Newton pre-step state while free
    predictions are generated recursively from the common initial state.
    """
    labels = plan['branch_order']
    horizon = plan['horizon']
    report_leads = plan['report_leads']
    consecutive = plan['onset_consecutive_leads']
    if (not report_leads or len(report_leads) != len(set(report_leads))
            or any(type(lead) is not int or not 1 <= lead <= horizon for lead in report_leads)
            or type(consecutive) is not int or not 1 <= consecutive <= horizon):
        raise ValueError('response diagnosis requires in-range report and consecutive onset leads')
    initial = _array(initial, (9,), 'shared initial state')
    for group in (truths, free, local):
        if set(group) != set(labels):
            raise ValueError('response diagnosis requires every frozen branch')
    truths, free, local = ({label: _array(group[label], (horizon, 9), label) for label in labels}
                           for group in (truths, free, local))
    branches = {label: _branch(truths[label], free[label], local[label], report_leads) for label in labels}
    nominal = [truths[label] for label in labels[:plan['budget']['baseline_replays']]]
    if len(nominal) != 2:
        raise ValueError('response diagnosis requires two nominal references')
    result = {}
    for direction in plan['directions']:
        plus, minus = f'{direction}_plus', f'{direction}_minus'
        jvp = _array(frozen_model['pairs'][direction]['directional_jvp'], (horizon, 9), direction)
        result[direction] = _pair((truths[plus], truths[minus]), (free[plus], free[minus]),
                                  (local[plus], local[minus]), plan['amplitude']*jvp,
                                  nominal, initial, plan['gates'], report_leads, consecutive)
    return {
        'branches': branches, 'directions': result,
        'definitions': {
            'free_error': 'free prediction minus Newton post-step state',
            'local_error': 'offline one-step prediction from Newton pre-step state minus Newton post-step state',
            'propagated_error': 'free prediction minus local prediction, with the same causal tool feature row',
            'pair_vectors': 'signed half-difference from positive and negative branches; JVP scaled by amplitude',
            'local_increment': 'local predicted signed half-difference minus inherited Newton pre-step half-difference',
            'increment_floor': 'same SI floor and multiple applied to nominal repeat increment noise',
            'onset': f'negative direction at {consecutive} consecutive leads with strong Newton magnitude',
            'rmse': 'root mean square of the per-lead Euclidean vector error norms',
        },
    }
