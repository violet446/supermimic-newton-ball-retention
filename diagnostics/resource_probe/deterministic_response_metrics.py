"""Compare frozen model and Newton directional responses at matched leads."""

import math

import numpy as np

from ball_retention.action_eval import _direction
from ball_retention.pair_data import DEFAULT_RESPONSE_FLOOR


HORIZON = 32
QUANTITIES = ('position', 'velocity', 'spin')


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite values with shape {shape}')
    return array


def _comparison(estimate, reference, floor, names):
    estimate_norm = float(np.linalg.norm(estimate))
    reference_norm = float(np.linalg.norm(reference))
    estimate_resolved = estimate_norm >= floor
    reference_resolved = reference_norm >= floor
    cosine, direction = _direction(estimate, reference, estimate_resolved, reference_resolved, names)
    return {'cosine': cosine, 'direction': direction,
            'relative_error': float(np.linalg.norm(estimate-reference))/reference_norm if reference_resolved else None,
            'magnitude_ratio': estimate_norm/reference_norm if reference_resolved else None}


def _quantity(physics_base, physics_positive, physics_negative, model_base, model_positive, model_negative,
              jvp, amplitude, floor, noise_multiplier, strong_multiplier, min_cosine, max_relative_error):
    newton_first, newton_second = physics_base
    newton_center = physics_base.mean(axis=0)
    noise = float(np.linalg.norm(newton_first-newton_second)/2)
    numerical_floor = float(32*np.finfo(float).eps * max(1., *(
        np.linalg.norm(state) for state in (newton_first, newton_second, physics_positive, physics_negative,
                                           model_base, model_positive, model_negative))))
    threshold = max(floor, noise_multiplier*noise, numerical_floor)
    newton_effect = (physics_positive-physics_negative)/2
    model_fd_effect = (model_positive-model_negative)/2
    model_jvp_effect = amplitude*jvp
    newton_signal, fd_signal, jvp_signal = (float(np.linalg.norm(effect))
                                          for effect in (newton_effect, model_fd_effect, model_jvp_effect))
    newton_strong = newton_signal >= strong_multiplier*threshold

    comparisons = {}
    for name, estimate in (('model_fd', model_fd_effect), ('model_jvp', model_jvp_effect)):
        comparison = _comparison(estimate, newton_effect, threshold, (name, 'newton'))
        cosine = comparison['cosine']
        comparison['agrees'] = (cosine is not None and cosine >= min_cosine
                                and comparison['relative_error'] <= max_relative_error) if newton_strong else None
        comparisons[f'{name}_vs_newton'] = comparison

    return {
        'newton_half_difference': newton_effect.tolist(),
        'model_fd_half_difference': model_fd_effect.tolist(),
        'model_jvp_linearized_displacement': model_jvp_effect.tolist(),
        'newton_fd_derivative': (newton_effect/amplitude).tolist(),
        'model_fd_derivative': (model_fd_effect/amplitude).tolist(),
        'model_jvp_derivative': jvp.tolist(),
        'newton_half_difference_norm': newton_signal,
        'model_fd_half_difference_norm': fd_signal,
        'model_jvp_linearized_displacement_norm': jvp_signal,
        'nominal_repeat_half_spread': noise,
        'numerical_floor': numerical_floor,
        'signal_floor': threshold,
        'strong_signal_floor': strong_multiplier*threshold,
        'newton_resolved': newton_signal >= threshold,
        'newton_strong': newton_strong,
        'model_fd_resolved': fd_signal >= threshold,
        'model_jvp_resolved': jvp_signal >= threshold,
        **comparisons,
        'model_jvp_vs_model_fd': _comparison(model_jvp_effect, model_fd_effect, threshold, ('model_jvp', 'model_fd')),
        'newton_midpoint_shift_norm': float(np.linalg.norm((physics_positive+physics_negative)/2-newton_center)),
        'model_midpoint_shift_norm': float(np.linalg.norm((model_positive+model_negative)/2-model_base)),
        'absolute_model_nominal_error_norm': float(np.linalg.norm(model_base-newton_center)),
        'absolute_model_positive_error_norm': float(np.linalg.norm(model_positive-physics_positive)),
        'absolute_model_negative_error_norm': float(np.linalg.norm(model_negative-physics_negative)),
    }


def response_metrics(newton_nominal, newton_positive, newton_negative, model_nominal, model_positive,
                     model_negative, model_jvp, amplitude, *, floors=DEFAULT_RESPONSE_FLOOR,
                     noise_multiplier=5., strong_multiplier=10., min_cosine=.8, max_relative_error=.5,
                     leads=(8, 16, 32)):
    """Evaluate SI tool-frame position, velocity and spin for one paired direction.

    Nominal Newton is (2, 32, 9), with one full trajectory per restored
    replay. The other arrays are (32, 9), including a model JVP in physical
    units per unit of the signed scalar multiplying the frozen direction.
    Newton/model finite differences use the same signed trial amplitude.
    Agreement is unavailable when the Newton signal is below its strong
    threshold. The caller chooses which lead and quantity control the result.
    """
    leads = tuple(leads)
    floors = _finite_array(floors, (len(QUANTITIES),), 'response floors')
    if not (0 < amplitude < 1 and np.all(floors > 0) and math.isfinite(noise_multiplier)
            and noise_multiplier > 0 and math.isfinite(strong_multiplier) and strong_multiplier >= 1
            and 0 <= min_cosine <= 1 and math.isfinite(max_relative_error) and max_relative_error >= 0):
        raise ValueError('invalid response amplitude or thresholds')
    if (not leads or any(type(lead) is not int or not 0 < lead <= HORIZON for lead in leads)
            or len(set(leads)) != len(leads)):
        raise ValueError('response leads must be distinct integers in the branch horizon')
    replays = _finite_array(newton_nominal, (2, HORIZON, 9), 'Newton nominal')
    paths = [_finite_array(value, (HORIZON, 9), name) for name, value in (
        ('Newton positive', newton_positive), ('Newton negative', newton_negative),
        ('model nominal', model_nominal), ('model positive', model_positive),
        ('model negative', model_negative), ('model JVP', model_jvp))]

    by_lead = {}
    for lead in leads:
        by_quantity = {}
        for i, quantity in enumerate(QUANTITIES):
            components = slice(3*i, 3*(i+1))
            by_quantity[quantity] = _quantity(
                replays[:, lead-1, components], *(path[lead-1, components] for path in paths),
                amplitude, float(floors[i]), noise_multiplier, strong_multiplier, min_cosine, max_relative_error)
        by_lead[str(lead)] = by_quantity
    return {'amplitude': amplitude, 'absolute_floors_si': dict(zip(QUANTITIES, floors.tolist())),
            'noise_multiplier': noise_multiplier, 'strong_multiplier': strong_multiplier,
            'min_cosine': min_cosine, 'max_relative_error': max_relative_error, 'by_lead': by_lead}
