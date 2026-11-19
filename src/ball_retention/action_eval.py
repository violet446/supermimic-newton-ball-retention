"""Evaluate paired action perturbations at matched Newton and model leads."""
import math
from numbers import Integral

import numpy as np


COMPONENTS = {'position': slice(0, 3), 'velocity': slice(3, 6),
              'angular_velocity': slice(6, 9)}
POSITION_FLOOR_M = 1e-6
REPLAY_NOISE_FACTOR = 5.


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return result


def full_step_states(states_raw, substeps, control_substeps):
    """Keep nominal lead indices while marking a shortened final step as None."""
    if not isinstance(control_substeps, Integral) or control_substeps < 1:
        raise ValueError('control_substeps must be a positive integer')
    if len(states_raw) != len(substeps):
        raise ValueError('physics state and substep counts disagree')
    result = []
    for index, (state, steps) in enumerate(zip(states_raw, substeps)):
        if not isinstance(steps, Integral) or not 1 <= steps <= control_substeps:
            raise ValueError('invalid executed substep count')
        state = _vector(state, 9, 'physics state')
        if steps == control_substeps:
            result.append(state)
        elif index == len(states_raw)-1:
            result.append(None)
        else:
            raise ValueError('only the final state may have a shortened step')
    return result


def _at_lead(trajectory, lead, name):
    if lead > len(trajectory):
        return None, 'not_reached'
    value = trajectory[lead-1]
    if value is None:
        return None, 'shortened_step'
    return _vector(value, 9, name), 'full_step'


def _direction(first, second, first_resolved, second_resolved, names):
    if not first_resolved or not second_resolved:
        if first_resolved or second_resolved:
            label = names[0] if not first_resolved else names[1]
        else:
            label = 'both'
        return None, f'{label}_below_floor'
    cosine = float(np.clip(np.dot(first/np.linalg.norm(first), second/np.linalg.norm(second)), -1., 1.))
    direction = 'orthogonal' if abs(cosine) <= 1e-12 else 'aligned' if cosine > 0 else 'opposed'
    return cosine, direction


def _pair_metric(model_plus, model_minus, newton_plus, newton_minus,
                 replay_first, replay_second, epsilon, floor, noise_multiplier):
    # Half the paired change gives the typical displacement at one signed offset.
    model_effect = (model_plus-model_minus)/2
    newton_effect = (newton_plus-newton_minus)/2
    replay_noise = float(np.linalg.norm(replay_first-replay_second)/2)
    numerical_floor = 32*np.finfo(float).eps * max(1., *(np.linalg.norm(state) for state in (
        model_plus, model_minus, newton_plus, newton_minus, replay_first, replay_second)))
    threshold = float(max(floor, noise_multiplier*replay_noise, numerical_floor))
    model_signal = float(np.linalg.norm(model_effect))
    newton_signal = float(np.linalg.norm(newton_effect))
    model_resolved, newton_resolved = model_signal >= threshold, newton_signal >= threshold
    cosine, direction = _direction(model_effect, newton_effect, model_resolved, newton_resolved,
                                   ('model', 'newton'))
    return {'model_derivative': (model_effect/epsilon).tolist(),
            'newton_derivative': (newton_effect/epsilon).tolist(),
            'model_norm': model_signal/epsilon, 'newton_norm': newton_signal/epsilon,
            'model_half_spread': model_signal, 'newton_half_spread': newton_signal,
            'replay_half_spread': replay_noise, 'signal_threshold': threshold,
            'model_resolved': model_resolved, 'newton_resolved': newton_resolved,
            'cosine': cosine, 'direction': direction,
            'normalized_error': float(np.linalg.norm(model_effect-newton_effect)/newton_signal)
                                if newton_resolved else None}


def evaluate_pair(model_plus, model_minus, newton_plus, newton_minus, epsilon, target_xy,
                  *, replay_first, replay_second, leads=(1, 4, 8),
                  position_floor_m=POSITION_FLOOR_M, noise_multiplier=REPLAY_NOISE_FACTOR):
    """Compare derivatives with respect to the held normalized action offset.

    Each trajectory contains raw tool-frame p, relative v and rotated ball
    omega at consecutive post-step leads. Use None for a shortened step and
    omit its unexecuted tail. Two zero-offset replays set each lead's noise
    floor. The XY scalar is negative distance to the chosen tool-frame target.
    It is a local diagnostic of movement toward the target.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be positive and finite')
    epsilon = float(epsilon)
    if (not math.isfinite(position_floor_m) or position_floor_m < POSITION_FLOOR_M
            or not math.isfinite(noise_multiplier) or noise_multiplier <= 0):
        raise ValueError('expected position floor of at least 1um and positive replay noise multiplier')
    target = _vector(target_xy, 2, 'XY target')
    leads = tuple(leads)
    if not leads or len(set(leads)) != len(leads) or any(
            not isinstance(lead, Integral) or lead < 1 for lead in leads):
        raise ValueError('leads must be distinct positive integers')

    def progress(state):
        return np.array([-np.linalg.norm(state[:2]-target)])

    pairs = {'newton_plus': newton_plus, 'newton_minus': newton_minus,
             'replay_first': replay_first, 'replay_second': replay_second}
    by_lead = {}
    for lead in leads:
        model = [_at_lead(path, lead, name)[0] for name, path in
                 (('model_plus', model_plus), ('model_minus', model_minus))]
        if any(value is None for value in model):
            raise ValueError(f'model prediction is incomplete at lead {lead}')
        physics, step_status = {}, {}
        for name, path in pairs.items():
            physics[name], step_status[name] = _at_lead(path, lead, name)
        row = {'status': 'comparable', 'step_status': step_status}
        by_lead[str(lead)] = row
        if any(physics[name] is None for name in ('newton_plus', 'newton_minus')):
            row['status'] = 'missing_newton_lead'
            continue
        if any(physics[name] is None for name in ('replay_first', 'replay_second')):
            row['status'] = 'missing_replay_lead'
            continue
        states = [*model, *physics.values()]
        for name, components in COMPONENTS.items():
            row[name] = _pair_metric(*(state[components] for state in states), epsilon,
                                     position_floor_m if name == 'position' else 0., noise_multiplier)
        score = _pair_metric(*(progress(state) for state in states), epsilon,
                             position_floor_m, noise_multiplier)
        score['model_derivative'] = score['model_derivative'][0]
        score['newton_derivative'] = score['newton_derivative'][0]
        score['sign_agrees'] = None if score['cosine'] is None else score['direction'] == 'aligned'
        row['progress_xy'] = score
    return {'epsilon': epsilon, 'position_floor_m': float(position_floor_m),
            'noise_multiplier': float(noise_multiplier), 'by_lead': by_lead}


def compare_scales(small_pair, large_pair, *, min_cosine=.8, max_relative_error=.5):
    """Check Newton derivative consistency across two positive action offsets.

    Compare the large-offset estimate against the small-offset estimate.
    Thresholds classify sensitivity across scales as a diagnostic.
    """
    small, large = small_pair['epsilon'], large_pair['epsilon']
    if not 0 < small < large:
        raise ValueError('expected ascending positive perturbation scales')
    if not 0 <= min_cosine <= 1 or not math.isfinite(max_relative_error) or max_relative_error < 0:
        raise ValueError('invalid scale consistency thresholds')
    small_leads, large_leads = small_pair['by_lead'], large_pair['by_lead']
    if small_leads.keys() != large_leads.keys():
        raise ValueError('perturbation scales must use the same leads')
    by_lead = {}
    for lead, small_row in small_leads.items():
        large_row = large_leads[lead]
        statuses = {'small': small_row['status'], 'large': large_row['status']}
        available = all(status == 'comparable' for status in statuses.values())
        row = {'status': 'comparable' if available else 'incomplete', 'pair_status': statuses}
        by_lead[lead] = row
        if not available:
            continue
        for key in (*COMPONENTS, 'progress_xy'):
            a, b = small_row[key], large_row[key]
            first, second = (np.atleast_1d(np.asarray(metric['newton_derivative'], dtype=float))
                             for metric in (a, b))
            cosine, direction = _direction(first, second, a['newton_resolved'], b['newton_resolved'],
                                           ('small', 'large'))
            error = float(np.linalg.norm(first-second)/a['newton_norm']) if a['newton_resolved'] else None
            consistent = None if cosine is None else cosine >= min_cosine and error <= max_relative_error
            row[key] = {'small_derivative': a['newton_derivative'], 'large_derivative': b['newton_derivative'],
                        'small_norm': a['newton_norm'], 'large_norm': b['newton_norm'],
                        'cosine': cosine, 'direction': direction, 'normalized_error': error,
                        'consistent': consistent}
    return {'epsilon_small': small, 'epsilon_large': large, 'min_cosine': min_cosine,
            'max_relative_error': max_relative_error, 'by_lead': by_lead}
