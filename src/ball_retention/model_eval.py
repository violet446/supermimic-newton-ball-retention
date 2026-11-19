"""Paired world-model and baseline errors on fixed-duration episode windows."""
import math

import numpy as np


def _episode_masks(episode_ids, count):
    if episode_ids is None:
        return {}
    ids = np.asarray(episode_ids, dtype=object)
    if ids.shape != (count,) or any(not isinstance(name, str) or not name for name in ids):
        raise ValueError('expected a nonempty episode ID for each sample')
    return {name: ids == name for name in dict.fromkeys(ids)}


def _ball_errors(predicted, target):
    # Each metric is the RMS Euclidean error of its three components.
    difference = (predicted - target).reshape(-1, 3, 3)
    position, velocity, omega = np.sqrt(np.mean(np.sum(difference**2, axis=-1), axis=0))
    return {'position_rmse_mm': float(1000 * position),
            'relative_velocity_rmse_m_s': float(velocity),
            'angular_velocity_rmse_rad_s': float(omega)}


def evaluate_windows(predicted_raw, target_raw, initial_raw, dt, episode_ids=None, *, leads=(1, 4, 8)):
    """Score raw [position, relative velocity, rotated angular velocity] arrays.

    Predictions and targets have shape (windows, horizon, 9), beginning at
    lead 1, and initial states have shape (windows, 9). The caller provides only
    windows of complete control intervals. Pass each horizon's own windows and
    leads=(horizon,) to compare all valid starts for that horizon.
    """
    predicted = np.asarray(predicted_raw, dtype=float)
    target = np.asarray(target_raw, dtype=float)
    initial = np.asarray(initial_raw, dtype=float)
    if (predicted.ndim != 3 or predicted.shape[2] != 9 or not predicted.shape[0]
            or target.shape != predicted.shape or initial.shape != (predicted.shape[0], 9)):
        raise ValueError('expected aligned nonempty (N,H,9) and (N,9) ball arrays')
    if not all(np.isfinite(values).all() for values in (predicted, target, initial)):
        raise ValueError('ball arrays must be finite')
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('control interval must be positive and finite')
    leads = tuple(leads)
    if (not leads or any(not isinstance(lead, (int, np.integer)) or not 1 <= lead <= predicted.shape[1]
                         for lead in leads) or len(set(leads)) != len(leads)):
        raise ValueError('leads must be distinct integers within the rollout horizon')

    def summarize(mask):
        starting = initial[mask]
        results = {}
        for lead in leads:
            actual = target[mask, lead - 1]
            constant_velocity = starting.copy()
            constant_velocity[:, :3] += (lead * dt) * starting[:, 3:6]
            results[str(lead)] = {
                'windows': len(starting),
                'model': _ball_errors(predicted[mask, lead - 1], actual),
                'persistence': _ball_errors(starting, actual),
                'constant_relative_velocity': _ball_errors(constant_velocity, actual),
            }
        return results

    masks = _episode_masks(episode_ids, len(initial))
    return {'by_lead': summarize(slice(None)),
            'by_episode': {name: summarize(mask) for name, mask in masks.items()}}


def evaluate_termination(probabilities, labels, prior_probability, *, threshold=0.5, episode_ids=None):
    """Score per-control departure probabilities and a supplied train-set prior.

    The event is physical termination within the requested control interval.
    Its target is positive even when the transition ended on a partial interval.
    """
    probability = np.asarray(probabilities, dtype=float)
    label = np.asarray(labels)
    if probability.ndim != 1 or not len(probability) or label.shape != probability.shape:
        raise ValueError('expected aligned nonempty termination vectors')
    if not np.all(np.isfinite(probability) & (probability >= 0) & (probability <= 1)):
        raise ValueError('termination probabilities must be within [0,1]')
    if not np.isin(label, [False, True]).all():
        raise ValueError('termination labels must be binary')
    if not (0 <= prior_probability <= 1 and 0 <= threshold <= 1):
        raise ValueError('prior probability and threshold must be within [0,1]')
    label = label.astype(bool)

    def scores(p, y):
        predicted_positive = p >= threshold
        assigned = np.where(y, p, 1 - p)
        return {'log_loss': float(-np.mean(np.log(np.maximum(assigned, 1e-12)))),
                'brier': float(np.mean((p - y)**2)),
                'true_positive': int(np.count_nonzero(predicted_positive & y)),
                'false_positive': int(np.count_nonzero(predicted_positive & ~y)),
                'true_negative': int(np.count_nonzero(~predicted_positive & ~y)),
                'false_negative': int(np.count_nonzero(~predicted_positive & y))}

    def summarize(mask):
        y = label[mask]
        positives = int(np.count_nonzero(y))
        return {'transitions': len(y), 'positive_count': positives, 'negative_count': len(y) - positives,
                'model': scores(probability[mask], y),
                'prior': scores(np.full(len(y), prior_probability), y)}

    masks = _episode_masks(episode_ids, len(label))
    return {**summarize(slice(None)), 'threshold': float(threshold), 'prior_probability': float(prior_probability),
            'by_episode': {name: summarize(mask) for name, mask in masks.items()},
            'scope': 'Descriptive transition metrics. Report positive counts and compare by episode; '
                     'rare events and within-episode dependence limit interpretation.'}
