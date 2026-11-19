"""Summarize position-response diagnostics on a fixed development set."""
from collections import Counter, defaultdict
from statistics import median


PAIR_COMPARISONS = ('jacobian_vs_newton', 'secant_vs_newton', 'jacobian_vs_secant')
SIGNAL_LEVELS = ('strong', 'near', 'unresolved', 'unavailable')


def _median_present(values):
    present = [float(value) for value in values if value is not None]
    return float(median(present)) if present else None


def _comparison(rows, key):
    values = [row['position'].get(key) or {} for row in rows]
    cosines = [float(value['cosine']) for value in values if value.get('cosine') is not None]
    return {
        'comparable': len(cosines),
        'unavailable': len(rows)-len(cosines),
        'positive_cosine': sum(cosine > 0 for cosine in cosines),
        'cosine_ge_0_5': sum(cosine >= .5 for cosine in cosines),
        'median_cosine': _median_present(cosines),
        'median_amplitude_ratio': _median_present(value.get('amplitude_ratio') for value in values),
        'median_relative_error': _median_present(value.get('relative_error') for value in values),
    }


def _threshold_transitions(rows, threshold, inclusive):
    result = {'both_pass': 0, 'jacobian_only': 0, 'secant_only': 0, 'neither_pass': 0,
              'unavailable': 0}
    for row in rows:
        position = row['position']
        jac = (position.get('jacobian_vs_newton') or {}).get('cosine')
        secant = (position.get('secant_vs_newton') or {}).get('cosine')
        if jac is None or secant is None:
            result['unavailable'] += 1
            continue
        jac_pass = jac >= threshold if inclusive else jac > threshold
        secant_pass = secant >= threshold if inclusive else secant > threshold
        key = {
            (True, True): 'both_pass', (True, False): 'jacobian_only',
            (False, True): 'secant_only', (False, False): 'neither_pass',
        }[jac_pass, secant_pass]
        result[key] += 1
    return result


def _snapshot_count(rows):
    return len({(row['group'], row['episode'], row['tick']) for row in rows})


def _single_summary(rows):
    counts = Counter(row['position']['signal'] for row in rows)
    strong = [row for row in rows if row['position']['signal'] == 'strong']
    strong_metrics = {key: _comparison(strong, key) for key in PAIR_COMPARISONS}
    strong_metrics['curvature_to_newton_norm_ratio'] = _median_present(
        row['position'].get('curvature_to_newton_norm_ratio') for row in strong)
    strong_metrics['threshold_transitions'] = {
        'positive_cosine': _threshold_transitions(strong, 0., inclusive=False),
        'cosine_ge_0_5': _threshold_transitions(strong, .5, inclusive=True),
    }
    return {
        'cases': len(rows),
        'snapshots': _snapshot_count(rows),
        'signal_counts': {level: int(counts[level]) for level in dict.fromkeys((*SIGNAL_LEVELS, *counts))},
        'strong': {'cases': len(strong), 'snapshots': _snapshot_count(strong), **strong_metrics},
    }


def _scale_summary(rows):
    result = {'cases': len(rows), 'snapshots': _snapshot_count(rows)}
    for key in ('newton', 'model_secant'):
        values = [row.get(key) or {} for row in rows]
        eligible = [value for value in values if value.get('stable') is not None]
        result[key] = {
            'eligible': len(eligible),
            'unavailable': len(rows)-len(eligible),
            'stable': sum(bool(value['stable']) for value in eligible),
            'median_cosine': _median_present(value.get('cosine') for value in values),
            'median_relative_error': _median_present(value.get('relative_error') for value in values),
        }
    transitions = Counter()
    for row in rows:
        physical = (row.get('newton') or {}).get('stable')
        model = (row.get('model_secant') or {}).get('stable')
        if physical is None or model is None:
            transitions['unavailable'] += 1
            continue
        key = {
            (True, True): 'both_stable', (True, False): 'newton_only',
            (False, True): 'model_only', (False, False): 'neither_stable',
        }[bool(physical), bool(model)]
        transitions[key] += 1
    result['stability_transitions'] = {key: transitions[key] for key in (
        'both_stable', 'newton_only', 'model_only', 'neither_stable', 'unavailable')}
    return result


def _dual_summary(rows):
    return {'all_cases': _scale_summary(rows),
            'both_newton_strong': _scale_summary([row for row in rows if row['both_newton_strong']])}


def _partitions(rows, fields, summarize_partition):
    result = {'all': summarize_partition(rows)}
    for name, columns in fields:
        partitions = defaultdict(list)
        for row in rows:
            label = '/'.join(str(row[column]) for column in columns)
            partitions[label].append(row)
        result[name] = {key: summarize_partition(value) for key, value in sorted(partitions.items())}
    return result


def summarize(rows, dual_rows):
    """Group fixed development cases by model, lead and physical provenance."""
    cases = defaultdict(list)
    scale_pairs = defaultdict(list)
    for row in rows:
        cases[row['model'], row['lead']].append(row)
    for row in dual_rows:
        scale_pairs[row['model'], row['lead']].append(row)
    if not scale_pairs.keys() <= cases.keys():
        raise ValueError('a dual-amplitude result has no corresponding single-amplitude result')

    by_model = {}
    shared_fields = [(name, (name,)) for name in ('cohort', 'history', 'axis')]
    for (model, lead), entries in sorted(cases.items()):
        summary = _partitions(entries, [('group_amplitude', ('group', 'amplitude')), *shared_fields],
                              _single_summary)
        summary['dual_amplitude'] = _partitions(scale_pairs[model, lead],
                                               [('group', ('group',)), *shared_fields], _dual_summary)
        by_model.setdefault(model, {})[str(lead)] = summary
    return {
        'by_model': by_model,
        'interpretation': [
            'Strong-set counts use the same Newton-signal cases for the local Jacobian and finite secant.',
            'Axes and amplitudes can share snapshots and are correlated observations.',
            'A Jacobian and its finite secant can differ because of finite perturbations along the request axis.',
            'The local-to-secant and secant-to-Newton error vectors can reinforce or cancel. Their norm ratios are not additive.',
            'Newton consistency at two finite amplitudes supports that specific interval. The infinitesimal physical derivative requires separate evidence.',
        ],
    }
