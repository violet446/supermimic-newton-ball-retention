"""Align recorded broad contact evidence without importing Newton."""
from collections import Counter
from math import isclose, isfinite


def _allowed_pairs(layout):
    labels = layout['shape_labels']
    expected = layout['expected_collision_shapes']
    if (len(set(labels)) != len(labels) or any(not isinstance(label, str) or not label for label in labels)
            or set(expected) != {'ball', 'shovel', 'floor'}
            or any(type(index) is not int or not 0 <= index < len(labels) or labels[index] != name
                   for name, index in expected.items())):
        raise ValueError('invalid saved shape IDs or labels')
    return {tuple(sorted((expected['ball'], expected[other]))) for other in ('shovel', 'floor')}


def _decode_trace(trace, label, horizon, substeps, origin, dt, allowed):
    elapsed = trace['elapsed_s']
    states = trace['states']
    records = trace['contacts']
    if (trace['substeps'] != [substeps]*horizon or len(elapsed) != horizon or len(states) != horizon
            or len(records) != horizon*substeps):
        raise ValueError(f'{label}: contact record does not cover all frozen leads and substeps')
    for lead, (age, state) in enumerate(zip(elapsed, states), 1):
        physical_offset = lead*substeps
        if (not isclose(age, physical_offset*dt, rel_tol=0., abs_tol=1e-9)
                or not isclose(state['time_s'], (origin+physical_offset)*dt, rel_tol=0., abs_tol=1e-9)):
            raise ValueError(f'{label}: contact and public state clocks disagree at lead {lead}')

    decoded = []
    for offset, record in enumerate(records, 1):
        step = record['physics_step']
        count = record['candidates']
        if type(step) is not int or step != origin+offset or type(count) is not int or count < 0:
            raise ValueError(f'{label}: invalid contact chronology or count at substep {offset}')
        pairs = tuple(tuple(pair) for pair in record['shape_pairs'])
        if (any(len(pair) != 2 or any(type(index) is not int for index in pair) or pair not in allowed
                for pair in pairs) or tuple(sorted(set(pairs))) != pairs
                or len(pairs) > count or bool(pairs) != bool(count)):
            raise ValueError(f'{label}: invalid distinct contact shape pairs at substep {offset}')
        decoded.append((count, pairs))
    return decoded


def _timeline(index, origin, substeps, dt):
    return {'lead': index//substeps+1, 'substep_in_lead': index%substeps+1,
            'physics_step': origin+index+1, 'elapsed_s': (index+1)*dt,
            'ball_state_elapsed_before_step_s': index*dt}


def _describe_branch(records, substeps, dt):
    leads = []
    overall_sets = Counter(pairs for _, pairs in records)
    for index in range(0, len(records), substeps):
        block = records[index:index+substeps]
        counts = [count for count, _ in block]
        sets = Counter(pairs for _, pairs in block)
        leads.append({'lead': index//substeps+1, 'elapsed_s': (index+substeps)*dt,
                      'candidates_by_substep': counts, 'candidate_range': [min(counts), max(counts)],
                      'shape_pair_sets': [{'pairs': pairs, 'substeps': n} for pairs, n in sorted(sets.items())]})
    return {'candidate_range': [min(count for count, _ in records), max(count for count, _ in records)],
            'shape_pair_sets': [{'pairs': pairs, 'substeps': n} for pairs, n in sorted(overall_sets.items())],
            'leads': leads}


def _compare(first_label, second_label, decoded, origin, substeps, dt):
    first, second = decoded[first_label], decoded[second_label]
    counts = [a[0]-b[0] for a, b in zip(first, second, strict=True)]
    shapes = [a[1] != b[1] for a, b in zip(first, second, strict=True)]

    def first_difference(values, field, name):
        index = next((i for i, value in enumerate(values) if value), None)
        if index is None:
            return None
        return {**_timeline(index, origin, substeps, dt),
                f'first_{name}': first[index][field], f'second_{name}': second[index][field]}

    leads = []
    for index in range(0, len(first), substeps):
        delta = counts[index:index+substeps]
        leads.append({'lead': index//substeps+1, 'elapsed_s': (index+substeps)*dt,
                      'candidate_count_different_substeps': sum(value != 0 for value in delta),
                      'max_abs_candidate_count_delta': max(map(abs, delta)),
                      'shape_set_different_substeps': sum(shapes[index:index+substeps])})
    return {'first_branch': first_label, 'second_branch': second_label,
            'first_candidate_count_difference': first_difference(counts, 0, 'candidate_count'),
            'first_shape_pair_set_difference': first_difference(shapes, 1, 'shape_pairs'),
            'candidate_count_different_substeps': sum(value != 0 for value in counts),
            'shape_set_different_substeps': sum(shapes), 'leads': leads}


def diagnose_contacts(plan, traces, layout):
    """Compare candidate counts and distinct shape pairs per control lead."""
    labels = plan['branch_order']
    horizon, substeps = plan['horizon'], plan['control_substeps']
    origin = plan['case']['tick']*substeps
    if (set(traces) != set(labels) or labels[:2] != ['nominal_1', 'nominal_2']
            or len(labels) != len(set(labels))):
        raise ValueError('contact traces do not match frozen branch labels')
    # Public endpoint time uses the absolute integer physics counter. Cross-check
    # the derived period against every recorded public and elapsed clock.
    dt = traces['nominal_1']['states'][-1]['time_s']/(origin+horizon*substeps)
    if not isfinite(dt) or dt <= 0:
        raise ValueError('invalid physics step period in nominal replay')
    allowed = _allowed_pairs(layout)
    decoded = {label: _decode_trace(traces[label], label, horizon, substeps, origin, dt, allowed) for label in labels}

    def compare(first, second):
        return _compare(first, second, decoded, origin, substeps, dt)

    comparisons = {}
    for direction in plan['directions']:
        plus, minus = f'{direction}_plus', f'{direction}_minus'
        comparisons[direction] = {'plus_vs_minus': compare(plus, minus),
                                  'plus_vs_nominal': compare(plus, 'nominal_1'),
                                  'minus_vs_nominal': compare(minus, 'nominal_1')}
    return {'scope': ('Counts measure collision candidates before integration. Shape pairs list distinct shape IDs. '
                      'The trace contains no contact points, normals, impulses, forces or active constraints.'),
            'physics_dt_s': dt, 'snapshot_physics_step': origin, 'shape_labels': layout['shape_labels'],
            'nominal_comparison': compare('nominal_1', 'nominal_2'),
            'branches': {label: _describe_branch(decoded[label], substeps, dt) for label in labels},
            'comparisons': comparisons}
