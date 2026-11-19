"""Attribute the fixed broad-data pair objective by its training rows."""
from collections import defaultdict

import numpy as np

from ball_retention.velocity_response_loss import paired_velocity_response_loss, response_components
from ball_retention.world_increment_loss import world_increment_loss
from preflight_s3_world_increment_supervision import add, dot, mul, norm
from train_s3_world_increment import packed


ABSOLUTE = 'absolute_rollout'
RESPONSE = 'paired_response'
NOMINAL = 'nominal_world_increment'
INCREMENT = 'paired_world_increment'
TERMS = (ABSOLUTE, RESPONSE, NOMINAL, INCREMENT)


def _rows(keys):
    result = defaultdict(list)
    for i, key in enumerate(keys):
        result[key].append(i)
    return {key: np.asarray(rows) for key, rows in result.items()}


def _add_parts(parts, zero):
    if not parts:
        return 0., 0., zero
    return (sum(v[0] for v in parts), sum(v[1] for v in parts),
            add(*(v[2] for v in parts)))


def _rollout_terms(model, fold, cfg, scales, leads, model_cfg, anchor, cells, guard):
    batch = fold.pairs
    initial = model.normalize(batch.initial)
    known = packed(model, np.stack((batch.plus_known, batch.minus_known)), model_cfg, anchor, fold)
    target = np.stack((batch.plus_target, batch.minus_target))
    mask = np.stack((batch.plus_mask, batch.minus_mask))
    branches, count, horizon, known_dim = known.shape
    predicted = model.rollout(np.concatenate((initial, initial)),
                              known.reshape(branches*count, horizon, known_dim)).reshape(target.shape)
    base, velocity = leads
    components = np.asarray(cfg['components'])
    absolute_units = np.asarray(scales['state_error_scale_si'])
    response_units = np.asarray(cfg['response_scales_si'])
    samples = fold.pair_weights
    absolute_coefficient = cfg['pairs']['absolute_weight']
    response_coefficient = cfg['pairs']['response_weight']
    base_components = base[:, None]*components
    response_weights = (base_components if np.array_equal(base, velocity) else
                        response_components(base, velocity, components, batch.noise_mask, samples))

    # Keep the full-fold denominator and the full-fold velocity redistribution.
    # A subset training loss would normalize both again and change the objective.
    abs_weights = mask[..., None]*base_components*samples[None, :, None, None]
    resp_weights = batch.noise_mask[..., None]*response_weights*samples[:, None, None]
    abs_mass, resp_mass = float(abs_weights.sum()), float(resp_weights.sum())
    if min(abs_mass, resp_mass) <= 0.:
        raise ValueError('missing globally eligible paired supervision')
    abs_weights /= abs_mass
    resp_weights /= resp_mass
    factor = model.state_error_factor(absolute_units)
    absolute_error = (predicted-model.normalize(target))*factor
    half_spread = (predicted[0]-predicted[1])*model.state_scale*.5
    target_spread = (target[0]-target[1])*.5
    unresolved_excess = np.maximum(abs(half_spread)-batch.response_floor, 0.)
    residual = np.where(batch.response_mask, half_spread-target_spread,
                        np.sign(half_spread)*unresolved_excess)/response_units
    robust = np.hypot(1., residual)

    # The source losses have no per-row attribution interface. Reconstruct their
    # per-output derivatives, then check against the unmodified objective below.
    abs_values = absolute_coefficient*abs_weights*absolute_error**2
    resp_values = response_coefficient*resp_weights*2.*(robust-1.)
    abs_cotangent = (2.*absolute_coefficient*abs_weights)*absolute_error*factor
    signed = response_coefficient*resp_weights*residual/robust*model.state_scale/response_units
    resp_cotangent = np.stack((signed, -signed))
    zero = {key: np.zeros_like(value) for key, value in model.params.items()}

    def gradient(inputs, features, derivatives, mass):
        if mass == 0.:
            return zero
        return model.rollout_vjp(inputs, features, derivatives)[2]

    parts = {}
    for key, rows in cells.items():
        size = branches*len(rows)
        inputs = np.concatenate((initial[rows], initial[rows]))
        features = known[:, rows].reshape(size, horizon, known_dim)
        absolute_mass = float(abs_weights[:, rows].sum())
        response_mass = float(resp_weights[rows].sum())
        parts[key] = {
            ABSOLUTE: (float(abs_values[:, rows].sum()), absolute_mass,
                       gradient(inputs, features, abs_cotangent[:, rows].reshape(size, horizon, -1), absolute_mass)),
            RESPONSE: (float(resp_values[rows].sum()), response_mass,
                       gradient(inputs, features, resp_cotangent[:, rows].reshape(size, horizon, -1), response_mass)),
        }
        guard('training_rollout_partition')

    metrics, expected = paired_velocity_response_loss(
        model, initial, known, target, mask, batch.response_mask, batch.noise_mask, batch.response_floor,
        base, components, response_units, absolute_coefficient, response_coefficient,
        velocity_lead_weights=velocity, sample_weights=samples, absolute_scale_si=absolute_units)
    return parts, metrics, expected, {ABSOLUTE: abs_mass, RESPONSE: resp_mass}


def _increment_terms(model, data, keys, coefficient, paired, guard):
    states, known, target, weights, scale = data
    if len(keys) != len(weights):
        raise ValueError('increment archive metadata does not match its target order')
    parts = {}
    for key, rows in _rows(keys).items():
        mass = float(weights[rows].sum())
        if mass == 0.:
            continue
        take = (slice(None), rows) if paired else rows
        value, gradient, _ = world_increment_loss(model, states[take], known[take], target[take],
                                                   weights[rows]/mass, scale, paired=paired)
        parts[key] = (coefficient*mass*value, mass, mul(gradient, coefficient*mass))
        guard('training_increment_partition')
    value, gradient, _ = world_increment_loss(model, *data, paired=paired)
    return parts, coefficient*value, mul(gradient, coefficient)


def _check_partition(actual, expected, label):
    if not all(np.isfinite(values).all() for values in (actual, expected)):
        raise FloatingPointError(f'nonfinite gradient attribution in {label}')
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13, err_msg=label)
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))


def attribute(model, fold, aux, coefficients, cfg, scales, leads, model_cfg, anchor,
              amplitudes, nominal_groups, guard):
    """Return the paired objective attribution and its full parameter gradient.

    Snapshot tick is the absolute start of a paired rollout, not a loss lead.
    Signed increment labels follow the paired archive suffix. Nominal labels
    provide a separate (group, snapshot tick) entry for each recorded replay.
    """
    batch = fold.pairs
    count = len(batch.initial)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.shape != (count,) or not np.all(np.isfinite(amplitudes) & (amplitudes > 0.)):
        raise ValueError('expected finite positive amplitude metadata for every pair')
    pair_keys = [(str(group), int(tick), float(amplitude))
                 for group, tick, amplitude in zip(batch.group_ids, batch.snapshot_ticks, amplitudes, strict=True)]
    nominal_keys = [(str(group), int(tick), None) for group, tick in nominal_groups]
    if {(g, t) for g, t, _ in nominal_keys}-{(g, t) for g, t, _ in pair_keys}:
        raise ValueError('nominal training metadata has no matching group and snapshot')
    n_signed = len(aux['half_difference'][3])
    if not 0 < n_signed <= count:
        raise ValueError('invalid fully recorded pair archive suffix')

    cells = _rows(pair_keys)
    parts, metrics, paired_grad, global_mass = _rollout_terms(
        model, fold, cfg, scales, leads, model_cfg, anchor, cells, guard)
    for term, name, keys, paired in (
            (NOMINAL, 'nominal', nominal_keys, False),
            (INCREMENT, 'half_difference', pair_keys[count-n_signed:], True)):
        terms, value, gradient = _increment_terms(
            model, aux[name], keys, coefficients[name], paired, guard)
        for cell, part in terms.items():
            parts.setdefault(cell, {})[term] = part
        if term == NOMINAL:
            nominal_value, nominal_grad = value, gradient
        else:
            signed_value, signed_grad = value, gradient

    zero = {key: np.zeros_like(value) for key, value in model.params.items()}
    expected = add(paired_grad, nominal_grad, signed_grad)
    by_term = {term: _add_parts([p[term] for p in parts.values() if term in p], zero) for term in TERMS}
    term_gradients = {key: val[2] for key, val in by_term.items()}
    total = add(*term_gradients.values())
    term_values = [by_term[term][0] for term in TERMS]
    expected_values = [cfg['pairs']['absolute_weight']*metrics['absolute'],
                       cfg['pairs']['response_weight']*metrics['response'], nominal_value, signed_value]
    loss_difference = _check_partition(term_values, expected_values, 'weighted per-term loss')
    _check_partition([by_term[term][1] for term in TERMS], np.ones(len(TERMS)), 'effective weight fractions')
    gradient_difference = max(_check_partition(total[key], expected[key], key) for key in total)
    for term, gradient in ((NOMINAL, nominal_grad), (INCREMENT, signed_grad)):
        gradient_difference = max(gradient_difference, *(
            _check_partition(by_term[term][2][key], gradient[key], f'{term}/{key}') for key in gradient))

    def describe(vector, ref):
        size, ref_size = norm(vector), norm(ref)
        return {'signed_projection_fraction': dot(vector, ref)/ref_size**2 if ref_size > 0. else None,
                'cosine': dot(vector, ref)/(size*ref_size) if size > 0. and ref_size > 0. else None}

    def describe_term(part):
        value, mass, gradient = part
        return {'weighted_loss': value, 'effective_weight_fraction': mass, 'gradient_norm': norm(gradient),
                'projection_on_pair_objective': describe(gradient, total)}

    nominal_cells = _rows(nominal_keys)

    def reduce(keys):
        reductions = {term: _add_parts([parts[key][term] for key in keys if term in parts[key]], zero)
                      for term in TERMS}
        vector = add(*(value[2] for value in reductions.values()))
        return {'pair_rows': sum(len(cells[key]) for key in keys if key in cells),
                'nominal_rollouts': sum(len(nominal_cells[key]) for key in keys if key in nominal_cells),
                'pair_weight_mass': sum(float(fold.pair_weights[cells[key]].sum()) for key in keys if key in cells),
                'weighted_loss': sum(value[0] for value in reductions.values()),
                'gradient_norm': norm(vector), 'projection_on_pair_objective': describe(vector, total),
                'projection_on_component': {term: describe(vector, term_gradients[term]) for term in TERMS},
                'objective_terms': {term: describe_term(value) for term, value in reductions.items()}}

    by_group, by_tick, by_amplitude = defaultdict(list), defaultdict(list), defaultdict(list)
    by_cell = {}
    for cell in parts:
        group, tick, amplitude = cell
        label = 'nominal' if amplitude is None else f'{amplitude:g}'
        by_group[group].append(cell)
        by_tick[str(tick)].append(cell)
        by_amplitude[label].append(cell)
        by_cell[f'{group}/{tick}/{label}'] = reduce([cell])
    report = {'scope': 'Raw gradient of the S3 paired objective at fixed parameters. The separate S2 batch and Adam are excluded.',
              'total_loss': sum(term_values), 'gradient_norm': norm(total),
              'objective_terms': {term: describe_term(value) for term, value in by_term.items()},
              'closure': {'max_parameter_abs_error': gradient_difference, 'max_loss_abs_error': loss_difference},
              'raw_rollout_weight_mass': global_mass,
              'partitions': {'by_group': {key: reduce(group) for key, group in by_group.items()},
                             'by_snapshot_tick': {key: reduce(group) for key, group in by_tick.items()},
                             'by_amplitude': {key: reduce(group) for key, group in by_amplitude.items()},
                             'by_cell': by_cell}}
    return report, total
