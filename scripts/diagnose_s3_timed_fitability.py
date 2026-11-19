"""P0 audit of fixed-endpoint curves and weighted loss gradients."""
import argparse
import csv
from io import BytesIO
from itertools import combinations
from pathlib import Path
from xml.sax.saxutils import escape

from s3_fitability_common import PARTS, ROOT, branch_arrays, forecast, frozen_data, run_phase, write_json

import numpy as np

from ball_retention.action_grad import _inputs
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.deterministic_response_model import _directional_rollout
from ball_retention.model_data import _ball_state
from ball_retention.pair_loss import paired_loss
from ball_retention.position_integral_model import load_position_model
from ball_retention.velocity_response_loss import paired_velocity_response_loss, response_components
from preflight_s3_velocity_response_leads import lead_vectors
from train_s3_timing_coverage import action_sequences, read_held


NAMES = ('original_46', 'trapezoid', 'dense_velocity')
SCALES = (1000, 1000, 1)
UNITS = ('mm', 'mm/s', 'rad/s')
METHODS = ('Free', 'Local', 'Propagation')
COLORS = ('#2166ac', '#b35806', '#1b7837')


def flatten(gradient):
    return np.concatenate([value.ravel() for value in gradient.values()])


def cosine(a, b):
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip((a@b)/norm, -1., 1.)) if norm else None


def saturation(model, initial, features):
    _, caches = model._rollout(initial, features, record=True)
    data = {}
    for i in (1, 2):
        slopes = 1.-np.stack([cache[i] for cache in caches])**2
        data[f'tanh_{i}'] = {'fraction_derivative_le_0_05': float(np.mean(slopes <= .05)),
                            'fraction_derivative_le_0_01': float(np.mean(slopes <= .01)),
                            'fraction_le_0_05_by_lead': np.mean(slopes <= .05, axis=(1, 2)).tolist(),
                            'derivative_p05_median': np.quantile(slopes, [.05, .5]).tolist()}
    return data


def gradient_audit(model, fold, cfg, scales, base, velocity, seed):
    initial, features, truth, masks = branch_arrays(fold)
    initial = model.normalize(initial)
    batch = fold.pairs
    samples = fold.pair_weights
    timed = np.arange(len(initial)) >= 40
    horizon = features.shape[2]
    spec = cfg['pairs']
    a, r = spec['absolute_weight'], spec['response_weight']
    components = np.asarray(cfg['components'])
    units = np.asarray(cfg['response_scales_si'])
    absolute_units = np.asarray(scales['state_error_scale_si'])
    common = (model, initial, features, truth, masks, batch.response_mask, batch.noise_mask, batch.response_floor)

    rng = np.random.default_rng(seed)
    anchors, anchor_losses, anchor_indices = {}, {}, {}
    for h in cfg['anchors']['horizons']:
        windows = fold.anchors[h]
        take = rng.integers(len(windows.initial), size=cfg['anchors']['batch_size'])
        loss, grad = model.loss_and_grad(model.normalize(windows.initial[take]),
            (windows.known[take]-fold.known_mean)/fold.known_scale, model.normalize(windows.target[take]),
            state_weights=components, state_error_scale_si=absolute_units)
        anchors[h], anchor_losses[h], anchor_indices[h] = grad, loss, take.tolist()
    s2 = {key: np.mean([grad[key] for grad in anchors.values()], axis=0) for key in model.params}

    weights = response_components(base, velocity, components, batch.noise_mask, samples)
    effective = batch.noise_mask[..., None]*weights*samples[:, None, None]
    mass = effective.sum()
    absolute, absolute_grad = paired_loss(*common, base, components, units, a, 0.,
                                           sample_weights=samples, absolute_scale_si=absolute_units)
    grads = {'S2': s2, 'absolute': absolute_grad}
    losses = {'S2': float(np.mean(list(anchor_losses.values()))), 'absolute': absolute['loss']}
    for name, subset in (('held_response', ~timed), ('timed_response', timed)):
        # Preserve the shared response denominator across the partition.
        share = effective[subset].sum()/mass
        metrics, grads[name] = paired_loss(*common, np.ones(horizon), weights, units, 0., r*share,
                                     sample_weights=samples*subset, absolute_scale_si=absolute_units)
        losses[name] = metrics['loss']
    actual, grad = paired_velocity_response_loss(*common, base, components, units, a, r,
            velocity_lead_weights=velocity, sample_weights=samples, absolute_scale_si=absolute_units)
    pair_total = sum(flatten(value) for name, value in grads.items() if name != 'S2')
    np.testing.assert_allclose(pair_total, flatten(grad), atol=5e-13, rtol=5e-12)
    np.testing.assert_allclose(sum(value for name, value in losses.items() if name != 'S2'), actual['loss'],
                               atol=5e-13, rtol=5e-12)
    vectors = {key: flatten(value) for key, value in grads.items()}
    combined = sum(vectors.values())
    pair_and_s2_by_horizon = {h: {
        'S2_gradient_norm': float(np.linalg.norm(flatten(value))),
        'cosine_to_each_pair_term': {name: cosine(flatten(value), grad_vec) for name, grad_vec in vectors.items()
                                     if name != 'S2'},
    } for h, value in anchors.items()}

    doubled = np.tile(initial, (2, 1))
    packed = features.reshape(-1, horizon, model.known_dim)
    projected = model.rollout(doubled, packed).reshape(truth.shape)
    prediction = .5*(projected[0]-projected[1])*model.state_scale
    target = .5*(truth[0]-truth[1])
    error = np.where(batch.response_mask, prediction-target,
                     np.sign(prediction)*np.maximum(0., abs(prediction)-batch.response_floor))
    residual = error/units
    tail = {}
    for name, subset in (('held', ~timed), ('timed', timed)):
        tail[name] = {}
        for component, part in PARTS.items():
            block = effective[subset, :, part]
            scaled = residual[subset, :, part]
            resolved = batch.response_mask[subset, :, part]
            eligible = block > 0
            linear_tail = abs(scaled) > 1.
            den = block.sum()
            tail[name][component] = {
                'response_mass_fraction': float(den/mass),
                'h32_mass_fraction_of_component_at_h32': float(block[:, -1].sum()/effective[:, -1, part].sum()),
                'resolved_weight_fraction': float(block[resolved].sum()/den),
                'tail_component_fraction': float(np.mean(linear_tail[eligible])),
                'tail_weight_fraction': float(block[linear_tail].sum()/den),
                'weighted_robust_to_quadratic_slope_norm': float(
                    np.linalg.norm(block*scaled/np.hypot(1., scaled))/np.linalg.norm(block*scaled))
                    if np.any(block*scaled) else None,
            }
    return {'weighted_loss': losses, 'norm': {key: float(np.linalg.norm(value)) for key, value in vectors.items()},
            'parameter_block_norm': {term: {key: float(np.linalg.norm(v)) for key, v in values.items()}
                                     for term, values in grads.items()},
            'cosine': {f'{k}/{j}': cosine(vectors[k], vectors[j]) for k, j in combinations(vectors, 2)},
            'joint_norm': float(np.linalg.norm(combined)),
            'cosine_to_joint': {key: cosine(value, combined) for key, value in vectors.items()},
            'timed_weight_fraction_all_response': float(effective[timed].sum()/mass),
            'timed_weight_fraction_absolute': float(np.sum(masks[:, timed, :, None]*base[:, None]*components
                                                           *samples[timed, None, None])/
                                                     np.sum(masks[..., None]*base[:, None]*components*samples[:, None, None])),
            'loss_gradient_recomposition_max_abs': float(abs(pair_total-flatten(grad)).max()),
            'anchor_indices': anchor_indices, 'anchor_losses': anchor_losses, 'anchor_by_horizon': pair_and_s2_by_horizon,
            'response_robust_tail_abs_scaled_error_gt_1': tail,
            'saturation': {'held': saturation(model, doubled[np.tile(~timed, 2)], packed[np.tile(~timed, 2)]),
                           'timed': saturation(model, doubled[np.tile(timed, 2)], packed[np.tile(timed, 2)])}}


def held_inputs(held, model, fold):
    capture, cfg, fields, traces, _ = held
    state = _parse_state(fields)
    initial = _ball_state(state)
    nominal = traces['nominal_1']['actions']
    directions, requests = action_sequences(capture, nominal)
    args = (model, initial, state.time_s, state.tilt_rad, state.tilt_rate_rad_s)
    tail = (cfg, fold.known_mean, fold.known_scale)
    features, targets = {}, {}
    for label, trace in traces.items():
        np.testing.assert_array_equal(trace['actions'], requests[label])
        features[label] = _inputs(*args, trace['actions'], *tail)[2]
        targets[label] = trace['raw']
    start, raw, exogenous = _inputs(*args, nominal, *tail)
    _, slope = _directional_rollout(model, start, raw, exogenous, nominal, np.asarray(list(directions.values())),
                                    cfg, fold.known_scale)
    pairs = [('nominal_1', 'nominal_2')]+[(f'{key}_plus', f'{key}_minus') for key in directions]
    jvp_effect = np.concatenate((np.zeros_like(slope[:1]), capture['amplitude']*slope))
    return (np.broadcast_to(initial, (len(pairs), 9)),
            np.stack([[features[labels[b]] for labels in pairs] for b in (0, 1)]),
            np.stack([[targets[labels[b]] for labels in pairs] for b in (0, 1)]), jvp_effect,
            ['nominal_replays', *directions])


def rollout_pair(model, initial, features, truth):
    flat = (np.tile(initial, (2, 1)), features.reshape(-1, *features.shape[2:]), truth.reshape(-1, *truth.shape[2:]))
    free = forecast(model, *flat).reshape(truth.shape)
    local = forecast(model, *flat, local=True).reshape(truth.shape)
    np.testing.assert_allclose(free[:, :, 0], local[:, :, 0], atol=1e-12, rtol=0.)
    if not all(np.isfinite(data).all() for data in (free, local, truth)):
        raise FloatingPointError('nonfinite endpoint curve')
    return free, local


def curve_arrays(truth, free, local):
    half = lambda q: .5*(q[0]-q[1])
    for label, vectors in zip(('plus', 'minus', 'half_difference'),
                               (*zip(truth, free, local, strict=True), tuple(map(half, (truth, free, local)))),
                               strict=True):
        target, predicted, reset = vectors
        errors = (predicted-target, reset-target, predicted-reset)
        np.testing.assert_allclose(errors[0], errors[1]+errors[2], atol=1e-12, rtol=0.)
        yield label, target, predicted, reset, errors


def record_curves(writer, scope, endpoint, identifiers, mask, truth, free, local, jvp=None):
    margins = {}
    for label, target, predicted, reset, errors in curve_arrays(truth, free, local):
        valid = {'plus': mask[0], 'minus': mask[1], 'half_difference': mask.all(axis=0)}[label]
        for component, part in PARTS.items():
            spans = [data[..., part] for data in (target, predicted, reset)]
            norms = [np.linalg.norm(value, axis=-1) for value in spans]
            err = np.stack([np.linalg.norm(data[..., part], axis=-1) for data in errors], axis=-1)
            margins[label, component] = err
            for row, lead in np.argwhere(valid):
                is_half = label == 'half_difference'
                direction = [cosine(spans[0][row, lead], data[row, lead]) for data in spans[1:]] if is_half else [None]*2
                ratio = [data[row, lead]/norms[0][row, lead] for data in norms[1:]] if is_half and norms[0][row, lead] else [None]*2
                slope_error = (float(np.linalg.norm(jvp[row, lead, part]-spans[0][row, lead]))
                               if jvp is not None and is_half and row > 0 else None)
                magnitude = [data[row, lead] for data in norms] if is_half else [None]*3
                writer.writerow([scope, endpoint, identifiers[row], lead+1, label, component,
                                 *err[row, lead], *magnitude,
                                 *direction, *ratio, slope_error])
    return margins


def svg(path, title, names, series):
    """Draw a small multiple with a common vertical scale per panel."""
    width, cell_w, cell_h, top = 1080, 350, 126, 72
    height = top+len(names)*cell_h+20
    tags = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>']
    def label(x, y, value, size=13, **style):
        options = ' '.join(f'{key.replace("_", "-")}="{val}"' for key, val in style.items())
        tags.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" {options}>{escape(value)}</text>')
    label(18, 27, title, 20, font_weight='bold')
    for i, method in enumerate(METHODS):
        x = 18+180*i
        tags.append(f'<path d="M{x},{48}h24" stroke="{COLORS[i]}" stroke-width="2"/>')
        label(x+31, 53, method)
    for i, name in enumerate(names):
        for j, (component, unit, scale) in enumerate(zip(PARTS, UNITS, SCALES, strict=True)):
            x, y = 42+j*cell_w, top+i*cell_h
            plot_w, plot_h = 285, 70
            values = scale*series[i, j]
            limit = max(float(values.max())*1.05, 1e-6)
            label(x, y+12, f'{name} / {component} ({unit})', font_weight='bold')
            tags.append(f'<path d="M{x},{y+19}v{plot_h}h{plot_w}" fill="none" stroke="#82919c"/>')
            label(x-3, y+25, f'{limit:.3g}', 10, text_anchor='end')
            label(x-3, y+92, '0', 10, text_anchor='end')
            for lead in (1, 8, 16, 24, 32):
                lx = x+(lead-1)*plot_w/31
                label(lx, y+105, str(lead), 10, text_anchor='middle')
            for k, color in enumerate(COLORS):
                points = ' '.join(f'{x+t*plot_w/31:.2f},{y+89-v/limit*plot_h:.2f}'
                                  for t, v in enumerate(values[:, k]))
                tags.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.7"/>')
    path.write_text('\n'.join([*tags, '</svg>'])+'\n', encoding='utf-8')


def run(plan, out, guard):
    design, parent, cfg, scales, fold, rows, matched, dense = frozen_data(plan, guard)
    source = design['nominal_gate_reference']
    with np.load(BytesIO(_verified_bytes(ROOT, source, plan['files_sha256'][source])), allow_pickle=False) as saved:
        original = load_position_model(saved)
        for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold, key))
    held = read_held(parent)
    names = [f'{row["group"]} / {row["offset_pattern"]}' for row in rows[40:]]
    train = branch_arrays(fold)
    base, dense_velocity = lead_vectors(design, 32)
    audits = {}
    with (out/'curves.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scope', 'checkpoint', 'case', 'lead', 'series', 'component',
                         'free_error_si', 'local_error_si', 'propagation_norm_si',
                         'newton_half_norm_si', 'free_half_norm_si', 'local_half_norm_si',
                         'free_cosine', 'local_cosine', 'free_magnitude_ratio', 'local_magnitude_ratio', 'jvp_error_si'])
        for name, model in zip(NAMES, (original, matched, dense), strict=True):
            guard('gradients', log=True)
            audits[name] = gradient_audit(model, fold, cfg, scales, base, dense_velocity if model is dense else base,
                                          plan['seed'])
            guard('curves', log=True)
            initial, features, truth, mask = train
            free, local = rollout_pair(model, initial, features, truth)
            errors = record_curves(writer, 'training', name, range(len(rows)), mask, truth, free, local)
            i, k, y, jvp, labels = held_inputs(held, model, fold)
            f, l = rollout_pair(model, i, k, y)
            dev = record_curves(writer, 'development', name, labels, np.ones(y.shape[:-1], bool), y, f, l, jvp)
            if model is dense:
                absolute = np.stack([np.sqrt(.5*(errors['plus', part][40:]**2+errors['minus', part][40:]**2))
                                     for part in PARTS], axis=1)
                svg(out/'timed_absolute.svg', 'Dense velocity endpoint | branch state errors (pair RMS)', names, absolute)
                svg(out/'timed_response.svg', 'Dense velocity endpoint | half-difference errors', names,
                    np.stack([errors['half_difference', part][40:] for part in PARTS], axis=1))
                svg(out/'development_response.svg', 'Dense velocity endpoint | frozen development half-difference errors', labels[1:],
                    np.stack([dev['half_difference', part][1:] for part in PARTS], axis=1))
            stream.flush()
            guard('curves', log=True)
    write_json(out/'gradients.json', {'checkpoints': audits, 'pair_identity': rows,
               'anchor_estimator': 'Mean gradient of one seeded training batch per H1/H4/H8. Pair terms use all rows.'})
    return {'training_pairs': len(rows), 'fixed_endpoints': NAMES}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(ROOT/'configs/s3_fitability_p0_plan.json', args.plan_sha256, 'P0', run)
