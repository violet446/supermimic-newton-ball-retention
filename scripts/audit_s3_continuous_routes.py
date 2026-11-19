"""Decompose fixed route-fit errors and inspect the frozen training gradients."""
import argparse
from io import BytesIO
import json
from pathlib import Path

from s3_fitability_common import BUDGET, ROOT, SCOPE, run_phase, write_json
import numpy as np

from audit_s3_ball_velocity_components import frame_series
from audit_s3_world_increment_drift import WINDOWS, branch_world_terms, require_close
from audit_s3_world_increment_generation import propagate, world_head
from audit_s3_world_increment_newton_reference import propagate_discrete_position, required_residual, tool_transition
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.model_data import KNOWN_NAMES, _ball_state, known_rollout
from ball_retention.velocity_response_loss import paired_velocity_response_loss
from ball_retention.world_increment_model import WorldIncrementMLP
from check_s3_actions import digest
from deterministic_archive_metrics import _full_branch
from preflight_s3_world_increment_supervision import add, dot, mul, norm
from train_s3_spin_supervision_window import SMALL_CAPTURE, accepted_artifact, score_small, small_inputs
from train_s3_world_increment import packed, score_candidate
from train_s3_world_increment_supervision import losses
import train_s3_continuous_routes as fit


PHASE = 'continuous_route_fit_audit'
PLAN = 'configs/s3_continuous_route_fit_audit_20261119.json'
OUTPUT = 'runs/s3_continuous_route_fit_audit_20261119'
COMPARISON = 'runs/s3_continuous_route_fit_comparison_20261119/summary.json'
FIT6 = 'runs/s3_continuous_route_fit_6_20261119'
SCORE6 = 'runs/s3_continuous_route_fit_6_score_20261119'
FIT12 = 'runs/s3_continuous_route_fit_12_20261119'
FIT_PLANS = {
    '6': ('configs/s3_continuous_route_fit_6_score_20261119.json',
          '29e479e09ff36834f5f47993117b4432771dbf401cf3b8fdf583306eb5193a9f'),
    '12': ('configs/s3_continuous_route_fit_12_verified_20261119.json',
           'a1710fd60e8ea77b29356177ea11a4db809ac36e2bccad34c2ebbdc7fd833122'),
}
SUMMARY_SHA = '48232ed34cb5d8e0b3354d10015aefc1e0b3d554ff1e41e2e06b7dc6103bada9'
BRANCHES = ('nominal_1', 'nominal_2', 'pitch_reversal_plus', 'pitch_reversal_minus',
            'roll_component_plus', 'roll_component_minus')
TRACKS = {'nominal': [.5, .5, 0., 0., 0., 0.], 'pitch_reversal': [0., 0., .5, -.5, 0., 0.],
          'roll_component': [0., 0., 0., 0., .5, -.5]}
REPORT_WINDOWS = (*WINDOWS, (1, 16), (17, 32), (1, 32))


def freeze():
    summary = json.loads(_verified_bytes(ROOT, COMPARISON, SUMMARY_SHA))
    inputs = {COMPARISON: SUMMARY_SHA}
    for folder, value in summary['acceptance_sha256'].items():
        path = f'{folder}/validation.json'
        accepted = json.loads(_verified_bytes(ROOT, path, value))
        inputs[path] = value
        for name in ('comparison.json', 'timed_rollouts.json', 'opening.json', 'execution_plan.json'):
            if name in accepted['artifacts_sha256']:
                inputs[f'{folder}/{name}'] = accepted['artifacts_sha256'][name]
    for key, (path, value) in FIT_PLANS.items():
        candidate = json.loads(_verified_bytes(ROOT, path, value))
        score_dir = SCORE6 if key == '6' else FIT12
        receipt_path = f'{score_dir}/execution_plan.json'
        receipt = json.loads(_verified_bytes(ROOT, receipt_path, inputs[receipt_path]))
        if receipt != {'plan_sha256': value, 'plan': candidate}:
            raise ValueError(f'{key}-episode fit receipt changed')
        for file, sha in candidate['files_sha256'].items():
            if inputs.setdefault(file, sha) != sha:
                raise ValueError(f'inconsistent accepted hash for {file}')
        inputs[path] = value
    for key, folder in (('61_pairs', fit.REFERENCE_RUN), ('6', FIT6), ('12', FIT12)):
        path = f'{folder}/{fit.ENDPOINT}'
        expected = summary['checkpoints_sha256'][key]
        if inputs.setdefault(path, expected) != expected:
            raise ValueError(f'checkpoint {key} differs from result summary')
    helpers = ('scripts/audit_s3_continuous_routes.py', 'scripts/audit_s3_world_increment_drift.py',
               'scripts/audit_s3_world_increment_generation.py', 'scripts/audit_s3_world_increment_newton_reference.py')
    for path in helpers:
        if inputs.setdefault(path, digest(ROOT/path)) != digest(ROOT/path):
            raise ValueError(f'audit helper changed: {path}')
    with (ROOT/PLAN).open('x', encoding='utf-8') as stream:
        json.dump({'phase': PHASE, 'output': OUTPUT, 'seed': 20261111, 'budget': BUDGET, 'scope': SCOPE,
                   'updates': 0, 'endpoint_updates': fit.UPDATES, 'training_draws': fit.UPDATES,
                   'branches': BRANCHES, 'files_sha256': inputs}, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load_endpoint(plan, key, fold, sampler_sha):
    fit_dir, score_dir = (FIT6, SCORE6) if key == '6' else (FIT12, FIT12)
    path = f'{score_dir}/validation.json'
    state = fit.read(plan, path)
    if (state['status'] != 'completed' or state['updates'] != (0 if key == '6' else fit.UPDATES)
            or state['checkpoint_updates'] != fit.UPDATES or state['score_only'] != (key == '6')
            or state['tier'] != key or state['sampler_sha256'] != sampler_sha
            or state['plan_sha256'] != plan['files_sha256'][FIT_PLANS[key][0]]
            or any(state[k] != v for k, v in SCOPE.items())):
        raise ValueError(f'unexpected training receipt for {key} episodes')
    ck = f'{fit_dir}/{fit.ENDPOINT}'
    if key == '12' and accepted_artifact(plan, path, state, fit.ENDPOINT) != Path(ck):
        raise ValueError('12-episode checkpoint path changed')
    with np.load(BytesIO(_verified_bytes(ROOT, ck, plan['files_sha256'][ck])), allow_pickle=False) as stored:
        if int(stored['updates']) != fit.UPDATES:
            raise ValueError('checkpoint is not from the fixed update')
        for name in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(stored[name], getattr(fold, name))
        np.testing.assert_array_equal(stored['known_names'], KNOWN_NAMES)
        model = WorldIncrementMLP.from_state_dict(stored)
    artifacts = [fit.read(plan, accepted_artifact(plan, path, state, name))
                 for name in ('comparison.json', 'timed_rollouts.json', 'opening.json')]
    return model, *artifacts


def window_stats(values, metric):
    return {f'{a}-{b}': metric(values[a-1:b]) for a,b in REPORT_WINDOWS}


def rms(values):
    return float(np.sqrt(np.mean(np.sum(values**2, axis=-1))))


def length(values):
    return float(np.linalg.norm(values))


def prepare_development(plan, held, small):
    capture, _, _, full, _ = held
    launch = fit.read(plan, SMALL_CAPTURE)
    path = f"{launch['output']}/validation.json"
    accepted = fit.read(plan, path)
    result = {key: full[key] for key in BRANCHES[:4]}
    for key in BRANCHES[4:]:
        trace = fit.read(plan, accepted_artifact(plan, path, accepted, f'worker/{key}.json'))
        arc = _full_branch(trace, capture['case']['tick'])
        for field in ('raw', 'actions'):
            np.testing.assert_array_equal(arc[field], small[1][key][field])
        result[key] = arc
    return result


def source_audit(model, fold, held, anchor, traces, forecast):
    _, cfg, fields, _, _ = held
    state = _parse_state(fields)
    dt = model.dt_s
    actions = np.asarray([traces[k]['actions'] for k in BRANCHES])
    known = known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s, actions, cfg, dt)
    features = model.features(known, cfg, anchor, fold.known_mean, fold.known_scale)
    initial = np.broadcast_to(_ball_state(state), (len(traces), 9))
    truth = np.stack([traces[k]['raw'] for k in BRANCHES])
    free = model.denormalize(model.rollout(model.normalize(initial), features))
    expected = np.asarray([forecast['nominal']]*2 + [forecast['pairs']['pitch_reversal'][s] for s in ('plus', 'minus')])
    closure = require_close(free[:4], expected)
    measured_pre, rolled_pre = (np.concatenate((initial[:, None], path[:, :-1]), axis=1) for path in (truth, free))
    independent = model.denormalize(model.step(model.normalize(measured_pre), features))
    transform, world, shift = tool_transition(features)
    required, err = required_residual(measured_pre, truth, transform, world, shift)
    local_head = world_head(model, features, measured_pre)
    free_head = world_head(model, features, rolled_pre)
    defect = measured_pre[..., :3]+.5*dt*(measured_pre[..., 3:6]+truth[..., 3:6])-truth[..., :3]
    terms = {'independent_increment': propagate(local_head-required, transform, world, dt),
             'prediction_feedback': propagate(free_head-local_head, transform, world, dt),
             'position_rule': propagate_discrete_position(defect, transform, dt)}
    closure = max(closure, err, require_close(sum(terms.values()).sum(axis=0), free[...,:6]-truth[...,:6]),
                  require_close((world@(local_head-required)[..., None])[..., 0], independent[..., 3:6]-truth[..., 3:6]))
    gravity = np.array([0., 0., -cfg['physics']['gravity_m_s2']*dt])
    increments = []
    for i, key in enumerate(BRANCHES):
        _, errors, err = branch_world_terms(frame_series(state, traces[key], dt), truth[i], independent[i], free[i], gravity)
        increments.append(errors)
        closure = max(closure, err)
    return truth, independent, free, terms, {k: np.stack([row[k] for row in increments]) for k in ('local', 'free')}, closure


def summarise_tracks(data, archived):
    result = {}
    for name, weights in TRACKS.items():
        reduce = lambda x: np.einsum('b,btv->tv', weights, x)
        endpoints, source_vectors = {}, {}
        for k, (truth, independent, free, terms, increments, _) in data.items():
            measured, predicted = reduce(truth)[-1,:3], reduce(free)[-1,:3]
            position_error = reduce(free-truth)[:, :3]
            bucket = 'small' if name == 'roll_component' else 'full'
            if name == 'nominal':
                require_close(length(position_error[-1]), archived['nominal_h32_mm'][k]/1000)
            else:
                match = archived['scores'][bucket][name][k]
                require_close(length(measured), match['Newton_mm']/1000)
                require_close(length(predicted), match['finite']['mm']/1000)
                require_close(length(position_error[-1]), match['finite']['error_mm']/1000)
            endpoint = {'h32_newton_m': measured.tolist(), 'h32_prediction_m': predicted.tolist(),
                'h32_error_m': length(position_error[-1]), 'h32_prediction_norm_m': length(predicted),
                'free_position_error_m_by_lead': position_error.tolist(),
                'single_position_rmse_m': window_stats(reduce(independent-truth)[:, :3], rms),
                'world_velocity_increment_rmse_m_s': {part: window_stats(reduce(x), rms) for part,x in increments.items()}}
            source_vectors[k] = {part: np.einsum('b,sbv->sv', weights, tensor[:, :, -1, :3]) for part,tensor in terms.items()}
            require_close(sum(source_vectors[k].values()).sum(axis=0), position_error[-1])
            endpoints[k] = endpoint
        changes = {}
        for key in ('6', '12'):
            first, second = (endpoints[k] for k in ('61_pairs', key))
            p, q = (np.asarray(e['h32_prediction_m']) for e in (first, second))
            truth = np.asarray(first['h32_newton_m'])
            projections = {}
            pairs = {'error_growth': (p-truth, q-truth)}
            if name == 'roll_component':
                pairs['amplitude_growth'] = (p,q)
            for kind, (a,b) in pairs.items():
                axis = (a+b)/(length(a)+length(b))
                parts = {part: window_stats((value-source_vectors['61_pairs'][part])@axis,
                                            lambda x: float(x.sum()))
                         for part,value in source_vectors[key].items()}
                growth = length(b)-length(a)
                closure = require_close(sum(v['1-32'] for v in parts.values()), growth)
                projections[kind] = {'h32_growth_m': growth, 'by_source_and_window_m': parts, 'closure_m': closure}
            changes[key] = projections
        result[name] = {'endpoints': endpoints, 'changes_vs_61_pairs': changes}
    return result


def pair_terms(plan, endpoint, fold, cfg, scales, held, anchor, previous, parent, guard):
    dense = fit.read(plan, fit.read(plan, previous['coverage_design'])['reference_design'])
    leads = fit.lead_vectors(dense, 32)
    base, velocity = leads
    components = np.asarray(cfg['components'])
    units = np.asarray(scales['state_error_scale_si'])
    batch = fold.pairs
    inputs = (endpoint, endpoint.normalize(batch.initial),
              packed(endpoint, np.stack((batch.plus_known, batch.minus_known)), held[1], anchor, fold),
              np.stack((batch.plus_target, batch.minus_target)), np.stack((batch.plus_mask, batch.minus_mask)),
              batch.response_mask, batch.noise_mask, batch.response_floor, base, components,
              np.asarray(cfg['response_scales_si']))
    weights = cfg['pairs']
    parts, values = {}, {}
    for label, absolute, response in (('absolute', weights['absolute_weight'], 0.),
                                      ('response', 0., weights['response_weight'])):
        score, parts[label] = paired_velocity_response_loss(*inputs, absolute, response, velocity_lead_weights=velocity,
                                              sample_weights=fold.pair_weights, absolute_scale_si=units)
        values[label] = score['loss']
    targets, _ = fit.supervision(previous, fold, previous['timed_manifests'], guard)
    coefficients = fit.read(plan, parent['window_protocol'])['auxiliary']['coefficients']
    for label, (value, gradient, _) in losses(endpoint, targets).items():
        parts[label] = mul(gradient, coefficients[label])
        values[label] = value*coefficients[label]
    original = fit.combine(endpoint, fit.objectives(endpoint, fold, cfg, scales, leads, held[1], anchor)[0],
                           targets, coefficients)
    return parts, values, original


def dot_record(reference, added):
    reference_norm, added_norm = norm(reference), norm(added)
    projection = dot(reference, added)
    return {'cosine': projection/(reference_norm*added_norm), 'added_norm_ratio': added_norm/reference_norm,
            'signed_fraction_of_reference': projection/(reference_norm**2), 'dot': projection}


def percentile(rows):
    data = np.asarray(rows, dtype=float)
    if not np.isfinite(data).all() or data.size != fit.UPDATES:
        raise FloatingPointError('missing or nonfinite frozen gradient draws')
    return {'min': float(data.min()), 'p50': float(np.median(data)), 'max': float(data.max()),
            'mean': float(data.mean()), 'negative_draws': int((data<0).sum())}


def inspect_gradients(endpoint, fold, cfg, scales, held, anchor, arrays, stream, streams,
                      parts, values, original, coefficient, guard):
    response = add(parts['response'], parts['half_difference'])
    pair = add(*parts.values())
    components = np.asarray(cfg['components'])
    units = np.asarray(scales['state_error_scale_si'])
    anchors = {h: (endpoint.normalize(w.initial), packed(endpoint, w.known, held[1], anchor, fold),
                   endpoint.normalize(w.target)) for h,w in fold.anchors.items()}
    ordinary = fit.route_objective(endpoint, arrays, cfg, scales)
    rows = {key: [] for key in streams}
    closure = 0.
    for i, (h, take) in enumerate(stream):
        anchor_loss, s2 = endpoint.loss_and_grad(*(x[take] for x in anchors[h]), state_weights=components,
                                                state_error_scale_si=units)
        combined = add(s2, pair)
        if i in (0, fit.UPDATES-1):
            expected_loss, expected_terms, expected_gradient = original(h, take)
            closure = max(closure, require_close(anchor_loss, expected_loss),
                          require_close(sum(values.values()), expected_terms['loss']),
                          *(require_close(combined[k], expected_gradient[k]) for k in combined))
        references = {**parts, 'response_combined': response, 'pair_combined': pair, 's2': s2, 'original_total': combined}
        for key, indices in streams.items():
            loss, grad = ordinary(indices[i])
            added = mul(grad, coefficient)
            original_effect = dot(combined, response)
            modified_effect = dot(add(combined, added), response)
            rows[key].append({'draw': i+1, 's2_horizon': h, 'ordinary_loss': loss, 's2_loss': anchor_loss,
                'weighted_ordinary_gradient_norm': norm(added), 'comparisons': {k: dot_record(g, added) for k,g in references.items()},
                'original_descent_reduces_response_linearized': original_effect > 0.,
                'augmented_descent_reduces_response_linearized': modified_effect > 0.,
                'original_response_dot': original_effect, 'augmented_response_dot': modified_effect})
        if (i+1)%30 == 0:
            guard('gradient_audit', 0, log=(i+1)%150 == 0)
    summary = {}
    for key, table in rows.items():
        summary[key] = {'draws': len(table), 'weighted_ordinary_gradient_norm': percentile([r['weighted_ordinary_gradient_norm'] for r in table]),
            'comparisons': {label: {stat: percentile([r['comparisons'][label][stat] for r in table])
                                    for stat in ('cosine', 'added_norm_ratio', 'signed_fraction_of_reference', 'dot')}
                            for label in table[0]['comparisons']},
            'old_descent_reduces_response': sum(r['original_descent_reduces_response_linearized'] for r in table),
            'augmented_descent_reduces_response': sum(r['augmented_descent_reduces_response_linearized'] for r in table),
            'reduction_reversed_by_ordinary_term': sum(r['original_descent_reduces_response_linearized'] and
                  not r['augmented_descent_reduces_response_linearized'] for r in table)}
    return {'coefficient': coefficient, 'fixed_original_61_checkpoint': True, 'training_draws_only': True,
            'gradient_projection_before_optimizer': True, 'pair_terms_weighted_loss': values,
            'pair_terms_gradient_norm': {key: norm(value) for key,value in {**parts, 'response_combined': response,
                                                                            'pair_combined': pair}.items()},
            'gradient_closure': closure, 'by_tier': summary, 'draw_details': rows}


def run(plan, out, guard):
    if ((plan['updates'], plan['endpoint_updates'], plan['training_draws']) != (0, fit.UPDATES, fit.UPDATES)
            or tuple(plan['branches']) != BRANCHES or plan['files_sha256'][COMPARISON] != SUMMARY_SHA):
        raise ValueError('unexpected frozen audit scope')
    (initial, endpoint, original_opening, _, _, stream, sampler, fold, cfg, scales, held, anchor,
        measured, old_score, previous, parent) = fit.reference(plan, guard)
    archived = fit.read(plan, COMPARISON)
    if archived['fixed_update'] != fit.UPDATES or archived['original_s2_sampler_sha256'] != sampler:
        raise ValueError('training comparison draws or endpoint changed')
    old_path = f'{fit.REFERENCE_RUN}/validation.json'
    old_status = fit.read(plan, old_path)
    old_rollout = fit.read(plan, accepted_artifact(plan, old_path, old_status, 'timed_rollouts.json'))
    models = {key: load_endpoint(plan, key, fold, sampler) for key in FIT_PLANS}
    parameter_digests = {key: fit.digest_arrays(m.params.values(), '<f8')
                         for key,m in [('61_pairs', endpoint), *((k,v[0]) for k,v in models.items())]}

    small = small_inputs(plan, held)
    traces = prepare_development(plan, held, small)
    evaluations = {'61_pairs': (endpoint, old_score['held_timing'], small[2], old_rollout)}
    for key, (model, score, forecasts, _) in models.items():
        if (model.dt_s != endpoint.dt_s or not np.array_equal(model.inc_center, endpoint.inc_center)
                or score['small']['61_pairs'] != small[2] or score['full']['61_pairs'] != old_score['held_timing']):
            raise ValueError('candidate rescoring baseline or checkpoint dynamics changed')
        evaluations[key] = (model, score['full']['candidate'], score['small']['candidate'], forecasts)
    trajectories = {}
    for key, (model, full, minor, forecasts) in evaluations.items():
        reproduced, rollout = score_candidate(model, held, fold, anchor, guard)
        if reproduced != full or rollout != forecasts or score_small(model, held, fold, anchor, small, guard) != minor:
            raise ValueError(f'archived development scores changed for {key}')
        trajectories[key] = source_audit(model, fold, held, anchor, traces, forecasts)
    track_report = summarise_tracks(trajectories, archived)
    closure = max(value[-1] for value in trajectories.values())
    write_json(out/'trajectory_audit.json', {'case': held[0]['case'], 'max_closure_si': closure, 'tracks': track_report})
    guard('trajectory_audit_complete', 0, log=True)

    arrays, streams, inventory = fit.route_windows(plan, initial, fold, held[1], anchor, guard)
    opening = fit.check_opening(initial, original_opening, fit.route_objective(initial, arrays, cfg, scales),
                                stream, streams, plan['seed'])
    expected = {'original': measured, 'ordinary': inventory, 'opening': opening, 'optimizer': cfg['optimizer'],
                'original_s2_sampler_sha256': sampler}
    if any(value[-1] != expected for value in models.values()):
        raise ValueError('ordinary window inventory, draws or coefficient changed')
    parts, values, original = pair_terms(plan, endpoint, fold, cfg, scales, held, anchor, previous, parent, guard)
    gradients = inspect_gradients(endpoint, fold, cfg, scales, held, anchor, arrays, stream, streams,
                                  parts, values, original, opening['coefficient'], guard)
    if any(fit.digest_arrays(values[0].params.values(), '<f8') != parameter_digests[key]
           for key, values in evaluations.items()):
        raise ValueError('audit changed a fixed checkpoint')
    write_json(out/'training_gradient_audit.json', {**gradients, 'original_s2_sampler_sha256': sampler,
               'ordinary_sampler_sha256_by_tier': inventory['sampler_sha256_by_tier']})
    return {'fixed_endpoints': 3, 'archived_development_scores_reproduced': True, 'diagnostic_tracks': len(TRACKS),
            'training_draws_per_tier': fit.UPDATES, 'max_trajectory_closure_si': closure,
            'max_gradient_closure': gradients['gradient_closure'], 'main_model_replaced': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'audit'))
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.phase == 'freeze':
        freeze()
    elif args.sha256:
        run_phase(ROOT/PLAN, args.sha256, PHASE, run)
    else:
        parser.error('audit requires a frozen plan SHA-256')
