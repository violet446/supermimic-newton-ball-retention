"""P1 diagnostic fit of six timed training pairs from a fixed checkpoint."""
import argparse
import json

from s3_fitability_common import PARTS, ROOT, branch_arrays, forecast, frozen_data, norm_stats, run_phase, write_json

import numpy as np

from ball_retention.model_data import KNOWN_NAMES
from ball_retention.pair_loss import paired_loss
from ball_retention.position_integral_model import load_position_model
from ball_retention.world_model import Adam
from check_s3_actions import digest


def pair_scores(predicted, truth, initial, features, model, rows, leads, gate):
    local = forecast(model, np.tile(initial, (2, 1)),
                     features.reshape(-1, *features.shape[2:]), truth.reshape(-1, *truth.shape[2:]),
                     local=True).reshape(truth.shape)
    if not all(np.isfinite(array).all() for array in (predicted, local)):
        raise FloatingPointError('nonfinite diagnostic trajectory')
    half = lambda v: .5*(v[0]-v[1])
    measured, free, one_step = map(half, (truth, predicted, local))
    pairs = []
    for i, row in enumerate(rows):
        pair = {'identity': row, 'branch_absolute_rmse_si': {
            sign: norm_stats(predicted[j, i]-truth[j, i], 1.) for j, sign in enumerate(('plus', 'minus'))},
            'by_lead': {}}
        for lead in leads:
            tick = lead-1
            absolute = predicted[:, i, tick]-truth[:, i, tick]
            by_part = {}
            for name, part in PARTS.items():
                target = measured[i, tick, part]
                estimate = free[i, tick, part]
                onestep = one_step[i, tick, part]
                norm, measured_norm = np.linalg.norm(estimate), np.linalg.norm(target)
                error = float(np.linalg.norm(estimate-target))
                by_part[name] = {'target_half_difference_si': target.tolist(),
                                 'free_half_difference_si': estimate.tolist(),
                                 'local_half_difference_si': onestep.tolist(),
                                 'free_error_si': error,
                                 'local_error_si': float(np.linalg.norm(onestep-target)),
                                 'propagation_norm_si': float(np.linalg.norm(estimate-onestep)),
                                 'cosine': float(np.clip(estimate@target/(norm*measured_norm), -1., 1.))
                                 if norm and measured_norm else None,
                                 'relative_error': float(error/measured_norm) if measured_norm else None}
            pair['by_lead'][lead] = {'absolute_pair_rmse_si': norm_stats(absolute, 1.), 'half_difference': by_part}
        result = pair['by_lead'][32]['half_difference']['velocity']
        pair['h32_velocity_agrees'] = (result['cosine'] is not None and result['relative_error'] is not None
                                       and result['cosine'] >= gate['min_cosine']
                                       and result['relative_error'] <= gate['max_relative_error'])
        pairs.append(pair)
    return {'absolute_rmse_all_leads_si': norm_stats(predicted-truth, 1.),
            'absolute_rmse_h32_si': norm_stats(predicted[:, :, -1]-truth[:, :, -1], 1.),
            'half_difference_rmse_h32_si': norm_stats(free[:, -1]-measured[:, -1], 1.),
            'h32_velocity_agrees_count': sum(row['h32_velocity_agrees'] for row in pairs), 'pairs': pairs}


def run(plan, out, guard):
    _, _, cfg, scales, fold, rows, _, model = frozen_data(plan, guard)
    p0 = ROOT/plan['p0_acceptance']
    evidence = json.loads(p0.read_text(encoding='utf-8'))
    audit = ROOT/plan['p0_gradients']
    previous = json.loads(audit.read_text(encoding='utf-8'))
    if (evidence['status'] != 'completed' or evidence['phase'] != 'P0' or evidence['updates'] != 0
            or evidence['artifacts_sha256'][audit.name] != plan['files_sha256'][plan['p0_gradients']]
            or rows != previous['pair_identity']):
        raise ValueError('P0 training data or evidence identity mismatch')
    guard('prepare', log=True)
    selected = plan['training_rows']
    if selected != [row['row'] for row in rows if row['timed']] or len(selected) != 6:
        raise ValueError('expected exactly the frozen six timed training pairs')
    rows = [rows[i] for i in selected]
    raw, known, targets, masks = branch_arrays(fold)
    initial, features, truth, masks = raw[selected], known[:, selected], targets[:, selected], masks[:, selected]
    batch = fold.pairs
    seen = batch.noise_mask[selected]
    if not (masks.all() and seen.all() and truth.shape == (2, 6, 32, 9)):
        raise ValueError('incomplete timed branch targets or replay mask')
    if cfg['optimizer'] != plan['optimizer'] or cfg['components'] != plan['components']:
        raise ValueError('optimizer or component weights changed')
    if plan['absolute_weight'] != cfg['pairs']['absolute_weight'] or plan['response_weight'] != cfg['pairs']['response_weight']:
        raise ValueError('pair coefficients changed')
    inputs = (model, model.normalize(initial), features, truth, masks,
              batch.response_mask[selected], seen, batch.response_floor[selected])
    leads = np.r_[np.full(31, plan['lead_mass_h1_h31']/31), plan['lead_mass_h32']]
    options = {'sample_weights': fold.pair_weights[selected], 'absolute_scale_si': np.asarray(scales['state_error_scale_si'])}
    response_units = np.asarray(plan['response_scales_si'])

    def evaluate(gradients=True):
        return paired_loss(*inputs, leads, np.asarray(plan['components']), response_units,
                           plan['absolute_weight'], plan['response_weight'], gradients=gradients, **options)

    packed = features.reshape(-1, *features.shape[2:])
    doubled = np.tile(initial, (2, 1))

    def measure(predictor):
        predicted = predictor.denormalize(predictor.rollout(predictor.normalize(doubled), packed)).reshape(truth.shape)
        return pair_scores(predicted, truth, initial, features, predictor, rows, plan['report_leads'], plan['gate'])

    def trace(update, metrics, summary, grad_norm):
        return {'update': update, 'loss': metrics, 'gradient_norm_before_last_update': grad_norm,
                'absolute_rmse_all_leads_si': summary['absolute_rmse_all_leads_si'],
                'half_difference_rmse_h32_si': summary['half_difference_rmse_h32_si'],
                'h32_velocity_agrees_count': summary['h32_velocity_agrees_count'],
                'pair_h32_velocity': [{
                    key: row['by_lead'][32]['half_difference']['velocity'][key]
                    for key in ('free_error_si', 'cosine', 'relative_error')} for row in summary['pairs']]}

    start_metrics, _ = evaluate(False)
    before = measure(model)
    if before['h32_velocity_agrees_count'] != 0:
        raise ValueError('P1 baseline does not reproduce the original six-row result')
    optimizer = Adam(model.params, learning_rate=cfg['optimizer']['learning_rate'],
                     max_grad_norm=cfg['optimizer']['max_grad_norm'])
    guard('fit', 0, log=True)
    with (out/'training.jsonl').open('x', encoding='utf-8') as log:
        log.write(json.dumps(trace(0, start_metrics, before, None), allow_nan=False)+'\n')
        for step in range(1, plan['updates']+1):
            _, grads = evaluate()
            norm = optimizer.step(grads)
            if not all(np.isfinite(value).all() for value in model.params.values()):
                raise FloatingPointError(f'nonfinite model after update {step}')
            periodic = step in plan['report_updates']
            guard('fit', step, log=periodic)
            if periodic:
                metrics, _ = evaluate(False)
                summary = measure(model)
                log.write(json.dumps(trace(step, metrics, summary, norm), allow_nan=False)+'\n')
                log.flush()
    if optimizer.iteration != plan['updates']:
        raise ValueError('fixed optimizer endpoint differs from the plan')
    guard('endpoint', optimizer.iteration, log=True)
    checkpoint = out/'diagnostic_timed_1800.npz'
    np.savez_compressed(checkpoint, **model.state_dict(), known_mean=fold.known_mean, known_scale=fold.known_scale,
                        known_names=np.asarray(KNOWN_NAMES), updates=np.asarray(optimizer.iteration),
                        state_error_scale_si=options['absolute_scale_si'],
                        residual_gain_si=np.asarray(scales['residual_gain_si']), diagnostic_only=True)
    with np.load(checkpoint, allow_pickle=False) as saved:
        restored = load_position_model(saved)
    np.testing.assert_array_equal(model.rollout(model.normalize(doubled), packed),
                                  restored.rollout(restored.normalize(doubled), packed))
    endpoint = measure(restored)
    passes = endpoint['h32_velocity_agrees_count'] == 6
    write_json(out/'fitability.json', {'scope': 'Six training pairs only', 'gate': plan['gate'],
               'initial': before, 'endpoint': endpoint, 'fitability_gate_passed': passes,
               'main_model_replaced': False, 'checkpoint_selection': False})
    return {'training_pairs': 6, 'h32_velocity_agrees_count': endpoint['h32_velocity_agrees_count'],
            'fitability_gate_passed': passes, 'checkpoint_sha256': digest(checkpoint),
            'checkpoint_selection': False, 'main_model_replaced': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(ROOT/'configs/s3_fitability_p1_serialization_fix_plan.json', args.plan_sha256, 'P1', run)
