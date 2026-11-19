"""Decompose archived discrete ball velocity increments and fixed-model errors."""
import argparse
from io import BytesIO
import json
from pathlib import Path

from s3_fitability_common import ROOT, SCOPE, forecast, run_phase, write_json

import numpy as np
from scipy.spatial.transform import Rotation

from ball_retention.action_grad import _inputs
from ball_retention.archive_state_import import _parse_state
from ball_retention.model_data import KNOWN_NAMES, _ball_state
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from deterministic_archive_metrics import _full_branch
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


ADVANCE = Path('runs/s3_training_timing_advance_20261116/validation.json')
VELOCITY = slice(3, 6)
ATOL = 1e-12
WINDOW = (8, 24)
LEADS = (1, 8, 16, 24, 32)
TERMS = ('frame', 'gravity', 'non_gravity')


def half(rows):
    return .5*(rows[0]-rows[1])


def rms(rows):
    return float(np.sqrt(np.mean(np.sum(rows**2, axis=-1))))


def closure(expected, result):
    np.testing.assert_allclose(result, expected, atol=ATOL, rtol=0.)
    return float(np.max(np.linalg.norm(result-expected, axis=-1)))


def load_cases(plan, read):
    capture, cfg, fields, held, _ = read_held(read(plan['held_design']))
    accepted, manifest = read(ADVANCE), read(ADVANCE.with_name('capture_manifest.json'))
    entry, = manifest['entries']
    manifest_sha = plan['files_sha256'][ADVANCE.with_name('capture_manifest.json').as_posix()]
    if (accepted['status'] != 'completed' or accepted['pairs'] != 3 or accepted['training_updates'] != 0
            or accepted['test_episode_files_read'] is not False
            or accepted['artifacts_sha256']['capture_manifest.json'] != manifest_sha
            or entry['split'] != 'train' or entry['history_id'] != 'advance' or entry['snapshot_tick'] != 168
            or [pair['direction_id'] for pair in entry['pairs']] != list(capture['directions'])
            or plan['files_sha256'][entry['source_config']] != entry['source_config_sha256']):
        raise ValueError('unexpected advance scope or source integrity')

    def archive(ref):
        file = ADVANCE.parent/ref['path']
        if (ref['sha256'] != plan['files_sha256'][file.as_posix()]
                or ref['sha256'] != accepted['artifacts_sha256'][ref['path']]):
            raise ValueError(f'advance artifact hash changed: {file}')
        return read(file)

    refs = [*entry['nominal_branches'], *(branch for pair in entry['pairs'] for branch in pair['branches'])]
    advance = {}
    for ref in refs:
        raw = archive(ref)
        np.testing.assert_array_equal(raw['actions'], ref['requested_actions_Hx2'])
        advance[ref['label']] = _full_branch(raw, entry['snapshot_tick'])
    return capture, {
        'development': (cfg, _parse_state(fields), held),
        'advance': (read(entry['source_config']), _parse_state(archive(entry['native_snapshot'])['state']), advance)}


def frame_series(initial, trace, dt):
    states = [initial, *trace['states']]
    np.testing.assert_allclose(np.diff([s.time_s for s in states]), dt, atol=ATOL, rtol=0.)
    tools = [s.tool for s in states]
    rotation = Rotation.from_quat([s.quaternion_xyzw for s in tools])
    shift = np.asarray([s.linear_velocity_origin for s in tools])
    omega = rotation.apply([s.angular_velocity for s in tools], inverse=True)
    archived_world = np.asarray([s.ball.linear_velocity_origin for s in states])
    return rotation, shift, omega, archived_world


def velocity_terms(frame, current, prior, world, prior_world, gravity_step_world):
    rotation, shift, omega = frame
    transport = -np.cross(omega, current[:, :3])
    previous_in_endpoint_frame = rotation.apply(prior_world-shift, inverse=True)+transport
    components = {
        'frame': previous_in_endpoint_frame-prior[:, VELOCITY],
        'gravity': rotation.apply(gravity_step_world, inverse=True),
        'non_gravity': rotation.apply(world-prior_world-gravity_step_world, inverse=True),
        'delta': current[:, VELOCITY]-prior[:, VELOCITY]}
    error = closure(components['delta'], sum(components[k] for k in TERMS))
    return components, error


def branch_terms(frame, truth, gravity_step_world, *, endpoint=None, independent=False):
    rotation, shift, omega, archived_world = frame
    ball_path = truth if endpoint is None else np.vstack((truth[0], endpoint))
    world = archived_world
    if endpoint is not None:
        # Use achieved tool motion only after inference to express the predictions in world coordinates.
        world = shift+rotation.apply(ball_path[:, VELOCITY]+np.cross(omega, ball_path[:, :3]))
    prior, prior_world = (truth[:-1], archived_world[:-1]) if independent else (ball_path[:-1], world[:-1])
    terms, error = velocity_terms((rotation[1:], shift[1:], omega[1:]), ball_path[1:], prior,
                                  world[1:], prior_world, gravity_step_world)
    return terms, world, error


def component_half(branches):
    return {key: half([row[key] for row in branches]) for key in branches[0]}


def physical_metrics(components, mask):
    delta = components['delta'][mask]
    energy = np.sum(delta**2)
    return {'rms_m_s': {key: rms(value[mask]) for key, value in components.items()},
            'signed_projection_fraction': {
                key: float(np.sum(components[key][mask]*delta)/energy) if energy > 0 else None for key in TERMS}}


def model_components(model, mean, scale, initial, cfg, traces, frames, truth, physics, gravity_step, mask):
    features = np.stack([_inputs(model, row[0], initial.time_s, initial.tilt_rad, initial.tilt_rate_rad_s,
                                  trace['actions'], cfg, mean, scale)[2] for row, trace in zip(truth, traces, strict=True)])
    targets = np.stack([row[1:] for row in truth])
    starts = np.stack([row[0] for row in truth])
    modes, predictions, max_closure = {}, {}, 0.
    for mode in ('local', 'free'):
        predicted = forecast(model, starts, features, targets, local=mode == 'local')
        if not np.isfinite(predicted).all():
            raise FloatingPointError('nonfinite fixed-model forecast')
        branches, state_errors = [], []
        for frame, actual, pred in zip(frames, truth, predicted, strict=True):
            terms, world, err = branch_terms(frame, actual, gravity_step, endpoint=pred, independent=mode == 'local')
            branches.append(terms)
            rotation, _, omega, archived_world = frame
            position_error = pred[:, :3]-actual[1:, :3]
            state = {'world_velocity': rotation[1:].apply(world[1:]-archived_world[1:], inverse=True),
                     'position_lever': -np.cross(omega[1:], position_error),
                     'total': pred[:, VELOCITY]-actual[1:, VELOCITY]}
            max_closure = max(max_closure, err, closure(state['total'], state['world_velocity']+state['position_lever']))
            state_errors.append(state)
        terms = component_half(branches)
        error = {key: value-physics[key] for key, value in terms.items()}
        state = component_half(state_errors)
        max_closure = max(max_closure, closure(error['delta'], sum(error[k] for k in TERMS)),
                          closure(state['total'], state['world_velocity']+state['position_lever']))
        if mode == 'local':
            max_closure = max(max_closure, closure(state['total'], error['delta']),
                              closure(state['world_velocity'], error['non_gravity']))
        modes[mode] = {
            'predicted_increment_half_difference_m_s': {k: v.tolist() for k, v in terms.items()},
            'velocity_state_error_half_difference_m_s': {k: v.tolist() for k, v in state.items()},
            'window_increment_error_rms_m_s': {k: rms(v[mask]) for k, v in error.items()},
            'window_velocity_state_error_rms_m_s': {k: rms(v[mask]) for k, v in state.items()},
            'h32_velocity_state_error_norm_m_s': {k: float(np.linalg.norm(v[-1])) for k, v in state.items()}}
        predictions[mode] = predicted
    return modes, predictions, max_closure


def run(plan, out, guard):
    if (plan['updates'] != 0 or plan['window_leads'] != list(WINDOW)
            or plan['closure_atol_m_s'] != ATOL or tuple(plan['models']) != ('dense_49', 'increment_2_5_mm_s')):
        raise ValueError('unexpected physical decomposition scope')

    def read(path):
        return json.loads(bound(plan, path))

    model_records = {}
    for label, folder in plan['models'].items():
        folder = Path(folder)
        accepted = read(folder/'validation.json')
        if (accepted['status'] != 'completed' or (accepted['updates'], accepted['training_pairs']) != (900, 49)
                or accepted['anchor_draws_sha256'] != plan['anchor_draws_sha256']
                or any(accepted[k] != v for k, v in SCOPE.items())
                or (label == 'increment_2_5_mm_s' and accepted['local_scale_m_s'] != .0025)):
            raise ValueError(f'unexpected fixed model: {label}')
        names = ('velocity_response_leads_900.npz', 'timed_rollouts.json', 'comparison.json')
        if any(plan['files_sha256'][(folder/name).as_posix()] != accepted['artifacts_sha256'][name] for name in names):
            raise ValueError(f'fixed model artifact hash changed: {label}')
        checkpoint, rollouts, comparison = [folder/name for name in names]
        saved_rollouts = read(rollouts)
        if saved_rollouts['checkpoint_sha256'] != accepted['checkpoint_sha256'] or (
                plan['files_sha256'][checkpoint.as_posix()] != accepted['checkpoint_sha256']):
            raise ValueError(f'fixed model rollout identity changed: {label}')
        with np.load(BytesIO(bound(plan, checkpoint)), allow_pickle=False) as saved:
            model = load_position_model(saved)
            if (type(model) is not TrapezoidPositionResidualMLP or int(saved['updates']) != 900
                    or tuple(saved['known_names']) != KNOWN_NAMES):
                raise ValueError(f'unexpected model architecture, endpoint or features: {label}')
            advance = read(comparison)['training_timed']['candidate_49']['advance']['pairs']
            advance_scores = {row['identity']['offset_pattern']: row for row in advance}
            model_records[label] = (model, saved['known_mean'], saved['known_scale'], saved_rollouts, advance_scores)

    capture, cases = load_cases(plan, read)
    results, physical_max, model_max = {}, 0., 0.
    for case, (cfg, initial, traces) in cases.items():
        if list(traces) != capture['branch_order'] or capture['amplitude'] != .2:
            raise ValueError(f'unexpected branch scope: {case}')
        dt = cfg['physics']['dt_s']*capture['control_substeps']
        if abs(dt*cfg['physics']['control_hz']-1.) > 1e-14 or capture['control_substeps'] != 16:
            raise ValueError('control clock changed')
        gravity_step = dt*np.array([0., 0., -cfg['physics']['gravity_m_s2']])
        frames, truth, absolute_terms = {}, {}, {}
        for label, trace in traces.items():
            frames[label] = frame_series(initial, trace, dt)
            truth[label] = np.vstack((_ball_state(initial), trace['raw']))
            if truth[label].shape != (33, 9) or not np.isfinite(truth[label]).all():
                raise ValueError(f'invalid 32-step archived state: {label}')
            terms, _, err = branch_terms(frames[label], truth[label], gravity_step)
            physical_max = max(physical_max, err)
            absolute_terms[label] = terms
        gates = capture['gates']
        nominal = half([truth[label][1:, VELOCITY] for label in ('nominal_1', 'nominal_2')])
        pairs = {}
        for direction in capture['directions']:
            labels = [f'{direction}_{side}' for side in ('plus', 'minus')]
            states = [truth[label] for label in labels]
            state_half = half([row[1:, VELOCITY] for row in states])
            signal = np.linalg.norm(state_half, axis=-1)
            floor = np.maximum.reduce([np.full(32, gates['velocity_floor_m_s']),
                                      gates['noise_multiplier']*np.linalg.norm(nominal, axis=-1),
                                      32*np.finfo(float).eps*np.maximum(1., signal)])
            mask = signal >= gates['strong_multiplier']*floor
            mask[:WINDOW[0]-1], mask[WINDOW[1]:] = False, False
            if not mask.any():
                raise ValueError(f'no strong velocity response in frozen window: {case}/{direction}')
            physical = component_half([absolute_terms[label] for label in labels])
            physical_max = max(physical_max, closure(physical['delta'], sum(physical[k] for k in TERMS)),
                               closure(physical['delta'], np.diff(np.vstack((np.zeros(3), state_half)), axis=0)))
            models = {}
            for key, (model, mean, scale, dev, saved_advance) in model_records.items():
                if abs(model.dt_s-dt) > 1e-14:
                    raise ValueError('model control clock changed')
                modes, predictions, err = model_components(model, mean, scale, initial, cfg,
                    [traces[label] for label in labels], [frames[label] for label in labels],
                    states, physical, gravity_step, mask)
                model_max = max(model_max, err)
                if case == 'development':
                    row = dev['pairs'][direction]
                    closure(np.asarray([row['plus'], row['minus']]), predictions['free'])
                else:
                    row = saved_advance[direction]
                    if (row['identity']['episode'], row['identity']['tick']) != ('history_train_yplus004_advance', 168):
                        raise ValueError('unexpected recorded advance identity')
                    for h in LEADS:
                        expected = row['by_lead'][str(h)]['half_difference']['velocity']
                        closure(expected['target_half_difference_si'], state_half[h-1])
                        for mode, values in predictions.items():
                            closure(expected[f'{mode}_half_difference_si'], half(values)[h-1, VELOCITY])
                models[key] = modes
            pairs[direction] = {
                'strong_window_leads': (np.flatnonzero(mask)+1).tolist(),
                'velocity_state_half_difference_m_s': state_half.tolist(),
                'velocity_signal_floor_m_s': floor.tolist(),
                'increment_half_difference_m_s': {k: v.tolist() for k, v in physical.items()},
                'window': physical_metrics(physical, mask), 'models': models}
            guard(case)
        results[case] = {'control_dt_s': dt, 'pairs': pairs,
                         'branches': {key: {k: v.tolist() for k, v in row.items()} for key, row in absolute_terms.items()}}
        guard(case, log=True)
    write_json(out/'components.json', {'window_leads': WINDOW, 'cases': results,
               'convention': 'Components use endpoint achieved tool coordinates before taking signed half differences. '
               'The non-gravity term is the measured world velocity increment minus integrated gravity, then rotated. '
               'Achieved tool motion is used after model inference only.'})
    return {'snapshots': len(results), 'branches_per_snapshot': 8, 'directions_per_snapshot': 3, 'checkpoints': 2,
            'leads_per_branch': 32, 'max_physical_closure_m_s': physical_max, 'max_model_closure_m_s': model_max}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'ball_velocity_components', run)
