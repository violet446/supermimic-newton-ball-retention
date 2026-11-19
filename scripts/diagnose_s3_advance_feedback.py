"""Locate local and propagated response errors at the fixed 46/49-pair endpoints."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from s3_fitability_common import FIT, ROOT, forecast, run_phase, write_json

import numpy as np

from ball_retention.action_grad import _inputs
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.model_data import KNOWN_NAMES, _ball_state
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.timed_pair_data import prepare_timed_pairs
from diagnose_s3_position_integral import ATOL, BANDS, differences, summarize, trapezoid
from response_contact_diagnosis import diagnose_contacts
from train_s3_position_integral import read_held, score_held


VELOCITY = slice(3, 6)
CANDIDATE = Path('runs/s3_fitability_p3_20261116/validation.json')
CAPTURE = Path('runs/s3_training_timing_advance_20261116/validation.json')


def half(branches):
    return .5*(branches[0]-branches[1])


def onset(mask, consecutive):
    return next((i+1 for i in range(len(mask)-consecutive+1) if mask[i:i+consecutive].all()), None)


def trace_error(reference, predicted, floor, ratio_limit, consecutive):
    size = np.linalg.norm(reference, axis=-1)
    estimated = np.linalg.norm(predicted, axis=-1)
    error = np.linalg.norm(predicted-reference, axis=-1)
    strong = size >= 10*floor
    dot = np.einsum('ij,ij->i', predicted, reference)
    return {'strong_velocity_signal': strong.tolist(),
            'error_norm_m_s': error.tolist(),
            'first_sustained_large_relative_error': onset(strong & (error > ratio_limit*size), consecutive),
            'first_sustained_opposition': onset(strong & (estimated >= floor) & (dot < 0.), consecutive),
            'peak_error_lead': int(error.argmax())+1}


def diagnose_pair(model, initial, features, truth, nominal, settings):
    """Use both orders of midpoint and half-difference replacement at each lead."""
    start = np.broadcast_to(initial, (2, 9))
    free = forecast(model, start, features, truth)
    local = forecast(model, start, features, truth, local=True)
    prior_truth, prior_free = (np.concatenate((start[:, None], v[:, :-1]), axis=1) for v in (truth, free))
    m0, m1 = prior_truth.mean(axis=0), prior_free.mean(axis=0)
    d0, d1 = half(prior_truth), half(prior_free)

    def step(mid, diff):
        mixed = model.normalize(np.stack((mid+diff, mid-diff)))
        return model.denormalize(model.step(mixed, features))

    end_truth = step(m0, d0)
    end_free = step(m1, d1)
    np.testing.assert_allclose(end_truth, local, rtol=0., atol=1e-12)
    np.testing.assert_allclose(end_free, free, rtol=0., atol=1e-12)
    v00, v10, v01, v11 = (half(a)[..., VELOCITY] for a in (local, step(m1, d0), step(m0, d1), free))
    measured = half(truth)
    v = measured[..., VELOCITY]
    midpoint = .5*((v10-v00)+(v11-v01))
    response = .5*((v01-v00)+(v11-v10))
    propagation = v11-v00
    local_error = v00-v
    free_error = v11-v
    np.testing.assert_allclose(propagation, midpoint+response, rtol=0., atol=ATOL)
    np.testing.assert_allclose(free_error, local_error+propagation, rtol=0., atol=ATOL)
    np.testing.assert_allclose(propagation[0], 0., rtol=0., atol=ATOL)
    if not all(np.isfinite(a).all() for a in (truth, free, local, midpoint, response)):
        raise FloatingPointError('nonfinite trajectory or contribution')
    dt = model.dt_s
    actual_position, predicted_position = measured[:, :3], half(free)[:, :3]
    np.testing.assert_allclose(differences(predicted_position), trapezoid(v11, dt), rtol=0., atol=ATOL)
    increments = {k: trapezoid(value, dt) for k, value in
                  {'one_step_velocity_defect': local_error, 'midpoint_feedback': midpoint,
                   'half_difference_feedback': response}.items()}
    increments['newton_discrete_remainder'] = trapezoid(v, dt)-differences(actual_position)
    position_error = predicted_position-actual_position
    rebuilt = np.cumsum(sum(increments.values()), axis=0)
    np.testing.assert_allclose(rebuilt, position_error, rtol=0., atol=ATOL)
    position_norm = np.linalg.norm(actual_position[-1])
    floor = settings['floors_si']
    if position_norm < 10*max(floor['position'], 5*np.linalg.norm(half(nominal)[-1, :3])):
        raise ValueError('expected a strong Newton H32 position response for projection')
    axis = actual_position[-1]/position_norm
    position = {key: summarize(data, axis) for key, data in increments.items()}
    velocity_noise = np.linalg.norm(half(nominal)[..., VELOCITY], axis=-1)
    epsilon = 32*np.finfo(float).eps*np.maximum(1., np.linalg.norm(v, axis=-1))
    threshold = np.maximum(np.maximum(floor['velocity'], 5*velocity_noise), epsilon)
    return {'velocity_m_s': {
                'newton': v.tolist(), 'free': v11.tolist(), 'local': v00.tolist(),
                'local_error': local_error.tolist(), 'propagation': propagation.tolist(),
                'midpoint_feedback': midpoint.tolist(), 'half_difference_feedback': response.tolist(),
                'replacement_interaction': (v11-v10-v01+v00).tolist(), 'signal_floor': threshold.tolist()},
            'onset': {key: trace_error(v, predicted, threshold, settings['relative_error_limit'],
                                      settings['consecutive_leads'])
                      for key, predicted in (('free', v11), ('local', v00))},
            'position_m': {'newton': actual_position.tolist(), 'free': predicted_position.tolist(),
                           'error': position_error.tolist(), 'error_contributions': position},
            'max_velocity_vector_closure_m_s': float(abs(free_error-local_error-midpoint-response).max()),
            'max_position_vector_closure_m': float(abs(rebuilt-position_error).max())}


def run(plan, out, guard):
    def load(path):
        return json.loads(_verified_bytes(ROOT, path, plan['files_sha256'][Path(path).as_posix()]))

    def artifact(validation, report, name):
        path = validation.with_name(name)
        expected = plan['files_sha256'][path.as_posix()]
        if expected != report['artifacts_sha256'][name]:
            raise ValueError(f'acceptance hash does not match {path}')
        return load(path)

    old, new = load(FIT), load(CANDIDATE)
    if (any(x['status'] != 'completed' or x['updates'] != 900 or x['simulation_steps'] != 0
            or x['test_episode_files_read'] is not False for x in (old, new))
            or (new['reference_pairs'], new['training_pairs']) != (46, 49)
            or new['anchor_draws_sha256'] != old['anchor_draws_sha256']):
        raise ValueError('unexpected 46/49 fixed endpoints')
    baseline_scores = artifact(FIT, old, 'comparison.json')
    candidate_scores = artifact(CANDIDATE, new, 'comparison.json')
    parent = load(plan['held_design'])
    held = read_held(parent)
    dev_case, cfg, state_raw, traces, _ = held
    dev_state = _parse_state(state_raw)
    dev_initial = _ball_state(dev_state)
    contact_path = Path(plan['held_contacts'])
    contact_validation = contact_path.with_name('validation.json')
    contact_report = load(contact_validation)
    if (contact_report['status'] != 'completed' or contact_report['case'] != dev_case['case']
            or contact_report['physical_substeps'] != 0 or contact_report['test_episode_files_read'] is not False):
        raise ValueError('existing contact timeline is outside the frozen development scope')
    contacts = artifact(contact_validation, contact_report, contact_path.name)
    names = tuple(dev_case['directions'])
    capture = load(CAPTURE)
    if (capture['status'] != 'completed' or capture['training_updates'] != 0
            or capture['test_episode_files_read'] is not False or capture['pairs'] != 3):
        raise ValueError('advance capture is outside the frozen scope')
    manifest = CAPTURE.with_name('capture_manifest.json')
    artifact(CAPTURE, capture, manifest.name)
    data = prepare_timed_pairs(ROOT, manifest, capture['artifacts_sha256'][manifest.name])
    if data.directions != names or len(data.entries) != 1:
        raise ValueError('mismatched advance/held timing directions')
    entry, = data.entries
    refs = entry['nominal_branches']+[branch for pair in entry['pairs'] for branch in pair['branches']]
    raw = {}
    for ref in refs:
        path = (CAPTURE.parent/ref['path']).as_posix()
        if ref['sha256'] != plan['files_sha256'][path] or ref['sha256'] != capture['artifacts_sha256'][ref['path']]:
            raise ValueError('advance trace and acceptance hash mismatch')
        raw[ref['label']] = load(path)
    layout_path = CAPTURE.parent/'worker/native_layout.json'
    if capture['artifacts_sha256']['worker/native_layout.json'] != plan['files_sha256'][layout_path.as_posix()]:
        raise ValueError('advance native layout identity changed')
    advance_contacts = diagnose_contacts({**dev_case, 'case': {'tick': entry['snapshot_tick']}}, raw, load(layout_path))
    all_contacts = {'development': contacts, 'advance': advance_contacts}
    results = {'development': {name: {} for name in names}, 'advance': {name: {} for name in names}}
    guard('load', log=True)
    for label, path, validation, recorded in (('reference_46', FIT, old, baseline_scores),
                                               ('candidate_49', CANDIDATE, new, candidate_scores)):
        checkpoint = path.with_name('velocity_response_leads_900.npz')
        _verified_bytes(ROOT, checkpoint, validation['checkpoint_sha256'])
        with np.load(ROOT/checkpoint, allow_pickle=False) as saved:
            model = load_position_model(saved)
            if (type(model) is not TrapezoidPositionResidualMLP or int(saved['updates']) != 900
                    or tuple(saved['known_names']) != KNOWN_NAMES or model.dt_s != data.dt_s):
                raise ValueError('unexpected model architecture, endpoint or clock')
            norms = SimpleNamespace(known_mean=saved['known_mean'], known_scale=saved['known_scale'])
        dev_scores, dev_forecasts = score_held(model, held, norms, lambda: guard('development'))
        if dev_scores != recorded['held_timing']:
            raise ValueError('held development scores did not reproduce')
        for i, name in enumerate(names):
            signed = [traces[f'{name}_{side}'] for side in ('plus', 'minus')]
            features = np.stack([_inputs(model, dev_initial, dev_state.time_s, dev_state.tilt_rad,
                                 dev_state.tilt_rate_rad_s, row['actions'], cfg, norms.known_mean, norms.known_scale)[2]
                                 for row in signed])
            dev = diagnose_pair(model, dev_initial, features, np.stack([row['raw'] for row in signed]),
                np.stack([traces[key]['raw'] for key in ('nominal_1', 'nominal_2')]), plan['onset'])
            expected = dev_forecasts['pairs'][name]
            np.testing.assert_allclose(dev['velocity_m_s']['free'],
                half(np.stack([expected['plus'], expected['minus']]))[..., VELOCITY], rtol=0., atol=1e-12)
            results['development'][name][label] = dev

            b = data.batch
            features = (np.stack([b.plus_known[i], b.minus_known[i]])-norms.known_mean)/norms.known_scale
            advance = diagnose_pair(model, b.initial[i], features, np.stack([b.plus_target[i], b.minus_target[i]]),
                                    np.stack([b.baseline_first[i], b.baseline_second[i]]), plan['onset'])
            saved = candidate_scores['training_timed'][label]['advance']['pairs'][i]['by_lead']['32']['half_difference']['velocity']
            for key in ('free', 'local'):
                np.testing.assert_allclose(advance['velocity_m_s'][key][-1], saved[f'{key}_half_difference_si'],
                                           rtol=0., atol=1e-12)
            results['advance'][name][label] = advance
            guard('decomposition')
    write_json(out/'contributions.json', {'bands': BANDS, 'settings': plan['onset'], 'cases': results,
               'decomposition': 'Mean of two replacement orders. Difference vectors precede norms and projections.'})
    write_json(out/'contacts.json', all_contacts)
    return {'checkpoints': 2, 'pair_trajectories': 6, 'leads_per_pair': 32,
            'max_velocity_closure_m_s': max(row['max_velocity_vector_closure_m_s'] for case in results.values()
                                           for pair in case.values() for row in pair.values()),
            'max_position_closure_m': max(row['max_position_vector_closure_m'] for case in results.values()
                                          for pair in case.values() for row in pair.values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'P4', run)
