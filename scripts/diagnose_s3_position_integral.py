"""Attribute the frozen H32 position responses to tool-frame velocity by lead."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from train_s3_position_integral import ROOT, REFERENCE, digest, read_held, score_held, write_json

import numpy as np

from ball_retention.action_grad import _inputs
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES, _ball_state
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.resources import directory_bytes


OUT = ROOT/'runs/s3_position_integral_diagnosis_20261115'
FIT = Path('runs/s3_position_integral_fit_20261115')
COVERAGE = Path('runs/s3_timing_coverage_fit_20261115_resume')
INPUTS = {
    (FIT/'validation.json').as_posix(): 'e050b747c3feb1f22777bbf6d62975a788fb581b464ed45894ffba50c7d628f4',
    (COVERAGE/'validation.json').as_posix(): '73ba0d9b301099beaf5c3f630fe1b425a6b1826db7277772b5c0a4ebfe6fbfe7',
}
BANDS = [(1, 8), (9, 16), (17, 24), (25, 32)]
ATOL = 1e-14
LIMITS = {'processes': 1, 'cpu_threads': 1, 'wall_s': 120, 'rss_gib': 1, 'private_commit_gib': 1.5, 'output_mib': 16}
SCOPE = {'training_updates': 0, 'simulation_steps': 0, 'test_episode_files_read': False}


def differences(series):
    return np.diff(series, axis=0, prepend=np.zeros_like(series[:1]))


def trapezoid(velocity, dt):
    before = np.vstack((np.zeros_like(velocity[:1]), velocity[:-1]))
    return dt*.5*(before+velocity)


def project(vector, unit):
    dot = float(vector@unit)
    return {'vector': vector.tolist(), 'norm': float(np.linalg.norm(vector)), 'projection_on_newton_h32': dot,
            'orthogonal_norm': float(np.linalg.norm(vector-dot*unit))}


def summarize(increments, unit):
    bands = [increments[a-1:b].sum(axis=0) for a, b in BANDS]
    total = increments.sum(axis=0)
    np.testing.assert_allclose(np.sum(bands, axis=0), total, rtol=0., atol=ATOL)
    return {'total': project(total, unit),
            'by_band': {f'{a}-{b}': project(vector, unit) for (a, b), vector in zip(BANDS, bands, strict=True)}}


def forecast(checkpoint, data, frozen, held, guard):
    with np.load(BytesIO(data), allow_pickle=False) as saved:
        if tuple(saved['known_names']) != KNOWN_NAMES or int(saved['updates']) != 900:
            raise ValueError(f'{checkpoint} has an unexpected feature order or endpoint')
        model = load_position_model(saved)
        norms = SimpleNamespace(known_mean=saved['known_mean'], known_scale=saved['known_scale'])
    scores, recomputed = score_held(model, held, norms, guard)
    for key in ('initial_state_raw_si', 'nominal', 'fd_step', 'amplitude'):
        np.testing.assert_array_equal(recomputed[key], frozen[key], err_msg=f'{checkpoint}: {key}')
    if recomputed['pairs'].keys() != frozen['pairs'].keys():
        raise ValueError(f'{checkpoint} direction set changed')
    for label, pair in recomputed['pairs'].items():
        for name, array in pair.items():
            np.testing.assert_array_equal(array, frozen['pairs'][label][name], err_msg=f'{checkpoint}: {label}.{name}')

    _, cfg, fields, traces, _ = held
    state = _parse_state(fields)
    initial = _ball_state(state)
    local = {}
    for direction, pair in recomputed['pairs'].items():
        for sign in ('plus', 'minus'):
            trace = traces[f'{direction}_{sign}']
            measured_prior = np.vstack((initial, trace['raw'][:-1]))
            _, _, features = _inputs(model, initial, state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                                     trace['actions'], cfg, norms.known_mean, norms.known_scale)
            predicted = model.denormalize(model.step(model.normalize(measured_prior), features))
            np.testing.assert_allclose(predicted[0], pair[sign][0], rtol=0., atol=2e-14)
            if not np.isfinite(predicted).all():
                raise ValueError(f'nonfinite conditional prediction for {direction}_{sign}')
            local[direction, sign] = predicted
        guard()
    return model, norms, recomputed, local, scores


def pair_analysis(name, held, forecasts, dt):
    capture, _, _, traces, _ = held
    plus, minus = (traces[f'{name}_{sign}']['raw'] for sign in ('plus', 'minus'))
    truth = (plus-minus)*.5
    h32 = truth[-1, :3]
    noise = np.linalg.norm((traces['nominal_1']['raw'][-1, :3]-traces['nominal_2']['raw'][-1, :3])*.5)
    gates = capture['gates']
    if np.linalg.norm(h32) < gates['strong_multiplier']*max(gates['position_floor_m'], gates['noise_multiplier']*noise):
        raise ValueError(f'{name} has no strong H32 projection axis')
    unit = h32/np.linalg.norm(h32)
    truth_position = differences(truth[:, :3])
    truth_velocity = truth[:, 3:6]
    truth_integrated = trapezoid(truth_velocity, dt)
    truth_remainder = truth_position-truth_integrated
    rows = {}
    closure = 0.
    for arm, (model, _, forecast, local, _) in forecasts.items():
        pair = forecast['pairs'][name]
        free = (np.asarray(pair['plus'])-pair['minus'])*.5
        jvp = capture['amplitude']*np.asarray(pair['directional_jvp'])
        finite_velocity = free[:, 3:6]
        local_velocity = (local[name, 'plus']-local[name, 'minus'])[:, 3:6]*.5
        inherited_velocity = np.vstack((np.zeros(3), truth_velocity[:-1]))
        local_error = local_velocity-truth_velocity
        propagated_error = finite_velocity-local_velocity
        np.testing.assert_allclose(finite_velocity-truth_velocity, local_error+propagated_error, rtol=0., atol=ATOL)
        reported = {}
        for key, path in (('finite', free), ('scaled_jvp', jvp)):
            position = differences(path[:, :3])
            velocity = trapezoid(path[:, 3:6], dt)
            remainder = position-velocity
            if isinstance(model, TrapezoidPositionResidualMLP):
                error = float(np.max(abs(np.cumsum(velocity, axis=0)-path[:, :3])))
                np.testing.assert_allclose(np.cumsum(velocity, axis=0), path[:, :3], rtol=0., atol=ATOL)
                closure = max(closure, error)
            reported[key] = {'position_increment_m': summarize(position, unit),
                             'velocity_integral_m': summarize(velocity, unit),
                             'position_minus_integral_m': summarize(remainder, unit),
                             'h32_relative_error_vs_newton_finite': float(np.linalg.norm(path[-1, :3]-h32)/np.linalg.norm(h32))}

        one_step_integrated = trapezoid(local_error, dt)
        propagated_integrated = trapezoid(propagated_error, dt)
        position_difference = differences(free[:, :3])-truth_position
        remainder_difference = reported['finite']['position_minus_integral_m']
        model_remainder = differences(free[:, :3])-trapezoid(finite_velocity, dt)
        rebuilt = one_step_integrated+propagated_integrated+model_remainder-truth_remainder
        np.testing.assert_allclose(position_difference, rebuilt, rtol=0., atol=ATOL)
        closure = max(closure, float(np.max(abs(position_difference-rebuilt))))
        velocity_errors = {'local': local_error, 'propagated': propagated_error, 'free': finite_velocity-truth_velocity}
        rows[arm] = {**reported, 'model_minus_newton_finite_position_error_m': {
                         'total': summarize(position_difference, unit),
                         'local_velocity_integral': summarize(one_step_integrated, unit),
                         'propagated_velocity_integral': summarize(propagated_integrated, unit),
                         'model_remainder': remainder_difference,
                         'minus_newton_remainder': summarize(-truth_remainder, unit)},
                     'finite_velocity_m_s_by_lead': {
                         'free': finite_velocity.tolist(), 'local': local_velocity.tolist(),
                         'local_increment': (local_velocity-inherited_velocity).tolist(),
                         **{f'{key}_error': value.tolist() for key, value in velocity_errors.items()}},
                     'finite_velocity_error_rmse_m_s_by_band': {
                         f'{a}-{b}': {key: float(np.sqrt(np.mean(np.sum(value[a-1:b]**2, axis=1))))
                                      for key, value in velocity_errors.items()} for a, b in BANDS}}

    return {'newton_h32_position_unit': unit.tolist(), 'newton_h32_position_signal_m': float(np.linalg.norm(h32)),
            'newton': {'position_increment_m': summarize(truth_position, unit),
                       'velocity_integral_m': summarize(truth_integrated, unit),
                       'position_minus_integral_m': summarize(truth_remainder, unit),
                       'finite_velocity_m_s_by_lead': truth_velocity.tolist(),
                       'finite_velocity_increment_m_s_by_lead': differences(truth_velocity).tolist()},
            'models': rows, 'max_closure_error_m': closure}


def run(out):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    inputs = dict(INPUTS)

    def verify(path, expected):
        name = Path(path).as_posix()
        raw = _verified_bytes(ROOT, name, expected)
        inputs[name] = expected
        return raw

    def read(path, expected):
        return json.loads(verify(path, expected))

    def guard():
        memory = process_memory(os.getpid())
        if (time.monotonic()-began >= LIMITS['wall_s'] or memory['peak_rss_bytes'] > LIMITS['rss_gib']*1024**3
                or memory['peak_private_commit_bytes'] > LIMITS['private_commit_gib']*1024**3
                or directory_bytes(out) > LIMITS['output_mib']*1024**2):
            raise RuntimeError('position integration diagnosis exceeded resource caps')
        return memory

    try:
        guard()
        candidate, reference = (read(d/'validation.json', INPUTS[(d/'validation.json').as_posix()]) for d in (FIT, COVERAGE))
        if (any(v['status'] != 'completed' or v['test_episode_files_read'] or v['simulation_steps'] for v in (candidate, reference))
                or candidate['updates'] != 900 or reference['updates'][REFERENCE] != 900
                or candidate['anchor_draws_sha256'] != reference['identical_anchor_draws_sha256']):
            raise ValueError('unexpected frozen fit endpoint scope')
        design = read(COVERAGE/'execution_plan.json', reference['artifacts_sha256']['execution_plan.json'])['design']
        held_path = Path(design['held_timing_run'])
        read(held_path, design['inputs_sha256'][held_path.as_posix()])
        held = read_held(design)
        for label in held[3]:
            name = f'worker/{label}.json'
            inputs[(held_path.parent/name).as_posix()] = held[-1]['artifacts_sha256'][name]
        models = {}
        for arm, folder, result, checkpoint, trajectory in (
            ('position_integral', FIT, candidate, 'position_integral_900.npz', 'timed_rollouts.json'),
            ('matched_46', COVERAGE, reference, f'{REFERENCE}_900.npz', f'{REFERENCE}_timing_rollouts.json'),
        ):
            files = result['artifacts_sha256']
            saved = read(folder/trajectory, files[trajectory])
            if saved['checkpoint_sha256'] != files[checkpoint]:
                raise ValueError('saved rollouts and checkpoint hash disagree')
            model = forecast(checkpoint, verify(folder/checkpoint, files[checkpoint]), saved, held, guard)
            recorded = read(folder/'comparison.json', files['comparison.json'])['held_timing']
            if arm == 'matched_46':
                recorded = recorded[REFERENCE]
            for name, row in recorded.items():
                for lead, values in row['by_lead'].items():
                    if model[-1][name]['by_lead'][lead] != values:
                        raise ValueError(f'{arm} metric did not reproduce at {name} H{lead}')
            models[arm] = model
        fit, fit_norms, *_ = models['position_integral']
        baseline, base_norms, *_ = models['matched_46']
        if not isinstance(fit, TrapezoidPositionResidualMLP) or isinstance(baseline, TrapezoidPositionResidualMLP):
            raise ValueError('frozen model types changed')
        for key in ('state_mean', 'state_scale', 'dt_s'):
            np.testing.assert_array_equal(getattr(fit, key), getattr(baseline, key))
        for key in ('known_mean', 'known_scale'):
            np.testing.assert_array_equal(getattr(fit_norms, key), getattr(base_norms, key))

        findings = {name: pair_analysis(name, held, models, fit.dt_s) for name in held[0]['directions']}
        write_json(out/'velocity_contributions.json', {'case': held[0]['case'], 'dt_s': fit.dt_s, 'bands': BANDS,
                   'amplitude': held[0]['amplitude'], 'directions': findings,
                   'convention': 'Project onto the actual Newton finite H32 position vector. Scale model JVP by amplitude. '
                                 'Local velocity comes from independent steps with each measured pre-step state and causal features. '
                                 'Its trapezoid integral is error accounting, not a simulated or teacher-forced path. '
                                 'Phase bands partition integration time and do not identify when a request injected the response.'})
        for name, expected in dict(inputs).items():
            _verified_bytes(ROOT, name, expected)
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline analysis imported physics')
        source = Path(__file__).resolve()
        write_json(out/'validation.json', {'status': 'completed', 'limits': LIMITS, **SCOPE,
                   'wall_s': time.monotonic()-began, **guard(), 'recorded_rollouts_and_metrics_reproduced': True,
                   'max_closure_error_m': max(row['max_closure_error_m'] for row in findings.values()),
                   'inputs_sha256': inputs, 'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted({
                       source, ROOT/'scripts/train_s3_position_integral.py', ROOT/'scripts/train_s3_timing_coverage.py',
                       ROOT/'diagnostics/resource_probe/deterministic_response_metrics.py', *(ROOT/'src/ball_retention').glob('*.py')})},
                   'artifacts_sha256': {'velocity_contributions.json': digest(out/'velocity_contributions.json')}})
        guard()
    except BaseException as error:
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error), 'wall_s': time.monotonic()-began, **SCOPE})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    target = parser.parse_args().output.resolve()
    if len(target.relative_to(ROOT/'runs').parts) != 1:
        raise ValueError('choose a new run directory immediately under runs')
    run(target)
