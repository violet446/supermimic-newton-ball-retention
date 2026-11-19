"""Split propagated velocity error by the source of the paired pre-step state."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from diagnose_s3_position_integral import (
    ATOL, BANDS, COVERAGE, FIT, LIMITS, REFERENCE, ROOT, SCOPE, digest, forecast,
    read_held, summarize, trapezoid, write_json,
)

import numpy as np

from ball_retention.action_grad import _inputs
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.guard import process_memory
from ball_retention.model_data import _ball_state
from ball_retention.resources import directory_bytes


SOURCE = Path('runs/s3_position_integral_diagnosis_20261115')
SOURCE_SHA256 = '65e929fe6fd54b04a888bd33624a6c803d8e79aefb67df6e0e003ae73ca0b653'
OUT = ROOT/'runs/s3_pair_state_diagnosis_20261115'
STATE_ATOL = 1e-12
CHECK_KEYS = ('max_endpoint_error_si', 'max_velocity_closure_error_m_s', 'max_integrated_closure_error_m')


def describe_pair(name, held, arm, frozen):
    model, norms, predicted, local, _ = arm
    _, cfg, state, traces, _ = held
    start = _parse_state(state)
    initial = _ball_state(start)
    paths = [traces[f'{name}_{sign}'] for sign in ('plus', 'minus')]
    measured = np.asarray([row['raw'] for row in paths])
    free = np.asarray([predicted['pairs'][name][sign] for sign in ('plus', 'minus')])
    conditional = np.asarray([local[name, sign] for sign in ('plus', 'minus')])
    initial_pair = np.broadcast_to(initial, (2, 1, 9))
    measured_prior, free_prior = (np.concatenate((initial_pair, states[:, :-1]), axis=1) for states in (measured, free))
    measured_midpoint, free_midpoint = (states.mean(axis=0) for states in (measured_prior, free_prior))
    measured_half, free_half = ((states[0]-states[1])*.5 for states in (measured_prior, free_prior))
    features = np.stack([_inputs(model, initial, start.time_s, start.tilt_rad, start.tilt_rate_rad_s,
                                 branch['actions'], cfg, norms.known_mean, norms.known_scale)[2] for branch in paths])

    def next_state(midpoint, half):
        state = np.stack((midpoint+half, midpoint-half))
        post = model.denormalize(model.step(model.normalize(state), features))
        if not np.isfinite(post).all():
            raise FloatingPointError('nonfinite independent step on reconstructed input')
        return post

    from_measured = next_state(measured_midpoint, measured_half)
    from_free = next_state(free_midpoint, free_half)
    np.testing.assert_allclose(from_measured, conditional, atol=STATE_ATOL, rtol=0.)
    np.testing.assert_allclose(from_free, free, atol=STATE_ATOL, rtol=0.)
    mixed = {
        'measured': from_measured,
        'predicted_midpoint_only': next_state(free_midpoint, measured_half),
        'predicted_half_difference_only': next_state(measured_midpoint, free_half),
        'predicted': from_free,
    }
    velocity = {key: (states[0, :, 3:6]-states[1, :, 3:6])*.5 for key, states in mixed.items()}
    base = velocity['measured']
    mid = velocity['predicted_midpoint_only']
    half = velocity['predicted_half_difference_only']
    end = velocity['predicted']
    total = end-base
    midpoint_first, midpoint_second = mid-base, end-half
    half_first, half_second = half-base, end-mid
    for reconstruct in (midpoint_first+half_second, half_first+midpoint_second):
        np.testing.assert_allclose(reconstruct, total, rtol=0., atol=ATOL)
    midpoint = .5*(midpoint_first+midpoint_second)
    half_difference = .5*(half_first+half_second)
    np.testing.assert_allclose(midpoint+half_difference, total, rtol=0., atol=ATOL)
    np.testing.assert_allclose(total, frozen['finite_velocity_m_s_by_lead']['propagated_error'], rtol=0., atol=ATOL)
    for array in velocity.values():
        np.testing.assert_allclose(array[0], base[0], rtol=0., atol=ATOL)
    np.testing.assert_allclose(np.stack((midpoint[0], half_difference[0])), 0., rtol=0., atol=ATOL)

    truth_h32 = .5*(measured[0, -1, :3]-measured[1, -1, :3])
    unit = truth_h32/np.linalg.norm(truth_h32)
    integrated = {key: trapezoid(value, model.dt_s) for key, value in {
        'total_propagated': total, 'midpoint': midpoint, 'half_difference': half_difference,
        'midpoint_when_replaced_first': midpoint_first, 'midpoint_when_replaced_second': midpoint_second,
        'half_difference_when_replaced_first': half_first, 'half_difference_when_replaced_second': half_second,
        'interaction': midpoint_second-midpoint_first,
    }.items()}
    whole = integrated['total_propagated']
    np.testing.assert_allclose(integrated['midpoint']+integrated['half_difference'], whole, rtol=0., atol=ATOL)
    frozen_integral = frozen['model_minus_newton_finite_position_error_m']['propagated_velocity_integral']
    blocks = {key: summarize(value, unit) for key, value in integrated.items()}
    for at, value in blocks['total_propagated']['by_band'].items():
        np.testing.assert_allclose(value['vector'], frozen_integral['by_band'][at]['vector'], rtol=0., atol=ATOL)
    np.testing.assert_allclose(whole.sum(axis=0), frozen_integral['total']['vector'], rtol=0., atol=ATOL)
    propagated = whole.sum(axis=0)
    denom = float(propagated@propagated)
    shares = {key: float(propagated@integrated[key].sum(axis=0))/denom if denom else None
              for key in ('midpoint', 'half_difference')}
    if denom:
        np.testing.assert_allclose(sum(shares.values()), 1., rtol=0., atol=ATOL)
    return {'integrated_velocity_contributions_m': blocks, 'h32_signed_share_of_propagated_error': shares,
            'post_step_velocity_half_difference_m_s_by_lead': {key: value.tolist() for key, value in velocity.items()},
            'velocity_contributions_m_s_by_lead': {'midpoint': midpoint.tolist(), 'half_difference': half_difference.tolist()},
            'checks': {'max_endpoint_error_si': float(max(np.max(abs(from_measured-conditional)), np.max(abs(from_free-free)))),
                       'max_velocity_closure_error_m_s': float(np.max(abs(total-midpoint-half_difference))),
                       'max_integrated_closure_error_m': float(np.max(abs(whole-integrated['midpoint']-integrated['half_difference'])))}}


def run(out):
    started = time.monotonic()
    out.mkdir(exist_ok=False)

    def guard():
        memory = process_memory(os.getpid())
        if (time.monotonic()-started >= LIMITS['wall_s'] or memory['peak_rss_bytes'] > LIMITS['rss_gib']*1024**3
                or memory['peak_private_commit_bytes'] > LIMITS['private_commit_gib']*1024**3
                or directory_bytes(out) > LIMITS['output_mib']*1024**2):
            raise RuntimeError('pair-state diagnosis exceeded a resource cap')
        return memory

    try:
        guard()
        source = SOURCE/'validation.json'
        accepted = json.loads(_verified_bytes(ROOT, source, SOURCE_SHA256))
        if (accepted['status'] != 'completed' or not accepted['recorded_rollouts_and_metrics_reproduced']
                or any(accepted[key] != value for key, value in SCOPE.items())):
            raise ValueError('unexpected integration diagnostic acceptance')
        prior = SOURCE/'velocity_contributions.json'
        inputs = {**accepted['inputs_sha256'], source.as_posix(): SOURCE_SHA256,
                  prior.as_posix(): accepted['artifacts_sha256'][prior.name]}
        for key, sha in {**inputs, **accepted['source_sha256']}.items():
            _verified_bytes(ROOT, key, sha)

        def load(path):
            return _verified_bytes(ROOT, path, inputs[path.as_posix()])

        earlier = json.loads(load(prior))
        parent = json.loads(load(COVERAGE/'execution_plan.json'))['design']
        held = read_held(parent)
        if earlier['case'] != held[0]['case'] or earlier['bands'] != [list(band) for band in BANDS]:
            raise ValueError('integration report uses a different held snapshot or lead partition')
        arms = {}
        for arm, folder, checkpoint, trajectory in (
            ('position_integral', FIT, 'position_integral_900.npz', 'timed_rollouts.json'),
            ('matched_46', COVERAGE, f'{REFERENCE}_900.npz', f'{REFERENCE}_timing_rollouts.json'),
        ):
            frozen = json.loads(load(folder/trajectory))
            if frozen['checkpoint_sha256'] != inputs[(folder/checkpoint).as_posix()]:
                raise ValueError('frozen trajectory is bound to a different checkpoint')
            arms[arm] = forecast(checkpoint, load(folder/checkpoint), frozen, held, guard)

        results = {}
        for name in held[0]['directions']:
            frozen = earlier['directions'][name]['models']
            results[name] = {key: describe_pair(name, held, model, frozen[key]) for key, model in arms.items()}
            guard()
        path = out/'pair_state_contributions.json'
        write_json(path, {'case': held[0]['case'], 'dt_s': earlier['dt_s'], 'bands': BANDS, 'directions': results,
                   'method': 'Mean of two replacement orders in one-step model evaluations with fixed signed tool features. '
                             'Each evaluation starts from a constructed pre-step state, with no feedback between leads. '
                             'Shares are signed projections onto the total integrated propagated error divided by its norm.'})
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline analysis imported physics')
        key = Path(__file__).resolve()
        write_json(out/'validation.json', {'status': 'completed', 'limits': LIMITS, 'wall_s': time.monotonic()-started,
                   **guard(), **SCOPE, 'inputs_sha256': inputs,
                   'source_sha256': {**accepted['source_sha256'], key.relative_to(ROOT).as_posix(): digest(key)},
                   'max_errors': {key: max(arm['checks'][key] for part in results.values() for arm in part.values())
                                  for key in CHECK_KEYS},
                   'artifacts_sha256': {path.name: digest(path)}})
        guard()
    except BaseException as error:
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error), 'wall_s': time.monotonic()-started, **SCOPE})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    target = parser.parse_args().output.resolve()
    if len(target.relative_to(ROOT/'runs').parts) != 1:
        raise ValueError('choose a new run directory immediately under runs')
    run(target)
