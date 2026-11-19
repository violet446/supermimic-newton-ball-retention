"""Locate the response tradeoffs of the two fixed timing-coverage endpoints."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from train_s3_timing_coverage import ROOT, digest, read_held, timed_score, write_json
from ball_retention.action_grad import _inputs
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.deterministic_response_model import evaluate_model
from ball_retention.guard import process_memory
from ball_retention.model_data import _ball_state
from ball_retention.pair_data import DEFAULT_RESPONSE_FLOOR, _snapshot_rows
from ball_retention.resources import directory_bytes
from ball_retention.response_error_diagnosis import diagnose_responses
from ball_retention.world_model import ResidualMLP


FIT = Path('runs/s3_timing_coverage_fit_20261115_resume')
OUT = ROOT/'runs/s3_timing_coverage_diagnosis_20261115'
LEGACY_PROTOCOL = 'configs/s3_action_response_diagnosis_plan.json'
LEGACY_PROTOCOL_SHA256 = '126a7f867e6b813906a42f55a1bf6200ed10ab06ac7a525d5007dea0281dfc3d'
VALIDATION_SHA256 = '73ba0d9b301099beaf5c3f630fe1b425a6b1826db7277772b5c0a4ebfe6fbfe7'
ARMS = ('held_offsets', 'held_offsets_plus_timing')
GROUP = 'history_validation_yminus006'
LIMITS = {'max_wall_s': 120, 'max_process_rss_gib': 1, 'max_process_commit_gib': 1.5, 'max_new_output_mib': 16}


def read_changed_pair(comparison):
    def eligible(arm):
        result = {}
        for row in comparison['legacy_development'][arm]['pairs']:
            if row['group'] != GROUP or row['amplitude'] != .2:
                continue
            metric = row['response']['by_lead']['32']['position']
            if metric['newton_half_spread'] >= 10.*metric['signal_threshold']:
                key = tuple(row[k] for k in ('episode', 'tick', 'axis', 'amplitude'))
                result[key] = metric
        return result

    before, after = map(eligible, ARMS)
    if before.keys() != after.keys() or len(before) != 6:
        raise ValueError('the six strong legacy pairs must be identical across arms')
    def positive(record):
        return record['cosine'] is not None and record['cosine'] > 0.

    lost, = [key for key in before if positive(before[key]) and not positive(after[key])]
    if sum(map(positive, before.values())) != 4 or sum(map(positive, after.values())) != 3:
        raise ValueError('changed direction does not account for the saved protection failure')
    return lost, {ARMS[0]: before[lost], ARMS[1]: after[lost]}


def legacy_case(plan, capture, key, verify):
    episode, tick, axis, amplitude = key
    base_path = Path(plan['base_collection'])
    collection = verify(base_path, plan['inputs_sha256'][base_path.as_posix()])
    history, = [row for row in collection['sources'] if row['cohort'] == 'history']
    source = Path(history['manifest'])
    manifest = verify(source, history['sha256'])
    entry, = [row for row in manifest['entries'] if row['episode_id'] == episode]
    if entry['split'] != 'validation' or entry['group_id'] != GROUP:
        raise ValueError('legacy pair is outside the targeted validation group')
    cfg = verify(source.parent/entry['config_path'], entry['config_sha256'])
    records = verify(source.parent/entry['path'], entry['sha256'], lines=True)
    record, = [row for row in records if row['snapshot_tick'] == tick]
    name = 'held_roll' if axis == 0 else 'held_pitch'
    analysis = {**capture, 'case': {'episode': episode, 'group': GROUP, 'tick': tick},
                'directions': {name: None}, 'branch_order': ['nominal_1', 'nominal_2', f'{name}_plus', f'{name}_minus']}
    dt, substeps, horizon = manifest['dt_s'], manifest['control_substeps'], manifest['horizon']
    row, = [row for row in _snapshot_rows(record, entry, cfg, horizon, amplitude, dt, substeps,
                                        np.repeat(DEFAULT_RESPONSE_FLOOR, 3), 5., [amplitude]) if row['axes'] == axis]
    if not all(np.all(row[k]) for k in ('plus_mask', 'minus_mask', 'baseline_mask')):
        raise ValueError('legacy diagnostic requires all 32 leads for the selected pair')
    nominal = row['nominal_actions']
    direction = np.broadcast_to(np.eye(2)[axis], nominal.shape)
    traces = {label: {'actions': nominal+sign*amplitude*direction, 'raw': row[target]}
              for label, sign, target in zip(analysis['branch_order'], (0, 0, 1, -1),
                                             ('baseline_first', 'baseline_second', 'plus_target', 'minus_target'), strict=True)}
    return analysis, cfg, row['initial_state'], traces, {name: direction}


def diagnose(capture, cfg, state, traces, forecasts, saved, guard):
    model = ResidualMLP.from_state_dict(saved)
    start = _parse_state(state)
    initial = _ball_state(start)
    np.testing.assert_array_equal(initial, forecasts['initial_state_raw_si'])
    expected = {n: forecasts['nominal'] for n in ('nominal_1', 'nominal_2')}
    expected.update({f'{name}_{sign}': row[sign] for name, row in forecasts['pairs'].items() for sign in ('plus', 'minus')})
    truth, free, one_step = {}, {}, {}
    for label, trace in traces.items():
        target = trace['raw']
        zero, _, features = _inputs(model, initial, start.time_s, start.tilt_rad, start.tilt_rate_rad_s,
                                    trace['actions'], cfg, saved['known_mean'], saved['known_scale'])
        prediction = model.denormalize(model.rollout(zero[None], features[None]))[0]
        np.testing.assert_array_equal(prediction, expected[label])
        truth[label], free[label] = target, prediction
        pre_state = np.vstack((initial, target[:-1]))
        one_step[label] = model.denormalize(model.step(model.normalize(pre_state), features))
        np.testing.assert_allclose(one_step[label][0], prediction[0], atol=2e-14, rtol=0.)
        guard()
    return diagnose_responses({**capture, 'report_leads': [1, 8, 16, 24, 32], 'onset_consecutive_leads': 3},
                              truth, free, one_step, forecasts, initial)


def run(out):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    inputs = {}

    def verify(name, expected, *, lines=False):
        path = Path(name).as_posix()
        raw = _verified_bytes(ROOT, path, expected)
        inputs[path] = expected
        return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)

    def guard():
        memory = process_memory(os.getpid())
        if (time.monotonic()-began >= LIMITS['max_wall_s'] or memory['peak_rss_bytes'] > LIMITS['max_process_rss_gib']*1024**3
                or memory['peak_private_commit_bytes'] > LIMITS['max_process_commit_gib']*1024**3
                or directory_bytes(out) > LIMITS['max_new_output_mib']*1024**2):
            raise RuntimeError('offline timing comparison resource limit reached')
        return memory

    try:
        accepted = verify(FIT/'validation.json', VALIDATION_SHA256)
        if (accepted['status'] != 'completed' or not accepted['control_reproduced_exactly']
                or list(accepted['updates']) != list(ARMS) or list(accepted['updates'].values()) != [900, 900]
                or accepted['test_episode_files_read'] is not False or accepted['simulation_steps'] != 0):
            raise ValueError('the source fit acceptance disagrees with this offline comparison')
        files = accepted['artifacts_sha256']
        plan = verify(FIT/'execution_plan.json', files['execution_plan.json'])['design']
        prior = verify(FIT/'comparison.json', files['comparison.json'])
        legacy_protocol = verify(LEGACY_PROTOCOL, LEGACY_PROTOCOL_SHA256)
        key, signs = read_changed_pair(prior)
        verify(plan['held_timing_run'], plan['inputs_sha256'][plan['held_timing_run']])
        held = read_held(plan)
        capture, cfg, state, traces, _ = held
        for label, trace in held[-1]['artifacts_sha256'].items():
            if label in {f'worker/{name}.json' for name in traces}:
                inputs[(Path(plan['held_timing_run']).parent/label).as_posix()] = trace
        old, old_cfg, old_state, old_traces, directions = legacy_case(plan, capture, key, verify)
        losses, timing = {}, {}
        for arm in ARMS:
            ck = FIT/f'{arm}_900.npz'
            raw = _verified_bytes(ROOT, ck, accepted['checkpoint_sha256'][arm])
            inputs[ck.as_posix()] = accepted['checkpoint_sha256'][arm]
            spec = {**{name: legacy_protocol[name] for name in ('fd_step', 'fd_rtol', 'fd_atol_si')},
                    'checkpoint': ck.as_posix(), 'checkpoint_sha256': inputs[ck.as_posix()],
                    'amplitude': key[-1]}
            old_prediction = evaluate_model(ROOT, spec, old_cfg, old_state, old_traces['nominal_1']['actions'], directions, guard)
            new_scores, new_prediction = timed_score(ROOT/ck, held, guard)
            frozen_name = f'{arm}_timing_rollouts.json'
            if new_prediction != verify(FIT/frozen_name, files[frozen_name]) or new_scores != prior['held_timing'][arm]:
                raise ValueError('timed free rollouts or responses differ from the frozen fit report')
            with np.load(BytesIO(raw), allow_pickle=False) as saved:
                losses[arm] = diagnose(old, old_cfg, old_state, old_traces, old_prediction, saved, guard)
                timing[arm] = diagnose(capture, cfg, state, traces, new_prediction, saved, guard)
            metric = next(iter(losses[arm]['directions'].values()))['by_lead']['32']['position']
            np.testing.assert_allclose(metric['comparisons']['free']['cosine'], signs[arm]['cosine'], atol=1e-10, rtol=0.)
            np.testing.assert_allclose(np.array(metric['vectors']['free'])/key[-1], signs[arm]['model_derivative'], atol=1e-12, rtol=0.)
        write_json(out/'legacy_flip.json', {'pair': dict(zip(('episode', 'tick', 'axis', 'amplitude'), key)),
                   'saved_h32': signs, 'arms': losses})
        write_json(out/'held_timing_diagnosis.json', {'case': capture['case'], 'arms': timing})
        for name, expected in inputs.items():
            _verified_bytes(ROOT, name, expected)
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline diagnosis loaded physics')
        sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted({Path(__file__).resolve(),
            ROOT/'scripts/train_s3_timing_coverage.py', ROOT/'scripts/preflight_s3_deterministic_action_response.py',
            *(ROOT/'src/ball_retention').glob('*.py'), *(ROOT/'diagnostics/resource_probe').glob('*.py')})}
        write_json(out/'validation.json', {'status': 'completed', 'limits': LIMITS, 'wall_s': time.monotonic()-began,
                   **guard(), 'simulation_steps': 0, 'training_updates': 0, 'test_episode_files_read': False,
                   'inputs_sha256': inputs, 'source_sha256': sources,
                   'artifacts_sha256': {p.name: digest(p) for p in out.iterdir()}})
    except BaseException as error:
        write_json(out/'stopped.json', {'reason': str(error), 'wall_s': time.monotonic()-began,
                   'simulation_steps': 0, 'training_updates': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    path = parser.parse_args().output.resolve()
    if not path.relative_to(ROOT/'runs').parts:
        raise ValueError('choose a fresh run subdirectory')
    run(path)
