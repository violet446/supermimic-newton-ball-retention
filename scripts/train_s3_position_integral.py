"""Fit and score the single frozen tool-frame trapezoid candidate."""
import argparse
import ctypes
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sys
import time

from preflight_s3_position_integral import (
    CONTROL, DESIGN, DESIGN_SHA256, REFERENCE, ROOT, action_sequences, digest, dormant_check,
    prepare_recipe, read_held, write_json,
)

import numpy as np

from train_s3_timing_coverage import primary, regression, report
from train_s3_timing_lead_weights import checks_for
from ball_retention.action_grad import _inputs, predict_ball
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.deterministic_response_model import _directional_rollout
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES, _ball_state
from ball_retention.pair_loss import paired_loss
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.resources import MemoryStatus, directory_bytes
from ball_retention.world_model import Adam
from deterministic_response_metrics import response_metrics


PREFLIGHT = 'runs/s3_position_integral_preflight_20261115_verified/preflight.json'
PREFLIGHT_SHA256 = 'c8eb095bf729e0a953e4fd2103c8245a14792026bf32c373066c6d96b6bf095e'
OUT = ROOT/'runs/s3_position_integral_fit_20261115'
LEADS = (1, 8, 16, 24, 32)
HELD_ROLL = ('history_validation_yminus006_advance', 72, 0, .2)
SCOPE = {'simulation_steps': 0, 'test_episode_files_read': False, 'control_accepted': False}


def fit(model, fold, cfg, scales, plan, out, track):
    """Use the original coverage optimizer and reduction order with the new transition."""
    def standardize(values):
        return (values-fold.known_mean)/fold.known_scale

    anchors = {h: (model.normalize(w.initial), standardize(w.known), model.normalize(w.target))
               for h, w in fold.anchors.items()}
    batch = fold.pairs
    initial = model.normalize(batch.initial)
    known = standardize(np.stack((batch.plus_known, batch.minus_known)))
    targets = np.stack((batch.plus_target, batch.minus_target))
    masks = np.stack((batch.plus_mask, batch.minus_mask))
    lead_weights = np.zeros(known.shape[2])
    spec = cfg['pairs']
    lead_weights[np.asarray(spec['lead_steps'])-1] = spec['lead_weights']
    units = np.asarray(scales['state_error_scale_si'])
    components = np.asarray(cfg['components'])
    response_units = np.asarray(cfg['response_scales_si'])
    horizons = cfg['anchors']['horizons']
    options = cfg['optimizer']
    if options['name'] != 'Adam':
        raise ValueError('expected the frozen Adam recipe')
    optimizer = Adam(model.params, learning_rate=options['learning_rate'], max_grad_norm=options['max_grad_norm'])
    rng = np.random.default_rng(plan['seed'])
    sample_hash = sha256()
    track('fit', 0)
    with (out/'training.jsonl').open('x', encoding='utf-8') as log:
        for step in range(plan['updates']):
            h = horizons[step%len(horizons)]
            x, features, y = anchors[h]
            take = rng.integers(len(x), size=cfg['anchors']['batch_size'])
            sample_hash.update(np.r_[h, take].astype('<i8').tobytes())
            anchor_loss, grads = model.loss_and_grad(x[take], features[take], y[take], state_weights=components,
                                                    state_error_scale_si=units)
            pair, pair_grads = paired_loss(
                model, initial, known, targets, masks, batch.response_mask, batch.noise_mask, batch.response_floor,
                lead_weights, components, response_units, spec['absolute_weight'], spec['response_weight'],
                sample_weights=fold.pair_weights, absolute_scale_si=units,
            )
            norm = optimizer.step({key: grads[key]+pair_grads[key] for key in grads})
            update = optimizer.iteration
            track('fit', update, record=update%30 == 0)
            if update%30 == 0:
                log.write(json.dumps({'update': update, 'anchor_loss': anchor_loss, 'pair': pair,
                                      'gradient_norm': norm}, allow_nan=False)+'\n')
                log.flush()

    draws = sample_hash.hexdigest()
    if update != plan['updates'] or draws != plan['anchor_draws_sha256']:
        raise ValueError('trained endpoint update count or anchor draws changed')
    dormant_check(model.params)
    if not all(np.isfinite(value).all() for value in model.params.values()):
        raise FloatingPointError('nonfinite endpoint parameters')
    checkpoint = out/'position_integral_900.npz'
    np.savez_compressed(checkpoint, **model.state_dict(), known_mean=fold.known_mean, known_scale=fold.known_scale,
                        known_names=np.asarray(KNOWN_NAMES), state_error_scale_si=units,
                        residual_gain_si=np.asarray(scales['residual_gain_si']), updates=np.asarray(update))
    with np.load(checkpoint, allow_pickle=False) as saved:
        restored = load_position_model(saved)
        np.testing.assert_array_equal(restored.rollout(initial, known[0]), model.rollout(initial, known[0]))
    return checkpoint, restored, draws


def score_held(model, held, fold, guard):
    capture, cfg, fields, traces, _ = held
    state = _parse_state(fields)
    nominal = np.asarray(traces['nominal_1']['actions'])
    directions, actions = action_sequences(capture, nominal)
    for label, requests in actions.items():
        np.testing.assert_array_equal(requests, traces[label]['actions'])
    rows = np.asarray(list(directions.values()))
    amplitude = capture['amplitude']
    spec = capture['model']
    fd_step = spec['fd_step']
    if not 0. < fd_step < amplitude or np.any(abs(nominal)+amplitude*abs(rows) >= 1.):
        raise ValueError('held direction is outside the frozen finite difference regime')
    if not np.isclose(model.dt_s, 1/cfg['physics']['control_hz'], rtol=0., atol=1e-14):
        raise ValueError('model and held physics time steps differ')

    start = (model, _ball_state(state), state.time_s, state.tilt_rad, state.tilt_rate_rad_s)
    tail = (cfg, fold.known_mean, fold.known_scale)
    initial, raw, features = _inputs(*start, nominal, *tail)
    prediction, slopes = _directional_rollout(model, initial, raw, features, nominal, rows, cfg, fold.known_scale)
    np.testing.assert_array_equal(prediction, predict_ball(*start, nominal, *tail))
    baselines = [traces[key]['raw'] for key in ('nominal_1', 'nominal_2')]
    gates = capture['gates']
    floors = tuple(gates[key] for key in ('position_floor_m', 'velocity_floor_m_s', 'spin_floor_rad_s'))
    thresholds = {key: gates[key] for key in ('noise_multiplier', 'strong_multiplier', 'min_cosine', 'max_relative_error')}
    metrics, forecasts = {}, {}
    for label, direction, slope in zip(directions, rows, slopes, strict=True):
        upper, lower = (predict_ball(*start, nominal+sign*fd_step*direction, *tail) for sign in (1., -1.))
        centered = (upper-lower)/(2*fd_step)
        np.testing.assert_allclose(slope, centered, rtol=spec['fd_rtol'], atol=spec['fd_atol_si'], err_msg=label)
        plus, minus = (predict_ball(*start, actions[f'{label}_{sign}'], *tail) for sign in ('plus', 'minus'))
        physics = [traces[f'{label}_{sign}']['raw'] for sign in ('plus', 'minus')]
        metrics[label] = response_metrics(baselines, *physics, prediction, plus, minus, slope, amplitude,
                                          leads=LEADS, floors=floors, **thresholds)
        forecasts[label] = {'plus': plus.tolist(), 'minus': minus.tolist(), 'directional_jvp': slope.tolist(),
                            'centered_fd': centered.tolist(), 'secant': ((plus-minus)/(2*amplitude)).tolist(),
                            'fd_max_abs_error_si_by_component': abs(centered-slope).max(axis=0).tolist()}
        guard()
    return metrics, {'initial_state_raw_si': start[1].tolist(), 'nominal': prediction.tolist(), 'pairs': forecasts,
                     'fd_step': fd_step, 'amplitude': amplitude, 'case': capture['case']}


def held_roll(score):
    row, = [row for row in score['pairs'] if row['split'] == 'validation'
            and tuple(row[key] for key in ('episode', 'tick', 'axis', 'amplitude')) == HELD_ROLL]
    return {lead: values['position'] for lead, values in row['response']['by_lead'].items()}


def compare(timed, matched_timed, score, before):
    protections = {key: regression(before['legacy_development'][key], score) for key in (CONTROL, REFERENCE)}
    if any(len(report['checks']) != 22 for report in protections.values()):
        raise ValueError('the 22 frozen protection checks were not scored')
    primary_metrics = primary(timed)
    nominal = primary_metrics['pitch_pulse']['absolute_model_nominal_error_norm']
    ratio = nominal/before['nominal_h32_position_m'][REFERENCE]
    checks = checks_for(primary_metrics, ratio, protections)
    h1 = {}
    for label, result in timed.items():
        pos = result['by_lead']['1']['position']
        matched = matched_timed[label]['by_lead']['1']['position']
        h1[label] = {'newton_half_difference_m': pos['newton_half_difference_norm'],
                     'candidate_half_difference_m': pos['model_fd_half_difference_norm'],
                     'matched_half_difference_m': matched['model_fd_half_difference_norm'],
                     'candidate_linearized_displacement_m': pos['model_jvp_linearized_displacement_norm'],
                     'matched_linearized_displacement_m': matched['model_jvp_linearized_displacement_norm']}
    return {'held_timing': timed, 'held_timing_h32_position': primary_metrics, 'matched_timing': matched_timed,
            'h1_position_response': h1, 'nominal_h32_position_m': nominal, 'nominal_h32_ratio_vs_matched': ratio,
            'legacy_protections': protections, 'legacy_validation': score, 'fixed_checks': checks,
            'held_roll': {'identity': dict(zip(('episode', 'tick', 'axis', 'amplitude'), HELD_ROLL, strict=True)),
                          'candidate': held_roll(score), **{key: held_roll(before['legacy_development'][key])
                                                         for key in (CONTROL, REFERENCE)}}}


def run(out):
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    stage, update = 'prepare', 0
    try:
        plan = json.loads(_verified_bytes(ROOT, DESIGN, DESIGN_SHA256))
        limits = plan['budget']
        gib = 1024**3
        host = MemoryStatus()
        host.length = ctypes.sizeof(host)
        get_ram = ctypes.windll.kernel32.GlobalMemoryStatusEx

        def track(name, completed=None, *, record=True):
            nonlocal stage, update
            stage = name
            if completed is not None:
                update = completed
            if not get_ram(ctypes.byref(host)):
                raise OSError('GlobalMemoryStatusEx failed')
            memory = process_memory(os.getpid())
            elapsed = time.monotonic()-started
            min_ram = limits['min_start_ram_gib'] if name == 'admission' else limits['min_running_ram_gib']
            output = directory_bytes(out)
            if record:
                with (out/'resources.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps({'stage': stage, 'update': update, 'wall_s': elapsed, 'output_bytes': output,
                                          'available_ram_bytes': host.avail_phys, **memory})+'\n')
            if (elapsed >= limits['max_wall_s'] or memory['peak_rss_bytes'] > limits['max_process_rss_gib']*gib
                    or memory['peak_private_commit_bytes'] > limits['max_process_commit_gib']*gib
                    or host.avail_phys < min_ram*gib or output > limits['max_new_output_mib']*1024**2):
                raise RuntimeError(f'position integral resource cap at {stage}, update {update}')
            return memory

        track('admission')
        if shutil.disk_usage(out).free < limits['max_new_output_mib']*1024**2:
            raise RuntimeError('insufficient disk space for the frozen output budget')
        checked = json.loads(_verified_bytes(ROOT, PREFLIGHT, PREFLIGHT_SHA256))
        if (checked['status'] != 'passed' or checked['design_sha256'] != DESIGN_SHA256
                or checked['pair_count'] != 46 or checked['training_updates'] != 0
                or checked['simulation_steps'] != 0 or checked['test_episode_files_read'] is not False
                or checked['anchor_draws_sha256'] != plan['anchor_draws_sha256']):
            raise ValueError('position integral preflight disagrees with the frozen design')
        inputs, parent, cfg, scales, fold, legacy, anchors, _ = prepare_recipe(plan, lambda: track('prepare'))
        accepted = json.loads(inputs[plan['parent_run']])
        parent_dir = Path(plan['parent_run']).parent
        upstream = {(parent_dir/name).as_posix(): accepted['artifacts_sha256'][name]
                    for name in ('comparison.json', 'execution_plan.json')}
        before, parent_run = (json.loads(_verified_bytes(ROOT, path, expected)) for path, expected in upstream.items())
        if parent_run['design'] != parent:
            raise ValueError('parent execution differs from the frozen training design')
        own = Path(__file__).resolve().relative_to(ROOT).as_posix()
        sources = {**parent_run['source_sha256'], **checked['source_sha256'], own: digest(ROOT/own)}
        history = cfg['history_plan']
        bound_inputs = {**plan['inputs_sha256'], **upstream, history: cfg['inputs_sha256'][history],
                        DESIGN: DESIGN_SHA256, PREFLIGHT: PREFLIGHT_SHA256}

        def verify():
            for name, expected in {**sources, **bound_inputs}.items():
                _verified_bytes(ROOT, name, expected)

        verify()
        histories = {row['episode_id']: row['history_id']
                     for row in json.loads(_verified_bytes(ROOT, history, bound_inputs[history]))['episodes']}
        held = read_held(parent)
        model = TrapezoidPositionResidualMLP(len(KNOWN_NAMES), fold.state_mean, fold.state_scale, anchors.dt_s,
                    plan['seed'], hidden=cfg['network'][1], residual_gain=np.asarray(scales['residual_gain_si'])/fold.state_scale)
        # Check the reused metrics and legacy identity before the sole fitting run.
        reference = plan['matched_reference']
        with np.load(BytesIO(inputs[reference]), allow_pickle=False) as saved:
            matched = load_position_model(saved)
        matched_timed, _ = score_held(matched, held, fold, lambda: track('matched_precheck'))
        if primary(matched_timed) != before['held_timing_h32_position'][REFERENCE]:
            raise ValueError('matched held timing scores did not exactly reproduce')
        compare(matched_timed, matched_timed, before['legacy_development'][REFERENCE], before)
        write_json(out/'execution_plan.json', {'design': plan, 'source_sha256': sources, 'inputs_sha256': bound_inputs,
                   'training_pairs': len(fold.pairs.initial), 'paired_loss': cfg['pairs'], 'optimizer': cfg['optimizer'],
                   'anchor_sampling': cfg['anchors'], 'matched_held_primary_reproduced': True})
        print('Fitting 46 frozen pairs with tool-frame trapezoid integration. 900 updates.', flush=True)
        checkpoint, model, draws = fit(model, fold, cfg, scales, plan, out, track)
        track('held_timing')
        timed, forecasts = score_held(model, held, fold, lambda: track('held_timing'))
        track('legacy_validation')
        score = report(model, fold.known_mean, fold.known_scale, legacy, anchors, histories)
        score['pairs'] = [row for row in score['pairs'] if row['split'] == 'validation']
        result = compare(timed, matched_timed, score, before)
        write_json(out/'comparison.json', result)
        write_json(out/'timed_rollouts.json', {**forecasts, 'checkpoint_sha256': digest(checkpoint)})
        verify()
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline fit imported a physics module')
        memory = track('complete')
        gates = result['fixed_checks']
        write_json(out/'validation.json', {'status': 'completed', 'updates': update, 'worker_starts': 1,
                   'anchor_draws_sha256': draws, 'checkpoint_sha256': digest(checkpoint), 'fixed_checks': gates,
                   'replacement_gate_passed': all(gates.values()), 'checkpoint_selection': False,
                   'wall_s': time.monotonic()-started, **SCOPE, **memory,
                   'artifacts_sha256': {p.name: digest(p) for p in sorted(out.iterdir())}})
        track('complete', record=False)
    except BaseException as error:
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error), 'stage': stage, 'updates': update,
                   'wall_s': time.monotonic()-started, **SCOPE})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    target = parser.parse_args().output.resolve()
    if len(target.relative_to(ROOT/'runs').parts) != 1:
        raise ValueError('choose a new run directory immediately under runs')
    run(target)
