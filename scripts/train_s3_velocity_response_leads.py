"""Fit one fixed trapezoid model with denser velocity half-difference supervision."""
import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import time

from preflight_s3_velocity_response_leads import (
    DESIGN, DESIGN_SHA256, ROOT, digest, dormant_check, lead_vectors, prepare_comparison, weight_audit, write_json,
)

import numpy as np

from train_s3_position_integral import SCOPE, compare, held_roll, read_held, regression, report, score_held
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.resources import MemoryStatus, directory_bytes
from ball_retention.velocity_response_loss import paired_velocity_response_loss
from ball_retention.world_model import Adam


PREFLIGHT = 'runs/s3_velocity_response_leads_preflight_20261115_verified/preflight.json'
PREFLIGHT_SHA256 = 'cddb25c64d51b8a2110a3ade2036c6622c118ca2abf64c799391e242e3f75bbf'
OUT = ROOT/'runs/s3_velocity_response_leads_fit_20261115'
MATCHED = 'position_integral'


def fit(model, fold, cfg, scales, plan, leads, out, track):
    def standardize(array):
        return (array-fold.known_mean)/fold.known_scale

    anchors = {h: (model.normalize(w.initial), standardize(w.known), model.normalize(w.target))
               for h, w in fold.anchors.items()}
    batch = fold.pairs
    initial = model.normalize(batch.initial)
    known = standardize(np.stack((batch.plus_known, batch.minus_known)))
    targets = np.stack((batch.plus_target, batch.minus_target))
    masks = np.stack((batch.plus_mask, batch.minus_mask))
    base, velocity = leads
    units = np.asarray(scales['state_error_scale_si'])
    components = np.asarray(cfg['components'])
    response_units = np.asarray(cfg['response_scales_si'])
    spec = cfg['pairs']
    adam = cfg['optimizer']
    if adam['name'] != 'Adam':
        raise ValueError('expected the frozen Adam recipe')
    optimizer = Adam(model.params, learning_rate=adam['learning_rate'], max_grad_norm=adam['max_grad_norm'])
    rng = np.random.default_rng(plan['seed'])
    sample_hash = sha256()
    horizons = cfg['anchors']['horizons']
    track('fit', 0)
    with (out/'training.jsonl').open('x', encoding='utf-8') as log:
        for step in range(plan['updates']):
            horizon = horizons[step%len(horizons)]
            x, features, target = anchors[horizon]
            take = rng.integers(len(x), size=cfg['anchors']['batch_size'])
            sample_hash.update(np.r_[horizon, take].astype('<i8').tobytes())
            anchor_loss, grads = model.loss_and_grad(x[take], features[take], target[take], state_weights=components,
                                                    state_error_scale_si=units)
            pair, pair_grads = paired_velocity_response_loss(
                model, initial, known, targets, masks, batch.response_mask, batch.noise_mask, batch.response_floor,
                base, components, response_units, spec['absolute_weight'], spec['response_weight'],
                velocity_lead_weights=velocity, sample_weights=fold.pair_weights, absolute_scale_si=units,
            )
            norm = optimizer.step({key: grads[key]+pair_grads[key] for key in grads})
            periodic = optimizer.iteration%30 == 0
            track('fit', optimizer.iteration, record=periodic)
            if periodic:
                log.write(json.dumps({'update': optimizer.iteration, 'anchor_loss': anchor_loss, 'pair': pair,
                                      'gradient_norm': norm}, allow_nan=False)+'\n')
                log.flush()

    draws = sample_hash.hexdigest()
    if optimizer.iteration != plan['updates'] or draws != plan['anchor_draws_sha256']:
        raise ValueError('fixed endpoint or S2 anchor stream differs from the plan')
    dormant_check(model.params)
    if not all(np.isfinite(value).all() for value in model.params.values()):
        raise FloatingPointError('nonfinite endpoint parameters')
    path = out/'velocity_response_leads_900.npz'
    np.savez_compressed(path, **model.state_dict(), known_mean=fold.known_mean, known_scale=fold.known_scale,
                        known_names=np.asarray(KNOWN_NAMES), state_error_scale_si=units,
                        residual_gain_si=np.asarray(scales['residual_gain_si']), updates=np.asarray(optimizer.iteration))
    with np.load(path, allow_pickle=False) as saved:
        restored = load_position_model(saved)
        np.testing.assert_array_equal(restored.rollout(initial, known[0]), model.rollout(initial, known[0]))
    return path, restored, draws


def velocity_errors(candidate, matched):
    """Report finite signed velocity half-difference errors in SI at all scored leads."""
    by_direction = {}
    for direction, report in candidate.items():
        by_lead = {}
        for lead, row in report['by_lead'].items():
            cand = row['velocity']
            ref = matched[direction]['by_lead'][lead]['velocity']
            truth = np.asarray(cand['newton_half_difference'])
            np.testing.assert_array_equal(truth, ref['newton_half_difference'])
            errors = {name: float(np.linalg.norm(np.asarray(values['model_fd_half_difference'])-truth))
                      for name, values in (('matched', ref), ('candidate', cand))}
            by_lead[lead] = {'newton_norm_m_s': float(np.linalg.norm(truth)),
                             'matched_error_m_s': errors['matched'], 'candidate_error_m_s': errors['candidate'],
                             'candidate_minus_matched_error_m_s': errors['candidate']-errors['matched']}
        by_direction[direction] = by_lead
    return by_direction


def compare_endpoints(timed, matched_timed, score, original, matched):
    result = compare(timed, matched_timed, score, original)
    matching = regression(matched['legacy_validation'], score)
    if len(matching['checks']) != 22:
        raise ValueError('the matched trapezoid did not score all 22 protections')
    result['legacy_protections'][MATCHED] = matching
    result['held_roll'][MATCHED] = held_roll(matched['legacy_validation'])
    result['nominal_h32_ratio_vs_original_46'] = result.pop('nominal_h32_ratio_vs_matched')
    result['nominal_h32_ratio_vs_matched_trapezoid'] = result['nominal_h32_position_m']/matched['nominal_h32_position_m']
    result['velocity_half_difference_errors'] = velocity_errors(timed, matched_timed)
    return result


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
            elapsed = time.monotonic()-started
            memory = process_memory(os.getpid())
            output = directory_bytes(out)
            floor = limits['min_start_ram_gib'] if stage == 'admission' else limits['min_running_ram_gib']
            if record:
                with (out/'resources.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps({'stage': stage, 'update': update, 'wall_s': elapsed, 'output_bytes': output,
                                          'available_ram_bytes': host.avail_phys, **memory})+'\n')
            if (elapsed >= limits['max_wall_s'] or memory['peak_rss_bytes'] > limits['max_process_rss_gib']*gib
                    or memory['peak_private_commit_bytes'] > limits['max_process_commit_gib']*gib
                    or host.avail_phys < floor*gib or output > limits['max_new_output_mib']*1024**2):
                raise RuntimeError(f'velocity response resource cap at {stage}, update {update}')
            return memory

        track('admission')
        if shutil.disk_usage(out).free < limits['max_new_output_mib']*1024**2:
            raise RuntimeError('insufficient disk space for the frozen output budget')
        preflight = json.loads(_verified_bytes(ROOT, PREFLIGHT, PREFLIGHT_SHA256))
        if (preflight['status'] != 'passed' or preflight['design_sha256'] != DESIGN_SHA256
                or preflight['pair_count'] != plan['pair_count'] or preflight['training_updates'] != 0
                or preflight['simulation_steps'] != 0 or preflight['test_episode_files_read'] is not False
                or preflight['anchor_draws_sha256'] != plan['anchor_draws_sha256']
                or preflight['inputs_sha256'] != plan['inputs_sha256']):
            raise ValueError('velocity response preflight disagrees with the frozen fit plan')
        bound = {**plan['inputs_sha256'], DESIGN: DESIGN_SHA256, PREFLIGHT: PREFLIGHT_SHA256}
        own = Path(__file__).resolve().relative_to(ROOT).as_posix()
        sources = {**preflight['source_sha256'], own: digest(ROOT/own)}
        for name, expected in {**sources, **bound}.items():
            _verified_bytes(ROOT, name, expected)
        parent, cfg, scales, fold, legacy, anchors, matched_model, _, _ = prepare_comparison(plan, lambda: track('prepare'))

        def read(path, sha):
            bound[Path(path).as_posix()] = sha
            return json.loads(_verified_bytes(ROOT, path, sha))

        def read_comparison(validation, sha):
            evidence = read(validation, sha)
            return read(Path(validation).parent/'comparison.json', evidence['artifacts_sha256']['comparison.json'])

        parent_design = read(plan['parent_design'], bound[plan['parent_design']])
        original_run = parent_design['parent_run']
        original = read_comparison(original_run, parent_design['inputs_sha256'][original_run])
        matched = read_comparison(plan['parent_run'], bound[plan['parent_run']])
        history = cfg['history_plan']
        histories = {item['episode_id']: item['history_id'] for item in read(history, cfg['inputs_sha256'][history])['episodes']}
        held = read_held(parent)
        matched_timed, _ = score_held(matched_model, held, fold, lambda: track('matched_precheck'))
        if matched_timed != matched['held_timing']:
            raise ValueError('matched trapezoid scores did not exactly reproduce')
        compare_endpoints(matched_timed, matched_timed, matched['legacy_validation'], original, matched)

        leads = lead_vectors(plan, fold.pairs.plus_target.shape[1])
        _, audit = weight_audit(*leads, np.asarray(cfg['components']), fold.pairs.noise_mask,
                                fold.pair_weights, plan['static_checks']['weight_atol'])
        if audit != preflight['checks']['h32_weights']:
            raise ValueError('effective velocity weights differ from the accepted preflight')
        write_json(out/'execution_plan.json', {'design': plan, 'source_sha256': sources, 'inputs_sha256': bound,
                   'training_pairs': len(fold.pairs.initial), 'paired_loss': cfg['pairs'], 'optimizer': cfg['optimizer'],
                   'anchor_sampling': cfg['anchors'], 'effective_velocity_weights': audit,
                   'matched_trapezoid_held_scores_reproduced': True})
        fresh = TrapezoidPositionResidualMLP(len(KNOWN_NAMES), fold.state_mean, fold.state_scale, anchors.dt_s,
                    plan['seed'], hidden=cfg['network'][1], residual_gain=np.asarray(scales['residual_gain_si'])/fold.state_scale)
        print('Fitting 46 frozen pairs with dense velocity response supervision. 900 updates.', flush=True)
        checkpoint, model, draws = fit(fresh, fold, cfg, scales, plan, leads, out, track)
        track('held_timing')
        timed, forecasts = score_held(model, held, fold, lambda: track('held_timing'))
        track('legacy_validation')
        score = report(model, fold.known_mean, fold.known_scale, legacy, anchors, histories)
        score['pairs'] = [row for row in score['pairs'] if row['split'] == 'validation']
        comparison = compare_endpoints(timed, matched_timed, score, original, matched)
        write_json(out/'comparison.json', comparison)
        sha = digest(checkpoint)
        write_json(out/'timed_rollouts.json', {**forecasts, 'checkpoint_sha256': sha})
        for name, expected in {**bound, **sources}.items():
            _verified_bytes(ROOT, name, expected)
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline fitting imported physics')
        memory = track('complete')
        checks = comparison['fixed_checks']
        write_json(out/'validation.json', {'status': 'completed', 'updates': update, 'worker_starts': 1,
                   'anchor_draws_sha256': draws, 'checkpoint_sha256': sha, 'fixed_checks': checks,
                   'replacement_gate_passed': all(checks.values()), 'checkpoint_selection': False, **SCOPE, **memory,
                   'wall_s': time.monotonic()-started, 'artifacts_sha256': {p.name: digest(p) for p in sorted(out.iterdir())}})
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
