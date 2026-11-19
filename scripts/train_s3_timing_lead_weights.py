"""Run the fixed 46-pair fit with only the paired lead weights changed."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

from diagnose_s3_timing_coverage import diagnose, legacy_case, read_changed_pair
from train_s3_timing_coverage import ROOT, digest, fit, primary, read_held, regression, report, timed_score, write_json
from ball_retention.action_grad import action_sensitivity, predict_ball
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.guard import process_memory
from ball_retention.model_data import _ball_state
from ball_retention.resources import directory_bytes, snapshot
from ball_retention.timing_coverage_fit import DESIGN as PARENT_DESIGN, prepare
from ball_retention.world_model import ResidualMLP


DESIGN = 'configs/s3_timing_lead_weight_fit_plan.json'
DESIGN_SHA256 = 'fef5d79c672e370f5fabef8827876ffe2490e7924d385337afbfa395c0e35dda'
PREFLIGHT = 'runs/s3_timing_coverage_diagnosis_20261115_verified/lead_weight_preflight.json'
PREFLIGHT_SHA256 = '39b25efdc3c569401ee3b26d571f61d62bef2064577ef84c94031c3c0696ae5f'
OUT = ROOT/'runs/s3_timing_lead_weight_fit_20261115'
REFERENCE = 'held_offsets_plus_timing'
CONTROL = 'held_offsets'
CANDIDATE = 'lead_weight_intervention'


def anchor_draws(cfg, fold, seed, updates):
    rng = np.random.default_rng(seed)
    hashes = sha256()
    horizons = cfg['anchors']['horizons']
    for update in range(updates):
        h = horizons[update%len(horizons)]
        take = rng.integers(len(fold.anchors[h].initial), size=cfg['anchors']['batch_size'])
        hashes.update(np.r_[h, take].astype('<i8').tobytes())
    return hashes.hexdigest()


def prepare_run(plan, guard):
    inputs = dict(plan['inputs_sha256'])

    def load(name, expected, *, lines=False):
        name = Path(name).as_posix()
        raw = _verified_bytes(ROOT, name, expected)
        inputs[name] = expected
        return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)

    for name, expected in inputs.items():
        _verified_bytes(ROOT, name, expected)
    checked = load(PREFLIGHT, PREFLIGHT_SHA256)
    if (checked['passed'] is not True or checked['design_sha256'] != DESIGN_SHA256
            or checked['pair_count'] != 46 or checked['training_updates'] != 0):
        raise ValueError('frozen lead-weight preflight disagrees with the execution plan')
    parent_path = Path(plan['parent_run'])
    accepted = load(parent_path, inputs[parent_path.as_posix()])
    parent_exec = load(parent_path.parent/'execution_plan.json', accepted['artifacts_sha256']['execution_plan.json'])
    comparison = load(parent_path.parent/'comparison.json', accepted['artifacts_sha256']['comparison.json'])
    parent, cfg, scales, folds, _, legacy, anchors, _ = prepare(ROOT, guard)
    if (plan['parent_design'] != PARENT_DESIGN or parent_exec['design'] != parent or accepted['status'] != 'completed'
            or not accepted['control_reproduced_exactly'] or accepted['updates'] != {CONTROL: 900, REFERENCE: 900}
            or accepted['identical_anchor_draws_sha256'] != plan['anchor_draws_sha256']
            or accepted['checkpoint_sha256'][REFERENCE] != inputs[plan['matched_reference']]
            or accepted['checkpoint_sha256'][CONTROL] != inputs[plan['legacy_protection_reference']]
            or accepted['simulation_steps'] != 0 or accepted['test_episode_files_read'] is not False):
        raise ValueError('parent endpoints do not match the frozen reference')
    fold = folds[REFERENCE]
    leads = plan['paired_leads']
    reference = dict(zip(cfg['pairs']['lead_steps'], cfg['pairs']['lead_weights'], strict=True))
    weights = plan['intervention_lead_weights']
    if (len(fold.pairs.initial) != 46 or parent['seed'] != plan['seed'] or parent['updates_per_arm'] != plan['updates']
            or not set(reference) <= set(leads) or [reference.get(h, 0.) for h in leads] != plan['reference_lead_weights']
            or len(leads) != len(weights) or not np.isclose(sum(weights), 1.) or weights[-1] != reference[32]
            or anchor_draws(cfg, fold, plan['seed'], plan['updates']) != plan['anchor_draws_sha256']):
        raise ValueError('same-data recipe or deterministic anchor draws changed')
    intervention = {**cfg, 'pairs': {**cfg['pairs'], 'lead_steps': leads, 'lead_weights': weights}}
    held = read_held(parent)
    pair, _ = read_changed_pair(comparison)
    legacy_snapshot = legacy_case(parent, held[0], pair, load)
    histories = {e['episode_id']: e['history_id'] for e in load(cfg['history_plan'],
                    cfg['inputs_sha256'][cfg['history_plan']])['episodes']}
    return parent, intervention, scales, fold, legacy, anchors, held, pair, legacy_snapshot, comparison, histories, inputs, parent_exec


def selected_diagnostics(result, leads):
    return {'directions': {name: {'position_onsets': row['position_onsets'],
                                  'position': {str(h): row['by_lead'][str(h)]['position'] for h in leads}}
                           for name, row in result['directions'].items()},
            'branch_errors': {name: {str(h): {family: {key: values[key] for key in
                    ('free_error_norm', 'local_error_norm', 'propagated_error_norm')}
                    for family, values in branch['by_lead'][str(h)].items()} for h in leads}
                    for name, branch in result['branches'].items()}}


def diagnose_legacy(legacy_snapshot, pair, model, saved, guard):
    analysis, cfg, fields, traces, directions = legacy_snapshot
    name, = directions
    state = _parse_state(fields)
    known = (cfg, saved['known_mean'], saved['known_scale'])
    args = (model, _ball_state(state), state.time_s, state.tilt_rad, state.tilt_rate_rad_s)
    sensitivity = action_sensitivity(*args, traces['nominal_1']['actions'], *known)
    forecasts = {'initial_state_raw_si': args[1], 'nominal': sensitivity['prediction'],
                 'pairs': {name: {**{s: predict_ball(*args, traces[f'{name}_{s}']['actions'], *known)
                                     for s in ('plus', 'minus')}, 'directional_jvp': sensitivity['jacobian'][..., pair[2]]}}}
    return diagnose(analysis, cfg, fields, traces, forecasts, saved, guard)


def checks_for(timed, nominal_ratio, protections):
    both = {name: row['model_jvp_vs_newton']['agrees'] is True and row['model_fd_vs_newton']['agrees'] is True
            for name, row in timed.items()}
    return {'short_pulse_retained': both['pitch_pulse'],
            'reversal_or_dual_new_pass': both['pitch_reversal'] or both['dual_axis_pulse'],
            'nominal_h32_within_1_10': nominal_ratio <= 1.10,
            'all_legacy_protections_vs_40_pairs': protections[CONTROL]['passed']}


def run(out):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    update, stage = 0, 'prepare'
    try:
        plan = json.loads(_verified_bytes(ROOT, DESIGN, DESIGN_SHA256))
        limits = plan['budget']

        def track(name='evaluation', completed=None, *, check=True):
            nonlocal update, stage
            stage = name
            if completed is not None:
                update = completed
            if not check:
                return
            memory, machine = process_memory(os.getpid()), snapshot(ROOT)
            elapsed, output = time.monotonic()-began, directory_bytes(out)
            with (out/'resources.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'stage': stage, 'update': update, 'wall_s': elapsed, 'output_bytes': output,
                                         **memory, **machine})+'\n')
            ram = limits['min_start_ram_gib'] if stage == 'admission' else limits['min_running_ram_gib']
            if (elapsed >= limits['max_wall_s'] or memory['peak_rss_bytes'] > limits['max_process_rss_gib']*1024**3
                    or memory['peak_private_commit_bytes'] > limits['max_process_commit_gib']*1024**3
                    or machine['available_ram_bytes'] < ram*1024**3 or output > limits['max_new_output_mib']*1024**2):
                raise RuntimeError(f'lead-weight resource cap at {stage}, update {update}')
            return memory

        track('admission')
        parent, cfg, scales, fold, legacy, anchors, held, pair, legacy_snapshot, before, histories, inputs, parent_exec = prepare_run(
            plan, lambda: track('prepare'))
        sources = {**parent_exec['source_sha256'], Path(__file__).resolve().relative_to(ROOT).as_posix(): digest(Path(__file__)),
                   'scripts/diagnose_s3_timing_coverage.py': digest(ROOT/'scripts/diagnose_s3_timing_coverage.py')}

        def verify_hashes():
            for name, expected in {**sources, **inputs, DESIGN: DESIGN_SHA256}.items():
                _verified_bytes(ROOT, name, expected)

        verify_hashes()
        write_json(out/'execution_plan.json', {'design': plan, 'design_sha256': DESIGN_SHA256,
                   'source_sha256': sources, 'inputs_sha256': inputs, 'training_pairs': len(fold.pairs.initial), 'recipe': cfg['pairs']})
        track(CANDIDATE, 0)
        print('Fitting 46 pairs with frozen lead redistribution, 900 updates.', flush=True)
        checkpoint, draws = fit(out, CANDIDATE, parent, cfg, scales, fold, anchors.dt_s, track, began+limits['max_wall_s'])
        if update != plan['updates'] or draws != plan['anchor_draws_sha256']:
            raise ValueError('trained endpoint update count or anchor draws changed')

        track('held_evaluation')
        timed, forecast = timed_score(checkpoint, held, lambda: track('held_evaluation'))
        with np.load(checkpoint, allow_pickle=False) as saved:
            model = ResidualMLP.from_state_dict(saved)
            score = report(model, saved['known_mean'], saved['known_scale'], legacy, anchors, histories)
            score['pairs'] = [row for row in score['pairs'] if row['split'] == 'validation']
            capture, physics, state, traces, _ = held
            timed_diagnostic = diagnose(capture, physics, state, traces, forecast, saved, lambda: track('diagnosis'))
            legacy_diagnostic = diagnose_legacy(legacy_snapshot, pair, model, saved, lambda: track('diagnosis'))

        protection = {name: regression(before['legacy_development'][name], score) for name in (CONTROL, REFERENCE)}
        if len(protection[CONTROL]['checks']) != 22 or len(protection[REFERENCE]['checks']) != 22:
            raise ValueError('the 22 frozen protection checks were not scored')
        episode, tick, axis, amplitude = pair
        row, = [r for r in score['pairs'] if (r['episode'], r['tick'], r['axis'], r['amplitude']) == pair]
        position = row['response']['by_lead']['32']['position']
        hit = legacy_diagnostic['directions']['held_roll']['by_lead']['32']['position']['comparisons']['free']['cosine']
        np.testing.assert_allclose(hit, position['cosine'], atol=1e-10, rtol=0.)
        held_primary = primary(timed)
        nominal = held_primary['pitch_pulse']['absolute_model_nominal_error_norm']
        ratio = nominal/before['nominal_h32_position_m'][REFERENCE]
        checks = checks_for(held_primary, ratio, protection)
        write_json(out/'comparison.json', {'held_timing': timed, 'held_timing_h32_position': held_primary,
                   'nominal_h32_position_m': nominal, 'nominal_h32_ratio_vs_matched': ratio,
                   'legacy_protections': protection, 'legacy_validation': score, 'fixed_checks': checks,
                   'held_roll': {'episode': episode, 'tick': tick, 'axis': axis, 'amplitude': amplitude,
                                 'candidate_h32': position,
                                 'reference_h32': {arm: next(r['response']['by_lead']['32']['position']
                                     for r in before['legacy_development'][arm]['pairs']
                                     if (r['episode'], r['tick'], r['axis'], r['amplitude']) == pair)
                                     for arm in (CONTROL, REFERENCE)}}})
        write_json(out/'per_lead_diagnosis.json', {'timed': selected_diagnostics(timed_diagnostic, plan['paired_leads']),
                   'held_roll': selected_diagnostics(legacy_diagnostic, plan['paired_leads'])})
        write_json(out/'timed_rollouts.json', forecast)
        verify_hashes()
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline lead-weight fit unexpectedly loaded physics')
        memory = track('complete')
        write_json(out/'validation.json', {'status': 'completed', 'updates': update, 'anchor_draws_sha256': draws,
                   'checkpoint_sha256': digest(checkpoint), 'fixed_checks': checks, 'replacement_gate_passed': all(checks.values()),
                   'checkpoint_selection': False, 'control_accepted': False, 'simulation_steps': 0, 'test_episode_files_read': False,
                   'wall_s': time.monotonic()-began, **memory,
                   'artifacts_sha256': {p.name: digest(p) for p in sorted(out.iterdir())}})
    except BaseException as error:
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error), 'stage': stage, 'updates': update,
                   'wall_s': time.monotonic()-began, 'simulation_steps': 0, 'test_episode_files_read': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    path = parser.parse_args().output.resolve()
    if not path.relative_to(ROOT/'runs').parts:
        raise ValueError('choose a fresh run subdirectory')
    run(path)
