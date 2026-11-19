"""Fit and score the two fixed endpoints of the timed-pair coverage ablation."""
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

from check_s3_actions import ROOT, digest, write_json
from preflight_s3_deterministic_action_response import action_sequences
from train_s3_history import regression, report
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.deterministic_response_model import evaluate_model
from ball_retention.deterministic_response_plan import load as load_response
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES
from ball_retention.pair_loss import paired_loss
from ball_retention.resources import directory_bytes, snapshot
from ball_retention.timing_coverage_fit import DESIGN, DESIGN_SHA256, prepare
from ball_retention.world_model import Adam, ResidualMLP
from deterministic_archive_metrics import _full_branch
from deterministic_response_metrics import response_metrics


PREFLIGHT = 'runs/s3_timing_coverage_fit_preflight_20261115/preflight.json'
PREFLIGHT_SHA256 = '09f868c67643c6f34619657dc6bf58ecd5615beaaee589c9ed1b890494cab21f'
OUT = ROOT/'runs/s3_timing_coverage_fit_20261115'
LEADS = (8, 16, 24, 32)


def fit(out, name, plan, cfg, scales, fold, dt, track, deadline):
    units = np.array(scales['state_error_scale_si'])
    gain = np.array(scales['residual_gain_si'])
    model = ResidualMLP(len(KNOWN_NAMES), fold.state_mean, fold.state_scale, dt, plan['seed'],
                        hidden=cfg['network'][1], residual_gain=gain/fold.state_scale)

    def known(values):
        return (values-fold.known_mean)/fold.known_scale

    anchors = {h: (model.normalize(w.initial), known(w.known), model.normalize(w.target)) for h, w in fold.anchors.items()}
    b = fold.pairs
    initial = model.normalize(b.initial)
    pair_known = known(np.stack((b.plus_known, b.minus_known)))
    target = np.stack((b.plus_target, b.minus_target))
    mask = np.stack((b.plus_mask, b.minus_mask))
    leads = np.zeros(b.plus_target.shape[1])
    pairs = cfg['pairs']
    leads[np.array(pairs['lead_steps'])-1] = pairs['lead_weights']
    components = np.array(cfg['components'])
    response_units = np.array(cfg['response_scales_si'])
    rng = np.random.default_rng(plan['seed'])
    optimizer_cfg = cfg['optimizer']
    if optimizer_cfg['name'] != 'Adam':
        raise ValueError('timing coverage optimizer must match the frozen Adam recipe')
    optimizer = Adam(model.params, learning_rate=optimizer_cfg['learning_rate'],
                     max_grad_norm=optimizer_cfg['max_grad_norm'])
    sample_hash = sha256()
    horizons = cfg['anchors']['horizons']
    with (out/f'{name}_training.jsonl').open('x', encoding='utf-8') as log:
        for update in range(1, plan['updates_per_arm']+1):
            if time.monotonic() >= deadline:
                raise RuntimeError('timing coverage fit hit the wall-clock cap')
            h = horizons[(update-1)%len(horizons)]
            x, k, y = anchors[h]
            take = rng.integers(len(x), size=cfg['anchors']['batch_size'])
            sample_hash.update(np.r_[h, take].astype('<i8').tobytes())
            anchor_loss, grads = model.loss_and_grad(x[take], k[take], y[take], state_weights=components,
                                                    state_error_scale_si=units)
            pair_loss, pair_grads = paired_loss(
                model, initial, pair_known, target, mask, b.response_mask, b.noise_mask, b.response_floor, leads,
                components, response_units, pairs['absolute_weight'], pairs['response_weight'],
                sample_weights=fold.pair_weights, absolute_scale_si=units,
            )
            norm = optimizer.step({key: grad+pair_grads[key] for key, grad in grads.items()})
            track(name, update, check=update%30 == 0)
            if update%30 == 0:
                log.write(json.dumps({'update': update, 'anchor_loss': anchor_loss, 'pair': pair_loss,
                                      'gradient_norm': norm}, allow_nan=False)+'\n')
                log.flush()

    arrays = {**model.state_dict(), 'known_mean': fold.known_mean, 'known_scale': fold.known_scale,
              'known_names': np.array(KNOWN_NAMES), 'state_error_scale_si': units, 'residual_gain_si': gain,
              'updates': np.asarray(optimizer.iteration)}
    if name == 'held_offsets':
        reference = plan['control_checkpoint']
        with np.load(BytesIO(_verified_bytes(ROOT, reference, plan['inputs_sha256'][reference])), allow_pickle=False) as saved:
            if set(saved.files) != arrays.keys():
                raise ValueError('matched control checkpoint keys differ from the pinned endpoint')
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, saved[key], err_msg=f'control endpoint {key}')
        write_json(out/'control_reproduction.json', {'array_equal': True, 'arrays': sorted(arrays),
                   'reference_sha256': plan['inputs_sha256'][reference], 'update': optimizer.iteration})
        print('Control endpoint reproduced exactly at update 900.', flush=True)
    checkpoint = out/f'{name}_900.npz'
    np.savez_compressed(checkpoint, **arrays)
    with np.load(checkpoint, allow_pickle=False) as loaded:
        restored = ResidualMLP.from_state_dict(loaded)
        np.testing.assert_array_equal(model.rollout(initial, pair_known[0]), restored.rollout(initial, pair_known[0]))
    track(name, optimizer.iteration)
    return checkpoint, sample_hash.hexdigest()


def read_held(plan):
    capture, cfg, evidence = load_response(ROOT)
    path = Path(plan['held_timing_run'])
    held = json.loads(_verified_bytes(ROOT, path, plan['inputs_sha256'][path.as_posix()]))
    if held['case'] != capture['case'] or held['branches'] != capture['branch_order']:
        raise ValueError('held timing evidence disagrees with the capture design')
    traces = {}
    for label in held['branches']:
        name = f'worker/{label}.json'
        trace = json.loads(_verified_bytes(ROOT, path.parent/name, held['artifacts_sha256'][name]))
        expected = evidence['branch_commands'][label]
        if trace['commands'] != expected['executed'] or trace['actions'] != expected['requested']:
            raise ValueError(f'held timing commands differ from frozen inputs: {label}')
        traces[label] = _full_branch(trace, held['case']['tick'])
    return capture, cfg, evidence['snapshot']['state'], traces, held


def timed_score(checkpoint, held, guard):
    capture, cfg, state, traces, _ = held
    nominal = traces['nominal_1']['actions']
    directions, actions = action_sequences(capture, nominal)
    for name, requested in actions.items():
        np.testing.assert_array_equal(requested, traces[name]['actions'])
    spec = {**capture['model'], 'checkpoint': checkpoint.relative_to(ROOT).as_posix(),
            'checkpoint_sha256': digest(checkpoint), 'amplitude': capture['amplitude']}
    model = evaluate_model(ROOT, spec, cfg, state, nominal, directions, guard)
    baselines = [traces[key]['raw'] for key in ('nominal_1', 'nominal_2')]
    results = {}
    for name, pred in model['pairs'].items():
        physical = [traces[f'{name}_{sign}']['raw'] for sign in ('plus', 'minus')]
        gates = capture['gates']
        results[name] = response_metrics(
            baselines, *physical, model['nominal'], pred['plus'], pred['minus'], pred['directional_jvp'],
            capture['amplitude'], leads=LEADS,
            floors=tuple(gates[key] for key in ('position_floor_m', 'velocity_floor_m_s', 'spin_floor_rad_s')),
            **{key: gates[key] for key in ('noise_multiplier', 'strong_multiplier', 'min_cosine', 'max_relative_error')},
        )
    return results, model


def primary(timed):
    result = {}
    for name, pair in timed.items():
        position = pair['by_lead']['32']['position']
        result[name] = {key: position[key] for key in ('newton_strong', 'absolute_model_nominal_error_norm',
                        'absolute_model_positive_error_norm', 'absolute_model_negative_error_norm',
                        'model_fd_vs_newton', 'model_jvp_vs_newton')}
    return result


def run(out):
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    updates, last = {}, {}
    try:
        plan = json.loads(_verified_bytes(ROOT, DESIGN, DESIGN_SHA256))
        limits = plan['budget']
        preflight = json.loads(_verified_bytes(ROOT, PREFLIGHT, PREFLIGHT_SHA256))
        if (preflight['status'] != 'passed' or preflight['design_sha256'] != DESIGN_SHA256
                or preflight['arm_pair_counts'] != plan['arms'] or preflight['training_updates'] != 0):
            raise ValueError('timing coverage preflight disagrees with the frozen design')
        sources = {**preflight['source_sha256'], **{path.relative_to(ROOT).as_posix(): digest(path) for path in {
            Path(__file__).resolve(), ROOT/'scripts/train_s3_history.py', ROOT/'scripts/evaluate_s3_history.py',
            ROOT/'scripts/evaluate_s3_coverage.py', ROOT/'scripts/preflight_s3_deterministic_action_response.py',
            ROOT/'diagnostics/resource_probe/deterministic_archive_metrics.py',
            ROOT/'diagnostics/resource_probe/archive_restore_metrics.py',
            ROOT/'diagnostics/resource_probe/deterministic_response_metrics.py',
        }}}

        def verify():
            for name, expected in {**plan['inputs_sha256'], **sources}.items():
                _verified_bytes(ROOT, name, expected)

        def track(stage='evaluation', update=None, *, check=True):
            last.update(stage=stage, update=update)
            if update is not None:
                updates[stage] = update
            if not check:
                return
            elapsed = time.monotonic()-started
            mem, machine = process_memory(os.getpid()), snapshot(ROOT)
            output = directory_bytes(out)
            with (out/'resources.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({**last, 'wall_s': elapsed, **mem, **machine, 'output_bytes': output})+'\n')
            min_ram = limits['min_start_ram_gib'] if stage == 'admission' else limits['min_running_ram_gib']
            if (time.monotonic()-started >= limits['max_wall_s']
                    or mem['peak_rss_bytes'] > limits['max_process_rss_gib']*1024**3
                    or mem['peak_private_commit_bytes'] > limits['max_process_commit_gib']*1024**3
                    or machine['available_ram_bytes'] < min_ram*1024**3 or output > limits['max_new_output_mib']*1024**2):
                raise RuntimeError(f'timing coverage resource cap at {stage}, update {update}')
            return mem

        track('admission')
        verify()
        plan, cfg, scales, folds, _, legacy, anchors, weights = prepare(ROOT, lambda: track('prepare'))
        held = read_held(plan)
        write_json(out/'execution_plan.json', {'design': plan, 'design_sha256': DESIGN_SHA256,
                   'frozen_preflight_sha256': PREFLIGHT_SHA256, 'source_sha256': sources, 'weights': weights})
        checkpoints, draws = {}, {}
        for name, fold in folds.items():
            track(name, 0)
            print(f'Fitting {name}: {len(fold.pairs.initial)} pairs, 900 fixed updates.', flush=True)
            checkpoints[name], draws[name] = fit(out, name, plan, cfg, scales, fold, anchors.dt_s, track,
                                                  started+limits['max_wall_s'])
        if len(set(draws.values())) != 1 or sum(updates.values()) != plan['total_updates']:
            raise ValueError('anchor draws or endpoint update count differ between arms')

        checkpoints['old_900'] = ROOT/plan['historical_reference']
        legacy_scores, timed_scores = {}, {}
        histories = {entry['episode_id']: entry['history_id'] for entry in
                     json.loads(_verified_bytes(ROOT, cfg['history_plan'], cfg['inputs_sha256'][cfg['history_plan']]))['episodes']}
        for name, checkpoint in checkpoints.items():
            track(f'evaluate_{name}')
            with np.load(checkpoint, allow_pickle=False) as saved:
                score = report(ResidualMLP.from_state_dict(saved), saved['known_mean'], saved['known_scale'],
                               legacy, anchors, histories)
            legacy_scores[name] = {**score, 'pairs': [row for row in score['pairs'] if row['split'] == 'validation']}
            timed_scores[name], model = timed_score(checkpoint, held, lambda: track(f'evaluate_{name}'))
            write_json(out/f'{name}_timing_rollouts.json', model)

        incumbent, candidate = 'held_offsets', 'held_offsets_plus_timing'
        protections = {key: regression(legacy_scores[key], legacy_scores[candidate]) for key in (incumbent, 'old_900')}
        comparison = {name: primary(values) for name, values in timed_scores.items()}
        historic = held[-1]['primary']
        for name, position in comparison['old_900'].items():
            if (position['model_jvp_vs_newton'] != historic[name]['model_jvp_vs_newton']
                    or position['model_fd_vs_newton'] != historic[name]['model_secant_vs_newton']):
                raise ValueError('historical response metrics did not reproduce on the frozen holdout')
        nominal = {name: next(iter(pair.values()))['absolute_model_nominal_error_norm'] for name, pair in comparison.items()}
        write_json(out/'comparison.json', {'legacy_development': legacy_scores, 'legacy_protections': protections,
                   'held_timing': timed_scores, 'held_timing_h32_position': comparison,
                   'nominal_h32_position_m': nominal, 'candidate_to_control_nominal_h32_ratio': nominal[candidate]/nominal[incumbent]})
        verify()
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline coverage fit unexpectedly loaded physics')
        mem = track('complete')
        write_json(out/'validation.json', {'status': 'completed', 'updates': updates, 'total_updates': sum(updates.values()),
                   'control_reproduced_exactly': True, 'identical_anchor_draws_sha256': draws[incumbent],
                   'legacy_protections_passed': {name: value['passed'] for name, value in protections.items()},
                   'checkpoint_sha256': {name: digest(path) for name, path in checkpoints.items()},
                   'artifacts_sha256': {p.name: digest(p) for p in sorted(out.iterdir())},
                   'wall_s': time.monotonic()-started, **mem, 'checkpoint_selection': False,
                   'test_episode_files_read': False, 'simulation_steps': 0, 'control_accepted': False})
    except BaseException as error:
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error), **last, 'updates': updates,
                   'wall_s': time.monotonic()-started, 'simulation_steps': 0, 'test_episode_files_read': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    output = parser.parse_args().output.resolve()
    if not output.relative_to(ROOT/'runs').parts:
        raise ValueError('choose a fresh run subdirectory')
    run(output)
