"""Check the frozen coverage ablation setup without optimizer updates."""
import argparse
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

from check_s3_actions import ROOT, digest, write_json
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES
from ball_retention.pair_loss import paired_loss
from ball_retention.resources import directory_bytes
from ball_retention.timing_coverage_fit import DESIGN, DESIGN_SHA256, prepare
from ball_retention.world_model import ResidualMLP


OUT = ROOT/'runs/s3_timing_coverage_fit_preflight_20261115'


def run(out):
    started = time.monotonic()
    out.mkdir(exist_ok=False)

    def guard():
        memory = process_memory(os.getpid())
        if (time.monotonic()-started >= 120 or memory['peak_rss_bytes'] > 1024**3
                or memory['peak_private_commit_bytes'] > 1.5*1024**3 or directory_bytes(out) > 16*1024**2):
            raise RuntimeError('coverage preparation exceeds static limits')
        return memory

    try:
        guard()
        plan, cfg, scales, folds, timed, legacy, anchors, weights = prepare(ROOT, guard)
        b = timed.batch
        features = np.stack((b.plus_known, b.minus_known))
        request = [KNOWN_NAMES.index(name) for name in ('roll_request', 'pitch_request')]
        np.testing.assert_array_equal(features[..., request], timed.branch_actions)
        np.testing.assert_array_equal(b.response_mask, b.noise_mask[..., None] &
                                      (np.abs(b.plus_target-b.minus_target)/2. >= b.response_floor))
        for entry in timed.entries:
            rows = {name: i for i, name in enumerate(timed.directions) if b.episode_ids[i] == entry['episode_id']}
            i, j = (rows[k] for k in ('pitch_pulse', 'pitch_reversal'))
            np.testing.assert_array_equal(features[:, i, :8], features[:, j, :8])
            if np.array_equal(features[:, i, 8:16], features[:, j, 8:16]):
                raise ValueError('post-pulse reversal must change causal inputs')
            np.testing.assert_array_equal(timed.branch_actions[:, list(rows.values()), 16:], 0.)

        loss = {}
        parameters = None
        lead_weights = np.zeros(legacy.horizon)
        lead_weights[np.asarray(cfg['pairs']['lead_steps'])-1] = cfg['pairs']['lead_weights']
        for name, fold in folds.items():
            model = ResidualMLP(len(KNOWN_NAMES), fold.state_mean, fold.state_scale, anchors.dt_s, plan['seed'],
                                hidden=cfg['network'][1], residual_gain=np.asarray(scales['residual_gain_si'])/fold.state_scale)
            if parameters is None:
                parameters = model.params
            else:
                for key in parameters:
                    np.testing.assert_array_equal(parameters[key], model.params[key])
            batch = fold.pairs
            known = (np.stack((batch.plus_known, batch.minus_known))-fold.known_mean)/fold.known_scale
            metrics, grad = paired_loss(model, model.normalize(batch.initial), known,
                                        np.stack((batch.plus_target, batch.minus_target)),
                                        np.stack((batch.plus_mask, batch.minus_mask)), batch.response_mask,
                                        batch.noise_mask, batch.response_floor, lead_weights, np.asarray(cfg['components']),
                                        np.asarray(cfg['response_scales_si']), cfg['pairs']['absolute_weight'],
                                        cfg['pairs']['response_weight'], gradients=False, sample_weights=fold.pair_weights,
                                        absolute_scale_si=np.asarray(scales['state_error_scale_si']))
            if grad is not None:
                raise RuntimeError('static paired loss should not compute gradients')
            loss[name] = metrics
            guard()

        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline preparation unexpectedly loaded physics')
        paths = {Path(__file__).resolve(), ROOT/DESIGN, ROOT/'scripts/check_s3_actions.py',
                 *(ROOT/'src/ball_retention').glob('*.py')}
        write_json(out/'preflight.json', {'status': 'passed', 'design_sha256': DESIGN_SHA256,
                   'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)},
                   'arm_pair_counts': plan['arms'], 'timed_pairs': len(b.initial), 'full_timed_pair_leads': int(b.pair_mask.sum()),
                   'timed_resolved_components': b.response_mask.sum(axis=(0, 1)).tolist(),
                   'nominal_noise_max_si': b.replay_noise.max(axis=(0, 1)).tolist(),
                   'weights': weights, 'legacy_validation': legacy.report['splits']['validation'],
                   'normalization_matches_control_checkpoint': True, 'initial_parameter_arrays_equal': True,
                   'causal_feature_prefix_and_zero_tail_verified': True, 'untrained_paired_loss': loss,
                   'simulation_steps': 0, 'training_updates': 0, 'test_episode_files_read': False,
                   'wall_s': time.monotonic()-started, **guard()})
    except BaseException as error:
        write_json(out/'stopped.json', {'reason': str(error), 'wall_s': time.monotonic()-started,
                   'simulation_steps': 0, 'training_updates': 0, 'test_episode_files_read': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    out = parser.parse_args().output.resolve()
    if not out.relative_to(ROOT/'runs').parts:
        raise ValueError('choose a fresh run subdirectory')
    run(out)
