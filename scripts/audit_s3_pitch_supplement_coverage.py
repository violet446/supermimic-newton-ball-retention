"""Compare pitch coverage before and after one accepted deterministic capture."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np

from audit_s3_ball_velocity_components import frame_series
from audit_s3_pitch_state_coverage import BRANCHES, URDF, context, coverage, matched_transfers
from audit_s3_world_increment_drift import half, require_close
from audit_s3_world_increment_generation import CASES, prior, world_head
from audit_s3_world_increment_pitch_drift import load_extended
from audit_s3_world_increment_supervision_drift import load_endpoint
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, known_rollout
from ball_retention.robot import Robot
from deterministic_archive_metrics import _full_branch
from preflight_s3_world_increment_supervision import SETTINGS
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'pitch_supplement_coverage'


def direct_evidence(state, cfg, traces, endpoint, anchor):
    model, fold, *_ = endpoint
    dt = model.dt_s
    gravity = np.array([0., 0., -cfg['physics']['gravity_m_s2']*dt])

    def measured(key):
        return np.diff(frame_series(state, traces[key], dt)[3], axis=0)-gravity

    noise = np.linalg.norm(half(np.stack([measured(k) for k in BRANCHES[:2]])), axis=-1)
    threshold = np.maximum(SETTINGS['floor_m_s'], SETTINGS['noise_multiplier']*noise)
    results = {}
    for label in CASES:
        keys = [f'{label}_{s}' for s in ('plus', 'minus')]
        targets = np.stack([measured(k) for k in keys])
        y = half(targets)
        truth = np.stack([traces[k]['raw'] for k in keys])
        known = known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                              np.stack([traces[k]['actions'] for k in keys]), cfg, dt)
        features = model.features(known, cfg, anchor, fold.known_mean, fold.known_scale)
        predicted = model.inc_center+world_head(model, features, prior(_ball_state(state), truth))
        results[label] = {'target_m_s': y.tolist(), 'strong': (np.linalg.norm(y, axis=-1) >= 10*threshold).tolist(),
            'per_step_errors': {'supervised': {'head_error_m_s': half(predicted-targets).tolist()}}}
    return results


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    def observed(path, expected):
        if plan['files_sha256'][Path(path).as_posix()] != expected:
            raise ValueError(f'accepted archive digest changed: {path}')
        return read(path)

    held_design = read(plan['held_design'])
    held = read_held(held_design)
    capture, cfg, fields, traces, held_status = held
    folder = Path(held_design['held_timing_run']).parent
    mesh, robot = load_tool(ROOT/'assets/shovel.STL'), Robot(ROOT/URDF)
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    endpoints = {'supervised': load_endpoint(plan, 'supervised', held, anchor, guard),
                 'extended': load_extended(plan, held, anchor, guard)}
    late = read(plan['archived_late_coverage'])
    old = read(plan['accepted_state_coverage'])
    scales = {key: np.asarray(v) for key, v in old['scales'].items()}
    raw = {k: observed(folder/'worker'/f'{k}.json', held_status['artifacts_sha256'][f'worker/{k}.json'])
           for k in BRANCHES}
    layout_path = 'worker/native_layout.json'
    layout = observed(folder/layout_path, held_status['artifacts_sha256'][layout_path])
    dev = context('development', capture['case']['group'], _parse_state(fields), cfg, traces, raw, layout, capture,
                  endpoints, mesh, robot, late['development'], guard)
    guard('development', log=True)

    def cohort(path, new):
        path = Path(path)
        parent = path.parent
        status = read(parent/'validation.json')
        manifest = observed(path, status['artifacts_sha256'][path.name])
        for entry in manifest['entries']:
            ep, group = entry['episode_id'], entry['group_id']
            if (entry['split'] != 'train' or group == dev['group'] or entry['snapshot_tick'] != 168
                    or (new and ep != plan['supplement_episode'])):
                raise ValueError('unexpected coverage case')
            refs = [*entry['nominal_branches'], *(b for pair in entry['pairs'] for b in pair['branches'] if b['label'] in BRANCHES)]
            raw_rows = {b['label']: observed(parent/b['path'], b['sha256']) for b in refs}
            parsed = {k: _full_branch(v, 168) for k, v in raw_rows.items()}
            record = entry['native_snapshot']
            state = _parse_state(observed(parent/record['path'], record['sha256'])['state'])
            layout_rel = (Path(record['path']).parent/'native_layout.json').as_posix()
            layout = observed(parent/layout_rel, status['artifacts_sha256'][layout_rel])
            conf = observed(entry['source_config'], entry['source_config_sha256'])
            anchor = support_point(mesh, conf['initial']['anchor_xy_tool_m'])[0]
            labels = direct_evidence(state, conf, parsed, endpoints['supervised'], anchor) if new else late['training'][ep]
            yield context(ep, group, state, conf, parsed, raw_rows, layout, capture, endpoints, mesh, robot, labels, guard)
            guard(ep, log=True)

    training = [row for p in plan['timed_manifests'] for row in cohort(p, False)]
    accepted = read(Path(plan['supplement_manifest']).parent/'validation.json')
    if (accepted['status'] != 'completed' or accepted['physics_steps'] != 6784
            or accepted['pairs'] != 3 or accepted['branches'] != 8
            or accepted['test_episode_files_read'] is not False):
        raise ValueError('published supplement did not pass the deterministic capture')
    extra, = cohort(plan['supplement_manifest'], True)
    expanded = [*training, extra]
    if len(training) != 3 or sorted(sum(x['group'] == g for x in expanded) for g in {x['group'] for x in expanded}) != [2, 2]:
        raise ValueError('expected a two-by-two training cohort')
    result = {}
    for action in CASES:
        before = coverage(dev, training, action, scales)
        if before != old['results'][action]['coverage']:
            raise ValueError('existing coverage did not reproduce')
        after = coverage(dev, expanded, action, scales)
        model_scores = {}
        for tag in endpoints:
            archived = old['results'][action]['transfers'][tag]
            prior_dev = matched_transfers(dev, training, action, tag, scales)
            for method, value in prior_dev.items():
                require_close(value['terminal_position_error_norm_before_after_m'],
                              np.asarray(archived['development'][method]['terminal_position_error_norm_before_after_m']))
            model_scores[tag] = {'development': matched_transfers(dev, expanded, action, tag, scales),
                'training_leave_group_out': {x['name']: matched_transfers(
                    x, [d for d in expanded if d['group'] != x['group']], action, tag, scales) for x in expanded}}
        supplement = extra['items'][action]
        result[action] = {'before': before, 'after': after, 'transfers_after': model_scores,
                          'supplement_strong': supplement['strong'], 'supplement_target_m_s': supplement['response'].tolist(),
                          'supplement_prior_contact_counts': supplement['prior_contact_counts'],
                          'supplement_current_contact_count_differences': supplement['current_contact_count_differences']}
    write_json(out/'supplement_coverage.json', {'supplement_episode': extra['name'], 'results': result})
    return {'training_snapshots_before': 3, 'training_snapshots_after': 4,
            'previous_coverage_reproduced': True, 'matching_group_weighting': 'equal groups for the diagnostic average'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
