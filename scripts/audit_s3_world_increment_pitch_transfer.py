"""Check whether the H9-H16 pitch increment deficit is shared across training groups."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, forecast, run_phase, write_json

import numpy as np

from audit_s3_world_increment_drift import half, require_close
from audit_s3_world_increment_generation import CASES, prior, propagate, world_head
from audit_s3_world_increment_newton_reference import required_residual, tool_transition
from audit_s3_world_increment_pitch_drift import load_extended
from audit_s3_world_increment_supervision_drift import load_endpoint
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, known_rollout
from deterministic_archive_metrics import _full_branch
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'world_increment_pitch_transfer'
WINDOW = slice(8, 16)
POSITION = slice(0, 3)


def alignment(a, b):
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom > 0. else None


def window_position(source, transform, world, dt):
    signal = np.zeros_like(source)
    signal[:, WINDOW] = source[:, WINDOW]
    vectors = propagate(signal, transform, world, dt).sum(axis=0)
    return half(vectors[:, -1, POSITION])


def snapshot(state, config, traces, endpoints, mesh):
    initial = _ball_state(state)
    dt = endpoints['supervised'][0].dt_s
    anchor = support_point(mesh, config['initial']['anchor_xy_tool_m'])[0]
    output = {}
    for action in CASES:
        keys = [f'{action}_{side}' for side in ('plus', 'minus')]
        truth = np.stack([traces[k]['raw'] for k in keys])
        measured_pre = prior(initial, truth)
        known = known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                              np.stack([traces[k]['actions'] for k in keys]), config, dt)
        models = {}
        for tag, (model, fold, *_) in endpoints.items():
            features = model.features(known, config, anchor, fold.known_mean, fold.known_scale)
            transform, world, shift = tool_transition(features)
            target, _ = required_residual(measured_pre, truth, transform, world, shift)
            free = forecast(model, np.broadcast_to(initial, (2, 9)), features, truth)
            local = world_head(model, features, measured_pre)
            rolled = world_head(model, features, prior(initial, free))
            response = half(truth[:, -1, POSITION])
            if np.linalg.norm(response) < 1e-5:
                raise ValueError('expected a strong H32 position response')
            axis = response/np.linalg.norm(response)
            models[tag] = {'target': window_position(target+model.inc_center, transform, world, dt),
                'local_error': window_position(local-target, transform, world, dt),
                'free_error': window_position(rolled-target, transform, world, dt),
                'terminal_error': half(free[:, -1, POSITION]-truth[:, -1, POSITION]),
                'correction': half(target-local)[WINDOW], 'transform': transform, 'world': world,
                'dt': dt, 'axis': axis}
        output[action] = models
    return output


def transfer(target, correction):
    increment = np.zeros(target['world'].shape[:2]+(3,))
    increment[0, WINDOW] = correction
    increment[1, WINDOW] = -correction
    effect = window_position(increment, target['transform'], target['world'], target['dt'])
    axis = target['axis']
    return {'applied_position_vector_m': effect.tolist(), 'applied_along_newton_h32_m': float(effect@axis),
            'window_position_error_norm_before_after_m': [float(np.linalg.norm(target['local_error']+x)) for x in (0., effect)],
            'terminal_position_error_norm_before_after_m': [float(np.linalg.norm(target['terminal_error']+x)) for x in (0., effect)]}


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    if plan['updates'] != 0 or plan['lead_window'] != [9, 16]:
        raise ValueError('unexpected pitch-transfer scope')
    held = read_held(read(plan['held_design']))
    capture, config, fields, traces, _ = held
    state = _parse_state(fields)
    mesh = load_tool(ROOT/'assets/shovel.STL')
    anchor = support_point(mesh, config['initial']['anchor_xy_tool_m'])[0]
    endpoints = {'supervised': load_endpoint(plan, 'supervised', held, anchor, guard),
                 'extended': load_extended(plan, held, anchor, guard)}
    if endpoints['supervised'][-1] != endpoints['extended'][-1]:
        raise ValueError('archive sampler changed')
    actual = snapshot(state, config, traces, endpoints, mesh)
    accepted = read(plan['accepted_newton_reference'])
    if accepted['development_case'] != capture['case']:
        raise ValueError('development case changed')
    for action, rows in actual.items():
        for tag, row in rows.items():
            ref = accepted['results'][action]['models'][tag]
            for key, expected in (('local_error', ref['split_learned_error']['on_measured_pre_state']),
                                  ('free_error', ref['learned_increment_error'])):
                require_close(row[key]@row['axis'], expected['window_projection_m']['9-16'])
    training = {}
    groups = {}
    for path in plan['timed_manifests']:
        for entry in read(path)['entries']:
            name, group = entry['episode_id'], entry['group_id']
            if entry['split'] != 'train' or name in training or group == capture['case']['group']:
                raise ValueError('unexpected archived training episode')
            if plan['files_sha256'][entry['source_config']] != entry['source_config_sha256']:
                raise ValueError('training config digest changed')

            def read_ref(ref):
                file = Path(path).parent/ref['path']
                if ref['sha256'] != plan['files_sha256'][file.as_posix()]:
                    raise ValueError('training branch digest changed')
                return read(file)

            by_label = {branch['label']: _full_branch(read_ref(branch), entry['snapshot_tick'])
                        for pair in entry['pairs'] if pair['direction_id'] in CASES for branch in pair['branches']}
            training[name] = snapshot(_parse_state(read_ref(entry['native_snapshot'])['state']),
                                      read(entry['source_config']), by_label, endpoints, mesh)
            groups.setdefault(group, []).append(name)
            guard(name, log=True)
    if len(training) != 3 or sorted(map(len, groups.values())) != [1, 2]:
        raise ValueError('unexpected set of training snapshots and groups')
    results = {}
    for action in CASES:
        by_model = {}
        for tag in endpoints:
            train = {ep: v[action][tag] for ep, v in training.items()}
            group_correction = {id_: np.mean([train[ep]['correction'] for ep in eps], axis=0)
                                for id_, eps in groups.items()}
            dev = actual[action][tag]
            scores = {}
            for group, eps in groups.items():
                donor, = [x for x in groups if x != group]
                for name in eps:
                    row = train[name]
                    scores[name] = {'group': group, 'donor_group': donor,
                        'target_cosine_vs_development': alignment(row['target'], dev['target']),
                        'local_error_cosine_vs_development': alignment(row['local_error'], dev['local_error']),
                        'target_along_own_newton_h32_m': float(row['target']@row['axis']),
                        'local_error_along_own_newton_h32_m': float(row['local_error']@row['axis']),
                        'free_error_along_own_newton_h32_m': float(row['free_error']@row['axis']),
                        'leave_group_out_correction': transfer(row, group_correction[donor])}
            by_model[tag] = {'training': scores,
                'development_target_along_newton_h32_m': float(dev['target']@dev['axis']),
                'development_local_error_along_newton_h32_m': float(dev['local_error']@dev['axis']),
                'development_free_error_along_newton_h32_m': float(dev['free_error']@dev['axis']),
                'equal_train_group_correction_on_development': transfer(dev, np.mean(list(group_correction.values()), axis=0))}
        results[action] = by_model
    write_json(out/'pitch_transfer.json', {'lead_window': plan['lead_window'], 'groups': groups, 'results': results,
        'convention': 'Corrections estimated only from the other training group (or equally from both groups for development). Terminal correction propagates with fixed recorded future learned increments; it is diagnostic only.'})
    return {'training_snapshots': len(training), 'training_groups': len(groups), 'fixed_endpoints': len(endpoints),
            'verified_development_reproduction': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
