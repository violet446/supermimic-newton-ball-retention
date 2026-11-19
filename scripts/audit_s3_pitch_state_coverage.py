"""Compare pitch increment errors across archived state, tool, and contact contexts."""
import argparse
from itertools import combinations
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np
from scipy.spatial.transform import Rotation

from audit_s3_tool_feature_clock import URDF, command_grid, world_to_tool
from audit_s3_world_increment_drift import half, require_close
from audit_s3_world_increment_generation import CASES, prior
from audit_s3_world_increment_pitch_drift import load_extended
from audit_s3_world_increment_pitch_transfer import WINDOW, alignment, snapshot, transfer
from audit_s3_world_increment_supervision_drift import load_endpoint
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state
from ball_retention.robot import Robot
from ball_retention.surface_diagnostic import SurfaceDiagnostic
from deterministic_archive_metrics import _full_branch
from response_contact_diagnosis import diagnose_contacts
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'pitch_state_coverage'
START = WINDOW.start
HORIZON = WINDOW.stop-START
BRANCHES = ('nominal_1', 'nominal_2', *(f'{case}_{side}' for case in CASES for side in ('plus', 'minus')))
DISTANCES = ('ball', 'tool', 'surface', 'combined')


def archived_tool(trace, robot, anchor, cfg, guard):
    before = trace['states'][START-1]
    after = trace['states'][WINDOW]
    dt = cfg['physics']['dt_s']
    substeps = 16
    ran = command_grid(robot, anchor, before, trace['commands'][WINDOW], substeps, guard)
    for key, expected in (
        ('origin', [s.tool.position for s in after]),
        ('omega', [s.tool.angular_velocity for s in after]),
        ('origin_velocity', [s.tool.linear_velocity_origin for s in after]),
        ('rotation', Rotation.from_quat([s.tool.quaternion_xyzw for s in after]).as_matrix()),
    ):
        np.testing.assert_allclose(ran[key][:, -1], expected, atol=2e-5, rtol=0.)
    midpoint = substeps//2
    rot = ran['rotation'][:, midpoint]
    gravity = np.broadcast_to([0., 0., -cfg['physics']['gravity_m_s2']], (HORIZON, 3))
    alpha = (ran['omega'][:, -1]-ran['omega'][:, 0])/(dt*substeps)
    accel = (ran['anchor_velocity'][:, -1]-ran['anchor_velocity'][:, 0])/(dt*substeps)
    return np.concatenate([world_to_tool(rot, v) for v in (gravity, ran['omega'][:, midpoint], alpha, accel)], axis=-1)


def context(name, group, state, config, parsed, raw, layout, capture, endpoints, mesh, robot, evidence, guard):
    anchor = support_point(mesh, config['initial']['anchor_xy_tool_m'])[0]
    surface = SurfaceDiagnostic(mesh, config['ball']['radius_m'], config['initial']['tool_rotation_world'])
    correlation = diagnose_contacts({**capture, 'branch_order': list(BRANCHES), 'case': {'tick': 168},
                                    'directions': {k: capture['directions'][k] for k in CASES}}, raw, layout)
    tool = {k: archived_tool(parsed[k], robot, anchor, config, guard) for k in BRANCHES[2:]}
    modeled = snapshot(state, config, parsed, endpoints, mesh)
    items = {}
    for label in CASES:
        keys = [f'{label}_{s}' for s in ('plus', 'minus')]
        states = np.stack([parsed[k]['raw'] for k in keys])
        pre = prior(_ball_state(state), states)[:, WINDOW]
        geom = [[surface.measure(p) for p in b[:, :3]] for b in pre]
        gaps = np.asarray([[x['sphere_distance_gap_m'] for x in b] for b in geom])
        normals = np.asarray([[x['normal_tool'] for x in b] for b in geom])
        previous_counts = np.asarray([[row['candidates_by_substep'] for row in
                                    correlation['branches'][k]['leads'][START-1:WINDOW.stop-1]] for k in keys])
        current_comparison = correlation['comparisons'][label]['plus_vs_minus']['leads'][WINDOW]
        response = np.asarray(evidence[label]['target_m_s'])[WINDOW]
        continuous = np.concatenate((normals, gaps[..., None]), axis=-1)
        items[label] = {'ball': pre, 'tool': np.stack([tool[k] for k in keys]), 'surface': continuous,
            'prior_contact_counts': previous_counts.tolist(),
            'current_contact_count_differences': [row['candidate_count_different_substeps'] for row in current_comparison],
            'current_shape_pair_differences': [row['shape_set_different_substeps'] for row in current_comparison],
            'response': response, 'strong': evidence[label]['strong'][WINDOW],
            'models': modeled[label]}
        exact = -np.asarray(evidence[label]['per_step_errors']['supervised']['head_error_m_s'])[WINDOW]
        # The earlier coverage archive predated the extended model. Its transfer scores are checked below.
        np.testing.assert_allclose(modeled[label]['supervised']['correction'], exact, atol=2e-6, rtol=0.)
    return {'name': name, 'group': group, 'items': items}


def distances(first, second, label, j, scales):
    a, b = [x['items'][label] for x in (first, second)]
    values = {key: float(np.sqrt(np.mean(((a[key][:, j]-b[key][:, j])/scale)**2)))
              for key, scale in scales.items()}
    values['combined'] = float(np.sqrt(np.mean([x*x for x in values.values()])))
    return values


def closest(target, donors, label, j, method, scales):
    metric = [distances(target, x, label, j, scales) for x in donors]
    i = min(range(len(donors)), key=lambda a: metric[a][method])
    return donors[i], metric[i]


def coverage(development, train, label, scales):
    rows = []
    item = development['items'][label]
    for j in range(HORIZON):
        cross = [distances(a, b, label, j, scales) for a, b in combinations(train, 2) if a['group'] != b['group']]
        max_cross = {k: max(v[k] for v in cross) for k in DISTANCES}
        matches = {}
        for name in DISTANCES:
            ref, ds = closest(development, train, label, j, name, scales)
            near = ref['items'][label]
            axis = item['response'][j]
            state_delta = item['ball'][:, j]-near['ball'][:, j]
            normal_cos = np.sum(item['surface'][:, j, :3]*near['surface'][:, j, :3], axis=-1)
            matches[name] = {'snapshot': ref['name'], 'distances': ds, 'training_cross_group_max': max_cross[name],
                'beyond_training_cross_group_max': ds[name] > max_cross[name],
                'state_block_distances_si_plus_minus': np.linalg.norm(state_delta.reshape(2, 3, 3), axis=-1).tolist(),
                'nearest_surface_normal_angle_deg_plus_minus': np.rad2deg(np.arccos(np.clip(normal_cos, -1., 1.))).tolist(),
                'nearest_surface_gap_difference_m_plus_minus': (item['surface'][:, j, 3]-near['surface'][:, j, 3]).tolist(),
                'newton_label_cosine': alignment(axis, near['response'][j]),
                'increment_error_cosine': {m: alignment(item['models'][m]['correction'][j], near['models'][m]['correction'][j])
                                           for m in item['models']},
                'same_previous_contact_counts': item['prior_contact_counts'][0][j] == near['prior_contact_counts'][0][j]
                                                and item['prior_contact_counts'][1][j] == near['prior_contact_counts'][1][j]}
        rows.append({'lead': j+START+1, 'strong': bool(item['strong'][j]), 'matches': matches})
    return rows


def matched_transfers(target, donors, label, model, scales):
    target_row = target['items'][label]['models'][model]
    group_names = dict.fromkeys(x['group'] for x in donors)
    uniform = np.mean([np.mean([x['items'][label]['models'][model]['correction'] for x in donors if x['group'] == group],
                               axis=0) for group in group_names], axis=0)
    corrections = {'group_average': uniform}
    for name in ('ball', 'combined'):
        sources = [closest(target, donors, label, j, name, scales)[0] for j in range(HORIZON)]
        corrections[name] = np.stack([x['items'][label]['models'][model]['correction'][j] for j, x in enumerate(sources)])
    return {k: transfer(target_row, v) for k, v in corrections.items()}


def run(plan, out, guard):
    def read(p):
        return json.loads(bound(plan, p))

    if plan['updates'] != 0 or plan['lead_window'] != [9, 16]:
        raise ValueError('unexpected coverage scope')
    held_design = read(plan['held_design'])
    held = read_held(held_design)
    capture, cfg, fields, traces, acceptance = held
    held_root = Path(held_design['held_timing_run']).parent
    mesh, robot = load_tool(ROOT/'assets/shovel.STL'), Robot(ROOT/URDF)
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    endpoints = {'supervised': load_endpoint(plan, 'supervised', held, anchor, guard),
                 'extended': load_extended(plan, held, anchor, guard)}
    if endpoints['supervised'][-1] != endpoints['extended'][-1]:
        raise ValueError('archive sampler changed')
    scales = {'ball': np.asarray(read(plan['physical_scales'])['state_error_scale_si']),
              'tool': endpoints['supervised'][1].known_scale[8:20],
              'surface': np.asarray(plan['normal_and_gap_reference_units'])}
    if not all(np.all(np.isfinite(v) & (v > 0.)) for v in scales.values()):
        raise ValueError('invalid diagnostic physical scales')
    late = read(plan['archived_late_coverage'])

    def observed(path, sha):
        if plan['files_sha256'][Path(path).as_posix()] != sha:
            raise ValueError(f'record digest disagrees with archive: {path}')
        return read(path)

    dev_raw = {key: observed(held_root/'worker'/f'{key}.json', acceptance['artifacts_sha256'][f'worker/{key}.json'])
               for key in BRANCHES}
    layout = observed(held_root/'worker/native_layout.json', acceptance['artifacts_sha256']['worker/native_layout.json'])
    dev = context('development', capture['case']['group'], _parse_state(fields), cfg, traces, dev_raw, layout, capture,
                  endpoints, mesh, robot, late['development'], guard)
    guard('development', log=True)
    training = []
    for path in plan['timed_manifests']:
        folder = Path(path).parent
        captured = read(folder/'validation.json')
        for entry in read(path)['entries']:
            name, group = entry['episode_id'], entry['group_id']
            if entry['split'] != 'train' or entry['snapshot_tick'] != 168 or group == dev['group']:
                raise ValueError('unexpected training capture')
            refs = [*entry['nominal_branches'], *(b for pair in entry['pairs'] for b in pair['branches'] if b['label'] in BRANCHES)]
            raw = {b['label']: observed(folder/b['path'], b['sha256']) for b in refs}
            parsed = {k: _full_branch(v, entry['snapshot_tick']) for k, v in raw.items()}
            snapshot_path = entry['native_snapshot']
            state = _parse_state(observed(folder/snapshot_path['path'], snapshot_path['sha256'])['state'])
            layout_path = (Path(snapshot_path['path']).parent/'native_layout.json').as_posix()
            layout = observed(folder/layout_path, captured['artifacts_sha256'][layout_path])
            conf = observed(entry['source_config'], entry['source_config_sha256'])
            training.append(context(name, group, state, conf, parsed, raw, layout, capture,
                                    endpoints, mesh, robot, late['training'][name], guard))
            guard(name, log=True)
    if len(training) != 3 or len({x['group'] for x in training}) != 2:
        raise ValueError('unexpected training coverage size')
    previous = read(plan['accepted_pitch_transfer'])['results']
    result = {}
    for label in CASES:
        matches = coverage(dev, training, label, scales)
        transfers = {}
        for model in endpoints:
            observed_train = {x['name']: matched_transfers(x, [d for d in training if d['group'] != x['group']],
                                                         label, model, scales) for x in training}
            observed_dev = matched_transfers(dev, training, label, model, scales)
            for new, old in [(observed_dev['group_average'], previous[label][model]['equal_train_group_correction_on_development']),
                             *((r['group_average'], previous[label][model]['training'][ep]['leave_group_out_correction'])
                               for ep, r in observed_train.items())]:
                require_close(new['terminal_position_error_norm_before_after_m'],
                              np.asarray(old['terminal_position_error_norm_before_after_m']))
            transfers[model] = {'training_leave_group_out': observed_train, 'development': observed_dev}
        result[label] = {'coverage': matches, 'transfers': transfers, 'contact': {
            x['name']: {k: x['items'][label][k] for k in ('prior_contact_counts', 'current_contact_count_differences',
                                                        'current_shape_pair_differences')}
            for x in (dev, *training)}}
    write_json(out/'pitch_state_coverage.json', {'scales': {k: a.tolist() for k, a in scales.items()}, 'results': result,
        'definitions': {'distance': 'Branchwise RMS for the same action and lead; combined gives equal mass to ball, tool and surface.',
                        'count': 'Archived collision candidates, not verified solver constraints.'}})
    return {'training_snapshots': 3, 'training_groups': 2, 'development_snapshots': 1,
            'executed_fk_checked': True, 'baseline_transfer_reproduced': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
