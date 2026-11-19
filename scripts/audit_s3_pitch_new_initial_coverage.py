"""Screen archived training groups for distinct coverage at the frozen pitch clock."""
import argparse
from itertools import combinations
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np

from audit_s3_world_increment_drift import require_close
from audit_s3_world_increment_pitch_transfer import WINDOW
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool
from ball_retention.model_data import _ball_state, known_rollout
from ball_retention.pair_collection import _initial_key
from ball_retention.surface_diagnostic import SurfaceDiagnostic
from deterministic_archive_metrics import _full_branch
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'pitch_new_initial_coverage'
TICK = 168
METRICS = ('ball', 'tool', 'surface')


def descriptors(start, truth, actions, cfg, mesh):
    state = np.vstack((_ball_state(start), truth))[WINDOW]
    dt = 1/cfg['physics']['control_hz']
    tool = known_rollout(start.time_s, start.tilt_rad, start.tilt_rate_rad_s, actions, cfg, dt)[WINDOW, 8:20]
    surface = SurfaceDiagnostic(mesh, cfg['ball']['radius_m'], cfg['initial']['tool_rotation_world'])
    geometry = [surface.measure(p) for p in state[:, :3]]
    data = np.asarray([(*v['normal_tool'], v['sphere_distance_gap_m']) for v in geometry])
    return {'ball': state, 'tool': tool, 'surface': data}


def distance(a, b, scales):
    columns = {k: np.sqrt(np.mean(((a[k]-b[k])/scales[k])**2, axis=-1)) for k in METRICS}
    return {**columns, 'combined': np.sqrt(np.mean([x*x for x in columns.values()], axis=0))}


def describe(values):
    return {k: {'median': float(np.median(v)), 'by_lead': np.asarray(v).tolist()} for k, v in values.items()}


def nominal_from_legacy(row):
    a, b = sorted((x for x in row['branches'] if x['axis'] is None), key=lambda x:x['repeat'])
    expected = row['nominal_actions']
    for trace in (a, b):
        if trace['actions'] != expected or trace['substeps'] != [16]*32 or trace['terminated'] or trace['truncated']:
            raise ValueError('archived nominal trace is incomplete or uses a different controller request')
    states = np.stack([x['states_raw'] for x in (a, b)])
    magnitude = np.abs(np.diff(states, axis=0)[0]).reshape(-1, 3, 3).max(axis=(0, 2))
    half_position = .5*np.linalg.norm(states[0, [7, 15, 23, 31], :3]-states[1, [7, 15, 23, 31], :3], axis=-1)
    return states.mean(axis=0), bool(np.all(magnitude <= [1e-5, 1e-3, 2e-2]) and np.all(half_position <= 1e-6)), half_position


def nominal_from_native(raw):
    counts = [row['candidates'] for row in raw['contacts'][8*16:16*16]]
    if len(counts) != 8*16:
        raise ValueError('missing accepted nominal collision-candidate record')
    return [min(counts), max(counts)]


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    def observed(path, expected):
        if plan['files_sha256'][Path(path).as_posix()] != expected:
            raise ValueError(f'archive digest changed: {path}')
        return read(path)

    capture, cfg, fields, traces, evidence = read_held(read(plan['held_design']))
    if plan['updates'] != 0 or capture['case']['tick'] != TICK:
        raise ValueError('unexpected coverage scope')
    held_run = Path(read(plan['held_design'])['held_timing_run']).parent
    mesh = load_tool(ROOT/'assets/shovel.STL')
    scales = {k: np.asarray(v) for k, v in read(plan['accepted_context'])['scales'].items()}
    require_close(traces['nominal_1']['raw'], traces['nominal_2']['raw'])
    dev = descriptors(_parse_state(fields), traces['nominal_1']['raw'], traces['nominal_1']['actions'], cfg, mesh)
    raw_dev = observed(held_run/'worker/nominal_1.json', evidence['artifacts_sha256']['worker/nominal_1.json'])
    dev_contact = nominal_from_native(raw_dev)
    native, groups, physical = {}, {}, {}
    for path in plan['native_training']:
        folder = Path(path).parent
        status = read(folder/'validation.json')
        if status['status'] != 'completed' or status['test_episode_files_read'] is not False:
            raise ValueError('unaccepted native training archive')
        for row in observed(path, status['artifacts_sha256'][Path(path).name])['entries']:
            if row['split'] != 'train' or row['group_id'] == capture['case']['group'] or row['snapshot_tick'] != TICK:
                raise ValueError('unexpected native training identity')
            name = row['episode_id']
            config = observed(row['source_config'], row['source_config_sha256'])
            saved = row['native_snapshot']
            start = _parse_state(observed(folder/saved['path'], saved['sha256'])['state'])
            n0 = row['nominal_branches'][0]
            trace = observed(folder/n0['path'], n0['sha256'])
            parsed = _full_branch(trace, TICK)
            native[name] = {'values': descriptors(start, parsed['raw'], parsed['actions'], config, mesh),
                            'nominal_contact_candidate_range': nominal_from_native(trace)}
            groups[name], physical[name] = row['group_id'], _initial_key(config)
            guard(name)
    if len(native) != 4 or len(set(groups.values())) != 2:
        raise ValueError('unexpected established native cohort')
    before = {key: distance(row['values'], dev, scales) for key, row in native.items()}
    nearest = {k: np.min(np.stack([x[k] for x in before.values()]), axis=0) for k in before[next(iter(before))]}
    training_max = {}
    cross = [distance(native[a]['values'], native[b]['values'], scales) for a, b in combinations(native, 2)
             if groups[a] != groups[b]]
    for k in nearest:
        training_max[k] = np.max(np.stack([v[k] for v in cross]), axis=0)
    result, control = [], {}
    seen = set()
    for source in read(plan['source_collection'])['sources']:
        manifest = Path(source['manifest'])
        catalog = observed(manifest, source['sha256'])
        for entry in catalog['entries']:
            if entry['split'] != 'train':
                continue
            if entry['group_id'] == capture['case']['group']:
                raise ValueError('development group leaked into candidate scope')
            cfg_path = (manifest.parent/entry['config_path']).as_posix()
            trace_path = (manifest.parent/entry['path']).as_posix()
            if trace_path in seen:
                continue  # Same trajectories occur in both amplitude manifests.
            seen.add(trace_path)
            config = observed(cfg_path, entry['config_sha256'])
            if plan['files_sha256'][trace_path] != entry['sha256']:
                raise ValueError('legacy archive digest disagrees with manifest')
            raw_rows = [json.loads(x) for x in bound(plan, trace_path).splitlines()]
            key = _initial_key(config)
            if (key == plan['development_initial_key']
                    or [row['snapshot_tick'] for row in raw_rows] != entry['snapshot_ticks']):
                raise ValueError('legacy physical identity or archive tick mismatch')
            for row in raw_rows:
                tick = row['snapshot_tick']
                name = entry['episode_id']
                y, acceptable, noise = nominal_from_legacy(row)
                start = _parse_state(row['initial_state'])
                require_close(_ball_state(start), np.asarray(row['initial_raw']))
                values = descriptors(start, y, row['nominal_actions'], config, mesh)
                ds = distance(values, dev, scales)
                record = {'episode': name, 'group': entry['group_id'], 'physical_initial_key': key,
                          'tick': tick, 'clock_matches': tick == TICK, 'seed': entry.get('seed'),
                          'config': cfg_path, 'trace': trace_path, 'manifest': manifest.as_posix(),
                          'new_physical_initial_group': key not in physical.values(),
                          'nominal_repeatability_passed': bool(acceptable and np.all(np.isfinite(y))),
                          'nominal_position_half_spread_m': noise.tolist(),
                          'archived_collision_candidates_available': all('contacts' in b for b in row['branches']),
                          'distance': describe(ds),
                          'beats_existing_nearest_leads': {k: int(np.sum(v < nearest[k])) for k, v in ds.items()}}
                if name in native and tick == TICK:
                    control[name] = describe(distance(values, native[name]['values'], scales))
                elif record['new_physical_initial_group']:
                    result.append(record)
            guard(entry['episode_id'], log=True)
    if len(control) != 4 or len(result) != 8:
        raise ValueError('unexpected archived candidate inventory')
    write_json(out/'initial_coverage.json', {'development_nominal_contact_candidate_range': dev_contact,
        'native_training': {k: {'distance': describe(before[k]), 'contact_candidate_range': x['nominal_contact_candidate_range']}
                            for k, x in native.items()},
        'training_cross_group_max': describe(training_max), 'frozen_nearest_distance': describe(nearest),
        'legacy_vs_native_same_case': control, 'candidates': result,
        'method': 'Nominal raw ball and nearest-surface geometry from archives; analytic requested tool kinematics for every case. Compare same-clock rows separately. Missing collision candidates remain unknown.'})
    return {'native_training_snapshots': len(native), 'archived_new_group_snapshots': len(result),
            'same_clock_new_group_snapshots': sum(x['clock_matches'] for x in result)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
