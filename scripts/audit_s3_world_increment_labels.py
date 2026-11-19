"""Inventory world velocity increment labels and test causal recovery of legacy targets."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np
from scipy.spatial.transform import Rotation

from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, _rotation, filter_advance
from ball_retention.pair_data import _branch
from ball_retention.surface_diagnostic import SurfaceDiagnostic
from ball_retention.task import reference
from ball_retention.timed_pair_data import prepare_timed_pairs
from train_s3_local_velocity_increment import bound


PHASE = 'world_increment_labels'
WINDOW = slice(7, 24)
FLOOR = 1e-5
NOISE_MULTIPLIER = 5.
MAX_RECOVERY_RELATIVE_RMS = .1


def world_to_local(rotation, value):
    return np.einsum('nji,nj->ni', rotation, value)


def to_world(rotation, value):
    return np.einsum('nij,nj->ni', rotation, value)


def analytic_pose(state, actions, cfg, anchor, dt):
    time = state.time_s+np.arange(len(actions)+1)*dt
    base = cfg['initial']['tool_rotation_world']
    tau = cfg['action']['tilt_filter_time_constant_s']
    limit = np.deg2rad(cfg['action']['tilt_limit_deg'])
    tilt, rate = np.asarray(state.tilt_rad), np.asarray(state.tilt_rate_rad_s)
    tilts, rates = [tilt], [rate]
    for action in actions:
        tilt, rate, _ = filter_advance(tilt, rate, limit*action, tau, dt)
        tilts.append(tilt)
        rates.append(rate)
    rot, roll_axis = _rotation(np.asarray(tilts), np.asarray(base))
    rates = np.asarray(rates)
    omega = roll_axis*rates[:, :1]+rates[:, 1:]*np.array([0., 1., 0.])
    origin_v = np.asarray([reference(t, cfg)['anchor_velocity'] for t in time])-np.cross(omega, rot@anchor)
    rot[0] = Rotation.from_quat(state.tool.quaternion_xyzw).as_matrix()
    origin_v[0], omega[0] = state.tool.linear_velocity_origin, state.tool.angular_velocity
    return rot, origin_v, omega


def labels(state, trace, cfg, anchor, surface, dt, count):
    raw = np.vstack((_ball_state(state), trace['states_raw'][:count]))
    actual = trace.get('states')
    gravity_step = np.array([0., 0., -cfg['physics']['gravity_m_s2']*dt])
    reference_normals = np.asarray([surface.measure(p)['normal_tool'] for p in raw[:-1, :3]])
    frames = {'analytic': analytic_pose(state, np.asarray(trace['actions'][:count]), cfg, anchor, dt)}
    if actual is not None:
        states = [state]+[_parse_state(x) for x in actual[:count]]
        times = np.asarray([s.time_s for s in states])
        np.testing.assert_allclose(np.diff(times), dt, atol=1e-12, rtol=0.)
        tools = [s.tool for s in states]
        frames['measured'] = (Rotation.from_quat([x.quaternion_xyzw for x in tools]).as_matrix(),
                              np.asarray([x.linear_velocity_origin for x in tools]),
                              np.asarray([x.angular_velocity for x in tools]))
        velocities = np.asarray([s.ball.linear_velocity_origin for s in states])
    output, closed = {}, None
    for key, (rot, origin, omega) in frames.items():
        lever = np.cross(omega, to_world(rot, raw[:, :3]))
        recovered = origin+to_world(rot, raw[:, 3:6])+lever
        if key == 'measured':
            closed = float(np.max(abs(recovered-velocities)))
            np.testing.assert_allclose(recovered, velocities, atol=1e-10, rtol=0.)
            recovered = velocities
        net = np.diff(recovered, axis=0)-gravity_step
        reconstructed = world_to_local(rot[1:], recovered[:-1]+gravity_step+net-origin[1:]-lever[1:])
        np.testing.assert_allclose(reconstructed, raw[1:, 3:6], atol=1e-10, rtol=0.)
        normal = to_world(rot[:-1], reference_normals)
        tangent = net-normal*np.einsum('ij,ij->i', normal, net)[:, None]
        output[key] = np.stack((net, tangent), axis=1)
    return output, closed


def rms(v):
    return float(np.sqrt(np.mean(np.sum(v*v, axis=-1)))) if len(v) else None


def pair_row(meta, signed, base):
    count = min(len(x) for x in (*signed, *base))
    a, b, n0, n1 = (x[:count] for x in (*signed, *base))
    half, nominal = (a-b)*.5, (n0-n1)*.5
    signal = np.linalg.norm(half[WINDOW, 1], axis=1)
    floor = np.maximum(FLOOR, NOISE_MULTIPLIER*np.linalg.norm(nominal[WINDOW, 1], axis=1))
    return {**meta, 'paired_full_steps': count, 'window_steps': len(signal),
            'net_rms_m_s': rms(np.concatenate((a[WINDOW, 0], b[WINDOW, 0]))),
            'tangent_half_rms_m_s': rms(half[WINDOW, 1]),
            'nominal_tangent_half_rms_m_s': rms(nominal[WINDOW, 1]),
            'resolved_window_steps': int(np.count_nonzero(signal >= floor)),
            'strong_window_steps': int(np.count_nonzero(signal >= 10.*floor))}, half


def complete_count(trace, substeps):
    return sum(n == substeps for n in trace['substeps'])


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    if plan['updates'] != 0 or plan['floor_m_s'] != FLOOR or plan['max_recovery_relative_rms'] != MAX_RECOVERY_RELATIVE_RMS:
        raise ValueError('unexpected audit scope')
    collection = read(plan['collection'])
    rows, snapshot_count, max_error = [], 0, 0.
    cache = {}
    mesh = load_tool(ROOT/'assets/shovel.STL')

    def setup(cfg):
        return (support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0],
                SurfaceDiagnostic(mesh, cfg['ball']['radius_m'], cfg['initial']['tool_rotation_world']))

    def checked(path, expected):
        if plan['files_sha256'][Path(path).as_posix()] != expected:
            raise ValueError(f'archive reference hash changed: {path}')
        return bound(plan, path)

    for source in collection['sources']:
        manifest_path = Path(source['manifest'])
        manifest = json.loads(checked(manifest_path, source['sha256']))
        dt, substeps, amplitude = manifest['dt_s'], manifest['control_substeps'], manifest['amplitude']
        for entry in manifest['entries']:
            if entry['split'] != 'train':
                continue
            cfg = json.loads(checked(manifest_path.parent/entry['config_path'], entry['config_sha256']))
            anchor, surface = setup(cfg)
            path = manifest_path.parent/entry['path']
            for line in checked(path, entry['sha256']).splitlines():
                record = json.loads(line)
                state = _parse_state(record['initial_state'])
                tick = record['snapshot_tick']
                branches = {(x['axis'], x['offset'], x['repeat']):x for x in record['branches']}
                data = {}
                for axis, offset, repeat in [(None, 0., 0), (None, 0., 1)]+[(a,s*amplitude,0) for a in (0,1) for s in (1,-1)]:
                    branch = branches[axis,offset,repeat]
                    if 'states' in branch:
                        raise ValueError('legacy full endpoint inventory changed')
                    request = np.asarray(record['nominal_actions']).copy()
                    if axis is not None:
                        request[:, axis] += offset
                    _branch(branch, request, dt, substeps)
                    key = (path, tick, axis, offset, repeat)
                    if key not in cache:
                        cache[key] = labels(state, branch, cfg, anchor, surface, dt*substeps, complete_count(branch, substeps))[0]['analytic']
                    data[axis,offset,repeat] = cache[key]
                for axis, action in enumerate(('held_roll','held_pitch')):
                    meta = {'episode':entry['episode_id'],'group':entry['group_id'],'tick':tick,
                            'history':cfg.get('history_capture', {}).get('history_id', 'original'),
                            'action':action, 'source':'analytic', 'amplitude':amplitude}
                    row, _ = pair_row(meta, [data[axis,s*amplitude,0] for s in (1,-1)], [data[None,0.,x] for x in (0,1)])
                    rows.append(row)
                guard()
    if len(rows) != 40:
        raise ValueError('legacy training inventory differs from frozen 40 rows')

    for path in plan['timed_manifests']:
        manifest = read(path)
        prepared = prepare_timed_pairs(ROOT, path, plan['files_sha256'][path])
        if prepared.dt_s != 1/60:
            raise ValueError('unexpected timed clock')
        for entry in manifest['entries']:
            snapshot_count += 1
            folder = Path(path).parent
            load_ref = lambda ref:json.loads(checked(folder/ref['path'], ref['sha256']))
            cfg = json.loads(checked(entry['source_config'], entry['source_config_sha256']))
            anchor, surface = setup(cfg)
            state = _parse_state(load_ref(entry['native_snapshot'])['state'])
            dt, substeps = entry['clock']['physics_dt_s'], entry['clock']['control_substeps']
            def extract(ref):
                data, closed = labels(state, load_ref(ref), cfg, anchor, surface, dt*substeps, 32)
                nonlocal max_error
                max_error = max(max_error, closed)
                return data
            base = [extract(ref) for ref in entry['nominal_branches']]
            for pair in entry['pairs']:
                signed = [extract(ref) for ref in sorted(pair['branches'], key=lambda ref: -ref['sign'])]
                meta = {'episode':entry['episode_id'],'group':entry['group_id'],'tick':entry['snapshot_tick'],
                        'history':entry['history_id'],'action':pair['direction_id'],'source':'measured','amplitude':pair['amplitude']}
                row, real = pair_row(meta, [x['measured'] for x in signed], [x['measured'] for x in base])
                _, approx = pair_row(meta, [x['analytic'] for x in signed], [x['analytic'] for x in base])
                error = rms(approx[WINDOW,1]-real[WINDOW,1])
                row['analytic_tangent_half_error_rms_m_s'] = error
                row['analytic_relative_tangent_half_error'] = error/row['tangent_half_rms_m_s']
                rows.append(row)
            guard(entry['episode_id'],log=True)
    exact = [x for x in rows if x['source']=='measured']
    if (len(rows),len(exact),snapshot_count) != (49,9,3):
        raise ValueError('timed training inventory differs from frozen 9 rows or 3 snapshots')
    enough = all(x['resolved_window_steps'] >= 9 for x in exact)
    compatible = all(x['analytic_relative_tangent_half_error'] <= MAX_RECOVERY_RELATIVE_RMS for x in exact)
    groups = {}
    for row in rows:
        key = '/'.join(row[k] for k in ('source','history','action'))
        g = groups.setdefault(key,{'pairs':0,'resolved':0,'strong':0,'available':0})
        for field, value in (('pairs',1),('resolved',row['resolved_window_steps']),
                             ('strong',row['strong_window_steps']),('available',row['window_steps'])):
            g[field] += value
    write_json(out/'label_audit.json', {'rows': rows,'groups':groups,'exact_snapshots':snapshot_count,
               'max_world_reconstruction_component_m_s':max_error,'timed_labels_resolved_for_pilot':enough,
               'analytic_recovery_within_10_percent_on_timed':compatible})
    return {'training_pairs':49,'measured_pairs':9,'analytic_pairs':40,'distinct_measured_snapshots':3,
            'timed_labels_resolved_for_pilot':enough,'analytic_recovery_within_10_percent_on_timed':compatible}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
