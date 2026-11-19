"""Check the single-contact tangent impulse relation against frozen Newton traces."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import SCOPE, run_phase, write_json

import numpy as np
from scipy.spatial.transform import Rotation

from audit_s3_ball_velocity_components import closure, frame_series, half, load_cases, rms
from train_s3_local_velocity_increment import bound


WINDOW = (8, 24)
MIN_COSINE = .8
MAX_RELATIVE_RMS = .5
MODES = ('start_normal', 'midpoint_transport')


def vector_skill(actual, estimate):
    error = estimate-actual
    signal_rms, estimate_rms, error_rms = (rms(x) for x in (actual, estimate, error))
    energy = np.sum(actual*actual)
    dot = np.sum(actual*estimate)
    cosine = float(dot/np.sqrt(energy*np.sum(estimate*estimate))) if signal_rms and estimate_rms else None
    ratio = error_rms/signal_rms if signal_rms else None
    return {'signal_rms_m_s': signal_rms, 'estimate_rms_m_s': estimate_rms, 'error_rms_m_s': error_rms,
            'cosine': cosine, 'relative_rms': ratio, 'signed_projection_gain': float(dot/energy) if energy else None,
            'fraction_explained_vs_zero': 1.-ratio**2 if ratio is not None else None,
            'passes': cosine is not None and cosine >= MIN_COSINE and ratio <= MAX_RELATIVE_RMS}


def grouped(vectors, groups):
    out = {}
    for name, leads in groups.items():
        if len(leads) == 0:
            out[name] = None
            continue
        a = {k: v[np.asarray(leads)-1] for k, v in vectors.items()}
        out[name] = {'tangent_from_spin': vector_skill(a['tangent'], a['tangent_from_spin']),
                     'spin_from_tangent': vector_skill(a['spin_equivalent'], a['spin_from_tangent']),
                     'spin_normal_equivalent_rms_m_s': rms(a['spin_normal_equivalent'])}
    return out


def relation(normal, increment, spin_equiv):
    normal_velocity = np.sum(normal*increment, axis=-1, keepdims=True)*normal
    tangent = increment-normal_velocity
    share_spin = np.sum(normal*spin_equiv, axis=-1, keepdims=True)*normal
    tangent_estimate = np.cross(normal, spin_equiv)
    spin_estimate = -np.cross(normal, tangent)
    check = max(closure(tangent, np.cross(normal, spin_estimate)),
                closure(spin_equiv-share_spin, -np.cross(normal, tangent_estimate)))
    return {'tangent': tangent, 'tangent_from_spin': tangent_estimate, 'spin_equivalent': spin_equiv,
            'spin_from_tangent': spin_estimate, 'spin_normal_equivalent': share_spin}, check


def run(plan, out, guard):
    if (plan['updates'] != 0 or plan['window'] != list(WINDOW) or plan['normal_modes'] != list(MODES)
            or plan['min_cosine'] != MIN_COSINE or plan['max_relative_rms'] != MAX_RELATIVE_RMS):
        raise ValueError('unexpected coupling audit scope')

    def read(path):
        return json.loads(bound(plan, path))

    inputs = {}
    for key, artifact in (('surface', 'surface_components.json'), ('velocity', 'components.json'),
                          ('contacts', 'contacts.json')):
        folder = Path(plan[f'{key}_run'])
        validated = read(folder/'validation.json')
        file = folder/artifact
        if (validated['status'] != 'completed' or (validated['updates'], validated['simulation_steps']) != (0, 0)
                or any(validated[name] != value for name, value in SCOPE.items())
                or validated['artifacts_sha256'][artifact] != plan['files_sha256'][file.as_posix()]):
            raise ValueError(f'prior accepted artifact changed: {key}')
        inputs[key] = read(file)
    capture, cases = load_cases(plan, read)
    output, max_closure, candidate_steps = {}, 0., 0
    pair_passes = {mode: {group: 0 for group in ('strong', 'stable', 'flagged')} for mode in MODES}
    for case, (cfg, initial, traces) in cases.items():
        geom = inputs['surface']['cases'][case]
        velocity = inputs['velocity']['cases'][case]
        contacts = inputs['contacts'][case]
        dt = capture['control_substeps']*cfg['physics']['dt_s']
        if dt != velocity['control_dt_s'] or contacts['physics_dt_s'] != cfg['physics']['dt_s']:
            raise ValueError(f'archive clock changed: {case}')
        mass, radius = cfg['ball']['mass_kg'], cfg['ball']['radius_m']
        if (mass, radius) != (.05, .02):
            raise ValueError(f'ball parameters changed: {case}')
        inertia = .4*mass*radius**2
        equiv_length = inertia/(mass*radius)
        gravity_step = dt*np.array([0., 0., -cfg['physics']['gravity_m_s2']])
        shape_pair = sorted([contacts['shape_labels'].index(name) for name in ('ball', 'shovel')])
        fields, endpoint_frames, evidence = {}, {}, {}
        window = np.arange(WINDOW[0], WINDOW[1]+1)
        for label, trace in traces.items():
            candidate_sets = contacts['branches'][label]['shape_pair_sets']
            if candidate_sets != [{'pairs': [shape_pair], 'substeps': 32*capture['control_substeps']}]:
                raise ValueError(f'archived candidate shapes changed: {case}/{label}')
            candidate_steps += candidate_sets[0]['substeps']
            rotation, _, _, world_velocity = frame_series(initial, trace, dt)
            net = np.diff(world_velocity, axis=0)-gravity_step
            states = [initial, *trace['states']]
            spin_equiv = equiv_length*np.diff([s.ball.angular_velocity for s in states], axis=0)
            rows = geom['geometry'][label]
            start_normal = rotation[:-1].apply([r['normal_tool'] for r in rows])
            max_closure = max(max_closure, closure(velocity['branches'][label]['non_gravity'],
                                                   rotation[1:].apply(net, inverse=True)),
                              closure([r['normal_in_step_end_tool'] for r in rows],
                                      rotation[1:].apply(start_normal, inverse=True)))
            half_turn = Rotation.from_rotvec(.5*(rotation[1:]*rotation[:-1].inv()).as_rotvec())
            normals = {MODES[0]: start_normal, MODES[1]: half_turn.apply(start_normal)}
            flagged = np.array([r['closest_feature'] != 'face' or r['step_start_normal_change_deg'] > plan['normal_change_deg']
                or not r['upward_bowl_face'] or not r['outward_side']
                or abs(r['sphere_distance_gap_m']) > plan['support_gap_m'] for r in rows])
            groups = {'window': window, 'stable': window[~flagged[window-1]], 'flagged': window[flagged[window-1]]}
            fields[label], evidence[label] = {}, {}
            for mode, normal in normals.items():
                terms, check = relation(normal, net, spin_equiv)
                max_closure = max(max_closure, check)
                fields[label][mode] = terms
                evidence[label][mode] = {'group_metrics': grouped(terms, groups),
                                         'vectors_world_m_s': {k: v.tolist() for k, v in terms.items()}}
            endpoint_frames[label] = rotation[1:]
            guard(case)
        pairs = {}
        for direction in capture['directions']:
            a, b = (f'{direction}_{side}' for side in ('plus', 'minus'))
            surface_pair = geom['pairs'][direction]
            groups = {key: surface_pair['leads_by_group'][key] for key in ('window', 'strong', 'stable', 'flagged')}
            measured = half([endpoint_frames[key].apply(fields[key][MODES[0]]['tangent'], inverse=True) for key in (a, b)])
            previous = surface_pair['components']['newton']['bases']['per_branch']['vectors_m_s']['tangent']
            max_closure = max(max_closure, closure(previous, measured))
            result = {}
            for mode in MODES:
                terms = {key: half((v, fields[b][mode][key])) for key, v in fields[a][mode].items()}
                metrics = grouped(terms, groups)
                for key in pair_passes[mode]:
                    pair_passes[mode][key] += int(metrics[key] is not None and all(
                        metrics[key][direction]['passes'] for direction in ('tangent_from_spin', 'spin_from_tangent')))
                result[mode] = {'group_metrics': metrics, 'vectors_world_m_s': {k: v.tolist() for k, v in terms.items()}}
            pairs[direction] = {'groups': groups, 'modes': result}
            guard(case)
        output[case] = {'ball_mass_kg': mass, 'ball_radius_m': radius, 'ball_inertia_kg_m2': inertia,
                        'spin_equivalent_length_m': equiv_length, 'branches': evidence, 'pairs': pairs}
        guard(case, log=True)
    write_json(out/'coupling.json', {'cases': output, 'paired_bidirectional_pass_counts': pair_passes,
               'convention': 'All vectors are in world coordinates. Spin increments are multiplied by I/(m*r) '
               'to use m/s. Estimate tangential velocity with n x spin_equivalent, and spin_equivalent with '
               '-n x measured tangential velocity. The midpoint sensitivity transports the same start facet '
               'by half of the observed tool rotation. No gains are fitted.'})
    return {'snapshots': 2, 'branches': 16, 'pairs': 6, 'candidate_substeps_verified': candidate_steps,
            'paired_bidirectional_pass_counts': pair_passes, 'max_vector_closure_m_s': max_closure}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'tangential_spin', run)
