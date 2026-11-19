"""Project archived net ball velocity increments and one-step errors onto the shovel surface."""
import argparse
from io import BytesIO
import json
from pathlib import Path

from s3_fitability_common import ROOT, SCOPE, forecast, run_phase, write_json

import numpy as np
from trimesh.triangles import points_to_barycentric

from audit_s3_ball_velocity_components import branch_terms, closure, frame_series, half, load_cases, rms
from ball_retention.action_grad import _inputs
from ball_retention.geometry import load_tool
from ball_retention.model_data import KNOWN_NAMES, _ball_state
from ball_retention.position_integral_model import TrapezoidPositionResidualMLP, load_position_model
from ball_retention.surface_diagnostic import SurfaceDiagnostic
from train_s3_local_velocity_increment import bound


WINDOW = (8, 24)
NORMAL_CHANGE_DEG = 1.
BARYCENTRIC_TOL = 1e-8
SUPPORT_GAP_M = .002
LABELS = ('dense_49', 'increment_2_5_mm_s')
NOMINALS = ('nominal_1', 'nominal_2')


def angle_deg(a, b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1)))


def surface_series(surface, truth, rotation):
    """Measure at step start and express normals in the step-end tool coordinates."""
    records = [surface.measure(p) for p in truth[:-1, :3]]
    faces = [row['triangle'] for row in records]
    closest = np.asarray([row['closest_point_tool_m'] for row in records])
    bary = points_to_barycentric(surface.triangles[faces], closest)
    normal_start = np.asarray([row['normal_tool'] for row in records])
    closure(np.ones((len(records), 1)), np.linalg.norm(normal_start, axis=-1, keepdims=True))
    if not np.isfinite(bary).all() or np.min(bary) < -BARYCENTRIC_TOL:
        raise ValueError('closest point is outside its selected triangle')
    kind = np.asarray(('face', 'edge', 'vertex'))[np.count_nonzero(bary <= BARYCENTRIC_TOL, axis=1)]
    jump = np.r_[0., angle_deg(normal_start[1:], normal_start[:-1])]
    endpoint = rotation[1:].apply(rotation[:-1].apply(normal_start), inverse=True)
    for i, row in enumerate(records):
        row.update(closest_feature=kind[i], closest_barycentric=bary[i].tolist(),
                   step_start_normal_change_deg=float(jump[i]), normal_in_step_end_tool=endpoint[i].tolist())
    return records, normal_start, endpoint, kind, jump


def split_pair(values, normals, expected):
    projections = np.sum(values*normals, axis=-1, keepdims=True)*normals
    tangents = values-projections
    total = half(values)
    normal, tangent = half(projections), half(tangents)
    error = max(closure(expected, total), closure(total, normal+tangent),
                closure(np.zeros_like(tangents[..., :1]), np.sum(tangents*normals, axis=-1, keepdims=True)))
    return {'total': total, 'normal': normal, 'tangent': tangent}, error


def metrics(split, mask):
    if not mask.any():
        return None
    a, n, t = (split[key][mask] for key in ('total', 'normal', 'tangent'))
    energy = np.sum(a*a)
    return {'rms_m_s': {key: rms(vec[mask]) for key, vec in split.items()},
            'normal_projection_fraction': float(np.sum(n*a)/energy) if energy else None,
            'tangent_projection_fraction': float(np.sum(t*a)/energy) if energy else None}


def paired_increments(models, cfg, initial, rows, frames, truth, gravity_step):
    starts = np.stack([a[0] for a in truth])
    targets = np.stack([a[1:] for a in truth])
    predictions = {}
    for label, (model, mean, scale) in models.items():
        features = np.stack([_inputs(model, start, initial.time_s, initial.tilt_rad, initial.tilt_rate_rad_s,
                                     row['actions'], cfg, mean, scale)[2]
                             for start, row in zip(starts, rows, strict=True)])
        predicted = forecast(model, starts, features, targets, local=True)
        predictions[label] = np.stack([branch_terms(f, a, gravity_step, endpoint=p, independent=True)[0]['non_gravity']
                                      for f, a, p in zip(frames, truth, predicted, strict=True)])
    return predictions


def audit_pair(direction, geometry, measured, predicted, prior_row):
    labels = [*NOMINALS, f'{direction}_plus', f'{direction}_minus']
    selected = [geometry[k] for k in labels]
    notes, start_normals, end_normals, kinds, jumps = zip(*selected, strict=True)
    closure(end_normals[0], end_normals[1])
    compare = np.stack([angle_deg(start_normals[a], start_normals[b]) for a, b in ((0, 2), (0, 3), (2, 3))])
    surfaces_ok = np.array([[r['upward_bowl_face'] and r['outward_side']
                             and abs(r['sphere_distance_gap_m']) <= SUPPORT_GAP_M for r in rs] for rs in notes])
    flags = {
        'edge': (np.asarray(kinds) == 'edge').any(axis=0),
        'vertex': (np.asarray(kinds) == 'vertex').any(axis=0),
        'normal_jump': (np.asarray(jumps) > NORMAL_CHANGE_DEG).any(axis=0),
        'normal_disagreement': (compare > NORMAL_CHANGE_DEG).any(axis=0),
        'outside_support_screen': ~surfaces_ok.all(axis=0)}
    unusual = np.logical_or.reduce(list(flags.values()))
    strong = np.isin(np.arange(1, 33), prior_row['strong_window_leads'])
    window = (np.arange(1, 33) >= WINDOW[0]) & (np.arange(1, 33) <= WINDOW[1])
    if not strong.any() or (strong & ~window).any():
        raise ValueError('unexpected prior strong-signal window')
    groups = {'window': window, 'strong': strong, 'stable': strong & ~unusual,
              'flagged': strong & unusual, **{key: strong & val for key, val in flags.items()}}
    expected = np.asarray(prior_row['increment_half_difference_m_s']['non_gravity'])
    target = {'newton': (measured, expected)}
    for key, values in predicted.items():
        ref = np.asarray(prior_row['models'][key]['local']['predicted_increment_half_difference_m_s']['non_gravity'])
        closure(ref, half(values))
        target[key] = (values-measured, ref-expected)
    results, max_closure = {}, 0.
    for label, (actual, reference) in target.items():
        bases, pairs = {}, {}
        for mode, normals in (('nominal', end_normals[0]), ('per_branch', np.stack(end_normals[2:]))):
            vectors, error = split_pair(actual, normals, reference)
            max_closure = max(max_closure, error)
            pairs[mode] = vectors
            bases[mode] = {'vectors_m_s': {k: v.tolist() for k, v in vectors.items()},
                           'metrics': {k: metrics(vectors, mask) for k, mask in groups.items()}}
        change = pairs['per_branch']['normal']-pairs['nominal']['normal']
        results[label] = {'bases': bases, 'normal_vector_change_rms_m_s': rms(change[strong])}
    return {'leads_by_group': {key: (np.flatnonzero(mask)+1).tolist() for key, mask in groups.items()},
            'material_normal_angle_deg': dict(zip(('nominal_plus', 'nominal_minus', 'plus_minus'),
                                                  compare.tolist(), strict=True)),
            'components': results}, max_closure


def run(plan, out, guard):
    if (plan['updates'] != 0 or plan['window'] != list(WINDOW) or tuple(plan['models']) != LABELS
            or plan['normal_change_deg'] != NORMAL_CHANGE_DEG or plan['barycentric_tol'] != BARYCENTRIC_TOL
            or plan['support_gap_m'] != SUPPORT_GAP_M):
        raise ValueError('unexpected geometry audit scope')

    def read(path):
        return json.loads(bound(plan, path))

    previous = Path(plan['velocity_run'])
    validation = read(previous/'validation.json')
    previous_plan = read(plan['velocity_plan'])
    if (validation['status'] != 'completed' or validation['updates'] != 0
            or any(validation[k] != value for k, value in SCOPE.items())
            or validation['plan_sha256'] != plan['files_sha256'][plan['velocity_plan']]
            or validation['artifacts_sha256']['components.json'] != plan['files_sha256'][(previous/'components.json').as_posix()]
            or previous_plan['models'] != plan['models']):
        raise ValueError('unexpected frozen velocity evidence')
    prior = read(previous/'components.json')
    capture, cases = load_cases(plan, read)
    models = {}
    for label, name in plan['models'].items():
        folder = Path(name)
        checkpoint = folder/'velocity_response_leads_900.npz'
        accepted = read(folder/'validation.json')
        if accepted['checkpoint_sha256'] != plan['files_sha256'][checkpoint.as_posix()]:
            raise ValueError('fixed checkpoint identity changed')
        with np.load(BytesIO(bound(plan, checkpoint)), allow_pickle=False) as saved:
            model = load_position_model(saved)
            if (type(model) is not TrapezoidPositionResidualMLP or int(saved['updates']) != 900
                    or tuple(saved['known_names']) != KNOWN_NAMES):
                raise ValueError('unexpected fixed model')
            models[label] = model, saved['known_mean'], saved['known_scale']

    mesh = load_tool(ROOT/plan['mesh'])
    output, max_closure, count = {}, 0., 0
    for case, (cfg, initial, traces) in cases.items():
        previous = prior['cases'][case]
        dt = cfg['physics']['dt_s']*capture['control_substeps']
        if dt != previous['control_dt_s'] or any(abs(m.dt_s-dt)>1e-14 for m, _, _ in models.values()):
            raise ValueError('control clock changed')
        gravity_step = dt*np.array([0., 0., -cfg['physics']['gravity_m_s2']])
        surface = SurfaceDiagnostic(mesh, cfg['ball']['radius_m'], cfg['initial']['tool_rotation_world'])
        truths, frames, geometry, measured = {}, {}, {}, {}
        for key, trace in traces.items():
            truth = np.vstack((_ball_state(initial), trace['raw']))
            frame = frame_series(initial, trace, dt)
            values = branch_terms(frame, truth, gravity_step)[0]['non_gravity']
            closure(previous['branches'][key]['non_gravity'], values)
            geometry[key] = surface_series(surface, truth, frame[0])
            truths[key], frames[key], measured[key] = truth, frame, values
            guard(case)
        pairs = {}
        for direction in capture['directions']:
            labels = [f'{direction}_{side}' for side in ('plus', 'minus')]
            predicted = paired_increments(models, cfg, initial, [traces[k] for k in labels],
                [frames[k] for k in labels], [truths[k] for k in labels], gravity_step)
            result, error = audit_pair(direction, geometry, np.stack([measured[k] for k in labels]), predicted,
                                       previous['pairs'][direction])
            pairs[direction] = result
            max_closure = max(max_closure, error)
            count += len(result['leads_by_group']['strong'])
        output[case] = {'geometry': {key: value[0] for key, value in geometry.items()}, 'pairs': pairs}
        guard(case, log=True)
    write_json(out/'surface_components.json', {'cases': output, 'window': WINDOW,
               'convention': 'Closest face normal at step start transformed to step-end tool coordinates. '
               'Project absolute branch vectors first, then take half-differences. '
               'Model terms are one-step net velocity increment errors. No mesh measurements enter the model.'})
    return {'snapshots': 2, 'branches': 16, 'pair_trajectories': 6, 'checkpoints': 2,
            'strong_pair_steps': count, 'max_vector_closure_m_s': max_closure}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.plan_sha256, 'surface_velocity_components', run)
