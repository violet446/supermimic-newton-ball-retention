"""Separate mesh support direction vectors from the weighting of their faces."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np
from scipy.special import expit

from audit_s3_causal_support_direction import (
    LABELS, REPLAY_PLAN, SETTINGS, MeshSupport, bound, frozen_case, score, start_geometry, support_query,
)
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _rotation, filter_advance
from ball_retention.surface_diagnostic import SurfaceDiagnostic


PHASE = 'support_direction_bias'
PRIOR = Path('runs/s3_causal_support_direction_20261117_verified')
SETUP = {'aligned_angle_deg': 20., 'position_fd_m': 1e-6, 'velocity_fd_m_s': 1e-4,
         'state_fd_rtol': 2e-5, 'position_fd_atol_per_s': 2e-5, 'velocity_fd_atol': 4e-7}
METHODS = ('radial_all', 'face_all', 'radial_aligned', 'face_aligned')
WINDOW = slice(7, 24)


class Direction:
    def __init__(self, mesh, normals, method, reference, *, record_groups=False):
        self.mesh = mesh
        self.face = normals
        self.use_face, self.use_filter = method.startswith('face'), method.endswith('aligned')
        self.aligned = self.face@reference >= np.cos(np.deg2rad(SETUP['aligned_angle_deg']))
        self.record_groups = record_groups
        self.stats = None
        self.jacobian = None

    def direction(self, center, jacobian=False):
        mesh = self.mesh
        offset = center[:, None, :]-mesh.points
        distance = np.linalg.norm(offset, axis=-1)
        if np.any(distance < 1e-10):
            raise ValueError('sphere center coincides with a quadrature point')
        radial = offset/distance[..., None]
        vectors = np.broadcast_to(self.face, radial.shape) if self.use_face else radial
        z = (mesh.reach-distance)/mesh.width
        w = mesh.areas*np.logaddexp(0., z)
        selected = w*self.aligned if self.use_filter else w
        total = np.einsum('qp,qpi->qi', selected, vectors)
        length = np.linalg.norm(total, axis=-1)
        if not np.isfinite(total).all() or np.any(length < 1e-20):
            raise ValueError('insufficient contact direction support')
        if self.record_groups:
            same = np.einsum('qp,qpi->qi', w*self.aligned, vectors)
            self.stats = {'aligned_weight_fraction': w[:, self.aligned].sum(axis=1)/w.sum(axis=1),
                          'aligned_vector': same/length[:, None], 'other_vector': (total-same)/length[:, None]}
        unit = total/length[:, None]
        if not jacobian:
            return unit
        dw = -mesh.areas*expit(z)/mesh.width
        if self.use_filter:
            dw *= self.aligned
            w *= self.aligned
        grad = np.einsum('qp,qpi,qpj->qij', dw, vectors, radial)
        if not self.use_face:
            c = w/distance
            grad += c.sum(axis=1)[:, None, None]*np.eye(3)-np.einsum('qp,qpi,qpj->qij', c, radial, radial)
        self.jacobian = (grad-unit[..., None]*np.einsum('qi,qij->qj', unit, grad)[:, None, :])/length[:, None, None]
        return unit, self.jacobian


def project(vectors, normal):
    return vectors-np.sum(vectors*normal, axis=-1, keepdims=True)*normal


def rotation_grid(state, action, cfg, r0):
    base = np.asarray(cfg['initial']['tool_rotation_world'])
    dt = cfg['physics']['dt_s']
    t = np.arange(1, 17)*dt
    actuator = cfg['action']
    tilt, _, _ = filter_advance(np.asarray(state.tilt_rad), np.asarray(state.tilt_rate_rad_s),
                                np.deg2rad(actuator['tilt_limit_deg'])*np.asarray(action),
                                actuator['tilt_filter_time_constant_s'], t[:, None])
    r, _ = _rotation(tilt, base)
    r_start, _ = _rotation(np.asarray(state.tilt_rad), base)
    return r@(r_start.T@r0), t


def rms_angle(estimate, actual):
    angle = np.arctan2(np.linalg.norm(np.cross(estimate, actual), axis=-1), np.einsum('ij,ij->i', estimate, actual))
    return float(np.sqrt(np.mean(np.rad2deg(angle[WINDOW])**2)))


def derivative_check(state, action, cfg, anchor, surface, support, normals, method, start):
    d = Direction(support, normals, method, start[1])
    _, ja = support_query(state, action, cfg, anchor, start, d, jacobian=True)
    rot, t = rotation_grid(state, action, cfg, start[0])
    dt = cfg['physics']['dt_s']
    scale = cfg['physics']['gravity_m_s2']*dt
    jp = scale*(rot@d.jacobian@rot.swapaxes(-1, -2))
    errors, boundaries = {}, 0

    def evaluate(s, a, geom):
        field = Direction(support, normals, method, geom[1])
        value = support_query(s, a, cfg, anchor, geom, field)['smooth_mesh']
        changes = field.use_filter and not np.array_equal(field.aligned, d.aligned)
        return value, changes

    for key, eps, jac, tolerance in (
        ('action', SETTINGS['fd_action_epsilon'], ja['smooth_mesh'], SETTINGS['fd_atol_m_s']),
        ('position', SETUP['position_fd_m'], jp, SETUP['position_fd_atol_per_s']),
        ('linear_velocity', SETUP['velocity_fd_m_s'], (t-dt)[:, None, None]*jp, SETUP['velocity_fd_atol']),
    ):
        errors[key] = 0.
        for axis, delta in enumerate(np.eye(jac.shape[-1])*eps):
            if key == 'action':
                a, b = (evaluate(state, np.asarray(action)+sign*delta, start) for sign in (1., -1.))
            else:
                attr = 'position' if key == 'position' else 'linear_velocity_origin'
                base = np.asarray(getattr(state.ball, attr))
                perturbed = [replace(state, ball=replace(state.ball, **{attr: (base+sign*delta).tolist()}))
                             for sign in (1., -1.)]
                a, b = (evaluate(s, action, start_geometry(s, surface) if key == 'position' else start)
                        for s in perturbed)
            if a[1] or b[1]:
                boundaries += 1
                continue
            finite = (a[0]-b[0])/(2.*eps)
            analytic = jac[..., axis]
            np.testing.assert_allclose(finite, analytic, atol=tolerance,
                                       rtol=SETUP['state_fd_rtol'] if key != 'action' else 0.)
            errors[key] = max(errors[key], float(np.max(abs(finite-analytic))))
    return errors, boundaries


def run(plan, out, guard):
    if plan['setup'] != SETUP or plan['methods'] != list(METHODS) or plan['updates'] != 0:
        raise ValueError('unexpected scope')
    replay = json.loads(bound(plan, REPLAY_PLAN))
    _, cfg, _, native, archive, _ = frozen_case(replay)
    previous = json.loads(bound(plan, PRIOR/'validation.json'))
    report_path = PRIOR/'support_direction.json'
    if previous['status'] != 'completed' or previous['artifacts_sha256'][report_path.name] != plan['files_sha256'][report_path.as_posix()]:
        raise ValueError('direction reference changed')
    mesh = load_tool(ROOT/'assets/shovel.STL')
    base = np.asarray(cfg['initial']['tool_rotation_world'])
    radius = cfg['ball']['radius_m']
    surface = SurfaceDiagnostic(mesh, radius, base)
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    support = MeshSupport(mesh, base, radius)
    keep = (np.asarray(mesh.triangles).mean(axis=1)[:, 0] >= .22) & (mesh.face_normals@base[2] >= .2) & (mesh.area_faces > 0.)
    normals = np.repeat(mesh.face_normals[keep], 3, axis=0)
    branches, rows = {}, {}
    for label in LABELS:
        starts = [native['state']]+[_parse_state(x) for x in archive[label]['states'][:23]]
        rows[label] = [(s, a, start_geometry(s, surface)) for s, a in zip(starts, archive[label]['actions'][:24], strict=True)]
        estimates = {key: [] for key in METHODS}
        stats = {key: [] for key in ('radial_all', 'face_all')}
        for state, request, start in rows[label]:
            for key in METHODS:
                field = Direction(support, normals, key, start[1], record_groups=key in stats)
                prediction = support_query(state, request, cfg, anchor, start, field)['smooth_mesh']
                estimates[key].append(prediction.sum(axis=0))
                if key in stats:
                    g = field.stats
                    parts = np.stack([project(g[name], start[1]) for name in ('aligned_vector','other_vector')])
                    stats[key].append({'weight_share': g['aligned_weight_fraction'].tolist(),
                                       'aligned_tangent': parts[0].tolist(), 'other_tangent': parts[1].tolist()})
            guard()
        branches[label] = {'prediction': {key: np.asarray(v) for key, v in estimates.items()}, 'geometry_stats': stats}
        guard(label, log=True)

    reference = json.loads(bound(plan, report_path))
    scores, all_tangent, stats_scores = {}, {}, {}
    for label in LABELS:
        branch = branches[label]
        prior = reference['per_branch'][label]
        plane = np.asarray(prior['start_plane_normal_world'])
        np.testing.assert_allclose([start[0]@start[1] for _, _, start in rows[label]], plane, atol=1e-12, rtol=0.)
        np.testing.assert_allclose(branch['prediction']['radial_all'], prior['normal_velocity_world_m_s']['smooth_mesh'],
                                   atol=1e-12, rtol=0.)
        actual = np.asarray(prior['normal_velocity_world_m_s']['xpbd'])
        branch['rms_angle_deg'] = {key: rms_angle(vec, actual) for key, vec in branch['prediction'].items()}
        all_tangent[label] = {key: project(vec, plane) for key, vec in {'xpbd': actual, **branch['prediction']}.items()}
        stats_scores[label] = {}
        for key, values in branch['geometry_stats'].items():
            a, b = (np.asarray([x[k] for x in values[WINDOW]]).reshape(-1, 3) for k in ('aligned_tangent','other_tangent'))
            y = a+b
            stats_scores[label][key] = {'mean_aligned_weight_fraction': float(np.mean([x['weight_share'] for x in values[WINDOW]])),
                                       'other_tangent_signed_fraction': float(np.sum(b*y)/np.sum(y*y))}
    plus, minus = (all_tangent[label] for label in LABELS)
    half = {k: (plus[k]-minus[k])*.5 for k in plus}
    for key in METHODS:
        scores[key] = score(half['xpbd'][WINDOW], half[key][WINDOW])
    qualified = [key for key in METHODS[1:] if scores[key]['relative_rms'] < scores[METHODS[0]]['relative_rms']
                 and all(b['rms_angle_deg'][key] < b['rms_angle_deg'][METHODS[0]] for b in branches.values())]
    derivatives = {}
    selected = min(qualified, key=lambda k: scores[k]['relative_rms'], default=None)
    if selected is not None:
        errors, boundaries = {}, 0
        guard(f'derivative_{selected}', log=True)
        for state, request, start in (row for label in LABELS for row in rows[label]):
            diff, switches = derivative_check(state, request, cfg, anchor, surface, support, normals, selected, start)
            errors = {axis: max(errors.get(axis, 0.), value) for axis, value in diff.items()}
            boundaries += switches
            guard()
        derivatives[selected] = {'max_fd_component_error': errors, 'mask_switch_fd_count': boundaries, 'queries': 48}
    write_json(out/'support_bias.json', {
        'angles_deg': {k:b['rms_angle_deg'] for k,b in branches.items()}, 'scores': scores,
        'groups': stats_scores, 'derivative_checks': derivatives,
        'half_differences_world_m_s': {k: v.tolist() for k, v in half.items()}})
    return {'branches': 2, 'horizon': 24, 'qualified_for_derivatives': qualified,
            'passed_response_gate': [k for k in METHODS if scores[k]['passed']]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
