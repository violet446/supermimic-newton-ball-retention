"""Evaluate a fixed differentiable mesh support direction in independent one-step queries."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np
from scipy.special import expit
from scipy.spatial.transform import Rotation

from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _rotation, filter_advance
from ball_retention.surface_diagnostic import SurfaceDiagnostic
from ball_retention.task import reference
from xpbd_contact_probe import LABELS, SUBSTEPS, accepted_artifact, bound, frozen_case


PHASE = 'causal_support_direction'
REPLAY = Path('diagnostics/resource_probe/runs/s3_xpbd_contact_probe_20261117_rebind')
REPLAY_PLAN = Path('diagnostics/resource_probe/s3_xpbd_contact_probe_rebind_plan.json')
SETTINGS = {'collision_gap_m': .001, 'smooth_width_m': .0002, 'fd_action_epsilon': .0001,
            'fd_atol_m_s': 2e-8, 'pair_leads': [8, 24], 'min_cosine': .8, 'max_relative_rms': .5}
METHODS = ('single_face', 'smooth_mesh')


class MeshSupport:
    """Area quadrature with smoothly weighted sphere-to-surface normals."""

    def __init__(self, mesh, base, radius):
        triangles = np.asarray(mesh.triangles)
        areas = np.asarray(mesh.area_faces)
        keep = (triangles.mean(axis=1)[:, 0] >= .22) & (mesh.face_normals@base[2] >= .2) & (areas > 0.)
        # Positive degree-two triangle rule, three points per face.
        barycentric = np.eye(3)*.5+1/6
        self.points = (barycentric@triangles[keep]).reshape(-1, 3)
        self.areas = np.repeat(areas[keep]/3, 3)
        self.reach = radius+SETTINGS['collision_gap_m']
        self.width = SETTINGS['smooth_width_m']
        if not len(self.points):
            raise ValueError('no upward bowl quadrature')

    def direction(self, center, jacobian=False):
        offset = center[:, None, :]-self.points
        distance = np.linalg.norm(offset, axis=-1)
        if np.any(distance < 1e-10):
            raise ValueError('sphere center coincides with a quadrature point')
        normals = offset/distance[..., None]
        z = (self.reach-distance)/self.width
        w = self.areas*np.logaddexp(0., z)
        total = np.einsum('qp,qpi->qi', w, normals)
        length = np.linalg.norm(total, axis=-1)
        if not np.isfinite(total).all() or np.any(length < 1e-20):
            raise ValueError('insufficient contact direction support')
        unit = total/length[:, None]
        if not jacobian:
            return unit
        radial = -self.areas*expit(z)/self.width
        coefficient = w/distance
        grad = coefficient.sum(axis=1)[:, None, None]*np.eye(3)
        grad += np.einsum('qp,qpi,qpj->qij', radial-coefficient, normals, normals)
        return unit, (grad-unit[..., None]*np.einsum('qi,qij->qj', unit, grad)[:, None, :])/length[:, None, None]


def start_geometry(state, surface):
    r0 = Rotation.from_quat(state.tool.quaternion_xyzw).as_matrix()
    local = r0.T@(np.asarray(state.ball.position)-state.tool.position)
    return r0, np.asarray(surface.measure(local)['normal_tool'])


def support_query(state, action, cfg, anchor, start, support, *, jacobian=False):
    """Use this step's public state and held request to predict a normal velocity increment."""
    action = np.asarray(action, dtype=float)
    if action.shape != (2,) or not np.isfinite(action).all() or np.max(abs(action)) >= 1.:
        raise ValueError('expected an interior normalized action')
    r0, n0 = start
    base = np.asarray(cfg['initial']['tool_rotation_world'])
    tau = cfg['action']['tilt_filter_time_constant_s']
    limit = np.deg2rad(cfg['action']['tilt_limit_deg'])
    dt = cfg['physics']['dt_s']
    times = np.arange(1, SUBSTEPS+1)*dt
    tilt, _, _ = filter_advance(np.array(state.tilt_rad), np.array(state.tilt_rate_rad_s),
                                limit*action, tau, times[:, None])
    r_analytic, roll_axis = _rotation(tilt, base)
    r_at_start, _ = _rotation(np.asarray(state.tilt_rad), base)
    rotation = r_analytic@(r_at_start.T@r0)
    ref_now = reference(state.time_s, cfg)['anchor_position']
    displacement = np.asarray([reference(state.time_s+t, cfg)['anchor_position']-ref_now for t in times])
    anchor_world = np.asarray(state.tool.position)+r0@anchor+displacement
    # The collision clock pairs endpoint tool pose with the start-of-substep ball center.
    ball = np.asarray(state.ball.position)+(times-dt)[:, None]*np.asarray(state.ball.linear_velocity_origin)
    delta = ball-anchor_world
    center = anchor+np.einsum('qji,qj->qi', rotation, delta)
    geometry = support.direction(center, jacobian)
    if jacobian:
        multi, dnormal = geometry
    else:
        multi = geometry
    scale = cfg['physics']['gravity_m_s2']*dt
    predicted = {name: scale*np.einsum('qij,qj->qi', rotation, direction)
                 for name, direction in zip(METHODS, (np.broadcast_to(n0, multi.shape), multi), strict=True)}
    if not jacobian:
        return predicted
    gain = limit*(1.-(1.+times/tau)*np.exp(-times/tau))
    axes = np.stack((roll_axis, np.broadcast_to([0., 1., 0.], roll_axis.shape)), axis=1)
    drot = np.cross(axes[:, :, None, :], rotation.swapaxes(1, 2)[:, None, :, :]).swapaxes(-1, -2)
    drot *= gain[:, None, None, None]
    dcenter = np.einsum('qaij,qi->qja', drot, delta)
    dlocal = np.einsum('qij,qja->qia', dnormal, dcenter)
    gradients = {
        METHODS[0]: scale*np.einsum('qaij,j->qia', drot, n0),
        METHODS[1]: scale*(np.einsum('qaij,qj->qia', drot, multi)+np.einsum('qij,qja->qia', rotation, dlocal))}
    return predicted, gradients


def score(actual, proposed):
    signal, estimate = (np.linalg.norm(x) for x in (actual, proposed))
    if not signal or not estimate:
        raise ValueError('unresolved normal impulse response')
    cosine = float(np.sum(actual*proposed)/(signal*estimate))
    relative = float(np.linalg.norm(proposed-actual)/signal)
    return {'cosine': cosine, 'relative_rms': relative,
            'signal_rms_m_s': float(signal/np.sqrt(len(actual))), 'estimate_rms_m_s': float(estimate/np.sqrt(len(actual))),
            'signed_projection_fraction': float(np.sum(actual*proposed)/signal**2),
            'passed': cosine >= SETTINGS['min_cosine'] and relative <= SETTINGS['max_relative_rms']}


def run(plan, out, guard):
    if plan['settings'] != SETTINGS or plan['updates'] != 0:
        raise ValueError('fixed support settings changed')
    # The archived scene seed is distinct from the enclosing offline runner seed.
    replay_plan = json.loads(bound(plan, REPLAY_PLAN))
    _, cfg, _, native, archive, _ = frozen_case(replay_plan)
    validation = json.loads(bound(plan, REPLAY/'validation.json'))
    if (validation['status'], validation['physics_steps'], validation['branches'], validation['training_updates']) != (
        'completed', 768, list(LABELS), 0):
        raise ValueError('unexpected target scope')
    mesh = load_tool(ROOT/'assets/shovel.STL')
    base = np.asarray(cfg['initial']['tool_rotation_world'])
    radius = cfg['ball']['radius_m']
    surface = SurfaceDiagnostic(mesh, radius, base)
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    support = MeshSupport(mesh, base, radius)
    queries, loss = {}, {m: 0. for m in METHODS}
    eps = SETTINGS['fd_action_epsilon']
    guard('predict', log=True)
    for label in LABELS:
        branch = archive[label]
        starts = [native['state']]+[_parse_state(x) for x in branch['states'][:23]]
        normals, estimates = [], {m: [] for m in METHODS}
        derivative_rows = []
        for state, action in zip(starts, branch['actions'][:24], strict=True):
            start = start_geometry(state, surface)
            normals.append(start[0]@start[1])
            pred, jac = support_query(state, action, cfg, anchor, start, support, jacobian=True)
            predicted_derivative = {}
            for axis, offset in enumerate(np.eye(2)*eps):
                a, b = (support_query(state, np.asarray(action)+sign*offset, cfg, anchor, start, support)
                        for sign in (1., -1.))
                for key in METHODS:
                    centered = (a[key]-b[key])/(2*eps)
                    np.testing.assert_allclose(centered, jac[key][..., axis], atol=SETTINGS['fd_atol_m_s'], rtol=0.)
                    loss[key] = max(loss[key], float(np.max(abs(centered-jac[key][..., axis]))))
            for key in METHODS:
                estimates[key].append(pred[key].sum(axis=0))
                predicted_derivative[key] = jac[key].sum(axis=0).tolist()
            derivative_rows.append(predicted_derivative)
            guard()
        queries[label] = {'normals': np.asarray(normals), 'estimates': {k: np.asarray(v) for k, v in estimates.items()},
                          'jacobians': derivative_rows}
        guard(label, log=True)

    window = slice(SETTINGS['pair_leads'][0]-1, SETTINGS['pair_leads'][1])
    results, branch_vectors = {}, []
    for label in LABELS:
        target = accepted_artifact(plan, REPLAY, f'{label}.json')
        if target['public_states'] != archive[label]['states'][:24] or len(target['substeps']) != 24*SUBSTEPS:
            raise ValueError('XPBD targets disagree with the archived branch')
        actual = np.asarray([row['ball_normal_contact_impulse_n_s'] for row in target['substeps']])
        actual = actual.reshape(24, SUBSTEPS, 3).sum(axis=1)/cfg['ball']['mass_kg']
        row = queries[label]
        normal = row['normals']
        tangent = lambda v: v-normal*np.sum(normal*v, axis=-1, keepdims=True)
        estimates = row['estimates']
        branch_vectors.append({k: tangent(v) for k, v in {'xpbd': actual, **estimates}.items()})
        direction_error = {k: np.rad2deg(np.arctan2(np.linalg.norm(np.cross(v, actual), axis=-1),
                                                 np.sum(v*actual, axis=-1))) for k, v in estimates.items()}
        results[label] = {
            'start_plane_normal_world': normal.tolist(), 'normal_velocity_world_m_s': {k:v.tolist() for k,v in {'xpbd':actual, **estimates}.items()},
            'direction_error_deg_rms_in_window': {k:float(np.sqrt(np.mean(v[window]**2))) for k,v in direction_error.items()},
            'one_step_request_jacobians_m_s': row['jacobians']}
    plus, minus = branch_vectors
    pair = {k: .5*(plus[k]-minus[k]) for k in plus}
    scores = {k: score(pair['xpbd'][window], pair[k][window]) for k in METHODS}
    momentum = accepted_artifact(plan, REPLAY, 'momentum.json')
    expected = momentum['pair']['control_period']['half_differences_m_s']['contact_normal_contribution']
    np.testing.assert_allclose(pair['xpbd'][window], expected, atol=1e-12, rtol=0.)
    write_json(out/'support_direction.json', {'per_branch': results, 'pair_tangent_half_difference_world_m_s': {
        k: v.tolist() for k, v in pair.items()}, 'pair_scores': scores, 'max_fd_component_error_m_s_per_action': loss,
        'angular_clock': 'Measured step-start pose, causal relative tool motion, frozen start velocity for ball center.',
        'normal_velocity_scale': 'World gravity magnitude times physics substep. No fit.'})
    return {'branches': 2, 'horizon': 24, 'quadrature_points': len(support.points),
            'pair_scores': scores, 'max_fd_component_error_m_s_per_action': loss}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
