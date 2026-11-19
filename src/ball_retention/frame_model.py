"""Inertial constant-velocity baseline with exact moving-frame transport.

Only the 20 causal midpoint features enter the neural network. The additional
90 numbers carry a standardized affine map, not trainable input features.
No gravity/contact assumption is added to the baseline; those forces belong
to the learned residual. This is a local ablation, not a contact solver.
"""
import numpy as np

from .model_data import KNOWN_NAMES, _rotation, filter_advance
from .task import reference
from .world_model import ResidualMLP

FEATURE_DIM = len(KNOWN_NAMES)
PACKED_DIM = FEATURE_DIM+81+9


def cross_matrix(vector):
    vector = np.asarray(vector)
    result = np.zeros((*vector.shape[:-1], 3, 3))
    x, y, z = np.moveaxis(vector, -1, 0)
    result[..., 0, 1], result[..., 0, 2] = -z, y
    result[..., 1, 0], result[..., 1, 2] = z, -x
    result[..., 2, 0], result[..., 2, 1] = -y, x
    return result


def frame_affine(known, config, dt_s, anchor_local):
    """Raw SI A,b from causal midpoint features; output shape (...,9,9), (...,9).

    Recover both endpoints using the exact held-target actuator solution.
    Inverting half a control period does not read a past/future measurement.
    """
    known = np.asarray(known, dtype=float)
    anchor = np.asarray(anchor_local, dtype=float)
    if known.ndim < 1 or known.shape[-1] != FEATURE_DIM or not np.isfinite(known).all():
        raise ValueError('expected finite causal midpoint features')
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError('invalid material anchor or control period')
    shape = known.shape[:-1]
    rows = known.reshape(-1, FEATURE_DIM)
    tau = float(config['action']['tilt_filter_time_constant_s'])
    target = np.deg2rad(config['action']['tilt_limit_deg'])*rows[:, 2:4]
    endpoints = []
    for sign in (-1, 1):
        tilt, rate, _ = filter_advance(rows[:, 4:6], rows[:, 6:8], target, tau, sign*dt_s/2)
        rotation, roll_axis = _rotation(tilt, np.asarray(config['initial']['tool_rotation_world']))
        omega_world = rate[:, :1]*roll_axis+rate[:, 1:]*np.array([0., 1., 0.])
        omega_local = np.einsum('nji,nj->ni', rotation, omega_world)
        refs = [reference(max(0., t+sign*dt_s/2), config) for t in rows[:, 0]]
        position = np.array([r['anchor_position'] for r in refs])
        velocity = np.array([r['anchor_velocity'] for r in refs])
        endpoints.append((rotation, omega_local, position, velocity))
    r0, w0, p0, v0 = endpoints[0]
    r1, w1, p1, v1 = endpoints[1]
    into = r1.swapaxes(-1, -2)
    rotation = into@r0
    c = rotation@cross_matrix(w0)
    d = cross_matrix(w1)
    pp = rotation+dt_s*c
    pv = dt_s*rotation
    vp = c-d@pp
    vv = rotation-d@pv
    translation = np.einsum('nij,nj->ni', into, p0+dt_s*v0-p1)
    velocity = np.einsum('nij,nj->ni', into, v0-v1)
    a = np.zeros((len(rows), 9, 9))
    a[:, :3, :3], a[:, :3, 3:6] = pp, pv
    a[:, 3:6, :3], a[:, 3:6, 3:6] = vp, vv
    a[:, 6:9, 6:9] = rotation
    b = np.zeros((len(rows), 9))
    b[:, :3] = anchor+translation-pp@anchor
    b[:, 3:6] = velocity-np.einsum('nij,nj->ni', d, translation)-vp@anchor
    return a.reshape(*shape, 9, 9), b.reshape(*shape, 9)


def pack_features(known, config, dt_s, anchor_local, state_mean, state_scale,
                  known_mean, known_scale):
    """Standardize the network features and affine map using train-only scales."""
    a, b = frame_affine(known, config, dt_s, anchor_local)
    mean, scale = np.asarray(state_mean), np.asarray(state_scale)
    az = a*scale[..., None, :]/scale[..., :, None]
    bz = (a@mean+b-mean)/scale
    return np.concatenate(((known-known_mean)/known_scale,
                           az.reshape(*a.shape[:-2], 81), bz), axis=-1)


class FrameResidualMLP(ResidualMLP):
    """Same 29-32-32-9 MLP; replace only the deterministic transition."""

    def __init__(self, known_dim, *args, **kwargs):
        if known_dim != PACKED_DIM:
            raise ValueError('expected frame map plus 20 network features')
        super().__init__(FEATURE_DIM, *args, **kwargs)
        self.known_dim = PACKED_DIM

    def _forward(self, state, known):
        prediction, network_cache = super()._forward(state, known[..., :FEATURE_DIM])
        prediction -= state
        prediction[..., :3] -= self._velocity_gain*state[..., 3:6]+self._position_bias
        a = known[..., FEATURE_DIM:FEATURE_DIM+81].reshape(*state.shape[:-1], 9, 9)
        prediction += np.einsum('...ij,...j->...i', a, state)+known[..., -9:]
        return prediction, (network_cache, state, a)

    def _backward(self, cache, cotangent):
        network_cache, state, a = cache
        ds, dk, grads = super()._backward(network_cache, cotangent)
        ds -= cotangent
        ds[..., 3:6] -= self._velocity_gain*cotangent[..., :3]
        ds += np.einsum('...ij,...i->...j', a, cotangent)
        da = (cotangent[..., :, None]*state[..., None, :]).reshape(*state.shape[:-1], 81)
        return ds, np.concatenate((dk, da, cotangent), axis=-1), grads

    def state_dict(self):
        return {**super().state_dict(), 'model_type': np.array('inertial_frame_residual_v1')}


def measured_response_scale(batch):
    """Per-lead, per-unit robust train response scale, shared by xyz axes.

    Only resolved components contribute to the median. Noise floors prevent
    near-zero measurements from giving an unbounded relative loss. Call with
    the training batch only and freeze the result for validation.
    """
    spread = np.abs(batch.plus_target-batch.minus_target)/2
    scale = np.empty(spread.shape[1:])
    for lead in range(spread.shape[1]):
        for start in (0, 3, 6):
            region = slice(start, start+3)
            values = spread[:, lead, region][batch.response_mask[:, lead, region]]
            floors = batch.response_floor[:, lead, region][batch.noise_mask[:, lead]]
            if not floors.size:
                raise ValueError('training lead has no replay-noise estimate')
            scale[lead, region] = max(float(np.median(floors)),
                                      float(np.median(values)) if values.size else 0.)
    return scale
