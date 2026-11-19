"""Teacher-forced world-frame increment objectives on full-state archives."""
import numpy as np

from .world_increment_model import FEATURES, WorldIncrementMLP
from .world_model import ResidualMLP


def world_increment_loss(model, states, known, target, weights, scale_m_s, *, paired=False):
    """Return a weighted MSE and its parameter gradient at measured pre-states.

    The paired form expects a positive and negative branch in its first
    dimension and applies weights to the half-difference. Geometry and state
    values are fixed here. Deployment still uses the model's own rollout.
    """
    if not isinstance(model, WorldIncrementMLP):
        raise ValueError('expected a world increment model')
    state = np.asarray(states,dtype=float)
    known, target, weights = (np.asarray(v,dtype=float) for v in (known,target,weights))
    if (state.ndim < 2 or state.shape[-1] != 9 or known.shape != (*state.shape[:-1],FEATURES)
            or target.shape != (*state.shape[:-1],3)
            or (paired and (state.ndim < 3 or len(state) != 2))
            or not np.isfinite(scale_m_s) or scale_m_s <= 0.
            or not all(np.isfinite(x).all() for x in (state,known,target,weights))):
        raise ValueError('invalid world increment loss inputs')
    expected_weights = state.shape[1:-1] if paired else state.shape[:-1]
    if weights.shape != expected_weights or np.any(weights < 0.) or not np.isclose(weights.sum(),1.):
        raise ValueError('expected normalized increment weights')
    z = model.normalize(state)
    forward, cache = ResidualMLP._forward(model,z,known)
    increment = model.inc_center+(forward[...,3:6]-z[...,3:6])*model.state_scale[3:6]
    error = increment-target
    if paired:
        error = .5*(error[0]-error[1])
    weighted = weights[...,None]/(3.*scale_m_s**2)
    loss = float(np.sum(weighted*error**2))
    dx = weighted*2.*error
    if paired:
        dx = .5*np.stack((dx,-dx))
    cotangent = np.zeros_like(z)
    cotangent[...,3:6] = dx*model.state_scale[3:6]
    _, _, grad = ResidualMLP._backward(model,cache,cotangent)
    return loss, grad, error
