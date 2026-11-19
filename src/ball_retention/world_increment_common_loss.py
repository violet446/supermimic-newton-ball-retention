"""Independent one-step supervision for the shared response to signed actions."""
import numpy as np

from .world_increment_model import FEATURES, WorldIncrementMLP
from .world_model import ResidualMLP


CONTRAST = np.asarray([-.5,-.5,.5,.5])
CONTRAST.setflags(write=False)


def common_increment_loss(model, states, known, targets, weights, scale_m_s):
    """Fit .5(plus+minus)-.5(nominal_1+nominal_2) at measured pre-states."""
    if not isinstance(model, WorldIncrementMLP):
        raise ValueError('expected a world increment model')
    state,features,targets,weights = (np.asarray(x,dtype=float) for x in (states,known,targets,weights))
    if (state.ndim!=4 or state.shape[0]!=4 or state.shape[-1]!=9
            or features.shape!=(*state.shape[:-1],FEATURES) or targets.shape!=(*state.shape[:-1],3)
            or weights.shape!=state.shape[1:-1] or not np.isfinite(scale_m_s) or scale_m_s<=0.
            or not all(np.isfinite(x).all() for x in (state,features,targets,weights))
            or np.any(weights<0.) or not np.isclose(weights.sum(),1.,rtol=0.,atol=2e-14)):
        raise ValueError('invalid common increment loss arrays or weights')
    normalized = model.normalize(state)
    forward,cache = ResidualMLP._forward(model,normalized,features)
    # The constant world-frame increment center cancels under this contrast.
    increments = (forward[...,3:6]-normalized[...,3:6])*model.state_scale[3:6]
    error = np.tensordot(CONTRAST,increments-targets,axes=1)
    weight = weights[...,None]/(3.*scale_m_s**2)
    loss = float(np.sum(weight*error**2))
    cotangent = np.zeros_like(normalized)
    cotangent[...,3:6] = CONTRAST[:,None,None,None]*(2.*weight*error*model.state_scale[3:6])
    _,_,gradient = ResidualMLP._backward(model,cache,cotangent)
    if not np.isfinite(loss) or not all(np.isfinite(x).all() for x in gradient.values()):
        raise FloatingPointError('nonfinite common increment loss or gradient')
    return loss,gradient,error
