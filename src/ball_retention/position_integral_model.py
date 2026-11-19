"""Residual dynamics with tool-frame position integrated from relative velocity."""
import numpy as np

from .world_model import ResidualMLP


class TrapezoidPositionResidualMLP(ResidualMLP):
    """Preserve the residual velocity and spin updates and couple position to them."""

    MODEL_TYPE = 'trapezoid_position_residual_v1'

    def _forward(self, state, known):
        prediction, cache = super()._forward(state, known)
        velocity_midpoint = .5*(state[..., 3:6]+prediction[..., 3:6])
        position_increment = self._velocity_gain*velocity_midpoint+self._position_bias
        prediction[..., :3] = state[..., :3]+position_increment
        return prediction, cache

    def _backward(self, cache, cotangent):
        position = cotangent[..., :3]
        velocity = .5*self._velocity_gain*position
        residual_cotangent = cotangent.copy()
        residual_cotangent[..., :3] = 0.
        residual_cotangent[..., 3:6] += velocity
        state_grad, known_grad, grads = super()._backward(cache, residual_cotangent)
        state_grad[..., :3] += position
        state_grad[..., 3:6] += velocity
        return state_grad, known_grad, grads

    def state_dict(self):
        state = super().state_dict()
        # The distinct gain key makes legacy loaders reject this transition.
        state['trapezoid_residual_gain'] = state.pop('residual_gain')
        state['model_type'] = np.array(self.MODEL_TYPE)
        return state

    @classmethod
    def from_state_dict(cls, state):
        if 'model_type' not in state or np.asarray(state['model_type']).item() != cls.MODEL_TYPE:
            raise ValueError('expected a trapezoid position checkpoint')
        return super().from_state_dict({**state, 'residual_gain': state['trapezoid_residual_gain']})


def load_position_model(state):
    """Dispatch the two position updates in this experiment by checkpoint format."""
    if 'model_type' not in state:
        return ResidualMLP.from_state_dict(state)
    return TrapezoidPositionResidualMLP.from_state_dict(state)
