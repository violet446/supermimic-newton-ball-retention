"""Trapezoid dynamics with the two direct request channels fixed at zero."""
from .model_data import KNOWN_NAMES
from .position_integral_model import TrapezoidPositionResidualMLP, load_position_model


REQUEST_COLUMNS = tuple(KNOWN_NAMES.index(name) for name in ('roll_request', 'pitch_request'))


class RequestMaskedTrapezoidMLP(TrapezoidPositionResidualMLP):
    """Mask standardized network inputs after analytic tool feature construction."""

    MODEL_TYPE = 'request_masked_trapezoid_v1'

    def __init__(self, known_dim, *args, **kwargs):
        if known_dim != len(KNOWN_NAMES):
            raise ValueError('request-masked model requires the frozen tool feature order')
        super().__init__(known_dim, *args, **kwargs)

    def _forward(self, state, known):
        masked = known.copy()
        masked[..., REQUEST_COLUMNS] = 0.
        return super()._forward(state, masked)

    def _backward(self, cache, cotangent):
        state_grad, known_grad, grads = super()._backward(cache, cotangent)
        known_grad[..., REQUEST_COLUMNS] = 0.
        return state_grad, known_grad, grads


def load_request_model(state):
    """Dispatch request masking from the checkpoint's transition type."""
    if 'model_type' in state and state['model_type'].item() == RequestMaskedTrapezoidMLP.MODEL_TYPE:
        return RequestMaskedTrapezoidMLP.from_state_dict(state)
    return load_position_model(state)
