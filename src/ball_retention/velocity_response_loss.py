"""Change the lead weights of the velocity response in a paired free rollout."""
import numpy as np

from .pair_loss import paired_loss
from .world_model import STATE_DIM, _nonnegative_weights


VELOCITY = slice(3, 6)


def response_components(lead_weights, velocity_lead_weights, component_weights, noise_mask, sample_weights):
    """Return lead-by-component weights before applying the pair weights and mask."""
    mask = np.asarray(noise_mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('expected a Boolean noise mask with shape (pairs, leads)')
    batch, horizon = mask.shape
    leads = _nonnegative_weights(lead_weights, horizon, 'lead weights')
    velocity_leads = _nonnegative_weights(velocity_lead_weights, horizon, 'velocity lead weights')
    components = _nonnegative_weights(component_weights, STATE_DIM, 'component weights')
    samples = _nonnegative_weights(sample_weights, batch, 'pair weights')
    weights = leads[:, None]*components
    replacement = velocity_leads[:, None]*components[VELOCITY]
    eligible = mask[..., None]*samples[:, None, None]
    original_mass = np.sum(eligible*weights[:, VELOCITY])
    new_mass = np.sum(eligible*replacement)
    if original_mass > 0. and new_mass == 0.:
        raise ValueError('velocity lead weights have no eligible replacement mass')
    weights[:, VELOCITY] = replacement*(original_mass/new_mass) if original_mass else 0.
    return weights


def paired_velocity_response_loss(model, initial, known, target, branch_mask, response_mask,
                                  noise_mask, response_floor, lead_weights, component_weights,
                                  response_scale, absolute_weight, response_weight, *, velocity_lead_weights=None,
                                  gradients=True, sample_weights=None, absolute_scale_si=None):
    """Apply independent lead weights only to the velocity half-difference term.

    The response retains the original effective velocity mass after pair weights
    and replay masks. Absent or equal velocity weights use the original loss.
    """
    branches = (model, initial, known, target, branch_mask, response_mask, noise_mask, response_floor)
    options = {'gradients': gradients, 'sample_weights': sample_weights, 'absolute_scale_si': absolute_scale_si}
    if velocity_lead_weights is None or np.array_equal(lead_weights, velocity_lead_weights):
        return paired_loss(*branches, lead_weights, component_weights, response_scale, absolute_weight,
                           response_weight, **options)

    adjusted = response_components(lead_weights, velocity_lead_weights, component_weights, noise_mask, sample_weights)
    # The shared loss accepts lead-by-component weights. Its two separate calls
    # keep the legacy weighting for absolute states and reuse the response VJP.
    state, state_grad = paired_loss(*branches, lead_weights, component_weights, response_scale, absolute_weight,
                                   0., **options)
    response, response_grad = paired_loss(*branches, np.ones(len(lead_weights)), adjusted, response_scale,
                                         0., response_weight, **options)
    metrics = {'absolute': state['absolute'], 'response': response['response'], 'loss': state['loss']+response['loss']}
    if not np.isfinite(metrics['loss']):
        raise FloatingPointError('non-finite velocity-weighted paired loss')
    if not gradients:
        return metrics, None
    return metrics, {name: gradient+response_grad[name] for name, gradient in state_grad.items()}
