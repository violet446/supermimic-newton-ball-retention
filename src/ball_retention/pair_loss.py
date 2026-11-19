"""Joint supervision for absolute Newton states and matched action responses."""
import numpy as np


def paired_loss(model, initial, known, target, branch_mask, response_mask,
                noise_mask, response_floor, lead_weights, component_weights,
                response_scale, absolute_weight, response_weight, *, gradients=True, sample_weights=None,
                absolute_scale_si=None):
    """Fit both free rollouts from one initial state, then fit their half-difference.

    Known features and initial states are standardized. Targets, the half-spread
    noise floor and response scales are SI. The first branch axis is plus/minus.
    """
    branch_count, batch, horizon, known_dim = known.shape
    if branch_count != 2 or initial.shape != (batch, 9):
        raise ValueError('expected two branches per paired initial state')
    sample_weights = np.ones(batch) if sample_weights is None else np.asarray(sample_weights,dtype=float)
    if (sample_weights.shape!=(batch,) or not np.isfinite(sample_weights).all()
            or np.any(sample_weights<0) or not np.any(sample_weights>0)):
        raise ValueError('expected finite nonnegative pair weights with positive mass')
    doubled_initial = np.concatenate((initial, initial), axis=0)
    exogenous = known.reshape(2*batch, horizon, known_dim)
    predicted = model.rollout(doubled_initial, exogenous).reshape(2, batch, horizon, 9)
    weights = lead_weights[:, None]*component_weights

    factor = model.state_error_factor(absolute_scale_si)
    absolute_error = (predicted-model.normalize(target))*factor
    absolute_weights = branch_mask[..., None]*weights*sample_weights[None,:,None,None]
    absolute_mass = absolute_weights.sum()
    if not absolute_mass > 0:
        raise ValueError('no complete branch targets at selected leads')
    absolute_weights = absolute_weights/absolute_mass
    absolute = float(np.sum(absolute_weights*absolute_error**2))

    half_spread = (predicted[0]-predicted[1])*model.state_scale*.5
    target_spread = (target[0]-target[1])*.5
    unresolved_excess = np.maximum(np.abs(half_spread)-response_floor, 0.)
    error_si = np.where(response_mask, half_spread-target_spread,
                        np.sign(half_spread)*unresolved_excess)
    residual = error_si/response_scale
    response_weights = noise_mask[..., None]*weights*sample_weights[:,None,None]
    response_mass = response_weights.sum()
    if not response_mass > 0:
        raise ValueError('no matched replay-noise estimates at selected leads')
    response_weights = response_weights/response_mass
    # Smooth L1-like tail keeps a few exaggerated rollout responses from
    # dominating the initial updates.
    robust_denominator = np.hypot(1., residual)
    response = float(np.sum(response_weights*2*(robust_denominator-1)))
    result = {'absolute':absolute, 'response':response,
              'loss':absolute_weight*absolute+response_weight*response}
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError('non-finite paired rollout loss')
    if not gradients:
        return result, None

    cotangent = (2*absolute_weight*absolute_weights)*absolute_error*factor
    response_grad = (response_weight*response_weights*residual/robust_denominator
                     *model.state_scale/response_scale)
    cotangent[0] += response_grad
    cotangent[1] -= response_grad
    _, _, grads = model.rollout_vjp(doubled_initial, exogenous, cotangent.reshape(2*batch, horizon, 9))
    return result, grads
