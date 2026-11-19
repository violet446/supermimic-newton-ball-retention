"""Blend free velocity half-difference supervision with independent local increments."""
import numpy as np

from .pair_loss import paired_loss
from .velocity_response_loss import VELOCITY


def prior(initial, endpoints):
    """Prepend the common initial state to an array of branch endpoints."""
    first = np.broadcast_to(initial[:, None], (*endpoints.shape[:2], 1, endpoints.shape[-1]))
    return np.concatenate((first, endpoints[:, :, :-1]), axis=2)


def half(branches):
    return .5*(branches[0]-branches[1])


class LocalVelocityResponse:
    def __init__(self, batch, original_weights, samples, fraction, floor_m_s):
        if not 0 <= fraction < 1 or not np.isfinite(floor_m_s) or floor_m_s <= 0:
            raise ValueError('expected a blend in [0, 1) and a positive finite noise floor')
        observed = np.stack((batch.plus_target, batch.minus_target))
        nominal = np.stack((batch.baseline_first, batch.baseline_second))
        self.prior = prior(batch.initial, observed)
        self.target = half((observed-self.prior)[..., VELOCITY])
        noise = abs(half((nominal-prior(batch.initial, nominal))[..., VELOCITY]))
        self.floor = np.maximum(floor_m_s, 5.*noise)
        self.resolved = abs(self.target) >= self.floor
        full = batch.plus_mask & batch.minus_mask & batch.baseline_mask
        previous_full = np.concatenate((np.ones_like(full[:, :1]), full[:, :-1]), axis=1)
        self.eligible = batch.noise_mask & full & previous_full

        effective = original_weights*batch.noise_mask[..., None]*samples[:, None, None]
        self.total_mass = float(effective.sum())
        if self.total_mass <= 0:
            raise ValueError('no original response supervision')
        moved = fraction*effective[..., VELOCITY].sum(axis=0)
        available = (self.eligible*samples[:, None]).sum(axis=0)
        if np.any((available == 0.) & (moved.sum(axis=1) > 0.)):
            raise ValueError('a weighted velocity lead has no adjacent complete replay')
        distributed = np.divide(moved, available[:, None], out=np.zeros_like(moved), where=available[:, None] > 0.)
        self.local_weights = distributed*self.eligible[..., None]*samples[:, None, None]
        self.free_components = original_weights.copy()
        self.free_components[..., VELOCITY] *= 1.-fraction
        remainder = self.free_components*batch.noise_mask[..., None]*samples[:, None, None]
        self.free_fraction = float(remainder.sum()/self.total_mass) if fraction else 1.
        combined = remainder.sum(axis=0)
        combined[..., VELOCITY] += self.local_weights.sum(axis=0)
        np.testing.assert_allclose(combined, effective.sum(axis=0), rtol=0., atol=1e-14)
        den = self.local_weights.sum()
        self.audit = {'response_mass': self.total_mass,
                      'moved_response_mass': float(den), 'original_velocity_mass': float(effective[..., VELOCITY].sum()),
                      'effective_eligible_pairs_by_lead': self.eligible.sum(axis=0).tolist(),
                      'original_noise_pairs_by_lead': batch.noise_mask.sum(axis=0).tolist(),
                      'max_nominal_increment_half_spread_m_s': float(noise[self.eligible].max()),
                      'max_weighted_floor_m_s': float(self.floor[self.eligible].max()),
                      'resolved_local_weight_fraction': float(self.local_weights[self.resolved].sum()/den) if den else None,
                      'max_lead_component_mass_delta': float(abs(combined-effective.sum(axis=0)).max())}

    def estimate(self, model, known):
        start = model.normalize(self.prior)
        prediction = model.step(start, known)
        value = half((prediction-start)[..., VELOCITY]*model.state_scale[VELOCITY])
        return start, prediction, value

    def local_loss(self, model, known, response_units, *, gradients):
        start, prediction, increment = self.estimate(model, known)
        error = np.where(self.resolved, increment-self.target,
                         np.sign(increment)*np.maximum(abs(increment)-self.floor, 0.))
        scale = np.asarray(response_units)[VELOCITY]
        residual = error/scale
        weights = self.local_weights/self.total_mass
        robust = np.hypot(1., residual)
        loss = float(np.sum(weights*2*(robust-1.)))
        if not gradients:
            return loss, None
        cotangent = np.zeros_like(prediction)
        cotangent[0, ..., VELOCITY] = weights*residual/robust*model.state_scale[VELOCITY]/scale
        cotangent[1] = -cotangent[0]
        return loss, model.step_vjp(start, known, cotangent)[2]

    def loss(self, model, initial, known, target, branch_mask, response_mask, noise_mask, response_floor,
             lead_weights, component_weights, response_scale, absolute_weight, response_weight, *,
             velocity_lead_weights=None, sample_weights=None, absolute_scale_si=None, gradients=True):
        """Keep the original absolute term, reusing the original free-response VJP."""
        args = (model, initial, known, target, branch_mask, response_mask, noise_mask, response_floor)
        options = {'gradients': gradients, 'sample_weights': sample_weights, 'absolute_scale_si': absolute_scale_si}
        state, state_grad = paired_loss(*args, lead_weights, component_weights, response_scale, absolute_weight,
                                       0., **options)
        free, free_grad = paired_loss(*args, np.ones(len(lead_weights)), self.free_components, response_scale,
                                     0., response_weight*self.free_fraction, **options)
        local, local_grad = self.local_loss(model, known, response_scale, gradients=gradients)
        response = self.free_fraction*free['response']+local
        total = state['loss']+free['loss']+response_weight*local
        if not np.isfinite(total):
            raise FloatingPointError('non-finite local velocity response loss')
        report = {'absolute': state['absolute'], 'response': response, 'loss': total,
                  'local_increment_contribution': local}
        if not gradients:
            return report, None
        return report, {name: value+free_grad[name]+response_weight*local_grad[name]
                        for name, value in state_grad.items()}
