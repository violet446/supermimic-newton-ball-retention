"""Residual dynamics conditioned on a local, differentiable tool surface."""
import numpy as np

from .model_data import KNOWN_NAMES, filter_advance
from .world_model import STATE_DIM, ResidualMLP, _array


FEATURE_DIM = len(KNOWN_NAMES)
SURFACE_DIM = 7
ACTUATOR_COLS = slice(2, 8)
MODEL_TYPE = 'local_surface_residual_v1'


class LocalSurfaceResidualMLP(ResidualMLP):
    """Keep the base transition and add seven state-dependent MLP inputs."""

    def __init__(self, surface, config, known_mean, known_scale, state_mean,
                 state_scale, dt_s, seed, hidden=32, residual_gain=.1):
        super().__init__(FEATURE_DIM, state_mean, state_scale, dt_s, seed,
                         hidden=hidden, residual_gain=residual_gain)
        self.surface = surface
        self.known_mean = _array(known_mean, (FEATURE_DIM,), 'known mean').copy()
        self.known_scale = _array(known_scale, (FEATURE_DIM,), 'known scale').copy()
        if np.any(self.known_scale <= 0.):
            raise ValueError('known scale must be positive')
        for values in (self.known_mean, self.known_scale):
            values.setflags(write=False)
        self.params['W1'] = np.vstack((self.params['W1'], np.zeros((SURFACE_DIM, hidden))))

        self._base_rotation = _array(config['initial']['tool_rotation_world'], (3, 3),
                                     'base tool rotation').copy()
        tau = float(config['action']['tilt_filter_time_constant_s'])
        limit = np.deg2rad(config['action']['tilt_limit_deg'])
        if not np.isfinite([tau, limit]).all() or min(tau, limit) <= 0.:
            raise ValueError('invalid tool filter time constant or tilt limit')
        # Apply the affine held-target filter to a basis of raw midpoint
        # request, tilt and tilt rate. Advance backwards by half a period.
        request, tilt, rate = np.moveaxis(np.eye(6).reshape(6, 3, 2), 1, 0)
        start_tilt, start_rate, _ = filter_advance(tilt, rate, limit*request, tau, -.5*self.dt_s)
        endpoint_map = np.column_stack((start_tilt[:, 0], start_rate))
        self._endpoint_map = self.known_scale[ACTUATOR_COLS, None]*endpoint_map
        self._endpoint_bias = self.known_mean[ACTUATOR_COLS]@endpoint_map

    def tool_omega(self, known, *, jacobian=False):
        """Derive start angular velocity from standardized causal tool features.

        The optional Jacobian is with respect to known[..., 2:8], covering
        the held request, midpoint tilt and midpoint tilt rate.
        """
        start = known[..., ACTUATOR_COLS]@self._endpoint_map+self._endpoint_bias
        roll, roll_rate, pitch_rate = np.moveaxis(start, -1, 0)
        cosine, sine = np.cos(roll), np.sin(roll)
        # Express world-Y pitch and the pitch-rotated roll axis in tool axes.
        # The pitch rotation cancels; the fixed tool mount remains.
        omega = np.stack((roll_rate, pitch_rate*cosine, -pitch_rate*sine), axis=-1)@self._base_rotation
        if not jacobian:
            return omega
        endpoint_jac = np.zeros((*start.shape[:-1], 3, 3))
        endpoint_jac[..., 0, 1] = 1.
        endpoint_jac[..., 1, 0] = -pitch_rate*sine
        endpoint_jac[..., 1, 2] = cosine
        endpoint_jac[..., 2, 0] = -pitch_rate*cosine
        endpoint_jac[..., 2, 2] = -sine
        return omega, self._base_rotation.T@endpoint_jac@self._endpoint_map.T

    def _surface_step(self, state, known, omega, omega_jac=None):
        if omega_jac is None:
            surface = self.surface.features(self.denormalize(state), omega)
            return super()._forward(state, np.concatenate((known, surface), axis=-1))[0], None
        surface, state_jac, angular_jac = self.surface.features(self.denormalize(state), omega, jacobian=True)
        predicted, net_cache = super()._forward(state, np.concatenate((known, surface), axis=-1))
        return predicted, (net_cache, state_jac, angular_jac, omega_jac)

    def _forward(self, state, known):
        return self._surface_step(state, known, *self.tool_omega(known, jacobian=True))

    def step(self, state, known):
        state, known = self._step_inputs(state, known)
        return self._surface_step(state, known, self.tool_omega(known))[0]

    def _backward(self, cache, cotangent):
        net_cache, state_jac, angular_jac, omega_jac = cache
        state_grad, input_grad, grads = super()._backward(net_cache, cotangent)
        surface_grad = input_grad[..., FEATURE_DIM:]
        state_grad += np.einsum('...i,...ij->...j', surface_grad, state_jac)*self.state_scale
        omega_grad = np.einsum('...i,...ij->...j', surface_grad, angular_jac)
        known_grad = input_grad[..., :FEATURE_DIM]
        known_grad[..., ACTUATOR_COLS] += np.einsum('...i,...ij->...j', omega_grad, omega_jac)
        return state_grad, known_grad, grads

    def _rollout(self, initial, known, record):
        initial = np.asarray(initial, dtype=float)
        known = np.asarray(known, dtype=float)
        if initial.ndim != 2 or initial.shape[1] != STATE_DIM or len(initial) == 0:
            raise ValueError('initial state must have shape (batch, 9)')
        if known.ndim != 3 or known.shape[0] != len(initial) or not known.shape[1]:
            raise ValueError('known features must have shape (batch, horizon, 20)')
        initial, known = _array(initial, initial.shape, 'initial state'), _array(
            known, (*known.shape[:2], FEATURE_DIM), 'known features')
        if record:
            omega, jacobian = self.tool_omega(known, jacobian=True)
        else:
            omega = self.tool_omega(known)
            jacobian = None
        prediction = np.empty((*known.shape[:2], STATE_DIM))
        state = initial
        caches = []
        for tick in range(known.shape[1]):
            omega_jac = jacobian[:, tick] if record else None
            state, cache = self._surface_step(state, known[:, tick], omega[:, tick], omega_jac)
            prediction[:, tick] = state
            if record:
                caches.append(cache)
        return prediction, caches

    def state_dict(self):
        return {**super().state_dict(), 'model_type': np.array(MODEL_TYPE),
                'known_mean': self.known_mean.copy(), 'known_scale': self.known_scale.copy()}

    @classmethod
    def from_checkpoint(cls, saved, surface, config):
        if str(saved['model_type']) != MODEL_TYPE or int(saved['known_dim']) != FEATURE_DIM:
            raise ValueError('invalid local surface residual checkpoint')
        model = cls(surface, config, saved['known_mean'], saved['known_scale'],
                    saved['state_mean'], saved['state_scale'], float(saved['dt_s']), seed=0,
                    hidden=int(saved['hidden']), residual_gain=saved['residual_gain'])
        for name, parameter in model.params.items():
            parameter[...] = _array(saved[name], parameter.shape, name)
        return model
