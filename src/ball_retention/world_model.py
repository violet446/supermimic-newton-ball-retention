"""Differentiable residual ball dynamics in standardized tool-frame coordinates.

The rollout takes the initial ball state and a sequence of known features. The
state consists of tool-frame position, relative velocity, and world angular
velocity expressed in the tool frame, in that order. The known features and
their derivatives are supplied by the caller.
"""
import numpy as np


STATE_DIM = 9


def _array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return array


def _nonnegative_weights(value, size, name):
    weights = np.ones(size) if value is None else _array(value, (size,), name)
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError(f'{name} must have nonnegative weights and a positive sum')
    return weights


class ResidualMLP:
    """Two tanh layers on top of a constant-relative-velocity transition."""

    def __init__(self, known_dim, state_mean, state_scale, dt_s, seed, hidden=32,
                 residual_gain=.1):
        if not isinstance(known_dim, (int, np.integer)) or known_dim < 0:
            raise ValueError('known_dim must be a nonnegative integer')
        if not isinstance(hidden, (int, np.integer)) or hidden < 1:
            raise ValueError('hidden must be a positive integer')
        self.known_dim, self.hidden = int(known_dim), int(hidden)
        self.dt_s = float(dt_s)
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError('dt_s must be positive and finite')
        self.state_mean = _array(state_mean, (STATE_DIM,), 'state_mean').copy()
        self.state_scale = _array(state_scale, (STATE_DIM,), 'state_scale').copy()
        if np.any(self.state_scale <= 0):
            raise ValueError('state_scale must be positive')
        gain = np.asarray(residual_gain, dtype=np.float64)
        if gain.ndim == 0:
            gain = np.full(STATE_DIM, gain)
        self.residual_gain = _array(gain, (STATE_DIM,), 'residual_gain').copy()
        if np.any(self.residual_gain < 0):
            raise ValueError('residual_gain must be nonnegative')
        for array in (self.state_mean, self.state_scale, self.residual_gain):
            array.setflags(write=False)

        self._velocity_gain = self.dt_s*self.state_scale[3:6]/self.state_scale[:3]
        self._position_bias = self.dt_s*self.state_mean[3:6]/self.state_scale[:3]
        rng = np.random.default_rng(seed)
        width = STATE_DIM+self.known_dim
        self.params = {
            'W1': rng.standard_normal((width, hidden))/np.sqrt(width),
            'b1': np.zeros(hidden),
            'W2': rng.standard_normal((hidden, hidden))/np.sqrt(hidden),
            'b2': np.zeros(hidden),
            'W3': np.zeros((hidden, STATE_DIM)),
            'b3': np.zeros(STATE_DIM),
        }

    def normalize(self, state):
        return (np.asarray(state, dtype=np.float64)-self.state_mean)/self.state_scale

    def denormalize(self, state):
        return np.asarray(state, dtype=np.float64)*self.state_scale+self.state_mean

    def _forward(self, state, known):
        params = self.params
        inputs = np.concatenate((state, known), axis=-1)
        first = np.tanh(inputs@params['W1']+params['b1'])
        second = np.tanh(first@params['W2']+params['b2'])
        delta = second@params['W3']+params['b3']
        prediction = state+self.residual_gain*delta
        prediction[..., :3] += self._velocity_gain*state[..., 3:6]+self._position_bias
        return prediction, (inputs, first, second)

    def _step_inputs(self, state, known):
        state = np.asarray(state, dtype=np.float64)
        if state.ndim < 1 or state.shape[-1] != STATE_DIM or not np.isfinite(state).all():
            raise ValueError('expected finite state with shape (..., 9)')
        known = _array(known, (*state.shape[:-1], self.known_dim), 'known features')
        return state, known

    def step(self, state, known):
        """Predict one standardized state from standardized inputs."""
        return self._forward(*self._step_inputs(state, known))[0]

    def _backward(self, cache, cotangent):
        inputs, first, second = cache
        params = self.params
        output_grad = (cotangent*self.residual_gain).reshape(-1, STATE_DIM)
        first_flat = first.reshape(-1, self.hidden)
        second_flat = second.reshape(-1, self.hidden)
        second_grad = (output_grad@params['W3'].T)*(1-second_flat**2)
        first_grad = (second_grad@params['W2'].T)*(1-first_flat**2)
        grads = {
            'W1': inputs.reshape(-1, inputs.shape[-1]).T@first_grad,
            'b1': first_grad.sum(axis=0),
            'W2': first_flat.T@second_grad,
            'b2': second_grad.sum(axis=0),
            'W3': second_flat.T@output_grad,
            'b3': output_grad.sum(axis=0),
        }
        inputs_grad = (first_grad@params['W1'].T).reshape(inputs.shape)
        state_grad = inputs_grad[..., :STATE_DIM]+cotangent
        state_grad[..., 3:6] += self._velocity_gain*cotangent[..., :3]
        return state_grad, inputs_grad[..., STATE_DIM:], grads

    def step_vjp(self, state, known, cotangent):
        """Return VJPs with respect to state, known features and parameters."""
        state, known = self._step_inputs(state, known)
        cotangent = _array(cotangent, state.shape, 'cotangent')
        _, cache = self._forward(state, known)
        return self._backward(cache, cotangent)

    def _rollout(self, initial, known, record):
        initial = np.asarray(initial, dtype=np.float64)
        known = np.asarray(known, dtype=np.float64)
        if initial.ndim != 2 or initial.shape[1] != STATE_DIM or len(initial) == 0:
            raise ValueError('initial state must have shape (batch, 9)')
        if (known.ndim != 3 or known.shape[0] != len(initial)
                or known.shape[1] == 0 or known.shape[2] != self.known_dim):
            raise ValueError('known features must have shape (batch, horizon, known_dim)')
        if not np.isfinite(initial).all() or not np.isfinite(known).all():
            raise ValueError('rollout inputs must be finite')
        batch, horizon = known.shape[:2]
        predictions = np.empty((batch, horizon, STATE_DIM))
        caches = []
        state = initial
        for tick in range(horizon):
            state, cache = self._forward(state, known[:, tick])
            predictions[:, tick] = state
            if record:
                caches.append(cache)
        return predictions, caches

    def rollout(self, initial, known):
        """Return one predicted state per step, feeding back each prediction."""
        return self._rollout(initial, known, record=False)[0]

    def _rollout_backward(self, caches, cotangent):
        batch, horizon = cotangent.shape[:2]
        known_grad = np.empty((batch, horizon, self.known_dim))
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        state_grad = np.zeros((batch, STATE_DIM))
        for tick in range(horizon-1, -1, -1):
            state_grad, known_grad[:, tick], step_grads = self._backward(
                caches[tick], state_grad+cotangent[:, tick])
            for name in grads:
                grads[name] += step_grads[name]
        return state_grad, known_grad, grads

    def rollout_vjp(self, initial, known, cotangent):
        """Backpropagate arbitrary costs on the rollout, including known features."""
        predictions, caches = self._rollout(initial, known, record=True)
        cotangent = _array(cotangent, predictions.shape, 'rollout cotangent')
        return self._rollout_backward(caches, cotangent)

    def state_error_factor(self, scale_si=None):
        """Convert standardized errors to fixed physical error units.

        This factor must not enter the weight normalizer: doing so would change
        total loss magnitude with the fitted input statistics.
        """
        if scale_si is None:
            return np.ones(STATE_DIM)
        scale = _array(scale_si, (STATE_DIM,), 'state error scale SI')
        if np.any(scale <= 0):
            raise ValueError('state error scale SI must be positive')
        return self.state_scale/scale

    def loss_and_grad(self, initial, known, target, *, step_weights=None, state_weights=None,
                      state_error_scale_si=None):
        """Weighted mean squared error and the BPTT parameter gradients."""
        predictions, caches = self._rollout(initial, known, record=True)
        target = _array(target, predictions.shape, 'target')
        step = _nonnegative_weights(step_weights, predictions.shape[1], 'step_weights')
        state = _nonnegative_weights(state_weights, STATE_DIM, 'state_weights')
        weights = (step/step.sum())[None, :, None]*(state/state.sum())[None, None, :]/len(predictions)
        factor = self.state_error_factor(state_error_scale_si)
        error = (predictions-target)*factor
        loss = float(np.sum(weights*error**2))
        if not np.isfinite(loss):
            raise FloatingPointError('non-finite rollout loss')
        _, _, grads = self._rollout_backward(caches, 2*weights*error*factor)
        return loss, grads

    def state_dict(self):
        """Array-only checkpoint suitable for np.savez without pickle."""
        state = {'state_mean': self.state_mean, 'state_scale': self.state_scale,
                 'residual_gain': self.residual_gain, 'dt_s': self.dt_s,
                 'known_dim': self.known_dim, 'hidden': self.hidden, **self.params}
        return {name: np.array(value, copy=True) for name, value in state.items()}

    @classmethod
    def from_state_dict(cls, state):
        model = cls(int(state['known_dim']), state['state_mean'], state['state_scale'],
                    float(state['dt_s']), seed=0, hidden=int(state['hidden']),
                    residual_gain=state['residual_gain'])
        for name, parameter in model.params.items():
            parameter[...] = _array(state[name], parameter.shape, name)
        return model


class Adam:
    """Adam with optional global gradient clipping and in-place updates."""

    def __init__(self, params, learning_rate=1e-3, max_grad_norm=None):
        self.params = params
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')
        if self.max_grad_norm is not None and (not np.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError('max_grad_norm must be positive and finite')
        self.first = {name: np.zeros_like(param) for name, param in params.items()}
        self.second = {name: np.zeros_like(param) for name, param in params.items()}
        self.iteration = 0

    def step(self, grads):
        if self.params.keys() != grads.keys():
            raise ValueError('optimizer gradient keys do not match parameters')
        grads = {name: _array(grads[name], value.shape, name) for name, value in self.params.items()}
        norm = float(np.sqrt(sum(np.sum(grad**2) for grad in grads.values())))
        if not np.isfinite(norm):
            raise FloatingPointError('non-finite gradient norm')
        scale = 1.
        if self.max_grad_norm is not None and norm > self.max_grad_norm:
            scale = self.max_grad_norm/norm
        self.iteration += 1
        correction_first = 1-.9**self.iteration
        correction_second = 1-.999**self.iteration
        for name, parameter in self.params.items():
            grad = scale*grads[name]
            first, second = self.first[name], self.second[name]
            first *= .9
            first += .1*grad
            second *= .999
            second += .001*grad**2
            parameter -= self.learning_rate*(first/correction_first)/(np.sqrt(second/correction_second)+1e-8)
        return norm
