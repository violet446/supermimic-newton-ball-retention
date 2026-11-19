import io
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.world_model import Adam, ResidualMLP


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4187)
        mean = [.2, -.03, .01, .15, -.07, .03, .8, -1.2, .1]
        scale = [.02, .01, .005, .2, .3, .4, 2., 3., 4.]
        self.model = ResidualMLP(5, mean, scale, 1/60, seed=20261111, hidden=8,
                                 residual_gain=[.05]*3+[.25]*6)
        self.initial = self.rng.normal(size=(3, 9))
        self.known = self.rng.normal(size=(3, 8, 5))

    def test_initial_model_matches_physical_constant_velocity(self):
        raw = self.model.denormalize(self.initial)
        expected = np.repeat(raw[:, None], 8, axis=1)
        elapsed = self.model.dt_s*np.arange(1, 9)
        expected[..., :3] += elapsed[None, :, None]*raw[:, None, 3:6]
        actual = self.model.denormalize(self.model.rollout(self.initial, self.known))
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)

    def test_rollout_gradients_reach_initial_state_and_every_feature_step(self):
        self.model.params['W3'][...] = self.rng.normal(scale=.2, size=(8, 9))
        costs = np.zeros((3, 8, 9))
        costs[:, -1] = self.rng.normal(size=(3, 9))
        ds, dk, dp = self.model.rollout_vjp(self.initial, self.known, costs)
        initial_direction = self.rng.normal(size=self.initial.shape)
        known_direction = self.rng.normal(size=self.known.shape)
        parameter_direction = {k: self.rng.normal(size=v.shape) for k, v in self.model.params.items()}
        original = {k: v.copy() for k, v in self.model.params.items()}

        def objective(initial, known):
            return float(np.sum(costs*self.model.rollout(initial, known)))

        epsilon = 1e-6
        state_fd = (objective(self.initial+epsilon*initial_direction, self.known)
                    -objective(self.initial-epsilon*initial_direction, self.known))/(2*epsilon)
        known_fd = (objective(self.initial, self.known+epsilon*known_direction)
                    -objective(self.initial, self.known-epsilon*known_direction))/(2*epsilon)
        objectives = []
        for sign in (1, -1):
            for key, value in original.items():
                self.model.params[key][...] = value+sign*epsilon*parameter_direction[key]
            objectives.append(objective(self.initial, self.known))
        param_fd = (objectives[0]-objectives[1])/(2*epsilon)
        self.assertAlmostEqual(float(np.sum(ds*initial_direction)), state_fd, places=7)
        self.assertAlmostEqual(float(np.sum(dk*known_direction)), known_fd, places=7)
        self.assertAlmostEqual(sum(float(np.sum(dp[k]*d)) for k, d in parameter_direction.items()),
                               param_fd, places=7)
        self.assertTrue(np.all(np.linalg.norm(dk, axis=(0, 2)) > 0))

    def test_weighted_fit_and_pickle_free_checkpoint(self):
        target = self.model.rollout(self.initial, self.known)+.2
        weights = {'step_weights': np.arange(1, 9), 'state_weights': [2.]*3+[1.]*6}
        loss, gradients = self.model.loss_and_grad(self.initial, self.known, target, **weights)
        self.assertAlmostEqual(loss, .04)
        optimizer = Adam(self.model.params, learning_rate=.005, max_grad_norm=1.)
        self.assertGreater(optimizer.step(gradients), 0.)
        updated, _ = self.model.loss_and_grad(self.initial, self.known, target, **weights)
        self.assertLess(updated, loss)
        buffer = io.BytesIO()
        np.savez(buffer, **self.model.state_dict())
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as checkpoint:
            restored = ResidualMLP.from_state_dict(checkpoint)
        np.testing.assert_array_equal(restored.rollout(self.initial, self.known),
                                      self.model.rollout(self.initial, self.known))


if __name__ == '__main__':
    unittest.main()
