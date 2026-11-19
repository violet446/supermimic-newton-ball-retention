import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.action_grad import action_sensitivity, predict_ball
from ball_retention.model_data import KNOWN_NAMES
from ball_retention.world_model import ResidualMLP


class ActionGradientTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261111)
        self.config = json.loads((ROOT/'configs/s1_online_episode.json').read_text())
        self.state = np.array([.24, .035, .017, .1, -.02, .01, .15, -.06, 1.1])
        n = len(KNOWN_NAMES)
        self.model = ResidualMLP(n, self.state, [.03]*3+[.2]*3+[1.]*3,
                                 1/60, seed=4187, hidden=8)
        self.model.params['W3'][...] = rng.normal(scale=.25, size=(8, 9))
        self.mean = np.linspace(-.1, .2, n)
        self.scale = np.linspace(.3, 4., n)
        self.actions = np.array([[.13, -.09], [.03, .05], [.12, -.14], [-.05, .03],
                                 [.09, .07], [-.03, .1], [.04, -.09], [.17, .04]])

    def calculate(self, fn, actions):
        return fn(self.model, self.state, 1.7, [.04, -.07], [-.09, .12], actions,
                  self.config, self.mean, self.scale)

    def test_shared_offsets_match_full_rollout_central_difference(self):
        result = self.calculate(action_sensitivity, self.actions)
        np.testing.assert_array_equal(result['prediction'], self.calculate(predict_ball, self.actions))
        epsilon = 1e-6
        for axis in range(2):
            offset = epsilon*np.eye(2)[axis]
            difference = (self.calculate(predict_ball, self.actions+offset)
                          -self.calculate(predict_ball, self.actions-offset))/(2*epsilon)
            np.testing.assert_allclose(result['jacobian'][:, :, axis], difference,
                                       rtol=3e-6, atol=3e-8)
            self.assertGreater(np.linalg.norm(difference[-1, :3]), 1e-5)

    def test_saturation_is_flat_and_exact_kinks_are_rejected(self):
        self.actions[:, 0] = 1.15
        result = self.calculate(action_sensitivity, self.actions)
        np.testing.assert_array_equal(result['jacobian'][:, :, 0], 0.)
        self.assertGreater(np.linalg.norm(result['jacobian'][:, :, 1]), 0.)
        self.actions[2, 0] = -1.
        with self.assertRaisesRegex(ValueError, 'clipping boundary'):
            self.calculate(action_sensitivity, self.actions)
        self.assertTrue(np.isfinite(self.calculate(predict_ball, self.actions)).all())


if __name__ == '__main__':
    unittest.main()
