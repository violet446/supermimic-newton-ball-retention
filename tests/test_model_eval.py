from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from ball_retention.model_eval import evaluate_termination, evaluate_windows


class ModelEvaluationTests(unittest.TestCase):
    def test_vector_rmse_units_and_same_initial_state_baselines(self):
        initial = np.zeros((2, 9))
        initial[:, 3:5] = [0.03, 0.04]
        target = np.zeros((2, 4, 9))
        target[:, -1, :3] = initial[:, 3:6] * 0.08
        target[:, -1, 3:] = initial[:, 3:]
        predicted = target.copy()
        predicted[0, -1, :3] += [0.003, 0.004, 0.]
        predicted[0, -1, 3:6] += [0.03, 0.04, 0.]
        predicted[0, -1, 6:] += [3., 4., 0.]

        metrics = evaluate_windows(predicted, target, initial, 0.02, ['a', 'b'], leads=(4,))
        lead = metrics['by_lead']['4']
        self.assertEqual(lead['windows'], 2)
        self.assertAlmostEqual(lead['model']['position_rmse_mm'], 5 / np.sqrt(2))
        self.assertAlmostEqual(lead['model']['relative_velocity_rmse_m_s'], 0.05 / np.sqrt(2))
        self.assertAlmostEqual(lead['model']['angular_velocity_rmse_rad_s'], 5 / np.sqrt(2))
        self.assertAlmostEqual(lead['persistence']['position_rmse_mm'], 4)
        self.assertAlmostEqual(lead['constant_relative_velocity']['position_rmse_mm'], 0)
        self.assertAlmostEqual(metrics['by_episode']['a']['4']['model']['position_rmse_mm'], 5)

    def test_rare_termination_counts_and_extreme_probabilities(self):
        metrics = evaluate_termination([1., 0., 0.9], [False, False, True], 0.01, episode_ids=['a', 'a', 'b'])
        self.assertEqual(metrics['positive_count'], 1)
        self.assertEqual(metrics['model']['true_positive'], 1)
        self.assertEqual(metrics['model']['false_positive'], 1)
        self.assertEqual(metrics['prior']['false_negative'], 1)
        self.assertAlmostEqual(metrics['model']['brier'], (1 + 0.01) / 3)
        self.assertTrue(np.isfinite(metrics['model']['log_loss']))
        self.assertEqual(metrics['by_episode']['a']['positive_count'], 0)
        self.assertEqual(metrics['by_episode']['b']['positive_count'], 1)


if __name__ == '__main__':
    unittest.main()
