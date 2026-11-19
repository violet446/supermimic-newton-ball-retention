import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from ball_retention.action_eval import compare_scales, evaluate_pair, full_step_states


class ActionEvaluationTests(unittest.TestCase):
    def test_direction_magnitude_progress_and_scale(self):
        base = np.array([.05, 0., 0., 0., 0., 0., 0., 0., 0.])
        physical = np.array([-.004, 0., 0., .01, 0., 0., 0., 0., .2])
        model = np.array([-.003, -.004, 0., -.01, 0., 0., 0., 0., .4])

        def evaluate(epsilon, sign=1.):
            return evaluate_pair([base+epsilon*model], [base-epsilon*model],
                                 [base+epsilon*sign*physical], [base-epsilon*sign*physical],
                                 epsilon, [0., 0.], replay_first=[base], replay_second=[base], leads=(1,))

        small = evaluate(.1)
        row = small['by_lead']['1']
        self.assertAlmostEqual(row['position']['model_norm'], .005)
        self.assertAlmostEqual(row['position']['cosine'], .6)
        self.assertAlmostEqual(row['position']['normalized_error'], np.hypot(.001, .004)/.004)
        self.assertEqual(row['velocity']['direction'], 'opposed')
        self.assertTrue(row['progress_xy']['sign_agrees'])
        self.assertGreater(row['progress_xy']['newton_derivative'], 0.)
        self.assertTrue(compare_scales(small, evaluate(.2))['by_lead']['1']['position']['consistent'])
        self.assertFalse(compare_scales(small, evaluate(.2, sign=-1.))['by_lead']['1']['position']['consistent'])
        json.dumps(small, allow_nan=False)

    def test_shortened_and_missing_leads_and_replay_noise_are_explicit(self):
        base = np.zeros(9)
        effect = np.zeros(9)
        effect[0] = 2e-6
        plus = full_step_states([effect, effect], [16, 5], 16)
        minus = [base, base, base]
        result = evaluate_pair([effect]*3, minus, plus, minus, .1, [.05, 0.],
                               replay_first=[base]*3, replay_second=[effect]*3, leads=(1, 2, 3))
        first, second, third = (result['by_lead'][str(i)] for i in (1, 2, 3))
        self.assertAlmostEqual(first['position']['signal_threshold'], 5e-6)
        self.assertEqual(first['position']['direction'], 'both_below_floor')
        self.assertIsNone(first['position']['cosine'])
        self.assertEqual(second['status'], 'missing_newton_lead')
        self.assertEqual(second['step_status']['newton_plus'], 'shortened_step')
        self.assertEqual(third['step_status']['newton_plus'], 'not_reached')
        self.assertIsNone(first['progress_xy']['sign_agrees'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
