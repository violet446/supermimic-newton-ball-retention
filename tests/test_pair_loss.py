from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.pair_loss import paired_loss
from ball_retention.world_model import ResidualMLP


class PairLossTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(24)
        self.model = ResidualMLP(4, rng.normal(size=9), np.linspace(.2, 2., 9), 1/60, 17, hidden=5)
        self.model.params['W3'][...] = rng.normal(scale=.1, size=(5,9))
        self.initial = rng.normal(size=(2,9))
        self.known = rng.normal(size=(2,2,4,4))
        self.target = self.model.denormalize(rng.normal(size=(2,2,4,9)))
        self.branch_mask = np.ones((2,2,4),dtype=bool)
        self.branch_mask[1,0,-1] = False
        self.noise_mask = self.branch_mask.all(axis=0)
        self.response_mask = rng.random((2,4,9)) > .5
        self.response_floor = np.full((2,4,9), .001)

    def evaluate(self, *, absolute=.5, response=.2, gradients=True, sample_weights=None):
        return paired_loss(self.model,self.initial,self.known,self.target,self.branch_mask,
                           self.response_mask,self.noise_mask,self.response_floor,np.array([0.,.2,.3,.5]),
                           np.linspace(.4,1.5,9),np.linspace(.1,1.,9),absolute,response,gradients=gradients,
                           sample_weights=sample_weights)

    def test_paired_vjp_matches_central_difference_with_partial_branch(self):
        result, gradients = self.evaluate()
        self.assertGreater(result['loss'], 0.)
        for key,index in [('W1',(2,1)),('W2',(2,3)),('W3',(2,4)),('b3',(4,))]:
            values = self.model.params[key]
            original = values[index]
            h = 1e-6
            values[index] = original+h
            plus = self.evaluate(gradients=False)[0]['loss']
            values[index] = original-h
            minus = self.evaluate(gradients=False)[0]['loss']
            values[index] = original
            self.assertAlmostEqual(gradients[key][index],(plus-minus)/(2*h),delta=1e-8)

    def test_unresolved_response_within_noise_band_has_no_loss_or_gradient(self):
        self.response_mask[...] = False
        self.response_floor[...] = 1.
        result, gradients = self.evaluate(absolute=0.,response=1.)
        self.assertEqual(result['response'], 0.)
        for grad in gradients.values():
            np.testing.assert_array_equal(grad,0.)

    def test_weighted_pair_gradient_and_zero_weight_isolation(self):
        weights = np.array([.2,.8])
        _,grads = self.evaluate(sample_weights=weights)
        value = self.model.params['W3'][2,4]
        eps = 1e-6
        self.model.params['W3'][2,4] = value+eps
        plus = self.evaluate(sample_weights=weights,gradients=False)[0]['loss']
        self.model.params['W3'][2,4] = value-eps
        minus = self.evaluate(sample_weights=weights,gradients=False)[0]['loss']
        self.model.params['W3'][2,4] = value
        self.assertAlmostEqual(grads['W3'][2,4],(plus-minus)/(2*eps),delta=1e-8)
        first = self.evaluate(sample_weights=[1.,0.])[0]
        self.target[:,1] += 1000.
        self.assertEqual(first,self.evaluate(sample_weights=[1.,0.])[0])


if __name__ == '__main__':
    unittest.main()
