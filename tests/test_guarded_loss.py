import copy
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ball_retention.guarded_loss import center_direction_loss,checkpoint_guard
from ball_retention.world_model import ResidualMLP

class GuardedLossTests(unittest.TestCase):
    def test_center_and_direction_parameter_gradients(self):
        rng=np.random.default_rng(84)
        model=ResidualMLP(3,np.zeros(9),np.linspace(.02,.2,9),1/60,seed=4,hidden=5)
        model.params['W3'][:]=rng.normal(scale=.1,size=(5,9))
        x=rng.normal(size=(2,9));known=rng.normal(size=(2,2,4,3))
        target=rng.normal(scale=.01,size=(2,2,4,9))
        mask=np.ones((2,2,4),bool);mask[0,1,-1]=False
        noise=np.ones((2,4),bool)
        baseline=np.zeros((2,4,9))
        cfg={'center_position_scale_m':.005,'center_position_weight':.5,'direction_weight':.2,'direction_target_cosine':.5}
        def objective():
            return center_direction_loss(model,x,known,target,mask,noise,baseline,baseline,
                                         np.array([0.,.2,.3,.5]),np.array([.3,.7]),cfg)
        for center,direction in [(1.,0.),(0.,1.),(.5,.2)]:
            cfg.update(center_position_weight=center,direction_weight=direction)
            loss,grads=objective()
            for key,index in [('W1',(3,2)),('W3',(2,0)),('b3',(1,))]:
                old=model.params[key][index];eps=1e-6
                model.params[key][index]=old+eps;plus=objective()[0]['loss']
                model.params[key][index]=old-eps;minus=objective()[0]['loss']
                model.params[key][index]=old
                np.testing.assert_allclose(grads[key][index],(plus-minus)/(2*eps),atol=2e-6,rtol=2e-5)
        noise[:]=False;cfg.update(center_position_weight=0.,direction_weight=1.)
        loss,grads=objective()
        self.assertEqual(loss['loss'],0.)
        for v in grads.values(): np.testing.assert_array_equal(v,0.)

    def test_guard_rejects_group_regression_even_if_aggregate_improves(self):
        metrics={'absolute_rmse_p_v_w_si':[.01,.1,1.],
                 'position_strong':{'resolved':4,'positive_cosine':4,'cosine_ge_0_5':3}}
        before={'summary':{'validation/group_a/0.2':{'32':metrics},
                           'validation/coverage/0.2':{'32':copy.deepcopy(metrics)}},
                'complete_route_ids':['route'],'s2_validation':{'8':{'route':{'rmse_p_v_w_si':[.003,.1,1.]}}}}
        cfg={'H32_position_rmse_max_ratio':1.1,'S2_complete_route_H8_position_rmse_max_ratio':1.1}
        self.assertTrue(checkpoint_guard(before,before,cfg)['passed'])
        after=copy.deepcopy(before)
        after['summary']['validation/coverage/0.2']['32']['absolute_rmse_p_v_w_si'][0]=.001
        after['summary']['validation/group_a/0.2']['32']['absolute_rmse_p_v_w_si'][0]=.012
        self.assertFalse(checkpoint_guard(before,after,cfg)['passed'])
        after=copy.deepcopy(before)
        after['summary']['validation/group_a/0.2']['32']['position_strong']['positive_cosine']=3
        self.assertFalse(checkpoint_guard(before,after,cfg)['passed'])
        after=copy.deepcopy(before);after['s2_validation']['8']['route']['rmse_p_v_w_si'][0]=.004
        self.assertFalse(checkpoint_guard(before,after,cfg)['passed'])
