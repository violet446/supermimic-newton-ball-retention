import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from train_s3_history import regression


class HistoryRefitTests(unittest.TestCase):
    def fixture(self):
        return {'summary':{'group/validation/g/0.2':{'32':{'absolute_rmse_p_v_w_si':[.01,0,0],
            'position_strong':{'pairs':4,'positive_cosine':4,'cosine_ge_0_5':3}}}},
            'complete_route_ids':['route'],'s2_validation':{'8':{'route':[.002,0,0]}}}

    def test_group_direction_and_route_protection_are_independent(self):
        a=self.fixture();b=copy.deepcopy(a)
        self.assertTrue(regression(a,b)['passed'])
        b['summary']['group/validation/g/0.2']['32']['absolute_rmse_p_v_w_si'][0]=.005
        b['summary']['group/validation/g/0.2']['32']['position_strong']['positive_cosine']=3
        b['s2_validation']['8']['route'][0]=.003
        failures=regression(a,b)['failures']
        self.assertIn('group/validation/g/0.2/positive_cosine',failures)
        self.assertIn('S2_route/route',failures)
        self.assertNotIn('group/validation/g/0.2/position',failures)

    def test_changed_physical_response_set_is_rejected(self):
        a=self.fixture();b=copy.deepcopy(a)
        b['summary']['group/validation/g/0.2']['32']['position_strong']['pairs']=3
        with self.assertRaises(ValueError):regression(a,b)
