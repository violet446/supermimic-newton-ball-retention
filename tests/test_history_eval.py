from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from ball_retention.action_eval import evaluate_pair
from evaluate_s3_history import aggregate


class HistoryEvaluationTests(unittest.TestCase):
    def row(self,signal,model_sign=1):
        plus=np.zeros((1,9));minus=plus.copy();plus[0,0]=signal;minus[0,0]=-signal
        replay=np.zeros((1,9))
        response=evaluate_pair(model_sign*plus,model_sign*minus,plus,minus,.2,[0,0],
            replay_first=replay,replay_second=replay,leads=(1,))
        return {'response':response,'absolute':{'1':{'branches':2,'squared_p_v_w_si':[8.,0.,0.]}}}

    def test_strong_direction_is_separate_from_near_noise_case(self):
        rows=[self.row(.001),self.row(.002,-1),self.row(2e-6)]
        result=aggregate(rows,[1],.5,10)['1']
        self.assertEqual(result['position_all']['resolved'],3)
        self.assertEqual(result['position_strong']['pairs'],2)
        self.assertEqual(result['position_strong']['positive_cosine'],1)
        self.assertEqual(result['position_strong']['cosine_ge_0_5'],1)

    def test_vector_rmse_pools_squared_errors_by_actual_branch_count(self):
        first=self.row(.001);second=self.row(.001)
        second['absolute']['1']={'branches':1,'squared_p_v_w_si':[9.,0.,0.]}
        result=aggregate([first,second],[1],.5,10)['1']
        self.assertAlmostEqual(result['absolute_rmse_p_v_w_si'][0],np.sqrt(17/3))
        self.assertEqual(result['branches'],3)
