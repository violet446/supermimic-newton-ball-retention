import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ball_retention.group_fold import make_group_fold
from ball_retention.model_data import Windows
from ball_retention.pair_data import PairBatch
from dataclasses import fields


class GroupFoldTests(unittest.TestCase):
    def fixture(self):
        rows=6
        values={f.name:np.zeros((rows,1)) for f in fields(PairBatch)}
        values.update(initial=np.zeros((rows,9)),group_ids=np.array(['a','a','b','b','b','b']),
                      plus_known=np.zeros((rows,2,20)),minus_known=np.zeros((rows,2,20)),
                      plus_target=np.zeros((rows,2,9)),minus_target=np.zeros((rows,2,9)),
                      plus_mask=np.ones((rows,2),bool),minus_mask=np.ones((rows,2),bool),
                      episode_ids=np.array(['ea','ea','eb','eb','eb','eb']),snapshot_ticks=np.zeros(rows,int),
                      initial_state=tuple({'id':i} for i in range(rows)))
        batch=PairBatch(**values)
        x=np.arange(36,dtype=float).reshape(4,9)
        windows={1:Windows(x,np.arange(80,dtype=float).reshape(4,1,20),x[:,None]+1,np.array(['ea','ea','eb','eb']),np.arange(4))}
        return windows,batch,{'ea':'a','eb':'b'}

    def test_holdout_cannot_change_training_normalization(self):
        w,b,m=self.fixture(); first=make_group_fold(w,b,m,'a')
        w[1].initial[:2]+=10000;w[1].target[:2]-=10000;w[1].known[:2]+=10000
        b.initial[:2]+=10000
        b.plus_known[:2]+=10000;b.minus_known[:2]-=10000
        b.plus_target[:2]+=10000;b.minus_target[:2]-=10000
        second=make_group_fold(w,b,m,'a')
        for key in ('state_mean','state_scale','known_mean','known_scale'):
            np.testing.assert_array_equal(getattr(first,key),getattr(second,key))
        self.assertEqual(first.report['retained_anchor_episodes'],['eb'])
        np.testing.assert_array_equal(first.held_indices,[0,1])
        np.testing.assert_array_equal(first.pair_weights,np.full(4,.25))

    def test_full_control_balances_groups_despite_extra_amplitudes(self):
        w,b,m=self.fixture();fold=make_group_fold(w,b,m)
        np.testing.assert_array_equal(fold.pair_weights,[.25,.25,.125,.125,.125,.125])
        self.assertEqual(len(fold.held_indices),0)
        with self.assertRaises(ValueError): make_group_fold(w,b,m,'unknown')

    def test_retained_pair_variation_prevents_floor_saturation(self):
        w,b,m=self.fixture();w[1].known[:]=0
        b.plus_known[:,:,3]=.5;b.minus_known[:,:,3]=-.5
        fold=make_group_fold(w,b,m,'a')
        self.assertAlmostEqual(fold.known_mean[3],0.)
        self.assertAlmostEqual(fold.known_scale[3],np.sqrt(.125))
        legacy=make_group_fold(w,b,m,'a',normalization='s2_only_v1')
        self.assertEqual(legacy.known_scale[3],.0001)

    def test_masked_targets_do_not_enter_statistics(self):
        w,b,m=self.fixture();b.plus_mask[:,1]=False
        before=make_group_fold(w,b,m)
        b.plus_target[:,1]=np.nan
        after=make_group_fold(w,b,m)
        np.testing.assert_array_equal(before.state_mean,after.state_mean)
        np.testing.assert_array_equal(before.state_scale,after.state_scale)

    def test_duplicate_group_rows_do_not_change_statistics(self):
        from ball_retention.group_fold import subset_pairs
        w,b,m=self.fixture();b.plus_known[:2]=4;b.plus_target[:2]=3
        before=make_group_fold(w,b,m)
        duplicated=subset_pairs(b,np.array([0,1,0,1,2,3,4,5]))
        after=make_group_fold(w,duplicated,m)
        for key in ('state_mean','state_scale','known_mean','known_scale'):
            np.testing.assert_allclose(getattr(before,key),getattr(after,key),rtol=1e-14,atol=1e-14)
