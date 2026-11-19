import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from ball_retention.history_capture import validate_history_case,DESIGN
from collect_s3_history import build_case


class HistoryCaptureTests(unittest.TestCase):
    def setUp(self):
        self.design=json.loads((ROOT/DESIGN).read_text())
        self.base=json.loads((ROOT/'configs/s1_online_episode.json').read_text())
        self.cfg=build_case(self.design,self.design['episodes'][0],self.base)

    def test_declared_histories_share_initial_and_split_but_not_schedule(self):
        other=build_case(self.design,self.design['episodes'][1],self.base)
        self.assertEqual(self.cfg['initial'],other['initial'])
        self.assertEqual(self.cfg['history_capture']['group_id'],other['history_capture']['group_id'])
        self.assertNotEqual(self.cfg['action_pairs']['schedule'],other['action_pairs']['schedule'])
        for cfg in (self.cfg,other):validate_history_case(ROOT,cfg)

    def test_seed_schedule_and_identity_mutation_rejected(self):
        for mutate in (lambda c:c.update(seed=1),lambda c:c['action_pairs'].update(master_seed=1),
                       lambda c:c['action_pairs']['schedule'][1].update(action=[.8,.8]),
                       lambda c:c['history_capture'].update(split='validation'),
                       lambda c:c['history_capture'].update(design_sha256='0'*64)):
            cfg=copy.deepcopy(self.cfg);mutate(cfg)
            with self.assertRaises(ValueError):validate_history_case(ROOT,cfg)

    def test_physics_initial_and_budget_changes_rejected(self):
        for mutate in (lambda c:c['physics'].update(iterations=10),
                       lambda c:c['initial'].update(ball_velocity_world_m_s=[.1,0,0]),
                       lambda c:c['budget'].update(wall_s=226)):
            cfg=copy.deepcopy(self.cfg);mutate(cfg)
            with self.assertRaises(ValueError):validate_history_case(ROOT,cfg)

    def test_capture_output_cap_is_separate_from_existing_shared_cache_guard(self):
        self.assertEqual(self.design['next_capture_budget']['max_output_mib'],512)
        self.assertEqual(self.cfg['budget']['max_output_cache_gib'],self.base['budget']['max_output_cache_gib'])
        cfg=copy.deepcopy(self.cfg);cfg['budget']['max_output_cache_gib']=.5
        with self.assertRaisesRegex(ValueError,'shared-cache guard'):validate_history_case(ROOT,cfg)
