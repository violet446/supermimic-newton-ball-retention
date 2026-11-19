import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.geometry import load_tool, sphere_clearance, vertical_hits
from ball_retention.robot import Robot
from ball_retention.support import CarryingRegion


class CarryingRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = json.loads((ROOT/'configs/s1_arm_path.json').read_text())
        cls.mesh = load_tool(ROOT/'assets/shovel.STL')
        cls.rotation = np.array(cfg['initial']['tool_rotation_world'])
        cls.radius = cfg['ball']['radius_m']
        cls.region = CarryingRegion(cls.mesh, cls.rotation, cls.radius)
        robot = Robot(ROOT/'assets/ec66_description.urdf')
        tool, _, _ = robot.fk(np.asarray(cfg['initial']['joint_q_rad']))
        cls.initial_ball = tool[:3, :3].T @ (np.array(cfg['initial']['ball_center_world_m'])-tool[:3, 3])

    def test_baseline_and_airborne_over_bowl(self):
        self.assertTrue(self.region.measure(self.initial_ball)['carrying_candidate'])
        airborne = self.region.measure(self.initial_ball+.010*self.rotation[2])
        self.assertTrue(airborne['within_bowl'])
        self.assertFalse(airborne['carrying_candidate'])

    def test_near_handle_and_underside_are_not_carrying(self):
        carrier_ball = self.rotation @ self.initial_ball
        hits = vertical_hits(self.region.carrier_mesh, carrier_ball[:2])
        lowest = min(hit['z'] for hit in hits)
        underside = self.rotation.T @ np.array([*carrier_ball[:2], lowest-self.radius])
        near_handle = np.array([.15, 0., .0424])
        for ball, reason in [(near_handle, 'handle_or_connector'), (underside, 'below_upper_face')]:
            with self.subTest(reason=reason):
                self.assertLess(abs(sphere_clearance(self.mesh, ball, self.radius)), .002)
                result = self.region.measure(ball)
                self.assertFalse(result['carrying_candidate'])
                self.assertEqual(result['reason'], reason)
