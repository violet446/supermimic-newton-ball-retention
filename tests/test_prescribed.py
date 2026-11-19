import copy
import json
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.prescribed import PrescribedPath, collision_exclusions
from ball_retention.clearance import excluded_pairs
from ball_retention.robot import Robot
from ball_retention.geometry import load_tool, support_point
from ball_retention.task import reference


class PrescribedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT/'configs/s1_arm_path.json').read_text())
        cls.path = PrescribedPath(ROOT/cls.cfg['prescribed']['joint_path'], cls.cfg, True)
        cls.robot = Robot(ROOT/'assets/ec66_description.urdf')

    def test_curve_matches_material_reference_and_derivatives(self):
        anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), self.cfg['initial']['anchor_xy_tool_m'])[0]
        for t in np.linspace(0, 4, 241):
            q, qd, qdd = self.path.command(t)
            self.assertTrue(np.all((q >= self.robot.low) & (q <= self.robot.high)))
            self.assertLess(np.max(abs(qd)), self.cfg['robot']['joint_speed_rad_s'])
            self.assertLess(np.max(abs(qdd)), self.cfg['robot']['joint_acceleration_rad_s2'])
            tool, jacobian, _ = self.robot.fk(q)
            expected = reference(t, self.cfg)
            np.testing.assert_allclose(tool[:3, 3]+tool[:3, :3]@anchor, expected['anchor_position'], atol=2e-5)
            self.assertLess(Rotation.from_matrix(tool[:3, :3]@np.asarray(self.cfg['initial']['tool_rotation_world']).T).magnitude(), 2e-4)
            twist = jacobian@qd
            np.testing.assert_allclose(twist[:3]+np.cross(twist[3:], tool[:3, :3]@anchor), expected['anchor_velocity'], atol=2e-5)
        t, h = 1.537, 1e-5
        np.testing.assert_allclose((self.path.command(t+h)[0]-self.path.command(t-h)[0])/(2*h), self.path.command(t)[1], atol=1e-8)

    def test_hold_and_end_do_not_extrapolate(self):
        fixed = PrescribedPath(ROOT/self.cfg['prescribed']['joint_path'], self.cfg, False)
        for t in (0, 2, 4.1):
            q, qd, qdd = fixed.command(t)
            np.testing.assert_array_equal(q, self.cfg['initial']['joint_q_rad'])
            np.testing.assert_array_equal(qd, np.zeros(6))
        for t in (4, 4.1, 100):
            q, qd, qdd = self.path.command(t)
            np.testing.assert_array_equal(q, self.path.command(4)[0])
            np.testing.assert_array_equal(qd, np.zeros(6))
            np.testing.assert_array_equal(qdd, np.zeros(6))
        with self.assertRaises(ValueError): self.path.command(float('nan'))

    def test_filter_preserves_ball_floor_and_nonadjacent_pairs(self):
        names = ['base_link']+[f'link{i}' for i in range(1, 7)]+['shovel', 'ball', 'floor']
        pairs = collision_exclusions(names, list(range(9))+[-1], excluded_pairs())
        self.assertEqual({frozenset((names[a], names[b])) for a, b in pairs}, excluded_pairs())
        self.assertFalse(any(8 in pair or 9 in pair for pair in pairs))
        self.assertNotIn((0, 2), pairs)

    def test_fk_isolation_protects_dynamic_state_and_restores_clock(self):
        # FK must receive the separate arm-only state, never the physical scene.
        from ball_retention.env import PrescribedArmEnv
        class Array:
            def __init__(self, x): self.x = np.array(x, dtype=float)
            def numpy(self): return self.x.copy()
            def assign(self, x): self.x = np.array(x, dtype=float)
        class State:
            def __init__(self):
                self.joint_q, self.joint_qd = Array(np.zeros(13)), Array(np.zeros(12))
                self.body_q, self.body_qd = Array(np.arange(14).reshape(2, 7)), Array(np.arange(12).reshape(2, 6))
        class Backend:
            @staticmethod
            def eval_fk(model, q, qd, state):
                if len(state.body_q.x) != 1: raise AssertionError('FK received physical ball')
        env = PrescribedArmEnv.__new__(PrescribedArmEnv)
        env.a, env.b, env.ball_id = State(), State(), 1
        env.cfg, env.robot, env.path, env.model = copy.deepcopy(self.cfg), self.robot, self.path, None
        env.newton, env.wp = Backend, type('Warp', (), {'array': Array})
        env.fk_model, env.fk_state, env.arm_body_count = None, State(), 1
        env.fk_state.body_q.x = env.fk_state.body_q.x[:1]
        env.fk_state.body_qd.x = env.fk_state.body_qd.x[:1]
        env.com_local, env.fk_com_local = np.zeros((2, 3)), np.zeros((1, 3))
        env.steps, env.max_fk_ball_change = 100, 0
        env._new_solver = lambda: None
        env._prescribe(1.5)
        saved = env.snapshot()
        env.a.body_q.x[:] = 1000
        env.steps = 200
        env._prescribe(2)
        env.restore(saved)
        self.assertEqual(env.steps, 100)
        np.testing.assert_array_equal(env.a.body_q.numpy(), saved['arrays']['body_q'])
        np.testing.assert_array_equal(env.b.body_q.numpy(), saved['arrays']['body_q'])
        np.testing.assert_array_equal(env.command_q, self.path.command(1.5)[0])


if __name__ == '__main__': unittest.main()
