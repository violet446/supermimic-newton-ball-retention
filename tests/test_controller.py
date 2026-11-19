import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.controller import OnlineTiltController
from ball_retention.geometry import load_tool, support_point
from ball_retention.prescribed import PrescribedPath
from ball_retention.robot import Robot
from ball_retention.task import reference


class OnlineTiltTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT/'configs/s1_arm_path.json').read_text())
        cls.robot = Robot(ROOT/'assets/ec66_description.urdf')
        cls.baseline = PrescribedPath(ROOT/cls.cfg['prescribed']['joint_path'], cls.cfg, True)
        cls.anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), cls.cfg['initial']['anchor_xy_tool_m'])[0]

    def make_controller(self):
        return OnlineTiltController(self.robot, self.cfg, self.anchor, self.baseline, ROOT/'assets')

    def test_zero_matches_prescribed_commands_exactly(self):
        controller = self.make_controller()
        saved = controller.snapshot()
        for command in controller.plan(0., [0., 0.], 16):
            q, qd, qdd = self.baseline.command(command.time_s)
            np.testing.assert_array_equal(command.q, q)
            np.testing.assert_array_equal(command.qd, qd)
            np.testing.assert_array_equal(command.qdd, qdd)
            np.testing.assert_array_equal(command.tilt, [0., 0.])
        self.assertEqual(controller.snapshot(), saved)

    def test_mixed_world_axes_preserve_anchor_pose_and_twist(self):
        controller = self.make_controller()
        t = 1.75
        q, qd, _ = self.baseline.command(t)
        controller.restore({**controller.snapshot(), 'time_s': t, 'q': q, 'qd': qd})
        commands = controller.plan(t, [.4, -.2], 16)
        for command in commands:
            pose, jacobian, _ = self.robot.fk(command.q)
            roll, pitch = command.tilt
            target_rotation = Rotation.from_euler('y', pitch).as_matrix()@Rotation.from_euler('x', roll).as_matrix()@controller.base_rotation
            anchor_offset = pose[:3, :3]@self.anchor
            twist = jacobian@command.qd
            expected = reference(command.time_s, self.cfg)
            np.testing.assert_allclose(pose[:3, :3], target_rotation, atol=1e-10, rtol=0)
            np.testing.assert_allclose(pose[:3, 3]+anchor_offset, expected['anchor_position'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(twist[:3]+np.cross(twist[3:], anchor_offset), expected['anchor_velocity'], atol=1e-10, rtol=0)
        for previous, command, following in zip(commands, commands[1:], commands[2:]):
            np.testing.assert_allclose((following.q-previous.q)/(2*controller.dt), command.qd, atol=1e-6, rtol=0)
            np.testing.assert_allclose((following.qd-previous.qd)/(2*controller.dt), command.qdd, atol=1e-6, rtol=0)

    def test_clipping_partial_commit_and_failed_plan_are_replayable(self):
        controller = self.make_controller()
        plan = controller.plan(0., [1.25, -1.25], 16)
        np.testing.assert_array_equal(plan[0].clipped_action, [1., -1.])
        for command in plan[:7]:
            controller.commit(command)
        saved = controller.snapshot()
        with self.assertRaises(ValueError):
            controller.plan(saved['time_s'], [np.nan, 0.], 16)
        self.assertEqual(controller.snapshot(), saved)
        first = controller.plan(saved['time_s'], [-.4, .2], 16)
        for command in first:
            controller.commit(command)
        controller.restore(saved)
        second = controller.plan(saved['time_s'], [-.4, .2], 16)
        for a, b in zip(first, second):
            self.assertEqual(a.time_s, b.time_s)
            for name in ('q', 'qd', 'qdd', 'tilt', 'tilt_rate', 'rotation', 'filter_input_state', 'clipped_action'):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


if __name__ == '__main__':
    unittest.main()
