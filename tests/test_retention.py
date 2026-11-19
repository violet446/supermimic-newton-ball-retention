import sys
import unittest
from pathlib import Path
import numpy as np
import trimesh
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.retention import ToolEnvelope, DepartureMonitor


class DepartureTests(unittest.TestCase):
    def region(self):
        # L-shaped union has a missing upper-right square: hull/box would fill it.
        a = trimesh.creation.box(extents=[2, 1, .02]); a.apply_translation([0, -.5, 0])
        b = trimesh.creation.box(extents=[1, 1, .02]); b.apply_translation([-.5, .5, 0])
        return ToolEnvelope(trimesh.util.concatenate([a, b]), np.eye(3), .02)

    def test_concavity_and_radius_overlap(self):
        r = self.region()
        self.assertFalse(r.measure([.5, .5, .1])['center_inside_projection'])
        self.assertAlmostEqual(r.measure([.5, .5, .1])['outside_projection_distance_m'], .5)
        self.assertFalse(r.measure([.019, .5, .1])['outside_with_margin'])
        self.assertTrue(r.measure([.023, .5, .1])['outside_with_margin'])

    def test_airborne_not_loss_and_below_tool_is_loss(self):
        m = DepartureMonitor(self.region())
        self.assertFalse(m.update(0, [-.5, .5, .3])['terminated'])
        self.assertFalse(m.update(.1, [-.5, .5, .031])['terminated'])
        self.assertEqual(m.update(.2, [-.5, .5, -.1])['reason'], 'below_entire_tool')

    def test_grace_reentry_latch_and_snapshot(self):
        m = DepartureMonitor(self.region())
        self.assertFalse(m.update(0, [.1, .5, .1])['terminated'])
        self.assertFalse(m.update(.04, [-.1, .5, .1])['terminated'])
        m.update(.05, [.1, .5, .1]); saved = m.snapshot()
        self.assertFalse(m.update(.099, [.1, .5, .1])['terminated'])
        first = m.update(.1, [.1, .5, .1])
        self.assertEqual(first['reason'], 'outside_projection')
        self.assertTrue(m.update(.2, [-.1, .5, .1])['terminated'])
        m.restore(saved)
        self.assertEqual(first, m.update(.1, [.1, .5, .1]))
        with self.assertRaises(ValueError): m.update(.09, [-.1, .5, .1])


if __name__ == '__main__': unittest.main()
