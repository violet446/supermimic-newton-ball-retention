"""Regression tests for conservative static certificates, not Newton physics."""
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from scipy.interpolate import CubicSpline
from ball_retention.clearance import separation_lower_bound,excluded_pairs
from ball_retention.robot import Robot
from check_paths import extrema,speed_bound

def box(center,half=.5):
    corners=np.array([[x,y,z] for x in (-half,half) for y in (-half,half) for z in (-half,half)])
    return {'vertices':corners+center,'normals':np.eye(3)}

class ClearanceTests(unittest.TestCase):
    def test_separated_boxes_exact_axis_gap(self):
        self.assertAlmostEqual(separation_lower_bound(box([0,0,0]),box([2,0,0])),1.)

    def test_enclosed_or_overlapping_shapes_not_certified(self):
        self.assertLessEqual(separation_lower_bound(box([0,0,0],1),box([0,0,0],.1)),0)
        self.assertLessEqual(separation_lower_bound(box([0,0,0]),box([.2,0,0])),0)

    def test_exclusions_do_not_skip_nonadjacent_wrist_or_base(self):
        self.assertNotIn(frozenset(('link5','shovel')),excluded_pairs())
        self.assertNotIn(frozenset(('base_link','link2')),excluded_pairs())
        self.assertIn(frozenset(('link6','shovel')),excluded_pairs())

    def test_extrema_include_interior_speed_peak(self):
        # Smoothstep q=3t^2-2t^3, max speed 1.5 at t=.5.
        curve=CubicSpline([0,1],np.array([[0]*6,[1]*6]),bc_type=((1,np.zeros(6)),(1,np.zeros(6))))
        low,high=extrema(curve,1)
        np.testing.assert_allclose(low,0,atol=1e-12)
        np.testing.assert_allclose(high,1.5,atol=1e-12)

    def test_common_ancestor_speed_bound_covers_relative_motion(self):
        robot=Robot(ROOT/'assets/ec66_description.urdf')
        q=np.array([.4,-.3,-1.5,2.3,-2.,3.]); qd=np.array([.2,-.1,.15,.3,-.25,.1]); h=1e-6
        point=np.array([.05,.02,.04]); row={'link':'link5','radius_m':np.linalg.norm(point)}
        positions=[]
        for sign in (-1,1):
            _,_,poses=robot.fk(q+sign*h*qd)
            local=np.linalg.inv(poses['link3'])@poses['link5']@np.r_[point,1]
            positions.append(local[:3])
        actual=np.linalg.norm((positions[1]-positions[0])/(2*h))
        bound=speed_bound(row,robot,abs(qd),shared=3)
        self.assertLessEqual(actual,bound+1e-8)

if __name__=='__main__': unittest.main()
