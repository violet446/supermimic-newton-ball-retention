import copy, json, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.spatial.transform import Rotation
from ball_retention.contracts import RigidState, State, Transition, validate_window
from ball_retention.task import reference,ball_in_tool,origin_velocity,tool_pose_for_anchor
from ball_retention.robot import Robot
from ball_retention.geometry import load_tool,support_point,sphere_clearance
from ball_retention.resources import admission

class StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((ROOT/'configs/s1_smoke.json').read_text())
        cls.robot=Robot(ROOT/'assets/ec66_description.urdf')

    def test_reference_endpoints_and_derivatives(self):
        for t in (0,1,3,4):
            r=reference(t,self.cfg)
            np.testing.assert_allclose(r['anchor_velocity'],0,atol=1e-12)
            np.testing.assert_allclose(r['anchor_acceleration'],0,atol=1e-12)
        t=1.63; h=1e-5
        before=reference(t-h,self.cfg); after=reference(t+h,self.cfg); at=reference(t,self.cfg)
        np.testing.assert_allclose((after['anchor_position']-before['anchor_position'])/(2*h),at['anchor_velocity'],atol=1e-9)
        np.testing.assert_allclose((after['anchor_velocity']-before['anchor_velocity'])/(2*h),at['anchor_acceleration'],atol=1e-9)

    def test_rotating_frame_stationary_attached_point(self):
        # A ball co-moving with rotating tool must have zero relative velocity.
        tool=RigidState([1,2,3],[0,0,0,1],[.1,.2,.3],[0,0,2])
        ball=RigidState([2,2,3],[0,0,0,1],[.1,2.2,.3],[0,0,2])
        np.testing.assert_allclose(ball_in_tool(ball,tool)['velocity'],0,atol=1e-12)

    def test_com_to_origin_velocity(self):
        np.testing.assert_allclose(origin_velocity([0,0,0],[0,0,2],[1,0,0],[0,0,0]),[0,2,0])

    def test_anchor_preserved_when_tilting(self):
        a=np.array([.32,0,.03]); target=np.array([.45,0,.35])
        R=Rotation.from_euler('xy',[.1,-.2]).as_matrix()
        T=tool_pose_for_anchor(target,a,R)
        np.testing.assert_allclose(T[:3,:3]@a+T[:3,3],target,atol=1e-12)

    def test_fk_jacobian_finite_difference(self):
        q=np.array([.4,-.8,1.,.2,.5,-.3]); T,J,_=self.robot.fk(q); h=1e-6
        for i in range(6):
            delta=np.eye(6)[i]*h
            plus,_,_=self.robot.fk(q+delta); minus,_,_=self.robot.fk(q-delta)
            np.testing.assert_allclose((plus[:3,3]-minus[:3,3])/(2*h),J[:3,i],atol=1e-8)
            w=Rotation.from_matrix(plus[:3,:3]@minus[:3,:3].T).as_rotvec()/(2*h)
            np.testing.assert_allclose(w,J[3:,i],atol=1e-8)

    def test_ik_recovers_known_fk_pose(self):
        q=np.array([.4,-.8,1.,.2,.5,-.3]); T,_,_=self.robot.fk(q)
        fit=self.robot.ik(T,q+.03)
        self.assertTrue(fit['pose_passed'])

    def test_actual_mesh_support_and_penetration_detection(self):
        mesh=load_tool(ROOT/'assets/shovel.STL'); p,n,_=support_point(mesh,[.32,0])
        self.assertAlmostEqual(sphere_clearance(mesh,p+n*.0202,.02),.0002,places=7)
        self.assertLess(sphere_clearance(mesh,p+n*.019,.02),-.0001)

    def test_ambiguous_stacked_surfaces_rejected(self):
        import trimesh
        low=trimesh.creation.box(extents=[1,1,.1]); high=low.copy(); high.apply_translation([0,0,1])
        with self.assertRaises(ValueError): support_point(trimesh.util.concatenate([low,high]),[.1,.2])

    def transition(self):
        body=RigidState([0,0,0],[0,0,0,1],[0,0,0],[0,0,0])
        s=State(0,body,body,[0]*6,[0]*6,[0]*2,[0]*2,[0]*2,0)
        ns=copy.deepcopy(s); ns.time_s=1/60
        return Transition('episode-0',0,s,[0,0],[[0]*6 for _ in range(8)],[[0]*6 for _ in range(8)],ns,False,False,None,None,{}, {})

    def test_reset_next_state_rejected(self):
        row=self.transition(); row.validate(1/480,8)
        row.next_state_before_reset.time_s=0
        with self.assertRaises(ValueError): row.validate(1/480,8)

    def test_partial_step_requires_physical_termination_in_episode_schema(self):
        from ball_retention.contracts import EPISODE_SCHEMA
        row = self.transition()
        row.executed_joint_q = row.executed_joint_q[:3]
        row.executed_joint_qd = row.executed_joint_qd[:3]
        row.next_state_before_reset.time_s = 3/480
        row.terminated = True
        with self.assertRaises(ValueError):
            row.validate(1/480, 8)
        row.schema_version = EPISODE_SCHEMA
        row.validate(1/480, 8)
        row.terminated, row.truncated = False, True
        with self.assertRaises(ValueError):
            row.validate(1/480, 8)

    def test_windows_reject_cross_episode_and_termination(self):
        a=self.transition(); b=copy.deepcopy(a); b.step_index=1
        b.state_t=copy.deepcopy(a.next_state_before_reset); b.next_state_before_reset.time_s=2/60
        validate_window([a,b],1/480,8)
        b.episode_id='episode-1'
        with self.assertRaises(ValueError): validate_window([a,b],1/480,8)
        b.episode_id=a.episode_id; a.terminated=True
        with self.assertRaises(ValueError): validate_window([a,b],1/480,8)

    def test_invalid_data_not_training_data(self):
        row=self.transition(); row.invalid_reason='constraint_violation'
        with self.assertRaises(ValueError): validate_window([row],1/480,8)

    def test_resource_gate_rejects_low_ram(self):
        sample={'available_ram_bytes':1024**3,'gpu_total_bytes':8*1024**3,'gpu_used_bytes':1024**3,'disk_free_bytes':20*1024**3}
        self.assertFalse(admission(sample,self.cfg['budget'])['passed'])

if __name__=='__main__': unittest.main()
