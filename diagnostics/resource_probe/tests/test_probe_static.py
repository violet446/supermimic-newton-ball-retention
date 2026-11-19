"""Pure geometry and resource refusal tests; no Newton or Warp imports."""
from pathlib import Path
import json,sys,unittest
import tempfile
ROOT=Path(__file__).resolve().parents[3]; sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from scipy.spatial.transform import Rotation
from diagnostics.resource_probe.probe import initial_geometry,diagnostic_metrics,FixedToolProbe
from ball_retention.guard import stop_reason
from diagnostics.resource_probe.runner import monitored_pid,storage_admission

class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((ROOT/'configs/s1_reachable_candidate.json').read_text())
        cls.mesh,cls.tool,cls.center=initial_geometry(ROOT,cls.cfg)

    def states(self):
        q=np.array([np.r_[self.tool[:3,3],Rotation.from_matrix(self.tool[:3,:3]).as_quat()],np.r_[self.center,0,0,0,1]])
        return q,np.zeros((2,6))

    def test_initial_world_transform_preserves_clearance(self):
        q,qd=self.states(); result=diagnostic_metrics(q,qd,1,0,self.mesh,self.cfg)
        self.assertAlmostEqual(result['surface_clearance_m'],.0002,places=7)
        self.assertFalse(result['floor_reached'])

    def test_prescribed_tool_origin_velocity_matches_pose_derivative(self):
        from types import SimpleNamespace
        class Array:
            def __init__(self,x): self.x=x.copy()
            def numpy(self): return self.x.copy()
            def assign(self,x): self.x=x.copy()
        q,qd=self.states(); probe=FixedToolProbe.__new__(FixedToolProbe)
        probe.base_tool=self.tool; probe.cfg=self.cfg; probe.tool_id=0
        probe.a=SimpleNamespace(body_q=Array(q),body_qd=Array(qd))
        t=1.4; h=1e-5
        probe.prescribe_tilt(t-h,1); before=probe.a.body_q.numpy()[0]
        probe.prescribe_tilt(t+h,1); after=probe.a.body_q.numpy()[0]
        probe.prescribe_tilt(t,1); pose=probe.a.body_q.numpy()[0]; velocity=probe.a.body_qd.numpy()[0]
        np.testing.assert_allclose((after[:3]-before[:3])/(2*h),velocity[:3],atol=1e-9)
        angular=Rotation.from_matrix(Rotation.from_quat(after[3:]).as_matrix()@Rotation.from_quat(before[3:]).as_matrix().T).as_rotvec()/(2*h)
        np.testing.assert_allclose(angular,velocity[3:],atol=1e-9)
        anchor=np.asarray(self.cfg['initial']['anchor_world_m']); local=self.tool[:3,:3].T@(anchor-self.tool[:3,3])
        np.testing.assert_allclose(Rotation.from_quat(pose[3:]).apply(local)+pose[:3],anchor,atol=1e-12)

    def test_corrupt_state_rejected(self):
        q,qd=self.states(); q[1,0]=np.nan
        with self.assertRaises(FloatingPointError): diagnostic_metrics(q,qd,1,0,self.mesh,self.cfg)
        q,qd=self.states(); q[1,3:]*=2
        with self.assertRaises(FloatingPointError): diagnostic_metrics(q,qd,1,0,self.mesh,self.cfg)

    def test_resource_stop_boundaries(self):
        sample={'gpu_used_bytes':1024**3,'disk_free_bytes':20*1024**3}; budget=self.cfg['budget']
        self.assertIsNone(stop_reason(sample,budget,1,100,100,4.9))
        self.assertEqual(stop_reason(sample,budget,1,100,100,5),'system_available_ram_below_2gib_for_5s')
        self.assertEqual(stop_reason(sample,budget,1800,100,100,0),'wall_budget')
        self.assertEqual(stop_reason(sample,budget,1,4*1024**3,100,0),'process_memory_budget')
        self.assertEqual(stop_reason(sample,budget,1,100,2*1024**3,0),'output_cache_budget')

    def test_static_imports_do_not_load_physics(self):
        self.assertNotIn('newton',sys.modules)
        self.assertNotIn('warp',sys.modules)

    def test_windows_launcher_pid_is_replaced_by_worker_handshake(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            path=Path(directory)/'worker_result.json'
            self.assertEqual(monitored_pid(path,123),123)
            path.write_text(json.dumps({'worker_pid':456}))
            self.assertEqual(monitored_pid(path,123),456)

    def test_probe_budget_has_separate_emergency_and_commit_limits(self):
        cfg=json.loads((ROOT/'diagnostics/resource_probe/config.json').read_text()); b=cfg['budget']
        sample={'available_ram_bytes':.1*1024**3,'gpu_used_bytes':1024**3,'disk_free_bytes':20*1024**3}
        self.assertEqual(stop_reason(sample,b,1,100,100,0),'critical_system_ram')
        sample['available_ram_bytes']=.5*1024**3
        self.assertEqual(stop_reason(sample,b,1,100,100,0,3*1024**3),'process_commit_budget')
        self.assertEqual(stop_reason(sample,b,1,100,100,2),'system_available_ram_floor')

    def test_existing_cache_is_checked_before_starting_worker(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            root=Path(directory);(root/'.cache').mkdir();(root/'out').mkdir()
            (root/'.cache'/'cached.bin').write_bytes(b'x'*100)
            (root/'out'/'result.json').write_bytes(b'x'*20)
            budget={'max_output_cache_gib':110/1024**3}
            result=storage_admission(root,root/'out',budget)
            self.assertFalse(result['passed']);self.assertEqual(result['total_bytes'],120)
            self.assertEqual(result['cache_bytes'],100)
            budget['max_output_cache_gib']=120/1024**3
            self.assertTrue(storage_admission(root,root/'out',budget)['passed'])

if __name__=='__main__': unittest.main()
