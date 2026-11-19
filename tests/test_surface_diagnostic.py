from pathlib import Path
import sys
import unittest
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ball_retention.surface_diagnostic import SurfaceDiagnostic,sustained_crossing


class SurfaceDiagnosticTests(unittest.TestCase):
    def test_face_and_edge_gaps_have_different_meanings(self):
        mesh=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],process=False)
        measure=SurfaceDiagnostic(mesh,.02,np.eye(3)).measure
        face=measure([.3,.3,.025]);self.assertAlmostEqual(face['sphere_distance_gap_m'],.005)
        self.assertTrue(face['plane_projection_on_triangle'])
        edge=measure([.3,-.03,.04]);self.assertAlmostEqual(edge['sphere_distance_gap_m'],.03)
        self.assertAlmostEqual(edge['normal_plane_gap_m'],.02)
        self.assertFalse(edge['plane_projection_on_triangle'])
        self.assertFalse(measure([.3,.3,-.03])['outward_side'])

    def test_pruning_matches_exhaustive_triangle_distance(self):
        mesh=trimesh.creation.icosphere(subdivisions=2,radius=.15)
        index=SurfaceDiagnostic(mesh,.02,np.eye(3));rng=np.random.default_rng(15)
        for point in rng.normal(size=(12,3))*.2:
            closest=trimesh.triangles.closest_point(mesh.triangles,np.broadcast_to(point,(len(mesh.faces),3)))
            distance=np.linalg.norm(point-closest,axis=1).min()
            self.assertAlmostEqual(index.measure(point)['surface_distance_m'],distance,places=12)

    def test_crossing_requires_consecutive_steps_and_reports_first(self):
        self.assertEqual(sustained_crossing([0,2,0,2,3,4],1),4)
        self.assertIsNone(sustained_crossing([2,0,2,2],1))
