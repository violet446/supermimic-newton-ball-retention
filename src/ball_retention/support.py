"""Positive geometric check for the carrying face of the pinned shovel mesh."""
import numpy as np
import trimesh

from .geometry import vertical_hits


class CarryingRegion:
    """Check that a sphere lies above and near the front bowl's upper face.

    The default lengthwise cut applies to assets/shovel.STL after load_tool's
    coordinate conversion. The carrier plane is fixed to the tool material
    frame, with its vertical aligned to world up at the reference orientation.
    """

    def __init__(self, mesh, carrier_rotation, radius, bowl_start_x_m=.22,
                 support_gap_m=.002, min_upward_normal=.2):
        self.rotation = np.asarray(carrier_rotation, dtype=float).copy()
        if (self.rotation.shape != (3, 3)
                or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.rotation), 1., atol=1e-6)):
            raise ValueError('invalid carrier rotation')
        if (not np.isfinite([radius, bowl_start_x_m, support_gap_m, min_upward_normal]).all()
                or radius <= 0 or not 0 < support_gap_m < radius
                or not 0 < min_upward_normal <= 1
                or not mesh.bounds[0, 0] < bowl_start_x_m < mesh.bounds[1, 0]):
            raise ValueError('invalid sphere or carrying region')
        self.radius = float(radius)
        self.bowl_start_x_m = float(bowl_start_x_m)
        self.support_gap_m = float(support_gap_m)
        self.min_upward_normal = float(min_upward_normal)
        self.carrier_mesh = mesh.copy()
        self.carrier_mesh.vertices = mesh.vertices @ self.rotation.T
        triangles = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.face_normals)
        front_upward = ((triangles[:, :, 0].max(axis=1) >= self.bowl_start_x_m)
                        & (normals @ self.rotation[2] >= self.min_upward_normal))
        self.triangles = triangles[front_upward]
        self.normals = normals[front_upward]
        if not len(self.triangles):
            raise ValueError('no upward triangles in carrying region')
        self.bounds_min = self.triangles.min(axis=1)
        self.bounds_max = self.triangles.max(axis=1)

    def measure(self, ball_local):
        ball = np.asarray(ball_local, dtype=float)
        if ball.shape != (3,) or not np.isfinite(ball).all():
            raise ValueError('invalid ball position')
        carrier_ball = self.rotation @ ball
        metrics = {
            'carrying_candidate': False, 'within_bowl': False,
            'reason': 'outside_tool_projection', 'surface_height_m': None,
            'bottom_clearance_m': None, 'bowl_margin_m': None,
            'nearest_support_clearance_m': None,
        }
        hits = vertical_hits(self.carrier_mesh, carrier_ball[:2])
        if not hits:
            return metrics
        highest = max(hits, key=lambda hit: (hit['z'], hit['normal'][2]))
        surface = np.array([*carrier_ball[:2], highest['z']])
        surface_local_x = float(self.rotation[:, 0] @ surface)
        bowl_margin = surface_local_x-self.bowl_start_x_m
        center_height = float(carrier_ball[2]-highest['z'])
        metrics.update(surface_height_m=highest['z'], bottom_clearance_m=center_height-self.radius,
                       bowl_margin_m=bowl_margin)
        if bowl_margin < 0:
            metrics['reason'] = 'handle_or_connector'
        elif center_height <= 0:
            metrics['reason'] = 'below_upper_face'
        elif highest['normal'][2] < self.min_upward_normal:
            metrics['reason'] = 'steep_or_back_face'
        else:
            metrics['within_bowl'] = True
            clearance = self._near_support_clearance(ball)
            metrics['nearest_support_clearance_m'] = clearance
            metrics['carrying_candidate'] = clearance is not None and abs(clearance) <= self.support_gap_m
            metrics['reason'] = 'above_near_bowl_face' if metrics['carrying_candidate'] else 'above_bowl_without_near_support'
        return metrics

    def _near_support_clearance(self, ball):
        # Bounding boxes only reject triangles beyond the sphere and gap.
        reach = self.radius+self.support_gap_m
        nearby = ((self.bounds_min <= ball+reach) & (self.bounds_max >= ball-reach)).all(axis=1)
        triangles = self.triangles[nearby]
        if not len(triangles):
            return None
        closest = trimesh.triangles.closest_point(triangles, np.broadcast_to(ball, (len(triangles), 3)))
        offsets = ball-closest
        on_front = closest[:, 0] >= self.bowl_start_x_m
        outward = np.einsum('ij,ij->i', offsets, self.normals[nearby]) > 0
        eligible = on_front & outward
        if not eligible.any():
            return None
        return float(np.linalg.norm(offsets[eligible], axis=1).min()-self.radius)
