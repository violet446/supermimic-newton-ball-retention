"""Conservative geometric departure monitor; no learned model or contact oracle.

Uses the union of actual projected mesh triangles, not a convex hull or box.
Remaining inside this projection is NOT proof of support or task success.
"""
import copy
import numpy as np


class ToolEnvelope:
    def __init__(self, mesh, carrier_rotation, radius, margin=.002):
        self.rotation = np.asarray(carrier_rotation, dtype=float)
        self.radius, self.margin = float(radius), float(margin)
        if self.rotation.shape != (3, 3) or not np.allclose(self.rotation.T@self.rotation, np.eye(3), atol=1e-6):
            raise ValueError('invalid carrier rotation')
        if self.radius <= 0 or self.margin < 0:
            raise ValueError('invalid sphere/edge margin')
        vertices = np.asarray(mesh.triangles)@self.rotation.T
        self.min_height = float(vertices[:, :, 2].min())
        triangles = vertices[:, :, :2]
        # Degenerate projected faces do not add area; their boundary is already
        # included by neighboring faces of the closed task mesh.
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        u, v = b-a, c-a
        det = u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
        self.triangles = triangles[np.abs(det) > 1e-14]
        self.a = self.triangles[:, 0]
        self.u, self.v = self.triangles[:, 1]-self.a, self.triangles[:, 2]-self.a
        self.det = self.u[:, 0]*self.v[:, 1]-self.u[:, 1]*self.v[:, 0]
        if not len(self.triangles):
            raise ValueError('empty projected region')
        self.edge_a = self.triangles.reshape(-1, 2)
        self.edge_b = self.triangles[:, [1, 2, 0]].reshape(-1, 2)
        self.edge_d = self.edge_b-self.edge_a
        self.edge_d2 = np.maximum(np.sum(self.edge_d**2, axis=1), 1e-30)

    def measure(self, ball_local):
        point = np.asarray(ball_local, dtype=float)@self.rotation.T
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('invalid ball position')
        delta = point[:2]-self.a
        b = (delta[:, 0]*self.v[:, 1]-delta[:, 1]*self.v[:, 0])/self.det
        c = (self.u[:, 0]*delta[:, 1]-self.u[:, 1]*delta[:, 0])/self.det
        inside = bool(np.any((b >= -1e-10) & (c >= -1e-10) & (b+c <= 1+1e-10)))
        distance = 0.
        if not inside:
            t = np.clip(np.sum((point[:2]-self.edge_a)*self.edge_d, axis=1)/self.edge_d2, 0, 1)
            distance = float(np.linalg.norm(self.edge_a+t[:, None]*self.edge_d-point[:2], axis=1).min())
        return {'center_inside_projection': inside, 'outside_projection_distance_m': distance,
            'sphere_projection_overlap_m': self.radius-distance,
            'outside_with_margin': distance > self.radius+self.margin,
            'below_entire_tool': bool(point[2]+self.radius < self.min_height-self.margin)}


class DepartureMonitor:
    def __init__(self, envelope, outside_grace_s=.05):
        if outside_grace_s < 0:
            raise ValueError('negative grace interval')
        self.envelope, self.grace_s = envelope, float(outside_grace_s)
        self.state = {'last_time_s': None, 'outside_since_s': None, 'terminated': False, 'reason': None}

    def update(self, time_s, ball_local, floor_reached=False):
        if not np.isfinite(time_s) or time_s < 0 or (self.state['last_time_s'] is not None and time_s <= self.state['last_time_s']):
            raise ValueError('time must strictly increase; restore monitor when replaying')
        metrics = self.envelope.measure(ball_local)
        if metrics['outside_with_margin']:
            if self.state['outside_since_s'] is None:
                self.state['outside_since_s'] = float(time_s)
        else:
            self.state['outside_since_s'] = None
        duration = 0 if self.state['outside_since_s'] is None else time_s-self.state['outside_since_s']
        reason = ('floor' if floor_reached else 'below_entire_tool' if metrics['below_entire_tool'] else
                  'outside_projection' if metrics['outside_with_margin'] and duration+1e-12 >= self.grace_s else None)
        if reason and not self.state['terminated']:
            self.state.update(terminated=True, reason=reason)
        self.state['last_time_s'] = float(time_s)
        return {**metrics, 'outside_duration_s': duration, 'terminated': self.state['terminated'], 'reason': self.state['reason']}

    def snapshot(self):
        return copy.deepcopy(self.state)

    def restore(self, state):
        self.state = copy.deepcopy(state)
