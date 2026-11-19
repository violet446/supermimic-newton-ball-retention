"""Exact nearest-triangle geometry for offline diagnosis, not contact labels."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree


class SurfaceDiagnostic:
    def __init__(self,mesh,radius,carrier_rotation):
        self.triangles=np.asarray(mesh.triangles).copy()
        self.normals=np.asarray(mesh.face_normals).copy()
        self.low=self.triangles.min(1);self.high=self.triangles.max(1)
        self.vertices=cKDTree(np.asarray(mesh.vertices))
        self.radius=float(radius);self.rotation=np.asarray(carrier_rotation)

    def measure(self,point):
        p=np.asarray(point,dtype=float)
        if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('invalid point')
        # Nearest vertex provides a valid distance upper bound. AABB distance
        # then excludes triangles that cannot contain a nearer surface point.
        upper=float(self.vertices.query(p)[0])
        delta=np.maximum(np.maximum(self.low-p,p-self.high),0)
        ids=np.flatnonzero(np.sum(delta*delta,axis=1)<=(upper+1e-12)**2)
        closest=trimesh.triangles.closest_point(self.triangles[ids],np.broadcast_to(p,(len(ids),3)))
        offsets=p-closest;distances=np.linalg.norm(offsets,axis=1)
        j=int(np.argmin(distances));face=int(ids[j]);normal=self.normals[face]
        projection=float(offsets[j]@normal)
        return {'triangle':face,'closest_point_tool_m':closest[j].tolist(),'normal_tool':normal.tolist(),
            'surface_distance_m':float(distances[j]),'sphere_distance_gap_m':float(distances[j]-self.radius),
            'normal_plane_gap_m':projection-self.radius,
            'plane_projection_on_triangle':bool(np.linalg.norm(offsets[j]-projection*normal)<1e-9),
            'outward_side':bool(projection>0),
            'upward_bowl_face':bool(closest[j,0]>=.22 and normal@self.rotation[2]>=.2)}


def sustained_crossing(values,threshold,count=3):
    """Return the first 1-based lead starting count consecutive exceedances."""
    above=np.asarray(values)>threshold
    for i in range(len(above)-count+1):
        if above[i:i+count].all():return i+1
    return None
