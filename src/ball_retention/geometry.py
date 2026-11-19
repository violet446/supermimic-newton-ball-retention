"""Local mesh geometry. No old loading data, FCL, viewer, or Newton imports."""
import numpy as np
import trimesh

def load_tool(path):
    mesh = trimesh.load(path, force='mesh')
    mesh.vertices = ((np.asarray(mesh.vertices)-[138.,30.,0.])*.001)[:,[2,0,1]]
    return mesh

def vertical_hits(mesh, xy):
    tri = np.asarray(mesh.triangles)
    a = tri[:,0]; u=tri[:,1]-a; v=tri[:,2]-a
    det = u[:,0]*v[:,1]-u[:,1]*v[:,0]
    valid = np.abs(det)>1e-14
    dx=xy[0]-a[:,0]; dy=xy[1]-a[:,1]
    b=np.zeros(len(tri)); c=b.copy()
    b[valid]=(dx[valid]*v[valid,1]-dy[valid]*v[valid,0])/det[valid]
    c[valid]=(u[valid,0]*dy[valid]-u[valid,1]*dx[valid])/det[valid]
    ids=np.flatnonzero(valid & (b>=-1e-9) & (c>=-1e-9) & (b+c<=1+1e-9))
    return [{'triangle':int(i),'z':float(a[i,2]+b[i]*u[i,2]+c[i]*v[i,2]),
             'normal':mesh.face_normals[i].tolist()} for i in ids]

def support_point(mesh, xy, min_normal_z=.5):
    hits=vertical_hits(mesh,xy)
    upward=[h for h in hits if h['normal'][2]>=min_normal_z]
    levels=[]
    for hit in sorted(upward,key=lambda h:h['z']):
        if not levels or abs(hit['z']-levels[-1]['z'])>1e-6:
            levels.append(hit)
    if len(levels)!=1:
        raise ValueError(f'ambiguous support: {len(levels)} upward surface levels')
    h=levels[0]
    if any(x['z']>h['z']+1e-6 for x in hits):
        raise ValueError('support is occluded by higher geometry')
    return np.array([*xy,h['z']]), np.array(h['normal']), hits

def sphere_clearance(mesh, center, radius):
    # One point against all triangles: bounded O(triangle count), no rtree dependency.
    points=np.broadcast_to(center,(len(mesh.faces),3))
    closest=trimesh.triangles.closest_point(mesh.triangles,points)
    return float(np.linalg.norm(closest-center,axis=1).min()-radius)
