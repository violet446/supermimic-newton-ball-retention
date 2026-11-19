"""Conservative separating-plane clearance certificates for full source meshes.

Positive bound proves separation. Nonpositive bound is inconclusive, NOT a collision.
Hull support extrema equal mesh support extrema; no convex hull replaces physics geometry.
"""
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from .robot import origin
from .geometry import load_tool

def geometry_rows(asset_root):
    rows=[]
    root=ET.parse(asset_root/'ec66_description.urdf').getroot()
    for link in root.findall('link'):
        for collision in link.findall('collision'):
            elem=collision.find('geometry/mesh')
            if elem is None: continue
            mesh=trimesh.load(asset_root/elem.get('filename'),force='mesh')
            mesh.apply_transform(origin(collision))
            rows.append((link.get('name'),link.get('name'),mesh))
    rows.append(('shovel','tool',load_tool(asset_root/'shovel.STL')))
    result=[]
    for name,link,mesh in rows:
        points=np.asarray(mesh.vertices); hull=ConvexHull(points)
        normals=hull.equations[:,:3]
        # Duplicate planes removed solely for speed, keeping more axes cannot reduce certificate.
        normals=np.unique(np.round(normals,7),axis=0)
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        result.append({'name':name,'link':link,'vertices':points[hull.vertices],
                       'normals':normals,'radius_m':float(np.linalg.norm(points,axis=1).max())})
    return result

def world_geometry(rows,poses,tool):
    result=[]
    for row in rows:
        T=tool if row['link']=='tool' else poses[row['link']]
        result.append({'name':row['name'],'vertices':row['vertices']@T[:3,:3].T+T[:3,3],
                       'normals':row['normals']@T[:3,:3].T})
    return result

def separating_plane(a,b):
    delta=b['vertices'].mean(0)-a['vertices'].mean(0)
    axes=np.vstack((a['normals'],b['normals'],np.eye(3)))
    if np.linalg.norm(delta)>1e-12: axes=np.vstack((axes,delta/np.linalg.norm(delta)))
    pa=a['vertices']@axes.T; pb=b['vertices']@axes.T
    gaps=np.maximum(pb.min(0)-pa.max(0),pa.min(0)-pb.max(0))
    best = np.argmax(gaps)
    return float(gaps[best]), axes[best]

def separation_lower_bound(a,b):
    return separating_plane(a,b)[0]

def excluded_pairs():
    # Only directly adjacent mesh links plus actual flange assembly attachment.
    chain=['base_link']+[f'link{i}' for i in range(1,7)]+['shovel']
    return {frozenset((a,b)) for a,b in zip(chain,chain[1:])}

def pose_clearance(rows,robot,q,floor_z=-.1):
    tool,_,poses=robot.fk(q); geometry=world_geometry(rows,poses,tool)
    excluded=excluded_pairs(); gaps={}
    for i,a in enumerate(geometry):
        for b in geometry[i+1:]:
            if frozenset((a['name'],b['name'])) in excluded: continue
            gaps[a['name']+' / '+b['name']]=separation_lower_bound(a,b)
    floor=min(float(g['vertices'][:,2].min()-floor_z) for g in geometry)
    return {'pair_lower_bounds_m':gaps,'floor_clearance_m':floor}
