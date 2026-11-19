"""Static local sphere support height, not a physics or training sample."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from diagnostics.resource_probe.probe import initial_geometry
from ball_retention.geometry import sphere_clearance

cfg=json.loads((ROOT/'configs/s1_contact_checks.json').read_text())
mesh,T,center=initial_geometry(ROOT,cfg); world=mesh.copy(); world.apply_transform(T)
def height(dx,dy):
    p=center.copy(); p[:2]+=[dx,dy]; lo=center[2]-.005; hi=center[2]+.005
    p[2]=lo
    if sphere_clearance(world,p,.02)>=0: raise ValueError('lower bracket not intersecting surface')
    for _ in range(30):
        mid=(lo+hi)/2; p[2]=mid
        if sphere_clearance(world,p,.02)>0: hi=mid
        else: lo=mid
    return (lo+hi)/2
offsets=[-.004,-.002,0,.002,.004]; heights=np.array([[height(x,y) for x in offsets] for y in offsets]); h=.002
H=np.array([[(heights[2,3]-2*heights[2,2]+heights[2,1])/h**2,(heights[3,3]-heights[3,1]-heights[1,3]+heights[1,1])/(4*h*h)],[(heights[3,3]-heights[3,1]-heights[1,3]+heights[1,1])/(4*h*h),(heights[3,2]-2*heights[2,2]+heights[1,2])/h**2]])
report={'offsets_world_xy_m':offsets,'center_heights_m_rows_y_columns_x':heights.tolist(),'relative_heights_m':(heights-heights[2,2]).tolist(),'finite_difference_hessian_per_m':H.tolist(),'hessian_eigenvalues':np.linalg.eigvalsh(H).tolist(),'caveat':'finite differences of piecewise mesh sphere support; no smoothness or global-stability proof','initial_center_world_m':center.tolist()}
out=ROOT/'runs/s1_equilibrium_20261112'; out.mkdir(exist_ok=True); (out/'support.json').write_text(json.dumps(report,indent=2))
for name in ('flat','offset','half_dt'):
    c=json.loads(json.dumps(cfg)); c['schema_version']='s1-equilibrium-'+name
    if name=='flat': c['diagnostic']={'geometry':'flat_patch'}
    if name=='offset': c['diagnostic']={'ball_center_world_m':[center[0],center[1]+.002,heights[3,2]+.0002]}
    if name=='half_dt': c['physics']['dt_s']/=2
    (out/(name+'_config.json')).write_text(json.dumps(c,indent=2))
print(json.dumps(report,indent=2))
