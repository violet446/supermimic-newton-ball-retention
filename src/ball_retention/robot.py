"""Independent EC66 URDF FK and initialization IK; not a differentiable controller.

Mount convention cross-checked against source manifest robot.py; no source module imported.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

def origin(node):
    T=np.eye(4); e=node.find('origin')
    if e is not None:
        T[:3,3]=np.fromstring(e.get('xyz','0 0 0'),sep=' ')
        T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(e.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return T

class Robot:
    def __init__(self,path):
        root=ET.parse(path).getroot(); self.joints=[]; self.low=[]; self.high=[]
        for j in root.findall('joint'):
            moving=j.get('type')=='revolute'
            if j.get('type') not in ('revolute','fixed'):
                raise ValueError('unsupported joint')
            axis=np.fromstring(j.find('axis').get('xyz'),sep=' ') if moving else None
            self.joints.append((j.find('parent').get('link'),j.find('child').get('link'),origin(j),axis))
            if moving:
                self.low.append(float(j.find('limit').get('lower')))
                self.high.append(float(j.find('limit').get('upper')))
        self.low=np.array(self.low); self.high=np.array(self.high)
        self.mount=Rotation.from_euler('x',90,degrees=True).as_matrix()@Rotation.from_euler('z',90,degrees=True).as_matrix()

    def fk(self,q):
        q=np.asarray(q)
        if q.shape!=(6,) or not np.isfinite(q).all(): raise ValueError('expected six finite joints')
        poses={'world':np.eye(4)}; k=0; axes=[]; centers=[]
        for parent,child,T,axis in self.joints:
            pose=poses[parent]@T
            if axis is not None:
                axes.append(pose[:3,:3]@axis); centers.append(pose[:3,3].copy())
                turn=np.eye(4); turn[:3,:3]=Rotation.from_rotvec(axis*q[k]).as_matrix()
                pose=pose@turn; k+=1
            poses[child]=pose
        tool=poses['flan'].copy(); tool[:3,:3]=tool[:3,:3]@self.mount
        J=np.vstack((np.cross(np.asarray(axes),tool[:3,3]-centers).T,np.asarray(axes).T))
        return tool,J,poses

    def ik(self,target,seed):
        def residual(q):
            T,_,_=self.fk(q)
            return np.r_[T[:3,3]-target[:3,3],.2*Rotation.from_matrix(target[:3,:3]@T[:3,:3].T).as_rotvec()]
        fit=least_squares(residual,np.clip(seed,self.low+1e-6,self.high-1e-6),bounds=(self.low+1e-6,self.high-1e-6),max_nfev=180,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        T,_,_=self.fk(fit.x)
        pe=float(np.linalg.norm(T[:3,3]-target[:3,3])); re=float(Rotation.from_matrix(target[:3,:3]@T[:3,:3].T).magnitude())
        return {'q':fit.x.tolist(),'position_error_m':pe,'rotation_error_rad':re,'pose_passed':pe<=2e-5 and re<=2e-4}
