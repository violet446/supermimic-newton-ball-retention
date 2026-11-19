"""Analytic references and coordinate transforms. No simulator or future measurements."""
import numpy as np
from scipy.spatial.transform import Rotation

def reference(t, cfg):
    r = cfg['reference']; duration = r['duration_s']
    u = np.clip((t-r['start_s'])/duration, 0., 1.)
    s = 10*u**3-15*u**4+6*u**5
    sd = (30*u**2-60*u**3+30*u**4)/duration
    sdd = (60*u-180*u**2+120*u**3)/duration**2
    delta = np.asarray(r['displacement_world_m'])
    return {'phase':float(u), 'anchor_position':np.asarray(cfg['initial']['anchor_world_m'])+s*delta,
            'anchor_velocity':sd*delta, 'anchor_acceleration':sdd*delta}

def ball_in_tool(ball, tool):
    R = Rotation.from_quat(tool.quaternion_xyzw).as_matrix()
    p = R.T@(np.asarray(ball.position)-tool.position)
    omega = R.T@np.asarray(tool.angular_velocity)
    v = R.T@(np.asarray(ball.linear_velocity_origin)-tool.linear_velocity_origin)-np.cross(omega,p)
    return {'position':p, 'velocity':v, 'ball_omega':R.T@np.asarray(ball.angular_velocity),
            'gravity_direction':R.T@np.array([0.,0.,-1.])}

def origin_velocity(v_com, omega, origin_world, com_world):
    return np.asarray(v_com)+np.cross(omega,np.asarray(origin_world)-com_world)

def tool_pose_for_anchor(anchor_world, anchor_local, rotation):
    """Tilting about the material anchor requires a translated tool origin."""
    T = np.eye(4); T[:3,:3] = rotation
    T[:3,3] = np.asarray(anchor_world)-rotation@anchor_local
    return T
