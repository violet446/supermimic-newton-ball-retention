"""Exact shared-action sensitivities including the affine moving-frame path."""
import numpy as np

from .action_grad import _known_jacobian
from .frame_model import PACKED_DIM, cross_matrix, pack_features
from .model_data import _rotation, filter_advance, known_rollout
from .task import reference


def _affine_tangent(known, tangent, config, dt, anchor):
    """Directional derivative of frame_affine for action-only perturbations."""
    if np.any(tangent[:, :2]):
        raise ValueError('time/phase derivatives are not shared-action derivatives')
    tau = config['action']['tilt_filter_time_constant_s']
    limit = np.deg2rad(config['action']['tilt_limit_deg'])
    axis_y = np.array([0., 1., 0.])
    endpoints = []
    for sign in (-1, 1):
        tilt, rate, _ = filter_advance(known[:,4:6],known[:,6:8],limit*known[:,2:4],tau,sign*dt/2)
        dtilt, drate, _ = filter_advance(tangent[:,4:6],tangent[:,6:8],limit*tangent[:,2:4],tau,sign*dt/2)
        r, axis = _rotation(tilt,np.asarray(config['initial']['tool_rotation_world']))
        delta = dtilt[:,:1]*axis+dtilt[:,1:]*axis_y
        dr = cross_matrix(delta)@r
        omega = rate[:,:1]*axis+rate[:,1:]*axis_y
        daxis = dtilt[:,1:]*np.cross(axis_y,axis)
        domega = drate[:,:1]*axis+rate[:,:1]*daxis+drate[:,1:]*axis_y
        w = np.einsum('nji,nj->ni',r,omega)
        dw = np.einsum('nji,nj->ni',dr,omega)+np.einsum('nji,nj->ni',r,domega)
        ref = [reference(max(0., t+sign*dt/2),config) for t in known[:,0]]
        p, v = (np.array([row[key] for row in ref]) for key in ('anchor_position','anchor_velocity'))
        endpoints.append((r,dr,w,dw,p,v))
    r0, dr0, w0, dw0, p0, v0 = endpoints[0]
    r1, dr1, w1, dw1, p1, v1 = endpoints[1]
    into, dinto = r1.swapaxes(-1,-2),dr1.swapaxes(-1,-2)
    rot, drot = into@r0,dinto@r0+into@dr0
    c, dc = rot@cross_matrix(w0),drot@cross_matrix(w0)+rot@cross_matrix(dw0)
    d, dd = cross_matrix(w1),cross_matrix(dw1)
    pp, dpp = rot+dt*c,drot+dt*dc
    pv, dpv = dt*rot,dt*drot
    dvp = dc-dd@pp-d@dpp
    dvv = drot-dd@pv-d@dpv
    world_shift = p0+dt*v0-p1
    shift = np.einsum('nij,nj->ni',into,world_shift)
    dshift = np.einsum('nij,nj->ni',dinto,world_shift)
    dvelocity = np.einsum('nij,nj->ni',dinto,v0-v1)
    da, db = np.zeros((len(known),9,9)),np.zeros((len(known),9))
    da[:,:3,:3], da[:,:3,3:6] = dpp,dpv
    da[:,3:6,:3], da[:,3:6,3:6],da[:,6:,6:] = dvp,dvv,drot
    db[:,:3] = dshift-dpp@anchor
    db[:,3:6] = dvelocity-np.einsum('nij,nj->ni',dd,shift)-np.einsum('nij,nj->ni',d,dshift)-dvp@anchor
    return da,db


def frame_action_sensitivity(model,state_raw,time_s,tilt_rad,tilt_rate_rad_s,actions,
                             config,known_mean,known_scale,anchor_local):
    """Differentiate every rollout state w.r.t. a held normalized action offset.

    Both the neural features and the deterministic affine map are differentiated.
    No finite differences, simulator calls, or future measured states are used.
    """
    if model.known_dim != PACKED_DIM:
        raise ValueError('expected a moving-frame residual model')
    anchor = np.asarray(anchor_local)
    known = known_rollout(time_s,tilt_rad,tilt_rate_rad_s,actions,config,model.dt_s)
    if known.ndim != 2:
        raise ValueError('expected one action sequence')
    dk = _known_jacobian(known,actions,config,model.dt_s)
    packed = pack_features(known,config,model.dt_s,anchor,model.state_mean,model.state_scale,known_mean,known_scale)
    jac = np.empty((*packed.shape,2))
    for axis in (0,1):
        da,db = _affine_tangent(known,dk[:,:,axis],config,model.dt_s,anchor)
        daz = da*model.state_scale[None,None,:]/model.state_scale[None,:,None]
        dbz = (da@model.state_mean+db)/model.state_scale
        jac[:,:,axis] = np.concatenate((dk[:,:,axis]/known_scale,daz.reshape(-1,81),dbz),axis=-1)
    initial = model.normalize(np.asarray(state_raw))
    prediction = model.rollout(initial[None],packed[None])[0]
    result = np.empty((*prediction.shape,2))
    state_jac = np.zeros((9,2))
    previous = initial
    for step,feature in enumerate(packed):
        ds,df,_ = model.step_vjp(np.broadcast_to(previous,(9,9)),np.broadcast_to(feature,(9,PACKED_DIM)),np.eye(9))
        state_jac = ds@state_jac+df@jac[step]
        result[step] = model.state_scale[:,None]*state_jac
        previous = prediction[step]
    return {'prediction':model.denormalize(prediction),'jacobian':result}
