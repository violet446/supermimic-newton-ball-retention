"""Tool-frame trapezoid integration with learned world-frame velocity increments."""
import numpy as np

from .action_grad import _known_jacobian
from .frame_model import cross_matrix
from .model_data import KNOWN_NAMES, _rotation, filter_advance, known_rollout
from .task import reference
from .world_model import STATE_DIM, ResidualMLP, _array


FEATURES = len(KNOWN_NAMES)
PACKED = FEATURES+18+3+9


def _matvec(matrix, vector):
    return (matrix@vector[..., None])[..., 0]


def _endpoints(rows, cfg, dt, anchor, sign, tangent=None):
    """Recover analytic orientation and tool-origin twist from midpoint features."""
    action = cfg['action']
    tau = action['tilt_filter_time_constant_s']
    limit = np.deg2rad(action['tilt_limit_deg'])
    base = np.asarray(cfg['initial']['tool_rotation_world'])
    tilt, rate, _ = filter_advance(rows[:, 4:6], rows[:, 6:8], limit*rows[:, 2:4], tau, sign*dt/2)
    r, axis = _rotation(tilt, base)
    y = np.array([0., 1., 0.])
    omega = rate[:, :1]*axis+rate[:, 1:]*y
    offset = r@anchor
    local_w = _matvec(r.swapaxes(-1, -2), omega)
    reference_v = np.array([reference(max(0., t+sign*dt/2), cfg)['anchor_velocity'] for t in rows[:, 0]])
    v = reference_v-np.cross(omega, offset)
    if tangent is None:
        return (r, local_w, v), None
    da, dw, _ = filter_advance(tangent[:, 4:6], tangent[:, 6:8], limit*tangent[:, 2:4], tau, sign*dt/2)
    dr = cross_matrix(da[:, :1]*axis+da[:, 1:]*y)@r
    domega = dw[:, :1]*axis+dw[:, 1:]*y+rate[:, :1]*da[:, 1:]*np.cross(y, axis)
    dlocal = _matvec(dr.swapaxes(-1,-2), omega)+_matvec(r.swapaxes(-1,-2), domega)
    dv = -np.cross(domega, offset)-np.cross(omega, dr@anchor)
    return (r, local_w, v), (dr, dlocal, dv)


def _coefficients(rows, cfg, dt, anchor, center, tangent=None):
    (r0, w0, v0), d0 = _endpoints(rows, cfg, dt, anchor, -1, tangent)
    (r1, w1, v1), d1 = _endpoints(rows, cfg, dt, anchor, 1, tangent)
    h = .5*dt
    d, c = cross_matrix(w1), cross_matrix(w0)
    system = np.eye(3)+h*d
    world = np.linalg.solve(system, r1.swapaxes(-1, -2))
    lever = np.linalg.solve(system, d)
    local = world@r0
    shift = v0-v1+np.array([0.,0.,-cfg['physics']['gravity_m_s2']*dt])+center
    if tangent is None:
        return local@c-lever, local-h*lever, _matvec(world, shift), world
    dr0, dw0, dv0 = d0
    dr1, dw1, dv1 = d1
    dd = cross_matrix(dw1)
    d_world = np.linalg.solve(system, dr1.swapaxes(-1,-2)-h*dd@world)
    d_lever = np.linalg.solve(system, dd-h*dd@lever)
    d_local = d_world@r0+world@dr0
    return (d_local@c+local@cross_matrix(dw0)-d_lever, d_local-h*d_lever,
            _matvec(d_world, shift)+_matvec(world, dv0-dv1), d_world)


def _pack(rows, network, cfg, dt, anchor, center, tangent=None):
    shape = rows.shape[:-1]
    x = rows.reshape(-1, FEATURES)
    p, u, shift, world = _coefficients(x, cfg, dt, anchor, center, tangent)
    affine = np.concatenate((p, u), axis=-1).reshape(*shape, 18)
    return np.concatenate((network, affine, shift.reshape(*shape,3), world.reshape(*shape,9)), axis=-1)


def pack_world_features(known, cfg, dt, anchor, center, mean, scale):
    """Keep normalized midpoint network features and append raw deterministic coefficients."""
    rows = np.asarray(known, dtype=float)
    if rows.ndim < 1 or rows.shape[-1] != FEATURES or not np.isfinite(rows).all():
        raise ValueError('expected finite causal midpoint features')
    mean = _array(mean, (FEATURES,), 'known mean')
    scale = _array(scale, (FEATURES,), 'known scale')
    if np.any(scale <= 0):
        raise ValueError('known scale must be positive')
    return _pack(rows, (rows-mean)/scale, cfg, dt, _array(anchor,(3,),'anchor'), _array(center,(3,),'center'))


class WorldIncrementMLP(ResidualMLP):
    """Same network as the reference, with velocity residuals in world SI axes."""

    MODEL_TYPE = 'world_increment_trapezoid_v1'

    def __init__(self, known_dim, state_mean, state_scale, dt_s, seed, *, inc_center, hidden=32, residual_gain=.1):
        if known_dim != PACKED:
            raise ValueError('expected packed world increment features')
        super().__init__(FEATURES, state_mean, state_scale, dt_s, seed, hidden=hidden, residual_gain=residual_gain)
        self.known_dim = PACKED
        self.inc_center = _array(inc_center, (3,), 'world increment center').copy()
        self.inc_center.setflags(write=False)

    def features(self, known, cfg, anchor, mean, scale):
        return pack_world_features(known, cfg, self.dt_s, anchor, self.inc_center, mean, scale)

    def _forward(self, state, known):
        neural, cache = super()._forward(state, known[..., :FEATURES])
        raw = self.denormalize(state)
        p, u = raw[..., :3], raw[..., 3:6]
        matrix = known[..., FEATURES:FEATURES+18].reshape(*state.shape[:-1],3,6)
        world = known[..., -9:].reshape(*state.shape[:-1],3,3)
        residual = (neural[..., 3:6]-state[..., 3:6])*self.state_scale[3:6]
        next_u = _matvec(matrix, raw[..., :6])+known[..., FEATURES+18:FEATURES+21]+_matvec(world, residual)
        next_p = p+.5*self.dt_s*(u+next_u)
        neural[..., :3] = (next_p-self.state_mean[:3])/self.state_scale[:3]
        neural[..., 3:6] = (next_u-self.state_mean[3:6])/self.state_scale[3:6]
        return neural, (cache, raw[..., :6], residual, matrix, world)

    def _backward(self, cache, cotangent):
        net, raw, residual, matrix, world = cache
        h = .5*self.dt_s
        p_grad = cotangent[..., :3]/self.state_scale[:3]
        u_grad = cotangent[..., 3:6]/self.state_scale[3:6]+h*p_grad
        neural = np.zeros_like(cotangent)
        neural[..., 3:6] = _matvec(world.swapaxes(-1,-2), u_grad)*self.state_scale[3:6]
        neural[..., 6:] = cotangent[..., 6:]
        state, inputs, params = super()._backward(net, neural)
        state[..., 3:6] -= neural[..., 3:6]
        state[..., :6] += (_matvec(matrix.swapaxes(-1,-2), u_grad)
                           +np.concatenate((p_grad,h*p_grad),axis=-1))*self.state_scale[:6]
        affine_grad = (u_grad[..., :, None]*raw[..., None, :]).reshape(*raw.shape[:-1],18)
        world_grad = (u_grad[..., :, None]*residual[..., None, :]).reshape(*raw.shape[:-1],9)
        return state, np.concatenate((inputs,affine_grad,u_grad,world_grad),axis=-1), params

    def state_dict(self):
        state = super().state_dict()
        state['world_increment_residual_gain'] = state.pop('residual_gain')
        return {**state, 'model_type':np.array(self.MODEL_TYPE), 'world_increment_center_m_s':self.inc_center.copy()}

    @classmethod
    def from_state_dict(cls, state):
        if np.asarray(state.get('model_type')).item() != cls.MODEL_TYPE:
            raise ValueError('expected a world increment checkpoint')
        model = cls(int(state['known_dim']), state['state_mean'], state['state_scale'], float(state['dt_s']), 0,
                    inc_center=state['world_increment_center_m_s'], hidden=int(state['hidden']),
                    residual_gain=state['world_increment_residual_gain'])
        for name, param in model.params.items():
            param[...] = _array(state[name],param.shape,name)
        return model


def world_action_jvp(model, state, time, tilt, rate, actions, direction, cfg, anchor, mean, scale):
    """Roll out and differentiate along one full (H,2) request direction."""
    actions = np.asarray(actions, dtype=float)
    direction = _array(direction, actions.shape, 'request direction')
    raw = known_rollout(time, tilt, rate, actions, cfg, model.dt_s)
    feature_tangent = np.zeros_like(raw)
    jumps = np.diff(direction,axis=0,prepend=0.)
    for i in np.flatnonzero(np.any(jumps,axis=1)):
        feature_tangent[i:] += _known_jacobian(raw[i:],actions[i:],cfg,model.dt_s)@jumps[i]
    packed = model.features(raw,cfg,anchor,mean,scale)
    tangent = _pack(raw,feature_tangent/scale,cfg,model.dt_s,anchor,model.inc_center,feature_tangent)
    initial = model.normalize(state)
    trajectory = model.rollout(initial[None],packed[None])[0]
    jvp = np.empty_like(trajectory)
    state_tangent = np.zeros(STATE_DIM)
    previous = initial
    outputs = np.eye(STATE_DIM)
    for i, features in enumerate(packed):
        ds, dk, _ = model.step_vjp(np.broadcast_to(previous,outputs.shape),
                                  np.broadcast_to(features,(STATE_DIM,PACKED)), outputs)
        state_tangent = ds@state_tangent+dk@tangent[i]
        jvp[i] = state_tangent*model.state_scale
        previous = trajectory[i]
    return model.denormalize(trajectory), jvp
