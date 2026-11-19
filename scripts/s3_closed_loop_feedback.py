"""Low-amplitude ball position and velocity feedback for the fixed Newton route."""
import numpy as np

from ball_retention.model_data import _rotation
from ball_retention.task import ball_in_tool


def _parameters(options):
    target = np.asarray(options['target_xy_m'], dtype=float)
    kp = float(options['position_gain_per_m'])
    kd = float(options['velocity_gain_s_per_m'])
    cap = float(options['correction_cap'])
    if (target.shape != (2,) or not np.isfinite(target).all()
            or not np.isfinite([kp, kd, cap]).all()
            or not (kp > 0. and kd > 0. and 0. < cap <= 1.)):
        raise ValueError('invalid position and velocity feedback options')
    return target, kp, kd, cap


def _action(position_xy, velocity_xy, parameters):
    target, kp, kd, cap = parameters
    position = np.asarray(position_xy, dtype=float)
    velocity = np.asarray(velocity_xy, dtype=float)
    if (position.shape != (2,) or velocity.shape != (2,)
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()):
        raise ValueError('nonfinite or malformed feedback observation')
    with np.errstate(over='raise', invalid='raise'):
        error_x, error_y = kp*(position-target)+kd*velocity
    # With the pinned orientation, positive world roll projects gravity toward
    # negative tool y. Positive world pitch projects it toward positive tool x.
    return np.clip([error_y, -error_x], -cap, cap).tolist()


def feedback_action(state, options):
    """Read only the current observation and return bounded [roll, pitch]."""
    obs = ball_in_tool(state.ball, state.tool)
    return _action(obs['position'][:2], obs['velocity'][:2], _parameters(options))


def preflight(base_rotation, options):
    """Check static gravity signs and action bounds without stepping Newton."""
    rotation = np.asarray(base_rotation, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), rtol=0., atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-6)):
        raise ValueError('invalid initial tool orientation')
    parameters = _parameters(options)
    target, kp, kd, cap = parameters
    zero = np.zeros(2)
    np.testing.assert_array_equal(_action(target, zero, parameters), zero)
    gravity = np.array([0., 0., -1.])
    epsilon = 1e-6
    gravity_jacobian = np.column_stack([
        ((_rotation(step, rotation)[0].T-_rotation(-step, rotation)[0].T)@gravity)[:2]/(2.*epsilon)
        for step in np.eye(2)*epsilon
    ])
    trials = []
    directions = np.vstack((np.eye(2), -np.eye(2)))
    for direction in directions:
        for kind, gain in (('position', kp), ('velocity', kd)):
            magnitude = min(1e-3, .25*cap/gain)
            deviation = magnitude*direction
            position, velocity = (target+deviation, zero) if kind == 'position' else (target, deviation)
            request = np.asarray(_action(position, velocity, parameters))
            gravity_change = gravity_jacobian@request
            if gravity_change@direction >= 0. or np.max(abs(request)) >= cap:
                raise ValueError(f'{kind} action did not oppose the measured direction')
            trials.append({'kind':kind, 'direction_xy':direction.tolist(), 'request_roll_pitch':request.tolist(),
                           'gravity_projection_change_per_g_xy':gravity_change.tolist()})
    limit_trials = []
    for direction in ([1.,1.], [1.,-1.], [-1.,1.], [-1.,-1.]):
        displacement = (2.*cap/kp)*np.asarray(direction)
        request = _action(target+displacement, zero, parameters)
        np.testing.assert_array_equal(request, cap*np.array([direction[1], -direction[0]]))
        limit_trials.append(request)
    return {'zero_at_target':True, 'gravity_jacobian_per_radian':gravity_jacobian.tolist(),
            'position_and_velocity_trials':trials, 'saturation_trials':limit_trials,
            'correction_cap':cap, 'simulation_steps':0}
