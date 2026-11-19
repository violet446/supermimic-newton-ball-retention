"""Action sensitivities of the learned ball dynamics and analytic tool motion."""
import numpy as np

from .model_data import KNOWN_NAMES, _rotation, filter_advance, known_rollout
from .world_model import STATE_DIM


def _inputs(model, state_raw, time_s, tilt_rad, tilt_rate_rad_s, actions, config,
            known_mean, known_scale):
    state = np.asarray(state_raw, dtype=float)
    actions = np.asarray(actions, dtype=float)
    mean, scale = (np.asarray(x, dtype=float) for x in (known_mean, known_scale))
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError('expected one finite nine-dimensional ball state')
    if actions.ndim != 2 or actions.shape[1] != 2 or not len(actions):
        raise ValueError('expected one action sequence with shape (H,2)')
    if (model.known_dim != len(KNOWN_NAMES) or mean.shape != (model.known_dim,)
            or scale.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError('invalid tool feature normalization')
    known = known_rollout(time_s, tilt_rad, tilt_rate_rad_s, actions, config, model.dt_s)
    return model.normalize(state), known, (known-mean)/scale


def predict_ball(model, state_raw, time_s, tilt_rad, tilt_rate_rad_s, actions, config,
                 known_mean, known_scale):
    """Predict the raw SI state at each lead for a single action sequence."""
    initial, _, normalized = _inputs(model, state_raw, time_s, tilt_rad, tilt_rate_rad_s,
                                     actions, config, known_mean, known_scale)
    return model.denormalize(model.rollout(initial[None], normalized[None])[0])


def _known_jacobian(known, actions, config, dt_s):
    """Differentiate known_rollout for an additive offset shared by all requests."""
    actions = np.asarray(actions, dtype=float)
    if np.any(abs(actions) == 1):
        raise ValueError('action derivative is undefined at the clipping boundary')
    tau = float(config['action']['tilt_filter_time_constant_s'])
    limit = np.deg2rad(config['action']['tilt_limit_deg'])
    base_rotation = np.asarray(config['initial']['tool_rotation_world'], dtype=float)
    pitch_axis = np.array([0., 1., 0.])

    # Rows of each tangent array correspond to the roll and pitch perturbations.
    dtilt, drate = np.zeros((2, 2)), np.zeros((2, 2))
    jacobian = np.zeros((*known.shape, 2))
    for k, row in enumerate(known):
        request, tilt, rate = row[2:8].reshape(3, 2)
        drequest = np.diag(abs(actions[k]) < 1).astype(float)
        dtarget = limit*drequest
        # The held-command filter is affine in its inputs, so the same formula
        # also advances a perturbation of its state and target.
        dmid, dmid_rate, dmid_accel = filter_advance(dtilt, drate, dtarget, tau, .5*dt_s)

        rotation, roll_axis = _rotation(tilt, base_rotation)
        droll_axis = dmid[:, 1, None]*np.cross(pitch_axis, roll_axis)
        delta_rotation_world = dmid[:, :1]*roll_axis+dmid[:, 1:]*pitch_axis
        accel = (limit*request-tilt-2*tau*rate)/tau**2
        roll_velocity, pitch_velocity = roll_axis*rate[0], pitch_axis*rate[1]
        droll_velocity = droll_axis*rate[0]+dmid_rate[:, :1]*roll_axis
        dpitch_velocity = dmid_rate[:, 1:]*pitch_axis
        dangular_accel = (droll_axis*accel[0]+dmid_accel[:, :1]*roll_axis
                          +dmid_accel[:, 1:]*pitch_axis
                          +np.cross(dpitch_velocity, roll_velocity)
                          +np.cross(pitch_velocity, droll_velocity))
        # Gravity and prescribed reference acceleration are fixed in world
        # coordinates. All four feature vectors still rotate with the tool.
        dvectors_world = np.zeros((2, 4, 3))
        dvectors_world[:, 1] = droll_velocity+dpitch_velocity
        dvectors_world[:, 2] = dangular_accel
        local_vectors = row[8:].reshape(4, 3)
        dvectors_local = dvectors_world@rotation-np.cross(
            (delta_rotation_world@rotation)[:, None, :], local_vectors)

        jacobian[k, 2:4] = drequest.T
        jacobian[k, 4:6] = dmid.T
        jacobian[k, 6:8] = dmid_rate.T
        jacobian[k, 8:] = dvectors_local.reshape(2, 12).T
        dtilt, drate, _ = filter_advance(dtilt, drate, dtarget, tau, dt_s)
    return jacobian


def action_sensitivity(model, state_raw, time_s, tilt_rad, tilt_rate_rad_s, actions,
                       config, known_mean, known_scale):
    """Return (H,9) SI predictions and an (H,9,2) shared-action Jacobian.

    An offset in normalized roll or pitch is added to every request in the
    sequence. The initial state and time are fixed. Strictly saturated request
    components have derivative zero; exact clipping kinks are rejected.
    """
    initial, known, normalized = _inputs(model, state_raw, time_s, tilt_rad, tilt_rate_rad_s,
                                         actions, config, known_mean, known_scale)
    feature_jacobian = _known_jacobian(known, actions, config, model.dt_s)
    feature_jacobian /= np.asarray(known_scale)[None, :, None]
    prediction = model.rollout(initial[None], normalized[None])[0]
    jacobian = np.empty((*prediction.shape, 2))
    state_jacobian = np.zeros((STATE_DIM, 2))
    previous = initial
    outputs = np.eye(STATE_DIM)
    for k, features in enumerate(normalized):
        ds, dk, _ = model.step_vjp(np.broadcast_to(previous, outputs.shape),
                                  np.broadcast_to(features, (STATE_DIM, model.known_dim)),
                                  outputs)
        state_jacobian = ds@state_jacobian+dk@feature_jacobian[k]
        jacobian[k] = model.state_scale[:, None]*state_jacobian
        previous = prediction[k]
    return {'prediction': model.denormalize(prediction), 'jacobian': jacobian}
