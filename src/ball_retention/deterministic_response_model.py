"""Evaluate frozen model responses to time-varying controls without Newton."""
from io import BytesIO
from pathlib import Path

import numpy as np

from .action_grad import _inputs, _known_jacobian, predict_ball
from .archive_restore_plan import _verified_bytes
from .archive_state_import import _parse_state
from .model_data import STATE_NAMES, _ball_state
from .world_model import STATE_DIM, ResidualMLP


def _directional_rollout(model, initial, known, normalized, actions, directions, cfg, scale):
    """Propagate model JVPs for a stack of action directions shaped (D,H,2)."""
    jumps = np.diff(directions, axis=1, prepend=0.)
    feature_tangent = np.zeros((*known.shape, len(directions)))
    # Each change in a direction is a shared offset from that lead onward.
    # Adding the causal suffix derivatives reconstructs the full direction.
    for index in np.flatnonzero(np.any(jumps != 0., axis=(0, 2))):
        suffix = _known_jacobian(known[index:], actions[index:], cfg, model.dt_s)
        feature_tangent[index:] += suffix @ jumps[:, index].T
    feature_tangent /= scale[None, :, None]

    prediction = model.rollout(initial[None], normalized[None])[0]
    tangent = np.zeros((STATE_DIM, len(directions)))
    trajectory = np.empty((*prediction.shape, len(directions)))
    previous = initial
    outputs = np.eye(STATE_DIM)
    for i, features in enumerate(normalized):
        ds, dk, _ = model.step_vjp(np.broadcast_to(previous, outputs.shape),
                                  np.broadcast_to(features, (STATE_DIM, model.known_dim)), outputs)
        tangent = ds @ tangent + dk @ feature_tangent[i]
        trajectory[i] = model.state_scale[:, None]*tangent
        previous = prediction[i]
    return model.denormalize(prediction), np.moveaxis(trajectory, -1, 0)


def evaluate_model(root, spec, cfg, snapshot_state_dict, nominal, directions, guard):
    """Return complete SI model rollouts and validate exact JVPs against centered FD.

    The public State at the snapshot is the only measured model input.
    The JVP is the derivative with respect to the coefficient of each direction.
    """
    checkpoint = spec['checkpoint']
    digest = spec['checkpoint_sha256']
    raw = _verified_bytes(Path(root).resolve(), checkpoint, digest)
    with np.load(BytesIO(raw), allow_pickle=False) as saved:
        model = ResidualMLP.from_state_dict(saved)
        mean, scale = saved['known_mean'], saved['known_scale']
    if not np.isclose(model.dt_s, 1/cfg['physics']['control_hz'], atol=1e-14, rtol=0.):
        raise ValueError('model and physical control periods disagree')
    state = _parse_state(snapshot_state_dict)
    state_raw = _ball_state(state)
    args = (model, state_raw, state.time_s, state.tilt_rad, state.tilt_rate_rad_s)
    tail = (cfg, mean, scale)
    nominal = np.asarray(nominal, dtype=float)
    initial, known, normalized = _inputs(*args, nominal, *tail)
    direction_rows = np.asarray(list(directions.values()), dtype=float)
    if (not len(directions) or direction_rows.shape != (len(directions), *nominal.shape)
            or not np.isfinite(direction_rows).all() or not np.all(np.any(direction_rows != 0., axis=(1, 2)))):
        raise ValueError('each response direction must be finite, nonzero and shaped like the nominal actions')
    amplitude, fd_step, rtol, atol = (float(spec[key]) for key in ('amplitude', 'fd_step', 'fd_rtol', 'fd_atol_si'))
    if not (np.isfinite([amplitude, fd_step, rtol, atol]).all() and 0. < fd_step < amplitude and rtol > 0. and atol > 0.):
        raise ValueError('invalid model response amplitude or centered FD tolerances')
    if np.any(np.abs(nominal) + amplitude*np.abs(direction_rows) >= 1.):
        raise ValueError('model response actions reach clipping')

    guard()
    prediction, tangents = _directional_rollout(model, initial, known, normalized, nominal, direction_rows, cfg, scale)
    np.testing.assert_array_equal(prediction, predict_ball(*args, nominal, *tail))
    pairs = {}
    for label, direction, tangent in zip(directions, direction_rows, tangents, strict=True):
        fd_plus, fd_minus = (predict_ball(*args, nominal+sign*fd_step*direction, *tail) for sign in (1., -1.))
        fd = (fd_plus-fd_minus)/(2*fd_step)
        if not np.isfinite(tangent).all() or not np.isfinite(fd).all():
            raise ValueError(f'non-finite model JVP or centered FD in {label}')
        np.testing.assert_allclose(tangent, fd, rtol=rtol, atol=atol, err_msg=f'directional FD for {label}')
        plus, minus = (predict_ball(*args, nominal+sign*amplitude*direction, *tail) for sign in (1., -1.))
        if not np.isfinite([prediction, plus, minus]).all():
            raise ValueError(f'non-finite frozen model predictions in {label}')
        pairs[label] = {'plus': plus.tolist(), 'minus': minus.tolist(), 'directional_jvp': tangent.tolist(),
                        'centered_fd': fd.tolist(), 'secant': ((plus-minus)/(2*amplitude)).tolist(),
                        'fd_max_abs_error_si_by_component': np.max(np.abs(tangent-fd), axis=0).tolist()}
        guard()
    return {'checkpoint': checkpoint, 'checkpoint_sha256': digest, 'amplitude': amplitude,
            'fd_step': fd_step, 'fd_rtol': rtol, 'fd_atol_si': atol, 'passed': True,
            'state_components': list(STATE_NAMES), 'initial_state_raw_si': state_raw.tolist(),
            'nominal': prediction.tolist(), 'pairs': pairs}
