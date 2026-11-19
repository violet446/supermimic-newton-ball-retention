"""Decode and validate the saved S3 native snapshot without stepping physics."""

import math

import numpy as np

from ball_retention.archive_state_import import _check_layout, _parse_state, _readback_errors
from ball_retention.task import reference


def _require_fields(mapping, fields, name):
    if not isinstance(mapping, dict) or mapping.keys() != set(fields.split()):
        raise ValueError(f'unexpected fields in {name}')


def _numeric_array(values, shape, dtype, name):
    items = np.asarray(values, dtype=object)
    if items.shape != shape or any(type(item) not in (int, float) for item in items.flat):
        raise ValueError(f'{name}: expected JSON numbers with shape {shape}')
    with np.errstate(over='ignore', invalid='ignore'):
        array = np.asarray(values, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError(f'{name}: values must remain finite as {dtype}')
    return array


def _validate_state_numbers(fields):
    # The public contract validates lengths and normalization. Check JSON types
    # before it can coerce a string or a boolean to a floating-point value.
    for value in fields.values():
        if isinstance(value, dict):
            _validate_state_numbers(value)
        else:
            shape = (len(value),) if isinstance(value, list) else ()
            _numeric_array(value, shape, 'float64', 'public State')


def decode_snapshot(document, layout, cfg, tick, substeps, command=None):
    """Reconstitute the original snapshot contract from its JSON representation."""
    _require_fields(document, 'schema physics controller monitor episode_id step_index state done invalid deep_count',
                    'episode snapshot')
    physical = document['physics']
    controller = document['controller']
    monitor = document['monitor']
    _require_fields(physical, 'schema steps arrays command', 'physics snapshot')
    _require_fields(controller, 'time_s q qd tilt filtered', 'controller snapshot')
    _require_fields(monitor, 'last_time_s outside_since_s terminated reason', 'departure snapshot')
    if (document['schema'] != 'ball-retention-episode-snapshot-v1' or physical['schema'] != 'prescribed-arm-v1'
            or not isinstance(document['episode_id'], str) or not document['episode_id']):
        raise ValueError('snapshot schema or episode ID mismatch')

    steps = tick*substeps
    time_s = steps*cfg['physics']['dt_s']
    controller_time = _numeric_array(controller['time_s'], (), 'float64', 'controller time').item()
    monitor_time = _numeric_array(monitor['last_time_s'], (), 'float64', 'monitor time').item()
    fields = document['state']
    if not isinstance(fields, dict):
        raise ValueError('snapshot requires a public State object')
    _validate_state_numbers(fields)
    state = _parse_state(fields)
    counters = ((document['step_index'], tick), (physical['steps'], steps), (document['deep_count'], 0))
    if (any(type(actual) is not int or actual != expected for actual, expected in counters)
            or document['done'] is not False or document['invalid'] is not False
            or monitor['terminated'] is not False or monitor['reason'] is not None
            or monitor['outside_since_s'] is not None or state.departure_outside_s != 0.
            or not math.isclose(controller_time, time_s, abs_tol=1e-12, rel_tol=0.)
            or state.time_s != time_s or monitor_time != time_s
            or state.reference_phase != reference(time_s, cfg)['phase']):
        raise ValueError('snapshot clocks or clean-interior counters differ from the frozen tick')

    count = layout['free_joint_count']
    if type(count) is not int or count <= 0 or len(layout['body_labels']) != count:
        raise ValueError('invalid body count in saved native layout')
    shapes = {'body_q': (count, 7), 'body_qd': (count, 6), 'body_f': (count, 6),
              'joint_q': (count*7,), 'joint_qd': (count*6,)}
    expected_arrays = {name: {'shape': list(shape), 'dtype': 'float32'} for name, shape in shapes.items()}
    if layout['state_arrays'] != expected_arrays:
        raise ValueError('unsupported saved state array layout')
    _require_fields(physical['arrays'], ' '.join(shapes), 'native arrays')
    arrays = {name: _numeric_array(physical['arrays'][name], shape, 'float32', name)
              for name, shape in shapes.items()}
    q, qd, qdd = _numeric_array(physical['command'], (3, 6), 'float64', 'native joint command')
    tilt = _numeric_array(controller['tilt'], (2,), 'float64', 'controller tilt')
    filtered = _numeric_array(controller['filtered'], (2,), 'float64', 'controller first filter stage')
    for values, expected, name in (
        (_numeric_array(controller['q'], (6,), 'float64', 'controller q'), q, 'controller q'),
        (_numeric_array(controller['qd'], (6,), 'float64', 'controller qd'), qd, 'controller qd'),
        (q, state.joint_q, 'public q'), (qd, state.joint_qd, 'public qd'), (tilt, state.tilt_rad, 'public tilt'),
    ):
        np.testing.assert_array_equal(values, expected, err_msg=name)

    if command is not None:
        if not math.isclose(controller_time, command.time_s, abs_tol=1e-12, rel_tol=0.):
            raise ValueError('saved and derived lead clocks differ')
        for values, expected, name in (
            (q, command.q, 'q'), (qd, command.qd, 'qd'), (qdd, command.qdd, 'qdd'),
            (tilt, command.tilt, 'tilt'), (state.tilt_rate_rad_s, command.tilt_rate, 'tilt rate'),
            (filtered, command.filter_input_state, 'filter state'),
            (state.previous_action, command.clipped_action, 'previous action'),
        ):
            np.testing.assert_array_equal(values, expected, err_msg=f'saved and derived {name}')

    return {**document, 'state': state, 'controller': dict(controller), 'monitor': dict(monitor),
            'physics': {**physical, 'arrays': arrays, 'command': (q.copy(), qd.copy(), qdd.copy())}}


def verify_live_layout(physics, recorded_layout):
    """Match the frozen metadata against both fresh Newton buffers and their template."""
    actual_layout = _check_layout(physics)
    if actual_layout != recorded_layout:
        raise ValueError('fresh scene layout differs from the saved native layout')
    return actual_layout


def restore_and_readback(env, saved, substeps):
    """Restore both buffers and verify measured State and discrete fields before stepping."""
    env.restore(saved)
    physics = env.physics
    physical = saved['physics']
    public = saved['state']
    if (env.step_index != saved['step_index'] or physics.steps != env.step_index*substeps
            or physics.steps != physical['steps'] or env.episode_id != saved['episode_id']
            or env.done != saved['done'] or env.invalid != saved['invalid'] or env.deep_count != saved['deep_count']
            or env.controller.snapshot() != saved['controller'] or env.departure.snapshot() != saved['monitor']
            or env.state != public):
        raise ValueError('restore changed the clock, controller, monitor, counters or stored State')
    for buffer in (physics.a, physics.b):
        for name, expected in physical['arrays'].items():
            actual = getattr(buffer, name).numpy()
            if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
                raise ValueError(f'restored native buffer changed {name}')
    for actual, expected in zip((physics.command_q, physics.command_qd, physics.command_qdd), physical['command']):
        np.testing.assert_array_equal(actual, expected)

    row = physics.read()
    monitor = env.departure.snapshot()
    outside_since = monitor['outside_since_s']
    outside_duration = 0. if outside_since is None else row['time_s']-outside_since
    controller = env.controller
    observed = env._state(row, controller.tilt, (controller.filtered-controller.tilt)/controller.tau,
                          public.previous_action, {'outside_duration_s': outside_duration})
    observed.validate()
    return {'readback_errors': _readback_errors(observed, public), 'both_buffers_exact': True,
            'controller_monitor_counters_match': True, 'tick': env.step_index, 'physics_tick': physics.steps,
            'controller': controller.snapshot(), 'monitor': monitor, 'deep_count': env.deep_count,
            'native_joint_command': [array.tolist() for array in physical['command']],
            'native_arrays': {name: {'shape': list(array.shape), 'dtype': str(array.dtype)}
                              for name, array in physical['arrays'].items()}}
