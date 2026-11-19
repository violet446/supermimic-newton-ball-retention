"""Pre-fit checks for the local surface world-model representation."""
import numpy as np

from ball_retention.action_grad import action_sensitivity, predict_ball
from ball_retention.model_data import _rotation
from ball_retention.surface_diagnostic import SurfaceDiagnostic
from ball_retention.world_model import ResidualMLP


def _training_rows(batch):
    """Return one row per physical snapshot, without counting axis/amplitude repeats."""
    first = {}
    for i, key in enumerate(zip(batch.group_ids, batch.episode_ids, batch.snapshot_ticks)):
        first.setdefault(key, i)
    return list(first.values())


def check_action_gradient(model, collection, seed):
    """Compare the raw SI rollout Jacobian for both shared request offsets."""
    batch = collection.train.batch
    rows = _training_rows(batch)
    rng = np.random.default_rng(seed)
    records = []
    epsilon = 1e-5
    for tick in sorted(set(batch.snapshot_ticks[rows])):
        i = int(rng.choice([row for row in rows if batch.snapshot_ticks[row] == tick]))
        state = batch.initial_state[i]
        args = (model, batch.initial[i], state['time_s'], state['tilt_rad'], state['tilt_rate_rad_s'])
        normalization = (collection.model_config, model.known_mean, model.known_scale)
        actions = batch.nominal_actions[i]
        analytical = action_sensitivity(*args, actions, *normalization)
        np.testing.assert_array_equal(analytical['prediction'], predict_ball(*args, actions, *normalization))
        for axis, offset in enumerate(epsilon*np.eye(2)):
            plus = predict_ball(*args, actions+offset, *normalization)
            minus = predict_ball(*args, actions-offset, *normalization)
            numerical = (plus-minus)/(2*epsilon)
            exact = analytical['jacobian'][..., axis]
            np.testing.assert_allclose(exact, numerical, rtol=2e-5, atol=3e-8)
            records.append({'episode': str(batch.episode_ids[i]), 'tick': int(tick), 'leads': len(actions),
                            'axis': axis, 'max_abs_error_si': float(np.max(np.abs(exact-numerical))),
                            'max_abs_jacobian_si': float(np.max(np.abs(exact)))})
    if not records or max(record['max_abs_jacobian_si'] for record in records) < 1e-8:
        raise ValueError('action VJP probe is degenerate')
    return records


def _coverage(model, collection, mesh):
    surface = model.surface
    batch = collection.train.batch
    rows = _training_rows(batch)
    points = batch.initial[rows, :3]
    normal, gap = surface.sample(points)
    weights = surface.weights(points, gap)
    if not np.isfinite(normal).all() or not np.isfinite(gap).all() or not np.isfinite(weights).all():
        raise ValueError('surface field outputs must be finite')
    if not np.any(weights > 0.) or np.max(np.ptp(surface.coefficients, axis=(0, 1, 2))) == 0.:
        raise ValueError('surface field is inactive or spatially constant on the training probes')
    geometry = SurfaceDiagnostic(mesh, surface.radius, collection.model_config['initial']['tool_rotation_world'])
    details = []
    for j, i in enumerate(rows):
        reference = geometry.measure(points[j])
        details.append({'group': str(batch.group_ids[i]), 'episode': str(batch.episode_ids[i]),
                        'tick': int(batch.snapshot_ticks[i]), 'weight': float(weights[j]),
                        'normal_tool': normal[j].tolist(), 'gap_m': float(gap[j]),
                        'reference': reference,
                        'normal_cosine': float(normal[j]@reference['normal_tool']),
                        'normal_gap_error_m': float(gap[j]-reference['normal_plane_gap_m'])})
    input_coverage = {}
    for label, values, mask in [('plus', batch.plus_target, batch.plus_mask),
                                ('minus', batch.minus_target, batch.minus_mask)]:
        previous = np.concatenate((batch.initial[:, None, :3], values[:, :-1, :3]), axis=1)
        positions = previous[mask]
        _, near_gaps = surface.sample(positions)
        input_weights = surface.weights(positions, near_gaps)
        input_coverage[label] = {'rows': len(positions), 'active_rows': int(np.count_nonzero(input_weights > 0)),
                                 'mean_weight': float(input_weights.mean())}
    return {'unique_snapshots': len(rows), 'active_snapshots': int(np.count_nonzero(weights > 0)),
            'full_weight_snapshots': int(np.count_nonzero(weights == 1)),
            'min_reference_normal_cosine': min(item['normal_cosine'] for item in details),
            'max_abs_gap_error_m': max(abs(item['normal_gap_error_m']) for item in details),
            'snapshots': details, 'paired_input_coverage': input_coverage}, rows[int(np.argmax(weights))]


def _field_probes(surface, state):
    """Choose interior, both tapers, and inactive positions for derivative probes."""
    low, spacing = surface.low, surface.spacing
    high = low+spacing*(np.array(surface.coefficients.shape[:3])-1)
    normal, _ = surface.sample(state[:3])
    offsets = np.r_[np.linspace(-.018, 0., 73), np.linspace(0., .018, 73)]
    motion = state[:3]+offsets[:, None]*normal
    _, motion_gaps = surface.sample(motion)
    weights = surface.weights(motion, motion_gaps)
    domain_clearance = np.min(np.minimum(motion-low, high-motion), axis=1)
    choices = np.flatnonzero((domain_clearance > 3*spacing) & (.0035 < np.abs(motion_gaps))
                             & (np.abs(motion_gaps) < .0075) & (weights > .05) & (weights < .95))
    if not choices.size:
        raise ValueError('preflight could not locate an active gap taper on the frozen field')
    gap_taper = motion[choices[np.argmin(np.abs(weights[choices]-.5))]]

    # Inspect all six interior edge slabs, including points away from the basin.
    # Some slabs may have no surface within the fixed gap band.
    grid = np.meshgrid(*[np.arange(3, size-3) for size in surface.coefficients.shape[:3]], indexing='ij')
    slab_core = low + spacing*np.stack(grid, axis=-1).reshape(-1, 3)
    boundary_points = []
    for axis in range(3):
        axis_mask = slab_core[:, axis] == slab_core[:, axis].min()
        slab = slab_core[axis_mask].copy()
        for coordinate in (low[axis]+2*spacing, high[axis]-2*spacing):
            face = slab.copy()
            face[:, axis] = coordinate
            boundary_points.extend(face)
    boundary_points = np.asarray(boundary_points)
    _, gaps = surface.sample(boundary_points)
    weights = surface.weights(boundary_points, gaps)
    choices = np.flatnonzero((np.abs(gaps) <= .003) & (weights > .05) & (weights < .95))
    if not choices.size:
        raise ValueError('preflight could not locate an active domain taper on the frozen field')
    edge_taper = boundary_points[choices[np.argmin(np.abs(gaps[choices]))]]
    edge_inactive = edge_taper.copy()
    closest_axis = int(np.argmin(np.minimum(edge_taper-low, high-edge_taper)))
    lower = edge_taper[closest_axis]-low[closest_axis] < high[closest_axis]-edge_taper[closest_axis]
    edge_inactive[closest_axis] = (low[closest_axis]+.5*spacing if lower else high[closest_axis]-.5*spacing)
    outside = edge_inactive.copy()
    outside[closest_axis] = low[closest_axis]-spacing if lower else high[closest_axis]+spacing

    # A distant point within the field isolates inactivity due to the gap.
    gap_zero = np.flatnonzero((domain_clearance > 3*spacing) & (np.abs(motion_gaps) >= .0085))
    if not gap_zero.size:
        raise ValueError('preflight could not locate an inactive gap probe')
    positions = np.stack((state[:3], gap_taper, edge_taper, edge_inactive, outside, motion[gap_zero[0]]))
    probes = np.broadcast_to(state, (len(positions), 9)).copy()
    probes[:, :3] = positions
    normals, gaps = surface.sample(positions)
    weights = surface.weights(positions, gaps)
    omega = np.tile([.03, -.02, .04], (len(probes), 1))
    features, state_jac, omega_jac = surface.features(probes, omega, jacobian=True)
    for array in (features, state_jac, omega_jac, normals, gaps, weights):
        if not np.isfinite(array).all():
            raise ValueError('surface field probe produced a nonfinite value or derivative')
    if not np.all(weights[:3] > .05) or not np.all(weights[1:3] < .95):
        raise ValueError('surface field preflight does not exercise both tapers')
    for array in (features, state_jac, omega_jac, weights):
        np.testing.assert_array_equal(array[3:], 0.)
    return probes, weights


def _step_vjps(model, states, known, rng):
    numerical_state = np.zeros_like(states)
    numerical_known = np.zeros_like(known)
    cotangent = rng.normal(size=states.shape)
    ds, dk, dp = model.step_vjp(states, known, cotangent)
    epsilon = 2e-6

    def objective(x, k):
        return np.sum(model.step(x, k)*cotangent, axis=1)

    for axis, basis in enumerate(epsilon*np.eye(states.shape[-1])):
        numerical_state[:, axis] = (objective(states+basis, known)-objective(states-basis, known))/(2*epsilon)
    for axis, basis in enumerate(epsilon*np.eye(known.shape[-1])):
        numerical_known[:, axis] = (objective(states, known+basis)-objective(states, known-basis))/(2*epsilon)
    np.testing.assert_allclose(ds, numerical_state, rtol=2e-5, atol=3e-8)
    np.testing.assert_allclose(dk, numerical_known, rtol=2e-5, atol=3e-8)

    errors = {}
    base_width = states.shape[-1]+known.shape[-1]
    for name, param in model.params.items():
        original = param.copy()
        regions = ([('base', slice(0, base_width)), ('surface', slice(base_width, None))]
                   if name == 'W1' else [('all', slice(None))])
        for region, indices in regions:
            direction = np.zeros_like(param)
            direction[indices] = rng.normal(size=param[indices].shape)
            direction /= np.linalg.norm(direction)
            try:
                param[...] = original+epsilon*direction
                plus = objective(states, known).sum()
                param[...] = original-epsilon*direction
                minus = objective(states, known).sum()
            finally:
                param[...] = original
            numerical = (plus-minus)/(2*epsilon)
            exact = np.sum(dp[name]*direction)
            np.testing.assert_allclose(exact, numerical, rtol=2e-5, atol=3e-8)
            errors[f'{name}/{region}'] = {'absolute_error': float(abs(exact-numerical)), 'derivative': float(exact)}
    if abs(errors['W1/surface']['derivative']) < 1e-7:
        raise ValueError('surface parameter VJP probe is degenerate')
    return {'state_max_abs_error': float(np.max(np.abs(ds-numerical_state))),
            'known_max_abs_error': float(np.max(np.abs(dk-numerical_known))), 'parameter_directions': errors}


def run_preflight(model, collection, mesh, seed, guard=None):
    """Verify field and model with retained training inputs and reversible probes."""
    rng = np.random.default_rng(seed)
    coverage, i = _coverage(model, collection, mesh)
    if guard is not None:
        guard('preflight_coverage')
    raw_states, weights = _field_probes(model.surface, collection.train.batch.initial[i])
    states = model.normalize(raw_states)
    batch = collection.train.batch
    known = (batch.plus_known[i]-model.known_mean)/model.known_scale
    step_known = np.broadcast_to(known[0], (len(states), len(known[0])))

    baseline = ResidualMLP(model.known_dim, model.state_mean, model.state_scale, model.dt_s,
                           seed, hidden=model.hidden, residual_gain=model.residual_gain)
    for name, value in baseline.params.items():
        candidate = model.params[name]
        if name == 'W1':
            np.testing.assert_array_equal(candidate[len(value):], 0.)
            candidate = candidate[:len(value)]
        np.testing.assert_array_equal(candidate, value)
    batched_known = np.broadcast_to(known, (len(states), *known.shape))
    np.testing.assert_array_equal(model.rollout(states, batched_known), baseline.rollout(states, batched_known))

    initial = batch.initial_state[i]
    base_rotation = np.array(collection.model_config['initial']['tool_rotation_world'])
    rotation, roll_axis = _rotation(np.array(initial['tilt_rad']), base_rotation)
    roll_rate, pitch_rate = initial['tilt_rate_rad_s']
    expected_omega = rotation.T@(roll_axis*roll_rate+np.array([0., pitch_rate, 0.]))
    np.testing.assert_allclose(model.tool_omega(known[0]), expected_omega, rtol=0., atol=2e-15)

    # Exercise the surface path before fitting without retaining any probe weights.
    original = {key: value.copy() for key, value in model.params.items()}
    try:
        model.params['W1'][-7:] = rng.normal(scale=.15, size=model.params['W1'][-7:].shape)
        model.params['W3'][...] = rng.normal(scale=.05, size=model.params['W3'].shape)
        vjps = _step_vjps(model, states, step_known, rng)
        for name, value in baseline.params.items():
            value[...] = model.params[name][:len(value)] if name == 'W1' else model.params[name]
        np.testing.assert_allclose(model.step(states[3:], step_known[3:]),
                                   baseline.step(states[3:], step_known[3:]), rtol=0., atol=1e-14)
        if guard is not None:
            guard('preflight_step_vjps')
        action = check_action_gradient(model, collection, seed)
    finally:
        for name, value in original.items():
            model.params[name][...] = value
    return {'status': 'passed', 'initial_parameters_and_predictions_equal': True,
            'outside_surface_feature_jacobians_zero': True, 'outside_step_matches_baseline': True,
            'initial_tool_omega_verified': True, 'field_probe_weights': weights.tolist(),
            'training_coverage': coverage, 'step_vjp': vjps, 'action_gradients': action}
