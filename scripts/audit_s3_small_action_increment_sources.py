"""Trace measured small-action differences back to their generating control leads."""
import argparse
from itertools import combinations
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from audit_s3_ball_velocity_components import frame_series, half
from audit_s3_small_action_response import read
from audit_s3_surface_velocity_components import surface_series
from audit_s3_world_increment_drift import WINDOWS, integral, require_close
from audit_s3_world_increment_generation import prior, propagate, world_head
from audit_s3_world_increment_newton_reference import propagate_discrete_position, required_residual, tool_transition
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, known_rollout
from ball_retention.surface_diagnostic import SurfaceDiagnostic
from ball_retention.world_increment_model import WorldIncrementMLP
from collect_s3_small_action_response import load_run
from deterministic_archive_metrics import _full_branch


PHASE = 'small_action_increment_sources'
PARTS = ('tangent', 'normal', 'trapezoid', 'tool_frame')
WINDOWS = (*WINDOWS, (17, 32), (20, 32))
BLOCKS = {'position': slice(0, 3), 'velocity': slice(3, 6), 'spin': slice(6, 9)}


def oriented_split(vector, normals):
    normal = np.sum(vector*normals, axis=-1, keepdims=True)*normals
    return {'tangent': vector-normal, 'normal': normal}


def measured_branch(initial, trace, surface, dt, gravity):
    rotation, _, _, velocity_world = frame_series(initial, trace, dt)
    states = [initial, *trace['states']]
    position_world = np.asarray([s.ball.position for s in states])
    tool_world = np.asarray([s.tool.position for s in states])
    truth = np.vstack((_ball_state(initial), trace['raw']))
    normals, start_normals, _, feature, jump = surface_series(surface, truth, rotation)
    return {'raw': trace['raw'], 'rotation': rotation, 'position_world': position_world, 'tool_world': tool_world,
            'velocity_world': velocity_world, 'increment_world': np.diff(velocity_world, axis=0)-gravity,
            'normals_world': rotation[:-1].apply(start_normals), 'geometry': normals,
            'closest_feature': feature.tolist(), 'normal_jump_deg': jump.tolist()}


def measured_sources(branch, nominal, dt, mode):
    """Use the actual branch frame at every output lead and along each source path."""
    transform = branch['rotation'][1:].inv().as_matrix()
    delta_velocity = branch['velocity_world']-nominal['velocity_world']
    delta_position = branch['position_world']-nominal['position_world']
    lead = np.arange(len(delta_velocity)-1)
    steps_apart = lead[:, None]-lead[None, :]
    velocity_weights = dt*np.maximum(steps_apart+.5, 0.)
    branch_normal = branch['normals_world'] if mode == 'branch' else nominal['normals_world']
    local = oriented_split(branch['increment_world'], branch_normal)
    reference = oriented_split(nominal['increment_world'], nominal['normals_world'])
    sources = {k: np.einsum('eij,es,sj->sei', transform, velocity_weights, local[k]-reference[k])
               for k in local}
    shift = branch['rotation'][1:].apply(
        nominal['position_world'][1:]-branch['tool_world'][1:], inverse=True)-nominal['raw'][:, :3]
    gap = np.diff(delta_position, axis=0)-integral(delta_velocity[1:], dt, delta_velocity[0])
    sources['trapezoid'] = np.einsum('eij,es,sj->sei', transform, steps_apart >= 0, gap)
    # This is a frame difference at the output time; it is not a model increment.
    sources['tool_frame'] = np.eye(len(lead))[..., None]*shift[:, None, :]
    expected = branch['raw'][:, :3]-nominal['raw'][:, :3]
    error = require_close(sum(sources.values()).sum(axis=0), expected)
    return sources, error


def summarize(components):
    trajectory = sum(components.values()).sum(axis=0)
    norm = np.linalg.norm(trajectory[-1])
    axis = trajectory[-1]/norm if norm else trajectory[-1]
    news = {}
    for name, values in {**components, 'total': sum(components.values())}.items():
        projected = values @ axis
        news[name] = {'projection_at_h32_m': float(projected[:, -1].sum()),
                      'vector_by_evaluation_lead_m': values.sum(axis=0).tolist(),
                      'projection_by_source_and_evaluation_lead_m': projected.tolist(),
                      'projection_at_h32_by_source_window_m': {
                          f'{a}-{b}': float(projected[a-1:b, -1].sum()) for a, b in WINDOWS}}
    return {'norm_h32_m': float(norm), 'axis_h32_tool': axis.tolist(), 'terms': news}


def score_contact_windows(traces, names, generated):
    shortlist = sorted(np.argsort(np.abs(generated))[-4:].tolist())
    result = {}
    for lead in shortlist:
        result[str(lead+1)] = {}
        for name in names:
            row = [traces[f'{name}_{sign}']['contacts'][16*lead:16*(lead+1)] for sign in ('plus', 'minus')]
            result[str(lead+1)][name] = {'absolute_substeps': [r['physics_step'] for r in row[0]],
                'candidate_counts_plus_minus': [[r['candidates'] for r in values] for values in row],
                'candidate_counts_differ': sum(a['candidates'] != b['candidates'] for a, b in zip(*row)),
                'shape_pair_lists_differ': sum(a['shape_pairs'] != b['shape_pairs'] for a, b in zip(*row))}
    return result


def roll_feedback_blocks(model, features, measured, rolled):
    """Three-variable Shapley attribution of the learned increment change."""
    outputs = {}
    for count in range(len(BLOCKS)+1):
        for subset in combinations(BLOCKS, count):
            mixed = measured.copy()
            for k in subset:
                mixed[..., BLOCKS[k]] = rolled[..., BLOCKS[k]]
            outputs[frozenset(subset)] = world_head(model, features, mixed)
    contributions = {}
    for k in BLOCKS:
        terms = [(outputs[s | {k}]-v)*([1/3, 1/6, 1/3][len(s)]) for s, v in outputs.items() if k not in s]
        contributions[k] = sum(terms)
    require_close(sum(contributions.values()), outputs[frozenset(BLOCKS)]-outputs[frozenset()])
    return contributions


def model_roll(model, saved, cfg, initial, anchor, traces, series, rollouts, nominal_normal):
    keys = [f'roll_component_{sign}' for sign in ('plus', 'minus')]
    dt = model.dt_s
    truth = np.stack([traces[key]['raw'] for key in keys])
    known = known_rollout(initial.time_s, initial.tilt_rad, initial.tilt_rate_rad_s,
                          np.stack([traces[key]['actions'] for key in keys]), cfg, dt)
    features = model.features(known, cfg, anchor, saved['known_mean'], saved['known_scale'])
    start = _ball_state(initial)
    predicted = model.denormalize(model.rollout(model.normalize(np.broadcast_to(start, (2, 9))), features))
    require_close(predicted, np.asarray([rollouts['plus'], rollouts['minus']]))
    transform, world, shift = tool_transition(features)
    true_pre, roll_pre = (prior(start, path) for path in (truth, predicted))
    required, _ = required_residual(true_pre, truth, transform, world, shift)
    local = world_head(model, features, true_pre)
    feedback = world_head(model, features, roll_pre)-local
    position = true_pre[..., :3]+.5*dt*(true_pre[..., 3:6]+truth[..., 3:6])-truth[..., :3]
    discrete = propagate_discrete_position(position, transform, dt)
    splits = {}
    for tag, normals in (('branch', np.stack([series[key]['normals_world'] for key in keys])),
                         ('nominal', np.broadcast_to(nominal_normal, (2, *nominal_normal.shape)))):
        paths = {'trapezoid': discrete}
        for label, values in (('independent', local-required), ('feedback', feedback)):
            parts = oriented_split(values, normals)
            paths.update({f'{label}_{k}': propagate(v, transform, world, dt) for k, v in parts.items()})
        require_close(sum(paths.values()).sum(axis=0), predicted[..., :6]-truth[..., :6])
        splits[tag] = summarize({k: .5*(v[:, 0, :, :3]-v[:, 1, :, :3]) for k, v in paths.items()})
    blocks = roll_feedback_blocks(model, features, true_pre, roll_pre)
    projected = {}
    for name, value in blocks.items():
        trace = propagate(value, transform, world, dt)
        projected[name] = .5*(trace[:, 0, :, :3]-trace[:, 1, :, :3])
    axis = np.asarray(splits['nominal']['axis_h32_tool'])
    return {'splits': splits, 'feedback_block_projection_h32_m': {
                k: float(v[:, -1].sum(axis=0) @ axis) for k, v in projected.items()},
            'per_lead_independent_half_increment_error_world_m_s': half(local-required).tolist(),
            'per_lead_feedback_half_increment_error_world_m_s': half(feedback).tolist(),
            'predicted_half_position_tool_m': half(predicted[..., :3]).tolist()}


def run(plan, out, guard):
    if plan['updates']:
        raise ValueError('expected an offline source diagnostic')
    capture = Path(plan['capture_plan'])
    launch, cfg, spec, native, _, _, frozen, load_input = load_run(ROOT/capture, plan['files_sha256'][capture.as_posix()])
    status = read(plan, plan['capture_validation'])
    if (status['status'] != 'completed' or status['plan_sha256'] != plan['files_sha256'][capture.as_posix()]
            or (status['worker_starts'], status['physics_steps'], status['training_updates'],
                status['test_episode_files_read']) != (1, 7168, 0, False)
            or not status['nominal_gate_passed'] or status['branches'] != spec['branch_order']
            or len(status['accepted_prefixes']) != 4):
        raise ValueError('missing accepted physical replay')
    cap_folder = Path(plan['capture_validation']).parent/'worker'
    audit = read(plan, plan['previous_audit'])
    tick = spec['case']['tick']
    traces = {}
    for label in status['branches']:
        name = cap_folder/f'{label}.json'
        if plan['files_sha256'][name.as_posix()] != status['artifacts_sha256'][f'worker/{label}.json']:
            raise ValueError('changed small-action source')
        document = read(plan, name)
        traces[label] = {**_full_branch(document, tick), 'contacts': document['contacts']}
    old_dir = Path(launch['archived_response_validation']).parent/'worker'
    for case in spec['scoring']['archived_full_half_states_tool_si']:
        for sign in ('plus', 'minus'):
            label = f'{case}_{sign}'
            document = load_input(old_dir/f'{label}.json')
            traces[label] = {**_full_branch(document, tick), 'contacts': document['contacts']}
    initial = _parse_state(native['snapshot']['state'])
    dt = spec['substeps_per_control']*cfg['physics']['dt_s']
    gravity = [0., 0., -cfg['physics']['gravity_m_s2']*dt]
    mesh = load_tool(ROOT/'assets/shovel.STL')
    surface = SurfaceDiagnostic(mesh, cfg['ball']['radius_m'], cfg['initial']['tool_rotation_world'])
    series = {label: measured_branch(initial, row, surface, dt, gravity) for label, row in traces.items()}
    nominal = series['nominal_1']
    require_close(nominal['raw'], series['nominal_2']['raw'])
    contrasts = {f'{shape}_{amp}': {f'{shape}_{amp}': factor, shape: -1.}
                 for shape in ('pitch_pulse', 'pitch_reversal') for amp, factor in (('quarter', 4.), ('half', 2.))}
    contrasts['dual_interaction'] = {'dual_axis_pulse': 1., 'roll_component': -1., 'pitch_component': -1.}
    results = {name: {} for name in contrasts}
    worst = 0.
    for mode in ('branch', 'nominal'):
        source = {}
        for label, row in series.items():
            source[label], closure = measured_sources(row, nominal, dt, mode)
            worst = max(worst, closure)
        for key, coeffs in contrasts.items():
            combined = {part: sum(w*.5*(source[f'{name}_plus'][part]-source[f'{name}_minus'][part])
                                  for name, w in coeffs.items()) for part in PARTS}
            curve = sum(combined.values()).sum(axis=0)
            if key == 'dual_interaction':
                expected = [x['position']['newton_vectors']['interaction'] for x in audit['dual'].values()]
            else:
                shape, ratio = key.rsplit('_', 1)
                expected = [coeffs[key]*np.asarray(x['position'][ratio]['newton_half_difference'])
                            -np.asarray(x['position']['full']['newton_half_difference']) for x in audit['pitch'][shape].values()]
            worst = max(worst, require_close(curve, np.asarray(expected)))
            results[key][mode] = summarize(combined)
        guard(mode)
    for case, coeffs in contrasts.items():
        tangent = results[case]['nominal']['terms']['tangent']['projection_by_source_and_evaluation_lead_m']
        normal = results[case]['nominal']['terms']['normal']['projection_by_source_and_evaluation_lead_m']
        results[case]['candidate_log_at_largest_world_increment_leads'] = score_contact_windows(
            traces, coeffs, np.asarray(tangent)[:, -1]+np.asarray(normal)[:, -1])
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    models = {}
    for name, folder in launch['fixed_endpoint_sources'].items():
        file = Path(folder)/'world_increment_900.npz'
        with np.load(ROOT/file, allow_pickle=False) as saved:
            model = WorldIncrementMLP.from_state_dict(saved)
            if float(saved['dt_s']) != dt or saved['updates'].item() != 900:
                raise ValueError('unexpected fixed endpoint')
            models[name] = model_roll(model, saved, cfg, initial, anchor, traces, series,
                                      frozen[name]['pairs']['roll_component'], nominal['normals_world'])
        guard(name)
    write_json(out/'increment_sources.json', {'case': spec['case'], 'dt_s': dt, 'contrasts': results, 'roll_models': models,
        'geometry': {k: {n: v[n] for n in ('closest_feature','normal_jump_deg')} for k,v in series.items()},
        'convention': 'Attribution holds subsequent increments fixed. Branch mode uses each archived branch nearest mesh normal. Nominal mode uses one common normal per lead. Geometry is not a contact-force estimate.'})
    return {'measured_contrasts': len(contrasts), 'fixed_models': len(models), 'archived_branches': len(traces),
            'measured_position_closure_m': worst}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
