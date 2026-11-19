"""Compare the four restored Newton snapshots with frozen model responses."""
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json

import numpy as np

from ball_retention.action_eval import COMPONENTS, evaluate_pair
from ball_retention.action_grad import action_sensitivity, predict_ball
from ball_retention.collection import replay_error
from ball_retention.local_surface import SurfaceField
from ball_retention.local_surface_model import LocalSurfaceResidualMLP
from ball_retention.pair_data import _branch
from ball_retention.world_model import ResidualMLP
from diagnose_s3_action_response import compare, component


CHECKPOINTS = {
    'old_900': 'runs/s3_physical_scales_20261113/full_training.npz',
    'expanded_1800': 'runs/s3_history_duration_20261114/expanded_1800.npz',
    'candidate_1800': 'runs/s3_local_surface_refit_20261114/candidate_1800.npz',
}
FIELD = 'runs/s3_local_surface_refit_20261114/field.npz'
ARCHIVED_AMPLITUDE = .20


def verified_file(path, digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f'frozen hash mismatch: {path}')
    return raw


def branch_states(record, nominal, amplitudes, dt, substeps):
    branches = {(b['axis'], b['offset'], b['repeat']): b for b in record['branches']}
    expected = {(None, 0., 0), (None, 0., 1)} | {
        (axis, sign*a, 0) for axis in (0, 1) for a in amplitudes for sign in (-1., 1.)}
    if len(branches) != len(record['branches']) or set(branches) != expected:
        raise ValueError('incomplete or duplicate action branches')
    states = {}
    for key, branch in branches.items():
        axis, offset, _ = key
        requested = nominal.copy()
        if axis is not None:
            requested[:, axis] += offset
        target, counts, ending = _branch(branch, requested, dt, substeps)
        if ending != (False, False, None) or not (counts == substeps).all():
            raise ValueError(f'branch did not complete its full 32 controls: {key}')
        states[key] = target
    return states


def physical_metrics(branches, axis, amplitude, center, leads):
    plus, minus = (branches[axis, sign*amplitude, 0] for sign in (1., -1.))
    replay_a, replay_b = (branches[None, 0., r] for r in (0, 1))
    # Passing each physical branch as the predicted branch too makes the
    # reference threshold depend on Newton and its own zero replays alone.
    metrics = evaluate_pair(plus, minus, plus, minus, amplitude, center,
                            replay_first=replay_a, replay_second=replay_b, leads=leads)
    result = {}
    for h, row in metrics['by_lead'].items():
        result[h] = {name: newton_response(row[name]) for name in COMPONENTS}
    return result


def newton_response(metric):
    threshold, spread = metric['signal_threshold'], metric['newton_half_spread']
    return {'secant_si': metric['newton_derivative'], 'norm_si': metric['newton_norm'],
            'half_spread_si': spread, 'threshold_si': threshold,
            'zero_replay_half_spread_si': metric['replay_half_spread'],
            'signal_to_threshold': spread/threshold,
            'signal': 'strong' if spread >= 10.*threshold else 'near' if metric['newton_resolved'] else 'unresolved'}


def scale_comparison(reference, other, *, require_strong):
    a, b = np.array(reference['secant_si']), np.array(other['secant_si'])
    resolved = (reference['signal'] != 'unresolved', other['signal'] != 'unresolved')
    both_strong = reference['signal'] == other['signal'] == 'strong'
    eligible = all(resolved) and (both_strong or not require_strong)
    metrics = compare(b, a, resolved[1], resolved[0])
    stable = metrics['cosine'] >= .8 and metrics['relative_error'] <= .5 if eligible else None
    return {**metrics, 'eligible': eligible, 'both_strong': both_strong, 'stable': stable,
            'reference_signal': reference['signal'], 'other_signal': other['signal']}


def model_responses(model, km, ks, record, cfg, branches, physics, axis, amplitude, nominal, derivative, leads):
    state = record['initial_state']
    args = (model, record['initial_raw'], state['time_s'], state['tilt_rad'], state['tilt_rate_rad_s'])
    tail = (cfg, km, ks)
    offset = np.eye(2)[axis]*amplitude
    plus, minus = (predict_ball(*args, nominal+sign*offset, *tail) for sign in (1., -1.))
    metrics = evaluate_pair(plus, minus, branches[axis, amplitude, 0], branches[axis, -amplitude, 0],
                            amplitude, cfg['initial']['anchor_xy_tool_m'], replay_first=branches[None, 0., 0],
                            replay_second=branches[None, 0., 1], leads=leads)
    results = {}
    for lead, row in metrics['by_lead'].items():
        h = int(lead)-1
        even_model = (plus[h]+minus[h])*.5-derivative['prediction'][h]
        even_newton = (branches[axis, amplitude, 0][h]+branches[axis, -amplitude, 0][h]
                       -branches[None, 0., 0][h]-branches[None, 0., 1][h])*.5
        results[lead] = {}
        for name, region in COMPONENTS.items():
            metric = row[name]
            physical_floor = physics[lead][name]['threshold_si']
            metric['signal_threshold'] = physical_floor
            metric['model_resolved'] = metric['model_half_spread'] >= physical_floor
            metric['newton_resolved'] = metric['newton_half_spread'] >= physical_floor
            results[lead][name] = component(metric, derivative['jacobian'][h, region, axis],
                                            even_model[region], even_newton[region], amplitude, 10.)
    return results


def stable_counts(records):
    return {'cases': len(records), 'eligible': sum(r['eligible'] for r in records),
            'both_strong': sum(r['both_strong'] for r in records),
            'stable': sum(r['stable'] is True for r in records),
            'unstable': sum(r['stable'] is False for r in records)}


def alignment(rows, model, h, amplitude):
    label = str(amplitude)
    subset = [row['models'][model][label][h]['position'] for row in rows
              if row['physics'][label][h]['position']['signal'] == 'strong']
    aligned = {'strong_cases': len(subset)}
    for key in ('jacobian_vs_newton', 'secant_vs_newton', 'jacobian_vs_secant'):
        cosines = [item[key]['cosine'] for item in subset if item[key]['cosine'] is not None]
        aligned[key] = {'comparable': len(cosines), 'positive': sum(x > 0. for x in cosines),
                        'cosine_ge_0_5': sum(x >= .5 for x in cosines)}
    return aligned


def evaluate(root, out, design, guard=None):
    """Write validated, same-session action comparisons and return compact counts."""
    root, out = root.resolve(), out.resolve()
    fixed = design['inputs_sha256']
    for path in [*CHECKPOINTS.values(), FIELD]:
        verified_file(root/path, fixed[path])
    with np.load(root/FIELD, allow_pickle=False) as saved:
        field = SurfaceField(saved['low'], float(saved['spacing']), saved['coefficients'], float(saved['radius']))
    amplitudes, substeps, leads = design['amplitudes'], design['control_substeps'], design['leads']
    amplitude_pairs = tuple(combinations(amplitudes, 2))
    rows, snapshots = [], []
    for case in design['cases']:
        episode, tick = case['episode'], case['tick']
        if guard is not None:
            guard(f'evaluate_{episode}')
        path = out/episode/'action_pairs.jsonl'
        raw = path.read_bytes()
        record, = (json.loads(line) for line in raw.splitlines())
        source = verified_file(root/case['source_trace'], case['source_trace_sha256'])
        archived, = (row for line in source.splitlines() if (row := json.loads(line))['snapshot_tick'] == tick)
        cfg = json.loads(verified_file(root/case['source_config'], case['source_config_sha256']))
        dt = cfg['physics']['dt_s']
        keys = ('scenario', 'nominal_actions', 'control_substeps', 'physics_dt_s')
        if (record['snapshot_tick'] != tick or record['amplitudes'] != amplitudes
                or record['control_substeps'] != substeps or record['physics_dt_s'] != dt
                or any(record[key] != archived[key] for key in keys)):
            raise ValueError(f'archived and newly captured schedules or metadata differ: {episode}')
        state_delta = replay_error(record['initial_state'], archived['initial_state'])
        raw_delta = replay_error(record['initial_raw'], archived['initial_raw'])
        anchor_delta = replay_error(record['anchor_local_m'], archived['anchor_local_m'])
        if max(state_delta, raw_delta, anchor_delta) > 1e-5:
            raise ValueError(f'archived and newly captured initial conditions differ: {episode}')
        nominal = np.asarray(record['nominal_actions'], dtype=float)
        if nominal.shape != (design['horizon'], 2) or not np.isfinite(nominal).all() or np.max(abs(nominal))+max(amplitudes) >= 1:
            raise ValueError(f'unexpected nominal control horizon or clipping: {episode}')
        branches = branch_states(record, nominal, amplitudes, dt, substeps)
        prior = branch_states(archived, nominal, archived['amplitudes'], dt, substeps)
        models = {}
        for name, checkpoint in CHECKPOINTS.items():
            with np.load(root/checkpoint, allow_pickle=False) as saved:
                if name == 'candidate_1800':
                    if str(saved['field_sha256']) != fixed[FIELD]:
                        raise ValueError('local-surface field hash does not match checkpoint')
                    model = LocalSurfaceResidualMLP.from_checkpoint(saved, field, cfg)
                else:
                    model = ResidualMLP.from_state_dict(saved)
                km, ks = saved['known_mean'], saved['known_scale']
            if not np.isclose(model.dt_s, dt*substeps, atol=1e-14, rtol=0.):
                raise ValueError(f'checkpoint clock does not match capture: {name}')
            state = record['initial_state']
            derivative = action_sensitivity(model, record['initial_raw'], state['time_s'], state['tilt_rad'],
                                            state['tilt_rate_rad_s'], nominal, cfg, km, ks)
            models[name] = (model, km, ks, derivative)
        snapshots.append({'episode': episode, 'group': case['group'], 'snapshot_tick': tick, 'seed': case['seed'],
                          'numeric_snapshot_delta': state_delta, 'initial_raw_delta': raw_delta,
                          'anchor_local_delta_m': anchor_delta, 'trace_sha256': sha256(raw).hexdigest()})
        center = cfg['initial']['anchor_xy_tool_m']
        for axis in (0, 1):
            physical = {str(a): physical_metrics(branches, axis, a, center, leads) for a in amplitudes}
            old_physical = physical_metrics(prior, axis, ARCHIVED_AMPLITUDE, center, leads)
            scale = {}
            for smaller, larger in amplitude_pairs:
                reference, other = physical[str(smaller)], physical[str(larger)]
                scale[f'{smaller}/{larger}'] = {
                    h: scale_comparison(reference[h]['position'], other[h]['position'], require_strong=True) for h in reference}
            new_main = physical[str(ARCHIVED_AMPLITUDE)]
            repeat = {h: scale_comparison(old_physical[h]['position'], new_main[h]['position'], require_strong=False)
                      for h in old_physical}
            model_rows = {name: {str(a): model_responses(model, km, ks, record, cfg, branches, physical[str(a)],
                                                        axis, a, nominal, derivative, leads) for a in amplitudes}
                          for name, (model, km, ks, derivative) in models.items()}
            rows.append({'episode': episode, 'group': case['group'], 'snapshot_tick': tick, 'axis': design['axes'][axis],
                         'physics': physical, 'archived_0_20': old_physical, 'newton_scale_stability': scale,
                         'newton_0_20_repeatability': repeat, 'models': model_rows})

    by_lead = {}
    for lead in leads:
        h = str(lead)
        by_lead[h] = {
            'position_signals': {str(a): dict(Counter(row['physics'][str(a)][h]['position']['signal']
                                                      for row in rows)) for a in amplitudes},
            'newton_scale_stability': {f'{a}/{b}': stable_counts([row['newton_scale_stability'][f'{a}/{b}'][h]
                                                                  for row in rows]) for a, b in amplitude_pairs},
            'newton_0_20_repeatability': stable_counts([row['newton_0_20_repeatability'][h] for row in rows]),
            'model_alignments': {name: {str(a): alignment(rows, name, h, a) for a in amplitudes} for name in CHECKPOINTS}}
    summary = {'snapshots': len(snapshots), 'snapshot_axes': len(rows), 'H32': by_lead['32']}
    with (out/'comparison.json').open('x', encoding='utf-8') as stream:
        json.dump({'status': 'evaluated', 'summary': summary, 'by_lead': by_lead, 'snapshots': snapshots, 'cases': rows,
                   'interpretation': ['Within-session strong/stable secants describe the measured finite widths.',
                                      'Archived and current 0.20 comparisons use separate zero-replay floors.',
                                      'Four previously observed development cases provide targeted diagnostics.'],
                   'control_accepted': False}, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if guard is not None:
        guard('evaluation_complete')
    return summary
