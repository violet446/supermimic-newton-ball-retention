"""Trace the shared pitch prefix and the effect of the later reversal request."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from s3_fitability_common import ROOT, SCOPE, run_phase, write_json

import numpy as np

from audit_s3_world_increment_drift import WINDOWS, audit, describe, half, require_close
from audit_s3_world_increment_supervision_drift import CHECKPOINT, ENDPOINTS, load_endpoint, paired_timeline
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import known_rollout
from ball_retention.world_increment_model import WorldIncrementMLP
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held
from train_s3_world_increment import score_candidate


PHASE = 'world_increment_pitch_drift'
EXTENDED = Path('runs/s3_world_increment_extended_window_fit_20261117')
LABELS = ('pitch_pulse', 'pitch_reversal', 'dual_axis_pulse')
PREFIX = 8
FRAME_SYSTEMS = ('integrated_state_errors', 'integrated_world_errors')


def load_extended(plan, held, anchor, guard):
    def read(name):
        return json.loads(bound(plan, EXTENDED/name))

    status = read('validation.json')
    if (status['status'] != 'completed' or status['updates'] != 900 or status['training_pairs'] != 49
            or not status['extended_window_preflight_reproduced'] or any(status[k] != v for k, v in SCOPE.items())):
        raise ValueError('unexpected full-window endpoint')
    for name in (CHECKPOINT, 'timed_rollouts.json', 'comparison.json'):
        if plan['files_sha256'][(EXTENDED/name).as_posix()] != status['artifacts_sha256'][name]:
            raise ValueError(f'full-window artifact changed: {name}')
    with np.load(ROOT/EXTENDED/CHECKPOINT, allow_pickle=False) as saved:
        if int(saved['updates']) != status['updates']:
            raise ValueError('full-window update count changed')
        model = WorldIncrementMLP.from_state_dict(saved)
        fold = SimpleNamespace(known_mean=saved['known_mean'], known_scale=saved['known_scale'])
    score, rollout = score_candidate(model, held, fold, anchor, guard)
    if score != read('comparison.json')['held_timing'] or rollout != read('timed_rollouts.json'):
        raise ValueError('full-window score or trajectory did not reproduce')
    return model, fold, score, rollout, status['sampler_sha256']


def compare_prefix(traces, endpoints, measured):
    pulse, reversal = LABELS[:2]
    for sign in ('plus', 'minus'):
        a, b = (traces[f'{label}_{sign}'] for label in (pulse, reversal))
        if a['commands'][:PREFIX] != b['commands'][:PREFIX]:
            raise ValueError('executed pitch prefix changed')
        np.testing.assert_array_equal(a['actions'][:PREFIX], b['actions'][:PREFIX])
        if np.array_equal(a['actions'][PREFIX], b['actions'][PREFIX]):
            raise ValueError('expected a change of request at lead nine')
        require_close(a['raw'][:PREFIX], b['raw'][:PREFIX])
    predictions = {'newton': measured[pulse][PREFIX-1].tolist()}
    for key, (_, _, _, rollout, _) in endpoints.items():
        a, b = (rollout['pairs'][label] for label in (pulse, reversal))
        for term in ('plus', 'minus', 'jvp'):
            require_close(np.asarray(a[term])[:PREFIX], np.asarray(b[term])[:PREFIX])
        predictions[key] = half(np.stack([a[side][PREFIX-1] for side in ('plus', 'minus')])).tolist()
    return predictions


def difference_summary(a, b):
    return {'vector_m': (np.asarray(a['vector_m'])-b['vector_m']).tolist(),
            'projection_m': a['projection_m']-b['projection_m'],
            'window_projection_m': {w: x-b['window_projection_m'][w] for w, x in a['window_projection_m'].items()}}


def verify_rebuild(components, target):
    require_close(sum(x['projection_m'] for x in components.values()), target['projection_m'])
    for w, value in target['window_projection_m'].items():
        require_close(sum(x['window_projection_m'][w] for x in components.values()), value)
    return require_close(sum(np.asarray(x['vector_m']) for x in components.values()), np.asarray(target['vector_m']))


def run(plan, out, guard):
    if plan['updates'] != 0 or tuple(plan['cases']) != LABELS or plan['shared_prefix_steps'] != PREFIX:
        raise ValueError('unexpected pitch-diagnostic scope')
    held = read_held(json.loads(bound(plan, plan['held_design'])))
    capture, cfg, fields, traces, _ = held
    if (capture['case']['episode'] != 'validation_lateral' or capture['case']['tick'] != 168
            or tuple(capture['directions']) != LABELS or len(traces) != 8):
        raise ValueError('unexpected archived development capture')
    state = _parse_state(fields)
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), cfg['initial']['anchor_xy_tool_m'])[0]
    endpoints = {key: load_endpoint(plan, key, held, anchor, guard) for key in ENDPOINTS}
    endpoints['extended'] = load_extended(plan, held, anchor, guard)
    reference = endpoints['reference']
    dt = reference[0].dt_s
    for model, fold, _, _, sampler in endpoints.values():
        if dt != model.dt_s or sampler != reference[-1]:
            raise ValueError('unmatched endpoint clock or sampler')
        for attribute in ('known_mean', 'known_scale'):
            np.testing.assert_array_equal(getattr(fold, attribute), getattr(reference[1], attribute))
    previous = json.loads(bound(plan, plan['accepted_previous_decomposition']))['cases']
    settings = {'floors_si': {'position': capture['gates']['position_floor_m'],
                              'velocity': capture['gates']['velocity_floor_m_s']},
                'consecutive_leads': 3, 'relative_error_limit': .5,
                'gravity_step': np.array([0., 0., -cfg['physics']['gravity_m_s2']*dt])}
    branches = {label: [f'{label}_{side}' for side in ('plus', 'minus')] for label in LABELS}
    measured = {label: half(np.stack([traces[k]['raw'] for k in keys])) for label, keys in branches.items()}
    prefix = compare_prefix(traces, endpoints, measured)
    axes = {k: v[-1, :3]/np.linalg.norm(v[-1, :3]) for k, v in measured.items()}
    pulse, reverse = LABELS[:2]
    reversal_effect = measured[reverse][:, :3]-measured[pulse][:, :3]
    contrast_axis = reversal_effect[-1]/np.linalg.norm(reversal_effect[-1])
    cases = {label: {} for label in LABELS}
    contrasts = {}
    max_close = 0.
    for tag, (model, fold, scores, rollout, _) in endpoints.items():
        contrast_terms = {}
        for label, keys in branches.items():
            raw = known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                                np.stack([traces[k]['actions'] for k in keys]), cfg, dt)
            features = model.features(raw, cfg, anchor, fold.known_mean, fold.known_scale)
            result = audit(model, features, state, traces, keys, axes[label], settings)
            expected = np.asarray([rollout['pairs'][label][s] for s in ('plus', 'minus')])
            require_close(result.pop('free_branch_states'), expected)
            result.pop('local_branch_states')
            if tag in ENDPOINTS:
                for system in FRAME_SYSTEMS:
                    if result[system] != previous[label]['models'][tag][system]:
                        raise ValueError('previous endpoint decomposition did not reproduce')
            result['timeline'] = paired_timeline(traces, keys, rollout, label, scores[label], axes[label])
            cases[label][tag] = result
            if label in (pulse, reverse):
                contrast_terms[label] = audit(model, features, state, traces, keys, contrast_axis, settings)
            guard(label, log=True)
        a, b = contrast_terms[reverse], contrast_terms[pulse]
        contrasts[tag] = {key: {term: difference_summary(summary, b[key][term]) for term, summary in a[key].items()}
                          for key in FRAME_SYSTEMS}
        contrasts[tag]['position_error'] = difference_summary(a['position_error'], b['position_error'])
        effects = [half(np.asarray(contrast_terms[label]['free_branch_states']))[:, :3] for label in (reverse, pulse)]
        change = effects[0]-effects[1]
        require_close(change[:PREFIX], np.zeros((PREFIX, 3)))
        require_close(contrasts[tag]['position_error']['vector_m'], change[-1]-reversal_effect[-1])
        contrasts[tag]['predicted_reversal_effect'] = describe(
            np.diff(change, axis=0, prepend=np.zeros((1, 3))), contrast_axis)
        for system in FRAME_SYSTEMS:
            max_close = max(max_close, verify_rebuild(contrasts[tag][system], contrasts[tag]['position_error']))
    for label in LABELS:
        a, b = cases[label]['extended'], cases[label]['supervised']
        change = {key: {term: difference_summary(summary, b[key][term]) for term, summary in a[key].items()}
                  for key in FRAME_SYSTEMS}
        change['position_error'] = difference_summary(a['position_error'], b['position_error'])
        for system in FRAME_SYSTEMS:
            max_close = max(max_close, verify_rebuild(change[system], change['position_error']))
        cases[label]['extended_minus_supervised'] = change
    write_json(out/'pitch_drift.json', {'development_case': capture['case'], 'dt_s': dt, 'windows': WINDOWS,
        'h8_shared_position_velocity_spin_half_difference_si': prefix,
        'cases': {k: {'axis': axes[k].tolist(), 'results': v} for k, v in cases.items()},
        'reversal_minus_pulse': {'axis': contrast_axis.tolist(), 'models': contrasts,
            'newton_effect': describe(np.diff(reversal_effect, axis=0, prepend=np.zeros((1, 3))), contrast_axis)}})
    return {'fixed_endpoints': len(endpoints), 'physical_branch_labels': len(traces), 'identical_pitch_prefix_steps': PREFIX,
            'newton_model_and_jvp_prefix_verified': True, 'old_scores_and_decompositions_reproduced': True,
            'max_difference_closure_m': max_close}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
