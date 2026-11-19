"""Locate changes in archived development responses at two fixed world-model endpoints."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from s3_fitability_common import ROOT, SCOPE, run_phase, write_json

import numpy as np

from audit_s3_world_increment_drift import WINDOWS, audit, describe, half, require_close
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import known_rollout
from ball_retention.world_increment_model import WorldIncrementMLP
from deterministic_response_metrics import HORIZON, response_metrics
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held
from train_s3_world_increment import score_candidate


PHASE = 'world_increment_supervision_drift'
ENDPOINTS = {
    'reference': Path('runs/s3_world_increment_fit_20261117'),
    'supervised': Path('runs/s3_world_increment_supervision_fit_20261117'),
}
CASES = ('nominal', 'pitch_pulse', 'pitch_reversal', 'dual_axis_pulse')
CHECKPOINT = 'world_increment_900.npz'


def load_endpoint(plan, key, held, anchor, guard):
    root = ENDPOINTS[key]
    status = json.loads(bound(plan, root/'validation.json'))
    if (status['status'] != 'completed' or status['updates'] != 900 or status['training_pairs'] != 49
            or any(status[k] != v for k, v in SCOPE.items())):
        raise ValueError(f'unexpected archived endpoint {key}')
    for name in (CHECKPOINT, 'timed_rollouts.json', 'comparison.json'):
        if plan['files_sha256'][(root/name).as_posix()] != status['artifacts_sha256'][name]:
            raise ValueError(f'archived endpoint artifact changed: {root/name}')
    with np.load(ROOT/root/CHECKPOINT, allow_pickle=False) as saved:
        if int(saved['updates']) != status['updates']:
            raise ValueError('checkpoint update count changed')
        model = WorldIncrementMLP.from_state_dict(saved)
        fold = SimpleNamespace(known_mean=saved['known_mean'], known_scale=saved['known_scale'])
    score, rollout = score_candidate(model, held, fold, anchor, guard)
    recorded = json.loads(bound(plan, root/'comparison.json'))
    if score != recorded['held_timing'] or rollout != json.loads(bound(plan, root/'timed_rollouts.json')):
        raise ValueError(f'archived development scoring failed to reproduce for {key}')
    return model, fold, score, rollout, status['sampler_sha256']


def terminal_start(by_lead, negative):
    """First lead in the uninterrupted negative run that reaches H32."""
    active = [i for i, row in enumerate(by_lead, 1) if negative(row)]
    terminal = None
    for expected, lead in zip(range(HORIZON, 0, -1), reversed(active)):
        if lead != expected:
            break
        terminal = lead
    return {'opposite_leads': active, 'terminal_opposite_run_start': terminal}


def paired_timeline(traces, keys, rollout, label, frozen, axis):
    branches = rollout['pairs'][label]
    nominal = np.stack([traces[k]['raw'] for k in ('nominal_1', 'nominal_2')])
    score = response_metrics(nominal, *(traces[k]['raw'] for k in keys),
        rollout['nominal'], branches['plus'], branches['minus'], branches['jvp'], frozen['amplitude'],
        floors=tuple(frozen['absolute_floors_si'][key] for key in ('position', 'velocity', 'spin')),
        leads=range(1, HORIZON+1),
        **{key: frozen[key] for key in ('noise_multiplier', 'strong_multiplier', 'min_cosine', 'max_relative_error')})
    if any(score['by_lead'][k] != v for k, v in frozen['by_lead'].items()):
        raise ValueError('dense scoring changed a frozen lead')
    rows = []
    for lead, scores in score['by_lead'].items():
        p = scores['position']
        rows.append({'lead': int(lead), 'newton_strong': p['newton_strong'],
            'newton_projection_m': float(np.asarray(p['newton_half_difference']) @ axis),
            'finite_projection_m': float(np.asarray(p['model_fd_half_difference']) @ axis),
            'jvp_projection_m': float(np.asarray(p['model_jvp_linearized_displacement']) @ axis),
            'finite_cosine': p['model_fd_vs_newton']['cosine'], 'jvp_cosine': p['model_jvp_vs_newton']['cosine']})
    reversal = {}
    for model_key in ('finite', 'jvp'):
        reversal[model_key] = terminal_start(rows, lambda r: r['newton_strong']
                                               and r[f'{model_key}_cosine'] is not None and r[f'{model_key}_cosine'] < 0.)
    return {'by_lead': rows, 'opposite_to_current_newton': reversal}


def run(plan, out, guard):
    if (plan['updates'] != 0 or tuple(plan['cases']) != CASES or tuple(plan['endpoints']) != tuple(ENDPOINTS)):
        raise ValueError('unexpected diagnostic scope')
    held = read_held(json.loads(bound(plan, plan['held_design'])))
    capture, cfg, fields, traces, _ = held
    if (capture['case']['tick'] != 168 or tuple(capture['directions']) != CASES[1:]
            or len(traces) != 8):
        raise ValueError('unexpected development capture')
    state = _parse_state(fields)
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), cfg['initial']['anchor_xy_tool_m'])[0]
    endpoints = {key: load_endpoint(plan, key, held, anchor, guard) for key in ENDPOINTS}
    ref, sup = endpoints.values()
    dt = ref[0].dt_s
    if ref[-1] != sup[-1] or dt != sup[0].dt_s:
        raise ValueError('mismatched endpoint sampler or clock')
    for key in ('known_mean', 'known_scale'):
        np.testing.assert_array_equal(getattr(ref[1], key), getattr(sup[1], key))
    settings = {'floors_si': {'position': capture['gates']['position_floor_m'],
                              'velocity': capture['gates']['velocity_floor_m_s']},
                'consecutive_leads': 3, 'relative_error_limit': .5,
                'gravity_step': np.array([0., 0., -cfg['physics']['gravity_m_s2']*dt])}
    require_close(traces['nominal_1']['raw'], traces['nominal_2']['raw'])
    nominal = traces['nominal_1']['raw'][:, :3]
    cases = {}
    all_closures = []
    for label in CASES:
        keys = ('nominal_1',) if label == 'nominal' else tuple(f'{label}_{side}' for side in ('plus', 'minus'))
        truth = nominal if label == 'nominal' else half(np.stack([traces[k]['raw'][:, :3] for k in keys]))
        direction = np.asarray(sup[3]['nominal'])[-1, :3]-truth[-1] if label == 'nominal' else truth[-1]
        axis = direction/np.linalg.norm(direction)
        known = known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                              np.stack([traces[k]['actions'] for k in keys]), cfg, dt)
        reports = {}
        for key, (model, fold, score, rollout, _) in endpoints.items():
            features = model.features(known, cfg, anchor, fold.known_mean, fold.known_scale)
            result = audit(model, features, state, traces, keys, axis, settings)
            free = np.asarray(result.pop('free_branch_states'))
            result.pop('local_branch_states')
            expected = [rollout['nominal']] if label == 'nominal' else [rollout['pairs'][label][s] for s in ('plus', 'minus')]
            require_close(free, np.asarray(expected))
            prediction = free[0, :, :3] if label == 'nominal' else half(free[:, :, :3])
            quantity = prediction-truth if label == 'nominal' else prediction
            result['projected_by_lead_m'] = (quantity @ axis).tolist()
            result['projected_by_window'] = describe(np.diff(quantity, axis=0, prepend=np.zeros((1, 3))), axis)
            if label != 'nominal':
                timeline = paired_timeline(traces, keys, rollout, label, score[label], axis)
                require_close([r['finite_projection_m'] for r in timeline['by_lead']], quantity @ axis)
                result['response_timeline'] = timeline
            all_closures.append((result['velocity_closure_m_s'], result['position_closure_m']))
            reports[key] = result
            guard(label, log=True)
        cases[label] = {'projection_axis_tool': axis.tolist(), 'models': reports}
    velocity, position = np.max(all_closures, axis=0)
    write_json(out/'supervision_drift.json', {'dt_s': dt, 'windows': WINDOWS, 'development_case': capture['case'],
        'cases': cases, 'projection': 'Nominal uses supervised H32 error. Paired uses Newton H32 half-difference.',
        'direction_timeline': 'Cosines use Newton half-difference at each lead and only strong Newton responses count toward opposition.'})
    return {'fixed_endpoints': 2, 'development_snapshots': 1, 'physical_branch_labels': len(traces),
            'archived_scores_reproduced': True, 'max_velocity_closure_m_s': float(velocity),
            'max_position_closure_m': float(position)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
