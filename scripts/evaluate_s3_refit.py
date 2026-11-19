"""Compare frozen and paired-refit models on the same validation snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.action_eval import evaluate_pair, full_step_states
from ball_retention.action_grad import action_sensitivity, predict_ball
from ball_retention.model_data import KNOWN_NAMES, prepare_dataset
from ball_retention.model_eval import evaluate_windows
from ball_retention.world_model import STATE_DIM, ResidualMLP

PAIR_LEADS = (8, 16, 32)
ROUTE_LEADS = (1, 4, 8)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path, tool_only):
    with np.load(path, allow_pickle=False) as data:
        if tuple(data['known_names'].tolist()) != KNOWN_NAMES:
            raise ValueError(f'known feature order changed: {path}')
        mean, scale = data['known_mean'], data['known_scale']
        if (mean.shape != (len(KNOWN_NAMES),) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError(f'invalid feature normalization: {path}')
        model = ResidualMLP.from_state_dict(data)
        request_columns = [KNOWN_NAMES.index(name) for name in ('roll_request', 'pitch_request')]
        expected_mask = np.ones(len(KNOWN_NAMES), dtype=bool)
        expected_mask[request_columns] = not tool_only
        if ('feature_keep_mask' in data and not np.array_equal(data['feature_keep_mask'], expected_mask)):
            raise ValueError(f'feature mask disagrees with the comparison arm: {path}')
        if tool_only and np.any(model.params['W1'][np.asarray(request_columns)+STATE_DIM]):
            raise ValueError('tool-only checkpoint has nonzero direct request weights')
    return model, mean, scale


def summarize_pair_lead(rows, epsilon):
    comparable = [row for row in rows if row['status']=='comparable']
    positions = [row['position'] for row in comparable]
    physics_resolved = [m for m in positions if m['newton_resolved']]
    directions = {name:sum(m['direction']==name for m in positions) for name in
                  ('aligned','opposed','orthogonal','model_below_floor','newton_below_floor','both_below_floor')}
    gain = [m['model_norm']/m['newton_norm'] for m in physics_resolved]
    cosines = [m['cosine'] for m in positions if m['cosine'] is not None]
    progress = [row['progress_xy']['sign_agrees'] for row in comparable]
    # Derivatives are per normalized unit. Recover the half-spread in meters
    # through the norm of the derivative difference times the common epsilon.
    absolute = [np.linalg.norm(np.asarray(m['model_derivative'])-m['newton_derivative'])
                *epsilon*1e6 for m in positions]
    return {'pairs':len(rows),'comparable':len(comparable),'missing':len(rows)-len(comparable),
            'newton_resolved':len(physics_resolved),'both_resolved':len(cosines),
            'direction_counts':directions,'median_cosine':float(np.median(cosines)) if cosines else None,
            'median_gain_ratio':float(np.median(gain)) if gain else None,
            'gain_ratio_range':[float(min(gain)),float(max(gain))] if gain else None,
            'median_resolved_relative_error':float(np.median([m['normalized_error'] for m in physics_resolved])) if physics_resolved else None,
            'position_half_spread_error_rmse_um':float(np.sqrt(np.mean(np.square(absolute)))) if absolute else None,
            'newton_half_spread_range_um': [float(1e6*min(m['newton_half_spread'] for m in positions)),
                                            float(1e6*max(m['newton_half_spread'] for m in positions))] if positions else None,
            'model_half_spread_range_um': [float(1e6*min(m['model_half_spread'] for m in positions)),
                                           float(1e6*max(m['model_half_spread'] for m in positions))] if positions else None,
            'progress_toward_basin':{'same':sum(s is True for s in progress),
                                     'opposite':sum(s is False for s in progress),
                                     'unresolved':sum(s is None for s in progress)}}


def observed_states(states, counts, control_substeps):
    """Keep the shortened terminal row distinct from unexecuted leads."""
    executed = np.count_nonzero(counts)
    return full_step_states(states[:executed],counts[:executed],control_substeps)


def paired_evaluation(model, mean, scale, pairs, config, epsilon, basin, control_substeps, leads=PAIR_LEADS):
    initial = model.normalize(pairs.initial)
    plus = model.denormalize(model.rollout(initial,(pairs.plus_known-mean)/scale))
    minus = model.denormalize(model.rollout(initial,(pairs.minus_known-mean)/scale))
    cases = []
    for i,(state,axis) in enumerate(zip(pairs.initial_state,pairs.axes,strict=True)):
        action = pairs.nominal_actions[i]
        args = (model,pairs.initial[i],state['time_s'],state['tilt_rad'],state['tilt_rate_rad_s'])
        sensitivity = action_sensitivity(*args,action,config,mean,scale)
        fd_step = 1e-5
        shift = fd_step*np.eye(2)[axis]
        numeric = (predict_ball(*args,action+shift,config,mean,scale)-
                   predict_ball(*args,action-shift,config,mean,scale))/(2*fd_step)
        analytic = sensitivity['jacobian'][:,:,axis]
        np.testing.assert_allclose(analytic,numeric,rtol=5e-6,atol=1e-8)
        actual_plus = observed_states(pairs.plus_target[i],pairs.plus_substeps[i],control_substeps)
        actual_minus = observed_states(pairs.minus_target[i],pairs.minus_substeps[i],control_substeps)
        replays = [values[i][pairs.baseline_mask[i]]
                   for values in (pairs.baseline_first,pairs.baseline_second)]

        def compare(a,b):
            return evaluate_pair(a,b,actual_plus,actual_minus,epsilon,basin,
                                 replay_first=replays[0],replay_second=replays[1],leads=leads)

        center = sensitivity['prediction']
        delta = epsilon*analytic
        cases.append({'group_id':str(pairs.group_ids[i]),'episode_id':str(pairs.episode_ids[i]),
                      'snapshot_tick':int(pairs.snapshot_ticks[i]),'axis':int(axis),
                      'jacobian_fd_relative_l2':float(np.linalg.norm(analytic-numeric)/max(np.linalg.norm(numeric),1e-15)),
                      'finite':compare(plus[i],minus[i]),'analytic':compare(center+delta,center-delta)})
    by_lead = {}
    for lead in map(str,leads):
        by_lead[lead] = {kind:summarize_pair_lead([case[kind]['by_lead'][lead] for case in cases],epsilon)
                         for kind in ('finite','analytic')}
    return {'by_lead':by_lead,'cases':cases}


def route_evaluation(model, mean, scale, old, complete_ids):
    result = {}
    for lead in ROUTE_LEADS:
        w = old.validation[lead]
        use = np.isin(w.episode_ids,complete_ids)
        prediction = model.denormalize(model.rollout(model.normalize(w.initial[use]),(w.known[use]-mean)/scale))
        row = evaluate_windows(prediction,w.target[use],w.initial[use],old.dt_s,w.episode_ids[use],leads=(lead,))
        result[str(lead)] = row['by_lead'][str(lead)]
    return result


def evaluate(out):
    from ball_retention.pair_data import prepare_pairs

    manifest = out/'capture_manifest.json'
    capture = json.loads(manifest.read_text(encoding='utf-8'))
    pairs = prepare_pairs(manifest)
    if pairs.horizon != max(PAIR_LEADS):
        raise ValueError('capture horizon differs from the paired comparison')
    s2 = ROOT/'runs/s2_world_model_20261112'
    plan = json.loads((s2/'plan.json').read_text(encoding='utf-8'))['config']
    dataset = ROOT/plan['dataset']
    dataset_manifest = dataset/'manifest.json'
    if digest(dataset_manifest) != plan['dataset_manifest_sha256']:
        raise ValueError('S2 validation manifest changed')
    old_entries = json.loads(dataset_manifest.read_text(encoding='utf-8'))['entries']
    old_groups = {row['group_id']:row['split'] for row in old_entries}
    if any(row['group_id'] in old_groups and old_groups[row['group_id']] != row['split']
           for row in capture['entries']):
        raise ValueError('capture group crossed the S2 anchor split')
    old = prepare_dataset(dataset,horizons=ROUTE_LEADS)
    complete_ids = [row['episode_id'] for row in old_entries
                    if row['split']=='validation' and row['truncated'] and not row['terminated']]
    original = s2/'world_model.npz'
    s2_sha = json.loads((s2/'summary.json').read_text(encoding='utf-8'))['checkpoint_sha256']
    if digest(original) != s2_sha:
        raise ValueError('frozen S2 checkpoint changed')
    prior_plan = ROOT/'runs/s3_action_pairs_20261112/plan.json'
    basin = json.loads(prior_plan.read_text(encoding='utf-8'))['basin_center_tool_xy_m']
    models = {}
    for name,path in [('s2',original),('raw_request',out/'fit/raw_request.npz'),('tool_only',out/'fit/tool_only.npz')]:
        model, mean, scale = load_checkpoint(path,tool_only=name=='tool_only')
        if not (model.dt_s==pairs.dt_s==old.dt_s):
            raise ValueError(f'control interval changed: {path}')
        models[name] = {'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':digest(path),
                        'pairs':paired_evaluation(model,mean,scale,pairs.validation,pairs.model_config,pairs.amplitude,
                                                  basin,capture['control_substeps']),
                        'complete_route':route_evaluation(model,mean,scale,old,complete_ids)}
    val = pairs.validation
    snapshots = {(str(e),int(t)) for e,t in zip(val.episode_ids,val.snapshot_ticks,strict=True)}
    report = {'scope':'Validation groups held out from gradient updates. Validation may select checkpoints. No test split scores.',
              'analytic_comparison':'Position Jacobian at nominal commands compared with Newton finite half-spread at +/- epsilon; basin progress from linearized positions.',
              'finite_comparison':'Model and Newton use matched positive/negative action sequences from the same snapshot.',
              'gain_definition':'Model position half-spread norm divided by Newton norm for Newton-resolved pairs.',
              'capture_manifest_sha256':digest(manifest),'old_dataset_manifest_sha256':digest(dataset_manifest),
              'epsilon':pairs.amplitude,'control_dt_s':pairs.dt_s,'leads':list(PAIR_LEADS),
              'evaluation_source_sha256':digest(Path(__file__)),
              'basin_center_tool_xy_m':basin,'basin_source':prior_plan.relative_to(ROOT).as_posix(),
              'validation_snapshot_count':len(snapshots),'validation_pair_count':len(val.axes),
              'validation_groups':sorted({str(group) for group in val.group_ids}),
              'old_complete_route_episode_ids':complete_ids,'models':models}
    path = out/'comparison.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,model in models.items():
        lead = model['pairs']['by_lead'][str(PAIR_LEADS[-1])]
        verdict = ', '.join(f'{kind} {stats["direction_counts"]["aligned"]}/{stats["newton_resolved"]} aligned, '
                            f'gain={stats["median_gain_ratio"]}' for kind,stats in lead.items())
        route_rmse = model['complete_route']['8']['model']['position_rmse_mm']
        print(f'{name}: 32 steps {verdict}; old complete route 8-step position RMSE {route_rmse:.3f} mm')
    print(path)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/s3_pair_refit_20261112')
    args = parser.parse_args()
    directory = (ROOT/args.run).resolve()
    directory.relative_to(ROOT/'runs')
    evaluate(directory)
