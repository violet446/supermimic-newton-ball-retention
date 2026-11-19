"""Compare the trained model's action responses with matched Newton branches."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.action_eval import compare_scales, evaluate_pair, full_step_states
from ball_retention.action_grad import action_sensitivity, predict_ball
from ball_retention.model_data import KNOWN_NAMES, known_rollout
from ball_retention.world_model import ResidualMLP


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_channel_contribution(model, raw, state, nominal, cfg, mean, scale, leads, jacobian):
    """Separate the direct request inputs from the actuator-mediated features."""
    features = known_rollout(state['time_s'], state['tilt_rad'], state['tilt_rate_rad_s'],
                             nominal, cfg, model.dt_s)
    known = ((features-mean)/scale)[None]
    initial = model.normalize(raw)[None]
    columns = [KNOWN_NAMES.index(name) for name in ('roll_request','pitch_request')]
    result = {}
    for h in leads:
        direct = np.empty((3,2))
        for component in range(3):
            cotangent = np.zeros((1,len(nominal),9))
            cotangent[0,h-1,component] = model.state_scale[component]
            _, feature_grad, _ = model.rollout_vjp(initial,known,cotangent)
            direct[component] = feature_grad[0][:,columns].sum(axis=0)/scale[columns]
        result[str(h)] = {'direct_request_dp_m_per_unit':direct.tolist(),
                          'tool_features_dp_m_per_unit':(jacobian[h-1,:3]-direct).tolist()}
    return result


def evaluate(out):
    recovering = (out/'recovery_plan.json').exists()
    plan = json.loads((out/('recovery_plan.json' if recovering else 'plan.json')).read_text())
    collected = json.loads((out/('recovery_collection.json' if recovering else 'collection.json')).read_text())
    if (len(collected['runs']) != len(plan['cases']) or collected['physics_steps']>plan['max_physics_steps']
            or collected['wall_s']>plan['max_wall_s'] or (recovering and not collected['capture_completed'])):
        raise ValueError('incomplete or out-of-budget branch capture')
    checkpoint = ROOT/plan['checkpoint']
    if digest(checkpoint) != plan['checkpoint_sha256']:
        raise ValueError('model checkpoint changed')
    with np.load(checkpoint, allow_pickle=False) as saved:
        model = ResidualMLP.from_state_dict(saved)
        mean, scale = saved['known_mean'], saved['known_scale']
    leads = plan['leads']
    horizon = plan['horizon']
    summaries, resources = [], []
    groups, finite_groups = {}, {}
    for case in plan['cases']:
        scenario = case['scenario']
        folder = out/case.get('directory',scenario)
        cfg_file = out/case.get('config',scenario+'_config.json')
        result = json.loads((folder/'result.json').read_text())
        if (result['status']!=case.get('worker_status','completed') or digest(cfg_file)!=case['config_sha256']
                or result['config_sha256']!=case['config_sha256']):
            raise ValueError('Newton run used a different configuration')
        cfg = json.loads(cfg_file.read_text())
        substeps = round(model.dt_s/cfg['physics']['dt_s'])
        resources.extend(json.loads(line) for line in (folder/'resources.jsonl').read_text().splitlines())
        trace = folder/case.get('data_file','action_pairs.jsonl')
        if 'data_sha256' in case and digest(trace)!=case['data_sha256']:
            raise ValueError('preserved snapshot trace changed')
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        if [r['snapshot_tick'] for r in rows] != case['snapshot_ticks']:
            raise ValueError('captured snapshot set differs from the plan')
        for row in rows:
            amplitudes = plan['amplitudes'][:1] if row['snapshot_tick'] in case.get('single_scale_ticks',[]) else plan['amplitudes']
            if row.get('amplitudes',amplitudes)!=amplitudes:
                raise ValueError('captured perturbation scales differ from the plan')
            state = row['initial_state']
            raw = row['initial_raw']
            nominal = np.asarray(row['nominal_actions'])
            if nominal.shape != (horizon,2):
                raise ValueError('nominal action schedule has wrong length')
            branches = {(b['axis'],b['offset'],b['repeat']):b for b in row['branches']}
            if len(branches) != len(row['branches']) or len(branches) != 2+4*len(amplitudes):
                raise ValueError('missing or duplicated branch')

            def matched(branch, expected):
                states = branch['states_raw']
                n = len(states)
                if not 1<=n<=horizon or len(branch['substeps'])!=n or len(branch['elapsed_s'])!=n:
                    raise ValueError('malformed Newton branch trace')
                applied = np.asarray(branch['actions'])
                if not (np.array_equal(applied,expected[:n]) or np.array_equal(applied,expected)):
                    raise ValueError('Newton and model received different requested actions')
                np.testing.assert_allclose(branch['elapsed_s'], np.cumsum(branch['substeps'])*cfg['physics']['dt_s'],
                                           rtol=0,atol=1e-10)
                return full_step_states(states,branch['substeps'],substeps)

            args = (model, raw, state['time_s'], state['tilt_rad'], state['tilt_rate_rad_s'])
            derivative = action_sensitivity(*args,nominal,cfg,mean,scale)
            eps = plan['model_difference_step']
            numeric = np.stack([(predict_ball(*args,nominal+eps*d,cfg,mean,scale)
                                -predict_ball(*args,nominal-eps*d,cfg,mean,scale))/(2*eps)
                                for d in np.eye(2)],axis=-1)
            np.testing.assert_allclose(derivative['jacobian'],numeric,rtol=plan['model_relative_tolerance'],
                                       atol=plan['model_absolute_tolerance'])
            base_a = matched(branches[None,0.,0],nominal)
            base_b = matched(branches[None,0.,1],nominal)
            pairs, stability = [], []
            for axis in (0,1):
                axis_pairs = []
                for size in amplitudes:
                    delta = np.eye(2)[axis]*size
                    pos, neg = nominal+delta,nominal-delta
                    actual_plus = matched(branches[axis,size,0],pos)
                    actual_minus = matched(branches[axis,-size,0],neg)
                    model_plus = predict_ball(*args,pos,cfg,mean,scale)
                    model_minus = predict_ball(*args,neg,cfg,mean,scale)

                    def compare(plus, minus):
                        return evaluate_pair(plus,minus,actual_plus,actual_minus,size,
                            plan['basin_center_tool_xy_m'], replay_first=base_a,replay_second=base_b,
                            leads=leads,position_floor_m=plan['position_signal_floor_m'],noise_multiplier=plan['noise_multiplier'])

                    paired = compare(model_plus, model_minus)
                    center = derivative['prediction']
                    linear_offset = size*derivative['jacobian'][...,axis]
                    analytic_pair = compare(center+linear_offset, center-linear_offset)
                    axis_pairs.append(paired)
                    pairs.append({'axis':axis,'epsilon':size,'analytic':analytic_pair,'finite_model':paired})
                    for h in leads:
                        for comparison, results in ((analytic_pair,groups),(paired,finite_groups)):
                            stats = comparison['by_lead'][str(h)]
                            counter = results.setdefault((scenario,h),Counter())
                            counter[stats['status']] += 1
                            if stats['status']=='comparable':
                                counter[stats['position']['direction']] += 1
                                sign = stats['progress_xy']['sign_agrees']
                                counter['progress_same' if sign is True else 'progress_opposite' if sign is False else 'progress_unresolved'] += 1
                if len(axis_pairs)==2:
                    stability.append({'axis':axis,**compare_scales(*axis_pairs)})
            summaries.append({'scenario':scenario,'tick':row['snapshot_tick'],'time_s':state['time_s'],
                              'amplitudes':amplitudes,'baseline_noise_raw_max':row['baseline_noise_raw_max'],
                              'model_jacobian_relative_l2_difference':float(np.linalg.norm(derivative['jacobian']-numeric)/max(np.linalg.norm(numeric),1e-15)),
                              'model_analytic_jacobian':derivative['jacobian'].tolist(),
                              'request_channel_contribution':request_channel_contribution(
                                  model,raw,state,nominal,cfg,mean,scale,leads,derivative['jacobian']),
                              'pairs':pairs,'scale_stability':stability})
    summary = {'status':'completed','physics_steps':collected['physics_steps'],'wall_s':collected['wall_s'],
               'attempts':collected['runs'],'recovery_used':recovering,
               'snapshots':summaries,
               'counts':{scenario:{str(h):dict(groups[scenario,h]) for h in leads} for scenario in ('route','edge')},
               'finite_model_counts':{scenario:{str(h):dict(finite_groups[scenario,h]) for h in leads} for scenario in ('route','edge')},
               'resource_peak_gib':{key:max(row[key] for row in resources)/1024**3 for key in
                                    ('peak_rss_bytes','peak_private_commit_bytes','gpu_used_bytes','output_cache_bytes')},
               'scope':'analytic gradient at nominal action vs Newton paired slopes; finite model pairs also recorded; horizons and perturbation sizes share snapshots',
               'progress_objective':'negative XY distance to the S1 stable basin start in tool coordinates'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return summary


if __name__ == '__main__':
    summary = evaluate(ROOT/'runs/s3_action_pairs_20261112')
    print(json.dumps({k:summary[k] for k in ('status','physics_steps','wall_s','counts','resource_peak_gib')},indent=2))
