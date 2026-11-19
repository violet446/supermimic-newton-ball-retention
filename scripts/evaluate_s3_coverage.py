"""Frozen-model evaluation on the newly declared initial groups and amplitudes."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ball_retention.action_eval import evaluate_pair, compare_scales
from ball_retention.frame_model import FrameResidualMLP, pack_features
from ball_retention.geometry import load_tool, support_point
from ball_retention.guard import process_memory
from ball_retention.pair_data import prepare_pairs
from ball_retention.resources import directory_bytes
from ball_retention.world_model import ResidualMLP

OUT = ROOT/'runs/s3_coverage_20261113'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def summarize(metrics,cosine_min):
    resolved = [m for m in metrics if m['newton_resolved']]
    ratios = [m['model_half_spread']/m['newton_half_spread'] for m in resolved]
    cosines = [m['cosine'] for m in resolved if m['cosine'] is not None]
    return {'pairs':len(metrics),'resolved':len(resolved),
        'positive_cosine':sum(m['cosine'] is not None and m['cosine']>0 for m in resolved),
        'cosine_ge_0_5':sum(m['cosine'] is not None and m['cosine']>=cosine_min for m in resolved),
        'model_unresolved':sum(not m['model_resolved'] for m in resolved),
        'median_cosine':float(np.median(cosines)) if cosines else None,
        'median_amplitude_ratio':float(np.median(ratios)) if ratios else None,
        'amplitude_ratio_within_half_to_twice':sum(.5<=r<=2. for r in ratios),
        'median_relative_vector_error':float(np.median([m['normalized_error'] for m in resolved])) if resolved else None}


def main():
    started = time.monotonic()
    plan = json.loads((OUT/'evaluation_plan.json').read_text())
    if json.loads((OUT/'capture_progress.json').read_text())['status']!='completed':
        raise ValueError('complete verified capture required')
    data = {a:prepare_pairs(OUT/f'capture_manifest_{a:.2f}.json') for a in plan['amplitudes']}
    first = next(iter(data.values()))
    cfg = first.model_config
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'),cfg['initial']['anchor_xy_tool_m'])[0]
    results = {}
    for name,spec in plan['models'].items():
        path = ROOT/spec['path']
        if digest(path)!=spec['sha256']:
            raise ValueError('model changed since evaluation was frozen')
        frame = name in ('frame_fixed','untrained_frame')
        cls = FrameResidualMLP if frame else ResidualMLP
        with np.load(path,allow_pickle=False) as saved:
            model = cls.from_state_dict(saved)
            mean,scale = saved['known_mean'],saved['known_scale']
        if name=='relative_cv':
            model.params['W3'].fill(0.)
            model.params['b3'].fill(0.)
        rows = []
        for amplitude,prepared in data.items():
            if prepared.model_config!=cfg or model.dt_s!=prepared.dt_s:
                raise ValueError('physical configuration or control clock mismatch')
            for split in ('train','validation'):
                b = getattr(prepared,split)
                raw_known = np.concatenate((b.plus_known,b.minus_known))
                features = (pack_features(raw_known,cfg,model.dt_s,anchor,model.state_mean,model.state_scale,mean,scale)
                            if frame else (raw_known-mean)/scale)
                initial = model.normalize(np.concatenate((b.initial,b.initial)))
                predictions = model.denormalize(model.rollout(initial,features)).reshape(2,-1,prepared.horizon,9)
                for i in range(len(b.initial)):
                    def masked(array,mask):
                        return [row if keep else None for row,keep in zip(array,mask)]
                    result = evaluate_pair(predictions[0,i],predictions[1,i],
                        masked(b.plus_target[i],b.plus_mask[i]),masked(b.minus_target[i],b.minus_mask[i]),amplitude,anchor[:2],
                        replay_first=masked(b.baseline_first[i],b.baseline_mask[i]),
                        replay_second=masked(b.baseline_second[i],b.baseline_mask[i]),leads=plan['leads'])
                    absolute = {}
                    for lead in plan['leads']:
                        k = lead-1
                        differences = [predictions[branch,i,k]-target[i,k] for branch,target,mask in
                            ((0,b.plus_target,b.plus_mask),(1,b.minus_target,b.minus_mask)) if mask[i,k]]
                        absolute[str(lead)] = {'full_branches':len(differences),
                            'squared_error_p_v_w':np.sum(np.array(differences).reshape(-1,3,3)**2,axis=(0,2)).tolist()
                            if differences else [0.,0.,0.]}
                    rows.append({'split':split,'group':str(b.group_ids[i]),'episode':str(b.episode_ids[i]),
                        'tick':int(b.snapshot_ticks[i]),'axis':int(b.axes[i]),'amplitude':amplitude,
                        'response':result,'absolute':absolute})
        summary = {}
        for split in ('train','validation'):
            summary[split] = {}
            for amplitude in plan['amplitudes']:
                select = [r for r in rows if r['split']==split and r['amplitude']==amplitude]
                leads = {}
                for lead in plan['leads']:
                    comparable = [r['response']['by_lead'][str(lead)] for r in select
                                  if r['response']['by_lead'][str(lead)]['status']=='comparable']
                    p = [r['position'] for r in comparable]
                    near = [m for m in p if m['newton_resolved'] and m['newton_half_spread']<10*m['signal_threshold']]
                    strong = [m for m in p if m['newton_half_spread']>=10*m['signal_threshold']]
                    total = sum(r['absolute'][str(lead)]['full_branches'] for r in select)
                    squared = np.sum([r['absolute'][str(lead)]['squared_error_p_v_w'] for r in select],axis=0)
                    leads[str(lead)] = {'position_all':summarize(p,plan['direction_cosine_min']),
                        'position_near_floor':summarize(near,plan['direction_cosine_min']),
                        'position_strong':summarize(strong,plan['direction_cosine_min']),
                        'velocity':summarize([r['velocity'] for r in comparable],plan['direction_cosine_min']),
                        'spin':summarize([r['angular_velocity'] for r in comparable],plan['direction_cosine_min']),
                        'absolute_rmse_p_v_w_si':np.sqrt(squared/total).tolist() if total else None,
                        'full_branches':total,'unavailable_pairs':len(select)-len(comparable)}
                summary[split][str(amplitude)] = leads
        results[name] = {'checkpoint_sha256':spec['sha256'],'summary':summary,'pairs':rows}
        memory = process_memory(os.getpid())
        if (time.monotonic()-started>plan['evaluation_wall_s'] or memory['rss_bytes']>plan['evaluation_rss_gib']*1024**3
                or memory['private_commit_bytes']>plan['evaluation_commit_gib']*1024**3):
            raise RuntimeError('CPU evaluation resource budget exceeded')

    # Newton's change across amplitudes is independent of the frozen model.
    grouped = {}
    for row in next(iter(results.values()))['pairs']:
        key = row['split'],row['group'],row['tick'],row['axis']
        grouped.setdefault(key,{})[row['amplitude']] = row['response']
    consistency = []
    small,large = plan['amplitudes']
    for (split,group,tick,axis),pair in grouped.items():
        comparison = compare_scales(pair[small],pair[large],**plan['scale_consistency'])
        consistency.append({'split':split,'group':group,'tick':tick,'axis':axis,**comparison})
    write(OUT/'frozen_comparison.json',results)
    write(OUT/'scale_consistency.json',consistency)
    if directory_bytes(OUT)>plan['output_cap_mib']*1024**2:
        raise RuntimeError('evaluation output budget exceeded')
    write(OUT/'evaluation_summary.json',{'status':'completed','wall_s':time.monotonic()-started,
        'training_updates':0,'checkpoint_selection':False,'test_used':False,'models':list(results),
        'source_sha256':{Path(__file__).relative_to(ROOT).as_posix():digest(Path(__file__)),
                         'src/ball_retention/pair_data.py':digest(ROOT/'src/ball_retention/pair_data.py')},
        'evaluation_plan_sha256':digest(OUT/'evaluation_plan.json'),**memory})
    for name,result in results.items():
        print(name,{a:v['32']['position_strong'] for a,v in result['summary']['validation'].items()},flush=True)


if __name__=='__main__':
    main()
