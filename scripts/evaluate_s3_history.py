"""Evaluate the predeclared frozen model on the crossed-history capture."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ball_retention.action_eval import evaluate_pair
from ball_retention.pair_data import prepare_pairs
from ball_retention.pair_collection import prepare_collection
from ball_retention.world_model import ResidualMLP
from ball_retention.guard import process_memory
from ball_retention.resources import directory_bytes
from evaluate_s3_coverage import summarize,digest,write


def aggregate(rows,leads,cosine_min,strong_multiplier):
    result={}
    for lead in leads:
        comparable=[r['response']['by_lead'][str(lead)] for r in rows
                    if r['response']['by_lead'][str(lead)]['status']=='comparable']
        p=[r['position'] for r in comparable]
        strong=[m for m in p if m['newton_half_spread']>=strong_multiplier*m['signal_threshold']]
        count=sum(r['absolute'][str(lead)]['branches'] for r in rows)
        square=np.sum([r['absolute'][str(lead)]['squared_p_v_w_si'] for r in rows],axis=0)
        result[str(lead)]={'position_all':summarize(p,cosine_min),'position_strong':summarize(strong,cosine_min),
            'velocity':summarize([r['velocity'] for r in comparable],cosine_min),
            'spin':summarize([r['angular_velocity'] for r in comparable],cosine_min),
            'absolute_rmse_p_v_w_si':np.sqrt(square/count).tolist() if count else None,
            'branches':count,'unavailable_pairs':len(rows)-len(comparable)}
    return result


def run(out):
    started=time.monotonic();plan=json.loads((out/'evaluation_plan.json').read_text())
    progress=json.loads((out/'capture_progress.json').read_text())
    if progress['status']!='completed':raise ValueError('complete capture required')
    if (out/'frozen_comparison.json').exists():raise ValueError('preserve existing frozen evaluation')
    checkpoint=ROOT/plan['model_path']
    if digest(checkpoint)!=plan['model_sha256']:raise ValueError('frozen model changed')
    capture=json.loads((out/'capture_plan.json').read_text())
    for path,expected in capture['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('capture source changed: '+path)
    manifest=json.loads((out/'capture_manifest.json').read_text());episodes={e['episode_id']:e for e in manifest['entries']}
    data=prepare_pairs(out/'capture_manifest.json');merged=prepare_collection(ROOT,out/'merged_collection.json')
    write(out/'evaluation_execution.json',{'source_sha256':digest(Path(__file__)),
        'summary_helper_sha256':digest(Path(__file__).with_name('evaluate_s3_coverage.py')),
        'evaluation_plan_sha256':digest(out/'evaluation_plan.json'),'capture_manifest_sha256':digest(out/'capture_manifest.json'),
        'checkpoint_sha256':digest(checkpoint),'teacher_forcing':False,'training_updates':0,'checkpoint_selection':False})
    with np.load(checkpoint,allow_pickle=False) as saved:
        model=ResidualMLP.from_state_dict(saved);km=saved['known_mean'];ks=saved['known_scale']
    if model.dt_s!=data.dt_s:raise ValueError('model/capture clock mismatch')
    rows=[]
    for split in ('train','validation'):
        b=getattr(data,split)
        if not (b.plus_mask.all() and b.minus_mask.all() and b.baseline_mask.all()):
            raise ValueError('frozen pilot requires complete branches and replays')
        known=np.concatenate((b.plus_known,b.minus_known));initial=np.concatenate((b.initial,b.initial))
        pred=model.denormalize(model.rollout(model.normalize(initial),(known-km)/ks)).reshape(2,-1,data.horizon,9)
        for i in range(len(b.initial)):
            response=evaluate_pair(pred[0,i],pred[1,i],b.plus_target[i],b.minus_target[i],data.amplitude,
                data.model_config['initial']['anchor_xy_tool_m'],replay_first=b.baseline_first[i],
                replay_second=b.baseline_second[i],leads=plan['leads'])
            absolute={}
            for lead in plan['leads']:
                error=np.stack((pred[0,i,lead-1]-b.plus_target[i,lead-1],pred[1,i,lead-1]-b.minus_target[i,lead-1]))
                absolute[str(lead)]={'branches':2,'squared_p_v_w_si':np.sum(error.reshape(2,3,3)**2,axis=(0,2)).tolist()}
            entry=episodes[str(b.episode_ids[i])]
            rows.append({'split':split,'group':str(b.group_ids[i]),'history':entry['history_id'],'episode':str(b.episode_ids[i]),
                'tick':int(b.snapshot_ticks[i]),'axis':int(b.axes[i]),'amplitude':data.amplitude,'response':response,'absolute':absolute})
        memory=process_memory(os.getpid())
        if time.monotonic()-started>plan['wall_s'] or memory['rss_bytes']>plan['rss_gib']*1024**3 or memory['private_commit_bytes']>plan['commit_gib']*1024**3:
            raise RuntimeError('evaluation resource budget exceeded')
    def metrics(selected):return aggregate(selected,plan['leads'],plan['direction_cosine_min'],plan['strong_noise_multiplier'])
    summary={}
    for split in ('train','validation'):
        selected=[r for r in rows if r['split']==split]
        summary[split]={'all':metrics(selected),'by_history':{},'by_group':{}}
        for history in sorted(set(r['history'] for r in selected)):
            summary[split]['by_history'][history]=metrics([r for r in selected if r['history']==history])
        for group in sorted(set(r['group'] for r in selected)):
            group_rows=[r for r in selected if r['group']==group]
            summary[split]['by_group'][group]={'all':metrics(group_rows),
                'by_history':{h:metrics([r for r in group_rows if r['history']==h]) for h in sorted(set(r['history'] for r in group_rows))}}
    write(out/'frozen_comparison.json',{'checkpoint_sha256':plan['model_sha256'],'summary':summary,'pairs':rows,
        'limitations':['Train means new intended training split; this frozen model has not fit any new cases.',
            'Two histories, axes and snapshots share physical initial groups; counts are not independent initial states.',
            'Prediction and finite-perturbation response diagnostics only, not action-gradient or control acceptance.']})
    resources=[];snapshots=branches=0
    for entry in manifest['entries']:
        folder=(out/entry['path']).parent;result=json.loads((folder/'result.json').read_text())
        if result['status']!='completed' or result['worker']['status']!='action_pairs_completed':raise ValueError('worker not complete')
        if result['config_sha256']!=entry['config_sha256']:raise ValueError('worker configuration mismatch')
        resources.extend(json.loads(line) for line in (folder/'resources.jsonl').read_text().splitlines())
        for line in (out/entry['path']).read_text().splitlines():
            record=json.loads(line);snapshots+=1;branches+=len(record['branches'])
            for branch in record['branches']:
                if branch['substeps']!=[16]*32 or branch['terminated'] or branch['truncated']:raise ValueError('branch incomplete')
    if snapshots!=16 or branches!=96 or len(rows)!=32:raise ValueError('frozen size mismatch')
    if digest(checkpoint)!=plan['model_sha256']:raise ValueError('evaluation mutated checkpoint')
    if directory_bytes(out)>512*1024**2:raise RuntimeError('capture output budget exceeded')
    # Keep the prior preparation certificate immutable; this certifies actual capture.
    write(out/'capture_validation.json',{'status':'completed_and_evaluated','worker_starts':8,'snapshots':snapshots,'branches':branches,
        'physics_steps':progress['physics_steps'],'capture_wall_s':progress['wall_s'],'evaluation_wall_s':time.monotonic()-started,
        'source_config_and_data_hashes_verified':True,'frozen_checkpoint_unchanged':True,'merged_collection':merged.report['splits'],
        'peak_rss_bytes':max(r['peak_rss_bytes'] for r in resources),'peak_commit_bytes':max(r['peak_private_commit_bytes'] for r in resources),
        'peak_gpu_used_bytes':max(r['gpu_used_bytes'] for r in resources),'min_available_ram_bytes':min(r['available_ram_bytes'] for r in resources),
        'training_updates':0,'checkpoint_selection':False,'test_metrics_used':False,'control_accepted':False,
        'artifacts_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='capture_validation.json'}})
    print(json.dumps({s:v['all']['32'] for s,v in summary.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'runs/s3_history_coverage_20261114')
    args=parser.parse_args();out=args.output.resolve();out.relative_to(ROOT/'runs');run(out)
