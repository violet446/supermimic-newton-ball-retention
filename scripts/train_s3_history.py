"""One fresh, fixed-final-update fit on the expanded training split."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ball_retention.group_fold import make_group_fold
from ball_retention.model_data import prepare_dataset,KNOWN_NAMES
from ball_retention.pair_collection import prepare_collection
from ball_retention.pair_loss import paired_loss
from ball_retention.world_model import ResidualMLP,Adam
from ball_retention.action_eval import evaluate_pair
from ball_retention.guard import process_memory
from ball_retention.resources import snapshot,directory_bytes
from evaluate_s3_history import aggregate
from check_s3_group_generalization import sha,write
OUT=ROOT/'runs/s3_history_refit_20261114'


def report(model,km,ks,collection,data,histories):
    rows=[];leads=[8,16,32]
    for split in ('train','validation'):
        mixed=getattr(collection,split);b=mixed.batch
        k=(np.concatenate((b.plus_known,b.minus_known))-km)/ks
        pred=model.denormalize(model.rollout(model.normalize(np.concatenate((b.initial,b.initial))),k)).reshape(2,-1,32,9)
        for i in range(len(b.initial)):
            def masked(v,m):return [x if valid else None for x,valid in zip(v,m)]
            response=evaluate_pair(pred[0,i],pred[1,i],masked(b.plus_target[i],b.plus_mask[i]),masked(b.minus_target[i],b.minus_mask[i]),
                float(mixed.amplitudes[i]),collection.model_config['initial']['anchor_xy_tool_m'],
                replay_first=masked(b.baseline_first[i],b.baseline_mask[i]),replay_second=masked(b.baseline_second[i],b.baseline_mask[i]),leads=leads)
            absolute={}
            for lead in leads:
                errors=[pred[j,i,lead-1]-target[i,lead-1] for j,target,mask in
                    ((0,b.plus_target,b.plus_mask),(1,b.minus_target,b.minus_mask)) if mask[i,lead-1]]
                absolute[str(lead)]={'branches':len(errors),'squared_p_v_w_si':np.sum(np.array(errors).reshape(-1,3,3)**2,axis=(0,2)).tolist()}
            rows.append({'split':split,'group':str(b.group_ids[i]),'cohort':str(mixed.cohorts[i]),
                'history':histories.get(str(b.episode_ids[i]),'original'),'episode':str(b.episode_ids[i]),
                'tick':int(b.snapshot_ticks[i]),'axis':int(b.axes[i]),'amplitude':float(mixed.amplitudes[i]),'response':response,'absolute':absolute})
    summaries={}
    for field in ('group','cohort','history'):
        for split,label,amp in sorted(set((r['split'],r[field],r['amplitude']) for r in rows)):
            take=[r for r in rows if r['split']==split and r[field]==label and r['amplitude']==amp]
            summaries[f'{field}/{split}/{label}/{amp}']=aggregate(take,leads,.5,10)
    terminal=set(data.validation_endings.episode_ids[data.validation_endings.terminated]);s2={}
    for h,w in data.validation.items():
        prediction=model.denormalize(model.rollout(model.normalize(w.initial),(w.known-km)/ks))[:,-1]
        s2[str(h)]={str(e):np.sqrt(np.mean(np.sum((prediction[w.episode_ids==e]-w.target[w.episode_ids==e,-1]).reshape(-1,3,3)**2,axis=2),axis=0)).tolist() for e in np.unique(w.episode_ids)}
    return {'summary':summaries,'pairs':rows,'s2_validation':s2,'complete_route_ids':sorted(set(data.validation[8].episode_ids)-terminal)}


def regression(before,after):
    checks={}
    for key,value in before['summary'].items():
        if not key.startswith('group/validation/'):continue
        a=value['32'];b=after['summary'][key]['32']
        checks[key+'/position']=b['absolute_rmse_p_v_w_si'][0]<=1.1*a['absolute_rmse_p_v_w_si'][0]+1e-12
        if a['position_strong']['pairs']!=b['position_strong']['pairs']:raise ValueError('strong physical set changed')
        for metric in ('positive_cosine','cosine_ge_0_5'):
            checks[key+'/'+metric]=b['position_strong'][metric]>=a['position_strong'][metric]
    for e in before['complete_route_ids']:
        checks['S2_route/'+e]=after['s2_validation']['8'][e][0]<=1.1*before['s2_validation']['8'][e][0]+1e-12
    if not checks:raise ValueError('empty regression checks')
    return {'passed':all(checks.values()),'checks':checks,'failures':[k for k,v in checks.items() if not v]}


def main():
    OUT.mkdir(exist_ok=False);started=time.monotonic()
    cfg=json.loads((ROOT/'configs/s3_history_refit_plan.json').read_text())
    for path,digest in cfg['inputs_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('plan input changed: '+path)
    sources=[Path(__file__),ROOT/'scripts/evaluate_s3_history.py',ROOT/'scripts/evaluate_s3_coverage.py']+list((ROOT/'src/ball_retention').glob('*.py'))
    write(OUT/'plan.json',{'design_sha256':sha(ROOT/'configs/s3_history_refit_plan.json'),'config':cfg,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources}})
    def guard(update):
        mem=process_memory(os.getpid());machine=snapshot(ROOT);wall=time.monotonic()-started
        with (OUT/'resources.jsonl').open('a') as f:f.write(json.dumps({'update':update,'wall_s':wall,**mem,**machine})+'\n')
        if wall>600 or mem['rss_bytes']>1024**3 or mem['private_commit_bytes']>1.5*1024**3 or machine['available_ram_bytes']<(.75 if update>=0 else 2)*1024**3 or directory_bytes(OUT)>64*1024**2:
            raise RuntimeError('history refit resource cap')
    guard(-1)
    collection=prepare_collection(ROOT,ROOT/cfg['collection']);data=prepare_dataset(ROOT/'data/s2_pilot_20261112')
    if collection.model_config!=data.model_config or collection.dt_s!=data.dt_s:raise ValueError('configuration mismatch')
    manifest=json.loads((ROOT/'data/s2_pilot_20261112/manifest.json').read_text())
    mapping={e['episode_id']:e['group_id'] for e in manifest['entries'] if e['split']=='train'}
    fold=make_group_fold(data.train,collection.train.batch,mapping)
    if len(fold.pairs.initial)!=40 or len(set(fold.pairs.group_ids))!=6:raise ValueError('unexpected training split')
    np.testing.assert_allclose(fold.pair_weights,collection.train.sample_weights,rtol=1e-14)
    write(OUT/'training_data.json',{'fold':fold.report,'collection':collection.report})
    physical=json.loads((ROOT/cfg['physical_scales']).read_text());units=np.array(physical['state_error_scale_si'])
    model=ResidualMLP(20,fold.state_mean,fold.state_scale,data.dt_s,cfg['seed'],residual_gain=np.array(physical['residual_gain_si'])/fold.state_scale)
    known=lambda k:(k-fold.known_mean)/fold.known_scale
    windows={h:(model.normalize(w.initial),known(w.known),model.normalize(w.target)) for h,w in fold.anchors.items()}
    b=fold.pairs;x=model.normalize(b.initial);k=known(np.stack((b.plus_known,b.minus_known)))
    target=np.stack((b.plus_target,b.minus_target));mask=np.stack((b.plus_mask,b.minus_mask))
    leads=np.zeros(collection.horizon);leads[np.array(cfg['pairs']['lead_steps'])-1]=cfg['pairs']['lead_weights']
    components=np.array(cfg['components']);rng=np.random.default_rng(cfg['seed'])
    optimizer=Adam(model.params,learning_rate=cfg['optimizer']['learning_rate'],max_grad_norm=cfg['optimizer']['max_grad_norm'])
    with (OUT/'training.jsonl').open('x') as log:
        for update in range(1,cfg['updates']+1):
            if time.monotonic()-started>600:raise RuntimeError('wall cap')
            h=cfg['anchors']['horizons'][(update-1)%3];sx,sk,sy=windows[h];take=rng.integers(len(sx),size=64)
            loss,g=model.loss_and_grad(sx[take],sk[take],sy[take],state_weights=components,state_error_scale_si=units)
            p,pg=paired_loss(model,x,k,target,mask,b.response_mask,b.noise_mask,b.response_floor,leads,components,
                np.array(cfg['response_scales_si']),.5,.2,sample_weights=fold.pair_weights,absolute_scale_si=units)
            norm=optimizer.step({key:g[key]+pg[key] for key in g})
            if update%30==0:guard(update);log.write(json.dumps({'update':update,'anchor_loss':loss,'pair':p,'gradient_norm':norm},allow_nan=False)+'\n')
    checkpoint=OUT/'candidate_900.npz'
    np.savez_compressed(checkpoint,**model.state_dict(),known_mean=fold.known_mean,known_scale=fold.known_scale,
        known_names=np.array(KNOWN_NAMES),state_error_scale_si=units,residual_gain_si=np.array(physical['residual_gain_si']),updates=np.array(900))
    expected=model.rollout(x,k[0])
    with np.load(checkpoint,allow_pickle=False) as s:restored=ResidualMLP.from_state_dict(s)
    np.testing.assert_array_equal(expected,restored.rollout(x,k[0]))
    histories={e['episode_id']:e['history_id'] for e in json.loads((ROOT/'configs/s3_history_coverage_plan.json').read_text())['episodes']}
    after=report(restored,fold.known_mean,fold.known_scale,collection,data,histories)
    with np.load(ROOT/cfg['baseline_checkpoint'],allow_pickle=False) as s:
        baseline=ResidualMLP.from_state_dict(s);before=report(baseline,s['known_mean'],s['known_scale'],collection,data,histories)
    # Reproduce previous frozen new-cohort validation, independent of training.
    old=json.loads((ROOT/'runs/s3_history_coverage_20261114/frozen_comparison.json').read_text())
    for split in ('train','validation'):
        np.testing.assert_allclose(before['summary'][f'cohort/{split}/history/0.2']['32']['absolute_rmse_p_v_w_si'],old['summary'][split]['all']['32']['absolute_rmse_p_v_w_si'],rtol=1e-12)
    gate=regression(before,after)
    write(OUT/'comparison.json',{'before':before,'after':after,'regression':gate})
    guard(900)
    write(OUT/'summary.json',{'status':'completed','updates':900,'fits':1,'wall_s':time.monotonic()-started,
        'reload_equal':True,'baseline_metrics_reproduced':True,'candidate_sha256':sha(checkpoint),'regression':gate,
        'checkpoint_selection':False,'control_accepted':False,'simulation_steps':0,'test_metrics_used':False})
    print(json.dumps({'regression':gate,'new_validation_before':before['summary']['cohort/validation/history/0.2']['32'],
        'new_validation_after':after['summary']['cohort/validation/history/0.2']['32']},indent=2))


if __name__=='__main__':main()
