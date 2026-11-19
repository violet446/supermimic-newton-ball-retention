"""Four leave-initial-group-out fits and a matched full-training control."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ball_retention.action_eval import evaluate_pair
from ball_retention.group_fold import make_group_fold
from ball_retention.model_data import prepare_dataset,KNOWN_NAMES
from ball_retention.pair_collection import prepare_collection
from ball_retention.pair_loss import paired_loss
from ball_retention.world_model import ResidualMLP,Adam
from ball_retention.guard import process_memory
from ball_retention.resources import snapshot,directory_bytes
from evaluate_s3_coverage import summarize

OUT=ROOT/'runs/s3_group_normalization_20261113'
CONFIG={'seed':20261111,'updates_per_fit':900,'fits':5,'total_updates':4500,'wall_s':600,
        'rss_gib':1.,'commit_gib':1.5,'output_mib':64,'lr':.0003,'clip':5.,'batch':64,
        'horizons':[1,4,8],'pair_leads':[8,16,32],'pair_lead_weights':[.2,.3,.5],
        'components':[1.5]*3+[.7]*3+[.4]*3,'response_scales':[.0005]*3+[.025]*3+[.5]*3,
        'absolute_weight':.5,'response_weight':.2,'initialization':'fresh zero-output residual MLP, identical seed',
        'selection':'fixed final update only; holdout scores never used to pick weights',
        'development_validation_used':False,'test_used':False,'simulation_steps':0,
        'normalization':'retained_training_v2'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run():
    OUT.mkdir(exist_ok=False);started=time.monotonic()
    sources=[Path(__file__),ROOT/'src/ball_retention/group_fold.py',ROOT/'src/ball_retention/world_model.py',
             ROOT/'src/ball_retention/pair_loss.py',ROOT/'configs/s3_pair_collection.json']
    if 'physical_scale_config' in CONFIG:
        sources.append(ROOT/CONFIG['physical_scale_config'])
    write(OUT/'plan.json',{'config':CONFIG,'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources}})
    def guard(fit,update):
        m=process_memory(os.getpid());machine=snapshot(ROOT)
        row={'fit':fit,'update':update,'wall_s':time.monotonic()-started,**m,**machine}
        with (OUT/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if (row['wall_s']>600 or m['rss_bytes']>1024**3 or m['private_commit_bytes']>1.5*1024**3
            or machine['available_ram_bytes']<(2 if update<0 else .75)*1024**3
            or machine['disk_free_bytes']<10*1024**3 or directory_bytes(OUT)>64*1024**2):
            raise RuntimeError('fold diagnostic resource budget exceeded')
    guard('prepare',-1)
    collection=prepare_collection(ROOT,ROOT/'configs/s3_pair_collection.json')
    data=prepare_dataset(ROOT/'data/s2_pilot_20261112')
    # Only original training windows and training pairs reach fold fitting or
    # scoring. Loader integrity checks may read other split files, never metrics.
    manifest=json.loads((ROOT/'data/s2_pilot_20261112/manifest.json').read_text())
    mapping={e['episode_id']:e['group_id'] for e in manifest['entries'] if e['split']=='train'}
    all_pairs=collection.train.batch
    groups=sorted(set(all_pairs.group_ids.tolist()))
    if len(groups)!=4:raise ValueError('expected four predeclared training groups')
    folds=[('full_training',None)]+[(f'holdout_{i}',g) for i,g in enumerate(groups)]
    components=np.array(CONFIG['components']);leads=np.zeros(collection.horizon)
    leads[np.array(CONFIG['pair_leads'])-1]=CONFIG['pair_lead_weights']
    results={}
    for name,held in folds:
        fold=make_group_fold(data.train,all_pairs,mapping,held,normalization=CONFIG['normalization'])
        if held in fold.report['retained_pair_groups'] or held in {mapping[e] for e in fold.report['retained_anchor_episodes']}:
            raise ValueError('holdout leaked into training')
        write(OUT/(name+'_data.json'),fold.report)
        gain=([.05]*3+[.25]*6 if 'residual_gain_si' not in CONFIG
              else np.array(CONFIG['residual_gain_si'])/fold.state_scale)
        error_scale=CONFIG.get('state_error_scale_si')
        model=ResidualMLP(20,fold.state_mean,fold.state_scale,data.dt_s,seed=CONFIG['seed'],residual_gain=gain)
        def known(k):return (k-fold.known_mean)/fold.known_scale
        windows={h:(model.normalize(w.initial),known(w.known),model.normalize(w.target)) for h,w in fold.anchors.items()}
        b=fold.pairs;k=known(np.stack((b.plus_known,b.minus_known)));x=model.normalize(b.initial)
        feature_values=np.concatenate([v[1].reshape(-1,20) for v in windows.values()]+[k.reshape(-1,20)])
        write(OUT/(name+'_normalization.json'),{
            'state_mean':fold.state_mean.tolist(),'state_scale':fold.state_scale.tolist(),
            'known_mean':fold.known_mean.tolist(),'known_scale':fold.known_scale.tolist(),
            'retained_network_known_max_abs_z':float(np.max(np.abs(feature_values))),
            'retained_network_known_fraction_abs_z_gt_10':float(np.mean(np.abs(feature_values)>10)),
            'paired_known_max_abs_z':float(np.max(np.abs(k))),
            'paired_known_fraction_abs_z_gt_10':float(np.mean(np.abs(k)>10)),
            'per_feature_max_abs_z':dict(zip(KNOWN_NAMES,np.max(np.abs(feature_values),axis=0).tolist()))})
        target=np.stack((b.plus_target,b.minus_target));mask=np.stack((b.plus_mask,b.minus_mask))
        rng=np.random.default_rng(CONFIG['seed']);optimizer=Adam(model.params,learning_rate=CONFIG['lr'],max_grad_norm=CONFIG['clip'])
        with (OUT/(name+'_training.jsonl')).open('x') as log:
            for update in range(1,901):
                if time.monotonic()-started>600:raise RuntimeError('fold wall budget exhausted')
                h=CONFIG['horizons'][(update-1)%3];sx,sk,sy=windows[h]
                take=rng.integers(len(sx),size=64)
                loss,gradient=model.loss_and_grad(sx[take],sk[take],sy[take],state_weights=components,
                    state_error_scale_si=error_scale)
                pair,pg=paired_loss(model,x,k,target,mask,b.response_mask,b.noise_mask,b.response_floor,leads,components,
                    np.array(CONFIG['response_scales']),.5,.2,sample_weights=fold.pair_weights,
                    absolute_scale_si=error_scale)
                norm=optimizer.step({key:v+pg[key] for key,v in gradient.items()})
                if update%30==0:
                    guard(name,update)
                    log.write(json.dumps({'update':update,'anchor_loss':loss,'pair':pair,'gradient_norm':norm},allow_nan=False)+'\n')
        checkpoint=OUT/(name+'.npz')
        physical_metadata={} if error_scale is None else {
            'state_error_scale_si':np.array(error_scale),'residual_gain_si':np.array(CONFIG['residual_gain_si'])}
        np.savez_compressed(checkpoint,**model.state_dict(),**physical_metadata,known_mean=fold.known_mean,known_scale=fold.known_scale,
                            known_names=np.array(KNOWN_NAMES),held_group=np.array(held or ''),updates=np.array(900))
        ak=known(np.concatenate((all_pairs.plus_known,all_pairs.minus_known)))
        ax=model.normalize(np.concatenate((all_pairs.initial,all_pairs.initial)))
        expected=model.rollout(ax,ak)
        with np.load(checkpoint,allow_pickle=False) as saved:model=ResidualMLP.from_state_dict(saved)
        np.testing.assert_array_equal(expected,model.rollout(ax,ak))
        pred=model.denormalize(expected).reshape(2,-1,collection.horizon,9)
        rows=[]
        for i in range(len(all_pairs.initial)):
            b=all_pairs
            def masked(v,m):return [row if keep else None for row,keep in zip(v,m)]
            response=evaluate_pair(pred[0,i],pred[1,i],masked(b.plus_target[i],b.plus_mask[i]),masked(b.minus_target[i],b.minus_mask[i]),
                float(collection.train.amplitudes[i]),[0.,0.],replay_first=masked(b.baseline_first[i],b.baseline_mask[i]),
                replay_second=masked(b.baseline_second[i],b.baseline_mask[i]),leads=(8,16,32))
            absolute={}
            for lead in (8,16,32):
                errors=[pred[j,i,lead-1]-t[i,lead-1] for j,t,m in [(0,b.plus_target,b.plus_mask),(1,b.minus_target,b.minus_mask)] if m[i,lead-1]]
                absolute[str(lead)]={'count':len(errors),'squared_p_v_w':np.sum(np.array(errors).reshape(-1,3,3)**2,axis=(0,2)).tolist()}
            rows.append({'group':str(b.group_ids[i]),'held_out':str(b.group_ids[i])==held,'tick':int(b.snapshot_ticks[i]),
                         'axis':int(b.axes[i]),'amplitude':float(collection.train.amplitudes[i]),'response':response,'absolute':absolute})
        summary={}
        for group in groups:
            summary[group]={}
            for amplitude in sorted(set(r['amplitude'] for r in rows if r['group']==group)):
                chosen=[r for r in rows if r['group']==group and r['amplitude']==amplitude]
                summary[group][str(amplitude)]={}
                for lead in (8,16,32):
                    metrics=[r['response']['by_lead'][str(lead)]['position'] for r in chosen if r['response']['by_lead'][str(lead)]['status']=='comparable']
                    n=sum(r['absolute'][str(lead)]['count'] for r in chosen)
                    square=np.sum([r['absolute'][str(lead)]['squared_p_v_w'] for r in chosen],axis=0)
                    summary[group][str(amplitude)][str(lead)]={'position':summarize(metrics,.5),
                        'strong_position':summarize([m for m in metrics if m['newton_half_spread']>=10*m['signal_threshold']],.5),
                        'absolute_rmse_p_v_w_si':np.sqrt(square/n).tolist() if n else None}
        results[name]={'held_group':held,'updates':900,'checkpoint_sha256':sha(checkpoint),'reload_equal':True,
                       'summary':summary,'cases':rows,'fold_data':fold.report}
        write(OUT/'comparison.json',results)
        print(name,'held',held,'complete',flush=True)
    guard('complete',4500)
    write(OUT/'summary.json',{'status':'completed','wall_s':time.monotonic()-started,'updates':4500,
          'fits':5,'checkpoint_selection':False,'pretrained_weights_used':False,'development_validation_metrics_used':False,
          'test_metrics_used':False,'simulation_steps':0,'limitation':'one seed, fixed 900 updates; training-loss convergence not assumed'})

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--normalization',choices=('s2_only_v1','retained_training_v2'),default=CONFIG['normalization'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--physical-scales',type=Path)
    args=parser.parse_args()
    CONFIG['normalization']=args.normalization
    if args.physical_scales is not None:
        physical_path=args.physical_scales.resolve()
        CONFIG['physical_scale_config']=physical_path.relative_to(ROOT).as_posix()
        physical=json.loads(physical_path.read_text(encoding='utf-8'))
        for key in ('state_error_scale_si','residual_gain_si'):
            values=np.asarray(physical[key],dtype=float)
            if values.shape!=(9,) or not np.isfinite(values).all() or np.any(values<=0):
                raise ValueError('physical scales must contain nine positive finite entries')
            CONFIG[key]=values.tolist()
    OUT=args.output.resolve();OUT.relative_to(ROOT)
    run()
