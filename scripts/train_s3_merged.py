"""Fixed-budget warm starts on group-balanced original and coverage pairs."""
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
from ball_retention.frame_model import FrameResidualMLP,pack_features
from ball_retention.geometry import load_tool,support_point
from ball_retention.guard import process_memory
from ball_retention.model_data import prepare_dataset
from ball_retention.pair_collection import prepare_collection
from ball_retention.pair_loss import paired_loss
from ball_retention.resources import snapshot,directory_bytes
from ball_retention.world_model import ResidualMLP,Adam
from evaluate_s3_coverage import summarize

OUT=ROOT/'runs/s3_merged_refit_20261113'
STARTS={'raw_request':'runs/s3_pair_refit_20261112/fit/raw_request.npz',
        'frame':'runs/s3_frame_refit_20261113/frame_fixed.npz'}
CFG={'seed':20261111,'updates_per_variant':1500,'total_updates':3000,'lr':.0003,'clip':5.,
     'anchor_batch':64,'anchor_horizons':[1,4,8],'leads':[8,16,32],'lead_weights':[.2,.3,.5],
     'components':[1.5]*3+[.7]*3+[.4]*3,'response_scale':[.0005]*3+[.025]*3+[.5]*3,
     'anchor_weight':1.,'absolute_weight':.5,'response_weight':.2,'validate_every':30,
     'wall_s':600,'rss_gib':1.,'commit_gib':1.5,'output_mib':64,'min_start_ram_gib':2.,
     'min_running_ram_gib':.75,'min_disk_gib':10.}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def run(start,experiment=None):
    def guard(update):
        m=process_memory(os.getpid()); s=snapshot(ROOT)
        row={'update':update,'wall_s':time.monotonic()-start,**m,**s,'output_bytes':directory_bytes(OUT)}
        with (OUT/'resources.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        floor=CFG['min_start_ram_gib'] if update<0 else CFG['min_running_ram_gib']
        if (row['wall_s']>CFG['wall_s'] or m['rss_bytes']>CFG['rss_gib']*1024**3
            or m['private_commit_bytes']>CFG['commit_gib']*1024**3 or s['available_ram_bytes']<floor*1024**3
            or s['disk_free_bytes']<CFG['min_disk_gib']*1024**3 or row['output_bytes']>CFG['output_mib']*1024**2):
            raise RuntimeError('CPU refit resource budget exceeded')
    guard(-1)
    collection=prepare_collection(ROOT,ROOT/'configs/s3_pair_collection.json')
    data=prepare_dataset(ROOT/'data/s2_pilot_20261112')
    if collection.model_config!=data.model_config or collection.dt_s!=data.dt_s:
        raise ValueError('pair and S2 configuration mismatch')
    anchor=support_point(load_tool(ROOT/'assets/shovel.STL'),data.model_config['initial']['anchor_xy_tool_m'])[0]
    write(OUT/'data_validation.json',collection.report)
    weights=np.zeros(collection.horizon);weights[np.array(CFG['leads'])-1]=CFG['lead_weights']
    components=np.array(CFG['components']); response_scale=np.array(CFG['response_scale'])
    terminal=set(data.validation_endings.episode_ids[data.validation_endings.terminated])
    results={}
    for variant,path in STARTS.items():
        frame=variant=='frame'; cls=FrameResidualMLP if frame else ResidualMLP
        with np.load(ROOT/path,allow_pickle=False) as saved:
            model=cls.from_state_dict(saved)
            for key,actual in [('known_mean',data.known_mean),('known_scale',data.known_scale),
                               ('state_mean',data.state_mean),('state_scale',data.state_scale)]:
                np.testing.assert_array_equal(saved[key],actual)
        def features(k):
            return (pack_features(k,data.model_config,data.dt_s,anchor,data.state_mean,data.state_scale,
                                  data.known_mean,data.known_scale) if frame else (k-data.known_mean)/data.known_scale)
        windows={s:{h:(model.normalize(w.initial),features(w.known),model.normalize(w.target)) for h,w in ds.items()}
                 for s,ds in [('train',data.train),('validation',data.validation)]}
        pair_features={s:features(np.stack((v.batch.plus_known,v.batch.minus_known)))
                       for s,v in [('train',collection.train),('validation',collection.validation)]}
        def objective(split,gradients):
            mixed=getattr(collection,split); b=mixed.batch
            result,grads=paired_loss(model,model.normalize(b.initial),pair_features[split],np.stack((b.plus_target,b.minus_target)),
                np.stack((b.plus_mask,b.minus_mask)),b.response_mask,b.noise_mask,b.response_floor,weights,components,
                response_scale,CFG['absolute_weight'],CFG['response_weight'],gradients=gradients,sample_weights=mixed.sample_weights)
            if experiment and variant=='center_and_direction' and gradients:
                from ball_retention.guarded_loss import center_direction_loss
                aux,ag=center_direction_loss(model,model.normalize(b.initial),pair_features[split],
                    np.stack((b.plus_target,b.minus_target)),np.stack((b.plus_mask,b.minus_mask)),b.noise_mask,
                    b.baseline_first,b.baseline_second,weights,mixed.sample_weights,experiment['additional_loss'])
                result={**result,'auxiliary':aux,'loss':result['loss']+aux['loss']}
                grads={key:grads[key]+ag[key] for key in grads}
            return result,grads
        def validation():
            losses=[]
            for h,(x,k,y) in windows['validation'].items():
                route=np.array([i not in terminal for i in data.validation[h].episode_ids])
                error=model.rollout(x[route],k[route])[:,-1]-y[route,-1]
                losses.append(float(np.average(np.mean(error**2,axis=0),weights=components)))
            p,_=objective('validation',False)
            return {'route':float(np.mean(losses)),**p,'score':float(np.mean(losses))+p['loss']}
        def report():
            pair_rows=[]
            for split in ('train','validation'):
                mixed=getattr(collection,split);b=mixed.batch
                k=pair_features[split].reshape(-1,collection.horizon,model.known_dim)
                x=model.normalize(np.concatenate((b.initial,b.initial)))
                prediction=model.denormalize(model.rollout(x,k)).reshape(2,-1,collection.horizon,9)
                for i in range(len(b.initial)):
                    def masked(v,m): return [x if keep else None for x,keep in zip(v,m)]
                    response=evaluate_pair(prediction[0,i],prediction[1,i],masked(b.plus_target[i],b.plus_mask[i]),
                        masked(b.minus_target[i],b.minus_mask[i]),float(mixed.amplitudes[i]),anchor[:2],
                        replay_first=masked(b.baseline_first[i],b.baseline_mask[i]),
                        replay_second=masked(b.baseline_second[i],b.baseline_mask[i]),leads=CFG['leads'])
                    absolute={}
                    for lead in CFG['leads']:
                        errors=[prediction[j,i,lead-1]-t[i,lead-1] for j,t,m in
                                [(0,b.plus_target,b.plus_mask),(1,b.minus_target,b.minus_mask)] if m[i,lead-1]]
                        absolute[str(lead)]={'count':len(errors),'squared_p_v_w':np.sum(np.array(errors).reshape(-1,3,3)**2,axis=(0,2)).tolist()}
                    pair_rows.append({'split':split,'cohort':str(mixed.cohorts[i]),'group':str(b.group_ids[i]),
                        'tick':int(b.snapshot_ticks[i]),'axis':int(b.axes[i]),'amplitude':float(mixed.amplitudes[i]),
                        'response':response,'absolute':absolute})
            summary={}
            selectors=sorted(set((r['split'],r['cohort'],r['amplitude']) for r in pair_rows))
            selectors+=sorted(set((r['split'],r['group'],r['amplitude']) for r in pair_rows))
            for split,group,amplitude in selectors:
                chosen=[r for r in pair_rows if r['split']==split and (r['cohort']==group or r['group']==group) and r['amplitude']==amplitude]
                rows={}
                for lead in CFG['leads']:
                    metrics=[r['response']['by_lead'][str(lead)]['position'] for r in chosen
                             if r['response']['by_lead'][str(lead)]['status']=='comparable']
                    count=sum(r['absolute'][str(lead)]['count'] for r in chosen)
                    squared=np.sum([r['absolute'][str(lead)]['squared_p_v_w'] for r in chosen],axis=0)
                    rows[str(lead)]={'position_all':summarize(metrics,.5),
                        'position_strong':summarize([m for m in metrics if m['newton_half_spread']>=10*m['signal_threshold']],.5),
                        'position_near_floor':summarize([m for m in metrics if m['newton_resolved'] and m['newton_half_spread']<10*m['signal_threshold']],.5),
                        'absolute_rmse_p_v_w_si':np.sqrt(squared/count).tolist() if count else None}
                summary[f'{split}/{group}/{amplitude}']=rows
            s2={}
            for h,(x,k,y) in windows['validation'].items():
                raw=model.denormalize(model.rollout(x,k))[:,-1];w=data.validation[h]
                s2[str(h)]={}
                for episode in np.unique(w.episode_ids):
                    take=w.episode_ids==episode;err=raw[take]-w.target[take,-1]
                    s2[str(h)][str(episode)]={'windows':int(take.sum()),'rmse_p_v_w_si':np.sqrt(np.mean(np.sum(err.reshape(-1,3,3)**2,axis=-1),axis=0)).tolist()}
            return {'summary':summary,'pairs':pair_rows,'s2_validation':s2,
                    'complete_route_ids':sorted(set(data.validation[8].episode_ids)-terminal)}
        before=report();best=validation();best_state=model.state_dict();best_update=0
        eligible_updates=[]
        unconstrained_best,unconstrained_update,unconstrained_state=best,0,best_state
        optimizer=Adam(model.params,learning_rate=CFG['lr'],max_grad_norm=CFG['clip'])
        rng=np.random.default_rng(CFG['seed'])
        with (OUT/(variant+'_training.jsonl')).open('x') as log:
            log.write(json.dumps({'update':0,'validation':best})+'\n')
            for update in range(1,CFG['updates_per_variant']+1):
                if time.monotonic()-start>CFG['wall_s']: raise RuntimeError('CPU time exhausted')
                h=CFG['anchor_horizons'][(update-1)%3];x,k,y=windows['train'][h]
                take=rng.integers(len(x),size=CFG['anchor_batch'])
                loss,grad=model.loss_and_grad(x[take],k[take],y[take],state_weights=components)
                pair,pg=objective('train',True)
                norm=optimizer.step({key:v+pg[key] for key,v in grad.items()})
                row={'update':update,'anchor':loss,'paired':pair,'gradient_norm':norm}
                if update%CFG['validate_every']==0:
                    guard(update); row['validation']=validation()
                    eligible=True
                    if experiment:
                        from ball_retention.guarded_loss import checkpoint_guard
                        row['guard']=checkpoint_guard(before,report(),experiment['checkpoint_guard'])
                        eligible=row['guard']['passed']
                    if eligible: eligible_updates.append(update)
                    if row['validation']['score']<unconstrained_best['score']:
                        unconstrained_best,unconstrained_update,unconstrained_state=row['validation'],update,model.state_dict()
                    if eligible and row['validation']['score']<best['score']:
                        best,best_update,best_state=row['validation'],update,model.state_dict()
                log.write(json.dumps(row,allow_nan=False)+'\n')
        model=cls.from_state_dict(best_state)
        checkpoint=OUT/(variant+'.npz')
        np.savez_compressed(checkpoint,**best_state,known_mean=data.known_mean,known_scale=data.known_scale,
            known_names=np.array(data.known_names),anchor_local=anchor,model_config_json=np.array(json.dumps(data.model_config)))
        x,k,_=windows['validation'][8]
        expected=model.rollout(x[:2],k[:2])
        with np.load(checkpoint,allow_pickle=False) as saved: model=cls.from_state_dict(saved)
        np.testing.assert_array_equal(expected,model.rollout(x[:2],k[:2]))
        results[variant]={'before':before,'after':report(),'best_update':best_update,'best_validation':best,
                          'checkpoint_sha256':sha(checkpoint),'reload_equal':True,'eligible_updates':eligible_updates}
        if experiment:
            from ball_retention.guarded_loss import checkpoint_guard
            results[variant]['selected_guard']=checkpoint_guard(before,results[variant]['after'],experiment['checkpoint_guard'])
            model=cls.from_state_dict(unconstrained_state)
            rejected_report=report()
            results[variant]['unconstrained_best']={'update':unconstrained_update,'validation':unconstrained_best,
                'guard':checkpoint_guard(before,rejected_report,experiment['checkpoint_guard']),'report':rejected_report}
            np.savez_compressed(OUT/(variant+'_unconstrained_diagnostic.npz'),**unconstrained_state,
                                known_mean=data.known_mean,known_scale=data.known_scale,known_names=np.array(data.known_names))
        write(OUT/'comparison.json',results)
        print(variant,'best_update',best_update,'score',best['score'],flush=True)
    guard(CFG['total_updates'])
    write(OUT/'summary.json',{'status':'completed','updates':CFG['total_updates'],'wall_s':time.monotonic()-start,
        'test_used':False,'simulation_steps':0,'validation_use':'old and new groups both used for checkpoint selection',
        'comparison_scope':'same-start objective ablation with regression guards' if experiment else
                           'warm starts with different histories; compare each before/after, not pure architecture ablation'})

if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),ROOT/'src/ball_retention/pair_collection.py',ROOT/'src/ball_retention/pair_loss.py',
           ROOT/'configs/s3_pair_collection.json',ROOT/'scripts/evaluate_s3_coverage.py']
    write(OUT/'plan.json',{'config':CFG,'threads':1,'device':'cpu','source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        'starting_checkpoints':{k:{'path':v,'sha256':sha(ROOT/v)} for k,v in STARTS.items()},
        'checkpoint_selection':'S2 complete-route mean loss plus group-balanced old/new validation pair loss',
        'normalization':'unchanged S2 train-only statistics','test_used':False})
    start=time.monotonic()
    try: run(start)
    except Exception as exc:
        write(OUT/'failure.json',{'error':str(exc),'wall_s':time.monotonic()-start})
        raise
