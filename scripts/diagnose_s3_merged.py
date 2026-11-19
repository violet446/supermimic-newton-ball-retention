"""Read-only error/selection audit. Teacher-forced steps are diagnostics only."""
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
from ball_retention.model_data import prepare_dataset
from ball_retention.pair_collection import prepare_collection
from ball_retention.pair_loss import paired_loss
from ball_retention.guard import process_memory
from ball_retention.resources import snapshot,directory_bytes

OUT=ROOT/'runs/s3_error_audit_20261113'
FIT=ROOT/'runs/s3_merged_refit_20261113'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def loss_parts(model,prediction,mixed,leads,components,scales):
    b=mixed.batch
    target=np.stack((b.plus_target,b.minus_target))
    mask=np.stack((b.plus_mask,b.minus_mask))
    weights=leads[:,None]*components
    absolute_weights=mask[...,None]*weights*mixed.sample_weights[None,:,None,None]
    absolute_weights/=absolute_weights.sum()
    absolute=absolute_weights*((prediction-target)/model.state_scale)**2
    half=(prediction[0]-prediction[1])/2
    truth=(target[0]-target[1])/2
    error=np.where(b.response_mask,half-truth,np.sign(half)*np.maximum(abs(half)-b.response_floor,0.))
    response_weights=b.noise_mask[...,None]*weights*mixed.sample_weights[:,None,None]
    response_weights/=response_weights.sum()
    response=response_weights*2*(np.hypot(1.,error/scales)-1)
    result={}
    for group in np.unique(b.group_ids):
        take=b.group_ids==group
        a=np.sum(absolute[:,take],axis=(0,1,2)).reshape(3,3).sum(axis=1)
        r=np.sum(response[take],axis=(0,1)).reshape(3,3).sum(axis=1)
        result[str(group)]={'weighted_absolute_p_v_w':a.tolist(),'weighted_response_p_v_w':r.tolist(),
                            'contribution_to_selection':float(.5*a.sum()+.2*r.sum())}
    return result,float(absolute.sum()),float(response.sum())

def main():
    OUT.mkdir(exist_ok=False); started=time.monotonic()
    plan=json.loads((FIT/'plan.json').read_text()); cfg=plan['config']
    inputs=[ROOT/'configs/s3_pair_collection.json',FIT/'comparison.json',FIT/'plan.json']
    for name,spec in plan['starting_checkpoints'].items(): inputs.extend([ROOT/spec['path'],FIT/(name+'.npz')])
    write(OUT/'plan.json',{'cpu_threads':1,'seed':None,'training_updates':0,'simulation_steps':0,
        'wall_s':600,'rss_gib':1.,'commit_gib':1.5,'output_mib':32,'test_used':False,
        'inputs_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},'source_sha256':sha(Path(__file__)),
        'teacher_forcing':'separate local one-step diagnostic only; never substituted into free predictions',
        'coverage':'nearest standardized training state distances are descriptive, not causal proof'})
    machine=snapshot(ROOT)
    if machine['available_ram_bytes']<2*1024**3: raise RuntimeError('insufficient startup RAM')
    collection=prepare_collection(ROOT,ROOT/'configs/s3_pair_collection.json')
    data=prepare_dataset(ROOT/'data/s2_pilot_20261112')
    mixed=collection.validation;b=mixed.batch; train=collection.train.batch
    if not (b.plus_mask.all() and b.minus_mask.all() and b.noise_mask.all()):
        raise ValueError('audit requires matched full leads')
    target=np.stack((b.plus_target,b.minus_target))
    true_inputs=np.concatenate((np.broadcast_to(b.initial,(2,*b.initial.shape))[:,:,None],target[:,:,:-1]),axis=2)
    train_states=np.concatenate((train.initial,train.plus_target.reshape(-1,9),train.minus_target.reshape(-1,9),data.train[1].initial))
    state_z=(train_states-data.state_mean)/data.state_scale
    leads=np.zeros(collection.horizon);leads[np.array(cfg['leads'])-1]=cfg['lead_weights']
    components=np.array(cfg['components']);scales=np.array(cfg['response_scale'])
    comparison=json.loads((FIT/'comparison.json').read_text()); result={}
    for variant,spec in plan['starting_checkpoints'].items():
        frame=variant=='frame'
        cls=FrameResidualMLP if frame else __import__('ball_retention.world_model',fromlist=['ResidualMLP']).ResidualMLP
        result[variant]={}
        for stage,path in [('before',ROOT/spec['path']),('after',FIT/(variant+'.npz'))]:
            with np.load(path,allow_pickle=False) as saved:
                model=cls.from_state_dict(saved);anchor=saved['anchor_local'] if frame else np.zeros(3)
                mean,scale=saved['known_mean'],saved['known_scale']
            def features(k):
                return pack_features(k,collection.model_config,model.dt_s,anchor,model.state_mean,model.state_scale,mean,scale) if frame else (k-mean)/scale
            k=features(np.stack((b.plus_known,b.minus_known)))
            predicted=model.denormalize(model.rollout(model.normalize(np.concatenate((b.initial,b.initial))),k.reshape(-1,32,model.known_dim))).reshape(2,-1,32,9)
            local=model.denormalize(model.step(model.normalize(true_inputs),k))
            parts,absolute,response=loss_parts(model,predicted,mixed,leads,components,scales)
            reference,_=paired_loss(model,model.normalize(b.initial),k,target,np.stack((b.plus_mask,b.minus_mask)),
                b.response_mask,b.noise_mask,b.response_floor,leads,components,scales,.5,.2,gradients=False,sample_weights=mixed.sample_weights)
            np.testing.assert_allclose([absolute,response],[reference['absolute'],reference['response']],rtol=1e-12)
            route=[]
            terminal=set(data.validation_endings.episode_ids[data.validation_endings.terminated])
            for h,w in data.validation.items():
                keep=np.array([i not in terminal for i in w.episode_ids])
                error=model.rollout(model.normalize(w.initial[keep]),features(w.known[keep]))[:,-1]-model.normalize(w.target[keep,-1])
                route.append(float(np.average(np.mean(error**2,axis=0),weights=components)))
            score=float(np.mean(route))+.5*absolute+.2*response
            logs=[json.loads(line) for line in (FIT/(variant+'_training.jsonl')).read_text().splitlines()]
            update=0 if stage=='before' else comparison[variant]['best_update']
            saved_score=next(r['validation']['score'] for r in logs if r['update']==update)
            np.testing.assert_allclose(score,saved_score,rtol=1e-10)
            rows=[]
            for i in range(len(b.initial)):
                center_error=(predicted[0,i]+predicted[1,i]-target[0,i]-target[1,i])/2
                response_error=(predicted[0,i]-predicted[1,i]-target[0,i]+target[1,i])/2
                free_error=predicted[:,i]-target[:,i]; local_error=local[:,i]-target[:,i]
                # Exact identity: mean branch error squared = center error squared + half-difference error squared.
                np.testing.assert_allclose(np.mean(free_error**2,axis=0),center_error**2+response_error**2,atol=1e-14)
                response_metrics=evaluate_pair(predicted[0,i],predicted[1,i],target[0,i],target[1,i],float(mixed.amplitudes[i]),[0.,0.],
                    replay_first=b.baseline_first[i],replay_second=b.baseline_second[i],leads=tuple(range(1,33)))
                z=(b.initial[i]-data.state_mean)/data.state_scale
                nearest=int(np.argmin(np.linalg.norm(state_z-z,axis=1)))
                per_lead=[]
                for t in range(32):
                    pos=response_metrics['by_lead'][str(t+1)]['position']
                    per_lead.append({'lead':t+1,'position_response':pos,
                        'free_rmse_p_v_w':np.sqrt(np.mean(np.sum(free_error[:,t].reshape(2,3,3)**2,axis=-1),axis=0)).tolist(),
                        'local_one_step_rmse_p_v_w':np.sqrt(np.mean(np.sum(local_error[:,t].reshape(2,3,3)**2,axis=-1),axis=0)).tolist(),
                        'center_error_p_v_w':np.linalg.norm(center_error[t].reshape(3,3),axis=-1).tolist(),
                        'half_difference_error_p_v_w':np.linalg.norm(response_error[t].reshape(3,3),axis=-1).tolist()})
                opposite=[x['lead'] for x in per_lead if x['position_response']['newton_resolved'] and x['position_response']['cosine'] is not None and x['position_response']['cosine']<0]
                rows.append({'group':str(b.group_ids[i]),'cohort':str(mixed.cohorts[i]),'tick':int(b.snapshot_ticks[i]),
                    'axis':int(b.axes[i]),'amplitude':float(mixed.amplitudes[i]),'first_resolved_opposed_lead':min(opposite) if opposite else None,
                    'nearest_training_state_z_distance':float(np.linalg.norm(state_z[nearest]-z)),
                    'nearest_training_state_delta_p_v_w_si':np.linalg.norm((train_states[nearest]-b.initial[i]).reshape(3,3),axis=-1).tolist(),
                    'by_lead':per_lead})
            result[variant][stage]={'selection_score':score,'route_contribution':float(np.mean(route)),
                                   'group_loss_contributions':parts,'cases':rows}
            memory=process_memory(os.getpid())
            if time.monotonic()-started>600 or memory['rss_bytes']>1024**3 or memory['private_commit_bytes']>1.5*1024**3:
                raise RuntimeError('audit resource budget exceeded')
    write(OUT/'audit.json',result)
    if directory_bytes(OUT)>32*1024**2: raise RuntimeError('output cap exceeded')
    write(OUT/'validation.json',{'status':'passed','wall_s':time.monotonic()-started,**memory,
        'selection_scores_reproduced':True,'loss_decomposition_checked':True,'center_response_identity_checked':True,
        'training_updates':0,'simulation_steps':0,'test_used':False,'audit_sha256':sha(OUT/'audit.json')})
    for v,stages in result.items():
        print(v,{s:{'score':x['selection_score'],'groups':x['group_loss_contributions']} for s,x in stages.items()})

if __name__=='__main__': main()
