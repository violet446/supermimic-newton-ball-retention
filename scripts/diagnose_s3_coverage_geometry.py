"""Read-only training-fold coverage and mesh geometry audit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ball_retention.geometry import load_tool
from ball_retention.surface_diagnostic import SurfaceDiagnostic,sustained_crossing
from ball_retention.group_fold import make_group_fold
from ball_retention.model_data import prepare_dataset
from ball_retention.pair_collection import prepare_collection
from ball_retention.world_model import ResidualMLP
from ball_retention.guard import process_memory
from ball_retention.resources import snapshot,directory_bytes
from check_s3_group_generalization import sha,write

OUT=ROOT/'runs/s3_coverage_geometry_20261113'
FIT=ROOT/'runs/s3_physical_scales_20261113'
UNITS=np.repeat([.005,.05,2.5],3)


def bank(fold,mapping):
    states=[];features=[];metadata=[]
    for h,w in fold.anchors.items():
        previous=np.concatenate((w.initial[:,None],w.target[:,:-1]),axis=1)
        states.append(previous.reshape(-1,9));features.append(w.known.reshape(-1,20))
        metadata.extend({'source':'S2','group':mapping[str(e)],'episode':str(e),'tick':int(t+j),'horizon':h}
                        for e,t in zip(w.episode_ids,w.ticks) for j in range(h))
    b=fold.pairs
    for sign,target,known,mask in (('plus',b.plus_target,b.plus_known,b.plus_mask),('minus',b.minus_target,b.minus_known,b.minus_mask)):
        if not mask.all():raise ValueError('audit requires complete paired branches')
        previous=np.concatenate((b.initial[:,None],target[:,:-1]),axis=1)
        states.append(previous.reshape(-1,9));features.append(known.reshape(-1,20))
        metadata.extend({'source':'pair_'+sign,'group':str(g),'episode':str(e),'tick':int(t+j),'row':i}
                        for i,(g,e,t) in enumerate(zip(b.group_ids,b.episode_ids,b.snapshot_ticks)) for j in range(known.shape[1]))
    return np.concatenate(states),np.concatenate(features),metadata


def main():
    OUT.mkdir(exist_ok=False);started=time.monotonic()
    inputs=[ROOT/'configs/s3_pair_collection.json',ROOT/'assets/shovel.STL',FIT/'comparison.json',FIT/'plan.json']+list(FIT.glob('*.npz'))
    sources=[Path(__file__),ROOT/'src/ball_retention/surface_diagnostic.py',ROOT/'src/ball_retention/geometry.py',ROOT/'src/ball_retention/group_fold.py']
    write(OUT/'plan.json',{'wall_s':600,'cpu_threads':1,'rss_gib':1,'commit_gib':1.5,'output_mib':64,
        'seed':None,'training_updates':0,'simulation_steps':0,'development_validation_metrics_used':False,'test_used':False,
        'position_crossing_thresholds_m':[.001,.005],'crossing_consecutive_steps':3,'geometric_distance_band_m':.002,
        'state_distance_units_si':UNITS.tolist(),'joint_distance':'equal mean-square state/known blocks; known scaled by retained fold statistics',
        'teacher_forcing':'local one-step diagnostic only, never fed into free rollouts',
        'geometry':'nearest actual mesh triangle; unsigned distance gap and oriented normal-plane gap, not Newton contact labels',
        'inputs_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
        'sources_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources}})
    def guard(stage):
        memory=process_memory(os.getpid());machine=snapshot(ROOT)
        row={'stage':stage,'wall_s':time.monotonic()-started,**memory,**machine}
        with (OUT/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if (row['wall_s']>600 or memory['rss_bytes']>1024**3 or memory['private_commit_bytes']>1.5*1024**3
            or machine['available_ram_bytes']<(.75 if stage!='start' else 2)*1024**3
            or machine['disk_free_bytes']<10*1024**3 or directory_bytes(OUT)>64*1024**2):
            raise RuntimeError('offline audit resource budget exceeded')
    guard('start')
    c=prepare_collection(ROOT,ROOT/'configs/s3_pair_collection.json');b=c.train.batch
    data=prepare_dataset(ROOT/'data/s2_pilot_20261112')
    manifest=json.loads((ROOT/'data/s2_pilot_20261112/manifest.json').read_text())
    mapping={e['episode_id']:e['group_id'] for e in manifest['entries'] if e['split']=='train'}
    surface=SurfaceDiagnostic(load_tool(ROOT/'assets/shovel.STL'),c.model_config['ball']['radius_m'],c.model_config['initial']['tool_rotation_world'])
    geometry_cache={}
    def geometry(p):
        key=tuple(p)
        if key not in geometry_cache:geometry_cache[key]=surface.measure(p)
        return geometry_cache[key]
    reference=json.loads((FIT/'comparison.json').read_text());results={}
    with np.load(FIT/'full_training.npz',allow_pickle=False) as s:
        full=ResidualMLP.from_state_dict(s);fkm=s['known_mean'];fks=s['known_scale']
    for name,record in reference.items():
        group=record['held_group']
        if group is None:continue
        fold=make_group_fold(data.train,b,mapping,group)
        states,features,metadata=bank(fold,mapping)
        assert all(m['group']!=group for m in metadata)
        state_tree=cKDTree(states/UNITS)
        position_tree=cKDTree(states[:,:3])
        joint_values=np.concatenate((states/UNITS/3,features/fold.known_scale/np.sqrt(20)),axis=1)
        joint_tree=cKDTree(joint_values)
        with np.load(FIT/(name+'.npz'),allow_pickle=False) as s:
            model=ResidualMLP.from_state_dict(s);km=s['known_mean'];ks=s['known_scale']
        def neighbors(x,k):
            ds,si=state_tree.query(x/UNITS);dp,pi=position_tree.query(x[:,:3])
            dj,ji=joint_tree.query(np.concatenate((x/UNITS/3,k/ks/np.sqrt(20)),axis=1))
            return ds,si,dp,pi,dj,ji
        rows=[]
        for i in fold.held_indices:
            for sign,target,known in (('plus',b.plus_target,b.plus_known),('minus',b.minus_target,b.minus_known)):
                truth=target[i];k=known[i]
                pred=model.denormalize(model.rollout(model.normalize(b.initial[i,None]),((k-km)/ks)[None]))[0]
                fit=full.denormalize(full.rollout(full.normalize(b.initial[i,None]),((k-fkm)/fks)[None]))[0]
                pre_truth=np.concatenate((b.initial[i,None],truth[:-1]));pre_pred=np.concatenate((b.initial[i,None],pred[:-1]))
                # Kept separate: a local diagnostic is not a free forecast.
                local=model.denormalize(model.step(model.normalize(pre_truth),(k-km)/ks))
                error=np.linalg.norm(pred[:,:3]-truth[:,:3],axis=1)
                one_step=np.linalg.norm(local[:,:3]-truth[:,:3],axis=1)
                onset={str(t):sustained_crossing(error,t) for t in (.001,.005)}
                ds,si,dp,pi,dj,ji=neighbors(pre_truth,k)
                fds,_,_,_,fdj,_=neighbors(pre_pred,k)
                selected=sorted(set([0,31]+[v-1 for v in onset.values() if v is not None]))
                detail=[]
                for t in selected:
                    detail.append({'lead':t+1,'query_timing':'before this step',
                        'truth_pre_state_si':pre_truth[t].tolist(),'free_pre_state_si':pre_pred[t].tolist(),
                        'tool_midpoint_tilt_rad':k[t,4:6].tolist(),'tool_midpoint_rate_rad_s':k[t,6:8].tolist(),
                        'true_pre_geometry':geometry(pre_truth[t,:3]),'free_pre_geometry':geometry(pre_pred[t,:3]),
                        'true_post_geometry':geometry(truth[t,:3]),'free_post_geometry':geometry(pred[t,:3]),
                        'nearest':{label:{'distance':float(dist[t]),'provenance':metadata[int(idx[t])],
                            'delta_p_v_w_si':np.linalg.norm((states[idx[t]]-pre_truth[t]).reshape(3,3),axis=1).tolist(),
                            'known_delta_z_norm':float(np.linalg.norm((features[idx[t]]-k[t])/ks)),
                            'geometry':geometry(states[idx[t],:3])}
                            for label,dist,idx in [('position_m',dp,pi),('state_units',ds,si),('joint_units',dj,ji)]}})
                rows.append({'group':group,'tick':int(b.snapshot_ticks[i]),'axis':int(b.axes[i]),'amplitude':float(c.train.amplitudes[i]),
                    'sign':sign,'row':int(i),'sustained_position_crossing_lead':onset,
                    'free_position_error_m':error.tolist(),'local_one_step_position_error_m':one_step.tolist(),
                    'full_fit_position_error_m':np.linalg.norm(fit[:,:3]-truth[:,:3],axis=1).tolist(),
                    'true_pre_nearest_position_m':dp.tolist(),'true_pre_nearest_state_units':ds.tolist(),'true_pre_nearest_joint_units':dj.tolist(),
                    'free_pre_nearest_state_units':fds.tolist(),'free_pre_nearest_joint_units':fdj.tolist(),
                    'geometry_and_neighbors':detail})
                guard(name+'_'+str(i)+'_'+sign)
        # Independently reproduce saved held H32 positions from branch errors.
        for amp,leads in record['summary'][group].items():
            selected=[r for r in rows if r['amplitude']==float(amp)]
            rmse=np.sqrt(np.mean([r['free_position_error_m'][-1]**2 for r in selected]))
            np.testing.assert_allclose(rmse,leads['32']['absolute_rmse_p_v_w_si'][0],rtol=1e-12)
        snapshots={}
        for tick in sorted(set(r['tick'] for r in rows)):
            selected=[r for r in rows if r['tick']==tick]
            first=selected[0]['geometry_and_neighbors'][0]
            snapshots[str(tick)]={'branches':len(selected),'initial':first,
                'crossing_1mm_leads':[r['sustained_position_crossing_lead']['0.001'] for r in selected],
                'crossing_5mm_leads':[r['sustained_position_crossing_lead']['0.005'] for r in selected],
                'H32_free_position_rmse_m':float(np.sqrt(np.mean([r['free_position_error_m'][-1]**2 for r in selected]))),
                'H32_full_fit_position_rmse_m':float(np.sqrt(np.mean([r['full_fit_position_error_m'][-1]**2 for r in selected]))),
                'all_lead_local_one_step_position_rmse_m':float(np.sqrt(np.mean(np.array([r['local_one_step_position_error_m'] for r in selected])**2)))}
        results[name]={'held_group':group,'retained_bank_rows':len(states),'snapshots':snapshots,'branches':rows}
        write(OUT/'audit.json',results)
        print(name,'completed',flush=True)
    guard('complete')
    write(OUT/'summary.json',{'status':'completed','wall_s':time.monotonic()-started,'snapshots':8,'branches':sum(len(v['branches']) for v in results.values()),
        'training_updates':0,'simulation_steps':0,'H32_metrics_reproduced':True,'retained_bank_group_isolation_verified':True,
        'geometry_contact_labels':False,'limitation':'Nearest states and surfaces are descriptive; thresholds mark diagnostic onset, not acceptance.'})


if __name__=='__main__':main()
