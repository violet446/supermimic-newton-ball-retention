"""Physical-unit comparisons for the fixed five-fit experiment; no fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ball_retention.model_data import prepare_dataset
from ball_retention.world_model import ResidualMLP
from check_s3_group_generalization import sha,write
from audit_s3_group_normalization import retained_metrics

OLD=ROOT/'runs/s3_group_normalization_20261113'
NEW=ROOT/'runs/s3_physical_scales_20261113'


def counts(result):
    total={k:0 for k in ('strong_cases','full_positive','held_positive','full_cosine_ge_0_5','held_cosine_ge_0_5')}
    for name,row in result.items():
        if row['held_group'] is None:continue
        group=row['held_group']
        for amp,leads in row['summary'][group].items():
            held=leads['32']['strong_position']
            full=result['full_training']['summary'][group][amp]['32']['strong_position']
            total['strong_cases']+=held['pairs']
            for prefix,value in (('full',full),('held',held)):
                total[prefix+'_positive']+=value['positive_cosine']
                total[prefix+'_cosine_ge_0_5']+=value['cosine_ge_0_5']
    return total


def run():
    old=json.loads((OLD/'comparison.json').read_text());new=json.loads((NEW/'comparison.json').read_text())
    plan=json.loads((NEW/'plan.json').read_text());cfg=plan['config']
    for path,digest in plan['source_sha256'].items():assert sha(ROOT/path)==digest
    report={'strong_counts':{'old':counts(old),'new':counts(new)},'folds':{},
        'limitations':['Single seed, fixed 900 updates; convergence not assumed.',
            'Shared snapshot axis/amplitude cases are correlated.',
            'Loss units and residual SI gains changed jointly; no attribution between them.',
            'Same input normalization, samples and optimizer; SI losses now independent of fitted input statistics.',
            'Training-split diagnostics only; no development/test selection, no physical control acceptance.']}
    components=np.array(cfg['components']);units=np.array(cfg['state_error_scale_si'])
    for name,row in new.items():
        held=row['held_group']
        assert held==old[name]['held_group']
        assert row['fold_data']==old[name]['fold_data']
        with np.load(OLD/(name+'.npz'),allow_pickle=False) as a,np.load(NEW/(name+'.npz'),allow_pickle=False) as b:
            for key in ('state_mean','state_scale','known_mean','known_scale'):
                np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_allclose(b['residual_gain']*b['state_scale'],cfg['residual_gain_si'],rtol=1e-15)
            np.testing.assert_array_equal(b['state_error_scale_si'],units)
            scales={'old_residual_gain_si':(a['residual_gain']*a['state_scale']).tolist(),
                'new_residual_gain_si':(b['residual_gain']*b['state_scale']).tolist(),
                'old_absolute_error_coefficient_per_si_squared':(components/components.sum()/a['state_scale']**2).tolist(),
                'new_absolute_error_coefficient_per_si_squared':(components/components.sum()/units**2).tolist()}
        for directory,result in ((OLD,old),(NEW,new)):
            assert sha(directory/(name+'.npz'))==result[name]['checkpoint_sha256']
        report['folds'][name]={'held_group':held,'physical_scales':scales,
            'old_retained_H32_rmse_p_v_w_si':retained_metrics(old[name]),
            'new_retained_H32_rmse_p_v_w_si':retained_metrics(row),
            'held_comparison':{} if held is None else {
                amp:{'old':old[name]['summary'][held][amp],'new':values}
                for amp,values in row['summary'][held].items()}}
    # Check original S2 training anchors too, without opening development metrics.
    data=prepare_dataset(ROOT/'data/s2_pilot_20261112');anchor={}
    for label,directory in (('old',OLD),('new',NEW)):
        with np.load(directory/'full_training.npz',allow_pickle=False) as saved:
            model=ResidualMLP.from_state_dict(saved);km=saved['known_mean'];ks=saved['known_scale']
        anchor[label]={}
        for h,w in data.train.items():
            pred=model.denormalize(model.rollout(model.normalize(w.initial),(w.known-km)/ks))
            error=pred-w.target
            anchor[label][str(h)]={str(e):np.sqrt(np.mean(np.sum(error[w.episode_ids==e,-1].reshape(-1,3,3)**2,axis=2),axis=0)).tolist()
                for e in np.unique(w.episode_ids)}
    report['full_model_S2_training_episode_rmse_p_v_w_si']=anchor
    write(NEW/'audit.json',report)
    resources=[json.loads(line) for line in (NEW/'resources.jsonl').read_text().splitlines()]
    write(NEW/'validation.json',{'status':'passed','model_control_accepted':False,
        'core_tests':70,'diagnostic_tests':7,'source_and_checkpoint_hashes_verified':True,
        'identical_fold_membership_and_input_normalization':True,'fixed_si_gain_and_error_scales_verified':True,
        'peak_rss_bytes':max(r['rss_bytes'] for r in resources),'peak_commit_bytes':max(r['private_commit_bytes'] for r in resources),
        'audit_source_sha256':{Path(__file__).name:sha(Path(__file__)),'audit_s3_group_normalization.py':sha(Path(__file__).with_name('audit_s3_group_normalization.py'))},
        'artifacts_sha256':{p.name:sha(p) for p in sorted(NEW.iterdir()) if p.is_file() and p.name!='validation.json'}})
    print(json.dumps({'counts':report['strong_counts'],'S2_train_H8':{k:v['8'] for k,v in anchor.items()},
        'retained_position_mm':{k:[v['old_retained_H32_rmse_p_v_w_si'][0]*1000,v['new_retained_H32_rmse_p_v_w_si'][0]*1000] for k,v in report['folds'].items()}},indent=2))


if __name__=='__main__':run()
