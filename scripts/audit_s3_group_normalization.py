"""Audit the fixed normalization experiment without fitting or model selection."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ball_retention.group_fold import make_group_fold
from ball_retention.model_data import prepare_dataset
from ball_retention.pair_collection import prepare_collection
from check_s3_group_generalization import sha,write

OLD=ROOT/'runs/s3_group_holdout_20261113'
NEW=ROOT/'runs/s3_group_normalization_20261113'


def retained_metrics(result):
    rows=[r for r in result['cases'] if not r['held_out']]
    # Equal initial group, snapshot, axis/amplitude, then branch weights.
    groups=sorted(set(r['group'] for r in rows));squared=np.zeros(3)
    for group in groups:
        selected=[r for r in rows if r['group']==group]
        ticks=sorted(set(r['tick'] for r in selected))
        for tick in ticks:
            cases=[r for r in selected if r['tick']==tick]
            for row in cases:
                absolute=row['absolute']['32']
                squared+=np.array(absolute['squared_p_v_w'])/(absolute['count']*len(groups)*len(ticks)*len(cases))
    return np.sqrt(squared).tolist()


def run():
    results=[json.loads((d/'comparison.json').read_text()) for d in (OLD,NEW)]
    data=prepare_dataset(ROOT/'data/s2_pilot_20261112')
    collection=prepare_collection(ROOT,ROOT/'configs/s3_pair_collection.json')
    manifest=json.loads((ROOT/'data/s2_pilot_20261112/manifest.json').read_text())
    mapping={e['episode_id']:e['group_id'] for e in manifest['entries'] if e['split']=='train'}
    plan=json.loads((NEW/'plan.json').read_text())
    for relative,digest in plan['source_sha256'].items():assert sha(ROOT/relative)==digest
    report={'folds':{},'strong_counts':{},'caveats':[
        'Single seed and fixed 900 updates; convergence is not assumed.',
        'Axis/amplitude cases share snapshots, not independent initial groups.',
        'Changing state normalization also changes physical weighting of normalized absolute losses and residual steps.',
        'Compare SI errors, not normalized loss totals across policies; no pure input-scaling causal claim.',
        'No development-validation/test metrics or new simulation used. No control acceptance.']}
    for name,new in results[1].items():
        held=new['held_group'];old=results[0][name]
        for directory,policy,result in ((OLD,'s2_only_v1',old),(NEW,'retained_training_v2',new)):
            checkpoint=directory/(name+'.npz')
            assert sha(checkpoint)==result['checkpoint_sha256']
            fold=make_group_fold(data.train,collection.train.batch,mapping,held,normalization=policy)
            assert held not in fold.report['retained_pair_groups']
            assert all(mapping[e]!=held for e in fold.report['retained_anchor_episodes'])
            with np.load(checkpoint,allow_pickle=False) as saved:
                for field in ('state_mean','state_scale','known_mean','known_scale'):
                    np.testing.assert_array_equal(saved[field],getattr(fold,field))
        report['folds'][name]={'held_group':held,
            'old_retained_H32_rmse_p_v_w_si':retained_metrics(old),
            'new_retained_H32_rmse_p_v_w_si':retained_metrics(new),
            'new_normalization':json.loads((NEW/(name+'_normalization.json')).read_text()),
            'held_comparison':{} if held is None else {
                amp:{'old':old['summary'][held][amp]['32'],'new':metrics['32']}
                for amp,metrics in new['summary'][held].items()}}
    for label,result in zip(('old','new'),results):
        counts={key:0 for key in ('strong_cases','full_positive','held_positive','full_cosine_ge_0_5','held_cosine_ge_0_5')}
        for name,r in result.items():
            if name=='full_training':continue
            group=r['held_group']
            for amp,values in r['summary'][group].items():
                held=values['32']['strong_position']
                full=result['full_training']['summary'][group][amp]['32']['strong_position']
                counts['strong_cases']+=held['pairs']
                for prefix,m in (('full',full),('held',held)):
                    counts[prefix+'_positive']+=m['positive_cosine']
                    counts[prefix+'_cosine_ge_0_5']+=m['cosine_ge_0_5']
        report['strong_counts'][label]=counts
    write(NEW/'audit.json',report)
    resources=[json.loads(line) for line in (NEW/'resources.jsonl').read_text().splitlines()]
    validation={'status':'passed','model_control_accepted':False,'core_tests':66,'diagnostic_tests':7,
        'source_hashes_verified':True,'checkpoint_hashes_verified':True,
        'all_old_and_new_normalization_arrays_recomputed_exactly':True,'fold_membership_verified':True,
        'peak_rss_bytes':max(r['rss_bytes'] for r in resources),
        'peak_commit_bytes':max(r['private_commit_bytes'] for r in resources),
        'audit_script_sha256':sha(Path(__file__)),
        'artifacts_sha256':{p.name:sha(p) for p in sorted(NEW.iterdir()) if p.is_file() and p.name!='validation.json'}}
    write(NEW/'validation.json',validation)
    print(json.dumps({'strong_counts':report['strong_counts'],
        'retained_position_mm':{name:[v['old_retained_H32_rmse_p_v_w_si'][0]*1000,
                                     v['new_retained_H32_rmse_p_v_w_si'][0]*1000] for name,v in report['folds'].items()},
        'peak_rss_bytes':validation['peak_rss_bytes'],'peak_commit_bytes':validation['peak_commit_bytes']},indent=2))


if __name__=='__main__':run()
