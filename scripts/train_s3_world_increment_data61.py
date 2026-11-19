"""Run the matched 49/61-pair comparison with frozen full-window supervision."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

import train_s3_world_increment as matched
from audit_s3_world_increment_pitch_drift import EXTENDED, load_extended
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.timed_pair_data import concatenate_batches, prepare_timed_pairs
from ball_retention.timing_coverage_fit import pair_cells
from ball_retention.world_increment_model import PACKED, WorldIncrementMLP
from preflight_s3_velocity_response_leads import lead_vectors
from preflight_s3_world_increment_supervision import SETTINGS, add, dot, examples, mul, norm, raw_example
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import regression
from train_s3_world_increment_supervision import losses, nominal_score


PRE = 'world_increment_data61_preflight'
FIT = 'world_increment_data61_fit'


def extend_fold(reference, timed):
    base = reference.pairs
    added = concatenate_batches(*(x.batch for x in timed))
    prior_cells, new_cells = pair_cells(base), pair_cells(added)
    w_old = reference.pair_weights.copy()
    w_new = np.empty(len(added.initial))
    records = {}
    for group in np.unique(base.group_ids):
        indices = np.flatnonzero(base.group_ids==group)
        mass = float(w_old[indices].sum())
        existing = {k:v for k,v in prior_cells.items() if k[0]==group}
        incoming = {k:v for k,v in new_cells.items() if k[0]==group}
        new_only = incoming.keys()-existing.keys()
        if incoming:
            reserve = mass/(len(existing)+len(new_only))
            w_old[indices] *= len(existing)/(len(existing)+len(new_only))
            for key, rows in incoming.items():
                if key in existing:
                    old_rows = existing[key]
                    shared = w_old[old_rows].sum()/(len(old_rows)+len(rows))
                    w_old[old_rows] = shared
                else:
                    shared = reserve/len(rows)
                w_new[rows] = shared
        total = float(w_old[indices].sum()+w_new[added.group_ids==group].sum())
        np.testing.assert_allclose(total,mass,atol=2e-15,rtol=0.)
        records[str(group)] = {'mass_before':mass,'mass_after':total,'existing_cells':len(existing),
                               'new_cells':len(new_only),'added_rows':sum(map(len,incoming.values()))}
    weights = np.concatenate((w_old,w_new))
    if not np.all(np.isfinite(weights)&(weights>0.)):
        raise ValueError('invalid expanded pair weights')
    return replace(reference,pairs=concatenate_batches(base,added),pair_weights=weights,
                   retained_indices=np.arange(len(weights))),records


def supervision(plan, fold, manifests, guard):
    nominal, signed, valid, inventory = [], [], [], []
    for manifest in manifests:
        folder=Path(manifest).parent
        for entry in json.loads(bound(plan,manifest))['entries']:
            cfg=json.loads(bound(plan,entry['source_config']))

            def archive(ref):
                f=folder/ref['path']
                if plan['files_sha256'][f.as_posix()]!=ref['sha256']:
                    raise ValueError('label source digest mismatch')
                return json.loads(bound(plan,f))

            state=_parse_state(archive(entry['native_snapshot'])['state'])
            base=[raw_example(state,archive(ref),cfg) for ref in entry['nominal_branches']]
            nominal.extend(base)
            noise=.5*np.linalg.norm(base[0][2]-base[1][2],axis=-1)
            floor=np.maximum(SETTINGS['floor_m_s'],SETTINGS['noise_multiplier']*noise)
            for pair in entry['pairs']:
                a,b=(raw_example(state,archive(ref),cfg) for ref in sorted(pair['branches'],key=lambda r:-r['sign']))
                row=40+len(signed)
                if (entry['split']!='train' or (entry['episode_id'],entry['snapshot_tick'],entry['group_id'])
                        != (fold.pairs.episode_ids[row],fold.pairs.snapshot_ticks[row],fold.pairs.group_ids[row])):
                    raise ValueError('training label row identity changed')
                np.testing.assert_array_equal(a[1],fold.pairs.plus_known[row])
                np.testing.assert_array_equal(b[1],fold.pairs.minus_known[row])
                magnitude=.5*np.linalg.norm(a[2]-b[2],axis=-1)
                valid.append(magnitude>=floor)
                signed.append((a,b))
                inventory.append({'episode':entry['episode_id'],'direction':pair['direction_id'],
                    'resolved_H1_H16':int(np.sum(valid[-1][:16])),
                    'resolved_H17_H32':int(np.sum(valid[-1][16:])),
                    'strong_H1_H32':int(np.sum(magnitude>=10*floor)),
                    'max_nominal_repeat_spread_m_s':float(noise.max())})
            guard(entry['episode_id'])
    if len(signed)!=len(fold.pairs.initial)-40 or len(nominal)*3!=2*len(signed):
        raise ValueError('unexpected number of signed or nominal labels')
    raw=np.asarray(valid)*fold.pair_weights[40:,None]
    mass=[float(raw[:,s].sum()) for s in (slice(0,16),slice(16,32))]
    if min(mass)<=0.:
        raise ValueError('empty half-difference window')
    weights=raw*np.repeat(.5/np.asarray(mass),16)
    np.testing.assert_allclose([weights[:,:16].sum(),weights[:,16:].sum()],[.5,.5],atol=2e-15,rtol=0.)
    ns,nk,nt=(np.stack([x[k] for x in nominal]) for k in range(3))
    ps,pk,pt=(np.stack([np.stack([x[k] for x in pair]) for pair in signed],axis=1) for k in range(3))
    return {'nominal':(ns,(nk-fold.known_mean)/fold.known_scale,nt,
                       np.full(ns.shape[:-1],1/np.prod(ns.shape[:-1])),SETTINGS['nominal_scale_m_s']),
            'half_difference':(ps,(pk-fold.known_mean)/fold.known_scale,pt,weights,SETTINGS['pair_scale_m_s'])},inventory


def combine(model, evaluate, data, coefficients):
    def step(h, take):
        anchor, pair, grad=evaluate(h,take)
        terms=losses(model,data)
        total=pair['loss']+sum(coefficients[k]*v[0] for k,v in terms.items())
        metrics = {**pair, 'loss': total, 'original_pair_loss': pair['loss'],
                   'auxiliary': {k: v[0] for k, v in terms.items()}}
        gradient = add(grad, *(mul(v[1], coefficients[k]) for k, v in terms.items()))
        return anchor, metrics, gradient
    return step


def probe(model, evaluate, h, take, seed):
    a,b,g=evaluate(h,take)
    rng=np.random.default_rng(seed)
    direction={k:rng.normal(size=v.shape) for k,v in model.params.items()}
    direction=mul(direction,1/norm(direction))
    saved={k:v.copy() for k,v in model.params.items()}
    eps=1e-6
    values=[]
    try:
        for sign in (-1,1):
            for k,v in model.params.items():
                v[...]=saved[k]+sign*eps*direction[k]
            x,y,_=evaluate(h,take)
            values.append(x+y['loss'])
    finally:
        for k,v in model.params.items():
            v[...]=saved[k]
    fd=(values[1]-values[0])/(2*eps)
    analytic=dot(g,direction)
    np.testing.assert_allclose(fd,analytic,atol=2e-6,rtol=2e-5)
    return {'s2_loss':a,'pair':b,'gradient_norm':norm(g),'fd_error':abs(fd-analytic)},g


def run(plan,out,guard):
    def read(p):
        return json.loads(bound(plan,p))

    if plan['updates'] not in (0,900):
        raise ValueError('unexpected capture or fit scope')
    old_run=matched.reference_data(plan,guard)
    protocol,cfg,scales,reference,held,draws=old_run
    old_path=Path(plan['matched_fit_plan'])
    base=read(old_path)
    base_manifests=base['timed_manifests']
    newer=plan['new_timed_manifests']
    manifests=[*base_manifests,*newer]
    if (len(newer)!=4 or set(newer)&set(base_manifests) or plan['timed_manifests']!=manifests
            or any(plan[k]!=base[k] for k in ('scope','budget','seed','held_design','coverage_design'))):
        raise ValueError('matched identities or optimizer configuration changed')
    coefficients=read(base['window_protocol'])['auxiliary']['coefficients']
    entries=[]
    for p in newer:
        v=read(Path(p).with_name('validation.json'))
        if (v['status']!='completed' or (v['physics_steps'],v['pairs'],v['training_updates'])!=(6784,3,0)
                or any(v[k]!=plan['scope'][k] for k in ('test_episode_files_read','control_accepted'))
                or v['artifacts_sha256']['capture_manifest.json']!=plan['files_sha256'][p]):
            raise ValueError('unaccepted added capture')
        entries.append(prepare_timed_pairs(ROOT,p,plan['files_sha256'][p]))
    groups=set(reference.pairs.group_ids)
    names=[e['episode_id'] for v in entries for e in v.entries]
    if (len(set(names))!=4 or any(set(x.batch.group_ids)-groups for x in entries)
            or any(x.model_config!=entries[0].model_config or x.dt_s!=entries[0].dt_s for x in entries)):
        raise ValueError('new capture groups, model configuration or clock changed')
    expanded,weights=extend_fold(reference,entries)
    if (len(reference.pairs.initial),len(expanded.pairs.initial))!=(49,61):
        raise ValueError('unexpected matched fold sizes')
    for key in ('state_mean','state_scale','known_mean','known_scale'):
        np.testing.assert_array_equal(getattr(reference,key),getattr(expanded,key))
    data49,i49=supervision(plan,reference,base_manifests,guard)
    data61,i61=supervision(plan,expanded,manifests,guard)
    archived_plan={**plan,'timed_manifests':base_manifests}
    old_aux,_,_=examples(archived_plan,reference,guard)
    for mode in old_aux:
        for field in (0,1,2):
            np.testing.assert_array_equal(old_aux[mode][field],data49[mode][field])
    for mode in ('nominal','half_difference'):
        count=len(i49) if mode=='half_difference' else data49[mode][0].shape[0]
        for field in (0,1,2):
            selected=data61[mode][field][:,:count] if mode=='half_difference' else data61[mode][field][:count]
            np.testing.assert_array_equal(selected,data49[mode][field])
    model_cfg=held[1]
    anchor=support_point(load_tool(ROOT/'assets/shovel.STL'),model_cfg['initial']['anchor_xy_tool_m'])[0]
    endpoint,_,_,_,sampler=load_extended(plan,held,anchor,guard)
    if sampler!=draws:
        raise ValueError('previous sampler changed')
    initial=WorldIncrementMLP(PACKED,reference.state_mean,reference.state_scale,endpoint.dt_s,plan['seed'],
            inc_center=protocol['world_increment_center_m_s'],hidden=endpoint.hidden,residual_gain=endpoint.residual_gain)
    dense=read(read(plan['coverage_design'])['reference_design'])
    leads=lead_vectors(dense,32)
    hs=cfg['anchors']['horizons']
    rng=np.random.default_rng(plan['seed'])
    stream=[(hs[i%len(hs)], rng.integers(len(reference.anchors[hs[i%len(hs)]].initial),size=cfg['anchors']['batch_size']))
            for i in range(900)]
    digest=sha256(b''.join(np.r_[h,t].astype('<i8').tobytes() for h,t in stream)).hexdigest()
    if digest!=draws:
        raise ValueError('S2 draw stream did not reproduce')
    stats={}
    original=matched.objectives
    for label,m in (('initialization',initial),('reference_endpoint',endpoint)):
        arms=[]; gs=[]
        for fold,data in ((reference,data49),(expanded,data61)):
            evaluate,*_=original(m,fold,cfg,scales,leads,model_cfg,anchor)
            result,g=probe(m,combine(m,evaluate,data,coefficients),*stream[0],plan['seed'])
            arms.append(result); gs.append(g)
        if arms[0]['s2_loss']!=arms[1]['s2_loss']:
            raise ValueError('data expansion changed the S2 anchor objective')
        stats[label]={'49_pairs':arms[0],'61_pairs':arms[1],'gradient_cosine':dot(*gs)/(norm(gs[0])*norm(gs[1]))}
    expected=read(EXTENDED/'fit_preflight.json')['initial_pair_loss']
    actual=stats['initialization']['49_pairs']['pair']
    terms=('nominal','half_difference')
    np.testing.assert_allclose([actual['loss'],*[actual['auxiliary'][k] for k in terms]],
                               [expected['loss'],*[expected['auxiliary'][k] for k in terms]],atol=1e-14,rtol=0.)
    binding={'sampler_sha256':draws,'group_weights':weights,'first49_weights':reference.pair_weights.tolist(),
             'expanded_weights':expanded.pair_weights.tolist(),'label_inventory':i61,'finite_differences':stats,
             'coefficients':coefficients,'new_manifest_count':len(newer)}
    write_json(out/'data_preflight.json',binding)
    if plan['updates']==0:
        return {'training_pairs':61,'sampler_sha256':draws,'matched_data_preflight_passed':True}

    pre=read(plan['accepted_61_preflight'])
    pre_artifact=Path(plan['accepted_61_preflight']).with_name('data_preflight.json')
    if (pre['status']!='completed' or pre['updates']!=0 or not pre['matched_data_preflight_passed']
            or pre['artifacts_sha256'][pre_artifact.name]!=plan['files_sha256'][pre_artifact.as_posix()]
            or read(pre_artifact)!=binding):
        raise ValueError('executable data preflight differs from accepted preflight')

    def objectives(model,fold,cfg,scales,leads,model_cfg,anchor):
        evaluate,initial,features=original(model,fold,cfg,scales,leads,model_cfg,anchor)
        return combine(model,evaluate,data61,coefficients),initial,features

    with patch.multiple(matched,reference_data=lambda *args:(protocol,cfg,scales,expanded,held,draws),
                        objectives=objectives):
        result=matched.run(plan,out,guard)
    reference_score=read(EXTENDED/'comparison.json')
    candidate=json.loads((out/'comparison.json').read_bytes())
    protection=regression(reference_score['legacy_validation'],candidate['legacy_validation'])
    with np.load(out/'world_increment_900.npz',allow_pickle=False) as saved:
        fitted=WorldIncrementMLP.from_state_dict(saved)
    write_json(out/'expanded_data_comparison.json',{'protections_vs_49_full_window':protection,
        'development_nominal':{k:nominal_score(m,held,expanded,anchor) for k,m in (('49_pairs',endpoint),('61_pairs',fitted))}})
    return {**result,'training_pairs':len(expanded.pairs.initial),'matched_data_preflight_passed':True,
            'replacement_gate_passed':result['replacement_gate_passed'] and protection['passed']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    plan_path=args.plan.resolve()
    phase=json.loads(plan_path.read_bytes())['phase']
    if phase not in (PRE,FIT):
        parser.error('unknown data comparison phase')
    run_phase(plan_path,args.sha256,phase,run)
