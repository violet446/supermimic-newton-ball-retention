"""Measure nominal and action response world-increment supervision at fixed parameters."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np

from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state, known_rollout
from ball_retention.velocity_response_loss import paired_velocity_response_loss
from ball_retention.world_increment_loss import world_increment_loss
from ball_retention.world_increment_model import PACKED, WorldIncrementMLP
from preflight_s3_velocity_response_leads import lead_vectors
from preflight_s3_world_increment import reference_data
from train_s3_local_velocity_increment import bound


PHASE = 'world_increment_supervision_preflight'
FIT = Path('runs/s3_world_increment_fit_20261117')
SETTINGS = {'nominal_scale_m_s': .025, 'pair_scale_m_s': .0025,
            'pair_first_lead':1,'pair_last_lead':16,'floor_m_s':1e-5,'noise_multiplier':5.,
            'endpoint_aux_norm_fraction':.2,'max_init_aux_norm_fraction':.5,'fd_step':1e-6}


def raw_example(state, trace, cfg):
    rows = [state]+[_parse_state(x) for x in trace['states']]
    if len(rows) != 33 or trace['substeps'] != [16]*32:
        raise ValueError('expected one complete archived H32 branch')
    dt = cfg['physics']['dt_s']*16
    np.testing.assert_allclose(np.diff([s.time_s for s in rows]),dt,atol=1e-12,rtol=0.)
    known = known_rollout(state.time_s,state.tilt_rad,state.tilt_rate_rad_s,trace['actions'],cfg,dt)
    raw = np.asarray([_ball_state(s) for s in rows[:-1]])
    world = np.asarray([s.ball.linear_velocity_origin for s in rows])
    net = np.diff(world,axis=0)-np.array([0.,0.,-cfg['physics']['gravity_m_s2']*dt])
    return raw,known,net


def examples(plan, fold, guard):
    nominal, signed, masks, reports = [], [], [], []
    counts = {}
    for path in plan['timed_manifests']:
        manifest = json.loads(bound(plan,path))
        folder = Path(path).parent
        for entry in manifest['entries']:
            if entry['split'] != 'train':
                raise ValueError('unexpected non-training episode')
            cfg = json.loads(bound(plan,entry['source_config']))

            def load_ref(ref):
                file = folder/ref['path']
                if plan['files_sha256'][file.as_posix()] != ref['sha256']:
                    raise ValueError('archive branch hash changed')
                return json.loads(bound(plan,file))

            state = _parse_state(load_ref(entry['native_snapshot'])['state'])
            a,b = (raw_example(state,load_ref(x),cfg) for x in entry['nominal_branches'])
            nominal.extend((a,b))
            noise = .5*np.linalg.norm(a[2]-b[2],axis=-1)
            floor = np.maximum(SETTINGS['floor_m_s'],SETTINGS['noise_multiplier']*noise)
            count = 0
            for pair in entry['pairs']:
                aa,bb = (raw_example(state,load_ref(x),cfg) for x in sorted(pair['branches'],key=lambda x:-x['sign']))
                row = 40+len(signed)
                if fold.pairs.episode_ids[row] != entry['episode_id'] or fold.pairs.snapshot_ticks[row] != entry['snapshot_tick']:
                    raise ValueError('exact pair order changed')
                np.testing.assert_array_equal(aa[1],fold.pairs.plus_known[row])
                np.testing.assert_array_equal(bb[1],fold.pairs.minus_known[row])
                signed.append((aa,bb))
                value = .5*np.linalg.norm(aa[2]-bb[2],axis=-1)
                mask = value >= floor
                mask[SETTINGS['pair_last_lead']:] = False
                mask[:SETTINGS['pair_first_lead']-1] = False
                masks.append(mask)
                count += int(mask.sum())
                reports.append({'episode':entry['episode_id'],'direction':pair['direction_id'],
                    'h1_h16_resolved':int(mask.sum()),'h1_h16_strong':int(np.sum(value[:16]>=10*floor[:16])),
                    'max_repeat_half_spread_m_s':float(noise.max()),
                    'h1_h16_target_half_rms_m_s':float(np.sqrt(np.mean(value[:16]**2)))})
            counts[entry['episode_id']] = count
            guard(entry['episode_id'])
    if len(nominal)!=6 or len(signed)!=9:
        raise ValueError('unexpected exact-label scope')
    nominal_state,nominal_known,nominal_target = (np.stack([x[i] for x in nominal]) for i in range(3))
    pair_state,pair_known,pair_target = (np.stack([np.stack([x[i] for x in pair]) for pair in signed],axis=1) for i in range(3))
    nominal_weights = np.full((6,32),1/(6*32))
    mask = np.stack(masks)
    raw = fold.pair_weights[40:,None]*mask
    denom = raw.sum()
    if denom <= 0.:
        raise ValueError('no identifiable one-step half differences')
    pair_weights = raw/denom
    assert fold.pairs.initial.shape[0] == 49
    args = {'nominal':(nominal_state,(nominal_known-fold.known_mean)/fold.known_scale,nominal_target,nominal_weights,
                       SETTINGS['nominal_scale_m_s']),
            'half_difference':(pair_state,(pair_known-fold.known_mean)/fold.known_scale,pair_target,pair_weights,
                               SETTINGS['pair_scale_m_s'])}
    return args,reports,counts


def add(*gradients):
    return {k:sum(g[k] for g in gradients) for k in gradients[0]}


def mul(gradient, weight):
    return {k:weight*v for k,v in gradient.items()}


def dot(a,b):
    return float(sum(np.vdot(a[k],b[k]) for k in a))


def norm(a):
    return dot(a,a)**.5


def original_gradient(model, fold, cfg, scales, leads, model_cfg, anchor, guard, draws):
    feature = lambda v:model.features(v,model_cfg,anchor,fold.known_mean,fold.known_scale)
    cfg_s2, rng = cfg['anchors'],np.random.default_rng(20261111)
    horizons = cfg_s2['horizons']
    units, components = np.asarray(scales['state_error_scale_si']),np.asarray(cfg['components'])
    anchor_rows = {h:(model.normalize(w.initial),feature(w.known),model.normalize(w.target)) for h,w in fold.anchors.items()}
    s2 = {k:np.zeros_like(v) for k,v in model.params.items()}
    a_loss, digest = 0., sha256()
    for i in range(900):
        h = horizons[i%len(horizons)]
        x,k,y = anchor_rows[h]
        indices = rng.integers(len(x),size=cfg_s2['batch_size'])
        digest.update(np.r_[h,indices].astype('<i8').tobytes())
        a,g = model.loss_and_grad(x[indices],k[indices],y[indices],state_weights=components,state_error_scale_si=units)
        for key in s2:
            s2[key] += g[key]/900
        a_loss += a/900
        if i%90==0:
            guard()
    if digest.hexdigest()!=draws:
        raise ValueError('archived S2 draws did not reproduce')
    b = fold.pairs
    args = (model,model.normalize(b.initial),feature(np.stack((b.plus_known,b.minus_known))),
            np.stack((b.plus_target,b.minus_target)),np.stack((b.plus_mask,b.minus_mask)),
            b.response_mask,b.noise_mask,b.response_floor,leads[0],components,np.asarray(cfg['response_scales_si']))
    options = dict(velocity_lead_weights=leads[1],sample_weights=fold.pair_weights,absolute_scale_si=units)
    a,g_a = paired_velocity_response_loss(*args,cfg['pairs']['absolute_weight'],0.,**options)
    p,g_p = paired_velocity_response_loss(*args,0.,cfg['pairs']['response_weight'],**options)
    direct,g_direct = paired_velocity_response_loss(*args,cfg['pairs']['absolute_weight'],cfg['pairs']['response_weight'],**options)
    for key,value in add(g_a,g_p).items():
        np.testing.assert_allclose(value,g_direct[key],atol=1e-13,rtol=0.)
    if abs(a['loss']+p['loss']-direct['loss'])>1e-13:
        raise ValueError('original terms failed loss closure')
    return {'s2':s2,'absolute':g_a,'response':g_p},{'s2_mean':a_loss,'pair':direct}


def finite_difference(model, name, data, gradient, rng):
    direction = {k:rng.normal(size=v.shape) for k,v in model.params.items()}
    direction = mul(direction,1./norm(direction))
    backups = {k:v.copy() for k,v in model.params.items()}
    values = []
    e = SETTINGS['fd_step']
    try:
        for sign in (1.,-1.):
            for k,p in model.params.items():
                p[...] = backups[k]+sign*e*direction[k]
            values.append(world_increment_loss(model,*data,paired=name=='half_difference')[0])
    finally:
        for k,p in model.params.items():
            p[...] = backups[k]
    actual, expected = (values[0]-values[1])/(2*e),dot(gradient,direction)
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=2e-5)
    return abs(actual-expected)


def run(plan,out,guard):
    if plan['updates'] != 0 or plan['settings'] != SETTINGS:
        raise ValueError('unexpected probe scope')
    trial = json.loads(bound(plan,FIT/'validation.json'))
    ck = FIT/'world_increment_900.npz'
    if (trial['updates']!=900 or trial['status']!='completed'
            or trial['artifacts_sha256'][ck.name]!=plan['files_sha256'][ck.as_posix()]):
        raise ValueError('unexpected fitted candidate checkpoint')
    protocol,cfg,scales,fold,held,draws = reference_data(plan,guard)
    config = held[1]
    design = json.loads(bound(plan,plan['coverage_design']))
    dense = json.loads(bound(plan,design['reference_design']))
    leads = lead_vectors(dense,32)
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'),config['initial']['anchor_xy_tool_m'])[0]
    dataset,coverage,counts = examples(plan,fold,guard)
    with np.load(ROOT/ck,allow_pickle=False) as saved:
        endpoint = WorldIncrementMLP.from_state_dict(saved)
    fresh = WorldIncrementMLP(PACKED,endpoint.state_mean,endpoint.state_scale,endpoint.dt_s,plan['seed'],
                              inc_center=protocol['world_increment_center_m_s'],hidden=endpoint.hidden,
                              residual_gain=endpoint.residual_gain)
    rng = np.random.default_rng(plan['seed'])
    checkpoints,gradients = {},{}
    for label,model in (('endpoint',endpoint),('initialization',fresh)):
        guard(label,log=True)
        g,loss = original_gradient(model,fold,cfg,scales,leads,config,anchor,guard,draws)
        baseline = add(*g.values())
        gradients[label] = {'baseline':baseline}
        records = {}
        for key,args in dataset.items():
            value, gradient, err = world_increment_loss(model,*args,paired=key=='half_difference')
            gradients[label][key] = gradient
            weight = args[3]
            records[key] = {'loss':value,'rms_error_m_s':float(np.sqrt(np.sum(weight*np.sum(err**2,axis=-1)))),
                'axis_bias_m_s':np.sum(err*weight[...,None],axis=tuple(range(weight.ndim))).tolist(),
                'direction_fd_error':finite_difference(model,key,args,gradient,rng),
                'cosine_to_original':dot(gradient,baseline)/(norm(gradient)*norm(baseline)),
                'cosine_to_original_parts':{name:dot(gradient,v)/(norm(gradient)*norm(v)) for name,v in g.items()}}
        checkpoints[label] = {'losses':loss,'auxiliary':records,'gradient_norms':{k:norm(v) for k,v in gradients[label].items()}}
        guard(label,log=True)
    ep = gradients['endpoint']
    coefficients = {key:SETTINGS['endpoint_aux_norm_fraction']*norm(ep['baseline'])/norm(ep[key]) for key in dataset}
    trial = {}
    for label,g in gradients.items():
        extra = add(*(mul(g[key],coefficients[key]) for key in dataset))
        full = add(g['baseline'],extra)
        trial[label] = {'auxiliary_norm_fraction':norm(extra)/norm(g['baseline']),
                        'combined_direction_cosine':{k:dot(full,v)/(norm(full)*norm(v)) for k,v in g.items()}}
    usable = all(count>=8 for count in counts.values()) and all(row['max_repeat_half_spread_m_s'] <= 1e-5 for row in coverage)
    qualifies = (usable and trial['initialization']['auxiliary_norm_fraction'] <= SETTINGS['max_init_aux_norm_fraction']
                 and all(c>0. for c in trial['endpoint']['combined_direction_cosine'].values()))
    write_json(out/'supervision.json',{'coverage':coverage,'coverage_by_snapshot':counts,'checkpoints':checkpoints,
        'nominal_weights':dataset['nominal'][3].tolist(),'half_difference_weights':dataset['half_difference'][3].tolist(),
        'coefficients':coefficients,'trial_gradient':trial,'coverage_usable':usable,'matched_loss_supported':qualifies})
    return {'exact_pairs':9,'nominal_branches':6,'matched_loss_supported':qualifies,'sampler_sha256':draws}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
