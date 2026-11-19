"""Check one group-balanced incremental common-response objective with no updates."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from s3_fitability_common import ROOT, SCOPE, run_phase, write_json
import numpy as np

from audit_s3_gradient_action_feedback import artifact, read
from ball_retention.geometry import load_tool, support_point
from ball_retention.timed_pair_data import prepare_timed_pairs
from ball_retention.world_increment_common_loss import CONTRAST, common_increment_loss
from ball_retention.world_increment_model import PACKED, WorldIncrementMLP
from preflight_s3_velocity_response_leads import lead_vectors
from preflight_s3_world_increment_supervision import SETTINGS, add, dot, mul, norm, original_gradient
from preflight_s3_world_increment_late_gradient import cosine, rms
from train_s3_world_increment import objectives, reference_data
from train_s3_world_increment_data61 import combine, extend_fold, probe, supervision
from train_s3_world_increment_supervision import losses


PHASE = 'gradient_common_supervision'
WINDOW = slice(16,24)
TERMS = ('nominal','half_difference')


def common_examples(data, fold, inventory, audit):
    nominal,signed = (data[name] for name in TERMS)
    episodes,groups = fold.pairs.episode_ids[40:],fold.pairs.group_ids[40:]
    if len(episodes)!=21 or len(set(groups))!=5 or len(nominal[0])!=14:
        raise ValueError('unexpected complete-label training membership')
    if not all(len(set(chunk))==1 for chunk in episodes.reshape(-1,3)) or len(set(episodes))!=7:
        raise ValueError('nominal-to-signed training branch order changed')
    if list(episodes)!=[row['episode'] for row in inventory]:
        raise ValueError('exact training labels no longer follow the matched fold')
    arrays = [np.concatenate((np.repeat(nominal[k].reshape(7,2,32,-1).swapaxes(0,1),3,axis=1),signed[k]),axis=0)
              [:,:,WINDOW] for k in range(3)]
    targets = arrays[2]
    observed = np.tensordot(CONTRAST,targets,axes=1)
    noise = .5*np.linalg.norm(targets[0]-targets[1],axis=-1)
    floor = np.maximum(SETTINGS['floor_m_s'],SETTINGS['noise_multiplier']*noise)
    resolved = np.linalg.norm(observed,axis=-1)>=floor
    audited = {(group,row['episode'],row['action']):row['by_lead']
               for group,rows in audit['training'].items() for row in rows}
    if len(audited)!=21:
        raise ValueError('previous coverage does not contain all complete training pairs')
    observed_model = []
    for i,(group,row) in enumerate(zip(groups,inventory,strict=True)):
        info = audited[group,row['episode'],row['direction']]
        np.testing.assert_allclose(observed[i],[info[str(h)]['physics_world_m_s'] for h in range(17,25)],
                                   atol=2e-14,rtol=0.)
        np.testing.assert_allclose(floor[i],[info[str(h)]['floor_m_s'] for h in range(17,25)],atol=2e-14,rtol=0.)
        observed_model.append([info[str(h)]['model_world_m_s'] for h in range(17,25)])
    raw = fold.pair_weights[40:,None]*resolved
    weights = np.zeros_like(raw)
    for group in np.unique(groups):
        incl = groups==group
        mass = raw[incl].sum()
        if not mass>0.:
            raise ValueError(f'no resolved common-response training steps: {group}')
        weights[incl] = .2*raw[incl]/mass
    dataset = (*arrays,weights,SETTINGS['pair_scale_m_s'])
    return dataset,groups,resolved,observed,np.asarray(observed_model)


def check_difference(model, args, gradient, seed):
    rng = np.random.default_rng(seed)
    random = {key:rng.normal(size=value.shape) for key,value in model.params.items()}
    saved = {key:value.copy() for key,value in model.params.items()}
    report = {}
    for name,vector in (('random',random),('common_gradient',gradient)):
        direction = mul(vector,1./norm(vector))
        try:
            values = []
            for sign in (1.,-1.):
                for key,value in model.params.items():
                    value[...] = saved[key]+sign*SETTINGS['fd_step']*direction[key]
                values.append(common_increment_loss(model,*args)[0])
        finally:
            for key,value in model.params.items():
                value[...] = saved[key]
        actual = (values[0]-values[1])/(2*SETTINGS['fd_step'])
        expected = dot(gradient,direction)
        np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=2e-5)
        report[name] = {'analytic':expected,'finite_difference':actual,'absolute_error':abs(actual-expected)}
    return report


def contributions(model, args, groups, resolved, value, gradient, error):
    states,known,target,weights,scale = args
    parts,report = {},{}
    grad_norm = norm(gradient)
    for group in np.unique(groups):
        incl = groups==group
        mass = float(weights[incl].sum())
        np.testing.assert_allclose(mass,.2,rtol=0.,atol=2e-15)
        subset,local,err = common_increment_loss(model,states[:,incl],known[:,incl],target[:,incl],weights[incl]/mass,scale)
        np.testing.assert_allclose(err,error[incl],atol=2e-14,rtol=0.)
        parts[group] = mul(local,mass)
        report[group] = {'pairs':int(incl.sum()),'resolved_steps':int(resolved[incl].sum()),'weight_mass':mass,
            'weighted_loss':mass*subset,'loss_fraction':mass*subset/value,
            'weighted_gradient_norm':norm(parts[group]),
            'projection_fraction_of_total_gradient':dot(parts[group],gradient)/grad_norm**2}
    np.testing.assert_allclose(sum(row['weighted_loss'] for row in report.values()),value,atol=2e-14,rtol=0.)
    for key,part in add(*parts.values()).items():
        np.testing.assert_allclose(part,gradient[key],atol=2e-13,rtol=0.)
    return report


def run(plan,out,guard):
    if plan['updates']!=0 or plan['policy']['first_last_lead']!=[17,24] or plan['policy']['group_weight']!=.2:
        raise ValueError('unexpected frozen common supervision scope')
    fit = read(plan,plan['fit_plan'])
    valid_path = plan['fit_validation']
    valid = read(plan,valid_path)
    if ((valid['status'],valid['training_pairs'],valid['updates'])!=('completed',61,900)
            or any(valid[k]!=v for k,v in SCOPE.items())
            or valid['plan_sha256']!=plan['files_sha256'][plan['fit_plan']]):
        raise ValueError('missing accepted 61-pair endpoint')
    protocol,cfg,scales,fold,held,draws = reference_data(fit,guard)
    new = [prepare_timed_pairs(ROOT,p,fit['files_sha256'][p]) for p in fit['new_timed_manifests']]
    fold,groups = extend_fold(fold,new)
    first_check = artifact(plan,valid_path,'data_preflight.json')
    old_fit = artifact(plan,valid_path,'fit_preflight.json')
    np.testing.assert_array_equal(fold.pair_weights,first_check['expanded_weights'])
    data,inventory = supervision(fit,fold,fit['timed_manifests'],guard)
    parent = read(plan,fit['matched_fit_plan'])
    coeff = read(plan,parent['window_protocol'])['auxiliary']['coefficients']
    if (len(fold.pair_weights)!=61 or groups!=first_check['group_weights']
            or inventory!=first_check['label_inventory'] or coeff!=first_check['coefficients']):
        raise ValueError('original matched weights, auxiliary labels or coefficients changed')
    audit = artifact(plan,plan['coverage_validation'],'common_coverage.json')
    args,common_groups,resolved,observed,audited_model = common_examples(data,fold,inventory,audit)
    model_cfg = held[1]
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'),model_cfg['initial']['anchor_xy_tool_m'])[0]
    leads = lead_vectors(read(plan,read(plan,fit['coverage_design'])['reference_design']),32)
    ck = Path(valid_path).with_name('world_increment_900.npz')
    if valid['artifacts_sha256'][ck.name]!=plan['files_sha256'][ck.as_posix()]:
        raise ValueError('checkpoint provenance changed')
    with np.load(ROOT/ck,allow_pickle=False) as saved:
        for key in ('known_mean','known_scale','state_mean','state_scale'):
            np.testing.assert_array_equal(saved[key],getattr(fold,key))
        if saved['updates'].item()!=900:
            raise ValueError('expected frozen update 900')
        endpoint = WorldIncrementMLP.from_state_dict(saved)
    initial = WorldIncrementMLP(PACKED,fold.state_mean,fold.state_scale,endpoint.dt_s,plan['seed'],
        inc_center=protocol['world_increment_center_m_s'],hidden=endpoint.hidden,residual_gain=endpoint.residual_gain)
    horizons = cfg['anchors']['horizons']
    rng = np.random.default_rng(plan['seed'])
    stream = [(h,rng.integers(len(fold.anchors[h].initial),size=cfg['anchors']['batch_size']))
              for h in (horizons[i%len(horizons)] for i in range(900))]
    digest = sha256(b''.join(np.r_[h,take].astype('<i8').tobytes() for h,take in stream)).hexdigest()
    if any(x!=digest for x in (draws,valid['sampler_sha256'],first_check['sampler_sha256'],old_fit['sampler_sha256'])):
        raise ValueError('S2 sampler differs from frozen 61-pair fit')

    gradients,report = {},{}
    for name,model in (('initialization',initial),('61_pairs_900',endpoint)):
        old_terms,old_losses = original_gradient(model,fold,cfg,scales,leads,model_cfg,anchor,guard,draws)
        old_aux = losses(model,data)
        old = add(*old_terms.values(),*(mul(old_aux[k][1],coeff[k]) for k in TERMS))
        evaluate,*_ = objectives(model,fold,cfg,scales,leads,model_cfg,anchor)
        first_probe,_ = probe(model,combine(model,evaluate,data,coeff),*stream[0],plan['seed'])
        if name=='initialization' and (first_probe['pair']!=old_fit['initial_pair_loss']
                or first_probe['s2_loss']!=old_fit['initial_s2_loss']):
            raise ValueError('original opening loss did not reproduce')
        loss,gradient,error = common_increment_loss(model,*args)
        if name=='61_pairs_900':
            np.testing.assert_allclose(error+observed,audited_model,atol=2e-14,rtol=0.)
        components = {'original':old,'nominal':old_aux['nominal'][1],
                      'half_difference':old_aux['half_difference'][1],'common':gradient}
        gradients[name]=components
        report[name]={'opening_batch':first_probe,'mean_original_parts_before_auxiliary':old_losses,
            'common_loss':loss,'weighted_label_rms_m_s':rms(observed,args[3]),
            'weighted_common_prediction_rms_m_s':rms(error+observed,args[3]),
            'weighted_common_error_rms_m_s':rms(error,args[3]),
            'gradient_norms':{k:norm(v) for k,v in components.items()},
            'common_cosine_to_existing':{k:cosine(gradient,v) for k,v in {**components,**old_terms}.items() if k!='common'},
            'finite_difference':check_difference(model,args,gradient,plan['seed']),
            'groups':contributions(model,args,common_groups,resolved,loss,gradient,error)}
        guard(name,log=True)

    policy = plan['policy']
    endpoint_terms = gradients['61_pairs_900']
    coefficient = policy['endpoint_norm_fraction']*norm(endpoint_terms['original'])/norm(endpoint_terms['common'])
    for name,terms in gradients.items():
        extra = mul(terms['common'],coefficient)
        merged = add(terms['original'],extra)
        report[name]['calibration']={'extra_fraction':norm(extra)/norm(terms['original']),
            'old_direction_cosines':{k:cosine(terms['original'],v) for k,v in terms.items()},
            'merged_direction_cosines':{k:cosine(merged,v) for k,v in terms.items()}}
    init_gate = report['initialization']['calibration']['extra_fraction']<=policy['max_initial_norm_fraction']
    endpoint_gate = all(x is not None and x>0. for x in report['61_pairs_900']['calibration']['merged_direction_cosines'].values())
    write_json(out/'common_supervision.json',{'checkpoints':report,'calibrated_coefficient':coefficient,
        'old_auxiliary_coefficients':coeff,'weights':args[3].tolist(),'group_ids':common_groups.tolist(),
        'label_inventory':inventory,'resolved_steps':int(resolved.sum()),'sampler_sha256':draws,
        'checks':{'initial_norm_cap':init_gate,'endpoint_joint_descent':endpoint_gate},
        'fixed_policy':policy})
    return {'fixed_parameter_sets':len(report),'training_pairs':61,'complete_common_pairs':len(inventory),
            'old_opening_loss_and_sampler_reproduced':True,'finite_differences_and_group_closure_verified':True,
            'match_ready':bool(init_gate and endpoint_gate)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
