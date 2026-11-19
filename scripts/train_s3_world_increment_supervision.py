"""Fit the frozen one-step auxiliary objective through the original matched driver."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from s3_fitability_common import ROOT, forecast, run_phase, write_json

import numpy as np

import train_s3_world_increment as matched
from audit_s3_ball_velocity_components import branch_terms, frame_series, rms
from ball_retention.archive_state_import import _parse_state
from ball_retention.model_data import _ball_state, known_rollout, prepare_dataset
from ball_retention.pair_collection import prepare_collection
from ball_retention.world_increment_loss import world_increment_loss
from ball_retention.world_increment_model import WorldIncrementMLP
from preflight_s3_world_increment_supervision import SETTINGS, add, examples, mul
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held, regression, report


PHASE = 'world_increment_supervision_fit'


def losses(model, dataset):
    return {k:world_increment_loss(model,*v,paired=k=='half_difference') for k,v in dataset.items()}


def loss_summary(data, dataset):
    return {k:{'loss':l,'rms_error_m_s':float(np.sqrt(np.sum(dataset[k][3]*np.sum(error**2,axis=-1))))}
            for k,(l,_,error) in data.items()}


def nominal_score(model, held, fold, anchor):
    _,cfg,fields,traces,_ = held
    state = _parse_state(fields)
    trace = traces['nominal_1']
    initial = _ball_state(state)
    truth = trace['raw']
    raw = known_rollout(state.time_s,state.tilt_rad,state.tilt_rate_rad_s,trace['actions'],cfg,model.dt_s)
    known = model.features(raw,cfg,anchor,fold.known_mean,fold.known_scale)
    frame = frame_series(state,trace,model.dt_s)
    path = np.vstack((initial,truth))
    gravity = np.array([0.,0.,-cfg['physics']['gravity_m_s2']*model.dt_s])
    target = branch_terms(frame,path,gravity)[0]['non_gravity']
    output = {}
    for mode in ('local','free'):
        pred = forecast(model,initial[None],known[None],truth[None],local=mode=='local')[0]
        terms = branch_terms(frame,path,gravity,endpoint=pred,independent=mode=='local')[0]
        world_error = frame[0][1:].apply(terms['non_gravity']-target)
        errors = pred-truth
        output[mode] = {'h8_h24_world_increment_rmse_m_s':rms(world_error[7:24]),
                        'h8_h24_world_increment_mean_error_m_s':world_error[7:24].mean(axis=0).tolist(),
                        'h32_position_error_m':float(np.linalg.norm(errors[-1,:3])),
                        'h32_velocity_error_m_s':float(np.linalg.norm(errors[-1,3:6]))}
    return output


def run(plan,out,guard):
    def read(p):
        return json.loads(bound(plan,p))

    protocol = read(plan['supervision_protocol'])
    audit_path = Path(protocol['accepted_preflight'])
    audit = read(audit_path)
    detail_path = audit_path.with_name('supervision.json')
    details = read(detail_path)
    if (audit['status']!='completed' or not audit['matched_loss_supported'] or audit['updates']!=0
            or audit['artifacts_sha256'][detail_path.name]!=plan['files_sha256'][detail_path.as_posix()]
            or protocol['auxiliary_settings']!=SETTINGS or protocol['auxiliary_coefficients']!=details['coefficients']
            or plan['updates']!=protocol['fixed_updates'] or protocol['seed']!=plan['seed']
            or protocol['budget']!=plan['budget'] or protocol['scope']!=plan['scope']):
        raise ValueError('unexpected auxiliary scope or acceptance')
    base = read(protocol['reference_fit_plan'])
    for key in ('budget','scope','seed','updates','held_design','coverage_design','accepted_preflight','timed_manifests'):
        if plan[key]!=base[key]:
            raise ValueError(f'matched fit recipe changed: {key}')
    ck = Path(protocol['reference_checkpoint'])
    status = read(ck.with_name('validation.json'))
    if (status['status'],status['updates'],status['training_pairs'],status['sampler_sha256']) != (
            'completed',900,49,protocol['sampler_sha256']):
        raise ValueError('world residual matched endpoint changed')
    for name in (ck.name,'comparison.json','timed_rollouts.json'):
        path = ck.with_name(name)
        if status['artifacts_sha256'][name] != plan['files_sha256'][path.as_posix()]:
            raise ValueError(f'world residual reference artifact changed: {name}')
    reference = read(ck.with_name('comparison.json'))
    with np.load(ROOT/ck,allow_pickle=False) as store:
        incumbent = WorldIncrementMLP.from_state_dict(store)
    coefficients = protocol['auxiliary_coefficients']
    original = matched.objectives
    context = {}

    def objectives(model,fold,cfg,scales,leads,model_cfg,anchor):
        evaluate,initial,features = original(model,fold,cfg,scales,leads,model_cfg,anchor)
        data,coverage,counts = examples(plan,fold,guard)
        if (coverage!=details['coverage'] or counts!=details['coverage_by_snapshot']
                or data['nominal'][3].tolist()!=details['nominal_weights']
                or data['half_difference'][3].tolist()!=details['half_difference_weights']):
            raise ValueError('auxiliary labels, order, or effective masks changed')
        held_design = read(plan['held_design'])
        held = read_held(held_design)
        score, trajectories = matched.score_candidate(incumbent,held,fold,anchor,guard)
        if score!=reference['held_timing'] or trajectories!=read(ck.with_name('timed_rollouts.json')):
            raise ValueError('world residual reference development scoring did not reproduce')
        legacy = prepare_collection(ROOT,ROOT/held_design['base_collection'])
        s2 = prepare_dataset((ROOT/held_design['s2_manifest']).parent,horizons=tuple(cfg['anchors']['horizons']),verify_test=False)
        histories = {r['episode_id']:r['history_id'] for r in read(cfg['history_plan'])['episodes']}
        evaluation = report(matched.LegacyEvaluation(incumbent,model_cfg,anchor,fold),
                            fold.known_mean,fold.known_scale,legacy,s2,histories)
        evaluation['pairs'] = [r for r in evaluation['pairs'] if r['split']=='validation']
        if evaluation!=reference['legacy_validation']:
            raise ValueError('world residual reference legacy protections did not reproduce')
        before,initial_aux = (loss_summary(losses(x,data),data) for x in (incumbent,model))
        for label,current in (('endpoint',before),('initialization',initial_aux)):
            for key,v in current.items():
                expected = details['checkpoints'][label]['auxiliary'][key]
                for quantity in ('loss','rms_error_m_s'):
                    np.testing.assert_allclose(v[quantity],expected[quantity],rtol=0.,atol=1e-14)
        context.update(dataset=data,before=before,held=held,fold=fold,anchor=anchor)
        write_json(out/'supervision_binding.json',{'coverage':coverage,'coefficients':coefficients,
            'weights_exact':True,'zero_output_losses_exact':True,'matched_reference_reproduced':True,
            'reference_auxiliary_endpoint':before})

        def combined(h,take):
            anchor_loss,pair,grad = evaluate(h,take)
            values = losses(model,data)
            extra = sum(coefficients[k]*v[0] for k,v in values.items())
            total = add(grad,*(mul(v[1],coefficients[k]) for k,v in values.items()))
            return anchor_loss,{**pair,'loss':pair['loss']+extra,'original_pair_loss':pair['loss'],
                                'auxiliary':{k:v[0] for k,v in values.items()}},total

        return combined,initial,features

    # Limit substitution to the objective factory. The frozen optimizer,
    # sampler, checkpoint write and score paths execute unmodified.
    with patch.object(matched,'objectives',objectives):
        result = matched.run(plan,out,guard)
    if result['sampler_sha256']!=protocol['sampler_sha256']:
        raise ValueError('trained sampler hash drift')
    with np.load(out/'world_increment_900.npz',allow_pickle=False) as store:
        fitted = WorldIncrementMLP.from_state_dict(store)
    comparison = json.loads((out/'comparison.json').read_bytes())
    protection = regression(reference['legacy_validation'],comparison['legacy_validation'])
    if len(protection['checks'])!=22:
        raise ValueError('expected 22 matched world-model protections')
    data = context['dataset']
    assessment = {'legacy_protection_vs_world_increment':protection,
        'reference_train_loss':context['before'],
        'new_train_loss':loss_summary(losses(fitted,data),data),
        'development_nominal':{k:nominal_score(x,context['held'],context['fold'],context['anchor'])
                               for k,x in (('reference',incumbent),('supervised',fitted))}}
    write_json(out/'supervision_comparison.json',assessment)
    return {**result,'reference_auxiliary_implementation_verified':True,
            'replacement_gate_passed':result['replacement_gate_passed'] and protection['passed']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
