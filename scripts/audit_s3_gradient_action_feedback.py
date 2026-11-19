"""Decompose lateral position error on the two frozen gradient-action replays."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from audit_s3_world_increment_drift import WINDOWS, require_close
from audit_s3_world_increment_generation import propagate, world_head
from audit_s3_world_increment_newton_reference import (
    propagate_discrete_position, required_residual, tool_transition,
)
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import KNOWN_NAMES, _ball_state, known_rollout
from ball_retention.world_increment_model import WorldIncrementMLP
from collect_s3_gradient_action import IDENTITIES, LABELS, load_case, load_plan, nominal_gate, score_case
from deterministic_archive_metrics import _full_branch
from train_s3_local_velocity_increment import bound


PHASE = 'gradient_action_feedback'
TRACKS = ('nominal','common','half_difference','suggestion_minus_nominal')
WEIGHTS = np.array([[.5,.5,0.,0.],[-.5,-.5,.5,.5],[0.,0.,.5,-.5],[-.5,-.5,1.,0.]])


def read(plan, path):
    return json.loads(bound(plan,path))


def artifact(plan, validation, relative):
    path = Path(validation).parent/relative
    if plan['files_sha256'][path.as_posix()]!=read(plan,validation)['artifacts_sha256'][relative]:
        raise ValueError(f'accepted artifact changed: {path}')
    return read(plan,path)


def prior(initial, states):
    return np.concatenate((initial[:,None],states[:,:-1]),axis=1)


def summarize(source_matrix):
    return {'by_source_and_evaluation_lead_m':source_matrix.tolist(),
            'total_by_evaluation_lead_m':source_matrix.sum(axis=0).tolist(),
            'by_source_window_at_h32_m':{f'{start}-{end}':float(source_matrix[start-1:end,-1].sum())
                                        for start,end in WINDOWS}}


def physical_onset(y, floor):
    """Persistent dominance of a common outward shift over the signed response."""
    nominal,common,odd = (WEIGHTS@y)[:3]
    same_side = np.all(np.sign(y)==np.sign(nominal),axis=0) & (nominal!=0.)
    dominates = same_side & (np.sign(nominal)*common > np.abs(odd))
    improvement = np.abs(nominal)-np.abs(y[2])
    beyond_margin = dominates & (-improvement>floor)

    def terminal_start(mask):
        suffix = np.logical_and.accumulate(mask[::-1])[::-1]
        leads = np.flatnonzero(suffix)+1
        return int(leads[0]) if len(leads) else None

    return {'outward_common_dominant_leads':(np.flatnonzero(dominates)+1).tolist(),
            'terminal_common_dominant_start':terminal_start(dominates),
            'terminal_common_dominant_loss_above_margin_start':terminal_start(beyond_margin),
            'suggestion_improvement_by_lead_m':improvement.tolist()}


def decompose(model, initial, features, predicted, truth):
    dt = model.dt_s
    measured_prior,rolled_prior = (prior(initial,path) for path in (truth,predicted))
    transform,world,shift = tool_transition(features)
    reference,velocity_check = required_residual(measured_prior,truth,transform,world,shift)
    independent = world_head(model,features,measured_prior)
    free = world_head(model,features,rolled_prior)
    position_defect = measured_prior[...,:3]+.5*dt*(measured_prior[...,3:6]+truth[...,3:6])-truth[...,:3]
    sources = {
        'independent_velocity_error':propagate(independent-reference,transform,world,dt),
        'learned_feedback_error':propagate(free-independent,transform,world,dt),
        'position_integration_error':propagate_discrete_position(position_defect,transform,dt),
    }
    closure = require_close(sum(sources.values()).sum(axis=0),predicted[...,:6]-truth[...,:6])
    # This identity independently checks that the local head and the Newton
    # reference produce the literal single-step prediction error.
    actual_local = model.denormalize(model.step(model.normalize(measured_prior),features))
    local_delta = (world@(independent-reference)[...,None])[...,0]
    local_check = require_close(local_delta,actual_local[...,3:6]-truth[...,3:6])
    local_position_check = require_close(.5*dt*local_delta+position_defect,actual_local[...,:3]-truth[...,:3])
    lateral = [{key:summarize(np.einsum('b,sbt->st',weights,arr[...,1]))
                for key,arr in sources.items()} for weights in WEIGHTS]
    checks = {'required_velocity_closure_m_s':velocity_check,'trajectory_closure_max_si':closure,
              'local_velocity_closure_m_s':local_check,'local_position_closure_m':local_position_check}
    return dict(zip(TRACKS,lateral,strict=True)),checks


def case_report(model, normalization, bundle, traces, archived_score):
    cfg,spec,native,commands,recorded,prediction = bundle
    if (score_case(spec,native,commands,traces,prediction)!=archived_score
            or not nominal_gate([traces[k] for k in LABELS[:2]],recorded,spec['scoring']['summary_leads'])['passed']):
        raise ValueError('archived physical score or nominal baseline did not reproduce')
    state = _parse_state(native['snapshot']['state'])
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'),cfg['initial']['anchor_xy_tool_m'])[0]
    requests = np.asarray([commands[name]['requested'] for name in LABELS])
    initial = np.broadcast_to(_ball_state(state),(len(LABELS),9))
    # Build and run free predictions before loading measured states for the
    # independent one-step and error-source calculations.
    known = known_rollout(state.time_s,state.tilt_rad,state.tilt_rate_rad_s,requests,cfg,model.dt_s)
    features = model.features(known,cfg,anchor,*normalization)
    predicted = model.denormalize(model.rollout(model.normalize(initial),features))
    expected = [prediction['predictions'][name]['h32_y_tool_m'] for name in LABELS]
    checks = {'frozen_model_h32_closure_m':require_close(predicted[:,-1,1],expected)}
    np.testing.assert_array_equal(predicted[0],predicted[1])
    truth = np.stack([_full_branch(traces[name],spec['case']['tick'])['raw'] for name in LABELS])
    attribution,closure = decompose(model,initial,features,predicted,truth)
    y0 = prediction['target_y_tool_m']
    y_physics,y_model = (path[...,1]-y0 for path in (truth,predicted))
    actual,pred = WEIGHTS@y_physics,WEIGHTS@y_model
    checks.update(closure)
    for index,track in enumerate(TRACKS):
        terms = attribution[track]
        checks[f'{track}_closure_m'] = require_close(sum(np.asarray(v['total_by_evaluation_lead_m'])
                                                        for v in terms.values()),pred[index]-actual[index])
    return {
        'case':spec['case'],
        'frozen_physical_score_passed':archived_score['passed'],
        'physics_timing':physical_onset(y_physics,archived_score['margin_m']),
        'trajectory_y_by_branch':{label:{'newton_m':y_physics[i].tolist(),'model_free_m':y_model[i].tolist()}
                                 for i,label in enumerate(LABELS)},
        'tracks':{track:{'newton_m':actual[i].tolist(),'model_free_m':pred[i].tolist(),
                         'error_sources':attribution[track]} for i,track in enumerate(TRACKS)},
        'closure':checks,
    }


def run(plan, out, guard):
    if plan['updates']!=0:
        raise ValueError('this is an offline frozen-model diagnosis')
    capture = plan['original_capture_plan']
    original = load_plan(ROOT/capture,plan['files_sha256'][capture])
    prior_path,finished_path = plan['capture_validations']
    first,second = (read(plan,p) for p in (prior_path,finished_path))
    continuation = artifact(plan,finished_path,'execution_plan.json')
    if (first['status']!='stopped_before_second_worker' or first['plan_sha256']!=plan['files_sha256'][capture]
            or first['second_worker_started'] or second['status']!='completed' or not second['nominal_gate_passed']
            or continuation['plan']['frozen_plan']!=capture or continuation['plan']['partial_validation']!=prior_path
            or continuation['plan']['partial_validation_sha256']!=plan['files_sha256'][prior_path]
            or second['combined_physics_steps']!=4096 or second['cases_scored']!=2
            or any(v['physics_steps']!=2048 or v['training_updates'] or v['test_episode_files_read']
                   or v['control_accepted'] for v in (first,second))):
        raise ValueError('two-case physical experiment does not match the frozen scope')
    comparison = artifact(plan,finished_path,'comparison.json')
    checkpoint = Path(plan['model_validation']).with_name('world_increment_900.npz')
    valid = read(plan,plan['model_validation'])
    if (valid['status'],valid['updates'],valid['training_pairs'],valid['test_episode_files_read'])!=('completed',900,61,False):
        raise ValueError('expected the frozen 61-pair checkpoint')
    if plan['files_sha256'][checkpoint.as_posix()]!=valid['artifacts_sha256'][checkpoint.name]:
        raise ValueError('61-pair checkpoint provenance changed')
    with np.load(ROOT/checkpoint,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['known_names'],KNOWN_NAMES)
        if saved['updates'].item()!=900:
            raise ValueError('unexpected checkpoint update')
        model = WorldIncrementMLP.from_state_dict(saved)
        normalization = saved['known_mean'],saved['known_scale']

    cases = {}
    for index,(name,_,_) in enumerate(IDENTITIES):
        validation = (prior_path,finished_path)[index]
        base = f'{name}/worker' if index==0 else 'worker'
        traces = {k:artifact(plan,validation,f'{base}/{k}.json') for k in LABELS}
        cases[name] = case_report(model,normalization,load_case(original,index),traces,comparison['cases'][name])
        guard(name,log=True)
    if sum(c['frozen_physical_score_passed'] for c in cases.values())!=second['cases_passed']:
        raise ValueError('aggregate frozen pass count changed')
    write_json(out/'feedback_decomposition.json',{'source_windows':WINDOWS,'cases':cases,'dt_s':model.dt_s,
        'scope':'Free trajectories use their own predicted states. Error terms are propagated with fixed analytic tool maps. Measured pre-state is used only for the independent local diagnostic.'})
    return {'development_cases':len(cases),'traces':len(cases)*len(LABELS),
            'archived_physics_scores_reproduced':True,'max_trajectory_error_closure_si':
            max(row['closure']['trajectory_closure_max_si'] for row in cases.values())}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
