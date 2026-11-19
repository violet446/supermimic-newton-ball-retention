"""Audit late common-response coverage in the frozen training and development archives."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from audit_s3_ball_velocity_components import frame_series
from audit_s3_gradient_action_feedback import WEIGHTS, artifact, prior, read
from audit_s3_world_increment_drift import require_close
from audit_s3_world_increment_generation import propagate, world_head
from audit_s3_world_increment_newton_reference import required_residual, tool_transition
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import KNOWN_NAMES, _ball_state, known_rollout
from ball_retention.world_increment_model import WorldIncrementMLP
from collect_s3_gradient_action import IDENTITIES, LABELS, load_case, load_plan, score_case
from deterministic_archive_metrics import _full_branch
from preflight_s3_world_increment_supervision import SETTINGS
from response_contact_diagnosis import _allowed_pairs, _decode_trace


PHASE = 'gradient_common_coverage'
LEADS = np.arange(17,25)
FOCUS = (20,21,22)
COMMON = WEIGHTS[1]
TICK, SUBSTEPS, HORIZON = 168,16,32


def compare(estimate, target, estimate_floor):
    a,b = np.linalg.norm(estimate),np.linalg.norm(target)
    cosine = float(np.clip(estimate@target/(a*b),-1.,1.)) if a>=estimate_floor and b>0 else None
    return cosine, (float(a/b) if b>0 else None)


def analyze(model, normalizer, cfg, native, traces, mesh):
    """Evaluate measured and independent model increments in world axes."""
    raw = [traces[key] for key in LABELS]
    state = _parse_state(native['snapshot']['state'])
    anchor = support_point(mesh,cfg['initial']['anchor_xy_tool_m'])[0]
    dt = model.dt_s
    parts = [_full_branch(branch,TICK) for branch in raw]
    truth = np.stack([row['raw'] for row in parts])
    start = np.broadcast_to(_ball_state(state),(4,9))
    measured = prior(start,truth)
    actions = np.stack([row['actions'] for row in parts])
    if not (np.array_equal(actions[0],actions[1])
            and np.array_equal(actions[2:,16:],actions[:2,16:])):
        raise ValueError('nominal schedule or post-pulse branch requests changed')
    known = known_rollout(state.time_s,state.tilt_rad,state.tilt_rate_rad_s,actions,cfg,dt)
    features = model.features(known,cfg,anchor,*normalizer)
    transform,world,shift = tool_transition(features)
    reference,_ = required_residual(measured,truth,transform,world,shift)
    local = world_head(model,features,measured)
    velocity = np.stack([frame_series(state,row,dt)[3] for row in parts])
    # Gravity and the model's constant world increment center cancel under COMMON.
    increments = np.diff(velocity,axis=1)
    response,reference_response,predicted = (np.tensordot(COMMON,x,axes=1) for x in (increments,reference,local))
    local_sources = np.einsum('b,sbt->st',COMMON,propagate(local-reference,transform,world,dt)[...,1])
    noise = .5*np.linalg.norm(increments[0]-increments[1],axis=-1)
    floor = np.maximum(SETTINGS['floor_m_s'],SETTINGS['noise_multiplier']*noise)
    allowed = _allowed_pairs(native['layout'])
    decoded = [_decode_trace(b,label,HORIZON,SUBSTEPS,TICK*SUBSTEPS,cfg['physics']['dt_s'],allowed)
               for label,b in zip(LABELS,raw,strict=True)]
    counts = np.array([[count for count,_ in branch] for branch in decoded]).reshape(4,HORIZON,SUBSTEPS)
    shapes = [[pairs for _,pairs in branch] for branch in decoded]
    return {'measured':measured,'normalized_tool':features[...,:len(KNOWN_NAMES)],
            'physics':response,'reference':reference_response,'local':predicted,'floor':floor,
            'local_source_matrix':local_sources,'counts':counts,'shapes':shapes}


def compact(data):
    rows = {}
    for h in LEADS:
        i = h-1
        physical,target,model = (data[key][i] for key in ('physics','reference','local'))
        floor = data['floor'][i]
        strong = np.linalg.norm(physical)>=10*floor
        cosine,ratio = compare(model,physical,floor)
        rows[str(h)] = {'physics_world_m_s':physical.tolist(),'analytic_reference_world_m_s':target.tolist(),
            'model_world_m_s':model.tolist(),'strong_physics':bool(strong),'floor_m_s':float(floor),
            'local_error_norm_m_s':float(np.linalg.norm(model-physical)),
            'analytic_local_error_norm_m_s':float(np.linalg.norm(model-target)),
            'model_direction_cosine':cosine,'model_magnitude_ratio':ratio,
            'attributed_local_error_to_h32_y_m':float(data['local_source_matrix'][i,-1]),
            'measured_pre_states_tool_si':data['measured'][:,i].tolist(),
            'candidate_range_previous':[[int(a.min()),int(a.max())] for a in data['counts'][:,i-1]],
            'candidate_range_current':[[int(a.min()),int(a.max())] for a in data['counts'][:,i]],
            'common_count_contrast_current':(COMMON@data['counts'][:,i]).tolist(),
            'shape_sets_in_step':sorted(set(pairs for b in data['shapes'] for pairs in b[i*SUBSTEPS:(i+1)*SUBSTEPS]))}
    return rows


def candidate(training, development, row, i, j, scales):
    """Rank with pre-state only. Report label and contact agreement separately."""
    options = ((0,1,2,3),(0,1,3,2))
    match = min(options,key=lambda order:np.mean(((training['measured'][list(order),j]-development['measured'][:,i])/scales)**2))
    match = list(match)
    distance = float(np.sqrt(np.mean(((training['measured'][match,j]-development['measured'][:,i])/scales)**2)))
    tool_distance = float(np.sqrt(np.mean((training['normalized_tool'][match,j]-development['normalized_tool'][:,i])**2)))
    label,target = training['physics'][j],development['physics'][i]
    cosine,ratio = compare(label,target,training['floor'][j])
    compatible = (np.linalg.norm(label)>=10*training['floor'][j] and cosine is not None
                  and cosine>=.8 and .5<=ratio<=2.)
    return {**row,'lead':int(j+1),'pre_state_rms':distance,'tool_feature_rms':tool_distance,
            'previous_count_mae':float(np.mean(abs(training['counts'][match,j-1]-development['counts'][:,i-1]))),
            'current_common_count_contrast_mae':float(np.mean(abs(COMMON@(training['counts'][:,j]-development['counts'][:,i])))),
            'cosine':cosine,'amplitude_ratio':ratio,'label_compatible':bool(compatible)}


def coverage(data, training, scales):
    result = {}
    for h in LEADS:
        i = h-1
        if np.linalg.norm(data['physics'][i])<10*data['floor'][i]:
            result[str(h)] = {'strong':False}
            continue
        groups = {}
        for group,examples in training.items():
            same = [candidate(pack,data,identity,i,i,scales) for identity,pack in examples]
            compatible = [row for row in same if row['label_compatible']]
            shifted = [candidate(pack,data,identity,i,j,scales) for identity,pack in examples
                       for j in LEADS-1 if 0<abs(j-i)<=2]
            near = lambda row:row['pre_state_rms']
            groups[group] = {'nearest_same_step':min(same,key=near),
                'nearest_label_compatible_same_step':min(compatible,key=near,default=None),
                'compatible_same_step_count':len(compatible),
                'nearest_label_compatible_shifted':min((r for r in shifted if r['label_compatible']),key=near,default=None)}
        result[str(h)] = {'strong':True,'groups':groups}
    return result


def training_archives(plan,model,normalizer,mesh,dev_groups,guard):
    prepared,report = {},{}
    seen = set()
    for manifest in plan['training_manifests']:
        folder = Path(manifest).parent
        validation = folder/'validation.json'
        valid = read(plan,validation)
        if valid['status']!='completed' or valid['test_episode_files_read']:
            raise ValueError('training capture is not accepted')
        for entry in artifact(plan,validation,'capture_manifest.json')['entries']:
            episode,group = entry['episode_id'],entry['group_id']
            if (entry['split']!='train' or entry['snapshot_tick']!=TICK or group in dev_groups or episode in seen
                    or len(entry['pairs'])!=3):
                raise ValueError('training archive group, uniqueness or clock changed')
            seen.add(episode)
            def branch(ref):
                source = folder/ref['path']
                if plan['files_sha256'][source.as_posix()]!=ref['sha256']:
                    raise ValueError('training branch hash differs from its manifest')
                return read(plan,source)

            cfg = read(plan,entry['source_config'])
            snapshot = entry['native_snapshot']
            layout_path = (Path(snapshot['path']).parent/'native_layout.json').as_posix()
            native = {'snapshot':branch(snapshot),'layout':artifact(plan,validation,layout_path)}
            if cfg['seed']!=entry['seed'] or native['snapshot']['episode_id']!=episode:
                raise ValueError('training seed or saved episode identity changed')
            baselines = dict(zip(LABELS[:2],(branch(r) for r in entry['nominal_branches']),strict=True))
            if baselines[LABELS[0]]!=baselines[LABELS[1]]:
                raise ValueError('frozen training nominal repeats differ')
            for pair in entry['pairs']:
                plus,minus = sorted(pair['branches'],key=lambda r:-r['sign'])
                if (plus['sign'],minus['sign'],pair['amplitude'])!=(1,-1,.2):
                    raise ValueError('unexpected archive training action')
                traces = {**baselines,LABELS[2]:branch(plus),LABELS[3]:branch(minus)}
                identity = {'episode':episode,'action':pair['direction_id'],'amplitude':pair['amplitude']}
                values = analyze(model,normalizer,cfg,native,traces,mesh)
                prepared.setdefault(group,[]).append((identity,values))
                report.setdefault(group,[]).append({**identity,'by_lead':compact(values)})
            guard(episode,log=True)
    if len(seen)!=7 or len(prepared)!=5 or sum(map(len,prepared.values()))!=21:
        raise ValueError('expected seven complete training snapshots from five groups')
    return prepared,report


def run(plan,out,guard):
    if plan['updates']!=0 or tuple(plan['focus_leads'])!=FOCUS:
        raise ValueError('unexpected offline coverage scope')
    fit = read(plan,plan['model_plan'])
    if fit['timed_manifests']!=plan['training_manifests']:
        raise ValueError('complete training snapshot inventory differs from the 61-pair fit')
    model_valid = read(plan,plan['model_validation'])
    checkpoint = Path(plan['model_validation']).with_name('world_increment_900.npz')
    if (model_valid['status'],model_valid['updates'],model_valid['training_pairs'],model_valid['test_episode_files_read'])!=(
            'completed',900,61,False) or plan['files_sha256'][checkpoint.as_posix()]!=model_valid['artifacts_sha256'][checkpoint.name]:
        raise ValueError('61-pair model provenance changed')
    with np.load(ROOT/checkpoint,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['known_names'],KNOWN_NAMES)
        if saved['updates'].item()!=900:
            raise ValueError('unexpected checkpoint update')
        model = WorldIncrementMLP.from_state_dict(saved)
        normalizer = saved['known_mean'],saved['known_scale']
    capture = plan['original_capture_plan']
    original = load_plan(ROOT/capture,plan['files_sha256'][capture])
    first,second = plan['capture_validations']
    baseline = artifact(plan,second,'comparison.json')['cases']
    feedback = artifact(plan,plan['feedback_validation'],'feedback_decomposition.json')['cases']
    mesh = load_tool(ROOT/'assets/shovel.STL')
    scales = np.asarray(read(plan,plan['context_reference'])['scales']['ball'])
    if scales.shape!=(9,) or not np.all(np.isfinite(scales)&(scales>0.)):
        raise ValueError('invalid frozen context scales')
    training,rows = training_archives(plan,model,normalizer,mesh,{r[1] for r in IDENTITIES},guard)
    development = {}
    for index,(name,group,_) in enumerate(IDENTITIES):
        cfg,spec,native,commands,_,prediction = load_case(original,index)
        validation = (first,second)[index]
        root = f'{name}/worker' if index==0 else 'worker'
        traces = {k:artifact(plan,validation,f'{root}/{k}.json') for k in LABELS}
        if score_case(spec,native,commands,traces,prediction)!=baseline[name]:
            raise ValueError('archived development score changed')
        data = analyze(model,normalizer,cfg,native,traces,mesh)
        expected = feedback[name]['tracks']['common']['error_sources']['independent_velocity_error']
        error = require_close(data['local_source_matrix'],expected['by_source_and_evaluation_lead_m'])
        development[name] = {'group':group,'local_source_reproduction_m':error,'by_lead':compact(data),
                             'coverage_by_lead':coverage(data,training,scales)}
        guard(name,log=True)
    write_json(out/'common_coverage.json',{'leads':LEADS.tolist(),'focus_leads':FOCUS,'training':rows,
        'development':development,'label_match_rule':{'min_cosine':.8,'amplitude_ratio_range':[.5,2.],
        'min_signal_relative_to_floor':10,'shifted_comparison_max_leads':2},
        'context_note':'Nearest archived pre-state uses the frozen nine ball scales and allows swapping both signed branches. Contact counts describe candidates, not impulses. All recorded amplitudes are used unscaled.'})
    return {'training_snapshots':7,'training_groups':5,'training_pairs':21,'development_cases':2,
            'independent_source_reproduced':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
