"""Replay the 61-pair recipe with 36 accepted Newton pairs and score update 900."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from s3_fitability_common import BUDGET, ROOT, SCOPE, run_phase, write_json
import numpy as np

import collect_s3_gradient_action as actions
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import KNOWN_NAMES, _ball_state, prepare_dataset
from ball_retention.pair_collection import prepare_collection
from ball_retention.timed_pair_data import concatenate_batches, prepare_timed_pairs
from ball_retention.world_increment_model import PACKED, WorldIncrementMLP, world_action_jvp
from ball_retention.world_model import Adam
from check_s3_actions import digest
from deterministic_response_metrics import response_metrics
from preflight_s3_velocity_response_leads import dormant_check, lead_vectors
from preflight_s3_world_increment import predict, reference_data
from preflight_s3_world_increment_supervision import SETTINGS, add, dot, mul, norm, raw_example
from train_s3_local_velocity_increment import REFERENCE as LEGACY_49_VALIDATION, bound
from train_s3_position_integral import CONTROL, REFERENCE, checks_for, primary, regression, report
from train_s3_spin_supervision_window import accepted_artifact, score_small, small_inputs
from train_s3_world_increment import LegacyEvaluation, objectives, score_candidate
from train_s3_world_increment_data61 import combine, extend_fold, probe, supervision


PHASE = 'world_increment_data97_fit'
CAPTURE = Path('runs/s3_broad_training_20261118')
FIT = Path('runs/s3_world_increment_data97_fit_20261118')
PLAN = CAPTURE/'fit_plan.json'
PREVIOUS = Path('configs/s3_world_increment_data61_fit_plan.json')
PREVIOUS_RUN = Path('runs/s3_world_increment_data61_fit_20261117')
PREVIOUS_SHA = 'bc252bfab4715fbf5ab21c7a096aad42c853a217de693d689507e3b878520ae3'
DRAW_SHA = 'e06e22a6d9979c90b66fc9da48cacefbab4f23ed45892c0cd0e003fb09945f5c'
INIT_SHA = 'd410b77e57aaa2489463a6ba8699a6b7b85515890aeecd77121ba180323b4a68'
SPIN_PLAN = Path('configs/s3_spin_supervision_window_fit_plan.json')
ACTION_PLAN = Path('configs/s3_gradient_action_capture_plan.json')
ACTION_RUNS = [(Path('runs/s3_gradient_action_capture_20261118'), 'partial_validation.json', 'validation_lateral/worker'),
               (Path('runs/s3_gradient_action_second_case_20261118_verified'), 'validation.json', 'worker')]
ENDPOINT = 'world_increment_900.npz'


def read(plan, path):
    return json.loads(bound(plan, path))


def accepted(plan, folder, validation, name):
    path = folder/validation
    return read(plan, accepted_artifact(plan, path, read(plan, path), name))


def latest_capture(plan):
    validation = read(plan, CAPTURE/'validation.json')
    if ((validation['status'], validation['physics_steps'], validation['pairs'])
            != ('completed', 54528, 36) or validation['test_episode_files_read']
            or validation['control_accepted'] or validation['training_updates'] != 0
            or validation['preflight_sha256'] != plan['files_sha256'][(CAPTURE/'preflight.json').as_posix()]):
        raise ValueError('all six new captures must be accepted before fitting')
    if 'resume_plan_sha256' in validation:
        path = CAPTURE/'resume_plan.json'
        resume = read(plan,path)
        if (validation['resume_plan_sha256'] != plan['files_sha256'][path.as_posix()]
                or (validation['worker_starts'], validation['previously_recorded_physics_steps'],
                    validation['new_physics_steps'], validation['unrecorded_physics_steps_upper_bound'],
                    validation['approved_physics_steps_upper_bound']) != (7,4736,49792,512,55040)
                or resume['max_physical_substeps'] != 55040):
            raise ValueError('continued capture exceeded or changed its approved budget')
    elif validation['worker_starts'] != 6:
        raise ValueError('unexpected number of physical workers')
    manifest = accepted_artifact(plan, CAPTURE/'validation.json', validation, 'capture_manifest.json')
    return validation, manifest


def reference_files():
    previous = json.loads((ROOT/PREVIOUS).read_bytes())
    if digest(ROOT/PREVIOUS) != PREVIOUS_SHA:
        raise ValueError('original 61-pair fit plan changed')
    source_hashes = previous['files_sha256'].copy()

    def bind(path, expected=None):
        path = Path(path)
        key = path.as_posix()
        actual = digest(ROOT/path)
        if actual != source_hashes.get(key, actual) or (expected is not None and actual != expected):
            raise ValueError(f'frozen file changed: {path}')
        source_hashes[key] = actual

    def bind_set(files):
        for path, expected in files.items():
            bind(path, expected)

    bind(PREVIOUS, PREVIOUS_SHA)
    old_status_path = PREVIOUS_RUN/'validation.json'
    bind(old_status_path)
    old = json.loads((ROOT/old_status_path).read_bytes())
    if old['plan_sha256'] != PREVIOUS_SHA:
        raise ValueError('original validation refers to another fit plan')
    for file in (ENDPOINT, 'comparison.json', 'data_preflight.json', 'fit_preflight.json'):
        bind(PREVIOUS_RUN/file, old['artifacts_sha256'][file])
    spin = json.loads((ROOT/SPIN_PLAN).read_bytes())
    bind(SPIN_PLAN)
    bind_set(spin['files_sha256'])
    bind(ACTION_PLAN)
    bind_set(json.loads((ROOT/ACTION_PLAN).read_bytes())['files_sha256'])
    for folder, name, worker in ACTION_RUNS:
        bind(folder/name)
        status = json.loads((ROOT/folder/name).read_bytes())
        for label in actions.LABELS:
            name = f'{worker}/{label}.json'
            bind(folder/name, status['artifacts_sha256'][name])
    bind(Path(__file__).resolve().relative_to(ROOT))
    return previous, source_hashes


def freeze():
    previous, source_hashes = reference_files()

    def bind(path, expected=None):
        key = Path(path).as_posix()
        actual = digest(ROOT/path)
        if expected is not None and actual != expected:
            raise ValueError(f'captured file changed: {path}')
        source_hashes[key] = actual

    for name in ('validation.json', 'preflight.json', 'capture_manifest.json'):
        bind(CAPTURE/name)
    if 'resume_plan_sha256' in json.loads((ROOT/CAPTURE/'validation.json').read_bytes()):
        bind(CAPTURE/'resume_plan.json')
    _, manifest = latest_capture({'files_sha256': source_hashes})
    for entry in json.loads((ROOT/manifest).read_bytes())['entries']:
        bind(entry['source_config'], entry['source_config_sha256'])
        refs = [entry['native_snapshot'], entry['worker_result'], entry['prefix_record'], entry['signal_report'],
                *entry['nominal_branches'], *(ref for pair in entry['pairs'] for ref in pair['branches'])]
        for ref in refs:
            bind(CAPTURE/ref['path'], ref['sha256'])
    plan = {**previous, 'phase': PHASE, 'output': FIT.as_posix(), 'schema': 's3-world-increment-data97-fit-v1',
            'files_sha256': source_hashes, 'added_training_pairs': 36, 'new_initial_groups': 6,
            'fixed_checkpoint': 900, 'sampler_sha256': DRAW_SHA, 'initial_parameter_sha256': INIT_SHA}
    with (ROOT/PLAN).open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write('\n')


def widen(old, timed):
    batch = timed.batch
    groups = set(batch.group_ids)
    old_groups = set(old.pairs.group_ids)
    if (len(batch.initial), len(old.pairs.initial), len(groups), len(old_groups)) != (36, 61, 6, 6) or groups & old_groups:
        raise ValueError('unexpected overlap or sample count')
    if any(np.count_nonzero(batch.group_ids == group) != 6 for group in groups):
        raise ValueError('each new group must have all six perturbation cells')
    weights = np.r_[old.pair_weights/2, np.full(len(batch.initial), 1/72)]
    pairs = concatenate_batches(old.pairs, batch)
    masses = {str(group): float(weights[pairs.group_ids == group].sum()) for group in sorted(groups | old_groups)}
    np.testing.assert_allclose(list(masses.values()), 1/12, rtol=0., atol=2e-15)
    return replace(old, pairs=pairs, pair_weights=weights, retained_indices=np.arange(len(weights))), masses


def new_auxiliary(plan, previous_data, timed, fold, guard):
    """Extend the original nominal and signed increment targets and reductions."""
    nominal, signed, valid, inventory = [], [], [], []
    for entry in timed.entries:
        cfg = read(plan, entry['source_config'])
        state = _parse_state(read(plan, CAPTURE/entry['native_snapshot']['path'])['state'])

        def example(ref):
            return raw_example(state, read(plan, CAPTURE/ref['path']), cfg)

        base = [example(ref) for ref in entry['nominal_branches']]
        nominal.extend(base)
        noise = .5*np.linalg.norm(base[0][2]-base[1][2], axis=-1)
        floor = np.maximum(SETTINGS['floor_m_s'], SETTINGS['noise_multiplier']*noise)
        for pair in entry['pairs']:
            a,b = (example(ref) for ref in sorted(pair['branches'], key=lambda ref: -ref['sign']))
            row = 61+len(signed)
            if (fold.pairs.episode_ids[row], fold.pairs.snapshot_ticks[row], fold.pairs.group_ids[row]) != (
                    entry['episode_id'], entry['snapshot_tick'], entry['group_id']):
                raise ValueError('new auxiliary row identity changed')
            np.testing.assert_array_equal(a[1], fold.pairs.plus_known[row])
            np.testing.assert_array_equal(b[1], fold.pairs.minus_known[row])
            magnitude = .5*np.linalg.norm(a[2]-b[2], axis=-1)
            eligible = magnitude >= floor
            signed.append((a,b))
            valid.append(eligible)
            inventory.append({'episode':entry['episode_id'], 'direction':pair['direction_id'],
                              'resolved_H1_H16':int(eligible[:16].sum()), 'resolved_H17_H32':int(eligible[16:].sum()),
                              'strong_H1_H32':int((magnitude >= 10*floor).sum())})
        guard('added_training_labels')
    if (len(nominal), len(signed)) != (12, 36):
        raise ValueError('new auxiliary label count changed')
    n0, n1, n2, _, nscale = previous_data['nominal']
    p0, p1, p2, old_weights, pscale = previous_data['half_difference']
    ns,nk,nt = (np.stack([row[k] for row in nominal]) for k in range(3))
    ps,pk,pt = (np.stack([np.stack([branch[k] for branch in pair]) for pair in signed], axis=1) for k in range(3))
    ns = np.concatenate((n0, ns))
    nk = np.concatenate((n1, (nk-fold.known_mean)/fold.known_scale))
    nt = np.concatenate((n2, nt))
    ps = np.concatenate((p0, ps), axis=1)
    pk = np.concatenate((p1, (pk-fold.known_mean)/fold.known_scale), axis=1)
    pt = np.concatenate((p2, pt), axis=1)
    raw = np.concatenate((old_weights > 0., np.asarray(valid)))*fold.pair_weights[40:, None]
    mass = np.array([raw[:, :16].sum(), raw[:, 16:].sum()])
    if not np.all(mass > 0.):
        raise ValueError('an auxiliary time window has no resolved responses')
    weights = raw*np.repeat(.5/mass, 16)
    np.testing.assert_allclose([weights[:, :16].sum(), weights[:, 16:].sum()], [.5, .5], rtol=0., atol=2e-15)
    return {'nominal': (ns,nk,nt,np.full(ns.shape[:-1],1/np.prod(ns.shape[:-1])),nscale),
            'half_difference': (ps,pk,pt,weights,pscale)}, inventory


def score_actions(plan, models, fold, guard):
    action_plan = read(plan, ACTION_PLAN)
    results = {}
    for i, (folder, validation, worker) in enumerate(ACTION_RUNS):
        cfg, spec, native, commands, _, frozen = actions.load_case(action_plan, i)
        traces = {name:accepted(plan, folder, validation, f'{worker}/{name}.json') for name in actions.LABELS}
        truth = actions.score_case(spec, native, commands, traces, frozen)
        state = _parse_state(native['snapshot']['state'])
        anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), cfg['initial']['anchor_xy_tool_m'])[0]
        nominal, plus, minus = (np.asarray(commands[k]['requested']) for k in ('nominal_1','gradient','reverse'))
        direction = (plus-minus)/.10
        scores = {}
        for label, model in models.items():
            def forward(request):
                return predict(model, state, request, cfg, anchor, fold.known_mean, fold.known_scale)

            base,jvp = world_action_jvp(model,_ball_state(state),state.time_s,state.tilt_rad,state.tilt_rate_rad_s,
                                       nominal,direction,cfg,anchor,fold.known_mean,fold.known_scale)
            np.testing.assert_allclose(jvp,(forward(nominal+1e-5*direction)-forward(nominal-1e-5*direction))/2e-5,
                                       atol=2e-6,rtol=2e-4)
            pos, neg = forward(plus), forward(minus)
            metric = response_metrics([traces[k]['states_raw'] for k in actions.LABELS[:2]],
                       traces['gradient']['states_raw'],traces['reverse']['states_raw'],base,pos,neg,jvp,.05,leads=(32,))
            absolute = {k:float(abs(path[-1,1]-truth['target_y_tool_m']))
                        for k,path in (('nominal',base),('gradient',pos),('reverse',neg))}
            scores[label] = {'h32_position':metric['by_lead']['32']['position'], 'h32_lateral_offset_m':absolute,
                             'rank':sorted(absolute,key=absolute.get)}
            guard('action_score')
        old_offsets = scores['61_pairs']['h32_lateral_offset_m']
        for label, value in truth['model_h32_abs_offset_m'].items():
            key = 'nominal' if label in ('nominal_1','nominal_2') else label
            np.testing.assert_allclose(old_offsets[key],value,rtol=0.,atol=1e-10)
        results[spec['case']['episode']] = {'physics':truth, 'models':scores}
    return results


def roll_spin(model, held, fold, anchor, small, guard):
    spec,traces,_ = small
    capture,cfg,state,_,_ = held
    label = 'roll_component'
    case = spec['cases'][label]
    design = {**capture, 'amplitude':case['amplitude'], 'directions':{label:case['segments']}}
    _, forecast = score_candidate(model,(design,cfg,state,traces,None),fold,anchor,guard)
    pair = forecast['pairs'][label]
    truth = (traces[f'{label}_plus']['raw']-traces[f'{label}_minus']['raw'])/2
    predicted = (np.asarray(pair['plus'])-np.asarray(pair['minus']))/2
    first, second = truth[8:16,6:9], predicted[8:16,6:9]
    norms = np.linalg.norm(first,axis=1)*np.linalg.norm(second,axis=1)
    cosines = [float(a/n) if n > 0. else None for a,n in zip(np.einsum('ij,ij->i',first,second),norms)]
    error = np.linalg.norm(second-first,axis=1)
    return {'leads':list(range(9,17)), 'newton_half_difference_rad_s':first.tolist(),
            'model_half_difference_rad_s':second.tolist(), 'vector_error_rad_s':error.tolist(),
            'direction_cosines':cosines, 'vector_error_rms_rad_s':float(np.sqrt(np.mean(error**2)))}


def run(plan, out, guard):
    if (plan['updates'],plan['budget'],plan['scope'],plan['seed'],plan['sampler_sha256'],plan['initial_parameter_sha256']) != (
            900,BUDGET,SCOPE,20261111,DRAW_SHA,INIT_SHA):
        raise ValueError('approved fit scope changed')
    previous = read(plan, PREVIOUS)
    old_valid = read(plan, PREVIOUS_RUN/'validation.json')
    if ((old_valid['status'],old_valid['updates'],old_valid['training_pairs'],old_valid['plan_sha256'])
            != ('completed',900,61,PREVIOUS_SHA) or any(old_valid[k] != v for k,v in SCOPE.items())):
        raise ValueError('original fit acceptance changed')
    old_pre = accepted(plan, PREVIOUS_RUN, 'validation.json', 'data_preflight.json')
    old_fit = accepted(plan, PREVIOUS_RUN, 'validation.json', 'fit_preflight.json')
    old_score = accepted(plan, PREVIOUS_RUN, 'validation.json', 'comparison.json')
    protocol,cfg,scales,old49,held,draws = reference_data(previous,guard)
    original_timed = [prepare_timed_pairs(ROOT,p,previous['files_sha256'][p]) for p in previous['new_timed_manifests']]
    old_fold, groups = extend_fold(old49, original_timed)
    old_aux, labels = supervision(previous, old_fold, previous['timed_manifests'], guard)
    np.testing.assert_array_equal(old_fold.pair_weights, old_pre['expanded_weights'])
    if groups != old_pre['group_weights'] or labels != old_pre['label_inventory']:
        raise ValueError('original 61-pair opening labels or weights changed')
    _, manifest = latest_capture(plan)
    timed = prepare_timed_pairs(ROOT, manifest, plan['files_sha256'][manifest.as_posix()])
    if timed.model_config != original_timed[0].model_config or timed.dt_s != original_timed[0].dt_s:
        raise ValueError('collection changed the inherited model inputs or clock')
    fold, masses = widen(old_fold, timed)
    aux, inventory = new_auxiliary(plan, old_aux, timed, fold, guard)
    inherited = read(plan, previous['matched_fit_plan'])
    coefs = read(plan, inherited['window_protocol'])['auxiliary']['coefficients']
    if coefs != old_pre['coefficients']:
        raise ValueError('auxiliary loss coefficients changed')

    checkpoint = accepted_artifact(plan,PREVIOUS_RUN/'validation.json',old_valid,ENDPOINT)
    with np.load(ROOT/checkpoint,allow_pickle=False) as saved:
        endpoint = WorldIncrementMLP.from_state_dict(saved)
        if saved['updates'].item() != 900:
            raise ValueError('original checkpoint is outside update 900')
        for key in ('state_mean','state_scale','known_mean','known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold,key))
        np.testing.assert_array_equal(saved['known_names'], KNOWN_NAMES)
    np.testing.assert_array_equal(endpoint.inc_center, protocol['world_increment_center_m_s'])
    model = WorldIncrementMLP(PACKED,fold.state_mean,fold.state_scale,endpoint.dt_s,plan['seed'],
                              inc_center=endpoint.inc_center,hidden=endpoint.hidden,residual_gain=endpoint.residual_gain)
    initial_sha = sha256(b''.join(p.astype('<f8').tobytes() for p in model.params.values())).hexdigest()
    horizons = cfg['anchors']['horizons']
    rng = np.random.default_rng(plan['seed'])
    stream = [(h,rng.integers(len(fold.anchors[h].initial),size=cfg['anchors']['batch_size']))
              for h in (horizons[i%len(horizons)] for i in range(900))]
    draw_sha = sha256(b''.join(np.r_[h,take].astype('<i8').tobytes() for h,take in stream)).hexdigest()
    if initial_sha != INIT_SHA or any(draw_sha != d for d in (DRAW_SHA,draws,old_fit['sampler_sha256'])):
        raise ValueError('initial parameters or S2 sample order changed')
    model_cfg = held[1]
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'),model_cfg['initial']['anchor_xy_tool_m'])[0]
    leads = lead_vectors(read(plan, read(plan, previous['coverage_design'])['reference_design']),32)
    old_eval,_,_ = objectives(model,old_fold,cfg,scales,leads,model_cfg,anchor)
    new_eval,initial,features = objectives(model,fold,cfg,scales,leads,model_cfg,anchor)
    original = combine(model, old_eval, old_aux, coefs)
    candidate = combine(model, new_eval, aux, coefs)
    guard('opening_gradient',0,log=True)
    a, g0 = probe(model, original, *stream[0], plan['seed'])
    b, g1 = probe(model, candidate, *stream[0], plan['seed'])
    # The new-data gradient has no archived value. Validate a second independent direction.
    b2, _ = probe(model, candidate, *stream[0], plan['seed']+1)
    finite = all(np.isfinite(x).all() for g in (g0,g1) for x in g.values())
    record = {'original':a, 'expanded':b, 'expanded_second_fd_error':b2['fd_error'],
              'total_gradient_cosine':dot(g0,g1)/(norm(g0)*norm(g1)),
              'delta_gradient_fraction':norm(add(g1,mul(g0,-1.)))/norm(g0),
              'sampler_sha256':draw_sha, 'initial_parameter_sha256':initial_sha,
              'training_pairs':97, 'group_weights':masses,'added_label_inventory':inventory,
              'optimizer':cfg['optimizer'], 'auxiliary_coefficients':coefs}
    write_json(out/'fit_preflight.json',record)
    previous_probe = old_pre['finite_differences']['initialization']['61_pairs']
    if (not finite or a['s2_loss'] != b['s2_loss'] or a['s2_loss'] != old_fit['initial_s2_loss']
            or a['pair'] != old_fit['initial_pair_loss'] or a['pair'] != previous_probe['pair']):
        raise ValueError('opening loss or gradient check failed')
    np.testing.assert_allclose(a['gradient_norm'],previous_probe['gradient_norm'],atol=2e-13,rtol=0.)
    adam = cfg['optimizer']
    if adam['name'] != 'Adam':
        raise ValueError('optimizer changed')
    optimizer = Adam(model.params,learning_rate=adam['learning_rate'],max_grad_norm=adam['max_grad_norm'])
    guard('fit',0,log=True)
    with (out/'training.jsonl').open('x',encoding='utf-8') as log:
        for update,(h,take) in enumerate(stream,1):
            s2,pair,grad = candidate(h,take)
            if not np.isfinite(s2+pair['loss']) or not all(np.isfinite(x).all() for x in grad.values()):
                raise FloatingPointError(f'nonfinite loss or gradient before update {update}')
            size = optimizer.step(grad)
            if not np.isfinite(size) or not all(np.isfinite(x).all() for x in model.params.values()):
                raise FloatingPointError(f'nonfinite optimizer state at update {update}')
            guard('fit',update,log=update%30==0)
            if update%30==0:
                print(json.dumps({'update':update,'s2_loss':s2,'pair':pair,'gradient_norm':size},allow_nan=False),
                      file=log,flush=True)
    dormant_check(model.params)
    np.savez_compressed(out/ENDPOINT,**model.state_dict(),known_names=KNOWN_NAMES,
                        known_mean=fold.known_mean,known_scale=fold.known_scale,updates=optimizer.iteration)
    with np.load(out/ENDPOINT,allow_pickle=False) as saved:
        fitted = WorldIncrementMLP.from_state_dict(saved)
    np.testing.assert_array_equal(fitted.rollout(initial,features[0]),model.rollout(initial,features[0]))

    full61,_ = score_candidate(endpoint,held,fold,anchor,guard)
    full97,forecasts = score_candidate(fitted,held,fold,anchor,guard)
    small = small_inputs(plan,held)
    small61 = score_small(endpoint,held,fold,anchor,small,guard)
    small97 = score_small(fitted,held,fold,anchor,small,guard)
    if full61 != old_score['held_timing'] or small61 != small[2]:
        raise ValueError('the frozen 61-pair development scores did not reproduce')
    action_scores = score_actions(plan,{'61_pairs':endpoint,'97_pairs':fitted},fold,guard)
    spin = {label:roll_spin(candidate,held,fold,anchor,small,guard)
            for label,candidate in (('61_pairs',endpoint),('97_pairs',fitted))}
    held_plan = read(plan, previous['held_design'])
    legacy = prepare_collection(ROOT,ROOT/held_plan['base_collection'])
    anchor_data = prepare_dataset((ROOT/held_plan['s2_manifest']).parent,horizons=tuple(horizons),verify_test=False)
    histories = {row['episode_id']:row['history_id'] for row in read(plan,cfg['history_plan'])['episodes']}
    legacy97 = report(LegacyEvaluation(fitted,model_cfg,anchor,fold),fold.known_mean,fold.known_scale,
                      legacy,anchor_data,histories)
    legacy97['pairs'] = [row for row in legacy97['pairs'] if row['split']=='validation']
    original_scores = read(plan, previous['original_scores'])
    references = {'40_pairs':original_scores['legacy_development'][CONTROL],
                  '46_pairs':read(plan,previous['reference_46_scores'])['legacy_validation'],
                  '49_pairs':read(plan,LEGACY_49_VALIDATION.with_name('comparison.json'))['legacy_validation'],
                  '49_pairs_full_window':read(plan,Path(inherited['output'])/'comparison.json')['legacy_validation'],
                  '61_pairs':old_score['legacy_validation']}
    protections = {label:regression(value,legacy97) for label,value in references.items()}
    if any(len(item['checks']) != 22 for item in protections.values()):
        raise ValueError('expected exactly 22 legacy protections for each reference')
    nominal = primary(full97)['pitch_pulse']['absolute_model_nominal_error_norm']
    gates = checks_for(primary(full97), nominal/original_scores['nominal_h32_position_m'][REFERENCE],
                       {CONTROL:protections['40_pairs']})
    gates.update({f'protections_vs_{k}':v['passed'] for k,v in protections.items() if k != '40_pairs'})
    write_json(out/'comparison.json',{'full':{'61_pairs':full61,'97_pairs':full97}, 'roll_spin_H9_H16':spin,
        'small':{'61_pairs':small61,'97_pairs':small97}, 'frozen_development_actions':action_scores,
        'nominal_h32_position_error_m':{'61_pairs':old_score['nominal_h32_position_m'],'97_pairs':nominal},
        'legacy_protections':protections,'legacy_validation':legacy97,'gates':gates})
    write_json(out/'timed_rollouts.json', forecasts)
    return {'training_pairs':97,'sampler_sha256':draw_sha,'opening_gradient_verified':True,
            'replacement_gate_passed':all(gates.values()),'main_model_replaced':False,'checkpoint_selection':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'fit'))
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.phase == 'freeze':
        freeze()
    elif args.sha256:
        run_phase(ROOT/PLAN, args.sha256, PHASE, run)
    else:
        parser.error('accepted fit plan digest is required')
