"""Fit the frozen spin-window comparison and score only the two fixed endpoints."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from s3_fitability_common import ROOT, SCOPE, run_phase, write_json
import numpy as np

from audit_s3_small_action_response import read
from ball_retention import velocity_response_loss
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import KNOWN_NAMES, prepare_dataset
from ball_retention.pair_collection import prepare_collection
from ball_retention.timed_pair_data import prepare_timed_pairs
from ball_retention.world_increment_model import PACKED, WorldIncrementMLP
from ball_retention.world_model import Adam
from deterministic_response_metrics import response_metrics
from preflight_s3_spin_supervision_window import SPIN, response_weights, spin_response_loss
from preflight_s3_velocity_response_leads import dormant_check, lead_vectors
from preflight_s3_world_increment_supervision import add, dot, mul, norm
from train_s3_local_velocity_increment import REFERENCE as LEGACY_49_VALIDATION
from train_s3_world_increment import LegacyEvaluation, objectives, reference_data, score_candidate
from train_s3_world_increment_data61 import combine, extend_fold, probe, supervision
from train_s3_position_integral import CONTROL, REFERENCE, checks_for, primary, regression, report


PHASE = 'spin_supervision_window_fit'
COMPARISON = 'configs/s3_spin_supervision_window_comparison_protocol.json'
SMALL_CAPTURE = 'configs/s3_small_action_response_capture_plan.json'
ENDPOINT = 'world_increment_900.npz'


def accepted_artifact(plan, validation, status, name):
    path = Path(validation).parent/name
    if plan['files_sha256'][path.as_posix()] != status['artifacts_sha256'][name]:
        raise ValueError(f'accepted artifact has changed: {path}')
    return path


def check_start(model, original, candidate, spin_args, weights, stream, expected, out):
    old, old_grad = probe(model, original, *stream[0], expected['seed'])
    new, new_grad = probe(model, candidate, *stream[0], expected['seed'])
    before, after, valid = weights
    mass = float(np.sum(valid*before))
    a, ga = spin_response_loss(*spin_args, before, valid, mass)
    b, gb = spin_response_loss(*spin_args, after, valid, mass)
    delta = add(new_grad, mul(old_grad, -1.))
    record = {'original_probe': old, 'candidate_probe': new, 'spin_loss_change': b-a,
              'delta_gradient_norm': norm(delta), 'delta_gradient_fraction_of_old_total': norm(delta)/norm(old_grad),
              'old_vs_new_total_gradient_cosine': dot(old_grad,new_grad)/(norm(old_grad)*norm(new_grad))}
    write_json(out/'opening_gradient.json', record)
    frozen = expected['opening']
    if old['pair'] != frozen['original_probe']['pair'] or old['pair'] != expected['old_fit']['initial_pair_loss']:
        raise ValueError('original opening objective did not reproduce')
    if old['s2_loss'] != new['s2_loss'] or old['s2_loss'] != expected['old_fit']['initial_s2_loss']:
        raise ValueError('opening S2 loss did not reproduce')
    if any(old['pair'][k] != new['pair'][k] for k in ('absolute', 'auxiliary')):
        raise ValueError('candidate changed an absolute or auxiliary opening loss')
    for name in ('original_probe', 'candidate_probe'):
        np.testing.assert_allclose(record[name]['gradient_norm'], frozen[name]['gradient_norm'], rtol=0., atol=2e-13)
    np.testing.assert_allclose(new['pair']['loss'], frozen['candidate_probe']['pair']['loss'], rtol=0., atol=2e-14)
    np.testing.assert_allclose(new['pair']['loss']-old['pair']['loss'], b-a, rtol=0., atol=2e-14)
    for key, value in add(gb, mul(ga, -1.)).items():
        np.testing.assert_allclose(delta[key], value, rtol=0., atol=2e-14)
    for key in ('delta_gradient_norm', 'delta_gradient_fraction_of_old_total', 'old_vs_new_total_gradient_cosine'):
        np.testing.assert_allclose(record[key], frozen[key], rtol=0., atol=2e-13)


def small_inputs(plan, held):
    launch = read(plan, SMALL_CAPTURE)
    validation = (Path(launch['output'])/'validation.json').as_posix()
    accepted = read(plan, validation)
    static_path = launch['preflight_validation']
    static = read(plan, static_path)
    if (accepted['plan_sha256'] != plan['files_sha256'][SMALL_CAPTURE]
            or accepted['status'] != 'completed' or not accepted['nominal_gate_passed']
            or accepted['physics_steps'] != 7168 or len(accepted['accepted_prefixes']) != 4
            or accepted['test_episode_files_read'] or accepted['control_accepted']
            or static['status'] != 'completed'
            or static['plan_sha256'] != launch['files_sha256'][launch['preflight_plan']]):
        raise ValueError('small-action archive not accepted')
    spec, commands = (read(plan, accepted_artifact(plan, static_path, static, name))
                      for name in ('replay_spec.json', 'branch_commands.json'))
    capture, cfg, state, full_traces, _ = held
    if (spec['case'] != capture['case'] or accepted['case'] != capture['case']
            or spec['split'] != 'validation' or accepted['branches'] != spec['branch_order']
            or len(spec['cases']) != 6 or spec['horizon'] != 32):
        raise ValueError('small-action scope changed')
    traces = {}
    for name in accepted['branches']:
        trace = read(plan, accepted_artifact(plan, validation, accepted, f'worker/{name}.json'))
        if (trace['actions'] != commands[name]['requested'] or trace['commands'] != commands[name]['executed']):
            raise ValueError(f'small-action commands changed: {name}')
        traces[name] = {'raw': np.asarray(trace['states_raw']), 'actions': trace['actions']}
    for name in ('nominal_1', 'nominal_2'):
        for field in ('raw', 'actions'):
            np.testing.assert_array_equal(traces[name][field], full_traces[name][field])
    scores = read(plan, accepted_artifact(plan, validation, accepted, 'same_amplitude_scores.json'))
    if scores['case'] != capture['case']:
        raise ValueError('small-action scores refer to a different snapshot')
    return spec, traces, scores['models']['61_pairs']


def score_small(model, held, fold, anchor, small, guard):
    spec, traces, _ = small
    capture, cfg, state, _, _ = held
    policy = spec['scoring']
    baseline = [traces[name]['raw'] for name in ('nominal_1', 'nominal_2')]
    scores = {}
    for name, case in spec['cases'].items():
        design = {**capture, 'amplitude': case['amplitude'], 'directions': {name: case['segments']}}
        _, rollout = score_candidate(model, (design, cfg, state, traces, None), fold, anchor, guard)
        pair = rollout['pairs'][name]
        plus, minus = (traces[f'{name}_{side}']['raw'] for side in ('plus', 'minus'))
        scores[name] = response_metrics(baseline, plus, minus, rollout['nominal'],
            pair['plus'], pair['minus'], pair['jvp'], case['amplitude'],
            leads=policy['diagnostic_leads'], floors=policy['absolute_floors_si'],
            **{key: policy[key] for key in ('noise_multiplier', 'strong_multiplier', 'min_cosine', 'max_relative_error')})
    return scores


def run(plan, out, guard):
    frozen = read(plan, COMPARISON)
    previous_plan_path = frozen['matched_reference']
    previous_plan = read(plan, previous_plan_path)
    verified_plan_path = 'configs/s3_spin_supervision_window_verified_plan.json'
    verified_plan = read(plan, verified_plan_path)
    for manifest in (frozen, previous_plan, verified_plan):
        if any(plan['files_sha256'][path] != digest for path,digest in manifest['files_sha256'].items()):
            raise ValueError('execution inputs differ from the frozen manifest')
    if (frozen['schema'] != 's3-spin-supervision-window-comparison-v1'
            or (plan['updates'], plan['seed'], plan['budget'], plan['scope']) !=
               (frozen['fixed_updates'], frozen['seed'], frozen['budget'], frozen['execution_scope'])
            or (plan['updates'],plan['seed']) != (900,20261111)
            or any(plan[key] != previous_plan[key] for key in ('updates','seed','budget','scope'))):
        raise ValueError('fit scope changed')
    valid_path = frozen['accepted_preflight']
    verified = read(plan, valid_path)
    if (verified['status'] != 'completed' or verified['updates'] != 0
            or verified['plan_sha256'] != plan['files_sha256'][verified_plan_path]
            or any(verified[key] != val for key,val in SCOPE.items())
            or not all(verified[key] for key in ('old_loss_and_sampler_reproduced',
                        'spin_weight_mass_preserved_per_pair', 'gradient_difference_and_fd_verified'))):
        raise ValueError('spin window preflight not accepted')
    gradient = read(plan, accepted_artifact(plan, valid_path, verified, 'spin_gradient_preflight.json'))
    old_path = verified_plan['fit_validation']
    old = read(plan, old_path)
    if (old['status'] != 'completed' or old['updates'] != 900 or old['training_pairs'] != 61
            or old['plan_sha256'] != plan['files_sha256'][previous_plan_path]
            or any(old[key] != val for key,val in SCOPE.items())):
        raise ValueError('61-pair reference fit not accepted')
    old_pre = read(plan, accepted_artifact(plan, old_path, old, 'data_preflight.json'))
    old_fit = read(plan, accepted_artifact(plan, old_path, old, 'fit_preflight.json'))
    old_score = read(plan, accepted_artifact(plan, old_path, old, 'comparison.json'))
    ck = accepted_artifact(plan, old_path, old, ENDPOINT)

    protocol, cfg, scales, base, held, draws = reference_data(previous_plan, guard)
    timed = [prepare_timed_pairs(ROOT, path, previous_plan['files_sha256'][path])
             for path in previous_plan['new_timed_manifests']]
    fold, groups = extend_fold(base, timed)
    data, labels = supervision(previous_plan, fold, previous_plan['timed_manifests'], guard)
    parent = read(plan, previous_plan['matched_fit_plan'])
    coefficients = read(plan, parent['window_protocol'])['auxiliary']['coefficients']
    np.testing.assert_array_equal(fold.pair_weights, old_pre['expanded_weights'])
    if (len(fold.pairs.initial) != frozen['training_pairs'] or len(fold.pairs.initial) != 61
            or groups != old_pre['group_weights'] or labels != old_pre['label_inventory']
            or coefficients != old_pre['coefficients']):
        raise ValueError('61-pair weights or auxiliary labels changed')
    with np.load(ROOT/ck, allow_pickle=False) as saved:
        endpoint = WorldIncrementMLP.from_state_dict(saved)
        if saved['updates'].item() != plan['updates']:
            raise ValueError('reference checkpoint is not update 900')
        for key in ('state_mean','state_scale','known_mean','known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold,key))
        np.testing.assert_array_equal(saved['known_names'], KNOWN_NAMES)
    np.testing.assert_array_equal(endpoint.inc_center, protocol['world_increment_center_m_s'])
    model = WorldIncrementMLP(PACKED, fold.state_mean, fold.state_scale, endpoint.dt_s, plan['seed'],
                              inc_center=endpoint.inc_center, hidden=endpoint.hidden, residual_gain=endpoint.residual_gain)
    horizons = cfg['anchors']['horizons']
    rng = np.random.default_rng(plan['seed'])
    stream = [(h, rng.integers(len(fold.anchors[h].initial), size=cfg['anchors']['batch_size']))
              for h in (horizons[i%len(horizons)] for i in range(plan['updates']))]
    digest = sha256(b''.join(np.r_[h,take].astype('<i8').tobytes() for h,take in stream)).hexdigest()
    if any(digest != x for x in (draws, frozen['sampler_sha256'], old['sampler_sha256'],
                                old_pre['sampler_sha256'], gradient['sampler_sha256'])):
        raise ValueError('S2 draw stream changed')
    model_cfg = held[1]
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), model_cfg['initial']['anchor_xy_tool_m'])[0]
    leads = lead_vectors(read(plan, read(plan, previous_plan['coverage_design'])['reference_design']), 32)
    weights = response_weights(fold, cfg, leads)
    before, after, valid = weights
    for matrix, prefix, mass_key in ((before,'old_', 'before'), (after,'', 'after')):
        np.testing.assert_array_equal(matrix[:,SPIN], frozen[prefix+'response_spin_weights_by_lead'])
        np.testing.assert_allclose((valid*matrix).sum(axis=(0,1))[SPIN],
                                   frozen['response_mass_per_axis_'+mass_key], rtol=0., atol=2e-15)
    evaluate, initial, features = objectives(model, fold, cfg, scales, leads, model_cfg, anchor)
    original = combine(model, evaluate, data, coefficients)

    def candidate(h, take):
        # Substitute only the response matrix. Keep the frozen absolute-loss
        # call, response normalization, auxiliary terms and S2 path intact.
        with patch.object(velocity_response_loss, 'response_components', new=lambda *args: after):
            return original(h, take)

    guard('opening_gradient', 0)
    check_start(model, original, candidate, (model,fold,cfg,scales,initial,features), weights, stream,
                {'seed':plan['seed'],'opening':gradient['checkpoints']['initialization'],'old_fit':old_fit}, out)
    small = small_inputs(plan, held)
    full_old, _ = score_candidate(endpoint, held, fold, anchor, guard)
    small_old = score_small(endpoint, held, fold, anchor, small, guard)
    if full_old != old_score['held_timing'] or small_old != small[2]:
        raise ValueError('61-pair development endpoint did not reproduce')
    adam = cfg['optimizer']
    if adam['name'] != 'Adam':
        raise ValueError('optimizer changed')
    initial_sha = sha256(b''.join(p.astype('<f8').tobytes() for p in model.params.values())).hexdigest()
    write_json(out/'fit_preflight.json', {'sampler_sha256':digest, 'training_pairs':61, 'optimizer':adam,
        'initial_parameter_sha256':initial_sha, 'old_spin_weights_by_lead':before[:,SPIN].tolist(),
        'candidate_spin_weights_by_lead':after[:,SPIN].tolist(), 'per_pair_axis_weight_preserved':True,
        'old_full_and_small_scores_reproduced':True, 'opening_gradient_verified':True})
    optimizer = Adam(model.params, learning_rate=adam['learning_rate'], max_grad_norm=adam['max_grad_norm'])
    guard('fit', 0, log=True)
    with (out/'training.jsonl').open('x', encoding='utf-8') as log:
        for update,(h,take) in enumerate(stream, 1):
            s2,pair,grad = candidate(h,take)
            if not np.isfinite(s2+pair['loss']):
                raise FloatingPointError(f'nonfinite objective before update {update}')
            size = optimizer.step(grad)
            guard('fit', update, log=update%30 == 0)
            if not all(np.isfinite(value).all() for value in model.params.values()):
                raise FloatingPointError(f'nonfinite parameters after update {update}')
            if update%30 == 0:
                print(json.dumps({'update':update,'anchor_loss':s2,'pair':pair,'gradient_norm':size},
                                 allow_nan=False), file=log, flush=True)
    dormant_check(model.params)
    np.savez_compressed(out/ENDPOINT, **model.state_dict(), known_names=KNOWN_NAMES,
                        known_mean=fold.known_mean, known_scale=fold.known_scale, updates=optimizer.iteration)
    with np.load(out/ENDPOINT, allow_pickle=False) as saved:
        fitted = WorldIncrementMLP.from_state_dict(saved)
    np.testing.assert_array_equal(fitted.rollout(initial,features[0]), model.rollout(initial,features[0]))
    full_new, forecasts = score_candidate(fitted, held, fold, anchor, guard)
    small_new = score_small(fitted, held, fold, anchor, small, guard)
    legacy_plan = read(plan, previous_plan['held_design'])
    legacy = prepare_collection(ROOT, ROOT/legacy_plan['base_collection'])
    anchors = prepare_dataset((ROOT/legacy_plan['s2_manifest']).parent,horizons=tuple(horizons),verify_test=False)
    histories = {row['episode_id']:row['history_id'] for row in read(plan, cfg['history_plan'])['episodes']}
    guard('legacy_protections')
    legacy_scores = report(LegacyEvaluation(fitted,model_cfg,anchor,fold),fold.known_mean,fold.known_scale,
                           legacy,anchors,histories)
    legacy_scores['pairs'] = [row for row in legacy_scores['pairs'] if row['split']=='validation']
    original_scores = read(plan, previous_plan['original_scores'])
    references = {'40_pairs':original_scores['legacy_development'][CONTROL],
                  '46_pairs':read(plan, previous_plan['reference_46_scores'])['legacy_validation'],
                  '49_pairs':read(plan, LEGACY_49_VALIDATION.with_name('comparison.json'))['legacy_validation'],
                  '49_pairs_full_window':read(plan, Path(parent['output'])/'comparison.json')['legacy_validation'],
                  '61_pairs':old_score['legacy_validation']}
    protections = {key:regression(value,legacy_scores) for key,value in references.items()}
    if any(len(value['checks']) != 22 for value in protections.values()):
        raise ValueError('expected 22 legacy checks for each reference')
    nominal = primary(full_new)['pitch_pulse']['absolute_model_nominal_error_norm']
    gates = checks_for(primary(full_new), nominal/original_scores['nominal_h32_position_m'][REFERENCE],
                       {CONTROL:protections['40_pairs']})
    gates.update({f'protections_vs_{key}':value['passed'] for key,value in protections.items() if key != '40_pairs'})
    write_json(out/'comparison.json', {'full':{'61_pairs':full_old,'spin_window':full_new},
        'small':{'61_pairs':small_old,'spin_window':small_new}, 'legacy_protections':protections,
        'legacy_validation':legacy_scores, 'nominal_h32_position_m':{'61_pairs':old_score['nominal_h32_position_m'],
        'spin_window':nominal}, 'gates':gates})
    write_json(out/'timed_rollouts.json', forecasts)
    return {'training_pairs':61,'sampler_sha256':digest,'opening_gradient_verified':True,
            'replacement_gate_passed':all(gates.values()),'main_model_replaced':False,'checkpoint_selection':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
