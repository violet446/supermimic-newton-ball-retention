"""Preflight a mass-matched shift of paired spin supervision to earlier leads."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from audit_s3_small_action_response import read
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import known_rollout
from ball_retention.pair_loss import paired_loss
from ball_retention.timed_pair_data import prepare_timed_pairs
from ball_retention.velocity_response_loss import response_components
from ball_retention.world_increment_model import PACKED, WorldIncrementMLP
from preflight_s3_velocity_response_leads import lead_vectors
from preflight_s3_world_increment_supervision import add, dot, mul, norm
from train_s3_world_increment import objectives, reference_data
from train_s3_world_increment_data61 import combine, extend_fold, probe, supervision


PHASE = 'spin_supervision_window_preflight'
SPIN = slice(6, 9)
EARLY = slice(8, 14)
LATE = slice(14, 16)


def gradient_info(loss, gradient, reference):
    size = norm(gradient)
    if not np.isfinite(size) or size == 0.:
        raise ValueError('undefined gradient projection')
    return {'loss': float(loss), 'gradient_norm': size,
            'cosine_to_reference': dot(gradient, reference)/(size*norm(reference))}


def nominal_examples(fold, cfg, dt):
    batch = fold.pairs
    rows = {}
    # Prefer the timed capture when the episode and snapshot also have a legacy view.
    for i, key in enumerate(zip(batch.episode_ids, batch.snapshot_ticks)):
        rows[key] = i
    indices = np.array(list(rows.values()))
    if len(indices) != 19 or not batch.baseline_mask[indices, :16].all():
        raise ValueError('nominal targets for the inspected window are incomplete')
    known = []
    for i in indices:
        state = _parse_state(batch.initial_state[i])
        known.append(known_rollout(state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                                   batch.nominal_actions[i], cfg, dt))
    target = .5*(batch.baseline_first[indices]+batch.baseline_second[indices])
    return batch.initial[indices], np.stack(known), target


def response_weights(fold, cfg, leads):
    """Replace half the raw H16 spin mass with six equally weighted early leads."""
    spin = cfg['components'][SPIN]
    batch = fold.pairs
    if not batch.noise_mask[:, 8:16].all():
        raise ValueError('some pairs cannot preserve their own shifted weight')
    original = response_components(*leads, cfg['components'], batch.noise_mask, fold.pair_weights)
    candidate = original.copy()
    removed = .5*original[15, SPIN]
    candidate[15, SPIN] -= removed
    candidate[EARLY, SPIN] += removed/6.
    valid = batch.noise_mask[..., None]*fold.pair_weights[:, None, None]
    axes = list(range(6))
    np.testing.assert_array_equal(candidate[:, axes], original[:, axes])
    np.testing.assert_allclose((valid*candidate).sum(axis=1), (valid*original).sum(axis=1), rtol=0., atol=2e-15)
    if not np.array_equal(original[[7,15,31],SPIN], leads[0][[7,15,31], None]*spin):
        raise ValueError('unexpected old spin weighting')
    return original, candidate, valid


def spin_response_loss(model, fold, cfg, scales, pair_initial, pair_known, weights, valid, full_mass):
    """Evaluate the selected response numerator using the original full denominator."""
    batch = fold.pairs
    spin_weights = np.zeros_like(weights)
    spin_weights[:, SPIN] = weights[:, SPIN]
    mass = float(np.sum(valid*spin_weights))
    options = {'sample_weights': fold.pair_weights, 'absolute_scale_si': scales['state_error_scale_si']}
    value, gradient = paired_loss(model, pair_initial, pair_known, np.stack((batch.plus_target,batch.minus_target)),
        np.stack((batch.plus_mask,batch.minus_mask)), batch.response_mask, batch.noise_mask, batch.response_floor,
        np.ones(32), spin_weights, cfg['response_scales_si'], 0., cfg['pairs']['response_weight'], **options)
    return value['loss']*mass/full_mass, mul(gradient,mass/full_mass)


def sampler_sensitivity(model, fold, cfg, scales, model_cfg, anchor, stream, baseline, delta, guard):
    anchors = {h: (model.normalize(w.initial), model.features(w.known, model_cfg, anchor, fold.known_mean,
                fold.known_scale), model.normalize(w.target)) for h, w in fold.anchors.items()}

    def anchor_gradient(h, take):
        x, known, y = anchors[h]
        return model.loss_and_grad(x[take], known[take], y[take], state_weights=cfg['components'],
                                   state_error_scale_si=scales['state_error_scale_si'])[1]

    fixed = add(baseline, mul(anchor_gradient(*stream[0]), -1.))
    delta_norm = norm(delta)
    means = {name: np.zeros_like(value) for name, value in baseline.items()}
    fractions = []
    for index, (h, take) in enumerate(stream):
        original = add(fixed, anchor_gradient(h, take))
        fractions.append(delta_norm/norm(original))
        for name in means:
            means[name] += original[name]/len(stream)
        if index % 90 == 0:
            guard('fixed_s2_sampler')
    return {'draw_count': len(fractions), 'delta_over_gradient_min_median_p95_max':
               np.quantile(fractions, [0.,.5,.95,1.]).tolist(),
            'delta_over_mean_gradient': delta_norm/norm(means)}


def endpoint_stats(model, fold, cfg, scales, leads, model_cfg, anchor, data, coeff, weight_info, nominal, stream, seed, guard):
    old_weights, new_weights, valid = weight_info
    full_mass = float(np.sum(valid*old_weights))
    evaluate, pair_initial, pair_known = objectives(model, fold, cfg, scales, leads, model_cfg, anchor)
    old = combine(model, evaluate, data, coeff)

    def signed(matrix):
        return spin_response_loss(model, fold, cfg, scales, pair_initial, pair_known, matrix, valid, full_mass)

    def candidate(h, take):
        s2, metrics, grad = old(h, take)
        old_loss, old_grad = signed(old_weights)
        new_loss, new_grad = signed(new_weights)
        return s2, {**metrics,'loss': metrics['loss']+new_loss-old_loss}, add(grad, mul(old_grad,-1),new_grad)

    baseline, original_gradient = probe(model, old, *stream[0], seed)
    changed, candidate_gradient = probe(model, candidate, *stream[0], seed)
    if baseline['s2_loss'] != changed['s2_loss']:
        raise ValueError('candidate changed the S2 metric')

    raw, known, target = nominal
    features = model.features(known, model_cfg, anchor, fold.known_mean, fold.known_scale)
    spin = np.r_[np.zeros(6),np.ones(3)]
    windows = {}
    for name, window in (('H9_H14',EARLY),('H15_H16',LATE)):
        step = np.zeros(32)
        step[window] = 1/(window.stop-window.start)
        # These 19 snapshots have validated targets through H16; later points are excluded.
        nl, ng = model.loss_and_grad(model.normalize(raw), features[:,:16], model.normalize(target[:,:16]),
            step_weights=step[:16], state_weights=spin, state_error_scale_si=scales['state_error_scale_si'])
        diagnostic = np.zeros_like(old_weights)
        diagnostic[window,SPIN] = step[window,None]*cfg['components'][SPIN]
        pl, pg = signed(diagnostic)
        windows[name] = {'nominal':gradient_info(nl,ng,original_gradient),
                         'paired_half_difference':gradient_info(pl,pg,original_gradient),
                         'nominal_vs_paired_gradient_cosine':dot(ng,pg)/(norm(ng)*norm(pg)),
                         'spin_components_eligible':int(np.sum(fold.pairs.response_mask[:,window,SPIN]))}
        guard(name)

    # Verify the delta construction against an independent recomputation of the entire response matrix.
    batch = fold.pairs
    args = (model,pair_initial,pair_known,np.stack((batch.plus_target,batch.minus_target)),
            np.stack((batch.plus_mask,batch.minus_mask)),batch.response_mask,batch.noise_mask,batch.response_floor)
    evaluate_parts = lambda weights: paired_loss(*args,np.ones(32),weights,cfg['response_scales_si'],0.,
        cfg['pairs']['response_weight'],sample_weights=fold.pair_weights,absolute_scale_si=scales['state_error_scale_si'])
    left, lg = evaluate_parts(old_weights)
    right, rg = evaluate_parts(new_weights)
    delta = add(candidate_gradient,mul(original_gradient,-1))
    np.testing.assert_allclose(changed['pair']['loss']-baseline['pair']['loss'],right['loss']-left['loss'],atol=2e-14,rtol=0.)
    for k in delta:
        np.testing.assert_allclose(delta[k],rg[k]-lg[k],rtol=0.,atol=2e-14)
    a, ga = signed(old_weights)
    b, gb = signed(new_weights)
    sensitivity = sampler_sensitivity(model, fold, cfg, scales, model_cfg, anchor, stream, original_gradient, delta, guard)
    return {'original_probe':baseline,'candidate_probe':changed,'windows':windows,'fixed_s2_sampler_sensitivity':sensitivity,
            'old_signed_spin':gradient_info(a,ga,original_gradient),
            'new_signed_spin':gradient_info(b,gb,original_gradient),
            'delta_gradient_norm':norm(delta),'delta_gradient_fraction_of_old_total':norm(delta)/norm(original_gradient),
            'old_vs_new_total_gradient_cosine':dot(original_gradient,candidate_gradient)/(norm(original_gradient)*norm(candidate_gradient)),
            'new_total_vs_new_signed_spin_gradient_cosine':dot(candidate_gradient,gb)/(norm(candidate_gradient)*norm(gb))}


def run(plan, out, guard):
    if plan['updates']:
        raise ValueError('spin window preflight cannot update the model')
    fit = read(plan, plan['fit_plan'])
    previous = read(plan, plan['fit_validation'])
    if (previous['status'],previous['training_pairs'],previous['updates'],previous['test_episode_files_read']) != ('completed',61,900,False):
        raise ValueError('missing accepted 61-pair endpoint')
    if previous['plan_sha256'] != plan['files_sha256'][plan['fit_plan']]:
        raise ValueError('61-pair endpoint plan changed')
    protocol,cfg,scales,fold,held,draws = reference_data(fit,guard)
    if previous['sampler_sha256'] != draws:
        raise ValueError('fixed sampler changed')
    new = [prepare_timed_pairs(ROOT,p,fit['files_sha256'][p]) for p in fit['new_timed_manifests']]
    fold, _ = extend_fold(fold,new)
    archived = read(plan, Path(plan['fit_validation']).with_name('data_preflight.json'))
    np.testing.assert_array_equal(fold.pair_weights,archived['expanded_weights'])
    data,_ = supervision(fit,fold,fit['timed_manifests'],guard)
    parent = read(plan, fit['matched_fit_plan'])
    coeff = read(plan,parent['window_protocol'])['auxiliary']['coefficients']
    cfg_dev = held[1]
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'),cfg_dev['initial']['anchor_xy_tool_m'])[0]
    leads = lead_vectors(read(plan,read(plan,fit['coverage_design'])['reference_design']),32)
    weights = response_weights(fold,cfg,leads)
    checkpoint = Path(plan['fit_validation']).with_name('world_increment_900.npz')
    if previous['artifacts_sha256'][checkpoint.name] != plan['files_sha256'][checkpoint.as_posix()]:
        raise ValueError('61-pair checkpoint changed')
    with np.load(ROOT/checkpoint,allow_pickle=False) as saved:
        endpoint = WorldIncrementMLP.from_state_dict(saved)
        if saved['updates'].item() != 900:
            raise ValueError('checkpoint is not the frozen endpoint')
    initial = WorldIncrementMLP(PACKED,fold.state_mean,fold.state_scale,endpoint.dt_s,plan['seed'],
        inc_center=protocol['world_increment_center_m_s'],hidden=endpoint.hidden,residual_gain=endpoint.residual_gain)
    horizons = cfg['anchors']['horizons']
    rng = np.random.default_rng(plan['seed'])
    stream = []
    for index in range(900):
        h = horizons[index%len(horizons)]
        stream.append((h,rng.integers(len(fold.anchors[h].initial),size=cfg['anchors']['batch_size'])))
    if sha256(b''.join(np.r_[h,t].astype('<i8').tobytes() for h,t in stream)).hexdigest() != draws:
        raise ValueError('S2 draw stream did not reproduce')
    nominal = nominal_examples(fold,cfg_dev,endpoint.dt_s)
    results = {name:endpoint_stats(model,fold,cfg,scales,leads,cfg_dev,anchor,data,coeff,weights,nominal,
             stream,plan['seed'],guard) for name,model in (('initialization',initial),('61_pairs_900',endpoint))}
    old = results['initialization']['original_probe']
    ref = read(plan,Path(plan['fit_validation']).with_name('fit_preflight.json'))
    np.testing.assert_allclose([old['s2_loss'],old['pair']['loss']],
                               [ref['initial_s2_loss'],ref['initial_pair_loss']['loss']],rtol=0.,atol=1e-14)
    old_weights,new_weights,valid = weights
    binding = {'nominal_snapshots':len(nominal[0]),'checkpoints':results,'sampler_sha256':draws,
               'old_spin_weights_by_lead':old_weights[:,SPIN].tolist(),'new_spin_weights_by_lead':new_weights[:,SPIN].tolist(),
               'original_per_axis_spin_mass':(valid*old_weights).sum(axis=(0,1))[SPIN].tolist(),
               'candidate_per_axis_spin_mass':(valid*new_weights).sum(axis=(0,1))[SPIN].tolist(),
               'eligible_spin_samples_by_lead':np.sum(fold.pairs.response_mask[...,SPIN],axis=0).tolist(),
               'candidate_changes_only_signed_spin':True}
    write_json(out/'spin_gradient_preflight.json',binding)
    return {'fixed_parameter_sets':len(results),'training_pairs':61,'old_loss_and_sampler_reproduced':True,
            'spin_weight_mass_preserved_per_pair':True,'gradient_difference_and_fd_verified':True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
