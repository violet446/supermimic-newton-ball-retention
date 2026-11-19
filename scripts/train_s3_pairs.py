"""Refit the S2 world model with bounded Newton paired action supervision."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.guard import process_memory
from ball_retention.resources import directory_bytes, snapshot

CONFIG = {
    'seed':20261111, 'updates_per_variant':900, 'variants':['raw_request','tool_only'],
    'learning_rate':0.0003, 'gradient_clip':5., 'anchor_batch_size':64,
    'anchor_horizons':[1,4,8], 'pair_leads':[8,16,32], 'pair_lead_weights':[.2,.3,.5],
    'component_weights':[1.5]*3+[.7]*3+[.4]*3,
    'response_scales_si':[.0005]*3+[.025]*3+[.5]*3,
    'anchor_weight':1., 'absolute_weight':.5, 'response_weight':.2, 'validate_every':30,
    'wall_s':480., 'rss_gib':1., 'commit_gib':1.5, 'max_output_mib':64.,
    'min_start_ram_gib':2., 'min_running_ram_gib':.75, 'min_disk_gib':10.,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def train(out, capture_manifest, s2_checkpoint, dataset, started):
    import numpy as np
    from ball_retention.model_data import KNOWN_NAMES, prepare_dataset
    from ball_retention.model_eval import evaluate_windows
    from ball_retention.pair_data import prepare_pairs
    from ball_retention.pair_loss import paired_loss
    from ball_retention.world_model import Adam, ResidualMLP

    cfg = CONFIG
    gib = 1024**3

    def check_resources(variant, update):
        elapsed = time.monotonic()-started
        machine = snapshot(ROOT)
        memory = process_memory(os.getpid())
        bytes_written = directory_bytes(out)
        row = {'variant':variant,'update':update,'elapsed_s':elapsed,'output_bytes':bytes_written,**machine,**memory}
        with (out/'resources.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        ram_floor = cfg['min_start_ram_gib'] if update < 0 else cfg['min_running_ram_gib']
        if (elapsed >= cfg['wall_s'] or machine['available_ram_bytes'] < ram_floor*gib
                or machine['disk_free_bytes'] < cfg['min_disk_gib']*gib or memory['rss_bytes'] > cfg['rss_gib']*gib
                or memory['private_commit_bytes'] > cfg['commit_gib']*gib or bytes_written > cfg['max_output_mib']*1024**2):
            raise RuntimeError('paired refit exceeded CPU training resource limits')

    check_resources(None, -1)
    source_manifest = json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    capture_entries = json.loads(capture_manifest.read_text(encoding='utf-8'))['entries']
    source_identity = {row['episode_id']:(row['group_id'],row['split']) for row in source_manifest['entries']}
    for entry in capture_entries:
        if (entry['split'] not in ('train','validation')
                or source_identity.get(entry['episode_id'])!=(entry['group_id'],entry['split'])):
            raise ValueError('paired capture changed an S2 source episode group or split')
    anchor = prepare_dataset(dataset, horizons=tuple(cfg['anchor_horizons']))
    paired = prepare_pairs(capture_manifest)
    with np.load(s2_checkpoint, allow_pickle=False) as saved:
        starting_state = {key:saved[key] for key in saved.files}
    mean, scale = starting_state['known_mean'], starting_state['known_scale']
    feature_names = starting_state['known_names']
    if (tuple(feature_names) != KNOWN_NAMES or not np.array_equal(mean,anchor.known_mean)
            or not np.array_equal(scale,anchor.known_scale) or not np.array_equal(starting_state['state_mean'],anchor.state_mean)
            or not np.array_equal(starting_state['state_scale'],anchor.state_scale)):
        raise ValueError('S2 normalization or feature schema mismatch')
    if (not np.isclose(paired.dt_s,anchor.dt_s,atol=1e-12,rtol=0)
            or not np.isclose(float(starting_state['dt_s']),anchor.dt_s,atol=1e-12,rtol=0)):
        raise ValueError('S2 and paired controls must share the physical clock')
    if paired.model_config!=anchor.model_config:
        raise ValueError('paired capture must preserve the S2 tool feature configuration')
    lead_weights = np.zeros(paired.horizon)
    lead_weights[np.asarray(cfg['pair_leads'])-1] = cfg['pair_lead_weights']
    component_weights = np.asarray(cfg['component_weights'])
    response_scales = np.asarray(cfg['response_scales_si'])
    direct_channels = np.asarray([KNOWN_NAMES.index('roll_request'), KNOWN_NAMES.index('pitch_request')])
    terminal_episode_ids = set(anchor.validation_endings.episode_ids[anchor.validation_endings.terminated])

    def fit_variant(variant):
        model = ResidualMLP.from_state_dict(starting_state)
        keep = np.ones(len(KNOWN_NAMES), dtype=bool)
        if variant == 'tool_only':
            keep[direct_channels] = False
        model.params['W1'][9+np.flatnonzero(~keep)] = 0.
        optimizer = Adam(model.params, learning_rate=cfg['learning_rate'], max_grad_norm=cfg['gradient_clip'])
        rng = np.random.default_rng(cfg['seed'])

        def normalize_known(known):
            return ((known-mean)/scale)*keep

        def prep_windows(windows):
            return model.normalize(windows.initial), normalize_known(windows.known), model.normalize(windows.target)

        def prep_pairs(data):
            return (model.normalize(data.initial),
                    normalize_known(np.stack((data.plus_known,data.minus_known))),
                    np.stack((data.plus_target,data.minus_target)), np.stack((data.plus_mask,data.minus_mask)),
                    data.response_mask,data.noise_mask,data.response_floor)

        train_windows = {h:prep_windows(windows) for h,windows in anchor.train.items()}
        validation_windows = {h:prep_windows(windows) for h,windows in anchor.validation.items()}
        train_pairs, val_pairs = prep_pairs(paired.train), prep_pairs(paired.validation)

        def pair_objective(values, *, gradients):
            return paired_loss(model,*values,lead_weights,component_weights,response_scales,
                               cfg['absolute_weight'],cfg['response_weight'],gradients=gradients)

        def validation_score():
            route_scores = []
            for h,(x,k,y) in validation_windows.items():
                ids = anchor.validation[h].episode_ids
                route = np.array([name not in terminal_episode_ids for name in ids])
                if not np.any(route):
                    raise ValueError('no complete-route S2 validation windows')
                diff = model.rollout(x[route],k[route])[:,-1]-y[route,-1]
                route_scores.append(float(np.average(np.mean(diff**2,axis=0),weights=component_weights)))
            pair, _ = pair_objective(val_pairs,gradients=False)
            route_score = float(np.mean(route_scores))
            return {'anchor_route':route_score,'paired_absolute':pair['absolute'],'paired_response':pair['response'],
                    'score':cfg['anchor_weight']*route_score+pair['loss']}

        best = initial = validation_score()
        best_state = model.state_dict()
        best_update = 0
        with (out/(variant+'_training.jsonl')).open('x', encoding='utf-8') as stream:
            stream.write(json.dumps({'update':0,'validation':initial},allow_nan=False)+'\n')
            for update in range(1,cfg['updates_per_variant']+1):
                if time.monotonic()-started >= cfg['wall_s']:
                    raise RuntimeError('paired refit reached wall budget')
                horizons = cfg['anchor_horizons']
                h = horizons[(update-1)%len(horizons)]
                x,k,y = train_windows[h]
                take = rng.integers(len(x),size=cfg['anchor_batch_size'])
                anchor_loss, anchor_grad = model.loss_and_grad(x[take],k[take],y[take],state_weights=component_weights)
                pair, pair_grad = pair_objective(train_pairs,gradients=True)
                grads = {name:cfg['anchor_weight']*grad+pair_grad[name] for name,grad in anchor_grad.items()}
                norm = optimizer.step(grads)
                if np.any(model.params['W1'][9+np.flatnonzero(~keep)]):
                    raise RuntimeError('masked action channel weights changed')
                record = {'update':update,'anchor_horizon':h,'anchor_loss':anchor_loss,'paired':pair,'gradient_norm':norm}
                if update%cfg['validate_every']==0 or update==cfg['updates_per_variant']:
                    check_resources(variant,update)
                    record['validation'] = validation_score()
                    if record['validation']['score'] < best['score']:
                        best, best_update, best_state = record['validation'], update, model.state_dict()
                stream.write(json.dumps(record,allow_nan=False)+'\n')

        restored = ResidualMLP.from_state_dict(best_state)
        path = out/(variant+'.npz')
        np.savez_compressed(path, **best_state, known_mean=mean, known_scale=scale, known_names=feature_names,
                            feature_keep_mask=keep, best_update=np.array(best_update))
        with np.load(path,allow_pickle=False) as saved:
            loaded = ResidualMLP.from_state_dict(saved)
        x,k,_ = validation_windows[8]
        np.testing.assert_array_equal(restored.rollout(x[:2],k[:2]),loaded.rollout(x[:2],k[:2]))
        predictions = {}
        for h,(x,k,_) in validation_windows.items():
            windows = anchor.validation[h]
            raw = loaded.denormalize(loaded.rollout(x,k))
            predictions[str(h)] = evaluate_windows(raw,windows.target,windows.initial,anchor.dt_s,windows.episode_ids,leads=(h,))
        print(f'{variant}: {cfg["updates_per_variant"]} updates, best {best_update}, score {best["score"]:.6g}',flush=True)
        return {'checkpoint':path.name,'checkpoint_sha256':sha256(path),'best_update':best_update,
                'updates':cfg['updates_per_variant'],'initial_validation':initial,'best_validation':best,
                'feature_keep_mask':keep.tolist(),'reload_equal':True,'s2_validation':predictions}

    results = {variant:fit_variant(variant) for variant in cfg['variants']}
    check_resources(None, sum(row['updates'] for row in results.values()))
    write_json(out/'summary.json',{'status':'completed','variants':results,'pair_data':paired.report,
                                  'total_updates':sum(row['updates'] for row in results.values()),
                                  'wall_s':time.monotonic()-started})


def main():
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'
    capture = ROOT/'runs/s3_pair_refit_20261112/capture_manifest.json'
    out = capture.parent/'fit'
    checkpoint = ROOT/'runs/s2_world_model_20261112/world_model.npz'
    s2 = json.loads((ROOT/'configs/s2_world_model.json').read_text(encoding='utf-8'))
    dataset = ROOT/s2['dataset']
    s2_summary = json.loads((checkpoint.parent/'summary.json').read_text(encoding='utf-8'))
    if sha256(checkpoint)!=s2_summary['checkpoint_sha256'] or sha256(dataset/'manifest.json')!=s2['dataset_manifest_sha256']:
        raise ValueError('S2 checkpoint or data split changed')
    out.mkdir(exist_ok=False)
    plan = {'config':CONFIG,'device':'cpu','threads':1,'total_update_cap':1800,'capture_manifest_sha256':sha256(capture),
            'starting_checkpoint_sha256':sha256(checkpoint),'s2_dataset_manifest_sha256':sha256(dataset/'manifest.json'),
            'checkpoint_selection':'S2 complete-route validation anchor plus paired validation absolute and response losses',
            'unresolved_response':'zero-centered deadzone with per-component replay floor',
            'selection_split':'validation','test_used_for_fitting_or_evaluation':False,
            'source_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in
                             (Path(__file__),ROOT/'src/ball_retention/pair_loss.py',ROOT/'src/ball_retention/pair_data.py')}}
    write_json(out/'plan.json',plan)
    started = time.monotonic()
    try:
        train(out,capture,checkpoint,dataset,started)
    except Exception as exc:
        write_json(out/'failure.json',{'status':'failed','error':str(exc),'wall_s':time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
