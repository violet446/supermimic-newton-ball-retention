"""Bounded 2x2 ablation: frame transport and train-measured response scales."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.action_eval import evaluate_pair
from ball_retention.frame_model import FrameResidualMLP, PACKED_DIM, measured_response_scale, pack_features
from ball_retention.geometry import load_tool, support_point
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES, prepare_dataset
from ball_retention.pair_data import prepare_pairs
from ball_retention.pair_loss import paired_loss
from ball_retention.resources import directory_bytes, snapshot
from ball_retention.world_model import Adam, ResidualMLP

OUT = ROOT/'runs/s3_frame_refit_20261113'
CAPTURE = ROOT/'runs/s3_pair_refit_20261112/capture_manifest.json'
DATA = ROOT/'data/s2_pilot_20261112'
CFG = {'seed':20261111, 'updates_per_variant':900, 'total_updates':3600,
       'variants':['cv_fixed', 'cv_measured', 'frame_fixed', 'frame_measured'],
       'initialization':'fresh identical neural parameters; zero residual output',
       'lr':.0003, 'clip':5., 'batch':64, 'horizons':[1,4,8],
       'leads':[8,16,32], 'lead_weights':[.2,.3,.5],
       'component_weights':[1.5]*3+[.7]*3+[.4]*3,
       'absolute_weight':.5, 'response_weight':.2, 'anchor_weight':1.,
       'validate_every':30, 'wall_s':600., 'rss_gib':1., 'commit_gib':1.5,
       'min_start_ram_gib':2., 'min_running_ram_gib':.75, 'min_disk_gib':10., 'max_output_mib':64.}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run(start):
    def guard(update):
        memory = process_memory(os.getpid())
        machine = snapshot(ROOT)
        row = {'update':update, 'elapsed_s':time.monotonic()-start, **memory, **machine,
               'output_bytes':directory_bytes(OUT)}
        with (OUT/'resources.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        ram = CFG['min_start_ram_gib'] if update < 0 else CFG['min_running_ram_gib']
        if (row['elapsed_s'] >= CFG['wall_s'] or memory['rss_bytes'] > CFG['rss_gib']*1024**3
                or memory['private_commit_bytes'] > CFG['commit_gib']*1024**3
                or machine['available_ram_bytes'] < ram*1024**3
                or machine['disk_free_bytes'] < CFG['min_disk_gib']*1024**3
                or row['output_bytes'] > CFG['max_output_mib']*1024**2):
            raise RuntimeError('frame ablation resource budget exceeded')

    guard(-1)
    data, pairs = prepare_dataset(DATA), prepare_pairs(CAPTURE)
    if data.model_config != pairs.model_config or data.dt_s != pairs.dt_s:
        raise ValueError('model configuration changed between datasets')
    source = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    identity = {e['episode_id']:(e['group_id'],e['split']) for e in source['entries']}
    for e in json.loads(CAPTURE.read_text(encoding='utf-8'))['entries']:
        if identity.get(e['episode_id']) != (e['group_id'],e['split']):
            raise ValueError('paired source group/split mismatch')
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), data.model_config['initial']['anchor_xy_tool_m'])[0]
    learned_scale = measured_response_scale(pairs.train)
    weights = np.zeros(pairs.horizon)
    weights[np.array(CFG['leads'])-1] = CFG['lead_weights']
    components = np.array(CFG['component_weights'])
    terminal_ids = set(data.validation_endings.episode_ids[data.validation_endings.terminated])

    def features(raw, frame):
        if frame:
            return pack_features(raw, data.model_config, data.dt_s, anchor, data.state_mean,
                                 data.state_scale, data.known_mean, data.known_scale)
        return (raw-data.known_mean)/data.known_scale

    # Computed from known actuator inputs, never from future measured tool poses.
    cached = {frame:{split:{h:features(w.known, frame) for h,w in windows.items()}
                     for split,windows in (('train',data.train),('validation',data.validation))}
              for frame in (False,True)}
    paired_features = {frame:{split:features(np.stack((b.plus_known,b.minus_known)), frame)
                             for split,b in (('train',pairs.train),('validation',pairs.validation))}
                       for frame in (False,True)}
    write(OUT/'preparation.json', {'anchor_local_m':anchor.tolist(), 'data':data.report, 'pairs':pairs.report,
                                  'measured_response_scale_si':learned_scale.tolist(),
                                  'scale_source':'training paired resolved components, per lead and physical unit',
                                  'frame_baseline':'constant world velocity and world spin; no gravity/contact force'})

    def pair_predictions(model, frame, split):
        b = getattr(pairs, split)
        x = model.normalize(np.concatenate((b.initial,b.initial)))
        k = paired_features[frame][split]
        return model.denormalize(model.rollout(x, k.reshape(-1,pairs.horizon,model.known_dim))).reshape(2,-1,pairs.horizon,9)

    def pair_report(prediction, batch):
        rows = []
        for i in range(len(batch.initial)):
            def masked(values, mask):
                return [v if keep else None for v,keep in zip(values,mask)]
            result = evaluate_pair(prediction[0,i], prediction[1,i],
                                   masked(batch.plus_target[i],batch.plus_mask[i]),
                                   masked(batch.minus_target[i],batch.minus_mask[i]), pairs.amplitude, anchor[:2],
                                   replay_first=masked(batch.baseline_first[i],batch.baseline_mask[i]),
                                   replay_second=masked(batch.baseline_second[i],batch.baseline_mask[i]), leads=CFG['leads'])
            rows.append({'episode':str(batch.episode_ids[i]), 'tick':int(batch.snapshot_ticks[i]),
                         'axis':int(batch.axes[i]), **result})
        summary = {}
        for lead in CFG['leads']:
            comparable = [r['by_lead'][str(lead)] for r in rows if r['by_lead'][str(lead)]['status']=='comparable']
            summary[str(lead)] = {}
            for component in ('position','velocity','angular_velocity'):
                resolved = [r[component] for r in comparable if r[component]['newton_resolved']]
                summary[str(lead)][component] = {'resolved':len(resolved),
                    'aligned':sum(r['direction']=='aligned' for r in resolved),
                    'opposed':sum(r['direction']=='opposed' for r in resolved),
                    'model_unresolved':sum(not r['model_resolved'] for r in resolved),
                    'median_amplitude_ratio':float(np.median([r['model_half_spread']/r['newton_half_spread'] for r in resolved])) if resolved else None,
                    'median_relative_vector_error':float(np.median([r['normalized_error'] for r in resolved])) if resolved else None}
        return {'summary':summary, 'pairs':rows}

    def report(model, frame):
        train_p = pair_predictions(model,frame,'train')
        validation_p = pair_predictions(model,frame,'validation')
        train_report = pair_report(train_p,pairs.train)
        ratios = [v['position']['newton_half_spread']/v['position']['model_half_spread']
                  for r in train_report['pairs'] for v in r['by_lead'].values()
                  if v['status']=='comparable' and v['position']['newton_resolved'] and v['position']['model_half_spread']>0]
        factor = float(np.median(ratios)) if ratios else 1.
        center = np.mean(validation_p,axis=0)
        half = (validation_p[0]-validation_p[1])/2*factor
        route = {}
        for h,w in data.validation.items():
            prediction = model.denormalize(model.rollout(model.normalize(w.initial),cached[frame]['validation'][h]))[:,-1]
            route[str(h)] = {}
            for group in sorted(set(w.episode_ids)):
                select = w.episode_ids==group
                error = prediction[select]-w.target[select,-1]
                route[str(h)][str(group)] = {'windows':int(select.sum()), 'complete_route':group not in terminal_ids,
                    'position_rmse_m':float(np.sqrt(np.mean(np.sum(error[:,:3]**2,axis=-1)))),
                    'velocity_rmse_m_s':float(np.sqrt(np.mean(np.sum(error[:,3:6]**2,axis=-1)))),
                    'spin_rmse_rad_s':float(np.sqrt(np.mean(np.sum(error[:,6:]**2,axis=-1))))}
        return {'validation':pair_report(validation_p,pairs.validation), 'train':train_report,
                'magnitude_calibration':{'train_factor':factor, 'changes_checkpoint':False,
                    'validation':pair_report(np.stack((center+half,center-half)),pairs.validation)},
                's2_by_episode':route}

    results = {}
    for index, name in enumerate(CFG['variants']):
        frame = name.startswith('frame')
        cls, dim = (FrameResidualMLP,PACKED_DIM) if frame else (ResidualMLP,20)
        model = cls(dim,data.state_mean,data.state_scale,data.dt_s,seed=CFG['seed'],residual_gain=[.05]*3+[.25]*6)
        if name.endswith('fixed'):
            results['untrained_'+('frame' if frame else 'cv')] = report(model,frame)
        response_scale = learned_scale if name.endswith('measured') else np.repeat([.0005,.025,.5],3)
        optimizer = Adam(model.params,learning_rate=CFG['lr'],max_grad_norm=CFG['clip'])
        rng = np.random.default_rng(CFG['seed'])
        def pair_objective(split, gradients):
            b = getattr(pairs,split)
            return paired_loss(model,model.normalize(b.initial),paired_features[frame][split],
                               np.stack((b.plus_target,b.minus_target)),np.stack((b.plus_mask,b.minus_mask)),
                               b.response_mask,b.noise_mask,b.response_floor,weights,components,response_scale,
                               CFG['absolute_weight'],CFG['response_weight'],gradients=gradients)
        def validation():
            losses = []
            for h,w in data.validation.items():
                mask = np.array([i not in terminal_ids for i in w.episode_ids])
                pred = model.rollout(model.normalize(w.initial[mask]),cached[frame]['validation'][h][mask])[:,-1]
                error = pred-model.normalize(w.target[mask,-1])
                losses.append(float(np.average(np.mean(error**2,axis=0),weights=components)))
            pair,_ = pair_objective('validation',False)
            return {'anchor':float(np.mean(losses)), **pair, 'score':float(np.mean(losses))+pair['loss']}
        best, best_update, best_state = validation(),0,model.state_dict()
        with (OUT/(name+'_training.jsonl')).open('x',encoding='utf-8') as log:
            log.write(json.dumps({'update':0,'validation':best})+'\n')
            for update in range(1,CFG['updates_per_variant']+1):
                if time.monotonic()-start >= CFG['wall_s']:
                    raise RuntimeError('CPU wall budget exceeded')
                h = CFG['horizons'][(update-1)%3]
                w = data.train[h]
                take = rng.integers(len(w.initial),size=CFG['batch'])
                loss,grads = model.loss_and_grad(model.normalize(w.initial[take]),cached[frame]['train'][h][take],
                                               model.normalize(w.target[take]),state_weights=components)
                pair,pg = pair_objective('train',True)
                norm = optimizer.step({k:v+pg[k] for k,v in grads.items()})
                row = {'update':update,'anchor_loss':loss,'paired':pair,'gradient_norm':norm}
                if update%CFG['validate_every']==0:
                    guard(index*CFG['updates_per_variant']+update)
                    row['validation'] = validation()
                    if row['validation']['score'] < best['score']:
                        best,best_update,best_state = row['validation'],update,model.state_dict()
                log.write(json.dumps(row,allow_nan=False)+'\n')
        model = cls.from_state_dict(best_state)
        path = OUT/(name+'.npz')
        np.savez_compressed(path,**best_state,known_mean=data.known_mean,known_scale=data.known_scale,
                            known_names=np.array(KNOWN_NAMES),anchor_local=anchor,
                            model_config_json=np.array(json.dumps(data.model_config)),response_scale=response_scale)
        with np.load(path,allow_pickle=False) as saved:
            loaded = cls.from_state_dict(saved)
        w = data.validation[8]
        np.testing.assert_array_equal(model.rollout(model.normalize(w.initial[:2]),cached[frame]['validation'][8][:2]),
                                      loaded.rollout(loaded.normalize(w.initial[:2]),cached[frame]['validation'][8][:2]))
        results[name] = {'best_update':best_update,'validation_selection':best,'checkpoint_sha256':digest(path),
                         'reload_equal':True,**report(loaded,frame)}
        write(OUT/'comparison.json',results)
        print(name, 'best',best_update, results[name]['validation']['summary']['32']['position'],flush=True)
    for name,path in (('s2_original',ROOT/'runs/s2_world_model_20261112/world_model.npz'),
                      ('prior_raw_refit',CAPTURE.parent/'fit/raw_request.npz')):
        with np.load(path,allow_pickle=False) as saved:
            if not np.array_equal(saved['known_mean'],data.known_mean) or not np.array_equal(saved['known_scale'],data.known_scale):
                raise ValueError('reference checkpoint normalization changed')
            model = ResidualMLP.from_state_dict(saved)
        results[name] = {'checkpoint_sha256':digest(path),**report(model,False)}
    guard(CFG['total_updates'])
    write(OUT/'comparison.json',results)
    write(OUT/'summary.json',{'status':'completed','total_updates':CFG['total_updates'], 'wall_s':time.monotonic()-start,
                             'selection_split':'validation (previously used); not independent test',
                             'test_used':False,'new_simulation_steps':0})


def main():
    OUT.mkdir(exist_ok=False)
    sources = [Path(__file__),ROOT/'src/ball_retention/frame_model.py',ROOT/'src/ball_retention/world_model.py',
               ROOT/'src/ball_retention/pair_loss.py',ROOT/'src/ball_retention/pair_data.py',ROOT/'src/ball_retention/model_data.py']
    write(OUT/'plan.json',{'config':CFG,'device':'cpu','threads':1,
                          'capture_manifest_sha256':digest(CAPTURE),'s2_manifest_sha256':digest(DATA/'manifest.json'),
                          'asset_sha256':digest(ROOT/'assets/shovel.STL'),
                          'source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},
                          'checkpoint_selection':'complete-route validation state loss plus paired validation loss',
                          'calibration':'positive magnitude scalar fitted only on resolved training pairs',
                          'test_used':False})
    start = time.monotonic()
    try:
        run(start)
    except Exception as exc:
        write(OUT/'failure.json',{'status':'failed','error':str(exc),'wall_s':time.monotonic()-start})
        raise


if __name__ == '__main__':
    main()
