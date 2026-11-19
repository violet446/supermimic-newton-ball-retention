"""Measure what a positive, train-only position response scale can correct."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    from ball_retention.guard import process_memory
    from ball_retention.pair_data import prepare_pairs
    from ball_retention.resources import snapshot
    from ball_retention.world_model import ResidualMLP

    root = ROOT/'runs/s3_pair_refit_20261112'
    output = root/'response_calibration.json'
    budget = {'threads':1,'wall_s':120,'rss_gib':1,'commit_gib':1.5,'max_output_mib':8}
    started = time.monotonic()
    limits = 1024**3

    def check():
        ram = snapshot(ROOT)['available_ram_bytes']
        memory = process_memory(os.getpid())
        if (ram < .75*limits or time.monotonic()-started>budget['wall_s'] or
                memory['rss_bytes']>budget['rss_gib']*limits or memory['private_commit_bytes']>budget['commit_gib']*limits):
            raise RuntimeError('calibration diagnostic resource limit reached')

    check()
    manifest = root/'capture_manifest.json'
    data = prepare_pairs(manifest)
    leads = [8,16,32]
    report = {'budget':budget,'leads':leads,'capture_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'method':'median Newton/model norm ratio on train-resolved pairs at 8/16/32, one positive scale per checkpoint',
              'scope':'position response diagnostic; checkpoint weights and direction preserved', 'models':{}}
    sources = {'s2':ROOT/'runs/s2_world_model_20261112/world_model.npz',
               'raw_request':root/'fit/raw_request.npz','tool_only':root/'fit/tool_only.npz'}
    for name,path in sources.items():
        with np.load(path,allow_pickle=False) as checkpoint:
            model = ResidualMLP.from_state_dict(checkpoint)
            mean,scale = checkpoint['known_mean'],checkpoint['known_scale']

        def differences(batch):
            initial = model.normalize(batch.initial)
            pos = model.denormalize(model.rollout(initial,(batch.plus_known-mean)/scale))
            neg = model.denormalize(model.rollout(initial,(batch.minus_known-mean)/scale))
            x = .5*(pos[...,:3]-neg[...,:3])
            y = .5*(batch.plus_target[...,:3]-batch.minus_target[...,:3])
            noise = np.linalg.norm(.5*(batch.baseline_first[...,:3]-batch.baseline_second[...,:3]),axis=-1)
            yn = np.linalg.norm(y,axis=-1)
            resolved = batch.noise_mask & (yn>=np.maximum(1e-6,5*noise))
            return x,y,yn,resolved

        x,_,yn,resolved = differences(data.train)
        xn = np.linalg.norm(x,axis=-1)
        indices = np.asarray(leads)-1
        fit_mask = resolved[:,indices] & (xn[:,indices]>1e-12)
        if not fit_mask.any():
            raise ValueError('no measured train responses available for calibration')
        factor = float(np.median(yn[:,indices][fit_mask]/xn[:,indices][fit_mask]))
        row = {'train_scale':factor,'fit_pairs_across_leads':int(fit_mask.sum()),'validation':{}}
        x,y,yn,resolved = differences(data.validation)
        xn = np.linalg.norm(x,axis=-1)
        for lead in leads:
            k = lead-1
            good = resolved[:,k]
            a,b,denom = x[good,k],y[good,k],yn[good,k]
            if len(denom)==0:
                row['validation'][str(lead)] = {'resolved_pairs':0}
                continue
            gains = xn[good,k]/denom
            row['validation'][str(lead)] = {'resolved_pairs':len(denom),
                'positive_dot_pairs':int(np.sum(np.sum(a*b,axis=-1)>0)),
                'median_gain_before':float(np.median(gains)), 'median_gain_after':float(np.median(gains*factor)),
                'median_relative_vector_error_before':float(np.median(np.linalg.norm(a-b,axis=-1)/denom)),
                'median_relative_vector_error_after':float(np.median(np.linalg.norm(a*factor-b,axis=-1)/denom))}
        report['models'][name] = row
        check()
    report['wall_s'] = time.monotonic()-started
    serialized = json.dumps(report,indent=2,allow_nan=False)+'\n'
    if len(serialized.encode('utf-8'))>budget['max_output_mib']*1024**2:
        raise RuntimeError('calibration output limit exceeded')
    with output.open('x',encoding='utf-8') as stream:
        stream.write(serialized)
    for name,row in report['models'].items():
        print(name,'train scale',row['train_scale'],'validation 32',row['validation']['32'])


if __name__=='__main__':
    main()
