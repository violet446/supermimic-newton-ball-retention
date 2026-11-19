"""Execute the previously fixed guarded loss ablation, preserving old runs."""
import json
from pathlib import Path
import time
import train_s3_merged as training

def main():
    root=training.ROOT
    plan_path=root/'configs/s3_guarded_refit_plan.json'
    spec=json.loads(plan_path.read_text())
    if training.sha(root/spec['starting_checkpoint'])!=spec['starting_checkpoint_sha256']:
        raise ValueError('starting checkpoint changed')
    training.OUT=root/'runs/s3_guarded_refit_20261113'
    training.OUT.mkdir(exist_ok=False)
    training.STARTS={name:spec['starting_checkpoint'] for name in spec['variants']}
    for key in ('seed','updates_per_variant','total_updates','wall_s','rss_gib','commit_gib','output_mib'):
        training.CFG[key]=spec[key]
    sources=[Path(__file__),root/'scripts/train_s3_merged.py',root/'src/ball_retention/guarded_loss.py',
             root/'src/ball_retention/pair_loss.py',root/'configs/s3_pair_collection.json',plan_path]
    training.write(training.OUT/'plan.json',{'experiment':spec,'effective_config':training.CFG,
        'source_sha256':{p.relative_to(root).as_posix():training.sha(p) for p in sources},
        'selection_cadence':30,'test_used':False,'simulation_steps':0})
    started=time.monotonic()
    try: training.run(started,experiment=spec)
    except Exception as exc:
        training.write(training.OUT/'failure.json',{'error':str(exc),'wall_s':time.monotonic()-started})
        raise

if __name__=='__main__': main()
