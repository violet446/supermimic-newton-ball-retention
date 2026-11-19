"""Finish the matched S3 capture after its mixed-unit replay gate stopped it."""
import json
import subprocess
import sys
import time

from check_s3_actions import ROOT, digest, write_json


def run():
    out = ROOT/'runs/s3_action_pairs_20261112'
    original = json.loads((out/'plan.json').read_text())
    first_run = json.loads((out/'collection.json').read_text())
    first_result = json.loads((out/'route/result.json').read_text())
    partial = out/'route/action_pairs.partial.jsonl'
    saved = [json.loads(line) for line in partial.read_text().splitlines()]
    if (first_result['status']!='failed' or len(first_run['runs'])!=1 or
            not first_result['worker']['error'].startswith('baseline restore mismatch at snapshot 90:') or
            [row['snapshot_tick'] for row in saved]!=[48] or len(saved[0]['branches'])!=10):
        raise ValueError('recovery expects a completed tick 48 and the tick 90 replay-gate failure')
    if (out/'recovery_plan.json').exists():
        raise ValueError('recovery evidence already exists')
    worker_steps = first_result['worker']['physics_steps']
    if worker_steps!=first_run['physics_steps']:
        raise ValueError('original physics step counts disagree')
    remaining_wall = original['max_wall_s']-first_run['wall_s']
    stages = []
    parts = [{**original['cases'][0], 'snapshot_ticks':[48], 'directory':'route', 'config':'route_config.json',
              'data_file':partial.name, 'data_sha256':digest(partial), 'worker_status':'failed'}]
    specs = [('route','route_recovery',[90,150],[150],4448,600), ('edge','edge',[0],[],1280,remaining_wall-600)]
    for scenario,directory,ticks,single,reserved,wall in specs:
        source_cfg = out/(scenario+'_config.json')
        cfg = json.loads(source_cfg.read_text())
        cfg['action_pairs'].update(snapshot_ticks=ticks,single_scale_ticks=single,replay_as_noise=True)
        cfg['budget'].update(wall_s=wall,max_trace_steps=reserved)
        path = out/(scenario+'_recovery_config.json')
        write_json(path,cfg)
        part = {'scenario':scenario,'directory':directory,'snapshot_ticks':ticks,'config':path.name,
                'config_sha256':digest(path),'data_file':'action_pairs.jsonl','worker_status':'completed',
                'reserved_steps':reserved,'single_scale_ticks':single}
        parts.append(part)
        stages.append((part,path))
    reserved_total = worker_steps+sum(part['reserved_steps'] for part,_ in stages)
    if remaining_wall<=600 or reserved_total>original['max_physics_steps']:
        raise ValueError('recovery does not fit the remaining round budget')
    recovery = {**original,'cases':parts,'max_worker_starts':3,'reserved_physics_steps':reserved_total,
                'original_plan_sha256':digest(out/'plan.json'),
                'change':'preserve tick 48; compare numeric repeat differences by lead and units, retain strict discrete checks; tick 150 small scale only'}
    write_json(out/'recovery_plan.json',recovery)
    started = time.monotonic()
    attempts = list(first_run['runs'])
    for part,path in stages:
        if time.monotonic()-started >= remaining_wall or worker_steps+part['reserved_steps']>original['max_physics_steps']:
            raise ValueError('round budget exhausted before next worker')
        command = [sys.executable,'-B',str(ROOT/'diagnostics/resource_probe/runner.py'),'--mode','s3-pairs',
                   '--config',str(path),'--output',str(out/part['directory']),'--steps',str(part['reserved_steps'])]
        print(f"Collecting {part['directory']} at {part['snapshot_ticks']}",flush=True)
        process = subprocess.run(command,cwd=ROOT)
        result = json.loads((out/part['directory']/'result.json').read_text())
        used = (result.get('worker') or {}).get('physics_steps',0)
        worker_steps += used
        attempts.append({'scenario':part['scenario'],'directory':part['directory'],
                         'status':result['status'],'physics_steps':used})
        complete = len(attempts)==3 and all(a['status']=='completed' for a in attempts[1:])
        write_json(out/'recovery_collection.json',{'runs':attempts,'capture_completed':complete,
                   'physics_steps':worker_steps,'wall_s':time.monotonic()-started+first_run['wall_s']})
        if process.returncode or result['status']!='completed':
            raise RuntimeError(f"recovery stopped at {part['directory']}: {result['status']}")


if __name__=='__main__':
    run()
