"""Collect the longer, group-separated Newton action pairs for model refitting."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from check_s3_actions import ROOT, digest, schedule_actions, write_json


def preflight(cfg, dt, substeps):
    import numpy as np
    from ball_retention.controller import OnlineTiltController
    from ball_retention.geometry import load_tool, support_point
    from ball_retention.prescribed import PrescribedPath
    from ball_retention.robot import Robot

    spec = cfg['action_pairs']
    ticks, horizon = spec['snapshot_ticks'], spec['horizon']
    actions = np.asarray(schedule_actions(spec['schedule'], 0, ticks[-1]+horizon))
    robot = Robot(ROOT/'assets/ec66_description.urdf')
    mesh = load_tool(ROOT/'assets/shovel.STL')
    anchor = support_point(mesh,cfg['initial']['anchor_xy_tool_m'])[0]
    path = PrescribedPath(ROOT/cfg['prescribed']['joint_path'],cfg,moving=True)
    controller = OnlineTiltController(robot,cfg,anchor,path,ROOT/'assets')
    offsets = [np.zeros(2)]+[d*sign*magnitude for d in np.eye(2)
                           for magnitude in spec['amplitudes'] for sign in (-1,1)]
    report = []
    for tick in range(ticks[-1]+1):
        if tick in ticks:
            saved = controller.snapshot()
            speed, accel = 0., 0.
            for offset in offsets:
                controller.restore(saved)
                for k,action in enumerate(actions[tick:tick+horizon]+offset):
                    if np.max(np.abs(action))>=1:
                        raise ValueError('preflight found an action at the clipping boundary')
                    for command in controller.plan((tick+k)*dt,action,substeps):
                        speed = max(speed,float(np.max(np.abs(command.qd))))
                        accel = max(accel,float(np.max(np.abs(command.qdd))))
                        controller.commit(command)
            controller.restore(saved)
            report.append({'tick':tick,'branches_checked':len(offsets),'max_joint_speed_rad_s':speed,
                           'max_joint_acceleration_rad_s2':accel})
        if tick<ticks[-1]:
            for command in controller.plan(tick*dt,actions[tick],substeps):
                controller.commit(command)
    return report


def run():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'
    out = ROOT/'runs/s3_pair_refit_20261112'
    out.mkdir(parents=True,exist_ok=False)
    dataset = ROOT/'data/s2_pilot_20261112'
    dataset_manifest = dataset/'manifest.json'
    original = json.loads(dataset_manifest.read_text())
    trained = ROOT/'runs/s2_world_model_20261112'
    model_plan = json.loads((trained/'plan.json').read_text())
    if digest(dataset_manifest)!=model_plan['config']['dataset_manifest_sha256']:
        raise ValueError('original S2 split changed')
    physics_dt = original['dt_s']
    substeps = original['control_substeps']
    dt = physics_dt*substeps
    horizon, amplitude, seed = 32, .2, 20261111
    max_steps, wall_limit = 27000, 1200
    selected = [('train_roll','train',[48,156]),('train_pitch','train',[72,152]),
                ('validation_lateral','validation',[72,168])]
    cases = []
    for episode,split,ticks in selected:
        entry = next(row for row in original['entries'] if row['episode_id']==episode and row['split']==split)
        source = dataset/entry['provenance']['config']
        if digest(source)!=entry['provenance']['config_sha256']:
            raise ValueError('source case configuration changed')
        cfg = json.loads(source.read_text())
        schedule = cfg.pop('collection')['case']['actions']
        cfg['action_pairs'] = {'scenario':'route','master_seed':seed,'schedule':schedule,'snapshot_ticks':ticks,
                               'horizon':horizon,'amplitudes':[amplitude],'replay_as_noise':True}
        reserved = (ticks[-1]+len(ticks)*6*horizon)*substeps
        cfg['budget'].update(wall_s=wall_limit/3,max_cases=3,max_physics_steps=max_steps,max_trace_steps=reserved)
        path = out/(episode+'_config.json')
        write_json(path,cfg)
        cases.append((cfg,{'group_id':entry['group_id'],'split':split,'episode_id':episode,'snapshot_ticks':ticks,
                           'config_path':path.name,'config_sha256':digest(path),'reserved_steps':reserved}))
    if len({c['group_id'] for _,c in cases})!=3 or sum(c['reserved_steps'] for _,c in cases)>max_steps:
        raise ValueError('group split or reserved step count violates the capture scope')
    plan = {'schema':'s3-refit-capture-plan-v1','master_seed':seed,'environments':1,'max_worker_starts':3,
            'max_physics_steps':max_steps,'max_wall_s':wall_limit,'horizon':horizon,'amplitude':amplitude,
            'control_leads':[8,16,32],'resource_limits':cases[0][0]['budget'],
            'stop_conditions':['controller preflight or planner rejects','physics or resource guard failure',
                               'discrete replay divergence','invalid capture format'],
            'source_dataset_manifest_sha256':digest(dataset_manifest),'source_script_sha256':digest(Path(__file__)),
            'cases':[case for _,case in cases]}
    write_json(out/'capture_plan.json',plan)
    for cfg,case in cases:
        case_report = preflight(cfg,dt,substeps)
        write_json(out/(case['episode_id']+'_preflight.json'),case_report)
        print(f"{case['episode_id']}: motion preflight passed",flush=True)

    steps, entries, attempts = 0, [], []
    started = time.monotonic()
    for cfg,case in cases:
        if steps+case['reserved_steps']>max_steps or time.monotonic()-started>=wall_limit:
            raise RuntimeError('capture budget exhausted')
        episode = case['episode_id']
        folder = out/episode
        command = [sys.executable,'-B',str(ROOT/'diagnostics/resource_probe/runner.py'),'--mode','s3-pairs',
                   '--config',str(out/case['config_path']),'--output',str(folder),'--steps',str(case['reserved_steps'])]
        print(f'Collecting {episode} {case["split"]} snapshots {case["snapshot_ticks"]}',flush=True)
        process = subprocess.run(command,cwd=ROOT)
        result = json.loads((folder/'result.json').read_text())
        worker = result.get('worker') or {}
        steps += worker.get('physics_steps',0)
        attempts.append({'episode_id':episode,'status':result['status'],'physics_steps':worker.get('physics_steps',0),
                         'reason':worker.get('error') or result.get('stop_reason')})
        write_json(out/'capture_progress.json',{'attempts':attempts,'physics_steps':steps,'wall_s':time.monotonic()-started})
        if process.returncode or result['status']!='completed' or worker.get('status')!='action_pairs_completed':
            raise RuntimeError(f'capture failed for {episode}: {result["status"]}')
        trace = folder/worker['data_file']
        snapshots = [json.loads(line) for line in trace.read_text().splitlines()]
        if ([row['snapshot_tick'] for row in snapshots]!=case['snapshot_ticks'] or
                any(len(row['branches'])!=6 for row in snapshots) or result['config_sha256']!=case['config_sha256']):
            raise ValueError('captured snapshot set differs from the plan')
        entries.append({**case,'path':trace.relative_to(out).as_posix(),'sha256':digest(trace)})
    if steps>max_steps or time.monotonic()-started>wall_limit:
        raise RuntimeError('capture round exceeded its budget')
    write_json(out/'capture_manifest.json',{'schema':'s3-long-pairs-v1','horizon':horizon,'amplitude':amplitude,
               'dt_s':physics_dt,'control_substeps':substeps,'entries':entries,
               'capture_plan_sha256':digest(out/'capture_plan.json'),'physics_steps':steps})
    print(f'Captured {len(entries)} groups, {steps} physics substeps',flush=True)


if __name__=='__main__':
    run()
