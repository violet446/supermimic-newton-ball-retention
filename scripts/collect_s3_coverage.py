"""Predeclared new initial groups and two perturbation scales; no model fitting."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable] = '1'

from check_s3_actions import ROOT, digest, write_json
from collect_s3_refit import preflight

OUT = ROOT/'runs/s3_coverage_20261113'
SEED, HORIZON, AMPLITUDES, TICKS = 20261111, 32, [.2,.35], [72,168]
MAX_STEPS, WALL = 52000, 1200.
SCHEDULE = [{'start_step':0,'action':[0.,0.]}, {'start_step':48,'action':[.18,.16]},
            {'start_step':120,'action':[-.14,-.12]}, {'start_step':192,'action':[0.,0.]}]
CASES = [('coverage_train_yplus','train',[0.,.008,.020]),
         ('coverage_train_yminus','train',[0.,-.008,.020]),
         ('coverage_validation_diagonal_a','validation',[-.004,.004,.020]),
         ('coverage_validation_diagonal_b','validation',[.004,-.004,.020])]


def prepare():
    import numpy as np
    from ball_retention.geometry import load_tool, sphere_clearance
    from ball_retention.guard import process_memory
    from ball_retention.resources import snapshot, admission
    from ball_retention.robot import Robot

    OUT.mkdir(exist_ok=True)
    if any(p.name != 'preflight_rejected.json' for p in OUT.iterdir()):
        raise ValueError('refuse to overwrite prepared capture evidence')
    started = time.monotonic()
    base_path = ROOT/'configs/s1_online_episode.json'
    base = json.loads(base_path.read_text(encoding='utf-8'))
    robot = Robot(ROOT/'assets/ec66_description.urdf')
    tool,_,_ = robot.fk(base['initial']['joint_q_rad'])
    mesh = load_tool(ROOT/'assets/shovel.STL')
    seeds = np.random.SeedSequence(SEED).spawn(len(CASES))
    reserved = (TICKS[-1]+len(TICKS)*(2+4*len(AMPLITUDES))*HORIZON)*16
    if reserved*len(CASES)>MAX_STEPS:
        raise ValueError('capture exceeds substep cap')
    prior = json.loads((ROOT/'data/s2_pilot_20261112/manifest.json').read_text(encoding='utf-8'))
    previous_initials = []
    for row in prior['entries']:
        path = ROOT/'data/s2_pilot_20261112'/row['provenance']['config']
        if digest(path)!=row['provenance']['config_sha256']:
            raise ValueError('prior config hash changed')
        previous_initials.append(json.loads(path.read_text(encoding='utf-8'))['initial']['ball_center_world_m'])
    cases = []
    for (name,split,offset),seed in zip(CASES,seeds):
        cfg = copy.deepcopy(base)
        cfg['seed'] = int(seed.generate_state(1)[0])
        center = np.array(base['initial']['ball_center_world_m'])+offset
        if any(np.linalg.norm(center-p)<1e-7 for p in previous_initials):
            raise ValueError('new group repeats a previous initial position')
        local = tool[:3,:3].T@(center-tool[:3,3])
        clearance = sphere_clearance(mesh,local,cfg['ball']['radius_m'])
        if clearance < .0001 or local[0] < .22 or center[2]-cfg['ball']['radius_m'] <= cfg['physics']['floor_z_m']:
            raise ValueError(f'initial geometry rejected: {name}, clearance={clearance}')
        cfg['initial']['ball_center_world_m'] = center.tolist()
        cfg['initial']['ball_velocity_world_m_s'] = [0.,0.,0.]
        cfg.pop('candidate_provenance',None)
        cfg['revision_reason'] = 'S3 new predeclared initial group and two-scale paired capture; dynamics unchanged'
        cfg['action_pairs'] = {'scenario':'route','master_seed':SEED,'schedule':SCHEDULE,
                               'snapshot_ticks':TICKS,'horizon':HORIZON,'amplitudes':AMPLITUDES,'replay_as_noise':True}
        cfg['budget'].update(wall_s=WALL/len(CASES),max_cases=len(CASES),max_physics_steps=MAX_STEPS,max_trace_steps=reserved)
        path = OUT/(name+'_config.json')
        write_json(path,cfg)
        cases.append({'episode_id':name,'group_id':name+'_initial','split':split,'seed':cfg['seed'],
            'snapshot_ticks':TICKS,'config_path':path.name,'config_sha256':digest(path),'reserved_steps':reserved,
            'ball_offset_world_m':offset,'initial_ball_world_m':center.tolist(),'initial_clearance_m':clearance})
    sources = [Path(__file__),ROOT/'scripts/collect_s3_refit.py',ROOT/'src/ball_retention/pair_data.py',
               ROOT/'diagnostics/resource_probe/action_pairs.py']
    plan = {'schema':'s3-coverage-plan-v1','master_seed':SEED,'environments':1,'max_worker_starts':len(CASES),
        'max_physics_steps':MAX_STEPS,'reserved_physics_steps':reserved*len(CASES),'max_wall_s':WALL,
        'preflight_wall_s':300,'horizon':HORIZON,'amplitudes':AMPLITUDES,'snapshot_ticks':TICKS,
        'resource_limits':cfg['budget'],'base_config_sha256':digest(base_path),
        'source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},'cases':cases,
        'fitting_this_round':False,'test_split_used':False,
        'validation_policy':'new groups evaluated with frozen models first; no checkpoint selection this round',
        'stop_conditions':['preflight or planner rejection','resource or wall budget','nominal premature termination',
                           'discrete replay divergence','invalid capture or hash mismatch'],
        'position_floor_m':1e-6,'noise_multiplier':5.,
        'signal_bands':'unresolved below threshold; near threshold [1,10); strong >=10 times threshold',
        'strong_direction_diagnostic':'cosine >=0.5, report amplitude and relative vector error separately'}
    write_json(OUT/'capture_plan.json',plan)
    machine = snapshot(ROOT)
    if not admission(machine,cfg['budget'])['passed']:
        raise RuntimeError('resource admission failed before preflight')
    # All four cases have identical tool controls; sphere initial position does
    # not enter the prescribed controller. One check covers all nine motions.
    report = preflight(cfg,1/60,16)
    memory = process_memory(os.getpid())
    if time.monotonic()-started>300 or memory['rss_bytes']>1024**3 or memory['private_commit_bytes']>1.5*1024**3:
        raise RuntimeError('preflight CPU budget exceeded')
    write_json(OUT/'preflight.json',{'passed':True,'motion_cases':report,'initial_geometry':cases,
        'shared_control_config_sha256':digest(OUT/cases[-1]['config_path']),
        'scope':'identical prescribed controls; no physical trajectory simulated',
        'wall_s':time.monotonic()-started,**memory})
    print(f'Prepared {len(cases)} groups, {reserved*len(CASES)} reserved substeps; CPU preflight passed',flush=True)


def collect():
    from ball_retention.pair_data import prepare_pairs
    plan = json.loads((OUT/'capture_plan.json').read_text(encoding='utf-8'))
    if not json.loads((OUT/'preflight.json').read_text(encoding='utf-8'))['passed']:
        raise ValueError('preflight required')
    for name,expected in plan['source_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('source changed after preflight: '+name)
    started = time.monotonic()
    steps,attempts,entries = 0,[],[]
    def progress(status,**extra):
        write_json(OUT/'capture_progress.json',{'status':status,'attempts':attempts,'physics_steps':steps,
                                               'wall_s':time.monotonic()-started,**extra})
    try:
        for case in plan['cases']:
            name = case['episode_id']
            path = OUT/case['config_path']
            if digest(path)!=case['config_sha256']:
                raise ValueError('configuration changed after preflight')
            if steps+case['reserved_steps']>MAX_STEPS or time.monotonic()-started+WALL/len(CASES)>WALL:
                raise RuntimeError('insufficient remaining capture budget')
            progress('collecting',active_episode=name)
            folder = OUT/name
            command = [sys.executable,'-B',str(ROOT/'diagnostics/resource_probe/runner.py'),'--mode','s3-pairs',
                       '--config',str(path),'--output',str(folder),'--steps',str(case['reserved_steps'])]
            print(f'Collecting {name} ({case["split"]})',flush=True)
            with (OUT/(name+'_launcher.log')).open('x',encoding='utf-8') as log:
                proc = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
            worker = result.get('worker') or {}
            steps += worker.get('physics_steps',0)
            attempts.append({'episode_id':name,'status':result['status'],'physics_steps':worker.get('physics_steps',0),
                'wall_s':result.get('wall_s'),'reason':worker.get('error') or result.get('stop_reason'),
                'snapshot_report':worker.get('snapshot_report')})
            progress('collecting')
            if proc.returncode or result['status']!='completed' or worker.get('status')!='action_pairs_completed':
                raise RuntimeError(f'capture failed: {name}: {result["status"]}')
            trace = folder/worker['data_file']
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            if [r['snapshot_tick'] for r in rows]!=TICKS or any(len(r['branches'])!=10 for r in rows):
                raise ValueError('snapshot set differs from plan')
            entries.append({**case,'path':trace.relative_to(OUT).as_posix(),'sha256':digest(trace)})
            print(f'Completed {name}: {worker["physics_steps"]} substeps',flush=True)
        for amplitude in AMPLITUDES:
            path = OUT/f'capture_manifest_{amplitude:.2f}.json'
            write_json(path,{'schema':'s3-long-pairs-v1','horizon':HORIZON,'amplitude':amplitude,
                'record_amplitudes':AMPLITUDES,'dt_s':1/960,'control_substeps':16,'entries':entries,
                'capture_plan_sha256':digest(OUT/'capture_plan.json'),'physics_steps':steps})
            prepared = prepare_pairs(path)
            write_json(OUT/f'data_validation_{amplitude:.2f}.json',prepared.report)
        progress('completed')
    except Exception as exc:
        progress('stopped',error=str(exc))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','collect'))
    args = parser.parse_args()
    prepare() if args.stage=='prepare' else collect()
