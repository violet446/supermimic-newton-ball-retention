"""Prepare and collect the frozen initial-state x action-history pilot."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from check_s3_actions import ROOT,digest,write_json
from collect_s3_refit import preflight
from ball_retention.history_capture import DESIGN,validate_history_case
from ball_retention.resources import snapshot,admission,directory_bytes
from ball_retention.guard import process_memory
from ball_retention.pair_collection import _initial_key,prepare_collection
from ball_retention.pair_data import prepare_pairs
sys.path.insert(0,str(ROOT))
from diagnostics.resource_probe.runner import stop_owned_tree,storage_admission

OUT=ROOT/'runs/s3_history_coverage_20261113'


def build_case(design,case,base):
    import numpy as np
    group=next(g for g in design['groups'] if g['group_id']==case['group_id'])
    cfg=copy.deepcopy(base);cfg['seed']=case['seed']
    cfg['initial']['ball_center_world_m']=(np.array(base['initial']['ball_center_world_m'])+group['ball_offset_world_m']).tolist()
    cfg['initial']['ball_velocity_world_m_s']=design['base_initial_velocity_world_m_s']
    cfg['history_capture']={**case,'design_sha256':digest(ROOT/DESIGN)}
    cfg['action_pairs']={'scenario':'route','master_seed':design['master_seed'],
        'schedule':design['histories'][case['history_id']],'snapshot_ticks':design['snapshot_ticks'],
        'horizon':design['horizon'],'amplitudes':design['amplitudes'],'replay_as_noise':True}
    limits=design['next_capture_budget']
    cfg['budget'].update({k:limits[k] for k in ('max_process_rss_gib','max_process_commit_gib','max_gpu_used_gib',
        'min_start_ram_gib','min_start_gpu_free_gib','min_disk_free_gib')})
    cfg['budget'].update(wall_s=limits['max_worker_wall_s'],max_cases=8,max_physics_steps=limits['max_physics_substeps'],
        max_trace_steps=design['reserved_substeps_per_episode'])
    # max_output_mib caps this capture directory in collect(). The inherited
    # output+shared-cache guard remains separate; old Warp caches are not new
    # capture output and must not be charged against the 512MiB capture limit.
    validate_history_case(ROOT,cfg)
    return cfg


def verify_design():
    design=json.loads((ROOT/DESIGN).read_text())
    for path,expected in design['inputs_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('design input changed: '+path)
    if (len(design['episodes'])!=8 or len(design['groups'])!=4 or design['reserved_substeps_total']!=70656
        or sum(design['reserved_substeps_per_episode'] for _ in design['episodes'])!=70656):
        raise ValueError('unexpected frozen capture size')
    return design


def prepare():
    import numpy as np
    from ball_retention.geometry import load_tool,sphere_clearance
    from ball_retention.robot import Robot
    OUT.mkdir(exist_ok=False);started=time.monotonic();reports={}
    try:
        design=verify_design();base=json.loads((ROOT/'configs/s1_online_episode.json').read_text())
        # Prior collection validates identities; anchor configs include test
        # identity metadata, never test trajectories or metrics.
        prepare_collection(ROOT,ROOT/'configs/s3_pair_collection.json')
        spec=json.loads((ROOT/'configs/s3_pair_collection.json').read_text())
        previous={}
        for source in [spec['anchor_dataset']]+spec['sources']:
            path=ROOT/source['manifest'];manifest=json.loads(path.read_text())
            for entry in manifest['entries']:
                prov=entry.get('provenance',entry)
                config=path.parent/prov.get('config',prov.get('config_path'))
                expected=prov.get('config_sha256')
                if digest(config)!=expected:raise ValueError('prior config changed')
                previous[_initial_key(json.loads(config.read_text()))]=entry['group_id']
        robot=Robot(ROOT/'assets/ec66_description.urdf');tool,_,_=robot.fk(base['initial']['joint_q_rad'])
        mesh=load_tool(ROOT/'assets/shovel.STL');cases=[];configs={};identities={}
        for case in design['episodes']:
            cfg=build_case(design,case,base);key=_initial_key(cfg)
            if key in previous:raise ValueError('new initial duplicates prior group')
            if identities.setdefault(key,(case['group_id'],case['split']))!=(case['group_id'],case['split']):
                raise ValueError('physical initial has multiple groups or splits')
            p=np.asarray(cfg['initial']['ball_center_world_m']);local=tool[:3,:3].T@(p-tool[:3,3])
            clearance=sphere_clearance(mesh,local,cfg['ball']['radius_m'])
            if clearance<.0001 or local[0]<.22 or p[2]-cfg['ball']['radius_m']<=cfg['physics']['floor_z_m']:
                raise ValueError('new initial geometry rejected: '+case['episode_id'])
            path=OUT/(case['episode_id']+'_config.json');write_json(path,cfg)
            cases.append({**case,'config_path':path.name,'config_sha256':digest(path),
                'snapshot_ticks':design['snapshot_ticks'],'reserved_steps':design['reserved_substeps_per_episode'],
                'initial_clearance_m':clearance})
            configs.setdefault(case['history_id'],cfg)
        if len(identities)!=4:raise ValueError('expected four physical groups')
        sources=[Path(__file__),ROOT/'scripts/collect_s3_refit.py']+list((ROOT/'src/ball_retention').glob('*.py'))+list((ROOT/'diagnostics/resource_probe').glob('*.py'))
        plan={'design_path':DESIGN,'design_sha256':digest(ROOT/DESIGN),'cases':cases,
            'source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},
            'resource_limits':design['next_capture_budget'],'training_updates':0,
            'storage_scopes':{'capture_directory_cap_mib':design['next_capture_budget']['max_output_mib'],
                'worker_output_plus_shared_cache_and_temp_cap_gib':base['budget']['max_output_cache_gib']}}
        write_json(OUT/'capture_plan.json',plan)
        for history,cfg in configs.items():
            if not admission(snapshot(ROOT),cfg['budget'])['passed']:raise RuntimeError('resource admission before preflight failed')
            storage=storage_admission(ROOT,OUT,cfg['budget'])
            if not storage['passed']:raise RuntimeError('storage admission before preflight failed')
            print('Preflight '+history,flush=True)
            reports[history]=preflight(cfg,1/60,16)
            memory=process_memory(os.getpid())
            if time.monotonic()-started>600 or memory['rss_bytes']>1024**3 or memory['private_commit_bytes']>1.5*1024**3:
                raise RuntimeError('preflight CPU budget exceeded')
            write_json(OUT/'preflight_progress.json',reports)
        write_json(OUT/'preflight.json',{'passed':True,'motion_checks':reports,'cases':cases,
            'scope':'two shared prescribed histories; no physics simulated','wall_s':time.monotonic()-started,
            'storage_admission':storage})
        write_json(OUT/'evaluation_plan.json',{'model_path':'runs/s3_physical_scales_20261113/full_training.npz',
            'model_sha256':design['inputs_sha256']['runs/s3_physical_scales_20261113/full_training.npz'],
            'leads':[8,16,32],'strong_noise_multiplier':10,'direction_cosine_min':.5,
            'wall_s':120,'rss_gib':1,'commit_gib':1.5,'training_updates':0,'checkpoint_selection':False,
            'groups_and_histories_reported_separately':True})
        print('Prepared 8 episodes; both motion histories passed',flush=True)
    except BaseException as exc:
        write_json(OUT/'preflight_rejected.json',{'passed':False,'error':str(exc),'wall_s':time.monotonic()-started,'completed_motion_checks':reports})
        raise


def collect():
    design=verify_design();plan=json.loads((OUT/'capture_plan.json').read_text());limits=design['next_capture_budget']
    if (OUT/'capture_progress.json').exists():raise ValueError('capture already attempted; no automatic retry')
    if not json.loads((OUT/'preflight.json').read_text())['passed']:raise ValueError('preflight required')
    if digest(ROOT/DESIGN)!=plan['design_sha256']:raise ValueError('design changed')
    for path,expected in plan['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('source changed since preflight: '+path)
    started=time.monotonic();attempts=[];entries=[];steps=0
    def progress(status,**extra):
        write_json(OUT/'capture_progress.json',{'status':status,'attempts':attempts,'physics_steps':steps,
            'wall_s':time.monotonic()-started,**extra})
    try:
        for case in plan['cases']:
            path=OUT/case['config_path']
            if digest(path)!=case['config_sha256']:raise ValueError('case config changed')
            cfg=json.loads(path.read_text());validate_history_case(ROOT,cfg)
            if (steps+case['reserved_steps']>limits['max_physics_substeps']
                or time.monotonic()-started+limits['max_worker_wall_s']>limits['max_wall_s']):
                raise RuntimeError('insufficient remaining capture budget')
            name=case['episode_id'];folder=OUT/name;progress('collecting',active_episode=name)
            print('Collecting '+name,flush=True)
            command=[sys.executable,'-B',str(ROOT/'diagnostics/resource_probe/runner.py'),'--mode','s3-pairs',
                '--config',str(path),'--output',str(folder),'--steps',str(case['reserved_steps'])]
            with (OUT/(name+'_launcher.log')).open('x',encoding='utf-8') as log:
                proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    while proc.poll() is None:
                        if time.monotonic()-started>limits['max_wall_s'] or directory_bytes(OUT)>limits['max_output_mib']*1024**2:
                            raise RuntimeError('capture aggregate time/output budget exceeded')
                        time.sleep(1)
                finally:
                    if proc.poll() is None:stop_owned_tree(proc);proc.wait(timeout=15)
            result=json.loads((folder/'result.json').read_text());worker=result.get('worker') or {}
            steps+=worker.get('physics_steps',0)
            attempts.append({'episode_id':name,'status':result['status'],'physics_steps':worker.get('physics_steps',0),
                'wall_s':result.get('wall_s'),'reason':worker.get('error') or result.get('stop_reason')})
            progress('collecting',active_episode=name)
            if proc.returncode or result['status']!='completed' or worker.get('status')!='action_pairs_completed':
                raise RuntimeError('worker stopped or failed: '+name)
            if result['config_sha256']!=case['config_sha256']:raise ValueError('worker config mismatch')
            trace=folder/worker['data_file'];rows=[json.loads(line) for line in trace.read_text().splitlines()]
            if [r['snapshot_tick'] for r in rows]!=design['snapshot_ticks']:raise ValueError('snapshot set mismatch')
            for row in rows:
                if len(row['branches'])!=6:raise ValueError('branch count mismatch')
                for branch in row['branches']:
                    if branch['terminated'] or branch['truncated'] or branch['substeps']!=[16]*32:
                        raise ValueError('incomplete branch; frozen scope requires full captures')
            entries.append({**case,'path':trace.relative_to(OUT).as_posix(),'sha256':digest(trace)})
            print('Completed '+name,flush=True)
        manifest=OUT/'capture_manifest.json'
        write_json(manifest,{'schema':'s3-long-pairs-v1','horizon':32,'amplitude':.2,'dt_s':1/960,'control_substeps':16,
            'entries':entries,'capture_plan_sha256':digest(OUT/'capture_plan.json'),'physics_steps':steps})
        prepared=prepare_pairs(manifest);write_json(OUT/'data_validation.json',prepared.report)
        spec=json.loads((ROOT/'configs/s3_pair_collection.json').read_text())
        spec['sources'].append({'cohort':'history','manifest':manifest.relative_to(ROOT).as_posix(),'sha256':digest(manifest)})
        spec_path=OUT/'merged_collection.json';write_json(spec_path,spec)
        merged=prepare_collection(ROOT,spec_path);write_json(OUT/'merged_validation.json',merged.report)
        progress('completed')
    except BaseException as exc:
        progress('stopped',error=str(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','collect'))
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output.resolve();OUT.relative_to(ROOT/'runs')
    prepare() if args.stage=='prepare' else collect()
