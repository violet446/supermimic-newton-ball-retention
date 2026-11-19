"""Freeze, run and verify four complete-route Newton baseline episodes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

from check_s3_deterministic_archive_import import ROOT, backend_preflight, digest, write_json
from collect_s3_small_action_response import monitor, storage_admission
import numpy as np

from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.controller import OnlineTiltController
from ball_retention.dataset import load_episode
from ball_retention.geometry import load_tool, sphere_clearance, support_point
from ball_retention.guard import process_memory
from ball_retention.pair_collection import _initial_key
from ball_retention.prescribed import PrescribedPath
from ball_retention.resources import admission, directory_bytes, snapshot
from ball_retention.robot import Robot
from ball_retention.support import CarryingRegion


BASE = ROOT/'runs/s3_closed_loop_baseline_20261118'
BROAD = 'runs/s3_broad_training_20261118'
BROAD_SHA = '6696d6a2b3193444f0791693488f10f93e1f4bd515151ccee0ac30dcbc6b87ca'
TARGET_SOURCE = 'runs/s1_episode_20261112/summary.json'
TARGET_SHA = '6a0ebec48fce92d8924a53cd4c925419a60c3d81d3032fd67b0901bbf4e80e99'
GROUPS = ('broad_train_72_center', 'broad_train_72_right')
ARMS = ('nominal', 'feedback')
HORIZON, SUBSTEPS, WARMUP = 240, 16, 60
TOTAL_STEPS = len(GROUPS)*len(ARMS)*HORIZON*SUBSTEPS
LIMITS = {'max_wall_s':900, 'max_new_output_mib':128}
RUNTIME = {'newton':'1.6.0','warp':'1.17.0','cuda_arch':89,
           'collision_pipeline_deterministic':True,'xpbd_deterministic_mode':'RUN_TO_RUN'}
SCOPE = {'training_updates':0,'test_episode_files_read':False,'main_model_replaced':False,
         'actor_critic_trained':False,'hardware_control':False}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def read(relative_path, sha):
    return json.loads(_verified_bytes(ROOT,relative_path,sha))


def check_stage_budget(out, budget, start):
    mem = process_memory(os.getpid())
    if (time.monotonic()-start >= budget['wall_s']
            or mem['peak_rss_bytes'] > budget['max_process_rss_gib']*1024**3
            or mem['peak_private_commit_bytes'] > budget['max_process_commit_gib']*1024**3
            or directory_bytes(out) > budget['max_new_output_mib']*1024**2):
        raise RuntimeError('local time, memory or output budget exceeded')
    return mem


def controller_check(cfg, robot, anchor, cap, budget, start):
    dt = cfg['physics']['dt_s']
    baseline = PrescribedPath(ROOT/cfg['prescribed']['joint_path'],cfg,moving=True)
    controller = OnlineTiltController(robot,cfg,anchor,baseline,ROOT/'assets')
    initial = controller.snapshot()
    corners = [(a*cap,b*cap) for a in (-1,1) for b in (-1,1)]
    requests = {'nominal':[[0.,0.]]*HORIZON}
    for i,corner in enumerate(corners):
        requests[f'corner_{i+1}'] = [[0.,0.]]*WARMUP+[corner]*(HORIZON-WARMUP)
    requests['switching_corners'] = [[0.,0.]]*WARMUP+[corners[(i//12)%4] for i in range(HORIZON-WARMUP)]
    record = {}
    for name,sequence in requests.items():
        controller.restore(initial)
        max_qd = max_qdd = 0.
        for tick,request in enumerate(sequence):
            commands = controller.plan(tick*SUBSTEPS*dt,request,SUBSTEPS)
            for command in commands:
                np.testing.assert_array_equal(command.clipped_action,request)
                controller.commit(command)
                max_qd = max(max_qd,float(np.max(np.abs(command.qd))))
                max_qdd = max(max_qdd,float(np.max(np.abs(command.qdd))))
            if (tick+1)%24 == 0:
                check_stage_budget(BASE,budget,start)
        np.testing.assert_allclose(controller.time_s,4.,rtol=0.,atol=1e-12)
        record[name] = {'steps':len(sequence),'max_joint_speed_rad_s':max_qd,'max_joint_acceleration_rad_s2':max_qdd}
    return record


def preflight():
    from s3_closed_loop_feedback import preflight as feedback_preflight

    began = time.monotonic()
    BASE.mkdir(exist_ok=False)
    budget = {'wall_s':120,'max_process_rss_gib':1.,'max_process_commit_gib':1.5,'max_new_output_mib':32.}
    try:
        archive = read(f'{BROAD}/validation.json',BROAD_SHA)
        manifest_path = f'{BROAD}/capture_manifest.json'
        manifest_sha = archive['artifacts_sha256']['capture_manifest.json']
        if (archive['status'] != 'completed' or archive['pairs'] != 36 or archive['worker_starts'] != 7
                or archive['test_episode_files_read']):
            raise ValueError('the training capture was not accepted')
        manifest = read(manifest_path,manifest_sha)
        selected = {entry['group_id']:entry for entry in manifest['entries'] if entry['group_id'] in GROUPS}
        equilibrium = read(TARGET_SOURCE,TARGET_SHA)['cases']['zero']
        if not equilibrium['final_task']['success'] or equilibrium['control_steps'] != HORIZON:
            raise ValueError('feedback target source did not pass its full route')
        p0 = equilibrium['final_task']['observation_t']['position']
        feedback = {'target_xy_m':p0[:2],'position_gain_per_m':8.,'velocity_gain_s_per_m':3.,'correction_cap':.12}
        sources = {f'{BROAD}/validation.json':BROAD_SHA,manifest_path:manifest_sha,TARGET_SOURCE:TARGET_SHA}
        sources.update({relative(p):digest(p) for p in [Path(__file__).resolve(),ROOT/'requirements.lock',
                    *sorted((ROOT/'src/ball_retention').glob('*.py')),
                    *(ROOT/'scripts'/name for name in ('s3_closed_loop_feedback.py','s3_closed_loop_episode.py',
                        'check_s3_actions.py','check_s3_deterministic_archive_import.py','collect_s3_small_action_response.py')),
                    *(ROOT/'diagnostics/resource_probe'/name for name in ('native_repeatability.py','runner.py'))]})
        asset_path = ROOT/'assets/manifest.json'
        sources[relative(asset_path)] = digest(asset_path)
        for asset in read(relative(asset_path),sources[relative(asset_path)])['assets']:
            _verified_bytes(ROOT,asset['local_path'],asset['sha256'])
            sources[asset['local_path']] = asset['sha256']
        mesh = load_tool(ROOT/'assets/shovel.STL')
        robot = Robot(ROOT/'assets/ec66_description.urdf')
        cases,geometry = {},{}
        common = None
        for group in GROUPS:
            entry = selected[group]
            if entry['split'] != 'train' or entry['snapshot_tick'] != 72:
                raise ValueError(f'{group} does not have the intended training provenance')
            path,sha = entry['source_config'],entry['source_config_sha256']
            cfg = read(path,sha)
            if (_initial_key(cfg) != entry['physical_initial_sha256']
                    or cfg['initial']['ball_velocity_world_m_s'] != [0.,0.,0.]
                    or cfg['history_capture']['split'] != 'train' or cfg['seed'] != entry['seed']):
                raise ValueError('the frozen training initial state changed')
            settings = {k:cfg[k] for k in ('ball','reference','physics','robot','action','prescribed','episode','budget')}
            settings['budget'] = {k:v for k,v in settings['budget'].items() if k not in ('max_physics_steps','max_trace_steps')}
            initial = {k:v for k,v in cfg['initial'].items() if k not in ('ball_center_world_m','ball_velocity_world_m_s')}
            if common is None:
                common = (settings,initial)
            elif common != (settings,initial):
                raise ValueError('non-ball settings differ between the two initial groups')
            ref = cfg['reference']
            if (ref['start_s'],ref['duration_s'],ref['hold_end_s'],ref['displacement_world_m'],
                    cfg['episode']['horizon_s'],cfg['physics']['control_hz'],cfg['physics']['dt_s']) != (
                    1.,2.,1.,[.08,0.,0.],4.,60,1/960):
                raise ValueError('the intended route or physics clock changed')
            state = cfg['initial']
            pose,_,_ = robot.fk(state['joint_q_rad'])
            ball = np.asarray(state['ball_center_world_m'])
            local = pose[:3,:3].T@(ball-pose[:3,3])
            clearance = float(sphere_clearance(mesh,local,cfg['ball']['radius_m']))
            carrying = CarryingRegion(mesh,state['tool_rotation_world'],cfg['ball']['radius_m'])
            if (clearance <= 0. or local[0] < carrying.bowl_start_x_m
                    or ball[2]-cfg['ball']['radius_m'] <= cfg['physics']['floor_z_m']
                    or not carrying.measure(p0)['carrying_candidate']):
                raise ValueError(f'{group} initial geometry or target is outside the admitted range')
            geometry[group] = {'physical_initial_sha256':_initial_key(cfg),'seed':cfg['seed'],
                               'center_in_tool_m':local.tolist(),'clearance_m':clearance,
                               'center_world_m':ball.tolist(),'feedback_target_supported':True}
            sources[path] = sha
            for arm in ARMS:
                name = f'{group}_{arm}'
                cases[name] = {'group':group,'arm':arm,'source_config':path,'sha256':sha}
        if len({_['physical_initial_sha256'] for _ in geometry.values()}) != len(GROUPS):
            raise ValueError('matched controllers must run on two distinct initial groups')
        route_path = cfg['prescribed']['joint_path']
        sources[route_path] = digest(ROOT/route_path)
        anchor = support_point(mesh,cfg['initial']['anchor_xy_tool_m'])[0]
        directions = feedback_preflight(np.asarray(cfg['initial']['tool_rotation_world']),feedback)
        kinematics = controller_check(cfg,robot,anchor,feedback['correction_cap'],budget,began)
        backend = backend_preflight({'runtime':RUNTIME})
        if int(backend['device']['native_sm']) != RUNTIME['cuda_arch'] or {'newton','warp'}.intersection(sys.modules):
            raise ValueError('unexpected backend or simulator imported in static preflight')
        mem = check_stage_budget(BASE,budget,began)
        write_json(BASE/'preflight.json',{'passed':True,'cases':cases,'initial_geometry':geometry,'feedback':feedback,
                   'warmup_steps':WARMUP,'route_time_s':4.,'physical_substeps_reserved':TOTAL_STEPS,
                   'trajectories_reserved':4,'backend':backend,'kinematics':kinematics,'feedback_signs':directions,
                   'static_budget':budget,'limits':LIMITS,'input_sha256':sources,'physics_steps':0,
                   'wall_s':time.monotonic()-began,'memory':mem,**SCOPE})
    except BaseException as error:
        write_json(BASE/'preflight_stopped.json',{'reason':str(error),'physics_steps':0,**SCOPE})
        raise


def load_plan(path, sha):
    if path != BASE/'preflight.json':
        raise ValueError('unexpected static plan location')
    plan = read(relative(path),sha)
    if (plan['passed'] is not True or plan['physics_steps'] != 0 or plan['physical_substeps_reserved'] != TOTAL_STEPS
            or plan['limits'] != LIMITS or plan['warmup_steps'] != WARMUP
            or list(plan['cases']) != [f'{g}_{a}' for g in GROUPS for a in ARMS]
            or any(plan[key] != val for key,val in SCOPE.items())):
        raise ValueError('unexpected frozen capture scope')
    for name,expected in plan['input_sha256'].items():
        _verified_bytes(ROOT,name,expected)
    return plan


def case_config(case):
    cfg = read(case['source_config'],case['sha256'])
    cfg['budget'].update(wall_s=LIMITS['max_wall_s'],max_cases=1,max_physics_steps=HORIZON*SUBSTEPS,
                         max_trace_steps=HORIZON*SUBSTEPS,max_new_output_mib=LIMITS['max_new_output_mib'])
    return cfg


def admit(out, cfg):
    budget = cfg['budget']
    started = time.monotonic()
    sample = snapshot(ROOT)
    while sample['available_ram_bytes'] < budget['min_start_ram_gib']*1024**3 and time.monotonic()-started < 45.:
        time.sleep(1)
        sample = snapshot(ROOT)
    hardware = admission(sample,budget)
    storage = storage_admission(ROOT,BASE,budget)
    environment = directory_bytes(ROOT/'.venv')
    write_json(out/'admission.json',{'sample':sample,'hardware':hardware,'storage':storage,
                                     'environment_bytes':environment,'startup_wait_s':time.monotonic()-started})
    if not hardware['passed'] or not storage['passed'] or environment > budget['max_environment_gib']*1024**3:
        raise RuntimeError('Newton startup resource admission failed')


def worker(out, plan):
    from s3_closed_loop_episode import run_episode
    from s3_closed_loop_feedback import feedback_action

    began = time.monotonic()
    if out.parent.parent != BASE or out.name != 'worker':
        raise ValueError('unexpected worker field or output path')
    name = out.parent.name
    case = plan['cases'][name]
    cfg = case_config(case)
    state = {'status':'starting','physics_steps':0,'worker_pid':os.getpid(),'case':case}

    def stage(label):
        state.update(stage=label,wall_s=time.monotonic()-began)
        write_json(out/'worker_result.json',state)
        return check_stage_budget(out,cfg['budget'],began)

    def action(tick, observation):
        if case['arm'] == 'nominal' or tick < WARMUP:
            return [0.,0.]
        return feedback_action(observation,plan['feedback'])

    try:
        stage('resource_recheck')
        if not admission(snapshot(ROOT),cfg['budget'])['passed']:
            raise RuntimeError('worker resource recheck failed before importing Newton')
        run_episode(ROOT,cfg,name,action,out,state,stage)
        state['status'] = 'completed'
        stage('complete')
    except BaseException as error:
        write_json(out/'worker_result.json',{**state,'status':'stopped','error':str(error)})
        raise


def verify_episode(cfg, state, path, name, cap):
    episodes = load_episode(path,cfg['physics']['dt_s'],SUBSTEPS,name)
    last = episodes[-1]
    observed = sum(len(row.executed_joint_q) for row in episodes)
    if (observed != state['physics_steps'] or observed > HORIZON*SUBSTEPS
            or (last.terminated == last.truncated) or (last.truncated and len(episodes) != HORIZON)
            or any(row.invalid_reason is not None for row in episodes)
            or any(any(abs(a)>cap for a in row.requested_action) for row in episodes)):
        raise ValueError(f'{name} invalid, incomplete or outside frozen physical trace scope')
    return episodes


def capture(plan, path, sha):
    from s3_closed_loop_feedback import feedback_action

    began = time.monotonic()
    completed,rows = {},{}
    current = None
    total = 0
    started = 0
    try:
        if backend_preflight({'runtime':RUNTIME}) != plan['backend'] or {'newton','warp'}.intersection(sys.modules):
            raise ValueError('launcher environment differs from preflight')
        for name,case in plan['cases'].items():
            current = BASE/name
            current.mkdir(exist_ok=False)
            cfg = case_config(case)
            admit(current,cfg)
            started += 1
            with patch.dict(monitor.__globals__,{'__file__':__file__,'directory_bytes':lambda _:directory_bytes(BASE),
                            'storage_admission':lambda root,_,budget:storage_admission(root,BASE,budget)}):
                state = monitor(current,path,sha,cfg['budget'],LIMITS,began)
            if state['case'] != case:
                raise ValueError('worker results do not identify their matched arm')
            cap = plan['feedback']['correction_cap']
            recorded = current/'worker/transitions.jsonl'
            episodes = verify_episode(cfg,state,recorded,name,cap)
            for tick,row in enumerate(episodes):
                request = ([0.,0.] if case['arm']=='nominal' or tick<WARMUP
                           else feedback_action(row.state_t,plan['feedback']))
                if row.requested_action != request:
                    raise ValueError(f'{name} tick {tick} does not replay its declared feedback law')
            completed[name] = state
            rows[name] = episodes
            total += state['physics_steps']
            if total > TOTAL_STEPS:
                raise ValueError('cumulative substep allowance exceeded')
            print(f'{name}: {len(episodes)} controls, success={episodes[-1].task["success"]}',flush=True)
        comparisons = {}
        for group in GROUPS:
            a,b = (rows[f'{group}_{arm}'] for arm in ARMS)
            n = min(WARMUP,len(a),len(b))
            for i in range(n):
                x,y = (episode[i].to_dict() for episode in (a,b))
                for row in (x,y):
                    row.pop('episode_id')
                if x != y:
                    raise ValueError(f'{group} matched zero-action prefix diverged at control {i}')
            if (min(len(a),len(b)) < WARMUP) and len(a) != len(b):
                raise ValueError(f'{group} matched warm-up ended asymmetrically')
            comparisons[group] = {'shared_prefix_verified_controls':n,
               'nominal':{'terminal':a[-1].task,'elapsed_sim_s':a[-1].next_state_before_reset.time_s},
               'feedback':{'terminal':b[-1].task,'elapsed_sim_s':b[-1].next_state_before_reset.time_s}}
        load_plan(path,sha)
        if time.monotonic()-began >= LIMITS['max_wall_s'] or directory_bytes(BASE) > LIMITS['max_new_output_mib']*1024**2:
            raise RuntimeError('aggregate time or output budget reached')
        result = {'status':'completed','preflight_sha256':sha,'physics_steps':total,
                  'worker_starts':started,'episodes':completed,'paired':comparisons,'wall_s':time.monotonic()-began,**SCOPE}
        write_json(BASE/'comparison.json',result)
        names = [p.relative_to(BASE).as_posix() for p in BASE.rglob('*') if p.is_file()]
        write_json(BASE/'validation.json',{'status':'completed','preflight_sha256':sha,'physics_steps':total,
                   'worker_starts':started,'case_count':len(completed),'wall_s':time.monotonic()-began,
                   'artifacts_sha256':{name:digest(BASE/name) for name in sorted(names)},**SCOPE})
    except BaseException as error:
        f = current/'worker/worker_result.json' if current else None
        state = json.loads(f.read_bytes()) if f and f.is_file() else {}
        completed_steps = sum(v['physics_steps'] for v in completed.values())
        write_json(BASE/'stopped.json',{'status':'stopped','reason':str(error),'worker_starts':started,
                   'completed_cases':list(completed),'completed_physics_steps':completed_steps,
                   'active_case':current.name if current else None,
                   'active_last_checkpoint':state,'wall_s':time.monotonic()-began,**SCOPE})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','capture','worker'))
    parser.add_argument('--plan',type=Path,default=BASE/'preflight.json')
    parser.add_argument('--sha256')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.phase == 'preflight':
        preflight()
    else:
        if not args.sha256:
            parser.error('frozen preflight hash required')
        path = args.plan.resolve()
        plan = load_plan(path,args.sha256)
        if args.phase == 'capture':
            capture(plan,path,args.sha256)
        else:
            worker(args.output.resolve(),plan)
