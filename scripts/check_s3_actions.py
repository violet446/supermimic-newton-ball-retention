"""Preflight action derivatives and collect bounded matched Newton branches."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def schedule_actions(schedule, start, stop):
    events = iter(schedule)
    event = next(events)
    action = None
    rows = []
    for tick in range(stop):
        if event is not None and tick == event['start_step']:
            action = event['action']
            event = next(events, None)
        if tick >= start:
            if action is None:
                raise ValueError('schedule needs a command at tick zero')
            rows.append(action)
    return rows


def ball_state(state):
    import numpy as np
    from ball_retention.task import ball_in_tool
    values = ball_in_tool(state.ball, state.tool)
    return np.r_[values['position'], values['velocity'], values['ball_omega']]


def preflight(cfg, source, model, mean, scale, dt, substeps, plan):
    import numpy as np
    from ball_retention.action_grad import action_sensitivity
    from ball_retention.controller import OnlineTiltController
    from ball_retention.dataset import load_episode
    from ball_retention.geometry import load_tool, support_point
    from ball_retention.model_data import known_rollout
    from ball_retention.prescribed import PrescribedPath
    from ball_retention.robot import Robot

    spec = cfg['action_pairs']
    horizon = spec['horizon']
    snapshots = spec['snapshot_ticks']
    schedule = np.asarray(schedule_actions(spec['schedule'], 0, max(snapshots)+horizon))
    rows = load_episode(source/'transitions.jsonl', cfg['physics']['dt_s'], substeps, source.name)
    robot = Robot(ROOT/'assets/ec66_description.urdf')
    mesh = load_tool(ROOT/'assets/shovel.STL')
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    path = PrescribedPath(ROOT/cfg['prescribed']['joint_path'], cfg, moving=True)
    controller = OnlineTiltController(robot, cfg, anchor, path, ROOT/'assets')
    results = []
    for tick in range(max(snapshots)+1):
        if tick in snapshots:
            saved = controller.snapshot()
            nominal = schedule[tick:tick+horizon]
            offsets = [np.zeros(2)]
            offsets += [np.eye(2)[axis]*sign*size for axis in range(2)
                        for size in spec['amplitudes'] for sign in (-1,1)]
            peak_speed = peak_accel = 0.
            for offset in offsets:
                controller.restore(saved)
                for k, action in enumerate(nominal+offset):
                    if np.max(np.abs(action)) >= 1:
                        raise ValueError('paired branch would hit request clipping')
                    commands = controller.plan((tick+k)*dt, action, substeps)
                    for command in commands:
                        peak_speed = max(peak_speed, float(np.max(np.abs(command.qd))))
                        peak_accel = max(peak_accel, float(np.max(np.abs(command.qdd))))
                        controller.commit(command)
            controller.restore(saved)

            state = rows[tick].state_t
            initial = ball_state(state)
            arguments = (state.time_s, state.tilt_rad, state.tilt_rate_rad_s)
            sensitivity = action_sensitivity(model, initial, *arguments, nominal, cfg, mean, scale)
            analytical = sensitivity['jacobian']

            def predict(actions):
                known = (known_rollout(*arguments, actions, cfg, dt)-mean)/scale
                return model.denormalize(model.rollout(model.normalize(initial)[None, :], known[None, ...])[0])

            eps = plan['model_difference_step']
            numeric = np.stack([(predict(nominal+direction*eps)-predict(nominal-direction*eps))/(2*eps)
                                for direction in np.eye(2)], axis=-1)
            error = float(np.max(np.abs(analytical-numeric)))
            forward_error = float(np.max(np.abs(sensitivity['prediction']-predict(nominal))))
            if not np.allclose(analytical, numeric, rtol=plan['model_relative_tolerance'],
                               atol=plan['model_absolute_tolerance']) or forward_error > 1e-10:
                raise ValueError(f'chain rule / central difference mismatch at {spec["scenario"]}:{tick}')
            results.append({'scenario':spec['scenario'], 'tick':tick, 'cpu_branches_checked':len(offsets),
                            'max_joint_speed_rad_s':peak_speed, 'max_joint_acceleration_rad_s2':peak_accel,
                            'model_gradient_max_absolute_difference':error,
                            'model_forward_max_absolute_difference':forward_error})
        if tick < max(snapshots):
            for command in controller.plan(tick*dt, schedule[tick], substeps):
                controller.commit(command)
    return results


def run():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    from ball_retention.geometry import load_tool
    from ball_retention.robot import Robot
    from ball_retention.world_model import ResidualMLP

    out = ROOT/'runs/s3_action_pairs_20261112'
    out.mkdir(exist_ok=False)
    model_dir = ROOT/'runs/s2_world_model_20261112'
    checkpoint = model_dir/'world_model.npz'
    model_report = json.loads((model_dir/'summary.json').read_text())
    if digest(checkpoint) != model_report['checkpoint_sha256']:
        raise ValueError('S2 checkpoint digest changed')
    dataset = ROOT/'data/s2_pilot_20261112'
    manifest = json.loads((dataset/'manifest.json').read_text())
    if digest(dataset/'manifest.json') != json.loads((model_dir/'plan.json').read_text())['config']['dataset_manifest_sha256']:
        raise ValueError('S2 data split changed')
    substeps = manifest['control_substeps']
    dt = substeps*manifest['dt_s']
    with np.load(checkpoint, allow_pickle=False) as saved:
        model = ResidualMLP.from_state_dict(saved)
        mean, scale = saved['known_mean'], saved['known_scale']
    if model.dt_s != dt:
        raise ValueError('checkpoint and Newton control periods differ')
    route = json.loads((ROOT/'configs/s1_online_episode.json').read_text())
    tool, _, _ = Robot(ROOT/'assets/ec66_description.urdf').fk(route['initial']['joint_q_rad'])
    basin = tool[:3,:3].T@(np.array(route['initial']['ball_center_world_m'])-tool[:3,3])
    horizon, amplitudes, seed = 8, [.1,.2], 20261111
    max_steps, max_wall_s = 9000, 900
    plan = {'master_seed':seed, 'environments':1, 'max_worker_starts':2, 'max_physics_steps':max_steps,
            'max_wall_s':max_wall_s, 'horizon':horizon, 'leads':[1,4,8], 'amplitudes':amplitudes,
            'position_signal_floor_m':1e-6, 'noise_multiplier':5., 'basin_center_tool_xy_m':basin[:2].tolist(),
            'model_difference_step':1e-5, 'model_relative_tolerance':5e-6, 'model_absolute_tolerance':1e-8,
            'resource_limits':route['budget'] | {'wall_s':max_wall_s,'max_cases':2,'max_physics_steps':max_steps},
            'checkpoint':checkpoint.relative_to(ROOT).as_posix(),'checkpoint_sha256':digest(checkpoint),
            'scope':'local derivative response at chosen validation states; fixed perturbation over horizon',
            'test_split_used':False, 'cases':[]}
    cases = []
    for scenario, episode_id, ticks, wall in [('route','validation_lateral',[48,90,150],600),
                                             ('edge','validation_edge',[0],300)]:
        entry = next(e for e in manifest['entries'] if e['episode_id']==episode_id and e['split']=='validation')
        config_path = dataset/entry['provenance']['config']
        trace_path = dataset/entry['path']
        if digest(config_path)!=entry['provenance']['config_sha256'] or digest(trace_path)!=entry['sha256']:
            raise ValueError('source episode digest changed')
        cfg = json.loads(config_path.read_text())
        old_case = cfg.pop('collection')['case']
        cfg['action_pairs'] = {'master_seed':seed,'scenario':scenario,'snapshot_ticks':ticks,'horizon':horizon,
                               'amplitudes':amplitudes,'schedule':old_case['actions']}
        reserved = (max(ticks)+len(ticks)*(2+4*len(amplitudes))*horizon)*substeps
        cfg['budget'].update(wall_s=wall,max_cases=2,max_physics_steps=max_steps,max_trace_steps=reserved)
        new_config = out/(scenario+'_config.json')
        write_json(new_config,cfg)
        plan['cases'].append({'scenario':scenario,'source_episode':episode_id,'snapshot_ticks':ticks,
                              'config_sha256':digest(new_config),'reserved_steps':reserved})
        cases.append((scenario, cfg, trace_path.parent, new_config, reserved))
    if sum(case[-1] for case in cases)>max_steps:
        raise ValueError('reserved branches exceed round budget')
    write_json(out/'plan.json',plan)
    started = time.monotonic()
    preflight_rows = []
    for scenario,cfg,source,_,_ in cases:
        preflight_rows.extend(preflight(cfg,source,model,mean,scale,dt,substeps,plan))
        print(f'{scenario}: model derivative and CPU motion preflight passed',flush=True)
    write_json(out/'preflight.json',{'passed':True,'cases':preflight_rows,'wall_s':time.monotonic()-started})
    actual_steps = 0
    runs = []
    for scenario,_,_,config,reserved in cases:
        if actual_steps+reserved>max_steps or time.monotonic()-started>=max_wall_s:
            raise ValueError('round budget exhausted before next case')
        command = [sys.executable,'-B',str(ROOT/'diagnostics/resource_probe/runner.py'),'--mode','s3-pairs',
                   '--config',str(config),'--output',str(out/scenario),'--steps',str(reserved)]
        print(f'{scenario}: collecting paired Newton branches',flush=True)
        proc = subprocess.run(command,cwd=ROOT)
        result = json.loads((out/scenario/'result.json').read_text())
        worker = result.get('worker') or {}
        actual_steps += worker.get('physics_steps',0)
        runs.append({'scenario':scenario,'status':result['status'],'physics_steps':worker.get('physics_steps',0)})
        write_json(out/'collection.json',{'runs':runs,'physics_steps':actual_steps,'wall_s':time.monotonic()-started})
        if proc.returncode or result['status']!='completed':
            raise RuntimeError(f'paired capture stopped: {scenario} ({result["status"]})')
    if actual_steps>max_steps or time.monotonic()-started>max_wall_s:
        raise RuntimeError('round budget exceeded')


if __name__=='__main__':
    run()
