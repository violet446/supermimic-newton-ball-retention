"""Verify one frozen batch initial condition against the accepted full tool replay."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from ball_retention.controller import OnlineTiltController
from ball_retention.geometry import load_tool, sphere_clearance, support_point
from ball_retention.pair_collection import _initial_key
from ball_retention.prescribed import PrescribedPath
from ball_retention.robot import Robot
from check_s3_actions import schedule_actions
from preflight_s3_deterministic_action_response import action_sequences, verify_commands
from train_s3_local_velocity_increment import bound


PHASE = 'pitch_full_tool_batch_static'


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    scope = read(plan['batch_scope'])
    item, = [x for x in scope['cases'] if x['source_episode'] == plan['source_episode']]
    accepted = read(plan['reference_static_validation'])
    keys = ('reference_config', 'reference_commands')
    for key in keys:
        path = Path(plan[key])
        if accepted['artifacts_sha256'][path.name] != plan['files_sha256'][path.as_posix()]:
            raise ValueError('accepted reference artifact changed')
    if (scope['status'] != 'static_commands_pending' or plan['updates'] != 0
            or not accepted['exact_development_tool_commands'] or accepted['simulation_steps'] != 0
            or accepted['test_episode_files_read'] is not False):
        raise ValueError('unexpected batch or reference scope')
    reference, evidence = (read(plan[k]) for k in keys)
    cfg = read(item['source_config'])
    protocol = read(scope['tool_protocol'])
    inventory = [p for p in scope['files_sha256'] if p.endswith(('capture_manifest.json','capture_manifest_0.20.json'))]
    sources = [e for p in inventory for e in read(p)['entries'] if e['episode_id']==item['source_episode']]
    archived, = sources
    if (archived['split']!='train' or archived['group_id']!=item['training_group']
            or _initial_key(cfg)!=item['physical_initial_sha256'] or cfg['seed']!=item['seed']
            or item['training_group']==protocol['case']['group']
            or any(cfg[k]!=reference[k] for k in ('physics','ball','action','robot','reference','prescribed'))
            or any(cfg['initial'][k]!=reference['initial'][k] for k in
                   ('joint_q_rad','tool_rotation_world','anchor_world_m','anchor_xy_tool_m'))):
        raise ValueError('invalid independent training case or controller settings')
    cfg['action_pairs']['schedule']=reference['action_pairs']['schedule']
    mesh=load_tool(ROOT/'assets/shovel.STL')
    robot=Robot(ROOT/'assets/ec66_description.urdf')
    anchor=support_point(mesh,cfg['initial']['anchor_xy_tool_m'])[0]
    pose,_,_=robot.fk(cfg['initial']['joint_q_rad'])
    ball=np.asarray(cfg['initial']['ball_center_world_m'])
    local=pose[:3,:3].T@(ball-pose[:3,3])
    radius=cfg['ball']['radius_m']
    clearance=sphere_clearance(mesh,local,radius)
    if (not np.isclose(clearance,item['checked_initial_clearance_m'],rtol=0.,atol=1e-12)
            or clearance<=0 or local[0]<.22 or ball[2]-radius<=cfg['physics']['floor_z_m']):
        raise ValueError('training ball placement changed')
    path=PrescribedPath(ROOT/cfg['prescribed']['joint_path'],cfg,moving=True)
    controller=OnlineTiltController(robot,cfg,anchor,path,ROOT/'assets')
    tick,n,horizon=scope['reference_clock_tick'],scope['physics_substeps_per_control'],scope['horizon']
    if (tick,n,horizon)!=(168,16,32):
        raise ValueError('unexpected batch clocks')
    dt=cfg['physics']['dt_s']
    timeline=schedule_actions(cfg['action_pairs']['schedule'],0,tick+horizon)
    nominal=[]
    for i,request in enumerate(timeline):
        steps=controller.plan(i*n*dt,request,n)
        for cmd in steps:
            controller.commit(cmd)
            np.testing.assert_array_equal(request,cmd.clipped_action)
        command={'q':[x.q.tolist() for x in steps],'qd':[x.qd.tolist() for x in steps]}
        if i<tick and (request!=evidence['prefix_requested'][i] or command!=evidence['prefix_executed'][i]):
            raise ValueError('replayed full controller history differs from reference')
        if i>=tick:
            nominal.append(command)
        if i+1==tick:
            saved={'controller':controller.snapshot(),'physics':{'steps':tick*n}}
            if (saved['controller']!=evidence['snapshot_control_only']
                    or cmd.qdd.tolist()!=evidence['snapshot_joint_qdd_rad_s2']):
                raise ValueError('identical controller did not reach the reference snapshot')
        guard()
    _,requests=action_sequences(protocol,timeline[tick:])
    commands,peaks=verify_commands(ROOT,protocol,cfg,saved,requests,nominal,guard)
    for label in protocol['branch_order']:
        if (commands[label]!=evidence['branches'][label]['executed']
                or requests[label].tolist()!=evidence['branches'][label]['requested']):
            raise ValueError(f'signed executed commands differ: {label}')
    path=out/'candidate_config.json'
    write_json(path,cfg)
    identity={'episode_id':item['source_episode']+'_matched_full_tool','group_id':item['training_group'],
              'history_id':'matched_full_tool','split':'train','seed':item['seed'],
              'physical_initial_sha256':item['physical_initial_sha256'],
              'source_config':path.relative_to(ROOT).as_posix(),
              'source_config_sha256':sha256(path.read_bytes()).hexdigest()}
    write_json(out/'commands.json',{**evidence,'case':identity})
    write_json(out/'static.json',{'training_initial':identity,'clearance_m':float(clearance),
        'exact_prefix_controls':tick,'exact_branch_controls':len(peaks)*horizon,'branch_peaks':peaks})
    return {'static_branches':len(peaks),'exact_development_tool_commands':True,'new_ball_trajectories':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
