"""Bounded CPU-only geometry/IK checks. Never imports or runs Newton."""
from pathlib import Path
import argparse, hashlib, importlib.metadata as md, json, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.spatial.transform import Rotation
from ball_retention.geometry import load_tool,support_point,sphere_clearance
from ball_retention.robot import Robot
from ball_retention.task import tool_pose_for_anchor
from ball_retention.resources import snapshot,admission,directory_bytes

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='runs/s1_20261111/preflight.json')
    args=parser.parse_args(); started=time.monotonic()
    cfg=json.loads((ROOT/'configs/s1_smoke.json').read_text())
    manifest=json.loads((ROOT/'assets/manifest.json').read_text())
    for row in manifest['assets']:
        if hashlib.sha256((ROOT/row['local_path']).read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('asset checksum mismatch')
    resource=snapshot(ROOT)
    report={'kind':'static CPU geometry and initialization IK; no Newton import/simulation',
            'seed':cfg['seed'],'effective_config':cfg,'resources':resource,'physics_resource_admission':admission(resource,cfg['budget']),
            'python':sys.executable,'packages':{name:md.version(name) for name in ('newton','warp-lang','numpy','scipy','trimesh')},
            'config_sha256':hashlib.sha256((ROOT/'configs/s1_smoke.json').read_bytes()).hexdigest(),
            'lock_sha256':hashlib.sha256((ROOT/'requirements.lock').read_bytes()).hexdigest()}
    tool=load_tool(ROOT/'assets/shovel.STL')
    point,normal,hits=support_point(tool,cfg['initial']['anchor_xy_tool_m'])
    center=point+normal*(cfg['ball']['radius_m']+cfg['initial']['gap_m'])
    clearance=sphere_clearance(tool,center,cfg['ball']['radius_m'])
    report['tool']={'watertight':bool(tool.is_watertight),'winding_consistent':bool(tool.is_winding_consistent),
                    'anchor_local_m':point.tolist(),'normal_local':normal.tolist(),'vertical_hits':hits,
                    'ball_center_local_m':center.tolist(),'sphere_clearance_m':clearance,
                    'initial_clearance_passed':clearance>=-cfg['admission']['max_initial_penetration_m']}
    # Place the local supporting normal along world +Z, with local forward in world XZ.
    forward=np.array([1.,0.,0.]); forward-=normal*np.dot(forward,normal); forward/=np.linalg.norm(forward)
    basis=np.column_stack((forward,np.cross(normal,forward),normal))
    target=tool_pose_for_anchor(cfg['initial']['anchor_world_m'],point,basis.T)
    robot=Robot(ROOT/'assets/ec66_description.urdf')
    # Deterministic new-task numerical seeds, no old trajectory or data dependency.
    rng=np.random.default_rng(cfg['seed']); seeds=[np.zeros(6)]+[rng.uniform(-np.pi,np.pi,6) for _ in range(7)]
    fits=[]
    for seed in seeds:
        if time.monotonic()-started>60: raise TimeoutError('60-second static preflight budget')
        fits.append(robot.ik(target,seed))
    report['initial_ik']={'target_tool_transform':target.tolist(),'candidates':fits,
                          'pose_solution_found':any(f['pose_passed'] for f in fits),
                          'self_collision_checked':False,'trajectory_checked':False,
                          'admitted_for_physics':False}
    report['sizes_bytes']={name:directory_bytes(ROOT/name) for name in ('.venv','.cache','.tmp','assets')}
    report['wall_s']=time.monotonic()-started
    report['s1_complete']=False
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'tool':report['tool'],'ik_pose_found':report['initial_ik']['pose_solution_found'],
                      'resources':report['physics_resource_admission'],'sizes_bytes':report['sizes_bytes'],'wall_s':report['wall_s']},indent=2))

if __name__=='__main__': main()
