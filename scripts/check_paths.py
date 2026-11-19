"""Five bounded, continuous joint-spline paths; CPU kinematics, not physics."""
from pathlib import Path
import copy,json,sys,time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.interpolate import CubicSpline,PPoly
from scipy.spatial.transform import Rotation
from ball_retention.robot import Robot
from ball_retention.task import reference,tool_pose_for_anchor
from ball_retention.clearance import geometry_rows,pose_clearance,excluded_pairs
from search_reachability import Budget

def extrema(curve,order):
    low=[]; high=[]
    for j in range(6):
        p=PPoly(curve.c[:,:,j],curve.x).derivative(order)
        roots=p.derivative().roots(extrapolate=False)
        roots=roots[np.isfinite(roots)&(roots>=curve.x[0])&(roots<=curve.x[-1])]
        values=p(np.r_[curve.x,roots]); low.append(float(values.min())); high.append(float(values.max()))
    return np.array(low),np.array(high)

def target_at(t,cfg,point,base_R,axis,sign):
    # 0->5deg->0 over the two-second translation, zero endpoint rates/accelerations.
    phase=np.clip(t-1,0,2); u=phase if phase<=1 else 2-phase
    angle=np.deg2rad(5)*sign*(10*u**3-15*u**4+6*u**5)
    R=Rotation.from_euler(axis,angle).as_matrix()@base_R
    return tool_pose_for_anchor(reference(t,cfg)['anchor_position'],point,R)

def speed_bound(row,robot,qd_max,shared=0):
    """Point speed in common-ancestor frame; only downstream rotations contribute."""
    chain=[]
    for parent,child,T,axis in robot.joints:
        chain.append((T,axis))
        if child==row['link'] or (row['link']=='tool' and child=='flan'): break
    index=0; value=0.
    for k,(T,axis) in enumerate(chain):
        if axis is None: continue
        index+=1
        if index<=shared: continue
        radius=row['radius_m']+sum(np.linalg.norm(e[0][:3,3]) for e in chain[k+1:])
        value+=qd_max[index-1]*radius
    return float(value)

def continuous_clearance(curve,rows,robot,qd_max,budget):
    # Ancestor depths in this verified serial URDF chain; tool shares all six joints.
    depths={'base_link':0,**{f'link{i}':i for i in range(1,7)},'tool':6}
    pair_speeds={}
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if frozenset((a['name'],b['name'])) in excluded_pairs(): continue
            shared=min(depths[a['link']],depths[b['link']])
            pair_speeds[a['name']+' / '+b['name']]=speed_bound(a,robot,qd_max,shared)+speed_bound(b,robot,qd_max,shared)
    max_floor_speed=max(speed_bound(r,robot,qd_max) for r in rows)
    intervals=[(a,b) for a,b in zip(np.linspace(0,4,17)[:-1],np.linspace(0,4,17)[1:])]
    accepted=[]; failed=[]; evaluations=0
    while intervals:
        budget.check()
        a,b=intervals.pop(); mid=(a+b)/2
        check=pose_clearance(rows,robot,curve(mid)); evaluations+=1
        lower={k:v-pair_speeds[k]*(b-a)/2-1e-7 for k,v in check['pair_lower_bounds_m'].items()}
        floor=check['floor_clearance_m']-max_floor_speed*(b-a)/2-1e-7
        if min(lower.values())>=.002 and floor>=.02:
            accepted.append({'start_s':a,'end_s':b,'pair_lower_bound_m':min(lower.values()),'floor_lower_bound_m':floor})
        elif b-a<=1/480 or evaluations>=250:
            failed.append({'start_s':a,'end_s':b,'pair_lower_bounds_m':lower,'floor_lower_bound_m':floor})
            break
        else:
            intervals.extend(((a,mid),(mid,b)))
    return {'certified':not failed and not intervals,'evaluations':evaluations,'accepted_intervals':accepted,
            'failed_intervals':failed,'remaining_intervals':len(intervals),'pair_speed_bounds_m_s':pair_speeds,
            'method':'midpoint separating plane minus common-ancestor relative point-speed bound * half interval; 1e-7m numerical allowance',
            'excluded_pairs':[sorted(p) for p in excluded_pairs()]}

def main():
    out=ROOT/'runs/s1_reachability_20261111'; budget=Budget(540)
    search=json.loads((out/'search.json').read_text()); candidates=json.loads((out/'candidates.json').read_text())
    chosen=next(r for r in candidates['results'] if r['anchor_world_m'][0]==.65 and r['seed_index']==0)
    if not chosen['pose_clearance_certified']: raise RuntimeError('chosen candidate not certified')
    cfg=json.loads((ROOT/'configs/s1_smoke.json').read_text()); cfg=copy.deepcopy(cfg)
    cfg['schema_version']='ball-retention-s1-v0.2-candidate'
    cfg['initial']['anchor_world_m']=chosen['anchor_world_m']; cfg['initial']['joint_q_rad']=chosen['q']
    cfg['initial']['tool_rotation_world']=search['base_rotation']
    cfg['revision_reason']='v0.1 initial IK failed; anchor X moved from 0.45 to 0.65m with unchanged criteria; static candidate only'
    robot=Robot(ROOT/'assets/ec66_description.urdf'); rows=geometry_rows(ROOT/'assets')
    point=np.array(search['anchor_local_m']); base_R=np.array(search['base_rotation'])
    reports=[]; arrays={}
    for name,axis,sign in [('zero','x',0),('roll_plus','x',1),('roll_minus','x',-1),('pitch_plus','y',1),('pitch_minus','y',-1)]:
        q=np.array(chosen['q']); qs=[]; times=np.linspace(0,4,81)
        for t in times:
            budget.check(); fit=robot.ik(target_at(t,cfg,point,base_R,axis,sign),q)
            if not fit['pose_passed']: raise RuntimeError(f'{name} IK failed at {t}')
            q=np.array(fit['q']); qs.append(q)
        curve=CubicSpline(times,qs,bc_type=((1,np.zeros(6)),(1,np.zeros(6))))
        low,high=extrema(curve,0); vlow,vhigh=extrema(curve,1); alow,ahigh=extrema(curve,2)
        vmax=np.maximum(abs(vlow),abs(vhigh)); amax=np.maximum(abs(alow),abs(ahigh))
        limits=bool(np.all(low>=robot.low)&np.all(high<=robot.high)&np.all(vmax<=.5)&np.all(amax<=1.))
        samples=np.linspace(0,4,1921); errors=[]
        for t in samples:
            budget.check(); actual,_,_=robot.fk(curve(t)); target=target_at(t,cfg,point,base_R,axis,sign)
            anchor_error=np.linalg.norm(actual[:3,:3]@point+actual[:3,3]-(target[:3,:3]@point+target[:3,3]))
            angle_error=Rotation.from_matrix(target[:3,:3]@actual[:3,:3].T).magnitude()
            errors.append([anchor_error,angle_error])
        max_errors=np.max(errors,axis=0)
        print(name,'vmax',vmax.max(),'amax',amax.max(),'tracking',max_errors,flush=True)
        collision=continuous_clearance(curve,rows,robot,vmax,budget) if limits else {'certified':False,'reason':'joint limits failed'}
        report={'name':name,'position_min_rad':low.tolist(),'position_max_rad':high.tolist(),'speed_max_rad_s':vmax.tolist(),
                'acceleration_max_rad_s2':amax.tolist(),'joint_limits_certified':limits,'sampled_anchor_error_max_m':float(max_errors[0]),
                'sampled_orientation_error_max_rad':float(max_errors[1]),'tracking_samples':1921,
                'collision':collision,'static_path_passed':bool(limits and collision['certified'] and max_errors[0]<=.005 and max_errors[1]<=np.deg2rad(1))}
        reports.append(report); arrays[name+'_knots_s']=times; arrays[name+'_coefficients']=curve.c
        print(name,'static_path_passed',report['static_path_passed'],flush=True)
        (out/'paths.json').write_text(json.dumps({'paths':reports,'complete':False,'wall_s':time.monotonic()-budget.started},indent=2))
    budget.check()
    report={'kind':'CPU kinematic paths, NOT physics trajectories','paths':reports,'complete':True,
            'wall_s':time.monotonic()-budget.started,'peak_rss_bytes':budget.peak_rss,
            'online_action_domain_validated':False,'physics_admitted':False}
    (out/'paths.json').write_text(json.dumps(report,indent=2)); np.savez_compressed(out/'joint_paths.npz',**arrays)
    (ROOT/'configs/s1_reachable_candidate.json').write_text(json.dumps(cfg,indent=2))

if __name__=='__main__': main()
