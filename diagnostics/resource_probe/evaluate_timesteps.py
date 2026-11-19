"""Compare physical traces at matched times with preregistered criteria."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'runs/s1_basin_20261112'
plan=json.loads((out/'plan.json').read_text()); crit=plan['criteria_before_runs']
arrays={}; cases=[]
for case in plan['cases']:
    hz=case['hz']; folder=out/f'hz{hz}'
    worker=json.loads((folder/'worker_result.json').read_text())
    if worker['status']!='diagnostic_completed': raise RuntimeError(f'case {hz} not completed')
    rows=[json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
    if len(rows)!=case['steps']: raise RuntimeError('trace incomplete')
    p=np.array([r['body_q'][1][:3] for r in rows]); v=np.array([r['body_qd_com_world'][1][:3] for r in rows]); w=np.array([r['body_qd_com_world'][1][3:] for r in rows])
    first=round(crit['ignore_initial_settling_s']*hz)-1
    excursion=float(np.linalg.norm(p[first:]-p[first],axis=1).max()); final_speed=float(np.linalg.norm(v[-1])); penetration=max(0,-min(r['surface_clearance_m'] for r in rows))
    passed=excursion<=crit['max_position_excursion_after_settling_m'] and final_speed<=crit['final_linear_speed_max_m_s'] and penetration<=crit['max_penetration_m'] and not any(r['floor_reached'] for r in rows)
    cases.append({'hz':hz,'physics_steps':worker['physics_steps'],'worker_wall_s':worker['wall_s'],'excursion_after_0p1s_m':excursion,'final_linear_speed_m_s':final_speed,'max_penetration_m':penetration,'initial_state_stability_passed':passed,'final_position_world_m':p[-1].tolist(),'replay_position_error_m':worker['replay_position_error_m'],'reset_max_error':worker['reset_max_error']})
    indices=np.arange(6,121)*(hz//60)-1
    arrays[hz]=(p[indices],v[indices],w[indices])
pairs=[]
for a,b in ((480,960),(960,1920),(480,1920)):
    errs=[float(np.linalg.norm(x-y,axis=1).max()) for x,y in zip(arrays[a],arrays[b])]
    pairs.append({'hz':[a,b],'max_position_error_m':errs[0],'max_linear_velocity_error_m_s':errs[1],'max_angular_velocity_error_rad_s':errs[2],'passed':all(e<=lim for e,lim in zip(errs,[crit['pair_max_position_error_after_settling_m'],crit['pair_max_linear_velocity_error_after_settling_m_s'],crit['pair_max_angular_velocity_error_after_settling_rad_s']]))})
result={'cases':cases,'pairs':pairs,'all_preregistered_checks_passed':all(c['initial_state_stability_passed'] for c in cases) and all(p['passed'] for p in pairs),'total_physics_steps':sum(c['physics_steps'] for c in cases),'total_worker_wall_s':sum(c['worker_wall_s'] for c in cases),'s1_complete':False,'comparison_times':'0.1 to 2.0 seconds inclusive at 60Hz; no temporal interpolation','scope':'one zero-velocity lower-basin initial state, fixed tool, no policy; not general contact convergence'}
(out/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
