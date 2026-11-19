from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from ball_retention.robot import Robot
from ball_retention.clearance import geometry_rows,pose_clearance
from search_reachability import Budget

out=ROOT/'runs/s1_reachability_20261111'; budget=Budget()
search=json.loads((out/'search.json').read_text()); robot=Robot(ROOT/'assets/ec66_description.urdf')
rows=geometry_rows(ROOT/'assets'); results=[]
for row in search['rows']:
    if row['anchor_world_m'][0] not in (.55,.60,.65,.70,.75): continue
    for index,fit in enumerate(row['fits']):
        if not fit['pose_passed']: continue
        budget.check(); checks=pose_clearance(rows,robot,fit['q'])
        failed={k:v for k,v in checks['pair_lower_bounds_m'].items() if v<.002}
        results.append({'anchor_world_m':row['anchor_world_m'],'seed_index':index,'q':fit['q'],**checks,
                        'uncertified_pairs':failed,'pose_clearance_certified':not failed and checks['floor_clearance_m']>=.02})
        print(row['anchor_world_m'][0],index,'uncertified',len(failed),'floor',round(checks['floor_clearance_m'],4),flush=True)
(out/'candidates.json').write_text(json.dumps({'results':results,'wall_s':time.monotonic()-budget.started,'peak_rss_bytes':budget.peak_rss},indent=2))
