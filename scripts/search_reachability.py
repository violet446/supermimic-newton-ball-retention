"""Bounded static search, no Newton imports. Keep old failed configuration intact."""
from pathlib import Path
import ctypes, hashlib, json, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ball_retention.geometry import load_tool,support_point
from ball_retention.robot import Robot
from ball_retention.task import tool_pose_for_anchor

class ProcessMemory(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]

class Budget:
    def __init__(self,seconds=600):
        self.started=time.monotonic(); self.seconds=seconds; self.peak_rss=0
    def check(self):
        m=ProcessMemory(); m.cb=ctypes.sizeof(m)
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
        if not ctypes.windll.psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(m),m.cb):
            raise OSError('cannot monitor process memory')
        self.peak_rss=max(self.peak_rss,m.ws)
        if m.ws>1024**3 or time.monotonic()-self.started>self.seconds:
            raise RuntimeError('static budget exceeded')

def main():
    out=ROOT/'runs/s1_reachability_20261111'; out.mkdir(exist_ok=True)
    budget=Budget(); robot=Robot(ROOT/'assets/ec66_description.urdf')
    mesh=load_tool(ROOT/'assets/shovel.STL'); point,normal,_=support_point(mesh,[.32,0])
    forward=np.array([1.,0,0]); forward-=normal*np.dot(forward,normal); forward/=np.linalg.norm(forward)
    R=np.column_stack((forward,np.cross(normal,forward),normal)).T
    rng=np.random.default_rng(20261111); seeds=[np.zeros(6)]+[rng.uniform(-np.pi,np.pi,6) for _ in range(7)]
    # Same orientation/height/y; change only radial target location, fixed in advance.
    rows=[]
    for x in (.45,.50,.55,.60,.65,.70,.75):
        target=tool_pose_for_anchor([x,0,.35],point,R)
        fits=[]
        for seed in seeds:
            budget.check(); fit=robot.ik(target,seed); fits.append(fit)
        rows.append({'anchor_world_m':[x,0,.35],'target_tool_origin_m':target[:3,3].tolist(),'fits':fits})
        print(f'x={x:.2f}: {sum(f["pose_passed"] for f in fits)}/8 pose solutions',flush=True)
    report={'kind':'static radial search; no physics or collision admission','seed':20261111,
            'source_config_sha256':hashlib.sha256((ROOT/'configs/s1_smoke.json').read_bytes()).hexdigest(),
            'anchor_local_m':point.tolist(),'base_rotation':R.tolist(),'rows':rows,
            'wall_s':time.monotonic()-budget.started,'peak_rss_bytes':budget.peak_rss}
    (out/'search.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__': main()
