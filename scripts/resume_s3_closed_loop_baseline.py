"""Reuse the accepted nominal route and finish the frozen baseline comparison."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import run_s3_closed_loop_baseline as baseline
from s3_closed_loop_feedback import feedback_action


ROOT, BASE = baseline.ROOT, baseline.BASE
OUT = BASE/'continuation'
PLAN = BASE/'resume_plan.json'
PREFLIGHT = BASE/'preflight.json'
STOP = BASE/'stopped.json'
PRE_SHA = '7857c1f7a34fe79b4678a2838f1dcff257e7b7190bb0b29bffd10d8f9df05e12'
STOP_SHA = 'b5dd5049dda5aa4427d0db69e4fc54dd4331a646a37be3968c1f3101b4dbef55'
RETAINED = f'{baseline.GROUPS[0]}_nominal'
INTERRUPTED = f'{baseline.GROUPS[0]}_feedback'
INTERRUPTED_CAP = 176
APPROVED_CAP = baseline.TOTAL_STEPS+INTERRUPTED_CAP
NEW_CAP = 3*baseline.HORIZON*baseline.SUBSTEPS


def matched(a,b,limit):
    n = min(limit,len(a),len(b))
    for k in range(n):
        x,y = (r[k].to_dict() for r in (a,b))
        for row in (x,y):
            row.pop('episode_id')
        if x != y:
            raise ValueError(f'matched prefix diverged at control {k}')
    if n < limit and len(a) != len(b):
        raise ValueError('matched zero-action prefix ended asymmetrically')
    return n


def verify_partial(original):
    stopped = baseline.read(baseline.relative(STOP),STOP_SHA)
    case = stopped['active_last_checkpoint']
    if (stopped['reason'],stopped['worker_starts'],stopped['completed_cases'],stopped['active_case'],
            stopped['completed_physics_steps'],case['physics_steps'],case['control_rows']) != (
            'system_available_ram_floor',2,[RETAINED],INTERRUPTED,3840,160,10):
        raise ValueError('the physical scope does not match the authorized interruption')
    if case['case'] != original['cases'][INTERRUPTED] or case['control_rows'] >= baseline.WARMUP:
        raise ValueError('the interruption does not belong to the frozen zero-action prefix')
    complete,partial = [(BASE/name/'worker/transitions.jsonl').read_bytes().splitlines()
                        for name in (RETAINED,INTERRUPTED)]
    if (len(complete),len(partial)) != (baseline.HORIZON,10):
        raise ValueError('retained or partial trajectory length changed')
    for k,(a,b) in enumerate(zip(complete,partial)):
        x,y = (json.loads(line) for line in (a,b))
        if (x.pop('episode_id'),y.pop('episode_id')) != (RETAINED,INTERRUPTED):
            raise ValueError('partial trajectory episode changed')
        if x != y or y['step_index'] != k or y['requested_action'] != [0.,0.]:
            raise ValueError('interrupted episode differs during the measured common prefix')
    return stopped


def freeze():
    original = baseline.load_plan(PREFLIGHT,PRE_SHA)
    stopped = verify_partial(original)
    accepted = baseline.HORIZON*baseline.SUBSTEPS
    max_wall = original['limits']['max_wall_s']-stopped['wall_s']
    if max_wall <= 0. or accepted+INTERRUPTED_CAP+NEW_CAP != APPROVED_CAP:
        raise ValueError('the approved cumulative continuation reservation is inconsistent')
    artifact_hashes = {baseline.relative(path):baseline.digest(path) for name in (RETAINED,INTERRUPTED)
                       for path in (BASE/name).rglob('*') if path.is_file()}
    with PLAN.open('x',encoding='utf-8') as stream:
        json.dump({'schema':'s3-closed-loop-baseline-resume-v1','preflight_sha256':PRE_SHA,
                   'interrupted_run_sha256':STOP_SHA,'resume_runner_sha256':baseline.digest(Path(__file__).resolve()),
                   'frozen_case_ids':list(original['cases']),'retained_case':RETAINED,
                   'interrupted_case':INTERRUPTED,'retained_physics_steps':accepted,
                   'interrupted_physics_steps_recorded':160,'interrupted_physics_steps_upper_bound':INTERRUPTED_CAP,
                   'new_physics_steps_cap':NEW_CAP,'approved_total_physics_steps_cap':APPROVED_CAP,
                   'limits':{**original['limits'],'max_wall_s':max_wall},'retained_sha256':artifact_hashes,
                   **baseline.SCOPE},stream,indent=2,allow_nan=False)
        stream.write('\n')


def load(path, sha):
    if path != PLAN:
        raise ValueError('unexpected continuation plan path')
    plan = baseline.read(baseline.relative(path),sha)
    original = baseline.load_plan(PREFLIGHT,PRE_SHA)
    if (plan['schema'] != 's3-closed-loop-baseline-resume-v1' or plan['preflight_sha256'] != PRE_SHA
            or plan['interrupted_run_sha256'] != STOP_SHA or plan['retained_case'] != RETAINED
            or plan['interrupted_case'] != INTERRUPTED or plan['frozen_case_ids'] != list(original['cases'])
            or plan['new_physics_steps_cap'] != NEW_CAP or plan['approved_total_physics_steps_cap'] != APPROVED_CAP
            or plan['interrupted_physics_steps_upper_bound'] != INTERRUPTED_CAP
            or plan['resume_runner_sha256'] != baseline.digest(Path(__file__).resolve())):
        raise ValueError('continuation scope changed')
    for name,digest in plan['retained_sha256'].items():
        baseline._verified_bytes(ROOT,name,digest)
    return original,plan


def verified_episode(original, root, name, state):
    case = original['cases'][name]
    if state['case'] != case or state['status'] != 'completed':
        raise ValueError('worker did not complete its declared arm')
    cfg = baseline.case_config(case)
    trace = root/name/'worker/transitions.jsonl'
    if baseline.digest(trace) != state['episode']['sha256']:
        raise ValueError('worker trajectory hash changed')
    options = original['feedback']
    rows = baseline.verify_episode(cfg,state,trace,name,options['correction_cap'])
    for tick,row in enumerate(rows):
        action = ([0.,0.] if case['arm'] == 'nominal' or tick < baseline.WARMUP
                  else feedback_action(row.state_t,options))
        if row.requested_action != action:
            raise ValueError('stored action no longer matches frozen causal policy')
    if bool(rows[-1].task['success']) != state['episode']['success']:
        raise ValueError('worker success differs from the standard hard criterion')
    return rows


def capture(path, sha, original, plan):
    OUT.mkdir(exist_ok=False)
    began = time.monotonic()
    states, rows, locations = {}, {}, {}
    active = None
    starts = 0
    new_steps = 0
    limits = plan['limits']
    try:
        if baseline.backend_preflight({'runtime':baseline.RUNTIME}) != original['backend']:
            raise ValueError('continuation runtime changed')
        previous = BASE/RETAINED/'worker/worker_result.json'
        state = baseline.read(baseline.relative(previous),plan['retained_sha256'][baseline.relative(previous)])
        rows[RETAINED] = verified_episode(original,BASE,RETAINED,state)
        states[RETAINED] = state
        locations[RETAINED] = baseline.relative(BASE/RETAINED)
        if state['physics_steps'] != plan['retained_physics_steps']:
            raise ValueError('retained nominal physical count changed')
        for name in plan['frozen_case_ids'][1:]:
            active = OUT/name
            case = original['cases'][name]
            active.mkdir(exist_ok=False)
            cfg = baseline.case_config(case)
            baseline.admit(active,cfg)
            if new_steps+baseline.HORIZON*baseline.SUBSTEPS > NEW_CAP:
                raise ValueError('continuation predicted to exceed its substep reservation')
            starts += 1
            with patch.dict(baseline.monitor.__globals__,{'__file__':__file__,
                    'directory_bytes':lambda _:baseline.directory_bytes(BASE),
                    'storage_admission':lambda root,_,budget:baseline.storage_admission(root,BASE,budget)}):
                state = baseline.monitor(active,path,sha,cfg['budget'],limits,began)
            rows[name] = verified_episode(original,OUT,name,state)
            states[name] = state
            locations[name] = baseline.relative(active)
            new_steps += state['physics_steps']
            print(f'{name}: {len(rows[name])} controls, success={state["episode"]["success"]}',flush=True)
        total = states[RETAINED]['physics_steps']+new_steps
        if (total > baseline.TOTAL_STEPS or total+INTERRUPTED_CAP > APPROVED_CAP
                or time.monotonic()-began >= limits['max_wall_s']
                or baseline.directory_bytes(BASE) > limits['max_new_output_mib']*1024**2):
            raise ValueError('continuation exceeded the aggregate resource reservation')
        comparison = {}
        for group in baseline.GROUPS:
            a,b = [f'{group}_{arm}' for arm in baseline.ARMS]
            if states[a]['initial_state'] != states[b]['initial_state']:
                raise ValueError('two controllers started with different physical states')
            comparison[group] = {'shared_prefix_verified_controls':matched(rows[a],rows[b],baseline.WARMUP),
                                 'nominal':states[a]['episode'],'feedback':states[b]['episode']}
        result = {'status':'completed','preflight_sha256':PRE_SHA,'resume_plan_sha256':sha,
                  'physics_steps':total,'new_physics_steps':new_steps,
                  'interrupted_physics_steps_recorded':160,'interrupted_physics_steps_upper_bound':INTERRUPTED_CAP,
                  'total_physics_steps_upper_bound':total+INTERRUPTED_CAP,
                  'approved_physics_steps_cap':APPROVED_CAP,'worker_starts':starts+2,'case_count':len(states),
                  'case_directories':locations,'paired':comparison,'continuation_wall_s':time.monotonic()-began,
                  **baseline.SCOPE}
        baseline.write_json(BASE/'comparison.json',result)
        hashes = {p.relative_to(BASE).as_posix():baseline.digest(p) for p in BASE.rglob('*') if p.is_file()}
        baseline.write_json(BASE/'validation.json',{k:v for k,v in result.items() if k!='paired'} |
                            {'artifacts_sha256':hashes})
    except BaseException as error:
        status = active/'worker/worker_result.json' if active else None
        state = json.loads(status.read_bytes()) if status and status.is_file() else {}
        baseline.write_json(OUT/'stopped.json',{'status':'stopped','reason':str(error),'worker_starts':starts,
                   'completed_cases':list(states),'completed_new_physics_steps':new_steps,
                   'active_case':active.name if active else None,'active_last_checkpoint':state,
                   'wall_s':time.monotonic()-began,**baseline.SCOPE})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','capture','worker'))
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--sha256')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.phase == 'freeze':
        freeze()
    else:
        if not args.sha256:
            parser.error('accepted continuation plan SHA256 required')
        source = args.plan.resolve()
        original,plan = load(source,args.sha256)
        if args.phase == 'capture':
            capture(source,args.sha256,original,plan)
        else:
            destination = args.output.resolve()
            if destination.parent.name == RETAINED:
                parser.error('continuation must reuse the completed nominal case')
            with patch.object(baseline,'BASE',OUT):
                baseline.worker(destination,original)
