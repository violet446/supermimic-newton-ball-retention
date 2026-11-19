"""Finish the same four baseline episodes after verifying the interrupted feedback tail."""
import argparse
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import run_s3_closed_loop_baseline as baseline
import resume_s3_closed_loop_baseline as earlier


ROOT, BASE = baseline.ROOT, baseline.BASE
OUT = BASE/'restored_completion'
PLAN = BASE/'restored_completion_plan.json'
PARTIAL = BASE/'continuation'/earlier.INTERRUPTED/'worker/transitions.jsonl'
FIRST_RESUME = BASE/'continuation/stopped.json'
FIRST_RESUME_SHA = '33b790fcd80596bdc59970a096edce627fc93f7a81c3fbc4e2f9472b289766a8'
RESUME_SHA = 'eb50871656ad99a1ead4955b9e7a6895c40b1f17db2be4657bd167db2e1b2da0'
PRELAUNCH = BASE/'continuation_prelaunch_attempt1/stopped.json'
PRELAUNCH_SHA = '300478b39e13c1f70bca24d78c5eb7d961a05e40a797f481ccc26507d1712ea6'
VERIFICATION_TICK = 75
TAIL_CAP = (baseline.HORIZON-VERIFICATION_TICK)*baseline.SUBSTEPS
FRESH_CAP = baseline.HORIZON*baseline.SUBSTEPS
PREVIOUS_PHYSICS_UPPER_BOUND = 3840+176+1232
ACCUMULATED_PHYSICS_CAP = 15568
NEW_CAP = TAIL_CAP+2*FRESH_CAP


def freeze():
    original, plan = earlier.load(earlier.PLAN,RESUME_SHA)
    stopped = baseline.read(baseline.relative(FIRST_RESUME),FIRST_RESUME_SHA)
    prelaunch = baseline.read(baseline.relative(PRELAUNCH),PRELAUNCH_SHA)
    state = stopped['active_last_checkpoint']
    if (stopped['reason'] != 'system_available_ram_floor'
            or stopped['active_case'] != earlier.INTERRUPTED or stopped['completed_new_physics_steps'] != 0
            or stopped['completed_cases'] != [earlier.RETAINED]
            or (state['physics_steps'],state['control_rows']) != (1216,76)
            or prelaunch['active_last_checkpoint']['physics_steps'] != 0
            or prelaunch['reason'] != 'worker resource recheck failed before importing Newton'):
        raise ValueError('feedback interruption does not match the approved reconstruction')
    # The original 176-step allowance already covered the first interruption.
    # The second contributes up to 16 unlogged steps and one validation repeat.
    if PREVIOUS_PHYSICS_UPPER_BOUND+NEW_CAP != ACCUMULATED_PHYSICS_CAP or (
            ACCUMULATED_PHYSICS_CAP-plan['approved_total_physics_steps_cap'] != 2*baseline.SUBSTEPS):
        raise ValueError('unexpected cumulative reservation')
    before = plan['limits']['max_wall_s']-stopped['wall_s']-prelaunch['wall_s']
    if before <= 0.:
        raise ValueError('no remaining wall time')
    sources = {baseline.relative(p):baseline.digest(p) for p in (Path(__file__).resolve(),
               ROOT/'scripts/s3_closed_loop_tail.py',PARTIAL)}
    sources.update({baseline.relative(FIRST_RESUME):FIRST_RESUME_SHA,baseline.relative(PRELAUNCH):PRELAUNCH_SHA})
    with PLAN.open('x',encoding='utf-8') as stream:
        json.dump({'schema':'s3-closed-loop-restore-tail-v1','input_sha256':sources,'resume_sha256':RESUME_SHA,
                   'preflight_sha256':earlier.PRE_SHA,'original_case_ids':list(original['cases']),
                   'reconstructed_tick':VERIFICATION_TICK,'verified_tick':VERIFICATION_TICK+1,
                   'old_physics_upper_bound':PREVIOUS_PHYSICS_UPPER_BOUND,'remaining_physics_cap':NEW_CAP,
                   'cumulative_physics_cap':ACCUMULATED_PHYSICS_CAP,
                   'limits':{**plan['limits'],'max_wall_s':before},**baseline.SCOPE},stream,indent=2,allow_nan=False)
        stream.write('\n')


def load(path, sha):
    if path != PLAN:
        raise ValueError('unexpected reconstruction plan path')
    plan = baseline.read(baseline.relative(path),sha)
    original,resume = earlier.load(earlier.PLAN,plan['resume_sha256'])
    if (plan['schema'] != 's3-closed-loop-restore-tail-v1' or plan['resume_sha256'] != RESUME_SHA
            or plan['preflight_sha256'] != earlier.PRE_SHA or plan['original_case_ids'] != list(original['cases'])
            or (plan['reconstructed_tick'],plan['verified_tick']) != (VERIFICATION_TICK,VERIFICATION_TICK+1)
            or (plan['old_physics_upper_bound'],plan['remaining_physics_cap'],plan['cumulative_physics_cap']) != (
                PREVIOUS_PHYSICS_UPPER_BOUND,NEW_CAP,ACCUMULATED_PHYSICS_CAP)):
        raise ValueError('reconstruction scope changed')
    for name,digest in plan['input_sha256'].items():
        baseline._verified_bytes(ROOT,name,digest)
    return original,resume,plan


def worker(out, original):
    from s3_closed_loop_feedback import feedback_action
    from s3_closed_loop_tail import run_tail

    name = out.parent.name
    if name not in list(original['cases'])[1:] or out.parent.parent != OUT or out.name != 'worker':
        raise ValueError('invalid reconstruction worker path')
    if name != earlier.INTERRUPTED:
        with patch.object(baseline,'BASE',OUT):
            baseline.worker(out,original)
        return
    began = time.monotonic()
    case = original['cases'][name]
    cfg = baseline.case_config(case)
    state = {'status':'starting','physics_steps':0,'worker_pid':os.getpid(),'case':case}

    def stage(label):
        state.update(stage=label,wall_s=time.monotonic()-began)
        baseline.write_json(out/'worker_result.json',state)
        if state['physics_steps'] > TAIL_CAP:
            raise RuntimeError('reconstruction worker exceeded physical scope')
        return baseline.check_stage_budget(out,cfg['budget'],began)

    def action(tick,obs):
        return [0.,0.] if tick < baseline.WARMUP else feedback_action(obs,original['feedback'])

    try:
        stage('resource_recheck')
        if not baseline.admission(baseline.snapshot(ROOT),cfg['budget'])['passed']:
            raise RuntimeError('worker resource recheck failed before importing Newton')
        run_tail(ROOT,cfg,name,action,out,state,stage,PARTIAL)
        state['status'] = 'completed'
        stage('complete')
    except BaseException as error:
        baseline.write_json(out/'worker_result.json',{**state,'status':'stopped','error':str(error)})
        raise


def run(path, sha, original, resume, plan):
    OUT.mkdir(exist_ok=False)
    began = time.monotonic()
    states,rows,locations = {},{},{}
    active = None
    attempts = 0
    new_steps = 0
    try:
        if baseline.backend_preflight({'runtime':baseline.RUNTIME}) != original['backend']:
            raise ValueError('reconstruction backend differs from original preflight')
        retained = earlier.RETAINED
        previous = BASE/retained/'worker/worker_result.json'
        state = baseline.read(baseline.relative(previous),resume['retained_sha256'][baseline.relative(previous)])
        rows[retained] = earlier.verified_episode(original,BASE,retained,state)
        states[retained] = state
        locations[retained] = baseline.relative(BASE/retained)
        for name in list(original['cases'])[1:]:
            active = OUT/name
            case = original['cases'][name]
            cfg = baseline.case_config(case)
            active.mkdir(exist_ok=False)
            cap = TAIL_CAP if name == earlier.INTERRUPTED else FRESH_CAP
            if new_steps+cap > NEW_CAP:
                raise ValueError('insufficient remaining physical reservation')
            for attempt in range(2):
                baseline.admit(active,cfg)
                attempts += 1
                try:
                    with patch.dict(baseline.monitor.__globals__,{'__file__':__file__,
                            'directory_bytes':lambda _:baseline.directory_bytes(BASE),
                            'storage_admission':lambda root,_,budget:baseline.storage_admission(root,BASE,budget)}):
                        state = baseline.monitor(active,path,sha,cfg['budget'],plan['limits'],began)
                except RuntimeError as error:
                    refused = active/'worker/worker_result.json'
                    if attempt or not refused.is_file():
                        raise
                    record = json.loads(refused.read_bytes())
                    if (str(error) != 'worker resource recheck failed before importing Newton'
                            or record['stage'] != 'resource_recheck' or record['physics_steps'] != 0):
                        raise
                    active.rename(OUT/f'{name}_prelaunch_refusal')
                    active.mkdir(exist_ok=False)
                    continue
                break
            physical = state['physics_steps']
            if not 0 < physical <= cap:
                raise ValueError('worker did not respect its physical reservation')
            full = state['episode']['physics_steps']
            expected = full-VERIFICATION_TICK*baseline.SUBSTEPS if name==earlier.INTERRUPTED else full
            if physical != expected:
                raise ValueError('worker physical and accepted trajectory clocks disagree')
            rows[name] = earlier.verified_episode(original,OUT,name,{**state,'physics_steps':full})
            states[name] = state
            locations[name] = baseline.relative(active)
            new_steps += physical
            print(f'{name}: {len(rows[name])} controls, success={state["episode"]["success"]}',flush=True)
        comparison = {}
        for group in baseline.GROUPS:
            a,b = [f'{group}_{arm}' for arm in baseline.ARMS]
            comparison[group] = {'shared_prefix_verified_controls':earlier.matched(rows[a],rows[b],baseline.WARMUP),
                                 'nominal':states[a]['episode'],'feedback':states[b]['episode']}
        accepted_steps = sum(row['episode']['physics_steps'] for row in states.values())
        if (new_steps+PREVIOUS_PHYSICS_UPPER_BOUND > ACCUMULATED_PHYSICS_CAP or accepted_steps > baseline.TOTAL_STEPS
                or time.monotonic()-began >= plan['limits']['max_wall_s']
                or baseline.directory_bytes(BASE) > plan['limits']['max_new_output_mib']*1024**2):
            raise ValueError('comparison exceeded the final resource reservation')
        result = {'status':'completed','preflight_sha256':earlier.PRE_SHA,'restored_completion_plan_sha256':sha,
                  'trajectories':len(states),'physics_steps':accepted_steps,'new_physics_steps':new_steps,
                  'archived_replay_verification_steps':baseline.SUBSTEPS,
                  'total_physics_steps_upper_bound':new_steps+PREVIOUS_PHYSICS_UPPER_BOUND,
                  'approved_physics_steps_cap':ACCUMULATED_PHYSICS_CAP,
                  'case_directories':locations,'worker_process_launches':attempts+4,
                  'wall_s':time.monotonic()-began,'paired':comparison,**baseline.SCOPE}
        baseline.write_json(BASE/'comparison.json',result)
        hashes = {p.relative_to(BASE).as_posix():baseline.digest(p) for p in BASE.rglob('*') if p.is_file()}
        baseline.write_json(BASE/'validation.json',{k:v for k,v in result.items() if k!='paired'} |
                            {'artifacts_sha256':hashes})
    except BaseException as error:
        status = active/'worker/worker_result.json' if active else None
        last = json.loads(status.read_bytes()) if status and status.is_file() else {}
        baseline.write_json(OUT/'stopped.json',{'status':'stopped','reason':str(error),'process_launches':attempts,
                   'completed_cases':list(states),'completed_new_physics_steps':new_steps,
                   'active_case':active.name if active else None,'active_last_checkpoint':last,
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
            parser.error('approved reconstruction plan hash required')
        path = args.plan.resolve()
        original,resume,plan = load(path,args.sha256)
        if args.phase == 'capture':
            run(path,args.sha256,original,resume,plan)
        else:
            worker(args.output.resolve(),original)
