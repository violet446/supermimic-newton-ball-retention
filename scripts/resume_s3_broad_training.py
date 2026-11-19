"""Resume the frozen 36-pair capture from its audited native snapshot."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import collect_s3_broad_training as capture
from saved_native_snapshot import decode_snapshot


ROOT, BASE = capture.ROOT, capture.BASE
RUN = BASE/'continuation'
AUDIT = ROOT/'runs/s3_world_increment_data97_partial_audit_20261118_verified/validation.json'
AUDIT_SHA = '0fb0255d3f217d26469393c2325ba86e71fbad24af4fe65b2b64cf95b01687c1'
PRE = BASE/'preflight.json'
STOP = BASE/'stopped.json'
STOP_SHA = '6c427778b4d60191291f7457844fd13a9d436bd43a59532a5bbd1602dbc1d150'
ALLOWED_TOTAL = 55040
CASE = 'broad_train_72_left'
CHUNK = capture.HORIZON*capture.SUBSTEPS
write_json = capture.write_json


def verified(path, sha):
    return capture.read(capture.relative(path), sha)


def freeze(path):
    accepted = verified(AUDIT, AUDIT_SHA)
    audit_path = AUDIT.with_name('partial_capture_audit.json')
    audit = verified(audit_path, accepted['artifacts_sha256'][audit_path.name])
    stopped = verified(STOP, STOP_SHA)
    original = verified(PRE, audit['source_preflight_sha256'])
    if (accepted['status'] != 'completed' or accepted['updates'] != 0 or audit['case'] != CASE
            or not audit['nominal_files_byte_identical'] or not audit['snapshot_decoding_passed']
            or stopped['reason'] != 'system_available_ram_floor' or stopped['completed_cases']
            or stopped['active_case'] != CASE or audit['physics_steps_completed'] != stopped['active_physics_steps_at_last_event']
            or audit['total_steps_including_interrupted_attempt_upper_bound'] != ALLOWED_TOTAL
            or audit['new_steps_from_saved_snapshot']+audit['physics_steps_completed'] != capture.TOTAL_STEPS):
        raise ValueError('audited partial run does not fit the approved continuation')
    for name, sha in {**original['source_sha256'], **audit['partial_artifacts_sha256']}.items():
        capture._verified_bytes(ROOT, name, sha)
    with path.open('x',encoding='utf-8') as stream:
        json.dump({'schema':'s3-broad-training-resume-v1','original_preflight_sha256':audit['source_preflight_sha256'],
                   'accepted_audit_sha256':AUDIT_SHA,'interrupted_run_sha256':STOP_SHA,
                   'continuation_runner_sha256':capture.digest(Path(__file__).resolve()),
                   'reused_case':CASE,'reused_steps':audit['physics_steps_completed'],
                   'possible_unrecorded_steps':audit['unrecorded_physics_steps_upper_bound'],
                   'new_steps_budget':audit['new_steps_from_saved_snapshot'],'max_physical_substeps':ALLOWED_TOTAL,
                   'completed_branches':audit['finished_branches'],'pending_branches':audit['pending_first_case_branches'],
                   'limits':{**capture.LIMITS,'max_wall_s':capture.LIMITS['max_wall_s']-stopped['wall_s']},
                   'retained_artifacts_sha256':audit['partial_artifacts_sha256']},stream,indent=2,allow_nan=False)
        stream.write('\n')


def load(path, expected):
    plan = verified(path, expected)
    if (plan['schema'] != 's3-broad-training-resume-v1' or plan['max_physical_substeps'] != ALLOWED_TOTAL
            or plan['reused_steps']+plan['possible_unrecorded_steps']+plan['new_steps_budget'] != ALLOWED_TOTAL
            or plan['possible_unrecorded_steps'] != CHUNK or plan['reused_case'] != CASE
            or plan['continuation_runner_sha256'] != capture.digest(Path(__file__).resolve())):
        raise ValueError('continuation scope or runner changed')
    original = verified(PRE, plan['original_preflight_sha256'])
    verified(AUDIT, plan['accepted_audit_sha256'])
    verified(STOP, plan['interrupted_run_sha256'])
    for name, sha in {**original['source_sha256'], **plan['retained_artifacts_sha256']}.items():
        capture._verified_bytes(ROOT, name, sha)
    return plan, original


def resume_case(out, plan, cfg, evidence, layout, cap, state, stage):
    retained = BASE/CASE/'worker'
    manifest = plan['retained_artifacts_sha256']

    def copy(name):
        source = retained/name
        sha = manifest[capture.relative(source)]
        capture._verified_bytes(ROOT,capture.relative(source),sha)
        shutil.copyfile(source,out/name)
        if capture.digest(out/name) != sha:
            raise ValueError(f'copied partial archive differs: {name}')
        return json.loads((out/name).read_bytes()) if name.endswith('.json') else None

    completed, pending = plan['completed_branches'], plan['pending_branches']
    if list(evidence['branches']) != completed+pending or cap-plan['reused_steps'] != len(pending)*CHUNK:
        raise ValueError('continuation would replay a completed branch or change the branch set')
    document = copy('snapshot.json')
    if copy('native_layout.json') != layout:
        raise ValueError('native continuation layout changed')
    nominal = copy('nominal_comparison.json')
    if not nominal['passed'] or not nominal['contacts_match']:
        raise ValueError('retained nominal replays failed')
    copy('prefix.jsonl')
    tick, substeps, dt = evidence['snapshot_tick'], capture.SUBSTEPS, cfg['physics']['dt_s']
    case = evidence['case']
    traces = {}
    for label in completed:
        traces[label] = copy(f'{label}.json')
        if copy(f'{label}_readback.json')['both_buffers_exact'] is not True:
            raise ValueError(f'invalid retained readback for {label}')
    native = decode_snapshot(document,layout,cfg,tick,substeps)
    if (native['episode_id'] != case['episode_id'] or native['controller'] != evidence['snapshot_control_only']
            or native['physics']['command'][2].tolist() != evidence['snapshot_joint_qdd_rad_s2']):
        raise ValueError('continuation snapshot does not match the frozen case')
    state.update(case=case, completed_branches=completed.copy(), prefix_substeps=tick*substeps,
                 reused_physics_steps=plan['reused_steps'])
    env = capture._make_env(ROOT,cfg,state,stage,expected_device_arch=capture.RUNTIME['cuda_arch'])
    env.reset(episode_id=case['episode_id'])
    capture.verify_live_layout(env.physics,layout)
    for label in pending:
        capture.write(out,f'{label}_readback.json',capture.restore_and_readback(env,native,substeps))
        stage(f'{label}_start')
        branch = evidence['branches'][label]

        def advance(action):
            before,index = env.physics.steps,env.step_index
            if state['physics_steps']+substeps > cap-plan['reused_steps']:
                raise RuntimeError('continuation would exceed its physical budget')
            try:
                row = env.step(action)
            finally:
                state['physics_steps'] += env.physics.steps-before
            nxt = row.next_state_before_reset
            if (row.episode_id != case['episode_id'] or row.step_index != index or env.step_index != index+1
                    or row.state_t.time_s != before*dt or before != index*substeps
                    or env.physics.steps-before != substeps or nxt.time_s != env.physics.steps*dt
                    or row.invalid_reason is not None or row.terminated or row.truncated
                    or row.requested_action != action or nxt.previous_action != action
                    or {'q':row.executed_joint_q,'qd':row.executed_joint_qd} != branch['executed'][index-tick]):
                capture.write(out,'failed_transition.json',{'branch':label,'transition':row.to_dict()})
                raise ValueError(f'{label} tick {index} failed the clock, command or episode check')
            return row

        trace = capture._record_nominal(env,native,branch['requested'],advance,substeps,substeps*dt)
        if [r['physics_step'] for r in trace['contacts']] != list(range(tick*substeps+1,(tick+capture.HORIZON)*substeps+1)):
            raise ValueError(f'{label} continuation contact clock is incomplete')
        capture.write(out,f'{label}.json',trace)
        trace = json.loads((out/f'{label}.json').read_bytes())
        capture._full_branch(trace,tick)
        traces[label] = trace
        state['completed_branches'].append(label)
        stage(f'{label}_complete')
    if state['physics_steps']+state['reused_physics_steps'] != cap:
        raise ValueError('continued case did not exhaust its frozen step count')
    capture.write(out,'signal_report.json',capture.signal_report({'directions':evidence['directions_Hx2']},traces))
    state.update(status='completed',nominal_passed=True)


def worker(out, plan, original):
    name = out.parent.name
    if out.parent.parent != RUN or out.name != 'worker':
        raise ValueError('continuation worker output outside frozen path')
    began = time.monotonic()
    cfg,evidence,cap = capture.load_evidence(original,name)
    budget = cfg['budget']
    state = {'status':'starting','worker_pid':os.getpid(),'physics_steps':0,'reused_physics_steps':0}

    def stage(label):
        memory = capture.process_memory(os.getpid())
        state.update(stage=label,memory=memory)
        write_json(out/'worker_result.json',state)
        if (time.monotonic()-began >= budget['wall_s']
                or capture.directory_bytes(out) > budget['max_new_output_mib']*1024**2
                or memory['peak_rss_bytes'] > budget['max_process_rss_gib']*1024**3
                or memory['peak_private_commit_bytes'] > budget['max_process_commit_gib']*1024**3):
            raise RuntimeError('continuation worker resource limit reached')

    try:
        stage('resource_recheck')
        if not capture.admission(capture.snapshot(ROOT),budget)['passed']:
            raise RuntimeError('continuation worker resource recheck failed before physics imports')
        layout = capture.read(original['layout']['path'],original['layout']['sha256'])
        if name == CASE:
            resume_case(out,plan,cfg,evidence,layout,cap,state,stage)
        else:
            capture.record_case(out,cfg,evidence,layout,cap,state,stage)
        stage('complete')
    except BaseException as error:
        write_json(out/'worker_result.json',{**state,'status':'stopped','error':str(error)})
        raise


def run(path, sha, plan, original):
    RUN.mkdir(exist_ok=False)
    began = time.monotonic()
    limit = plan['limits']
    accumulated = 0
    started = 0
    entries, states = [], {}
    active = None
    try:
        if capture.backend_preflight({'runtime':capture.RUNTIME}) != original['backend']:
            raise ValueError('continuation backend does not match the accepted device and runtime')
        for name in original['cases']:
            cfg,evidence,cap = capture.load_evidence(original,name)
            active = RUN/name
            active.mkdir(exist_ok=False)
            capture.admit(active,cfg['budget'])
            started += 1
            with patch.dict(capture.monitor.__globals__,{'__file__':__file__,
                        'directory_bytes':lambda _:capture.directory_bytes(BASE),
                        'storage_admission':lambda root,_,budget:capture.storage_admission(root,BASE,budget)}):
                state = capture.monitor(active,path,sha,cfg['budget'],limit,began)
            if (state['case'] != evidence['case'] or state['physics_steps']+state['reused_physics_steps'] != cap
                    or state['completed_branches'] != list(evidence['branches']) or state['nominal_passed'] is not True):
                raise ValueError('continued worker does not match its complete case')
            accumulated += state['physics_steps']
            entries.append(capture.manifest_entry(active/'worker',cfg,evidence))
            states[name] = state['physics_steps']
            print(f'{name} complete, {len(entries)}/6 cases and {accumulated} new substeps.',flush=True)
        if accumulated != plan['new_steps_budget']:
            raise ValueError('continuation did not exhaust the frozen new step count')
        pending = BASE/'capture_manifest.pending.json'
        write_json(pending,{'schema':capture.SCHEMA,'horizon':capture.HORIZON,'design_sha256':plan['original_preflight_sha256'],
                   'continuation_sha256':sha,'development_holdout_group':'negative_y_basin','entries':entries})
        manifest_sha = capture.digest(pending)
        data = capture.prepare_timed_pairs(ROOT,capture.relative(pending),manifest_sha)
        expected_directions = {pair['direction_id']:6 for pair in entries[0]['pairs']}
        if (len(data.batch.initial) != 36 or len(set(data.batch.group_ids)) != 6
                or Counter(data.directions) != expected_directions):
            raise ValueError('archive reader did not recover the fixed groups and action cells')
        if time.monotonic()-began >= limit['max_wall_s'] or capture.directory_bytes(BASE) > limit['max_new_output_mib']*1024**2:
            raise RuntimeError('continuation aggregate resource cap reached')
        pending.rename(BASE/'capture_manifest.json')
        write_json(BASE/'validation.json',{'status':'completed','preflight_sha256':plan['original_preflight_sha256'],
                   'resume_plan_sha256':sha,'physics_steps':accumulated+plan['reused_steps'],
                   'previously_recorded_physics_steps':plan['reused_steps'],'new_physics_steps':accumulated,
                   'unrecorded_physics_steps_upper_bound':plan['possible_unrecorded_steps'],
                   'approved_physics_steps_upper_bound':ALLOWED_TOTAL,'pairs':36,'worker_starts':started+1,
                   'cases':list(states),'wall_s':time.monotonic()-began,**capture.SCOPE,
                   'artifacts_sha256':{'capture_manifest.json':manifest_sha}})
    except BaseException as error:
        current = active/'worker/worker_result.json' if active else None
        last = json.loads(current.read_bytes()) if current and current.is_file() else {}
        write_json(RUN/'stopped.json',{'status':'stopped','reason':str(error),'worker_starts':started,
                    'completed_cases':states,'completed_new_steps':accumulated,'active_case':active.name if active else None,
                    'active_worker_last_checkpoint':last,'wall_s':time.monotonic()-began,**capture.SCOPE})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','capture','worker'))
    parser.add_argument('--plan',type=Path,default=BASE/'resume_plan.json')
    parser.add_argument('--sha256')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    path = args.plan.resolve()
    if path != BASE/'resume_plan.json':
        parser.error('unexpected continuation plan path')
    if args.phase == 'freeze':
        freeze(path)
    else:
        if not args.sha256:
            parser.error('frozen continuation plan hash is required')
        plan, original = load(path,args.sha256)
        if args.phase == 'capture':
            run(path,args.sha256,plan,original)
        else:
            worker(args.output.resolve(),plan,original)
