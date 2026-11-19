"""Recover the second gradient-test snapshot and verify two nominal replays."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import collect_s3_small_action_response as collector

from archive_restore import write
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import reconstruct
from ball_retention.pair_collection import _initial_key
from ball_retention.resources import directory_bytes
from deterministic_archive_metrics import _full_branch, historical_import_metrics
from native_repeatability import _make_env, _nominal_schedule, _record_nominal
from response_contact_diagnosis import _allowed_pairs, _decode_trace
from saved_native_snapshot import decode_snapshot, restore_and_readback, verify_live_layout


ROOT = collector.ROOT
CASE = 'history_validation_yplus006_reverse'
SEED = 3392718248
TICK, SUBSTEPS, HORIZON = 168, 16, 32
LABELS = ['nominal_1', 'nominal_2']
PHYSICS_STEPS = len(LABELS)*HORIZON*SUBSTEPS


def load_run(path, expected):
    plan = json.loads(_verified_bytes(ROOT, path.relative_to(ROOT), expected))
    if plan['schema'] != 's3-gradient-public-restore-v1' or plan['scope'] != collector.SCOPE:
        raise ValueError('unexpected public restore plan')
    files = plan['files_sha256']
    for name, digest in files.items():
        _verified_bytes(ROOT, name, digest)

    def read(path):
        return json.loads(_verified_bytes(ROOT, path, files[Path(path).as_posix()]))

    def artifact(validation, name):
        path = Path(validation).parent/name
        if files[path.as_posix()] != read(validation)['artifacts_sha256'][name]:
            raise ValueError(f'accepted artifact changed: {path}')
        return read(path)

    static_path = plan['gradient_preflight_validation']
    static = read(static_path)
    if (static['status'],static['updates'],static['simulation_steps'],static['test_episode_files_read'],
            static['physical_restore_verified_cases'],static['rank_order_cases'],static['plan_sha256']) != (
                'completed',0,0,False,1,2,files[plan['gradient_preflight_plan']]):
        raise ValueError('frozen gradient preflight not accepted')
    candidates = artifact(static_path, 'candidate_commands.json')
    choice = candidates['cases'][CASE]
    case, restore = choice['identity'], choice['restoration']
    if (case['episode'],case['tick'],case['seed'],restore['kind'],restore['physical_restore_validated']) != (
            CASE,TICK,SEED,'public_state_only',False):
        raise ValueError('public snapshot identity changed')
    history = restore['source_validation']
    historic = read(history)
    if (historic['status'],historic['training_updates'],historic['test_metrics_used'],historic['control_accepted']) != (
            'completed_and_evaluated',0,False,False):
        raise ValueError('source development capture not accepted')
    manifest = artifact(history, 'capture_manifest.json')
    entry, = [row for row in manifest['entries'] if row['episode_id']==CASE]
    folder = Path(history).parent
    if ((entry['split'],entry['group_id'],entry['seed']) != ('validation',case['group'],SEED)
            or TICK not in entry['snapshot_ticks'] or
            ((folder/entry['path']).as_posix(),(folder/entry['config_path']).as_posix()) != (
                restore['source_trace'],restore['source_config'])
            or files[restore['source_trace']] != entry['sha256']
            or files[restore['source_config']] != entry['config_sha256']):
        raise ValueError('public source is outside its frozen development split')
    cfg = artifact(history, entry['config_path'])
    if cfg['seed'] != SEED or _initial_key(cfg) != choice['physical_initial_sha256']:
        raise ValueError('physical initial group or seed changed')
    target, = [row for row in map(json.loads, _verified_bytes(ROOT, restore['source_trace'],entry['sha256']).splitlines())
               if row['snapshot_tick']==TICK]
    commands = {name:choice['commands'][name] for name in LABELS}
    if (target['initial_state'] != restore['state'] or target['control_substeps'] != SUBSTEPS
            or target['physics_dt_s'] != 1/960 or cfg['physics']['dt_s'] != target['physics_dt_s']
            or any(target['nominal_actions'] != row['requested'] for row in commands.values())
            or commands[LABELS[0]] != commands[LABELS[1]]):
        raise ValueError('public state, clock or archived nominal requests changed')
    old_nominals = [row for row in target['branches'] if row['axis'] is None and row['offset']==0.]
    if ([row['repeat'] for row in old_nominals] != [0,1]
            or any(row['terminated'] or row['truncated'] or row['actions']!=target['nominal_actions']
                   or row['substeps']!=[SUBSTEPS]*HORIZON for row in old_nominals)):
        raise ValueError('archived development nominal references are incomplete')

    small = read(plan['small_capture_plan'])
    small_valid = (Path(small['output'])/'validation.json').as_posix()
    small_status = read(small_valid)
    small_static = small['preflight_validation']
    spec = artifact(small_static,'replay_spec.json')
    if (small_status['status']!='completed' or small_status['test_episode_files_read']
            or not small_status['nominal_gate_passed']
            or small_status['plan_sha256'] != files[plan['small_capture_plan']]
            or read(small_static)['plan_sha256'] != small['files_sha256'][small['preflight_plan']]
            or (spec['horizon'],spec['substeps_per_control']) != (HORIZON,SUBSTEPS)):
        raise ValueError('deterministic runtime evidence changed')
    limits = {**small['limits'], 'max_physics_substeps':PHYSICS_STEPS}
    if plan['limits'] != limits or (limits['max_worker_starts'],limits['sequential_environments'],
            limits['baseline_replays'],limits['nominal_prefix_substeps']) != (1,1,2,0):
        raise ValueError('physical or resource scope changed')
    baseline_cfg = artifact(small_valid,'execution_plan.json')['worker_config']
    if any(cfg[name]!=baseline_cfg[name] for name in ('physics','ball','reference','action','robot','prescribed','episode')):
        raise ValueError('physical scenario differs from the deterministic replay configuration family')
    cfg['budget'] = {**baseline_cfg['budget'],'max_physics_steps':PHYSICS_STEPS,'max_trace_steps':PHYSICS_STEPS}
    return plan, cfg, spec, case, restore, target, commands, old_nominals, choice['score']['target_y_tool_m']


def prepare(cfg, restore, target, commands, guard):
    prefix, nominal, checkpoint, executed = _nominal_schedule(ROOT, cfg, {TICK:target}, TICK,SUBSTEPS,guard)
    rebuilt = {'time_s':checkpoint.time_s,'q':checkpoint.q.tolist(),'qd':checkpoint.qd.tolist(),
               'tilt':checkpoint.tilt.tolist(),'filtered':checkpoint.filter_input_state.tolist()}
    if (prefix != restore['nominal_prefix_actions'] or rebuilt != restore['controller']
            or checkpoint.qdd.tolist() != restore['static_audit']['joint_qdd_from_nominal_prefix_rad_s2']
            or {'requested':nominal,'executed':executed} != commands[LABELS[0]]):
        raise ValueError('CPU controller replay did not reproduce frozen restoration and nominal commands')
    return checkpoint, nominal, executed


def preflight(cfg, spec, restore, target, commands, guard):
    backend = collector.backend_preflight(spec)
    if backend != spec['static_backend']:
        raise ValueError('installed backend differs from accepted deterministic capture')
    prepare(cfg, restore, target, commands, guard)
    if {'newton','warp'}.intersection(sys.modules):
        raise RuntimeError('static restoration preflight imported physics')
    return {'backend':backend,'nominal_controller_exact':True,'physics_steps':0}


def worker(out, path, expected):
    started = time.monotonic()
    state = {'status':'starting','worker_pid':os.getpid(),'physics_steps':0,'completed_branches':[]}

    def stage(name):
        state.setdefault('phase_events',[]).append({'stage':name,'elapsed_s':time.monotonic()-started,
            **collector.process_memory(state['worker_pid'])})
        collector.write_json(out/'worker_result.json',state)

    stage('starting')
    try:
        plan,cfg,spec,case,restore,target,commands,old_nominals,y0 = load_run(path,expected)
        limits = plan['limits']

        def guard():
            memory = collector.process_memory(state['worker_pid'])
            if (time.monotonic()-started >= limits['max_worker_wall_s']
                    or memory['peak_rss_bytes'] > limits['max_process_rss_gib']*1024**3
                    or memory['peak_private_commit_bytes'] > limits['max_process_commit_gib']*1024**3
                    or directory_bytes(out)>limits['max_new_output_mib']*1024**2):
                raise RuntimeError('public restore worker budget reached')

        checkpoint,nominal,executed = prepare(cfg,restore,target,commands,guard)
        if not collector.admission(collector.snapshot(ROOT),cfg['budget'])['passed']:
            raise RuntimeError('public restore worker resource recheck failed')
        stage('controller_and_resource_admission_verified')
        env = _make_env(ROOT,cfg,state,stage,expected_device_arch=spec['runtime']['cuda_arch'])
        env.reset(episode_id=case['episode'])
        if env.substeps!=SUBSTEPS or env.dt!=target['physics_dt_s'] or env.step_index or env.physics.steps:
            raise ValueError('fresh scene did not start at the zero-prefix clock')
        state['nominal_prefix_substeps'] = 0
        layout = verify_live_layout(env.physics,spec['static_snapshot_checks']['native_layout'])
        saved,evidence = reconstruct(env,restore['state'],TICK,checkpoint)
        if (saved['controller']!=restore['controller'] or not evidence['both_buffers_equal']
                or evidence['native_layout']!=layout or state['physics_steps']!=0):
            raise ValueError('restored controller, native buffers or physical scope changed')
        write(out,'public_readback.json',evidence)
        write(out,'native_layout.json',layout)
        document = {**saved,'state':asdict(saved['state']),'physics':{**saved['physics'],
            'arrays':{k:v.tolist() for k,v in saved['physics']['arrays'].items()},
            'command':[v.tolist() for v in saved['physics']['command']]}}
        saved = decode_snapshot(document,layout,cfg,TICK,SUBSTEPS,checkpoint)
        stage('public_snapshot_readback_verified')

        def advance(action):
            if state['physics_steps']+SUBSTEPS > PHYSICS_STEPS or time.monotonic()-started >= limits['max_worker_wall_s']:
                raise RuntimeError('nominal replay reached the frozen substep or wall cap')
            before,previous = env.physics.steps,env.state
            try:
                row = env.step(action)
            finally:
                state['physics_steps'] += env.physics.steps-before
            lead = env.step_index-TICK
            after = row.next_state_before_reset
            if (not 1<=lead<=HORIZON or row.step_index!=env.step_index-1 or row.episode_id!=case['episode']
                    or row.state_t!=previous or row.state_t.time_s!=before*env.dt
                    or env.physics.steps!=env.step_index*SUBSTEPS or env.physics.steps-before!=SUBSTEPS
                    or after.time_s!=env.physics.steps*env.dt or row.invalid_reason is not None
                    or row.terminated or row.truncated or row.requested_action!=action or after.previous_action!=action
                    or {'q':row.executed_joint_q,'qd':row.executed_joint_qd}!=executed[lead-1]):
                raise RuntimeError('nominal transition differs from frozen commands, clock or episode scope')
            return row

        branches = []
        for label in LABELS:
            write(out,f'{label}_readback.json',restore_and_readback(env,saved,SUBSTEPS))
            guard()
            stage(f'{label}_start')
            trace = _record_nominal(env,saved,nominal,advance,SUBSTEPS,SUBSTEPS*env.dt)
            filename = f'{label}.json'
            write(out,filename,trace)
            trace = json.loads((out/filename).read_bytes())
            _full_branch(trace,TICK)
            _decode_trace(trace,label,HORIZON,SUBSTEPS,TICK*SUBSTEPS,env.dt,_allowed_pairs(layout))
            branches.append(trace)
            state['completed_branches'].append(label)
            guard()
            stage(f'{label}_complete')

        comparison = collector.nominal_gate(branches,branches,spec['scoring']['summary_leads'])
        comparison['contact_records_equal'] = branches[0]['contacts']==branches[1]['contacts']
        comparison['passed'] = comparison['passed'] and comparison['contact_records_equal']
        comparison['repeats_byte_identical'] = (out/'nominal_1.json').read_bytes()==(out/'nominal_2.json').read_bytes()
        comparison['h32_signed_offset_from_frozen_y_m'] = [b['states_raw'][-1][1]-y0 for b in branches]
        historical = historical_import_metrics(branches,old_nominals)['cross_configuration']
        write(out,'nominal_comparison.json',comparison)
        write(out,'historical_context.json',historical)
        if not comparison['passed'] or state['physics_steps']!=PHYSICS_STEPS:
            raise RuntimeError('nominal repeatability or frozen step count failed')
        write(out,'snapshot.json',document)
        guard()
        state.update(status='completed',nominal_gate_passed=True,public_readback_passed=True)
        stage('nominal_recovery_completed')
    except BaseException as error:
        state.update(status='stopped',error=str(error))
        stage('stopped')
        raise


def capture(out, path, expected):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    try:
        plan,cfg,spec,case,restore,target,commands,_,_ = load_run(path,expected)
        limits,budget = plan['limits'],cfg['budget']

        def guard():
            if time.monotonic()-began>=limits['max_wall_s'] or directory_bytes(out)>limits['max_new_output_mib']*1024**2:
                raise RuntimeError('end-to-end public restore budget reached')

        collector.write_json(out/'static_preflight.json',preflight(cfg,spec,restore,target,commands,guard))
        collector.write_json(out/'execution_plan.json',{'plan':plan,'plan_sha256':expected,'worker_config':cfg})
        sample = collector.snapshot(ROOT)
        hardware = collector.admission(sample,budget)
        storage = collector.storage_admission(ROOT,out/'worker',budget)
        environment = directory_bytes(ROOT/'.venv')
        collector.write_json(out/'admission.json',{'hardware':hardware,'sample':sample,
            'storage':storage,'environment_bytes':environment})
        if not hardware['passed'] or not storage['passed'] or environment>budget['max_environment_gib']*1024**3:
            raise RuntimeError('public restore resource admission failed')
        with patch.object(collector,'__file__',__file__):
            result = collector.monitor(out,path,expected,budget,limits,began)
        if ((result['physics_steps'],result['nominal_prefix_substeps'],result['completed_branches']) !=
                (PHYSICS_STEPS,0,LABELS) or not result['public_readback_passed'] or not result['nominal_gate_passed']):
            raise ValueError('worker did not meet its frozen recovery gates')
        for name,digest in plan['files_sha256'].items():
            _verified_bytes(ROOT,name,digest)
        hashes = {p.relative_to(out).as_posix():collector.digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
        guard()
        collector.write_json(out/'validation.json',{'status':'completed','plan_sha256':expected,'case':case,
            'worker_starts':1,'physics_steps':PHYSICS_STEPS,'nominal_prefix_substeps':0,'branches':LABELS,
            'public_readback_passed':True,'nominal_gate_passed':True,'wall_s':time.monotonic()-began,
            **collector.SCOPE,'artifacts_sha256':hashes})
    except BaseException as error:
        path = out/'worker/worker_result.json'
        last = json.loads(path.read_bytes()) if path.exists() else {}
        collector.write_json(out/'stopped.json',{'status':'stopped','reason':str(error),
            'wall_s':time.monotonic()-began,'last_reported_physics_steps':last.get('physics_steps',0),
            'branches':last.get('completed_branches',[]),**collector.SCOPE})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('capture','worker'))
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    path = args.plan.resolve()
    plan = json.loads(_verified_bytes(ROOT,path.relative_to(ROOT),args.sha256))
    expected = (ROOT/plan['output']).resolve()
    out = args.output.resolve()
    if (len(expected.relative_to(ROOT/'runs').parts)!=1
            or out != (expected/'worker' if args.phase=='worker' else expected)):
        parser.error('output differs from frozen public restore plan')
    (worker if args.phase=='worker' else capture)(out,path,args.sha256)
