"""Headless resource-gated diagnostic launcher. A refusal imports no physics libraries."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from ball_retention.resources import snapshot,admission,directory_bytes
from ball_retention.guard import process_memory,stop_reason
from ball_retention.native_repeatability_plan import MARKER as NATIVE_MARKER, MODE as NATIVE_MODE
from ball_retention.deterministic_archive_import_plan import MARKER as IMPORT_MARKER, MODE as IMPORT_MODE
from ball_retention.deterministic_response_plan import MARKER as RESPONSE_MARKER, MODE as RESPONSE_MODE
from ball_retention.training_timing_capture_plan import MARKER as TIMING_MARKER, MODE as TIMING_MODE

def stop_owned_tree(proc):
    # Windows venv python.exe is a launcher; terminating it alone may orphan the worker.
    if proc.poll() is None:
        subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                       timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)

def monitored_pid(worker_path,launcher_pid):
    try:
        record=json.loads(worker_path.read_text())
        return int(record.get('worker_pid',launcher_pid))
    except (FileNotFoundError,json.JSONDecodeError):
        return launcher_pid


def storage_admission(root,output_scope,budget):
    sizes={name:directory_bytes(path) for name,path in
           (('output_bytes',output_scope),('cache_bytes',root/'.cache'),('temp_bytes',root/'.tmp'))}
    total=sum(sizes.values());limit=budget['max_output_cache_gib']*1024**3
    return {'passed':total<=limit,'total_bytes':total,'limit_bytes':limit,**sizes}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='diagnostics/resource_probe/config.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['initialize','fixed-tool','tilt-plus','tilt-minus','drop','fk','arm-static','arm-move','arm-episode','s2-collect','s3-pairs','s3-archive-restore',NATIVE_MODE,IMPORT_MODE,RESPONSE_MODE,TIMING_MODE],default='initialize')
    parser.add_argument('--steps',type=int,default=240)
    args=parser.parse_args(); out=(ROOT/args.output).resolve(); out.relative_to(ROOT)
    if out.exists(): raise ValueError('use a fresh output directory to preserve evidence')
    out.mkdir(parents=True); config=(ROOT/args.config).resolve(); cfg=json.loads(config.read_text())
    collection=cfg.get('collection') if args.mode=='s2-collect' else None
    output_scope=out
    if collection is not None:
        output_scope=(ROOT/collection['dataset_dir']).resolve()
        output_scope.relative_to(ROOT/'data')
        if out!=output_scope/'episodes'/collection['case']['id'] or cfg['seed']!=collection['case']['seed']:
            raise ValueError('worker identity does not match the collection plan')
    if args.mode=='s3-pairs':
        master_seed=cfg['action_pairs']['master_seed']
    else:
        master_seed=collection['master_seed'] if collection is not None else cfg['seed']
    small_offset = 'small_offset_probe' in cfg
    archive_restore = args.mode == 's3-archive-restore'
    native_repeatability = args.mode == NATIVE_MODE
    deterministic_import = args.mode == IMPORT_MODE
    deterministic_response = args.mode == RESPONSE_MODE
    timing_capture = args.mode == TIMING_MODE
    if timing_capture or TIMING_MARKER in cfg:
        if not timing_capture:
            raise ValueError('training timing case requires s3-training-timing-capture mode')
        from ball_retention.training_timing_capture_plan import validate_case
        validate_case(ROOT, cfg, args.steps)
    elif deterministic_response or RESPONSE_MARKER in cfg:
        if not deterministic_response:
            raise ValueError('deterministic response case requires s3-deterministic-action-response mode')
        from ball_retention.deterministic_response_plan import validate_case
        validate_case(ROOT, cfg, args.steps)
    elif deterministic_import or IMPORT_MARKER in cfg:
        if not deterministic_import:
            raise ValueError('deterministic archive case requires s3-deterministic-archive-import mode')
        from ball_retention.deterministic_archive_import_plan import validate_case
        validate_case(ROOT, cfg, args.steps)
    elif native_repeatability or NATIVE_MARKER in cfg:
        if not native_repeatability:
            raise ValueError('native repeatability case requires s3-native-repeatability mode')
        from ball_retention.native_repeatability_plan import validate_case
        validate_case(ROOT, cfg, args.steps)
    elif archive_restore or 'archive_restore_pilot' in cfg:
        if not archive_restore:
            raise ValueError('archive restoration case requires s3-archive-restore mode')
        from ball_retention.archive_restore_plan import validate_case
        validate_case(ROOT, cfg, args.steps)
    elif small_offset:
        if args.mode != 's3-pairs':
            raise ValueError('small-offset case requires s3-pairs mode')
        from ball_retention.small_offset_probe import validate_small_offset_case
        case = validate_small_offset_case(ROOT, cfg)
        if args.steps != case['reserved_substeps']:
            raise ValueError('small-offset worker requires the exact frozen substep count')
    elif master_seed!=20261111:
        if args.mode!='s3-pairs':raise ValueError('outside current probe seed scope')
        from ball_retention.history_capture import validate_history_case
        validate_history_case(ROOT,cfg)
    if not 1<=args.steps<=cfg['budget'].get('max_trace_steps',1920): raise ValueError('outside current probe scope')
    if not (small_offset or archive_restore or native_repeatability or deterministic_import or deterministic_response or timing_capture) and args.steps+32>cfg['budget']['max_physics_steps']: raise ValueError('step budget exceeded')
    manifest=json.loads((ROOT/'assets/manifest.json').read_text())
    for row in manifest['assets']:
        if hashlib.sha256((ROOT/row['local_path']).read_bytes()).hexdigest()!=row['sha256']: raise ValueError('asset changed')
    sample=snapshot(ROOT); result={'mode':args.mode,'requested_steps':args.steps,'seed':cfg['seed'],
               'admission':admission(sample,cfg['budget']),'resources_before':sample,
               'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'worker_started':False,'s1_complete':False}
    (out/'effective_config.json').write_text(json.dumps(cfg,indent=2))
    # Bind evidence to the implementation and prescribed path, including refused runs.
    sources=list((ROOT/'src/ball_retention').glob('*.py'))+list((ROOT/'diagnostics/resource_probe').glob('*.py'))
    if 'prescribed' in cfg: sources.append(ROOT/cfg['prescribed']['joint_path'])
    result['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    def save(): (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['admission']['passed']:
        result['status']='blocked_resource_admission'; save(); print(json.dumps(result,indent=2)); return 2
    if directory_bytes(ROOT/'.venv')>cfg['budget']['max_environment_gib']*1024**3:
        result['status']='blocked_environment_size'; save(); return 2
    result['storage_admission']=storage_admission(ROOT,output_scope,cfg['budget'])
    if not result['storage_admission']['passed']:
        result.update(status='blocked_storage_admission',stop_reason='output_cache_budget')
        save();return 2
    env=os.environ.copy()
    for name,path in {'TEMP':'.tmp','TMP':'.tmp','TMPDIR':'.tmp','WARP_CACHE_PATH':'.cache/warp','NEWTON_CACHE_PATH':'.cache/newton'}.items():
        folder=ROOT/path; folder.mkdir(parents=True,exist_ok=True); env[name]=str(folder)
    env['OPENBLAS_NUM_THREADS']='1'; env['OMP_NUM_THREADS']='1'
    started=time.monotonic(); low_since=None; proc=None; reason=None
    try:
        with (out/'worker.log').open('w',encoding='utf-8') as log, (out/'resources.jsonl').open('w',encoding='utf-8') as timeline:
            proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'diagnostics/resource_probe/worker.py'),'--config',str(config),'--output',str(out),'--mode',args.mode,'--steps',str(args.steps)],
                                  cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            result.update(worker_started=True,pid=proc.pid,status='running'); save()
            while proc.poll() is None:
                sample=snapshot(ROOT); now=time.monotonic()
                if sample['available_ram_bytes']<cfg['budget'].get('min_running_ram_gib',2)*1024**3:
                    if low_since is None: low_since=now
                else: low_since=None
                if proc.poll() is not None: break
                target_pid=monitored_pid(out/'worker_result.json',proc.pid)
                try: memory=process_memory(target_pid)
                except OSError:
                    if proc.poll() is not None: break
                    raise
                output_bytes=directory_bytes(output_scope)
                size=output_bytes+directory_bytes(ROOT/'.cache')+directory_bytes(ROOT/'.tmp')
                reason=stop_reason(sample,cfg['budget'],now-started,memory['rss_bytes'],size,0 if low_since is None else now-low_since,memory['private_commit_bytes'])
                if (archive_restore or native_repeatability or deterministic_import or deterministic_response or timing_capture) and reason is None and output_bytes>cfg['budget']['max_new_output_mib']*1024**2:
                    reason='new_output_budget'
                timeline.write(json.dumps({'elapsed_s':now-started,'monitored_pid':target_pid,**memory,'output_cache_bytes':size,**sample,'stop_reason':reason})+'\n'); timeline.flush()
                if reason: stop_owned_tree(proc); break
                time.sleep(1)
            proc.wait(timeout=15)
        worker_path=out/'worker_result.json'; worker=json.loads(worker_path.read_text()) if worker_path.exists() else None
        result.update(status='stopped' if reason else ('completed' if proc.returncode==0 else 'failed'),
                      stop_reason=reason,exit_code=proc.returncode,worker=worker,wall_s=time.monotonic()-started)
        save(); return 0 if result['status']=='completed' else 1
    except BaseException as exc:
        result.update(status='supervisor_failed',error=str(exc)); save(); raise
    finally:
        if proc is not None and proc.poll() is None: stop_owned_tree(proc); proc.wait(timeout=15)

if __name__=='__main__': sys.exit(main())
