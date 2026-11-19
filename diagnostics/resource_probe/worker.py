"""Internal worker launched by check_physics.py after resource admission."""
from pathlib import Path
import argparse,json,sys,time,traceback,os
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from ball_retention.native_repeatability_plan import MARKER as NATIVE_MARKER, MODE as NATIVE_MODE
from ball_retention.deterministic_archive_import_plan import MARKER as IMPORT_MARKER, MODE as IMPORT_MODE
from ball_retention.deterministic_response_plan import MARKER as RESPONSE_MARKER, MODE as RESPONSE_MODE
from ball_retention.training_timing_capture_plan import MARKER as TIMING_MARKER, MODE as TIMING_MODE

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['initialize','fixed-tool','tilt-plus','tilt-minus','drop','fk','arm-static','arm-move','arm-episode','s2-collect','s3-pairs','s3-archive-restore',NATIVE_MODE,IMPORT_MODE,RESPONSE_MODE,TIMING_MODE],required=True)
    parser.add_argument('--steps',type=int,default=240); args=parser.parse_args()
    out=Path(args.output); cfg=json.loads(Path(args.config).read_text()); began=time.monotonic()
    summary={'mode':args.mode,'status':'starting','s1_complete':False,'physics_steps':0,'worker_pid':os.getpid()}
    def save(): (out/'worker_result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    def stage(name):
        from ball_retention.guard import process_memory
        event={'stage':name,'elapsed_s':time.monotonic()-began,**process_memory(os.getpid())}
        summary.setdefault('phase_events',[]).append(event); save()
    save()
    try:
        from ball_retention.resources import snapshot,admission
        if not admission(snapshot(ROOT),cfg['budget'])['passed']: raise RuntimeError('worker resource recheck failed')
        if not 1<=args.steps<=cfg['budget'].get('max_trace_steps',1920): raise ValueError('outside configured trace step budget')
        if 'small_offset_probe' in cfg and args.mode != 's3-pairs':
            raise ValueError('small-offset case requires s3-pairs mode')
        if 'archive_restore_pilot' in cfg and args.mode != 's3-archive-restore':
            raise ValueError('archive restoration case requires s3-archive-restore mode')
        if NATIVE_MARKER in cfg and args.mode != NATIVE_MODE:
            raise ValueError('native repeatability case requires s3-native-repeatability mode')
        if IMPORT_MARKER in cfg and args.mode != IMPORT_MODE:
            raise ValueError('deterministic archive case requires s3-deterministic-archive-import mode')
        if RESPONSE_MARKER in cfg and args.mode != RESPONSE_MODE:
            raise ValueError('deterministic response case requires s3-deterministic-action-response mode')
        if TIMING_MARKER in cfg and args.mode != TIMING_MODE:
            raise ValueError('training timing case requires s3-training-timing-capture mode')
        if args.mode=='fk':
            from fk_crosscheck import crosscheck
            summary.update(crosscheck(ROOT,cfg)); summary['status']='fk_passed'
        elif args.mode=='arm-episode':
            from episode_smoke import run
            run(ROOT,cfg,out,args.steps,summary,stage)
        elif args.mode=='s2-collect':
            from ball_retention.collection import collect_episode
            collect_episode(ROOT,cfg,out,args.steps,summary,stage)
        elif args.mode=='s3-pairs':
            from action_pairs import run
            run(ROOT,cfg,out,args.steps,summary,stage)
        elif args.mode=='s3-archive-restore':
            from archive_restore import run
            run(ROOT,cfg,out,args.steps,summary,stage)
        elif args.mode==NATIVE_MODE:
            from native_repeatability import run
            run(ROOT,cfg,out,args.steps,summary,stage)
        elif args.mode==IMPORT_MODE:
            from deterministic_archive_import import run
            run(ROOT,cfg,out,args.steps,summary,stage)
        elif args.mode==RESPONSE_MODE:
            from deterministic_action_response import run
            run(ROOT,cfg,out,args.steps,summary,stage)
        elif args.mode==TIMING_MODE:
            from training_timing_capture import run
            run(ROOT,cfg,out,args.steps,summary,stage)
        else:
            stage('before_probe_module')
            if args.mode.startswith('arm-'):
                from ball_retention.env import PrescribedArmEnv
                probe=PrescribedArmEnv(ROOT,cfg,'cuda:0',moving=args.mode=='arm-move',report_stage=stage)
                summary['assembly']=probe.assembly
            else:
                from probe import FixedToolProbe
                probe=FixedToolProbe(ROOT,cfg,'cuda:0',report_stage=stage)
            if args.mode=='drop': probe.place_ball_outside()
            def advance():
                if args.mode in ('tilt-plus','tilt-minus'):
                    probe.prescribe_tilt((probe.steps+1)*cfg['physics']['dt_s'],1 if args.mode=='tilt-plus' else -1)
                return probe.step()
            summary['initialization_wall_s']=time.monotonic()-began
            summary['phase_times_s']=getattr(probe,'phase_times_s',{})
            summary['effective_material']=getattr(probe,'mass_inertia',None)
            summary['initial_state']=probe.read(); save()
            departure=None
            if 'departure_monitor' in cfg:
                from ball_retention.retention import ToolEnvelope,DepartureMonitor
                spec=cfg['departure_monitor']
                departure=DepartureMonitor(ToolEnvelope(probe.mesh,cfg['initial']['tool_rotation_world'],cfg['ball']['radius_m'],spec['margin_m']),spec['outside_grace_s'])
                summary['initial_departure']=departure.update(0,summary['initial_state']['ball_local_m'])
                summary['departure_monitor_scope']='diagnostic labels only; not yet an integrated S2 termination/reset interface'
            if args.mode in ('fixed-tool','tilt-plus','tilt-minus','drop','arm-static','arm-move'):
                replay_saved=None
                with (out/'trace.jsonl').open('w',encoding='utf-8') as stream:
                    deep_count=0
                    stage('before_first_step')
                    for _ in range(args.steps):
                        row=advance(); summary['physics_steps']+=1
                        if args.mode=='arm-move' and summary['physics_steps']==args.steps//2:
                            replay_saved=probe.snapshot()
                        if summary['physics_steps']==1: stage('first_step_complete')
                        if departure is not None:
                            row['departure']=departure.update(row['time_s'],row['ball_local_m'],row['floor_reached'])
                        stream.write(json.dumps(row)+'\n'); stream.flush()
                        deep_count=deep_count+1 if row['surface_clearance_m']<-.002 else 0
                        if deep_count>=3: raise RuntimeError('deep penetration for three substeps')
                        if departure is not None and row['departure']['terminated']:
                            if cfg['departure_monitor'].get('expected_departure',False):
                                summary['expected_departure_detected']=row['departure']; summary['departure_time_s']=row['time_s']; break
                            raise RuntimeError('unexpected tool departure')
                        if row['floor_reached']:
                            if args.mode=='drop':
                                summary['expected_drop_detected']=True; break
                            raise RuntimeError('ball reached floor in retention diagnostic')
                if args.mode=='drop' and not summary.get('expected_drop_detected'): raise RuntimeError('expected floor/drop not detected')
                if departure is not None and cfg['departure_monitor'].get('expected_departure',False) and not summary.get('expected_departure_detected'):
                    raise RuntimeError('expected departure not detected within bounded trace')
                summary['final_trace_state']=probe.read()
                saved=replay_saved if replay_saved is not None else probe.snapshot()
                summary['replay_start_step']=saved['steps']
                # Moving-arm replay must exercise the command clock during motion,
                # rather than only the stationary hold after the path has finished.
                if replay_saved is not None: probe.restore(saved)
                first=[]
                for _ in range(16): first.append(advance()); summary['physics_steps']+=1
                probe.restore(saved); second=[]
                for _ in range(16): second.append(advance()); summary['physics_steps']+=1
                import numpy as np
                a=np.asarray([r['body_q'] for r in first]); b=np.asarray([r['body_q'] for r in second])
                summary['replay_position_error_m']=float(abs(a[:,:,:3]-b[:,:,:3]).max())
                summary['replay_quaternion_component_error']=float(abs(a[:,:,3:]-b[:,:,3:]).max())
                summary['replay_velocity_error']=float(abs(np.asarray([r['body_qd_com_world'] for r in first])-np.asarray([r['body_qd_com_world'] for r in second])).max())
                reset=probe.reset(); initial=summary['initial_state']
                summary['reset_max_error']=float(max(abs(np.asarray(reset[k])-initial[k]).max() for k in ('body_q','body_qd_com_world')))
                if summary['replay_position_error_m']>1e-5 or summary['replay_quaternion_component_error']>1e-5 or summary['replay_velocity_error']>1e-5 or summary['reset_max_error']>1e-6:
                    raise RuntimeError('restore/reset mismatch')
            summary['status']='diagnostic_completed'
        stage('finished')
        summary['wall_s']=time.monotonic()-began; save()
    except Exception as exc:
        summary.update(status='failed',error=str(exc),wall_s=time.monotonic()-began); save()
        traceback.print_exc(); return 1
    return 0

if __name__=='__main__': sys.exit(main())
