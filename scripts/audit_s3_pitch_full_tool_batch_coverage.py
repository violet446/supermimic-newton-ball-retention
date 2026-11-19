"""Measure the change in direction and state coverage from the matched-tool batch."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from audit_s3_pitch_full_tool_coverage import WINDOW, aligned, pair_data
from audit_s3_world_increment_drift import require_close
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool
from deterministic_archive_metrics import _full_branch
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'pitch_full_tool_batch_coverage'


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    def accepted(path, digest):
        if plan['files_sha256'][Path(path).as_posix()] != digest:
            raise ValueError(f'archive identity changed: {path}')
        return read(path)

    held_design = read(plan['held_design'])
    tool, cfg, state, traces, status = read_held(held_design)
    if plan['updates'] or tool['case']['tick'] != 168:
        raise ValueError('unexpected offline scope')
    mesh = load_tool(ROOT/'assets/shovel.STL')
    folder = Path(held_design['held_timing_run']).parent
    labels = tool['branch_order']
    dev_raw = {k: accepted(folder/'worker'/f'{k}.json',status['artifacts_sha256'][f'worker/{k}.json']) for k in labels}
    layout = accepted(folder/'worker/native_layout.json',status['artifacts_sha256']['worker/native_layout.json'])
    dev = pair_data(tool,_parse_state(state),cfg,traces,dev_raw,layout,mesh)

    def cohort(manifests, matched):
        items = {}
        for manifest in manifests:
            p=Path(manifest)
            base=p.parent
            st=read(base/'validation.json')
            if st['status']!='completed' or st['test_episode_files_read'] is not False:
                raise ValueError('unaccepted training capture')
            if matched and (st['worker_starts'],st['physics_steps'],st['branches'],st['pairs'])!=(1,6784,8,3):
                raise ValueError('new training capture exceeds scope')
            for entry in accepted(p,st['artifacts_sha256'][p.name])['entries']:
                if (entry['split']!='train' or entry['snapshot_tick']!=168 or entry['episode_id'] in items
                        or entry['group_id']==tool['case']['group']):
                    raise ValueError('unstable training identity, clock or split')
                refs=[*entry['nominal_branches'],*(b for pair in entry['pairs'] for b in pair['branches'])]
                raw={b['label']:accepted(base/b['path'],b['sha256']) for b in refs}
                nominal=entry['native_snapshot']
                state=_parse_state(accepted(base/nominal['path'],nominal['sha256'])['state'])
                layout=(Path(nominal['path']).parent/'native_layout.json').as_posix()
                native_layout=accepted(base/layout,st['artifacts_sha256'][layout])
                config=accepted(entry['source_config'],entry['source_config_sha256'])
                traces={k:_full_branch(v,168) for k,v in raw.items()}
                if matched and any(raw[k]['commands']!=dev_raw[k]['commands'] or raw[k]['actions']!=dev_raw[k]['actions'] for k in labels):
                    raise ValueError('matched training commands differ from development')
                items[entry['episode_id']] = (entry['group_id'],pair_data(tool,state,config,traces,raw,native_layout,mesh))
                guard(entry['episode_id'],log=True)
        return items

    old = cohort(plan['baseline_manifests'],False)
    new = cohort(plan['batch_manifests'],True)
    if (len(old)!=5 or not 1<=len(new)<=2 or set(new)&set(old)
            or set(new)-set(plan['approved_episodes'])
            or set(x[0] for x in new.values())&set(x[0] for x in old.values())):
        raise ValueError('unexpected independent training groups')
    benchmark=read(plan['baseline_coverage'])['result']
    scales=np.asarray(read(plan['context_reference'])['scales']['ball'])
    results={}
    for action,target in dev.items():
        signal=np.linalg.norm(target['label'],axis=-1)
        lead=np.flatnonzero((signal>=10*target['floor'])&(np.arange(len(signal))>=8)&(np.arange(len(signal))<16))
        old_cosine=np.max(np.stack([aligned(row[action],target,lead) for _,row in old.values()]),axis=0)
        archived=benchmark[action]
        expected=np.maximum(archived['direction_cosine_best_previous_by_lead'],archived['direction_cosine_new_by_lead'])
        require_close(old_cosine,expected)
        np.testing.assert_array_equal(lead+1,archived['development_strong_leads'])
        candidate_cosine={name:aligned(row[action],target,lead) for name,(_,row) in new.items()}
        best=np.maximum(old_cosine,np.max(np.stack(list(candidate_cosine.values())),axis=0))

        def distance(row):
            return np.sqrt(np.mean(((row['start']-target['start'])/scales)**2,axis=(0,2)))

        prior=np.min(np.stack([distance(row[action]) for _,row in old.values()]),axis=0)
        rows={}
        for name,(group,blob) in new.items():
            row=blob[action]
            norm=float(np.linalg.norm(row['position']))
            dev_norm=float(np.linalg.norm(target['position']))
            rows[name]={'group':group,'cosine_by_development_strong_lead':candidate_cosine[name].tolist(),
                'new_alignment_leads':(lead[(old_cosine<.8)&(candidate_cosine[name]>=.8)]+1).tolist(),
                'strong_label_count':int(np.sum(np.linalg.norm(row['label'][WINDOW],axis=-1)>=10*row['floor'][WINDOW])),
                'ball_nearest_leads':(lead[distance(row)[lead-8]<prior[lead-8]]+1).tolist(),
                'candidate_count_range':row['candidate_range'],
                'h32_half_difference_norm_m':norm,
                'h32_position_cosine':float(row['position']@target['position']/(norm*dev_norm)) if norm*dev_norm>0 else None}
        results[action]={'development_leads':(lead+1).tolist(),'aligned_before':int(np.sum(old_cosine>=.8)),
            'aligned_after':int(np.sum(best>=.8)),'new_alignment_leads':(lead[(old_cosine<.8)&(best>=.8)]+1).tolist(),
            'development_candidate_range':target['candidate_range'],
            'development_h32_position_half_norm_m':float(np.linalg.norm(target['position'])),'by_case':rows}
    write_json(out/'coverage.json',{'before_training_snapshots':len(old),
        'after_training_snapshots':len(old)+len(new),'results':results})
    return {'accepted_batch_cases':len(new),'baseline_direction_scores_reproduced':True,
            'batch_tool_commands_exact':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
