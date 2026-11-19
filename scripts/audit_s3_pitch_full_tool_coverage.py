"""Audit the matched-tool capture using measured increments from approved replays."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np

from audit_s3_ball_velocity_components import frame_series
from audit_s3_world_increment_drift import half, require_close
from audit_s3_world_increment_generation import prior
from ball_retention.archive_state_import import _parse_state
from ball_retention.geometry import load_tool
from ball_retention.model_data import _ball_state
from ball_retention.surface_diagnostic import SurfaceDiagnostic
from deterministic_archive_metrics import _full_branch
from preflight_s3_world_increment_supervision import SETTINGS
from response_contact_diagnosis import diagnose_contacts
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import read_held


PHASE = 'pitch_full_tool_coverage'
WINDOW = slice(8,16)


def pair_data(capture, state, cfg, traces, raw, layout, mesh):
    dt = 1/cfg['physics']['control_hz']
    data = {}

    def increments(k):
        return np.diff(frame_series(state,traces[k],dt)[3],axis=0)

    nominal = np.linalg.norm(half(np.stack([increments(k) for k in ('nominal_1','nominal_2')])),axis=-1)
    floor = np.maximum(SETTINGS['floor_m_s'], SETTINGS['noise_multiplier']*nominal)
    settings = {**capture,'case':{'tick':168}}
    contacts = diagnose_contacts(settings,raw,layout)
    surface = SurfaceDiagnostic(mesh,cfg['ball']['radius_m'],cfg['initial']['tool_rotation_world'])
    for action in capture['directions']:
        keys = [f'{action}_{sign}' for sign in ('plus','minus')]
        label = half(np.stack([increments(k) for k in keys]))
        truth = np.stack([traces[k]['raw'] for k in keys])
        start = prior(_ball_state(state),truth)[:,WINDOW]
        geom = [[surface.measure(row[:3]) for row in branch] for branch in start]
        counts = [x for key in keys for row in contacts['branches'][key]['leads'][7:15] for x in row['candidates_by_substep']]
        data[action] = {'label':label, 'floor':floor, 'position':half(truth[:,-1,:3]),'start':start,
            'geometry':np.asarray([[(*r['normal_tool'],r['sphere_distance_gap_m']) for r in branch] for branch in geom]),
            'candidate_range':[min(counts),max(counts)],
            'branch_count_disagreement':[x['candidate_count_different_substeps']
                for x in contacts['comparisons'][action]['plus_vs_minus']['leads'][WINDOW]]}
    return data


def aligned(candidate, target, use):
    x, y = candidate['label'][use], target['label'][use]
    nx = np.linalg.norm(x,axis=-1)
    den = nx*np.linalg.norm(y,axis=-1)
    resolved = nx >= candidate['floor'][use]
    return np.divide(np.sum(x*y,axis=-1),den,out=np.full(len(x),-1.),where=resolved & (den>0.))


def run(plan,out,guard):
    def read(p):
        return json.loads(bound(plan,p))

    def artifact(path,digest):
        if plan['files_sha256'][Path(path).as_posix()] != digest:
            raise ValueError(f'archive digest changed: {path}')
        return read(path)

    held_plan=read(plan['held_design'])
    capture,cfg,fields,dev_traces,dev_v=read_held(held_plan)
    dev_folder=Path(held_plan['held_timing_run']).parent
    labels=capture['branch_order']
    if capture['case']['tick']!=168 or plan['updates']!=0:
        raise ValueError('unexpected audit scope')
    mesh=load_tool(ROOT/'assets/shovel.STL')
    dev_raw={k:artifact(dev_folder/'worker'/f'{k}.json',dev_v['artifacts_sha256'][f'worker/{k}.json']) for k in labels}
    dev_layout=artifact(dev_folder/'worker/native_layout.json',dev_v['artifacts_sha256']['worker/native_layout.json'])
    dev=pair_data(capture,_parse_state(fields),cfg,dev_traces,dev_raw,dev_layout,mesh)
    training={}
    for manifest in [*plan['baseline_manifests'],plan['new_manifest']]:
        path=Path(manifest)
        root=path.parent
        status=read(root/'validation.json')
        if status['status']!='completed' or status['test_episode_files_read'] is not False:
            raise ValueError('unaccepted training capture')
        for e in artifact(path,status['artifacts_sha256'][path.name])['entries']:
            name=e['episode_id']
            if (name in training or e['split']!='train' or e['snapshot_tick']!=168
                    or e['group_id']==capture['case']['group']):
                raise ValueError('unexpected training identity')
            refs=[*e['nominal_branches'],*(b for p in e['pairs'] for b in p['branches'])]
            raw={b['label']:artifact(root/b['path'],b['sha256']) for b in refs}
            traces={k:_full_branch(v,168) for k,v in raw.items()}
            saved=e['native_snapshot']
            start=_parse_state(artifact(root/saved['path'],saved['sha256'])['state'])
            layout_rel=(Path(saved['path']).parent/'native_layout.json').as_posix()
            layout=artifact(root/layout_rel,status['artifacts_sha256'][layout_rel])
            case_cfg=artifact(e['source_config'],e['source_config_sha256'])
            training[name]=pair_data(capture,start,case_cfg,traces,raw,layout,mesh)
            if manifest==plan['new_manifest']:
                if (status['worker_starts'],status['physics_steps'],status['pairs'],status['branches'])!=(1,6784,3,8):
                    raise ValueError('supplement exceeded or missed capture scope')
                if any(raw[k]['commands']!=dev_raw[k]['commands'] or raw[k]['actions']!=dev_raw[k]['actions'] for k in labels):
                    raise ValueError('actual new tool commands differ from the development match')
            guard(name,log=True)
    new=training.pop(plan['new_episode'])
    if len(training)!=4:
        raise ValueError('unexpected baseline size')
    late=read(plan['late_reference'])
    scales=np.asarray(read(plan['context_reference'])['scales']['ball'])
    result={}
    for action,reference in dev.items():
        require_close(reference['label'],np.asarray(late['development'][action]['target_m_s']))
        zz=np.linalg.norm(reference['label'],axis=-1)
        leads=np.flatnonzero((np.arange(len(zz))>=8)&(np.arange(len(zz))<16)&(zz>=10*reference['floor']))
        before=np.max(np.stack([aligned(v[action],reference,leads) for v in training.values()]),axis=0)
        extra=new[action]
        after=aligned(extra,reference,leads)
        distance=lambda x: np.sqrt(np.mean(((x['start']-reference['start'])/scales)**2,axis=(0,2)))
        existing=np.min(np.stack([distance(v[action]) for v in training.values()]),axis=0)
        p,q=extra['position'],reference['position']
        p_norm, q_norm = np.linalg.norm(p), np.linalg.norm(q)
        result[action]={'development_strong_leads':(leads+1).tolist(),
            'direction_cosine_best_previous_by_lead':before.tolist(),'direction_cosine_new_by_lead':after.tolist(),
            'aligned_before':int(np.sum(before>=.8)),'aligned_after':int(np.sum(np.maximum(before,after)>=.8)),
            'newly_aligned_leads':(leads[(before<.8)&(after>=.8)]+1).tolist(),
            'new_strong_in_full_window':int(np.sum(np.linalg.norm(extra['label'][WINDOW],axis=-1)>=10*extra['floor'][WINDOW])),
            'development_candidate_range':reference['candidate_range'],'new_candidate_range':extra['candidate_range'],
            'existing_candidate_ranges':{k:v[action]['candidate_range'] for k,v in training.items()},
            'new_ball_block_distance_m_s_rad_s_median':np.median(
                np.linalg.norm((extra['start']-reference['start']).reshape(2,8,3,3),axis=-1),axis=(0,1)).tolist(),
            'new_ball_nearest_strong_leads':(leads[distance(extra)[leads-8]<existing[leads-8]]+1).tolist(),
            'median_geometry_normal_angle_deg':float(np.median(np.rad2deg(np.arccos(np.clip(
                np.sum(extra['geometry'][...,:3]*reference['geometry'][...,:3],axis=-1),-1.,1.))))),
            'new_h32_position_half_norm_m':float(p_norm),
            'development_h32_position_half_norm_m':float(q_norm),
            'h32_position_direction_cosine':float(p@q/(p_norm*q_norm)) if p_norm > 0 and q_norm > 0 else None}
    write_json(out/'coverage.json',{'result':result,'actual_tool_commands_exact':True})
    return {'old_training_snapshots':4,'new_training_snapshots':5,'actual_tool_commands_exact':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
