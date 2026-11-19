"""Bind the new capture seed to its immutable, bounded task-local design."""
import hashlib
import json
import numpy as np


DESIGN='configs/s3_history_coverage_plan.json'


def validate_history_case(root,cfg):
    raw=(root/DESIGN).read_bytes();design=json.loads(raw)
    binding=cfg.get('history_capture',{})
    if binding.get('design_sha256')!=hashlib.sha256(raw).hexdigest():
        raise ValueError('history design hash mismatch')
    cases=[e for e in design['episodes'] if e['episode_id']==binding.get('episode_id')]
    if len(cases)!=1:raise ValueError('history episode is not declared')
    case=cases[0];group=next(g for g in design['groups'] if g['group_id']==case['group_id'])
    spec=cfg['action_pairs']
    if (cfg['seed']!=case['seed'] or spec['master_seed']!=design['master_seed']
        or binding.get('group_id')!=case['group_id'] or binding.get('split')!=case['split']
        or spec['schedule']!=design['histories'][case['history_id']]
        or spec['snapshot_ticks']!=design['snapshot_ticks'] or spec['horizon']!=design['horizon']
        or spec['amplitudes']!=design['amplitudes']):
        raise ValueError('history case differs from frozen design')
    base_path=root/'configs/s1_online_episode.json';base_raw=base_path.read_bytes()
    if hashlib.sha256(base_raw).hexdigest()!=design['inputs_sha256']['configs/s1_online_episode.json']:
        raise ValueError('history base config changed')
    base=json.loads(base_raw)
    expected=dict(base['initial'])
    expected['ball_center_world_m']=(np.array(expected['ball_center_world_m'])+group['ball_offset_world_m']).tolist()
    expected['ball_velocity_world_m_s']=design['base_initial_velocity_world_m_s']
    if cfg['initial']!=expected:raise ValueError('history initial state changed')
    for key in ('physics','ball','action','robot','reference','prescribed','admission'):
        if cfg[key]!=base[key]:raise ValueError('history physical/control configuration changed')
    budget=cfg['budget'];limits=design['next_capture_budget']
    if budget['max_output_cache_gib']!=base['budget']['max_output_cache_gib']:
        raise ValueError('shared-cache guard must retain the base cap; capture output has a separate cap')
    for key,limit in [('wall_s',limits['max_worker_wall_s']),('max_physics_steps',limits['max_physics_substeps']),
                      ('max_trace_steps',design['reserved_substeps_per_episode']),
                      ('max_process_rss_gib',limits['max_process_rss_gib']),
                      ('max_process_commit_gib',limits['max_process_commit_gib']),('max_gpu_used_gib',limits['max_gpu_used_gib'])]:
        if budget[key]>limit:raise ValueError('history resource scope expanded')
    return case
