"""Score frozen models on matched-tool captures absent from their training."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

from audit_s3_world_increment_pitch_drift import load_extended
from audit_s3_world_increment_supervision_drift import load_endpoint
from ball_retention.geometry import load_tool, support_point
from deterministic_archive_metrics import _full_branch
from train_s3_local_velocity_increment import bound
from train_s3_position_integral import primary, read_held
from train_s3_world_increment import score_candidate


PHASE = 'pitch_full_tool_unseen_models'


def run(plan, out, guard):
    def read(p):
        return json.loads(bound(plan, p))

    def verify(p, expected):
        if plan['files_sha256'][Path(p).as_posix()] != expected:
            raise ValueError('archive reference changed')
        return read(p)

    held=read_held(read(plan['held_design']))
    capture,cfg,_,_,_=held
    mesh=load_tool(ROOT/'assets/shovel.STL')
    anchor=support_point(mesh,cfg['initial']['anchor_xy_tool_m'])[0]
    models={k:load_endpoint(plan,k,held,anchor,guard) for k in ('reference','supervised')}
    models['extended']=load_extended(plan,held,anchor,guard)
    if plan['updates'] or len({p[-1] for p in models.values()})!=1:
        raise ValueError('unexpected fixed-model comparison')
    results={}
    for path in plan['unseen_manifests']:
        file=Path(path)
        run=file.parent
        status=read(run/'validation.json')
        if (status['status']!='completed' or status['physics_steps']!=6784 or status['pairs']!=3
                or status['test_episode_files_read'] is not False):
            raise ValueError('unaccepted matched-tool trajectory')
        item,=verify(file,status['artifacts_sha256'][file.name])['entries']
        if (item['split']!='train' or item['group_id']==capture['case']['group']
                or item['episode_id'] in results or item['snapshot_tick']!=168):
            raise ValueError('case was not an independent matched-tool training initial state')
        snap=item['native_snapshot']
        initial=verify(run/snap['path'],snap['sha256'])['state']
        config=verify(item['source_config'],item['source_config_sha256'])
        refs=[*item['nominal_branches'],*(b for pair in item['pairs'] for b in pair['branches'])]
        traces={b['label']:_full_branch(verify(run/b['path'],b['sha256']),168) for b in refs}
        local_anchor=support_point(mesh,config['initial']['anchor_xy_tool_m'])[0]
        results[item['episode_id']]={'group':item['group_id'],'models':{}}
        for name,(model,fold,*_) in models.items():
            scores, forecasts=score_candidate(model,(capture,config,initial,traces,status),fold,local_anchor,guard)
            directions=primary(scores)
            finite_pass=sum(row['model_fd_vs_newton']['agrees'] is True for row in directions.values())
            paired_pass=sum(row['model_fd_vs_newton']['agrees'] is True and row['model_jvp_vs_newton']['agrees'] is True
                            for row in directions.values())
            reports={action:{'h32_position':row,
                             'h32_measured_position_half_norm_m':scores[action]['by_lead']['32']['position']['newton_half_difference_norm'],
                             'max_action_jvp_fd_difference':forecasts['pairs'][action]['max_jvp_fd_diff']}
                     for action,row in directions.items()}
            results[item['episode_id']]['models'][name]={'finite_pass':finite_pass,'joint_pass':paired_pass,'actions':reports}
            guard(name,log=True)
    if len(results)<2 or len({r['group'] for r in results.values()})!=len(results):
        raise ValueError('insufficient independent matched-tool snapshots')
    write_json(out/'scores.json',{'results':results,
        'scope':'Only fixed pre-capture checkpoints; these new training-side trajectories were not used for their fitting.'})
    return {'unseen_training_snapshots':len(results),'frozen_models':len(models),'action_derivative_checks_passed':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
