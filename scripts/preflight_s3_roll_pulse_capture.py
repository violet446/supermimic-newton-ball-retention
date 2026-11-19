"""Select one archived training snapshot for a possible pure-roll pulse replay."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from ball_retention.model_data import _ball_state
from ball_retention.pair_collection import _initial_key
from check_s3_actions import schedule_actions
from deterministic_archive_metrics import CONTROL_FIELDS, _full_branch
from preflight_s3_deterministic_action_response import action_sequences, verify_commands
from preflight_s3_training_timing_capture import sequence_digest
from response_contact_diagnosis import _allowed_pairs, _decode_trace
from saved_native_snapshot import decode_snapshot
from train_s3_local_velocity_increment import bound


PHASE = 'roll_pulse_capture_preflight'
NOMINAL = ('nominal_1', 'nominal_2')
ROLL = ('roll_component_plus', 'roll_component_minus')
DUAL = ('dual_axis_pulse_plus', 'dual_axis_pulse_minus')
PITCH = ('pitch_pulse_plus', 'pitch_pulse_minus', 'pitch_reversal_plus', 'pitch_reversal_minus')
LEADS = (15, 16)


def read(plan, path):
    return json.loads(bound(plan, path))


def accepted(plan, base, relative, status):
    path = base/relative
    if plan['files_sha256'][path.as_posix()] != status['artifacts_sha256'][relative]:
        raise ValueError(f'accepted archive digest differs: {path}')
    return read(plan, path)


def trace_fields(trace, name, layout, tick, substeps):
    states = _full_branch(trace, tick)
    contacts = _decode_trace(trace, name, 32, substeps, tick*substeps, 1/960, _allowed_pairs(layout))
    return states['raw'], contacts


def compare_context(training, development, scales, substeps):
    """Compare archived ball states and preceding intervals for both target leads."""
    train_ball, train_contacts = training
    dev_ball, dev_contacts = development
    report = {}
    for lead in LEADS:
        # The endpoint of H(k-1) is the input to Hk. Candidate logs in
        # H(k-1) describe the interval leading up to that endpoint.
        previous = lead-2
        delta = train_ball[previous]-dev_ball[previous]
        block = slice(previous*substeps, (previous+1)*substeps)
        rows = list(zip(train_contacts[block], dev_contacts[block], strict=True))
        counts, reference = np.asarray([[a[0],b[0]] for a,b in rows]).T
        report[str(lead)] = {
            'training_ball_tool_si':train_ball[previous].tolist(),
            'development_ball_tool_si':dev_ball[previous].tolist(),
            'position_distance_m':float(np.linalg.norm(delta[:3])),
            'velocity_distance_m_s':float(np.linalg.norm(delta[3:6])),
            'spin_distance_rad_s':float(np.linalg.norm(delta[6:])),
            'scaled_ball_rms':float(np.sqrt(np.mean((delta/scales)**2))),
            'training_contact_candidate_range':[int(counts.min()),int(counts.max())],
            'development_contact_candidate_range':[int(reference.min()),int(reference.max())],
            'candidate_count_mae':float(np.mean(abs(counts-reference))),
            'shape_pair_disagreement_substeps':sum(a[1]!=b[1] for a,b in rows),
        }
    return report


def training_case(plan, manifest, dev, guard):
    root = Path(manifest).parent
    status = read(plan, root/'validation.json')
    if (status['status'],status['worker_starts'],status['snapshots'],status['physics_steps'],
            status['branches'],status['pairs'],status['training_updates'],status['test_episode_files_read'],
            status['control_accepted']) != ('completed',1,1,6784,8,3,0,False,False):
        raise ValueError('training source did not pass the full-tool capture gates')
    collection = accepted(plan, root, 'capture_manifest.json', status)
    entry, = collection['entries']
    case = status['case']
    capture, roll, dev_cfg, snapshot, layout, reference = dev
    tick, substeps, horizon = capture['case']['tick'], capture['control_substeps'], capture['horizon']
    config = read(plan, entry['source_config'])
    execution = accepted(plan, root, 'execution_plan.json', status)
    if (collection['horizon'] != horizon or collection['development_holdout_group'] != capture['case']['group']
            or execution['design']['runtime'] != capture['runtime']
            or entry['snapshot_tick'] != tick or any(case[key]!=entry[key] for key in case)
            or entry['split'] != 'train' or entry['group_id'] == capture['case']['group']
            or entry['seed'] != config['seed']
            or plan['files_sha256'][entry['source_config']] != entry['source_config_sha256']
            or _initial_key(config) != entry['physical_initial_sha256'] or _initial_key(config) == _initial_key(dev_cfg)
            or any(config[key]!=dev_cfg[key] for key in ('physics','ball','action','reference','robot','prescribed'))
            or any(config['initial'][key]!=dev_cfg['initial'][key] for key in
                   ('joint_q_rad','anchor_xy_tool_m','anchor_world_m','tool_rotation_world'))
            or config['action_pairs']['schedule'] != dev_cfg['action_pairs']['schedule']):
        raise ValueError('training split, scene, clock or full tool history changed')
    nominal_requests = schedule_actions(config['action_pairs']['schedule'], 0, tick+horizon)
    if sequence_digest(nominal_requests[:tick]) != entry['prefix_actions_sha256']:
        raise ValueError('archived training prefix requests differ from the matched history')

    native = entry['native_snapshot']
    if native['sha256'] != status['artifacts_sha256'][native['path']] or native['sha256'] != entry['native_snapshot_sha256']:
        raise ValueError('training native snapshot digest changed')
    raw = accepted(plan, root, native['path'], status)
    if accepted(plan, root, 'worker/native_layout.json', status) != layout:
        raise ValueError('training and development native layouts differ')
    saved = decode_snapshot(raw, layout, config, tick, substeps)
    if (saved['episode_id'] != entry['episode_id']
            or saved['controller'] != snapshot['controller']
            or raw['physics']['command'] != snapshot['physics']['command']
            or any(raw['state'][key]!=snapshot['state'][key] for key in CONTROL_FIELDS)):
        raise ValueError('training snapshot controller differs from the matched development snapshot')

    refs = [*entry['nominal_branches'], *(ref for pair in entry['pairs'] for ref in pair['branches'])]
    archives = {}
    for ref in refs:
        name = ref['label']
        if ref['sha256'] != status['artifacts_sha256'][ref['path']] or name in archives:
            raise ValueError('training branch identity changed')
        archives[name] = accepted(plan, root, ref['path'], status)
    if set(archives) != set((*NOMINAL,*PITCH,*DUAL)):
        raise ValueError('unexpected archived training branch inventory')
    if archives[NOMINAL[0]] != archives[NOMINAL[1]]:
        raise ValueError('archived training nominal repeats differ')

    _, requests = action_sequences(capture, nominal_requests[tick:])
    roll_design = {**capture, 'amplitude':roll['amplitude'], 'directions':{'roll_component':roll['segments']}}
    directions, roll_requests = action_sequences(roll_design, nominal_requests[tick:])
    requests.update(roll_requests)
    commands, peaks = verify_commands(ROOT, capture, config, saved, requests, reference[NOMINAL[0]]['commands'], guard)
    for name in requests:
        trace = reference[name]
        if requests[name].tolist()!=trace['actions'] or commands[name]!=trace['commands']:
            raise ValueError(f'training and development commands differ: {name}')
        if name in archives and (archives[name]['actions']!=trace['actions'] or archives[name]['commands']!=trace['commands']):
            raise ValueError(f'archived training tool history differs: {name}')
    if np.any(directions['roll_component'][8:]):
        raise ValueError('pure roll disturbance continues beyond H8')
    for name in ROLL:
        np.testing.assert_array_equal(requests[name][8:], requests[NOMINAL[0]][8:])

    train_proxies = {name:trace_fields(archives[name], name, layout, tick, substeps) for name in (*NOMINAL,*DUAL)}
    paths = {'source_manifest':manifest, 'source_validation':(root/'validation.json').as_posix(),
             'snapshot':(root/native['path']).as_posix(), 'layout':(root/'worker/native_layout.json').as_posix()}
    provenance = {name:{'path':path, 'sha256':plan['files_sha256'][path]} for name,path in paths.items()}
    future = {'case':case, 'source':provenance, 'snapshot_tick':tick, 'control_substeps':substeps,
              'horizon':horizon, 'runtime':capture['runtime'], 'branch_order':list((*NOMINAL,*ROLL)),
              'branches':{name:{'requested':requests[name].tolist(),'executed':commands[name]}
                          for name in (*NOMINAL,*ROLL)},
              'archived_nominal_references':entry['nominal_branches'],
              'peak_commands':{name:peaks[name] for name in (*NOMINAL,*ROLL)}}
    return future, train_proxies, _ball_state(saved['state'])


def run(plan, out, guard):
    if plan['updates'] != 0 or len(plan['training_manifests']) != 3:
        raise ValueError('unexpected static scope')
    launch = read(plan, plan['small_capture_plan'])
    root = Path(launch['output'])
    status = read(plan, root/'validation.json')
    if (status['status'],status['physics_steps'],status['training_updates'],status['test_episode_files_read'],
            status['control_accepted'],status['nominal_gate_passed']) != ('completed',7168,0,False,False,True) or (
            status['plan_sha256'] != plan['files_sha256'][plan['small_capture_plan']]):
        raise ValueError('small-action development replay was not accepted')
    static_path = Path(launch['preflight_validation'])
    static = read(plan, static_path)
    if static['status'] != 'completed' or static['plan_sha256'] != plan['files_sha256'][launch['preflight_plan']]:
        raise ValueError('small-action operating point was not accepted')
    spec = accepted(plan, static_path.parent, 'replay_spec.json', static)
    archive = read(plan, plan['development_archive_plan'])
    original = read(plan, launch['original_response_plan'])
    setup = accepted(plan, root, 'execution_plan.json', status)
    if setup['plan'] != launch or setup['plan_sha256'] != status['plan_sha256']:
        raise ValueError('development worker configuration has changed')
    cfg = setup['worker_config']
    tick, n, horizon = spec['case']['tick'], spec['substeps_per_control'], spec['horizon']
    if spec['case'] != status['case'] or spec['case'] != original['case'] or (tick,n,horizon)!=(168,16,32):
        raise ValueError('unexpected development clock or identity')
    snapshot, layout = (read(plan, archive['native_reference'][name]) for name in ('snapshot','layout'))
    dev_state = decode_snapshot(snapshot, layout, cfg, tick, n)['state']
    reference = {name:accepted(plan, root, f'worker/{name}.json', status) for name in (*NOMINAL,*ROLL)}
    full_path = Path(launch['archived_response_validation'])
    full = read(plan, full_path)
    if full['status'] != 'completed' or not full['execution_gates_passed'] or full['case'] != spec['case']:
        raise ValueError('full-amplitude context was not accepted')
    reference.update({name:accepted(plan, full_path.parent, f'worker/{name}.json', full) for name in (*PITCH,*DUAL)})

    roll = spec['cases']['roll_component']
    if roll['segments'] != [{'start':0,'stop':8,'axes':[1.,0.]}] or not (0.14 < roll['amplitude'] < 0.15):
        raise ValueError('unexpected roll request shape or amplitude')
    profiles = {name:trace_fields(reference[name],name,layout,tick,n) for name in (*NOMINAL,*DUAL,*ROLL)}
    scales = np.asarray(read(plan, plan['context_reference'])['scales']['ball'])
    if scales.shape != (9,) or not np.all(np.isfinite(scales)&(scales>0.)):
        raise ValueError('invalid frozen ball scales')

    dev = (original, roll, cfg, snapshot, layout, reference)
    rows, commands = {}, {}
    for manifest in plan['training_manifests']:
        future, proxy, start = training_case(plan, manifest, dev, guard)
        name = future['case']['episode_id']
        commands[name] = future
        matches = {key:compare_context(proxy[key],profiles[key],scales,n) for key in (NOMINAL[0],*DUAL)}
        target = {roll_name:{key:compare_context(proxy[key],profiles[roll_name],scales,n)
                             for key in (NOMINAL[0],dual_name)} for roll_name,dual_name in zip(ROLL,DUAL,strict=True)}
        rows[name] = {'group_id':future['case']['group_id'],'seed':future['case']['seed'],
                      'static_branch_count':len(future['branch_order']),
                      'initial_position_distance_m':float(np.linalg.norm(start[:3]-_ball_state(dev_state)[:3])),
                      'like_for_like':matches,'proxies_vs_development_roll':target,
                      'target_proxy_mean_ball_rms':float(np.mean([x['scaled_ball_rms'] for by_proxy in target.values()
                            for by_lead in by_proxy.values() for x in by_lead.values()])),
                      'target_proxy_mean_contact_count_mae':float(np.mean([x['candidate_count_mae'] for by_proxy in target.values()
                            for by_lead in by_proxy.values() for x in by_lead.values()]))}
        guard(name, log=True)
    if len({row['group_id'] for row in rows.values()})!=3 or len({row['case']['physical_initial_sha256'] for row in commands.values()})!=3:
        raise ValueError('training sources do not comprise three independent initial states')
    ranked = sorted(rows, key=lambda name:(rows[name]['target_proxy_mean_ball_rms'],
                                           rows[name]['target_proxy_mean_contact_count_mae'], name))
    residual_tilt = {str(lead): (np.asarray(reference[ROLL[0]]['states'][lead-1]['tilt_rad'])-
                                 reference[ROLL[1]]['states'][lead-1]['tilt_rad']).tolist()
                     for lead in (8,9,14,15,16,24,32)}
    write_json(out/'candidate_coverage.json', {'development_case':spec['case'], 'roll_amplitude':roll['amplitude'],
        'signed_pulse_leads':[1,8], 'ball_scales_si':scales.tolist(), 'target_leads':LEADS,
        'tool_commands_equal_through_H32':True, 'nominal_requests_equal_after_H8':True,
        'roll_plus_minus_tilt_rad_from_development_archive':residual_tilt, 'candidates':rows,
        'ranked_by_proxy_ball_distance':ranked, 'selected_episode':ranked[0],
        'newton_roll_states_measured_in_this_run':False})
    write_json(out/'selected_commands.json', commands[ranked[0]])
    write_json(out/'candidate_commands.json', commands)
    return {'candidate_groups':len(rows), 'static_branches':sum(x['static_branch_count'] for x in rows.values()),
            'selected_episode':ranked[0], 'newton_workers':0, 'new_ball_trajectories':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
