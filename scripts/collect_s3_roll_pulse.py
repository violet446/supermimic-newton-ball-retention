"""Replay one training snapshot and compare its pure-roll response to development."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import collect_s3_small_action_response as collector
import numpy as np

from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.pair_collection import _initial_key
from deterministic_archive_metrics import _full_branch
from response_contact_diagnosis import diagnose_contacts
from saved_native_snapshot import decode_snapshot
import small_action_response


ROOT = collector.ROOT
NOMINAL = ('nominal_1', 'nominal_2')
ROLL = ('roll_component_plus', 'roll_component_minus')
LABELS = [*NOMINAL, *ROLL]
CASE = 'coverage_train_yminus_matched_full_tool'
SEED = 1666646064
TICK = 168
HORIZON = 32
SUBSTEPS = 16
CAP = len(LABELS)*HORIZON*SUBSTEPS


def load_run(path, expected):
    plan = json.loads(_verified_bytes(ROOT, path.relative_to(ROOT), expected))
    if plan['schema'] != 's3-roll-pulse-capture-v1' or plan['scope'] != collector.SCOPE:
        raise ValueError('unexpected roll pulse capture plan')
    files = plan['files_sha256']
    for name, digest in files.items():
        _verified_bytes(ROOT, name, digest)

    def read(path):
        return json.loads(_verified_bytes(ROOT, path, files[Path(path).as_posix()]))

    def artifact(validation, name, accepted):
        path = Path(validation).parent/name
        if files[path.as_posix()] != accepted['artifacts_sha256'][name]:
            raise ValueError(f'accepted artifact changed: {path}')
        return read(path)

    static_path = plan['static_validation']
    static = read(static_path)
    if (static['status'],static['updates'],static['simulation_steps'],static['newton_workers'],
            static['test_episode_files_read'],static['selected_episode'],static['plan_sha256']) != (
                'completed',0,0,0,False,CASE,files[plan['static_plan']]):
        raise ValueError('selected training snapshot did not pass static preflight')
    selection = artifact(static_path, 'selected_commands.json', static)
    case = selection['case']
    if (case['episode_id'],case['seed'],case['split'],selection['snapshot_tick'],
            selection['horizon'],selection['control_substeps'],selection['branch_order']) != (
                CASE,SEED,'train',TICK,HORIZON,SUBSTEPS,LABELS):
        raise ValueError('unexpected selected snapshot or branch scope')
    if any(files[row['path']] != row['sha256'] for row in selection['source'].values()):
        raise ValueError('selected snapshot sources differ from the static preflight')
    source = selection['source']['source_validation']['path']
    accepted = read(source)
    manifest = artifact(source, 'capture_manifest.json', accepted)
    entry, = manifest['entries']
    if (accepted['status'] != 'completed' or accepted['case'] != case
            or accepted['test_episode_files_read'] or accepted['control_accepted']
            or any(case[k] != entry[k] for k in case) or entry['snapshot_tick'] != TICK
            or selection['source']['source_manifest']['path'] != Path(source).with_name('capture_manifest.json').as_posix()
            or selection['archived_nominal_references'] != entry['nominal_branches']):
        raise ValueError('roll source is outside the accepted training split')
    execution = artifact(source, 'execution_plan.json', accepted)
    cfg = execution['worker_config']
    if (cfg['seed'] != SEED or _initial_key(cfg) != case['physical_initial_sha256']
            or execution['design']['runtime'] != selection['runtime']
            or files[case['source_config']] != case['source_config_sha256']):
        raise ValueError('worker identity differs from the frozen training source')
    native = {key:read(selection['source'][key]['path']) for key in ('snapshot','layout')}
    for key in native:
        row = selection['source'][key]
        if accepted['artifacts_sha256'][Path(row['path']).relative_to(Path(source).parent).as_posix()] != row['sha256']:
            raise ValueError(f'training native {key} differs from accepted evidence')
    if decode_snapshot(native['snapshot'], native['layout'], cfg, TICK, SUBSTEPS)['episode_id'] != CASE:
        raise ValueError('decoded snapshot has a different episode identity')
    recorded = [artifact(source, item['path'], accepted) for item in entry['nominal_branches']]
    commands = selection['branches']
    for label, trace in zip(NOMINAL, recorded, strict=True):
        if commands[label] != {'requested':trace['actions'], 'executed':trace['commands']}:
            raise ValueError('frozen nominal commands differ from training archive')

    old_plan = read(plan['small_capture_plan'])
    dev_valid_path = (Path(old_plan['output'])/'validation.json').as_posix()
    dev = read(dev_valid_path)
    dev_static_path = old_plan['preflight_validation']
    dev_static = read(dev_static_path)
    dev_spec = artifact(dev_static_path, 'replay_spec.json', dev_static)
    if (dev['status'] != 'completed' or not dev['nominal_gate_passed']
            or dev['plan_sha256'] != files[plan['small_capture_plan']]
            or dev['test_episode_files_read'] or dev['control_accepted'] or dev_spec['split'] != 'validation'
            or dev_spec['case'] != dev['case'] or dev_spec['case']['tick'] != TICK
            or dev_spec['runtime'] != selection['runtime']
            or dev_spec['static_snapshot_checks']['native_layout'] != native['layout']
            or manifest['development_holdout_group'] != dev['case']['group']
            or case['group_id'] == dev['case']['group']):
        raise ValueError('development comparison provenance changed')
    truth = {name:artifact(dev_valid_path, f'worker/{name}.json', dev) for name in LABELS}
    for name in LABELS:
        if commands[name] != {'requested':truth[name]['actions'], 'executed':truth[name]['commands']}:
            raise ValueError(f'command path differs from frozen development {name}')
    limits = {**old_plan['limits'], 'max_physics_substeps':CAP}
    if plan['limits'] != limits or (limits['baseline_replays'],limits['nominal_prefix_substeps'],
            limits['max_worker_starts'],limits['sequential_environments']) != (2,0,1,1):
        raise ValueError('resource envelope or physical scope changed')
    cfg['budget'].update(wall_s=limits['max_worker_wall_s'], max_physics_steps=CAP, max_trace_steps=CAP)
    dev_config = artifact(dev_valid_path, 'execution_plan.json', dev)['worker_config']
    if any(cfg['budget'][k] != value for k,value in dev_config['budget'].items()
           if k not in ('max_physics_steps','max_trace_steps')):
        raise ValueError('worker resource limits differ from accepted small-action run')
    spec = {**dev_spec, 'case':{**case,'tick':TICK}, 'split':'train', 'branch_order':LABELS,
            'same_sign_prefixes':[], 'future_capture_limits':limits}
    return plan, cfg, spec, native, commands, recorded, {'development':truth,'layout':native['layout']}, read


def exact_nominal(new, archived, summary_leads):
    result = collector.nominal_gate(new, archived, summary_leads)
    result['all_contact_records_match'] = all(row['contacts'] == archived[0]['contacts'] for row in (*new,*archived))
    result['passed'] = result['passed'] and result['all_contact_records_match']
    return result


def preflight(_plan, spec, recorded, _read):
    if collector.backend_preflight(spec) != spec['static_backend']:
        raise ValueError('installed backend differs from the accepted small-action capture')
    if not exact_nominal(recorded, recorded, spec['scoring']['summary_leads'])['passed']:
        raise ValueError('archived training nominal reference failed its gate')
    if {'newton', 'warp'}.intersection(sys.modules):
        raise RuntimeError('static roll replay preflight imported physics')


def effects(traces, lead, field):
    def value(name):
        row = traces[name]
        return np.asarray(row['states'][lead-1]['ball']['angular_velocity'] if field == 'world_spin'
                          else row['states_raw'][lead-1][:3])

    half = (value(ROLL[0])-value(ROLL[1]))*.5
    noise = float(np.linalg.norm(value(NOMINAL[0])-value(NOMINAL[1]))*.5)
    return half, noise


def comparison(training, development, floor):
    vector, noise = training
    ref, ref_noise = development
    magnitude, ref_magnitude = (float(np.linalg.norm(v)) for v in (vector,ref))
    resolved = magnitude >= max(floor,5*noise) and ref_magnitude >= max(floor,5*ref_noise)
    cosine = float(np.clip(vector @ ref/(magnitude*ref_magnitude), -1., 1.)) if resolved else None
    ratio = magnitude/ref_magnitude if ref_magnitude >= max(floor,5*ref_noise) else None
    return {'training_vector':vector.tolist(), 'development_vector':ref.tolist(),
            'training_norm':magnitude, 'development_norm':ref_magnitude,
            'training_nominal_repeat_half_spread':noise, 'development_nominal_repeat_half_spread':ref_noise,
            'training_strong':magnitude >= 10*max(floor,5*noise), 'development_strong':ref_magnitude >= 10*max(floor,5*ref_noise),
            'direction_cosine':cosine, 'amplitude_ratio':ratio, 'vector_distance':float(np.linalg.norm(vector-ref))}


def score(spec, inputs, traces):
    dev = inputs['development']
    for label in LABELS:
        _full_branch(dev[label], TICK)
    spin = {}
    for lead in range(1,HORIZON+1):
        row = comparison(effects(traces, lead, 'world_spin'), effects(dev, lead, 'world_spin'), 1e-4)
        row['comparable'] = bool(row['training_strong'] and row['development_strong']
                                 and row['direction_cosine'] >= .8 and .5 <= row['amplitude_ratio'] <= 2.)
        spin[str(lead)] = row
    position_floor = spec['scoring']['absolute_floors_si'][0]
    position = comparison(effects(traces, 32, 'tool_position'), effects(dev, 32, 'tool_position'), position_floor)
    settings = {'case':spec['case'],'horizon':HORIZON,'control_substeps':SUBSTEPS,
                'branch_order':LABELS,'directions':['roll_component']}
    return {'training_case':spec['case'],'spin_world_rad_s':spin,
            'matched_target_leads':[lead for lead in (15,16) if spin[str(lead)]['comparable']],
            'position_h32_tool_m':position,
            'contacts_training':diagnose_contacts(settings,traces,inputs['layout']),
            'contacts_development':diagnose_contacts(settings,dev,inputs['layout'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight','capture','worker'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    path = args.plan.resolve()
    if args.phase == 'preflight':
        plan, _, spec, _, _, recorded, _, read = load_run(path, args.sha256)
        preflight(plan, spec, recorded, read)
        print('Frozen files, archive nominal reference, resource envelope and backend verified. Physics steps: 0.')
        return
    out = args.output.resolve()
    frozen = json.loads(_verified_bytes(ROOT, path.relative_to(ROOT), args.sha256))
    target = (ROOT/frozen['output']).resolve()
    if (len(target.relative_to(ROOT/'runs').parts) != 1
            or out != (target/'worker' if args.phase == 'worker' else target)):
        parser.error('output differs from frozen capture plan')
    # Reuse the archived four-branch replay contract and its resource monitor.
    with patch.multiple(collector, load_run=load_run, preflight=preflight, score=score, __file__=__file__), \
            patch.object(small_action_response, 'nominal_gate', exact_nominal):
        (collector.worker if args.phase == 'worker' else collector.capture)(out, path, args.sha256)


if __name__ == '__main__':
    main()
