"""Freeze one model-gradient action and symmetric controls for each development group."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from ball_retention.archive_state_import import _parse_state
from ball_retention.controller import OnlineTiltController
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import KNOWN_NAMES, _ball_state
from ball_retention.pair_collection import _initial_key
from ball_retention.prescribed import PrescribedPath
from ball_retention.robot import Robot
from ball_retention.world_increment_model import WorldIncrementMLP, world_action_jvp
from check_s3_actions import schedule_actions
from check_s3_archived_snapshot import audit
from deterministic_archive_metrics import CONTROL_FIELDS, _full_branch
from preflight_s3_world_increment import predict
from saved_native_snapshot import decode_snapshot
from train_s3_local_velocity_increment import bound


PHASE = 'gradient_action_preflight'
LABELS = ('nominal_1', 'nominal_2', 'gradient', 'reverse')
TICK, SUBSTEPS, HORIZON = 168, 16, 32


def read(plan, path):
    return json.loads(bound(plan, path))


def artifact(plan, validation, relative):
    path = Path(validation).parent/relative
    if plan['files_sha256'][path.as_posix()] != read(plan, validation)['artifacts_sha256'][relative]:
        raise ValueError(f'accepted artifact changed: {path}')
    return read(plan, path)


def controller(config, mesh, robot):
    anchor = support_point(mesh, config['initial']['anchor_xy_tool_m'])[0]
    path = PrescribedPath(ROOT/config['prescribed']['joint_path'], config, moving=True)
    return OnlineTiltController(robot, config, anchor, path, ROOT/'assets'), anchor


def public_case(plan, record, mesh, robot, guard):
    history = plan['historical_validation']
    status = read(plan, history)
    if (status['status'],status['training_updates'],status['test_metrics_used'],status['control_accepted']) != (
            'completed_and_evaluated',0,False,False):
        raise ValueError('historical development capture was not accepted')
    manifest = artifact(plan, history, 'capture_manifest.json')
    entry, = [e for e in manifest['entries'] if e['episode_id'] == record['episode']]
    root = Path(history).parent
    src, cfg_path = root/entry['path'], root/entry['config_path']
    if ((entry['split'],entry['group_id'],entry['seed']) != ('validation',record['group'],record['seed'])
            or TICK not in entry['snapshot_ticks']
            or plan['files_sha256'][src.as_posix()] != entry['sha256']
            or plan['files_sha256'][cfg_path.as_posix()] != entry['config_sha256']):
        raise ValueError('independent public development source changed')
    cfg = artifact(plan, history, entry['config_path'])
    row, = [x for x in map(json.loads, bound(plan, src).splitlines()) if x['snapshot_tick']==TICK]
    clock = (row['control_substeps'], float(row['physics_dt_s']))
    if clock != (SUBSTEPS,1/960) or cfg['seed'] != record['seed']:
        raise ValueError('historical clock or seed changed')
    historical_baselines = [x for x in row['branches'] if x['axis'] is None and x['offset']==0.]
    if ([x['repeat'] for x in historical_baselines] != [0,1]
            or any(x['terminated'] or x['truncated'] or x['actions'] != row['nominal_actions']
                   or x['substeps'] != [SUBSTEPS]*HORIZON for x in historical_baselines)):
        raise ValueError('missing complete historical nominal controls')

    audit_row = audit(ROOT, cfg, record, row, SUBSTEPS, mesh, robot, guard)
    old_audit = read(plan, plan['historical_static_audit'])
    recorded, = [item for item in old_audit['cases'] if item['episode']==record['episode']]
    if old_audit['status']!='passed' or audit_row!=recorded or not audit_row['interior_geometry']['center_inside_projection']:
        raise ValueError('public snapshot static audit did not reproduce')
    active, _ = controller(cfg, mesh, robot)
    prefix = schedule_actions(cfg['action_pairs']['schedule'], 0, TICK)
    dt = cfg['physics']['dt_s']
    for index, action in enumerate(prefix):
        for cmd in active.plan((index*SUBSTEPS)*dt, action, SUBSTEPS):
            active.commit(cmd)
        if index%24==0:
            guard()
    np.testing.assert_array_equal(cmd.qdd, audit_row['joint_qdd_from_nominal_prefix_rad_s2'])
    state = _parse_state(row['initial_state'])
    for key, expected in zip(CONTROL_FIELDS, (cmd.q, cmd.qd, cmd.tilt, cmd.tilt_rate, cmd.clipped_action), strict=True):
        np.testing.assert_array_equal(getattr(state,key), expected)
    if schedule_actions(cfg['action_pairs']['schedule'], TICK, TICK+HORIZON) != row['nominal_actions']:
        raise ValueError('archived nominal does not reproduce from the source schedule')
    restoration = {'kind':'public_state_only', 'physical_restore_validated':False, 'state':row['initial_state'],
                   'controller':active.snapshot(), 'nominal_prefix_actions':prefix, 'static_audit':audit_row,
                   'source_config':cfg_path.as_posix(), 'source_trace':src.as_posix(),
                   'source_validation':history}
    return cfg, state, row['nominal_actions'], restoration, None


def native_case(plan, record):
    archive = read(plan, plan['native_archive_plan'])
    native = archive['native_reference']
    validation = plan['native_validation']
    status = read(plan, validation)
    if (status['status']!='completed' or not status['passed'] or status['test_episode_files_read']
            or any(archive['case'][key]!=record[key] for key in ('episode','group','seed','tick'))):
        raise ValueError('native repeatability evidence changed')
    small = read(plan, plan['small_capture_validation'])
    if (small['status']!='completed' or small['test_episode_files_read'] or not small['nominal_gate_passed']
            or small['case']!=record):
        raise ValueError('accepted native development operating point changed')
    cfg = artifact(plan, plan['small_capture_validation'], 'execution_plan.json')['worker_config']
    snapshot, layout = (artifact(plan, validation, Path(native[key]).relative_to(Path(validation).parent).as_posix())
                        for key in ('snapshot','layout'))
    saved = decode_snapshot(snapshot, layout, cfg, TICK, SUBSTEPS)
    traces = [artifact(plan, plan['small_capture_validation'], f'worker/{label}.json') for label in LABELS[:2]]
    for trace in traces:
        _full_branch(trace, TICK)
    if traces[0]!=traces[1] or cfg['seed']!=record['seed']:
        raise ValueError('nominal Newton repeats or identity did not reproduce')
    restoration = {'kind':'native_snapshot','physical_restore_validated':True,
                   'snapshot':native['snapshot'], 'layout':native['layout'], 'controller':saved['controller'],
                   'source_validation':validation, 'nominal_validation':plan['small_capture_validation']}
    return cfg, saved['state'], traces[0]['actions'], restoration, traces[0]['commands']


def suggest(model, normalization, cfg, state, nominal, anchor, policy):
    initial = _ball_state(state)
    nominal = np.asarray(nominal)
    target_y = initial[1]
    pulse = np.zeros_like(nominal)
    pulse[:policy['pulse_steps']] = 1.
    def forecast(actions):
        return predict(model, state, actions, cfg, anchor, *normalization)

    y_nominal = forecast(nominal)
    nominal_offset = float(y_nominal[-1,1]-target_y)
    slopes = []
    eps = policy['centered_difference_step']
    for axis in range(2):
        basis = pulse*np.eye(2)[axis]
        y, tangent = world_action_jvp(model, initial, state.time_s, state.tilt_rad, state.tilt_rate_rad_s,
                                       nominal, basis, cfg, anchor, *normalization)
        np.testing.assert_array_equal(y, y_nominal)
        central = (forecast(nominal+eps*basis)-forecast(nominal-eps*basis))/(2*eps)
        np.testing.assert_allclose(tangent[-1,1], central[-1,1], rtol=2e-5, atol=5e-11)
        slopes.append(float(tangent[-1,1]))
    grad = nominal_offset*np.asarray(slopes)
    size = np.linalg.norm(grad)
    if not np.isfinite(size) or size==0.:
        raise ValueError('undefined bounded descent direction')
    step = -policy['amplitude_l2']*(grad/size)
    delta = pulse*step
    requests = dict(zip(LABELS, (nominal, nominal, nominal+delta, nominal-delta), strict=True))
    if any(np.any(np.abs(actions)>=1.) for actions in requests.values()):
        raise ValueError('candidate crosses the action clipping boundary')
    predictions = {}
    for key, action in requests.items():
        y = y_nominal if key.startswith('nominal') else forecast(action)
        if not np.isfinite(y).all():
            raise ValueError('nonfinite model rollout')
        predictions[key] = {'h32_y_tool_m':float(y[-1,1]),'h32_signed_lateral_offset_m':float(y[-1,1]-target_y),
                            'h32_abs_lateral_offset_m':float(abs(y[-1,1]-target_y))}
    costs = [predictions[name]['h32_abs_lateral_offset_m'] for name in ('gradient','nominal_1','reverse')]
    ready = min(costs[1]-costs[0],costs[2]-costs[1]) > policy['position_margin_floor_m']
    return requests, {'target_y_tool_m':float(target_y),'dy_h32_m_per_unit_pulse':slopes,
                      'gradient_delta_roll_pitch':step.tolist(),'predictions':predictions,
                      'gradient_nominal_reverse_order_has_margin':ready}


def validate_commands(active, restoration, cfg, requests, reference, guard):
    dt = cfg['physics']['dt_s']
    result = {}
    for label, actions in requests.items():
        active.restore(restoration['controller'])
        executed = []
        for lead, action in enumerate(actions):
            step = active.plan((TICK+lead)*SUBSTEPS*dt,action,SUBSTEPS)
            for command in step:
                active.commit(command)
                np.testing.assert_array_equal(command.clipped_action,action)
            executed.append({'q':[x.q.tolist() for x in step],'qd':[x.qd.tolist() for x in step]})
        if label.startswith('nominal') and reference is not None and executed!=reference:
            raise ValueError('computed nominal commands differ from accepted Newton execution')
        result[label] = {'requested':actions.tolist(),'executed':executed}
        guard(label)
    return result


def run(plan, out, guard):
    if plan['updates']!=0 or (plan['tick'],plan['horizon'],plan['control_substeps'])!=(TICK,HORIZON,SUBSTEPS):
        raise ValueError('unexpected scope for gradient test preflight')
    model_path = plan['model_validation']
    accepted = read(plan, model_path)
    if (accepted['status'],accepted['updates'],accepted['training_pairs'],accepted['test_episode_files_read'],
            accepted['main_model_replaced']) != ('completed',900,61,False,False):
        raise ValueError('original 61-pair endpoint not accepted')
    ck = Path(model_path).with_name('world_increment_900.npz')
    if plan['files_sha256'][ck.as_posix()]!=accepted['artifacts_sha256'][ck.name]:
        raise ValueError('model checkpoint digest changed')
    training = artifact(plan,model_path,'data_preflight.json')['group_weights']
    with np.load(ROOT/ck,allow_pickle=False) as data:
        if data['updates'].item()!=900:
            raise ValueError('model checkpoint update changed')
        np.testing.assert_array_equal(data['known_names'], KNOWN_NAMES)
        model = WorldIncrementMLP.from_state_dict(data)
        normalization = data['known_mean'],data['known_scale']
    if model.dt_s!=1/60 or not all(np.isfinite(v).all() for v in normalization) or np.any(normalization[1]<=0.):
        raise ValueError('invalid frozen model clock or normalization')

    inventory = {}
    for path, expected in plan['native_snapshot_inventory'].items():
        state = read(plan,path)
        if state['episode_id']!=expected['episode_id']:
            raise ValueError(f'native snapshot identity changed: {path}')
        inventory[path] = expected
    if set(x['group_id'] for x in inventory.values() if x['split']=='validation') != {plan['native_case']['group']}:
        raise ValueError('development native inventory changed')
    mesh, robot = load_tool(ROOT/'assets/shovel.STL'), Robot(ROOT/'assets/ec66_description.urdf')
    groups = {c['group'] for c in (plan['native_case'],plan['public_case'])}
    if len(groups)!=2 or groups.intersection(training):
        raise ValueError('development groups overlap each other or model training')
    guard('development_nominal')
    cases = {}
    keys = []
    for kind, record in (('native',plan['native_case']),('public',plan['public_case'])):
        config, state, nominal, restore, reference = (native_case(plan,record) if kind=='native'
                                                      else public_case(plan,record,mesh,robot,guard))
        active, anchor = controller(config,mesh,robot)
        keys.append(_initial_key(config))
        requested, scores = suggest(model,normalization,config,state,nominal,anchor,plan['policy'])
        if kind=='native':
            expected = artifact(plan,model_path,'timed_rollouts.json')['nominal'][-1][1]
            np.testing.assert_equal(scores['predictions']['nominal_1']['h32_y_tool_m'],expected)
        commands = validate_commands(active,restore,config,requested,reference,guard)
        cases[record['episode']] = {'identity':record,'physical_initial_sha256':keys[-1],
                                    'restoration':restore,'score':scores,'commands':commands}
        guard(f'{kind}_ready', log=True)
    if keys[0]==keys[1]:
        raise ValueError('development conditions share a physical initial state')
    write_json(out/'candidate_commands.json', {'policy':plan['policy'],'cases':cases})
    write_json(out/'restore_inventory.json', {'native_snapshots':inventory,
               'independent_development_native_groups':sorted(
                   {x['group_id'] for x in inventory.values() if x['split']=='validation'}),
               'public_source_requires_physical_readback_and_repeatability':plan['public_case']['episode']})
    return {'static_cases':len(cases),'static_branches':sum(len(row['commands']) for row in cases.values()),
            'rank_order_cases':sum(row['score']['gradient_nominal_reverse_order_has_margin'] for row in cases.values()),
            'physical_restore_verified_cases':1,'cross_case_execution_ready':False,'newton_workers':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(),args.sha256,PHASE,run)
