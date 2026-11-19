"""Freeze and check the one-snapshot deterministic response pilot without physics."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from check_s3_deterministic_archive_import import ROOT, backend_preflight, digest, write_json
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.controller import OnlineTiltController
from ball_retention.deterministic_archive_import_plan import DESIGN as ARCHIVE, load as load_archive
from ball_retention.deterministic_response_model import evaluate_model
from ball_retention.geometry import load_tool, support_point
from ball_retention.guard import process_memory
from ball_retention.prescribed import PrescribedPath
from ball_retention.resources import directory_bytes
from ball_retention.robot import Robot
from deterministic_archive_metrics import native_reference_metrics
from native_repeatability import _nominal_schedule
from saved_native_snapshot import decode_snapshot
import numpy as np


DESIGN = 'configs/s3_deterministic_action_response_plan.json'
DESIGN_SHA256 = 'f1ea393adce1fadf3f80a749634d1eae162031537fe94bdbd33d8179bbd39a66'
OUTPUT = ROOT/'runs/s3_deterministic_action_response_preflight_20261115'


def load_plan(root):
    """Bind all references to the accepted nominal replay and original saved snapshot."""
    root = Path(root).resolve()
    plan = json.loads(_verified_bytes(root, DESIGN, DESIGN_SHA256))
    inputs = {name: _verified_bytes(root, name, value) for name, value in plan['inputs_sha256'].items()}
    archive, cfg, rows = load_archive(root)
    case = plan['case']
    if plan['checkpoint_source'] != ARCHIVE or any(value != archive['case'][key] for key, value in case.items()):
        raise ValueError('response case differs from the saved native snapshot')
    h, substeps = plan['horizon'], plan['control_substeps']
    budget = plan['budget']
    labels = ['nominal_1', 'nominal_2', *(f'{name}_{sign}' for name in plan['directions'] for sign in ('plus', 'minus'))]
    if (h != cfg['action_pairs']['horizon'] or substeps != rows[case['tick']]['control_substeps']
            or labels != plan['branch_order'] or len(labels)*h*substeps != budget['max_physics_substeps']
            or budget['baseline_replays'] != 2 or budget['nominal_prefix_substeps'] != 0
            or budget['max_worker_starts'] != 1 or budget['sequential_environments'] != 1):
        raise ValueError('response pilot clock or budget disagrees with its fixed branch list')
    validation_path = plan['nominal_validation']
    validation = json.loads(inputs[validation_path])
    if (validation['passed'] is not True or validation['tick'] != case['tick'] or validation['seed'] != case['seed']
            or validation['nominal_prefix_substeps'] != 0
            or validation['comparisons']['native_reference_comparison']['all_native_files_byte_identical'] is not True):
        raise ValueError('nominal response reference did not pass the acceptance gates')
    native_layout = archive['native_reference']
    replays = []
    for name, reference in zip(plan['nominal_replays'], native_layout['replays'], strict=True):
        relative = Path(name).relative_to(Path(validation_path).parent).as_posix()
        if validation['artifacts_sha256'][relative] != plan['inputs_sha256'][name] or (
                archive['inputs_sha256'][reference] != plan['inputs_sha256'][name]):
            raise ValueError('response reference is inconsistent with accepted nominal replay')
        replays.append(json.loads(inputs[name]))
    data = {key: json.loads(_verified_bytes(root, native_layout[key], archive['inputs_sha256'][native_layout[key]]))
            for key in ('snapshot', 'layout')}
    return plan, cfg, rows, replays, data


def action_sequences(plan, nominal):
    nominal = np.asarray(nominal, dtype=float)
    if nominal.shape != (plan['horizon'], 2) or not np.isfinite(nominal).all():
        raise ValueError('expected finite nominal roll/pitch actions for all control steps')
    directions = {}
    actions = {name: nominal.copy() for name in ('nominal_1', 'nominal_2')}
    for name, segments in plan['directions'].items():
        direction = np.zeros_like(nominal)
        previous_stop = 0
        for step in segments:
            start, stop = step['start'], step['stop']
            axes = np.asarray(step['axes'], dtype=float)
            if (type(start) is not int or type(stop) is not int or not previous_stop <= start < stop <= len(nominal)
                    or axes.shape != (2,) or not np.isclose(np.linalg.norm(axes), 1., rtol=0., atol=1e-12)):
                raise ValueError(f'invalid unit direction segment in {name}')
            direction[start:stop] = axes
            previous_stop = stop
        directions[name] = direction
        for sign, multiplier in (('plus', 1), ('minus', -1)):
            request = nominal + multiplier*plan['amplitude']*direction
            if np.any(np.abs(request) >= 1.):
                raise ValueError(f'{name} would reach the controller clipping boundary')
            actions[f'{name}_{sign}'] = request
    return directions, actions


def verify_commands(root, plan, cfg, saved, actions, expected_nominal, guard):
    controller = OnlineTiltController(Robot(root/'assets/ec66_description.urdf'), cfg,
                                     support_point(load_tool(root/'assets/shovel.STL'),
                                                   cfg['initial']['anchor_xy_tool_m'])[0],
                                     PrescribedPath(root/cfg['prescribed']['joint_path'], cfg, moving=True),
                                     root/'assets')
    origin = saved['physics']['steps']
    substeps = plan['control_substeps']
    dt = cfg['physics']['dt_s']
    commands, peaks = {}, {}
    for label, sequence in actions.items():
        controller.restore(saved['controller'])
        series = []
        max_qd = max_qdd = 0.
        for lead, request in enumerate(sequence):
            step = controller.plan((origin+lead*substeps)*dt, request, substeps)
            for command in step:
                controller.commit(command)
                np.testing.assert_array_equal(command.clipped_action, request)
            series.append({'q': [command.q.tolist() for command in step], 'qd': [command.qd.tolist() for command in step]})
            max_qd = max(max_qd, float(np.max(np.abs([command.qd for command in step]))))
            max_qdd = max(max_qdd, float(np.max(np.abs([command.qdd for command in step]))))
        if label.startswith('nominal') and series != expected_nominal:
            raise ValueError('nominal controller commands differ from saved replay')
        commands[label] = series
        peaks[label] = {'max_requested_component': float(np.max(np.abs(sequence))),
                        'max_joint_speed_rad_s': max_qd, 'max_joint_accel_rad_s2': max_qdd}
        guard()
    for sign in ('plus', 'minus'):
        if commands[f'pitch_pulse_{sign}'][:8] != commands[f'pitch_reversal_{sign}'][:8]:
            raise ValueError('pulse and reversal shared commands disagree before reversal')
    return commands, peaks


def run(out):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    try:
        plan, cfg, rows, replays, data = load_plan(ROOT)
        budget = plan['static_budget']

        def guard():
            memory = process_memory(os.getpid())
            if (time.monotonic()-began >= budget['wall_s']
                    or memory['peak_rss_bytes'] > budget['max_process_rss_gib']*1024**3
                    or memory['peak_private_commit_bytes'] > budget['max_process_commit_gib']*1024**3
                    or directory_bytes(out) > budget['max_new_output_mib']*1024**2):
                raise RuntimeError('response static preflight resource cap')
            return memory

        guard()
        backend = backend_preflight(plan)
        if int(backend['device']['native_sm']) != plan['runtime']['cuda_arch']:
            raise ValueError('response pilot GPU architecture differs from the saved scene')
        tick, substeps = plan['case']['tick'], plan['control_substeps']
        _, nominal, command, expected = _nominal_schedule(ROOT, cfg, rows, tick, substeps, guard)
        saved = decode_snapshot(data['snapshot'], data['layout'], cfg, tick, substeps, command)
        replay_check = native_reference_metrics(replays, replays)
        if (not replay_check['passed'] or any(replay['commands'] != expected or replay['actions'] != nominal for replay in replays)
                or any(lead['position_m'] > replay_check['position_resolution_m']
                       for lead in replay_check['all_branch_max_half_spread_by_lead'].values())):
            raise ValueError('saved nominal baseline disagrees with the response preflight')
        directions, actions = action_sequences(plan, nominal)
        commands, peaks = verify_commands(ROOT, plan, cfg, saved, actions, expected, guard)
        spec = {**plan['model'], 'amplitude': plan['amplitude'],
                'checkpoint_sha256': plan['inputs_sha256'][plan['model']['checkpoint']]}
        model = evaluate_model(ROOT, spec, cfg, data['snapshot']['state'], nominal, directions, guard)
        for sign in ('plus', 'minus'):
            np.testing.assert_array_equal(model['pairs']['pitch_pulse'][sign][:8],
                                          model['pairs']['pitch_reversal'][sign][:8])
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('response static preflight imported physics')
        sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted({Path(__file__).resolve(),
                   ROOT/'scripts/check_s3_deterministic_archive_import.py', ROOT/'scripts/check_s3_actions.py',
                   *ROOT.glob('src/ball_retention/*.py'), *ROOT.glob('diagnostics/resource_probe/*.py')})}
        write_json(out/'branch_commands.json', {label: {'requested': actions[label].tolist(), 'executed': commands[label]}
                                                for label in plan['branch_order']})
        write_json(out/'model_rollouts.json', model)
        memory = guard()
        write_json(out/'preflight.json', {'passed': True, 'design_sha256': DESIGN_SHA256, 'case': plan['case'],
                   'source_sha256': sources, 'backend': backend, 'branch_order': plan['branch_order'],
                   'nominal_reference_position_half_spread': replay_check['all_branch_max_half_spread_by_lead'],
                   'controller_peaks': peaks, 'shared_first_eight_commands_and_model_states_exact': True,
                   'model_id': plan['model']['id'], 'wall_s': time.monotonic()-began, **memory,
                   'physical_substeps': 0, 'worker_starts': 0, 'test_episode_files_read': False,
                   'artifacts_sha256': {p.name: digest(p) for p in out.iterdir()}})
    except BaseException as error:
        write_json(out/'stopped.json', {'reason': str(error), 'physical_substeps': 0,
                   'wall_s': time.monotonic()-began, 'test_episode_files_read': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    output = parser.parse_args().output.resolve()
    if not output.relative_to(ROOT/'runs').parts:
        raise ValueError('choose a fresh run subdirectory')
    run(output)
