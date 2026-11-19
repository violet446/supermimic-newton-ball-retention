"""Run the frozen same-GPU Newton repeatability diagnostic."""
from dataclasses import asdict
from importlib.metadata import version
import inspect
import os
import time
from unittest.mock import patch

import numpy as np

from ball_retention.archive_state_import import _check_layout
from ball_retention.collection import replay_error
from ball_retention.controller import OnlineTiltController
from ball_retention.geometry import load_tool, support_point
from ball_retention.guard import process_memory
from ball_retention.native_repeatability_plan import validate_case
from ball_retention.prescribed import PrescribedPath
from ball_retention.resources import directory_bytes
from ball_retention.robot import Robot
from ball_retention.task import ball_in_tool

from archive_restore import write
from archive_restore_metrics import native_metrics


def _nominal_schedule(root, cfg, archived, tick, substeps, guard):
    spec = cfg['action_pairs']
    target = archived[tick]
    nominal = target['nominal_actions']
    if substeps != 16 or len(nominal) != 32:
        raise ValueError('unexpected control clock or replay length')
    events = {event['start_step']: event['action'] for event in spec['schedule']}
    actions = []
    action = events[0]
    for i in range(tick+len(nominal)):
        if i in events:
            action = events[i]
        actions.append(action)
    np.testing.assert_array_equal(actions[tick:], nominal)
    if np.max(np.abs(actions)) >= 1.:
        raise ValueError('nominal request reaches the action clipping boundary')

    mesh = load_tool(root/'assets/shovel.STL')
    anchor = support_point(mesh, cfg['initial']['anchor_xy_tool_m'])[0]
    np.testing.assert_array_equal(anchor, target['anchor_local_m'])
    path = PrescribedPath(root/cfg['prescribed']['joint_path'], cfg, moving=True)
    controller = OnlineTiltController(Robot(root/'assets/ec66_description.urdf'), cfg, anchor, path, root/'assets')
    dt = cfg['physics']['dt_s']
    replay_commands = []
    for i, request in enumerate(actions):
        commands = controller.plan((i*substeps)*dt, request, substeps)
        for command in commands:
            controller.commit(command)
        if i+1 == tick:
            checkpoint_command = command
            for values, name in ((command.q, 'joint_q'), (command.qd, 'joint_qd'), (command.tilt, 'tilt_rad'),
                                 (command.tilt_rate, 'tilt_rate_rad_s'), (command.clipped_action, 'previous_action')):
                np.testing.assert_array_equal(values, target['initial_state'][name])
        if i >= tick:
            replay_commands.append({'q': [command.q.tolist() for command in commands],
                                    'qd': [command.qd.tolist() for command in commands]})
        if i % 24 == 0:
            guard()
    return actions[:tick], nominal, checkpoint_command, replay_commands


def _nominal_commands(root, cfg, archived, tick, substeps, guard):
    return _nominal_schedule(root, cfg, archived, tick, substeps, guard)[:2]


def _make_env(root, cfg, summary, stage, expected_device_arch=None):
    package_versions = {name: version(name) for name in ('newton', 'warp-lang')}
    if package_versions != {'newton': '1.6.0', 'warp-lang': '1.17.0'}:
        raise ValueError(f'installed Newton/Warp versions differ from the plan: {package_versions}')
    summary['physics_versions'] = package_versions
    stage('native_repeatability_before_physics_imports')
    import newton
    import warp as wp
    from newton._src.solvers.xpbd import kernels
    from ball_retention import episode

    if expected_device_arch is not None:
        device = wp.get_device('cuda:0')
        if device.arch != expected_device_arch:
            raise ValueError(f'CUDA architecture differs from the native reference: sm_{device.arch}')
        summary['physics_device'] = {'alias': 'cuda:0', 'name': device.name, 'architecture': f'sm_{device.arch}'}
    required_options = [(newton.CollisionPipeline, 'deterministic'), (newton.solvers.SolverXPBD, 'deterministic')]
    if any(name not in inspect.signature(factory).parameters for factory, name in required_options):
        raise ValueError('pinned backend does not expose both deterministic mode parameters')
    mode = wp.DeterministicMode.RUN_TO_RUN
    summary['backend_options'] = {'collision_pipeline_deterministic': True, 'xpbd_deterministic_mode': mode.name,
                                  'warp_process_default_mode': wp.config.deterministic.name}
    summary['verified_solver_builds'] = 0
    stage('native_repeatability_backend_signatures_verified')

    class NativeRepeatabilityArm(episode.PrescribedArmEnv):
        def _new_solver(self):
            self.solver = self.newton.solvers.SolverXPBD(self.model, iterations=self.cfg['physics']['iterations'],
                                                        enable_restitution=False, deterministic=mode)
            self.pipeline = self.newton.CollisionPipeline(self.model, deterministic=True)
            self.contacts = self.pipeline.contacts()
            active_mode = wp.get_module(kernels.__name__).options.get('deterministic')
            if self.pipeline.deterministic is not True or active_mode != mode:
                raise RuntimeError('Newton collision or XPBD did not retain the requested deterministic option')
            summary['verified_solver_builds'] += 1

        def step(self, *args, **kwargs):
            row = super().step(*args, **kwargs)
            if self.replay_contacts is not None:
                self.replay_contacts.append({'physics_step': row['step'], 'candidates': row['contacts_before_step'],
                                             'shape_pairs': row['contact_shape_pairs']})
            return row

        replay_contacts = None

    # Hold the substitution only for construction of this worker's single episode.
    # Its private physics instance keeps the subclass on each subsequent restore.
    with patch.object(episode, 'PrescribedArmEnv', NativeRepeatabilityArm):
        env = episode.EpisodeEnv(root, cfg, report_stage=stage)
    summary['warp_kernel_cache_dir'] = str(wp.config.kernel_cache_dir)
    return env


def _record_nominal(env, saved, nominal, advance, substeps, control_dt):
    contacts, states, raw, elapsed, commands, retention = [], [], [], [], [], []
    env.physics.replay_contacts = contacts
    try:
        for action in nominal:
            row = advance(action)
            state = row.next_state_before_reset
            states.append(asdict(state))
            terms = ball_in_tool(state.ball, state.tool)
            raw.append(np.r_[terms['position'], terms['velocity'], terms['ball_omega']].tolist())
            elapsed.append(state.time_s-saved['state'].time_s)
            commands.append({'q': row.executed_joint_q, 'qd': row.executed_joint_qd})
            retention.append({'terminated': row.terminated, 'truncated': row.truncated,
                              'reason': row.task['termination_reason'],
                              'carrying_candidate': row.task['carrying']['carrying_candidate'],
                              'outside_grace_active': state.departure_outside_s > 0})
    finally:
        env.physics.replay_contacts = None
    np.testing.assert_allclose(elapsed, np.arange(1,len(nominal)+1)*control_dt, atol=1e-9, rtol=0.)
    if len(contacts) != len(nominal)*substeps:
        raise ValueError('contact records must cover every branch physics substep')
    return {'actions': nominal, 'substeps': [substeps]*len(nominal), 'elapsed_s': elapsed,
            'terminated': False, 'truncated': False, 'termination_reason': None,
            'states': states, 'states_raw': raw, 'commands': commands, 'retention': retention, 'contacts': contacts}


def run(root, cfg, out, max_steps, summary, stage):
    started = time.monotonic()
    plan, archive = validate_case(root, cfg, max_steps)
    tick = plan['case']['tick']
    limits = plan['budget']
    substeps = round(1/(cfg['physics']['control_hz']*cfg['physics']['dt_s']))
    control_dt = substeps*cfg['physics']['dt_s']

    def guard():
        memory = process_memory(os.getpid())
        if (time.monotonic()-started >= limits['max_worker_wall_s']
                or memory['peak_rss_bytes'] > limits['max_process_rss_gib']*1024**3
                or memory['peak_private_commit_bytes'] > limits['max_process_commit_gib']*1024**3
                or directory_bytes(out) > limits['max_new_output_mib']*1024**2):
            raise RuntimeError('native repeatability worker budget reached')

    guard()
    prefix, nominal = _nominal_commands(root, cfg, archive, tick, substeps, guard)
    stage('native_nominal_controller_verified')
    env = _make_env(root, cfg, summary, stage)
    env.reset(episode_id='s3-native-repeatability')
    if env.substeps != substeps:
        raise ValueError('backend substep clock differs from configuration')
    layout = _check_layout(env.physics)
    write(out, 'native_layout.json', layout)
    summary['completed_branches'] = []
    summary['prefix_restore_ticks'] = []
    guard()
    stage('native_layout_verified')

    def advance(action):
        if summary['physics_steps']+substeps > max_steps or time.monotonic()-started >= limits['max_worker_wall_s']:
            raise RuntimeError('native repeatability substep or worker wall cap')
        before = env.physics.steps
        try:
            row = env.step(action)
        finally:
            summary['physics_steps'] += env.physics.steps-before
        if (len(row.executed_joint_q) != substeps or env.physics.steps-before != substeps
                or row.invalid_reason is not None or row.terminated or row.truncated
                or row.requested_action != action or row.next_state_before_reset.previous_action != action):
            raise RuntimeError('nominal transition invalid, constrained or incomplete')
        return row

    def restore(saved):
        env.restore(saved)
        if (replay_error(asdict(env.state), asdict(saved['state']))
                or env.step_index != saved['step_index'] or env.physics.steps != saved['step_index']*substeps):
            raise RuntimeError('native restore changed the saved controller clock or public state')
        for buffer in (env.physics.a, env.physics.b):
            for name, array in saved['physics']['arrays'].items():
                np.testing.assert_array_equal(getattr(buffer, name).numpy(), array)

    for i, action in enumerate(prefix):
        if i in archive:
            restore(env.snapshot())
            summary['prefix_restore_ticks'].append(i)
            stage(f'native_prefix_restore_{i}')
        if i == 0:
            stage('first_native_step_started')
        advance(action)
        if i == 0:
            guard()
            stage('first_native_step_complete')
        if i % 24 == 0:
            guard()
    if summary['physics_steps'] != limits['nominal_prefix_substeps'] or env.step_index != tick:
        raise RuntimeError('nominal prefix did not end at the frozen checkpoint')

    saved = env.snapshot()
    write(out, 'snapshot.json', {**saved, 'state': asdict(saved['state']),
                                 'physics': {**saved['physics'],
                                  'arrays': {name: value.tolist() for name, value in saved['physics']['arrays'].items()},
                                  'command': [value.tolist() for value in saved['physics']['command']]}})
    stage('native_target_snapshot_saved')

    def replay(label):
        restore(saved)
        stage(f'{label}_start')
        trace = _record_nominal(env, saved, nominal, advance, substeps, control_dt)
        write(out, f'{label}.json', trace)
        summary['completed_branches'].append(label)
        guard()
        stage(f'{label}_complete')
        return trace

    branches = [replay(f'native_replay_{i}') for i in (1, 2)]
    if summary['physics_steps'] != max_steps:
        raise ValueError('complete native repeatability count differs from the frozen reservation')
    comparison = native_metrics(branches)
    comparison['contact_candidates_and_shape_pairs_match'] = branches[0]['contacts'] == branches[1]['contacts']
    comparison['contact_candidate_counts'] = sorted({row['candidates'] for trace in branches for row in trace['contacts']})
    write(out, 'comparison.json', comparison)
    guard()
    summary.update(status='native_repeatability_completed', native_comparison=comparison, control_accepted=False)
