"""Control-rate single-ball episode with Newton physics and explicit boundaries."""
import copy
import math
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import EPISODE_SCHEMA, RigidState, State, Transition, vector
from .env import PrescribedArmEnv
from .retention import DepartureMonitor, ToolEnvelope
from .task import ball_in_tool, origin_velocity, reference


class EpisodeEnv:
    def __init__(self, root, cfg, device='cuda:0', report_stage=None):
        from .controller import OnlineTiltController
        from .support import CarryingRegion

        self.cfg = copy.deepcopy(cfg)
        self.dt = cfg['physics']['dt_s']
        self.control_hz = cfg['physics']['control_hz']
        ratio = 1/(self.control_hz*self.dt)
        self.substeps = round(ratio)
        if self.substeps < 1 or not math.isclose(ratio, self.substeps, abs_tol=1e-10):
            raise ValueError('control interval must contain a whole number of physics steps')
        spec = cfg['episode']
        horizon_steps = spec['horizon_s']*self.control_hz
        self.horizon_steps = round(horizon_steps)
        if self.horizon_steps < 1 or not math.isclose(horizon_steps, self.horizon_steps, abs_tol=1e-10):
            raise ValueError('episode horizon must be aligned to control ticks')
        ref = cfg['reference']
        self.route_end = ref['start_s']+ref['duration_s']+ref['hold_end_s']
        self.physics = PrescribedArmEnv(root, cfg, device, moving=True, report_stage=report_stage)
        self.controller = OnlineTiltController(self.physics.robot, self.cfg, self.physics.anchor_local,
                                               self.physics.path, root/'assets')
        rotation = cfg['initial']['tool_rotation_world']
        radius = cfg['ball']['radius_m']
        self.carrying = CarryingRegion(self.physics.mesh, rotation, radius)
        envelope = ToolEnvelope(self.physics.mesh, rotation, radius, spec['departure_margin_m'])
        self.departure = DepartureMonitor(envelope, spec['outside_grace_s'])
        self.initial_monitor = self.departure.snapshot()
        self.initial_controller = self.controller.snapshot()
        self.episode_id = None
        self.state = None
        self.done = True
        self.invalid = False
        self.step_index = 0
        self.deep_count = 0
        self.context = {'ball': copy.deepcopy(cfg['ball']), 'physics': copy.deepcopy(cfg['physics']),
                        'action': copy.deepcopy(cfg['action']), 'robot': copy.deepcopy(cfg['robot']),
                        'reference': copy.deepcopy(cfg['reference']), 'episode': copy.deepcopy(spec),
                        'seed': cfg['seed']}
        self.context['action']['tilt_filter_time_constant_s'] = self.controller.tau
        self.context['episode']['carrying_bowl_start_x_m'] = self.carrying.bowl_start_x_m

    def _state(self, row, tilt, tilt_rate, action, departure):
        bid = self.physics.ball_id
        ball_pose = row['body_q'][bid]
        ball_v = row['body_qd_com_world'][bid]
        rotation = Rotation.from_quat(ball_pose[3:]).as_matrix()
        ball_com = np.asarray(ball_pose[:3])+rotation@self.physics.com_local[bid]
        ball_origin_v = origin_velocity(ball_v[:3], ball_v[3:], ball_pose[:3], ball_com)
        time_s = row['time_s']
        return State(time_s, RigidState(ball_pose[:3], ball_pose[3:], ball_origin_v.tolist(), ball_v[3:]),
            RigidState(row['tool_position_world_m'], row['tool_quaternion_xyzw'],
                       row['tool_origin_velocity_world_m_s'], row['tool_angular_velocity_world_rad_s']),
            row['command_q_rad'], row['command_qd_rad_s'], np.asarray(tilt).tolist(),
            np.asarray(tilt_rate).tolist(), np.asarray(action).tolist(), reference(time_s, self.cfg)['phase'],
            departure['outside_duration_s'])

    def reset(self, episode_id=None):
        new_id = str(uuid.uuid4()) if episode_id is None else episode_id
        if not isinstance(new_id, str) or not new_id or new_id == self.episode_id:
            raise ValueError('reset requires a new nonempty episode ID')
        row = self.physics.reset()
        self.controller.restore(self.initial_controller)
        self.departure.restore(self.initial_monitor)
        departure = self.departure.update(0., row['ball_local_m'], row['floor_reached'])
        if departure['terminated']:
            raise ValueError('episode starts in a terminal state')
        self.episode_id, self.step_index = new_id, 0
        self.done = self.invalid = False
        self.deep_count = 0
        self.state = self._state(row, [0., 0.], [0., 0.], [0., 0.], departure)
        return copy.deepcopy(self.state)

    def step(self, action):
        if self.done or self.invalid:
            raise RuntimeError('episode requires reset or restore before stepping')
        requested = vector(action, 2, 'requested_action')
        previous = self.state
        commands = self.controller.plan(previous.time_s, requested, self.substeps)
        if len(commands) != self.substeps:
            raise ValueError('controller returned an incomplete plan')
        applied_q, applied_qd = [], []
        max_anchor_error = max_rotation_error = 0.
        try:
            for command in commands:
                row = self.physics.step((command.q, command.qd, command.qdd), command.rotation, allow_floor_contact=True)
                self.controller.commit(command)
                applied_q.append(row['command_q_rad'])
                applied_qd.append(row['command_qd_rad_s'])
                max_anchor_error = max(max_anchor_error, row['anchor_error_m'])
                max_rotation_error = max(max_rotation_error, row['tool_rotation_error_rad'])
                self.deep_count = self.deep_count+1 if row['surface_clearance_m'] < -.002 else 0
                if self.deep_count >= 3:
                    raise RuntimeError('deep penetration for three substeps')
                departure = self.departure.update(row['time_s'], row['ball_local_m'], row['floor_reached'])
                if departure['terminated']:
                    break
            next_state = self._state(row, command.tilt, command.tilt_rate, command.clipped_action, departure)
            terminated = departure['terminated']
            truncated = self.step_index+1 == self.horizon_steps and not terminated
            motion_complete = row['time_s']+1e-10 >= self.route_end
            carrying = self.carrying.measure(row['ball_local_m'])
            success = (truncated and motion_complete and carrying['carrying_candidate']
                       and abs(row['surface_clearance_m']) <= self.cfg['episode']['final_surface_gap_m'])
            ref = reference(previous.time_s, self.cfg)
            transition = Transition(self.episode_id, self.step_index, previous, requested.tolist(), applied_q, applied_qd,
                next_state, terminated, truncated, None, None, self.context,
                {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in ref.items()},
                schema_version=EPISODE_SCHEMA, task={'termination_reason': departure['reason'],
                    'departure_time_s': row['time_s'] if terminated else None, 'motion_complete': motion_complete,
                    'success': bool(success), 'carrying': carrying, 'max_anchor_error_m': max_anchor_error,
                    'max_tool_rotation_error_rad': max_rotation_error,
                    'observation_t': {key: value.tolist() for key, value in ball_in_tool(previous.ball, previous.tool).items()}})
            transition.validate(self.dt, self.substeps)
        except Exception:
            self.invalid = True
            raise
        self.state = next_state
        self.step_index += 1
        self.done = terminated or truncated
        return copy.deepcopy(transition)

    def snapshot(self):
        return {'schema': 'ball-retention-episode-snapshot-v1',
                'physics': self.physics.snapshot(), 'controller': self.controller.snapshot(),
                'monitor': self.departure.snapshot(), 'episode_id': self.episode_id, 'step_index': self.step_index,
                'state': copy.deepcopy(self.state), 'done': self.done, 'invalid': self.invalid,
                'deep_count': self.deep_count}

    def restore(self, saved):
        if saved['schema'] != 'ball-retention-episode-snapshot-v1':
            raise ValueError('episode snapshot schema mismatch')
        self.physics.restore(saved['physics'])
        self.controller.restore(saved['controller'])
        self.departure.restore(saved['monitor'])
        self.episode_id, self.step_index = saved['episode_id'], saved['step_index']
        self.state = copy.deepcopy(saved['state'])
        self.done, self.invalid, self.deep_count = saved['done'], saved['invalid'], saved['deep_count']
