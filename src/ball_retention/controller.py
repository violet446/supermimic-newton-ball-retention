"""Online tilt targets and checked kinematic substep commands for the EC66."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .clearance import excluded_pairs, geometry_rows, separating_plane, world_geometry
from .contracts import vector
from .task import reference, tool_pose_for_anchor


@dataclass(frozen=True)
class TiltCommand:
    time_s: float
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    tilt: np.ndarray
    tilt_rate: np.ndarray
    rotation: np.ndarray
    clipped_action: np.ndarray
    filter_input_state: np.ndarray


class OnlineTiltController:
    """Hold each normalized target for one control interval.

    The target is filtered by two identical first-order stages. Their exact
    constant-input evolution gives a continuous tilt and tilt rate across
    action changes. A plan is checked before the environment commits any part.
    """

    def __init__(self, robot, cfg, anchor_local, baseline_path, asset_root):
        self.robot, self.cfg, self.baseline_path = robot, cfg, baseline_path
        self.anchor_local = vector(anchor_local, 3, 'material anchor')
        self.base_rotation = np.array(cfg['initial']['tool_rotation_world'], dtype=float, copy=True)
        self.dt = float(cfg['physics']['dt_s'])
        action = cfg['action']
        self.angle_limit, self.rate_limit, self.accel_limit = np.deg2rad([
            action['tilt_limit_deg'], action['tilt_rate_deg_s'], action['tilt_acceleration_deg_s2']])
        self.joint_speed = float(cfg['robot']['joint_speed_rad_s'])
        self.joint_accel = float(cfg['robot']['joint_acceleration_rad_s2'])
        limits = [self.dt, self.angle_limit, self.rate_limit, self.accel_limit, self.joint_speed, self.joint_accel]
        if not np.isfinite(limits).all() or min(limits) <= 0:
            raise ValueError('controller limits must be positive and finite')
        # For a bounded input, the double-pole impulse response has integral 1
        # and its first derivative has L1 norm 2/(e*tau). Bounding each stage by
        # the input range also gives |tilt acceleration| <= 4*limit/tau**2.
        min_tau = max(2*self.angle_limit/(math.e*self.rate_limit), 2*math.sqrt(self.angle_limit/self.accel_limit))
        self.tau = float(action.get('tilt_filter_time_constant_s', min_tau))
        if not math.isfinite(self.tau) or self.tau < min_tau:
            raise ValueError('tilt filter time constant is too short for configured limits')
        self.geometry = geometry_rows(asset_root)
        self.floor_z = float(cfg['physics']['floor_z_m'])
        self.excluded = excluded_pairs()
        self.reaches = []
        self.depths = []
        for row in self.geometry:
            link = 'flan' if row['link'] == 'tool' else row['link']
            chain = []
            for _, child, transform, axis in robot.joints:
                chain.append((float(np.linalg.norm(transform[:3, 3])), axis is not None))
                if child == link:
                    break
            else:
                raise ValueError(f'collision link absent from kinematic chain: {link}')
            distances = [row['radius_m']+sum(length for length, _ in chain[k+1:])
                         for k, (_, moving) in enumerate(chain) if moving]
            self.depths.append(len(distances))
            self.reaches.append(np.pad(distances, (0, 6-len(distances))))
        self.reaches = np.asarray(self.reaches)
        initial_q, initial_qd, _ = self.baseline_path.command(0.)
        self.time_s = 0.
        self.q = initial_q.copy()
        self.qd = initial_qd.copy()
        self.tilt = np.zeros(2)
        self.filtered = np.zeros(2)
        tool, _, poses = self.robot.fk(self.q)
        rows = world_geometry(self.geometry, poses, tool)
        self.pairs = []
        for i, row in enumerate(rows):
            link = self.geometry[i]['link']
            rotation = (tool if link == 'tool' else poses[link])[:3, :3]
            for j in range(i+1, len(rows)):
                other = rows[j]
                if frozenset((row['name'], other['name'])) in self.excluded:
                    continue
                _, axis = separating_plane(row, other)
                shared = min(self.depths[i], self.depths[j])
                self.pairs.append((i, j, shared, rotation.T@axis))

    def snapshot(self):
        return {'time_s': self.time_s, 'q': self.q.tolist(), 'qd': self.qd.tolist(),
                'tilt': self.tilt.tolist(), 'filtered': self.filtered.tolist()}

    def restore(self, state):
        time_s = float(state['time_s'])
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError('invalid controller snapshot time')
        q = vector(state['q'], 6, 'snapshot joint position')
        qd = vector(state['qd'], 6, 'snapshot joint velocity')
        tilt = vector(state['tilt'], 2, 'snapshot tilt')
        filtered = vector(state['filtered'], 2, 'snapshot first filter state')
        if (np.any(q < self.robot.low) or np.any(q > self.robot.high)
                or np.max(abs(qd)) > self.joint_speed
                or np.max(abs(np.r_[tilt, filtered])) > self.angle_limit+1e-12):
            raise ValueError('controller snapshot exceeds configured limits')
        self.time_s, self.q, self.qd, self.tilt, self.filtered = time_s, q, qd, tilt, filtered

    def commit(self, command):
        if not math.isclose(command.time_s, self.time_s+self.dt, rel_tol=0, abs_tol=1e-10):
            raise ValueError('controller command does not continue the current state')
        self.time_s, self.q, self.qd = command.time_s, command.q.copy(), command.qd.copy()
        self.tilt, self.filtered = command.tilt.copy(), command.filter_input_state.copy()

    def plan(self, time_s, requested_action, substeps):
        if not math.isclose(time_s, self.time_s, rel_tol=0, abs_tol=1e-10):
            raise ValueError('controller and environment clocks disagree')
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError('expected a positive substep count')
        action = np.clip(vector(requested_action, 2, 'requested action'), -1., 1.)
        target = self.angle_limit*action
        pure_baseline = not np.any(np.r_[target, self.filtered, self.tilt])
        previous_q, previous_qd = self.q, self.qd
        travel = np.zeros(6)
        commands = []
        for i in range(1, substeps+1):
            elapsed = i*self.dt
            t = time_s+elapsed
            u = elapsed/self.tau
            decay = math.exp(-u)
            filtered = target+(self.filtered-target)*decay
            tilt = target+(self.tilt-target+u*(self.filtered-target))*decay
            tilt_rate = (filtered-tilt)/self.tau
            tilt_accel = (target-2*filtered+tilt)/self.tau**2
            if pure_baseline:
                q, qd, qdd = self.baseline_path.command(t)
                rotation = self.base_rotation
            else:
                q, qd, qdd, rotation = self._solve(t, tilt, tilt_rate, tilt_accel, previous_q)
            if (not all(np.isfinite(x).all() for x in (q, qd, qdd))
                    or np.any(q < self.robot.low) or np.any(q > self.robot.high)
                    or np.max(abs(qd)) > self.joint_speed or np.max(abs(qdd)) > self.joint_accel
                    or np.max(abs(q-previous_q))/self.dt > self.joint_speed
                    or np.max(abs(qd-previous_qd))/self.dt > self.joint_accel):
                raise ValueError(f'joint constraint exceeded at {t:.6f}s')
            travel += abs(q-previous_q)
            commands.append(TiltCommand(t, q.copy(), qd.copy(), qdd.copy(), tilt, tilt_rate,
                                        rotation.copy(), action.copy(), filtered))
            previous_q, previous_qd = q, qd
        self._check_clearance(travel)
        return commands

    def _solve(self, t, tilt, tilt_rate, tilt_accel, seed):
        ref = reference(t, self.cfg)
        roll, pitch = tilt
        rotate_y = Rotation.from_rotvec([0., pitch, 0.]).as_matrix()
        rotation = rotate_y@Rotation.from_rotvec([roll, 0., 0.]).as_matrix()@self.base_rotation
        roll_axis = rotate_y[:, 0]
        pitch_axis = np.array([0., 1., 0.])
        omega = roll_axis*tilt_rate[0]+pitch_axis*tilt_rate[1]
        alpha = (roll_axis*tilt_accel[0]+pitch_axis*tilt_accel[1]
                 +np.cross(pitch_axis*tilt_rate[1], roll_axis*tilt_rate[0]))
        arm = rotation@self.anchor_local
        target = tool_pose_for_anchor(ref['anchor_position'], self.anchor_local, rotation)
        q = seed.copy()
        for _ in range(8):
            actual, jacobian, _ = self.robot.fk(q)
            error = np.r_[target[:3, 3]-actual[:3, 3],
                          Rotation.from_matrix(rotation@actual[:3, :3].T).as_rotvec()]
            if np.max(abs(error)) < 1e-11:
                break
            q += np.linalg.solve(jacobian, error)
        else:
            raise ValueError(f'online IK did not converge at {t:.6f}s')
        twist = np.r_[ref['anchor_velocity']-np.cross(omega, arm), omega]
        qd = np.linalg.solve(jacobian, twist)
        twist_derivative = np.r_[ref['anchor_acceleration']-np.cross(alpha, arm)
                                 -np.cross(omega, np.cross(omega, arm)), alpha]
        h = 1e-4
        jac_plus = self.robot.fk(q+h*qd)[1]
        jac_minus = self.robot.fk(q-h*qd)[1]
        jacobian_dot = (jac_plus-jac_minus)/(2*h)
        qdd = np.linalg.solve(jacobian, twist_derivative-jacobian_dot@qd)
        return q, qd, qdd, rotation

    def _check_clearance(self, travel):
        """Certify all endpoints and linear joint segments in the plan.

        The accumulated angle variation bounds swept vertex displacement from
        the initial pose. Shared upstream rotations cancel in a pair's common
        link frame. Ground clearance instead uses the world-frame displacement.
        """
        tool, _, poses = self.robot.fk(self.q)
        rows = world_geometry(self.geometry, poses, tool)
        epsilon = 1e-7
        for i, row in enumerate(rows):
            floor_clearance = row['vertices'][:, 2].min()-self.floor_z-self.reaches[i]@travel
            if floor_clearance < .02+epsilon:
                raise ValueError(f'cannot certify planned floor clearance for {row["name"]}')
        for i, j, shared, local_axis in self.pairs:
            row, other = rows[i], rows[j]
            motion = (self.reaches[i, shared:]+self.reaches[j, shared:])@travel[shared:]
            required = motion+.002+epsilon
            link = self.geometry[i]['link']
            rotation = (tool if link == 'tool' else poses[link])[:3, :3]
            axis = rotation@local_axis
            a, b = row['vertices']@axis, other['vertices']@axis
            gap = max(b.min()-a.max(), a.min()-b.max())
            # The initial witness is fast to recheck. If it no longer separates,
            # search the current pose without modifying the fixed witness cache.
            if gap < required and separating_plane(row, other)[0] < required:
                raise ValueError(f'cannot certify planned clearance for {row["name"]}/{other["name"]}')
