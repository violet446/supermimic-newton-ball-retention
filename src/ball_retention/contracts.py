"""Versioned physical transition contract; SI units, world frame, xyzw quaternions."""
from dataclasses import dataclass, asdict, field
import math
import numpy as np

SCHEMA = 'ball-retention-transition-v0.1'
EPISODE_SCHEMA = 'ball-retention-transition-v0.2'

def vector(value, size, name):
    a = np.array(value, dtype=float, copy=True)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected {size} finite values')
    return a

@dataclass
class RigidState:
    position: list[float]
    quaternion_xyzw: list[float]
    linear_velocity_origin: list[float]
    angular_velocity: list[float]

    def validate(self):
        for name in ('position', 'linear_velocity_origin', 'angular_velocity'):
            vector(getattr(self, name), 3, name)
        q = vector(self.quaternion_xyzw, 4, 'quaternion_xyzw')
        if abs(np.linalg.norm(q) - 1) > 1e-5:
            raise ValueError('quaternion must be normalized')

@dataclass
class State:
    time_s: float
    ball: RigidState
    tool: RigidState
    joint_q: list[float]
    joint_qd: list[float]
    tilt_rad: list[float]
    tilt_rate_rad_s: list[float]
    previous_action: list[float]
    reference_phase: float
    departure_outside_s: float = 0.

    def validate(self):
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError('invalid time')
        if not 0 <= self.reference_phase <= 1:
            raise ValueError('invalid reference phase')
        if not math.isfinite(self.departure_outside_s) or self.departure_outside_s < 0:
            raise ValueError('invalid departure duration')
        self.ball.validate(); self.tool.validate()
        for name in ('joint_q', 'joint_qd'):
            vector(getattr(self, name), 6, name)
        for name in ('tilt_rad', 'tilt_rate_rad_s', 'previous_action'):
            vector(getattr(self, name), 2, name)

@dataclass
class Transition:
    episode_id: str
    step_index: int
    state_t: State
    requested_action: list[float]
    # Every actually applied substep command, not only the final target.
    executed_joint_q: list[list[float]]
    executed_joint_qd: list[list[float]]
    next_state_before_reset: State
    terminated: bool
    truncated: bool
    invalid_reason: str | None
    reward: float | None  # None in S1: reward has not been specified or trained.
    context: dict
    reference_t: dict
    schema_version: str = SCHEMA
    task: dict = field(default_factory=dict)

    def validate(self, dt_s, substeps):
        if self.schema_version not in (SCHEMA, EPISODE_SCHEMA) or not self.episode_id or self.step_index < 0:
            raise ValueError('invalid identity/schema')
        self.state_t.validate(); self.next_state_before_reset.validate()
        vector(self.requested_action, 2, 'requested_action')
        applied = len(self.executed_joint_q)
        if applied != substeps and not (self.schema_version == EPISODE_SCHEMA and self.terminated and 0 < applied < substeps):
            raise ValueError('only a physical termination can shorten the control step')
        for name in ('executed_joint_q', 'executed_joint_qd'):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (applied, 6) or not np.isfinite(a).all():
                raise ValueError(f'invalid {name}')
        elapsed = self.next_state_before_reset.time_s-self.state_t.time_s
        if not math.isclose(elapsed, dt_s*applied, abs_tol=1e-9):
            raise ValueError('transition must end before reset after the recorded substeps')
        if self.terminated and self.truncated:
            raise ValueError('physical termination takes precedence over timeout')
        if self.reward is not None and not math.isfinite(self.reward):
            raise ValueError('non-finite reward')

    def to_dict(self):
        row = asdict(self)
        if self.schema_version == SCHEMA:
            row.pop('task')
            row['state_t'].pop('departure_outside_s')
            row['next_state_before_reset'].pop('departure_outside_s')
        return row

def validate_window(rows, dt_s, substeps):
    if not rows:
        raise ValueError('empty window')
    for row in rows:
        row.validate(dt_s, substeps)
        if row.invalid_reason is not None:
            raise ValueError('invalid trajectories are diagnostics, not training windows')
    for prev, row in zip(rows, rows[1:]):
        if prev.terminated or prev.truncated or row.episode_id != prev.episode_id:
            raise ValueError('window crosses an episode boundary')
        if row.schema_version != prev.schema_version:
            raise ValueError('window mixes transition schema versions')
        if row.step_index != prev.step_index+1 or asdict(prev.next_state_before_reset) != asdict(row.state_t):
            raise ValueError('non-contiguous window')
