"""Bounded, recorded joint paths for S1. Not an online policy executor."""
import numpy as np
from scipy.interpolate import PPoly


class PrescribedPath:
    def __init__(self, archive, cfg, moving):
        with np.load(archive, allow_pickle=False) as data:
            self.curve = PPoly(data['zero_coefficients'], data['zero_knots_s'], extrapolate=False)
        self.cfg = cfg
        self.moving = moving

    def command(self, time_s):
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError('invalid command time')
        if not self.moving:
            return np.asarray(self.cfg['initial']['joint_q_rad']), np.zeros(6), np.zeros(6)
        t = np.clip(time_s, self.curve.x[0], self.curve.x[-1])
        q = self.curve(t)
        # Replay may continue beyond the final knot: hold the endpoint, never extrapolate.
        if time_s >= self.curve.x[-1]:
            return q, np.zeros(6), np.zeros(6)
        return q, self.curve(t, 1), self.curve(t, 2)


def collision_exclusions(names, bodies, adjacent):
    """Explicit shape pairs; never suppress ball/robot or ball/tool collisions."""
    return [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))
            if (bodies[i] >= 0 and bodies[i] == bodies[j])
            or frozenset((names[i], names[j])) in adjacent]
