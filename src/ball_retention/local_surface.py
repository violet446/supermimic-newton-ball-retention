"""Smooth tool-local surface features and their state and tool-spin derivatives."""
import numpy as np
from scipy.ndimage import gaussian_filter, spline_filter
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point

from .frame_model import cross_matrix


def _smoothstep(increasing):
    u = np.clip(increasing, 0., 1.)
    return u*u*(3.-2.*u), 6.*u*(1.-u)


def _basis(t):
    """Cubic B-spline values and derivatives for the four adjacent nodes."""
    weights = np.stack(((1.-t)**3, 4.-6.*t*t+3.*t**3,
                        1.+3.*t+3.*t*t-3.*t**3, t**3), axis=-1)/6.
    derivatives = np.stack((-.5*(1.-t)**2, -2.*t+1.5*t*t,
                            .5+t-1.5*t*t, .5*t*t), axis=-1)
    return weights, derivatives


class SurfaceField:
    """Fixed mesh field with seven tapered, dimensionless state features."""

    def __init__(self, low, spacing, coefficients, radius):
        self.low = np.array(low, dtype=float, copy=True)
        self.spacing = float(spacing)
        self.coefficients = np.array(coefficients, dtype=float, copy=True)
        self.radius = float(radius)
        if (self.low.shape != (3,) or self.coefficients.ndim != 4
                or self.coefficients.shape[-1] != 4 or min(self.coefficients.shape[:3]) < 7
                or not np.isfinite(self.low).all() or not np.isfinite(self.coefficients).all()
                or not np.isfinite([self.spacing, self.radius]).all()
                or min(self.spacing, self.radius) <= 0.):
            raise ValueError('invalid spline grid or sphere radius')
        self._last = np.asarray(self.coefficients.shape[:3])-1
        self._high = self.low+self.spacing*self._last
        self.low.setflags(write=False)
        self.coefficients.setflags(write=False)

    @classmethod
    def from_mesh(cls, mesh, radius, base_rotation, field_spec, guard=None):
        """Build exact nearest-triangle samples, then smooth and prefilter the channels."""
        low = np.asarray(field_spec['grid_low_tool_m'], dtype=float)
        high = np.asarray(field_spec['grid_high_tool_m'], dtype=float)
        spacing = float(field_spec['spacing_m'])
        cells = np.rint((high-low)/spacing).astype(int)
        if not np.allclose(low+spacing*cells, high, atol=1e-12, rtol=0.):
            raise ValueError('surface domain must span whole grid cells')
        grid = np.indices(tuple(cells+1), dtype=float).reshape(3, -1).T*spacing+low
        normals = np.asarray(mesh.face_normals)
        selected = ((np.asarray(mesh.triangles_center)[:, 0] >= .22)
                    & (normals@np.asarray(base_rotation)[2] >= .2))
        triangles = np.asarray(mesh.triangles)[selected]
        normals = normals[selected]
        if not len(triangles):
            raise ValueError('surface field has no upward-facing triangles')
        bounds_low, bounds_high = triangles.min(axis=1), triangles.max(axis=1)
        distances = cKDTree(triangles.reshape(-1, 3)).query(grid)[0]
        # A selected vertex is on the surface, so its distance is an upper
        # bound. Exclude boxes that cannot intersect any such search sphere.
        reach = distances.max()+1e-12
        near = np.all((bounds_high >= low-reach) & (bounds_low <= high+reach), axis=1)
        triangles, normals = triangles[near], normals[near]
        bounds_low, bounds_high = bounds_low[near], bounds_high[near]
        values = np.empty((len(grid), 4))
        for start in range(0, len(grid), 32):
            end = min(start+32, len(grid))
            if guard is not None and start % 2048 == 0:
                guard(f'surface_nodes_{start}')
            points = grid[start:end]
            delta = np.maximum(np.maximum(bounds_low-points[:, None], points[:, None]-bounds_high), 0.)
            rows, faces = np.nonzero(np.sum(delta**2, axis=-1) <= (distances[start:end, None]+1e-12)**2)
            offset = points[rows]-closest_point(triangles[faces], points[rows])
            square_distance = np.sum(offset**2, axis=-1)
            minimum = np.full(len(points), np.inf)
            np.minimum.at(minimum, rows, square_distance)
            tied = np.flatnonzero(square_distance == minimum[rows])
            winners = np.full(len(points), len(rows))
            np.minimum.at(winners, rows[tied], tied)
            if np.any(winners == len(rows)):
                raise ValueError('surface query failed to find a finite nearest face')
            nearest_normals = normals[faces[winners]]
            values[start:end, :3] = nearest_normals
            values[start:end, 3] = np.sum(offset[winners]*nearest_normals, axis=-1)-radius
        values = values.reshape(*tuple(cells+1), 4)
        smooth = gaussian_filter(values, (1., 1., 1., 0.), mode='nearest')
        coefficients = np.stack([spline_filter(smooth[..., i], mode='nearest') for i in range(4)], axis=-1)
        return cls(low, spacing, coefficients, radius)

    def _sample(self, points, jacobian):
        coordinates = (points-self.low)/self.spacing
        clamped = np.clip(coordinates, 0, self._last)
        cell = np.floor(clamped).astype(int)
        weights, derivatives = _basis(clamped-cell)
        nodes = np.clip(cell[..., None]+np.arange(-1, 3), 0, self._last[:, None])
        x, y, z = np.moveaxis(nodes, -2, 0)
        neighborhood = self.coefficients[x[:, :, None, None], y[:, None, :, None], z[:, None, None, :]]
        wx, wy, wz = np.moveaxis(weights, -2, 0)
        field = np.einsum('nijkf,ni,nj,nk->nf', neighborhood, wx, wy, wz)
        norm = np.sqrt(np.sum(field[:, :3]**2, axis=-1, keepdims=True)+1e-12)
        normal = field[:, :3]/norm
        gap = field[:, 3]
        if not jacobian:
            return normal, gap
        derivatives *= ((coordinates > 0) & (coordinates < self._last))[..., None]/self.spacing
        dx, dy, dz = np.moveaxis(derivatives, -2, 0)
        field_jacobian = np.stack((
            np.einsum('nijkf,ni,nj,nk->nf', neighborhood, dx, wy, wz),
            np.einsum('nijkf,ni,nj,nk->nf', neighborhood, wx, dy, wz),
            np.einsum('nijkf,ni,nj,nk->nf', neighborhood, wx, wy, dz),
        ), axis=-1)
        normal_jacobian = field_jacobian[:, :3]/norm[..., None]
        normal_jacobian -= normal[..., None]*np.sum(normal[..., None]*normal_jacobian, axis=1, keepdims=True)
        return normal, gap, normal_jacobian, field_jacobian[:, 3]

    def sample(self, points):
        """Return interpolated unit normals and signed sphere gaps, for diagnostics."""
        points = np.asarray(points, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError('expected finite points of shape (..., 3)')
        normal, gap = self._sample(points.reshape(-1, 3), jacobian=False)
        return normal.reshape(points.shape), gap.reshape(points.shape[:-1])

    def _weights(self, points, gap, jacobian):
        lower, dlower = _smoothstep(((points-self.low)/self.spacing-1.)/2.)
        upper, dupper = _smoothstep(((self._high-points)/self.spacing-1.)/2.)
        per_axis = lower*upper
        domain = per_axis.prod(axis=-1)
        band, dband = _smoothstep((.008-np.abs(gap))/.005)
        weight = domain*band
        if not jacobian:
            return weight
        d_axis = (dlower*upper-lower*dupper)/(2.*self.spacing)
        d_domain = d_axis*np.roll(per_axis, 1, axis=-1)*np.roll(per_axis, -1, axis=-1)
        return weight, d_domain*band[:, None], -domain*dband*np.sign(gap)/.005

    def weights(self, points, gap):
        """Return the product of the domain and gap taper weights."""
        points, gap = np.asarray(points, dtype=float), np.asarray(gap, dtype=float)
        if points.shape != (*gap.shape, 3):
            raise ValueError('point and gap shapes disagree')
        return self._weights(points, gap, jacobian=False)

    def features(self, raw_state, omega_tool, jacobian=False):
        """Return feature values and, optionally, Jacobians in raw SI coordinates."""
        state, tool = np.asarray(raw_state, dtype=float), np.asarray(omega_tool, dtype=float)
        if (state.shape[-1:] != (9,) or tool.shape != (*state.shape[:-1], 3)
                or not np.isfinite(state).all() or not np.isfinite(tool).all()):
            raise ValueError('invalid local surface state or tool spin')
        shape = state.shape[:-1]
        state, tool = state.reshape(-1, 9), tool.reshape(-1, 3)
        points = state[:, :3]
        active = np.all((points > self.low+self.spacing) & (points < self._high-self.spacing), axis=-1)
        point = points[active]
        sampled = self._sample(point, jacobian)
        normal, gap = sampled[:2]
        taper = self._weights(point, gap, jacobian)
        weight = taper[0] if jacobian else taper
        angular = state[active, 6:9]-tool[active]
        point_velocity = state[active, 3:6]-self.radius*np.cross(angular, normal)
        projection = np.sum(normal*point_velocity, axis=-1, keepdims=True)
        slip = point_velocity-normal*projection
        gap_input = np.tanh(gap/.005)
        slip_input = np.tanh(slip/.05)
        ungated = np.column_stack((normal, gap_input, slip_input))
        output = np.zeros((len(state), 7))
        output[active] = weight[:, None]*ungated
        values = output.reshape(*shape, 7)
        if not jacobian:
            return values

        normal_grad, gap_grad = sampled[2:]
        weight_grad = taper[1]+taper[2][:, None]*gap_grad
        projector = np.eye(3)-normal[..., None]*normal[:, None, :]
        point_velocity_grad = -self.radius*cross_matrix(angular)@normal_grad
        slip_grad = (projector@point_velocity_grad - projection[..., None]*normal_grad
                     - normal[..., None]*(point_velocity[:, None, :]@normal_grad))
        slip_scale = (1.-slip_input**2)/.05
        ungated_grad = np.concatenate((normal_grad, ((1.-gap_input**2)/.005)[:, None, None]*gap_grad[:, None],
                                       slip_scale[..., None]*slip_grad), axis=1)
        state_grad = np.zeros((len(state), 7, 9))
        state_grad[active, :, :3] = weight[:, None, None]*ungated_grad+ungated[..., None]*weight_grad[:, None]
        state_grad[active, 4:, 3:6] = (weight[:, None]*slip_scale)[..., None]*projector
        spin_grad = self.radius*state_grad[active, 4:, 3:6]@cross_matrix(normal)
        state_grad[active, 4:, 6:] = spin_grad
        tool_grad = np.zeros((len(state), 7, 3))
        tool_grad[active, 4:] = -spin_grad
        return values, state_grad.reshape(*shape, 7, 9), tool_grad.reshape(*shape, 7, 3)
