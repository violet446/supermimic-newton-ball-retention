import io
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.frame_model import FrameResidualMLP, PACKED_DIM, frame_affine, pack_features, measured_response_scale
from ball_retention.frame_grad import frame_action_sensitivity
from ball_retention.model_data import _rotation, filter_advance, known_rollout
from ball_retention.task import reference


class FrameModelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'configs/s1_online_episode.json').read_text(encoding='utf-8'))
        self.anchor = np.array([.32, 0., -.02])
        self.dt = 1/60
        self.rng = np.random.default_rng(4187)
        self.mean = np.array([.3, -.03, .01, .15, -.07, .03, .8, -1.2, .1])
        self.scale = np.array([.02, .01, .005, .2, .3, .4, 2., 3., 4.])

    def endpoint(self, t, tilt, rate):
        r, axis = _rotation(tilt, np.array(self.cfg['initial']['tool_rotation_world']))
        omega = axis*rate[0]+np.array([0., rate[1], 0.])
        ref = reference(t, self.cfg)
        p = ref['anchor_position']-r@self.anchor
        v = ref['anchor_velocity']-np.cross(omega, r@self.anchor)
        return r, omega, p, v

    def test_affine_matches_independent_world_motion_and_rotating_velocity(self):
        for time in (.0, 1.4, 2.6, 3.4):
            tilt, rate = np.array([.02, -.01]), np.array([.01, .04])
            actions = self.rng.uniform(-.6, .6, (8, 2))
            known = known_rollout(time, tilt, rate, actions, self.cfg, self.dt)
            a, b = frame_affine(known, self.cfg, self.dt, self.anchor)
            raw = self.mean.copy()
            r, omega, p, v = self.endpoint(time, tilt, rate)
            world_p = p+r@raw[:3]
            world_v = v+r@raw[3:6]+np.cross(omega, r@raw[:3])
            world_w = r@raw[6:9]
            for k in range(8):
                tilt, rate, _ = filter_advance(tilt, rate, np.deg2rad(5)*actions[k],
                                              self.cfg['action']['tilt_filter_time_constant_s'], self.dt)
                r, omega, p, v = self.endpoint(time+(k+1)*self.dt, tilt, rate)
                world_p += self.dt*world_v
                expected = np.r_[r.T@(world_p-p), r.T@(world_v-v-np.cross(omega, world_p-p)), r.T@world_w]
                raw = a[k]@raw+b[k]
                np.testing.assert_allclose(raw, expected, atol=3e-13, rtol=0)

    def test_static_frame_is_constant_relative_velocity(self):
        known = known_rollout(0., [0., 0.], [0., 0.], np.zeros((8, 2)), self.cfg, self.dt)
        model = FrameResidualMLP(PACKED_DIM, self.mean, self.scale, self.dt, seed=9)
        packed = pack_features(known, self.cfg, self.dt, self.anchor, self.mean, self.scale, np.zeros(20), np.ones(20))
        actual = model.denormalize(model.rollout(model.normalize(self.mean)[None], packed[None]))[0]
        expected = np.tile(self.mean, (8, 1))
        expected[:, :3] += np.arange(1, 9)[:, None]*self.dt*self.mean[3:6]
        np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=0)

    def test_multistep_vjp_and_reload(self):
        model = FrameResidualMLP(PACKED_DIM, self.mean, self.scale, self.dt, seed=9, hidden=8)
        model.params['W3'][:] = self.rng.normal(scale=.2, size=(8, 9))
        known = known_rollout(1.4, [.01, .02], [.01, -.02], self.rng.uniform(-.5, .5, (3, 8, 2)), self.cfg, self.dt)
        packed = pack_features(known, self.cfg, self.dt, self.anchor, self.mean, self.scale, np.zeros(20), np.ones(20))
        x = self.rng.normal(size=(3, 9))
        cost = self.rng.normal(size=(3, 8, 9))
        ds, dk, dp = model.rollout_vjp(x, packed, cost)
        dx, df = self.rng.normal(size=x.shape), self.rng.normal(size=packed.shape)
        def objective(x, k):
            return np.sum(model.rollout(x, k)*cost)
        eps = 1e-6
        for numerical, analytical in (
                ((objective(x+eps*dx, packed)-objective(x-eps*dx, packed))/(2*eps), np.sum(ds*dx)),
                ((objective(x, packed+eps*df)-objective(x, packed-eps*df))/(2*eps), np.sum(dk*df))):
            np.testing.assert_allclose(numerical, analytical, rtol=2e-7, atol=2e-7)
        original = {key:value.copy() for key,value in model.params.items()}
        direction = {key:self.rng.normal(size=value.shape) for key,value in model.params.items()}
        values = []
        for sign in (1, -1):
            for key in model.params:
                model.params[key][:] = original[key]+sign*eps*direction[key]
            values.append(objective(x, packed))
        np.testing.assert_allclose((values[0]-values[1])/(2*eps), sum(np.sum(dp[k]*direction[k]) for k in dp), rtol=2e-7, atol=2e-7)
        with io.BytesIO() as stream:
            np.savez(stream, **model.state_dict())
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as saved:
                loaded = FrameResidualMLP.from_state_dict(saved)
        np.testing.assert_array_equal(model.rollout(x, packed), loaded.rollout(x, packed))

    def test_exact_action_gradient_includes_transport_and_neural_paths(self):
        model = FrameResidualMLP(PACKED_DIM, self.mean, self.scale, self.dt, seed=9, hidden=8)
        actions = self.rng.uniform(-.5, .5, (32, 2))
        for trained in (False, True):
            if trained:
                model.params['W3'][:] = self.rng.normal(scale=.2, size=(8, 9))
            result = frame_action_sensitivity(model,self.mean,1.4,[.02,-.01],[.01,.04],actions,
                                              self.cfg,np.zeros(20),np.ones(20),self.anchor)
            for axis in (0,1):
                eps = 1e-5
                offset = eps*np.eye(2)[axis]
                predictions = []
                for sign in (1,-1):
                    known = known_rollout(1.4,[.02,-.01],[.01,.04],actions+sign*offset,self.cfg,self.dt)
                    packed = pack_features(known,self.cfg,self.dt,self.anchor,self.mean,self.scale,np.zeros(20),np.ones(20))
                    predictions.append(model.denormalize(model.rollout(model.normalize(self.mean)[None],packed[None]))[0])
                numerical = (predictions[0]-predictions[1])/(2*eps)
                np.testing.assert_allclose(numerical,result['jacobian'][:,:,axis],atol=1e-8,rtol=2e-5)
        clipped = np.full((8,2),1.2)
        result = frame_action_sensitivity(model,self.mean,1.4,[.02,-.01],[.01,.04],clipped,
                                          self.cfg,np.zeros(20),np.ones(20),self.anchor)
        np.testing.assert_array_equal(result['jacobian'],0.)
        with self.assertRaises(ValueError):
            frame_action_sensitivity(model,self.mean,1.4,[.02,-.01],[.01,.04],np.ones((8,2)),
                                     self.cfg,np.zeros(20),np.ones(20),self.anchor)

    def test_response_scale_uses_resolved_training_values_and_replay_floors(self):
        spread = np.ones((2,8,9))*.01
        resolved = np.ones_like(spread,dtype=bool)
        resolved[0,:,0] = False
        spread[0,:,0] = 1000.  # Masked observation must not influence the scale.
        resolved[:,0,:3] = False
        batch = SimpleNamespace(plus_target=spread,minus_target=-spread,response_mask=resolved,
                                response_floor=np.full_like(spread,.001),noise_mask=np.ones((2,8),dtype=bool))
        scale = measured_response_scale(batch)
        np.testing.assert_array_equal(scale[0,:3],.001)
        np.testing.assert_array_equal(scale[1:],.01)
        np.testing.assert_array_equal(scale[0,3:],.01)


if __name__ == '__main__':
    unittest.main()
