import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.contracts import EPISODE_SCHEMA, RigidState, State, Transition
from ball_retention.dataset import write_manifest
from ball_retention.model_data import KNOWN_NAMES, known_rollout, prepare_dataset


class ModelDataTests(unittest.TestCase):
    config = {'physics': {'dt_s': 1/960, 'control_hz': 60, 'gravity_m_s2': 9.81},
              'ball': {'radius_m': .02},
              'action': {'tilt_filter_time_constant_s': .6, 'tilt_limit_deg': 5.},
              'reference': {'start_s': 1., 'duration_s': 2., 'displacement_world_m': [.08, 0., 0.]},
              'initial': {'tool_rotation_world': np.eye(3).tolist(), 'anchor_world_m': [0.]*3,
                          'anchor_xy_tool_m': [0.]*2}}

    def test_known_rollout_uses_clipped_requests_and_advances_filter(self):
        period = 1/60
        known = known_rollout(0., [0., 0.], [0., 0.], [[2., 0.], [2., 0.]], self.config, period)
        u = np.array([.5, 1.5])*period/self.config['action']['tilt_filter_time_constant_s']
        expected = np.deg2rad(5.)*(1-(1+u)*np.exp(-u))
        np.testing.assert_allclose(known[:, KNOWN_NAMES.index('midpoint_roll_rad')], expected, atol=1e-16)
        np.testing.assert_array_equal(known[:, KNOWN_NAMES.index('roll_request')], [1., 1.])
        changed = known_rollout(0., [0., 0.], [0., 0.], [[2., 0.], [-1., .5]], self.config, period)
        np.testing.assert_array_equal(known[0], changed[0])
        self.assertFalse(np.array_equal(known[1], changed[1]))

    def _episode(self, root, episode_id, split, x, changed_tilt):
        tool = RigidState([0.]*3, [0., 0., 0., 1.], [0.]*3, [0.]*3)
        ball = copy.deepcopy(tool)
        ball.position[0] = x
        state = State(0., ball, tool, [0.]*6, [0.]*6, [0.]*2, [0.]*2, [0.]*2, 0.)
        rows = []
        for tick, count in enumerate((16, 16, 5)):
            terminated = count < 16
            next_state = copy.deepcopy(state)
            next_state.time_s += count/960
            next_state.ball.position[0] += .01
            # The observed future controller state differs from the planned one.
            # This must not enter a rollout that started at tick zero.
            next_state.tilt_rad[0] = changed_tilt
            next_state.departure_outside_s = .04 if tick == 1 else 0.
            row = Transition(episode_id, tick, state, [0., 0.], [[0.]*6]*count, [[0.]*6]*count,
                next_state, terminated, False, None, None, self.config, {}, schema_version=EPISODE_SCHEMA,
                task={'success': False, 'termination_reason': 'outside_projection' if terminated else None})
            rows.append(row.to_dict())
            state = next_state
        filename = episode_id+'.jsonl'
        (root/filename).write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        config_file = episode_id+'.json'
        config_bytes = json.dumps(self.config).encode('utf-8')
        (root/config_file).write_bytes(config_bytes)
        return {'episode_id': episode_id, 'group_id': episode_id, 'split': split, 'path': filename,
                'provenance': {'config': config_file, 'config_sha256': hashlib.sha256(config_bytes).hexdigest()}}

    def test_training_scales_fixed_windows_and_ending_metadata(self):
        scratch = ROOT/'.tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            entries = [self._episode(root, 'training', 'train', 0., .1),
                       self._episode(root, 'validation', 'validation', 3., .2)]
            write_manifest(root, entries, 1/960, 16)
            data = prepare_dataset(root, horizons=(1, 2))
        self.assertAlmostEqual(data.state_mean[0], .01)
        self.assertEqual(data.train[1].ticks.tolist(), [0, 1])
        self.assertEqual(data.train[2].ticks.tolist(), [0])
        np.testing.assert_allclose(data.train[2].target[0, :, 0], [.01, .02])
        roll = KNOWN_NAMES.index('midpoint_roll_rad')
        np.testing.assert_array_equal(data.train[2].known[0, :, roll], [0., 0.])
        self.assertGreater(data.train[1].known[1, 0, roll], .09)
        self.assertEqual(data.train_endings.terminated.tolist(), [False, False, True])
        self.assertAlmostEqual(data.train_endings.departure_outside_s[-1], .04)
        self.assertAlmostEqual(data.train_endings.elapsed_s[-1], 5/960)

    def test_dev_only_preparation_skips_test_episode_files(self):
        scratch = ROOT/'.tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            entries = [self._episode(root, split, split, x, .1)
                       for x, split in enumerate(('train', 'validation', 'test'))]
            write_manifest(root, entries, 1/960, 16)
            (root/entries[2]['path']).unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_dataset(root, horizons=(1, 2))
            data = prepare_dataset(root, horizons=(1, 2), verify_test=False)
            np.testing.assert_array_equal(data.train[1].episode_ids, ['train', 'train'])
            np.testing.assert_array_equal(data.validation[1].episode_ids, ['validation', 'validation'])
            (root/entries[1]['path']).unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_dataset(root, horizons=(1, 2), verify_test=False)


if __name__ == '__main__':
    unittest.main()
