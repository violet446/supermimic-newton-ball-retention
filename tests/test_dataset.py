import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from ball_retention.contracts import EPISODE_SCHEMA, RigidState, State, Transition
from ball_retention.dataset import load_episode, read_manifest, validate_episode, validate_split_entries, write_manifest


class DatasetTests(unittest.TestCase):
    dt, substeps = 1/960, 16

    def setUp(self):
        scratch = ROOT/'.tmp'
        scratch.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def episode(self, episode_id):
        body = RigidState([0., 0., 0.], [0., 0., 0., 1.], [0.]*3, [0.]*3)
        state = State(0., body, copy.deepcopy(body), [0.]*6, [0.]*6, [0.]*2, [0.]*2, [0.]*2, 0.)
        rows = []
        for step, count in enumerate((self.substeps, 3)):
            done = step == 1
            next_state = copy.deepcopy(state)
            next_state.time_s += count*self.dt
            next_state.ball.position[0] += .01
            rows.append(Transition(episode_id, step, state, [0., .2*step], [[0.]*6]*count, [[0.]*6]*count,
                next_state, done, False, None, None, {}, {}, schema_version=EPISODE_SCHEMA,
                task={'success': False, 'termination_reason': 'outside_projection' if done else None}).to_dict())
            state = next_state
        return rows

    def write(self, path, rows):
        (self.root/path).write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')

    def test_partial_terminal_preserves_actual_time_and_displacement(self):
        path = 'edge.jsonl'
        self.write(path, self.episode('edge'))
        report = validate_episode(self.root/path, self.dt, self.substeps, expected_id='edge')
        self.assertEqual(report['physics_steps'], 19)
        self.assertEqual(report['terminal_substeps'], 3)
        self.assertEqual(report['transitions'], 2)
        self.assertTrue(report['terminated'])
        self.assertAlmostEqual(report['elapsed_s'], 19*self.dt)
        self.assertAlmostEqual(report['max_ball_tool_displacement_m'], .02)
        self.assertEqual(report['requested_action_max'], [0., .2])
        self.assertEqual(len(load_episode(self.root/path, self.dt, self.substeps)), 2)

    def test_incomplete_episodes_and_nonfinite_values_are_rejected(self):
        path = 'edge.jsonl'
        rows = self.episode('edge')
        self.write(path, rows[:1])
        with self.assertRaisesRegex(ValueError, 'no ending'):
            validate_episode(self.root/path, self.dt, self.substeps)
        rows[-1]['task']['diagnostic'] = {'nonfinite': float('nan')}
        self.write(path, rows)
        with self.assertRaisesRegex(ValueError, 'non-finite JSON'):
            validate_episode(self.root/path, self.dt, self.substeps)

    def test_manifest_checks_group_leakage_and_episode_digest(self):
        entries = [dict(episode_id=name, group_id=group, split=split, path=f'{name}.jsonl')
                   for name, group, split in (('a', 'pair', 'train'), ('b', 'pair', 'train'),
                                              ('c', 'held-out', 'validation'))]
        for entry in entries:
            self.write(entry['path'], self.episode(entry['episode_id']))
        manifest = write_manifest(self.root, entries, self.dt, self.substeps, {'seed': 20261111})
        self.assertEqual(manifest['splits']['train']['groups'], 1)
        self.assertEqual(manifest['splits']['train']['episodes'], 2)
        self.assertEqual(read_manifest(self.root), manifest)
        leaked = copy.deepcopy(entries)
        leaked[1]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'crosses dataset splits'):
            write_manifest(self.root, leaked, self.dt, self.substeps)
        self.assertEqual(read_manifest(self.root), manifest)
        leaked[1]['split'] = 'train'
        leaked[1]['path'] = '../escape.jsonl'
        with self.assertRaisesRegex(ValueError, 'inside dataset root'):
            validate_split_entries(leaked)
        path = self.root/'a.jsonl'
        path.write_bytes(b' '+path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            read_manifest(self.root)


if __name__ == '__main__':
    unittest.main()
