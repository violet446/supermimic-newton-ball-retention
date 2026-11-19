import hashlib
import copy
import json
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.contracts import RigidState, State
from ball_retention.model_data import KNOWN_NAMES
from ball_retention.pair_data import SCHEMA, prepare_pairs


class PairDataTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT/'.tmp'
        scratch.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name)
        cfg = json.loads((ROOT/'configs/s1_online_episode.json').read_text())
        config = self._write('config.json', cfg)
        self.manifest = {'schema':SCHEMA, 'horizon':4, 'amplitude':.2, 'dt_s':1/960,
                         'control_substeps':16, 'entries':[]}
        for group, split in [('a','train'), ('b','validation')]:
            episode = f'episode_{group}'
            record = self._record(.28 if group=='a' else .27)
            trace = self._write(episode+'.jsonl', record)
            self.manifest['entries'].append({'group_id':group, 'split':split, 'episode_id':episode,
                'path':trace.name, 'sha256':self._sha(trace), 'config_path':config.name,
                'config_sha256':self._sha(config), 'snapshot_ticks':[0]})
        self.manifest_path = self._write('capture_manifest.json', self.manifest)

    def _write(self, name, value):
        path = self.out/name
        path.write_text(json.dumps(value)+'\n', encoding='utf-8')
        return path

    @staticmethod
    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _record(x):
        ball = RigidState([x,0.,0.], [0.,0.,0.,1.], [0.]*3, [0.]*3)
        tool = RigidState([0.]*3, [0.,0.,0.,1.], [0.]*3, [0.]*3)
        state = State(0.,ball,tool,[0.]*6,[0.]*6,[0.]*2,[0.]*2,[0.]*2,0.)
        initial = [x]+[0.]*8
        branches = []
        for axis, offset, repeat in [(None,0.,0),(None,0.,1),(0,-.2,0),(0,.2,0),(1,-.2,0),(1,.2,0)]:
            short = axis==0 and offset>0
            counts = [16,16,5] if short else [16]*4
            request = [0.,0.]
            if axis is not None:
                request[axis] = offset
            raw = np.tile(initial,(len(counts),1))
            if axis is None and repeat:
                raw[:,0] += .8e-6
            if axis==0:
                raw[:,0] += np.sign(offset)*np.array([3e-6,1e-6,4e-6,5e-6])[:len(counts)]
            branches.append({'axis':axis,'offset':offset,'repeat':repeat,'actions':[request]*len(counts),
                'states_raw':raw.tolist(),'substeps':counts,'elapsed_s':(np.cumsum(counts)/960).tolist(),
                'terminated':short,'truncated':False,'termination_reason':'outside_projection' if short else None})
        return {'scenario':'route','snapshot_tick':0,'initial_state':asdict(state),'initial_raw':initial,
                'nominal_actions':[[0.,0.]]*4,'branches':branches,'amplitudes':[.2],
                'physics_dt_s':1/960,'control_substeps':16}

    def test_matches_full_steps_and_masks_noise_without_losing_short_target(self):
        data = prepare_pairs(self.manifest_path)
        self.assertEqual(data.train.initial.shape,(2,9))
        self.assertEqual(data.train.plus_known.shape,(2,4,20))
        np.testing.assert_array_equal(data.train.plus_substeps[0],[16,16,5,0])
        np.testing.assert_array_equal(data.train.pair_mask[0],[True,True,False,False])
        np.testing.assert_array_equal(data.train.response_mask[0,:,0],[True,False,False,False])
        np.testing.assert_allclose(data.train.response_floor[0,:,0],2e-6,atol=1e-15)
        self.assertAlmostEqual(data.train.plus_target[0,2,0],.280004)
        self.assertTrue(data.train.plus_terminated[0])
        self.assertEqual(data.train.plus_target[0,3,0],0.)
        col = KNOWN_NAMES.index('roll_request')
        np.testing.assert_array_equal(data.train.plus_known[0,:,col],[.2]*4)
        self.assertEqual(data.report['splits']['train']['shortened_plus'],1)

    def test_timing_and_group_integrity(self):
        trace = self.out/'episode_a.jsonl'
        record = json.loads(trace.read_text())
        record['branches'][3]['elapsed_s'][2] = 3/60
        self._write(trace.name,record)
        with self.assertRaisesRegex(ValueError,'capture hash mismatch'):
            prepare_pairs(self.manifest_path)
        self.manifest['entries'][0]['sha256'] = self._sha(trace)
        self._write(self.manifest_path.name,self.manifest)
        with self.assertRaisesRegex(ValueError,'elapsed time disagrees'):
            prepare_pairs(self.manifest_path)
        self._write(trace.name,self._record(.28))
        self.manifest['entries'][0]['sha256'] = self._sha(trace)
        self.manifest['entries'][1]['group_id'] = 'a'
        self._write(self.manifest_path.name,self.manifest)
        with self.assertRaisesRegex(ValueError,'group crosses splits'):
            prepare_pairs(self.manifest_path)

    def test_two_amplitude_views_share_states_and_validate_unselected_branches(self):
        for entry in self.manifest['entries']:
            path = self.out/entry['path']
            record = json.loads(path.read_text())
            extra = []
            for branch in record['branches']:
                if branch['axis'] is None:
                    continue
                branch = copy.deepcopy(branch)
                branch['offset'] = np.sign(branch['offset'])*.35
                for action in branch['actions']:
                    action[branch['axis']] = branch['offset']
                extra.append(branch)
            record['branches'] += extra
            record['amplitudes'] = [.2,.35]
            self._write(path.name,record)
            entry['sha256'] = self._sha(path)
        self.manifest['record_amplitudes'] = [.2,.35]
        self._write(self.manifest_path.name,self.manifest)
        small = prepare_pairs(self.manifest_path)
        self.manifest['amplitude'] = .35
        self._write(self.manifest_path.name,self.manifest)
        large = prepare_pairs(self.manifest_path)
        np.testing.assert_array_equal(small.train.initial,large.train.initial)
        np.testing.assert_array_equal(small.validation.group_ids,large.validation.group_ids)
        np.testing.assert_array_equal(large.train.plus_known[0,:,2],.35)
        # Malformed data in the other amplitude must still invalidate this view.
        entry = self.manifest['entries'][0]
        path = self.out/entry['path']
        record = json.loads(path.read_text())
        record['branches'][3]['elapsed_s'][0] = 0.
        self._write(path.name,record)
        entry['sha256'] = self._sha(path)
        self._write(self.manifest_path.name,self.manifest)
        with self.assertRaisesRegex(ValueError,'elapsed time disagrees'):
            prepare_pairs(self.manifest_path)


if __name__=='__main__':
    unittest.main()
