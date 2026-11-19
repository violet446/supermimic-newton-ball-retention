import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ball_retention.pair_collection import prepare_collection
import test_pair_data


class PairCollectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT/'.tmp')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self,path,value):
        path.write_text(json.dumps(value)+'\n',encoding='utf-8')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def source(self,name,positions,amplitudes=(.2,)):
        folder = self.root/name
        folder.mkdir()
        entries = []
        for i,(split,x) in enumerate(zip(('train','validation'),positions)):
            cfg = json.loads((ROOT/'configs/s1_online_episode.json').read_text())
            cfg['initial']['ball_center_world_m'] = [x,0.,0.]
            record = test_pair_data.PairDataTests._record(x)
            for amplitude in amplitudes[1:]:
                for branch in list(record['branches'][2:6]):
                    branch = copy.deepcopy(branch)
                    branch['offset'] = float(np.sign(branch['offset'])*amplitude)
                    for action in branch['actions']:
                        action[branch['axis']] = branch['offset']
                    record['branches'].append(branch)
            record['amplitudes'] = list(amplitudes)
            cfg_name,trace_name = f'{i}.json',f'{i}.jsonl'
            entries.append({'episode_id':f'{name}_{i}','group_id':f'{name}_{i}','split':split,
                'config_path':cfg_name,'config_sha256':self.write(folder/cfg_name,cfg),
                'path':trace_name,'sha256':self.write(folder/trace_name,record),'snapshot_ticks':[0]})
        views = []
        for a in amplitudes:
            manifest = {'schema':'s3-long-pairs-v1','horizon':4,'amplitude':a,'record_amplitudes':list(amplitudes),
                        'dt_s':1/960,'control_substeps':16,'entries':entries}
            path = folder/f'{a}.json'
            sha = self.write(path,manifest)
            views.append({'manifest':path.relative_to(self.root).as_posix(),'sha256':sha,'cohort':name})
        return views

    def load(self,sources):
        self.write(self.root/'collection.json',{'schema':'s3-pair-collection-v1','sources':sources})
        return prepare_collection(self.root,self.root/'collection.json')

    def test_amplitudes_share_group_mass_and_provenance(self):
        sources = self.source('old',(.28,.27))+self.source('new',(.26,.25),(.2,.35))
        data = self.load(sources)
        self.assertEqual(len(data.train.batch.initial),6)
        self.assertEqual(data.report['splits']['train']['snapshots'],2)
        np.testing.assert_allclose(data.train.sample_weights,[.25,.25,.125,.125,.125,.125])
        np.testing.assert_array_equal(data.train.amplitudes,[.2,.2,.2,.2,.35,.35])
        np.testing.assert_allclose(data.train.sample_weights.sum(),1.)
        self.assertFalse(data.report['normalization_fitted'])

    def test_duplicate_views_and_digest_change_are_rejected(self):
        source = self.source('old',(.28,.27))
        with self.assertRaisesRegex(ValueError,'duplicate trace/amplitude'):
            self.load(source+source)
        source[0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'digest mismatch'):
            self.load(source)

    def test_renaming_groups_cannot_hide_initial_state_leakage(self):
        old = self.source('old',(.28,.27))
        renamed = self.source('renamed',(.27,.28))
        with self.assertRaisesRegex(ValueError,'crosses splits'):
            self.load(old+renamed)

    def test_same_snapshot_cannot_be_relabelled_as_a_new_sample(self):
        old = self.source('old',(.28,.27))
        renamed = self.source('renamed',(.28,.27))
        with self.assertRaisesRegex(ValueError,'crosses splits|duplicate'):
            self.load(old+renamed)

    def test_distinct_history_snapshots_share_one_initial_group(self):
        old=self.source('old',(.28,.27));new=self.source('history',(.281,.271))
        path=self.root/new[0]['manifest'];manifest=json.loads(path.read_text())
        for i,entry in enumerate(manifest['entries']):
            cfg_path=path.parent/entry['config_path'];cfg=json.loads(cfg_path.read_text())
            cfg['initial']['ball_center_world_m']=[(.28,.27)[i],0.,0.]
            entry['config_sha256']=self.write(cfg_path,cfg)
            entry['group_id']=f'old_{i}'
        new[0]['sha256']=self.write(path,manifest)
        data=self.load(old+new)
        self.assertEqual(data.report['splits']['train']['groups'],1)
        self.assertEqual(data.report['splits']['train']['snapshots'],2)
        manifest['entries'][0]['group_id']='aliased_group'
        new[0]['sha256']=self.write(path,manifest)
        with self.assertRaisesRegex(ValueError,'duplicate physical initial'):
            self.load(old+new)
