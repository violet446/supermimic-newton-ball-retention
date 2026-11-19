"""Join immutable amplitude views without crossing physical initial groups."""
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from .pair_data import PairBatch, prepare_pairs


@dataclass(frozen=True)
class MixedPairs:
    batch: PairBatch
    amplitudes: np.ndarray
    cohorts: np.ndarray
    sample_weights: np.ndarray


@dataclass(frozen=True)
class PairCollection:
    train: MixedPairs
    validation: MixedPairs
    dt_s: float
    horizon: int
    model_config: dict
    report: dict


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _signature(value):
    return _digest(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def _verified(root, relative, expected):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or ':' in relative or '\\' in relative:
        raise ValueError('collection path must stay under its declared root')
    full = (root/path).resolve()
    full.relative_to(root)
    raw = full.read_bytes()
    if _digest(raw)!=expected:
        raise ValueError('collection source digest mismatch')
    return full,json.loads(raw)


def _initial_key(cfg):
    initial = cfg['initial']
    # Defaults must canonicalize identically to explicit zero velocity. Seeds,
    # schedules and episode names cannot turn one physical initial state into
    # separate train/validation groups.
    state = {k:initial[k] for k in ('ball_center_world_m','joint_q_rad','tool_rotation_world','anchor_world_m')}
    state['ball_velocity_world_m_s'] = initial.get('ball_velocity_world_m_s',[0.,0.,0.])
    state = {k:np.asarray(v,dtype=float).tolist() for k,v in state.items()}
    return _signature(state)


def prepare_collection(root, spec_path):
    """Load a hash-pinned collection; preserve cohort/amplitude provenance.

    Duplicate views are rejected, while two different amplitudes of one trace
    share group and snapshot identity. Each initial group receives equal mass,
    each snapshot equal mass within its group, each axis/amplitude equal mass
    within its snapshot. No normalization is fitted here.
    """
    root = Path(root).resolve()
    spec_path = Path(spec_path).resolve()
    spec_path.relative_to(root)
    spec = json.loads(spec_path.read_bytes())
    if spec.get('schema')!='s3-pair-collection-v1' or not spec.get('sources'):
        raise ValueError('invalid pair collection spec')
    mappings = {name:{} for name in ('group','episode','trace','initial','snapshot')}
    group_initials,initial_groups = {},{}
    def bind(kind,key,split):
        if mappings[kind].setdefault(key,split)!=split:
            raise ValueError(f'{kind} crosses splits across collection sources')
    def bind_config(cfg,group,split):
        key = _initial_key(cfg)
        bind('initial',key,split)
        if initial_groups.setdefault(key,group)!=group:
            raise ValueError('duplicate physical initial state under different group names')
        if group_initials.setdefault(group,key)!=key:
            raise ValueError('group names different physical initial states')

    # Read only reference metadata/configs, including test identity; no test
    # trajectories, metrics or targets are loaded.
    if 'anchor_dataset' in spec:
        entry = spec['anchor_dataset']
        anchor_path,manifest = _verified(root,entry['manifest'],entry['sha256'])
        for row in manifest['entries']:
            split,group = row['split'],row['group_id']
            bind('group',group,split)
            bind('episode',row['episode_id'],(group,split))
            provenance = row['provenance']
            _,cfg = _verified(anchor_path.parent,provenance['config'],provenance['config_sha256'])
            bind_config(cfg,group,split)
    prepared, seen_views, seen_rows, clock, model_cfg = [],set(),set(),None,None
    for source in spec['sources']:
        cohort = source['cohort']
        if not isinstance(cohort,str) or not cohort:
            raise ValueError('source cohort must be named')
        manifest_path,manifest = _verified(root,source['manifest'],source['sha256'])
        data = prepare_pairs(manifest_path)
        if clock is None:
            clock,model_cfg = (data.dt_s,data.horizon),data.model_config
        if clock!=(data.dt_s,data.horizon) or model_cfg!=data.model_config:
            raise ValueError('collection changed physical configuration or control clock')
        for entry in manifest['entries']:
            group,split,episode = entry['group_id'],entry['split'],entry['episode_id']
            bind('group',group,split)
            bind('episode',episode,(group,split))
            bind('trace',entry['sha256'],(group,split))
            view = entry['sha256'],data.amplitude
            if view in seen_views:
                raise ValueError('duplicate trace/amplitude view')
            seen_views.add(view)
            _,cfg = _verified(manifest_path.parent,entry['config_path'],entry['config_sha256'])
            bind_config(cfg,group,split)
        for split in ('train','validation'):
            batch = getattr(data,split)
            for index,state in enumerate(batch.initial_state):
                # Ignore bookkeeping when recognizing the physical snapshot.
                physical = {k:v for k,v in state.items() if k not in ('episode_id',)}
                signature = _signature(physical)
                bind('snapshot',signature,(str(batch.group_ids[index]),split))
                key = signature,int(batch.axes[index]),data.amplitude
                if key in seen_rows:
                    raise ValueError('duplicate physical snapshot/axis/amplitude')
                seen_rows.add(key)
        prepared.append((cohort,data))
    batches,report = {},{'sources':spec['sources'],'normalization_fitted':False,'splits':{}}
    for split in ('train','validation'):
        inputs = [getattr(d,split) for _,d in prepared]
        batch = PairBatch(**{f.name:sum((getattr(b,f.name) for b in inputs),()) if f.name=='initial_state'
                            else np.concatenate([getattr(b,f.name) for b in inputs]) for f in fields(PairBatch)})
        amplitudes = np.concatenate([np.full(len(getattr(d,split).initial),d.amplitude) for _,d in prepared])
        cohorts = np.concatenate([np.full(len(getattr(d,split).initial),c) for c,d in prepared])
        weights = np.zeros(len(batch.initial))
        group_ids = np.unique(batch.group_ids)
        for group in group_ids:
            indices = np.flatnonzero(batch.group_ids==group)
            snapshots = {}
            for i in indices:
                snapshots.setdefault(_signature(batch.initial_state[i]),[]).append(i)
            for rows in snapshots.values():
                weights[rows] = 1/(len(group_ids)*len(snapshots)*len(rows))
        batches[split] = MixedPairs(batch,amplitudes,cohorts,weights)
        report['splits'][split] = {'groups':len(group_ids),
            'snapshots':len(set(_signature(s) for s in batch.initial_state)),
            'axis_amplitude_pairs':len(batch.initial),
            'amplitude_counts':{str(a):int(np.sum(amplitudes==a)) for a in np.unique(amplitudes)},
            'group_weight_mass':{str(g):float(weights[batch.group_ids==g].sum()) for g in group_ids},
            'cohort_counts':{str(c):int(np.sum(cohorts==c)) for c in np.unique(cohorts)}}
    return PairCollection(batches['train'],batches['validation'],*clock,model_cfg,report)
