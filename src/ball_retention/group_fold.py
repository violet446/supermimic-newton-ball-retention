"""Leakage-free fold preparation for the existing training split only."""
from dataclasses import dataclass,fields
import numpy as np
from .model_data import Windows
from .pair_data import PairBatch


@dataclass(frozen=True)
class GroupFold:
    anchors: dict
    pairs: PairBatch
    retained_indices: np.ndarray
    held_indices: np.ndarray
    pair_weights: np.ndarray
    state_mean: np.ndarray
    state_scale: np.ndarray
    known_mean: np.ndarray
    known_scale: np.ndarray
    report: dict


def subset_pairs(batch,indices):
    return PairBatch(**{f.name:tuple(batch.initial_state[i] for i in indices) if f.name=='initial_state'
                        else getattr(batch,f.name)[indices] for f in fields(PairBatch)})


def _moments(chunks, floor):
    """Stable weighted moments; each chunk specifies its total probability mass."""
    values=np.concatenate([v for v,_ in chunks])
    weights=np.concatenate([np.full(len(v),mass/len(v)) for v,mass in chunks])
    weights/=weights.sum()
    if not np.isfinite(values).all(): raise ValueError('nonfinite training statistics')
    mean=np.sum(values*weights[:,None],axis=0)
    scale=np.sqrt(np.sum((values-mean)**2*weights[:,None],axis=0))
    return mean,np.maximum(scale,floor)


def _training_statistics(anchors,pairs,pair_weights,episode_groups):
    states=[];known=[]
    for w in anchors.values():
        group_ids=np.array([episode_groups[str(e)] for e in w.episode_ids])
        groups=np.unique(group_ids)
        for group in groups:
            episodes=np.unique(w.episode_ids[group_ids==group])
            for episode in episodes:
                selected=w.episode_ids==episode
                mass=.5/(len(anchors)*len(groups)*len(episodes))
                # Initial and observed trajectory states establish a training-only
                # coordinate scale; targets are never fed back during rollout.
                states.append((np.concatenate((w.initial[selected,None],w.target[selected]),axis=1).reshape(-1,9),mass))
                known.append((w.known[selected].reshape(-1,w.known.shape[-1]),mass))
    for i,weight in enumerate(pair_weights):
        for features,target,mask in ((pairs.plus_known,pairs.plus_target,pairs.plus_mask),
                                     (pairs.minus_known,pairs.minus_target,pairs.minus_mask)):
            mass=.25*weight
            states.append((np.concatenate((pairs.initial[i,None],target[i,mask[i]])),mass))
            # The complete causal control schedule is evaluated by the network,
            # even when a physical target mask excludes a shortened branch.
            known.append((features[i],mass))
    sm,ss=_moments(states,np.repeat([1e-4,1e-3,1e-2],3))
    km,ks=_moments(known,1e-4)
    return sm,ss,km,ks


def make_group_fold(train_windows,train_pairs,episode_groups,held_group=None,
                    *,normalization='retained_training_v2'):
    """Refit statistics after filtering, using only retained training groups.

    No incoming normalization or checkpoint is accepted. Entire paired initial
    groups are removed, including both amplitudes and both axes.
    """
    if normalization not in ('s2_only_v1','retained_training_v2'):
        raise ValueError('unknown normalization policy')
    groups=set(train_pairs.group_ids)
    if held_group is not None and held_group not in groups:
        raise ValueError('held group is not in the pair training split')
    anchors={}
    for h,w in train_windows.items():
        if any(str(i) not in episode_groups for i in w.episode_ids):
            raise ValueError('anchor episode group missing')
        keep=np.array([episode_groups[str(i)]!=held_group for i in w.episode_ids])
        if not keep.any(): raise ValueError('fold has no retained anchor windows')
        anchors[h]=Windows(*(getattr(w,f.name)[keep] for f in fields(Windows)))
    keep=np.flatnonzero(train_pairs.group_ids!=held_group)
    held=np.flatnonzero(train_pairs.group_ids==held_group)
    if not len(keep): raise ValueError('fold has no retained paired groups')
    pairs=subset_pairs(train_pairs,keep)
    weights=np.zeros(len(keep))
    retained_groups=np.unique(pairs.group_ids)
    for group in retained_groups:
        rows=np.flatnonzero(pairs.group_ids==group)
        snapshots=sorted(set(zip(pairs.episode_ids[rows],pairs.snapshot_ticks[rows])))
        for episode,tick in snapshots:
            indices=rows[(pairs.episode_ids[rows]==episode)&(pairs.snapshot_ticks[rows]==tick)]
            weights[indices]=1/(len(retained_groups)*len(snapshots)*len(indices))
    w=anchors[1]
    states=np.concatenate((w.initial,w.target[:,0]))
    known=w.known[:,0]
    if normalization=='s2_only_v1':
        stats=(states.mean(0),np.maximum(states.std(0),np.repeat([1e-4,1e-3,1e-2],3)),
               known.mean(0),np.maximum(known.std(0),1e-4))
        policy='retained S2 train one-step initial/target states and causal known features only'
    else:
        stats=_training_statistics(anchors,pairs,weights,episode_groups)
        policy={'source_mass':{'S2':.5,'pairs':.5},
                'S2':'equal horizons, initial groups, episodes within group, windows, and times',
                'pairs':'equal groups, snapshots within group, axis/amplitude rows, branches, and times',
                'state_samples':'initial and valid observed training targets; no rollout truth feedback',
                'known_samples':'all causal features evaluated in retained training windows/branches',
                'state_scale_floor_si':np.repeat([1e-4,1e-3,1e-2],3).tolist(),
                'known_scale_floor_in_feature_units':1e-4}
    report={'held_group':held_group,'retained_pair_groups':retained_groups.tolist(),
            'retained_anchor_episodes':sorted(set(w.episode_ids.tolist())),
            'anchor_window_counts':{str(h):len(w.initial) for h,w in anchors.items()},
            'retained_axis_amplitude_pairs':len(keep),'held_axis_amplitude_pairs':len(held),
            'normalization':normalization,'normalization_source':policy,
            'pretrained_weights_used':False}
    return GroupFold(anchors,pairs,keep,held,weights,*stats,report)
