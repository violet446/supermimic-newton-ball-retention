"""Load complete v0.2 episodes and index them by disjoint collection groups."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile

import numpy as np

from .contracts import EPISODE_SCHEMA, RigidState, State, Transition, validate_window
from .task import ball_in_tool


MANIFEST_SCHEMA = 'ball-retention-dataset-v1'
SPLITS = ('train', 'validation', 'test')


def _parse_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f'non-finite JSON number: {token}')
    return value


def _load_json(data):
    return json.loads(data, parse_float=_parse_float, parse_constant=_parse_float)


def decode_transition(record):
    """Rebuild typed states from the serialized physical transition."""
    values = dict(record)
    for key in ('state_t', 'next_state_before_reset'):
        state = dict(values[key])
        state['ball'] = RigidState(**state['ball'])
        state['tool'] = RigidState(**state['tool'])
        values[key] = State(**state)
    return Transition(**values)


def _read_episode(path, dt, substeps, expected_id):
    if not math.isfinite(dt) or dt <= 0 or type(substeps) is not int or substeps <= 0:
        raise ValueError('expected positive physics dt and control substeps')
    rows = []
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for line_no, line in enumerate(stream, 1):
            digest.update(line)
            try:
                row = decode_transition(_load_json(line))
                if row.schema_version != EPISODE_SCHEMA:
                    raise ValueError('expected transition schema v0.2')
                if not isinstance(row.episode_id, str) or not row.episode_id:
                    raise ValueError('missing episode ID')
                if type(row.step_index) is not int:
                    raise ValueError('step index must be an integer')
                if any(type(flag) is not bool for flag in (row.terminated, row.truncated)):
                    raise ValueError('terminal flags must be boolean')
                if any(not isinstance(value, dict) for value in (row.context, row.reference_t, row.task)):
                    raise ValueError('context, reference and task fields must be objects')
                success = row.task.get('success')
                if type(success) is not bool or (success and not (row.truncated and row.task.get('motion_complete') is True)):
                    raise ValueError('success label contradicts the episode boundary')
                reason = row.task.get('termination_reason')
                if row.terminated:
                    if not isinstance(reason, str) or not reason:
                        raise ValueError('physical termination requires a reason')
                elif reason is not None:
                    raise ValueError('termination reason requires physical termination')
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'{path}:{line_no}: {exc}') from exc
            rows.append(row)

    validate_window(rows, dt, substeps)
    first, last = rows[0], rows[-1]
    if first.step_index != 0 or first.state_t.time_s != 0:
        raise ValueError('episode must start at step 0 and time 0')
    if expected_id is not None and first.episode_id != expected_id:
        raise ValueError(f'expected episode {expected_id}, got {first.episode_id}')
    if not (last.terminated or last.truncated):
        raise ValueError('episode has no ending')
    if any(row.context != first.context for row in rows[1:]):
        raise ValueError('physical context changed within the episode')
    return rows, digest.hexdigest()


def load_episode(path, dt, substeps, expected_id=None):
    """Read a complete validated episode for replay or window construction."""
    rows, _ = _read_episode(path, dt, substeps, expected_id)
    return rows


def validate_episode(path, dt, substeps, expected_id=None):
    """Verify the trajectory and summarize its boundaries, action and ball coverage."""
    rows, sha256 = _read_episode(path, dt, substeps, expected_id)
    first, last = rows[0], rows[-1]
    states = [first.state_t]+[row.next_state_before_reset for row in rows]
    positions = np.stack([ball_in_tool(state.ball, state.tool)['position'] for state in states])
    actions = np.asarray([row.requested_action for row in rows], dtype=float)
    tilts = np.asarray([state.tilt_rad for state in states], dtype=float)
    return {'episode_id': first.episode_id, 'sha256': sha256,
            'transitions': len(rows), 'physics_steps': sum(len(row.executed_joint_q) for row in rows),
            'elapsed_s': last.next_state_before_reset.time_s, 'terminal_substeps': len(last.executed_joint_q),
            'terminated': last.terminated, 'truncated': last.truncated, 'success': last.task['success'],
            'termination_reason': last.task.get('termination_reason'),
            'requested_action_min': actions.min(axis=0).tolist(), 'requested_action_max': actions.max(axis=0).tolist(),
            'requested_action_std': actions.std(axis=0).tolist(),
            'nonzero_action_steps': int(np.count_nonzero(np.any(actions != 0, axis=1))),
            'max_abs_tilt_rad': np.max(np.abs(tilts), axis=0).tolist(),
            'ball_tool_position_min_m': positions.min(axis=0).tolist(),
            'ball_tool_position_max_m': positions.max(axis=0).tolist(),
            'max_ball_tool_displacement_m': float(np.linalg.norm(positions-positions[0], axis=1).max())}


def _entry_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('episode path must be relative POSIX')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('episode path must stay inside dataset root')
    return path


def validate_split_entries(entries):
    """Enforce unique episode identities and keep every group in one split."""
    if not entries:
        raise ValueError('dataset requires episodes')
    groups, ids, paths = {}, set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('dataset entry must be an object')
        episode_id, group_id, split = (entry.get(key) for key in ('episode_id', 'group_id', 'split'))
        if any(not isinstance(value, str) or not value.strip() for value in (episode_id, group_id)):
            raise ValueError('missing episode or group identity')
        if split not in SPLITS:
            raise ValueError(f'invalid split: {split}')
        path = _entry_path(entry.get('path')).as_posix().casefold()
        if episode_id in ids or path in paths:
            raise ValueError('duplicate episode identity or path')
        if groups.setdefault(group_id, split) != split:
            raise ValueError(f'group {group_id} crosses dataset splits')
        ids.add(episode_id)
        paths.add(path)


def _verify_entries(root, entries, dt, substeps):
    validate_split_entries(entries)
    checked = []
    for entry in entries:
        relative = _entry_path(entry['path'])
        path = (root/relative).resolve()
        path.relative_to(root)
        stats = validate_episode(path, dt, substeps, entry['episode_id'])
        if 'sha256' in entry and entry['sha256'] != stats['sha256']:
            raise ValueError(f'episode SHA-256 mismatch: {entry["path"]}')
        checked.append({**entry, **stats, 'path': relative.as_posix()})
    return checked


def _split_summary(entries):
    summary = {}
    for split in SPLITS:
        rows = [entry for entry in entries if entry['split'] == split]
        summary[split] = {'episodes': len(rows), 'groups': len({row['group_id'] for row in rows}),
                          'transitions': sum(row['transitions'] for row in rows),
                          'physics_steps': sum(row['physics_steps'] for row in rows),
                          'terminated': sum(row['terminated'] for row in rows),
                          'truncated': sum(row['truncated'] for row in rows),
                          'successes': sum(row['success'] for row in rows)}
    return summary


def write_manifest(dataset_root, entries, dt, substeps, provenance=None):
    """Publish manifest.json atomically after validating all referenced episodes."""
    root = Path(dataset_root).resolve()
    checked = _verify_entries(root, list(entries), dt, substeps)
    manifest = {'schema_version': MANIFEST_SCHEMA, 'transition_schema': EPISODE_SCHEMA,
                'dt_s': dt, 'control_substeps': substeps, 'entries': checked,
                'splits': _split_summary(checked), 'provenance': provenance if provenance is not None else {}}
    serialized = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    with tempfile.NamedTemporaryFile(dir=root, prefix='.manifest-', suffix='.tmp', mode='w',
                                     encoding='utf-8', delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(serialized)
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, root/'manifest.json')
    finally:
        temp.unlink(missing_ok=True)
    return manifest


def read_manifest(dataset_root, *, verify_splits=SPLITS):
    """Check all split identities and verify episode files for the selected splits."""
    root = Path(dataset_root).resolve()
    manifest = _load_json((root/'manifest.json').read_bytes())
    if (manifest.get('schema_version') != MANIFEST_SCHEMA
            or manifest.get('transition_schema') != EPISODE_SCHEMA):
        raise ValueError('dataset schema mismatch')
    entries = manifest['entries']
    if any('sha256' not in entry for entry in entries):
        raise ValueError('published episodes require a SHA-256 digest')
    validate_split_entries(entries)
    if not verify_splits or not set(verify_splits) <= set(SPLITS):
        raise ValueError('invalid manifest verification splits')
    selected = [entry for entry in entries if entry['split'] in verify_splits]
    checked = _verify_entries(root, selected, manifest['dt_s'], manifest['control_substeps'])
    if checked != selected or _split_summary(entries) != manifest['splits']:
        raise ValueError('manifest summaries disagree with episode contents')
    return manifest
