"""Collect complete S3 episodes with a native checkpoint at every control boundary."""
from dataclasses import asdict
import ctypes
import hashlib
import json
import os
import time

import numpy as np

from check_s3_deterministic_archive_import import ROOT
from ball_retention.archive_state_import import _check_layout, _parse_state
from ball_retention.dataset import decode_transition, validate_episode
from ball_retention.resources import MemoryStatus, admission, snapshot
from native_repeatability import _make_env
from saved_native_snapshot import restore_and_readback
from s3_closed_loop_feedback import feedback_action


HORIZON, SUBSTEPS, WARMUP = 240, 16, 60
DT = 1/960
TRACE = 'transitions.jsonl'
CHECKPOINT = 'checkpoint.json'


def encode(document):
    return (json.dumps(document, allow_nan=False, separators=(',', ':'))+'\n').encode('utf-8')


def atomic_json(path, document):
    temp = path.with_suffix('.tmp')
    temp.write_bytes(encode(document))
    deadline = time.monotonic()+.1
    while True:
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.005)


def action(case, tick, state, feedback):
    if case['policy'] == 'feedback' and tick >= WARMUP:
        return feedback_action(state, feedback)
    for start, end, values in case.get('pulses', []):
        if start <= tick < end:
            return values
    return [0., 0.]


def native_document(env):
    saved = env.snapshot()
    saved['state'] = asdict(saved['state'])
    physics = saved['physics']
    physics['arrays'] = {name: array.tolist() for name, array in physics['arrays'].items()}
    physics['command'] = [array.tolist() for array in physics['command']]
    return saved


def restore(env, saved, layout, name):
    tick = saved['step_index']
    physical = saved['physics']
    public = _parse_state(saved['state'])
    if (saved['episode_id'] != name or type(tick) is not int or not 0 <= tick < HORIZON
            or saved['done'] is not False or saved['invalid'] is not False
            or type(saved['deep_count']) is not int or not 0 <= saved['deep_count'] < 3
            or physical['steps'] != tick*SUBSTEPS or public.time_s != tick*SUBSTEPS*DT
            or saved['monitor']['terminated'] is not False
            or saved['monitor']['last_time_s'] != public.time_s):
        raise ValueError('native checkpoint has an invalid control boundary')
    specs = layout['state_arrays']
    raw = physical['arrays']
    if raw.keys() != specs.keys():
        raise ValueError('native checkpoint arrays differ from live layout')
    arrays = {key: np.asarray(raw[key], dtype=spec['dtype']) for key, spec in specs.items()}
    command = np.asarray(physical['command'], dtype=float)
    if (command.shape != (3, 6) or not np.isfinite(command).all()
            or any(list(arrays[key].shape) != spec['shape'] or not np.isfinite(arrays[key]).all()
                   for key, spec in specs.items())):
        raise ValueError('native checkpoint has invalid arrays or commands')
    restored = {**saved, 'state': public, 'physics': {**physical, 'arrays': arrays, 'command': command}}
    return restore_and_readback(env, restored, SUBSTEPS)


def checkpoint(out, name):
    """Verify the committed prefix and distinguish an interrupted append."""
    path = out/CHECKPOINT
    if not path.exists():
        return None, b''
    saved = json.loads(path.read_bytes())
    trace = out/TRACE
    data = trace.read_bytes() if trace.exists() else b''
    if saved['episode_id'] != name or not 0 <= saved['trace_bytes'] <= len(data):
        raise ValueError('checkpoint and trace do not identify the same committed prefix')
    prefix = data[:saved['trace_bytes']]
    lines = prefix.splitlines()
    if (hashlib.sha256(prefix).hexdigest() != saved['trace_sha256'] or len(lines) != saved['control_rows']
            or saved['committed_physics_steps'] > len(lines)*SUBSTEPS
            or len(lines) > HORIZON):
        raise ValueError('checkpoint prefix hash or counts differ')
    if lines:
        last = decode_transition(json.loads(lines[-1]))
        last.validate(DT, SUBSTEPS)
        if (last.episode_id != name or last.step_index+1 != len(lines)
                or saved['committed_physics_steps'] != last.step_index*SUBSTEPS+len(last.executed_joint_q)
                or saved['terminal'] != (last.terminated or last.truncated)
                or (not saved['terminal'] and saved['native']['state'] != asdict(last.next_state_before_reset))):
            raise ValueError('checkpoint and committed transition disagree')
    elif saved['committed_physics_steps'] != 0 or saved['terminal']:
        raise ValueError('empty checkpoint cannot have advanced physics')
    return saved, prefix


def wait_for_control_boundary_ram(budget):
    """Pause new control cycles on low RAM while the parent hard guard stays active."""
    began = time.monotonic()
    floor = budget['min_running_ram_gib']*1024**3
    while True:
        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError('GlobalMemoryStatusEx failed')
        if memory.avail_phys >= floor:
            return
        if (memory.avail_phys < budget['critical_ram_gib']*1024**3
                or time.monotonic()-began >= budget['low_ram_grace_s']):
            raise RuntimeError('system_available_ram_floor_at_control_boundary')
        time.sleep(.05)


def collect(output, cfg, plan, case, stage, state, max_new_steps):
    name = case['episode_id']
    folder = output.parent.parent
    saved, prefix = checkpoint(folder, name)
    start_tick = saved['control_rows'] if saved else 0
    prior_steps = saved['committed_physics_steps'] if saved else 0
    state.update(start_tick=start_tick, start_physics_steps=prior_steps,
                 reserved_new_steps=0, actual_new_steps=0)
    stage('resource_recheck')
    if not admission(snapshot(ROOT), cfg['budget'])['passed']:
        raise RuntimeError('worker resource recheck failed before Newton imports')
    if saved and saved['terminal']:
        raise ValueError('terminal episode should be finalized from its trace')
    env = _make_env(ROOT, cfg, state, stage, expected_device_arch=89)
    first = env.reset(episode_id=name)
    first.validate()
    if (env.substeps, env.horizon_steps, env.dt, env.physics.steps, env.route_end) != (SUBSTEPS, HORIZON, DT, 0, 4.):
        raise ValueError('worker episode clock differs from the frozen route')
    layout = _check_layout(env.physics)
    trace = folder/TRACE
    hasher = hashlib.sha256(prefix)

    def persist(stream, terminal):
        document = {'episode_id': name, 'control_rows': env.step_index,
                    'committed_physics_steps': env.physics.steps, 'trace_bytes': stream.tell(),
                    'trace_sha256': hasher.hexdigest(), 'terminal': terminal,
                    'native': None if terminal else native_document(env)}
        atomic_json(folder/CHECKPOINT, document)
        return document

    if saved:
        proof = restore(env, saved['native'], layout, name)
        atomic_json(output/'restore_readback.json', proof)
        # Keep any interrupted append as evidence before restoring the committed prefix.
        tail = trace.read_bytes()[len(prefix):] if trace.exists() else b''
        if tail:
            (output/'discarded_uncommitted_tail.bin').write_bytes(tail)
    with trace.open('r+b' if trace.exists() else 'xb') as stream:
        stream.seek(len(prefix))
        stream.truncate()
        if not saved:
            document = persist(stream, False)
            atomic_json(output/'reset_readback.json', restore(env, document['native'], layout, name))
        stage('episode_ready')
        for tick in range(start_tick, HORIZON):
            wait_for_control_boundary_ram(cfg['budget'])
            request = action(case, tick, env.state, plan['feedback'])
            before = env.physics.steps
            reserve = before-prior_steps+SUBSTEPS
            if before != tick*SUBSTEPS or reserve > max_new_steps:
                raise RuntimeError('physical step cap before next control tick')
            state['reserved_new_steps'] = reserve
            stage('control_in_flight')
            try:
                row = env.step(request)
            finally:
                state['actual_new_steps'] = env.physics.steps-prior_steps
            if (row.step_index != tick or row.episode_id != name or row.invalid_reason is not None
                    or row.requested_action != request or row.next_state_before_reset.previous_action != request
                    or max(map(abs, request)) > plan['feedback']['correction_cap']
                    or len(row.executed_joint_q) != env.physics.steps-before
                    or env.done != (row.terminated or row.truncated)):
                raise ValueError('invalid control transition')
            line = encode(row.to_dict())
            stream.write(line)
            stream.flush()
            hasher.update(line)
            document = persist(stream, env.done)
            state['reserved_new_steps'] = state['actual_new_steps']
            state['last_committed_tick'] = env.step_index
            stage('control_committed')
            if not saved and tick == 0 and not env.done:
                # The bit-exact restore is exercised past reset, with no extra integration.
                atomic_json(output/'tick_001_readback.json', restore(env, document['native'], layout, name))
            if env.done:
                break
    state['episode'] = validate_episode(trace, DT, SUBSTEPS, name)
    if not env.done or state['episode']['physics_steps'] != prior_steps+state['actual_new_steps']:
        raise ValueError('terminal trace and physics counters disagree')
