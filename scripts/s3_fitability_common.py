"""Shared frozen inputs and resource guard for the offline fitability experiments."""
import ctypes
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sys
from threading import Event, Thread
import time

STARTED = time.monotonic()
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from audit_s3_velocity_training_coverage import DESIGN, DESIGN_SHA256, FIT, FIT_SHA, ROOT, metadata
from preflight_s3_velocity_response_leads import prepare_comparison, write_json

import numpy as np

from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES
from ball_retention.position_integral_model import load_position_model
from ball_retention.resources import MemoryStatus, directory_bytes
from ball_retention.timed_pair_data import prepare_timed_pairs
from check_s3_actions import digest


BUDGET = {'cpu_threads': 1, 'gpu_compute': False, 'worker_starts': 1, 'max_wall_s': 120,
          'max_rss_gib': 1, 'max_private_commit_gib': 1.5, 'max_new_output_mib': 32,
          'min_start_ram_gib': 2, 'min_running_ram_gib': .75}
SCOPE = {'simulation_steps': 0, 'test_episode_files_read': False, 'control_accepted': False}
SEED = 20261111
PARTS = {'position': slice(0, 3), 'velocity': slice(3, 6), 'spin': slice(6, 9)}


def frozen_data(plan, guard):
    design = json.loads(_verified_bytes(ROOT, DESIGN, DESIGN_SHA256))
    parent, cfg, scales, fold, legacy, _, matched, draws, _ = prepare_comparison(design, guard)
    accepted = json.loads(_verified_bytes(ROOT, FIT, FIT_SHA))
    if (accepted['status'] != 'completed' or accepted['updates'] != 900
            or accepted['anchor_draws_sha256'] != draws or any(accepted[k] != v for k, v in SCOPE.items())):
        raise ValueError('unexpected dense velocity acceptance')
    ck = FIT.parent/'velocity_response_leads_900.npz'
    if plan['files_sha256'][ck.as_posix()] != accepted['artifacts_sha256'][ck.name]:
        raise ValueError('dense checkpoint hash differs from acceptance')
    manifest = parent['timed_manifest']
    timed = prepare_timed_pairs(ROOT, manifest, parent['inputs_sha256'][manifest])
    history = cfg['history_plan']
    rows, _ = metadata(fold, legacy, timed, json.loads(_verified_bytes(ROOT, history, cfg['inputs_sha256'][history])))
    if (len(rows) != 46 or [row['row'] for row in rows if row['timed']] != list(range(40, 46))
            or set(fold.pairs.group_ids) & set(legacy.validation.batch.group_ids)):
        raise ValueError('unexpected scope or held group in training inputs')
    with np.load(ROOT/ck, allow_pickle=False) as saved:
        dense = load_position_model(saved)
        for key in ('state_mean', 'state_scale', 'known_mean', 'known_scale'):
            np.testing.assert_array_equal(saved[key], getattr(fold, key))
        np.testing.assert_array_equal(saved['known_names'], KNOWN_NAMES)
        if type(dense) is not type(matched) or dense.dt_s != matched.dt_s or int(saved['updates']) != 900:
            raise ValueError('unexpected dense velocity checkpoint')
    return design, parent, cfg, scales, fold, rows, matched, dense


def branch_arrays(fold):
    batch = fold.pairs
    features = (np.stack((batch.plus_known, batch.minus_known))-fold.known_mean)/fold.known_scale
    truth = np.stack((batch.plus_target, batch.minus_target))
    mask = np.stack((batch.plus_mask, batch.minus_mask))
    return batch.initial, features, truth, mask


def forecast(model, initial, features, truth, *, local=False):
    """Use truth in a separate independent one-step diagnostic only."""
    if local:
        prior = np.concatenate((initial[:, None], truth[:, :-1]), axis=1)
        predicted = model.step(model.normalize(prior), features)
    else:
        predicted = model.rollout(model.normalize(initial), features)
    return model.denormalize(predicted)


def norm_stats(values, weights):
    """Summarize three-dimensional vector errors in physical units."""
    weight = np.broadcast_to(weights, values.shape[:-1])
    return {name: float(np.sqrt(np.sum(weight*np.sum(values[..., part]**2, axis=-1))/weight.sum()))
            for name, part in PARTS.items()}


def run_phase(path, expected_sha, phase, work):
    name = path.relative_to(ROOT).as_posix()
    plan = json.loads(_verified_bytes(ROOT, name, expected_sha))
    if (plan['phase'] != phase or plan['seed'] != SEED or plan['budget'] != BUDGET
            or plan['scope'] != SCOPE or not 0 <= plan['updates'] <= (0 if phase == 'P0' else 1800)):
        raise ValueError('plan exceeds frozen execution scope')
    out = (ROOT/plan['output']).resolve()
    if len(out.relative_to(ROOT/'runs').parts) != 1:
        raise ValueError('expected a new run directory directly under runs')
    out.mkdir(exist_ok=False)
    stage, update = 'admission', 0
    min_ram = float('inf')
    host = MemoryStatus()
    host.length = ctypes.sizeof(host)
    done = Event()

    def check_limits(host, stage_name, floor, size):
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(host)):
            raise OSError('GlobalMemoryStatusEx failed')
        elapsed = time.monotonic()-STARTED
        memory = process_memory(os.getpid())
        gib = 1024**3
        if (elapsed >= BUDGET['max_wall_s'] or memory['peak_rss_bytes'] > BUDGET['max_rss_gib']*gib
                or memory['peak_private_commit_bytes'] > BUDGET['max_private_commit_gib']*gib
                or host.avail_phys < floor*gib or size > BUDGET['max_new_output_mib']*1024**2):
            raise RuntimeError(f'resource cap at {stage_name}, update {update}')
        return elapsed, memory

    def watch():
        watchdog_host = MemoryStatus()
        watchdog_host.length = ctypes.sizeof(watchdog_host)
        while not done.wait(.05):
            try:
                check_limits(watchdog_host, stage, BUDGET['min_running_ram_gib'], directory_bytes(out))
            except FileNotFoundError:
                continue
            except (RuntimeError, OSError) as error:
                try:
                    write_json(out/'watchdog_stop.json', {'status': 'stopped', 'reason': str(error),
                               'stage': stage, 'updates': update, 'wall_s': time.monotonic()-STARTED, **SCOPE})
                finally:
                    os._exit(2)

    def guard(at=None, completed=None, *, log=False):
        nonlocal stage, update, min_ram
        if at is not None:
            stage = at
        if completed is not None:
            update = completed
        size = directory_bytes(out) if log else 0
        floor = BUDGET['min_start_ram_gib'] if stage == 'admission' else BUDGET['min_running_ram_gib']
        elapsed, memory = check_limits(host, stage, floor, size)
        min_ram = min(min_ram, host.avail_phys)
        record = {'stage': stage, 'update': update, 'wall_s': elapsed, 'available_ram_bytes': host.avail_phys,
                  'min_available_ram_bytes': min_ram, **memory}
        if log:
            record['output_bytes_before_record'] = size
            with (out/'resources.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
        return record

    watchdog = Thread(target=watch, daemon=True)
    try:
        guard(log=True)
        if shutil.disk_usage(out).free < BUDGET['max_new_output_mib']*1024**2:
            raise RuntimeError('insufficient disk space')
        guard('verify')
        watchdog.start()
        for key, sha in plan['files_sha256'].items():
            _verified_bytes(ROOT, key, sha)
            guard()
        write_json(out/'execution_plan.json', {'plan_sha256': expected_sha, 'plan': plan})
        result = work(plan, out, guard)
        if {'newton', 'warp'}.intersection(sys.modules) or update != plan['updates']:
            raise RuntimeError('unexpected physics import or optimizer step count')
        for key, sha in plan['files_sha256'].items():
            _verified_bytes(ROOT, key, sha)
            guard()
        artifacts = {file.name: digest(file) for file in sorted(out.iterdir()) if file.name != 'resources.jsonl'}
        write_json(out/'validation.json', {'status': 'completed', 'phase': phase, 'plan_sha256': expected_sha,
                   'updates': update, **SCOPE, **result, 'resources': guard('complete', log=True),
                   'artifacts_sha256': {**artifacts, 'resources.jsonl': digest(out/'resources.jsonl')}})
        if directory_bytes(out) > BUDGET['max_new_output_mib']*1024**2:
            raise RuntimeError('final output exceeds budget')
        guard()
    except BaseException as error:
        write_json(out/'stopped.json', {'status': 'stopped', 'stage': stage, 'updates': update,
                   'reason': str(error), 'wall_s': time.monotonic()-STARTED, **SCOPE})
        raise
    finally:
        done.set()
        if watchdog.ident is not None:
            watchdog.join()
