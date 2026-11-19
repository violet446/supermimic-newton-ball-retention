"""Decompose action JVPs along a frozen nominal world-model rollout."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from diagnose_s3_timing_coverage import legacy_case, read_changed_pair
from train_s3_timing_coverage import ROOT, action_sequences, digest, read_held, write_json
from ball_retention.action_grad import _inputs, _known_jacobian, action_sensitivity
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.deterministic_response_model import _directional_rollout
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES, STATE_NAMES, _ball_state, known_rollout
from ball_retention.resources import directory_bytes
from ball_retention.world_model import STATE_DIM, ResidualMLP


COVERAGE = Path('runs/s3_timing_coverage_fit_20261115_resume')
LEAD_FIT = Path('runs/s3_timing_lead_weight_fit_20261115')
OUT = ROOT/'runs/s3_action_input_gradients_20261115'
INPUTS = {
    (COVERAGE/'validation.json').as_posix(): '73ba0d9b301099beaf5c3f630fe1b425a6b1826db7277772b5c0a4ebfe6fbfe7',
    (LEAD_FIT/'validation.json').as_posix(): '073a0f522f96052a220600e28361bb6dc9f9e73a24820d19308764aec567f9bc',
}
GROUPS = {
    'request': KNOWN_NAMES[2:4],
    'filtered_angles_and_rates': KNOWN_NAMES[4:8],
    'tool_angular_velocity_and_acceleration': KNOWN_NAMES[11:17],
    'time_phase_gravity_and_anchor_acceleration': (*KNOWN_NAMES[:2], *KNOWN_NAMES[8:11], *KNOWN_NAMES[17:]),
}
REPORT_LEADS = (1, 8, 16, 24, 32)
BANDS = ('1–8', '9–16', '17–24', '25–32')
LIMITS = {'cpu_threads': 1, 'wall_s': 120, 'rss_gib': 1, 'private_commit_gib': 1.5, 'output_mib': 16}


def feature_tangents(known, actions, directions, cfg, dt):
    """Add causal suffixes of the existing held-offset tool Jacobian."""
    jumps = np.diff(directions, axis=1, prepend=0.)
    features = np.zeros((*known.shape, len(directions)))
    for t in np.flatnonzero(np.any(jumps != 0., axis=(0, 2))):
        features[t:] += _known_jacobian(known[t:], actions[t:], cfg, dt) @ jumps[:, t].T
    return features


def decompose(model, mean, scale, fields, cfg, actions, directions, reference=None):
    """Separate first-order injections by input columns and injection lead."""
    state = _parse_state(fields)
    start = (state.time_s, state.tilt_rad, state.tilt_rate_rad_s)
    raw = _ball_state(state)
    initial, known, normalized = _inputs(model, raw, *start, actions, cfg, mean, scale)
    direction = np.stack(list(directions.values()))
    feature = feature_tangents(known, actions, direction, cfg, model.dt_s)
    if len(known) != 32:
        raise ValueError('expected the frozen H32 rollout')
    mask = np.array([[name in columns for name in KNOWN_NAMES] for columns in GROUPS.values()])
    np.testing.assert_array_equal(mask.sum(axis=0), 1)
    epsilon = 1e-5
    plus, minus = (known_rollout(*start, actions+sign*epsilon*direction, cfg, model.dt_s) for sign in (1., -1.))
    np.testing.assert_allclose(feature, ((plus-minus)/(2*epsilon)).transpose(1, 2, 0), rtol=2e-6, atol=2e-9)

    prediction = model.rollout(initial[None], normalized[None])[0]
    scale_state = model.state_scale[:, None, None, None]
    band = np.zeros((STATE_DIM, len(GROUPS), len(BANDS), len(direction)))
    emitted, carried, cumulative = [], [], []
    eye = np.eye(STATE_DIM)
    previous = initial
    for t, row in enumerate(normalized):
        state_jac, tool_jac, _ = model.step_vjp(np.broadcast_to(previous, eye.shape),
                                               np.broadcast_to(row, (STATE_DIM, model.known_dim)), eye)
        band = np.einsum('ij,jgbd->igbd', state_jac, band)
        direct = np.einsum('if,fd,gf->igd', tool_jac, feature[t]/scale[:, None], mask)
        carried.append((band*scale_state).sum(axis=2).transpose(2, 1, 0))
        band[:, :, t//8, :] += direct
        emitted.append((direct*model.state_scale[:, None, None]).transpose(2, 1, 0))
        cumulative.append((band*scale_state).transpose(3, 1, 2, 0))
        previous = prediction[t]

    bands = np.stack(cumulative, axis=1)       # direction, lead, group, band, state
    direct, inherited = (np.stack(rows, axis=1) for rows in (emitted, carried))
    grouped = bands.sum(axis=3)
    np.testing.assert_allclose(grouped, direct+inherited, rtol=1e-12, atol=5e-15)
    full = grouped.sum(axis=2)
    existing_pred, existing_jvp = _directional_rollout(model, initial, known, normalized, actions, direction, cfg, scale)
    np.testing.assert_array_equal(existing_pred, model.denormalize(prediction))
    np.testing.assert_allclose(full, existing_jvp, rtol=1e-12, atol=5e-15)
    if reference is not None:
        np.testing.assert_array_equal(existing_pred, reference['nominal'])
        stored = np.array([reference['pairs'][name]['directional_jvp'] for name in directions])
        np.testing.assert_allclose(full, stored, rtol=1e-12, atol=5e-15)
    return full, grouped, direct, inherited, bands, float(np.max(np.abs(full-existing_jvp)))


def compare_position(full, grouped, direct, carried, bands, trace, base, amplitude):
    effect, group, injected, inherited, by_band = (amplitude*value[..., :3] for value in (full, grouped, direct, carried, bands))
    physical = (trace[0][:, :3]-trace[1][:, :3])/2.
    noise = np.linalg.norm((base[0][:, :3]-base[1][:, :3])/2., axis=1)
    floor = np.maximum(1e-6, 5*noise)
    rows = {}
    for h in REPORT_LEADS:
        i = h-1
        ref = physical[i]
        norm_ref = float(np.linalg.norm(ref))
        strong = bool(norm_ref >= 10*floor[i])
        unit = ref/norm_ref if strong else None
        value = effect[i]
        remainder = value-group[i, 0]
        local_norm = np.linalg.norm(value)
        rows[str(h)] = {'newton_half_m': ref.tolist(), 'newton_norm_m': norm_ref, 'newton_strong': strong,
                        'linearized_half_m': value.tolist(), 'linearized_norm_m': float(local_norm),
                        'jvp_cosine': float(unit@value/local_norm) if strong and local_norm > floor[i] else None,
                        'group_position_m': {name: vec.tolist() for name, vec in zip(GROUPS, group[i])},
                        'group_current_injection_m': {name: vec.tolist() for name, vec in zip(GROUPS, injected[i])},
                        'group_carried_m': {name: vec.tolist() for name, vec in zip(GROUPS, inherited[i])},
                        'group_projection_on_newton_m': ({name: float(unit@vec) for name, vec in zip(GROUPS, group[i])}
                                                          if strong else None),
                        'projection_by_injection_band_m': ({name: dict(zip(BANDS, (vectors @ unit).tolist()))
                                                            for name, vectors in zip(GROUPS, by_band[i])} if strong else None),
                        'jvp_without_request_tangent_position_m': remainder.tolist(),
                        'jvp_without_request_tangent_projection_on_newton_m': float(unit@remainder) if strong else None}
    return rows


def run(out):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    inputs = dict(INPUTS)

    def load(name, expected, *, lines=False):
        path = Path(name).as_posix()
        raw = _verified_bytes(ROOT, path, expected)
        inputs[path] = expected
        return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)

    def guard():
        mem = process_memory(os.getpid())
        if (time.monotonic()-began >= LIMITS['wall_s'] or mem['peak_rss_bytes'] > LIMITS['rss_gib']*1024**3
                or mem['peak_private_commit_bytes'] > LIMITS['private_commit_gib']*1024**3
                or directory_bytes(out) > LIMITS['output_mib']*1024**2):
            raise RuntimeError('action input gradient diagnosis resource cap reached')
        return mem

    try:
        old, new = (load(path/'validation.json', inputs[(path/'validation.json').as_posix()]) for path in (COVERAGE, LEAD_FIT))
        if (any(v['status'] != 'completed' or v['test_episode_files_read'] or v['simulation_steps'] for v in (old, new))
                or list(old['updates'].values()) != [900, 900] or new['updates'] != 900):
            raise ValueError('unexpected frozen endpoint scope')
        plan = load(COVERAGE/'execution_plan.json', old['artifacts_sha256']['execution_plan.json'])['design']
        summary = load(COVERAGE/'comparison.json', old['artifacts_sha256']['comparison.json'])
        probe_path = Path(plan['held_timing_run'])
        held_record = load(probe_path, plan['inputs_sha256'][probe_path.as_posix()])
        capture, cfg, state, timing, _ = read_held(plan)
        for name in timing:
            item = f'worker/{name}.json'
            inputs[(probe_path.parent/item).as_posix()] = held_record['artifacts_sha256'][item]
        held_roll, _ = read_changed_pair(summary)
        _, legacy_cfg, legacy_state, legacy, legacy_dirs = legacy_case(plan, capture, held_roll, load)
        nominal = timing['nominal_1']['actions']
        timing_dirs, _ = action_sequences(capture, nominal)
        models = {'held_offsets': (COVERAGE, old, 'held_offsets_900.npz', 'held_offsets_timing_rollouts.json'),
                  'held_offsets_plus_timing': (COVERAGE, old, 'held_offsets_plus_timing_900.npz', 'held_offsets_plus_timing_timing_rollouts.json'),
                  'lead_weight_intervention': (LEAD_FIT, new, 'lead_weight_intervention_900.npz', 'timed_rollouts.json')}
        output = {}
        for name, (folder, result, checkpoint, trajectory) in models.items():
            ref = load(folder/trajectory, result['artifacts_sha256'][trajectory])
            checkpoint_path = (folder/checkpoint).as_posix()
            ck_sha = result['artifacts_sha256'][checkpoint]
            if ref['checkpoint_sha256'] != ck_sha:
                raise ValueError('saved rollout checkpoint hash disagrees')
            inputs[checkpoint_path] = ck_sha
            with np.load(BytesIO(_verified_bytes(ROOT, checkpoint_path, ck_sha)), allow_pickle=False) as saved:
                if tuple(saved['known_names']) != KNOWN_NAMES:
                    raise ValueError('frozen model feature order changed')
                model = ResidualMLP.from_state_dict(saved)
                mean, scale = saved['known_mean'], saved['known_scale']
            values = decompose(model, mean, scale, state, cfg, nominal, timing_dirs, ref)
            legacy_actions = legacy['nominal_1']['actions']
            old_values = decompose(model, mean, scale, legacy_state, legacy_cfg, legacy_actions, legacy_dirs)
            old_state = _parse_state(legacy_state)
            independently = action_sensitivity(model, _ball_state(old_state), old_state.time_s, old_state.tilt_rad,
                                               old_state.tilt_rate_rad_s, legacy_actions, legacy_cfg, mean, scale)
            np.testing.assert_allclose(old_values[0][0], independently['jacobian'][..., held_roll[2]], rtol=1e-12, atol=5e-15)
            packed = {}
            for data, dirs, arrays, epsilon in ((timing, timing_dirs, values, capture['amplitude']),
                                                 (legacy, legacy_dirs, old_values, held_roll[-1])):
                base = [data[f'nominal_{i}']['raw'] for i in (1, 2)]
                for i, direction in enumerate(dirs):
                    branches = [data[f'{direction}_{side}']['raw'] for side in ('plus', 'minus')]
                    packed[direction] = {'position': compare_position(*(a[i] for a in arrays[:5]), branches, base, epsilon),
                                         'jvp_si': arrays[0][i].tolist(), 'groups_si': arrays[1][i].tolist(),
                                         'current_injection_si': arrays[2][i].tolist(), 'carried_state_si': arrays[3][i].tolist(),
                                         'by_injection_band_si': arrays[4][i].tolist()}
            output[name] = {'directions': packed, 'max_closure_error_si': max(values[-1], old_values[-1])}
            guard()

        write_json(out/'decomposition.json', {'held_timing': capture['case'], 'held_roll': held_roll, 'models': output,
                   'input_groups': GROUPS, 'state_component_order': STATE_NAMES, 'injection_lead_bands': BANDS,
                   'convention': 'JVP per normalized action coefficient. Position summary multiplies by the frozen 0.20 amplitude. '
                                 'Projection is along the saved Newton finite response at strong leads. '
                                 'Input-column attribution follows a shared nominal model rollout; it is a model diagnostic.'})
        for name, expected in inputs.items():
            _verified_bytes(ROOT, name, expected)
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline diagnosis loaded physics')
        mem = guard()
        write_json(out/'validation.json', {'status': 'completed', 'limits': LIMITS, 'wall_s': time.monotonic()-began, **mem,
                   'models': len(output), 'directions_per_model': 4, 'max_closure_error_si': max(v['max_closure_error_si'] for v in output.values()),
                   'training_updates': 0, 'simulation_steps': 0, 'test_episode_files_read': False,
                   'inputs_sha256': inputs, 'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted({
                        Path(__file__).resolve(), ROOT/'scripts/diagnose_s3_timing_coverage.py', ROOT/'scripts/train_s3_timing_coverage.py',
                        *(ROOT/'src/ball_retention').glob('*.py')})},
                   'artifacts_sha256': {'decomposition.json': digest(out/'decomposition.json')}})
    except BaseException as error:
        write_json(out/'stopped.json', {'reason': str(error), 'training_updates': 0, 'simulation_steps': 0,
                                       'wall_s': time.monotonic()-began})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    path = parser.parse_args().output.resolve()
    if not path.relative_to(ROOT/'runs').parts:
        raise ValueError('choose a fresh run subdirectory')
    run(path)
