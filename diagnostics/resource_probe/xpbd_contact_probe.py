"""Replay two frozen signed branches and audit XPBD contact reporting."""
import argparse
import ast
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.spatial.transform import Rotation

from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.deterministic_response_plan import load
from ball_retention.guard import process_memory, stop_reason
from ball_retention.resources import admission, directory_bytes, snapshot
from ball_retention.task import ball_in_tool
from deterministic_archive_metrics import _full_branch
from native_repeatability import _make_env
from runner import monitored_pid, stop_owned_tree, storage_admission
from saved_native_snapshot import decode_snapshot, restore_and_readback, verify_live_layout


LABELS = ('pitch_pulse_plus', 'pitch_pulse_minus')
HORIZON = 24
SUBSTEPS = 16
STEP_CAP = len(LABELS)*HORIZON*SUBSTEPS
SCOPE = {'training_updates': 0, 'test_episode_files_read': False, 'control_accepted': False}
BUDGET = {'wall_s': 240, 'max_process_rss_gib': 2, 'max_process_commit_gib': 3,
          'min_start_ram_gib': 2, 'min_running_ram_gib': 1, 'critical_ram_gib': .75, 'low_ram_grace_s': 5,
          'min_start_gpu_free_gib': 4, 'max_gpu_used_gib': 3, 'min_disk_free_gib': 10,
          'max_new_output_mib': 32, 'max_output_cache_gib': 1}
STOCK_LINEAR_CAP = 1e-7  # N s, public contact force versus applied linear correction
VELOCITY_CAP = .00015  # m/s, allows an XPBD iteration's 1e-4 low-speed clamp
SPIN_CAP = .00015  # rad/s, same XPBD clamp
TAP_LIMITS = np.array([STOCK_LINEAR_CAP, VELOCITY_CAP, SPIN_CAP])
PACKAGE = '.venv/Lib/site-packages/newton/_src/solvers/xpbd/'
RESPONSE_RUN = Path('runs/s3_deterministic_action_response_20261115')
SURFACE_RUN = Path('runs/s3_surface_velocity_components_20261116')


def put(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def bound(plan, path):
    return _verified_bytes(ROOT, path, plan['files_sha256'][Path(path).as_posix()])


def validate_plan(path, sha):
    plan = json.loads(_verified_bytes(ROOT, path.relative_to(ROOT).as_posix(), sha))
    if (plan['scope'] != SCOPE or plan['budget'] != BUDGET or plan['labels'] != list(LABELS)
            or plan['horizon'] != HORIZON or plan['substeps'] != SUBSTEPS or plan['step_cap'] != STEP_CAP
            or plan['seed'] != 536873541 or plan['limits'] != TAP_LIMITS.tolist()):
        raise ValueError('probe differs from frozen replay scope')
    for file in plan['files_sha256']:
        bound(plan, file)
    out = (ROOT/plan['output']).resolve()
    if len(out.relative_to(ROOT/'diagnostics/resource_probe/runs').parts) != 1:
        raise ValueError('expected a dedicated resource probe output directory')
    return plan, out


def accepted_artifact(plan, run, name):
    validation = json.loads(bound(plan, run/'validation.json'))
    path = run/name
    hashes = plan['files_sha256']
    if (validation['status'] != 'completed' or validation['test_episode_files_read'] is not False
            or validation['artifacts_sha256'][name] != hashes[path.as_posix()]):
        raise ValueError(f'unexpected accepted artifact: {path}')
    return json.loads(bound(plan, path))


def frozen_case(plan):
    case, cfg, evidence = load(ROOT)
    tick, substeps = case['case']['tick'], case['control_substeps']
    if ((case['case']['episode'], tick, substeps, cfg['seed']) != ('validation_lateral', 168, SUBSTEPS, plan['seed'])
            or case['amplitude'] != .2 or case['runtime']['cuda_arch'] != 89
            or (cfg['ball']['mass_kg'], cfg['ball']['radius_m']) != (.05, .02)):
        raise ValueError('unexpected frozen scene, perturbation or sphere')
    native = decode_snapshot(evidence['snapshot'], evidence['layout'], cfg, tick, substeps)
    archive = {}
    for label in LABELS:
        row = accepted_artifact(plan, RESPONSE_RUN, f'worker/{label}.json')
        _full_branch(row, tick)
        commands = evidence['branch_commands'][label]
        if row['actions'] != commands['requested'] or row['commands'] != commands['executed']:
            raise ValueError(f'archived requests or commands differ: {label}')
        archive[label] = row
    geometry = accepted_artifact(plan, SURFACE_RUN, 'surface_components.json')['cases']['development']
    return case, cfg, evidence['layout'], native, archive, geometry


def preflight(plan):
    case, cfg, layout, native, archive, _ = frozen_case(plan)
    installed = {pkg: version(pkg) for pkg in ('newton', 'warp-lang')}
    if installed != {'newton': '1.6.0', 'warp-lang': '1.17.0'}:
        raise ValueError(f'unexpected local dependencies: {installed}')
    solver = ast.parse(bound(plan, PACKAGE+'solver_xpbd.py'))
    functions = {node.name: node for node in ast.walk(solver) if isinstance(node, ast.FunctionDef)}
    tap_args = [node.arg for node in functions['_apply_body_deltas'].args.args]
    kernels = bound(plan, PACKAGE+'kernels.py').decode('utf-8')
    if (tap_args != ['self', 'model', 'state_in', 'state_out', 'body_deltas', 'dt', 'rigid_contact_inv_weight']
            or 'contact_impulse' not in ast.unparse(functions['update_contacts'])
            or 'wp.atomic_add(contact_impulse, tid, wp.spatial_vector(lin_delta_a, ang_delta_a))' not in kernels
            or 'wp.spatial_bottom(delta) * weight' not in kernels
            or 'r_a = bx_a - wp.transform_point(X_wb_a, com_a)' not in kernels):
        raise ValueError('XPBD contact reporting or correction interface changed')
    for asset in json.loads(bound(plan, 'assets/manifest.json'))['assets']:
        _verified_bytes(ROOT, asset['local_path'], asset['sha256'])
    if not {'newton', 'warp'}.isdisjoint(sys.modules):
        raise RuntimeError('loaded physics during static preflight')
    return {'passed': True, 'case': case['case'], 'dependencies': installed, 'physics_steps': 0,
            'archive_branches': list(archive), 'native_tick': native['step_index'],
            'shape_labels': layout['shape_labels'], 'physics_dt_s': cfg['physics']['dt_s'],
            'force_sign': 'contacts.force[:3] acts on shape 0. For ball as shape 1, reverse this force.',
            'torque_reference': 'contacts.force[3:] is about body 0 COM. Record the ball-side applied '
                                'angular correction directly using the same constraint count as XPBD.',
            **SCOPE}


def contacts_for_ball(physics, row, dt):
    contacts = physics.contacts
    physics.solver.update_contacts(contacts, physics.a)
    count = row['contacts_before_step']
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    valid = ((shape0 == physics.ball_shape) & (shape1 == physics.shovel_shape)
             | (shape0 == physics.shovel_shape) & (shape1 == physics.ball_shape))
    normal_raw = contacts.rigid_contact_normal.numpy()[:count].astype(float)
    wrench_raw = contacts.force.numpy()[:count].astype(float)
    p0, p1 = (getattr(contacts, f'rigid_contact_point{i}').numpy()[:count].astype(float) for i in (0, 1))
    if not valid.all() or not all(np.isfinite(a).all() for a in (normal_raw, wrench_raw, p0, p1)):
        raise RuntimeError('unexpected or nonfinite reported contact')
    np.testing.assert_allclose(np.linalg.norm(normal_raw, axis=1), 1., atol=1e-5, rtol=0.)
    sign = np.where(shape0 == physics.ball_shape, 1., -1.)[:, None]
    normals = -sign*normal_raw
    impulses = sign*wrench_raw[:, :3]*dt
    normal_impulses = np.sum(normals*impulses, axis=1, keepdims=True)*normals
    tangent_impulses = impulses-normal_impulses
    estimate_torque = np.cross(-.02*normals, tangent_impulses).sum(axis=0)
    return {'shape0': shape0.tolist(), 'shape1': shape1.tolist(), 'normal_to_ball_world': normals.tolist(),
            'point0_body_m': p0.tolist(), 'point1_body_m': p1.tolist(), 'wrench_on_shape0': wrench_raw.tolist(),
            'ball_linear_contact_impulses_n_s': impulses.tolist(),
            'ball_normal_contact_impulse_n_s': normal_impulses.sum(axis=0).tolist(),
            'ball_tangent_contact_impulse_n_s': tangent_impulses.sum(axis=0).tolist(),
            'torque_impulse_from_contact_normals_nm_s': estimate_torque.tolist()}


def replay(physics, env, label, record, dt, native, cfg, progress, checkpoint):
    mass, radius = cfg['ball']['mass_kg'], cfg['ball']['radius_m']
    inertia = .4*mass*radius**2
    gdt = np.array([0., 0., -cfg['physics']['gravity_m_s2']*dt])
    np.testing.assert_allclose(physics.assembly['ball_inertia_kg_m2'], inertia*np.eye(3), atol=1e-12, rtol=1e-6)
    np.testing.assert_array_equal(physics.com_local[physics.ball_id], np.zeros(3))
    readback = restore_and_readback(env, native, SUBSTEPS)
    # PrescribedArmEnv.restore rebuilds the solver and contacts on every branch.
    solver = physics.solver
    if physics.contacts.force is None or not solver.rigid_contact_con_weighting:
        raise RuntimeError(f'restored solver lost the force attribute or contact weighting: {label}')
    advance_physics = physics.step
    apply_original = solver._apply_body_deltas
    previous_twist = np.array(physics.a.body_qd.numpy()[physics.ball_id], dtype=float)
    contact_delta, joint_delta = np.zeros(6), np.zeros(6)
    iterations_seen = 0
    records = []

    def tap(model, state_in, state_out, body_deltas, dt, rigid_contact_inv_weight=None):
        nonlocal iterations_seen
        delta = body_deltas.numpy()[physics.ball_id].astype(float)
        if not np.isfinite(delta).all():
            raise FloatingPointError('nonfinite ball correction')
        if rigid_contact_inv_weight is None:
            joint_delta[:] += delta
        else:
            counter = float(rigid_contact_inv_weight.numpy()[physics.ball_id])
            if not np.isfinite(counter) or counter < 0 or counter != int(counter):
                raise RuntimeError('invalid ball constraint count')
            contact_delta[:] += delta/max(counter, 1.)
            iterations_seen += 1
        return apply_original(model, state_in, state_out, body_deltas, dt, rigid_contact_inv_weight)

    def step(*args, **kwargs):
        nonlocal previous_twist, iterations_seen
        start = physics.steps
        if progress['physics_steps'] >= STEP_CAP:
            raise RuntimeError('physics substep cap reached')
        contact_delta.fill(0.)
        joint_delta.fill(0.)
        iterations_seen = 0
        try:
            row = advance_physics(*args, **kwargs)
        finally:
            progress['physics_steps'] += physics.steps-start
        k = len(records)
        archive_mark = {'physics_step': row['step'], 'candidates': row['contacts_before_step'],
                        'shape_pairs': [list(pair) for pair in row['contact_shape_pairs']]}
        if archive_mark != record['contacts'][k]:
            raise RuntimeError(f'contact candidates differ at {label} substep {k+1}: '
                               f'expected {record["contacts"][k]}, observed {archive_mark}')
        if iterations_seen != cfg['physics']['iterations'] or np.any(joint_delta):
            raise RuntimeError(f'unexpected solver corrections at {label} substep {k+1}: '
                               f'observed {iterations_seen}/{cfg["physics"]["iterations"]} '
                               f'contact iterations, ball joint delta {joint_delta.tolist()}')
        actual = np.asarray(row['body_qd_com_world'][physics.ball_id])
        wrench = contacts_for_ball(physics, row, dt)
        impulse = np.array(wrench['ball_linear_contact_impulses_n_s']).reshape(-1, 3).sum(axis=0)
        delta_v = actual[:3]-previous_twist[:3]-gdt
        delta_w = actual[3:]-previous_twist[3:]
        residuals = np.array([np.linalg.norm(impulse-contact_delta[:3]),
                              np.linalg.norm(delta_v-contact_delta[:3]/mass),
                              np.linalg.norm(delta_w-contact_delta[3:]/inertia)])
        records.append({**archive_mark, **wrench, 'ball_twist_after': actual.tolist(),
                        'observed_ball_delta_v_without_gravity_m_s': delta_v.tolist(),
                        'observed_ball_delta_omega_rad_s': delta_w.tolist(),
                        'applied_ball_linear_impulse_n_s': contact_delta[:3].tolist(),
                        'applied_ball_angular_impulse_nm_s': contact_delta[3:].tolist(),
                        'residual_norms_stock_linear_n_s_velocity_m_s_spin_rad_s': residuals.tolist()})
        previous_twist = actual
        return row

    errors = []
    raw = []
    with patch.object(physics, 'step', step), patch.object(solver, '_apply_body_deltas', tap):
        for i, action in enumerate(record['actions'][:HORIZON]):
            result = env.step(action)
            state = result.next_state_before_reset
            retained = {'terminated': result.terminated, 'truncated': result.truncated,
                        'reason': result.task['termination_reason'],
                        'carrying_candidate': result.task['carrying']['carrying_candidate'],
                        'outside_grace_active': state.departure_outside_s > 0}
            if (result.invalid_reason is not None or retained != record['retention'][i]
                    or result.requested_action != action or {'q': result.executed_joint_q, 'qd': result.executed_joint_qd} != record['commands'][i]
                    or len(records) != (i+1)*SUBSTEPS):
                raise RuntimeError(f'control, retention or clock differ at {label} H{i+1}')
            archived = record['states'][i]
            public = asdict(state)
            if any(public[k] != archived[k] for k in public if k not in ('ball', 'tool')):
                raise RuntimeError(f'public controller state differs at {label} H{i+1}')
            terms = ball_in_tool(state.ball, state.tool)
            actual = np.r_[terms['position'], terms['velocity'], terms['ball_omega']]
            errors.append(np.abs(actual-record['states_raw'][i]))
            raw.append(public)
            caps = np.repeat([1e-6, 1e-5, 1e-4], 3)
            if np.any(errors[-1] > caps):
                raise RuntimeError(f'public ball state differs at {label} H{i+1}: {errors[-1].tolist()}')
            checkpoint(f'{label}_H{i+1}')
    return {'snapshot_readback': readback, 'public_states': raw, 'substeps': records,
            'max_archive_tool_state_component_errors_si': np.max(errors, axis=0).tolist()}


def compare_vectors(actual, predicted):
    a, b = np.linalg.norm(actual), np.linalg.norm(predicted)
    return {'signal_rms_m_s': float(a/np.sqrt(len(actual))), 'estimate_rms_m_s': float(b/np.sqrt(len(actual))),
            'relative_rms': float(np.linalg.norm(actual-predicted)/a) if a else None,
            'cosine': float(np.sum(actual*predicted)/(a*b)) if a and b else None}


def analyze(traces, geometry, native):
    start = (8-1)*SUBSTEPS
    times = slice(start, HORIZON*SUBSTEPS)
    values = []
    mass, radius = .05, .02
    for label, trace in traces.items():
        rows = trace['substeps']
        vectors = {key: np.asarray([r[key] for r in rows]) for key in (
            'observed_ball_delta_v_without_gravity_m_s', 'observed_ball_delta_omega_rad_s',
            'ball_normal_contact_impulse_n_s', 'ball_tangent_contact_impulse_n_s',
            'applied_ball_angular_impulse_nm_s', 'torque_impulse_from_contact_normals_nm_s')}
        material = [r['normal_tool'] for r in geometry['geometry'][label][:HORIZON]]
        q = [native['state'].tool.quaternion_xyzw, *[r['tool']['quaternion_xyzw'] for r in trace['public_states'][:-1]]]
        normals = np.repeat(Rotation.from_quat(q).apply(material), SUBSTEPS, axis=0)

        def tangent(a):
            return a-normals*np.sum(normals*a, axis=-1, keepdims=True)

        values.append({
            'control_plane_tangent': tangent(vectors['observed_ball_delta_v_without_gravity_m_s']),
            'contact_normal_contribution': tangent(vectors['ball_normal_contact_impulse_n_s']/mass),
            'contact_tangent_contribution': tangent(vectors['ball_tangent_contact_impulse_n_s']/mass),
            'spin_prediction_single_normal': np.cross(normals, .4*radius*vectors['observed_ball_delta_omega_rad_s']),
            'observed_spin_equivalent': .4*radius*vectors['observed_ball_delta_omega_rad_s'],
            'multi_normal_spin_equivalent': vectors['torque_impulse_from_contact_normals_nm_s']/(mass*radius),
            'applied_spin_equivalent': vectors['applied_ball_angular_impulse_nm_s']/(mass*radius)})
    a, b = values
    half = {key: (a[key]-b[key])[times]*.5 for key in a}
    summaries = {}
    for level in ('physics_substep', 'control_period'):
        terms = half if level == 'physics_substep' else {k: v.reshape(-1, SUBSTEPS, 3).sum(axis=1) for k, v in half.items()}
        base = terms['control_plane_tangent']
        energy = np.sum(base*base)
        summaries[level] = {
            'single_normal_spin_to_tangent': compare_vectors(base, terms['spin_prediction_single_normal']),
            'contact_projection_closure': compare_vectors(base, terms['contact_normal_contribution']+terms['contact_tangent_contribution']),
            'normal_impulse_projection_fraction': float(np.sum(base*terms['contact_normal_contribution'])/energy),
            'tangent_impulse_projection_fraction': float(np.sum(base*terms['contact_tangent_contribution'])/energy),
            'multi_normal_torque_model_vs_applied': compare_vectors(terms['applied_spin_equivalent'], terms['multi_normal_spin_equivalent']),
            'applied_torque_vs_observed_spin': compare_vectors(terms['observed_spin_equivalent'], terms['applied_spin_equivalent']),
            'half_differences_m_s': {key: v.tolist() for key, v in terms.items()}}
    return summaries


def worker(plan, out):
    progress = {'worker_pid': os.getpid(), 'physics_steps': 0, 'completed_branches': [], **SCOPE}
    status = out/'worker_result.json'

    def checkpoint(stage):
        progress['stage'] = stage
        put(status, progress)

    checkpoint('worker_admission')
    try:
        if not admission(snapshot(ROOT), BUDGET)['passed']:
            raise RuntimeError('worker resource recheck failed')
        case, cfg, layout, native, archive, geometry = frozen_case(plan)
        env = _make_env(ROOT, cfg, progress, checkpoint, expected_device_arch=case['runtime']['cuda_arch'])
        env.reset(episode_id='s3-xpbd-contact-probe')
        physics = env.physics
        if env.substeps != SUBSTEPS or physics.steps != 0 or not physics.solver.rigid_contact_con_weighting:
            raise RuntimeError('initialized scene differs from fixed clock or XPBD contact weighting')
        verify_live_layout(physics, layout)
        physics.model.request_contact_attributes('force')
        physics.contacts = physics.pipeline.contacts()
        if physics.contacts.force is None:
            raise RuntimeError('XPBD force attribute did not allocate')
        traces, maxima = {}, {}
        for label in LABELS:
            checkpoint(f'{label}_restore')
            trace = replay(physics, env, label, archive[label], env.dt, native, cfg, progress, checkpoint)
            put(out/f'{label}.json', trace)
            residual = np.asarray([row['residual_norms_stock_linear_n_s_velocity_m_s_spin_rad_s'] for row in trace['substeps']])
            maxima[label] = {'peak_stock_linear_n_s_velocity_m_s_spin_rad_s': residual.max(axis=0).tolist(),
                             'rms_stock_linear_n_s_velocity_m_s_spin_rad_s': np.sqrt(np.mean(residual**2, axis=0)).tolist()}
            if np.any(residual > TAP_LIMITS):
                raise RuntimeError(f'momentum or force tap gate failed: {label}: {maxima[label]}')
            progress['completed_branches'].append(label)
            traces[label] = trace
            checkpoint(f'{label}_verified')
        if progress['physics_steps'] != STEP_CAP:
            raise RuntimeError('replay did not finish at the frozen cap')
        put(out/'momentum.json', {'limits': TAP_LIMITS.tolist(), 'by_branch': maxima,
                                 'window': [8, HORIZON], 'pair': analyze(traces, geometry, native)})
        progress['status'] = 'completed'
        checkpoint('complete')
    except BaseException as e:
        progress.update(status='stopped', error=str(e))
        put(status, progress)
        raise


def supervise(plan, out, plan_file, sha):
    began = time.monotonic()
    out.mkdir()
    proc = None

    def output_hashes():
        return {p.name: fingerprint(p) for p in sorted(out.iterdir()) if p.is_file()}

    try:
        put(out/'execution_plan.json', {'plan': plan, 'plan_sha256': sha})
        put(out/'preflight.json', preflight(plan))
        sample = snapshot(ROOT)
        hardware, storage = admission(sample, BUDGET), storage_admission(ROOT, out, BUDGET)
        put(out/'admission.json', {'sample': sample, 'hardware': hardware, 'storage': storage})
        if not hardware['passed'] or not storage['passed']:
            raise RuntimeError('resource admission failed')
        env = os.environ.copy()
        for key, part in (('WARP_CACHE_PATH', '.cache/warp'), ('NEWTON_CACHE_PATH', '.cache/newton'),
                          ('TEMP', '.tmp'), ('TMP', '.tmp'), ('TMPDIR', '.tmp')):
            folder = ROOT/part
            folder.mkdir(parents=True, exist_ok=True)
            env[key] = str(folder)
        low_since = None
        with (out/'worker.log').open('x', encoding='utf-8') as log, (out/'resources.jsonl').open('x', encoding='utf-8') as timeline:
            proc = subprocess.Popen([sys.executable, '-B', __file__, '--plan', str(plan_file), '--sha256', sha, '--worker'],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            while proc.poll() is None:
                sample, now = snapshot(ROOT), time.monotonic()
                low = sample['available_ram_bytes'] < BUDGET['min_running_ram_gib']*1024**3
                low_since = (now if low_since is None else low_since) if low else None
                try:
                    memory = process_memory(monitored_pid(out/'worker_result.json', proc.pid))
                except OSError:
                    if proc.poll() is not None:
                        break
                    raise
                size = directory_bytes(out)
                combined = size+directory_bytes(ROOT/'.cache')+directory_bytes(ROOT/'.tmp')
                reason = stop_reason(sample, BUDGET, now-began, memory['peak_rss_bytes'], combined,
                                     now-low_since if low_since is not None else 0., memory['peak_private_commit_bytes'])
                if reason is None and size > BUDGET['max_new_output_mib']*1024**2:
                    reason = 'new_output_budget'
                timeline.write(json.dumps({'wall_s': now-began, **sample, **memory, 'output_bytes': size,
                                           'output_and_cache_bytes': combined, 'stop_reason': reason})+'\n')
                timeline.flush()
                if reason:
                    stop_owned_tree(proc)
                    raise RuntimeError(reason)
                time.sleep(1)
        result = json.loads((out/'worker_result.json').read_bytes())
        if (proc.returncode != 0 or result['status'] != 'completed' or result['physics_steps'] != STEP_CAP
                or result['completed_branches'] != list(LABELS)):
            raise RuntimeError(result.get('error', 'worker scope incomplete'))
        validate_plan(plan_file, sha)
        put(out/'validation.json', {'status': 'completed', 'plan_sha256': sha, 'physics_steps': result['physics_steps'],
                                    'branches': result['completed_branches'], 'wall_s': time.monotonic()-began,
                                    'artifacts_sha256': output_hashes(), **SCOPE})
    except BaseException as e:
        status = out/'worker_result.json'
        last = json.loads(status.read_bytes()) if status.exists() else {}
        put(out/'stopped.json', {'status': 'stopped', 'reason': str(e), 'worker_started': proc is not None,
                                'physics_steps': 0 if proc is None else last.get('physics_steps'),
                                'last_reported_stage': last.get('stage'), 'artifacts_sha256': output_hashes(), **SCOPE})
        raise
    finally:
        if proc is not None and proc.poll() is None:
            stop_owned_tree(proc)
            proc.wait(timeout=15)


if __name__ == '__main__':
    opts = argparse.ArgumentParser(description=__doc__)
    opts.add_argument('--plan', required=True, type=Path)
    opts.add_argument('--sha256', required=True)
    opts.add_argument('--worker', action='store_true')
    args = opts.parse_args()
    plan_file = args.plan.resolve()
    validated, out_dir = validate_plan(plan_file, args.sha256)
    if args.worker:
        worker(validated, out_dir)
    else:
        supervise(validated, out_dir, plan_file, args.sha256)
