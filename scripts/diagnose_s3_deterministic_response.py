"""Locate error in accepted S3 response trajectories using only frozen evidence."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

from check_s3_actions import ROOT, digest, write_json
from ball_retention.action_grad import _inputs
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.archive_state_import import _parse_state
from ball_retention.deterministic_response_plan import DESIGN as CAPTURE, load as load_capture
from ball_retention.guard import process_memory
from ball_retention.model_data import KNOWN_NAMES, _ball_state
from ball_retention.resources import directory_bytes
from ball_retention.response_error_diagnosis import diagnose_responses
from ball_retention.world_model import ResidualMLP

sys.path.insert(0, str(ROOT/'diagnostics/resource_probe'))
from deterministic_archive_metrics import _full_branch
from response_contact_diagnosis import diagnose_contacts


DESIGN = 'configs/s3_deterministic_response_diagnosis_plan.json'
DESIGN_SHA256 = '8201c2e739ecc82bbfac9167963377954c545b69d160649fb37572f1c8394f1e'
OUT = ROOT/'runs/s3_deterministic_response_diagnosis_20261115'


def load_run(root, diagnosis):
    """Tie trajectories and model inputs to the accepted response validation."""
    source = Path(diagnosis['source_run'])
    manifest = diagnosis['inputs_sha256']
    validation, execution = (json.loads(_verified_bytes(root, source/name, manifest[(source/name).as_posix()]))
                             for name in ('validation.json', 'execution_plan.json'))
    plan, cfg, evidence = load_capture(root)
    labels = plan['branch_order']
    if (execution['design'] != plan or validation['status'] != 'completed'
            or not validation['execution_gates_passed'] or validation['case'] != diagnosis['case']
            or validation['case'] != plan['case'] or validation['branches'] != labels
            or validation['physics_steps'] != plan['budget']['max_physics_substeps']
            or validation['nominal_prefix_substeps'] != 0 or validation['test_episode_files_read'] is not False):
        raise ValueError('source response run does not match the frozen offline diagnosis')

    files = validation['artifacts_sha256']

    def artifact(name):
        path = (source/name).as_posix()
        manifest[path] = files[name]
        return json.loads(_verified_bytes(root, path, files[name]))

    manifest = dict(manifest)
    traces = {label: artifact(f'worker/{label}.json') for label in labels}
    model = artifact('static/model_rollouts.json')
    layout = artifact('worker/native_layout.json')
    comparison = artifact('worker/response_comparison.json')
    if (artifact('worker/effective_config.json') != cfg or model != evidence['model_rollouts']
            or layout != evidence['layout']):
        raise ValueError('model, physics config or layout disagrees with the capture preflight')
    for label, trace in traces.items():
        branch = evidence['branch_commands'][label]
        if trace['actions'] != branch['requested'] or trace['commands'] != branch['executed']:
            raise ValueError(f'executed commands disagree with the frozen branch {label}')

    checkpoint = plan['model']['checkpoint']
    manifest[checkpoint] = plan['inputs_sha256'][checkpoint]
    with np.load(BytesIO(_verified_bytes(root, checkpoint, manifest[checkpoint])), allow_pickle=False) as state:
        fit = ResidualMLP.from_state_dict(state)
        mean, scale = state['known_mean'], state['known_scale']
        if tuple(state['known_names']) != tuple(KNOWN_NAMES):
            raise ValueError('model checkpoint has a different causal feature order')
    start = _parse_state(evidence['snapshot']['state'])
    return plan, cfg, fit, mean, scale, start, traces, model, layout, comparison, manifest


def evaluate(fit, mean, scale, cfg, start, traces, frozen, plan, guard):
    """Reproduce free forecasts and perform one independent conditional step per lead."""
    initial = _ball_state(start)
    np.testing.assert_array_equal(initial, frozen['initial_state_raw_si'])
    nominal = np.asarray(frozen['nominal'], dtype=float)
    saved_outputs = {'nominal_1': nominal, 'nominal_2': nominal}
    for name, pair in frozen['pairs'].items():
        saved_outputs.update({f'{name}_{side}': np.asarray(pair[side], dtype=float) for side in ('plus', 'minus')})

    measured, free, local = {}, {}, {}
    for label, trace in traces.items():
        series = _full_branch(trace, plan['case']['tick'])
        truth = series['raw']
        pre_truth = np.vstack((initial, truth[:-1]))
        normalized, _, features = _inputs(fit, initial, start.time_s, start.tilt_rad, start.tilt_rate_rad_s,
                                          series['actions'], cfg, mean, scale)
        rollout = fit.denormalize(fit.rollout(normalized[None], features[None]))[0]
        np.testing.assert_array_equal(rollout, saved_outputs[label])
        one_step = fit.denormalize(fit.step(fit.normalize(pre_truth), features))
        np.testing.assert_allclose(one_step[0], rollout[0], rtol=0., atol=2e-14)
        if not (np.isfinite(rollout).all() and np.isfinite(one_step).all()):
            raise ValueError(f'nonfinite diagnostic forecast in {label}')
        measured[label], free[label], local[label] = truth, rollout, one_step
        guard()
    return initial, measured, free, local


def check_recorded_responses(responses, recorded):
    for direction, group in recorded['directions'].items():
        for lead, families in group['by_lead'].items():
            fresh = responses['directions'][direction]['by_lead'][lead]
            for family, expected in families.items():
                row = fresh[family]
                for key, reference in (('newton', 'newton_half_difference'), ('free', 'model_fd_half_difference'),
                                       ('jvp_effect', 'model_jvp_linearized_displacement')):
                    np.testing.assert_array_equal(row['vectors'][key], expected[reference])
                for key, reference in (('free', 'model_fd_vs_newton'), ('jvp', 'model_jvp_vs_newton')):
                    if row['comparisons'][key]['agrees'] != expected[reference]['agrees']:
                        raise ValueError(f'recomputed agreement changed for {direction} H{lead} {family}')


def run(out):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    try:
        diagnosis = json.loads(_verified_bytes(ROOT, DESIGN, DESIGN_SHA256))
        limits = diagnosis['budget']

        def guard():
            memory = process_memory(os.getpid())
            if (time.monotonic()-began >= limits['max_wall_s']
                    or memory['peak_rss_bytes'] > limits['max_process_rss_gib']*1024**3
                    or memory['peak_private_commit_bytes'] > limits['max_process_commit_gib']*1024**3
                    or directory_bytes(out) > limits['max_new_output_mib']*1024**2):
                raise RuntimeError('offline response diagnosis resource cap reached')
            return memory

        guard()
        plan, cfg, fit, mean, scale, start, traces, model, layout, recorded, inputs = load_run(ROOT, diagnosis)
        initial, measured, free, local = evaluate(fit, mean, scale, cfg, start, traces, model, plan, guard)
        analysis = {**plan, **{key: diagnosis[key] for key in ('onset_consecutive_leads', 'report_leads')}}
        responses = diagnose_responses(analysis, measured, free, local, model, initial)
        check_recorded_responses(responses, recorded)
        contacts = diagnose_contacts(plan, traces, layout)
        write_json(out/'response_diagnosis.json', responses)
        write_json(out/'contact_timeline.json', contacts)
        for name, expected in inputs.items():
            _verified_bytes(ROOT, name, expected)
        if {'newton', 'warp'}.intersection(sys.modules):
            raise RuntimeError('offline diagnosis loaded a physics library')
        source_paths = {Path(__file__).resolve(), ROOT/DESIGN, ROOT/CAPTURE,
                        *(ROOT/'src/ball_retention').glob('*.py'),
                        *(ROOT/'diagnostics/resource_probe').glob('*.py'), ROOT/'scripts/check_s3_actions.py'}
        sources = {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(source_paths)}
        memory = guard()
        write_json(out/'validation.json', {'status': 'completed', 'case': diagnosis['case'],
                   'design_sha256': DESIGN_SHA256, 'source_run': diagnosis['source_run'], 'branches': len(traces),
                   'leads_per_branch': plan['horizon'], 'frozen_free_rollouts_exact': True,
                   'frozen_lead_responses_match': True,
                   'one_step_uses_measured_pre_state_only': True, 'physical_substeps': 0,
                   'training_updates': 0, 'test_episode_files_read': False, 'control_accepted': False,
                   'wall_s': time.monotonic()-began, **memory, 'inputs_sha256': inputs, 'source_sha256': sources,
                   'artifacts_sha256': {p.name: digest(p) for p in sorted(out.iterdir())}})
    except BaseException as error:
        write_json(out/'stopped.json', {'reason': str(error), 'wall_s': time.monotonic()-began,
                   'physical_substeps': 0, 'training_updates': 0, 'test_episode_files_read': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    output = parser.parse_args().output.resolve()
    if not output.relative_to(ROOT/'runs').parts:
        raise ValueError('choose a fresh run subdirectory')
    run(output)
