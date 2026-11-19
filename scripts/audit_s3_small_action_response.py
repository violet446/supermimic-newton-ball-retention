"""Compare measured Newton secants across action amplitudes and axes."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from collect_s3_small_action_response import load_run, score_case
from train_s3_local_velocity_increment import bound


PHASE = 'small_action_response_audit'
QUANTITIES = ('position', 'velocity', 'spin')


def read(plan, name):
    return json.loads(bound(plan, name))


def aligned(estimate, reference):
    a, b = np.linalg.norm(estimate), np.linalg.norm(reference)
    if min(a, b) == 0.:
        return None
    return float(np.clip(estimate @ reference / (a*b), -1., 1.))


def newton_data(row):
    return {k: row[k] for k in ('newton_half_difference', 'newton_half_difference_norm',
                                'newton_fd_derivative', 'signal_floor', 'newton_resolved', 'newton_strong')}


def pitch_amplitudes(shape, scores, archived, leads):
    rows = {}
    for lead in leads:
        key = str(lead)
        quantities = {}
        for group in QUANTITIES:
            full = newton_data(archived[shape]['by_lead'][key][group])
            reference = np.asarray(full['newton_fd_derivative'])
            stages = {}
            for fraction in ('quarter', 'half'):
                case = f'{shape}_{fraction}'
                sample = scores['61_pairs'][case]['by_lead'][key][group]
                measured = newton_data(sample)
                slope = np.asarray(measured['newton_fd_derivative'])
                both_resolved = measured['newton_resolved'] and full['newton_resolved']
                stages[fraction] = {**measured,
                    'slope_cosine_vs_full': aligned(slope, reference) if both_resolved else None,
                    'slope_relative_error_vs_full': float(np.linalg.norm(slope-reference)/np.linalg.norm(reference))
                        if both_resolved else None,
                    'slope_norm_ratio_vs_full': float(np.linalg.norm(slope)/np.linalg.norm(reference)) if both_resolved else None,
                    'same_amplitude_scores': {
                        label: {name: trace[case]['by_lead'][key][group][name] for name in ('model_fd_vs_newton','model_jvp_vs_newton')}
                        for label, trace in scores.items()},
                }
            quantities[group] = {'full': full, **stages}
        rows[key] = quantities
    return rows


def dual_decomposition(scores, models_full, archived, leads):
    rows = {}
    for lead in leads:
        key = str(lead)
        quantities = {}
        for i, group in enumerate(QUANTITIES):
            columns = slice(3*i, 3*(i+1))
            native = scores['61_pairs']
            original = archived['dual_axis_pulse']['by_lead'][key][group]
            measured_fields = {'dual': original, **{ax: native[ax+'_component']['by_lead'][key][group]
                                                      for ax in ('roll', 'pitch')}}
            measured = {k: np.asarray(v['newton_half_difference']) for k, v in measured_fields.items()}
            measured['interaction'] = measured['dual']-measured['roll']-measured['pitch']
            floor = sum(v['signal_floor'] for v in measured_fields.values())
            model_results = {}
            for name, result in scores.items():
                old_full = models_full[name]['pairs']['dual_axis_pulse']
                dual = .5*(np.asarray(old_full['plus'][lead-1])-np.asarray(old_full['minus'][lead-1]))
                prediction = {'dual': dual[columns],
                              **{ax: np.asarray(result[ax+'_component']['by_lead'][key][group]['model_fd_half_difference'])
                                 for ax in ('roll', 'pitch')}}
                prediction['interaction'] = prediction['dual']-prediction['roll']-prediction['pitch']
                errors = {k: v-measured[k] for k,v in prediction.items()}
                total = errors['dual']
                parts = ('roll','pitch','interaction')
                np.testing.assert_allclose(sum(errors[k] for k in parts), total, rtol=0., atol=2e-12)
                length = np.linalg.norm(total)
                model_results[name] = {'predicted_vectors': {k: v.tolist() for k,v in prediction.items()},
                    'error_vectors': {k: v.tolist() for k,v in errors.items()},
                    'error_norms': {k: float(np.linalg.norm(v)) for k,v in errors.items()},
                    'signed_error_projection_on_total': {k: float(errors[k] @ total / length) if length else 0. for k in parts}}
            residual = float(np.linalg.norm(measured['interaction']))
            quantities[group] = {'newton_vectors': {k:v.tolist() for k,v in measured.items()},
                'newton_norms': {k:float(np.linalg.norm(v)) for k,v in measured.items()},
                'newton_interaction_floor': floor, 'newton_interaction_resolved': residual >= floor,
                'models': model_results}
        rows[key] = quantities
    return rows


def candidate_log_differences(spec, status, plan):
    source = Path(plan['capture_validation']).parent
    n = spec['substeps_per_control']
    clock = spec['case']['tick']*n
    logs = {}
    for label in status['branches']:
        name = f'worker/{label}.json'
        path = source/name
        if plan['files_sha256'][path.as_posix()] != status['artifacts_sha256'][name]:
            raise ValueError('unaccepted Newton branch contact log')
        logs[label] = read(plan, path)['contacts']
        if ([row['physics_step'] for row in logs[label]]
                != list(range(clock+1, clock+1+n*spec['horizon']))):
            raise ValueError('contact log does not match the accepted control clock')

    report = {}
    for case in spec['cases']:
        plus, minus = (logs[f'{case}_{sign}'] for sign in ('plus', 'minus'))
        fields = {}
        for key in ('candidates', 'shape_pairs'):
            different = [i for i, (a, b) in enumerate(zip(plus, minus)) if a[key] != b[key]]
            fields[key] = {'different_substeps': len(different),
                           'first_different_lead': different[0]//n+1 if different else None,
                           'first_different_substep': different[0]+1 if different else None}
        report[case] = fields
    return report


def run(plan, out, guard):
    if plan['updates']:
        raise ValueError('multiamplitude audit must remain offline')
    capture = Path(plan['capture_plan'])
    launch, _, spec, _, _, _, _, load_input = load_run(ROOT/capture, plan['files_sha256'][capture.as_posix()])
    status = read(plan, plan['capture_validation'])
    comparison_file = Path(plan['capture_validation']).with_name('same_amplitude_scores.json')
    if (status['status'] != 'completed' or status['plan_sha256'] != plan['files_sha256'][capture.as_posix()]
            or (status['worker_starts'], status['physics_steps'], status['training_updates'],
                status['test_episode_files_read'], status['control_accepted']) != (1, 7168, 0, False, False)
            or not status['nominal_gate_passed'] or len(status['accepted_prefixes']) != 4
            or status['artifacts_sha256'][comparison_file.name] != plan['files_sha256'][comparison_file.as_posix()]):
        raise ValueError('missing accepted development capture')
    policy = spec['scoring']
    leads = policy['diagnostic_leads']
    results = read(plan, comparison_file)['models']
    for case in spec['cases']:
        for lead in leads:
            for quantity in QUANTITIES:
                a, b = (newton_data(result[case]['by_lead'][str(lead)][quantity]) for result in results.values())
                if a != b:
                    raise ValueError(f'Newton response differs between model reports for {case}')
    folder = Path(launch['archived_response_validation']).parent/'worker'
    nominal = [load_input(folder/f'{k}.json')['states_raw'] for k in ('nominal_1','nominal_2')]
    models_full = {k: load_input(Path(v)/'timed_rollouts.json') for k,v in launch['fixed_endpoint_sources'].items()}
    source_model = models_full['61_pairs']
    full = {}
    for case in policy['archived_full_half_states_tool_si']:
        pair = source_model['pairs'][case]
        positive, negative = (load_input(folder/f'{case}_{sign}.json')['states_raw'] for sign in ('plus','minus'))
        full[case] = score_case(policy, nominal, positive, negative, source_model['nominal'],
                                {**pair, 'directional_jvp': pair['jvp']}, policy['archived_full_amplitude'])
    guard('full_amplitude')
    pitch = {shape: pitch_amplitudes(shape, results, full, leads) for shape in ('pitch_pulse', 'pitch_reversal')}
    dual = dual_decomposition(results, models_full, full, leads)
    contacts = candidate_log_differences(spec, status, plan)
    write_json(out/'amplitude_and_axis_attribution.json', {'case': spec['case'], 'pitch': pitch, 'dual': dual,
        'contact_candidate_log_differences': contacts,
        'interpretation': 'Newton data use finite signed measurements at each declared amplitude. Dual interaction equals dual response minus the two matched component responses.'})
    return {'newton_actuation_cases': len(spec['cases']), 'amplitude_leads': len(leads), 'models': len(results)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
