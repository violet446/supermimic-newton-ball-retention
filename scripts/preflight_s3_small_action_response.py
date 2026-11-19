"""Freeze small-amplitude development requests and run the controller without Newton."""
import argparse
from itertools import combinations
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json
import numpy as np

from archive_restore_metrics import _half_spreads
from audit_s3_world_increment_data61_response import fixed_models, read
from ball_retention.geometry import load_tool, support_point
from ball_retention.model_data import _ball_state
from ball_retention.world_increment_model import world_action_jvp
from check_s3_deterministic_archive_import import backend_preflight
from deterministic_archive_metrics import native_reference_metrics
from native_repeatability import _nominal_schedule
from preflight_s3_deterministic_action_response import action_sequences, load_plan, verify_commands
from preflight_s3_world_increment import predict
from saved_native_snapshot import decode_snapshot
from train_s3_timing_coverage import read_held


PHASE = 'small_action_response_preflight'
NOMINALS = ('nominal_1', 'nominal_2')
PITCHES = ('pitch_pulse', 'pitch_reversal')
LEADS = (8, 16, 24, 32)


def nominal_gate(new, archived):
    metrics = native_reference_metrics(new, archived)
    trajectories = [{'raw': np.asarray(branch['states_raw'])} for branch in (*new, *archived)]
    diagnostic_leads = range(1, len(trajectories[0]['raw'])+1)
    spread = _half_spreads(trajectories, list(combinations(range(len(trajectories)), 2)), diagnostic_leads)
    resolved = all(spread[str(lead)]['position_m'] <= metrics['position_resolution_m'] for lead in LEADS)
    return {**metrics, 'passed': metrics['numerical_equivalent'] and resolved,
            'position_resolution_passed': resolved, 'position_resolution_gate_leads': LEADS,
            'all_branch_max_half_spread_by_lead': spread}


def build_branches(parent, nominal):
    """Keep the original unit directions and change only each requested amplitude."""
    full_amplitude = parent['amplitude']
    dual_scale = parent['directions']['dual_axis_pulse'][0]['axes'][0]
    dual_segment, = parent['directions']['dual_axis_pulse']
    if dual_segment['axes'] != [dual_scale, dual_scale]:
        raise ValueError('expected equal roll and pitch components')
    cases = {}
    for tag, fraction in (('quarter', .25), ('half', .5)):
        for shape in PITCHES:
            cases[f'{shape}_{tag}'] = {
                'amplitude': full_amplitude*fraction, 'segments': parent['directions'][shape],
                'archive_comparison': shape, 'full_amplitude_fraction': fraction,
            }
    for name, axes in (('roll_component', [1., 0.]), ('pitch_component', [0., 1.])):
        cases[name] = {'amplitude': full_amplitude*dual_scale, 'segments': [{**dual_segment, 'axes': axes}]}

    directions, requests = {}, {label: np.asarray(nominal) for label in NOMINALS}
    for case, spec in cases.items():
        layout = {'horizon': parent['horizon'], 'amplitude': spec['amplitude'],
                  'directions': {case: spec['segments']}}
        shaped, branches = action_sequences(layout, nominal)
        directions.update(shaped)
        for sign in ('plus', 'minus'):
            label = f'{case}_{sign}'
            requests[label] = branches[label]
    _, original = action_sequences(parent, nominal)
    for sign in ('plus', 'minus'):
        base = np.asarray(nominal)
        np.testing.assert_allclose(requests[f'roll_component_{sign}']+requests[f'pitch_component_{sign}']-base,
                                   original[f'dual_axis_pulse_{sign}'], atol=1e-15, rtol=0.)
        for tag in ('quarter', 'half'):
            p, r = (requests[f'{shape}_{tag}_{sign}'] for shape in PITCHES)
            np.testing.assert_array_equal(p[:8], r[:8])
            if np.array_equal(p[8], r[8]):
                raise ValueError('reversal missing at lead nine')
    return cases, directions, requests


def check_controller(parent, cfg, saved, requests, expected, guard):
    # The existing verifier expects unsuffixed pitch labels for its built-in prefix check.
    aliases = {f'{shape}_quarter_{sign}': f'{shape}_{sign}' for shape in PITCHES for sign in ('plus', 'minus')}
    inputs = {aliases.get(label, label): sequence for label, sequence in requests.items()}
    checked, limits = verify_commands(ROOT, parent, cfg, saved, inputs, expected, lambda: guard('controller'))
    commands = {label: checked[aliases.get(label, label)] for label in requests}
    peaks = {label: limits[aliases.get(label, label)] for label in requests}
    for tag in ('quarter', 'half'):
        for sign in ('plus', 'minus'):
            p, r = (commands[f'{shape}_{tag}_{sign}'] for shape in PITCHES)
            if p[:8] != r[:8] or p[8] == r[8]:
                raise ValueError('controller prefix or reversal time differs from its frozen requests')
    return commands, peaks


def model_rollouts(parent, cfg, state, anchor, nominal, cases, directions, requests, endpoints, audit, guard):
    report = {}
    for tag, (model, fold, _, reference) in endpoints.items():
        def roll(actions):
            return predict(model, state, actions, cfg, anchor, fold.known_mean, fold.known_scale)

        baseline = roll(nominal)
        np.testing.assert_array_equal(baseline, reference['nominal'])
        rows = {}
        for name, spec in cases.items():
            direction = directions[name]
            prediction, jvp = world_action_jvp(model, _ball_state(state), state.time_s, state.tilt_rad,
                                              state.tilt_rate_rad_s, nominal, direction, cfg, anchor,
                                              fold.known_mean, fold.known_scale)
            np.testing.assert_array_equal(prediction, baseline)
            step = parent['model']['fd_step']*.25
            finite_slope = (roll(nominal + step*direction)-roll(nominal - step*direction))/(2*step)
            np.testing.assert_allclose(jvp, finite_slope, atol=parent['model']['fd_atol_si'],
                                       rtol=parent['model']['fd_rtol'])
            plus, minus = (roll(requests[f'{name}_{sign}']) for sign in ('plus', 'minus'))
            half = .5*(plus-minus)
            if 'archive_comparison' in spec:
                archive = audit['development'][spec['archive_comparison']]['models'][tag]['sweep']['rows']
                fraction = spec['full_amplitude_fraction']
                old, = [row for row in archive if row['amplitude_fraction'] == fraction]
                np.testing.assert_allclose(half[:, :3]/fraction, old['rescaled_half_response_by_lead_m'],
                                           rtol=0., atol=2e-12)
            rows[name] = {'plus': plus.tolist(), 'minus': minus.tolist(), 'directional_jvp': jvp.tolist(),
                          'max_jvp_fd_difference': float(np.max(np.abs(jvp-finite_slope))),
                          'h32_position_half_norm_m': float(np.linalg.norm(half[-1, :3]))}
            guard(name)
        for level in ('quarter', 'half'):
            p, r = (rows[f'{shape}_{level}'] for shape in PITCHES)
            for key in ('plus', 'minus', 'directional_jvp'):
                np.testing.assert_array_equal(p[key][:8], r[key][:8])
        jvp_sum = (np.asarray(rows['roll_component']['directional_jvp'])
                   + np.asarray(rows['pitch_component']['directional_jvp']))
        dual = parent['directions']['dual_axis_pulse'][0]['axes'][0]
        np.testing.assert_allclose(dual*jvp_sum, reference['pairs']['dual_axis_pulse']['jvp'], rtol=0., atol=2e-12)
        report[tag] = {'nominal': baseline.tolist(), 'pairs': rows}
    return report


def score_protocol(parent, traces, endpoints):
    """Bind the archived full-amplitude measurements and the future scoring rules."""
    truth = {}
    for case in parent['directions']:
        positive, negative = (np.asarray(traces[f'{case}_{sign}']['raw']) for sign in ('plus', 'minus'))
        effect = .5*(positive-negative)
        for _, _, scores, _ in endpoints.values():
            for lead in LEADS:
                np.testing.assert_array_equal(
                    effect[lead-1, :3], scores[case]['by_lead'][str(lead)]['position']['newton_half_difference'])
        truth[case] = effect.tolist()
    gates = parent['gates']
    return {
        'quantity_order': ['position', 'velocity', 'spin'],
        'diagnostic_leads': list(range(1, parent['horizon']+1)),
        'summary_leads': LEADS, 'primary_lead': parent['primary_lead'],
        'absolute_floors_si': [gates[k] for k in ('position_floor_m', 'velocity_floor_m_s', 'spin_floor_rad_s')],
        **{key: gates[key] for key in ('noise_multiplier', 'strong_multiplier', 'min_cosine', 'max_relative_error')},
        'noise_gate': 'Use max(absolute SI floor, 5 times the same-session nominal half spread, float roundoff floor).',
        'eligibility': 'Unresolved below the floor. Report resolved weak signals with raw vectors. Gate cosine and relative error only for signals at least ten times the floor.',
        'secant_and_jvp': 'Store signed plus/minus half difference, half difference divided by actual branch amplitude, and amplitude times the local model JVP. Compare each model to Newton at the same amplitude.',
        'pitch_comparison': 'Compare Newton slopes at 1/4, 1/2 and archived full amplitude using only leads with measured signals above the relevant floors. These are finite secants at all three amplitudes.',
        'dual_comparison': 'Subtract the observed single-axis half difference vectors from the archived full dual half difference at their matching component amplitudes; report the residual and use the sum of the three corresponding floors as its conservative resolution floor.',
        'archived_full_amplitude': parent['amplitude'], 'archived_full_half_states_tool_si': truth,
    }


def run(plan, out, guard):
    parent, cfg, rows, replays, native = load_plan(ROOT)
    held = read_held(read(plan, plan['held_design']))
    capture, held_cfg, state, traces, _ = held
    if plan['updates'] or parent != capture or parent['case'] != plan['development_case'] or cfg['physics'] != held_cfg['physics']:
        raise ValueError('unexpected frozen development case or preflight scope')
    tick, substeps = parent['case']['tick'], parent['control_substeps']
    backend = backend_preflight(parent)
    if int(backend['device']['native_sm']) != parent['runtime']['cuda_arch']:
        raise ValueError('installed backend differs from the accepted snapshot runtime')
    _, nominal, command, expected = _nominal_schedule(ROOT, cfg, rows, tick, substeps, guard)
    saved = decode_snapshot(native['snapshot'], native['layout'], cfg, tick, substeps, command)
    if state != native['snapshot']['state']:
        raise ValueError('archived model state differs from the original native snapshot')
    same_session = nominal_gate(replays, replays)
    if not same_session['passed'] or any(r['commands'] != expected or r['actions'] != nominal for r in replays):
        raise ValueError('nominal replay archive failed admission')
    cases, directions, requests = build_branches(parent, nominal)
    future = plan['future_capture']
    if (list(requests) != plan['branch_order'] or future['baseline_replays'] != len(NOMINALS)
            or future['nominal_prefix_substeps'] != 0 or future['max_worker_starts'] != 1
            or future['sequential_environments'] != 1
            or future['max_physics_substeps'] != len(requests)*parent['horizon']*substeps):
        raise ValueError('next-phase envelope disagrees with the frozen controller branches')
    commands, peaks = check_controller(parent, cfg, saved, requests, expected, guard)
    anchor = support_point(load_tool(ROOT/'assets/shovel.STL'), cfg['initial']['anchor_xy_tool_m'])[0]
    endpoints = fixed_models(plan, held, anchor, guard)
    audited = read(plan, plan['accepted_audit'])
    audit_path = Path(plan['accepted_audit']).with_name('response_attribution.json')
    if (audited['status'] != 'completed' or audited['updates'] or audited['simulation_steps']
            or audited['test_episode_files_read']
            or audited['artifacts_sha256'][audit_path.name] != plan['files_sha256'][audit_path.as_posix()]):
        raise ValueError('missing accepted model-amplitude audit')
    models = model_rollouts(parent, cfg, saved['state'], anchor, np.asarray(nominal), cases, directions, requests,
                            endpoints, read(plan, audit_path), guard)
    write_json(out/'branch_commands.json', {label: {'requested': seq.tolist(), 'executed': commands[label]}
                                            for label, seq in requests.items()})
    write_json(out/'model_rollouts.json', models)
    write_json(out/'replay_spec.json', {
        'case': parent['case'], 'split': 'validation', 'runtime': parent['runtime'], 'static_backend': backend,
        'horizon': parent['horizon'], 'substeps_per_control': substeps, 'cases': cases, 'branch_order': list(requests),
        'same_sign_prefixes': [[f'pitch_pulse_{level}_{sign}', f'pitch_reversal_{level}_{sign}', 8]
                               for level in ('quarter', 'half') for sign in ('plus', 'minus')],
        'static_snapshot_checks': {'snapshot_time_s': saved['state'].time_s,
                                   'saved_physics_steps': saved['physics']['steps'],
                                   'native_layout': native['layout'], 'archived_nominal_gate': same_session},
        'live_capture_requirements': plan['capture_requirements'], 'future_capture_limits': future,
        'scoring': score_protocol(parent, traces, endpoints), 'controller_peaks': peaks,
    })
    return {'development_cases': len(cases), 'static_branches': len(requests),
            'shared_prefixes_verified': 4, 'old_development_scores_reproduced': True, 'snapshot_decoded': True,
            'newton_worker_starts': 0, 'future_capture_max_physics_substeps': future['max_physics_substeps'],
            'live_capture_authorized_by_this_plan': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
