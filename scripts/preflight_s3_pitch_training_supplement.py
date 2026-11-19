"""Validate the missing advance history in the negative-y training group."""
import argparse
import json
from pathlib import Path

from s3_fitability_common import ROOT, run_phase, write_json

import numpy as np

from ball_retention.archive_state_import import _parse_state
from ball_retention.model_data import _ball_state
from preflight_s3_training_timing_capture import load_cases, verify_case
from train_s3_local_velocity_increment import bound


PHASE = 'pitch_training_supplement_preflight'


def run(plan, out, guard):
    def read(path):
        return json.loads(bound(plan, path))

    if plan['updates'] or plan['timing_design']['snapshot_tick'] != 168:
        raise ValueError('unexpected supplement preflight scope')
    prior = read(plan['accepted_coverage'])
    if not prior['executed_fk_checked'] or not prior['baseline_transfer_reproduced']:
        raise ValueError('missing accepted historical coverage')
    capture = plan['timing_design']
    design, rows = load_cases(ROOT, capture)
    (case, config, archives, clearance), = rows
    if case['history_id'] != 'advance' or case['split'] != 'train':
        raise ValueError('unexpected candidate history')
    existing = [(Path(path).parent, entry) for path in plan['existing_timed_manifests']
                for entry in read(path)['entries']]
    group_cells = {(entry['group_id'], entry['history_id']) for _, entry in existing}
    if (len(existing) != 3 or (case['group_id'], 'advance') in group_cells
            or (case['group_id'], 'reverse') not in group_cells):
        raise ValueError('expected one missing advance cell')
    (folder, reference), = [x for x in existing if x[1]['group_id'] == case['group_id']]
    if case['physical_initial_sha256'] != reference['physical_initial_sha256']:
        raise ValueError('candidate changes the existing negative-y initial group')
    old = read(plan['training_inventory'])['advance_sampling_candidates']['options']
    option, = [row for row in old if row['episode'] == case['episode_id']]
    if not option['eligible_from_archives'] or option['tick'] != 168:
        raise ValueError('candidate failed archived nominal or held-axis checks')
    payload, report = verify_case(ROOT, design, case, config, archives, clearance, guard)
    initial = np.asarray(archives[-1]['initial_raw'])
    saved = folder/reference['native_snapshot']['path']
    ref_initial = _ball_state(_parse_state(read(saved)['state']))
    report['archived_tick_168_ball_distances_si'] = np.linalg.norm((initial-ref_initial).reshape(3,3), axis=1).tolist()
    report['same_physical_initial_group_as_reverse'] = True
    report['candidate'] = case
    report['archived_nominal_noise_probe'] = option
    write_json(out/'commands.json', payload)
    write_json(out/'capture_preflight.json', report)
    return {'training_case': case['episode_id'], 'deterministic_controller_preflight_passed': True,
            'simulator_worker_started': False, 'new_training_pairs_published': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run_phase(args.plan.resolve(), args.sha256, PHASE, run)
