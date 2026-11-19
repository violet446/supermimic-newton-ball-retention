from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from diagnostics.resource_probe.deterministic_archive_metrics import (
    historical_import_metrics,
    native_reference_metrics,
    same_configuration_metrics,
    validate_recorded_replays,
)


def nominal_replay():
    body = {'position': [0., 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.],
            'linear_velocity_origin': [0., 0., 0.], 'angular_velocity': [0., 0., 0.]}
    replay = {'actions': [], 'substeps': [], 'elapsed_s': [], 'states': [], 'states_raw': [],
              'commands': [], 'retention': [], 'terminated': False, 'truncated': False,
              'termination_reason': None}
    for tick in range(169, 201):
        now = (tick * 16) * (1 / 960)
        action = [.16, 0.]
        replay['actions'].append(action)
        replay['substeps'].append(16)
        replay['elapsed_s'].append(now - (168 * 16) * (1 / 960))
        replay['states'].append({'time_s': now, 'ball': deepcopy(body), 'tool': deepcopy(body),
                                 'joint_q': [0.] * 6, 'joint_qd': [0.] * 6, 'tilt_rad': [0.] * 2,
                                 'tilt_rate_rad_s': [0.] * 2, 'previous_action': action,
                                 'reference_phase': .9, 'departure_outside_s': 0.})
        replay['states_raw'].append([0.] * 9)
        replay['commands'].append({'q': [[0.] * 6 for _ in range(16)], 'qd': [[0.] * 6 for _ in range(16)]})
        replay['retention'].append({'terminated': False, 'truncated': False, 'reason': None,
                                    'carrying_candidate': True, 'outside_grace_active': False})
    return replay


def shift_position(replay, lead, delta):
    replay['states'][lead - 1]['ball']['position'][0] += delta
    replay['states_raw'][lead - 1][0] += delta


class DeterministicArchiveMetricsTests(unittest.TestCase):
    def setUp(self):
        self.native = nominal_replay()
        self.native_pair = [self.native, deepcopy(self.native)]

    def test_all_pairs_and_h32_cross_source_resolution(self):
        changed = deepcopy(self.native)
        shift_position(changed, 8, 3e-6)
        same = same_configuration_metrics(self.native_pair, [self.native, changed])
        self.assertTrue(same['passed'])
        self.assertEqual(len(same['pairwise']), 6)
        self.assertEqual(same['cross_source_max_half_spread_by_lead']['8']['position_m'], 1.5e-6)
        shift_position(changed, 32, 3e-6)
        self.assertFalse(same_configuration_metrics(self.native_pair, [self.native, changed])['passed'])
        self.assertFalse(native_reference_metrics([self.native, changed], self.native_pair)['passed'])

    def test_old_config_divergence_does_not_gate_new_repeatability(self):
        old = {key: deepcopy(self.native[key]) for key in ('actions', 'substeps', 'elapsed_s', 'states_raw',
                                                          'terminated', 'truncated', 'termination_reason')}
        old['states_raw'][31][0] = .001
        result = historical_import_metrics(self.native_pair, [old, old])
        self.assertTrue(result['passed'])
        divergence = result['cross_configuration']['cross_source_max_half_spread_by_lead']
        self.assertEqual(divergence['32']['position_m'], .0005)
        changed = deepcopy(self.native)
        shift_position(changed, 8, 3e-6)
        self.assertFalse(historical_import_metrics([self.native, changed], [old, old])['passed'])

    def test_controls_and_retention_must_match_at_every_lead(self):
        changed = deepcopy(self.native)
        changed['commands'][6]['qd'][2][0] = 1e-12
        self.assertFalse(native_reference_metrics(self.native_pair, [self.native, changed])['passed'])
        changed['commands'][6]['qd'][2][0] = 0.
        changed['retention'][19]['carrying_candidate'] = False
        self.assertFalse(same_configuration_metrics(self.native_pair, [self.native, changed])['passed'])

    def test_preflight_rejects_corrupt_lead_clock_and_public_values(self):
        changed = deepcopy(self.native)
        changed['states'][10]['time_s'] += 1e-10
        with self.assertRaisesRegex(ValueError, 'replay clocks'):
            validate_recorded_replays([self.native, changed], self.native_pair, self.native['actions'])
        changed['states'][10]['time_s'] = self.native['states'][10]['time_s']
        changed['states_raw'][10][0] = 1e-6
        with self.assertRaisesRegex(ValueError, 'tool-frame values'):
            validate_recorded_replays([self.native, changed], self.native_pair, self.native['actions'])
