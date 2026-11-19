"""CPU-only regressions for the frozen response worker's sequencing and guards."""
from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'diagnostics/resource_probe')]

from ball_retention.archive_state_import import _parse_state
from ball_retention.contracts import EPISODE_SCHEMA, Transition
from ball_retention.deterministic_response_plan import load
import deterministic_action_response as worker


class ReplayScene:
    """Supply documented command/state/contact data while the real worker orchestrates."""

    def __init__(self, plan, cfg, evidence, altered_position=None, partial_step=False):
        self.branches = iter(plan['branch_order'])
        self.origin = plan['case']['tick']
        self.substeps = plan['control_substeps']
        self.dt = cfg['physics']['dt_s']
        self.commands = evidence['branch_commands']
        self.recorded = evidence['recorded'][0]
        self.altered_position = altered_position
        self.partial_step = partial_step
        self.physics = SimpleNamespace(steps=0, replay_contacts=None)
        self.restores = []

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.step_index = 0

    def restore(self, saved):
        self.label = next(self.branches)
        self.restores.append(self.label)
        self.episode_id = saved['episode_id']
        self.step_index = saved['step_index']
        self.physics.steps = saved['physics']['steps']
        self.state = deepcopy(saved['state'])

    def step(self, action):
        lead = self.step_index-self.origin
        branch = self.commands[self.label]
        if action != branch['requested'][lead]:
            raise AssertionError(f'worker supplied an unexpected action in {self.label}')
        fields = deepcopy(self.recorded['states'][lead])
        requested = branch['requested'][lead]
        command = branch['executed'][lead]
        fields.update(joint_q=command['q'][-1], joint_qd=command['qd'][-1], previous_action=requested)
        if self.label.endswith('plus'):
            fields['ball']['position'][0] += 2e-4
        elif self.label.endswith('minus'):
            fields['ball']['position'][0] -= 2e-4
        if (self.label, lead) == self.altered_position:
            fields['ball']['position'][0] += 3e-6
        applied = self.substeps-1 if self.partial_step else self.substeps
        start = lead*self.substeps
        self.physics.replay_contacts.extend(self.recorded['contacts'][start:start+applied])
        self.physics.steps += applied
        fields['time_s'] = self.physics.steps*self.dt
        state = _parse_state(fields)
        row = Transition(self.episode_id, self.step_index, self.state, requested, command['q'][:applied],
                         command['qd'][:applied], state, self.partial_step, False, None, None, {}, {},
                         schema_version=EPISODE_SCHEMA,
                         task={'termination_reason': 'mock_departure' if self.partial_step else None,
                               'carrying': {'carrying_candidate': not self.partial_step}})
        self.step_index += 1
        self.state = state
        return row


class DeterministicResponseWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.cfg, cls.evidence = load(ROOT)
        cls.steps = cls.plan['budget']['max_physics_substeps']

    def setUp(self):
        temporary = TemporaryDirectory(dir=ROOT/'.tmp')
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.summary = {'physics_steps': 0}
        self.stages = []

    def run_scene(self, scene, steps=None):
        def restore(env, saved, substeps):
            self.assertEqual(substeps, scene.substeps)
            env.restore(saved)
            return {'both_buffers_exact': True}

        with (patch.object(worker, '_make_env', return_value=scene),
              patch.object(worker, 'verify_live_layout', return_value=self.evidence['layout']),
              patch.object(worker, 'restore_and_readback', side_effect=restore)):
            worker.run(ROOT, self.cfg, self.out, self.steps if steps is None else steps,
                       self.summary, self.stages.append)
        self.assertFalse({'newton', 'warp'}.intersection(sys.modules))

    def test_nominal_h8_spread_blocks_all_perturbations(self):
        scene = ReplayScene(self.plan, self.cfg, self.evidence, altered_position=('nominal_2', 7))
        with self.assertRaisesRegex(RuntimeError, 'nominal.*gate'):
            self.run_scene(scene)
        self.assertEqual(scene.restores, self.plan['branch_order'][:2])
        self.assertEqual(self.summary['physics_steps'], 1024)
        self.assertTrue((self.out/'nominal_comparison.json').is_file())
        self.assertFalse((self.out/'pitch_pulse_plus.json').exists())

    def test_shared_h8_prefix_failure_stops_later_branches(self):
        scene = ReplayScene(self.plan, self.cfg, self.evidence, altered_position=('pitch_reversal_plus', 7))
        with self.assertRaisesRegex(RuntimeError, 'shared action prefix'):
            self.run_scene(scene)
        self.assertEqual(scene.restores, self.plan['branch_order'][:5])
        self.assertEqual(self.summary['physics_steps'], 2560)

    def test_incomplete_transition_preserves_actual_substep_count(self):
        scene = ReplayScene(self.plan, self.cfg, self.evidence, partial_step=True)
        with self.assertRaisesRegex(RuntimeError, 'transition'):
            self.run_scene(scene)
        self.assertEqual(scene.restores, self.plan['branch_order'][:1])
        self.assertEqual(self.summary['physics_steps'], 15)

    def test_frozen_cap_mismatch_never_restores_a_branch(self):
        scene = ReplayScene(self.plan, self.cfg, self.evidence)
        with self.assertRaises(ValueError):
            self.run_scene(scene, self.steps+16)
        self.assertEqual(scene.restores, [])
        self.assertEqual(self.summary['physics_steps'], 0)

    def test_valid_sequence_runs_only_eight_frozen_branches(self):
        scene = ReplayScene(self.plan, self.cfg, self.evidence)
        self.run_scene(scene)
        self.assertEqual(scene.restores, self.plan['branch_order'])
        self.assertEqual(self.summary['completed_branches'], self.plan['branch_order'])
        self.assertEqual(self.summary['status'], 'deterministic_action_response_completed')
        self.assertEqual(self.summary['physics_steps'], self.steps)
        self.assertLess(self.stages.index('nominal_comparison'), self.stages.index('pitch_pulse_plus_start'))
        self.assertLess(self.stages.index('shared_prefix_minus'), self.stages.index('dual_axis_pulse_plus_start'))


if __name__ == '__main__':
    unittest.main()
