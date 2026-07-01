"""S01 policy-interface tests for original cell-view policies."""

from __future__ import annotations

import random
import unittest
from dataclasses import dataclass

from src.e02.deterministic_simulator import (
    EventTracingStatusProbe,
    SimulatorConfig,
    _build_cells,
)
from src.e03.policy_interface import (
    OriginalCellPolicyWrapper,
    PolicyAction,
    cells_signature,
    observe_cell,
    policy_for_cell,
)


@dataclass(frozen=True)
class DecisionCase:
    name: str
    config: SimulatorConfig
    actor_index: int
    policy_seed: int
    expected_behavior: str


DECISION_CASES = (
    DecisionCase(
        name="bubble_swap_right",
        config=SimulatorConfig(values=(2, 1), algorithm="bubble"),
        actor_index=0,
        policy_seed=1,
        expected_behavior="bubble",
    ),
    DecisionCase(
        name="bubble_reverse_swap_right",
        config=SimulatorConfig(values=(1, 2), algorithm="bubble", reverse_directions=(True, True)),
        actor_index=0,
        policy_seed=1,
        expected_behavior="bubble",
    ),
    DecisionCase(
        name="insertion_swap_left",
        config=SimulatorConfig(values=(2, 1, 3), algorithm="insertion"),
        actor_index=1,
        policy_seed=11,
        expected_behavior="insertion",
    ),
    DecisionCase(
        name="selection_swap_to_ideal",
        config=SimulatorConfig(values=(2, 1, 3), algorithm="selection"),
        actor_index=1,
        policy_seed=21,
        expected_behavior="selection",
    ),
    DecisionCase(
        name="selection_update_ideal_without_swap",
        config=SimulatorConfig(values=(1, 2, 3), algorithm="selection"),
        actor_index=1,
        policy_seed=31,
        expected_behavior="selection",
    ),
)


def build_cells(config: SimulatorConfig):
    probe = EventTracingStatusProbe()
    cells, _cell_status = _build_cells(config, probe)
    return cells, probe


def probe_counts(probe: EventTracingStatusProbe) -> tuple[int, int, int]:
    return (
        int(probe.compare_and_swap_count),
        int(probe.swap_count),
        int(probe.frozen_swap_attempts),
    )


class PolicyInterfaceTests(unittest.TestCase):
    def test_observation_exposes_cell_local_fields_and_state(self) -> None:
        cells, _probe = build_cells(SimulatorConfig(values=(3, 1, 2), algorithm="selection"))
        cell = cells[1]
        observation = observe_cell(cell)
        self.assertEqual(observation.behavior, "selection")
        self.assertEqual(observation.actor_index, 1)
        self.assertEqual(observation.actor_value, 1)
        self.assertEqual(observation.values, (3, 1, 2))
        self.assertEqual(observation.statuses, ("ACTIVE", "ACTIVE", "ACTIVE"))
        self.assertEqual(observation.state.ideal_position, 0)

    def test_original_wrappers_match_direct_public_move_decisions(self) -> None:
        for case in DECISION_CASES:
            with self.subTest(case=case.name):
                direct_cells, direct_probe = build_cells(case.config)
                wrapper_cells, wrapper_probe = build_cells(case.config)

                random.seed(case.policy_seed)
                direct_cells[case.actor_index].move()
                direct_signature = cells_signature(direct_cells)
                direct_counts = probe_counts(direct_probe)

                random.seed(case.policy_seed)
                wrapper = policy_for_cell(wrapper_cells[case.actor_index])
                result = wrapper.step(wrapper_cells[case.actor_index])
                wrapper_signature = cells_signature(wrapper_cells)
                wrapper_counts = probe_counts(wrapper_probe)

                self.assertEqual(wrapper.behavior, case.expected_behavior)
                self.assertEqual(result.behavior, case.expected_behavior)
                self.assertEqual(wrapper_signature, direct_signature)
                self.assertEqual(wrapper_counts, direct_counts)

    def test_proposed_actions_match_applied_actions_on_decision_fixtures(self) -> None:
        equivalent_applied = {
            ("update_state", "wait"),
        }
        for case in DECISION_CASES:
            with self.subTest(case=case.name):
                cells, _probe = build_cells(case.config)
                random.seed(case.policy_seed)
                wrapper = policy_for_cell(cells[case.actor_index])
                result = wrapper.step(cells[case.actor_index])
                pair = (result.proposed_action.action_type, result.applied_action.action_type)
                self.assertTrue(
                    pair[0] == pair[1] or pair in equivalent_applied,
                    f"proposal/applied mismatch: {pair}",
                )
                self.assertEqual(result.proposed_action.compare_counted, result.applied_action.compare_counted)
                if result.proposed_action.action_type == "swap":
                    self.assertEqual(result.proposed_action.target_index, result.applied_action.target_index)
                self.assertTrue(result.proposed_action.constraints)

    def test_mixed_algotype_labels_resolve_to_configured_original_behavior(self) -> None:
        config = SimulatorConfig(
            values=(4, 1, 3, 2),
            algotypes=("bubble_label_a", "bubble_label_b", "insertion", "selection"),
        )
        cells, _probe = build_cells(config)
        observed = [policy_for_cell(cell, config.label_to_behavior).behavior for cell in cells]
        self.assertEqual(observed, ["bubble", "bubble", "insertion", "selection"])

    def test_policy_action_allowed_reflects_constraint_checks(self) -> None:
        action = PolicyAction(
            action_type="swap",
            constraints=(
                OriginalCellPolicyWrapper("bubble")
                .propose_action(observe_cell(build_cells(SimulatorConfig(values=(2, 1), algorithm="bubble"))[0][0]), random.Random(1))
                .constraints
            ),
        )
        self.assertTrue(action.allowed)


if __name__ == "__main__":
    unittest.main()
