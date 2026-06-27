from __future__ import annotations

import json
import unittest

from morphospace import BubblePolicy
from memory_repair import (
    FatigueDamageConfig,
    FatigueDamageEventSimulator,
    RepairRuleConfig,
    RepairableFrozenEventSimulator,
)


def _same_core_result(left, right) -> None:
    testcase = unittest.TestCase()
    testcase.assertEqual(left.completed, right.completed)
    testcase.assertEqual(left.stop_reason, right.stop_reason)
    testcase.assertEqual(left.final_values, right.final_values)
    testcase.assertEqual(left.final_frozen_positions, right.final_frozen_positions)
    testcase.assertEqual(left.swap_count, right.swap_count)
    testcase.assertEqual(left.comparison_count, right.comparison_count)
    testcase.assertEqual(left.activation_count, right.activation_count)
    testcase.assertEqual([row["state_hash"] for row in left.trace_rows], [row["state_hash"] for row in right.trace_rows])


class TestE04FatigueDamage(unittest.TestCase):
    def test_disabled_fatigue_preserves_permanent_baseline(self) -> None:
        values = [8, 4, 7, 2, 6, 1, 5, 3]
        kwargs = {
            "frozen_positions": [2],
            "frozen_variant": "stuck",
            "scheduler_seed": 1234,
            "tie_breaker_seed": 5678,
            "trace_signal_activations": False,
            "trace_memory_activations": False,
        }
        base = RepairableFrozenEventSimulator(
            values,
            BubblePolicy(),
            signal_config="no_signal",
            repair_config="permanent",
            **kwargs,
        ).run(max_activations=200000)
        fatigue = FatigueDamageEventSimulator(
            values,
            BubblePolicy(),
            signal_config="no_signal",
            repair_config="permanent",
            fatigue_config="none",
            **kwargs,
        ).run(max_activations=200000)
        _same_core_result(fatigue, base)

    def test_disabled_fatigue_preserves_repairable_baseline(self) -> None:
        values = [2, 1, 3]
        kwargs = {
            "frozen_positions": [1],
            "frozen_variant": "stuck",
            "scheduler_seed": 1,
            "tie_breaker_seed": 1,
            "trace_signal_activations": False,
            "trace_memory_activations": False,
        }
        repair = RepairRuleConfig("nudge_count", nudge_threshold=1)
        base = RepairableFrozenEventSimulator(values, BubblePolicy(), repair_config=repair, **kwargs)
        fatigue = FatigueDamageEventSimulator(values, BubblePolicy(), repair_config=repair, fatigue_config="none", **kwargs)
        for _ in range(4):
            left = fatigue.step(forced_cell_id=0, forced_direction=1)
            right = base.step(forced_cell_id=0, forced_direction=1)
            self.assertEqual(left.swapped, right.swapped)
            self.assertEqual(left.blocked_move_attempt, right.blocked_move_attempt)
            self.assertEqual(fatigue.current_values(), base.current_values())
            self.assertEqual(fatigue.current_frozen_positions(), base.current_frozen_positions())

    def test_movement_threshold_induces_and_recovers_fatigue(self) -> None:
        sim = FatigueDamageEventSimulator(
            [3, 2, 1],
            BubblePolicy(),
            fatigue_config=FatigueDamageConfig("movement", movement_threshold=1, recovery_activations=2),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        first = sim.step(forced_cell_id=2, forced_direction=-1)
        self.assertTrue(first.swapped)
        state = sim.fatigue_states[2]
        self.assertEqual(state["fatigue_cooldown_remaining"], 2)
        self.assertEqual(state["impaired_direction"], "either")

        second = sim.step(forced_cell_id=2, forced_direction=-1)
        self.assertTrue(second.blocked_move_attempt)
        self.assertEqual(second.reason, "fatigue_impaired")
        self.assertEqual(sim.fatigue_states[2]["fatigue_cooldown_remaining"], 1)

        third = sim.step(forced_cell_id=2, forced_direction=-1)
        self.assertTrue(third.blocked_move_attempt)
        self.assertEqual(sim.fatigue_states[2]["fatigue_cooldown_remaining"], 0)
        self.assertFalse(sim.fatigue_states[2]["fatigued"])

        fourth = sim.step(forced_cell_id=2, forced_direction=-1)
        self.assertTrue(fourth.swapped)

    def test_failed_swap_threshold_induces_fatigue(self) -> None:
        sim = FatigueDamageEventSimulator(
            [2, 1],
            BubblePolicy(),
            frozen_positions=[1],
            frozen_variant="stuck",
            repair_config="permanent",
            fatigue_config=FatigueDamageConfig("failed_swap", failed_swap_threshold=2, recovery_activations=3),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        sim.step(forced_cell_id=0, forced_direction=1)
        self.assertEqual(sim.fatigue_states[0]["failed_swap_count"], 1)
        sim.step(forced_cell_id=0, forced_direction=1)
        self.assertEqual(sim.fatigue_states[0]["fatigue_cooldown_remaining"], 3)
        self.assertEqual(sim.fatigue_events[-1]["reason"], "failed_swap_threshold_met")

    def test_frustration_threshold_from_waits(self) -> None:
        sim = FatigueDamageEventSimulator(
            [1, 2],
            BubblePolicy(),
            fatigue_config=FatigueDamageConfig("frustration", frustration_threshold=2, recovery_activations=2),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        sim.step(forced_cell_id=0, forced_direction=-1)
        self.assertEqual(sim.fatigue_states[0]["frustration_count"], 1)
        sim.step(forced_cell_id=0, forced_direction=-1)
        self.assertEqual(sim.fatigue_states[0]["fatigue_cooldown_remaining"], 2)
        self.assertEqual(sim.fatigue_events[-1]["reason"], "frustration_threshold_met")

    def test_direction_specific_impairment_blocks_only_matching_direction(self) -> None:
        sim = FatigueDamageEventSimulator(
            [3, 2, 1],
            BubblePolicy(),
            frozen_positions=[2],
            frozen_variant="stuck",
            repair_config="permanent",
            fatigue_config=FatigueDamageConfig(
                "directional",
                failed_swap_threshold=1,
                recovery_activations=3,
                impaired_direction="attempted",
            ),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        first = sim.step(forced_cell_id=1, forced_direction=1)
        self.assertTrue(first.blocked_move_attempt)
        self.assertEqual(sim.fatigue_states[1]["impaired_direction"], "right")

        second = sim.step(forced_cell_id=1, forced_direction=1)
        self.assertTrue(second.blocked_move_attempt)
        self.assertEqual(second.reason, "fatigue_impaired")

        third = sim.step(forced_cell_id=1, forced_direction=-1)
        self.assertTrue(third.swapped)
        self.assertEqual(sim.current_values(), [2, 3, 1])

    def test_stochastic_damage_probability_and_recovery(self) -> None:
        sim = FatigueDamageEventSimulator(
            [3, 2, 1],
            BubblePolicy(),
            fatigue_config=FatigueDamageConfig(
                "stochastic_damage",
                damage_probability=1.0,
                damage_cooldown_activations=1,
                random_seed=7,
            ),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        first = sim.step(forced_cell_id=2, forced_direction=-1)
        self.assertTrue(first.swapped)
        self.assertTrue(sim.fatigue_states[2]["damaged"])
        self.assertEqual(sim.fatigue_states[2]["damage_cooldown_remaining"], 1)

        second = sim.step(forced_cell_id=2, forced_direction=-1)
        self.assertTrue(second.blocked_move_attempt)
        self.assertEqual(second.reason, "damage_impaired")
        self.assertFalse(sim.fatigue_states[2]["damaged"])
        self.assertEqual(sim.fatigue_states[2]["damage_cooldown_remaining"], 0)

    def test_fatigue_trace_fields_are_json(self) -> None:
        sim = FatigueDamageEventSimulator(
            [3, 2, 1],
            BubblePolicy(),
            fatigue_config=FatigueDamageConfig("movement", movement_threshold=1, recovery_activations=2),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        sim.step(forced_cell_id=2, forced_direction=-1)
        row = sim.trace_rows[-1]
        json.loads(row["fatigue_config_json"])
        states = json.loads(row["fatigue_states_json"])
        events = json.loads(row["fatigue_events_json"])
        self.assertEqual(states[2]["fatigue_cooldown_remaining"], 2)
        self.assertEqual(events[-1]["event_type"], "fatigue_induced")


if __name__ == "__main__":
    unittest.main(verbosity=2)
