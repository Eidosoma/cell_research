from __future__ import annotations

import json
import unittest

from morphospace import BubblePolicy, PolicyEventSimulator
from memory_repair import (
    RepairRuleConfig,
    RepairableFrozenEventSimulator,
    SignalConfig,
    SignalEventSimulator,
)


class TestE04RepairableFrozen(unittest.TestCase):
    def test_permanent_stuck_baseline_matches_signal_simulator(self) -> None:
        values = [8, 4, 7, 2, 6, 1, 5, 3]
        kwargs = {
            "frozen_positions": [2],
            "frozen_variant": "stuck",
            "scheduler_seed": 1234,
            "tie_breaker_seed": 5678,
            "trace_signal_activations": False,
            "trace_memory_activations": False,
        }
        base = SignalEventSimulator(values, BubblePolicy(), signal_config="no_signal", **kwargs).run(
            max_activations=200000
        )
        repair = RepairableFrozenEventSimulator(
            values,
            BubblePolicy(),
            signal_config="no_signal",
            repair_config="permanent",
            **kwargs,
        ).run(max_activations=200000)
        self.assertEqual(repair.completed, base.completed)
        self.assertEqual(repair.stop_reason, base.stop_reason)
        self.assertEqual(repair.final_values, base.final_values)
        self.assertEqual(repair.final_frozen_positions, base.final_frozen_positions)
        self.assertEqual(repair.swap_count, base.swap_count)
        self.assertEqual(repair.comparison_count, base.comparison_count)
        self.assertEqual([row["state_hash"] for row in repair.trace_rows], [row["state_hash"] for row in base.trace_rows])

    def test_nudge_threshold_recovery(self) -> None:
        sim = RepairableFrozenEventSimulator(
            [2, 1],
            BubblePolicy(),
            frozen_positions=[1],
            frozen_variant="stuck",
            repair_config=RepairRuleConfig("nudge_count", nudge_threshold=2),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        first = sim.step(forced_cell_id=0, forced_direction=1)
        self.assertTrue(first.blocked_move_attempt)
        self.assertEqual(sim.current_frozen_positions(), [1])
        second = sim.step(forced_cell_id=0, forced_direction=1)
        self.assertTrue(second.blocked_move_attempt)
        self.assertEqual(sim.current_frozen_positions(), [])
        third = sim.step(forced_cell_id=0, forced_direction=1)
        self.assertTrue(third.swapped)
        self.assertEqual(sim.current_values(), [1, 2])

    def test_elapsed_time_recovery(self) -> None:
        sim = RepairableFrozenEventSimulator(
            [2, 1],
            BubblePolicy(),
            frozen_positions=[1],
            frozen_variant="stuck",
            repair_config=RepairRuleConfig("elapsed_time", elapsed_activations=2),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        sim.step(forced_cell_id=0, forced_direction=-1)
        self.assertEqual(sim.current_frozen_positions(), [1])
        sim.step(forced_cell_id=0, forced_direction=-1)
        self.assertEqual(sim.current_frozen_positions(), [])

    def test_direction_contact_recovery(self) -> None:
        sim = RepairableFrozenEventSimulator(
            [2, 1],
            BubblePolicy(),
            frozen_positions=[1],
            frozen_variant="stuck",
            repair_config=RepairRuleConfig("direction_contact", nudge_threshold=1, approach_direction="from_left"),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        outcome = sim.step(forced_cell_id=0, forced_direction=1)
        self.assertTrue(outcome.blocked_move_attempt)
        self.assertEqual(sim.current_frozen_positions(), [])

    def test_direction_contact_threshold_counts_matching_direction_only(self) -> None:
        sim = RepairableFrozenEventSimulator(
            [3, 2, 1],
            BubblePolicy(),
            frozen_positions=[1],
            frozen_variant="stuck",
            repair_config=RepairRuleConfig("direction_contact", nudge_threshold=2, approach_direction="from_left"),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        sim.step(forced_cell_id=2, forced_direction=-1)
        self.assertEqual(sim.current_frozen_positions(), [1])
        sim.step(forced_cell_id=0, forced_direction=1)
        self.assertEqual(sim.current_frozen_positions(), [1])
        sim.step(forced_cell_id=0, forced_direction=1)
        self.assertEqual(sim.current_frozen_positions(), [])

    def test_signal_threshold_recovery(self) -> None:
        config = SignalConfig("nearest_neighbor", signal_range=1)
        sim = RepairableFrozenEventSimulator(
            [2, 1, 3],
            BubblePolicy(),
            frozen_positions=[1],
            frozen_variant="stuck",
            signal_config=config,
            repair_config=RepairRuleConfig("signal_threshold", signal_channel="blocked", signal_threshold=1.0),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        outcome = sim.step(forced_cell_id=0, forced_direction=1)
        self.assertTrue(outcome.blocked_move_attempt)
        self.assertEqual(sim.current_frozen_positions(), [])
        state = json.loads(sim.trace_rows[-1]["repair_states_json"])[0]
        self.assertGreaterEqual(state["signal_exposure"], 1.0)

    def test_repair_trace_fields_are_json(self) -> None:
        sim = RepairableFrozenEventSimulator(
            [2, 1],
            BubblePolicy(),
            frozen_positions=[1],
            frozen_variant="stuck",
            repair_config=RepairRuleConfig("nudge_count", nudge_threshold=1),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        sim.step(forced_cell_id=0, forced_direction=1)
        json.loads(sim.trace_rows[-1]["repair_config_json"])
        json.loads(sim.trace_rows[-1]["repair_states_json"])
        events = json.loads(sim.trace_rows[-1]["recovery_events_json"])
        self.assertEqual(len(events), 1)

    def test_original_policy_baseline_still_runs(self) -> None:
        result = PolicyEventSimulator(
            [2, 1],
            "bubble",
            frozen_positions=[1],
            frozen_variant="stuck",
            scheduler_seed=1,
            tie_breaker_seed=1,
        ).run(max_activations=20)
        self.assertFalse(result.completed)
        self.assertEqual(result.final_frozen_positions, [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
