from __future__ import annotations

import json
import unittest

from morphospace import BubblePolicy, DSLPolicy, PolicyEventSimulator, local_inversion_program
from memory_repair import (
    MEMORY_STATE_KEY,
    MemoryConfig,
    MemoryEventSimulator,
    MemoryPolicyWrapper,
    build_memory_variants,
    memory_policy_from_json,
    memory_policy_to_json,
    memory_state_for_trace,
    reset_all_memory,
)


class TestE04MemoryExtension(unittest.TestCase):
    def test_no_memory_wrapper_matches_e03(self) -> None:
        values = [6, 2, 4, 1, 5, 3]
        base = PolicyEventSimulator(values, "bubble", scheduler_seed=123, tie_breaker_seed=456).run(
            max_activations=200000
        )
        wrapped = MemoryEventSimulator(
            values,
            MemoryPolicyWrapper(BubblePolicy(), "no_memory"),
            scheduler_seed=123,
            tie_breaker_seed=456,
            trace_memory_activations=False,
        ).run(max_activations=200000)
        self.assertEqual(wrapped.completed, base.completed)
        self.assertEqual(wrapped.stop_reason, base.stop_reason)
        self.assertEqual(wrapped.final_values, base.final_values)
        self.assertEqual(wrapped.swap_count, base.swap_count)
        self.assertEqual(wrapped.comparison_count, base.comparison_count)
        self.assertEqual(
            [row["state_hash"] for row in wrapped.trace_rows],
            [row["state_hash"] for row in base.trace_rows],
        )

    def test_memory_state_updates_and_resets(self) -> None:
        policy = MemoryPolicyWrapper(BubblePolicy(), MemoryConfig("bounded_counter", counter_max=3))
        sim = MemoryEventSimulator(
            [2, 1],
            policy,
            frozen_positions=[1],
            frozen_variant="stuck",
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        sim.step(forced_cell_id=0, forced_direction=1)
        state = sim.cells[sim.positions_by_id[0]].state
        self.assertIn(MEMORY_STATE_KEY, state)
        self.assertEqual(memory_state_for_trace(state)["recent_failed_swaps"], 1)
        reset_all_memory(sim.cells)
        self.assertEqual(memory_state_for_trace(state)["recent_failed_swaps"], 0)

    def test_neighbor_memory_trace_is_json_serializable(self) -> None:
        policy = MemoryPolicyWrapper(BubblePolicy(), MemoryConfig("neighbor_memory", counter_max=5, neighbor_history=2))
        sim = MemoryEventSimulator([3, 1, 2], policy, scheduler_seed=4, tie_breaker_seed=5)
        sim.run(max_activations=20)
        self.assertTrue(sim.trace_rows)
        parsed = json.loads(sim.trace_rows[-1]["memory_states_json"])
        self.assertEqual(len(parsed), 3)
        memory_lengths = [
            len(row["memory_state"].get("recent_neighbor_ids", []))
            for row in parsed
            if row["memory_variant"] == "neighbor_memory"
        ]
        self.assertTrue(all(length <= 2 for length in memory_lengths))

    def test_memory_policy_round_trip(self) -> None:
        policy = MemoryPolicyWrapper(DSLPolicy(local_inversion_program()), MemoryConfig("one_bit"))
        restored = memory_policy_from_json(memory_policy_to_json(policy))
        self.assertEqual(restored.to_spec().to_dict(), policy.to_spec().to_dict())

    def test_build_memory_variants(self) -> None:
        variants = build_memory_variants(BubblePolicy())
        self.assertEqual([policy.memory_config.variant for policy in variants], [
            "no_memory",
            "one_bit",
            "bounded_counter",
            "neighbor_memory",
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
