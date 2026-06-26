from __future__ import annotations

import unittest

from e02_deterministic_simulator import (
    DeterministicEventSimulator,
    aggregation,
    monotonicity_error,
    sortedness_percent,
    sortedness_raw,
)


class TestDeterministicEventSimulator(unittest.TestCase):
    def test_metric_contract(self) -> None:
        self.assertEqual(sortedness_raw([1, 2, 3, 4]), 3)
        self.assertEqual(sortedness_percent([1, 2, 3, 4]), 100.0)
        self.assertAlmostEqual(sortedness_percent([1, 3, 2, 4]), 100.0 * 2 / 3)
        self.assertEqual(monotonicity_error([1, 3, 2, 4]), 1)
        self.assertAlmostEqual(aggregation(["bubble", "bubble", "selection"]), 0.5)

    def test_passive_frozen_target_can_move(self) -> None:
        sim = DeterministicEventSimulator(
            [2, 1],
            "bubble",
            frozen_positions=[1],
            frozen_variant="passive",
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        outcome = sim.step(forced_cell_id=0, forced_direction=1)
        self.assertTrue(outcome.swapped)
        self.assertEqual(sim.current_values(), [1, 2])
        self.assertEqual(sim.current_frozen_positions(), [0])
        self.assertTrue(sim.value_counts_conserved())

    def test_stuck_frozen_target_blocks_swap(self) -> None:
        sim = DeterministicEventSimulator(
            [2, 1],
            "bubble",
            frozen_positions=[1],
            frozen_variant="stuck",
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        outcome = sim.step(forced_cell_id=0, forced_direction=1)
        self.assertFalse(outcome.swapped)
        self.assertTrue(outcome.blocked_move_attempt)
        self.assertEqual(sim.current_values(), [2, 1])
        self.assertEqual(sim.current_frozen_positions(), [1])

    def test_seed_replay_is_exact(self) -> None:
        kwargs = {
            "initial_values": [6, 2, 4, 1, 5, 3],
            "algotypes": "bubble",
            "scheduler_seed": 123,
            "tie_breaker_seed": 456,
            "condition_id": "seed_replay",
        }
        first = DeterministicEventSimulator(**kwargs).run(max_activations=20000)
        second = DeterministicEventSimulator(**kwargs).run(max_activations=20000)
        self.assertTrue(first.completed)
        self.assertEqual(first.final_values, second.final_values)
        self.assertEqual(first.swap_count, second.swap_count)
        self.assertEqual(first.activation_count, second.activation_count)
        self.assertEqual([row["state_hash"] for row in first.trace_rows], [row["state_hash"] for row in second.trace_rows])

    def test_all_core_algorithms_sort_small_arrays(self) -> None:
        for algorithm in ["bubble", "insertion", "selection"]:
            with self.subTest(algorithm=algorithm):
                result = DeterministicEventSimulator(
                    [5, 1, 4, 2, 3],
                    algorithm,
                    scheduler_seed=10,
                    tie_breaker_seed=20,
                    condition_id=f"small_{algorithm}",
                ).run(max_activations=100000)
                self.assertTrue(result.completed)
                self.assertEqual(result.final_values, [1, 2, 3, 4, 5])
                self.assertEqual(result.final_monotonicity_error, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
