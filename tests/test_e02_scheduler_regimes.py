from __future__ import annotations

import unittest

from scripts.e02_s02_scheduler_regimes import (
    algotypes_from_seed,
    run_deterministic_scheduler,
    scheduler_order,
)
from e02_deterministic_simulator import DeterministicEventSimulator


class TestE02SchedulerRegimes(unittest.TestCase):
    def test_algotype_assignment_counts_are_scaled_and_seeded(self) -> None:
        first = algotypes_from_seed("bubble_insertion", 7, 123)
        second = algotypes_from_seed("bubble_insertion", 7, 123)
        self.assertEqual(first, second)
        self.assertEqual(first.count("bubble"), 3)
        self.assertEqual(first.count("insertion"), 4)

    def test_left_and_right_orders_follow_current_positions(self) -> None:
        sim = DeterministicEventSimulator([3, 2, 1], "bubble", scheduler_seed=1, tie_breaker_seed=1)
        self.assertEqual(scheduler_order(sim, "left_to_right", sim.scheduler_rng), [0, 1, 2])
        self.assertEqual(scheduler_order(sim, "right_to_left", sim.scheduler_rng), [2, 1, 0])
        sim.step(forced_cell_id=0, forced_direction=1)
        self.assertEqual([cell.cell_id for cell in sim.cells], [1, 0, 2])
        self.assertEqual(scheduler_order(sim, "left_to_right", sim.scheduler_rng), [1, 0, 2])

    def test_round_schedulers_sort_small_pure_bubble_case(self) -> None:
        for regime in [
            "s01_random_sequential",
            "synchronous_rounds",
            "priority_queue",
            "adversarial_order",
            "left_to_right",
            "right_to_left",
        ]:
            with self.subTest(regime=regime):
                result = run_deterministic_scheduler(
                    regime=regime,
                    initial_values=[4, 1, 3, 2],
                    initial_algotypes=["bubble"] * 4,
                    condition_id=f"test_{regime}",
                    scheduler_seed=10,
                    tie_breaker_seed=20,
                    max_activations=5000,
                    max_swaps=1000,
                    max_comparisons=10000,
                    no_move_checks_required=2,
                )
                self.assertTrue(result.completed)
                self.assertEqual(result.final_values, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
