from __future__ import annotations

import math
import unittest

from e02_deterministic_simulator import aggregation
from morphospace2d import (
    EMBEDDED_1D_SCHEMA_VERSION,
    delayed_gratification_from_sortedness,
    run_embedded_row,
)
from scripts.e01_s04_reproduce_figure03 import run_cell_view


class TestE05Embedded1D(unittest.TestCase):
    def test_embedded_row_uses_2d_substrate_and_stays_on_row(self) -> None:
        result = run_embedded_row(
            "bubble",
            [4, 1, 3, 2],
            scheduler_seed=123,
            tie_breaker_seed=456,
            row_y=1,
            height=3,
            max_swaps=1000,
        )
        self.assertEqual(result.schema_version, EMBEDDED_1D_SCHEMA_VERSION)
        self.assertEqual(result.substrate.substrate_type, "square_grid_2d")
        self.assertEqual(result.substrate.metadata["height"], 3)
        self.assertTrue(result.completed)
        self.assertTrue(result.row_restricted)
        self.assertTrue(all(row["row_restricted"] for row in result.trace_rows))
        self.assertEqual(result.final_values, [1, 2, 3, 4])
        self.assertEqual(result.final_monotonicity_error, 0)

    def test_classic_algorithms_match_e01_cell_view_small_reference(self) -> None:
        values = [6, 2, 4, 1, 5, 3]
        for algorithm in ["bubble", "insertion", "selection"]:
            with self.subTest(algorithm=algorithm):
                embedded = run_embedded_row(
                    algorithm,
                    values,
                    scheduler_seed=321,
                    tie_breaker_seed=654,
                    row_y=1,
                    height=3,
                    max_swaps=10000,
                )
                reference = run_cell_view(
                    algorithm,
                    list(values),
                    scheduler_seed=321,
                    tie_breaker_seed=654,
                    max_swaps=10000,
                )
                self.assertEqual(embedded.final_values, reference.states[-1])
                self.assertEqual(embedded.swap_count, reference.final_swap_count)
                self.assertEqual(embedded.comparison_count, reference.final_comparison_count)
                self.assertEqual(
                    embedded.archived_compare_and_swap_count,
                    reference.final_archived_compare_and_swap_count,
                )
                self.assertEqual(embedded.event_count, len(reference.states))
                self.assertEqual(embedded.scheduler_rounds, reference.scheduler_rounds)
                self.assertTrue(embedded.row_restricted)

    def test_delayed_gratification_formula_matches_e01_contract_cases(self) -> None:
        self.assertEqual(delayed_gratification_from_sortedness([])["delayedGratification"], 0.0)
        self.assertEqual(delayed_gratification_from_sortedness([40.0, 50.0, 60.0])["dgEventCount"], 0)
        result = delayed_gratification_from_sortedness([50.0, 40.0, 70.0])
        self.assertEqual(result["dgEventCount"], 1)
        self.assertTrue(math.isclose(result["delayedGratification"], 2.0))

    def test_aggregation_contract_for_pure_and_mixed_rows(self) -> None:
        self.assertEqual(aggregation(["bubble"] * 4), 1.0)
        self.assertEqual(aggregation(["bubble", "insertion", "bubble", "insertion"]), 0.0)
        self.assertAlmostEqual(aggregation(["bubble", "bubble", "insertion", "insertion"]), 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
