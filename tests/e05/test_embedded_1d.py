"""E05 S06 embedded 1D continuity tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.embedded_1d import (
    build_embedded_state,
    embedded_sorted_row_target,
    execute_row_swap,
    final_metric_summary,
    monotonicity_error_count,
    row_neighbor_swap_allowed,
    row_site_ids,
    row_values,
    run_adjacent_sort,
    sortedness_percent,
    stable_json_sha256,
)
from src.e05.morphospace_metrics import aggregate_morphospace_error, evaluate_morphology_metrics


class Embedded1DTests(unittest.TestCase):
    def test_sortedness_matches_e01_adjacent_pair_definition(self) -> None:
        self.assertEqual(sortedness_percent((1, 2, 3)), 100.0)
        self.assertEqual(sortedness_percent((2, 1, 3)), 100.0 * 2 / 3)
        self.assertEqual(monotonicity_error_count((2, 1, 3)), 1)
        self.assertEqual(
            stable_json_sha256([3, 1, 2]),
            "51bda7ab4e44726cde71fcb6e4b515357059bb6b6dd5146d1fc50f73f11678c6",
        )

    def test_1d_and_embedded_bubble_match_exactly(self) -> None:
        values = (5, 1, 4, 2, 3)
        array_result = run_adjacent_sort(values, algorithm="bubble", substrate_kind="array_1d")
        embedded_result = run_adjacent_sort(values, algorithm="bubble", substrate_kind="embedded_square_grid_2d")

        self.assertEqual(array_result.final_values, tuple(sorted(values)))
        self.assertEqual(embedded_result.final_values, tuple(sorted(values)))
        self.assertEqual(array_result.swap_count, embedded_result.swap_count)
        self.assertEqual(array_result.comparison_count, embedded_result.comparison_count)
        self.assertEqual(
            [row["sortedness_percent"] for row in array_result.records],
            [row["sortedness_percent"] for row in embedded_result.records],
        )

    def test_1d_and_embedded_insertion_match_exactly(self) -> None:
        values = (4, 3, 2, 1)
        array_result = run_adjacent_sort(values, algorithm="insertion", substrate_kind="array_1d")
        embedded_result = run_adjacent_sort(values, algorithm="insertion", substrate_kind="embedded_square_grid_2d")

        self.assertEqual(array_result.swap_count, 6)
        self.assertEqual(embedded_result.swap_count, 6)
        self.assertEqual(array_result.comparison_count, embedded_result.comparison_count)
        self.assertEqual(array_result.final_sortedness_percent, 100.0)
        self.assertEqual(embedded_result.final_sortedness_percent, 100.0)

    def test_row_restriction_rejects_vertical_and_off_row_swaps(self) -> None:
        values = (3, 1, 2)
        target = embedded_sorted_row_target(values, grid_height=3, row_y=1)
        state = build_embedded_state(values, target=target, grid_height=3, row_y=1)
        row_ids = row_site_ids(3, 1, grid_height=3)

        self.assertEqual(row_values(state, row_ids), values)
        self.assertEqual(row_neighbor_swap_allowed(state, row_ids, row_ids[0], row_ids[1]), (True, "allowed"))
        self.assertEqual(row_neighbor_swap_allowed(state, row_ids, row_ids[0], 0), (False, "target_not_in_embedded_row"))
        self.assertEqual(row_neighbor_swap_allowed(state, row_ids, 0, row_ids[0]), (False, "source_not_in_embedded_row"))

        vertical = execute_row_swap(state, row_ids, row_ids[0], 0)
        self.assertFalse(vertical.allowed)
        self.assertFalse(vertical.state_changed)
        self.assertEqual(vertical.energy_cost_charged, 0.0)

    def test_embedded_final_state_has_zero_s05_metrics(self) -> None:
        values = (5, 1, 4, 2, 3)
        result = run_adjacent_sort(values, algorithm="bubble", substrate_kind="embedded_square_grid_2d")
        summary = final_metric_summary(result)
        self.assertEqual(summary["final_row_morphospace_error"], 0.0)
        self.assertEqual(summary["final_embedded_morphospace_error"], 0.0)

        metrics = evaluate_morphology_metrics(result.embedded_target, result.final_state)
        self.assertEqual(aggregate_morphospace_error(metrics), 0.0)


if __name__ == "__main__":
    unittest.main()
