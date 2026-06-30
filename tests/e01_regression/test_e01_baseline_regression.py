"""Small deterministic E01 regression checks.

These tests intentionally use the standard-library unittest runner so the E01
baseline can be checked without adding a test dependency.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import pandas as pd


ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def monotonicity_error_count(values: list[int]) -> int:
    return sum(1 for left, right in zip(values, values[1:]) if left > right)


def sortedness_percent(values: list[int]) -> float:
    if not values:
        return 100.0
    return 100.0 * (len(values) - monotonicity_error_count(values)) / len(values)


def left_neighbor_aggregation_percent(labels: list[str]) -> float:
    if not labels:
        return 0.0
    same_left = sum(1 for index in range(1, len(labels)) if labels[index] == labels[index - 1])
    return 100.0 * same_left / len(labels)


class E01BaselineRegressionTests(unittest.TestCase):
    def test_small_fixture_metrics_are_stable(self) -> None:
        values = [3, 1, 2, 4]
        labels = ["bubble", "bubble", "insertion", "insertion"]
        self.assertEqual(monotonicity_error_count(values), 1)
        self.assertEqual(sortedness_percent(values), 75.0)
        self.assertEqual(left_neighbor_aggregation_percent(labels), 50.0)

    def test_s13_non_replications_are_explicit(self) -> None:
        status = pd.read_csv(ARTIFACTS_DIR / "tables/e01_replication_status.csv")
        non_replicated = set(status.loc[status["classification"] == "not replicated", "claim_id"])
        expected = {
            "figure4_exact_fold_change_magnitudes",
            "figure5_all_cell_view_lower_error_than_traditional",
            "figure8_unique_aggregation_exact_peak_magnitudes",
        }
        self.assertEqual(non_replicated, expected)

    def test_s14_scaled_package_keeps_non_replications(self) -> None:
        noise = pd.read_csv(ARTIFACTS_DIR / "tables/e01_noise_estimates.csv")
        fig4 = noise[noise["claim_family"] == "figure4_efficiency"]
        fig8 = noise[noise["claim_family"] == "figure8_aggregation"]
        fig5 = noise[noise["claim_family"] == "figure5_frozen_cell"]
        unsupported_fig4 = fig4[
            (fig4["paper_inside_scaled_ci95"] == False)  # noqa: E712
            & (fig4["condition_key"] != "bubble_swap_only_steps")
            & (fig4["condition_key"] != "insertion_swap_only_steps")
        ]
        self.assertGreaterEqual(len(unsupported_fig4), 4)
        self.assertTrue((fig8.loc[~fig8["condition_key"].str.contains("control"), "paper_inside_scaled_ci95"] == False).all())  # noqa: E712
        self.assertFalse((fig5["stability_result"] == "cell_lower_supported").all())

    def test_s15_core_results_and_schema_exist(self) -> None:
        core = pd.read_parquet(ARTIFACTS_DIR / "results/e01_core_results.parquet")
        self.assertGreaterEqual(len(core), 80)
        self.assertIn("figure4_exact_fold_change_magnitudes", set(core["claim_id"].dropna()))
        self.assertIn("figure8_unique_aggregation_exact_peak_magnitudes", set(core["claim_id"].dropna()))

        schema_path = ARTIFACTS_DIR / "results/e01_trace_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        families = {entry["traceFamily"] for entry in schema["traceArtifacts"]}
        self.assertGreaterEqual(
            families,
            {
                "figure3_unperturbed",
                "frozen_cell",
                "same_goal_chimeras",
                "duplicate_value_chimeras",
                "opposite_direction_chimeras",
            },
        )
        for entry in schema["traceArtifacts"]:
            self.assertGreater(entry["rowCount"], 0)
            column_names = {col["name"] for col in entry["columns"]}
            if entry["traceFamily"] == "s14_scaled_aggregation_curves":
                self.assertIn("progress_percent", column_names)
                self.assertIn("mean_aggregation_left_neighbor_percent", column_names)
            else:
                self.assertIn("research_step_id", column_names)


if __name__ == "__main__":
    unittest.main()
