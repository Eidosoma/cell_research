"""S04 competence-vector metric tests."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.e03.classic_policies import classic_policy_library
from src.e03.competence_metrics import (
    CompetenceSourcePaths,
    build_policy_competence_vectors,
    normalize_higher_better,
    normalize_lower_better,
    validate_competence_vectors,
    vector_column_spec,
)


def _write_policy_library(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(classic_policy_library(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_fixture_inputs(root: Path) -> tuple[Path, Path, Path]:
    policy_path = root / "policies/e03_classic_policy_library.json"
    e01 = root / "E01"
    e02 = root / "E02"
    (e01 / "tables").mkdir(parents=True)
    (e01 / "results").mkdir(parents=True)
    (e02 / "research_steps/S09").mkdir(parents=True)
    _write_policy_library(policy_path)

    pd.DataFrame(
        [
            {"algorithm": "bubble", "count_metric": "swap_only_steps", "cell_view_mean": 10.0},
            {"algorithm": "insertion", "count_metric": "swap_only_steps", "cell_view_mean": 10.0},
            {"algorithm": "selection", "count_metric": "swap_only_steps", "cell_view_mean": 5.0},
            {"algorithm": "bubble", "count_metric": "comparison_steps_observed", "cell_view_mean": 20.0},
            {"algorithm": "insertion", "count_metric": "comparison_steps_observed", "cell_view_mean": 15.0},
            {"algorithm": "selection", "count_metric": "comparison_steps_observed", "cell_view_mean": 40.0},
            {"algorithm": "bubble", "count_metric": "compare_plus_swap_steps", "cell_view_mean": 30.0},
            {"algorithm": "insertion", "count_metric": "compare_plus_swap_steps", "cell_view_mean": 25.0},
            {"algorithm": "selection", "count_metric": "compare_plus_swap_steps", "cell_view_mean": 45.0},
        ]
    ).to_csv(e01 / "tables/e01_efficiency_numeric_table.csv", index=False)
    pd.DataFrame(
        [
            {"mode": "cell_view", "algorithm": algorithm, "repeat_index": repeat, "final_sortedness_percent": 100.0}
            for algorithm in ("bubble", "insertion", "selection")
            for repeat in range(2)
        ]
    ).to_parquet(e01 / "results/e01_efficiency_counts.parquet", index=False)
    pd.DataFrame(
        [
            {"condition_id": "b1", "mode": "cell_view", "algorithm": "bubble", "mean_final_sortedness_percent": 98.0, "mean_final_monotonicity_error": 2.0},
            {"condition_id": "b2", "mode": "cell_view", "algorithm": "bubble", "mean_final_sortedness_percent": 98.0, "mean_final_monotonicity_error": 2.0},
            {"condition_id": "i1", "mode": "cell_view", "algorithm": "insertion", "mean_final_sortedness_percent": 97.0, "mean_final_monotonicity_error": 3.0},
            {"condition_id": "i2", "mode": "cell_view", "algorithm": "insertion", "mean_final_sortedness_percent": 97.0, "mean_final_monotonicity_error": 3.0},
            {"condition_id": "s1", "mode": "cell_view", "algorithm": "selection", "mean_final_sortedness_percent": 99.0, "mean_final_monotonicity_error": 1.0},
            {"condition_id": "s2", "mode": "cell_view", "algorithm": "selection", "mean_final_sortedness_percent": 99.0, "mean_final_monotonicity_error": 1.0},
        ]
    ).to_csv(e01 / "tables/e01_frozen_cell_robustness_numeric_table.csv", index=False)
    pd.DataFrame(
        [
            {"mode": "cell_view", "algorithm": "bubble", "frozen_semantics": "none", "frozen_count": 0, "mean_dg_primary": 0.4, "mean_dg_total_ratio": 0.1, "mean_dg_event_count": 30.0},
            {"mode": "cell_view", "algorithm": "insertion", "frozen_semantics": "none", "frozen_count": 0, "mean_dg_primary": 1.0, "mean_dg_total_ratio": 1.0, "mean_dg_event_count": 20.0},
            {"mode": "cell_view", "algorithm": "selection", "frozen_semantics": "none", "frozen_count": 0, "mean_dg_primary": 5.0, "mean_dg_total_ratio": 4.0, "mean_dg_event_count": 10.0},
        ]
    ).to_csv(e01 / "tables/e01_dg_numeric_table.csv", index=False)
    pd.DataFrame(
        [
            {"algotype_mix": "same_goal_bubble_insertion", "is_negative_control": False, "peak_minus_expected_random_left_neighbor_percent": 10.0, "mean_curve_peak_aggregation_left_neighbor_percent": 60.0},
            {"algotype_mix": "same_goal_bubble_selection", "is_negative_control": False, "peak_minus_expected_random_left_neighbor_percent": 8.0, "mean_curve_peak_aggregation_left_neighbor_percent": 58.0},
            {"algotype_mix": "same_goal_insertion_selection", "is_negative_control": False, "peak_minus_expected_random_left_neighbor_percent": 12.0, "mean_curve_peak_aggregation_left_neighbor_percent": 62.0},
            {"algotype_mix": "same_goal_bubble_insertion_selection", "is_negative_control": False, "peak_minus_expected_random_left_neighbor_percent": 15.0, "mean_curve_peak_aggregation_left_neighbor_percent": 55.0},
            {"algotype_mix": "same_algorithm_bubble_label_control", "is_negative_control": True, "peak_minus_expected_random_left_neighbor_percent": 1.0, "mean_curve_peak_aggregation_left_neighbor_percent": 50.0},
        ]
    ).to_csv(e01 / "tables/e01_aggregation_peak_table.csv", index=False)
    pd.DataFrame(
        [
            {"algotype_mix": "opposite_unique_bubble_down_selection_up", "component_algorithms_json": '["bubble","selection"]', "dominant_label_counts_json": '{"bubble":100}', "repetitions_observed": 100},
            {"algotype_mix": "opposite_unique_bubble_up_insertion_down", "component_algorithms_json": '["bubble","insertion"]', "dominant_label_counts_json": '{"bubble":100}', "repetitions_observed": 100},
            {"algotype_mix": "opposite_unique_insertion_up_selection_down", "component_algorithms_json": '["insertion","selection"]', "dominant_label_counts_json": '{"selection":100}', "repetitions_observed": 100},
            {"algotype_mix": "opposite_duplicate_bubble_down_selection_up", "component_algorithms_json": '["bubble","selection"]', "dominant_label_counts_json": '{"bubble":90,"selection":10}', "repetitions_observed": 100},
            {"algotype_mix": "opposite_duplicate_bubble_up_insertion_down", "component_algorithms_json": '["bubble","insertion"]', "dominant_label_counts_json": '{"bubble":100}', "repetitions_observed": 100},
            {"algotype_mix": "opposite_duplicate_insertion_up_selection_down", "component_algorithms_json": '["insertion","selection"]', "dominant_label_counts_json": '{"insertion":40,"selection":39,"tie":21}', "repetitions_observed": 100},
        ]
    ).to_csv(e01 / "tables/e01_conflict_equilibria_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "task_family": "efficiency",
                "mode": "cell_view",
                "algorithm": algorithm,
                "frozen_semantics": "none",
                "frozen_count": 0,
                "array_length": 100,
                "repeat_index": repeat,
                "final_sortedness_percent": 100.0,
            }
            for algorithm in ("bubble", "insertion", "selection")
            for repeat in range(2)
        ]
    ).to_parquet(e01 / "results/e01_scaled_replication.parquet", index=False)
    pd.DataFrame(
        [
            {"run_family": "unperturbed_baseline", "metric_name": "sortedness_adjacency_distance", "algotype_mix": "bubble", "mean_path_curvature_ratio": 25.0, "n_runs": 2},
            {"run_family": "unperturbed_baseline", "metric_name": "sortedness_adjacency_distance", "algotype_mix": "insertion", "mean_path_curvature_ratio": 3.0, "n_runs": 2},
            {"run_family": "unperturbed_baseline", "metric_name": "sortedness_adjacency_distance", "algotype_mix": "selection", "mean_path_curvature_ratio": 2.0, "n_runs": 2},
        ]
    ).to_csv(e02 / "research_steps/S09/e02_alternative_metric_summary.csv", index=False)
    return policy_path, e01, e02


class CompetenceMetricTests(unittest.TestCase):
    def test_normalization_helpers_handle_ties_and_missing_values(self) -> None:
        self.assertEqual(normalize_lower_better([5.0, 5.0]), [1.0, 1.0])
        self.assertEqual(normalize_higher_better([5.0, 10.0, 15.0]), [0.0, 0.5, 1.0])
        lower = normalize_lower_better([1.0, math.nan, 3.0])
        self.assertEqual(lower[0], 1.0)
        self.assertTrue(math.isnan(lower[1]))
        self.assertEqual(lower[2], 0.0)

    def test_build_vectors_and_validation_from_fixture_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path, e01, e02 = _write_fixture_inputs(Path(tmpdir))
            vectors = build_policy_competence_vectors(CompetenceSourcePaths(policy_path, e01, e02))
            self.assertEqual(len(vectors), 9)
            self.assertEqual(int(vectors["canonical_classic_baseline"].sum()), 3)
            canonical = vectors[vectors["canonical_classic_baseline"] == True]  # noqa: E712
            self.assertEqual(
                canonical.sort_values("compare_plus_swap_steps")["algorithm"].tolist(),
                ["insertion", "bubble", "selection"],
            )
            validation = validate_competence_vectors(vectors)
            self.assertTrue(validation["success"].all(), validation.to_string(index=False))

    def test_vector_spec_names_required_dimensions(self) -> None:
        names = {entry["column"] for entry in vector_column_spec()}
        for column in {
            "efficiency_score",
            "final_sortedness_score",
            "frozen_cell_robustness_score",
            "dg_tendency_score",
            "aggregation_tendency_score",
            "conflict_dominance_score",
            "movement_energy_score",
            "path_directness_score",
            "transfer_across_array_sizes_score",
        }:
            self.assertIn(column, names)


if __name__ == "__main__":
    unittest.main()
