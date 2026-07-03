"""E06 S05 compatibility metric tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e06.compatibility_metrics import (
    S05Config,
    build_control_scores,
    build_pure_baselines,
    score_source_summary,
    summarize_dimensions,
    validate_s05_outputs,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "algotype_id": "a",
                "display_name": "alpha",
                "source_category": "original",
                "baseline_competence_score": 0.90,
                "mean_final_inversion_sortedness": 0.90,
                "mean_work_count": 10.0,
                "mean_swap_count": 5.0,
                "validation_run_count": 2,
                "invalid_run_count": 0,
                "pure_execution_success": True,
            },
            {
                "algotype_id": "b",
                "display_name": "beta",
                "source_category": "control",
                "baseline_competence_score": 0.50,
                "mean_final_inversion_sortedness": 0.50,
                "mean_work_count": 2.0,
                "mean_swap_count": 1.0,
                "validation_run_count": 2,
                "invalid_run_count": 0,
                "pure_execution_success": True,
            },
        ]
    )


def _validation_runs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"algotype_id": "a", "final_inversion_sortedness": 0.90, "work_count": 10.0, "array_size": 8, "invalid": False},
            {"algotype_id": "a", "final_inversion_sortedness": 0.90, "work_count": 10.0, "array_size": 8, "invalid": False},
            {"algotype_id": "b", "final_inversion_sortedness": 0.50, "work_count": 2.0, "array_size": 8, "invalid": False},
            {"algotype_id": "b", "final_inversion_sortedness": 0.50, "work_count": 2.0, "array_size": 8, "invalid": False},
        ]
    )


class CompatibilityMetricTests(unittest.TestCase):
    def test_weighted_baseline_and_synergy_use_ratios(self) -> None:
        baselines = build_pure_baselines(_metadata(), _validation_runs())
        summary = pd.DataFrame(
            [
                {
                    "research_step_id": "S02",
                    "condition_id": "unit",
                    "condition_kind": "pair",
                    "panel": "unit",
                    "policy_ids_json": json.dumps(["a", "b"], separators=(",", ":")),
                    "display_names_json": json.dumps(["alpha", "beta"], separators=(",", ":")),
                    "source_categories_json": json.dumps(["original", "control"], separators=(",", ":")),
                    "ratio_targets_json": json.dumps([0.25, 0.75], separators=(",", ":")),
                    "seed_count": 2,
                    "run_count": 2,
                    "mean_final_inversion_sortedness": 0.65,
                    "mean_aggregation_delta_percent": 0.0,
                    "mean_interface_count": 10.0,
                    "mean_largest_block_fraction": 0.4,
                    "mean_position_bias_margin": 0.0,
                    "mean_work_count": 100.0,
                    "invalid_action_count": 0,
                    "final_state_class_mode": "unit",
                }
            ]
        )
        scores = score_source_summary(summary, baselines, S05Config(), source_step="S02")
        row = scores.iloc[0]
        self.assertAlmostEqual(row["expected_pure_quality"], 0.60)
        self.assertAlmostEqual(row["quality_synergy"], 0.05)
        self.assertEqual(row["source_research_step_id"], "S02")

    def test_pure_and_dummy_controls_center_synergy(self) -> None:
        baselines = build_pure_baselines(_metadata(), _validation_runs())
        controls = build_control_scores(baselines, S05Config(), max_dummy_controls=2)
        pure = controls[controls["control_type"] == "pure_policy_baseline_control"]
        dummy = controls[controls["control_type"] == "dummy_label_same_policy_control"]
        self.assertEqual(len(pure), 2)
        self.assertEqual(len(dummy), 2)
        self.assertLessEqual(pure["quality_synergy"].abs().max(), 1e-12)
        self.assertLessEqual(dummy["quality_synergy"].abs().max(), 1e-12)
        self.assertEqual(int(dummy["metric_disagreement_count"].max()), 0)
        self.assertEqual(set(controls["compatibility_pattern_class"]), {"pure_or_dummy_label_control"})

    def test_goal_conflict_is_kept_separate_from_quality(self) -> None:
        baselines = build_pure_baselines(_metadata(), _validation_runs())
        summary = pd.DataFrame(
            [
                {
                    "research_step_id": "S04",
                    "s03_candidate_id": "unit",
                    "panel": "unit",
                    "candidate_reason": "unit",
                    "arrangement": "contiguous_patch",
                    "goal_profile_id": "opposite_direction",
                    "goal_compatibility_class": "opposite_goal",
                    "policy_ids_json": json.dumps(["a", "b"], separators=(",", ":")),
                    "display_names_json": json.dumps(["alpha", "beta"], separators=(",", ":")),
                    "ratio_targets_json": json.dumps([0.5, 0.5], separators=(",", ":")),
                    "goal_assignments_json": json.dumps({"a": "increasing", "b": "decreasing"}, separators=(",", ":")),
                    "seed_count": 2,
                    "run_count": 2,
                    "mean_reference_increasing_score": 0.40,
                    "mean_assigned_policy_sortedness": 0.85,
                    "mean_shared_goal_score": float("nan"),
                    "mean_best_goal_score": 0.90,
                    "mean_goal_alignment_gap": 0.50,
                    "mean_final_inversion_sortedness": 0.40,
                    "mean_aggregation_delta_percent": 12.0,
                    "mean_interface_count": 5.0,
                    "mean_work_count": 100.0,
                    "invalid_action_count": 0,
                    "goal_state_class_mode": "unit",
                }
            ]
        )
        scores = score_source_summary(summary, baselines, S05Config(), source_step="S04")
        row = scores.iloc[0]
        conflicts = json.loads(row["conflicting_dimensions_json"])
        self.assertEqual(row["goal_conflict_index"], 0.50)
        self.assertIn("goal_alignment_conflict", conflicts)
        self.assertEqual(row["compatibility_pattern_class"], "goal_tension_local_success")
        self.assertNotIn("compatibility_score", scores.columns)

    def test_validation_checks_expected_counts_and_controls(self) -> None:
        baselines = build_pure_baselines(_metadata(), _validation_runs())
        summary = pd.DataFrame(
            [
                {
                    "research_step_id": "S02",
                    "condition_id": "unit",
                    "condition_kind": "pair",
                    "panel": "unit",
                    "policy_ids_json": json.dumps(["a", "b"], separators=(",", ":")),
                    "display_names_json": json.dumps(["alpha", "beta"], separators=(",", ":")),
                    "source_categories_json": json.dumps(["original", "control"], separators=(",", ":")),
                    "ratio_targets_json": json.dumps([0.5, 0.5], separators=(",", ":")),
                    "seed_count": 2,
                    "run_count": 2,
                    "mean_final_inversion_sortedness": 0.90,
                    "mean_aggregation_delta_percent": 20.0,
                    "mean_interface_count": 2.0,
                    "mean_largest_block_fraction": 0.9,
                    "mean_position_bias_margin": 0.0,
                    "mean_work_count": 100.0,
                    "invalid_action_count": 0,
                    "final_state_class_mode": "unit",
                }
            ]
        )
        scores = score_source_summary(summary, baselines, S05Config(), source_step="S02")
        controls = build_control_scores(baselines, S05Config(), max_dummy_controls=1)
        validation = validate_s05_outputs(
            scores,
            baselines,
            controls,
            expected_source_counts={"S02": 1},
            unit_tests_success=True,
        )
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertTrue(cases["source_row_counts_match_inputs"])
        self.assertTrue(cases["pure_policy_controls_center_synergy"])
        self.assertTrue(cases["dummy_label_controls_do_not_create_interference"])
        self.assertTrue(cases["conflicting_dimensions_kept_separate"])
        self.assertFalse(summarize_dimensions(scores).empty)


if __name__ == "__main__":
    unittest.main()
