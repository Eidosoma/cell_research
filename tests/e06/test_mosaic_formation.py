"""E06 S07 mosaic formation tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e06.mosaic_formation import (
    S07Config,
    add_continuum_coordinates,
    add_rule_labels,
    attach_explanatory_contexts,
    classification_mode,
    classify_mosaic_label,
    condition_seed_stability,
    harmonize_all_runs,
    metric_continua_summary,
    run_classifier_settings,
    select_exemplars,
    summarize_classes,
    validate_s07_outputs,
)


def _json(values: list) -> str:
    return json.dumps(values, separators=(",", ":"))


def _source_row(step: str, condition_id: str, seed: int, *, target: float, aggregation: float, goal_gap: float = 0.0) -> dict:
    row = {
        "schema": f"unit.{step}",
        "experiment_id": "E06",
        "research_step_id": step,
        "condition_id": condition_id,
        "condition_kind": "pair",
        "panel": "unit",
        "arrangement": "random_permutation",
        "array_size": 20,
        "event_cap": 40,
        "events_executed": 40,
        "seed": seed,
        "policy_ids_json": _json(["a", "b"]),
        "display_names_json": _json(["alpha", "beta"]),
        "source_categories_json": _json(["original", "control"]),
        "ratio_targets_json": _json([0.5, 0.5]),
        "final_inversion_sortedness": target,
        "initial_inversion_sortedness": 0.5,
        "inversion_sortedness_delta": target - 0.5,
        "aggregation_delta_percent": aggregation,
        "interface_count": 4,
        "contiguous_run_count": 5,
        "largest_block_fraction": 0.5,
        "label_entropy": 0.9,
        "position_bias_margin": 0.1,
        "work_count": 20,
        "swap_count": 5,
        "final_state_class": "unit",
        "final_labels_head_json": _json(["a", "b"] * 5),
        "final_labels_tail_json": _json(["b", "a"] * 5),
        "final_values_head_json": _json(list(range(1, 11))),
        "final_values_tail_json": _json(list(range(11, 21))),
    }
    if step in {"S04", "S06"}:
        row["goal_profile_id"] = "opposite_direction"
        row["goal_compatibility_class"] = "opposite_goal"
        row["assigned_policy_mean_sortedness"] = target
        row["reference_increasing_score"] = 1.0 - goal_gap
        row["goal_alignment_gap"] = goal_gap
        row["goal_state_class"] = "unit_goal"
    else:
        row["goal_profile_id"] = "same_increasing"
        row["goal_compatibility_class"] = "same_goal"
    if step == "S06":
        row["base_contest_id"] = "base1"
        row["orientation"] = "as_selected"
        row["value_profile"] = "random_permutation"
        row["perturbation_profile"] = "none"
        row["dominant_policy_id"] = "a"
        row["dominated_policy_id"] = "b"
        row["dominance_margin_abs"] = 0.3
    return row


class MosaicFormationTests(unittest.TestCase):
    def test_rule_labels_cover_goal_conflict_and_integrated_cases(self) -> None:
        self.assertEqual(
            classify_mosaic_label(
                {
                    "final_target_quality": 0.9,
                    "goal_conflict_index": 0.6,
                    "aggregation_delta_percent": 0.0,
                    "largest_block_fraction": 0.2,
                    "label_entropy": 1.0,
                    "position_bias_abs": 0.0,
                    "interface_density": 0.5,
                    "work_rate": 0.5,
                    "frozen_block_rate": 0.0,
                    "ratio_max_fraction": 0.5,
                }
            ),
            "polarized_goal_conflict",
        )
        self.assertEqual(
            classify_mosaic_label(
                {
                    "final_target_quality": 0.95,
                    "goal_conflict_index": 0.0,
                    "aggregation_delta_percent": 1.0,
                    "largest_block_fraction": 0.2,
                    "label_entropy": 1.0,
                    "position_bias_abs": 0.0,
                    "interface_density": 0.5,
                    "work_rate": 0.5,
                    "frozen_block_rate": 0.0,
                    "ratio_max_fraction": 0.5,
                }
            ),
            "integrated_high_quality",
        )

    def test_harmonize_and_attach_contexts(self) -> None:
        frames = {
            "S02": pd.DataFrame([_source_row("S02", "c1", 1, target=0.9, aggregation=1.0)]),
            "S06": pd.DataFrame([_source_row("S06", "c2", 1, target=0.8, aggregation=20.0, goal_gap=0.6)]),
        }
        runs = harmonize_all_runs(frames)
        scores = pd.DataFrame(
            [
                {
                    "source_research_step_id": "S02",
                    "policy_ids_json": _json(["a", "b"]),
                    "ratio_targets_json": _json([0.5, 0.5]),
                    "arrangement": "random_permutation",
                    "goal_profile_id": "same_increasing",
                    "score_id": "s05_unit",
                    "primary_target_quality": 0.9,
                    "expected_pure_quality": 0.8,
                    "quality_synergy": 0.1,
                    "work_interference_log_ratio": 0.0,
                    "cooperative_efficiency_score": 0.7,
                    "integration_score": 0.9,
                    "interface_stability_score": 0.8,
                    "dominance_abs_margin": 0.1,
                    "goal_conflict_index": 0.0,
                    "metric_disagreement_count": 0,
                    "conflicting_dimensions_json": "[]",
                    "compatibility_pattern_class": "unit",
                }
            ]
        )
        hierarchy = pd.DataFrame(
            [
                {"policy_id": "a", "display_name": "alpha", "net_dominance_score": 2.0, "win_fraction": 0.7},
                {"policy_id": "b", "display_name": "beta", "net_dominance_score": -1.0, "win_fraction": 0.3},
            ]
        )
        context = pd.DataFrame(
            [
                {
                    "base_contest_id": "base1",
                    "context_dependent_dominance": True,
                    "dominance_margin_range": 0.5,
                }
            ]
        )
        enriched = attach_explanatory_contexts(runs, scores, hierarchy, context)
        self.assertEqual(enriched.loc[enriched["source_research_step_id"] == "S02", "s05_score_id"].iloc[0], "s05_unit")
        self.assertTrue(enriched.loc[enriched["source_research_step_id"] == "S06", "s06_context_context_dependent_dominance"].iloc[0])
        self.assertEqual(enriched.loc[0, "s06_hierarchy_top_policy_id"], "a")

    def test_classifier_stability_and_continuum_fallback(self) -> None:
        frames = {"S02": pd.DataFrame([_source_row("S02", "c1", seed, target=0.6 + 0.1 * (seed % 2), aggregation=seed) for seed in range(1, 7)])}
        runs = add_rule_labels(harmonize_all_runs(frames))
        runs, _pca = add_continuum_coordinates(runs)
        config = S07Config(seed_stability_threshold=0.75, classifier_stability_threshold=0.99, classifier_k_values=(2,), classifier_random_states=(1,))
        clustered, settings, pairwise, classifier_summary = run_classifier_settings(runs, config)
        condition_rows, seed_summary = condition_seed_stability(clustered, config)
        mode = classification_mode(seed_summary, classifier_summary)
        self.assertIn(mode, {"stable_discrete_taxonomy", "metric_continuum_with_exemplars"})
        self.assertFalse(settings.empty)
        self.assertFalse(condition_rows.empty)
        if not bool(seed_summary["seed_labels_stable"]) or not bool(classifier_summary["classifier_settings_stable"]):
            self.assertEqual(mode, "metric_continuum_with_exemplars")
        self.assertIsInstance(pairwise, pd.DataFrame)

    def test_validation_accepts_continuum_fallback(self) -> None:
        frames = {"S02": pd.DataFrame([_source_row("S02", f"c{seed}", seed, target=0.9, aggregation=1.0) for seed in range(1, 5)])}
        mosaic = add_rule_labels(harmonize_all_runs(frames))
        mosaic, _pca = add_continuum_coordinates(mosaic)
        mosaic["s07_label"] = mosaic["s07_rule_label"]
        mosaic["s07_classifier_consensus_label"] = mosaic["s07_label"]
        mosaic["s07_classifier_consensus_fraction"] = 1.0
        mosaic["s05_score_id"] = "s05_unit"
        mosaic["s06_context_context_dependent_dominance"] = pd.NA
        class_summary = summarize_classes(mosaic)
        continua = metric_continua_summary(mosaic)
        condition_rows, _seed_summary = condition_seed_stability(mosaic, S07Config())
        validation = validate_s07_outputs(
            mosaic,
            {"S02": len(mosaic)},
            class_summary,
            continua,
            condition_rows,
            pd.DataFrame({"setting_id": ["unit"]}),
            pd.DataFrame({"left_setting_id": ["a"], "right_setting_id": ["b"], "semantic_label_agreement": [1.0]}),
            select_exemplars(mosaic, S07Config()),
            {
                "classification_mode": "metric_continuum_with_exemplars",
                "seed_labels_stable": False,
                "classifier_settings_stable": False,
                "stable_condition_fraction": 0.0,
                "mean_pairwise_semantic_agreement": 0.0,
            },
            S07Config(),
            figure_written=True,
            unit_tests_success=True,
        )
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertTrue(cases["discrete_labels_stable_or_continuum_fallback"])
        self.assertTrue(cases["metric_continua_written"])


if __name__ == "__main__":
    unittest.main()
