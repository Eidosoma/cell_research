"""E06 S13 final-state model tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e06.final_state_models import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    S13Config,
    audit_predictor_leakage,
    build_feature_catalog,
    build_model_input_matrix,
    build_s06_context_annotations,
    evaluate_models,
    group_splits,
    summarize_performance,
    validate_s13_outputs,
)


def _json(values: list) -> str:
    return json.dumps(values, separators=(",", ":"))


def _sample_s07_mosaic() -> pd.DataFrame:
    rows = []
    classes = ["segregated_low_sort", "mixed_intermediate", "patchy_mosaic"]
    for idx in range(18):
        source_step = ["S02", "S03", "S04", "S06"][idx % 4]
        condition = f"c_{idx // 3}"
        rows.append(
            {
                "source_research_step_id": source_step,
                "run_id": f"run_{idx}",
                "analysis_condition_id": condition,
                "source_condition_id": condition,
                "condition_id": condition,
                "seed": idx,
                "panel": "unit",
                "condition_kind": "pair",
                "candidate_reason": "unit",
                "arrangement": "alternating" if idx % 2 else "random_permutation",
                "orientation": "as_selected",
                "value_profile": "random",
                "perturbation_profile": "none",
                "goal_profile_id": "same_increasing",
                "goal_compatibility_class": "same_goal",
                "policy_ids_json": _json(["a", "b"]),
                "display_names_json": _json(["alpha", "beta"]),
                "source_categories_json": _json(["original", "control"]),
                "ratio_targets_json": _json([0.5, 0.5]),
                "array_size": 20,
                "event_cap": 50,
                "final_state_class": classes[idx % len(classes)],
                "final_target_quality": 0.4 + 0.03 * idx,
                "aggregation_delta_percent": float(idx % 5),
                "goal_conflict_index": 0.0 if idx % 3 else 0.5,
                "largest_block_fraction": 0.2 + 0.02 * idx,
                "base_contest_id": "bc_unit" if source_step == "S06" else "",
            }
        )
    return pd.DataFrame(rows)


def _sample_direct(step: str) -> pd.DataFrame:
    rows = []
    classes = ["segregated_low_sort", "mixed_intermediate", "patchy_mosaic"]
    for idx in range(9):
        rows.append(
            {
                "condition_id": f"{step.lower()}_condition_{idx // 3}",
                "seed": 100 + idx,
                "panel": "unit",
                "candidate_reason": "unit",
                "arrangement": "contiguous_patch" if idx % 2 else "random_permutation",
                "goal_profile_id": "opposite_direction" if idx % 3 == 0 else "same_increasing",
                "goal_compatibility_class": "opposite_goal" if idx % 3 == 0 else "same_goal",
                "policy_ids_json": _json(["a", "b"]),
                "display_names_json": _json(["alpha", "beta"]),
                "source_categories_json": _json(["original", "control"]),
                "ratio_targets_json": _json([0.25, 0.75]),
                "array_size": 20,
                "event_cap": 50,
                "interface_rule_id": "behavior_only",
                "governance_mechanism_id": "no_governance",
                "final_state_class": classes[(idx + 1) % len(classes)],
                "final_target_quality": 0.5 + 0.02 * idx,
                "aggregation_delta_percent": float(idx % 4),
                "goal_conflict_index": 0.5 if idx % 3 == 0 else 0.0,
                "largest_block_fraction": 0.25 + 0.01 * idx,
                "history_id": "simultaneous_mixed_start" if step == "S12" else "",
                "history_family": "simultaneous" if step == "S12" else "",
                "clone_initial_fraction": 0.1 if step == "S11" else None,
                "graft_size": 4 if step == "S10" else None,
            }
        )
    return pd.DataFrame(rows)


class FinalStateModelTests(unittest.TestCase):
    def test_model_input_covers_sources_and_excludes_s05_s07_predictors(self) -> None:
        matrix = build_model_input_matrix(
            _sample_s07_mosaic(),
            {step: _sample_direct(step) for step in ("S08", "S09", "S10", "S11", "S12")},
        )
        self.assertTrue({"S02", "S03", "S04", "S06", "S08", "S09", "S10", "S11", "S12"}.issubset(set(matrix["source_research_step_id"])))
        self.assertNotIn("s07_label", set(NUMERIC_FEATURES + CATEGORICAL_FEATURES))
        self.assertNotIn("s05_quality_synergy", set(NUMERIC_FEATURES + CATEGORICAL_FEATURES))

    def test_leakage_audit_flags_no_allowed_predictor_risks(self) -> None:
        matrix = build_model_input_matrix(_sample_s07_mosaic(), {"S08": _sample_direct("S08")})
        audit = audit_predictor_leakage(matrix, list(NUMERIC_FEATURES + CATEGORICAL_FEATURES))
        risks = audit[(audit["used_as_predictor"]) & (audit["leakage_status"] == "leakage_risk")]
        self.assertTrue(risks.empty)
        targets = audit[audit["column_name"].isin(["final_target_quality", "final_state_class"])]
        self.assertTrue((targets["used_as_predictor"] == False).all())  # noqa: E712

    def test_group_splits_hold_out_condition_groups(self) -> None:
        matrix = build_model_input_matrix(
            _sample_s07_mosaic(),
            {step: _sample_direct(step) for step in ("S08", "S09", "S10", "S11", "S12")},
        )
        config = S13Config(split_count=2, test_fraction=0.30, random_forest_trees=8, workers=1)
        for train_idx, test_idx in group_splits(matrix, config):
            train_groups = set(matrix.iloc[train_idx]["condition_group_id"])
            test_groups = set(matrix.iloc[test_idx]["condition_group_id"])
            self.assertFalse(train_groups & test_groups)

    def test_small_evaluation_and_validation_runs(self) -> None:
        matrix = build_model_input_matrix(
            _sample_s07_mosaic(),
            {step: _sample_direct(step) for step in ("S08", "S09", "S10", "S11", "S12")},
        )
        config = S13Config(split_count=2, test_fraction=0.30, random_forest_trees=8, workers=1, max_permutation_rows=20, permutation_repeats=1)
        feature_catalog = build_feature_catalog()
        leakage = audit_predictor_leakage(matrix, list(NUMERIC_FEATURES + CATEGORICAL_FEATURES))
        performance, predictions, importance = evaluate_models(matrix, config)
        summary = summarize_performance(performance)
        s06_annotations = build_s06_context_annotations(
            matrix,
            pd.DataFrame(
                [
                    {
                        "base_contest_id": "bc_unit",
                        "context_dependent_dominance": True,
                        "dominance_margin_range": 0.5,
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {"policy_id": "a", "net_dominance_score": 1.0, "win_fraction": 0.6},
                    {"policy_id": "b", "net_dominance_score": -1.0, "win_fraction": 0.4},
                ]
            ),
        )
        validation = validate_s13_outputs(
            matrix,
            performance,
            predictions,
            feature_catalog,
            leakage,
            s06_annotations,
            importance.assign(
                claim_scope="exploratory_association_not_causal_proof",
            ),
            model_artifact_count=2,
            figure_written=True,
            unit_tests_success=True,
            s05_context_available=True,
            s07_context_available=True,
            config=config,
        )
        self.assertFalse(summary.empty)
        self.assertFalse(performance.empty)
        self.assertFalse(predictions.empty)
        self.assertTrue(validation["success"].all())


if __name__ == "__main__":
    unittest.main()
