"""E06 S14 intervention-search tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e06.governance_mechanisms import DEFAULT_GOVERNANCE_MECHANISMS
from src.e06.interface_rules import DEFAULT_INTERFACE_RULES
from src.e06.intervention_search import (
    BASELINE_SPEC_ID,
    DEFAULT_INTERVENTION_SPECS,
    SEARCH_STAGE,
    VALIDATION_STAGE,
    InterventionSpec,
    S14Config,
    compare_to_baseline,
    intervention_spec_table,
    rank_interventions,
    run_s14_search,
    select_s14_failure_contexts,
    summarize_s14_runs,
    validate_s14_outputs,
)


def _json(values: list) -> str:
    return json.dumps(values, separators=(",", ":"))


def _records() -> list[dict]:
    return [
        {
            "algotypeId": "orig_bubble",
            "sourceCategory": "original",
            "displayName": "original_bubble",
            "executionBackend": "e02_public_cell_simulator",
            "algorithm": "bubble",
        },
        {
            "algotypeId": "orig_insertion",
            "sourceCategory": "original",
            "displayName": "original_insertion",
            "executionBackend": "e02_public_cell_simulator",
            "algorithm": "insertion",
        },
        {
            "algotypeId": "orig_selection",
            "sourceCategory": "original",
            "displayName": "original_selection",
            "executionBackend": "e02_public_cell_simulator",
            "algorithm": "selection",
        },
    ]


def _matrix() -> pd.DataFrame:
    rows = []
    specs = [
        ("S06", "ctx_a", ["orig_insertion", "orig_selection"], ["original_insertion", "original_selection"], "75:25", "gradient", "s06_opposite_direction", 0.25, 0.95),
        ("S09", "ctx_b", ["orig_bubble", "orig_selection"], ["original_bubble", "original_selection"], "50:50", "contiguous_patch", "opposite_direction", 0.40, 0.70),
        ("S12", "ctx_c", ["orig_bubble", "orig_insertion"], ["original_bubble", "original_insertion"], "50:50", "random_permutation", "same_increasing", 0.52, 0.10),
    ]
    for idx, (step, context, policy_ids, displays, ratio, arrangement, goal, quality, conflict) in enumerate(specs):
        for seed in (101, 102):
            rows.append(
                {
                    "schema": "unit",
                    "prediction_row_id": f"pred_{idx}_{seed}",
                    "source_research_step_id": step,
                    "source_condition_id": context,
                    "source_run_condition_id": context,
                    "condition_group_id": f"{step}:{context}",
                    "seed": seed,
                    "policy_count": 2.0,
                    "policy_signature": _json(policy_ids),
                    "display_signature": _json(displays),
                    "source_category_signature": _json(["original", "original"]),
                    "ratio_label": ratio,
                    "arrangement": arrangement,
                    "goal_profile_id": goal,
                    "goal_compatibility_class": "opposite_goal" if "opposite" in goal else "same_goal",
                    "value_profile": "random_permutation",
                    "perturbation_profile": "none",
                    "panel": "unit",
                    "candidate_reason": "unit_failure",
                    "final_state_class": "segregated_low_sort",
                    "final_target_quality": quality,
                    "goal_conflict_index": conflict,
                }
            )
    return pd.DataFrame(rows)


def _predictions(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw in matrix.to_dict(orient="records"):
        rows.append(
            {
                "prediction_row_id": raw["prediction_row_id"],
                "target_name": "final_state_class",
                "target_type": "classification",
                "model_id": "random_forest",
                "observed_value": raw["final_state_class"],
                "predicted_value": "mixed_intermediate",
                "observed_numeric": None,
                "predicted_numeric": None,
            }
        )
        rows.append(
            {
                "prediction_row_id": raw["prediction_row_id"],
                "target_name": "final_target_quality",
                "target_type": "regression",
                "model_id": "random_forest",
                "observed_value": "",
                "predicted_value": "",
                "observed_numeric": raw["final_target_quality"],
                "predicted_numeric": raw["final_target_quality"] + 0.05,
            }
        )
    return pd.DataFrame(rows)


def _feature_screen() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature_name": "goal_compatibility_class", "mean_importance": 0.5},
            {"feature_name": "governance_information_access_class", "mean_importance": 0.4},
            {"feature_name": "interface_rule_id", "mean_importance": 0.3},
        ]
    )


class InterventionSearchTests(unittest.TestCase):
    def test_intervention_specs_document_minimality_and_strata(self) -> None:
        specs = intervention_spec_table(feature_screen=_feature_screen())
        self.assertIn(BASELINE_SPEC_ID, set(specs["intervention_spec_id"]))
        self.assertIn("global_controller_like", set(specs["access_stratum"]))
        self.assertTrue(specs["minimality_score"].notna().all())
        self.assertTrue((specs["duration_fraction"] >= 0).all())

    def test_select_failure_contexts_has_disjoint_search_and_holdout(self) -> None:
        config = S14Config(max_search_contexts=1, max_holdout_contexts=1)
        selected = select_s14_failure_contexts(_matrix(), _predictions(_matrix()), _feature_screen(), _records(), config)
        self.assertEqual(set(selected["evaluation_stage"]), {SEARCH_STAGE, VALIDATION_STAGE})
        search = set(selected[selected["evaluation_stage"] == SEARCH_STAGE]["base_context_id"])
        holdout = set(selected[selected["evaluation_stage"] == VALIDATION_STAGE]["base_context_id"])
        self.assertTrue(search.isdisjoint(holdout))
        self.assertTrue(selected["selection_reason"].str.contains("s13_failure_context").all())

    def test_small_s14_search_runs_and_validates(self) -> None:
        local_spec = next(spec for spec in DEFAULT_INTERVENTION_SPECS if spec.governance_mechanism_id == "conflict_resolution_radius2")
        global_spec = next(spec for spec in DEFAULT_INTERVENTION_SPECS if spec.global_controller_like)
        baseline = next(spec for spec in DEFAULT_INTERVENTION_SPECS if spec.intervention_spec_id == BASELINE_SPEC_ID)
        config = S14Config(
            array_size=20,
            event_cap=60,
            search_seeds=(301,),
            validation_seeds=(401,),
            max_search_contexts=1,
            max_holdout_contexts=1,
            validation_local_intervention_count=1,
            intervention_specs=(baseline, local_spec, global_spec),
            worker_count=1,
        )
        run_df, condition_df, selected, specs, summary, comparison, rankings = run_s14_search(
            _records(),
            _matrix(),
            _predictions(_matrix()),
            _feature_screen(),
            config,
        )
        validation = validate_s14_outputs(
            run_df,
            condition_df,
            selected,
            specs,
            summary,
            comparison,
            rankings,
            config,
            figure_written=True,
            unit_tests_success=True,
        )
        self.assertFalse(run_df.empty)
        self.assertFalse(summary.empty)
        self.assertFalse(comparison.empty)
        self.assertFalse(rankings.empty)
        self.assertTrue(validation["success"].all())
        self.assertIn(VALIDATION_STAGE, set(run_df["evaluation_stage"]))
        self.assertIn(True, set(run_df["global_controller_like"].astype(bool)))


if __name__ == "__main__":
    unittest.main()
