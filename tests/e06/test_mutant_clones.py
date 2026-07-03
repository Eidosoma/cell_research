"""E06 S11 mutant clone experiment tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e06.algotype_library import builtin_control_records
from src.e06.governance_mechanisms import DEFAULT_GOVERNANCE_MECHANISMS, DEFAULT_INTERFACE_RULES
from src.e06.mutant_clones import (
    CloneSpec,
    S11Config,
    clone_layout_labels,
    compare_to_behavior_baseline,
    mutant_strata_summary,
    objective_table,
    run_s11_sweep,
    select_s11_base_contexts,
    summarize_s11_runs,
    validate_s11_outputs,
)


def _json(values: list) -> str:
    return json.dumps(values, separators=(",", ":"))


def _sample_s10_selected(records: list[dict]) -> pd.DataFrame:
    left = records[2]
    right = records[3]
    rows = []
    for idx, ratios in enumerate(([0.75, 0.25], [0.50, 0.50])):
        rows.append(
            {
                "base_context_id": f"s10_base_unit_{idx}",
                "selection_rank": idx + 1,
                "selection_reason": "s07_exemplar" if idx == 0 else "metric_extreme:high_conflict_continuum",
                "source_s07_run_id": f"s07_unit_{idx}",
                "source_research_step_id": "S07",
                "source_condition_id": f"s07_condition_{idx}",
                "s07_label": "stable_segregated_patch" if idx == 0 else "partial_mosaic",
                "s07_classification_mode": "metric_continuum",
                "panel": "unit",
                "candidate_reason": "unit_fixture",
                "source_arrangement": "graft_like_insertions",
                "goal_profile_id": "same_increasing" if idx == 0 else "opposite_direction",
                "goal_compatibility_class": "same_goal" if idx == 0 else "opposite_goal",
                "policy_ids_json": _json([left["algotypeId"], right["algotypeId"]]),
                "display_names_json": _json([left["displayName"], right["displayName"]]),
                "source_categories_json": _json([left["sourceCategory"], right["sourceCategory"]]),
                "ratio_targets_json": _json(ratios),
                "host_policy_id": left["algotypeId"] if ratios[0] >= ratios[1] else right["algotypeId"],
                "graft_policy_id": right["algotypeId"] if ratios[0] >= ratios[1] else left["algotypeId"],
                "host_display_name": left["displayName"] if ratios[0] >= ratios[1] else right["displayName"],
                "graft_display_name": right["displayName"] if ratios[0] >= ratios[1] else left["displayName"],
                "host_source_category": left["sourceCategory"] if ratios[0] >= ratios[1] else right["sourceCategory"],
                "graft_source_category": right["sourceCategory"] if ratios[0] >= ratios[1] else left["sourceCategory"],
                "host_policy_index": 0,
                "graft_policy_index": 1,
                "value_profile": "random_permutation",
                "perturbation_profile": "none",
                "source_final_target_quality": 0.7 - idx * 0.2,
                "source_goal_conflict_index": 0.1 + idx * 0.5,
                "source_aggregation_delta_percent": 10.0 + idx,
                "source_largest_block_fraction": 0.5,
                "source_mosaic_continuum_score": 0.3 + idx * 0.4,
                "source_conflict_continuum_score": 0.2 + idx * 0.6,
            }
        )
    return pd.DataFrame(rows)


def _sample_s10_runs(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = ["stable_segregated_patch", "rejected_or_edge_segregated", "partial_mosaic_or_contained"]
    for _, row in selected.iterrows():
        for idx, outcome in enumerate(outcomes):
            rows.append(
                {
                    "research_step_id": "S10",
                    "base_context_id": row["base_context_id"],
                    "graft_outcome_class": outcome,
                    "target_delta_vs_pre_graft": -0.02 * idx,
                    "graft_relative_largest_block_fraction": 0.4 + idx * 0.1,
                }
            )
    return pd.DataFrame(rows)


def _sample_s09_rankings(config: S11Config) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "governance_mechanism_id": mechanism.mechanism_id,
                "global_controller_like": mechanism.global_controller_like,
            }
            for mechanism in config.governance_mechanisms
        ]
    )


class MutantCloneTests(unittest.TestCase):
    def test_selfish_objectives_are_explicitly_defined(self) -> None:
        table = objective_table()
        self.assertEqual(set(table["objective_id"]), {"edge_position_left", "clone_cohesion", "replicative_takeover"})
        self.assertTrue(table["definition"].astype(str).str.len().gt(20).all())
        self.assertTrue(table["biological_analogy_caveat"].astype(str).str.contains("computational").all())

    def test_clone_layout_logs_size_and_location(self) -> None:
        spec = CloneSpec("unit_center", 5, 4, "center", "graft_like_insertions", "unit")
        labels = clone_layout_labels("host", "mutant", spec, array_size=12, seed=7, condition_id="unit")
        self.assertEqual(labels.count("mutant"), 4)
        self.assertEqual(labels[4:8], ("mutant", "mutant", "mutant", "mutant"))

        edge_spec = CloneSpec("unit_left", 5, 3, "left_edge", "contiguous_patch", "unit")
        edge_labels = clone_layout_labels("host", "mutant", edge_spec, array_size=10, seed=7, condition_id="unit")
        self.assertEqual(edge_labels[:3], ("mutant", "mutant", "mutant"))
        self.assertEqual(edge_labels.count("host"), 7)

    def test_select_s11_contexts_uses_s10_sensitivity(self) -> None:
        records = builtin_control_records()
        selected = _sample_s10_selected(records)
        config = S11Config(max_base_contexts=1)
        contexts = select_s11_base_contexts(selected, _sample_s10_runs(selected), config)
        self.assertEqual(len(contexts), 1)
        self.assertIn("s10_graft_sensitive", contexts.iloc[0]["s11_selection_reason"])
        self.assertGreater(contexts.iloc[0]["s10_graft_sensitivity_score"], 0)

    def test_small_s11_sweep_runs_and_validates_strata(self) -> None:
        records = builtin_control_records()
        config = S11Config(
            array_size=20,
            event_cap=50,
            seeds=(11, 12),
            max_base_contexts=1,
            clone_specs=(CloneSpec("unit_center", 10, 3, "center", "graft_like_insertions", "unit"),),
            interface_rules=(DEFAULT_INTERFACE_RULES[0], DEFAULT_INTERFACE_RULES[1]),
            governance_mechanisms=(DEFAULT_GOVERNANCE_MECHANISMS[0], DEFAULT_GOVERNANCE_MECHANISMS[1], DEFAULT_GOVERNANCE_MECHANISMS[-1]),
        )
        selected = _sample_s10_selected(records)
        run_df, condition_df, contexts, initial_state_df, objectives = run_s11_sweep(
            records,
            selected,
            _sample_s10_runs(selected),
            _sample_s09_rankings(config),
            config,
        )
        summary = summarize_s11_runs(run_df)
        comparison = compare_to_behavior_baseline(summary)
        strata = mutant_strata_summary(comparison, config)
        validation = validate_s11_outputs(
            run_df,
            condition_df,
            contexts,
            initial_state_df,
            objectives,
            summary,
            comparison,
            strata,
            config,
            figure_written=True,
            unit_tests_success=True,
        )
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertEqual(len(run_df), 36)
        self.assertEqual(initial_state_df["initial_clone_state_id"].nunique(), 6)
        self.assertTrue(cases["selfish_objectives_explicitly_defined"])
        self.assertTrue(cases["clone_initial_size_location_logged"])
        self.assertTrue(cases["behavior_explicit_finite_global_strata_preserved"])
        self.assertTrue(cases["matched_seed_sets_by_condition"])
        self.assertTrue(cases["replicative_objective_changes_clone_count"])
        self.assertTrue(cases["clone_counts_bounded_and_total_size_preserved"])


if __name__ == "__main__":
    unittest.main()
