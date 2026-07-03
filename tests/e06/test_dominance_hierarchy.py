"""E06 S06 dominance hierarchy tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e06.algotype_library import builtin_control_records
from src.e06.dominance_hierarchy import (
    S06Config,
    build_s06_conditions,
    context_dependence,
    dominance_hierarchy,
    frozen_indices_for_profile,
    initial_values_for_profile,
    relabeling_metric_control,
    run_s06_sweep,
    select_s06_base_contests,
    summarize_s06_runs,
    validate_s06_outputs,
)


def _records_and_metadata() -> tuple[list[dict], pd.DataFrame]:
    records = builtin_control_records()
    names = ["original_bubble", "original_insertion", "original_selection", "random_noisy_inversion_cleaner"]
    categories = ["original", "original", "original", "randomized"]
    rows = []
    for record, name, category in zip(records, names, categories, strict=True):
        rows.append(
            {
                "algotype_id": record["algotypeId"],
                "display_name": name,
                "source_category": category,
            }
        )
    return records, pd.DataFrame(rows)


def _candidate_and_score_tables(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_ids = [records[0]["algotypeId"], records[2]["algotypeId"]]
    display_names = ["original_bubble", "original_selection"]
    source_categories = ["original", "original"]
    ratios = [0.25, 0.75]
    candidates = pd.DataFrame(
        [
            {
                "score_id": "score_unit",
                "source_condition_id": "s04_unit",
                "panel": "unit",
                "arrangement": "alternating",
                "goal_profile_id": "opposite_direction",
                "goal_compatibility_class": "opposite_goal",
                "display_names_json": json.dumps(display_names, separators=(",", ":")),
                "ratio_targets_json": json.dumps(ratios, separators=(",", ":")),
                "primary_target_quality": 0.8,
                "quality_synergy": -0.1,
                "goal_conflict_index": 0.6,
                "dominance_abs_margin": 0.3,
                "aggregation_delta_percent": 4.0,
                "compatibility_pattern_class": "goal_tension_local_success",
                "conflicting_dimensions_json": "[]",
                "s06_priority_score": 2.5,
            }
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "score_id": "score_unit",
                "source_condition_id": "s04_unit",
                "policy_ids_json": json.dumps(policy_ids, separators=(",", ":")),
                "display_names_json": json.dumps(display_names, separators=(",", ":")),
                "source_categories_json": json.dumps(source_categories, separators=(",", ":")),
                "ratio_targets_json": json.dumps(ratios, separators=(",", ":")),
            }
        ]
    )
    return candidates, scores


class DominanceHierarchyTests(unittest.TestCase):
    def test_value_and_perturbation_profiles_are_deterministic(self) -> None:
        self.assertEqual(initial_values_for_profile(4, 11, "reverse_sorted"), (4, 3, 2, 1))
        self.assertEqual(frozen_indices_for_profile(5, "frozen_edges_2"), frozenset({0, 4}))
        self.assertEqual(frozen_indices_for_profile(1, "frozen_edges_2"), frozenset({0}))

    def test_select_s06_contests_joins_policy_ids_and_adds_original_controls(self) -> None:
        records, metadata = _records_and_metadata()
        candidates, scores = _candidate_and_score_tables(records)
        selected = select_s06_base_contests(candidates, scores, metadata, S06Config(max_s05_candidates=1))
        self.assertEqual(len(selected), 4)
        self.assertIn("s05_candidate", set(selected["source_kind"]))
        self.assertEqual(int((selected["source_kind"] == "paper_original_control").sum()), 3)
        self.assertFalse(selected["policy_ids_json"].isna().any())

    def test_conditions_include_relabel_swaps(self) -> None:
        records, metadata = _records_and_metadata()
        candidates, scores = _candidate_and_score_tables(records)
        selected = select_s06_base_contests(candidates, scores, metadata, S06Config(max_s05_candidates=1))
        config = S06Config(
            array_size=12,
            event_cap=10,
            seeds=(1,),
            max_s05_candidates=1,
            value_profiles=("random_permutation",),
            perturbation_profiles=("none",),
        )
        conditions = build_s06_conditions(selected, config)
        self.assertEqual(len(conditions), 8)
        for group_id in {condition.symmetry_group_id for condition in conditions}:
            orientations = {condition.orientation for condition in conditions if condition.symmetry_group_id == group_id}
            self.assertEqual(orientations, {"as_selected", "relabel_swap"})

    def test_small_s06_sweep_runs_and_validates(self) -> None:
        records, metadata = _records_and_metadata()
        candidates, scores = _candidate_and_score_tables(records)
        config = S06Config(
            array_size=12,
            event_cap=16,
            seeds=(1, 2),
            max_s05_candidates=1,
            value_profiles=("random_permutation",),
            perturbation_profiles=("none",),
        )
        run_df, condition_df, base_contests = run_s06_sweep(records, metadata, candidates, scores, config)
        summary = summarize_s06_runs(run_df)
        hierarchy = dominance_hierarchy(run_df, metadata)
        validation = validate_s06_outputs(
            run_df,
            condition_df,
            base_contests,
            hierarchy,
            config,
            figure_written=True,
            unit_tests_success=True,
        )
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertEqual(len(run_df), 16)
        self.assertTrue(cases["paper_originals_included"])
        self.assertTrue(cases["opposite_goal_assignments_valid"])
        self.assertTrue(cases["matched_relabel_run_pairs_by_seed"])
        self.assertTrue(cases["realized_ratios_and_counts_match"])
        self.assertFalse(summary.empty)
        self.assertFalse(hierarchy.empty)
        self.assertFalse(context_dependence(summary).empty)

    def test_relabeling_metric_control_passes(self) -> None:
        self.assertTrue(relabeling_metric_control())


if __name__ == "__main__":
    unittest.main()
