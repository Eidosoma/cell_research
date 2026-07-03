"""E06 S03 spatial-arrangement tests."""

from __future__ import annotations

import json
import unittest
from collections import Counter

import pandas as pd

from src.e06.algotype_library import builtin_control_records
from src.e06.spatial_arrangements import (
    DEFAULT_ARRANGEMENTS,
    S03Config,
    arrangement_metric_row,
    labels_for_arrangement,
    run_s03_sweep,
    select_s03_candidates,
    validate_s03_outputs,
)


class SpatialArrangementTests(unittest.TestCase):
    def test_arrangement_generators_preserve_counts(self) -> None:
        policy_ids = ("a", "b")
        counts = (5, 15)
        observed_runs = {}
        for arrangement in DEFAULT_ARRANGEMENTS:
            labels = labels_for_arrangement(policy_ids, counts, seed=17, condition_id="unit", arrangement=arrangement)
            self.assertEqual(len(labels), 20)
            self.assertEqual(Counter(labels), Counter({"a": 5, "b": 15}))
            observed_runs[arrangement] = arrangement_metric_row(labels, policy_ids)["initial_contiguous_run_count"]
        self.assertGreaterEqual(len(set(observed_runs.values())), 2)
        self.assertGreater(observed_runs["alternating"], observed_runs["contiguous_patch"])

    def test_select_s03_candidates_prioritizes_aggregation_and_memory(self) -> None:
        records = [
            {
                "policy_ids_json": json.dumps([f"p{i}", f"q{i}"], separators=(",", ":")),
                "display_names_json": json.dumps([f"policy_{i}", "e04_memory_repair_01" if i == 5 else f"other_{i}"], separators=(",", ":")),
                "panel": "memory_vs_original" if i == 5 else "frontier_vs_original",
                "sortedness_range": 0.10 + i / 100,
                "aggregation_delta_range": 20.0 if i == 4 else (15.0 if i == 5 else 2.0),
                "largest_sortedness_jump": 0.20 - i / 100,
                "candidate_threshold_between_ratios_json": "[0.25,0.5]",
                "minority_effect_candidate": i < 3,
                "aggregation_sensitive_candidate": i in {4, 5},
            }
            for i in range(8)
        ]
        selected = select_s03_candidates(pd.DataFrame(records), S03Config(max_candidate_pairs=6, top_sortedness_pairs=3, top_aggregation_pairs=2))
        self.assertLessEqual(len(selected), 6)
        self.assertTrue(selected["candidate_reason"].str.contains("aggregation_sensitive").any())
        self.assertTrue(selected["candidate_reason"].str.contains("memory_repair").any())

    def test_small_s03_sweep_runs_and_validates(self) -> None:
        records = builtin_control_records()
        candidates = pd.DataFrame(
            [
                {
                    "policy_ids_json": json.dumps([records[0]["algotypeId"], records[2]["algotypeId"]], separators=(",", ":")),
                    "display_names_json": json.dumps([records[0]["displayName"], records[2]["displayName"]], separators=(",", ":")),
                    "panel": "unit",
                    "sortedness_range": 0.2,
                    "aggregation_delta_range": 12.0,
                    "largest_sortedness_jump": 0.1,
                    "candidate_threshold_between_ratios_json": "[0.25,0.5]",
                    "minority_effect_candidate": True,
                    "aggregation_sensitive_candidate": True,
                }
            ]
        )
        config = S03Config(array_size=20, event_cap=25, seeds=(1, 2), arrangements=("random_permutation", "contiguous_patch", "alternating"), max_candidate_pairs=1)
        run_df, condition_df, selected = run_s03_sweep(records, candidates, config)
        validation = validate_s03_outputs(run_df, condition_df, selected, config, figure_written=True, unit_tests_success=True)
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertEqual(len(run_df), 12)
        self.assertTrue(cases["initial_arrangement_metrics_computed"])
        self.assertTrue(cases["matched_seed_sets_by_condition"])
        self.assertTrue(cases["matched_arrangements_by_seed"])
        self.assertTrue(cases["realized_ratios_and_counts_match"])


if __name__ == "__main__":
    unittest.main()
