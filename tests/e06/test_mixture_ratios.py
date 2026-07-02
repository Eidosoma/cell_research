"""E06 S02 mixture-ratio tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e06.algotype_library import builtin_control_records
from src.e06.mixture_ratios import (
    MixtureCondition,
    S02Config,
    build_initial_policy_ids,
    counts_for_ratios,
    run_s02_sweep,
    select_s02_panel,
    simulate_mixture_condition,
    validate_s02_outputs,
)


class MixtureRatioTests(unittest.TestCase):
    def test_counts_for_ratios_preserves_total_and_positive_minorities(self) -> None:
        self.assertEqual(counts_for_ratios((0.01, 0.99), 100), (1, 99))
        self.assertEqual(sum(counts_for_ratios((0.34, 0.33, 0.33), 100)), 100)
        self.assertTrue(all(count > 0 for count in counts_for_ratios((0.01, 0.01, 0.98), 100)))

    def test_initial_policy_ids_realize_counts_deterministically(self) -> None:
        labels1 = build_initial_policy_ids(("a", "b"), (5, 95), seed=17, condition_id="unit")
        labels2 = build_initial_policy_ids(("a", "b"), (5, 95), seed=17, condition_id="unit")
        self.assertEqual(labels1, labels2)
        self.assertEqual(labels1.count("a"), 5)
        self.assertEqual(labels1.count("b"), 95)

    def test_small_control_mixture_runs_and_validates_ratios(self) -> None:
        records = builtin_control_records()
        records_by_id = {record["algotypeId"]: record for record in records}
        condition = MixtureCondition(
            condition_id="unit_pair",
            condition_kind="pair",
            panel="unit",
            policy_ids=(records[0]["algotypeId"], records[2]["algotypeId"]),
            ratio_targets=(0.25, 0.75),
            label="unit",
        )
        config = S02Config(array_size=20, event_cap=30, seeds=(101,), top_discovered_count=0)
        row = simulate_mixture_condition(condition, records_by_id, seed=101, config=config)
        self.assertTrue(row["count_preservation_success"])
        self.assertEqual(row["invalid_action_count"], 0)
        self.assertLessEqual(row["realized_ratio_max_abs_error"], 1e-12)

    def test_validation_detects_balanced_seed_sets(self) -> None:
        run_df = pd.DataFrame(
            [
                {
                    "condition_id": condition_id,
                    "seed": seed,
                    "realized_ratio_max_abs_error": 0.0,
                    "count_preservation_success": True,
                    "invalid_action_count": 0,
                }
                for condition_id in ("pair_unit", "threeway_unit")
                for seed in (1, 2)
            ]
        )
        condition_df = pd.DataFrame(
            [
                {
                    "condition_id": "pair_unit",
                    "condition_kind": "pair",
                    "source_categories_json": json.dumps(["original", "null_control", "randomized"]),
                },
                {
                    "condition_id": "threeway_unit",
                    "condition_kind": "threeway",
                    "source_categories_json": json.dumps(["discovered", "memory_repair"]),
                },
            ]
        )
        panel_df = pd.DataFrame({"algotype_id": ["a", "b", "c"]})
        validation = validate_s02_outputs(run_df, condition_df, panel_df, S02Config(array_size=20, event_cap=20, seeds=(1, 2), top_discovered_count=0), figure_written=True, unit_tests_success=True)
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertTrue(cases["realized_ratios_match_expected_counts"])
        self.assertTrue(cases["balanced_seed_sets_by_condition"])
        self.assertTrue(cases["bounded_panel_includes_priority_categories"])


if __name__ == "__main__":
    unittest.main()
