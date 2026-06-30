from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from platonic_space.invariant_search import (
    INVARIANT_SEARCH_CLAIM_BOUNDARY,
    build_invariant_feature_frame,
    confound_adjusted_associations,
    cross_world_holdout_search,
    invariant_candidate_table,
    invariant_feature_columns,
    model_comparison_table,
    summarize_coefficients,
    summarize_cross_world_metrics,
    summarize_group_ablation,
    target_error_weights,
    validate_invariant_search,
)


class TestE07InvariantSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            Path("/artifacts/research_steps/S05/modeling_frame.parquet"),
            Path("/artifacts/research_steps/S02/policy_abstract_catalog.parquet"),
            Path("/artifacts/research_steps/S03/goal_catalog.parquet"),
            Path("/artifacts/research_steps/S01/normalized_world_catalog.parquet"),
            Path("/artifacts/research_steps/S05/error_limits.parquet"),
            Path("/artifacts/models/e07_behavior_predictor/model_card.json"),
            Path("/artifacts/research_steps/S08/platonic_distances.parquet"),
            Path("/artifacts/research_steps/S08/distance_entities.parquet"),
        )
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("S01-S08 artifacts are required")
        cls.frame = pd.read_parquet(required[0])
        cls.policy_catalog = pd.read_parquet(required[1])
        cls.goal_catalog = pd.read_parquet(required[2])
        cls.world_catalog = pd.read_parquet(required[3])
        cls.error_limits = pd.read_parquet(required[4])
        cls.target_columns = json.loads(required[5].read_text(encoding="utf-8")).get("targetColumns", [])
        cls.s08_distances = pd.read_parquet(required[6])
        cls.s08_entities = pd.read_parquet(required[7])
        cls.target_weights = target_error_weights(cls.frame, cls.error_limits, cls.target_columns)
        cls.feature_frame, cls.centrality = build_invariant_feature_frame(
            cls.frame,
            cls.policy_catalog,
            cls.goal_catalog,
            cls.world_catalog,
            cls.s08_distances,
            cls.s08_entities,
        )

    def test_invariant_feature_frame_uses_s08_context(self) -> None:
        self.assertTrue(INVARIANT_SEARCH_CLAIM_BOUNDARY.startswith("Empirical invariant search"))
        columns = invariant_feature_columns(self.feature_frame)
        self.assertGreaterEqual(len(self.feature_frame), 80000)
        self.assertGreaterEqual(len(columns), 30)
        self.assertTrue(any(column.endswith("s08_mean_distance") for column in columns))
        self.assertGreaterEqual(self.centrality["entityType"].nunique(), 3)

    def test_cross_world_metrics_and_validation(self) -> None:
        metrics, ablations, coefficients, eligibility = cross_world_holdout_search(
            self.feature_frame,
            target_columns=self.target_columns,
            max_targets=3,
            n_splits=3,
        )
        model_summary = summarize_cross_world_metrics(metrics, self.target_weights)
        comparisons = model_comparison_table(model_summary)
        ablation_summary = summarize_group_ablation(ablations)
        associations = confound_adjusted_associations(self.feature_frame, self.target_columns, max_targets=3)
        coefficient_summary = summarize_coefficients(coefficients)
        candidates = invariant_candidate_table(ablation_summary, associations, coefficient_summary)
        validation = validate_invariant_search(self.feature_frame, metrics, model_summary, ablation_summary, candidates, associations)
        hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

        self.assertGreaterEqual(eligibility["eligible"].sum(), 3)
        self.assertEqual({"global_mean", "invariant_feature_ridge", "confound_metadata_ridge", "combined_ridge"}, set(metrics["modelName"]))
        self.assertFalse(comparisons.empty)
        self.assertFalse(ablation_summary.empty)
        self.assertFalse(associations.empty)
        self.assertFalse(candidates.empty)
        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
