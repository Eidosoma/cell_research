"""Tests for E07 S07 goal-embedding helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.e07.goal_embedding_schema import (
    conflict_separation_summary,
    conflict_separation_table,
    policy_support_weight,
    unique_conflict_pairs,
    validate_goal_embedding_artifacts,
    weighted_frequency_profile,
    weighted_numeric_means,
)


class GoalEmbeddingSchemaTests(unittest.TestCase):
    def test_policy_support_weight_uses_status_and_count(self) -> None:
        self.assertEqual(policy_support_weight("behavior_profile_supported", 10), 1.0)
        self.assertGreater(policy_support_weight("sparse_behavior_profile", 4), 0.1)
        self.assertLess(policy_support_weight("sparse_behavior_profile", 4), 1.0)
        self.assertEqual(policy_support_weight("no_behavior_observations", 0), 0.0)
        self.assertEqual(policy_support_weight("policy_missing", None), 0.0)

    def test_weighted_frequency_profile_returns_shares(self) -> None:
        frame = pd.DataFrame(
            {
                "goal": ["g1", "g1", "g1", "g2"],
                "feature": ["a", "a", "b", "b"],
                "weight": [1.0, 2.0, 1.0, 4.0],
            }
        )
        profile = weighted_frequency_profile(frame, entity_column="goal", feature_column="feature", weight_column="weight")
        self.assertAlmostEqual(float(profile.loc["g1", "a"]), 0.75)
        self.assertAlmostEqual(float(profile.loc["g1", "b"]), 0.25)
        self.assertAlmostEqual(float(profile.loc["g2", "b"]), 1.0)

    def test_weighted_numeric_means(self) -> None:
        frame = pd.DataFrame(
            {
                "goal": ["g1", "g1", "g2"],
                "weight": [1.0, 3.0, 2.0],
                "x": [1.0, 3.0, 5.0],
                "y": [0.0, 4.0, 6.0],
            }
        )
        means = weighted_numeric_means(frame, entity_column="goal", numeric_columns=["x", "y"], weight_column="weight")
        self.assertAlmostEqual(float(means.loc["g1", "x"]), 2.5)
        self.assertAlmostEqual(float(means.loc["g1", "y"]), 3.0)
        self.assertAlmostEqual(float(means.loc["g2", "x"]), 5.0)

    def test_conflict_pairs_and_separation_summary(self) -> None:
        conflicts = pd.DataFrame(
            {
                "canonical_goal_id_a": ["g1", "g2"],
                "canonical_goal_id_b": ["g2", "g1"],
                "conflict_type": ["opposes", "opposes"],
                "conflict_group_id": ["grp", "grp"],
            }
        )
        pairs = unique_conflict_pairs(conflicts)
        self.assertEqual(len(pairs), 1)
        embeddings = pd.DataFrame(
            {
                "canonical_goal_id": ["g1", "g2", "g3"],
                "goal_embedding_dim_00": [0.0, 10.0, 1.0],
                "goal_embedding_dim_01": [0.0, 0.0, 0.0],
            }
        )
        distances = conflict_separation_table(
            embeddings,
            conflicts,
            coordinate_columns=["goal_embedding_dim_00", "goal_embedding_dim_01"],
        )
        summary = conflict_separation_summary(distances)
        lift = float(summary.set_index("metric_name").loc["known_conflict_distance_lift_over_reference_median", "metric_value"])
        self.assertGreater(lift, 0.0)

    def test_goal_embedding_validation_accepts_minimal_valid_artifacts(self) -> None:
        embeddings = pd.DataFrame(
            {
                "canonical_goal_id": ["g1", "g2"],
                "embedding_x": [0.0, 1.0],
                "embedding_y": [0.0, 1.0],
                "goal_embedding_dim_00": [0.0, 1.0],
                "goal_embedding_dim_01": [0.0, 1.0],
                "goal_behavior_row_count": [10, 20],
                "supported_policy_row_count": [5, 10],
                "goal_support_status": ["supported_policy_evidence", "supported_policy_evidence"],
            }
        )
        stability = pd.DataFrame({"pairwise_distance_spearman": [0.8], "neighbor_overlap_at_3": [0.5]})
        sensitivity = pd.DataFrame({"analysis_name": ["unit"], "metric_name": ["overlap"], "metric_value": [0.5]})
        conflict_summary = pd.DataFrame(
            {
                "metric_name": ["known_conflict_distance_lift_over_reference_median"],
                "metric_value": [0.2],
            }
        )
        checks = validate_goal_embedding_artifacts(
            embeddings,
            stability,
            sensitivity,
            conflict_summary,
            expected_goal_ids=["g1", "g2"],
            min_modeled_goals=2,
            min_supported_goals=2,
        )
        self.assertTrue(bool(checks["success"].all()))


if __name__ == "__main__":
    unittest.main()
