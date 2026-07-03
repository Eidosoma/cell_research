"""Tests for E07 S06 policy-embedding helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.e07.policy_embedding_schema import (
    add_neighbor_labels,
    cosine_neighbor_table,
    neighbor_overlap_at_k,
    sampled_pairwise_distance_spearman,
    stable_fraction,
    validate_policy_embedding_artifacts,
    weighted_mean_profile,
)


class PolicyEmbeddingSchemaTests(unittest.TestCase):
    def test_stable_fraction_is_deterministic(self) -> None:
        left = stable_fraction("policy-a", salt="unit")
        right = stable_fraction("policy-a", salt="unit")
        other = stable_fraction("policy-b", salt="unit")
        self.assertEqual(left, right)
        self.assertNotEqual(left, other)
        self.assertGreaterEqual(left, 0.0)
        self.assertLess(left, 1.0)

    def test_weighted_mean_profile_computes_entity_feature_means(self) -> None:
        frame = pd.DataFrame(
            {
                "policy": ["p1", "p1", "p1", "p2"],
                "feature": ["f1", "f1", "f2", "f1"],
                "value": [1.0, 3.0, 10.0, 4.0],
                "weight": [1.0, 3.0, 1.0, 1.0],
            }
        )
        profile = weighted_mean_profile(
            frame,
            entity_column="policy",
            feature_column="feature",
            value_column="value",
            weight_column="weight",
        )
        self.assertAlmostEqual(float(profile.loc["p1", "f1"]), 2.5)
        self.assertAlmostEqual(float(profile.loc["p1", "f2"]), 10.0)
        self.assertAlmostEqual(float(profile.loc["p2", "f1"]), 4.0)

    def test_neighbors_and_labels_are_consistent(self) -> None:
        ids = ["p1", "p2", "p3"]
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        labels = pd.DataFrame(
            {
                "canonical_policy_id": ids,
                "policy_family": ["a", "a", "b"],
                "algorithm": ["x", "x", "y"],
                "source_experiment_id": ["E1", "E1", "E2"],
                "representation_type": ["dsl", "dsl", "source"],
            }
        )
        neighbors = cosine_neighbor_table(embeddings, ids, k=1)
        labeled = add_neighbor_labels(neighbors, labels)
        p1_neighbor = labeled[labeled["query_policy_id"] == "p1"].iloc[0]
        self.assertEqual(p1_neighbor["neighbor_policy_id"], "p2")
        self.assertTrue(bool(p1_neighbor["same_policy_family"]))

    def test_neighbor_overlap_and_distance_correlation(self) -> None:
        left = pd.DataFrame(
            {
                "query_policy_id": ["p1", "p1", "p2", "p2"],
                "neighbor_policy_id": ["p2", "p3", "p1", "p3"],
                "neighbor_rank": [1, 2, 1, 2],
            }
        )
        right = pd.DataFrame(
            {
                "query_policy_id": ["p1", "p1", "p2", "p2"],
                "neighbor_policy_id": ["p2", "p4", "p1", "p4"],
                "neighbor_rank": [1, 2, 1, 2],
            }
        )
        self.assertAlmostEqual(neighbor_overlap_at_k(left, right, k=2), 0.5)
        values = np.arange(12, dtype=float).reshape(4, 3)
        corr = sampled_pairwise_distance_spearman(values, values * 2.0, seed=1)
        self.assertAlmostEqual(corr, 1.0)

    def test_embedding_validation_accepts_minimal_valid_artifacts(self) -> None:
        embeddings = pd.DataFrame(
            {
                "canonical_policy_id": ["p1", "p2"],
                "embedding_x": [0.1, 0.2],
                "embedding_y": [0.0, 0.3],
                "behavior_embedding_dim_00": [0.1, 0.2],
                "behavior_embedding_dim_01": [0.0, 0.3],
                "behavior_observation_count": [20, 30],
            }
        )
        neighbors = pd.DataFrame(
            {
                "embedding_name": ["behavior", "behavior"],
                "query_policy_id": ["p1", "p2"],
                "neighbor_policy_id": ["p2", "p1"],
                "neighbor_rank": [1, 1],
            }
        )
        stability = pd.DataFrame(
            {
                "pairwise_distance_spearman": [0.9],
                "neighbor_overlap_at_5": [0.8],
            }
        )
        baseline = pd.DataFrame(
            {
                "metric_name": [
                    "behavior_policy_family_agreement_lift_over_random_at_5",
                    "behavior_source_experiment_agreement_at_5",
                    "behavior_syntax_neighbor_overlap_at_5",
                ],
                "metric_value": [0.2, 0.5, 0.4],
            }
        )
        checks = validate_policy_embedding_artifacts(
            embeddings,
            neighbors,
            stability,
            baseline,
            expected_policy_ids=["p1", "p2"],
            min_supported_policies=2,
        )
        self.assertTrue(bool(checks["success"].all()))


if __name__ == "__main__":
    unittest.main()
