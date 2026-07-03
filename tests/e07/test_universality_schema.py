"""Tests for E07 S10 universality helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.e07.universality_schema import (
    cautious_cluster_label,
    classify_cluster_status,
    dominance,
    finite_standardize,
    neighbor_distance_matrix,
    validate_universality_artifacts,
)


class UniversalitySchemaTests(unittest.TestCase):
    def test_finite_standardize_drops_constant_columns(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "constant": [5.0, 5.0, 5.0], "bad": [np.nan, np.inf, -np.inf]})
        scaled, stats = finite_standardize(frame)
        self.assertEqual(stats["inputFeatureCount"], 3)
        self.assertEqual(stats["retainedFeatureCount"], 1)
        self.assertIn("a", scaled.columns)
        self.assertTrue(np.isfinite(scaled.to_numpy()).all())

    def test_dominance_and_status_classification(self) -> None:
        label, rate, count = dominance(["E03", "E03", "E05", ""])
        self.assertEqual(label, "E03")
        self.assertAlmostEqual(rate, 0.5)
        self.assertEqual(count, 3)
        row = {
            "entity_count": 10,
            "stability_mean_ari": 0.8,
            "source_dominance_rate": 0.95,
            "support_status_dominance_rate": 0.4,
            "dominant_support_status": "behavior_profile_supported",
            "max_control_ari": 0.2,
        }
        self.assertEqual(classify_cluster_status(row), "source_dominated_constraint")
        row["source_dominance_rate"] = 0.4
        row["max_control_ari"] = 0.9
        self.assertEqual(classify_cluster_status(row), "control_aligned_constraint")

    def test_cautious_label_uses_bounded_feature_language(self) -> None:
        row = {
            "entity_type": "policy",
            "class_status": "bounded_interpretable_class",
            "dominant_family": "rule_recombination",
            "dominant_support_status": "behavior_profile_supported",
            "mean_policy_memory_depth_proxy": 1.4,
            "mean_policy_stochastic_choice": 0.3,
        }
        label = cautious_cluster_label(row)
        self.assertIn("prox", label)
        self.assertIn("policy", label)

    def test_neighbor_distance_matrix_is_symmetric_and_filled(self) -> None:
        neighbors = pd.DataFrame(
            [
                {"entity_type": "policy", "distance_variant": "platonic_supported_only", "query_entity_id": "a", "neighbor_entity_id": "b", "distance": 0.2},
                {"entity_type": "policy", "distance_variant": "platonic_supported_only", "query_entity_id": "b", "neighbor_entity_id": "c", "distance": 0.4},
            ]
        )
        matrix, stats = neighbor_distance_matrix(neighbors, entity_type="policy", distance_variant="platonic_supported_only", entity_ids=["a", "b", "c"])
        self.assertEqual(stats["knownEdgeCount"], 2)
        self.assertAlmostEqual(matrix.loc["a", "b"], 0.2)
        self.assertAlmostEqual(matrix.loc["b", "a"], 0.2)
        self.assertGreater(matrix.loc["a", "c"], 0.4)

    def test_validate_universality_artifacts_minimal(self) -> None:
        summary = pd.DataFrame(
            {
                "entity_type": ["policy", "goal", "world", "policy"],
                "branch_name": ["policy_supported_only", "goal_supported_only", "world_supported_only", "policy_all"],
                "branch_role": ["primary", "primary", "primary", "sensitivity"],
                "class_id": ["UC-P-01", "UC-G-01", "UC-W-01", "UC-P-A01"],
                "class_status": ["bounded_interpretable_class", "source_dominated_constraint", "bounded_interpretable_class", "unstable_constraint"],
                "entity_count": [4, 4, 4, 4],
                "cautious_label": ["a", "b", "c", "d"],
            }
        )
        assignments = pd.DataFrame(
            {
                "entity_type": ["policy", "goal", "world"],
                "branch_name": ["policy_supported_only", "goal_supported_only", "world_supported_only"],
                "entity_id": ["p1", "g1", "w1"],
                "class_id": ["UC-P-01", "UC-G-01", "UC-W-01"],
                "map_x": [0.0, 1.0, 2.0],
                "map_y": [0.0, 1.0, 2.0],
            }
        )
        stability = pd.DataFrame({"branch_name": ["policy_supported_only"], "mean_adjusted_rand_index": [0.7]})
        controls = pd.DataFrame({"control_family": ["source_metadata", "metric_only"], "adjusted_rand_index": [0.1, 0.2]})
        exemplars = pd.DataFrame({"entity_id": ["p1"]})
        counterexamples = pd.DataFrame({"entity_id": ["p2"]})
        checks = validate_universality_artifacts(
            summary,
            assignments,
            stability,
            controls,
            exemplars,
            counterexamples,
            {"stableInvariantFeatures": ["policy_memory_depth_proxy"], "usedInvariantFeatures": ["policy_memory_depth_proxy"]},
        )
        self.assertTrue(checks["success"].all(), checks.to_string(index=False))


if __name__ == "__main__":
    unittest.main()
