"""Tests for E07 S08 Platonic-distance helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.e07.distance_schema import (
    attach_neighbor_metadata,
    euclidean_neighbor_table,
    sampled_pairwise_distance_spearman,
    standardized_numeric_frame,
    support_weight_from_status,
    validate_platonic_distance_artifacts,
)


class DistanceSchemaTests(unittest.TestCase):
    def test_standardized_numeric_frame_drops_constants(self) -> None:
        scaled, stats = standardized_numeric_frame(pd.DataFrame({"vary": [1.0, 2.0, 3.0], "constant": [5.0, 5.0, 5.0]}))
        self.assertEqual(stats["featureCount"], 2)
        self.assertEqual(stats["retainedFeatureCount"], 1)
        self.assertIn("vary", scaled.columns)
        self.assertNotIn("constant", scaled.columns)
        self.assertAlmostEqual(float(scaled["vary"].mean()), 0.0)

    def test_euclidean_neighbors_identify_nearest_entity(self) -> None:
        neighbors = euclidean_neighbor_table(
            np.array([[0.0, 0.0], [0.2, 0.0], [3.0, 0.0]]),
            ["a", "b", "c"],
            k=1,
            entity_type="policy",
            distance_variant="unit",
            baseline_family="platonic",
        )
        a_neighbor = neighbors[neighbors["query_entity_id"] == "a"].iloc[0]
        self.assertEqual(a_neighbor["neighbor_entity_id"], "b")
        self.assertAlmostEqual(float(a_neighbor["distance"]), 0.2)

    def test_attach_neighbor_metadata_adds_support_and_labels(self) -> None:
        neighbors = euclidean_neighbor_table(
            np.array([[0.0], [1.0]]),
            ["a", "b"],
            k=1,
            entity_type="goal",
            distance_variant="unit",
            baseline_family="platonic",
        )
        metadata = pd.DataFrame(
            {
                "entity_type": ["goal", "goal"],
                "entity_id": ["a", "b"],
                "source_experiment_id": ["E1", "E2"],
                "family": ["order", "order"],
                "support_status": ["supported", "sparse"],
                "support_weight": [1.0, 0.25],
            }
        )
        labeled = attach_neighbor_metadata(neighbors, metadata)
        self.assertFalse(bool(labeled.iloc[0]["same_source_experiment"]))
        self.assertTrue(bool(labeled.iloc[0]["same_family"]))
        self.assertAlmostEqual(float(labeled.iloc[0]["pair_support_weight"]), 0.25)

    def test_support_weight_and_distance_spearman(self) -> None:
        self.assertEqual(support_weight_from_status("behavior_profile_supported", 10), 1.0)
        self.assertGreater(support_weight_from_status("sparse_behavior_profile", 4), 0.1)
        self.assertEqual(support_weight_from_status("no_behavior_observations", 0), 0.0)
        values = np.arange(12, dtype=float).reshape(4, 3)
        self.assertAlmostEqual(sampled_pairwise_distance_spearman(values, values * 3.0, seed=2), 1.0)

    def test_validation_flags_missingness_and_conflict_failures(self) -> None:
        benchmarks = pd.DataFrame(
            {
                "entity_type": ["policy", "policy", "policy", "policy"],
                "distance_variant": [
                    "platonic_uncertainty_weighted",
                    "platonic_supported_only",
                    "syntax_baseline",
                    "metric_only_baseline",
                ],
                "baseline_family": ["platonic", "platonic", "syntax", "metric_only"],
            }
        )
        benchmarks = pd.concat(
            [
                benchmarks,
                pd.DataFrame(
                    {
                        "entity_type": ["goal"],
                        "distance_variant": ["source_metadata_baseline"],
                        "baseline_family": ["source_metadata"],
                    }
                ),
            ],
            ignore_index=True,
        )
        neighbors = pd.DataFrame({"query_entity_id": ["p1"], "neighbor_entity_id": ["p2"], "distance_variant": ["x"], "distance": [0.1]})
        audit = pd.DataFrame({"audit_case": ["source"], "success": [False]})
        conflict = pd.DataFrame(
            {
                "distance_variant": ["platonic_supported_only"],
                "eligible_mode": ["supported_only"],
                "known_conflict_distance_lift_over_reference_median": [-1.0],
                "success": [False],
            }
        )
        checks = validate_platonic_distance_artifacts(
            benchmarks,
            neighbors,
            audit,
            conflict,
            required_variants=[
                "policy:platonic_uncertainty_weighted",
                "policy:platonic_supported_only",
                "policy:syntax_baseline",
            ],
        )
        lookup = checks.set_index("validation_case")["success"].to_dict()
        self.assertFalse(bool(lookup["source_missingness_dominance_not_detected"]))
        self.assertFalse(bool(lookup["supported_platonic_conflict_separation_passed"]))


if __name__ == "__main__":
    unittest.main()
