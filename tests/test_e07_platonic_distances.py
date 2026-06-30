from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from platonic_space.platonic_distances import (
    PLATONIC_DISTANCE_CLAIM_BOUNDARY,
    build_world_profiles,
    distance_baseline_comparison,
    fit_world_embedding,
    goal_sparse_penalty,
    nearest_neighbor_sanity,
    nearest_neighbor_table,
    pairwise_distance_tables,
    policy_sparse_penalty,
    prior_neighbor_overlap,
    symmetry_triangle_diagnostics,
    target_error_weights,
    validate_platonic_distances,
    world_sparse_penalty,
)


POLICY_LABEL_COLUMNS = ("policyLabel", "sourceExperimentId", "policyFamily", "policyKind", "representationType", "stochasticity")
GOAL_LABEL_COLUMNS = ("goalLabel", "goalFamily", "goalKind", "representationType", "targetDimensionality", "sparseUncertaintyLevel")
WORLD_LABEL_COLUMNS = ("title", "experimentId", "worldFamily", "substrateClass", "replayability", "metadataCompleteness")
BASELINE_LABEL_COLUMNS = {
    "policy": ("sourceExperimentId", "policyFamily", "policyKind", "representationType"),
    "goal": ("goalFamily", "goalKind", "representationType", "targetDimensionality"),
    "world": ("experimentId", "worldFamily", "substrateClass", "replayability"),
}


class TestE07PlatonicDistances(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            Path("/artifacts/research_steps/S05/modeling_frame.parquet"),
            Path("/artifacts/research_steps/S01/normalized_world_catalog.parquet"),
            Path("/artifacts/research_steps/S05/error_limits.parquet"),
            Path("/artifacts/models/e07_behavior_predictor/model_card.json"),
            Path("/artifacts/research_steps/S06/policy_embeddings.parquet"),
            Path("/artifacts/research_steps/S06/policy_embedding_coverage.parquet"),
            Path("/artifacts/research_steps/S06/nearest_neighbors.parquet"),
            Path("/artifacts/research_steps/S07/goal_embeddings.parquet"),
            Path("/artifacts/research_steps/S07/goal_uncertainty.parquet"),
            Path("/artifacts/research_steps/S07/nearest_neighbors.parquet"),
        )
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("S01/S05/S06/S07 artifacts are required")
        cls.frame = pd.read_parquet(required[0])
        cls.world_catalog = pd.read_parquet(required[1])
        cls.error_limits = pd.read_parquet(required[2])
        cls.target_columns = json.loads(required[3].read_text(encoding="utf-8")).get("targetColumns", [])
        cls.policy_embeddings = pd.read_parquet(required[4])
        cls.policy_coverage = pd.read_parquet(required[5])
        cls.policy_neighbors = pd.read_parquet(required[6])
        cls.goal_embeddings = pd.read_parquet(required[7])
        cls.goal_uncertainty = pd.read_parquet(required[8])
        cls.goal_neighbors = pd.read_parquet(required[9])
        cls.target_weights = target_error_weights(cls.frame, cls.error_limits, cls.target_columns)
        cls.world_profiles, cls.world_coverage = build_world_profiles(
            cls.frame,
            cls.world_catalog,
            cls.policy_embeddings,
            cls.goal_embeddings,
            cls.target_weights,
            cls.target_columns,
        )
        cls.world_embeddings, cls.world_model, _world_feature_matrix = fit_world_embedding(cls.world_profiles)

    def test_world_profiles_and_claim_boundary(self) -> None:
        self.assertTrue(PLATONIC_DISTANCE_CLAIM_BOUNDARY.startswith("Empirical computational distances"))
        self.assertGreaterEqual(len(self.world_profiles), 30)
        self.assertGreaterEqual(len(self.world_embeddings), 20)
        self.assertIn("s06_policy::embedding_00::mean", " ".join(self.world_model["featureColumns"]))
        self.assertIn("s07_goal::embedding_00::mean", " ".join(self.world_model["featureColumns"]))
        self.assertTrue((self.target_weights["s05ReliabilityWeight"].between(0, 1)).all())

    def test_pairwise_distances_and_metric_diagnostics(self) -> None:
        policy_penalty = policy_sparse_penalty(self.policy_embeddings, self.policy_coverage)
        goal_penalty = goal_sparse_penalty(self.goal_embeddings, self.goal_uncertainty)
        world_penalty = world_sparse_penalty(self.world_embeddings, self.world_coverage)

        policy_distances, policy_entities, policy_matrix = pairwise_distance_tables(
            self.policy_embeddings, "policy", "abstractPolicyId", POLICY_LABEL_COLUMNS, policy_penalty
        )
        goal_distances, goal_entities, goal_matrix = pairwise_distance_tables(
            self.goal_embeddings, "goal", "abstractGoalId", GOAL_LABEL_COLUMNS, goal_penalty
        )
        world_distances, world_entities, world_matrix = pairwise_distance_tables(
            self.world_embeddings, "world", "worldId", WORLD_LABEL_COLUMNS, world_penalty
        )
        distances = pd.concat([policy_distances, goal_distances, world_distances], ignore_index=True)
        entities = pd.concat([policy_entities, goal_entities, world_entities], ignore_index=True)

        diagnostics = pd.concat(
            [
                symmetry_triangle_diagnostics(
                    "policy", self.policy_embeddings.sort_values("abstractPolicyId")["abstractPolicyId"].astype(str).tolist(), policy_matrix
                ),
                symmetry_triangle_diagnostics("goal", self.goal_embeddings.sort_values("abstractGoalId")["abstractGoalId"].astype(str).tolist(), goal_matrix),
                symmetry_triangle_diagnostics("world", self.world_embeddings.sort_values("worldId")["worldId"].astype(str).tolist(), world_matrix),
            ],
            ignore_index=True,
        )

        self.assertGreater(len(distances), 100000)
        self.assertEqual(set(distances["entityType"]), {"policy", "goal", "world"})
        self.assertTrue(np.isfinite(distances["platonicDistance"].to_numpy(dtype=float)).all())
        self.assertTrue((diagnostics["maxSymmetryAbsError"] <= 1e-10).all())
        self.assertTrue((diagnostics["maxDiagonalAbsDistance"] <= 1e-10).all())
        self.assertTrue((diagnostics["triangleViolationRate"] <= 1e-8).all())
        self.assertEqual(int(diagnostics["triangleViolationCount"].sum()), 0)

        policy_nearest = nearest_neighbor_table(self.policy_embeddings, "policy", "abstractPolicyId", POLICY_LABEL_COLUMNS, policy_matrix)
        goal_nearest = nearest_neighbor_table(self.goal_embeddings, "goal", "abstractGoalId", GOAL_LABEL_COLUMNS, goal_matrix)
        world_nearest = nearest_neighbor_table(self.world_embeddings, "world", "worldId", WORLD_LABEL_COLUMNS, world_matrix)
        nearest = pd.concat([policy_nearest, goal_nearest, world_nearest], ignore_index=True)
        sanity = pd.concat(
            [
                nearest_neighbor_sanity(nearest, "policy", BASELINE_LABEL_COLUMNS["policy"]),
                nearest_neighbor_sanity(nearest, "goal", BASELINE_LABEL_COLUMNS["goal"]),
                nearest_neighbor_sanity(nearest, "world", BASELINE_LABEL_COLUMNS["world"]),
                prior_neighbor_overlap(nearest, self.policy_neighbors, "policy"),
                prior_neighbor_overlap(nearest, self.goal_neighbors, "goal"),
            ],
            ignore_index=True,
        )
        baselines = distance_baseline_comparison(distances, BASELINE_LABEL_COLUMNS)
        validation = validate_platonic_distances(distances, entities, diagnostics, sanity, baselines, self.world_embeddings)
        hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

        self.assertFalse(sanity.empty)
        self.assertFalse(baselines.empty)
        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
