from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from platonic_space.universality_classes import (
    S09_CONSTRAINING_CAVEAT,
    UNIVERSALITY_CLASS_CLAIM_BOUNDARY,
    build_policy_taxonomy_features,
    build_taxonomy_tables,
    build_upstream_label_table,
    policy_distance_matrix,
    select_cluster_count,
    stability_tables,
    upstream_label_agreement,
    validation_table,
)


class TestE07UniversalityClasses(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            Path("/artifacts/research_steps/S06/policy_embeddings.parquet"),
            Path("/artifacts/research_steps/S06/policy_behavior_profiles.parquet"),
            Path("/artifacts/research_steps/S07/goal_embeddings.parquet"),
            Path("/artifacts/research_steps/S08/platonic_distances.parquet"),
            Path("/artifacts/research_steps/S08/distance_entities.parquet"),
            Path("/artifacts/research_steps/S09/invariant_feature_frame.parquet"),
            Path("/artifacts/research_steps/S09/invariant_candidates.parquet"),
            Path("/artifacts/research_steps/S02/policy_abstract_catalog.parquet"),
        )
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("S02 and S06-S09 artifacts are required")
        cls.policy_embeddings = pd.read_parquet(required[0])
        cls.policy_profiles = pd.read_parquet(required[1])
        cls.goal_embeddings = pd.read_parquet(required[2])
        cls.s08_distances = pd.read_parquet(required[3])
        cls.s08_entities = pd.read_parquet(required[4])
        cls.s09_feature_frame = pd.read_parquet(required[5])
        cls.s09_candidates = pd.read_parquet(required[6])
        cls.policy_catalog = pd.read_parquet(required[7])
        cls.e03_assignments = pd.read_parquet("/previous-artifacts/E03/research_steps/S11/policy_class_assignments.parquet")
        cls.e06_panel = pd.read_parquet("/previous-artifacts/E06/research_steps/S01/algotype_panel.parquet")
        cls.e06_dominance = pd.read_parquet("/previous-artifacts/E06/results/e06_dominance_hierarchy.parquet")
        cls.e06_mosaic = pd.read_parquet("/previous-artifacts/E06/results/e06_mosaic_classifications.parquet")

    def test_builds_non_degenerate_taxonomy(self) -> None:
        self.assertTrue(UNIVERSALITY_CLASS_CLAIM_BOUNDARY.startswith("Empirical computational taxonomy"))
        self.assertIn("no robust cross-world invariant", S09_CONSTRAINING_CAVEAT)
        upstream_labels, label_audit = build_upstream_label_table(
            self.policy_catalog,
            self.policy_embeddings["abstractPolicyId"].astype(str).tolist(),
            e03_assignments=self.e03_assignments,
            e06_panel=self.e06_panel,
            e06_dominance=self.e06_dominance,
            e06_mosaic=self.e06_mosaic,
        )
        policies, matrix, feature_sources = build_policy_taxonomy_features(
            self.policy_embeddings,
            self.goal_embeddings,
            self.s09_feature_frame,
            self.policy_catalog,
            self.s08_entities,
            upstream_labels,
            policy_behavior_profiles=self.policy_profiles,
        )
        s08_matrix = policy_distance_matrix(self.s08_distances, policies["abstractPolicyId"].astype(str).tolist())
        selected_k, labels, cluster_selection = select_cluster_count(matrix, s08_matrix, cluster_counts=range(6, 11))
        assignments, class_summary, exemplars = build_taxonomy_tables(policies, matrix, s08_matrix, labels, selected_k)
        stability = stability_tables(matrix, s08_matrix, labels, selected_k, resamples=4)
        agreement = upstream_label_agreement(assignments)
        validation = validation_table(assignments, class_summary, stability, agreement, cluster_selection, self.s09_candidates)
        hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

        self.assertEqual(len(assignments), len(self.policy_embeddings))
        self.assertGreaterEqual(class_summary["universalityClassId"].nunique(), 4)
        self.assertGreaterEqual(class_summary["policyCount"].min(), 5)
        self.assertFalse(exemplars.empty)
        self.assertGreater(feature_sources["s06PolicyEmbeddingColumns"].__len__(), 0)
        self.assertGreater(feature_sources["s07GoalContextColumns"].__len__(), 0)
        self.assertGreaterEqual(label_audit["embeddedPolicyMatches"].max(), 400)
        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
