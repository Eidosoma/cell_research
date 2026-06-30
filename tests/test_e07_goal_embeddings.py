from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from platonic_space.goal_embeddings import (
    GOAL_EMBEDDING_CLAIM_BOUNDARY,
    build_goal_profiles,
    feature_columns,
    fit_goal_embedding,
    goal_metadata_baseline_matrix,
    heldout_goal_retrieval_metrics,
    heldout_truth_columns,
    label_retrieval_metrics,
    policy_neighborhood_baseline_matrix,
    split_behavior_frame,
    stability_metrics,
    target_error_weights,
    validate_goal_embeddings,
)


class TestE07GoalEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            Path("/artifacts/research_steps/S05/modeling_frame.parquet"),
            Path("/artifacts/data/e07_unified_behavior_corpus.parquet"),
            Path("/artifacts/research_steps/S03/goal_catalog.parquet"),
            Path("/artifacts/research_steps/S05/error_limits.parquet"),
            Path("/artifacts/models/e07_behavior_predictor/model_card.json"),
            Path("/artifacts/research_steps/S06/policy_embeddings.parquet"),
            Path("/artifacts/research_steps/S06/nearest_neighbors.parquet"),
        )
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("S04/S05/S06 artifacts and S03 goal catalog are required")
        cls.frame = pd.read_parquet(required[0])
        cls.corpus = pd.read_parquet(required[1])
        cls.goal_catalog = pd.read_parquet(required[2])
        cls.error_limits = pd.read_parquet(required[3])
        cls.target_columns = json.loads(required[4].read_text(encoding="utf-8")).get("targetColumns", [])
        cls.policy_embeddings = pd.read_parquet(required[5])
        cls.policy_neighbors = pd.read_parquet(required[6])
        cls.weights = target_error_weights(cls.frame, cls.error_limits, cls.target_columns)
        cls.profiles, cls.coverage, cls.uncertainty = build_goal_profiles(
            cls.frame,
            cls.goal_catalog,
            cls.policy_embeddings,
            cls.policy_neighbors,
            cls.target_columns,
            cls.weights,
            corpus=cls.corpus,
        )
        cls.embedded = cls.profiles[cls.profiles["embeddingStatus"].eq("embedded_behavior")].sort_values("abstractGoalId").reset_index(drop=True)

    def test_goal_coverage_and_uncertainty(self) -> None:
        self.assertTrue(GOAL_EMBEDDING_CLAIM_BOUNDARY.startswith("Computational goal embeddings"))
        self.assertEqual(len(self.coverage), len(self.goal_catalog))
        self.assertGreaterEqual(len(self.embedded), 30)
        self.assertGreaterEqual((self.coverage["embeddingStatus"].eq("unobserved_in_s04")).sum(), 5)
        self.assertGreaterEqual((self.uncertainty["sparseUncertaintyLevel"].eq("high")).sum(), 5)

    def test_goal_embedding_and_baselines(self) -> None:
        columns = feature_columns(self.embedded)
        embeddings, model, matrix = fit_goal_embedding(self.embedded, columns, n_components=6)
        metadata_matrix, _ = goal_metadata_baseline_matrix(self.goal_catalog, embeddings["abstractGoalId"].tolist(), n_components=6)
        policy_matrix, _ = policy_neighborhood_baseline_matrix(self.embedded, n_components=6)
        label_metrics = label_retrieval_metrics(embeddings["abstractGoalId"].tolist(), matrix, self.embedded, "weighted_goal_behavior_pca")

        self.assertEqual(len(embeddings), len(self.embedded))
        self.assertGreater(len(columns), 20)
        self.assertEqual(matrix.shape[0], len(self.embedded))
        self.assertEqual(len(model["explainedVarianceRatio"]), matrix.shape[1])
        self.assertEqual(metadata_matrix.shape[0], len(self.embedded))
        self.assertEqual(policy_matrix.shape[0], len(self.embedded))
        self.assertFalse(label_metrics.empty)

    def test_heldout_retrieval_stability_and_validation(self) -> None:
        columns = feature_columns(self.embedded)
        train_frame, holdout_frame = split_behavior_frame(self.frame)
        train_profiles, _train_cov, _train_unc = build_goal_profiles(
            train_frame,
            self.goal_catalog,
            self.policy_embeddings,
            self.policy_neighbors,
            self.target_columns,
            self.weights,
            corpus=self.corpus,
            min_goal_rows=1,
        )
        holdout_profiles, _holdout_cov, _holdout_unc = build_goal_profiles(
            holdout_frame,
            self.goal_catalog,
            self.policy_embeddings,
            self.policy_neighbors,
            self.target_columns,
            self.weights,
            corpus=self.corpus,
            min_goal_rows=1,
        )
        train_index = train_profiles.set_index("abstractGoalId")
        holdout_index = holdout_profiles.set_index("abstractGoalId")
        goal_ids = [
            goal_id
            for goal_id in self.embedded["abstractGoalId"].tolist()
            if goal_id in train_index.index
            and goal_id in holdout_index.index
            and int(train_index.loc[goal_id, "rowCount"]) >= 5
            and int(holdout_index.loc[goal_id, "rowCount"]) >= 5
            and int(train_index.loc[goal_id, "observedTargetCount"]) > 0
            and int(holdout_index.loc[goal_id, "observedTargetCount"]) > 0
        ]
        self.assertGreaterEqual(len(goal_ids), 20)
        train_common = train_index.loc[goal_ids].reset_index()
        holdout_common = holdout_index.loc[goal_ids].reset_index()
        _train_embeddings, _train_model, train_matrix = fit_goal_embedding(train_common, columns, n_components=6)
        _holdout_embeddings, _holdout_model, holdout_matrix = fit_goal_embedding(holdout_common, columns, n_components=6)
        metadata_matrix, _ = goal_metadata_baseline_matrix(self.goal_catalog, goal_ids, n_components=6)
        policy_matrix, _ = policy_neighborhood_baseline_matrix(train_common, n_components=6)
        truth_columns = heldout_truth_columns(holdout_common)
        heldout_metrics = heldout_goal_retrieval_metrics(
            goal_ids,
            {
                "weighted_goal_behavior_pca": train_matrix,
                "goal_metadata_baseline": metadata_matrix,
                "policy_neighborhood_baseline": policy_matrix,
            },
            holdout_common,
            truth_columns,
            ks=(3,),
        )
        stability = stability_metrics(goal_ids, train_matrix, holdout_matrix, k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "goal_embedding_model.joblib"
            joblib.dump({"ok": True}, model_path)
            embeddings, _model, _matrix = fit_goal_embedding(self.embedded, columns, n_components=6)
            validation = validate_goal_embeddings(embeddings, self.coverage, self.uncertainty, heldout_metrics, stability, model_path)
        hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

        self.assertFalse(heldout_metrics.empty)
        self.assertFalse(stability.empty)
        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
