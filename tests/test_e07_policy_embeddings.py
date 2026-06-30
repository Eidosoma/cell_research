from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import joblib
import pandas as pd

from platonic_space.policy_embeddings import (
    POLICY_EMBEDDING_CLAIM_BOUNDARY,
    build_policy_profiles,
    family_baseline_matrix,
    feature_columns,
    fit_behavior_embedding,
    heldout_behavior_retrieval_metrics,
    heldout_truth_columns,
    label_retrieval_metrics,
    source_code_baseline_matrix,
    split_behavior_frame,
    stability_metrics,
    target_error_weights,
    validate_policy_embeddings,
)


class TestE07PolicyEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            Path("/artifacts/research_steps/S05/modeling_frame.parquet"),
            Path("/artifacts/research_steps/S05/error_limits.parquet"),
            Path("/artifacts/research_steps/S02/policy_abstract_catalog.parquet"),
            Path("/artifacts/models/e07_behavior_predictor/model_card.json"),
        )
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("S05 modeling frame, S05 error limits, and S02 policy catalog are required")
        cls.frame = pd.read_parquet(required[0])
        cls.error_limits = pd.read_parquet(required[1])
        cls.policy_catalog = pd.read_parquet(required[2])
        cls.target_columns = json.loads(required[3].read_text(encoding="utf-8")).get("targetColumns", [])
        if not cls.target_columns:
            cls.target_columns = [
                "final_sortedness_percent",
                "completed",
                "aggregation",
            ]
        cls.weights = target_error_weights(cls.frame, cls.error_limits, cls.target_columns)
        cls.profiles, cls.coverage = build_policy_profiles(cls.frame, cls.policy_catalog, cls.target_columns, cls.weights)
        cls.embedded = cls.profiles[cls.profiles["embeddingStatus"].eq("embedded_behavior")].sort_values("abstractPolicyId").reset_index(drop=True)

    def test_target_weights_and_profile_coverage(self) -> None:
        self.assertTrue(POLICY_EMBEDDING_CLAIM_BOUNDARY.startswith("Computational policy embeddings"))
        self.assertIn("final_sortedness_percent", set(self.weights["target"]))
        self.assertTrue((self.weights["s05ReliabilityWeight"].between(0, 1)).all())
        self.assertGreaterEqual(len(self.embedded), 500)
        self.assertGreaterEqual((self.coverage["embeddingStatus"].eq("unobserved_in_s04_primary_policy")).sum(), 1000)

    def test_behavior_embedding_and_label_metrics(self) -> None:
        columns = feature_columns(self.embedded)
        embeddings, model, matrix = fit_behavior_embedding(self.embedded, columns, n_components=6)
        family_matrix, _ = family_baseline_matrix(self.policy_catalog, embeddings["abstractPolicyId"].tolist(), n_components=6)
        label_metrics = label_retrieval_metrics(embeddings["abstractPolicyId"].tolist(), matrix, self.embedded, "weighted_behavior_pca")

        self.assertEqual(len(embeddings), len(self.embedded))
        self.assertGreater(len(columns), 20)
        self.assertEqual(matrix.shape[0], len(self.embedded))
        self.assertEqual(len(model["explainedVarianceRatio"]), matrix.shape[1])
        self.assertFalse(label_metrics.empty)
        self.assertEqual(family_matrix.shape[0], len(self.embedded))

    def test_heldout_retrieval_stability_and_validation(self) -> None:
        columns = feature_columns(self.embedded)
        train_frame, holdout_frame = split_behavior_frame(self.frame)
        train_profiles, _ = build_policy_profiles(train_frame, self.policy_catalog, self.target_columns, self.weights, min_policy_rows=1)
        holdout_profiles, _ = build_policy_profiles(holdout_frame, self.policy_catalog, self.target_columns, self.weights, min_policy_rows=1)
        train_index = train_profiles.set_index("abstractPolicyId")
        holdout_index = holdout_profiles.set_index("abstractPolicyId")
        policy_ids = [
            policy_id
            for policy_id in self.embedded["abstractPolicyId"].tolist()
            if policy_id in train_index.index
            and policy_id in holdout_index.index
            and int(train_index.loc[policy_id, "rowCount"]) >= 5
            and int(holdout_index.loc[policy_id, "rowCount"]) >= 5
        ][:120]
        self.assertGreater(len(policy_ids), 50)
        train_common = train_index.loc[policy_ids].reset_index()
        holdout_common = holdout_index.loc[policy_ids].reset_index()
        _train_embeddings, _train_model, train_matrix = fit_behavior_embedding(train_common, columns, n_components=6)
        _holdout_embeddings, _holdout_model, holdout_matrix = fit_behavior_embedding(holdout_common, columns, n_components=6)
        family_matrix, _ = family_baseline_matrix(self.policy_catalog, policy_ids, n_components=6)
        source_matrix, _ = source_code_baseline_matrix(self.policy_catalog, policy_ids, n_components=6)
        truth_columns = heldout_truth_columns(holdout_common)
        heldout_metrics = heldout_behavior_retrieval_metrics(
            policy_ids,
            {
                "weighted_behavior_pca": train_matrix,
                "family_metadata_baseline": family_matrix,
                "source_code_baseline": source_matrix,
            },
            holdout_common,
            truth_columns,
            ks=(5,),
        )
        stability = stability_metrics(policy_ids, train_matrix, holdout_matrix, k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "embedding_model.joblib"
            joblib.dump({"ok": True}, model_path)
            embeddings, _model, _matrix = fit_behavior_embedding(self.embedded, columns, n_components=6)
            validation = validate_policy_embeddings(embeddings, self.coverage, heldout_metrics, stability, model_path)
        hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

        self.assertFalse(heldout_metrics.empty)
        self.assertFalse(stability.empty)
        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
