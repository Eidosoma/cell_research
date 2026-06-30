from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib

from platonic_space.behavior_predictor import (
    PREDICTOR_CLAIM_BOUNDARY,
    SPLIT_GROUP_COLUMNS,
    build_modeling_frame,
    selected_targets,
    target_coverage,
    train_evaluate_predictor,
    validation_checks,
)


class TestE07BehaviorPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            Path("/artifacts/data/e07_unified_behavior_corpus.parquet"),
            Path("/artifacts/research_steps/S01/normalized_world_catalog.parquet"),
            Path("/artifacts/research_steps/S02/policy_abstract_catalog.parquet"),
            Path("/artifacts/research_steps/S03/goal_catalog.parquet"),
        )
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("S04 corpus plus S01/S02/S03 catalogs are required")
        cls.frame = build_modeling_frame()

    def test_modeling_frame_has_targets_and_partitions(self) -> None:
        coverage = target_coverage(self.frame)
        targets = selected_targets(coverage, min_rows=500)

        self.assertGreater(len(self.frame), 10_000)
        self.assertIn("final_sortedness_percent", targets)
        self.assertTrue(PREDICTOR_CLAIM_BOUNDARY.startswith("Computational surrogate model"))
        for split_name in SPLIT_GROUP_COLUMNS:
            partitions = set(self.frame[f"{split_name}Partition"])
            self.assertIn("train", partitions)
            self.assertIn("test", partitions)

    def test_group_heldout_splits_are_disjoint(self) -> None:
        for split_name, group_column in SPLIT_GROUP_COLUMNS.items():
            if split_name == "random":
                continue
            partition_column = f"{split_name}Partition"
            train_groups = set(self.frame.loc[self.frame[partition_column].eq("train"), group_column].astype(str))
            validation_groups = set(self.frame.loc[self.frame[partition_column].eq("validation"), group_column].astype(str))
            test_groups = set(self.frame.loc[self.frame[partition_column].eq("test"), group_column].astype(str))

            self.assertTrue(train_groups.isdisjoint(test_groups), split_name)
            self.assertTrue(train_groups.isdisjoint(validation_groups), split_name)

    def test_sparse_ridge_predictor_smoke(self) -> None:
        frame = self.frame.head(12_000).copy()
        coverage = target_coverage(frame)
        targets = selected_targets(coverage, min_rows=800)[:2]
        self.assertGreaterEqual(len(targets), 1)

        metrics, _calibration, residuals, model_bundle, split_manifest = train_evaluate_predictor(
            frame,
            targets,
            alphas=(1.0,),
            min_train_rows=50,
            min_eval_rows=10,
        )

        evaluated_models = set(metrics.loc[metrics["status"].eq("evaluated"), "modelName"])
        self.assertIn("sparse_ridge", evaluated_models)
        self.assertIn("global_mean", evaluated_models)
        self.assertIn("source_group_mean", evaluated_models)
        self.assertFalse(residuals.empty)
        self.assertEqual(set(model_bundle["targetColumns"]), set(targets))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model_bundle.joblib"
            joblib.dump(model_bundle, model_path)
            validation = validation_checks(frame, coverage, metrics, split_manifest, model_path)
        hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
