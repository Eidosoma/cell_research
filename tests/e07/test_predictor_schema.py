"""Tests for E07 S05 behavior-predictor helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.e07.predictor_schema import (
    assign_group_holdout,
    calibration_bins,
    metrics_for_predictions,
    split_summary,
    stable_fraction,
    transformed_target,
    validate_prediction_artifacts,
)


class PredictorSchemaTests(unittest.TestCase):
    def test_stable_fraction_is_deterministic_and_bounded(self) -> None:
        left = stable_fraction("policy-a", salt="unit")
        right = stable_fraction("policy-a", salt="unit")
        other = stable_fraction("policy-b", salt="unit")
        self.assertEqual(left, right)
        self.assertNotEqual(left, other)
        self.assertGreaterEqual(left, 0.0)
        self.assertLess(left, 1.0)

    def test_transformed_target_orients_minimize_metrics(self) -> None:
        self.assertGreater(transformed_target(9.0, "maximize"), 0.0)
        self.assertLess(transformed_target(9.0, "minimize"), 0.0)
        self.assertEqual(transformed_target(0.0, "maximize"), 0.0)

    def test_holdout_split_has_no_group_overlap(self) -> None:
        frame = pd.DataFrame({"group": ["a", "a", "b", "c", "d", "e"]})
        test_mask = assign_group_holdout(frame, group_column="group", split_name="unit", test_fraction=0.4)
        summary = split_summary(frame, split_name="unit", group_column="group", test_mask=test_mask)
        self.assertEqual(summary["leaked_groups"], 0)
        self.assertGreater(summary["test_groups"], 0)
        self.assertGreater(summary["train_groups"], 0)

    def test_metrics_and_calibration_bins_are_finite(self) -> None:
        y_true = np.array([0.0, 1.0, 2.0, 3.0])
        y_pred = np.array([0.1, 1.2, 1.8, 2.7])
        metrics = metrics_for_predictions(y_true, y_pred)
        self.assertLess(metrics["mae"], 0.3)
        bins = calibration_bins(y_true, y_pred, n_bins=2)
        self.assertEqual(int(bins["row_count"].sum()), 4)

    def test_prediction_artifact_validation(self) -> None:
        metrics = pd.DataFrame(
            [
                {"split_name": "heldout_policy", "model_name": "global_median", "mae": 1.0, "rmse": 1.2},
                {"split_name": "heldout_world", "model_name": "source_metric_median", "mae": 0.9, "rmse": 1.1},
                {"split_name": "heldout_goal", "model_name": "hashed_linear_sgd", "mae": 0.8, "rmse": 1.0},
                {"split_name": "heldout_perturbation", "model_name": "neural_embedding_mlp", "mae": 0.7, "rmse": 0.9},
            ]
        )
        split_summaries = pd.DataFrame(
            [
                {"split_name": "heldout_policy", "leaked_groups": 0},
                {"split_name": "heldout_world", "leaked_groups": 0},
                {"split_name": "heldout_goal", "leaked_groups": 0},
                {"split_name": "heldout_perturbation", "leaked_groups": 0},
            ]
        )
        checks = validate_prediction_artifacts(metrics, split_summaries)
        self.assertTrue(bool(checks["success"].all()))


if __name__ == "__main__":
    unittest.main()
