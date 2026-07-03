"""Tests for E07 S11 counterfactual helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.e07.counterfactual_schema import (
    POSITIVE_CLASS_STATUS,
    capability_target_for_record,
    class_status_bucket,
    feature_dicts,
    grouped_median_predictions,
    stable_record_hash,
    timestamp_order_ok,
    validate_counterfactual_artifacts,
)


class CounterfactualSchemaTests(unittest.TestCase):
    def test_capability_target_for_record(self) -> None:
        self.assertEqual(capability_target_for_record({"metric_family": "aggregation"}), "high_aggregation")
        self.assertEqual(capability_target_for_record({"source_metric_name": "mean_dg_primary"}), "high_dg")
        self.assertEqual(capability_target_for_record({"source_metric_name": "target_recovery_fraction"}), "high_repair_robustness")
        self.assertEqual(capability_target_for_record({"metric_family": "sorting_order"}), "high_order_quality")
        self.assertEqual(capability_target_for_record({"metric_family": "prediction"}), "other")

    def test_class_status_bucket(self) -> None:
        self.assertEqual(class_status_bucket(POSITIVE_CLASS_STATUS), "positive_bounded")
        self.assertEqual(class_status_bucket("source_dominated_constraint"), "caution_constraining")
        self.assertEqual(class_status_bucket("missingness_driven_constraint"), "caution_constraining")
        self.assertEqual(class_status_bucket(""), "unclassified")

    def test_grouped_median_predictions(self) -> None:
        train = pd.DataFrame(
            {
                "source_experiment_id": ["E01", "E01", "E02"],
                "metric_family": ["a", "a", "b"],
                "target_transformed": [1.0, 3.0, 10.0],
            }
        )
        test = pd.DataFrame(
            {
                "source_experiment_id": ["E01", "E02", "E03"],
                "metric_family": ["a", "b", "z"],
            }
        )
        pred = grouped_median_predictions(train, test, ("source_experiment_id", "metric_family"))
        np.testing.assert_allclose(pred, [2.0, 10.0, 3.0])

    def test_feature_dicts_and_hash_stability(self) -> None:
        frame = pd.DataFrame({"source_experiment_id": ["E01"], "has_world_link": [1.0]})
        features = feature_dicts(frame)
        self.assertEqual(len(features), 1)
        self.assertIn("source_experiment_id=E01", features[0])
        self.assertIn("num:has_world_link", features[0])
        first = stable_record_hash({"b": 2, "a": 1})
        second = stable_record_hash({"a": 1, "b": 2})
        self.assertEqual(first, second)

    def test_validate_counterfactual_artifacts(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "prediction_id": "p1",
                    "candidate_role": "positive_candidate",
                    "capability_target": "high_aggregation",
                    "policy_class_status": POSITIVE_CLASS_STATUS,
                    "world_class_status": "unclassified",
                    "goal_class_status": POSITIVE_CLASS_STATUS,
                    "source_metric_median_prediction": 1.0,
                    "metric_family_median_prediction": 1.0,
                    "missingness_pattern_median_prediction": 1.0,
                    "class_status_median_prediction": 1.0,
                },
                {
                    "prediction_id": "p2",
                    "candidate_role": "caution_control",
                    "capability_target": "high_dg",
                    "policy_class_status": POSITIVE_CLASS_STATUS,
                    "world_class_status": "source_dominated_constraint",
                    "goal_class_status": "unclassified",
                    "source_metric_median_prediction": 1.0,
                    "metric_family_median_prediction": 1.0,
                    "missingness_pattern_median_prediction": 1.0,
                    "class_status_median_prediction": 1.0,
                },
            ]
        )
        validations = pd.DataFrame(
            [
                {"prediction_id": "p1", "validation_kind": "heldout_retrospective", "observed_target_transformed": 1.2},
                {"prediction_id": "p2", "validation_kind": "fresh_simulator_blocker", "observed_target_transformed": np.nan},
            ]
        )
        manifest = {
            "predictionArtifactSha256": "abc",
            "frozenAtUtc": "2026-07-03T00:00:00+00:00",
            "validationStartedAtUtc": "2026-07-03T00:01:00+00:00",
            "selectionGaps": [{"capability_target": "high_repair_robustness"}],
        }
        checks = validate_counterfactual_artifacts(predictions, validations, manifest)
        self.assertTrue(timestamp_order_ok(manifest["frozenAtUtc"], manifest["validationStartedAtUtc"]))
        self.assertEqual(int(checks["success"].sum()), len(checks))


if __name__ == "__main__":
    unittest.main()
