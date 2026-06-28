from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from chimera.compatibility import (
    CORE_SCORE_COLUMNS,
    REQUIRED_COMPONENTS,
    build_control_anchor_validation,
    compute_compatibility_metrics,
    load_optional_table,
    load_s04_results,
    metric_definitions,
    validation_checks,
)


class TestE06CompatibilityMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s04 = load_s04_results()
        cls.metrics = compute_compatibility_metrics(cls.s04)
        cls.definitions = metric_definitions()
        cls.pure = load_optional_table(Path("/artifacts/research_steps/S01/pure_policy_validation.parquet"))
        cls.dummy_diag = load_optional_table(Path("/previous-artifacts/E02/results/e02_dummy_algotypes_diagnostics.parquet"))
        cls.dummy_summary = load_optional_table(Path("/previous-artifacts/E02/results/e02_dummy_algotypes_summary.parquet"))
        cls.anchors = build_control_anchor_validation(cls.metrics, cls.pure, cls.dummy_diag, cls.dummy_summary)

    def test_metric_definitions_cover_required_components(self) -> None:
        self.assertTrue(set(REQUIRED_COMPONENTS) <= set(self.definitions["componentName"]))
        self.assertEqual(len(self.definitions), len(REQUIRED_COMPONENTS))

    def test_vector_has_one_bounded_row_per_s04_condition(self) -> None:
        self.assertEqual(len(self.metrics), len(self.s04))
        self.assertTrue(self.metrics["conditionId"].is_unique)
        for column in CORE_SCORE_COLUMNS:
            values = pd.to_numeric(self.metrics[column], errors="coerce")
            self.assertFalse(values.isna().any(), column)
            self.assertTrue(values.between(0.0, 1.0).all(), column)

    def test_same_goal_and_opposite_goal_anchor_behavior(self) -> None:
        same = self.metrics[self.metrics["goalMode"] == "same_goal"]
        opposite = self.metrics[self.metrics["goalMode"] == "opposite_goal"]
        self.assertFalse(same.empty)
        self.assertFalse(opposite.empty)
        self.assertTrue((same["cooperativeEfficiencyScore"] == 1.0).all())
        self.assertTrue((same["mutualInterferenceScore"] == 0.0).all())
        self.assertGreater(opposite["dominanceProxyScore"].mean(), same["dominanceProxyScore"].mean())
        self.assertGreater(opposite["mutualInterferenceScore"].mean(), same["mutualInterferenceScore"].mean())

    def test_control_anchors_and_validation_pass(self) -> None:
        self.assertTrue(self.anchors["success"].all(), self.anchors.to_string())
        checks = validation_checks(self.metrics, self.definitions, self.anchors, self.s04)
        self.assertTrue(checks["success"].all(), checks.to_string())


if __name__ == "__main__":
    unittest.main(verbosity=2)
