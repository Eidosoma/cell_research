"""E05 S14 shape delayed-gratification tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.shape_delayed_gratification import (  # noqa: E402
    build_shape_dg_tables,
    compute_delayed_gratification_metrics,
    matched_increment_null_summary,
)


class ShapeDelayedGratificationTests(unittest.TestCase):
    def test_constructive_drawup_requires_later_larger_improvement(self) -> None:
        metrics = compute_delayed_gratification_metrics([1.0, 0.7, 0.9, 0.2], [0, 1, 2, 3])

        self.assertTrue(metrics["dg_detected"])
        self.assertAlmostEqual(metrics["max_prefix_drawup"], 0.2)
        self.assertGreater(metrics["post_drawup_repair"], metrics["max_prefix_drawup"])
        self.assertAlmostEqual(metrics["dg_score"], 0.2)

    def test_monotone_improvement_has_no_dg(self) -> None:
        metrics = compute_delayed_gratification_metrics([1.0, 0.8, 0.4], [0, 1, 2])

        self.assertFalse(metrics["dg_detected"])
        self.assertTrue(metrics["monotone_nonincreasing"])
        self.assertEqual(metrics["temporary_away_steps"], 0)
        self.assertEqual(metrics["dg_score"], 0.0)

    def test_null_permutation_preserves_length_and_endpoint(self) -> None:
        values = [1.0, 0.7, 0.9, 0.2]
        observed = compute_delayed_gratification_metrics(values)
        nulls = matched_increment_null_summary(values, observed["dg_score"], n_null=25, seed=3)

        self.assertEqual(nulls["null_match_status"], "matched")
        self.assertEqual(nulls["null_length_match_rate"], 1.0)
        self.assertLessEqual(nulls["max_null_endpoint_abs_error"], 1e-12)
        self.assertGreaterEqual(nulls["null_p_value"], 0.0)
        self.assertLessEqual(nulls["null_p_value"], 1.0)

    def test_artifact_backed_s14_builder_smoke_when_inputs_exist(self) -> None:
        artifacts_dir = Path("/artifacts")
        if not (artifacts_dir / "traces" / "e05_regeneration_trace_table.parquet").exists():
            self.skipTest("S08 artifacts are not mounted")

        tables = build_shape_dg_tables(artifacts_dir, n_null=9)

        self.assertEqual(len(tables["shape_dg"]), 360)
        self.assertEqual(tables["shape_dg"]["metric_id"].nunique(), 2)
        self.assertEqual(len(tables["metric_sensitivity"]), 180)
        self.assertTrue(tables["validation"]["success"].all())


if __name__ == "__main__":
    unittest.main()

