"""E05 S15 morphology benchmark-suite smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.benchmark_suite import (  # noqa: E402
    STANDARD_TASK_SPECS,
    benchmark_config_payloads,
    build_benchmark_suite_tables,
)


class BenchmarkSuiteTests(unittest.TestCase):
    def test_standard_task_ids_match_declared_suite(self) -> None:
        task_ids = {spec["benchmark_task_id"] for spec in STANDARD_TASK_SPECS}

        self.assertEqual(
            task_ids,
            {
                "sort_row_embedded_1d",
                "restore_gradient_scramble",
                "repair_hole_missing_patch",
                "recover_boundary_scramble",
                "regenerate_limb_like_appendage",
                "chimeric_pattern_conflict",
            },
        )

    def test_config_payloads_have_required_schema_fields(self) -> None:
        artifacts_dir = Path("/artifacts")
        if not (artifacts_dir / "results" / "e05_local_vs_global_control.parquet").exists():
            self.skipTest("E05 artifacts are not mounted")
        tables = build_benchmark_suite_tables(artifacts_dir)
        payloads = benchmark_config_payloads(tables["task_catalog"])

        self.assertEqual(len(payloads), 6)
        for task_id, payload in payloads.items():
            self.assertEqual(payload["schema"], "eidosoma.e05.benchmark_task_config.v1")
            self.assertEqual(payload["benchmarkTaskId"], task_id)
            self.assertTrue(payload["target"]["targetId"])
            self.assertTrue(payload["metrics"]["metricIds"])
            self.assertGreater(payload["referenceOutputs"]["observedReferenceRows"], 0)

    def test_artifact_backed_builder_produces_reference_rows_and_smoke_passes(self) -> None:
        artifacts_dir = Path("/artifacts")
        if not (artifacts_dir / "results" / "e05_1d_embedded_run_summary.parquet").exists():
            self.skipTest("E05 artifacts are not mounted")

        tables = build_benchmark_suite_tables(artifacts_dir)

        self.assertEqual(len(tables["task_catalog"]), 6)
        self.assertEqual(len(tables["reference_results"]), 495)
        self.assertTrue(tables["smoke_tests"]["success"].all())
        self.assertTrue(tables["validation"]["success"].all())


if __name__ == "__main__":
    unittest.main()

