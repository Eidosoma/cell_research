from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_claim_registry.py"
SOURCE = Path(
    "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/"
    "pdf-markdown.md"
)


def load_builder():
    spec = importlib.util.spec_from_file_location("build_claim_registry", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClaimRegistryTests(unittest.TestCase):
    def test_registry_schema_coverage_and_uniqueness(self) -> None:
        builder = load_builder()
        frame = builder.to_frame(builder.build_records())
        validation = builder.validate_frame(frame)

        self.assertTrue(validation["passed"], validation["errors"])
        self.assertEqual(validation["rowCount"], 118)
        self.assertEqual(validation["uniqueClaimIdCount"], 118)
        self.assertEqual(set(frame["figure"].astype(int)), set(range(3, 11)))
        self.assertEqual(validation["evidenceStatusCounts"]["not_recoverable"], 3)
        self.assertEqual(
            validation["executableSpecStatusCounts"],
            {"needs_decision": 114, "not_applicable": 1, "unavailable": 3},
        )

    def test_registered_contradiction_and_unavailable_values_are_explicit(self) -> None:
        builder = load_builder()
        frame = builder.to_frame(builder.build_records()).set_index("claim_id")

        traditional = frame.loc["F05-B-PASSIVE-INSERTION-F2-TRADITIONAL"]
        cell_view = frame.loc["F05-B-PASSIVE-INSERTION-F2-CELL_VIEW"]
        self.assertEqual(traditional["reported_mean"], 4.65)
        self.assertEqual(cell_view["reported_mean"], 5.28)
        self.assertEqual(cell_view["evidence_status"], "contradictory")
        self.assertIn("PASSIVE_INSERTION_CONTRADICTION", cell_view["ambiguity_codes"])

        figure_10 = frame.loc[
            ["F10-A-REPEATED-OPPOSITE", "F10-B-REPEATED-OPPOSITE", "F10-C-REPEATED-OPPOSITE"]
        ]
        self.assertTrue(figure_10["reported_mean"].isna().all())
        self.assertTrue((figure_10["evidence_status"] == "not_recoverable").all())
        self.assertTrue(
            figure_10["ambiguity_codes"].str.contains("F10_NUMERIC_NOT_RECOVERABLE").all()
        )

    def test_cli_writes_round_trippable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--source-markdown",
                    str(SOURCE),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

            csv_frame = pd.read_csv(output / "claim_registry.csv")
            parquet_frame = pd.read_parquet(output / "claim_registry.parquet")
            self.assertEqual(len(csv_frame), 118)
            self.assertEqual(len(parquet_frame), 118)
            self.assertEqual(csv_frame["claim_id"].tolist(), parquet_frame["claim_id"].tolist())

            validation = json.loads((output / "validation_summary.json").read_text())
            self.assertTrue(validation["passed"])
            self.assertEqual(validation["sourceValidationErrors"], [])
            self.assertTrue((output / "ambiguity_log.md").read_text().startswith("# S01 Ambiguity Log"))
            self.assertTrue(
                (output / "paper_metric_definitions.md")
                .read_text()
                .startswith("# S01 Paper Metric Definitions")
            )
            manifest = json.loads((output / "artifact_manifest.json").read_text())
            self.assertEqual(manifest["researchStepId"], "S01")


if __name__ == "__main__":
    unittest.main()
