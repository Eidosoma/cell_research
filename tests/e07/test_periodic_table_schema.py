"""Tests for E07 S15 periodic-table schema helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.e07.periodic_table_schema import (
    PERIODIC_TABLE_SCHEMA_VERSION,
    extract_local_hrefs,
    report_relative_href,
    stable_hash,
    validate_periodic_table_artifacts,
    validation_summary,
)


class PeriodicTableSchemaTests(unittest.TestCase):
    def test_stable_hash_is_deterministic(self) -> None:
        payload_a = {"b": [2, 3], "a": 1}
        payload_b = {"a": 1, "b": [2, 3]}
        self.assertEqual(stable_hash(payload_a), stable_hash(payload_b))

    def test_extract_local_hrefs_ignores_external_and_anchor_links(self) -> None:
        html = """
        <a href="../tables/a.csv">A</a>
        <a href="https://example.org/x">X</a>
        <a href="#section">section</a>
        <a href="../reports/b.md#top">B</a>
        """
        self.assertEqual(extract_local_hrefs(html), ["../tables/a.csv", "../reports/b.md"])

    def test_report_relative_href_targets_artifacts_from_reports_dir(self) -> None:
        href = report_relative_href(Path("/tmp/artifacts/results/final.json"), Path("/tmp/artifacts/reports"))
        self.assertEqual(href, "../results/final.json")

    def test_validate_periodic_table_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports = root / "reports"
            step = root / "research_steps" / "S15"
            results = root / "results"
            reports.mkdir(parents=True)
            step.mkdir(parents=True)
            results.mkdir(parents=True)

            linked = root / "tables" / "e07_law_evidence_matrix.csv"
            linked.parent.mkdir()
            linked.write_text("law_id,claim_status\nS14-SPEC-001,unsupported_speculation\n", encoding="utf-8")

            html_path = reports / "e07_periodic_table.html"
            summary_path = reports / "e07_periodic_table_summary.md"
            handoff_path = reports / "e07_report_bundle_handoff.md"
            full_results_path = step / "research_step_full_results.md"
            manifest_path = results / "e07_final_corpus_manifest.json"

            html_path.write_text(
                """
                <html><body>
                <a href="../tables/e07_law_evidence_matrix.csv">matrix</a>
                S14-SPEC-001 unsupported_speculation S13 embedded-row only.
                This bounded computational atlas is not causal and not biological; do not generalize.
                </body></html>
                """,
                encoding="utf-8",
            )
            top_summary = """# Report

## Top Summary

- Research step ID: S15
- Completion status: Completed
- Artifacts written: listed
- Validation result: passed
- Outcome classification: supportive
- Caveats or blockers: bounded computational output; not causal and not biological; S13 embedded-row only, do not generalize.
- Lay summary: Atlas package.
- Recommended next action: Chief review.
"""
            summary_path.write_text(top_summary, encoding="utf-8")
            handoff_path.write_text(top_summary, encoding="utf-8")
            full_results_path.write_text(top_summary, encoding="utf-8")
            manifest_path.write_text("{}", encoding="utf-8")

            manifest = {
                "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
                "manifestPath": str(manifest_path),
                "corpusVersion": "E07-S15",
                "s14ClaimStatusCounts": {"unsupported_speculation": 1},
                "unsupportedSpeculationLawIds": ["S14-SPEC-001"],
                "s14LawRowCount": 1,
                "sourceStepsRepresented": [f"S{idx:02d}" for idx in range(1, 16)],
                "artifacts": [
                    {"artifactKey": "periodic_table_html", "path": str(html_path), "sha256": "abc"},
                    {"artifactKey": "periodic_table_summary", "path": str(summary_path), "sha256": "abc"},
                    {"artifactKey": "final_corpus_manifest", "path": str(manifest_path), "checksumOmittedReason": "self"},
                    {"artifactKey": "report_bundle_handoff", "path": str(handoff_path), "sha256": "abc"},
                    {"artifactKey": "full_results", "path": str(full_results_path), "sha256": "abc"},
                ],
            }
            law_matrix = pd.DataFrame(
                {
                    "law_id": ["S14-SPEC-001"],
                    "claim_status": ["unsupported_speculation"],
                    "unsupported_speculation": [True],
                }
            )
            source_manifest = pd.DataFrame({"sha256": ["abc"], "path": [str(linked)], "source_step_id": ["S14"]})

            checks = validate_periodic_table_artifacts(
                manifest=manifest,
                html_path=html_path,
                summary_path=summary_path,
                handoff_path=handoff_path,
                full_results_path=full_results_path,
                law_matrix=law_matrix,
                source_manifest=source_manifest,
                expected_claim_status_counts={"unsupported_speculation": 1},
                expected_unsupported_ids=["S14-SPEC-001"],
            )
            self.assertTrue(bool(checks["success"].all()), checks.to_string(index=False))
            self.assertTrue(validation_summary(checks)["allPassed"])
            json.dumps(manifest, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
