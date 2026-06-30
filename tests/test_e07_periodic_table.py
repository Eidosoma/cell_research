from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from platonic_space.periodic_table import (
    PERIODIC_TABLE_CLAIM_BOUNDARY,
    S13_S14_CAVEAT,
    build_periodic_table_bundle,
    extract_html_links,
    render_periodic_table_html,
    validate_periodic_table_outputs,
)


class TestE07PeriodicTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifacts_dir = Path("/artifacts")
        if not (artifacts_dir / "research_steps" / "S14" / "status.json").exists():
            raise unittest.SkipTest("S01-S14 artifacts are required for S15 periodic-table tests")
        cls.artifacts_dir = artifacts_dir
        cls.bundle = build_periodic_table_bundle(artifacts_dir)

    def test_bundle_has_expected_entity_counts_and_caveats(self) -> None:
        payload = self.bundle.report_bundle_payload

        self.assertGreaterEqual(payload["entityCounts"]["policies"], 500)
        self.assertGreaterEqual(payload["entityCounts"]["goals"], 30)
        self.assertGreaterEqual(payload["entityCounts"]["worlds"], 50)
        self.assertEqual(payload["entityCounts"]["laws"], 8)
        self.assertIn("simple S08-distance transfer-success law", payload["s13S14Caveat"])
        self.assertIn("Computational atlas", payload["claimBoundary"])

    def test_html_is_standalone_and_preserves_claim_boundaries(self) -> None:
        html_text = render_periodic_table_html(bundle=self.bundle, created_utc="2026-06-30T00:00:00+00:00")
        links = extract_html_links(html_text)

        self.assertIn("E07 Periodic Table Atlas", html_text)
        self.assertIn(PERIODIC_TABLE_CLAIM_BOUNDARY, html_text)
        self.assertIn(S13_S14_CAVEAT, html_text)
        self.assertFalse(links["target"].astype(str).str.startswith(("http://", "https://")).any())
        self.assertIn("/artifacts/results/e07_periodic_table_catalog.parquet", set(links["target"].astype(str)))

    def test_model_card_and_artifact_indexes_resolve_existing_paths(self) -> None:
        self.assertGreaterEqual(len(self.bundle.model_card_index), 8)
        self.assertTrue(self.bundle.model_card_index["exists"].astype(bool).all())
        required = {
            "/artifacts/research_steps/S14/empirical_law_catalog.parquet",
            "/artifacts/research_steps/S13/s08_distance_transfer_prediction.parquet",
            "/artifacts/research_steps/S10/universality_classes.parquet",
        }
        indexed = set(self.bundle.artifact_index["artifactPath"].astype(str))

        self.assertTrue(required.issubset(indexed))

    def test_validation_passes_for_rendered_temp_outputs(self) -> None:
        if not Path("/artifacts/results/e07_periodic_table_catalog.parquet").exists():
            raise unittest.SkipTest("S15 static exports are required for full atlas link validation")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            html_path = tmp_path / "atlas.html"
            report_path = tmp_path / "report.md"
            bundle_path = tmp_path / "bundle.json"
            export_path = tmp_path / "export.csv"
            html_path.write_text(render_periodic_table_html(bundle=self.bundle, created_utc="2026-06-30T00:00:00+00:00"), encoding="utf-8")
            report_path.write_text(PERIODIC_TABLE_CLAIM_BOUNDARY + "\n" + S13_S14_CAVEAT, encoding="utf-8")
            bundle_path.write_text('{"s13S14Caveat": "present"}', encoding="utf-8")
            export_path.write_text("ok\n1\n", encoding="utf-8")
            checks = validate_periodic_table_outputs(
                artifacts_dir=self.artifacts_dir,
                bundle=self.bundle,
                html_path=html_path,
                report_path=report_path,
                bundle_path=bundle_path,
                static_export_paths=(export_path,),
            )
        hard_failures = checks[checks["severity"].eq("error") & ~checks["success"]]

        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
