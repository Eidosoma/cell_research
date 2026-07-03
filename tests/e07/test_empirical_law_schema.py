"""Tests for E07 S14 empirical-law schema helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.e07.empirical_law_schema import (
    dataframe_from_law_records,
    empirical_law_id,
    validate_empirical_law_artifacts,
    validation_summary,
)


class EmpiricalLawSchemaTests(unittest.TestCase):
    def test_law_ids_are_stable(self) -> None:
        self.assertEqual(empirical_law_id("abc", "A title"), empirical_law_id("abc", "A title"))
        self.assertNotEqual(empirical_law_id("abc", "A title"), empirical_law_id("abc", "Other"))

    def test_dataframe_serializes_json_fields(self) -> None:
        frame = dataframe_from_law_records(
            [
                {
                    "law_slug": "narrow_transfer",
                    "law_title": "Narrow transfer law",
                    "claim_status": "supported_narrow",
                    "outcome_classification": "supportive_narrow",
                    "unsupported_speculation": False,
                    "s13_scope_limited": True,
                    "evidence_steps_json": ["S13"],
                    "evidence_summary": "Exact embedded-row transfer.",
                    "quantitative_support_json": {"fresh_rows": 8},
                    "scope": "Only the existing E05 embedded-row path.",
                    "counterexamples": "Other transfer mappings are blocked.",
                    "falsification_tests": "Validate graph transfer with a frozen adapter.",
                    "caveats": "Do not generalize.",
                    "recommended_use": "Use as continuity evidence only.",
                    "source_artifacts_json": [{"path": "x"}],
                }
            ]
        )
        self.assertEqual(len(frame), 1)
        self.assertIn("S13", frame.loc[0, "evidence_steps_json"])
        self.assertIn("fresh_rows", frame.loc[0, "quantitative_support_json"])

    def test_validate_empirical_law_artifacts(self) -> None:
        laws = dataframe_from_law_records(
            [
                {
                    "law_slug": "narrow_transfer",
                    "law_title": "Narrow transfer law",
                    "claim_status": "supported_narrow",
                    "outcome_classification": "supportive_narrow",
                    "unsupported_speculation": False,
                    "s13_scope_limited": True,
                    "evidence_steps_json": ["S13"],
                    "evidence_summary": "Exact embedded-row transfer.",
                    "quantitative_support_json": {"fresh_rows": 8},
                    "scope": "Only the existing E05 embedded-row path and blocked mappings.",
                    "counterexamples": "S12 DSL-to-E05 mappings are blocked.",
                    "falsification_tests": "Pre-freeze graph and E06 adapters and validate transfer.",
                    "caveats": "Do not generalize beyond embedded row.",
                    "recommended_use": "Use as continuity evidence only.",
                    "source_artifacts_json": [{"path": "x"}],
                },
                {
                    "law_slug": "memory_threshold",
                    "law_title": "Unsupported speculation: memory-depth threshold for repair",
                    "claim_status": "unsupported_speculation",
                    "outcome_classification": "unsupported",
                    "unsupported_speculation": True,
                    "s13_scope_limited": False,
                    "evidence_steps_json": ["S09", "S11", "S12"],
                    "evidence_summary": "Memory proxy exists but repair validation is blocked.",
                    "quantitative_support_json": {"blocker_rows": 4},
                    "scope": "No current executable repair-threshold claim.",
                    "counterexamples": "Memory candidates have counterexamples.",
                    "falsification_tests": "Run repair holdouts with memory-depth ablations.",
                    "caveats": "Speculation only.",
                    "recommended_use": "Do not use as a law.",
                    "source_artifacts_json": [{"path": "y"}],
                },
            ]
        )
        source_manifest = pd.DataFrame(
            {
                "source_step_id": ["S13"],
                "path": ["x"],
                "sha256": ["abc"],
                "row_count": [1],
            }
        )
        checks = validate_empirical_law_artifacts(laws, source_manifest, expected_report_law_count=2)
        self.assertTrue(bool(checks["success"].all()), checks.to_string(index=False))
        self.assertTrue(validation_summary(checks)["allPassed"])


if __name__ == "__main__":
    unittest.main()
