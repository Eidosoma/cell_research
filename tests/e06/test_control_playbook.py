"""E06 S15 control-playbook tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.e06.control_playbook import (
    S15Config,
    build_handoff_markdown,
    build_playbook_markdown,
    claim_row,
    evidence_link,
    validate_s15_outputs,
)


class ControlPlaybookTests(unittest.TestCase):
    def test_evidence_link_records_row_specific_values(self) -> None:
        rows = pd.DataFrame(
            [
                {"claim_row_id": "r1", "metric": "final_target_quality", "value": 0.8},
                {"claim_row_id": "r2", "metric": "goal_conflict_index", "value": 0.2},
            ]
        )
        link = evidence_link(
            "E01",
            "C01",
            "S99",
            "tables/unit.csv",
            "metric in selected metrics",
            rows,
            {"mean_value": 0.5},
            "unit evidence",
        )
        self.assertEqual(link["source_row_count"], 2)
        self.assertIn("tables/unit.csv", link["artifact_path"])
        self.assertIn("mean_value", link["evidence_values_json"])
        self.assertIn("metric", link["source_columns_json"])

    def test_markdown_preserves_evidence_links_and_access_terms(self) -> None:
        claims = pd.DataFrame(
            [
                claim_row(
                    "C14",
                    "S14 null rescue result",
                    "Do not recommend S14 pulses.",
                    "high_internal_validity_null",
                    "null",
                    "finite_radius_governance,global_controller_like",
                    "global-controller-like comparator is separate.",
                    "computational proxy with bounded contexts",
                    "Prune intervention claims.",
                    "Do not claim no intervention could ever work.",
                )
            ]
        )
        evidence = pd.DataFrame(
            [
                evidence_link(
                    "E14",
                    "C14",
                    "S14",
                    "tables/e06_s14_intervention_rankings.csv",
                    "prospective_holdout_validation rows",
                    pd.DataFrame([{"intervention": "early_quorum_signal_radius2_15pct"}]),
                    {"supportive_holdout_count": 0},
                    "S14 held-out ranking",
                )
            ]
        )
        text = build_playbook_markdown(
            claims,
            evidence,
            artifacts_written=["reports/e06_chimeric_control_playbook.md"],
            validation_passed=True,
        )
        self.assertIn("E14", text)
        self.assertIn("finite_radius_governance", text)
        self.assertIn("global_controller_like", text)
        self.assertIn("No finite-radius local candidate", text)

    def test_validation_requires_claim_evidence_caveats_and_handoff_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_paths = {}
            for name in [
                "full_results_report",
                "playbook",
                "handoff",
                "phase_matrix",
                "claim_registry",
                "evidence_links",
                "history_effect_summary",
                "config",
            ]:
                path = root / f"{name}.txt"
                path.write_text("content\n", encoding="utf-8")
                artifact_paths[name] = path
            prior_reports = []
            for idx in range(1, 3):
                path = root / f"S{idx:02d}_report.md"
                path.write_text("prior\n", encoding="utf-8")
                prior_reports.append(path)
            claims = pd.DataFrame(
                [
                    claim_row(
                        "C01",
                        "Unit claim",
                        "Use the unit claim.",
                        "unit",
                        "supportive",
                        "behavior_only,global_controller_like",
                        "local and global are separate.",
                        "computational proxy caveat is explicit",
                        "Use for testing.",
                        "Do not overclaim.",
                    )
                ]
            )
            evidence = pd.DataFrame(
                [
                    evidence_link(
                        "E01",
                        "C01",
                        "S01",
                        "tables/unit.csv",
                        "all rows",
                        pd.DataFrame([{"x": 1}]),
                        {"x": 1},
                        "unit",
                    )
                ]
            )
            playbook = (
                "computational not causal proof behavior_only explicit_interface "
                "finite_radius_governance global_controller_like S14 null "
                "No finite-radius local candidate"
            )
            handoff = "Chief report-bundle. Do not start E07. computational-proxy."
            report = (
                "Step ID: S15\nCompletion status:\nArtifacts written:\nValidation result:\n"
                "Outcome classification:\nCaveats or blockers:\nLay summary:\nRecommended next action:\n"
            )
            validation = validate_s15_outputs(
                claims,
                evidence,
                artifact_paths,
                prior_reports,
                S15Config(min_claim_count=1, required_claim_ids=("C01",)),
                playbook_text=playbook,
                handoff_text=handoff,
                full_report_text=report,
                unit_tests_success=True,
            )
            self.assertTrue(validation["success"].all(), validation.to_string(index=False))


if __name__ == "__main__":
    unittest.main()
