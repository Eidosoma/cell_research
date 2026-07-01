"""Focused tests for the E02 S15 claim-audit runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.e02_s15_claim_audit import (
    ALTERNATIVE_KEYS,
    AlternativeEvidence,
    base_verdict_for_claim,
    build_audit_tables,
    claim_family_for_claim,
    evidence_level_for_claim,
    validate_outputs,
)


class ClaimAuditTests(unittest.TestCase):
    def make_alternatives(self) -> dict[str, AlternativeEvidence]:
        alternatives: dict[str, AlternativeEvidence] = {}
        for key in ALTERNATIVE_KEYS:
            alternatives[key] = AlternativeEvidence(
                key=key,
                title=key.replace("_", " "),
                step_ids=("S99",),
                artifact_paths=(f"/artifacts/{key}.csv",),
                summary=f"{key} summary",
                supports_families=frozenset(),
                constrains_families=frozenset(),
                caveat=f"{key} caveat",
            )
        alternatives["trajectory_label_shuffle"] = AlternativeEvidence(
            key="trajectory_label_shuffle",
            title="label shuffle",
            step_ids=("S04",),
            artifact_paths=("/artifacts/s04.csv",),
            summary="label shuffle supports aggregation",
            supports_families=frozenset({"aggregation"}),
            constrains_families=frozenset(),
            caveat="metric null",
        )
        alternatives["dg_matched_null"] = AlternativeEvidence(
            key="dg_matched_null",
            title="DG matched null",
            step_ids=("S08",),
            artifact_paths=("/artifacts/s08.csv",),
            summary="DG null constrains DG",
            supports_families=frozenset(),
            constrains_families=frozenset({"delayed_gratification"}),
            caveat="synthetic bridge",
        )
        alternatives["strong_statistics_fdr"] = AlternativeEvidence(
            key="strong_statistics_fdr",
            title="strong stats",
            step_ids=("S14",),
            artifact_paths=("/artifacts/s14.parquet",),
            summary="strong stats summary",
            supports_families=frozenset({"aggregation"}),
            constrains_families=frozenset({"delayed_gratification", "efficiency"}),
            caveat="artifact meta-analysis",
        )
        return alternatives

    def test_claim_family_mapping_covers_original_claim_groups(self) -> None:
        self.assertEqual(claim_family_for_claim("methods_n100_repeats100_unique_values"), "baseline_sorting")
        self.assertEqual(claim_family_for_claim("figure4_bubble_compare_plus_swap_steps"), "efficiency")
        self.assertEqual(claim_family_for_claim("figure5_cell_view_passive_ranking"), "frozen_robustness")
        self.assertEqual(claim_family_for_claim("figure7_all_algorithms_show_dg"), "delayed_gratification")
        self.assertEqual(claim_family_for_claim("figure7_bubble_insertion_dg_increases_with_frozen_count"), "delayed_gratification")
        self.assertEqual(claim_family_for_claim("figure8_unique_aggregation_above_control"), "aggregation")
        self.assertEqual(claim_family_for_claim("figure9_unique_opposite_dominance_and_aggregation"), "conflict_governance")
        self.assertEqual(claim_family_for_claim("figure10_duplicate_opposite_similar_conflict_pattern"), "conflict_governance")

    def test_verdict_rules_are_conservative_for_dg_and_nonreplication(self) -> None:
        self.assertEqual(base_verdict_for_claim("figure3_all_sorts_complete", "baseline_sorting", "exact"), "robust")
        self.assertEqual(
            base_verdict_for_claim("figure7_cell_vs_traditional_directions", "delayed_gratification", "directionally consistent"),
            "partly_explained_by_alternative",
        )
        self.assertEqual(
            base_verdict_for_claim("figure8_unique_aggregation_exact_peak_magnitudes", "aggregation", "not replicated"),
            "not_replicated_or_unresolved",
        )
        self.assertEqual(
            evidence_level_for_claim("conflict_governance", "directionally consistent", "robust_but_narrower"),
            "E01_replication_with_limited_E02_direct_stress_tests",
        )

    def test_build_audit_tables_represents_all_alternatives_per_claim(self) -> None:
        e01 = pd.DataFrame(
            [
                {
                    "claim_id": "figure8_unique_aggregation_above_control",
                    "figure_or_section": "Figure 8",
                    "paper_claim": "Aggregation above control",
                    "classification": "directionally consistent",
                    "replication_evidence": "aggregation evidence",
                    "evidence_artifacts": "/artifacts/e01.csv",
                    "caveats": "aggregation caveat",
                    "divergence_cause": "none",
                },
                {
                    "claim_id": "figure7_cell_vs_traditional_directions",
                    "figure_or_section": "Figure 7",
                    "paper_claim": "DG directions",
                    "classification": "directionally consistent",
                    "replication_evidence": "DG evidence",
                    "evidence_artifacts": "/artifacts/e01_dg.csv",
                    "caveats": "DG caveat",
                    "divergence_cause": "none",
                },
            ]
        )
        claims, matrix = build_audit_tables(
            e01,
            self.make_alternatives(),
            {
                "aggregation": "aggregation S14 summary",
                "delayed_gratification": "DG S14 summary",
            },
        )
        self.assertEqual(len(claims), 2)
        self.assertEqual(len(matrix), 2 * len(ALTERNATIVE_KEYS))
        self.assertEqual(set(matrix["alternative_key"]), set(ALTERNATIVE_KEYS))
        agg = claims[claims["claim_family"] == "aggregation"].iloc[0]
        self.assertEqual(agg["final_verdict"], "robust_but_narrower")
        self.assertIn("trajectory_label_shuffle", agg["alternative_explanations"])
        dg = claims[claims["claim_family"] == "delayed_gratification"].iloc[0]
        self.assertEqual(dg["final_verdict"], "partly_explained_by_alternative")
        self.assertIn("dg_matched_null", dg["alternative_explanations"])
        self.assertTrue(json.loads(agg["source_artifacts_json"]))

    def test_validate_outputs_requires_top_summaries_and_complete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_paths = {
                "claims_csv": root / "claims.csv",
                "matrix_csv": root / "matrix.csv",
                "claim_audit_report": root / "claim.md",
                "handoff_report": root / "handoff.md",
                "full_results_report": root / "full.md",
                "validation_json": root / "validation.json",
                "status_json": root / "status.json",
                "manifest_json": root / "manifest.json",
                "src_manifest_json": root / "src.json",
                "log": root / "run.log",
            }
            input_paths = {"input": root / "input.csv"}
            for path in [*output_paths.values(), *input_paths.values()]:
                path.write_text("x\n", encoding="utf-8")
            top = """# report
## Top Summary
- Step ID: S15
- Completion status: completed
- Artifacts written: x
- Validation result: passed
- Outcome classification: constraining/contradictory
- Caveats or blockers: caveat
- Lay summary: summary
- Recommended next action: stop
"""
            for path in (output_paths["claim_audit_report"], output_paths["handoff_report"], output_paths["full_results_report"]):
                path.write_text(top, encoding="utf-8")
            claims = pd.DataFrame(
                [
                    {
                        "claim_id": "figure3_all_sorts_complete",
                        "claim_family": "baseline_sorting",
                        "final_verdict": "robust",
                        "e02_evidence_summary": "evidence",
                        "strong_statistics_summary": "stats",
                        "source_artifacts_json": "[\"/artifacts/e01.csv\"]",
                    }
                ]
            )
            matrix = pd.DataFrame(
                [
                    {
                        "claim_id": "figure3_all_sorts_complete",
                        "alternative_key": key,
                        "support_level": "not_directly_tested",
                    }
                    for key in ALTERNATIVE_KEYS
                ]
            )
            validation = validate_outputs(
                claims=claims,
                matrix=matrix,
                verdict_summary=pd.DataFrame({"x": [1]}),
                alternative_summary=pd.DataFrame({"x": [1]}),
                output_paths=output_paths,
                input_paths=input_paths,
                unit_result={"success": True},
            )
            self.assertFalse(validation["success"])
            self.assertEqual(validation["alternativeMatrixRowCount"], len(ALTERNATIVE_KEYS))
            self.assertTrue(validation["allAlternativesRepresented"])


if __name__ == "__main__":
    unittest.main()
