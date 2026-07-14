from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from analysis.baseline_release import (
    CLASSIFICATIONS,
    build_claim_matrix,
    report_input_records,
    run_reference_smoke,
    source_hash_manifest,
    validate_claim_matrix,
)


class BaselineReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = build_claim_matrix()

    def test_all_118_registry_claims_are_reconciled_once(self) -> None:
        validation = validate_claim_matrix(self.matrix)
        self.assertTrue(validation["success"], validation)
        self.assertEqual(len(self.matrix), 118)
        self.assertEqual(self.matrix.claim_id.nunique(), 118)
        self.assertEqual(set(self.matrix.final_replication_classification) - CLASSIFICATIONS, set())

    def test_figure_five_uses_all_registered_layers(self) -> None:
        figure5 = self.matrix[self.matrix.figure.eq(5)]
        self.assertEqual(len(figure5), 40)
        self.assertEqual(int(figure5.claim_kind.eq("panel_mean").sum()), 36)
        self.assertEqual(int(figure5.claim_kind.eq("definition").sum()), 2)
        self.assertEqual(int(figure5.claim_kind.eq("qualitative_rank").sum()), 2)
        passive_rank = figure5[figure5.claim_id.eq("F05-B-PASSIVE-RANK")].iloc[0]
        self.assertEqual(passive_rank.final_replication_classification, "implementation_sensitive")

    def test_nonempirical_and_unavailable_claims_are_not_forced_supportive(self) -> None:
        context = self.matrix[self.matrix.claim_id.eq("F06-A-CONCEPTUAL-CONTEXT")].iloc[0]
        self.assertEqual(context.final_replication_classification, "not_testable")
        duplicate_example = self.matrix[self.matrix.claim_id.eq("F08-E-DUPLICATE-EXAMPLES")].iloc[0]
        self.assertEqual(duplicate_example.final_replication_classification, "not_testable")

    def test_duplicate_and_censoring_boundaries_are_explicit(self) -> None:
        figure10 = self.matrix[self.matrix.figure.eq(10)]
        self.assertTrue(figure10.strict_duplicate_sensitivity.str.contains("both_retained").all())
        self.assertEqual(int(self.matrix.event_budget_censoring_present.sum()), 6)
        b = self.matrix[self.matrix.claim_id.eq("F09-B-FINAL-SORTEDNESS")].iloc[0]
        self.assertEqual(b.final_replication_classification, "implementation_sensitive")

    def test_reference_smoke_is_exact_and_invariant_checked(self) -> None:
        result = run_reference_smoke()
        self.assertTrue(result["success"], result)
        self.assertEqual(len(result["samples"]), 3)

    def test_source_manifest_is_pointer_only(self) -> None:
        commit = __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
        result = source_hash_manifest(commit)
        self.assertFalse(result["sourceArchiveCreated"])
        self.assertFalse(result["historicalSourceIncluded"])
        self.assertGreater(len(result["files"]), 20)
        self.assertFalse(any(item["path"].startswith("modules/") for item in result["files"]))

    def test_report_input_manifest_excludes_itself(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_inputs = Path(temporary_directory)
            (report_inputs / "evidence.json").write_text("{}\n")
            (report_inputs / "report_bundle_manifest.json").write_text("{}\n")

            records = report_input_records(report_inputs)

        self.assertEqual([record["label"] for record in records], ["evidence.json"])


if __name__ == "__main__":
    unittest.main()
