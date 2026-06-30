from __future__ import annotations

import unittest
from pathlib import Path

from platonic_space.empirical_laws import (
    EMPIRICAL_LAW_CLAIM_BOUNDARY,
    build_empirical_law_synthesis,
    validation_checks,
)


class TestE07EmpiricalLaws(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifacts_dir = Path("/artifacts")
        if not (artifacts_dir / "research_steps" / "S13" / "status.json").exists():
            raise unittest.SkipTest("S01-S13 artifacts are required for S14 empirical-law tests")
        cls.artifacts_dir = artifacts_dir
        cls.synthesis = build_empirical_law_synthesis(artifacts_dir)

    def test_catalog_contains_required_bounded_laws(self) -> None:
        laws = self.synthesis.law_catalog

        self.assertEqual(len(laws), 8)
        self.assertTrue(laws["lawId"].is_unique)
        for column in ("lawStatement", "scope", "excludedScope", "recommendedUse", "claimBoundary"):
            with self.subTest(column=column):
                self.assertTrue(laws[column].astype(str).str.len().gt(40).all())
        self.assertTrue(laws["claimBoundary"].astype(str).eq(EMPIRICAL_LAW_CLAIM_BOUNDARY).all())

    def test_every_law_has_evidence_counterexamples_uncertainty_and_falsification(self) -> None:
        law_ids = set(self.synthesis.law_catalog["lawId"])
        evidence_counts = self.synthesis.evidence_links.groupby("lawId").size().to_dict()
        counter_counts = self.synthesis.counterexamples.groupby("lawId").size().to_dict()
        uncertainty_counts = self.synthesis.uncertainty_register.groupby("lawId").size().to_dict()
        falsification_counts = self.synthesis.falsification_register.groupby("lawId").size().to_dict()

        for law_id in law_ids:
            with self.subTest(lawId=law_id):
                self.assertGreaterEqual(evidence_counts.get(law_id, 0), 2)
                self.assertGreaterEqual(counter_counts.get(law_id, 0), 1)
                self.assertGreaterEqual(uncertainty_counts.get(law_id, 0), 1)
                self.assertGreaterEqual(falsification_counts.get(law_id, 0), 1)

    def test_s13_transfer_constraint_is_explicit(self) -> None:
        laws = self.synthesis.law_catalog
        transfer = laws[laws["lawId"].eq("LAW08_transfer_requires_mechanism_mapping_not_raw_distance")].iloc[0]
        evidence = self.synthesis.evidence_links[
            self.synthesis.evidence_links["lawId"].eq("LAW08_transfer_requires_mechanism_mapping_not_raw_distance")
        ]

        self.assertEqual(transfer["supportLevel"], "constraining")
        self.assertIn("simple", str(transfer["lawStatement"]).lower())
        self.assertIn("S13", str(transfer["s13ConstraintRole"]))
        self.assertTrue(evidence["artifactPath"].astype(str).str.contains("/S13/", regex=False).any())

    def test_evidence_paths_resolve(self) -> None:
        missing = [
            path
            for path in self.synthesis.evidence_links["artifactPath"].astype(str)
            if not Path(path).exists()
        ]

        self.assertEqual(missing, [])

    def test_validation_checks_pass_without_hard_failures(self) -> None:
        checks = validation_checks(self.synthesis, artifacts_dir=self.artifacts_dir)
        hard_failures = checks[checks["severity"].eq("error") & ~checks["success"]]

        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
