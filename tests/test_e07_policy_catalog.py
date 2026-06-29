from __future__ import annotations

import unittest
from pathlib import Path

from platonic_space.policy_catalog import (
    HARD_COVERAGE_REQUIREMENTS,
    POLICY_CLAIM_BOUNDARY,
    POLICY_SCHEMA_VERSION,
    REQUIRED_POLICY_FIELDS,
    build_policy_catalog,
    validate_policy_catalog,
)


class TestE07PolicyCatalog(unittest.TestCase):
    def setUp(self) -> None:
        if not Path("/previous-artifacts").exists() or not Path("/artifacts/research_steps/S01/normalized_world_catalog.parquet").exists():
            self.skipTest("S01 world catalog and upstream mounts are required")

    def test_catalog_covers_policy_families_and_sources(self) -> None:
        records = build_policy_catalog()

        self.assertGreaterEqual(len(records), 2500)
        self.assertEqual({record["sourceExperimentId"] for record in records}, {f"E{i:02d}" for i in range(1, 7)})
        kinds = {record["policyKind"] for record in records}
        for kind in HARD_COVERAGE_REQUIREMENTS:
            self.assertIn(kind, kinds)
        self.assertTrue(all(record["policySchemaVersion"] == POLICY_SCHEMA_VERSION for record in records))
        self.assertTrue(all(record["claimBoundary"] == POLICY_CLAIM_BOUNDARY for record in records))

    def test_required_fields_and_linked_worlds_present(self) -> None:
        records = build_policy_catalog()

        for record in records[:50]:
            for field in REQUIRED_POLICY_FIELDS:
                self.assertIn(field, record)
            self.assertTrue(record["linkedWorldIds"])
            self.assertTrue(record["sourceArtifactPaths"])

    def test_validation_has_no_hard_failures(self) -> None:
        records = build_policy_catalog()
        validation = validate_policy_catalog(records)
        hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
