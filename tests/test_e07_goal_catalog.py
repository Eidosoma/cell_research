from __future__ import annotations

import unittest
from pathlib import Path

from platonic_space.goal_catalog import (
    GOAL_CLAIM_BOUNDARY,
    GOAL_SCHEMA_VERSION,
    HARD_COVERAGE_REQUIREMENTS,
    REQUIRED_GOAL_FIELDS,
    build_goal_catalog,
    validate_goal_catalog,
)


class TestE07GoalCatalog(unittest.TestCase):
    def setUp(self) -> None:
        if (
            not Path("/previous-artifacts").exists()
            or not Path("/artifacts/research_steps/S01/normalized_world_catalog.parquet").exists()
            or not Path("/artifacts/research_steps/S02/policy_abstract_catalog.parquet").exists()
        ):
            self.skipTest("S01 world catalog, S02 policy catalog, and upstream mounts are required")

    def test_catalog_covers_required_goal_families_and_sources(self) -> None:
        records = build_goal_catalog()

        self.assertGreaterEqual(len(records), 30)
        families = {record["goalFamily"] for record in records}
        for family in HARD_COVERAGE_REQUIREMENTS:
            self.assertIn(family, families)
        source_experiments = {experiment for record in records for experiment in record["sourceExperimentIds"]}
        self.assertEqual(source_experiments, {f"E{i:02d}" for i in range(1, 7)})
        self.assertTrue(all(record["goalSchemaVersion"] == GOAL_SCHEMA_VERSION for record in records))
        self.assertTrue(all(record["claimBoundary"] == GOAL_CLAIM_BOUNDARY for record in records))

    def test_required_fields_and_substrate_links_present(self) -> None:
        records = build_goal_catalog()

        for record in records:
            for field in REQUIRED_GOAL_FIELDS:
                self.assertIn(field, record)
            self.assertTrue(record["linkedWorldIds"])
            self.assertTrue(record["metricBindingsJson"])
            self.assertTrue(record["sourceArtifactPaths"])
            self.assertEqual(record["exampleValidationStatus"], "validated")
            self.assertTrue(record["perfectExampleJson"]["validated"])
            self.assertTrue(record["failedExampleJson"]["validated"])

    def test_validation_has_no_hard_failures(self) -> None:
        records = build_goal_catalog()
        validation = validate_goal_catalog(records)
        hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
