from __future__ import annotations

import unittest
from pathlib import Path

from platonic_space.behavior_corpus import (
    BEHAVIOR_CLAIM_BOUNDARY,
    BEHAVIOR_SCHEMA_VERSION,
    REQUIRED_BEHAVIOR_FIELDS,
    build_behavior_corpus,
    validate_behavior_corpus,
)


class TestE07BehaviorCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            Path("/previous-artifacts"),
            Path("/artifacts/research_steps/S01/normalized_world_catalog.parquet"),
            Path("/artifacts/research_steps/S02/policy_abstract_catalog.parquet"),
            Path("/artifacts/research_steps/S03/goal_catalog.parquet"),
        )
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("S01/S02/S03 catalogs and upstream previous artifacts are required")
        cls.records, cls.artifact_index = build_behavior_corpus()

    def test_corpus_covers_all_upstream_experiments(self) -> None:
        source_experiments = {record["sourceExperimentId"] for record in self.records}

        self.assertGreater(len(self.records), 50_000)
        self.assertEqual(source_experiments, {f"E{i:02d}" for i in range(1, 7)})
        self.assertTrue(all(record["behaviorSchemaVersion"] == BEHAVIOR_SCHEMA_VERSION for record in self.records))
        self.assertTrue(all(record["claimBoundary"] == BEHAVIOR_CLAIM_BOUNDARY for record in self.records))

    def test_required_fields_and_source_index_present(self) -> None:
        available_sources = self.artifact_index[self.artifact_index["available"]]

        self.assertGreaterEqual(len(available_sources), 60)
        self.assertTrue((available_sources["sourceTableSha256"].astype(str).str.len() == 64).all())
        self.assertTrue((available_sources["integratedRowCount"] == available_sources["rowCount"]).all())
        for record in self.records[:500]:
            for field in REQUIRED_BEHAVIOR_FIELDS:
                self.assertIn(field, record)
            self.assertTrue(record["worldId"])
            self.assertTrue(record["abstractGoalId"])
            self.assertTrue(record["metricValuesJson"] or record["behaviorVectorJson"])

    def test_validation_has_no_hard_failures(self) -> None:
        validation = validate_behavior_corpus(self.records, self.artifact_index)
        hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
