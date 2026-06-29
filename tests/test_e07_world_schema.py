from __future__ import annotations

import unittest
from pathlib import Path

from platonic_space.world_schema import (
    REQUIRED_RECORD_FIELDS,
    SCHEMA_VERSION,
    TUPLE_FIELDS,
    build_world_catalog,
    validate_world_catalog,
    world_schema_document,
)


class TestE07WorldSchema(unittest.TestCase):
    def test_schema_declares_formal_tuple_fields(self) -> None:
        schema = world_schema_document()

        self.assertEqual(schema["schemaVersion"], SCHEMA_VERSION)
        declared = {item["field"] for item in schema["formalTuple"]}
        self.assertEqual(declared, set(TUPLE_FIELDS))
        for field in REQUIRED_RECORD_FIELDS:
            self.assertIn(field, schema["worldRecordSchema"]["required"])

    def test_catalog_covers_upstream_experiments_and_major_families(self) -> None:
        records = build_world_catalog("/previous-artifacts")

        self.assertGreaterEqual(len(records), 40)
        self.assertEqual({record["experimentId"] for record in records}, {f"E{i:02d}" for i in range(1, 7)})
        families = {record["worldFamily"] for record in records}
        for required in {
            "sorting",
            "scheduler_control",
            "null_model",
            "policy_morphospace",
            "memory_repair",
            "morphology_target",
            "morphology_benchmark",
            "chimera_goal_compatibility",
        }:
            self.assertIn(required, families)

    def test_catalog_validation_passes_for_mounted_artifacts(self) -> None:
        if not Path("/previous-artifacts").exists():
            self.skipTest("upstream artifact mounts are not present")
        records = build_world_catalog("/previous-artifacts")
        validation = validate_world_catalog(records, world_schema_document())
        hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
