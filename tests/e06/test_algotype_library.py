"""E06 S01 Algotype-library curation tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.e06.algotype_library import (
    S01ValidationConfig,
    builtin_control_records,
    curate_algotype_records,
    summarize_validation,
    validate_algotype_records,
    validation_checks,
)
from src.e06.algotype_library import S01SourcePaths


class AlgotypeLibraryTests(unittest.TestCase):
    def test_builtin_controls_have_unique_ids_and_control_labels(self) -> None:
        records = builtin_control_records()
        self.assertEqual(len(records), 4)
        self.assertEqual(len({record["algotypeId"] for record in records}), 4)
        categories = {record["sourceCategory"] for record in records}
        self.assertEqual(categories, {"null_control", "randomized"})
        self.assertTrue(all(record["representationType"] == "dsl" for record in records))

    def test_missing_upstream_paths_fall_back_to_originals_and_controls(self) -> None:
        paths = S01SourcePaths(
            e03_classic_library=Path("/definitely/missing/classics.json"),
            e03_frontier_candidates=Path("/definitely/missing/frontier.jsonl"),
            e03_frontier_table=Path("/definitely/missing/frontier.csv"),
            e04_repair_capable=Path("/definitely/missing/e04.jsonl"),
            e04_evolved_repair=Path("/definitely/missing/e04_evolved.jsonl"),
        )
        records, status = curate_algotype_records(paths)
        self.assertEqual(status["originals"]["loaded"], 3)
        self.assertTrue(status["originals"]["fallbackUsed"])
        self.assertTrue(status["discovered"]["fallbackUsed"])
        self.assertTrue(status["memoryRepair"]["fallbackUsed"])
        self.assertGreaterEqual(len(records), 7)
        self.assertIn("original", {record["sourceCategory"] for record in records})
        self.assertIn("null_control", {record["sourceCategory"] for record in records})

    def test_control_validation_labels_nulls_without_swaps(self) -> None:
        records = builtin_control_records()
        config = S01ValidationConfig(seeds=(101,), array_size=5, dsl_event_cap=20, memory_event_cap=20)
        validation = validate_algotype_records(records, config)
        metadata = summarize_validation(records, validation)
        checks = validation_checks(records, metadata, validation, {"discovered": {"loaded": 1}, "memoryRepair": {"loaded": 1}})
        self.assertFalse(validation["invalid"].any())
        nulls = metadata[metadata["source_category"] == "null_control"]
        self.assertTrue((nulls["mean_swap_count"] == 0.0).all())
        self.assertTrue(bool(checks.loc[checks["validation_case"] == "null_controls_do_not_swap", "success"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
