"""Tests for E07 S01 world schema metadata."""

from __future__ import annotations

import json
import unittest

from src.e07.world_schema import (
    REQUIRED_TUPLE_FIELDS,
    WorldRecord,
    dataframe_from_records,
    make_world_id,
    stable_hash,
    stable_json,
    validate_inventory,
)


def complete_record(task_id: str = "unit") -> WorldRecord:
    return WorldRecord(
        world_id=make_world_id("E07", "S01", task_id, family="unit"),
        experiment_id="E07",
        source_step_id="S01",
        record_granularity="unit",
        world_family="unit",
        task_id=task_id,
        task_label="Unit task",
        substrate_kind="array_1d",
        state_space="finite array state",
        local_observations="local neighbors",
        action_set="swap, wait",
        transition_rules="deterministic local transition",
        goal_predicate="sorted increasing",
        perturbation_model="none",
        measurement_functions="sortedness",
        source_artifacts=("/tmp/source.json",),
        metadata={"a": 1},
    )


class WorldSchemaTests(unittest.TestCase):
    def test_stable_json_and_hash_are_order_invariant(self) -> None:
        left = {"b": [2, 1], "a": {"x": 3}}
        right = {"a": {"x": 3}, "b": [2, 1]}
        self.assertEqual(stable_json(left), stable_json(right))
        self.assertEqual(stable_hash(left), stable_hash(right))

    def test_make_world_id_is_stable_and_slugged(self) -> None:
        world_id = make_world_id("E01", "S03", "Bubble/Insertion 50:50", family="Same Goal")
        self.assertEqual(world_id, "world:e01:s03:same-goal:bubble-insertion-50-50")

    def test_record_validation_marks_missing_tuple_fields(self) -> None:
        record = complete_record()
        self.assertEqual(record.with_validation().missing_fields, ())
        broken = WorldRecord(**{**record.__dict__, "goal_predicate": ""}).with_validation()
        self.assertEqual(broken.missing_fields, ("goal_predicate",))
        self.assertEqual(broken.completeness, "partial_explicit_missing")

    def test_dataframe_roundtrips_json_columns(self) -> None:
        df = dataframe_from_records([complete_record()])
        self.assertEqual(set(REQUIRED_TUPLE_FIELDS).issubset(df.columns), True)
        self.assertEqual(json.loads(df.loc[0, "metadata_json"]), {"a": 1})
        self.assertEqual(json.loads(df.loc[0, "source_artifacts_json"]), ["/tmp/source.json"])
        self.assertEqual(json.loads(df.loc[0, "missing_fields_json"]), [])
        self.assertEqual(df.loc[0, "completeness"], "complete")

    def test_inventory_validation_accepts_explicit_partial_rows(self) -> None:
        complete = complete_record("complete")
        partial = WorldRecord(**{**complete_record("partial").__dict__, "action_set": ""})
        df = dataframe_from_records([complete, partial])
        checks = validate_inventory(df, required_experiments=("E07",))
        results = dict(zip(checks["validation_case"], checks["success"], strict=True))
        self.assertTrue(results["required_columns_present"])
        self.assertTrue(results["world_ids_unique"])
        self.assertTrue(results["missing_fields_explicitly_marked"])
        self.assertTrue(results["json_columns_roundtrip"])

    def test_inventory_validation_detects_duplicate_ids(self) -> None:
        record = complete_record("same")
        df = dataframe_from_records([record, record])
        checks = validate_inventory(df, required_experiments=("E07",))
        results = dict(zip(checks["validation_case"], checks["success"], strict=True))
        self.assertFalse(results["world_ids_unique"])


if __name__ == "__main__":
    unittest.main()
