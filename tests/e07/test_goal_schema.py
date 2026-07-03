"""Tests for E07 S03 goal representation metadata."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.e07.goal_schema import (
    GoalRepresentation,
    dataframe_from_goal_records,
    goal_uid,
    stable_hash,
    stable_json,
    validate_goal_table,
)


THIS_FILE = Path(__file__)


def complete_goal(
    source_experiment_id: str = "E07",
    source_record_id: str = "unit-record",
    *,
    source_goal_id: str = "unit:goal",
    partial: bool = False,
    limitations: tuple[str, ...] = ("unit limitation",),
    linked_world_ids: tuple[str, ...] = ("world:unit",),
    conflicts_with_goal_ids: tuple[str, ...] = (),
    linked_policy_ids: tuple[str, ...] = ("policy:unit",),
) -> GoalRepresentation:
    return GoalRepresentation(
        source_experiment_id=source_experiment_id,
        source_step_id="S03",
        source_record_id=source_record_id,
        source_goal_id=source_goal_id,
        display_name="Unit goal",
        goal_family="unit_family",
        goal_kind="unit_kind",
        abstraction_kind="scalar_metric",
        target_structure="unit target",
        representation_status="partial_explicit" if partial else "complete",
        partial=partial,
        target_direction="partial_unknown" if partial else "maximize",
        unit="unitless",
        primary_metric_id="unit_metric",
        metric_family="unit_metric_family",
        target_value="1.0",
        lower_bound=0.0,
        upper_bound=1.0,
        zero_point_definition="0 means absent unit score.",
        energy_function="1 - unit_metric",
        constraint_predicate="unit predicate",
        conflict_group_id="unit_conflict" if conflicts_with_goal_ids else "",
        conflicts_with_goal_ids=conflicts_with_goal_ids,
        linked_world_ids=linked_world_ids,
        linked_policy_ids=linked_policy_ids,
        source_artifacts=(str(THIS_FILE),),
        limitations=limitations if partial else (),
        metric_contract={"direction": "higher_is_better"},
        representation_payload={"kind": "unit"},
        direction_validation_status="partial_direction_declared" if partial else "explicit_direction_declared",
        unit_validation_status="partial_unit_declared" if partial else "explicit_unit_declared",
        conflict_validation_status="encoded" if conflicts_with_goal_ids else "no_known_conflict",
        source_validation_status="source_artifacts_present",
    )


class GoalSchemaTests(unittest.TestCase):
    def test_stable_json_and_hash_are_order_invariant(self) -> None:
        left = {"b": [2, 1], "a": {"x": 3}}
        right = {"a": {"x": 3}, "b": [2, 1]}
        self.assertEqual(stable_json(left), stable_json(right))
        self.assertEqual(stable_hash(left), stable_hash(right))

    def test_dataframe_roundtrips_json_columns_and_ids(self) -> None:
        record = complete_goal()
        df = dataframe_from_goal_records([record])
        row = df.iloc[0]
        self.assertEqual(row["goal_uid"], goal_uid("E07", "unit-record"))
        self.assertTrue(str(row["canonical_goal_id"]).startswith("goalcanon:"))
        self.assertEqual(json.loads(row["linked_world_ids_json"]), ["world:unit"])
        self.assertEqual(json.loads(row["linked_policy_ids_json"]), ["policy:unit"])
        self.assertEqual(json.loads(row["metric_contract_json"]), {"direction": "higher_is_better"})

    def test_goal_table_validation_accepts_complete_partial_and_conflicts(self) -> None:
        complete = complete_goal(source_record_id="complete", source_goal_id="unit:complete", conflicts_with_goal_ids=("unit:partial",))
        partial = complete_goal(source_record_id="partial", source_goal_id="unit:partial", partial=True)
        df = dataframe_from_goal_records([complete, partial])
        checks = validate_goal_table(
            df,
            required_sources=("E07",),
            s01_world_ids=("world:unit",),
            known_policy_ids=("policy:unit",),
        )
        results = dict(zip(checks["validation_case"], checks["success"], strict=True))
        self.assertTrue(results["required_columns_present"])
        self.assertTrue(results["goal_uids_unique"])
        self.assertTrue(results["target_directions_declared"])
        self.assertTrue(results["metric_units_declared"])
        self.assertTrue(results["partial_records_have_limitations"])
        self.assertTrue(results["conflict_encodings_resolve"])
        self.assertTrue(results["s01_world_goal_links_cover_inventory"])
        self.assertTrue(results["s02_policy_links_resolve"])
        self.assertTrue(results["source_artifacts_exist"])
        self.assertTrue(results["json_columns_roundtrip"])

    def test_goal_table_validation_detects_partial_without_limitations(self) -> None:
        complete = complete_goal(source_record_id="complete", source_goal_id="unit:complete", conflicts_with_goal_ids=("unit:partial",))
        partial = complete_goal(source_record_id="partial", source_goal_id="unit:partial", partial=True, limitations=())
        df = dataframe_from_goal_records([complete, partial])
        checks = validate_goal_table(df, required_sources=("E07",), s01_world_ids=("world:unit",), known_policy_ids=("policy:unit",))
        results = dict(zip(checks["validation_case"], checks["success"], strict=True))
        self.assertFalse(results["partial_records_have_limitations"])

    def test_goal_table_validation_detects_unresolved_world_and_policy_links(self) -> None:
        complete = complete_goal(source_record_id="complete", source_goal_id="unit:complete", conflicts_with_goal_ids=("unit:partial",), linked_policy_ids=("missing-policy",))
        partial = complete_goal(source_record_id="partial", source_goal_id="unit:partial", partial=True, linked_world_ids=("world:missing",))
        df = dataframe_from_goal_records([complete, partial])
        checks = validate_goal_table(df, required_sources=("E07",), s01_world_ids=("world:unit",), known_policy_ids=("policy:unit",))
        results = dict(zip(checks["validation_case"], checks["success"], strict=True))
        self.assertFalse(results["s01_world_goal_links_cover_inventory"])
        self.assertFalse(results["s02_policy_links_resolve"])


if __name__ == "__main__":
    unittest.main()
