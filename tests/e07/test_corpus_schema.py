"""Tests for E07 S04 unified corpus schema helpers."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e07.corpus_schema import (
    CORPUS_COLUMNS,
    MISSINGNESS_COLUMNS,
    SOURCE_MANIFEST_COLUMNS,
    dataframe_from_records,
    stable_hash,
    stable_json,
    validate_unified_corpus,
)


def complete_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "source_experiment_id": "E07",
        "source_artifact_path": __file__,
        "source_artifact_sha256": "unit-sha",
        "source_table": "unit_table",
        "source_row_index": 0,
        "source_record_id": "unit-row",
        "source_metric_name": "unit_metric",
        "metric_value": 1.0,
        "metric_unit": "unitless",
        "metric_direction": "maximize",
        "metric_family": "unit_family",
        "evidence_kind": "metric_observation",
        "row_granularity": "unit_row",
        "world_id": "world:unit",
        "world_link_status": "resolved",
        "source_policy_id": "unit_policy",
        "policy_uid": "policy:unit",
        "canonical_policy_id": "canon:unit",
        "policy_link_status": "resolved",
        "source_goal_id": "unit_goal",
        "goal_uid": "goal:unit",
        "canonical_goal_id": "goalcanon:unit",
        "goal_link_status": "resolved",
        "perturbation_type": "none",
        "seed": "1",
        "repeat_index": "0",
        "trace_artifacts_json": [],
        "missingness_json": {},
        "source_columns_json": {"unit_metric": 1.0},
    }
    record.update(overrides)
    return record


class CorpusSchemaTests(unittest.TestCase):
    def test_stable_json_and_hash_are_order_invariant(self) -> None:
        left = {"b": [2, 1], "a": {"x": 3}}
        right = {"a": {"x": 3}, "b": [2, 1]}
        self.assertEqual(stable_json(left), stable_json(right))
        self.assertEqual(stable_hash(left), stable_hash(right))

    def test_dataframe_from_records_fills_ids_hashes_and_json(self) -> None:
        df = dataframe_from_records([complete_record()])
        row = df.iloc[0]
        self.assertEqual(tuple(df.columns), CORPUS_COLUMNS)
        self.assertTrue(str(row["corpus_row_id"]).startswith("e07corp:"))
        self.assertTrue(str(row["record_hash"]))
        self.assertEqual(json.loads(row["trace_artifacts_json"]), [])
        self.assertEqual(json.loads(row["missingness_json"]), {})

    def test_validation_accepts_resolved_and_documented_missing_records(self) -> None:
        corpus = dataframe_from_records(
            [
                complete_record(),
                complete_record(
                    source_row_index=1,
                    source_metric_name="unkeyed_metric",
                    world_id="",
                    world_link_status="unkeyed_missing_world",
                    policy_uid="",
                    policy_link_status="unkeyed_missing_policy",
                    goal_uid="",
                    goal_link_status="world_unkeyed",
                    missingness_json={"world": "unit test missing key"},
                ),
            ]
        )
        missingness = pd.DataFrame(
            [
                {
                    "missingness_id": "miss:unit",
                    "source_experiment_id": "E07",
                    "source_artifact_path": __file__,
                    "source_table": "unit_table",
                    "missingness_type": "unkeyed_world",
                    "affected_rows": 1,
                    "affected_metric_rows": 1,
                    "detail": "unit test",
                    "documented_limitation": "unit test",
                }
            ],
            columns=list(MISSINGNESS_COLUMNS),
        )
        manifest = pd.DataFrame(
            [
                {
                    "source_experiment_id": "E07",
                    "source_table": "unit_table",
                    "source_artifact_path": __file__,
                    "source_artifact_sha256": "unit-sha",
                    "row_count": 2,
                    "column_count": 1,
                    "selected_metric_column_count": 1,
                    "metric_observation_rows": 2,
                    "world_keyed_rows": 1,
                    "world_unkeyed_rows": 1,
                    "policy_keyed_rows": 1,
                    "policy_unkeyed_rows": 1,
                    "goal_keyed_rows": 1,
                    "goal_unkeyed_rows": 1,
                    "ingest_status": "ingested",
                }
            ],
            columns=list(SOURCE_MANIFEST_COLUMNS),
        )
        checks = validate_unified_corpus(
            corpus,
            missingness,
            manifest,
            required_sources=("E07",),
            s01_world_ids=("world:unit",),
            known_policy_ids=("policy:unit",),
            known_goal_ids=("goal:unit",),
        )
        results = dict(zip(checks["validation_case"], checks["success"], strict=True))
        self.assertTrue(all(results.values()))

    def test_validation_detects_bad_foreign_keys(self) -> None:
        corpus = dataframe_from_records([complete_record(world_id="world:missing", policy_uid="policy:missing", goal_uid="goal:missing")])
        missingness = pd.DataFrame(columns=list(MISSINGNESS_COLUMNS))
        manifest = pd.DataFrame(
            [
                {
                    "source_experiment_id": "E07",
                    "source_table": "unit_table",
                    "source_artifact_path": __file__,
                    "source_artifact_sha256": "unit-sha",
                    "row_count": 1,
                    "column_count": 1,
                    "selected_metric_column_count": 1,
                    "metric_observation_rows": 1,
                    "world_keyed_rows": 1,
                    "world_unkeyed_rows": 0,
                    "policy_keyed_rows": 1,
                    "policy_unkeyed_rows": 0,
                    "goal_keyed_rows": 1,
                    "goal_unkeyed_rows": 0,
                    "ingest_status": "ingested",
                }
            ],
            columns=list(SOURCE_MANIFEST_COLUMNS),
        )
        checks = validate_unified_corpus(
            corpus,
            missingness,
            manifest,
            required_sources=("E07",),
            s01_world_ids=("world:unit",),
            known_policy_ids=("policy:unit",),
            known_goal_ids=("goal:unit"),
        )
        results = dict(zip(checks["validation_case"], checks["success"], strict=True))
        self.assertFalse(results["s01_world_foreign_keys_resolve"])
        self.assertFalse(results["s02_policy_foreign_keys_resolve"])
        self.assertFalse(results["s03_goal_foreign_keys_resolve"])


if __name__ == "__main__":
    unittest.main()
