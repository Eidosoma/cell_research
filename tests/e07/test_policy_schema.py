"""Tests for E07 S02 policy representation metadata."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.e07.policy_schema import (
    PolicyRepresentation,
    dataframe_from_policy_records,
    policy_uid,
    sha256_text,
    stable_hash,
    stable_json,
    validate_policy_table,
)


THIS_FILE = Path(__file__)


def complete_policy(
    source_experiment_id: str = "E07",
    source_record_id: str = "unit-record",
    *,
    metadata_only: bool = False,
    limitations: tuple[str, ...] = ("unit limitation",),
    applicable_world_ids: tuple[str, ...] = ("world:unit",),
) -> PolicyRepresentation:
    return PolicyRepresentation(
        source_experiment_id=source_experiment_id,
        source_step_id="S02",
        source_record_id=source_record_id,
        source_policy_id="unit-policy",
        display_name="Unit policy",
        source_category="unit",
        policy_family="unit_family",
        algorithm="unit_algorithm",
        representation_type="metadata_only" if metadata_only else "dsl",
        abstraction_kind="metadata_reference" if metadata_only else "local_rule_dsl",
        interpretable=not metadata_only,
        metadata_only=metadata_only,
        requires_memory=False,
        requires_signaling=False,
        uses_global_oracle=False,
        information_access="local",
        execution_backend="metadata_only" if metadata_only else "dsl_interpreter",
        direction_support="bidirectional",
        observation_contract="unit neighborhood",
        action_vocabulary=("swap", "wait"),
        state_variables=("left", "right"),
        source_artifacts=(str(THIS_FILE),),
        applicable_world_ids=applicable_world_ids,
        limitations=limitations,
        upstream_metrics={"score": 1.0},
        representation_payload={"kind": "unit", "rules": [{"action": "swap"}]},
        declared_dsl_sha256=sha256_text("unit") if not metadata_only else None,
        computed_dsl_canonical_sha256=sha256_text("unit") if not metadata_only else None,
        computed_dsl_source_sha256=sha256_text("source") if not metadata_only else None,
        source_payload_sha256=sha256_text("payload"),
        hash_validation_status="matches_declared_canonical_dsl_hash" if not metadata_only else "metadata_only_no_artifact_hash",
        parser_validation_status="parsed" if not metadata_only else "metadata_only_explicit",
    )


class PolicySchemaTests(unittest.TestCase):
    def test_stable_json_and_hash_are_order_invariant(self) -> None:
        left = {"b": [2, 1], "a": {"x": 3}}
        right = {"a": {"x": 3}, "b": [2, 1]}
        self.assertEqual(stable_json(left), stable_json(right))
        self.assertEqual(stable_hash(left), stable_hash(right))

    def test_dataframe_roundtrips_json_columns_and_ids(self) -> None:
        record = complete_policy()
        df = dataframe_from_policy_records([record])
        row = df.iloc[0]
        self.assertEqual(row["policy_uid"], policy_uid("E07", "unit-record"))
        self.assertTrue(str(row["canonical_policy_id"]).startswith("canon:"))
        self.assertEqual(json.loads(row["action_vocabulary_json"]), ["swap", "wait"])
        self.assertEqual(json.loads(row["state_variables_json"]), ["left", "right"])
        self.assertEqual(json.loads(row["source_artifacts_json"]), [str(THIS_FILE)])

    def test_policy_table_validation_accepts_parsed_and_metadata_rows(self) -> None:
        parsed = complete_policy(source_record_id="parsed")
        metadata = complete_policy(source_record_id="metadata", metadata_only=True)
        df = dataframe_from_policy_records([parsed, metadata])
        checks = validate_policy_table(df, required_sources=("E07",), s01_world_ids=("world:unit",))
        results = dict(zip(checks["validation_case"], checks["success"], strict=True))
        self.assertTrue(results["required_columns_present"])
        self.assertTrue(results["policy_uids_unique"])
        self.assertTrue(results["required_sources_covered"])
        self.assertTrue(results["representations_parse_or_are_explicit_metadata"])
        self.assertTrue(results["hashes_match_or_are_documented"])
        self.assertTrue(results["metadata_only_records_have_limitations"])
        self.assertTrue(results["s01_world_links_valid"])
        self.assertTrue(results["source_artifacts_exist"])

    def test_policy_table_validation_detects_metadata_without_limitations(self) -> None:
        df = dataframe_from_policy_records([complete_policy(metadata_only=True, limitations=())])
        checks = validate_policy_table(df, required_sources=("E07",), s01_world_ids=("world:unit",))
        results = dict(zip(checks["validation_case"], checks["success"], strict=True))
        self.assertFalse(results["metadata_only_records_have_limitations"])

    def test_policy_table_validation_detects_bad_world_links(self) -> None:
        df = dataframe_from_policy_records([complete_policy(applicable_world_ids=("world:missing",))])
        checks = validate_policy_table(df, required_sources=("E07",), s01_world_ids=("world:unit",))
        results = dict(zip(checks["validation_case"], checks["success"], strict=True))
        self.assertFalse(results["s01_world_links_valid"])


if __name__ == "__main__":
    unittest.main()
