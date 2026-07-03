"""Goal representation schema for E07 cross-world analysis."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


GOAL_SCHEMA_VERSION = "eidosoma.e07.goal_representation.v1"

ALLOWED_TARGET_DIRECTIONS = {
    "maximize",
    "minimize",
    "match_target",
    "maintain_at_or_above",
    "maintain_range",
    "mixed_profile",
    "constraint_set",
    "partial_unknown",
}

GOAL_COLUMNS = (
    "schema_version",
    "goal_uid",
    "canonical_goal_id",
    "source_experiment_id",
    "source_step_id",
    "source_record_id",
    "source_goal_id",
    "display_name",
    "goal_family",
    "goal_kind",
    "abstraction_kind",
    "target_structure",
    "representation_status",
    "partial",
    "target_direction",
    "unit",
    "primary_metric_id",
    "metric_family",
    "target_value",
    "lower_bound",
    "upper_bound",
    "zero_point_definition",
    "energy_function",
    "constraint_predicate",
    "conflict_group_id",
    "conflicts_with_goal_ids_json",
    "compatible_with_goal_ids_json",
    "linked_world_ids_json",
    "linked_policy_ids_json",
    "source_artifacts_json",
    "limitations_json",
    "metric_contract_json",
    "representation_payload_json",
    "target_hash",
    "representation_hash",
    "direction_validation_status",
    "unit_validation_status",
    "conflict_validation_status",
    "source_validation_status",
    "record_hash",
)


def sanitize_json(value: Any) -> Any:
    """Convert pandas/numpy/NaN values into strict JSON-compatible values."""

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def stable_json(payload: Any) -> str:
    return json.dumps(sanitize_json(payload), sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def goal_uid(source_experiment_id: str, source_record_id: str) -> str:
    digest = stable_hash({"sourceExperimentId": source_experiment_id, "sourceRecordId": source_record_id})
    return f"e07goal:{digest[:20]}"


def canonical_goal_id(representation_hash: str) -> str:
    return f"goalcanon:{representation_hash[:20]}"


@dataclass(frozen=True)
class GoalRepresentation:
    """One abstract goal, target predicate, metric objective, or explicit partial proxy."""

    source_experiment_id: str
    source_step_id: str
    source_record_id: str
    source_goal_id: str
    display_name: str
    goal_family: str
    goal_kind: str
    abstraction_kind: str
    target_structure: str
    representation_status: str
    partial: bool
    target_direction: str
    unit: str
    primary_metric_id: str
    metric_family: str
    target_value: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    zero_point_definition: str = ""
    energy_function: str = ""
    constraint_predicate: str = ""
    conflict_group_id: str = ""
    conflicts_with_goal_ids: tuple[str, ...] = ()
    compatible_with_goal_ids: tuple[str, ...] = ()
    linked_world_ids: tuple[str, ...] = ()
    linked_policy_ids: tuple[str, ...] = ()
    source_artifacts: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    metric_contract: Mapping[str, Any] = field(default_factory=dict)
    representation_payload: Mapping[str, Any] = field(default_factory=dict)
    target_hash: str | None = None
    representation_hash: str | None = None
    direction_validation_status: str = "not_checked"
    unit_validation_status: str = "not_checked"
    conflict_validation_status: str = "not_checked"
    source_validation_status: str = "not_checked"
    schema_version: str = GOAL_SCHEMA_VERSION

    def with_hashes(self) -> "GoalRepresentation":
        representation_hash = self.representation_hash or stable_hash(
            {
                "sourceGoalId": self.source_goal_id,
                "goalFamily": self.goal_family,
                "goalKind": self.goal_kind,
                "targetDirection": self.target_direction,
                "primaryMetricId": self.primary_metric_id,
                "targetStructure": self.target_structure,
                "targetValue": self.target_value,
                "metricContract": dict(self.metric_contract),
                "payload": dict(self.representation_payload),
                "targetHash": self.target_hash,
            }
        )
        return GoalRepresentation(**{**self.__dict__, "representation_hash": representation_hash})

    def to_flat_dict(self) -> dict[str, Any]:
        checked = self.with_hashes()
        payload = {
            "schema_version": checked.schema_version,
            "goal_uid": goal_uid(checked.source_experiment_id, checked.source_record_id),
            "canonical_goal_id": canonical_goal_id(str(checked.representation_hash)),
            "source_experiment_id": checked.source_experiment_id,
            "source_step_id": checked.source_step_id,
            "source_record_id": checked.source_record_id,
            "source_goal_id": checked.source_goal_id,
            "display_name": checked.display_name,
            "goal_family": checked.goal_family,
            "goal_kind": checked.goal_kind,
            "abstraction_kind": checked.abstraction_kind,
            "target_structure": checked.target_structure,
            "representation_status": checked.representation_status,
            "partial": bool(checked.partial),
            "target_direction": checked.target_direction,
            "unit": checked.unit,
            "primary_metric_id": checked.primary_metric_id,
            "metric_family": checked.metric_family,
            "target_value": checked.target_value,
            "lower_bound": checked.lower_bound,
            "upper_bound": checked.upper_bound,
            "zero_point_definition": checked.zero_point_definition,
            "energy_function": checked.energy_function,
            "constraint_predicate": checked.constraint_predicate,
            "conflict_group_id": checked.conflict_group_id,
            "conflicts_with_goal_ids_json": stable_json(list(checked.conflicts_with_goal_ids)),
            "compatible_with_goal_ids_json": stable_json(list(checked.compatible_with_goal_ids)),
            "linked_world_ids_json": stable_json(list(checked.linked_world_ids)),
            "linked_policy_ids_json": stable_json(list(checked.linked_policy_ids)),
            "source_artifacts_json": stable_json(list(checked.source_artifacts)),
            "limitations_json": stable_json(list(checked.limitations)),
            "metric_contract_json": stable_json(dict(checked.metric_contract)),
            "representation_payload_json": stable_json(dict(checked.representation_payload)),
            "target_hash": checked.target_hash,
            "representation_hash": checked.representation_hash,
            "direction_validation_status": checked.direction_validation_status,
            "unit_validation_status": checked.unit_validation_status,
            "conflict_validation_status": checked.conflict_validation_status,
            "source_validation_status": checked.source_validation_status,
        }
        payload["record_hash"] = stable_hash({key: payload[key] for key in payload if key != "record_hash"})
        return payload


def dataframe_from_goal_records(records: Sequence[GoalRepresentation]) -> pd.DataFrame:
    rows = [record.to_flat_dict() for record in records]
    df = pd.DataFrame(rows, columns=list(GOAL_COLUMNS))
    if not df.empty:
        df = df.sort_values(["source_experiment_id", "source_record_id"], kind="mergesort").reset_index(drop=True)
    return df


def parse_json_list(text: Any) -> list[Any]:
    if text is None:
        return []
    try:
        return list(json.loads(str(text)))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def validate_goal_table(
    table: pd.DataFrame,
    *,
    required_sources: Iterable[str] = ("E01", "E04", "E05", "E06", "S01_derived"),
    s01_world_ids: Iterable[str] = (),
    known_policy_ids: Iterable[str] = (),
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    missing_columns = [column for column in GOAL_COLUMNS if column not in table.columns]
    checks.append(
        {
            "validation_case": "required_columns_present",
            "success": not missing_columns,
            "detail": "all goal columns present" if not missing_columns else f"missing columns: {missing_columns}",
        }
    )

    duplicate_uid_count = int(table["goal_uid"].duplicated().sum()) if "goal_uid" in table else len(table)
    checks.append(
        {
            "validation_case": "goal_uids_unique",
            "success": duplicate_uid_count == 0,
            "detail": f"duplicate goal_uid rows: {duplicate_uid_count}",
        }
    )

    observed_sources = set(table.get("source_experiment_id", pd.Series(dtype=str)).astype(str))
    missing_sources = sorted(set(required_sources) - observed_sources)
    checks.append(
        {
            "validation_case": "required_sources_covered",
            "success": not missing_sources,
            "detail": "covered required goal sources" if not missing_sources else f"missing sources: {missing_sources}",
        }
    )

    direction_failures = table[
        table["target_direction"].isna()
        | ~table["target_direction"].astype(str).isin(ALLOWED_TARGET_DIRECTIONS)
        | (table["direction_validation_status"].astype(str).str.len() == 0)
    ]
    checks.append(
        {
            "validation_case": "target_directions_declared",
            "success": direction_failures.empty,
            "detail": "all rows declare allowed target directions" if direction_failures.empty else f"direction failures: {len(direction_failures)}",
        }
    )

    unit_failures = table[
        table["unit"].isna()
        | (table["unit"].astype(str).str.len() == 0)
        | (table["unit_validation_status"].astype(str).str.len() == 0)
    ]
    checks.append(
        {
            "validation_case": "metric_units_declared",
            "success": unit_failures.empty,
            "detail": "all rows declare metric units or proxy units" if unit_failures.empty else f"unit failures: {len(unit_failures)}",
        }
    )

    partial_rows = table[table["partial"].astype(bool)] if "partial" in table else table.iloc[0:0]
    undocumented_partial = partial_rows[partial_rows["limitations_json"].map(lambda text: len(parse_json_list(text)) == 0)]
    checks.append(
        {
            "validation_case": "partial_records_have_limitations",
            "success": undocumented_partial.empty,
            "detail": "partial rows document limitations" if undocumented_partial.empty else f"undocumented partial rows: {len(undocumented_partial)}",
        }
    )

    source_goal_ids = set(table.get("source_goal_id", pd.Series(dtype=str)).astype(str))
    conflict_failures: list[str] = []
    conflict_encoded_count = 0
    for row in table.to_dict(orient="records"):
        conflicts = [str(item) for item in parse_json_list(row.get("conflicts_with_goal_ids_json"))]
        if conflicts:
            conflict_encoded_count += 1
        if str(row.get("conflict_validation_status", "")) == "encoded" and not conflicts:
            conflict_failures.append(str(row.get("source_goal_id")))
        missing = [goal_id for goal_id in conflicts if goal_id not in source_goal_ids]
        if missing:
            conflict_failures.append(f"{row.get('source_goal_id')} missing {missing}")
    checks.append(
        {
            "validation_case": "conflict_encodings_resolve",
            "success": not conflict_failures and conflict_encoded_count > 0,
            "detail": f"encoded conflict rows: {conflict_encoded_count}"
            if not conflict_failures
            else f"conflict failures: {len(conflict_failures)}",
        }
    )

    world_id_set = set(str(item) for item in s01_world_ids)
    linked_worlds: set[str] = set()
    bad_world_rows: list[str] = []
    if world_id_set:
        for row in table.to_dict(orient="records"):
            for world_id in parse_json_list(row.get("linked_world_ids_json")):
                world_id = str(world_id)
                linked_worlds.add(world_id)
                if world_id not in world_id_set:
                    bad_world_rows.append(str(row.get("source_goal_id")))
        missing_worlds = world_id_set - linked_worlds
    else:
        missing_worlds = set()
    checks.append(
        {
            "validation_case": "s01_world_goal_links_cover_inventory",
            "success": not bad_world_rows and not missing_worlds,
            "detail": "all S01 worlds linked to at least one goal"
            if not bad_world_rows and not missing_worlds
            else f"bad world rows: {len(bad_world_rows)}; missing worlds: {len(missing_worlds)}",
        }
    )

    known_policy_set = set(str(item) for item in known_policy_ids if str(item))
    missing_policy_links: list[str] = []
    if known_policy_set:
        for row in table.to_dict(orient="records"):
            for policy_id in parse_json_list(row.get("linked_policy_ids_json")):
                if str(policy_id) not in known_policy_set:
                    missing_policy_links.append(str(policy_id))
    checks.append(
        {
            "validation_case": "s02_policy_links_resolve",
            "success": not missing_policy_links,
            "detail": "all explicit policy links resolve in S02 policy table"
            if not missing_policy_links
            else f"missing policy links: {len(missing_policy_links)}",
        }
    )

    source_missing: list[str] = []
    for row in table.to_dict(orient="records"):
        for path in parse_json_list(row.get("source_artifacts_json")):
            if path and not Path(path).exists():
                source_missing.append(str(path))
    checks.append(
        {
            "validation_case": "source_artifacts_exist",
            "success": not source_missing,
            "detail": "all listed source artifacts exist" if not source_missing else f"missing source artifacts: {len(source_missing)}",
        }
    )

    json_failures = 0
    for column in [column for column in GOAL_COLUMNS if column.endswith("_json")]:
        for value in table[column].tolist():
            try:
                json.loads(str(value))
            except (json.JSONDecodeError, TypeError, ValueError):
                json_failures += 1
    checks.append(
        {
            "validation_case": "json_columns_roundtrip",
            "success": json_failures == 0,
            "detail": "all JSON columns parse" if json_failures == 0 else f"JSON parse failures: {json_failures}",
        }
    )

    return pd.DataFrame(checks)
