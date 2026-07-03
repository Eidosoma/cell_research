"""World metadata schema for E07 cross-substrate abstraction.

The E07 S01 contract represents each upstream world or task as the tuple
specified in the research plan: state space, local observations, action set,
transition rules, goal predicate, perturbation model, and measurements.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


WORLD_SCHEMA_VERSION = "eidosoma.e07.world_schema.v1"

REQUIRED_TUPLE_FIELDS = (
    "state_space",
    "local_observations",
    "action_set",
    "transition_rules",
    "goal_predicate",
    "perturbation_model",
    "measurement_functions",
)

INVENTORY_COLUMNS = (
    "schema_version",
    "world_id",
    "experiment_id",
    "source_step_id",
    "record_granularity",
    "world_family",
    "task_id",
    "task_label",
    "substrate_kind",
    "state_space",
    "local_observations",
    "action_set",
    "transition_rules",
    "goal_predicate",
    "perturbation_model",
    "measurement_functions",
    "scheduler",
    "source_artifacts_json",
    "metadata_json",
    "missing_fields_json",
    "completeness",
    "record_hash",
)


@dataclass(frozen=True)
class WorldRecord:
    """One normalized world or task entry in the E07 S01 inventory."""

    world_id: str
    experiment_id: str
    source_step_id: str
    record_granularity: str
    world_family: str
    task_id: str
    task_label: str
    substrate_kind: str
    state_space: str
    local_observations: str
    action_set: str
    transition_rules: str
    goal_predicate: str
    perturbation_model: str
    measurement_functions: str
    scheduler: str = "not_specified"
    source_artifacts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    completeness: str = "unchecked"
    schema_version: str = WORLD_SCHEMA_VERSION

    def with_validation(self) -> "WorldRecord":
        missing = tuple(field_name for field_name in REQUIRED_TUPLE_FIELDS if _is_missing(getattr(self, field_name)))
        completeness = "complete" if not missing else "partial_explicit_missing"
        return replace(self, missing_fields=missing, completeness=completeness)

    def to_flat_dict(self) -> dict[str, Any]:
        checked = self.with_validation()
        payload = {
            "schema_version": checked.schema_version,
            "world_id": checked.world_id,
            "experiment_id": checked.experiment_id,
            "source_step_id": checked.source_step_id,
            "record_granularity": checked.record_granularity,
            "world_family": checked.world_family,
            "task_id": checked.task_id,
            "task_label": checked.task_label,
            "substrate_kind": checked.substrate_kind,
            "state_space": checked.state_space,
            "local_observations": checked.local_observations,
            "action_set": checked.action_set,
            "transition_rules": checked.transition_rules,
            "goal_predicate": checked.goal_predicate,
            "perturbation_model": checked.perturbation_model,
            "measurement_functions": checked.measurement_functions,
            "scheduler": checked.scheduler,
            "source_artifacts_json": stable_json(list(checked.source_artifacts)),
            "metadata_json": stable_json(dict(checked.metadata)),
            "missing_fields_json": stable_json(list(checked.missing_fields)),
            "completeness": checked.completeness,
        }
        payload["record_hash"] = stable_hash({key: payload[key] for key in payload if key != "record_hash"})
        return payload


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"unknown", "missing", "not_available", "not documented"}


def stable_json(payload: Any) -> str:
    """Return stable, compact JSON for nested CSV fields and hashes."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def slugify(value: Any, *, max_length: int = 72) -> str:
    text = str(value).strip().lower()
    chars: list[str] = []
    previous_dash = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    slug = "".join(chars).strip("-")
    if not slug:
        slug = "unnamed"
    return slug[:max_length].strip("-") or "unnamed"


def make_world_id(experiment_id: str, source_step_id: str, task_id: str, *, family: str = "") -> str:
    core = ":".join(
        item
        for item in (
            str(experiment_id).lower(),
            str(source_step_id).lower(),
            slugify(family, max_length=32) if family else "",
            slugify(task_id, max_length=80),
        )
        if item
    )
    return f"world:{core}"


def dataframe_from_records(records: Sequence[WorldRecord]) -> pd.DataFrame:
    rows = [record.to_flat_dict() for record in records]
    df = pd.DataFrame(rows, columns=list(INVENTORY_COLUMNS))
    if not df.empty:
        df = df.sort_values(["experiment_id", "source_step_id", "world_id"], kind="mergesort").reset_index(drop=True)
    return df


def validate_inventory(
    inventory: pd.DataFrame,
    *,
    required_experiments: Iterable[str] = ("E01", "E02", "E03", "E04", "E05", "E06"),
    require_complete_records: bool = False,
) -> pd.DataFrame:
    """Validate inventory structure and coverage.

    Partial records are acceptable for S01 only when their missing tuple fields
    are explicitly recorded in ``missing_fields_json``.
    """

    checks: list[dict[str, Any]] = []
    columns = set(inventory.columns)
    missing_columns = [column for column in INVENTORY_COLUMNS if column not in columns]
    checks.append(
        {
            "validation_case": "required_columns_present",
            "success": not missing_columns,
            "detail": "all inventory columns present" if not missing_columns else f"missing columns: {missing_columns}",
        }
    )

    if "world_id" in inventory:
        duplicate_count = int(inventory["world_id"].duplicated().sum())
    else:
        duplicate_count = len(inventory)
    checks.append(
        {
            "validation_case": "world_ids_unique",
            "success": duplicate_count == 0,
            "detail": f"duplicate world_id rows: {duplicate_count}",
        }
    )

    required_set = set(str(item) for item in required_experiments)
    observed_set = set(inventory.get("experiment_id", pd.Series(dtype=str)).astype(str))
    missing_experiments = sorted(required_set - observed_set)
    checks.append(
        {
            "validation_case": "required_upstream_experiments_covered",
            "success": not missing_experiments,
            "detail": "covered E01-E06" if not missing_experiments else f"missing experiments: {missing_experiments}",
        }
    )

    explicit_missing_success = True
    undocumented_rows: list[str] = []
    if not inventory.empty:
        for row in inventory.to_dict(orient="records"):
            missing = json.loads(str(row.get("missing_fields_json", "[]")))
            completeness = str(row.get("completeness", ""))
            if missing and completeness != "partial_explicit_missing":
                explicit_missing_success = False
                undocumented_rows.append(str(row.get("world_id", "")))
            if require_complete_records and missing:
                explicit_missing_success = False
                undocumented_rows.append(str(row.get("world_id", "")))
    checks.append(
        {
            "validation_case": "missing_fields_explicitly_marked",
            "success": explicit_missing_success,
            "detail": "all tuple gaps are explicit" if explicit_missing_success else f"undocumented rows: {undocumented_rows[:10]}",
        }
    )

    json_success = True
    json_errors: list[str] = []
    for column in ("source_artifacts_json", "metadata_json", "missing_fields_json"):
        if column not in inventory:
            json_success = False
            json_errors.append(f"{column}: column missing")
            continue
        for idx, value in enumerate(inventory[column].head(250).tolist()):
            try:
                json.loads(str(value))
            except json.JSONDecodeError as exc:
                json_success = False
                json_errors.append(f"{column} row {idx}: {exc}")
                break
    checks.append(
        {
            "validation_case": "json_columns_roundtrip",
            "success": json_success,
            "detail": "json columns parse" if json_success else "; ".join(json_errors),
        }
    )

    nonempty_tuple = bool(len(inventory) > 0)
    if nonempty_tuple:
        required_present = [
            field_name
            for field_name in REQUIRED_TUPLE_FIELDS
            if field_name in inventory and inventory[field_name].astype(str).str.strip().ne("").any()
        ]
        nonempty_tuple = len(required_present) == len(REQUIRED_TUPLE_FIELDS)
    checks.append(
        {
            "validation_case": "tuple_fields_populated_somewhere",
            "success": nonempty_tuple,
            "detail": "all required tuple fields have populated values in at least one record",
        }
    )

    return pd.DataFrame(checks)


def source_paths_exist(inventory: pd.DataFrame) -> pd.DataFrame:
    """Check whether recorded source artifact paths currently exist."""

    rows: list[dict[str, Any]] = []
    for record in inventory.to_dict(orient="records"):
        paths = json.loads(str(record.get("source_artifacts_json", "[]")))
        missing = [path for path in paths if path and not Path(path).exists()]
        rows.append(
            {
                "world_id": record["world_id"],
                "source_artifact_count": len(paths),
                "missing_source_artifact_count": len(missing),
                "missing_source_artifacts_json": stable_json(missing),
                "success": len(missing) == 0,
            }
        )
    return pd.DataFrame(rows)
