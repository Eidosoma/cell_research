"""Unified E07 behavior-corpus schema and validation helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


CORPUS_SCHEMA_VERSION = "eidosoma.e07.unified_behavior_corpus.v1"

CORPUS_COLUMNS = (
    "schema_version",
    "corpus_row_id",
    "source_experiment_id",
    "source_artifact_path",
    "source_artifact_sha256",
    "source_table",
    "source_row_index",
    "source_record_id",
    "source_metric_name",
    "metric_value",
    "metric_unit",
    "metric_direction",
    "metric_family",
    "evidence_kind",
    "row_granularity",
    "world_id",
    "world_link_status",
    "source_policy_id",
    "policy_uid",
    "canonical_policy_id",
    "policy_link_status",
    "source_goal_id",
    "goal_uid",
    "canonical_goal_id",
    "goal_link_status",
    "perturbation_type",
    "seed",
    "repeat_index",
    "trace_artifacts_json",
    "missingness_json",
    "source_columns_json",
    "raw_row_hash",
    "record_hash",
)

MISSINGNESS_COLUMNS = (
    "missingness_id",
    "source_experiment_id",
    "source_artifact_path",
    "source_table",
    "missingness_type",
    "affected_rows",
    "affected_metric_rows",
    "detail",
    "documented_limitation",
)

SOURCE_MANIFEST_COLUMNS = (
    "source_experiment_id",
    "source_table",
    "source_artifact_path",
    "source_artifact_sha256",
    "row_count",
    "column_count",
    "selected_metric_column_count",
    "metric_observation_rows",
    "world_keyed_rows",
    "world_unkeyed_rows",
    "policy_keyed_rows",
    "policy_unkeyed_rows",
    "goal_keyed_rows",
    "goal_unkeyed_rows",
    "ingest_status",
)

JSON_COLUMNS = ("trace_artifacts_json", "missingness_json", "source_columns_json")

MISSING_WORLD_STATUSES = {"unkeyed_missing_world", "not_applicable", "source_table_missing"}
MISSING_POLICY_STATUSES = {"unkeyed_missing_policy", "not_applicable", "source_table_missing"}
MISSING_GOAL_STATUSES = {"unkeyed_missing_goal", "world_unkeyed", "not_applicable", "source_table_missing"}


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
    return json.dumps(sanitize_json(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json(text: Any) -> Any:
    if text is None:
        return None
    try:
        return json.loads(str(text))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def corpus_row_id(source_artifact_path: str, source_row_index: int, metric_name: str) -> str:
    digest = stable_hash(
        {
            "sourceArtifactPath": source_artifact_path,
            "sourceRowIndex": int(source_row_index),
            "metricName": metric_name,
        }
    )
    return f"e07corp:{digest[:24]}"


def finalize_corpus_record(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = {column: record.get(column, "") for column in CORPUS_COLUMNS if column != "record_hash"}
    payload["schema_version"] = payload.get("schema_version") or CORPUS_SCHEMA_VERSION
    payload["source_row_index"] = int(payload.get("source_row_index") or 0)
    payload["corpus_row_id"] = payload.get("corpus_row_id") or corpus_row_id(
        str(payload.get("source_artifact_path", "")),
        int(payload["source_row_index"]),
        str(payload.get("source_metric_name", "")),
    )
    for column in JSON_COLUMNS:
        value = payload.get(column)
        if value in ("", None):
            payload[column] = stable_json({})
        elif not isinstance(value, str):
            payload[column] = stable_json(value)
    payload["raw_row_hash"] = payload.get("raw_row_hash") or hashlib.sha256(str(payload.get("source_columns_json", "{}")).encode("utf-8")).hexdigest()
    hash_text = "\x1f".join(str(payload.get(column, "")) for column in CORPUS_COLUMNS if column != "record_hash")
    payload["record_hash"] = hashlib.sha256(hash_text.encode("utf-8")).hexdigest()
    return payload


def dataframe_from_records(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [finalize_corpus_record(record) for record in records]
    df = pd.DataFrame(rows, columns=list(CORPUS_COLUMNS))
    if not df.empty:
        df = df.sort_values(["source_experiment_id", "source_table", "source_row_index", "source_metric_name"], kind="mergesort").reset_index(drop=True)
    return df


def status_allows_missing(status: str, allowed: set[str]) -> bool:
    return status in allowed or status.startswith("unkeyed_") or status.startswith("missing_")


def _add_check(checks: list[dict[str, Any]], name: str, success: bool, detail: str) -> None:
    checks.append({"validation_case": name, "success": bool(success), "detail": detail})


def validate_unified_corpus(
    corpus: pd.DataFrame,
    missingness: pd.DataFrame,
    source_manifest: pd.DataFrame,
    *,
    required_sources: Iterable[str] = ("E01", "E02", "E03", "E04", "E05", "E06"),
    s01_world_ids: Iterable[str] = (),
    known_policy_ids: Iterable[str] = (),
    known_goal_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Validate S04 corpus structure, foreign keys, and documented missingness."""

    checks: list[dict[str, Any]] = []

    missing_columns = [column for column in CORPUS_COLUMNS if column not in corpus.columns]
    _add_check(
        checks,
        "required_columns_present",
        not missing_columns,
        "all corpus columns present" if not missing_columns else f"missing corpus columns: {missing_columns}",
    )

    duplicate_rows = int(corpus["corpus_row_id"].duplicated().sum()) if "corpus_row_id" in corpus else len(corpus)
    _add_check(checks, "corpus_row_ids_unique", duplicate_rows == 0, f"duplicate corpus_row_id rows: {duplicate_rows}")

    observed_sources = set(corpus.get("source_experiment_id", pd.Series(dtype=str)).astype(str))
    missing_sources = sorted(set(required_sources) - observed_sources)
    _add_check(
        checks,
        "required_sources_covered",
        not missing_sources,
        "all required sources covered" if not missing_sources else f"missing sources: {missing_sources}",
    )

    world_id_set = set(str(item) for item in s01_world_ids)
    bad_world_rows = corpus[(corpus["world_id"].astype(str) != "") & ~corpus["world_id"].astype(str).isin(world_id_set)]
    _add_check(
        checks,
        "s01_world_foreign_keys_resolve",
        bad_world_rows.empty,
        "all populated world_id values resolve to S01" if bad_world_rows.empty else f"bad world links: {len(bad_world_rows)}",
    )

    missing_world_rows = corpus[(corpus["world_id"].astype(str) == "") & ~corpus["world_link_status"].astype(str).map(lambda value: status_allows_missing(value, MISSING_WORLD_STATUSES))]
    _add_check(
        checks,
        "missing_world_links_are_documented",
        missing_world_rows.empty,
        "all blank world_id rows use documented missing statuses"
        if missing_world_rows.empty
        else f"undocumented blank world_id rows: {len(missing_world_rows)}",
    )

    policy_id_set = set(str(item) for item in known_policy_ids)
    populated_policies = corpus[corpus["policy_uid"].astype(str) != ""]
    bad_policy_rows = populated_policies[~populated_policies["policy_uid"].astype(str).isin(policy_id_set)]
    _add_check(
        checks,
        "s02_policy_foreign_keys_resolve",
        bad_policy_rows.empty,
        "all populated policy_uid values resolve to S02" if bad_policy_rows.empty else f"bad policy links: {len(bad_policy_rows)}",
    )

    missing_policy_rows = corpus[(corpus["policy_uid"].astype(str) == "") & ~corpus["policy_link_status"].astype(str).map(lambda value: status_allows_missing(value, MISSING_POLICY_STATUSES))]
    _add_check(
        checks,
        "missing_policy_links_are_documented",
        missing_policy_rows.empty,
        "all blank policy_uid rows use documented missing statuses"
        if missing_policy_rows.empty
        else f"undocumented blank policy_uid rows: {len(missing_policy_rows)}",
    )

    goal_id_set = set(str(item) for item in known_goal_ids)
    populated_goals = corpus[corpus["goal_uid"].astype(str) != ""]
    bad_goal_rows = populated_goals[~populated_goals["goal_uid"].astype(str).isin(goal_id_set)]
    _add_check(
        checks,
        "s03_goal_foreign_keys_resolve",
        bad_goal_rows.empty,
        "all populated goal_uid values resolve to S03" if bad_goal_rows.empty else f"bad goal links: {len(bad_goal_rows)}",
    )

    missing_goal_rows = corpus[(corpus["goal_uid"].astype(str) == "") & ~corpus["goal_link_status"].astype(str).map(lambda value: status_allows_missing(value, MISSING_GOAL_STATUSES))]
    _add_check(
        checks,
        "missing_goal_links_are_documented",
        missing_goal_rows.empty,
        "all blank goal_uid rows use documented missing statuses"
        if missing_goal_rows.empty
        else f"undocumented blank goal_uid rows: {len(missing_goal_rows)}",
    )

    manifest_missing = [column for column in SOURCE_MANIFEST_COLUMNS if column not in source_manifest.columns]
    _add_check(
        checks,
        "source_manifest_schema_present",
        not manifest_missing,
        "all source manifest columns present" if not manifest_missing else f"missing manifest columns: {manifest_missing}",
    )

    hash_failures: list[str] = []
    for row in source_manifest.to_dict(orient="records"):
        path = Path(str(row.get("source_artifact_path", "")))
        sha = str(row.get("source_artifact_sha256", ""))
        if str(row.get("ingest_status")) == "missing":
            continue
        if not path.exists() or not sha:
            hash_failures.append(str(path))
    _add_check(
        checks,
        "source_hashes_recorded",
        not hash_failures,
        "source hashes recorded for all available ingested artifacts" if not hash_failures else f"hash failures: {len(hash_failures)}",
    )

    missingness_missing = [column for column in MISSINGNESS_COLUMNS if column not in missingness.columns]
    _add_check(
        checks,
        "missingness_schema_present",
        not missingness_missing,
        "all missingness columns present" if not missingness_missing else f"missing missingness columns: {missingness_missing}",
    )

    documented_types = set(missingness.get("missingness_type", pd.Series(dtype=str)).astype(str))
    needs_missingness = any(
        [
            (corpus["world_id"].astype(str) == "").any(),
            (corpus["policy_uid"].astype(str) == "").any(),
            (corpus["goal_uid"].astype(str) == "").any(),
        ]
    )
    _add_check(
        checks,
        "missingness_records_cover_unkeyed_data",
        (not needs_missingness) or bool(documented_types),
        "missingness records present for unkeyed data" if needs_missingness else "no unkeyed data requires missingness records",
    )

    metric_rows = corpus[corpus["evidence_kind"].astype(str) == "metric_observation"]
    numeric_metric_failures = metric_rows[pd.to_numeric(metric_rows["metric_value"], errors="coerce").isna()]
    _add_check(
        checks,
        "metric_values_numeric",
        numeric_metric_failures.empty,
        "all metric observation rows have numeric metric_value"
        if numeric_metric_failures.empty
        else f"non-numeric metric values: {len(numeric_metric_failures)}",
    )

    json_failures: list[str] = []
    for column in JSON_COLUMNS:
        for index, value in corpus[column].items():
            if parse_json(value) is None:
                json_failures.append(f"{column}:{index}")
                break
    _add_check(
        checks,
        "json_columns_roundtrip",
        not json_failures,
        "all JSON columns parse" if not json_failures else f"JSON parse failures: {json_failures[:5]}",
    )

    return pd.DataFrame(checks)
