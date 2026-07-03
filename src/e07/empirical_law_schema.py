"""Helpers for E07 S14 empirical-law synthesis."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


EMPIRICAL_LAW_SCHEMA_VERSION = "eidosoma.e07.empirical_laws.v1"

ALLOWED_EVIDENCE_STEPS = {"S09", "S10", "S11", "S12", "S13"}
ALLOWED_CLAIM_STATUSES = {
    "supported_narrow",
    "provisional_constraining",
    "provisional_descriptor",
    "unsupported_speculation",
}
LAW_REQUIRED_COLUMNS = {
    "schema_version",
    "research_step_id",
    "law_id",
    "law_slug",
    "law_title",
    "claim_status",
    "outcome_classification",
    "unsupported_speculation",
    "s13_scope_limited",
    "evidence_steps_json",
    "evidence_summary",
    "quantitative_support_json",
    "scope",
    "counterexamples",
    "falsification_tests",
    "caveats",
    "recommended_use",
    "source_artifacts_json",
}


def sanitize_json(value: Any) -> Any:
    """Convert pandas/numpy/NaN values into strict JSON-compatible values."""

    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
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


def parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    try:
        decoded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return decoded if isinstance(decoded, list) else []


def empirical_law_id(slug: str, title: str) -> str:
    return f"s14law:{stable_hash({'slug': slug, 'title': title})[:16]}"


def validation_summary(checks: pd.DataFrame) -> dict[str, Any]:
    if checks.empty or "success" not in checks:
        return {"passed": 0, "total": 0, "allPassed": False}
    return {
        "passed": int(checks["success"].sum()),
        "total": int(len(checks)),
        "allPassed": bool(checks["success"].all()),
    }


def validate_empirical_law_artifacts(
    laws: pd.DataFrame,
    source_manifest: pd.DataFrame,
    *,
    expected_report_law_count: int | None = None,
) -> pd.DataFrame:
    """Validate that S14 law candidates are bounded, sourced, and caveated."""

    checks: list[dict[str, Any]] = []

    def add(name: str, success: bool, detail: str) -> None:
        checks.append({"validation_case": name, "success": bool(success), "detail": detail})

    missing = sorted(LAW_REQUIRED_COLUMNS - set(laws.columns))
    add("law_table_required_columns_present", not missing and not laws.empty, f"rows={len(laws)} missing={missing}")

    claim_statuses = set(laws.get("claim_status", pd.Series(dtype=str)).astype(str))
    add(
        "claim_statuses_are_allowed_and_bounded",
        bool(claim_statuses) and claim_statuses <= ALLOWED_CLAIM_STATUSES,
        f"claim_statuses={sorted(claim_statuses)}",
    )

    bad_steps: list[str] = []
    for row in laws.to_dict(orient="records"):
        steps = {str(item) for item in parse_json_list(row.get("evidence_steps_json"))}
        if not steps or not steps <= ALLOWED_EVIDENCE_STEPS:
            bad_steps.append(str(row.get("law_id", "")))
    add("evidence_steps_within_s09_s13", not bad_steps, f"bad_law_ids={bad_steps}")

    required_text_cols = ["evidence_summary", "scope", "counterexamples", "falsification_tests", "caveats", "recommended_use"]
    empty_required = []
    for column in required_text_cols:
        empty_count = int(laws.get(column, pd.Series(dtype=str)).fillna("").astype(str).str.strip().eq("").sum())
        if empty_count:
            empty_required.append(f"{column}:{empty_count}")
    add("every_law_has_evidence_scope_counterexamples_falsification_and_caveats", not empty_required, f"empty={empty_required}")

    unsupported = laws[laws.get("unsupported_speculation", pd.Series(dtype=bool)).astype(bool)]
    unsupported_ok = (
        not unsupported.empty
        and unsupported.get("claim_status", pd.Series(dtype=str)).astype(str).eq("unsupported_speculation").all()
        and unsupported.get("law_title", pd.Series(dtype=str)).astype(str).str.lower().str.contains("unsupported").all()
    )
    add("unsupported_speculation_explicitly_labeled", bool(unsupported_ok), f"unsupported_rows={len(unsupported)}")

    s13_rows = laws[
        laws.get("evidence_steps_json", pd.Series(dtype=str)).astype(str).str.contains("S13", regex=False)
        | laws.get("law_title", pd.Series(dtype=str)).astype(str).str.lower().str.contains("transfer", regex=False)
    ]
    s13_ok = (
        s13_rows.empty
        or (
            s13_rows.get("s13_scope_limited", pd.Series(dtype=bool)).astype(bool).all()
            and s13_rows.get("scope", pd.Series(dtype=str)).astype(str).str.lower().str.contains("embedded-row|embedded row|block", regex=True).all()
        )
    )
    add("s13_scope_not_overgeneralized", bool(s13_ok), f"s13_related_rows={len(s13_rows)}")

    source_paths_ok = not source_manifest.empty and {"source_step_id", "path", "sha256", "row_count"} <= set(source_manifest.columns)
    nonmissing_hashes = source_manifest.get("sha256", pd.Series(dtype=str)).astype(str).ne("").all() if source_paths_ok else False
    add("source_manifest_hashes_present", bool(source_paths_ok and nonmissing_hashes), f"source_rows={len(source_manifest)}")

    quantitative_parse_failures: list[str] = []
    source_artifact_parse_failures: list[str] = []
    for row in laws.to_dict(orient="records"):
        for column, failures in (
            ("quantitative_support_json", quantitative_parse_failures),
            ("source_artifacts_json", source_artifact_parse_failures),
        ):
            try:
                parsed = json.loads(str(row.get(column, "")))
            except (json.JSONDecodeError, TypeError, ValueError):
                failures.append(str(row.get("law_id", "")))
                continue
            if not isinstance(parsed, (dict, list)):
                failures.append(str(row.get("law_id", "")))
    add("quantitative_support_json_parses", not quantitative_parse_failures, f"bad_law_ids={quantitative_parse_failures}")
    add("source_artifacts_json_parses", not source_artifact_parse_failures, f"bad_law_ids={source_artifact_parse_failures}")

    unique_ids = laws.get("law_id", pd.Series(dtype=str)).astype(str)
    add("law_ids_unique", unique_ids.nunique(dropna=False) == len(laws), f"rows={len(laws)} unique={unique_ids.nunique(dropna=False)}")

    if expected_report_law_count is not None:
        add(
            "report_law_count_matches_matrix",
            int(expected_report_law_count) == len(laws),
            f"report_count={expected_report_law_count} matrix_rows={len(laws)}",
        )

    return pd.DataFrame(checks)


def dataframe_from_law_records(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row.setdefault("schema_version", EMPIRICAL_LAW_SCHEMA_VERSION)
        row.setdefault("research_step_id", "S14")
        row.setdefault("law_id", empirical_law_id(str(row.get("law_slug", "")), str(row.get("law_title", ""))))
        for column in ("evidence_steps_json", "quantitative_support_json", "source_artifacts_json"):
            value = row.get(column)
            if not isinstance(value, str):
                row[column] = stable_json(value if value is not None else ([] if column.endswith("_json") else {}))
        rows.append(row)
    frame = pd.DataFrame(rows)
    ordered = [column for column in LAW_REQUIRED_COLUMNS if column in frame.columns]
    remaining = [column for column in frame.columns if column not in ordered]
    return frame[ordered + remaining].sort_values("law_id", kind="mergesort").reset_index(drop=True)
