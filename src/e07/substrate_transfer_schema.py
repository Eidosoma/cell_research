"""Helpers for E07 S13 substrate-transfer tests."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


SUBSTRATE_TRANSFER_SCHEMA_VERSION = "eidosoma.e07.substrate_transfer.v1"

REQUIRED_CONTROL_FAMILIES = (
    "within_1d_baseline",
    "cross_substrate_transfer",
    "e05_context_control",
    "blocked_transfer",
)


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


def stable_payload_hash(payload: Mapping[str, Any], *, prefix: str) -> str:
    text = json.dumps(sanitize_json(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{prefix}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def transfer_mapping_id(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key not in {"mapping_id", "frozen_at_utc"}}
    return stable_payload_hash(payload, prefix="s13map")


def timestamp_order_ok(first: str, second: str) -> bool:
    try:
        return datetime.fromisoformat(first) <= datetime.fromisoformat(second)
    except ValueError:
        return False


def validation_summary(checks: pd.DataFrame) -> dict[str, Any]:
    if checks.empty or "success" not in checks:
        return {"passed": 0, "total": 0, "allPassed": False}
    return {
        "passed": int(checks["success"].sum()),
        "total": int(len(checks)),
        "allPassed": bool(checks["success"].all()),
    }


def transfer_score(
    *,
    target_final_sortedness_percent: float | None = None,
    exact_trajectory_match: bool = False,
    target_recovery_fraction: float | None = None,
    blocked: bool = False,
) -> float:
    """Return a compact [0, 1] score for heterogeneous transfer evidence."""

    if blocked:
        return 0.0
    if exact_trajectory_match:
        return 1.0
    if target_final_sortedness_percent is not None and np.isfinite(float(target_final_sortedness_percent)):
        return float(np.clip(float(target_final_sortedness_percent) / 100.0, 0.0, 1.0))
    if target_recovery_fraction is not None and np.isfinite(float(target_recovery_fraction)):
        return float(np.clip(float(target_recovery_fraction), 0.0, 1.0))
    return 0.0


def validate_substrate_transfer_artifacts(
    mappings: pd.DataFrame,
    results: pd.DataFrame,
    freeze_manifest: Mapping[str, Any],
    *,
    required_controls: Iterable[str] = REQUIRED_CONTROL_FAMILIES,
) -> pd.DataFrame:
    """Validate S13 mapping freeze, scope, baselines, blockers, and results."""

    checks: list[dict[str, Any]] = []

    def add(name: str, success: bool, detail: str) -> None:
        checks.append({"validation_case": name, "success": bool(success), "detail": detail})

    add(
        "transfer_mappings_frozen_and_hashed",
        bool(freeze_manifest.get("transferMappingArtifactSha256")) and bool(freeze_manifest.get("mappingsFrozenAtUtc")),
        f"hash={freeze_manifest.get('transferMappingArtifactSha256')} frozenAt={freeze_manifest.get('mappingsFrozenAtUtc')}",
    )
    add(
        "mapping_freeze_precedes_evaluation",
        timestamp_order_ok(str(freeze_manifest.get("mappingsFrozenAtUtc", "")), str(freeze_manifest.get("evaluationStartedAtUtc", ""))),
        f"mappingsFrozenAt={freeze_manifest.get('mappingsFrozenAtUtc')} evaluationStartedAt={freeze_manifest.get('evaluationStartedAtUtc')}",
    )
    add(
        "no_new_e05_or_e06_adapters_declared",
        bool(freeze_manifest.get("noNewE05OrE06AdaptersBuilt")) and not mappings.get("adapter_built_in_s13", pd.Series(dtype=bool)).fillna(False).astype(bool).any(),
        f"noNewAdapters={freeze_manifest.get('noNewE05OrE06AdaptersBuilt')}",
    )

    executable = mappings[mappings.get("mapping_status", pd.Series(dtype=str)).astype(str) == "executable_existing_path"].copy()
    add("executable_existing_path_mappings_present", not executable.empty, f"executable_mappings={len(executable)}")
    target_experiments = (
        executable["target_experiment_id"].astype(str)
        if "target_experiment_id" in executable
        else pd.Series("", index=executable.index, dtype=str)
    )
    control_families = (
        executable["control_family"].astype(str)
        if "control_family" in executable
        else pd.Series("", index=executable.index, dtype=str)
    )
    cross_target_ok = target_experiments[control_families.eq("cross_substrate_transfer")].eq("E05").all()
    existing_paths_ok = (
        not executable.empty
        and executable.get("execution_path", pd.Series(dtype=str)).astype(str).eq("src.e05.embedded_1d.run_adjacent_sort").all()
        and target_experiments.isin({"E03", "E05"}).all()
        and bool(cross_target_ok)
    )
    add("executable_mappings_use_only_existing_e03_e05_paths", bool(existing_paths_ok), f"executable_mappings={len(executable)}")

    observed_controls = set(results.get("control_family", pd.Series(dtype=str)).astype(str))
    missing_controls = sorted(set(required_controls) - observed_controls)
    add("required_baselines_and_blockers_present", not missing_controls, f"missing_controls={missing_controls}")

    result_mapping_ids = set(results.get("mapping_id", pd.Series(dtype=str)).astype(str))
    executable_ids = set(executable.get("mapping_id", pd.Series(dtype=str)).astype(str))
    add(
        "results_cover_executable_mappings",
        bool(executable_ids) and executable_ids <= result_mapping_ids,
        f"executable_mappings={len(executable_ids)} result_mapping_ids={len(result_mapping_ids)}",
    )
    blocker_rows = results[results.get("evaluation_kind", pd.Series(dtype=str)).astype(str) == "transfer_blocker"]
    add("transfer_blocker_records_present", not blocker_rows.empty, f"blocker_rows={len(blocker_rows)}")
    s12_blockers = blocker_rows[blocker_rows.get("mapping_kind", pd.Series(dtype=str)).astype(str).str.contains("s12", regex=False)]
    add("s12_e03_only_caveat_blockers_present", not s12_blockers.empty, f"s12_blocker_rows={len(s12_blockers)}")

    fresh = results[results.get("evaluation_kind", pd.Series(dtype=str)).astype(str) == "fresh_existing_path_evaluation"]
    transfer = fresh[fresh.get("control_family", pd.Series(dtype=str)).astype(str) == "cross_substrate_transfer"]
    exact_values = transfer.get("exact_trajectory_match", pd.Series(dtype=bool))
    exact_ok = not transfer.empty and exact_values.astype("boolean").fillna(False).astype(bool).all()
    add("existing_embedded_row_transfer_exact", bool(exact_ok), f"fresh_transfer_rows={len(transfer)}")

    score = pd.to_numeric(results.get("transfer_score", pd.Series(dtype=float)), errors="coerce")
    add(
        "transfer_scores_finite_or_blocked",
        bool(np.isfinite(score.fillna(0.0).to_numpy(dtype=float)).all()),
        f"result_rows={len(results)} score_nonmissing={int(score.notna().sum())}",
    )
    non_e05_e06 = fresh[
        ~fresh.get("target_experiment_id", pd.Series(dtype=str)).astype(str).isin({"E03", "E05"})
        | fresh.get("target_experiment_id", pd.Series(dtype=str)).astype(str).eq("E06")
    ]
    add("no_non_e03_e05_fresh_evaluation", non_e05_e06.empty, f"bad_fresh_rows={len(non_e05_e06)}")
    return pd.DataFrame(checks)
