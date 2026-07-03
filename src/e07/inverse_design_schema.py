"""Helpers for E07 S12 executable-proxy inverse design."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


INVERSE_DESIGN_SCHEMA_VERSION = "eidosoma.e07.inverse_design.v1"

REQUIRED_CONTROL_FAMILIES = (
    "source_control",
    "metric_only_control",
    "missingness_control",
    "class_status_control",
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def stable_payload_hash(payload: Mapping[str, Any], *, prefix: str) -> str:
    """Return a short stable ID for JSON-serializable records."""

    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"), allow_nan=False)
    return f"{prefix}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def target_profile_id(profile: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in profile.items() if key not in {"target_profile_id", "frozen_at_utc"}}
    return stable_payload_hash(payload, prefix="s12target")


def design_record_id(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key not in {"design_id", "frozen_at_utc"}}
    return stable_payload_hash(payload, prefix="s12design")


def timestamp_order_ok(first: str, second: str) -> bool:
    try:
        return datetime.fromisoformat(first) <= datetime.fromisoformat(second)
    except ValueError:
        return False


def bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if not np.isfinite(value):
        return lower
    return float(min(upper, max(lower, value)))


def e03_proxy_target_score(
    *,
    final_sortedness: float,
    work_per_item: float,
    dg_recovery_proxy: float,
    invalid: bool,
    target_sortedness: float = 0.9,
    target_work_per_item: float = 3.5,
    target_dg_recovery_proxy: float = 0.03,
    dg_tolerance: float = 0.08,
) -> float:
    """Score the scoped E03 proxy target without using heldout labels."""

    order_component = bounded(float(final_sortedness) / target_sortedness)
    energy_component = bounded(1.0 - max(0.0, float(work_per_item) - target_work_per_item) / max(target_work_per_item, 1e-9))
    dg_component = bounded(1.0 - abs(float(dg_recovery_proxy) - target_dg_recovery_proxy) / max(dg_tolerance, 1e-9))
    penalty = 0.35 if invalid else 0.0
    return bounded(0.5 * order_component + 0.25 * energy_component + 0.25 * dg_component - penalty)


def validation_summary(checks: pd.DataFrame) -> dict[str, Any]:
    if checks.empty or "success" not in checks:
        return {"passed": 0, "total": 0, "allPassed": False}
    return {
        "passed": int(checks["success"].sum()),
        "total": int(len(checks)),
        "allPassed": bool(checks["success"].all()),
    }


def validate_inverse_design_artifacts(
    targets: pd.DataFrame,
    designs: pd.DataFrame,
    validations: pd.DataFrame,
    target_manifest: Mapping[str, Any],
    design_manifest: Mapping[str, Any],
    *,
    required_controls: Iterable[str] = REQUIRED_CONTROL_FAMILIES,
) -> pd.DataFrame:
    """Validate S12 freeze order, scope, controls, and heldout simulation outputs."""

    checks: list[dict[str, Any]] = []

    def add(name: str, success: bool, detail: str) -> None:
        checks.append({"validation_case": name, "success": bool(success), "detail": detail})

    add(
        "target_profiles_frozen_and_hashed",
        bool(target_manifest.get("targetProfileArtifactSha256")) and bool(target_manifest.get("targetProfilesFrozenAtUtc")),
        f"hash={target_manifest.get('targetProfileArtifactSha256')} frozenAt={target_manifest.get('targetProfilesFrozenAtUtc')}",
    )
    add(
        "candidate_designs_frozen_and_hashed",
        bool(design_manifest.get("candidateDesignArtifactSha256")) and bool(design_manifest.get("candidateDesignsFrozenAtUtc")),
        f"hash={design_manifest.get('candidateDesignArtifactSha256')} frozenAt={design_manifest.get('candidateDesignsFrozenAtUtc')}",
    )
    add(
        "target_freeze_precedes_candidate_freeze",
        timestamp_order_ok(
            str(target_manifest.get("targetProfilesFrozenAtUtc", "")),
            str(design_manifest.get("candidateDesignsFrozenAtUtc", "")),
        ),
        f"targetFrozenAt={target_manifest.get('targetProfilesFrozenAtUtc')} candidateFrozenAt={design_manifest.get('candidateDesignsFrozenAtUtc')}",
    )
    add(
        "candidate_freeze_precedes_validation",
        timestamp_order_ok(
            str(design_manifest.get("candidateDesignsFrozenAtUtc", "")),
            str(design_manifest.get("validationStartedAtUtc", "")),
        ),
        f"candidateFrozenAt={design_manifest.get('candidateDesignsFrozenAtUtc')} validationStartedAt={design_manifest.get('validationStartedAtUtc')}",
    )

    executable = designs[designs.get("executable_in_s12", pd.Series(dtype=bool)).astype(bool)].copy()
    add("candidate_designs_nonempty", not executable.empty, f"executable_design_rows={len(executable)}")
    scope_ok = (
        not executable.empty
        and executable.get("design_scope", pd.Series(dtype=str)).astype(str).eq("e03_dsl_array_world").all()
        and executable.get("source_experiment_id", pd.Series(dtype=str)).astype(str).eq("E03").all()
        and executable.get("representation_type", pd.Series(dtype=str)).astype(str).eq("dsl").all()
    )
    add("executable_designs_restricted_to_e03_dsl", bool(scope_ok), f"executable_design_rows={len(executable)}")
    parse_ok = not executable.empty and executable.get("parser_validation_status", pd.Series(dtype=str)).astype(str).eq("parsed").all()
    add("executable_designs_parse", bool(parse_ok), f"parsed={int(executable.get('parser_validation_status', pd.Series(dtype=str)).astype(str).eq('parsed').sum())}")

    heldout = validations[validations.get("validation_kind", pd.Series(dtype=str)).astype(str) == "heldout_e03_simulation"].copy()
    heldout_ok = (
        not heldout.empty
        and heldout.get("validation_split", pd.Series(dtype=str)).astype(str).eq("heldout").all()
        and heldout.get("source_experiment_id", pd.Series(dtype=str)).astype(str).eq("E03").all()
    )
    add("heldout_e03_validation_rows_present", bool(heldout_ok), f"heldout_rows={len(heldout)}")
    design_ids = set(executable.get("design_id", pd.Series(dtype=str)).astype(str))
    heldout_design_ids = set(heldout.get("design_id", pd.Series(dtype=str)).astype(str))
    add(
        "heldout_rows_cover_all_executable_designs",
        bool(design_ids) and design_ids <= heldout_design_ids,
        f"executable_designs={len(design_ids)} heldout_designs={len(heldout_design_ids)}",
    )
    add(
        "heldout_worlds_are_multiple",
        heldout.get("world_id", pd.Series(dtype=str)).astype(str).nunique() >= 3,
        f"heldout_worlds={heldout.get('world_id', pd.Series(dtype=str)).astype(str).nunique()}",
    )

    observed_controls = set(validations.get("control_family", pd.Series(dtype=str)).astype(str))
    missing_controls = sorted(set(required_controls) - observed_controls)
    add(
        "source_metric_missingness_class_controls_present",
        not missing_controls,
        f"missing_controls={missing_controls}",
    )
    blocker_rows = validations[validations.get("validation_kind", pd.Series(dtype=str)).astype(str) == "simulator_blocker"]
    add("simulator_blockers_recorded", not blocker_rows.empty, f"blocker_rows={len(blocker_rows)}")
    non_e03_simulated = heldout[heldout.get("source_experiment_id", pd.Series(dtype=str)).astype(str) != "E03"]
    add("no_non_e03_adapter_validation", non_e03_simulated.empty, f"non_e03_simulated_rows={len(non_e03_simulated)}")

    target_axes = set(targets.get("target_axis_id", pd.Series(dtype=str)).astype(str))
    add(
        "target_profile_records_scope_limitations",
        {"order_quality_proxy", "energy_efficiency_proxy", "moderate_dg_proxy", "aggregation_blocker", "repair_robustness_blocker"} <= target_axes,
        f"target_axes={sorted(target_axes)}",
    )
    return pd.DataFrame(checks)
