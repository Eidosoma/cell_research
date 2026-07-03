"""Helpers for E07 S11 counterfactual prediction tests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


COUNTERFACTUAL_SCHEMA_VERSION = "eidosoma.e07.counterfactual_predictions.v1"

S05_CATEGORICAL_COLUMNS = (
    "source_experiment_id",
    "source_table",
    "source_metric_name",
    "metric_unit",
    "metric_direction",
    "metric_family",
    "world_id",
    "world_family",
    "substrate_kind",
    "world_record_granularity",
    "world_completeness",
    "canonical_policy_id",
    "policy_source_experiment_id",
    "policy_family",
    "policy_algorithm",
    "policy_representation_type",
    "policy_abstraction_kind",
    "policy_information_access",
    "policy_execution_backend",
    "policy_direction_support",
    "canonical_goal_id",
    "goal_source_experiment_id",
    "goal_family",
    "goal_kind",
    "goal_abstraction_kind",
    "goal_target_direction",
    "goal_unit",
    "goal_metric_family",
    "goal_representation_status",
    "goal_conflict_group_id",
    "perturbation_type",
    "world_link_status",
    "policy_link_status",
    "goal_link_status",
)

S05_NUMERIC_COLUMNS = (
    "has_world_link",
    "has_policy_link",
    "has_goal_link",
    "policy_metadata_only",
    "policy_requires_memory",
    "policy_requires_signaling",
    "policy_uses_global_oracle",
    "goal_partial",
)

CAPABILITY_TARGETS = (
    "high_dg",
    "high_repair_robustness",
    "high_aggregation",
    "high_order_quality",
)

POSITIVE_CLASS_STATUS = "bounded_interpretable_class"


def clean_text(value: Any) -> str:
    """Return a stable missing-safe text value."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def class_status_bucket(status: Any) -> str:
    """Map an S10 class status into the S11 selection bucket."""

    text = clean_text(status)
    if text == POSITIVE_CLASS_STATUS:
        return "positive_bounded"
    if "constraint" in text or "source_dominated" in text or "missingness" in text:
        return "caution_constraining"
    if not text or text == "unclassified":
        return "unclassified"
    return "other"


def has_constraining_status(statuses: Iterable[Any]) -> bool:
    """Return true if any class status must be treated as a caution/control label."""

    return any(class_status_bucket(status) == "caution_constraining" for status in statuses)


def capability_target_for_record(record: Mapping[str, Any]) -> str:
    """Classify the requested competence family from metric and goal labels."""

    text = " ".join(
        clean_text(record.get(column))
        for column in ("metric_family", "source_metric_name", "goal_family", "goal_metric_family", "display_name")
    ).lower()
    if "aggregation" in text:
        return "high_aggregation"
    if "delayed" in text or "dg" in text:
        return "high_dg"
    if any(token in text for token in ("repair", "robust", "damage", "frozen", "recovery")):
        return "high_repair_robustness"
    if any(token in text for token in ("sorting", "sorted", "order", "inversion", "monotonic")):
        return "high_order_quality"
    return "other"


def feature_dicts(frame: pd.DataFrame) -> list[dict[str, float]]:
    """Build the hashed-feature records used by the frozen S05 linear surrogate."""

    records: list[dict[str, float]] = []
    missing_safe = frame.copy()
    for column in S05_CATEGORICAL_COLUMNS:
        if column not in missing_safe.columns:
            missing_safe[column] = "__missing__"
        missing_safe[column] = missing_safe[column].fillna("__missing__").astype(str).map(lambda value: value or "__missing__")
    for column in S05_NUMERIC_COLUMNS:
        if column not in missing_safe.columns:
            missing_safe[column] = 0.0
        missing_safe[column] = pd.to_numeric(missing_safe[column], errors="coerce").fillna(0.0)
    for row in missing_safe[list(S05_CATEGORICAL_COLUMNS) + list(S05_NUMERIC_COLUMNS)].to_dict(orient="records"):
        features: dict[str, float] = {}
        for column in S05_CATEGORICAL_COLUMNS:
            features[f"{column}={row[column]}"] = 1.0
        for column in S05_NUMERIC_COLUMNS:
            features[f"num:{column}"] = float(row[column])
        records.append(features)
    return records


def grouped_median_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_columns: Iterable[str],
    *,
    target_column: str = "target_transformed",
) -> np.ndarray:
    """Predict test rows with train medians over a missing-safe group key."""

    columns = tuple(group_columns)
    if train.empty:
        return np.full(len(test), np.nan, dtype=float)
    global_median = float(pd.to_numeric(train[target_column], errors="coerce").median())
    train_keys = train.copy()
    test_keys = test.copy()
    for column in columns:
        if column not in train_keys.columns:
            train_keys[column] = "__missing__"
        if column not in test_keys.columns:
            test_keys[column] = "__missing__"
        train_keys[column] = train_keys[column].fillna("__missing__").astype(str).map(lambda value: value or "__missing__")
        test_keys[column] = test_keys[column].fillna("__missing__").astype(str).map(lambda value: value or "__missing__")
    medians = train_keys.groupby(list(columns), dropna=False)[target_column].median().to_dict()
    predictions: list[float] = []
    for row in test_keys[list(columns)].to_dict(orient="records"):
        key = tuple(row[column] for column in columns)
        predictions.append(float(medians.get(key, global_median)))
    return np.asarray(predictions, dtype=float)


def stable_record_hash(record: Mapping[str, Any], *, prefix: str = "s11pred") -> str:
    """Return a stable content hash for a candidate prediction record."""

    payload = json.dumps(record, sort_keys=True, default=str, separators=(",", ":"), allow_nan=False)
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def timestamp_order_ok(first: str, second: str) -> bool:
    """Return true if two ISO timestamps are parseable and first <= second."""

    try:
        return datetime.fromisoformat(first) <= datetime.fromisoformat(second)
    except ValueError:
        return False


def validation_summary(checks: pd.DataFrame) -> dict[str, Any]:
    """Return a compact pass/fail summary for validation checks."""

    if checks.empty or "success" not in checks:
        return {"passed": 0, "total": 0, "allPassed": False}
    return {
        "passed": int(checks["success"].sum()),
        "total": int(len(checks)),
        "allPassed": bool(checks["success"].all()),
    }


def validate_counterfactual_artifacts(
    predictions: pd.DataFrame,
    validations: pd.DataFrame,
    freeze_manifest: Mapping[str, Any],
    *,
    expected_capabilities: Iterable[str] = CAPABILITY_TARGETS,
) -> pd.DataFrame:
    """Validate S11 prediction-freeze, controls, and validation artifacts."""

    checks: list[dict[str, Any]] = []
    expected = set(expected_capabilities)
    observed = set(predictions.get("capability_target", pd.Series(dtype=str)).astype(str))

    def add(name: str, success: bool, detail: str) -> None:
        checks.append({"validation_case": name, "success": bool(success), "detail": detail})

    add(
        "predictions_frozen_and_hashed",
        bool(freeze_manifest.get("predictionArtifactSha256")) and bool(freeze_manifest.get("frozenAtUtc")),
        f"hash={freeze_manifest.get('predictionArtifactSha256')} frozenAt={freeze_manifest.get('frozenAtUtc')}",
    )
    add(
        "freeze_precedes_validation",
        timestamp_order_ok(str(freeze_manifest.get("frozenAtUtc")), str(freeze_manifest.get("validationStartedAtUtc"))),
        f"frozenAt={freeze_manifest.get('frozenAtUtc')} validationStartedAt={freeze_manifest.get('validationStartedAtUtc')}",
    )
    positive = predictions[predictions.get("candidate_role", pd.Series(dtype=str)).astype(str) == "positive_candidate"]
    positive_ok = not positive.empty and positive["policy_class_status"].astype(str).eq(POSITIVE_CLASS_STATUS).all()
    if not positive.empty:
        positive_ok = positive_ok and not positive[["policy_class_status", "world_class_status", "goal_class_status"]].apply(
            lambda row: has_constraining_status(row), axis=1
        ).any()
    add("positive_candidates_use_only_bounded_supported_classes", bool(positive_ok), f"positive_rows={len(positive)}")
    caution = predictions[predictions.get("candidate_role", pd.Series(dtype=str)).astype(str).str.contains("caution", regex=False)]
    caution_ok = not caution.empty and caution[["policy_class_status", "world_class_status", "goal_class_status"]].apply(
        lambda row: has_constraining_status(row), axis=1
    ).any()
    add("caution_controls_include_constraining_classes", bool(caution_ok), f"caution_rows={len(caution)}")
    baseline_columns = {
        "source_metric_median_prediction",
        "metric_family_median_prediction",
        "missingness_pattern_median_prediction",
        "class_status_median_prediction",
    }
    add(
        "required_baselines_present",
        baseline_columns <= set(predictions.columns),
        f"missing={sorted(baseline_columns.difference(predictions.columns))}",
    )
    add(
        "validation_rows_match_predictions",
        set(predictions.get("prediction_id", pd.Series(dtype=str))) <= set(validations.get("prediction_id", pd.Series(dtype=str))),
        f"prediction_rows={len(predictions)} validation_prediction_ids={validations.get('prediction_id', pd.Series(dtype=str)).nunique()}",
    )
    add(
        "expected_capability_targets_represented_or_gap_recorded",
        expected <= observed or bool(freeze_manifest.get("selectionGaps")),
        f"observed={sorted(observed)} gaps={freeze_manifest.get('selectionGaps')}",
    )
    add(
        "simulator_blockers_recorded",
        "fresh_simulator_blocker" in set(validations.get("validation_kind", pd.Series(dtype=str)).astype(str)),
        f"validation_kinds={sorted(set(validations.get('validation_kind', pd.Series(dtype=str)).astype(str)))}",
    )
    add(
        "retrospective_observed_targets_present",
        bool((validations.get("validation_kind", pd.Series(dtype=str)).astype(str) == "heldout_retrospective").any())
        and validations.loc[
            validations.get("validation_kind", pd.Series(dtype=str)).astype(str) == "heldout_retrospective",
            "observed_target_transformed",
        ].notna().all(),
        "heldout retrospective rows require observed targets",
    )
    return pd.DataFrame(checks)
