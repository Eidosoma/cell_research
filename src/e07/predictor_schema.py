"""Helpers for E07 S05 behavior-predictor training and validation."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


PREDICTOR_SCHEMA_VERSION = "eidosoma.e07.behavior_predictor.v1"

SPLIT_DEFINITIONS = {
    "heldout_policy": "canonical_policy_id",
    "heldout_world": "world_id",
    "heldout_goal": "canonical_goal_id",
    "heldout_perturbation": "perturbation_type",
}


def stable_fraction(label: str, *, salt: str = "e07-s05") -> float:
    digest = hashlib.sha256(f"{salt}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def signed_log1p(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(numeric):
        return float("nan")
    return math.copysign(math.log1p(abs(numeric)), numeric)


def oriented_metric_value(metric_value: Any, metric_direction: str) -> float:
    numeric = float(metric_value)
    direction = str(metric_direction)
    if direction == "minimize":
        return -numeric
    return numeric


def transformed_target(metric_value: Any, metric_direction: str) -> float:
    return signed_log1p(oriented_metric_value(metric_value, metric_direction))


def assign_group_holdout(
    frame: pd.DataFrame,
    *,
    group_column: str,
    split_name: str,
    test_fraction: float = 0.2,
) -> pd.Series:
    groups = sorted({str(value) for value in frame[group_column].fillna("").astype(str) if str(value)})
    if not groups:
        return pd.Series(False, index=frame.index)
    group_is_test = {group: stable_fraction(group, salt=split_name) < test_fraction for group in groups}
    if not any(group_is_test.values()):
        first_group = min(groups, key=lambda group: stable_fraction(group, salt=split_name))
        group_is_test[first_group] = True
    if all(group_is_test.values()) and len(groups) > 1:
        last_group = max(groups, key=lambda group: stable_fraction(group, salt=split_name))
        group_is_test[last_group] = False
    return frame[group_column].fillna("").astype(str).map(lambda value: bool(value) and group_is_test.get(value, False))


def split_summary(frame: pd.DataFrame, *, split_name: str, group_column: str, test_mask: pd.Series) -> dict[str, Any]:
    group_values = frame[group_column].fillna("").astype(str)
    eligible = group_values != ""
    train_groups = set(group_values[eligible & ~test_mask])
    test_groups = set(group_values[eligible & test_mask])
    return {
        "split_name": split_name,
        "group_column": group_column,
        "eligible_rows": int(eligible.sum()),
        "train_rows": int((eligible & ~test_mask).sum()),
        "test_rows": int((eligible & test_mask).sum()),
        "eligible_groups": int(len(train_groups | test_groups)),
        "train_groups": int(len(train_groups)),
        "test_groups": int(len(test_groups)),
        "leaked_groups": int(len(train_groups & test_groups)),
        "test_fraction_rows": float((eligible & test_mask).sum() / max(1, eligible.sum())),
    }


def metrics_for_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0 else float("nan")
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        slope, intercept = np.polyfit(y_pred, y_true, 1)
        corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    else:
        slope, intercept, corr = float("nan"), float("nan"), float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "pearson_r": corr,
        "predicted_mean": float(np.mean(y_pred)),
        "observed_mean": float(np.mean(y_true)),
    }


def calibration_bins(y_true: np.ndarray, y_pred: np.ndarray, *, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"observed": np.asarray(y_true, dtype=float), "predicted": np.asarray(y_pred, dtype=float)})
    if frame.empty:
        return pd.DataFrame(columns=["bin_index", "row_count", "predicted_mean", "observed_mean", "abs_gap"])
    rank = frame["predicted"].rank(method="first")
    try:
        frame["bin_index"] = pd.qcut(rank, q=min(n_bins, len(frame)), labels=False, duplicates="drop")
    except ValueError:
        frame["bin_index"] = 0
    grouped = (
        frame.groupby("bin_index", dropna=False)
        .agg(row_count=("observed", "size"), predicted_mean=("predicted", "mean"), observed_mean=("observed", "mean"))
        .reset_index()
    )
    grouped["abs_gap"] = (grouped["predicted_mean"] - grouped["observed_mean"]).abs()
    return grouped


def validate_prediction_artifacts(
    metrics: pd.DataFrame,
    split_summaries: pd.DataFrame,
    *,
    expected_splits: Iterable[str] = SPLIT_DEFINITIONS.keys(),
    expected_models: Iterable[str] = ("global_median", "source_metric_median", "hashed_linear_sgd", "neural_embedding_mlp"),
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    expected_split_set = set(expected_splits)
    expected_model_set = set(expected_models)
    observed_split_set = set(metrics.get("split_name", pd.Series(dtype=str)).astype(str))
    observed_model_set = set(metrics.get("model_name", pd.Series(dtype=str)).astype(str))
    checks.append(
        {
            "validation_case": "all_expected_splits_evaluated",
            "success": expected_split_set <= observed_split_set,
            "detail": f"observed splits: {sorted(observed_split_set)}",
        }
    )
    checks.append(
        {
            "validation_case": "all_expected_models_evaluated",
            "success": expected_model_set <= observed_model_set,
            "detail": f"observed models: {sorted(observed_model_set)}",
        }
    )
    leaked = int(split_summaries.get("leaked_groups", pd.Series(dtype=int)).sum())
    checks.append(
        {
            "validation_case": "no_group_leakage",
            "success": leaked == 0,
            "detail": f"leaked groups across split summaries: {leaked}",
        }
    )
    finite_metric_rows = metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=["mae", "rmse"])
    checks.append(
        {
            "validation_case": "metrics_are_finite",
            "success": len(finite_metric_rows) == len(metrics),
            "detail": f"finite rows: {len(finite_metric_rows)}/{len(metrics)}",
        }
    )
    neural_rows = metrics[metrics.get("model_name", pd.Series(dtype=str)).astype(str) == "neural_embedding_mlp"]
    baseline_rows = metrics[metrics.get("model_name", pd.Series(dtype=str)).astype(str).isin({"global_median", "source_metric_median"})]
    checks.append(
        {
            "validation_case": "baselines_and_neural_present",
            "success": not neural_rows.empty and not baseline_rows.empty,
            "detail": f"neural rows: {len(neural_rows)}; baseline rows: {len(baseline_rows)}",
        }
    )
    return pd.DataFrame(checks)


def stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
