"""Emergent tissue-field predictors for E04 S12.

S12 is an offline analysis step.  It never feeds predictors back into local
policy decisions.  Target-derived labels are used only as future outcomes, and
predictor feature sets are restricted to no-field context, local order proxies,
and the S07/S10-allowed signal fields: ``blocked`` and ``frustrated``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.e02.deterministic_simulator import stable_json_sha256
from src.e04.no_oracle_protocol import (
    ALLOWED_TRAINING_SIGNAL_FIELDS,
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    LOCAL_ONLY_PROTOCOL_ID,
)


S12_PROTOCOL_ID = f"{LOCAL_ONLY_PROTOCOL_ID}:e04_s12_tissue_field_predictors"
S12_ALLOWED_SIGNAL_FIELDS = ("blocked", "frustrated")
S12_FORBIDDEN_PREDICTOR_TOKENS = (
    "sorted",
    "target",
    "target_seeking",
    "morphogen",
    "rank",
    "full_array",
    "whole_array",
    "all_values",
    "final",
)
PREDICTION_TARGETS = (
    "future_repair_12",
    "future_failure_12",
    "future_movement_4",
    "future_frustration_4",
)
NO_FIELD_FEATURES = (
    "event_fraction",
    "array_size_norm",
    "damage_level_norm",
    "current_frozen_fraction",
    "perturbation_event_flag",
    "policy_mode_order_norm",
)
LOCAL_ORDER_FEATURES = (
    "order_pair_fraction",
    "order_error_fraction",
    "local_actor_order_fraction",
    "current_frozen_neighbor_fraction",
)
SIGNAL_FIELD_FEATURES = (
    "blocked_field_mean",
    "blocked_field_max",
    "blocked_field_std",
    "blocked_field_local",
    "blocked_field_gradient_mean",
    "frustrated_field_mean",
    "frustrated_field_max",
    "frustrated_field_std",
    "frustrated_field_local",
    "frustrated_field_gradient_mean",
)
FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
    "no_field_baseline": NO_FIELD_FEATURES,
    "local_order_baseline": (*NO_FIELD_FEATURES, *LOCAL_ORDER_FEATURES),
    "allowed_signal_fields": SIGNAL_FIELD_FEATURES,
    "local_order_plus_allowed_signal_fields": (*NO_FIELD_FEATURES, *LOCAL_ORDER_FEATURES, *SIGNAL_FIELD_FEATURES),
}


@dataclass(frozen=True)
class FieldModelResult:
    """One target/feature-set predictor result."""

    target_name: str
    feature_set: str
    model_status: str
    train_rows: int
    test_rows: int
    train_positive_fraction: float | None
    test_positive_fraction: float | None
    test_roc_auc: float | None
    test_average_precision: float | None
    test_accuracy: float | None
    test_brier: float | None
    local_order_auc_delta: float | None
    no_field_auc_delta: float | None
    feature_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "feature_set": self.feature_set,
            "model_status": self.model_status,
            "train_rows": int(self.train_rows),
            "test_rows": int(self.test_rows),
            "train_positive_fraction": self.train_positive_fraction,
            "test_positive_fraction": self.test_positive_fraction,
            "test_roc_auc": self.test_roc_auc,
            "test_average_precision": self.test_average_precision,
            "test_accuracy": self.test_accuracy,
            "test_brier": self.test_brier,
            "local_order_auc_delta": self.local_order_auc_delta,
            "no_field_auc_delta": self.no_field_auc_delta,
            "feature_columns_json": json.dumps(list(self.feature_columns), separators=(",", ":")),
        }


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, str) and value:
        return list(json.loads(value))
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return list(value)
    return []


def local_order_features(values: Sequence[int], actor_position: int | None, frozen_positions: Sequence[int]) -> dict[str, float]:
    """Compute offline local-order baseline features without using target-derived signal fields."""

    values = [int(value) for value in values]
    if len(values) <= 1:
        pair_fraction = 1.0
    else:
        ordered = sum(1 for idx in range(1, len(values)) if values[idx - 1] <= values[idx])
        pair_fraction = float(ordered / (len(values) - 1))
    if actor_position is None or actor_position < 0 or actor_position >= len(values):
        actor_fraction = pair_fraction
        neighbor_fraction = 0.0
    else:
        checks: list[bool] = []
        if actor_position > 0:
            checks.append(values[actor_position - 1] <= values[actor_position])
        if actor_position < len(values) - 1:
            checks.append(values[actor_position] <= values[actor_position + 1])
        actor_fraction = float(sum(1 for item in checks if item) / len(checks)) if checks else 1.0
        frozen_set = {int(position) for position in frozen_positions}
        neighbors = [idx for idx in (actor_position - 1, actor_position + 1) if 0 <= idx < len(values)]
        neighbor_fraction = float(sum(1 for idx in neighbors if idx in frozen_set) / len(neighbors)) if neighbors else 0.0
    return {
        "order_pair_fraction": pair_fraction,
        "order_error_fraction": 1.0 - pair_fraction,
        "local_actor_order_fraction": actor_fraction,
        "current_frozen_neighbor_fraction": neighbor_fraction,
    }


def diffuse_field(field: np.ndarray, *, rate: float = 0.25, decay: float = 0.0) -> np.ndarray:
    """Apply one S02-style 1D diffusion step to a scalar field."""

    if len(field) == 0:
        return field
    retained = 1.0 - float(decay)
    updated = np.zeros_like(field, dtype=float)
    for idx, value in enumerate(field):
        left = field[idx - 1] if idx > 0 else value
        right = field[idx + 1] if idx < len(field) - 1 else value
        updated[idx] = retained * ((1.0 - 2.0 * rate) * value + rate * left + rate * right)
    return updated


def field_features(blocked_field: np.ndarray, frustrated_field: np.ndarray, actor_position: int | None) -> dict[str, float]:
    def stats(prefix: str, values: np.ndarray) -> dict[str, float]:
        if len(values) == 0:
            return {
                f"{prefix}_field_mean": 0.0,
                f"{prefix}_field_max": 0.0,
                f"{prefix}_field_std": 0.0,
                f"{prefix}_field_local": 0.0,
                f"{prefix}_field_gradient_mean": 0.0,
            }
        if actor_position is None or actor_position < 0 or actor_position >= len(values):
            local = float(values.mean())
        else:
            local = float(values[int(actor_position)])
        gradient = float(np.mean(np.abs(np.diff(values)))) if len(values) > 1 else 0.0
        return {
            f"{prefix}_field_mean": float(values.mean()),
            f"{prefix}_field_max": float(values.max()),
            f"{prefix}_field_std": float(values.std()),
            f"{prefix}_field_local": local,
            f"{prefix}_field_gradient_mean": gradient,
        }

    return {**stats("blocked", blocked_field), **stats("frustrated", frustrated_field)}


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return int(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return float(value)


def _future_window(values: Sequence[Any], start: int, horizon: int) -> list[Any]:
    return list(values[start + 1 : min(len(values), start + 1 + int(horizon))])


def _labels_for_run(run_trace: pd.DataFrame) -> dict[str, list[float | None]]:
    target_met = [bool(value) for value in run_trace["target_met"].tolist()]
    swap_delta = [_safe_int(value) for value in run_trace["swap_delta"].tolist()]
    frozen_delta = [_safe_int(value) for value in run_trace["frozen_attempt_delta"].tolist()]
    fatigue_delta = [_safe_int(value) for value in run_trace["fatigue_transition_delta"].tolist()]
    applied = [str(value) for value in run_trace["applied_action"].tolist()]
    labels: dict[str, list[float | None]] = {name: [] for name in PREDICTION_TARGETS}
    for idx, current_target in enumerate(target_met):
        target_window = _future_window(target_met, idx, 12)
        movement_window = _future_window(swap_delta, idx, 4)
        frozen_window = _future_window(frozen_delta, idx, 4)
        fatigue_window = _future_window(fatigue_delta, idx, 4)
        action_window = _future_window(applied, idx, 4)
        labels["future_repair_12"].append(
            None if current_target or not target_window else float(any(target_window))
        )
        labels["future_failure_12"].append(
            None if (not current_target) or not target_window else float(any(not item for item in target_window))
        )
        labels["future_movement_4"].append(
            None if not movement_window else float(any(value > 0 for value in movement_window))
        )
        frustration = any(value > 0 for value in frozen_window) or any(value > 0 for value in fatigue_window) or any(
            item in {"blocked_swap_attempt", "impaired_skip"} for item in action_window
        )
        labels["future_frustration_4"].append(None if not action_window else float(frustration))
    return labels


def build_tissue_field_feature_rows(
    outcomes_df: pd.DataFrame,
    traces_df: pd.DataFrame,
    *,
    diffusion_rate: float = 0.25,
    decay: float = 0.02,
) -> pd.DataFrame:
    """Build event-level S12 predictor rows from S11 outcomes and traces."""

    outcome_by_run = outcomes_df.set_index("run_id", drop=False)
    rows: list[dict[str, Any]] = []
    for run_id, run_trace in traces_df.groupby("run_id", sort=False):
        if run_id not in outcome_by_run.index:
            continue
        meta = outcome_by_run.loc[run_id]
        run_trace = run_trace.sort_values("event_step").reset_index(drop=True)
        array_size = int(meta["array_size"])
        blocked_field = np.zeros(array_size, dtype=float)
        frustrated_field = np.zeros(array_size, dtype=float)
        labels = _labels_for_run(run_trace)
        for idx, trace in run_trace.iterrows():
            actor_position = None if pd.isna(trace["activated_position"]) else int(trace["activated_position"])
            values = [int(value) for value in parse_json_list(trace["current_values_json"])]
            frozen_positions = [int(value) for value in parse_json_list(trace["current_frozen_positions_json"])]
            if actor_position is not None and 0 <= actor_position < array_size:
                blocked_deposit = 1.0 if _safe_int(trace["frozen_attempt_delta"]) > 0 or trace["applied_action"] == "blocked_swap_attempt" else 0.0
                no_progress = _safe_int(trace["swap_delta"]) == 0 and int(trace["event_step"]) > 0
                frustrated_deposit = 1.0 if blocked_deposit or no_progress or _safe_int(trace["fatigue_transition_delta"]) > 0 else 0.0
                blocked_field[actor_position] += blocked_deposit
                frustrated_field[actor_position] += frustrated_deposit
            blocked_field = diffuse_field(blocked_field, rate=diffusion_rate, decay=decay)
            frustrated_field = diffuse_field(frustrated_field, rate=diffusion_rate, decay=decay)
            order = local_order_features(values, actor_position, frozen_positions)
            fields = field_features(blocked_field, frustrated_field, actor_position)
            max_events = max(1, int(meta["max_events"]))
            row = {
                "predictor_row_id": stable_json_sha256({"run_id": run_id, "event_step": int(trace["event_step"])}),
                "run_id": str(run_id),
                "event_step": int(trace["event_step"]),
                "split": "test" if int(meta["activation_seed"]) % 2 == 0 else "train",
                "split_seed_group": int(meta["activation_seed"]),
                "split_perturbation_group": str(meta["schedule_sha256"]),
                "competency_axis": str(meta["competency_axis"]),
                "scenario_name": str(meta["scenario_name"]),
                "policy_mode": str(meta["policy_mode"]),
                "policy_mode_order": int(meta["policy_mode_order"]),
                "s08_source_policy_id": str(meta["s08_source_policy_id"]),
                "activation_seed": int(meta["activation_seed"]),
                "schedule_seed": int(meta["schedule_seed"]),
                "array_size": array_size,
                "damage_level": int(meta["damage_level"]),
                "event_fraction": float(int(trace["event_step"]) / max_events),
                "array_size_norm": float(array_size / 12.0),
                "damage_level_norm": float(int(meta["damage_level"]) / 3.0),
                "current_frozen_fraction": float(len(frozen_positions) / max(1, array_size)),
                "perturbation_event_flag": float(_safe_int(trace["perturbation_count_at_event"]) > 0),
                "policy_mode_order_norm": float(int(meta["policy_mode_order"]) / 6.0),
                "label_current_target_met": bool(trace["target_met"]),
                "source_trace_sortedness_percent": float(trace["sortedness_percent"]),
                "source_trace_applied_action": str(trace["applied_action"]),
                "source_trace_feature_protocol_id": str(trace["feature_protocol_id"]),
                "uses_global_oracle": bool(meta["uses_global_oracle"]) or bool(trace["feature_audit_uses_oracle"]),
                **order,
                **fields,
            }
            for target_name, values_for_target in labels.items():
                row[target_name] = values_for_target[idx]
            rows.append(row)
    return pd.DataFrame(rows)


def predictor_feature_audit(feature_sets: Mapping[str, Sequence[str]] = FEATURE_SETS) -> dict[str, Any]:
    """Audit that predictor feature columns exclude S07/S10 forbidden fields."""

    feature_columns = sorted({column for columns in feature_sets.values() for column in columns})
    serialized = json.dumps(feature_columns, sort_keys=True, separators=(",", ":")).lower()
    forbidden_hits = [token for token in S12_FORBIDDEN_PREDICTOR_TOKENS if token in serialized]
    signal_columns = [column for column in feature_columns if "_field_" in column]
    disallowed_signal_columns = [
        column for column in signal_columns if not any(column.startswith(f"{field}_field_") for field in S12_ALLOWED_SIGNAL_FIELDS)
    ]
    return {
        "success": not forbidden_hits and not disallowed_signal_columns,
        "featureColumns": feature_columns,
        "forbiddenTokenHits": forbidden_hits,
        "signalFieldFeatureColumns": signal_columns,
        "disallowedSignalFieldColumns": disallowed_signal_columns,
        "allowedSignalFields": list(S12_ALLOWED_SIGNAL_FIELDS),
        "excludedSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
    }


def split_audit(feature_df: pd.DataFrame) -> dict[str, Any]:
    train = feature_df[feature_df["split"] == "train"]
    test = feature_df[feature_df["split"] == "test"]
    train_seeds = set(map(int, train["activation_seed"].unique()))
    test_seeds = set(map(int, test["activation_seed"].unique()))
    train_perturbations = set(map(str, train["split_perturbation_group"].unique()))
    test_perturbations = set(map(str, test["split_perturbation_group"].unique()))
    return {
        "success": bool(
            len(train) > 0
            and len(test) > 0
            and not (train_seeds & test_seeds)
            and not (train_perturbations & test_perturbations)
        ),
        "trainRows": int(len(train)),
        "testRows": int(len(test)),
        "trainSeeds": sorted(train_seeds),
        "testSeeds": sorted(test_seeds),
        "trainPerturbationGroups": len(train_perturbations),
        "testPerturbationGroups": len(test_perturbations),
        "overlapSeeds": sorted(train_seeds & test_seeds),
        "overlapPerturbationGroups": sorted(train_perturbations & test_perturbations),
    }


def _safe_metric(metric_fn: Any, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        return float(metric_fn(y_true, y_score))
    except ValueError:
        return None


def evaluate_field_predictors(
    feature_df: pd.DataFrame,
    *,
    targets: Sequence[str] = PREDICTION_TARGETS,
    feature_sets: Mapping[str, Sequence[str]] = FEATURE_SETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test grouped logistic predictors for all target and feature sets."""

    result_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for target_name in targets:
        target_df = feature_df[feature_df[target_name].notna()].copy()
        for feature_set, columns in feature_sets.items():
            columns = tuple(columns)
            train = target_df[target_df["split"] == "train"]
            test = target_df[target_df["split"] == "test"]
            train_rows = int(len(train))
            test_rows = int(len(test))
            train_pos = float(train[target_name].mean()) if train_rows else None
            test_pos = float(test[target_name].mean()) if test_rows else None
            status = "ok"
            if train_rows == 0 or test_rows == 0:
                status = "missing_split_rows"
            elif train[target_name].nunique() < 2:
                status = "single_train_class"
            elif test[target_name].nunique() < 2:
                status = "single_test_class"
            if status != "ok":
                result_rows.append(
                    FieldModelResult(
                        target_name=target_name,
                        feature_set=feature_set,
                        model_status=status,
                        train_rows=train_rows,
                        test_rows=test_rows,
                        train_positive_fraction=train_pos,
                        test_positive_fraction=test_pos,
                        test_roc_auc=None,
                        test_average_precision=None,
                        test_accuracy=None,
                        test_brier=None,
                        local_order_auc_delta=None,
                        no_field_auc_delta=None,
                        feature_columns=columns,
                    ).to_dict()
                )
                continue
            x_train = train[list(columns)].to_numpy(dtype=float)
            y_train = train[target_name].to_numpy(dtype=int)
            x_test = test[list(columns)].to_numpy(dtype=float)
            y_test = test[target_name].to_numpy(dtype=int)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0),
            )
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            coefficients = model.named_steps["logisticregression"].coef_[0]
            for column, coefficient in zip(columns, coefficients, strict=True):
                importance_rows.append(
                    {
                        "target_name": target_name,
                        "feature_set": feature_set,
                        "feature_name": column,
                        "coefficient": float(coefficient),
                        "abs_coefficient": float(abs(coefficient)),
                    }
                )
            result_rows.append(
                FieldModelResult(
                    target_name=target_name,
                    feature_set=feature_set,
                    model_status=status,
                    train_rows=train_rows,
                    test_rows=test_rows,
                    train_positive_fraction=train_pos,
                    test_positive_fraction=test_pos,
                    test_roc_auc=_safe_metric(roc_auc_score, y_test, probabilities),
                    test_average_precision=_safe_metric(average_precision_score, y_test, probabilities),
                    test_accuracy=float(accuracy_score(y_test, predictions)),
                    test_brier=float(brier_score_loss(y_test, probabilities)),
                    local_order_auc_delta=None,
                    no_field_auc_delta=None,
                    feature_columns=columns,
                ).to_dict()
            )
    results = pd.DataFrame(result_rows)
    if results.empty:
        return results, pd.DataFrame(importance_rows)
    baselines = results.pivot(index="target_name", columns="feature_set", values="test_roc_auc")
    local_order = baselines.get("local_order_baseline", pd.Series(dtype=float))
    no_field = baselines.get("no_field_baseline", pd.Series(dtype=float))
    results["local_order_auc_delta"] = [
        None if pd.isna(row["test_roc_auc"]) or row["target_name"] not in local_order.index or pd.isna(local_order[row["target_name"]])
        else float(row["test_roc_auc"] - local_order[row["target_name"]])
        for _, row in results.iterrows()
    ]
    results["no_field_auc_delta"] = [
        None if pd.isna(row["test_roc_auc"]) or row["target_name"] not in no_field.index or pd.isna(no_field[row["target_name"]])
        else float(row["test_roc_auc"] - no_field[row["target_name"]])
        for _, row in results.iterrows()
    ]
    return results, pd.DataFrame(importance_rows)


def infer_field_predictor_outcome(results_df: pd.DataFrame) -> dict[str, Any]:
    """Classify whether allowed field predictors add test-set value."""

    rows = results_df[results_df["feature_set"] == "local_order_plus_allowed_signal_fields"].copy()
    rows = rows[rows["model_status"] == "ok"]
    if rows.empty:
        return {
            "outcome": "null",
            "criterion": "mean AUROC delta over local-order baseline >= 0.03 and positive target fraction >= 0.50",
            "reason": "no evaluable local-order-plus-signal models",
        }
    deltas = rows["local_order_auc_delta"].dropna()
    if deltas.empty:
        return {
            "outcome": "null",
            "criterion": "mean AUROC delta over local-order baseline >= 0.03 and positive target fraction >= 0.50",
            "reason": "no evaluable AUROC deltas",
        }
    mean_delta = float(deltas.mean())
    positive_fraction = float((deltas > 0.0).mean())
    if mean_delta >= 0.03 and positive_fraction >= 0.50:
        outcome = "supportive"
    elif mean_delta <= 0.0 and positive_fraction <= 0.25:
        outcome = "constraining/contradictory"
    else:
        outcome = "null"
    return {
        "outcome": outcome,
        "criterion": "mean AUROC delta over local-order baseline >= 0.03 and positive target fraction >= 0.50",
        "meanAurocDeltaVsLocalOrder": mean_delta,
        "positiveTargetFractionVsLocalOrder": positive_fraction,
        "evaluatedTargets": rows["target_name"].tolist(),
    }
