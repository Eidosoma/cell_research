"""E06 S13 final-state predictive and exploratory causal-style models.

This module keeps predictor variables deliberately upstream of final outcomes.
Final metrics, S05/S07 derived descriptors, and S06 dominance/context evidence
are allowed as labels or explanatory annotations, but not as model predictors.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.e06.mixture_ratios import EXPERIMENT_ID, stable_hash


STEP_ID = "S13"
MODEL_INPUT_SCHEMA = "eidosoma.e06.s13_model_input.v1"
PREDICTION_SCHEMA = "eidosoma.e06.s13_heldout_prediction.v1"
PERFORMANCE_SCHEMA = "eidosoma.e06.s13_model_performance.v1"
FEATURE_IMPORTANCE_SCHEMA = "eidosoma.e06.s13_feature_importance.v1"
FEATURE_CATALOG_SCHEMA = "eidosoma.e06.s13_feature_catalog.v1"
LEAKAGE_AUDIT_SCHEMA = "eidosoma.e06.s13_leakage_audit.v1"
S06_CONTEXT_SCHEMA = "eidosoma.e06.s13_s06_context_annotation.v1"
CAUSAL_SCREEN_SCHEMA = "eidosoma.e06.s13_exploratory_causal_screen.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s13_validation.v1"

EXPECTED_MODEL_ROW_SOURCE_STEPS = ("S02", "S03", "S04", "S06", "S08", "S09", "S10", "S11", "S12")
ANNOTATION_ONLY_SOURCE_STEPS = ("S05", "S07")

NUMERIC_FEATURES = (
    "array_size",
    "event_cap",
    "policy_count",
    "ratio_max_fraction",
    "ratio_min_fraction",
    "ratio_entropy",
    "ratio_imbalance",
    "explicit_recognition_used",
    "global_controller_like",
    "governance_influence_radius_cells",
    "memory_policy_present",
    "history_introduction_event_fraction",
    "history_transient_start_fraction",
    "history_transient_duration_fraction",
    "history_prior_exposure_fraction",
    "history_reset_after_exposure",
    "history_preserves_memory_across_reset",
    "graft_event_fraction",
    "graft_size_fraction",
    "clone_introduction_event_fraction",
    "clone_initial_fraction",
    "selfish_allows_replication",
)

CATEGORICAL_FEATURES = (
    "source_research_step_id",
    "source_table_kind",
    "panel",
    "condition_kind",
    "candidate_reason",
    "arrangement",
    "arrangement_role",
    "orientation",
    "value_profile",
    "perturbation_profile",
    "goal_profile_id",
    "goal_compatibility_class",
    "policy_signature",
    "display_signature",
    "source_category_signature",
    "ratio_label",
    "interface_rule_id",
    "interface_mechanism_family",
    "interface_access_stratum",
    "governance_mechanism_id",
    "governance_mechanism_family",
    "governance_access_stratum",
    "governance_information_access_class",
    "intervention_stratum",
    "history_id",
    "history_family",
    "history_start_mode",
    "source_context_family",
    "graft_location",
    "graft_mode",
    "clone_initial_location",
    "clone_mode",
    "selfish_objective_id",
    "selfish_objective_family",
)

CLASSIFICATION_TARGET = "final_state_class"
REGRESSION_TARGETS = (
    "final_target_quality",
    "aggregation_delta_percent",
    "goal_conflict_index",
    "largest_block_fraction",
)

OUTCOME_DERIVED_PATTERNS = (
    "final_",
    "source_final",
    "target_quality",
    "reference_increasing",
    "goal_conflict",
    "goal_state_class",
    "aggregation_delta_percent",
    "largest_block_fraction",
    "interface_count",
    "position_bias",
    "work_count",
    "swap_count",
    "block_fraction",
    "governance_cost",
    "graft_outcome",
    "clone_growth",
    "clone_fraction_peak",
    "takeover",
    "damage",
    "s05_",
    "s06_",
    "s07_",
    "dominance",
    "mosaic",
    "continuum",
)


@dataclass(frozen=True)
class S13Config:
    """Configuration for S13 predictive modeling."""

    random_state: int = 2026070313
    split_count: int = 3
    test_fraction: float = 0.25
    random_forest_trees: int = 96
    random_forest_min_leaf: int = 3
    max_permutation_rows: int = 300
    permutation_repeats: int = 1
    permutation_split_limit: int = 1
    workers: int = 4
    supportive_accuracy_delta_threshold: float = 0.08
    supportive_regression_r2_threshold: float = 0.10


def parse_json_list(value: Any) -> tuple[Any, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return tuple()
    if isinstance(value, str):
        if not value:
            return tuple()
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return tuple()
        if isinstance(parsed, list):
            return tuple(parsed)
        return tuple()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return tuple(value)
    return tuple()


def json_list(values: Sequence[Any]) -> str:
    return json.dumps(list(values), separators=(",", ":"), default=str)


def safe_str(value: Any, default: str = "not_applicable") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    text = str(value)
    return text if text and text.lower() != "nan" else default


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def first_finite(*values: Any, default: float = math.nan) -> float:
    for value in values:
        result = safe_float(value, math.nan)
        if math.isfinite(result):
            return result
    return default


def bool_feature(value: Any) -> float:
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"true", "1", "yes"} else 0.0
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return 1.0 if bool(value) else 0.0


def ratio_features(ratio_text: Any) -> dict[str, Any]:
    ratios = [float(value) for value in parse_json_list(ratio_text) if math.isfinite(safe_float(value))]
    if not ratios:
        return {
            "policy_count": 0.0,
            "ratio_max_fraction": 0.0,
            "ratio_min_fraction": 0.0,
            "ratio_entropy": 0.0,
            "ratio_imbalance": 0.0,
            "ratio_label": "not_applicable",
        }
    total = sum(ratios) or 1.0
    normalized = [max(0.0, value / total) for value in ratios]
    entropy = -sum(value * math.log(value) for value in normalized if value > 0.0)
    max_entropy = math.log(len(normalized)) if len(normalized) > 1 else 1.0
    return {
        "policy_count": float(len(normalized)),
        "ratio_max_fraction": float(max(normalized)),
        "ratio_min_fraction": float(min(normalized)),
        "ratio_entropy": float(entropy / max_entropy) if max_entropy else 0.0,
        "ratio_imbalance": float(max(normalized) - min(normalized)),
        "ratio_label": ":".join(str(int(round(value * 100))) for value in normalized),
    }


def fraction_of_event(value: Any, event_cap: Any) -> float:
    numerator = safe_float(value, math.nan)
    denominator = safe_float(event_cap, math.nan)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def signature_from_json(value: Any, *, sort_values: bool = False) -> str:
    items = [str(item) for item in parse_json_list(value)]
    if sort_values:
        items = sorted(items)
    return json_list(items) if items else "not_applicable"


def _prediction_row_id(source_step: str, raw: Mapping[str, Any], source_table_kind: str) -> str:
    payload = {
        "sourceStep": source_step,
        "sourceTableKind": source_table_kind,
        "conditionId": raw.get("condition_id", raw.get("source_condition_id", "")),
        "analysisConditionId": raw.get("analysis_condition_id", ""),
        "seed": raw.get("seed", ""),
        "historyId": raw.get("history_id", ""),
        "interfaceRuleId": raw.get("interface_rule_id", ""),
        "governanceMechanismId": raw.get("governance_mechanism_id", ""),
        "graftSpecId": raw.get("graft_spec_id", ""),
        "cloneSpecId": raw.get("clone_spec_id", ""),
        "selfishObjectiveId": raw.get("selfish_objective_id", ""),
    }
    return f"s13_{stable_hash(payload)[:18]}"


def harmonize_raw_row(source_step: str, raw: Mapping[str, Any], *, source_table_kind: str) -> dict[str, Any]:
    """Convert one source row into the S13 model input schema."""

    event_cap = first_finite(raw.get("event_cap"), default=0.0)
    condition_id = safe_str(raw.get("analysis_condition_id", raw.get("condition_id", raw.get("source_condition_id", ""))))
    source_condition_id = safe_str(raw.get("source_condition_id", raw.get("condition_id", condition_id)))
    row: dict[str, Any] = {
        "schema": MODEL_INPUT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "prediction_row_id": _prediction_row_id(source_step, raw, source_table_kind),
        "source_research_step_id": source_step,
        "source_table_kind": source_table_kind,
        "source_condition_id": source_condition_id,
        "source_run_condition_id": safe_str(raw.get("condition_id", source_condition_id)),
        "condition_group_id": f"{source_step}:{condition_id}",
        "seed": int(first_finite(raw.get("seed"), default=0.0)),
        "final_state_class": safe_str(raw.get("final_state_class"), default="unlabeled"),
        "goal_state_class": safe_str(raw.get("goal_state_class"), default="not_available"),
        "final_target_quality": first_finite(raw.get("final_target_quality"), raw.get("assigned_policy_mean_sortedness"), raw.get("final_inversion_sortedness"), default=math.nan),
        "aggregation_delta_percent": first_finite(raw.get("aggregation_delta_percent"), default=math.nan),
        "goal_conflict_index": first_finite(raw.get("goal_conflict_index"), raw.get("goal_alignment_gap"), default=0.0),
        "largest_block_fraction": first_finite(raw.get("largest_block_fraction"), default=math.nan),
        "base_contest_id": safe_str(raw.get("base_contest_id"), default=""),
        "context_annotation_key": f"{source_step}:{safe_str(raw.get('base_contest_id'), default='')}",
    }
    row.update(ratio_features(raw.get("ratio_targets_json")))

    for column in (
        "panel",
        "condition_kind",
        "candidate_reason",
        "arrangement",
        "arrangement_role",
        "orientation",
        "value_profile",
        "perturbation_profile",
        "goal_profile_id",
        "goal_compatibility_class",
        "interface_rule_id",
        "interface_mechanism_family",
        "interface_access_stratum",
        "governance_mechanism_id",
        "governance_mechanism_family",
        "governance_access_stratum",
        "governance_information_access_class",
        "intervention_stratum",
        "history_id",
        "history_family",
        "history_start_mode",
        "source_context_family",
        "graft_location",
        "graft_mode",
        "clone_initial_location",
        "clone_mode",
        "selfish_objective_id",
        "selfish_objective_family",
    ):
        row[column] = safe_str(raw.get(column))

    row["policy_signature"] = signature_from_json(raw.get("policy_ids_json"), sort_values=True)
    row["display_signature"] = signature_from_json(raw.get("display_names_json"), sort_values=True)
    row["source_category_signature"] = signature_from_json(raw.get("source_categories_json"), sort_values=True)
    row["array_size"] = first_finite(raw.get("array_size"), default=0.0)
    row["event_cap"] = event_cap
    row["explicit_recognition_used"] = bool_feature(raw.get("explicit_recognition_used"))
    row["global_controller_like"] = bool_feature(raw.get("global_controller_like"))
    row["memory_policy_present"] = bool_feature(raw.get("memory_policy_present"))
    row["selfish_allows_replication"] = bool_feature(raw.get("selfish_allows_replication"))
    row["history_reset_after_exposure"] = bool_feature(raw.get("reset_after_exposure"))
    row["history_preserves_memory_across_reset"] = bool_feature(raw.get("preserves_memory_across_reset"))
    row["governance_influence_radius_cells"] = first_finite(raw.get("governance_influence_radius_cells"), default=0.0)
    row["history_introduction_event_fraction"] = fraction_of_event(raw.get("introduction_event"), event_cap)
    row["history_transient_start_fraction"] = fraction_of_event(raw.get("transient_start_event"), event_cap)
    transient_start = safe_float(raw.get("transient_start_event"), math.nan)
    transient_end = safe_float(raw.get("transient_end_event"), math.nan)
    row["history_transient_duration_fraction"] = float((transient_end - transient_start) / event_cap) if math.isfinite(transient_start) and math.isfinite(transient_end) and event_cap > 0 else 0.0
    row["history_prior_exposure_fraction"] = fraction_of_event(raw.get("prior_exposure_events"), event_cap)
    row["graft_event_fraction"] = fraction_of_event(raw.get("graft_event"), event_cap)
    row["graft_size_fraction"] = first_finite(raw.get("graft_size"), default=0.0) / max(1.0, first_finite(raw.get("array_size"), default=1.0))
    row["clone_introduction_event_fraction"] = fraction_of_event(raw.get("clone_introduction_event"), event_cap)
    row["clone_initial_fraction"] = first_finite(raw.get("clone_initial_fraction"), default=0.0)
    return row


def build_model_input_matrix(
    s07_mosaic: pd.DataFrame,
    direct_frames_by_step: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build one row-level S13 modeling matrix from S02-S12 outcomes."""

    rows: list[dict[str, Any]] = []
    if not s07_mosaic.empty:
        for raw in s07_mosaic.to_dict(orient="records"):
            source_step = safe_str(raw.get("source_research_step_id"))
            if source_step in {"S02", "S03", "S04", "S06"}:
                rows.append(harmonize_raw_row(source_step, raw, source_table_kind="s07_mosaic_harmonized"))
    for source_step, frame in direct_frames_by_step.items():
        if frame.empty:
            continue
        for raw in frame.to_dict(orient="records"):
            rows.append(harmonize_raw_row(source_step, raw, source_table_kind="direct_run"))
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return matrix
    for column in NUMERIC_FEATURES + REGRESSION_TARGETS:
        if column in matrix.columns:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    for column in CATEGORICAL_FEATURES + (CLASSIFICATION_TARGET,):
        if column in matrix.columns:
            matrix[column] = matrix[column].fillna("not_applicable").astype(str)
    matrix = matrix.drop_duplicates("prediction_row_id", keep="first").reset_index(drop=True)
    return matrix


def build_feature_catalog() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in NUMERIC_FEATURES:
        rows.append(
            {
                "schema": FEATURE_CATALOG_SCHEMA,
                "feature_name": column,
                "feature_type": "numeric",
                "predictor_role": "allowed_predictor",
                "leakage_review": "upstream experimental design variable or predeclared mechanism metadata",
            }
        )
    for column in CATEGORICAL_FEATURES:
        rows.append(
            {
                "schema": FEATURE_CATALOG_SCHEMA,
                "feature_name": column,
                "feature_type": "categorical",
                "predictor_role": "allowed_predictor",
                "leakage_review": "upstream experimental design variable or policy/context identifier",
            }
        )
    for column in (CLASSIFICATION_TARGET, *REGRESSION_TARGETS):
        rows.append(
            {
                "schema": FEATURE_CATALOG_SCHEMA,
                "feature_name": column,
                "feature_type": "target",
                "predictor_role": "target_only",
                "leakage_review": "final outcome label; excluded from predictor matrix",
            }
        )
    for prefix in ("s05_*", "s06_*", "s07_*", "dominance_context", "source_final_*"):
        rows.append(
            {
                "schema": FEATURE_CATALOG_SCHEMA,
                "feature_name": prefix,
                "feature_type": "annotation_pattern",
                "predictor_role": "excluded_annotation",
                "leakage_review": "derived outcome/context annotation kept outside model predictors",
            }
        )
    return pd.DataFrame(rows)


def audit_predictor_leakage(matrix: pd.DataFrame, predictor_columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in matrix.columns:
        lower = column.lower()
        is_predictor = column in set(predictor_columns)
        matched_patterns = [pattern for pattern in OUTCOME_DERIVED_PATTERNS if pattern in lower]
        allowed_initial = lower.startswith("initial_")
        status = "predictor_allowed" if is_predictor else "not_used_as_predictor"
        if is_predictor and matched_patterns and not allowed_initial:
            status = "leakage_risk"
        rows.append(
            {
                "schema": LEAKAGE_AUDIT_SCHEMA,
                "column_name": column,
                "used_as_predictor": bool(is_predictor),
                "matched_outcome_patterns_json": json.dumps(matched_patterns, separators=(",", ":")),
                "leakage_status": status,
                "review_note": "Initial-condition metrics are allowed only if explicitly listed; final/S05/S06/S07/source-outcome descriptors are not predictors.",
            }
        )
    return pd.DataFrame(rows)


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="not_applicable")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def classification_models(config: S13Config) -> dict[str, Any]:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_l2": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=config.random_state)),
            ]
        ),
        "decision_tree_depth8": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("model", DecisionTreeClassifier(max_depth=8, min_samples_leaf=8, class_weight="balanced", random_state=config.random_state)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=config.random_forest_trees,
                        min_samples_leaf=config.random_forest_min_leaf,
                        class_weight="balanced_subsample",
                        random_state=config.random_state,
                        n_jobs=max(1, int(config.workers)),
                    ),
                ),
            ]
        ),
    }


def regression_models(config: S13Config) -> dict[str, Any]:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge_l2": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=config.random_forest_trees,
                        min_samples_leaf=config.random_forest_min_leaf,
                        random_state=config.random_state,
                        n_jobs=max(1, int(config.workers)),
                    ),
                ),
            ]
        ),
    }


def group_splits(matrix: pd.DataFrame, config: S13Config) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupShuffleSplit(n_splits=config.split_count, test_size=config.test_fraction, random_state=config.random_state)
    groups = matrix["condition_group_id"].astype(str).to_numpy()
    dummy_y = matrix[CLASSIFICATION_TARGET].astype(str).to_numpy()
    return [(np.asarray(train), np.asarray(test)) for train, test in splitter.split(matrix, dummy_y, groups)]


def _metric_or_nan(func: Any, *args: Any, **kwargs: Any) -> float:
    try:
        value = func(*args, **kwargs)
    except ValueError:
        return math.nan
    return float(value) if math.isfinite(float(value)) else math.nan


def _sample_for_permutation(x_test: pd.DataFrame, y_test: pd.Series, config: S13Config) -> tuple[pd.DataFrame, pd.Series]:
    if len(x_test) <= config.max_permutation_rows:
        return x_test, y_test
    rng = np.random.default_rng(config.random_state)
    indices = np.sort(rng.choice(np.arange(len(x_test)), size=config.max_permutation_rows, replace=False))
    return x_test.iloc[indices].copy(), y_test.iloc[indices].copy()


def _permutation_rows(
    estimator: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    target_name: str,
    model_id: str,
    split_index: int,
    scoring: str,
    config: S13Config,
) -> list[dict[str, Any]]:
    x_perm, y_perm = _sample_for_permutation(x_test, y_test, config)
    try:
        result = permutation_importance(
            estimator,
            x_perm,
            y_perm,
            n_repeats=config.permutation_repeats,
            random_state=config.random_state + split_index,
            scoring=scoring,
            n_jobs=1,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for column, mean, std in zip(x_perm.columns, result.importances_mean, result.importances_std, strict=True):
        rows.append(
            {
                "schema": FEATURE_IMPORTANCE_SCHEMA,
                "target_name": target_name,
                "model_id": model_id,
                "split_index": int(split_index),
                "feature_name": str(column),
                "importance_mean": float(mean),
                "importance_std": float(std),
                "importance_method": "heldout_permutation_importance",
            }
        )
    return rows


def evaluate_models(matrix: pd.DataFrame, config: S13Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run group-held-out classification and regression models."""

    feature_columns = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    x = matrix[feature_columns].copy()
    splits = group_splits(matrix, config)
    performance_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    class_models = classification_models(config)
    for split_index, (train_idx, test_idx) in enumerate(splits):
        train_groups = set(matrix.iloc[train_idx]["condition_group_id"].astype(str))
        test_groups = set(matrix.iloc[test_idx]["condition_group_id"].astype(str))
        overlap = len(train_groups & test_groups)
        y_train = matrix.iloc[train_idx][CLASSIFICATION_TARGET].astype(str)
        y_test = matrix.iloc[test_idx][CLASSIFICATION_TARGET].astype(str)
        x_train = x.iloc[train_idx].copy()
        x_test = x.iloc[test_idx].copy()
        for model_id, model in class_models.items():
            estimator = model
            estimator.fit(x_train, y_train)
            pred = pd.Series(estimator.predict(x_test), index=x_test.index, dtype=str)
            performance_rows.append(
                {
                    "schema": PERFORMANCE_SCHEMA,
                    "target_name": CLASSIFICATION_TARGET,
                    "target_type": "classification",
                    "model_id": model_id,
                    "split_index": int(split_index),
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "train_group_count": int(len(train_groups)),
                    "test_group_count": int(len(test_groups)),
                    "train_test_group_overlap_count": int(overlap),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "balanced_accuracy": _metric_or_nan(balanced_accuracy_score, y_test, pred),
                    "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                    "mae": math.nan,
                    "rmse": math.nan,
                    "r2": math.nan,
                    "class_count": int(y_train.nunique()),
                }
            )
            for row_idx, observed, predicted in zip(x_test.index, y_test, pred, strict=True):
                prediction_rows.append(
                    {
                        "schema": PREDICTION_SCHEMA,
                        "prediction_row_id": matrix.loc[row_idx, "prediction_row_id"],
                        "source_research_step_id": matrix.loc[row_idx, "source_research_step_id"],
                        "condition_group_id": matrix.loc[row_idx, "condition_group_id"],
                        "target_name": CLASSIFICATION_TARGET,
                        "target_type": "classification",
                        "model_id": model_id,
                        "split_index": int(split_index),
                        "observed_value": str(observed),
                        "predicted_value": str(predicted),
                        "observed_numeric": math.nan,
                        "predicted_numeric": math.nan,
                    }
                )
            if model_id == "random_forest" and split_index < config.permutation_split_limit:
                importance_rows.extend(
                    _permutation_rows(
                        estimator,
                        x_test,
                        y_test,
                        target_name=CLASSIFICATION_TARGET,
                        model_id=model_id,
                        split_index=split_index,
                        scoring="balanced_accuracy",
                        config=config,
                    )
                )

        for target_name in REGRESSION_TARGETS:
            target_values = pd.to_numeric(matrix[target_name], errors="coerce")
            valid_train = train_idx[np.isfinite(target_values.iloc[train_idx].to_numpy(dtype=float))]
            valid_test = test_idx[np.isfinite(target_values.iloc[test_idx].to_numpy(dtype=float))]
            y_train_reg = target_values.iloc[valid_train]
            y_test_reg = target_values.iloc[valid_test]
            x_train_reg = x.iloc[valid_train].copy()
            x_test_reg = x.iloc[valid_test].copy()
            for model_id, model in regression_models(config).items():
                estimator = model
                if len(valid_train) < 4 or len(valid_test) < 4:
                    continue
                estimator.fit(x_train_reg, y_train_reg)
                pred_reg = pd.Series(estimator.predict(x_test_reg), index=x_test_reg.index, dtype=float)
                mse = mean_squared_error(y_test_reg, pred_reg)
                performance_rows.append(
                    {
                        "schema": PERFORMANCE_SCHEMA,
                        "target_name": target_name,
                        "target_type": "regression",
                        "model_id": model_id,
                        "split_index": int(split_index),
                        "train_rows": int(len(valid_train)),
                        "test_rows": int(len(valid_test)),
                        "train_group_count": int(len(set(matrix.iloc[valid_train]["condition_group_id"].astype(str)))),
                        "test_group_count": int(len(set(matrix.iloc[valid_test]["condition_group_id"].astype(str)))),
                        "train_test_group_overlap_count": int(overlap),
                        "accuracy": math.nan,
                        "balanced_accuracy": math.nan,
                        "macro_f1": math.nan,
                        "mae": float(mean_absolute_error(y_test_reg, pred_reg)),
                        "rmse": float(math.sqrt(mse)),
                        "r2": _metric_or_nan(r2_score, y_test_reg, pred_reg),
                        "class_count": math.nan,
                    }
                )
                for row_idx, observed, predicted in zip(x_test_reg.index, y_test_reg, pred_reg, strict=True):
                    prediction_rows.append(
                        {
                            "schema": PREDICTION_SCHEMA,
                            "prediction_row_id": matrix.loc[row_idx, "prediction_row_id"],
                            "source_research_step_id": matrix.loc[row_idx, "source_research_step_id"],
                            "condition_group_id": matrix.loc[row_idx, "condition_group_id"],
                            "target_name": target_name,
                            "target_type": "regression",
                            "model_id": model_id,
                            "split_index": int(split_index),
                            "observed_value": "",
                            "predicted_value": "",
                            "observed_numeric": float(observed),
                            "predicted_numeric": float(predicted),
                        }
                    )
                if model_id == "random_forest" and split_index < config.permutation_split_limit:
                    importance_rows.extend(
                        _permutation_rows(
                            estimator,
                            x_test_reg,
                            y_test_reg,
                            target_name=target_name,
                            model_id=model_id,
                            split_index=split_index,
                            scoring="r2",
                            config=config,
                        )
                    )
    return pd.DataFrame(performance_rows), pd.DataFrame(prediction_rows), pd.DataFrame(importance_rows)


def summarize_performance(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty:
        return performance.copy()
    rows: list[dict[str, Any]] = []
    for (target_name, target_type, model_id), group in performance.groupby(["target_name", "target_type", "model_id"], dropna=False):
        if target_type == "classification":
            rows.append(
                {
                    "target_name": target_name,
                    "target_type": target_type,
                    "model_id": model_id,
                    "split_count": int(group["split_index"].nunique()),
                    "mean_accuracy": float(group["accuracy"].mean()),
                    "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                    "mean_macro_f1": float(group["macro_f1"].mean()),
                    "mean_mae": math.nan,
                    "mean_rmse": math.nan,
                    "mean_r2": math.nan,
                }
            )
        else:
            rows.append(
                {
                    "target_name": target_name,
                    "target_type": target_type,
                    "model_id": model_id,
                    "split_count": int(group["split_index"].nunique()),
                    "mean_accuracy": math.nan,
                    "mean_balanced_accuracy": math.nan,
                    "mean_macro_f1": math.nan,
                    "mean_mae": float(group["mae"].mean()),
                    "mean_rmse": float(group["rmse"].mean()),
                    "mean_r2": float(group["r2"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_name", "target_type", "model_id"], kind="mergesort").reset_index(drop=True)


def build_causal_screen(feature_importance: pd.DataFrame, performance_summary: pd.DataFrame) -> pd.DataFrame:
    if feature_importance.empty:
        return pd.DataFrame()
    grouped = (
        feature_importance.groupby(["target_name", "model_id", "feature_name"], dropna=False)
        .agg(mean_importance=("importance_mean", "mean"), max_importance=("importance_mean", "max"), split_count=("split_index", "nunique"))
        .reset_index()
    )
    grouped = grouped.sort_values(["target_name", "mean_importance"], ascending=[True, False], kind="mergesort")
    top = grouped.groupby("target_name", group_keys=False).head(12).copy()
    top["schema"] = CAUSAL_SCREEN_SCHEMA
    top["screen_type"] = "predictive_permutation_importance"
    top["claim_scope"] = "exploratory_association_not_causal_proof"
    top["causal_caveat"] = "Feature ranking is from held-out predictive perturbations in an observational design matrix; it does not prove mechanistic causation."
    return top[
        [
            "schema",
            "target_name",
            "model_id",
            "feature_name",
            "mean_importance",
            "max_importance",
            "split_count",
            "screen_type",
            "claim_scope",
            "causal_caveat",
        ]
    ].reset_index(drop=True)


def build_s06_context_annotations(
    matrix: pd.DataFrame,
    dominance_context: pd.DataFrame,
    dominance_hierarchy: pd.DataFrame,
) -> pd.DataFrame:
    s06 = matrix[matrix["source_research_step_id"] == "S06"].copy()
    if s06.empty:
        return pd.DataFrame()
    context = dominance_context.copy() if not dominance_context.empty else pd.DataFrame()
    if not context.empty and "base_contest_id" in context.columns:
        context = context.add_prefix("s06_context_").rename(columns={"s06_context_base_contest_id": "base_contest_id"})
        s06 = s06.merge(context, on="base_contest_id", how="left")
    hierarchy_map = {}
    if not dominance_hierarchy.empty:
        hierarchy_map = {
            str(row["policy_id"]): {
                "net": safe_float(row.get("net_dominance_score")),
                "win_fraction": safe_float(row.get("win_fraction")),
            }
            for row in dominance_hierarchy.to_dict(orient="records")
        }
    rows: list[dict[str, Any]] = []
    for raw in s06.to_dict(orient="records"):
        policy_ids = [str(item) for item in parse_json_list(raw.get("policy_signature"))]
        nets = [hierarchy_map[item]["net"] for item in policy_ids if item in hierarchy_map and math.isfinite(hierarchy_map[item]["net"])]
        rows.append(
            {
                "schema": S06_CONTEXT_SCHEMA,
                "prediction_row_id": raw["prediction_row_id"],
                "source_condition_id": raw["source_condition_id"],
                "base_contest_id": raw.get("base_contest_id", ""),
                "context_dependent_dominance": raw.get("s06_context_context_dependent_dominance", ""),
                "dominance_margin_range": safe_float(raw.get("s06_context_dominance_margin_range"), math.nan),
                "policy_hierarchy_net_mean": float(np.mean(nets)) if nets else math.nan,
                "policy_hierarchy_net_spread": float(max(nets) - min(nets)) if nets else math.nan,
                "used_as_model_predictor": False,
                "annotation_role": "separate_explanatory_context",
            }
        )
    return pd.DataFrame(rows)


def train_final_models(matrix: pd.DataFrame, config: S13Config) -> dict[str, Any]:
    """Train final random-forest models on all rows for reuse."""

    x = matrix[list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)].copy()
    models: dict[str, Any] = {}
    classifier = classification_models(config)["random_forest"]
    classifier.fit(x, matrix[CLASSIFICATION_TARGET].astype(str))
    models["final_state_class_random_forest"] = classifier
    for target_name in REGRESSION_TARGETS:
        target = pd.to_numeric(matrix[target_name], errors="coerce")
        valid = np.isfinite(target.to_numpy(dtype=float))
        if valid.sum() < 8:
            continue
        model = regression_models(config)["random_forest"]
        model.fit(x.loc[valid], target.loc[valid])
        models[f"{target_name}_random_forest"] = model
    return models


def infer_s13_outcome(performance_summary: pd.DataFrame, config: S13Config) -> dict[str, Any]:
    if performance_summary.empty:
        return {
            "outcomeClassification": "null",
            "primaryReason": "no evaluable models",
            "bestClassifierAccuracy": math.nan,
            "dummyClassifierAccuracy": math.nan,
            "positiveRegressionTargetCount": 0,
        }
    cls = performance_summary[(performance_summary["target_name"] == CLASSIFICATION_TARGET) & (performance_summary["target_type"] == "classification")]
    best_cls = cls[cls["model_id"] == "random_forest"]["mean_accuracy"].max() if not cls.empty else math.nan
    dummy_cls = cls[cls["model_id"] == "dummy_most_frequent"]["mean_accuracy"].max() if not cls.empty else math.nan
    reg = performance_summary[(performance_summary["target_type"] == "regression") & (performance_summary["model_id"] == "random_forest")]
    positive_reg = int((reg["mean_r2"] >= config.supportive_regression_r2_threshold).sum()) if not reg.empty else 0
    if math.isfinite(best_cls) and math.isfinite(dummy_cls) and best_cls - dummy_cls >= config.supportive_accuracy_delta_threshold and positive_reg >= 2:
        outcome = "supportive"
        reason = "held-out final-state classifier beat the dummy baseline and multiple continuous outcome models had positive held-out R2"
    elif math.isfinite(best_cls) and math.isfinite(dummy_cls) and best_cls > dummy_cls:
        outcome = "null"
        reason = "classification improved over dummy, but continuous outcome support was limited under the configured threshold"
    else:
        outcome = "constraining/contradictory"
        reason = "held-out models did not improve enough over baseline to support predictive claims"
    return {
        "outcomeClassification": outcome,
        "primaryReason": reason,
        "bestClassifierAccuracy": float(best_cls) if math.isfinite(best_cls) else math.nan,
        "dummyClassifierAccuracy": float(dummy_cls) if math.isfinite(dummy_cls) else math.nan,
        "positiveRegressionTargetCount": positive_reg,
    }


def validate_s13_outputs(
    matrix: pd.DataFrame,
    performance: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    s06_annotations: pd.DataFrame,
    causal_screen: pd.DataFrame,
    model_artifact_count: int,
    figure_written: bool,
    unit_tests_success: bool,
    *,
    s05_context_available: bool,
    s07_context_available: bool,
    config: S13Config,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(case: str, success: bool, observed: Any, expected: str) -> None:
        rows.append(
            {
                "schema": VALIDATION_SCHEMA,
                "validation_case": case,
                "success": bool(success),
                "observed": str(observed),
                "expected": expected,
            }
        )

    source_steps = sorted(matrix["source_research_step_id"].dropna().astype(str).unique().tolist()) if not matrix.empty else []
    add(
        "model_input_matrix_nonempty_and_source_rows_covered",
        bool(len(matrix) > 0 and set(EXPECTED_MODEL_ROW_SOURCE_STEPS).issubset(source_steps)),
        f"rows={len(matrix)} sources={source_steps}",
        f"row-level model inputs cover {list(EXPECTED_MODEL_ROW_SOURCE_STEPS)}",
    )
    add(
        "s05_s07_contexts_handled_as_annotations",
        bool(s05_context_available and s07_context_available),
        f"s05_available={s05_context_available}; s07_available={s07_context_available}",
        "S05 and S07 context outputs are available but not used as predictor columns",
    )
    leakage_risks = leakage_audit[(leakage_audit["used_as_predictor"]) & (leakage_audit["leakage_status"] == "leakage_risk")]
    add(
        "no_outcome_derived_predictor_leakage",
        bool(leakage_risks.empty),
        f"leakage_risk_features={leakage_risks['column_name'].tolist() if not leakage_risks.empty else []}",
        "No final/S05/S06/S07/source-outcome descriptor is used as a predictor",
    )
    predictor_features = set(feature_catalog[feature_catalog["predictor_role"] == "allowed_predictor"]["feature_name"])
    add(
        "feature_catalog_matches_predictor_matrix",
        bool(set(NUMERIC_FEATURES + CATEGORICAL_FEATURES).issubset(predictor_features)),
        f"predictor_feature_count={len(predictor_features)}",
        "All model predictor columns are cataloged with leakage review notes",
    )
    heldout_ok = bool(
        not performance.empty
        and "train_test_group_overlap_count" in performance.columns
        and int(performance["train_test_group_overlap_count"].max()) == 0
        and performance["split_index"].nunique() == config.split_count
    )
    add(
        "group_heldout_validation_used",
        heldout_ok,
        f"splits={performance['split_index'].nunique() if not performance.empty else 0}; max_group_overlap={performance['train_test_group_overlap_count'].max() if not performance.empty else 'n/a'}",
        "Grouped held-out validation has disjoint train/test condition groups for every split",
    )
    add(
        "classification_and_regression_performance_reported",
        bool({"classification", "regression"}.issubset(set(performance["target_type"].astype(str))) if not performance.empty else False),
        f"target_types={sorted(performance['target_type'].dropna().astype(str).unique().tolist()) if not performance.empty else []}",
        "Performance is reported for final-state classification and continuous outcome regression",
    )
    add(
        "heldout_predictions_written",
        bool(not predictions.empty and predictions["target_name"].nunique() >= 2),
        f"prediction_rows={len(predictions)} targets={predictions['target_name'].nunique() if not predictions.empty else 0}",
        "Held-out prediction rows are written for classification and regression targets",
    )
    s06_bad_features = [feature for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES if "s06" in feature.lower() or "dominance" in feature.lower()]
    add(
        "s06_dominance_context_kept_separate",
        bool(not s06_annotations.empty and not s06_bad_features),
        f"s06_annotation_rows={len(s06_annotations)}; s06_or_dominance_predictors={s06_bad_features}",
        "S06 dominance/context annotations are written separately and are not predictor features",
    )
    add(
        "causal_screen_cautious_scope",
        bool(not causal_screen.empty and set(causal_screen["claim_scope"]) == {"exploratory_association_not_causal_proof"}),
        f"rows={len(causal_screen)} claim_scopes={sorted(causal_screen['claim_scope'].unique().tolist()) if not causal_screen.empty else []}",
        "Causal-style output is labeled exploratory association, not causal proof",
    )
    add(
        "model_artifacts_and_figure_written",
        bool(model_artifact_count >= 2 and figure_written),
        f"model_artifact_count={model_artifact_count}; figure_written={figure_written}",
        "Reusable model artifacts and feature-importance figure are written",
    )
    add("unit_tests_passed", unit_tests_success, unit_tests_success, "Focused S13 unit tests pass")
    return pd.DataFrame(rows)
