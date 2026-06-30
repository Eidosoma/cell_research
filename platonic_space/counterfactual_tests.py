from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .behavior_corpus import BEHAVIOR_CLAIM_BOUNDARY
from .behavior_predictor import (
    DEFAULT_TARGET_COLUMNS,
    SPLIT_GROUP_COLUMNS,
    _feature_dicts,
    fit_sparse_ridge_matrix,
    group_mean_predictions,
)
from .world_schema import compact_json, sha256_path


COUNTERFACTUAL_SCHEMA_VERSION = "e07_s11_counterfactual_tests.v1"
COUNTERFACTUAL_MODEL_VERSION = "e07_s11_split_specific_sparse_ridge_replay.v1"
COUNTERFACTUAL_CLAIM_BOUNDARY = (
    "Empirical computational counterfactual tests over S04 simulation-derived behavior records, S05 predictor "
    "features/error limits, S06-S08 embeddings/distances, and S10 classes used only for stratified coverage. "
    "These predictions and replays are bounded simulator-proxy evidence, not causal invariant laws, biological "
    "validation, clinical advice, cognition, agency, sentience, or evidence about living chimeras."
)
CLASS_STRATIFICATION_NOTE = (
    "S10 classes are used only as stratification and coverage aids for candidate selection. They are not treated "
    "as causal invariants or as proof of universality."
)

HELDOUT_SPLITS = ("heldout_policy", "heldout_world", "heldout_goal", "heldout_perturbation")
SCENARIO_COLUMNS = (
    "sourceExperimentId",
    "sourceStepId",
    "sourceTableName",
    "sourceKind",
    "worldId",
    "abstractGoalId",
    "primaryPolicyId",
    "perturbationHash",
    "schedulerHash",
)
PREDICTION_METADATA_COLUMNS = (
    "behaviorRecordId",
    *SCENARIO_COLUMNS,
    "worldFamily",
    "substrateClass",
    "goalFamily",
    "goalKind",
    "policyFamily",
    "policyKind",
    "policyRepresentationType",
    "sourceRowIndex",
)

DEFAULT_TEST_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "counterfactualFamily": "high_delayed_gratification",
        "target": "delayed_gratification",
        "direction": "max",
        "allowedSplits": ("heldout_policy", "heldout_perturbation"),
        "competenceAxis": "delayed gratification",
    },
    {
        "counterfactualFamily": "high_aggregation",
        "target": "aggregation",
        "direction": "max",
        "allowedSplits": ("heldout_policy", "heldout_world", "heldout_perturbation"),
        "competenceAxis": "aggregation",
    },
    {
        "counterfactualFamily": "high_completion",
        "target": "completed",
        "direction": "max",
        "allowedSplits": HELDOUT_SPLITS,
        "competenceAxis": "robust completion",
    },
    {
        "counterfactualFamily": "high_repair_success",
        "target": "repair_success",
        "direction": "max",
        "allowedSplits": ("heldout_policy", "heldout_world", "heldout_perturbation"),
        "competenceAxis": "repair robustness",
    },
    {
        "counterfactualFamily": "high_error_reduction",
        "target": "error_reduction",
        "direction": "max",
        "allowedSplits": HELDOUT_SPLITS,
        "competenceAxis": "morphology error reduction",
    },
    {
        "counterfactualFamily": "high_compatibility",
        "target": "compatibility",
        "direction": "max",
        "allowedSplits": ("heldout_policy", "heldout_world", "heldout_perturbation"),
        "competenceAxis": "policy-goal compatibility",
    },
    {
        "counterfactualFamily": "high_final_sortedness",
        "target": "final_sortedness_percent",
        "direction": "max",
        "allowedSplits": HELDOUT_SPLITS,
        "competenceAxis": "sorting competence",
    },
    {
        "counterfactualFamily": "low_swap_count",
        "target": "swap_count",
        "direction": "min",
        "allowedSplits": HELDOUT_SPLITS,
        "competenceAxis": "low work cost",
    },
    {
        "counterfactualFamily": "low_activation_count",
        "target": "activation_count",
        "direction": "min",
        "allowedSplits": ("heldout_policy", "heldout_world", "heldout_perturbation"),
        "competenceAxis": "low activation cost",
    },
    {
        "counterfactualFamily": "unusual_transfer_final_sortedness",
        "target": "final_sortedness_percent",
        "direction": "max",
        "allowedSplits": ("heldout_world",),
        "competenceAxis": "cross-world transfer",
    },
)


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    return text if text else default


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _stable_id(prefix: str, payload: Mapping[str, Any], length: int = 18) -> str:
    digest = hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def _ci95(values: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if len(arr) == 0:
        return math.nan, math.nan, math.nan
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, math.nan, math.nan
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    delta = 1.96 * se
    return mean, float(mean - delta), float(mean + delta)


def _metrics_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"rmse": math.nan, "mae": math.nan, "bias": math.nan, "predictionCorrelation": math.nan}
    residual = y_pred - y_true
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    else:
        corr = math.nan
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(residual)),
        "predictionCorrelation": corr,
    }


def error_limit_table(error_limits: pd.DataFrame) -> pd.DataFrame:
    """Return one S05 error-limit row per held-out target/split pair with fallback target medians."""

    if error_limits.empty:
        return pd.DataFrame(columns=["target", "splitName", "rmse", "mae", "absErrorP90", "absErrorP95", "errorLimitSource"])
    test = error_limits[error_limits["partition"].astype(str).eq("test")].copy()
    if test.empty:
        test = error_limits.copy()
    cols = ["target", "splitName", "n", "rmse", "mae", "absErrorP50", "absErrorP90", "absErrorP95", "predictionCorrelation"]
    direct = test[[column for column in cols if column in test.columns]].copy()
    direct["errorLimitSource"] = "S05_test_target_split"

    medians = []
    for target, subset in test.groupby("target", dropna=False, sort=True):
        row: dict[str, Any] = {"target": target, "splitName": "__target_median__", "n": int(subset.get("n", pd.Series(dtype=float)).sum())}
        for column in ("rmse", "mae", "absErrorP50", "absErrorP90", "absErrorP95", "predictionCorrelation"):
            if column in subset.columns:
                row[column] = float(pd.to_numeric(subset[column], errors="coerce").median())
        row["errorLimitSource"] = "S05_test_target_median"
        medians.append(row)
    return pd.concat([direct, pd.DataFrame(medians)], ignore_index=True, sort=False)


def _lookup_error_limits(error_limits: pd.DataFrame, target: str, split_name: str) -> dict[str, Any]:
    if error_limits.empty:
        return {}
    exact = error_limits[error_limits["target"].astype(str).eq(target) & error_limits["splitName"].astype(str).eq(split_name)]
    if exact.empty:
        exact = error_limits[error_limits["target"].astype(str).eq(target) & error_limits["splitName"].astype(str).eq("__target_median__")]
    if exact.empty:
        return {}
    return exact.iloc[0].to_dict()


def evaluated_target_splits(metrics: pd.DataFrame, target_columns: Sequence[str], split_names: Sequence[str]) -> set[tuple[str, str]]:
    if metrics.empty:
        return {(target, split) for target in target_columns for split in split_names}
    rows = metrics[
        metrics["target"].astype(str).isin([str(target) for target in target_columns])
        & metrics["splitName"].astype(str).isin([str(split) for split in split_names])
        & metrics["partition"].astype(str).eq("test")
        & metrics["modelName"].astype(str).eq("sparse_ridge")
        & metrics["status"].astype(str).eq("evaluated")
    ]
    return {(str(row.target), str(row.splitName)) for row in rows.itertuples(index=False)}


def score_heldout_predictions(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target_columns: Sequence[str] | None = None,
    split_names: Sequence[str] = HELDOUT_SPLITS,
    min_train_rows: int = 200,
    min_test_rows: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train split-specific S05-style sparse-ridge models and predict excluded test rows."""

    target_columns = list(target_columns or DEFAULT_TARGET_COLUMNS)
    frame = frame.reset_index(drop=True).copy()
    feature_vectorizer = DictVectorizer(sparse=True)
    feature_matrix = feature_vectorizer.fit_transform(_feature_dicts(frame))
    eligible = evaluated_target_splits(metrics, target_columns, split_names)
    if not eligible:
        eligible = {(target, split) for target in target_columns for split in split_names}

    prediction_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    metric_lookup = metrics[
        metrics.get("modelName", pd.Series(dtype=str)).astype(str).eq("sparse_ridge")
        & metrics.get("partition", pd.Series(dtype=str)).astype(str).eq("test")
    ].copy()

    for split_name in split_names:
        if split_name not in SPLIT_GROUP_COLUMNS:
            continue
        partition_col = f"{split_name}Partition"
        group_col = SPLIT_GROUP_COLUMNS[split_name]
        if partition_col not in frame.columns or group_col not in frame.columns:
            continue
        train_groups_all = set(frame.loc[frame[partition_col].eq("train"), group_col].astype(str))
        for target in target_columns:
            if (target, split_name) not in eligible or target not in frame.columns:
                continue
            target_frame = frame[frame[target].notna()].copy()
            train = target_frame[target_frame[partition_col].eq("train")]
            test = target_frame[target_frame[partition_col].eq("test")]
            metric_row = metric_lookup[
                metric_lookup["target"].astype(str).eq(target) & metric_lookup["splitName"].astype(str).eq(split_name)
            ]
            alpha = _finite(metric_row["bestAlpha"].iloc[0], 1.0) if not metric_row.empty else 1.0
            status = "evaluated"
            reason = ""
            if len(train) < min_train_rows or len(test) < min_test_rows:
                status = "skipped_insufficient_rows"
                reason = f"train={len(train)}, test={len(test)}"
                diagnostic_rows.append(
                    {
                        "target": target,
                        "splitName": split_name,
                        "status": status,
                        "reason": reason,
                        "trainRows": int(len(train)),
                        "testRows": int(len(test)),
                        "bestAlpha": alpha,
                    }
                )
                continue

            model = fit_sparse_ridge_matrix(
                feature_matrix[train.index.to_numpy()],
                train[target].to_numpy(dtype=float),
                alpha,
            )
            x_test = feature_matrix[test.index.to_numpy()]
            y_true = test[target].to_numpy(dtype=float)
            predictions = model.predict(x_test)
            global_predictions = np.full(len(test), float(train[target].mean()))
            group_predictions = group_mean_predictions(train, test, target)
            summary = _metrics_summary(y_true, predictions)
            diagnostic_rows.append(
                {
                    "target": target,
                    "splitName": split_name,
                    "status": status,
                    "reason": reason,
                    "trainRows": int(len(train)),
                    "testRows": int(len(test)),
                    "bestAlpha": alpha,
                    **summary,
                }
            )

            out_cols = [column for column in PREDICTION_METADATA_COLUMNS if column in test.columns]
            out = test[out_cols].copy()
            out["target"] = target
            out["splitName"] = split_name
            out["partition"] = "test"
            out["prediction"] = predictions.astype(float)
            out["observed"] = y_true.astype(float)
            out["globalMeanPrediction"] = global_predictions.astype(float)
            out["sourceGroupMeanPrediction"] = group_predictions.astype(float)
            out["excludedGroupColumn"] = group_col
            out["excludedGroupValue"] = test[group_col].astype(str).to_numpy()
            out["trainContainsExcludedGroup"] = out["excludedGroupValue"].map(lambda value: str(value) in train_groups_all)
            out["modelTrainingRows"] = int(len(train))
            out["modelTestRows"] = int(len(test))
            out["bestAlpha"] = alpha
            prediction_frames.append(out)

    predictions_df = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    diagnostics = pd.DataFrame(diagnostic_rows)
    return predictions_df, diagnostics


def attach_universality_classes(predictions: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    class_cols = [
        "abstractPolicyId",
        "universalityClassId",
        "className",
        "classLabel",
        "classAssignmentConfidence",
        "distanceUncertaintyLevel",
        "sparseCoveragePenalty",
    ]
    available_cols = [column for column in class_cols if column in classes.columns]
    if not available_cols:
        out = predictions.copy()
    else:
        out = predictions.merge(
            classes[available_cols].drop_duplicates("abstractPolicyId"),
            left_on="primaryPolicyId",
            right_on="abstractPolicyId",
            how="left",
        )
    out["universalityClassId"] = out.get("universalityClassId", pd.Series(index=out.index, dtype=object)).fillna("S10_UNASSIGNED")
    out["classLabel"] = out.get("classLabel", pd.Series(index=out.index, dtype=object)).fillna("unassigned")
    out["classUse"] = "stratification_only"
    out["classCausalInvariantAssumption"] = False
    out["classStratificationNote"] = CLASS_STRATIFICATION_NOTE
    return out


def _aggregate_candidate_group(group: pd.DataFrame, spec: Mapping[str, Any]) -> dict[str, Any]:
    first = group.iloc[0]
    observed_values = group["observed"].astype(float).tolist()
    observed_mean, observed_ci_low, observed_ci_high = _ci95(observed_values)
    source_experiments = sorted(group["sourceExperimentId"].astype(str).dropna().unique().tolist())
    row: dict[str, Any] = {
        "counterfactualFamily": spec["counterfactualFamily"],
        "target": spec["target"],
        "direction": spec["direction"],
        "competenceAxis": spec["competenceAxis"],
        "splitName": first["splitName"],
        "partition": "test",
        "predictionMean": float(group["prediction"].mean()),
        "predictionMin": float(group["prediction"].min()),
        "predictionMax": float(group["prediction"].max()),
        "observedMean": observed_mean,
        "observedCi95Lower": observed_ci_low,
        "observedCi95Upper": observed_ci_high,
        "observedStd": float(np.std(observed_values, ddof=1)) if len(observed_values) > 1 else math.nan,
        "observedRecordCount": int(len(group)),
        "globalMeanPredictionMean": float(group["globalMeanPrediction"].mean()),
        "sourceGroupMeanPredictionMean": float(group["sourceGroupMeanPrediction"].mean()),
        "behaviorRecordIdsJson": group["behaviorRecordId"].astype(str).dropna().unique().tolist(),
        "sourceExperimentIdsJson": source_experiments,
        "sourceRowIndicesJson": [int(value) for value in pd.to_numeric(group.get("sourceRowIndex", pd.Series(dtype=int)), errors="coerce").dropna().tolist()],
    }
    for column in SCENARIO_COLUMNS:
        row[column] = first.get(column)
    for column in (
        "worldFamily",
        "substrateClass",
        "goalFamily",
        "goalKind",
        "policyFamily",
        "policyKind",
        "policyRepresentationType",
        "universalityClassId",
        "className",
        "classLabel",
        "classAssignmentConfidence",
        "distanceUncertaintyLevel",
        "sparseCoveragePenalty",
        "classUse",
        "classCausalInvariantAssumption",
        "classStratificationNote",
        "excludedGroupColumn",
        "excludedGroupValue",
        "modelTrainingRows",
        "modelTestRows",
        "bestAlpha",
    ):
        if column in group.columns:
            row[column] = first.get(column)
    score = row["predictionMean"] if spec["direction"] == "max" else -row["predictionMean"]
    row["selectionScore"] = float(score)
    row["candidateId"] = _stable_id(
        "S11_CF",
        {
            "family": row["counterfactualFamily"],
            "target": row["target"],
            "splitName": row["splitName"],
            "scenario": {column: row.get(column) for column in SCENARIO_COLUMNS},
        },
    )
    return row


def select_counterfactual_candidates(
    predictions: pd.DataFrame,
    test_families: Sequence[Mapping[str, Any]] = DEFAULT_TEST_FAMILIES,
    max_candidates_per_spec: int = 24,
    max_candidates_per_class: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidate_rows: list[dict[str, Any]] = []
    selection_diagnostics: list[dict[str, Any]] = []
    group_cols = [*SCENARIO_COLUMNS, "target", "splitName"]

    for spec in test_families:
        target = str(spec["target"])
        allowed_splits = {str(split) for split in spec.get("allowedSplits", HELDOUT_SPLITS)}
        subset = predictions[
            predictions["target"].astype(str).eq(target)
            & predictions["splitName"].astype(str).isin(allowed_splits)
            & predictions["observed"].notna()
            & (~predictions["trainContainsExcludedGroup"].astype(bool))
        ].copy()
        for split_name, split_subset in subset.groupby("splitName", sort=True):
            grouped_rows = [_aggregate_candidate_group(group, spec) for _, group in split_subset.groupby(group_cols, dropna=False, sort=False)]
            grouped = pd.DataFrame(grouped_rows)
            if grouped.empty:
                selection_diagnostics.append(
                    {
                        "counterfactualFamily": spec["counterfactualFamily"],
                        "target": target,
                        "splitName": split_name,
                        "eligibleScenarioCount": 0,
                        "selectedScenarioCount": 0,
                        "selectedClassCount": 0,
                        "selectionRule": "top prediction by S10 class strata",
                    }
                )
                continue
            grouped = grouped.sort_values(["universalityClassId", "selectionScore"], ascending=[True, False])
            stratified = grouped.groupby("universalityClassId", group_keys=False, sort=True).head(max_candidates_per_class)
            stratified = stratified.sort_values("selectionScore", ascending=False).head(max_candidates_per_spec).copy()
            stratified["selectionRankWithinSpec"] = np.arange(1, len(stratified) + 1)
            stratified["selectionRule"] = (
                f"Top {max_candidates_per_class} per S10 class, capped at {max_candidates_per_spec}; classes used as strata only."
            )
            candidate_rows.extend(stratified.to_dict(orient="records"))
            selection_diagnostics.append(
                {
                    "counterfactualFamily": spec["counterfactualFamily"],
                    "target": target,
                    "splitName": split_name,
                    "eligibleScenarioCount": int(len(grouped)),
                    "selectedScenarioCount": int(len(stratified)),
                    "selectedClassCount": int(stratified["universalityClassId"].nunique()),
                    "selectionRule": "top prediction by S10 class strata",
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.drop_duplicates("candidateId").sort_values(
            ["counterfactualFamily", "splitName", "selectionRankWithinSpec", "candidateId"]
        )
    return candidates.reset_index(drop=True), pd.DataFrame(selection_diagnostics)


def _behavior_ids(value: Any) -> list[str]:
    parsed = _json_loads(value, [])
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(value, str):
        return [value]
    return []


def _raw_trace_link_count(value: Any) -> tuple[int, int]:
    parsed = _json_loads(value, [])
    if not isinstance(parsed, list):
        return 0, 0
    total = len(parsed)
    available = 0
    for item in parsed:
        if not isinstance(item, Mapping):
            continue
        if bool(item.get("exists")):
            available += 1
            continue
        for key in ("resolvedPath", "path", "value"):
            path = item.get(key)
            if path and Path(str(path)).exists():
                available += 1
                break
    return total, available


def replay_records_for_candidates(candidates: pd.DataFrame, s04_corpus: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    selected_ids = sorted({record_id for value in candidates["behaviorRecordIdsJson"] for record_id in _behavior_ids(value)})
    s04 = s04_corpus[s04_corpus["behaviorRecordId"].astype(str).isin(selected_ids)].copy()
    if s04.empty:
        return pd.DataFrame()

    table_status: dict[str, dict[str, Any]] = {}
    for path_text, subset in s04.groupby("sourceTablePath", dropna=False):
        path = Path(str(path_text))
        exists = path.exists()
        readable = False
        row_count = math.nan
        sha_matches = False
        actual_sha = None
        if exists and path.is_file():
            actual_sha = sha256_path(path)
            expected = str(subset["sourceTableSha256"].dropna().iloc[0]) if subset["sourceTableSha256"].notna().any() else ""
            sha_matches = actual_sha == expected
            try:
                if path.suffix.lower() == ".parquet":
                    source_df = pd.read_parquet(path)
                elif path.suffix.lower() == ".csv":
                    source_df = pd.read_csv(path)
                else:
                    source_df = pd.DataFrame()
                row_count = int(len(source_df))
                readable = True
            except Exception:
                readable = False
                row_count = math.nan
        table_status[str(path_text)] = {
            "sourceTablePath": str(path_text),
            "sourceTableExists": bool(exists),
            "sourceTableReadable": bool(readable),
            "sourceTableShaMatches": bool(sha_matches),
            "sourceTableActualSha256": actual_sha,
            "sourceTableRowCount": row_count,
        }

    rows: list[dict[str, Any]] = []
    for _, row in s04.iterrows():
        status = table_status.get(str(row.get("sourceTablePath")), {})
        row_count = _finite(status.get("sourceTableRowCount"), math.nan)
        source_index = int(row.get("sourceRowIndex")) if pd.notna(row.get("sourceRowIndex")) else -1
        in_bounds = bool(math.isfinite(row_count) and 0 <= source_index < int(row_count))
        raw_total, raw_available = _raw_trace_link_count(row.get("rawTraceLinksJson"))
        rows.append(
            {
                "behaviorRecordId": str(row.get("behaviorRecordId")),
                "sourceExperimentId": row.get("sourceExperimentId"),
                "sourceStepId": row.get("sourceStepId"),
                "sourceTableName": row.get("sourceTableName"),
                "sourceTablePath": row.get("sourceTablePath"),
                "sourceTableSha256": row.get("sourceTableSha256"),
                "sourceRowIndex": source_index,
                "sourceRowHash": row.get("sourceRowHash"),
                "sourceTableExists": bool(status.get("sourceTableExists")),
                "sourceTableReadable": bool(status.get("sourceTableReadable")),
                "sourceTableShaMatches": bool(status.get("sourceTableShaMatches")),
                "sourceRowIndexInBounds": in_bounds,
                "sourceTableVerified": bool(status.get("sourceTableExists") and status.get("sourceTableReadable") and status.get("sourceTableShaMatches") and in_bounds),
                "rawTraceLinkCount": int(raw_total),
                "availableRawTraceLinkCount": int(raw_available),
                "behaviorVectorJson": row.get("behaviorVectorJson"),
                "metricValuesJson": row.get("metricValuesJson"),
                "trajectorySummaryJson": row.get("trajectorySummaryJson"),
                "rawTraceLinksJson": row.get("rawTraceLinksJson"),
            }
        )
    return pd.DataFrame(rows)


def validate_candidates(
    candidates: pd.DataFrame,
    replay_records: pd.DataFrame,
    error_limits: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    replay_by_id = replay_records.set_index("behaviorRecordId", drop=False) if not replay_records.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        ids = _behavior_ids(candidate.get("behaviorRecordIdsJson"))
        replay_subset = replay_by_id.loc[replay_by_id.index.intersection(ids)] if not replay_by_id.empty else pd.DataFrame()
        limits = _lookup_error_limits(error_limits, str(candidate["target"]), str(candidate["splitName"]))
        p90 = _finite(limits.get("absErrorP90"), math.nan)
        p95 = _finite(limits.get("absErrorP95"), math.nan)
        rmse = _finite(limits.get("rmse"), math.nan)
        prediction = _finite(candidate.get("predictionMean"), math.nan)
        observed = _finite(candidate.get("observedMean"), math.nan)
        abs_error = abs(prediction - observed) if math.isfinite(prediction) and math.isfinite(observed) else math.nan
        verified_count = int(replay_subset["sourceTableVerified"].sum()) if not replay_subset.empty else 0
        raw_trace_records = int((replay_subset.get("rawTraceLinkCount", pd.Series(dtype=int)).fillna(0).astype(int) > 0).sum()) if not replay_subset.empty else 0
        available_raw_records = (
            int((replay_subset.get("availableRawTraceLinkCount", pd.Series(dtype=int)).fillna(0).astype(int) > 0).sum())
            if not replay_subset.empty
            else 0
        )
        mode = "s04_behavior_vector_only"
        if verified_count == len(ids) and raw_trace_records > 0:
            mode = "source_table_row_replay_plus_raw_trace_links"
        elif verified_count == len(ids) and ids:
            mode = "source_table_row_replay"
        elif verified_count > 0:
            mode = "partial_source_table_row_replay"
        rows.append(
            {
                **candidate.to_dict(),
                "directValidationMode": mode,
                "directReplayRecordCount": int(len(ids)),
                "sourceTableVerifiedRecordCount": verified_count,
                "sourceTableVerifiedFraction": float(verified_count / len(ids)) if ids else 0.0,
                "rawTraceLinkedRecordCount": raw_trace_records,
                "availableRawTraceLinkedRecordCount": available_raw_records,
                "sourceTablePathsJson": sorted(replay_subset["sourceTablePath"].astype(str).dropna().unique().tolist()) if not replay_subset.empty else [],
                "sourceRowHashesJson": sorted(replay_subset["sourceRowHash"].astype(str).dropna().unique().tolist()) if not replay_subset.empty else [],
                "predictionError": float(prediction - observed) if math.isfinite(prediction) and math.isfinite(observed) else math.nan,
                "absoluteError": float(abs_error) if math.isfinite(abs_error) else math.nan,
                "s05RmseLimit": rmse,
                "s05AbsErrorP90": p90,
                "s05AbsErrorP95": p95,
                "predictionInterval95Lower": float(prediction - p95) if math.isfinite(prediction) and math.isfinite(p95) else math.nan,
                "predictionInterval95Upper": float(prediction + p95) if math.isfinite(prediction) and math.isfinite(p95) else math.nan,
                "observedWithinS05P90": bool(math.isfinite(abs_error) and math.isfinite(p90) and abs_error <= p90),
                "observedWithinS05P95": bool(math.isfinite(abs_error) and math.isfinite(p95) and abs_error <= p95),
                "globalMeanAbsoluteError": abs(_finite(candidate.get("globalMeanPredictionMean"), math.nan) - observed)
                if math.isfinite(observed) and math.isfinite(_finite(candidate.get("globalMeanPredictionMean"), math.nan))
                else math.nan,
                "sourceGroupMeanAbsoluteError": abs(_finite(candidate.get("sourceGroupMeanPredictionMean"), math.nan) - observed)
                if math.isfinite(observed) and math.isfinite(_finite(candidate.get("sourceGroupMeanPredictionMean"), math.nan))
                else math.nan,
                "intervalCaveat": "single observed record; observed CI unavailable"
                if int(candidate.get("observedRecordCount", 0)) <= 1
                else "observed CI uses normal approximation across replay rows",
            }
        )
    return pd.DataFrame(rows)


def target_performance_summary(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["counterfactualFamily", "target", "splitName", "direction", "competenceAxis"]
    for keys, subset in validation.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, keys, strict=False))
        abs_error = pd.to_numeric(subset["absoluteError"], errors="coerce")
        global_abs = pd.to_numeric(subset["globalMeanAbsoluteError"], errors="coerce")
        source_abs = pd.to_numeric(subset["sourceGroupMeanAbsoluteError"], errors="coerce")
        row.update(
            {
                "candidateCount": int(len(subset)),
                "sourceExperimentCount": int(len({item for values in subset["sourceExperimentIdsJson"] for item in _json_loads(values, []) or []})),
                "classCount": int(subset["universalityClassId"].nunique()),
                "meanPredicted": float(pd.to_numeric(subset["predictionMean"], errors="coerce").mean()),
                "meanObserved": float(pd.to_numeric(subset["observedMean"], errors="coerce").mean()),
                "mae": float(abs_error.mean()),
                "rmse": float(math.sqrt(np.nanmean(pd.to_numeric(subset["predictionError"], errors="coerce").to_numpy(dtype=float) ** 2))),
                "medianAbsoluteError": float(abs_error.median()),
                "withinS05P90Fraction": float(subset["observedWithinS05P90"].mean()),
                "withinS05P95Fraction": float(subset["observedWithinS05P95"].mean()),
                "sourceTableVerifiedFraction": float(subset["sourceTableVerifiedFraction"].mean()),
                "modelBeatsGlobalMeanFraction": float((abs_error < global_abs).mean()) if global_abs.notna().any() else math.nan,
                "modelBeatsSourceGroupMeanFraction": float((abs_error < source_abs).mean()) if source_abs.notna().any() else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def class_coverage_summary(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (class_id, class_label), subset in validation.groupby(["universalityClassId", "classLabel"], dropna=False, sort=True):
        rows.append(
            {
                "universalityClassId": class_id,
                "classLabel": class_label,
                "candidateCount": int(len(subset)),
                "targetCount": int(subset["target"].nunique()),
                "splitCount": int(subset["splitName"].nunique()),
                "sourceExperimentCount": int(len({item for values in subset["sourceExperimentIdsJson"] for item in _json_loads(values, []) or []})),
                "meanAbsoluteError": float(pd.to_numeric(subset["absoluteError"], errors="coerce").mean()),
                "withinS05P95Fraction": float(subset["observedWithinS05P95"].mean()),
                "sourceTableVerifiedFraction": float(subset["sourceTableVerifiedFraction"].mean()),
                "classUse": "stratification_only",
            }
        )
    return pd.DataFrame(rows)


def validation_checks(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    validation: pd.DataFrame,
    replay_records: pd.DataFrame,
    diagnostics: pd.DataFrame,
    min_evaluated_models: int = 8,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(scope: str, check: str, success: bool, severity: str = "error", detail: str = "") -> None:
        checks.append({"scope": scope, "check": check, "success": bool(success), "severity": severity, "detail": detail})

    add("predictions", "heldout_prediction_rows_present", len(predictions) > 0, detail=str(len(predictions)))
    add("candidates", "counterfactual_candidates_selected", len(candidates) > 0, detail=str(len(candidates)))
    add("candidates", "all_candidates_are_test_partition", bool((candidates.get("partition", pd.Series(dtype=str)).astype(str) == "test").all()) if not candidates.empty else False)
    add(
        "candidates",
        "no_candidate_group_seen_in_training_partition",
        bool((~predictions.get("trainContainsExcludedGroup", pd.Series(dtype=bool)).astype(bool)).all()) if not predictions.empty else False,
        detail="checked row-level split group membership",
    )
    add(
        "classes",
        "s10_classes_marked_stratification_only",
        bool((candidates.get("classUse", pd.Series(dtype=str)).astype(str) == "stratification_only").all())
        and bool((candidates.get("classCausalInvariantAssumption", pd.Series(dtype=bool)).astype(bool) == False).all())
        if not candidates.empty
        else False,
        detail=CLASS_STRATIFICATION_NOTE,
    )
    add("validation", "direct_validation_rows_present", len(validation) == len(candidates) and len(validation) > 0, detail=f"{len(validation)} validation rows")
    verified_fraction = float(validation["sourceTableVerifiedFraction"].mean()) if not validation.empty else 0.0
    add("replay", "source_table_replay_verified_for_all_candidates", verified_fraction >= 0.95, detail=f"mean verified fraction {verified_fraction:.4f}")
    has_limits = (
        validation["s05AbsErrorP95"].notna().all()
        and validation["predictionInterval95Lower"].notna().all()
        and validation["predictionInterval95Upper"].notna().all()
        if not validation.empty
        else False
    )
    add("intervals", "s05_prediction_intervals_available", bool(has_limits))
    evaluated_models = diagnostics[diagnostics["status"].astype(str).eq("evaluated")] if not diagnostics.empty else pd.DataFrame()
    add(
        "modeling",
        "split_specific_models_evaluated",
        len(evaluated_models) >= min_evaluated_models,
        detail=f"{len(evaluated_models)} evaluated, minimum {min_evaluated_models}",
    )

    if not validation.empty:
        p95_fraction = float(validation["observedWithinS05P95"].mean())
        add(
            "intervals",
            "most_candidates_within_s05_p95_error_limit",
            p95_fraction >= 0.50,
            severity="warning",
            detail=f"within-S05-P95 fraction {p95_fraction:.4f}",
        )
        single_count = int((pd.to_numeric(validation["observedRecordCount"], errors="coerce") <= 1).sum())
        add(
            "replicates",
            "some_candidates_have_single_observed_replay_record",
            single_count == 0,
            severity="warning",
            detail=f"{single_count} candidates have n<=1; prediction interval still uses S05 held-out error limits",
        )
    if not replay_records.empty:
        add(
            "replay",
            "s04_records_have_source_table_links",
            bool(replay_records["sourceTablePath"].astype(str).ne("").all()),
            severity="warning",
            detail=f"{len(replay_records)} replay rows checked",
        )
    return pd.DataFrame(checks)


def performance_summary(validation: pd.DataFrame, target_summary: pd.DataFrame, checks: pd.DataFrame) -> dict[str, Any]:
    hard_failures = checks[(checks["severity"].eq("error")) & (~checks["success"])] if not checks.empty else pd.DataFrame()
    warnings = checks[(checks["severity"].eq("warning")) & (~checks["success"])] if not checks.empty else pd.DataFrame()
    if validation.empty:
        return {
            "candidateCount": 0,
            "targetCount": 0,
            "splitCount": 0,
            "classCount": 0,
            "overallWithinS05P90Fraction": math.nan,
            "overallWithinS05P95Fraction": math.nan,
            "overallMeanAbsoluteError": math.nan,
            "overallSourceTableVerifiedFraction": math.nan,
            "targetFamiliesBeatingGlobalMeanCount": 0,
            "targetFamilyCount": 0,
            "hardValidationFailureCount": int(len(hard_failures)),
            "warningValidationFailureCount": int(len(warnings)),
        }
    abs_error = pd.to_numeric(validation["absoluteError"], errors="coerce")
    return {
        "candidateCount": int(len(validation)),
        "targetCount": int(validation["target"].nunique()),
        "splitCount": int(validation["splitName"].nunique()),
        "classCount": int(validation["universalityClassId"].nunique()),
        "overallWithinS05P90Fraction": float(validation["observedWithinS05P90"].mean()),
        "overallWithinS05P95Fraction": float(validation["observedWithinS05P95"].mean()),
        "overallMeanAbsoluteError": float(abs_error.mean()),
        "overallMedianAbsoluteError": float(abs_error.median()),
        "overallSourceTableVerifiedFraction": float(validation["sourceTableVerifiedFraction"].mean()),
        "rawTraceLinkedCandidateFraction": float((validation["rawTraceLinkedRecordCount"] > 0).mean()),
        "targetFamiliesBeatingGlobalMeanCount": int((target_summary.get("modelBeatsGlobalMeanFraction", pd.Series(dtype=float)) > 0.5).sum())
        if not target_summary.empty
        else 0,
        "targetFamiliesBeatingSourceGroupMeanCount": int((target_summary.get("modelBeatsSourceGroupMeanFraction", pd.Series(dtype=float)) > 0.5).sum())
        if not target_summary.empty
        else 0,
        "targetFamilyCount": int(len(target_summary)),
        "hardValidationFailureCount": int(len(hard_failures)),
        "warningValidationFailureCount": int(len(warnings)),
    }


def dataframe_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if column.endswith("Json"):
            out[column] = out[column].map(compact_json)
    return out


def claim_boundary() -> str:
    return f"{COUNTERFACTUAL_CLAIM_BOUNDARY} S04 corpus boundary: {BEHAVIOR_CLAIM_BOUNDARY}"
