from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .behavior_corpus import BEHAVIOR_CORPUS_VERSION, compact_json, json_ready
from .policy_catalog import read_table, safe_json_loads
from .world_schema import sha256_path


PREDICTOR_SCHEMA_VERSION = "e07_s05_behavior_predictor.v1"
PREDICTOR_MODEL_VERSION = "e07_s05_sparse_ridge_behavior_predictor.v1"
PREDICTOR_CLAIM_BOUNDARY = (
    "Computational surrogate model over S04 simulation-derived behavior records only. Predictions are bounded "
    "proxy estimates of simulator metrics and do not validate biological morphogenesis, cognition, agency, "
    "clinical behavior, sentience, or living chimeras."
)

DEFAULT_TARGET_COLUMNS = (
    "initial_sortedness_percent",
    "final_sortedness_percent",
    "final_monotonicity_error",
    "completed",
    "swap_count",
    "comparison_count",
    "activation_count",
    "event_count",
    "blocked_move_attempts",
    "delayed_gratification",
    "aggregation",
    "target_error",
    "error_reduction",
    "repair_success",
    "compatibility",
    "fitness_or_score",
)

SPLIT_GROUP_COLUMNS = {
    "random": "behaviorRecordId",
    "heldout_policy": "primaryPolicyId",
    "heldout_world": "worldId",
    "heldout_goal": "abstractGoalId",
    "heldout_perturbation": "perturbationHash",
}

CAT_FEATURE_COLUMNS = (
    "sourceExperimentId",
    "sourceStepId",
    "sourceTableName",
    "sourceKind",
    "worldId",
    "worldFamily",
    "substrateClass",
    "abstractGoalId",
    "goalFamily",
    "goalKind",
    "goalRepresentationType",
    "primaryPolicyId",
    "policyFamily",
    "policyKind",
    "policyRepresentationType",
    "policyStochasticity",
    "policyResolutionStatus",
    "worldResolutionStatus",
    "goalResolutionStatus",
    "perturbationHash",
    "schedulerHash",
    "exactPolicyLinked",
)

NUMERIC_FEATURE_COLUMNS = (
    "policyComplexityScore",
    "policyMemoryDepth",
    "policySignalingHorizon",
    "policyDslAvailable",
    "policyAutomatonAvailable",
    "policyDecisionGraphAvailable",
    "policyNeuralMetadataAvailable",
    "seedFieldCount",
    "schedulerFieldCount",
    "perturbationFieldCount",
    "candidatePolicyCount",
    "exactPolicyCount",
)


def parse_json(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    parsed = safe_json_loads(value, default)
    return default if parsed is None else parsed


def stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()[:length]


def partition_for_group(value: Any, salt: str) -> str:
    bucket = int(hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def _first_json_item(value: Any) -> str:
    parsed = parse_json(value, [])
    if isinstance(parsed, list) and parsed:
        return str(parsed[0])
    if isinstance(parsed, dict) and parsed:
        return str(next(iter(parsed)))
    return "unknown"


def _json_len(value: Any) -> int:
    parsed = parse_json(value, {})
    if isinstance(parsed, (dict, list)):
        return len(parsed)
    return 0


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _lookup_frame(df: pd.DataFrame, id_column: str, columns: Sequence[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if df.empty or id_column not in df.columns:
        return out
    for _, row in df.iterrows():
        key = str(row[id_column])
        out[key] = {column: row.get(column) for column in columns if column in df.columns}
    return out


def _policy_lookup(policies: pd.DataFrame) -> dict[str, dict[str, Any]]:
    columns = [
        "policyFamily",
        "policyKind",
        "representationType",
        "memoryDepth",
        "signalingHorizon",
        "stochasticity",
        "complexityScore",
        "dslAvailable",
        "automatonAvailable",
        "decisionGraphAvailable",
        "neuralMetadataAvailable",
    ]
    return _lookup_frame(policies, "abstractPolicyId", columns)


def target_coverage(frame: pd.DataFrame, target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in target_columns:
        if target not in frame.columns:
            count = 0
            source_counts: dict[str, int] = {}
        else:
            mask = frame[target].notna()
            count = int(mask.sum())
            source_counts = frame.loc[mask, "sourceExperimentId"].value_counts().sort_index().astype(int).to_dict()
        rows.append(
            {
                "target": target,
                "availableRows": count,
                "coverageFraction": count / len(frame) if len(frame) else 0.0,
                "sourceExperimentCountsJson": source_counts,
                "selectedForTraining": bool(count >= 1000),
            }
        )
    return pd.DataFrame(rows)


def build_modeling_frame(
    corpus_path: str | Path = "/artifacts/data/e07_unified_behavior_corpus.parquet",
    world_catalog_path: str | Path = "/artifacts/research_steps/S01/normalized_world_catalog.parquet",
    policy_catalog_path: str | Path = "/artifacts/research_steps/S02/policy_abstract_catalog.parquet",
    goal_catalog_path: str | Path = "/artifacts/research_steps/S03/goal_catalog.parquet",
    max_rows: int | None = None,
) -> pd.DataFrame:
    corpus = read_table(corpus_path)
    if corpus.empty:
        raise FileNotFoundError(f"S04 unified behavior corpus not found or empty: {corpus_path}")
    if max_rows is not None:
        corpus = corpus.head(max_rows).copy()

    worlds = read_table(world_catalog_path)
    policies = read_table(policy_catalog_path)
    goals = read_table(goal_catalog_path)
    world_lookup = _lookup_frame(worlds, "worldId", ["worldFamily", "substrateClass", "replayability", "metadataCompleteness"])
    policy_lookup = _policy_lookup(policies)
    goal_lookup = _lookup_frame(goals, "abstractGoalId", ["goalFamily", "goalKind", "representationType", "localObservability"])

    rows: list[dict[str, Any]] = []
    for record in corpus.to_dict(orient="records"):
        abstract_policy_ids = parse_json(record.get("abstractPolicyIdsJson"), []) or []
        candidate_policy_ids = parse_json(record.get("candidatePolicyIdsJson"), []) or []
        primary_policy_id = str(abstract_policy_ids[0]) if abstract_policy_ids else (str(candidate_policy_ids[0]) if candidate_policy_ids else "unknown")
        perturbation = parse_json(record.get("perturbationJson"), {}) or {}
        scheduler = parse_json(record.get("schedulerJson"), {}) or {}
        seeds = parse_json(record.get("seedJson"), {}) or {}
        behavior = parse_json(record.get("behaviorVectorJson"), {}) or {}
        world = world_lookup.get(str(record.get("worldId")), {})
        policy = policy_lookup.get(primary_policy_id, {})
        goal = goal_lookup.get(str(record.get("abstractGoalId")), {})

        row: dict[str, Any] = {
            "behaviorRecordId": record.get("behaviorRecordId"),
            "sourceExperimentId": record.get("sourceExperimentId"),
            "sourceStepId": record.get("sourceStepId"),
            "sourceTableName": record.get("sourceTableName"),
            "sourceKind": record.get("sourceKind"),
            "sourceRowIndex": record.get("sourceRowIndex"),
            "worldId": record.get("worldId"),
            "worldFamily": world.get("worldFamily", "unknown"),
            "substrateClass": world.get("substrateClass", "unknown"),
            "abstractGoalId": record.get("abstractGoalId"),
            "goalFamily": goal.get("goalFamily", "unknown"),
            "goalKind": goal.get("goalKind", "unknown"),
            "goalRepresentationType": goal.get("representationType", "unknown"),
            "primaryPolicyId": primary_policy_id,
            "exactPolicyLinked": "true" if bool(abstract_policy_ids) else "false",
            "policyResolutionStatus": record.get("policyResolutionStatus"),
            "worldResolutionStatus": record.get("worldResolutionStatus"),
            "goalResolutionStatus": record.get("goalResolutionStatus"),
            "policyFamily": policy.get("policyFamily", "unknown"),
            "policyKind": policy.get("policyKind", "unknown"),
            "policyRepresentationType": policy.get("representationType", "unknown"),
            "policyStochasticity": policy.get("stochasticity", "unknown"),
            "policyComplexityScore": _float_or_nan(policy.get("complexityScore")),
            "policyMemoryDepth": _float_or_nan(policy.get("memoryDepth")),
            "policySignalingHorizon": _float_or_nan(policy.get("signalingHorizon")),
            "policyDslAvailable": float(bool(policy.get("dslAvailable"))) if "dslAvailable" in policy else 0.0,
            "policyAutomatonAvailable": float(bool(policy.get("automatonAvailable"))) if "automatonAvailable" in policy else 0.0,
            "policyDecisionGraphAvailable": float(bool(policy.get("decisionGraphAvailable"))) if "decisionGraphAvailable" in policy else 0.0,
            "policyNeuralMetadataAvailable": float(bool(policy.get("neuralMetadataAvailable"))) if "neuralMetadataAvailable" in policy else 0.0,
            "candidatePolicyCount": len(candidate_policy_ids),
            "exactPolicyCount": len(abstract_policy_ids),
            "seedFieldCount": _json_len(seeds),
            "schedulerFieldCount": _json_len(scheduler),
            "perturbationFieldCount": _json_len(perturbation),
            "perturbationHash": stable_hash(perturbation),
            "schedulerHash": stable_hash(scheduler),
            "behaviorCorpusVersion": BEHAVIOR_CORPUS_VERSION,
        }
        for target in DEFAULT_TARGET_COLUMNS:
            row[target] = _float_or_nan(behavior.get(target))
        rows.append(row)

    frame = pd.DataFrame(rows)
    for split_name, group_column in SPLIT_GROUP_COLUMNS.items():
        frame[f"{split_name}Partition"] = frame[group_column].map(lambda value, salt=split_name: partition_for_group(value, salt))
    return frame


def selected_targets(coverage: pd.DataFrame, min_rows: int = 1000) -> list[str]:
    selected = coverage[coverage["availableRows"].ge(min_rows)]["target"].astype(str).tolist()
    return [target for target in DEFAULT_TARGET_COLUMNS if target in selected]


def row_to_features(row: Mapping[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for column in CAT_FEATURE_COLUMNS:
        value = row.get(column, "unknown")
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            value = "unknown"
        features[f"{column}={value}"] = 1.0
    for column in NUMERIC_FEATURE_COLUMNS:
        value = _float_or_nan(row.get(column))
        features[column] = 0.0 if math.isnan(value) else float(value)
    return features


def _feature_dicts(frame: pd.DataFrame) -> list[dict[str, float]]:
    return [row_to_features(row) for row in frame.to_dict(orient="records")]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    variance = float(np.var(y_true))
    metrics = {
        "n": int(len(y_true)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(residual)),
        "absErrorP50": float(np.quantile(np.abs(residual), 0.50)),
        "absErrorP90": float(np.quantile(np.abs(residual), 0.90)),
        "absErrorP95": float(np.quantile(np.abs(residual), 0.95)),
        "targetMean": float(np.mean(y_true)),
        "targetStd": float(np.std(y_true)),
    }
    metrics["r2"] = float(r2_score(y_true, y_pred)) if variance > 0 else math.nan
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        design = np.column_stack([y_pred, np.ones(len(y_pred))])
        slope, intercept = np.linalg.lstsq(design, y_true, rcond=None)[0]
        metrics["calibrationSlope"] = float(slope)
        metrics["calibrationIntercept"] = float(intercept)
        metrics["predictionCorrelation"] = float(np.corrcoef(y_pred, y_true)[0, 1])
    else:
        metrics["calibrationSlope"] = math.nan
        metrics["calibrationIntercept"] = math.nan
        metrics["predictionCorrelation"] = math.nan
    if np.nanmin(y_true) >= 0 and np.nanmax(y_true) <= 1:
        clipped = np.clip(y_pred, 0, 1)
        metrics["brierScore"] = float(np.mean((clipped - y_true) ** 2))
        metrics["binaryEce10"] = expected_calibration_error(y_true, clipped, bins=10)
    else:
        metrics["brierScore"] = math.nan
        metrics["binaryEce10"] = math.nan
    return metrics


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1:
            mask = (y_pred >= low) & (y_pred <= high)
        else:
            mask = (y_pred >= low) & (y_pred < high)
        if not np.any(mask):
            continue
        total += float(mask.mean()) * abs(float(np.mean(y_pred[mask])) - float(np.mean(y_true[mask])))
    return total


def group_mean_predictions(train: pd.DataFrame, eval_frame: pd.DataFrame, target: str) -> np.ndarray:
    global_mean = float(train[target].mean())
    by_pair = train.groupby(["sourceExperimentId", "sourceKind"], observed=True)[target].mean().to_dict()
    by_experiment = train.groupby("sourceExperimentId", observed=True)[target].mean().to_dict()
    predictions: list[float] = []
    for _, row in eval_frame.iterrows():
        key = (row["sourceExperimentId"], row["sourceKind"])
        if key in by_pair:
            predictions.append(float(by_pair[key]))
        elif row["sourceExperimentId"] in by_experiment:
            predictions.append(float(by_experiment[row["sourceExperimentId"]]))
        else:
            predictions.append(global_mean)
    return np.asarray(predictions, dtype=float)


def calibration_rows(
    target: str,
    splitName: str,
    partition: str,
    modelName: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 10,
) -> list[dict[str, Any]]:
    if len(y_true) == 0:
        return []
    order = np.argsort(y_pred)
    chunks = np.array_split(order, min(bins, len(order)))
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        pred = y_pred[chunk]
        obs = y_true[chunk]
        rows.append(
            {
                "target": target,
                "splitName": splitName,
                "partition": partition,
                "modelName": modelName,
                "binIndex": index,
                "n": int(len(chunk)),
                "predictionMean": float(np.mean(pred)),
                "observedMean": float(np.mean(obs)),
                "residualMean": float(np.mean(pred - obs)),
                "rmse": float(math.sqrt(mean_squared_error(obs, pred))),
            }
        )
    return rows


def fit_sparse_ridge(train: pd.DataFrame, target: str, alpha: float):
    pipeline = make_pipeline(
        DictVectorizer(sparse=True),
        StandardScaler(with_mean=False),
        Ridge(alpha=alpha, solver="lsqr", tol=1e-3),
    )
    pipeline.fit(_feature_dicts(train), train[target].to_numpy(dtype=float))
    return pipeline


def fit_sparse_ridge_matrix(x_train: Any, y_train: np.ndarray, alpha: float):
    pipeline = make_pipeline(
        StandardScaler(with_mean=False),
        Ridge(alpha=alpha, solver="lsqr", tol=1e-3),
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def train_evaluate_predictor(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
    alphas: Sequence[float] = (1.0,),
    min_train_rows: int = 200,
    min_eval_rows: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    frame = frame.reset_index(drop=True).copy()
    feature_vectorizer = DictVectorizer(sparse=True)
    feature_matrix = feature_vectorizer.fit_transform(_feature_dicts(frame))
    metric_rows: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    residual_samples: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []
    chosen_alphas: dict[str, list[float]] = defaultdict(list)

    for split_name in SPLIT_GROUP_COLUMNS:
        partition_col = f"{split_name}Partition"
        group_col = SPLIT_GROUP_COLUMNS[split_name]
        split_rows.extend(split_manifest_rows(frame, split_name, partition_col, group_col))
        for target in target_columns:
            target_frame = frame[frame[target].notna()].copy()
            train = target_frame[target_frame[partition_col].eq("train")]
            validation = target_frame[target_frame[partition_col].eq("validation")]
            test = target_frame[target_frame[partition_col].eq("test")]
            if len(train) < min_train_rows or len(test) < min_eval_rows:
                metric_rows.append(
                    {
                        "target": target,
                        "splitName": split_name,
                        "partition": "test",
                        "modelName": "sparse_ridge",
                        "status": "skipped_insufficient_rows",
                        "trainRows": int(len(train)),
                        "validationRows": int(len(validation)),
                        "testRows": int(len(test)),
                        "bestAlpha": math.nan,
                    }
                )
                continue

            best_alpha = float(alphas[0])
            best_score = math.inf
            best_model = None
            x_train = feature_matrix[train.index.to_numpy()]
            y_train = train[target].to_numpy(dtype=float)
            for alpha in alphas:
                model = fit_sparse_ridge_matrix(x_train, y_train, float(alpha))
                if len(validation) >= min_eval_rows:
                    pred = model.predict(feature_matrix[validation.index.to_numpy()])
                    score = math.sqrt(mean_squared_error(validation[target].to_numpy(dtype=float), pred))
                else:
                    score = 0.0
                if best_model is None or score < best_score:
                    best_score = score
                    best_alpha = float(alpha)
                    best_model = model
            assert best_model is not None
            chosen_alphas[target].append(best_alpha)

            for partition_name, eval_frame in (("validation", validation), ("test", test)):
                if len(eval_frame) < min_eval_rows:
                    continue
                y = eval_frame[target].to_numpy(dtype=float)
                model_pred = best_model.predict(feature_matrix[eval_frame.index.to_numpy()])
                global_pred = np.full(len(eval_frame), float(train[target].mean()))
                group_pred = group_mean_predictions(train, eval_frame, target)
                models = {
                    "sparse_ridge": model_pred,
                    "global_mean": global_pred,
                    "source_group_mean": group_pred,
                }
                for model_name, pred in models.items():
                    row = {
                        "target": target,
                        "splitName": split_name,
                        "partition": partition_name,
                        "modelName": model_name,
                        "status": "evaluated",
                        "trainRows": int(len(train)),
                        "validationRows": int(len(validation)),
                        "testRows": int(len(test)),
                        "bestAlpha": best_alpha if model_name == "sparse_ridge" else math.nan,
                    }
                    row.update(_metrics(y, pred))
                    metric_rows.append(row)
                    calibration.extend(calibration_rows(target, split_name, partition_name, model_name, y, pred))
                sample = eval_frame[
                    [
                        "behaviorRecordId",
                        "sourceExperimentId",
                        "sourceTableName",
                        "worldId",
                        "abstractGoalId",
                        "primaryPolicyId",
                        "perturbationHash",
                        target,
                    ]
                ].copy()
                sample["target"] = target
                sample["splitName"] = split_name
                sample["partition"] = partition_name
                sample["prediction"] = model_pred
                sample["residual"] = model_pred - y
                residual_samples.append(sample.head(200))

    final_models: dict[str, Any] = {}
    target_model_rows: list[dict[str, Any]] = []
    for target in target_columns:
        target_frame = frame[frame[target].notna()].copy()
        if len(target_frame) < min_train_rows:
            continue
        alpha_counts = Counter(chosen_alphas.get(target) or [1.0])
        alpha = float(alpha_counts.most_common(1)[0][0])
        final_models[target] = fit_sparse_ridge_matrix(
            feature_matrix[target_frame.index.to_numpy()],
            target_frame[target].to_numpy(dtype=float),
            alpha,
        )
        target_model_rows.append({"target": target, "finalTrainingRows": int(len(target_frame)), "finalAlpha": alpha})

    model_bundle = {
        "predictorSchemaVersion": PREDICTOR_SCHEMA_VERSION,
        "predictorModelVersion": PREDICTOR_MODEL_VERSION,
        "claimBoundary": PREDICTOR_CLAIM_BOUNDARY,
        "featureColumns": {"categorical": list(CAT_FEATURE_COLUMNS), "numeric": list(NUMERIC_FEATURE_COLUMNS)},
        "targetColumns": list(target_columns),
        "splitGroupColumns": dict(SPLIT_GROUP_COLUMNS),
        "featureVectorizer": feature_vectorizer,
        "modelInputMode": "DictVectorizer sparse matrix built from row_to_features metadata",
        "finalModels": final_models,
        "targetModelRows": target_model_rows,
    }
    residual_df = pd.concat(residual_samples, ignore_index=True) if residual_samples else pd.DataFrame()
    return pd.DataFrame(metric_rows), pd.DataFrame(calibration), residual_df, model_bundle, pd.DataFrame(split_rows)


def split_manifest_rows(frame: pd.DataFrame, split_name: str, partition_col: str, group_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition, subset in frame.groupby(partition_col):
        rows.append(
            {
                "splitName": split_name,
                "partition": partition,
                "groupColumn": group_col,
                "rowCount": int(len(subset)),
                "uniqueGroupCount": int(subset[group_col].nunique()),
                "targetNonNullRowsJson": {
                    target: int(subset[target].notna().sum()) for target in DEFAULT_TARGET_COLUMNS if target in subset.columns
                },
            }
        )
    return rows


def baseline_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    evaluated = metrics[metrics["status"].eq("evaluated")].copy()
    if evaluated.empty:
        return pd.DataFrame()
    model = evaluated[evaluated["modelName"].eq("sparse_ridge")]
    baselines = evaluated[evaluated["modelName"].isin(["global_mean", "source_group_mean"])]
    rows: list[dict[str, Any]] = []
    for _, model_row in model.iterrows():
        subset = baselines[
            baselines["target"].eq(model_row["target"])
            & baselines["splitName"].eq(model_row["splitName"])
            & baselines["partition"].eq(model_row["partition"])
        ]
        for _, baseline in subset.iterrows():
            rows.append(
                {
                    "target": model_row["target"],
                    "splitName": model_row["splitName"],
                    "partition": model_row["partition"],
                    "baselineName": baseline["modelName"],
                    "modelRmse": model_row["rmse"],
                    "baselineRmse": baseline["rmse"],
                    "rmseImprovement": baseline["rmse"] - model_row["rmse"],
                    "rmseImprovementFraction": (baseline["rmse"] - model_row["rmse"]) / baseline["rmse"]
                    if baseline["rmse"]
                    else math.nan,
                    "modelR2": model_row["r2"],
                    "baselineR2": baseline["r2"],
                    "modelBeatsBaseline": bool(model_row["rmse"] < baseline["rmse"]),
                }
            )
    return pd.DataFrame(rows)


def validation_checks(
    frame: pd.DataFrame,
    target_coverage_df: pd.DataFrame,
    metrics: pd.DataFrame,
    split_manifest: pd.DataFrame,
    model_bundle_path: str | Path,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(scope: str, check: str, success: bool, severity: str = "error", detail: str = "") -> None:
        checks.append({"scope": scope, "check": check, "success": bool(success), "severity": severity, "detail": detail})

    add("inputs", "modeling_frame_nonempty", len(frame) > 0, detail=str(len(frame)))
    selected_count = int(target_coverage_df["selectedForTraining"].sum()) if "selectedForTraining" in target_coverage_df else 0
    add("targets", "at_least_five_targets_selected", selected_count >= 5, detail=str(selected_count))
    split_names = set(split_manifest["splitName"].astype(str)) if not split_manifest.empty else set()
    add("splits", "all_required_splits_present", set(SPLIT_GROUP_COLUMNS).issubset(split_names), detail=compact_json(sorted(split_names)))
    evaluated = metrics[metrics["status"].eq("evaluated")] if not metrics.empty else pd.DataFrame()
    add("metrics", "metrics_include_sparse_ridge", "sparse_ridge" in set(evaluated.get("modelName", [])), detail=str(len(evaluated)))
    add(
        "metrics",
        "metrics_include_simple_baselines",
        {"global_mean", "source_group_mean"}.issubset(set(evaluated.get("modelName", []))),
        detail=compact_json(sorted(set(evaluated.get("modelName", [])))),
    )
    test_model = evaluated[evaluated["modelName"].eq("sparse_ridge") & evaluated["partition"].eq("test")]
    add("metrics", "heldout_test_metrics_present", not test_model.empty, detail=str(len(test_model)))
    add("model", "model_bundle_written", Path(model_bundle_path).exists(), detail=str(model_bundle_path))

    for split_name, group_col in SPLIT_GROUP_COLUMNS.items():
        if split_name == "random":
            continue
        partition_col = f"{split_name}Partition"
        if partition_col not in frame.columns:
            add(split_name, "group_partition_column_present", False)
            continue
        train_groups = set(frame.loc[frame[partition_col].eq("train"), group_col].astype(str))
        test_groups = set(frame.loc[frame[partition_col].eq("test"), group_col].astype(str))
        validation_groups = set(frame.loc[frame[partition_col].eq("validation"), group_col].astype(str))
        add(split_name, "train_test_groups_disjoint", train_groups.isdisjoint(test_groups), detail=f"{len(train_groups)} train, {len(test_groups)} test")
        add(
            split_name,
            "train_validation_groups_disjoint",
            train_groups.isdisjoint(validation_groups),
            detail=f"{len(train_groups)} train, {len(validation_groups)} validation",
        )

    return pd.DataFrame(checks)


def performance_summary(metrics: pd.DataFrame, comparisons: pd.DataFrame) -> dict[str, Any]:
    evaluated = metrics[metrics["status"].eq("evaluated")] if not metrics.empty else pd.DataFrame()
    sparse_test = evaluated[evaluated["modelName"].eq("sparse_ridge") & evaluated["partition"].eq("test")]
    comparison_test = comparisons[comparisons["partition"].eq("test")] if not comparisons.empty else pd.DataFrame()
    global_test = comparison_test[comparison_test["baselineName"].eq("global_mean")] if not comparison_test.empty else pd.DataFrame()
    group_test = comparison_test[comparison_test["baselineName"].eq("source_group_mean")] if not comparison_test.empty else pd.DataFrame()
    return {
        "evaluatedMetricRows": int(len(evaluated)),
        "sparseRidgeTestRows": int(len(sparse_test)),
        "testTargets": sorted(sparse_test["target"].unique().tolist()) if not sparse_test.empty else [],
        "testSplitNames": sorted(sparse_test["splitName"].unique().tolist()) if not sparse_test.empty else [],
        "medianTestRmse": float(sparse_test["rmse"].median()) if not sparse_test.empty else math.nan,
        "medianTestR2": float(sparse_test["r2"].median()) if not sparse_test.empty else math.nan,
        "beatsGlobalMeanCount": int(global_test["modelBeatsBaseline"].sum()) if not global_test.empty else 0,
        "globalMeanComparisonCount": int(len(global_test)),
        "beatsSourceGroupMeanCount": int(group_test["modelBeatsBaseline"].sum()) if not group_test.empty else 0,
        "sourceGroupMeanComparisonCount": int(len(group_test)),
    }


def artifact_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)}
