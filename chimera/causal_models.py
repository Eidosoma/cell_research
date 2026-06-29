"""E06 S13 causal/predictive models for final morphology.

The causal language in this module is deliberately bounded. These models use
simulation rows and paired computational interventions to generate hypotheses
about local-policy collectives; they are not biological causal evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .mixtures import artifact_records, compact_result_csv, git_value, write_json
from .panel import CLAIM_BOUNDARY, json_ready


STEP_ID = "S13"
STEP_NUMBER = 13
CAUSAL_MODEL_VERSION = "e06_s13_causal_predictive_models.v1"
DEFAULT_S12_RESULTS_PATH = Path("/artifacts/results/e06_developmental_history.parquet")
GROUP_SPLIT_SEED = 13013
HISTORY_SENSITIVE_THRESHOLD = 0.15
CAUSAL_HYPOTHESIS_CAVEAT = (
    "Causal hypotheses are generated from simulation contrasts and predictive associations only; "
    "they are not biological causal proof."
)
LEAKAGE_AUDIT_CAVEAT = (
    "Primary models exclude same-row final states, scores, deltas, post-run event counts, hashes, "
    "and verbose state arrays; upstream context labels are retained only as audited simulation-context proxies."
)

REFERENCE_PROTOCOLS = {"matched_no_mutant_replay", "simultaneous_clone_reference"}
EXACT_REPLAY_FEATURES = ["simultaneousReplayMatchesS11Reference", "noHistoryReplayMatchesS11NoMutant"]

CATEGORICAL_FEATURE_CANDIDATES: tuple[str, ...] = (
    "pairCategory",
    "ratioLabel",
    "arrangementType",
    "valueProfile",
    "perturbationType",
    "goalMode",
    "goalCompatibilityClass",
    "hostPolicyId",
    "donorPolicyId",
    "leftPanelPolicyId",
    "rightPanelPolicyId",
    "policyAPanelPolicyId",
    "policyBPanelPolicyId",
    "mutantBehaviorVariant",
    "mutantObjectiveFamily",
    "mutantClonePositionName",
    "s11MutantOutcomeClass",
    "graftOutcomeClass",
    "governanceVariant",
    "governanceFamily",
    "interfaceRuleVariant",
    "interfaceRuleFamily",
    "historyProtocol",
    "historyConditionKind",
    "historyTimingLabel",
    "memoryCarryoverMode",
    "policyStateCarryoverMode",
    "signalFieldCarryoverMode",
    "baselineGovernanceVariant",
    "baselineInterfaceRuleVariant",
    "preGraftTimingLabel",
)

NUMERIC_FEATURE_CANDIDATES: tuple[str, ...] = (
    "n",
    "targetPolicyCount",
    "policyCount",
    "influenceRange",
    "mutantCloneSize",
    "mutantCloneStartIndex",
    "mutantCloneEndExclusive",
    "preExposureActivationCap",
    "cloneIntroductionActivation",
    "transientPerturbationStartActivation",
    "transientPerturbationEndActivation",
    "historyTotalActivationCap",
    "historyActualStageCount",
    "graftSize",
    "graftStartIndex",
    "graftEndExclusive",
    "preGraftActivationCap",
    "s10StressScore",
    "containmentScore",
    "takeoverProxyScore",
    "s11ContextPriorityScore",
    "s12PriorityScore",
)

BOOLEAN_FEATURE_CANDIDATES: tuple[str, ...] = (
    "governanceEnabled",
    "localInformationOnly",
    "usesGlobalState",
    "usesTargetMap",
    "usesOrganizer",
    "broadControlLike",
    "upperBoundControlProxy",
    "interfaceRuleEnabled",
    "clonePresentAtStart",
    "cloneIntroducedDuringRun",
    "divisionEnabled",
    "memoryEligiblePolicyPresent",
    "simultaneousReplayMatchesS11Reference",
    "noHistoryReplayMatchesS11NoMutant",
    "graftApplied",
)

DERIVED_FEATURES: tuple[str, ...] = (
    "isResetProtocol",
    "isCarryoverProtocol",
    "isReferenceReplayProtocol",
    "isTransientProtocol",
    "isStagedProtocol",
    "exactReplayIntegrityAllMatched",
)

DIRECT_TARGET_COLUMNS: tuple[str, ...] = (
    "s07MosaicClass",
    "finalTargetQualityScore",
    "dominanceProxyScore",
    "finalAggregation",
    "historyOutcomeClass",
    "historyEffectMagnitudeScore",
    "historyDisruptionProxyScore",
)

UPSTREAM_CONTEXT_PROXY_FEATURES: tuple[str, ...] = (
    "s11MutantOutcomeClass",
    "graftOutcomeClass",
    "containmentScore",
    "takeoverProxyScore",
    "s10StressScore",
    "s11ContextPriorityScore",
    "s12PriorityScore",
)


def load_s12_history_results(path: Path = DEFAULT_S12_RESULTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"S12 developmental-history results missing: {path}")
    df = pd.read_parquet(path)
    required = {
        "conditionId",
        "s12SourceS11ConditionId",
        "s07MosaicClass",
        "finalTargetQualityScore",
        "dominanceProxyScore",
        "finalAggregation",
        "historyOutcomeClass",
        "historyEffectMagnitudeScore",
        "historyProtocol",
        "memoryCarryoverMode",
        "simultaneousReplayMatchesS11Reference",
        "noHistoryReplayMatchesS11NoMutant",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"S12 results missing required S13 columns: {missing}")
    if df.empty:
        raise ValueError("S12 results are empty")
    return df.copy().reset_index(drop=True)


def target_definitions() -> pd.DataFrame:
    rows = [
        {
            "targetName": "final_morphology_class",
            "targetColumn": "finalMorphologyClassTarget",
            "sourceColumn": "s07MosaicClass",
            "taskType": "multiclass_classification",
            "primaryMetric": "balancedAccuracy",
            "definition": "S07 final mosaic class assigned to the S12 final state.",
            "metricBoundaryCaveat": "Mosaic classes are one-dimensional computational final-state proxies.",
        },
        {
            "targetName": "rescue_success_proxy",
            "targetColumn": "rescueSuccessProxyTarget",
            "sourceColumn": "historyOutcomeClass",
            "taskType": "binary_classification",
            "primaryMetric": "rocAucOrBalancedAccuracy",
            "definition": "`historyOutcomeClass == history_rescue_proxy`.",
            "metricBoundaryCaveat": "Rescue is a bounded target-quality/min-goal computational proxy.",
        },
        {
            "targetName": "history_sensitive_proxy",
            "targetColumn": "historySensitiveProxyTarget",
            "sourceColumn": "historyEffectMagnitudeScore",
            "taskType": "binary_classification",
            "primaryMetric": "rocAucOrBalancedAccuracy",
            "definition": f"`historyEffectMagnitudeScore >= {HISTORY_SENSITIVE_THRESHOLD}`.",
            "metricBoundaryCaveat": "History sensitivity is a thresholded final-state proxy, not biological memory.",
        },
        {
            "targetName": "final_target_quality",
            "targetColumn": "finalTargetQualityTarget",
            "sourceColumn": "finalTargetQualityScore",
            "taskType": "regression",
            "primaryMetric": "r2",
            "definition": "S12 final target-quality score after goal-aware scoring.",
            "metricBoundaryCaveat": "Target quality is a simulation score over policy-goal objectives.",
        },
        {
            "targetName": "dominance_proxy",
            "targetColumn": "dominanceProxyTarget",
            "sourceColumn": "dominanceProxyScore",
            "taskType": "regression",
            "primaryMetric": "r2",
            "definition": "S05/S06-derived dominance proxy score carried through S12 final states.",
            "metricBoundaryCaveat": "Dominance is a final-state computational proxy.",
        },
        {
            "targetName": "final_aggregation",
            "targetColumn": "finalAggregationTarget",
            "sourceColumn": "finalAggregation",
            "taskType": "regression",
            "primaryMetric": "r2",
            "definition": "Final policy-label Aggregation score in the one-dimensional array.",
            "metricBoundaryCaveat": "Aggregation is a spatial label-clustering proxy.",
        },
    ]
    for row in rows:
        row["causalCaveat"] = CAUSAL_HYPOTHESIS_CAVEAT
        row["claimBoundary"] = CLAIM_BOUNDARY
        row["causalModelVersion"] = CAUSAL_MODEL_VERSION
    return pd.DataFrame(rows)


def _present(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def derive_s13_fields(s12_results: pd.DataFrame) -> pd.DataFrame:
    df = s12_results.copy()
    protocol = df["historyProtocol"].astype(str)
    carryover = df["memoryCarryoverMode"].astype(str)
    df["isResetProtocol"] = carryover.str.contains("reset", case=False, na=False)
    df["isCarryoverProtocol"] = carryover.str.contains("carryover", case=False, na=False)
    df["isReferenceReplayProtocol"] = protocol.isin(REFERENCE_PROTOCOLS)
    df["isTransientProtocol"] = protocol.str.contains("transient", case=False, na=False)
    df["isStagedProtocol"] = protocol.str.contains("staged", case=False, na=False)
    df["exactReplayIntegrityAllMatched"] = (
        df["simultaneousReplayMatchesS11Reference"].fillna(False).map(bool)
        & df["noHistoryReplayMatchesS11NoMutant"].fillna(False).map(bool)
    )
    df["finalMorphologyClassTarget"] = df["s07MosaicClass"].astype(str)
    df["rescueSuccessProxyTarget"] = df["historyOutcomeClass"].astype(str).eq("history_rescue_proxy")
    df["historySensitiveProxyTarget"] = (
        pd.to_numeric(df["historyEffectMagnitudeScore"], errors="coerce").fillna(0.0) >= HISTORY_SENSITIVE_THRESHOLD
    )
    df["finalTargetQualityTarget"] = pd.to_numeric(df["finalTargetQualityScore"], errors="coerce")
    df["dominanceProxyTarget"] = pd.to_numeric(df["dominanceProxyScore"], errors="coerce")
    df["finalAggregationTarget"] = pd.to_numeric(df["finalAggregation"], errors="coerce")
    return df


def selected_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    categorical = _present(df, CATEGORICAL_FEATURE_CANDIDATES)
    numeric = _present(df, NUMERIC_FEATURE_CANDIDATES)
    boolean = _present(df, BOOLEAN_FEATURE_CANDIDATES) + [col for col in DERIVED_FEATURES if col in df.columns]
    boolean = list(dict.fromkeys(boolean))
    feature_cols = categorical + numeric + boolean
    return feature_cols, categorical, numeric, boolean


def _is_high_risk_output_column(column: str) -> bool:
    lower = column.lower()
    if column in DIRECT_TARGET_COLUMNS:
        return True
    if column.startswith("deltaVs") or column.startswith("simultaneous") or column.startswith("noHistory"):
        return column not in EXACT_REPLAY_FEATURES
    blocked_tokens = (
        "finalstatehash",
        "finalvaluesjson",
        "finalpanelpolicyidsjson",
        "historychanged",
        "outcomeclassifier",
        "effectmagnitudescore",
        "disruptionproxyscore",
    )
    return any(token in lower for token in blocked_tokens)


def feature_leakage_audit(input_columns: Sequence[str], included_features: Sequence[str]) -> pd.DataFrame:
    included = set(included_features)
    rows: list[dict[str, Any]] = []
    for column in input_columns:
        included_flag = column in included
        if included_flag and column in UPSTREAM_CONTEXT_PROXY_FEATURES:
            risk = "medium_context_proxy"
            reason = "Included as prior simulation-context metadata; interpret as predictive context, not direct S12 causality."
        elif included_flag and column in EXACT_REPLAY_FEATURES:
            risk = "low_integrity_flag"
            reason = "Included as requested exact-replay integrity indicator; constant/pass-through flags do not encode final morphology."
        elif included_flag:
            risk = "low_design_or_context_feature"
            reason = "Included as predeclared design, policy, governance, interface, graft, mutant, or history descriptor."
        elif _is_high_risk_output_column(column):
            risk = "high_excluded_same_row_outcome"
            reason = "Excluded because it is a same-row final outcome, reference outcome, delta, hash, or post-hoc label."
        elif column.endswith("Json") or "Hash" in column or column in {"conditionId", "s12SourceS11ConditionId"}:
            risk = "excluded_identifier_or_verbose_state"
            reason = "Excluded to prevent memorization of identities, hashes, or verbose state arrays."
        elif any(token in column for token in ["swapCount", "activationCount", "comparisonCount", "eventCount", "Evaluations", "Interventions"]):
            risk = "excluded_post_run_process_measure"
            reason = "Excluded because it is measured after or during the run being predicted."
        else:
            risk = "excluded_not_predeclared"
            reason = "Not in the predeclared S13 primary feature set."
        rows.append(
            {
                "featureName": column,
                "featureRole": "input_source_column",
                "includedInPrimaryModel": bool(included_flag),
                "leakageRisk": risk,
                "reason": reason,
                "sourceColumnPresent": True,
                "causalModelVersion": CAUSAL_MODEL_VERSION,
                "claimBoundary": CLAIM_BOUNDARY,
            }
        )
    for column in DERIVED_FEATURES:
        rows.append(
            {
                "featureName": column,
                "featureRole": "derived_feature",
                "includedInPrimaryModel": column in included,
                "leakageRisk": "low_derived_from_protocol_or_replay_flags",
                "reason": "Derived only from S12 protocol/reset/carryover labels or exact-replay integrity flags.",
                "sourceColumnPresent": column in included,
                "causalModelVersion": CAUSAL_MODEL_VERSION,
                "claimBoundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["featureName", "featureRole"], keep="last").reset_index(drop=True)


def build_feature_matrix(s12_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], list[str], list[str]]:
    derived = derive_s13_fields(s12_results)
    feature_cols, categorical, numeric, boolean = selected_feature_columns(derived)
    target_defs = target_definitions()
    target_cols = [str(row.targetColumn) for row in target_defs.itertuples(index=False)]
    id_cols = [
        col
        for col in [
            "conditionId",
            "s12SourceS11ConditionId",
            "historyProtocol",
            "s12ContextIndex",
            "s11MutantOutcomeClass",
            "governanceVariant",
            "interfaceRuleVariant",
        ]
        if col in derived.columns
    ]
    matrix_columns = list(dict.fromkeys([*id_cols, *feature_cols, *target_cols]))
    matrix = derived[matrix_columns].copy()
    for col in categorical:
        matrix[col] = matrix[col].astype("string").fillna("missing").astype(str)
    for col in numeric:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
    for col in boolean:
        matrix[col] = matrix[col].fillna(False).map(bool).astype(int)
    audit = feature_leakage_audit(s12_results.columns, feature_cols)
    return matrix, target_defs, audit, feature_cols, categorical, numeric, boolean


def make_group_holdout_split(
    matrix: pd.DataFrame,
    *,
    group_col: str = "s12SourceS11ConditionId",
    test_size: float = 0.25,
    random_state: int = GROUP_SPLIT_SEED,
) -> pd.DataFrame:
    if group_col not in matrix.columns:
        raise ValueError(f"group column missing from feature matrix: {group_col}")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(matrix, groups=matrix[group_col].astype(str)))
    split = matrix[["conditionId", group_col]].copy()
    split["split"] = "train"
    split.loc[split.index[test_idx], "split"] = "test"
    split["groupSplitSeed"] = int(random_state)
    split["groupSplitTestSize"] = float(test_size)
    split["causalModelVersion"] = CAUSAL_MODEL_VERSION
    if set(split.loc[split["split"] == "train", group_col]).intersection(set(split.loc[split["split"] == "test", group_col])):
        raise ValueError("group holdout split leaked source contexts between train and test")
    return split


def _preprocessor(categorical: Sequence[str], numeric: Sequence[str], boolean: Sequence[str]) -> ColumnTransformer:
    numeric_features = list(dict.fromkeys([*numeric, *boolean]))
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(categorical),
            )
        )
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def _classification_models(random_state: int) -> dict[str, Any]:
    return {
        "decision_tree_classifier": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
    }


def _regression_models(random_state: int) -> dict[str, Any]:
    return {
        "decision_tree_regressor": DecisionTreeRegressor(max_depth=4, min_samples_leaf=3, random_state=random_state),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(random_state=random_state, max_depth=2),
    }


def _classes_json(values: Sequence[Any]) -> str:
    return json.dumps([str(value) for value in values], sort_keys=True)


def _multiclass_brier(y_true: Sequence[Any], classes: Sequence[Any], proba: np.ndarray) -> float:
    class_lookup = {value: index for index, value in enumerate(classes)}
    encoded = np.zeros_like(proba, dtype=float)
    for row_index, value in enumerate(y_true):
        if value in class_lookup:
            encoded[row_index, class_lookup[value]] = 1.0
    return float(np.mean(np.sum((proba - encoded) ** 2, axis=1)))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _classification_performance(
    *,
    target_name: str,
    model_name: str,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    classes: Sequence[Any],
) -> dict[str, Any]:
    row = {
        "targetName": target_name,
        "modelName": model_name,
        "taskType": "classification",
        "trainCount": int(len(y_train)),
        "testCount": int(len(y_test)),
        "trainClassCountsJson": json.dumps({str(k): int(v) for k, v in Counter(y_train).items()}, sort_keys=True),
        "testClassCountsJson": json.dumps({str(k): int(v) for k, v in Counter(y_test).items()}, sort_keys=True),
        "classesJson": _classes_json(classes),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balancedAccuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macroF1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weightedF1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "logLoss": float("nan"),
        "multiclassBrier": float("nan"),
        "rocAuc": float("nan"),
        "averagePrecision": float("nan"),
        "brierScore": float("nan"),
        "primaryMetricName": "balancedAccuracy",
        "primaryMetricValue": float(balanced_accuracy_score(y_test, y_pred)),
        "validationFold": "group_holdout_by_s12SourceS11ConditionId",
        "causalModelVersion": CAUSAL_MODEL_VERSION,
        "claimBoundary": CLAIM_BOUNDARY,
    }
    if proba is not None and len(proba):
        row["multiclassBrier"] = _multiclass_brier(list(y_test), list(classes), proba)
        try:
            row["logLoss"] = float(log_loss(y_test, proba, labels=list(classes)))
        except ValueError:
            row["logLoss"] = float("nan")
        if len(classes) == 2:
            positive_index = 1
            positive_label = classes[positive_index]
            positive_proba = proba[:, positive_index]
            y_binary = np.array([value == positive_label for value in y_test], dtype=int)
            if len(np.unique(y_binary)) == 2:
                row["rocAuc"] = float(roc_auc_score(y_binary, positive_proba))
                row["averagePrecision"] = float(average_precision_score(y_binary, positive_proba))
            row["brierScore"] = float(brier_score_loss(y_binary, positive_proba))
            if np.isfinite(row["rocAuc"]):
                row["primaryMetricName"] = "rocAuc"
                row["primaryMetricValue"] = row["rocAuc"]
    return row


def _regression_performance(
    *,
    target_name: str,
    model_name: str,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    mse = float(mean_squared_error(y_test, y_pred))
    try:
        rho = float(spearmanr(y_test, y_pred, nan_policy="omit").statistic)
    except Exception:
        rho = float("nan")
    row = {
        "targetName": target_name,
        "modelName": model_name,
        "taskType": "regression",
        "trainCount": int(len(y_train)),
        "testCount": int(len(y_test)),
        "trainClassCountsJson": "",
        "testClassCountsJson": "",
        "classesJson": "",
        "meanAbsoluteError": float(mean_absolute_error(y_test, y_pred)),
        "rootMeanSquaredError": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else float("nan"),
        "spearmanR": rho,
        "primaryMetricName": "r2",
        "primaryMetricValue": float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else float("nan"),
        "validationFold": "group_holdout_by_s12SourceS11ConditionId",
        "causalModelVersion": CAUSAL_MODEL_VERSION,
        "claimBoundary": CLAIM_BOUNDARY,
    }
    return row


def _source_feature_from_transformed(name: str, categorical: Sequence[str], numeric_boolean: Sequence[str]) -> str:
    if name.startswith("num__"):
        return name[len("num__") :]
    if name.startswith("cat__"):
        rest = name[len("cat__") :]
        for feature in sorted(categorical, key=len, reverse=True):
            if rest == feature or rest.startswith(f"{feature}_"):
                return feature
    for feature in sorted(numeric_boolean, key=len, reverse=True):
        if name.endswith(feature):
            return feature
    return name


def _feature_importance_rows(
    pipeline: Pipeline,
    *,
    target_name: str,
    model_name: str,
    categorical: Sequence[str],
    numeric: Sequence[str],
    boolean: Sequence[str],
) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return []
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    importances = np.asarray(model.feature_importances_, dtype=float)
    numeric_boolean = list(dict.fromkeys([*numeric, *boolean]))
    rows = []
    for name, importance in zip(names, importances, strict=False):
        source = _source_feature_from_transformed(name, categorical, numeric_boolean)
        rows.append(
            {
                "targetName": target_name,
                "modelName": model_name,
                "sourceFeature": source,
                "transformedFeature": name,
                "importance": float(importance),
                "featureImportanceType": "tree_gini_or_variance_reduction",
                "causalModelVersion": CAUSAL_MODEL_VERSION,
                "claimBoundary": CLAIM_BOUNDARY,
            }
        )
    return rows


def _prediction_rows(
    *,
    matrix: pd.DataFrame,
    indices: np.ndarray,
    target_name: str,
    model_name: str,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    proba: np.ndarray | None = None,
    classes: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    classes = list(classes or [])
    for offset, row_index in enumerate(indices):
        payload = {
            "conditionId": str(matrix.iloc[int(row_index)]["conditionId"]),
            "s12SourceS11ConditionId": str(matrix.iloc[int(row_index)]["s12SourceS11ConditionId"]),
            "split": "test",
            "targetName": target_name,
            "modelName": model_name,
            "trueValue": str(y_true[offset]),
            "predictedValue": str(y_pred[offset]),
            "predictedProbabilityJson": "",
            "causalModelVersion": CAUSAL_MODEL_VERSION,
        }
        if proba is not None:
            payload["predictedProbabilityJson"] = json.dumps(
                {str(cls): float(proba[offset, class_index]) for class_index, cls in enumerate(classes)},
                sort_keys=True,
            )
        rows.append(payload)
    return rows


def _calibration_rows(
    *,
    target_name: str,
    model_name: str,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    proba: np.ndarray,
    classes: Sequence[Any],
    bins: int = 5,
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    confidence = np.max(proba, axis=1)
    correct = np.array([yt == yp for yt, yp in zip(y_true, y_pred, strict=False)], dtype=float)
    for i in range(bins):
        lower, upper = float(edges[i]), float(edges[i + 1])
        mask = (confidence >= lower) & (confidence <= upper if i == bins - 1 else confidence < upper)
        if not mask.any():
            continue
        rows.append(
            {
                "targetName": target_name,
                "modelName": model_name,
                "calibrationType": "top_label_confidence",
                "binLower": lower,
                "binUpper": upper,
                "binCount": int(mask.sum()),
                "meanPredictedProbability": float(confidence[mask].mean()),
                "observedFrequency": float(correct[mask].mean()),
                "absoluteCalibrationError": float(abs(confidence[mask].mean() - correct[mask].mean())),
                "causalModelVersion": CAUSAL_MODEL_VERSION,
            }
        )
    if len(classes) == 2:
        positive_index = 1
        positive_label = classes[positive_index]
        positive = np.array([value == positive_label for value in y_true], dtype=float)
        positive_proba = proba[:, positive_index]
        for i in range(bins):
            lower, upper = float(edges[i]), float(edges[i + 1])
            mask = (positive_proba >= lower) & (positive_proba <= upper if i == bins - 1 else positive_proba < upper)
            if not mask.any():
                continue
            rows.append(
                {
                    "targetName": target_name,
                    "modelName": model_name,
                    "calibrationType": "positive_class_probability",
                    "positiveClass": str(positive_label),
                    "binLower": lower,
                    "binUpper": upper,
                    "binCount": int(mask.sum()),
                    "meanPredictedProbability": float(positive_proba[mask].mean()),
                    "observedFrequency": float(positive[mask].mean()),
                    "absoluteCalibrationError": float(abs(positive_proba[mask].mean() - positive[mask].mean())),
                    "causalModelVersion": CAUSAL_MODEL_VERSION,
                }
            )
    return rows


def fit_predictive_models(
    matrix: pd.DataFrame,
    target_defs: pd.DataFrame,
    split: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical: Sequence[str],
    numeric: Sequence[str],
    boolean: Sequence[str],
    *,
    random_state: int = GROUP_SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_lookup = split.set_index("conditionId")["split"].to_dict()
    split_labels = matrix["conditionId"].map(split_lookup).astype(str)
    train_mask = split_labels.eq("train").to_numpy()
    test_mask = split_labels.eq("test").to_numpy()
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    X = matrix[list(feature_cols)].copy()
    performance_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    importance: list[dict[str, Any]] = []

    for target in target_defs.to_dict(orient="records"):
        target_name = str(target["targetName"])
        target_col = str(target["targetColumn"])
        task = str(target["taskType"])
        valid = matrix[target_col].notna().to_numpy()
        target_train_mask = train_mask & valid
        target_test_mask = test_mask & valid
        y_train = matrix.loc[target_train_mask, target_col]
        y_test = matrix.loc[target_test_mask, target_col]
        X_train = X.loc[target_train_mask]
        X_test = X.loc[target_test_mask]
        if "classification" in task:
            if target_name == "final_morphology_class":
                y_train_model = y_train.astype(str)
                y_test_model = y_test.astype(str)
            else:
                y_train_model = y_train.map(bool)
                y_test_model = y_test.map(bool)
            models = _classification_models(random_state)
            for model_name, estimator in models.items():
                pipeline = Pipeline(steps=[("preprocess", _preprocessor(categorical, numeric, boolean)), ("model", estimator)])
                pipeline.fit(X_train, y_train_model)
                y_pred = pipeline.predict(X_test)
                proba = pipeline.predict_proba(X_test) if hasattr(pipeline.named_steps["model"], "predict_proba") else None
                classes = list(pipeline.named_steps["model"].classes_)
                performance_rows.append(
                    _classification_performance(
                        target_name=target_name,
                        model_name=model_name,
                        y_train=y_train_model,
                        y_test=y_test_model,
                        y_pred=y_pred,
                        proba=proba,
                        classes=classes,
                    )
                )
                prediction_rows.extend(
                    _prediction_rows(
                        matrix=matrix,
                        indices=test_indices[valid[test_indices]],
                        target_name=target_name,
                        model_name=model_name,
                        y_true=list(y_test_model),
                        y_pred=list(y_pred),
                        proba=proba,
                        classes=classes,
                    )
                )
                if proba is not None:
                    calibration.extend(
                        _calibration_rows(
                            target_name=target_name,
                            model_name=model_name,
                            y_true=list(y_test_model),
                            y_pred=list(y_pred),
                            proba=proba,
                            classes=classes,
                        )
                    )
                importance.extend(
                    _feature_importance_rows(
                        pipeline,
                        target_name=target_name,
                        model_name=model_name,
                        categorical=categorical,
                        numeric=numeric,
                        boolean=boolean,
                    )
                )
        else:
            y_train_model = pd.to_numeric(y_train, errors="coerce")
            y_test_model = pd.to_numeric(y_test, errors="coerce")
            finite_train = np.isfinite(y_train_model.to_numpy(dtype=float))
            finite_test = np.isfinite(y_test_model.to_numpy(dtype=float))
            X_train_reg = X_train.loc[finite_train]
            X_test_reg = X_test.loc[finite_test]
            y_train_reg = y_train_model.loc[finite_train]
            y_test_reg = y_test_model.loc[finite_test]
            models = _regression_models(random_state)
            for model_name, estimator in models.items():
                pipeline = Pipeline(steps=[("preprocess", _preprocessor(categorical, numeric, boolean)), ("model", estimator)])
                pipeline.fit(X_train_reg, y_train_reg)
                y_pred = pipeline.predict(X_test_reg)
                performance_rows.append(
                    _regression_performance(
                        target_name=target_name,
                        model_name=model_name,
                        y_train=y_train_reg,
                        y_test=y_test_reg,
                        y_pred=y_pred,
                    )
                )
                prediction_rows.extend(
                    _prediction_rows(
                        matrix=matrix,
                        indices=test_indices[valid[test_indices]][finite_test],
                        target_name=target_name,
                        model_name=model_name,
                        y_true=list(y_test_reg),
                        y_pred=[_safe_float(value) for value in y_pred],
                    )
                )
                importance.extend(
                    _feature_importance_rows(
                        pipeline,
                        target_name=target_name,
                        model_name=model_name,
                        categorical=categorical,
                        numeric=numeric,
                        boolean=boolean,
                    )
                )
    performance = pd.DataFrame(performance_rows)
    predictions = pd.DataFrame(prediction_rows)
    calibration_df = pd.DataFrame(calibration)
    importance_df = pd.DataFrame(importance)
    for frame in [performance, predictions, calibration_df, importance_df]:
        if not frame.empty:
            frame["groupSplitSeed"] = int(random_state)
            frame["claimBoundary"] = CLAIM_BOUNDARY
    return performance, predictions, calibration_df, importance_df


def paired_protocol_contrasts(scored: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "finalTargetQualityScore",
        "dominanceProxyScore",
        "finalAggregation",
        "historyEffectMagnitudeScore",
    ]
    comparisons = [
        ("early_staged_introduction_reset", "early_staged_introduction_carryover", "carryover_minus_reset_early_staged"),
        ("late_staged_introduction_reset", "late_staged_introduction_carryover", "carryover_minus_reset_late_staged"),
        ("early_transient_rule_release_reset", "early_transient_rule_release_carryover", "carryover_minus_reset_early_transient"),
        ("late_transient_rule_release_reset", "late_transient_rule_release_carryover", "carryover_minus_reset_late_transient"),
        ("simultaneous_clone_reference", "early_staged_introduction_reset", "early_staged_reset_minus_simultaneous"),
        ("simultaneous_clone_reference", "late_staged_introduction_reset", "late_staged_reset_minus_simultaneous"),
        ("simultaneous_clone_reference", "early_transient_rule_release_reset", "early_transient_reset_minus_simultaneous"),
        ("simultaneous_clone_reference", "late_transient_rule_release_reset", "late_transient_reset_minus_simultaneous"),
    ]
    rows: list[dict[str, Any]] = []
    keyed = scored.set_index(["s12SourceS11ConditionId", "historyProtocol"], drop=False)
    groups = sorted(scored["s12SourceS11ConditionId"].astype(str).unique())
    for reference, treatment, comparison_name in comparisons:
        pair_rows = []
        for group in groups:
            key_ref = (group, reference)
            key_treat = (group, treatment)
            if key_ref not in keyed.index or key_treat not in keyed.index:
                continue
            ref = keyed.loc[key_ref]
            treat = keyed.loc[key_treat]
            if isinstance(ref, pd.DataFrame):
                ref = ref.iloc[0]
            if isinstance(treat, pd.DataFrame):
                treat = treat.iloc[0]
            pair_rows.append((str(group), ref, treat))
        for metric in metrics:
            diffs = []
            for _, ref, treat in pair_rows:
                diffs.append(_safe_float(treat.get(metric)) - _safe_float(ref.get(metric)))
            diffs_array = np.asarray([value for value in diffs if np.isfinite(value)], dtype=float)
            if len(diffs_array) == 0:
                continue
            rows.append(
                {
                    "comparisonName": comparison_name,
                    "referenceProtocol": reference,
                    "treatmentProtocol": treatment,
                    "metricName": metric,
                    "nPairs": int(len(diffs_array)),
                    "meanTreatmentMinusReference": float(np.mean(diffs_array)),
                    "medianTreatmentMinusReference": float(np.median(diffs_array)),
                    "stdTreatmentMinusReference": float(np.std(diffs_array, ddof=1)) if len(diffs_array) > 1 else float("nan"),
                    "positivePairFraction": float(np.mean(diffs_array > 0)),
                    "negativePairFraction": float(np.mean(diffs_array < 0)),
                    "evidenceType": "paired_within_s12_source_context_simulation_contrast",
                    "causalCaveat": CAUSAL_HYPOTHESIS_CAVEAT,
                    "causalModelVersion": CAUSAL_MODEL_VERSION,
                }
            )
        mismatch = []
        rescue_diff = []
        for _, ref, treat in pair_rows:
            mismatch.append(str(treat.get("s07MosaicClass")) != str(ref.get("s07MosaicClass")))
            rescue_diff.append(
                float(str(treat.get("historyOutcomeClass")) == "history_rescue_proxy")
                - float(str(ref.get("historyOutcomeClass")) == "history_rescue_proxy")
            )
        if mismatch:
            rows.append(
                {
                    "comparisonName": comparison_name,
                    "referenceProtocol": reference,
                    "treatmentProtocol": treatment,
                    "metricName": "mosaicClassMismatch",
                    "nPairs": int(len(mismatch)),
                    "meanTreatmentMinusReference": float(np.mean(mismatch)),
                    "medianTreatmentMinusReference": float(np.median(mismatch)),
                    "stdTreatmentMinusReference": float(np.std(mismatch, ddof=1)) if len(mismatch) > 1 else float("nan"),
                    "positivePairFraction": float(np.mean(mismatch)),
                    "negativePairFraction": 0.0,
                    "evidenceType": "paired_within_s12_source_context_simulation_contrast",
                    "causalCaveat": CAUSAL_HYPOTHESIS_CAVEAT,
                    "causalModelVersion": CAUSAL_MODEL_VERSION,
                }
            )
        if rescue_diff:
            diffs_array = np.asarray(rescue_diff, dtype=float)
            rows.append(
                {
                    "comparisonName": comparison_name,
                    "referenceProtocol": reference,
                    "treatmentProtocol": treatment,
                    "metricName": "rescueSuccessProxyTarget",
                    "nPairs": int(len(diffs_array)),
                    "meanTreatmentMinusReference": float(np.mean(diffs_array)),
                    "medianTreatmentMinusReference": float(np.median(diffs_array)),
                    "stdTreatmentMinusReference": float(np.std(diffs_array, ddof=1)) if len(diffs_array) > 1 else float("nan"),
                    "positivePairFraction": float(np.mean(diffs_array > 0)),
                    "negativePairFraction": float(np.mean(diffs_array < 0)),
                    "evidenceType": "paired_within_s12_source_context_simulation_contrast",
                    "causalCaveat": CAUSAL_HYPOTHESIS_CAVEAT,
                    "causalModelVersion": CAUSAL_MODEL_VERSION,
                }
            )
    return pd.DataFrame(rows)


def aggregate_feature_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    return (
        importance.groupby(["targetName", "modelName", "sourceFeature"], dropna=False)
        .agg(
            totalImportance=("importance", "sum"),
            maxImportance=("importance", "max"),
            transformedFeatureCount=("transformedFeature", "nunique"),
        )
        .reset_index()
        .sort_values(["targetName", "modelName", "totalImportance"], ascending=[True, True, False], kind="mergesort")
    )


def causal_hypothesis_table(
    contrasts: pd.DataFrame,
    aggregate_importance: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not contrasts.empty:
        contrast_top = contrasts.assign(absEffect=lambda frame: frame["meanTreatmentMinusReference"].abs()).sort_values(
            ["absEffect", "nPairs"], ascending=[False, False], kind="mergesort"
        )
        for row in contrast_top.head(12).to_dict(orient="records"):
            direction = "increase" if _safe_float(row["meanTreatmentMinusReference"]) > 0 else "decrease"
            rows.append(
                {
                    "hypothesisId": f"s13_paired_{len(rows) + 1:03d}",
                    "featureFamily": "history_protocol_intervention",
                    "candidateDeterminant": str(row["comparisonName"]),
                    "targetName": str(row["metricName"]),
                    "hypothesis": (
                        f"Within matched S12 source contexts, {row['treatmentProtocol']} may {direction} "
                        f"{row['metricName']} relative to {row['referenceProtocol']}."
                    ),
                    "evidenceType": str(row["evidenceType"]),
                    "meanEffect": _safe_float(row["meanTreatmentMinusReference"]),
                    "nEvidenceRows": int(row["nPairs"]),
                    "supportingModelMetric": "",
                    "causalCaveat": CAUSAL_HYPOTHESIS_CAVEAT,
                    "metricBoundaryCaveat": "Effect sizes are paired simulation-proxy contrasts over selected S12 contexts.",
                    "claimBoundary": CLAIM_BOUNDARY,
                    "causalModelVersion": CAUSAL_MODEL_VERSION,
                }
            )
    if not aggregate_importance.empty:
        primary_models = aggregate_importance[
            aggregate_importance["modelName"].astype(str).isin({"random_forest_classifier", "random_forest_regressor"})
        ].copy()
        primary_models = primary_models.sort_values(
            ["targetName", "totalImportance"], ascending=[True, False], kind="mergesort"
        )
        best_metric = performance.sort_values(["targetName", "primaryMetricValue"], ascending=[True, False], kind="mergesort")
        metric_lookup = {
            str(row.targetName): f"{row.primaryMetricName}={_safe_float(row.primaryMetricValue):.3f}"
            for row in best_metric.itertuples(index=False)
        }
        for row in primary_models.groupby("targetName", sort=False).head(3).to_dict(orient="records"):
            rows.append(
                {
                    "hypothesisId": f"s13_predictive_{len(rows) + 1:03d}",
                    "featureFamily": "predictive_feature_importance",
                    "candidateDeterminant": str(row["sourceFeature"]),
                    "targetName": str(row["targetName"]),
                    "hypothesis": (
                        f"`{row['sourceFeature']}` is a predictive determinant candidate for `{row['targetName']}` "
                        "in leakage-audited held-out S12 models."
                    ),
                    "evidenceType": "heldout_predictive_association_feature_importance",
                    "meanEffect": float(row["totalImportance"]),
                    "nEvidenceRows": int(row["transformedFeatureCount"]),
                    "supportingModelMetric": metric_lookup.get(str(row["targetName"]), ""),
                    "causalCaveat": CAUSAL_HYPOTHESIS_CAVEAT,
                    "metricBoundaryCaveat": "Feature importance is associational and can distribute across correlated design variables.",
                    "claimBoundary": CLAIM_BOUNDARY,
                    "causalModelVersion": CAUSAL_MODEL_VERSION,
                }
            )
    return pd.DataFrame(rows)


def validation_checks(
    s12_results: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    target_defs: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    split: pd.DataFrame,
    performance: pd.DataFrame,
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    importance: pd.DataFrame,
    hypotheses: pd.DataFrame,
) -> pd.DataFrame:
    train_groups = set(split.loc[split["split"] == "train", "s12SourceS11ConditionId"].astype(str))
    test_groups = set(split.loc[split["split"] == "test", "s12SourceS11ConditionId"].astype(str))
    high_risk_included = leakage_audit[
        leakage_audit["includedInPrimaryModel"].map(bool)
        & leakage_audit["leakageRisk"].astype(str).str.startswith("high_", na=False)
    ]
    exact_flags_included = set(EXACT_REPLAY_FEATURES) <= set(
        leakage_audit.loc[leakage_audit["includedInPrimaryModel"].map(bool), "featureName"].astype(str)
    )
    classification_targets_ok = True
    for target_col in ["finalMorphologyClassTarget", "rescueSuccessProxyTarget", "historySensitiveProxyTarget"]:
        if target_col not in feature_matrix.columns:
            classification_targets_ok = False
            continue
        train_values = feature_matrix.loc[split["split"].to_numpy() == "train", target_col]
        test_values = feature_matrix.loc[split["split"].to_numpy() == "test", target_col]
        classification_targets_ok = classification_targets_ok and train_values.nunique(dropna=True) >= 2 and test_values.nunique(dropna=True) >= 2
    required_targets = set(target_defs["targetName"].astype(str))
    modeled_targets = set(performance["targetName"].astype(str)) if not performance.empty else set()
    causal_caveats_ok = bool(
        not hypotheses.empty
        and hypotheses["causalCaveat"].astype(str).str.contains("not biological causal proof", case=False, na=False).all()
    )
    checks = [
        {
            "checkId": "s12_input_loaded",
            "success": bool(len(s12_results) == 120 and s12_results["runSucceeded"].fillna(False).map(bool).all()),
            "detail": f"{len(s12_results)} S12 rows loaded; run success rate {s12_results['runSucceeded'].fillna(False).map(bool).mean():.3f}",
        },
        {
            "checkId": "feature_leakage_audit_present",
            "success": bool(not leakage_audit.empty and high_risk_included.empty and exact_flags_included),
            "detail": (
                f"{int(leakage_audit['includedInPrimaryModel'].sum())} included features audited; "
                f"{len(high_risk_included)} high-risk outcome features included; exact replay flags included={exact_flags_included}"
            ),
        },
        {
            "checkId": "heldout_group_split",
            "success": bool(train_groups and test_groups and train_groups.isdisjoint(test_groups) and len(test_groups) == 3),
            "detail": f"{len(train_groups)} train source contexts and {len(test_groups)} held-out source contexts with no overlap",
        },
        {
            "checkId": "target_coverage",
            "success": bool(classification_targets_ok and required_targets <= modeled_targets),
            "detail": f"modeled targets={sorted(modeled_targets)}; required targets={sorted(required_targets)}",
        },
        {
            "checkId": "heldout_evaluation_written",
            "success": bool(not performance.empty and not predictions.empty and performance["testCount"].min() > 0),
            "detail": f"{len(performance)} model-performance rows and {len(predictions)} held-out prediction rows written",
        },
        {
            "checkId": "calibration_written",
            "success": bool(not calibration.empty and calibration["calibrationType"].astype(str).nunique() >= 1),
            "detail": f"{len(calibration)} calibration-bin rows written for classification models",
        },
        {
            "checkId": "feature_importance_written",
            "success": bool(not importance.empty and importance["importance"].notna().any()),
            "detail": f"{len(importance)} transformed feature-importance rows written",
        },
        {
            "checkId": "causal_hypotheses_caveated",
            "success": causal_caveats_ok,
            "detail": f"{len(hypotheses)} causal-hypothesis rows; caveats present={causal_caveats_ok}",
        },
        {
            "checkId": "claim_boundaries_retained",
            "success": bool(
                target_defs["claimBoundary"].astype(str).str.contains("Computational", na=False).all()
                and performance["claimBoundary"].astype(str).str.contains("Computational", na=False).all()
            ),
            "detail": "computational claim boundaries retained on target definitions and model performance rows",
        },
    ]
    return pd.DataFrame(checks)


def write_model_plots(performance: pd.DataFrame, aggregate_importance: pd.DataFrame, figure_dir: Path, step_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not performance.empty:
        plot = performance.copy()
        plot["label"] = plot["targetName"].astype(str) + "\n" + plot["modelName"].astype(str).str.replace("_", " ")
        fig, ax = plt.subplots(figsize=(13, 6))
        colors = ["#4c78a8" if task == "classification" else "#f58518" for task in plot["taskType"].astype(str)]
        ax.bar(np.arange(len(plot)), plot["primaryMetricValue"].fillna(0.0), color=colors)
        ax.set_xticks(np.arange(len(plot)), labels=plot["label"], rotation=65, ha="right")
        ax.set_ylabel("Held-out primary metric")
        ax.set_title("E06 S13 held-out predictive model performance")
        fig.tight_layout()
        for path in [
            figure_dir / "e06_s13_model_performance.png",
            figure_dir / "e06_s13_model_performance.pdf",
            step_dir / "model_performance.png",
            step_dir / "model_performance.pdf",
        ]:
            fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
            paths.append(path)
        plt.close(fig)
    if not aggregate_importance.empty:
        plot = (
            aggregate_importance[aggregate_importance["modelName"].astype(str).str.contains("random_forest", na=False)]
            .groupby("sourceFeature", dropna=False)
            .agg(totalImportance=("totalImportance", "sum"))
            .reset_index()
            .sort_values("totalImportance", ascending=False, kind="mergesort")
            .head(18)
        )
        if not plot.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(plot["sourceFeature"], plot["totalImportance"], color="#54a24b")
            ax.invert_yaxis()
            ax.set_xlabel("Aggregate random-forest importance")
            ax.set_title("E06 S13 top leakage-audited predictive features")
            fig.tight_layout()
            for path in [
                figure_dir / "e06_s13_feature_importance.png",
                figure_dir / "e06_s13_feature_importance.pdf",
                step_dir / "feature_importance.png",
                step_dir / "feature_importance.pdf",
            ]:
                fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
                paths.append(path)
            plt.close(fig)
    return paths


def write_markdown_report(
    *,
    step_dir: Path,
    status: Mapping[str, Any],
    artifacts_written: Sequence[str],
    performance: pd.DataFrame,
    aggregate_importance: pd.DataFrame,
    hypotheses: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Path:
    top_models = performance.sort_values(["targetName", "primaryMetricValue"], ascending=[True, False], kind="mergesort")
    top_models = top_models.groupby("targetName", sort=False).head(1)
    model_lines = "\n".join(
        f"- `{row.targetName}`: `{row.modelName}` {row.primaryMetricName}={_safe_float(row.primaryMetricValue):.3f}"
        for row in top_models.itertuples(index=False)
    ) or "- No held-out model results available."
    top_features = (
        aggregate_importance[aggregate_importance["modelName"].astype(str).str.contains("random_forest", na=False)]
        .groupby("sourceFeature", dropna=False)
        .agg(totalImportance=("totalImportance", "sum"))
        .reset_index()
        .sort_values("totalImportance", ascending=False, kind="mergesort")
        .head(8)
    )
    feature_lines = "\n".join(
        f"- `{row.sourceFeature}`: aggregate importance {row.totalImportance:.3f}" for row in top_features.itertuples(index=False)
    ) or "- No feature-importance rows available."
    hypothesis_lines = "\n".join(
        f"- `{row.hypothesisId}` / `{row.candidateDeterminant}` -> `{row.targetName}`: {row.hypothesis}"
        for row in hypotheses.head(8).itertuples(index=False)
    ) or "- No hypotheses available."
    text = f"""# Research Step S13: Build causal/predictive models of final morphology

## Completion status

Research step ID: `S13`. {status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{status['validationResult']}. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}. Held-out split used {status['trainSourceContextCount']} training S12 source contexts and {status['testSourceContextCount']} held-out contexts.

## Caveats or blockers

{LEAKAGE_AUDIT_CAVEAT} {CAUSAL_HYPOTHESIS_CAVEAT} S12 reset/carryover comparisons are policy-state history probes because the selected contexts had no E04 memory-policy rows. {CLAIM_BOUNDARY}

## Lay summary

S13 trained leakage-audited held-out models to predict final morphology, rescue/history sensitivity, target quality, dominance, and Aggregation from policy, governance, interface, graft, mutant, and history descriptors. It also compared reset/carryover and timing protocols within matched S12 contexts to generate cautious intervention hypotheses.

## Held-out model anchors

{model_lines}

## Top audited predictors

{feature_lines}

## Causal-hypothesis candidates

{hypothesis_lines}

## Recommended next action

{status['recommendedNextAction']}
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_s13_causal_predictive_models(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    s12_results_path: Path = DEFAULT_S12_RESULTS_PATH,
    group_split_seed: int = GROUP_SPLIT_SEED,
) -> dict[str, Any]:
    artifacts_dir = artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e06"
    provenance_dir = artifacts_dir / "provenance"
    for path in [step_dir, results_dir, figures_dir, provenance_dir]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    s12_results = load_s12_history_results(s12_results_path)
    feature_matrix, targets, leakage_audit, feature_cols, categorical, numeric, boolean = build_feature_matrix(s12_results)
    split = make_group_holdout_split(feature_matrix, random_state=group_split_seed)
    feature_matrix = feature_matrix.merge(split[["conditionId", "split"]], on="conditionId", how="left")
    performance, predictions, calibration, importance = fit_predictive_models(
        feature_matrix,
        targets,
        split,
        feature_cols,
        categorical,
        numeric,
        boolean,
        random_state=group_split_seed,
    )
    aggregate_importance = aggregate_feature_importance(importance)
    contrasts = paired_protocol_contrasts(derive_s13_fields(s12_results))
    hypotheses = causal_hypothesis_table(contrasts, aggregate_importance, performance)
    validation_df = validation_checks(
        s12_results,
        feature_matrix,
        targets,
        leakage_audit,
        split,
        performance,
        predictions,
        calibration,
        importance,
        hypotheses,
    )

    matrix_csv = step_dir / "model_feature_matrix.csv"
    matrix_parquet = step_dir / "model_feature_matrix.parquet"
    target_csv = step_dir / "model_target_definitions.csv"
    target_parquet = step_dir / "model_target_definitions.parquet"
    audit_csv = step_dir / "feature_leakage_audit.csv"
    audit_parquet = step_dir / "feature_leakage_audit.parquet"
    split_csv = step_dir / "heldout_split.csv"
    split_parquet = step_dir / "heldout_split.parquet"
    performance_csv = step_dir / "model_performance.csv"
    performance_parquet = step_dir / "model_performance.parquet"
    predictions_csv = step_dir / "model_predictions.csv"
    predictions_parquet = step_dir / "model_predictions.parquet"
    calibration_csv = step_dir / "calibration_summary.csv"
    calibration_parquet = step_dir / "calibration_summary.parquet"
    importance_csv = step_dir / "feature_importance.csv"
    importance_parquet = step_dir / "feature_importance.parquet"
    aggregate_importance_csv = step_dir / "feature_importance_by_source.csv"
    aggregate_importance_parquet = step_dir / "feature_importance_by_source.parquet"
    contrast_csv = step_dir / "paired_protocol_contrasts.csv"
    contrast_parquet = step_dir / "paired_protocol_contrasts.parquet"
    hypothesis_csv = step_dir / "causal_hypothesis_table.csv"
    hypothesis_parquet = step_dir / "causal_hypothesis_table.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    result_csv = results_dir / "e06_causal_predictive_models.csv"
    result_parquet = results_dir / "e06_causal_predictive_models.parquet"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    model_summary = performance.copy()
    if not model_summary.empty:
        model_summary["resultTable"] = "model_performance"
        model_summary["featureCount"] = int(len(feature_cols))
        model_summary["trainSourceContextCount"] = int(split.loc[split["split"] == "train", "s12SourceS11ConditionId"].nunique())
        model_summary["testSourceContextCount"] = int(split.loc[split["split"] == "test", "s12SourceS11ConditionId"].nunique())
        model_summary["featureLeakageAuditPassed"] = bool(validation_df.loc[validation_df["checkId"] == "feature_leakage_audit_present", "success"].iloc[0])

    for path, df in [
        (matrix_csv, compact_result_csv(feature_matrix)),
        (target_csv, targets),
        (audit_csv, leakage_audit),
        (split_csv, split),
        (performance_csv, performance),
        (predictions_csv, predictions),
        (calibration_csv, calibration),
        (importance_csv, importance),
        (aggregate_importance_csv, aggregate_importance),
        (contrast_csv, contrasts),
        (hypothesis_csv, hypotheses),
        (validation_csv, validation_df),
        (result_csv, model_summary),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (matrix_parquet, feature_matrix),
        (target_parquet, targets),
        (audit_parquet, leakage_audit),
        (split_parquet, split),
        (performance_parquet, performance),
        (predictions_parquet, predictions),
        (calibration_parquet, calibration),
        (importance_parquet, importance),
        (aggregate_importance_parquet, aggregate_importance),
        (contrast_parquet, contrasts),
        (hypothesis_parquet, hypotheses),
        (validation_parquet, validation_df),
        (result_parquet, model_summary),
    ]:
        df.to_parquet(path, index=False)

    figure_paths = write_model_plots(performance, aggregate_importance, figures_dir, step_dir)
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() else "failed"
    outcome = "supportive" if validation_result == "passed" and not hypotheses.empty else "null"
    artifacts = [
        matrix_csv,
        matrix_parquet,
        target_csv,
        target_parquet,
        audit_csv,
        audit_parquet,
        split_csv,
        split_parquet,
        performance_csv,
        performance_parquet,
        predictions_csv,
        predictions_parquet,
        calibration_csv,
        calibration_parquet,
        importance_csv,
        importance_parquet,
        aggregate_importance_csv,
        aggregate_importance_parquet,
        contrast_csv,
        contrast_parquet,
        hypothesis_csv,
        hypothesis_parquet,
        validation_csv,
        validation_parquet,
        result_csv,
        result_parquet,
        *figure_paths,
        step_dir / "summary.md",
        status_path,
        manifest_path,
        run_manifest_path,
    ]
    train_context_count = int(split.loc[split["split"] == "train", "s12SourceS11ConditionId"].nunique())
    test_context_count = int(split.loc[split["split"] == "test", "s12SourceS11ConditionId"].nunique())
    best_models = (
        performance.sort_values(["targetName", "primaryMetricValue"], ascending=[True, False], kind="mergesort")
        .groupby("targetName", sort=False)
        .head(1)
    )
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_result == "passed",
        "status": "completed" if validation_result == "passed" else "completed_with_validation_failure",
        "artifactsWritten": [str(path) for path in artifacts],
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            LEAKAGE_AUDIT_CAVEAT,
            CAUSAL_HYPOTHESIS_CAVEAT,
            "S13 is a derived analysis of selected S12 contexts and does not add new simulation interventions.",
            "Upstream S10/S11 context labels are audited predictive context proxies, not direct S12 causal variables.",
            "S12 reset/carryover comparisons are policy-state history probes because selected contexts had no E04 memory-policy rows.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then S14 should use S13 predictors and paired protocol hypotheses to design "
            "minimal interventions with paired no-intervention controls and held-out validation; do not start S14 until review is complete."
        ),
        "laySummary": (
            "S13 trained leakage-audited held-out models on S12 history rows and produced cautious predictive determinants "
            "plus paired computational causal-hypothesis candidates for final morphology and related proxy scores."
        ),
        "outcomeClassification": outcome,
        "conditionCount": int(len(s12_results)),
        "featureCount": int(len(feature_cols)),
        "targetCount": int(len(targets)),
        "modelCount": int(len(performance)),
        "predictionRowCount": int(len(predictions)),
        "calibrationRowCount": int(len(calibration)),
        "featureImportanceRowCount": int(len(importance)),
        "causalHypothesisCount": int(len(hypotheses)),
        "pairedContrastCount": int(len(contrasts)),
        "trainSourceContextCount": train_context_count,
        "testSourceContextCount": test_context_count,
        "groupSplitSeed": int(group_split_seed),
        "bestHeldoutModels": [
            {
                "targetName": str(row.targetName),
                "modelName": str(row.modelName),
                "primaryMetricName": str(row.primaryMetricName),
                "primaryMetricValue": _safe_float(row.primaryMetricValue),
            }
            for row in best_models.itertuples(index=False)
        ],
        "workerCount": 1,
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
        "newDependenciesInstalled": [],
        "pythonPackages": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "sourceCodeLocation": str(repo_root),
        "sourceCodeArtifactPolicy": "repository-backed source is committed to git; source files are not copied into artifacts per workspace instructions",
    }
    summary_path = write_markdown_report(
        step_dir=step_dir,
        status=status,
        artifacts_written=[str(path) for path in artifacts],
        performance=performance,
        aggregate_importance=aggregate_importance,
        hypotheses=hypotheses,
        validation_df=validation_df,
    )
    if summary_path not in artifacts:
        artifacts.append(summary_path)
    status["artifactsWritten"] = [str(path) for path in artifacts]
    write_json(status_path, status)

    manifest_payload = {
        "schema": "eidosoma.step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "causalModelVersion": CAUSAL_MODEL_VERSION,
        "inputArtifacts": {
            "s12Results": str(s12_results_path),
        },
        "featureColumns": list(feature_cols),
        "targetDefinitions": targets.to_dict(orient="records"),
        "heldoutSplit": {
            "groupColumn": "s12SourceS11ConditionId",
            "randomState": int(group_split_seed),
            "trainSourceContextCount": train_context_count,
            "testSourceContextCount": test_context_count,
        },
        "artifacts": artifact_records(artifacts),
    }
    write_json(manifest_path, manifest_payload)

    existing_manifest: dict[str, Any] = {}
    if run_manifest_path.exists():
        try:
            existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = {}
    research_steps = dict(existing_manifest.get("researchSteps", {}))
    research_steps[STEP_ID] = status
    run_manifest = {
        **existing_manifest,
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": "E06",
        "lastResearchStepId": STEP_ID,
        "lastStepNumber": STEP_NUMBER,
        "updatedAt": completed_at,
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
            "commit": git_value(["rev-parse", "HEAD"], repo_root),
            "dirtyStatus": git_value(["status", "--short"], repo_root),
            "remote": git_value(["remote", "-v"], repo_root),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
                if os.environ.get(key) is not None
            },
            "newDependenciesInstalled": [],
        },
        "researchSteps": research_steps,
        "artifacts": artifact_records(artifacts),
    }
    write_json(run_manifest_path, run_manifest)
    manifest_payload["artifacts"] = artifact_records(artifacts)
    write_json(manifest_path, manifest_payload)
    return {
        "status": status,
        "featureMatrix": feature_matrix,
        "targetDefinitions": targets,
        "featureLeakageAudit": leakage_audit,
        "split": split,
        "modelPerformance": performance,
        "modelPredictions": predictions,
        "calibrationSummary": calibration,
        "featureImportance": importance,
        "featureImportanceBySource": aggregate_importance,
        "pairedProtocolContrasts": contrasts,
        "causalHypotheses": hypotheses,
        "validation": validation_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E06 S13 causal/predictive models")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--s12-results-path", type=Path, default=DEFAULT_S12_RESULTS_PATH)
    parser.add_argument("--group-split-seed", type=int, default=GROUP_SPLIT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_s13_causal_predictive_models(
        artifacts_dir=args.artifacts_dir,
        s12_results_path=args.s12_results_path,
        group_split_seed=args.group_split_seed,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['modelCount']} models over {status['conditionCount']} S12 rows, "
        f"{status['causalHypothesisCount']} hypotheses, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
