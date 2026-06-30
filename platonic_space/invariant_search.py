from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .behavior_predictor import DEFAULT_TARGET_COLUMNS
from .policy_catalog import safe_json_loads
from .policy_embeddings import target_error_weights
from .world_schema import compact_json


INVARIANT_SEARCH_SCHEMA_VERSION = "e07_s09_invariant_search.v1"
INVARIANT_SEARCH_MODEL_VERSION = "e07_s09_cross_world_interpretable_ridge.v1"
INVARIANT_SEARCH_CLAIM_BOUNDARY = (
    "Empirical invariant search over S04-S08 simulation-derived records, catalogs, embeddings, surrogate error "
    "limits, and Platonic distances only. Associations are computational regularities within toy simulated worlds; "
    "they are not causal proof and do not validate biological morphogenesis, cognition, agency, clinical behavior, "
    "sentience, or living chimeras."
)

CONFOUND_COLUMNS = (
    "sourceExperimentId",
    "sourceKind",
    "policyFamily",
    "policyKind",
    "policyRepresentationType",
    "policyStochasticity",
    "worldFamily",
    "substrateClass",
    "goalFamily",
    "goalKind",
    "goalRepresentationType",
)

INVARIANT_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "memory_feedback": (
        "inv__policy__memory_depth",
        "inv__policy__has_internal_memory",
        "inv__policy__feedback_strength",
        "inv__policy__automaton_available",
        "inv__policy__neural_metadata_available",
    ),
    "locality_signaling": (
        "inv__policy__signaling_horizon",
        "inv__policy__locality_horizon",
        "inv__policy__local_observation_only",
        "inv__policy__global_information_proxy",
    ),
    "target_position_knowledge": (
        "inv__policy__target_position_knowledge",
        "inv__goal__target_dimensionality",
    ),
    "stochasticity_and_policy_form": (
        "inv__policy__stochasticity_flag",
        "inv__policy__dsl_available",
        "inv__policy__decision_graph_available",
    ),
    "policy_complexity_action_space": (
        "inv__policy__complexity_score",
        "inv__policy__action_repertoire_size",
        "inv__policy__observation_repertoire_size",
        "inv__policy__source_row_count_log1p",
    ),
    "world_substrate_challenge": (
        "inv__world__substrate_dimension",
        "inv__world__perturbation_intensity",
        "inv__world__scheduler_random",
        "inv__world__repair_challenge",
        "inv__world__chimeric_or_conflict",
        "inv__world__analysis_layer",
        "inv__world__source_record_count_log1p",
    ),
    "goal_demand_observability": (
        "inv__goal__local_observable",
        "inv__goal__metric_proxy",
        "inv__goal__benchmark_goal",
        "inv__goal__repair_or_homeostasis",
        "inv__goal__aggregation_or_symmetry",
    ),
    "s08_platonic_position": (
        "inv__policy__s08_mean_distance",
        "inv__policy__s08_nearest_distance",
        "inv__policy__s08_distance_std",
        "inv__goal__s08_mean_distance",
        "inv__goal__s08_nearest_distance",
        "inv__goal__s08_distance_std",
        "inv__world__s08_mean_distance",
        "inv__world__s08_nearest_distance",
        "inv__world__s08_distance_std",
    ),
    "coverage_and_missingness_controls": (
        "inv__policy__s08_sparse_penalty",
        "inv__goal__s08_sparse_penalty",
        "inv__world__s08_sparse_penalty",
        "inv__policy__s08_row_count_log1p",
        "inv__goal__s08_row_count_log1p",
        "inv__world__s08_row_count_log1p",
    ),
}


def invariant_feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith("inv__"))


def _json(value: Any, default: Any = None) -> Any:
    parsed = safe_json_loads(value, default)
    return default if parsed is None else parsed


def _json_text(value: Any) -> str:
    parsed = _json(value, value)
    if isinstance(parsed, (dict, list, tuple)):
        return compact_json(parsed)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(parsed)


def _safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _label(value: Any) -> str:
    text = _safe_text(value).strip()
    return text if text and text.lower() != "nan" else "unknown"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _indicator(text: str, patterns: Sequence[str]) -> float:
    lowered = text.lower()
    return 1.0 if any(pattern in lowered for pattern in patterns) else 0.0


def _json_len(value: Any) -> int:
    parsed = _json(value, [])
    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("actions"), list):
            return len(parsed["actions"])
        return len(parsed)
    return 0


def _actions_len(value: Any) -> int:
    parsed = _json(value, {})
    if isinstance(parsed, dict) and isinstance(parsed.get("actions"), list):
        return len(parsed["actions"])
    if isinstance(parsed, list):
        return len(parsed)
    return 0


def _dimensionality_proxy(value: Any) -> float:
    text = _json_text(value).lower()
    if any(token in text for token in ("3d", "volume")):
        return 3.0
    if any(token in text for token in ("2d", "grid", "graph", "morphology", "boundary", "shape")):
        return 2.0
    if any(token in text for token in ("1d", "row", "mixed_1d")):
        return 1.0
    if "abstract" in text or "panel" in text:
        return 0.5
    return 0.0


def s08_entity_centrality(distances: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entity_type, subset in distances.groupby("entityType", sort=True):
        long_a = subset[["entityIdA", "platonicDistance"]].rename(columns={"entityIdA": "entityId"})
        long_b = subset[["entityIdB", "platonicDistance"]].rename(columns={"entityIdB": "entityId"})
        long = pd.concat([long_a, long_b], ignore_index=True)
        for entity_id, values in long.groupby("entityId")["platonicDistance"]:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            if numeric.empty:
                continue
            rows.append(
                {
                    "entityType": entity_type,
                    "entityId": str(entity_id),
                    "s08MeanDistance": float(numeric.mean()),
                    "s08MedianDistance": float(numeric.median()),
                    "s08NearestDistance": float(numeric.min()),
                    "s08DistanceStd": float(numeric.std(ddof=0)) if len(numeric) > 1 else 0.0,
                    "s08DistanceP90": float(numeric.quantile(0.90)),
                    "s08NeighborCount": int(len(numeric)),
                }
            )
    centrality = pd.DataFrame(rows)
    if entities.empty:
        return centrality
    keep = ["entityType", "entityId", "sparseCoveragePenalty", "distanceUncertaintyLevel", "rowCount", "observedTargetCount"]
    entity_meta = entities[[column for column in keep if column in entities.columns]].copy()
    return centrality.merge(entity_meta, on=["entityType", "entityId"], how="left")


def policy_invariant_table(policy_catalog: pd.DataFrame, centrality: pd.DataFrame) -> pd.DataFrame:
    s08 = centrality[centrality["entityType"].eq("policy")].set_index("entityId", drop=False)
    rows: list[dict[str, Any]] = []
    for record in policy_catalog.to_dict(orient="records"):
        policy_id = str(record["abstractPolicyId"])
        representation_text = " ".join(
            [
                _json_text(record.get("representationJson")),
                _json_text(record.get("observationRequirementsJson")),
                _safe_text(record.get("policyLabel")),
                _safe_text(record.get("policyFamily")),
                _safe_text(record.get("policyKind")),
                _safe_text(record.get("representationType")),
                _safe_text(record.get("lineageJson")),
            ]
        )
        stochastic_text = " ".join([_safe_text(record.get("stochasticity")), representation_text])
        memory_depth = _float(record.get("memoryDepth"))
        signaling_horizon = _float(record.get("signalingHorizon"))
        target_knowledge = _indicator(
            representation_text,
            ("target_position", "position_compare", "selection", "global target", "rank", "oracle", "target index"),
        )
        global_info = _indicator(representation_text, ("global", "oracle", "centralized", "rank"))
        feedback = _indicator(
            representation_text,
            ("feedback", "learning", "adaptive", "homeostasis", "signal", "emit", "diffuse", "governance", "intervention", "repair", "fatigue"),
        )
        stochastic = 0.0 if "deterministic" in stochastic_text.lower() else _indicator(stochastic_text, ("stochastic", "random", "probability", "resampling", "mutat"))
        local_observation = _indicator(_json_text(record.get("observationRequirementsJson")), ("own", "neighbor", "local", "left", "right"))
        s08_row = s08.loc[policy_id].to_dict() if policy_id in s08.index else {}
        rows.append(
            {
                "abstractPolicyId": policy_id,
                "inv__policy__memory_depth": memory_depth,
                "inv__policy__has_internal_memory": float(memory_depth > 0 or "memory" in representation_text.lower()),
                "inv__policy__signaling_horizon": signaling_horizon,
                "inv__policy__feedback_strength": float(feedback + min(memory_depth, 3.0) / 3.0 + min(signaling_horizon, 3.0) / 3.0),
                "inv__policy__target_position_knowledge": target_knowledge,
                "inv__policy__global_information_proxy": global_info,
                "inv__policy__locality_horizon": float(1.0 + signaling_horizon + (2.0 * target_knowledge) + (3.0 * global_info)),
                "inv__policy__local_observation_only": float(local_observation and not target_knowledge and not global_info),
                "inv__policy__stochasticity_flag": stochastic,
                "inv__policy__complexity_score": _float(record.get("complexityScore"), default=0.0),
                "inv__policy__action_repertoire_size": float(_actions_len(record.get("actionSpaceJson"))),
                "inv__policy__observation_repertoire_size": float(_json_len(record.get("observationRequirementsJson"))),
                "inv__policy__dsl_available": float(bool(record.get("dslAvailable"))),
                "inv__policy__automaton_available": float(bool(record.get("automatonAvailable"))),
                "inv__policy__decision_graph_available": float(bool(record.get("decisionGraphAvailable"))),
                "inv__policy__neural_metadata_available": float(bool(record.get("neuralMetadataAvailable"))),
                "inv__policy__source_row_count_log1p": float(np.log1p(_float(record.get("sourceRowCount"), default=0.0))),
                "inv__policy__s08_mean_distance": _float(s08_row.get("s08MeanDistance"), default=0.0),
                "inv__policy__s08_nearest_distance": _float(s08_row.get("s08NearestDistance"), default=0.0),
                "inv__policy__s08_distance_std": _float(s08_row.get("s08DistanceStd"), default=0.0),
                "inv__policy__s08_sparse_penalty": _float(s08_row.get("sparseCoveragePenalty"), default=1.0),
                "inv__policy__s08_row_count_log1p": float(np.log1p(_float(s08_row.get("rowCount"), default=0.0))),
            }
        )
    return pd.DataFrame(rows)


def goal_invariant_table(goal_catalog: pd.DataFrame, centrality: pd.DataFrame) -> pd.DataFrame:
    s08 = centrality[centrality["entityType"].eq("goal")].set_index("entityId", drop=False)
    rows: list[dict[str, Any]] = []
    for record in goal_catalog.to_dict(orient="records"):
        goal_id = str(record["abstractGoalId"])
        text = " ".join(
            [
                _safe_text(record.get("goalFamily")),
                _safe_text(record.get("goalKind")),
                _safe_text(record.get("representationType")),
                _safe_text(record.get("localObservability")),
                _json_text(record.get("constraintOrEnergyJson")),
            ]
        ).lower()
        local_text = _safe_text(record.get("localObservability")).lower()
        s08_row = s08.loc[goal_id].to_dict() if goal_id in s08.index else {}
        rows.append(
            {
                "abstractGoalId": goal_id,
                "inv__goal__local_observable": float("local" in local_text and "global baseline" not in local_text and "not usually" not in local_text),
                "inv__goal__metric_proxy": _indicator(text, ("metric_proxy", "taxonomy_proxy", "compatibility", "report-facing", "proxy")),
                "inv__goal__benchmark_goal": _indicator(text, ("benchmark",)),
                "inv__goal__target_dimensionality": _dimensionality_proxy(record.get("targetDimensionality")),
                "inv__goal__repair_or_homeostasis": _indicator(text, ("repair", "regeneration", "homeostasis", "boundary", "recover", "restore")),
                "inv__goal__aggregation_or_symmetry": _indicator(text, ("aggregation", "symmetry", "dispersion")),
                "inv__goal__s08_mean_distance": _float(s08_row.get("s08MeanDistance"), default=0.0),
                "inv__goal__s08_nearest_distance": _float(s08_row.get("s08NearestDistance"), default=0.0),
                "inv__goal__s08_distance_std": _float(s08_row.get("s08DistanceStd"), default=0.0),
                "inv__goal__s08_sparse_penalty": _float(s08_row.get("sparseCoveragePenalty"), default=1.0),
                "inv__goal__s08_row_count_log1p": float(np.log1p(_float(s08_row.get("rowCount"), default=0.0))),
            }
        )
    return pd.DataFrame(rows)


def world_invariant_table(world_catalog: pd.DataFrame, centrality: pd.DataFrame) -> pd.DataFrame:
    s08 = centrality[centrality["entityType"].eq("world")].set_index("entityId", drop=False)
    rows: list[dict[str, Any]] = []
    for record in world_catalog.to_dict(orient="records"):
        world_id = str(record["worldId"])
        text = " ".join(
            [
                _safe_text(record.get("worldFamily")),
                _safe_text(record.get("substrateClass")),
                _json_text(record.get("stateSpace")),
                _json_text(record.get("perturbationModel")),
                _json_text(record.get("scheduler")),
                _json_text(record.get("transitionRules")),
            ]
        ).lower()
        s08_row = s08.loc[world_id].to_dict() if world_id in s08.index else {}
        perturb_words = ("frozen", "damage", "repair", "chimera", "conflict", "perturb", "regeneration", "mutant", "graft", "scrambled", "noise")
        perturb_score = sum(1.0 for word in perturb_words if word in text)
        rows.append(
            {
                "worldId": world_id,
                "inv__world__substrate_dimension": _dimensionality_proxy(record.get("substrateClass")),
                "inv__world__perturbation_intensity": float(perturb_score + min(_float(record.get("sourceRecordCount"), default=0.0), 500.0) / 500.0),
                "inv__world__scheduler_random": _indicator(_json_text(record.get("scheduler")), ("random", "asynchronous", "stochastic", "rate_weighted", "resampling")),
                "inv__world__repair_challenge": _indicator(text, ("repair", "defect", "frozen", "damage", "regeneration", "robustness", "homeostasis", "recover")),
                "inv__world__chimeric_or_conflict": _indicator(text, ("chimera", "mixed", "governance", "conflict", "graft", "mutant")),
                "inv__world__analysis_layer": float(_safe_text(record.get("replayability")) in {"derived", "summary_only"} or _safe_text(record.get("metadataCompleteness")) != "complete"),
                "inv__world__source_record_count_log1p": float(np.log1p(_float(record.get("sourceRecordCount"), default=0.0))),
                "inv__world__s08_mean_distance": _float(s08_row.get("s08MeanDistance"), default=0.0),
                "inv__world__s08_nearest_distance": _float(s08_row.get("s08NearestDistance"), default=0.0),
                "inv__world__s08_distance_std": _float(s08_row.get("s08DistanceStd"), default=0.0),
                "inv__world__s08_sparse_penalty": _float(s08_row.get("sparseCoveragePenalty"), default=1.0),
                "inv__world__s08_row_count_log1p": float(np.log1p(_float(s08_row.get("rowCount"), default=0.0))),
            }
        )
    return pd.DataFrame(rows)


def build_invariant_feature_frame(
    frame: pd.DataFrame,
    policy_catalog: pd.DataFrame,
    goal_catalog: pd.DataFrame,
    world_catalog: pd.DataFrame,
    s08_distances: pd.DataFrame,
    s08_entities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    centrality = s08_entity_centrality(s08_distances, s08_entities)
    policy_features = policy_invariant_table(policy_catalog, centrality)
    goal_features = goal_invariant_table(goal_catalog, centrality)
    world_features = world_invariant_table(world_catalog, centrality)

    keep_columns = [
        "behaviorRecordId",
        "sourceExperimentId",
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
        *DEFAULT_TARGET_COLUMNS,
    ]
    output = frame[[column for column in keep_columns if column in frame.columns]].copy()
    output = output.merge(policy_features, left_on="primaryPolicyId", right_on="abstractPolicyId", how="left")
    output = output.merge(goal_features, on="abstractGoalId", how="left")
    output = output.merge(world_features, on="worldId", how="left")
    for column in invariant_feature_columns(output):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    for column in CONFOUND_COLUMNS:
        if column not in output.columns:
            output[column] = "unknown"
        output[column] = output[column].map(_label)
    output["s09FeatureFrameVersion"] = INVARIANT_SEARCH_SCHEMA_VERSION
    return output, centrality


def target_eligibility(frame: pd.DataFrame, target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS, min_rows: int = 500, min_worlds: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in target_columns:
        if target not in frame.columns:
            rows.append({"target": target, "availableRows": 0, "availableWorlds": 0, "eligible": False, "reason": "missing_column"})
            continue
        mask = pd.to_numeric(frame[target], errors="coerce").notna()
        available_rows = int(mask.sum())
        available_worlds = int(frame.loc[mask, "worldId"].nunique()) if available_rows else 0
        eligible = available_rows >= min_rows and available_worlds >= min_worlds
        reason = "eligible" if eligible else f"requires_at_least_{min_rows}_rows_and_{min_worlds}_worlds"
        rows.append({"target": target, "availableRows": available_rows, "availableWorlds": available_worlds, "eligible": eligible, "reason": reason})
    return pd.DataFrame(rows)


def _group_for_feature(column: str) -> str:
    for group, prefixes in INVARIANT_FEATURE_GROUPS.items():
        if any(column.startswith(prefix) for prefix in prefixes):
            return group
    return "other_invariant_features"


def _columns_for_group(columns: Sequence[str], group: str) -> list[str]:
    prefixes = INVARIANT_FEATURE_GROUPS.get(group, ())
    return [column for column in columns if any(column.startswith(prefix) for prefix in prefixes)]


def _make_pipeline(numeric_columns: Sequence[str], categorical_columns: Sequence[str]) -> Pipeline:
    transformers: list[tuple[str, Any, Sequence[str]]] = []
    if numeric_columns:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(numeric_columns)))
    if categorical_columns:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), list(categorical_columns)))
    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required")
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers, sparse_threshold=0.3)),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str] = (),
) -> tuple[np.ndarray, Pipeline, pd.DataFrame]:
    model = _make_pipeline(numeric_columns, categorical_columns)
    y_train = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
    model.fit(train[list(numeric_columns) + list(categorical_columns)], y_train)
    pred = model.predict(test[list(numeric_columns) + list(categorical_columns)])
    coefs = coefficient_table(model, numeric_columns, categorical_columns, target)
    return pred, model, coefs


def coefficient_table(model: Pipeline, numeric_columns: Sequence[str], categorical_columns: Sequence[str], target: str) -> pd.DataFrame:
    preprocess = model.named_steps["preprocess"]
    names = preprocess.get_feature_names_out()
    coef = np.asarray(model.named_steps["ridge"].coef_, dtype=float).reshape(-1)
    rows: list[dict[str, Any]] = []
    for name, value in zip(names, coef):
        clean = re.sub(r"^(num|cat)__", "", str(name))
        is_invariant = clean.startswith("inv__")
        rows.append(
            {
                "target": target,
                "featureName": clean,
                "featureGroup": _group_for_feature(clean) if is_invariant else "confound_metadata",
                "isInvariantFeature": bool(is_invariant),
                "coefficient": float(value),
                "absCoefficient": float(abs(value)),
            }
        )
    return pd.DataFrame(rows)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"r2": math.nan, "rmse": math.nan, "mae": math.nan}
    return {
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else math.nan,
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def cross_world_holdout_search(
    feature_frame: pd.DataFrame,
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
    max_targets: int | None = None,
    n_splits: int = 5,
    min_rows: int = 500,
    min_worlds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligibility = target_eligibility(feature_frame, target_columns, min_rows=min_rows, min_worlds=min_worlds)
    eligible_targets = eligibility[eligibility["eligible"]]["target"].astype(str).tolist()
    if max_targets is not None:
        eligible_targets = eligible_targets[:max_targets]
    inv_columns = invariant_feature_columns(feature_frame)
    cat_columns = [column for column in CONFOUND_COLUMNS if column in feature_frame.columns]

    metric_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    coef_rows: list[pd.DataFrame] = []
    for target in eligible_targets:
        subset = feature_frame[pd.to_numeric(feature_frame[target], errors="coerce").notna()].copy()
        subset[target] = pd.to_numeric(subset[target], errors="coerce")
        world_count = int(subset["worldId"].nunique())
        folds = GroupKFold(n_splits=min(n_splits, world_count))
        for fold, (train_idx, test_idx) in enumerate(folds.split(subset, groups=subset["worldId"]), start=1):
            train = subset.iloc[train_idx].copy()
            test = subset.iloc[test_idx].copy()
            y_test = test[target].to_numpy(dtype=float)
            mean_pred = np.full(len(test), float(train[target].mean()), dtype=float)
            for model_name, pred, coefs in [
                ("global_mean", mean_pred, pd.DataFrame()),
            ]:
                row = _metric_row(target, fold, model_name, train, test, y_test, pred)
                metric_rows.append(row)

            for model_name, numeric, categorical in [
                ("invariant_feature_ridge", inv_columns, ()),
                ("confound_metadata_ridge", (), cat_columns),
                ("combined_ridge", inv_columns, cat_columns),
            ]:
                pred, _model, coefs = _fit_predict(train, test, target, numeric, categorical)
                metric_rows.append(_metric_row(target, fold, model_name, train, test, y_test, pred))
                if not coefs.empty:
                    coefs.insert(0, "fold", fold)
                    coef_rows.append(coefs)

            full_pred, _full_model, _full_coefs = _fit_predict(train, test, target, inv_columns, cat_columns)
            full_r2 = _metrics(y_test, full_pred)["r2"]
            for group in INVARIANT_FEATURE_GROUPS:
                keep_inv = [column for column in inv_columns if column not in set(_columns_for_group(inv_columns, group))]
                if len(keep_inv) == len(inv_columns):
                    continue
                pred, _model, _coefs = _fit_predict(train, test, target, keep_inv, cat_columns)
                r2 = _metrics(y_test, pred)["r2"]
                ablation_rows.append(
                    {
                        "validationTask": "cross_world_group_ablation",
                        "target": target,
                        "fold": int(fold),
                        "featureGroup": group,
                        "fullCombinedR2": full_r2,
                        "ablatedCombinedR2": r2,
                        "deltaR2WhenRemoved": float(full_r2 - r2) if math.isfinite(full_r2) and math.isfinite(r2) else math.nan,
                        "trainRows": int(len(train)),
                        "testRows": int(len(test)),
                        "heldoutWorldCount": int(test["worldId"].nunique()),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    ablations = pd.DataFrame(ablation_rows)
    coefficients = pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame()
    return metrics, ablations, coefficients, eligibility


def _metric_row(target: str, fold: int, model_name: str, train: pd.DataFrame, test: pd.DataFrame, y_test: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    values = _metrics(y_test, pred)
    return {
        "validationTask": "cross_world_holdout",
        "target": target,
        "fold": int(fold),
        "modelName": model_name,
        "trainRows": int(len(train)),
        "testRows": int(len(test)),
        "trainWorldCount": int(train["worldId"].nunique()),
        "heldoutWorldCount": int(test["worldId"].nunique()),
        "r2": values["r2"],
        "rmse": values["rmse"],
        "mae": values["mae"],
    }


def summarize_cross_world_metrics(metrics: pd.DataFrame, target_weights: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weight_lookup = target_weights.set_index("target") if not target_weights.empty and "target" in target_weights.columns else pd.DataFrame()
    for (target, model_name), subset in metrics.groupby(["target", "modelName"], sort=True):
        r2 = pd.to_numeric(subset["r2"], errors="coerce")
        rmse = pd.to_numeric(subset["rmse"], errors="coerce")
        mae = pd.to_numeric(subset["mae"], errors="coerce")
        se = float(r2.std(ddof=1) / math.sqrt(r2.notna().sum())) if r2.notna().sum() > 1 else math.nan
        rows.append(
            {
                "target": target,
                "modelName": model_name,
                "foldCount": int(len(subset)),
                "meanR2": float(r2.mean()) if r2.notna().any() else math.nan,
                "r2Std": float(r2.std(ddof=1)) if r2.notna().sum() > 1 else math.nan,
                "r2Ci95HalfWidth": float(1.96 * se) if math.isfinite(se) else math.nan,
                "meanRmse": float(rmse.mean()) if rmse.notna().any() else math.nan,
                "meanMae": float(mae.mean()) if mae.notna().any() else math.nan,
                "totalTestRows": int(subset["testRows"].sum()),
                "meanHeldoutWorldCount": float(subset["heldoutWorldCount"].mean()),
                "s05ReliabilityWeight": _float(weight_lookup.loc[target, "s05ReliabilityWeight"], default=math.nan) if not weight_lookup.empty and target in weight_lookup.index else math.nan,
            }
        )
    return pd.DataFrame(rows)


def model_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, subset in summary.groupby("target", sort=True):
        values = subset.set_index("modelName")["meanR2"].to_dict()
        rows.append(
            {
                "target": target,
                "globalMeanR2": values.get("global_mean", math.nan),
                "invariantRidgeR2": values.get("invariant_feature_ridge", math.nan),
                "confoundRidgeR2": values.get("confound_metadata_ridge", math.nan),
                "combinedRidgeR2": values.get("combined_ridge", math.nan),
                "invariantMinusGlobalR2": values.get("invariant_feature_ridge", math.nan) - values.get("global_mean", math.nan),
                "combinedMinusGlobalR2": values.get("combined_ridge", math.nan) - values.get("global_mean", math.nan),
                "combinedMinusConfoundR2": values.get("combined_ridge", math.nan) - values.get("confound_metadata_ridge", math.nan),
                "invariantMinusConfoundR2": values.get("invariant_feature_ridge", math.nan) - values.get("confound_metadata_ridge", math.nan),
            }
        )
    return pd.DataFrame(rows)


def summarize_group_ablation(ablations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if ablations.empty:
        return pd.DataFrame()
    for group, subset in ablations.groupby("featureGroup", sort=True):
        deltas = pd.to_numeric(subset["deltaR2WhenRemoved"], errors="coerce").dropna()
        se = float(deltas.std(ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else math.nan
        rows.append(
            {
                "featureGroup": group,
                "validationTask": "cross_world_combined_model_group_ablation",
                "evaluatedFoldTargets": int(len(deltas)),
                "meanDeltaR2WhenRemoved": float(deltas.mean()) if len(deltas) else math.nan,
                "medianDeltaR2WhenRemoved": float(deltas.median()) if len(deltas) else math.nan,
                "deltaR2Ci95HalfWidth": float(1.96 * se) if math.isfinite(se) else math.nan,
                "positiveDeltaFraction": float((deltas > 0).mean()) if len(deltas) else math.nan,
                "targetsWithPositiveMeanDelta": int(
                    (subset.groupby("target")["deltaR2WhenRemoved"].mean() > 0).sum()
                ),
                "targetCount": int(subset["target"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def confound_adjusted_associations(
    feature_frame: pd.DataFrame,
    target_columns: Sequence[str],
    max_targets: int | None = None,
    min_rows: int = 500,
    min_worlds: int = 5,
) -> pd.DataFrame:
    eligibility = target_eligibility(feature_frame, target_columns, min_rows=min_rows, min_worlds=min_worlds)
    targets = eligibility[eligibility["eligible"]]["target"].astype(str).tolist()
    if max_targets is not None:
        targets = targets[:max_targets]
    inv_cols = invariant_feature_columns(feature_frame)
    cat_cols = [column for column in CONFOUND_COLUMNS if column in feature_frame.columns]
    rows: list[dict[str, Any]] = []
    for target in targets:
        subset = feature_frame[pd.to_numeric(feature_frame[target], errors="coerce").notna()].copy()
        y = pd.to_numeric(subset[target], errors="coerce").to_numpy(dtype=float)
        confounds = _confound_matrix(subset, cat_cols)
        y_resid = _residualize_values(confounds, y)
        x_values = subset[inv_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x_resid = _residualize_values(confounds, x_values)
        correlations = _column_correlations(x_resid, y_resid)
        for feature, corr in zip(inv_cols, correlations):
            n = int(len(subset))
            se = 1.0 / math.sqrt(max(n - 3, 1)) if math.isfinite(corr) and abs(corr) < 1 else math.nan
            z = math.atanh(max(min(corr, 0.999999), -0.999999)) if math.isfinite(corr) else math.nan
            rows.append(
                {
                    "target": target,
                    "featureName": feature,
                    "featureGroup": _group_for_feature(feature),
                    "n": n,
                    "confoundAdjustedPearsonR": corr,
                    "absConfoundAdjustedPearsonR": abs(corr) if math.isfinite(corr) else math.nan,
                    "fisherZCi95Low": math.tanh(z - 1.96 * se) if math.isfinite(z) and math.isfinite(se) else math.nan,
                    "fisherZCi95High": math.tanh(z + 1.96 * se) if math.isfinite(z) and math.isfinite(se) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _confound_matrix(frame: pd.DataFrame, cat_cols: Sequence[str]):
    records: list[dict[str, float]] = []
    for values in frame[list(cat_cols)].itertuples(index=False, name=None):
        record = {f"{column}={_label(value)}": 1.0 for column, value in zip(cat_cols, values)}
        records.append(record)
    vectorizer = DictVectorizer(sparse=True)
    return vectorizer.fit_transform(records)


def _residualize_values(confound_matrix: Any, values: np.ndarray) -> np.ndarray:
    model = Ridge(alpha=10.0)
    model.fit(confound_matrix, values)
    return values - model.predict(confound_matrix)


def _column_correlations(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(vector, dtype=float).reshape(-1)
    x_centered = x - np.nanmean(x, axis=0, keepdims=True)
    y_centered = y - np.nanmean(y)
    numerator = x_centered.T @ y_centered
    denominator = np.sqrt(np.sum(x_centered * x_centered, axis=0) * np.sum(y_centered * y_centered))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numerator / denominator
    corr[~np.isfinite(corr)] = math.nan
    return corr


def summarize_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    invariant = coefficients[coefficients["isInvariantFeature"].astype(bool)].copy()
    if invariant.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (group, feature), subset in invariant.groupby(["featureGroup", "featureName"], sort=True):
        coef = pd.to_numeric(subset["coefficient"], errors="coerce").dropna()
        rows.append(
            {
                "featureGroup": group,
                "featureName": feature,
                "coefficientCount": int(len(coef)),
                "meanCoefficient": float(coef.mean()) if len(coef) else math.nan,
                "meanAbsCoefficient": float(coef.abs().mean()) if len(coef) else math.nan,
                "positiveCoefficientFraction": float((coef > 0).mean()) if len(coef) else math.nan,
                "stableSignFraction": float(max((coef > 0).mean(), (coef < 0).mean())) if len(coef) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def invariant_candidate_table(
    ablation_summary: pd.DataFrame,
    association_table: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
) -> pd.DataFrame:
    assoc_group = (
        association_table.groupby("featureGroup")["absConfoundAdjustedPearsonR"].agg(["mean", "max", "count"]).reset_index()
        if not association_table.empty
        else pd.DataFrame(columns=["featureGroup", "mean", "max", "count"])
    )
    coef_group = (
        coefficient_summary.groupby("featureGroup")["meanAbsCoefficient"].agg(["mean", "max", "count"]).reset_index()
        if not coefficient_summary.empty
        else pd.DataFrame(columns=["featureGroup", "mean", "max", "count"])
    )
    rows: list[dict[str, Any]] = []
    groups = sorted(set(INVARIANT_FEATURE_GROUPS) | set(ablation_summary.get("featureGroup", [])) | set(assoc_group.get("featureGroup", [])))
    for group in groups:
        ab = ablation_summary[ablation_summary["featureGroup"].eq(group)] if not ablation_summary.empty else pd.DataFrame()
        assoc = assoc_group[assoc_group["featureGroup"].eq(group)]
        coef = coef_group[coef_group["featureGroup"].eq(group)]
        mean_delta = _float(ab["meanDeltaR2WhenRemoved"].iloc[0], default=math.nan) if not ab.empty else math.nan
        positive_fraction = _float(ab["positiveDeltaFraction"].iloc[0], default=math.nan) if not ab.empty else math.nan
        mean_abs_corr = _float(assoc["mean"].iloc[0], default=math.nan) if not assoc.empty else math.nan
        mean_abs_coef = _float(coef["mean"].iloc[0], default=math.nan) if not coef.empty else math.nan
        score = (
            max(mean_delta, 0.0) * 8.0
            + (positive_fraction if math.isfinite(positive_fraction) else 0.0) * 0.25
            + (mean_abs_corr if math.isfinite(mean_abs_corr) else 0.0)
            + min(mean_abs_coef if math.isfinite(mean_abs_coef) else 0.0, 1.0) * 0.05
        )
        if group == "coverage_and_missingness_controls":
            tier = "coverage_caveat"
        elif math.isfinite(mean_delta) and mean_delta > 0.002 and positive_fraction >= 0.55:
            tier = "candidate_invariant"
        elif (math.isfinite(mean_delta) and mean_delta > 0 and positive_fraction >= 0.45) or (math.isfinite(mean_abs_corr) and mean_abs_corr >= 0.04):
            tier = "weak_or_contextual_candidate"
        else:
            tier = "not_supported_or_underpowered"
        rows.append(
            {
                "featureGroup": group,
                "candidateInvariantScore": float(score),
                "evidenceTier": tier,
                "meanDeltaR2WhenRemoved": mean_delta,
                "positiveDeltaFraction": positive_fraction,
                "meanAbsConfoundAdjustedCorrelation": mean_abs_corr,
                "meanAbsStandardizedCoefficient": mean_abs_coef,
                "interpretation": _group_interpretation(group),
                "claimBoundary": INVARIANT_SEARCH_CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(["candidateInvariantScore", "featureGroup"], ascending=[False, True]).reset_index(drop=True)


def apply_cross_world_signal_caveat(candidates: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or comparisons.empty:
        return candidates
    out = candidates.copy()
    combined_beats_global = int((pd.to_numeric(comparisons["combinedMinusGlobalR2"], errors="coerce") > 0).sum())
    combined_beats_confound = int((pd.to_numeric(comparisons["combinedMinusConfoundR2"], errors="coerce") > 0).sum())
    if combined_beats_global == 0 and combined_beats_confound == 0:
        mask = out["evidenceTier"].isin(["candidate_invariant", "weak_or_contextual_candidate"])
        out.loc[mask, "evidenceTier"] = "unstable_cross_world_signal"
        out.loc[mask, "interpretation"] = out.loc[mask, "interpretation"].astype(str) + (
            " Cross-world predictive models did not beat global or confound baselines, so this is retained as an unstable diagnostic signal, not an invariant candidate."
        )
    elif combined_beats_confound == 0:
        mask = out["evidenceTier"].isin(["candidate_invariant", "weak_or_contextual_candidate"])
        out.loc[mask, "evidenceTier"] = "confound_limited_contextual_signal"
        out.loc[mask, "interpretation"] = out.loc[mask, "interpretation"].astype(str) + (
            " Combined models did not beat confound-only metadata, so this remains confound-limited."
        )
    return out


def _group_interpretation(group: str) -> str:
    return {
        "memory_feedback": "Internal state, memory, feedback, and adaptive policy structure.",
        "locality_signaling": "Locality horizon, signaling range, and global-information proxies.",
        "target_position_knowledge": "Target-position or rank-like policy information and target dimensionality.",
        "stochasticity_and_policy_form": "Stochastic policy behavior and formal representation availability.",
        "policy_complexity_action_space": "Policy complexity and observation/action repertoire size.",
        "world_substrate_challenge": "Substrate dimensionality, perturbation intensity, repair, and chimera/conflict context.",
        "goal_demand_observability": "Goal observability, metric-proxy status, and repair/aggregation/symmetry demands.",
        "s08_platonic_position": "Mean, nearest, and dispersion position in S08 empirical distance space.",
        "coverage_and_missingness_controls": "Sparse-coverage and row-count controls; predictive signal here is a caveat, not a mechanistic invariant.",
    }.get(group, "Other invariant proxy features.")


def validate_invariant_search(
    feature_frame: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    associations: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(scope: str, check: str, success: bool, severity: str = "error", detail: str = "") -> None:
        checks.append({"scope": scope, "check": check, "success": bool(success), "severity": severity, "detail": detail})

    add("feature_frame", "invariant_feature_rows_present", len(feature_frame) >= 50000, detail=str(len(feature_frame)))
    add("feature_frame", "s08_features_present", any("s08_" in column for column in invariant_feature_columns(feature_frame)), detail=str(len(invariant_feature_columns(feature_frame))))
    add("metrics", "cross_world_holdout_metrics_present", not metrics.empty and metrics["validationTask"].eq("cross_world_holdout").all(), detail=str(len(metrics)))
    models = set(metrics["modelName"].astype(str)) if not metrics.empty else set()
    add("metrics", "required_models_present", {"global_mean", "invariant_feature_ridge", "confound_metadata_ridge", "combined_ridge"}.issubset(models), detail=compact_json(sorted(models)))
    target_count = int(summary["target"].nunique()) if not summary.empty else 0
    add("summary", "target_model_summary_present", not summary.empty and target_count >= 3, detail=str(target_count))
    add("summary", "target_model_summary_has_broad_target_coverage", target_count >= 5, severity="warning", detail=str(target_count))
    add("ablation", "group_ablation_present", not ablation_summary.empty and ablation_summary["featureGroup"].nunique() >= 5, detail=str(len(ablation_summary)))
    add("associations", "confound_adjusted_associations_present", not associations.empty, detail=str(len(associations)))
    add("candidates", "candidate_table_present", not candidates.empty, detail=str(len(candidates)))
    supported = candidates[candidates["evidenceTier"].isin(["candidate_invariant", "weak_or_contextual_candidate"])]
    add("candidates", "at_least_one_candidate_or_contextual_candidate", not supported.empty, severity="warning", detail=compact_json(supported["featureGroup"].tolist()))
    return pd.DataFrame(checks)
