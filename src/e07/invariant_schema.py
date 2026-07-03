"""Helpers for E07 S09 invariant discovery and validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


INVARIANT_SCHEMA_VERSION = "eidosoma.e07.invariant_discovery.v1"


def stable_fraction(label: str, *, salt: str = "e07-s09") -> float:
    digest = hashlib.sha256(f"{salt}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def parse_json_field(value: Any, default: Any | None = None) -> Any:
    if default is None:
        default = {}
    if value is None:
        return default
    if isinstance(value, (dict, list, tuple)):
        return value
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def flatten_text(value: Any) -> str:
    parts: list[str] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                parts.append(str(key))
                walk(child)
        elif isinstance(item, (list, tuple, set)):
            for child in item:
                walk(child)
        else:
            parts.append(str(item))

    walk(value)
    return " ".join(parts).lower()


def json_item_count(value: Any) -> int:
    payload = parse_json_field(value, default=[])
    if isinstance(payload, Mapping):
        return len(payload)
    if isinstance(payload, (list, tuple, set)):
        return len(payload)
    return 0


def json_string_items(value: Any) -> set[str]:
    payload = parse_json_field(value, default=[])
    if isinstance(payload, Mapping):
        items = payload.keys()
    elif isinstance(payload, (list, tuple, set)):
        items = payload
    else:
        items = []
    return {str(item).lower() for item in items}


def boolish(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    return 1.0 if text in {"true", "1", "yes"} else 0.0


def keyword_flag(text: str, keywords: Iterable[str]) -> float:
    lowered = text.lower()
    return float(any(keyword.lower() in lowered for keyword in keywords))


def extract_policy_invariants(row: Mapping[str, Any]) -> dict[str, float]:
    """Extract candidate invariant proxies from one S02 policy row."""

    payload = parse_json_field(row.get("representation_payload_json"), default={})
    payload_text = flatten_text(payload)
    action_items = json_string_items(row.get("action_vocabulary_json"))
    state_items = json_string_items(row.get("state_variables_json"))
    parameter_count = json_item_count(row.get("parameter_keys_json"))
    representation_text = " ".join(
        [
            payload_text,
            flatten_text(parse_json_field(row.get("limitations_json"), default=[])),
            str(row.get("policy_family", "")),
            str(row.get("algorithm", "")),
            str(row.get("information_access", "")),
            str(row.get("observation_contract", "")),
        ]
    ).lower()
    dsl_source = str(payload.get("dslSource", "")) if isinstance(payload, Mapping) else ""
    rule_count = payload.get("ruleCount") if isinstance(payload, Mapping) else None
    try:
        rule_count_value = float(rule_count)
    except (TypeError, ValueError):
        rule_count_value = float(len(re.findall(r"(^|\n)\s*rule\b", dsl_source)))
    condition_count = 0.0
    if isinstance(payload, Mapping) and isinstance(payload.get("conditionVocabulary"), (list, tuple)):
        condition_count = float(len(payload.get("conditionVocabulary", [])))

    uses_global = boolish(row.get("uses_global_oracle"))
    info_access = str(row.get("information_access", "")).lower()
    locality_horizon = 3.0 if uses_global or "global" in info_access else 1.0
    if "neighbor" in info_access or "local" in info_access:
        locality_horizon = min(locality_horizon, 1.0)
    if "endpoint" in info_access:
        locality_horizon = max(locality_horizon, 2.0)

    has_memory = boolish(row.get("requires_memory")) or keyword_flag(
        representation_text,
        ["remember", "memory", "last_move", "time_since", "frustration", "fatigue"],
    )
    has_signal = boolish(row.get("requires_signaling")) or "signal" in action_items or keyword_flag(
        representation_text,
        ["signal", "communication", "inbox"],
    )
    has_stochastic = keyword_flag(representation_text, ["random_lt", "choose(", "probability", "stochastic", "random"])
    has_target_position = keyword_flag(
        representation_text,
        ["ideal_position", "set_ideal", "estimate_target_position", "target_position", "left_boundary", "right_boundary"],
    )
    has_feedback = keyword_flag(
        representation_text,
        ["remember", "signal", "last_failure", "frustration", "fatigue", "damage", "repair", "homeostasis", "learn"],
    )
    state_count = float(len(state_items))
    action_count = float(len(action_items))
    memory_depth = float(has_memory) + state_count + 0.25 * parameter_count
    feedback_strength = float(has_feedback) + float(has_signal) + 0.05 * parameter_count

    return {
        "policy_requires_memory": float(has_memory),
        "policy_requires_signaling": float(has_signal),
        "policy_uses_global_oracle": uses_global,
        "policy_target_position_knowledge": float(has_target_position),
        "policy_stochastic_choice": float(has_stochastic),
        "policy_feedback_or_adaptation": float(has_feedback),
        "policy_locality_horizon_score": float(locality_horizon),
        "policy_memory_depth_proxy": float(memory_depth),
        "policy_feedback_strength_proxy": float(feedback_strength),
        "policy_rule_count": float(rule_count_value),
        "policy_condition_count": float(condition_count),
        "policy_action_count": float(action_count),
        "policy_has_swap_action": float("swap" in action_items),
        "policy_has_compare_action": float("compare" in action_items),
        "policy_has_wait_action": float("wait" in action_items),
        "policy_has_signal_action": float("signal" in action_items),
        "policy_has_remember_action": float("remember" in action_items),
        "policy_interpretable": boolish(row.get("interpretable")),
    }


def extract_world_invariants(row: Mapping[str, Any]) -> dict[str, float]:
    """Extract candidate invariant proxies from one S01 world row."""

    metadata = parse_json_field(row.get("metadata_json"), default={})
    text = " ".join(
        [
            str(row.get("world_family", "")),
            str(row.get("substrate_kind", "")),
            str(row.get("state_space", "")),
            str(row.get("local_observations", "")),
            str(row.get("action_set", "")),
            str(row.get("transition_rules", "")),
            str(row.get("goal_predicate", "")),
            str(row.get("perturbation_model", "")),
            str(row.get("measurement_functions", "")),
            str(row.get("scheduler", "")),
            flatten_text(metadata),
        ]
    ).lower()
    substrate = str(row.get("substrate_kind", "")).lower()
    dimension = 1.0
    if "3d" in substrate or "3d" in text:
        dimension = 3.0
    elif "2d" in substrate or "2d" in text or "grid" in substrate:
        dimension = 2.0
    elif "graph" in substrate:
        dimension = 2.5
    frozen_match = re.search(r"frozen_count\s*[=:]\s*(\d+)", text)
    frozen_count = 0.0
    if frozen_match:
        frozen_count = float(frozen_match.group(1))
    elif isinstance(metadata, Mapping) and "frozen_count" in metadata:
        try:
            frozen_count = float(metadata.get("frozen_count") or 0.0)
        except (TypeError, ValueError):
            frozen_count = 0.0
    observation_words = len(str(row.get("local_observations", "")).split())
    action_text = str(row.get("action_set", ""))
    measurement_text = str(row.get("measurement_functions", ""))
    action_count = len([part for part in re.split(r"[;,]", action_text) if part.strip()])
    measurement_count = len([part for part in re.split(r"[;,]", measurement_text) if part.strip()])
    nonlocal_baseline = keyword_flag(text, ["full array", "traditional controller", "global reference"])
    local_observation = keyword_flag(text, ["local", "neighbor", "cell-view", "bounded"])

    return {
        "world_dimension_proxy": float(dimension),
        "world_has_2d_or_graph_substrate": float(dimension > 1.0),
        "world_has_frozen_perturbation": keyword_flag(text, ["frozen", "stuck", "passive"]),
        "world_frozen_count": float(frozen_count),
        "world_has_chimera_or_mixed_identity": keyword_flag(text, ["chimera", "chimeric", "algotype mix", "mixed"]),
        "world_has_damage_or_repair": keyword_flag(text, ["damage", "repair", "recover", "regeneration", "hole", "appendage"]),
        "world_has_homeostasis": keyword_flag(text, ["homeostasis", "homeostatic", "target range", "maintenance"]),
        "world_has_governance_or_dominance": keyword_flag(text, ["governance", "dominance", "graft", "mutant", "interface"]),
        "world_has_target_morphology": keyword_flag(text, ["target morphology", "shape", "boundary", "topology", "gradient", "organ"]),
        "world_has_null_or_control": keyword_flag(text, ["null", "dummy", "label shuffle", "random walker", "control"]),
        "world_scheduler_cyclic": keyword_flag(text, ["cyclic"]),
        "world_scheduler_uniform_or_random": keyword_flag(text, ["uniform", "random"]),
        "world_timeout_guard": keyword_flag(text, ["timeout", "guard"]),
        "world_nonlocal_baseline_context": float(nonlocal_baseline),
        "world_local_observation_context": float(local_observation),
        "world_observation_richness_proxy": float(observation_words),
        "world_action_count": float(action_count),
        "world_measurement_count": float(measurement_count),
    }


def source_balance_weights(frame: pd.DataFrame, source_column: str = "source_experiment_id") -> pd.Series:
    if frame.empty or source_column not in frame.columns:
        return pd.Series(dtype=float, index=frame.index)
    counts = frame.groupby(source_column, observed=True).size().astype(float)
    median = float(counts.median()) if not counts.empty else 1.0
    weights = frame[source_column].map(lambda value: median / max(1.0, float(counts.get(value, 1.0))))
    return weights.clip(lower=0.05, upper=20.0).astype(float)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = numeric.notna() & (numeric_weights > 0)
    if not mask.any():
        return float("nan")
    total = float(numeric_weights[mask].sum())
    return float((numeric[mask] * numeric_weights[mask]).sum() / total)


def weighted_correlation(x: pd.Series, y: pd.Series, weights: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = x_num.notna() & y_num.notna() & (w > 0)
    if int(mask.sum()) < 3:
        return float("nan")
    x_vals = x_num[mask].to_numpy(dtype=float)
    y_vals = y_num[mask].to_numpy(dtype=float)
    w_vals = w[mask].to_numpy(dtype=float)
    w_vals = w_vals / np.sum(w_vals)
    x_mean = float(np.sum(w_vals * x_vals))
    y_mean = float(np.sum(w_vals * y_vals))
    x_center = x_vals - x_mean
    y_center = y_vals - y_mean
    cov = float(np.sum(w_vals * x_center * y_center))
    x_var = float(np.sum(w_vals * x_center * x_center))
    y_var = float(np.sum(w_vals * y_center * y_center))
    if x_var <= 0 or y_var <= 0:
        return float("nan")
    return cov / math.sqrt(x_var * y_var)


def residualize_by_group_median(frame: pd.DataFrame, *, target_column: str, group_columns: Sequence[str]) -> pd.Series:
    medians = frame.groupby(list(group_columns), observed=True)[target_column].transform("median")
    return pd.to_numeric(frame[target_column], errors="coerce") - pd.to_numeric(medians, errors="coerce")


def deterministic_cap(frame: pd.DataFrame, max_rows: int, *, id_column: str, salt: str) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    hashes = pd.util.hash_pandas_object(frame[id_column].astype(str) + f"|{salt}", index=False)
    return frame.assign(_cap_hash=hashes).sort_values("_cap_hash", kind="mergesort").head(max_rows).drop(columns=["_cap_hash"])


def validate_invariant_artifacts(
    candidates: pd.DataFrame,
    model_metrics: pd.DataFrame,
    counterexamples: pd.DataFrame,
    s08_audit: pd.DataFrame,
    *,
    min_stable_candidates: int = 1,
) -> pd.DataFrame:
    """Validate S09 invariant discovery artifacts."""

    checks: list[dict[str, Any]] = []
    required_candidate_columns = {
        "candidate_feature",
        "feature_group",
        "effect_full",
        "effect_heldout_policy",
        "effect_heldout_world",
        "source_dominance_rate",
        "candidate_status",
    }
    missing = sorted(required_candidate_columns - set(candidates.columns))
    checks.append(
        {
            "validation_case": "candidate_table_required_columns_present",
            "success": not missing and not candidates.empty,
            "detail": "candidate table populated" if not missing and not candidates.empty else f"missing columns: {missing}; rows={len(candidates)}",
        }
    )
    stable = candidates[candidates.get("candidate_status", pd.Series(dtype=str)).astype(str) == "stable_supported_candidate"]
    checks.append(
        {
            "validation_case": "stable_supported_candidate_present",
            "success": len(stable) >= min_stable_candidates,
            "detail": f"stable supported candidates: {len(stable)}",
        }
    )
    heldout_columns = {"split_name", "model_name", "mae", "rmse", "r2"}
    observed_splits = set(model_metrics.get("split_name", pd.Series(dtype=str)).astype(str))
    observed_models = set(model_metrics.get("model_name", pd.Series(dtype=str)).astype(str))
    checks.append(
        {
            "validation_case": "heldout_policy_and_world_models_evaluated",
            "success": heldout_columns <= set(model_metrics.columns) and {"heldout_policy", "heldout_world"} <= observed_splits,
            "detail": f"splits={sorted(observed_splits)}; models={sorted(observed_models)}",
        }
    )
    checks.append(
        {
            "validation_case": "source_metric_missingness_controls_present",
            "success": bool("source_metric_median" in observed_models and any(model.startswith("source_missingness_control") for model in observed_models)),
            "detail": f"models={sorted(observed_models)}",
        }
    )
    finite_metrics = not model_metrics.empty and model_metrics[["mae", "rmse"]].replace([np.inf, -np.inf], np.nan).notna().all().all()
    checks.append(
        {
            "validation_case": "model_metrics_are_finite",
            "success": bool(finite_metrics),
            "detail": f"model metric rows: {len(model_metrics)}",
        }
    )
    checks.append(
        {
            "validation_case": "counterexamples_recorded",
            "success": not counterexamples.empty and {"candidate_feature", "counterexample_type", "residual_vs_source_metric"} <= set(counterexamples.columns),
            "detail": f"counterexample rows: {len(counterexamples)}",
        }
    )
    caveat_cases = set(s08_audit.get("s08_caveat", pd.Series(dtype=str)).astype(str))
    checks.append(
        {
            "validation_case": "s08_caveats_carried_forward",
            "success": {"source_dominance_failure", "goal_conflict_separation_failure"} <= caveat_cases,
            "detail": f"S08 caveats recorded: {sorted(caveat_cases)}",
        }
    )
    return pd.DataFrame(checks)
