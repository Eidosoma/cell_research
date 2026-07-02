"""Higher-dimensional delayed-gratification trajectory analysis for E05 S14."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


STEP_ID = "S14"
EXPERIMENT_ID = "E05"
EPSILON = 1e-12
DEFAULT_NULL_REPLICATES = 499
NULL_MODEL_ID = "increment_permutation_length_endpoint_matched"
RUN_KEY_COLUMNS = (
    "target_id",
    "target_kind",
    "perturbation_type",
    "mask_id",
    "policy_id",
    "policy_family",
    "simulation_seed",
)
JOIN_KEY_COLUMNS = ("target_id", "perturbation_type", "mask_id", "policy_id", "simulation_seed")
TASK_JOIN_KEY_COLUMNS = ("target_id", "perturbation_type", "mask_id", "simulation_seed")
METRIC_SPECS = {
    "target_error_proxy": {
        "column": "target_error",
        "metric_family": "target_identity_proxy",
        "source": "S03 target-error trajectory column from S08 traces",
    },
    "s05_aggregate_morphospace_error": {
        "column": "aggregate_morphospace_error",
        "metric_family": "aggregate_morphospace_proxy",
        "source": "S05 aggregate morphospace-error trajectory column from S08 traces",
    },
}


def stable_json(obj: Any) -> str:
    """Return a deterministic compact JSON representation."""

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stable_seed(parts: Sequence[Any]) -> int:
    """Return a deterministic 32-bit RNG seed from a sequence of identifiers."""

    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def compute_delayed_gratification_metrics(
    values: Sequence[float],
    event_steps: Sequence[int] | None = None,
    *,
    eps: float = EPSILON,
) -> dict[str, Any]:
    """Measure transient error increases that are followed by larger later repair.

    All metrics assume lower error is better. A constructive DG event requires:
    an increase above the best-so-far error, a lower final error than the initial
    state, and post-peak repair larger than the transient drawup.
    """

    arr = np.asarray(values, dtype=float)
    if event_steps is None:
        steps = np.arange(len(arr), dtype=int)
    else:
        steps = np.asarray(event_steps, dtype=int)
    if len(arr) != len(steps):
        raise ValueError("values and event_steps must have the same length")
    if len(arr) == 0:
        raise ValueError("at least one trajectory point is required")
    if not np.isfinite(arr).all():
        raise ValueError("trajectory values must be finite")

    diffs = np.diff(arr)
    prefix_min = np.minimum.accumulate(arr)
    drawups = arr - prefix_min
    max_drawup_index = int(np.argmax(drawups))
    max_prefix_drawup = float(max(0.0, drawups[max_drawup_index]))
    post_drawup_repair = float(arr[max_drawup_index] - arr[-1])
    net_improvement = float(arr[0] - arr[-1])
    away_steps = int(np.sum(diffs > eps))
    positive_diffs = diffs[diffs > eps]
    largest_step_increase = float(positive_diffs.max()) if len(positive_diffs) else 0.0
    drawup_integral = float(np.sum(np.maximum(0.0, drawups)))
    repair_exceeds_drawup = bool(post_drawup_repair > max_prefix_drawup + eps)
    dg_detected = bool(net_improvement > eps and max_prefix_drawup > eps and repair_exceeds_drawup)
    dg_score = float(max_prefix_drawup if dg_detected else 0.0)

    return {
        "point_count": int(len(arr)),
        "initial_event_step": int(steps[0]),
        "final_event_step": int(steps[-1]),
        "initial_error": float(arr[0]),
        "final_error": float(arr[-1]),
        "min_error": float(np.min(arr)),
        "max_error": float(np.max(arr)),
        "net_improvement": net_improvement,
        "endpoint_error_delta": float(arr[-1] - arr[0]),
        "temporary_away_steps": away_steps,
        "temporary_away_fraction": float(away_steps / max(1, len(arr) - 1)),
        "largest_step_increase": largest_step_increase,
        "max_prefix_drawup": max_prefix_drawup,
        "max_prefix_drawup_event_step": int(steps[max_drawup_index]),
        "post_drawup_repair": post_drawup_repair,
        "repair_exceeds_drawup": repair_exceeds_drawup,
        "drawup_integral": drawup_integral,
        "monotone_nonincreasing": bool(np.all(diffs <= eps)),
        "dg_detected": dg_detected,
        "dg_score": dg_score,
    }


def matched_increment_null_summary(
    values: Sequence[float],
    observed_score: float,
    *,
    n_null: int = DEFAULT_NULL_REPLICATES,
    seed: int = 0,
    eps: float = EPSILON,
) -> dict[str, Any]:
    """Summarize length- and endpoint-matched null DG scores.

    The null model permutes observed step increments. This preserves the number
    of trajectory points, initial error, final error, endpoint delta, and the
    multiset of one-step error changes.
    """

    arr = np.asarray(values, dtype=float)
    if len(arr) < 3:
        return {
            "null_model_id": NULL_MODEL_ID,
            "null_match_status": "blocked_insufficient_trace",
            "null_replicates": 0,
            "null_length_match_rate": 0.0,
            "max_null_initial_abs_error": np.nan,
            "max_null_endpoint_abs_error": np.nan,
            "null_dg_detection_rate": np.nan,
            "null_mean_dg_score": np.nan,
            "null_median_dg_score": np.nan,
            "null_p95_dg_score": np.nan,
            "null_p99_dg_score": np.nan,
            "null_max_dg_score": np.nan,
            "null_p_value": np.nan,
            "dg_exceeds_null_p95": False,
            "dg_exceeds_null_p99": False,
        }
    if n_null <= 0:
        raise ValueError("n_null must be positive")

    increments = np.diff(arr)
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    detected: list[bool] = []
    length_matches = 0
    initial_errors: list[float] = []
    endpoint_errors: list[float] = []
    for _ in range(n_null):
        permuted = rng.permutation(increments)
        null_values = arr[0] + np.concatenate(([0.0], np.cumsum(permuted)))
        length_matches += int(len(null_values) == len(arr))
        initial_errors.append(float(abs(null_values[0] - arr[0])))
        endpoint_errors.append(float(abs(null_values[-1] - arr[-1])))
        metrics = compute_delayed_gratification_metrics(null_values, eps=eps)
        scores.append(float(metrics["dg_score"]))
        detected.append(bool(metrics["dg_detected"]))

    score_array = np.asarray(scores, dtype=float)
    p95 = float(np.quantile(score_array, 0.95))
    p99 = float(np.quantile(score_array, 0.99))
    p_value = float((1 + np.sum(score_array >= observed_score - eps)) / (len(score_array) + 1))
    return {
        "null_model_id": NULL_MODEL_ID,
        "null_match_status": "matched",
        "null_replicates": int(n_null),
        "null_length_match_rate": float(length_matches / n_null),
        "max_null_initial_abs_error": float(max(initial_errors)),
        "max_null_endpoint_abs_error": float(max(endpoint_errors)),
        "null_dg_detection_rate": float(np.mean(detected)),
        "null_mean_dg_score": float(np.mean(score_array)),
        "null_median_dg_score": float(np.quantile(score_array, 0.50)),
        "null_p95_dg_score": p95,
        "null_p99_dg_score": p99,
        "null_max_dg_score": float(np.max(score_array)),
        "null_p_value": p_value,
        "dg_exceeds_null_p95": bool(observed_score > p95 + eps),
        "dg_exceeds_null_p99": bool(observed_score > p99 + eps),
    }


def build_shape_dg_tables(artifacts_dir: Path, *, n_null: int = DEFAULT_NULL_REPLICATES) -> dict[str, pd.DataFrame]:
    """Load S05/S08/S12/S13 inputs and return S14 result tables."""

    artifacts_dir = Path(artifacts_dir)
    source_inputs = source_input_table(artifacts_dir)
    trace_df = load_s08_trace_table(artifacts_dir)
    route_context = load_s12_route_context(artifacts_dir)
    s13_context = load_s13_context(artifacts_dir)
    s13_task_context = summarize_s13_task_context(s13_context)

    dg_df = compute_shape_dg(trace_df, n_null=n_null)
    dg_df = attach_s12_context(dg_df, route_context)
    dg_df = attach_s13_context(dg_df, s13_context, s13_task_context)
    sensitivity_df = compute_metric_sensitivity(dg_df)
    summary_df = summarize_shape_dg(dg_df, sensitivity_df)
    null_validation_df = summarize_null_validation(dg_df)
    validation_df = validate_shape_dg_tables(
        dg_df=dg_df,
        summary_df=summary_df,
        sensitivity_df=sensitivity_df,
        null_validation_df=null_validation_df,
        source_inputs_df=source_inputs,
        route_context_df=route_context,
        s13_context_df=s13_context,
        required_artifact_paths=None,
    )
    return {
        "shape_dg": dg_df,
        "summary": summary_df,
        "metric_sensitivity": sensitivity_df,
        "null_validation": null_validation_df,
        "source_inputs": source_inputs,
        "s12_route_context": route_context,
        "s13_context": s13_context,
        "validation": validation_df,
    }


def source_input_table(artifacts_dir: Path) -> pd.DataFrame:
    """Return compact provenance rows for S14 upstream inputs."""

    specs = [
        ("S05", "metric_catalog", artifacts_dir / "results" / "e05_metric_catalog.parquet", True),
        ("S05", "metric_validation", artifacts_dir / "results" / "e05_metric_validation.parquet", True),
        ("S08", "regeneration_trace_table", artifacts_dir / "traces" / "e05_regeneration_trace_table.parquet", True),
        ("S08", "regeneration_results", artifacts_dir / "results" / "e05_regeneration_results.parquet", True),
        ("S08", "regeneration_metric_rows", artifacts_dir / "results" / "e05_regeneration_metric_rows.parquet", True),
        ("S12", "morphospace_route_metrics", artifacts_dir / "results" / "e05_morphospace_route_metrics.parquet", True),
        ("S13", "local_global_control", artifacts_dir / "results" / "e05_local_vs_global_control.parquet", True),
    ]
    rows: list[dict[str, Any]] = []
    for source_step, input_type, path, required in specs:
        exists = path.exists()
        row_count = 0
        columns: list[str] = []
        if exists and path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            row_count = int(len(frame))
            columns = list(frame.columns)
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": source_step,
                "input_type": input_type,
                "source_path": str(path),
                "required": bool(required),
                "exists": bool(exists),
                "row_count": row_count,
                "column_count": int(len(columns)),
                "columns_json": stable_json(columns),
                "used_for_primary_dg": bool(source_step in {"S08"}),
                "used_for_context_or_validation": bool(source_step in {"S05", "S12", "S13"}),
            }
        )
    return pd.DataFrame(rows)


def load_s08_trace_table(artifacts_dir: Path) -> pd.DataFrame:
    path = Path(artifacts_dir) / "traces" / "e05_regeneration_trace_table.parquet"
    if not path.exists():
        raise FileNotFoundError(f"S08 trace table is required for S14: {path}")
    df = pd.read_parquet(path)
    required = set(RUN_KEY_COLUMNS) | {"event_step", "target_error", "aggregate_morphospace_error"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"S08 trace table is missing required columns: {missing}")
    return df.copy()


def load_s12_route_context(artifacts_dir: Path) -> pd.DataFrame:
    path = Path(artifacts_dir) / "results" / "e05_morphospace_route_metrics.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "source_research_step_id" not in df.columns:
        return pd.DataFrame()
    df = df[df["source_research_step_id"].eq("S08")].copy()
    if df.empty:
        return df
    df["perturbation_type"] = df["source_task_type"].astype(str)
    selected = [
        "target_id",
        "perturbation_type",
        "policy_id",
        "simulation_seed",
        "task_id",
        "point_count",
        "initial_target_error",
        "final_target_error",
        "target_error_delta",
        "temporary_away_steps",
        "temporary_away_fraction",
        "max_overshoot_above_initial",
        "embedding_path_length",
        "embedding_direct_distance",
        "path_curvature",
        "route_diversity_to_group_mean",
        "group_mean_pairwise_route_distance",
    ]
    out = df[[column for column in selected if column in df.columns]].copy()
    rename = {
        "task_id": "s12_task_id",
        "point_count": "s12_point_count",
        "initial_target_error": "s12_initial_target_error",
        "final_target_error": "s12_final_target_error",
        "target_error_delta": "s12_target_error_delta",
        "temporary_away_steps": "s12_temporary_away_steps",
        "temporary_away_fraction": "s12_temporary_away_fraction",
        "max_overshoot_above_initial": "s12_max_overshoot_above_initial",
        "embedding_path_length": "s12_embedding_path_length",
        "embedding_direct_distance": "s12_embedding_direct_distance",
        "path_curvature": "s12_path_curvature",
        "route_diversity_to_group_mean": "s12_route_diversity_to_group_mean",
        "group_mean_pairwise_route_distance": "s12_group_mean_pairwise_route_distance",
    }
    return out.rename(columns=rename)


def load_s13_context(artifacts_dir: Path) -> pd.DataFrame:
    path = Path(artifacts_dir) / "results" / "e05_local_vs_global_control.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "source_research_step_id" not in df.columns:
        return pd.DataFrame()
    df = df[df["source_research_step_id"].eq("S08")].copy()
    if df.empty:
        return df
    selected = [
        "target_id",
        "target_kind",
        "task_id",
        "perturbation_type",
        "mask_id",
        "policy_id",
        "policy_family",
        "control_class",
        "information_access_tier",
        "declared_information_access",
        "global_state_access",
        "global_target_access",
        "organizer_cue_added",
        "birth_death_allowed",
        "identity_conversion_allowed",
        "unfreeze_allowed",
        "local_policy_information_access_changed",
        "simulation_seed",
        "baseline_blocked",
        "blocked_reason",
        "semantic_blocker",
        "repairability_class",
        "target_recovery_fraction",
        "final_target_error",
        "final_aggregate_morphospace_error",
        "total_energy_cost",
        "s12_path_curvature",
        "s12_temporary_away_fraction",
    ]
    return df[[column for column in selected if column in df.columns]].copy()


def summarize_s13_task_context(s13_context: pd.DataFrame) -> pd.DataFrame:
    """Summarize S13 local/global endpoint context for each S08 task seed."""

    if s13_context.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in s13_context.groupby(list(TASK_JOIN_KEY_COLUMNS), sort=False):
        control_classes = sorted(str(value) for value in group["control_class"].dropna().unique())
        global_rows = group[group["control_class"].ne("local")]
        rows.append(
            {
                "target_id": key[0],
                "perturbation_type": key[1],
                "mask_id": key[2],
                "simulation_seed": int(key[3]),
                "s13_control_classes_json": stable_json(control_classes),
                "s13_context_rows": int(len(group)),
                "s13_global_context_rows": int(len(global_rows)),
                "s13_any_global_context_blocked": bool(global_rows["baseline_blocked"].fillna(False).any()) if not global_rows.empty else False,
                "s13_best_global_final_target_error": (
                    float(global_rows["final_target_error"].min()) if not global_rows.empty else np.nan
                ),
                "s13_best_global_recovery_fraction": (
                    float(global_rows["target_recovery_fraction"].max()) if not global_rows.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_shape_dg(trace_df: pd.DataFrame, *, n_null: int = DEFAULT_NULL_REPLICATES) -> pd.DataFrame:
    """Compute observed and null-matched DG rows for each S08 run and metric."""

    records: list[dict[str, Any]] = []
    grouped = trace_df.groupby(list(RUN_KEY_COLUMNS), sort=False, dropna=False)
    for key, group in grouped:
        group = group.sort_values("event_step")
        first = group.iloc[0]
        base = {column: first[column] for column in RUN_KEY_COLUMNS}
        base.update(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": "S08",
                "task_id": f"S08:{first['target_id']}:{first['perturbation_type']}",
                "event_step_min": int(group["event_step"].min()),
                "event_step_max": int(group["event_step"].max()),
                "accepted_swaps_final": int(group["accepted_swaps"].iloc[-1]) if "accepted_swaps" in group.columns else np.nan,
                "attempted_swaps_final": int(group["attempted_swaps"].iloc[-1]) if "attempted_swaps" in group.columns else np.nan,
                "rejected_actions_final": int(group["rejected_actions"].iloc[-1]) if "rejected_actions" in group.columns else np.nan,
                "wait_actions_final": int(group["wait_actions"].iloc[-1]) if "wait_actions" in group.columns else np.nan,
                "energy_cost_final": float(group["energy_cost"].iloc[-1]) if "energy_cost" in group.columns else np.nan,
                "population_count_final": (
                    int(group["population_count"].iloc[-1]) if "population_count" in group.columns and pd.notna(group["population_count"].iloc[-1]) else np.nan
                ),
            }
        )
        steps = group["event_step"].to_numpy(dtype=int)
        for metric_id, spec in METRIC_SPECS.items():
            values = group[spec["column"]].to_numpy(dtype=float)
            record = {
                **base,
                "metric_id": metric_id,
                "metric_column": spec["column"],
                "metric_family": spec["metric_family"],
                "metric_source": spec["source"],
                "lower_is_better": True,
            }
            try:
                observed = compute_delayed_gratification_metrics(values, steps)
                seed = stable_seed([*key, metric_id, n_null, NULL_MODEL_ID])
                nulls = matched_increment_null_summary(values, observed["dg_score"], n_null=n_null, seed=seed)
                record.update(observed)
                record.update(nulls)
                record["null_rng_seed"] = int(seed)
                record["analysis_status"] = "computed"
                record["caveat"] = ""
            except ValueError as exc:
                record.update(_blocked_metric_record(str(exc), len(values), steps))
            records.append(record)
    return pd.DataFrame(records)


def _blocked_metric_record(reason: str, point_count: int, steps: np.ndarray) -> dict[str, Any]:
    first_step = int(steps[0]) if len(steps) else -1
    last_step = int(steps[-1]) if len(steps) else -1
    return {
        "point_count": int(point_count),
        "initial_event_step": first_step,
        "final_event_step": last_step,
        "initial_error": np.nan,
        "final_error": np.nan,
        "min_error": np.nan,
        "max_error": np.nan,
        "net_improvement": np.nan,
        "endpoint_error_delta": np.nan,
        "temporary_away_steps": 0,
        "temporary_away_fraction": np.nan,
        "largest_step_increase": np.nan,
        "max_prefix_drawup": np.nan,
        "max_prefix_drawup_event_step": -1,
        "post_drawup_repair": np.nan,
        "repair_exceeds_drawup": False,
        "drawup_integral": np.nan,
        "monotone_nonincreasing": False,
        "dg_detected": False,
        "dg_score": np.nan,
        "null_model_id": NULL_MODEL_ID,
        "null_match_status": "blocked_missing_or_invalid_metric",
        "null_replicates": 0,
        "null_length_match_rate": 0.0,
        "max_null_initial_abs_error": np.nan,
        "max_null_endpoint_abs_error": np.nan,
        "null_dg_detection_rate": np.nan,
        "null_mean_dg_score": np.nan,
        "null_median_dg_score": np.nan,
        "null_p95_dg_score": np.nan,
        "null_p99_dg_score": np.nan,
        "null_max_dg_score": np.nan,
        "null_p_value": np.nan,
        "dg_exceeds_null_p95": False,
        "dg_exceeds_null_p99": False,
        "null_rng_seed": -1,
        "analysis_status": "blocked",
        "caveat": reason,
    }


def attach_s12_context(dg_df: pd.DataFrame, route_context: pd.DataFrame) -> pd.DataFrame:
    if route_context.empty:
        out = dg_df.copy()
        out["s12_context_joined"] = False
        return out
    merge_keys = ["target_id", "perturbation_type", "policy_id", "simulation_seed"]
    out = dg_df.merge(route_context, on=merge_keys, how="left", validate="many_to_one")
    out["s12_context_joined"] = out["s12_task_id"].notna() if "s12_task_id" in out.columns else False
    return out


def attach_s13_context(dg_df: pd.DataFrame, s13_context: pd.DataFrame, s13_task_context: pd.DataFrame) -> pd.DataFrame:
    out = dg_df.copy()
    if not s13_context.empty:
        local_context = s13_context[s13_context["control_class"].eq("local")].copy()
        rename = {
            "task_id": "s13_task_id",
            "control_class": "s13_control_class",
            "information_access_tier": "s13_information_access_tier",
            "declared_information_access": "s13_declared_information_access",
            "global_state_access": "s13_global_state_access",
            "global_target_access": "s13_global_target_access",
            "organizer_cue_added": "s13_organizer_cue_added",
            "birth_death_allowed": "s13_birth_death_allowed",
            "identity_conversion_allowed": "s13_identity_conversion_allowed",
            "unfreeze_allowed": "s13_unfreeze_allowed",
            "local_policy_information_access_changed": "s13_local_policy_information_access_changed",
            "baseline_blocked": "s13_baseline_blocked",
            "blocked_reason": "s13_blocked_reason",
            "semantic_blocker": "s13_semantic_blocker",
            "repairability_class": "s13_repairability_class",
            "target_recovery_fraction": "s13_target_recovery_fraction",
            "final_target_error": "s13_final_target_error",
            "final_aggregate_morphospace_error": "s13_final_aggregate_morphospace_error",
            "total_energy_cost": "s13_total_energy_cost",
        }
        selected = list(JOIN_KEY_COLUMNS) + [column for column in rename if column in local_context.columns]
        local_context = local_context[selected].rename(columns=rename)
        out = out.merge(local_context, on=list(JOIN_KEY_COLUMNS), how="left", validate="many_to_one")
    if not s13_task_context.empty:
        out = out.merge(s13_task_context, on=list(TASK_JOIN_KEY_COLUMNS), how="left", validate="many_to_one")
    out["s13_local_context_joined"] = out["s13_control_class"].notna() if "s13_control_class" in out.columns else False
    out["s13_global_endpoint_context_available"] = (
        out["s13_global_context_rows"].fillna(0).astype(int) > 0 if "s13_global_context_rows" in out.columns else False
    )
    return out


def compute_metric_sensitivity(dg_df: pd.DataFrame) -> pd.DataFrame:
    """Compare target-error and S05 aggregate-error DG calls per S08 run."""

    records: list[dict[str, Any]] = []
    key_cols = list(RUN_KEY_COLUMNS)
    for key, group in dg_df.groupby(key_cols, sort=False, dropna=False):
        by_metric = {str(row["metric_id"]): row for row in group.to_dict(orient="records")}
        target = by_metric.get("target_error_proxy")
        aggregate = by_metric.get("s05_aggregate_morphospace_error")
        if target is None or aggregate is None:
            status = "blocked_missing_metric_pair"
            detected_agreement = False
            exceeds_agreement = False
        else:
            status = "computed"
            detected_agreement = bool(target["dg_detected"]) == bool(aggregate["dg_detected"])
            exceeds_agreement = bool(target["dg_exceeds_null_p95"]) == bool(aggregate["dg_exceeds_null_p95"])
        base = {column: value for column, value in zip(key_cols, key, strict=True)}
        if target is None or aggregate is None:
            records.append(
                {
                    "research_step_id": STEP_ID,
                    **base,
                    "analysis_status": status,
                    "metric_agreement_detected": detected_agreement,
                    "metric_agreement_exceeds_null_p95": exceeds_agreement,
                    "metric_inconsistent_away_from_target": True,
                    "sensitivity_classification": "blocked_missing_metric_pair",
                }
            )
            continue
        target_detected = bool(target["dg_detected"])
        aggregate_detected = bool(aggregate["dg_detected"])
        target_exceeds = bool(target["dg_exceeds_null_p95"])
        aggregate_exceeds = bool(aggregate["dg_exceeds_null_p95"])
        if target_exceeds and aggregate_exceeds:
            classification = "metric_consistent_null_exceeding_shape_dg"
        elif target_detected and aggregate_detected:
            classification = "metric_consistent_observed_shape_dg_not_null_exceeding"
        elif not target_detected and not aggregate_detected:
            classification = "metric_consistent_no_constructive_dg"
        elif target_detected:
            classification = "metric_inconsistent_target_only_dg"
        else:
            classification = "metric_inconsistent_aggregate_only_dg"
        records.append(
            {
                "research_step_id": STEP_ID,
                **base,
                "analysis_status": status,
                "target_error_dg_detected": target_detected,
                "aggregate_error_dg_detected": aggregate_detected,
                "target_error_dg_exceeds_null_p95": target_exceeds,
                "aggregate_error_dg_exceeds_null_p95": aggregate_exceeds,
                "metric_agreement_detected": detected_agreement,
                "metric_agreement_exceeds_null_p95": exceeds_agreement,
                "metric_inconsistent_away_from_target": bool(not detected_agreement or not exceeds_agreement),
                "target_error_dg_score": float(target["dg_score"]),
                "aggregate_error_dg_score": float(aggregate["dg_score"]),
                "dg_score_difference_target_minus_aggregate": float(target["dg_score"] - aggregate["dg_score"]),
                "target_error_temporary_away_fraction": float(target["temporary_away_fraction"]),
                "aggregate_error_temporary_away_fraction": float(aggregate["temporary_away_fraction"]),
                "temporary_away_fraction_difference_target_minus_aggregate": float(
                    target["temporary_away_fraction"] - aggregate["temporary_away_fraction"]
                ),
                "target_error_null_p_value": float(target["null_p_value"]),
                "aggregate_error_null_p_value": float(aggregate["null_p_value"]),
                "sensitivity_classification": classification,
            }
        )
    return pd.DataFrame(records)


def summarize_shape_dg(dg_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize S14 evidence by perturbation, policy, and metric."""

    group_cols = ["perturbation_type", "policy_id", "policy_family", "metric_id"]
    rows = (
        dg_df.groupby(group_cols, as_index=False)
        .agg(
            runs=("metric_id", "count"),
            computed_runs=("analysis_status", lambda values: int(np.sum(pd.Series(values).eq("computed")))),
            mean_initial_error=("initial_error", "mean"),
            mean_final_error=("final_error", "mean"),
            mean_net_improvement=("net_improvement", "mean"),
            mean_temporary_away_fraction=("temporary_away_fraction", "mean"),
            mean_max_prefix_drawup=("max_prefix_drawup", "mean"),
            mean_dg_score=("dg_score", "mean"),
            dg_detected_rate=("dg_detected", "mean"),
            dg_exceeds_null_p95_rate=("dg_exceeds_null_p95", "mean"),
            dg_exceeds_null_p99_rate=("dg_exceeds_null_p99", "mean"),
            median_null_p_value=("null_p_value", "median"),
            mean_null_p95_dg_score=("null_p95_dg_score", "mean"),
            mean_s12_path_curvature=("s12_path_curvature", "mean"),
            mean_s12_temporary_away_fraction=("s12_temporary_away_fraction", "mean"),
            s13_local_context_join_rate=("s13_local_context_joined", "mean"),
            s13_global_endpoint_context_available_rate=("s13_global_endpoint_context_available", "mean"),
        )
        .sort_values(group_cols)
    )
    if not sensitivity_df.empty:
        sensitivity_summary = (
            sensitivity_df.groupby(["perturbation_type", "policy_id"], as_index=False)
            .agg(
                metric_inconsistency_rate=("metric_inconsistent_away_from_target", "mean"),
                target_and_aggregate_detected_agreement_rate=("metric_agreement_detected", "mean"),
                target_and_aggregate_null_exceed_agreement_rate=("metric_agreement_exceeds_null_p95", "mean"),
            )
        )
        rows = rows.merge(sensitivity_summary, on=["perturbation_type", "policy_id"], how="left")
    rows["group_outcome_classification"] = rows.apply(_classify_group_outcome, axis=1)
    return rows


def _classify_group_outcome(row: pd.Series) -> str:
    if row["computed_runs"] == 0:
        return "blocked"
    if row["dg_exceeds_null_p95_rate"] >= 0.25:
        return "supportive_for_metric_specific_shape_dg"
    if row["dg_detected_rate"] == 0.0:
        return "constraining_no_constructive_dg"
    return "null_constructive_dg_not_above_matched_null"


def summarize_null_validation(dg_df: pd.DataFrame) -> pd.DataFrame:
    rows = (
        dg_df.groupby(["metric_id", "null_match_status"], as_index=False)
        .agg(
            rows=("metric_id", "count"),
            min_null_replicates=("null_replicates", "min"),
            mean_null_length_match_rate=("null_length_match_rate", "mean"),
            max_null_initial_abs_error=("max_null_initial_abs_error", "max"),
            max_null_endpoint_abs_error=("max_null_endpoint_abs_error", "max"),
            mean_null_dg_detection_rate=("null_dg_detection_rate", "mean"),
            max_null_p95_dg_score=("null_p95_dg_score", "max"),
        )
        .sort_values(["metric_id", "null_match_status"])
    )
    rows["research_step_id"] = STEP_ID
    return rows[
        [
            "research_step_id",
            "metric_id",
            "null_match_status",
            "rows",
            "min_null_replicates",
            "mean_null_length_match_rate",
            "max_null_initial_abs_error",
            "max_null_endpoint_abs_error",
            "mean_null_dg_detection_rate",
            "max_null_p95_dg_score",
        ]
    ]


def validate_shape_dg_tables(
    *,
    dg_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    null_validation_df: pd.DataFrame,
    source_inputs_df: pd.DataFrame,
    route_context_df: pd.DataFrame,
    s13_context_df: pd.DataFrame,
    required_artifact_paths: Sequence[Path] | None,
) -> pd.DataFrame:
    """Return S14 validation rows."""

    rows: list[dict[str, Any]] = []
    source_by_type = {row["input_type"]: row for row in source_inputs_df.to_dict(orient="records")}
    run_count = int(dg_df[list(RUN_KEY_COLUMNS)].drop_duplicates().shape[0]) if not dg_df.empty else 0
    metric_count = int(dg_df["metric_id"].nunique()) if "metric_id" in dg_df.columns else 0
    expected_dg_rows = run_count * len(METRIC_SPECS)
    rows.append(
        _validation_row(
            "s08_trace_input_loaded",
            "input_provenance",
            bool(source_by_type.get("regeneration_trace_table", {}).get("exists") and run_count == 180 and len(dg_df) == 360),
            {"required_input": "S08 regeneration trace table", "expected_runs": 180, "expected_dg_rows": 360},
            {"runs": run_count, "dg_rows": len(dg_df), "source": source_by_type.get("regeneration_trace_table", {})},
            "S08 regeneration traces provide the primary time series for S14.",
        )
    )
    rows.append(
        _validation_row(
            "s05_metric_context_loaded",
            "input_provenance",
            bool(
                source_by_type.get("metric_catalog", {}).get("exists")
                and source_by_type.get("regeneration_metric_rows", {}).get("exists")
                and {"target_error_proxy", "s05_aggregate_morphospace_error"}.issubset(set(dg_df["metric_id"]))
            ),
            {"S05_catalog": True, "S08_metric_rows": True, "metric_ids": list(METRIC_SPECS)},
            {
                "metric_catalog_rows": source_by_type.get("metric_catalog", {}).get("row_count"),
                "regeneration_metric_rows": source_by_type.get("regeneration_metric_rows", {}).get("row_count"),
                "metric_ids": sorted(set(dg_df["metric_id"])),
            },
            "S14 uses target-error and S05 aggregate morphospace-error trajectory columns as sensitivity metrics.",
        )
    )
    rows.append(
        _validation_row(
            "s12_route_context_joined",
            "context_join",
            bool(len(route_context_df) == 180 and dg_df["s12_context_joined"].all()),
            {"s12_s08_route_rows": 180, "all_dg_rows_joined": True},
            {
                "route_context_rows": len(route_context_df),
                "joined_rows": int(dg_df["s12_context_joined"].sum()) if "s12_context_joined" in dg_df.columns else 0,
            },
            "S12 route curvature and temporary-away context joined to all S08 local-policy DG rows.",
        )
    )
    rows.append(
        _validation_row(
            "s13_context_joined_without_expanding_local_access",
            "context_join",
            bool(
                len(s13_context_df) >= 450
                and dg_df["s13_local_context_joined"].all()
                and not dg_df["s13_global_state_access"].fillna(False).any()
                and not dg_df["s13_local_policy_information_access_changed"].fillna(True).any()
                and dg_df["s13_global_endpoint_context_available"].all()
            ),
            {"s13_s08_context_rows_min": 450, "local_rows_joined": True, "local_global_access": False},
            {
                "s13_context_rows": len(s13_context_df),
                "local_joined_rows": int(dg_df["s13_local_context_joined"].sum()),
                "global_context_available_rows": int(dg_df["s13_global_endpoint_context_available"].sum()),
            },
            "S13 local access labels join to DG rows and global baselines are retained only as endpoint context.",
        )
    )
    matched = dg_df[dg_df["analysis_status"].eq("computed")]
    max_endpoint_error = float(matched["max_null_endpoint_abs_error"].max()) if not matched.empty else np.nan
    rows.append(
        _validation_row(
            "dg_nulls_length_and_endpoint_matched",
            "null_model",
            bool(
                not matched.empty
                and matched["null_match_status"].eq("matched").all()
                and (matched["null_length_match_rate"] == 1.0).all()
                and max_endpoint_error <= 1e-10
            ),
            {"null_match_status": "matched", "length_match_rate": 1.0, "max_endpoint_abs_error_lte": 1e-10},
            {
                "computed_rows": len(matched),
                "statuses": sorted(set(dg_df["null_match_status"])),
                "min_length_match_rate": float(matched["null_length_match_rate"].min()) if not matched.empty else None,
                "max_endpoint_abs_error": max_endpoint_error,
            },
            "Increment-permutation nulls preserve each trajectory's length and endpoint error.",
        )
    )
    rows.append(
        _validation_row(
            "dg_metrics_computed_for_target_and_aggregate",
            "output_completeness",
            bool(len(dg_df) == expected_dg_rows and metric_count == len(METRIC_SPECS) and dg_df["analysis_status"].eq("computed").all()),
            {"rows_equal_runs_times_metrics": True, "metric_count": len(METRIC_SPECS)},
            {"runs": run_count, "metric_count": metric_count, "rows": len(dg_df), "summary_rows": len(summary_df)},
            "Observed DG metrics were computed for both target-error and S05 aggregate-error series.",
        )
    )
    pvals = pd.to_numeric(dg_df["null_p_value"], errors="coerce")
    rows.append(
        _validation_row(
            "null_p_values_bounded_and_replicated",
            "null_model",
            bool(pvals.between(0.0, 1.0).all() and (dg_df["null_replicates"] > 0).all()),
            {"p_values_between_0_and_1": True, "null_replicates_positive": True},
            {
                "min_p_value": float(pvals.min()),
                "max_p_value": float(pvals.max()),
                "min_null_replicates": int(dg_df["null_replicates"].min()),
                "null_validation_rows": len(null_validation_df),
            },
            "Matched-null one-sided p-values are bounded and based on positive replicate counts.",
        )
    )
    inconsistency_rate = (
        float(sensitivity_df["metric_inconsistent_away_from_target"].mean()) if not sensitivity_df.empty else np.nan
    )
    rows.append(
        _validation_row(
            "metric_sensitivity_tested",
            "metric_sensitivity",
            bool(len(sensitivity_df) == run_count and "sensitivity_classification" in sensitivity_df.columns),
            {"sensitivity_rows_equal_runs": True, "classification_present": True},
            {
                "sensitivity_rows": len(sensitivity_df),
                "run_count": run_count,
                "metric_inconsistency_rate": inconsistency_rate,
                "classifications": sorted(set(sensitivity_df["sensitivity_classification"])) if not sensitivity_df.empty else [],
            },
            "Target-error and aggregate-error DG calls were compared per run to test whether away-from-target is metric stable.",
        )
    )
    artifact_observed = {}
    artifact_success = True
    if required_artifact_paths is not None:
        for path in required_artifact_paths:
            artifact_observed[str(path)] = {"exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
        artifact_success = all(value["exists"] and value["size_bytes"] > 0 for value in artifact_observed.values())
    rows.append(
        _validation_row(
            "required_s14_artifacts_present",
            "artifact_completeness",
            bool(artifact_success),
            {"required_artifacts_present": True},
            artifact_observed,
            "Required S14 result table, example figure, and supporting compact artifacts are present.",
        )
    )
    return pd.DataFrame(rows)


def _validation_row(
    validation_case: str,
    case_type: str,
    success: bool,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    detail: str,
) -> dict[str, Any]:
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "validation_case": validation_case,
        "case_type": case_type,
        "success": bool(success),
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": detail,
    }


def summarize_for_report(dg_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> dict[str, Any]:
    """Return headline S14 statistics for scripts and reports."""

    computed = dg_df[dg_df["analysis_status"].eq("computed")]
    target = computed[computed["metric_id"].eq("target_error_proxy")]
    aggregate = computed[computed["metric_id"].eq("s05_aggregate_morphospace_error")]
    return {
        "dg_rows": int(len(dg_df)),
        "run_count": int(dg_df[list(RUN_KEY_COLUMNS)].drop_duplicates().shape[0]) if not dg_df.empty else 0,
        "target_dg_detected_rate": float(target["dg_detected"].mean()) if not target.empty else np.nan,
        "aggregate_dg_detected_rate": float(aggregate["dg_detected"].mean()) if not aggregate.empty else np.nan,
        "target_dg_exceeds_null_p95_rate": float(target["dg_exceeds_null_p95"].mean()) if not target.empty else np.nan,
        "aggregate_dg_exceeds_null_p95_rate": float(aggregate["dg_exceeds_null_p95"].mean()) if not aggregate.empty else np.nan,
        "metric_inconsistency_rate": (
            float(sensitivity_df["metric_inconsistent_away_from_target"].mean()) if not sensitivity_df.empty else np.nan
        ),
        "target_mean_temporary_away_fraction": float(target["temporary_away_fraction"].mean()) if not target.empty else np.nan,
        "aggregate_mean_temporary_away_fraction": float(aggregate["temporary_away_fraction"].mean()) if not aggregate.empty else np.nan,
    }
