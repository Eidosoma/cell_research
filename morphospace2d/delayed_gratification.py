"""Higher-dimensional Delayed Gratification utilities for E05 S14."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DG_SCHEMA_VERSION = "e05_s14_higher_dimensional_dg.v1"


@dataclass(frozen=True)
class DGDetectionConfig:
    """Thresholds for detecting temporary target-error worsening followed by recovery."""

    epsilon: float = 1e-9
    min_worsening: float = 1e-9
    min_recovery: float = 1e-9


LOWER_IS_BETTER_FEATURES = [
    "error_proxy",
    "target_energy_feature",
    "neighborhood_error_feature",
    "earth_mover_feature",
    "graph_edit_feature",
    "boundary_error_feature",
    "topology_error_feature",
    "shape_moment_error_feature",
    "hausdorff_feature",
    "edge_disagreement_feature",
]


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def _bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    return values.map(
        lambda item: default if pd.isna(item) else str(item).strip().lower() in {"true", "1", "yes"}
    ).fillna(default).astype(bool)


def _stable_seed(label: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}::{label}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _segments_from_error(values: np.ndarray, config: DGDetectionConfig) -> list[dict[str, Any]]:
    deltas = np.diff(values)
    segments: list[dict[str, Any]] = []
    current_sign = 0
    start_delta = 0
    magnitude = 0.0
    for idx, delta in enumerate(deltas):
        sign = 1 if delta > config.epsilon else -1 if delta < -config.epsilon else 0
        if sign == 0:
            continue
        if current_sign == 0:
            current_sign = sign
            start_delta = idx
            magnitude = float(delta)
            continue
        if sign == current_sign:
            magnitude += float(delta)
            continue
        segments.append(
            {
                "sign": current_sign,
                "start_index": int(start_delta),
                "end_index": int(idx),
                "delta_sum": float(magnitude),
            }
        )
        current_sign = sign
        start_delta = idx
        magnitude = float(delta)
    if current_sign != 0:
        segments.append(
            {
                "sign": current_sign,
                "start_index": int(start_delta),
                "end_index": int(len(deltas)),
                "delta_sum": float(magnitude),
            }
        )
    return segments


def detect_dg_events(
    error_values: Sequence[float],
    *,
    steps: Sequence[float] | None = None,
    config: DGDetectionConfig | None = None,
) -> pd.DataFrame:
    """Detect error-worsening runs followed by immediate recovery runs.

    A DG event is a contiguous increase in target-error proxy followed by a
    contiguous decrease. Productive DG is stricter: recovery must more than
    compensate for the temporary worsening, matching the paper's
    backtrack-then-gain interpretation.
    """

    cfg = config or DGDetectionConfig()
    values = np.asarray([_finite_float(value) for value in error_values], dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        values = values[finite_mask]
        if steps is not None:
            step_arr = np.asarray(steps, dtype=float)[finite_mask]
        else:
            step_arr = np.arange(len(values), dtype=float)
    else:
        step_arr = np.asarray(steps, dtype=float) if steps is not None else np.arange(len(values), dtype=float)
    if len(values) < 3:
        return pd.DataFrame(
            columns=[
                "event_index",
                "start_snapshot_index",
                "worsened_snapshot_index",
                "recovered_snapshot_index",
                "start_step",
                "worsened_step",
                "recovered_step",
                "start_error",
                "worsened_error",
                "recovered_error",
                "worsening_magnitude",
                "recovery_magnitude",
                "net_error_improvement_after_backtrack",
                "dg_ratio",
                "raw_dg_index",
                "productive_dg_index",
                "productive_event",
            ]
        )

    rows: list[dict[str, Any]] = []
    segments = _segments_from_error(values, cfg)
    for index, segment in enumerate(segments[:-1]):
        next_segment = segments[index + 1]
        if segment["sign"] != 1 or next_segment["sign"] != -1:
            continue
        start_idx = int(segment["start_index"])
        worsened_idx = int(segment["end_index"])
        recovered_idx = int(next_segment["end_index"])
        start_error = float(values[start_idx])
        worsened_error = float(values[worsened_idx])
        recovered_error = float(values[recovered_idx])
        worsening = worsened_error - start_error
        recovery = worsened_error - recovered_error
        if worsening < cfg.min_worsening or recovery < cfg.min_recovery:
            continue
        raw_index = (recovery - worsening) / max(worsening, cfg.epsilon)
        productive_index = max(0.0, raw_index)
        rows.append(
            {
                "event_index": int(len(rows)),
                "start_snapshot_index": start_idx,
                "worsened_snapshot_index": worsened_idx,
                "recovered_snapshot_index": recovered_idx,
                "start_step": float(step_arr[start_idx]),
                "worsened_step": float(step_arr[worsened_idx]),
                "recovered_step": float(step_arr[recovered_idx]),
                "start_error": start_error,
                "worsened_error": worsened_error,
                "recovered_error": recovered_error,
                "worsening_magnitude": float(worsening),
                "recovery_magnitude": float(recovery),
                "net_error_improvement_after_backtrack": float(start_error - recovered_error),
                "dg_ratio": float(recovery / max(worsening, cfg.epsilon)),
                "raw_dg_index": float(raw_index),
                "productive_dg_index": float(productive_index),
                "productive_event": bool(productive_index > 0.0),
            }
        )
    return pd.DataFrame(rows)


def _event_summary(event_df: pd.DataFrame) -> dict[str, Any]:
    if event_df.empty:
        return {
            "dg_event_count": 0,
            "productive_dg_event_count": 0,
            "total_error_worsening": 0.0,
            "max_error_worsening": 0.0,
            "total_recovery_after_worsening": 0.0,
            "total_raw_dg_index": 0.0,
            "total_productive_dg_index": 0.0,
            "mean_dg_ratio": 0.0,
            "max_productive_dg_index": 0.0,
            "any_dg_event": False,
            "any_productive_dg_event": False,
        }
    productive = event_df["productive_event"].astype(bool)
    return {
        "dg_event_count": int(len(event_df)),
        "productive_dg_event_count": int(productive.sum()),
        "total_error_worsening": float(event_df["worsening_magnitude"].sum()),
        "max_error_worsening": float(event_df["worsening_magnitude"].max()),
        "total_recovery_after_worsening": float(event_df["recovery_magnitude"].sum()),
        "total_raw_dg_index": float(event_df["raw_dg_index"].sum()),
        "total_productive_dg_index": float(event_df["productive_dg_index"].sum()),
        "mean_dg_ratio": float(event_df["dg_ratio"].mean()),
        "max_productive_dg_index": float(event_df["productive_dg_index"].max()),
        "any_dg_event": True,
        "any_productive_dg_event": bool(productive.any()),
    }


def _ordered_trajectory(group: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [column for column in ["step", "snapshot_index"] if column in group.columns]
    if not sort_cols:
        return group.reset_index(drop=True)
    return group.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _metadata_from_group(group: pd.DataFrame) -> dict[str, Any]:
    ordered = _ordered_trajectory(group)
    row = ordered.iloc[0]
    metadata_cols = [
        "source_step_id",
        "source_trace_kind",
        "source_trace_path",
        "source_run_path",
        "trajectory_id",
        "run_id",
        "target_id",
        "task_id",
        "motif",
        "policy_id",
        "policy_family",
        "seed",
        "perturbation_id",
        "perturbation_family",
        "size_class",
        "size_key",
        "scale_rule_id",
        "is_heldout_size",
        "information_scope",
        "control_class",
        "is_local_only_policy",
        "is_global_information_baseline",
        "uses_target_map",
        "uses_global_gradient",
        "uses_organizer",
        "node_count",
        "device",
        "convergence_basin",
        "route_cluster_id",
        "failure_cluster_id",
    ]
    return {column: row[column] if column in ordered.columns else None for column in metadata_cols}


def attach_control_metadata(state_df: pd.DataFrame, control_df: pd.DataFrame | None) -> pd.DataFrame:
    """Attach S13 run-level control metadata while preserving state-feature rows."""

    out = state_df.copy()
    if control_df is None or control_df.empty:
        for column in ["control_class", "baseline_flag", "target_error_metric", "intervention_complexity_score", "local_only_leakage_violation"]:
            if column not in out.columns:
                out[column] = np.nan
        return out
    cols = [
        "source_step_id",
        "run_id",
        "information_scope",
        "control_class",
        "baseline_flag",
        "is_local_only_policy",
        "is_global_information_baseline",
        "uses_target_map",
        "uses_global_gradient",
        "uses_organizer",
        "target_error_metric",
        "intervention_complexity_score",
        "local_only_leakage_violation",
        "hidden_global_leakage_flag",
        "policy_audit_success",
        "robustness_score",
    ]
    available = [column for column in cols if column in control_df.columns]
    meta = control_df[available].drop_duplicates(["source_step_id", "run_id"])
    merged = out.merge(meta, on=["source_step_id", "run_id"], how="left", suffixes=("", "_s13"))
    for column in available:
        if column in {"source_step_id", "run_id"}:
            continue
        s13_column = f"{column}_s13"
        if s13_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna(), merged[s13_column])
        else:
            merged[column] = merged[s13_column]
        merged = merged.drop(columns=[s13_column])
    return merged


def trajectory_dg_summary(
    state_df: pd.DataFrame,
    *,
    config: DGDetectionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-trajectory DG summaries and event rows."""

    cfg = config or DGDetectionConfig()
    summary_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    for trajectory_id, group in state_df.groupby("trajectory_id", sort=False):
        ordered = _ordered_trajectory(group)
        errors = _numeric_series(ordered, "error_proxy").to_numpy(dtype=float)
        steps = _numeric_series(ordered, "step").to_numpy(dtype=float)
        event_df = detect_dg_events(errors, steps=steps, config=cfg)
        metadata = _metadata_from_group(ordered)
        if not event_df.empty:
            for key, value in metadata.items():
                event_df[key] = value
            event_df["schema_version"] = DG_SCHEMA_VERSION
            event_df["research_step_id"] = "S14"
            event_df["error_metric"] = "s12_error_proxy"
            event_frames.append(event_df)

        event_summary = _event_summary(event_df)
        positive_error_steps = np.diff(errors)
        positive_error_steps = positive_error_steps[np.isfinite(positive_error_steps) & (positive_error_steps > cfg.epsilon)]
        duration = float(steps[-1] - steps[0]) if len(steps) else 0.0
        row = {
            "schema_version": DG_SCHEMA_VERSION,
            "research_step_id": "S14",
            **metadata,
            "trajectory_id": str(trajectory_id),
            "snapshot_count": int(len(ordered)),
            "duration_steps": duration,
            "initial_error": float(errors[0]) if len(errors) else math.nan,
            "final_error": float(errors[-1]) if len(errors) else math.nan,
            "best_error": float(np.nanmin(errors)) if len(errors) else math.nan,
            "worst_error": float(np.nanmax(errors)) if len(errors) else math.nan,
            "error_reduction": float(errors[0] - errors[-1]) if len(errors) else math.nan,
            "relative_error_reduction": float((errors[0] - errors[-1]) / max(abs(errors[0]), cfg.epsilon)) if len(errors) else math.nan,
            "positive_delta_count": int(len(positive_error_steps)),
            "total_positive_error_delta": float(positive_error_steps.sum()) if len(positive_error_steps) else 0.0,
            "dg_event_density_per_snapshot": float(event_summary["dg_event_count"] / max(len(ordered) - 1, 1)),
            **event_summary,
        }
        summary_rows.append(row)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return pd.DataFrame(summary_rows), events


def hand_constructed_validation_examples(config: DGDetectionConfig | None = None) -> pd.DataFrame:
    """Validate DG event detection on simple hand-constructed trajectories."""

    cfg = config or DGDetectionConfig()
    examples = [
        ("monotone_improvement", [1.0, 0.8, 0.6, 0.4], 0, 0),
        ("productive_single_backtrack", [1.0, 0.7, 0.8, 0.4], 1, 1),
        ("nonproductive_backtrack", [1.0, 0.7, 0.9, 0.8], 1, 0),
        ("two_productive_backtracks", [1.0, 0.8, 0.9, 0.55, 0.6, 0.2], 2, 2),
        ("flat_then_backtrack", [1.0, 1.0, 0.9, 0.95, 0.7], 1, 1),
    ]
    rows: list[dict[str, Any]] = []
    for name, series, expected_events, expected_productive in examples:
        events = detect_dg_events(series, steps=list(range(len(series))), config=cfg)
        summary = _event_summary(events)
        rows.append(
            {
                "schema_version": DG_SCHEMA_VERSION,
                "research_step_id": "S14",
                "example_id": name,
                "error_series": ",".join(f"{value:.6g}" for value in series),
                "expected_dg_event_count": expected_events,
                "observed_dg_event_count": int(summary["dg_event_count"]),
                "expected_productive_dg_event_count": expected_productive,
                "observed_productive_dg_event_count": int(summary["productive_dg_event_count"]),
                "validation_success": bool(
                    int(summary["dg_event_count"]) == expected_events
                    and int(summary["productive_dg_event_count"]) == expected_productive
                ),
                "total_productive_dg_index": float(summary["total_productive_dg_index"]),
            }
        )
    return pd.DataFrame(rows)


def synthetic_start_end_null_summary(
    summary_df: pd.DataFrame,
    *,
    replicates: int = 32,
    random_state: int = 14014,
    config: DGDetectionConfig | None = None,
) -> pd.DataFrame:
    """Generate deterministic random-bridge null trajectories with identical start/end error."""

    cfg = config or DGDetectionConfig()
    rows: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        n = int(max(_finite_float(row.get("snapshot_count"), 0), 2))
        start = _finite_float(row.get("initial_error"))
        end = _finite_float(row.get("final_error"))
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        base = np.linspace(start, end, n)
        span = max(abs(start - end), _finite_float(row.get("max_error_worsening"), 0.0), _finite_float(row.get("total_positive_error_delta"), 0.0), 0.05)
        for replicate in range(int(replicates)):
            rng = np.random.default_rng(_stable_seed(f"{row.get('trajectory_id')}::{replicate}", random_state))
            increments = rng.normal(loc=0.0, scale=1.0, size=n)
            walk = np.cumsum(increments)
            bridge = walk - np.linspace(walk[0], walk[-1], n)
            bridge = bridge - bridge[0]
            denom = np.std(bridge)
            bridge = bridge / denom if denom > 1e-12 else bridge
            series = np.clip(base + bridge * span * 0.25, 0.0, None)
            series[0] = start
            series[-1] = end
            events = detect_dg_events(series, steps=np.arange(n), config=cfg)
            metrics = _event_summary(events)
            rows.append(
                {
                    "schema_version": DG_SCHEMA_VERSION,
                    "research_step_id": "S14",
                    "null_model": "synthetic_start_end_random_bridge",
                    "observed_trajectory_id": row.get("trajectory_id"),
                    "source_step_id": row.get("source_step_id"),
                    "run_id": row.get("run_id"),
                    "target_id": row.get("target_id"),
                    "motif": row.get("motif"),
                    "policy_id": row.get("policy_id"),
                    "policy_family": row.get("policy_family"),
                    "null_replicate": int(replicate),
                    "snapshot_count": n,
                    "start_error": start,
                    "end_error": end,
                    "start_end_match_abs_error": float(abs(series[0] - start) + abs(series[-1] - end)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _is_random_local_null(df: pd.DataFrame) -> pd.Series:
    policy_family = df["policy_family"].fillna("").astype(str).str.lower() if "policy_family" in df.columns else pd.Series("", index=df.index)
    policy_id = df["policy_id"].fillna("").astype(str).str.lower() if "policy_id" in df.columns else pd.Series("", index=df.index)
    local_only = _bool_series(df, "is_local_only_policy", default=True)
    global_baseline = _bool_series(df, "is_global_information_baseline", default=False)
    return (policy_family.str.contains("random") | policy_id.str.contains("random")) & local_only & ~global_baseline


def empirical_local_move_null_matches(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Match each non-null trajectory to the closest empirical random local-move null."""

    summary = summary_df.copy()
    summary["is_empirical_random_local_move_null"] = _is_random_local_null(summary)
    null_pool = summary[summary["is_empirical_random_local_move_null"]].copy()
    rows: list[dict[str, Any]] = []
    if null_pool.empty:
        return pd.DataFrame()

    match_levels = [
        ("source_target_policy_context", ["source_step_id", "target_id", "motif"]),
        ("source_motif", ["source_step_id", "motif"]),
        ("source_only", ["source_step_id"]),
        ("all_random_local_nulls", []),
    ]
    scale_cols = ["initial_error", "final_error", "snapshot_count", "duration_steps"]
    scale = summary[scale_cols].apply(pd.to_numeric, errors="coerce").std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    for _, observed in summary[~summary["is_empirical_random_local_move_null"]].iterrows():
        chosen_level = None
        candidates = pd.DataFrame()
        for level_name, cols in match_levels:
            candidates = null_pool
            for column in cols:
                candidates = candidates[candidates[column].astype(str).eq(str(observed.get(column)))]
            if not candidates.empty:
                chosen_level = level_name
                break
        if candidates.empty:
            rows.append(
                {
                    "schema_version": DG_SCHEMA_VERSION,
                    "research_step_id": "S14",
                    "observed_trajectory_id": observed.get("trajectory_id"),
                    "observed_run_id": observed.get("run_id"),
                    "source_step_id": observed.get("source_step_id"),
                    "matched": False,
                    "match_level": "none",
                }
            )
            continue
        obs_vec = pd.Series({column: _finite_float(observed.get(column), 0.0) for column in scale_cols})
        cand_values = candidates[scale_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        distances = np.sqrt(np.square((cand_values - obs_vec) / scale).sum(axis=1))
        match_idx = distances.idxmin()
        match = candidates.loc[match_idx]
        rows.append(
            {
                "schema_version": DG_SCHEMA_VERSION,
                "research_step_id": "S14",
                "observed_trajectory_id": observed.get("trajectory_id"),
                "observed_run_id": observed.get("run_id"),
                "matched_null_trajectory_id": match.get("trajectory_id"),
                "matched_null_run_id": match.get("run_id"),
                "source_step_id": observed.get("source_step_id"),
                "target_id": observed.get("target_id"),
                "motif": observed.get("motif"),
                "policy_id": observed.get("policy_id"),
                "policy_family": observed.get("policy_family"),
                "matched_null_policy_id": match.get("policy_id"),
                "matched_null_policy_family": match.get("policy_family"),
                "matched": True,
                "match_level": chosen_level,
                "match_distance": float(distances.loc[match_idx]),
                "observed_initial_error": _finite_float(observed.get("initial_error")),
                "observed_final_error": _finite_float(observed.get("final_error")),
                "null_initial_error": _finite_float(match.get("initial_error")),
                "null_final_error": _finite_float(match.get("final_error")),
                "start_end_distance": float(
                    abs(_finite_float(observed.get("initial_error"), 0.0) - _finite_float(match.get("initial_error"), 0.0))
                    + abs(_finite_float(observed.get("final_error"), 0.0) - _finite_float(match.get("final_error"), 0.0))
                ),
                "observed_dg_event_count": int(observed.get("dg_event_count", 0)),
                "null_dg_event_count": int(match.get("dg_event_count", 0)),
                "observed_productive_dg_event_count": int(observed.get("productive_dg_event_count", 0)),
                "null_productive_dg_event_count": int(match.get("productive_dg_event_count", 0)),
                "observed_total_productive_dg_index": _finite_float(observed.get("total_productive_dg_index"), 0.0),
                "null_total_productive_dg_index": _finite_float(match.get("total_productive_dg_index"), 0.0),
                "observed_minus_null_productive_dg_index": _finite_float(observed.get("total_productive_dg_index"), 0.0)
                - _finite_float(match.get("total_productive_dg_index"), 0.0),
            }
        )
    return pd.DataFrame(rows)


def normalization_artifact_audit(
    state_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    config: DGDetectionConfig | None = None,
    tiny_threshold: float = 1e-6,
) -> pd.DataFrame:
    """Audit whether primary error-proxy DG is supported by component metrics."""

    cfg = config or DGDetectionConfig()
    summary_lookup = summary_df.set_index("trajectory_id")
    rows: list[dict[str, Any]] = []
    for trajectory_id, group in state_df.groupby("trajectory_id", sort=False):
        ordered = _ordered_trajectory(group)
        row = summary_lookup.loc[trajectory_id] if trajectory_id in summary_lookup.index else pd.Series(dtype=object)
        component_metrics: list[str] = []
        component_dg = 0
        component_productive = 0
        component_event_counts: dict[str, int] = {}
        for column in LOWER_IS_BETTER_FEATURES:
            if column not in ordered.columns or column == "error_proxy":
                continue
            values = _numeric_series(ordered, column)
            if values.notna().sum() < 3:
                continue
            events = detect_dg_events(values.to_numpy(dtype=float), steps=_numeric_series(ordered, "step").to_numpy(dtype=float), config=cfg)
            component_metrics.append(column)
            event_count = int(len(events))
            productive_count = int(events["productive_event"].astype(bool).sum()) if len(events) else 0
            component_event_counts[column] = event_count
            component_dg += int(event_count > 0)
            component_productive += int(productive_count > 0)
        primary_event_count = int(row.get("dg_event_count", 0)) if len(row) else 0
        primary_productive_count = int(row.get("productive_dg_event_count", 0)) if len(row) else 0
        total_worsening = _finite_float(row.get("total_error_worsening"), 0.0) if len(row) else 0.0
        comparable_count = len(component_metrics)
        rows.append(
            {
                "schema_version": DG_SCHEMA_VERSION,
                "research_step_id": "S14",
                **_metadata_from_group(ordered),
                "trajectory_id": str(trajectory_id),
                "primary_error_metric": "s12_error_proxy",
                "primary_dg_event_count": primary_event_count,
                "primary_productive_dg_event_count": primary_productive_count,
                "primary_total_error_worsening": total_worsening,
                "comparable_component_metric_count": comparable_count,
                "component_metrics_checked": ",".join(component_metrics),
                "component_metric_event_counts": ";".join(f"{key}:{value}" for key, value in sorted(component_event_counts.items())),
                "component_metrics_with_any_dg_count": int(component_dg),
                "component_metrics_with_productive_dg_count": int(component_productive),
                "primary_dg_supported_by_component": bool(primary_event_count == 0 or comparable_count == 0 or component_dg > 0),
                "normalization_artifact_flag": bool(primary_event_count > 0 and comparable_count > 0 and component_dg == 0),
                "productive_normalization_artifact_flag": bool(primary_productive_count > 0 and comparable_count > 0 and component_productive == 0),
                "tiny_primary_worsening_flag": bool(primary_event_count > 0 and total_worsening <= tiny_threshold),
                "audit_coverage": "component_metrics_available" if comparable_count > 0 else "primary_proxy_only",
            }
        )
    return pd.DataFrame(rows)


def source_policy_dg_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate DG trajectory metrics by source, motif, and policy family."""

    rows: list[dict[str, Any]] = []
    group_cols = ["source_step_id", "motif", "policy_family", "control_class", "information_scope"]
    safe = summary_df.copy()
    for column in group_cols:
        if column not in safe.columns:
            safe[column] = ""
    for keys, group in safe.groupby(group_cols, dropna=False, sort=True):
        source_step_id, motif, policy_family, control_class, information_scope = keys
        rows.append(
            {
                "schema_version": DG_SCHEMA_VERSION,
                "research_step_id": "S14",
                "source_step_id": source_step_id,
                "motif": motif,
                "policy_family": policy_family,
                "control_class": control_class,
                "information_scope": information_scope,
                "trajectory_count": int(len(group)),
                "any_dg_event_rate": float(group["any_dg_event"].astype(bool).mean()),
                "productive_dg_event_rate": float(group["any_productive_dg_event"].astype(bool).mean()),
                "mean_dg_event_count": float(group["dg_event_count"].astype(float).mean()),
                "mean_productive_dg_event_count": float(group["productive_dg_event_count"].astype(float).mean()),
                "mean_total_error_worsening": float(group["total_error_worsening"].astype(float).mean()),
                "mean_total_productive_dg_index": float(group["total_productive_dg_index"].astype(float).mean()),
                "mean_initial_error": float(group["initial_error"].astype(float).mean()),
                "mean_final_error": float(group["final_error"].astype(float).mean()),
                "mean_relative_error_reduction": float(group["relative_error_reduction"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def null_comparison_summary(
    summary_df: pd.DataFrame,
    synthetic_null_df: pd.DataFrame,
    empirical_match_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize observed DG against synthetic and empirical nulls."""

    rows: list[dict[str, Any]] = []
    observed_mean = float(summary_df["total_productive_dg_index"].astype(float).mean()) if len(summary_df) else math.nan
    observed_rate = float(summary_df["any_productive_dg_event"].astype(bool).mean()) if len(summary_df) else math.nan
    if len(synthetic_null_df):
        null_by_obs = synthetic_null_df.groupby("observed_trajectory_id", sort=False).agg(
            synthetic_mean_productive_dg_index=("total_productive_dg_index", "mean"),
            synthetic_productive_event_rate=("any_productive_dg_event", "mean"),
            synthetic_replicates=("null_replicate", "count"),
        ).reset_index()
        merged = summary_df.merge(null_by_obs, left_on="trajectory_id", right_on="observed_trajectory_id", how="left")
        rows.append(
            {
                "schema_version": DG_SCHEMA_VERSION,
                "research_step_id": "S14",
                "comparison_type": "observed_vs_synthetic_start_end_random_bridge",
                "observed_trajectory_count": int(len(summary_df)),
                "matched_trajectory_count": int(merged["synthetic_mean_productive_dg_index"].notna().sum()),
                "observed_mean_productive_dg_index": observed_mean,
                "null_mean_productive_dg_index": float(merged["synthetic_mean_productive_dg_index"].mean()),
                "observed_minus_null_productive_dg_index": float(observed_mean - merged["synthetic_mean_productive_dg_index"].mean()),
                "observed_productive_event_rate": observed_rate,
                "null_productive_event_rate": float(merged["synthetic_productive_event_rate"].mean()),
                "observed_minus_null_productive_event_rate": float(observed_rate - merged["synthetic_productive_event_rate"].mean()),
            }
        )
    matched = empirical_match_df[empirical_match_df.get("matched", pd.Series(dtype=bool)).astype(bool)] if len(empirical_match_df) else pd.DataFrame()
    if len(matched):
        rows.append(
            {
                "schema_version": DG_SCHEMA_VERSION,
                "research_step_id": "S14",
                "comparison_type": "observed_vs_empirical_random_local_move_null",
                "observed_trajectory_count": int(summary_df[~_is_random_local_null(summary_df)].shape[0]),
                "matched_trajectory_count": int(len(matched)),
                "observed_mean_productive_dg_index": float(matched["observed_total_productive_dg_index"].astype(float).mean()),
                "null_mean_productive_dg_index": float(matched["null_total_productive_dg_index"].astype(float).mean()),
                "observed_minus_null_productive_dg_index": float(matched["observed_minus_null_productive_dg_index"].astype(float).mean()),
                "observed_productive_event_rate": float((matched["observed_productive_dg_event_count"].astype(float) > 0.0).mean()),
                "null_productive_event_rate": float((matched["null_productive_dg_event_count"].astype(float) > 0.0).mean()),
                "observed_minus_null_productive_event_rate": float(
                    (matched["observed_productive_dg_event_count"].astype(float) > 0.0).mean()
                    - (matched["null_productive_dg_event_count"].astype(float) > 0.0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def validation_rows(
    *,
    state_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    event_df: pd.DataFrame,
    hand_validation_df: pd.DataFrame,
    synthetic_null_df: pd.DataFrame,
    empirical_match_df: pd.DataFrame,
    normalization_audit_df: pd.DataFrame,
    expected_sources: Iterable[str] = ("S07", "S08", "S09", "S10", "S11"),
) -> pd.DataFrame:
    """Build S14 validation checks."""

    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: Any) -> None:
        rows.append(
            {
                "schema_version": DG_SCHEMA_VERSION,
                "research_step_id": "S14",
                "check_id": check_id,
                "success": bool(success),
                "detail": detail,
            }
        )

    expected = set(expected_sources)
    observed = set(state_df["source_step_id"].astype(str).unique()) if len(state_df) else set()
    add("expected_sources_present", expected.issubset(observed), {"expected": sorted(expected), "observed": sorted(observed)})
    add("trajectory_and_run_ids_preserved", bool(summary_df["trajectory_id"].notna().all() and summary_df["run_id"].notna().all()), "trajectory_id and run_id are populated")
    add("one_summary_per_trajectory", int(summary_df["trajectory_id"].nunique()) == len(summary_df), {"summaryRows": int(len(summary_df)), "uniqueTrajectories": int(summary_df["trajectory_id"].nunique())})
    add("dg_event_ids_preserved", bool(event_df.empty or event_df[["trajectory_id", "run_id", "source_step_id"]].notna().all().all()), {"eventRows": int(len(event_df))})
    add("hand_constructed_validation_passed", bool(hand_validation_df["validation_success"].astype(bool).all()), {"rows": int(len(hand_validation_df))})
    max_start_end_deviation = float(synthetic_null_df["start_end_match_abs_error"].max()) if len(synthetic_null_df) else math.nan
    add("synthetic_start_end_nulls_matched", bool(len(synthetic_null_df) > 0 and max_start_end_deviation <= 1e-9), {"rows": int(len(synthetic_null_df)), "maxDeviation": max_start_end_deviation})
    matched_count = int(empirical_match_df["matched"].astype(bool).sum()) if len(empirical_match_df) and "matched" in empirical_match_df.columns else 0
    add("empirical_local_move_nulls_matched_where_available", matched_count > 0, {"matchedRows": matched_count, "totalRows": int(len(empirical_match_df))})
    add("normalization_audit_written", len(normalization_audit_df) == len(summary_df), {"rows": int(len(normalization_audit_df)), "summaryRows": int(len(summary_df))})
    finite_cols = ["dg_event_count", "productive_dg_event_count", "total_error_worsening", "total_productive_dg_index"]
    finite_ok = np.isfinite(summary_df[finite_cols].astype(float).to_numpy()).all() if len(summary_df) else False
    add("dg_summary_metrics_finite", bool(finite_ok), {"rows": int(len(summary_df)), "columns": finite_cols})
    component_coverage = float((normalization_audit_df["comparable_component_metric_count"].astype(float) > 0.0).mean()) if len(normalization_audit_df) else 0.0
    add("normalization_component_coverage_recorded", component_coverage > 0.0, {"componentCoverage": component_coverage})
    return pd.DataFrame(rows)
