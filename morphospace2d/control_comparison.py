"""Local-versus-global control comparison utilities for E05 S13."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


CONTROL_COMPARISON_SCHEMA_VERSION = "e05_s13_local_global_control.v1"


@dataclass(frozen=True)
class ControlSourceSpec:
    """Run-result and policy-audit artifacts for one upstream control source."""

    source_step_id: str
    run_path: str
    leakage_audit_path: str | None = None
    source_kind: str = "benchmark"


def standard_control_source_specs(artifacts_dir: str | Path = "/artifacts") -> list[ControlSourceSpec]:
    """Return S07-S11 run artifacts used by S13."""

    step = Path(artifacts_dir) / "research_steps"
    return [
        ControlSourceSpec(
            "S07",
            str(step / "S07" / "scrambled_embryo_run_results.parquet"),
            str(step / "S07" / "policy_leakage_audit.parquet"),
            "scrambled_recovery",
        ),
        ControlSourceSpec(
            "S08",
            str(step / "S08" / "regeneration_run_results.parquet"),
            str(step / "S08" / "policy_leakage_audit.parquet"),
            "regeneration",
        ),
        ControlSourceSpec(
            "S09",
            str(step / "S09" / "scaling_run_results.parquet"),
            str(step / "S09" / "policy_scaling_leakage_audit.parquet"),
            "scale_transfer",
        ),
        ControlSourceSpec(
            "S10",
            str(step / "S10" / "symmetry_breaking_run_results.parquet"),
            str(step / "S10" / "policy_symmetry_leakage_audit.parquet"),
            "symmetry_breaking",
        ),
        ControlSourceSpec(
            "S11",
            str(step / "S11" / "gpu_sweep_run_results.parquet"),
            str(step / "S11" / "gpu_policy_leakage_audit.parquet"),
            "gpu_label_dynamics",
        ),
    ]


def available_control_sources(specs: Sequence[ControlSourceSpec]) -> tuple[list[ControlSourceSpec], pd.DataFrame]:
    """Filter to available sources and return a source-artifact catalog."""

    available: list[ControlSourceSpec] = []
    rows: list[dict[str, Any]] = []
    for spec in specs:
        run_path = Path(spec.run_path)
        audit_path = Path(spec.leakage_audit_path) if spec.leakage_audit_path else None
        is_available = run_path.exists()
        if is_available:
            available.append(spec)
        rows.append(
            {
                "schema_version": CONTROL_COMPARISON_SCHEMA_VERSION,
                "research_step_id": "S13",
                "source_step_id": spec.source_step_id,
                "source_kind": spec.source_kind,
                "run_path": str(run_path),
                "leakage_audit_path": str(audit_path) if audit_path else None,
                "run_available": bool(run_path.exists()),
                "leakage_audit_available": bool(audit_path.exists()) if audit_path else False,
                "run_size_bytes": int(run_path.stat().st_size) if run_path.exists() else 0,
                "leakage_audit_size_bytes": int(audit_path.stat().st_size) if audit_path and audit_path.exists() else 0,
            }
        )
    return available, pd.DataFrame(rows)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def _bool(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    return values.map(lambda item: str(item).strip().lower() in {"true", "1", "yes"}).fillna(default).astype(bool)


def _text(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=object)
    return df[column].fillna(default).astype(str)


def _first_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for column in columns:
        if column in df.columns:
            out = out.fillna(_numeric(df, column))
    return out


def _safe_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _attach_policy_audit(run: pd.DataFrame, audit_df: pd.DataFrame | None) -> pd.DataFrame:
    out = run.copy()
    if audit_df is None or audit_df.empty or "policy_id" not in audit_df.columns:
        out["policy_audit_success"] = True
        out["policy_audit_errors_json"] = "[]"
        out["policy_audit_payload_hash"] = None
        return out
    audit = audit_df.drop_duplicates("policy_id").set_index("policy_id")
    out["policy_audit_success"] = out["policy_id"].map(audit["audit_success"]) if "audit_success" in audit.columns else True
    out["policy_audit_success"] = out["policy_audit_success"].fillna(True).astype(bool)
    out["policy_audit_errors_json"] = out["policy_id"].map(audit["audit_errors_json"]) if "audit_errors_json" in audit.columns else "[]"
    out["policy_audit_errors_json"] = out["policy_audit_errors_json"].fillna("[]").map(_safe_json)
    out["policy_audit_payload_hash"] = out["policy_id"].map(audit["audit_payload_hash"]) if "audit_payload_hash" in audit.columns else None
    return out


def _attach_s12_trajectory(run: pd.DataFrame, trajectory_df: pd.DataFrame | None, source_step_id: str) -> pd.DataFrame:
    out = run.copy()
    if trajectory_df is None or trajectory_df.empty:
        out["trajectory_id"] = out["run_id"].astype(str)
        out["s12_route_joined"] = False
        return out
    cols = [
        "source_step_id",
        "run_id",
        "trajectory_id",
        "path_curvature_proxy",
        "temporary_worsening_count",
        "monotonicity_error_proxy",
        "convergence_basin",
        "route_cluster_id",
        "failure_cluster_id",
        "exact_or_near_target",
        "initial_error_proxy",
        "final_error_proxy",
        "relative_error_reduction_proxy",
    ]
    available = [column for column in cols if column in trajectory_df.columns]
    traj = trajectory_df[trajectory_df["source_step_id"].astype(str).eq(source_step_id)][available].drop_duplicates(["source_step_id", "run_id"])
    out = out.merge(traj, on=["source_step_id", "run_id"], how="left", suffixes=("", "_s12"))
    out["trajectory_id"] = out["trajectory_id"].fillna(out["run_id"].astype(str))
    out["s12_route_joined"] = out["path_curvature_proxy"].notna()
    return out


def _target_error_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    if "final_composite_error" in df.columns:
        return _numeric(df, "initial_composite_error"), _numeric(df, "final_composite_error"), _numeric(df, "best_composite_error"), "composite_error"
    if "final_hamming_error" in df.columns:
        return _numeric(df, "initial_hamming_error"), _numeric(df, "final_hamming_error"), _numeric(df, "final_hamming_error"), "hamming_error"
    if "pattern_score" in df.columns:
        final_error = (1.0 - _numeric(df, "pattern_score")).clip(lower=0.0)
        initial_error = _first_numeric(df, ["initial_error_proxy", "initial_error_proxy_s12"])
        initial_error = initial_error.fillna(1.0)
        return initial_error, final_error, final_error, "pattern_score_complement"
    final_error = _first_numeric(df, ["final_error_proxy", "final_error_proxy_s12"])
    initial_error = _first_numeric(df, ["initial_error_proxy", "initial_error_proxy_s12"])
    return initial_error, final_error, final_error, "s12_error_proxy"


def _classify_information_access(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["uses_target_map"] = _bool(out, "uses_target_map")
    out["uses_global_gradient"] = _bool(out, "uses_global_gradient")
    out["uses_organizer"] = _bool(out, "uses_organizer")
    out["uses_whole_target_leakage"] = _bool(out, "uses_whole_target_leakage")
    out["uses_hidden_target_map_leakage"] = _bool(out, "uses_hidden_target_map_leakage")
    out["uses_hidden_axis_leakage"] = _bool(out, "uses_hidden_axis_leakage")
    out["uses_target_map_leakage"] = _bool(out, "uses_target_map_leakage")
    out["uses_hidden_global_size"] = _bool(out, "uses_hidden_global_size")
    out["is_global_information_baseline"] = _bool(out, "is_global_information_baseline")
    out["is_local_only_policy"] = _bool(out, "is_local_only_policy", default=True)

    policy_family = _text(out, "policy_family").str.lower()
    policy_id = _text(out, "policy_id").str.lower()
    inferred_target_map = policy_family.str.contains("target_map") | policy_id.str.contains("target_relaxation")
    inferred_gradient = policy_family.str.contains("global_gradient") | policy_id.str.contains("global_gradient")
    inferred_organizer = policy_family.str.contains("organizer") | policy_id.str.contains("organizer")
    out["uses_target_map"] = out["uses_target_map"] | inferred_target_map
    out["uses_global_gradient"] = out["uses_global_gradient"] | inferred_gradient
    out["uses_organizer"] = out["uses_organizer"] | inferred_organizer
    out["is_global_information_baseline"] = (
        out["is_global_information_baseline"]
        | out["uses_target_map"]
        | out["uses_global_gradient"]
        | out["uses_organizer"]
    )
    out["is_local_only_policy"] = out["is_local_only_policy"] & ~out["is_global_information_baseline"]

    information_scope = _text(out, "information_scope", default="")
    information_scope = information_scope.mask(information_scope.eq(""), np.nan).astype(object)
    out["information_scope"] = information_scope
    out.loc[out["uses_target_map"], "information_scope"] = "explicit_target_map_baseline"
    out.loc[out["uses_global_gradient"], "information_scope"] = "explicit_global_gradient_baseline"
    out.loc[out["uses_organizer"], "information_scope"] = "explicit_organizer_baseline"
    out.loc[out["is_local_only_policy"] & out["information_scope"].isna(), "information_scope"] = "local_target_map_free"
    out["information_scope"] = out["information_scope"].fillna("target_map_free_nonlocal_unspecified")

    control_class = np.full(len(out), "local_only", dtype=object)
    control_class = np.where(out["uses_target_map"], "target_map_baseline", control_class)
    control_class = np.where(out["uses_global_gradient"], "global_gradient_baseline", control_class)
    control_class = np.where(out["uses_organizer"], "organizer_baseline", control_class)
    control_class = np.where(
        out["is_global_information_baseline"] & ~out["uses_target_map"] & ~out["uses_global_gradient"] & ~out["uses_organizer"],
        "other_global_information",
        control_class,
    )
    out["control_class"] = control_class
    out["baseline_flag"] = out["is_global_information_baseline"].map(lambda flag: "explicit_global_information_baseline" if flag else "local_or_target_map_free")
    return out


def _intervention_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    information_score = pd.Series(0.0, index=out.index)
    information_score += out["uses_global_gradient"].astype(float) * 3.0
    information_score += out["uses_organizer"].astype(float) * 3.0
    information_score += out["uses_target_map"].astype(float) * 4.0
    information_score += out["is_global_information_baseline"].astype(float) * 1.0
    memory_signal = _text(out, "policy_family").str.contains("memory|signal", case=False, regex=True)
    nonconservative = (_numeric(out, "nonconservative_action_count").fillna(0.0) > 0.0) | (_bool(out, "requires_nonconservative_repair"))
    out["information_access_score"] = information_score
    out["intervention_complexity_score"] = information_score + memory_signal.astype(float) + nonconservative.astype(float)
    out["intervention_components"] = [
        ",".join(
            component
            for component, present in [
                ("target_map", bool(target_map)),
                ("global_gradient", bool(global_gradient)),
                ("organizer", bool(organizer)),
                ("memory_or_signal", bool(mem_sig)),
                ("nonconservative_actions", bool(noncons)),
            ]
            if present
        )
        or "local_target_map_free"
        for target_map, global_gradient, organizer, mem_sig, noncons in zip(
            out["uses_target_map"],
            out["uses_global_gradient"],
            out["uses_organizer"],
            memory_signal,
            nonconservative,
            strict=True,
        )
    ]
    return out


def harmonize_control_runs(
    spec: ControlSourceSpec,
    run_df: pd.DataFrame,
    leakage_audit_df: pd.DataFrame | None = None,
    trajectory_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize one upstream run table to the S13 comparison schema."""

    run = run_df.copy()
    if "research_step_id" not in run.columns:
        run["research_step_id"] = spec.source_step_id
    run["source_step_id"] = spec.source_step_id
    if "target_id" not in run.columns and "task_id" in run.columns:
        run["target_id"] = run["task_id"].astype(str)
    if "target_id" not in run.columns:
        run["target_id"] = "unknown"
    if "task_id" not in run.columns:
        run["task_id"] = None
    if "run_id" not in run.columns:
        run["run_id"] = spec.source_step_id + "::" + run["target_id"].astype(str) + "::" + _text(run, "policy_id") + "::seed" + _numeric(run, "seed").fillna(-1).astype(int).astype(str)
    for column in ["motif", "policy_id", "policy_family"]:
        if column not in run.columns:
            run[column] = "unknown"
    run = _attach_policy_audit(run, leakage_audit_df)
    run = _attach_s12_trajectory(run, trajectory_df, spec.source_step_id)
    run = _classify_information_access(run)

    initial_error, final_error, best_error, metric_name = _target_error_columns(run)
    run["initial_target_error"] = initial_error
    run["final_target_error"] = final_error
    run["best_target_error"] = best_error
    run["target_error_metric"] = metric_name
    run["relative_error_reduction_normalized"] = _numeric(run, "relative_error_reduction")
    missing_reduction = run["relative_error_reduction_normalized"].isna()
    denominator = run["initial_target_error"].where(run["initial_target_error"].abs() > 1e-12, np.nan)
    run.loc[missing_reduction, "relative_error_reduction_normalized"] = (
        (run.loc[missing_reduction, "initial_target_error"] - run.loc[missing_reduction, "final_target_error"]) / denominator.loc[missing_reduction]
    ).fillna(0.0)

    run["exact_success"] = (
        _bool(run, "exact_recovery")
        | _bool(run, "success")
        | _bool(run, "exact_match")
        | _bool(run, "exact_or_near_target")
    )
    run["improved"] = _bool(run, "improved") | (run["final_target_error"] < run["initial_target_error"])
    run["occupancy_preserved"] = _bool(run, "occupancy_preserved", default=True)
    run["cell_ids_preserved"] = _bool(run, "cell_ids_preserved", default=True)
    run["robustness_score"] = np.where(run["exact_success"], 1.0, np.where(run["improved"], 0.5, 0.0))
    run.loc[~run["occupancy_preserved"], "robustness_score"] *= 0.5

    run["final_target_energy_proxy"] = _first_numeric(run, ["final_target_energy", "final_target_energy_s12"])
    reported_energy = run["final_target_energy_proxy"].notna()
    run.loc[~reported_energy, "final_target_energy_proxy"] = run.loc[~reported_energy, "final_target_error"] * _numeric(run, "node_count").fillna(1.0)
    run["target_energy_source"] = np.where(reported_energy, "reported_target_energy", "final_error_times_node_count_proxy")
    run["initial_target_energy_proxy"] = _first_numeric(run, ["initial_target_energy"]).fillna(run["initial_target_error"] * _numeric(run, "node_count").fillna(1.0))

    action_cols = [
        "accepted_swap_count",
        "swap_count",
        "crawl_count",
        "divide_count",
        "die_count",
        "rotate_count",
        "memory_update_count",
        "signal_update_count",
    ]
    action_count = pd.Series(0.0, index=run.index)
    for column in action_cols:
        action_count += _numeric(run, column).fillna(0.0)
    run["action_count_proxy"] = action_count
    run["wait_count_proxy"] = _numeric(run, "wait_count").fillna(0.0)
    run["work_cost_proxy"] = action_count + _numeric(run, "nonconservative_action_count").fillna(0.0) * 2.0
    run = _intervention_scores(run)

    leakage_flags = [
        "uses_whole_target_leakage",
        "uses_hidden_target_map_leakage",
        "uses_hidden_axis_leakage",
        "uses_target_map_leakage",
        "uses_hidden_global_size",
    ]
    run["hidden_global_leakage_flag"] = False
    for column in leakage_flags:
        run["hidden_global_leakage_flag"] = run["hidden_global_leakage_flag"] | _bool(run, column)
    run["local_only_leakage_violation"] = run["is_local_only_policy"] & (
        run["uses_target_map"]
        | run["uses_global_gradient"]
        | run["uses_organizer"]
        | run["hidden_global_leakage_flag"]
        | (~run["policy_audit_success"].astype(bool))
    )
    run["global_baseline_explicitly_flagged"] = (~run["is_global_information_baseline"]) | (
        run["uses_target_map"] | run["uses_global_gradient"] | run["uses_organizer"]
    )

    keep = [
        "schema_version",
        "research_step_id",
        "source_step_id",
        "source_kind",
        "run_id",
        "trajectory_id",
        "s12_route_joined",
        "target_id",
        "task_id",
        "motif",
        "policy_id",
        "policy_family",
        "seed",
        "node_count",
        "information_scope",
        "control_class",
        "baseline_flag",
        "is_local_only_policy",
        "is_global_information_baseline",
        "uses_target_map",
        "uses_global_gradient",
        "uses_organizer",
        "hidden_global_leakage_flag",
        "local_only_leakage_violation",
        "global_baseline_explicitly_flagged",
        "policy_audit_success",
        "policy_audit_errors_json",
        "policy_audit_payload_hash",
        "initial_target_error",
        "final_target_error",
        "best_target_error",
        "target_error_metric",
        "relative_error_reduction_normalized",
        "exact_success",
        "improved",
        "robustness_score",
        "final_target_energy_proxy",
        "target_energy_source",
        "initial_target_energy_proxy",
        "action_count_proxy",
        "wait_count_proxy",
        "work_cost_proxy",
        "information_access_score",
        "intervention_complexity_score",
        "intervention_components",
        "path_curvature_proxy",
        "temporary_worsening_count",
        "monotonicity_error_proxy",
        "convergence_basin",
        "route_cluster_id",
        "failure_cluster_id",
    ]
    run["schema_version"] = CONTROL_COMPARISON_SCHEMA_VERSION
    run["research_step_id"] = "S13"
    run["source_kind"] = spec.source_kind
    for column in keep:
        if column not in run.columns:
            run[column] = np.nan
    return run[keep]


def load_control_comparison(
    specs: Sequence[ControlSourceSpec],
    trajectory_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load S07-S11 run results and attach S12 trajectory route metadata."""

    available, catalog = available_control_sources(specs)
    trajectory_df = pd.read_parquet(trajectory_path) if trajectory_path and Path(trajectory_path).exists() else None
    frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    for spec in available:
        run_df = pd.read_parquet(spec.run_path)
        audit_df = pd.read_parquet(spec.leakage_audit_path) if spec.leakage_audit_path and Path(spec.leakage_audit_path).exists() else None
        if audit_df is not None:
            audit = audit_df.copy()
            audit["source_step_id"] = spec.source_step_id
            audit["source_kind"] = spec.source_kind
            audit_frames.append(audit)
        frames.append(harmonize_control_runs(spec, run_df, audit_df, trajectory_df))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    audits = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    return combined, catalog, audits


def information_access_summary(control_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize run counts and outcomes by explicit information-access class."""

    rows: list[dict[str, Any]] = []
    group_cols = ["source_step_id", "source_kind", "control_class", "information_scope", "policy_family"]
    for keys, group in control_df.groupby(group_cols, dropna=False, sort=True):
        source_step_id, source_kind, control_class, information_scope, policy_family = keys
        rows.append(
            {
                "schema_version": CONTROL_COMPARISON_SCHEMA_VERSION,
                "research_step_id": "S13",
                "source_step_id": source_step_id,
                "source_kind": source_kind,
                "control_class": control_class,
                "information_scope": information_scope,
                "policy_family": policy_family,
                "run_count": int(len(group)),
                "local_only_run_count": int(group["is_local_only_policy"].astype(bool).sum()),
                "global_baseline_run_count": int(group["is_global_information_baseline"].astype(bool).sum()),
                "mean_final_target_error": float(group["final_target_error"].astype(float).mean()),
                "median_final_target_error": float(group["final_target_error"].astype(float).median()),
                "mean_relative_error_reduction": float(group["relative_error_reduction_normalized"].astype(float).mean()),
                "exact_success_rate": float(group["exact_success"].astype(bool).mean()),
                "improved_rate": float(group["improved"].astype(bool).mean()),
                "mean_robustness_score": float(group["robustness_score"].astype(float).mean()),
                "mean_final_target_energy_proxy": float(group["final_target_energy_proxy"].astype(float).mean()),
                "mean_action_count_proxy": float(group["action_count_proxy"].astype(float).mean()),
                "mean_work_cost_proxy": float(group["work_cost_proxy"].astype(float).mean()),
                "mean_information_access_score": float(group["information_access_score"].astype(float).mean()),
                "mean_intervention_complexity_score": float(group["intervention_complexity_score"].astype(float).mean()),
                "local_only_leakage_violation_count": int(group["local_only_leakage_violation"].astype(bool).sum()),
                "s12_route_join_rate": float(group["s12_route_joined"].astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def target_control_gap_summary(control_df: pd.DataFrame) -> pd.DataFrame:
    """Compare local-only policies against explicit global baselines where both exist."""

    rows: list[dict[str, Any]] = []
    group_cols = ["source_step_id", "source_kind", "target_id", "motif"]
    for keys, group in control_df.groupby(group_cols, dropna=False, sort=True):
        source_step_id, source_kind, target_id, motif = keys
        local = group[group["is_local_only_policy"].astype(bool)]
        global_info = group[group["is_global_information_baseline"].astype(bool)]
        if local.empty and global_info.empty:
            continue
        local_error = float(local["final_target_error"].astype(float).mean()) if len(local) else math.nan
        global_error = float(global_info["final_target_error"].astype(float).mean()) if len(global_info) else math.nan
        local_success = float(local["exact_success"].astype(bool).mean()) if len(local) else math.nan
        global_success = float(global_info["exact_success"].astype(bool).mean()) if len(global_info) else math.nan
        rows.append(
            {
                "schema_version": CONTROL_COMPARISON_SCHEMA_VERSION,
                "research_step_id": "S13",
                "source_step_id": source_step_id,
                "source_kind": source_kind,
                "target_id": target_id,
                "motif": motif,
                "local_run_count": int(len(local)),
                "global_baseline_run_count": int(len(global_info)),
                "has_local_and_global": bool(len(local) > 0 and len(global_info) > 0),
                "local_mean_final_target_error": local_error,
                "global_mean_final_target_error": global_error,
                "local_minus_global_final_error": local_error - global_error if math.isfinite(local_error) and math.isfinite(global_error) else math.nan,
                "local_exact_success_rate": local_success,
                "global_exact_success_rate": global_success,
                "global_minus_local_success_rate": global_success - local_success if math.isfinite(local_success) and math.isfinite(global_success) else math.nan,
                "local_mean_robustness_score": float(local["robustness_score"].astype(float).mean()) if len(local) else math.nan,
                "global_mean_robustness_score": float(global_info["robustness_score"].astype(float).mean()) if len(global_info) else math.nan,
                "local_mean_intervention_complexity": float(local["intervention_complexity_score"].astype(float).mean()) if len(local) else math.nan,
                "global_mean_intervention_complexity": float(global_info["intervention_complexity_score"].astype(float).mean()) if len(global_info) else math.nan,
                "global_control_classes": ",".join(sorted(global_info["control_class"].astype(str).unique())) if len(global_info) else "",
            }
        )
    return pd.DataFrame(rows)


def leakage_audit_summary(control_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    """Produce a policy-level leakage and baseline-flag audit table."""

    rows: list[dict[str, Any]] = []
    for keys, group in control_df.groupby(["source_step_id", "policy_id", "policy_family"], dropna=False, sort=True):
        source_step_id, policy_id, policy_family = keys
        rows.append(
            {
                "schema_version": CONTROL_COMPARISON_SCHEMA_VERSION,
                "research_step_id": "S13",
                "source_step_id": source_step_id,
                "policy_id": policy_id,
                "policy_family": policy_family,
                "run_count": int(len(group)),
                "control_classes": ",".join(sorted(group["control_class"].astype(str).unique())),
                "information_scopes": ",".join(sorted(group["information_scope"].astype(str).unique())),
                "is_local_only_policy": bool(group["is_local_only_policy"].astype(bool).all()),
                "is_global_information_baseline": bool(group["is_global_information_baseline"].astype(bool).any()),
                "uses_target_map": bool(group["uses_target_map"].astype(bool).any()),
                "uses_global_gradient": bool(group["uses_global_gradient"].astype(bool).any()),
                "uses_organizer": bool(group["uses_organizer"].astype(bool).any()),
                "policy_audit_success": bool(group["policy_audit_success"].astype(bool).all()),
                "local_only_leakage_violation_count": int(group["local_only_leakage_violation"].astype(bool).sum()),
                "global_baseline_explicitly_flagged": bool(group["global_baseline_explicitly_flagged"].astype(bool).all()),
                "audit_error_examples": ";".join(sorted(set(group["policy_audit_errors_json"].astype(str))))[:500],
            }
        )
    if audit_df is not None and not audit_df.empty:
        audited = set(zip(audit_df["source_step_id"].astype(str), audit_df["policy_id"].astype(str)))
        for row in rows:
            row["policy_audit_record_present"] = (str(row["source_step_id"]), str(row["policy_id"])) in audited
    else:
        for row in rows:
            row["policy_audit_record_present"] = False
    return pd.DataFrame(rows)


def validation_rows(
    *,
    source_catalog: pd.DataFrame,
    control_df: pd.DataFrame,
    access_summary_df: pd.DataFrame,
    gap_summary_df: pd.DataFrame,
    leakage_summary_df: pd.DataFrame,
    expected_sources: Iterable[str] = ("S07", "S08", "S09", "S10", "S11"),
) -> pd.DataFrame:
    """Build S13 validation checks."""

    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: Any) -> None:
        rows.append(
            {
                "schema_version": CONTROL_COMPARISON_SCHEMA_VERSION,
                "research_step_id": "S13",
                "check_id": check_id,
                "success": bool(success),
                "detail": detail,
            }
        )

    expected = set(expected_sources)
    observed = set(control_df["source_step_id"].astype(str).unique()) if len(control_df) else set()
    add("source_run_artifacts_available", expected.issubset(set(source_catalog[source_catalog["run_available"]]["source_step_id"])), {"expected": sorted(expected), "observed": sorted(observed)})
    add("expected_sources_present", observed == expected, {"expected": sorted(expected), "observed": sorted(observed)})
    add("run_and_trajectory_identifiers_preserved", bool(control_df["run_id"].notna().all() and control_df["trajectory_id"].notna().all()), "run_id and trajectory_id are populated")
    add("s12_route_metadata_joined", bool(control_df["s12_route_joined"].astype(bool).all()), {"joinRate": float(control_df["s12_route_joined"].astype(bool).mean()) if len(control_df) else 0.0})
    duplicate_count = int(control_df.duplicated(["source_step_id", "run_id"]).sum())
    add("run_rows_unique_by_source_and_run_id", duplicate_count == 0, {"duplicateCount": duplicate_count})
    local = control_df[control_df["is_local_only_policy"].astype(bool)]
    global_info = control_df[control_df["is_global_information_baseline"].astype(bool)]
    add("local_and_global_conditions_separable", bool(len(local) > 0 and len(global_info) > 0 and not (local["is_global_information_baseline"].astype(bool).any()) and not (global_info["is_local_only_policy"].astype(bool).any())), {"localRuns": int(len(local)), "globalRuns": int(len(global_info))})
    add("local_only_leakage_audit_passed", bool(local["local_only_leakage_violation"].astype(bool).sum() == 0), {"violations": int(local["local_only_leakage_violation"].astype(bool).sum()), "localRuns": int(len(local))})
    add("global_baselines_explicitly_flagged", bool(len(global_info) > 0 and global_info["global_baseline_explicitly_flagged"].astype(bool).all()), {"globalRuns": int(len(global_info)), "controlClasses": sorted(global_info["control_class"].astype(str).unique())})
    add("target_error_metrics_finite", bool(np.isfinite(control_df["final_target_error"].astype(float).to_numpy()).all()), "final target-error proxies are finite")
    add("energy_and_complexity_metrics_finite", bool(np.isfinite(control_df[["final_target_energy_proxy", "information_access_score", "intervention_complexity_score"]].astype(float).to_numpy()).all()), "energy and intervention-complexity proxies are finite")
    add("access_summary_written", len(access_summary_df) > 0, {"rows": int(len(access_summary_df))})
    add("target_gap_summary_written", len(gap_summary_df) > 0 and gap_summary_df["has_local_and_global"].astype(bool).any(), {"rows": int(len(gap_summary_df)), "pairedRows": int(gap_summary_df["has_local_and_global"].astype(bool).sum()) if len(gap_summary_df) else 0})
    add("policy_leakage_summary_written", len(leakage_summary_df) > 0 and leakage_summary_df["policy_audit_success"].astype(bool).all(), {"rows": int(len(leakage_summary_df))})
    return pd.DataFrame(rows)
