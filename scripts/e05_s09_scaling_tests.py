#!/usr/bin/env python3
"""Run E05 S09 scaling tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.e05.scaling import (
    DEFAULT_EVENT_MULTIPLIER,
    DEFAULT_RECORDS_PER_RUN,
    DEFAULT_SCALING_SEEDS,
    SCALING_TASK_TYPES,
    blocked_s08_perturbation_rows,
    default_scaling_target_configs,
    run_scaling_benchmark,
    scaling_summary_rows,
    target_config_rows,
    validate_scaling_target_configs,
)
from src.e05.scrambled_embryo import RUNNABLE_POLICY_IDS


STEP_ID = "S09"
STEP_NUMBER = 9
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SCALING_SEEDS))
    parser.add_argument("--policy-ids", default=",".join(RUNNABLE_POLICY_IDS))
    parser.add_argument("--task-types", default=",".join(SCALING_TASK_TYPES))
    parser.add_argument("--event-multiplier", type=int, default=DEFAULT_EVENT_MULTIPLIER)
    parser.add_argument("--records-per-run", type=int, default=DEFAULT_RECORDS_PER_RUN)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_path(path),
        "sizeBytes": path.stat().st_size if path.is_file() else sum(child.stat().st_size for child in path.rglob("*") if child.is_file()),
        "artifactType": "directory" if path.is_dir() else "file",
    }


def manifest_self_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum omitted to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("at least one seed is required")
    return values


def parse_str_list(raw: str, allowed: Sequence[str], label: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"at least one {label} is required")
    unsupported = sorted(set(values) - set(allowed))
    if unsupported:
        raise ValueError(f"unsupported {label}: {unsupported}")
    return values


def _row(
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


def load_source_results(artifacts_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    s07_path = artifacts_dir / "results" / "e05_scrambled_embryo_results.parquet"
    s08_path = artifacts_dir / "results" / "e05_regeneration_results.parquet"
    if not s07_path.exists() or not s08_path.exists():
        missing = [str(path) for path in (s07_path, s08_path) if not path.exists()]
        raise FileNotFoundError(f"S09 requires S07/S08 source result tables; missing: {missing}")
    return pd.read_parquet(s07_path), pd.read_parquet(s08_path)


def source_anchor_rows(s07_df: pd.DataFrame, s08_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_id, group in s07_df.groupby("policy_id"):
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": "S07",
                "source_result_family": "scrambled_embryo",
                "policy_id": policy_id,
                "perturbation_type": "full_scramble",
                "runs": int(len(group)),
                "mean_target_recovery_fraction": float(group["target_recovery_fraction"].mean()),
                "mean_final_target_error": float(group["final_target_error"].mean()),
                "mean_total_energy_cost": float(group["total_energy_cost"].mean()),
                "source_note": "S07 small-grid scrambled benchmark anchor used to freeze runnable policy IDs and access labels.",
            }
        )
    for (perturbation_type, policy_id), group in s08_df.groupby(["perturbation_type", "policy_id"]):
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": "S08",
                "source_result_family": "regeneration",
                "policy_id": policy_id,
                "perturbation_type": perturbation_type,
                "runs": int(len(group)),
                "mean_target_recovery_fraction": float(group["target_recovery_fraction"].mean()),
                "mean_final_target_error": float(group["post_repair_target_error"].mean()),
                "mean_total_energy_cost": float(group["total_energy_cost"].mean()),
                "source_note": (
                    "S08 rotated-patch rows supply the repairable regeneration task; blocked perturbations are carried as blocker context only."
                    if perturbation_type == "rotated_patch"
                    else "S08 blocker context; not scaled because current S07 controls lack the needed repair semantics."
                ),
            }
        )
    return rows


def validation_source_inputs_loaded(s07_df: pd.DataFrame, s08_df: pd.DataFrame) -> dict[str, Any]:
    required_s07 = {"policy_id", "target_recovery_fraction", "final_target_error", "total_energy_cost"}
    required_s08 = {
        "policy_id",
        "perturbation_type",
        "target_recovery_fraction",
        "post_repair_target_error",
        "semantic_blocker",
        "total_energy_cost",
    }
    observed = {
        "s07_rows": int(len(s07_df)),
        "s08_rows": int(len(s08_df)),
        "s07_required_columns_present": bool(required_s07.issubset(s07_df.columns)),
        "s08_required_columns_present": bool(required_s08.issubset(s08_df.columns)),
        "s08_rotated_patch_rows": int((s08_df["perturbation_type"] == "rotated_patch").sum()),
    }
    expected = {
        "s07_rows_gt": 0,
        "s08_rows_gt": 0,
        "s07_required_columns_present": True,
        "s08_required_columns_present": True,
        "s08_rotated_patch_rows_gt": 0,
    }
    success = (
        observed["s07_rows"] > 0
        and observed["s08_rows"] > 0
        and observed["s07_required_columns_present"]
        and observed["s08_required_columns_present"]
        and observed["s08_rotated_patch_rows"] > 0
    )
    return _row(
        "s07_s08_source_results_loaded",
        "input",
        success,
        expected,
        observed,
        "Required S07 and S08 source result tables were loaded before running S09.",
    )


def validation_scaled_targets(config_df: pd.DataFrame, pair_validation_df: pd.DataFrame) -> dict[str, Any]:
    observed = {
        "target_config_rows": int(len(config_df)),
        "source_target_count": int(config_df["source_target_id"].nunique()),
        "scale_labels": sorted(config_df["scale_label"].unique().tolist()),
        "all_constructed_errors_zero": bool((config_df["constructed_target_error"].astype(float) == 0.0).all()),
        "all_pair_validation_cases_passed": bool(pair_validation_df["success"].all()),
        "min_large_to_small_site_ratio": float(config_df.loc[config_df["scale_label"] == "large", "large_to_small_site_ratio"].min()),
    }
    expected = {
        "target_config_rows": 12,
        "scale_labels": ["large", "small"],
        "all_constructed_errors_zero": True,
        "all_pair_validation_cases_passed": True,
        "min_large_to_small_site_ratio_gt": 1.0,
    }
    success = (
        observed["target_config_rows"] == expected["target_config_rows"]
        and observed["scale_labels"] == expected["scale_labels"]
        and observed["all_constructed_errors_zero"]
        and observed["all_pair_validation_cases_passed"]
        and observed["min_large_to_small_site_ratio"] > 1.0
    )
    return _row(
        "scaled_targets_generated_consistently",
        "target_config",
        success,
        expected,
        observed,
        "Small/large paired S03-compatible target configs were generated and their constructed states have zero target error.",
    )


def validation_run_matrix(
    summary_df: pd.DataFrame,
    *,
    target_config_count: int,
    task_count: int,
    policy_count: int,
    seed_count: int,
) -> dict[str, Any]:
    expected_rows = target_config_count * task_count * policy_count * seed_count
    observed = {
        "run_count": int(len(summary_df)),
        "expected_run_count": int(expected_rows),
        "target_config_count": int(summary_df["config_id"].nunique()),
        "task_count": int(summary_df["task_type"].nunique()),
        "policy_count": int(summary_df["policy_id"].nunique()),
        "seed_count": int(summary_df["simulation_seed"].nunique()),
    }
    return _row(
        "scaling_run_matrix_complete",
        "run_matrix",
        observed["run_count"] == expected_rows,
        {"run_count": expected_rows},
        observed,
        "S09 produced the full target-config by task by policy by seed run matrix.",
    )


def validation_no_retuning(summary_df: pd.DataFrame) -> dict[str, Any]:
    grouped = summary_df.groupby(["task_type", "source_target_id", "policy_id"])
    signature_consistent = True
    event_multiplier_consistent = True
    scale_pairs_present = True
    for _key, group in grouped:
        signature_consistent = signature_consistent and group["policy_parameter_signature"].nunique() == 1
        event_multiplier_consistent = event_multiplier_consistent and group["event_multiplier"].nunique() == 1
        scale_pairs_present = scale_pairs_present and set(group["scale_label"].unique()) == {"small", "large"}
    observed = {
        "retuning_allowed_any": bool(summary_df["retuning_allowed"].any()),
        "large_grid_retuned_any": bool(summary_df["large_grid_retuned"].any()),
        "policy_signature_consistent_across_scales": bool(signature_consistent),
        "event_multiplier_consistent_across_scales": bool(event_multiplier_consistent),
        "small_large_pairs_present_for_each_task_target_policy": bool(scale_pairs_present),
    }
    expected = {
        "retuning_allowed_any": False,
        "large_grid_retuned_any": False,
        "policy_signature_consistent_across_scales": True,
        "event_multiplier_consistent_across_scales": True,
        "small_large_pairs_present_for_each_task_target_policy": True,
    }
    success = all(observed[key] == expected[key] for key in expected)
    return _row(
        "no_retuning_on_large_grid_tests",
        "policy_transfer",
        success,
        expected,
        observed,
        "Large-grid runs reused the same S07 policy IDs, declared access labels, and per-site event budget as small-grid runs.",
    )


def validation_metrics_quantified(summary_df: pd.DataFrame, metric_df: pd.DataFrame, scaling_df: pd.DataFrame) -> dict[str, Any]:
    expected_metric_rows = len(summary_df) * 2 * 8
    required_scaling = {
        "mean_target_recovery_fraction",
        "mean_final_target_error",
        "mean_energy_per_site",
        "low_recovery_failure_rate",
        "large_to_small_recovery_ratio",
    }
    observed = {
        "metric_rows": int(len(metric_df)),
        "expected_metric_rows": int(expected_metric_rows),
        "metric_values_non_null": bool(metric_df["value"].notna().all()),
        "scaling_summary_rows": int(len(scaling_df)),
        "required_scaling_columns_present": bool(required_scaling.issubset(scaling_df.columns)),
        "run_level_recovery_finite": bool(np.isfinite(summary_df["target_recovery_fraction"].astype(float)).all()),
        "run_level_energy_finite": bool(np.isfinite(summary_df["energy_per_site"].astype(float)).all()),
    }
    expected = {
        "metric_rows": expected_metric_rows,
        "metric_values_non_null": True,
        "required_scaling_columns_present": True,
        "run_level_recovery_finite": True,
        "run_level_energy_finite": True,
    }
    success = (
        observed["metric_rows"] == expected_metric_rows
        and observed["metric_values_non_null"]
        and observed["required_scaling_columns_present"]
        and observed["run_level_recovery_finite"]
        and observed["run_level_energy_finite"]
        and observed["scaling_summary_rows"] > 0
    )
    return _row(
        "recovery_error_energy_failure_scaling_quantified",
        "metric",
        success,
        expected,
        observed,
        "Run-level and grouped scaling metrics cover recovery, final error, energy per site, and failure rates.",
    )


def validation_blocked_s08_not_overextended(summary_df: pd.DataFrame, blocker_df: pd.DataFrame) -> dict[str, Any]:
    observed_task_perturbations = set(summary_df["perturbation_type"].unique().tolist())
    blocked_types = set(blocker_df.loc[~blocker_df["included_in_scaling"].astype(bool), "perturbation_type"].tolist())
    observed = {
        "observed_perturbation_types": sorted(observed_task_perturbations),
        "blocked_types": sorted(blocked_types),
        "blocked_types_absent_from_scaling_runs": bool(observed_task_perturbations.isdisjoint(blocked_types)),
        "blocked_rows_with_reason": bool(blocker_df.loc[~blocker_df["included_in_scaling"].astype(bool), "reason"].astype(str).str.len().gt(0).all()),
        "rotated_patch_included": bool((summary_df["perturbation_type"] == "rotated_patch").any()),
    }
    expected = {
        "blocked_types_absent_from_scaling_runs": True,
        "blocked_rows_with_reason": True,
        "rotated_patch_included": True,
    }
    success = all(observed[key] == expected[key] for key in expected)
    return _row(
        "s08_blocked_perturbations_excluded_with_reason",
        "blocker",
        success,
        expected,
        observed,
        "S09 scales only the S07 scramble and S08 swap-repairable rotated patch, while preserving S08 blocker labels for other perturbations.",
    )


def validation_local_control_reported(summary_df: pd.DataFrame) -> dict[str, Any]:
    large = summary_df[summary_df["scale_label"] == "large"]
    by_policy = large.groupby("policy_id", as_index=False).agg(mean_recovery=("target_recovery_fraction", "mean"))
    local_mean = float(
        by_policy.loc[by_policy["policy_id"] == "s07_local_target_neighbor_descent", "mean_recovery"].iloc[0]
    )
    random_mean = float(by_policy.loc[by_policy["policy_id"] == "s07_random_adjacent_swap_control", "mean_recovery"].iloc[0])
    observed = {
        "large_local_mean_recovery": local_mean,
        "large_random_mean_recovery": random_mean,
        "large_recovery_values_finite": bool(np.isfinite(large["target_recovery_fraction"].astype(float)).all()),
        "large_local_exceeds_random": bool(local_mean > random_mean),
    }
    expected = {"large_recovery_values_finite": True, "large_local_exceeds_random": True}
    success = observed["large_recovery_values_finite"] and observed["large_local_exceeds_random"]
    return _row(
        "large_grid_local_control_recovery_reported",
        "outcome",
        success,
        expected,
        observed,
        "The frozen local target-aware control remains measurable on large grids and is compared to the random adjacent-swap null.",
    )


def validation_figure(figure_path: Path) -> dict[str, Any]:
    size = figure_path.stat().st_size if figure_path.exists() else 0
    observed = {"exists": figure_path.exists(), "size_bytes": int(size)}
    expected = {"exists": True, "size_bytes_min": 1000}
    return _row(
        "scaling_curves_figure_written",
        "artifact",
        observed["exists"] and observed["size_bytes"] > expected["size_bytes_min"],
        expected,
        observed,
        "The required S09 scaling-curve figure was written and is nonempty.",
    )


def run_validations(
    *,
    s07_df: pd.DataFrame,
    s08_df: pd.DataFrame,
    config_df: pd.DataFrame,
    pair_validation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    blocker_df: pd.DataFrame,
    figure_path: Path,
    task_count: int,
    policy_count: int,
    seed_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            validation_source_inputs_loaded(s07_df, s08_df),
            validation_scaled_targets(config_df, pair_validation_df),
            validation_run_matrix(
                summary_df,
                target_config_count=len(config_df),
                task_count=task_count,
                policy_count=policy_count,
                seed_count=seed_count,
            ),
            validation_no_retuning(summary_df),
            validation_metrics_quantified(summary_df, metric_df, scaling_df),
            validation_blocked_s08_not_overextended(summary_df, blocker_df),
            validation_local_control_reported(summary_df),
            validation_figure(figure_path),
        ]
    )


def render_scaling_curves(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_run = (
        summary_df.groupby(["task_type", "target_kind", "policy_id", "scale_label", "site_count"], as_index=False)
        .agg(
            mean_recovery=("target_recovery_fraction", "mean"),
            mean_final_error=("final_target_error", "mean"),
            mean_energy_per_site=("energy_per_site", "mean"),
            failure_rate=("low_recovery_failure", "mean"),
        )
        .sort_values(["task_type", "target_kind", "policy_id", "site_count"])
    )
    policies = sorted(by_run["policy_id"].unique())
    policy_colors = {
        "s07_local_target_neighbor_descent": "#2F6B4F",
        "s07_random_adjacent_swap_control": "#8D4E2F",
    }
    panels = [
        ("mean_recovery", "Target recovery fraction"),
        ("mean_final_error", "Final target error"),
        ("mean_energy_per_site", "Energy per site"),
        ("failure_rate", "Low-recovery failure rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    for ax, (column, title) in zip(axes.reshape(-1), panels, strict=True):
        for task_type in sorted(by_run["task_type"].unique()):
            task_df = by_run[by_run["task_type"] == task_type]
            linestyle = "-" if task_type == "scrambled_embryo" else "--"
            for policy_id in policies:
                policy_df = task_df[task_df["policy_id"] == policy_id]
                for _target_kind, target_df in policy_df.groupby("target_kind"):
                    ax.plot(
                        target_df["site_count"],
                        target_df[column],
                        marker="o",
                        linewidth=1.1,
                        alpha=0.45,
                        color=policy_colors.get(policy_id, "#444444"),
                        linestyle=linestyle,
                    )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("target sites")
        ax.grid(True, linewidth=0.4, alpha=0.35)
    legend_lines = [
        plt.Line2D([0], [0], color=policy_colors["s07_local_target_neighbor_descent"], marker="o", label="local target-aware"),
        plt.Line2D([0], [0], color=policy_colors["s07_random_adjacent_swap_control"], marker="o", label="random adjacent swap"),
        plt.Line2D([0], [0], color="#555555", linestyle="-", label="scrambled"),
        plt.Line2D([0], [0], color="#555555", linestyle="--", label="rotated patch"),
    ]
    fig.legend(handles=legend_lines, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("E05 S09 scaling tests: small-to-large target transfer", fontsize=13)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def top_summary_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
    lay_summary: str,
) -> str:
    artifact_lines = "\n".join(f"- `{entry['path']}`" for entry in artifacts if entry.get("path"))
    return f"""## Top Summary

- Research step ID: {STEP_ID}
- Completion status: Completed
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {outcome_classification}
- Caveats or blockers: {caveats_or_blockers}
- Lay summary: {lay_summary}
- Recommended next action: {recommended_next_action}
"""


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    config_df: pd.DataFrame,
    source_anchor_df: pd.DataFrame,
    blocker_df: pd.DataFrame,
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    args: argparse.Namespace,
    seeds: Sequence[int],
    policy_ids: Sequence[str],
    task_types: Sequence[str],
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
    lay_summary: str,
) -> str:
    command_lines = "\n".join(
        f"- `{command['command']}`: return code {command['returnCode']}, success={command['success']}, elapsed={command['elapsedSeconds']:.3f}s"
        for command in test_commands
    )
    source_lines = "\n".join(
        f"- `{entry['relativePath']}` sha256 `{entry['sha256']}` ({entry['sizeBytes']} bytes)"
        for entry in source_files
    )
    transfer_summary = (
        summary_df.groupby(["task_type", "scale_label", "policy_id"], as_index=False)
        .agg(
            runs=("target_id", "count"),
            mean_target_recovery=("target_recovery_fraction", "mean"),
            mean_final_target_error=("final_target_error", "mean"),
            mean_energy_per_site=("energy_per_site", "mean"),
            low_recovery_failure_rate=("low_recovery_failure", "mean"),
        )
        .sort_values(["task_type", "scale_label", "policy_id"])
    )
    target_summary = config_df[
        [
            "source_target_id",
            "target_kind",
            "scale_label",
            "dimension_signature",
            "site_count",
            "large_to_small_site_ratio",
            "constructed_target_error",
            "proportion_rule",
        ]
    ].copy()
    scaling_focus = scaling_df[
        [
            "task_type",
            "target_kind",
            "policy_id",
            "scale_label",
            "runs",
            "site_count",
            "mean_target_recovery_fraction",
            "mean_energy_per_site",
            "low_recovery_failure_rate",
            "large_to_small_recovery_ratio",
            "large_to_small_energy_per_site_ratio",
        ]
    ].copy()
    for frame in (transfer_summary, scaling_focus):
        for column in frame.select_dtypes(include=[float]).columns:
            frame[column] = frame[column].round(6)
    return f"""{top_summary_markdown(artifacts, validation_result, outcome_classification, caveats_or_blockers, recommended_next_action, lay_summary)}

# Research Step Full Results: {STEP_ID} Run Scaling Tests

## Lay Summary

{lay_summary}

## Frozen Question

Do policies tuned on small grids transfer to larger grids while preserving target morphology and relative proportions?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S09.
- S07 source results: `$ARTIFACTS_DIR/results/e05_scrambled_embryo_results.parquet`.
- S08 source results: `$ARTIFACTS_DIR/results/e05_regeneration_results.parquet`.
- S07 controls: `s07_local_target_neighbor_descent` and `s07_random_adjacent_swap_control`, used without changing declared information access.
- S03 target constructors: gradient, stripes, boundary, ring, symmetry, and toy organ-like 2D targets.
- S05 metrics: all default morphospace metrics at initial and final states.
- Datasets: none required.

## Detailed Methods

S09 froze the runnable S07 controls and treated the previous S07/S08 small-grid outcomes as policy-selection context, not as a new tuning loop. The large-grid tests use the same policy IDs, same local observation labels, the same S04 swap/wait legality and energy semantics, and the same `{args.event_multiplier}` scheduler events per site as the small-grid runs. The inner loop uses an audit-free S09 fast path for swap/wait proposals because the full S04 `ActionExecutor` hashes the whole state around every action; unit tests compare the fast path against the S04 executor on representative swap and wait actions.

For every genuine 2D S03 target family, S09 generated one small target matching the S03 default dimensions and one larger target with approximately doubled dimensions. The bilateral-symmetry target keeps odd width so the central axis remains defined. Each target state was constructed and checked for zero target error before perturbation.

Two task families were run. The `scrambled_embryo` task repeats the S07 full-permutation challenge. The `rotated_patch_regeneration` task scales the S08 swap-repairable rotated-patch perturbation so patch area grows on large targets. S08 perturbations that require birth, death, identity conversion, or unfreezing were not silently overextended; they are listed as blocker context.

Each run records initial/final target error, aggregate S05 morphospace error, recovery fractions, S04 action accounting, energy per site, low-recovery failure flags, and small-to-large scaling ratios.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- Matplotlib: `{matplotlib.__version__}`
- New packages installed: none.
- CPU/GPU use: serial CPU benchmark and validation; no GPU use was needed for this compact scaling matrix.

## Parameters

- Seeds: `{", ".join(str(seed) for seed in seeds)}`
- Runnable policies: `{", ".join(policy_ids)}`
- Task types: `{", ".join(task_types)}`
- Event multiplier: `{args.event_multiplier}` local scheduler events per target site
- Records per run: `{args.records_per_run}`; retained as run metadata for compatibility with S07/S08, but S09 stores run summaries rather than dense traces.
- Target configs: 12 small/large configs across six S03 target families.

## Results

Validation summary:

{markdown_table(validation_df, ["validation_case", "case_type", "success", "detail"])}

Target configuration summary:

{markdown_table(target_summary, ["source_target_id", "target_kind", "scale_label", "dimension_signature", "site_count", "large_to_small_site_ratio", "constructed_target_error", "proportion_rule"])}

Transfer summary by task, scale, and policy:

{markdown_table(transfer_summary, ["task_type", "scale_label", "policy_id", "runs", "mean_target_recovery", "mean_final_target_error", "mean_energy_per_site", "low_recovery_failure_rate"])}

Small-to-large scaling summary:

{markdown_table(scaling_focus, ["task_type", "target_kind", "policy_id", "scale_label", "runs", "site_count", "mean_target_recovery_fraction", "mean_energy_per_site", "low_recovery_failure_rate", "large_to_small_recovery_ratio", "large_to_small_energy_per_site_ratio"])}

S07/S08 source anchors:

{markdown_table(source_anchor_df, ["source_research_step_id", "source_result_family", "perturbation_type", "policy_id", "runs", "mean_target_recovery_fraction", "mean_final_target_error", "source_note"])}

S08 blocker handling:

{markdown_table(blocker_df, ["perturbation_type", "included_in_scaling", "reason"])}

## Metrics

- Recovery: `(initial target error - final target error) / initial target error`, bounded by the same helper used in S07/S08.
- Error: S03 target identity error and S05 aggregate morphospace error at initial and final states.
- Energy: S04 charged action energy, summarized as total energy and energy per site.
- Failure: `failure_to_improve` means final target error did not decrease; `low_recovery_failure` means target recovery stayed below the fixed `{0.05}` threshold.
- Scaling ratios: large-grid recovery and energy-per-site values divided by matched small-grid values for each task, target family, and policy.

## Figures And Tables

- The required scaling figure is `$ARTIFACTS_DIR/figures/e05/scaling_curves.png`.
- The required run table is `$ARTIFACTS_DIR/results/e05_scaling_tests.parquet` with a CSV companion.
- Additional machine-readable tables record target configs, per-metric rows, source anchors, excluded S08 blockers, and validation cases.

## Validation Checks

- Loaded S07 and S08 source result tables before S09 execution.
- Generated consistent small/large target configs and checked constructed-state zero target error.
- Produced the full target-config by task by policy by seed matrix.
- Verified that large-grid runs were not retuned: no policy parameter, access label, or per-site event-budget changes were introduced.
- Computed recovery, final error, energy, and failure scaling summaries.
- Excluded S08 blocked perturbations with explicit reasons instead of adding undeclared repair semantics.
- Confirmed large-grid local-control recovery is finite and exceeds the random local-swap null on average.
- Wrote the required nonempty scaling-curve figure.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- S09 is a computational proxy benchmark, not a biological validation.
- The large targets are S03-compatible scaled constructions. Some discrete target rules preserve their generator rather than exact area fraction; for example, one-cell boundaries remain one-cell boundaries, and stripe width is a cell-level alternation rule.
- No new policy tuning is performed in S09. That preserves the no-retuning test but does not search for better large-grid policies.
- S09 scales only S07 scrambling and the S08 rotated-patch repairable rearrangement. Missing, frozen, duplicated, and foreign patches remain blocked under current semantics.
- Some small-to-large ratios are infinite when the matched small-grid denominator is zero; the underlying run-level recovery, error, energy, and failure values remain finite in the required result table.
- S05 topology, Hausdorff, graph-edit, and Earth-mover metrics remain labeled approximations.

## Blockers And Failed Assumptions

No execution blocker was encountered. The main failed assumption is that all S08 perturbations can be used in scaling tests under swap/wait controls: S08 already showed that missing, frozen, duplicated, and foreign patches need extra repair semantics, so S09 preserves that constraint and scales only the repairable rotated-patch case.

## Recommended Next Action

{recommended_next_action}
"""


def determine_outcome(summary_df: pd.DataFrame, validation_success: bool) -> tuple[str, str, str]:
    large = summary_df[summary_df["scale_label"] == "large"]
    by_policy = large.groupby("policy_id", as_index=False).agg(mean_recovery=("target_recovery_fraction", "mean"))
    local_mean = float(by_policy.loc[by_policy["policy_id"] == "s07_local_target_neighbor_descent", "mean_recovery"].iloc[0])
    random_mean = float(by_policy.loc[by_policy["policy_id"] == "s07_random_adjacent_swap_control", "mean_recovery"].iloc[0])
    if validation_success and local_mean > random_mean and local_mean > 0.05:
        outcome = "supportive"
        lay = (
            "S09 shows that the frozen local target-aware control still reduces morphology error on larger grids better than the random "
            "local-swap null, while energy and failure rates are now quantified across scale."
        )
    else:
        outcome = "constraining/contradictory"
        lay = (
            "S09 quantified scaling, but the large-grid transfer did not clearly preserve recovery over the random local-swap null, "
            "so larger targets would need new mechanisms or policy search before being treated as robust."
        )
    caveats = (
        "Large-grid tests used fixed S07 controls with no retuning and only scaled the S07 scramble plus the S08 rotated-patch repairable "
        "perturbation. S08 missing, frozen, duplicated, and foreign perturbations remain blocked because they require birth/death, "
        "identity conversion, or unfreeze semantics."
    )
    return outcome, caveats, lay


def write_checksums(paths: list[Path], checksum_path: Path, artifacts_dir: Path) -> None:
    lines = []
    for path in sorted(paths):
        if path == checksum_path:
            continue
        lines.append(f"{sha256_path(path)}  {path.relative_to(artifacts_dir)}")
    write_text(checksum_path, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e05"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, figures_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    seeds = parse_int_list(args.seeds)
    policy_ids = parse_str_list(args.policy_ids, RUNNABLE_POLICY_IDS, "policy IDs")
    task_types = parse_str_list(args.task_types, SCALING_TASK_TYPES, "task types")
    s07_df, s08_df = load_source_results(artifacts_dir)

    configs = default_scaling_target_configs()
    config_rows = target_config_rows(configs)
    pair_validation_rows = validate_scaling_target_configs(configs)
    summary_rows, metric_rows, _run_results = run_scaling_benchmark(
        configs=configs,
        task_types=task_types,
        policy_ids=policy_ids,
        seeds=seeds,
        event_multiplier=args.event_multiplier,
        records_per_run=args.records_per_run,
    )
    scaling_rows = scaling_summary_rows(summary_rows)
    blocker_rows = list(blocked_s08_perturbation_rows())
    source_rows = source_anchor_rows(s07_df, s08_df)

    summary_df = pd.DataFrame(summary_rows)
    metric_df = pd.DataFrame(metric_rows)
    scaling_df = pd.DataFrame(scaling_rows)
    config_df = pd.DataFrame(config_rows)
    pair_validation_df = pd.DataFrame(pair_validation_rows)
    blocker_df = pd.DataFrame(blocker_rows)
    source_anchor_df = pd.DataFrame(source_rows)

    results_path = results_dir / "e05_scaling_tests.parquet"
    results_csv_path = tables_dir / "e05_scaling_tests.csv"
    metric_rows_path = results_dir / "e05_scaling_metric_rows.parquet"
    metric_rows_csv_path = tables_dir / "e05_scaling_metric_rows.csv"
    scaling_summary_path = results_dir / "e05_scaling_summary.parquet"
    scaling_summary_csv_path = tables_dir / "e05_scaling_summary.csv"
    target_config_path = results_dir / "e05_scaling_target_configs.parquet"
    target_config_csv_path = tables_dir / "e05_scaling_target_configs.csv"
    source_anchor_path = results_dir / "e05_scaling_source_anchors.parquet"
    source_anchor_csv_path = tables_dir / "e05_scaling_source_anchors.csv"
    blocker_path = results_dir / "e05_scaling_s08_blockers.parquet"
    blocker_csv_path = tables_dir / "e05_scaling_s08_blockers.csv"
    pair_validation_path = results_dir / "e05_scaling_target_pair_validation.parquet"
    validation_path = results_dir / "e05_scaling_validation.parquet"
    validation_csv_path = tables_dir / "e05_scaling_validation.csv"
    figure_path = figures_dir / "scaling_curves.png"
    config_path = configs_dir / "e05_s09_scaling_tests.json"
    source_manifest_path = src_snapshot_dir / "e05_scaling_manifest.json"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    summary_df.to_parquet(results_path, index=False)
    summary_df.to_csv(results_csv_path, index=False)
    metric_df.to_parquet(metric_rows_path, index=False)
    metric_df.to_csv(metric_rows_csv_path, index=False)
    scaling_df.to_parquet(scaling_summary_path, index=False)
    scaling_df.to_csv(scaling_summary_csv_path, index=False)
    config_df.to_parquet(target_config_path, index=False)
    config_df.to_csv(target_config_csv_path, index=False)
    source_anchor_df.to_parquet(source_anchor_path, index=False)
    source_anchor_df.to_csv(source_anchor_csv_path, index=False)
    blocker_df.to_parquet(blocker_path, index=False)
    blocker_df.to_csv(blocker_csv_path, index=False)
    pair_validation_df.to_parquet(pair_validation_path, index=False)
    render_scaling_curves(summary_df, figure_path)

    validation_df = run_validations(
        s07_df=s07_df,
        s08_df=s08_df,
        config_df=config_df,
        pair_validation_df=pair_validation_df,
        summary_df=summary_df,
        metric_df=metric_df,
        scaling_df=scaling_df,
        blocker_df=blocker_df,
        figure_path=figure_path,
        task_count=len(task_types),
        policy_count=len(policy_ids),
        seed_count=len(seeds),
    )
    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"

    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )
    test_success = all(command["success"] for command in test_commands)

    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)
    write_json(
        config_path,
        {
            "schema": "eidosoma.e05_s09.scaling_config.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "targetConfigCount": len(configs),
            "tasks": list(task_types),
            "policyIds": list(policy_ids),
            "seeds": list(seeds),
            "eventMultiplier": int(args.event_multiplier),
            "recordsPerRun": int(args.records_per_run),
            "noRetuningOnLargeGridTests": True,
            "unitTestsRun": bool(args.run_unit_tests),
            "createdAt": started_at,
        },
    )

    source_files = [
        source_entry(args.repo_dir / "src/e05/scaling.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_scaling.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s09_scaling_tests.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/scrambled_embryo.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/regeneration.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/actions.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/targets.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/morphospace_metrics.py", args.repo_dir),
    ]
    write_json(
        source_manifest_path,
        {
            "schema": "eidosoma.source_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "sourceFiles": source_files,
            "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
            "createdAt": utc_now(),
        },
    )

    outcome_classification, caveats_or_blockers, lay_summary = determine_outcome(summary_df, validation_success and test_success)
    recommended_next_action = "Stop before S10 and let the Chief Scientist review S09; if accepted, proceed to symmetry-breaking tests in S10."

    artifacts_for_summary = [
        artifact_entry(results_path, artifacts_dir, "Required S09 run-level scaling results table."),
        artifact_entry(results_csv_path, artifacts_dir, "CSV mirror of required S09 scaling results."),
        artifact_entry(figure_path, artifacts_dir, "Required S09 scaling-curves figure."),
        artifact_entry(metric_rows_path, artifacts_dir, "S09 initial/final S05 metric rows."),
        artifact_entry(metric_rows_csv_path, artifacts_dir, "CSV mirror of S09 metric rows."),
        artifact_entry(scaling_summary_path, artifacts_dir, "Grouped small-to-large scaling summary."),
        artifact_entry(scaling_summary_csv_path, artifacts_dir, "CSV mirror of grouped scaling summary."),
        artifact_entry(target_config_path, artifacts_dir, "S09 small/large target config table."),
        artifact_entry(target_config_csv_path, artifacts_dir, "CSV mirror of S09 target config table."),
        artifact_entry(source_anchor_path, artifacts_dir, "S07/S08 source-anchor summary used by S09."),
        artifact_entry(source_anchor_csv_path, artifacts_dir, "CSV mirror of S07/S08 source-anchor summary."),
        artifact_entry(blocker_path, artifacts_dir, "S08 blocker table carried into S09."),
        artifact_entry(blocker_csv_path, artifacts_dir, "CSV mirror of S08 blocker table."),
        artifact_entry(pair_validation_path, artifacts_dir, "Scaled target pair validation rows."),
        artifact_entry(validation_path, artifacts_dir, "S09 validation table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV mirror of S09 validation table."),
        artifact_entry(config_path, artifacts_dir, "S09 benchmark configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "Repository source hashes for S09 code and tests."),
    ]
    planned_artifact_paths = [
        {"path": str(full_results_path), "description": "S09 full-results handoff report."},
        {"path": str(artifact_manifest_path), "description": "S09 artifact manifest."},
        {"path": str(run_manifest_path), "description": "Experiment run manifest updated for S09."},
        {"path": str(checksum_path), "description": "Checksums for key S09 artifacts."},
    ]
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts=[*artifacts_for_summary, *planned_artifact_paths],
            validation_result=f"{validation_result}; unit-test commands success={test_success}",
            validation_df=validation_df,
            summary_df=summary_df,
            scaling_df=scaling_df,
            config_df=config_df,
            source_anchor_df=source_anchor_df,
            blocker_df=blocker_df,
            test_commands=test_commands,
            source_files=source_files,
            args=args,
            seeds=seeds,
            policy_ids=policy_ids,
            task_types=task_types,
            outcome_classification=outcome_classification,
            caveats_or_blockers=caveats_or_blockers,
            recommended_next_action=recommended_next_action,
            lay_summary=lay_summary,
        ),
    )
    artifacts_final = [
        *artifacts_for_summary,
        artifact_entry(full_results_path, artifacts_dir, "S09 full-results handoff report."),
    ]
    write_json(
        artifact_manifest_path,
        {
            "schema": "eidosoma.artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "success": bool(validation_success and test_success),
            "artifacts": [*artifacts_final, manifest_self_entry(artifact_manifest_path, artifacts_dir, "S09 artifact manifest.")],
            "validationResult": f"{validation_result}; unit-test commands success={test_success}",
            "outcomeClassification": outcome_classification,
            "caveatsOrBlockers": caveats_or_blockers,
            "recommendedNextAction": recommended_next_action,
            "createdAt": utc_now(),
        },
    )
    run_manifest_payload = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "experimentTitle": EXPERIMENT_TITLE,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "success": bool(validation_success and test_success),
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "packages": {"pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "commands": test_commands,
        "artifacts": [*artifacts_final, artifact_entry(artifact_manifest_path, artifacts_dir, "S09 artifact manifest.")],
        "validationResult": f"{validation_result}; unit-test commands success={test_success}",
        "outcomeClassification": outcome_classification,
    }
    write_json(run_manifest_path, run_manifest_payload)
    checksum_inputs = [
        results_path,
        results_csv_path,
        figure_path,
        metric_rows_path,
        metric_rows_csv_path,
        scaling_summary_path,
        scaling_summary_csv_path,
        target_config_path,
        target_config_csv_path,
        source_anchor_path,
        source_anchor_csv_path,
        blocker_path,
        blocker_csv_path,
        pair_validation_path,
        validation_path,
        validation_csv_path,
        config_path,
        source_manifest_path,
        full_results_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    write_checksums(checksum_inputs, checksum_path, artifacts_dir)

    if not validation_success or not test_success:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
