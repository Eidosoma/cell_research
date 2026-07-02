#!/usr/bin/env python3
"""Run E05 S10 symmetry-breaking tests."""

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

from src.e05.scrambled_embryo import RUNNABLE_POLICY_IDS
from src.e05.symmetry_breaking import (
    AXIS_CHOICES,
    DEFAULT_EVENT_MULTIPLIER,
    DEFAULT_RECORDS_PER_RUN,
    DEFAULT_SYMMETRY_SEEDS,
    default_initial_conditions,
    default_symmetry_target,
    initial_condition_rows,
    run_symmetry_breaking_benchmark,
    source_guardrail_rows,
)


STEP_ID = "S10"
STEP_NUMBER = 10
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SYMMETRY_SEEDS))
    parser.add_argument("--policy-ids", default=",".join(RUNNABLE_POLICY_IDS))
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


def load_s09_scaling_results(artifacts_dir: Path) -> pd.DataFrame:
    path = artifacts_dir / "results" / "e05_scaling_tests.parquet"
    if not path.exists():
        raise FileNotFoundError(f"S10 requires S09 scaling guardrails; missing {path}")
    return pd.read_parquet(path)


def s09_source_anchor_rows(s09_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    grouped = (
        s09_df.groupby(["policy_id", "scale_label"], as_index=False)
        .agg(
            runs=("target_id", "count"),
            mean_target_recovery=("target_recovery_fraction", "mean"),
            retuning_allowed_any=("retuning_allowed", "max"),
            large_grid_retuned_any=("large_grid_retuned", "max"),
        )
        .sort_values(["policy_id", "scale_label"])
    )
    for record in grouped.to_dict(orient="records"):
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": "S09",
                "policy_id": record["policy_id"],
                "scale_label": record["scale_label"],
                "runs": int(record["runs"]),
                "mean_target_recovery_fraction": float(record["mean_target_recovery"]),
                "retuning_allowed_any": bool(record["retuning_allowed_any"]),
                "large_grid_retuned_any": bool(record["large_grid_retuned_any"]),
                "source_note": "S09 no-retuning guardrail and performance context for reusing S07 controls in S10.",
            }
        )
    return rows


def validation_s09_guardrails_loaded(s09_df: pd.DataFrame) -> dict[str, Any]:
    required = {"policy_id", "retuning_allowed", "large_grid_retuned", "target_recovery_fraction"}
    observed = {
        "s09_rows": int(len(s09_df)),
        "required_columns_present": bool(required.issubset(s09_df.columns)),
        "retuning_allowed_any": bool(s09_df["retuning_allowed"].any()),
        "large_grid_retuned_any": bool(s09_df["large_grid_retuned"].any()),
    }
    expected = {
        "s09_rows_gt": 0,
        "required_columns_present": True,
        "retuning_allowed_any": False,
        "large_grid_retuned_any": False,
    }
    success = (
        observed["s09_rows"] > 0
        and observed["required_columns_present"]
        and observed["retuning_allowed_any"] is False
        and observed["large_grid_retuned_any"] is False
    )
    return _row(
        "s09_no_retuning_guardrails_loaded",
        "input",
        success,
        expected,
        observed,
        "S09 scaling results were loaded and confirm no large-grid policy retuning before S10 reuse.",
    )


def validation_initial_states(initial_df: pd.DataFrame) -> dict[str, Any]:
    exact = initial_df[initial_df["symmetry_class"] == "exact_axis_mirror"]
    near = initial_df[initial_df["symmetry_class"] == "near_axis_mirror"]
    observed = {
        "initial_condition_rows": int(len(initial_df)),
        "exact_construction_axis_error_zero": bool((exact["construction_axis_label_symmetry_error"].astype(float) == 0.0).all()),
        "near_construction_axis_error_positive": bool((near["construction_axis_label_symmetry_error"].astype(float) > 0.0).all()),
        "near_construction_axis_error_max": float(near["construction_axis_label_symmetry_error"].max()),
        "all_initial_target_errors_positive": bool((initial_df["initial_target_error"].astype(float) > 0.0).all()),
    }
    expected = {
        "initial_condition_rows": 4,
        "exact_construction_axis_error_zero": True,
        "near_construction_axis_error_positive": True,
        "near_construction_axis_error_max_lte": 0.25,
        "all_initial_target_errors_positive": True,
    }
    success = (
        observed["initial_condition_rows"] == expected["initial_condition_rows"]
        and observed["exact_construction_axis_error_zero"]
        and observed["near_construction_axis_error_positive"]
        and observed["near_construction_axis_error_max"] <= expected["near_construction_axis_error_max_lte"]
        and observed["all_initial_target_errors_positive"]
    )
    return _row(
        "symmetric_and_near_symmetric_initial_states_validated",
        "initial_state",
        success,
        expected,
        observed,
        "Exact initial states have zero construction-axis organ-label mirror error; near states have bounded positive error.",
    )


def validation_run_matrix(summary_df: pd.DataFrame, *, condition_count: int, policy_count: int, seed_count: int) -> dict[str, Any]:
    expected_rows = condition_count * policy_count * seed_count
    observed = {
        "run_count": int(len(summary_df)),
        "expected_run_count": int(expected_rows),
        "condition_count": int(summary_df["condition_id"].nunique()),
        "policy_count": int(summary_df["policy_id"].nunique()),
        "seed_count": int(summary_df["simulation_seed"].nunique()),
    }
    return _row(
        "symmetry_breaking_run_matrix_complete",
        "run_matrix",
        observed["run_count"] == expected_rows,
        {"run_count": expected_rows},
        observed,
        "S10 produced the full initial-condition by policy by seed matrix.",
    )


def validation_axis_metrics(summary_df: pd.DataFrame) -> dict[str, Any]:
    required = {
        "initial_chosen_axis",
        "final_chosen_axis",
        "initial_axis_margin",
        "final_axis_margin",
        "initial_vertical_orientation_error",
        "initial_horizontal_orientation_error",
        "final_vertical_orientation_error",
        "final_horizontal_orientation_error",
        "axis_switch_count",
        "ambiguous_step_fraction",
    }
    observed = {
        "required_columns_present": bool(required.issubset(summary_df.columns)),
        "final_axis_choices_valid": bool(summary_df["final_chosen_axis"].isin(AXIS_CHOICES).all()),
        "axis_margins_finite": bool(np.isfinite(summary_df[["initial_axis_margin", "final_axis_margin"]].astype(float).to_numpy()).all()),
        "orientation_errors_finite": bool(
            np.isfinite(
                summary_df[
                    [
                        "initial_vertical_orientation_error",
                        "initial_horizontal_orientation_error",
                        "final_vertical_orientation_error",
                        "final_horizontal_orientation_error",
                    ]
                ]
                .astype(float)
                .to_numpy()
            ).all()
        ),
    }
    expected = {
        "required_columns_present": True,
        "final_axis_choices_valid": True,
        "axis_margins_finite": True,
        "orientation_errors_finite": True,
    }
    success = all(observed[key] == expected[key] for key in expected)
    return _row(
        "axis_choice_metrics_defined_and_reported",
        "metric",
        success,
        expected,
        observed,
        "Every S10 run has vertical/horizontal orientation errors, chosen-axis labels, margins, switches, and ambiguity fractions.",
    )


def validation_no_organizer(summary_df: pd.DataFrame, guardrail_df: pd.DataFrame) -> dict[str, Any]:
    observed = {
        "organizer_cue_added_any": bool(summary_df["organizer_cue_added"].any()),
        "guardrail_organizer_cue_added_any": bool(guardrail_df["organizer_cue_added"].any()),
        "global_axis_observation_added_any": bool(guardrail_df["global_axis_observation_added"].any()),
        "policy_ids": sorted(summary_df["policy_id"].unique().tolist()),
        "declared_information_access_nonempty": bool(summary_df["declared_information_access"].astype(str).str.len().gt(0).all()),
    }
    expected = {
        "organizer_cue_added_any": False,
        "guardrail_organizer_cue_added_any": False,
        "global_axis_observation_added_any": False,
        "policy_ids": sorted(RUNNABLE_POLICY_IDS),
        "declared_information_access_nonempty": True,
    }
    success = all(observed[key] == expected[key] for key in expected)
    return _row(
        "no_undeclared_organizer_or_axis_cues_added",
        "policy_access",
        success,
        expected,
        observed,
        "S10 reuses S07 runnable policies and records that no organizer cells or global-axis observations were added.",
    )


def validation_trace_and_oscillation(summary_df: pd.DataFrame, trace_df: pd.DataFrame) -> dict[str, Any]:
    run_keys = ["condition_id", "policy_id", "simulation_seed"]
    start_rows = trace_df[trace_df["event_step"] == 0].groupby(run_keys).size()
    final_step_rows = trace_df.groupby(run_keys)["event_step"].max().reset_index()
    observed = {
        "trace_rows": int(len(trace_df)),
        "run_count": int(summary_df[run_keys].drop_duplicates().shape[0]),
        "trace_run_count": int(trace_df[run_keys].drop_duplicates().shape[0]),
        "runs_with_step_zero": int((start_rows >= 1).sum()),
        "runs_with_final_step": int(len(final_step_rows)),
        "axis_switch_count_nonnegative": bool((summary_df["axis_switch_count"].astype(int) >= 0).all()),
        "ambiguous_fraction_in_range": bool(((summary_df["ambiguous_step_fraction"].astype(float) >= 0.0) & (summary_df["ambiguous_step_fraction"].astype(float) <= 1.0)).all()),
    }
    expected = {
        "trace_rows_gt": 0,
        "trace_run_count_matches": True,
        "runs_with_step_zero_matches": True,
        "runs_with_final_step_matches": True,
        "axis_switch_count_nonnegative": True,
        "ambiguous_fraction_in_range": True,
    }
    success = (
        observed["trace_rows"] > 0
        and observed["trace_run_count"] == observed["run_count"]
        and observed["runs_with_step_zero"] == observed["run_count"]
        and observed["runs_with_final_step"] == observed["run_count"]
        and observed["axis_switch_count_nonnegative"]
        and observed["ambiguous_fraction_in_range"]
    )
    return _row(
        "axis_trace_and_oscillation_metrics_recorded",
        "trace_metric",
        success,
        expected,
        observed,
        "Downsampled axis traces support reporting axis switches, ambiguous phases, and final-axis stability.",
    )


def validation_s05_metrics(summary_df: pd.DataFrame, metric_df: pd.DataFrame) -> dict[str, Any]:
    expected_metric_rows = len(summary_df) * 2 * 8
    observed = {
        "metric_rows": int(len(metric_df)),
        "expected_metric_rows": int(expected_metric_rows),
        "metric_values_non_null": bool(metric_df["value"].notna().all()),
        "state_labels": sorted(metric_df["benchmark_state_label"].unique().tolist()),
    }
    expected = {"metric_rows": expected_metric_rows, "metric_values_non_null": True, "state_labels": ["final", "initial"]}
    success = (
        observed["metric_rows"] == expected_metric_rows
        and observed["metric_values_non_null"]
        and observed["state_labels"] == expected["state_labels"]
    )
    return _row(
        "s05_metrics_computed_for_initial_and_final_states",
        "metric",
        success,
        expected,
        observed,
        "S05 metrics were computed for every initial and final S10 state.",
    )


def validation_action_accounting(summary_df: pd.DataFrame) -> dict[str, Any]:
    observed = {
        "population_conserved": bool((summary_df["population_delta_total"].astype(int) == 0).all()),
        "population_min_equals_site_count": bool((summary_df["population_min"].astype(int) == summary_df["site_count"].astype(int)).all()),
        "population_max_equals_site_count": bool((summary_df["population_max"].astype(int) == summary_df["site_count"].astype(int)).all()),
        "total_rejected_actions": int(summary_df["rejected_actions"].sum()),
    }
    expected = {
        "population_conserved": True,
        "population_min_equals_site_count": True,
        "population_max_equals_site_count": True,
        "total_rejected_actions": 0,
    }
    success = all(observed[key] == expected[key] for key in expected)
    return _row(
        "s04_swap_wait_action_accounting_preserved",
        "action",
        success,
        expected,
        observed,
        "S10 swap/wait trajectories conserve population and reject no legal local moves.",
    )


def validation_figure(figure_path: Path) -> dict[str, Any]:
    size = figure_path.stat().st_size if figure_path.exists() else 0
    observed = {"exists": figure_path.exists(), "size_bytes": int(size)}
    expected = {"exists": True, "size_bytes_min": 1000}
    return _row(
        "symmetry_breaking_outcomes_figure_written",
        "artifact",
        observed["exists"] and observed["size_bytes"] > expected["size_bytes_min"],
        expected,
        observed,
        "The required S10 symmetry-breaking outcome figure was written and is nonempty.",
    )


def run_validations(
    *,
    s09_df: pd.DataFrame,
    initial_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    guardrail_df: pd.DataFrame,
    figure_path: Path,
    condition_count: int,
    policy_count: int,
    seed_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            validation_s09_guardrails_loaded(s09_df),
            validation_initial_states(initial_df),
            validation_run_matrix(summary_df, condition_count=condition_count, policy_count=policy_count, seed_count=seed_count),
            validation_axis_metrics(summary_df),
            validation_no_organizer(summary_df, guardrail_df),
            validation_trace_and_oscillation(summary_df, trace_df),
            validation_s05_metrics(summary_df, metric_df),
            validation_action_accounting(summary_df),
            validation_figure(figure_path),
        ]
    )


def render_symmetry_figure(summary_df: pd.DataFrame, trace_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    policy_labels = {
        "s07_local_target_neighbor_descent": "local target-aware",
        "s07_random_adjacent_swap_control": "random adjacent swap",
    }
    colors = {
        "s07_local_target_neighbor_descent": "#2F6B4F",
        "s07_random_adjacent_swap_control": "#8D4E2F",
    }
    outcome = (
        summary_df.groupby(["condition_id", "policy_id"], as_index=False)
        .agg(
            canonical_rate=("canonical_axis_selected_final", "mean"),
            mean_final_margin=("final_axis_margin", "mean"),
            mean_axis_switches=("axis_switch_count", "mean"),
            mean_recovery=("target_recovery_fraction", "mean"),
        )
        .sort_values(["condition_id", "policy_id"])
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), constrained_layout=True)
    conditions = sorted(summary_df["condition_id"].unique())
    x = np.arange(len(conditions))
    width = 0.36
    for offset, policy_id in zip((-width / 2, width / 2), sorted(summary_df["policy_id"].unique()), strict=True):
        data = outcome[outcome["policy_id"] == policy_id].set_index("condition_id").reindex(conditions)
        axes[0, 0].bar(x + offset, data["canonical_rate"], width=width, color=colors.get(policy_id, "#555555"), label=policy_labels.get(policy_id, policy_id))
        axes[0, 1].bar(x + offset, data["mean_final_margin"], width=width, color=colors.get(policy_id, "#555555"), label=policy_labels.get(policy_id, policy_id))
        axes[1, 0].bar(x + offset, data["mean_axis_switches"], width=width, color=colors.get(policy_id, "#555555"), label=policy_labels.get(policy_id, policy_id))
    for ax, title, ylabel in (
        (axes[0, 0], "Canonical vertical axis selected", "fraction of runs"),
        (axes[0, 1], "Final axis-choice margin", "orientation-error margin"),
        (axes[1, 0], "Axis switches in downsampled trace", "mean switches"),
    ):
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    example = trace_df[
        (trace_df["condition_id"] == "exact_horizontal_mirror")
        & (trace_df["policy_id"] == "s07_local_target_neighbor_descent")
        & (trace_df["simulation_seed"] == int(summary_df["simulation_seed"].min()))
    ].sort_values("event_step")
    if example.empty:
        example = trace_df.head(0)
    axes[1, 1].plot(example["event_step"], example["vertical_orientation_error"], color="#2F6B4F", label="vertical error")
    axes[1, 1].plot(example["event_step"], example["horizontal_orientation_error"], color="#8D4E2F", label="horizontal error")
    axes[1, 1].set_title("Example local target-aware axis trajectory")
    axes[1, 1].set_xlabel("event step")
    axes[1, 1].set_ylabel("orientation target error")
    axes[1, 1].grid(True, linewidth=0.4, alpha=0.35)
    axes[1, 1].legend(frameon=False, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("E05 S10 symmetry-breaking outcomes without added organizer cues", fontsize=13)
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


def determine_outcome(summary_df: pd.DataFrame, validation_success: bool) -> tuple[str, str, str]:
    by_policy = summary_df.groupby("policy_id", as_index=False).agg(canonical_rate=("canonical_axis_selected_final", "mean"))
    local_rate = float(by_policy.loc[by_policy["policy_id"] == "s07_local_target_neighbor_descent", "canonical_rate"].iloc[0])
    random_rate = float(by_policy.loc[by_policy["policy_id"] == "s07_random_adjacent_swap_control", "canonical_rate"].iloc[0])
    if validation_success and local_rate > random_rate and local_rate >= 0.60:
        outcome = "supportive"
        lay = (
            "S10 found that the declared local target-aware control resolves symmetric and near-symmetric starts toward the canonical "
            "vertical axis more often than the random adjacent-swap control, without adding organizer cells or new axis observations."
        )
    else:
        outcome = "constraining/contradictory"
        lay = (
            "S10 quantified axis choice, but the available local controls did not robustly select the canonical axis from symmetric starts; "
            "this suggests reliable symmetry breaking may need additional declared cues or mechanisms."
        )
    caveats = (
        "S10 uses a square toy bilateral target and compares canonical vertical versus horizontal target orientations. The local target-aware "
        "control keeps its S07 local target-site access, which is not an added organizer cue but is still a target-template cue. Random "
        "adjacent swaps have no target access."
    )
    return outcome, caveats, lay


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
    initial_df: pd.DataFrame,
    guardrail_df: pd.DataFrame,
    source_anchor_df: pd.DataFrame,
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    args: argparse.Namespace,
    seeds: Sequence[int],
    policy_ids: Sequence[str],
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
    outcome_summary = (
        summary_df.groupby(["condition_id", "policy_id"], as_index=False)
        .agg(
            runs=("simulation_seed", "count"),
            canonical_axis_rate=("canonical_axis_selected_final", "mean"),
            construction_axis_retained_rate=("construction_axis_retained_final", "mean"),
            mean_initial_target_error=("initial_target_error", "mean"),
            mean_final_target_error=("final_target_error", "mean"),
            mean_target_recovery=("target_recovery_fraction", "mean"),
            mean_final_axis_margin=("final_axis_margin", "mean"),
            mean_axis_switch_count=("axis_switch_count", "mean"),
            mean_ambiguous_step_fraction=("ambiguous_step_fraction", "mean"),
        )
        .sort_values(["condition_id", "policy_id"])
    )
    for frame in (outcome_summary, initial_df, source_anchor_df):
        for column in frame.select_dtypes(include=[float]).columns:
            frame[column] = frame[column].round(6)
    return f"""{top_summary_markdown(artifacts, validation_result, outcome_classification, caveats_or_blockers, recommended_next_action, lay_summary)}

# Research Step Full Results: {STEP_ID} Run Symmetry-Breaking Tests

## Lay Summary

{lay_summary}

## Frozen Question

Can local policies reliably choose a consistent global axis from symmetric random states?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S10.
- S03 target constructor: `symmetry_target`, instantiated as a square 7x7 bilateral target so vertical and horizontal axis alternatives are both valid.
- S07 controls: `s07_local_target_neighbor_descent` and `s07_random_adjacent_swap_control`, reused without changing declared information access.
- S05 metrics: all default morphospace metrics at initial and final states.
- S09 scaling results: `$ARTIFACTS_DIR/results/e05_scaling_tests.parquet`, used as a no-retuning guardrail.
- Datasets: none required.

## Detailed Methods

S10 constructs four initial-state classes: exact vertical mirror, near vertical mirror, exact horizontal mirror, and near horizontal mirror. Exact states have zero organ-label mirror-symmetry error about their construction axis. Near states start from the corresponding exact state and apply `{default_initial_conditions()[1].near_symmetry_break_swaps}` cross-label swaps to make symmetry slightly imperfect while preserving population.

Axis choice is measured without adding organizer cells or global-axis observations. For each state, S10 computes identity error to the canonical vertical S03 target and to a horizontal alternative obtained by transposing the square target's identity map. If the two orientation errors differ by at most the fixed ambiguity tolerance, the state is labeled `ambiguous`; otherwise the lower-error orientation is the chosen axis. Downsampled traces record chosen axis, axis margin, orientation errors, organ-label mirror errors, and target error.

The local target-aware control keeps the same declared S07 local access: actor identity, adjacent-neighbor identities, and target identities at the actor and adjacent-neighbor sites. The random control keeps no target access. S10 uses the same S04 swap/wait legality and energy semantics as S09's optimized inner loop; no new policy observation or organizer cue is introduced.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- Matplotlib: `{matplotlib.__version__}`
- New packages installed: none.
- CPU/GPU use: serial CPU benchmark and validation; no GPU use was needed for this compact S10 matrix.

## Parameters

- Seeds: `{", ".join(str(seed) for seed in seeds)}`
- Runnable policies: `{", ".join(policy_ids)}`
- Event multiplier: `{args.event_multiplier}` local scheduler events per target site
- Records per run: `{args.records_per_run}` downsampled axis-trace records per run
- Initial conditions: `{", ".join(default_initial_conditions()[idx].condition_id for idx in range(len(default_initial_conditions())))}`

## Results

Validation summary:

{markdown_table(validation_df, ["validation_case", "case_type", "success", "detail"])}

Initial-state construction summary:

{markdown_table(initial_df, ["condition_id", "construction_axis", "symmetry_class", "initial_target_error", "construction_axis_label_symmetry_error", "initial_chosen_axis", "initial_axis_margin"])}

Axis outcome summary:

{markdown_table(outcome_summary, ["condition_id", "policy_id", "runs", "canonical_axis_rate", "construction_axis_retained_rate", "mean_initial_target_error", "mean_final_target_error", "mean_target_recovery", "mean_final_axis_margin", "mean_axis_switch_count", "mean_ambiguous_step_fraction"])}

Policy access guardrails:

{markdown_table(guardrail_df, ["policy_id", "policy_family", "organizer_cue_added", "global_axis_observation_added", "declared_information_access"])}

S09 source guardrails:

{markdown_table(source_anchor_df, ["policy_id", "scale_label", "runs", "mean_target_recovery_fraction", "retuning_allowed_any", "large_grid_retuned_any", "source_note"])}

## Metrics

- Vertical orientation error: S03 identity error against the canonical bilateral-symmetry target.
- Horizontal orientation error: S03 identity error against a transposed target-identity map on the same square substrate.
- Chosen axis: `vertical`, `horizontal`, or `ambiguous` according to the lower orientation error and the fixed ambiguity tolerance.
- Axis margin: absolute difference between vertical and horizontal orientation errors.
- Organ-label symmetry error: fraction of mirror-paired sites with different observed organ labels for a candidate axis.
- Oscillation proxy: number of chosen-axis label changes in the downsampled trace plus the fraction of trace points marked ambiguous.
- Recovery and energy: S07/S09-compatible target-recovery fraction and S04 swap/wait charged energy.

## Figures And Tables

- The required outcome figure is `$ARTIFACTS_DIR/figures/e05/symmetry_breaking_outcomes.png`.
- The required run table is `$ARTIFACTS_DIR/results/e05_symmetry_breaking.parquet` with a CSV companion.
- Additional machine-readable tables record axis traces, S05 metric rows, initial-condition validation, S09 guardrail summaries, policy-access guardrails, and validation cases.

## Validation Checks

- Loaded S09 no-retuning guardrails before S10 execution.
- Validated exact and near symmetric initial-state construction.
- Produced the full condition by policy by seed run matrix.
- Defined and reported axis-choice metrics for all runs.
- Verified that no undeclared organizer or global-axis cue was added.
- Recorded downsampled axis traces and oscillation metrics.
- Computed S05 metrics for every initial and final state.
- Preserved S04 swap/wait population accounting.
- Wrote the required nonempty symmetry-breaking outcome figure.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- S10 uses a square toy target so vertical and horizontal target orientations can be compared on the same substrate.
- The local target-aware control has declared local target-site access from S07. This is not an added organizer cue, but it means the control is not a no-target spontaneous symmetry-breaking mechanism.
- The random adjacent-swap control has no target access and is expected to be weak on target-axis selection.
- Axis-choice metrics are computational proxies over a fixed target identity map; they are not evidence for biological organizer formation.
- Oscillation is measured on downsampled axis labels, not on every individual event.

## Blockers And Failed Assumptions

No execution blocker was encountered. The main interpretive constraint is that reliable canonical-axis selection by the local target-aware control depends on its declared local target-template access. S10 therefore distinguishes target-template-guided axis resolution from spontaneous symmetry breaking without target cues.

## Recommended Next Action

{recommended_next_action}
"""


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
    s09_df = load_s09_scaling_results(artifacts_dir)
    target = default_symmetry_target()
    conditions = default_initial_conditions()
    initial_df = pd.DataFrame(initial_condition_rows(target, conditions, seed=seeds[0]))
    summary_rows, trace_rows, metric_rows, run_results = run_symmetry_breaking_benchmark(
        target=target,
        conditions=conditions,
        policy_ids=policy_ids,
        seeds=seeds,
        event_multiplier=args.event_multiplier,
        records_per_run=args.records_per_run,
    )
    summary_df = pd.DataFrame(summary_rows)
    trace_df = pd.DataFrame(trace_rows)
    metric_df = pd.DataFrame(metric_rows)
    guardrail_df = pd.DataFrame(source_guardrail_rows())
    source_anchor_df = pd.DataFrame(s09_source_anchor_rows(s09_df))

    results_path = results_dir / "e05_symmetry_breaking.parquet"
    results_csv_path = tables_dir / "e05_symmetry_breaking.csv"
    trace_path = results_dir / "e05_symmetry_breaking_axis_trace.parquet"
    trace_csv_path = tables_dir / "e05_symmetry_breaking_axis_trace.csv"
    metric_rows_path = results_dir / "e05_symmetry_breaking_metric_rows.parquet"
    metric_rows_csv_path = tables_dir / "e05_symmetry_breaking_metric_rows.csv"
    initial_path = results_dir / "e05_symmetry_breaking_initial_conditions.parquet"
    initial_csv_path = tables_dir / "e05_symmetry_breaking_initial_conditions.csv"
    guardrail_path = results_dir / "e05_symmetry_breaking_policy_guardrails.parquet"
    guardrail_csv_path = tables_dir / "e05_symmetry_breaking_policy_guardrails.csv"
    source_anchor_path = results_dir / "e05_symmetry_breaking_s09_source_anchors.parquet"
    source_anchor_csv_path = tables_dir / "e05_symmetry_breaking_s09_source_anchors.csv"
    validation_path = results_dir / "e05_symmetry_breaking_validation.parquet"
    validation_csv_path = tables_dir / "e05_symmetry_breaking_validation.csv"
    figure_path = figures_dir / "symmetry_breaking_outcomes.png"
    config_path = configs_dir / "e05_s10_symmetry_breaking.json"
    source_manifest_path = src_snapshot_dir / "e05_symmetry_breaking_manifest.json"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    summary_df.to_parquet(results_path, index=False)
    summary_df.to_csv(results_csv_path, index=False)
    trace_df.to_parquet(trace_path, index=False)
    trace_df.to_csv(trace_csv_path, index=False)
    metric_df.to_parquet(metric_rows_path, index=False)
    metric_df.to_csv(metric_rows_csv_path, index=False)
    initial_df.to_parquet(initial_path, index=False)
    initial_df.to_csv(initial_csv_path, index=False)
    guardrail_df.to_parquet(guardrail_path, index=False)
    guardrail_df.to_csv(guardrail_csv_path, index=False)
    source_anchor_df.to_parquet(source_anchor_path, index=False)
    source_anchor_df.to_csv(source_anchor_csv_path, index=False)
    render_symmetry_figure(summary_df, trace_df, figure_path)

    validation_df = run_validations(
        s09_df=s09_df,
        initial_df=initial_df,
        summary_df=summary_df,
        trace_df=trace_df,
        metric_df=metric_df,
        guardrail_df=guardrail_df,
        figure_path=figure_path,
        condition_count=len(conditions),
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
            "schema": "eidosoma.e05_s10.symmetry_breaking_config.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "targetId": target.target_id,
            "targetKind": target.target_kind,
            "conditions": [condition.compact_dict() for condition in conditions],
            "policyIds": list(policy_ids),
            "seeds": list(seeds),
            "eventMultiplier": int(args.event_multiplier),
            "recordsPerRun": int(args.records_per_run),
            "axisChoices": list(AXIS_CHOICES),
            "noOrganizerCueAdded": True,
            "unitTestsRun": bool(args.run_unit_tests),
            "createdAt": started_at,
        },
    )

    source_files = [
        source_entry(args.repo_dir / "src/e05/symmetry_breaking.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_symmetry_breaking.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s10_symmetry_breaking.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/scaling.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/scrambled_embryo.py", args.repo_dir),
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
    recommended_next_action = "Stop before S11 and let the Chief Scientist review S10; if accepted, proceed to GPU batched tissue simulations in S11."

    artifacts_for_summary = [
        artifact_entry(results_path, artifacts_dir, "Required S10 run-level symmetry-breaking results table."),
        artifact_entry(results_csv_path, artifacts_dir, "CSV mirror of required S10 results."),
        artifact_entry(figure_path, artifacts_dir, "Required S10 symmetry-breaking outcome figure."),
        artifact_entry(trace_path, artifacts_dir, "S10 downsampled axis trace table."),
        artifact_entry(trace_csv_path, artifacts_dir, "CSV mirror of S10 axis trace table."),
        artifact_entry(metric_rows_path, artifacts_dir, "S10 initial/final S05 metric rows."),
        artifact_entry(metric_rows_csv_path, artifacts_dir, "CSV mirror of S10 metric rows."),
        artifact_entry(initial_path, artifacts_dir, "S10 initial-condition validation table."),
        artifact_entry(initial_csv_path, artifacts_dir, "CSV mirror of S10 initial-condition table."),
        artifact_entry(guardrail_path, artifacts_dir, "S10 policy-access guardrail table."),
        artifact_entry(guardrail_csv_path, artifacts_dir, "CSV mirror of S10 policy-access guardrails."),
        artifact_entry(source_anchor_path, artifacts_dir, "S09 source guardrail summary used by S10."),
        artifact_entry(source_anchor_csv_path, artifacts_dir, "CSV mirror of S09 source guardrails."),
        artifact_entry(validation_path, artifacts_dir, "S10 validation table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV mirror of S10 validation table."),
        artifact_entry(config_path, artifacts_dir, "S10 benchmark configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "Repository source hashes for S10 code and tests."),
    ]
    planned_artifact_paths = [
        {"path": str(full_results_path), "description": "S10 full-results handoff report."},
        {"path": str(artifact_manifest_path), "description": "S10 artifact manifest."},
        {"path": str(run_manifest_path), "description": "Experiment run manifest updated for S10."},
        {"path": str(checksum_path), "description": "Checksums for key S10 artifacts."},
    ]
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts=[*artifacts_for_summary, *planned_artifact_paths],
            validation_result=f"{validation_result}; unit-test commands success={test_success}",
            validation_df=validation_df,
            summary_df=summary_df,
            initial_df=initial_df,
            guardrail_df=guardrail_df,
            source_anchor_df=source_anchor_df,
            test_commands=test_commands,
            source_files=source_files,
            args=args,
            seeds=seeds,
            policy_ids=policy_ids,
            outcome_classification=outcome_classification,
            caveats_or_blockers=caveats_or_blockers,
            recommended_next_action=recommended_next_action,
            lay_summary=lay_summary,
        ),
    )
    artifacts_final = [
        *artifacts_for_summary,
        artifact_entry(full_results_path, artifacts_dir, "S10 full-results handoff report."),
    ]
    write_json(
        artifact_manifest_path,
        {
            "schema": "eidosoma.artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "success": bool(validation_success and test_success),
            "artifacts": [*artifacts_final, manifest_self_entry(artifact_manifest_path, artifacts_dir, "S10 artifact manifest.")],
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
        "artifacts": [*artifacts_final, artifact_entry(artifact_manifest_path, artifacts_dir, "S10 artifact manifest.")],
        "validationResult": f"{validation_result}; unit-test commands success={test_success}",
        "outcomeClassification": outcome_classification,
    }
    write_json(run_manifest_path, run_manifest_payload)
    checksum_inputs = [
        results_path,
        results_csv_path,
        figure_path,
        trace_path,
        trace_csv_path,
        metric_rows_path,
        metric_rows_csv_path,
        initial_path,
        initial_csv_path,
        guardrail_path,
        guardrail_csv_path,
        source_anchor_path,
        source_anchor_csv_path,
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
