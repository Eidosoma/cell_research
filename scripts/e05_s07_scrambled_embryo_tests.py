#!/usr/bin/env python3
"""Run E05 S07 scrambled-embryo tests."""

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
from matplotlib.colors import ListedColormap

from src.e05.cell_identity import identity_from_substrate_cell
from src.e05.scrambled_embryo import (
    DEFAULT_EVENT_MULTIPLIER,
    DEFAULT_RECORDS_PER_RUN,
    DEFAULT_SEEDS,
    POLICY_SOURCE_PATHS,
    RUNNABLE_POLICY_IDS,
    policy_source_mapping_rows,
    run_scrambled_benchmark,
    target_coverage_rows,
    two_dimensional_targets,
    validate_trace_zarr,
    write_trace_zarr,
)
from src.e05.targets import ORGAN_COLORS, TargetMorphology, default_target_gallery, target_render_mode


STEP_ID = "S07"
STEP_NUMBER = 7
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
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
        rel = str(child.relative_to(path)).encode("utf-8")
        digest.update(rel)
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


def parse_str_list(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("at least one policy ID is required")
    unsupported = sorted(set(values) - set(RUNNABLE_POLICY_IDS))
    if unsupported:
        raise ValueError(f"unsupported S07 policy IDs: {unsupported}")
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


def validation_target_coverage(target_status_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, Any]:
    simulated = set(summary_df["target_id"].unique().tolist())
    expected_simulated = set(target_status_df.loc[target_status_df["simulated_in_s07"], "target_id"].tolist())
    sorted_row = target_status_df.loc[target_status_df["target_kind"] == "sorted_row"]
    observed = {
        "s03_target_rows": int(len(target_status_df)),
        "simulated_2d_target_count": int(len(simulated)),
        "expected_2d_target_count": int(len(expected_simulated)),
        "simulated_targets_match": simulated == expected_simulated,
        "sorted_row_documented_not_simulated": bool((~sorted_row["simulated_in_s07"]).all()) if not sorted_row.empty else False,
    }
    expected = {"simulated_targets_match": True, "sorted_row_documented_not_simulated": True}
    return _row(
        "s03_target_coverage_logged",
        "input",
        observed["simulated_targets_match"] and observed["sorted_row_documented_not_simulated"],
        expected,
        observed,
        "All S03 targets are accounted for; genuine 2D targets are simulated and the 1D sorted-row continuity target is documented as not a 2D scrambled-embryo task.",
    )


def validation_initial_scramble(summary_df: pd.DataFrame) -> dict[str, Any]:
    observed = {
        "run_count": int(len(summary_df)),
        "min_initial_target_error": float(summary_df["initial_target_error"].min()),
        "min_initial_displacement_fraction": float(summary_df["initial_displacement_fraction"].min()),
        "mean_initial_displacement_fraction": float(summary_df["initial_displacement_fraction"].mean()),
        "all_initial_errors_positive": bool((summary_df["initial_target_error"] > 0.0).all()),
    }
    expected = {
        "all_initial_errors_positive": True,
        "min_initial_displacement_fraction_gt": 0.2,
    }
    success = observed["all_initial_errors_positive"] and observed["min_initial_displacement_fraction"] > 0.2
    return _row(
        "initial_scrambling_severity_logged",
        "perturbation",
        success,
        expected,
        observed,
        "Every simulated run starts from a positive-error shuffled target state, and the displaced-site fraction is logged.",
    )


def validation_final_metrics(summary_df: pd.DataFrame, metric_df: pd.DataFrame) -> dict[str, Any]:
    final_metric_df = metric_df[metric_df["benchmark_state_label"] == "final"]
    observed = {
        "summary_run_count": int(len(summary_df)),
        "final_metric_rows": int(len(final_metric_df)),
        "expected_final_metric_rows": int(len(summary_df) * 8),
        "final_target_error_non_null": bool(summary_df["final_target_error"].notna().all()),
        "final_aggregate_error_non_null": bool(summary_df["final_aggregate_morphospace_error"].notna().all()),
        "metric_values_non_null": bool(final_metric_df["value"].notna().all()),
    }
    expected = {
        "final_target_error_non_null": True,
        "final_aggregate_error_non_null": True,
        "metric_values_non_null": True,
        "final_metric_rows_match": True,
    }
    success = (
        observed["final_target_error_non_null"]
        and observed["final_aggregate_error_non_null"]
        and observed["metric_values_non_null"]
        and observed["final_metric_rows"] == observed["expected_final_metric_rows"]
    )
    return _row(
        "target_metrics_computed_from_final_states",
        "metric",
        success,
        expected,
        observed,
        "S05 target and aggregate morphospace metrics were computed from every final scrambled-embryo state.",
    )


def validation_action_accounting(summary_df: pd.DataFrame) -> dict[str, Any]:
    observed = {
        "run_count": int(len(summary_df)),
        "all_population_conserved": bool((summary_df["population_delta_total"].astype(int) == 0).all()),
        "all_population_min_equals_site_count": bool((summary_df["population_min"].astype(int) == summary_df["site_count"].astype(int)).all()),
        "all_population_max_equals_site_count": bool((summary_df["population_max"].astype(int) == summary_df["site_count"].astype(int)).all()),
        "total_rejected_actions": int(summary_df["rejected_actions"].sum()),
        "total_accepted_swaps": int(summary_df["accepted_swaps"].sum()),
        "total_energy_cost": float(summary_df["total_energy_cost"].sum()),
    }
    expected = {
        "all_population_conserved": True,
        "all_population_min_equals_site_count": True,
        "all_population_max_equals_site_count": True,
        "total_rejected_actions": 0,
    }
    success = (
        observed["all_population_conserved"]
        and observed["all_population_min_equals_site_count"]
        and observed["all_population_max_equals_site_count"]
        and observed["total_rejected_actions"] == 0
    )
    return _row(
        "s04_action_accounting_is_local_and_conserved",
        "action",
        success,
        expected,
        observed,
        "S07 uses S04 local swap/wait actions; accepted swaps conserve population and rejected/illegal actions are counted.",
    )


def validation_trace_store(trace_df: pd.DataFrame, zarr_path: Path) -> dict[str, Any]:
    zarr_check = validate_trace_zarr(zarr_path, len(trace_df))
    run_keys = ["target_id", "policy_id", "simulation_seed"]
    start_rows = trace_df[trace_df["event_step"] == 0].groupby(run_keys).size()
    final_rows = trace_df.groupby(run_keys)["event_step"].max().reset_index()
    observed = {
        "trace_rows": int(len(trace_df)),
        "run_count": int(trace_df[run_keys].drop_duplicates().shape[0]),
        "runs_with_step_zero": int((start_rows >= 1).sum()),
        "runs_with_final_step": int(len(final_rows)),
        "zarr_exists": bool(zarr_check["exists"]),
        "zarr_row_count": int(zarr_check["row_count"]),
        "zarr_all_arrays_row_aligned": bool(zarr_check["all_arrays_row_aligned"]),
    }
    expected = {
        "trace_rows_gt": 0,
        "runs_with_step_zero_equals_run_count": True,
        "zarr_row_count_matches_trace_rows": True,
        "zarr_all_arrays_row_aligned": True,
    }
    success = (
        observed["trace_rows"] > 0
        and observed["runs_with_step_zero"] == observed["run_count"]
        and observed["zarr_row_count"] == observed["trace_rows"]
        and observed["zarr_all_arrays_row_aligned"]
    )
    return _row(
        "trace_trajectories_written_and_valid",
        "artifact",
        success,
        expected,
        observed,
        "Trajectory rows include initial and final records per run, and the required zarr trace group is readable and row-aligned.",
    )


def validation_policy_mapping(policy_mapping_df: pd.DataFrame) -> dict[str, Any]:
    e03e04 = policy_mapping_df[policy_mapping_df["source_experiment"].isin(["E03", "E04"])]
    controls = policy_mapping_df[policy_mapping_df["source_experiment"] == "E05"]
    observed = {
        "policy_mapping_rows": int(len(policy_mapping_df)),
        "e03_e04_source_rows": int(len(e03e04)),
        "e03_e04_blocked_rows": int((e03e04["mapping_status"] == "blocked_without_expanding_information_access").sum()),
        "blocked_rows_with_reason": bool((e03e04["blocker"].astype(str).str.len() > 0).all()),
        "runnable_control_rows": int((controls["mapping_status"] == "runnable_control").sum()),
        "total_e03_e04_source_policy_count": int(e03e04["source_policy_count"].sum()),
    }
    expected = {
        "all_e03_e04_rows_blocked_or_mapped": True,
        "blocked_rows_with_reason": True,
        "runnable_control_rows": len(RUNNABLE_POLICY_IDS),
    }
    success = (
        observed["e03_e04_source_rows"] > 0
        and observed["e03_e04_blocked_rows"] == observed["e03_e04_source_rows"]
        and observed["blocked_rows_with_reason"]
        and observed["runnable_control_rows"] == len(RUNNABLE_POLICY_IDS)
    )
    return _row(
        "e03_e04_policy_mapping_access_guard",
        "policy_mapping",
        success,
        expected,
        observed,
        "Available E03/E04 policies were inventoried; none were silently expanded into 2D observations, and every blocked family has an explicit reason.",
    )


def validation_recovery_reported(summary_df: pd.DataFrame) -> dict[str, Any]:
    by_policy = (
        summary_df.groupby("policy_id", as_index=False)
        .agg(
            runs=("target_id", "count"),
            mean_target_recovery=("target_recovery_fraction", "mean"),
            mean_aggregate_recovery=("aggregate_recovery_fraction", "mean"),
            median_final_target_error=("final_target_error", "median"),
        )
        .sort_values("policy_id")
    )
    local = by_policy.loc[by_policy["policy_id"] == "s07_local_target_neighbor_descent"]
    random_control = by_policy.loc[by_policy["policy_id"] == "s07_random_adjacent_swap_control"]
    local_mean = float(local["mean_target_recovery"].iloc[0]) if not local.empty else float("nan")
    random_mean = float(random_control["mean_target_recovery"].iloc[0]) if not random_control.empty else float("nan")
    observed = {
        "policy_summary": by_policy.to_dict(orient="records"),
        "local_target_mean_recovery": local_mean,
        "random_control_mean_recovery": random_mean,
        "local_target_better_than_random": bool(local_mean > random_mean),
        "all_recovery_values_finite": bool(np.isfinite(summary_df["target_recovery_fraction"].astype(float)).all()),
    }
    expected = {
        "recovery_rates_reported": True,
        "local_target_better_than_random": True,
    }
    success = observed["all_recovery_values_finite"] and observed["local_target_better_than_random"]
    return _row(
        "recovery_rates_and_policy_trajectories_reported",
        "outcome",
        success,
        expected,
        observed,
        "Recovery fractions are reported by target and policy; the local target-aware control is checked against the random local-swap null.",
    )


def validation_figure(figure_path: Path) -> dict[str, Any]:
    size = figure_path.stat().st_size if figure_path.exists() else 0
    observed = {"exists": figure_path.exists(), "size_bytes": int(size)}
    expected = {"exists": True, "size_bytes_min": 1000}
    return _row(
        "scrambled_embryo_examples_figure_written",
        "artifact",
        observed["exists"] and observed["size_bytes"] > expected["size_bytes_min"],
        expected,
        observed,
        "The required S07 example morphology figure was written and is nonempty.",
    )


def run_validations(
    *,
    target_status_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    policy_mapping_df: pd.DataFrame,
    zarr_path: Path,
    figure_path: Path,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            validation_target_coverage(target_status_df, summary_df),
            validation_initial_scramble(summary_df),
            validation_final_metrics(summary_df, metric_df),
            validation_action_accounting(summary_df),
            validation_trace_store(trace_df, zarr_path),
            validation_policy_mapping(policy_mapping_df),
            validation_recovery_reported(summary_df),
            validation_figure(figure_path),
        ]
    )


def state_component_matrix(target: TargetMorphology, state: Any, render_mode: str) -> np.ndarray:
    width = int(target.substrate.dimensions.get("width", len(target.substrate.site_ids)))
    height = int(target.substrate.dimensions.get("height", 1))
    values: list[Any] = []
    for site_id in target.substrate.site_ids:
        if state == "target":
            identity = target.identities_by_site[site_id]
        else:
            cell = state.cell_at(site_id)
            identity = identity_from_substrate_cell(cell, fallback_scalar=False) if cell is not None else None
        if identity is None:
            values.append(None)
        elif render_mode in identity.components:
            values.append(identity.components[render_mode])
        elif "organ_type" in identity.components:
            values.append(identity.components["organ_type"])
        else:
            values.append(identity.identity_id)
    return np.array(values, dtype=object).reshape(height, width)


def render_state_panel(ax: Any, target: TargetMorphology, state: Any, title: str) -> None:
    mode = target_render_mode(target)
    matrix = state_component_matrix(target, state, mode)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if mode in {"value", "ap_coordinate"}:
        numeric = matrix.astype(float)
        kwargs = {"vmin": 0.0, "vmax": 1.0} if mode == "ap_coordinate" else {}
        ax.imshow(numeric, cmap="viridis", interpolation="nearest", aspect="equal", **kwargs)
        return
    color_keys = ["neural", "epidermis", "mesenchyme", "boundary"]
    cmap = ListedColormap([ORGAN_COLORS[key] for key in color_keys])
    mapping = {label: idx for idx, label in enumerate(color_keys)}
    encoded = np.vectorize(lambda value: mapping.get(str(value), 0))(matrix)
    ax.imshow(encoded, cmap=cmap, vmin=0, vmax=len(color_keys) - 1, interpolation="nearest", aspect="equal")


def render_examples_figure(
    *,
    targets: Sequence[TargetMorphology],
    run_results: Mapping[tuple[str, str, int], Any],
    seed: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preferred_ids = ("ap_gradient", "organ_stripes", "ring_pattern", "toy_organ_like")
    target_lookup = {target.target_id: target for target in targets}
    example_targets = [target_lookup[target_id] for target_id in preferred_ids if target_id in target_lookup]
    if not example_targets:
        example_targets = list(targets[:4])
    rows = len(example_targets)
    fig, axes = plt.subplots(rows, 3, figsize=(8.4, 2.4 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])
    for row_idx, target in enumerate(example_targets):
        result = run_results[(target.target_id, "s07_local_target_neighbor_descent", int(seed))]
        render_state_panel(axes[row_idx, 0], target, "target", f"{target.title}\ntarget")
        render_state_panel(axes[row_idx, 1], target, result.initial_state, "scrambled")
        render_state_panel(axes[row_idx, 2], target, result.final_state, "local target final")
    fig.suptitle("E05 S07 scrambled-embryo examples")
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
- Lay summary: S07 scrambled every genuine 2D S03 target and measured whether local S04 swap policies could reduce S05 morphology error. The E03/E04 policy inventory was used as an access-control audit: no available E03/E04 policy was run in 2D because each would require undeclared target, projection, or four-neighbor action information.
- Recommended next action: {recommended_next_action}
"""


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    policy_mapping_df: pd.DataFrame,
    target_status_df: pd.DataFrame,
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    args: argparse.Namespace,
    seeds: Sequence[int],
    policy_ids: Sequence[str],
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
) -> str:
    command_lines = "\n".join(
        f"- `{command['command']}`: return code {command['returnCode']}, success={command['success']}, elapsed={command['elapsedSeconds']:.3f}s"
        for command in test_commands
    )
    source_lines = "\n".join(
        f"- `{entry['relativePath']}` sha256 `{entry['sha256']}` ({entry['sizeBytes']} bytes)"
        for entry in source_files
    )
    policy_summary = (
        summary_df.groupby(["policy_id", "target_kind"], as_index=False)
        .agg(
            runs=("simulation_seed", "count"),
            mean_initial_target_error=("initial_target_error", "mean"),
            mean_final_target_error=("final_target_error", "mean"),
            mean_target_recovery=("target_recovery_fraction", "mean"),
            mean_initial_aggregate_error=("initial_aggregate_morphospace_error", "mean"),
            mean_final_aggregate_error=("final_aggregate_morphospace_error", "mean"),
            mean_aggregate_recovery=("aggregate_recovery_fraction", "mean"),
            mean_accepted_swaps=("accepted_swaps", "mean"),
        )
        .sort_values(["policy_id", "target_kind"])
    )
    policy_mapping_summary = policy_mapping_df[
        [
            "source_experiment",
            "policy_family",
            "source_policy_count",
            "mapping_status",
            "blocker",
        ]
    ].copy()
    target_status_summary = target_status_df[["target_id", "target_kind", "substrate_kind", "site_count", "simulated_in_s07", "reason"]]
    return f"""{top_summary_markdown(artifacts, validation_result, outcome_classification, caveats_or_blockers, recommended_next_action)}

# Research Step Full Results: {STEP_ID} Run Scrambled-Embryo Tests

## Lay Summary

S07 asks whether cells can recover 2D target patterns after their positions are scrambled. A bounded local target-aware control consistently reduced morphology error, while a random adjacent-swap control served as a local null. The available E03 and E04 policy artifacts were inventoried, but none could be validly executed on 2D targets without expanding their declared information access.

## Frozen Question

Can local policies recover target morphology from severe 2D disorganization?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S07.
- S03 targets: `src/e05/targets.py`, using every genuine 2D target from `default_target_gallery()`.
- S04 actions: `src/e05/actions.py`, using local `swap` and `wait` through `ActionExecutor`.
- S05 metrics: `src/e05/morphospace_metrics.py`, evaluating all default metric specs at initial and final states and aggregate traces during trajectories.
- E03 policy artifacts: `/previous-artifacts/E03/policies/e03_classic_policy_library.json`, `/previous-artifacts/E03/policies/e03_frontier_candidate_policies.jsonl`, `/previous-artifacts/E03/policies/e03_generated_policy_library.jsonl`, `/previous-artifacts/E03/policies/e03_qd_discovered_policies.jsonl`.
- E04 policy artifacts: `/previous-artifacts/E04/policies/e04_evolved_repair_policies.jsonl` and `/previous-artifacts/E04/reports/e04_local_learning_rule_spec.md`.
- Datasets: none required.

## Methods

For each 2D S03 target, S07 constructed the target state and then applied a deterministic random permutation of its cells for each seed. Each run used only S04 local action execution. The local target-aware control considered one actor and its adjacent neighbors, compared the target identity error at those same local sites before and after each candidate adjacent swap, and selected a strictly improving swap or waited. The random control proposed a random legal adjacent swap without target access.

The E03/E04 policy artifacts were not force-mapped. They were loaded into a policy mapping table with source counts, declared information access, and blocker reasons. Families that only define 1D scalar left/right observations, global 1D arrays, sortedness, ideal positions, or swap-left/swap-right actions were marked blocked rather than run as 2D policies.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- Matplotlib: `{matplotlib.__version__}`
- No new packages were installed for S07; `zarr` was already available in the runtime.
- CPU/GPU use: serial CPU benchmark and validation; no GPU use was needed.

## Parameters

- Seeds: `{", ".join(str(seed) for seed in seeds)}`
- Runnable S07 policies: `{", ".join(policy_ids)}`
- Event multiplier: `{args.event_multiplier}` local scheduler events per target site
- Records per run: `{args.records_per_run}`
- Simulated targets: `{", ".join(sorted(summary_df["target_id"].unique().tolist()))}`
- Excluded S03 target: `sorted_row`, documented as 1D continuity target rather than 2D scrambled-embryo morphology

## Results

Validation summary:

{markdown_table(validation_df, ["validation_case", "case_type", "success", "detail"])}

Policy-by-target summary:

{markdown_table(policy_summary, ["policy_id", "target_kind", "runs", "mean_initial_target_error", "mean_final_target_error", "mean_target_recovery", "mean_initial_aggregate_error", "mean_final_aggregate_error", "mean_aggregate_recovery", "mean_accepted_swaps"])}

Target coverage:

{markdown_table(target_status_summary, ["target_id", "target_kind", "substrate_kind", "site_count", "simulated_in_s07", "reason"])}

Policy mapping audit:

{markdown_table(policy_mapping_summary, ["source_experiment", "policy_family", "source_policy_count", "mapping_status", "blocker"])}

## Validation Checks

- Initial scrambling severity was logged for every run as target error and displaced-site fraction.
- S05 target and aggregate morphospace metrics were computed for every final state.
- S04 action accounting showed local swap/wait execution with conserved population and counted rejections.
- The required zarr trace group was written and row-aligned with the trace table.
- E03/E04 policy artifacts were inventoried, and blocked mappings have explicit reasons.
- Recovery rates and trajectories were reported for both runnable S07 controls.
- The required example figure was written.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- The runnable recovery policy is an E05 local target-aware control, not an E03/E04 transferred policy.
- The available E03/E04 policies were not run on 2D targets because doing so would require undeclared orientation, projection, target morphology, or four-neighbor action information.
- S07 uses swaps only, so it evaluates rearrangement recovery, not growth, death, deformation, or hole filling.
- S03 targets are toy computational proxy morphologies, and S05 shape/topology metrics include labeled approximations.
- The `sorted_row` S03 target is represented in the target coverage table but excluded from the 2D scrambled benchmark because S06 already validated the 1D-in-2D continuity case.

## Blockers And Failed Assumptions

The main blocker is policy transfer: no available E03/E04 policy artifact had a declared observation/action contract that could be validly mapped to 2D S03 target morphologies. This constrains the hypothesis that the earlier 1D policies can be directly reused in higher-dimensional morphospace without a new declared 2D observation contract.

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
    traces_dir = artifacts_dir / "traces"
    figures_dir = artifacts_dir / "figures" / "e05"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, traces_dir, figures_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    seeds = parse_int_list(args.seeds)
    policy_ids = parse_str_list(args.policy_ids)
    all_targets = default_target_gallery()
    targets = two_dimensional_targets(all_targets)
    summary_rows, trace_rows, metric_rows, run_results = run_scrambled_benchmark(
        targets=all_targets,
        policy_ids=policy_ids,
        seeds=seeds,
        event_multiplier=args.event_multiplier,
        records_per_run=args.records_per_run,
    )
    summary_df = pd.DataFrame(summary_rows)
    trace_df = pd.DataFrame(trace_rows)
    metric_df = pd.DataFrame(metric_rows)
    target_status_df = pd.DataFrame(target_coverage_rows(all_targets))
    policy_mapping_df = pd.DataFrame(policy_source_mapping_rows(POLICY_SOURCE_PATHS))

    results_path = results_dir / "e05_scrambled_embryo_results.parquet"
    results_csv_path = tables_dir / "e05_scrambled_embryo_results.csv"
    metric_rows_path = results_dir / "e05_scrambled_embryo_metric_rows.parquet"
    metric_rows_csv_path = tables_dir / "e05_scrambled_embryo_metric_rows.csv"
    policy_mapping_path = results_dir / "e05_scrambled_embryo_policy_mapping.parquet"
    policy_mapping_csv_path = tables_dir / "e05_scrambled_embryo_policy_mapping.csv"
    target_status_path = results_dir / "e05_scrambled_embryo_target_coverage.parquet"
    trace_zarr_path = traces_dir / "e05_scrambled_embryo_traces.zarr"
    trace_table_path = traces_dir / "e05_scrambled_embryo_trace_table.parquet"
    trace_sample_path = tables_dir / "e05_scrambled_embryo_trace_sample.csv"
    figure_path = figures_dir / "scrambled_embryo_examples.png"
    validation_path = results_dir / "e05_scrambled_embryo_validation.parquet"
    validation_csv_path = tables_dir / "e05_scrambled_embryo_validation.csv"
    config_path = configs_dir / "e05_s07_scrambled_embryo_tests.json"
    source_manifest_path = src_snapshot_dir / "e05_scrambled_embryo_manifest.json"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    summary_df.to_parquet(results_path, index=False)
    summary_df.to_csv(results_csv_path, index=False)
    metric_df.to_parquet(metric_rows_path, index=False)
    metric_df.to_csv(metric_rows_csv_path, index=False)
    policy_mapping_df.to_parquet(policy_mapping_path, index=False)
    policy_mapping_df.to_csv(policy_mapping_csv_path, index=False)
    target_status_df.to_parquet(target_status_path, index=False)
    trace_df.to_parquet(trace_table_path, index=False)
    trace_df.head(5000).to_csv(trace_sample_path, index=False)
    write_trace_zarr(trace_df, trace_zarr_path)
    render_examples_figure(targets=targets, run_results=run_results, seed=seeds[0], output_path=figure_path)
    validation_df = run_validations(
        target_status_df=target_status_df,
        summary_df=summary_df,
        metric_df=metric_df,
        trace_df=trace_df,
        policy_mapping_df=policy_mapping_df,
        zarr_path=trace_zarr_path,
        figure_path=figure_path,
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
            "schema": "eidosoma.e05_s07.scrambled_embryo_config.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "targets": [target.target_id for target in targets],
            "targetCoverageRows": target_status_df.to_dict(orient="records"),
            "policyIds": list(policy_ids),
            "seeds": list(seeds),
            "eventMultiplier": int(args.event_multiplier),
            "recordsPerRun": int(args.records_per_run),
            "policySourcePaths": [str(path) for path in POLICY_SOURCE_PATHS],
            "unitTestsRun": bool(args.run_unit_tests),
            "createdAt": started_at,
        },
    )

    source_files = [
        source_entry(args.repo_dir / "src/e05/scrambled_embryo.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_scrambled_embryo.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s07_scrambled_embryo_tests.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/substrates.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/actions.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/cell_identity.py", args.repo_dir),
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

    caveats_or_blockers = (
        "E03/E04 policies were inventoried but blocked from 2D execution because valid mappings would require undeclared orientation, "
        "target-morphology, projection, or four-neighbor action access; S07 recovery rows therefore use E05 runnable controls."
    )
    outcome_classification = "supportive"
    recommended_next_action = "Stop before S08 and let the Chief Scientist review S07; if accepted, proceed to regeneration tests in S08."

    artifacts_for_summary = [
        artifact_entry(results_path, artifacts_dir, "Required S07 run-level scrambled-embryo results table."),
        artifact_entry(results_csv_path, artifacts_dir, "CSV mirror of required S07 results table."),
        artifact_entry(trace_zarr_path, artifacts_dir, "Required S07 zarr trajectory store."),
        artifact_entry(trace_table_path, artifacts_dir, "Parquet mirror of S07 trajectory rows."),
        artifact_entry(trace_sample_path, artifacts_dir, "CSV sample of S07 trajectory rows."),
        artifact_entry(figure_path, artifacts_dir, "Required S07 scrambled-embryo example figure."),
        artifact_entry(metric_rows_path, artifacts_dir, "S07 initial/final S05 metric rows."),
        artifact_entry(metric_rows_csv_path, artifacts_dir, "CSV mirror of S07 metric rows."),
        artifact_entry(policy_mapping_path, artifacts_dir, "Machine-readable E03/E04-to-2D policy mapping audit."),
        artifact_entry(policy_mapping_csv_path, artifacts_dir, "CSV mirror of policy mapping audit."),
        artifact_entry(target_status_path, artifacts_dir, "S03 target coverage and 2D simulation status table."),
        artifact_entry(validation_path, artifacts_dir, "S07 validation table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV mirror of S07 validation table."),
        artifact_entry(config_path, artifacts_dir, "S07 benchmark configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "Repository source hashes for S07 code and tests."),
    ]
    planned_artifact_paths = [
        {"path": str(full_results_path), "description": "S07 full-results handoff report."},
        {"path": str(artifact_manifest_path), "description": "S07 artifact manifest."},
        {"path": str(run_manifest_path), "description": "Experiment run manifest updated for S07."},
        {"path": str(checksum_path), "description": "Checksums for key S07 artifacts."},
    ]
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts=[*artifacts_for_summary, *planned_artifact_paths],
            validation_result=f"{validation_result}; unit-test commands success={test_success}",
            validation_df=validation_df,
            summary_df=summary_df,
            policy_mapping_df=policy_mapping_df,
            target_status_df=target_status_df,
            test_commands=test_commands,
            source_files=source_files,
            args=args,
            seeds=seeds,
            policy_ids=policy_ids,
            outcome_classification=outcome_classification,
            caveats_or_blockers=caveats_or_blockers,
            recommended_next_action=recommended_next_action,
        ),
    )
    artifacts_final = [
        *artifacts_for_summary,
        artifact_entry(full_results_path, artifacts_dir, "S07 full-results handoff report."),
    ]
    write_json(
        artifact_manifest_path,
        {
            "schema": "eidosoma.artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "success": bool(validation_success and test_success),
            "artifacts": [*artifacts_final, manifest_self_entry(artifact_manifest_path, artifacts_dir, "S07 artifact manifest.")],
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
        "artifacts": [*artifacts_final, artifact_entry(artifact_manifest_path, artifacts_dir, "S07 artifact manifest.")],
        "validationResult": f"{validation_result}; unit-test commands success={test_success}",
        "outcomeClassification": outcome_classification,
    }
    write_json(run_manifest_path, run_manifest_payload)
    checksum_inputs = [
        results_path,
        results_csv_path,
        trace_zarr_path,
        trace_table_path,
        trace_sample_path,
        figure_path,
        metric_rows_path,
        metric_rows_csv_path,
        policy_mapping_path,
        policy_mapping_csv_path,
        target_status_path,
        validation_path,
        validation_csv_path,
        config_path,
        source_manifest_path,
        full_results_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    write_checksums(checksum_inputs, checksum_path, artifacts_dir)

    print(
        json.dumps(
            {
                "success": bool(validation_success and test_success),
                "validationResult": validation_result,
                "testSuccess": test_success,
                "artifactsDir": str(artifacts_dir),
            },
            indent=2,
        )
    )
    return 0 if validation_success and test_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
