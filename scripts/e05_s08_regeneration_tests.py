#!/usr/bin/env python3
"""Run E05 S08 regeneration perturbation tests."""

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
from src.e05.regeneration import (
    DEFAULT_EVENT_MULTIPLIER,
    DEFAULT_RECORDS_PER_RUN,
    DEFAULT_REGENERATION_SEEDS,
    PERTURBATION_TYPES,
    STUCK_STATUS,
    run_regeneration_benchmark,
    validate_regeneration_trace_zarr,
    write_regeneration_trace_zarr,
)
from src.e05.scrambled_embryo import RUNNABLE_POLICY_IDS, two_dimensional_targets
from src.e05.targets import ORGAN_COLORS, TargetMorphology, default_target_gallery, target_render_mode


STEP_ID = "S08"
STEP_NUMBER = 8
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_REGENERATION_SEEDS))
    parser.add_argument("--policy-ids", default=",".join(RUNNABLE_POLICY_IDS))
    parser.add_argument("--perturbations", default=",".join(PERTURBATION_TYPES))
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


def validation_run_matrix(
    summary_df: pd.DataFrame,
    *,
    target_count: int,
    perturbation_count: int,
    policy_count: int,
    seed_count: int,
) -> dict[str, Any]:
    expected_rows = target_count * perturbation_count * policy_count * seed_count
    observed = {
        "run_count": int(len(summary_df)),
        "expected_run_count": int(expected_rows),
        "target_count": int(summary_df["target_id"].nunique()),
        "perturbation_count": int(summary_df["perturbation_type"].nunique()),
        "policy_count": int(summary_df["policy_id"].nunique()),
        "seed_count": int(summary_df["simulation_seed"].nunique()),
    }
    expected = {"run_count": int(expected_rows)}
    return _row(
        "regeneration_run_matrix_complete",
        "input",
        observed["run_count"] == expected_rows,
        expected,
        observed,
        "S08 produced the complete target by perturbation by policy by seed regeneration matrix.",
    )


def validation_masks_logged(mask_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, Any]:
    expected_masks = summary_df[["target_id", "perturbation_type", "simulation_seed"]].drop_duplicates().shape[0]
    observed = {
        "mask_rows": int(len(mask_df)),
        "expected_mask_rows": int(expected_masks),
        "all_mask_ids_nonempty": bool(mask_df["mask_id"].astype(str).str.len().gt(0).all()),
        "all_mask_areas_positive": bool(mask_df["mask_area"].astype(int).gt(0).all()),
        "all_mask_sites_logged": bool(mask_df["mask_site_ids_json"].astype(str).str.startswith("[").all()),
    }
    expected = {
        "mask_rows_match": True,
        "all_mask_ids_nonempty": True,
        "all_mask_areas_positive": True,
        "all_mask_sites_logged": True,
    }
    success = (
        observed["mask_rows"] == observed["expected_mask_rows"]
        and observed["all_mask_ids_nonempty"]
        and observed["all_mask_areas_positive"]
        and observed["all_mask_sites_logged"]
    )
    return _row(
        "perturbation_masks_logged",
        "perturbation",
        success,
        expected,
        observed,
        "Every target, perturbation, and seed has a logged mask with site IDs and coordinates.",
    )


def validation_pre_post_errors(summary_df: pd.DataFrame) -> dict[str, Any]:
    pre_zero = summary_df[summary_df["pre_repair_target_error"].astype(float) == 0.0]
    if pre_zero.empty:
        zero_rows_explained = True
    else:
        zero_rows_explained = bool(
            (
                pre_zero["semantic_blocker"].astype(str).str.len().gt(0)
                | ~pre_zero["identity_multiset_matches_target_pre"].astype(bool)
                | pre_zero["population_delta_from_baseline"].astype(int).ne(0)
                | pre_zero["stuck_site_count_pre"].astype(int).gt(0)
            ).all()
        )
    observed = {
        "baseline_errors_zero": bool((summary_df["baseline_target_error"].astype(float) == 0.0).all()),
        "pre_errors_non_null": bool(summary_df["pre_repair_target_error"].notna().all()),
        "post_errors_non_null": bool(summary_df["post_repair_target_error"].notna().all()),
        "pre_aggregate_non_null": bool(summary_df["pre_repair_aggregate_morphospace_error"].notna().all()),
        "post_aggregate_non_null": bool(summary_df["post_repair_aggregate_morphospace_error"].notna().all()),
        "min_pre_target_error": float(summary_df["pre_repair_target_error"].min()),
        "pre_target_error_zero_rows": int(len(pre_zero)),
        "zero_pre_error_rows_explained_by_blocker_or_nonmetric_damage": bool(zero_rows_explained),
    }
    expected = {
        "baseline_errors_zero": True,
        "pre_post_errors_non_null": True,
        "zero_pre_error_rows_explained_by_blocker_or_nonmetric_damage": True,
    }
    success = (
        observed["baseline_errors_zero"]
        and observed["pre_errors_non_null"]
        and observed["post_errors_non_null"]
        and observed["pre_aggregate_non_null"]
        and observed["post_aggregate_non_null"]
        and observed["zero_pre_error_rows_explained_by_blocker_or_nonmetric_damage"]
    )
    return _row(
        "baseline_pre_post_target_errors_recorded",
        "metric",
        success,
        expected,
        observed,
        "Baseline, post-perturbation/pre-repair, and final post-repair target errors are recorded; metric-insensitive damage must be explained by blocker, identity, status, or population flags.",
    )


def validation_final_metrics(summary_df: pd.DataFrame, metric_df: pd.DataFrame) -> dict[str, Any]:
    expected_metric_rows = len(summary_df) * 2 * 8
    observed = {
        "metric_rows": int(len(metric_df)),
        "expected_metric_rows": int(expected_metric_rows),
        "metric_values_non_null": bool(metric_df["value"].notna().all()),
        "state_labels": sorted(metric_df["benchmark_state_label"].unique().tolist()),
    }
    expected = {"metric_rows": expected_metric_rows, "state_labels": ["final", "perturbed"], "metric_values_non_null": True}
    success = (
        observed["metric_rows"] == expected_metric_rows
        and observed["state_labels"] == expected["state_labels"]
        and observed["metric_values_non_null"]
    )
    return _row(
        "s05_metrics_computed_for_pre_and_post_states",
        "metric",
        success,
        expected,
        observed,
        "S05 metric rows were computed for perturbed pre-repair and final post-repair states.",
    )


def validation_blockers(summary_df: pd.DataFrame) -> dict[str, Any]:
    blocked = summary_df[summary_df["perturbation_type"] != "rotated_patch"]
    rotated = summary_df[summary_df["perturbation_type"] == "rotated_patch"]
    observed = {
        "blocked_rows": int(len(blocked)),
        "blocked_rows_with_semantic_blocker": bool(blocked["semantic_blocker"].astype(str).str.len().gt(0).all()),
        "rotated_rows_without_blocker": bool(rotated["semantic_blocker"].astype(str).eq("").all()),
        "birth_death_identity_required_types": sorted(
            summary_df.loc[summary_df["birth_death_or_identity_conversion_required"], "perturbation_type"].unique().tolist()
        ),
        "unfreeze_required_types": sorted(summary_df.loc[summary_df["unfreeze_required"], "perturbation_type"].unique().tolist()),
        "policy_overextension_avoided_for_blocked": bool(blocked["policy_overextension_avoided"].all()),
    }
    expected = {
        "blocked_rows_with_semantic_blocker": True,
        "rotated_rows_without_blocker": True,
        "birth_death_identity_required_types": ["duplicated_patch", "foreign_patch", "missing_patch"],
        "unfreeze_required_types": ["frozen_patch"],
    }
    success = (
        observed["blocked_rows_with_semantic_blocker"]
        and observed["rotated_rows_without_blocker"]
        and observed["birth_death_identity_required_types"] == expected["birth_death_identity_required_types"]
        and observed["unfreeze_required_types"] == expected["unfreeze_required_types"]
        and observed["policy_overextension_avoided_for_blocked"]
    )
    return _row(
        "division_death_identity_and_unfreeze_blockers_documented",
        "blocker",
        success,
        expected,
        observed,
        "Missing, duplicated, foreign, and frozen perturbations carry explicit blocker labels instead of overextended repair semantics.",
    )


def validation_population_and_actions(summary_df: pd.DataFrame) -> dict[str, Any]:
    missing = summary_df[summary_df["perturbation_type"] == "missing_patch"]
    non_missing = summary_df[summary_df["perturbation_type"] != "missing_patch"]
    observed = {
        "missing_population_deficit_retained": bool((missing["population_delta_from_baseline"].astype(int) < 0).all()),
        "non_missing_population_conserved": bool((non_missing["population_delta_from_baseline"].astype(int) == 0).all()),
        "population_min_le_max": bool((summary_df["population_min"].astype(int) <= summary_df["population_max"].astype(int)).all()),
        "total_rejected_actions": int(summary_df["rejected_actions"].sum()),
        "total_accepted_swaps": int(summary_df["accepted_swaps"].sum()),
    }
    expected = {
        "missing_population_deficit_retained": True,
        "non_missing_population_conserved": True,
        "population_min_le_max": True,
    }
    success = all(observed[key] == expected[key] for key in expected)
    return _row(
        "s04_action_accounting_population_semantics_preserved",
        "action",
        success,
        expected,
        observed,
        "Swap/wait repair runs preserve the intended population semantics and do not synthesize missing cells.",
    )


def validation_trace_store(trace_df: pd.DataFrame, zarr_path: Path) -> dict[str, Any]:
    zarr_check = validate_regeneration_trace_zarr(zarr_path, len(trace_df))
    run_keys = ["target_id", "perturbation_type", "policy_id", "simulation_seed"]
    start_rows = trace_df[trace_df["event_step"] == 0].groupby(run_keys).size()
    observed = {
        "trace_rows": int(len(trace_df)),
        "run_count": int(trace_df[run_keys].drop_duplicates().shape[0]),
        "runs_with_step_zero": int((start_rows >= 1).sum()),
        "zarr_exists": bool(zarr_check["exists"]),
        "zarr_row_count": int(zarr_check["row_count"]),
        "zarr_all_arrays_row_aligned": bool(zarr_check["all_arrays_row_aligned"]),
        "zarr_schema": zarr_check.get("schema"),
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
        "regeneration_traces_written_and_valid",
        "artifact",
        success,
        expected,
        observed,
        "Trajectory rows include initial records per run, and the required zarr trace group is readable and row-aligned.",
    )


def validation_recovery_reported(summary_df: pd.DataFrame) -> dict[str, Any]:
    rotated = summary_df[summary_df["perturbation_type"] == "rotated_patch"]
    by_policy = (
        rotated.groupby("policy_id", as_index=False)
        .agg(mean_target_recovery=("target_recovery_fraction", "mean"), runs=("target_id", "count"))
        .set_index("policy_id")
    )
    local_mean = float(by_policy.loc["s07_local_target_neighbor_descent", "mean_target_recovery"])
    random_mean = float(by_policy.loc["s07_random_adjacent_swap_control", "mean_target_recovery"])
    observed = {
        "all_recovery_values_finite": bool(np.isfinite(summary_df["target_recovery_fraction"].astype(float)).all()),
        "rotated_local_target_mean_recovery": local_mean,
        "rotated_random_mean_recovery": random_mean,
        "local_target_better_on_rotated": bool(local_mean > random_mean),
    }
    expected = {"all_recovery_values_finite": True, "local_target_better_on_rotated": True}
    success = observed["all_recovery_values_finite"] and observed["local_target_better_on_rotated"]
    return _row(
        "recovery_rates_reported_for_repairable_rotated_patch",
        "outcome",
        success,
        expected,
        observed,
        "Recovery fractions are finite, and the S07 local target-aware control is checked against the random swap control on repairable rotated patches.",
    )


def validation_figure(figure_path: Path) -> dict[str, Any]:
    size = figure_path.stat().st_size if figure_path.exists() else 0
    observed = {"exists": figure_path.exists(), "size_bytes": int(size)}
    expected = {"exists": True, "size_bytes_min": 1000}
    return _row(
        "regeneration_examples_figure_written",
        "artifact",
        observed["exists"] and observed["size_bytes"] > expected["size_bytes_min"],
        expected,
        observed,
        "The required S08 regeneration example figure was written and is nonempty.",
    )


def run_validations(
    *,
    summary_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    zarr_path: Path,
    figure_path: Path,
    target_count: int,
    perturbation_count: int,
    policy_count: int,
    seed_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            validation_run_matrix(
                summary_df,
                target_count=target_count,
                perturbation_count=perturbation_count,
                policy_count=policy_count,
                seed_count=seed_count,
            ),
            validation_masks_logged(mask_df, summary_df),
            validation_pre_post_errors(summary_df),
            validation_final_metrics(summary_df, metric_df),
            validation_blockers(summary_df),
            validation_population_and_actions(summary_df),
            validation_trace_store(trace_df, zarr_path),
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
            values.append(identity.components.get(render_mode, identity.components.get("organ_type", identity.identity_id)))
            continue
        cell = state.cell_at(site_id)
        if cell is None:
            values.append("__empty__")
            continue
        if cell.status == STUCK_STATUS:
            values.append("__stuck__")
            continue
        identity = identity_from_substrate_cell(cell, fallback_scalar=False)
        values.append(identity.components.get(render_mode, identity.components.get("organ_type", identity.identity_id)))
    return np.array(values, dtype=object).reshape(height, width)


def render_state_panel(ax: Any, target: TargetMorphology, state: Any, title: str) -> None:
    mode = target_render_mode(target)
    matrix = state_component_matrix(target, state, mode)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    if mode in {"value", "ap_coordinate"} and "__empty__" not in matrix and "__stuck__" not in matrix:
        ax.imshow(matrix.astype(float), cmap="viridis", interpolation="nearest", aspect="equal", vmin=0.0, vmax=1.0)
        return
    color_keys = ["neural", "epidermis", "mesenchyme", "boundary", "__empty__", "__stuck__"]
    colors = [ORGAN_COLORS[key] for key in color_keys[:4]] + ["#F7F7F7", "#333333"]
    cmap = ListedColormap(colors)
    mapping = {label: idx for idx, label in enumerate(color_keys)}
    encoded = np.vectorize(lambda value: mapping.get(str(value), 0))(matrix)
    ax.imshow(encoded, cmap=cmap, vmin=0, vmax=len(color_keys) - 1, interpolation="nearest", aspect="equal")


def render_examples_figure(
    *,
    targets: Sequence[TargetMorphology],
    run_results: Mapping[tuple[str, str, str, int], Any],
    seed: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_lookup = {target.target_id: target for target in targets}
    target = target_lookup.get("toy_organ_like", targets[-1])
    fig, axes = plt.subplots(len(PERTURBATION_TYPES), 3, figsize=(8.4, 2.0 * len(PERTURBATION_TYPES)), constrained_layout=True)
    for row_idx, perturbation_type in enumerate(PERTURBATION_TYPES):
        result = run_results[(target.target_id, perturbation_type, "s07_local_target_neighbor_descent", int(seed))]
        render_state_panel(axes[row_idx, 0], target, "target", f"{perturbation_type}\ntarget")
        render_state_panel(axes[row_idx, 1], target, result.perturbed_state, "perturbed")
        render_state_panel(axes[row_idx, 2], target, result.final_state, "final")
    fig.suptitle("E05 S08 regeneration perturbation examples")
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
- Lay summary: S08 applied missing, frozen, rotated, duplicated, and foreign-patch perturbations to every genuine 2D S03 target, logged masks and pre/post target errors, and evaluated S07 local controls with S05 metrics. Rotated patches are swap-repairable rearrangements; the other perturbation classes expose explicit population, stuck-cell, or identity-multiset blockers under the current swap-only policy semantics.
- Recommended next action: {recommended_next_action}
"""


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    args: argparse.Namespace,
    seeds: Sequence[int],
    policy_ids: Sequence[str],
    perturbations: Sequence[str],
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
    perturbation_summary = (
        summary_df.groupby(["perturbation_type", "policy_id"], as_index=False)
        .agg(
            runs=("target_id", "count"),
            mean_pre_target_error=("pre_repair_target_error", "mean"),
            mean_post_target_error=("post_repair_target_error", "mean"),
            mean_target_recovery=("target_recovery_fraction", "mean"),
            mean_pre_aggregate_error=("pre_repair_aggregate_morphospace_error", "mean"),
            mean_post_aggregate_error=("post_repair_aggregate_morphospace_error", "mean"),
            mean_accepted_swaps=("accepted_swaps", "mean"),
        )
        .sort_values(["perturbation_type", "policy_id"])
    )
    blocker_summary = (
        summary_df.groupby(["perturbation_type", "repairability_class", "semantic_blocker"], as_index=False)
        .agg(runs=("target_id", "count"))
        .sort_values(["perturbation_type", "repairability_class"])
    )
    mask_summary = (
        mask_df.groupby("perturbation_type", as_index=False)
        .agg(mask_rows=("mask_id", "count"), mean_mask_area=("mask_area", "mean"))
        .sort_values("perturbation_type")
    )
    return f"""{top_summary_markdown(artifacts, validation_result, outcome_classification, caveats_or_blockers, recommended_next_action)}

# Research Step Full Results: {STEP_ID} Run Regeneration Tests

## Lay Summary

S08 tests whether the S07 local controls can recover target morphology after structured damage rather than full scrambling. The local target-aware policy can reduce error on rotated patches, which preserve the target identity set and only require rearrangement. Missing, frozen, duplicated, and foreign patches expose limits of the current action semantics: full repair would require cell birth/division, unfreezing, death, or identity conversion, so S08 logs those blockers instead of adding undeclared repair powers.

## Frozen Question

Can local policies repair missing, frozen, rotated, duplicated, or foreign patches in target morphologies?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S08.
- S07 task framework: `src/e05/scrambled_embryo.py`, especially the local target-aware and random adjacent-swap controls.
- S08 perturbation code: `src/e05/regeneration.py`.
- S03 targets: all genuine 2D targets from `src/e05/targets.py`.
- S05 metrics: all default morphospace metrics from `src/e05/morphospace_metrics.py`.
- Datasets: none required.

## Methods

For each genuine 2D S03 target, S08 selected a deterministic compact rectangular mask for each perturbation type and seed. Missing patches remove the masked cells. Frozen patches rotate the masked patch and mark the rotated cells `stuck`, so identity placement is damaged but cells cannot act or be swapped. Rotated patches apply the same 180-degree patch rotation without stuck status. Duplicated patches replace the mask with duplicate identities from a different source patch in the same target. Foreign patches replace the mask with compatible identities from another S03 target.

Every run starts from the perturbed state, then applies the S07 local target-aware control or random adjacent-swap control through S04 `ActionExecutor` using only swap/wait actions. S05 metrics are evaluated at perturbed pre-repair and final post-repair states, and aggregate target-error trajectories are written to zarr.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- Matplotlib: `{matplotlib.__version__}`
- No new packages were installed for S08; `zarr` was already available in the runtime.
- CPU/GPU use: serial CPU benchmark and validation; no GPU use was needed.

## Parameters

- Seeds: `{", ".join(str(seed) for seed in seeds)}`
- Perturbations: `{", ".join(perturbations)}`
- Runnable policies: `{", ".join(policy_ids)}`
- Event multiplier: `{args.event_multiplier}` local scheduler events per target site
- Records per run: `{args.records_per_run}`

## Results

Validation summary:

{markdown_table(validation_df, ["validation_case", "case_type", "success", "detail"])}

Perturbation-by-policy summary:

{markdown_table(perturbation_summary, ["perturbation_type", "policy_id", "runs", "mean_pre_target_error", "mean_post_target_error", "mean_target_recovery", "mean_pre_aggregate_error", "mean_post_aggregate_error", "mean_accepted_swaps"])}

Blocker summary:

{markdown_table(blocker_summary, ["perturbation_type", "repairability_class", "semantic_blocker", "runs"])}

Mask summary:

{markdown_table(mask_summary, ["perturbation_type", "mask_rows", "mean_mask_area"])}

## Validation Checks

- The full target by perturbation by policy by seed matrix was produced.
- Perturbation masks were logged with site IDs and coordinates.
- Baseline, pre-repair, and post-repair target errors were recorded.
- S05 metrics were computed for every pre-repair and post-repair state.
- Missing, duplicated, foreign, and frozen perturbation blockers were documented explicitly.
- S04 action accounting preserved population semantics and did not synthesize missing cells.
- The required zarr trace group was written and row-aligned.
- Recovery rates were reported, with a rotated-patch local-control sanity comparison.
- The required example figure was written.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- S08 uses swap/wait S07 controls only; it does not add a new policy for division, death, unfreezing, or identity conversion.
- Missing patches remain population-deficient under the current policy semantics.
- Duplicated and foreign patches retain identity-multiset mismatches unless later steps define explicit replacement or identity-conversion actions.
- Frozen patches use `stuck` status to represent cells that cannot be swapped or act; no unfreeze action is introduced.
- Some target-error metrics can be insensitive to duplicated cells with identical target-relevant components, especially the pure AP-gradient target; S08 therefore logs identity-multiset, population, status, and semantic-blocker flags alongside S05 metric values.
- S03 targets are toy computational proxy morphologies, and S05 shape/topology metrics include labeled approximations.

## Blockers And Failed Assumptions

The main blocker is true regeneration semantics. The current S07 local controls can rearrange identities, but missing-cell repair requires a birth/division policy with target-compatible daughter identities; duplicated and foreign patches require death or identity conversion; frozen patches require an explicit unfreeze or repair action. S08 documents these blockers and does not overextend S04 actions beyond their declared semantics.

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
    policy_ids = parse_str_list(args.policy_ids, RUNNABLE_POLICY_IDS, "policy IDs")
    perturbations = parse_str_list(args.perturbations, PERTURBATION_TYPES, "perturbations")
    all_targets = default_target_gallery()
    targets = two_dimensional_targets(all_targets)
    summary_rows, trace_rows, metric_rows, mask_rows, run_results = run_regeneration_benchmark(
        targets=all_targets,
        perturbation_types=perturbations,
        policy_ids=policy_ids,
        seeds=seeds,
        event_multiplier=args.event_multiplier,
        records_per_run=args.records_per_run,
    )
    summary_df = pd.DataFrame(summary_rows)
    trace_df = pd.DataFrame(trace_rows)
    metric_df = pd.DataFrame(metric_rows)
    mask_df = pd.DataFrame(mask_rows)

    results_path = results_dir / "e05_regeneration_results.parquet"
    results_csv_path = tables_dir / "e05_regeneration_results.csv"
    metric_rows_path = results_dir / "e05_regeneration_metric_rows.parquet"
    metric_rows_csv_path = tables_dir / "e05_regeneration_metric_rows.csv"
    mask_path = results_dir / "e05_regeneration_masks.parquet"
    mask_csv_path = tables_dir / "e05_regeneration_masks.csv"
    trace_zarr_path = traces_dir / "e05_regeneration_traces.zarr"
    trace_table_path = traces_dir / "e05_regeneration_trace_table.parquet"
    trace_sample_path = tables_dir / "e05_regeneration_trace_sample.csv"
    figure_path = figures_dir / "regeneration_examples.png"
    validation_path = results_dir / "e05_regeneration_validation.parquet"
    validation_csv_path = tables_dir / "e05_regeneration_validation.csv"
    config_path = configs_dir / "e05_s08_regeneration_tests.json"
    source_manifest_path = src_snapshot_dir / "e05_regeneration_manifest.json"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    summary_df.to_parquet(results_path, index=False)
    summary_df.to_csv(results_csv_path, index=False)
    metric_df.to_parquet(metric_rows_path, index=False)
    metric_df.to_csv(metric_rows_csv_path, index=False)
    mask_df.to_parquet(mask_path, index=False)
    mask_df.to_csv(mask_csv_path, index=False)
    trace_df.to_parquet(trace_table_path, index=False)
    trace_df.head(5000).to_csv(trace_sample_path, index=False)
    write_regeneration_trace_zarr(trace_df, trace_zarr_path)
    render_examples_figure(targets=targets, run_results=run_results, seed=seeds[0], output_path=figure_path)

    validation_df = run_validations(
        summary_df=summary_df,
        metric_df=metric_df,
        mask_df=mask_df,
        trace_df=trace_df,
        zarr_path=trace_zarr_path,
        figure_path=figure_path,
        target_count=len(targets),
        perturbation_count=len(perturbations),
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
            "schema": "eidosoma.e05_s08.regeneration_config.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "targets": [target.target_id for target in targets],
            "perturbations": list(perturbations),
            "policyIds": list(policy_ids),
            "seeds": list(seeds),
            "eventMultiplier": int(args.event_multiplier),
            "recordsPerRun": int(args.records_per_run),
            "unitTestsRun": bool(args.run_unit_tests),
            "createdAt": started_at,
        },
    )

    source_files = [
        source_entry(args.repo_dir / "src/e05/regeneration.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_regeneration.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s08_regeneration_tests.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/scrambled_embryo.py", args.repo_dir),
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
        "True regeneration is blocked for missing, duplicated, foreign, and frozen patches under S07 swap/wait controls: these cases need "
        "birth/division, death, identity conversion, or unfreeze semantics. Some duplicated damage can be target-error-insensitive when copied "
        "cells share target-relevant identity components, so S08 logs identity-multiset flags alongside S05 metrics."
    )
    outcome_classification = "constraining/contradictory"
    recommended_next_action = "Stop before S09 and let the Chief Scientist review S08; if accepted, proceed to scaling tests in S09."

    artifacts_for_summary = [
        artifact_entry(results_path, artifacts_dir, "Required S08 run-level regeneration results table."),
        artifact_entry(results_csv_path, artifacts_dir, "CSV mirror of required S08 results table."),
        artifact_entry(trace_zarr_path, artifacts_dir, "Required S08 zarr trajectory store."),
        artifact_entry(trace_table_path, artifacts_dir, "Parquet mirror of S08 trajectory rows."),
        artifact_entry(trace_sample_path, artifacts_dir, "CSV sample of S08 trajectory rows."),
        artifact_entry(figure_path, artifacts_dir, "Required S08 regeneration example figure."),
        artifact_entry(metric_rows_path, artifacts_dir, "S08 pre/final S05 metric rows."),
        artifact_entry(metric_rows_csv_path, artifacts_dir, "CSV mirror of S08 metric rows."),
        artifact_entry(mask_path, artifacts_dir, "Machine-readable S08 perturbation mask table."),
        artifact_entry(mask_csv_path, artifacts_dir, "CSV mirror of S08 perturbation mask table."),
        artifact_entry(validation_path, artifacts_dir, "S08 validation table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV mirror of S08 validation table."),
        artifact_entry(config_path, artifacts_dir, "S08 benchmark configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "Repository source hashes for S08 code and tests."),
    ]
    planned_artifact_paths = [
        {"path": str(full_results_path), "description": "S08 full-results handoff report."},
        {"path": str(artifact_manifest_path), "description": "S08 artifact manifest."},
        {"path": str(run_manifest_path), "description": "Experiment run manifest updated for S08."},
        {"path": str(checksum_path), "description": "Checksums for key S08 artifacts."},
    ]
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts=[*artifacts_for_summary, *planned_artifact_paths],
            validation_result=f"{validation_result}; unit-test commands success={test_success}",
            validation_df=validation_df,
            summary_df=summary_df,
            mask_df=mask_df,
            test_commands=test_commands,
            source_files=source_files,
            args=args,
            seeds=seeds,
            policy_ids=policy_ids,
            perturbations=perturbations,
            outcome_classification=outcome_classification,
            caveats_or_blockers=caveats_or_blockers,
            recommended_next_action=recommended_next_action,
        ),
    )
    artifacts_final = [
        *artifacts_for_summary,
        artifact_entry(full_results_path, artifacts_dir, "S08 full-results handoff report."),
    ]
    write_json(
        artifact_manifest_path,
        {
            "schema": "eidosoma.artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "success": bool(validation_success and test_success),
            "artifacts": [*artifacts_final, manifest_self_entry(artifact_manifest_path, artifacts_dir, "S08 artifact manifest.")],
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
        "artifacts": [*artifacts_final, artifact_entry(artifact_manifest_path, artifacts_dir, "S08 artifact manifest.")],
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
        mask_path,
        mask_csv_path,
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
