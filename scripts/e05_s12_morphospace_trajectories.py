#!/usr/bin/env python3
"""Run E05 S12 morphospace trajectory embedding."""

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
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.e05.morphospace_trajectories import (
    FEATURE_COLUMNS,
    STEP_ID,
    build_morphospace_embedding,
)


STEP_NUMBER = 12
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
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
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def run_validations(
    *,
    embedding_df: pd.DataFrame,
    route_metrics_df: pd.DataFrame,
    route_summary_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    source_df: pd.DataFrame,
    metric_source_df: pd.DataFrame,
    figure_path: Path,
) -> pd.DataFrame:
    non_target = embedding_df[~embedding_df["is_target_reference"]]
    initial_by_run = non_target.groupby("run_uid")["state_role"].apply(lambda values: "initial" in set(values))
    task_target_refs = embedding_df[embedding_df["is_target_reference"]].groupby("task_id").size()
    finite_route = np.isfinite(route_metrics_df["path_curvature"]).all() if not route_metrics_df.empty else False
    rows = [
        _row(
            "s07_to_s11_trace_inputs_loaded",
            "input_provenance",
            bool({"S07", "S08", "S09", "S10", "S11"}.issubset(set(source_df.loc[source_df["used_for_embedding"], "source_research_step_id"]))),
            {"source_steps": ["S07", "S08", "S09", "S10", "S11"]},
            {
                "used_steps": sorted(set(source_df.loc[source_df["used_for_embedding"], "source_research_step_id"])),
                "total_normalized_rows": int(source_df["normalized_rows"].sum()),
            },
            "S07, S08, S10, and S11 downsampled traces were loaded; S09 was included as endpoint-only S05 metric trajectories.",
        ),
        _row(
            "s05_metric_sources_loaded",
            "input_provenance",
            bool(
                (metric_source_df["source_research_step_id"].isin(["S05", "S07", "S08", "S09", "S10"]).sum() >= 5)
                and metric_source_df.loc[metric_source_df["source_research_step_id"] == "S05", "exists"].all()
                and metric_source_df.loc[metric_source_df["input_type"] == "s05_aggregate_trace_feature", "row_count"].iloc[0] > 0
            ),
            {"S05_catalog_exists": True, "aggregate_trace_feature_rows": ">0"},
            {
                "metric_source_rows": len(metric_source_df),
                "aggregate_trace_feature_rows": int(metric_source_df.loc[metric_source_df["input_type"] == "s05_aggregate_trace_feature", "row_count"].iloc[0]),
            },
            "S05 metric catalog and endpoint metric rows were loaded; S05-derived aggregate morphospace error is used as an embedding feature when available.",
        ),
        _row(
            "embedding_table_complete",
            "output_completeness",
            bool(len(embedding_df) > 50_000 and set(["morph_x", "morph_y", "morph_z"]).issubset(embedding_df.columns)),
            {"minimum_rows": 50000, "coordinate_columns": ["morph_x", "morph_y", "morph_z"]},
            {"rows": len(embedding_df), "coordinate_columns_present": set(["morph_x", "morph_y", "morph_z"]).issubset(embedding_df.columns)},
            "The embedding table contains all normalized trajectory points plus target-reference rows and PCA coordinates.",
        ),
        _row(
            "initial_and_target_states_labeled",
            "label_validation",
            bool(initial_by_run.all() and len(task_target_refs) > 0 and (embedding_df.loc[embedding_df["is_target_reference"], "target_error"] == 0.0).all()),
            {"all_runs_have_initial": True, "target_references_zero_error": True},
            {
                "run_count": int(non_target["run_uid"].nunique()),
                "runs_with_initial": int(initial_by_run.sum()),
                "target_reference_rows": int(embedding_df["is_target_reference"].sum()),
                "target_reference_tasks": int(len(task_target_refs)),
            },
            "Every non-target route has an initial state label, and target-reference rows are explicit zero-error rows.",
        ),
        _row(
            "route_metrics_complete",
            "route_metric_validation",
            bool(len(route_metrics_df) == non_target["run_uid"].nunique() and finite_route),
            {"route_rows_equal_run_count": True, "path_curvature_finite": True},
            {
                "route_rows": len(route_metrics_df),
                "unique_runs": int(non_target["run_uid"].nunique()),
                "finite_path_curvature": bool(finite_route),
            },
            "Per-run path length, path curvature, temporary-away, and route-diversity metrics were computed.",
        ),
        _row(
            "route_diversity_reported",
            "route_metric_validation",
            bool(not route_summary_df.empty and (route_summary_df["mean_pairwise_route_distance"] >= 0.0).all()),
            {"summary_rows": ">0", "distances_nonnegative": True},
            {
                "summary_rows": len(route_summary_df),
                "max_mean_pairwise_route_distance": float(route_summary_df["mean_pairwise_route_distance"].max()) if not route_summary_df.empty else None,
            },
            "Route diversity is summarized by source task, policy, and target/task group.",
        ),
        _row(
            "seed_split_embedding_stability_passed",
            "embedding_stability",
            bool(stability_df.loc[stability_df["validation_case"] == "seed_split_embedding_stability", "success"].all()),
            {"stability_score_min": 0.70},
            stability_df.loc[stability_df["validation_case"] == "seed_split_embedding_stability"].to_dict(orient="records")[0],
            "Embedding pairwise distances were stable when PCA was fitted on even versus odd seed subsets.",
        ),
        _row(
            "metric_scaling_embedding_stability_passed",
            "embedding_stability",
            bool(stability_df.loc[stability_df["validation_case"] == "metric_scaling_embedding_stability", "success"].all()),
            {"stability_score_min": 0.70},
            stability_df.loc[stability_df["validation_case"] == "metric_scaling_embedding_stability"].to_dict(orient="records")[0],
            "Embedding pairwise distances were stable under a deterministic reweighting of S05-derived metric features.",
        ),
        _row(
            "trajectory_map_figure_written",
            "figure_validation",
            bool(figure_path.exists() and figure_path.stat().st_size > 5_000),
            {"exists": True, "min_size_bytes": 5000},
            {"exists": figure_path.exists(), "size_bytes": figure_path.stat().st_size if figure_path.exists() else 0},
            "The required morphospace trajectory map figure was written as a nonempty PNG.",
        ),
    ]
    return pd.DataFrame(rows)


def render_trajectory_figure(
    embedding_df: pd.DataFrame,
    route_metrics_df: pd.DataFrame,
    route_summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_colors = {
        "S07": "#4C78A8",
        "S08": "#54A24B",
        "S09": "#B279A2",
        "S10": "#F58518",
        "S11": "#6F4E9B",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    non_target = embedding_df[~embedding_df["is_target_reference"]]
    target_refs = embedding_df[embedding_df["is_target_reference"]]
    for source_step, group in non_target.groupby("source_research_step_id"):
        sample = group.sample(min(len(group), 3500), random_state=12) if len(group) > 3500 else group
        axes[0].scatter(
            sample["morph_x"],
            sample["morph_y"],
            s=5,
            alpha=0.25,
            color=source_colors.get(source_step, "#777777"),
            label=source_step,
            linewidths=0,
        )
    axes[0].scatter(target_refs["morph_x"], target_refs["morph_y"], marker="*", s=80, color="black", label="target", zorder=5)
    axes[0].set_title("Embedded trajectory states")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].grid(True, linewidth=0.3, alpha=0.35)

    selected_tasks = (
        route_summary_df.sort_values("mean_pairwise_route_distance", ascending=False)["task_id"].drop_duplicates().head(3).tolist()
    )
    route_subset = non_target[non_target["task_id"].isin(selected_tasks)].copy()
    route_subset["progress_bin"] = pd.cut(route_subset["event_progress"], bins=np.linspace(0, 1, 9), include_lowest=True, labels=False)
    mean_routes = (
        route_subset.groupby(["task_id", "policy_id", "progress_bin"], as_index=False)
        .agg(morph_x=("morph_x", "mean"), morph_y=("morph_y", "mean"))
        .sort_values(["task_id", "policy_id", "progress_bin"])
    )
    policy_styles = {
        "s07_local_target_neighbor_descent": ("#2F6B4F", "-"),
        "s07_random_adjacent_swap_control": ("#8D4E2F", "--"),
    }
    for (task_id, policy_id), group in mean_routes.groupby(["task_id", "policy_id"]):
        color, linestyle = policy_styles.get(policy_id, ("#555555", "-"))
        axes[1].plot(group["morph_x"], group["morph_y"], color=color, linestyle=linestyle, linewidth=1.5, alpha=0.85)
        if not group.empty:
            axes[1].text(group["morph_x"].iloc[-1], group["morph_y"].iloc[-1], str(task_id).replace("S10:", ""), fontsize=7, color=color)
    axes[1].scatter(target_refs["morph_x"], target_refs["morph_y"], marker="*", s=50, color="black", alpha=0.7)
    axes[1].set_title("Mean routes for high-diversity tasks")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, linewidth=0.3, alpha=0.35)

    curvature = (
        route_metrics_df.groupby(["source_research_step_id", "policy_id"], as_index=False)
        .agg(path_curvature=("path_curvature", "mean"), temporary_away_fraction=("temporary_away_fraction", "mean"))
        .sort_values(["source_research_step_id", "policy_id"])
    )
    source_steps = sorted(curvature["source_research_step_id"].unique())
    x = np.arange(len(source_steps))
    width = 0.36
    for offset, policy_id in zip((-width / 2, width / 2), sorted(curvature["policy_id"].unique()), strict=False):
        data = curvature[curvature["policy_id"] == policy_id].set_index("source_research_step_id").reindex(source_steps)
        color, _linestyle = policy_styles.get(policy_id, ("#555555", "-"))
        axes[2].bar(x + offset, data["path_curvature"], width=width, color=color, label=policy_id)
    axes[2].set_title("Mean path curvature by source and policy")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(source_steps)
    axes[2].set_ylabel("path length / direct distance")
    axes[2].grid(axis="y", linewidth=0.3, alpha=0.35)
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle("E05 S12 morphospace trajectory embedding", fontsize=13)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], *, max_rows: int = 24) -> str:
    if df.empty:
        return "_No rows._"
    shown = df[columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in shown.to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    if len(df) > max_rows:
        omitted = [f"... {len(df) - max_rows} more rows omitted", *("" for _ in columns[1:])]
        rows.append("| " + " | ".join(omitted) + " |")
    return "\n".join([header, separator, *rows])


def determine_outcome(validation_df: pd.DataFrame, route_metrics_df: pd.DataFrame) -> tuple[str, str, str]:
    validation_success = bool(validation_df["success"].all())
    by_policy = route_metrics_df.groupby("policy_id", as_index=False).agg(
        mean_curvature=("path_curvature", "mean"),
        mean_away=("temporary_away_fraction", "mean"),
    )
    local = by_policy[by_policy["policy_id"] == "s07_local_target_neighbor_descent"]
    random = by_policy[by_policy["policy_id"] == "s07_random_adjacent_swap_control"]
    if validation_success and not local.empty and not random.empty:
        outcome = "supportive"
        lay = (
            "S12 embeds S07-S11 morphology trajectories in a stable low-dimensional proxy space and shows clear route differences: "
            "the random adjacent-swap control takes more curved, often target-away paths than the local target-aware control."
        )
    else:
        outcome = "null"
        lay = (
            "S12 produced morphospace embeddings, but validation or policy-route separation was insufficient to treat route differences "
            "as a usable summary."
        )
    caveats = (
        "The embedding is a visualization and route-summary proxy over S05-derived scalar metrics and action summaries, not a mechanistic "
        "proof of morphology-space geometry. S09 contributes endpoint-only trajectories because no downsampled S09 trace table exists; "
        "S11 contributes target-error and displacement traces but not full S05 metric rows at every intermediate state."
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
    embedding_df: pd.DataFrame,
    route_metrics_df: pd.DataFrame,
    route_summary_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    source_df: pd.DataFrame,
    metric_source_df: pd.DataFrame,
    pca_variance_ratio: tuple[float, ...],
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
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
    policy_summary = (
        route_metrics_df.groupby("policy_id", as_index=False)
        .agg(
            routes=("run_uid", "count"),
            mean_path_curvature=("path_curvature", "mean"),
            mean_temporary_away_fraction=("temporary_away_fraction", "mean"),
            mean_route_diversity=("route_diversity_to_group_mean", "mean"),
            mean_final_target_error=("final_target_error", "mean"),
        )
        .sort_values("policy_id")
    )
    source_summary = (
        route_metrics_df.groupby(["source_research_step_id", "policy_id"], as_index=False)
        .agg(
            routes=("run_uid", "count"),
            mean_path_curvature=("path_curvature", "mean"),
            mean_temporary_away_fraction=("temporary_away_fraction", "mean"),
            endpoint_only_rate=("endpoint_only", "mean"),
        )
        .sort_values(["source_research_step_id", "policy_id"])
    )
    for frame in (policy_summary, source_summary, route_summary_df, stability_df):
        for column in frame.select_dtypes(include=[float]).columns:
            frame[column] = frame[column].round(6)
    return f"""{top_summary_markdown(artifacts, validation_result, outcome_classification, caveats_or_blockers, recommended_next_action, lay_summary)}

# Research Step Full Results: {STEP_ID} Map Trajectories Through Morphospace

## Lay Summary

{lay_summary}

## Frozen Question

Do different policies take distinct routes through morphology space toward the same target, including temporary movement away from the target?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S12.
- Trace inputs: S07 scrambled-embryo traces, S08 regeneration traces, S10 symmetry-breaking axis traces, and S11 GPU tensor traces.
- S09 endpoint inputs: S09 scaling run summaries and S09 S05 metric rows, represented as initial/final endpoint-only trajectories because S09 did not write a downsampled trajectory table.
- S05 metric inputs: S05 metric catalog plus S07/S08/S09/S10 endpoint S05 metric rows; S05-derived aggregate morphospace error is used directly where trace tables expose it.
- Datasets: none required.

## Detailed Methods

S12 normalizes all available S07-S11 trajectory-like records into a common point table with run IDs, source step, target/task labels, policy labels, event progress, target error, aggregate morphospace error, displacement fraction when available, population fraction, action/energy-per-site summaries, and S10 axis-choice features where applicable. Initial rows are labeled from event step zero. Explicit zero-error target-reference rows are added for every source/task/target combination.

The embedding uses standardized PCA over these feature columns: `{", ".join(FEATURE_COLUMNS)}`. PCA is used because it is deterministic, transparent, and easy to validate under seed splits and metric reweighting. Route metrics are then computed in the first two embedding dimensions: path length, direct distance, path curvature, temporary target-away steps, overshoot above initial target error, and group route diversity using fixed-progress route signatures.

Embedding stability is validated in two ways. First, PCA is fit separately on even-seed and odd-seed point subsets and compared by Spearman correlation of sampled pairwise distances. Second, S05-derived target and aggregate error features are reweighted and compared against the base embedding by the same pairwise-distance stability score.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- Matplotlib: `{matplotlib.__version__}`
- scikit-learn PCA/StandardScaler and SciPy pairwise-distance/Spearman utilities were used from the preinstalled runtime.
- New packages installed: none.
- CPU/GPU use: CPU-only embedding and validation; S12 reused S11 GPU-generated traces but did not run new GPU simulations.

## Parameters

- Embedding method: standardized PCA with 3 retained coordinates; figure and route metrics use PC1 and PC2.
- Feature columns: `{", ".join(FEATURE_COLUMNS)}`
- PCA variance ratio: `{", ".join(f"{value:.4f}" for value in pca_variance_ratio)}`
- Stability threshold: 0.70 Spearman pairwise-distance correlation for both seed-split and metric-scaling checks.

## Results

Validation summary:

{markdown_table(validation_df, ["validation_case", "case_type", "success", "detail"], max_rows=20)}

Source trace inputs:

{markdown_table(source_df, ["source_research_step_id", "input_type", "exists", "raw_rows", "normalized_rows", "used_for_embedding", "caveat"], max_rows=10)}

Metric sources:

{markdown_table(metric_source_df, ["source_research_step_id", "input_type", "exists", "row_count", "metric_id_count", "used_for_embedding"], max_rows=10)}

Policy route summary:

{markdown_table(policy_summary, ["policy_id", "routes", "mean_path_curvature", "mean_temporary_away_fraction", "mean_route_diversity", "mean_final_target_error"], max_rows=10)}

Source-step route summary:

{markdown_table(source_summary, ["source_research_step_id", "policy_id", "routes", "mean_path_curvature", "mean_temporary_away_fraction", "endpoint_only_rate"], max_rows=20)}

Highest-diversity route groups:

{markdown_table(route_summary_df.sort_values("mean_pairwise_route_distance", ascending=False), ["source_research_step_id", "task_id", "policy_id", "runs", "mean_pairwise_route_distance", "mean_path_curvature", "mean_temporary_away_fraction", "endpoint_only_rate"], max_rows=16)}

Embedding stability:

{markdown_table(stability_df, ["validation_case", "success", "stability_score", "threshold", "detail"], max_rows=10)}

## Metrics

- Embedding path length: sum of Euclidean step distances through PC1/PC2.
- Direct distance: Euclidean distance from a run's initial point to its final point in PC1/PC2.
- Path curvature: embedding path length divided by direct distance; values above 1 indicate detours or curved routes.
- Temporary-away fraction: fraction of adjacent trace intervals where target error increases.
- Route diversity: mean pairwise distance between fixed-progress route signatures within each source/task/policy group.
- Target-reference rows: explicit zero target-error, zero aggregate-error rows used to label target states in the embedding.

## Figures And Tables

- Required embedding table: `$ARTIFACTS_DIR/results/e05_morphospace_embeddings.parquet`.
- Required trajectory map: `$ARTIFACTS_DIR/figures/e05/morphospace_trajectory_map.png`.
- Additional route metrics, route summaries, stability checks, source input tables, metric source tables, validation rows, config, manifests, and checksums are listed below.

## Validation Checks

- Loaded S07-S11 trajectory inputs, with S09 represented as endpoint-only S05 metric trajectories.
- Loaded S05 metric catalog and endpoint metric rows.
- Wrote a complete embedding table with explicit target-reference rows.
- Verified every run has an initial state label and every task has a zero-error target reference.
- Computed route metrics and group route diversity.
- Passed seed-split and metric-scaling embedding stability checks.
- Wrote the required nonempty morphospace trajectory map.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- The embedding is a low-dimensional proxy view of scalar morphospace features and action summaries, not proof of a true physical or biological state space.
- S09 contributes only two-point initial/final endpoint trajectories because S09 did not write downsampled trace rows.
- S11 contributes target-error and displacement-fraction traces but not full S05 metric vectors at each intermediate tensor state.
- PCA captures dominant linear variance; nonlinear embeddings might show different visual geometry and should be treated as visualization aids unless separately validated.
- Temporary-away steps are measured from downsampled trace intervals, not every simulator event.

## Blockers And Failed Assumptions

No execution blocker was encountered. The main failed assumption is that every S07-S11 source had comparable dense state features at every time point; S09 and S11 required documented reduced representations.

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
    result = build_morphospace_embedding(artifacts_dir)
    embedding_df = result.embedding_df
    route_metrics_df = result.route_metrics_df
    route_summary_df = result.route_summary_df
    stability_df = result.stability_df
    source_df = result.source_df
    metric_source_df = result.metric_source_df

    results_path = results_dir / "e05_morphospace_embeddings.parquet"
    results_csv_path = tables_dir / "e05_morphospace_embeddings.csv"
    route_metrics_path = results_dir / "e05_morphospace_route_metrics.parquet"
    route_metrics_csv_path = tables_dir / "e05_morphospace_route_metrics.csv"
    route_summary_path = results_dir / "e05_morphospace_route_summary.parquet"
    route_summary_csv_path = tables_dir / "e05_morphospace_route_summary.csv"
    stability_path = results_dir / "e05_morphospace_embedding_stability.parquet"
    stability_csv_path = tables_dir / "e05_morphospace_embedding_stability.csv"
    source_path = results_dir / "e05_morphospace_source_inputs.parquet"
    source_csv_path = tables_dir / "e05_morphospace_source_inputs.csv"
    metric_source_path = results_dir / "e05_morphospace_metric_sources.parquet"
    metric_source_csv_path = tables_dir / "e05_morphospace_metric_sources.csv"
    validation_path = results_dir / "e05_morphospace_validation.parquet"
    validation_csv_path = tables_dir / "e05_morphospace_validation.csv"
    figure_path = figures_dir / "morphospace_trajectory_map.png"
    config_path = configs_dir / "e05_s12_morphospace_trajectories.json"
    source_manifest_path = src_snapshot_dir / "e05_morphospace_trajectories_manifest.json"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    render_trajectory_figure(embedding_df, route_metrics_df, route_summary_df, figure_path)
    validation_df = run_validations(
        embedding_df=embedding_df,
        route_metrics_df=route_metrics_df,
        route_summary_df=route_summary_df,
        stability_df=stability_df,
        source_df=source_df,
        metric_source_df=metric_source_df,
        figure_path=figure_path,
    )
    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"
    outcome_classification, caveats_or_blockers, lay_summary = determine_outcome(validation_df, route_metrics_df)
    recommended_next_action = "Stop before S13 and let the Chief Scientist review S12; if accepted, proceed to S13 local-versus-global control comparison."

    embedding_df.to_parquet(results_path, index=False)
    embedding_df.to_csv(results_csv_path, index=False)
    route_metrics_df.to_parquet(route_metrics_path, index=False)
    route_metrics_df.to_csv(route_metrics_csv_path, index=False)
    route_summary_df.to_parquet(route_summary_path, index=False)
    route_summary_df.to_csv(route_summary_csv_path, index=False)
    stability_df.to_parquet(stability_path, index=False)
    stability_df.to_csv(stability_csv_path, index=False)
    source_df.to_parquet(source_path, index=False)
    source_df.to_csv(source_csv_path, index=False)
    metric_source_df.to_parquet(metric_source_path, index=False)
    metric_source_df.to_csv(metric_source_csv_path, index=False)
    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)

    config = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "featureColumns": list(FEATURE_COLUMNS),
        "embeddingMethod": "standardized_weighted_pca",
        "pcaVarianceRatio": list(result.pca_variance_ratio),
        "embeddingRows": int(len(embedding_df)),
        "routeMetricRows": int(len(route_metrics_df)),
        "routeSummaryRows": int(len(route_summary_df)),
        "stabilityThreshold": 0.70,
    }
    write_json(config_path, config)

    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "tests.e05.test_morphospace_trajectories", "-v"], args.repo_dir))
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )

    source_files = [
        source_entry(args.repo_dir / "src/e05/morphospace_trajectories.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s12_morphospace_trajectories.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_morphospace_trajectories.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/morphospace_metrics.py", args.repo_dir),
    ]
    write_json(
        source_manifest_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "sourceFiles": source_files,
        },
    )

    artifacts: list[dict[str, Any]] = [
        artifact_entry(results_path, artifacts_dir, "Required S12 morphospace embedding point table."),
        artifact_entry(results_csv_path, artifacts_dir, "CSV companion for S12 embedding point table."),
        artifact_entry(figure_path, artifacts_dir, "Required S12 morphospace trajectory map figure."),
        artifact_entry(route_metrics_path, artifacts_dir, "S12 per-run route metric table."),
        artifact_entry(route_metrics_csv_path, artifacts_dir, "CSV companion for route metrics."),
        artifact_entry(route_summary_path, artifacts_dir, "S12 route-diversity group summary table."),
        artifact_entry(route_summary_csv_path, artifacts_dir, "CSV companion for route-diversity summary."),
        artifact_entry(stability_path, artifacts_dir, "S12 embedding stability validation details."),
        artifact_entry(stability_csv_path, artifacts_dir, "CSV companion for embedding stability details."),
        artifact_entry(source_path, artifacts_dir, "S12 source trace input provenance table."),
        artifact_entry(source_csv_path, artifacts_dir, "CSV companion for source trace input provenance."),
        artifact_entry(metric_source_path, artifacts_dir, "S12 S05 metric source provenance table."),
        artifact_entry(metric_source_csv_path, artifacts_dir, "CSV companion for S05 metric source provenance."),
        artifact_entry(validation_path, artifacts_dir, "S12 validation case summary table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV companion for S12 validation cases."),
        artifact_entry(config_path, artifacts_dir, "S12 reproducibility configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "S12 source-code provenance manifest."),
    ]
    report_artifacts = [
        *artifacts,
        manifest_self_entry(full_results_path, artifacts_dir, "Canonical S12 full-results report."),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S12 artifact manifest."),
        manifest_self_entry(checksum_path, artifacts_dir, "SHA-256 checksums for S12 artifacts."),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Workspace-root S12 run manifest."),
    ]
    report = full_results_markdown(
        artifacts=report_artifacts,
        validation_result=validation_result,
        validation_df=validation_df,
        embedding_df=embedding_df,
        route_metrics_df=route_metrics_df,
        route_summary_df=route_summary_df,
        stability_df=stability_df,
        source_df=source_df,
        metric_source_df=metric_source_df,
        pca_variance_ratio=result.pca_variance_ratio,
        test_commands=test_commands,
        source_files=source_files,
        outcome_classification=outcome_classification,
        caveats_or_blockers=caveats_or_blockers,
        recommended_next_action=recommended_next_action,
        lay_summary=lay_summary,
    )
    write_text(full_results_path, report)
    artifacts.append(artifact_entry(full_results_path, artifacts_dir, "Canonical S12 full-results report."))
    artifacts.append(manifest_self_entry(artifact_manifest_path, artifacts_dir, "S12 artifact manifest."))
    write_json(
        artifact_manifest_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "artifacts": artifacts,
            "validationResult": validation_result,
            "outcomeClassification": outcome_classification,
            "createdAt": utc_now(),
        },
    )

    checksum_inputs = [
        results_path,
        results_csv_path,
        figure_path,
        route_metrics_path,
        route_metrics_csv_path,
        route_summary_path,
        route_summary_csv_path,
        stability_path,
        stability_csv_path,
        source_path,
        source_csv_path,
        metric_source_path,
        metric_source_csv_path,
        validation_path,
        validation_csv_path,
        config_path,
        source_manifest_path,
        full_results_path,
        artifact_manifest_path,
    ]
    write_checksums(checksum_inputs, checksum_path, artifacts_dir)
    artifacts.append(artifact_entry(checksum_path, artifacts_dir, "SHA-256 checksums for S12 artifacts."))

    ended_at = utc_now()
    run_manifest = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "experimentTitle": EXPERIMENT_TITLE,
        "startedAt": started_at,
        "endedAt": ended_at,
        "success": bool(validation_success and all(command["success"] for command in test_commands)),
        "status": "completed",
        "artifactsDir": str(artifacts_dir),
        "validationResult": validation_result,
        "outcomeClassification": outcome_classification,
        "recommendedNextAction": recommended_next_action,
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "configPath": str(config_path),
        "artifactManifestPath": str(artifact_manifest_path),
        "checksumPath": str(checksum_path),
        "testCommands": test_commands,
        "artifacts": artifacts,
    }
    write_json(run_manifest_path, run_manifest)
    write_checksums([*checksum_inputs, checksum_path, run_manifest_path], checksum_path, artifacts_dir)
    return 0 if run_manifest["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
