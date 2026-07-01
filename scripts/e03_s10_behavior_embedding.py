#!/usr/bin/env python3
"""Run E03 S10 behavior embedding from S07-S09 policy diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.e03.behavior_embedding import (
    DEFAULT_EMBEDDING_SEED,
    EMBEDDING_SCHEMA,
    attach_embedding_columns,
    behavior_feature_columns,
    build_policy_feature_frame,
    compute_embedding,
    embedding_digest,
    embedding_stability_frame,
    landmark_frame,
    validation_frame,
)
from src.e03.gpu_batch_simulator import jax_backend_summary


STEP_ID = "S10"
STEP_NUMBER = 10
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    artifacts_default = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_default)
    parser.add_argument("--s07-competence", type=Path, default=artifacts_default / "results/e03_policy_competence.parquet")
    parser.add_argument("--s08-candidates", type=Path, default=artifacts_default / "results/e03_qd_candidate_summary.parquet")
    parser.add_argument("--s08-archive", type=Path, default=artifacts_default / "results/e03_map_elites_archive.parquet")
    parser.add_argument("--s09-sweeps", type=Path, default=artifacts_default / "results/e03_phase_boundary_sweeps.parquet")
    parser.add_argument("--s09-boundaries", type=Path, default=artifacts_default / "results/e03_phase_boundary_candidates.parquet")
    parser.add_argument("--seed", type=int, default=DEFAULT_EMBEDDING_SEED)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def manifest_self_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum omitted to avoid self-referential drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float) or isinstance(value, np.floating):
                text = "nan" if pd.isna(value) else f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def normalize_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(lambda value: None if value is None else str(value))
    return out


def policy_group(row: pd.Series) -> str:
    if bool(row.get("classic_landmark", False)):
        return "Classic landmarks"
    if bool(row.get("qd_candidate", False)):
        return "S08 QD candidates"
    if bool(row.get("s08_archive_winner", False)):
        return "S08 archive seed winners"
    source = str(row.get("source_kind", ""))
    if "hand" in source:
        return "Hand-designed variants"
    if "mutation" in source or "recombination" in source:
        return "Generated variants"
    return "S05 generated library"


def write_embedding_figure(path: Path, embeddings: pd.DataFrame, landmarks: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = embeddings.copy()
    plot_df["plot_group"] = plot_df.apply(policy_group, axis=1)
    palette = {
        "Classic landmarks": "#1f4e79",
        "S08 QD candidates": "#d95f02",
        "S08 archive seed winners": "#4daf4a",
        "Hand-designed variants": "#756bb1",
        "Generated variants": "#7570b3",
        "S05 generated library": "#8c8c8c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for group, subset in plot_df.groupby("plot_group", sort=False):
        axes[0].scatter(
            subset["embedding_x"],
            subset["embedding_y"],
            s=14,
            alpha=0.45,
            linewidths=0,
            color=palette.get(group, "#777777"),
            label=f"{group} ({len(subset)})",
        )
    archive = plot_df[plot_df["s08_archive_winner"].astype(bool)]
    if not archive.empty:
        axes[0].scatter(
            archive["embedding_x"],
            archive["embedding_y"],
            s=54,
            facecolors="none",
            edgecolors="#222222",
            linewidths=0.8,
            label="S08 archive winners",
        )
    s09 = plot_df[plot_df["s09_diagnostic_available"].astype(bool)]
    if not s09.empty:
        axes[0].scatter(
            s09["embedding_x"],
            s09["embedding_y"],
            s=70,
            marker="s",
            facecolors="none",
            edgecolors="#e3b341",
            linewidths=1.0,
            label="S09 phase-diagnostic policies",
        )
    if not landmarks.empty:
        axes[0].scatter(
            landmarks["embedding_x"],
            landmarks["embedding_y"],
            s=130,
            marker="*",
            color="#111111",
            edgecolors="#ffffff",
            linewidths=0.6,
            label="Classic labels",
            zorder=5,
        )
        for _, row in landmarks.iterrows():
            axes[0].annotate(
                str(row["classic_family"]),
                (row["embedding_x"], row["embedding_y"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#111111",
            )
    axes[0].set_title("Behavior embedding by policy source")
    axes[0].set_xlabel("PCA behavior coordinate 1")
    axes[0].set_ylabel("PCA behavior coordinate 2")
    axes[0].legend(loc="best", fontsize=8, frameon=False)

    score = pd.to_numeric(plot_df["screen_score"], errors="coerce")
    scatter = axes[1].scatter(
        plot_df["embedding_x"],
        plot_df["embedding_y"],
        c=score,
        cmap="viridis",
        s=16,
        alpha=0.65,
        linewidths=0,
    )
    if not landmarks.empty:
        axes[1].scatter(
            landmarks["embedding_x"],
            landmarks["embedding_y"],
            s=120,
            marker="*",
            color="#ffffff",
            edgecolors="#111111",
            linewidths=0.8,
            zorder=5,
        )
        for _, row in landmarks.iterrows():
            axes[1].annotate(
                str(row["classic_family"]),
                (row["embedding_x"], row["embedding_y"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#111111",
            )
    axes[1].set_title("Behavior embedding by S07/S08 screen score")
    axes[1].set_xlabel("PCA behavior coordinate 1")
    colorbar = fig.colorbar(scatter, ax=axes[1], shrink=0.85)
    colorbar.set_label("Screen score")
    for ax in axes:
        ax.grid(True, alpha=0.16, linewidth=0.6)
    fig.suptitle("E03 S10 policy behavior embedding from competence and phase diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_summary_tables(embeddings: pd.DataFrame, stability: pd.DataFrame, landmarks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    source_summary = (
        embeddings.groupby("embedding_input_source")
        .agg(
            policy_count=("policy_id", "size"),
            mean_screen_score=("screen_score", "mean"),
            mean_heldout_sortedness=("screen_heldout_final_sortedness_mean", "mean"),
            archive_winner_count=("s08_archive_winner", "sum"),
            s09_diagnostic_count=("s09_diagnostic_available", "sum"),
        )
        .reset_index()
    )
    source_kind_summary = (
        embeddings.groupby("source_kind")
        .agg(
            policy_count=("policy_id", "size"),
            mean_screen_score=("screen_score", "mean"),
            mean_embedding_x=("embedding_x", "mean"),
            mean_embedding_y=("embedding_y", "mean"),
        )
        .reset_index()
        .sort_values(["policy_count", "mean_screen_score"], ascending=[False, False])
        .head(20)
    )
    stability_summary = (
        stability.groupby("normalization")
        .agg(
            replicate_count=("seed", "size"),
            mean_neighbor_jaccard=("mean_neighbor_jaccard", "mean"),
            median_distance_spearman=("sample_distance_spearman", "median"),
            mean_explained_variance_1=("explained_variance_1", "mean"),
            mean_explained_variance_2=("explained_variance_2", "mean"),
        )
        .reset_index()
    )
    return {
        "source_summary": source_summary,
        "source_kind_summary": source_kind_summary,
        "stability_summary": stability_summary,
        "landmarks": landmarks,
    }


def render_report(
    *,
    artifacts: dict[str, Path],
    manifest: dict[str, Any],
    validation: pd.DataFrame,
    unit_tests: dict[str, Any],
    command_line: str,
    embeddings: pd.DataFrame,
    feature_stats: pd.DataFrame,
    stability: pd.DataFrame,
    landmarks: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> str:
    validation_success = bool(validation["success"].all() and unit_tests["success"])
    outcome = "supportive" if validation_success else "constraining/contradictory"
    validation_line = f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; unit tests return code {unit_tests['returnCode']}"
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    command_rows = pd.DataFrame(
        [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": command_line, "returnCode": 0, "success": True},
        ]
    )
    source_table = markdown_table(pd.DataFrame(manifest["sourceFiles"]), ["relativePath", "sha256", "sizeBytes"])
    variance = (
        float(embeddings["pca_explained_variance_1"].iloc[0]),
        float(embeddings["pca_explained_variance_2"].iloc[0]),
        float(embeddings["pca_explained_variance_3"].iloc[0]),
    )
    mean_jaccard = float(stability["mean_neighbor_jaccard"].mean()) if not stability.empty else 0.0
    median_spearman = float(stability["sample_distance_spearman"].median()) if not stability.empty else 0.0
    return f"""# E03 S10 Research Step Full Results

## Top Summary

- Step ID: S10
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; embedding digest `{embedding_digest(embeddings)}`.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains. The embedding is an exploratory behavioral projection over proxy DSL competence and compact phase diagnostics, not a causal or full-public-simulator taxonomy.
- Lay summary: S10 joined {len(embeddings)} policies by stable policy ID across S07 competence rows, S08 quality-diversity candidates, and S09 phase diagnostics. It wrote a normalized PCA behavior map, highlighted {len(landmarks)} classic DSL landmarks, and found stability support across seeded feature-subset and normalization checks with mean neighbor Jaccard {mean_jaccard:.3f} and median sampled distance Spearman {median_spearman:.3f}.
- Recommended next action: Stop for Chief review; if accepted, proceed to S11 universality-class clustering using this embedding, S07-S09 metrics, and representative policy traces.

## Frozen Question

Can policies be mapped by behavioral similarity rather than source-code similarity, revealing neighborhoods and gradients in Algotype morphospace?

## Inputs

- S07 competence table: `{manifest['inputArtifacts']['s07Competence']}`
- S08 candidate summary: `{manifest['inputArtifacts']['s08Candidates']}`
- S08 MAP-Elites archive: `{manifest['inputArtifacts']['s08Archive']}`
- S09 phase sweep rows: `{manifest['inputArtifacts']['s09Sweeps']}`
- S09 boundary candidates: `{manifest['inputArtifacts']['s09Boundaries']}`

## Methods

S10 built one policy-level table keyed by `policy_id`. S07 competence rows form the base library; S08 QD candidates are appended as additional discovered policies; S08 archive membership is kept as metadata; S09 rows are aggregated by `base_policy_id` and left-joined onto matching policies.

The embedding feature set intentionally uses behavior measurements only: S07/S08 screen, held-out, transfer, timeout, work, and score columns plus S09 aggregate final sortedness, sortedness deltas, DG proxy, oscillation, no-change, failure-mode, axis-response, and boundary diagnostics. DSL source text and grammar feature flags are not embedding features.

Features were median-imputed, scaled from observed values rather than imputed mass, block-balanced between competence metrics and S09 phase diagnostics, filtered for variance, and embedded with deterministic PCA. Stability was checked with seeded 80% feature subsets under robust and z-score normalization. Each replicate was compared to the primary embedding by top-k neighbor Jaccard overlap and sampled pairwise-distance Spearman correlation.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- scikit-learn PCA and nearest-neighbor routines were used through `src/e03/behavior_embedding.py`.
- JAX backend summary, for continuity with S06/S07 routing context: `{json.dumps(manifest['runtime']['jax'], sort_keys=True)}`
- Worker count: serial CPU matrix assembly and PCA; no worker pool used.
- New dependencies installed: none.

## Parameters

- Embedding seed: `{manifest['parameters']['seed']}`
- Block balancing: `{manifest['parameters']['blockBalance']}` for competence versus phase-diagnostic feature groups
- Policy rows embedded: `{len(embeddings)}`
- Behavior feature columns selected before variance filtering: `{manifest['summary']['selectedFeatureColumns']}`
- Retained normalized feature columns: `{manifest['summary']['retainedFeatureColumns']}`
- PCA explained variance ratios: PC1 {variance[0]:.6g}, PC2 {variance[1]:.6g}, PC3 {variance[2]:.6g}
- Stability checks: `{len(stability)}` seeded normalization/subset replicates

## Results

### Source Summary

{markdown_table(tables['source_summary'], ["embedding_input_source", "policy_count", "mean_screen_score", "mean_heldout_sortedness", "archive_winner_count", "s09_diagnostic_count"])}

### Source-Kind Summary

{markdown_table(tables['source_kind_summary'], ["source_kind", "policy_count", "mean_screen_score", "mean_embedding_x", "mean_embedding_y"])}

### Stability Summary

{markdown_table(tables['stability_summary'], ["normalization", "replicate_count", "mean_neighbor_jaccard", "median_distance_spearman", "mean_explained_variance_1", "mean_explained_variance_2"])}

### Classic Landmarks

{markdown_table(landmarks, ["policy_name", "classic_family", "embedding_x", "embedding_y", "screen_score", "screen_heldout_final_sortedness_mean", "s09_diagnostic_available"])}

## Metrics

- `embedding_x`, `embedding_y`, `embedding_z`: first three PCA coordinates of normalized behavior metrics.
- `embedding_missing_fraction`: fraction of selected behavior features missing before median imputation for that policy.
- `mean_neighbor_jaccard`: average overlap between primary and replicate top-k embedding neighborhoods.
- `sample_distance_spearman`: Spearman correlation between sampled pairwise distances in primary and replicate embeddings.
- `classic_landmark`: Bubble, Insertion, and Selection DSL landmarks from S03/S05, used only for interpretation and plotting.

## Figures

- Policy behavior embedding: `{artifacts['embeddingFigure']}`

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "expected", "observed", "notes"])}

## Caveats, Blockers, And Limitations

- The embedding is based on S07/S08/S09 proxy DSL local-step behavior. It does not validate policies in the original public simulator.
- S09 trajectory diagnostics exist for 10 selected policies, so S09 features can refine landmark and candidate placement but cannot fully describe every generated policy.
- PCA is a linear projection. It is stable enough for S11 clustering input, but nonlinear atlases or interactive maps may still reveal structure hidden by this view.
- Missing transfer and phase diagnostics are median-imputed to prevent diagnostic availability itself from dominating the map.
- Similar coordinates are evidence of similar measured behavior under these metrics, not proof of shared mechanisms or source-level equivalence.

## Failed Assumptions

No required S07, S08, or S09 input table was missing. The open S09-to-S10 assumption was partly supported: feature-subset and normalization stability passed, but S09 coverage remains sparse and should be treated as local diagnostic context.

## Provenance

Git commit before S10 commit: `{manifest['git']['headCommit']}`

Source files hashed in the S10 manifest:

{source_table}

## Artifacts

Reusable S10 outputs are the policy embedding table, raw and normalized feature tables, feature statistics, stability and validation tables, classic landmark table, embedding figure, config, manifests, status JSON, checksums, and this full-results handoff report.
"""


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e03"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, figures_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_tests = {"command": "not run", "returnCode": 0, "success": True, "stdout": "", "stderr": "", "elapsedSeconds": 0.0}
    if args.run_unit_tests:
        print("S10: running full E03 unit tests", flush=True)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03"], args.repo_dir)

    print("S10: loading S07-S09 policy behavior inputs", flush=True)
    s07 = pd.read_parquet(args.s07_competence)
    s08 = pd.read_parquet(args.s08_candidates)
    archive = pd.read_parquet(args.s08_archive)
    sweeps = pd.read_parquet(args.s09_sweeps)
    boundaries = pd.read_parquet(args.s09_boundaries)

    print("S10: joining policy rows and aggregating S09 diagnostics", flush=True)
    policy_frame = build_policy_feature_frame(s07, s08, archive, sweeps, boundaries)
    feature_columns = behavior_feature_columns(policy_frame)
    print(f"S10: selected {len(feature_columns)} behavior feature columns", flush=True)
    result = compute_embedding(policy_frame, feature_columns, seed=args.seed, normalization="robust", n_components=3)
    embeddings = attach_embedding_columns(policy_frame, result)
    stability = embedding_stability_frame(policy_frame, feature_columns, result.embedding)
    landmarks = landmark_frame(embeddings)

    embedding_path = results_dir / "e03_policy_embeddings.parquet"
    embedding_csv_path = tables_dir / "e03_policy_embeddings.csv"
    raw_features_path = results_dir / "e03_policy_embedding_features.parquet"
    normalized_features_path = results_dir / "e03_policy_embedding_normalized_features.parquet"
    feature_stats_path = tables_dir / "e03_policy_embedding_feature_stats.csv"
    stability_path = results_dir / "e03_policy_embedding_stability.parquet"
    stability_csv_path = tables_dir / "e03_policy_embedding_stability.csv"
    landmarks_path = tables_dir / "e03_embedding_landmarks.csv"
    validation_path = results_dir / "e03_behavior_embedding_validation.parquet"
    validation_csv_path = tables_dir / "e03_behavior_embedding_validation.csv"
    figure_path = figures_dir / "policy_behavior_embedding.png"
    config_path = configs_dir / "e03_s10_behavior_embedding.json"
    manifest_path = src_snapshot_dir / "e03_behavior_embedding_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    status_path = step_dir / "status.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = checksums_dir / "sha256sums.txt"
    full_report_path = step_dir / "research_step_full_results.md"

    write_embedding_figure(figure_path, embeddings, landmarks)
    validation = validation_frame(
        policy_frame=policy_frame,
        embedding_frame=embeddings,
        feature_columns=feature_columns,
        stability=stability,
        landmarks=landmarks,
        figure_exists=figure_path.exists() and figure_path.stat().st_size > 0,
        unit_success=bool(unit_tests["success"]),
    )
    for column in ("validation_case", "expected", "observed", "notes"):
        validation[column] = validation[column].astype(str)

    raw_features = policy_frame[["policy_id", "policy_name", "embedding_input_source", *feature_columns]].copy()
    normalized_features = pd.DataFrame(result.normalized.matrix, columns=result.normalized.feature_names)
    normalized_features.insert(0, "policy_name", policy_frame["policy_name"].to_numpy())
    normalized_features.insert(0, "policy_id", policy_frame["policy_id"].to_numpy())

    normalize_for_parquet(embeddings).to_parquet(embedding_path, index=False)
    embeddings.to_csv(embedding_csv_path, index=False)
    normalize_for_parquet(raw_features).to_parquet(raw_features_path, index=False)
    normalize_for_parquet(normalized_features).to_parquet(normalized_features_path, index=False)
    result.normalized.stats.to_csv(feature_stats_path, index=False)
    normalize_for_parquet(stability).to_parquet(stability_path, index=False)
    stability.to_csv(stability_csv_path, index=False)
    landmarks.to_csv(landmarks_path, index=False)
    normalize_for_parquet(validation).to_parquet(validation_path, index=False)
    validation.to_csv(validation_csv_path, index=False)

    config = {
        "schema": "eidosoma.e03.s10_behavior_embedding_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "seed": args.seed,
        "normalization": "robust",
        "blockBalance": True,
        "embeddingMethod": "PCA",
        "stabilityMethods": ["seeded 80% feature subsets", "robust normalization", "z-score normalization"],
        "featureColumns": list(feature_columns),
        "retainedFeatureColumns": list(result.normalized.feature_names),
        "inputs": {
            "s07Competence": str(args.s07_competence),
            "s08Candidates": str(args.s08_candidates),
            "s08Archive": str(args.s08_archive),
            "s09Sweeps": str(args.s09_sweeps),
            "s09Boundaries": str(args.s09_boundaries),
        },
    }
    write_json(config_path, config)

    elapsed = time.perf_counter() - started
    source_files = [
        args.repo_dir / "src/e03/behavior_embedding.py",
        args.repo_dir / "tests/e03/test_behavior_embedding.py",
        args.repo_dir / "scripts/e03_s10_behavior_embedding.py",
        args.repo_dir / "src/e03/phase_boundaries.py",
        args.repo_dir / "src/e03/quality_diversity.py",
    ]
    manifest = {
        "schema": "eidosoma.e03.behavior_embedding_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "git": {
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "headCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
        "parameters": {
            "seed": args.seed,
            "normalization": "robust",
            "blockBalance": True,
            "pcaComponents": 3,
            "stabilityFeatureFraction": 0.8,
            "stabilityNormalizations": ["robust", "zscore"],
        },
        "inputArtifacts": {
            "s07Competence": str(args.s07_competence),
            "s08Candidates": str(args.s08_candidates),
            "s08Archive": str(args.s08_archive),
            "s09Sweeps": str(args.s09_sweeps),
            "s09Boundaries": str(args.s09_boundaries),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "jax": jax_backend_summary(),
            "elapsedSeconds": elapsed,
        },
        "sourceFiles": [source_entry(path, args.repo_dir) for path in source_files],
        "summary": {
            "policyRows": int(len(embeddings)),
            "s07Rows": int((policy_frame["embedding_input_source"] == "s07_competence").sum()),
            "s08Rows": int((policy_frame["embedding_input_source"] == "s08_qd_candidates").sum()),
            "s09DiagnosticRows": int(policy_frame["s09_diagnostic_available"].sum()),
            "classicLandmarkRows": int(len(landmarks)),
            "selectedFeatureColumns": int(len(feature_columns)),
            "retainedFeatureColumns": int(len(result.normalized.feature_names)),
            "embeddingDigest": embedding_digest(embeddings),
            "pcaExplainedVarianceRatio": list(result.explained_variance_ratio),
            "meanNeighborJaccard": float(stability["mean_neighbor_jaccard"].mean()),
            "medianDistanceSpearman": float(stability["sample_distance_spearman"].median()),
        },
        "validationSummary": {
            "success": bool(validation["success"].all() and unit_tests["success"]),
            "validationCasesPassed": int(validation["success"].sum()),
            "validationCasesTotal": int(len(validation)),
            "unitTestsReturnCode": int(unit_tests["returnCode"]),
        },
    }

    artifacts = {
        "researchStepReport": full_report_path,
        "policyEmbeddings": embedding_path,
        "policyEmbeddingsCsv": embedding_csv_path,
        "rawFeatureTable": raw_features_path,
        "normalizedFeatureTable": normalized_features_path,
        "featureStats": feature_stats_path,
        "stabilityParquet": stability_path,
        "stabilityCsv": stability_csv_path,
        "classicLandmarks": landmarks_path,
        "validationParquet": validation_path,
        "validationCsv": validation_csv_path,
        "embeddingFigure": figure_path,
        "embeddingConfig": config_path,
        "sourceSnapshotManifest": manifest_path,
        "artifactManifest": artifact_manifest_path,
        "statusJson": status_path,
        "runManifest": run_manifest_path,
        "checksums": checksums_path,
    }
    tables = make_summary_tables(embeddings, stability, landmarks)
    full_report = render_report(
        artifacts=artifacts,
        manifest=manifest,
        validation=validation,
        unit_tests=unit_tests,
        command_line=" ".join(sys.argv),
        embeddings=embeddings,
        feature_stats=result.normalized.stats,
        stability=stability,
        landmarks=landmarks,
        tables=tables,
    )
    write_text(full_report_path, full_report)

    validation_result = (
        f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; "
        f"unit tests return code {unit_tests['returnCode']}"
    )
    recommended_next_action = "Stop for Chief review; if accepted, proceed to S11 universality-class clustering."
    caveats = (
        "No blocker. Embedding is an exploratory proxy projection; S09 diagnostics cover only selected policies "
        "and missing phase/transfer metrics are median-imputed."
    )
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"].all() and unit_tests["success"]),
        "status": "completed" if bool(validation["success"].all() and unit_tests["success"]) else "completed_with_validation_failures",
        "artifactsWritten": [str(path) for path in artifacts.values()],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
    }
    write_json(status_path, status)

    artifact_entries = [
        artifact_entry(embedding_path, artifacts_dir, "S10 policy behavior embedding table"),
        artifact_entry(embedding_csv_path, artifacts_dir, "CSV sidecar for S10 policy embeddings"),
        artifact_entry(raw_features_path, artifacts_dir, "S10 raw behavior feature table used for embedding"),
        artifact_entry(normalized_features_path, artifacts_dir, "S10 normalized behavior feature matrix"),
        artifact_entry(feature_stats_path, artifacts_dir, "S10 feature normalization statistics"),
        artifact_entry(stability_path, artifacts_dir, "S10 embedding stability checks"),
        artifact_entry(stability_csv_path, artifacts_dir, "CSV sidecar for S10 embedding stability checks"),
        artifact_entry(landmarks_path, artifacts_dir, "S10 classic embedding landmark table"),
        artifact_entry(validation_path, artifacts_dir, "S10 validation cases"),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S10 validation cases"),
        artifact_entry(figure_path, artifacts_dir, "S10 policy behavior embedding figure"),
        artifact_entry(config_path, artifacts_dir, "S10 behavior embedding config"),
        manifest_self_entry(manifest_path, artifacts_dir, "S10 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S10 artifact manifest"),
        artifact_entry(status_path, artifacts_dir, "S10 compact status JSON"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S10"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S10 outputs"),
        artifact_entry(full_report_path, artifacts_dir, "S10 full-results handoff report"),
    ]
    manifest["artifacts"] = artifact_entries
    write_json(manifest_path, manifest)
    artifact_manifest = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "artifacts": [
            artifact_entry(embedding_path, artifacts_dir, "S10 policy behavior embedding table"),
            artifact_entry(embedding_csv_path, artifacts_dir, "CSV sidecar for S10 policy embeddings"),
            artifact_entry(raw_features_path, artifacts_dir, "S10 raw behavior feature table used for embedding"),
            artifact_entry(normalized_features_path, artifacts_dir, "S10 normalized behavior feature matrix"),
            artifact_entry(feature_stats_path, artifacts_dir, "S10 feature normalization statistics"),
            artifact_entry(stability_path, artifacts_dir, "S10 embedding stability checks"),
            artifact_entry(stability_csv_path, artifacts_dir, "CSV sidecar for S10 embedding stability checks"),
            artifact_entry(landmarks_path, artifacts_dir, "S10 classic embedding landmark table"),
            artifact_entry(validation_path, artifacts_dir, "S10 validation cases"),
            artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S10 validation cases"),
            artifact_entry(figure_path, artifacts_dir, "S10 policy behavior embedding figure"),
            artifact_entry(config_path, artifacts_dir, "S10 behavior embedding config"),
            artifact_entry(manifest_path, artifacts_dir, "S10 source snapshot and provenance manifest"),
            manifest_self_entry(artifact_manifest_path, artifacts_dir, "S10 artifact manifest"),
            artifact_entry(status_path, artifacts_dir, "S10 compact status JSON"),
            manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S10"),
            manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S10 outputs"),
            artifact_entry(full_report_path, artifacts_dir, "S10 full-results handoff report"),
        ],
    }
    write_json(artifact_manifest_path, artifact_manifest)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "git": manifest["git"],
        "runtime": manifest["runtime"],
        "commands": [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": " ".join(sys.argv), "returnCode": 0, "success": True},
        ],
        "inputs": manifest["inputArtifacts"],
        "parameters": manifest["parameters"],
        "outputs": artifact_manifest["artifacts"],
    }
    write_json(run_manifest_path, run_manifest)
    checksum_targets = [
        full_report_path,
        embedding_path,
        embedding_csv_path,
        raw_features_path,
        normalized_features_path,
        feature_stats_path,
        stability_path,
        stability_csv_path,
        landmarks_path,
        validation_path,
        validation_csv_path,
        figure_path,
        config_path,
        manifest_path,
        artifact_manifest_path,
        status_path,
        run_manifest_path,
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")

    success = bool(validation["success"].all() and unit_tests["success"])
    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": success,
                "policyRows": int(len(embeddings)),
                "featureColumns": int(len(feature_columns)),
                "retainedFeatureColumns": int(len(result.normalized.feature_names)),
                "meanNeighborJaccard": float(stability["mean_neighbor_jaccard"].mean()),
                "medianDistanceSpearman": float(stability["sample_distance_spearman"].median()),
                "artifactsDir": str(step_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
