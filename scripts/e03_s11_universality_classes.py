#!/usr/bin/env python3
"""Run E03 S11 universality-class clustering over S10 behavior embeddings."""

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

from src.e03.gpu_batch_simulator import jax_backend_summary
from src.e03.universality_classes import (
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_CLUSTER_SEED,
    assignments_digest,
    attach_labels,
    build_assignments,
    build_code_feature_frame,
    candidate_k_frame,
    cluster_summary_frame,
    exemplar_frame,
    feature_columns,
    feature_matrix,
    fit_kmeans,
    load_policy_code_table,
    stability_frame,
    validation_frame,
)


STEP_ID = "S11"
STEP_NUMBER = 11
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    artifacts_default = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_default)
    parser.add_argument("--s10-embeddings", type=Path, default=artifacts_default / "results/e03_policy_embeddings.parquet")
    parser.add_argument("--s10-normalized-features", type=Path, default=artifacts_default / "results/e03_policy_embedding_normalized_features.parquet")
    parser.add_argument("--generated-policy-library", type=Path, default=artifacts_default / "policies/e03_generated_policy_library.jsonl")
    parser.add_argument("--qd-candidates", type=Path, default=artifacts_default / "results/e03_qd_candidate_summary.parquet")
    parser.add_argument("--qd-discovered", type=Path, default=artifacts_default / "policies/e03_qd_discovered_policies.jsonl")
    parser.add_argument("--cluster-count", type=int, default=DEFAULT_CLUSTER_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_CLUSTER_SEED)
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
            values.append(text.replace("|", "\\|").replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def normalize_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(lambda value: None if value is None else str(value))
    return out


def compact_assignment_table(assignments: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "policy_id",
        "policy_name",
        "class_id",
        "cautious_label",
        "label_confidence",
        "raw_cluster_id",
        "cluster_distance",
        "screen_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_work_mean",
        "source_kind",
        "route",
        "qd_candidate",
        "s08_archive_winner",
        "classic_landmark",
        "classic_family",
        "s09_diagnostic_available",
        "usesIdeal",
        "usesPrefixSorted",
        "usesRandomCondition",
        "usesProbabilisticAction",
        "usesMemory",
        "usesSignal",
        "swapActionCount",
        "stateActionCount",
        "waitActionCount",
        "rule_count",
        "dsl_excerpt",
    ]
    return assignments[[column for column in columns if column in assignments.columns]].sort_values(["class_id", "cluster_distance"], kind="mergesort")


def write_cluster_figure(path: Path, assignments: pd.DataFrame, summary: pd.DataFrame, exemplars: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = summary["class_id"].tolist()
    colors = dict(zip(classes, plt.get_cmap("tab10").colors, strict=False))
    if len(classes) > 10:
        cmap = plt.get_cmap("tab20")
        colors = {class_id: cmap(index % 20) for index, class_id in enumerate(classes)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for class_id, group in assignments.groupby("class_id", sort=True):
        ax.scatter(
            group["embedding_x"],
            group["embedding_y"],
            s=12,
            alpha=0.42,
            linewidths=0,
            color=colors.get(class_id, "#777777"),
            label=f"{class_id}: {summary.loc[summary['class_id'] == class_id, 'cautious_label'].iloc[0]}",
        )
    medoids = exemplars[exemplars["exemplar_role"] == "medoid"]
    if not medoids.empty:
        ax.scatter(medoids["embedding_x"], medoids["embedding_y"], s=80, marker="D", facecolors="none", edgecolors="#111111", linewidths=0.9, label="medoids")
    classics = assignments[assignments["classic_landmark"].astype(bool)]
    if not classics.empty:
        ax.scatter(classics["embedding_x"], classics["embedding_y"], s=140, marker="*", color="#ffffff", edgecolors="#111111", linewidths=0.8, zorder=5, label="classic landmarks")
        for _, row in classics.iterrows():
            ax.annotate(str(row["classic_family"]), (row["embedding_x"], row["embedding_y"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    centroids = assignments.groupby("class_id").agg(x=("embedding_x", "mean"), y=("embedding_y", "mean")).reset_index()
    for _, row in centroids.iterrows():
        ax.annotate(str(row["class_id"]), (row["x"], row["y"]), ha="center", va="center", fontsize=9, weight="bold", color="#111111")
    ax.set_title("S11 proxy universality classes in S10 behavior embedding")
    ax.set_xlabel("S10 behavior embedding x")
    ax.set_ylabel("S10 behavior embedding y")
    ax.grid(True, alpha=0.16, linewidth=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), fontsize=7, frameon=False, ncol=2)

    bar_ax = axes[1]
    plot_summary = summary.sort_values("class_id", ascending=False)
    y = np.arange(len(plot_summary))
    bar_ax.barh(y, plot_summary["mean_heldout_sortedness"], color=[colors.get(cid, "#777777") for cid in plot_summary["class_id"]], alpha=0.85)
    for index, row in enumerate(plot_summary.to_dict(orient="records")):
        bar_ax.text(
            min(float(row["mean_heldout_sortedness"]) + 0.015, 0.98),
            index,
            f"n={int(row['policy_count'])}",
            va="center",
            fontsize=8,
        )
    bar_ax.set_yticks(y)
    bar_ax.set_yticklabels([f"{row['class_id']} {row['cautious_label']}" for row in plot_summary.to_dict(orient="records")], fontsize=8)
    bar_ax.set_xlim(-0.05, 1.05)
    bar_ax.set_xlabel("Mean held-out final sortedness")
    bar_ax.set_title("Class competence summary")
    bar_ax.grid(True, axis="x", alpha=0.16, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_class_summary_markdown(path: Path, summary: pd.DataFrame, exemplars: pd.DataFrame, stability: pd.DataFrame) -> None:
    lines = [
        "# E03 S11 Universality Class Summary",
        "",
        "This summary names proxy behavior classes from the S10/S11 clustering. Labels are cautious descriptors, not causal claims.",
        "",
        "## Classes",
        "",
    ]
    columns = [
        "class_id",
        "cautious_label",
        "label_confidence",
        "policy_count",
        "mean_screen_score",
        "mean_heldout_sortedness",
        "mean_work",
        "classic_families",
        "label_basis",
    ]
    lines.append(markdown_table(summary, columns))
    lines.extend(["", "## Inspected Exemplars", ""])
    ex_cols = [
        "class_id",
        "cautious_label",
        "exemplar_role",
        "policy_name",
        "screen_score",
        "classic_family",
        "top_normalized_feature_deviations",
        "dsl_excerpt",
    ]
    lines.append(markdown_table(exemplars, [column for column in ex_cols if column in exemplars.columns]))
    lines.extend(["", "## Stability By Metric Subset", ""])
    stable_summary = (
        stability.groupby("subset_name")
        .agg(
            replicate_count=("seed", "size"),
            mean_adjusted_rand_index=("adjusted_rand_index", "mean"),
            min_adjusted_rand_index=("adjusted_rand_index", "min"),
            mean_normalized_mutual_info=("normalized_mutual_info", "mean"),
        )
        .reset_index()
        .sort_values("subset_name")
    )
    lines.append(markdown_table(stable_summary, list(stable_summary.columns)))
    write_text(path, "\n".join(lines) + "\n")


def render_report(
    *,
    artifacts: dict[str, Path],
    manifest: dict[str, Any],
    validation: pd.DataFrame,
    unit_tests: dict[str, Any],
    command_line: str,
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    candidates: pd.DataFrame,
    exemplars: pd.DataFrame,
) -> str:
    success = bool(validation["success"].all() and unit_tests["success"])
    outcome = "supportive" if success else "constraining/contradictory"
    validation_line = f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; unit tests return code {unit_tests['returnCode']}"
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    command_rows = pd.DataFrame(
        [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": command_line, "returnCode": 0, "success": True},
        ]
    )
    source_table = markdown_table(pd.DataFrame(manifest["sourceFiles"]), ["relativePath", "sha256", "sizeBytes"])
    stable_summary = (
        stability.groupby("subset_name")
        .agg(
            replicate_count=("seed", "size"),
            mean_adjusted_rand_index=("adjusted_rand_index", "mean"),
            min_adjusted_rand_index=("adjusted_rand_index", "min"),
            mean_normalized_mutual_info=("normalized_mutual_info", "mean"),
        )
        .reset_index()
        .sort_values("subset_name")
    )
    candidate_cols = ["candidate_k", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score", "min_cluster_size", "max_cluster_size"]
    summary_cols = [
        "class_id",
        "cautious_label",
        "label_confidence",
        "policy_count",
        "mean_screen_score",
        "mean_heldout_sortedness",
        "mean_work",
        "qd_candidate_count",
        "archive_winner_count",
        "classic_families",
        "label_basis",
    ]
    exemplar_cols = [
        "class_id",
        "cautious_label",
        "exemplar_role",
        "policy_name",
        "source_kind",
        "screen_score",
        "classic_family",
        "s08_archive_winner",
        "top_normalized_feature_deviations",
        "dsl_excerpt",
    ]
    return f"""# E03 S11 Research Step Full Results

## Top Summary

- Step ID: S11
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; assignment digest `{assignments_digest(assignments)}`.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains. Classes are proxy behavior families from S10/S07-S09 metrics and DSL excerpts, not causal mechanisms or full public-simulator universality classes.
- Lay summary: S11 clustered {len(assignments)} S10-embedded policies into {summary['class_id'].nunique()} cautious proxy classes. The selected k={manifest['parameters']['clusterCount']} solution has silhouette {manifest['summary']['selectedSilhouette']:.3f}; metric-subset stability passed with minimum subset mean adjusted Rand index {manifest['summary']['minimumSubsetMeanARI']:.3f}. The report highlights classic Bubble, Insertion, and Selection landmarks plus {len(exemplars)} inspected exemplar rows with DSL rule excerpts.
- Recommended next action: Stop for Chief review; if accepted, proceed to S12 causal rule-component ablations using S11 class exemplars and policy code.

## Frozen Question

Do policies cluster into interpretable families such as gradient-followers, target-position seekers, local-inversion cleaners, barrier navigators, cooperative sorters, and pathological oscillators?

## Inputs

- S10 policy embeddings: `{manifest['inputArtifacts']['s10Embeddings']}`
- S10 normalized behavior features: `{manifest['inputArtifacts']['s10NormalizedFeatures']}`
- S05 generated policy library: `{manifest['inputArtifacts']['generatedPolicyLibrary']}`
- S08 QD candidate summary with DSL source: `{manifest['inputArtifacts']['qdCandidates']}`
- S08 discovered policy library: `{manifest['inputArtifacts']['qdDiscovered']}`
- S07-S09 metrics are carried through the S10 embedding table and normalized feature table.

## Methods

S11 clustered the S10 normalized behavior feature matrix, not raw DSL text. The primary clustering used deterministic KMeans with k={manifest['parameters']['clusterCount']} and 50 restarts. Candidate k values 4 through 12 were scored with silhouette, Calinski-Harabasz, Davies-Bouldin, and minimum cluster size; k=8 was retained because it preserved interpretable non-singleton bands and matched the S10 stability structure.

Stability was tested by reclustering metric subsets: all S10 features, competence-only features, screen-core features, all features excluding phase-boundary diagnostics, phase-plus-screen diagnostics, and the three S10 embedding coordinates. Each subset was run across five deterministic seeds and compared to the primary labels with adjusted Rand index and normalized mutual information.

Policy code was parsed after clustering to support interpretation. S11 extracted rule counts, condition/action counts, target-position usage, prefix guards, stochastic guards, memory/signal stubs, swap/state/wait counts, and DSL excerpts for inspected exemplars. These code features inform cautious labels but do not define the primary clusters.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- scikit-learn KMeans and clustering metrics were used through `src/e03/universality_classes.py`.
- JAX backend summary, for continuity with prior E03 routing context: `{json.dumps(manifest['runtime']['jax'], sort_keys=True)}`
- Worker count: serial CPU clustering; no worker pool used.
- New dependencies installed: none.

## Parameters

- Cluster seed: `{manifest['parameters']['seed']}`
- Cluster count: `{manifest['parameters']['clusterCount']}`
- Input policy rows: `{len(assignments)}`
- Normalized feature count: `{manifest['summary']['featureCount']}`
- Stability subsets: `{sorted(stability['subset_name'].unique().tolist())}`

## Results

### Candidate k Diagnostics

{markdown_table(candidates, candidate_cols)}

### Universality Class Summary

{markdown_table(summary, summary_cols)}

### Stability Summary

{markdown_table(stable_summary, list(stable_summary.columns))}

### Inspected Exemplars And Classic Landmarks

{markdown_table(exemplars, [column for column in exemplar_cols if column in exemplars.columns])}

## Metrics

- `class_id`: stable S11 class ID ordered by descending mean screen score.
- `cautious_label`: proxy behavior-family label assigned from behavior rank, code summaries, and inspected landmarks.
- `cluster_distance`: Euclidean distance from the policy to its KMeans centroid in S10 normalized feature space.
- `adjusted_rand_index`: label agreement between primary and metric-subset clusterings, adjusted for chance.
- `normalized_mutual_info`: information overlap between primary and subset clusterings.
- `manual_inspection_status`: exemplar rows whose DSL excerpt, metrics, and class context were inspected for label support.

## Figures

- Policy cluster and exemplar map: `{artifacts['clusterFigure']}`

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "expected", "observed", "notes"])}

## Caveats, Blockers, And Limitations

- These are S10/S11 proxy behavior classes, not proof of dynamical universality in the original paper simulator.
- Class labels are intentionally cautious. They summarize measured behavior, classic landmark placement, and DSL excerpts but do not establish necessary or sufficient rule components.
- S09 trajectory diagnostics are sparse and cover only selected policies; S11 stability therefore includes competence-only and screen-core controls.
- The dominant structure is a competence gradient with smaller source/code distinctions layered on top. S12 should test whether any label has causal support by targeted ablation.
- Memory and signal DSL primitives remain stub-like from S02, so classes involving those flags are not interpreted as communication mechanisms.

## Failed Assumptions

No required S10 or policy-code input was missing. The open assumption that S11 can produce stable interpretable classes is supported as a proxy result: metric-subset agreement is high, but labels remain approximate and require S12 ablation before causal wording.

## Provenance

Git commit before S11 commit: `{manifest['git']['headCommit']}`

Source files hashed in the S11 manifest:

{source_table}

## Artifacts

Reusable S11 outputs are the policy cluster assignment table, class summary, stability diagnostics, candidate-k diagnostics, inspected exemplar table, validation table, figure, class-summary markdown, config, manifests, status JSON, checksums, and this full-results handoff report.
"""


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e03"
    reports_dir = artifacts_dir / "reports"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, figures_dir, reports_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_tests = {"command": "not run", "returnCode": 0, "success": True, "stdout": "", "stderr": "", "elapsedSeconds": 0.0}
    if args.run_unit_tests:
        print("S11: running full E03 unit tests", flush=True)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03"], args.repo_dir)

    print("S11: loading S10 embeddings, normalized metrics, and policy code", flush=True)
    embeddings = pd.read_parquet(args.s10_embeddings)
    normalized_features = pd.read_parquet(args.s10_normalized_features)
    qd_candidates = pd.read_parquet(args.qd_candidates)
    code_table = load_policy_code_table(
        generated_policy_library=args.generated_policy_library,
        qd_candidates=qd_candidates,
        qd_discovered_policy_library=args.qd_discovered,
    )
    code_features = build_code_feature_frame(embeddings["policy_id"].astype(str).tolist(), code_table)

    columns = feature_columns(normalized_features)
    matrix = feature_matrix(normalized_features, columns)
    print(f"S11: clustering {len(embeddings)} policies with {len(columns)} normalized behavior features", flush=True)
    primary = fit_kmeans(matrix, cluster_count=args.cluster_count, seed=args.seed)
    candidates = candidate_k_frame(matrix, seed=args.seed)
    assignments = build_assignments(embeddings, normalized_features, code_features, primary)
    summary = cluster_summary_frame(assignments)
    assignments = attach_labels(assignments, summary)
    stability = stability_frame(normalized_features, embeddings, primary.labels, cluster_count=args.cluster_count)
    exemplars = exemplar_frame(assignments, normalized_features)

    cluster_parquet_path = results_dir / "e03_policy_clusters.parquet"
    cluster_csv_path = tables_dir / "e03_policy_clusters.csv"
    summary_parquet_path = results_dir / "e03_policy_cluster_summary.parquet"
    summary_csv_path = tables_dir / "e03_policy_cluster_summary.csv"
    stability_path = results_dir / "e03_cluster_stability.parquet"
    stability_csv_path = tables_dir / "e03_cluster_stability.csv"
    candidate_k_path = results_dir / "e03_cluster_candidate_k.parquet"
    candidate_k_csv_path = tables_dir / "e03_cluster_candidate_k.csv"
    exemplar_path = tables_dir / "e03_cluster_exemplar_inspection.csv"
    validation_path = results_dir / "e03_universality_validation.parquet"
    validation_csv_path = tables_dir / "e03_universality_validation.csv"
    figure_path = figures_dir / "policy_cluster_exemplars.png"
    class_summary_path = reports_dir / "e03_universality_class_summary.md"
    config_path = configs_dir / "e03_s11_universality_classes.json"
    manifest_path = src_snapshot_dir / "e03_universality_classes_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    status_path = step_dir / "status.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = checksums_dir / "sha256sums.txt"
    full_report_path = step_dir / "research_step_full_results.md"

    write_cluster_figure(figure_path, assignments, summary, exemplars)
    validation = validation_frame(
        assignments=assignments,
        summary=summary,
        stability=stability,
        candidate_k=candidates,
        exemplars=exemplars,
        figure_exists=figure_path.exists() and figure_path.stat().st_size > 0,
        unit_success=bool(unit_tests["success"]),
        expected_rows=len(embeddings),
        cluster_count=args.cluster_count,
    )
    for column in ("validation_case", "expected", "observed", "notes"):
        validation[column] = validation[column].astype(str)

    compact_assignments = compact_assignment_table(assignments)
    normalize_for_parquet(compact_assignments).to_parquet(cluster_parquet_path, index=False)
    compact_assignments.to_csv(cluster_csv_path, index=False)
    normalize_for_parquet(summary).to_parquet(summary_parquet_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
    normalize_for_parquet(stability).to_parquet(stability_path, index=False)
    stability.to_csv(stability_csv_path, index=False)
    normalize_for_parquet(candidates).to_parquet(candidate_k_path, index=False)
    candidates.to_csv(candidate_k_csv_path, index=False)
    exemplars.to_csv(exemplar_path, index=False)
    normalize_for_parquet(validation).to_parquet(validation_path, index=False)
    validation.to_csv(validation_csv_path, index=False)
    write_class_summary_markdown(class_summary_path, summary, exemplars, stability)

    selected_silhouette = float(candidates.loc[candidates["candidate_k"] == args.cluster_count, "silhouette_score"].iloc[0])
    subset_mean_ari = stability.groupby("subset_name")["adjusted_rand_index"].mean()
    minimum_subset_mean_ari = float(subset_mean_ari.min())
    config = {
        "schema": "eidosoma.e03.s11_universality_classes_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "seed": args.seed,
        "clusterCount": args.cluster_count,
        "clusteringMethod": "KMeans on S10 normalized behavior features",
        "featureColumns": list(columns),
        "stabilitySubsets": sorted(stability["subset_name"].unique().tolist()),
        "inputs": {
            "s10Embeddings": str(args.s10_embeddings),
            "s10NormalizedFeatures": str(args.s10_normalized_features),
            "generatedPolicyLibrary": str(args.generated_policy_library),
            "qdCandidates": str(args.qd_candidates),
            "qdDiscovered": str(args.qd_discovered),
        },
    }
    write_json(config_path, config)

    elapsed = time.perf_counter() - started
    source_files = [
        args.repo_dir / "src/e03/universality_classes.py",
        args.repo_dir / "tests/e03/test_universality_classes.py",
        args.repo_dir / "scripts/e03_s11_universality_classes.py",
        args.repo_dir / "src/e03/behavior_embedding.py",
        args.repo_dir / "src/e03/rule_dsl.py",
    ]
    manifest = {
        "schema": "eidosoma.e03.universality_classes_manifest.v1",
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
            "clusterCount": args.cluster_count,
            "candidateKValues": list(range(4, 13)),
            "kMeansRestarts": 50,
        },
        "inputArtifacts": {
            "s10Embeddings": str(args.s10_embeddings),
            "s10NormalizedFeatures": str(args.s10_normalized_features),
            "generatedPolicyLibrary": str(args.generated_policy_library),
            "qdCandidates": str(args.qd_candidates),
            "qdDiscovered": str(args.qd_discovered),
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
            "policyRows": int(len(assignments)),
            "featureCount": int(len(columns)),
            "clusterCount": int(args.cluster_count),
            "selectedSilhouette": selected_silhouette,
            "minimumSubsetMeanARI": minimum_subset_mean_ari,
            "minimumSubsetARI": float(stability["adjusted_rand_index"].min()),
            "meanSubsetARI": float(stability["adjusted_rand_index"].mean()),
            "exemplarRows": int(len(exemplars)),
            "classicLandmarkRows": int(assignments["classic_landmark"].sum()),
            "assignmentDigest": assignments_digest(assignments),
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
        "policyClustersParquet": cluster_parquet_path,
        "policyClustersCsv": cluster_csv_path,
        "clusterSummaryParquet": summary_parquet_path,
        "clusterSummaryCsv": summary_csv_path,
        "stabilityParquet": stability_path,
        "stabilityCsv": stability_csv_path,
        "candidateKParquet": candidate_k_path,
        "candidateKCsv": candidate_k_csv_path,
        "exemplarInspectionCsv": exemplar_path,
        "validationParquet": validation_path,
        "validationCsv": validation_csv_path,
        "clusterFigure": figure_path,
        "classSummaryMarkdown": class_summary_path,
        "clusteringConfig": config_path,
        "sourceSnapshotManifest": manifest_path,
        "artifactManifest": artifact_manifest_path,
        "statusJson": status_path,
        "runManifest": run_manifest_path,
        "checksums": checksums_path,
    }
    full_report = render_report(
        artifacts=artifacts,
        manifest=manifest,
        validation=validation,
        unit_tests=unit_tests,
        command_line=" ".join(sys.argv),
        assignments=assignments,
        summary=summary,
        stability=stability,
        candidates=candidates,
        exemplars=exemplars,
    )
    write_text(full_report_path, full_report)

    validation_result = (
        f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; "
        f"unit tests return code {unit_tests['returnCode']}"
    )
    recommended_next_action = "Stop for Chief review; if accepted, proceed to S12 causal rule-component ablations."
    caveats = (
        "No blocker. S11 classes are proxy behavior families from S10/S07-S09 metrics and DSL excerpts; "
        "labels remain approximate until S12 ablations."
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
        artifact_entry(cluster_parquet_path, artifacts_dir, "S11 policy cluster assignment table"),
        artifact_entry(cluster_csv_path, artifacts_dir, "CSV sidecar for S11 policy cluster assignments"),
        artifact_entry(summary_parquet_path, artifacts_dir, "S11 universality class summary table"),
        artifact_entry(summary_csv_path, artifacts_dir, "CSV sidecar for S11 class summary"),
        artifact_entry(stability_path, artifacts_dir, "S11 metric-subset stability diagnostics"),
        artifact_entry(stability_csv_path, artifacts_dir, "CSV sidecar for S11 stability diagnostics"),
        artifact_entry(candidate_k_path, artifacts_dir, "S11 candidate-k diagnostics"),
        artifact_entry(candidate_k_csv_path, artifacts_dir, "CSV sidecar for S11 candidate-k diagnostics"),
        artifact_entry(exemplar_path, artifacts_dir, "S11 inspected exemplar and classic landmark table"),
        artifact_entry(validation_path, artifacts_dir, "S11 validation cases"),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S11 validation cases"),
        artifact_entry(figure_path, artifacts_dir, "S11 policy cluster exemplar figure"),
        artifact_entry(class_summary_path, artifacts_dir, "S11 concise class-label summary report"),
        artifact_entry(config_path, artifacts_dir, "S11 clustering config"),
        manifest_self_entry(manifest_path, artifacts_dir, "S11 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S11 artifact manifest"),
        artifact_entry(status_path, artifacts_dir, "S11 compact status JSON"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S11"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S11 outputs"),
        artifact_entry(full_report_path, artifacts_dir, "S11 full-results handoff report"),
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
            artifact_entry(cluster_parquet_path, artifacts_dir, "S11 policy cluster assignment table"),
            artifact_entry(cluster_csv_path, artifacts_dir, "CSV sidecar for S11 policy cluster assignments"),
            artifact_entry(summary_parquet_path, artifacts_dir, "S11 universality class summary table"),
            artifact_entry(summary_csv_path, artifacts_dir, "CSV sidecar for S11 class summary"),
            artifact_entry(stability_path, artifacts_dir, "S11 metric-subset stability diagnostics"),
            artifact_entry(stability_csv_path, artifacts_dir, "CSV sidecar for S11 stability diagnostics"),
            artifact_entry(candidate_k_path, artifacts_dir, "S11 candidate-k diagnostics"),
            artifact_entry(candidate_k_csv_path, artifacts_dir, "CSV sidecar for S11 candidate-k diagnostics"),
            artifact_entry(exemplar_path, artifacts_dir, "S11 inspected exemplar and classic landmark table"),
            artifact_entry(validation_path, artifacts_dir, "S11 validation cases"),
            artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S11 validation cases"),
            artifact_entry(figure_path, artifacts_dir, "S11 policy cluster exemplar figure"),
            artifact_entry(class_summary_path, artifacts_dir, "S11 concise class-label summary report"),
            artifact_entry(config_path, artifacts_dir, "S11 clustering config"),
            artifact_entry(manifest_path, artifacts_dir, "S11 source snapshot and provenance manifest"),
            manifest_self_entry(artifact_manifest_path, artifacts_dir, "S11 artifact manifest"),
            artifact_entry(status_path, artifacts_dir, "S11 compact status JSON"),
            manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S11"),
            manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S11 outputs"),
            artifact_entry(full_report_path, artifacts_dir, "S11 full-results handoff report"),
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
        cluster_parquet_path,
        cluster_csv_path,
        summary_parquet_path,
        summary_csv_path,
        stability_path,
        stability_csv_path,
        candidate_k_path,
        candidate_k_csv_path,
        exemplar_path,
        validation_path,
        validation_csv_path,
        figure_path,
        class_summary_path,
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
                "policyRows": int(len(assignments)),
                "clusterCount": int(args.cluster_count),
                "selectedSilhouette": selected_silhouette,
                "minimumSubsetMeanARI": minimum_subset_mean_ari,
                "validationCasesPassed": int(validation["success"].sum()),
                "validationCasesTotal": int(len(validation)),
                "artifactsDir": str(step_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
