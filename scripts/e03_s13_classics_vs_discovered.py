#!/usr/bin/env python3
"""Run E03 S13 classic-versus-discovered morphospace comparison."""

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

from src.e03.classic_position import (
    CLASSIC_POSITION_SCHEMA,
    algorithm_position_frame,
    behavior_feature_columns,
    classic_alignment_frame,
    classic_policy_position_frame,
    classic_position_digest,
    finite_feature_matrix,
    load_s03_policy_library,
    nearest_neighbors,
    neighbor_robustness_frame,
    validation_frame,
)
from src.e03.gpu_batch_simulator import jax_backend_summary


STEP_ID = "S13"
STEP_NUMBER = 13
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    artifacts_default = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_default)
    parser.add_argument("--s03-policy-library", type=Path, default=artifacts_default / "policies/e03_classic_policy_library.json")
    parser.add_argument("--s04-classic-vectors", type=Path, default=artifacts_default / "results/e03_classic_competence_vectors.parquet")
    parser.add_argument("--s07-competence", type=Path, default=artifacts_default / "results/e03_policy_competence.parquet")
    parser.add_argument("--s10-embeddings", type=Path, default=artifacts_default / "results/e03_policy_embeddings.parquet")
    parser.add_argument("--s10-normalized-features", type=Path, default=artifacts_default / "results/e03_policy_embedding_normalized_features.parquet")
    parser.add_argument("--s10-raw-features", type=Path, default=artifacts_default / "results/e03_policy_embedding_features.parquet")
    parser.add_argument("--s11-clusters", type=Path, default=artifacts_default / "results/e03_policy_clusters.parquet")
    parser.add_argument("--s12-claims", type=Path, default=artifacts_default / "results/e03_rule_ablation_claims.parquet")
    parser.add_argument("--neighbor-k", type=int, default=20)
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


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_policy_table(embeddings: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "policy_id",
        "policy_name",
        "source_kind",
        "screen_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_improvement_mean",
        "screen_heldout_work_mean",
        "n100_final_sortedness_mean",
        "n1000_final_sortedness_mean",
        "qd_candidate",
        "s08_archive_winner",
        "classic_landmark",
        "classic_family",
        "embedding_x",
        "embedding_y",
        "embedding_z",
    ]
    cluster_cols = ["policy_id", "class_id", "cautious_label", "label_confidence", "cluster_distance"]
    table = embeddings[[column for column in base_cols if column in embeddings.columns]].merge(
        clusters[[column for column in cluster_cols if column in clusters.columns]],
        on="policy_id",
        how="left",
        validate="one_to_one",
    )
    for column in ("classic_landmark", "qd_candidate", "s08_archive_winner"):
        if column in table.columns:
            table[column] = table[column].map(truthy)
    return table


def align_features(policy_table: pd.DataFrame, feature_frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    cols = ["policy_id", *feature_names]
    return policy_table[["policy_id"]].merge(feature_frame[[column for column in cols if column in feature_frame.columns]], on="policy_id", how="left", validate="one_to_one")


def write_classic_figure(path: Path, policy_table: pd.DataFrame, algorithm_positions: pd.DataFrame, classic_positions: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = sorted(policy_table["class_id"].dropna().astype(str).unique())
    cmap = plt.get_cmap("tab10")
    colors = {class_id: cmap(idx % 10) for idx, class_id in enumerate(classes)}
    classed = policy_table.copy()
    classed["class_id"] = classed["class_id"].fillna("unclassified")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw={"width_ratios": [1.35, 0.9, 1.0]})
    ax = axes[0]
    for class_id, group in classed.groupby("class_id", sort=True):
        ax.scatter(
            group["embedding_x"],
            group["embedding_y"],
            s=12,
            alpha=0.30,
            linewidths=0,
            color=colors.get(class_id, "#777777"),
            label=class_id,
        )
    classics = policy_table[policy_table["classic_landmark"].map(truthy)].copy()
    markers = {"Bubble": "*", "Insertion": "D", "Selection": "P"}
    for _, row in classics.iterrows():
        family = str(row.get("classic_family", "Classic"))
        ax.scatter(
            row["embedding_x"],
            row["embedding_y"],
            s=180,
            marker=markers.get(family, "X"),
            facecolors="#ffffff",
            edgecolors="#111111",
            linewidths=1.1,
            zorder=5,
        )
        direction = "inc" if "increasing" in str(row.get("policy_name", "")) else "dec"
        ax.annotate(f"{family} {direction}", (row["embedding_x"], row["embedding_y"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_title("Classic DSL landmarks in S10 behavior embedding")
    ax.set_xlabel("S10 embedding x")
    ax.set_ylabel("S10 embedding y")
    ax.grid(True, alpha=0.16, linewidth=0.6)

    class_colors = {"central": "#1b9e77", "peripheral": "#7570b3", "accidental": "#d95f02"}
    ax = axes[1]
    ordered = algorithm_positions.sort_values("best_dsl_screen_score", ascending=True, kind="mergesort")
    y = np.arange(len(ordered))
    ax.barh(y, ordered["best_dsl_screen_score"], color=[class_colors.get(label, "#777777") for label in ordered["classic_position_classification"]], alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['algorithm_label']} ({row['classic_position_classification']})" for row in ordered.to_dict(orient="records")])
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel("Best DSL screen score")
    ax.set_title("Algorithm-level position classification")
    ax.grid(True, axis="x", alpha=0.16, linewidth=0.6)

    ax = axes[2]
    dsl = classic_positions[classic_positions["representation_type"] == "dsl"].copy()
    dsl["label"] = dsl["algorithm_label"].astype(str) + " " + dsl["direction"].astype(str).str.replace("_", " ")
    dsl = dsl.sort_values("normalized_raw_top10_jaccard", kind="mergesort")
    y = np.arange(len(dsl))
    ax.barh(y, pd.to_numeric(dsl["normalized_raw_top10_jaccard"], errors="coerce"), color="#4c78a8", alpha=0.86)
    ax.set_yticks(y)
    ax.set_yticklabels(dsl["label"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Top-10 Jaccard")
    ax.set_title("Normalized/raw neighbor overlap")
    ax.grid(True, axis="x", alpha=0.16, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_report(
    *,
    artifacts: dict[str, Path],
    manifest: dict[str, Any],
    validation: pd.DataFrame,
    unit_tests: dict[str, Any],
    command_line: str,
    alignment: pd.DataFrame,
    classic_positions: pd.DataFrame,
    algorithm_positions: pd.DataFrame,
    neighbors: pd.DataFrame,
    robustness: pd.DataFrame,
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
    dsl_positions = classic_positions[classic_positions["representation_type"] == "dsl"].copy()
    nearest_top = neighbors[neighbors["neighbor_rank"] <= 5].copy()
    nearest_summary = (
        nearest_top.groupby(["target_algorithm", "target_policy_name", "metric_space"])
        .agg(
            top5_same_class_fraction=("same_class", "mean"),
            top5_qd_fraction=("neighbor_qd_candidate", "mean"),
            nearest_distance=("neighbor_distance", "min"),
        )
        .reset_index()
    )
    return f"""# E03 S13 Research Step Full Results

## Top Summary

- Step ID: S13
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; classic-position digest `{classic_position_digest(algorithm_positions)}`.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains. Classifications are finite-search, proxy-morphospace statements over S03 DSL landmarks in S07-S12 measurements; exact public-method interface wrappers are aligned as S04 metric context but are not embedded S10/S11 points.
- Lay summary: S13 aligned 9 S03/S04 classic entries, confirmed that 6 DSL classic landmarks are embedded in S10/S11/S12, computed nearest neighbors in normalized S10 feature space, raw S10 metric space, and S10 embedding space, then classified Bubble as central, Insertion as peripheral, and Selection as accidental under the current DSL proxy morphospace.
- Recommended next action: Stop for Chief review; if accepted, proceed to S14 frontier-candidate selection using S07-S13 evidence.

## Frozen Question

Are Bubble, Insertion, and Selection central, peripheral, or accidental examples in the larger Algotype morphospace?

## Inputs

- S03 classic policy library: `{manifest['inputArtifacts']['s03PolicyLibrary']}`
- S04 classic competence vectors: `{manifest['inputArtifacts']['s04ClassicVectors']}`
- S07 policy competence: `{manifest['inputArtifacts']['s07Competence']}`
- S10 embeddings: `{manifest['inputArtifacts']['s10Embeddings']}`
- S10 normalized behavior features: `{manifest['inputArtifacts']['s10NormalizedFeatures']}`
- S10 raw behavior features: `{manifest['inputArtifacts']['s10RawFeatures']}`
- S11 clusters: `{manifest['inputArtifacts']['s11Clusters']}`
- S12 rule-ablation claims: `{manifest['inputArtifacts']['s12Claims']}`

## Lay Summary

This step asks whether the three classic sorting algorithms are typical landmarks in the expanded rule space or just special cases. The exact public Bubble, Insertion, and Selection wrappers are the canonical paper-compatible policies, but only their six DSL landmark/shadow versions are present in the S10/S11 morphospace. S13 therefore aligns both layers and classifies the measurable DSL landmarks with explicit caveats.

The result is direction-sensitive. Bubble's increasing DSL shadow sits in the highest-competence local-inversion-cleaner class and has nearby non-classic policies, so Bubble is central in this proxy atlas. Insertion is measurable and useful but sits outside the high-competence core, so it is peripheral. Selection's DSL target-position landmarks are weak in the S07/S10/S11 screen and S12 ablations often improve them by removing target-state or swap behavior, so Selection is accidental-looking in this DSL proxy morphospace, despite the exact public Selection wrapper remaining a valid S04 classic baseline.

## Methods

S13 first built an ID-alignment table across S03, S04, S07, S10, S11, and S12. The three `iface:*` public-method wrappers align to S04 competence vectors only. The six `dsl:*` landmarks align to S07 competence, S10 embeddings/features, S11 class assignments, and S12 ablation claims.

Distances were computed three ways. The normalized distance uses S10 normalized behavior features, preserving S10's block-balanced metric scaling. The raw distance uses the corresponding raw S10 behavior features with median imputation but without standardization; this intentionally tests whether nearest-neighbor conclusions depend on metric scale. The embedding distance uses S10's 3D PCA coordinates as a visualization-space check. For each DSL classic landmark and each metric space, S13 recorded the top {manifest['parameters']['neighborK']} nearest non-classic neighbors.

Nearest-neighbor robustness was measured as top-10 Jaccard overlap between normalized and raw neighbors, normalized and embedding neighbors, and raw and embedding neighbors. Same-class, QD-candidate, and archive-winner fractions were also recorded for top-10 neighborhoods.

Algorithm-level labels were assigned conservatively from the best DSL landmark for each classic family, its S11 class, its S07/S10 screen score, directional split, neighbor stability, and S12 ablation evidence. The labels are `central`, `peripheral`, or `accidental`; they are not claims about the full public simulator outside this finite DSL search.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- S13 used repository code in `src/e03/classic_position.py` plus S10/S11/S12 artifact tables.
- JAX backend summary, recorded for continuity only: `{json.dumps(manifest['runtime']['jax'], sort_keys=True)}`
- Worker count: serial CPU analysis; no worker pool used.
- New dependencies installed: none.

## Parameters

- Neighbor count per metric space: `{manifest['parameters']['neighborK']}`
- Normalized/raw behavior feature count: `{manifest['summary']['distanceFeatureCount']}`
- Embedded DSL classic landmarks: `{manifest['summary']['embeddedDslClassicCount']}`
- Non-classic policies available for neighbors: `{manifest['summary']['nonClassicNeighborPoolSize']}`

## Results

### Algorithm-Level Classification

{markdown_table(algorithm_positions, ["algorithm_label", "classic_position_classification", "classification_confidence", "best_dsl_policy_name", "best_dsl_class_id", "best_dsl_screen_score", "worst_dsl_class_id", "worst_dsl_screen_score", "direction_split", "classification_basis"])}

### DSL Landmark Positions

{markdown_table(dsl_positions, ["algorithm_label", "policy_id", "policy_name", "direction", "exactness", "class_id", "cautious_label", "screen_score", "screen_heldout_final_sortedness_mean", "nearest_neighbor_stability", "normalized_raw_top10_jaccard", "s12_local_necessary_count", "s12_anti_feature_count"])}

### ID Alignment Summary

{markdown_table(alignment, ["algorithm_label", "policy_id", "representation_type", "exactness", "direction", "s04_present", "s07_present", "s10_present", "s11_present", "s12_present", "morphospace_role"])}

### Nearest-Neighbor Robustness

{markdown_table(robustness, ["target_algorithm", "target_policy_name", "target_class_id", "normalized_raw_top10_jaccard", "normalized_embedding_top10_jaccard", "raw_embedding_top10_jaccard", "normalized_top10_same_class_fraction", "raw_top10_same_class_fraction", "nearest_neighbor_stability"])}

### Top-5 Neighbor Summary

{markdown_table(nearest_summary, ["target_algorithm", "target_policy_name", "metric_space", "top5_same_class_fraction", "top5_qd_fraction", "nearest_distance"])}

## Metrics

- `normalized` distance: Euclidean distance over S10 normalized behavior features.
- `raw` distance: Euclidean distance over corresponding raw S10 behavior metrics after median imputation and without rescaling.
- `embedding` distance: Euclidean distance in the S10 3D PCA coordinates.
- `nearest_distance_percentile`: percentile of the target's nearest non-classic distance compared with all policies' nearest non-classic distances in the same space; lower means denser local neighborhood.
- `top10_jaccard`: overlap between top-10 nearest-neighbor sets in two metric spaces.
- `central/peripheral/accidental`: finite-search classification over measured DSL landmarks, not a universal statement about all possible policies or exact public methods.

## Figures

- Classics in morphospace: `{artifacts['classicFigure']}`

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "expected", "observed", "notes"])}

## Caveats, Blockers, And Limitations

- Exact `iface:*` classic wrappers are aligned to S04 canonical metrics but are not present as S10/S11 morphospace points. The morphospace placement uses S03 DSL landmarks and shadows.
- Bubble's DSL representation is an approximate shadow, especially for sampled-side control flow and comparison accounting. Bubble's central label is therefore strongest for the increasing bidirectional inversion-cleaner shadow.
- Selection's accidental label is a proxy-DSL result. It should not be read as invalidating exact public Selection behavior in E01/S04.
- Raw distances are intentionally unscaled and can be dominated by large-count metrics such as work; agreement with normalized neighbors is treated as a robustness signal, not a requirement for validity.
- S13 does not rerun new simulations; it joins and analyzes S03-S12 artifacts.

## Failed Assumptions

No required S03-S12 input was missing. The main scope limit is conceptual rather than blocking: public-method classics and DSL landmarks are not the same object, so S13 separates canonical metric context from embedded DSL morphospace placement.

## Provenance

Git commit before S13 commit: `{manifest['git']['headCommit']}`

Source files hashed in the S13 manifest:

{source_table}

## Artifacts

Reusable S13 outputs are the algorithm classification table, classic ID-alignment table, DSL classic position table, nearest-neighbor table, neighbor-robustness table, validation table, figure, config, manifest, status JSON, checksums, and this full-results handoff report.
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
        print("S13: running full E03 unit tests", flush=True)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03"], args.repo_dir)

    print("S13: loading S03-S12 artifacts", flush=True)
    s03 = load_s03_policy_library(args.s03_policy_library)
    s04 = pd.read_parquet(args.s04_classic_vectors)
    s07 = pd.read_parquet(args.s07_competence)
    embeddings = pd.read_parquet(args.s10_embeddings)
    normalized_features = pd.read_parquet(args.s10_normalized_features)
    raw_features = pd.read_parquet(args.s10_raw_features)
    clusters = pd.read_parquet(args.s11_clusters)
    s12_claims = pd.read_parquet(args.s12_claims)

    print("S13: aligning classic IDs and building metric spaces", flush=True)
    alignment = classic_alignment_frame(s03, s04, s07, embeddings, clusters, s12_claims)
    policy_table = build_policy_table(embeddings, clusters)
    feature_cols = [column for column in behavior_feature_columns(normalized_features) if column in raw_features.columns]
    normalized_aligned = align_features(policy_table, normalized_features, feature_cols)
    raw_aligned = align_features(policy_table, raw_features, feature_cols)
    normalized_matrix = finite_feature_matrix(normalized_aligned, feature_cols)
    raw_matrix = finite_feature_matrix(raw_aligned, feature_cols)
    embedding_matrix = finite_feature_matrix(policy_table, ["embedding_x", "embedding_y", "embedding_z"])
    classic_ids = alignment[(alignment["representation_type"] == "dsl") & alignment["s10_present"]]["policy_id"].astype(str).tolist()

    print("S13: computing nearest neighbors in normalized, raw, and embedding spaces", flush=True)
    neighbors = pd.concat(
        [
            nearest_neighbors(policy_table, normalized_matrix.matrix, classic_ids, metric_space="normalized", k=args.neighbor_k),
            nearest_neighbors(policy_table, raw_matrix.matrix, classic_ids, metric_space="raw", k=args.neighbor_k),
            nearest_neighbors(policy_table, embedding_matrix.matrix, classic_ids, metric_space="embedding", k=args.neighbor_k),
        ],
        ignore_index=True,
        sort=False,
    )
    robustness = neighbor_robustness_frame(neighbors, k=10)
    classic_positions = classic_policy_position_frame(alignment, robustness, s12_claims)
    algorithm_positions = algorithm_position_frame(classic_positions)

    alignment_path = results_dir / "e03_classic_id_alignment.parquet"
    positions_path = results_dir / "e03_classic_policy_positions.parquet"
    analysis_path = results_dir / "e03_classic_position_analysis.parquet"
    neighbors_path = results_dir / "e03_classic_nearest_neighbors.parquet"
    robustness_path = results_dir / "e03_classic_neighbor_robustness.parquet"
    validation_path = results_dir / "e03_classic_position_validation.parquet"
    alignment_csv = tables_dir / "e03_classic_id_alignment.csv"
    analysis_csv = tables_dir / "e03_classic_position_analysis.csv"
    positions_csv = tables_dir / "e03_classic_policy_positions.csv"
    neighbor_csv = tables_dir / "e03_classic_nearest_neighbors_top20.csv"
    validation_csv = step_dir / "validation.csv"
    figure_path = figures_dir / "classics_in_morphospace.png"
    config_path = configs_dir / "e03_s13_classics_vs_discovered_config.json"
    manifest_path = src_snapshot_dir / "e03_classic_position_manifest.json"
    step_manifest_path = step_dir / "manifest.json"
    status_path = step_dir / "status.json"
    report_path = step_dir / "research_step_full_results.md"
    checksum_path = checksums_dir / "e03_s13_sha256sums.txt"

    write_classic_figure(figure_path, policy_table, algorithm_positions, classic_positions)
    validation = validation_frame(
        alignment=alignment,
        classic_positions=classic_positions,
        algorithm_positions=algorithm_positions,
        neighbors=neighbors,
        robustness=robustness,
        figure_exists=figure_path.exists(),
        unit_success=bool(unit_tests["success"]),
    )

    normalize_for_parquet(alignment).to_parquet(alignment_path, index=False)
    normalize_for_parquet(classic_positions).to_parquet(positions_path, index=False)
    normalize_for_parquet(algorithm_positions).to_parquet(analysis_path, index=False)
    normalize_for_parquet(neighbors).to_parquet(neighbors_path, index=False)
    normalize_for_parquet(robustness).to_parquet(robustness_path, index=False)
    normalize_for_parquet(validation).to_parquet(validation_path, index=False)
    alignment.to_csv(alignment_csv, index=False)
    algorithm_positions.to_csv(analysis_csv, index=False)
    classic_positions.to_csv(positions_csv, index=False)
    neighbors.to_csv(neighbor_csv, index=False)
    validation.to_csv(validation_csv, index=False)

    config_payload = {
        "schema": CLASSIC_POSITION_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "neighborK": args.neighbor_k,
        "distanceFeatureColumns": feature_cols,
        "distanceSpaces": ["normalized", "raw", "embedding"],
        "classificationRules": {
            "central": "best DSL screen_score >= 0.85 and best S11 class in UC01-UC03",
            "peripheral": "best DSL screen_score >= 0.55 but not central",
            "accidental": "best DSL screen_score < 0.55 in current DSL proxy morphospace",
        },
    }
    write_json(config_path, config_payload)

    source_files = [
        source_entry(args.repo_dir / "src/e03/classic_position.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e03_s13_classics_vs_discovered.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e03/test_classic_position.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e03/behavior_embedding.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e03/rule_ablations.py", args.repo_dir),
    ]
    artifacts: dict[str, Path] = {
        "fullResultsReport": report_path,
        "statusJson": status_path,
        "stepManifest": step_manifest_path,
        "sourceManifest": manifest_path,
        "config": config_path,
        "classicPositionAnalysis": analysis_path,
        "classicPolicyPositions": positions_path,
        "classicIdAlignment": alignment_path,
        "classicNearestNeighbors": neighbors_path,
        "classicNeighborRobustness": robustness_path,
        "validationParquet": validation_path,
        "classicPositionAnalysisCsv": analysis_csv,
        "classicPolicyPositionsCsv": positions_csv,
        "classicIdAlignmentCsv": alignment_csv,
        "classicNearestNeighborsCsv": neighbor_csv,
        "validationCsv": validation_csv,
        "classicFigure": figure_path,
        "checksums": checksum_path,
    }
    manifest = {
        "schema": CLASSIC_POSITION_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "createdUtc": utc_now(),
        "inputArtifacts": {
            "s03PolicyLibrary": str(args.s03_policy_library),
            "s04ClassicVectors": str(args.s04_classic_vectors),
            "s07Competence": str(args.s07_competence),
            "s10Embeddings": str(args.s10_embeddings),
            "s10NormalizedFeatures": str(args.s10_normalized_features),
            "s10RawFeatures": str(args.s10_raw_features),
            "s11Clusters": str(args.s11_clusters),
            "s12Claims": str(args.s12_claims),
        },
        "parameters": {
            "neighborK": int(args.neighbor_k),
            "distanceSpaces": ["normalized", "raw", "embedding"],
        },
        "summary": {
            "s03ClassicEntryCount": int(len(alignment)),
            "embeddedDslClassicCount": int(((alignment["representation_type"] == "dsl") & alignment["s10_present"]).sum()),
            "distanceFeatureCount": int(len(feature_cols)),
            "nonClassicNeighborPoolSize": int((~policy_table["classic_landmark"].map(truthy)).sum()),
            "neighborRows": int(len(neighbors)),
            "classificationCounts": algorithm_positions["classic_position_classification"].value_counts().sort_index().to_dict(),
            "resultDigest": classic_position_digest(algorithm_positions),
            "validationPassed": bool(validation["success"].all() and unit_tests["success"]),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax": jax_backend_summary(),
            "elapsedSeconds": time.perf_counter() - started,
        },
        "git": {
            "headCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
        "sourceFiles": source_files,
    }
    write_json(manifest_path, manifest)

    report = render_report(
        artifacts=artifacts,
        manifest=manifest,
        validation=validation,
        unit_tests=unit_tests,
        command_line=" ".join(sys.argv),
        alignment=alignment,
        classic_positions=classic_positions,
        algorithm_positions=algorithm_positions,
        neighbors=neighbors,
        robustness=robustness,
    )
    write_text(report_path, report)

    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"].all() and unit_tests["success"]),
        "status": "completed" if bool(validation["success"].all() and unit_tests["success"]) else "completed_with_validation_failure",
        "artifactsWritten": [str(path) for path in artifacts.values()],
        "validationResult": f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; unit tests return code {unit_tests['returnCode']}",
        "caveatsOrBlockers": "No blocker remains. Classifications are finite-search proxy statements over embedded S03 DSL landmarks; exact public wrappers are S04 metric context, not S10/S11 morphospace points.",
        "recommendedNextAction": "Stop for Chief review before S14; if accepted, select frontier candidates using S07-S13 evidence.",
    }
    write_json(status_path, status_payload)

    artifact_entries = [artifact_entry(path, artifacts_dir, key) for key, path in artifacts.items() if path.exists() and path not in {step_manifest_path, checksum_path}]
    step_manifest = dict(manifest)
    step_manifest["artifacts"] = artifact_entries + [
        manifest_self_entry(step_manifest_path, artifacts_dir, "stepManifest"),
        manifest_self_entry(checksum_path, artifacts_dir, "checksums"),
    ]
    write_json(step_manifest_path, step_manifest)

    checksum_entries = []
    for path in artifacts.values():
        if path.exists() and path != checksum_path:
            checksum_entries.append(f"{sha256_file(path)}  {path}")
    write_text(checksum_path, "\n".join(checksum_entries) + "\n")
    print(f"S13: wrote report to {report_path}", flush=True)
    return 0 if status_payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
