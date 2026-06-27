#!/usr/bin/env python3
"""Execute E03 S11: identify empirical universality classes.

S11 consumes completed S10 behavior features, embeddings, and nearest-neighbor
records together with S07-S09 competence, QD, and phase-boundary anchors.  It
clusters policies into empirical computational families, documents robustness,
classic/null/elite placements, exemplars, and caveats, then stops before S12.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from morphospace import (  # noqa: E402
    UNIVERSALITY_CLASS_VERSION,
    identify_universality_classes,
    validate_universality_outputs,
)
from morphospace.competence import canonical_json  # noqa: E402


EXPERIMENT_ID = "E03"
STEP_ID = "S11"
STEP_NUMBER = 11
TITLE = "Identify universality classes"
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"

CANDIDATE_K = tuple(range(5, 13))
PRIMARY_SEED = 0
REDUCTION_COMPONENTS = 24
EXEMPLARS_PER_CLASS = 5

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_DIR = ARTIFACTS_DIR / "research_steps" / STEP_ID
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = STEP_DIR / "figures"
SHARED_CODE_DIR = ARTIFACTS_DIR / "code" / "e03_universality_classes"

S10_FEATURES_PATH = RESULTS_DIR / "e03_s10_embedding_features.parquet"
S10_NORMALIZED_PATH = ARTIFACTS_DIR / "research_steps" / "S10" / "normalized_feature_matrix.parquet"
S10_EMBEDDINGS_PATH = RESULTS_DIR / "e03_policy_embeddings.parquet"
S10_NEIGHBORS_PATH = RESULTS_DIR / "e03_s10_embedding_neighbors.parquet"
S07_COMPETENCE_PATH = RESULTS_DIR / "e03_morphospace_metrics.parquet"
S08_ELITES_PATH = RESULTS_DIR / "e03_qd_elites.parquet"
S09_BOUNDARIES_PATH = RESULTS_DIR / "e03_phase_boundaries.parquet"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def table_ready(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in prepared.columns:
        series = prepared[column]
        if series.map(lambda value: isinstance(value, (dict, list, tuple))).any():
            prepared[column] = series.map(
                lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple))
                else value
            )
    return prepared


def write_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> list[Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out = table_ready(df)
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists() and path.is_file():
            rows.append({"path": str(path), "sizeBytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return rows


def run_command(command: list[str], *, cwd: Path = REPO_ROOT, timeout: int = 900) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returnCode": int(completed.returncode),
        "success": completed.returncode == 0,
        "runtimeSeconds": time.perf_counter() - started,
        "output": completed.stdout,
    }


def get_git_metadata() -> dict[str, Any]:
    def read(command: list[str]) -> str:
        result = subprocess.run(command, cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return result.stdout.strip()

    return {
        "branch": read(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": read(["git", "rev-parse", "HEAD"]),
        "statusShort": read(["git", "status", "--short"]),
        "remote": read(["git", "remote", "get-url", "origin"]),
    }


def load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"success": False, "missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs() -> dict[str, pd.DataFrame]:
    required_paths = {
        "s10_features": S10_FEATURES_PATH,
        "s10_normalized": S10_NORMALIZED_PATH,
        "s10_embeddings": S10_EMBEDDINGS_PATH,
        "s10_neighbors": S10_NEIGHBORS_PATH,
        "s07_competence": S07_COMPETENCE_PATH,
        "s08_elites": S08_ELITES_PATH,
        "s09_boundaries": S09_BOUNDARIES_PATH,
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"S11 requires completed S07-S10 artifacts; missing: {missing}")
    for step in ["S07", "S08", "S09", "S10"]:
        status = load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json")
        if not status.get("success"):
            raise RuntimeError(f"S11 requires successful {step}; got {status}")
    return {key: pd.read_parquet(path) for key, path in required_paths.items()}


def write_taxonomy_report(
    *,
    path: Path,
    validation_result: str,
    artifacts_written: list[Path],
    class_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    exemplars: pd.DataFrame,
    placements: pd.DataFrame,
    neighbor_alignment: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> None:
    robustness_summary = (
        robustness[robustness["success"].astype(bool)]
        .groupby("comparisonType")[["adjustedRandIndex", "normalizedMutualInfo", "silhouetteScore"]]
        .mean()
        .round(4)
        .reset_index()
        if not robustness.empty
        else pd.DataFrame()
    )
    role_counts = {}
    if not placements.empty:
        for role in ["classic", "null", "s08_elite"]:
            role_counts[role] = int(placements["placementRole"].astype(str).str.contains(role, regex=False).sum())
    rows = [
        "# S11 Universality Taxonomy Report",
        "",
        "- Research step ID: S11",
        f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
        f"- Artifacts written: {len(artifacts_written)} files, including `{RESULTS_DIR / 'e03_universality_classes.parquet'}`, `{RESULTS_DIR / 'e03_s11_policy_class_assignments.parquet'}`, and `{FIGURES_DIR}/`",
        f"- Validation result: {validation_result}",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        (
            "- Lay summary: S11 groups policies into empirical behavior families using the completed S10 feature matrix, "
            "embedding neighborhoods, S07-S09 competence evidence, QD elite metadata, and phase-boundary records."
        ),
        "",
        "## Method Boundary",
        "",
        "- Primary clustering uses S10 normalized behavior features reduced by bounded PCA, then deterministic k-means.",
        "- Robustness is documented across repeated seeds, feature subsets, and S10 PCA/spectral/MDS coordinate spaces.",
        "- Class names are descriptive computational labels over observed proxy behavior, not mathematical universality claims.",
        "",
        "## Empirical Classes",
        "",
        "| Class | Label | Policies | Primary Role Counts | Completion | Sortedness | Oscillation | Interpretation |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in class_summary.iterrows():
        rows.append(
            "| "
            f"`{row['className']}` | `{row['classLabel']}` | {int(row['policyCount'])} | "
            f"`{row['primaryRoleCountsJson']}` | {float(row['meanCompletionSuccess']):.3f} | "
            f"{float(row['meanSortednessScore']):.3f} | {float(row['meanOscillationProxy']):.3f} | "
            f"{str(row['classInterpretation']).replace('|', '/')} |"
        )
    rows.extend(
        [
            "",
            "## Robustness",
            "",
            f"- Successful robustness rows: {int(robustness['success'].sum()) if not robustness.empty else 0}/{len(robustness)}",
            f"- Mean same-class S10 nearest-neighbor rate: {float(neighbor_alignment.loc[neighbor_alignment['universalityClassId'].eq('ALL'), 'meanSameClassNeighborRate'].iloc[0]):.3f}"
            if not neighbor_alignment.empty and neighbor_alignment["universalityClassId"].eq("ALL").any()
            else "- Mean same-class S10 nearest-neighbor rate: not available",
            "",
        ]
    )
    if not robustness_summary.empty:
        rows.extend(["| Comparison Type | Mean ARI | Mean NMI | Mean Silhouette |", "| --- | ---: | ---: | ---: |"])
        for _, row in robustness_summary.iterrows():
            rows.append(
                f"| `{row['comparisonType']}` | {float(row['adjustedRandIndex']):.3f} | "
                f"{float(row['normalizedMutualInfo']):.3f} | {float(row['silhouetteScore']):.3f} |"
            )
        rows.append("")
    rows.extend(
        [
            "## Classic, Null, and Elite Placements",
            "",
            f"- Placement counts: `{canonical_json(role_counts)}`",
            "",
            "| Role | Policy | Class | Lineage | Distance |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for _, row in placements.sort_values(["placementRole", "distanceToCentroid", "policyId"], kind="mergesort").head(30).iterrows():
        rows.append(
            f"| `{row['placementRole']}` | `{row['policyId']}` | `{row['className']}` | "
            f"`{str(row['lineageId'])[:60]}` | {float(row['distanceToCentroid']):.3f} |"
        )
    rows.extend(["", "## Class Exemplars", "", "| Class | Rank | Policy | Role | Lineage | Neighbors |", "| --- | ---: | --- | --- | --- | --- |"])
    for _, row in exemplars.sort_values(["universalityClassId", "exemplarRank"], kind="mergesort").iterrows():
        neighbors = json.loads(row["nearestNeighborsJson"]) if isinstance(row.get("nearestNeighborsJson"), str) else []
        neighbor_ids = ",".join(str(item.get("neighborPolicyId")) for item in neighbors[:3])
        rows.append(
            f"| `{row['className']}` | {int(row['exemplarRank'])} | `{row['policyId']}` | `{row['primaryRole']}` | "
            f"`{str(row['lineageId'])[:60]}` | `{neighbor_ids}` |"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def render_summary(
    *,
    validation_result: str,
    artifacts_written: list[Path],
    class_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    placements: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    class_counts = class_summary.set_index("className")["policyCount"].astype(int).to_dict()
    robustness_brief = (
        robustness[robustness["success"].astype(bool)]
        .groupby("comparisonType")[["adjustedRandIndex", "normalizedMutualInfo"]]
        .mean()
        .round(4)
        .to_dict()
        if not robustness.empty
        else {}
    )
    placement_counts = {}
    if not placements.empty:
        for role in ["classic", "null", "s08_elite"]:
            placement_counts[role] = int(placements["placementRole"].astype(str).str.contains(role, regex=False).sum())
    return "\n".join(
        [
            "# S11 Universality Class Summary",
            "",
            "- Research step ID: S11",
            f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
            f"- Artifacts written: {len(artifacts_written)} files, including `{RESULTS_DIR / 'e03_universality_classes.parquet'}`, `{STEP_DIR / 'universality_taxonomy_report.md'}`, and `{FIGURES_DIR}/`",
            f"- Validation result: {validation_result}",
            f"- Caveats or blockers: {'; '.join(caveats)}",
            f"- Recommended next action: {recommended_next_action}",
            (
                "- Lay summary: S11 turns the S10 behavior map into a bounded taxonomy of empirical computational policy classes, "
                "with robustness checks and representative policies for Chief Scientist review."
            ),
            "",
            "## Key Counts",
            "",
            f"- Classes: {len(class_summary)}",
            f"- Policy counts by class: `{canonical_json(class_counts)}`",
            f"- Classic/null/elite placement counts: `{canonical_json(placement_counts)}`",
            f"- Robustness summary: `{canonical_json(robustness_brief)}`",
        ]
    ) + "\n"


def render_validation_report(
    validation_df: pd.DataFrame,
    artifacts_written: list[Path],
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    rows = [
        "# S11 Validation Report",
        "",
        "- Research step ID: S11",
        f"- Completion status: {STATUS}",
        f"- Artifacts written: {len(artifacts_written)} files",
        f"- Validation result: {'passed' if validation_df['success'].all() else 'failed'}; {int(validation_df['success'].sum())}/{len(validation_df)} checks passed",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        (
            "- Lay summary: S11 validation checks that the taxonomy covers every S10 policy, documents robustness, "
            "places classic/null/elite policies, writes exemplars and figures, and avoids starting S12."
        ),
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for _, row in validation_df.iterrows():
        rows.append(f"| `{row['checkId']}` | {'pass' if row['success'] else 'fail'} | {str(row['detail']).replace('|', '/')} |")
    return "\n".join(rows) + "\n"


def plot_outputs(
    *,
    embeddings: pd.DataFrame,
    assignments: pd.DataFrame,
    class_summary: pd.DataFrame,
    robustness: pd.DataFrame,
) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    pca = embeddings[embeddings["embeddingMethod"].eq("pca") & embeddings["embeddingSeed"].eq(0)].copy()
    merge_columns = ["policyId", "className"]
    for column in ["primaryRole", "isClassicPolicy", "isNullPolicy", "isS08Elite"]:
        if column not in pca.columns and column in assignments.columns:
            merge_columns.append(column)
    pca = pca.merge(assignments[merge_columns], on="policyId", how="left")
    classes = sorted(pca["className"].dropna().unique())
    cmap = plt.get_cmap("tab20")
    color_map = {class_name: cmap(index % 20) for index, class_name in enumerate(classes)}
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for class_name in classes:
        subset = pca[pca["className"].eq(class_name)]
        ax.scatter(
            subset["embeddingDim1"],
            subset["embeddingDim2"],
            s=18,
            alpha=0.72,
            label=class_name,
            color=color_map[class_name],
            edgecolors="none",
        )
    label_subset = pca[pca[["isClassicPolicy", "isNullPolicy", "isS08Elite"]].fillna(False).any(axis=1)].head(45)
    for _, row in label_subset.iterrows():
        label = str(row["policyId"])
        ax.annotate(label[:28], (row["embeddingDim1"], row["embeddingDim2"]), fontsize=6, alpha=0.75)
    ax.set_title("S11 empirical classes on S10 PCA behavior map")
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")
    ax.legend(loc="best", fontsize=6, ncols=2)
    fig.tight_layout()
    path = FIGURES_DIR / "s11_classes_on_s10_pca.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    role_rows = []
    for _, row in class_summary.iterrows():
        counts = json.loads(row["primaryRoleCountsJson"])
        for role, count in counts.items():
            role_rows.append({"className": row["className"], "role": role, "count": int(count)})
    role_df = pd.DataFrame(role_rows)
    role_pivot = role_df.pivot_table(index="className", columns="role", values="count", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.38 * len(role_pivot))))
    image = ax.imshow(role_pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(role_pivot.index)))
    ax.set_yticklabels(role_pivot.index, fontsize=7)
    ax.set_xticks(np.arange(len(role_pivot.columns)))
    ax.set_xticklabels(role_pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_title("Primary role composition by S11 class")
    fig.colorbar(image, ax=ax, label="policy count")
    fig.tight_layout()
    path = FIGURES_DIR / "s11_class_role_composition.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    successful = robustness[robustness["success"].astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(12, 5.8))
    labels = [f"{row.comparisonType}:{row.featureChoice}" for row in successful.itertuples()]
    x = np.arange(len(successful))
    ax.bar(x - 0.18, successful["adjustedRandIndex"], width=0.36, label="ARI", color="#2f6f7e")
    ax.bar(x + 0.18, successful["normalizedMutualInfo"], width=0.36, label="NMI", color="#9b5c8f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("S11 class robustness against seed, feature, and embedding choices")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "s11_cluster_robustness.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def copy_code_artifacts() -> list[Path]:
    paths: list[Path] = []
    for root in [STEP_DIR / "code", SHARED_CODE_DIR]:
        targets = [
            (REPO_ROOT / "scripts" / "e03_s11_universality_classes.py", root / "scripts" / "e03_s11_universality_classes.py"),
            (REPO_ROOT / "morphospace" / "universality_classes.py", root / "morphospace" / "universality_classes.py"),
            (REPO_ROOT / "morphospace" / "behavior_embeddings.py", root / "morphospace" / "behavior_embeddings.py"),
            (REPO_ROOT / "tests" / "test_e03_universality_classes.py", root / "tests" / "test_e03_universality_classes.py"),
        ]
        for source, dest in targets:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            paths.append(dest)
    return paths


def main() -> None:
    started_at = utc_now()
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_written: list[Path] = []

    inputs = load_inputs()
    config = {
        "researchStepId": STEP_ID,
        "version": UNIVERSALITY_CLASS_VERSION,
        "candidateK": list(CANDIDATE_K),
        "primarySeed": PRIMARY_SEED,
        "reductionComponents": REDUCTION_COMPONENTS,
        "exemplarsPerClass": EXEMPLARS_PER_CLASS,
        "inputs": {
            "s10Features": str(S10_FEATURES_PATH),
            "s10Normalized": str(S10_NORMALIZED_PATH),
            "s10Embeddings": str(S10_EMBEDDINGS_PATH),
            "s10Neighbors": str(S10_NEIGHBORS_PATH),
            "s07Competence": str(S07_COMPETENCE_PATH),
            "s08Elites": str(S08_ELITES_PATH),
            "s09Boundaries": str(S09_BOUNDARIES_PATH),
        },
    }
    config_path = STEP_DIR / "universality_class_config.json"
    write_json(config_path, config)
    artifacts_written.append(config_path)

    result = identify_universality_classes(
        feature_df=inputs["s10_features"],
        normalized_matrix_df=inputs["s10_normalized"],
        embeddings_df=inputs["s10_embeddings"],
        neighbors_df=inputs["s10_neighbors"],
        candidate_k=CANDIDATE_K,
        seed=PRIMARY_SEED,
        n_components=REDUCTION_COMPONENTS,
        exemplars_per_class=EXEMPLARS_PER_CLASS,
    )
    candidates_df = result["candidate_clusters"]
    assignments_df = result["class_assignments"]
    class_summary_df = result["class_summary"]
    robustness_df = result["cluster_robustness"]
    neighbor_alignment_df = result["neighbor_alignment"]
    exemplars_df = result["class_exemplars"]
    placements_df = result["classic_null_elite_placements"]

    artifacts_written.extend(write_table(candidates_df, STEP_DIR / "candidate_cluster_scores.csv", STEP_DIR / "candidate_cluster_scores.parquet"))
    artifacts_written.extend(write_table(assignments_df, STEP_DIR / "policy_class_assignments.csv", STEP_DIR / "policy_class_assignments.parquet"))
    artifacts_written.extend(write_table(assignments_df, RESULTS_DIR / "e03_s11_policy_class_assignments.csv", RESULTS_DIR / "e03_s11_policy_class_assignments.parquet"))
    artifacts_written.extend(write_table(class_summary_df, STEP_DIR / "universality_classes.csv", STEP_DIR / "universality_classes.parquet"))
    artifacts_written.extend(write_table(class_summary_df, RESULTS_DIR / "e03_universality_classes.csv", RESULTS_DIR / "e03_universality_classes.parquet"))
    artifacts_written.extend(write_table(robustness_df, STEP_DIR / "cluster_robustness.csv", STEP_DIR / "cluster_robustness.parquet"))
    artifacts_written.extend(write_table(robustness_df, RESULTS_DIR / "e03_s11_cluster_robustness.csv", RESULTS_DIR / "e03_s11_cluster_robustness.parquet"))
    artifacts_written.extend(write_table(neighbor_alignment_df, STEP_DIR / "neighbor_class_alignment.csv", STEP_DIR / "neighbor_class_alignment.parquet"))
    artifacts_written.extend(write_table(neighbor_alignment_df, RESULTS_DIR / "e03_s11_neighbor_class_alignment.csv", RESULTS_DIR / "e03_s11_neighbor_class_alignment.parquet"))
    artifacts_written.extend(write_table(exemplars_df, STEP_DIR / "class_exemplars.csv", STEP_DIR / "class_exemplars.parquet"))
    artifacts_written.extend(write_table(exemplars_df, RESULTS_DIR / "e03_s11_class_exemplars.csv", RESULTS_DIR / "e03_s11_class_exemplars.parquet"))
    artifacts_written.extend(
        write_table(
            placements_df,
            STEP_DIR / "classic_null_elite_placements.csv",
            STEP_DIR / "classic_null_elite_placements.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            placements_df,
            RESULTS_DIR / "e03_s11_classic_null_elite_placements.csv",
            RESULTS_DIR / "e03_s11_classic_null_elite_placements.parquet",
        )
    )

    figure_paths = plot_outputs(
        embeddings=inputs["s10_embeddings"],
        assignments=assignments_df,
        class_summary=class_summary_df,
        robustness=robustness_df,
    )
    artifacts_written.extend(figure_paths)

    repo_test_payload = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_e03_universality_classes",
            "tests.test_e03_behavior_embeddings",
            "tests.test_e03_phase_boundaries",
            "tests.test_e03_quality_diversity",
            "tests.test_e03_competence",
        ],
        timeout=900,
    )
    repo_log = STEP_DIR / "repo_unit_test_log.txt"
    repo_log.write_text(repo_test_payload["output"], encoding="utf-8")
    artifacts_written.append(repo_log)

    caveats = [
        "S11 universality classes are empirical computational labels over bounded simulator artifacts, not mathematical universality proofs.",
        "Small-array S07-S09 measurements and S10 feature engineering dominate the taxonomy; larger-array and full-program-token evidence remain deferred.",
        "Robustness is documented across seeds, feature choices, and embedding methods, but imperfect ARI/NMI is treated as a boundary caveat rather than hidden.",
        "Unsupported stochastic, stateful, target, signal, full frozen-cell, and larger-array GPU behavior still require later CPU expansion or kernel extensions.",
    ]
    recommended_next_action = (
        "Stop before S12 for Chief Scientist review; if approved, use the S11 class labels and exemplars to design causal feature ablations in S12."
    )

    taxonomy_report_path = STEP_DIR / "universality_taxonomy_report.md"
    write_taxonomy_report(
        path=taxonomy_report_path,
        validation_result="pending until final validation table is written",
        artifacts_written=artifacts_written,
        class_summary=class_summary_df,
        robustness=robustness_df,
        exemplars=exemplars_df,
        placements=placements_df,
        neighbor_alignment=neighbor_alignment_df,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
    )
    artifacts_written.append(taxonomy_report_path)

    upstream_statuses = {
        step: load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json") for step in ["S07", "S08", "S09", "S10"]
    }
    validation_df = validate_universality_outputs(
        feature_df=inputs["s10_features"],
        normalized_matrix_df=inputs["s10_normalized"],
        embeddings_df=inputs["s10_embeddings"],
        neighbors_df=inputs["s10_neighbors"],
        class_assignments_df=assignments_df,
        class_summary_df=class_summary_df,
        robustness_df=robustness_df,
        exemplars_df=exemplars_df,
        placements_df=placements_df,
        neighbor_alignment_df=neighbor_alignment_df,
        upstream_statuses=upstream_statuses,
        repo_test_payload={key: value for key, value in repo_test_payload.items() if key != "output"},
        taxonomy_report_exists=taxonomy_report_path.exists(),
        figure_paths=[str(path) for path in figure_paths if path.exists()],
        s12_dir_exists=(ARTIFACTS_DIR / "research_steps" / "S12").exists(),
    )
    artifacts_written.extend(write_table(validation_df, STEP_DIR / "universality_class_validation.csv", STEP_DIR / "universality_class_validation.parquet"))
    artifacts_written.extend(
        write_table(validation_df, RESULTS_DIR / "e03_s11_universality_class_validation.csv", RESULTS_DIR / "e03_s11_universality_class_validation.parquet")
    )

    validation_passed = bool(validation_df["success"].all())
    validation_result = (
        f"passed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
        if validation_passed
        else f"failed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
    )
    write_taxonomy_report(
        path=taxonomy_report_path,
        validation_result=validation_result,
        artifacts_written=artifacts_written,
        class_summary=class_summary_df,
        robustness=robustness_df,
        exemplars=exemplars_df,
        placements=placements_df,
        neighbor_alignment=neighbor_alignment_df,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
    )

    code_paths = copy_code_artifacts()
    artifacts_written.extend(code_paths)

    summary_path = STEP_DIR / "summary.md"
    validation_report_path = STEP_DIR / "validation_report.md"
    summary_path.write_text(
        render_summary(
            validation_result=validation_result,
            artifacts_written=artifacts_written,
            class_summary=class_summary_df,
            robustness=robustness_df,
            placements=placements_df,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )
    validation_report_path.write_text(
        render_validation_report(validation_df, artifacts_written, caveats, recommended_next_action),
        encoding="utf-8",
    )
    artifacts_written.extend([summary_path, validation_report_path])
    write_taxonomy_report(
        path=taxonomy_report_path,
        validation_result=validation_result,
        artifacts_written=artifacts_written,
        class_summary=class_summary_df,
        robustness=robustness_df,
        exemplars=exemplars_df,
        placements=placements_df,
        neighbor_alignment=neighbor_alignment_df,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
    )

    status_path = STEP_DIR / "status.json"
    manifest_path = STEP_DIR / "artifact_manifest.json"
    git_metadata = get_git_metadata()
    selected_k = int(candidates_df.loc[candidates_df["selectedBySilhouette"], "candidateK"].iloc[0])
    robustness_summary = (
        robustness_df[robustness_df["success"].astype(bool)]
        .groupby("comparisonType")[["adjustedRandIndex", "normalizedMutualInfo", "silhouetteScore"]]
        .mean()
        .round(6)
        .to_dict()
    )
    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": TITLE,
        "success": validation_passed,
        "status": STATUS if validation_passed else "completed_with_validation_failures",
        "outcomeClassification": OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory",
        "artifactsWritten": [str(path) for path in artifacts_written] + [str(status_path), str(manifest_path)],
        "validationResult": validation_result,
        "validationChecksPassed": int(validation_df["success"].sum()),
        "validationChecksTotal": int(len(validation_df)),
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "laySummary": (
            "S11 grouped 2,060 policies into empirical computational behavior classes using S10 features, embeddings, "
            "nearest neighbors, and S07-S09 competence/QD/phase-boundary evidence."
        ),
        "universalityClassSummary": {
            "policyCount": int(assignments_df["policyId"].nunique()),
            "classCount": int(class_summary_df["universalityClassId"].nunique()),
            "selectedClusterCount": selected_k,
            "candidateK": list(CANDIDATE_K),
            "classCounts": {str(row["className"]): int(row["policyCount"]) for _, row in class_summary_df.iterrows()},
            "meanSameClassNeighborRate": float(
                neighbor_alignment_df.loc[neighbor_alignment_df["universalityClassId"].eq("ALL"), "meanSameClassNeighborRate"].iloc[0]
            )
            if not neighbor_alignment_df.empty and neighbor_alignment_df["universalityClassId"].eq("ALL").any()
            else None,
            "robustnessSummary": robustness_summary,
            "classicNullElitePlacements": {
                "classic": int(placements_df["placementRole"].astype(str).str.contains("classic", regex=False).sum()),
                "null": int(placements_df["placementRole"].astype(str).str.contains("null", regex=False).sum()),
                "s08Elite": int(placements_df["placementRole"].astype(str).str.contains("s08_elite", regex=False).sum()),
            },
        },
        "inputArtifactCounts": {
            "s10FeaturePolicies": int(inputs["s10_features"]["policyId"].nunique()),
            "s10NormalizedPolicies": int(inputs["s10_normalized"]["policyId"].nunique()),
            "s10EmbeddingRows": int(len(inputs["s10_embeddings"])),
            "s10NeighborRows": int(len(inputs["s10_neighbors"])),
            "s07CompetenceRows": int(len(inputs["s07_competence"])),
            "s08EliteRows": int(len(inputs["s08_elites"])),
            "s09BoundaryRows": int(len(inputs["s09_boundaries"])),
        },
        "repoUnitTests": {key: value for key, value in repo_test_payload.items() if key != "output"},
        "versions": {
            "universalityClassVersion": UNIVERSALITY_CLASS_VERSION,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
        },
        "git": git_metadata,
        "startedAt": started_at,
        "completedAt": utc_now(),
    }
    write_json(status_path, status_payload)
    artifacts_written.append(status_path)

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_passed,
        "status": status_payload["status"],
        "artifactsWritten": [str(path) for path in artifacts_written] + [str(manifest_path)],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "artifactCount": len(artifacts_written) + 1,
        "checksummedArtifactCount": len(artifact_rows(artifacts_written)),
        "artifacts": artifact_rows(artifacts_written),
        "manifestSelfReference": {
            "path": str(manifest_path),
            "sha256": "omitted_self_referential_manifest",
        },
        "git": git_metadata,
        "versions": status_payload["versions"],
    }
    write_json(manifest_path, manifest_payload)
    artifacts_written.append(manifest_path)

    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": validation_passed,
                "validationResult": validation_result,
                "policyCount": int(assignments_df["policyId"].nunique()),
                "classCount": int(class_summary_df["universalityClassId"].nunique()),
                "selectedClusterCount": selected_k,
                "statusPath": str(status_path),
                "summaryPath": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
