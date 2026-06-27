#!/usr/bin/env python3
"""Execute E03 S10: embed policy behavior.

S10 consumes completed S07-S09 competence, trajectory-summary, elite, and
phase-boundary artifacts, builds a normalized behavior feature matrix, computes
PCA, spectral/diffusion-style, and MDS embeddings across repeated seeds, and
records embedding stability before stopping for S11 review.
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
    BEHAVIOR_EMBEDDING_VERSION,
    build_policy_feature_table,
    compute_embeddings,
    embedding_config_dict,
    embedding_stability_table,
    nearest_neighbors_table,
    normalize_feature_table,
    validate_embedding_outputs,
)
from morphospace.competence import canonical_json  # noqa: E402


EXPERIMENT_ID = "E03"
STEP_ID = "S10"
STEP_NUMBER = 10
TITLE = "Embed policy behavior"
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
EMBEDDING_METHODS = ("pca", "spectral", "mds")
EMBEDDING_SEEDS = (0, 1, 2)
NEIGHBOR_K = 8
STABILITY_NEIGHBOR_K = 10

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_DIR = ARTIFACTS_DIR / "research_steps" / STEP_ID
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = STEP_DIR / "figures"
SHARED_CODE_DIR = ARTIFACTS_DIR / "code" / "e03_behavior_embeddings"

CORPUS_PATH = ARTIFACTS_DIR / "data" / "e03_policy_corpus.parquet"
S07_COMPETENCE_PATH = RESULTS_DIR / "e03_morphospace_metrics.parquet"
S07_RUNS_PATH = RESULTS_DIR / "e03_coarse_sweep.parquet"
S08_ELITES_PATH = RESULTS_DIR / "e03_qd_elites.parquet"
S08_VALIDATION_PATH = RESULTS_DIR / "e03_s08_cpu_elite_validation.parquet"
S09_COMPETENCE_PATH = RESULTS_DIR / "e03_s09_phase_boundary_competence_vectors.parquet"
S09_RUNS_PATH = RESULTS_DIR / "e03_s09_phase_boundary_runs.parquet"
S09_BOUNDARIES_PATH = RESULTS_DIR / "e03_phase_boundaries.parquet"
S09_VARIANCE_PATH = RESULTS_DIR / "e03_s09_phase_boundary_variance.parquet"


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
        "corpus": CORPUS_PATH,
        "s07_competence": S07_COMPETENCE_PATH,
        "s07_runs": S07_RUNS_PATH,
        "s08_elites": S08_ELITES_PATH,
        "s08_validation": S08_VALIDATION_PATH,
        "s09_competence": S09_COMPETENCE_PATH,
        "s09_runs": S09_RUNS_PATH,
        "s09_boundaries": S09_BOUNDARIES_PATH,
        "s09_variance": S09_VARIANCE_PATH,
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"S10 requires completed S07-S09 artifacts; missing: {missing}")
    for step in ["S07", "S08", "S09"]:
        status = load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json")
        if not status.get("success"):
            raise RuntimeError(f"S10 requires successful {step}; got {status}")
    return {key: pd.read_parquet(path) for key, path in required_paths.items()}


def metadata_for_embeddings(feature_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "policyId",
        "primaryRole",
        "family",
        "generationMethod",
        "lineageId",
        "classicFamily",
        "isNullPolicy",
        "isClassicPolicy",
        "isS08Elite",
        "isPhaseBoundaryPolicy",
        "isPathologicalPolicy",
        "evidenceLayerCount",
        "phaseBoundaryInvolvementCount",
        "s08Elite_eliteRank",
        "s08Elite_qualityScore",
        "s09Trace_completedNumericMean",
        "s09Trace_finalSortednessScoreMean",
        "s09Trace_oscillationProxyMean",
    ]
    available = [column for column in columns if column in feature_df.columns]
    return feature_df[available].copy()


def write_feature_normalization_report(normalization_df: pd.DataFrame, feature_df: pd.DataFrame, path: Path) -> None:
    missing_features = int(normalization_df["isMissingIndicator"].sum())
    rows = [
        "# S10 Feature Normalization Report",
        "",
        "- Research step ID: S10",
        "- Completion status: completed",
        f"- Artifacts written: `{STEP_DIR}/feature_normalization_spec.parquet` and `{STEP_DIR}/normalized_feature_matrix.parquet`",
        "- Validation result: documented by the final S10 validation table after the full embedding run completes.",
        "- Caveats or blockers: normalization is a screening representation over small-array computational proxies, not a biological metric space.",
        "- Recommended next action: use these features only after S10 validation passes and Chief Scientist review approves S11.",
        "",
        "## Normalization Contract",
        "",
        "- Numeric features are median-imputed, centered by the median, scaled by IQR, and clipped to [-5, 5].",
        "- Features with zero IQR use population standard deviation; constant features use unit scale.",
        "- Every feature with missing values gets an additional binary `__missing` indicator.",
        "- Categorical fields such as family and lineage are retained as metadata rather than one-hot embedded, except role booleans and structural counts.",
        "",
        "## Counts",
        "",
        f"- Policies: {len(feature_df)}",
        f"- Normalized feature columns: {len(normalization_df)}",
        f"- Missing-indicator columns: {missing_features}",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_embeddings(embeddings_df: pd.DataFrame, metadata_df: pd.DataFrame, stability_df: pd.DataFrame) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    merged = embeddings_df.merge(metadata_df, on="policyId", how="left")
    role_order = ["classic", "null", "elite", "phase_boundary", "pathological", "generated"]
    colors = {
        "classic": "#1f77b4",
        "null": "#7f7f7f",
        "elite": "#2ca02c",
        "phase_boundary": "#ff7f0e",
        "pathological": "#d62728",
        "generated": "#9467bd",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), squeeze=False)
    for ax, method in zip(axes[0], EMBEDDING_METHODS):
        subset = merged[merged["embeddingMethod"].eq(method) & merged["embeddingSeed"].eq(0)]
        for role in role_order:
            role_subset = subset[subset["primaryRole"].eq(role)]
            if role_subset.empty:
                continue
            ax.scatter(
                role_subset["embeddingDim1"],
                role_subset["embeddingDim2"],
                s=18 if role not in {"classic", "elite"} else 32,
                alpha=0.75,
                label=role,
                color=colors[role],
                edgecolors="none",
            )
        ax.set_title(f"{method} seed 0")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.tight_layout()
    path = FIGURES_DIR / "behavior_embedding_methods_seed0.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    pca = merged[merged["embeddingMethod"].eq("pca") & merged["embeddingSeed"].eq(0)].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for role in role_order:
        role_subset = pca[pca["primaryRole"].eq(role)]
        if role_subset.empty:
            continue
        ax.scatter(
            role_subset["embeddingDim1"],
            role_subset["embeddingDim2"],
            s=20 if role not in {"classic", "elite"} else 42,
            alpha=0.78,
            label=role,
            color=colors[role],
            edgecolors="none",
        )
    label_subset = pca[pca["isClassicPolicy"].fillna(False) | pca["isNullPolicy"].fillna(False) | pca["isS08Elite"].fillna(False)].head(40)
    for _, row in label_subset.iterrows():
        label = str(row["lineageId"])
        if len(label) > 28:
            label = label[:25] + "..."
        ax.annotate(label, (row["embeddingDim1"], row["embeddingDim2"]), fontsize=6, alpha=0.75)
    ax.set_title("PCA behavior embedding with classic, null, and elite labels")
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    path = FIGURES_DIR / "pca_classic_null_elite_placements.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    if not stability_df.empty:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        labels = [
            f"{row.comparisonType}:{row.leftMethod}-{row.rightMethod}:{row.leftSeed}-{row.rightSeed}"
            for row in stability_df.itertuples()
        ]
        x = np.arange(len(stability_df))
        ax.bar(x - 0.2, stability_df["meanNeighborJaccard"], width=0.4, label="neighbor Jaccard", color="#2f6f7e")
        ax.bar(x + 0.2, stability_df["pairwiseDistanceSpearman"], width=0.4, label="distance Spearman", color="#8d5a97")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("S10 embedding stability across methods and seeds")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = FIGURES_DIR / "embedding_stability_summary.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)
    return paths


def placement_table(embeddings_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    merged = embeddings_df.merge(metadata_df, on="policyId", how="left")
    subset = merged[
        merged["embeddingSeed"].eq(0)
        & (
            merged["isClassicPolicy"].fillna(False)
            | merged["isNullPolicy"].fillna(False)
            | merged["isS08Elite"].fillna(False)
            | merged["isPathologicalPolicy"].fillna(False)
            | merged["isPhaseBoundaryPolicy"].fillna(False)
        )
    ].copy()
    return subset.sort_values(["embeddingMethod", "primaryRole", "policyId"], kind="mergesort").reset_index(drop=True)


def render_summary(
    *,
    validation_result: str,
    artifacts_written: list[Path],
    feature_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    role_counts = feature_df["primaryRole"].value_counts().to_dict()
    method_counts = embeddings_df.groupby("embeddingMethod")["policyId"].nunique().to_dict()
    stability_brief = (
        stability_df.groupby("comparisonType")[["meanNeighborJaccard", "pairwiseDistanceSpearman"]].mean().round(4).to_dict()
        if not stability_df.empty
        else {}
    )
    return "\n".join(
        [
            "# S10 Behavior Embedding Summary",
            "",
            "- Research step ID: S10",
            f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
            f"- Artifacts written: {len(artifacts_written)} files, including `{RESULTS_DIR / 'e03_policy_embeddings.parquet'}`, `{STEP_DIR / 'feature_normalization_report.md'}`, and `{FIGURES_DIR}/`",
            f"- Validation result: {validation_result}",
            f"- Caveats or blockers: {'; '.join(caveats)}",
            f"- Recommended next action: {recommended_next_action}",
            (
                "- Lay summary: S10 turned S07-S09 policy measurements into a normalized behavior feature matrix, "
                "embedded policies with PCA, spectral/diffusion-style, and MDS methods, and measured how stable those maps are across random seeds and methods."
            ),
            "",
            "## Key Counts",
            "",
            f"- Policies embedded: {feature_df['policyId'].nunique()}",
            f"- Embedding rows: {len(embeddings_df)}",
            f"- Embedding methods: `{canonical_json(method_counts)}`",
            f"- Primary role counts: `{canonical_json(role_counts)}`",
            f"- Stability summary: `{canonical_json(stability_brief)}`",
        ]
    ) + "\n"


def render_validation_report(validation_df: pd.DataFrame, artifacts_written: list[Path], caveats: list[str], recommended_next_action: str) -> str:
    rows = [
        "# S10 Validation Report",
        "",
        "- Research step ID: S10",
        f"- Completion status: {STATUS}",
        f"- Artifacts written: {len(artifacts_written)} files",
        f"- Validation result: {'passed' if validation_df['success'].all() else 'failed'}; {int(validation_df['success'].sum())}/{len(validation_df)} checks passed",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for _, row in validation_df.iterrows():
        rows.append(f"| `{row['checkId']}` | {'pass' if row['success'] else 'fail'} | {str(row['detail']).replace('|', '/')} |")
    return "\n".join(rows) + "\n"


def copy_code_artifacts() -> list[Path]:
    paths: list[Path] = []
    for root in [STEP_DIR / "code", SHARED_CODE_DIR]:
        targets = [
            (REPO_ROOT / "scripts" / "e03_s10_behavior_embeddings.py", root / "scripts" / "e03_s10_behavior_embeddings.py"),
            (REPO_ROOT / "morphospace" / "behavior_embeddings.py", root / "morphospace" / "behavior_embeddings.py"),
            (REPO_ROOT / "morphospace" / "phase_boundaries.py", root / "morphospace" / "phase_boundaries.py"),
            (REPO_ROOT / "morphospace" / "competence.py", root / "morphospace" / "competence.py"),
            (REPO_ROOT / "tests" / "test_e03_behavior_embeddings.py", root / "tests" / "test_e03_behavior_embeddings.py"),
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
    config = embedding_config_dict(methods=EMBEDDING_METHODS, seeds=EMBEDDING_SEEDS, neighbor_k=STABILITY_NEIGHBOR_K)
    config_path = STEP_DIR / "behavior_embedding_config.json"
    write_json(config_path, config)
    artifacts_written.append(config_path)

    feature_df = build_policy_feature_table(
        corpus_df=inputs["corpus"],
        s07_competence_df=inputs["s07_competence"],
        s07_run_df=inputs["s07_runs"],
        s08_elites_df=inputs["s08_elites"],
        s08_validation_df=inputs["s08_validation"],
        s09_competence_df=inputs["s09_competence"],
        s09_run_df=inputs["s09_runs"],
        s09_boundaries_df=inputs["s09_boundaries"],
        s09_variance_df=inputs["s09_variance"],
    )
    normalized_matrix_df, normalization_df = normalize_feature_table(feature_df)
    embeddings_raw_df = compute_embeddings(normalized_matrix_df, methods=EMBEDDING_METHODS, seeds=EMBEDDING_SEEDS)
    metadata_df = metadata_for_embeddings(feature_df)
    embeddings_df = embeddings_raw_df.merge(metadata_df, on="policyId", how="left")
    stability_df = embedding_stability_table(embeddings_raw_df, neighbor_k=STABILITY_NEIGHBOR_K)
    neighbors_df = nearest_neighbors_table(normalized_matrix_df, neighbor_k=NEIGHBOR_K)
    neighbors_df = (
        neighbors_df.merge(metadata_df[["policyId", "primaryRole"]].rename(columns={"primaryRole": "policyPrimaryRole"}), on="policyId", how="left")
        .merge(
            metadata_df[["policyId", "primaryRole"]].rename(
                columns={"policyId": "neighborPolicyId", "primaryRole": "neighborPrimaryRole"}
            ),
            on="neighborPolicyId",
            how="left",
        )
    )
    placements_df = placement_table(embeddings_raw_df, metadata_df)
    feature_norm_report_path = STEP_DIR / "feature_normalization_report.md"
    write_feature_normalization_report(normalization_df, feature_df, feature_norm_report_path)

    artifacts_written.extend(write_table(feature_df, STEP_DIR / "embedding_feature_table.csv", STEP_DIR / "embedding_feature_table.parquet"))
    artifacts_written.extend(write_table(feature_df, RESULTS_DIR / "e03_s10_embedding_features.csv", RESULTS_DIR / "e03_s10_embedding_features.parquet"))
    artifacts_written.extend(write_table(normalized_matrix_df, STEP_DIR / "normalized_feature_matrix.csv", STEP_DIR / "normalized_feature_matrix.parquet"))
    artifacts_written.extend(write_table(normalization_df, STEP_DIR / "feature_normalization_spec.csv", STEP_DIR / "feature_normalization_spec.parquet"))
    artifacts_written.extend(write_table(embeddings_df, STEP_DIR / "policy_embeddings.csv", STEP_DIR / "policy_embeddings.parquet"))
    artifacts_written.extend(write_table(embeddings_df, RESULTS_DIR / "e03_policy_embeddings.csv", RESULTS_DIR / "e03_policy_embeddings.parquet"))
    artifacts_written.extend(write_table(stability_df, STEP_DIR / "embedding_stability.csv", STEP_DIR / "embedding_stability.parquet"))
    artifacts_written.extend(write_table(stability_df, RESULTS_DIR / "e03_s10_embedding_stability.csv", RESULTS_DIR / "e03_s10_embedding_stability.parquet"))
    artifacts_written.extend(write_table(neighbors_df, STEP_DIR / "embedding_neighbors.csv", STEP_DIR / "embedding_neighbors.parquet"))
    artifacts_written.extend(write_table(neighbors_df, RESULTS_DIR / "e03_s10_embedding_neighbors.csv", RESULTS_DIR / "e03_s10_embedding_neighbors.parquet"))
    artifacts_written.extend(write_table(placements_df, STEP_DIR / "embedding_policy_placements.csv", STEP_DIR / "embedding_policy_placements.parquet"))
    artifacts_written.extend(write_table(placements_df, RESULTS_DIR / "e03_s10_embedding_policy_placements.csv", RESULTS_DIR / "e03_s10_embedding_policy_placements.parquet"))
    artifacts_written.append(feature_norm_report_path)

    figure_paths = plot_embeddings(embeddings_raw_df, metadata_df, stability_df)
    artifacts_written.extend(figure_paths)

    repo_test_payload = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
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

    upstream_statuses = {
        step: load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json") for step in ["S07", "S08", "S09"]
    }
    validation_df = validate_embedding_outputs(
        feature_df=feature_df,
        normalized_matrix_df=normalized_matrix_df,
        normalization_df=normalization_df,
        embeddings_df=embeddings_df,
        stability_df=stability_df,
        neighbors_df=neighbors_df,
        upstream_statuses=upstream_statuses,
        repo_test_payload={key: value for key, value in repo_test_payload.items() if key != "output"},
        figure_paths=[str(path) for path in figure_paths if path.exists()],
        s11_dir_exists=(ARTIFACTS_DIR / "research_steps" / "S11").exists(),
    )
    artifacts_written.extend(write_table(validation_df, STEP_DIR / "behavior_embedding_validation.csv", STEP_DIR / "behavior_embedding_validation.parquet"))
    artifacts_written.extend(write_table(validation_df, RESULTS_DIR / "e03_s10_behavior_embedding_validation.csv", RESULTS_DIR / "e03_s10_behavior_embedding_validation.parquet"))

    code_paths = copy_code_artifacts()
    artifacts_written.extend(code_paths)

    caveats = [
        "S10 embeddings are exploratory maps over computational proxy features from S07-S09, not proof of causal policy classes.",
        "Small-array S07-S09 measurements dominate the feature matrix; larger-array transfer remains deferred.",
        "MDS and spectral layouts are seed-sensitive, so S10 reports stability rather than treating one layout as canonical.",
        "Categorical DSL structure is represented mainly through structural counts and role metadata, not full program-token embeddings.",
    ]
    recommended_next_action = (
        "Stop before S11 for Chief Scientist review; if approved, use S10 normalized features, embeddings, neighbors, and stability records "
        "to identify empirical universality classes in S11."
    )
    validation_passed = bool(validation_df["success"].all())
    validation_result = (
        f"passed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
        if validation_passed
        else f"failed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
    )

    summary_path = STEP_DIR / "summary.md"
    validation_report_path = STEP_DIR / "validation_report.md"
    summary_path.write_text(
        render_summary(
            validation_result=validation_result,
            artifacts_written=artifacts_written,
            feature_df=feature_df,
            embeddings_df=embeddings_df,
            stability_df=stability_df,
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

    status_path = STEP_DIR / "status.json"
    manifest_path = STEP_DIR / "artifact_manifest.json"
    git_metadata = get_git_metadata()
    role_counts = feature_df["primaryRole"].value_counts().to_dict()
    stability_summary = (
        stability_df.groupby("comparisonType")[["meanNeighborJaccard", "pairwiseDistanceSpearman", "procrustesDisparity"]]
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
            "S10 converted S07-S09 policy behavior into normalized features, generated PCA, spectral, and MDS maps, "
            "and measured seed/method stability before S11."
        ),
        "embeddingSummary": {
            "policyCount": int(feature_df["policyId"].nunique()),
            "normalizedFeatureCount": int(normalized_matrix_df.shape[1] - 1),
            "embeddingRowCount": int(len(embeddings_df)),
            "embeddingMethods": list(EMBEDDING_METHODS),
            "embeddingSeeds": list(EMBEDDING_SEEDS),
            "nearestNeighborRows": int(len(neighbors_df)),
            "stabilityRows": int(len(stability_df)),
            "roleCounts": {str(key): int(value) for key, value in role_counts.items()},
            "stabilitySummary": stability_summary,
        },
        "repoUnitTests": {key: value for key, value in repo_test_payload.items() if key != "output"},
        "versions": {
            "behaviorEmbeddingVersion": BEHAVIOR_EMBEDDING_VERSION,
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
                "policyCount": int(feature_df["policyId"].nunique()),
                "normalizedFeatureCount": int(normalized_matrix_df.shape[1] - 1),
                "embeddingRows": int(len(embeddings_df)),
                "statusPath": str(status_path),
                "summaryPath": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
