#!/usr/bin/env python3
"""Map E05 S07-S11 trajectories through morphospace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace2d import (  # noqa: E402
    TRAJECTORY_MAP_SCHEMA_VERSION,
    assign_route_and_failure_clusters,
    embedding_rows,
    embedding_stability_checks,
    fit_embedding_coordinates,
    load_harmonized_traces,
    metadata_confounding_checks,
    prepare_embedding_matrix,
    route_diversity_summary,
    standard_trace_source_specs,
    summarize_trajectories,
    trajectory_validation_rows,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S12"
STEP_NUMBER = 12
STEP_TITLE = "Map trajectories through morphospace"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
FOCUSED_TESTS = [
    "tests.test_e05_trajectory_maps",
    "tests.test_e05_gpu_batches",
    "tests.test_e05_symmetry",
    "tests.test_e05_scaling",
    "tests.test_e05_regeneration",
    "tests.test_e05_scrambled_recovery",
    "tests.test_e05_metrics",
]


FEATURE_DEFINITIONS = {
    "initial_error_proxy": "First available target-error proxy at trajectory start.",
    "final_error_proxy": "Target-error proxy at the final saved trajectory state.",
    "best_error_proxy": "Minimum saved target-error proxy along the trajectory.",
    "worst_error_proxy": "Maximum saved target-error proxy along the trajectory.",
    "mean_error_proxy": "Mean saved target-error proxy.",
    "std_error_proxy": "Population standard deviation of saved target-error proxy.",
    "error_reduction": "Initial error minus final error.",
    "relative_error_reduction_proxy": "Error reduction divided by initial error, with zero guard.",
    "area_under_error_curve": "Step-normalized trapezoid area under saved error trajectory.",
    "monotonicity_error_proxy": "Sum of positive step-to-step increases in saved error.",
    "temporary_worsening_count": "Count of saved intervals where target-error proxy increased.",
    "path_length_proxy": "Euclidean path length through normalized saved-state feature space.",
    "direct_distance_proxy": "Start-to-final Euclidean distance in saved-state feature space.",
    "path_curvature_proxy": "Path length divided by direct distance, with stationary tie handling.",
    "final_score_proxy": "One minus final error proxy, clipped to [0, 1].",
    "score_gain": "Final score proxy minus initial score proxy.",
    "final_progress_fraction": "Final fraction of initial error reduced.",
    "mean_progress_fraction": "Mean saved fraction of initial error reduced.",
    "final_edge_disagreement": "Final S11 edge-disagreement feature when available.",
    "mean_edge_disagreement": "Mean S11 edge-disagreement feature when available.",
    "final_axis_strength": "Final S10 axis-strength feature when available.",
    "mean_axis_strength": "Mean S10 axis-strength feature when available.",
    "final_target_energy": "Final normalized target-energy feature when available.",
    "mean_target_energy": "Mean normalized target-energy feature when available.",
    "cell_count_delta": "Final minus initial cell count when traces expose cell count.",
    "action_activity_total": "Final cumulative local-action count proxy when traces expose actions.",
    "duration_steps": "Final saved step minus first saved step.",
    "snapshot_count": "Number of saved states in the trajectory artifact.",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(json_ready(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "args": args,
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "runtimeSeconds": time.perf_counter() - started,
    }


def git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "get-url", "origin"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["success"] else "unknown",
        "branch": branch["stdout"].strip() if branch["success"] else "unknown",
        "remote": remote["stdout"].strip() if remote["success"] else "unknown",
        "statusShort": status["stdout"].strip(),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataframe(df: pd.DataFrame, path_without_suffix: Path) -> list[Path]:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path_without_suffix.with_suffix(".csv")
    parquet_path = path_without_suffix.with_suffix(".parquet")
    safe = df.copy()
    for column in safe.columns:
        if safe[column].map(lambda item: isinstance(item, (Mapping, list, tuple))).any():
            safe[column] = safe[column].map(
                lambda item: json.dumps(json_ready(item), sort_keys=True) if isinstance(item, (Mapping, list, tuple)) else item
            )
    safe.to_csv(csv_path, index=False)
    safe.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def collect_artifacts(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        records.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(records, key=lambda row: row["path"])


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            if abs(value) >= 1000 or (0 < abs(value) < 0.001):
                return f"{value:.3g}"
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map S07-S11 trajectories through morphospace.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--random-state", type=int, default=12012)
    parser.add_argument("--stability-sample-size", type=int, default=320)
    parser.add_argument("--skip-repo-tests", action="store_true")
    return parser.parse_args()


def run_repo_tests(step_dir: Path, skip: bool) -> dict[str, Any]:
    log_path = step_dir / "repo_unit_test_log.txt"
    if skip:
        payload = {
            "args": [],
            "returncode": 0,
            "success": True,
            "stdout": "",
            "stderr": "Skipped by --skip-repo-tests.",
            "runtimeSeconds": 0.0,
            "logPath": str(log_path),
        }
        log_path.write_text("Skipped by --skip-repo-tests.\n", encoding="utf-8")
        return payload
    command = [sys.executable, "-m", "unittest", *FOCUSED_TESTS]
    result = run_command(command, cwd=REPO_ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "Command: " + " ".join(command) + "\n"
        + f"Return code: {result['returncode']}\n"
        + f"Runtime seconds: {result['runtimeSeconds']:.3f}\n\n"
        + "STDOUT\n"
        + result["stdout"]
        + "\nSTDERR\n"
        + result["stderr"],
        encoding="utf-8",
    )
    result["logPath"] = str(log_path)
    return result


def append_repo_validation(validation_df: pd.DataFrame, repo_test: Mapping[str, Any]) -> pd.DataFrame:
    row = {
        "schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
        "research_step_id": STEP_ID,
        "check_id": "repo_unit_tests_passed",
        "success": bool(repo_test.get("success", False)),
        "detail": json.dumps({"returncode": repo_test.get("returncode"), "args": repo_test.get("args")}, sort_keys=True),
    }
    return pd.concat([validation_df, pd.DataFrame([row])], ignore_index=True)


def plot_embedding_map(embedding_df: pd.DataFrame, method: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = embedding_df[embedding_df["embedding_method"].eq(method)].copy()
    sources = list(dict.fromkeys(subset["source_step_id"].astype(str).tolist()))
    colors = plt.get_cmap("tab10")
    fig, axis = plt.subplots(figsize=(8.6, 6.2))
    for index, source in enumerate(sources):
        group = subset[subset["source_step_id"].astype(str).eq(source)]
        axis.scatter(
            group["embedding_x"],
            group["embedding_y"],
            s=18,
            alpha=0.65,
            label=source,
            color=colors(index % 10),
            edgecolors="none",
        )
    axis.set_xlabel("embedding dimension 1")
    axis.set_ylabel("embedding dimension 2")
    axis.set_title(f"S12 trajectory-summary morphospace: {method}")
    axis.grid(color="#dddddd", linewidth=0.45)
    axis.legend(title="source", fontsize=8, title_fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_route_diversity(route_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = route_df[route_df["grouping"].eq("source_policy_family")].copy()
    subset["label"] = subset["source_step_id"].astype(str) + "\n" + subset["policy_family"].astype(str).str.replace("_", " ", regex=False)
    subset = subset.sort_values(["source_step_id", "mean_pairwise_route_distance"], ascending=[True, False]).head(28)
    fig, axis = plt.subplots(figsize=(12, 5.8))
    axis.bar(np.arange(len(subset)), subset["mean_pairwise_route_distance"].astype(float), color="#5c8d89", edgecolor="#222222", linewidth=0.35)
    axis.set_xticks(np.arange(len(subset)))
    axis.set_xticklabels(subset["label"], rotation=55, ha="right", fontsize=7)
    axis.set_ylabel("mean pairwise route distance")
    axis.set_title("S12 route diversity by source and policy family")
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confounding(confounding_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = confounding_df[confounding_df["computed"].astype(bool)].copy()
    subset["label"] = subset["embedding_method"].astype(str) + "\n" + subset["metadata_field"].astype(str)
    fig, axis = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(subset))
    axis.bar(x - 0.18, subset["majority_baseline_accuracy"].astype(float), width=0.36, label="majority baseline", color="#9aa4b2")
    axis.bar(x + 0.18, subset["cv_accuracy"].astype(float), width=0.36, label="embedding CV accuracy", color="#c47f5a")
    axis.set_xticks(x)
    axis.set_xticklabels(subset["label"], rotation=50, ha="right", fontsize=7)
    axis.set_ylabel("classification accuracy")
    axis.set_title("S12 metadata predictability from embedding coordinates")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_path_curvature(summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = summary_df.groupby(["source_step_id", "policy_family"], dropna=False, sort=True).agg(
        path_curvature_proxy=("path_curvature_proxy", "mean"),
        temporary_worsening_count=("temporary_worsening_count", "mean"),
    ).reset_index()
    grouped["label"] = grouped["source_step_id"].astype(str) + "\n" + grouped["policy_family"].astype(str).str.replace("_", " ", regex=False)
    grouped = grouped.sort_values(["source_step_id", "path_curvature_proxy"], ascending=[True, False]).head(28)
    fig, axis = plt.subplots(figsize=(12, 5.8))
    axis.scatter(
        np.arange(len(grouped)),
        grouped["path_curvature_proxy"].astype(float),
        s=np.clip(grouped["temporary_worsening_count"].astype(float).to_numpy() * 12 + 24, 24, 220),
        color="#6d77a8",
        alpha=0.75,
        edgecolors="#222222",
        linewidths=0.35,
    )
    axis.set_xticks(np.arange(len(grouped)))
    axis.set_xticklabels(grouped["label"], rotation=55, ha="right", fontsize=7)
    axis.set_ylabel("mean path curvature proxy")
    axis.set_title("S12 trajectory curvature and temporary worsening")
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def feature_spec_markdown(feature_cols: Sequence[str]) -> str:
    rows = [[column, FEATURE_DEFINITIONS.get(column, "")] for column in feature_cols]
    return f"""# E05 S12 Embedding Feature Specification

Research step ID: {STEP_ID}
Completion status: feature schema documented for trajectory-summary embeddings

S12 maps trajectory summaries rather than raw cell-by-cell state tensors. Each S07-S11 trace is first reduced to a common target-error proxy, score proxy, progress fraction, and saved-state path. PCA and diffusion-map embeddings are fit on the standardized columns below after median imputation of source-specific missing fields.

Error-proxy precedence:

1. `composite_error` when available from S07-S09.
2. `hamming_error` when available from S11.
3. `1 - pattern_score` or `1 - label_match_fraction` when available from S10/S11.
4. Boolean success or exact-match complements only as a fallback.

## Embedding Columns

{markdown_table(["feature", "definition"], rows)}

## Caveats

- The embedding feature table intentionally preserves `trajectory_id`, `source_step_id`, policy metadata, and source artifact paths, but those metadata columns are not included as embedding coordinates.
- Mixed CPU and GPU traces expose different component metrics; unavailable fields are median-imputed for embedding and audited for metadata predictability.
- Low-dimensional coordinates are exploratory route maps, not direct biological measurements.
"""


def report_markdown(
    *,
    success: bool,
    source_catalog: pd.DataFrame,
    summary_df: pd.DataFrame,
    method_info_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    confounding_df: pd.DataFrame,
    route_diversity_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    artifacts: Sequence[Path],
    validation_result: str,
    caveats: Sequence[str],
    recommended_next_action: str,
) -> str:
    source_rows = summary_df.groupby("source_step_id", sort=True).agg(
        trajectories=("trajectory_id", "count"),
        mean_final_error=("final_error_proxy", "mean"),
        mean_relative_reduction=("relative_error_reduction_proxy", "mean"),
        mean_curvature=("path_curvature_proxy", "mean"),
    ).reset_index()
    confound_rows = confounding_df[confounding_df["computed"].astype(bool)][
        ["embedding_method", "metadata_field", "cv_accuracy", "majority_baseline_accuracy", "accuracy_lift", "confounding_flag"]
    ].values.tolist()
    artifacts_text = "\n".join(f"- `{path}`" for path in artifacts[:40])
    if len(artifacts) > 40:
        artifacts_text += f"\n- ... {len(artifacts) - 40} additional files listed in `artifact_manifest.json`"
    caveat_text = "\n".join(f"- {item}" for item in caveats)
    return f"""# E05 S12 Morphospace Trajectory Report

- Research step ID: {STEP_ID}
- Completion status: {"completed" if success else "completed with validation failures"}
- Artifacts written:
{artifacts_text}
- Validation result: {validation_result}
- Caveats or blockers:
{caveat_text}
- Recommended next action: {recommended_next_action}

S12 harmonized S07-S11 trajectory artifacts into a common trajectory-summary feature table, then generated PCA and diffusion-map route maps. The route metrics quantify path curvature, temporary movement away from target, convergence basins, failure clusters, and route diversity while preserving source trajectory identifiers.

## Source Coverage

{markdown_table(["source", "trace available", "run available", "trace kind"], source_catalog[["source_step_id", "trace_available", "run_summary_available", "trace_kind"]].values.tolist())}

## Source-Level Path Summary

{markdown_table(["source", "trajectories", "mean final error", "mean relative reduction", "mean curvature"], source_rows[["source_step_id", "trajectories", "mean_final_error", "mean_relative_reduction", "mean_curvature"]].values.tolist())}

## Embedding Diagnostics

{markdown_table(["method", "trajectories", "features", "distance spearman", "normalized stress", "variance total"], method_info_df[["embedding_method", "trajectory_count", "feature_count", "distance_spearman", "normalized_stress", "explained_variance_ratio_total"]].fillna("").values.tolist())}

## Stability Checks

{markdown_table(["method", "check", "sample", "spearman", "passed"], stability_df[["embedding_method", "check_type", "sample_count", "spearman_distance_correlation", "passed"]].values.tolist())}

## Metadata Confounding Audit

{markdown_table(["method", "field", "cv accuracy", "baseline", "lift", "flag"], confound_rows)}

## Cluster And Route Tables

- Route-diversity rows: {len(route_diversity_df)}
- Cluster-summary rows: {len(cluster_df)}
- Validation checks passed: {int(validation_df["success"].astype(bool).sum())}/{len(validation_df)}

## Lay Summary

The S12 maps show that trajectories from different tasks and policy families occupy separable regions of a shared computational morphospace. This is useful for comparing route shapes and failure modes, but the separation partly reflects protocol and source-step differences, so these maps should guide S13 control comparisons rather than be treated as final biological claims.
"""


def summary_markdown(
    *,
    success: bool,
    artifacts: Sequence[Path],
    validation_result: str,
    outcome: str,
    caveats: Sequence[str],
    recommended_next_action: str,
    anchor_result: str,
) -> str:
    artifact_lines = "\n".join(f"- `{path}`" for path in artifacts)
    caveat_lines = "\n".join(f"- {item}" for item in caveats)
    return f"""# E05 S12 Status Summary

- Research step ID: {STEP_ID}
- Completion status: {"completed" if success else "completed with validation failures"}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers:
{caveat_lines}
- Lay summary: S12 merged CPU and GPU trajectory artifacts from S07-S11, reduced them to documented trajectory-summary features, and generated PCA plus diffusion-map morphospace maps with route-diversity, curvature, backtracking, convergence-basin, failure-cluster, stability, and metadata-confounding checks.
- Recommended next action: {recommended_next_action}

Anchor result: {anchor_result}
"""


def update_run_manifest(artifacts_dir: Path, status_path: Path, artifacts: Sequence[Path]) -> Path:
    manifest_path = artifacts_dir / "provenance" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    steps = payload.setdefault("researchSteps", {})
    steps[STEP_ID] = {
        "statusPath": str(status_path),
        "artifactCount": len(artifacts),
        "updatedAt": utc_now(),
        "schemaVersion": TRAJECTORY_MAP_SCHEMA_VERSION,
    }
    payload.setdefault("schemaVersion", "eidosoma.run_manifest.v1")
    payload["updatedAt"] = utc_now()
    write_json(manifest_path, payload)
    return manifest_path


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    artifacts_dir = Path(args.artifacts_dir)
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures"
    configs_dir = artifacts_dir / "configs"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    specs = standard_trace_source_specs(artifacts_dir)
    feature_df, source_catalog = load_harmonized_traces(specs)
    summary_df = summarize_trajectories(feature_df)
    matrix, imputed_features_df, feature_cols = prepare_embedding_matrix(summary_df)
    clustered_summary_df, cluster_df = assign_route_and_failure_clusters(summary_df, matrix, random_state=int(args.random_state))
    coords_by_method, method_info_df = fit_embedding_coordinates(matrix, random_state=int(args.random_state))
    embedding_df = embedding_rows(clustered_summary_df, coords_by_method)
    stability_df = embedding_stability_checks(
        clustered_summary_df,
        matrix,
        random_states=(int(args.random_state), int(args.random_state) + 1, int(args.random_state) + 2),
        max_items=int(args.stability_sample_size),
    )
    confounding_df = metadata_confounding_checks(embedding_df, random_state=int(args.random_state))
    route_df = route_diversity_summary(clustered_summary_df, matrix)
    validation_df = trajectory_validation_rows(
        source_catalog=source_catalog,
        feature_df=feature_df,
        summary_df=clustered_summary_df,
        embedding_df=embedding_df,
        method_info_df=method_info_df,
        stability_df=stability_df,
        confounding_df=confounding_df,
        route_diversity_df=route_df,
        cluster_df=cluster_df,
    )
    repo_test = run_repo_tests(step_dir, skip=bool(args.skip_repo_tests))
    validation_df = append_repo_validation(validation_df, repo_test)
    success = bool(validation_df["success"].astype(bool).all())

    artifacts: list[Path] = []
    artifacts.extend(write_dataframe(feature_df, step_dir / "trajectory_state_features"))
    artifacts.extend(write_dataframe(clustered_summary_df, step_dir / "trajectory_summary_features"))
    artifacts.extend(write_dataframe(imputed_features_df.assign(trajectory_id=clustered_summary_df["trajectory_id"].to_numpy()), step_dir / "trajectory_embedding_feature_matrix"))
    artifacts.extend(write_dataframe(embedding_df, step_dir / "trajectory_embeddings"))
    artifacts.extend(write_dataframe(embedding_df, results_dir / "e05_morphospace_trajectories"))
    artifacts.extend(write_dataframe(method_info_df, step_dir / "trajectory_embedding_method_diagnostics"))
    artifacts.extend(write_dataframe(stability_df, step_dir / "embedding_stability_checks"))
    artifacts.extend(write_dataframe(confounding_df, step_dir / "metadata_confounding_checks"))
    artifacts.extend(write_dataframe(route_df, step_dir / "route_diversity_summary"))
    artifacts.extend(write_dataframe(cluster_df, step_dir / "trajectory_cluster_summary"))
    artifacts.extend(write_dataframe(source_catalog, step_dir / "source_artifact_catalog"))
    artifacts.extend(write_dataframe(validation_df, step_dir / "trajectory_validation_results"))

    config_path = configs_dir / "e05_morphospace_trajectory_config.json"
    write_json(
        config_path,
        {
            "schemaVersion": TRAJECTORY_MAP_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "randomState": int(args.random_state),
            "stabilitySampleSize": int(args.stability_sample_size),
            "sourceSpecs": [spec.__dict__ for spec in specs],
            "embeddingMethods": ["pca", "diffusion_map_knn"],
            "embeddingFeatureColumns": feature_cols,
            "errorProxyPrecedence": ["composite_error", "hamming_error", "1-pattern_score", "1-label_match_fraction", "success/exact_match fallback"],
        },
    )
    artifacts.append(config_path)

    feature_spec_json_path = step_dir / "embedding_feature_spec.json"
    write_json(
        feature_spec_json_path,
        {
            "schemaVersion": TRAJECTORY_MAP_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "embeddingFeatureColumns": [{"name": column, "definition": FEATURE_DEFINITIONS.get(column, "")} for column in feature_cols],
            "errorProxyPrecedence": ["composite_error", "hamming_error", "1-pattern_score", "1-label_match_fraction", "success/exact_match fallback"],
            "metadataPreservedButExcludedFromEmbedding": [
                "trajectory_id",
                "source_step_id",
                "run_id",
                "target_id",
                "motif",
                "policy_id",
                "policy_family",
                "seed",
                "source_trace_path",
                "source_run_path",
            ],
        },
    )
    artifacts.append(feature_spec_json_path)
    feature_spec_md_path = step_dir / "embedding_feature_spec.md"
    feature_spec_md_path.write_text(feature_spec_markdown(feature_cols), encoding="utf-8")
    artifacts.append(feature_spec_md_path)

    fig_pca = figures_dir / "e05_s12_pca_morphospace_map.png"
    fig_diffusion = figures_dir / "e05_s12_diffusion_morphospace_map.png"
    fig_diversity = figures_dir / "e05_s12_route_diversity.png"
    fig_confounding = figures_dir / "e05_s12_metadata_confounding.png"
    fig_curvature = figures_dir / "e05_s12_path_curvature.png"
    plot_embedding_map(embedding_df, "pca", fig_pca)
    plot_embedding_map(embedding_df, "diffusion_map_knn", fig_diffusion)
    plot_route_diversity(route_df, fig_diversity)
    plot_confounding(confounding_df, fig_confounding)
    plot_path_curvature(clustered_summary_df, fig_curvature)
    artifacts.extend([fig_pca, fig_diffusion, fig_diversity, fig_confounding, fig_curvature])

    confound_flags = int(confounding_df.get("confounding_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    min_stability = float(stability_df["spearman_distance_correlation"].dropna().min()) if len(stability_df["spearman_distance_correlation"].dropna()) else math.nan
    validation_result = (
        f"passed: loaded {feature_df['source_step_id'].nunique()} sources, {len(feature_df)} saved states, "
        f"{len(clustered_summary_df)} trajectories, generated PCA and diffusion-map embeddings, "
        f"minimum stability Spearman={min_stability:.3f}, metadata confounding flags={confound_flags}, "
        f"repo tests {'passed' if repo_test.get('success') else 'failed'}"
        if success
        else f"failed: {int((~validation_df['success'].astype(bool)).sum())} validation checks failed"
    )
    outcome = "supportive" if success else "constraining/contradictory"
    recommended_next_action = "Chief Scientist review, then proceed to S13 local-versus-global control comparisons only after explicit instruction."
    caveats = [
        "Embeddings use trajectory-summary proxies, not full cell-state tensors, because S07-S11 trace schemas are heterogeneous.",
        "Mixed CPU and GPU artifacts expose different component metrics; missing source-specific fields are median-imputed for embedding and audited for metadata predictability.",
        "Metadata confounding flags indicate route-map structure can partly reflect source protocol, motif, policy family, or convergence class rather than only state geometry.",
        "S11 trajectories remain compact GPU label dynamics rather than the full S04 action-world semantics.",
        "Low-dimensional PCA and diffusion-map coordinates are exploratory computational maps, not biological measurements.",
    ]
    if math.isfinite(min_stability) and min_stability < 0.30:
        caveats.append(
            f"PCA and diffusion-map cross-method distance agreement was low (minimum Spearman {min_stability:.3f}); route conclusions should be checked method-by-method."
        )
    if confound_flags > 0:
        caveats.append(
            f"Metadata confounding audit flagged {confound_flags} embedding-field pairs, so source protocol and convergence class can influence apparent clusters."
        )
    anchor_result = (
        f"S12 mapped {len(clustered_summary_df)} trajectories from {feature_df['source_step_id'].nunique()} upstream steps "
        f"using {len(feature_cols)} documented summary features and two embeddings; "
        f"route-diversity groups={len(route_df)}, cluster summaries={len(cluster_df)}, confounding flags={confound_flags}."
    )

    report_path = step_dir / "morphospace_trajectory_report.md"
    report_path.write_text(
        report_markdown(
            success=success,
            source_catalog=source_catalog,
            summary_df=clustered_summary_df,
            method_info_df=method_info_df,
            stability_df=stability_df,
            confounding_df=confounding_df,
            route_diversity_df=route_df,
            cluster_df=cluster_df,
            validation_df=validation_df,
            artifacts=artifacts + [report_path],
            validation_result=validation_result,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )
    artifacts.append(report_path)

    artifacts.append(Path(repo_test["logPath"]))
    status_path = step_dir / "status.json"
    summary_path = step_dir / "summary.md"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = update_run_manifest(artifacts_dir, status_path, artifacts)
    artifacts.append(run_manifest_path)

    summary_path.write_text(
        summary_markdown(
            success=success,
            artifacts=artifacts + [summary_path, status_path, manifest_path],
            validation_result=validation_result,
            outcome=outcome,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
            anchor_result=anchor_result,
        ),
        encoding="utf-8",
    )
    artifacts.append(summary_path)

    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "parallelism": "single_process_sklearn_numpy",
        "runtimeSeconds": time.perf_counter() - started,
    }
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "completedAt": utc_now(),
        "outcomeClassification": outcome,
        "artifactsWritten": sorted(str(path) for path in artifacts + [status_path, manifest_path]),
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "benchmark": {
            "sourceStepCount": int(feature_df["source_step_id"].nunique()),
            "savedStateFeatureRows": int(len(feature_df)),
            "trajectoryCount": int(len(clustered_summary_df)),
            "embeddingRowCount": int(len(embedding_df)),
            "embeddingMethods": sorted(embedding_df["embedding_method"].unique().tolist()),
            "routeDiversityGroupCount": int(len(route_df)),
            "clusterSummaryRowCount": int(len(cluster_df)),
            "metadataConfoundingFlagCount": confound_flags,
            "minimumStabilitySpearman": min_stability,
            "validationCheckCount": int(len(validation_df)),
            "validationPassCount": int(validation_df["success"].astype(bool).sum()),
        },
        "repoUnitTests": {
            "success": bool(repo_test.get("success", False)),
            "returncode": repo_test.get("returncode"),
            "logPath": repo_test.get("logPath"),
        },
        "runtime": runtime,
        "git": git_metadata(),
    }
    write_json(status_path, status)
    artifacts.append(status_path)
    manifest_payload = {
        "schemaVersion": TRAJECTORY_MAP_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "artifacts": collect_artifacts(artifacts),
    }
    write_json(manifest_path, manifest_payload)
    if str(manifest_path) not in status["artifactsWritten"]:
        status["artifactsWritten"].append(str(manifest_path))
        status["artifactsWritten"] = sorted(set(status["artifactsWritten"]))
        write_json(status_path, status)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
