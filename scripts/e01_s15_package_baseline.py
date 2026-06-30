#!/usr/bin/env python3
"""Package the E01 replication baseline for downstream experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


STEP_ID = "S15"
STEP_NUMBER = 15
EXPERIMENT_ID = "E01"
TITLE = "Package the baseline"
PAPER_MARKDOWN_PATH = Path(
    "/workspace/input-attachments/f93afdc5-f2e5-4ecc-80bb-e088f93acf3c/pdf-markdown.md"
)
NON_REPLICATION_CLAIMS = {
    "figure4_exact_fold_change_magnitudes",
    "figure5_all_cell_view_lower_error_than_traditional",
    "figure8_unique_aggregation_exact_peak_magnitudes",
}
CAVEAT_THEMES = [
    "Traditional Bubble, Insertion, and Selection baselines are reconstructed because the public repository does not include standalone traditional-generation runners.",
    "Cell-view comparison counts use the S05 actionable StatusProbe comparison proxy, not a complete read census.",
    "Several later steps use deterministic public-method wrappers to preserve frozen configs and assignment semantics rather than the original public threaded runner.",
    "Aggregation uses the frozen S03/S09 left-neighbor primary denominator and preserves the S10 right-neighbor sensitivity field.",
    "Duplicate-value Sortedness uses nondecreasing tie semantics; strict unique-rank Sortedness is not meaningful for ten copies each of values 1..10.",
    "Opposite-direction stable-equilibrium stopping criteria are reconstructed because executable thresholds are absent from the paper and public code.",
    "S14 scaled selected high-value divergences at N=1000, not every S04-S12 condition and not N=10000.",
    "All biological or morphogenesis interpretations remain computational proxy evidence only.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
    parser.add_argument("--paper-markdown-path", type=Path, default=PAPER_MARKDOWN_PATH)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_status(repo_dir: Path) -> str:
    try:
        result = subprocess.run(["git", "status", "--short"], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    stringified = df.copy()
    for col in stringified.columns:
        if pd.api.types.is_float_dtype(stringified[col]):
            stringified[col] = stringified[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    stringified = stringified.astype("string").fillna("").astype(str)
    headers = list(stringified.columns)
    rows = stringified.values.tolist()
    widths = [max(len(str(header)), *(len(row[col_idx]) for row in rows)) for col_idx, header in enumerate(headers)]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    divider_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body_lines = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body_lines])


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def required_input_paths(artifacts_dir: Path, research_plan_path: Path, paper_markdown_path: Path) -> dict[str, Path]:
    step_reports = {
        f"s{step:02d}_full_results": artifacts_dir / f"research_steps/S{step:02d}/research_step_full_results.md"
        for step in range(1, 15)
    }
    return {
        "research_plan": research_plan_path,
        "paper_markdown": paper_markdown_path,
        "run_manifest": artifacts_dir / "run_manifest.json",
        "original_repo_manifest": artifacts_dir / "code/original_repo_manifest.json",
        "baseline_config": artifacts_dir / "configs/e01_baseline_configs.json",
        "condition_matrix": artifacts_dir / "tables/e01_condition_matrix.csv",
        "code_paper_mapping": artifacts_dir / "tables/code_paper_mapping.csv",
        "codebase_map": artifacts_dir / "reports/e01_codebase_map.md",
        "statistical_methods": artifacts_dir / "reports/e01_statistical_methods.md",
        "divergence_log": artifacts_dir / "reports/e01_divergence_log.md",
        "replication_status_csv": artifacts_dir / "tables/e01_replication_status.csv",
        "replication_status_parquet": artifacts_dir / "results/e01_replication_status.parquet",
        "noise_estimates": artifacts_dir / "tables/e01_noise_estimates.csv",
        "scaled_replication": artifacts_dir / "results/e01_scaled_replication.parquet",
        "scaled_aggregation_curves": artifacts_dir / "results/e01_scaled_aggregation_curves.parquet",
        "figure3_summary": artifacts_dir / "results/e01_figure3_summary.csv",
        "efficiency_numeric": artifacts_dir / "tables/e01_efficiency_numeric_table.csv",
        "z_tests": artifacts_dir / "tables/e01_z_tests.csv",
        "bootstrap_intervals": artifacts_dir / "tables/e01_bootstrap_intervals.csv",
        "frozen_numeric": artifacts_dir / "tables/e01_frozen_cell_robustness_numeric_table.csv",
        "frozen_comparison": artifacts_dir / "tables/e01_frozen_cell_robustness_comparison_table.csv",
        "dg_numeric": artifacts_dir / "tables/e01_dg_numeric_table.csv",
        "dg_comparison": artifacts_dir / "tables/e01_dg_comparison_table.csv",
        "chimera_efficiency": artifacts_dir / "tables/e01_chimera_efficiency_numeric_table.csv",
        "aggregation_peak": artifacts_dir / "tables/e01_aggregation_peak_table.csv",
        "duplicate_summary": artifacts_dir / "tables/e01_duplicate_aggregation_summary.csv",
        "duplicate_within_block": artifacts_dir / "tables/e01_duplicate_within_block_summary.csv",
        "conflict_summary": artifacts_dir / "tables/e01_conflict_equilibria_summary.csv",
        "figure3_trace": artifacts_dir / "traces/e01_figure3_trajectories.parquet",
        "frozen_trace": artifacts_dir / "traces/e01_frozen_cell_trajectories.parquet",
        "same_goal_trace": artifacts_dir / "traces/e01_same_goal_chimeras.parquet",
        "duplicate_trace": artifacts_dir / "traces/e01_duplicate_value_chimeras.parquet",
        "opposite_trace": artifacts_dir / "traces/e01_opposite_direction_chimeras.parquet",
        "figure3_png": artifacts_dir / "figures/e01/figure3_reproduction.png",
        "figure4_png": artifacts_dir / "figures/e01/figure4_efficiency_reproduction.png",
        "figure5_png": artifacts_dir / "figures/e01/figure5_frozen_robustness.png",
        "figure7_png": artifacts_dir / "figures/e01/figure7_dg_reproduction.png",
        "figure8_aggregation_png": artifacts_dir / "figures/e01/figure8_aggregation_reproduction.png",
        "figure8_chimera_png": artifacts_dir / "figures/e01/figure8_chimera_sortedness.png",
        "figure8_duplicate_png": artifacts_dir / "figures/e01/figure8_duplicate_chimeras.png",
        "figure9_png": artifacts_dir / "figures/e01/figure9_unique_conflict.png",
        "figure10_png": artifacts_dir / "figures/e01/figure10_repeated_conflict.png",
        "s14_scaled_png": artifacts_dir / "figures/e01/scaled_uncertainty_summary.png",
        **step_reports,
    }


def trace_paths(artifacts_dir: Path) -> dict[str, Path]:
    return {
        "figure3_unperturbed": artifacts_dir / "traces/e01_figure3_trajectories.parquet",
        "frozen_cell": artifacts_dir / "traces/e01_frozen_cell_trajectories.parquet",
        "same_goal_chimeras": artifacts_dir / "traces/e01_same_goal_chimeras.parquet",
        "duplicate_value_chimeras": artifacts_dir / "traces/e01_duplicate_value_chimeras.parquet",
        "opposite_direction_chimeras": artifacts_dir / "traces/e01_opposite_direction_chimeras.parquet",
        "s14_scaled_replication_runs": artifacts_dir / "results/e01_scaled_replication.parquet",
        "s14_scaled_aggregation_curves": artifacts_dir / "results/e01_scaled_aggregation_curves.parquet",
    }


def add_core_row(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    template = {
        "experiment_id": EXPERIMENT_ID,
        "packaging_step_id": STEP_ID,
        "source_step_id": None,
        "result_family": None,
        "result_id": None,
        "figure_or_section": None,
        "claim_id": None,
        "condition_id": None,
        "condition_key": None,
        "algorithm": None,
        "mode_or_mix": None,
        "metric": None,
        "value": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "unit": None,
        "classification": None,
        "source_artifact": None,
        "notes": None,
        "caveats": None,
    }
    template.update(kwargs)
    rows.append(template)


def build_core_results(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    status = read_csv(paths["replication_status_csv"])
    noise = read_csv(paths["noise_estimates"])
    figure3 = read_csv(paths["figure3_summary"])
    efficiency = read_csv(paths["efficiency_numeric"])
    frozen = read_csv(paths["frozen_comparison"])
    dg = read_csv(paths["dg_comparison"])
    aggregation = read_csv(paths["aggregation_peak"])
    duplicate = read_csv(paths["duplicate_summary"])
    conflict = read_csv(paths["conflict_summary"])

    for row in status.itertuples(index=False):
        add_core_row(
            rows,
            source_step_id="S13",
            result_family="claim_status",
            result_id=row.claim_id,
            figure_or_section=row.figure_or_section,
            claim_id=row.claim_id,
            metric="replication_classification",
            classification=row.classification,
            source_artifact=str(paths["replication_status_csv"]),
            notes=row.replication_evidence,
            caveats=row.caveats,
        )

    for row in noise.itertuples(index=False):
        add_core_row(
            rows,
            source_step_id="S14",
            result_family=row.claim_family,
            result_id=row.condition_key,
            claim_id=row.source_claim_id,
            condition_key=row.condition_key,
            algorithm=row.algorithm,
            mode_or_mix=row.mode_or_mix,
            metric=row.metric,
            value=float(row.scaled_effect),
            ci95_low=float(row.scaled_ci95_low),
            ci95_high=float(row.scaled_ci95_high),
            unit="ratio_or_difference_or_percent",
            classification=row.stability_result,
            source_artifact=str(paths["noise_estimates"]),
            notes=f"paper_reference_effect={row.paper_reference_effect}; paper_inside_scaled_ci95={row.paper_inside_scaled_ci95}",
            caveats="N=1000 selected scaled sensitivity; deterministic wrapper and count caveats remain.",
        )

    for row in figure3.itertuples(index=False):
        add_core_row(
            rows,
            source_step_id="S04",
            result_family="figure3_trajectory_completion",
            result_id=f"{row.algorithm}_{row.mode}",
            figure_or_section="Figure 3",
            algorithm=row.algorithm,
            mode_or_mix=row.mode,
            metric="mean_final_sortedness_percent",
            value=float(row.mean_final_sortedness_percent),
            unit="percent",
            classification="exact" if bool(row.all_final_sorted) else "not replicated",
            source_artifact=str(paths["figure3_summary"]),
            notes=f"n_runs={row.repetitions_observed}; all_final_sorted={row.all_final_sorted}",
            caveats="Traditional modes are reconstructed baselines.",
        )

    for row in efficiency[efficiency["included_in_figure4"] == True].itertuples(index=False):  # noqa: E712
        add_core_row(
            rows,
            source_step_id="S05",
            result_family="figure4_efficiency",
            result_id=f"{row.algorithm}_{row.count_metric}",
            figure_or_section="Figure 4",
            algorithm=row.algorithm,
            metric=row.count_metric,
            value=float(row.ratio_of_means_cell_over_traditional),
            ci95_low=float(row.ratio_of_means_ci95_low),
            ci95_high=float(row.ratio_of_means_ci95_high),
            unit="cell_view_over_traditional_ratio",
            classification=row.replication_status,
            source_artifact=str(paths["efficiency_numeric"]),
            notes=f"paper_ratio={row.paper_ratio_cell_over_traditional}; direction_matches_paper={row.direction_matches_paper}",
            caveats="Comparison counts use S05 actionable-comparison proxy.",
        )

    for row in frozen.itertuples(index=False):
        add_core_row(
            rows,
            source_step_id="S07",
            result_family="figure5_frozen_cell",
            result_id=f"{row.algorithm}_{row.frozen_semantics}_f{row.frozen_count}",
            figure_or_section="Figure 5",
            algorithm=row.algorithm,
            mode_or_mix=row.frozen_semantics,
            metric="cell_minus_traditional_final_monotonicity_error",
            value=float(row.cell_minus_traditional_mean_error),
            unit="monotonicity_error_count",
            classification="cell_lower" if bool(row.cell_lower_error_than_traditional) else "not_cell_lower_or_tied",
            source_artifact=str(paths["frozen_comparison"]),
            caveats="S07 deterministic wrapper and reconstructed traditional Frozen Cell caveats remain.",
        )

    for row in dg.itertuples(index=False):
        add_core_row(
            rows,
            source_step_id="S08",
            result_family="figure7_delayed_gratification",
            result_id=f"{row.algorithm}_{row.plot_frozen_semantics}_f{row.frozen_count}",
            figure_or_section="Figure 7",
            algorithm=row.algorithm,
            mode_or_mix=row.plot_frozen_semantics,
            metric="cell_minus_traditional_mean_dg",
            value=float(row.cell_minus_traditional_mean_dg),
            unit="delayed_gratification_area",
            classification="cell_greater" if bool(row.cell_greater_than_traditional) else "cell_not_greater",
            source_artifact=str(paths["dg_comparison"]),
            caveats="DG formula ambiguity preserved; primary stuck layer was supportive.",
        )

    for row in aggregation.itertuples(index=False):
        add_core_row(
            rows,
            source_step_id="S10",
            result_family="figure8_unique_aggregation",
            result_id=row.algotype_mix,
            condition_id=row.condition_id,
            figure_or_section="Figure 8",
            mode_or_mix=row.algotype_mix,
            metric="mean_curve_peak_aggregation_left_neighbor_percent",
            value=float(row.mean_curve_peak_aggregation_left_neighbor_percent),
            ci95_low=float(row.mean_curve_peak_left_neighbor_ci95_low),
            ci95_high=float(row.mean_curve_peak_left_neighbor_ci95_high),
            unit="percent",
            classification="above_control" if bool(row.above_negative_control_peak) else "control_or_not_above_control",
            source_artifact=str(paths["aggregation_peak"]),
            notes=f"paper_peak_percent={row.paper_peak_percent}; peak_minus_paper={row.peak_percent_minus_paper}",
            caveats="Exact paper peak magnitudes were not fully reproduced under left-neighbor primary definition.",
        )

    for row in duplicate.itertuples(index=False):
        add_core_row(
            rows,
            source_step_id="S11",
            result_family="figure8_duplicate_value_aggregation",
            result_id=row.algotype_mix,
            condition_id=row.condition_id,
            figure_or_section="Figure 8 duplicate-value sensitivity",
            mode_or_mix=row.algotype_mix,
            metric="final_mean_aggregation_left_neighbor_percent",
            value=float(row.final_mean_aggregation_left_neighbor_percent),
            unit="percent",
            classification="above_unique_peak" if bool(row.final_above_s10_unique_peak) else "not_above_unique_peak",
            source_artifact=str(paths["duplicate_summary"]),
            notes=f"within_equal_value_same_label_pair_percent={row.final_mean_within_equal_value_same_label_pair_percent}",
            caveats="Duplicate Sortedness uses nondecreasing ties.",
        )

    for row in conflict.itertuples(index=False):
        add_core_row(
            rows,
            source_step_id="S12",
            result_family="figure9_10_opposite_direction",
            result_id=row.algotype_mix,
            condition_id=row.condition_id,
            figure_or_section="Figures 9-10",
            mode_or_mix=row.algotype_mix,
            metric="mean_final_sortedness_percent",
            value=float(row.mean_final_sortedness_percent),
            unit="percent",
            classification="paper_order_supported" if bool(row.paper_dominance_order_supported) else "paper_order_not_supported",
            source_artifact=str(paths["conflict_summary"]),
            notes=f"paper_reported_unique_final_sortedness_percent={row.paper_reported_unique_final_sortedness_percent}",
            caveats="Stable-equilibrium stopping criteria are reconstructed.",
        )

    return pd.DataFrame(rows)


def build_trace_schema(paths: dict[str, Path]) -> dict[str, Any]:
    trace_artifacts: list[dict[str, Any]] = []
    for family, path in paths.items():
        parquet_file = pq.ParquetFile(path)
        schema = parquet_file.schema_arrow
        trace_artifacts.append(
            {
                "traceFamily": family,
                "path": str(path),
                "sha256": sha256_file(path),
                "sizeBytes": path.stat().st_size,
                "rowCount": parquet_file.metadata.num_rows,
                "numRowGroups": parquet_file.metadata.num_row_groups,
                "columns": [
                    {
                        "name": field.name,
                        "type": str(field.type),
                        "nullable": bool(field.nullable),
                    }
                    for field in schema
                ],
            }
        )
    return {
        "researchStepId": STEP_ID,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": utc_now(),
        "description": "Schema inventory for reusable E01 trace and scaled-run artifacts. Bulk traces remain in their original step outputs.",
        "traceArtifacts": trace_artifacts,
    }


def build_artifact_index(paths: dict[str, Path], output_paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, path in {**paths, **output_paths}.items():
        rows.append(
            {
                "label": label,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else np.nan,
                "sha256": sha256_file(path) if path.exists() else None,
                "producer_or_role": infer_producer(label, path),
            }
        )
    return pd.DataFrame(rows)


def infer_producer(label: str, path: Path) -> str:
    for step in range(1, 16):
        token = f"S{step:02d}"
        if token in str(path) or label.lower().startswith(f"s{step:02d}"):
            return token
    if "figure" in label:
        return "S04-S14"
    if label in {"research_plan", "paper_markdown"}:
        return "input"
    return "E01"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_checksum_file(checksum_path: Path, paths: list[Path]) -> None:
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    order: list[str] = []
    if checksum_path.exists():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if "  " not in line:
                continue
            digest, path_str = line.split("  ", 1)
            existing[path_str] = digest
            order.append(path_str)
    for path in paths:
        if not path.exists() or path == checksum_path:
            continue
        resolved = str(path)
        existing[resolved] = sha256_file(path)
        if resolved not in order:
            order.append(resolved)
    checksum_path.write_text("".join(f"{existing[path]}  {path}\n" for path in order), encoding="utf-8")


def validate_checksum_entries(checksum_path: Path, paths: list[Path]) -> tuple[bool, list[str]]:
    if not checksum_path.exists():
        return False, [f"missing checksum file: {checksum_path}"]
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if "  " in line:
            digest, path_str = line.split("  ", 1)
            entries[path_str] = digest
    errors: list[str] = []
    for path in paths:
        if not path.exists():
            errors.append(f"missing path for checksum validation: {path}")
            continue
        digest = entries.get(str(path))
        if digest is None:
            errors.append(f"missing checksum entry: {path}")
        elif digest != sha256_file(path):
            errors.append(f"checksum mismatch: {path}")
    return not errors, errors


def run_regression_tests(repo_dir: Path, artifacts_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "e01_regression_unittest.log"
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests/e01_regression", "-p", "test_*.py"]
    env = os.environ.copy()
    env["ARTIFACTS_DIR"] = str(artifacts_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True, env=env)
    elapsed = time.monotonic() - started
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(command),
                "",
                "STDOUT:",
                result.stdout,
                "STDERR:",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "researchStepId": STEP_ID,
        "command": " ".join(command),
        "returnCode": result.returncode,
        "success": result.returncode == 0,
        "elapsedSeconds": elapsed,
        "logPath": str(log_path),
    }
    write_json(output_dir / "regression_test_results.json", summary)
    write_json(
        output_dir / "regression_test_manifest.json",
        {
            "researchStepId": STEP_ID,
            "suiteLocation": str(repo_dir / "tests/e01_regression"),
            "runner": "python -m unittest discover",
            "artifactsDir": str(artifacts_dir),
            "resultSummary": summary,
            "note": "Test source remains in the repository; artifact directory stores runner logs and manifest only.",
        },
    )
    return summary


def artifact_validation(paths: dict[str, Path], output_paths: dict[str, Path], checksum_path: Path) -> dict[str, Any]:
    all_paths = [*paths.values(), *output_paths.values()]
    missing = [str(path) for path in all_paths if not path.exists()]
    empty = [str(path) for path in all_paths if path.exists() and path.stat().st_size == 0]
    checksum_ok, checksum_errors = validate_checksum_entries(checksum_path, all_paths)
    return {
        "missingArtifactPaths": missing,
        "emptyArtifactPaths": empty,
        "referencedArtifactsExist": not missing,
        "referencedArtifactsNonEmpty": not empty,
        "checksumsValidated": checksum_ok,
        "checksumErrors": checksum_errors,
    }


def validate_package(
    core_results: pd.DataFrame,
    trace_schema: dict[str, Any],
    regression_summary: dict[str, Any],
    artifact_checks: dict[str, Any],
    status: pd.DataFrame,
    noise: pd.DataFrame,
) -> dict[str, Any]:
    nonrep = set(status.loc[status["classification"] == "not replicated", "claim_id"])
    trace_families = {entry["traceFamily"] for entry in trace_schema["traceArtifacts"]}
    required_families = {
        "figure3_unperturbed",
        "frozen_cell",
        "same_goal_chimeras",
        "duplicate_value_chimeras",
        "opposite_direction_chimeras",
    }
    errors: list[str] = []
    warnings: list[str] = []
    if nonrep != NON_REPLICATION_CLAIMS:
        errors.append(f"S13 non-replication set changed: {sorted(nonrep)}")
    if core_results.empty or len(core_results) < 80:
        errors.append(f"Core results table unexpectedly small: {len(core_results)} rows")
    if not required_families.issubset(trace_families):
        errors.append(f"Trace schema missing families: {sorted(required_families - trace_families)}")
    if any(int(entry["rowCount"]) <= 0 for entry in trace_schema["traceArtifacts"]):
        errors.append("At least one trace schema entry has zero rows.")
    if not regression_summary["success"]:
        errors.append("Regression tests failed.")
    if not artifact_checks["referencedArtifactsExist"] or not artifact_checks["referencedArtifactsNonEmpty"]:
        errors.append("One or more referenced artifacts are missing or empty.")
    if not artifact_checks["checksumsValidated"]:
        errors.extend(artifact_checks["checksumErrors"])
    fig4_unsupported = noise[
        (noise["claim_family"] == "figure4_efficiency")
        & (noise["paper_inside_scaled_ci95"] == False)  # noqa: E712
        & (~noise["condition_key"].isin(["bubble_swap_only_steps", "insertion_swap_only_steps"]))
    ]
    if len(fig4_unsupported) < 4:
        errors.append("S14 Figure 4 non-replication evidence was not preserved in noise estimates.")
    if warnings:
        status_label = "completed_with_warnings"
    else:
        status_label = "completed_with_caveats"
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": not errors,
        "status": status_label if not errors else "failed_validation",
        "validationResult": "passed_package_validation_with_caveats" if not errors else "failed_package_validation",
        "outcomeClassification": "supportive",
        "validationErrors": errors,
        "validationWarnings": warnings,
        "coreResultRows": int(len(core_results)),
        "traceSchemaArtifactCount": int(len(trace_schema["traceArtifacts"])),
        "regressionTestsPassed": bool(regression_summary["success"]),
        "artifactValidation": artifact_checks,
        "nonReplicationClaimsCarriedForward": sorted(nonrep),
        "caveatsOrBlockers": CAVEAT_THEMES,
        "recommendedNextAction": "Hand control back to the Chief workflow. E01 is ready for report-bundle generation and downstream E02 launch only after explicit instruction; do not start E02 inside S15.",
    }


def write_artifact_manifest(path: Path, artifact_paths: dict[str, Path], validation: dict[str, Any], repo_dir: Path) -> None:
    payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "validationResult": validation["validationResult"],
        "artifacts": [
            {
                "label": label,
                "path": str(artifact_path),
                "sha256": None if artifact_path == path else (sha256_file(artifact_path) if artifact_path.exists() else None),
                "sizeBytes": None if artifact_path == path else (artifact_path.stat().st_size if artifact_path.exists() else None),
            }
            for label, artifact_path in artifact_paths.items()
        ],
    }
    write_json(path, payload)


def update_run_manifest(
    artifacts_dir: Path,
    artifact_paths: dict[str, Path],
    validation: dict[str, Any],
    repo_dir: Path,
    command: list[str],
) -> None:
    manifest_path = artifacts_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest["researchStepId"] = STEP_ID
    manifest["updatedAtUtc"] = utc_now()
    manifest.setdefault("artifacts", {}).update({label: str(path) for label, path in artifact_paths.items()})
    manifest.setdefault("checksums", {})
    for path in artifact_paths.values():
        if path.exists():
            manifest["checksums"][str(path)] = sha256_file(path)
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": TITLE,
        "status": validation["status"],
        "success": validation["success"],
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
        "artifactsWritten": [str(path) for path in artifact_paths.values()],
        "caveatsOrBlockers": validation["caveatsOrBlockers"],
        "recommendedNextAction": validation["recommendedNextAction"],
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "script": str(repo_dir / "scripts/e01_s15_package_baseline.py"),
        "command": " ".join(command),
        "summary": {
            "coreResultRows": validation["coreResultRows"],
            "traceSchemaArtifactCount": validation["traceSchemaArtifactCount"],
            "regressionTestsPassed": validation["regressionTestsPassed"],
            "nonReplicationClaimsCarriedForward": validation["nonReplicationClaimsCarriedForward"],
        },
    }
    write_json(manifest_path, manifest)


def top_summary_md(validation: dict[str, Any], artifact_paths: dict[str, Path], lay_summary: str) -> str:
    return f"""## Top Summary

- Step ID: {STEP_ID}
- Completion status: {validation["status"]}
- Artifacts written: {", ".join(str(path) for path in artifact_paths.values())}
- Validation result: {validation["validationResult"]}
- Outcome classification: {validation["outcomeClassification"]}
- Caveats or blockers: {"; ".join(validation["caveatsOrBlockers"])}
- Lay summary: {lay_summary}
- Recommended next action: {validation["recommendedNextAction"]}
"""


def write_replication_report(
    path: Path,
    validation: dict[str, Any],
    artifact_paths: dict[str, Path],
    status: pd.DataFrame,
    noise: pd.DataFrame,
    core_results: pd.DataFrame,
) -> None:
    class_counts = status["classification"].value_counts().rename_axis("classification").reset_index(name="claim_count")
    nonrep = status[status["claim_id"].isin(NON_REPLICATION_CLAIMS)][
        ["claim_id", "figure_or_section", "classification", "divergence_cause"]
    ]
    scaled = noise[
        [
            "condition_key",
            "claim_family",
            "scaled_effect",
            "scaled_ci95_low",
            "scaled_ci95_high",
            "paper_reference_effect",
            "paper_inside_scaled_ci95",
            "stability_result",
        ]
    ]
    lay_summary = (
        "E01 is now packaged as a reusable replication baseline. The package reproduces many qualitative and statistical directions, "
        "but it explicitly preserves three non-replicated claims from S13/S14: exact Figure 4 fold-change magnitudes, broad Figure 5 "
        "cell-view Frozen Cell superiority, and exact Figure 8 unique-value Aggregation peak magnitudes."
    )
    report = f"""# E01 Replication Report

{top_summary_md(validation, artifact_paths, lay_summary)}

## Scope

This report packages E01, the full core replication of Zhang, Goldstein, and Levin's sorting-array morphogenesis model. It is a baseline for later Experiments, not a report-bundle generation step and not an E02 launch.

## Lay Summary

The replication pipeline rebuilt the public code environment, froze replacement seed banks, reran the major figure families, classified the paper claims, scaled the most important divergences, and packaged the outputs into a traceable baseline. The main result is mixed: many behaviors reproduce qualitatively or statistically, but three important exact or broad claims remain unsupported under the frozen public-code definitions.

## Claim Status

{dataframe_to_markdown(class_counts)}

Non-replicated or magnitude-divergent claims carried forward:

{dataframe_to_markdown(nonrep)}

## Key Evidence

- S04: all six unperturbed traditional/cell-view algorithm runs reached final Sortedness 100% over 100 matched repeats per condition.
- S05-S06: Figure 4 directions reproduced for the available count definitions, but exact fold-change magnitudes did not reproduce under the actionable-comparison count convention.
- S07: Frozen Cell robustness was constraining because cell-view did not beat reconstructed traditional in every comparison.
- S08: primary stuck Frozen Cell Delayed Gratification directions were supportive, with documented formula ambiguity.
- S09-S10: same-goal chimeras sorted and showed Aggregation above same-algorithm control/chance, but exact unique-value peak magnitudes did not reproduce.
- S11: duplicate-value chimeras sorted under nondecreasing tie semantics and preserved within-equal-value-block aggregation evidence.
- S12: opposite-direction chimeras reproduced the unique-value dominance order and final Sortedness values close to paper reports under reconstructed stopping criteria.
- S13-S14: the non-replications persisted after tolerance classification and N=1000 selected scaled sweeps.

## S14 Scaled Divergence Evidence

{dataframe_to_markdown(scaled)}

## Core Results Table

The machine-readable baseline table is `{artifact_paths["core_results_parquet"]}` and contains `{len(core_results)}` long-form result rows spanning claim statuses, efficiency ratios, Frozen Cell comparisons, Delayed Gratification, unique and duplicate Aggregation, opposite-direction conflict results, and S14 scaled estimates.

## Caveats

{chr(10).join(f"- {item}" for item in CAVEAT_THEMES)}

## Provenance

- Repository commit at package time: `{validation["repositoryCommitAtRunTime"]}`
- Repository status during package generation: `{validation["repositoryStatusDuringReportGeneration"]}`
- Python: `{sys.version.splitlines()[0]}`
- Platform: `{platform.platform()}`
- Input paper markdown: `{PAPER_MARKDOWN_PATH}`
- Global manifest: `/artifacts/run_manifest.json`
- Checksum file: `/artifacts/checksums/sha256sums.txt`

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    path.write_text(report, encoding="utf-8")


def write_trace_schema_report(path: Path, validation: dict[str, Any], artifact_paths: dict[str, Path], trace_schema: dict[str, Any]) -> None:
    rows = pd.DataFrame(
        [
            {
                "trace_family": entry["traceFamily"],
                "path": entry["path"],
                "row_count": entry["rowCount"],
                "column_count": len(entry["columns"]),
                "sha256": entry["sha256"],
            }
            for entry in trace_schema["traceArtifacts"]
        ]
    )
    lay_summary = "The E01 trace handoff keeps bulk traces in their original Parquet files and records row counts, column names, types, hashes, and family labels for downstream reuse."
    report = f"""# E01 Trace Schema Handoff

{top_summary_md(validation, artifact_paths, lay_summary)}

## Trace Families

{dataframe_to_markdown(rows)}

## Schema Details

```json
{json.dumps(trace_schema, indent=2)}
```
"""
    path.write_text(report, encoding="utf-8")


def write_report_bundle_handoff(
    path: Path,
    validation: dict[str, Any],
    artifact_paths: dict[str, Path],
    status: pd.DataFrame,
    artifact_index: pd.DataFrame,
) -> None:
    nonrep = status[status["claim_id"].isin(NON_REPLICATION_CLAIMS)][
        ["claim_id", "figure_or_section", "classification", "divergence_cause", "caveats"]
    ]
    featured = artifact_index[
        artifact_index["label"].isin(
            [
                "replication_report",
                "core_results_parquet",
                "trace_schema_json",
                "trace_schema_report",
                "provenance_manifest",
                "regression_test_results",
                "divergence_log",
            ]
        )
    ][["label", "path", "sha256"]]
    lay_summary = "This handoff tells the Chief workflow which E01 package artifacts to use for report-bundle generation and which caveats must remain visible."
    report = f"""# E01 Report-Bundle Handoff

{top_summary_md(validation, artifact_paths, lay_summary)}

## Bundle Inputs

{dataframe_to_markdown(featured)}

## Claims To Surface Explicitly

{dataframe_to_markdown(nonrep)}

## Required Caveats For Any Report Bundle

{chr(10).join(f"- {item}" for item in CAVEAT_THEMES)}

## Validation Evidence

- Regression tests passed: `{validation["regressionTestsPassed"]}`
- Core result rows: `{validation["coreResultRows"]}`
- Trace/schema artifacts indexed: `{validation["traceSchemaArtifactCount"]}`
- Referenced artifacts exist: `{validation["artifactValidation"]["referencedArtifactsExist"]}`
- Checksums validated: `{validation["artifactValidation"]["checksumsValidated"]}`

## Stop Boundary

S15 packages E01 only. Do not start E02 or generate the final report bundle from inside this step.
"""
    path.write_text(report, encoding="utf-8")


def write_full_results_report(
    path: Path,
    validation: dict[str, Any],
    artifact_paths: dict[str, Path],
    input_paths: dict[str, Path],
    core_results: pd.DataFrame,
    trace_schema: dict[str, Any],
    regression_summary: dict[str, Any],
    status: pd.DataFrame,
    noise: pd.DataFrame,
    command: list[str],
    wall_seconds: float,
) -> None:
    lay_summary = (
        "S15 packaged the full E01 replication baseline for reuse. It wrote the replication report, core results table, "
        "trace schema, provenance handoff, regression-test artifacts, and report-bundle handoff while carrying forward all S13/S14 non-replications and caveats."
    )
    nonrep = status[status["claim_id"].isin(NON_REPLICATION_CLAIMS)][
        ["claim_id", "classification", "divergence_cause"]
    ]
    report = f"""# E01 S15 Research Step Full Results

{top_summary_md(validation, artifact_paths, lay_summary)}

## Frozen Question

Can the full replication be packaged into a reusable baseline that later Experiments can trust as a regression standard?

## Inputs

{chr(10).join(f"- `{label}`: `{path}` (`{sha256_file(path)}`)" for label, path in input_paths.items() if path.exists())}

## Detailed Methods

S15 performed evidence synthesis and packaging only. It did not rerun simulations, change frozen assumptions, start E02, or generate the final report bundle. The packager loaded S04-S14 metrics and manifests, built a long-form core results table, inventoried trace schemas from Parquet metadata, wrote report-handoff files, generated artifact/provenance manifests, updated checksums and the global run manifest, and ran standard-library regression tests against deterministic fixtures and packaged artifacts.

The package explicitly preserves the S13/S14 non-replications and every documented caveat from the completed E01 steps.

## Commands

```bash
{" ".join(command)}
```

Validation command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/e01_s15_package_baseline.py tests/e01_regression/test_e01_baseline_regression.py
PYTHONDONTWRITEBYTECODE=1 ARTIFACTS_DIR=/artifacts python -m unittest discover -s tests/e01_regression -p 'test_*.py'
```

## Dependencies And Parameters

- Python executable: `{sys.executable}`
- Python version: `{sys.version.splitlines()[0]}`
- Platform: `{platform.platform()}`
- Pandas version: `{pd.__version__}`
- NumPy version: `{np.__version__}`
- PyArrow version: `{pa.__version__}`
- Repository commit at run time: `{validation["repositoryCommitAtRunTime"]}`
- Repository status during report generation: `{validation["repositoryStatusDuringReportGeneration"]}`
- Wall time: `{wall_seconds:.3f}` seconds

No new Python, system, R, Rust, or Node dependencies were installed for S15.

## Results

- Core results rows: `{len(core_results)}`.
- Trace schema artifacts: `{len(trace_schema["traceArtifacts"])}`.
- Regression test success: `{regression_summary["success"]}`.
- S13/S14 non-replications preserved: `{", ".join(sorted(NON_REPLICATION_CLAIMS))}`.

Non-replicated claims:

{dataframe_to_markdown(nonrep)}

S14 scaled selected divergence rows:

{dataframe_to_markdown(noise[["condition_key", "claim_family", "scaled_effect", "scaled_ci95_low", "scaled_ci95_high", "paper_reference_effect", "paper_inside_scaled_ci95", "stability_result"]])}

## Validation

```json
{json.dumps(validation, indent=2)}
```

Regression test summary:

```json
{json.dumps(regression_summary, indent=2)}
```

## Artifacts And Provenance

{chr(10).join(f"- `{label}`: `{path}`" for label, path in artifact_paths.items())}

The global run manifest and checksum file were updated after artifact creation. The S15 artifact manifest records paths, sizes, and SHA256 hashes for package outputs.

## Caveats, Blockers, Failed Assumptions, And Limitations

{chr(10).join(f"- {item}" for item in CAVEAT_THEMES)}

## Recommended Next Action

{validation["recommendedNextAction"]}
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    s15_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    reports_dir = artifacts_dir / "reports"
    regression_dir = artifacts_dir / "regression_tests"
    checksums_dir = artifacts_dir / "checksums"
    ensure_dirs(s15_dir, results_dir, tables_dir, reports_dir, regression_dir, checksums_dir)

    input_paths = required_input_paths(artifacts_dir, args.research_plan_path.resolve(), args.paper_markdown_path.resolve())
    missing_inputs = [str(path) for path in input_paths.values() if not path.exists()]
    if missing_inputs:
        raise SystemExit("Missing S15 inputs: " + "; ".join(missing_inputs))

    core_results_path = results_dir / "e01_core_results.parquet"
    core_results_csv_path = tables_dir / "e01_core_results.csv"
    trace_schema_json_path = results_dir / "e01_trace_schema.json"
    trace_schema_report_path = reports_dir / "e01_trace_schema.md"
    provenance_manifest_path = reports_dir / "e01_provenance_manifest.json"
    artifact_index_path = s15_dir / "e01_artifact_index.csv"
    regression_test_results_path = regression_dir / "regression_test_results.json"
    regression_test_manifest_path = regression_dir / "regression_test_manifest.json"
    regression_test_log_path = regression_dir / "e01_regression_unittest.log"
    replication_report_path = artifacts_dir / "replication_report.md"
    report_bundle_handoff_path = reports_dir / "e01_report_bundle_handoff.md"
    validation_path = s15_dir / "s15_validation.json"
    manifest_path = s15_dir / "artifact_manifest.json"
    full_results_path = s15_dir / "research_step_full_results.md"

    output_paths = {
        "full_results_report": full_results_path,
        "replication_report": replication_report_path,
        "core_results_parquet": core_results_path,
        "core_results_csv": core_results_csv_path,
        "trace_schema_json": trace_schema_json_path,
        "trace_schema_report": trace_schema_report_path,
        "provenance_manifest": provenance_manifest_path,
        "artifact_index_csv": artifact_index_path,
        "regression_test_results": regression_test_results_path,
        "regression_test_manifest": regression_test_manifest_path,
        "regression_test_log": regression_test_log_path,
        "report_bundle_handoff": report_bundle_handoff_path,
        "validation_json": validation_path,
        "artifact_manifest": manifest_path,
    }

    status = read_csv(input_paths["replication_status_csv"])
    noise = read_csv(input_paths["noise_estimates"])
    core_results = build_core_results(input_paths)
    trace_schema = build_trace_schema(trace_paths(artifacts_dir))
    core_results.to_parquet(core_results_path, index=False)
    core_results.to_csv(core_results_csv_path, index=False)
    write_json(trace_schema_json_path, trace_schema)

    initial_artifact_index = build_artifact_index(input_paths, {k: v for k, v in output_paths.items() if v.exists()})
    provenance = {
        "researchStepId": STEP_ID,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": utc_now(),
        "repositoryCommitAtRunTime": git_commit(repo_dir),
        "repositoryStatus": git_status(repo_dir) or "clean",
        "python": sys.version,
        "platform": platform.platform(),
        "inputArtifactCount": int(len(input_paths)),
        "outputArtifactCount": int(len(output_paths)),
        "caveatThemes": CAVEAT_THEMES,
        "nonReplicationClaims": sorted(NON_REPLICATION_CLAIMS),
        "artifactIndexPreviewRows": initial_artifact_index.head(20).to_dict(orient="records"),
    }
    write_json(provenance_manifest_path, provenance)

    regression_summary = run_regression_tests(repo_dir, artifacts_dir, regression_dir)

    pre_validation_paths = {k: v for k, v in output_paths.items() if k not in {"validation_json", "artifact_manifest", "full_results_report", "replication_report", "trace_schema_report", "report_bundle_handoff"}}
    artifact_index = build_artifact_index(input_paths, pre_validation_paths)
    artifact_index.to_csv(artifact_index_path, index=False)

    artifact_paths_for_validation = {
        key: value
        for key, value in output_paths.items()
        if key not in {"validation_json", "artifact_manifest", "full_results_report", "replication_report", "trace_schema_report", "report_bundle_handoff"}
    }
    update_checksum_file(checksums_dir / "sha256sums.txt", [*input_paths.values(), *artifact_paths_for_validation.values()])
    artifact_checks = artifact_validation(input_paths, artifact_paths_for_validation, checksums_dir / "sha256sums.txt")
    validation = validate_package(core_results, trace_schema, regression_summary, artifact_checks, status, noise)
    validation.update(
        {
            "createdAtUtc": utc_now(),
            "repositoryCommitAtRunTime": git_commit(repo_dir),
            "repositoryStatusDuringReportGeneration": git_status(repo_dir) or "clean",
            "script": str(repo_dir / "scripts/e01_s15_package_baseline.py"),
            "artifactsWritten": [str(path) for path in output_paths.values()],
            "inputArtifactCount": int(len(input_paths)),
            "outputArtifactCount": int(len(output_paths)),
            "wallSeconds": time.monotonic() - started,
            "threadEnvironment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
        }
    )

    write_replication_report(replication_report_path, validation, output_paths, status, noise, core_results)
    write_trace_schema_report(trace_schema_report_path, validation, output_paths, trace_schema)
    artifact_index = build_artifact_index(input_paths, output_paths)
    artifact_index.to_csv(artifact_index_path, index=False)
    write_report_bundle_handoff(report_bundle_handoff_path, validation, output_paths, status, artifact_index)
    write_json(validation_path, validation)
    write_artifact_manifest(manifest_path, output_paths, validation, repo_dir)
    write_full_results_report(
        full_results_path,
        validation,
        output_paths,
        input_paths,
        core_results,
        trace_schema,
        regression_summary,
        status,
        noise,
        sys.argv,
        validation["wallSeconds"],
    )
    update_run_manifest(artifacts_dir, output_paths, validation, repo_dir, sys.argv)
    update_checksum_file(checksums_dir / "sha256sums.txt", [args.research_plan_path.resolve(), *input_paths.values(), *output_paths.values(), artifacts_dir / "run_manifest.json"])
    final_checks = artifact_validation(input_paths, output_paths, checksums_dir / "sha256sums.txt")
    if not (final_checks["referencedArtifactsExist"] and final_checks["referencedArtifactsNonEmpty"] and final_checks["checksumsValidated"]):
        validation["success"] = False
        validation["status"] = "failed_validation"
        validation["validationResult"] = "failed_final_checksum_validation"
        validation["artifactValidation"] = final_checks
        write_json(validation_path, validation)
        raise SystemExit(json.dumps(validation, indent=2, sort_keys=True))

    print(json.dumps(validation, indent=2, sort_keys=True))
    if not validation["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
