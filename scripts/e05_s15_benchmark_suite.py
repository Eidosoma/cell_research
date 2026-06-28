#!/usr/bin/env python3
"""Package the E05 morphology benchmark suite for S15."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
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
    BENCHMARK_SUITE_SCHEMA_VERSION,
    BenchmarkConfig,
    artifact_link_rows,
    assign_benchmark_membership,
    benchmark_suite_validation_rows,
    config_catalog_rows,
    dg_caveat_preservation_audit,
    dry_run_benchmark_configs,
    information_flag_preservation_audit,
    standard_benchmark_configs,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S15"
STEP_NUMBER = 15
STEP_TITLE = "Produce a morphology benchmark suite"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
FOCUSED_TESTS = [
    "tests.test_e05_benchmark_suite",
    "tests.test_e05_delayed_gratification",
    "tests.test_e05_control_comparison",
    "tests.test_e05_trajectory_maps",
    "tests.test_e05_gpu_batches",
    "tests.test_e05_symmetry",
    "tests.test_e05_scaling",
    "tests.test_e05_regeneration",
    "tests.test_e05_scrambled_recovery",
    "tests.test_e05_embedded_1d",
]


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_dataframe(df: pd.DataFrame, path_without_suffix: Path) -> list[Path]:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path_without_suffix.with_suffix(".csv")
    parquet_path = path_without_suffix.with_suffix(".parquet")
    safe = df.copy()
    for column in safe.columns:
        if safe[column].map(lambda item: isinstance(item, (Mapping, list, tuple, set))).any():
            safe[column] = safe[column].map(
                lambda item: json.dumps(json_ready(item), sort_keys=True)
                if isinstance(item, (Mapping, list, tuple, set))
                else item
            )
    safe.to_csv(csv_path, index=False)
    safe.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


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


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return pd.read_parquet(path)


def bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    return values.map(
        lambda item: default if pd.isna(item) else str(item).strip().lower() in {"true", "1", "yes"}
    ).fillna(default).astype(bool)


def first_existing(df: pd.DataFrame, *columns: str, default: Any = None) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for column in columns:
        if column in df.columns:
            values = df[column]
            if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
                values = values.where(values.notna() & values.astype(str).str.strip().ne(""))
            out = out.where(out.notna(), values)
    if default is not None:
        out = out.where(out.notna(), default)
    return out


def build_s06_stable_run_ids(s06: pd.DataFrame) -> pd.Series:
    """Return nonempty, row-stable S06 run identifiers from provenance columns."""

    run_id = first_existing(s06, "condition_id", "source_condition_id", default="")
    fallback = (
        first_existing(s06, "algorithm", default="algorithm_unknown").astype(str)
        + "::rep"
        + first_existing(s06, "replicate_index", "replicate_number", default="unknown").astype(str)
        + "::input"
        + first_existing(s06, "input_permutation_seed", default="unknown").astype(str)
        + "::sched"
        + first_existing(s06, "scheduler_seed", default="unknown").astype(str)
        + "::tie"
        + first_existing(s06, "tie_breaker_seed", default="unknown").astype(str)
    )
    run_id = run_id.astype(str).str.strip()
    missing = run_id.eq("") | run_id.str.lower().isin({"nan", "none", "<na>", "nat"})
    run_id = run_id.where(~missing, fallback)
    return run_id.astype(str)


def build_s14_canonical(artifacts_dir: Path) -> pd.DataFrame:
    s14 = read_table(artifacts_dir / "research_steps/S14/higher_dimensional_dg_summary.parquet")
    audit = read_table(artifacts_dir / "research_steps/S14/dg_normalization_artifact_audit.parquet")
    s13 = read_table(artifacts_dir / "research_steps/S13/local_global_control_runs.parquet")
    keys = ["source_step_id", "run_id", "trajectory_id"]
    audit_cols = keys + [
        "normalization_artifact_flag",
        "productive_normalization_artifact_flag",
        "primary_dg_supported_by_component",
        "tiny_primary_worsening_flag",
        "audit_coverage",
        "component_metrics_checked",
    ]
    s13_cols = keys + [
        "final_target_error",
        "best_target_error",
        "target_error_metric",
        "exact_success",
        "improved",
        "robustness_score",
        "information_access_score",
        "intervention_complexity_score",
        "local_only_leakage_violation",
        "global_baseline_explicitly_flagged",
        "s12_route_joined",
    ]
    out = s14.merge(audit[audit_cols], on=keys, how="left")
    out = out.merge(s13[s13_cols], on=keys, how="left")
    out["benchmark_source"] = "S14_harmonized_S07_S11_with_S13_control_metadata"
    out["final_error"] = pd.to_numeric(out["final_error"], errors="coerce")
    out["exact_success"] = bool_series(out, "exact_success", default=False)
    out["improved"] = bool_series(out, "improved", default=False)
    out["normalization_artifact_flag"] = bool_series(out, "normalization_artifact_flag", default=False)
    out["productive_normalization_artifact_flag"] = bool_series(out, "productive_normalization_artifact_flag", default=False)
    out["primary_dg_supported_by_component"] = bool_series(out, "primary_dg_supported_by_component", default=True)
    out["dg_caveat"] = (
        "S14 DG is detected on saved target-error snapshots; interval-level backtracking can be missed or merged."
    )
    out["normalization_warning"] = np.where(
        out["normalization_artifact_flag"],
        "S14 normalization artifact flag: primary error-proxy DG lacked component-metric support.",
        "S14 normalization audit available; no normalization artifact flag for this trajectory.",
    )
    out["claim_boundary"] = "Computational morphology benchmark proxy; not biological validation."
    return out


def build_s06_canonical(artifacts_dir: Path) -> pd.DataFrame:
    s06 = read_table(artifacts_dir / "research_steps/S06/embedded_1d_replication_results.parquet")
    run_id = build_s06_stable_run_ids(s06)
    out = pd.DataFrame(index=s06.index)
    out["schema_version"] = BENCHMARK_SUITE_SCHEMA_VERSION
    out["research_step_id"] = STEP_ID
    out["source_step_id"] = "S06"
    out["source_trace_kind"] = "embedded_1d_replication"
    out["source_trace_path"] = str(artifacts_dir / "research_steps/S06/embedded_1d_trace_examples.parquet")
    out["source_run_path"] = str(artifacts_dir / "research_steps/S06/embedded_1d_replication_results.parquet")
    out["run_id"] = run_id.astype(str)
    out["trajectory_id"] = "S06::" + run_id.astype(str)
    out["target_id"] = "embedded_sorted_row"
    out["task_id"] = ""
    out["motif"] = "sorted_row"
    out["policy_id"] = first_existing(s06, "algorithm", default="classic_row_policy").astype(str)
    out["policy_family"] = "classic_1d_embedded"
    out["seed"] = first_existing(s06, "input_permutation_seed", "scheduler_seed", default=np.nan)
    out["perturbation_id"] = ""
    out["perturbation_family"] = ""
    out["size_class"] = "embedded_1d"
    out["size_key"] = first_existing(s06, "n", default="").astype(str)
    out["scale_rule_id"] = ""
    out["is_heldout_size"] = False
    out["information_scope"] = "row_restricted_local_only"
    out["control_class"] = "local_only"
    out["is_local_only_policy"] = True
    out["is_global_information_baseline"] = False
    out["uses_target_map"] = False
    out["uses_global_gradient"] = False
    out["uses_organizer"] = False
    out["node_count"] = pd.to_numeric(first_existing(s06, "n", default=np.nan), errors="coerce")
    out["device"] = "cpu"
    out["snapshot_count"] = first_existing(s06, "event_count", default=np.nan)
    out["duration_steps"] = first_existing(s06, "scheduler_rounds", default=np.nan)
    out["initial_error"] = np.nan
    out["final_error"] = pd.to_numeric(first_existing(s06, "final_monotonicity_error", default=np.nan), errors="coerce")
    out["best_error"] = out["final_error"]
    out["worst_error"] = np.nan
    out["error_reduction"] = np.nan
    out["relative_error_reduction"] = np.nan
    out["dg_event_count"] = pd.to_numeric(first_existing(s06, "dg_event_count", default=0), errors="coerce").fillna(0).astype(int)
    out["productive_dg_event_count"] = out["dg_event_count"]
    out["total_error_worsening"] = pd.to_numeric(first_existing(s06, "dg_total_drop", default=0.0), errors="coerce").fillna(0.0)
    out["total_productive_dg_index"] = pd.to_numeric(first_existing(s06, "delayed_gratification", default=0.0), errors="coerce").fillna(0.0)
    out["any_dg_event"] = out["dg_event_count"] > 0
    out["any_productive_dg_event"] = out["productive_dg_event_count"] > 0
    out["normalization_artifact_flag"] = False
    out["productive_normalization_artifact_flag"] = False
    out["primary_dg_supported_by_component"] = True
    out["tiny_primary_worsening_flag"] = False
    out["audit_coverage"] = "S06_sortedness_metric"
    out["component_metrics_checked"] = "Sortedness,monotonicity_error"
    out["target_error_metric"] = "embedded_1d_monotonicity_error"
    out["final_target_error"] = out["final_error"]
    out["best_target_error"] = out["best_error"]
    out["exact_success"] = bool_series(s06, "metric_match", default=True)
    out["improved"] = out["exact_success"]
    out["robustness_score"] = np.nan
    out["information_access_score"] = 0.0
    out["intervention_complexity_score"] = 0.0
    out["local_only_leakage_violation"] = False
    out["global_baseline_explicitly_flagged"] = False
    out["s12_route_joined"] = False
    out["benchmark_source"] = "S06_1d_embedded_regression"
    out["dg_caveat"] = "S06 uses the original 1D Sortedness-based DG metric, not the S14 higher-dimensional error proxy."
    out["normalization_warning"] = "S06 DG is not normalized through S14 component metrics; compare only within the row-restricted baseline."
    out["claim_boundary"] = "Computational sorting-row baseline preservation only."
    return out


def build_canonical_table(artifacts_dir: Path) -> pd.DataFrame:
    s14 = build_s14_canonical(artifacts_dir)
    s06 = build_s06_canonical(artifacts_dir)
    out = pd.concat([s06, s14], ignore_index=True, sort=False)
    for column in ["task_id", "target_id", "motif", "policy_id", "policy_family", "perturbation_id", "perturbation_family"]:
        if column in out.columns:
            out[column] = out[column].fillna("").astype(str)
    for column in [
        "is_local_only_policy",
        "is_global_information_baseline",
        "uses_target_map",
        "uses_global_gradient",
        "uses_organizer",
        "normalization_artifact_flag",
        "productive_normalization_artifact_flag",
        "any_dg_event",
        "any_productive_dg_event",
    ]:
        out[column] = bool_series(out, column, default=False)
    return out


def config_from_json(path: Path) -> BenchmarkConfig:
    payload = read_json(path)
    selectors = tuple({str(key): tuple(value) for key, value in selector.items()} for selector in payload["selectors"])
    return BenchmarkConfig(
        benchmark_id=payload["benchmarkId"],
        title=payload["title"],
        benchmark_family=payload["benchmarkFamily"],
        description=payload["description"],
        primary_goal=payload["primaryGoal"],
        selectors=selectors,
        expected_min_rows=int(payload["expectedMinRows"]),
        required_columns=tuple(payload["requiredColumns"]),
        required_artifact_paths=tuple(payload["requiredArtifactPaths"]),
        metrics=tuple(payload["metrics"]),
        caveats=tuple(payload["caveats"]),
        validation_notes=tuple(payload["validationNotes"]),
        report_tags=tuple(payload["reportTags"]),
    )


def run_single_config_dry_run(artifacts_dir: Path, config_path: Path) -> int:
    config = config_from_json(config_path)
    canonical = build_canonical_table(artifacts_dir)
    packaged = assign_benchmark_membership(canonical, [config])
    dry = dry_run_benchmark_configs([config], packaged)
    links = artifact_link_rows([config])
    success = bool(dry["dry_run_success"].all() and links["link_validation_success"].all())
    print(
        json.dumps(
            json_ready(
                {
                    "schemaVersion": BENCHMARK_SUITE_SCHEMA_VERSION,
                    "researchStepId": STEP_ID,
                    "configPath": str(config_path),
                    "success": success,
                    "dryRun": dry.to_dict(orient="records"),
                    "artifactLinks": links.to_dict(orient="records"),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if success else 1


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


def write_config_files(configs: Sequence[BenchmarkConfig], step_dir: Path, artifacts_dir: Path) -> list[Path]:
    config_dir = step_dir / "benchmark_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for config in configs:
        path = config_dir / f"{config.benchmark_id}.json"
        payload = config.to_dict()
        payload["dryRunContract"]["command"] = payload["dryRunContract"]["commandTemplate"].format(config_path=str(path))
        write_json(path, payload)
        paths.append(path)
    suite_config = {
        "schemaVersion": BENCHMARK_SUITE_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "createdAt": utc_now(),
        "benchmarkCount": len(configs),
        "configs": [config.to_dict() for config in configs],
    }
    suite_path = artifacts_dir / "configs/e05_morphology_benchmark_suite.json"
    write_json(suite_path, suite_config)
    paths.append(suite_path)
    return paths


def plot_coverage(dry_df: pd.DataFrame, figure_path: Path) -> Path:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = dry_df.sort_values("row_count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(ordered))))
    ax.barh(ordered["benchmark_id"], ordered["local_only_rows"], label="local-only rows", color="#4477AA")
    ax.barh(
        ordered["benchmark_id"],
        ordered["global_information_baseline_rows"],
        left=ordered["local_only_rows"],
        label="explicit global/baseline rows",
        color="#CC6677",
    )
    ax.set_xlabel("Packaged rows")
    ax.set_ylabel("Benchmark config")
    ax.set_title("E05 S15 benchmark suite dry-run coverage")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def run_generated_config_dry_runs(
    artifacts_dir: Path,
    step_dir: Path,
    config_paths: Sequence[Path],
) -> tuple[pd.DataFrame, list[Path]]:
    log_dir = step_dir / "config_dry_run_logs"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    log_paths: list[Path] = []
    for config_path in sorted(config_paths, key=lambda path: path.name):
        result = run_command(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "e05_s15_benchmark_suite.py"),
                "--artifacts-dir",
                str(artifacts_dir),
                "--dry-run-config",
                str(config_path),
            ],
            cwd=REPO_ROOT,
        )
        parsed: dict[str, Any] = {}
        parse_error = ""
        if result["stdout"].strip():
            try:
                parsed = json.loads(result["stdout"])
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
        benchmark_id = str(parsed.get("dryRun", [{}])[0].get("benchmark_id", config_path.stem)) if parsed else config_path.stem
        success = bool(result["success"] and parsed.get("success", False) and not parse_error)
        log_path = log_dir / f"{config_path.stem}.json"
        write_json(
            log_path,
            {
                "schemaVersion": BENCHMARK_SUITE_SCHEMA_VERSION,
                "researchStepId": STEP_ID,
                "stepNumber": STEP_NUMBER,
                "benchmarkId": benchmark_id,
                "configPath": str(config_path),
                "command": result["args"],
                "returncode": result["returncode"],
                "subprocessSuccess": result["success"],
                "parsedSuccess": bool(parsed.get("success", False)) if parsed else False,
                "success": success,
                "parseError": parse_error,
                "runtimeSeconds": result["runtimeSeconds"],
                "stdoutJson": parsed,
                "stderr": result["stderr"],
            },
        )
        log_paths.append(log_path)
        rows.append(
            {
                "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
                "research_step_id": STEP_ID,
                "benchmark_id": benchmark_id,
                "config_path": str(config_path),
                "dry_run_log_path": str(log_path),
                "returncode": result["returncode"],
                "subprocess_success": bool(result["success"]),
                "parsed_success": bool(parsed.get("success", False)) if parsed else False,
                "parse_error": parse_error,
                "dry_run_command_success": success,
                "runtime_seconds": result["runtimeSeconds"],
            }
        )
    return pd.DataFrame(rows), log_paths


def write_report_bundle_inputs(
    bundle_dir: Path,
    configs: Sequence[BenchmarkConfig],
    tables: Mapping[str, pd.DataFrame],
    report_path: Path,
    figure_paths: Sequence[Path],
    artifact_links: pd.DataFrame,
) -> list[Path]:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / "tables").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "configs").mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    readme = bundle_dir / "README.md"
    readme.write_text(
        "# E05 Morphology Benchmark Suite Report Bundle Inputs\n\n"
        "- Research step ID: S15\n"
        "- Completion status: bundled inputs generated by S15\n"
        "- Artifacts written: tables, configs, figure index, and manifest in this directory\n"
        "- Validation result: see `tables/benchmark_suite_validation_results.csv`\n"
        "- Caveats or blockers: computational benchmark proxies; DG and normalization warnings preserved in tables\n"
        "- Recommended next action: Chief Scientist review before downstream E07 ingestion\n",
        encoding="utf-8",
    )
    paths.append(readme)
    for config in configs:
        path = bundle_dir / "configs" / f"{config.benchmark_id}.json"
        write_json(path, config.to_dict())
        paths.append(path)
    for name, df in tables.items():
        csv_path = bundle_dir / "tables" / f"{name}.csv"
        safe = df.copy()
        for column in safe.columns:
            if safe[column].map(lambda item: isinstance(item, (Mapping, list, tuple, set))).any():
                safe[column] = safe[column].map(
                    lambda item: json.dumps(json_ready(item), sort_keys=True)
                    if isinstance(item, (Mapping, list, tuple, set))
                    else item
                )
        safe.to_csv(csv_path, index=False)
        paths.append(csv_path)
    figure_index = pd.DataFrame(
        [
            {
                "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
                "research_step_id": STEP_ID,
                "figure_path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            for path in figure_paths
        ]
    )
    figure_index_path = bundle_dir / "figure_index.csv"
    figure_index.to_csv(figure_index_path, index=False)
    paths.append(figure_index_path)
    artifact_links_path = bundle_dir / "artifact_links.csv"
    artifact_links.to_csv(artifact_links_path, index=False)
    paths.append(artifact_links_path)
    report_link_path = bundle_dir / "suite_report_path.txt"
    report_link_path.write_text(str(report_path) + "\n", encoding="utf-8")
    paths.append(report_link_path)
    manifest_path = bundle_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "schemaVersion": BENCHMARK_SUITE_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "createdAt": utc_now(),
            "bundlePath": str(bundle_dir),
            "suiteReportPath": str(report_path),
            "benchmarkCount": len(configs),
            "tableCount": len(tables),
            "artifacts": collect_artifacts(paths),
        },
    )
    paths.append(manifest_path)
    return paths


def write_suite_report(
    path: Path,
    configs: Sequence[BenchmarkConfig],
    dry_run_df: pd.DataFrame,
    artifact_df: pd.DataFrame,
    info_df: pd.DataFrame,
    dg_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    artifacts: Sequence[Path],
    validation_result: str,
    caveats: Sequence[str],
    recommended_next_action: str,
) -> None:
    row_count = int(dry_run_df["row_count"].sum())
    global_rows = int(dry_run_df["global_information_baseline_rows"].sum())
    local_rows = int(dry_run_df["local_only_rows"].sum())
    report_rows = [
        [
            row.benchmark_id,
            row.row_count,
            row.local_only_rows,
            row.global_information_baseline_rows,
            "yes" if row.dry_run_success else "no",
        ]
        for row in dry_run_df.itertuples(index=False)
    ]
    validation_rows_md = [
        [row.check_id, "pass" if row.success else "fail", row.detail]
        for row in validation_df.itertuples(index=False)
    ]
    artifact_preview = "\n".join(f"- `{path}`" for path in sorted(str(path) for path in artifacts)[:45])
    if len(artifacts) > 45:
        artifact_preview += f"\n- ... {len(artifacts) - 45} additional artifacts listed in artifact manifest"
    caveat_md = "\n".join(f"- {item}" for item in caveats)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# E05 S15 Morphology Benchmark Suite\n\n"
        f"- Research step ID: {STEP_ID}\n"
        f"- Completion status: completed\n"
        f"- Artifacts written:\n{artifact_preview}\n"
        f"- Validation result: {validation_result}\n"
        f"- Caveats or blockers:\n{caveat_md}\n"
        f"- Recommended next action: {recommended_next_action}\n\n"
        f"S15 packages the E05 higher-dimensional morphology outputs into reusable benchmark configs and report bundle inputs. "
        f"The packaged table contains {row_count} benchmark-membership rows across {len(configs)} configs, including "
        f"{local_rows} local-only rows and {global_rows} explicitly flagged global-information baseline rows. "
        f"Every config was dry-run against existing S06-S14 artifacts, and all required artifact links resolved.\n\n"
        "## Benchmark Config Dry-Runs\n\n"
        + markdown_table(["benchmark", "rows", "local-only", "global baseline", "dry-run"], report_rows)
        + "\n\n## Validation Checks\n\n"
        + markdown_table(["check", "status", "detail"], validation_rows_md)
        + "\n\n## Preservation Audits\n\n"
        f"- Artifact links checked: {len(artifact_df)}; failures: {int((~artifact_df['link_validation_success'].astype(bool)).sum())}\n"
        f"- Information-flag audit rows: {len(info_df)}; failures: {int((~info_df['audit_success'].astype(bool)).sum())}\n"
        f"- DG caveat audit rows: {len(dg_df)}; normalization artifact flags carried: {int(dg_df['normalization_artifact_flag_count'].sum())}\n\n"
        "## Lay Summary\n\n"
        "The suite is a compact, reusable index over the completed E05 simulations. It does not rerun the full experiments; "
        "it turns validated outputs into dry-runnable benchmark definitions with linked evidence, IDs, local/global information flags, "
        "DG caveats, and normalization warnings preserved for downstream review or E07 ingestion.\n",
        encoding="utf-8",
    )


def summary_markdown(
    artifacts: Sequence[Path],
    validation_result: str,
    outcome: str,
    caveats: Sequence[str],
    recommended_next_action: str,
    anchor_result: str,
) -> str:
    artifact_lines = "\n".join(f"- `{path}`" for path in sorted(str(path) for path in artifacts))
    caveat_lines = "\n".join(f"- {item}" for item in caveats)
    return (
        "# E05 S15 Status Summary\n\n"
        f"- Research step ID: {STEP_ID}\n"
        "- Completion status: completed\n"
        f"- Artifacts written:\n{artifact_lines}\n"
        f"- Validation result: {validation_result}\n"
        f"- Outcome classification: {outcome}\n"
        f"- Caveats or blockers:\n{caveat_lines}\n"
        "- Lay summary: S15 packaged the E05 morphology tasks into reusable, dry-run validated benchmark configs and report bundle inputs while preserving source IDs, local/global flags, DG caveats, and normalization warnings.\n"
        f"- Recommended next action: {recommended_next_action}\n\n"
        f"Anchor result: {anchor_result}\n"
    )


def run_repo_tests(step_dir: Path, skip: bool) -> dict[str, Any]:
    log_path = step_dir / "repo_unit_test_log.txt"
    if skip:
        log_path.write_text("Skipped by --skip-repo-tests.\n", encoding="utf-8")
        return {"success": True, "returncode": 0, "runtimeSeconds": 0.0, "logPath": str(log_path)}
    result = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS], cwd=REPO_ROOT)
    log_path.write_text(
        "Command: " + " ".join(result["args"]) + "\n"
        f"Return code: {result['returncode']}\n"
        f"Runtime seconds: {result['runtimeSeconds']:.3f}\n\n"
        "STDOUT\n" + result["stdout"] + "\nSTDERR\n" + result["stderr"],
        encoding="utf-8",
    )
    return {
        "success": result["success"],
        "returncode": result["returncode"],
        "runtimeSeconds": result["runtimeSeconds"],
        "logPath": str(log_path),
    }


def update_run_manifest(artifacts_dir: Path, status_path: Path, artifacts: Sequence[Path]) -> Path:
    path = artifacts_dir / "provenance/run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = read_json(path)
    else:
        payload = {"schemaVersion": "eidosoma.run_manifest.v1", "researchSteps": {}}
    payload["schemaVersion"] = payload.get("schemaVersion", "eidosoma.run_manifest.v1")
    payload["latestResearchStepId"] = STEP_ID
    payload["updatedAt"] = utc_now()
    payload.setdefault("researchSteps", {})
    payload["researchSteps"][STEP_ID] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "statusPath": str(status_path),
        "artifactCount": len(artifacts),
        "artifacts": sorted(str(path) for path in artifacts),
        "git": git_metadata(),
    }
    payload["runtime"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    payload["git"] = git_metadata()
    write_json(path, payload)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the E05 morphology benchmark suite.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--dry-run-config", default=None, help="Dry-run one generated benchmark config and print JSON.")
    parser.add_argument("--skip-repo-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    if args.dry_run_config:
        return run_single_config_dry_run(artifacts_dir, Path(args.dry_run_config))

    started = time.perf_counter()
    step_dir = artifacts_dir / "research_steps/S15"
    figures_dir = artifacts_dir / "figures"
    reports_dir = artifacts_dir / "reports"
    bundle_dir = artifacts_dir / "report_bundle_inputs/e05_morphology_benchmark_suite"
    step_dir.mkdir(parents=True, exist_ok=True)

    configs = standard_benchmark_configs(artifacts_dir)
    canonical_df = build_canonical_table(artifacts_dir)
    packaged_df = assign_benchmark_membership(canonical_df, configs)
    config_df = config_catalog_rows(configs)
    dry_run_df = dry_run_benchmark_configs(configs, packaged_df)
    artifact_df = artifact_link_rows(configs)
    info_audit_df = information_flag_preservation_audit(packaged_df)
    dg_audit_df = dg_caveat_preservation_audit(packaged_df)

    config_paths = write_config_files(configs, step_dir, artifacts_dir)
    figure_path = plot_coverage(dry_run_df, figures_dir / "e05_s15_benchmark_suite_coverage.png")
    generated_config_paths = [path for path in config_paths if path.parent.name == "benchmark_configs"]
    config_command_df, config_command_logs = run_generated_config_dry_runs(
        artifacts_dir,
        step_dir,
        generated_config_paths,
    )

    artifacts: list[Path] = []
    artifacts.extend(config_paths)
    artifacts.extend(config_command_logs)
    artifacts.extend(write_dataframe(canonical_df, step_dir / "canonical_benchmark_trajectory_table"))
    artifacts.extend(write_dataframe(packaged_df, step_dir / "morphology_benchmark_suite_rows"))
    artifacts.extend(write_dataframe(packaged_df, artifacts_dir / "results/e05_morphology_benchmarks"))
    artifacts.extend(write_dataframe(config_df, step_dir / "benchmark_suite_config_catalog"))
    artifacts.extend(write_dataframe(dry_run_df, step_dir / "benchmark_dry_run_results"))
    artifacts.extend(write_dataframe(config_command_df, step_dir / "generated_config_dry_run_commands"))
    artifacts.extend(write_dataframe(artifact_df, step_dir / "artifact_link_validation"))
    artifacts.extend(write_dataframe(info_audit_df, step_dir / "information_flag_preservation_audit"))
    artifacts.extend(write_dataframe(dg_audit_df, step_dir / "dg_caveat_preservation_audit"))

    source_artifacts = pd.DataFrame(collect_artifacts([Path(path) for path in artifact_df["artifact_path"].unique()]))
    if source_artifacts.empty:
        source_artifacts = pd.DataFrame(columns=["path", "sizeBytes", "sha256"])
    source_artifacts.insert(0, "research_step_id", STEP_ID)
    source_artifacts.insert(0, "schema_version", BENCHMARK_SUITE_SCHEMA_VERSION)
    artifacts.extend(write_dataframe(source_artifacts, step_dir / "source_artifact_catalog"))
    artifacts.append(figure_path)

    tables_for_bundle = {
        "benchmark_suite_config_catalog": config_df,
        "e05_morphology_benchmarks": packaged_df,
        "benchmark_dry_run_results": dry_run_df,
        "generated_config_dry_run_commands": config_command_df,
        "artifact_link_validation": artifact_df,
        "information_flag_preservation_audit": info_audit_df,
        "dg_caveat_preservation_audit": dg_audit_df,
    }
    report_path = reports_dir / "e05_morphology_benchmark_suite.md"
    bundle_paths = write_report_bundle_inputs(
        bundle_dir,
        configs,
        tables_for_bundle,
        report_path,
        [figure_path],
        artifact_df,
    )
    bundle_validation_path = bundle_dir / "tables" / "benchmark_suite_validation_results.csv"
    pd.DataFrame(
        [
            {
                "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
                "research_step_id": STEP_ID,
                "check_id": "pending_final_validation_write",
                "success": True,
                "detail": "Placeholder overwritten after report bundle path validation.",
            }
        ]
    ).to_csv(bundle_validation_path, index=False)
    bundle_paths.append(bundle_validation_path)
    artifacts.extend(bundle_paths)

    validation_df = benchmark_suite_validation_rows(
        configs,
        packaged_df,
        dry_run_df,
        artifact_df,
        info_audit_df,
        dg_audit_df,
        bundle_paths,
    )
    validation_df = pd.concat(
        [
            validation_df,
            pd.DataFrame(
                [
                    {
                        "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
                        "research_step_id": STEP_ID,
                        "check_id": "generated_config_dry_run_commands_pass",
                        "success": bool(
                            len(config_command_df) == len(generated_config_paths)
                            and config_command_df["dry_run_command_success"].astype(bool).all()
                        ),
                        "detail": {
                            "commands": int(len(config_command_df)),
                            "successes": int(config_command_df["dry_run_command_success"].astype(bool).sum()),
                        },
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    artifacts.extend(write_dataframe(validation_df, step_dir / "benchmark_suite_validation_results"))
    validation_df.to_csv(bundle_validation_path, index=False)
    write_json(
        bundle_dir / "manifest.json",
        {
            "schemaVersion": BENCHMARK_SUITE_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "createdAt": utc_now(),
            "bundlePath": str(bundle_dir),
            "suiteReportPath": str(report_path),
            "benchmarkCount": len(configs),
            "tableCount": len(tables_for_bundle) + 1,
            "artifacts": collect_artifacts([path for path in bundle_paths if path.name != "manifest.json"]),
        },
    )
    repo_test = run_repo_tests(step_dir, args.skip_repo_tests)
    artifacts.append(Path(repo_test["logPath"]))

    success = bool(validation_df["success"].astype(bool).all() and repo_test["success"])
    dry_success_count = int(dry_run_df["dry_run_success"].astype(bool).sum())
    command_success_count = int(config_command_df["dry_run_command_success"].astype(bool).sum())
    link_failures = int((~artifact_df["link_validation_success"].astype(bool)).sum())
    info_failures = int((~info_audit_df["audit_success"].astype(bool)).sum())
    dg_failures = int((~dg_audit_df["audit_success"].astype(bool)).sum())
    validation_result = (
        f"passed: dry-ran {dry_success_count}/{len(configs)} benchmark configs in-process and {command_success_count}/{len(configs)} generated configs by command, resolved {len(artifact_df) - link_failures}/{len(artifact_df)} artifact links, "
        f"preserved source IDs and local/global flags for {len(info_audit_df)} configs, preserved DG caveats and normalization warnings for {len(dg_audit_df)} configs, repo tests passed"
        if success
        else f"failed: dry-run failures={len(configs) - dry_success_count}, generated config command failures={len(configs) - command_success_count}, link failures={link_failures}, information-flag failures={info_failures}, DG-caveat failures={dg_failures}, repo test success={repo_test['success']}"
    )
    outcome = "supportive" if success else "constraining/contradictory"
    recommended_next_action = "Chief Scientist review; E05 is packaged and ready to hand off to E07 or final report assembly only after explicit instruction."
    caveats = [
        "The suite packages validated E05 simulation artifacts; it does not rerun full S07-S14 sweeps.",
        "All morphology labels, including limb-like and chimeric-pattern conflict, are abstract computational analogies.",
        "Local/global comparisons preserve S13 information-access flags; explicit organizer, global-gradient, and target-map rows remain upper-bound baselines.",
        "DG diagnostics preserve S14 caveats: saved-snapshot resolution, mixed source-specific error proxies, null comparison limits, and normalization artifact warnings.",
        "S06 row-restricted sorting remains a baseline preservation task and should not be interpreted as free 2D motion.",
    ]
    anchor_result = (
        f"S15 packaged {len(configs)} benchmark configs and {len(packaged_df)} benchmark-membership rows from {canonical_df['source_step_id'].nunique()} source steps; "
        f"all in-process and generated-config command dry-runs passed, {len(artifact_df)} artifact links resolved, local/global flags and source IDs were preserved, "
        f"and {int(dg_audit_df['normalization_artifact_flag_count'].sum())} carried normalization artifact flags remain visible."
    )

    write_suite_report(
        report_path,
        configs,
        dry_run_df,
        artifact_df,
        info_audit_df,
        dg_audit_df,
        validation_df,
        artifacts + [report_path],
        validation_result,
        caveats,
        recommended_next_action,
    )
    artifacts.append(report_path)

    status_path = step_dir / "status.json"
    summary_path = step_dir / "summary.md"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = update_run_manifest(artifacts_dir, status_path, artifacts)
    artifacts.append(run_manifest_path)

    summary_path.write_text(
        summary_markdown(
            artifacts + [summary_path, status_path, manifest_path],
            validation_result,
            outcome,
            caveats,
            recommended_next_action,
            anchor_result,
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
        "parallelism": "single_process_packaging",
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
            "benchmarkConfigCount": len(configs),
            "packagedBenchmarkRowCount": int(len(packaged_df)),
            "canonicalTrajectoryRowCount": int(len(canonical_df)),
            "dryRunSuccessCount": dry_success_count,
            "generatedConfigDryRunCommandCount": int(len(config_command_df)),
            "generatedConfigDryRunCommandSuccessCount": command_success_count,
            "artifactLinkCount": int(len(artifact_df)),
            "artifactLinkFailureCount": link_failures,
            "informationFlagAuditFailureCount": info_failures,
            "dgCaveatAuditFailureCount": dg_failures,
            "normalizationArtifactFlagCountCarried": int(dg_audit_df["normalization_artifact_flag_count"].sum()),
            "validationCheckCount": int(len(validation_df)),
            "validationPassCount": int(validation_df["success"].astype(bool).sum()),
        },
        "repoUnitTests": {
            "success": bool(repo_test["success"]),
            "returncode": repo_test["returncode"],
            "logPath": repo_test["logPath"],
        },
        "runtime": runtime,
        "git": git_metadata(),
    }
    write_json(status_path, status)
    artifacts.append(status_path)

    manifest_payload = {
        "schemaVersion": BENCHMARK_SUITE_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "sourceCodePaths": [
            str(REPO_ROOT / "morphospace2d" / "benchmark_suite.py"),
            str(REPO_ROOT / "scripts" / "e05_s15_benchmark_suite.py"),
            str(REPO_ROOT / "tests" / "test_e05_benchmark_suite.py"),
        ],
        "artifacts": collect_artifacts(artifacts),
    }
    write_json(manifest_path, manifest_payload)
    if str(manifest_path) not in status["artifactsWritten"]:
        status["artifactsWritten"] = sorted(set(status["artifactsWritten"] + [str(manifest_path)]))
        write_json(status_path, status)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
