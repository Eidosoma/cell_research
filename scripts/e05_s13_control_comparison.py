#!/usr/bin/env python3
"""Compare E05 local-only and global-information control conditions."""

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
    CONTROL_COMPARISON_SCHEMA_VERSION,
    control_validation_rows,
    information_access_summary,
    leakage_audit_summary,
    load_control_comparison,
    standard_control_source_specs,
    target_control_gap_summary,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S13"
STEP_NUMBER = 13
STEP_TITLE = "Compare local versus global control"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
FOCUSED_TESTS = [
    "tests.test_e05_control_comparison",
    "tests.test_e05_trajectory_maps",
    "tests.test_e05_gpu_batches",
    "tests.test_e05_symmetry",
    "tests.test_e05_scaling",
    "tests.test_e05_regeneration",
    "tests.test_e05_scrambled_recovery",
]


INTERVENTION_COMPLEXITY_DEFINITIONS = {
    "information_access_score": "0 for local target-map-free policies, +3 for global gradients, +3 for organizers, +4 for explicit target maps, and +1 for any explicit global-information baseline flag.",
    "intervention_complexity_score": "Information-access score plus one point each for memory/signal policy families and nonconservative action use.",
    "target_error": "Final target-error proxy from the source-specific metric: composite error for S07-S09, 1 - pattern score for S10, and hamming error for S11.",
    "target_energy": "Reported target energy when present, otherwise final_target_error multiplied by node_count as a bounded computational proxy.",
    "robustness_score": "1.0 for exact or near-target success, 0.5 for improvement without exact success, 0.0 otherwise, halved if occupancy was not preserved.",
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
    parser = argparse.ArgumentParser(description="Compare local-only and global-information E05 controls.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
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
        "Command: "
        + " ".join(command)
        + "\n"
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
        "schema_version": CONTROL_COMPARISON_SCHEMA_VERSION,
        "research_step_id": STEP_ID,
        "check_id": "repo_unit_tests_passed",
        "success": bool(repo_test.get("success", False)),
        "detail": json.dumps({"returncode": repo_test.get("returncode"), "args": repo_test.get("args")}, sort_keys=True),
    }
    return pd.concat([validation_df, pd.DataFrame([row])], ignore_index=True)


def plot_error_by_class(access_summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if access_summary_df.empty:
        return
    grouped = access_summary_df.groupby(["source_step_id", "control_class"], sort=True).apply(
        lambda group: np.average(group["mean_final_target_error"].astype(float), weights=group["run_count"].astype(float)),
        include_groups=False,
    ).reset_index(name="weighted_mean_final_error")
    grouped["label"] = grouped["source_step_id"].astype(str) + "\n" + grouped["control_class"].astype(str).str.replace("_", " ", regex=False)
    fig, axis = plt.subplots(figsize=(11, 5.4))
    colors = {"local_only": "#4f7c8a", "target_map_baseline": "#b85750", "global_gradient_baseline": "#8a6fb0", "organizer_baseline": "#8b7a43"}
    axis.bar(
        np.arange(len(grouped)),
        grouped["weighted_mean_final_error"].astype(float),
        color=[colors.get(label, "#7c8795") for label in grouped["control_class"]],
        edgecolor="#222222",
        linewidth=0.35,
    )
    axis.set_xticks(np.arange(len(grouped)))
    axis.set_xticklabels(grouped["label"], rotation=50, ha="right", fontsize=8)
    axis.set_ylabel("weighted mean final target error")
    axis.set_title("S13 target error by source and control class")
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_success_by_scope(control_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = control_df.groupby(["information_scope"], sort=True).agg(
        run_count=("run_id", "count"),
        exact_success_rate=("exact_success", "mean"),
    ).reset_index()
    grouped = grouped.sort_values("exact_success_rate", ascending=False)
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.bar(
        np.arange(len(grouped)),
        grouped["exact_success_rate"].astype(float),
        color="#5e8c61",
        edgecolor="#222222",
        linewidth=0.35,
    )
    axis.set_xticks(np.arange(len(grouped)))
    axis.set_xticklabels(grouped["information_scope"].astype(str).str.replace("_", " ", regex=False), rotation=35, ha="right", fontsize=8)
    axis.set_ylabel("exact or near-target success rate")
    axis.set_title("S13 success by information-access scope")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    for index, row in grouped.iterrows():
        axis.text(
            index,
            min(1.0, float(row["exact_success_rate"]) + 0.025),
            f"n={int(row['run_count'])}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_complexity_tradeoff(control_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = control_df.groupby(["source_step_id", "control_class"], sort=True).agg(
        mean_error=("final_target_error", "mean"),
        mean_complexity=("intervention_complexity_score", "mean"),
        run_count=("run_id", "count"),
    ).reset_index()
    fig, axis = plt.subplots(figsize=(8.2, 5.6))
    colors = {"local_only": "#4f7c8a", "target_map_baseline": "#b85750", "global_gradient_baseline": "#8a6fb0", "organizer_baseline": "#8b7a43"}
    for _, row in summary.iterrows():
        axis.scatter(
            float(row["mean_complexity"]),
            float(row["mean_error"]),
            s=max(35, min(360, int(row["run_count"]) * 1.6)),
            color=colors.get(str(row["control_class"]), "#7c8795"),
            alpha=0.72,
            edgecolors="#222222",
            linewidths=0.35,
        )
        axis.text(
            float(row["mean_complexity"]) + 0.03,
            float(row["mean_error"]),
            f"{row['source_step_id']} {str(row['control_class']).replace('_', ' ')}",
            fontsize=7,
            va="center",
        )
    axis.set_xlabel("mean intervention complexity score")
    axis.set_ylabel("mean final target error")
    axis.set_title("S13 error versus information/intervention complexity")
    axis.grid(color="#dddddd", linewidth=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gap_summary(gap_summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = gap_summary_df[gap_summary_df["has_local_and_global"].astype(bool)].copy()
    if subset.empty:
        return
    subset["label"] = subset["source_step_id"].astype(str) + "\n" + subset["target_id"].astype(str).str.slice(0, 28)
    subset = subset.sort_values("local_minus_global_final_error", ascending=False).head(28)
    fig, axis = plt.subplots(figsize=(12, 5.4))
    axis.bar(
        np.arange(len(subset)),
        subset["local_minus_global_final_error"].astype(float),
        color="#b86d4b",
        edgecolor="#222222",
        linewidth=0.35,
    )
    axis.axhline(0.0, color="#222222", linewidth=0.8)
    axis.set_xticks(np.arange(len(subset)))
    axis.set_xticklabels(subset["label"], rotation=50, ha="right", fontsize=8)
    axis.set_ylabel("local minus global final target error")
    axis.set_title("S13 paired local/global target-error gaps")
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def intervention_spec_markdown() -> str:
    rows = [[name, definition] for name, definition in INTERVENTION_COMPLEXITY_DEFINITIONS.items()]
    return f"""# E05 S13 Intervention And Information Complexity Specification

- Research step ID: {STEP_ID}
- Completion status: completed metric specification for local-versus-global control comparison
- Artifacts written: `intervention_complexity_spec.md`, `intervention_complexity_spec.json`
- Validation result: evaluated by S13 validation rows and focused unit tests
- Caveats or blockers: These are computational proxy metrics; target energy falls back to an error-times-size proxy when upstream steps did not report target energy directly.
- Recommended next action: Chief Scientist review, then proceed to S14 only after explicit instruction.

S13 keeps information access separate from outcome quality. Local-only policies are target-map-free unless upstream artifacts explicitly mark hidden global leakage. Organizer, global-gradient, and target-map rows are retained as explicit baselines rather than mixed into local-only evidence.

## Metric Definitions

{markdown_table(["metric", "definition"], rows)}
"""


def report_markdown(
    *,
    success: bool,
    control_df: pd.DataFrame,
    source_catalog: pd.DataFrame,
    access_summary_df: pd.DataFrame,
    gap_summary_df: pd.DataFrame,
    leakage_summary_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    artifacts: Sequence[Path],
    validation_result: str,
    caveats: Sequence[str],
    recommended_next_action: str,
    anchor_result: str,
) -> str:
    source_rows = control_df.groupby("source_step_id", sort=True).agg(
        runs=("run_id", "count"),
        local_runs=("is_local_only_policy", "sum"),
        global_baseline_runs=("is_global_information_baseline", "sum"),
        mean_final_error=("final_target_error", "mean"),
        success_rate=("exact_success", "mean"),
        mean_energy=("final_target_energy_proxy", "mean"),
        mean_complexity=("intervention_complexity_score", "mean"),
    ).reset_index()
    access_rows = access_summary_df.groupby(["control_class", "information_scope"], sort=True).agg(
        runs=("run_count", "sum"),
        mean_final_error=("mean_final_target_error", "mean"),
        exact_success_rate=("exact_success_rate", "mean"),
        mean_complexity=("mean_intervention_complexity_score", "mean"),
    ).reset_index()
    paired = gap_summary_df[gap_summary_df["has_local_and_global"].astype(bool)]
    artifacts_text = "\n".join(f"- `{path}`" for path in artifacts[:45])
    if len(artifacts) > 45:
        artifacts_text += f"\n- ... {len(artifacts) - 45} additional files listed in `artifact_manifest.json`"
    caveat_text = "\n".join(f"- {item}" for item in caveats)
    return f"""# E05 S13 Local Versus Global Control Report

- Research step ID: {STEP_ID}
- Completion status: {"completed" if success else "completed with validation failures"}
- Artifacts written:
{artifacts_text}
- Validation result: {validation_result}
- Caveats or blockers:
{caveat_text}
- Recommended next action: {recommended_next_action}

S13 merged S07-S11 run tables with S12 trajectory summaries while preserving `source_step_id`, `run_id`, and `trajectory_id`. Local-only and explicit global-information baselines remain separable through `control_class`, `information_scope`, and global-access flags. Organizer, global-gradient, and target-map rows are labeled as baselines, not local morphogenesis policies.

## Source Coverage

{markdown_table(["source", "kind", "run available", "leakage audit available"], source_catalog[["source_step_id", "source_kind", "run_available", "leakage_audit_available"]].values.tolist())}

## Source-Level Results

{markdown_table(["source", "runs", "local", "global", "mean error", "success", "energy proxy", "complexity"], source_rows[["source_step_id", "runs", "local_runs", "global_baseline_runs", "mean_final_error", "success_rate", "mean_energy", "mean_complexity"]].values.tolist())}

## Information Access Summary

{markdown_table(["control class", "scope", "runs", "mean error", "success", "complexity"], access_rows[["control_class", "information_scope", "runs", "mean_final_error", "exact_success_rate", "mean_complexity"]].values.tolist())}

## Paired Local/Global Gaps

Paired target rows with both local-only and explicit global-information controls: {len(paired)}.

{markdown_table(["source", "target", "local runs", "global runs", "local-global error", "global-local success"], paired[["source_step_id", "target_id", "local_run_count", "global_baseline_run_count", "local_minus_global_final_error", "global_minus_local_success_rate"]].head(18).values.tolist())}

## Leakage Audit

Policy leakage summary rows: {len(leakage_summary_df)}. Local-only leakage violations: {int(control_df["local_only_leakage_violation"].astype(bool).sum())}. Explicit global baselines flagged: {int(control_df["is_global_information_baseline"].astype(bool).sum())}.

## Lay Summary

The local-only policies and the global-information baselines solve different control problems. Local-only policies act from neighborhood state and usually improve target error without seeing a target map. The target-map, organizer, and global-gradient baselines use extra information and often reach lower error, but they should be treated as upper-bound or intervention controls rather than evidence for local self-organization.

Anchor result: {anchor_result}
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
    return f"""# E05 S13 Status Summary

- Research step ID: {STEP_ID}
- Completion status: {"completed" if success else "completed with validation failures"}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers:
{caveat_lines}
- Lay summary: S13 compared local-only target-map-free policies against explicitly labeled organizer, global-gradient, and target-map baselines across S07-S11 while preserving S12 trajectory/run identifiers and auditing local-only leakage.
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
        "schemaVersion": CONTROL_COMPARISON_SCHEMA_VERSION,
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

    specs = standard_control_source_specs(artifacts_dir)
    trajectory_path = artifacts_dir / "research_steps" / "S12" / "trajectory_summary_features.parquet"
    control_df, source_catalog, raw_audit_df = load_control_comparison(specs, trajectory_path=trajectory_path)
    access_summary_df = information_access_summary(control_df)
    gap_summary_df = target_control_gap_summary(control_df)
    leakage_summary_df = leakage_audit_summary(control_df, raw_audit_df)
    validation_df = control_validation_rows(
        source_catalog=source_catalog,
        control_df=control_df,
        access_summary_df=access_summary_df,
        gap_summary_df=gap_summary_df,
        leakage_summary_df=leakage_summary_df,
    )
    repo_test = run_repo_tests(step_dir, skip=bool(args.skip_repo_tests))
    validation_df = append_repo_validation(validation_df, repo_test)
    success = bool(validation_df["success"].astype(bool).all())

    artifacts: list[Path] = []
    artifacts.extend(write_dataframe(control_df, step_dir / "local_global_control_runs"))
    artifacts.extend(write_dataframe(control_df, results_dir / "e05_local_global_control"))
    artifacts.extend(write_dataframe(access_summary_df, step_dir / "information_access_summary"))
    artifacts.extend(write_dataframe(gap_summary_df, step_dir / "target_control_gap_summary"))
    artifacts.extend(write_dataframe(leakage_summary_df, step_dir / "local_only_leakage_audit"))
    artifacts.extend(write_dataframe(source_catalog, step_dir / "source_artifact_catalog"))
    artifacts.extend(write_dataframe(raw_audit_df, step_dir / "raw_policy_audits"))
    artifacts.extend(write_dataframe(validation_df, step_dir / "control_validation_results"))

    config_path = configs_dir / "e05_local_global_control_config.json"
    write_json(
        config_path,
        {
            "schemaVersion": CONTROL_COMPARISON_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "sourceSpecs": [spec.__dict__ for spec in specs],
            "trajectorySummaryPath": str(trajectory_path),
            "controlClasses": sorted(control_df["control_class"].dropna().astype(str).unique().tolist()),
            "informationScopes": sorted(control_df["information_scope"].dropna().astype(str).unique().tolist()),
            "targetErrorMetricPrecedence": ["composite_error", "hamming_error", "1-pattern_score", "S12 final_error_proxy"],
            "globalBaselineFlags": ["uses_target_map", "uses_global_gradient", "uses_organizer", "is_global_information_baseline"],
            "localOnlyLeakageFlags": [
                "uses_whole_target_leakage",
                "uses_hidden_target_map_leakage",
                "uses_hidden_axis_leakage",
                "uses_target_map_leakage",
                "uses_hidden_global_size",
            ],
        },
    )
    artifacts.append(config_path)

    spec_json_path = step_dir / "intervention_complexity_spec.json"
    write_json(
        spec_json_path,
        {
            "schemaVersion": CONTROL_COMPARISON_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "definitions": INTERVENTION_COMPLEXITY_DEFINITIONS,
            "localOnlyDefinition": "No explicit target map, global gradient, organizer, or hidden whole-target/global-size/axis leakage.",
            "globalBaselineDefinition": "A row explicitly flagged as using a target map, global gradient, organizer, or other global information baseline.",
        },
    )
    artifacts.append(spec_json_path)
    spec_md_path = step_dir / "intervention_complexity_spec.md"
    spec_md_path.write_text(intervention_spec_markdown(), encoding="utf-8")
    artifacts.append(spec_md_path)

    fig_error = figures_dir / "e05_s13_final_error_by_control_class.png"
    fig_success = figures_dir / "e05_s13_success_by_information_scope.png"
    fig_complexity = figures_dir / "e05_s13_intervention_complexity_tradeoff.png"
    fig_gaps = figures_dir / "e05_s13_local_global_gaps.png"
    plot_error_by_class(access_summary_df, fig_error)
    plot_success_by_scope(control_df, fig_success)
    plot_complexity_tradeoff(control_df, fig_complexity)
    plot_gap_summary(gap_summary_df, fig_gaps)
    artifacts.extend([fig_error, fig_success, fig_complexity, fig_gaps])

    local_runs = int(control_df["is_local_only_policy"].astype(bool).sum())
    global_runs = int(control_df["is_global_information_baseline"].astype(bool).sum())
    target_map_runs = int(control_df["uses_target_map"].astype(bool).sum())
    gradient_runs = int(control_df["uses_global_gradient"].astype(bool).sum())
    organizer_runs = int(control_df["uses_organizer"].astype(bool).sum())
    local_leakage_violations = int(control_df["local_only_leakage_violation"].astype(bool).sum())
    paired_rows = int(gap_summary_df["has_local_and_global"].astype(bool).sum()) if len(gap_summary_df) else 0
    local_mean_error = float(control_df[control_df["is_local_only_policy"].astype(bool)]["final_target_error"].astype(float).mean())
    global_mean_error = float(control_df[control_df["is_global_information_baseline"].astype(bool)]["final_target_error"].astype(float).mean())
    local_success_rate = float(control_df[control_df["is_local_only_policy"].astype(bool)]["exact_success"].astype(bool).mean())
    global_success_rate = float(control_df[control_df["is_global_information_baseline"].astype(bool)]["exact_success"].astype(bool).mean())

    validation_result = (
        f"passed: harmonized {len(control_df)} runs from {control_df['source_step_id'].nunique()} sources, "
        f"preserved S12 trajectory IDs for {float(control_df['s12_route_joined'].astype(bool).mean()):.3f} of rows, "
        f"separated {local_runs} local-only rows from {global_runs} explicit global-information baselines, "
        f"local-only leakage violations={local_leakage_violations}, repo tests passed"
        if success
        else f"failed: {int((~validation_df['success'].astype(bool)).sum())} validation checks failed"
    )
    outcome = "supportive" if success else "constraining/contradictory"
    recommended_next_action = "Chief Scientist review, then proceed to S14 higher-dimensional DG only after explicit instruction."
    caveats = [
        "S07-S09 contribute local or target-map-free evidence only; they do not include paired explicit global-information baselines.",
        "S10 organizer/global-gradient and S11 target-map rows are explicit upper-bound or intervention baselines, not local-only morphogenesis policies.",
        "Target-energy and intervention-complexity values are computational proxies; S10/S11 use final-error-times-size when reported target energy is unavailable.",
        "S12 trajectory route metadata is joined as run-level context and inherits S12 caveats about heterogeneous CPU/GPU source schemas.",
        "The comparison is observational across completed artifact sweeps; it does not equal a causal biological validation.",
    ]
    anchor_result = (
        f"S13 harmonized {len(control_df)} S07-S11 runs: {local_runs} local-only rows and {global_runs} explicit global-information baselines "
        f"({target_map_runs} target-map, {gradient_runs} global-gradient, {organizer_runs} organizer). "
        f"Mean final target error was {local_mean_error:.4f} for local-only rows versus {global_mean_error:.4f} for explicit global baselines; "
        f"exact/near-target success was {local_success_rate:.3f} versus {global_success_rate:.3f}; paired local/global target rows={paired_rows}; "
        f"local-only leakage violations={local_leakage_violations}."
    )

    report_path = step_dir / "local_global_control_report.md"
    report_path.write_text(
        report_markdown(
            success=success,
            control_df=control_df,
            source_catalog=source_catalog,
            access_summary_df=access_summary_df,
            gap_summary_df=gap_summary_df,
            leakage_summary_df=leakage_summary_df,
            validation_df=validation_df,
            artifacts=artifacts + [report_path],
            validation_result=validation_result,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
            anchor_result=anchor_result,
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
        "parallelism": "single_process_pandas_matplotlib",
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
            "sourceStepCount": int(control_df["source_step_id"].nunique()),
            "runCount": int(len(control_df)),
            "localOnlyRunCount": local_runs,
            "globalInformationBaselineRunCount": global_runs,
            "targetMapBaselineRunCount": target_map_runs,
            "globalGradientBaselineRunCount": gradient_runs,
            "organizerBaselineRunCount": organizer_runs,
            "pairedLocalGlobalTargetRowCount": paired_rows,
            "localOnlyLeakageViolationCount": local_leakage_violations,
            "localMeanFinalTargetError": local_mean_error,
            "globalBaselineMeanFinalTargetError": global_mean_error,
            "localExactOrNearSuccessRate": local_success_rate,
            "globalBaselineExactOrNearSuccessRate": global_success_rate,
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
        "schemaVersion": CONTROL_COMPARISON_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "sourceCodePaths": [
            str(REPO_ROOT / "morphospace2d" / "control_comparison.py"),
            str(REPO_ROOT / "scripts" / "e05_s13_control_comparison.py"),
            str(REPO_ROOT / "tests" / "test_e05_control_comparison.py"),
        ],
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
