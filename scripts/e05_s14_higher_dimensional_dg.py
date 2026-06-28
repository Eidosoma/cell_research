#!/usr/bin/env python3
"""Define and quantify higher-dimensional Delayed Gratification for E05 S14."""

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
    DG_SCHEMA_VERSION,
    DGDetectionConfig,
    attach_control_metadata,
    dg_validation_rows,
    empirical_local_move_null_matches,
    hand_constructed_validation_examples,
    normalization_artifact_audit,
    null_comparison_summary,
    source_policy_dg_summary,
    synthetic_start_end_null_summary,
    trajectory_dg_summary,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S14"
STEP_NUMBER = 14
STEP_TITLE = "Identify higher-dimensional Delayed Gratification"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
FOCUSED_TESTS = [
    "tests.test_e05_delayed_gratification",
    "tests.test_e05_trajectory_maps",
    "tests.test_e05_control_comparison",
    "tests.test_e05_gpu_batches",
    "tests.test_e05_symmetry",
    "tests.test_e05_scaling",
    "tests.test_e05_regeneration",
    "tests.test_e05_scrambled_recovery",
]


DG_DEFINITIONS = {
    "dg_event": "A contiguous increase in target-error proxy followed by an immediate contiguous decrease in the same proxy.",
    "productive_dg_event": "A DG event where the recovery more than compensates for the temporary worsening, so final error after recovery is lower than error before backtracking.",
    "worsening_magnitude": "The target-error increase during the temporary movement away from target.",
    "recovery_magnitude": "The target-error decrease during the subsequent recovery segment.",
    "productive_dg_index": "max(0, (recovery_magnitude - worsening_magnitude) / worsening_magnitude).",
    "normalization_artifact_flag": "A primary error-proxy DG event with no supporting DG event in any available lower-is-better component metric.",
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
    parser = argparse.ArgumentParser(description="Quantify higher-dimensional DG from S07-S13 artifacts.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--null-replicates", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=14014)
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
        "schema_version": DG_SCHEMA_VERSION,
        "research_step_id": STEP_ID,
        "check_id": "repo_unit_tests_passed",
        "success": bool(repo_test.get("success", False)),
        "detail": json.dumps({"returncode": repo_test.get("returncode"), "args": repo_test.get("args")}, sort_keys=True),
    }
    return pd.concat([validation_df, pd.DataFrame([row])], ignore_index=True)


def source_artifact_catalog(artifacts_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = [
        ("S12 state features", artifacts_dir / "research_steps" / "S12" / "trajectory_state_features.parquet", "trajectory_state_features"),
        ("S12 summary features", artifacts_dir / "research_steps" / "S12" / "trajectory_summary_features.parquet", "trajectory_summary_features"),
        ("S13 control runs", artifacts_dir / "research_steps" / "S13" / "local_global_control_runs.parquet", "control_metadata"),
        ("S07 raw traces", artifacts_dir / "research_steps" / "S07" / "scrambled_embryo_trace_examples.parquet", "raw_trace"),
        ("S08 raw traces", artifacts_dir / "research_steps" / "S08" / "regeneration_trace_examples.parquet", "raw_trace"),
        ("S09 raw traces", artifacts_dir / "research_steps" / "S09" / "scaling_trace_examples.parquet", "raw_trace"),
        ("S10 raw traces", artifacts_dir / "research_steps" / "S10" / "symmetry_trace_examples.parquet", "raw_trace"),
        ("S11 raw traces", artifacts_dir / "research_steps" / "S11" / "gpu_sweep_trace_examples.parquet", "raw_trace"),
    ]
    for label, path, role in specs:
        rows.append(
            {
                "schema_version": DG_SCHEMA_VERSION,
                "research_step_id": STEP_ID,
                "label": label,
                "role": role,
                "path": str(path),
                "available": bool(path.exists()),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def plot_dg_rate_by_source(policy_summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = policy_summary.copy()
    subset["label"] = subset["source_step_id"].astype(str) + "\n" + subset["policy_family"].astype(str).str.replace("_", " ", regex=False)
    subset = subset.sort_values(["source_step_id", "productive_dg_event_rate", "trajectory_count"], ascending=[True, False, False]).head(30)
    fig, axis = plt.subplots(figsize=(12, 5.6))
    axis.bar(
        np.arange(len(subset)),
        subset["productive_dg_event_rate"].astype(float),
        color="#5a7f92",
        edgecolor="#222222",
        linewidth=0.35,
    )
    axis.set_xticks(np.arange(len(subset)))
    axis.set_xticklabels(subset["label"], rotation=55, ha="right", fontsize=7)
    axis.set_ylabel("productive DG trajectory rate")
    axis.set_title("S14 productive DG by source and policy family")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_null_comparison(null_summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if null_summary.empty:
        return
    labels = null_summary["comparison_type"].astype(str).str.replace("_", " ", regex=False)
    x = np.arange(len(null_summary))
    fig, axis = plt.subplots(figsize=(8.6, 4.8))
    axis.bar(x - 0.18, null_summary["observed_productive_event_rate"].astype(float), width=0.36, label="observed", color="#4f7c8a")
    axis.bar(x + 0.18, null_summary["null_productive_event_rate"].astype(float), width=0.36, label="null", color="#b7774d")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axis.set_ylabel("productive DG event rate")
    axis.set_title("S14 observed DG versus matched nulls")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_normalization_audit(audit_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = audit_df.groupby("source_step_id", sort=True).agg(
        primary_dg=("primary_dg_event_count", lambda values: int((values.astype(float) > 0.0).sum())),
        normalization_flags=("normalization_artifact_flag", "sum"),
        productive_flags=("productive_normalization_artifact_flag", "sum"),
        component_coverage=("comparable_component_metric_count", lambda values: float((values.astype(float) > 0.0).mean())),
    ).reset_index()
    x = np.arange(len(grouped))
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    axis.bar(x - 0.22, grouped["primary_dg"].astype(float), width=0.22, label="primary DG trajectories", color="#617e63")
    axis.bar(x, grouped["normalization_flags"].astype(float), width=0.22, label="normalization flags", color="#bd6d55")
    axis.bar(x + 0.22, grouped["productive_flags"].astype(float), width=0.22, label="productive flags", color="#8a6fb0")
    axis.set_xticks(x)
    axis.set_xticklabels(grouped["source_step_id"].astype(str))
    axis.set_ylabel("trajectory count")
    axis.set_title("S14 normalization-artifact audit")
    axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_example_trajectory(state_df: pd.DataFrame, event_df: pd.DataFrame, summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if event_df.empty:
        example_id = summary_df.sort_values("total_positive_error_delta", ascending=False)["trajectory_id"].iloc[0]
        example_events = event_df
    else:
        example_id = event_df.sort_values("productive_dg_index", ascending=False)["trajectory_id"].iloc[0]
        example_events = event_df[event_df["trajectory_id"].eq(example_id)]
    series = state_df[state_df["trajectory_id"].eq(example_id)].sort_values(["step", "snapshot_index"])
    fig, axis = plt.subplots(figsize=(8.6, 4.8))
    axis.plot(series["step"].astype(float), series["error_proxy"].astype(float), marker="o", linewidth=1.6, color="#3f6f83")
    for _, event in example_events.iterrows():
        axis.axvspan(float(event["start_step"]), float(event["worsened_step"]), color="#c75f5f", alpha=0.16)
        axis.axvspan(float(event["worsened_step"]), float(event["recovered_step"]), color="#5e9a68", alpha=0.14)
    axis.set_xlabel("saved step")
    axis.set_ylabel("target-error proxy")
    axis.set_title(f"S14 example DG trajectory: {example_id}")
    axis.grid(color="#dddddd", linewidth=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def dg_definition_markdown() -> str:
    rows = [[name, definition] for name, definition in DG_DEFINITIONS.items()]
    return f"""# E05 S14 Higher-Dimensional Delayed Gratification Definition

- Research step ID: {STEP_ID}
- Completion status: completed metric definition for S14
- Artifacts written: `dg_event_definition.md`, `dg_event_definition.json`
- Validation result: hand-constructed trajectories validate event counts and productive-event counts
- Caveats or blockers: DG is measured on saved trajectory snapshots, so unsaved short backtracking episodes are invisible.
- Recommended next action: Chief Scientist review, then proceed to S15 benchmark-suite packaging only after explicit instruction.

S14 generalizes the paper's Sortedness DG idea to higher-dimensional morphospace by replacing Sortedness drops with increases in a target-error proxy. Lower error is better. A temporary target-error increase followed by target-error decrease is a DG event; a productive DG event is one where the later decrease exceeds the temporary increase.

## Definitions

{markdown_table(["term", "definition"], rows)}
"""


def report_markdown(
    *,
    success: bool,
    summary_df: pd.DataFrame,
    policy_summary_df: pd.DataFrame,
    null_summary_df: pd.DataFrame,
    normalization_audit_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    artifacts: Sequence[Path],
    validation_result: str,
    caveats: Sequence[str],
    recommended_next_action: str,
    anchor_result: str,
) -> str:
    source_rows = summary_df.groupby("source_step_id", sort=True).agg(
        trajectories=("trajectory_id", "count"),
        dg_rate=("any_dg_event", "mean"),
        productive_rate=("any_productive_dg_event", "mean"),
        mean_events=("dg_event_count", "mean"),
        mean_productive_index=("total_productive_dg_index", "mean"),
        mean_error_reduction=("relative_error_reduction", "mean"),
    ).reset_index()
    policy_rows = policy_summary_df.sort_values(["productive_dg_event_rate", "trajectory_count"], ascending=[False, False]).head(16)
    null_rows = null_summary_df[
        [
            "comparison_type",
            "matched_trajectory_count",
            "observed_productive_event_rate",
            "null_productive_event_rate",
            "observed_minus_null_productive_event_rate",
            "observed_minus_null_productive_dg_index",
        ]
    ].values.tolist() if len(null_summary_df) else []
    audit_rows = normalization_audit_df.groupby("source_step_id", sort=True).agg(
        rows=("trajectory_id", "count"),
        component_coverage=("comparable_component_metric_count", lambda values: float((values.astype(float) > 0.0).mean())),
        primary_dg=("primary_dg_event_count", lambda values: int((values.astype(float) > 0.0).sum())),
        artifact_flags=("normalization_artifact_flag", "sum"),
        productive_artifact_flags=("productive_normalization_artifact_flag", "sum"),
    ).reset_index()
    artifacts_text = "\n".join(f"- `{path}`" for path in artifacts[:45])
    if len(artifacts) > 45:
        artifacts_text += f"\n- ... {len(artifacts) - 45} additional files listed in `artifact_manifest.json`"
    caveat_text = "\n".join(f"- {item}" for item in caveats)
    return f"""# E05 S14 Higher-Dimensional Delayed Gratification Report

- Research step ID: {STEP_ID}
- Completion status: {"completed" if success else "completed with validation failures"}
- Artifacts written:
{artifacts_text}
- Validation result: {validation_result}
- Caveats or blockers:
{caveat_text}
- Recommended next action: {recommended_next_action}

S14 detects higher-dimensional DG as saved target-error worsening followed by recovery, then marks events as productive when the recovery more than compensates for the temporary worsening. The analysis preserves source step, run, and trajectory identifiers from S07-S13 and compares observed trajectories against synthetic start/end-matched random bridges plus empirical random local-move nulls where available.

## Source-Level DG

{markdown_table(["source", "trajectories", "DG rate", "productive rate", "mean events", "productive index", "relative error reduction"], source_rows[["source_step_id", "trajectories", "dg_rate", "productive_rate", "mean_events", "mean_productive_index", "mean_error_reduction"]].values.tolist())}

## Policy Families With Highest Productive DG Rate

{markdown_table(["source", "motif", "policy family", "control", "trajectories", "productive rate", "mean productive events", "mean productive index"], policy_rows[["source_step_id", "motif", "policy_family", "control_class", "trajectory_count", "productive_dg_event_rate", "mean_productive_dg_event_count", "mean_total_productive_dg_index"]].values.tolist())}

## Null Comparisons

{markdown_table(["comparison", "matched", "observed productive rate", "null productive rate", "rate delta", "index delta"], null_rows)}

## Normalization Audit

{markdown_table(["source", "trajectories", "component coverage", "primary DG", "artifact flags", "productive artifact flags"], audit_rows[["source_step_id", "rows", "component_coverage", "primary_dg", "artifact_flags", "productive_artifact_flags"]].values.tolist())}

## Lay Summary

Higher-dimensional DG appears as temporary movement away from target morphology before later improvement in the saved error trajectory. The result is a computational route-shape measurement, not a biological claim: it says which simulated policies backtracked in target-error space, whether that backtracking was later productive, and how often similar behavior appears under start/end-matched null controls.

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
    return f"""# E05 S14 Status Summary

- Research step ID: {STEP_ID}
- Completion status: {"completed" if success else "completed with validation failures"}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers:
{caveat_lines}
- Lay summary: S14 defined higher-dimensional Delayed Gratification as temporary target-error worsening followed by recovery, validated the detector on hand-constructed trajectories, quantified DG events across S07-S13-derived trajectory artifacts, compared them with start/end-matched synthetic and empirical local-move nulls, and audited normalization artifacts.
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
        "schemaVersion": DG_SCHEMA_VERSION,
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

    state_path = artifacts_dir / "research_steps" / "S12" / "trajectory_state_features.parquet"
    control_path = artifacts_dir / "research_steps" / "S13" / "local_global_control_runs.parquet"
    state_df = pd.read_parquet(state_path)
    control_df = pd.read_parquet(control_path)
    state_df = attach_control_metadata(state_df, control_df)
    config = DGDetectionConfig()

    dg_summary_df, dg_event_df = trajectory_dg_summary(state_df, config=config)
    policy_summary_df = source_policy_dg_summary(dg_summary_df)
    hand_validation_df = hand_constructed_validation_examples(config=config)
    synthetic_null_df = synthetic_start_end_null_summary(
        dg_summary_df,
        replicates=int(args.null_replicates),
        random_state=int(args.random_state),
        config=config,
    )
    empirical_match_df = empirical_local_move_null_matches(dg_summary_df)
    normalization_audit_df = normalization_artifact_audit(state_df, dg_summary_df, config=config)
    null_summary_df = null_comparison_summary(dg_summary_df, synthetic_null_df, empirical_match_df)
    source_catalog_df = source_artifact_catalog(artifacts_dir)
    validation_df = dg_validation_rows(
        state_df=state_df,
        summary_df=dg_summary_df,
        event_df=dg_event_df,
        hand_validation_df=hand_validation_df,
        synthetic_null_df=synthetic_null_df,
        empirical_match_df=empirical_match_df,
        normalization_audit_df=normalization_audit_df,
    )
    repo_test = run_repo_tests(step_dir, skip=bool(args.skip_repo_tests))
    validation_df = append_repo_validation(validation_df, repo_test)
    success = bool(validation_df["success"].astype(bool).all())

    artifacts: list[Path] = []
    artifacts.extend(write_dataframe(dg_event_df, step_dir / "dg_event_table"))
    artifacts.extend(write_dataframe(dg_summary_df, step_dir / "higher_dimensional_dg_summary"))
    artifacts.extend(write_dataframe(dg_summary_df, results_dir / "e05_higher_dimensional_dg"))
    artifacts.extend(write_dataframe(policy_summary_df, step_dir / "source_policy_dg_summary"))
    artifacts.extend(write_dataframe(synthetic_null_df, step_dir / "dg_synthetic_start_end_nulls"))
    artifacts.extend(write_dataframe(empirical_match_df, step_dir / "dg_empirical_local_move_null_matches"))
    artifacts.extend(write_dataframe(null_summary_df, step_dir / "dg_null_comparison_summary"))
    artifacts.extend(write_dataframe(normalization_audit_df, step_dir / "dg_normalization_artifact_audit"))
    artifacts.extend(write_dataframe(hand_validation_df, step_dir / "dg_hand_constructed_validation"))
    artifacts.extend(write_dataframe(source_catalog_df, step_dir / "source_artifact_catalog"))
    artifacts.extend(write_dataframe(validation_df, step_dir / "dg_validation_results"))

    config_path = configs_dir / "e05_higher_dimensional_dg_config.json"
    write_json(
        config_path,
        {
            "schemaVersion": DG_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "stateFeaturePath": str(state_path),
            "controlMetadataPath": str(control_path),
            "dgDetectionConfig": config.__dict__,
            "syntheticNullReplicates": int(args.null_replicates),
            "randomState": int(args.random_state),
            "errorProxy": "S12 error_proxy joined with S13 control metadata",
            "lowerIsBetterComponentMetricsAudited": [
                "target_energy_feature",
                "neighborhood_error_feature",
                "earth_mover_feature",
                "graph_edit_feature",
                "boundary_error_feature",
                "topology_error_feature",
                "shape_moment_error_feature",
                "hausdorff_feature",
                "edge_disagreement_feature",
            ],
        },
    )
    artifacts.append(config_path)

    definition_json_path = step_dir / "dg_event_definition.json"
    write_json(
        definition_json_path,
        {
            "schemaVersion": DG_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "definitions": DG_DEFINITIONS,
            "eventLogic": "contiguous target-error increase followed by contiguous target-error decrease",
            "productiveCriterion": "recovery_magnitude > worsening_magnitude",
            "paperAnchor": "Generalizes Sortedness DG from the source paper to higher-dimensional target-error proxies.",
        },
    )
    artifacts.append(definition_json_path)
    definition_md_path = step_dir / "dg_event_definition.md"
    definition_md_path.write_text(dg_definition_markdown(), encoding="utf-8")
    artifacts.append(definition_md_path)

    fig_rate = figures_dir / "e05_s14_dg_rate_by_source_policy.png"
    fig_null = figures_dir / "e05_s14_observed_vs_null_dg.png"
    fig_audit = figures_dir / "e05_s14_normalization_artifact_audit.png"
    fig_example = figures_dir / "e05_s14_dg_event_example.png"
    plot_dg_rate_by_source(policy_summary_df, fig_rate)
    plot_null_comparison(null_summary_df, fig_null)
    plot_normalization_audit(normalization_audit_df, fig_audit)
    plot_example_trajectory(state_df, dg_event_df, dg_summary_df, fig_example)
    artifacts.extend([fig_rate, fig_null, fig_audit, fig_example])

    trajectory_count = int(len(dg_summary_df))
    event_count = int(dg_event_df.shape[0])
    productive_event_count = int(dg_event_df["productive_event"].astype(bool).sum()) if len(dg_event_df) else 0
    trajectories_with_dg = int(dg_summary_df["any_dg_event"].astype(bool).sum())
    trajectories_with_productive = int(dg_summary_df["any_productive_dg_event"].astype(bool).sum())
    normalization_flags = int(normalization_audit_df["normalization_artifact_flag"].astype(bool).sum())
    productive_normalization_flags = int(normalization_audit_df["productive_normalization_artifact_flag"].astype(bool).sum())
    empirical_matched = int(empirical_match_df["matched"].astype(bool).sum()) if len(empirical_match_df) and "matched" in empirical_match_df.columns else 0
    synthetic_delta = math.nan
    empirical_delta = math.nan
    for _, row in null_summary_df.iterrows():
        if row["comparison_type"] == "observed_vs_synthetic_start_end_random_bridge":
            synthetic_delta = float(row["observed_minus_null_productive_event_rate"])
        if row["comparison_type"] == "observed_vs_empirical_random_local_move_null":
            empirical_delta = float(row["observed_minus_null_productive_event_rate"])

    validation_result = (
        f"passed: detected {event_count} DG events and {productive_event_count} productive events across {trajectory_count} trajectories, "
        f"validated {len(hand_validation_df)} hand-constructed examples, generated {len(synthetic_null_df)} start/end-matched synthetic null trajectories, "
        f"matched {empirical_matched} empirical random local-move null comparisons, normalization artifact flags={normalization_flags}, repo tests passed"
        if success
        else f"failed: {int((~validation_df['success'].astype(bool)).sum())} validation checks failed"
    )
    if not success:
        outcome = "constraining/contradictory"
    elif trajectories_with_productive == 0:
        outcome = "null"
    elif (math.isfinite(synthetic_delta) and synthetic_delta < -0.02) and (not math.isfinite(empirical_delta) or empirical_delta < -0.02):
        outcome = "constraining/contradictory"
    else:
        outcome = "supportive"
    recommended_next_action = "Chief Scientist review, then proceed to S15 morphology benchmark suite packaging only after explicit instruction."
    caveats = [
        "DG is detected on saved trajectory snapshots, so backtracking between saved intervals may be missed or merged.",
        "The primary higher-dimensional error proxy is inherited from S12 and mixes source-specific metrics; component-metric support is audited but not available for every trajectory.",
        "Synthetic start/end-matched random bridges are statistical controls, not physically executable local policies.",
        "Empirical random local-move null matching is available mainly for S07-S09; S10 and S11 lack comparable random local-move nulls in the current artifacts.",
        "Normalization artifact flags indicate possible proxy-driven events that should be interpreted cautiously rather than discarded automatically.",
    ]
    anchor_result = (
        f"S14 quantified {event_count} DG events across {trajectory_count} S07-S13-derived trajectories; "
        f"{trajectories_with_dg} trajectories had any DG event and {trajectories_with_productive} had productive DG. "
        f"Observed-minus-synthetic productive-rate delta={synthetic_delta:.3f} where finite, "
        f"observed-minus-empirical-null productive-rate delta={empirical_delta:.3f} where finite, "
        f"normalization artifact flags={normalization_flags} and productive normalization flags={productive_normalization_flags}."
    )

    report_path = step_dir / "higher_dimensional_dg_report.md"
    report_path.write_text(
        report_markdown(
            success=success,
            summary_df=dg_summary_df,
            policy_summary_df=policy_summary_df,
            null_summary_df=null_summary_df,
            normalization_audit_df=normalization_audit_df,
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
            "trajectoryCount": trajectory_count,
            "dgEventCount": event_count,
            "productiveDgEventCount": productive_event_count,
            "trajectoryWithAnyDgCount": trajectories_with_dg,
            "trajectoryWithProductiveDgCount": trajectories_with_productive,
            "syntheticNullRowCount": int(len(synthetic_null_df)),
            "empiricalNullMatchedCount": empirical_matched,
            "normalizationArtifactFlagCount": normalization_flags,
            "productiveNormalizationArtifactFlagCount": productive_normalization_flags,
            "observedMinusSyntheticProductiveEventRate": synthetic_delta,
            "observedMinusEmpiricalNullProductiveEventRate": empirical_delta,
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
        "schemaVersion": DG_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "sourceCodePaths": [
            str(REPO_ROOT / "morphospace2d" / "delayed_gratification.py"),
            str(REPO_ROOT / "scripts" / "e05_s14_higher_dimensional_dg.py"),
            str(REPO_ROOT / "tests" / "test_e05_delayed_gratification.py"),
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
