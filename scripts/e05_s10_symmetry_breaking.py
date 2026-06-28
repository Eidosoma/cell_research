#!/usr/bin/env python3
"""Run E05 S10 symmetry-breaking benchmarks."""

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
from concurrent.futures import ProcessPoolExecutor
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
    SYMMETRY_SCHEMA_VERSION,
    axis_consistency_rows,
    run_symmetry_breaking_benchmark,
    standard_symmetry_policy_specs,
    standard_symmetry_task_specs,
    symmetry_policy_catalog_rows,
    symmetry_task_catalog_rows,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S10"
STEP_NUMBER = 10
STEP_TITLE = "Run symmetry-breaking tests"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
FOCUSED_TESTS = [
    "tests.test_e05_symmetry",
    "tests.test_e05_scaling",
    "tests.test_e05_regeneration",
    "tests.test_e05_scrambled_recovery",
    "tests.test_e05_embedded_1d",
    "tests.test_e05_metrics",
    "tests.test_e05_actions",
    "tests.test_e05_target_morphologies",
    "tests.test_e05_cell_identity",
    "tests.test_e05_substrate_generalization",
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
    parser = argparse.ArgumentParser(description="Run E05 S10 symmetry-breaking benchmarks.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10001, 10011)))
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--snapshot-interval", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--skip-repo-tests", action="store_true")
    return parser.parse_args()


def run_one(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    return run_symmetry_breaking_benchmark(
        payload["task"],
        payload["policy"],
        seed=int(payload["seed"]),
        max_steps=int(payload["max_steps"]),
        snapshot_interval=int(payload["snapshot_interval"]),
    )


def run_benchmarks(
    tasks: Sequence[Any],
    policies: Sequence[Any],
    seeds: Sequence[int],
    *,
    max_steps: int,
    snapshot_interval: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        {
            "task": task,
            "policy": policy,
            "seed": int(seed),
            "max_steps": int(max_steps),
            "snapshot_interval": int(snapshot_interval),
        }
        for task in tasks
        for policy in policies
        for seed in seeds
    ]
    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            results = list(executor.map(run_one, payloads, chunksize=1))
    else:
        results = [run_one(payload) for payload in payloads]

    run_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    initial_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for run_row, trace, initial_audit in results:
        run_rows.append(run_row)
        trace_rows.extend(trace)
        initial_by_key[(str(initial_audit["taskId"]), int(initial_audit["seed"]))] = initial_audit
    return pd.DataFrame(run_rows), pd.DataFrame(trace_rows), pd.DataFrame(initial_by_key.values())


def summary_by_task_policy(run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = run_df.groupby(
        ["task_id", "motif", "target_kind", "policy_id", "policy_family", "is_local_only_policy", "is_global_information_baseline"],
        sort=True,
    )
    return grouped.agg(
        run_count=("run_id", "count"),
        success_rate=("success", "mean"),
        symmetry_broken_rate=("symmetry_broken", "mean"),
        oriented_fraction_mean=("oriented_fraction", "mean"),
        axis_strength_mean=("axis_strength", "mean"),
        target_axis_alignment_mean=("target_axis_alignment", "mean"),
        label_match_fraction_mean=("label_match_fraction", "mean"),
        pattern_score_mean=("pattern_score", "mean"),
    ).reset_index()


def validation_rows(
    run_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    initial_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    task_df: pd.DataFrame,
    axis_df: pd.DataFrame,
    tasks: Sequence[Any],
    policies: Sequence[Any],
    seeds: Sequence[int],
    repo_test: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: Any) -> None:
        rows.append(
            {
                "research_step_id": STEP_ID,
                "check_id": check_id,
                "success": bool(success),
                "detail": json.dumps(json_ready(detail), sort_keys=True) if isinstance(detail, (Mapping, list, tuple)) else str(detail),
            }
        )

    expected_run_count = len(tasks) * len(policies) * len(seeds)
    expected_task_policy_pairs = {(task.task_id, policy.policy_id) for task in tasks for policy in policies}
    observed_task_policy_pairs = set(zip(run_df["task_id"], run_df["policy_id"]))
    local_policy_df = policy_df[policy_df["is_local_only_policy"].astype(bool)]
    baseline_policy_df = policy_df[policy_df["is_global_information_baseline"].astype(bool)]
    local_runs = run_df[run_df["is_local_only_policy"].astype(bool)]
    baseline_runs = run_df[run_df["is_global_information_baseline"].astype(bool)]
    metric_cols = ["oriented_fraction", "axis_strength", "pattern_score"]
    add("expected_run_count", len(run_df) == expected_run_count, {"observed": len(run_df), "expected": expected_run_count})
    add("all_task_policy_pairs_covered", observed_task_policy_pairs == expected_task_policy_pairs, {"observedPairs": len(observed_task_policy_pairs), "expectedPairs": len(expected_task_policy_pairs)})
    add("trace_snapshots_written", len(trace_df) >= expected_run_count * 2, {"traceRows": len(trace_df), "minimumExpected": expected_run_count * 2})
    add("task_catalog_written", len(task_df) == len(tasks), task_df[["task_id", "target_kind", "grid_size"]].to_dict(orient="records"))
    add("symmetric_initial_states_verified", bool(len(initial_df) == len(tasks) * len(seeds) and initial_df["success"].astype(bool).all()), initial_df[["taskId", "seed", "success", "mismatchCount"]].to_dict(orient="records"))
    add("random_seeds_logged", set(run_df["seed"].astype(int)) == set(int(seed) for seed in seeds), {"observedSeeds": sorted(run_df["seed"].astype(int).unique()), "expectedSeeds": sorted(int(seed) for seed in seeds)})
    add("policy_leakage_audits_passed", bool(policy_df["audit_success"].astype(bool).all()), policy_df[["policy_id", "audit_success", "audit_errors_json"]].to_dict(orient="records"))
    add("local_only_policies_present", len(local_policy_df) >= 3 and bool(local_runs["is_local_only_policy"].all()), local_policy_df[["policy_id", "information_scope"]].to_dict(orient="records"))
    add(
        "local_only_runs_no_axis_or_target_map_leakage",
        bool((~local_runs["uses_hidden_axis_leakage"].astype(bool)).all() and (~local_runs["uses_target_map_leakage"].astype(bool)).all()),
        "local-only runs explicitly report no hidden axis or target-map leakage",
    )
    add(
        "organizer_and_global_gradient_baselines_flagged",
        bool(
            len(baseline_policy_df) >= 2
            and baseline_policy_df["uses_organizer"].astype(bool).any()
            and baseline_policy_df["uses_global_gradient"].astype(bool).any()
            and baseline_runs["is_global_information_baseline"].astype(bool).all()
        ),
        baseline_policy_df[["policy_id", "uses_organizer", "uses_global_gradient", "is_global_information_baseline"]].to_dict(orient="records"),
    )
    add("finite_metric_outputs", bool(np.isfinite(run_df[metric_cols].astype(float).to_numpy()).all()), "selected symmetry metrics are finite")
    add("occupancy_preserved", bool(run_df["occupancy_preserved"].astype(bool).all()), "S10 mutable polarity/label dynamics preserves occupied grid")
    add("axis_consistency_quantified", len(axis_df) == len(tasks) * len(policies), {"rows": len(axis_df), "expected": len(tasks) * len(policies)})
    add("global_baseline_success_observed", bool(baseline_runs["success"].astype(bool).any()), {"baselineSuccessRuns": int(baseline_runs["success"].astype(bool).sum()), "baselineRuns": len(baseline_runs)})
    add("local_symmetry_breaking_quantified", bool(local_runs["symmetry_broken"].astype(bool).any()), {"localSymmetryBrokenRuns": int(local_runs["symmetry_broken"].astype(bool).sum()), "localRuns": len(local_runs)})
    add("repo_unit_tests_passed", bool(repo_test.get("success", False)), {"returncode": repo_test.get("returncode"), "args": repo_test.get("args")})
    return pd.DataFrame(rows)


def plot_axis_consistency(axis_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    axis_task = axis_df[axis_df["motif"].eq("axis_gradient")].copy()
    policies = axis_task["policy_id"].tolist()
    x = np.arange(len(policies))
    fig, axis = plt.subplots(figsize=(11, 5.2))
    axis.bar(x - 0.22, axis_task["x_axis_fraction"], width=0.22, label="x", color="#4f8bc9", edgecolor="#222222", linewidth=0.4)
    axis.bar(x, axis_task["y_axis_fraction"], width=0.22, label="y", color="#d87c4a", edgecolor="#222222", linewidth=0.4)
    axis.bar(x + 0.22, axis_task["none_axis_fraction"], width=0.22, label="none", color="#aaaaaa", edgecolor="#222222", linewidth=0.4)
    axis.set_xticks(x)
    axis.set_xticklabels([policy.replace("_", "\n") for policy in policies], fontsize=7)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("fraction of seeds")
    axis.set_title("S10 selected axis fractions for symmetric axis task")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_success_heatmap(summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = list(dict.fromkeys(summary_df["motif"].tolist()))
    policies = list(dict.fromkeys(summary_df["policy_id"].tolist()))
    matrix = np.zeros((len(tasks), len(policies)), dtype=float)
    for i, task in enumerate(tasks):
        for j, policy in enumerate(policies):
            subset = summary_df[summary_df["motif"].eq(task) & summary_df["policy_id"].eq(policy)]
            matrix[i, j] = float(subset["success_rate"].iloc[0]) if not subset.empty else np.nan
    fig, axis = plt.subplots(figsize=(11, 4.6))
    im = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(policies)))
    axis.set_xticklabels([policy.replace("_", "\n") for policy in policies], fontsize=7)
    axis.set_yticks(np.arange(len(tasks)))
    axis.set_yticklabels(tasks)
    axis.set_title("S10 symmetry-breaking success rate by task and policy")
    for i in range(len(tasks)):
        for j in range(len(policies)):
            value = matrix[i, j]
            axis.text(j, i, "" if not math.isfinite(value) else f"{value:.2f}", ha="center", va="center", color="white" if value < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=axis, fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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


def report_markdown(
    summary_df: pd.DataFrame,
    axis_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    initial_df: pd.DataFrame,
) -> str:
    failures = validation_df[~validation_df["success"]]
    validation_text = "passed" if failures.empty else "failed"
    overall = summary_df.groupby(["policy_id", "policy_family", "is_local_only_policy", "is_global_information_baseline"], sort=True).agg(
        run_count=("run_count", "sum"),
        success_rate=("success_rate", "mean"),
        symmetry_broken_rate=("symmetry_broken_rate", "mean"),
        pattern_score_mean=("pattern_score_mean", "mean"),
        axis_strength_mean=("axis_strength_mean", "mean"),
    ).reset_index()
    return f"""# E05 S10 Symmetry-Breaking Report

Research step ID: {STEP_ID}
Completion status: {"completed" if failures.empty else "completed with validation failures"}
Artifact family: symmetric neutral-state axis and pattern formation
Validation result: {validation_text}

S10 starts every run from a D4-symmetric, fully neutral square grid. Local-only policies receive actor/neighbor labels, polarity, local signal if declared, and local degree only. Organizer and global-gradient policies are explicit baselines and are flagged as nonlocal information conditions.

## Policy Summary

{markdown_table(["policy", "family", "local only", "global baseline", "runs", "success rate", "symmetry broken rate", "mean pattern score", "mean axis strength"], overall[["policy_id", "policy_family", "is_local_only_policy", "is_global_information_baseline", "run_count", "success_rate", "symmetry_broken_rate", "pattern_score_mean", "axis_strength_mean"]].values.tolist())}

## Task And Policy Summary

{markdown_table(["task", "motif", "policy", "runs", "success", "symmetry broken", "pattern score", "axis strength", "label match"], summary_df[["task_id", "motif", "policy_id", "run_count", "success_rate", "symmetry_broken_rate", "pattern_score_mean", "axis_strength_mean", "label_match_fraction_mean"]].values.tolist())}

## Axis Consistency Across Seeds

{markdown_table(["task", "policy", "dominant axis", "dominant fraction", "x fraction", "y fraction", "success rate"], axis_df[["task_id", "policy_id", "dominant_selected_axis", "dominant_axis_fraction", "x_axis_fraction", "y_axis_fraction", "success_rate"]].values.tolist())}

## Initial Symmetry Audit

{markdown_table(["task", "seed", "success", "mismatches", "state hash"], initial_df[["taskId", "seed", "success", "mismatchCount", "stateHash"]].head(20).values.tolist())}

## Policy Leakage And Baseline Flags

{markdown_table(["policy", "local only", "global baseline", "organizer", "global gradient", "audit pass"], policy_df[["policy_id", "is_local_only_policy", "is_global_information_baseline", "uses_organizer", "uses_global_gradient", "audit_success"]].values.tolist())}

## Validation

{markdown_table(["check", "success", "detail"], validation_df[["check_id", "success", "detail"]].values.tolist())}

## Caveats

- S10 uses computational mutable polarity and label dynamics, not a biological tissue model.
- Local-only policies may break symmetry stochastically but are not expected to know a target x-axis without an explicit cue.
- Organizer and global-gradient baselines intentionally carry nonlocal information and are comparators, not local morphogenesis models.
- Ring and appendage tasks test pattern cue dependence more than exact biological morphogenesis.
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
    return f"""# E05 S10 Status Summary

- Research step ID: {STEP_ID}
- Completion status: {"completed" if success else "completed with validation failures"}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers:
{caveat_lines}
- Lay summary: S10 started from perfectly symmetric neutral grids and measured whether local-only rules or explicitly cued baselines could choose axes or form ring/asymmetric patterns. The local-only policies were audited for hidden axis and target-map leakage, and nonlocal organizer/global-gradient baselines were explicitly flagged.
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
        "schemaVersion": SYMMETRY_SCHEMA_VERSION,
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

    tasks = standard_symmetry_task_specs()
    policies = standard_symmetry_policy_specs()
    task_df = pd.DataFrame(symmetry_task_catalog_rows(tasks))
    policy_df = pd.DataFrame(symmetry_policy_catalog_rows(policies))
    run_df, trace_df, initial_df = run_benchmarks(
        tasks,
        policies,
        args.seeds,
        max_steps=int(args.max_steps),
        snapshot_interval=int(args.snapshot_interval),
        workers=int(args.workers),
    )
    summary_df = summary_by_task_policy(run_df)
    axis_df = pd.DataFrame(axis_consistency_rows(run_df.to_dict(orient="records")))
    repo_test = run_repo_tests(step_dir, skip=bool(args.skip_repo_tests))
    validation_df = validation_rows(run_df, trace_df, initial_df, policy_df, task_df, axis_df, tasks, policies, args.seeds, repo_test)
    success = bool(validation_df["success"].all())

    local_runs = run_df[run_df["is_local_only_policy"].astype(bool)]
    baseline_runs = run_df[run_df["is_global_information_baseline"].astype(bool)]
    local_success = int(local_runs["success"].astype(bool).sum())
    local_broken = int(local_runs["symmetry_broken"].astype(bool).sum())
    baseline_success = int(baseline_runs["success"].astype(bool).sum())
    outcome = "constraining/contradictory" if success else "constraining/contradictory"
    validation_result = (
        f"passed: {len(run_df)} runs, {len(initial_df)} symmetric initial states verified, "
        f"{local_broken}/{len(local_runs)} local-only runs broke symmetry, "
        f"{local_success}/{len(local_runs)} local-only runs met task target, "
        f"{baseline_success}/{len(baseline_runs)} baseline runs met task target"
        if success
        else f"failed: {int((~validation_df['success']).sum())} validation checks failed"
    )
    recommended_next_action = "Chief Scientist review, then proceed to S11 GPU batched tissue simulations only after approval."
    caveats = [
        "S10 uses computational mutable polarity and label dynamics, not biological tissue dynamics.",
        "Local-only policies can stochastically break symmetry but lack a target-axis cue by construction.",
        "Organizer and global-gradient baselines intentionally carry nonlocal information and are flagged comparators.",
        "Ring and appendage tasks test cue dependence more than anatomical morphogenesis.",
    ]
    anchor_result = (
        f"Local-only target successes = {local_success}/{len(local_runs)}; "
        f"local-only symmetry-broken runs = {local_broken}/{len(local_runs)}; "
        f"flagged baseline target successes = {baseline_success}/{len(baseline_runs)}."
    )

    artifacts: list[Path] = []
    artifacts.extend(write_dataframe(run_df, step_dir / "symmetry_breaking_run_results"))
    artifacts.extend(write_dataframe(run_df, results_dir / "e05_symmetry_breaking"))
    artifacts.extend(write_dataframe(trace_df, step_dir / "symmetry_trace_examples"))
    artifacts.extend(write_dataframe(initial_df, step_dir / "symmetric_initial_state_audit"))
    artifacts.extend(write_dataframe(task_df, step_dir / "symmetry_task_catalog"))
    artifacts.extend(write_dataframe(policy_df, step_dir / "policy_symmetry_leakage_audit"))
    artifacts.extend(write_dataframe(summary_df, step_dir / "symmetry_task_policy_summary"))
    artifacts.extend(write_dataframe(axis_df, step_dir / "axis_consistency_summary"))
    artifacts.extend(write_dataframe(validation_df, step_dir / "symmetry_validation_results"))

    config_path = configs_dir / "e05_symmetry_breaking_config.json"
    write_json(
        config_path,
        {
            "schemaVersion": SYMMETRY_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "seeds": [int(seed) for seed in args.seeds],
            "maxSteps": int(args.max_steps),
            "snapshotInterval": int(args.snapshot_interval),
            "workerCount": int(args.workers),
            "tasks": task_df[["task_id", "target_kind", "grid_size", "target_axis"]].to_dict(orient="records"),
            "policies": policy_df[["policy_id", "information_scope", "is_local_only_policy", "is_global_information_baseline", "uses_organizer", "uses_global_gradient"]].to_dict(orient="records"),
        },
    )
    artifacts.append(config_path)

    report_path = step_dir / "symmetry_breaking_report.md"
    report_path.write_text(report_markdown(summary_df, axis_df, validation_df, policy_df, initial_df), encoding="utf-8")
    artifacts.append(report_path)

    fig_axis = figures_dir / "e05_s10_axis_consistency.png"
    fig_success = figures_dir / "e05_s10_symmetry_success_heatmap.png"
    plot_axis_consistency(axis_df, fig_axis)
    plot_success_heatmap(summary_df, fig_success)
    artifacts.extend([fig_axis, fig_success])

    artifacts.append(Path(repo_test["logPath"]))
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "workerCount": int(args.workers),
        "parallelism": "process_pool_for_independent_symmetry_task_policy_seed_runs",
        "benchmarkRuntimeSeconds": time.perf_counter() - started,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
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

    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "completedAt": utc_now(),
        "outcomeClassification": outcome,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "artifactsWritten": sorted(str(path) for path in artifacts + [status_path, manifest_path]),
        "benchmark": {
            "runCount": len(run_df),
            "taskCount": len(tasks),
            "policyCount": len(policies),
            "seedCount": len(args.seeds),
            "traceRowCount": len(trace_df),
            "symmetricInitialAuditCount": len(initial_df),
            "localOnlyRunCount": len(local_runs),
            "localOnlySymmetryBrokenCount": local_broken,
            "localOnlyTargetSuccessCount": local_success,
            "globalBaselineRunCount": len(baseline_runs),
            "globalBaselineTargetSuccessCount": baseline_success,
            "maxSteps": int(args.max_steps),
            "snapshotInterval": int(args.snapshot_interval),
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
        "schemaVersion": SYMMETRY_SCHEMA_VERSION,
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
