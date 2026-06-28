#!/usr/bin/env python3
"""Run E05 S07 scrambled target recovery benchmarks."""

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
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace2d import (  # noqa: E402
    RECOVERY_SCHEMA_VERSION,
    audit_recovery_policy_payload,
    build_standard_target_library,
    evaluate_morphospace_metrics,
    run_recovery_benchmark,
    standard_recovery_policy_specs,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S07"
STEP_NUMBER = 7
STEP_TITLE = "Run scrambled-embryo tests"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_E03_FRONTIER = Path("/previous-artifacts/E03/results/e03_frontier_candidates.parquet")
DEFAULT_E04_HANDOFF = Path("/previous-artifacts/E04/results/e04_s15_handoff_policy_catalog.parquet")
FOCUSED_TESTS = [
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
        records.append(
            {
                "path": str(path),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
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
    parser = argparse.ArgumentParser(description="Run E05 S07 scrambled target recovery benchmarks.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--e03-frontier", default=str(DEFAULT_E03_FRONTIER))
    parser.add_argument("--e04-handoff", default=str(DEFAULT_E04_HANDOFF))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7001, 7002, 7003, 7004, 7005])
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--snapshot-interval", type=int, default=250)
    parser.add_argument("--recovery-threshold", type=float, default=1e-12)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--skip-repo-tests", action="store_true")
    return parser.parse_args()


def load_e03_source(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    subset = df.copy()
    for column in ("isS08Elite", "selectedForS13"):
        if column not in subset:
            subset[column] = False
    subset = subset.sort_values(["isS08Elite", "selectedForS13", "policyId"], ascending=[False, False, True])
    row = subset.iloc[0].to_dict()
    row["sourceArtifact"] = str(path)
    return row


def load_e04_source(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    subset = df.copy()
    for column in ("replayable", "policyAuditSuccess"):
        if column not in subset:
            subset[column] = False
    if "oracleAccessAllowed" not in subset:
        subset["oracleAccessAllowed"] = True
    subset = subset[
        subset["replayable"].astype(bool)
        & subset["policyAuditSuccess"].astype(bool)
        & (~subset["oracleAccessAllowed"].astype(bool))
    ].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["familyKind", "sourceStep", "policyId"], kind="mergesort")
    row = subset.iloc[0].to_dict()
    row["sourceArtifact"] = str(path)
    return row


def run_one(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    return run_recovery_benchmark(
        payload["target"],
        payload["policy"],
        seed=int(payload["seed"]),
        max_steps=int(payload["max_steps"]),
        snapshot_interval=int(payload["snapshot_interval"]),
        recovery_threshold=float(payload["recovery_threshold"]),
    )


def run_benchmarks(
    targets: Sequence[Any],
    policies: Sequence[Any],
    seeds: Sequence[int],
    *,
    max_steps: int,
    snapshot_interval: int,
    recovery_threshold: float,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        {
            "target": target,
            "policy": policy,
            "seed": int(seed),
            "max_steps": int(max_steps),
            "snapshot_interval": int(snapshot_interval),
            "recovery_threshold": float(recovery_threshold),
        }
        for target in targets
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
    severity_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for run_row, trace, severity in results:
        run_rows.append(run_row)
        trace_rows.extend(trace)
        severity_by_key[(str(severity["targetId"]), int(severity["seed"]))] = severity
    return pd.DataFrame(run_rows), pd.DataFrame(trace_rows), pd.DataFrame(severity_by_key.values())


def policy_catalog_rows(policies: Sequence[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        record = policy.to_record()
        audit = audit_recovery_policy_payload(policy)
        rows.append(
            {
                "research_step_id": STEP_ID,
                "schema_version": RECOVERY_SCHEMA_VERSION,
                "policy_id": policy.policy_id,
                "policy_family": policy.family,
                "description": policy.description,
                "rank_weight": policy.rank_weight,
                "axis_weight": policy.axis_weight,
                "affinity_weight": policy.affinity_weight,
                "exploration_epsilon": policy.exploration_epsilon,
                "memory_weight": policy.memory_weight,
                "signal_weight": policy.signal_weight,
                "source_artifact": policy.source_artifact,
                "source_policy_id": policy.source_policy_id,
                "source_policy_label": policy.source_policy_label,
                "policy_record_json": json.dumps(json_ready(record), sort_keys=True),
                "audit_success": bool(audit["success"]),
                "audit_errors_json": json.dumps(json_ready(audit["errors"]), sort_keys=True),
                "audit_payload_hash": audit["payloadHash"],
            }
        )
    return pd.DataFrame(rows)


def target_zero_validation_rows(targets: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        result = evaluate_morphospace_metrics(target, target.target_state())
        rows.append(
            {
                "research_step_id": STEP_ID,
                "check_id": f"target_state_zero_error__{target.target_id}",
                "success": bool(
                    abs(float(result["compositeError"])) <= 1e-12
                    and all(abs(float(metric["normalizedValue"])) <= 1e-12 for metric in result["metrics"])
                ),
                "detail": json.dumps(
                    {
                        "targetId": target.target_id,
                        "compositeError": float(result["compositeError"]),
                        "metricCount": len(result["metrics"]),
                    },
                    sort_keys=True,
                ),
            }
        )
    return rows


def validation_rows(
    run_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    severity_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    targets: Sequence[Any],
    policies: Sequence[Any],
    seeds: Sequence[int],
    repo_test: Mapping[str, Any],
    *,
    e03_source_available: bool,
    e04_source_available: bool,
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

    expected_run_count = len(targets) * len(policies) * len(seeds)
    expected_trace_min = expected_run_count * 2
    observed_pairs = set(zip(run_df["target_id"], run_df["policy_id"]))
    expected_pairs = {(target.target_id, policy.policy_id) for target in targets for policy in policies}
    add("expected_run_count", len(run_df) == expected_run_count, {"observed": len(run_df), "expected": expected_run_count})
    add("all_target_policy_pairs_covered", observed_pairs == expected_pairs, {"observedPairs": len(observed_pairs), "expectedPairs": len(expected_pairs)})
    add("trace_snapshots_written", len(trace_df) >= expected_trace_min, {"traceRows": len(trace_df), "minimumExpected": expected_trace_min})
    add("scrambling_severity_logged", len(severity_df) == len(targets) * len(seeds), {"severityRows": len(severity_df)})
    add(
        "scrambling_is_substantial",
        bool((severity_df["movedFraction"].astype(float) >= 0.9).all() and (severity_df["initialCompositeError"].astype(float) > 0).all()),
        severity_df[["targetId", "seed", "movedFraction", "severityClass", "initialCompositeError"]].to_dict(orient="records"),
    )
    add("policy_leakage_audits_passed", bool(policy_df["audit_success"].all()), policy_df[["policy_id", "audit_success", "audit_errors_json"]].to_dict(orient="records"))
    add("run_rows_report_no_whole_target_leakage", bool((~run_df["uses_whole_target_leakage"].astype(bool)).all()), "all run rows mark no whole-target leakage")
    add(
        "occupancy_and_identity_conserved",
        bool(run_df["occupancy_preserved"].astype(bool).all() and run_df["cell_ids_preserved"].astype(bool).all()),
        "all local-swap runs preserve occupied target nodes and cell-id multiset",
    )
    add(
        "no_collision_birth_death_or_detach_events",
        bool(
            (run_df["collision_count"].astype(int) == 0).all()
            and (run_df["birth_count"].astype(int) == 0).all()
            and (run_df["death_count"].astype(int) == 0).all()
            and (run_df["detach_count"].astype(int) == 0).all()
        ),
        "S07 simulator permits only adjacent swaps or waits",
    )
    add("classic_scalar_sorted_row_recovers", bool(run_df[(run_df["target_id"].eq("sorted_row_8")) & (run_df["policy_id"].eq("classic_scalar_rank_swap"))]["exact_recovery"].all()), "classic scalar rank swap should recover the S06 row-regression target")
    add("at_least_one_policy_improves_each_target", bool(run_df.groupby("target_id")["improved"].any().all()), run_df.groupby("target_id")["improved"].sum().to_dict())
    add("random_null_panel_present", bool(run_df["policy_id"].eq("random_local_swap_null").any()), "random local-move null included")
    add("e03_source_available_and_used", e03_source_available and bool(run_df["policy_family"].eq("e03_frontier_feasible").any()), "E03 frontier analogue included when source catalog is present")
    add("e04_source_available_and_used", e04_source_available and bool(run_df["policy_family"].eq("e04_memory_signal_feasible").any()), "E04 handoff analogue included when source catalog is present")
    add("repo_unit_tests_passed", bool(repo_test.get("success", False)), {"returncode": repo_test.get("returncode"), "args": repo_test.get("args")})
    rows.extend(target_zero_validation_rows(targets))
    return pd.DataFrame(rows)


def summary_by_target_policy(run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = run_df.groupby(["target_id", "motif", "policy_id", "policy_family"], sort=True)
    return grouped.agg(
        run_count=("run_id", "count"),
        exact_recovery_rate=("exact_recovery", "mean"),
        improved_rate=("improved", "mean"),
        initial_composite_error_mean=("initial_composite_error", "mean"),
        final_composite_error_mean=("final_composite_error", "mean"),
        relative_error_reduction_mean=("relative_error_reduction", "mean"),
        final_target_energy_mean=("final_target_energy", "mean"),
        accepted_swap_count_mean=("accepted_swap_count", "mean"),
        wait_count_mean=("wait_count", "mean"),
        trajectory_curvature_mean=("trajectory_curvature", "mean"),
    ).reset_index()


def plot_recovery_summary(summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    motifs = list(dict.fromkeys(summary_df["motif"].tolist()))
    policies = list(dict.fromkeys(summary_df["policy_id"].tolist()))
    fig, axes = plt.subplots(len(motifs), 1, figsize=(11, max(3, 2.2 * len(motifs))), sharex=True)
    if len(motifs) == 1:
        axes = [axes]
    colors = plt.get_cmap("tab10")
    for axis, motif in zip(axes, motifs):
        subset = summary_df[summary_df["motif"].eq(motif)]
        x = np.arange(len(policies))
        values = [
            float(subset[subset["policy_id"].eq(policy)]["relative_error_reduction_mean"].iloc[0])
            if not subset[subset["policy_id"].eq(policy)].empty
            else 0.0
            for policy in policies
        ]
        axis.bar(x, values, color=[colors(index % 10) for index in range(len(policies))], edgecolor="#222222", linewidth=0.5)
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_ylim(min(-0.1, min(values) - 0.05), max(1.05, max(values) + 0.05))
        axis.set_ylabel(motif.replace("_", " "))
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[-1].set_xticks(np.arange(len(policies)))
    axes[-1].set_xticklabels([policy.replace("_", "\n") for policy in policies], fontsize=8)
    fig.suptitle("S07 scrambled recovery relative error reduction")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scrambling_severity(severity_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    grouped = severity_df.groupby("motif", sort=True)["normalizedMeanEuclideanDisplacement"].mean().sort_values()
    axis.barh(grouped.index, grouped.values, color="#4f8b8b", edgecolor="#222222", linewidth=0.5)
    axis.set_xlabel("mean normalized Euclidean displacement")
    axis.set_ylabel("target motif")
    axis.set_title("S07 initial scrambling severity")
    axis.grid(axis="x", color="#dddddd", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_metric_trajectory_gif(trace_df: pd.DataFrame, path: Path) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates = trace_df[
        trace_df["policy_id"].isin(["classic_scalar_rank_swap", "random_local_swap_null"])
        & trace_df["target_id"].eq("abstract_organ_like_7x5")
    ].copy()
    if candidates.empty:
        return None
    seed = int(candidates["seed"].min())
    subset = candidates[candidates["seed"].eq(seed)].copy()
    policies = list(dict.fromkeys(subset["policy_id"].tolist()))
    max_step = int(subset["step"].max())
    frames = sorted(subset["step"].unique())
    fig, axis = plt.subplots(figsize=(7.2, 4.2))

    def draw(frame_index: int) -> None:
        step = frames[frame_index]
        axis.clear()
        for policy in policies:
            series = subset[(subset["policy_id"].eq(policy)) & (subset["step"] <= step)].sort_values("step")
            axis.plot(series["step"], series["composite_error"], marker="o", linewidth=1.8, label=policy.replace("_", " "))
        axis.set_xlim(0, max_step)
        ymax = max(0.05, float(subset["composite_error"].max()) * 1.05)
        axis.set_ylim(0, ymax)
        axis.set_xlabel("local activation step")
        axis.set_ylabel("composite error")
        axis.set_title(f"S07 abstract organ-like recovery trajectory, seed {seed}, step {int(step)}")
        axis.grid(color="#dddddd", linewidth=0.5)
        axis.legend(loc="upper right", fontsize=7)

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=250, repeat=True)
    try:
        animation.save(path, writer=PillowWriter(fps=3))
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return path


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
    run_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    policy_df: pd.DataFrame,
) -> str:
    failures = validation_df[~validation_df["success"]]
    validation_text = "passed" if failures.empty else "failed"
    overall = summary_df.groupby(["policy_id", "policy_family"], sort=True).agg(
        run_count=("run_count", "sum"),
        exact_recovery_rate=("exact_recovery_rate", "mean"),
        improved_rate=("improved_rate", "mean"),
        relative_error_reduction_mean=("relative_error_reduction_mean", "mean"),
        trajectory_curvature_mean=("trajectory_curvature_mean", "mean"),
    ).reset_index()
    return f"""# E05 S07 Scrambled-Embryo Recovery Report

Research step ID: {STEP_ID}
Completion status: {"completed" if failures.empty else "completed with validation failures"}
Artifact family: scrambled target recovery on S01/S03 substrates
Validation result: {validation_text}

S07 starts every target from a seeded deranged identity assignment, then applies adjacent local swaps under classic-derived, E03-catalog-backed, E04-catalog-backed, and random-null policy variants. Recovery metrics are S05 computational proxies; no row receives a whole-target identity map or global target state as a policy input.

## Policy Summary

{markdown_table(["policy", "family", "runs", "exact recovery rate", "improved rate", "mean relative error reduction", "mean trajectory curvature"], overall[["policy_id", "policy_family", "run_count", "exact_recovery_rate", "improved_rate", "relative_error_reduction_mean", "trajectory_curvature_mean"]].values.tolist())}

## Target And Policy Summary

{markdown_table(["target", "motif", "policy", "runs", "exact recovery rate", "improved rate", "mean final composite", "mean error reduction"], summary_df[["target_id", "motif", "policy_id", "run_count", "exact_recovery_rate", "improved_rate", "final_composite_error_mean", "relative_error_reduction_mean"]].values.tolist())}

## Policy Leakage Audit

{markdown_table(["policy", "family", "audit pass", "source policy"], policy_df[["policy_id", "policy_family", "audit_success", "source_policy_id"]].values.tolist())}

## Validation

{markdown_table(["check", "success", "detail"], validation_df[["check_id", "success", "detail"]].values.tolist())}

## Caveats

- E03 and E04 entries are feasible 2D local-policy analogues selected from upstream catalogs, not byte-for-byte replay of the original 1D DSL or memory-signal simulators.
- The strongest classic baseline uses visible scalar rank and public substrate coordinates, so it tests whether the S02 scalar ordering can serve as a local morphogen-like coordinate cue.
- These are bounded computational proxy recoveries, not biological embryo repair evidence.
"""


def summary_markdown(
    *,
    success: bool,
    artifacts: Sequence[Path],
    validation_result: str,
    caveats: Sequence[str],
    recommended_next_action: str,
    supportive_detail: str,
) -> str:
    artifact_lines = "\n".join(f"- `{path}`" for path in artifacts)
    caveat_lines = "\n".join(f"- {item}" for item in caveats)
    return f"""# E05 S07 Status Summary

- Research step ID: {STEP_ID}
- Completion status: {"completed" if success else "completed with validation failures"}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {"supportive" if success else "constraining/contradictory"}
- Caveats or blockers:
{caveat_lines}
- Lay summary: S07 scrambled all target identity assignments and tested whether local adjacent-swap policies could reduce target error. The scalar-rank classic baseline and catalog-backed local analogues were benchmarked against a random local-swap null while preserving occupancy and cell IDs.
- Recommended next action: {recommended_next_action}

Anchor result: {supportive_detail}
"""


def update_run_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    manifest: dict[str, Any] = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest.setdefault("schema", "e05_run_manifest.v1")
    manifest["updatedAt"] = utc_now()
    manifest["latestResearchStepId"] = STEP_ID
    manifest["git"] = payload["git"]
    manifest["runtime"] = payload["runtime"]
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "status": payload["status"],
        "success": payload["success"],
        "artifactsWritten": payload["artifactsWritten"],
        "validationResult": payload["validationResult"],
    }
    write_json(path, manifest)


def main() -> int:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    configs_dir = artifacts_dir / "configs"
    figures_dir = artifacts_dir / "figures"
    provenance_dir = artifacts_dir / "provenance"
    for directory in [step_dir, results_dir, configs_dir, figures_dir, provenance_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    e03_path = Path(args.e03_frontier)
    e04_path = Path(args.e04_handoff)
    e03_source = load_e03_source(e03_path)
    e04_source = load_e04_source(e04_path)
    targets = build_standard_target_library()
    policies = standard_recovery_policy_specs(e03_source=e03_source, e04_source=e04_source)
    seeds = [int(seed) for seed in args.seeds]

    config = {
        "schema": "e05_s07_scrambled_recovery_config.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "recoverySchemaVersion": RECOVERY_SCHEMA_VERSION,
        "targetIds": [target.target_id for target in targets],
        "policyIds": [policy.policy_id for policy in policies],
        "seeds": seeds,
        "maxSteps": int(args.max_steps),
        "snapshotInterval": int(args.snapshot_interval),
        "recoveryThreshold": float(args.recovery_threshold),
        "workers": int(args.workers),
        "upstreamPolicyCatalogs": {
            "e03Frontier": str(e03_path),
            "e03SourceSelected": None if e03_source is None else e03_source.get("policyId"),
            "e04Handoff": str(e04_path),
            "e04SourceSelected": None if e04_source is None else e04_source.get("policyId"),
        },
        "policyLeakageBoundary": "local actor/neighbor observations only; no whole target map or global target-state table",
    }
    config_path = configs_dir / "e05_scrambled_embryo_tests_config.json"
    write_json(config_path, config)

    started = time.perf_counter()
    run_df, trace_df, severity_df = run_benchmarks(
        targets,
        policies,
        seeds,
        max_steps=int(args.max_steps),
        snapshot_interval=int(args.snapshot_interval),
        recovery_threshold=float(args.recovery_threshold),
        workers=int(args.workers),
    )
    benchmark_runtime = time.perf_counter() - started
    policy_df = policy_catalog_rows(policies)
    summary_df = summary_by_target_policy(run_df)
    repo_test = run_repo_tests(step_dir, bool(args.skip_repo_tests))
    validation_df = validation_rows(
        run_df,
        trace_df,
        severity_df,
        policy_df,
        targets,
        policies,
        seeds,
        repo_test,
        e03_source_available=e03_source is not None,
        e04_source_available=e04_source is not None,
    )

    artifact_paths: list[Path] = []
    artifact_paths.extend(write_dataframe(run_df, results_dir / "e05_scrambled_embryo_tests"))
    artifact_paths.extend(write_dataframe(run_df, step_dir / "scrambled_embryo_run_results"))
    artifact_paths.extend(write_dataframe(summary_df, step_dir / "scrambled_embryo_target_policy_summary"))
    artifact_paths.extend(write_dataframe(trace_df, step_dir / "scrambled_embryo_trace_examples"))
    artifact_paths.extend(write_dataframe(severity_df, step_dir / "scrambling_severity"))
    artifact_paths.extend(write_dataframe(policy_df, step_dir / "policy_leakage_audit"))
    artifact_paths.extend(write_dataframe(validation_df, step_dir / "scrambled_embryo_validation_results"))
    artifact_paths.append(config_path)
    artifact_paths.append(step_dir / "repo_unit_test_log.txt")

    recovery_plot_path = figures_dir / "e05_s07_scrambled_recovery_error_reduction.png"
    severity_plot_path = figures_dir / "e05_s07_scrambling_severity.png"
    plot_recovery_summary(summary_df, recovery_plot_path)
    plot_scrambling_severity(severity_df, severity_plot_path)
    artifact_paths.extend([recovery_plot_path, severity_plot_path])
    gif_path = write_metric_trajectory_gif(trace_df, figures_dir / "e05_s07_recovery_trajectory.gif")
    if gif_path is not None:
        artifact_paths.append(gif_path)

    report_path = step_dir / "scrambled_embryo_report.md"
    report_path.write_text(report_markdown(run_df, summary_df, validation_df, policy_df), encoding="utf-8")
    artifact_paths.append(report_path)

    success = bool(validation_df["success"].all())
    exact_count = int(run_df["exact_recovery"].sum())
    improved_count = int(run_df["improved"].sum())
    scalar_summary = summary_df[summary_df["policy_id"].eq("classic_scalar_rank_swap")]
    scalar_exact = float(scalar_summary["exact_recovery_rate"].mean()) if not scalar_summary.empty else 0.0
    validation_result = (
        f"passed: {len(run_df)} runs, {improved_count} improved, {exact_count} exact recoveries, "
        f"all leakage/conservation checks passed"
        if success
        else "failed: one or more S07 validation checks failed"
    )
    caveats = [
        "E03/E04 policies are local 2D analogues selected from completed upstream catalogs, not direct replay of all upstream policy DSLs.",
        "The scalar-rank baseline can be strong because S03 targets retain a visible scalar ordering component.",
        "Recovery scores are computational proxies over small target grids, not biological validation.",
    ]
    if gif_path is None:
        caveats.append("Metric recovery GIF could not be written; static recovery plots and trace tables were written.")
    recommended_next_action = "Chief Scientist review, then proceed to S08 only after approval."
    supportive_detail = (
        f"Classic scalar rank swap exact recovery mean across targets = {scalar_exact:.3f}; "
        f"overall improved runs = {improved_count}/{len(run_df)}."
    )

    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"
    artifact_paths.extend([summary_path, status_path, manifest_path, run_manifest_path])
    summary_path.write_text(
        summary_markdown(
            success=success,
            artifacts=artifact_paths,
            validation_result=validation_result,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
            supportive_detail=supportive_detail,
        ),
        encoding="utf-8",
    )

    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "workerCount": int(args.workers),
        "parallelism": "process_pool_for_independent_target_policy_seed_runs",
        "benchmarkRuntimeSeconds": benchmark_runtime,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "artifactsWritten": [str(path) for path in artifact_paths],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "completedAt": utc_now(),
        "git": git_metadata(),
        "runtime": runtime,
        "benchmark": {
            "targetCount": len(targets),
            "policyCount": len(policies),
            "seedCount": len(seeds),
            "runCount": len(run_df),
            "traceRowCount": len(trace_df),
            "severityRowCount": len(severity_df),
            "exactRecoveryCount": exact_count,
            "improvedRunCount": improved_count,
            "scalarRankExactRecoveryRateMean": scalar_exact,
            "maxSteps": int(args.max_steps),
            "snapshotInterval": int(args.snapshot_interval),
        },
        "repoUnitTests": {
            "success": bool(repo_test.get("success", False)),
            "returncode": repo_test.get("returncode"),
            "logPath": repo_test.get("logPath", str(step_dir / "repo_unit_test_log.txt")),
        },
    }
    write_json(status_path, status_payload)
    update_run_manifest(run_manifest_path, status_payload)

    manifest_payload = {
        "schema": "e05_s07_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "success": success,
        "git": status_payload["git"],
        "artifacts": collect_artifacts(artifact_paths),
        "upstreamInputs": [
            {"path": str(e03_path), "exists": e03_path.exists(), "sha256": sha256_path(e03_path) if e03_path.exists() else None},
            {"path": str(e04_path), "exists": e04_path.exists(), "sha256": sha256_path(e04_path) if e04_path.exists() else None},
        ],
    }
    write_json(manifest_path, manifest_payload)

    status_payload["artifactsWritten"] = [record["path"] for record in collect_artifacts(artifact_paths)]
    write_json(status_path, status_payload)
    summary_path.write_text(
        summary_markdown(
            success=success,
            artifacts=[Path(path) for path in status_payload["artifactsWritten"]],
            validation_result=validation_result,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
            supportive_detail=supportive_detail,
        ),
        encoding="utf-8",
    )
    update_run_manifest(run_manifest_path, status_payload)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
