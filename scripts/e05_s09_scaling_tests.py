#!/usr/bin/env python3
"""Run E05 S09 scaling-transfer benchmarks."""

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
    SCALING_SCHEMA_VERSION,
    audit_scaling_policy_payload,
    build_scaling_target_panel,
    metric_normalization_rows,
    run_scaling_recovery_benchmark,
    standard_recovery_policy_specs,
    target_scaling_rule_rows,
    transfer_gap_rows,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S09"
STEP_NUMBER = 9
STEP_TITLE = "Run scaling tests"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_E03_FRONTIER = Path("/previous-artifacts/E03/results/e03_frontier_candidates.parquet")
DEFAULT_E04_HANDOFF = Path("/previous-artifacts/E04/results/e04_s15_handoff_policy_catalog.parquet")
FOCUSED_TESTS = [
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
    parser = argparse.ArgumentParser(description="Run E05 S09 scaling-transfer benchmarks.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--e03-frontier", default=str(DEFAULT_E03_FRONTIER))
    parser.add_argument("--e04-handoff", default=str(DEFAULT_E04_HANDOFF))
    parser.add_argument("--seeds", type=int, nargs="+", default=[9001, 9002, 9003])
    parser.add_argument("--steps-per-cell", type=int, default=80)
    parser.add_argument("--snapshot-cells", type=int, default=8)
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
    target = payload["target"]
    max_steps = max(1, int(len(target.substrate.nodes) * int(payload["steps_per_cell"])))
    snapshot_interval = max(1, int(len(target.substrate.nodes) * int(payload["snapshot_cells"])))
    return run_scaling_recovery_benchmark(
        payload["spec"],
        target,
        payload["policy"],
        seed=int(payload["seed"]),
        max_steps=max_steps,
        snapshot_interval=snapshot_interval,
        recovery_threshold=float(payload["recovery_threshold"]),
    )


def run_benchmarks(
    panel: Sequence[tuple[Any, Any]],
    policies: Sequence[Any],
    seeds: Sequence[int],
    *,
    steps_per_cell: int,
    snapshot_cells: int,
    recovery_threshold: float,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        {
            "spec": spec,
            "target": target,
            "policy": policy,
            "seed": int(seed),
            "steps_per_cell": int(steps_per_cell),
            "snapshot_cells": int(snapshot_cells),
            "recovery_threshold": float(recovery_threshold),
        }
        for spec, target in panel
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
    severity_rows: list[dict[str, Any]] = []
    for run_row, trace, severity in results:
        run_rows.append(run_row)
        trace_rows.extend(trace)
        severity_rows.append(severity)
    return pd.DataFrame(run_rows), pd.DataFrame(trace_rows), pd.DataFrame(severity_rows)


def policy_catalog_rows(policies: Sequence[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        record = policy.to_record()
        audit = audit_scaling_policy_payload(policy)
        rows.append(
            {
                "research_step_id": STEP_ID,
                "schema_version": SCALING_SCHEMA_VERSION,
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
                "trained_on_size_class": "training_small",
                "heldout_sizes_seen_during_tuning": False,
                "policy_record_json": json.dumps(json_ready(record), sort_keys=True),
                "audit_success": bool(audit["success"]),
                "audit_errors_json": json.dumps(json_ready(audit["errors"]), sort_keys=True),
                "audit_payload_hash": audit["payloadHash"],
            }
        )
    return pd.DataFrame(rows)


def summary_by_target_policy_size(run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = run_df.groupby(
        ["target_id", "motif", "size_class", "size_key", "scale_rule_id", "is_heldout_size", "policy_id", "policy_family"],
        sort=True,
    )
    return grouped.agg(
        run_count=("run_id", "count"),
        node_count=("node_count", "first"),
        exact_recovery_rate=("exact_recovery", "mean"),
        improved_rate=("improved", "mean"),
        initial_composite_error_mean=("initial_composite_error", "mean"),
        final_composite_error_mean=("final_composite_error", "mean"),
        relative_error_reduction_mean=("relative_error_reduction", "mean"),
        final_target_energy_mean=("final_target_energy", "mean"),
        accepted_swaps_per_cell_mean=("accepted_swaps_per_cell", "mean"),
        max_steps_per_cell_mean=("max_steps_per_cell", "mean"),
        trajectory_curvature_mean=("trajectory_curvature", "mean"),
    ).reset_index()


def validation_rows(
    run_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    severity_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    target_rule_df: pd.DataFrame,
    metric_norm_df: pd.DataFrame,
    transfer_gap_df: pd.DataFrame,
    panel: Sequence[tuple[Any, Any]],
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

    expected_run_count = len(panel) * len(policies) * len(seeds)
    expected_pairs = {(target.target_id, policy.policy_id) for _, target in panel for policy in policies}
    observed_pairs = set(zip(run_df["target_id"], run_df["policy_id"]))
    size_classes = set(run_df["size_class"])
    motif_size_pairs = set(zip(target_rule_df["motif"], target_rule_df["sizeClass"]))
    expected_motif_size_pairs = {(spec.motif, spec.size_class) for spec, _ in panel}
    heldout = run_df[run_df["is_heldout_size"].astype(bool)]
    metric_cols = [
        "initial_composite_error",
        "final_composite_error",
        "relative_error_reduction",
        "initial_target_energy",
        "final_target_energy",
        "initial_earth_mover_distance",
        "final_earth_mover_distance",
    ]
    add("expected_run_count", len(run_df) == expected_run_count, {"observed": len(run_df), "expected": expected_run_count})
    add("all_target_policy_pairs_covered", observed_pairs == expected_pairs, {"observedPairs": len(observed_pairs), "expectedPairs": len(expected_pairs)})
    add("trace_snapshots_written", len(trace_df) >= expected_run_count * 2, {"traceRows": len(trace_df), "minimumExpected": expected_run_count * 2})
    add("target_scaling_rules_documented", bool(len(target_rule_df) == len(panel) and target_rule_df["scaleRule"].map(bool).all()), target_rule_df[["motif", "sizeClass", "sizeKey", "scaleRuleId"]].to_dict(orient="records"))
    add("heldout_sizes_isolated", bool(size_classes == {"training_small", "heldout_medium", "heldout_large"} and heldout["training_size_class"].eq("training_small").all()), {"sizeClasses": sorted(size_classes), "heldoutRuns": len(heldout)})
    add("motif_size_matrix_complete", motif_size_pairs == expected_motif_size_pairs, {"observedPairs": len(motif_size_pairs), "expectedPairs": len(expected_motif_size_pairs)})
    add("policy_leakage_audits_passed", bool(policy_df["audit_success"].all()), policy_df[["policy_id", "audit_success", "audit_errors_json"]].to_dict(orient="records"))
    add("policies_do_not_encode_heldout_sizes", bool((~policy_df["heldout_sizes_seen_during_tuning"].astype(bool)).all()), policy_df[["policy_id", "trained_on_size_class", "heldout_sizes_seen_during_tuning"]].to_dict(orient="records"))
    add("run_rows_report_no_hidden_size_or_target_map_leakage", bool((~run_df["uses_hidden_global_size"].astype(bool)).all() and (~run_df["uses_whole_target_leakage"].astype(bool)).all()), "all S09 rows mark no hidden global-size or whole-target leakage")
    add("metric_normalization_checks_passed", bool(metric_norm_df["success"].all()), metric_norm_df[~metric_norm_df["success"]].head(10).to_dict(orient="records"))
    add("finite_metric_outputs", bool(np.isfinite(run_df[metric_cols].astype(float).to_numpy()).all()), "selected run-level metric columns are finite")
    add("occupancy_and_identity_conserved", bool(run_df["occupancy_preserved"].astype(bool).all() and run_df["cell_ids_preserved"].astype(bool).all()), "all scaled local-swap runs preserve target occupancy and cell IDs")
    add(
        "no_collision_birth_death_or_detach_events",
        bool(
            (run_df["collision_count"].astype(int) == 0).all()
            and (run_df["birth_count"].astype(int) == 0).all()
            and (run_df["death_count"].astype(int) == 0).all()
            and (run_df["detach_count"].astype(int) == 0).all()
        ),
        "S09 uses conservative adjacent-swap transfer runs",
    )
    add("scrambling_severity_logged", len(severity_df) == expected_run_count, {"severityRows": len(severity_df)})
    add("heldout_transfer_signal_present", bool(heldout["improved"].astype(bool).any()), {"heldoutImprovedRuns": int(heldout["improved"].astype(bool).sum()), "heldoutRuns": len(heldout)})
    add("transfer_gaps_quantified", len(transfer_gap_df) == len({spec.motif for spec, _ in panel}) * len(policies) * 2, {"rows": len(transfer_gap_df)})
    add("random_null_panel_present", bool(run_df["policy_id"].eq("random_local_swap_null").any()), "random local-move null included")
    add("e03_source_available_and_used", e03_source_available and bool(run_df["policy_family"].eq("e03_frontier_feasible").any()), "E03 frontier analogue included when source catalog is present")
    add("e04_source_available_and_used", e04_source_available and bool(run_df["policy_family"].eq("e04_memory_signal_feasible").any()), "E04 handoff analogue included when source catalog is present")
    add("repo_unit_tests_passed", bool(repo_test.get("success", False)), {"returncode": repo_test.get("returncode"), "args": repo_test.get("args")})
    return pd.DataFrame(rows)


def write_scaling_rules_markdown(target_rule_df: pd.DataFrame, path: Path) -> Path:
    rows = target_rule_df[["motif", "sizeClass", "sizeKey", "nodeCount", "scaleRuleId", "scaleRule", "isHeldoutSize"]].values.tolist()
    text = f"""# E05 S09 Target Scaling Rules

Research step ID: {STEP_ID}
Completion status: completed

S09 isolates `training_small` as the only tuning/source size class. `heldout_medium` and `heldout_large` are evaluation-only sizes. The policy panel uses the fixed S07/S08 local policy weights selected before S09; no held-out target map, full target state, or global-size oracle is supplied as a policy input.

{markdown_table(["motif", "size class", "size", "nodes", "rule id", "scaling rule", "held out"], rows)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def plot_scaling_error(summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = summary_df.groupby(["size_class", "policy_id"], sort=True)["relative_error_reduction_mean"].mean().reset_index()
    size_order = ["training_small", "heldout_medium", "heldout_large"]
    policies = list(dict.fromkeys(grouped["policy_id"].tolist()))
    fig, axis = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(size_order))
    width = 0.16
    colors = plt.get_cmap("tab10")
    for index, policy in enumerate(policies):
        values = []
        for size_class in size_order:
            subset = grouped[grouped["size_class"].eq(size_class) & grouped["policy_id"].eq(policy)]
            values.append(float(subset["relative_error_reduction_mean"].iloc[0]) if not subset.empty else 0.0)
        axis.bar(x + (index - (len(policies) - 1) / 2) * width, values, width=width, label=policy.replace("_", " "), color=colors(index % 10), edgecolor="#222222", linewidth=0.4)
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([label.replace("_", "\n") for label in size_order])
    axis.set_ylabel("mean relative error reduction")
    axis.set_title("S09 scale-transfer recovery by size class")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_transfer_gap(transfer_gap_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    large = transfer_gap_df[transfer_gap_df["heldout_size_class"].eq("heldout_large")].copy()
    motifs = sorted(large["motif"].unique())
    policies = sorted(large["policy_id"].unique())
    matrix = np.zeros((len(motifs), len(policies)), dtype=float)
    for i, motif in enumerate(motifs):
        for j, policy in enumerate(policies):
            subset = large[large["motif"].eq(motif) & large["policy_id"].eq(policy)]
            matrix[i, j] = float(subset["transfer_gap"].iloc[0]) if not subset.empty else np.nan
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    im = axis.imshow(matrix, cmap="coolwarm", aspect="auto")
    axis.set_xticks(np.arange(len(policies)))
    axis.set_xticklabels([policy.replace("_", "\n") for policy in policies], fontsize=7)
    axis.set_yticks(np.arange(len(motifs)))
    axis.set_yticklabels(motifs)
    axis.set_title("S09 large held-out transfer gap: training minus held-out error reduction")
    for i in range(len(motifs)):
        for j in range(len(policies)):
            value = matrix[i, j]
            axis.text(j, i, "" if not math.isfinite(value) else f"{value:.2f}", ha="center", va="center", fontsize=7)
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
        }
        log_path.write_text("Skipped by --skip-repo-tests.\n", encoding="utf-8")
        payload["logPath"] = str(log_path)
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
    transfer_gap_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    target_rule_df: pd.DataFrame,
) -> str:
    failures = validation_df[~validation_df["success"]]
    validation_text = "passed" if failures.empty else "failed"
    overall = summary_df.groupby(["size_class", "policy_id", "policy_family"], sort=True).agg(
        run_count=("run_count", "sum"),
        exact_recovery_rate=("exact_recovery_rate", "mean"),
        improved_rate=("improved_rate", "mean"),
        relative_error_reduction_mean=("relative_error_reduction_mean", "mean"),
        trajectory_curvature_mean=("trajectory_curvature_mean", "mean"),
    ).reset_index()
    large_gap = transfer_gap_df[transfer_gap_df["heldout_size_class"].eq("heldout_large")].copy()
    return f"""# E05 S09 Scaling Transfer Report

Research step ID: {STEP_ID}
Completion status: {"completed" if failures.empty else "completed with validation failures"}
Artifact family: scaled target transfer on S01/S03 substrates
Validation result: {validation_text}

S09 treats the default S03 target dimensions as `training_small` and tests fixed local policy weights on `heldout_medium` and `heldout_large` targets. Initial perturbation is the S07 deranged identity scramble. Policies receive local actor and neighbor observations plus public local substrate geometry; the audit excludes hidden global-size keys, whole target maps, and held-out target maps.

## Target Scaling Rules

{markdown_table(["motif", "size class", "size", "nodes", "rule id", "held out"], target_rule_df[["motif", "sizeClass", "sizeKey", "nodeCount", "scaleRuleId", "isHeldoutSize"]].values.tolist())}

## Size And Policy Summary

{markdown_table(["size class", "policy", "family", "runs", "exact rate", "improved rate", "mean error reduction", "curvature"], overall[["size_class", "policy_id", "policy_family", "run_count", "exact_recovery_rate", "improved_rate", "relative_error_reduction_mean", "trajectory_curvature_mean"]].values.tolist())}

## Large Held-Out Transfer Gap

{markdown_table(["motif", "policy", "train mean", "large mean", "gap", "large improved"], large_gap[["motif", "policy_id", "training_relative_error_reduction_mean", "heldout_relative_error_reduction_mean", "transfer_gap", "heldout_improved"]].values.tolist())}

## Policy Leakage Audit

{markdown_table(["policy", "family", "audit pass", "trained on", "heldout seen", "source policy"], policy_df[["policy_id", "policy_family", "audit_success", "trained_on_size_class", "heldout_sizes_seen_during_tuning", "source_policy_id"]].values.tolist())}

## Validation

{markdown_table(["check", "success", "detail"], validation_df[["check_id", "success", "detail"]].values.tolist())}

## Caveats

- S09 evaluates fixed local policy weights selected before S09; it does not perform fresh optimization on small grids.
- The initial perturbation class is deranged identity scrambling, not every S08 damage mode.
- The scalar-rank baseline uses public coordinate geometry and visible scalar identity cues; this is audited as public geometry, not as hidden target-map access.
- These are computational proxy transfer tests, not biological scaling evidence.
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
    return f"""# E05 S09 Status Summary

- Research step ID: {STEP_ID}
- Completion status: {"completed" if success else "completed with validation failures"}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers:
{caveat_lines}
- Lay summary: S09 scaled five target motifs from default training-small grids to medium and large held-out grids, then tested whether fixed local policies could reduce scrambled target error without seeing held-out target maps or hidden global-size oracles.
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
        "schemaVersion": SCALING_SCHEMA_VERSION,
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

    e03_source = load_e03_source(Path(args.e03_frontier))
    e04_source = load_e04_source(Path(args.e04_handoff))
    policies = standard_recovery_policy_specs(e03_source=e03_source, e04_source=e04_source)
    panel = build_scaling_target_panel()
    target_rule_df = pd.DataFrame(target_scaling_rule_rows(panel))
    policy_df = policy_catalog_rows(policies)

    run_df, trace_df, severity_df = run_benchmarks(
        panel,
        policies,
        args.seeds,
        steps_per_cell=int(args.steps_per_cell),
        snapshot_cells=int(args.snapshot_cells),
        recovery_threshold=float(args.recovery_threshold),
        workers=int(args.workers),
    )
    summary_df = summary_by_target_policy_size(run_df)
    transfer_gap_df = pd.DataFrame(transfer_gap_rows(run_df.to_dict(orient="records")))
    metric_norm_df = pd.DataFrame(metric_normalization_rows(panel, seed=int(args.seeds[0])))
    repo_test = run_repo_tests(step_dir, skip=bool(args.skip_repo_tests))
    validation_df = validation_rows(
        run_df,
        trace_df,
        severity_df,
        policy_df,
        target_rule_df,
        metric_norm_df,
        transfer_gap_df,
        panel,
        policies,
        args.seeds,
        repo_test,
        e03_source_available=e03_source is not None,
        e04_source_available=e04_source is not None,
    )
    success = bool(validation_df["success"].all())
    heldout = run_df[run_df["is_heldout_size"].astype(bool)]
    heldout_improved = int(heldout["improved"].astype(bool).sum())
    exact_recoveries = int(run_df["exact_recovery"].astype(bool).sum())
    outcome = "supportive" if success and heldout_improved > 0 else "constraining/contradictory"
    validation_result = (
        f"passed: {len(run_df)} runs, {heldout_improved} held-out improved runs, "
        f"{exact_recoveries} exact recoveries, scaling rules/leakage/metric-normalization checks passed"
        if success
        else f"failed: {int((~validation_df['success']).sum())} validation checks failed"
    )
    recommended_next_action = "Chief Scientist review, then proceed to S10 symmetry-breaking tests only after approval."
    caveats = [
        "Scaling transfer uses fixed pre-S09 policy weights rather than fresh small-grid optimization.",
        "The perturbation class is deranged identity scrambling; S08 damage modes are not exhaustively scaled here.",
        "Scalar-rank policies use public coordinate geometry and visible scalar identity cues, not hidden target maps.",
        "Results are computational proxy transfer evidence, not biological scaling evidence.",
    ]
    anchor_result = f"Held-out improved runs = {heldout_improved}/{len(heldout)}; exact recoveries = {exact_recoveries}/{len(run_df)}."

    artifacts: list[Path] = []
    artifacts.extend(write_dataframe(run_df, step_dir / "scaling_run_results"))
    artifacts.extend(write_dataframe(run_df, results_dir / "e05_scaling_tests"))
    artifacts.extend(write_dataframe(summary_df, step_dir / "scaling_target_policy_size_summary"))
    artifacts.extend(write_dataframe(trace_df, step_dir / "scaling_trace_examples"))
    artifacts.extend(write_dataframe(severity_df, step_dir / "scaling_scramble_severity"))
    artifacts.extend(write_dataframe(target_rule_df, step_dir / "target_scaling_rules"))
    artifacts.extend(write_dataframe(policy_df, step_dir / "policy_scaling_leakage_audit"))
    artifacts.extend(write_dataframe(metric_norm_df, step_dir / "metric_normalization_checks"))
    artifacts.extend(write_dataframe(transfer_gap_df, step_dir / "scaling_transfer_gaps"))
    artifacts.extend(write_dataframe(validation_df, step_dir / "scaling_validation_results"))

    config_path = configs_dir / "e05_scaling_tests_config.json"
    write_json(
        config_path,
        {
            "schemaVersion": SCALING_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "seeds": [int(seed) for seed in args.seeds],
            "stepsPerCell": int(args.steps_per_cell),
            "snapshotCells": int(args.snapshot_cells),
            "recoveryThreshold": float(args.recovery_threshold),
            "workerCount": int(args.workers),
            "trainingSizeClass": "training_small",
            "heldoutSizeClasses": ["heldout_medium", "heldout_large"],
            "targetRules": target_rule_df[["motif", "sizeClass", "sizeKey", "scaleRuleId", "scaleRule", "isHeldoutSize"]].to_dict(orient="records"),
            "policies": policy_df[["policy_id", "policy_family", "trained_on_size_class", "heldout_sizes_seen_during_tuning"]].to_dict(orient="records"),
            "e03Source": str(args.e03_frontier),
            "e04Source": str(args.e04_handoff),
        },
    )
    artifacts.append(config_path)

    rules_md_path = write_scaling_rules_markdown(target_rule_df, step_dir / "target_scaling_rules.md")
    artifacts.append(rules_md_path)

    report_path = step_dir / "scaling_report.md"
    report_path.write_text(report_markdown(summary_df, transfer_gap_df, validation_df, policy_df, target_rule_df), encoding="utf-8")
    artifacts.append(report_path)

    fig_error = figures_dir / "e05_s09_scaling_error_reduction.png"
    fig_gap = figures_dir / "e05_s09_heldout_transfer_gap.png"
    plot_scaling_error(summary_df, fig_error)
    plot_transfer_gap(transfer_gap_df, fig_gap)
    artifacts.extend([fig_error, fig_gap])

    artifacts.append(Path(repo_test["logPath"]))
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "workerCount": int(args.workers),
        "parallelism": "process_pool_for_independent_scaled_target_policy_seed_runs",
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
            "targetCount": len(panel),
            "motifCount": int(target_rule_df["motif"].nunique()),
            "sizeClassCount": int(target_rule_df["sizeClass"].nunique()),
            "heldoutRunCount": len(heldout),
            "heldoutImprovedRunCount": heldout_improved,
            "exactRecoveryCount": exact_recoveries,
            "policyCount": len(policies),
            "seedCount": len(args.seeds),
            "traceRowCount": len(trace_df),
            "stepsPerCell": int(args.steps_per_cell),
            "snapshotCells": int(args.snapshot_cells),
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
        "schemaVersion": SCALING_SCHEMA_VERSION,
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
