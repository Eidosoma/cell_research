#!/usr/bin/env python3
"""Run E05 S11 GPU batched tissue simulations."""

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
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace2d import (  # noqa: E402
    GPU_BATCH_SCHEMA_VERSION,
    cpu_gpu_validation_rows,
    gpu_policy_catalog_rows,
    gpu_sweep_summary_rows,
    gpu_target_catalog_rows,
    labels_to_rgb,
    run_batched_gpu_sweep,
    seed_reproducibility_rows,
    simulate_label_frame_sequence,
    standard_gpu_policy_specs,
    standard_gpu_sweep_targets,
    standard_gpu_validation_targets,
    torch_device_summary,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S11"
STEP_NUMBER = 11
STEP_TITLE = "Use GPU for batched tissue simulations"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
FOCUSED_TESTS = [
    "tests.test_e05_gpu_batches",
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
    parser = argparse.ArgumentParser(description="Run E05 S11 GPU batched tissue simulations.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=list(range(11101, 11109)))
    parser.add_argument("--sweep-seeds", type=int, nargs="+", default=list(range(11201, 11265)))
    parser.add_argument("--validation-steps", type=int, default=16)
    parser.add_argument("--sweep-steps", type=int, default=96)
    parser.add_argument("--trace-interval", type=int, default=24)
    parser.add_argument("--movie-interval", type=int, default=8)
    parser.add_argument("--skip-repo-tests", action="store_true")
    return parser.parse_args()


def run_all_sweeps(targets: Sequence[Any], policies: Sequence[Any], seeds: Sequence[int], *, steps: int, trace_interval: int, device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for target in targets:
        for policy in policies:
            rows, trace, _ = run_batched_gpu_sweep(
                target,
                policy,
                seeds,
                steps=int(steps),
                device=device,
                trace_interval=int(trace_interval),
            )
            run_rows.extend(rows)
            trace_rows.extend(trace)
    return pd.DataFrame(run_rows), pd.DataFrame(trace_rows)


def validation_rows(
    *,
    device_info: Mapping[str, Any],
    cpu_gpu_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    target_df: pd.DataFrame,
    run_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    targets: Sequence[Any],
    policies: Sequence[Any],
    seeds: Sequence[int],
    movie_paths: Sequence[Path],
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

    expected_sweep_runs = len(targets) * len(policies) * len(seeds)
    expected_pairs = {(target.target_id, policy.policy_id) for target in targets for policy in policies}
    observed_pairs = set(zip(run_df["target_id"], run_df["policy_id"]))
    local_runs = run_df[run_df["is_local_only_policy"].astype(bool)]
    baseline_runs = run_df[run_df["is_global_information_baseline"].astype(bool)]
    metric_cols = ["initial_hamming_error", "final_hamming_error", "relative_error_reduction", "final_label_match_fraction", "final_edge_disagreement"]

    add("cuda_available", bool(device_info.get("cudaAvailable", False) and str(device_info.get("selectedDevice", "")).startswith("cuda")), device_info)
    add("cpu_gpu_validation_cases_passed", bool(len(cpu_gpu_df) > 0 and cpu_gpu_df["success"].astype(bool).all()), cpu_gpu_df[["target_id", "policy_id", "success", "max_abs_metric_delta"]].to_dict(orient="records"))
    add("seed_reproducibility_passed", bool(len(seed_df) == len(set(seed_df["seed"].astype(int))) and seed_df["success"].astype(bool).all()), seed_df[["seed", "success", "initial_hash_first", "final_hash_first"]].to_dict(orient="records"))
    add("target_catalog_written", len(target_df) == len(targets), target_df[["target_id", "motif", "width", "height", "target_hash"]].to_dict(orient="records"))
    add("policy_leakage_audits_passed", bool(policy_df["audit_success"].astype(bool).all()), policy_df[["policy_id", "audit_success", "audit_errors_json"]].to_dict(orient="records"))
    add("local_only_runs_no_hidden_target_map_leakage", bool((~local_runs["uses_hidden_target_map_leakage"].astype(bool)).all() and (~local_runs["uses_target_map"].astype(bool)).all()), "local-only GPU runs report no target-map access")
    add("global_target_baseline_explicitly_flagged", bool(len(baseline_runs) > 0 and baseline_runs["uses_target_map"].astype(bool).all() and baseline_runs["is_global_information_baseline"].astype(bool).all()), {"baselineRuns": len(baseline_runs)})
    add("expected_gpu_sweep_run_count", len(run_df) == expected_sweep_runs, {"observed": len(run_df), "expected": expected_sweep_runs})
    add("all_target_policy_pairs_covered", observed_pairs == expected_pairs, {"observedPairs": len(observed_pairs), "expectedPairs": len(expected_pairs)})
    add("trace_snapshots_written", len(trace_df) >= expected_sweep_runs * 2, {"traceRows": len(trace_df), "minimumExpected": expected_sweep_runs * 2})
    add("finite_metric_outputs", bool(np.isfinite(run_df[metric_cols].astype(float).to_numpy()).all()), "selected S11 metrics are finite")
    add("gpu_device_used_for_sweeps", bool(run_df["device"].map(lambda value: str(value).startswith("cuda")).all()), sorted(run_df["device"].unique()))
    add("occupancy_preserved", bool(run_df["occupancy_preserved"].astype(bool).all()), "S11 label dynamics preserves fixed occupied grid size")
    add("summary_rows_complete", len(summary_df) == len(targets) * len(policies), {"rows": len(summary_df), "expected": len(targets) * len(policies)})
    add("expanded_sweep_signal_present", bool(run_df["improved"].astype(bool).any()), {"improvedRuns": int(run_df["improved"].astype(bool).sum()), "runCount": len(run_df)})
    add("target_baseline_exact_recovery_observed", bool(baseline_runs["exact_match"].astype(bool).all()), {"baselineExactRuns": int(baseline_runs["exact_match"].astype(bool).sum()), "baselineRuns": len(baseline_runs)})
    add("movies_written", bool(len(movie_paths) >= 2 and all(path.exists() and path.stat().st_size > 0 for path in movie_paths)), [str(path) for path in movie_paths])
    add("repo_unit_tests_passed", bool(repo_test.get("success", False)), {"returncode": repo_test.get("returncode"), "args": repo_test.get("args")})
    return pd.DataFrame(rows)


def plot_error_reduction(summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    motifs = list(dict.fromkeys(summary_df["motif"].tolist()))
    policies = list(dict.fromkeys(summary_df["policy_id"].tolist()))
    x = np.arange(len(motifs))
    width = 0.24
    fig, axis = plt.subplots(figsize=(11, 5.2))
    colors = plt.get_cmap("tab10")
    for index, policy in enumerate(policies):
        values = []
        for motif in motifs:
            subset = summary_df[summary_df["motif"].eq(motif) & summary_df["policy_id"].eq(policy)]
            values.append(float(subset["relative_error_reduction_mean"].iloc[0]) if not subset.empty else 0.0)
        axis.bar(x + (index - (len(policies) - 1) / 2) * width, values, width=width, label=policy.replace("_", " "), color=colors(index), edgecolor="#222222", linewidth=0.4)
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(motifs)
    axis.set_ylabel("mean relative Hamming-error reduction")
    axis.set_title("S11 GPU batched sweep error reduction")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.legend(fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_validation_deltas(cpu_gpu_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{row.motif}\n{row.policy_id.replace('_gpu', '')}" for row in cpu_gpu_df.itertuples()]
    values = cpu_gpu_df["max_abs_metric_delta"].fillna(np.nan).astype(float).to_numpy()
    fig, axis = plt.subplots(figsize=(12, 4.8))
    axis.bar(np.arange(len(labels)), values, color="#5b8db8", edgecolor="#222222", linewidth=0.4)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("max absolute metric delta")
    axis.set_title("S11 CPU-vs-GPU validation deltas")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_label_gif(frames: Sequence[np.ndarray], *, class_count: int, path: Path, scale: int = 12) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    images: list[Image.Image] = []
    for frame in frames:
        rgb = labels_to_rgb(frame, class_count=class_count)
        image = Image.fromarray(rgb, mode="RGB").resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.Resampling.NEAREST)
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:], duration=180, loop=0)
    return path


def make_movies(targets: Sequence[Any], policies: Sequence[Any], *, seed: int, steps: int, device: str, frame_interval: int, figures_dir: Path) -> list[Path]:
    target = next(target for target in targets if target.motif == "ring")
    baseline = next(policy for policy in policies if policy.policy_id == "explicit_target_relaxation_gpu")
    local = next(policy for policy in policies if policy.policy_id == "local_neighbor_majority_gpu")
    baseline_frames = simulate_label_frame_sequence(target, baseline, seed=int(seed), steps=int(steps), device=device, frame_interval=int(frame_interval))
    local_frames = simulate_label_frame_sequence(target, local, seed=int(seed), steps=int(steps), device=device, frame_interval=int(frame_interval))
    return [
        write_label_gif(
            baseline_frames,
            class_count=target.class_count,
            path=figures_dir / "e05_s11_gpu_global_target_relaxation.gif",
        ),
        write_label_gif(
            local_frames,
            class_count=target.class_count,
            path=figures_dir / "e05_s11_gpu_local_neighbor_majority.gif",
        ),
    ]


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
    validation_df: pd.DataFrame,
    cpu_gpu_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> str:
    failures = validation_df[~validation_df["success"]]
    validation_text = "passed" if failures.empty else "failed"
    overall = summary_df.groupby(["policy_id", "policy_family", "is_local_only_policy", "is_global_information_baseline"], sort=True).agg(
        run_count=("run_count", "sum"),
        exact_match_rate=("exact_match_rate", "mean"),
        improved_rate=("improved_rate", "mean"),
        relative_error_reduction_mean=("relative_error_reduction_mean", "mean"),
        final_hamming_error_mean=("final_hamming_error_mean", "mean"),
    ).reset_index()
    return f"""# E05 S11 GPU Batched Simulation Report

Research step ID: {STEP_ID}
Completion status: {"completed" if failures.empty else "completed with validation failures"}
Artifact family: GPU batched square-grid label dynamics
Validation result: {validation_text}

S11 validates a deterministic torch tensor kernel against the CPU tensor reference before running expanded GPU batches. Local-only policies receive only actor and von Neumann neighbor labels. The explicit target-map relaxation policy is a flagged global-information baseline.

## Target Catalog

{markdown_table(["target", "motif", "width", "height", "classes", "hash"], target_df[["target_id", "motif", "width", "height", "class_count", "target_hash"]].values.tolist())}

## Policy Summary

{markdown_table(["policy", "family", "local only", "global baseline", "runs", "exact rate", "improved rate", "mean reduction", "final error"], overall[["policy_id", "policy_family", "is_local_only_policy", "is_global_information_baseline", "run_count", "exact_match_rate", "improved_rate", "relative_error_reduction_mean", "final_hamming_error_mean"]].values.tolist())}

## Target And Policy Summary

{markdown_table(["target", "motif", "policy", "runs", "exact", "improved", "mean reduction", "final match", "final edge disagreement"], summary_df[["target_id", "motif", "policy_id", "run_count", "exact_match_rate", "improved_rate", "relative_error_reduction_mean", "final_label_match_fraction_mean", "final_edge_disagreement_mean"]].values.tolist())}

## CPU-GPU Validation

{markdown_table(["target", "policy", "success", "max delta", "detail"], cpu_gpu_df[["target_id", "policy_id", "success", "max_abs_metric_delta", "detail"]].values.tolist())}

## Seed Reproducibility

{markdown_table(["seed", "success", "initial hash", "final hash"], seed_df[["seed", "success", "initial_hash_first", "final_hash_first"]].values.tolist())}

## Policy Leakage Audit

{markdown_table(["policy", "local only", "global baseline", "uses target map", "audit pass"], policy_df[["policy_id", "is_local_only_policy", "is_global_information_baseline", "uses_target_map", "audit_success"]].values.tolist())}

## Validation

{markdown_table(["check", "success", "detail"], validation_df[["check_id", "success", "detail"]].values.tolist())}

## Caveats

- S11 uses compact label-grid dynamics as a GPU batching validation layer, not the full CPU action-world semantics from S04.
- Local-only GPU policies are intentionally target-map-free and therefore mostly smooth labels rather than solve global morphology targets.
- The explicit target-map relaxation policy is a nonlocal comparator and should not be interpreted as local morphogenesis.
- GIF movies are compact qualitative traces for report review, while Parquet tables are the quantitative evidence.
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
    return f"""# E05 S11 Status Summary

- Research step ID: {STEP_ID}
- Completion status: {"completed" if success else "completed with validation failures"}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {outcome}
- Caveats or blockers:
{caveat_lines}
- Lay summary: S11 validated a torch GPU batch kernel against CPU tensor outputs, checked seed reproducibility and policy information boundaries, then ran expanded GPU sweeps over target motifs with local-only policies and an explicitly flagged target-map baseline. Two compact GIFs show a cued success and an uncued local-only trajectory.
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
        "schemaVersion": GPU_BATCH_SCHEMA_VERSION,
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

    device_info = torch_device_summary(args.device)
    device = str(device_info["selectedDevice"])
    targets = standard_gpu_sweep_targets()
    validation_targets = standard_gpu_validation_targets()
    policies = standard_gpu_policy_specs()
    target_df = pd.DataFrame(gpu_target_catalog_rows(targets))
    validation_target_df = pd.DataFrame(gpu_target_catalog_rows(validation_targets))
    policy_df = pd.DataFrame(gpu_policy_catalog_rows(policies))

    cpu_gpu_df = pd.DataFrame(
        cpu_gpu_validation_rows(
            validation_targets,
            policies,
            args.validation_seeds,
            steps=int(args.validation_steps),
            trace_interval=max(1, int(args.validation_steps) // 2),
            gpu_device=device,
        )
    )
    seed_policy = next(policy for policy in policies if policy.policy_id == "local_inertia_smoothing_gpu")
    seed_target = next(target for target in validation_targets if target.motif == "gradient")
    seed_df = pd.DataFrame(
        seed_reproducibility_rows(
            seed_target,
            seed_policy,
            args.validation_seeds,
            steps=int(args.validation_steps),
            trace_interval=max(1, int(args.validation_steps) // 2),
            device=device,
        )
    )

    run_df, trace_df = run_all_sweeps(
        targets,
        policies,
        args.sweep_seeds,
        steps=int(args.sweep_steps),
        trace_interval=int(args.trace_interval),
        device=device,
    )
    summary_df = pd.DataFrame(gpu_sweep_summary_rows(run_df.to_dict(orient="records")))
    movie_paths = make_movies(
        targets,
        policies,
        seed=int(args.sweep_seeds[0]),
        steps=int(args.sweep_steps),
        device=device,
        frame_interval=int(args.movie_interval),
        figures_dir=figures_dir,
    )
    repo_test = run_repo_tests(step_dir, skip=bool(args.skip_repo_tests))
    validation_df = validation_rows(
        device_info=device_info,
        cpu_gpu_df=cpu_gpu_df,
        seed_df=seed_df,
        policy_df=policy_df,
        target_df=target_df,
        run_df=run_df,
        trace_df=trace_df,
        summary_df=summary_df,
        targets=targets,
        policies=policies,
        seeds=args.sweep_seeds,
        movie_paths=movie_paths,
        repo_test=repo_test,
    )
    success = bool(validation_df["success"].all())

    baseline_runs = run_df[run_df["is_global_information_baseline"].astype(bool)]
    local_runs = run_df[run_df["is_local_only_policy"].astype(bool)]
    baseline_exact = int(baseline_runs["exact_match"].astype(bool).sum())
    local_improved = int(local_runs["improved"].astype(bool).sum())
    local_exact = int(local_runs["exact_match"].astype(bool).sum())
    validation_result = (
        f"passed: {len(cpu_gpu_df)} CPU-vs-GPU validation cases, {len(seed_df)} seed reproducibility checks, "
        f"{len(run_df)} GPU sweep runs, {baseline_exact}/{len(baseline_runs)} baseline exact recoveries, "
        f"{local_improved}/{len(local_runs)} local-only improved runs"
        if success
        else f"failed: {int((~validation_df['success']).sum())} validation checks failed"
    )
    outcome = "supportive" if success else "constraining/contradictory"
    recommended_next_action = "Chief Scientist review, then proceed to S12 morphospace trajectory mapping only after approval."
    caveats = [
        "S11 uses compact batched label dynamics rather than the full S04 action-world semantics.",
        "Local-only GPU policies are target-map-free and should be interpreted as smoothing or diffusion controls.",
        "The target-map relaxation policy is an explicitly flagged nonlocal baseline comparator.",
        "GIF movies are qualitative trace views; Parquet outputs are the quantitative evidence.",
    ]
    anchor_result = (
        f"GPU sweeps produced {len(run_df)} runs on {device}; "
        f"CPU-vs-GPU parity passed for {int(cpu_gpu_df['success'].astype(bool).sum())}/{len(cpu_gpu_df)} cases; "
        f"baseline exact recoveries = {baseline_exact}/{len(baseline_runs)}; "
        f"local-only exact recoveries = {local_exact}/{len(local_runs)}."
    )

    artifacts: list[Path] = []
    artifacts.extend(write_dataframe(run_df, step_dir / "gpu_sweep_run_results"))
    artifacts.extend(write_dataframe(run_df, results_dir / "e05_gpu_sweeps"))
    artifacts.extend(write_dataframe(trace_df, step_dir / "gpu_sweep_trace_examples"))
    artifacts.extend(write_dataframe(summary_df, step_dir / "gpu_sweep_summary"))
    artifacts.extend(write_dataframe(cpu_gpu_df, step_dir / "cpu_gpu_validation_results"))
    artifacts.extend(write_dataframe(seed_df, step_dir / "gpu_seed_reproducibility"))
    artifacts.extend(write_dataframe(policy_df, step_dir / "gpu_policy_leakage_audit"))
    artifacts.extend(write_dataframe(target_df, step_dir / "gpu_target_catalog"))
    artifacts.extend(write_dataframe(validation_target_df, step_dir / "gpu_validation_target_catalog"))
    artifacts.extend(write_dataframe(validation_df, step_dir / "gpu_validation_results"))

    config_path = configs_dir / "e05_gpu_sweeps_config.json"
    write_json(
        config_path,
        {
            "schemaVersion": GPU_BATCH_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "deviceInfo": device_info,
            "validationSeeds": [int(seed) for seed in args.validation_seeds],
            "sweepSeeds": [int(seed) for seed in args.sweep_seeds],
            "validationSteps": int(args.validation_steps),
            "sweepSteps": int(args.sweep_steps),
            "traceInterval": int(args.trace_interval),
            "movieInterval": int(args.movie_interval),
            "targets": target_df[["target_id", "motif", "width", "height", "class_count", "target_hash"]].to_dict(orient="records"),
            "policies": policy_df[["policy_id", "information_scope", "is_local_only_policy", "is_global_information_baseline", "uses_target_map"]].to_dict(orient="records"),
        },
    )
    artifacts.append(config_path)

    fig_reduction = figures_dir / "e05_s11_gpu_error_reduction.png"
    fig_validation = figures_dir / "e05_s11_cpu_gpu_validation_deltas.png"
    plot_error_reduction(summary_df, fig_reduction)
    plot_validation_deltas(cpu_gpu_df, fig_validation)
    artifacts.extend([fig_reduction, fig_validation])
    artifacts.extend(movie_paths)

    report_path = step_dir / "gpu_batched_simulation_report.md"
    report_path.write_text(report_markdown(summary_df, validation_df, cpu_gpu_df, seed_df, policy_df, target_df), encoding="utf-8")
    artifacts.append(report_path)

    artifacts.append(Path(repo_test["logPath"]))
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "deviceInfo": device_info,
        "parallelism": "single_cuda_device_batched_torch_tensors",
        "benchmarkRuntimeSeconds": time.perf_counter() - started,
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
            "cpuGpuValidationCaseCount": len(cpu_gpu_df),
            "cpuGpuValidationPassCount": int(cpu_gpu_df["success"].astype(bool).sum()) if len(cpu_gpu_df) else 0,
            "seedReproducibilityCheckCount": len(seed_df),
            "seedReproducibilityPassCount": int(seed_df["success"].astype(bool).sum()) if len(seed_df) else 0,
            "gpuSweepRunCount": len(run_df),
            "targetCount": len(targets),
            "policyCount": len(policies),
            "seedCount": len(args.sweep_seeds),
            "traceRowCount": len(trace_df),
            "localOnlyRunCount": len(local_runs),
            "localOnlyImprovedCount": local_improved,
            "localOnlyExactMatchCount": local_exact,
            "globalBaselineRunCount": len(baseline_runs),
            "globalBaselineExactMatchCount": baseline_exact,
            "sweepSteps": int(args.sweep_steps),
            "traceInterval": int(args.trace_interval),
            "movieCount": len(movie_paths),
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
        "schemaVersion": GPU_BATCH_SCHEMA_VERSION,
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
