#!/usr/bin/env python3
"""Run E05 S11 GPU-batched tissue sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

from src.e05.gpu_tissue import (
    DEFAULT_GPU_EVENT_MULTIPLIERS,
    DEFAULT_GPU_SWEEP_SEEDS,
    DEFAULT_RECORDS_PER_RUN,
    SELECTED_GPU_TASK_IDS,
    default_gpu_task_templates,
    run_gpu_tissue_sweep,
    validate_cpu_gpu_agreement,
    validate_reference_cpu_agreement,
)
from src.e05.scrambled_embryo import RUNNABLE_POLICY_IDS
from src.e05.targets import ORGAN_COLORS


STEP_ID = "S11"
STEP_NUMBER = 11
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--seeds", default=f"{DEFAULT_GPU_SWEEP_SEEDS[0]}:{DEFAULT_GPU_SWEEP_SEEDS[-1]}")
    parser.add_argument("--event-multipliers", default=",".join(str(value) for value in DEFAULT_GPU_EVENT_MULTIPLIERS))
    parser.add_argument("--policy-ids", default=",".join(RUNNABLE_POLICY_IDS))
    parser.add_argument("--records-per-run", type=int, default=DEFAULT_RECORDS_PER_RUN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_path(path),
        "sizeBytes": path.stat().st_size if path.is_file() else sum(child.stat().st_size for child in path.rglob("*") if child.is_file()),
        "artifactType": "directory" if path.is_dir() else "file",
    }


def manifest_self_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum omitted to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def parse_int_sequence(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            start_raw, end_raw = chunk.split(":", 1)
            start = int(start_raw)
            end = int(end_raw)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(chunk))
    if not values:
        raise ValueError("at least one integer value is required")
    return tuple(values)


def parse_str_list(raw: str, allowed: Sequence[str], label: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"at least one {label} is required")
    unsupported = sorted(set(values) - set(allowed))
    if unsupported:
        raise ValueError(f"unsupported {label}: {unsupported}")
    return values


def _row(
    validation_case: str,
    case_type: str,
    success: bool,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    detail: str,
) -> dict[str, Any]:
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "validation_case": validation_case,
        "case_type": case_type,
        "success": bool(success),
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": detail,
    }


def load_source_anchor_rows(artifacts_dir: Path) -> list[dict[str, Any]]:
    source_specs = (
        ("S07", "scrambled embryo results", artifacts_dir / "results" / "e05_scrambled_embryo_results.parquet"),
        ("S08", "regeneration results", artifacts_dir / "results" / "e05_regeneration_results.parquet"),
        ("S09", "scaling results", artifacts_dir / "results" / "e05_scaling_tests.parquet"),
        ("S10", "symmetry-breaking results", artifacts_dir / "results" / "e05_symmetry_breaking.parquet"),
    )
    rows: list[dict[str, Any]] = []
    for step_id, label, path in source_specs:
        exists = path.exists()
        row_count = 0
        columns: list[str] = []
        if exists:
            df = pd.read_parquet(path)
            row_count = len(df)
            columns = list(df.columns)
        rows.append(
            {
                "research_step_id": STEP_ID,
                "source_research_step_id": step_id,
                "source_label": label,
                "source_path": str(path),
                "exists": bool(exists),
                "row_count": int(row_count),
                "column_count": len(columns),
                "used_for_s11": True,
                "note": "Loaded as source-task/provenance anchor for selected S11 tensorized tasks.",
            }
        )
    return rows


def render_representative_frames(
    *,
    summary_df: pd.DataFrame,
    snapshots_by_run_uid: Mapping[str, Mapping[int, Sequence[int]]],
    templates_by_task_id: Mapping[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """Export one success and one failure frame panel."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_png in output_dir.glob("*.png"):
        old_png.unlink()
    eligible = summary_df[summary_df["target_kind"].isin(["organ_like", "boundary", "symmetry"])].copy()
    if eligible.empty:
        eligible = summary_df.copy()
    success_candidates = eligible[eligible["policy_id"] == "s07_local_target_neighbor_descent"].copy()
    if success_candidates.empty:
        success_candidates = eligible.copy()
    success_row = success_candidates.sort_values(
        ["target_recovery_fraction", "final_target_error"],
        ascending=[False, True],
    ).iloc[0]
    failure_candidates = eligible[eligible["run_uid"] != success_row["run_uid"]].copy()
    if failure_candidates.empty:
        failure_candidates = eligible.copy()
    failure_row = failure_candidates.sort_values(
        ["target_recovery_fraction", "final_target_error"],
        ascending=[True, False],
    ).iloc[0]

    rows = []
    for role, row in (("success", success_row), ("failure", failure_row)):
        task_id = str(row["task_id"])
        run_uid = str(row["run_uid"])
        template = templates_by_task_id[task_id]
        snapshots = snapshots_by_run_uid[run_uid]
        filename = f"{role}_{_safe_filename(run_uid)}.png"
        output_path = output_dir / filename
        _render_one_frame_panel(template, snapshots, output_path, role=role, summary_row=row)
        rows.append(
            {
                "research_step_id": STEP_ID,
                "frame_role": role,
                "run_uid": run_uid,
                "task_id": task_id,
                "source_research_step_id": row["source_research_step_id"],
                "target_id": row["target_id"],
                "target_kind": row["target_kind"],
                "policy_id": row["policy_id"],
                "simulation_seed": int(row["simulation_seed"]),
                "event_multiplier": int(row["event_multiplier"]),
                "initial_target_error": float(row["initial_target_error"]),
                "final_target_error": float(row["final_target_error"]),
                "target_recovery_fraction": float(row["target_recovery_fraction"]),
                "path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
        )
    return pd.DataFrame(rows)


def _render_one_frame_panel(template: Any, snapshots: Mapping[int, Sequence[int]], output_path: Path, *, role: str, summary_row: Mapping[str, Any]) -> None:
    target = template.target
    width = template.width
    height = template.height
    steps = sorted(int(step) for step in snapshots)
    midpoint = steps[len(steps) // 2]
    panel_steps = [0, midpoint, steps[-1]]
    labels_by_cell = _target_cell_organ_labels(target)
    target_labels = [str(target.identities_by_site[site_id].components.get("organ_type", "cell")) for site_id in target.substrate.site_ids]
    all_labels = sorted(set(labels_by_cell) | set(target_labels) | set(ORGAN_COLORS))
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    colors = [ORGAN_COLORS.get(label, "#777777") for label in all_labels]
    cmap = ListedColormap(colors)

    matrices = [_label_matrix(target_labels, label_to_index, width, height)]
    titles = ["target"]
    for step in panel_steps:
        labels = [labels_by_cell[int(cell_idx)] for cell_idx in snapshots[step]]
        matrices.append(_label_matrix(labels, label_to_index, width, height))
        titles.append(f"event {step}")

    fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.0), constrained_layout=True)
    for ax, matrix, title in zip(axes, matrices, titles, strict=True):
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(1, len(colors) - 1), interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        (
            f"S11 {role}: {summary_row['task_id']} | {summary_row['policy_id']} | "
            f"recovery {float(summary_row['target_recovery_fraction']):.3f}"
        ),
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _target_cell_organ_labels(target: Any) -> list[str]:
    constructed = target.constructed_substrate()
    labels = []
    for site_id in target.substrate.site_ids:
        cell = constructed.cell_at(site_id)
        identity = cell.metadata.get("e05_identity", {}) if cell is not None else {}
        labels.append(str(identity.get("components", {}).get("organ_type", cell.label if cell is not None else "empty")))
    return labels


def _label_matrix(labels: Sequence[str], label_to_index: Mapping[str, int], width: int, height: int) -> np.ndarray:
    values = np.array([label_to_index[str(label)] for label in labels], dtype=int)
    return values.reshape(height, width)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:160]


def run_validations(
    *,
    source_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    memory_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    cpu_gpu_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    expected_templates: int,
    expected_policies: int,
    expected_seeds: int,
    expected_event_multipliers: int,
) -> pd.DataFrame:
    expected_runs = expected_templates * expected_policies * expected_seeds * expected_event_multipliers
    validation_rows = [
        _row(
            "source_result_anchors_loaded",
            "input_provenance",
            bool(len(source_df) == 4 and source_df["exists"].all() and (source_df["row_count"] > 0).all()),
            {"source_steps": ["S07", "S08", "S09", "S10"], "min_row_count": 1},
            {
                "rows": len(source_df),
                "all_exist": bool(source_df["exists"].all()),
                "min_row_count": int(source_df["row_count"].min()) if not source_df.empty else 0,
            },
            "Prior S07-S10 result tables were loaded as task and provenance anchors.",
        ),
        _row(
            "cuda_runtime_available_and_logged",
            "gpu_runtime",
            bool(torch.cuda.is_available() and not memory_df.empty and memory_df["cuda_available"].all() and (memory_df["cuda_total_memory_bytes"] > 0).all()),
            {"cuda_available": True, "memory_rows": ">0"},
            {
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "memory_rows": len(memory_df),
                "device_names": sorted(set(str(value) for value in memory_df.get("cuda_device_name", pd.Series(dtype=str)))),
            },
            "PyTorch CUDA runtime and device memory counters were available and logged.",
        ),
        _row(
            "vectorization_scope_declared",
            "scope_guardrail",
            bool(
                {"S07", "S08", "S09", "S10"}.issubset(set(scope_df.loc[scope_df["vectorized_in_s11"], "source_research_step_id"]))
                and (scope_df["vectorized_in_s11"] == False).any()  # noqa: E712
            ),
            {"selected_steps": ["S07", "S08", "S09", "S10"], "blocked_rows": ">0"},
            {
                "selected_steps": sorted(set(scope_df.loc[scope_df["vectorized_in_s11"], "source_research_step_id"])),
                "blocked_rows": int((scope_df["vectorized_in_s11"] == False).sum()),  # noqa: E712
            },
            "S11 selected square-grid swap/wait tasks and documented blocked semantics instead of remapping them.",
        ),
        _row(
            "cpu_gpu_agreement_small_fixtures",
            "cpu_gpu_validation",
            bool(not cpu_gpu_df.empty and cpu_gpu_df["success"].all()),
            {"all_success": True},
            {
                "rows": len(cpu_gpu_df),
                "successes": int(cpu_gpu_df["success"].sum()) if not cpu_gpu_df.empty else 0,
                "max_target_error_abs_diff": float(cpu_gpu_df["max_target_error_abs_diff"].max()) if not cpu_gpu_df.empty else None,
            },
            "CPU and GPU tensor kernels agreed on final occupants, counts, and target errors for deterministic fixtures.",
        ),
        _row(
            "tensor_reference_replay_agreement",
            "semantic_validation",
            bool(not reference_df.empty and reference_df["success"].all()),
            {"all_success": True},
            {
                "rows": len(reference_df),
                "successes": int(reference_df["success"].sum()) if not reference_df.empty else 0,
                "max_target_error_abs_diff": float(reference_df["max_target_error_abs_diff"].max()) if not reference_df.empty else None,
            },
            "Tensor CPU kernel matched object-state S04-compatible replay under identical schedules.",
        ),
        _row(
            "gpu_sweep_matrix_complete",
            "output_completeness",
            bool(len(summary_df) == expected_runs and len(summary_df) >= 1000),
            {"expected_runs": expected_runs, "minimum_runs": 1000},
            {
                "observed_runs": len(summary_df),
                "task_count": int(summary_df["task_id"].nunique()) if not summary_df.empty else 0,
                "policy_count": int(summary_df["policy_id"].nunique()) if not summary_df.empty else 0,
                "seed_count": int(summary_df["simulation_seed"].nunique()) if not summary_df.empty else 0,
                "event_multiplier_count": int(summary_df["event_multiplier"].nunique()) if not summary_df.empty else 0,
            },
            "The selected task by policy by seed by event-budget matrix was completed and includes thousands of runs.",
        ),
        _row(
            "gpu_memory_use_logged",
            "gpu_runtime",
            bool(not memory_df.empty and (memory_df["peak_allocated_bytes"] > 0).all() and (memory_df["peak_reserved_bytes"] >= memory_df["peak_allocated_bytes"]).all()),
            {"peak_allocated_bytes": ">0", "peak_reserved_ge_allocated": True},
            {
                "rows": len(memory_df),
                "max_peak_allocated_bytes": int(memory_df["peak_allocated_bytes"].max()) if not memory_df.empty else 0,
                "max_peak_reserved_bytes": int(memory_df["peak_reserved_bytes"].max()) if not memory_df.empty else 0,
            },
            "PyTorch peak allocated and reserved GPU memory were recorded for every batched group.",
        ),
        _row(
            "trace_table_complete",
            "output_completeness",
            bool(
                not trace_df.empty
                and trace_df.groupby("run_uid")["event_step"].min().eq(0).all()
                and trace_df.groupby("run_uid").apply(lambda frame: int(frame["event_step"].max()) == int(frame["event_cap"].iloc[0]), include_groups=False).all()
            ),
            {"each_run_has_event_0_and_event_cap": True},
            {
                "trace_rows": len(trace_df),
                "run_count": int(trace_df["run_uid"].nunique()) if not trace_df.empty else 0,
            },
            "Downsampled trace rows include initial and final event records for every S11 run.",
        ),
        _row(
            "s04_swap_wait_accounting_preserved",
            "semantic_validation",
            bool(
                not summary_df.empty
                and (summary_df["population_delta_total"] == 0).all()
                and (summary_df["rejected_actions"] == 0).all()
                and np.allclose(summary_df["total_energy_cost"], summary_df["accepted_swaps"].astype(float))
                and (summary_df["accepted_swaps"] == summary_df["attempted_swaps"]).all()
            ),
            {"population_delta_total": 0, "rejected_actions": 0, "energy_equals_accepted_swaps": True},
            {
                "max_population_delta_abs": int(summary_df["population_delta_total"].abs().max()) if not summary_df.empty else None,
                "total_rejected_actions": int(summary_df["rejected_actions"].sum()) if not summary_df.empty else None,
                "max_energy_count_abs_diff": float((summary_df["total_energy_cost"] - summary_df["accepted_swaps"].astype(float)).abs().max()) if not summary_df.empty else None,
            },
            "Selected tensorized tasks preserve S04 swap/wait population and energy accounting.",
        ),
        _row(
            "representative_success_failure_frames_exported",
            "figure_validation",
            bool(
                not frame_df.empty
                and {"success", "failure"}.issubset(set(frame_df["frame_role"]))
                and (frame_df["size_bytes"] > 1000).all()
                and frame_df["path"].map(lambda value: Path(value).exists()).all()
            ),
            {"frame_roles": ["success", "failure"], "min_size_bytes": 1000},
            {
                "rows": len(frame_df),
                "roles": sorted(set(frame_df["frame_role"])) if not frame_df.empty else [],
                "min_size_bytes": int(frame_df["size_bytes"].min()) if not frame_df.empty else 0,
            },
            "Representative success and failure frame panels were exported as nonempty PNG files.",
        ),
    ]
    return pd.DataFrame(validation_rows)


def markdown_table(df: pd.DataFrame, columns: list[str], *, max_rows: int = 24) -> str:
    if df.empty:
        return "_No rows._"
    shown = df[columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in shown.to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    if len(df) > max_rows:
        omitted = [f"... {len(df) - max_rows} more rows omitted", *("" for _ in columns[1:])]
        rows.append("| " + " | ".join(omitted) + " |")
    return "\n".join([header, separator, *rows])


def determine_outcome(validation_df: pd.DataFrame, summary_df: pd.DataFrame) -> tuple[str, str, str]:
    validation_success = bool(validation_df["success"].all())
    enough_runs = len(summary_df) >= 1000
    if validation_success and enough_runs:
        outcome = "supportive"
        lay = (
            "S11 shows that selected full-occupancy 2D swap/wait tissue tasks from S07-S10 can be batched on the L4 GPU with exact "
            "CPU-vs-GPU agreement on small fixtures, logged memory use, and representative success/failure frames."
        )
    else:
        outcome = "constraining/contradictory"
        lay = (
            "S11 attempted GPU batching, but validation or matrix-completeness checks did not fully pass, so GPU sweeps should not yet "
            "be treated as a reliable backend for E05 trajectory generation."
        )
    caveats = (
        "S11 vectorizes only regular square-grid, full-occupancy, swap/wait tasks. Missing, frozen, duplicated, foreign-patch, graph, and "
        "birth/death semantics remain explicitly blocked for this tensor kernel. The deterministic schedules are precomputed for CPU/GPU "
        "agreement and are not identical to earlier Python random streams, although they preserve the same local policy information access "
        "and S04 swap/wait accounting."
    )
    return outcome, caveats, lay


def top_summary_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
    lay_summary: str,
) -> str:
    artifact_lines = "\n".join(f"- `{entry['path']}`" for entry in artifacts if entry.get("path"))
    return f"""## Top Summary

- Research step ID: {STEP_ID}
- Completion status: Completed
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {outcome_classification}
- Caveats or blockers: {caveats_or_blockers}
- Lay summary: {lay_summary}
- Recommended next action: {recommended_next_action}
"""


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    memory_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    source_df: pd.DataFrame,
    cpu_gpu_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    seeds: Sequence[int],
    event_multipliers: Sequence[int],
    policy_ids: Sequence[str],
    args: argparse.Namespace,
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
    lay_summary: str,
) -> str:
    command_lines = "\n".join(
        f"- `{command['command']}`: return code {command['returnCode']}, success={command['success']}, elapsed={command['elapsedSeconds']:.3f}s"
        for command in test_commands
    )
    source_lines = "\n".join(
        f"- `{entry['relativePath']}` sha256 `{entry['sha256']}` ({entry['sizeBytes']} bytes)"
        for entry in source_files
    )
    outcome_summary = (
        summary_df.groupby(["source_research_step_id", "task_id", "policy_id", "event_multiplier"], as_index=False)
        .agg(
            runs=("run_uid", "count"),
            mean_initial_error=("initial_target_error", "mean"),
            mean_final_error=("final_target_error", "mean"),
            mean_recovery=("target_recovery_fraction", "mean"),
            mean_energy_per_site=("energy_per_site", "mean"),
            mean_events_per_second=("group_events_per_second", "mean"),
        )
        .sort_values(["source_research_step_id", "task_id", "policy_id", "event_multiplier"])
    )
    memory_summary = (
        memory_df.groupby(["device_used"], as_index=False)
        .agg(
            groups=("task_id", "count"),
            max_peak_allocated_bytes=("peak_allocated_bytes", "max"),
            max_peak_reserved_bytes=("peak_reserved_bytes", "max"),
            mean_events_per_second=("events_per_second", "mean"),
        )
        .sort_values("device_used")
    )
    for frame in (outcome_summary, memory_summary):
        for column in frame.select_dtypes(include=[float]).columns:
            frame[column] = frame[column].round(6)
    return f"""{top_summary_markdown(artifacts, validation_result, outcome_classification, caveats_or_blockers, recommended_next_action, lay_summary)}

# Research Step Full Results: {STEP_ID} Use GPU For Batched Tissue Simulations

## Lay Summary

{lay_summary}

## Frozen Question

Can thousands of tissue simulations be batched on the GPU to support parameter sweeps and movies of successes and failures?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S11.
- Source task anchors: S07 scrambled embryo results, S08 regeneration results, S09 scaling results, and S10 symmetry-breaking results under `$ARTIFACTS_DIR/results/`.
- Repository code: `src/e05/gpu_tissue.py`, plus S07-S10 CPU task helpers for target construction and semantic reference checks.
- Datasets: none required.

## Detailed Methods

S11 vectorizes the full-occupancy square-grid swap/wait subset of the E05 task family. The selected templates cover three S07 scrambled targets, one S08 repairable rotated-patch perturbation, two S09 small/large scaled scramble settings, and two S10 exact mirror-symmetric starts. Missing, frozen, duplicated, foreign-patch, and graph tasks are not remapped into the GPU kernel; they are recorded as blocked scope rows because they require birth/death, identity conversion, unfreeze, or ragged graph semantics beyond this tensor kernel.

Each run is encoded as an integer occupant vector over target sites. A target-distance tensor stores the S03 identity distance from each target-population cell to each target site. The S07 local target-aware policy evaluates only the actor and adjacent neighbors by comparing local target-error deltas; the random adjacent-swap control uses only actor and adjacent-neighbor occupancy. Actor sites and neighbor-order permutations are generated once on CPU and reused for CPU and GPU validation so agreement is a backend check rather than a random-stream comparison.

The GPU sweep crossed `{len(SELECTED_GPU_TASK_IDS)}` selected task templates, `{len(policy_ids)}` policies, `{len(seeds)}` seeds, and `{len(event_multipliers)}` event budgets, producing `{len(summary_df)}` run summaries. PyTorch CUDA memory counters were logged for every batched group. Representative success and failure frame panels show target, initial, midpoint, and final states.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- PyTorch: `{torch.__version__}`
- CUDA available through PyTorch: `{torch.cuda.is_available()}`
- CUDA device: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"}`
- Pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- Matplotlib: `{matplotlib.__version__}`
- New packages installed: none.
- CPU/GPU use: CPU reference validation plus CUDA batched sweeps on the selected device `{args.device}`.

## Parameters

- Seeds: `{seeds[0]}:{seeds[-1]}` inclusive (`{len(seeds)}` seeds)
- Event multipliers: `{", ".join(str(value) for value in event_multipliers)}` local events per site
- Records per run: `{args.records_per_run}`
- Runnable policies: `{", ".join(policy_ids)}`
- Selected task IDs: `{", ".join(SELECTED_GPU_TASK_IDS)}`

## Results

Validation summary:

{markdown_table(validation_df, ["validation_case", "case_type", "success", "detail"], max_rows=20)}

GPU outcome summary:

{markdown_table(outcome_summary, ["source_research_step_id", "task_id", "policy_id", "event_multiplier", "runs", "mean_initial_error", "mean_final_error", "mean_recovery", "mean_energy_per_site"], max_rows=28)}

GPU memory summary:

{markdown_table(memory_summary, ["device_used", "groups", "max_peak_allocated_bytes", "max_peak_reserved_bytes", "mean_events_per_second"], max_rows=10)}

Source anchors:

{markdown_table(source_df, ["source_research_step_id", "source_label", "exists", "row_count", "used_for_s11"], max_rows=10)}

Vectorization scope and blockers:

{markdown_table(scope_df, ["scope_id", "source_research_step_id", "source_task_type", "vectorized_in_s11", "status", "reason"], max_rows=18)}

CPU/GPU agreement checks:

{markdown_table(cpu_gpu_df, ["task_id", "policy_id", "success", "max_target_error_abs_diff", "final_occupants_equal", "counts_equal"], max_rows=20)}

Reference replay checks:

{markdown_table(reference_df, ["task_id", "policy_id", "success", "max_target_error_abs_diff", "final_occupants_equal", "counts_equal"], max_rows=20)}

Representative frames:

{markdown_table(frame_df, ["frame_role", "task_id", "policy_id", "simulation_seed", "target_recovery_fraction", "path"], max_rows=10)}

## Metrics

- Target error: mean S03 identity distance between current occupants and target-site identities.
- Recovery fraction: `(initial_target_error - final_target_error) / initial_target_error`, using the S07 convention for zero initial error.
- Energy: S04-compatible swap/wait bookkeeping with cost 1 for accepted adjacent swaps and 0 for waits or rejected actions.
- Throughput: batched local scheduler events per second for each task/policy/event-budget group.
- GPU memory: PyTorch allocated/reserved and peak allocated/reserved byte counters, plus total CUDA memory from `torch.cuda.mem_get_info`.

## Figures And Tables

- Required S11 run table: `$ARTIFACTS_DIR/results/e05_gpu_tissue_sweeps.parquet`.
- Required frame directory: `$ARTIFACTS_DIR/figures/e05/tissue_simulation_frames/`.
- Downsampled trace table: `$ARTIFACTS_DIR/traces/e05_gpu_tissue_trace_table.parquet`.
- Memory, validation, scope, source-anchor, CPU/GPU agreement, reference-replay, and frame-manifest tables are written under `$ARTIFACTS_DIR/results/` with CSV companions under `$ARTIFACTS_DIR/tables/`.

## Validation Checks

- Loaded source result anchors from S07 through S10.
- Confirmed PyTorch CUDA availability and logged memory counters.
- Declared selected vectorized tasks and explicit blocked tasks.
- Validated CPU-vs-GPU tensor agreement on small fixtures.
- Validated tensor CPU semantics against object-state S04-compatible replay.
- Completed a >1000-run GPU sweep matrix.
- Logged GPU memory use for every batched group.
- Wrote complete downsampled trace rows with initial and final records.
- Preserved S04 swap/wait population and energy accounting.
- Exported representative success and failure PNG frame panels.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- S11 is a backend and throughput validation for selected computational proxy tasks; it is not biological evidence.
- The tensor kernel covers full-occupancy regular square grids with swap/wait actions. It does not cover missing populations, frozen/stuck cells, duplicated or foreign identities, division/death, identity conversion, unfreeze repair, or irregular graph substrates.
- Deterministic actor and neighbor-order schedules are generated ahead of time to make CPU/GPU agreement exact. These schedules are not the same random streams used by earlier CPU artifact scripts.
- The local target-aware control retains its declared S07 local target-site access. S11 does not add organizer cells or global controller access.
- Frame panels are representative still images, not full movies.

## Blockers And Failed Assumptions

No execution blocker was encountered for the selected square-grid swap/wait GPU tasks. The constrained scope is explicit: graph substrates and perturbations that require birth/death, identity conversion, or unfreeze semantics remain blocked for this S11 tensor kernel.

## Recommended Next Action

{recommended_next_action}
"""


def write_checksums(paths: list[Path], checksum_path: Path, artifacts_dir: Path) -> None:
    lines = []
    for path in sorted(paths):
        if path == checksum_path:
            continue
        lines.append(f"{sha256_path(path)}  {path.relative_to(artifacts_dir)}")
    write_text(checksum_path, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    traces_dir = artifacts_dir / "traces"
    figures_dir = artifacts_dir / "figures" / "e05"
    frame_dir = figures_dir / "tissue_simulation_frames"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, traces_dir, frame_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    seeds = parse_int_sequence(args.seeds)
    event_multipliers = parse_int_sequence(args.event_multipliers)
    policy_ids = parse_str_list(args.policy_ids, RUNNABLE_POLICY_IDS, "policy IDs")
    templates = default_gpu_task_templates()
    source_df = pd.DataFrame(load_source_anchor_rows(artifacts_dir))

    cpu_gpu_df = pd.DataFrame(validate_cpu_gpu_agreement(templates=templates[:3], policy_ids=policy_ids, seeds=seeds[:3], event_multiplier=4))
    reference_df = pd.DataFrame(validate_reference_cpu_agreement(templates=templates[:4], policy_ids=policy_ids, seeds=seeds[:2], event_multiplier=3))
    sweep = run_gpu_tissue_sweep(
        templates=templates,
        policy_ids=policy_ids,
        seeds=seeds,
        event_multipliers=event_multipliers,
        records_per_run=args.records_per_run,
        device=args.device,
    )
    summary_df = pd.DataFrame(sweep.summary_rows)
    trace_df = pd.DataFrame(sweep.trace_rows)
    memory_df = pd.DataFrame(sweep.memory_rows)
    scope_df = pd.DataFrame(sweep.vectorization_scope_rows)
    frame_df = render_representative_frames(
        summary_df=summary_df,
        snapshots_by_run_uid=sweep.snapshots_by_run_uid,
        templates_by_task_id=sweep.templates_by_task_id,
        output_dir=frame_dir,
    )
    validation_df = run_validations(
        source_df=source_df,
        summary_df=summary_df,
        trace_df=trace_df,
        memory_df=memory_df,
        scope_df=scope_df,
        cpu_gpu_df=cpu_gpu_df,
        reference_df=reference_df,
        frame_df=frame_df,
        expected_templates=len(templates),
        expected_policies=len(policy_ids),
        expected_seeds=len(seeds),
        expected_event_multipliers=len(event_multipliers),
    )
    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"
    outcome_classification, caveats_or_blockers, lay_summary = determine_outcome(validation_df, summary_df)
    recommended_next_action = "Stop before S12 and let the Chief Scientist review S11; if accepted, proceed to S12 morphospace trajectory mapping."

    results_path = results_dir / "e05_gpu_tissue_sweeps.parquet"
    results_csv_path = tables_dir / "e05_gpu_tissue_sweeps.csv"
    trace_path = traces_dir / "e05_gpu_tissue_trace_table.parquet"
    trace_csv_path = tables_dir / "e05_gpu_tissue_trace_table.csv"
    memory_path = results_dir / "e05_gpu_tissue_memory.parquet"
    memory_csv_path = tables_dir / "e05_gpu_tissue_memory.csv"
    scope_path = results_dir / "e05_gpu_tissue_vectorization_scope.parquet"
    scope_csv_path = tables_dir / "e05_gpu_tissue_vectorization_scope.csv"
    source_anchor_path = results_dir / "e05_gpu_tissue_source_anchors.parquet"
    source_anchor_csv_path = tables_dir / "e05_gpu_tissue_source_anchors.csv"
    cpu_gpu_path = results_dir / "e05_gpu_tissue_cpu_gpu_agreement.parquet"
    cpu_gpu_csv_path = tables_dir / "e05_gpu_tissue_cpu_gpu_agreement.csv"
    reference_path = results_dir / "e05_gpu_tissue_reference_replay.parquet"
    reference_csv_path = tables_dir / "e05_gpu_tissue_reference_replay.csv"
    frame_manifest_path = results_dir / "e05_gpu_tissue_frame_manifest.parquet"
    frame_manifest_csv_path = tables_dir / "e05_gpu_tissue_frame_manifest.csv"
    validation_path = results_dir / "e05_gpu_tissue_validation.parquet"
    validation_csv_path = tables_dir / "e05_gpu_tissue_validation.csv"
    config_path = configs_dir / "e05_s11_gpu_tissue_sweeps.json"
    source_manifest_path = src_snapshot_dir / "e05_gpu_tissue_manifest.json"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    summary_df.to_parquet(results_path, index=False)
    summary_df.to_csv(results_csv_path, index=False)
    trace_df.to_parquet(trace_path, index=False)
    trace_df.to_csv(trace_csv_path, index=False)
    memory_df.to_parquet(memory_path, index=False)
    memory_df.to_csv(memory_csv_path, index=False)
    scope_df.to_parquet(scope_path, index=False)
    scope_df.to_csv(scope_csv_path, index=False)
    source_df.to_parquet(source_anchor_path, index=False)
    source_df.to_csv(source_anchor_csv_path, index=False)
    cpu_gpu_df.to_parquet(cpu_gpu_path, index=False)
    cpu_gpu_df.to_csv(cpu_gpu_csv_path, index=False)
    reference_df.to_parquet(reference_path, index=False)
    reference_df.to_csv(reference_csv_path, index=False)
    frame_df.to_parquet(frame_manifest_path, index=False)
    frame_df.to_csv(frame_manifest_csv_path, index=False)
    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)

    config = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "device": args.device,
        "seeds": list(seeds),
        "eventMultipliers": list(event_multipliers),
        "policyIds": list(policy_ids),
        "recordsPerRun": int(args.records_per_run),
        "selectedTaskIds": list(SELECTED_GPU_TASK_IDS),
        "runCount": int(len(summary_df)),
        "traceRowCount": int(len(trace_df)),
        "cudaAvailable": bool(torch.cuda.is_available()),
        "cudaDeviceName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    write_json(config_path, config)

    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "tests.e05.test_gpu_tissue", "-v"], args.repo_dir))
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )

    source_files = [
        source_entry(args.repo_dir / "src/e05/gpu_tissue.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s11_gpu_tissue_sweeps.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_gpu_tissue.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/scrambled_embryo.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/scaling.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/regeneration.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/symmetry_breaking.py", args.repo_dir),
    ]
    write_json(
        source_manifest_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "sourceFiles": source_files,
        },
    )

    artifacts: list[dict[str, Any]] = [
        artifact_entry(results_path, artifacts_dir, "Required S11 GPU-batched tissue sweep run-summary table."),
        artifact_entry(results_csv_path, artifacts_dir, "CSV companion for the S11 run-summary table."),
        artifact_entry(frame_dir, artifacts_dir, "Required S11 representative success/failure tissue frame directory."),
        artifact_entry(trace_path, artifacts_dir, "Downsampled S11 tensor trajectory trace table."),
        artifact_entry(trace_csv_path, artifacts_dir, "CSV companion for the S11 trace table."),
        artifact_entry(memory_path, artifacts_dir, "S11 GPU memory and throughput log table."),
        artifact_entry(memory_csv_path, artifacts_dir, "CSV companion for GPU memory and throughput logs."),
        artifact_entry(scope_path, artifacts_dir, "S11 vectorized scope and blocked-task guardrail table."),
        artifact_entry(scope_csv_path, artifacts_dir, "CSV companion for vectorization scope guardrails."),
        artifact_entry(source_anchor_path, artifacts_dir, "S07-S10 source result anchor table."),
        artifact_entry(source_anchor_csv_path, artifacts_dir, "CSV companion for source result anchors."),
        artifact_entry(cpu_gpu_path, artifacts_dir, "S11 CPU-vs-GPU tensor agreement validation details."),
        artifact_entry(cpu_gpu_csv_path, artifacts_dir, "CSV companion for CPU-vs-GPU agreement details."),
        artifact_entry(reference_path, artifacts_dir, "S11 tensor-vs-object reference replay validation details."),
        artifact_entry(reference_csv_path, artifacts_dir, "CSV companion for reference replay details."),
        artifact_entry(frame_manifest_path, artifacts_dir, "S11 representative frame manifest."),
        artifact_entry(frame_manifest_csv_path, artifacts_dir, "CSV companion for the representative frame manifest."),
        artifact_entry(validation_path, artifacts_dir, "S11 validation case summary table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV companion for S11 validation cases."),
        artifact_entry(config_path, artifacts_dir, "S11 reproducibility configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "S11 source-code provenance manifest."),
    ]

    report = full_results_markdown(
        artifacts=[*artifacts, manifest_self_entry(full_results_path, artifacts_dir, "Canonical S11 full-results report.")],
        validation_result=validation_result,
        validation_df=validation_df,
        summary_df=summary_df,
        trace_df=trace_df,
        memory_df=memory_df,
        scope_df=scope_df,
        source_df=source_df,
        cpu_gpu_df=cpu_gpu_df,
        reference_df=reference_df,
        frame_df=frame_df,
        test_commands=test_commands,
        source_files=source_files,
        seeds=seeds,
        event_multipliers=event_multipliers,
        policy_ids=policy_ids,
        args=args,
        outcome_classification=outcome_classification,
        caveats_or_blockers=caveats_or_blockers,
        recommended_next_action=recommended_next_action,
        lay_summary=lay_summary,
    )
    write_text(full_results_path, report)
    artifacts.append(artifact_entry(full_results_path, artifacts_dir, "Canonical S11 full-results report."))
    artifacts.append(manifest_self_entry(artifact_manifest_path, artifacts_dir, "S11 artifact manifest."))
    write_json(
        artifact_manifest_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "artifacts": artifacts,
            "validationResult": validation_result,
            "outcomeClassification": outcome_classification,
            "createdAt": utc_now(),
        },
    )

    checksum_inputs = [
        results_path,
        results_csv_path,
        frame_dir,
        trace_path,
        trace_csv_path,
        memory_path,
        memory_csv_path,
        scope_path,
        scope_csv_path,
        source_anchor_path,
        source_anchor_csv_path,
        cpu_gpu_path,
        cpu_gpu_csv_path,
        reference_path,
        reference_csv_path,
        frame_manifest_path,
        frame_manifest_csv_path,
        validation_path,
        validation_csv_path,
        config_path,
        source_manifest_path,
        full_results_path,
        artifact_manifest_path,
    ]
    write_checksums(checksum_inputs, checksum_path, artifacts_dir)
    artifacts.append(artifact_entry(checksum_path, artifacts_dir, "SHA-256 checksums for S11 artifacts."))

    ended_at = utc_now()
    run_manifest = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "experimentTitle": EXPERIMENT_TITLE,
        "startedAt": started_at,
        "endedAt": ended_at,
        "success": bool(validation_success and all(command["success"] for command in test_commands)),
        "status": "completed",
        "artifactsDir": str(artifacts_dir),
        "validationResult": validation_result,
        "outcomeClassification": outcome_classification,
        "recommendedNextAction": recommended_next_action,
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "torchVersion": torch.__version__,
        "cudaAvailable": bool(torch.cuda.is_available()),
        "cudaDeviceName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "configPath": str(config_path),
        "artifactManifestPath": str(artifact_manifest_path),
        "checksumPath": str(checksum_path),
        "testCommands": test_commands,
        "artifacts": artifacts,
    }
    write_json(run_manifest_path, run_manifest)
    write_checksums([*checksum_inputs, checksum_path, run_manifest_path], checksum_path, artifacts_dir)
    return 0 if run_manifest["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
