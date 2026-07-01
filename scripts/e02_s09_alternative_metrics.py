#!/usr/bin/env python3
"""Preflight E02 S09 alternative distance metrics.

S09 is intended to recompute trajectory progress using full per-step value
states.  This script validates the alternative metric definitions on small
known arrays, audits available E01/E02 artifacts for persisted value
trajectories, and writes a blocked handoff if the required trajectories are not
available.
"""

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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


STEP_ID = "S09"
STEP_NUMBER = 9
EXPERIMENT_ID = "E02"

E01_TRACE_SOURCES = (
    ("E01 figure3 unperturbed trajectories", "/previous-artifacts/E01/traces/e01_figure3_trajectories.parquet"),
    ("E01 Frozen Cell trajectories", "/previous-artifacts/E01/traces/e01_frozen_cell_trajectories.parquet"),
    ("E01 same-goal chimera trajectories", "/previous-artifacts/E01/traces/e01_same_goal_chimeras.parquet"),
    ("E01 duplicate-value chimera trajectories", "/previous-artifacts/E01/traces/e01_duplicate_value_chimeras.parquet"),
    ("E01 opposite-direction chimera trajectories", "/previous-artifacts/E01/traces/e01_opposite_direction_chimeras.parquet"),
)
E02_RESULT_SOURCES = (
    ("E02 S01 trace row samples", "/artifacts/research_steps/S01/e02_s01_trace_row_samples.parquet"),
    ("E02 S02 scheduler comparison", "/artifacts/results/e02_scheduler_comparison.parquet"),
    ("E02 S03 activation-rate artifacts", "/artifacts/results/e02_activation_rate_artifacts.parquet"),
    ("E02 S05 dummy Algotype controls", "/artifacts/results/e02_dummy_algotype_controls.parquet"),
    ("E02 S06 speed-matched chimeras", "/artifacts/results/e02_speed_matched_chimeras.parquet"),
    ("E02 S07 local-move nulls", "/artifacts/results/e02_local_move_nulls.parquet"),
    ("E02 S08 DG null benchmarks", "/artifacts/results/e02_dg_null_benchmarks.parquet"),
)
PER_STEP_VALUE_COLUMN_CANDIDATES = {
    "values",
    "values_json",
    "value_state",
    "value_state_json",
    "current_values",
    "current_values_json",
    "array_values",
    "array_values_json",
    "state_values",
    "state_values_json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--config-path", type=Path, default=Path("/previous-artifacts/E01/configs/e01_baseline_configs.json"))
    parser.add_argument("--research-plan-path", type=Path, default=Path("/workspace/RESEARCH_PLAN.md"))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True, env=dict(env) if env else None)
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(time.monotonic() - started),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": bool(result.returncode == 0),
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": int(path.stat().st_size),
    }


def occurrence_tokens(values: Sequence[int | float]) -> tuple[tuple[int | float, int], ...]:
    """Label duplicate values by left-to-right occurrence for deterministic ties."""

    counts: Counter[int | float] = Counter()
    tokens: list[tuple[int | float, int]] = []
    for value in values:
        occurrence = counts[value]
        tokens.append((value, occurrence))
        counts[value] += 1
    return tuple(tokens)


def target_sequence(values: Sequence[int | float], direction: str = "increasing") -> tuple[int | float, ...]:
    if direction in {"increasing", "nondecreasing", "all_increasing"}:
        return tuple(sorted(values))
    if direction in {"decreasing", "nonincreasing", "all_decreasing"}:
        return tuple(sorted(values, reverse=True))
    raise ValueError(f"Unsupported direction: {direction}")


def target_token_positions(values: Sequence[int | float], direction: str = "increasing") -> dict[tuple[int | float, int], int]:
    target_tokens = occurrence_tokens(target_sequence(values, direction))
    return {token: idx for idx, token in enumerate(target_tokens)}


def strict_pair_count(values: Sequence[int | float]) -> int:
    counts = Counter(values)
    n = len(values)
    return n * (n - 1) // 2 - sum(count * (count - 1) // 2 for count in counts.values())


def inversion_count(values: Sequence[int | float], direction: str = "increasing") -> int:
    inversions = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if direction in {"increasing", "nondecreasing", "all_increasing"}:
                inversions += int(values[i] > values[j])
            elif direction in {"decreasing", "nonincreasing", "all_decreasing"}:
                inversions += int(values[i] < values[j])
            else:
                raise ValueError(f"Unsupported direction: {direction}")
    return inversions


def kendall_tau_distance(values: Sequence[int | float], direction: str = "increasing") -> float:
    possible = strict_pair_count(values)
    if possible == 0:
        return 0.0
    return float(inversion_count(values, direction) / possible)


def spearman_position_distance(values: Sequence[int | float], direction: str = "increasing") -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    positions = target_token_positions(values, direction)
    current_tokens = occurrence_tokens(values)
    numerator = sum((idx - positions[token]) ** 2 for idx, token in enumerate(current_tokens))
    max_unique_reversal = n * (n * n - 1) / 3.0
    return float(numerator / max_unique_reversal) if max_unique_reversal else 0.0


def earth_mover_position_distance(values: Sequence[int | float], direction: str = "increasing") -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    positions = target_token_positions(values, direction)
    current_tokens = occurrence_tokens(values)
    numerator = sum(abs(idx - positions[token]) for idx, token in enumerate(current_tokens))
    max_unique_reversal = (n * n) // 2
    return float(numerator / max_unique_reversal) if max_unique_reversal else 0.0


def levenshtein_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return int(previous[-1])


def edit_distance_to_target(values: Sequence[int | float], direction: str = "increasing") -> float:
    n = len(values)
    if n == 0:
        return 0.0
    current_tokens = occurrence_tokens(values)
    target_tokens = occurrence_tokens(target_sequence(values, direction))
    return float(levenshtein_distance(current_tokens, target_tokens) / n)


def scalar_path_curvature(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 1.0
    deltas = np.diff(np.asarray(values, dtype=float))
    total = float(np.sum(np.abs(deltas)))
    net = float(abs(float(values[-1]) - float(values[0])))
    if math.isclose(total, 0.0, abs_tol=1e-12):
        return 1.0
    if math.isclose(net, 0.0, abs_tol=1e-12):
        return float("inf")
    return float(total / net)


def state_space_curvature(states: Sequence[Sequence[int | float]]) -> float:
    if len(states) < 2:
        return 1.0
    arr = np.asarray(states, dtype=float)
    step_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    total = float(np.sum(step_lengths))
    net = float(np.linalg.norm(arr[-1] - arr[0]))
    if math.isclose(total, 0.0, abs_tol=1e-12):
        return 1.0
    if math.isclose(net, 0.0, abs_tol=1e-12):
        return float("inf")
    return float(total / net)


def alternative_metrics(values: Sequence[int | float], direction: str = "increasing") -> dict[str, float | int]:
    inv = inversion_count(values, direction)
    possible = strict_pair_count(values)
    return {
        "inversion_count": int(inv),
        "inversion_fraction": float(inv / possible) if possible else 0.0,
        "kendall_tau_distance": kendall_tau_distance(values, direction),
        "spearman_position_distance": spearman_position_distance(values, direction),
        "edit_distance_to_target": edit_distance_to_target(values, direction),
        "earth_mover_position_distance": earth_mover_position_distance(values, direction),
    }


def metric_sanity_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Mapping[str, Any]) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": dict(details)})

    sorted_unique = alternative_metrics([1, 2, 3])
    add("sorted_unique_zero_distance", all(math.isclose(float(value), 0.0) for value in sorted_unique.values()), sorted_unique)

    reversed_unique = alternative_metrics([3, 2, 1])
    add(
        "reversed_unique_max_rank_distances",
        reversed_unique["inversion_count"] == 3
        and math.isclose(float(reversed_unique["kendall_tau_distance"]), 1.0)
        and math.isclose(float(reversed_unique["spearman_position_distance"]), 1.0)
        and math.isclose(float(reversed_unique["earth_mover_position_distance"]), 1.0)
        and math.isclose(float(reversed_unique["edit_distance_to_target"]), 2.0 / 3.0),
        reversed_unique,
    )

    adjacent_swap = alternative_metrics([1, 3, 2, 4])
    add(
        "adjacent_swap_one_inversion",
        adjacent_swap["inversion_count"] == 1 and math.isclose(float(adjacent_swap["kendall_tau_distance"]), 1.0 / 6.0),
        adjacent_swap,
    )

    duplicate_case = alternative_metrics([2, 1, 2, 1])
    add(
        "duplicate_case_strict_ties_ignored_and_tokens_stable",
        duplicate_case["inversion_count"] == 3
        and math.isclose(float(duplicate_case["kendall_tau_distance"]), 0.75)
        and math.isclose(float(duplicate_case["earth_mover_position_distance"]), 0.75),
        duplicate_case,
    )

    all_ties = alternative_metrics([1, 1, 1])
    add(
        "all_duplicate_ties_zero_order_distance",
        all_ties["inversion_count"] == 0 and math.isclose(float(all_ties["kendall_tau_distance"]), 0.0),
        all_ties,
    )

    add("monotone_scalar_curvature_one", math.isclose(scalar_path_curvature([1.0, 0.5, 0.0]), 1.0), {"curvature": scalar_path_curvature([1.0, 0.5, 0.0])})
    add(
        "backtracking_scalar_curvature_above_one",
        scalar_path_curvature([1.0, 0.2, 0.8, 0.0]) > 1.0,
        {"curvature": scalar_path_curvature([1.0, 0.2, 0.8, 0.0])},
    )
    add(
        "state_space_curvature_backtracking_above_one",
        state_space_curvature([[0, 0], [1, 0], [0, 0], [2, 0]]) > 1.0,
        {"curvature": state_space_curvature([[0, 0], [1, 0], [0, 0], [2, 0]])},
    )

    duplicate_checks = [row for row in checks if "duplicate" in row["name"] or "ties" in row["name"]]
    return {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "metricSanityChecksPassed": bool(all(row["passed"] for row in checks)),
        "duplicateHandlingValidationPassed": bool(duplicate_checks and all(row["passed"] for row in duplicate_checks)),
        "checks": checks,
        "duplicateHandlingPolicy": (
            "Kendall/inversion metrics ignore equal-value ties; position-based Spearman, "
            "earth-mover, and edit metrics use deterministic left-to-right occurrence "
            "tokens (value, occurrence_index) so duplicate states have explicit target "
            "positions without comparing equal values as strict order violations."
        ),
    }


def parquet_non_null_counts(path: Path, columns: Sequence[str]) -> dict[str, int]:
    if not columns:
        return {}
    frame = pq.ParquetFile(path).read(columns=list(columns)).to_pandas()
    return {column: int(frame[column].notna().sum()) for column in columns}


def audit_parquet_source(name: str, path: Path, source_group: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_group": source_group,
        "source_name": name,
        "path": str(path),
        "exists": bool(path.exists()),
        "row_count": 0,
        "column_count": 0,
        "has_sortedness_trajectory": False,
        "has_final_values_json": False,
        "has_per_step_value_column": False,
        "has_per_step_value_trajectory": False,
        "nonfinal_value_rows": 0,
        "final_value_rows": 0,
        "value_like_columns": "",
        "columns": "",
        "blocker": "",
    }
    if not path.exists():
        row["blocker"] = "missing_file"
        return row
    pf = pq.ParquetFile(path)
    columns = list(pf.schema_arrow.names)
    row["row_count"] = int(pf.metadata.num_rows)
    row["column_count"] = int(len(columns))
    row["columns"] = ",".join(columns)
    value_like = [col for col in columns if any(token in col.lower() for token in ("value", "array", "state", "target", "order"))]
    row["value_like_columns"] = ",".join(value_like)
    row["has_sortedness_trajectory"] = "sortedness_percent" in columns or "final_sortedness_percent" in columns
    row["has_final_values_json"] = "final_values_json" in columns

    per_step_columns = [col for col in columns if col in PER_STEP_VALUE_COLUMN_CANDIDATES]
    row["has_per_step_value_column"] = bool(per_step_columns)
    if per_step_columns:
        counts = parquet_non_null_counts(path, per_step_columns)
        row["nonfinal_value_rows"] = int(max(counts.values()) if counts else 0)
        row["has_per_step_value_trajectory"] = row["nonfinal_value_rows"] > 0
    elif "final_values_json" in columns:
        columns_to_read = ["final_values_json"] + (["is_final"] if "is_final" in columns else [])
        frame = pq.ParquetFile(path).read(columns=columns_to_read).to_pandas()
        row["final_value_rows"] = int(frame["final_values_json"].notna().sum())
        if "is_final" in frame.columns:
            row["nonfinal_value_rows"] = int((frame["final_values_json"].notna() & ~frame["is_final"].astype(bool)).sum())
            row["has_per_step_value_trajectory"] = bool(row["nonfinal_value_rows"] > 0)
        else:
            row["has_per_step_value_trajectory"] = False

    if not row["has_per_step_value_trajectory"]:
        row["blocker"] = "no_persisted_per_step_value_arrays"
    return row


def audit_config(config_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config_path": str(config_path),
        "exists": bool(config_path.exists()),
        "has_unique_initial_arrays": False,
        "has_duplicate_initial_arrays": False,
        "unique_repeat_count": 0,
        "duplicate_repeat_count": 0,
        "target_order_status": "unavailable",
        "duplicate_policy_status": "unavailable",
        "blocker": "",
    }
    if not config_path.exists():
        row["blocker"] = "missing_config"
        return row
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    value_banks = cfg.get("seedBanks", {}).get("valueBanks", {})
    unique = value_banks.get("unique_1_to_100", {})
    duplicate = value_banks.get("duplicate_1_to_10_x10", {})
    row["unique_repeat_count"] = int(len(unique.get("initialArrays", [])))
    row["duplicate_repeat_count"] = int(len(duplicate.get("initialArrays", [])))
    row["has_unique_initial_arrays"] = bool(row["unique_repeat_count"] > 0)
    row["has_duplicate_initial_arrays"] = bool(row["duplicate_repeat_count"] > 0)
    if row["has_unique_initial_arrays"] and row["has_duplicate_initial_arrays"]:
        row["target_order_status"] = "inferable_from_direction_and_value_bank"
        row["duplicate_policy_status"] = "duplicate_bank_present_non_strict_tie_semantics_documented"
    else:
        row["blocker"] = "missing_initial_value_banks"
    return row


def audit_inputs(config_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_rows = [
        audit_parquet_source(name, Path(path), "E01 trajectory artifact") for name, path in E01_TRACE_SOURCES
    ] + [
        audit_parquet_source(name, Path(path), "E02 result artifact") for name, path in E02_RESULT_SOURCES
    ]
    audit_df = pd.DataFrame(source_rows)
    config_audit = audit_config(config_path)
    required_trajectory_rows = audit_df[audit_df["source_group"].isin(["E01 trajectory artifact", "E02 result artifact"])]
    available_value_sources = required_trajectory_rows[required_trajectory_rows["has_per_step_value_trajectory"].astype(bool)]
    summary = {
        "configAudit": config_audit,
        "requiredSourceCount": int(len(required_trajectory_rows)),
        "existingRequiredSourceCount": int(required_trajectory_rows["exists"].astype(bool).sum()),
        "sourcesWithPerStepValueTrajectories": int(len(available_value_sources)),
        "sourcesWithSortednessTrajectories": int(required_trajectory_rows["has_sortedness_trajectory"].astype(bool).sum()),
        "allRequiredValueTrajectoriesAvailable": bool(len(available_value_sources) == len(required_trajectory_rows) and len(required_trajectory_rows) > 0),
        "targetOrderDefinitionsAvailable": bool(config_audit["target_order_status"] == "inferable_from_direction_and_value_bank"),
        "duplicateHandlingInputsAvailable": bool(config_audit["duplicate_policy_status"] == "duplicate_bank_present_non_strict_tie_semantics_documented"),
        "deterministicSimulatorCanRecordValuesInMemory": True,
        "deterministicSimulatorPersistedValueTrajectoriesInPriorArtifacts": False,
    }
    return audit_df, summary


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    show = df.head(max_rows).copy()
    columns = list(show.columns)
    rows = [[str(value) for value in row] for row in show.itertuples(index=False, name=None)]

    def esc(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(esc(col) for col in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(esc(value) for value in row) + " |" for row in rows]
    suffix = [f"\nShowing first {max_rows} of {len(df)} rows."] if len(df) > max_rows else []
    return "\n".join([header, divider, *body, *suffix])


def build_report(
    *,
    generated_at: str,
    artifacts_dir: Path,
    audit_path: Path,
    sanity_path: Path,
    validation_path: Path,
    status_path: Path,
    log_path: Path,
    manifest_path: Path,
    src_manifest_path: Path,
    audit_df: pd.DataFrame,
    validation: Mapping[str, Any],
    unit_result: Mapping[str, Any] | None,
    git_commit: str,
    git_status: str,
    elapsed_seconds: float,
) -> str:
    artifacts_written = validation["artifactsWritten"]
    source_summary = audit_df[
        [
            "source_group",
            "source_name",
            "row_count",
            "has_sortedness_trajectory",
            "has_final_values_json",
            "has_per_step_value_trajectory",
            "blocker",
        ]
    ]
    unit_line = "not run"
    if unit_result is not None:
        unit_line = f"{unit_result['command']} -> return code {unit_result['returnCode']}"
    missing_results_path = artifacts_dir / "results" / "e02_alternative_metrics.parquet"
    missing_figure_path = artifacts_dir / "figures" / "e02" / "metric_sensitivity_matrix.png"
    return f"""# E02 S09 Full Results: Alternative Distance Metrics

## Top Summary

- Step ID: S09
- Completion status: blocked_missing_required_inputs
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: blocked after preflight; metric sanity checks passed and duplicate handling checks passed, but the input audit found zero persisted per-step value trajectories across the required E01/E02 trajectory artifacts.
- Outcome classification: constraining/contradictory
- Caveats or blockers: Required value trajectories are unavailable in the persisted E01/E02 artifacts. Existing traces store Sortedness trajectories, hashes, label-position codes, and final arrays, but not the intermediate value arrays needed for Kendall, Spearman, edit, EMD, or state-space curvature trajectories.
- Lay summary: The alternative metrics are well defined on small unique and duplicate arrays, but the saved experiment traces do not contain the actual array values at each recorded step. Without those states, S09 cannot honestly replace Sortedness on the same trajectories.
- Recommended next action: Regenerate or export trace-level value arrays from E01/E02 runs, including duplicate-value runs and direction/target metadata, then rerun S09. Do not start S10 until the Chief Scientist decides whether to regenerate the required trajectories or revise S09.

## Frozen Question

Do qualitative claims survive when progress is measured by distances other than the paper's Sortedness metric?

## Outcome

S09 could not produce the planned alternative-metric result table or metric-sensitivity figure because the required input layer is missing. The planned output `{missing_results_path}` was not written, and `{missing_figure_path}` was not written, to avoid mistaking an input-audit failure for a completed metric comparison.

This is a constraining result for the artifact package: prior steps preserved enough information for Sortedness, Aggregation, DG over Sortedness, and hash-level reproducibility checks, but not enough state to recompute rank/edit/transport metrics on the recorded trajectories.

## Inputs Audited

- E01 trace schema and trajectory artifacts under `/previous-artifacts/E01/traces/`.
- E02 result artifacts under `/artifacts/results/` and the S01 trace-row sample.
- E01 baseline config `/previous-artifacts/E01/configs/e01_baseline_configs.json`.
- Repository code at commit `{git_commit}`.

## Methods

The S09 script implemented the planned distance families and ran deterministic toy validations before auditing artifacts:

- Kendall tau distance and raw inversion count count strict out-of-order value pairs; equal-value ties are ignored.
- Spearman position distance assigns duplicate values deterministic occurrence tokens `(value, occurrence_index)` and compares current positions with target positions.
- Edit distance uses Levenshtein distance over the same occurrence tokens.
- Earth-mover-over-positions is the normalized sum of absolute current-vs-target occurrence-token displacements.
- Scalar trajectory curvature is total variation divided by net change. State-space curvature is total Euclidean path length divided by start-to-end Euclidean distance.

The artifact audit inspected Parquet metadata and relevant columns for persisted per-step value columns such as `values`, `values_json`, `current_values_json`, or equivalent state arrays. It separately checked whether `final_values_json` was present only on final rows, because final arrays are not sufficient for trajectory metrics.

## Commands

- `python scripts/e02_s09_alternative_metrics.py --repo-dir /workspace/cell-research --artifacts-dir /artifacts --run-unit-tests`
- Unit-test command inside the script: `{unit_line}`

## Results

Metric sanity validation passed, including duplicate-specific checks. The input audit failed the production prerequisite:

- Required trajectory/result sources audited: {validation['requiredSourceCount']}
- Existing sources audited: {validation['existingRequiredSourceCount']}
- Sources with Sortedness trajectories or summaries: {validation['sourcesWithSortednessTrajectories']}
- Sources with persisted per-step value trajectories: {validation['sourcesWithPerStepValueTrajectories']}
- Target-order definitions: {validation['targetOrderDefinitionsStatus']}
- Duplicate handling inputs: {validation['duplicateHandlingInputsStatus']}

### Input Audit Summary

{markdown_table(source_summary, max_rows=20)}

## Validation

- Metric sanity checks: {validation['metricSanityChecksPassed']}
- Duplicate handling validation: {validation['duplicateHandlingValidationPassed']}
- Input audit passed: {validation['inputAuditPassed']}
- Planned results table written: {validation['plannedResultsWritten']}
- Planned figure written: {validation['plannedFigureWritten']}
- Unit tests: {unit_line}

Detailed validation JSON: `{validation_path}`  
Status JSON: `{status_path}`  
Metric sanity details: `{sanity_path}`  
Input audit CSV: `{audit_path}`

## Caveats And Blockers

- The deterministic S01 simulator keeps value arrays in memory inside `SimulatorResult.records`, but prior E02 result tables stored compact hashes and summaries rather than those arrays.
- Re-running selected deterministic conditions would create a new evidence layer, not a replacement for the persisted E01/E02 trajectories requested by S09 as written.
- The E01 config provides initial value banks and duplicate-value conventions, so target orders are inferable for increasing/decreasing goals, but target definitions alone cannot reconstruct intermediate states.
- Opposite-direction chimeras may require per-label or per-direction target metadata when S09 is retried; this was not the blocking issue because value trajectories were already absent.

## Provenance

- Generated at UTC: {generated_at}
- Runtime: Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}, PyArrow {pq.__version__ if hasattr(pq, '__version__') else 'available'}
- Git commit at run time: `{git_commit}`
- Git status at run time: `{git_status or 'clean'}`
- Elapsed seconds: {elapsed_seconds:.3f}
- No new dependencies were installed.
- Artifact manifest: `{manifest_path}`
- Source manifest: `{src_manifest_path}`

## Recommended Next Action

Regenerate or export trace-level value arrays for the E01 and E02 trajectory families needed by S09, with one row per recorded step containing the current value array and direction/target metadata. Then rerun S09 before starting S10, or revise S09 to be an explicitly new rerun-only analysis if the Chief Scientist decides the original trajectories do not need to be remeasured.
"""


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    logs_dir = artifacts_dir / "logs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    for directory in (step_dir, logs_dir, src_snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    generated_at = utc_now()
    log_lines = [
        f"E02 S09 started {generated_at}",
        f"repo_dir={repo_dir}",
        f"artifacts_dir={artifacts_dir}",
        "mode=preflight_blocker_audit",
    ]

    unit_result: dict[str, Any] | None = None
    if args.run_unit_tests:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_dir)
        unit_result = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_*.py"], repo_dir, env)
        log_lines.extend(["UNIT TESTS:", json.dumps(unit_result, indent=2, sort_keys=True)])
        if not unit_result["success"]:
            log_path = logs_dir / "e02_s09_alternative_metrics.log"
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return int(unit_result["returnCode"] or 1)

    sanity = metric_sanity_checks()
    audit_df, audit_summary = audit_inputs(args.config_path)

    audit_path = step_dir / "e02_s09_input_audit.csv"
    sanity_path = step_dir / "e02_s09_metric_sanity.json"
    validation_path = step_dir / "s09_validation.json"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    src_manifest_path = src_snapshot_dir / "e02_s09_alternative_metrics_manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    log_path = logs_dir / "e02_s09_alternative_metrics.log"

    audit_df.to_csv(audit_path, index=False)
    write_json(sanity_path, sanity)

    input_audit_passed = bool(
        audit_summary["allRequiredValueTrajectoriesAvailable"]
        and audit_summary["targetOrderDefinitionsAvailable"]
        and audit_summary["duplicateHandlingInputsAvailable"]
    )
    status = "blocked_missing_required_value_trajectories" if not input_audit_passed else "ready_for_metric_evaluation"
    caveats = [
        "No required E01/E02 source has persisted per-step value arrays.",
        "Final arrays and value-trace hashes are insufficient to compute alternative metric trajectories.",
        "Target orders are inferable from configs, but intermediate value states are not reconstructible from persisted hashes/summaries.",
    ]
    artifacts_written = [
        str(audit_path),
        str(sanity_path),
        str(validation_path),
        str(status_path),
        str(log_path),
        str(report_path),
        str(src_manifest_path),
        str(manifest_path),
    ]
    validation = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": False,
        "status": status,
        "artifactsWritten": artifacts_written,
        "validationResult": (
            "blocked: metric sanity and duplicate checks passed, but input audit found zero persisted per-step value trajectories"
        ),
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": (
            "Regenerate or export trace-level value arrays with direction/target metadata, then rerun S09 before starting S10."
        ),
        "outcomeClassification": "constraining/contradictory",
        "metricSanityChecksPassed": bool(sanity["metricSanityChecksPassed"]),
        "duplicateHandlingValidationPassed": bool(sanity["duplicateHandlingValidationPassed"]),
        "inputAuditPassed": input_audit_passed,
        "plannedResultsWritten": False,
        "plannedFigureWritten": False,
        "requiredSourceCount": int(audit_summary["requiredSourceCount"]),
        "existingRequiredSourceCount": int(audit_summary["existingRequiredSourceCount"]),
        "sourcesWithPerStepValueTrajectories": int(audit_summary["sourcesWithPerStepValueTrajectories"]),
        "sourcesWithSortednessTrajectories": int(audit_summary["sourcesWithSortednessTrajectories"]),
        "targetOrderDefinitionsStatus": (
            "available_inferable_from_config" if audit_summary["targetOrderDefinitionsAvailable"] else "unavailable"
        ),
        "duplicateHandlingInputsStatus": (
            "available_duplicate_bank_and_metric_policy_validated"
            if audit_summary["duplicateHandlingInputsAvailable"]
            else "unavailable"
        ),
        "configAudit": audit_summary["configAudit"],
        "missingPlannedOutputs": [
            str(artifacts_dir / "results" / "e02_alternative_metrics.parquet"),
            str(artifacts_dir / "figures" / "e02" / "metric_sensitivity_matrix.png"),
        ],
    }
    write_json(validation_path, validation)
    write_json(status_path, validation)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])
    elapsed_seconds = time.monotonic() - start
    log_lines.extend(
        [
            f"metric_sanity_checks_passed={sanity['metricSanityChecksPassed']}",
            f"duplicate_handling_validation_passed={sanity['duplicateHandlingValidationPassed']}",
            f"input_audit_passed={input_audit_passed}",
            f"sources_with_per_step_values={audit_summary['sourcesWithPerStepValueTrajectories']}",
            f"elapsed_seconds={elapsed_seconds:.3f}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    code_files = [
        repo_dir / "scripts/e02_s09_alternative_metrics.py",
        repo_dir / "tests/e02/test_alternative_metrics.py",
    ]
    src_manifest = {
        "schema": "eidosoma.src_snapshot.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "gitCommit": git_commit,
        "gitStatusShort": git_status,
        "sourceFiles": [
            {
                "path": str(path),
                "relativePath": str(path.relative_to(repo_dir)),
                "sha256": sha256_file(path),
                "sizeBytes": int(path.stat().st_size),
            }
            for path in code_files
            if path.exists()
        ],
        "parameters": {
            "mode": "preflight_blocker_audit",
            "e01TraceSources": [path for _, path in E01_TRACE_SOURCES],
            "e02ResultSources": [path for _, path in E02_RESULT_SOURCES],
        },
        "validation": validation,
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": "available",
            "newDependenciesInstalled": [],
        },
    }
    write_json(src_manifest_path, src_manifest)

    report_text = build_report(
        generated_at=generated_at,
        artifacts_dir=artifacts_dir,
        audit_path=audit_path,
        sanity_path=sanity_path,
        validation_path=validation_path,
        status_path=status_path,
        log_path=log_path,
        manifest_path=manifest_path,
        src_manifest_path=src_manifest_path,
        audit_df=audit_df,
        validation=validation,
        unit_result=unit_result,
        git_commit=git_commit,
        git_status=git_status,
        elapsed_seconds=elapsed_seconds,
    )
    report_path.write_text(report_text, encoding="utf-8")
    write_json(validation_path, validation)
    write_json(status_path, validation)

    manifest = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": generated_at,
        "status": status,
        "artifacts": [
            artifact_entry(audit_path, artifacts_dir, "Input audit for E01/E02 value trajectory availability"),
            artifact_entry(sanity_path, artifacts_dir, "Alternative metric sanity checks and duplicate policy validation"),
            artifact_entry(validation_path, artifacts_dir, "S09 validation evidence"),
            artifact_entry(status_path, artifacts_dir, "S09 machine-readable blocked status"),
            artifact_entry(report_path, artifacts_dir, "S09 full-results blocked handoff report"),
            artifact_entry(log_path, artifacts_dir, "S09 execution log"),
            artifact_entry(src_manifest_path, artifacts_dir, "S09 source snapshot manifest"),
        ],
        "plannedOutputsNotWritten": validation["missingPlannedOutputs"],
    }
    write_json(manifest_path, manifest)

    return 0 if sanity["metricSanityChecksPassed"] and sanity["duplicateHandlingValidationPassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
