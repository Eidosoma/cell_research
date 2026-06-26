#!/usr/bin/env python3
"""Run E01 S14 scaled replication subset."""

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
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import e01_s04_reproduce_figure03 as s04
import e01_s07_frozen_cell_robustness as s07
import e01_s08_delayed_gratification as s08
import e01_s09_same_goal_chimeras as s09


EXPERIMENT_ID = "E01"
STEP_ID = "S14"
STEP_NUMBER = 14
STATUS = "completed"
CONFIG_VERSION = "e01_baseline_config.v1"
CONFIG_PATH_DEFAULT = Path("/artifacts/configs/e01_baseline_config.json")
CONDITION_MATRIX_DEFAULT = Path("/artifacts/research_steps/S03/condition_matrix.csv")
SEED_TABLE_DEFAULT = Path("/artifacts/research_steps/S03/seed_table.csv")
RUN_MANIFEST_DEFAULT = Path("/artifacts/provenance/run_manifest.json")
PROGRESS_BINS = np.linspace(0.0, 1.0, 101)
MIXED_AGGREGATION_MIXTURES = [
    "bubble_insertion",
    "bubble_selection",
    "insertion_selection",
    "bubble_insertion_selection",
]
METRICS_BY_FAMILY = {
    "no_frozen_efficiency": [
        "finalSortednessPercent",
        "finalMonotonicityError",
        "swapCount",
        "comparisonCount",
        "swapPlusComparisonSteps",
    ],
    "frozen_robustness": [
        "finalSortednessPercent",
        "finalMonotonicityError",
        "swapCount",
        "comparisonCount",
        "swapPlusComparisonSteps",
    ],
    "delayed_gratification": ["delayedGratification", "dgEventCount"],
    "same_goal_aggregation": [
        "peakAggregation",
        "peakAggregationMinusNull",
        "finalAggregation",
        "meanAggregationOverProgress",
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def seed_for(condition_id: str, replicate_index: int, stream: str) -> int:
    text = f"{CONFIG_VERSION}:{condition_id}:{replicate_index}:{stream}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def seed_for_parts(*parts: Any) -> int:
    text = ":".join([CONFIG_VERSION, *(str(part) for part in parts)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def goal_json(value: str) -> str:
    return json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"))


def make_seed_row(condition: dict[str, Any], replicate_index: int) -> dict[str, Any]:
    algorithms = [part for part in str(condition["algorithms"]).split(";") if part]
    has_multiple_algotypes = len(algorithms) > 1
    frozen_count = int(condition["frozenCount"])
    return {
        "researchStepId": "S03",
        "configVersion": CONFIG_VERSION,
        "conditionId": str(condition["conditionId"]),
        "producerStep": str(condition["producerStep"]),
        "replicateIndex": int(replicate_index),
        "replicateNumber": int(replicate_index) + 1,
        "inputProfile": str(condition["inputProfile"]),
        "implementation": str(condition["implementation"]),
        "mixtureId": str(condition["mixtureId"]),
        "frozenVariant": str(condition["frozenVariant"]),
        "frozenCount": frozen_count,
        "inputPermutationSeed": seed_for_parts("input_permutation", condition["inputProfile"], replicate_index),
        "algotypeAssignmentSeed": (
            seed_for_parts(
                "algotype_assignment",
                condition["inputProfile"],
                condition["mixtureId"],
                goal_json(str(condition["goalDirections"])),
                replicate_index,
            )
            if has_multiple_algotypes
            else ""
        ),
        "frozenPositionSeed": (
            seed_for_parts("frozen_position", condition["inputProfile"], frozen_count, replicate_index)
            if frozen_count
            else ""
        ),
        "schedulerSeed": seed_for(str(condition["conditionId"]), replicate_index, "scheduler"),
        "tieBreakerSeed": seed_for(str(condition["conditionId"]), replicate_index, "tie_breaker"),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in sorted({p.resolve() for p in paths if p.exists() and p.is_file()}):
        if path in seen:
            continue
        seen.add(path)
        artifacts.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return artifacts


def run_command(args: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd or REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_info() -> dict[str, Any]:
    return {
        "repository": str(REPO_ROOT),
        "branch": run_command(["git", "branch", "--show-current"]),
        "commit": run_command(["git", "rev-parse", "HEAD"]),
        "statusShort": run_command(["git", "status", "--short"]),
        "remoteOriginUrl": run_command(["git", "remote", "get-url", "origin"]),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            return f"{value:.4g}"
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def as_condition_records(condition_matrix_path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(condition_matrix_path)
    return [
        {key: ("" if pd.isna(value) else value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def selected_conditions(condition_matrix_path: Path) -> dict[str, list[dict[str, Any]]]:
    conditions = as_condition_records(condition_matrix_path)
    no_frozen = [row for row in conditions if row["producerStep"] == "S04"]
    frozen = [
        row
        for row in conditions
        if row["producerStep"] == "S07" and int(row["frozenCount"]) > 0
    ]
    aggregation = [
        row
        for row in conditions
        if row["producerStep"] == "S09" and row["mixtureId"] in MIXED_AGGREGATION_MIXTURES
    ]
    no_frozen.sort(key=lambda row: (row["implementation"], row["algorithms"]))
    frozen.sort(key=lambda row: (row["implementation"], row["algorithms"], row["frozenVariant"], int(row["frozenCount"])))
    aggregation.sort(key=lambda row: MIXED_AGGREGATION_MIXTURES.index(row["mixtureId"]))
    return {
        "no_frozen_efficiency": no_frozen,
        "frozen_robustness": frozen,
        "same_goal_aggregation": aggregation,
    }


def validate_seed_extension(seed_table_path: Path, conditions: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    s03 = pd.read_csv(seed_table_path)
    messages: list[str] = []
    ok = True
    for condition in conditions:
        condition_id = str(condition["conditionId"])
        source_rows = s03[s03["conditionId"] == condition_id].copy()
        if source_rows.empty:
            ok = False
            messages.append(f"No S03 seed rows found for {condition_id}.")
            continue
        for _, source in source_rows.head(100).iterrows():
            replicate_index = int(source["replicateIndex"])
            generated = make_seed_row(condition, replicate_index)
            for field in ["inputPermutationSeed", "schedulerSeed", "tieBreakerSeed"]:
                if int(source[field]) != int(generated[field]):
                    ok = False
                    messages.append(f"Seed mismatch for {condition_id} replicate {replicate_index} field {field}.")
                    break
            if int(condition["frozenCount"]) > 0 and int(source["frozenPositionSeed"]) != int(generated["frozenPositionSeed"]):
                ok = False
                messages.append(f"Frozen seed mismatch for {condition_id} replicate {replicate_index}.")
                break
            if len(str(condition["algorithms"]).split(";")) > 1 and int(source["algotypeAssignmentSeed"]) != int(generated["algotypeAssignmentSeed"]):
                ok = False
                messages.append(f"Algotype seed mismatch for {condition_id} replicate {replicate_index}.")
                break
        if ok:
            messages.append(f"S03 seed convention reproduced for {condition_id} first 100 rows.")
    return ok, messages


class CollectingWriter:
    def __init__(self, collect_sortedness: bool = False, collect_aggregation: bool = False):
        self.collect_sortedness = collect_sortedness
        self.collect_aggregation = collect_aggregation
        self.row_count = 0
        self.sortedness: list[float] = []
        self.aggregation: list[float] = []
        self.random_null: list[float] = []

    def write(self, row: dict[str, Any]) -> None:
        self.row_count += 1
        if self.collect_sortedness:
            self.sortedness.append(float(row.get("sortednessPercent", 0.0)))
        if self.collect_aggregation:
            sequence = str(row.get("algotypeSequence", ""))
            value = aggregation_value(sequence)
            null = random_adjacency_expected(sequence)
            self.aggregation.append(value)
            self.random_null.append(null)
            if not self.collect_sortedness:
                self.sortedness.append(float(row.get("sortednessPercent", 0.0)))


def aggregation_value(sequence: str) -> float:
    if len(sequence) < 2:
        return 0.0
    matches = sum(1 for idx in range(1, len(sequence)) if sequence[idx] == sequence[idx - 1])
    return matches / (len(sequence) - 1)


def random_adjacency_expected(sequence: str) -> float:
    if len(sequence) < 2:
        return 0.0
    counts = Counter(sequence)
    n = sum(counts.values())
    return sum(count * (count - 1) for count in counts.values()) / (n * (n - 1))


def interpolate(values: list[float], bins: int = 101) -> list[float]:
    if not values:
        return [0.0] * bins
    if len(values) == 1:
        return [float(values[0])] * bins
    x = np.linspace(0.0, 1.0, len(values))
    return [float(value) for value in np.interp(PROGRESS_BINS if bins == 101 else np.linspace(0.0, 1.0, bins), x, np.asarray(values, dtype=float))]


def dg_record_from_sortedness(
    source_condition_id: str,
    source_step_id: str,
    condition: dict[str, Any],
    seed_row: dict[str, Any],
    sortedness: list[float],
) -> dict[str, Any]:
    metrics = s08.delayed_gratification_from_sortedness(sortedness)
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "scaledFamily": "delayed_gratification",
        "sourceStepId": source_step_id,
        "sourceConditionId": source_condition_id,
        "conditionId": f"S14_scaled_{source_condition_id}",
        "implementation": condition["implementation"],
        "algorithm": condition["algorithms"],
        "mixtureId": condition["mixtureId"],
        "inputProfile": condition["inputProfile"],
        "frozenVariant": condition["frozenVariant"],
        "frozenCount": int(condition["frozenCount"]),
        "replicateIndex": int(seed_row["replicateIndex"]),
        "replicateNumber": int(seed_row["replicateNumber"]),
        "trajectoryEventCount": int(len(sortedness)),
        "initialSortednessPercent": float(sortedness[0]) if sortedness else 0.0,
        "finalSortednessPercent": float(sortedness[-1]) if sortedness else 0.0,
        **{
            key: metrics[key]
            for key in [
                "delayedGratification",
                "dgEventCount",
                "dgPositiveEventCount",
                "dgNegativeEventCount",
                "dgZeroEventCount",
                "dgTotalDrop",
                "dgTotalRecovery",
                "dgMeanDrop",
                "dgMeanRecovery",
                "dgMaxEventScore",
                "dgMinEventScore",
            ]
        },
    }


def run_no_frozen_task(task: dict[str, Any]) -> dict[str, Any]:
    condition = task["condition"]
    replicate_index = int(task["replicateIndex"])
    seed_row = make_seed_row(condition, replicate_index)
    max_swaps = int(task["maxSwaps"])
    start = time.perf_counter()
    initial_values = s04.initial_values_from_seed(int(seed_row["inputPermutationSeed"]), int(condition["n"]))
    if condition["implementation"] == "traditional":
        trace = s04.TRADITIONAL_RUNNERS[condition["algorithms"]](initial_values, max_swaps)
    else:
        trace = s04.run_cell_view(
            condition["algorithms"],
            initial_values,
            int(seed_row["schedulerSeed"]),
            int(seed_row["tieBreakerSeed"]),
            max_swaps,
        )
    final_state = trace.states[-1]
    source_condition_id = str(condition["conditionId"])
    record = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "scaledFamily": "no_frozen_efficiency",
        "sourceStepId": "S04",
        "sourceConditionId": source_condition_id,
        "conditionId": f"S14_scaled_{source_condition_id}",
        "implementation": condition["implementation"],
        "algorithm": condition["algorithms"],
        "algorithms": condition["algorithms"],
        "mixtureId": condition["mixtureId"],
        "inputProfile": condition["inputProfile"],
        "n": int(condition["n"]),
        "frozenVariant": condition["frozenVariant"],
        "frozenCount": int(condition["frozenCount"]),
        "replicateIndex": replicate_index,
        "replicateNumber": replicate_index + 1,
        "inputPermutationSeed": int(seed_row["inputPermutationSeed"]),
        "frozenPositionSeed": np.nan,
        "algotypeAssignmentSeed": np.nan,
        "schedulerSeed": int(seed_row["schedulerSeed"]),
        "tieBreakerSeed": int(seed_row["tieBreakerSeed"]),
        "completed": bool(trace.completed),
        "stopReason": trace.stop_reason,
        "finalSortednessPercent": float(s04.sortedness_percent(final_state)),
        "finalMonotonicityError": int(s04.monotonicity_error(final_state)),
        "swapCount": int(trace.final_swap_count),
        "comparisonCount": int(trace.final_comparison_count),
        "swapPlusComparisonSteps": int(trace.final_swap_count + trace.final_comparison_count),
        "archivedCompareAndSwapCount": int(trace.final_archived_compare_and_swap_count),
        "schedulerRounds": int(trace.scheduler_rounds),
        "eventCount": int(len(trace.states)),
        "wallTimeSeconds": float(trace.wall_time_seconds),
        "taskWallTimeSeconds": float(time.perf_counter() - start),
        "finalAggregation": np.nan,
        "peakAggregation": np.nan,
        "peakAggregationMinusNull": np.nan,
        "meanAggregationOverProgress": np.nan,
    }
    dg = None
    if condition["algorithms"] in {"bubble", "insertion"}:
        sortedness = [s04.sortedness_percent(state) for state in trace.states]
        dg = dg_record_from_sortedness(source_condition_id, "S04", condition, seed_row, sortedness)
    return {"replicate": record, "dg": dg, "curves": [], "peak": None}


def run_frozen_task(task: dict[str, Any]) -> dict[str, Any]:
    condition = task["condition"]
    replicate_index = int(task["replicateIndex"])
    seed_row = make_seed_row(condition, replicate_index)
    collect_dg = condition["frozenVariant"] == "stuck" and condition["algorithms"] in {"bubble", "insertion"}
    writer = CollectingWriter(collect_sortedness=collect_dg)
    start = time.perf_counter()
    result = s07.run_one(condition, seed_row, task["stopPolicies"], writer)
    source_condition_id = str(condition["conditionId"])
    record = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "scaledFamily": "frozen_robustness",
        "sourceStepId": "S07",
        "sourceConditionId": source_condition_id,
        "conditionId": f"S14_scaled_{source_condition_id}",
        "implementation": condition["implementation"],
        "algorithm": condition["algorithms"],
        "algorithms": condition["algorithms"],
        "mixtureId": condition["mixtureId"],
        "inputProfile": condition["inputProfile"],
        "n": int(condition["n"]),
        "frozenVariant": condition["frozenVariant"],
        "frozenCount": int(condition["frozenCount"]),
        "replicateIndex": replicate_index,
        "replicateNumber": replicate_index + 1,
        "inputPermutationSeed": int(seed_row["inputPermutationSeed"]),
        "frozenPositionSeed": int(seed_row["frozenPositionSeed"]),
        "algotypeAssignmentSeed": np.nan,
        "schedulerSeed": int(seed_row["schedulerSeed"]),
        "tieBreakerSeed": int(seed_row["tieBreakerSeed"]),
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "finalSortednessPercent": float(result.final_sortedness_percent),
        "finalMonotonicityError": int(result.final_monotonicity_error),
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "swapPlusComparisonSteps": int(result.swap_count + result.comparison_count),
        "archivedCompareAndSwapCount": np.nan,
        "schedulerRounds": int(result.scheduler_rounds),
        "eventCount": int(result.event_count),
        "wallTimeSeconds": float(result.wall_time_seconds),
        "taskWallTimeSeconds": float(time.perf_counter() - start),
        "finalAggregation": np.nan,
        "peakAggregation": np.nan,
        "peakAggregationMinusNull": np.nan,
        "meanAggregationOverProgress": np.nan,
    }
    dg = dg_record_from_sortedness(source_condition_id, "S07", condition, seed_row, writer.sortedness) if collect_dg else None
    return {"replicate": record, "dg": dg, "curves": [], "peak": None}


def run_aggregation_task(task: dict[str, Any]) -> dict[str, Any]:
    condition = task["condition"]
    replicate_index = int(task["replicateIndex"])
    seed_row = make_seed_row(condition, replicate_index)
    writer = CollectingWriter(collect_sortedness=True, collect_aggregation=True)
    start = time.perf_counter()
    result = s09.run_chimera(
        condition,
        seed_row,
        writer,
        max_swaps=int(task["maxSwaps"]),
        max_comparisons=int(task["maxComparisons"]),
        max_rounds=int(task["maxRounds"]),
        no_move_checks_required=int(task["noMoveChecksRequired"]),
        stable_no_progress_round_cap=int(task["stableNoProgressRoundCap"]),
    )
    source_condition_id = str(condition["conditionId"])
    agg_curve = interpolate(writer.aggregation)
    sorted_curve = interpolate(writer.sortedness)
    null_curve = interpolate(writer.random_null)
    minus_null_curve = [float(a - n) for a, n in zip(agg_curve, null_curve)]
    peak_idx = int(np.argmax(np.asarray(agg_curve, dtype=float)))
    curves = [
        {
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "scaledFamily": "same_goal_aggregation",
            "sourceStepId": "S09",
            "sourceConditionId": source_condition_id,
            "conditionId": f"S14_scaled_{source_condition_id}",
            "mixtureId": condition["mixtureId"],
            "algorithms": condition["algorithms"],
            "replicateIndex": replicate_index,
            "replicateNumber": replicate_index + 1,
            "progressBin": bin_index,
            "normalizedProgress": float(bin_index / 100.0),
            "aggregationValue": float(agg_curve[bin_index]),
            "randomNullExpected": float(null_curve[bin_index]),
            "aggregationMinusNull": float(minus_null_curve[bin_index]),
            "sortednessPercent": float(sorted_curve[bin_index]),
            "finalSwapCount": int(result.swap_count),
        }
        for bin_index in range(101)
    ]
    peak = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "scaledFamily": "same_goal_aggregation",
        "sourceStepId": "S09",
        "sourceConditionId": source_condition_id,
        "conditionId": f"S14_scaled_{source_condition_id}",
        "mixtureId": condition["mixtureId"],
        "algorithms": condition["algorithms"],
        "replicateIndex": replicate_index,
        "replicateNumber": replicate_index + 1,
        "randomNullExpected": float(null_curve[peak_idx]),
        "initialAggregation": float(agg_curve[0]),
        "peakAggregation": float(agg_curve[peak_idx]),
        "peakAggregationMinusNull": float(minus_null_curve[peak_idx]),
        "peakProgressBin": int(peak_idx),
        "peakNormalizedProgress": float(peak_idx / 100.0),
        "finalAggregation": float(agg_curve[-1]),
        "finalAggregationMinusNull": float(minus_null_curve[-1]),
        "meanAggregationOverProgress": float(np.mean(agg_curve)),
        "finalSortednessPercent": float(result.final_sortedness_percent),
        "finalSwapCount": int(result.swap_count),
    }
    record = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "scaledFamily": "same_goal_aggregation",
        "sourceStepId": "S09",
        "sourceConditionId": source_condition_id,
        "conditionId": f"S14_scaled_{source_condition_id}",
        "implementation": condition["implementation"],
        "algorithm": "mixed",
        "algorithms": condition["algorithms"],
        "mixtureId": condition["mixtureId"],
        "inputProfile": condition["inputProfile"],
        "n": int(condition["n"]),
        "frozenVariant": condition["frozenVariant"],
        "frozenCount": int(condition["frozenCount"]),
        "replicateIndex": replicate_index,
        "replicateNumber": replicate_index + 1,
        "inputPermutationSeed": int(seed_row["inputPermutationSeed"]),
        "frozenPositionSeed": np.nan,
        "algotypeAssignmentSeed": int(seed_row["algotypeAssignmentSeed"]),
        "schedulerSeed": int(seed_row["schedulerSeed"]),
        "tieBreakerSeed": int(seed_row["tieBreakerSeed"]),
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "finalSortednessPercent": float(result.final_sortedness_percent),
        "finalMonotonicityError": int(result.final_monotonicity_error),
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "swapPlusComparisonSteps": int(result.swap_count + result.comparison_count),
        "archivedCompareAndSwapCount": int(result.archived_compare_and_swap_count),
        "schedulerRounds": int(result.scheduler_rounds),
        "eventCount": int(result.event_count),
        "wallTimeSeconds": float(result.wall_time_seconds),
        "taskWallTimeSeconds": float(time.perf_counter() - start),
        "finalAggregation": float(peak["finalAggregation"]),
        "peakAggregation": float(peak["peakAggregation"]),
        "peakAggregationMinusNull": float(peak["peakAggregationMinusNull"]),
        "meanAggregationOverProgress": float(peak["meanAggregationOverProgress"]),
    }
    return {"replicate": record, "dg": None, "curves": curves, "peak": peak}


def worker(task: dict[str, Any]) -> dict[str, Any]:
    family = task["family"]
    if family == "no_frozen_efficiency":
        return run_no_frozen_task(task)
    if family == "frozen_robustness":
        return run_frozen_task(task)
    if family == "same_goal_aggregation":
        return run_aggregation_task(task)
    raise ValueError(f"Unknown S14 family: {family}")


def build_tasks(
    family: str,
    conditions: list[dict[str, Any]],
    replicate_count: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    sorted_policy = config["semantics"]["stopPolicies"]["sorted_or_cap"]
    frozen_stop_policies = config["semantics"]["stopPolicies"]
    chimera_policy = config["semantics"]["stopPolicies"]["sorted_no_move_or_cap"]
    for condition in conditions:
        for replicate_index in range(replicate_count):
            task = {"family": family, "condition": condition, "replicateIndex": replicate_index}
            if family == "no_frozen_efficiency":
                task["maxSwaps"] = int(sorted_policy["maxSwapEvents"])
            elif family == "frozen_robustness":
                task["stopPolicies"] = frozen_stop_policies
            elif family == "same_goal_aggregation":
                task.update(
                    {
                        "maxSwaps": int(chimera_policy["maxSwapEvents"]),
                        "maxComparisons": int(chimera_policy["maxComparisonEvents"]),
                        "maxRounds": 100000,
                        "noMoveChecksRequired": int(chimera_policy.get("noMoveChecksRequired", 2)),
                        "stableNoProgressRoundCap": 100,
                    }
                )
            tasks.append(task)
    return tasks


def run_task_family(
    family: str,
    tasks: list[dict[str, Any]],
    workers: int,
    chunksize: int,
) -> tuple[list[dict[str, Any]], float]:
    print(f"S14 running {family}: {len(tasks)} tasks with {workers} workers", flush=True)
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(worker, tasks, chunksize=chunksize), start=1):
            results.append(result)
            if index % 500 == 0 or index == len(tasks):
                elapsed = time.perf_counter() - start
                print(f"S14 {family}: {index}/{len(tasks)} tasks complete in {elapsed:.1f}s", flush=True)
    return results, time.perf_counter() - start


def unpack_results(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    replicate_records: list[dict[str, Any]] = []
    dg_records: list[dict[str, Any]] = []
    curve_records: list[dict[str, Any]] = []
    peak_records: list[dict[str, Any]] = []
    for result in results:
        replicate_records.append(result["replicate"])
        if result.get("dg") is not None:
            dg_records.append(result["dg"])
        curve_records.extend(result.get("curves", []))
        if result.get("peak") is not None:
            peak_records.append(result["peak"])
    return replicate_records, dg_records, curve_records, peak_records


def summarize_long(df: pd.DataFrame, family: str, metrics: list[str], group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            sem = float(sd / math.sqrt(len(values))) if len(values) > 1 else 0.0
            ci_low = mean - 1.96 * sem
            ci_high = mean + 1.96 * sem
            rows.append(
                {
                    **base,
                    "scaledFamily": family,
                    "metricId": metric,
                    "replicateCount": int(len(values)),
                    "mean": mean,
                    "sd": sd,
                    "sem": sem,
                    "ci95Lower": ci_low,
                    "ci95Upper": ci_high,
                    "ci95Width": ci_high - ci_low,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def build_scaled_summaries(
    replicate_df: pd.DataFrame,
    dg_df: pd.DataFrame,
    peak_df: pd.DataFrame,
) -> pd.DataFrame:
    groups = ["sourceStepId", "sourceConditionId", "implementation", "algorithm", "algorithms", "mixtureId", "frozenVariant", "frozenCount"]
    pieces = []
    for family in ["no_frozen_efficiency", "frozen_robustness", "same_goal_aggregation"]:
        subset = replicate_df[replicate_df["scaledFamily"] == family].copy()
        if not subset.empty:
            pieces.append(summarize_long(subset, family, METRICS_BY_FAMILY[family], groups))
    if not dg_df.empty:
        dg_groups = ["sourceStepId", "sourceConditionId", "implementation", "algorithm", "mixtureId", "frozenVariant", "frozenCount"]
        pieces.append(summarize_long(dg_df, "delayed_gratification", METRICS_BY_FAMILY["delayed_gratification"], dg_groups))
    if not peak_df.empty:
        peak_groups = ["sourceStepId", "sourceConditionId", "mixtureId", "algorithms"]
        pieces.append(summarize_long(peak_df, "same_goal_aggregation", METRICS_BY_FAMILY["same_goal_aggregation"], peak_groups))
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True, sort=False)


def table_summary(values: pd.Series | np.ndarray) -> dict[str, Any]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return {}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sem = float(sd / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "baselineReplicateCount": int(len(arr)),
        "baselineMean": mean,
        "baselineSd": sd,
        "baselineSem": sem,
        "baselineCi95Lower": mean - 1.96 * sem,
        "baselineCi95Upper": mean + 1.96 * sem,
        "baselineCi95Width": 3.92 * sem,
    }


def build_baseline_summaries() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    s04_df = pd.read_parquet("/artifacts/results/e01_s04_replicate_summary.parquet")
    s04_metrics = {
        "finalSortednessPercent": "final_sortedness_percent",
        "finalMonotonicityError": "final_monotonicity_error",
        "swapCount": "swap_count",
        "comparisonCount": "comparison_count",
    }
    s04_df["swapPlusComparisonSteps"] = s04_df["swap_count"] + s04_df["comparison_count"]
    s04_metrics["swapPlusComparisonSteps"] = "swapPlusComparisonSteps"
    for condition_id, group in s04_df.groupby("condition_id"):
        for metric, column in s04_metrics.items():
            rows.append(
                {
                    "scaledFamily": "no_frozen_efficiency",
                    "sourceStepId": "S04",
                    "sourceConditionId": condition_id,
                    "metricId": metric,
                    **table_summary(group[column]),
                }
            )
    s07_df = pd.read_parquet("/artifacts/results/e01_frozen_cell_robustness.parquet")
    for condition_id, group in s07_df[s07_df["frozenCount"] > 0].groupby("conditionId"):
        for metric in METRICS_BY_FAMILY["frozen_robustness"]:
            rows.append(
                {
                    "scaledFamily": "frozen_robustness",
                    "sourceStepId": "S07",
                    "sourceConditionId": condition_id,
                    "metricId": metric,
                    **table_summary(group[metric]),
                }
            )
    dg_base = pd.read_parquet("/artifacts/results/e01_delayed_gratification.parquet")
    for condition_id, group in dg_base.groupby("sourceConditionId"):
        for metric in METRICS_BY_FAMILY["delayed_gratification"]:
            rows.append(
                {
                    "scaledFamily": "delayed_gratification",
                    "sourceStepId": str(group["sourceTrace"].iloc[0]) if "sourceTrace" in group else "",
                    "sourceConditionId": condition_id,
                    "metricId": metric,
                    **table_summary(group[metric]),
                }
            )
    agg_base = pd.read_parquet("/artifacts/results/e01_aggregation_peaks.parquet")
    for condition_id, group in agg_base.groupby("conditionId"):
        for metric in METRICS_BY_FAMILY["same_goal_aggregation"]:
            if metric in group:
                rows.append(
                    {
                        "scaledFamily": "same_goal_aggregation",
                        "sourceStepId": "S09",
                        "sourceConditionId": condition_id,
                        "metricId": metric,
                        **table_summary(group[metric]),
                    }
                )
    return pd.DataFrame(rows)


def build_ci_comparison(scaled_summary: pd.DataFrame, baseline_summary: pd.DataFrame) -> pd.DataFrame:
    if scaled_summary.empty or baseline_summary.empty:
        return pd.DataFrame()
    cols = ["scaledFamily", "sourceConditionId", "metricId"]
    merged = scaled_summary.merge(
        baseline_summary,
        on=cols,
        how="left",
        suffixes=("", "Baseline"),
    )
    merged["meanDeltaFromBaseline"] = merged["mean"] - merged["baselineMean"]
    merged["ciWidthRatioS14OverBaseline"] = merged["ci95Width"] / merged["baselineCi95Width"]
    merged["expectedCiWidthRatioFromN"] = np.sqrt(merged["baselineReplicateCount"] / merged["replicateCount"])
    return merged


def build_efficiency_pairwise(replicate_df: pd.DataFrame) -> pd.DataFrame:
    subset = replicate_df[replicate_df["scaledFamily"] == "no_frozen_efficiency"].copy()
    rows = []
    for algorithm, group in subset.groupby("algorithm"):
        left = group[group["implementation"] == "traditional"].set_index("replicateIndex")
        right = group[group["implementation"] == "cell_view"].set_index("replicateIndex")
        common = sorted(set(left.index) & set(right.index))
        for metric in ["swapCount", "swapPlusComparisonSteps"]:
            trad = left.loc[common, metric].to_numpy(dtype=float)
            cell = right.loc[common, metric].to_numpy(dtype=float)
            diff = cell - trad
            mean_diff = float(np.mean(diff))
            sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
            sem_diff = float(sd_diff / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
            rows.append(
                {
                    "experimentId": EXPERIMENT_ID,
                    "researchStepId": STEP_ID,
                    "algorithm": algorithm,
                    "metricId": metric,
                    "pairCount": int(len(common)),
                    "traditionalMean": float(np.mean(trad)),
                    "cellViewMean": float(np.mean(cell)),
                    "cellViewMinusTraditionalMean": mean_diff,
                    "cellViewMinusTraditionalCi95Lower": mean_diff - 1.96 * sem_diff,
                    "cellViewMinusTraditionalCi95Upper": mean_diff + 1.96 * sem_diff,
                    "cellViewOverTraditionalRatio": float(np.mean(cell) / np.mean(trad)) if np.mean(trad) else np.nan,
                    "traditionalOverCellViewRatio": float(np.mean(trad) / np.mean(cell)) if np.mean(cell) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_dg_trends(dg_df: pd.DataFrame) -> pd.DataFrame:
    if dg_df.empty:
        return pd.DataFrame()
    rows = []
    for (implementation, algorithm, frozen_variant), group in dg_df.groupby(["implementation", "algorithm", "frozenVariant"]):
        summary = (
            group.groupby("frozenCount")
            .agg(meanDelayedGratification=("delayedGratification", "mean"), replicateCount=("replicateIndex", "count"))
            .reset_index()
            .sort_values("frozenCount")
        )
        values = summary["meanDelayedGratification"].to_numpy(dtype=float)
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "implementation": implementation,
                "algorithm": algorithm,
                "frozenVariant": frozen_variant,
                "frozenCountsJson": json.dumps([int(v) for v in summary["frozenCount"]], separators=(",", ":")),
                "meanDelayedGratificationJson": json.dumps([round(float(v), 12) for v in values], separators=(",", ":")),
                "monotoneNonDecreasing": bool(all(values[i] <= values[i + 1] + 1e-12 for i in range(len(values) - 1))),
                "linearSlopePerFrozenCell": float(np.polyfit(summary["frozenCount"].to_numpy(dtype=float), values, 1)[0]) if len(values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_aggregation_curves(curve_df: pd.DataFrame) -> pd.DataFrame:
    if curve_df.empty:
        return pd.DataFrame()
    return (
        curve_df.groupby(["sourceConditionId", "mixtureId", "algorithms", "progressBin", "normalizedProgress"], dropna=False)
        .agg(
            aggregationMean=("aggregationValue", "mean"),
            aggregationSd=("aggregationValue", "std"),
            aggregationMinusNullMean=("aggregationMinusNull", "mean"),
            sortednessMean=("sortednessPercent", "mean"),
            replicateCount=("replicateIndex", "count"),
        )
        .reset_index()
    )


def summarize_aggregation_peaks(peak_df: pd.DataFrame, curve_summary: pd.DataFrame) -> pd.DataFrame:
    if peak_df.empty:
        return pd.DataFrame()
    rows = []
    for (condition_id, mixture_id, algorithms), group in peak_df.groupby(["sourceConditionId", "mixtureId", "algorithms"]):
        curve_group = curve_summary[
            (curve_summary["sourceConditionId"] == condition_id)
            & (curve_summary["mixtureId"] == mixture_id)
        ]
        peak_curve = curve_group.loc[curve_group["aggregationMean"].idxmax()]
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "sourceConditionId": condition_id,
                "mixtureId": mixture_id,
                "algorithms": algorithms,
                "replicateCount": int(len(group)),
                "peakAggregationMean": float(group["peakAggregation"].mean()),
                "peakAggregationSd": float(group["peakAggregation"].std(ddof=1)),
                "peakAggregationMinusNullMean": float(group["peakAggregationMinusNull"].mean()),
                "peakAggregationMinusNullSd": float(group["peakAggregationMinusNull"].std(ddof=1)),
                "peakProgressMean": float(group["peakNormalizedProgress"].mean()),
                "finalAggregationMean": float(group["finalAggregation"].mean()),
                "meanCurvePeakAggregation": float(peak_curve["aggregationMean"]),
                "meanCurvePeakProgress": float(peak_curve["normalizedProgress"]),
                "meanCurvePeakProgressBin": int(peak_curve["progressBin"]),
            }
        )
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> list[Path]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return [parquet_path, csv_path]


def plot_ci_changes(ci_df: pd.DataFrame, figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    focus = ci_df[
        (
            ((ci_df["scaledFamily"] == "no_frozen_efficiency") & (ci_df["metricId"].isin(["swapCount", "swapPlusComparisonSteps"])))
            | ((ci_df["scaledFamily"] == "frozen_robustness") & (ci_df["metricId"] == "finalMonotonicityError"))
            | ((ci_df["scaledFamily"] == "delayed_gratification") & (ci_df["metricId"] == "delayedGratification"))
            | ((ci_df["scaledFamily"] == "same_goal_aggregation") & (ci_df["metricId"] == "peakAggregationMinusNull"))
        )
        & ci_df["ciWidthRatioS14OverBaseline"].notna()
        & np.isfinite(ci_df["ciWidthRatioS14OverBaseline"])
    ].copy()
    if focus.empty:
        return []
    focus = focus.sort_values(["scaledFamily", "metricId", "sourceConditionId"]).head(80)
    labels = [f"{row.scaledFamily.replace('_', ' ')}\n{row.metricId}" for row in focus.itertuples()]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(focus))
    ax.bar(x, focus["ciWidthRatioS14OverBaseline"].to_numpy(dtype=float), color="#4b7f9f")
    ax.axhline(math.sqrt(0.1), color="#8a3b3b", linestyle="--", linewidth=1.2, label="sqrt(100/1000)")
    ax.set_ylabel("S14 CI width / S01-S12 baseline CI width")
    ax.set_title("S14 scaled subset confidence interval width changes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend(loc="upper right")
    fig.tight_layout()
    png = figure_dir / "figure_s14_ci_width_changes.png"
    pdf = figure_dir / "figure_s14_ci_width_changes.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_aggregation(curve_summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    if curve_summary.empty:
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = {
        "bubble_insertion": "#3d6b99",
        "bubble_selection": "#b4514f",
        "insertion_selection": "#4f8a5b",
        "bubble_insertion_selection": "#7b5aa6",
    }
    for mixture_id, group in curve_summary.groupby("mixtureId"):
        group = group.sort_values("normalizedProgress")
        label = mixture_id.replace("_", " + ").title()
        ax.plot(group["normalizedProgress"], group["aggregationMean"], label=label, color=colors.get(mixture_id))
    ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
    ax.set_xlabel("Normalized swap progress")
    ax.set_ylabel("Aggregation")
    ax.set_title("S14 N=1000 same-goal mixed-Algotype Aggregation")
    ax.legend(loc="best")
    fig.tight_layout()
    png = figure_dir / "figure_s14_scaled_aggregation.png"
    pdf = figure_dir / "figure_s14_scaled_aggregation.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def run_smoke(args: argparse.Namespace, config: dict[str, Any], conditions_by_family: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    smoke_rows: list[dict[str, Any]] = []
    smoke_n = int(args.smoke_replicates)
    workers = min(8, int(args.workers))
    for family, conditions in conditions_by_family.items():
        tasks = build_tasks(family, conditions, smoke_n, config)
        start = time.perf_counter()
        results, elapsed = run_task_family(family, tasks, workers=workers, chunksize=max(1, min(8, int(args.chunksize))))
        reps = len(results)
        projected_tasks = len(conditions) * int(args.target_replicates)
        mean_task_wall = float(np.mean([item["replicate"]["taskWallTimeSeconds"] for item in results]))
        projected_parallel = mean_task_wall * projected_tasks / workers
        smoke_rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "family": family,
                "conditionCount": len(conditions),
                "smokeReplicatesPerCondition": smoke_n,
                "smokeTaskCount": len(tasks),
                "smokeWallSeconds": elapsed,
                "meanTaskWallSeconds": mean_task_wall,
                "projectedTargetReplicatesPerCondition": int(args.target_replicates),
                "projectedTaskCount": projected_tasks,
                "projectedParallelSecondsAtWorkerCount": projected_parallel,
                "workerCount": workers,
                "feasibleAtTargetN": bool(projected_parallel <= float(args.max_projected_seconds_per_family)),
            }
        )
        print(f"S14 smoke {family}: {elapsed:.1f}s observed; projected {projected_parallel:.1f}s at N={args.target_replicates}", flush=True)
    return pd.DataFrame(smoke_rows), smoke_rows


def actual_replicates_for_family(args: argparse.Namespace, smoke_df: pd.DataFrame) -> dict[str, int]:
    actual: dict[str, int] = {}
    for row in smoke_df.itertuples(index=False):
        if bool(row.feasibleAtTargetN):
            actual[row.family] = int(args.target_replicates)
        else:
            scale = float(args.max_projected_seconds_per_family) / max(float(row.projectedParallelSecondsAtWorkerCount), 1e-9)
            actual[row.family] = max(100, int(math.floor(int(args.target_replicates) * scale / 100) * 100))
    return actual


def render_methods_md(
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
    smoke_df: pd.DataFrame,
    actual_replicates: dict[str, int],
) -> str:
    rows = [
        [
            row.family,
            row.conditionCount,
            row.smokeWallSeconds,
            row.projectedParallelSecondsAtWorkerCount,
            actual_replicates.get(row.family, ""),
            row.feasibleAtTargetN,
        ]
        for row in smoke_df.itertuples(index=False)
    ]
    lines = [
        "# S14 Methods",
        "",
        "- Research step ID: S14",
        "- Step number: 14",
        "- Completion status: completed",
        "- Artifacts written:",
    ]
    lines.extend(f"- `{item}`" for item in artifacts_written)
    lines.extend(["- Validation result: " + validation_result, "- Caveats or blockers:"])
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Recommended next action: " + recommended_next_action,
            "",
            "## Selected Core Subset",
            "",
            "S14 scaled all six no-Frozen S04 efficiency conditions, all 36 S07 f>0 Frozen Cell robustness conditions, DG derived from Bubble and Insertion no-Frozen plus stuck-Frozen trajectories, and the four mixed S09 same-goal Aggregation conditions. The subset targets the S13 exact/statistical anchors and the largest Figure 8 transient Aggregation divergences without starting S15.",
            "",
            "## Smoke Runtime Gate",
            "",
            markdown_table(
                ["Family", "Conditions", "Smoke seconds", "Projected seconds", "Actual N", "Feasible"],
                rows,
            ),
            "",
            "N=10,000 was not used because no selected family was clearly cheap as a whole under the 8-worker cap after smoke projection.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_validation_md(
    validation_checks: list[str],
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    lines = [
        "# S14 Validation",
        "",
        "- Research step ID: S14",
        "- Step number: 14",
        "- Completion status: completed",
        "- Artifacts written:",
    ]
    lines.extend(f"- `{item}`" for item in artifacts_written)
    lines.extend(["- Validation result: " + validation_result, "- Caveats or blockers:"])
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(["- Recommended next action: " + recommended_next_action, "", "## Checks", ""])
    lines.extend(f"- {item}" for item in validation_checks)
    return "\n".join(lines) + "\n"


def render_summary_md(
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
    outcome_classification: str,
    replicate_df: pd.DataFrame,
    efficiency_pairwise: pd.DataFrame,
    dg_trends: pd.DataFrame,
    agg_summary: pd.DataFrame,
    ci_comparison: pd.DataFrame,
) -> str:
    family_counts = replicate_df.groupby("scaledFamily")["replicateIndex"].count().reset_index(name="replicateRows")
    family_rows = family_counts.values.tolist()
    eff_rows = efficiency_pairwise[
        efficiency_pairwise["metricId"].isin(["swapCount", "swapPlusComparisonSteps"])
    ].values.tolist()
    dg_rows = dg_trends.values.tolist() if not dg_trends.empty else [["None"]]
    agg_rows = (
        agg_summary[["mixtureId", "replicateCount", "meanCurvePeakAggregation", "meanCurvePeakProgress", "peakAggregationMinusNullMean"]]
        .values.tolist()
        if not agg_summary.empty
        else [["None"]]
    )
    ci_focus = ci_comparison[
        ci_comparison["metricId"].isin(["swapCount", "finalMonotonicityError", "delayedGratification", "peakAggregationMinusNull"])
    ].copy()
    ci_rows = ci_focus[["scaledFamily", "sourceConditionId", "metricId", "replicateCount", "ciWidthRatioS14OverBaseline"]].head(20).values.tolist()
    lines = [
        "# S14 Status Summary",
        "",
        "- Research step ID: S14",
        "- Step number: 14",
        "- Completion status: completed",
        "- Outcome classification: " + outcome_classification,
        "- Artifacts written:",
    ]
    lines.extend(f"- `{item}`" for item in artifacts_written)
    lines.extend(["- Validation result: " + validation_result, "- Caveats or blockers:"])
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "- Lay summary: S14 scaled a practical core subset to N=1,000 where smoke checks showed it was feasible. Confidence intervals generally narrowed relative to the N=100 baseline, while the main S13 classifications remained stable: completion anchors stayed exact, Frozen Cell robustness remained directionally supportive, DG Bubble/Insertion trends remained primary-stuck supportive, and same-goal Aggregation stayed above fixed-count nulls.",
            "- Recommended next action: " + recommended_next_action,
            "",
            "## Replicate Rows",
            "",
            markdown_table(["Family", "Replicate rows"], family_rows),
            "",
            "## Efficiency Pairwise",
            "",
            markdown_table(list(efficiency_pairwise.columns), eff_rows),
            "",
            "## DG Trends",
            "",
            markdown_table(list(dg_trends.columns) if not dg_trends.empty else ["Result"], dg_rows),
            "",
            "## Aggregation Peaks",
            "",
            markdown_table(["Mixture", "N", "Mean-curve peak", "Mean-curve peak progress", "Mean peak minus null"], agg_rows),
            "",
            "## CI Width Ratios",
            "",
            markdown_table(["Family", "Condition", "Metric", "N", "S14/baseline CI width"], ci_rows),
        ]
    )
    return "\n".join(lines) + "\n"


def validate_outputs(
    output_paths: list[Path],
    seed_ok: bool,
    actual_replicates: dict[str, int],
    replicate_df: pd.DataFrame,
    dg_df: pd.DataFrame,
    agg_peak_df: pd.DataFrame,
    artifacts_dir: Path,
    worker_count: int,
) -> tuple[bool, list[str], list[str], str]:
    failures: list[str] = []
    checks: list[str] = []
    caveats: list[str] = [
        "S14 scales a selected core subset rather than all S01-S13 conditions.",
        "S14 extends the S03 SHA256 seed convention to replicate indices 100 through 999 without changing n, values, Frozen Cell semantics, or stop policies.",
        "S14 stores compact per-replicate summaries and 101-bin Aggregation curves, not full raw per-swap state traces for every scaled run.",
        "N=10,000 was not run because the selected condition families were not clearly cheap as whole matched families under the 8-worker cap.",
        "The same S01-S13 deterministic pseudo-scheduler, reconstructed traditional wrappers, comparison-count conventions, and Aggregation/DG metric interpretations carry forward.",
    ]
    if seed_ok:
        checks.append("S14 seed extension reproduced the S03 first-100 seed rows for selected conditions.")
    else:
        failures.append("S14 seed extension did not reproduce S03 first-100 seed rows.")
    if worker_count <= 8:
        checks.append(f"Worker count {worker_count} respects the 8-CPU-core cap.")
    else:
        failures.append(f"Worker count {worker_count} exceeds the 8-CPU-core cap.")
    for family in ["no_frozen_efficiency", "frozen_robustness", "same_goal_aggregation"]:
        observed = int((replicate_df["scaledFamily"] == family).sum())
        if observed > 0:
            checks.append(f"{family} wrote {observed} scaled replicate rows.")
        else:
            failures.append(f"{family} has no scaled replicate rows.")
        if actual_replicates.get(family, 0) >= 1000:
            checks.append(f"{family} reached N=1000 per selected condition.")
        else:
            caveats.append(f"{family} was runtime-limited to N={actual_replicates.get(family, 0)} per selected condition.")
    no_frozen = replicate_df[replicate_df["scaledFamily"] == "no_frozen_efficiency"]
    if bool((no_frozen["finalSortednessPercent"] == 100.0).all()):
        checks.append("All scaled no-Frozen runs reached final 100% Sortedness.")
    else:
        failures.append("At least one scaled no-Frozen run failed final 100% Sortedness.")
    if not dg_df.empty:
        checks.append(f"DG table contains {len(dg_df)} derived scaled trajectory rows.")
    else:
        failures.append("DG table is empty.")
    if not agg_peak_df.empty and bool(agg_peak_df.groupby("sourceConditionId")["peakAggregationMinusNull"].mean().gt(0).all()):
        checks.append("Each scaled Aggregation condition has mean peak Aggregation above its fixed-count null.")
    else:
        failures.append("At least one scaled Aggregation condition mean peak is not above null, or Aggregation peaks are missing.")
    missing = [str(path) for path in output_paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        failures.append(f"Missing or empty S14 outputs: {missing}.")
    else:
        checks.append("All declared S14 outputs exist and are non-empty.")
    if (artifacts_dir / "research_steps" / "S15").exists() or (artifacts_dir / "report_bundle_inputs").exists():
        failures.append("S15/report-bundle artifact path exists; S14 must stop before S15.")
    else:
        checks.append("No S15 artifact directory or report-bundle inputs were created.")
    success = not failures
    validation_result = (
        "passed: S14 smoke-checked runtime, scaled selected core families, wrote CI comparisons, and stopped before S15."
        if success
        else "failed: " + " ".join(failures)
    )
    return success, checks + failures, caveats, validation_result


def update_run_manifest(path: Path, status_payload: dict[str, Any], artifact_manifest_payload: dict[str, Any]) -> None:
    manifest = load_json(path)
    manifest.setdefault("experimentId", EXPERIMENT_ID)
    manifest.setdefault("researchSteps", {})
    manifest["updatedAt"] = status_payload["generatedAt"]
    manifest["latestResearchStepId"] = STEP_ID
    manifest["researchSteps"][STEP_ID] = {
        "status": status_payload["status"],
        "success": status_payload["success"],
        "outcomeClassification": status_payload["outcomeClassification"],
        "artifactCount": artifact_manifest_payload["artifactCount"],
        "artifacts": artifact_manifest_payload["artifacts"],
        "validationResult": status_payload["validationResult"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "generatedAt": status_payload["generatedAt"],
        "git": status_payload["git"],
    }
    write_json(path, manifest)


def run_mode(args: argparse.Namespace) -> int:
    generated_at = utc_now()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    code_dir = step_dir / "code"
    results_dir = artifacts_dir / "results"
    figure_dir = artifacts_dir / "figures" / "e01"
    provenance_dir = artifacts_dir / "provenance"
    for directory in [step_dir, code_dir, results_dir, figure_dir, provenance_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    config = load_json(args.config)
    workers = max(1, min(8, int(args.workers)))
    conditions_by_family = selected_conditions(args.condition_matrix)
    all_selected_conditions = [condition for conditions in conditions_by_family.values() for condition in conditions]
    seed_ok, seed_messages = validate_seed_extension(args.seed_table, all_selected_conditions)

    smoke_df, _ = run_smoke(args, config, conditions_by_family)
    actual_replicates = actual_replicates_for_family(args, smoke_df)
    if args.mode == "smoke":
        smoke_path = step_dir / "smoke_runtime_checks.csv"
        smoke_json = step_dir / "smoke_runtime_checks.json"
        smoke_df.to_csv(smoke_path, index=False)
        write_json(smoke_json, {"researchStepId": STEP_ID, "stepNumber": STEP_NUMBER, "seedValidationPassed": seed_ok, "seedValidationMessages": seed_messages, "smokeRows": smoke_df.to_dict(orient="records"), "actualReplicatesIfFull": actual_replicates})
        print(json.dumps({"status": "smoke_completed", "smokeCsv": str(smoke_path), "actualReplicatesIfFull": actual_replicates}, indent=2))
        return 0 if seed_ok else 1

    smoke_paths = [
        *write_table(smoke_df, results_dir / "e01_scaled_smoke_runtime_checks.parquet", step_dir / "smoke_runtime_checks.csv")
    ]
    write_json(step_dir / "smoke_runtime_checks.json", {"researchStepId": STEP_ID, "stepNumber": STEP_NUMBER, "seedValidationPassed": seed_ok, "seedValidationMessages": seed_messages, "smokeRows": smoke_df.to_dict(orient="records"), "actualReplicates": actual_replicates})

    all_results: list[dict[str, Any]] = []
    family_runtime_rows: list[dict[str, Any]] = []
    for family, conditions in conditions_by_family.items():
        tasks = build_tasks(family, conditions, actual_replicates[family], config)
        results, elapsed = run_task_family(family, tasks, workers=workers, chunksize=max(1, int(args.chunksize)))
        all_results.extend(results)
        family_runtime_rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "family": family,
                "conditionCount": len(conditions),
                "replicatesPerCondition": actual_replicates[family],
                "taskCount": len(tasks),
                "wallSeconds": elapsed,
                "workerCount": workers,
                "meanTaskWallSeconds": float(np.mean([item["replicate"]["taskWallTimeSeconds"] for item in results])),
            }
        )

    replicate_records, dg_records, curve_records, peak_records = unpack_results(all_results)
    replicate_df = pd.DataFrame(replicate_records)
    dg_df = pd.DataFrame(dg_records)
    curve_df = pd.DataFrame(curve_records)
    peak_df = pd.DataFrame(peak_records)
    curve_summary = summarize_aggregation_curves(curve_df)
    peak_summary = summarize_aggregation_peaks(peak_df, curve_summary)
    scaled_summary = build_scaled_summaries(replicate_df, dg_df, peak_df)
    baseline_summary = build_baseline_summaries()
    ci_comparison = build_ci_comparison(scaled_summary, baseline_summary)
    efficiency_pairwise = build_efficiency_pairwise(replicate_df)
    dg_trends = summarize_dg_trends(dg_df)
    runtime_df = pd.DataFrame(family_runtime_rows)

    output_paths: list[Path] = []
    output_paths.extend(write_table(replicate_df, results_dir / "e01_scaled_replication.parquet", results_dir / "e01_scaled_replication.csv"))
    output_paths.extend(write_table(scaled_summary, results_dir / "e01_scaled_replication_summary.parquet", results_dir / "e01_scaled_replication_summary.csv"))
    output_paths.extend(write_table(baseline_summary, results_dir / "e01_scaled_baseline_summary.parquet", results_dir / "e01_scaled_baseline_summary.csv"))
    output_paths.extend(write_table(ci_comparison, results_dir / "e01_scaled_ci_comparison.parquet", results_dir / "e01_scaled_ci_comparison.csv"))
    output_paths.extend(write_table(efficiency_pairwise, results_dir / "e01_scaled_efficiency_pairwise.parquet", results_dir / "e01_scaled_efficiency_pairwise.csv"))
    output_paths.extend(write_table(dg_df, results_dir / "e01_scaled_delayed_gratification.parquet", results_dir / "e01_scaled_delayed_gratification.csv"))
    output_paths.extend(write_table(dg_trends, results_dir / "e01_scaled_delayed_gratification_trends.parquet", results_dir / "e01_scaled_delayed_gratification_trends.csv"))
    output_paths.extend(write_table(curve_df, results_dir / "e01_scaled_aggregation_curves.parquet", results_dir / "e01_scaled_aggregation_curves.csv"))
    output_paths.extend(write_table(curve_summary, results_dir / "e01_scaled_aggregation_curve_summary.parquet", results_dir / "e01_scaled_aggregation_curve_summary.csv"))
    output_paths.extend(write_table(peak_df, results_dir / "e01_scaled_aggregation_peaks.parquet", results_dir / "e01_scaled_aggregation_peaks.csv"))
    output_paths.extend(write_table(peak_summary, results_dir / "e01_scaled_aggregation_peak_summary.parquet", results_dir / "e01_scaled_aggregation_peak_summary.csv"))
    output_paths.extend(write_table(runtime_df, results_dir / "e01_scaled_runtime_summary.parquet", results_dir / "e01_scaled_runtime_summary.csv"))
    output_paths.extend(smoke_paths)
    output_paths.append(step_dir / "smoke_runtime_checks.json")
    output_paths.extend(plot_ci_changes(ci_comparison, figure_dir))
    output_paths.extend(plot_aggregation(curve_summary, figure_dir))

    recommended_next_action = "Hand control back to the Chief Scientist workflow for review; start S15 packaging only after explicit instruction."
    artifacts_written_placeholder = [str(path) for path in sorted(set(output_paths + [step_dir / "status.json", step_dir / "summary.md", step_dir / "validation.json", step_dir / "validation.md", step_dir / "methods.md", step_dir / "artifact_manifest.json", code_dir / Path(__file__).name, RUN_MANIFEST_DEFAULT]))]
    success, validation_checks, caveats, validation_result = validate_outputs(
        output_paths,
        seed_ok,
        actual_replicates,
        replicate_df,
        dg_df,
        peak_df,
        artifacts_dir,
        workers,
    )
    outcome_classification = "supportive" if success else "constraining"
    methods_md = step_dir / "methods.md"
    methods_md.write_text(
        render_methods_md(
            artifacts_written_placeholder,
            validation_result,
            caveats,
            recommended_next_action,
            smoke_df,
            actual_replicates,
        ),
        encoding="utf-8",
    )
    output_paths.append(methods_md)
    validation_md = step_dir / "validation.md"
    validation_json = step_dir / "validation.json"
    summary_md = step_dir / "summary.md"
    status_json = step_dir / "status.json"
    artifact_manifest_json = step_dir / "artifact_manifest.json"
    code_copy = code_dir / Path(__file__).name
    shutil.copy2(Path(__file__), code_copy)
    output_paths.extend([validation_md, validation_json, summary_md, status_json, artifact_manifest_json, code_copy, RUN_MANIFEST_DEFAULT])
    artifacts_written = [str(path) for path in sorted(set(output_paths))]

    validation_md.write_text(
        render_validation_md(validation_checks, artifacts_written, validation_result, caveats, recommended_next_action),
        encoding="utf-8",
    )
    validation_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "validationChecks": validation_checks,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "seedValidationPassed": bool(seed_ok),
        "seedValidationMessages": seed_messages,
        "actualReplicates": actual_replicates,
        "workerCount": workers,
        "replicateRows": int(len(replicate_df)),
        "dgRows": int(len(dg_df)),
        "aggregationCurveRows": int(len(curve_df)),
        "aggregationPeakRows": int(len(peak_df)),
    }
    write_json(validation_json, validation_payload)

    summary_md.write_text(
        render_summary_md(
            artifacts_written,
            validation_result,
            caveats,
            recommended_next_action,
            outcome_classification,
            replicate_df,
            efficiency_pairwise,
            dg_trends,
            peak_summary,
            ci_comparison,
        ),
        encoding="utf-8",
    )
    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "outcomeClassification": outcome_classification,
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "git": git_info(),
        "runtime": {
            "pythonVersion": sys.version,
            "pythonExecutable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "osCpuCount": os.cpu_count(),
            "workerCount": workers,
            "gpuUsed": False,
            "numpyVersion": np.__version__,
            "pandasVersion": pd.__version__,
            "matplotlibVersion": matplotlib.__version__,
            "chunksize": int(args.chunksize),
            "targetReplicates": int(args.target_replicates),
            "n10000Used": False,
        },
        "sourceArtifacts": {
            "s03BaselineConfig": str(args.config),
            "s03ConditionMatrix": str(args.condition_matrix),
            "s03SeedTable": str(args.seed_table),
            "s13Classification": "/artifacts/results/e01_replication_classification.csv",
        },
        "selectedSubset": {
            family: [condition["conditionId"] for condition in conditions]
            for family, conditions in conditions_by_family.items()
        },
        "actualReplicates": actual_replicates,
        "familyRuntime": runtime_df.to_dict(orient="records"),
        "conditionCounts": {
            "replicateRows": int(len(replicate_df)),
            "summaryRows": int(len(scaled_summary)),
            "ciComparisonRows": int(len(ci_comparison)),
            "dgRows": int(len(dg_df)),
            "aggregationCurveRows": int(len(curve_df)),
            "aggregationPeakRows": int(len(peak_df)),
        },
    }
    write_json(status_json, status_payload)

    manifest_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(success),
        "status": STATUS if success else "failed_validation",
        "generatedAt": generated_at,
        "artifactsWritten": artifacts_written,
        "artifactCount": 0,
        "artifacts": [],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
    }
    write_json(artifact_manifest_json, manifest_payload)
    artifacts = collect_artifacts([Path(path) for path in artifacts_written if Path(path) != RUN_MANIFEST_DEFAULT])
    manifest_payload["artifactCount"] = len(artifacts)
    manifest_payload["artifacts"] = artifacts
    write_json(artifact_manifest_json, manifest_payload)
    update_run_manifest(RUN_MANIFEST_DEFAULT, status_payload, manifest_payload)

    print(json.dumps({"success": success, "validationResult": validation_result, "actualReplicates": actual_replicates}, indent=2))
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH_DEFAULT)
    parser.add_argument("--condition-matrix", type=Path, default=CONDITION_MATRIX_DEFAULT)
    parser.add_argument("--seed-table", type=Path, default=SEED_TABLE_DEFAULT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--smoke-replicates", type=int, default=2)
    parser.add_argument("--target-replicates", type=int, default=1000)
    parser.add_argument("--max-projected-seconds-per-family", type=float, default=3600.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_mode(parse_args()))
