#!/usr/bin/env python3
"""Execute E02 S09 alternative metric sensitivity analysis.

S09 computes tie-aware distance-to-target metrics for sorting states, applies
them to S07 full trajectories and S01/E01 final-state context rows, joins S08
DG-null context where available, and writes metric sensitivity artifacts.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from e02_deterministic_simulator import initial_values_from_seed, monotonicity_error, sortedness_percent, sortedness_raw


EXPERIMENT_ID = "E02"
STEP_ID = "S09"
STEP_NUMBER = 9
DEFAULT_S01_SUMMARY = Path("/artifacts/results/e02_s01_replicate_summary.parquet")
DEFAULT_S07_SOURCES = Path("/artifacts/results/e02_local_move_null_sources.parquet")
DEFAULT_S07_NULLS = Path("/artifacts/results/e02_local_move_nulls.parquet")
DEFAULT_S07_TRACE = Path("/artifacts/traces/e02/S07/e02_local_move_null_trace_events.parquet")
DEFAULT_S08_DG_SOURCE_SUMMARY = Path("/artifacts/results/e02_dg_source_summary.parquet")
DEFAULT_E01_S04_SUMMARY = Path("/previous-artifacts/E01/results/e01_s04_replicate_summary.parquet")
DEFAULT_E01_S04_TRACE = Path("/previous-artifacts/E01/traces/e01/S04/trajectory_events.parquet")
DEFAULT_E01_FROZEN = Path("/previous-artifacts/E01/results/e01_frozen_cell_robustness.parquet")
DEFAULT_E01_SAME_GOAL = Path("/previous-artifacts/E01/results/e01_same_goal_chimeras.parquet")
DEFAULT_E01_DUPLICATE = Path("/previous-artifacts/E01/results/e01_duplicate_value_chimeras.parquet")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "args": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "ok": proc.returncode == 0,
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {"args": args, "returncode": None, "stdout": "", "stderr": repr(exc), "ok": False}


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["ok"] else "unknown",
        "branch": branch["stdout"].strip() if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"].strip(),
        "remote": remote["stdout"].strip(),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), separators=(",", ":"))


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            if abs(value) >= 1000 or (0 < abs(value) < 0.001):
                return f"{value:.3g}"
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def parse_json_array(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [int(item) for item in json.loads(text)]
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if pd.isna(value):
        return []
    raise TypeError(f"cannot parse array from {type(value)}")


def direction_key(value: int, direction: str) -> int:
    if direction == "increasing":
        return int(value)
    if direction == "decreasing":
        return -int(value)
    raise ValueError(f"unknown direction: {direction}")


def target_position_assignment(values: list[int] | np.ndarray, direction: str = "increasing") -> np.ndarray:
    """Assign each occurrence to a valid sorted target slot, preserving ties.

    Equal values are matched to their sorted target positions in left-to-right
    occurrence order, making already nondecreasing duplicate arrays distance 0.
    """

    values_list = [int(value) for value in values]
    if direction == "increasing":
        sorted_values = sorted(values_list)
    elif direction == "decreasing":
        sorted_values = sorted(values_list, reverse=True)
    else:
        raise ValueError(f"unknown direction: {direction}")
    target_slots: dict[int, deque[int]] = defaultdict(deque)
    for index, value in enumerate(sorted_values):
        target_slots[int(value)].append(index)
    assigned: list[int] = []
    for value in values_list:
        assigned.append(target_slots[int(value)].popleft())
    return np.asarray(assigned, dtype=np.int64)


def tie_aware_inversion_count(values: list[int] | np.ndarray, direction: str = "increasing") -> int:
    values_list = [direction_key(int(value), direction) for value in values]
    inversions = 0
    for left_index, left in enumerate(values_list):
        for right in values_list[left_index + 1 :]:
            if left > right:
                inversions += 1
    return int(inversions)


def comparable_pair_count(values: list[int] | np.ndarray) -> int:
    values_list = [int(value) for value in values]
    total = len(values_list) * (len(values_list) - 1) // 2
    tie_pairs = 0
    counts: dict[int, int] = defaultdict(int)
    for value in values_list:
        counts[int(value)] += 1
    for count in counts.values():
        tie_pairs += count * (count - 1) // 2
    return int(total - tie_pairs)


def longest_nondecreasing_subsequence_length(values: list[int] | np.ndarray, direction: str = "increasing") -> int:
    transformed = [direction_key(int(value), direction) for value in values]
    tails: list[int] = []
    for value in transformed:
        position = bisect.bisect_right(tails, value)
        if position == len(tails):
            tails.append(value)
        else:
            tails[position] = value
    return len(tails)


def target_displacements(values: list[int] | np.ndarray, direction: str = "increasing") -> np.ndarray:
    assigned = target_position_assignment(values, direction=direction)
    positions = np.arange(len(assigned), dtype=np.int64)
    return assigned - positions


def reverse_target_footrule(values: list[int] | np.ndarray, direction: str = "increasing") -> int:
    values_list = [int(value) for value in values]
    if direction == "increasing":
        reverse_values = sorted(values_list, reverse=True)
    else:
        reverse_values = sorted(values_list)
    return int(np.abs(target_displacements(reverse_values, direction=direction)).sum())


def state_distance_metrics(values: list[int] | np.ndarray, direction: str = "increasing") -> dict[str, Any]:
    values_list = [int(value) for value in values]
    n = len(values_list)
    raw_sortedness = sortedness_raw(values_list, direction=direction) if n else 0
    percent_sortedness = sortedness_percent(values_list, direction=direction) if n else 100.0
    inversions = tie_aware_inversion_count(values_list, direction=direction)
    comparable = comparable_pair_count(values_list)
    displacement = target_displacements(values_list, direction=direction)
    footrule = int(np.abs(displacement).sum())
    squared = int(np.square(displacement).sum())
    reverse_footrule = reverse_target_footrule(values_list, direction=direction)
    reverse_values = sorted(values_list, reverse=(direction == "increasing"))
    reverse_squared = int(np.square(target_displacements(reverse_values, direction=direction)).sum())
    lnds = longest_nondecreasing_subsequence_length(values_list, direction=direction)
    edit_distance = max(0, n - lnds)
    unique_count = len(set(values_list))
    return {
        "n": int(n),
        "hasDuplicates": bool(unique_count < n),
        "uniqueValueCount": int(unique_count),
        "sortednessRawCount": int(raw_sortedness),
        "sortednessPercent": float(percent_sortedness),
        "sortednessDistanceNormalized": float(1.0 - (percent_sortedness / 100.0)),
        "monotonicityError": int(monotonicity_error(values_list, direction=direction)) if n else 0,
        "inversionCount": int(inversions),
        "comparablePairCount": int(comparable),
        "inversionDistanceNormalized": float(inversions / comparable) if comparable else 0.0,
        "kendallTauDistanceNormalized": float(inversions / comparable) if comparable else 0.0,
        "spearmanFootruleDistance": int(footrule),
        "spearmanFootruleDistanceNormalized": float(footrule / reverse_footrule) if reverse_footrule else 0.0,
        "spearmanSquaredDistance": int(squared),
        "spearmanSquaredDistanceNormalized": float(squared / reverse_squared) if reverse_squared else 0.0,
        "earthMoverPositionDistance": float(footrule / n) if n else 0.0,
        "earthMoverPositionDistanceNormalized": float(footrule / reverse_footrule) if reverse_footrule else 0.0,
        "editDistanceToTargetOrder": int(edit_distance),
        "editDistanceToTargetOrderNormalized": float(edit_distance / n) if n else 0.0,
    }


def prefixed_metrics(prefix: str, values: list[int], direction: str = "increasing") -> dict[str, Any]:
    metrics = state_distance_metrics(values, direction=direction)
    return {f"{prefix}{key[0].upper()}{key[1:]}": value for key, value in metrics.items()}


def trajectory_curvature_metrics(states: list[list[int]]) -> dict[str, Any]:
    if len(states) < 2:
        return {
            "trajectoryStateCount": int(len(states)),
            "trajectoryTurnCount": 0,
            "trajectoryPathLengthL2": 0.0,
            "trajectoryDirectDistanceL2": 0.0,
            "trajectoryExcessPathRatio": 0.0,
            "stateSpaceCurvatureTotalTurnRadians": 0.0,
            "stateSpaceCurvatureMeanTurnRadians": 0.0,
            "stateSpaceCurvatureMeanOneMinusCosine": 0.0,
        }
    arr = np.asarray(states, dtype=float)
    deltas = np.diff(arr, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    path_length = float(norms.sum())
    direct = float(np.linalg.norm(arr[-1] - arr[0]))
    angles: list[float] = []
    one_minus_cosines: list[float] = []
    for left, right, left_norm, right_norm in zip(deltas[:-1], deltas[1:], norms[:-1], norms[1:]):
        if left_norm <= 0 or right_norm <= 0:
            continue
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
        cosine = max(-1.0, min(1.0, cosine))
        angles.append(float(math.acos(cosine)))
        one_minus_cosines.append(float(1.0 - cosine))
    return {
        "trajectoryStateCount": int(len(states)),
        "trajectoryTurnCount": int(len(angles)),
        "trajectoryPathLengthL2": path_length,
        "trajectoryDirectDistanceL2": direct,
        "trajectoryExcessPathRatio": float((path_length / direct) - 1.0) if direct > 0 else 0.0,
        "stateSpaceCurvatureTotalTurnRadians": float(np.sum(angles)) if angles else 0.0,
        "stateSpaceCurvatureMeanTurnRadians": float(np.mean(angles)) if angles else 0.0,
        "stateSpaceCurvatureMeanOneMinusCosine": float(np.mean(one_minus_cosines)) if one_minus_cosines else 0.0,
    }


def run_metric_unit_cases() -> tuple[bool, pd.DataFrame]:
    cases: list[dict[str, Any]] = []

    def record(case_id: str, observed: Any, expected: Any, passed: bool) -> None:
        cases.append({"caseId": case_id, "observed": observed, "expected": expected, "passed": bool(passed)})

    sorted_unique = state_distance_metrics([1, 2, 3, 4])
    record("sorted_unique_kendall", sorted_unique["kendallTauDistanceNormalized"], 0.0, math.isclose(sorted_unique["kendallTauDistanceNormalized"], 0.0))
    record("sorted_unique_edit", sorted_unique["editDistanceToTargetOrder"], 0, sorted_unique["editDistanceToTargetOrder"] == 0)
    record("sorted_unique_emd", sorted_unique["earthMoverPositionDistance"], 0.0, math.isclose(sorted_unique["earthMoverPositionDistance"], 0.0))

    reverse_unique = state_distance_metrics([4, 3, 2, 1])
    record("reverse_unique_inversions", reverse_unique["inversionCount"], 6, reverse_unique["inversionCount"] == 6)
    record("reverse_unique_kendall", reverse_unique["kendallTauDistanceNormalized"], 1.0, math.isclose(reverse_unique["kendallTauDistanceNormalized"], 1.0))
    record("reverse_unique_edit", reverse_unique["editDistanceToTargetOrder"], 3, reverse_unique["editDistanceToTargetOrder"] == 3)
    record("reverse_unique_footrule_norm", reverse_unique["spearmanFootruleDistanceNormalized"], 1.0, math.isclose(reverse_unique["spearmanFootruleDistanceNormalized"], 1.0))

    sorted_duplicate = state_distance_metrics([1, 1, 2, 2])
    record("sorted_duplicate_zero_inversions", sorted_duplicate["inversionCount"], 0, sorted_duplicate["inversionCount"] == 0)
    record("sorted_duplicate_zero_displacement", sorted_duplicate["spearmanFootruleDistance"], 0, sorted_duplicate["spearmanFootruleDistance"] == 0)
    record("sorted_duplicate_has_duplicates", sorted_duplicate["hasDuplicates"], True, sorted_duplicate["hasDuplicates"] is True)

    unsorted_duplicate = state_distance_metrics([2, 1, 2, 1])
    record("unsorted_duplicate_inversions", unsorted_duplicate["inversionCount"], 3, unsorted_duplicate["inversionCount"] == 3)
    record("unsorted_duplicate_comparable_pairs", unsorted_duplicate["comparablePairCount"], 4, unsorted_duplicate["comparablePairCount"] == 4)
    record("unsorted_duplicate_kendall", unsorted_duplicate["kendallTauDistanceNormalized"], 0.75, math.isclose(unsorted_duplicate["kendallTauDistanceNormalized"], 0.75))
    record("unsorted_duplicate_footrule", unsorted_duplicate["spearmanFootruleDistance"], 6, unsorted_duplicate["spearmanFootruleDistance"] == 6)
    record("unsorted_duplicate_edit", unsorted_duplicate["editDistanceToTargetOrder"], 2, unsorted_duplicate["editDistanceToTargetOrder"] == 2)

    curve = trajectory_curvature_metrics([[3, 1, 2], [1, 3, 2], [1, 2, 3]])
    record("trajectory_curvature_turn_count", curve["trajectoryTurnCount"], 1, curve["trajectoryTurnCount"] == 1)
    record("trajectory_curvature_positive", curve["stateSpaceCurvatureTotalTurnRadians"] > 0, True, curve["stateSpaceCurvatureTotalTurnRadians"] > 0)

    df = pd.DataFrame(cases)
    return bool(df["passed"].all()), df


def base_record(
    *,
    metric_source: str,
    metric_scope: str,
    source_run_id: str,
    source_context: str,
    source_condition_id: str,
    algorithm: str,
    implementation: str,
    input_profile: str,
    n: int,
    replicate_index: int | None,
    replicate_number: int | None,
    frozen_variant: str,
    frozen_count: int,
    null_model: str,
    null_replicate_index: int | None,
    input_permutation_seed: int | None,
    source_trajectory_hash: str,
    trajectory_metrics_available: bool,
) -> dict[str, Any]:
    return {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "metricSource": metric_source,
        "metricScope": metric_scope,
        "sourceRunId": source_run_id,
        "sourceContext": source_context,
        "sourceConditionId": source_condition_id,
        "algorithm": algorithm,
        "implementation": implementation,
        "inputProfile": input_profile,
        "n": int(n),
        "replicateIndex": None if replicate_index is None else int(replicate_index),
        "replicateNumber": None if replicate_number is None else int(replicate_number),
        "frozenVariant": frozen_variant,
        "frozenCount": int(frozen_count),
        "nullModel": null_model,
        "nullReplicateIndex": None if null_replicate_index is None else int(null_replicate_index),
        "inputPermutationSeed": None if input_permutation_seed is None else int(input_permutation_seed),
        "sourceTrajectoryHash": source_trajectory_hash,
        "trajectoryMetricsAvailable": bool(trajectory_metrics_available),
    }


def add_state_metrics(record: dict[str, Any], initial_values: list[int], final_values: list[int]) -> dict[str, Any]:
    initial = prefixed_metrics("initial", initial_values)
    final = prefixed_metrics("final", final_values)
    record.update(initial)
    record.update(final)
    for key in [
        "SortednessDistanceNormalized",
        "KendallTauDistanceNormalized",
        "InversionDistanceNormalized",
        "SpearmanFootruleDistanceNormalized",
        "SpearmanSquaredDistanceNormalized",
        "EarthMoverPositionDistanceNormalized",
        "EditDistanceToTargetOrderNormalized",
    ]:
        record[f"improvement{key}"] = float(record[f"initial{key}"] - record[f"final{key}"])
    record["valueCountPreservedForMetric"] = sorted(initial_values) == sorted(final_values)
    return record


def s07_meta_maps(s07_sources_path: Path, s07_nulls_path: Path) -> tuple[dict[str, pd.Series], dict[tuple[str, str, int], pd.Series]]:
    sources = pd.read_parquet(s07_sources_path)
    nulls = pd.read_parquet(s07_nulls_path)
    source_map = {str(row.sourceRunId): row for row in sources.itertuples(index=False)}
    null_map = {
        (str(row.sourceRunId), str(row.nullModel), int(row.nullReplicateIndex)): row
        for row in nulls.itertuples(index=False)
    }
    return source_map, null_map


def load_s08_dg_context(s08_dg_source_summary_path: Path) -> dict[str, dict[str, Any]]:
    if not s08_dg_source_summary_path.exists():
        return {}
    df = pd.read_parquet(s08_dg_source_summary_path)
    wanted = [
        "observedDelayedGratification",
        "nullDgMean",
        "nullDgSd",
        "observedMinusNullDgMean",
        "empiricalPHighDg",
        "empiricalPLowDg",
    ]
    return {
        str(row["sourceRunId"]): {f"s08{col[0].upper()}{col[1:]}": row[col] for col in wanted if col in row}
        for _, row in df.iterrows()
    }


def compute_s07_full_trace_metrics(
    s07_trace_path: Path,
    s07_sources_path: Path,
    s07_nulls_path: Path,
    s08_dg_source_summary_path: Path,
) -> pd.DataFrame:
    trace = pd.read_parquet(s07_trace_path)
    trace = trace.sort_values(["sourceRunId", "nullModel", "nullReplicateIndex", "eventIndex"]).reset_index(drop=True)
    source_map, null_map = s07_meta_maps(s07_sources_path, s07_nulls_path)
    dg_context = load_s08_dg_context(s08_dg_source_summary_path)
    records: list[dict[str, Any]] = []

    group_cols = ["sourceRunId", "nullModel", "nullReplicateIndex"]
    for (source_run_id, null_model, null_replicate_index), group in trace.groupby(group_cols, sort=False):
        states = [parse_json_array(value) for value in group["values"].tolist()]
        first = group.iloc[0]
        if str(null_model) == "real_policy_source":
            meta = source_map[str(source_run_id)]
            algorithm = str(meta.algorithm)
            input_seed = None if pd.isna(meta.inputPermutationSeed) else int(meta.inputPermutationSeed)
            replicate_index = int(meta.replicateIndex)
            replicate_number = int(meta.replicateNumber)
            source_condition_id = str(meta.sourceConditionId)
            source_context = str(meta.sourceContext)
            metric_source = "S07_full_trace_real_policy"
        else:
            meta = null_map[(str(source_run_id), str(null_model), int(null_replicate_index))]
            algorithm = str(meta.algorithm)
            input_seed = None if pd.isna(meta.inputPermutationSeed) else int(meta.inputPermutationSeed)
            replicate_index = int(meta.replicateIndex)
            replicate_number = int(meta.replicateNumber)
            source_condition_id = str(meta.sourceConditionId)
            source_context = str(meta.sourceContext)
            metric_source = "S07_full_trace_local_null"

        record = base_record(
            metric_source=metric_source,
            metric_scope="full_trajectory",
            source_run_id=str(source_run_id),
            source_context=source_context,
            source_condition_id=source_condition_id,
            algorithm=algorithm,
            implementation="deterministic_event_simulator",
            input_profile="unique_1_100",
            n=int(first["values"].count(",") + 1) if isinstance(first["values"], str) and first["values"] != "[]" else len(states[0]),
            replicate_index=replicate_index,
            replicate_number=replicate_number,
            frozen_variant=str(first.frozenVariant),
            frozen_count=int(first.frozenCount),
            null_model=str(null_model),
            null_replicate_index=int(null_replicate_index),
            input_permutation_seed=input_seed,
            source_trajectory_hash=str(first.sourceTrajectoryHash),
            trajectory_metrics_available=True,
        )
        add_state_metrics(record, states[0], states[-1])
        record.update(trajectory_curvature_metrics(states))
        record["eventCountForMetric"] = int(len(states))
        record["swapCountForMetric"] = int(group["swapCount"].max())
        record["s08DgContextAvailable"] = str(null_model) == "real_policy_source" and str(source_run_id) in dg_context
        if str(null_model) == "real_policy_source" and str(source_run_id) in dg_context:
            record.update(dg_context[str(source_run_id)])
        records.append(record)

    return pd.DataFrame(records)


def add_summary_rows_from_arrays(
    *,
    df: pd.DataFrame,
    metric_source: str,
    source_context: str,
    condition_col: str,
    algorithm_col: str,
    implementation_col: str,
    input_profile_col: str,
    n_col: str,
    replicate_index_col: str,
    replicate_number_col: str,
    frozen_variant_col: str,
    frozen_count_col: str,
    initial_values_col: str | None,
    final_values_col: str,
    input_seed_col: str | None,
    source_id_prefix: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        n = int(row_dict.get(n_col, 0) or 0)
        input_profile = str(row_dict.get(input_profile_col, "unknown"))
        if initial_values_col is None:
            input_seed = row_dict.get(input_seed_col) if input_seed_col else None
            if input_seed is None or pd.isna(input_seed):
                continue
            initial_values = initial_values_from_seed(int(input_seed), n=n, profile=input_profile)
        else:
            initial_values = parse_json_array(row_dict.get(initial_values_col))
        final_values = parse_json_array(row_dict.get(final_values_col))
        if not initial_values or not final_values:
            continue
        source_condition_id = str(row_dict.get(condition_col, "unknown"))
        replicate_index = None if replicate_index_col not in row_dict or pd.isna(row_dict.get(replicate_index_col)) else int(row_dict.get(replicate_index_col))
        replicate_number = None if replicate_number_col not in row_dict or pd.isna(row_dict.get(replicate_number_col)) else int(row_dict.get(replicate_number_col))
        source_run_id = f"{source_id_prefix}_{source_condition_id}_rep{replicate_index if replicate_index is not None else 0:03d}"
        algorithm = str(row_dict.get(algorithm_col, "unknown")).replace(";", "+")
        record = base_record(
            metric_source=metric_source,
            metric_scope="final_state_only",
            source_run_id=source_run_id,
            source_context=source_context,
            source_condition_id=source_condition_id,
            algorithm=algorithm,
            implementation=str(row_dict.get(implementation_col, "unknown")),
            input_profile=input_profile,
            n=len(initial_values),
            replicate_index=replicate_index,
            replicate_number=replicate_number,
            frozen_variant=str(row_dict.get(frozen_variant_col, "none")),
            frozen_count=int(0 if pd.isna(row_dict.get(frozen_count_col, 0)) else row_dict.get(frozen_count_col, 0)),
            null_model="not_applicable",
            null_replicate_index=None,
            input_permutation_seed=None
            if input_seed_col is None or pd.isna(row_dict.get(input_seed_col))
            else int(row_dict.get(input_seed_col)),
            source_trajectory_hash=str(row_dict.get("sourceTrajectoryHash", "")),
            trajectory_metrics_available=False,
        )
        add_state_metrics(record, initial_values, final_values)
        records.append(record)
    return records


def compute_context_final_state_metrics(
    s01_summary_path: Path,
    e01_s04_summary_path: Path,
    e01_frozen_path: Path,
    e01_same_goal_path: Path,
    e01_duplicate_path: Path,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    if s01_summary_path.exists():
        s01 = pd.read_parquet(s01_summary_path)
        records.extend(
            add_summary_rows_from_arrays(
                df=s01,
                metric_source="S01_deterministic_summary",
                source_context="s01_deterministic_baseline",
                condition_col="condition_id",
                algorithm_col="algorithm",
                implementation_col="implementation",
                input_profile_col="input_profile",
                n_col="n",
                replicate_index_col="replicate_index",
                replicate_number_col="replicate_number",
                frozen_variant_col="frozen_variant",
                frozen_count_col="frozen_count",
                initial_values_col="initial_values_json",
                final_values_col="final_values_json",
                input_seed_col="input_permutation_seed",
                source_id_prefix="S01",
            )
        )

    if e01_s04_summary_path.exists():
        e01_s04 = pd.read_parquet(e01_s04_summary_path)
        records.extend(
            add_summary_rows_from_arrays(
                df=e01_s04,
                metric_source="E01_S04_summary",
                source_context="e01_baseline",
                condition_col="condition_id",
                algorithm_col="algorithm",
                implementation_col="implementation",
                input_profile_col="input_profile",
                n_col="n",
                replicate_index_col="replicate_index",
                replicate_number_col="replicate_number",
                frozen_variant_col="input_profile",
                frozen_count_col="replicate_index",
                initial_values_col="initial_values_json",
                final_values_col="final_values_json",
                input_seed_col="input_permutation_seed",
                source_id_prefix="E01S04",
            )
        )
        for record in records:
            if record["metricSource"] == "E01_S04_summary":
                record["frozenVariant"] = "none"
                record["frozenCount"] = 0

    if e01_frozen_path.exists():
        e01_frozen = pd.read_parquet(e01_frozen_path)
        records.extend(
            add_summary_rows_from_arrays(
                df=e01_frozen,
                metric_source="E01_S07_frozen_summary",
                source_context="e01_frozen_cell",
                condition_col="conditionId",
                algorithm_col="algorithm",
                implementation_col="implementation",
                input_profile_col="inputProfile",
                n_col="n",
                replicate_index_col="replicateIndex",
                replicate_number_col="replicateNumber",
                frozen_variant_col="frozenVariant",
                frozen_count_col="frozenCount",
                initial_values_col=None,
                final_values_col="finalValues",
                input_seed_col="inputPermutationSeed",
                source_id_prefix="E01S07",
            )
        )

    if e01_same_goal_path.exists():
        e01_same = pd.read_parquet(e01_same_goal_path)
        records.extend(
            add_summary_rows_from_arrays(
                df=e01_same,
                metric_source="E01_S09_same_goal_summary",
                source_context="e01_same_goal_chimera",
                condition_col="conditionId",
                algorithm_col="algorithms",
                implementation_col="implementation",
                input_profile_col="inputProfile",
                n_col="n",
                replicate_index_col="replicateIndex",
                replicate_number_col="replicateNumber",
                frozen_variant_col="inputProfile",
                frozen_count_col="replicateIndex",
                initial_values_col="initialValuesJson",
                final_values_col="finalValuesJson",
                input_seed_col="inputPermutationSeed",
                source_id_prefix="E01S09",
            )
        )
        for record in records:
            if record["metricSource"] == "E01_S09_same_goal_summary":
                record["frozenVariant"] = "none"
                record["frozenCount"] = 0

    if e01_duplicate_path.exists():
        e01_dup = pd.read_parquet(e01_duplicate_path)
        records.extend(
            add_summary_rows_from_arrays(
                df=e01_dup,
                metric_source="E01_S11_duplicate_summary",
                source_context="e01_duplicate_chimera",
                condition_col="conditionId",
                algorithm_col="algorithms",
                implementation_col="implementation",
                input_profile_col="inputProfile",
                n_col="n",
                replicate_index_col="replicateIndex",
                replicate_number_col="replicateNumber",
                frozen_variant_col="inputProfile",
                frozen_count_col="replicateIndex",
                initial_values_col="initialValuesJson",
                final_values_col="finalValuesJson",
                input_seed_col="inputPermutationSeed",
                source_id_prefix="E01S11",
            )
        )
        for record in records:
            if record["metricSource"] == "E01_S11_duplicate_summary":
                record["frozenVariant"] = "none"
                record["frozenCount"] = 0

    return pd.DataFrame(records)


def compute_e01_trace_context(e01_trace_path: Path) -> pd.DataFrame:
    if not e01_trace_path.exists():
        return pd.DataFrame()
    trace = pd.read_parquet(e01_trace_path)
    records: list[dict[str, Any]] = []
    group_cols = ["condition_id", "implementation", "algorithm", "replicate_index", "replicate_number"]
    for key, group in trace.groupby(group_cols, sort=False):
        condition_id, implementation, algorithm, replicate_index, replicate_number = key
        ordered = group.sort_values("event_index")
        sortedness = ordered["sortedness_percent"].astype(float).to_numpy()
        deltas = np.diff(sortedness)
        signs = np.sign(deltas[np.abs(deltas) > 1e-12])
        sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
        records.append(
            {
                "experimentId": "E01",
                "researchStepId": "S04",
                "conditionId": condition_id,
                "implementation": implementation,
                "algorithm": algorithm,
                "replicateIndex": int(replicate_index),
                "replicateNumber": int(replicate_number),
                "trajectoryEventCount": int(len(ordered)),
                "initialSortednessPercent": float(sortedness[0]) if len(sortedness) else None,
                "finalSortednessPercent": float(sortedness[-1]) if len(sortedness) else None,
                "sortednessTotalVariation": float(np.abs(deltas).sum()) if len(deltas) else 0.0,
                "sortednessNetGain": float(sortedness[-1] - sortedness[0]) if len(sortedness) else 0.0,
                "sortednessSignChanges": sign_changes,
                "fullValueArraysAvailable": False,
                "contextUse": "baseline sortedness trajectory context only; full alternative metrics require value arrays",
            }
        )
    return pd.DataFrame(records)


def summarize_metrics(metric_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["metricSource", "metricScope", "sourceContext", "nullModel"]
    rows: list[dict[str, Any]] = []
    for key, group in metric_df.groupby(group_cols, dropna=False):
        metric_source, metric_scope, source_context, null_model = key
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "metricSource": metric_source,
                "metricScope": metric_scope,
                "sourceContext": source_context,
                "nullModel": null_model,
                "rowCount": int(len(group)),
                "trajectoryMetricsAvailableRate": float(group["trajectoryMetricsAvailable"].mean()),
                "duplicateRowRate": float(group["finalHasDuplicates"].mean()),
                "finalSortednessDistanceMean": float(group["finalSortednessDistanceNormalized"].mean()),
                "finalKendallTauDistanceMean": float(group["finalKendallTauDistanceNormalized"].mean()),
                "finalSpearmanFootruleDistanceMean": float(group["finalSpearmanFootruleDistanceNormalized"].mean()),
                "finalEarthMoverPositionDistanceMean": float(group["finalEarthMoverPositionDistanceNormalized"].mean()),
                "finalEditDistanceMean": float(group["finalEditDistanceToTargetOrderNormalized"].mean()),
                "stateSpaceCurvatureMeanTurnRadiansMean": float(group["stateSpaceCurvatureMeanTurnRadians"].dropna().mean())
                if "stateSpaceCurvatureMeanTurnRadians" in group
                else None,
            }
        )
    return pd.DataFrame(rows)


def sensitivity_summary(metric_df: pd.DataFrame) -> pd.DataFrame:
    s07 = metric_df[metric_df["metricSource"].str.startswith("S07_full_trace")].copy()
    real = s07[s07["nullModel"].eq("real_policy_source")].set_index("sourceRunId")
    nulls = s07[~s07["nullModel"].eq("real_policy_source")].copy()
    rows: list[dict[str, Any]] = []
    for _, null_row in nulls.iterrows():
        real_row = real.loc[null_row["sourceRunId"]]
        rows.append(
            {
                "sourceRunId": null_row["sourceRunId"],
                "sourceContext": null_row["sourceContext"],
                "sourceConditionId": null_row["sourceConditionId"],
                "algorithm": null_row["algorithm"],
                "frozenVariant": null_row["frozenVariant"],
                "frozenCount": int(null_row["frozenCount"]),
                "nullModel": null_row["nullModel"],
                "nullReplicateIndex": int(null_row["nullReplicateIndex"]),
                "realMinusNullFinalSortednessPercent": float(
                    real_row["finalSortednessPercent"] - null_row["finalSortednessPercent"]
                ),
                "nullMinusRealFinalKendallTauDistance": float(
                    null_row["finalKendallTauDistanceNormalized"] - real_row["finalKendallTauDistanceNormalized"]
                ),
                "nullMinusRealFinalEarthMoverDistance": float(
                    null_row["finalEarthMoverPositionDistanceNormalized"] - real_row["finalEarthMoverPositionDistanceNormalized"]
                ),
                "nullMinusRealFinalEditDistance": float(
                    null_row["finalEditDistanceToTargetOrderNormalized"] - real_row["finalEditDistanceToTargetOrderNormalized"]
                ),
                "nullMinusRealCurvatureMeanTurn": float(
                    null_row["stateSpaceCurvatureMeanTurnRadians"] - real_row["stateSpaceCurvatureMeanTurnRadians"]
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    summary = (
        comparison.groupby(["sourceContext", "nullModel"], dropna=False)
        .agg(
            comparisonRows=("sourceRunId", "size"),
            sourceRunCount=("sourceRunId", "nunique"),
            realMinusNullFinalSortednessPercentMean=("realMinusNullFinalSortednessPercent", "mean"),
            realBetterSortednessFraction=("realMinusNullFinalSortednessPercent", lambda s: float(np.mean(np.asarray(s) > 0))),
            nullMinusRealFinalKendallTauDistanceMean=("nullMinusRealFinalKendallTauDistance", "mean"),
            realBetterKendallFraction=("nullMinusRealFinalKendallTauDistance", lambda s: float(np.mean(np.asarray(s) > 0))),
            nullMinusRealFinalEarthMoverDistanceMean=("nullMinusRealFinalEarthMoverDistance", "mean"),
            realBetterEarthMoverFraction=("nullMinusRealFinalEarthMoverDistance", lambda s: float(np.mean(np.asarray(s) > 0))),
            nullMinusRealFinalEditDistanceMean=("nullMinusRealFinalEditDistance", "mean"),
            realBetterEditFraction=("nullMinusRealFinalEditDistance", lambda s: float(np.mean(np.asarray(s) > 0))),
            nullMinusRealCurvatureMeanTurnMean=("nullMinusRealCurvatureMeanTurn", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "researchStepId", STEP_ID)
    summary.insert(0, "experimentId", EXPERIMENT_ID)
    return summary


def correlation_table(metric_df: pd.DataFrame) -> pd.DataFrame:
    corr_cols = [
        "finalSortednessDistanceNormalized",
        "finalKendallTauDistanceNormalized",
        "finalSpearmanFootruleDistanceNormalized",
        "finalSpearmanSquaredDistanceNormalized",
        "finalEarthMoverPositionDistanceNormalized",
        "finalEditDistanceToTargetOrderNormalized",
    ]
    available = metric_df[corr_cols].apply(pd.to_numeric, errors="coerce")
    corr = available.corr(method="spearman")
    rows: list[dict[str, Any]] = []
    for left in corr_cols:
        for right in corr_cols:
            rows.append(
                {
                    "experimentId": EXPERIMENT_ID,
                    "researchStepId": STEP_ID,
                    "correlationScope": "all_metric_rows",
                    "method": "spearman",
                    "metricA": left,
                    "metricB": right,
                    "correlation": None if pd.isna(corr.loc[left, right]) else float(corr.loc[left, right]),
                }
            )
    return pd.DataFrame(rows)


def diagnostics_table(
    metric_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    e01_trace_context_df: pd.DataFrame,
    unit_passed: bool,
) -> pd.DataFrame:
    s07_real = metric_df[metric_df["metricSource"].eq("S07_full_trace_real_policy")]
    s07_null = metric_df[metric_df["metricSource"].eq("S07_full_trace_local_null")]
    checks = [
        ("unit_cases_passed", unit_passed, "Sorted, reverse, duplicate, and curvature toy unit cases passed."),
        ("metric_table_nonempty", len(metric_df) > 0, "Alternative metrics table is nonempty."),
        ("s07_real_rows_present", len(s07_real) == 39, "S07 real-source full trajectory rows are present."),
        ("s07_null_rows_present", len(s07_null) == 3900, "S07 local-null full trajectory rows are present."),
        (
            "s08_dg_context_joined",
            int(s07_real["s08DgContextAvailable"].eq(True).sum()) == 39,
            "S08 DG context joined onto all S07 real-source rows.",
        ),
        (
            "duplicate_rows_present",
            int(metric_df["finalHasDuplicates"].fillna(False).sum()) > 0,
            "Duplicate-array rows are represented for tie-aware metric checks.",
        ),
        (
            "e01_trace_context_present",
            len(e01_trace_context_df) > 0,
            "E01 baseline trace context was loaded and summarized.",
        ),
        (
            "all_value_counts_preserved",
            bool(metric_df["valueCountPreservedForMetric"].all()),
            "Initial/final arrays preserve value counts for all metric rows.",
        ),
        (
            "finite_core_metrics",
            bool(
                np.isfinite(
                    metric_df[
                        [
                            "finalKendallTauDistanceNormalized",
                            "finalEarthMoverPositionDistanceNormalized",
                            "finalEditDistanceToTargetOrderNormalized",
                        ]
                    ].to_numpy(dtype=float)
                ).all()
            ),
            "Core final alternative metrics are finite.",
        ),
        ("summary_nonempty", len(summary_df) > 0, "Metric summary table is nonempty."),
        ("sensitivity_nonempty", len(sensitivity_df) > 0, "S07 real-vs-null sensitivity summary is nonempty."),
        ("correlation_nonempty", corr_df["correlation"].notna().any(), "Correlation table contains finite values."),
    ]
    return pd.DataFrame(
        [
            {"experimentId": EXPERIMENT_ID, "researchStepId": STEP_ID, "check": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in checks
        ]
    )


def validate_outputs(diagnostics_df: pd.DataFrame) -> dict[str, Any]:
    failures = diagnostics_df.loc[~diagnostics_df["passed"], "detail"].tolist()
    return {
        "success": not failures,
        "validationResult": "passed" if not failures else "failed",
        "checksPassed": int(diagnostics_df["passed"].sum()),
        "checksTotal": int(len(diagnostics_df)),
        "failures": failures,
    }


def write_tables(
    metric_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    e01_trace_context_df: pd.DataFrame,
    unit_df: pd.DataFrame,
    artifacts_dir: Path,
) -> dict[str, Path]:
    result_dir = artifacts_dir / "results"
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "alternative_metrics_parquet": result_dir / "e02_alternative_metrics.parquet",
        "alternative_metrics_csv": result_dir / "e02_alternative_metrics.csv",
        "summary_parquet": result_dir / "e02_alternative_metric_summary.parquet",
        "summary_csv": result_dir / "e02_alternative_metric_summary.csv",
        "sensitivity_parquet": result_dir / "e02_alternative_metric_sensitivity.parquet",
        "sensitivity_csv": result_dir / "e02_alternative_metric_sensitivity.csv",
        "correlations_parquet": result_dir / "e02_alternative_metric_correlations.parquet",
        "correlations_csv": result_dir / "e02_alternative_metric_correlations.csv",
        "diagnostics_parquet": result_dir / "e02_alternative_metric_diagnostics.parquet",
        "diagnostics_csv": result_dir / "e02_alternative_metric_diagnostics.csv",
        "e01_trace_context_parquet": result_dir / "e02_alternative_metric_e01_trace_context.parquet",
        "e01_trace_context_csv": result_dir / "e02_alternative_metric_e01_trace_context.csv",
        "unit_tests_json": step_dir / "unit_tests.json",
    }
    metric_df.to_parquet(paths["alternative_metrics_parquet"], index=False)
    metric_df.to_csv(paths["alternative_metrics_csv"], index=False)
    summary_df.to_parquet(paths["summary_parquet"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    sensitivity_df.to_parquet(paths["sensitivity_parquet"], index=False)
    sensitivity_df.to_csv(paths["sensitivity_csv"], index=False)
    corr_df.to_parquet(paths["correlations_parquet"], index=False)
    corr_df.to_csv(paths["correlations_csv"], index=False)
    diagnostics_df.to_parquet(paths["diagnostics_parquet"], index=False)
    diagnostics_df.to_csv(paths["diagnostics_csv"], index=False)
    e01_trace_context_df.to_parquet(paths["e01_trace_context_parquet"], index=False)
    e01_trace_context_df.to_csv(paths["e01_trace_context_csv"], index=False)
    write_json(paths["unit_tests_json"], {"researchStepId": STEP_ID, "passed": bool(unit_df["passed"].all()), "cases": unit_df.to_dict(orient="records")})
    return paths


def plot_correlation_heatmap(corr_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    matrix = corr_df.pivot(index="metricA", columns="metricB", values="correlation")
    labels = [label.replace("final", "").replace("DistanceNormalized", "").replace("ToTargetOrderNormalized", "Edit") for label in matrix.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Alternative final-distance metric correlations")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png = figure_dir / "e02_alternative_metrics_correlation_heatmap.png"
    pdf = figure_dir / "e02_alternative_metrics_correlation_heatmap.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_sensitivity(metric_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    s07 = metric_df[metric_df["metricSource"].str.startswith("S07_full_trace")].copy()
    real = s07[s07["nullModel"].eq("real_policy_source")].set_index("sourceRunId")
    nulls = s07[~s07["nullModel"].eq("real_policy_source")].copy()
    records: list[dict[str, Any]] = []
    for _, row in nulls.iterrows():
        real_row = real.loc[row["sourceRunId"]]
        records.append(
            {
                "nullModel": row["nullModel"],
                "sourceContext": row["sourceContext"],
                "nullMinusRealFinalKendall": row["finalKendallTauDistanceNormalized"]
                - real_row["finalKendallTauDistanceNormalized"],
                "nullMinusRealFinalEdit": row["finalEditDistanceToTargetOrderNormalized"]
                - real_row["finalEditDistanceToTargetOrderNormalized"],
                "nullMinusRealFinalEMD": row["finalEarthMoverPositionDistanceNormalized"]
                - real_row["finalEarthMoverPositionDistanceNormalized"],
            }
        )
    plot_df = pd.DataFrame(records)
    models = sorted(plot_df["nullModel"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    plot_cols = [
        ("nullMinusRealFinalKendall", "Kendall distance"),
        ("nullMinusRealFinalEdit", "Edit-order distance"),
        ("nullMinusRealFinalEMD", "Position EMD"),
    ]
    for ax, (col, title) in zip(axes, plot_cols):
        data = [plot_df.loc[plot_df["nullModel"].eq(model), col].to_numpy(dtype=float) for model in models]
        ax.boxplot(data, tick_labels=[model.replace("_", "\n") for model in models], showfliers=False)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("null minus real")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("S07 local null distance excess over real policies")
    fig.tight_layout()
    png = figure_dir / "e02_alternative_metrics_sensitivity.png"
    pdf = figure_dir / "e02_alternative_metrics_sensitivity.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.exists() and path.is_file():
            records.append({"path": str(path), "sha256": sha256_path(path), "sizeBytes": path.stat().st_size})
    return sorted(records, key=lambda row: row["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    paths = [
        (REPO_ROOT / "scripts" / "e02_s09_alternative_metrics.py", code_dir / "scripts" / "e02_s09_alternative_metrics.py"),
        (REPO_ROOT / "tests" / "test_e02_alternative_metrics.py", code_dir / "tests" / "test_e02_alternative_metrics.py"),
    ]
    copied: list[Path] = []
    for src, dst in paths:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def run_repo_tests(step_dir: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "tests.test_e02_alternative_metrics"]
    result = run_command(command, cwd=REPO_ROOT)
    log_path = step_dir / "repo_unit_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT:\n" + result["stdout"] + "\nSTDERR:\n" + result["stderr"],
        encoding="utf-8",
    )
    return {
        "command": command,
        "returncode": result["returncode"],
        "passed": bool(result["ok"]),
        "logPath": str(log_path),
    }


def outcome_classification(metric_df: pd.DataFrame, sensitivity_df: pd.DataFrame, validation_success: bool) -> str:
    if not validation_success:
        return "null"
    s07_real = metric_df[metric_df["metricSource"].eq("S07_full_trace_real_policy")]
    real_sorted_rate = float((s07_real["finalKendallTauDistanceNormalized"] == 0).mean()) if len(s07_real) else 0.0
    better_fraction = float(sensitivity_df["realBetterKendallFraction"].mean()) if len(sensitivity_df) else 0.0
    if real_sorted_rate == 1.0 and better_fraction > 0.5:
        return "supportive"
    return "constraining/contradictory"


def write_validation_report(step_dir: Path, diagnostics_df: pd.DataFrame, validation: dict[str, Any]) -> Path:
    lines = [
        "# E02 S09 Validation Report",
        "",
        "- Research step ID: S09",
        "- Completion status: completed" if validation["success"] else "- Completion status: failed",
        "- Artifacts written: alternative metric tables, E01 trace context, summaries, diagnostics, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {validation['validationResult']}",
        "- Caveats or blockers: E01 and S01 trace rows do not store full value arrays, so S09 computes full trajectory state-space metrics from S07 traces and final-state context metrics from S01/E01 summary arrays; E01 baseline traces are summarized as Sortedness-only context.",
        "- Recommended next action: stop before S10 for Chief Scientist review.",
        "",
        "## Checks",
    ]
    for row in diagnostics_df.itertuples(index=False):
        prefix = "passed" if row.passed else "failed"
        lines.append(f"- {prefix}: {row.detail}")
    path = step_dir / "validation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_reports_and_manifests(
    *,
    artifacts_dir: Path,
    started_at: float,
    metric_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    e01_trace_context_df: pd.DataFrame,
    table_paths: dict[str, Path],
    figure_paths: tuple[Path, ...],
    code_paths: list[Path],
    repo_tests: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Path]:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    step_dir.mkdir(parents=True, exist_ok=True)
    validation_path = write_validation_report(step_dir, diagnostics_df, validation)
    outcome = outcome_classification(metric_df, sensitivity_df, validation["success"])
    caveat = (
        "E01 and S01 trace rows lack full value arrays, so full trajectory state-space curvature is computed from S07 only; "
        "S01/E01 summary rows provide initial/final alternative metrics, and E01 traces provide Sortedness-only trajectory context. "
        "Tie-aware duplicate handling is validated on toy arrays and exercised on E01 duplicate-value chimera rows."
    )
    recommended = "Stop before S10 for Chief Scientist review; if accepted, proceed to S10 input-distribution variation."

    summary_rows = summary_df.sort_values(["metricSource", "sourceContext", "nullModel"]).head(12)
    summary_table = markdown_table(
        ["Metric source", "Context", "Null model", "Rows", "Final Kendall mean", "Final edit mean"],
        [
            [
                row.metricSource,
                row.sourceContext,
                row.nullModel,
                row.rowCount,
                row.finalKendallTauDistanceMean,
                row.finalEditDistanceMean,
            ]
            for row in summary_rows.itertuples(index=False)
        ],
    )

    sensitivity_table = markdown_table(
        [
            "Context",
            "Null model",
            "Rows",
            "Real better Kendall frac.",
            "Null-real Kendall mean",
            "Null-real edit mean",
        ],
        [
            [
                row.sourceContext,
                row.nullModel,
                row.comparisonRows,
                row.realBetterKendallFraction,
                row.nullMinusRealFinalKendallTauDistanceMean,
                row.nullMinusRealFinalEditDistanceMean,
            ]
            for row in sensitivity_df.sort_values(["sourceContext", "nullModel"]).itertuples(index=False)
        ],
    )

    summary_path = step_dir / "summary.md"
    summary_lines = [
        "# E02 S09 Summary",
        "",
        "- Research step ID: S09",
        "- Completion status: completed" if validation["success"] else "- Completion status: failed",
        "- Artifacts written: `$ARTIFACTS_DIR/results/e02_alternative_metrics.parquet`, metric summaries, sensitivity and correlation tables, E01 trace context, diagnostics, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {validation['validationResult']}",
        f"- Caveats or blockers: {caveat}",
        "- Lay summary: S09 remeasured sorting progress with metrics that look beyond adjacent-pair Sortedness. No-Frozen and same-goal chimera sources scored as sorted under Kendall/inversion, edit-order, and position-transport distances, but representative Frozen Cell sources retained residual distance under these stricter order metrics. Duplicate arrays are handled by assigning equal values to any valid target slots, so sorted duplicate arrays score zero distance.",
        f"- Recommended next action: {recommended}",
        f"- Outcome classification: {outcome}",
        "",
        "## Run Counts",
        "",
        markdown_table(
            ["Item", "Count"],
            [
                ["Alternative metric rows", len(metric_df)],
                ["S07 full-trajectory rows", int(metric_df["metricSource"].str.startswith("S07_full_trace").sum())],
                ["Rows with duplicate values", int(metric_df["finalHasDuplicates"].fillna(False).sum())],
                ["E01 sortedness-only trace context rows", len(e01_trace_context_df)],
                ["S08 DG context joins", int(metric_df["s08DgContextAvailable"].eq(True).sum()) if "s08DgContextAvailable" in metric_df else 0],
            ],
        ),
        "",
        "## Metric Summary",
        "",
        summary_table,
        "",
        "## S07 Real-Vs-Null Sensitivity",
        "",
        sensitivity_table,
        "",
        "## Validation",
    ]
    for row in diagnostics_df.itertuples(index=False):
        summary_lines.append(f"- {'passed' if row.passed else 'failed'}: {row.detail}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "provenance" / "run_manifest.json"
    artifact_inputs = list(table_paths.values()) + list(figure_paths) + code_paths + [
        validation_path,
        summary_path,
        status_path,
        artifact_manifest_path,
        run_manifest_path,
        step_dir / "repo_unit_test_log.txt",
    ]
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"] and repo_tests["passed"]),
        "status": "completed" if validation["success"] and repo_tests["passed"] else "failed",
        "artifactsWritten": [str(path) for path in sorted(set(artifact_inputs), key=str) if path.exists()],
        "validationResult": validation["validationResult"] if repo_tests["passed"] else "failed",
        "caveatsOrBlockers": caveat,
        "recommendedNextAction": recommended,
        "outcomeClassification": outcome,
        "repoTests": repo_tests,
        "runSeconds": time.perf_counter() - started_at,
        "metricRows": int(len(metric_df)),
        "sensitivityRows": int(len(sensitivity_df)),
        "e01TraceContextRows": int(len(e01_trace_context_df)),
    }
    write_json(status_path, status_payload)

    all_artifacts = collect_artifacts(list(table_paths.values()) + list(figure_paths) + code_paths + [validation_path, summary_path, status_path, step_dir / "repo_unit_test_log.txt"])
    manifest_payload = {
        "schema": "eidosoma.research_step_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "git": get_git_metadata(),
        "inputs": {
            "s01Summary": str(DEFAULT_S01_SUMMARY),
            "s07Sources": str(DEFAULT_S07_SOURCES),
            "s07Nulls": str(DEFAULT_S07_NULLS),
            "s07Trace": str(DEFAULT_S07_TRACE),
            "s08DgSourceSummary": str(DEFAULT_S08_DG_SOURCE_SUMMARY),
            "e01S04Summary": str(DEFAULT_E01_S04_SUMMARY),
            "e01S04Trace": str(DEFAULT_E01_S04_TRACE),
            "e01Frozen": str(DEFAULT_E01_FROZEN),
            "e01SameGoal": str(DEFAULT_E01_SAME_GOAL),
            "e01Duplicate": str(DEFAULT_E01_DUPLICATE),
        },
        "outputs": all_artifacts,
        "validation": validation,
        "repoTests": repo_tests,
        "metricRows": int(len(metric_df)),
        "summaryRows": int(len(summary_df)),
        "sensitivityRows": int(len(sensitivity_df)),
        "correlationRows": int(len(corr_df)),
        "e01TraceContextRows": int(len(e01_trace_context_df)),
        "outcomeClassification": outcome,
    }
    write_json(artifact_manifest_path, manifest_payload)

    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "latestResearchStepId": STEP_ID,
        "generatedAt": utc_now(),
        "startedAt": utc_now(),
        "statusPath": str(status_path),
        "git": get_git_metadata(),
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpuCount": os.cpu_count(),
        },
        "packageVersions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "runs": [
            {
                "researchStepId": STEP_ID,
                "status": status_payload["status"],
                "runSeconds": status_payload["runSeconds"],
                "metricRows": int(len(metric_df)),
            }
        ],
        "artifacts": all_artifacts,
    }
    write_json(run_manifest_path, run_manifest)

    return {
        "summary": summary_path,
        "status": status_path,
        "validation": validation_path,
        "artifact_manifest": artifact_manifest_path,
        "run_manifest": run_manifest_path,
    }


def run_s09(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    figure_dir = artifacts_dir / "figures" / "e02"
    step_dir.mkdir(parents=True, exist_ok=True)

    unit_passed, unit_df = run_metric_unit_cases()
    s07_metric_df = compute_s07_full_trace_metrics(
        args.s07_trace,
        args.s07_sources,
        args.s07_nulls,
        args.s08_dg_source_summary,
    )
    context_df = compute_context_final_state_metrics(
        args.s01_summary,
        args.e01_s04_summary,
        args.e01_frozen,
        args.e01_same_goal,
        args.e01_duplicate,
    )
    metric_df = pd.concat([s07_metric_df, context_df], ignore_index=True, sort=False)
    e01_trace_context_df = compute_e01_trace_context(args.e01_s04_trace)
    summary_df = summarize_metrics(metric_df)
    sensitivity_df = sensitivity_summary(metric_df)
    corr_df = correlation_table(metric_df)
    diagnostics_df = diagnostics_table(metric_df, summary_df, sensitivity_df, corr_df, e01_trace_context_df, unit_passed)
    validation = validate_outputs(diagnostics_df)
    table_paths = write_tables(metric_df, summary_df, sensitivity_df, corr_df, diagnostics_df, e01_trace_context_df, unit_df, artifacts_dir)
    figure_paths = plot_correlation_heatmap(corr_df, figure_dir) + plot_sensitivity(metric_df, figure_dir)
    code_paths = copy_code_artifacts(step_dir)
    repo_tests = run_repo_tests(step_dir)
    if not repo_tests["passed"]:
        validation["success"] = False
        validation["validationResult"] = "failed"
        validation.setdefault("failures", []).append("Repository S09 unit tests failed.")
    write_reports_and_manifests(
        artifacts_dir=artifacts_dir,
        started_at=started_at,
        metric_df=metric_df,
        summary_df=summary_df,
        sensitivity_df=sensitivity_df,
        corr_df=corr_df,
        diagnostics_df=diagnostics_df,
        e01_trace_context_df=e01_trace_context_df,
        table_paths=table_paths,
        figure_paths=figure_paths,
        code_paths=code_paths,
        repo_tests=repo_tests,
        validation=validation,
    )
    return 0 if validation["success"] and repo_tests["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--s01-summary", type=Path, default=DEFAULT_S01_SUMMARY)
    parser.add_argument("--s07-sources", type=Path, default=DEFAULT_S07_SOURCES)
    parser.add_argument("--s07-nulls", type=Path, default=DEFAULT_S07_NULLS)
    parser.add_argument("--s07-trace", type=Path, default=DEFAULT_S07_TRACE)
    parser.add_argument("--s08-dg-source-summary", type=Path, default=DEFAULT_S08_DG_SOURCE_SUMMARY)
    parser.add_argument("--e01-s04-summary", type=Path, default=DEFAULT_E01_S04_SUMMARY)
    parser.add_argument("--e01-s04-trace", type=Path, default=DEFAULT_E01_S04_TRACE)
    parser.add_argument("--e01-frozen", type=Path, default=DEFAULT_E01_FROZEN)
    parser.add_argument("--e01-same-goal", type=Path, default=DEFAULT_E01_SAME_GOAL)
    parser.add_argument("--e01-duplicate", type=Path, default=DEFAULT_E01_DUPLICATE)
    return parser.parse_args()


def main() -> int:
    return run_s09(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
