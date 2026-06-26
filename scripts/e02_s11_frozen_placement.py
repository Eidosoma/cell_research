#!/usr/bin/env python3
"""Execute E02 S11 Frozen Cell placement sensitivity tests.

S11 varies where f=2 Frozen Cells are placed while keeping initial arrays,
policies, and simulator semantics fixed. It uses the S01 deterministic event
simulator, the S08 E01-compatible DG helper, and the S09 tie-aware metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
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

from e02_deterministic_simulator import DeterministicEventSimulator, aggregation, state_hash
from scripts.e02_s08_dg_nulls import delayed_gratification_from_sortedness
from scripts.e02_s09_alternative_metrics import prefixed_metrics
from scripts.e02_s10_input_distributions import compact_json, input_values_for_profile, json_ready, markdown_table, sha256_path, stable_seed, write_json


EXPERIMENT_ID = "E02"
STEP_ID = "S11"
STEP_NUMBER = 11
ALGORITHMS = ["bubble", "insertion", "selection"]
FROZEN_VARIANTS = ["passive", "stuck"]
FROZEN_PLACEMENT_CATEGORIES = [
    "array_ends",
    "center",
    "random",
    "high_value",
    "low_value",
    "clustered",
    "evenly_spaced",
]
CONTROL_PLACEMENT_CATEGORY = "no_frozen_control"
DEFAULT_E01_FROZEN_PATH = Path("/previous-artifacts/E01/results/e01_frozen_cell_robustness.parquet")


@dataclass(frozen=True)
class PlacementCondition:
    algorithm: str
    frozen_variant: str
    placement_category: str
    frozen_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
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


def placement_conditions() -> list[PlacementCondition]:
    conditions: list[PlacementCondition] = []
    for algorithm in ALGORITHMS:
        conditions.append(
            PlacementCondition(
                algorithm=algorithm,
                frozen_variant="none",
                placement_category=CONTROL_PLACEMENT_CATEGORY,
                frozen_count=0,
            )
        )
        for frozen_variant in FROZEN_VARIANTS:
            for category in FROZEN_PLACEMENT_CATEGORIES:
                conditions.append(
                    PlacementCondition(
                        algorithm=algorithm,
                        frozen_variant=frozen_variant,
                        placement_category=category,
                        frozen_count=2,
                    )
                )
    return conditions


def placement_label(category: str) -> str:
    labels = {
        CONTROL_PLACEMENT_CATEGORY: "No-Frozen control",
        "array_ends": "Array ends",
        "center": "Center",
        "random": "Random positions",
        "high_value": "Highest values",
        "low_value": "Lowest values",
        "clustered": "Clustered adjacent",
        "evenly_spaced": "Evenly spaced",
    }
    return labels[category]


def placement_family(category: str) -> str:
    if category == CONTROL_PLACEMENT_CATEGORY:
        return "control"
    if category in {"high_value", "low_value"}:
        return "value_selected"
    return "position_selected"


def placement_confound_note(category: str) -> str:
    if category == "high_value":
        return (
            "Highest-value Frozen Cells deliberately confound value identity with their current positions; "
            "S11 controls the initial array across placement categories and logs value/position quantiles, "
            "but does not separate value effects from position effects."
        )
    if category == "low_value":
        return (
            "Lowest-value Frozen Cells deliberately confound value identity with their current positions; "
            "S11 controls the initial array across placement categories and logs value/position quantiles, "
            "but does not separate value effects from position effects."
        )
    if category == CONTROL_PLACEMENT_CATEGORY:
        return "No Frozen Cells; included as a matched same-initial-array competence baseline."
    return "Position-selected placement with matched initial arrays across all S11 placement categories."


def value_position_confound(category: str) -> bool:
    return category in {"high_value", "low_value"}


def _unique_spaced_positions(n: int, frozen_count: int) -> list[int]:
    raw = np.linspace(0, n - 1, frozen_count + 2)[1:-1]
    positions: list[int] = []
    for value in raw:
        candidate = int(round(float(value)))
        candidate = max(0, min(n - 1, candidate))
        while candidate in positions and candidate < n - 1:
            candidate += 1
        while candidate in positions and candidate > 0:
            candidate -= 1
        positions.append(candidate)
    return sorted(positions)


def frozen_positions_for_category(category: str, initial_values: list[int], seed: int, frozen_count: int = 2) -> list[int]:
    n = len(initial_values)
    if frozen_count == 0:
        if category != CONTROL_PLACEMENT_CATEGORY:
            raise ValueError("Only no_frozen_control may have frozen_count=0")
        return []
    if frozen_count < 1 or frozen_count > n:
        raise ValueError(f"invalid frozen_count={frozen_count} for n={n}")
    rng = np.random.default_rng(int(seed))
    if category == "array_ends":
        if frozen_count != 2:
            raise ValueError("array_ends is defined for f=2 in S11")
        return [0, n - 1]
    if category == "center":
        midpoint = (n - 1) / 2.0
        return sorted(sorted(range(n), key=lambda pos: (abs(pos - midpoint), pos))[:frozen_count])
    if category == "random":
        return sorted(int(value) for value in rng.choice(np.arange(n), size=frozen_count, replace=False))
    if category == "high_value":
        return sorted(sorted(range(n), key=lambda pos: (-initial_values[pos], pos))[:frozen_count])
    if category == "low_value":
        return sorted(sorted(range(n), key=lambda pos: (initial_values[pos], pos))[:frozen_count])
    if category == "clustered":
        start = int(rng.integers(0, n - frozen_count + 1))
        return list(range(start, start + frozen_count))
    if category == "evenly_spaced":
        return _unique_spaced_positions(n, frozen_count)
    raise ValueError(f"unknown placement category: {category}")


def validate_placement_category(category: str, positions: list[int], initial_values: list[int], frozen_count: int) -> bool:
    n = len(initial_values)
    if len(positions) != frozen_count:
        return False
    if len(set(positions)) != len(positions):
        return False
    if any(position < 0 or position >= n for position in positions):
        return False
    if category == CONTROL_PLACEMENT_CATEGORY:
        return frozen_count == 0 and positions == []
    if category == "array_ends":
        return positions == [0, n - 1]
    if category == "center":
        midpoint = (n - 1) / 2.0
        expected = sorted(sorted(range(n), key=lambda pos: (abs(pos - midpoint), pos))[:frozen_count])
        return positions == expected
    if category == "random":
        return frozen_count > 0
    if category == "high_value":
        expected = sorted(sorted(range(n), key=lambda pos: (-initial_values[pos], pos))[:frozen_count])
        return positions == expected
    if category == "low_value":
        expected = sorted(sorted(range(n), key=lambda pos: (initial_values[pos], pos))[:frozen_count])
        return positions == expected
    if category == "clustered":
        return max(positions) - min(positions) == frozen_count - 1
    if category == "evenly_spaced":
        if frozen_count < 2:
            return True
        gaps = np.diff(positions)
        return bool(np.min(gaps) >= max(1, int(math.floor(n / (frozen_count + 2)))))
    return False


def frozen_value_metadata(initial_values: list[int], frozen_positions: list[int]) -> dict[str, Any]:
    n = len(initial_values)
    ascending_positions = sorted(range(n), key=lambda pos: (initial_values[pos], pos))
    rank_by_position = {position: rank + 1 for rank, position in enumerate(ascending_positions)}
    values = [int(initial_values[position]) for position in frozen_positions]
    ranks = [int(rank_by_position[position]) for position in frozen_positions]
    value_quantiles = [float((rank - 1) / (n - 1)) if n > 1 else 1.0 for rank in ranks]
    position_quantiles = [float(position / (n - 1)) if n > 1 else 1.0 for position in frozen_positions]
    return {
        "frozenInitialValues": values,
        "frozenInitialValueRanksAscending": ranks,
        "frozenInitialValueQuantiles": value_quantiles,
        "frozenInitialPositionQuantiles": position_quantiles,
        "frozenMeanInitialValueQuantile": float(np.mean(value_quantiles)) if value_quantiles else None,
        "frozenMeanInitialPositionQuantile": float(np.mean(position_quantiles)) if position_quantiles else None,
    }


def add_identity_logging(sim: DeterministicEventSimulator, frozen_cell_ids: list[int]) -> None:
    """Attach frozen identity snapshots to each simulator trace row."""

    def enrich_last_row() -> None:
        positions = sim.position_by_cell_id()
        id_positions = {str(cell_id): int(positions[cell_id]) for cell_id in frozen_cell_ids}
        sim.trace_rows[-1]["frozen_cell_ids_json"] = compact_json(frozen_cell_ids)
        sim.trace_rows[-1]["frozen_cell_id_positions_json"] = compact_json(id_positions)

    enrich_last_row()
    original_append = sim._append_trace_row

    def wrapped_append(**kwargs: Any) -> None:
        original_append(**kwargs)
        enrich_last_row()

    sim._append_trace_row = wrapped_append  # type: ignore[method-assign]


def sortedness_trajectory(result: Any) -> np.ndarray:
    return np.asarray([float(row["sortedness_percent"]) for row in result.trace_rows], dtype=float)


def aggregation_trajectory(result: Any) -> np.ndarray:
    values: list[float] = []
    for row in result.trace_rows:
        algotypes = json.loads(row["algotypes_json"])
        values.append(float(aggregation(algotypes)))
    return np.asarray(values, dtype=float)


def trajectory_proxy_metrics(result: Any) -> dict[str, Any]:
    sortedness = sortedness_trajectory(result)
    aggregation_values = aggregation_trajectory(result)
    deltas = np.diff(sortedness)
    signs = np.sign(deltas[np.abs(deltas) > 1e-12])
    sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
    dg = delayed_gratification_from_sortedness(sortedness)
    return {
        "initialAggregation": float(aggregation_values[0]) if len(aggregation_values) else None,
        "peakAggregation": float(np.max(aggregation_values)) if len(aggregation_values) else None,
        "finalAggregation": float(aggregation_values[-1]) if len(aggregation_values) else None,
        "aucAggregation": float(np.mean(aggregation_values)) if len(aggregation_values) else None,
        "sortednessTotalVariation": float(np.abs(deltas).sum()) if len(deltas) else 0.0,
        "sortednessNetGain": float(sortedness[-1] - sortedness[0]) if len(sortedness) else 0.0,
        "sortednessSignChanges": sign_changes,
        "delayedGratification": float(dg["delayedGratification"]),
        "dgEventCount": int(dg["dgEventCount"]),
        "dgTotalDrop": float(dg["dgTotalDrop"]),
        "dgTotalRecovery": float(dg["dgTotalRecovery"]),
        "dgMaxEventScore": float(dg["dgMaxEventScore"]),
        "dgMinEventScore": float(dg["dgMinEventScore"]),
    }


def paired_random_condition_id(condition: PlacementCondition, replicate_index: int) -> str | None:
    if condition.placement_category not in {"high_value", "low_value"}:
        return None
    return f"S11_{condition.algorithm}_random_{condition.frozen_variant}_rep{replicate_index:03d}"


def run_one_condition(
    *,
    condition: PlacementCondition,
    n: int,
    replicate_index: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_seed = stable_seed(STEP_ID, "random_unique_input", replicate_index)
    scheduler_seed = stable_seed(STEP_ID, condition.algorithm, condition.placement_category, condition.frozen_variant, "scheduler", replicate_index)
    tie_seed = stable_seed(STEP_ID, condition.algorithm, condition.placement_category, condition.frozen_variant, "tie", replicate_index)
    frozen_seed = stable_seed(STEP_ID, condition.algorithm, condition.placement_category, condition.frozen_variant, "frozen", replicate_index)
    initial_values = input_values_for_profile("random_unique", n, input_seed)
    frozen_positions = frozen_positions_for_category(
        condition.placement_category,
        initial_values,
        frozen_seed,
        condition.frozen_count,
    )
    condition_id = f"S11_{condition.algorithm}_{condition.placement_category}_{condition.frozen_variant}_rep{replicate_index:03d}"
    sim = DeterministicEventSimulator(
        initial_values,
        condition.algorithm,
        frozen_positions=frozen_positions,
        frozen_variant=condition.frozen_variant,
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_seed,
        condition_id=condition_id,
        implementation="cell_view",
        research_step_id=STEP_ID,
    )
    frozen_cell_ids = sim.frozen_cell_ids()
    initial_id_positions = {str(cell_id): int(sim.position_by_cell_id()[cell_id]) for cell_id in frozen_cell_ids}
    add_identity_logging(sim, frozen_cell_ids)
    result = sim.run(
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
        no_move_checks_required=2,
        no_move_check_interval=max(1, n),
    )
    final_id_positions = {str(cell_id): int(sim.position_by_cell_id()[cell_id]) for cell_id in frozen_cell_ids}
    final_frozen_cell_ids = sim.frozen_cell_ids()
    value_meta = frozen_value_metadata(initial_values, frozen_positions)
    displacement_by_id = {
        str(cell_id): int(final_id_positions[str(cell_id)] - initial_id_positions[str(cell_id)])
        for cell_id in frozen_cell_ids
    }
    abs_displacements = [abs(value) for value in displacement_by_id.values()]
    signed_displacements = list(displacement_by_id.values())

    initial_metrics = prefixed_metrics("initial", initial_values)
    final_metrics = prefixed_metrics("final", result.final_values)
    trajectory_metrics = trajectory_proxy_metrics(result)
    frozen_count_preserved = len(result.initial_frozen_positions) == len(result.final_frozen_positions) == condition.frozen_count
    frozen_identity_preserved = sorted(frozen_cell_ids) == sorted(final_frozen_cell_ids)
    placement_valid = validate_placement_category(condition.placement_category, frozen_positions, initial_values, condition.frozen_count)
    record: dict[str, Any] = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "conditionId": condition_id,
        "conditionTemplateId": f"{condition.algorithm}_{condition.placement_category}_{condition.frozen_variant}",
        "implementation": "S01_deterministic_event_simulator",
        "algorithm": condition.algorithm,
        "inputProfile": "random_unique",
        "n": int(n),
        "replicateIndex": int(replicate_index),
        "replicateNumber": int(replicate_index + 1),
        "inputSeed": int(input_seed),
        "schedulerSeed": int(scheduler_seed),
        "tieBreakerSeed": int(tie_seed),
        "frozenPositionSeed": None if condition.frozen_count == 0 else int(frozen_seed),
        "frozenVariant": condition.frozen_variant,
        "frozenCount": int(condition.frozen_count),
        "placementCategory": condition.placement_category,
        "placementCategoryLabel": placement_label(condition.placement_category),
        "placementFamily": placement_family(condition.placement_category),
        "placementCategoryValid": bool(placement_valid),
        "valuePositionConfound": bool(value_position_confound(condition.placement_category)),
        "highLowValueConfoundDocumented": bool(
            condition.placement_category not in {"high_value", "low_value"}
            or placement_confound_note(condition.placement_category)
        ),
        "confoundNote": placement_confound_note(condition.placement_category),
        "pairedRandomControlConditionId": paired_random_condition_id(condition, replicate_index),
        "initialFrozenPositions": compact_json(frozen_positions),
        "finalFrozenPositions": compact_json(result.final_frozen_positions),
        "initialFrozenCellIds": compact_json(frozen_cell_ids),
        "finalFrozenCellIds": compact_json(final_frozen_cell_ids),
        "initialFrozenCellIdPositions": compact_json(initial_id_positions),
        "finalFrozenCellIdPositions": compact_json(final_id_positions),
        "frozenCellDisplacements": compact_json(displacement_by_id),
        "frozenCellIdentityLogged": bool(condition.frozen_count == 0 or len(frozen_cell_ids) == condition.frozen_count),
        "frozenCellIdPreserved": bool(frozen_identity_preserved),
        "frozenPositionCountPreserved": bool(frozen_count_preserved),
        "frozenCellsMoved": bool(any(value != 0 for value in signed_displacements)),
        "frozenMeanAbsoluteDisplacement": float(np.mean(abs_displacements)) if abs_displacements else 0.0,
        "frozenMaxAbsoluteDisplacement": int(max(abs_displacements)) if abs_displacements else 0,
        "frozenMeanSignedDisplacement": float(np.mean(signed_displacements)) if signed_displacements else 0.0,
        "initialValuesHash": state_hash(initial_values),
        "finalValuesHash": state_hash(result.final_values),
        "initialValues": compact_json(initial_values),
        "finalValues": compact_json(result.final_values),
        "initialAlgotypes": compact_json([condition.algorithm] * n),
        "finalAlgotypes": compact_json(result.final_algotypes),
        "initialAlgotypeCounts": compact_json({condition.algorithm: n}),
        "finalAlgotypeCounts": compact_json(Counter(result.final_algotypes)),
        "completed": bool(result.completed),
        "stopReason": result.stop_reason,
        "capStop": result.stop_reason.startswith("max_"),
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "archivedCompareAndSwapCount": int(result.archived_compare_and_swap_count),
        "activationCount": int(result.activation_count),
        "eventCount": int(result.event_count),
        "blockedMoveAttempts": int(result.blocked_move_attempts),
        "frozenSwapAttempts": int(result.frozen_swap_attempts),
        "reroutingEventProxy": int(result.blocked_move_attempts + result.frozen_swap_attempts),
        "wallTimeSeconds": float(result.wall_time_seconds),
        "valueCountPreserved": sorted(initial_values) == sorted(result.final_values),
        "algotypeCountPreserved": sorted([condition.algorithm] * n) == sorted(result.final_algotypes),
        "hasDuplicateValues": len(set(initial_values)) < len(initial_values),
        "uniqueValueCount": int(len(set(initial_values))),
        "maxValueMultiplicity": int(pd.Series(initial_values).value_counts().max()),
    }
    record.update({key: compact_json(value) if isinstance(value, list) else value for key, value in value_meta.items()})
    record.update(initial_metrics)
    record.update(final_metrics)
    for key in [
        "SortednessDistanceNormalized",
        "KendallTauDistanceNormalized",
        "SpearmanFootruleDistanceNormalized",
        "EarthMoverPositionDistanceNormalized",
        "EditDistanceToTargetOrderNormalized",
    ]:
        record[f"improvement{key}"] = float(record[f"initial{key}"] - record[f"final{key}"])
    record.update(trajectory_metrics)

    trace_rows: list[dict[str, Any]] = []
    for row in result.trace_rows:
        trace_rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "conditionId": condition_id,
                "conditionTemplateId": record["conditionTemplateId"],
                "algorithm": condition.algorithm,
                "inputProfile": "random_unique",
                "replicateIndex": int(replicate_index),
                "frozenVariant": condition.frozen_variant,
                "frozenCount": int(condition.frozen_count),
                "placementCategory": condition.placement_category,
                "placementFamily": placement_family(condition.placement_category),
                "eventIndex": int(row["event_index"]),
                "eventKind": row["event_kind"],
                "activationIndex": int(row["activation_index"]),
                "swapCount": int(row["swap_count"]),
                "comparisonCount": int(row["comparison_count"]),
                "sortednessRawCount": int(row["sortedness_raw_count"]),
                "sortednessPercent": float(row["sortedness_percent"]),
                "monotonicityError": int(row["monotonicity_error"]),
                "aggregation": float(aggregation(json.loads(row["algotypes_json"]))),
                "stateHash": row["state_hash"],
                "initialStateHash": row["initial_state_hash"],
                "frozenPositions": row["frozen_positions_json"],
                "frozenCellIds": row.get("frozen_cell_ids_json", "[]"),
                "frozenCellIdPositions": row.get("frozen_cell_id_positions_json", "{}"),
            }
        )
    return record, trace_rows


def run_s11_matrix(
    *,
    n: int,
    replicates: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for condition in placement_conditions():
        for replicate_index in range(replicates):
            record, rows = run_one_condition(
                condition=condition,
                n=n,
                replicate_index=replicate_index,
                max_activations=max_activations,
                max_swaps=max_swaps,
                max_comparisons=max_comparisons,
            )
            records.append(record)
            trace_rows.extend(rows)
    return pd.DataFrame(records), pd.DataFrame(trace_rows)


def summarize_results(result_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        result_df.groupby(["algorithm", "frozenVariant", "placementCategory", "placementFamily"], dropna=False)
        .agg(
            runCount=("conditionId", "size"),
            completedRate=("completed", "mean"),
            capStopRate=("capStop", "mean"),
            finalSortednessPercentMean=("finalSortednessPercent", "mean"),
            finalKendallDistanceMean=("finalKendallTauDistanceNormalized", "mean"),
            finalEditDistanceMean=("finalEditDistanceToTargetOrderNormalized", "mean"),
            swapCountMean=("swapCount", "mean"),
            activationCountMean=("activationCount", "mean"),
            blockedMoveAttemptsMean=("blockedMoveAttempts", "mean"),
            frozenSwapAttemptsMean=("frozenSwapAttempts", "mean"),
            reroutingEventProxyMean=("reroutingEventProxy", "mean"),
            frozenMovedRate=("frozenCellsMoved", "mean"),
            frozenMeanAbsoluteDisplacementMean=("frozenMeanAbsoluteDisplacement", "mean"),
            delayedGratificationMean=("delayedGratification", "mean"),
            dgEventCountMean=("dgEventCount", "mean"),
        )
        .reset_index()
        .sort_values(["algorithm", "frozenVariant", "placementCategory"])
    )
    summary.insert(0, "researchStepId", STEP_ID)
    summary.insert(0, "experimentId", EXPERIMENT_ID)
    return summary


def category_diagnostics(result_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (category, variant), group in result_df.groupby(["placementCategory", "frozenVariant"], dropna=False):
        frozen = group[group["frozenCount"].gt(0)]
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "placementCategory": category,
                "frozenVariant": variant,
                "placementFamily": placement_family(category),
                "runCount": int(len(group)),
                "placementCategoryValidRate": float(group["placementCategoryValid"].mean()),
                "frozenIdentityLoggedRate": float(group["frozenCellIdentityLogged"].mean()),
                "frozenCellIdPreservedRate": float(group["frozenCellIdPreserved"].mean()),
                "frozenPositionCountPreservedRate": float(group["frozenPositionCountPreserved"].mean()),
                "valuePositionConfound": bool(value_position_confound(category)),
                "confoundNote": placement_confound_note(category),
                "meanFrozenInitialValueQuantile": float(frozen["frozenMeanInitialValueQuantile"].mean()) if len(frozen) else None,
                "meanFrozenInitialPositionQuantile": float(frozen["frozenMeanInitialPositionQuantile"].mean()) if len(frozen) else None,
                "completedRate": float(group["completed"].mean()),
                "finalKendallDistanceMean": float(group["finalKendallTauDistanceNormalized"].mean()),
                "reroutingEventProxyMean": float(group["reroutingEventProxy"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["placementCategory", "frozenVariant"])


def load_e01_context(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        empty = pd.DataFrame()
        return empty, empty
    df = pd.read_parquet(path)
    keep = df[
        df["algorithm"].isin(ALGORITHMS)
        & df["frozenVariant"].isin(["none", "passive", "stuck"])
        & df["frozenCount"].isin([0, 2])
    ].copy()
    keep.insert(0, "contextForResearchStepId", STEP_ID)
    keep["placementCategory"] = "e01_original_seeded_positions"
    summary = (
        keep.groupby(["algorithm", "frozenVariant", "frozenCount"], dropna=False)
        .agg(
            runCount=("conditionId", "size"),
            completedRate=("completed", "mean"),
            finalSortednessPercentMean=("finalSortednessPercent", "mean"),
            finalMonotonicityErrorMean=("finalMonotonicityError", "mean"),
            swapCountMean=("swapCount", "mean"),
            blockedMoveAttemptsMean=("blockedMoveAttempts", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "contextForResearchStepId", STEP_ID)
    return keep, summary


def diagnostics_table(
    *,
    result_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    category_df: pd.DataFrame,
    e01_context_df: pd.DataFrame,
    expected_rows: int,
    repo_tests_passed: bool,
) -> pd.DataFrame:
    frozen_rows = result_df[result_df["frozenCount"].gt(0)]
    high_low = result_df[result_df["placementCategory"].isin(["high_value", "low_value"])]
    matched_initial = result_df.groupby("replicateIndex")["initialValuesHash"].nunique().max() == 1
    checks = [
        ("expected_row_count", len(result_df) == expected_rows, f"Placement table has expected {expected_rows} rows."),
        ("all_placement_categories_represented", set(FROZEN_PLACEMENT_CATEGORIES).issubset(set(result_df["placementCategory"])), "All seven planned Frozen Cell placement categories are represented."),
        ("control_rows_represented", CONTROL_PLACEMENT_CATEGORY in set(result_df["placementCategory"]), "Matched no-Frozen control rows are represented."),
        ("all_algorithms_represented", set(result_df["algorithm"]) == set(ALGORITHMS), "Bubble, Insertion, and Selection are represented."),
        ("frozen_variants_represented", set(frozen_rows["frozenVariant"]) == set(FROZEN_VARIANTS), "Passive and stuck Frozen Cell variants are represented."),
        ("matched_initial_arrays", bool(matched_initial), "Initial arrays are matched across algorithms, variants, and placement categories within each replicate."),
        ("placement_categories_valid", bool(result_df["placementCategoryValid"].all()), "Every row passes exact placement-category validation."),
        ("frozen_identities_logged", bool(result_df["frozenCellIdentityLogged"].all()), "Frozen cell identities are logged for every Frozen Cell row."),
        ("frozen_identity_preserved", bool(result_df["frozenCellIdPreserved"].all()), "Frozen cell identities are preserved through each run."),
        ("frozen_count_preserved", bool(result_df["frozenPositionCountPreserved"].all()), "Frozen Cell counts are preserved through each run."),
        ("value_counts_preserved", bool(result_df["valueCountPreserved"].all()), "All runs preserve value counts."),
        ("algotype_counts_preserved", bool(result_df["algotypeCountPreserved"].all()), "All runs preserve Algotype counts."),
        ("high_low_confound_documented", bool(high_low["highLowValueConfoundDocumented"].all()) and bool(high_low["confoundNote"].str.len().gt(0).all()), "High-/low-value placement confounds are explicitly documented."),
        ("paired_random_controls_logged", bool(high_low["pairedRandomControlConditionId"].notna().all()), "High-/low-value rows point to matched random-placement controls."),
        ("stop_reasons_logged", bool(result_df["stopReason"].notna().all()), "Every run records a stop reason or cap."),
        ("trace_rows_written", len(trace_df) >= len(result_df), "Trace rows were written for every run."),
        ("trace_identity_snapshots_written", bool(trace_df["frozenCellIds"].notna().all()) if len(trace_df) else False, "Trace rows include frozen identity snapshots."),
        ("summary_written", len(summary_df) > 0, "Placement summary table is nonempty."),
        ("category_diagnostics_written", len(category_df) > 0, "Placement category diagnostics table is nonempty."),
        ("e01_context_loaded", len(e01_context_df) > 0, "E01 Frozen Cell context table was loaded."),
        ("repo_tests_passed", bool(repo_tests_passed), "Repository S11 unit tests passed."),
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
    *,
    artifacts_dir: Path,
    result_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    category_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    e01_context_df: pd.DataFrame,
    e01_summary_df: pd.DataFrame,
) -> dict[str, Path]:
    result_dir = artifacts_dir / "results"
    trace_dir = artifacts_dir / "traces" / "e02" / STEP_ID
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "frozen_placement_parquet": result_dir / "e02_frozen_placement.parquet",
        "frozen_placement_csv": result_dir / "e02_frozen_placement.csv",
        "summary_parquet": result_dir / "e02_frozen_placement_summary.parquet",
        "summary_csv": result_dir / "e02_frozen_placement_summary.csv",
        "category_diagnostics_parquet": result_dir / "e02_frozen_placement_category_diagnostics.parquet",
        "category_diagnostics_csv": result_dir / "e02_frozen_placement_category_diagnostics.csv",
        "diagnostics_parquet": result_dir / "e02_frozen_placement_diagnostics.parquet",
        "diagnostics_csv": result_dir / "e02_frozen_placement_diagnostics.csv",
        "e01_context_parquet": result_dir / "e02_frozen_placement_e01_context.parquet",
        "e01_context_csv": result_dir / "e02_frozen_placement_e01_context.csv",
        "e01_context_summary_parquet": result_dir / "e02_frozen_placement_e01_context_summary.parquet",
        "e01_context_summary_csv": result_dir / "e02_frozen_placement_e01_context_summary.csv",
        "trace_parquet": trace_dir / "e02_frozen_placement_trace_events.parquet",
        "trace_csv_gz": trace_dir / "e02_frozen_placement_trace_events.csv.gz",
    }
    result_df.to_parquet(paths["frozen_placement_parquet"], index=False)
    result_df.to_csv(paths["frozen_placement_csv"], index=False)
    summary_df.to_parquet(paths["summary_parquet"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    category_df.to_parquet(paths["category_diagnostics_parquet"], index=False)
    category_df.to_csv(paths["category_diagnostics_csv"], index=False)
    diagnostics_df.to_parquet(paths["diagnostics_parquet"], index=False)
    diagnostics_df.to_csv(paths["diagnostics_csv"], index=False)
    e01_context_df.to_parquet(paths["e01_context_parquet"], index=False)
    e01_context_df.to_csv(paths["e01_context_csv"], index=False)
    e01_summary_df.to_parquet(paths["e01_context_summary_parquet"], index=False)
    e01_summary_df.to_csv(paths["e01_context_summary_csv"], index=False)
    trace_df.to_parquet(paths["trace_parquet"], index=False)
    trace_df.to_csv(paths["trace_csv_gz"], index=False, compression="gzip")
    return paths


def plot_summary(summary_df: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    frozen = summary_df[summary_df["placementCategory"].ne(CONTROL_PLACEMENT_CATEGORY)].copy()
    categories = FROZEN_PLACEMENT_CATEGORIES
    labels = [placement_label(category).replace(" ", "\n") for category in categories]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for algorithm in ALGORITHMS:
        subset = frozen[frozen["algorithm"].eq(algorithm)]
        completion = subset.groupby("placementCategory")["completedRate"].mean().reindex(categories)
        distance = subset.groupby("placementCategory")["finalKendallDistanceMean"].mean().reindex(categories)
        reroute = subset.groupby("placementCategory")["reroutingEventProxyMean"].mean().reindex(categories)
        axes[0].plot(range(len(categories)), completion, marker="o", label=algorithm)
        axes[1].plot(range(len(categories)), distance, marker="o", label=algorithm)
        axes[2].plot(range(len(categories)), reroute, marker="o", label=algorithm)
    for ax, ylabel, title in [
        (axes[0], "completion rate", "Completion by placement"),
        (axes[1], "mean final Kendall distance", "Final order distance"),
        (axes[2], "blocked/frozen attempts", "Rerouting proxy"),
    ]:
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    summary_png = figure_dir / "e02_frozen_placement_summary.png"
    summary_pdf = figure_dir / "e02_frozen_placement_summary.pdf"
    fig.savefig(summary_png, dpi=180)
    fig.savefig(summary_pdf)
    plt.close(fig)

    pivot = (
        frozen.groupby(["placementCategory", "frozenVariant"])["completedRate"]
        .mean()
        .unstack("frozenVariant")
        .reindex(categories)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    width = 0.35
    for idx, variant in enumerate(pivot.columns):
        ax.bar(x + (idx - (len(pivot.columns) - 1) / 2) * width, pivot[variant], width=width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("completion rate")
    ax.set_title("Frozen Cell completion by placement and variant")
    ax.legend()
    fig.tight_layout()
    completion_png = figure_dir / "e02_frozen_placement_completion.png"
    completion_pdf = figure_dir / "e02_frozen_placement_completion.pdf"
    fig.savefig(completion_png, dpi=180)
    fig.savefig(completion_pdf)
    plt.close(fig)

    dg = frozen.groupby(["placementCategory", "frozenVariant"])["delayedGratificationMean"].mean().unstack("frozenVariant").reindex(categories)
    fig, ax = plt.subplots(figsize=(10, 5))
    for variant in dg.columns:
        ax.plot(range(len(categories)), dg[variant], marker="o", label=variant)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean DG")
    ax.set_title("E01-compatible DG proxy by placement")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    dg_png = figure_dir / "e02_frozen_placement_dg.png"
    dg_pdf = figure_dir / "e02_frozen_placement_dg.pdf"
    fig.savefig(dg_png, dpi=180)
    fig.savefig(dg_pdf)
    plt.close(fig)
    return summary_png, summary_pdf, completion_png, completion_pdf, dg_png, dg_pdf


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.exists() and path.is_file():
            records.append({"path": str(path), "sha256": sha256_path(path), "sizeBytes": path.stat().st_size})
    return sorted(records, key=lambda row: row["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    paths = [
        (REPO_ROOT / "scripts" / "e02_s11_frozen_placement.py", code_dir / "scripts" / "e02_s11_frozen_placement.py"),
        (REPO_ROOT / "tests" / "test_e02_frozen_placement.py", code_dir / "tests" / "test_e02_frozen_placement.py"),
        (REPO_ROOT / "scripts" / "e02_s10_input_distributions.py", code_dir / "scripts" / "e02_s10_input_distributions.py"),
        (REPO_ROOT / "scripts" / "e02_s09_alternative_metrics.py", code_dir / "scripts" / "e02_s09_alternative_metrics.py"),
        (REPO_ROOT / "scripts" / "e02_s08_dg_nulls.py", code_dir / "scripts" / "e02_s08_dg_nulls.py"),
        (REPO_ROOT / "e02_deterministic_simulator" / "__init__.py", code_dir / "e02_deterministic_simulator" / "__init__.py"),
        (REPO_ROOT / "e02_deterministic_simulator" / "metrics.py", code_dir / "e02_deterministic_simulator" / "metrics.py"),
        (REPO_ROOT / "e02_deterministic_simulator" / "simulator.py", code_dir / "e02_deterministic_simulator" / "simulator.py"),
    ]
    copied: list[Path] = []
    for src, dst in paths:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def run_repo_tests(step_dir: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "tests.test_e02_frozen_placement"]
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


def outcome_classification(summary_df: pd.DataFrame, validation_success: bool) -> str:
    if not validation_success:
        return "null"
    frozen = summary_df[summary_df["placementCategory"].ne(CONTROL_PLACEMENT_CATEGORY)]
    if frozen.empty:
        return "null"
    completion_range = (
        frozen.groupby("placementCategory")["completedRate"].mean().max()
        - frozen.groupby("placementCategory")["completedRate"].mean().min()
    )
    distance_max = frozen["finalKendallDistanceMean"].max()
    if completion_range > 0.25 or distance_max > 0.05:
        return "constraining/contradictory"
    return "supportive"


def write_validation_report(step_dir: Path, diagnostics_df: pd.DataFrame, validation: dict[str, Any]) -> Path:
    lines = [
        "# E02 S11 Validation Report",
        "",
        "- Research step ID: S11",
        "- Completion status: completed" if validation["success"] else "- Completion status: failed",
        "- Artifacts written: Frozen Cell placement table, placement summary, category diagnostics, E01 context tables, trace table with frozen identity snapshots, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {validation['validationResult']}",
        "- Caveats or blockers: S11 uses f=2 deterministic S01 simulator rows for new placement categories; high-/low-value categories intentionally confound value identity with current position and are documented with matched initial arrays, paired random controls, and quantile diagnostics.",
        "- Recommended next action: stop before S12 for Chief Scientist review.",
        "",
        "## Checks",
    ]
    for row in diagnostics_df.itertuples(index=False):
        lines.append(f"- {'passed' if row.passed else 'failed'}: {row.detail}")
    path = step_dir / "validation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_reports_and_manifests(
    *,
    artifacts_dir: Path,
    started_at: float,
    result_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    category_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    e01_context_df: pd.DataFrame,
    e01_summary_df: pd.DataFrame,
    table_paths: dict[str, Path],
    figure_paths: tuple[Path, ...],
    code_paths: list[Path],
    repo_tests: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Path]:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    step_dir.mkdir(parents=True, exist_ok=True)
    validation_path = write_validation_report(step_dir, diagnostics_df, validation)
    outcome = outcome_classification(summary_df, validation["success"])
    caveat = (
        "S11 fixes Frozen Cell count at f=2 to isolate placement categories in the bounded n=30 audit matrix. "
        "High-/low-value placements intentionally confound value identity with current position; the run controls initial arrays across categories and logs value/position quantiles plus paired random-placement condition IDs, but does not fully separate value effects from position effects."
    )
    recommended = "Stop before S12 for Chief Scientist review; if accepted, proceed to S12 Frozen Cell behavior variation."

    frozen_summary = (
        summary_df[summary_df["placementCategory"].ne(CONTROL_PLACEMENT_CATEGORY)]
        .groupby(["placementCategory", "frozenVariant"], dropna=False)
        .agg(
            completedRate=("completedRate", "mean"),
            finalKendallDistance=("finalKendallDistanceMean", "mean"),
            reroutingProxy=("reroutingEventProxyMean", "mean"),
            delayedGratification=("delayedGratificationMean", "mean"),
        )
        .reset_index()
        .sort_values(["placementCategory", "frozenVariant"])
    )
    frozen_table = markdown_table(
        ["Placement", "Variant", "Completed", "Final Kendall", "Rerouting", "DG"],
        [
            [
                placement_label(row.placementCategory),
                row.frozenVariant,
                row.completedRate,
                row.finalKendallDistance,
                row.reroutingProxy,
                row.delayedGratification,
            ]
            for row in frozen_summary.itertuples(index=False)
        ],
    )
    algorithm_rows = (
        summary_df[summary_df["placementCategory"].ne(CONTROL_PLACEMENT_CATEGORY)]
        .groupby("algorithm", dropna=False)
        .agg(
            completedRate=("completedRate", "mean"),
            finalKendallDistance=("finalKendallDistanceMean", "mean"),
            reroutingProxy=("reroutingEventProxyMean", "mean"),
            capStopRate=("capStopRate", "mean"),
        )
        .reset_index()
        .sort_values("algorithm")
    )
    algorithm_table = markdown_table(
        ["Algorithm", "Completed", "Final Kendall", "Rerouting", "Cap stops"],
        [[row.algorithm, row.completedRate, row.finalKendallDistance, row.reroutingProxy, row.capStopRate] for row in algorithm_rows.itertuples(index=False)],
    )
    summary_path = step_dir / "summary.md"
    summary_lines = [
        "# E02 S11 Summary",
        "",
        "- Research step ID: S11",
        "- Completion status: completed" if validation["success"] else "- Completion status: failed",
        "- Artifacts written: `$ARTIFACTS_DIR/results/e02_frozen_placement.parquet`, placement summaries, category diagnostics, E01 context tables, trace table, figures, copied code, manifest, status JSON, and run manifest.",
        f"- Validation result: {validation['validationResult']}",
        f"- Caveats or blockers: {caveat}",
        "- Lay summary: S11 held the initial arrays and Frozen Cell count fixed while moving f=2 passive or stuck Frozen Cells to seven placement categories. The run records frozen cell IDs, their initial and final positions, and value/position quantiles so placement-sensitive failures can be separated from missing provenance.",
        f"- Recommended next action: {recommended}",
        f"- Outcome classification: {outcome}",
        "",
        "## Run Counts",
        "",
        markdown_table(
            ["Item", "Count"],
            [
                ["Placement stress runs", len(result_df)],
                ["Trace rows", len(trace_df)],
                ["Frozen placement categories", len(FROZEN_PLACEMENT_CATEGORIES)],
                ["Algorithms", result_df["algorithm"].nunique()],
                ["E01 context rows", len(e01_context_df)],
                ["Diagnostics", len(diagnostics_df)],
            ],
        ),
        "",
        "## Placement Summary",
        "",
        frozen_table,
        "",
        "## Algorithm Summary",
        "",
        algorithm_table,
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
    artifacts_written = [str(path) for path in sorted(set(artifact_inputs), key=str)]
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"] and repo_tests["passed"]),
        "status": "completed" if validation["success"] and repo_tests["passed"] else "failed",
        "artifactsWritten": artifacts_written,
        "validationResult": validation["validationResult"] if repo_tests["passed"] else "failed",
        "caveatsOrBlockers": caveat,
        "recommendedNextAction": recommended,
        "outcomeClassification": outcome,
        "repoTests": repo_tests,
        "runSeconds": time.perf_counter() - started_at,
        "placementRunRows": int(len(result_df)),
        "traceRows": int(len(trace_df)),
        "summaryRows": int(len(summary_df)),
        "categoryDiagnosticRows": int(len(category_df)),
        "e01ContextRows": int(len(e01_context_df)),
    }
    write_json(status_path, status_payload)

    all_artifacts = collect_artifacts(
        list(table_paths.values())
        + list(figure_paths)
        + code_paths
        + [validation_path, summary_path, status_path, step_dir / "repo_unit_test_log.txt"]
    )
    manifest_payload = {
        "schema": "eidosoma.research_step_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "git": get_git_metadata(),
        "inputs": {
            "deterministicSimulator": "e02_deterministic_simulator",
            "s08DgHelper": "scripts/e02_s08_dg_nulls.py",
            "s09MetricHelper": "scripts/e02_s09_alternative_metrics.py",
            "s10InputGeneratorHelper": "scripts/e02_s10_input_distributions.py",
            "e01FrozenContext": str(DEFAULT_E01_FROZEN_PATH),
        },
        "parameters": {
            "inputProfile": "random_unique",
            "n": int(result_df["n"].iloc[0]) if len(result_df) else None,
            "replicates": int(result_df["replicateIndex"].nunique()) if len(result_df) else 0,
            "algorithms": ALGORITHMS,
            "frozenVariants": FROZEN_VARIANTS,
            "frozenPlacementCategories": FROZEN_PLACEMENT_CATEGORIES,
            "frozenCount": 2,
        },
        "outputs": all_artifacts,
        "validation": validation,
        "repoTests": repo_tests,
        "placementRunRows": int(len(result_df)),
        "traceRows": int(len(trace_df)),
        "summaryRows": int(len(summary_df)),
        "categoryDiagnosticRows": int(len(category_df)),
        "e01ContextRows": int(len(e01_context_df)),
        "outcomeClassification": outcome,
    }
    write_json(artifact_manifest_path, manifest_payload)

    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "latestResearchStepId": STEP_ID,
        "generatedAt": utc_now(),
        "statusPath": str(status_path),
        "git": get_git_metadata(),
        "hardware": {"platform": platform.platform(), "python": sys.version, "cpuCount": os.cpu_count()},
        "packageVersions": {"numpy": np.__version__, "pandas": pd.__version__},
        "runs": [
            {
                "researchStepId": STEP_ID,
                "status": status_payload["status"],
                "runSeconds": status_payload["runSeconds"],
                "placementRunRows": int(len(result_df)),
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


def run_s11(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    figure_dir = artifacts_dir / "figures" / "e02"
    step_dir.mkdir(parents=True, exist_ok=True)
    result_df, trace_df = run_s11_matrix(
        n=args.n,
        replicates=args.replicates,
        max_activations=args.max_activations,
        max_swaps=args.max_swaps,
        max_comparisons=args.max_comparisons,
    )
    summary_df = summarize_results(result_df)
    category_df = category_diagnostics(result_df)
    e01_context_df, e01_summary_df = load_e01_context(args.e01_frozen_context)
    expected_rows = len(placement_conditions()) * args.replicates
    diagnostics_df = diagnostics_table(
        result_df=result_df,
        trace_df=trace_df,
        summary_df=summary_df,
        category_df=category_df,
        e01_context_df=e01_context_df,
        expected_rows=expected_rows,
        repo_tests_passed=True,
    )
    validation = validate_outputs(diagnostics_df)
    table_paths = write_tables(
        artifacts_dir=artifacts_dir,
        result_df=result_df,
        trace_df=trace_df,
        summary_df=summary_df,
        category_df=category_df,
        diagnostics_df=diagnostics_df,
        e01_context_df=e01_context_df,
        e01_summary_df=e01_summary_df,
    )
    figure_paths = plot_summary(summary_df, figure_dir)
    code_paths = copy_code_artifacts(step_dir)
    repo_tests = run_repo_tests(step_dir)
    diagnostics_df = diagnostics_table(
        result_df=result_df,
        trace_df=trace_df,
        summary_df=summary_df,
        category_df=category_df,
        e01_context_df=e01_context_df,
        expected_rows=expected_rows,
        repo_tests_passed=repo_tests["passed"],
    )
    validation = validate_outputs(diagnostics_df)
    diagnostics_df.to_parquet(table_paths["diagnostics_parquet"], index=False)
    diagnostics_df.to_csv(table_paths["diagnostics_csv"], index=False)
    if not repo_tests["passed"]:
        validation["success"] = False
        validation["validationResult"] = "failed"
        validation.setdefault("failures", []).append("Repository S11 unit tests failed.")
    write_reports_and_manifests(
        artifacts_dir=artifacts_dir,
        started_at=started_at,
        result_df=result_df,
        trace_df=trace_df,
        summary_df=summary_df,
        category_df=category_df,
        diagnostics_df=diagnostics_df,
        e01_context_df=e01_context_df,
        e01_summary_df=e01_summary_df,
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
    parser.add_argument("--e01-frozen-context", type=Path, default=DEFAULT_E01_FROZEN_PATH)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--max-activations", type=int, default=300_000)
    parser.add_argument("--max-swaps", type=int, default=100_000)
    parser.add_argument("--max-comparisons", type=int, default=800_000)
    return parser.parse_args()


def main() -> int:
    return run_s11(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
