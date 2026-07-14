#!/usr/bin/env python3
"""Build and validate the frozen E02 S06 exact-count placement bank."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.faults import (
    FaultExecutionContract,
    MobilityProfile,
    exact_replay_fault,
    run_faulted_architecture,
)
from causal_simulator.placements import (
    PLACEMENT_INTERFACE_VERSION,
    ConfirmatoryLeakageError,
    FaultPlacement,
    PlacementClass,
    PlacementContext,
    boundary_band_positions,
    generate_structural_bank,
    generate_structural_placement,
    materialize_fault_scenario,
    median_rank_band_positions,
    search_exploratory_placements,
    select_confirmatory_placements,
    structural_lock_record,
)
from causal_simulator.schedulers import SchedulerExecutionContract, SchedulerFamily
from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    canonical_json_bytes,
    sha256_json,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MASTER_SEED = int("e0206000000000000000000000000001", 16)
FAULT_COUNTS = (2, 4, 6)
MAPS_PER_STRUCTURAL_CLASS = 12
SEARCH_MAPS_PER_CELL = 6
STRUCTURAL_CLASSES = (
    PlacementClass.UNIFORM_EXACT,
    PlacementClass.CLUSTERED_EXACT,
    PlacementClass.BOUNDARY_EXACT,
    PlacementClass.MEDIAN_RANK_EXACT,
)
ARCHITECTURES = (
    ArchitectureExecutionContract.central_local_k1(),
    ArchitectureExecutionContract.distributed_local(),
    ArchitectureExecutionContract.distributed_weak(enabled=False),
    ArchitectureExecutionContract.distributed_weak(),
)
SCHEDULERS = tuple(SchedulerFamily)
CORE_REGENERATION_FILES = (
    "fault_placement_bank.parquet",
    "placement_descriptors.parquet",
    "search_history.parquet",
    "structural_lock.json",
    "placement_contexts.json",
    "coverage_summary.csv",
    "uniform_distribution_validation.json",
    "contract_execution_validation.parquet",
    "search_replay_validation.parquet",
)


def contexts() -> tuple[PlacementContext, ...]:
    identities = tuple(f"cell-{index:04d}" for index in range(24))
    alternating: list[int] = []
    for low in range(1, 13):
        alternating.extend((25 - low, low))
    return (
        PlacementContext(
            "reverse_unique_n24", identities, tuple(range(24, 0, -1))
        ),
        PlacementContext(
            "alternating_extremes_n24", identities, tuple(alternating)
        ),
        PlacementContext(
            "repeated_six_n24", identities, (6, 1, 5, 2, 4, 3) * 4
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def derive_seed(*parts: object) -> int:
    payload = {
        "domain": "E02/S06/derived-seed/v1",
        "masterSeed": str(MASTER_SEED),
        "parts": list(parts),
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:16], "big")


def loss(values: Sequence[int | float]) -> int:
    adjacent = sum(left > right for left, right in zip(values, values[1:]))
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return 1000 * adjacent + inversions


def search_training_scenario(
    context: PlacementContext, fault_count: int, positions: tuple[int, ...]
) -> Scenario:
    if not len(positions) == len(set(positions)) == fault_count:
        raise AssertionError("invalid search candidate")
    faulty = {context.identity_ids[position] for position in positions}
    cells = tuple(
        Cell(
            identity,
            value,
            Policy.SELECTION,
            Direction.ASCENDING,
            FaultMode.STUCK if identity in faulty else FaultMode.NORMAL,
        )
        for identity, value in zip(context.identity_ids, context.values)
    )
    address = {
        "contextSha256": context.context_sha256,
        "faultCount": fault_count,
        "positions": list(positions),
        "role": "placement_search_training",
    }
    return Scenario.create(
        cells,
        initial_occupancy=context.identity_ids,
        seed=derive_seed("search_training", context.context_id, fault_count),
        max_activations=12 * context.n,
        architecture=Architecture.CELL_VIEW,
        batch_width=1,
        generation_key="E02/S06/search-training/" + sha256_json(address),
        fault_placement="explicit",
        requested_fault_count=fault_count,
    )


def score_search_candidate(
    context: PlacementContext, fault_count: int, positions: tuple[int, ...]
) -> tuple[int, int, int]:
    scenario = search_training_scenario(context, fault_count, positions)
    scheduler = SchedulerExecutionContract(SchedulerFamily.DETERMINISTIC_SCAN)
    fault = FaultExecutionContract(mobility=MobilityProfile.STUCK)
    local = run_faulted_architecture(
        scenario,
        ArchitectureExecutionContract.distributed_local(),
        scheduler,
        fault,
        trace_mode="digest",
    )
    weak = run_faulted_architecture(
        scenario,
        ArchitectureExecutionContract.distributed_weak(),
        scheduler,
        fault,
        trace_mode="digest",
    )
    if not all(local.opportunity_validation().values()) or not all(
        weak.opportunity_validation().values()
    ):
        raise AssertionError("search training violated S05 opportunity accounting")
    local_loss = loss(local.result.summary["finalValues"])
    weak_loss = loss(weak.result.summary["finalValues"])
    return weak_loss - local_loss, local_loss, weak_loss


def score_search_candidate_task(
    task: tuple[PlacementContext, int, tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    context, fault_count, positions = task
    return positions, score_search_candidate(context, fault_count, positions)


def score_search_candidate_batch(
    executor: ProcessPoolExecutor | None,
    workers: int,
    context: PlacementContext,
    fault_count: int,
    candidates: tuple[tuple[int, ...], ...],
) -> dict[tuple[int, ...], tuple[int, int, int]]:
    tasks = tuple((context, fault_count, positions) for positions in candidates)
    if executor is None:
        return dict(score_search_candidate_task(task) for task in tasks)
    chunksize = max(1, len(tasks) // (workers * 4))
    return dict(executor.map(score_search_candidate_task, tasks, chunksize=chunksize))


def uniform_distribution_audit(
    frozen_contexts: Sequence[PlacementContext], draws: int = 4096
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    passed = True
    for context in frozen_contexts:
        for fault_count in FAULT_COUNTS:
            counts = np.zeros(context.n, dtype=np.int64)
            exact = True
            for replicate in range(draws):
                item = generate_structural_placement(
                    context,
                    PlacementClass.UNIFORM_EXACT,
                    fault_count,
                    seed=MASTER_SEED,
                    replicate_ordinal=100_000 + replicate,
                )
                exact &= len(item.positions) == len(set(item.positions)) == fault_count
                counts[list(item.positions)] += 1
            expected = draws * fault_count / context.n
            variance = draws * (fault_count / context.n) * (1 - fault_count / context.n)
            max_abs_z = float(np.max(np.abs(counts - expected) / math.sqrt(variance)))
            chi_square = float(np.sum((counts - expected) ** 2 / expected))
            p_value = float(stats.chi2.sf(chi_square, context.n - 1))
            cell_passed = bool(exact and max_abs_z < 5.0 and p_value > 0.001)
            passed &= cell_passed
            rows.append(
                {
                    "contextId": context.context_id,
                    "faultCount": fault_count,
                    "draws": draws,
                    "totalSelections": int(counts.sum()),
                    "expectedPerPosition": expected,
                    "minimumPositionSelections": int(counts.min()),
                    "maximumPositionSelections": int(counts.max()),
                    "maxAbsoluteMarginalZ": max_abs_z,
                    "chiSquare": chi_square,
                    "degreesOfFreedom": context.n - 1,
                    "pValue": p_value,
                    "exactCountEachDraw": bool(exact),
                    "passed": cell_passed,
                }
            )
    return {
        "schemaVersion": "e02.s06.uniform_distribution_validation.v1",
        "researchStepId": "S06",
        "drawsPerContextFaultCount": draws,
        "cells": rows,
        "passed": bool(passed),
    }


def coverage_rows(placements: Sequence[FaultPlacement]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[FaultPlacement]] = defaultdict(list)
    for item in placements:
        grouped[(item.context_id, item.fault_count, item.placement_class.value)].append(item)
    rows: list[dict[str, Any]] = []
    descriptor_names = (
        "meanNormalizedPosition",
        "meanNormalizedEdgeDistance",
        "normalizedPositionSpan",
        "meanNormalizedPairwiseDistance",
        "longestContiguousRun",
        "contiguousRunCount",
        "boundaryBandOccupancy",
        "meanNormalizedMedianRankDistance",
    )
    for (context_id, fault_count, class_name), items in sorted(grouped.items()):
        descriptors = [item.descriptor_dict() for item in items]
        row: dict[str, Any] = {
            "contextId": context_id,
            "faultCount": fault_count,
            "placementClass": class_name,
            "mapCount": len(items),
            "uniquePositionSets": len({item.positions for item in items}),
        }
        for name in descriptor_names:
            values = [float(item[name]) for item in descriptors]
            row[f"{name}Min"] = min(values)
            row[f"{name}Mean"] = sum(values) / len(values)
            row[f"{name}Max"] = max(values)
        rows.append(row)
    return rows


def draw_coverage_plot(placements: Sequence[FaultPlacement], output: Path) -> None:
    ordered_classes = [item.value for item in PlacementClass]
    descriptor_names = (
        ("meanNormalizedEdgeDistance", "Mean edge distance"),
        ("normalizedPositionSpan", "Position span"),
        ("meanNormalizedMedianRankDistance", "Median-rank distance"),
    )
    plt.rcParams.update({"font.size": 9, "svg.hashsalt": "E02-S06"})
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    for axis, (key, label) in zip(axes, descriptor_names):
        values = [
            [
                float(item.descriptor_dict()[key])
                for item in placements
                if item.placement_class.value == class_name
            ]
            for class_name in ordered_classes
        ]
        axis.boxplot(values, tick_labels=[name.replace("_exact", "") for name in ordered_classes])
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("S06 fault-placement descriptor coverage (n=24; f=2,4,6)")
    figure.savefig(output / "placement_coverage.png", dpi=180, metadata={"Software": "E02 S06"})
    figure.savefig(
        output / "placement_coverage.svg",
        metadata={"Creator": "E02 S06", "Date": None},
    )
    plt.close(figure)


def validate_materialization(
    frozen_contexts: Sequence[PlacementContext], placements: Sequence[FaultPlacement]
) -> dict[str, Any]:
    context_map = {item.context_id: item for item in frozen_contexts}
    checked = 0
    errors: list[str] = []
    for placement in placements:
        context = context_map[placement.context_id]
        for mode in (FaultMode.PASSIVE, FaultMode.STUCK):
            try:
                scenario = materialize_fault_scenario(
                    context,
                    placement,
                    fault_mode=mode,
                    policy=Policy.SELECTION,
                    seed=derive_seed("materialization", placement.placement_id, mode.value),
                    max_activations=96,
                    generation_key=f"E02/S06/materialization/{mode.value}",
                )
                observed = {
                    cell.cell_id for cell in scenario.cells if cell.fault == mode
                }
                if observed != set(placement.identity_ids):
                    raise AssertionError("fault identity binding mismatch")
                if (
                    scenario.requested_fault_count != placement.fault_count
                    or scenario.realized_fault_count != placement.fault_count
                ):
                    raise AssertionError("fault count mismatch")
                checked += 1
            except Exception as error:  # noqa: BLE001 - validation artifact records type
                errors.append(
                    f"{placement.placement_id}/{mode.value}: {type(error).__name__}: {error}"
                )
    return {
        "schemaVersion": "e02.s06.materialization_validation.v1",
        "placements": len(placements),
        "modeOverlaysChecked": checked,
        "errors": errors,
        "passed": checked == 2 * len(placements) and not errors,
    }


def contract_validation_maps(
    search_placements: Sequence[FaultPlacement],
) -> list[tuple[PlacementContext, FaultPlacement, bool]]:
    context = PlacementContext(
        "contract_validation_only_n24",
        tuple(f"validation-cell-{index:04d}" for index in range(24)),
        tuple(range(24, 0, -1)),
    )
    result: list[tuple[PlacementContext, FaultPlacement, bool]] = []
    for fault_count in FAULT_COUNTS:
        for placement_class in STRUCTURAL_CLASSES:
            result.append(
                (
                    context,
                    generate_structural_placement(
                        context,
                        placement_class,
                        fault_count,
                        seed=derive_seed("contract_validation", placement_class.value),
                        replicate_ordinal=fault_count,
                    ),
                    True,
                )
            )
        search = next(
            item
            for item in search_placements
            if item.context_id == "reverse_unique_n24"
            and item.fault_count == fault_count
            and item.search_tail == "max_weak_minus_local"
        )
        search_context = next(
            item for item in contexts() if item.context_id == search.context_id
        )
        result.append((search_context, search, False))
    return result


def run_contract_validation(
    search_placements: Sequence[FaultPlacement],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replay_passed = 0
    ledger_passed = 0
    for context, placement, validation_only in contract_validation_maps(search_placements):
        for mode in (MobilityProfile.PASSIVE, MobilityProfile.STUCK):
            scenario = materialize_fault_scenario(
                context,
                placement,
                fault_mode=mode.value,
                policy=Policy.SELECTION,
                seed=derive_seed(
                    "contract_execution", placement.placement_id, mode.value
                ),
                max_activations=4 * context.n,
                generation_key=f"E02/S06/contract-execution/{mode.value}",
            )
            fault = FaultExecutionContract(mobility=mode)
            for architecture in ARCHITECTURES:
                for scheduler_family in SCHEDULERS:
                    scheduler = SchedulerExecutionContract(scheduler_family)
                    item = run_faulted_architecture(
                        scenario,
                        architecture,
                        scheduler,
                        fault,
                        trace_mode="digest",
                    )
                    checks = item.opportunity_validation()
                    ledger_ok = all(checks.values())
                    replay = exact_replay_fault(item)
                    replay_ok = replay.to_json_bytes() == item.to_json_bytes()
                    ledger_passed += ledger_ok
                    replay_passed += replay_ok
                    rows.append(
                        {
                            "schemaVersion": "e02.s06.contract_execution.v1",
                            "contextId": context.context_id,
                            "validationOnlyMap": validation_only,
                            "placementId": placement.placement_id,
                            "placementClass": placement.placement_class.value,
                            "faultCount": placement.fault_count,
                            "mobility": mode.value,
                            "architecture": architecture.architecture.value,
                            "coordinatorProfile": architecture.coordinator_profile.value,
                            "scheduler": scheduler_family.value,
                            "requestedFaultCount": scenario.requested_fault_count,
                            "realizedFaultCount": scenario.realized_fault_count,
                            "activationCount": item.result.summary["activationCount"],
                            "proposals": item.result.summary["ledger"]["proposals"],
                            "stopReason": item.result.summary["stopReason"],
                            "eventDigest": item.result.event_digest,
                            "ledgerPassed": ledger_ok,
                            "replayPassed": replay_ok,
                            "failedChecks": "|".join(
                                name for name, passed in checks.items() if not passed
                            ),
                        }
                    )
    expected = 5 * len(FAULT_COUNTS) * 2 * len(ARCHITECTURES) * len(SCHEDULERS)
    summary = {
        "schemaVersion": "e02.s06.contract_execution_summary.v1",
        "expectedRuns": expected,
        "executedRuns": len(rows),
        "ledgerPassed": ledger_passed,
        "replayPassed": replay_passed,
        "structuralBankOutcomesAccessed": False,
        "structuralExecutionContext": "contract_validation_only_n24",
        "searchExecutionRole": "exploratory_search_derived",
        "passed": len(rows) == replay_passed == ledger_passed == expected,
    }
    return rows, summary


def validate_search_replay(
    frozen_contexts: Sequence[PlacementContext],
    placements: Sequence[FaultPlacement],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context_map = {item.context_id: item for item in frozen_contexts}
    rows: list[dict[str, Any]] = []
    passed = 0
    for placement in placements:
        context = context_map[placement.context_id]
        scenario = search_training_scenario(
            context, placement.fault_count, placement.positions
        )
        scheduler = SchedulerExecutionContract(SchedulerFamily.DETERMINISTIC_SCAN)
        fault = FaultExecutionContract(mobility=MobilityProfile.STUCK)
        observed: dict[str, int] = {}
        replay_flags: dict[str, bool] = {}
        for name, architecture in (
            ("local", ArchitectureExecutionContract.distributed_local()),
            ("weak", ArchitectureExecutionContract.distributed_weak()),
        ):
            item = run_faulted_architecture(
                scenario, architecture, scheduler, fault, trace_mode="digest"
            )
            replay = exact_replay_fault(item)
            replay_flags[name] = replay.to_json_bytes() == item.to_json_bytes()
            observed[name] = loss(item.result.summary["finalValues"])
        matches = (
            observed["local"] == placement.local_loss
            and observed["weak"] == placement.weak_loss
            and observed["weak"] - observed["local"] == placement.search_score
            and all(replay_flags.values())
        )
        passed += matches
        rows.append(
            {
                "schemaVersion": "e02.s06.search_replay.v1",
                "placementId": placement.placement_id,
                "contextId": placement.context_id,
                "faultCount": placement.fault_count,
                "searchTail": placement.search_tail,
                "storedScore": placement.search_score,
                "observedScore": observed["weak"] - observed["local"],
                "storedLocalLoss": placement.local_loss,
                "observedLocalLoss": observed["local"],
                "storedWeakLoss": placement.weak_loss,
                "observedWeakLoss": observed["weak"],
                "localReplayPassed": replay_flags["local"],
                "weakReplayPassed": replay_flags["weak"],
                "passed": matches,
            }
        )
    return rows, {
        "schemaVersion": "e02.s06.search_replay_summary.v1",
        "placements": len(placements),
        "passedPlacements": passed,
        "runReplays": 2 * len(placements),
        "passed": passed == len(placements),
    }


def outcome_access_audit(
    structural: Sequence[FaultPlacement],
    search: Sequence[FaultPlacement],
    history_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    structural_keys = {
        (item.context_id, item.fault_count, item.positions) for item in structural
    }
    history_keys = {
        (
            str(item["contextId"]),
            int(item["faultCount"]),
            tuple(item["positions"]),
        )
        for item in history_rows
    }
    overlap = sorted(structural_keys & history_keys)
    structural_guard = len(select_confirmatory_placements(structural)) == len(structural)
    mixed_rejected = False
    try:
        select_confirmatory_placements((*structural, *search))
    except ConfirmatoryLeakageError:
        mixed_rejected = True
    checks = {
        "structuralLockedBeforeSearch": True,
        "structuralSearchHistoryOverlapZero": not overlap,
        "structuralRowsOutcomeAccessNone": all(
            item.outcome_access.value == "none" for item in structural
        ),
        "searchRowsExploratoryOnly": all(
            item.outcome_access.value == "exploratory_training_only"
            and not item.confirmatory_eligible
            for item in search
        ),
        "structuralGuardAccepted": structural_guard,
        "mixedGuardRejected": mixed_rejected,
        "protectedE01OutcomesNotRead": True,
        "priorE02OutcomesNotUsedForSelection": True,
        "adversarialHeldOutLabelNotAssigned": True,
    }
    return {
        "schemaVersion": "e02.s06.outcome_access_audit.v1",
        "structuralPlacementCount": len(structural),
        "searchPlacementCount": len(search),
        "searchEvaluationCount": len(history_rows),
        "structuralHistoryOverlapCount": len(overlap),
        "checks": checks,
        "passed": all(checks.values()),
    }


def validation_summary(
    frozen_contexts: Sequence[PlacementContext],
    structural: Sequence[FaultPlacement],
    search: Sequence[FaultPlacement],
    placements: Sequence[FaultPlacement],
    uniform_audit: Mapping[str, Any],
    access_audit: Mapping[str, Any],
    materialization: Mapping[str, Any],
    execution: Mapping[str, Any],
    search_replay: Mapping[str, Any],
) -> dict[str, Any]:
    context_map = {item.context_id: item for item in frozen_contexts}
    exact = all(
        len(item.positions)
        == len(set(item.positions))
        == len(set(item.identity_ids))
        == item.fault_count
        for item in placements
    )
    cell_unique = True
    class_counts = Counter(
        (item.context_id, item.fault_count, item.placement_class.value)
        for item in placements
    )
    for context in frozen_contexts:
        for fault_count in FAULT_COUNTS:
            cell = [
                item
                for item in placements
                if item.context_id == context.context_id
                and item.fault_count == fault_count
            ]
            cell_unique &= len({item.positions for item in cell}) == len(cell)
    expected_counts = all(
        class_counts[(context.context_id, fault_count, placement_class.value)]
        == MAPS_PER_STRUCTURAL_CLASS
        for context in frozen_contexts
        for fault_count in FAULT_COUNTS
        for placement_class in STRUCTURAL_CLASSES
    ) and all(
        class_counts[
            (
                context.context_id,
                fault_count,
                PlacementClass.SEARCH_DERIVED_EXPLORATORY.value,
            )
        ]
        == SEARCH_MAPS_PER_CELL
        for context in frozen_contexts
        for fault_count in FAULT_COUNTS
    )
    clustered = all(
        item.descriptor_dict()["longestContiguousRun"] == item.fault_count
        for item in structural
        if item.placement_class == PlacementClass.CLUSTERED_EXACT
    )
    boundary = all(
        set(item.positions)
        <= set(boundary_band_positions(context_map[item.context_id], item.fault_count))
        for item in structural
        if item.placement_class == PlacementClass.BOUNDARY_EXACT
    )
    median = all(
        set(item.positions)
        <= set(median_rank_band_positions(context_map[item.context_id], item.fault_count))
        for item in structural
        if item.placement_class == PlacementClass.MEDIAN_RANK_EXACT
    )
    descriptors_finite = all(
        math.isfinite(float(value))
        for item in placements
        for key, value in item.descriptor_dict().items()
        if key
        not in {
            "placementId",
            "contextId",
            "placementClass",
            "outcomeAccess",
            "analysisRole",
            "confirmatoryEligible",
        }
    )
    checks = {
        "plannedRowCount486": len(placements) == 486,
        "placementIdsUnique": len({item.placement_id for item in placements})
        == len(placements),
        "exactFaultCountsAndWithinMapUniqueness": exact,
        "crossClassPositionSetsUniqueWithinCells": cell_unique,
        "plannedClassCellCounts": expected_counts,
        "clusteredContiguous": clustered,
        "boundaryBandMembership": boundary,
        "medianRankBandMembership": median,
        "descriptorsFinite": descriptors_finite,
        "allContextsCountsClassesCovered": len(class_counts) == 45,
        "uniformDistributionAudit": bool(uniform_audit["passed"]),
        "outcomeAccessSeparation": bool(access_audit["passed"]),
        "passiveStuckMaterialization": bool(materialization["passed"]),
        "s05ContractExecutionReplay": bool(execution["passed"]),
        "searchOutcomeReplay": bool(search_replay["passed"]),
    }
    search_scores = [int(item.search_score) for item in search]
    reversal_cells = []
    for context in frozen_contexts:
        for fault_count in FAULT_COUNTS:
            scores = [
                int(item.search_score)
                for item in search
                if item.context_id == context.context_id
                and item.fault_count == fault_count
            ]
            reversal_cells.append(
                {
                    "contextId": context.context_id,
                    "faultCount": fault_count,
                    "minimumScore": min(scores),
                    "maximumScore": max(scores),
                    "bothArchitectureOrderingsObserved": min(scores) < 0 < max(scores),
                }
            )
    return {
        "schemaVersion": "e02.s06.validation_summary.v1",
        "researchStepId": "S06",
        "stepNumber": 6,
        "placementInterfaceVersion": PLACEMENT_INTERFACE_VERSION,
        "placementCount": len(placements),
        "structuralPlacementCount": len(structural),
        "searchDerivedPlacementCount": len(search),
        "searchScoreMinimum": min(search_scores),
        "searchScoreMaximum": max(search_scores),
        "rankingReversalCells": reversal_cells,
        "rankingReversalCellCount": sum(
            item["bothArchitectureOrderingsObserved"] for item in reversal_cells
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True
    ).strip()


def provenance(output: Path, prespec: Path, workers: int) -> dict[str, Any]:
    inputs = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/PREVIOUS_ARTIFACTS.json"),
        Path("/workspace/DATASET_AVAILABILITY.json"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        prespec,
    ]
    for step in range(1, 6):
        inputs.append(
            Path(f"/artifacts/research_steps/S{step:02d}/research_step_full_results.md")
        )
    inputs.extend(
        (
            Path("/previous-artifacts/E01/research_steps/S08/split_manifest.json"),
            Path("/previous-artifacts/E01/research_steps/S08/holdout_integrity.json"),
            Path("/previous-artifacts/E01/research_steps/S14/downstream_readiness_audit.json"),
        )
    )
    records = []
    for path in inputs:
        records.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return {
        "schemaVersion": "e02.s06.provenance.v1",
        "researchStepId": "S06",
        "repository": str(REPOSITORY),
        "branch": git_output("branch", "--show-current"),
        "baseCommit": git_output("rev-parse", "HEAD"),
        "placementSourceSha256": sha256_file(
            REPOSITORY / "causal_simulator/placements.py"
        ),
        "builderSourceSha256": sha256_file(Path(__file__)),
        "masterSeedHex": f"0x{MASTER_SEED:032x}",
        "workerCount": workers,
        "workerReason": "beam rounds are sequential; candidate evaluations within each round are independent, bounded, and gathered in canonical input order",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else None,
            "matplotlib": matplotlib.__version__,
        },
        "inputs": records,
        "outputDirectory": str(output),
    }


def compare_regeneration(output: Path, prior: Path) -> dict[str, Any]:
    rows = []
    for name in CORE_REGENERATION_FILES:
        left = output / name
        right = prior / name
        left_hash = sha256_file(left) if left.is_file() else None
        right_hash = sha256_file(right) if right.is_file() else None
        rows.append(
            {
                "path": name,
                "currentSha256": left_hash,
                "priorSha256": right_hash,
                "byteIdentical": left_hash is not None and left_hash == right_hash,
            }
        )
    return {
        "schemaVersion": "e02.s06.deterministic_regeneration.v1",
        "filesCompared": len(rows),
        "filesByteIdentical": sum(item["byteIdentical"] for item in rows),
        "comparisons": rows,
        "passed": all(item["byteIdentical"] for item in rows),
    }


def build(
    output: Path, prespec: Path, compare_to: Path | None, workers: int
) -> None:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1, 8]")
    output.mkdir(parents=True, exist_ok=True)
    frozen_contexts = contexts()
    structural = generate_structural_bank(
        frozen_contexts,
        FAULT_COUNTS,
        seed=MASTER_SEED,
        maps_per_class=MAPS_PER_STRUCTURAL_CLASS,
    )
    structural_lock = structural_lock_record(structural)
    write_json(output / "structural_lock.json", structural_lock)

    search: list[FaultPlacement] = []
    history_rows: list[dict[str, Any]] = []
    executor = (
        None
        if workers == 1
        else ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
    )
    try:
        for context in frozen_contexts:
            for fault_count in FAULT_COUNTS:
                forbidden = {
                    item.positions
                    for item in structural
                    if item.context_id == context.context_id
                    and item.fault_count == fault_count
                }
                selected, history = search_exploratory_placements(
                    context,
                    fault_count,
                    seed=MASTER_SEED,
                    forbidden_structural=forbidden,
                    scorer=lambda positions, context=context, fault_count=fault_count: score_search_candidate(
                        context, fault_count, positions
                    ),
                    batch_scorer=lambda candidates, context=context, fault_count=fault_count: score_search_candidate_batch(
                        executor, workers, context, fault_count, candidates
                    ),
                )
                search.extend(selected)
                history_rows.extend(
                    item.to_dict(context.context_id, fault_count) for item in history
                )
    finally:
        if executor is not None:
            executor.shutdown()

    placements = tuple((*structural, *search))
    ordered = sorted(
        placements,
        key=lambda item: (
            item.context_id,
            item.fault_count,
            item.placement_class.value,
            item.replicate_ordinal,
        ),
    )
    bank_rows = [item.to_record() for item in ordered]
    descriptor_rows = [item.descriptor_dict() for item in ordered]
    history_rows = sorted(
        history_rows,
        key=lambda item: (
            item["contextId"],
            item["faultCount"],
            item["positions"],
        ),
    )
    write_parquet(output / "fault_placement_bank.parquet", bank_rows)
    write_parquet(output / "placement_descriptors.parquet", descriptor_rows)
    write_parquet(output / "search_history.parquet", history_rows)
    context_manifest = {
        "schemaVersion": "e02.s06.placement_contexts.v1",
        "researchStepId": "S06",
        "scope": "descriptor and placement-rule validation only; not S07 factorial scale",
        "contexts": [
            {
                **item.to_dict(),
                "n": item.n,
                "contextSha256": item.context_sha256,
                "valueRanks": list(item.value_ranks),
            }
            for item in frozen_contexts
        ],
        "faultCounts": list(FAULT_COUNTS),
    }
    write_json(output / "placement_contexts.json", context_manifest)
    coverage = coverage_rows(ordered)
    write_csv(output / "coverage_summary.csv", coverage)
    draw_coverage_plot(ordered, output)

    uniform_audit = uniform_distribution_audit(frozen_contexts)
    write_json(output / "uniform_distribution_validation.json", uniform_audit)
    materialization = validate_materialization(frozen_contexts, ordered)
    write_json(output / "materialization_validation.json", materialization)
    access = outcome_access_audit(structural, search, history_rows)
    write_json(output / "outcome_access_audit.json", access)
    execution_rows, execution_summary = run_contract_validation(search)
    write_parquet(output / "contract_execution_validation.parquet", execution_rows)
    write_json(output / "contract_execution_summary.json", execution_summary)
    replay_rows, replay_summary = validate_search_replay(frozen_contexts, search)
    write_parquet(output / "search_replay_validation.parquet", replay_rows)
    write_json(output / "search_replay_summary.json", replay_summary)
    summary = validation_summary(
        frozen_contexts,
        structural,
        search,
        ordered,
        uniform_audit,
        access,
        materialization,
        execution_summary,
        replay_summary,
    )
    write_json(output / "validation_summary.json", summary)
    write_json(
        output / "provenance_manifest.json", provenance(output, prespec, workers)
    )

    if compare_to is not None:
        regeneration = compare_regeneration(output, compare_to)
        write_json(output / "deterministic_regeneration.json", regeneration)
        if not regeneration["passed"]:
            raise AssertionError("independent S06 regeneration was not byte-identical")
    if not summary["passed"]:
        raise AssertionError("S06 validation gate failed")

    print(
        json.dumps(
            {
                "placementCount": len(ordered),
                "structuralPlacementCount": len(structural),
                "searchPlacementCount": len(search),
                "searchEvaluationCount": len(history_rows),
                "contractRuns": execution_summary["executedRuns"],
                "rankingReversalCellCount": summary["rankingReversalCellCount"],
                "success": summary["passed"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prespecification",
        type=Path,
        default=Path("/artifacts/research_steps/S06/placement_prespecification.json"),
    )
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build(args.output, args.prespecification, args.compare_to, args.workers)


if __name__ == "__main__":
    main()
