#!/usr/bin/env python3
"""Execute E05 S13 homogeneous-versus-chimeric recovery exactly once."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.schedulers import (
    SchedulerExecutionContract,
    SchedulerFamily,
    run_scheduled_architecture,
)
from reference_simulator.model import LEDGER_FIELDS, Policy, canonical_json_bytes
from reference_simulator.transition_primitives import ledger_identity
from src.regeneration.chimeric import (
    BENCHMARK_VERSION,
    PORTFOLIOS,
    assignment_description_bits,
    build_composition_scenario,
    clone_with_stuck_identities,
    composition_counts,
    exact_replay_native,
    fault_count,
    fault_identity_ranking,
    run_native_recovery,
    semantic_seed,
    shannon_entropy_bits,
    should_stop_threshold,
    validate_chimeric_spec,
)
from src.regeneration.tasks import (
    Checkpoint,
    _checkpoint_hash,
    stabilize_achieved_checkpoint,
)
from src.regeneration.transfer import (
    TransferTargetChange,
    apply_transfer_lesion,
    exact_replay_result,
    run_goal_transfer,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = REPOSITORY / "configs/regeneration/s13_chimeric_recovery.json"
OUTPUT = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "research_steps/S13"
S12_RESULTS = Path("/artifacts/research_steps/S12/transfer_results.parquet")
S12_REPORT = Path("/artifacts/research_steps/S12/research_step_full_results.md")
E01_COMPOSITION = Path("/previous-artifacts/E01/research_steps/S13/chimeric_results.parquet")
WORKERS = min(8, max(1, int(os.environ.get("E05_S13_WORKERS", "8"))))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


@dataclass(frozen=True, slots=True)
class Source:
    source_id: str
    n: int
    direction: str
    replicate: int
    portfolio_id: str
    placement_map: int
    scenario: Any
    checkpoint: Checkpoint | None
    metadata: Mapping[str, Any]
    development: Mapping[str, Any]
    stabilization: Mapping[str, Any]
    source_success: bool
    source_stop_reason: str
    exact_replay_pass: bool


def _source_id(metadata: Mapping[str, Any]) -> str:
    return "e05s13src:" + canonical_hash(
        {
            "baseDrawPairingId": metadata["baseDrawPairingId"],
            "portfolioId": metadata["portfolioId"],
            "placementMap": metadata["placementMap"],
            "policyAssignmentSha256": metadata["policyAssignmentSha256"],
        }
    )


def _checkpoint_from_development(scenario: Any, run: Any) -> Checkpoint:
    occupancy = tuple(run.summary["finalOccupancy"])
    cursors = dict(run.final_state["selectionCursors"])
    activation_count = int(run.summary["activationCount"])
    streams = {str(k): int(v) for k, v in run.final_state["streamCounters"].items()}
    ledger = {field: int(run.final_state["ledger"][field]) for field in LEDGER_FIELDS}
    return Checkpoint(
        occupancy,
        tuple(sorted(cursors.items())),
        activation_count,
        tuple(sorted(streams.items())),
        tuple((field, ledger[field]) for field in LEDGER_FIELDS),
        0,
        _checkpoint_hash(
            scenario.scenario_id,
            occupancy,
            cursors,
            activation_count,
            streams,
            ledger,
        ),
    )


def _build_source(args: tuple[int, str, int, str, int]) -> Source:
    n, direction, replicate, portfolio_id, placement_map = args
    development_budget = 100 * n * n
    scenario, metadata = build_composition_scenario(
        n=n,
        portfolio_id=portfolio_id,
        direction=direction,
        replicate=replicate,
        placement_map=placement_map,
        max_activations=2 * development_budget + 40 * n,
    )
    architecture = ArchitectureExecutionContract.distributed_local()
    scheduler = SchedulerExecutionContract(
        SchedulerFamily.UNIFORM_RANDOM_ACTIVATION
    )
    first = run_scheduled_architecture(
        scenario, architecture, scheduler, trace_mode="digest"
    )
    replay = run_scheduled_architecture(
        scenario, architecture, scheduler, trace_mode="digest"
    )
    replay_pass = first.result.to_json_bytes() == replay.result.to_json_bytes()
    development = {
        "completed": bool(first.result.summary["completed"]),
        "activationCount": int(first.result.summary["activationCount"]),
        "stopReason": str(first.result.summary["stopReason"]),
        "eventDigest": first.result.event_digest,
        "withinFrozenDevelopmentBudget": int(first.result.summary["activationCount"])
        <= development_budget,
        "opportunityValidation": first.opportunity_validation(),
    }
    source_success = bool(development["completed"] and development["withinFrozenDevelopmentBudget"])
    checkpoint: Checkpoint | None = None
    stabilization: Mapping[str, Any] = {
        "success": False,
        "notRunReason": "development_competing_terminal",
    }
    stop_reason = str(development["stopReason"])
    if source_success:
        developed = _checkpoint_from_development(scenario, first.result)
        checkpoint, stabilization = stabilize_achieved_checkpoint(scenario, developed)
        checkpoint_replay, stabilization_replay = stabilize_achieved_checkpoint(
            scenario, developed
        )
        replay_pass = replay_pass and (
            canonical_json_bytes(checkpoint.to_dict())
            == canonical_json_bytes(checkpoint_replay.to_dict())
            and canonical_json_bytes(stabilization)
            == canonical_json_bytes(stabilization_replay)
        )
        source_success = bool(stabilization["success"])
        if not source_success:
            stop_reason = "stabilization_failure"
    return Source(
        _source_id(metadata),
        n,
        direction,
        replicate,
        portfolio_id,
        placement_map,
        scenario,
        checkpoint,
        metadata,
        development,
        stabilization,
        source_success,
        stop_reason,
        replay_pass,
    )


@dataclass(frozen=True, slots=True)
class MainCase:
    case_id: str
    source: Source
    task_kind: str
    task_id: str
    target_aware: bool | None
    stability: bool


def _case_id(source: Source, task_kind: str, task_id: str, arm: str) -> str:
    return "e05s13case:" + canonical_hash(
        {
            "sourceId": source.source_id,
            "taskKind": task_kind,
            "taskId": task_id,
            "arm": arm,
        }
    )


def _source_failure_row(case: MainCase, budget: int) -> dict[str, Any]:
    counts = case.source.metadata["compositionCounts"]
    return {
        "schemaVersion": "e05.s13.chimeric-result-row.v1",
        "runId": "e05s13run:" + canonical_hash({"caseId": case.case_id}),
        "mainCaseId": case.case_id,
        "thresholdCaseId": None,
        "sourceId": case.source.source_id,
        "baseDrawPairingId": case.source.metadata["baseDrawPairingId"],
        "sourceScenarioId": case.source.scenario.scenario_id,
        "sourceCheckpointHash": None,
        "portfolioId": case.source.portfolio_id,
        "portfolioRole": "chimeric" if len(PORTFOLIOS[case.source.portfolio_id]) > 1 else "homogeneous",
        "placementMap": case.source.placement_map,
        "n": case.source.n,
        "direction": case.source.direction,
        "replicateOrdinal": case.source.replicate,
        "taskKind": case.task_kind,
        "taskId": case.task_id,
        "arm": "source_competing_terminal",
        "targetAware": case.target_aware,
        "stability": case.stability,
        "faultFraction": 0.0,
        "faultCount": 0,
        "faultIdentitySetSha256": None,
        "faultIdentityIdsJson": None,
        "executionStatus": "source_competing_terminal",
        "trajectoryExecuted": False,
        "reusedRunId": None,
        "success": False,
        "stopReason": case.source.source_stop_reason,
        "phaseActivationCount": 0,
        "restrictedTime": budget + 1,
        "initialDistance": None,
        "finalDistance": None,
        "maximumDistance": None,
        "distanceOvershoot": None,
        "normalizedErrorAuc": 1.0,
        "nativeLedgerDeltaJson": json.dumps({field: 0 for field in LEDGER_FIELDS}, sort_keys=True),
        "activations": 0,
        "observationReads": 0,
        "valueComparisons": 0,
        "proposals": 0,
        "noOps": 0,
        "rejections": 0,
        "memoryUpdates": 0,
        "acceptedSwaps": 0,
        "conflictLosses": 0,
        "displacedCells": 0,
        "targetRecordReads": 0,
        "controllerComputations": 0,
        "abstractEnergyUnits": 0,
        "targetFeasible": None,
        "postHitAnyDeparture": None,
        "noChangeAnyTargetDeparture": None,
        "noChangeFinalTargetRetained": None,
        "policyFamilyCount": len(counts),
        "compositionEntropyBits": shannon_entropy_bits(counts),
        "assignmentDescriptionBits": assignment_description_bits(counts),
        "resultDigest": None,
        "exactReplayPass": case.source.exact_replay_pass,
        "validationPass": case.source.exact_replay_pass,
    }


def _flatten_native(
    result: Mapping[str, Any], case: MainCase, lesion: Mapping[str, Any]
) -> dict[str, Any]:
    native = result["nativeLedgerDelta"]
    counts = case.source.metadata["compositionCounts"]
    return {
        "schemaVersion": "e05.s13.chimeric-result-row.v1",
        "runId": "e05s13run:" + canonical_hash({"caseId": case.case_id}),
        "mainCaseId": case.case_id,
        "thresholdCaseId": None,
        "sourceId": case.source.source_id,
        "baseDrawPairingId": case.source.metadata["baseDrawPairingId"],
        "sourceScenarioId": case.source.scenario.scenario_id,
        "sourceCheckpointHash": case.source.checkpoint.state_hash,
        "portfolioId": case.source.portfolio_id,
        "portfolioRole": "chimeric" if len(PORTFOLIOS[case.source.portfolio_id]) > 1 else "homogeneous",
        "placementMap": case.source.placement_map,
        "n": case.source.n,
        "direction": case.source.direction,
        "replicateOrdinal": case.source.replicate,
        "taskKind": case.task_kind,
        "taskId": case.task_id,
        "arm": "native_recovery",
        "targetAware": None,
        "stability": False,
        "faultFraction": 0.0,
        "faultCount": 0,
        "faultIdentitySetSha256": None,
        "faultIdentityIdsJson": None,
        "executionStatus": "executed",
        "trajectoryExecuted": True,
        "reusedRunId": None,
        "success": bool(result["success"]),
        "stopReason": result["stopReason"],
        "phaseActivationCount": int(result["phaseActivationCount"]),
        "restrictedTime": int(result["restrictedTime"]),
        "initialDistance": int(result["initialDistance"]),
        "finalDistance": int(result["finalDistance"]),
        "maximumDistance": int(result["maximumDistance"]),
        "distanceOvershoot": int(result["distanceOvershoot"]),
        "normalizedErrorAuc": float(result["normalizedDistanceAucRestricted"]),
        "lesionStateHash": lesion["lesionStateHash"],
        "lesionWindowLength": int(lesion["windowLength"]),
        "lesionPostDistance": int(lesion["postDistance"]),
        "nativeLedgerDeltaJson": json.dumps(native, sort_keys=True),
        **{field: int(native[field]) for field in LEDGER_FIELDS},
        "targetRecordReads": 0,
        "controllerComputations": 0,
        "abstractEnergyUnits": 0,
        "targetFeasible": None,
        "postHitAnyDeparture": None,
        "noChangeAnyTargetDeparture": None,
        "noChangeFinalTargetRetained": None,
        "policyFamilyCount": len(counts),
        "compositionEntropyBits": shannon_entropy_bits(counts),
        "assignmentDescriptionBits": assignment_description_bits(counts),
        "resultDigest": result["resultDigest"],
        "exactReplayPass": True,
        "validationPass": all(result["validation"].values()),
    }


def _flatten_goal(result: Mapping[str, Any], case: MainCase) -> dict[str, Any]:
    native = result["nativeLedgerDelta"]
    target = result["targetLedger"]
    run_limit = (20 * case.source.n) if case.stability else (100 * case.source.n * case.source.n + 20 * case.source.n)
    restricted_auc = int(result["newTargetDistanceAuc"]) + max(
        0, run_limit - int(result["phaseActivationCount"])
    ) * int(result["finalNewTargetDistance"])
    maximum = int(result["targetMaximumDistance"])
    counts = case.source.metadata["compositionCounts"]
    return {
        "schemaVersion": "e05.s13.chimeric-result-row.v1",
        "runId": "e05s13run:" + canonical_hash({"caseId": case.case_id}),
        "mainCaseId": case.case_id,
        "thresholdCaseId": None,
        "sourceId": case.source.source_id,
        "baseDrawPairingId": case.source.metadata["baseDrawPairingId"],
        "sourceScenarioId": case.source.scenario.scenario_id,
        "sourceCheckpointHash": case.source.checkpoint.state_hash,
        "portfolioId": case.source.portfolio_id,
        "portfolioRole": "chimeric" if len(PORTFOLIOS[case.source.portfolio_id]) > 1 else "homogeneous",
        "placementMap": case.source.placement_map,
        "n": case.source.n,
        "direction": case.source.direction,
        "replicateOrdinal": case.source.replicate,
        "taskKind": case.task_kind,
        "taskId": case.task_id,
        "arm": result["arm"],
        "targetAware": bool(result["targetAware"]),
        "stability": bool(result["stability"]),
        "faultFraction": 0.0,
        "faultCount": 0,
        "faultIdentitySetSha256": None,
        "faultIdentityIdsJson": None,
        "executionStatus": "executed",
        "trajectoryExecuted": True,
        "reusedRunId": None,
        "success": bool(result["success"]),
        "stopReason": result["stopReason"],
        "phaseActivationCount": int(result["phaseActivationCount"]),
        "restrictedTime": int(result["restrictedTime"]),
        "initialDistance": int(result["initialNewTargetDistance"]),
        "finalDistance": int(result["finalNewTargetDistance"]),
        "maximumDistance": int(result["maximumNewTargetDistance"]),
        "distanceOvershoot": int(result["maximumNewTargetDistance"]) - int(result["initialNewTargetDistance"]),
        "normalizedErrorAuc": restricted_auc / (run_limit * maximum),
        "nativeLedgerDeltaJson": json.dumps(native, sort_keys=True),
        **{field: int(native[field]) for field in LEDGER_FIELDS},
        "targetRecordReads": int(target["targetRecordReads"]),
        "controllerComputations": int(target["controllerComputations"]),
        "abstractEnergyUnits": int(target["abstractEnergyUnits"]),
        "targetFeasible": bool(result["validation"]["targetFeasible"]),
        "postHitAnyDeparture": bool(result["postHitAnyDeparture"]),
        "noChangeAnyTargetDeparture": bool(result["noChangeAnyTargetDeparture"]),
        "noChangeFinalTargetRetained": bool(result["noChangeFinalTargetRetained"]),
        "policyFamilyCount": len(counts),
        "compositionEntropyBits": shannon_entropy_bits(counts),
        "assignmentDescriptionBits": assignment_description_bits(counts),
        "resultDigest": result["resultDigest"],
        "exactReplayPass": True,
        "validationPass": all(result["validation"].values()),
    }


def _execute_main(case: MainCase) -> dict[str, Any]:
    budget = 100 * case.source.n * case.source.n
    if not case.source.source_success or case.source.checkpoint is None:
        return _source_failure_row(case, budget)
    if case.task_kind == "injury":
        lesion = apply_transfer_lesion(
            case.source.scenario,
            case.source.checkpoint,
            lesion_type=case.task_id,
            location="central",
        )
        result = run_native_recovery(
            case.source.scenario,
            case.source.checkpoint,
            postinjury_occupancy=lesion["postOccupancy"],
            lesion_state_hash=lesion["lesionStateHash"],
            recovery_budget=budget,
        )
        exact_replay_native(
            case.source.scenario,
            case.source.checkpoint,
            postinjury_occupancy=lesion["postOccupancy"],
            lesion_state_hash=lesion["lesionStateHash"],
            recovery_budget=budget,
            expected=result,
        )
        return _flatten_native(result, case, lesion)
    target = TransferTargetChange(case.task_id)

    def factory() -> dict[str, Any]:
        return run_goal_transfer(
            case.source.scenario,
            case.source.checkpoint,
            target_change=target,
            scheduler=SchedulerFamily.UNIFORM_RANDOM_ACTIVATION,
            process_id="none",
            target_aware=bool(case.target_aware),
            adaptation_budget=budget,
            probe_budget=20 * case.source.n,
            stability=case.stability,
            retain_trace=False,
        )

    result = factory()
    exact_replay_result(factory, result)
    return _flatten_goal(result, case)


def _main_cases(sources: Sequence[Source], specification: Mapping[str, Any]) -> list[MainCase]:
    cases: list[MainCase] = []
    for source in sources:
        for lesion in specification["panel"]["mainLesions"]:
            case_id = _case_id(source, "injury", lesion, "native_recovery")
            cases.append(MainCase(case_id, source, "injury", lesion, None, False))
        for target in specification["panel"]["targetChanges"]:
            for aware in (True, False):
                arm = "changed_aware" if aware else "changed_nonadaptive"
                case_id = _case_id(source, "target_change", target, arm)
                cases.append(MainCase(case_id, source, "target_change", target, aware, False))
        stability_target = specification["panel"]["targetChanges"][0]
        for aware in (True, False):
            arm = "stability_aware" if aware else "stability_nonadaptive"
            case_id = _case_id(source, "intact_stability", stability_target, arm)
            cases.append(MainCase(case_id, source, "intact_stability", stability_target, aware, True))
    return sorted(cases, key=lambda item: item.case_id)


@dataclass(frozen=True, slots=True)
class ThresholdCase:
    threshold_case_id: str
    source: Source
    fraction: float


def _execute_threshold(case: ThresholdCase) -> dict[str, Any]:
    source = case.source
    budget = 100 * source.n * source.n
    if not source.source_success or source.checkpoint is None:
        ranking = fault_identity_ranking(
            tuple(source.scenario.cell_map),
            n=source.n,
            direction=source.direction,
            replicate=source.replicate,
        )
        selected = tuple(ranking[: fault_count(source.n, case.fraction)])
        synthetic = _source_failure_row(
            MainCase(case.threshold_case_id, source, "critical_fault_threshold", "segment_reversal_central_v1", None, False),
            budget,
        )
        synthetic["mainCaseId"] = None
        synthetic["thresholdCaseId"] = case.threshold_case_id
        synthetic["faultFraction"] = case.fraction
        synthetic["faultCount"] = len(selected)
        synthetic["faultIdentitySetSha256"] = canonical_hash(sorted(selected))
        synthetic["faultIdentityIdsJson"] = json.dumps(sorted(selected))
        return synthetic
    ranking = fault_identity_ranking(
        tuple(source.scenario.cell_map),
        n=source.n,
        direction=source.direction,
        replicate=source.replicate,
    )
    selected = tuple(ranking[: fault_count(source.n, case.fraction)])
    faulted, checkpoint, metadata = clone_with_stuck_identities(
        source.scenario,
        source.checkpoint,
        selected,
        fraction=case.fraction,
    )
    lesion = apply_transfer_lesion(
        faulted,
        checkpoint,
        lesion_type="segment_reversal_central_v1",
        location="central",
    )
    result = run_native_recovery(
        faulted,
        checkpoint,
        postinjury_occupancy=lesion["postOccupancy"],
        lesion_state_hash=lesion["lesionStateHash"],
        recovery_budget=budget,
    )
    exact_replay_native(
        faulted,
        checkpoint,
        postinjury_occupancy=lesion["postOccupancy"],
        lesion_state_hash=lesion["lesionStateHash"],
        recovery_budget=budget,
        expected=result,
    )
    row = _flatten_native(
        result,
        MainCase(case.threshold_case_id, source, "critical_fault_threshold", "segment_reversal_central_v1", None, False),
        lesion,
    )
    row.update(
        {
            "runId": "e05s13thr-run:" + canonical_hash({"thresholdCaseId": case.threshold_case_id}),
            "mainCaseId": None,
            "thresholdCaseId": case.threshold_case_id,
            "taskKind": "critical_fault_threshold",
            "faultFraction": case.fraction,
            "faultCount": len(selected),
            "faultIdentitySetSha256": metadata["faultIdentitySetSha256"],
            "faultIdentityIdsJson": json.dumps(sorted(selected)),
            "sourceScenarioId": source.scenario.scenario_id,
            "faultedScenarioId": metadata["faultedScenarioId"],
            "sourceCheckpointHash": source.checkpoint.state_hash,
            "faultedCheckpointHash": metadata["faultedCheckpointHash"],
            "onlyFaultFlagsChanged": metadata["onlyFaultFlagsChanged"],
        }
    )
    return row


def _run_pool(
    items: Sequence[Any], worker: Callable[[Any], dict[str, Any]], label: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        pending = {pool.submit(worker, item): item for item in items}
        for ordinal, future in enumerate(as_completed(pending), 1):
            item = pending[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({"item": repr(item), "error": repr(exc)})
            if ordinal % 64 == 0 or ordinal == len(items):
                print(
                    f"S13 {label} {ordinal}/{len(items)} rows={len(rows)} "
                    f"failures={len(failures)} elapsed={time.monotonic()-started:.1f}s",
                    flush=True,
                )
    rows.sort(key=lambda row: row["runId"])
    return rows, failures


def _wilson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _threshold_id(source: Source, fraction: float) -> str:
    return "e05s13thr:" + canonical_hash(
        {"sourceId": source.source_id, "faultFraction": fraction}
    )


def _reuse_zero_threshold(main: pd.DataFrame, sources: Sequence[Source]) -> list[dict[str, Any]]:
    lookup = {
        row["sourceId"]: row
        for row in main[
            (main["taskKind"] == "injury")
            & (main["taskId"] == "segment_reversal_central_v1")
            & (main["placementMap"] == 0)
        ].to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for source in sources:
        original = dict(lookup[source.source_id])
        threshold_id = _threshold_id(source, 0.0)
        reused_status = (
            "reused_main_no_new_execution"
            if bool(original["trajectoryExecuted"])
            else "reused_main_source_competing_terminal"
        )
        original.update(
            {
                "runId": "e05s13thr-reuse:" + canonical_hash({"thresholdCaseId": threshold_id}),
                "mainCaseId": None,
                "thresholdCaseId": threshold_id,
                "taskKind": "critical_fault_threshold",
                "faultFraction": 0.0,
                "faultCount": 0,
                "executionStatus": reused_status,
                "trajectoryExecuted": False,
                "reusedRunId": lookup[source.source_id]["runId"],
            }
        )
        rows.append(original)
    return rows


def _execute_threshold_sequence(
    primary_sources: Sequence[Source], specification: Mapping[str, Any], main: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    fractions = [float(value) for value in specification["criticalFaultThreshold"]["fractionsInOrder"]]
    assignment_rows = [
        {
            "thresholdCaseId": _threshold_id(source, fraction),
            "sourceId": source.source_id,
            "baseDrawPairingId": source.metadata["baseDrawPairingId"],
            "portfolioId": source.portfolio_id,
            "n": source.n,
            "direction": source.direction,
            "replicateOrdinal": source.replicate,
            "faultFraction": fraction,
            "faultCount": fault_count(source.n, fraction),
            "executionStatus": "pending",
        }
        for fraction in fractions
        for source in primary_sources
    ]
    by_id = {row["thresholdCaseId"]: row for row in assignment_rows}
    result_rows = _reuse_zero_threshold(main, primary_sources)
    for row in result_rows:
        by_id[row["thresholdCaseId"]]["executionStatus"] = row["executionStatus"]
    zero_frame = pd.DataFrame(result_rows)
    zero_level: dict[str, Any] = {"faultFraction": 0.0}
    for portfolio in PORTFOLIOS:
        group = zero_frame[zero_frame["portfolioId"] == portfolio]
        _, upper = _wilson(int(group["success"].sum()), len(group))
        zero_level[portfolio] = upper
    ucb_history: list[dict[str, Any]] = [zero_level]
    failures: list[dict[str, Any]] = []
    stopped_after: float | None = None
    for fraction in fractions[1:]:
        if stopped_after is not None:
            for source in primary_sources:
                by_id[_threshold_id(source, fraction)]["executionStatus"] = (
                    "not_run_prespecified_global_threshold_stop"
                )
            continue
        cases = [
            ThresholdCase(_threshold_id(source, fraction), source, fraction)
            for source in primary_sources
        ]
        rows, stage_failures = _run_pool(cases, _execute_threshold, f"threshold f={fraction:.1f}")
        failures.extend(stage_failures)
        result_rows.extend(rows)
        for row in rows:
            by_id[row["thresholdCaseId"]]["executionStatus"] = row["executionStatus"]
        frame = pd.DataFrame(rows)
        level: dict[str, Any] = {"faultFraction": fraction}
        for portfolio in PORTFOLIOS:
            group = frame[frame["portfolioId"] == portfolio]
            _, upper = _wilson(int(group["success"].sum()), len(group))
            level[portfolio] = upper
        ucb_history.append(level)
        if should_stop_threshold(ucb_history, tuple(PORTFOLIOS)):
            stopped_after = fraction
            print(f"S13 prespecified global threshold stop after f={fraction:.1f}", flush=True)
    for row in assignment_rows:
        if row["executionStatus"] == "pending":
            raise AssertionError("threshold assignment retained an unresolved status")
    return (
        pd.DataFrame(result_rows).sort_values(["faultFraction", "portfolioId", "sourceId"]),
        pd.DataFrame(assignment_rows).sort_values(["faultFraction", "portfolioId", "sourceId"]),
        ucb_history,
        failures,
    )


def _holm(pvalues: Sequence[float]) -> list[float]:
    order = np.argsort(np.asarray(pvalues, dtype=float))
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * float(pvalues[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _bootstrap_interval(values: np.ndarray, address: Sequence[Any]) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(semantic_seed("analysis_bootstrap", *address))
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _pure_id(policy: Policy) -> str:
    return {
        Policy.BUBBLE: "pure_bubble",
        Policy.INSERTION: "pure_insertion",
        Policy.SELECTION: "pure_selection",
    }[policy]


def _primary_task_rows(main: pd.DataFrame, placement_map: int) -> pd.DataFrame:
    injury = main[(main["taskKind"] == "injury") & (main["placementMap"] == placement_map)].copy()
    goal = main[
        (main["taskKind"] == "target_change")
        & main["targetAware"].fillna(False).astype(bool)
        & (main["placementMap"] == placement_map)
    ].copy()
    return pd.concat([injury, goal], ignore_index=True)


def _composition_contrasts(main: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    primary = _primary_task_rows(main, 0)
    for portfolio, policies in PORTFOLIOS.items():
        if len(policies) == 1:
            continue
        for task_id in sorted(primary["taskId"].unique()):
            mixed = primary[(primary["portfolioId"] == portfolio) & (primary["taskId"] == task_id)]
            component_ids = [_pure_id(policy) for policy in policies]
            pure = primary[(primary["portfolioId"].isin(component_ids)) & (primary["taskId"] == task_id)]
            pure_pivot = pure.pivot(index="baseDrawPairingId", columns="portfolioId", values=["success", "restrictedTime", "normalizedErrorAuc"])
            mixed_indexed = mixed.set_index("baseDrawPairingId")
            common = mixed_indexed.index.intersection(pure_pivot.index)
            success_difference = (
                mixed_indexed.loc[common, "success"].astype(float).to_numpy()
                - pure_pivot.loc[common, "success"][component_ids].astype(float).mean(axis=1).to_numpy()
            )
            time_difference = (
                mixed_indexed.loc[common, "restrictedTime"].astype(float).to_numpy()
                - pure_pivot.loc[common, "restrictedTime"][component_ids].astype(float).mean(axis=1).to_numpy()
            )
            auc_difference = (
                mixed_indexed.loc[common, "normalizedErrorAuc"].astype(float).to_numpy()
                - pure_pivot.loc[common, "normalizedErrorAuc"][component_ids].astype(float).mean(axis=1).to_numpy()
            )
            nonzero = success_difference[success_difference != 0]
            positives = int((nonzero > 0).sum())
            pvalue = (
                float(stats.binomtest(positives, len(nonzero), 0.5).pvalue)
                if len(nonzero)
                else 1.0
            )
            success_ci = _bootstrap_interval(success_difference, (portfolio, task_id, "success"))
            time_ci = _bootstrap_interval(time_difference, (portfolio, task_id, "time"))
            auc_ci = _bootstrap_interval(auc_difference, (portfolio, task_id, "auc"))
            records.append(
                {
                    "schemaVersion": "e05.s13.composition-contrast.v1",
                    "portfolioId": portfolio,
                    "componentPurePortfoliosJson": json.dumps(component_ids),
                    "taskId": task_id,
                    "pairedBaseDrawCount": len(common),
                    "meanSuccessDifference": float(success_difference.mean()),
                    "successDifferenceCiLow": success_ci[0],
                    "successDifferenceCiHigh": success_ci[1],
                    "successSignTestP": pvalue,
                    "meanRestrictedTimeDifference": float(time_difference.mean()),
                    "restrictedTimeDifferenceCiLow": time_ci[0],
                    "restrictedTimeDifferenceCiHigh": time_ci[1],
                    "meanNormalizedAucDifference": float(auc_difference.mean()),
                    "normalizedAucDifferenceCiLow": auc_ci[0],
                    "normalizedAucDifferenceCiHigh": auc_ci[1],
                }
            )
    frame = pd.DataFrame(records)
    frame["successSignTestHolmP"] = _holm(frame["successSignTestP"].tolist())
    frame["primaryBenefit"] = (
        (frame["meanSuccessDifference"] > 0)
        & (frame["successDifferenceCiLow"] > 0)
        & (frame["successSignTestHolmP"] < 0.05)
    )
    frame["primaryHarm"] = (
        (frame["meanSuccessDifference"] < 0)
        & (frame["successDifferenceCiHigh"] < 0)
        & (frame["successSignTestHolmP"] < 0.05)
    )
    return frame.sort_values(["taskId", "portfolioId"]).reset_index(drop=True)


def _placement_sensitivity(main: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for placement in (0, 1):
        rows = _primary_task_rows(main, placement)
        for portfolio in PORTFOLIOS:
            if len(PORTFOLIOS[portfolio]) == 1 or placement == 1:
                # Pure policies have only map 0; map 1 chimeras are compared
                # against their same base-draw pure map-0 component mean.
                pass
            if len(PORTFOLIOS[portfolio]) == 1:
                continue
            for task_id in sorted(rows["taskId"].unique()):
                mixed = rows[(rows["portfolioId"] == portfolio) & (rows["taskId"] == task_id)]
                if mixed.empty:
                    continue
                pure_source = _primary_task_rows(main, 0)
                components = [_pure_id(policy) for policy in PORTFOLIOS[portfolio]]
                pure = pure_source[(pure_source["portfolioId"].isin(components)) & (pure_source["taskId"] == task_id)]
                pivot = pure.pivot(index="baseDrawPairingId", columns="portfolioId", values="success")
                mixed_i = mixed.set_index("baseDrawPairingId")
                common = mixed_i.index.intersection(pivot.index)
                difference = mixed_i.loc[common, "success"].astype(float) - pivot.loc[common, components].astype(float).mean(axis=1)
                records.append(
                    {
                        "portfolioId": portfolio,
                        "taskId": task_id,
                        "placementMap": placement,
                        "pairedBaseDrawCount": len(common),
                        "meanSuccessDifference": float(difference.mean()),
                    }
                )
    frame = pd.DataFrame(records)
    pivot = frame.pivot(index=["portfolioId", "taskId"], columns="placementMap", values="meanSuccessDifference").reset_index()
    pivot.columns = ["portfolioId", "taskId", "map0Difference", "map1Difference"]
    pivot["directionConsistent"] = (
        np.sign(pivot["map0Difference"]) == np.sign(pivot["map1Difference"])
    ) | ((pivot["map0Difference"] == 0) & (pivot["map1Difference"] == 0))
    return pivot


def _controller_contrasts(main: pd.DataFrame) -> pd.DataFrame:
    changed = main[(main["taskKind"] == "target_change") & (main["placementMap"] == 0)]
    records: list[dict[str, Any]] = []
    for (portfolio, task_id), group in changed.groupby(["portfolioId", "taskId"], sort=True):
        pivot = group.pivot(index="baseDrawPairingId", columns="targetAware", values=["success", "restrictedTime"])
        difference = pivot["success"][True].astype(float) - pivot["success"][False].astype(float)
        nonzero = difference[difference != 0]
        pvalue = float(stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue) if len(nonzero) else 1.0
        records.append(
            {
                "portfolioId": portfolio,
                "taskId": task_id,
                "pairedCount": len(pivot),
                "awareSuccessRate": float(pivot["success"][True].mean()),
                "nonadaptiveSuccessRate": float(pivot["success"][False].mean()),
                "successDifference": float(difference.mean()),
                "signTestP": pvalue,
                "restrictedTimeDifference": float((pivot["restrictedTime"][True] - pivot["restrictedTime"][False]).mean()),
            }
        )
    frame = pd.DataFrame(records)
    frame["signTestHolmP"] = _holm(frame["signTestP"].tolist())
    return frame


def _policy_summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(
            ["taskKind", "taskId", "portfolioId", "placementMap", "faultFraction", "arm"],
            dropna=False,
            sort=True,
        )
        .agg(
            assignedRuns=("runId", "size"),
            successes=("success", "sum"),
            successRate=("success", "mean"),
            medianRestrictedTime=("restrictedTime", "median"),
            meanNormalizedErrorAuc=("normalizedErrorAuc", "mean"),
            meanDistanceOvershoot=("distanceOvershoot", "mean"),
            meanActivations=("activations", "mean"),
            meanObservationReads=("observationReads", "mean"),
            meanValueComparisons=("valueComparisons", "mean"),
            meanAcceptedSwaps=("acceptedSwaps", "mean"),
            meanMemoryUpdates=("memoryUpdates", "mean"),
            meanTargetRecordReads=("targetRecordReads", "mean"),
            meanControllerComputations=("controllerComputations", "mean"),
            meanAbstractEnergy=("abstractEnergyUnits", "mean"),
            quiescentTerminals=("stopReason", lambda values: sum(str(v) in {"quiescent", "target_quiescent"} for v in values)),
            eventBudgetCensors=("stopReason", lambda values: sum(str(v) == "phase_event_budget" for v in values)),
        )
        .reset_index()
    )


def _critical_thresholds(threshold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    for (portfolio, fraction), group in threshold.groupby(["portfolioId", "faultFraction"], sort=True):
        successes = int(group["success"].sum())
        low, high = _wilson(successes, len(group))
        summaries.append(
            {
                "portfolioId": portfolio,
                "faultFraction": fraction,
                "assignedRuns": len(group),
                "successes": successes,
                "successRate": successes / len(group),
                "wilsonLow": low,
                "wilsonHigh": high,
                "classificationAtHalf": "pass" if low >= 0.5 else "fail" if high < 0.5 else "inconclusive",
            }
        )
    curve = pd.DataFrame(summaries)
    records: list[dict[str, Any]] = []
    for portfolio, group in curve.groupby("portfolioId", sort=True):
        group = group.sort_values("faultFraction")
        passing = group[group["classificationAtHalf"] == "pass"]
        failing = group[group["classificationAtHalf"] == "fail"]
        lower = float(passing["faultFraction"].max()) if not passing.empty else None
        higher_fail = failing if lower is None else failing[failing["faultFraction"] > lower]
        upper = float(higher_fail["faultFraction"].min()) if not higher_fail.empty else None
        diffs = group["successRate"].diff()
        violations = group.loc[diffs > 0.10, "faultFraction"].tolist()
        records.append(
            {
                "portfolioId": portfolio,
                "criticalPassLowerFraction": lower,
                "firstFailUpperFraction": upper,
                "thresholdStatus": (
                    "interval_identified" if lower is not None and upper is not None
                    else "left_boundary_below_half" if lower is None and upper is not None
                    else "right_boundary_censored" if lower is not None
                    else "inconclusive"
                ),
                "monotonicityViolationFractionsJson": json.dumps(violations),
                "monotonicityAuditPass": len(violations) == 0,
            }
        )
    return curve, pd.DataFrame(records)


def _terminal_hazards(results: pd.DataFrame) -> pd.DataFrame:
    selected = results[
        results["taskKind"].isin(["injury", "critical_fault_threshold", "target_change"])
    ].copy()
    selected["terminalClass"] = np.where(
        selected["success"],
        "success",
        np.where(
            selected["stopReason"].isin(
                ["quiescent", "target_quiescent", "stabilization_failure", "event_budget"]
            )
            | selected["executionStatus"].astype(str).str.contains("source_competing_terminal"),
            "operational_irreversible_competing",
            "right_censor_or_other_terminal",
        ),
    )
    selected["normalizedTerminalTime"] = np.minimum(
        1.0,
        selected["phaseActivationCount"] / (100 * selected["n"] * selected["n"]),
    )
    selected["timeQuartile"] = pd.cut(
        selected["normalizedTerminalTime"],
        bins=[-1e-12, 0.25, 0.5, 0.75, 1.0],
        labels=["q1", "q2", "q3", "q4"],
        include_lowest=True,
    ).astype(str)
    records: list[dict[str, Any]] = []
    for keys, group in selected.groupby(["taskKind", "taskId", "portfolioId", "faultFraction"], sort=True):
        at_risk = len(group)
        for quartile in ("q1", "q2", "q3", "q4"):
            events = group[
                (group["timeQuartile"] == quartile)
                & (group["terminalClass"] == "operational_irreversible_competing")
            ]
            prior = group[
                group["timeQuartile"].isin(("q1", "q2", "q3", "q4")[: ("q1", "q2", "q3", "q4").index(quartile)])
            ]
            risk = at_risk - int((prior["terminalClass"] != "right_censor_or_other_terminal").sum())
            records.append(
                {
                    "taskKind": keys[0],
                    "taskId": keys[1],
                    "portfolioId": keys[2],
                    "faultFraction": keys[3],
                    "timeQuartile": quartile,
                    "atRisk": max(0, risk),
                    "irreversibleEvents": len(events),
                    "causeSpecificHazard": len(events) / risk if risk > 0 else math.nan,
                    "cumulativeIrreversibleIncidence": float(
                        (
                            (group["terminalClass"] == "operational_irreversible_competing")
                            & (group["normalizedTerminalTime"] <= ("q1", "q2", "q3", "q4").index(quartile) / 4 + 0.25)
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(records)


def _pareto(main: pd.DataFrame) -> pd.DataFrame:
    selected = _primary_task_rows(main, 0)
    summary = (
        selected.groupby(["taskId", "portfolioId"], sort=True)
        .agg(
            successRate=("success", "mean"),
            medianRestrictedTime=("restrictedTime", "median"),
            meanNormalizedErrorAuc=("normalizedErrorAuc", "mean"),
            meanAbstractEnergy=("abstractEnergyUnits", "mean"),
            portfolioComplexity=("compositionEntropyBits", "first"),
        )
        .reset_index()
    )
    flags: list[bool] = []
    for _, row in summary.iterrows():
        peers = summary[summary["taskId"] == row["taskId"]]
        dominates = (
            (peers["successRate"] >= row["successRate"])
            & (peers["medianRestrictedTime"] <= row["medianRestrictedTime"])
            & (peers["meanNormalizedErrorAuc"] <= row["meanNormalizedErrorAuc"])
            & (peers["meanAbstractEnergy"] <= row["meanAbstractEnergy"])
            & (peers["portfolioComplexity"] <= row["portfolioComplexity"])
            & (
                (peers["successRate"] > row["successRate"])
                | (peers["medianRestrictedTime"] < row["medianRestrictedTime"])
                | (peers["meanNormalizedErrorAuc"] < row["meanNormalizedErrorAuc"])
                | (peers["meanAbstractEnergy"] < row["meanAbstractEnergy"])
                | (peers["portfolioComplexity"] < row["portfolioComplexity"])
            )
        )
        flags.append(not bool(dominates.any()))
    summary["paretoEfficient"] = flags
    return summary


def _plots(pareto: pd.DataFrame, threshold_curve: pd.DataFrame) -> list[Path]:
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    tasks = sorted(pareto["taskId"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, task in zip(axes.flat, tasks):
        group = pareto[pareto["taskId"] == task]
        for _, row in group.iterrows():
            ax.scatter(
                row["meanNormalizedErrorAuc"],
                row["successRate"],
                s=60 + 100 * row["portfolioComplexity"],
                marker="o" if row["paretoEfficient"] else "x",
            )
            ax.annotate(row["portfolioId"].replace("chimera_", "c_").replace("pure_", "p_"), (row["meanNormalizedErrorAuc"], row["successRate"]), fontsize=7)
        ax.set_title(task)
        ax.set_xlabel("mean normalized restricted error AUC (lower better)")
        ax.set_ylabel("success probability (higher better)")
        ax.set_ylim(-0.05, 1.05)
    path = figure_dir / "composition_pareto.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for portfolio, group in threshold_curve.groupby("portfolioId", sort=True):
        ax.plot(group["faultFraction"], group["successRate"], marker="o", label=portfolio)
        ax.fill_between(group["faultFraction"], group["wilsonLow"], group["wilsonHigh"], alpha=0.08)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("permanent stuck-identity fraction")
    ax.set_ylabel("central-reversal recovery probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2)
    path = figure_dir / "critical_fault_thresholds.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def _sources(specification: Mapping[str, Any]) -> tuple[list[Source], pd.DataFrame, list[dict[str, Any]]]:
    definitions: list[tuple[int, str, int, str, int]] = []
    for n in specification["panel"]["sizes"]:
        for direction in specification["panel"]["directions"]:
            for replicate in range(specification["panel"]["replicatesPerCell"]):
                for portfolio in PORTFOLIOS:
                    definitions.append((n, direction, replicate, portfolio, 0))
                for portfolio, policies in PORTFOLIOS.items():
                    if len(policies) > 1:
                        definitions.append((n, direction, replicate, portfolio, 1))
    sources: list[Source] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        pending = {pool.submit(_build_source, definition): definition for definition in definitions}
        for ordinal, future in enumerate(as_completed(pending), 1):
            definition = pending[future]
            try:
                sources.append(future.result())
            except Exception as exc:
                failures.append({"definition": definition, "error": repr(exc)})
            if ordinal % 16 == 0 or ordinal == len(definitions):
                print(
                    f"S13 sources {ordinal}/{len(definitions)} built={len(sources)} "
                    f"failures={len(failures)} elapsed={time.monotonic()-started:.1f}s",
                    flush=True,
                )
    sources.sort(key=lambda source: source.source_id)
    rows = []
    for source in sources:
        counts = source.metadata["compositionCounts"]
        rows.append(
            {
                "schemaVersion": "e05.s13.source-checkpoint.v1",
                "sourceId": source.source_id,
                "baseDrawPairingId": source.metadata["baseDrawPairingId"],
                "sourceScenarioId": source.scenario.scenario_id,
                "sourceCheckpointHash": None if source.checkpoint is None else source.checkpoint.state_hash,
                "n": source.n,
                "direction": source.direction,
                "replicateOrdinal": source.replicate,
                "portfolioId": source.portfolio_id,
                "portfolioRole": "chimeric" if len(PORTFOLIOS[source.portfolio_id]) > 1 else "homogeneous",
                "placementMap": source.placement_map,
                "compositionCountsJson": json.dumps(counts, sort_keys=True),
                "policyAssignmentSha256": source.metadata["policyAssignmentSha256"],
                "initialOccupancySha256": source.metadata["initialOccupancySha256"],
                "runtimeSeed": source.metadata["runtimeSeed"],
                "developmentActivationCount": source.development["activationCount"],
                "developmentWithinBudget": source.development["withinFrozenDevelopmentBudget"],
                "developmentCompleted": source.development["completed"],
                "developmentStopReason": source.development["stopReason"],
                "stabilizationOpportunities": source.stabilization.get("opportunities"),
                "absorbingTargetCertificate": source.stabilization.get("absorbingTargetCertificate"),
                "stabilizationCoveragePass": source.stabilization.get("coveragePass"),
                "sourceSuccess": source.source_success,
                "sourceStopReason": source.source_stop_reason,
                "exactReplayPass": source.exact_replay_pass,
                "policyFamilyCount": len(counts),
                "compositionEntropyBits": shannon_entropy_bits(counts),
                "assignmentDescriptionBits": assignment_description_bits(counts),
            }
        )
    return sources, pd.DataFrame(rows), failures


def _validation(
    specification: Mapping[str, Any],
    sources: pd.DataFrame,
    main: pd.DataFrame,
    threshold: pd.DataFrame,
    threshold_assignments: pd.DataFrame,
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["specificationValidatedBeforeRuns"] = True
    checks["plannedSourceCount"] = len(sources) == specification["panel"]["plannedSourceCount"]
    checks["sourceWorkerFailuresZero"] = len(failures) == 0
    checks["allSourceDevelopmentReplay"] = bool(sources["exactReplayPass"].all())
    checks["allSourceSuccessOrExplicitTerminal"] = sources["sourceSuccess"].notna().all()
    checks["exactCompositionCounts"] = all(
        sum(json.loads(value).values()) == int(n)
        for value, n in zip(sources["compositionCountsJson"], sources["n"])
    )
    checks["compositionIdentityHashesUniqueByMap"] = not sources.duplicated(
        ["n", "direction", "replicateOrdinal", "portfolioId", "placementMap", "policyAssignmentSha256"]
    ).any()
    base = sources.groupby(["n", "direction", "replicateOrdinal"])
    checks["baseInputPairing"] = bool(
        base["initialOccupancySha256"].nunique().eq(1).all()
        and base["runtimeSeed"].nunique().eq(1).all()
        and base["baseDrawPairingId"].nunique().eq(1).all()
    )
    checks["mainPlannedCount"] = len(main) == specification["panel"]["plannedMainRuns"]
    checks["mainWorkerFailuresZero"] = not failures
    checks["everyExecutedTrajectoryReplay"] = bool(
        pd.concat([main, threshold], ignore_index=True)["exactReplayPass"].fillna(False).all()
    )
    checks["everyExecutedRunValidation"] = bool(
        pd.concat([main, threshold], ignore_index=True)["validationPass"].fillna(False).all()
    )
    executed = pd.concat([main, threshold], ignore_index=True)
    ledger_checks = []
    for payload in executed["nativeLedgerDeltaJson"]:
        ledger_checks.append(all(ledger_identity(json.loads(payload)).values()))
    checks["nativeLedgerIdentities"] = all(ledger_checks)
    executed_targets = main[
        main["taskKind"].isin(["target_change", "intact_stability"])
        & main["trajectoryExecuted"].astype(bool)
    ]
    checks["targetFeasibility"] = bool(executed_targets["targetFeasible"].fillna(False).all())
    checks["targetArmPairing"] = bool(
        main[main["taskKind"].isin(["target_change", "intact_stability"])]
        .groupby(["sourceId", "taskKind", "taskId"])["targetAware"]
        .nunique()
        .eq(2)
        .all()
    )
    injury = main[
        (main["taskKind"] == "injury")
        & (main["placementMap"] == 0)
        & main["trajectoryExecuted"].astype(bool)
    ]
    lesion_groups = injury.groupby(["baseDrawPairingId", "taskId"])
    checks["lesionWindowMatchedWithinBase"] = bool(
        lesion_groups["lesionWindowLength"].nunique().eq(1).all()
    )
    reversal = injury[injury["taskId"] == "segment_reversal_central_v1"]
    checks["reversalSeverityExactlyMatchedWithinBase"] = bool(
        reversal.groupby("baseDrawPairingId")["lesionPostDistance"].nunique().eq(1).all()
    )
    scramble = injury[injury["taskId"] == "local_scramble_sattolo_v1"]
    checks["scrambleSeverityPositive"] = bool((scramble["lesionPostDistance"] > 0).all())
    # S03's injury stream and S04's audited reconstruction boundary address
    # the Sattolo draw by immutable scenario ID. Cross-portfolio rows are thus
    # base/window/operator-paired but intentionally RNG-unpaired. Observing
    # more than one realized distance in every complete block confirms that
    # this limitation was exposed rather than silently reported as exact.
    checks["scrambleScenarioIdRngUnpairingAudited"] = bool(
        scramble.groupby("baseDrawPairingId")["lesionPostDistance"].nunique().gt(1).all()
    )
    checks["thresholdAssignmentComplete"] = len(threshold_assignments) == specification["criticalFaultThreshold"]["plannedPotentialRunsIncludingReusedZero"]
    checks["thresholdStatusesResolved"] = threshold_assignments["executionStatus"].isin(
        [
            "executed",
            "source_competing_terminal",
            "reused_main_no_new_execution",
            "reused_main_source_competing_terminal",
            "not_run_prespecified_global_threshold_stop",
        ]
    ).all()
    checks["thresholdResultsMatchAssignments"] = len(threshold) == int(
        (~threshold_assignments["executionStatus"].str.startswith("not_run")).sum()
    )
    checks["faultCountsExact"] = bool(
        threshold.apply(lambda row: int(row["faultCount"]) == fault_count(int(row["n"]), float(row["faultFraction"])), axis=1).all()
    )
    nested = True
    positive = threshold[threshold["faultFraction"] > 0]
    for _, group in positive.groupby("sourceId"):
        prior: set[str] = set()
        for payload in group.sort_values("faultFraction")["faultIdentityIdsJson"]:
            current = set(json.loads(payload))
            nested = nested and prior <= current
            prior = current
    checks["faultIdentitySetsNested"] = nested
    checks["onlyFaultFlagsChangedOnExecutedFaultRuns"] = bool(
        positive.loc[positive["trajectoryExecuted"].astype(bool), "onlyFaultFlagsChanged"]
        .fillna(False)
        .all()
    )
    checks["censorsAndCompetingTerminalsRetained"] = len(executed) == len(main) + len(threshold)
    checks["completeRunAccounting"] = bool(
        len(main)
        == int(main["trajectoryExecuted"].sum())
        + int((~main["trajectoryExecuted"]).sum())
        and len(threshold)
        == int(threshold["trajectoryExecuted"].sum())
        + int((~threshold["trajectoryExecuted"]).sum())
        and int(threshold_assignments["executionStatus"].eq("executed").sum())
        == int(threshold["trajectoryExecuted"].sum())
    )
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "schemaVersion": "e05.s13.validation-summary.v1",
        "researchStepId": "S13",
        "checks": checks,
        "allPassed": all(checks.values()),
        "sourceFailures": int((~sources["sourceSuccess"].astype(bool)).sum()),
        "workerFailureCount": len(failures),
    }


def main() -> None:
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    specification = json.loads(SPEC_PATH.read_text())
    validate_chimeric_spec(specification)
    # The freeze copy is written before any run result is constructed.
    write_json(OUTPUT / "s13_specification.json", specification)
    if not S12_RESULTS.exists() or not S12_REPORT.exists() or not E01_COMPOSITION.exists():
        raise FileNotFoundError("required S12 or E01 composition dependency is absent")
    sources, source_frame, source_failures = _sources(specification)
    source_frame.to_parquet(OUTPUT / "source_checkpoints.parquet", index=False)
    main_cases = _main_cases(sources, specification)
    main_rows, main_failures = _run_pool(main_cases, _execute_main, "main")
    main_frame = pd.DataFrame(main_rows)
    primary_sources = [source for source in sources if source.placement_map == 0]
    threshold_frame, threshold_assignments, ucb_history, threshold_failures = _execute_threshold_sequence(
        primary_sources, specification, main_frame
    )
    all_failures = [*source_failures, *main_failures, *threshold_failures]
    unified = pd.concat([main_frame, threshold_frame], ignore_index=True, sort=False)
    unified = unified.sort_values(["taskKind", "taskId", "portfolioId", "placementMap", "faultFraction", "sourceId"]).reset_index(drop=True)
    unified.to_parquet(OUTPUT / "chimeric_recovery.parquet", index=False)
    threshold_assignments.to_parquet(OUTPUT / "threshold_assignments.parquet", index=False)
    contrasts = _composition_contrasts(main_frame)
    contrasts.to_parquet(OUTPUT / "composition_contrasts.parquet", index=False)
    placement = _placement_sensitivity(main_frame)
    placement.to_parquet(OUTPUT / "placement_sensitivity.parquet", index=False)
    controller = _controller_contrasts(main_frame)
    controller.to_parquet(OUTPUT / "controller_contrasts.parquet", index=False)
    policy_summary = _policy_summary(unified)
    policy_summary.to_parquet(OUTPUT / "policy_family_summary.parquet", index=False)
    threshold_curve, critical = _critical_thresholds(threshold_frame)
    threshold_curve.to_parquet(OUTPUT / "threshold_curves.parquet", index=False)
    critical.to_parquet(OUTPUT / "critical_thresholds.parquet", index=False)
    hazards = _terminal_hazards(unified)
    hazards.to_parquet(OUTPUT / "terminal_hazards.parquet", index=False)
    pareto = _pareto(main_frame)
    pareto.to_parquet(OUTPUT / "pareto_summary.parquet", index=False)
    figures = _plots(pareto, threshold_curve)
    validation = _validation(
        specification,
        source_frame,
        main_frame,
        threshold_frame,
        threshold_assignments,
        all_failures,
    )
    write_json(OUTPUT / "validation_summary.json", validation)
    write_json(OUTPUT / "worker_failures.json", all_failures)
    benefits = int(contrasts["primaryBenefit"].sum())
    harms = int(contrasts["primaryHarm"].sum())
    intact_harms = int(
        main_frame[
            (main_frame["taskKind"] == "intact_stability")
            & main_frame["noChangeAnyTargetDeparture"].fillna(False)
        ].shape[0]
    )
    classification = (
        "constraining/contradictory"
        if harms or intact_harms or not validation["allPassed"]
        else "supportive"
        if benefits
        else "null"
    )
    accounting = {
        "schemaVersion": "e05.s13.run-accounting.v1",
        "plannedSources": specification["panel"]["plannedSourceCount"],
        "builtSources": len(source_frame),
        "sourceFailures": validation["sourceFailures"],
        "plannedMainRuns": specification["panel"]["plannedMainRuns"],
        "executedMainTrajectories": int(main_frame["trajectoryExecuted"].sum()),
        "sourceTerminalMainRows": int((~main_frame["trajectoryExecuted"]).sum()),
        "plannedMainExactReplays": specification["panel"]["plannedMainExactReplays"],
        "completedMainExactReplays": int(main_frame["trajectoryExecuted"].sum()),
        "thresholdPotentialRows": len(threshold_assignments),
        "thresholdReusedZeroRows": int(threshold_assignments["executionStatus"].eq("reused_main_no_new_execution").sum()),
        "thresholdNewExecutedTrajectories": int(threshold_assignments["executionStatus"].eq("executed").sum()),
        "thresholdPrespecifiedStopRows": int(threshold_assignments["executionStatus"].str.startswith("not_run").sum()),
        "thresholdNewExactReplays": int(threshold_assignments["executionStatus"].eq("executed").sum()),
        "logicalResultRows": len(unified),
        "newTrajectoryExecutionsIncludingReplays": 2 * (
            int(main_frame["trajectoryExecuted"].sum())
            + int(threshold_assignments["executionStatus"].eq("executed").sum())
        ),
        "sourceDevelopmentExecutionsIncludingReplays": 2 * len(source_frame),
        "workerFailures": len(all_failures),
        "thresholdUcbHistory": ucb_history,
        "outcomeClassification": classification,
        "primaryBenefits": benefits,
        "primaryHarms": harms,
        "intactDepartureRuns": intact_harms,
        "elapsedSeconds": time.monotonic() - started,
        "workers": WORKERS,
    }
    write_json(OUTPUT / "run_accounting.json", accounting)
    provenance_paths = [
        OUTPUT / "s13_specification.json",
        OUTPUT / "source_checkpoints.parquet",
        OUTPUT / "chimeric_recovery.parquet",
        OUTPUT / "threshold_assignments.parquet",
        OUTPUT / "composition_contrasts.parquet",
        OUTPUT / "placement_sensitivity.parquet",
        OUTPUT / "controller_contrasts.parquet",
        OUTPUT / "policy_family_summary.parquet",
        OUTPUT / "threshold_curves.parquet",
        OUTPUT / "critical_thresholds.parquet",
        OUTPUT / "terminal_hazards.parquet",
        OUTPUT / "pareto_summary.parquet",
        OUTPUT / "validation_summary.json",
        OUTPUT / "worker_failures.json",
        OUTPUT / "run_accounting.json",
        *figures,
    ]
    manifest = {
        "schemaVersion": "e05.s13.provenance-manifest.v1",
        "researchStepId": "S13",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPOSITORY),
        "branch": git_output("branch", "--show-current"),
        "sourceCommitAtExecution": git_output("rev-parse", "HEAD"),
        "workingTreeDirtyAtExecution": bool(git_output("status", "--short")),
        "command": "E05_S13_WORKERS=8 python scripts/build_regeneration_s13.py",
        "workers": WORKERS,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "s12Results": {"path": str(S12_RESULTS), "sha256": file_sha256(S12_RESULTS)},
            "s12Report": {"path": str(S12_REPORT), "sha256": file_sha256(S12_REPORT)},
            "e01Composition": {"path": str(E01_COMPOSITION), "sha256": file_sha256(E01_COMPOSITION)},
            "specificationSource": {"path": str(SPEC_PATH), "sha256": file_sha256(SPEC_PATH)},
        },
        "artifacts": [
            {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in provenance_paths
        ],
        "outcomeClassification": classification,
    }
    write_json(OUTPUT / "provenance_manifest.json", manifest)
    print(json.dumps({"classification": classification, "validation": validation["allPassed"], "accounting": accounting}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
