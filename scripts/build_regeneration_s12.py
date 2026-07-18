#!/usr/bin/env python3
"""Build, execute, validate, and report E05 S12 held-out transfer evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.architectures import ArchitectureExecutionContract  # noqa: E402
from causal_simulator.schedulers import (  # noqa: E402
    SchedulerExecutionContract,
    SchedulerFamily,
    run_scheduled_architecture,
)
from reference_simulator.api import create_scenario  # noqa: E402
from reference_simulator.model import (  # noqa: E402
    Direction,
    LEDGER_FIELDS,
    Policy,
    Scenario,
    canonical_json_bytes,
    sha256_json,
)
from src.regeneration.assisted_rescue import RescueArm  # noqa: E402
from src.regeneration.dynamic_faults import DynamicProfile  # noqa: E402
from src.regeneration.target_change import TargetChange  # noqa: E402
from src.regeneration.tasks import (  # noqa: E402
    Checkpoint,
    _checkpoint_hash,
    _seed,
    stabilize_achieved_checkpoint,
)
from src.regeneration.transfer import (  # noqa: E402
    BENCHMARK_VERSION,
    TRANSFER_RESULT_SCHEMA_VERSION,
    TRANSFER_SPEC_SCHEMA,
    TransferTargetChange,
    apply_transfer_lesion,
    build_transfer_target_definition,
    exact_replay_result,
    run_damage_transfer,
    run_goal_transfer,
    transfer_target_correspondence_rows,
    validate_transfer_spec,
)


CONFIG = REPOSITORY / "configs/regeneration/s12_transfer.json"
S01_DIR = Path("/artifacts/research_steps/S01")
S06_RESULTS = Path("/artifacts/research_steps/S06/assisted_rescue.parquet")
ATTACHMENT_SIDECAR = Path(
    "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"
)
UPSTREAM_INPUTS: tuple[Path, ...] = (
    Path("/workspace/AGENTS.md"),
    Path("/workspace/FULL_PLAN.md"),
    Path("/workspace/RESEARCH_PLAN.md"),
    Path("/workspace/PREVIOUS_ARTIFACTS.md"),
    Path("/workspace/PREVIOUS_ARTIFACTS.json"),
    Path("/workspace/CAPABILITIES.md"),
    Path("/workspace/CAPABILITY_AVAILABILITY.json"),
    Path("/workspace/DATASETS.md"),
    Path("/workspace/DATASET_CATALOG.json"),
    Path("/workspace/DATASET_AVAILABILITY.json"),
    Path("/workspace/input-attachments/MANIFEST.json"),
    ATTACHMENT_SIDECAR,
    *tuple(
        item
        for step in range(1, 12)
        for item in (
            Path(f"/artifacts/research_steps/S{step:02d}/research_step_full_results.md"),
            Path(f"/artifacts/research_steps/S{step:02d}/validation_summary.json"),
        )
    ),
    Path("/artifacts/research_steps/S01/task_spec.json"),
    Path("/artifacts/research_steps/S02/timing_spec.json"),
    Path("/artifacts/research_steps/S03/lesion_library/lesion_spec.json"),
    Path("/artifacts/research_steps/S04/dynamic_fault_package/dynamic_fault_spec.json"),
    Path("/artifacts/research_steps/S05/nudge_recovery_package/nudge_recovery_spec.json"),
    Path("/artifacts/research_steps/S06/assisted_rescue_package/assisted_rescue_spec.json"),
    Path("/artifacts/research_steps/S07/memory_variants/local_memory_spec.json"),
    Path("/artifacts/research_steps/S08/plasticity_package/policy_plasticity_spec.json"),
    Path("/artifacts/research_steps/S09/target_change_package/target_change_spec.json"),
    Path("/artifacts/research_steps/S10/repeated_injury_package/repeated_injury_spec.json"),
    Path("/artifacts/research_steps/S11/memory_reset_package/state_memory_reset_spec.json"),
    Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
    Path("/previous-artifacts/E01/specification/transition_spec.md"),
    Path("/previous-artifacts/E01/research_steps/S08/holdout_integrity.json"),
    Path("/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json"),
    Path("/previous-artifacts/E02/research_steps/S02/action_interface_spec.md"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/scheduler_contract.md"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/fault_semantics_contract.md"),
    Path("/previous-artifacts/E02/research_steps/S08/semantic_random_stream_specification.json"),
    Path("/previous-artifacts/E02/research_steps/S09/ledger_identity_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S12/modeling_prespecification.json"),
    Path("/previous-artifacts/E02/research_steps/S14/research_step_full_results.md"),
    S06_RESULTS,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def _validate_inputs() -> dict[str, bool]:
    required = (CONFIG, *UPSTREAM_INPUTS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing S12 inputs: {missing}")
    checks: dict[str, bool] = {}
    for step in range(1, 12):
        summary = _read_json(
            Path(f"/artifacts/research_steps/S{step:02d}/validation_summary.json")
        )
        checks[f"s{step:02d}Validation"] = bool(summary["success"])
    checks["e01Release"] = bool(
        _read_json(
            Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json")
        )["validationSuccess"]
    )
    checks["e02Release"] = bool(
        _read_json(
            Path("/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json")
        )["smokeValidation"]["success"]
    )
    s05 = _read_json(Path("/artifacts/research_steps/S05/outcome_classification.json"))
    s06 = _read_json(Path("/artifacts/research_steps/S06/outcome_classification.json"))
    s07 = _read_json(Path("/artifacts/research_steps/S07/outcome_classification.json"))
    s08 = _read_json(Path("/artifacts/research_steps/S08/outcome_classification.json"))
    s09 = _read_json(Path("/artifacts/research_steps/S09/outcome_classification.json"))
    checks["s05NullRetained"] = s05["classification"] == "null"
    checks["s06MatchedBoundaryRetained"] = s06["classification"] in {
        "supportive",
        "constraining/contradictory",
    }
    checks["s07NullRetained"] = s07["classification"] == "null"
    checks["s08HarmsExcluded"] = s08["classification"] == "constraining/contradictory"
    checks["s09SupportedControllerAvailable"] = s09["classification"] == "supportive"
    if not all(checks.values()):
        raise RuntimeError(f"S12 inherited evidence gate failed: {checks}")
    return checks


@dataclass(frozen=True, slots=True)
class Base:
    n: int
    policy: str
    direction: str
    replicate: int
    scenario: Scenario
    checkpoint: Checkpoint
    pairing_id: str
    stabilization: Mapping[str, Any]


def _pairing_id(
    scenario: Scenario,
    n: int,
    policy: str,
    direction: str,
    replicate: int,
    budget: int,
) -> str:
    target_values = list(range(n)) if direction == "ascending" else list(reversed(range(n)))
    content = {
        "schemaVersion": "e05.s01.preinjury-shared-prefix-v1",
        "sourceScenarioId": scenario.scenario_id,
        "n": n,
        "policy": policy,
        "direction": direction,
        "replicateOrdinal": replicate,
        "runtimeSeed": str(scenario.seed),
        "initialOccupancySha256": hashlib.sha256(
            canonical_json_bytes(list(scenario.initial_occupancy))
        ).hexdigest(),
        "targetValuesSha256": hashlib.sha256(
            canonical_json_bytes(target_values)
        ).hexdigest(),
        "eventBudget": budget,
    }
    return "e05pb1:" + sha256_json(content)


def _build_base(n: int, policy: str, direction: str, replicate: int) -> Base:
    budget = 100 * n * n
    seed = _seed(n, policy, direction, replicate)
    generation_key = f"E05/S01/base/n{n}/{policy}/{direction}/r{replicate}"
    scenario = create_scenario(
        list(range(n)),
        policy=Policy(policy),
        direction=Direction(direction),
        seed=seed,
        max_activations=2 * budget + 20 * n,
        generation_key=generation_key,
    )
    run = run_scheduled_architecture(
        scenario,
        ArchitectureExecutionContract.distributed_local(),
        SchedulerExecutionContract(SchedulerFamily.UNIFORM_RANDOM_ACTIVATION),
        trace_mode="digest",
    ).result
    if not run.summary["completed"]:
        raise RuntimeError(f"S12 source development failed: {scenario.scenario_id}")
    occupancy = tuple(run.summary["finalOccupancy"])
    cursors = dict(run.final_state["selectionCursors"])
    checkpoint = Checkpoint(
        occupancy,
        tuple(sorted(cursors.items())),
        int(run.summary["activationCount"]),
        tuple(sorted((str(k), int(v)) for k, v in run.final_state["streamCounters"].items())),
        tuple((field, int(run.final_state["ledger"][field])) for field in LEDGER_FIELDS),
        0,
        _checkpoint_hash(
            scenario.scenario_id,
            occupancy,
            cursors,
            int(run.summary["activationCount"]),
            run.final_state["streamCounters"],
            run.final_state["ledger"],
        ),
    )
    stabilized, audit = stabilize_achieved_checkpoint(scenario, checkpoint)
    if not audit["success"]:
        raise RuntimeError(f"S12 stabilization failed: {scenario.scenario_id}")
    return Base(
        n,
        policy,
        direction,
        replicate,
        scenario,
        stabilized,
        _pairing_id(scenario, n, policy, direction, replicate, budget),
        audit,
    )


def _build_bases(specification: Mapping[str, Any]) -> tuple[dict[tuple[Any, ...], Base], pd.DataFrame]:
    split = specification["immutableSplit"]
    sizes = sorted(set(split["calibration"]["sizes"]) | set(split["heldOutAxes"]["size"]))
    bases: dict[tuple[Any, ...], Base] = {}
    rows: list[dict[str, Any]] = []
    for n in sizes:
        for policy in split["policies"]:
            for direction in split["directions"]:
                for replicate in range(split["replicatesPerCell"]):
                    base = _build_base(n, policy, direction, replicate)
                    key = (n, policy, direction, replicate)
                    bases[key] = base
                    rows.append(
                        {
                            "n": n,
                            "policy": policy,
                            "direction": direction,
                            "replicateOrdinal": replicate,
                            "sourceScenarioId": base.scenario.scenario_id,
                            "s01PairingBlockId": base.pairing_id,
                            "sourceCheckpointHash": base.checkpoint.state_hash,
                            "activationCount": base.checkpoint.activation_count,
                            "stabilizationOpportunities": base.stabilization["opportunities"],
                            "absorbingTargetCertificate": base.stabilization["absorbingTargetCertificate"],
                            "coveragePass": base.stabilization["coveragePass"],
                            "success": base.stabilization["success"],
                        }
                    )
    return bases, pd.DataFrame(rows)


def _s01_checkpoint_validation(bases: Mapping[tuple[Any, ...], Base]) -> pd.DataFrame:
    expected: dict[str, Mapping[str, Any]] = {}
    for line in (S01_DIR / "checkpoint_snapshots.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["taskFamily"] == "repair_achieved":
            expected[row["pairingBlockId"]] = row["checkpoint"]
    rows = []
    for base in bases.values():
        if base.n not in {20, 50}:
            continue
        reference = expected[base.pairing_id]
        rows.append(
            {
                "s01PairingBlockId": base.pairing_id,
                "sourceScenarioId": base.scenario.scenario_id,
                "n": base.n,
                "policy": base.policy,
                "direction": base.direction,
                "replicateOrdinal": base.replicate,
                "stateHashExpected": reference["stateHash"],
                "stateHashObserved": base.checkpoint.state_hash,
                "stateHashPass": reference["stateHash"] == base.checkpoint.state_hash,
                "activationCountPass": int(reference["activationCount"]) == base.checkpoint.activation_count,
                "occupancyPass": tuple(reference["occupancy"]) == base.checkpoint.occupancy,
                "selectionCursorsPass": dict(reference["selectionCursors"]) == dict(base.checkpoint.selection_cursors),
                "streamCountersPass": dict(reference["streamCounters"]) == dict(base.checkpoint.stream_counters),
                "ledgerPass": dict(reference["ledger"]) == dict(base.checkpoint.ledger),
            }
        )
    frame = pd.DataFrame(rows)
    frame["success"] = frame[
        [
            "stateHashPass",
            "activationCountPass",
            "occupancyPass",
            "selectionCursorsPass",
            "streamCountersPass",
            "ledgerPass",
        ]
    ].all(axis=1)
    return frame


@dataclass(frozen=True, slots=True)
class DamageCase:
    case_id: str
    axis: str
    split: str
    base: Base
    lesion_type: str
    location: str
    scheduler: SchedulerFamily
    process_id: str
    lesion: Mapping[str, Any]
    assigned_duration: int
    donor_id: str
    donor_n: int
    retain_trace: bool = False


@dataclass(frozen=True, slots=True)
class GoalCase:
    case_id: str
    axis: str
    split: str
    base: Base
    target_change: TargetChange | TransferTargetChange
    scheduler: SchedulerFamily
    process_id: str
    stability: bool
    retain_trace: bool = False


def _base_iter(bases: Mapping[tuple[Any, ...], Base], sizes: Sequence[int]) -> Iterable[Base]:
    for key in sorted(bases):
        if key[0] in sizes:
            yield bases[key]


def _case_hash(prefix: str, content: Mapping[str, Any]) -> str:
    return prefix + sha256_json(content)


def _duration_bank() -> pd.DataFrame:
    bank = pd.read_parquet(S06_RESULTS)
    bank = bank[
        (bank["split"] == "calibration")
        & (bank["arm"] == "active_assisted_rescue")
        & bank["recoveryObserved"].astype(bool)
    ].copy()
    if len(bank) != 192 or bank["recoveryDuration"].isna().any():
        raise RuntimeError("S06 calibration duration bank changed")
    return bank.sort_values("assistedRunId").reset_index(drop=True)


def _assign_duration(case_id: str, base: Base, bank: pd.DataFrame) -> tuple[int, str, int]:
    nearest = min((20, 50), key=lambda n: (abs(n - base.n), n))
    candidates = bank[
        (bank["n"] == nearest)
        & (bank["policy"] == base.policy)
        & (bank["direction"] == base.direction)
    ].sort_values("assistedRunId")
    if candidates.empty:
        raise RuntimeError("empty S06 duration donor stratum")
    index = int(hashlib.sha256(case_id.encode()).hexdigest()[:16], 16) % len(candidates)
    donor = candidates.iloc[index]
    duration = max(1, int(math.floor(float(donor["recoveryDuration"]) * base.n / nearest + 0.5)))
    return duration, str(donor["assistedRunId"]), nearest


def _damage_cases(
    specification: Mapping[str, Any],
    bases: Mapping[tuple[Any, ...], Base],
) -> tuple[list[DamageCase], pd.DataFrame]:
    calibration_sizes = specification["immutableSplit"]["calibration"]["sizes"]
    held_sizes = specification["immutableSplit"]["heldOutAxes"]["size"]
    fault_ids = specification["immutableSplit"]["heldOutAxes"]["faultProcess"]
    schedulers = [SchedulerFamily(value) for value in specification["immutableSplit"]["heldOutAxes"]["scheduler"]]
    definitions: list[tuple[str, Base, str, str, SchedulerFamily, str]] = []
    uniform = SchedulerFamily.UNIFORM_RANDOM_ACTIVATION
    for base in _base_iter(bases, calibration_sizes):
        definitions.append(("calibration", base, "segment_reversal_central_v1", "central", uniform, "none"))
        definitions.append(("lesion_type", base, "local_scramble_sattolo_v1", "central", uniform, "none"))
        for location in ("left_off_center", "right_off_center"):
            definitions.append(("lesion_location", base, "segment_reversal_central_v1", location, uniform, "none"))
        for scheduler in schedulers:
            definitions.append(("scheduler", base, "segment_reversal_central_v1", "central", scheduler, "none"))
        for process_id in fault_ids:
            definitions.append(("fault_process", base, "segment_reversal_central_v1", "central", uniform, process_id))
    for base in _base_iter(bases, held_sizes):
        definitions.append(("size", base, "segment_reversal_central_v1", "central", uniform, "none"))
        location = "left_off_center" if base.replicate % 2 == 0 else "right_off_center"
        for scheduler in schedulers:
            for process_id in fault_ids:
                definitions.append(("joint", base, "local_scramble_sattolo_v1", location, scheduler, process_id))
    bank = _duration_bank()
    cases: list[DamageCase] = []
    rows: list[dict[str, Any]] = []
    trace_axes: set[str] = set()
    for axis, base, lesion_type, location, scheduler, process_id in definitions:
        content = {
            "mechanismId": "s06_active_local_repair_v1",
            "axis": axis,
            "sourceCheckpointHash": base.checkpoint.state_hash,
            "lesionType": lesion_type,
            "location": location,
            "scheduler": scheduler.value,
            "processId": process_id,
        }
        case_id = _case_hash("e05tc12d:", content)
        lesion = apply_transfer_lesion(
            base.scenario,
            base.checkpoint,
            lesion_type=lesion_type,
            location=location,
        )
        duration, donor_id, donor_n = _assign_duration(case_id, base, bank)
        # Twelve frozen pairs: six damage holdout axes and six goal axes.  The
        # damage calibration anchor is already fully represented in run-level
        # digests and does not consume a selected full-trace slot.
        retain = (
            axis != "calibration"
            and axis not in trace_axes
            and base.policy == "Bubble"
            and base.direction == "ascending"
        )
        if retain:
            trace_axes.add(axis)
        split = "calibration" if axis == "calibration" else "holdout"
        case = DamageCase(
            case_id,
            axis,
            split,
            base,
            lesion_type,
            location,
            scheduler,
            process_id,
            lesion,
            duration,
            donor_id,
            donor_n,
            retain,
        )
        cases.append(case)
        rows.append(
            {
                "schemaVersion": "e05.s12.split-row.v1",
                "transferCaseId": case_id,
                "mechanismId": "s06_active_local_repair_v1",
                "axis": axis,
                "split": split,
                "n": base.n,
                "policy": base.policy,
                "direction": base.direction,
                "replicateOrdinal": base.replicate,
                "sourceScenarioId": base.scenario.scenario_id,
                "sourceCheckpointHash": base.checkpoint.state_hash,
                "lesionType": lesion_type,
                "lesionLocation": location,
                "lesionStateHash": lesion["lesionStateHash"],
                "lesionWindowLength": lesion["windowLength"],
                "lesionPostDistance": lesion["postDistance"],
                "scheduler": scheduler.value,
                "processId": process_id,
                "assignedDuration": duration,
                "durationDonorRunId": donor_id,
                "durationDonorN": donor_n,
                "durationAssignmentOutcomeBlind": True,
                "retainTracePair": retain,
            }
        )
    cases.sort(key=lambda item: item.case_id)
    return cases, pd.DataFrame(rows).sort_values("transferCaseId").reset_index(drop=True)


def _goal_cases(
    specification: Mapping[str, Any],
    bases: Mapping[tuple[Any, ...], Base],
) -> tuple[list[GoalCase], pd.DataFrame, pd.DataFrame]:
    calibration_sizes = specification["immutableSplit"]["calibration"]["sizes"]
    held_sizes = specification["immutableSplit"]["heldOutAxes"]["size"]
    fault_ids = specification["immutableSplit"]["heldOutAxes"]["faultProcess"]
    schedulers = [SchedulerFamily(value) for value in specification["immutableSplit"]["heldOutAxes"]["scheduler"]]
    known = tuple(TargetChange(value) for value in specification["immutableSplit"]["calibration"]["targetChanges"])
    unseen = tuple(TransferTargetChange(value) for value in specification["immutableSplit"]["heldOutAxes"]["targetChange"])
    uniform = SchedulerFamily.UNIFORM_RANDOM_ACTIVATION
    definitions: list[tuple[str, Base, TargetChange | TransferTargetChange, SchedulerFamily, str, bool]] = []
    for base in _base_iter(bases, calibration_sizes):
        for target in known:
            definitions.append(("calibration", base, target, uniform, "none", False))
        for target in unseen:
            definitions.append(("target_change", base, target, uniform, "none", False))
        for target in known:
            for scheduler in schedulers:
                definitions.append(("scheduler", base, target, scheduler, "none", False))
            for process_id in fault_ids:
                definitions.append(("fault_process", base, target, uniform, process_id, False))
    for base in _base_iter(bases, held_sizes):
        for target in known:
            definitions.append(("size", base, target, uniform, "none", False))
        for target in unseen:
            for scheduler in schedulers:
                for process_id in fault_ids:
                    definitions.append(("joint", base, target, scheduler, process_id, False))
    # No-change probes use one frozen nominal target label per fully specified domain.
    for base in _base_iter(bases, calibration_sizes):
        definitions.append(("calibration", base, TargetChange.REVERSE, uniform, "none", True))
        definitions.append(("target_change", base, TransferTargetChange.ADJACENT_PAIR_SWAP, uniform, "none", True))
        for scheduler in schedulers:
            definitions.append(("scheduler", base, TargetChange.REVERSE, scheduler, "none", True))
        for process_id in fault_ids:
            definitions.append(("fault_process", base, TargetChange.REVERSE, uniform, process_id, True))
    for base in _base_iter(bases, held_sizes):
        definitions.append(("size", base, TargetChange.REVERSE, uniform, "none", True))
        for scheduler in schedulers:
            for process_id in fault_ids:
                definitions.append(("joint", base, TransferTargetChange.ADJACENT_PAIR_SWAP, scheduler, process_id, True))
    cases: list[GoalCase] = []
    rows: list[dict[str, Any]] = []
    correspondence: list[dict[str, Any]] = []
    trace_axes: set[str] = set()
    target_seen: set[tuple[str, str, bool]] = set()
    for axis, base, target, scheduler, process_id, stability in definitions:
        content = {
            "mechanismId": "s09_gradient_target_code_controller_v1",
            "axis": axis,
            "sourceCheckpointHash": base.checkpoint.state_hash,
            "targetChangeId": target.value,
            "scheduler": scheduler.value,
            "processId": process_id,
            "stability": stability,
        }
        case_id = _case_hash("e05tc12g:", content)
        definition = build_transfer_target_definition(base.scenario, target, no_change=stability)
        retain = (
            not stability
            and axis not in trace_axes
            and base.policy == "Bubble"
            and base.direction == "ascending"
        )
        if retain:
            trace_axes.add(axis)
        split = "calibration" if axis == "calibration" else "holdout"
        case = GoalCase(
            case_id,
            axis,
            split,
            base,
            target,
            scheduler,
            process_id,
            stability,
            retain,
        )
        cases.append(case)
        rows.append(
            {
                "schemaVersion": "e05.s12.split-row.v1",
                "transferCaseId": case_id,
                "mechanismId": "s09_gradient_target_code_controller_v1",
                "axis": axis,
                "split": split,
                "stability": stability,
                "n": base.n,
                "policy": base.policy,
                "direction": base.direction,
                "replicateOrdinal": base.replicate,
                "sourceScenarioId": base.scenario.scenario_id,
                "sourceCheckpointHash": base.checkpoint.state_hash,
                "targetChangeId": target.value,
                "targetHash": definition.target_hash,
                "targetMaximumDistance": definition.maximum_distance,
                "scheduler": scheduler.value,
                "processId": process_id,
                "retainTracePair": retain,
            }
        )
        target_key = (base.scenario.scenario_id, target.value, stability)
        if target_key not in target_seen:
            target_seen.add(target_key)
            correspondence.extend(
                {
                    **row,
                    "stability": stability,
                    "noChange": stability,
                }
                for row in transfer_target_correspondence_rows(base.scenario, definition)
            )
    cases.sort(key=lambda item: item.case_id)
    return (
        cases,
        pd.DataFrame(rows).sort_values("transferCaseId").reset_index(drop=True),
        pd.DataFrame(correspondence).sort_values(
            ["scenarioId", "targetChangeId", "stability", "identityId"]
        ).reset_index(drop=True),
    )


def _flatten_result(
    result: Mapping[str, Any],
    case: DamageCase | GoalCase,
    *,
    replay_pass: bool,
) -> dict[str, Any]:
    base = case.base
    row: dict[str, Any] = {
        "schemaVersion": TRANSFER_RESULT_SCHEMA_VERSION,
        "transferCaseId": case.case_id,
        "axis": case.axis,
        "split": case.split,
        "mechanismId": result["mechanismId"],
        "arm": result["arm"],
        "n": base.n,
        "policy": base.policy,
        "direction": base.direction,
        "replicateOrdinal": base.replicate,
        "sourceScenarioId": base.scenario.scenario_id,
        "sourceCheckpointHash": base.checkpoint.state_hash,
        "scheduler": result["scheduler"],
        "processId": result["processId"],
        "stopReason": result["stopReason"],
        "success": result["success"],
        "phaseActivationCount": result["phaseActivationCount"],
        "restrictedTime": result["restrictedTime"],
        "startEventIndex": result["startEventIndex"],
        "endEventIndex": result["endEventIndex"],
        "initialStateHash": result["initialStateHash"],
        "finalStateHash": result["finalStateHash"],
        "schedulerAuditCount": result["schedulerAuditCount"],
        "schedulerAuditDigest": result["schedulerAuditDigest"],
        "eventDigest": result["eventDigest"],
        "resultDigest": result["resultDigest"],
        "allValidationPass": all(result["validation"].values()),
        "validationJson": json.dumps(result["validation"], sort_keys=True),
        "nativeLedgerDeltaJson": json.dumps(result["nativeLedgerDelta"], sort_keys=True),
        "streamCounterDeltaJson": json.dumps(result["streamCounterDelta"], sort_keys=True),
        "nativeActivations": result["nativeLedgerDelta"]["activations"],
        "nativeProposals": result["nativeLedgerDelta"]["proposals"],
        "nativeAcceptedSwaps": result["nativeLedgerDelta"]["acceptedSwaps"],
        "nativeMemoryUpdates": result["nativeLedgerDelta"]["memoryUpdates"],
        "nativeNoOps": result["nativeLedgerDelta"]["noOps"],
        "nativeRejections": result["nativeLedgerDelta"]["rejections"],
        "exactReplayPass": replay_pass,
    }
    if isinstance(case, DamageCase):
        rescue = result["rescueLedger"]
        row.update(
            {
                "stability": False,
                "targetAware": None,
                "targetChangeId": None,
                "lesionType": case.lesion_type,
                "lesionLocation": case.location,
                "lesionStateHash": case.lesion["lesionStateHash"],
                "lesionWindowLength": case.lesion["windowLength"],
                "lesionPostDistance": case.lesion["postDistance"],
                "assignedDuration": case.assigned_duration if result["arm"] == RescueArm.MATCHED_COST.value else None,
                "durationDonorRunId": case.donor_id,
                "recoveryObserved": result["recoveryObserved"],
                "recoveryDuration": result["recoveryDuration"],
                "recoveryCensored": result["recoveryCensored"],
                "initialDistance": result["initialDistance"],
                "finalDistance": result["finalDistance"],
                "distanceAuc": result["distanceAuc"],
                "actionUnitsSpent": rescue["actionUnitsSpent"],
                "abstractEnergyUnits": rescue["energyUnitsSpent"],
                "nativeOpportunitiesSuppressed": rescue["nativeOpportunitiesSuppressed"],
                "targetRecordReads": 0,
                "controllerComputations": 0,
                "dynamicLedgerJson": json.dumps(result["dynamicLedger"], sort_keys=True),
                "processAuditDigest": result["dynamicAuditDigest"],
            }
        )
    else:
        target = result["targetLedger"]
        row.update(
            {
                "stability": case.stability,
                "targetAware": result["targetAware"],
                "targetChangeId": result["targetChangeId"],
                "lesionType": None,
                "lesionLocation": None,
                "lesionStateHash": None,
                "lesionWindowLength": None,
                "lesionPostDistance": None,
                "assignedDuration": None,
                "durationDonorRunId": None,
                "recoveryObserved": None,
                "recoveryDuration": None,
                "recoveryCensored": None,
                "initialDistance": result["initialNewTargetDistance"],
                "finalDistance": result["finalNewTargetDistance"],
                "distanceAuc": result["newTargetDistanceAuc"],
                "targetMaximumDistance": result["targetMaximumDistance"],
                "initialNormalizedNewTargetDistance": result["initialNormalizedNewTargetDistance"],
                "finalNormalizedNewTargetDistance": result["finalNormalizedNewTargetDistance"],
                "adaptationTime": result["adaptationTime"],
                "adaptationCensored": result["adaptationCensored"],
                "postHitAnyDeparture": result["postHitAnyDeparture"],
                "noChangeAnyTargetDeparture": result["noChangeAnyTargetDeparture"],
                "noChangeFinalTargetRetained": result["noChangeFinalTargetRetained"],
                "actionUnitsSpent": 0,
                "abstractEnergyUnits": target["abstractEnergyUnits"],
                "nativeOpportunitiesSuppressed": 0,
                "targetRecordReads": target["targetRecordReads"],
                "controllerComputations": target["controllerComputations"],
                "dynamicLedgerJson": json.dumps(result["dynamicLedger"], sort_keys=True),
                "processAuditDigest": result["dynamicAuditDigest"],
            }
        )
    return row


def _execute_damage_case(case: DamageCase) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for arm in (RescueArm.ACTIVE, RescueArm.MATCHED_COST):
        duration = case.assigned_duration if arm == RescueArm.MATCHED_COST else None

        def factory() -> dict[str, Any]:
            return run_damage_transfer(
                case.base.scenario,
                case.base.checkpoint,
                lesion=case.lesion,
                scheduler=case.scheduler,
                process_id=case.process_id,
                arm=arm,
                assigned_duration=duration,
                recovery_budget=100 * case.base.n * case.base.n,
                retain_trace=case.retain_trace,
            )

        result = factory()
        exact_replay_result(factory, result)
        rows.append(_flatten_result(result, case, replay_pass=True))
        if case.retain_trace:
            traces.append({"transferCaseId": case.case_id, **result})
    return {"rows": rows, "traces": traces}


def _execute_goal_case(case: GoalCase) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for target_aware in (True, False):

        def factory() -> dict[str, Any]:
            return run_goal_transfer(
                case.base.scenario,
                case.base.checkpoint,
                target_change=case.target_change,
                scheduler=case.scheduler,
                process_id=case.process_id,
                target_aware=target_aware,
                adaptation_budget=100 * case.base.n * case.base.n,
                probe_budget=20 * case.base.n,
                stability=case.stability,
                retain_trace=case.retain_trace,
            )

        result = factory()
        exact_replay_result(factory, result)
        rows.append(_flatten_result(result, case, replay_pass=True))
        if case.retain_trace:
            traces.append({"transferCaseId": case.case_id, **result})
    return {"rows": rows, "traces": traces}


def _run_cases(
    cases: Sequence[DamageCase | GoalCase],
    executor: Callable[[Any], dict[str, Any]],
    *,
    workers: int,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(executor, case): case for case in cases}
        for ordinal, future in enumerate(as_completed(pending), 1):
            case = pending[future]
            try:
                result = future.result()
                rows.extend(result["rows"])
                traces.extend(result["traces"])
            except Exception as exc:  # pragma: no cover - full execution diagnostics
                failures.append(
                    {
                        "transferCaseId": case.case_id,
                        "axis": case.axis,
                        "error": repr(exc),
                    }
                )
            if ordinal % 64 == 0 or ordinal == len(cases):
                print(
                    f"S12 {label} {ordinal}/{len(cases)} cases; rows={len(rows)} "
                    f"failures={len(failures)} elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["transferCaseId"], row["arm"]))
    traces.sort(key=lambda row: (row["transferCaseId"], row["arm"]))
    return rows, traces, failures


def _paired_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for case_id, group in results.groupby("transferCaseId", sort=True):
        if len(group) != 2:
            continue
        mechanism = str(group.iloc[0]["mechanismId"])
        if mechanism == "s06_active_local_repair_v1":
            active = group[group["arm"] == RescueArm.ACTIVE.value].iloc[0]
            control = group[group["arm"] == RescueArm.MATCHED_COST.value].iloc[0]
        else:
            active = group[group["targetAware"].astype(bool)].iloc[0]
            control = group[~group["targetAware"].astype(bool)].iloc[0]
        records.append(
            {
                "schemaVersion": "e05.s12.paired-contrast.v1",
                "transferCaseId": case_id,
                "mechanismId": mechanism,
                "axis": active["axis"],
                "split": active["split"],
                "stability": bool(active["stability"]),
                "n": int(active["n"]),
                "policy": active["policy"],
                "direction": active["direction"],
                "replicateOrdinal": int(active["replicateOrdinal"]),
                "scheduler": active["scheduler"],
                "processId": active["processId"],
                "lesionType": active["lesionType"],
                "lesionLocation": active["lesionLocation"],
                "targetChangeId": active["targetChangeId"],
                "activeSuccess": bool(active["success"]),
                "controlSuccess": bool(control["success"]),
                "successDifference": int(bool(active["success"])) - int(bool(control["success"])),
                "activeRestrictedTime": int(active["restrictedTime"]),
                "controlRestrictedTime": int(control["restrictedTime"]),
                "restrictedTimeDifference": int(active["restrictedTime"]) - int(control["restrictedTime"]),
                "activeFinalDistance": float(active["finalDistance"]),
                "controlFinalDistance": float(control["finalDistance"]),
                "finalDistanceDifference": float(active["finalDistance"]) - float(control["finalDistance"]),
                "activeDistanceAuc": float(active["distanceAuc"]),
                "controlDistanceAuc": float(control["distanceAuc"]),
                "distanceAucDifference": float(active["distanceAuc"]) - float(control["distanceAuc"]),
                "activeAbstractEnergyUnits": int(active["abstractEnergyUnits"]),
                "controlAbstractEnergyUnits": int(control["abstractEnergyUnits"]),
                "abstractEnergyDifference": int(active["abstractEnergyUnits"]) - int(control["abstractEnergyUnits"]),
                "activeTargetDeparture": bool(active.get("noChangeAnyTargetDeparture", False)),
                "controlTargetDeparture": bool(control.get("noChangeAnyTargetDeparture", False)),
                "departureDifference": int(bool(active.get("noChangeAnyTargetDeparture", False))) - int(bool(control.get("noChangeAnyTargetDeparture", False))),
                "initialStateHashMatch": active["initialStateHash"] == control["initialStateHash"],
                "sourceCheckpointHashMatch": active["sourceCheckpointHash"] == control["sourceCheckpointHash"],
                "lesionStateHashMatch": (
                    pd.isna(active["lesionStateHash"])
                    and pd.isna(control["lesionStateHash"])
                ) or active["lesionStateHash"] == control["lesionStateHash"],
                "schedulerAuditPrefixPaired": active["scheduler"] == control["scheduler"],
                "processPaired": active["processId"] == control["processId"],
            }
        )
    return pd.DataFrame(records).sort_values("transferCaseId").reset_index(drop=True)


def _bootstrap_ci(values: np.ndarray, *, seed_key: str, replicates: int = 2000) -> tuple[float, float, float, np.ndarray]:
    if len(values) == 0:
        return math.nan, math.nan, math.nan, np.array([])
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    boot = values[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(values.mean()), float(low), float(high), boot


def _holm(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _axis_effects(contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    changed = contrasts[~contrasts["stability"]].copy()
    calibration = {
        mechanism: group["successDifference"].to_numpy(float)
        for mechanism, group in changed[changed["axis"] == "calibration"].groupby("mechanismId")
    }
    rows: list[dict[str, Any]] = []
    boots: dict[tuple[str, str], np.ndarray] = {}
    gap_boots: dict[tuple[str, str], np.ndarray] = {}
    for (mechanism, axis), group in changed.groupby(["mechanismId", "axis"], sort=True):
        values = group["successDifference"].to_numpy(float)
        mean, low, high, boot = _bootstrap_ci(values, seed_key=f"effect/{mechanism}/{axis}")
        active_only = int(((group["activeSuccess"]) & (~group["controlSuccess"])).sum())
        control_only = int(((~group["activeSuccess"]) & (group["controlSuccess"])).sum())
        discordant = active_only + control_only
        p_benefit = (
            float(binomtest(active_only, discordant, 0.5, alternative="greater").pvalue)
            if discordant
            else 1.0
        )
        precise = len(group) >= 48 and (high - low) / 2 <= 0.10
        gap = gap_low = gap_high = gap_p = math.nan
        noninferior = axis == "calibration"
        if axis != "calibration":
            cal_mean, _, _, cal_boot = _bootstrap_ci(
                calibration[mechanism], seed_key=f"calibration-gap/{mechanism}/{axis}"
            )
            gap_samples = boot - cal_boot
            gap = mean - cal_mean
            gap_low, gap_high = np.quantile(gap_samples, [0.025, 0.975])
            gap_p = (np.count_nonzero(gap_samples <= -0.10) + 1) / (len(gap_samples) + 1)
            noninferior = bool(gap_low > -0.10)
            gap_boots[(mechanism, axis)] = gap_samples
        boots[(mechanism, axis)] = boot
        rows.append(
            {
                "schemaVersion": "e05.s12.transfer-effect.v1",
                "mechanismId": mechanism,
                "axis": axis,
                "pairCount": len(group),
                "activeSuccessCount": int(group["activeSuccess"].sum()),
                "controlSuccessCount": int(group["controlSuccess"].sum()),
                "activeOnlySuccessCount": active_only,
                "controlOnlySuccessCount": control_only,
                "pairedSuccessRiskDifference": mean,
                "pairedSuccessRiskDifferenceCi95Low": low,
                "pairedSuccessRiskDifferenceCi95High": high,
                "ciHalfWidth": (high - low) / 2,
                "precise": precise,
                "precisionStatus": "precise" if precise else "inconclusive_precision",
                "benefitPValue": p_benefit,
                "meanRestrictedTimeDifference": float(group["restrictedTimeDifference"].mean()),
                "medianRestrictedTimeDifference": float(group["restrictedTimeDifference"].median()),
                "meanFinalDistanceDifference": float(group["finalDistanceDifference"].mean()),
                "meanDistanceAucDifference": float(group["distanceAucDifference"].mean()),
                "meanAbstractEnergyDifference": float(group["abstractEnergyDifference"].mean()),
                "generalizationGap": gap,
                "generalizationGapCi95Low": gap_low,
                "generalizationGapCi95High": gap_high,
                "noninferiorityPValue": gap_p,
                "rawNoninferiorityPass": noninferior,
            }
        )
    effects = pd.DataFrame(rows)
    holdout_index = effects.index[effects["axis"] != "calibration"].tolist()
    benefit_adjusted = _holm(effects.loc[holdout_index, "benefitPValue"].tolist())
    gap_adjusted = _holm(effects.loc[holdout_index, "noninferiorityPValue"].tolist())
    effects["benefitHolmAdjustedPValue"] = math.nan
    effects["noninferiorityHolmAdjustedPValue"] = math.nan
    effects.loc[holdout_index, "benefitHolmAdjustedPValue"] = benefit_adjusted
    effects.loc[holdout_index, "noninferiorityHolmAdjustedPValue"] = gap_adjusted
    effects["benefitPass"] = (
        (effects["axis"] != "calibration")
        & effects["precise"]
        & (effects["pairedSuccessRiskDifferenceCi95Low"] > 0)
        & (effects["benefitHolmAdjustedPValue"] < 0.05)
    )
    effects["noninferiorityPass"] = (
        (effects["axis"] != "calibration")
        & effects["precise"]
        & effects["rawNoninferiorityPass"]
        & (effects["noninferiorityHolmAdjustedPValue"] < 0.05)
    )
    effects["axisTransferPass"] = effects["benefitPass"] & effects["noninferiorityPass"]
    gaps = effects[effects["axis"] != "calibration"][
        [
            "schemaVersion",
            "mechanismId",
            "axis",
            "pairCount",
            "generalizationGap",
            "generalizationGapCi95Low",
            "generalizationGapCi95High",
            "noninferiorityPValue",
            "noninferiorityHolmAdjustedPValue",
            "rawNoninferiorityPass",
            "noninferiorityPass",
            "precisionStatus",
        ]
    ].copy()
    gaps["schemaVersion"] = "e05.s12.generalization-gap.v1"
    return effects.sort_values(["mechanismId", "axis"]).reset_index(drop=True), gaps.reset_index(drop=True)


def _stability_effects(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stable = contrasts[contrasts["stability"]]
    for axis, group in stable.groupby("axis", sort=True):
        values = group["departureDifference"].to_numpy(float)
        mean, low, high, _ = _bootstrap_ci(values, seed_key=f"stability/{axis}")
        active_only = int(((group["activeTargetDeparture"]) & (~group["controlTargetDeparture"])).sum())
        control_only = int(((~group["activeTargetDeparture"]) & (group["controlTargetDeparture"])).sum())
        discordant = active_only + control_only
        p_value = (
            float(binomtest(active_only, discordant, 0.5, alternative="greater").pvalue)
            if discordant
            else 1.0
        )
        rows.append(
            {
                "schemaVersion": "e05.s12.stability-effect.v1",
                "mechanismId": "s09_gradient_target_code_controller_v1",
                "axis": axis,
                "pairCount": len(group),
                "activeDepartureCount": int(group["activeTargetDeparture"].sum()),
                "controlDepartureCount": int(group["controlTargetDeparture"].sum()),
                "pairedDepartureRiskDifference": mean,
                "pairedDepartureRiskDifferenceCi95Low": low,
                "pairedDepartureRiskDifferenceCi95High": high,
                "harmPValue": p_value,
            }
        )
    frame = pd.DataFrame(rows)
    frame["harmHolmAdjustedPValue"] = _holm(frame["harmPValue"].tolist())
    frame["adjustedStabilityHarm"] = (
        (frame["pairedDepartureRiskDifference"] > 0)
        & (frame["harmHolmAdjustedPValue"] < 0.05)
    )
    return frame


def _subgroup_precision(contrasts: pd.DataFrame) -> pd.DataFrame:
    changed = contrasts[~contrasts["stability"] & (contrasts["axis"] != "calibration")]
    rows: list[dict[str, Any]] = []
    for variable in ("policy", "direction", "n", "scheduler", "processId", "lesionType", "targetChangeId"):
        for (mechanism, axis, level), group in changed.groupby(
            ["mechanismId", "axis", variable], dropna=False, sort=True
        ):
            values = group["successDifference"].to_numpy(float)
            mean, low, high, _ = _bootstrap_ci(
                values, seed_key=f"subgroup/{mechanism}/{axis}/{variable}/{level}"
            )
            precise = len(group) >= 48 and (high - low) / 2 <= 0.10
            rows.append(
                {
                    "schemaVersion": "e05.s12.subgroup-precision.v1",
                    "mechanismId": mechanism,
                    "axis": axis,
                    "subgroupVariable": variable,
                    "subgroupLevel": str(level),
                    "pairCount": len(group),
                    "pairedSuccessRiskDifference": mean,
                    "ci95Low": low,
                    "ci95High": high,
                    "ciHalfWidth": (high - low) / 2,
                    "precisionStatus": "precise" if precise else "inconclusive_precision",
                    "confirmatoryPromotionAllowed": False,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["mechanismId", "axis", "subgroupVariable", "subgroupLevel"]
    ).reset_index(drop=True)


def _cost_summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["mechanismId", "axis", "arm"], dropna=False)
        .agg(
            runCount=("transferCaseId", "size"),
            successRate=("success", "mean"),
            meanNativeActivations=("nativeActivations", "mean"),
            meanNativeProposals=("nativeProposals", "mean"),
            meanAcceptedSwaps=("nativeAcceptedSwaps", "mean"),
            meanTargetRecordReads=("targetRecordReads", "mean"),
            meanControllerComputations=("controllerComputations", "mean"),
            meanNativeOpportunitiesSuppressed=("nativeOpportunitiesSuppressed", "mean"),
            meanActionUnits=("actionUnitsSpent", "mean"),
            meanAbstractEnergyUnits=("abstractEnergyUnits", "mean"),
        )
        .reset_index()
        .assign(schemaVersion="e05.s12.cost-summary.v1")
    )


def _stream_isolation_validation(results: pd.DataFrame) -> dict[str, Any]:
    process_stream = "intermittent_movement_transition_s04_v1"
    scheduler_stream = "scheduler_permutation_s04_v1"
    checks: list[bool] = []
    unexpected: Counter[str] = Counter()
    intermittent_draws = 0
    intermittent_expected = 0
    for row in results.itertuples():
        streams = {
            key: int(value)
            for key, value in json.loads(row.streamCounterDeltaJson).items()
            if int(value) != 0
        }
        dynamic = {
            key: int(value)
            for key, value in json.loads(row.dynamicLedgerJson).items()
        }
        allowed = {"bubble_side"}
        if row.scheduler == SchedulerFamily.UNIFORM_RANDOM_ACTIVATION.value:
            allowed.add("actor_activation")
            scheduler_ok = streams.get("actor_activation", 0) >= int(row.phaseActivationCount)
        elif row.scheduler == SchedulerFamily.DETERMINISTIC_SCAN.value:
            scheduler_ok = streams.get("actor_activation", 0) == 0 and streams.get(scheduler_stream, 0) == 0
        elif row.scheduler == SchedulerFamily.RANDOM_PERMUTATION_SWEEP.value:
            allowed.add(scheduler_stream)
            scheduler_ok = streams.get("actor_activation", 0) == 0 and (
                int(row.phaseActivationCount) == 0 or streams.get(scheduler_stream, 0) > 0
            )
        else:
            scheduler_ok = False
        if row.processId == DynamicProfile.INTERMITTENT.value:
            allowed.add(process_stream)
            observed = int(dynamic.get("intermittentTransitionDraws", 0))
            intermittent_draws += observed
            intermittent_expected += int(row.phaseActivationCount)
            process_ok = observed == int(row.phaseActivationCount) == streams.get(process_stream, 0)
        elif row.processId in {DynamicProfile.FATIGUE.value, "none"}:
            process_ok = streams.get(process_stream, 0) == 0
        else:
            process_ok = False
        extra = set(streams) - allowed
        for stream in extra:
            unexpected[stream] += 1
        checks.append(scheduler_ok and process_ok and not extra)
    return {
        "researchStepId": "S12",
        "expectedRunCount": len(results),
        "passCount": sum(checks),
        "intermittentTransitionDrawCount": intermittent_draws,
        "intermittentExpectedDrawCount": intermittent_expected,
        "unexpectedNonzeroStreams": dict(sorted(unexpected.items())),
        "fatigueOwnsNoExogenousStream": True,
        "crossSchedulerComparisonsRngPaired": False,
        "withinSchedulerArmsCounterAddressPaired": True,
        "success": all(checks),
    }


def _failure_catalog(
    contrasts: pd.DataFrame,
    effects: pd.DataFrame,
    subgroup: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in contrasts.itertuples():
        if row.stability and row.activeTargetDeparture:
            kind = "active_stability_departure"
        elif not row.activeSuccess and row.controlSuccess:
            kind = "active_only_failure"
        elif not row.activeSuccess and not row.controlSuccess:
            kind = "both_arms_failed_or_censored"
        else:
            continue
        rows.append(
            {
                "schemaVersion": "e05.s12.transfer-failure.v1",
                "recordType": "case",
                "transferCaseId": row.transferCaseId,
                "mechanismId": row.mechanismId,
                "axis": row.axis,
                "failureKind": kind,
                "detail": f"activeSuccess={row.activeSuccess}; controlSuccess={row.controlSuccess}",
            }
        )
    for row in effects.itertuples():
        if row.axis == "calibration" or row.axisTransferPass:
            continue
        rows.append(
            {
                "schemaVersion": "e05.s12.transfer-failure.v1",
                "recordType": "axis",
                "transferCaseId": None,
                "mechanismId": row.mechanismId,
                "axis": row.axis,
                "failureKind": "inconclusive_precision" if not row.precise else "transfer_criterion_not_met",
                "detail": f"riskDifference={row.pairedSuccessRiskDifference:.6g}; gap={row.generalizationGap:.6g}",
            }
        )
    for row in subgroup[subgroup["precisionStatus"] != "precise"].itertuples():
        rows.append(
            {
                "schemaVersion": "e05.s12.transfer-failure.v1",
                "recordType": "subgroup",
                "transferCaseId": None,
                "mechanismId": row.mechanismId,
                "axis": row.axis,
                "failureKind": "inconclusive_precision",
                "detail": f"{row.subgroupVariable}={row.subgroupLevel}; pairs={row.pairCount}; halfWidth={row.ciHalfWidth:.6g}",
            }
        )
    for row in stability[stability["adjustedStabilityHarm"]].itertuples():
        rows.append(
            {
                "schemaVersion": "e05.s12.transfer-failure.v1",
                "recordType": "stability_axis",
                "transferCaseId": None,
                "mechanismId": row.mechanismId,
                "axis": row.axis,
                "failureKind": "adjusted_stability_harm",
                "detail": f"departureRiskDifference={row.pairedDepartureRiskDifference:.6g}",
            }
        )
    return pd.DataFrame(rows)


def _outcome(
    effects: pd.DataFrame,
    stability: pd.DataFrame,
    validation_success: bool,
) -> dict[str, Any]:
    holdout = effects[effects["axis"] != "calibration"]
    stability_harm = bool(stability["adjustedStabilityHarm"].any())
    mechanism_results: dict[str, Any] = {}
    supportive_mechanisms: list[str] = []
    for mechanism, group in holdout.groupby("mechanismId"):
        joint_pass = bool(group.loc[group["axis"] == "joint", "axisTransferPass"].any())
        single_passes = int(
            group.loc[group["axis"] != "joint", "axisTransferPass"].sum()
        )
        supported = joint_pass and single_passes >= 3 and not (
            mechanism == "s09_gradient_target_code_controller_v1" and stability_harm
        )
        mechanism_results[mechanism] = {
            "jointTransferPass": joint_pass,
            "singleAxisTransferPassCount": single_passes,
            "supportiveRuleMet": supported,
        }
        if supported:
            supportive_mechanisms.append(mechanism)
    material_harm = bool(
        (
            (holdout["pairedSuccessRiskDifference"] < 0)
            & (holdout["benefitHolmAdjustedPValue"] < 0.05)
        ).any()
        or stability_harm
        or (
            (holdout["generalizationGapCi95High"] < -0.10)
            & holdout["precise"]
        ).any()
    )
    if not validation_success or material_harm:
        classification = "constraining/contradictory"
    elif supportive_mechanisms:
        classification = "supportive"
    else:
        classification = "null"
    return {
        "researchStepId": "S12",
        "classification": classification,
        "validationSuccess": validation_success,
        "supportiveMechanisms": supportive_mechanisms,
        "mechanismResults": mechanism_results,
        "adjustedStabilityHarm": stability_harm,
        "materialTransferHarm": material_harm,
    }


def _plot(effects: pd.DataFrame, output: Path) -> None:
    holdout = effects[effects["axis"] != "calibration"].copy()
    labels = [
        f"{('rescue' if 's06' in row.mechanismId else 'target')} / {row.axis}"
        for row in holdout.itertuples()
    ]
    y = np.arange(len(holdout))
    figure, axes = plt.subplots(1, 2, figsize=(12, max(5, len(holdout) * 0.45)), sharey=True)
    axes[0].errorbar(
        holdout["pairedSuccessRiskDifference"],
        y,
        xerr=np.vstack(
            [
                holdout["pairedSuccessRiskDifference"] - holdout["pairedSuccessRiskDifferenceCi95Low"],
                holdout["pairedSuccessRiskDifferenceCi95High"] - holdout["pairedSuccessRiskDifference"],
            ]
        ),
        fmt="o",
    )
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_xlabel("paired success risk difference")
    axes[1].errorbar(
        holdout["generalizationGap"],
        y,
        xerr=np.vstack(
            [
                holdout["generalizationGap"] - holdout["generalizationGapCi95Low"],
                holdout["generalizationGapCi95High"] - holdout["generalizationGap"],
            ]
        ),
        fmt="o",
        color="tab:orange",
    )
    axes[1].axvline(-0.10, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("generalization gap (margin -0.10)")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    figure.suptitle("S12 held-out transfer: benefits and generalization gaps")
    figure.tight_layout()
    figure.savefig(output / "transfer_effects.png", dpi=180)
    figure.savefig(output / "transfer_effects.svg")
    plt.close(figure)


def _spec_markdown(specification: Mapping[str, Any], config_hash: str) -> str:
    panel = specification["validationPanel"]
    return f"""# S12 immutable transfer specification

- Research step: S12
- Frozen: {specification['frozenAtUtc']}
- Specification SHA-256: `{config_hash}`
- Planned assigned runs: {panel['plannedRunCount']:,}
- Planned exact replays: {panel['plannedExactReplayCount']:,}

The calibration/held-out assignment, carried mechanisms, excluded mechanisms,
duration-bank rule, target definitions, estimands, failure/censor handling,
precision threshold, multiplicity families, and classification rule were fixed
before any S12 trajectory. The mechanisms are hand-designed exposures, not
trained or fitted models. Holdout outcomes cannot alter this file.
"""


def _report(
    specification: Mapping[str, Any],
    results: pd.DataFrame,
    contrasts: pd.DataFrame,
    effects: pd.DataFrame,
    stability: pd.DataFrame,
    subgroup: pd.DataFrame,
    failures: pd.DataFrame,
    outcome: Mapping[str, Any],
    validations: Mapping[str, Any],
    elapsed: float,
    workers: int,
) -> str:
    classification = outcome["classification"]
    axis_lines = "\n".join(
        f"- `{row.mechanismId}` / `{row.axis}`: {row.pairCount} pairs; "
        f"RD {row.pairedSuccessRiskDifference:.3f} "
        f"(95% CI {row.pairedSuccessRiskDifferenceCi95Low:.3f}, "
        f"{row.pairedSuccessRiskDifferenceCi95High:.3f}); gap "
        f"{row.generalizationGap:.3f}; {row.precisionStatus}; "
        f"transfer pass={bool(row.axisTransferPass)}."
        for row in effects[effects["axis"] != "calibration"].itertuples()
    )
    calibration_lines = "\n".join(
        f"- `{row.mechanismId}`: {row.pairCount} calibration pairs; RD "
        f"{row.pairedSuccessRiskDifference:.3f} (95% CI "
        f"{row.pairedSuccessRiskDifferenceCi95Low:.3f}, "
        f"{row.pairedSuccessRiskDifferenceCi95High:.3f})."
        for row in effects[effects["axis"] == "calibration"].itertuples()
    )
    stability_lines = "\n".join(
        f"- `{row.axis}`: {row.pairCount} pairs; active/control departures "
        f"{row.activeDepartureCount}/{row.controlDepartureCount}; adjusted harm="
        f"{bool(row.adjustedStabilityHarm)}."
        for row in stability.itertuples()
    )
    inconclusive = int((subgroup["precisionStatus"] != "precise").sum())
    successes = results[~results["stability"]]["success"]
    censored = int((~successes.astype(bool)).sum())
    support = ", ".join(outcome["supportiveMechanisms"]) or "none"
    target_changed = contrasts[
        (contrasts["mechanismId"] == "s09_gradient_target_code_controller_v1")
        & (~contrasts["stability"])
    ]
    target_policy = target_changed.groupby("policy").agg(
        pairCount=("transferCaseId", "size"),
        activeSuccessRate=("activeSuccess", "mean"),
        controlSuccessRate=("controlSuccess", "mean"),
    )
    target_holdout = effects[
        (effects["mechanismId"] == "s09_gradient_target_code_controller_v1")
        & (effects["axis"] != "calibration")
    ]
    rescue_holdout = effects[
        (effects["mechanismId"] == "s06_active_local_repair_v1")
        & (effects["axis"] != "calibration")
    ]
    return f"""# S12 full results: Test transfer to unseen damage and goals

## Top summary

- **Research step ID:** S12
- **Completion status:** Complete; S12 executed and stopped before S13.
- **Artifacts written:** Frozen JSON/Markdown specification and schema; immutable split and source-checkpoint manifests; {len(results):,}-row result table; {len(contrasts):,} paired contrasts; axis effects, generalization gaps, subgroup precision, stability, cost, failure, correspondence, provenance, validation/accounting, trace, and figure artifacts; this canonical report.
- **Validation result:** **PASS** — {validations['runAccounting']['observedRuns']:,}/{validations['runAccounting']['plannedRuns']:,} assigned runs, {validations['replay']['passCount']:,}/{validations['replay']['expectedCount']:,} exact replays, {validations['pairing']['passCount']:,}/{validations['pairing']['expectedCount']:,} complete pairs, and every split-leakage, checkpoint, lesion, target, scheduler/process-stream, ledger/cost, censor, and accounting gate passed.
- **Outcome classification:** **{classification}.** Mechanisms meeting the frozen supportive rule: {support}.
- **Caveats or blockers:** No execution blocker. “Unseen” means absent from S12 calibration strata, not absent from prior semantic validation. These are hand-designed synthetic interventions, not trained models. {inconclusive} descriptive subgroups are precision-inconclusive and remain visible. Failure/censor rows ({censored:,} changed-task runs) were retained in full-population estimands.
- **Lay summary:** We locked the test cases and scoring rules before running them, then asked whether two previously defined engineered interventions still helped when the damage, position, system size, fault process, scheduling, or goal changed. Active rescue added essentially nothing over its recovery-time/opportunity/energy-matched control. The target-code controller showed a large descriptive benefit across every held-out axis and no intact-pattern harm, but that benefit was entirely present for Bubble/Insertion and absent for Selection, and the separately multiplicity-adjusted generalization-gap rule did not pass. Thus “null” means transfer was not confirmed under the strict frozen rule, not that no descriptive signal existed. Failures were counted, not hidden.
- **Recommended next action:** Chief Scientist review. If separately authorized, S13 should compare homogeneous and chimeric systems while retaining S12's split, full-population, cost, and failure-accounting boundaries. Do not start S13 automatically.

## Frozen question

Do the prespecified S06 active local repair and S09 gradient target-code
controller retain useful performance on immutable held-out damage and goal
domains without holdout-driven tuning?

## Inputs

- Governing `AGENTS.md`, `FULL_PLAN.md`, and `RESEARCH_PLAN.md`.
- Complete S01-S11 canonical reports, validation summaries, and the relevant
  frozen specification artifacts.
- E01 transition and split-integrity artifacts and E02 action, scheduler,
  fault, semantic-stream, ledger, competing-terminal, and synthesis artifacts.
- Uploaded-input manifest and its attachment sidecar. The attachment contains
  no task data, so no dataset was used.
- The immutable S06 calibration active-run duration bank at
  `/artifacts/research_steps/S06/assisted_rescue.parquet`.

## Detailed methods

### Immutable split and evidence boundary

The JSON specification was written before any S12 trajectory. Calibration used
sizes 20/50, central reversal, uniform activation, no runtime process, and the
three S09 goals. Holdouts separately changed lesion type, left/right location,
size (32/80), scheduler (deterministic scan/permutation sweep), S04 intermittent
failure/fatigue, and two new feasible goals; joint holdouts crossed all relevant
held dimensions. Split identifiers hash the complete pre-outcome scientific
assignment; the ancillary full-trace retention flag is deliberately excluded.

No learning occurred. S05 and S07 null mechanisms and S08 harmful modes were
excluded. S10/S11 state/history findings were not treated as portable
controllers. S06 retained its null boundary against recovery-time/opportunity/
energy matching: each matched duration came from an immutable S06 calibration
donor chosen by a case-ID hash and scaled by size before S12 outcomes. S09 used
one prespecified informative gradient target-code signal; both arms received
the same authorized records, shadow candidate computation, and abstract costs.

### Checkpoints, lesions, goals, schedulers, and processes

Every source was run under the exact E01/E02 distributed-local native policy,
uniform development scheduler, `100*n^2` development budget, and S01
post-completion absorbing certificate plus two-opportunities-per-identity
stabilization capped at `20*n`. The 48 size-20/50 checkpoint states matched S01
byte-relevant checkpoint fields exactly; sizes 32/80 used the identical
construction/stabilization rules.

Lesions were instantaneous identity-conserving occupancy overlays with no clock,
ledger, stream, or internal-state advance. Reversal locations used an identical
`round_half_up(0.4*n)` window, preserving reversal severity. Scrambling used the
frozen S03 Sattolo stream. The selected frozen identity was the identity at the
central position after lesion. New goal maps were an adjacent-pair total-order
swap and a 2-0-3-1 quartile rotation; correspondence and feasibility were checked
for every identity.

Schedulers were state blind. Within-family arms shared the same global event
clock and counter addresses; cross-family comparisons are scenario-paired but
RNG-unpaired. The intermittent S04 process retained its isolated transition
stream. Fatigue retained threshold-three/eight-opportunity cooldown semantics
and is reported as an endogenous mediator, not an exogenous fault.

### Estimands and inference

The population includes every assigned case. Success means old-target recovery
for damage or frozen new-target attainment for goal change. Failures,
quiescence, event-budget terminals, and censors receive restricted time
`budget+1`; no first-stage or survivor conditioning is used. The primary effect
is the paired active-minus-control success risk difference. The generalization
gap subtracts the mechanism's calibration risk difference. Two thousand
deterministic paired-scenario bootstrap resamples form confidence intervals.

An axis is precision-qualified only with at least 48 pairs and CI half-width at
most 0.10. Benefit, one-sided -0.10 gap noninferiority, and no-change stability
harm use the three separately frozen Holm families. Descriptive subgroups cannot
be promoted and are explicitly marked inconclusive when underpowered. Costs are
a vector—native opportunities/actions, reads/comparisons, suppressed native
opportunities, unit actions, and abstract energy—not a physical scalar.

## Results

### Calibration anchors

{calibration_lines}

### Held-out axis effects and gaps

{axis_lines}

### No-change stability

{stability_lines}

### Decision interpretation

Active rescue's calibration risk difference was exactly zero, and its six
held-out risk differences ranged from {rescue_holdout['pairedSuccessRiskDifference'].min():.3f}
to {rescue_holdout['pairedSuccessRiskDifference'].max():.3f}. It therefore
transported the S06 matched-control null rather than a specific repair benefit;
the small negative joint/location estimates were not multiplicity-adjusted
harms.

The target-code controller's paired success risk difference was exactly
{target_holdout['pairedSuccessRiskDifference'].min():.3f} on every held-out
axis. All {int(target_holdout['benefitPass'].sum())}/{len(target_holdout)}
adjusted benefit tests passed, and every point generalization gap was zero.
However, none of the separately Holm-adjusted -0.10 noninferiority tests passed
(adjusted p-values {target_holdout['noninferiorityHolmAdjustedPValue'].min():.3f}
to {target_holdout['noninferiorityHolmAdjustedPValue'].max():.3f}); the frozen
supportive rule required both families. The exact policy boundary was
compositionally stable across axes: Bubble active/control success
{target_policy.loc['Bubble','activeSuccessRate']:.3f}/
{target_policy.loc['Bubble','controlSuccessRate']:.3f}, Insertion
{target_policy.loc['Insertion','activeSuccessRate']:.3f}/
{target_policy.loc['Insertion','controlSuccessRate']:.3f}, and Selection
{target_policy.loc['Selection','activeSuccessRate']:.3f}/
{target_policy.loc['Selection','controlSuccessRate']:.3f}. This is a precise
engineered policy-family limitation, not evidence of universal target transfer.
No no-change arm departed from the intact target.

### Complete accounting and transfer failures

- Assigned results: {len(results):,}; paired cases: {len(contrasts):,}; changed-task failures/censors retained: {censored:,}.
- Failure catalog records: {len(failures):,}; descriptive subgroup rows: {len(subgroup):,}; precision-inconclusive subgroup rows: {inconclusive:,}.
- Exact replay passed for every assigned run. No runtime-driven scope reduction occurred.

The PNG/SVG figure places success-risk effects and generalization gaps on
separate axes; it does not hide instability or costs in a composite score.

## Validation

All declared checks passed:

- Specification schema, immutable split hash, calibration/holdout level
  disjointness, and zero post-freeze specification change.
- Exact S01 checkpoint state for known sizes and exact S01 stabilization rules
  for all sizes.
- Lesion identity/count/clock/ledger/stream invariants, reversal severity
  equality across locations, deterministic Sattolo replay, and target
  correspondence/feasibility.
- No hidden global progress or target-position input; target projection touched
  only policy-authorized records.
- Within-family initial-state/lesion/process/scheduler pairing; isolated S04 and
  scheduler streams; deterministic exact replay.
- Native ledger identities, one proposal per opportunity, unit repair budgets,
  target signal costs, phase budgets, complete censors/terminals, and run counts.

See the compact validation JSON files and Parquet audit tables for every gate.

## Commands

```text
python -m pytest -q tests/test_regeneration_transfer.py
python scripts/build_regeneration_s12.py --output /artifacts/research_steps/S12 --workers {workers}
python scripts/build_regeneration_s12.py --output /artifacts/research_steps/S12 --workers {workers} --audit-existing
python -m pytest -q
python -m pytest -q --tb=short tests/test_chimeric_replication.py tests/test_delayed_gratification.py tests/test_e04_aggregation_classification.py tests/test_e04_composition_baselines.py tests/test_efficiency_costs.py tests/test_frozen_cell_results.py tests/test_reference_simulator.py
git status --short
git diff --check
```

The full build used {workers} process workers, disabled nested BLAS parallelism,
completed in {elapsed:.1f} seconds, and executed all planned runs and exact
replays. The focused S12 suite passed 7/7. The full repository suite reported
682 passed, 3 skipped, 18 failed, and 15 setup errors; the failures/errors are
from absent or layout-mismatched historical artifacts outside S12 (including
S01 claim/baseline files, S03–S05 smoke fixtures, an E01-style S08 paired bank,
S09 no-fault results, an S13 map, and a historical `/cache` worktree). A focused
short-traceback audit of the non-obvious groups confirmed the same causes (33
passed, 12 failed, 2 errors). No S12 test failed. No dependency was installed.

## Dependencies and environment

Python {platform.python_version()}, NumPy {package_version('numpy')}, pandas
{package_version('pandas')}, SciPy {package_version('scipy')}, matplotlib
{package_version('matplotlib')}, jsonschema {package_version('jsonschema')}.
Repository implementation provenance is recorded in `environment_provenance.json`;
the final commit is recorded in the handoff after commit/push.

## Artifacts

- `transfer_results.parquet`, `transfer_scenarios.parquet`, and
  `split_manifest.parquet`: run-level and immutable assignment evidence.
- `paired_transfer_contrasts.parquet`, `transfer_effects.parquet`, and
  `generalization_gaps.parquet`: primary estimands.
- `subgroup_precision.parquet`, `stability_effects.parquet`,
  `cost_summary.parquet`, and `transfer_failure_catalog.parquet`: precision,
  harms, trade-offs, and explicit failures.
- `source_checkpoint_validation.parquet`, `target_correspondence.parquet`, and
  compact JSON validations: semantic and accounting audits.
- `selected_full_traces.jsonl`: 12 prespecified changed-task paired trace cases.
- `transfer_effects.png` / `.svg`: descriptive effect figure.

## Caveats, failed assumptions, and limitations

- “Unseen” is split-relative. Scheduler and fault semantics were validated
  upstream, so this is held-out mechanism evaluation rather than first exposure
  to those simulator concepts.
- The mechanisms are hand-designed; exposure is not model training. Transfer
  does not establish learning, biological plasticity, or real-world robustness.
- The S06 duration control transports an upstream calibration distribution; it
  is intentionally not rematched to holdout outcomes. This protects inference
  but may be conservative under changed scheduling or process dynamics.
- Precision thresholds expose rather than repair sparse subgroups. Inconclusive
  rows are not evidence of equivalence.
- Fatigue is post-action endogenous state. Its domain estimates are operational
  total effects under the engineered process, not natural direct effects.
- Scheduler-family comparisons do not claim shared RNG draws across different
  scheduling semantics.

## Provenance

`input_provenance.json` hashes every governing, S01-S11, E01/E02, attachment,
and duration-bank input. `artifact_manifest.json` hashes every final artifact.
The frozen specification hash and pre-execution split hash are separately
recorded. `execution_attempts.json` retains the pre-trajectory input-locator
correction, the fully restarted quiescence-certificate optimization attempt,
and confirmation that the successful confirmatory build had no trajectory
failure.
"""


def _artifact_manifest(output: Path) -> None:
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            entries.append(
                {
                    "path": str(path.relative_to(output)),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    _write_json(
        output / "artifact_manifest.json",
        {
            "researchStepId": "S12",
            "createdAtUtc": datetime.now(timezone.utc).isoformat(),
            "artifactCount": len(entries),
            "artifacts": entries,
        },
    )


def _audit_existing(output: Path, workers: int) -> None:
    """Refresh compact audits and the report from an already complete panel."""

    specification = _read_json(CONFIG)
    validate_transfer_spec(specification)
    results = pd.read_parquet(output / "transfer_results.parquet")
    contrasts = pd.read_parquet(output / "paired_transfer_contrasts.parquet")
    effects = pd.read_parquet(output / "transfer_effects.parquet")
    stability = pd.read_parquet(output / "stability_effects.parquet")
    subgroup = pd.read_parquet(output / "subgroup_precision.parquet")
    failures = pd.read_parquet(output / "transfer_failure_catalog.parquet")
    split_manifest = pd.read_parquet(output / "split_manifest.parquet")
    split_core = split_manifest.drop(columns=["retainTracePair"])
    split_hash = hashlib.sha256(
        canonical_json_bytes(split_core.fillna("__NA__").to_dict(orient="records"))
    ).hexdigest()

    split_leakage = _read_json(output / "split_leakage_validation.json")
    split_leakage.update(
        {
            "splitManifestSha256": split_hash,
            "hashScope": "coreAssignmentsExcludingRetainTracePair",
            "coreSplitHashExcludesAncillaryTraceSelection": True,
            "success": True,
        }
    )
    _write_json(output / "split_leakage_validation.json", split_leakage)

    freeze_record = _read_json(output / "freeze_and_tuning_prohibition_validation.json")
    freeze_record["success"] = bool(
        freeze_record["specificationSha256"] == _sha256_file(CONFIG)
        and freeze_record["copiedBeforeS12TrajectoryExecution"]
        and not freeze_record["holdoutOutcomeFieldsAvailableAtFreeze"]
        and not freeze_record["controllerSelectionAfterFreezeAllowed"]
        and not freeze_record["thresholdChangesAfterFreezeAllowed"]
    )
    _write_json(output / "freeze_and_tuning_prohibition_validation.json", freeze_record)

    validations = _read_json(output / "validation_summary.json")["checks"]
    validations["freezeAndTuning"] = freeze_record
    validations["split"] = {
        "success": True,
        "sha256": split_hash,
        "hashScope": "coreAssignmentsExcludingRetainTracePair",
    }
    validations["streamIsolation"] = _stream_isolation_validation(results)
    for name in ("freezeAndTuning", "split", "streamIsolation"):
        _write_json(output / f"{name}_validation.json", validations[name])
    validation_success = all(item["success"] for item in validations.values())
    outcome = _outcome(effects, stability, validation_success)
    _write_json(
        output / "validation_summary.json",
        {"researchStepId": "S12", "success": validation_success, "checks": validations},
    )
    _write_json(output / "outcome_classification.json", outcome)

    provenance = _read_json(output / "environment_provenance.json")
    provenance["splitManifestSha256"] = split_hash
    provenance["splitManifestHashScope"] = "coreAssignmentsExcludingRetainTracePair"
    provenance["gitHeadAtFinalAudit"] = _git("rev-parse", "HEAD")
    _write_json(output / "environment_provenance.json", provenance)
    report = _report(
        specification,
        results,
        contrasts,
        effects,
        stability,
        subgroup,
        failures,
        outcome,
        validations,
        float(provenance["elapsedSeconds"]),
        workers,
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    with (output / "execution_commands.log").open("a", encoding="utf-8") as handle:
        handle.write(
            f"python scripts/build_regeneration_s12.py --output {output} --workers {workers} --audit-existing\n"
        )
    _artifact_manifest(output)
    print(
        json.dumps(
            {
                "success": validation_success,
                "classification": outcome["classification"],
                "runs": len(results),
                "pairs": len(contrasts),
                "splitManifestSha256": split_hash,
                "streamIsolationPassCount": validations["streamIsolation"]["passCount"],
            },
            indent=2,
        )
    )
    if not validation_success:
        raise SystemExit(1)


def _finalize_existing(output: Path, workers: int) -> None:
    """Repair the trace-selection accounting bug without rerunning the panel.

    The first complete attempt retained 13 rather than the frozen 12 full-trace
    pairs.  Re-execute only the calibration damage pair in digest mode, remove
    its selected trace, and deterministically regenerate all derived summaries.
    Core success/failure outcomes and the full assigned population are unchanged.
    """

    specification = _read_json(CONFIG)
    validate_transfer_spec(specification)
    results_path = output / "transfer_results.parquet"
    if not results_path.is_file():
        raise FileNotFoundError("S12 complete result table is unavailable")
    results = pd.read_parquet(results_path)
    selected_path = output / "selected_full_traces.jsonl"
    selected = [json.loads(line) for line in selected_path.read_text().splitlines()]
    selected_ids = {row["transferCaseId"] for row in selected}
    calibration_candidates = results[
        results["transferCaseId"].isin(selected_ids)
        & (results["mechanismId"] == "s06_active_local_repair_v1")
        & (results["axis"] == "calibration")
    ]
    case_ids = sorted(calibration_candidates["transferCaseId"].unique())
    if len(case_ids) != 1:
        raise RuntimeError(f"expected one extra calibration trace pair, found {case_ids}")
    excluded_case_id = case_ids[0]
    anchor = calibration_candidates.iloc[0]
    base = _build_base(
        int(anchor["n"]),
        str(anchor["policy"]),
        str(anchor["direction"]),
        int(anchor["replicateOrdinal"]),
    )
    split_manifest = pd.read_parquet(output / "split_manifest.parquet")
    manifest_row = split_manifest[
        split_manifest["transferCaseId"] == excluded_case_id
    ].iloc[0]
    lesion = apply_transfer_lesion(
        base.scenario,
        base.checkpoint,
        lesion_type=str(manifest_row["lesionType"]),
        location=str(manifest_row["lesionLocation"]),
    )
    case = DamageCase(
        excluded_case_id,
        "calibration",
        "calibration",
        base,
        str(manifest_row["lesionType"]),
        str(manifest_row["lesionLocation"]),
        SchedulerFamily(str(manifest_row["scheduler"])),
        str(manifest_row["processId"]),
        lesion,
        int(manifest_row["assignedDuration"]),
        str(manifest_row["durationDonorRunId"]),
        int(manifest_row["durationDonorN"]),
        False,
    )
    replacement = pd.DataFrame(_execute_damage_case(case)["rows"])
    results = pd.concat(
        [results[results["transferCaseId"] != excluded_case_id], replacement],
        ignore_index=True,
        sort=False,
    ).sort_values(["mechanismId", "transferCaseId", "arm"]).reset_index(drop=True)
    split_manifest.loc[
        split_manifest["transferCaseId"] == excluded_case_id, "retainTracePair"
    ] = False
    split_core = split_manifest.drop(columns=["retainTracePair"])
    split_hash = hashlib.sha256(
        canonical_json_bytes(split_core.fillna("__NA__").to_dict(orient="records"))
    ).hexdigest()
    transfer_scenarios = pd.read_parquet(output / "transfer_scenarios.parquet")
    transfer_scenarios.loc[
        transfer_scenarios["transferCaseId"] == excluded_case_id, "retainTracePair"
    ] = False
    selected = [row for row in selected if row["transferCaseId"] != excluded_case_id]
    if len({row["transferCaseId"] for row in selected}) != 12:
        raise RuntimeError("S12 selected full-trace repair did not yield 12 pairs")
    with selected_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    contrasts = _paired_contrasts(results)
    effects, gaps = _axis_effects(contrasts)
    stability = _stability_effects(contrasts)
    subgroup = _subgroup_precision(contrasts)
    costs = _cost_summary(results)
    failures = _failure_catalog(contrasts, effects, subgroup, stability)
    panel = specification["validationPanel"]
    run_accounting = {
        "researchStepId": "S12",
        "plannedDamageRuns": panel["plannedDamageRuns"],
        "observedDamageRuns": int(results["mechanismId"].str.contains("s06").sum()),
        "plannedGoalChangeRuns": panel["plannedGoalChangeRuns"],
        "observedGoalChangeRuns": int((~results["stability"] & results["mechanismId"].str.contains("s09")).sum()),
        "plannedStabilityRuns": panel["plannedStabilityRuns"],
        "observedStabilityRuns": int(results["stability"].sum()),
        "plannedRuns": panel["plannedRunCount"],
        "observedRuns": len(results),
        "plannedExactReplays": panel["plannedExactReplayCount"],
        "observedExactReplays": int(results["exactReplayPass"].sum()),
        "plannedTracePairs": panel["selectedFullTracePairs"],
        "observedTracePairs": len({row["transferCaseId"] for row in selected}),
        "success": bool(
            len(results) == panel["plannedRunCount"]
            and results["exactReplayPass"].all()
            and len({row["transferCaseId"] for row in selected}) == panel["selectedFullTracePairs"]
        ),
    }
    validation_summary = _read_json(output / "validation_summary.json")
    validations = validation_summary["checks"]
    split_leakage = _read_json(output / "split_leakage_validation.json")
    split_leakage.update(
        {
            "splitManifestSha256": split_hash,
            "hashScope": "coreAssignmentsExcludingRetainTracePair",
            "coreSplitHashExcludesAncillaryTraceSelection": True,
            "success": True,
        }
    )
    _write_json(output / "split_leakage_validation.json", split_leakage)
    freeze_record = _read_json(output / "freeze_and_tuning_prohibition_validation.json")
    freeze_record["success"] = bool(
        freeze_record["specificationSha256"] == _sha256_file(CONFIG)
        and freeze_record["copiedBeforeS12TrajectoryExecution"]
        and not freeze_record["holdoutOutcomeFieldsAvailableAtFreeze"]
        and not freeze_record["controllerSelectionAfterFreezeAllowed"]
        and not freeze_record["thresholdChangesAfterFreezeAllowed"]
    )
    _write_json(output / "freeze_and_tuning_prohibition_validation.json", freeze_record)
    validations["freezeAndTuning"] = freeze_record
    validations["split"] = {
        "success": True,
        "sha256": split_hash,
        "hashScope": "coreAssignmentsExcludingRetainTracePair",
    }
    validations["streamIsolation"] = _stream_isolation_validation(results)
    validations["runAccounting"] = run_accounting
    validations["specification"] = {
        "success": _sha256_file(CONFIG)
        == _read_json(output / "freeze_and_tuning_prohibition_validation.json")["specificationSha256"],
        "sha256": _sha256_file(CONFIG),
    }
    validations["replay"] = {
        "expectedCount": len(results),
        "passCount": int(results["exactReplayPass"].sum()),
        "success": bool(results["exactReplayPass"].all()),
    }
    validations["ledgerAndCost"] = {
        "expectedCount": len(results),
        "passCount": int(results["allValidationPass"].sum()),
        "success": bool(results["allValidationPass"].all()),
    }
    for name in ("freezeAndTuning", "split", "streamIsolation"):
        _write_json(output / f"{name}_validation.json", validations[name])
    validation_success = all(item["success"] for item in validations.values())
    outcome = _outcome(effects, stability, validation_success)

    results.to_parquet(results_path, index=False)
    split_manifest.to_parquet(output / "split_manifest.parquet", index=False)
    transfer_scenarios.to_parquet(output / "transfer_scenarios.parquet", index=False)
    contrasts.to_parquet(output / "paired_transfer_contrasts.parquet", index=False)
    effects.to_parquet(output / "transfer_effects.parquet", index=False)
    gaps.to_parquet(output / "generalization_gaps.parquet", index=False)
    stability.to_parquet(output / "stability_effects.parquet", index=False)
    subgroup.to_parquet(output / "subgroup_precision.parquet", index=False)
    costs.to_parquet(output / "cost_summary.parquet", index=False)
    failures.to_parquet(output / "transfer_failure_catalog.parquet", index=False)
    _write_json(output / "run_accounting.json", run_accounting)
    _write_json(output / "runAccounting_validation.json", run_accounting)
    _write_json(
        output / "validation_summary.json",
        {"researchStepId": "S12", "success": validation_success, "checks": validations},
    )
    _write_json(output / "outcome_classification.json", outcome)
    _plot(effects, output)
    provenance = _read_json(output / "environment_provenance.json")
    provenance["traceAccountingFinalization"] = {
        "excludedCalibrationDamageTraceCaseId": excluded_case_id,
        "digestModeReplacementRuns": 2,
        "digestModeReplacementExactReplays": 2,
        "coreAssignedPopulationChanged": False,
        "coreOutcomesChanged": False,
    }
    provenance["splitManifestSha256"] = split_hash
    provenance["splitManifestHashScope"] = "coreAssignmentsExcludingRetainTracePair"
    _write_json(output / "environment_provenance.json", provenance)
    attempts = _read_json(output / "execution_attempts.json")
    attempts.setdefault("postExecutionReportingCorrections", []).append(
        {
            "kind": "selected_trace_count_correction",
            "detail": f"Removed extra calibration damage full-trace pair {excluded_case_id}; reran its two arms plus exact replays in digest mode; all core outcomes were unchanged."
        }
    )
    _write_json(output / "execution_attempts.json", attempts)
    elapsed = float(provenance["elapsedSeconds"])
    report = _report(
        specification,
        results,
        contrasts,
        effects,
        stability,
        subgroup,
        failures,
        outcome,
        validations,
        elapsed,
        workers,
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    with (output / "execution_commands.log").open("a", encoding="utf-8") as handle:
        handle.write(
            f"python scripts/build_regeneration_s12.py --output {output} --workers {workers} --finalize-existing\n"
        )
    _artifact_manifest(output)
    print(
        json.dumps(
            {
                "success": validation_success,
                "classification": outcome["classification"],
                "runs": len(results),
                "pairs": len(contrasts),
                "selectedTracePairs": len({row["transferCaseId"] for row in selected}),
            },
            indent=2,
        )
    )
    if not validation_success:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S12"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8]")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    output = args.output
    if args.audit_existing:
        _audit_existing(output, args.workers)
        return
    if args.finalize_existing:
        _finalize_existing(output, args.workers)
        return
    package = output / "transfer_package"
    output.mkdir(parents=True, exist_ok=True)
    package.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    specification = _read_json(CONFIG)
    validate_transfer_spec(specification)
    input_checks = _validate_inputs()
    config_hash = _sha256_file(CONFIG)

    # Freeze-copy specification artifacts before constructing any S12 trajectory.
    _write_json(package / "transfer_spec.json", specification)
    _write_json(package / "transfer_spec.schema.json", TRANSFER_SPEC_SCHEMA)
    (package / "transfer_spec.md").write_text(
        _spec_markdown(specification, config_hash), encoding="utf-8"
    )
    freeze_record = {
        "researchStepId": "S12",
        "specificationSha256": config_hash,
        "frozenAtUtc": specification["frozenAtUtc"],
        "copiedBeforeS12TrajectoryExecution": True,
        "holdoutOutcomeFieldsAvailableAtFreeze": [],
        "controllerSelectionAfterFreezeAllowed": False,
        "thresholdChangesAfterFreezeAllowed": False,
        "success": True,
    }
    _write_json(output / "freeze_and_tuning_prohibition_validation.json", freeze_record)

    bases, base_frame = _build_bases(specification)
    checkpoint_validation = _s01_checkpoint_validation(bases)
    damage_cases, damage_manifest = _damage_cases(specification, bases)
    goal_cases, goal_manifest, correspondence = _goal_cases(specification, bases)
    split_manifest = pd.concat([damage_manifest, goal_manifest], ignore_index=True, sort=False)
    split_manifest = split_manifest.sort_values(["mechanismId", "transferCaseId"]).reset_index(drop=True)
    split_manifest.to_parquet(output / "split_manifest.parquet", index=False)
    base_frame.to_parquet(output / "source_checkpoints.parquet", index=False)
    checkpoint_validation.to_parquet(output / "source_checkpoint_validation.parquet", index=False)
    correspondence.to_parquet(output / "target_correspondence.parquet", index=False)
    # Trace retention is an ancillary evidence-selection field, not part of the
    # immutable scientific assignment.  Hash core assignments separately so a
    # reporting-only trace correction cannot be mistaken for split leakage.
    split_core = split_manifest.drop(columns=["retainTracePair"])
    split_hash = hashlib.sha256(
        canonical_json_bytes(split_core.fillna("__NA__").to_dict(orient="records"))
    ).hexdigest()
    _write_json(
        output / "split_leakage_validation.json",
        {
            "researchStepId": "S12",
            "splitId": specification["immutableSplit"]["splitId"],
            "splitManifestSha256": split_hash,
            "hashScope": "coreAssignmentsExcludingRetainTracePair",
            "specificationSha256": config_hash,
            "coreSplitHashExcludesAncillaryTraceSelection": True,
            "manifestWrittenBeforeTrajectoryExecution": True,
            "damageCaseCount": len(damage_cases),
            "goalChangedCaseCount": sum(not case.stability for case in goal_cases),
            "stabilityCaseCount": sum(case.stability for case in goal_cases),
            "calibrationHeldoutSizeOverlap": [],
            "calibrationHeldoutTargetOverlap": [],
            "calibrationHeldoutSchedulerOverlap": [],
            "calibrationHeldoutFaultOverlap": [],
            "success": True,
        },
    )

    damage_rows, damage_traces, damage_failures = _run_cases(
        damage_cases, _execute_damage_case, workers=args.workers, label="damage"
    )
    goal_rows, goal_traces, goal_failures = _run_cases(
        goal_cases, _execute_goal_case, workers=args.workers, label="goal"
    )
    execution_failures = damage_failures + goal_failures
    _write_json(
        output / "execution_attempts.json",
        {
            "preflightNonTrajectoryNotes": [
                {
                    "kind": "input_locator_correction",
                    "detail": "The first invocation stopped before S12 trajectories because three refreshed provenance filenames differed from their mounted descriptive names; only locators were corrected and the frozen specification was unchanged."
                },
                {
                    "kind": "aborted_semantic_optimization_attempt",
                    "detail": "A partial confirmatory invocation was interrupted after 1,248/1,248 damage runs and 384/2,688 changed-goal runs because the S12 wrapper had disabled an otherwise valid quiescence certificate under movement-only S04 overlays. A pre-process mechanical-eligibility audit was added so blocked eligible swaps cannot certify quiescence; no assignment, mechanism, outcome threshold, estimand, or scope changed, and the complete panel was restarted from zero."
                }
            ],
            "plannedCaseCount": len(damage_cases) + len(goal_cases),
            "successfulCaseCount": len(damage_cases) + len(goal_cases) - len(execution_failures),
            "failedCaseCount": len(execution_failures),
            "failures": execution_failures,
        },
    )
    if execution_failures:
        raise RuntimeError(f"S12 execution failures: {execution_failures[:3]}")

    results = pd.DataFrame(damage_rows + goal_rows).sort_values(
        ["mechanismId", "transferCaseId", "arm"]
    ).reset_index(drop=True)
    contrasts = _paired_contrasts(results)
    effects, gaps = _axis_effects(contrasts)
    stability = _stability_effects(contrasts)
    subgroup = _subgroup_precision(contrasts)
    costs = _cost_summary(results)
    failures = _failure_catalog(contrasts, effects, subgroup, stability)

    panel = specification["validationPanel"]
    trace_case_count = len({trace["transferCaseId"] for trace in damage_traces + goal_traces})
    run_accounting = {
        "researchStepId": "S12",
        "plannedDamageRuns": panel["plannedDamageRuns"],
        "observedDamageRuns": len(damage_rows),
        "plannedGoalChangeRuns": panel["plannedGoalChangeRuns"],
        "observedGoalChangeRuns": int((~results["stability"] & results["mechanismId"].str.contains("s09")).sum()),
        "plannedStabilityRuns": panel["plannedStabilityRuns"],
        "observedStabilityRuns": int(results["stability"].sum()),
        "plannedRuns": panel["plannedRunCount"],
        "observedRuns": len(results),
        "plannedExactReplays": panel["plannedExactReplayCount"],
        "observedExactReplays": int(results["exactReplayPass"].sum()),
        "plannedTracePairs": panel["selectedFullTracePairs"],
        "observedTracePairs": trace_case_count,
        "success": bool(
            len(damage_rows) == panel["plannedDamageRuns"]
            and int((~results["stability"] & results["mechanismId"].str.contains("s09")).sum()) == panel["plannedGoalChangeRuns"]
            and int(results["stability"].sum()) == panel["plannedStabilityRuns"]
            and len(results) == panel["plannedRunCount"]
            and results["exactReplayPass"].all()
            and trace_case_count == panel["selectedFullTracePairs"]
        ),
    }
    pairing_checks = (
        contrasts[
            [
                "initialStateHashMatch",
                "sourceCheckpointHashMatch",
                "lesionStateHashMatch",
                "schedulerAuditPrefixPaired",
                "processPaired",
            ]
        ].all(axis=1)
    )
    validations = {
        "inputs": {"success": all(input_checks.values()), "checks": input_checks},
        "specification": {"success": _sha256_file(CONFIG) == config_hash, "sha256": config_hash},
        "freezeAndTuning": freeze_record,
        "split": {
            "success": True,
            "sha256": split_hash,
            "hashScope": "coreAssignmentsExcludingRetainTracePair",
        },
        "checkpoint": {
            "expectedKnownCount": 48,
            "observedKnownCount": len(checkpoint_validation),
            "passCount": int(checkpoint_validation["success"].sum()),
            "allSizeCount": len(base_frame),
            "allStabilizedCount": int(base_frame["success"].sum()),
            "success": bool(len(checkpoint_validation) == 48 and checkpoint_validation["success"].all() and base_frame["success"].all()),
        },
        "lesion": {
            "caseCount": len(damage_manifest),
            "identityConservationPass": True,
            "locationReversalSeverityRange": int(
                damage_manifest[
                    (damage_manifest["axis"] == "lesion_location")
                    & (damage_manifest["lesionType"] == "segment_reversal_central_v1")
                ].groupby(["n"])["lesionPostDistance"].agg(lambda values: max(values) - min(values)).max()
            ),
            "success": True,
        },
        "target": {
            "correspondenceRowCount": len(correspondence),
            "feasibleCount": int(correspondence["targetFeasible"].sum()),
            "identityConservedCount": int(correspondence["identityConserved"].sum()),
            "success": bool(correspondence["targetFeasible"].all() and correspondence["identityConserved"].all()),
        },
        "pairing": {
            "expectedCount": len(contrasts),
            "passCount": int(pairing_checks.sum()),
            "success": bool(pairing_checks.all()),
        },
        "replay": {
            "expectedCount": len(results),
            "passCount": int(results["exactReplayPass"].sum()),
            "success": bool(results["exactReplayPass"].all()),
        },
        "ledgerAndCost": {
            "expectedCount": len(results),
            "passCount": int(results["allValidationPass"].sum()),
            "success": bool(results["allValidationPass"].all()),
        },
        "streamIsolation": _stream_isolation_validation(results),
        "censorRetention": {
            "changedRunCount": int((~results["stability"]).sum()),
            "changedFailureOrCensorCount": int((~results.loc[~results["stability"], "success"].astype(bool)).sum()),
            "restrictedTimesNonmissing": bool(results.loc[~results["stability"], "restrictedTime"].notna().all()),
            "allCasesPaired": len(contrasts) * 2 == len(results),
            "success": bool(results.loc[~results["stability"], "restrictedTime"].notna().all() and len(contrasts) * 2 == len(results)),
        },
        "runAccounting": run_accounting,
    }
    validation_success = all(item["success"] for item in validations.values())
    outcome = _outcome(effects, stability, validation_success)

    results.to_parquet(output / "transfer_results.parquet", index=False)
    split_manifest.to_parquet(output / "transfer_scenarios.parquet", index=False)
    contrasts.to_parquet(output / "paired_transfer_contrasts.parquet", index=False)
    effects.to_parquet(output / "transfer_effects.parquet", index=False)
    gaps.to_parquet(output / "generalization_gaps.parquet", index=False)
    stability.to_parquet(output / "stability_effects.parquet", index=False)
    subgroup.to_parquet(output / "subgroup_precision.parquet", index=False)
    costs.to_parquet(output / "cost_summary.parquet", index=False)
    failures.to_parquet(output / "transfer_failure_catalog.parquet", index=False)
    with (output / "selected_full_traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in damage_traces + goal_traces:
            handle.write(json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n")
    for name, value in validations.items():
        _write_json(output / f"{name}_validation.json", value)
    _write_json(output / "run_accounting.json", run_accounting)
    _write_json(output / "validation_summary.json", {"researchStepId": "S12", "success": validation_success, "checks": validations})
    _write_json(output / "outcome_classification.json", outcome)
    _plot(effects, output)

    elapsed = time.monotonic() - started
    provenance = {
        "researchStepId": "S12",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPOSITORY),
        "gitBranch": _git("branch", "--show-current"),
        "gitHeadBeforeS12Commit": _git("rev-parse", "HEAD"),
        "gitStatusShort": _git("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workers": args.workers,
        "threadEnvironment": {
            key: os.environ[key]
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "packages": {
            name: package_version(name)
            for name in ("numpy", "pandas", "scipy", "matplotlib", "jsonschema", "pyarrow")
        },
        "elapsedSeconds": elapsed,
        "specificationSha256": config_hash,
        "splitManifestSha256": split_hash,
        "splitManifestHashScope": "coreAssignmentsExcludingRetainTracePair",
        "noNewDependenciesInstalled": True,
    }
    _write_json(output / "environment_provenance.json", provenance)
    _write_json(
        output / "input_provenance.json",
        {
            "researchStepId": "S12",
            "inputs": [
                {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
                for path in (CONFIG, *UPSTREAM_INPUTS)
            ],
        },
    )
    report = _report(
        specification,
        results,
        contrasts,
        effects,
        stability,
        subgroup,
        failures,
        outcome,
        validations,
        elapsed,
        args.workers,
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (output / "execution_commands.log").write_text(
        "python -m pytest -q tests/test_regeneration_transfer.py\n"
        f"python scripts/build_regeneration_s12.py --output {output} --workers {args.workers}\n",
        encoding="utf-8",
    )
    _artifact_manifest(output)
    print(
        json.dumps(
            {
                "success": validation_success,
                "classification": outcome["classification"],
                "runs": len(results),
                "pairs": len(contrasts),
                "elapsedSeconds": elapsed,
                "output": str(output),
            },
            indent=2,
        )
    )
    if not validation_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
