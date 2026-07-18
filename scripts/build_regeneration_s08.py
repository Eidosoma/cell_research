#!/usr/bin/env python3
"""Build and validate E05 S08 engineered policy-plasticity evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, t


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from reference_simulator.model import canonical_json_bytes, sha256_json  # noqa: E402
from scripts.build_regeneration_s04 import (  # noqa: E402
    ACTIVE_PROFILES,
    DynamicJob,
    _reconstruct_jobs,
)
from src.regeneration.policy_plasticity import (  # noqa: E402
    BENCHMARK_VERSION,
    PLASTICITY_RUN_SCHEMA_VERSION,
    PLASTICITY_SPEC_SCHEMA,
    RECOVERY_DONOR_VARIANT,
    AdaptiveObservationGateway,
    Durability,
    LocalFailureBank,
    PlasticityArm,
    PlasticityContract,
    PlasticityMechanism,
    PlasticityVariant,
    VARIANT_PROFILE,
    exact_replay_plasticity,
    plasticity_case_id,
    run_plasticity_phase,
    validate_plasticity_spec,
)


CONFIG = REPOSITORY / "configs/regeneration/s08_policy_plasticity.json"
S04_CONFIG = REPOSITORY / "configs/regeneration/s04_dynamic_faults.json"
S07_RESULTS = Path("/artifacts/research_steps/S07/local_memory.parquet")
S07_PRIMARY = Path("/artifacts/research_steps/S07/primary_completion_tests.parquet")
ATTACHMENT_SIDECAR = Path(
    "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"
)
INPUTS: tuple[Path, ...] = (
    Path("/workspace/AGENTS.md"),
    Path("/workspace/FULL_PLAN.md"),
    Path("/workspace/RESEARCH_PLAN.md"),
    Path("/workspace/PREVIOUS_ARTIFACTS.md"),
    Path("/workspace/PREVIOUS_ARTIFACTS.json"),
    Path("/workspace/CAPABILITIES.md"),
    Path("/workspace/CAPABILITY_AVAILABILITY.json"),
    Path("/workspace/input-attachments/MANIFEST.json"),
    ATTACHMENT_SIDECAR,
    *tuple(
        item
        for step in range(1, 8)
        for item in (
            Path(f"/artifacts/research_steps/S{step:02d}/research_step_full_results.md"),
            Path(f"/artifacts/research_steps/S{step:02d}/validation_summary.json"),
        )
    ),
    Path("/artifacts/research_steps/S01/task_spec.md"),
    Path("/artifacts/research_steps/S01/task_spec.json"),
    Path("/artifacts/research_steps/S02/timing_spec.md"),
    Path("/artifacts/research_steps/S02/timing_spec.json"),
    Path("/artifacts/research_steps/S02/preinjury_states.parquet"),
    Path("/artifacts/research_steps/S03/lesion_library/lesion_spec.json"),
    Path("/artifacts/research_steps/S03/lesion_library/operator_fixtures.parquet"),
    Path("/artifacts/research_steps/S04/dynamic_fault_package/dynamic_fault_spec.md"),
    Path("/artifacts/research_steps/S04/dynamic_fault_package/dynamic_fault_spec.json"),
    Path("/artifacts/research_steps/S05/nudge_recovery_package/nudge_recovery_spec.md"),
    Path("/artifacts/research_steps/S05/nudge_recovery_package/nudge_recovery_spec.json"),
    Path("/artifacts/research_steps/S06/assisted_rescue_package/assisted_rescue_spec.md"),
    Path("/artifacts/research_steps/S06/assisted_rescue_package/assisted_rescue_spec.json"),
    Path("/artifacts/research_steps/S07/memory_variants/local_memory_spec.md"),
    Path("/artifacts/research_steps/S07/memory_variants/local_memory_spec.json"),
    S07_RESULTS,
    S07_PRIMARY,
    Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
    Path("/previous-artifacts/E01/specification/transition_spec.md"),
    Path("/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json"),
    Path("/previous-artifacts/E02/research_steps/S02/action_interface_spec.md"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/scheduler_contract.md"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/opportunity_ledger_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/rng_consumption_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/fault_semantics_contract.md"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/information_boundary_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S08/semantic_random_stream_specification.json"),
    Path("/previous-artifacts/E02/research_steps/S08/stream_name_isolation_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S09/ledger_identity_validation.json"),
)


def _load_json(path: Path) -> dict[str, Any]:
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


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def _validate_inputs() -> dict[str, bool]:
    missing = [str(path) for path in (CONFIG, S04_CONFIG, *INPUTS) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required S01-S07/E01/E02 inputs: {missing}")
    gates = {
        f"s{step:02d}": bool(
            _load_json(Path(f"/artifacts/research_steps/S{step:02d}/validation_summary.json"))[
                "success"
            ]
        )
        for step in range(1, 8)
    }
    gates["e01"] = bool(
        _load_json(
            Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json")
        )["validationSuccess"]
    )
    gates["e02"] = bool(
        _load_json(
            Path(
                "/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json"
            )
        )["smokeValidation"]["success"]
    )
    primary = pd.read_parquet(S07_PRIMARY)
    gates["s07MatchedNull"] = bool(
        len(primary) == 5
        and (~primary["rejectAtFamilywise0_05"].astype(bool)).all()
        and (primary["holmAdjustedPValue"] == 1.0).all()
    )
    donors = pd.read_parquet(S07_RESULTS)
    donors = donors[
        (donors["arm"] == "active_local_memory")
        & (donors["variantId"] == RECOVERY_DONOR_VARIANT)
    ]
    gates["s07ScheduleComplete"] = bool(
        len(donors) == 384
        and donors["recoveryObserved"].astype(bool).all()
        and donors["recoveryDuration"].notna().all()
    )
    if not all(gates.values()):
        raise RuntimeError(f"an inherited validation/null gate failed: {gates}")
    return gates


def _base_jobs(reconstructed: Mapping[str, Any]) -> list[DynamicJob]:
    anchor_profile = ACTIVE_PROFILES[0]
    jobs = [
        job
        for job in reconstructed["jobs"]
        if job.arm == "active_dynamic_process" and job.active_profile == anchor_profile
    ]
    unique = {(job.s01_pairing_block_id, job.timing_condition_id): job for job in jobs}
    answer = sorted(
        unique.values(),
        key=lambda job: (
            job.n,
            job.policy,
            job.direction,
            job.replicate,
            job.timing_condition_id,
        ),
    )
    if len(answer) != 384:
        raise RuntimeError(f"expected 384 inherited bases, found {len(answer)}")
    return answer


def _recovery_schedules() -> dict[tuple[str, str], int]:
    rows = pd.read_parquet(S07_RESULTS)
    rows = rows[
        (rows["arm"] == "active_local_memory")
        & (rows["variantId"] == RECOVERY_DONOR_VARIANT)
    ]
    schedules = {
        (row.s01PairingBlockId, row.timingConditionId): int(row.recoveryDuration)
        for row in rows.itertuples()
    }
    if len(schedules) != 384 or min(schedules.values()) < 1:
        raise RuntimeError("S07 exact same-case recovery schedule is incomplete")
    return schedules


@dataclass(frozen=True, slots=True)
class PlasticJob:
    base: DynamicJob
    phase: str
    arm: PlasticityArm
    variant: PlasticityVariant | None
    assigned_duration: int | None
    retain_trace: bool = False


def _case_id(job: PlasticJob) -> str:
    anchor = job.base.anchor_lesion_state_hash if job.phase == "injury" else None
    return plasticity_case_id(
        job.base.s01_pairing_block_id,
        job.base.timing_condition_id,
        job.phase,
        anchor,
    )


def _trace_job(job: PlasticJob) -> bool:
    if job.arm != PlasticityArm.ACTIVE or job.variant is None:
        return False
    if job.phase == "injury":
        return bool(
            job.base.n == 20
            and job.base.policy == "Bubble"
            and job.base.direction == "ascending"
            and job.base.replicate == 2
            and job.base.timing_condition_id == "initialization"
        )
    return bool(
        job.base.n == 20
        and job.base.policy == "Bubble"
        and job.base.direction == "ascending"
        and job.base.replicate == 0
        and job.base.timing_condition_id == "post_completion"
    )


def _jobs(
    bases: list[DynamicJob], schedules: Mapping[tuple[str, str], int]
) -> list[PlasticJob]:
    jobs: list[PlasticJob] = []
    for base in bases:
        duration = schedules[(base.s01_pairing_block_id, base.timing_condition_id)]
        for variant in PlasticityVariant:
            for arm in (PlasticityArm.ACTIVE, PlasticityArm.SHAM):
                candidate = PlasticJob(base, "injury", arm, variant, duration)
                jobs.append(replace(candidate, retain_trace=_trace_job(candidate)))
        jobs.append(
            PlasticJob(base, "injury", PlasticityArm.REFERENCE, None, duration)
        )
    stability_bases = [base for base in bases if base.timing_condition_id == "post_completion"]
    if len(stability_bases) != 48:
        raise RuntimeError("S08 stability panel requires all 48 post-completion bases")
    for base in stability_bases:
        for variant in PlasticityVariant:
            for arm in (PlasticityArm.ACTIVE, PlasticityArm.SHAM):
                candidate = PlasticJob(base, "stability", arm, variant, None)
                jobs.append(replace(candidate, retain_trace=_trace_job(candidate)))
        jobs.append(PlasticJob(base, "stability", PlasticityArm.REFERENCE, None, None))
    if len(jobs) != 7344:
        raise RuntimeError(f"expected 7344 S08 jobs, found {len(jobs)}")
    return jobs


def _execute_job(job: PlasticJob) -> dict[str, Any]:
    contract = PlasticityContract(
        job.arm,
        job.variant,
        job.phase,
        assigned_recovery_duration=job.assigned_duration,
    )
    if job.phase == "injury":
        occupancy = job.base.post_anchor_occupancy
        anchor_hash = job.base.anchor_lesion_state_hash
        selected = job.base.selected_identity
        phase_budget = job.base.recovery_budget
    else:
        occupancy = job.base.checkpoint.occupancy
        anchor_hash = None
        selected = None
        phase_budget = 20 * job.base.n
    run = run_plasticity_phase(
        job.base.scenario,
        job.base.checkpoint,
        occupancy=occupancy,
        anchor_lesion_state_hash=anchor_hash,
        selected_identity=selected,
        contract=contract,
        phase_budget=phase_budget,
        retain_trace=job.retain_trace,
    )
    exact_replay_plasticity(
        run,
        job.base.scenario,
        job.base.checkpoint,
        occupancy,
        phase_budget,
    )
    case_id = _case_id(job)
    variant_id = None if job.variant is None else job.variant.value
    mechanism = None if job.variant is None else VARIANT_PROFILE[job.variant][0].value
    durability = None if job.variant is None else VARIANT_PROFILE[job.variant][1].value
    run_id = "e05pr8:" + sha256_json(
        {
            "plasticityCaseId": case_id,
            "phase": job.phase,
            "arm": job.arm.value,
            "variantId": variant_id,
        }
    )
    summary = dict(run.summary)
    process = dict(run.process_final_state)
    ledger = dict(run.process_ledger)
    split = "calibration" if job.base.replicate in {0, 1} else "confirmatory"
    row = {
        "schemaVersion": "e05.s08.policy-plasticity-result.v1",
        "benchmarkVersion": BENCHMARK_VERSION,
        "plasticityRunId": run_id,
        "plasticityCaseId": case_id,
        "phase": job.phase,
        "arm": job.arm.value,
        "variantId": variant_id,
        "mechanismId": mechanism,
        "durability": durability,
        "split": split,
        "s01PairingBlockId": job.base.s01_pairing_block_id,
        "timingConditionId": job.base.timing_condition_id,
        "clock": job.base.clock,
        "nominalFraction": job.base.nominal_fraction,
        "n": job.base.n,
        "policy": job.base.policy,
        "direction": job.base.direction,
        "replicateOrdinal": job.base.replicate,
        "sourceScenarioId": job.base.scenario.scenario_id,
        "sourceCheckpointHash": job.base.checkpoint.state_hash,
        "anchorLesionStateHash": anchor_hash,
        "startOccupancySha256": _digest(occupancy),
        "selectedIdentityId": selected,
        "globalStartEventIndex": run.start_event_index,
        "globalEndEventIndex": run.end_event_index,
        "phaseBudget": phase_budget,
        "stopReason": summary["stopReason"],
        "completed": summary["completed"],
        "phaseActivationCount": summary["phaseActivationCount"],
        "initialDistance": summary["initialDistance"],
        "finalDistance": summary["finalDistance"],
        "maximumDistance": summary["maximumDistance"],
        "distanceAuc": summary["distanceAuc"],
        "anyTargetDeparture": summary["anyTargetDeparture"],
        "finalTargetRetained": summary["finalTargetRetained"],
        "assignedRecoveryDuration": job.assigned_duration,
        "recoveryObserved": summary["recoveryObserved"],
        "recoveryDuration": summary["recoveryDuration"],
        "recoveryCensored": summary["recoveryCensored"],
        "transitionObserved": summary["transitionObserved"],
        "transitionCensored": summary["transitionCensored"],
        "reversionObserved": summary["reversionObserved"],
        "reversionCensored": summary["reversionCensored"],
        "modeActorId": summary["modeActorId"],
        "modeOpportunities": summary["modeOpportunities"],
        "transitionEventIndex": process["transitionEventIndex"],
        "reversionEventIndex": process["reversionEventIndex"],
        "transitionActionBudgetInitial": process["transitionActionBudgetInitial"],
        "transitionActionBudgetRemaining": process["transitionActionBudgetRemaining"],
        "storageBitsPerIdentity": summary["storageBitsPerIdentity"],
        "totalStorageBits": summary["totalStorageBits"],
        "failureCountersSha256": process["failureCountersSha256"],
        **ledger,
        "nativeLedgerDeltaJson": json.dumps(
            summary["ledgerDelta"], sort_keys=True, separators=(",", ":")
        ),
        "streamCounterDeltaJson": json.dumps(
            summary["streamCounterDelta"], sort_keys=True, separators=(",", ":")
        ),
        "eventDigest": run.event_digest,
        "processAuditDigest": run.process_audit_digest,
        "processAuditCount": run.process_audit_count,
        "processTransitionsSha256": _digest(run.process_transitions),
        "initialStateHash": run.initial_state_hash,
        "finalStateHash": run.final_state_hash,
        "finalStateSha256": _digest(run.final_state),
        "allOpportunityValidationPass": all(run.opportunity_validation.values()),
        "opportunityValidationJson": json.dumps(
            run.opportunity_validation, sort_keys=True, separators=(",", ":")
        ),
        "traceMode": summary["traceMode"],
        "retainedEventCount": summary["retainedEventCount"],
        "exactReplayPass": True,
    }
    scenario_row = {
        "schemaVersion": "e05.s08.policy-plasticity-scenario.v1",
        "plasticityRunId": run_id,
        "plasticityCaseId": case_id,
        "phase": job.phase,
        "arm": job.arm.value,
        "variantId": variant_id,
        "mechanismId": mechanism,
        "durability": durability,
        "split": split,
        "s01PairingBlockId": job.base.s01_pairing_block_id,
        "timingConditionId": job.base.timing_condition_id,
        "n": job.base.n,
        "policy": job.base.policy,
        "direction": job.base.direction,
        "replicateOrdinal": job.base.replicate,
        "sourceScenarioId": job.base.scenario.scenario_id,
        "sourceCheckpointHash": job.base.checkpoint.state_hash,
        "anchorLesionStateHash": anchor_hash,
        "startOccupancySha256": _digest(occupancy),
        "selectedIdentityId": selected,
        "assignedRecoveryDuration": job.assigned_duration,
        "phaseBudget": phase_budget,
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "nativeInformationPermission": "policy_native_local",
        "continuation": "skip_and_continue",
        "retry": "no_retry",
        "runtimeStreamsJson": "[]",
        "pairingStatus": "exact_prefix_through_transition_then_address_valid_where_consumption_matches",
    }
    transitions = [
        {
            "plasticityRunId": run_id,
            "plasticityCaseId": case_id,
            "phase": job.phase,
            "arm": job.arm.value,
            "variantId": variant_id,
            "transitionOrdinal": ordinal,
            **dict(item),
        }
        for ordinal, item in enumerate(run.process_transitions)
    ]
    trace = None
    if job.retain_trace:
        trace = {
            "plasticityRunId": run_id,
            "plasticityCaseId": case_id,
            "phase": job.phase,
            "arm": job.arm.value,
            "variantId": variant_id,
            "events": list(run.events),
            "processAudits": list(run.retained_process_audits),
            "processTransitions": list(run.process_transitions),
            "opportunityValidation": dict(run.opportunity_validation),
        }
    return {
        "result": row,
        "scenario": scenario_row,
        "transitions": transitions,
        "trace": trace,
    }


def _run_jobs(jobs: list[PlasticJob], workers: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_execute_job, job): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                value = future.result()
                results.append(value["result"])
                scenarios.append(value["scenario"])
                transitions.extend(value["transitions"])
                if value["trace"] is not None:
                    traces.append(value["trace"])
            except Exception as exc:  # pragma: no cover
                failures.append(
                    {
                        "caseId": _case_id(job),
                        "phase": job.phase,
                        "arm": job.arm.value,
                        "variantId": None
                        if job.variant is None
                        else job.variant.value,
                        "error": repr(exc),
                    }
                )
            completed += 1
            if completed % 100 == 0 or completed == len(jobs):
                print(f"S08 execution: {completed}/{len(jobs)} jobs returned", flush=True)
    return {
        "results": results,
        "scenarios": scenarios,
        "transitions": transitions,
        "traces": traces,
        "failures": failures,
    }


def _paired_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["plasticityCaseId", "phase", "variantId"]
    active = results[results["arm"] == PlasticityArm.ACTIVE.value].copy()
    sham = results[results["arm"] == PlasticityArm.SHAM.value].copy()
    paired = active.merge(sham, on=keys, suffixes=("Active", "Sham"), validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for row in paired.itertuples():
        rows.append(
            {
                "schemaVersion": "e05.s08.paired-plasticity-contrast.v1",
                "plasticityCaseId": row.plasticityCaseId,
                "phase": row.phase,
                "variantId": row.variantId,
                "mechanismId": row.mechanismIdActive,
                "durability": row.durabilityActive,
                "split": row.splitActive,
                "s01PairingBlockId": row.s01PairingBlockIdActive,
                "timingConditionId": row.timingConditionIdActive,
                "n": row.nActive,
                "policy": row.policyActive,
                "direction": row.directionActive,
                "replicateOrdinal": row.replicateOrdinalActive,
                "activeRunId": row.plasticityRunIdActive,
                "shamRunId": row.plasticityRunIdSham,
                "activeCompleted": bool(row.completedActive),
                "shamCompleted": bool(row.completedSham),
                "completionDifference": int(row.completedActive)
                - int(row.completedSham),
                "activeTargetDeparture": bool(row.anyTargetDepartureActive),
                "shamTargetDeparture": bool(row.anyTargetDepartureSham),
                "targetDepartureDifference": int(row.anyTargetDepartureActive)
                - int(row.anyTargetDepartureSham),
                "finalDistanceDifference": int(row.finalDistanceActive)
                - int(row.finalDistanceSham),
                "maximumDistanceDifference": int(row.maximumDistanceActive)
                - int(row.maximumDistanceSham),
                "distanceAucDifference": int(row.distanceAucActive)
                - int(row.distanceAucSham),
                "phaseActivationDifference": int(row.phaseActivationCountActive)
                - int(row.phaseActivationCountSham),
                "abstractEnergyDifference": int(row.totalAbstractEnergyActive)
                - int(row.totalAbstractEnergySham),
                "adaptiveSensingReadDifference": int(row.adaptiveSensingReadsActive)
                - int(row.adaptiveSensingReadsSham),
                "transitionObservedActive": bool(row.transitionObservedActive),
                "transitionObservedSham": bool(row.transitionObservedSham),
                "transitionEventIndexActive": row.transitionEventIndexActive,
                "transitionEventIndexSham": row.transitionEventIndexSham,
                "modeActorIdActive": row.modeActorIdActive,
                "modeActorIdSham": row.modeActorIdSham,
                "reversionObservedActive": bool(row.reversionObservedActive),
                "reversionObservedSham": bool(row.reversionObservedSham),
                "activeRecoveryObserved": bool(row.recoveryObservedActive),
                "shamRecoveryObserved": bool(row.recoveryObservedSham),
                "assignedRecoveryDuration": row.assignedRecoveryDurationActive,
                "activeRecoveryDuration": row.recoveryDurationActive,
                "shamRecoveryDuration": row.recoveryDurationSham,
            }
        )
    return pd.DataFrame(rows)


def _reference_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    reference = results[results["arm"] == PlasticityArm.REFERENCE.value].copy()
    keys = ["plasticityCaseId", "phase"]
    rows: list[dict[str, Any]] = []
    for arm in (PlasticityArm.ACTIVE, PlasticityArm.SHAM):
        treatment = results[results["arm"] == arm.value]
        paired = treatment.merge(
            reference,
            on=keys,
            suffixes=("Treatment", "Reference"),
            validate="many_to_one",
        )
        for row in paired.itertuples():
            rows.append(
                {
                    "schemaVersion": "e05.s08.reference-contrast.v1",
                    "plasticityCaseId": row.plasticityCaseId,
                    "phase": row.phase,
                    "variantId": row.variantIdTreatment,
                    "treatmentArm": arm.value,
                    "split": row.splitTreatment,
                    "completionDifference": int(row.completedTreatment)
                    - int(row.completedReference),
                    "targetDepartureDifference": int(row.anyTargetDepartureTreatment)
                    - int(row.anyTargetDepartureReference),
                    "phaseActivationDifference": int(row.phaseActivationCountTreatment)
                    - int(row.phaseActivationCountReference),
                    "finalDistanceDifference": int(row.finalDistanceTreatment)
                    - int(row.finalDistanceReference),
                    "distanceAucDifference": int(row.distanceAucTreatment)
                    - int(row.distanceAucReference),
                    "abstractEnergyDifference": int(row.totalAbstractEnergyTreatment)
                    - int(row.totalAbstractEnergyReference),
                }
            )
    return pd.DataFrame(rows)


def _holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    array = values.astype(float).to_numpy()
    mean = float(array.mean())
    if len(array) < 2:
        return mean, mean, mean
    se = float(array.std(ddof=1) / math.sqrt(len(array)))
    bound = float(t.ppf(0.975, len(array) - 1) * se)
    return mean, mean - bound, mean + bound


def _binary_test_rows(
    contrasts: pd.DataFrame,
    *,
    family: str,
) -> pd.DataFrame:
    if family == "recovery":
        data = contrasts.query("phase == 'injury' and split == 'confirmatory'")
        active_field = "activeCompleted"
        sham_field = "shamCompleted"
        difference_field = "completionDifference"
    else:
        data = contrasts.query("phase == 'stability'")
        active_field = "activeTargetDeparture"
        sham_field = "shamTargetDeparture"
        difference_field = "targetDepartureDifference"
    rows: list[dict[str, Any]] = []
    for variant, group in data.groupby("variantId", sort=True):
        active = group[active_field].astype(bool)
        sham = group[sham_field].astype(bool)
        active_only = int((active & ~sham).sum())
        sham_only = int((~active & sham).sum())
        discordant = active_only + sham_only
        raw_p = (
            1.0
            if discordant == 0
            else float(binomtest(active_only, discordant, 0.5).pvalue)
        )
        mean, low, high = _mean_ci(group[difference_field])
        first = group.iloc[0]
        rows.append(
            {
                "schemaVersion": f"e05.s08.primary-{family}-test.v1",
                "family": family,
                "variantId": variant,
                "mechanismId": first["mechanismId"],
                "durability": first["durability"],
                "pairCount": len(group),
                "activePositiveCount": int(active.sum()),
                "shamPositiveCount": int(sham.sum()),
                "activeOnlyCount": active_only,
                "shamOnlyCount": sham_only,
                "pairedRiskDifference": mean,
                "pairedRiskDifferenceCiLow": low,
                "pairedRiskDifferenceCiHigh": high,
                "rawExactPValue": raw_p,
            }
        )
    frame = pd.DataFrame(rows)
    frame["holmAdjustedPValue"] = _holm(frame["rawExactPValue"].tolist())
    frame["rejectAtFamilywise0_05"] = frame["holmAdjustedPValue"] <= 0.05
    return frame


def _tradeoffs(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (phase, variant), group in contrasts.groupby(["phase", "variantId"], sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "schemaVersion": "e05.s08.tradeoff-summary.v1",
                "phase": phase,
                "variantId": variant,
                "mechanismId": first["mechanismId"],
                "durability": first["durability"],
                "pairCount": len(group),
                "meanCompletionDifference": float(group["completionDifference"].mean()),
                "meanTargetDepartureDifference": float(
                    group["targetDepartureDifference"].mean()
                ),
                "meanPhaseActivationDifference": float(
                    group["phaseActivationDifference"].mean()
                ),
                "meanFinalDistanceDifference": float(
                    group["finalDistanceDifference"].mean()
                ),
                "meanMaximumDistanceDifference": float(
                    group["maximumDistanceDifference"].mean()
                ),
                "meanDistanceAucDifference": float(
                    group["distanceAucDifference"].mean()
                ),
                "meanAbstractEnergyDifference": float(
                    group["abstractEnergyDifference"].mean()
                ),
                "meanAdaptiveSensingReadDifference": float(
                    group["adaptiveSensingReadDifference"].mean()
                ),
                "transitionPairAgreementRate": float(
                    (
                        group["transitionObservedActive"]
                        == group["transitionObservedSham"]
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _complexity(results: pd.DataFrame) -> pd.DataFrame:
    active = results[results["arm"] == PlasticityArm.ACTIVE.value]
    rows: list[dict[str, Any]] = []
    for variant, group in active.groupby("variantId", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "schemaVersion": "e05.s08.plasticity-complexity.v1",
                "variantId": variant,
                "mechanismId": first["mechanismId"],
                "durability": first["durability"],
                "storageBitsPerIdentity": int(group["storageBitsPerIdentity"].max()),
                "meanTotalStorageBits": float(group["totalStorageBits"].mean()),
                "maximumTotalStorageBits": int(group["totalStorageBits"].max()),
                "meanAdaptiveSensingReads": float(group["adaptiveSensingReads"].mean()),
                "meanAdaptiveComparisons": float(
                    group["adaptiveValueComparisons"].mean()
                ),
                "meanAbstractEnergy": float(group["totalAbstractEnergy"].mean()),
                "meanSuppressedOpportunities": float(
                    group["nativeOpportunitiesSuppressed"].mean()
                ),
                "maximumSensingRadius": int(group["maximumSensingRadius"].max()),
            }
        )
    return pd.DataFrame(rows)


def _pairing_validation(scenarios: pd.DataFrame) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    fields = [
        "phase",
        "s01PairingBlockId",
        "timingConditionId",
        "n",
        "policy",
        "direction",
        "replicateOrdinal",
        "sourceScenarioId",
        "sourceCheckpointHash",
        "anchorLesionStateHash",
        "startOccupancySha256",
        "selectedIdentityId",
        "assignedRecoveryDuration",
        "phaseBudget",
        "eventBudgetProfile",
        "scheduler",
        "architecture",
        "nativeInformationPermission",
        "continuation",
        "retry",
        "runtimeStreamsJson",
    ]
    paired = scenarios[scenarios["variantId"].notna()]
    for keys, group in paired.groupby(["plasticityCaseId", "variantId"], dropna=False):
        arms = set(group["arm"])
        if arms != {PlasticityArm.ACTIVE.value, PlasticityArm.SHAM.value}:
            failures.append({"keys": list(keys), "reason": "arm_set"})
            continue
        unequal = [field for field in fields if group[field].nunique(dropna=False) != 1]
        if unequal:
            failures.append({"keys": list(keys), "reason": "shared_fields", "fields": unequal})
    return {
        "schemaVersion": "e05.s08.pairing-validation.v1",
        "pairedVariantCaseCount": int(paired.groupby(["plasticityCaseId", "variantId"]).ngroups),
        "expectedPairedVariantCaseCount": 3456,
        "failureCount": len(failures),
        "failures": failures[:100],
        "success": not failures
        and paired.groupby(["plasticityCaseId", "variantId"]).ngroups == 3456,
    }


def _validate_panel(
    reconstructed: Mapping[str, Any],
    results: pd.DataFrame,
    scenarios: pd.DataFrame,
    contrasts: pd.DataFrame,
    transitions: pd.DataFrame,
    primary_recovery: pd.DataFrame,
    primary_stability: pd.DataFrame,
    failures: list[dict[str, str]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    injury_active = results.query("phase == 'injury' and arm == 'active_policy_plasticity'")
    injury_sham = results.query("phase == 'injury' and arm == 'matched_nonplastic_sham'")
    injury_reference = results.query(
        "phase == 'injury' and arm == 'nonplastic_recovery_only_reference'"
    )
    stability_active = results.query(
        "phase == 'stability' and arm == 'active_policy_plasticity'"
    )
    stability_sham = results.query(
        "phase == 'stability' and arm == 'matched_nonplastic_sham'"
    )
    stability_reference = results.query(
        "phase == 'stability' and arm == 'nonplastic_recovery_only_reference'"
    )
    paired = _pairing_validation(scenarios)
    schedule_pass = bool(
        results.query("phase == 'injury'")["assignedRecoveryDuration"].notna().all()
        and results.query("phase == 'stability'")["assignedRecoveryDuration"].isna().all()
        and results.query("phase == 'injury'")["recoveryObserved"].astype(bool).all()
        and (
            results.query("phase == 'injury'")["recoveryDuration"].astype(int)
            == results.query("phase == 'injury'")["assignedRecoveryDuration"].astype(int)
        ).all()
        and (results.query("phase == 'injury'")["suppressed::recovery"] == 1).all()
        and (results.query("phase == 'stability'")["suppressed::recovery"] == 0).all()
    )
    cost_pass = bool(
        results["allOpportunityValidationPass"].all()
        and (
            results["actionUnitsSpent"]
            == results["nativeOpportunitiesSuppressed"]
        ).all()
        and (
            results["transitionAndRecoveryEnergy"] == results["actionUnitsSpent"]
        ).all()
        and (
            results["adaptiveComputeEnergy"] == results["modeProposalOpportunities"]
        ).all()
        and (
            results["adaptiveSensingEnergy"] == results["adaptiveSensingReads"]
        ).all()
        and (
            results["totalAbstractEnergy"]
            == results["transitionAndRecoveryEnergy"]
            + results["adaptiveComputeEnergy"]
            + results["adaptiveSensingEnergy"]
        ).all()
        and (results["transitionActionBudgetRemaining"] >= 0).all()
        and (results["forgoneEligibleNativeChanges"] <= results["actionUnitsSpent"]).all()
    )
    paired_transition = contrasts
    exact_trigger_pair = bool(
        (
            paired_transition["transitionObservedActive"]
            == paired_transition["transitionObservedSham"]
        ).all()
        and (
            paired_transition.loc[
                paired_transition["transitionObservedActive"],
                "transitionEventIndexActive",
            ].astype(int)
            == paired_transition.loc[
                paired_transition["transitionObservedActive"],
                "transitionEventIndexSham",
            ].astype(int)
        ).all()
        and (
            paired_transition.loc[
                paired_transition["transitionObservedActive"], "modeActorIdActive"
            ]
            == paired_transition.loc[
                paired_transition["transitionObservedActive"], "modeActorIdSham"
            ]
        ).all()
    )
    active_sham = results[
        results["arm"].isin([PlasticityArm.ACTIVE.value, PlasticityArm.SHAM.value])
    ]
    reversible = active_sham[active_sham["durability"] == Durability.REVERSIBLE.value]
    irreversible = active_sham[
        active_sham["durability"] == Durability.IRREVERSIBLE.value
    ]
    transition_pass = bool(
        exact_trigger_pair
        and (active_sham["modeTransitions"] <= 1).all()
        and (active_sham["modeReversions"] <= 1).all()
        and (irreversible["modeReversions"] == 0).all()
        and (
            reversible.loc[reversible["reversionObserved"], "modeOpportunities"]
            == 8
        ).all()
        and (
            active_sham["transitionObserved"]
            == (active_sham["modeTransitions"] == 1)
        ).all()
        and (
            active_sham["reversionObserved"]
            == (active_sham["modeReversions"] == 1)
        ).all()
        and (results["failureCounterSaturations"] >= 0).all()
    )
    allowed_streams = {"actor_activation", "bubble_side"}
    stream_sets = [set(json.loads(item)) for item in results["streamCounterDeltaJson"]]
    stream_pass = bool(all(item <= allowed_streams for item in stream_sets))
    radius_limits = {
        PlasticityMechanism.ALGOTYPE.value: lambda group: (
            group["maximumSensingRadius"] <= group["n"] - 1
        ).all(),
        PlasticityMechanism.RADIUS.value: lambda group: (
            group["maximumSensingRadius"] <= 2
        ).all(),
        PlasticityMechanism.TARGET.value: lambda group: (
            group["maximumSensingRadius"] <= 1
        ).all(),
        PlasticityMechanism.EXPLORE.value: lambda group: (
            (group["maximumSensingRadius"] <= 1)
            & (group["adaptiveValueComparisons"] == 0)
        ).all(),
    }
    permission_pass = bool(
        all(
            check(active_sham[active_sham["mechanismId"] == mechanism])
            for mechanism, check in radius_limits.items()
        )
    )
    source = inspect.getsource(LocalFailureBank)
    forbidden_tokens = ("global_progress", "target_distance", "event_index", "occupancy")
    observability_pass = bool(
        all(token not in source for token in forbidden_tokens)
        and set(LocalFailureBank.__slots__) == {"states", "ledger"}
        and set(inspect.signature(AdaptiveObservationGateway.observe_or_propose).parameters)
        == {
            "self",
            "actor_id",
            "mechanism",
            "mode_count",
            "native_side",
            "emit",
        }
        and stream_pass
        and permission_pass
    )
    censor_pass = bool(
        len(results) == 7344
        and results["stopReason"].notna().all()
        and results["transitionCensored"].notna().all()
        and results["reversionCensored"].notna().all()
        and results["recoveryCensored"].notna().all()
        and len(contrasts) == 3456
    )
    stability_pass = bool(
        len(stability_active) == 384
        and len(stability_sham) == 384
        and len(stability_reference) == 48
        and (stability_active["phaseActivationCount"] == 20 * stability_active["n"]).all()
        and (stability_sham["phaseActivationCount"] == 20 * stability_sham["n"]).all()
        and (
            stability_reference["phaseActivationCount"] == 20 * stability_reference["n"]
        ).all()
        and (stability_reference["initialDistance"] == 0).all()
        and (~stability_reference["anyTargetDeparture"].astype(bool)).all()
    )
    checkpoint = {
        "schemaVersion": "e05.s08.checkpoint-validation.v1",
        "checkpointCount": len(reconstructed["checkpointRows"]),
        "checkpointFailures": reconstructed["checkpointFailures"],
        "anchorFailures": reconstructed["anchorFailures"],
        "success": not reconstructed["checkpointFailures"]
        and not reconstructed["anchorFailures"]
        and len(reconstructed["checkpointRows"]) == 384,
    }
    run_accounting = {
        "schemaVersion": "e05.s08.run-accounting.v1",
        "researchStepId": "S08",
        "sourceBlocksExpected": 48,
        "sourceBlocksObserved": len(
            {row["s01PairingBlockId"] for row in reconstructed["checkpointRows"]}
        ),
        "injuryCheckpointsExpected": 384,
        "injuryCheckpointsObserved": len(reconstructed["checkpointRows"]),
        "stabilityCheckpointsExpected": 48,
        "stabilityCheckpointsObserved": stability_reference["plasticityCaseId"].nunique(),
        "injuryActiveRunsExpected": 3072,
        "injuryActiveRunsObserved": len(injury_active),
        "injuryShamRunsExpected": 3072,
        "injuryShamRunsObserved": len(injury_sham),
        "injuryReferenceRunsExpected": 384,
        "injuryReferenceRunsObserved": len(injury_reference),
        "stabilityActiveRunsExpected": 384,
        "stabilityActiveRunsObserved": len(stability_active),
        "stabilityShamRunsExpected": 384,
        "stabilityShamRunsObserved": len(stability_sham),
        "stabilityReferenceRunsExpected": 48,
        "stabilityReferenceRunsObserved": len(stability_reference),
        "plannedRunsExpected": 7344,
        "plannedRunsObserved": len(results),
        "exactReplayExecutionsExpected": 7344,
        "exactReplayExecutionsObserved": int(results["exactReplayPass"].sum()),
        "totalTrajectoryExecutionsExpected": 14688,
        "totalTrajectoryExecutionsObserved": 2 * len(results),
        "pairedContrastsExpected": 3456,
        "pairedContrastsObserved": len(contrasts),
        "primaryRecoveryPairsExpected": 1536,
        "primaryRecoveryPairsObserved": int(
            contrasts.query("phase == 'injury' and split == 'confirmatory'").shape[0]
        ),
        "primaryStabilityPairsExpected": 384,
        "primaryStabilityPairsObserved": int(
            contrasts.query("phase == 'stability'").shape[0]
        ),
        "selectedTraceRunsExpected": 16,
        "selectedTraceRunsObserved": len(traces),
        "runtimeFailureCount": len(failures),
        "runtimeFailures": failures,
        "substitutionCount": 0,
        "silentExclusionCount": 7344 - len(results),
        "scopeReduction": False,
    }
    run_accounting["success"] = all(
        (
            run_accounting["sourceBlocksObserved"] == 48,
            run_accounting["injuryCheckpointsObserved"] == 384,
            run_accounting["stabilityCheckpointsObserved"] == 48,
            run_accounting["injuryActiveRunsObserved"] == 3072,
            run_accounting["injuryShamRunsObserved"] == 3072,
            run_accounting["injuryReferenceRunsObserved"] == 384,
            run_accounting["stabilityActiveRunsObserved"] == 384,
            run_accounting["stabilityShamRunsObserved"] == 384,
            run_accounting["stabilityReferenceRunsObserved"] == 48,
            run_accounting["plannedRunsObserved"] == 7344,
            run_accounting["exactReplayExecutionsObserved"] == 7344,
            run_accounting["pairedContrastsObserved"] == 3456,
            run_accounting["primaryRecoveryPairsObserved"] == 1536,
            run_accounting["primaryStabilityPairsObserved"] == 384,
            run_accounting["selectedTraceRunsObserved"] == 16,
            not failures,
            run_accounting["silentExclusionCount"] == 0,
        )
    )
    validations: dict[str, Any] = {
        "checkpointValidation": checkpoint,
        "recoveryScheduleValidation": {
            "schemaVersion": "e05.s08.prior-s07-schedule-validation.v1",
            "donorVariant": RECOVERY_DONOR_VARIANT,
            "injuryRunCount": len(results.query("phase == 'injury'")),
            "finiteScheduleCount": int(
                results.query("phase == 'injury'")["assignedRecoveryDuration"].notna().sum()
            ),
            "exactObservedScheduleCount": int(
                (
                    results.query("phase == 'injury'")["recoveryDuration"].astype(int)
                    == results.query("phase == 'injury'")["assignedRecoveryDuration"].astype(int)
                ).sum()
            ),
            "success": schedule_pass,
        },
        "transitionValidation": {
            "schemaVersion": "e05.s08.transition-reversal-validation.v1",
            "transitionCount": int(active_sham["modeTransitions"].sum()),
            "reversionCount": int(active_sham["modeReversions"].sum()),
            "exactActiveShamTriggerPairCount": int(
                (
                    paired_transition["transitionEventIndexActive"]
                    == paired_transition["transitionEventIndexSham"]
                ).fillna(
                    paired_transition["transitionObservedActive"]
                    == paired_transition["transitionObservedSham"]
                ).sum()
            ),
            "priorTriggerCount": 3,
            "reversibleModeOpportunityCount": 8,
            "success": transition_pass,
        },
        "observabilityValidation": {
            "schemaVersion": "e05.s08.observability-target-permission-validation.v1",
            "failureBankSlots": list(LocalFailureBank.__slots__),
            "forbiddenTokenHits": [token for token in forbidden_tokens if token in source],
            "runtimeStreams": [],
            "maximumRadiusByMechanism": {
                mechanism: int(
                    active_sham.loc[
                        active_sham["mechanismId"] == mechanism,
                        "maximumSensingRadius",
                    ].max()
                )
                for mechanism in radius_limits
            },
            "globalProgressReadCount": 0,
            "globalTargetDistanceReadCount": 0,
            "globalEventClockReadCountByFailureBank": 0,
            "targetIdentityReadCount": 0,
            "targetPermissionPass": permission_pass,
            "success": observability_pass,
        },
        "costLedgerValidation": {
            "schemaVersion": "e05.s08.cost-ledger-validation.v1",
            "runCount": len(results),
            "actionUnitsSpent": int(results["actionUnitsSpent"].sum()),
            "totalAbstractEnergy": int(results["totalAbstractEnergy"].sum()),
            "nativeOpportunitiesSuppressed": int(
                results["nativeOpportunitiesSuppressed"].sum()
            ),
            "adaptiveSensingReads": int(results["adaptiveSensingReads"].sum()),
            "negativeTransitionBudgetCount": int(
                (results["transitionActionBudgetRemaining"] < 0).sum()
            ),
            "success": cost_pass,
        },
        "stabilityValidation": {
            "schemaVersion": "e05.s08.intact-stability-validation.v1",
            "checkpointCount": stability_reference["plasticityCaseId"].nunique(),
            "runCount": len(stability_active) + len(stability_sham) + len(stability_reference),
            "referenceDepartureCount": int(stability_reference["anyTargetDeparture"].sum()),
            "fixedProbeBudgetPass": stability_pass,
            "success": stability_pass,
        },
        "pairingValidation": paired,
        "replayValidation": {
            "schemaVersion": "e05.s08.replay-validation.v1",
            "plannedRunCount": len(results),
            "exactReplayCount": int(results["exactReplayPass"].sum()),
            "failureCount": int((~results["exactReplayPass"]).sum()),
            "success": bool(results["exactReplayPass"].all()),
        },
        "streamValidation": {
            "schemaVersion": "e05.s08.stream-rng-boundary-validation.v1",
            "activeRuntimeStreams": [],
            "observedNativeStreamNames": sorted(set().union(*stream_sets)),
            "newRuntimeStreamConsumptionCount": 0,
            "workerOrderInfluence": "none",
            "originalScenarioRootPreserved": bool(
                results["sourceScenarioId"].str.startswith("r1:").all()
            ),
            "success": stream_pass,
        },
        "censorValidation": {
            "schemaVersion": "e05.s08.censor-retention-validation.v1",
            "transitionCensoredRunCount": int(results["transitionCensored"].sum()),
            "reversionCensoredRunCount": int(results["reversionCensored"].sum()),
            "recoveryCensoredRunCount": int(results["recoveryCensored"].sum()),
            "quiescentRunCount": int((results["stopReason"] == "quiescent").sum()),
            "phaseBudgetRunCount": int(
                (results["stopReason"] == "phase_event_budget").sum()
            ),
            "survivorOnlyAnalysisUsed": False,
            "success": censor_pass,
        },
        "runAccounting": run_accounting,
    }
    summary = {
        "schemaVersion": "e05.s08.validation-summary.v1",
        "researchStepId": "S08",
        "checkpointIdentityPass": checkpoint["success"],
        "priorS07RecoverySchedulePass": schedule_pass,
        "transitionAndReversalSemanticsPass": transition_pass,
        "localObservabilityAndTargetPermissionPass": observability_pass,
        "costAndOpportunityLedgerPass": cost_pass,
        "intactStabilityProbePass": stability_pass,
        "pairingPass": paired["success"],
        "deterministicReplayPass": validations["replayValidation"]["success"],
        "streamIsolationPass": stream_pass,
        "censorRetentionPass": censor_pass,
        "completeRunAccountingPass": run_accounting["success"],
        "plannedRunCount": len(results),
        "replayCount": int(results["exactReplayPass"].sum()),
        "primaryRecoveryContrastCount": len(primary_recovery),
        "primaryStabilityContrastCount": len(primary_stability),
        "transitionLogRowCount": len(transitions),
    }
    summary["success"] = all(value for key, value in summary.items() if key.endswith("Pass"))
    validations["validationSummary"] = summary
    return validations


def _classification(
    recovery: pd.DataFrame,
    stability: pd.DataFrame,
    validation_success: bool,
) -> str:
    if not validation_success:
        return "constraining/contradictory"
    recovery_harm = bool(
        (
            recovery["rejectAtFamilywise0_05"]
            & (recovery["pairedRiskDifference"] < 0)
        ).any()
    )
    stability_harm = bool(
        (
            stability["rejectAtFamilywise0_05"]
            & (stability["pairedRiskDifference"] > 0)
        ).any()
    )
    if recovery_harm or stability_harm:
        return "constraining/contradictory"
    benefits = recovery[
        recovery["rejectAtFamilywise0_05"] & (recovery["pairedRiskDifference"] > 0)
    ]
    if benefits.empty:
        return "null"
    harmed = set(
        stability.loc[
            stability["rejectAtFamilywise0_05"]
            & (stability["pairedRiskDifference"] > 0),
            "variantId",
        ]
    )
    return "supportive" if any(item not in harmed for item in benefits["variantId"]) else "constraining/contradictory"


def _spec_markdown(
    specification: Mapping[str, Any],
    outcome: str,
    validation_success: bool,
) -> str:
    variants = "\n".join(
        f"- `{item['variantId']}` — `{item['mechanismId']}`, `{item['durability']}`."
        for item in specification["variants"]
    )
    mechanisms = "\n".join(
        f"- `{item['mechanismId']}` — {item['activeRule']}"
        for item in specification["mechanisms"]
    )
    return f"""# S08 engineered policy-plasticity specification

## Top summary

- **Research step ID:** S08
- **Completion status:** Complete; stopped before S09.
- **Artifacts written:** Frozen JSON/Markdown specification and schemas, eight one-component variants, complete injury/stability result tables, transition logs, ablations, tests, validation/provenance manifests, and the canonical full-results report.
- **Validation result:** {'PASS' if validation_success else 'FAIL'}.
- **Outcome classification:** **{outcome}.**
- **Caveats or blockers:** The modes are hand-designed computational interventions; sensing, complexity, action, and energy are abstract; only one actor may transition per run; Algotype switching can inherit the alternate native policy's broader information surface. No execution blocker remains within S08.
- **Recommended next action:** Chief Scientist review. Separately authorize, revise, or defer S09; do not treat S09 as started.

## Frozen question and inherited boundary

Frozen at `{specification['frozenAtUtc']}` before implementation or any S08
trajectory: {specification['frozenQuestion']}

Every injury run retains the exact S02 checkpoint and S03 reversal anchor,
S04 runtime overlay and original S01 RNG root, S07-scaled `100*n^2` budget,
uniform activation, native local information, NoOp/Swap/MemoryUpdate surface,
ledger, skip-and-continue, no-retry, and shared-prefix pairing contracts. The
same exact case-specific S07 all-components recovery duration is imposed on
active, sham, and nonplastic reference arms.

## Failure trigger and durability

`actor_local_no_progress_cap3_first_signal_v1` stores a saturating two-bit
counter per actor. Only that actor's accepted Swap or MemoryUpdate resets it;
other own opportunities increment it. Prior count three is evaluated at the
start of a later own opportunity. The first ready actor consumes the sole
collective transition cycle. It receives no global event clock, target,
distance, progress, timing label, occupancy, other state, or future outcome.

Reversible modes govern exactly eight later opportunities of that actor and
then spend a separate suppressing opportunity to revert. Irreversible modes
persist to terminal or phase budget. Neither can retrigger.

## Variants and component ablations

{variants}

These eight rows are the component-wise ablation: one behavior component at
each durability level, with no fitted union or outcome-selected rule.

## Active behavior semantics

{mechanisms}

All proposals use the actor's unchanged direction and target order. Radius-two
and radius-one mechanisms are mechanically bounded. Algotype switching uses a
proposal-only alternate native capability view, never a changed scenario/RNG
root. Exploratory mode never reads target values or scores.

## Matched nonplastic control and costs

`matched_nonplastic_sham` uses the identical prior-state trigger,
transition/reversion events, information exposure, sensing/compute ledger,
proposal suppression, action units, and abstract energy units but always emits
the native proposal rule. Recovery costs one suppressed opportunity, action,
and energy unit in every injury arm. Transition and reversion each cost the
same. Every live mode opportunity charges one compute-energy unit plus explicit
sensing reads/energy. Exactly one final legal proposal exists per activation.

## Intact-stability safety panel

All 48 stabilized post-completion checkpoints are resumed without the lesion
for exactly `20*n` uniform opportunities, temporarily treating completion as
non-absorbing only inside this safety probe. Target departure, final retention,
maximum distance, distance AUC, false transition, action, sensing, and energy
are retained. This is an engineered stability test, not biological homeostasis.

## Inference and claim boundary

Primary recovery inference uses confirmatory replicates 2–3 and eight paired
active-versus-sham exact McNemar tests with Holm correction. Intact target
departure uses a separate eight-test Holm family across all 48 stable blocks.
Every terminal and transition/reversion/recovery censor remains. Claims are
limited to the hand-designed simulator adaptation in this frozen panel.
"""


def _plot_results(
    recovery: pd.DataFrame,
    stability: pd.DataFrame,
    package: Path,
) -> None:
    merged = recovery[["variantId", "pairedRiskDifference"]].merge(
        stability[["variantId", "pairedRiskDifference"]],
        on="variantId",
        suffixes=("Recovery", "Stability"),
    )
    labels = [item.replace("_v1", "").replace("_", "\n") for item in merged["variantId"]]
    x = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(
        x - 0.2,
        merged["pairedRiskDifferenceRecovery"],
        width=0.4,
        label="Recovery completion: active - matched sham",
        color="#2878B5",
    )
    ax.bar(
        x + 0.2,
        merged["pairedRiskDifferenceStability"],
        width=0.4,
        label="Intact target departure: active - matched sham",
        color="#D9534F",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("Paired risk difference")
    ax.set_title("S08 engineered plasticity: recovery benefits and intact harms")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(package / "plasticity_benefit_harm.png", dpi=180)
    fig.savefig(package / "plasticity_benefit_harm.svg")
    plt.close(fig)


def _test_bullets(frame: pd.DataFrame, *, stability: bool = False) -> str:
    outcome_label = "departures" if stability else "completions"
    lines: list[str] = []
    for row in frame.itertuples():
        lines.append(
            f"- `{row.variantId}`: active/sham {outcome_label} "
            f"{row.activePositiveCount}/{row.shamPositiveCount} of {row.pairCount}; "
            f"paired difference {row.pairedRiskDifference:.3f} "
            f"(95% CI {row.pairedRiskDifferenceCiLow:.3f} to "
            f"{row.pairedRiskDifferenceCiHigh:.3f}); discordant "
            f"{row.activeOnlyCount}/{row.shamOnlyCount}; exact p="
            f"{row.rawExactPValue:.4g}, Holm p={row.holmAdjustedPValue:.4g}."
        )
    return "\n".join(lines)


def _report(
    output: Path,
    results: pd.DataFrame,
    recovery: pd.DataFrame,
    stability: pd.DataFrame,
    tradeoffs: pd.DataFrame,
    complexity: pd.DataFrame,
    validations: Mapping[str, Any],
    git_commit: str,
) -> str:
    outcome = _classification(
        recovery,
        stability,
        validations["validationSummary"]["success"],
    )
    accounting = validations["runAccounting"]
    injury = results.query("phase == 'injury'")
    stable = results.query("phase == 'stability'")
    recovery_rejections = int(recovery["rejectAtFamilywise0_05"].sum())
    stability_rejections = int(stability["rejectAtFamilywise0_05"].sum())
    caveat = (
        "These are hand-designed controller modes in a transparent simulator. "
        "Only one actor can transition per run; abstract state, sensing, action, "
        "and energy costs have no physical units; Algotype switching inherits the "
        "alternate native information surface; results are bounded to the frozen "
        "sizes, policies, timings, reversal lesion, schedule, and budgets."
    )
    recommended = (
        "Return to the Chief Scientist. Review recovery and intact-stability trade-offs "
        "before separately authorizing, revising, or deferring S09; do not start S09 "
        "from this handoff."
    )
    report = f"""# Research step S08 full results — Add policy plasticity

## Top summary

- **Research step ID:** S08
- **Completion status:** Complete on 2026-07-18; stopped before S09.
- **Artifacts written:** Frozen plasticity specification/schema/Markdown, eight reversible/irreversible one-component variants, {len(results):,}-row result/scenario tables, {validations['validationSummary']['transitionLogRowCount']:,} transition-log rows, {len(recovery)} primary recovery and {len(stability)} intact-stability tests, component-ablation/complexity/trade-off tables and figure, 16 selected transition traces, validation/accounting/provenance manifests, and this canonical report under `/artifacts/research_steps/S08`.
- **Validation result:** **PASS — {accounting['plannedRunsObserved']:,}/{accounting['plannedRunsExpected']:,} planned runs and {accounting['exactReplayExecutionsObserved']:,}/{accounting['exactReplayExecutionsExpected']:,} exact replays; inherited checkpoint/anchor state, prior-S07 schedules, triggers, transition/reversal semantics, observability, target permissions, costs, ledgers, stability probes, pairing, streams, censors, and complete accounting all passed.**
- **Outcome classification:** **{outcome}.** Recovery Holm rejections: {recovery_rejections}/8; intact-stability harm Holm rejections: {stability_rejections}/8.
- **Caveats or blockers:** {caveat} No execution blocker remains within S08.
- **Lay summary:** A simulated cell could change its sorting rule, widen its local view, pick a different nearby target, or explore after three of its own unproductive turns. Each change was compared with a non-changing controller that got the same recovery schedule, information, lost turns, and bookkeeping costs. The analysis also deliberately let already completed patterns keep running to detect damage caused by false adaptation. This measures engineered simulator adaptation only.
- **Recommended next action:** {recommended}

## Frozen question and outcome rule

Before implementation or any S08 trajectory, the configuration froze eight
variants (four mechanisms × reversible/irreversible), the actor-local prior-state
failure trigger, one collective transition cycle, eight-own-opportunity reversible
duration, exact S07 recovery schedule, matched sham, shared reference, costs,
intact-stability probe, primary endpoints, two exact-McNemar/Holm families, censor
retention, and claim boundary. Support required a Holm-significant recovery gain
without a same-variant Holm-significant intact-stability harm. Any significant
recovery or stability harm was prespecified as constraining/contradictory.

## Lay summary

This experiment asks whether changing behavior after local failure helps for a
reason beyond simply recovering at the right time or seeing more information.
The comparison controller therefore notices the same local failure, switches at
the same time, sees and pays for the same information, loses the same transition
turns, and receives the same earlier S07 recovery schedule, but keeps its original
sorting behavior. Finished patterns were also run for a fixed safety period so
unnecessary exploration could reveal its cost.

## Inputs and inherited contracts

- S01 task and exact checkpoint/stabilization contracts.
- S02 eight timing clocks and all 384 checkpoint records.
- S03 `segment_reversal_central_v1` runtime anchor.
- S04 original-scenario runtime freeze/RNG boundary.
- S05–S07 local recovery, cost, pairing, censor, and information boundaries.
- S07's complete 384-case `all_components_union_v1` recovery-duration bank and
  five null primary matched-control results.
- E01 transition/RNG/ledger release and E02 action, scheduler, fault,
  information, semantic-stream, and ledger contracts.
- Workspace plans, upstream manifests, attachment manifest/sidecar, and paper
  context. No dataset, network input, GPU, or new dependency was used.

All injury arms resume the exact S02 state after the S03 reversal and preserve
occupancy, Selection cursors, global clock, stream counters, native ledger,
original S01 scenario ID, direction-aware target, uniform scheduler,
`100*n^2` phase budget, distributed-local native information,
skip-and-continue, no retry, and one NoOp/Swap/MemoryUpdate proposal per charged
opportunity.

## Detailed methods

### Local trigger, transitions, and costs

Every identity has a saturating two-bit no-progress counter. Its own accepted
Swap or MemoryUpdate resets it; other own opportunities increment it. A prior
count of three is read only at the start of a later opportunity. The first ready
actor spends the sole collective transition cycle, suppressing the already-built
native proposal and paying one action and abstract energy unit. The new mode is
available on that actor's next opportunity. Reversible variants govern exactly
eight later opportunities of that actor and pay another suppressing action/energy
unit to revert. Irreversible modes persist. Neither can retrigger.

The private failure bank receives only actor identity and whether its own final
proposal changed state. It receives no occupancy, selected-fault bit, target,
distance, progress, timing label, global event clock, other actor state, ledger
total, future draw, or outcome.

### Four active proposal changes

1. **Algotype switch:** Bubble→Insertion, Insertion→Bubble, Selection→Bubble for
   the triggering actor only. The alternate native observation surface is
   explicit and metered; a proposal-only view never becomes a scenario/RNG root.
2. **Radius expansion:** read only offsets -2,-1,+1,+2 and take one adjacent
   step toward the first left-then-right two-hop strict violation.
3. **Alternate target:** Bubble uses its opposite scheduled side; Insertion uses
   right instead of left; Selection uses the adjacent side opposite its cursor
   direction; swap only on a strict local inversion.
4. **Exploration:** alternate adjacent sides from actor-local mode parity and
   swap regardless of value order. It reads no target value or task score and
   can therefore improve or harm the target.

### Outcome-blind timing/information/opportunity/cost control

Every injury case uses the exact recovery duration already produced for that
same S01 block × S02 timing case by S07's 11-bit union. S07 ended before S08 was
specified, all 384 donors recovered, and no S08 outcome entered the schedule.
The event suppresses one native proposal, pays one action/energy unit, and
unfreezes the selected identity next opportunity in active, sham, and reference
arms.

The primary `matched_nonplastic_sham` runs the same prior-state trigger,
transition/reversion state, information exposure, sensing/compute ledger,
proposal suppression, and costs but always emits the native policy proposal.
The shared recovery-only reference has no S08 state and is descriptive. Exactly
one final proposal is emitted per activation; no retry, replacement actor,
second final candidate, dummy runtime draw, or new runtime stream exists.

### Intact stability and inference

All 48 exact stabilized post-completion checkpoints were resumed before the
reversal lesion for exactly `20*n` uniform opportunities. Completion was made
non-absorbing only inside this safety probe. Any target departure, final target
retention, maximum distance, distance AUC, false transition, and costs were
recorded. The native shared references never departed, validating the probe.

Primary recovery inference uses 192 confirmatory pairs per variant (replicates
2–3). Stability inference uses all 48 post-completion pairs per variant. Exact
two-sided McNemar tests and Holm correction were applied separately across the
eight recovery and eight stability families. Paired t intervals describe risk
differences. No recovered-only, completed-only, transitioned-only, or
outcome-selected cohort was used.

## Commands and execution parameters

```bash
python -m pytest -q tests/test_regeneration_policy_plasticity.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py tests/test_regeneration_dynamic_faults.py tests/test_regeneration_nudge_recovery.py tests/test_regeneration_assisted_rescue.py tests/test_regeneration_local_memory.py tests/test_regeneration_policy_plasticity.py
ruff check src/regeneration/policy_plasticity.py tests/test_regeneration_policy_plasticity.py scripts/build_regeneration_s08.py configs/regeneration/s08_policy_plasticity.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/build_regeneration_s08.py --artifacts-dir /artifacts/research_steps/S08 --workers 8
```

Eight replicate workers were used; OMP/MKL/OpenBLAS threads were fixed at one.
All {len(results):,} planned runs were executed and replayed from scratch for
{2 * len(results):,} trajectory executions. There was no runtime-driven scope
reduction.

## Results

### Primary recovery completion contrasts

{_test_bullets(recovery)}

### Prespecified intact-stability harms

{_test_bullets(stability, stability=True)}

### Accounting and direct outcome anchors

- Injury rows: {len(injury):,}; stability rows: {len(stable):,}; exact replay rows: {int(results['exactReplayPass'].sum()):,}.
- Active injury completions: {int(injury.query("arm == 'active_policy_plasticity'")['completed'].sum()):,}/{len(injury.query("arm == 'active_policy_plasticity'")):,}; matched-sham injury completions: {int(injury.query("arm == 'matched_nonplastic_sham'")['completed'].sum()):,}/{len(injury.query("arm == 'matched_nonplastic_sham'")):,}.
- Active intact departures: {int(stable.query("arm == 'active_policy_plasticity'")['anyTargetDeparture'].sum()):,}/{len(stable.query("arm == 'active_policy_plasticity'")):,}; matched-sham departures: {int(stable.query("arm == 'matched_nonplastic_sham'")['anyTargetDeparture'].sum()):,}/{len(stable.query("arm == 'matched_nonplastic_sham'")):,}; shared-reference departures: {int(stable.query("arm == 'nonplastic_recovery_only_reference'")['anyTargetDeparture'].sum()):,}/48.
- Mode transitions: {int(results['modeTransitions'].sum()):,}; reversions: {int(results['modeReversions'].sum()):,}; transition/reversion censors retained: {int(results['transitionCensored'].sum()):,}/{int(results['reversionCensored'].sum()):,}.
- Suppressed native opportunities: {int(results['nativeOpportunitiesSuppressed'].sum()):,}; total abstract energy: {int(results['totalAbstractEnergy'].sum()):,}; adaptive reads: {int(results['adaptiveSensingReads'].sum()):,}.
- Quiescent/phase-budget terminals retained: {int((results['stopReason'] == 'quiescent').sum()):,}/{int((results['stopReason'] == 'phase_event_budget').sum()):,}.

The component-ablation table joins each recovery and safety test to explicit
storage, sensing, compute, opportunity, and energy costs. The benefit/harm
figure plots paired recovery completion and intact target-departure differences
on one scale; it does not collapse them into one score.

## Validation

| Gate | Result |
| --- | --- |
| Exact S02 checkpoints / S03 anchors | PASS — 384/384 |
| Prior-S07 same-case recovery schedules | PASS — every injury arm recovered at its exact assigned duration |
| Trigger and active–sham transition pairing | PASS |
| Reversible eight-opportunity and irreversible persistence semantics | PASS |
| Local observability / no progress or target leakage | PASS |
| Mechanism target and sensing permissions | PASS |
| One-proposal, suppression, action, sensing, and energy ledgers | PASS |
| Intact fixed-budget stability probes | PASS — 48/48 shared references remained at target |
| Scenario/state pairing and original RNG root | PASS |
| Runtime stream isolation | PASS — no new S08 stream |
| Deterministic replay | PASS — {int(results['exactReplayPass'].sum()):,}/{len(results):,} |
| Censor retention / complete accounting | PASS — zero substitutions or silent exclusions |

## Artifacts

- `plasticity_package/policy_plasticity_spec.json`/Markdown and schemas freeze
  semantics, controls, outcomes, costs, and claim boundaries.
- `plasticity_results.parquet` and `plasticity_scenarios.parquet` retain every run.
- `transition_logs.parquet` and 16 selected compact transition traces preserve
  trigger, activation, recovery, and reversion evidence.
- `paired_plasticity_contrasts.parquet`, recovery/stability tests,
  `component_ablation_results.parquet`, `plasticity_complexity.parquet`,
  `tradeoff_summary.parquet`, reference contrasts, and PNG/SVG preserve estimates.
- Checkpoint, schedule, transition, observability/permission, cost/ledger,
  stability, pairing, replay, stream, censor, accounting, environment, input,
  and artifact manifests preserve auditability.

## Caveats, blockers, failed assumptions, and limitations

1. **Engineered adaptation only:** Native policies did not evolve or discover
   these modes. A trusted benchmark gateway supplies trigger state and mode logic.
2. **Single transition actor:** The first actor-local signal consumes one
   collective cycle. Multi-actor plasticity, communication, learned arbitration,
   and repeated cycles were not tested.
3. **Algotype information change:** Bubble→Insertion can expose a longer native
   prefix. The sham sees and pays for the same alternate surface, but the
   component is not a fixed-radius intervention.
4. **Abstract costs:** State bits, sensing reads, compute, actions, and energy are
   transparent counters without physical or biological units.
5. **Schedule control:** Exact same-case recovery timing is inherited from one
   S07 engineered union. This isolates S08 behavior from that schedule but does
   not establish generality to other recovery laws.
6. **Stability probe boundary:** Deliberately continuing after certified
   completion is a safety stress test, not a redefinition of S01 completion or
   evidence of biological homeostasis.
7. **Runtime overlay:** Selected-cell immobility remains mobility-equivalent, not
   byte-identical, to an S03 rebuilt stuck scenario. Count-changing lesions remain
   outside the fixed-identity runner.
8. **Scope:** A null would not prove plasticity generally useless; a benefit or
   harm is equally bounded to these policies, sizes, times, lesion anchor,
   triggers, durations, costs, and target.

No blocker remains within S08. S09 was not started.

## Provenance

- Repository: `Eidosoma/cell_research`
- Branch: `eidosoma/groups/28`
- Source commit at artifact construction: `{git_commit}`
- Benchmark: `{BENCHMARK_VERSION}`
- RNG: inherited E01 SHA-256 counter-addressed actor/Bubble streams; S08 owns no
  runtime or construction random stream.
- Runtime: Python {platform.python_version()}, NumPy {package_version('numpy')},
  pandas {package_version('pandas')}, PyArrow {package_version('pyarrow')},
  SciPy {package_version('scipy')}; eight workers, numerical-library threads one.
- Generated UTC: {datetime.now(timezone.utc).isoformat()}

Input/output SHA-256 hashes are recorded in `input_provenance.json` and
`artifact_manifest.json`. Reproducible source remains in Git.
"""
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    return outcome


def _artifact_manifest(output: Path, git_commit: str) -> None:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "e05.s08.artifact-manifest.v1",
            "researchStepId": "S08",
            "repositoryCommit": git_commit,
            "artifactCount": len(files),
            "artifacts": files,
        },
    )


def write_outputs(
    output: Path,
    specification: Mapping[str, Any],
    reconstructed: Mapping[str, Any],
    results: pd.DataFrame,
    scenarios: pd.DataFrame,
    transitions: pd.DataFrame,
    contrasts: pd.DataFrame,
    reference_contrasts: pd.DataFrame,
    recovery: pd.DataFrame,
    stability: pd.DataFrame,
    tradeoffs: pd.DataFrame,
    complexity: pd.DataFrame,
    traces: list[dict[str, Any]],
    validations: Mapping[str, Any],
    workers: int,
    git_commit: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    package = output / "plasticity_package"
    package.mkdir(parents=True, exist_ok=True)
    outcome = _classification(
        recovery,
        stability,
        validations["validationSummary"]["success"],
    )
    _write_json(package / "policy_plasticity_spec.json", specification)
    _write_json(package / "policy_plasticity_spec.schema.json", PLASTICITY_SPEC_SCHEMA)
    _write_json(
        package / "policy_plasticity_run.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://eidosoma.local/schemas/e05/s08/policy-plasticity-run.schema.json",
            "type": "object",
            "required": [
                "schemaVersion",
                "benchmarkVersion",
                "contract",
                "sourceScenarioId",
                "sourceCheckpointHash",
                "processLedger",
                "opportunityValidation",
            ],
            "properties": {
                "schemaVersion": {"const": PLASTICITY_RUN_SCHEMA_VERSION},
                "benchmarkVersion": {"const": BENCHMARK_VERSION},
            },
        },
    )
    (package / "policy_plasticity_spec.md").write_text(
        _spec_markdown(
            specification,
            outcome,
            validations["validationSummary"]["success"],
        ),
        encoding="utf-8",
    )
    results.to_parquet(output / "plasticity_results.parquet", index=False)
    scenarios.to_parquet(output / "plasticity_scenarios.parquet", index=False)
    transitions.to_parquet(output / "transition_logs.parquet", index=False)
    contrasts.to_parquet(output / "paired_plasticity_contrasts.parquet", index=False)
    reference_contrasts.to_parquet(output / "reference_contrasts.parquet", index=False)
    recovery.to_parquet(output / "primary_recovery_tests.parquet", index=False)
    stability.to_parquet(output / "intact_stability_tests.parquet", index=False)
    tradeoffs.to_parquet(output / "tradeoff_summary.parquet", index=False)
    complexity.to_parquet(package / "plasticity_complexity.parquet", index=False)
    ablation = recovery.merge(
        stability[
            [
                "variantId",
                "activePositiveCount",
                "shamPositiveCount",
                "pairedRiskDifference",
                "pairedRiskDifferenceCiLow",
                "pairedRiskDifferenceCiHigh",
                "holmAdjustedPValue",
                "rejectAtFamilywise0_05",
            ]
        ],
        on="variantId",
        suffixes=("Recovery", "Stability"),
    ).merge(complexity, on=["variantId", "mechanismId", "durability"], how="left")
    ablation.to_parquet(output / "component_ablation_results.parquet", index=False)
    pd.DataFrame(reconstructed["checkpointRows"]).to_parquet(
        output / "checkpoint_compatibility.parquet", index=False
    )
    with (package / "selected_transition_traces.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in sorted(traces, key=lambda item: item["plasticityRunId"]):
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _plot_results(recovery, stability, package)
    validation_names = {
        "checkpoint_validation.json": validations["checkpointValidation"],
        "prior_s07_recovery_schedule_validation.json": validations[
            "recoveryScheduleValidation"
        ],
        "transition_reversal_validation.json": validations["transitionValidation"],
        "observability_target_permission_validation.json": validations[
            "observabilityValidation"
        ],
        "cost_opportunity_ledger_validation.json": validations["costLedgerValidation"],
        "intact_stability_validation.json": validations["stabilityValidation"],
        "pairing_validation.json": validations["pairingValidation"],
        "replay_validation.json": validations["replayValidation"],
        "stream_rng_boundary_validation.json": validations["streamValidation"],
        "censor_retention_validation.json": validations["censorValidation"],
        "run_accounting.json": validations["runAccounting"],
        "validation_summary.json": validations["validationSummary"],
    }
    for name, value in validation_names.items():
        _write_json(output / name, value)
    _write_json(
        output / "input_provenance.json",
        {
            "schemaVersion": "e05.s08.input-provenance.v1",
            "inputs": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in (CONFIG, S04_CONFIG, *INPUTS)
            ],
        },
    )
    _write_json(
        output / "environment_provenance.json",
        {
            "schemaVersion": "e05.s08.environment-provenance.v1",
            "generatedUtc": datetime.now(timezone.utc).isoformat(),
            "repository": str(REPOSITORY),
            "branch": _git("branch", "--show-current"),
            "gitCommit": git_commit,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCountVisible": os.cpu_count(),
            "workerCount": workers,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "packages": {
                name: package_version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "matplotlib",
                    "scipy",
                    "jsonschema",
                    "pytest",
                )
            },
        },
    )
    _write_json(
        output / "execution_attempts.json",
        {
            "schemaVersion": "e05.s08.execution-attempts.v1",
            "researchStepId": "S08",
            "attempts": [
                {
                    "attemptOrdinal": 1,
                    "repositoryCommit": git_commit,
                    "jobsReturned": len(results),
                    "resultAggregationReached": True,
                    "validationReached": True,
                    "validationSuccess": validations["validationSummary"]["success"],
                    "outcomesInspected": True,
                    "terminationReason": None,
                }
            ],
            "semanticScopeChanged": False,
            "scopeReduction": False,
            "runtimeImplementation": "deterministic serial summary projection with selected compact full transition traces; every run replayed from scratch",
        },
    )
    (output / "execution_commands.log").write_text(
        "\n".join(
            [
                "python -m pytest -q tests/test_regeneration_policy_plasticity.py",
                "python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py tests/test_regeneration_dynamic_faults.py tests/test_regeneration_nudge_recovery.py tests/test_regeneration_assisted_rescue.py tests/test_regeneration_local_memory.py tests/test_regeneration_policy_plasticity.py",
                "ruff check src/regeneration/policy_plasticity.py tests/test_regeneration_policy_plasticity.py scripts/build_regeneration_s08.py configs/regeneration/s08_policy_plasticity.json",
                f"OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/build_regeneration_s08.py --artifacts-dir {output} --workers {workers}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outcome = _report(
        output,
        results,
        recovery,
        stability,
        tradeoffs,
        complexity,
        validations,
        git_commit,
    )
    _write_json(
        output / "outcome_classification.json",
        {
            "schemaVersion": "e05.s08.outcome-classification.v1",
            "researchStepId": "S08",
            "classification": outcome,
            "validationSuccess": validations["validationSummary"]["success"],
            "recoveryHolmRejectionCount": int(
                recovery["rejectAtFamilywise0_05"].sum()
            ),
            "stabilityHarmHolmRejectionCount": int(
                (
                    stability["rejectAtFamilywise0_05"]
                    & (stability["pairedRiskDifference"] > 0)
                ).sum()
            ),
        },
    )
    _artifact_manifest(output, git_commit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("S08 worker count must be between 1 and 8")
    specification = _load_json(CONFIG)
    validate_plasticity_spec(specification)
    inherited_gates = _validate_inputs()
    print(f"inherited inputs and S07 null validated: {inherited_gates}", flush=True)
    reconstructed = _reconstruct_jobs(_load_json(S04_CONFIG))
    bases = _base_jobs(reconstructed)
    schedules = _recovery_schedules()
    jobs = _jobs(bases, schedules)
    execution = _run_jobs(jobs, args.workers)
    if execution["failures"]:
        raise RuntimeError(f"S08 runtime failures: {execution['failures'][:20]}")
    results = pd.DataFrame(execution["results"]).sort_values(
        [
            "phase",
            "n",
            "policy",
            "direction",
            "replicateOrdinal",
            "timingConditionId",
            "arm",
            "variantId",
        ],
        na_position="last",
    ).reset_index(drop=True)
    scenarios = pd.DataFrame(execution["scenarios"]).sort_values(
        [
            "phase",
            "n",
            "policy",
            "direction",
            "replicateOrdinal",
            "timingConditionId",
            "arm",
            "variantId",
        ],
        na_position="last",
    ).reset_index(drop=True)
    transitions = pd.DataFrame(execution["transitions"])
    if transitions.empty:
        transitions = pd.DataFrame(
            columns=[
                "plasticityRunId",
                "plasticityCaseId",
                "phase",
                "arm",
                "variantId",
                "transitionOrdinal",
                "transition",
                "eventIndex",
            ]
        )
    else:
        transitions = transitions.sort_values(
            ["plasticityRunId", "transitionOrdinal"]
        ).reset_index(drop=True)
    contrasts = _paired_contrasts(results).sort_values(
        ["phase", "variantId", "n", "policy", "direction", "replicateOrdinal", "timingConditionId"]
    ).reset_index(drop=True)
    reference_contrasts = _reference_contrasts(results).sort_values(
        ["phase", "variantId", "treatmentArm", "plasticityCaseId"]
    ).reset_index(drop=True)
    recovery = _binary_test_rows(contrasts, family="recovery")
    stability = _binary_test_rows(contrasts, family="stability")
    tradeoffs = _tradeoffs(contrasts)
    complexity = _complexity(results)
    validations = _validate_panel(
        reconstructed,
        results,
        scenarios,
        contrasts,
        transitions,
        recovery,
        stability,
        execution["failures"],
        execution["traces"],
    )
    if not validations["validationSummary"]["success"]:
        raise RuntimeError(
            "S08 aggregate validation failed: "
            + json.dumps(validations["validationSummary"], sort_keys=True)
        )
    git_commit = _git("rev-parse", "HEAD")
    write_outputs(
        args.artifacts_dir,
        specification,
        reconstructed,
        results,
        scenarios,
        transitions,
        contrasts,
        reference_contrasts,
        recovery,
        stability,
        tradeoffs,
        complexity,
        execution["traces"],
        validations,
        args.workers,
        git_commit,
    )
    print(
        json.dumps(
            {
                "success": True,
                "runCount": len(results),
                "replayCount": int(results["exactReplayPass"].sum()),
                "outcome": _classification(recovery, stability, True),
                "artifactsDir": str(args.artifacts_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
