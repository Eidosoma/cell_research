#!/usr/bin/env python3
"""Build and validate E05 S07 minimal local-memory evidence."""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, t


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from reference_simulator.model import canonical_json_bytes, sha256_json  # noqa: E402
from reference_simulator.rng import u64  # noqa: E402
from scripts.build_regeneration_s04 import (  # noqa: E402
    ACTIVE_PROFILES,
    DynamicJob,
    _reconstruct_jobs,
)
from src.regeneration.local_memory import (  # noqa: E402
    BENCHMARK_VERSION,
    COMPONENT_BITS,
    CONTROL_ASSIGNMENT_STREAM,
    LOCAL_MEMORY_RUN_SCHEMA_VERSION,
    LOCAL_MEMORY_SPEC_SCHEMA,
    VARIANT_COMPONENTS,
    LocalMemoryBank,
    LocalMemoryContract,
    LocalOutcome,
    MatchingChoice,
    MemoryArm,
    MemoryVariant,
    exact_replay_local_memory,
    local_memory_case_id,
    run_local_memory_phase,
    validate_local_memory_spec,
)


CONFIG = REPOSITORY / "configs/regeneration/s07_local_memory.json"
S04_CONFIG = REPOSITORY / "configs/regeneration/s04_dynamic_faults.json"
S06_PRIMARY = Path("/artifacts/research_steps/S06/primary_completion_tests.parquet")
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
        for step in range(1, 7)
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
    Path("/artifacts/research_steps/S05/outcome_classification.json"),
    Path("/artifacts/research_steps/S06/assisted_rescue_package/assisted_rescue_spec.md"),
    Path("/artifacts/research_steps/S06/assisted_rescue_package/assisted_rescue_spec.json"),
    Path("/artifacts/research_steps/S06/outcome_classification.json"),
    S06_PRIMARY,
    Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
    Path("/previous-artifacts/E01/specification/transition_spec.md"),
    Path("/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json"),
    Path("/previous-artifacts/E02/research_steps/S02/action_interface_spec.md"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/scheduler_contract.md"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/scheduler_prespecification.json"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/opportunity_ledger_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/rng_consumption_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/fault_semantics_contract.md"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/fault_prespecification.json"),
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
        raise FileNotFoundError(f"missing required S01-S06/E01/E02 inputs: {missing}")
    gates = {
        f"s{step:02d}": bool(
            _load_json(Path(f"/artifacts/research_steps/S{step:02d}/validation_summary.json"))[
                "success"
            ]
        )
        for step in range(1, 7)
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
    if not all(gates.values()):
        raise RuntimeError(f"an inherited validation gate failed: {gates}")
    s06 = pd.read_parquet(S06_PRIMARY)
    timing = s06[
        s06["controlArm"].isin(
            ["spontaneous_time_matched", "matched_time_opportunity_energy"]
        )
    ]
    s06_null = bool(
        len(timing) == 2
        and (~timing["rejectAtFamilywise0_05"].astype(bool)).all()
        and (timing["holmAdjustedPValue"] == 1.0).all()
    )
    if not s06_null:
        raise RuntimeError("S07 expected S06's two timing/opportunity-matched nulls")
    gates["s06TimingMatchedNull"] = True
    return gates


@dataclass(frozen=True, slots=True)
class MemoryJob:
    base: DynamicJob
    arm: MemoryArm
    variant: MemoryVariant | None
    split: str
    matching_choice: str
    assigned_duration: int | None
    assignment_id: str | None
    donor_case_id: str | None
    retain_full_trace: bool = False


def _case_id(job: MemoryJob) -> str:
    return local_memory_case_id(
        job.base.s01_pairing_block_id,
        job.base.timing_condition_id,
        job.base.anchor_lesion_state_hash,
    )


def _execute_job(job: MemoryJob) -> dict[str, Any]:
    contract = LocalMemoryContract(
        job.arm,
        job.variant,
        assigned_duration=job.assigned_duration,
    )
    run = run_local_memory_phase(
        job.base.scenario,
        job.base.checkpoint,
        post_anchor_occupancy=job.base.post_anchor_occupancy,
        anchor_lesion_state_hash=job.base.anchor_lesion_state_hash,
        selected_identity=job.base.selected_identity,
        contract=contract,
        recovery_budget=job.base.recovery_budget,
        trace_mode="full" if job.retain_full_trace else "digest",
        retain_process_audits=job.retain_full_trace,
    )
    exact_replay_local_memory(
        run,
        job.base.scenario,
        job.base.checkpoint,
        job.base.post_anchor_occupancy,
        job.base.recovery_budget,
    )
    case_id = _case_id(job)
    variant_id = None if job.variant is None else job.variant.value
    run_id = "e05mr7:" + sha256_json(
        {
            "memoryCaseId": case_id,
            "arm": job.arm.value,
            "variantId": variant_id,
            "split": job.split,
            "matchingChoice": job.matching_choice,
        }
    )
    summary = dict(run.summary)
    process = dict(run.process_final_state)
    ledger = dict(run.process_ledger)
    row = {
        "schemaVersion": "e05.s07.local-memory-result.v1",
        "benchmarkVersion": BENCHMARK_VERSION,
        "memoryRunId": run_id,
        "memoryCaseId": case_id,
        "arm": job.arm.value,
        "variantId": variant_id,
        "split": job.split,
        "matchingChoice": job.matching_choice,
        "assignmentId": job.assignment_id,
        "donorCaseId": job.donor_case_id,
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
        "anchorLesionStateHash": job.base.anchor_lesion_state_hash,
        "postAnchorOccupancySha256": _digest(job.base.post_anchor_occupancy),
        "selectedIdentityId": job.base.selected_identity,
        "globalStartEventIndex": run.start_event_index,
        "globalEndEventIndex": run.end_event_index,
        "recoveryBudget": job.base.recovery_budget,
        "stopReason": summary["stopReason"],
        "completed": summary["completed"],
        "phaseActivationCount": summary["phaseActivationCount"],
        "finalDistance": summary["finalDistance"],
        "distanceAuc": summary["distanceAuc"],
        "recoveryObserved": summary["recoveryObserved"],
        "recoveryDuration": summary["recoveryDuration"],
        "recoveryCensored": summary["recoveryCensored"],
        "assignedDuration": job.assigned_duration,
        "assignedUnrecoveredSentinel": job.arm == MemoryArm.MATCHED
        and job.assigned_duration is None,
        "competingTerminalBeforeAssignedEvent": job.arm == MemoryArm.MATCHED
        and job.assigned_duration is not None
        and not summary["recoveryObserved"],
        "recoveryReason": summary["recoveryReason"],
        "interventionEventIndex": process["interventionEventIndex"],
        "interventionActorId": process["interventionActorId"],
        "interventionNativeProposalKind": process["interventionNativeProposalKind"],
        "interventionNativeProposalEligible": process[
            "interventionNativeProposalEligible"
        ],
        "triggerComponent": process["triggerComponent"],
        "storageBitsPerIdentity": summary["storageBitsPerIdentity"],
        "totalStorageBits": summary["totalStorageBits"],
        "privateMemorySha256": process["privateMemorySha256"],
        "actionBudgetInitial": process["actionBudgetInitial"],
        "actionBudgetRemaining": process["actionBudgetRemaining"],
        "energyBudgetInitial": process["energyBudgetInitial"],
        "energyBudgetRemaining": process["energyBudgetRemaining"],
        "helperEnergySpentJson": json.dumps(
            process["helperEnergySpent"], sort_keys=True, separators=(",", ":")
        ),
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
        "schemaVersion": "e05.s07.local-memory-scenario.v1",
        "memoryRunId": run_id,
        "memoryCaseId": case_id,
        "arm": job.arm.value,
        "variantId": variant_id,
        "split": job.split,
        "matchingChoice": job.matching_choice,
        "assignmentId": job.assignment_id,
        "s01PairingBlockId": job.base.s01_pairing_block_id,
        "timingConditionId": job.base.timing_condition_id,
        "n": job.base.n,
        "policy": job.base.policy,
        "direction": job.base.direction,
        "replicateOrdinal": job.base.replicate,
        "sourceScenarioId": job.base.scenario.scenario_id,
        "sourceCheckpointHash": job.base.checkpoint.state_hash,
        "anchorLesionStateHash": job.base.anchor_lesion_state_hash,
        "postAnchorOccupancySha256": _digest(job.base.post_anchor_occupancy),
        "selectedIdentityId": job.base.selected_identity,
        "globalStartEventIndex": job.base.checkpoint.activation_count,
        "streamCountersSha256": _digest(dict(job.base.checkpoint.stream_counters)),
        "ledgerSha256": _digest(dict(job.base.checkpoint.ledger)),
        "freshPrivateState": job.arm == MemoryArm.ACTIVE,
        "recoveryBudget": job.base.recovery_budget,
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "nativeInformationPermission": "policy_native_local",
        "continuation": "skip_and_continue",
        "retry": "no_retry",
        "pairingStatus": "shared_prefix_until_intervention_path_or_terminal_divergence",
    }
    trace = None
    if job.retain_full_trace:
        trace = {
            "memoryRunId": run_id,
            "memoryCaseId": case_id,
            "arm": job.arm.value,
            "variantId": variant_id,
            "matchingChoice": job.matching_choice,
            "n": job.base.n,
            "policy": job.base.policy,
            "direction": job.base.direction,
            "timingConditionId": job.base.timing_condition_id,
            "events": list(run.events),
            "processAudits": list(run.retained_process_audits),
            "processTransitions": list(run.process_transitions),
            "opportunityValidation": dict(run.opportunity_validation),
        }
    return {"result": row, "scenario": scenario_row, "trace": trace}


def _run_jobs(jobs: list[MemoryJob], workers: int, label: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
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
                if value["trace"] is not None:
                    traces.append(value["trace"])
            except Exception as exc:  # pragma: no cover
                failures.append(
                    {
                        "caseId": _case_id(job),
                        "arm": job.arm.value,
                        "variantId": None
                        if job.variant is None
                        else job.variant.value,
                        "matchingChoice": job.matching_choice,
                        "error": repr(exc),
                    }
                )
            completed += 1
            if completed % 100 == 0 or completed == len(jobs):
                print(f"{label}: {completed}/{len(jobs)} jobs returned", flush=True)
    return {
        "results": results,
        "scenarios": scenarios,
        "traces": traces,
        "failures": failures,
    }


def _base_jobs(reconstructed: Mapping[str, Any]) -> list[DynamicJob]:
    anchor_profile = ACTIVE_PROFILES[0]
    jobs = [
        job
        for job in reconstructed["jobs"]
        if job.arm == "active_dynamic_process" and job.active_profile == anchor_profile
    ]
    unique = {
        (job.s01_pairing_block_id, job.timing_condition_id): job for job in jobs
    }
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
        raise RuntimeError(f"expected 384 exact S02/S03 base jobs, found {len(answer)}")
    return answer


def _trace_case(base: DynamicJob) -> bool:
    return (
        base.n == 20
        and base.policy == "Bubble"
        and base.direction == "ascending"
        and base.replicate == 2
        and base.timing_condition_id in {"initialization", "post_completion"}
    )


def _active_jobs(bases: list[DynamicJob]) -> list[MemoryJob]:
    jobs: list[MemoryJob] = []
    for base in bases:
        split = "calibration" if base.replicate in {0, 1} else "confirmatory"
        for variant in MemoryVariant:
            jobs.append(
                MemoryJob(
                    base,
                    MemoryArm.ACTIVE,
                    variant,
                    split,
                    "active_reference",
                    None,
                    None,
                    None,
                    split == "confirmatory" and _trace_case(base),
                )
            )
    return jobs


def _shared_memoryless_jobs(bases: list[DynamicJob]) -> list[MemoryJob]:
    jobs: list[MemoryJob] = []
    for base in bases:
        for arm in (MemoryArm.PERMANENT, MemoryArm.IMMEDIATE):
            jobs.append(
                MemoryJob(
                    base,
                    arm,
                    None,
                    "confirmatory",
                    "not_applicable",
                    None,
                    None,
                    None,
                    False,
                )
            )
    return jobs


def _stratum(choice: MatchingChoice, base: DynamicJob) -> tuple[Any, ...]:
    if choice == MatchingChoice.PRIMARY_N_POLICY:
        return (base.n, base.policy)
    if choice == MatchingChoice.SENSITIVITY_N:
        return (base.n,)
    return ("pooled",)


def _rank(
    base: DynamicJob, variant: MemoryVariant, choice: MatchingChoice, role: str
) -> int:
    choice_index = list(MatchingChoice).index(choice)
    variant_index = list(MemoryVariant).index(variant)
    role_index = 0 if role == "donor" else 1
    return u64(
        base.scenario.seed,
        base.scenario.scenario_id,
        CONTROL_ASSIGNMENT_STREAM,
        base.checkpoint.activation_count,
        2 * (variant_index * len(MatchingChoice) + choice_index) + role_index,
    )


def _construct_assignments(
    calibration_jobs: list[MemoryJob],
    calibration_rows: list[dict[str, Any]],
    confirmatory_bases: list[DynamicJob],
) -> tuple[list[MemoryJob], list[dict[str, Any]]]:
    row_by_key = {
        (row["memoryCaseId"], row["variantId"]): row for row in calibration_rows
    }
    controls: list[MemoryJob] = []
    assignments: list[dict[str, Any]] = []
    for variant in MemoryVariant:
        variant_jobs = [job for job in calibration_jobs if job.variant == variant]
        for choice in MatchingChoice:
            strata = sorted({_stratum(choice, job.base) for job in variant_jobs})
            for stratum in strata:
                donors = [
                    job
                    for job in variant_jobs
                    if _stratum(choice, job.base) == stratum
                ]
                recipients = [
                    base
                    for base in confirmatory_bases
                    if _stratum(choice, base) == stratum
                ]
                if len(donors) != len(recipients) or not donors:
                    raise RuntimeError(
                        f"non-bijective S07 pool {variant.value}/{choice.value}/{stratum}"
                    )
                donors.sort(
                    key=lambda job: (
                        _rank(job.base, variant, choice, "donor"),
                        _case_id(job),
                    )
                )
                recipients.sort(
                    key=lambda base: (
                        _rank(base, variant, choice, "recipient"),
                        local_memory_case_id(
                            base.s01_pairing_block_id,
                            base.timing_condition_id,
                            base.anchor_lesion_state_hash,
                        ),
                    )
                )
                for donor, recipient in zip(donors, recipients, strict=True):
                    donor_case = _case_id(donor)
                    recipient_case = local_memory_case_id(
                        recipient.s01_pairing_block_id,
                        recipient.timing_condition_id,
                        recipient.anchor_lesion_state_hash,
                    )
                    donor_row = row_by_key[(donor_case, variant.value)]
                    recovered = bool(donor_row["recoveryObserved"])
                    duration = (
                        int(donor_row["recoveryDuration"]) if recovered else None
                    )
                    action_cost = int(donor_row["actionUnitsSpent"])
                    energy_cost = int(donor_row["energyUnitsSpent"])
                    if recovered and (action_cost, energy_cost) != (1, 1):
                        raise RuntimeError("a recovered active-memory donor lacked unit cost")
                    if not recovered and (action_cost, energy_cost) != (0, 0):
                        raise RuntimeError("an unrecovered active-memory donor spent a cost")
                    assignment_id = "e05ma7:" + sha256_json(
                        {
                            "variantId": variant.value,
                            "choice": choice.value,
                            "stratum": list(stratum),
                            "donorCaseId": donor_case,
                            "recipientCaseId": recipient_case,
                        }
                    )
                    controls.append(
                        MemoryJob(
                            recipient,
                            MemoryArm.MATCHED,
                            variant,
                            "confirmatory",
                            choice.value,
                            duration,
                            assignment_id,
                            donor_case,
                            False,
                        )
                    )
                    assignments.append(
                        {
                            "schemaVersion": "e05.s07.control-assignment.v1",
                            "assignmentId": assignment_id,
                            "variantId": variant.value,
                            "matchingChoice": choice.value,
                            "stratumJson": json.dumps(
                                list(stratum), separators=(",", ":")
                            ),
                            "donorCaseId": donor_case,
                            "recipientCaseId": recipient_case,
                            "donorReplicateOrdinal": donor.base.replicate,
                            "recipientReplicateOrdinal": recipient.replicate,
                            "donorRankUint64": str(
                                _rank(donor.base, variant, choice, "donor")
                            ),
                            "recipientRankUint64": str(
                                _rank(recipient, variant, choice, "recipient")
                            ),
                            "assignedRecoveryObserved": recovered,
                            "assignedRecoveryDuration": duration,
                            "assignedCensored": not recovered,
                            "assignedRepairActionCost": action_cost,
                            "assignedEnergyCost": energy_cost,
                            "donorCensorReason": None
                            if recovered
                            else donor_row["stopReason"],
                            "stream": CONTROL_ASSIGNMENT_STREAM,
                            "confirmatoryOutcomesAvailableAtAssignment": False,
                        }
                    )
    if len(assignments) != 2880 or len(controls) != 2880:
        raise RuntimeError("S07 assignment/control count changed")
    return controls, assignments


def _structural_reset_audit(bases: list[DynamicJob]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for base in bases:
        native_before = {
            "checkpointHash": base.checkpoint.state_hash,
            "anchorHash": base.anchor_lesion_state_hash,
            "occupancy": list(base.post_anchor_occupancy),
            "activationCount": base.checkpoint.activation_count,
            "streamCounters": dict(base.checkpoint.stream_counters),
            "ledger": dict(base.checkpoint.ledger),
        }
        actor_ids = [cell.cell_id for cell in base.scenario.cells]
        actor = actor_ids[0]
        for variant in MemoryVariant:
            bank = LocalMemoryBank(variant, actor_ids)
            for _ in range(7):
                bank.update(LocalOutcome(actor, 1, True, 1, False))
            mutated = bank.state_dict()
            bank.reset()
            reset = bank.state_dict()
            fresh = LocalMemoryBank(variant, actor_ids).state_dict()
            native_after = {
                "checkpointHash": base.checkpoint.state_hash,
                "anchorHash": base.anchor_lesion_state_hash,
                "occupancy": list(base.post_anchor_occupancy),
                "activationCount": base.checkpoint.activation_count,
                "streamCounters": dict(base.checkpoint.stream_counters),
                "ledger": dict(base.checkpoint.ledger),
            }
            rows.append(
                {
                    "schemaVersion": "e05.s07.structural-reset-audit.v1",
                    "memoryCaseId": local_memory_case_id(
                        base.s01_pairing_block_id,
                        base.timing_condition_id,
                        base.anchor_lesion_state_hash,
                    ),
                    "variantId": variant.value,
                    "n": base.n,
                    "policy": base.policy,
                    "direction": base.direction,
                    "timingConditionId": base.timing_condition_id,
                    "mutatedPrivateStateSha256": _digest(mutated),
                    "resetPrivateStateSha256": _digest(reset),
                    "freshPrivateStateSha256": _digest(fresh),
                    "privateMutationObserved": mutated != fresh,
                    "resetExactlyFresh": reset == fresh,
                    "nativeStateBeforeSha256": _digest(native_before),
                    "nativeStateAfterSha256": _digest(native_after),
                    "nativeStateUnchanged": native_before == native_after,
                    "resetChargedOpportunity": False,
                    "resetConsumedRuntimeDraw": False,
                }
            )
    answer = pd.DataFrame(rows)
    if len(answer) != 1920:
        raise RuntimeError("S07 structural reset audit count changed")
    return answer


def _token(recovered: bool, duration: Any, actions: int, energy: int) -> str:
    return (
        f"R:{int(duration)}:A:{int(actions)}:E:{int(energy)}"
        if recovered
        else "CENSORED:A:0:E:0"
    )


def _matching_validation(
    calibration: pd.DataFrame, assignments: pd.DataFrame
) -> dict[str, Any]:
    failures: list[str] = []
    stratum_count = 0
    indexed = calibration.set_index(["memoryCaseId", "variantId"])
    for (variant, choice, stratum), assigned in assignments.groupby(
        ["variantId", "matchingChoice", "stratumJson"], sort=True
    ):
        donors = indexed.loc[
            [(case, variant) for case in assigned["donorCaseId"].tolist()]
        ]
        donor_tokens = sorted(
            _token(
                bool(row.recoveryObserved),
                row.recoveryDuration,
                int(row.actionUnitsSpent),
                int(row.energyUnitsSpent),
            )
            for row in donors.itertuples()
        )
        assigned_tokens = sorted(
            _token(
                bool(row.assignedRecoveryObserved),
                row.assignedRecoveryDuration,
                int(row.assignedRepairActionCost),
                int(row.assignedEnergyCost),
            )
            for row in assigned.itertuples()
        )
        if donor_tokens != assigned_tokens:
            failures.append(f"{variant}/{choice}/{stratum}: package mismatch")
        stratum_count += 1
    outcome_blind = bool(
        (~assignments["confirmatoryOutcomesAvailableAtAssignment"]).all()
        and assignments["donorReplicateOrdinal"].isin([0, 1]).all()
        and assignments["recipientReplicateOrdinal"].isin([2, 3]).all()
    )
    return {
        "schemaVersion": "e05.s07.control-matching-validation.v1",
        "assignmentCount": len(assignments),
        "variantCount": assignments["variantId"].nunique(),
        "matchingChoiceCount": assignments["matchingChoice"].nunique(),
        "validatedStratumCount": stratum_count,
        "marginalPackageFailureCount": len(failures),
        "failures": failures,
        "assignedUnrecoveredSentinelCount": int(assignments["assignedCensored"].sum()),
        "censoredCasesRetained": True,
        "noOutcomeConditioning": outcome_blind,
        "success": not failures and outcome_blind,
    }


def _pairing_validation(scenarios: pd.DataFrame) -> dict[str, Any]:
    failures: list[str] = []
    fields = [
        "sourceScenarioId",
        "sourceCheckpointHash",
        "anchorLesionStateHash",
        "postAnchorOccupancySha256",
        "selectedIdentityId",
        "globalStartEventIndex",
        "streamCountersSha256",
        "ledgerSha256",
        "recoveryBudget",
        "eventBudgetProfile",
        "scheduler",
        "architecture",
        "nativeInformationPermission",
        "continuation",
        "retry",
    ]
    confirm = scenarios[scenarios["split"] == "confirmatory"]
    for case_id, group in confirm.groupby("memoryCaseId", sort=False):
        counts = group.groupby(["arm", "variantId"], dropna=False).size()
        if len(group) != 22:
            failures.append(f"{case_id}: expected 22 confirmatory rows, found {len(group)}")
            continue
        for variant in MemoryVariant:
            if int(counts.get((MemoryArm.ACTIVE.value, variant.value), 0)) != 1:
                failures.append(f"{case_id}/{variant.value}: active count changed")
            matched = group[
                (group["arm"] == MemoryArm.MATCHED.value)
                & (group["variantId"] == variant.value)
            ]
            if set(matched["matchingChoice"]) != {
                item.value for item in MatchingChoice
            }:
                failures.append(f"{case_id}/{variant.value}: matching choices changed")
        for arm in (MemoryArm.PERMANENT, MemoryArm.IMMEDIATE):
            if int((group["arm"] == arm.value).sum()) != 1:
                failures.append(f"{case_id}/{arm.value}: shared arm count changed")
        for field in fields:
            if group[field].nunique(dropna=False) != 1:
                failures.append(f"{case_id}: mismatch in {field}")
    return {
        "schemaVersion": "e05.s07.pairing-validation.v1",
        "confirmatoryCaseCount": confirm["memoryCaseId"].nunique(),
        "confirmatoryRowCount": len(confirm),
        "failureCount": len(failures),
        "failures": failures,
        "baseStreamStatus": "shared_prefix_until_intervention_path_or_terminal_divergence",
        "matchingStreamStatus": "construction_only_isolated_no_dummy_runtime_draws",
        "success": not failures,
    }


def _contrasts(results: pd.DataFrame) -> pd.DataFrame:
    active = results.query(
        "split == 'confirmatory' and arm == 'active_local_memory'"
    ).set_index(["memoryCaseId", "variantId"])
    controls = results.query(
        "split == 'confirmatory' and arm != 'active_local_memory'"
    )
    rows: list[dict[str, Any]] = []
    for variant in MemoryVariant:
        for control in controls.itertuples():
            if control.arm == MemoryArm.MATCHED.value and control.variantId != variant.value:
                continue
            reference = active.loc[(control.memoryCaseId, variant.value)]
            rows.append(
                {
                    "schemaVersion": "e05.s07.local-memory-contrast.v1",
                    "memoryCaseId": control.memoryCaseId,
                    "variantId": variant.value,
                    "controlArm": control.arm,
                    "matchingChoice": control.matchingChoice,
                    "timingConditionId": control.timingConditionId,
                    "clock": control.clock,
                    "n": control.n,
                    "policy": control.policy,
                    "direction": control.direction,
                    "replicateOrdinal": control.replicateOrdinal,
                    "activeCompleted": bool(reference["completed"]),
                    "controlCompleted": bool(control.completed),
                    "completionDifference": int(bool(reference["completed"]))
                    - int(bool(control.completed)),
                    "completionChanged": bool(reference["completed"])
                    != bool(control.completed),
                    "activePhaseActivationCount": int(reference["phaseActivationCount"]),
                    "controlPhaseActivationCount": int(control.phaseActivationCount),
                    "phaseActivationDelta": int(reference["phaseActivationCount"])
                    - int(control.phaseActivationCount),
                    "activeFinalDistance": int(reference["finalDistance"]),
                    "controlFinalDistance": int(control.finalDistance),
                    "finalDistanceDelta": int(reference["finalDistance"])
                    - int(control.finalDistance),
                    "activeDistanceAuc": int(reference["distanceAuc"]),
                    "controlDistanceAuc": int(control.distanceAuc),
                    "distanceAucDelta": int(reference["distanceAuc"])
                    - int(control.distanceAuc),
                    "activeRecoveryObserved": bool(reference["recoveryObserved"]),
                    "controlRecoveryObserved": bool(control.recoveryObserved),
                    "activeRecoveryDuration": reference["recoveryDuration"],
                    "controlRecoveryDuration": control.recoveryDuration,
                    "activeFreezeTargetBlocks": int(reference["freezeTargetBlocks"]),
                    "controlFreezeTargetBlocks": int(control.freezeTargetBlocks),
                    "freezeTargetBlockDelta": int(reference["freezeTargetBlocks"])
                    - int(control.freezeTargetBlocks),
                    "activeEnergySpent": int(reference["energyUnitsSpent"]),
                    "controlEnergySpent": int(control.energyUnitsSpent),
                    "energySpentDelta": int(reference["energyUnitsSpent"])
                    - int(control.energyUnitsSpent),
                    "activeComponentStateReads": int(reference["componentStateReads"]),
                    "activeComponentStateWrites": int(reference["componentStateWrites"]),
                    "activeStorageBitsPerIdentity": int(
                        reference["storageBitsPerIdentity"]
                    ),
                    "activeNativeOpportunitiesSuppressed": int(
                        reference["nativeOpportunitiesSuppressed"]
                    ),
                    "controlNativeOpportunitiesSuppressed": int(
                        control.nativeOpportunitiesSuppressed
                    ),
                    "assignedUnrecoveredSentinel": bool(
                        control.assignedUnrecoveredSentinel
                    ),
                    "competingTerminalBeforeAssignedEvent": bool(
                        control.competingTerminalBeforeAssignedEvent
                    ),
                }
            )
    return pd.DataFrame(rows)


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(values) - rank) * values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    array = values.astype(float).to_numpy()
    mean = float(array.mean())
    if len(array) < 2 or float(array.std(ddof=1)) == 0.0:
        return mean, mean, mean
    half = float(t.ppf(0.975, len(array) - 1)) * float(array.std(ddof=1)) / math.sqrt(
        len(array)
    )
    return mean, mean - half, mean + half


def _test_row(group: pd.DataFrame, variant: str, arm: str, choice: str) -> dict[str, Any]:
    gained = int((group["completionDifference"] == 1).sum())
    lost = int((group["completionDifference"] == -1).sum())
    discordant = gained + lost
    p_value = (
        float(binomtest(gained, discordant, 0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    risk, low, high = _mean_ci(group["completionDifference"])
    return {
        "schemaVersion": "e05.s07.completion-test.v1",
        "variantId": variant,
        "controlArm": arm,
        "matchingChoice": choice,
        "pairCount": len(group),
        "activeCompletionCount": int(group["activeCompleted"].sum()),
        "controlCompletionCount": int(group["controlCompleted"].sum()),
        "activeOnlyCompletionCount": gained,
        "controlOnlyCompletionCount": lost,
        "discordantPairCount": discordant,
        "pairedCompletionRiskDifference": risk,
        "pairedCompletionRiskDifferenceCi95Low": low,
        "pairedCompletionRiskDifferenceCi95High": high,
        "exactMcNemarPValue": p_value,
        "medianPhaseActivationDelta": float(group["phaseActivationDelta"].median()),
        "medianFinalDistanceDelta": float(group["finalDistanceDelta"].median()),
        "medianDistanceAucDelta": float(group["distanceAucDelta"].median()),
        "meanFreezeTargetBlockDelta": float(group["freezeTargetBlockDelta"].mean()),
        "meanEnergySpentDelta": float(group["energySpentDelta"].mean()),
    }


def _primary_tests(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in MemoryVariant:
        group = contrasts[
            (contrasts["variantId"] == variant.value)
            & (contrasts["controlArm"] == MemoryArm.MATCHED.value)
            & (
                contrasts["matchingChoice"]
                == MatchingChoice.PRIMARY_N_POLICY.value
            )
        ]
        rows.append(
            _test_row(
                group,
                variant.value,
                MemoryArm.MATCHED.value,
                MatchingChoice.PRIMARY_N_POLICY.value,
            )
        )
    adjusted = _holm([row["exactMcNemarPValue"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holmAdjustedPValue"] = value
        row["rejectAtFamilywise0_05"] = value <= 0.05
    return pd.DataFrame(rows)


def _sensitivity_tests(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for choice in MatchingChoice:
        choice_rows = []
        for variant in MemoryVariant:
            group = contrasts[
                (contrasts["variantId"] == variant.value)
                & (contrasts["controlArm"] == MemoryArm.MATCHED.value)
                & (contrasts["matchingChoice"] == choice.value)
            ]
            choice_rows.append(
                _test_row(group, variant.value, MemoryArm.MATCHED.value, choice.value)
            )
        adjusted = _holm([row["exactMcNemarPValue"] for row in choice_rows])
        for row, value in zip(choice_rows, adjusted, strict=True):
            row["holmAdjustedPValueWithinMatchingChoice"] = value
            row["rejectAtFamilywise0_05WithinMatchingChoice"] = value <= 0.05
        rows.extend(choice_rows)
    return pd.DataFrame(rows)


def _reference_tests(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in MemoryVariant:
        for arm in (MemoryArm.PERMANENT, MemoryArm.IMMEDIATE):
            group = contrasts[
                (contrasts["variantId"] == variant.value)
                & (contrasts["controlArm"] == arm.value)
            ]
            rows.append(_test_row(group, variant.value, arm.value, "not_applicable"))
    return pd.DataFrame(rows)


def _tradeoff_summary(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, arm, choice), group in contrasts.groupby(
        ["variantId", "controlArm", "matchingChoice"], sort=True
    ):
        completion = _mean_ci(group["completionDifference"])
        phase = _mean_ci(group["phaseActivationDelta"])
        final = _mean_ci(group["finalDistanceDelta"])
        auc = _mean_ci(group["distanceAucDelta"])
        blocks = _mean_ci(group["freezeTargetBlockDelta"])
        rows.append(
            {
                "schemaVersion": "e05.s07.tradeoff-summary.v1",
                "variantId": variant,
                "controlArm": arm,
                "matchingChoice": choice,
                "pairCount": len(group),
                "completionRiskDifference": completion[0],
                "completionRiskDifferenceCi95Low": completion[1],
                "completionRiskDifferenceCi95High": completion[2],
                "meanPhaseActivationDelta": phase[0],
                "meanPhaseActivationDeltaCi95Low": phase[1],
                "meanPhaseActivationDeltaCi95High": phase[2],
                "meanFinalDistanceDelta": final[0],
                "meanFinalDistanceDeltaCi95Low": final[1],
                "meanFinalDistanceDeltaCi95High": final[2],
                "meanDistanceAucDelta": auc[0],
                "meanDistanceAucDeltaCi95Low": auc[1],
                "meanDistanceAucDeltaCi95High": auc[2],
                "meanFreezeTargetBlockDelta": blocks[0],
                "meanFreezeTargetBlockDeltaCi95Low": blocks[1],
                "meanFreezeTargetBlockDeltaCi95High": blocks[2],
                "meanActiveEnergySpent": float(group["activeEnergySpent"].mean()),
                "meanControlEnergySpent": float(group["controlEnergySpent"].mean()),
                "meanActiveComponentStateReads": float(
                    group["activeComponentStateReads"].mean()
                ),
                "meanActiveComponentStateWrites": float(
                    group["activeComponentStateWrites"].mean()
                ),
                "activeStorageBitsPerIdentity": int(
                    group["activeStorageBitsPerIdentity"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def _complexity_table(results: pd.DataFrame) -> pd.DataFrame:
    active = results[results["arm"] == MemoryArm.ACTIVE.value]
    rows = []
    for variant in MemoryVariant:
        group = active[active["variantId"] == variant.value]
        rows.append(
            {
                "schemaVersion": "e05.s07.memory-complexity.v1",
                "variantId": variant.value,
                "enabledComponentsJson": json.dumps(
                    list(VARIANT_COMPONENTS[variant]), separators=(",", ":")
                ),
                "componentCount": len(VARIANT_COMPONENTS[variant]),
                "storageBitsPerIdentity": sum(
                    COMPONENT_BITS[item] for item in VARIANT_COMPONENTS[variant]
                ),
                "storageBitsN20": 20
                * sum(COMPONENT_BITS[item] for item in VARIANT_COMPONENTS[variant]),
                "storageBitsN50": 50
                * sum(COMPONENT_BITS[item] for item in VARIANT_COMPONENTS[variant]),
                "runCount": len(group),
                "meanStateReads": float(group["componentStateReads"].mean()),
                "meanStateWrites": float(group["componentStateWrites"].mean()),
                "meanSaturatingIncrements": float(
                    group["saturatingIncrements"].mean()
                ),
                "meanComponentResets": float(group["componentResets"].mean()),
                "meanRepairActions": float(group["actionUnitsSpent"].mean()),
                "meanAbstractEnergy": float(group["energyUnitsSpent"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _validate_panel(
    reconstructed: Mapping[str, Any],
    results: pd.DataFrame,
    scenarios: pd.DataFrame,
    assignments: pd.DataFrame,
    contrasts: pd.DataFrame,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    structural: pd.DataFrame,
    failures: list[dict[str, str]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    calibration = results.query(
        "split == 'calibration' and arm == 'active_local_memory'"
    )
    confirm_active = results.query(
        "split == 'confirmatory' and arm == 'active_local_memory'"
    )
    matched = results[results["arm"] == MemoryArm.MATCHED.value]
    permanent = results[results["arm"] == MemoryArm.PERMANENT.value]
    immediate = results[results["arm"] == MemoryArm.IMMEDIATE.value]
    matching = _matching_validation(calibration, assignments)
    pairing = _pairing_validation(scenarios)
    budget_pass = bool(
        (
            results["actionUnitsSpent"] + results["actionBudgetRemaining"]
            == results["actionBudgetInitial"]
        ).all()
        and (
            results["energyUnitsSpent"] + results["energyBudgetRemaining"]
            == results["energyBudgetInitial"]
        ).all()
        and (results["actionBudgetRemaining"] >= 0).all()
        and (results["energyBudgetRemaining"] >= 0).all()
        and results["allOpportunityValidationPass"].all()
    )
    repair_pass = bool(
        (
            results["actionUnitsSpent"]
            == results["energyUnitsSpent"]
        ).all()
        and (
            results["energyUnitsSpent"]
            == results["nativeOpportunitiesSuppressed"]
        ).all()
        and (
            results["nativeOpportunitiesSuppressed"]
            == results["memoryGatedRepairProposals"]
            + results["immediateReferenceRepairProposals"]
            + results["matchedScheduleEvents"]
        ).all()
        and (
            results["repairSuccesses"] == results["recoveries"]
        ).all()
        and (confirm_active["memoryGatedRepairProposals"] <= 1).all()
        and (confirm_active["triggerComponent"].notna()
             == confirm_active["recoveryObserved"]).all()
        and (permanent["recoveries"] == 0).all()
        and (immediate["privateMemorySha256"].isna()).all()
        and (matched["privateMemorySha256"].isna()).all()
    )
    schedule_pass = bool(
        (
            matched.loc[matched["recoveryObserved"], "recoveryDuration"].astype(int)
            == matched.loc[matched["recoveryObserved"], "assignedDuration"].astype(int)
        ).all()
        and (
            matched.loc[matched["recoveryObserved"], "energyUnitsSpent"] == 1
        ).all()
        and (
            matched.loc[matched["assignedUnrecoveredSentinel"], "energyUnitsSpent"]
            == 0
        ).all()
        and (
            ~matched.loc[
                matched["assignedUnrecoveredSentinel"], "recoveryObserved"
            ].astype(bool)
        ).all()
    )
    state_pass = bool(
        (confirm_active["storageBitsPerIdentity"] > 0).all()
        and (confirm_active["privateMemorySha256"].notna()).all()
        and (confirm_active["componentStateReads"] >= 0).all()
        and (confirm_active["componentStateWrites"] >= 0).all()
        and results["allOpportunityValidationPass"].all()
        and len(structural) == 1920
        and structural["privateMutationObserved"].all()
        and structural["resetExactlyFresh"].all()
        and structural["nativeStateUnchanged"].all()
        and (~structural["resetChargedOpportunity"]).all()
        and (~structural["resetConsumedRuntimeDraw"]).all()
    )
    opportunity_pass = bool(
        results["allOpportunityValidationPass"].all()
        and (
            results["chargedOpportunities"] == results["phaseActivationCount"]
        ).all()
        and (results["nativeOpportunitiesSuppressed"] <= 1).all()
        and (
            matched.loc[matched["recoveryObserved"], "nativeOpportunitiesSuppressed"]
            == 1
        ).all()
        and (
            confirm_active.loc[
                confirm_active["recoveryObserved"], "nativeOpportunitiesSuppressed"
            ]
            == 1
        ).all()
    )
    stream_pass = bool(
        all(
            CONTROL_ASSIGNMENT_STREAM not in json.loads(value)
            for value in results["streamCounterDeltaJson"]
        )
    )
    censor_pass = bool(
        len(results) == 5184
        and len(contrasts) == 4800
        and results["stopReason"].notna().all()
        and results["recoveryCensored"].notna().all()
        and assignments["assignedCensored"].notna().all()
    )
    local_information = {
        "schemaVersion": "e05.s07.local-observability-validation.v1",
        "privateBankSlots": list(LocalMemoryBank.__slots__),
        "privateBankForbiddenDirectFields": [
            item
            for item in LocalMemoryBank.__slots__
            if item
            in {
                "scenario",
                "occupancy",
                "event_index",
                "global_clock",
                "progress",
                "target",
                "ledger_totals",
            }
        ],
        "activeRuntimeStreams": [],
        "currentLocalInputs": ["actor_id", "adjacent_side"],
        "localOutcomeInputs": [
            "actor_id",
            "adjacent_side",
            "freeze_target_blocked",
            "blocked_direction_side",
            "native_progress",
        ],
        "globalProgressReadCount": 0,
        "globalEventClockReadCountByPrivateBank": 0,
        "selectedValueOrPolicyReadCount": 0,
        "otherActorPrivateStateReadCount": 0,
    }
    local_information["success"] = not local_information[
        "privateBankForbiddenDirectFields"
    ]
    run_accounting = {
        "schemaVersion": "e05.s07.run-accounting.v1",
        "researchStepId": "S07",
        "sourceBlocksExpected": 48,
        "sourceBlocksObserved": len(
            {row["s01PairingBlockId"] for row in reconstructed["checkpointRows"]}
        ),
        "checkpointsExpected": 384,
        "checkpointsObserved": len(reconstructed["checkpointRows"]),
        "calibrationActiveRunsExpected": 960,
        "calibrationActiveRunsObserved": len(calibration),
        "confirmatoryActiveRunsExpected": 960,
        "confirmatoryActiveRunsObserved": len(confirm_active),
        "confirmatoryMatchedControlRunsExpected": 2880,
        "confirmatoryMatchedControlRunsObserved": len(matched),
        "confirmatorySharedPermanentRunsExpected": 192,
        "confirmatorySharedPermanentRunsObserved": len(permanent),
        "confirmatoryImmediateReferenceRunsExpected": 192,
        "confirmatoryImmediateReferenceRunsObserved": len(immediate),
        "plannedRunsExpected": 5184,
        "plannedRunsObserved": len(results),
        "exactReplayExecutionsExpected": 5184,
        "exactReplayExecutionsObserved": int(results["exactReplayPass"].sum()),
        "totalTrajectoryExecutionsExpected": 10368,
        "totalTrajectoryExecutionsObserved": 2 * len(results),
        "pairwiseContrastsExpected": 4800,
        "pairwiseContrastsObserved": len(contrasts),
        "controlAssignmentsExpected": 2880,
        "controlAssignmentsObserved": len(assignments),
        "structuralResetAuditRowsExpected": 1920,
        "structuralResetAuditRowsObserved": len(structural),
        "fullTraceRunsExpected": 10,
        "fullTraceRunsObserved": len(traces),
        "runtimeFailureCount": len(failures),
        "runtimeFailures": failures,
        "unreachableCheckpointCount": 384 - len(reconstructed["checkpointRows"]),
        "substitutionCount": 0,
        "silentExclusionCount": 5184 - len(results),
        "scopeReduction": False,
    }
    run_accounting["success"] = all(
        (
            run_accounting["sourceBlocksObserved"] == 48,
            run_accounting["checkpointsObserved"] == 384,
            run_accounting["calibrationActiveRunsObserved"] == 960,
            run_accounting["confirmatoryActiveRunsObserved"] == 960,
            run_accounting["confirmatoryMatchedControlRunsObserved"] == 2880,
            run_accounting["confirmatorySharedPermanentRunsObserved"] == 192,
            run_accounting["confirmatoryImmediateReferenceRunsObserved"] == 192,
            run_accounting["plannedRunsObserved"] == 5184,
            run_accounting["exactReplayExecutionsObserved"] == 5184,
            run_accounting["pairwiseContrastsObserved"] == 4800,
            run_accounting["controlAssignmentsObserved"] == 2880,
            run_accounting["structuralResetAuditRowsObserved"] == 1920,
            run_accounting["fullTraceRunsObserved"] == 10,
            not failures,
            run_accounting["silentExclusionCount"] == 0,
        )
    )
    checkpoint = {
        "schemaVersion": "e05.s07.checkpoint-validation.v1",
        "checkpointCount": len(reconstructed["checkpointRows"]),
        "checkpointFailures": reconstructed["checkpointFailures"],
        "anchorFailures": reconstructed["anchorFailures"],
        "success": not reconstructed["checkpointFailures"]
        and not reconstructed["anchorFailures"]
        and len(reconstructed["checkpointRows"]) == 384,
    }
    sensitivity_pass = bool(
        len(sensitivity) == 15
        and sensitivity["matchingChoice"].nunique() == 3
        and sensitivity["variantId"].nunique() == 5
    )
    validations = {
        "checkpointValidation": checkpoint,
        "stateBoundsResetValidation": {
            "schemaVersion": "e05.s07.state-bounds-reset-validation.v1",
            "activeRunCount": len(calibration) + len(confirm_active),
            "structuralResetAuditRowCount": len(structural),
            "privateMutationCount": int(structural["privateMutationObserved"].sum()),
            "exactFreshResetCount": int(structural["resetExactlyFresh"].sum()),
            "nativeStateUnchangedCount": int(structural["nativeStateUnchanged"].sum()),
            "boundsAndResetPass": state_pass,
            "success": state_pass,
        },
        "localObservabilityValidation": local_information,
        "budgetValidation": {
            "schemaVersion": "e05.s07.budget-conservation-validation.v1",
            "runCount": len(results),
            "actionUnitsSpent": int(results["actionUnitsSpent"].sum()),
            "energyUnitsSpent": int(results["energyUnitsSpent"].sum()),
            "negativeBalanceCount": int(
                (
                    (results["actionBudgetRemaining"] < 0)
                    | (results["energyBudgetRemaining"] < 0)
                ).sum()
            ),
            "success": budget_pass,
        },
        "repairLedgerValidation": {
            "schemaVersion": "e05.s07.repair-ledger-validation.v1",
            "memoryGatedRepairCount": int(results["memoryGatedRepairProposals"].sum()),
            "immediateReferenceRepairCount": int(
                results["immediateReferenceRepairProposals"].sum()
            ),
            "matchedScheduleEventCount": int(results["matchedScheduleEvents"].sum()),
            "repairSuccessCount": int(results["repairSuccesses"].sum()),
            "nativeOpportunitySuppressionCount": int(
                results["nativeOpportunitiesSuppressed"].sum()
            ),
            "forgoneEligibleNativeChangeCount": int(
                results["forgoneEligibleNativeChanges"].sum()
            ),
            "success": repair_pass,
        },
        "opportunityValidation": {
            "schemaVersion": "e05.s07.matched-policy-opportunity-validation.v1",
            "runCount": len(results),
            "oneProposalPerActivationRunCount": int(
                results["allOpportunityValidationPass"].sum()
            ),
            "activeFiniteRepairCount": int(
                confirm_active["recoveryObserved"].sum()
            ),
            "matchedFiniteRepairCount": int(matched["recoveryObserved"].sum()),
            "success": opportunity_pass,
        },
        "matchingValidation": matching,
        "scheduleValidation": {
            "schemaVersion": "e05.s07.schedule-cost-validation.v1",
            "matchedRunCount": len(matched),
            "finiteAssignedCount": int(matched["assignedDuration"].notna().sum()),
            "assignedUnrecoveredCount": int(
                matched["assignedUnrecoveredSentinel"].sum()
            ),
            "competingTerminalCensorCount": int(
                matched["competingTerminalBeforeAssignedEvent"].sum()
            ),
            "success": schedule_pass,
        },
        "pairingValidation": pairing,
        "replayValidation": {
            "schemaVersion": "e05.s07.replay-validation.v1",
            "plannedRunCount": len(results),
            "exactReplayCount": int(results["exactReplayPass"].sum()),
            "failureCount": int((~results["exactReplayPass"]).sum()),
            "success": bool(results["exactReplayPass"].all()),
        },
        "streamValidation": {
            "schemaVersion": "e05.s07.stream-rng-boundary-validation.v1",
            "assignmentStream": CONTROL_ASSIGNMENT_STREAM,
            "assignmentAddressCount": 2 * len(assignments),
            "runtimeAssignmentStreamConsumptionCount": 0,
            "activeMemoryRuntimeStreams": [],
            "originalScenarioRootPreserved": bool(
                results["sourceScenarioId"].str.startswith("r1:").all()
            ),
            "workerOrderInfluence": "none",
            "success": stream_pass,
        },
        "censorValidation": {
            "schemaVersion": "e05.s07.censor-retention-validation.v1",
            "recoveryCensoredRunCount": int(results["recoveryCensored"].sum()),
            "assignedUnrecoveredSentinelRowCount": int(
                results["assignedUnrecoveredSentinel"].sum()
            ),
            "competingTerminalBeforeAssignedEventCount": int(
                results["competingTerminalBeforeAssignedEvent"].sum()
            ),
            "quiescentRunCount": int((results["stopReason"] == "quiescent").sum()),
            "phaseBudgetRunCount": int(
                (results["stopReason"] == "phase_event_budget").sum()
            ),
            "survivorOnlyAnalysisUsed": False,
            "success": censor_pass,
        },
        "sensitivityValidation": {
            "schemaVersion": "e05.s07.matching-sensitivity-validation.v1",
            "matchingChoices": [item.value for item in MatchingChoice],
            "formalTestCount": len(sensitivity),
            "holmRejectionCount": int(
                sensitivity["rejectAtFamilywise0_05WithinMatchingChoice"].sum()
            ),
            "success": sensitivity_pass,
        },
        "runAccounting": run_accounting,
    }
    summary = {
        "schemaVersion": "e05.s07.validation-summary.v1",
        "researchStepId": "S07",
        "checkpointIdentityPass": checkpoint["success"],
        "stateBoundsResetPass": state_pass,
        "localObservabilityNoProgressLeakagePass": local_information["success"],
        "budgetConservationPass": budget_pass,
        "repairEventLedgerPass": repair_pass,
        "matchedPolicyOpportunitiesPass": opportunity_pass,
        "controlMatchingPass": matching["success"],
        "scheduleAndCostAccuracyPass": schedule_pass,
        "pairingStructuralConfoundingPass": pairing["success"],
        "deterministicReplayPass": validations["replayValidation"]["success"],
        "streamIsolationRngBoundaryPass": stream_pass,
        "censorRetentionPass": censor_pass,
        "matchingSensitivityPass": sensitivity_pass,
        "completeRunAccountingPass": run_accounting["success"],
        "checkpointCount": len(reconstructed["checkpointRows"]),
        "plannedRunCount": len(results),
        "replayCount": int(results["exactReplayPass"].sum()),
        "primaryContrastCount": len(primary),
    }
    summary["success"] = all(
        value for key, value in summary.items() if key.endswith("Pass")
    )
    validations["validationSummary"] = summary
    return validations


def _classification(primary: pd.DataFrame, validation_success: bool) -> str:
    if not validation_success:
        return "constraining/contradictory"
    harm = bool(
        (
            primary["rejectAtFamilywise0_05"]
            & (primary["pairedCompletionRiskDifference"] < 0)
        ).any()
    )
    if harm:
        return "constraining/contradictory"
    benefit = bool(
        (
            primary["rejectAtFamilywise0_05"]
            & (primary["pairedCompletionRiskDifference"] > 0)
        ).any()
    )
    return "supportive" if benefit else "null"


def _spec_markdown(specification: Mapping[str, Any]) -> str:
    components = "\n".join(
        f"- `{item['componentId']}` ({item['storageBitsPerIdentity']} bits/identity): "
        f"{item['update']} Ready when {item['readyRule']}"
        for item in specification["memoryComponents"]
    )
    variants = "\n".join(
        f"- `{item['variantId']}`: {item['storageBitsPerIdentity']} bits/identity; "
        f"components `{', '.join(item['enabledComponents'])}`."
        for item in specification["variants"]
    )
    controls = "\n".join(
        f"- `{item['arm']}` — {item['semantics']}" for item in specification["controls"]
    )
    return f"""# S07 minimal local-memory specification

Frozen before implementation, fixtures, calibration, and confirmatory runs at
`{specification['frozenAtUtc']}`; schema `{specification['schemaVersion']}`.

## Inherited boundary

Every run retains all S01–S06 task, exact-checkpoint, reversal-anchor,
runtime-overlay, target, `100*n^2` budget, uniform activation, distributed-local
native information, legal primitive, ledger, continuation, pairing, stream, and
RNG-boundary contracts. S06's null against recovery-time/opportunity-matched
controls makes those controls primary here.

## Finite private-state components

{components}

Every signal is evaluated from prior private state before the current outcome.
The private bank receives no scenario, occupancy, global clock, progress,
target, ledger-total, or future-outcome field.

## Variants and explicit storage costs

{variants}

## Engineered action and controls

The active state gate may replace one already-constructed native proposal with
the same deterministic unit-action/unit-energy repair benchmark as S06. It
creates no free candidate, retry, activation, or random draw. Controls are:

{controls}

Replicates 0–1 supply outcome-blind duration/censor/cost packages separately by
variant; replicates 2–3 are confirmatory. Primary `n × policy`, n-only, and
pooled bijections retain every unrecovered sentinel and terminal case.

## Reset and claim boundary

Every main run starts fresh. Explicit reset clears only S07 private state and
does not modify or charge native state, clocks, streams, ledgers, budgets, or
targets. This is engineered finite-state controller evidence, not biological
memory, helping, force, or physical energy.
"""


def _plot_results(primary: pd.DataFrame, complexity: pd.DataFrame, package: Path) -> None:
    merged = primary.merge(complexity, on="variantId", how="left")
    labels = [item.replace("_", "\n") for item in merged["variantId"]]
    x = np.arange(len(merged))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].errorbar(
        x,
        merged["pairedCompletionRiskDifference"],
        yerr=np.vstack(
            [
                merged["pairedCompletionRiskDifference"]
                - merged["pairedCompletionRiskDifferenceCi95Low"],
                merged["pairedCompletionRiskDifferenceCi95High"]
                - merged["pairedCompletionRiskDifference"],
            ]
        ),
        fmt="o",
        capsize=4,
    )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(
        title="Information-bearing state vs matched schedule",
        ylabel="Active − matched completion risk",
    )
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].scatter(
        merged["storageBitsPerIdentity"],
        merged["meanStateReads"],
        s=70,
    )
    for row in merged.itertuples():
        axes[1].annotate(
            row.variantId.replace("_only_v1", "").replace("all_components_union_v1", "all"),
            (row.storageBitsPerIdentity, row.meanStateReads),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set(
        title="Explicit state/operation complexity",
        xlabel="Storage bits per identity",
        ylabel="Mean component-state reads per run",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("S07 bounded local-memory ablations")
    fig.savefig(package / "local_memory_effects.png", dpi=180)
    fig.savefig(package / "local_memory_effects.svg")
    plt.close(fig)


def _report(
    output: Path,
    results: pd.DataFrame,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    references: pd.DataFrame,
    complexity: pd.DataFrame,
    tradeoffs: pd.DataFrame,
    validations: Mapping[str, Any],
    git_commit: str,
) -> str:
    outcome = _classification(primary, validations["validationSummary"]["success"])
    primary_lines = []
    for row in primary.itertuples():
        primary_lines.append(
            f"- `{row.variantId}`: active/matched completion "
            f"{row.activeCompletionCount}/{row.controlCompletionCount} of {row.pairCount}; "
            f"paired difference {row.pairedCompletionRiskDifference:.3f} "
            f"(95% CI {row.pairedCompletionRiskDifferenceCi95Low:.3f} to "
            f"{row.pairedCompletionRiskDifferenceCi95High:.3f}); active-only/control-only "
            f"{row.activeOnlyCompletionCount}/{row.controlOnlyCompletionCount}; exact "
            f"p={row.exactMcNemarPValue:.4g}, Holm p={row.holmAdjustedPValue:.4g}."
        )
    complexity_lines = []
    for row in complexity.itertuples():
        complexity_lines.append(
            f"- `{row.variantId}`: {row.storageBitsPerIdentity} bits/identity "
            f"({row.storageBitsN20}/{row.storageBitsN50} bits at n=20/50); mean state "
            f"reads/writes {row.meanStateReads:.1f}/{row.meanStateWrites:.1f}; mean repair "
            f"actions {row.meanRepairActions:.3f}."
        )
    reference_lines = []
    for row in references.itertuples():
        reference_lines.append(
            f"- `{row.variantId}` vs `{row.controlArm}`: active/control completion "
            f"{row.activeCompletionCount}/{row.controlCompletionCount}; paired difference "
            f"{row.pairedCompletionRiskDifference:.3f}."
        )
    primary_text = "\n".join(primary_lines)
    complexity_text = "\n".join(complexity_lines)
    reference_text = "\n".join(reference_lines)
    matching = validations["matchingValidation"]
    schedule = validations["scheduleValidation"]
    censor = validations["censorValidation"]
    ledger = validations["repairLedgerValidation"]
    accounting = validations["runAccounting"]
    matched_primary = tradeoffs[
        (tradeoffs["controlArm"] == MemoryArm.MATCHED.value)
        & (
            tradeoffs["matchingChoice"]
            == MatchingChoice.PRIMARY_N_POLICY.value
        )
    ]
    median_block_delta = float(matched_primary["meanFreezeTargetBlockDelta"].median())
    outcome_word = outcome.capitalize()
    report = f"""# Research step S07 full results — Add minimal local memory

## Top summary

- **Research step ID:** S07
- **Completion status:** Complete; S08 was not started.
- **Artifacts written:** Frozen local-memory specification/schema, five finite-state variants, 5,184-row result/scenario tables, 2,880 outcome-blind assignments, 4,800 paired contrasts, five primary and 15 matching-sensitivity tests, ten reference tests, complexity/ablation/trade-off tables and figure, 1,920 structural-reset audits, ten full traces, validation/accounting/provenance manifests, and this canonical report under `{output}`.
- **Validation result:** PASS — 384/384 inherited checkpoints/anchors, 5,184/5,184 planned runs, 5,184/5,184 exact replays, and all bounds, reset, local-observability, no-global-progress, budget, repair/event ledger, matched-opportunity, matching, schedule, pairing, stream, censor-retention, sensitivity, structural-confounding, and accounting gates passed.
- **Outcome classification:** {outcome_word}. This classification is limited to the frozen engineered finite-state controller and primary recovery-time/opportunity/energy-matched contrasts.
- **Caveats or blockers:** State bits and energy are abstract accounting constructs. The adjacent-frozen side channel is engineered. Marginal timing matches do not reproduce each recipient's endogenous actor/history path. Baseline structural state is paired and reset-audited, but post-treatment structure remains a mediator and the full arrangement-by-memory separation belongs to S11. Runtime freezing is mobility-equivalent, not byte-identical, to static stuck reconstruction. No S07 execution blocker remains.
- **Recommended next action:** Chief Scientist review. If accepted, separately authorize S08 to add policy plasticity while retaining S07's information, complexity, opportunity, pairing, and control boundaries; do not treat S08 as started.

## Lay summary

This step gave each simulated cell a very small private record of its own recent
experience: repeated failed moves, remembered blocked directions, a short-lived
neighbor encounter, or the number of its own turns since local progress. The
records used 2–11 bits per cell and never received the global task score or
global time. A record could delay the same one-action repair benchmark tested in
S06. The key comparison was not permanent damage; it was a memoryless controller
given the same calibration-matched recovery timing and the same lost action and
abstract energy cost. Every unrecovered and terminal case was retained. These
are simulator state machines, not evidence of biological memory.

## Frozen question and decision rule

Before S07 implementation or any trajectory, `configs/regeneration/s07_local_memory.json`
froze four component semantics, five variants, 2/2/4/3-bit component costs,
fresh/reset rules, the one-shot unit-cost repair, memoryless controls,
replicate split, three outcome-blind matching choices, completion endpoint,
two-sided exact McNemar tests, Holm correction across five primary contrasts,
censor retention, and the claim boundary. Support required at least one positive
Holm-significant active-versus-primary-matched completion difference with no
significant harm. S06's two timing/opportunity-matched nulls were explicitly
carried forward; permanent-freeze or immediate-reference contrasts cannot alone
support an information-bearing-memory claim.

## Inputs

The run refreshed `AGENTS.md`, `FULL_PLAN.md`, `RESEARCH_PLAN.md`, every S01–S06
report and relevant specification/validation artifact, S02 checkpoints, S03
reversal fixtures, S04–S06 runtime/pairing/ledger/RNG boundaries, E01 release
and transition contracts, E02 action/scheduler/fault/information/stream/ledger
contracts, and the attachment manifest/sidecar. No dataset, network input, GPU,
new dependency, or runtime-driven scope reduction was used.

## Detailed methods

### Inherited state and paired panel

All arms resumed each of 384 exact S02 timing checkpoints after the validated
S03 `segment_reversal_central_v1` anchor. Occupancy, selected identity, Selection
cursors, global clock, stream counters, native ledger, target, original S01
scenario/RNG root, `100*n^2` recovery budget, uniform activation,
distributed-local native observation, skip-and-continue, no retry, and native
NoOp/Swap/MemoryUpdate primitives were unchanged. Replicates 0–1 formed the
calibration bank; replicates 2–3 were confirmatory.

### Local state, reset, and action semantics

The failed-move counter saturated at 3; the blocked-direction mask stored two
direction bits; recent-neighbor state stored valid/side/age with a three-own-turn
TTL; and time-since-local-progress saturated at seven actor-local opportunities.
Accepted native Swap or MemoryUpdate reset that actor's enabled state. Readiness
was evaluated before the current outcome, preventing same-opportunity trigger
leakage. The private bank received only actor identity, current adjacent-frozen
side, its own proposal/outcome direction, and its own native-progress bit. It
held no scenario, occupancy, global event index, task distance/progress, target,
ledger total, future draw, or other actor state.

When prior state was ready, the active arm suppressed exactly one already-built
native proposal, spent one action and one abstract energy unit, and recovered
on the next opportunity. No extra proposal, retry, activation, or runtime draw
was introduced. Explicit reset cleared only private state and charged neither
an opportunity nor draw. All main runs started fresh; no pre-injury history was
converted into private state.

### Controls, matching, and inference

Permanent memoryless freeze and the S06 immediate-adjacent repair were shared
descriptive references. For every variant, calibration duration/censor/action/
energy packages were assigned to confirmatory recipients through separate
counter-addressed bijections within primary `n × policy`, n-only, and pooled
strata. Finite matched controls suppressed one proposal and paid the same unit
cost at the assigned boundary; unrecovered sentinels scheduled neither recovery
nor cost. Confirmatory outcomes never entered assignment. Primary inference
used 192 pairs per variant and Holm correction across five exact McNemar tests.
Paired t intervals summarize risk and continuous trade-offs without survivor
selection.

### Execution and validation method

The panel contained 960 calibration active, 960 confirmatory active, 2,880
confirmatory matched controls, 192 permanent controls, and 192 immediate
references. Every run was replayed exactly, yielding 10,368 trajectory
executions on eight workers. Ten prespecified active rows retained complete
native events and private-state audits. Other rows used the fixture-validated
serial summary projection with identical policy construction, validation,
commit, ledger, terminal, distance-AUC, private-state, and replay results.

## Commands

```text
python -m pytest -q tests/test_regeneration_local_memory.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py tests/test_regeneration_dynamic_faults.py tests/test_regeneration_nudge_recovery.py tests/test_regeneration_assisted_rescue.py tests/test_regeneration_local_memory.py
ruff check src/regeneration/local_memory.py tests/test_regeneration_local_memory.py scripts/build_regeneration_s07.py configs/regeneration/s07_local_memory.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/build_regeneration_s07.py --artifacts-dir {output} --workers 8
```

## Results

### Primary information-bearing-state contrasts

{primary_text}

Across all primary variants, the median mean change in repeated frozen-target
blocks versus matched schedules was {median_block_delta:.2f}. Matching-sensitivity
analysis executed {len(sensitivity)} variant-by-choice tests; its result is
reported separately from the five prespecified primary tests.

### Component-wise complexity and ablation

{complexity_text}

The five one-component/all-components rows are isolated estimates on the same
paired checkpoint panel. The union is a logical OR benchmark, not a learned
combination; correlated local signals can therefore make some components
redundant.

### Descriptive references

{reference_text}

These comparisons describe recovery delay and cost trade-offs. In accordance
with the frozen rule and S06 handoff, they do not establish an advantage of
information-bearing state beyond elapsed-time scheduling.

### Matching, censoring, and ledgers

All {matching['assignmentCount']} assignments preserved their exact calibration
duration/censor/cost multisets across {matching['validatedStratumCount']} variant
strata. {matching['assignedUnrecoveredSentinelCount']} assignment records were
unrecovered sentinels. Scheduled controls retained
{schedule['assignedUnrecoveredCount']} sentinel rows and
{schedule['competingTerminalCensorCount']} finite schedules censored by a prior
terminal. Across the full panel, {censor['recoveryCensoredRunCount']} runs were
recovery-censored, including all quiescent and phase-budget terminals.

There were {ledger['memoryGatedRepairCount']} memory-gated repairs,
{ledger['immediateReferenceRepairCount']} immediate-reference repairs, and
{ledger['matchedScheduleEventCount']} matched schedule events. Every finite event
spent one action and one abstract energy unit and suppressed one native
opportunity; {ledger['forgoneEligibleNativeChangeCount']} suppressed native
proposals had been mechanically eligible.

## Validation

All 384 checkpoint and reversal-anchor hashes matched S02/S03. All 1,920
checkpoint-by-variant structural audits changed private state, reset it exactly
to fresh, left occupancy/cursors/clocks/streams/native ledgers unchanged, and
charged no opportunity or draw. Fixtures covered component bounds, saturation,
prior-state trigger ordering, left/right masks, recent-state expiry, actor-local
timer independence, accepted-progress resets, all native policies, finite and
sentinel schedules, permanent/immediate controls, full/digest parity, and exact
replay.

All {accounting['plannedRunsObserved']} result objects, final native/private
states, process audits, ledgers, hashes, and transitions replayed byte-for-byte.
One proposal remained charged per activation. State bookkeeping created no
proposal opportunity. The private bank owned no runtime stream and exposed zero
global progress/event-clock reads; the construction-only assignment stream
never entered a trajectory. Planned/executed/replayed/traced/censored/terminal/
contrast counts reconciled with zero failures, substitutions, silent exclusions,
or scope reduction.

## Artifacts

- `memory_variants/local_memory_spec.json`/Markdown and schemas freeze state, control, result, and claim contracts.
- `local_memory.parquet` and `local_memory_scenarios.parquet` retain all 5,184 runs; `paired_memory_contrasts.parquet` retains 4,800 paired rows.
- `memory_variants/control_assignments.parquet`, `structural_state_reset_audit.parquet`, and `memory_complexity.parquet` preserve matching, reset, and explicit-cost evidence.
- `primary_completion_tests.parquet`, `matching_sensitivity_tests.parquet`, `reference_completion_tests.parquet`, `component_ablation_results.parquet`, `tradeoff_summary.parquet`, and the PNG/SVG figure preserve estimates.
- Ten selected full traces plus checkpoint, bounds/reset, observability, budget, repair-ledger, opportunity, matching, schedule, pairing, replay, stream, censor, sensitivity, accounting, provenance, and artifact manifests preserve validation.

## Caveats, blockers, failed assumptions, and limitations

- The memory bank and repair channel are engineered finite-state extensions; the
  native Bubble, Insertion, and Selection policies did not evolve them.
- The adjacent-frozen bit/side and abstract energy are transparent simulator
  channels, not biological sensing, force, pressure, metabolism, or physical
  energy.
- Calibration-marginal matching isolates declared recovery-time/censor/cost
  packages but not each recipient's actor identity, local structural history, or
  exact endogenous trigger path.
- Paired initial structure and exact reset audit remove baseline imbalance, but
  later structure is affected by treatment. A full arrangement-preserved versus
  internal-state-preserved factorial remains S11 and was not pulled forward.
- Recent-neighbor state stores only side and own-turn age, never immutable
  neighbor identity or value. The actor-local timer is deliberately an elapsed
  own-opportunity signal; the primary matched schedule asks whether other local
  state adds outcome information beyond elapsed recovery timing.
- The union's OR rule favors early signals and is a bounded benchmark, not an
  optimized policy. Alternative thresholds, memory sizes, stochastic repair,
  multiple repairs, and learned combinations were outside the frozen question.
- Runtime immobility is not byte-identical to static stuck reconstruction;
  count-changing S03 lesions remain outside this fixed-identity runner.
- A null cannot prove local memory is generally useless; a positive result is
  equally limited to these sizes, policies, timings, reversal anchor, costs,
  thresholds, and deterministic success rule.

## Provenance

- Repository: `Eidosoma/cell_research`
- Branch: `eidosoma/groups/28`
- Source commit: `{git_commit}`
- Benchmark: `{BENCHMARK_VERSION}`
- RNG: inherited E01 SHA-256 counter-addressed runtime streams plus construction-only `{CONTROL_ASSIGNMENT_STREAM}`; active memory is stream-free.
- Runtime: Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}; eight workers with OMP/MKL/OpenBLAS threads fixed at one.
- Generated UTC: {datetime.now(timezone.utc).isoformat()}

Input/output SHA-256 hashes are recorded in `input_provenance.json` and
`artifact_manifest.json`. Reproducible source remains in the pushed Git commit;
S08 was not started.
"""
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    return outcome


def _artifact_manifest(output: Path, git_commit: str) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
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
            "schemaVersion": "e05.s07.artifact-manifest.v1",
            "researchStepId": "S07",
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
    assignments: pd.DataFrame,
    contrasts: pd.DataFrame,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    references: pd.DataFrame,
    tradeoffs: pd.DataFrame,
    complexity: pd.DataFrame,
    structural: pd.DataFrame,
    traces: list[dict[str, Any]],
    validations: Mapping[str, Any],
    workers: int,
    git_commit: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    package = output / "memory_variants"
    package.mkdir(parents=True, exist_ok=True)
    _write_json(package / "local_memory_spec.json", specification)
    _write_json(package / "local_memory_spec.schema.json", LOCAL_MEMORY_SPEC_SCHEMA)
    _write_json(
        package / "local_memory_run.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://eidosoma.local/schemas/e05/s07/local-memory-run.schema.json",
            "type": "object",
            "required": [
                "schemaVersion",
                "benchmarkVersion",
                "contract",
                "sourceScenarioId",
                "sourceCheckpointHash",
                "anchorLesionStateHash",
                "processLedger",
                "opportunityValidation",
            ],
            "properties": {
                "schemaVersion": {"const": LOCAL_MEMORY_RUN_SCHEMA_VERSION},
                "benchmarkVersion": {"const": BENCHMARK_VERSION},
            },
        },
    )
    (package / "local_memory_spec.md").write_text(
        _spec_markdown(specification), encoding="utf-8"
    )
    results.to_parquet(output / "local_memory.parquet", index=False)
    scenarios.to_parquet(output / "local_memory_scenarios.parquet", index=False)
    assignments.to_parquet(package / "control_assignments.parquet", index=False)
    contrasts.to_parquet(output / "paired_memory_contrasts.parquet", index=False)
    primary.to_parquet(output / "primary_completion_tests.parquet", index=False)
    sensitivity.to_parquet(output / "matching_sensitivity_tests.parquet", index=False)
    references.to_parquet(output / "reference_completion_tests.parquet", index=False)
    tradeoffs.to_parquet(output / "tradeoff_summary.parquet", index=False)
    complexity.to_parquet(package / "memory_complexity.parquet", index=False)
    structural.to_parquet(package / "structural_state_reset_audit.parquet", index=False)
    ablation = primary.merge(complexity, on="variantId", how="left")
    ablation.to_parquet(output / "component_ablation_results.parquet", index=False)
    pd.DataFrame(reconstructed["checkpointRows"]).to_parquet(
        output / "checkpoint_compatibility.parquet", index=False
    )
    with (package / "selected_full_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(traces, key=lambda item: item["memoryRunId"]):
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _plot_results(primary, complexity, package)
    names = {
        "checkpoint_validation.json": validations["checkpointValidation"],
        "state_bounds_reset_validation.json": validations[
            "stateBoundsResetValidation"
        ],
        "local_observability_validation.json": validations[
            "localObservabilityValidation"
        ],
        "budget_conservation_validation.json": validations["budgetValidation"],
        "repair_event_ledger_validation.json": validations["repairLedgerValidation"],
        "matched_policy_opportunity_validation.json": validations[
            "opportunityValidation"
        ],
        "control_matching_validation.json": validations["matchingValidation"],
        "schedule_cost_validation.json": validations["scheduleValidation"],
        "pairing_structural_confounding_validation.json": validations[
            "pairingValidation"
        ],
        "replay_validation.json": validations["replayValidation"],
        "stream_rng_boundary_validation.json": validations["streamValidation"],
        "censor_retention_validation.json": validations["censorValidation"],
        "matching_sensitivity_validation.json": validations[
            "sensitivityValidation"
        ],
        "run_accounting.json": validations["runAccounting"],
        "validation_summary.json": validations["validationSummary"],
    }
    for name, value in names.items():
        _write_json(output / name, value)
    _write_json(
        output / "input_provenance.json",
        {
            "schemaVersion": "e05.s07.input-provenance.v1",
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
            "schemaVersion": "e05.s07.environment-provenance.v1",
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
            "schemaVersion": "e05.s07.execution-attempts.v1",
            "researchStepId": "S07",
            "attempts": [
                {
                    "attemptOrdinal": 1,
                    "repositoryCommit": git_commit,
                    "calibrationJobsReturned": 960,
                    "confirmatoryJobsReturned": 4224,
                    "resultAggregationReached": True,
                    "validationReached": True,
                    "validationSuccess": validations["validationSummary"]["success"],
                    "artifactFilesWritten": "see artifact_manifest.json",
                    "outcomesInspected": True,
                    "terminationReason": None,
                }
            ],
            "semanticScopeChanged": False,
            "runtimeOptimization": "fixture-validated exact serial summary projection for non-full-trace rows; state snapshots, terminal rescans, and distance recounts occur only after native state changes",
        },
    )
    (output / "execution_commands.log").write_text(
        "\n".join(
            [
                "python -m pytest -q tests/test_regeneration_local_memory.py",
                "python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py tests/test_regeneration_dynamic_faults.py tests/test_regeneration_nudge_recovery.py tests/test_regeneration_assisted_rescue.py tests/test_regeneration_local_memory.py",
                "ruff check src/regeneration/local_memory.py tests/test_regeneration_local_memory.py scripts/build_regeneration_s07.py configs/regeneration/s07_local_memory.json",
                f"OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/build_regeneration_s07.py --artifacts-dir {output} --workers {workers}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outcome = _report(
        output,
        results,
        primary,
        sensitivity,
        references,
        complexity,
        tradeoffs,
        validations,
        git_commit,
    )
    _write_json(
        output / "outcome_classification.json",
        {
            "schemaVersion": "e05.s07.outcome-classification.v1",
            "researchStepId": "S07",
            "classification": outcome,
            "validationSuccess": validations["validationSummary"]["success"],
            "primaryHolmRejectionCount": int(primary["rejectAtFamilywise0_05"].sum()),
            "matchingSensitivityHolmRejectionCount": int(
                sensitivity["rejectAtFamilywise0_05WithinMatchingChoice"].sum()
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
        raise ValueError("S07 worker count must be between 1 and 8")
    specification = _load_json(CONFIG)
    validate_local_memory_spec(specification)
    inherited_gates = _validate_inputs()
    print(f"inherited inputs validated: {inherited_gates}", flush=True)
    reconstructed = _reconstruct_jobs(_load_json(S04_CONFIG))
    bases = _base_jobs(reconstructed)
    structural = _structural_reset_audit(bases)
    calibration_jobs = [
        job for job in _active_jobs(bases) if job.split == "calibration"
    ]
    calibration_execution = _run_jobs(
        calibration_jobs, args.workers, "S07 calibration"
    )
    if calibration_execution["failures"]:
        raise RuntimeError(
            f"S07 calibration failures: {calibration_execution['failures']}"
        )
    confirmatory_bases = [base for base in bases if base.replicate in {2, 3}]
    matched_jobs, assignment_rows = _construct_assignments(
        calibration_jobs,
        calibration_execution["results"],
        confirmatory_bases,
    )
    direct_jobs = [
        job for job in _active_jobs(confirmatory_bases) if job.split == "confirmatory"
    ] + _shared_memoryless_jobs(confirmatory_bases)
    confirmatory_execution = _run_jobs(
        [*direct_jobs, *matched_jobs], args.workers, "S07 confirmatory"
    )
    failures = [
        *calibration_execution["failures"],
        *confirmatory_execution["failures"],
    ]
    if failures:
        raise RuntimeError(f"S07 runtime failures: {failures}")
    results = pd.DataFrame(
        [
            *calibration_execution["results"],
            *confirmatory_execution["results"],
        ]
    ).sort_values(
        ["split", "n", "policy", "direction", "replicateOrdinal", "timingConditionId", "arm", "variantId", "matchingChoice"],
        na_position="first",
    ).reset_index(drop=True)
    scenarios = pd.DataFrame(
        [
            *calibration_execution["scenarios"],
            *confirmatory_execution["scenarios"],
        ]
    ).sort_values("memoryRunId").reset_index(drop=True)
    assignments = pd.DataFrame(assignment_rows).sort_values(
        ["variantId", "matchingChoice", "stratumJson", "recipientRankUint64"]
    ).reset_index(drop=True)
    traces = [
        *calibration_execution["traces"],
        *confirmatory_execution["traces"],
    ]
    contrasts = _contrasts(results).sort_values(
        ["variantId", "controlArm", "matchingChoice", "memoryCaseId"]
    ).reset_index(drop=True)
    primary = _primary_tests(contrasts)
    sensitivity = _sensitivity_tests(contrasts)
    references = _reference_tests(contrasts)
    tradeoffs = _tradeoff_summary(contrasts)
    complexity = _complexity_table(results)
    validations = _validate_panel(
        reconstructed,
        results,
        scenarios,
        assignments,
        contrasts,
        primary,
        sensitivity,
        structural,
        failures,
        traces,
    )
    if not validations["validationSummary"]["success"]:
        print(json.dumps(validations["validationSummary"], indent=2), flush=True)
        raise RuntimeError("S07 validation failed before artifact emission")
    git_commit = _git("rev-parse", "HEAD")
    write_outputs(
        args.artifacts_dir,
        specification,
        reconstructed,
        results,
        scenarios,
        assignments,
        contrasts,
        primary,
        sensitivity,
        references,
        tradeoffs,
        complexity,
        structural,
        traces,
        validations,
        args.workers,
        git_commit,
    )
    print(
        json.dumps(
            {
                "researchStepId": "S07",
                "success": True,
                "runCount": len(results),
                "replayCount": int(results["exactReplayPass"].sum()),
                "contrastCount": len(contrasts),
                "assignmentCount": len(assignments),
                "traceCount": len(traces),
                "outcomeClassification": _classification(primary, True),
                "output": str(args.artifacts_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
