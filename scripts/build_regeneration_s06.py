#!/usr/bin/env python3
"""Build and validate E05 S06 neighbor-assisted rescue evidence."""

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
from src.regeneration.assisted_rescue import (  # noqa: E402
    ASSISTED_RUN_SCHEMA_VERSION,
    ASSISTED_SPEC_SCHEMA,
    BENCHMARK_VERSION,
    CONTROL_ASSIGNMENT_STREAM,
    AssistedRescueContract,
    MatchingChoice,
    RescueArm,
    assisted_case_id,
    exact_replay_assisted_rescue,
    run_assisted_rescue_phase,
    validate_assisted_spec,
)


CONFIG = REPOSITORY / "configs/regeneration/s06_assisted_rescue.json"
S04_CONFIG = REPOSITORY / "configs/regeneration/s04_dynamic_faults.json"
S01_DIR = Path("/artifacts/research_steps/S01")
S02_DIR = Path("/artifacts/research_steps/S02")
S03_DIR = Path("/artifacts/research_steps/S03")
S04_DIR = Path("/artifacts/research_steps/S04")
S05_DIR = Path("/artifacts/research_steps/S05")
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
    S01_DIR / "research_step_full_results.md",
    S01_DIR / "task_spec.md",
    S01_DIR / "task_spec.json",
    S01_DIR / "validation_summary.json",
    S02_DIR / "research_step_full_results.md",
    S02_DIR / "timing_spec.md",
    S02_DIR / "timing_spec.json",
    S02_DIR / "preinjury_states.parquet",
    S02_DIR / "validation_summary.json",
    S03_DIR / "research_step_full_results.md",
    S03_DIR / "lesion_library/lesion_spec.json",
    S03_DIR / "lesion_library/operator_fixtures.parquet",
    S03_DIR / "validation_summary.json",
    S04_DIR / "research_step_full_results.md",
    S04_DIR / "dynamic_fault_package/dynamic_fault_spec.md",
    S04_DIR / "dynamic_fault_package/dynamic_fault_spec.json",
    S04_DIR / "validation_summary.json",
    S05_DIR / "research_step_full_results.md",
    S05_DIR / "nudge_recovery_package/nudge_recovery_spec.md",
    S05_DIR / "nudge_recovery_package/nudge_recovery_spec.json",
    S05_DIR / "validation_summary.json",
    S05_DIR / "outcome_classification.json",
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
        raise FileNotFoundError(f"missing required S01-S05/E01/E02 inputs: {missing}")
    gates = {
        "s01": bool(_load_json(S01_DIR / "validation_summary.json")["success"]),
        "s02": bool(_load_json(S02_DIR / "validation_summary.json")["success"]),
        "s03": bool(_load_json(S03_DIR / "validation_summary.json")["success"]),
        "s04": bool(_load_json(S04_DIR / "validation_summary.json")["success"]),
        "s05": bool(_load_json(S05_DIR / "validation_summary.json")["success"]),
        "e01": bool(
            _load_json(
                Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json")
            )["validationSuccess"]
        ),
        "e02": bool(
            _load_json(
                Path("/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json")
            )["smokeValidation"]["success"]
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"an inherited validation gate failed: {gates}")
    if _load_json(S05_DIR / "outcome_classification.json")["classification"] != "null":
        raise RuntimeError("S06 expected the validated S05 null handoff")
    return gates


@dataclass(frozen=True, slots=True)
class RescueJob:
    base: DynamicJob
    arm: RescueArm
    split: str
    matching_choice: str
    assigned_duration: int | None
    assignment_id: str | None
    donor_case_id: str | None
    retain_full_trace: bool = False


def _case_id(job: RescueJob) -> str:
    return assisted_case_id(
        job.base.s01_pairing_block_id,
        job.base.timing_condition_id,
        job.base.anchor_lesion_state_hash,
    )


def _execute_job(job: RescueJob) -> dict[str, Any]:
    contract = AssistedRescueContract(
        job.arm,
        assigned_duration=job.assigned_duration,
    )
    run = run_assisted_rescue_phase(
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
    exact_replay_assisted_rescue(
        run,
        job.base.scenario,
        job.base.checkpoint,
        job.base.post_anchor_occupancy,
        job.base.recovery_budget,
    )
    case_id = _case_id(job)
    run_id = "e05ar6:" + sha256_json(
        {
            "assistedCaseId": case_id,
            "arm": job.arm.value,
            "split": job.split,
            "matchingChoice": job.matching_choice,
        }
    )
    summary = dict(run.summary)
    process = dict(run.process_final_state)
    ledger = dict(run.process_ledger)
    row = {
        "schemaVersion": "e05.s06.assisted-rescue-result.v1",
        "benchmarkVersion": BENCHMARK_VERSION,
        "assistedRunId": run_id,
        "assistedCaseId": case_id,
        "arm": job.arm.value,
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
        "assignedUnrecoveredSentinel": job.arm
        in {RescueArm.SPONTANEOUS, RescueArm.MATCHED_COST}
        and job.assigned_duration is None,
        "competingTerminalBeforeAssignedEvent": job.arm
        in {RescueArm.SPONTANEOUS, RescueArm.MATCHED_COST}
        and job.assigned_duration is not None
        and not summary["recoveryObserved"],
        "recoveryReason": summary["recoveryReason"],
        "interventionEventIndex": process["interventionEventIndex"],
        "interventionActorId": process["interventionActorId"],
        "interventionNativeProposalKind": process[
            "interventionNativeProposalKind"
        ],
        "interventionNativeProposalEligible": process[
            "interventionNativeProposalEligible"
        ],
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
        "schemaVersion": "e05.s06.assisted-rescue-scenario.v1",
        "assistedRunId": run_id,
        "assistedCaseId": case_id,
        "arm": job.arm.value,
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
            "assistedRunId": run_id,
            "assistedCaseId": case_id,
            "arm": job.arm.value,
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


def _run_jobs(jobs: list[RescueJob], workers: int, label: str) -> dict[str, Any]:
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


def _active_job(base: DynamicJob) -> RescueJob:
    split = "calibration" if base.replicate in {0, 1} else "confirmatory"
    return RescueJob(
        base,
        RescueArm.ACTIVE,
        split,
        "active_reference",
        None,
        None,
        None,
        split == "confirmatory" and _trace_case(base),
    )


def _stratum(choice: MatchingChoice, base: DynamicJob) -> tuple[Any, ...]:
    if choice == MatchingChoice.PRIMARY_N_POLICY:
        return (base.n, base.policy)
    if choice == MatchingChoice.SENSITIVITY_N:
        return (base.n,)
    return ("pooled",)


def _rank(base: DynamicJob, choice: MatchingChoice, role: str) -> int:
    choice_index = list(MatchingChoice).index(choice)
    role_index = 0 if role == "donor" else 1
    return u64(
        base.scenario.seed,
        base.scenario.scenario_id,
        CONTROL_ASSIGNMENT_STREAM,
        base.checkpoint.activation_count,
        2 * choice_index + role_index,
    )


def _construct_assignments(
    calibration_jobs: list[RescueJob],
    calibration_rows: list[dict[str, Any]],
    confirmatory_bases: list[DynamicJob],
) -> tuple[list[RescueJob], list[dict[str, Any]]]:
    row_by_case = {row["assistedCaseId"]: row for row in calibration_rows}
    controls: list[RescueJob] = []
    assignments: list[dict[str, Any]] = []
    for choice in MatchingChoice:
        strata = sorted({_stratum(choice, job.base) for job in calibration_jobs})
        for stratum in strata:
            donors = [
                job for job in calibration_jobs if _stratum(choice, job.base) == stratum
            ]
            recipients = [
                base for base in confirmatory_bases if _stratum(choice, base) == stratum
            ]
            if len(donors) != len(recipients) or not donors:
                raise RuntimeError(f"non-bijective S06 matching pool {choice.value}/{stratum}")
            donors.sort(
                key=lambda job: (_rank(job.base, choice, "donor"), _case_id(job))
            )
            recipients.sort(
                key=lambda base: (
                    _rank(base, choice, "recipient"),
                    assisted_case_id(
                        base.s01_pairing_block_id,
                        base.timing_condition_id,
                        base.anchor_lesion_state_hash,
                    ),
                )
            )
            for donor, recipient in zip(donors, recipients, strict=True):
                donor_case = _case_id(donor)
                recipient_case = assisted_case_id(
                    recipient.s01_pairing_block_id,
                    recipient.timing_condition_id,
                    recipient.anchor_lesion_state_hash,
                )
                donor_row = row_by_case[donor_case]
                recovered = bool(donor_row["recoveryObserved"])
                duration = int(donor_row["recoveryDuration"]) if recovered else None
                action_cost = int(donor_row["actionUnitsSpent"])
                energy_cost = int(donor_row["energyUnitsSpent"])
                if recovered and (action_cost, energy_cost) != (1, 1):
                    raise RuntimeError("a recovered calibration active run lacked unit costs")
                if not recovered and (action_cost, energy_cost) != (0, 0):
                    raise RuntimeError("an unrecovered calibration active run spent a cost")
                assignment_id = "e05ma6:" + sha256_json(
                    {
                        "choice": choice.value,
                        "stratum": list(stratum),
                        "donorCaseId": donor_case,
                        "recipientCaseId": recipient_case,
                    }
                )
                retain = choice == MatchingChoice.PRIMARY_N_POLICY and _trace_case(
                    recipient
                )
                for arm in (RescueArm.SPONTANEOUS, RescueArm.MATCHED_COST):
                    controls.append(
                        RescueJob(
                            recipient,
                            arm,
                            "confirmatory",
                            choice.value,
                            duration,
                            assignment_id,
                            donor_case,
                            retain,
                        )
                    )
                assignments.append(
                    {
                        "schemaVersion": "e05.s06.control-assignment.v1",
                        "assignmentId": assignment_id,
                        "matchingChoice": choice.value,
                        "stratumJson": json.dumps(list(stratum), separators=(",", ":")),
                        "donorCaseId": donor_case,
                        "recipientCaseId": recipient_case,
                        "donorReplicateOrdinal": donor.base.replicate,
                        "recipientReplicateOrdinal": recipient.replicate,
                        "donorRankUint64": str(_rank(donor.base, choice, "donor")),
                        "recipientRankUint64": str(
                            _rank(recipient, choice, "recipient")
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
    if len(assignments) != 576 or len(controls) != 1152:
        raise RuntimeError("S06 assignment/control count changed")
    return controls, assignments


def _direct_confirmatory_jobs(bases: list[DynamicJob]) -> list[RescueJob]:
    jobs: list[RescueJob] = []
    for base in bases:
        for arm in (RescueArm.ACTIVE, RescueArm.PASSIVE, RescueArm.SHAM):
            jobs.append(
                RescueJob(
                    base,
                    arm,
                    "confirmatory",
                    "active_reference" if arm == RescueArm.ACTIVE else "not_applicable",
                    None,
                    None,
                    None,
                    _trace_case(base),
                )
            )
    return jobs


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
    indexed = calibration.set_index("assistedCaseId")
    for (choice, stratum), assigned in assignments.groupby(
        ["matchingChoice", "stratumJson"], sort=True
    ):
        donors = indexed.loc[assigned["donorCaseId"].tolist()]
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
            failures.append(f"{choice}/{stratum}: duration/censor/cost mismatch")
        stratum_count += 1
    outcome_blind = bool(
        (~assignments["confirmatoryOutcomesAvailableAtAssignment"]).all()
        and assignments["donorReplicateOrdinal"].isin([0, 1]).all()
        and assignments["recipientReplicateOrdinal"].isin([2, 3]).all()
    )
    return {
        "schemaVersion": "e05.s06.control-matching-validation.v1",
        "assignmentCount": len(assignments),
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
    for case_id, group in confirm.groupby("assistedCaseId", sort=False):
        counts = group["arm"].value_counts().to_dict()
        expected = {
            RescueArm.ACTIVE.value: 1,
            RescueArm.PASSIVE.value: 1,
            RescueArm.SHAM.value: 1,
            RescueArm.SPONTANEOUS.value: 3,
            RescueArm.MATCHED_COST.value: 3,
        }
        if counts != expected:
            failures.append(f"{case_id}: arm counts {counts}")
            continue
        for arm in (RescueArm.SPONTANEOUS, RescueArm.MATCHED_COST):
            choices = set(group.loc[group["arm"] == arm.value, "matchingChoice"])
            if choices != {item.value for item in MatchingChoice}:
                failures.append(f"{case_id}/{arm.value}: matching choices changed")
        for field in fields:
            if group[field].nunique(dropna=False) != 1:
                failures.append(f"{case_id}: mismatch in {field}")
    return {
        "schemaVersion": "e05.s06.pairing-validation.v1",
        "confirmatoryCaseCount": confirm["assistedCaseId"].nunique(),
        "confirmatoryRowCount": len(confirm),
        "failureCount": len(failures),
        "failures": failures,
        "baseStreamStatus": "shared_prefix_until_intervention_path_or_terminal_divergence",
        "matchingStreamStatus": "construction_only_isolated_no_dummy_runtime_draws",
        "success": not failures,
    }


def _contrasts(results: pd.DataFrame) -> pd.DataFrame:
    active = results.query(
        "split == 'confirmatory' and arm == 'active_assisted_rescue'"
    ).set_index("assistedCaseId")
    controls = results.query(
        "split == 'confirmatory' and arm != 'active_assisted_rescue'"
    )
    rows: list[dict[str, Any]] = []
    for control in controls.itertuples():
        reference = active.loc[control.assistedCaseId]
        rows.append(
            {
                "schemaVersion": "e05.s06.assisted-rescue-contrast.v1",
                "assistedCaseId": control.assistedCaseId,
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
                "activeEnergySpent": int(reference["energyUnitsSpent"]),
                "controlEnergySpent": int(control.energyUnitsSpent),
                "energySpentDelta": int(reference["energyUnitsSpent"])
                - int(control.energyUnitsSpent),
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


def _test_row(group: pd.DataFrame, control_arm: str, choice: str) -> dict[str, Any]:
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
        "schemaVersion": "e05.s06.completion-test.v1",
        "controlArm": control_arm,
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
        "meanEnergySpentDelta": float(group["energySpentDelta"].mean()),
    }


def _primary_tests(contrasts: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        (RescueArm.PASSIVE.value, "not_applicable"),
        (RescueArm.SHAM.value, "not_applicable"),
        (RescueArm.SPONTANEOUS.value, MatchingChoice.PRIMARY_N_POLICY.value),
        (RescueArm.MATCHED_COST.value, MatchingChoice.PRIMARY_N_POLICY.value),
    ]
    rows = []
    for arm, choice in definitions:
        group = contrasts[
            (contrasts["controlArm"] == arm)
            & (contrasts["matchingChoice"] == choice)
        ]
        rows.append(_test_row(group, arm, choice))
    adjusted = _holm([row["exactMcNemarPValue"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holmAdjustedPValue"] = value
        row["rejectAtFamilywise0_05"] = value <= 0.05
    return pd.DataFrame(rows)


def _sensitivity_tests(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in (RescueArm.SPONTANEOUS, RescueArm.MATCHED_COST):
        arm_rows = []
        for choice in MatchingChoice:
            group = contrasts[
                (contrasts["controlArm"] == arm.value)
                & (contrasts["matchingChoice"] == choice.value)
            ]
            arm_rows.append(_test_row(group, arm.value, choice.value))
        adjusted = _holm([row["exactMcNemarPValue"] for row in arm_rows])
        for row, value in zip(arm_rows, adjusted, strict=True):
            row["holmAdjustedPValueWithinControl"] = value
            row["rejectAtFamilywise0_05WithinControl"] = value <= 0.05
        rows.extend(arm_rows)
    return pd.DataFrame(rows)


def _tradeoff_summary(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, choice), group in contrasts.groupby(
        ["controlArm", "matchingChoice"], sort=True
    ):
        completion, completion_low, completion_high = _mean_ci(
            group["completionDifference"]
        )
        phase, phase_low, phase_high = _mean_ci(group["phaseActivationDelta"])
        final, final_low, final_high = _mean_ci(group["finalDistanceDelta"])
        auc, auc_low, auc_high = _mean_ci(group["distanceAucDelta"])
        mean_active_energy = float(group["activeEnergySpent"].mean())
        rows.append(
            {
                "schemaVersion": "e05.s06.tradeoff-summary.v1",
                "controlArm": arm,
                "matchingChoice": choice,
                "pairCount": len(group),
                "completionRiskDifference": completion,
                "completionRiskDifferenceCi95Low": completion_low,
                "completionRiskDifferenceCi95High": completion_high,
                "meanPhaseActivationDelta": phase,
                "meanPhaseActivationDeltaCi95Low": phase_low,
                "meanPhaseActivationDeltaCi95High": phase_high,
                "meanFinalDistanceDelta": final,
                "meanFinalDistanceDeltaCi95Low": final_low,
                "meanFinalDistanceDeltaCi95High": final_high,
                "meanDistanceAucDelta": auc,
                "meanDistanceAucDeltaCi95Low": auc_low,
                "meanDistanceAucDeltaCi95High": auc_high,
                "meanActiveEnergySpent": mean_active_energy,
                "meanControlEnergySpent": float(group["controlEnergySpent"].mean()),
                "completionBenefitPerMeanActiveEnergy": completion
                / mean_active_energy
                if mean_active_energy > 0
                else None,
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
    failures: list[dict[str, str]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    calibration = results.query(
        "split == 'calibration' and arm == 'active_assisted_rescue'"
    )
    confirm_active = results.query(
        "split == 'confirmatory' and arm == 'active_assisted_rescue'"
    )
    passive = results[results["arm"] == RescueArm.PASSIVE.value]
    sham = results[results["arm"] == RescueArm.SHAM.value]
    spontaneous = results[results["arm"] == RescueArm.SPONTANEOUS.value]
    matched = results[results["arm"] == RescueArm.MATCHED_COST.value]
    matching = _matching_validation(calibration, assignments)
    pairing = _pairing_validation(scenarios)
    active_sham = confirm_active.set_index("assistedCaseId").join(
        sham.set_index("assistedCaseId"), lsuffix="Active", rsuffix="Sham"
    )
    repair_ledger_pass = bool(
        (results["repairProposals"]
         == results["repairSuccesses"] + results["shamRepairFailures"]).all()
        and (
            results["actionUnitsSpent"]
            == results["energyUnitsSpent"]
        ).all()
        and (
            results["energyUnitsSpent"]
            == results["nativeOpportunitiesSuppressed"]
        ).all()
        and (
            results["nativeOpportunitiesSuppressed"]
            == results["repairProposals"] + results["matchedCostEvents"]
        ).all()
        and (active_sham["interventionEventIndexActive"].fillna(-1)
             == active_sham["interventionEventIndexSham"].fillna(-1)).all()
        and (active_sham["repairProposalsActive"]
             == active_sham["repairProposalsSham"]).all()
        and (active_sham["energyUnitsSpentActive"]
             == active_sham["energyUnitsSpentSham"]).all()
        and (confirm_active["repairSuccesses"] == confirm_active["recoveries"]).all()
        and (sham["recoveries"] == 0).all()
        and (passive["recoveries"] == 0).all()
    )
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
    schedule_pass = bool(
        (
            spontaneous.loc[spontaneous["recoveryObserved"], "recoveryDuration"].astype(int)
            == spontaneous.loc[spontaneous["recoveryObserved"], "assignedDuration"].astype(int)
        ).all()
        and (
            matched.loc[matched["recoveryObserved"], "recoveryDuration"].astype(int)
            == matched.loc[matched["recoveryObserved"], "assignedDuration"].astype(int)
        ).all()
        and (spontaneous["energyUnitsSpent"] == 0).all()
        and (
            matched.loc[matched["recoveryObserved"], "energyUnitsSpent"] == 1
        ).all()
        and (
            ~results.loc[
                results["assignedUnrecoveredSentinel"], "recoveryObserved"
            ].astype(bool)
        ).all()
    )
    stream_pass = bool(
        all(
            CONTROL_ASSIGNMENT_STREAM not in json.loads(value)
            for value in results["streamCounterDeltaJson"]
        )
    )
    censor_pass = bool(
        len(results) == 1920
        and len(contrasts) == 1536
        and results["stopReason"].notna().all()
        and results["recoveryCensored"].notna().all()
        and assignments["assignedCensored"].notna().all()
    )
    run_accounting = {
        "schemaVersion": "e05.s06.run-accounting.v1",
        "researchStepId": "S06",
        "sourceBlocksExpected": 48,
        "sourceBlocksObserved": len(
            {row["s01PairingBlockId"] for row in reconstructed["checkpointRows"]}
        ),
        "checkpointsExpected": 384,
        "checkpointsObserved": len(reconstructed["checkpointRows"]),
        "calibrationActiveRunsExpected": 192,
        "calibrationActiveRunsObserved": len(calibration),
        "confirmatoryActiveRunsExpected": 192,
        "confirmatoryActiveRunsObserved": len(confirm_active),
        "confirmatoryPassiveRunsExpected": 192,
        "confirmatoryPassiveRunsObserved": len(passive),
        "confirmatoryShamRunsExpected": 192,
        "confirmatoryShamRunsObserved": len(sham),
        "confirmatorySpontaneousRunsExpected": 576,
        "confirmatorySpontaneousRunsObserved": len(spontaneous),
        "confirmatoryMatchedCostRunsExpected": 576,
        "confirmatoryMatchedCostRunsObserved": len(matched),
        "plannedRunsExpected": 1920,
        "plannedRunsObserved": len(results),
        "exactReplayExecutionsExpected": 1920,
        "exactReplayExecutionsObserved": int(results["exactReplayPass"].sum()),
        "totalTrajectoryExecutionsExpected": 3840,
        "totalTrajectoryExecutionsObserved": 2 * len(results),
        "pairwiseContrastsExpected": 1536,
        "pairwiseContrastsObserved": len(contrasts),
        "fullTraceRunsExpected": 10,
        "fullTraceRunsObserved": len(traces),
        "runtimeFailureCount": len(failures),
        "runtimeFailures": failures,
        "unreachableCheckpointCount": 384 - len(reconstructed["checkpointRows"]),
        "substitutionCount": 0,
        "silentExclusionCount": 1920 - len(results),
        "scopeReduction": False,
    }
    run_accounting["success"] = all(
        (
            run_accounting["sourceBlocksObserved"] == 48,
            run_accounting["checkpointsObserved"] == 384,
            run_accounting["calibrationActiveRunsObserved"] == 192,
            run_accounting["confirmatoryActiveRunsObserved"] == 192,
            run_accounting["confirmatoryPassiveRunsObserved"] == 192,
            run_accounting["confirmatoryShamRunsObserved"] == 192,
            run_accounting["confirmatorySpontaneousRunsObserved"] == 576,
            run_accounting["confirmatoryMatchedCostRunsObserved"] == 576,
            run_accounting["plannedRunsObserved"] == 1920,
            run_accounting["exactReplayExecutionsObserved"] == 1920,
            run_accounting["pairwiseContrastsObserved"] == 1536,
            run_accounting["fullTraceRunsObserved"] == 10,
            not failures,
            run_accounting["silentExclusionCount"] == 0,
        )
    )
    checkpoint = {
        "schemaVersion": "e05.s06.checkpoint-validation.v1",
        "checkpointCount": len(reconstructed["checkpointRows"]),
        "checkpointFailures": reconstructed["checkpointFailures"],
        "anchorFailures": reconstructed["anchorFailures"],
        "success": not reconstructed["checkpointFailures"]
        and not reconstructed["anchorFailures"]
        and len(reconstructed["checkpointRows"]) == 384,
    }
    validations = {
        "checkpointValidation": checkpoint,
        "budgetValidation": {
            "schemaVersion": "e05.s06.budget-conservation-validation.v1",
            "runCount": len(results),
            "actionUnitsSpent": int(results["actionUnitsSpent"].sum()),
            "energyUnitsSpent": int(results["energyUnitsSpent"].sum()),
            "negativeBalanceCount": int(
                ((results["actionBudgetRemaining"] < 0)
                 | (results["energyBudgetRemaining"] < 0)).sum()
            ),
            "budgetConservationPass": budget_pass,
            "energyInterpretation": "abstract_intervention_accounting_unit",
            "success": budget_pass,
        },
        "repairLedgerValidation": {
            "schemaVersion": "e05.s06.repair-action-ledger-validation.v1",
            "runCount": len(results),
            "repairProposalCount": int(results["repairProposals"].sum()),
            "repairSuccessCount": int(results["repairSuccesses"].sum()),
            "shamFailureCount": int(results["shamRepairFailures"].sum()),
            "matchedCostEventCount": int(results["matchedCostEvents"].sum()),
            "nativeOpportunitySuppressionCount": int(
                results["nativeOpportunitiesSuppressed"].sum()
            ),
            "forgoneEligibleNativeChangeCount": int(
                results["forgoneEligibleNativeChanges"].sum()
            ),
            "activeShamEventPairingPass": repair_ledger_pass,
            "success": repair_ledger_pass,
        },
        "matchingValidation": matching,
        "scheduleValidation": {
            "schemaVersion": "e05.s06.schedule-cost-validation.v1",
            "spontaneousRunCount": len(spontaneous),
            "matchedCostRunCount": len(matched),
            "finiteAssignedCount": int(
                results["assignedDuration"].notna().sum()
            ),
            "assignedUnrecoveredCount": int(
                results["assignedUnrecoveredSentinel"].sum()
            ),
            "competingTerminalCensorCount": int(
                results["competingTerminalBeforeAssignedEvent"].sum()
            ),
            "exactDurationAndCostPass": schedule_pass,
            "success": schedule_pass,
        },
        "pairingValidation": pairing,
        "replayValidation": {
            "schemaVersion": "e05.s06.replay-validation.v1",
            "plannedRunCount": len(results),
            "exactReplayCount": int(results["exactReplayPass"].sum()),
            "repairEventLedgerReplayCount": int(results["exactReplayPass"].sum()),
            "failureCount": int((~results["exactReplayPass"]).sum()),
            "success": bool(results["exactReplayPass"].all()),
        },
        "streamValidation": {
            "schemaVersion": "e05.s06.stream-rng-boundary-validation.v1",
            "assignmentStream": CONTROL_ASSIGNMENT_STREAM,
            "assignmentAddressCount": 2 * len(assignments),
            "runtimeAssignmentStreamConsumptionCount": 0,
            "repairRuntimeStreams": [],
            "runtimeIsolationPass": stream_pass,
            "originalScenarioRootPreserved": bool(
                (results["sourceScenarioId"].str.startswith("r1:")).all()
            ),
            "workerOrderInfluence": "none",
            "staticReconstructionExecuted": False,
            "success": stream_pass,
        },
        "censorValidation": {
            "schemaVersion": "e05.s06.censor-retention-validation.v1",
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
            "censorRetentionPass": censor_pass,
            "success": censor_pass,
        },
        "sensitivityValidation": {
            "schemaVersion": "e05.s06.matching-sensitivity-validation.v1",
            "matchingChoices": [item.value for item in MatchingChoice],
            "formalTestCount": len(sensitivity),
            "holmRejectionCount": int(
                sensitivity["rejectAtFamilywise0_05WithinControl"].sum()
            ),
            "allChoicesExecuted": sensitivity["matchingChoice"].nunique() == 3,
            "success": len(sensitivity) == 6
            and sensitivity["matchingChoice"].nunique() == 3,
        },
        "runAccounting": run_accounting,
    }
    summary = {
        "schemaVersion": "e05.s06.validation-summary.v1",
        "researchStepId": "S06",
        "checkpointIdentityPass": checkpoint["success"],
        "budgetConservationPass": validations["budgetValidation"]["success"],
        "repairActionLedgerPass": validations["repairLedgerValidation"]["success"],
        "controlMatchingPass": matching["success"],
        "scheduleAndCostAccuracyPass": validations["scheduleValidation"]["success"],
        "pairingPass": pairing["success"],
        "deterministicReplayPass": validations["replayValidation"]["success"],
        "streamIsolationRngBoundaryPass": validations["streamValidation"]["success"],
        "censorRetentionPass": validations["censorValidation"]["success"],
        "matchingSensitivityPass": validations["sensitivityValidation"]["success"],
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


def _spec_markdown(specification: Mapping[str, Any]) -> str:
    controls = "\n".join(
        f"- `{item['arm']}` — {item['semantics']}" for item in specification["controls"]
    )
    return f"""# S06 neighbor-assisted rescue specification

Frozen before implementation, fixtures, calibration, and confirmatory runs at
`{specification['frozenAtUtc']}`; schema `{specification['schemaVersion']}`.

## Inherited boundary

Every run retains the exact S02 checkpoint and S03 reversal anchor, S04/S05
runtime overlay and original S01 RNG root, `100*n^2` recovery budget, uniform
activation, native policy-local observations, native NoOp/Swap/MemoryUpdate
primitives, native ledger, target, skip-and-continue, and no-retry contracts.

## Minimal local repair proposal

A separate engineered gateway receives only the scheduled actor identity and
position plus a Boolean adjacent-frozen-neighbor signal. When eligible, it
replaces the one already-constructed and already-charged native proposal. It
does not create a second proposal, retry, replacement activation, or random
draw. Active success is deterministic and becomes effective next opportunity;
the sham pays the same cost without recovery.

## Costs and budgets

Each proposal costs one action opportunity and one abstract energy unit. The
collective action budget, collective energy budget, and per-helper energy
budget are each one. “Energy” is an accounting unit, not physical energy.

## Controls

{controls}

Replicates 0–1 form the calibration bank and replicates 2–3 are confirmatory.
Duration/censor/cost packages are assigned by outcome-blind counter-addressed
bijections within primary `n × policy`, n-only, and pooled choices. Unrecovered
sentinels remain in the bank and all terminal cases remain in analysis.
"""


def _plot_tradeoffs(tradeoffs: pd.DataFrame, package: Path) -> None:
    primary = tradeoffs[
        (tradeoffs["matchingChoice"] == "not_applicable")
        | (tradeoffs["matchingChoice"] == MatchingChoice.PRIMARY_N_POLICY.value)
    ].copy()
    labels = [
        item.replace("_", "\n") for item in primary["controlArm"].tolist()
    ]
    x = np.arange(len(primary))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].errorbar(
        x,
        primary["completionRiskDifference"],
        yerr=np.vstack(
            [
                primary["completionRiskDifference"]
                - primary["completionRiskDifferenceCi95Low"],
                primary["completionRiskDifferenceCi95High"]
                - primary["completionRiskDifference"],
            ]
        ),
        fmt="o",
        capsize=4,
    )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Completion benefit", ylabel="Active − control risk")
    axes[1].scatter(x, primary["meanPhaseActivationDelta"], s=55)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(title="Opportunity trade-off", ylabel="Mean active − control opportunities")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(alpha=0.2)
    fig.suptitle("S06 engineered rescue benefits and task costs")
    fig.savefig(package / "rescue_tradeoffs.png", dpi=180)
    fig.savefig(package / "rescue_tradeoffs.svg")
    plt.close(fig)


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
    sham = primary[primary["controlArm"] == RescueArm.SHAM.value].iloc[0]
    if bool(sham["rejectAtFamilywise0_05"]) and float(
        sham["pairedCompletionRiskDifference"]
    ) > 0:
        return "supportive"
    return "null"


def _report(
    output: Path,
    results: pd.DataFrame,
    contrasts: pd.DataFrame,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    tradeoffs: pd.DataFrame,
    validations: Mapping[str, Any],
    git_commit: str,
) -> str:
    outcome = _classification(primary, validations["validationSummary"]["success"])
    primary_lines = []
    for row in primary.itertuples():
        primary_lines.append(
            f"- `{row.controlArm}` ({row.matchingChoice}): active/control completion "
            f"{row.activeCompletionCount}/{row.controlCompletionCount} of {row.pairCount}; "
            f"paired difference {row.pairedCompletionRiskDifference:.3f} "
            f"(95% CI {row.pairedCompletionRiskDifferenceCi95Low:.3f} to "
            f"{row.pairedCompletionRiskDifferenceCi95High:.3f}); discordant active-only/"
            f"control-only {row.activeOnlyCompletionCount}/{row.controlOnlyCompletionCount}; "
            f"exact p={row.exactMcNemarPValue:.4g}, Holm p={row.holmAdjustedPValue:.4g}."
        )
    primary_text = "\n".join(primary_lines)
    trade_lines = []
    primary_trade = tradeoffs[
        (tradeoffs["matchingChoice"] == "not_applicable")
        | (tradeoffs["matchingChoice"] == MatchingChoice.PRIMARY_N_POLICY.value)
    ]
    for row in primary_trade.itertuples():
        trade_lines.append(
            f"- `{row.controlArm}`: mean opportunity delta {row.meanPhaseActivationDelta:.2f}, "
            f"mean final-distance delta {row.meanFinalDistanceDelta:.2f}, mean distance-AUC "
            f"delta {row.meanDistanceAucDelta:.2f}, active/control mean abstract energy "
            f"{row.meanActiveEnergySpent:.3f}/{row.meanControlEnergySpent:.3f}."
        )
    trade_text = "\n".join(trade_lines)
    matching = validations["matchingValidation"]
    schedule = validations["scheduleValidation"]
    repair = validations["repairLedgerValidation"]
    report = f"""# Research step S06 full results — Test neighbor-assisted rescue

## Top summary

- **Research step ID:** S06
- **Completion status:** Complete; S07 was not started.
- **Artifacts written:** Frozen assisted-rescue specification/schema, 1,920-row result/scenario tables, 576 calibration control assignments, 1,536 paired contrasts, four primary tests, six matching-sensitivity tests, trade-off table/plot, ten full traces, validation/accounting/provenance manifests, and this canonical report under `{output}`.
- **Validation result:** PASS — 384/384 inherited checkpoints and anchors, 1,920/1,920 planned runs, 1,920/1,920 exact replays, and all local eligibility, budget, action/event ledger, matching, schedule/cost, pairing, stream, censor-retention, sensitivity, and accounting gates passed.
- **Outcome classification:** {outcome.capitalize()}. The classification is limited to the prespecified engineered intervention and confirmatory panel.
- **Caveats or blockers:** The repair channel is an engineered one-bit local extension, not an emergent native policy or biological mechanism. Energy is an abstract accounting unit. Calibration-marginal controls do not reproduce each recipient’s endogenous helper history. Runtime freezing remains mobility-equivalent, not byte-equivalent, to static stuck reconstruction. No execution blocker remains within S06.
- **Recommended next action:** Chief Scientist review. If this bounded engineered-intervention result is accepted, separately authorize S07 to test minimal local memory; do not treat S07 as started.

## Lay summary

This step gave a scheduled neighbor one new local option: spend its current
action and one bookkeeping energy unit to unfreeze an adjacent damaged cell.
The proposal used no global progress information and did not create a free extra
turn. It was compared with permanent freezing, a proposal that paid the same
cost but did nothing, spontaneous recovery at calibration-matched times, and
recovery with both matched timing and matched cost. Every failed and censored
case was retained. The result describes this deliberately engineered simulator
intervention only; it is not evidence of biological helping or physical energy.

## Frozen question and decision rule

Before implementation or any S06 trajectory, the question, one-bit local
gateway, deterministic success/sham rules, unit costs, one-action/one-energy
budgets, calibration/confirmatory split, four primary contrasts, completion
endpoint, exact McNemar tests, Holm correction, trade-off metrics, and censor
handling were frozen in `configs/regeneration/s06_assisted_rescue.json` at
`2026-07-18T04:33:31Z`. S05’s null was treated as evidence only against a tested
contact-timing distinction; S06 tested a separate action that suppresses a
native proposal and spends an abstract energy unit.

## Inputs

The run refreshed workspace plans; every S01–S05 report and specification; S02
checkpoint records; S03 reversal fixtures; S04/S05 runtime, ledger, pairing,
and RNG boundaries; E01 release/transition contracts; E02 action, scheduler,
fault, information, semantic-stream, and ledger contracts; and the attachment
manifest/sidecar plus the paper’s nudge-motivation passage. No dataset, network
input, GPU, or new dependency was used.

## Detailed methods

### Inherited state and runtime boundary

Every arm resumed one of all 384 exact S02 timing checkpoints after the
validated S03 `segment_reversal_central_v1` anchor. Occupancy, selected identity,
Selection cursors, global clock, stream counters, native ledger, target,
`100*n^2` recovery budget, uniform activation, distributed-local native policy
observations, skip-and-continue, no retry, and native NoOp/Swap/MemoryUpdate
primitives remained fixed. The original S01 scenario ID was retained.

### Repair proposal, costs, and controls

The separate intervention gateway received only scheduled actor identity and
position plus whether the selected frozen identity was immediately adjacent.
An eligible normal non-selected neighbor replaced exactly one already-built
native proposal; that charged opportunity made no native state change, spent
one action unit and one abstract energy unit, and—only in the active arm—made
recovery effective next opportunity. Collective action, collective energy, and
per-helper energy budgets were each one. The sham used the same eligibility and
cost but never recovered.

Permanent passive controls never recovered or spent intervention cost.
Replicates 0–1 supplied active calibration duration/censor/action/energy
packages; replicates 2–3 supplied confirmatory recipients. Outcome-blind
counter-addressed bijections assigned packages under primary `n × policy`,
n-only, and pooled choices. Spontaneous controls used assigned recovery timing
without cost. Matched-cost controls suppressed one native opportunity, spent
one energy unit, and recovered at the assigned boundary. Unrecovered sentinels
scheduled neither recovery nor cost. A competing terminal before a finite
schedule remained censored.

### Panel, endpoints, and uncertainty

The panel comprised 192 calibration active runs and 1,728 confirmatory runs:
192 each active, passive, and sham; 576 each spontaneous and matched-cost.
Every run was repeated for exact result/state/ledger replay, yielding 3,840
trajectory executions on eight workers. Ten prespecified runs retained full
native events; remaining rows used the fixture-validated transition-equivalent
summary path and retained state, ledger, intervention-audit, and result hashes.
Its opportunity-clock distance AUC cached the current inversion distance and
recomputed it only after accepted swaps; parity fixtures covered the exact AUC.
Primary completion inference used 192 paired cases
per contrast, exact two-sided McNemar tests, and Holm correction across four
contrasts. Paired mean-difference 95% t intervals describe completion risk and
trade-off metrics. Post-opportunity strict-inversion distance AUC, phase time,
final distance, suppressed native proposals, and abstract energy were retained.

## Commands

```text
python -m pytest -q tests/test_regeneration_assisted_rescue.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py tests/test_regeneration_dynamic_faults.py tests/test_regeneration_nudge_recovery.py tests/test_regeneration_assisted_rescue.py
ruff check src/regeneration/assisted_rescue.py tests/test_regeneration_assisted_rescue.py scripts/build_regeneration_s06.py src/regeneration/__init__.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/build_regeneration_s06.py --artifacts-dir {output} --workers 8
```

## Results

### Primary confirmatory completion contrasts

{primary_text}

### Action, energy, and task trade-offs

{trade_text}

Across all runs, {repair['repairProposalCount']} local repair proposals produced
{repair['repairSuccessCount']} active successes and {repair['shamFailureCount']}
sham failures. There were {repair['matchedCostEventCount']} matched-cost events,
{repair['nativeOpportunitySuppressionCount']} suppressed native opportunities,
and {repair['forgoneEligibleNativeChangeCount']} suppressed proposals that the
native validator had judged state-change eligible. Energy is strictly an
abstract intervention ledger unit.

### Recovery matching and censoring

All {matching['assignmentCount']} calibration assignments matched their declared
duration-plus-censor-plus-cost multisets in {matching['validatedStratumCount']}
strata. {matching['assignedUnrecoveredSentinelCount']} assignment records were
unrecovered sentinels. Across scheduled control rows,
{schedule['assignedUnrecoveredCount']} retained sentinel schedules and
{schedule['competingTerminalCensorCount']} finite schedules met a competing
terminal before the assigned event. No row was discarded or substituted.

## Validation

All 384 checkpoint and reversal-anchor hashes matched S02/S03. Native ledger
identities, one proposal per activation, action/energy/suppression identities,
per-helper and collective budget conservation, active/sham event pairing,
next-opportunity recovery, exact assigned schedules, sentinel behavior, and
local adjacency rules passed. Fixtures covered invalid native proposals,
non-neighbors, all three native primitive kinds, sham exhaustion, passive
freezing, finite and unrecovered schedules, matched cost, and exact replay.

All 1,920 result objects, final states, intervention audits, ledgers, hashes,
and transitions replayed byte-for-byte within their declared trace mode. Ten
full-trace rows additionally replayed every native event byte. A fixture proved
that the summary path and full-event authority produce identical state, ledger,
intervention, terminal, distance, and cost results across every native policy
and rescue/control arm. The repair channel owned no
runtime random stream; the construction-only assignment stream never entered a
trajectory; worker order entered no address. Planned, executed, replayed,
traced, censored, terminal, and contrast counts reconciled with zero runtime
failures, substitutions, silent exclusions, or scope reduction.

## Artifacts

- `assisted_rescue_package/assisted_rescue_spec.json`/Markdown and schemas freeze the intervention and result contracts.
- `assisted_rescue_package/control_assignments.parquet` records all donor/recipient ranks, durations, censor sentinels, and costs.
- `assisted_rescue.parquet` and `assisted_rescue_scenarios.parquet` retain all 1,920 runs; `paired_rescue_contrasts.parquet` retains 1,536 comparisons.
- `primary_completion_tests.parquet`, `matching_sensitivity_tests.parquet`, `tradeoff_summary.parquet`, the PNG/SVG trade-off figure, and ten selected full traces preserve direct evidence.
- Checkpoint, budget, repair-ledger, matching, schedule/cost, pairing, replay, stream, censor, sensitivity, accounting, input, environment, and artifact manifests preserve validation and provenance.
- `execution_attempts.json` records every outcome-blind rejected attempt before the accepted run.

## Caveats, blockers, failed assumptions, and limitations

- The native Bubble, Insertion, and Selection policies did not evolve or infer
  a repair rule. A separate engineered gateway supplied the adjacent-frozen bit
  and deterministic action.
- “Energy” has no physical units, force, stress, metabolism, or biological
  interpretation. It is a transparent budget counter.
- A one-action unit-cost intervention is a minimal benchmark, not a calibrated
  universal repair economy. Larger budgets, stochastic success, communication,
  and learned allocation were not tested.
- Marginal schedule/cost matching cannot reproduce a recipient’s endogenous
  helper identity or local history. The matched-cost arm isolates the declared
  opportunity-time package, not every path feature.
- The selected identity is normal in native policy observations while the
  repair gateway receives a separate local fault bit. Runtime immobility is not
  byte-identical to a rebuilt static-stuck scenario.
- Count-changing S03 lesions remain outside the fixed-identity runner. Matching
  sensitivities reuse the same calibration bank and active outcomes and are not
  independent replications.
- Four outcome-blind canonical attempts preceded the accepted run. The first
  three completed calibration and entered the confirmatory panel before
  aggregation. The first was terminated because
  digest-only rows still serialized full JSON events; the second exposed an
  exact but redundant full inversion recount on every unchanged opportunity;
  the third exposed generic state-clone and terminal-rescan overhead on every
  rejected/no-op summary opportunity. A fourth full panel reached aggregate
  validation but was rejected before artifact emission because
  intervention-suppressed Selection memory updates were not classified in the
  native rejection bucket. No outcome table was written or inspected in any
  rejected attempt. The full panel was rerun from the beginning after
  fixture-validating the transition-equivalent serial summary projection across
  every policy and arm and correcting the ledger disposition label; rescue
  semantics, accounting quantities, budgets, and scope did not change.
- Failure to find a contrast in this panel would not prove active rescue is
  generally ineffective; any positive result is equally bounded to the frozen
  sizes, policies, timings, lesion anchor, costs, and success rule.

## Provenance

- Repository: `Eidosoma/cell_research`
- Branch: `eidosoma/groups/28`
- Source commit: `{git_commit}`
- Benchmark: `{BENCHMARK_VERSION}`
- RNG: inherited E01 SHA-256 counter-addressed runtime streams plus construction-only `{CONTROL_ASSIGNMENT_STREAM}`.
- Runtime: Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}; eight workers with numerical-library threads fixed at one.
- Generated UTC: {datetime.now(timezone.utc).isoformat()}

Input and output SHA-256 hashes are recorded in `input_provenance.json` and
`artifact_manifest.json`. Reproducible source remains in the pushed Git commit;
S07 was not started.
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
            "schemaVersion": "e05.s06.artifact-manifest.v1",
            "researchStepId": "S06",
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
    tradeoffs: pd.DataFrame,
    traces: list[dict[str, Any]],
    validations: Mapping[str, Any],
    workers: int,
    git_commit: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    package = output / "assisted_rescue_package"
    package.mkdir(parents=True, exist_ok=True)
    _write_json(package / "assisted_rescue_spec.json", specification)
    _write_json(package / "assisted_rescue_spec.schema.json", ASSISTED_SPEC_SCHEMA)
    _write_json(
        package / "assisted_rescue_run.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://eidosoma.local/schemas/e05/s06/assisted-rescue-run.schema.json",
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
                "schemaVersion": {"const": ASSISTED_RUN_SCHEMA_VERSION},
                "benchmarkVersion": {"const": BENCHMARK_VERSION},
            },
        },
    )
    (package / "assisted_rescue_spec.md").write_text(
        _spec_markdown(specification), encoding="utf-8"
    )
    results.to_parquet(output / "assisted_rescue.parquet", index=False)
    scenarios.to_parquet(output / "assisted_rescue_scenarios.parquet", index=False)
    assignments.to_parquet(package / "control_assignments.parquet", index=False)
    contrasts.to_parquet(output / "paired_rescue_contrasts.parquet", index=False)
    primary.to_parquet(output / "primary_completion_tests.parquet", index=False)
    sensitivity.to_parquet(output / "matching_sensitivity_tests.parquet", index=False)
    tradeoffs.to_parquet(output / "tradeoff_summary.parquet", index=False)
    pd.DataFrame(reconstructed["checkpointRows"]).to_parquet(
        output / "checkpoint_compatibility.parquet", index=False
    )
    with (package / "selected_full_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(traces, key=lambda item: item["assistedRunId"]):
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _plot_tradeoffs(tradeoffs, package)
    names = {
        "checkpoint_validation.json": validations["checkpointValidation"],
        "budget_conservation_validation.json": validations["budgetValidation"],
        "repair_action_ledger_validation.json": validations["repairLedgerValidation"],
        "control_matching_validation.json": validations["matchingValidation"],
        "schedule_cost_validation.json": validations["scheduleValidation"],
        "pairing_validation.json": validations["pairingValidation"],
        "replay_validation.json": validations["replayValidation"],
        "stream_rng_boundary_validation.json": validations["streamValidation"],
        "censor_retention_validation.json": validations["censorValidation"],
        "matching_sensitivity_validation.json": validations["sensitivityValidation"],
        "run_accounting.json": validations["runAccounting"],
        "validation_summary.json": validations["validationSummary"],
    }
    for name, value in names.items():
        _write_json(output / name, value)
    _write_json(
        output / "input_provenance.json",
        {
            "schemaVersion": "e05.s06.input-provenance.v1",
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
            "schemaVersion": "e05.s06.environment-provenance.v1",
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
            "schemaVersion": "e05.s06.execution-attempts.v1",
            "researchStepId": "S06",
            "attempts": [
                {
                    "attemptOrdinal": 1,
                    "repositoryCommit": "13883059c3a1f39a0de2e9aeef50b912f864e746",
                    "calibrationJobsReturned": 192,
                    "confirmatoryJobsReturnedBeforeTermination": 300,
                    "resultAggregationReached": False,
                    "artifactFilesWritten": 0,
                    "outcomesInspected": False,
                    "terminationReason": "digest-only rows were still serializing and hashing full JSON events, making full-budget censoring runs impractically slow",
                },
                {
                    "attemptOrdinal": 2,
                    "repositoryCommit": "e4079b9c2c672c6b2b75544f47a3f026cdd5daff",
                    "calibrationJobsReturned": 192,
                    "confirmatoryJobsReturnedBeforeTermination": 300,
                    "resultAggregationReached": False,
                    "artifactFilesWritten": 0,
                    "outcomesInspected": False,
                    "terminationReason": "summary rows recomputed the full inversion-distance AUC on every unchanged opportunity, making retained full-budget censoring runs impractically slow",
                },
                {
                    "attemptOrdinal": 3,
                    "repositoryCommit": "4d9f0752d69ef7e19426c22814623189b48bdd97",
                    "calibrationJobsReturned": 192,
                    "confirmatoryJobsReturnedBeforeTermination": 200,
                    "resultAggregationReached": False,
                    "artifactFilesWritten": 0,
                    "outcomesInspected": False,
                    "terminationReason": "generic summary execution still cloned the complete state and rescanned terminal conditions on every rejected or no-op opportunity",
                },
                {
                    "attemptOrdinal": 4,
                    "repositoryCommit": "cfbec86de0aa693e7125bfbf629a4e407e55f635",
                    "calibrationJobsReturned": 192,
                    "confirmatoryJobsReturned": 1728,
                    "resultAggregationReached": True,
                    "validationReached": True,
                    "validationSuccess": False,
                    "failedValidationGates": ["budgetConservationPass"],
                    "artifactFilesWritten": 0,
                    "outcomesInspected": False,
                    "terminationReason": "intervention-suppressed Selection MemoryUpdate proposals lacked an explicit native rejection-bucket disposition, violating the inherited proposal-partition identity",
                },
                {
                    "attemptOrdinal": 5,
                    "repositoryCommit": git_commit,
                    "calibrationJobsReturned": 192,
                    "confirmatoryJobsReturned": 1728,
                    "resultAggregationReached": True,
                    "artifactFilesWritten": "see artifact_manifest.json",
                    "outcomesInspected": True,
                    "terminationReason": None,
                },
            ],
            "semanticScopeChanged": False,
            "runtimeOptimization": "fixture-validated exact serial summary projection for non-full-trace rows, with state snapshots and terminal rescans only after native state changes and distance AUC recomputed only after accepted swaps",
            "accountingCorrection": "intervention-suppressed Swap and MemoryUpdate proposals use explicit rejected_s06_* dispositions so the inherited native proposal partition remains exhaustive",
        },
    )
    (output / "execution_commands.log").write_text(
        "\n".join(
            [
                "python -m pytest -q tests/test_regeneration_assisted_rescue.py",
                "python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py tests/test_regeneration_dynamic_faults.py tests/test_regeneration_nudge_recovery.py tests/test_regeneration_assisted_rescue.py",
                "ruff check src/regeneration/assisted_rescue.py tests/test_regeneration_assisted_rescue.py scripts/build_regeneration_s06.py src/regeneration/__init__.py",
                f"OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/build_regeneration_s06.py --artifacts-dir {output} --workers {workers}",
                "NOTE: four outcome-blind attempts were rejected before artifact emission (three performance restarts and one ledger-validation failure); execution_attempts.json records all attempts and the unchanged scientific scope.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outcome = _report(
        output,
        results,
        contrasts,
        primary,
        sensitivity,
        tradeoffs,
        validations,
        git_commit,
    )
    _write_json(
        output / "outcome_classification.json",
        {
            "schemaVersion": "e05.s06.outcome-classification.v1",
            "researchStepId": "S06",
            "classification": outcome,
            "validationSuccess": validations["validationSummary"]["success"],
            "holmRejectionCount": int(primary["rejectAtFamilywise0_05"].sum()),
        },
    )
    _artifact_manifest(output, git_commit)


def build(specification: Mapping[str, Any], workers: int) -> dict[str, Any]:
    _validate_inputs()
    validate_assisted_spec(specification)
    reconstructed = _reconstruct_jobs(_load_json(S04_CONFIG))
    bases = _base_jobs(reconstructed)
    calibration_bases = [base for base in bases if base.replicate in {0, 1}]
    confirmatory_bases = [base for base in bases if base.replicate in {2, 3}]
    calibration_jobs = [_active_job(base) for base in calibration_bases]
    calibration_run = _run_jobs(calibration_jobs, workers, "S06 calibration active")
    if calibration_run["failures"]:
        raise RuntimeError(f"S06 calibration failures: {calibration_run['failures'][:3]}")
    control_jobs, assignment_rows = _construct_assignments(
        calibration_jobs,
        calibration_run["results"],
        confirmatory_bases,
    )
    confirmatory_jobs = _direct_confirmatory_jobs(confirmatory_bases) + control_jobs
    confirmatory_run = _run_jobs(confirmatory_jobs, workers, "S06 confirmatory panel")
    all_failures = calibration_run["failures"] + confirmatory_run["failures"]
    results = pd.DataFrame(
        calibration_run["results"] + confirmatory_run["results"]
    ).sort_values(["assistedCaseId", "split", "arm", "matchingChoice"]).reset_index(
        drop=True
    )
    scenarios = pd.DataFrame(
        calibration_run["scenarios"] + confirmatory_run["scenarios"]
    ).sort_values(["assistedCaseId", "split", "arm", "matchingChoice"]).reset_index(
        drop=True
    )
    assignments = pd.DataFrame(assignment_rows).sort_values(
        ["matchingChoice", "stratumJson", "recipientCaseId"]
    ).reset_index(drop=True)
    contrasts = _contrasts(results).sort_values(
        ["controlArm", "matchingChoice", "assistedCaseId"]
    ).reset_index(drop=True)
    primary = _primary_tests(contrasts)
    sensitivity = _sensitivity_tests(contrasts)
    tradeoffs = _tradeoff_summary(contrasts)
    traces = calibration_run["traces"] + confirmatory_run["traces"]
    validations = _validate_panel(
        reconstructed,
        results,
        scenarios,
        assignments,
        contrasts,
        primary,
        sensitivity,
        all_failures,
        traces,
    )
    if not validations["validationSummary"]["success"]:
        failed = [
            key
            for key, value in validations["validationSummary"].items()
            if key.endswith("Pass") and not value
        ]
        raise RuntimeError(f"S06 validation failed: {failed}")
    return {
        "reconstructed": reconstructed,
        "results": results,
        "scenarios": scenarios,
        "assignments": assignments,
        "contrasts": contrasts,
        "primary": primary,
        "sensitivity": sensitivity,
        "tradeoffs": tradeoffs,
        "traces": traces,
        "validations": validations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/artifacts/research_steps/S06"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be between 1 and 8")
    if _git("status", "--porcelain"):
        raise SystemExit("repository must be clean before canonical S06 execution")
    specification = _load_json(CONFIG)
    panel = build(specification, args.workers)
    git_commit = _git("rev-parse", "HEAD")
    write_outputs(
        args.artifacts_dir,
        specification,
        panel["reconstructed"],
        panel["results"],
        panel["scenarios"],
        panel["assignments"],
        panel["contrasts"],
        panel["primary"],
        panel["sensitivity"],
        panel["tradeoffs"],
        panel["traces"],
        panel["validations"],
        args.workers,
        git_commit,
    )
    print(json.dumps(panel["validations"]["validationSummary"], indent=2))


if __name__ == "__main__":
    main()
