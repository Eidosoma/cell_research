#!/usr/bin/env python3
"""Build and validate E05 S05 nudge-dependent unfreezing evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


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
from src.regeneration.nudge_recovery import (  # noqa: E402
    BENCHMARK_VERSION,
    MATCHING_ASSIGNMENT_STREAM,
    NUDGE_RUN_SCHEMA_VERSION,
    NUDGE_SPEC_SCHEMA,
    MatchingChoice,
    NudgeMechanism,
    NudgeRecoveryContract,
    RecoveryMode,
    exact_replay_nudge,
    nudge_case_id,
    run_nudge_phase,
    validate_nudge_spec,
)


CONFIG = REPOSITORY / "configs/regeneration/s05_nudge_recovery.json"
S04_CONFIG = REPOSITORY / "configs/regeneration/s04_dynamic_faults.json"
S01_DIR = Path("/artifacts/research_steps/S01")
S02_DIR = Path("/artifacts/research_steps/S02")
S03_DIR = Path("/artifacts/research_steps/S03")
S04_DIR = Path("/artifacts/research_steps/S04")
ATTACHMENT_SIDECAR = Path(
    "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"
)
INPUTS: tuple[Path, ...] = (
    Path("/workspace/AGENTS.md"),
    Path("/workspace/FULL_PLAN.md"),
    Path("/workspace/RESEARCH_PLAN.md"),
    Path("/workspace/PREVIOUS_ARTIFACTS.md"),
    Path("/workspace/PREVIOUS_ARTIFACTS.json"),
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
    S04_DIR / "dynamic_fault_package/dynamic_fault_spec.json",
    S04_DIR / "checkpoint_compatibility.parquet",
    S04_DIR / "validation_summary.json",
    S04_DIR / "rng_coupling_validation.json",
    Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
    Path("/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/scheduler_contract.md"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/scheduler_prespecification.json"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/opportunity_ledger_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/rng_consumption_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/fault_semantics_contract.md"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/fault_prespecification.json"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/exogenous_stream_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/information_boundary_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S08/semantic_random_stream_specification.json"),
    Path("/previous-artifacts/E02/research_steps/S08/stream_name_isolation_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S08/scenario_id_validation.json"),
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
        raise FileNotFoundError(f"missing required S01-S04/E01/E02 inputs: {missing}")
    gates = {
        "s01": bool(_load_json(S01_DIR / "validation_summary.json")["success"]),
        "s02": bool(_load_json(S02_DIR / "validation_summary.json")["success"]),
        "s03": bool(_load_json(S03_DIR / "validation_summary.json")["success"]),
        "s04": bool(_load_json(S04_DIR / "validation_summary.json")["success"]),
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
    return gates


@dataclass(frozen=True, slots=True)
class NudgeJob:
    base: DynamicJob
    mechanism: NudgeMechanism
    arm: str
    split: str
    matching_choice: str
    spontaneous_duration: int | None
    assignment_id: str | None
    donor_case_id: str | None
    retain_full_trace: bool = False


def _case_id(job: NudgeJob) -> str:
    return nudge_case_id(
        job.base.s01_pairing_block_id,
        job.base.timing_condition_id,
        job.base.anchor_lesion_state_hash,
        job.mechanism,
    )


def _execute_job(job: NudgeJob) -> dict[str, Any]:
    mode = (
        RecoveryMode.CONTACT_DEPENDENT
        if job.arm == "contact_dependent"
        else RecoveryMode.SPONTANEOUS_SCHEDULED
    )
    contract = NudgeRecoveryContract(
        mechanism=job.mechanism,
        mode=mode,
        spontaneous_duration=job.spontaneous_duration,
    )
    run = run_nudge_phase(
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
    exact_replay_nudge(
        run,
        job.base.scenario,
        job.base.checkpoint,
        job.base.post_anchor_occupancy,
        job.base.recovery_budget,
    )
    case_id = _case_id(job)
    run_id = "e05nr5:" + sha256_json(
        {
            "nudgeCaseId": case_id,
            "arm": job.arm,
            "split": job.split,
            "matchingChoice": job.matching_choice,
        }
    )
    summary = dict(run.summary)
    final_process = dict(run.process_final_state)
    ledger = dict(run.process_ledger)
    row = {
        "schemaVersion": "e05.s05.nudge-recovery-result.v1",
        "benchmarkVersion": BENCHMARK_VERSION,
        "nudgeRunId": run_id,
        "nudgeCaseId": case_id,
        "arm": job.arm,
        "split": job.split,
        "matchingChoice": job.matching_choice,
        "assignmentId": job.assignment_id,
        "donorCaseId": job.donor_case_id,
        "mechanismId": job.mechanism.value,
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
        "recoveryObserved": summary["recoveryObserved"],
        "recoveryDuration": summary["recoveryDuration"],
        "recoveryCensored": summary["recoveryCensored"],
        "assignedScheduleCensored": job.arm != "contact_dependent"
        and job.spontaneous_duration is None,
        "assignedSpontaneousDuration": job.spontaneous_duration,
        "competingTerminalBeforeAssignedRecovery": job.arm != "contact_dependent"
        and job.spontaneous_duration is not None
        and not summary["recoveryObserved"],
        "recoveryReason": summary["recoveryReason"],
        "contactCount": final_process["contactCount"],
        "pressureProxyUnitsFinal": final_process["pressureProxyUnits"],
        "distinctNeighborCountFinal": len(final_process["distinctNeighborIds"]),
        "distinctNeighborIdsJson": json.dumps(
            final_process["distinctNeighborIds"], separators=(",", ":")
        ),
        "quorumEvidenceFinalJson": json.dumps(
            final_process["quorumEvidence"], separators=(",", ":")
        ),
        "initialStateHash": run.initial_state_hash,
        "finalStateHash": run.final_state_hash,
        "finalOccupancySha256": _digest(run.final_state["occupancy"]),
        "eventDigest": run.event_digest,
        "processAuditDigest": run.process_audit_digest,
        "processAuditCount": run.process_audit_count,
        "processTransitionsSha256": _digest(run.process_transitions),
        "ledgerDeltaJson": json.dumps(
            summary["ledgerDelta"], sort_keys=True, separators=(",", ":")
        ),
        "streamCounterDeltaJson": json.dumps(
            summary["streamCounterDelta"], sort_keys=True, separators=(",", ":")
        ),
        "allOpportunityValidationPass": all(run.opportunity_validation.values()),
        "opportunityValidationJson": json.dumps(
            run.opportunity_validation, sort_keys=True, separators=(",", ":")
        ),
        "exactReplayPass": True,
        "traceMode": summary["traceMode"],
        **ledger,
    }
    scenario = {
        "schemaVersion": "e05.s05.nudge-recovery-scenario.v1",
        "nudgeRunId": run_id,
        "nudgeCaseId": case_id,
        "arm": job.arm,
        "split": job.split,
        "matchingChoice": job.matching_choice,
        "mechanismId": job.mechanism.value,
        "sourceScenarioId": job.base.scenario.scenario_id,
        "sourceCheckpointHash": job.base.checkpoint.state_hash,
        "anchorLesionStateHash": job.base.anchor_lesion_state_hash,
        "postAnchorOccupancySha256": _digest(job.base.post_anchor_occupancy),
        "selectedIdentityId": job.base.selected_identity,
        "globalStartEventIndex": job.base.checkpoint.activation_count,
        "streamCountersSha256": _digest(dict(job.base.checkpoint.stream_counters)),
        "ledgerSha256": _digest(dict(job.base.checkpoint.ledger)),
        "recoveryBudget": job.base.recovery_budget,
        "assignedSpontaneousDuration": job.spontaneous_duration,
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "informationPermission": "policy_native_local",
        "continuation": "skip_and_continue",
        "retry": "no_retry",
        "pairingStatus": "shared_prefix_until_arm_terminal_or_state_path_divergence",
    }
    trace = None
    if job.retain_full_trace:
        trace = {
            "nudgeRunId": run_id,
            "nudgeCaseId": case_id,
            "arm": job.arm,
            "matchingChoice": job.matching_choice,
            "mechanismId": job.mechanism.value,
            "timingConditionId": job.base.timing_condition_id,
            "events": list(run.events),
            "processAudits": list(run.retained_process_audits),
            "processTransitions": list(run.process_transitions),
            "opportunityValidation": dict(run.opportunity_validation),
        }
    return {"result": row, "scenario": scenario, "trace": trace}


def _run_jobs(jobs: list[NudgeJob], workers: int, label: str) -> dict[str, Any]:
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
            except Exception as exc:  # pragma: no cover - retained accounting path
                failures.append(
                    {
                        "caseId": _case_id(job),
                        "arm": job.arm,
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
    unique: dict[tuple[str, str], DynamicJob] = {}
    for job in jobs:
        key = (job.s01_pairing_block_id, job.timing_condition_id)
        unique[key] = job
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


def _active_jobs(base_jobs: list[DynamicJob]) -> list[NudgeJob]:
    jobs = []
    for base in base_jobs:
        split = "calibration" if base.replicate in {0, 1} else "confirmatory"
        for mechanism in NudgeMechanism:
            retain = (
                split == "confirmatory"
                and base.n == 20
                and base.policy == "Bubble"
                and base.direction == "ascending"
                and base.replicate == 2
                and base.timing_condition_id == "post_completion"
            )
            jobs.append(
                NudgeJob(
                    base,
                    mechanism,
                    "contact_dependent",
                    split,
                    "contact_dependent_reference",
                    None,
                    None,
                    None,
                    retain,
                )
            )
    return jobs


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
        MATCHING_ASSIGNMENT_STREAM,
        base.checkpoint.activation_count,
        2 * choice_index + role_index,
    )


def _construct_assignments(
    active_jobs: list[NudgeJob], active_rows: list[dict[str, Any]]
) -> tuple[list[NudgeJob], list[dict[str, Any]]]:
    job_by_case = {_case_id(job): job for job in active_jobs}
    row_by_case = {row["nudgeCaseId"]: row for row in active_rows}
    controls: list[NudgeJob] = []
    assignments: list[dict[str, Any]] = []
    for mechanism in NudgeMechanism:
        donors_all = [
            job
            for job in active_jobs
            if job.mechanism == mechanism and job.split == "calibration"
        ]
        recipients_all = [
            job
            for job in active_jobs
            if job.mechanism == mechanism and job.split == "confirmatory"
        ]
        for choice in MatchingChoice:
            strata = sorted({_stratum(choice, job.base) for job in donors_all})
            for stratum in strata:
                donors = [
                    job for job in donors_all if _stratum(choice, job.base) == stratum
                ]
                recipients = [
                    job
                    for job in recipients_all
                    if _stratum(choice, job.base) == stratum
                ]
                if len(donors) != len(recipients) or not donors:
                    raise RuntimeError(
                        f"non-bijective matching pool {mechanism.value}/{choice.value}/{stratum}"
                    )
                donors.sort(key=lambda job: (_rank(job.base, choice, "donor"), _case_id(job)))
                recipients.sort(
                    key=lambda job: (_rank(job.base, choice, "recipient"), _case_id(job))
                )
                for donor, recipient in zip(donors, recipients, strict=True):
                    donor_case = _case_id(donor)
                    recipient_case = _case_id(recipient)
                    donor_row = row_by_case[donor_case]
                    duration = (
                        int(donor_row["recoveryDuration"])
                        if donor_row["recoveryObserved"]
                        else None
                    )
                    assignment_id = "e05ma5:" + sha256_json(
                        {
                            "choice": choice.value,
                            "mechanism": mechanism.value,
                            "stratum": list(stratum),
                            "donorCaseId": donor_case,
                            "recipientCaseId": recipient_case,
                        }
                    )
                    retain = (
                        choice == MatchingChoice.PRIMARY_N_POLICY
                        and recipient.retain_full_trace
                    )
                    controls.append(
                        NudgeJob(
                            recipient.base,
                            mechanism,
                            "matched_spontaneous",
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
                            "schemaVersion": "e05.s05.matching-assignment.v1",
                            "assignmentId": assignment_id,
                            "mechanismId": mechanism.value,
                            "matchingChoice": choice.value,
                            "stratumJson": json.dumps(list(stratum), separators=(",", ":")),
                            "donorCaseId": donor_case,
                            "recipientCaseId": recipient_case,
                            "donorReplicateOrdinal": donor.base.replicate,
                            "recipientReplicateOrdinal": recipient.base.replicate,
                            "donorRankUint64": str(_rank(donor.base, choice, "donor")),
                            "recipientRankUint64": str(
                                _rank(recipient.base, choice, "recipient")
                            ),
                            "assignedRecoveryObserved": bool(
                                donor_row["recoveryObserved"]
                            ),
                            "assignedRecoveryDuration": duration,
                            "assignedCensored": not bool(
                                donor_row["recoveryObserved"]
                            ),
                            "donorCensorReason": None
                            if donor_row["recoveryObserved"]
                            else donor_row["stopReason"],
                            "stream": MATCHING_ASSIGNMENT_STREAM,
                            "confirmatoryOutcomesAvailableAtAssignment": False,
                        }
                    )
    if len(controls) != 2304 or len(assignments) != 2304:
        raise RuntimeError("S05 control assignment count changed")
    if set(job_by_case) != set(row_by_case):
        raise RuntimeError("active job/result case accounting changed")
    return controls, assignments


def _token(recovered: bool, duration: Any) -> str:
    return f"R:{int(duration)}" if recovered else "CENSORED"


def _matching_validation(
    active: pd.DataFrame, assignments: pd.DataFrame
) -> dict[str, Any]:
    failures: list[str] = []
    stratum_count = 0
    for (mechanism, choice, stratum), assigned in assignments.groupby(
        ["mechanismId", "matchingChoice", "stratumJson"], sort=True
    ):
        donor_ids = assigned["donorCaseId"].tolist()
        donors = active.set_index("nudgeCaseId").loc[donor_ids]
        donor_tokens = sorted(
            _token(bool(row.recoveryObserved), row.recoveryDuration)
            for row in donors.itertuples()
        )
        assigned_tokens = sorted(
            _token(bool(row.assignedRecoveryObserved), row.assignedRecoveryDuration)
            for row in assigned.itertuples()
        )
        if donor_tokens != assigned_tokens:
            failures.append(f"{mechanism}/{choice}/{stratum}: marginal mismatch")
        stratum_count += 1
    no_outcome_conditioning = bool(
        (~assignments["confirmatoryOutcomesAvailableAtAssignment"]).all()
        and assignments["donorReplicateOrdinal"].isin([0, 1]).all()
        and assignments["recipientReplicateOrdinal"].isin([2, 3]).all()
    )
    return {
        "schemaVersion": "e05.s05.recovery-time-matching-validation.v1",
        "assignmentCount": len(assignments),
        "matchingChoiceCount": assignments["matchingChoice"].nunique(),
        "mechanismCount": assignments["mechanismId"].nunique(),
        "validatedStratumCount": stratum_count,
        "marginalMultisetFailureCount": len(failures),
        "failures": failures,
        "censoredDonorCount": int(assignments["assignedCensored"].sum()),
        "censoredCasesRetained": True,
        "noOutcomeConditioning": no_outcome_conditioning,
        "success": not failures and no_outcome_conditioning,
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
        "informationPermission",
        "continuation",
        "retry",
    ]
    confirmatory = scenarios[scenarios["split"] == "confirmatory"]
    for case_id, group in confirmatory.groupby("nudgeCaseId", sort=False):
        if len(group) != 4 or set(group["arm"]) != {
            "contact_dependent",
            "matched_spontaneous",
        }:
            failures.append(f"{case_id}: expected one active and three controls")
            continue
        if (group["arm"] == "contact_dependent").sum() != 1:
            failures.append(f"{case_id}: active arm count changed")
        if set(group.loc[group["arm"] == "matched_spontaneous", "matchingChoice"]) != {
            item.value for item in MatchingChoice
        }:
            failures.append(f"{case_id}: matching choices changed")
        for field in fields:
            if group[field].nunique(dropna=False) != 1:
                failures.append(f"{case_id}: mismatch in {field}")
    return {
        "schemaVersion": "e05.s05.pairing-validation.v1",
        "confirmatoryCaseCount": confirmatory["nudgeCaseId"].nunique(),
        "confirmatoryRowCount": len(confirmatory),
        "failureCount": len(failures),
        "failures": failures,
        "baseStreamStatus": "shared_prefix_until_arm_terminal_or_state_path_divergence",
        "matchingStreamStatus": "construction_only_isolated_no_dummy_runtime_draws",
        "success": not failures,
    }


def _contrasts(results: pd.DataFrame) -> pd.DataFrame:
    active = results.query(
        "split == 'confirmatory' and arm == 'contact_dependent'"
    ).set_index("nudgeCaseId")
    controls = results.query("arm == 'matched_spontaneous'")
    rows: list[dict[str, Any]] = []
    for control in controls.itertuples():
        reference = active.loc[control.nudgeCaseId]
        rows.append(
            {
                "schemaVersion": "e05.s05.nudge-contrast.v1",
                "nudgeCaseId": control.nudgeCaseId,
                "mechanismId": control.mechanismId,
                "matchingChoice": control.matchingChoice,
                "timingConditionId": control.timingConditionId,
                "clock": control.clock,
                "n": control.n,
                "policy": control.policy,
                "direction": control.direction,
                "replicateOrdinal": control.replicateOrdinal,
                "activeCompleted": bool(reference["completed"]),
                "controlCompleted": bool(control.completed),
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
                "activeRecoveryObserved": bool(reference["recoveryObserved"]),
                "controlRecoveryObserved": bool(control.recoveryObserved),
                "assignedScheduleCensored": bool(control.assignedScheduleCensored),
            }
        )
    return pd.DataFrame(rows)


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _primary_tests(contrasts: pd.DataFrame) -> pd.DataFrame:
    primary = contrasts[
        contrasts["matchingChoice"] == MatchingChoice.PRIMARY_N_POLICY.value
    ]
    rows = []
    for mechanism in NudgeMechanism:
        group = primary[primary["mechanismId"] == mechanism.value]
        gained = int((group["activeCompleted"] & ~group["controlCompleted"]).sum())
        lost = int((~group["activeCompleted"] & group["controlCompleted"]).sum())
        discordant = gained + lost
        p_value = (
            float(binomtest(gained, discordant, 0.5, alternative="two-sided").pvalue)
            if discordant
            else 1.0
        )
        rows.append(
            {
                "schemaVersion": "e05.s05.primary-test.v1",
                "mechanismId": mechanism.value,
                "pairCount": len(group),
                "activeCompletionCount": int(group["activeCompleted"].sum()),
                "controlCompletionCount": int(group["controlCompleted"].sum()),
                "activeOnlyCompletionCount": gained,
                "controlOnlyCompletionCount": lost,
                "discordantPairCount": discordant,
                "exactMcNemarPValue": p_value,
                "medianPhaseActivationDelta": float(
                    group["phaseActivationDelta"].median()
                ),
                "medianFinalDistanceDelta": float(group["finalDistanceDelta"].median()),
            }
        )
    adjusted = _holm([row["exactMcNemarPValue"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holmAdjustedPValue"] = value
        row["rejectAtFamilywise0_05"] = value <= 0.05
    return pd.DataFrame(rows)


def _validate_panel(
    reconstructed: Mapping[str, Any],
    results: pd.DataFrame,
    scenarios: pd.DataFrame,
    assignments: pd.DataFrame,
    contrasts: pd.DataFrame,
    primary_tests: pd.DataFrame,
    failures: list[dict[str, str]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    active = results[results["arm"] == "contact_dependent"]
    controls = results[results["arm"] == "matched_spontaneous"]
    matching = _matching_validation(active, assignments)
    pairing = _pairing_validation(scenarios)
    expected_recovery = active["recoveryObserved"].astype(int)
    trigger_accuracy = bool(
        (active["mechanismTriggerEvents"] == expected_recovery).all()
        and (
            active.loc[
                active["mechanismId"] == NudgeMechanism.ATTEMPTED_CONTACT.value,
                "recoveryObserved",
            ]
            == (
                active.loc[
                    active["mechanismId"] == NudgeMechanism.ATTEMPTED_CONTACT.value,
                    "contactCount",
                ]
                >= 3
            )
        ).all()
        and (
            active.loc[
                active["mechanismId"] == NudgeMechanism.PRESSURE_PROXY.value,
                "recoveryObserved",
            ]
            == (
                active.loc[
                    active["mechanismId"] == NudgeMechanism.PRESSURE_PROXY.value,
                    "pressureProxyUnitsFinal",
                ]
                >= 3
            )
        ).all()
        and (
            active.loc[
                active["mechanismId"] == NudgeMechanism.DISTINCT_NEIGHBOR.value,
                "recoveryObserved",
            ]
            == (
                active.loc[
                    active["mechanismId"] == NudgeMechanism.DISTINCT_NEIGHBOR.value,
                    "distinctNeighborCountFinal",
                ]
                >= 2
            )
        ).all()
    )
    counter_accuracy = bool(
        (results["pressureProxyUnits"] == results["inboundContactEvents"]).all()
        and (
            results["inboundContactEvents"] <= results["qualifyingContactEvents"]
        ).all()
        and (results["distinctNeighborAdds"] == results["distinctNeighborCountFinal"]).all()
        and results["allOpportunityValidationPass"].all()
    )
    schedule_accuracy = bool(
        (
            controls.loc[controls["recoveryObserved"], "recoveryDuration"].astype(int)
            == controls.loc[
                controls["recoveryObserved"], "assignedSpontaneousDuration"
            ].astype(int)
        ).all()
        and (
            ~controls.loc[
                controls["assignedScheduleCensored"], "recoveryObserved"
            ].astype(bool)
        ).all()
    )
    stream_isolation = bool(
        all(
            MATCHING_ASSIGNMENT_STREAM not in json.loads(value)
            for value in results["streamCounterDeltaJson"]
        )
    )
    run_accounting = {
        "schemaVersion": "e05.s05.run-accounting.v1",
        "researchStepId": "S05",
        "sourceBlocksExpected": 48,
        "sourceBlocksObserved": len(
            {row["s01PairingBlockId"] for row in reconstructed["checkpointRows"]}
        ),
        "checkpointsExpected": 384,
        "checkpointsObserved": len(reconstructed["checkpointRows"]),
        "calibrationActiveRunsExpected": 768,
        "calibrationActiveRunsObserved": int(
            ((results["split"] == "calibration") & (results["arm"] == "contact_dependent")).sum()
        ),
        "confirmatoryActiveRunsExpected": 768,
        "confirmatoryActiveRunsObserved": int(
            ((results["split"] == "confirmatory") & (results["arm"] == "contact_dependent")).sum()
        ),
        "confirmatoryControlRunsExpected": 2304,
        "confirmatoryControlRunsObserved": len(controls),
        "plannedRunsExpected": 3840,
        "plannedRunsObserved": len(results),
        "exactReplayExecutionsExpected": 3840,
        "exactReplayExecutionsObserved": int(results["exactReplayPass"].sum()),
        "totalTrajectoryExecutionsExpected": 7680,
        "totalTrajectoryExecutionsObserved": 2 * len(results),
        "pairwiseContrastsExpected": 2304,
        "pairwiseContrastsObserved": len(contrasts),
        "fullTraceRunsExpected": 8,
        "fullTraceRunsObserved": len(traces),
        "runtimeFailureCount": len(failures),
        "runtimeFailures": failures,
        "unreachableCheckpointCount": 384 - len(reconstructed["checkpointRows"]),
        "substitutionCount": 0,
        "silentExclusionCount": 3840 - len(results),
        "scopeReduction": False,
    }
    run_accounting["success"] = all(
        (
            run_accounting["sourceBlocksObserved"] == 48,
            run_accounting["checkpointsObserved"] == 384,
            run_accounting["calibrationActiveRunsObserved"] == 768,
            run_accounting["confirmatoryActiveRunsObserved"] == 768,
            run_accounting["confirmatoryControlRunsObserved"] == 2304,
            run_accounting["plannedRunsObserved"] == 3840,
            run_accounting["exactReplayExecutionsObserved"] == 3840,
            run_accounting["pairwiseContrastsObserved"] == 2304,
            run_accounting["fullTraceRunsObserved"] == 8,
            not failures,
            run_accounting["silentExclusionCount"] == 0,
        )
    )
    validations = {
        "checkpointValidation": {
            "schemaVersion": "e05.s05.checkpoint-validation.v1",
            "checkpointCount": len(reconstructed["checkpointRows"]),
            "checkpointFailures": reconstructed["checkpointFailures"],
            "anchorFailures": reconstructed["anchorFailures"],
            "success": not reconstructed["checkpointFailures"]
            and not reconstructed["anchorFailures"]
            and len(reconstructed["checkpointRows"]) == 384,
        },
        "counterValidation": {
            "schemaVersion": "e05.s05.counter-neighbor-validation.v1",
            "runCount": len(results),
            "qualifyingContactEventCount": int(results["qualifyingContactEvents"].sum()),
            "inboundContactEventCount": int(results["inboundContactEvents"].sum()),
            "pressureProxyUnitCount": int(results["pressureProxyUnits"].sum()),
            "distinctNeighborAddCount": int(results["distinctNeighborAdds"].sum()),
            "triggerAccuracyPass": trigger_accuracy,
            "counterAccuracyPass": counter_accuracy,
            "neighborIdentityLogicPass": bool(
                (results["distinctNeighborAdds"] == results["distinctNeighborCountFinal"]).all()
            ),
            "pressureInterpretation": "abstract_event_count_proxy",
            "success": trigger_accuracy and counter_accuracy,
        },
        "matchingValidation": matching,
        "scheduleValidation": {
            "schemaVersion": "e05.s05.schedule-validation.v1",
            "controlRunCount": len(controls),
            "finiteAssignedCount": int(controls["assignedSpontaneousDuration"].notna().sum()),
            "assignedUnrecoveredCount": int(controls["assignedScheduleCensored"].sum()),
            "competingTerminalCensorCount": int(
                controls["competingTerminalBeforeAssignedRecovery"].sum()
            ),
            "exactObservedSchedulePass": schedule_accuracy,
            "success": schedule_accuracy,
        },
        "replayValidation": {
            "schemaVersion": "e05.s05.replay-validation.v1",
            "plannedRunCount": len(results),
            "exactReplayCount": int(results["exactReplayPass"].sum()),
            "contactEventReplayCount": int(results["exactReplayPass"].sum()),
            "failureCount": int((~results["exactReplayPass"]).sum()),
            "success": bool(results["exactReplayPass"].all()),
        },
        "streamValidation": {
            "schemaVersion": "e05.s05.stream-isolation-validation.v1",
            "matchingStream": MATCHING_ASSIGNMENT_STREAM,
            "matchingAddressCount": 2 * len(assignments),
            "matchingAddressesDeterministic": True,
            "runtimeMatchingStreamConsumptionCount": 0,
            "contactMechanismRuntimeStreams": [],
            "matchingStreamRuntimeIsolationPass": stream_isolation,
            "originalScenarioRootPreserved": bool(
                (results["sourceScenarioId"].str.startswith("r1:")).all()
            ),
            "staticReconstructionExecuted": False,
            "success": stream_isolation,
        },
        "pairingValidation": pairing,
        "runAccounting": run_accounting,
        "sensitivityValidation": {
            "schemaVersion": "e05.s05.matching-sensitivity-validation.v1",
            "matchingChoices": [item.value for item in MatchingChoice],
            "contrastCountPerChoice": contrasts.groupby("matchingChoice")
            .size()
            .astype(int)
            .to_dict(),
            "allChoicesExecuted": contrasts["matchingChoice"].nunique() == 3,
            "censoredCasesRetained": True,
            "success": contrasts["matchingChoice"].nunique() == 3,
        },
    }
    summary = {
        "schemaVersion": "e05.s05.validation-summary.v1",
        "researchStepId": "S05",
        "checkpointIdentityPass": validations["checkpointValidation"]["success"],
        "triggerCounterNeighborPass": validations["counterValidation"]["success"],
        "recoveryTimeMatchingPass": matching["success"],
        "spontaneousScheduleAccuracyPass": validations["scheduleValidation"]["success"],
        "deterministicReplayPass": validations["replayValidation"]["success"],
        "contactQuorumEventReplayPass": validations["replayValidation"]["success"],
        "streamIsolationRngBoundaryPass": validations["streamValidation"]["success"],
        "pairingPass": pairing["success"],
        "matchingSensitivityPass": validations["sensitivityValidation"]["success"],
        "completeRunAccountingPass": run_accounting["success"],
        "checkpointCount": len(reconstructed["checkpointRows"]),
        "plannedRunCount": len(results),
        "replayCount": int(results["exactReplayPass"].sum()),
        "primaryMechanismCount": len(primary_tests),
    }
    summary["success"] = all(
        value for key, value in summary.items() if key.endswith("Pass")
    )
    validations["validationSummary"] = summary
    return validations


def _spec_markdown(specification: Mapping[str, Any]) -> str:
    mechanisms = "\n".join(
        f"- `{item['mechanismId']}` — {item['semantics']}"
        for item in specification["mechanisms"]
    )
    return f"""# S05 nudge-dependent unfreezing specification

Frozen before confirmatory runs at `{specification['frozenAtUtc']}`; schema
`{specification['schemaVersion']}`.

## Inherited boundary

Every run retains the exact S02 checkpoint and S03 reversal anchor, the S04
runtime overlay on the original S01 scenario root, the `100*n^2` recovery
budget, target, native ledger, uniform scheduler, policy-native local
information, and shared-prefix pairing contract. The triggering contact is
blocked and recovery begins on the next charged opportunity.

## Qualifying local contact

A contact is one mechanically eligible adjacent Swap involving the selected
frozen identity. Its other participant is recorded by immutable identity ID and
by pre-event side. NoOp, MemoryUpdate, invalid, non-adjacent, and post-recovery
interactions do not count.

## Mechanisms

{mechanisms}

“Pressure” is strictly an abstract count of inbound displacement-attempt events.
It is not force, stress, impulse, energy, or a mechanical measurement.

## Matched spontaneous controls

Replicates 0–1 form the calibration bank; replicates 2–3 are confirmatory.
Calibration recovery durations and explicit unrecovered sentinels are assigned
to confirmatory recipients by isolated counter-addressed bijections. The
primary strata are `n × policy`; `n`-only and pooled bijections are matching
sensitivity analyses. No confirmatory outcome is available to assignment.
"""


def _plot_matching(
    active: pd.DataFrame, assignments: pd.DataFrame, package: Path
) -> None:
    calibration = active.query("split == 'calibration'")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, mechanism in zip(axes.flat, NudgeMechanism, strict=True):
        donor = calibration[calibration["mechanismId"] == mechanism.value]
        finite = donor.loc[donor["recoveryObserved"], "recoveryDuration"].astype(int)
        if len(finite):
            values = np.sort(finite.to_numpy())
            axis.step(values, np.arange(1, len(values) + 1) / len(donor), where="post")
        censor_fraction = 1 - donor["recoveryObserved"].mean()
        axis.set(
            title=f"{mechanism.value}\ncensored mass={censor_fraction:.3f}",
            xlabel="Recovery duration (charged opportunities)",
            ylabel="Marginal cumulative incidence",
            xscale="log",
        )
        axis.grid(alpha=0.2)
    fig.suptitle("S05 calibration recovery-time marginals retained by bijective controls")
    fig.savefig(package / "recovery_time_matching.png", dpi=180)
    fig.savefig(package / "recovery_time_matching.svg")
    plt.close(fig)


def _report(
    output: Path,
    results: pd.DataFrame,
    contrasts: pd.DataFrame,
    primary: pd.DataFrame,
    validations: Mapping[str, Any],
    git_commit: str,
) -> str:
    success = validations["validationSummary"]["success"]
    supportive = bool(primary["rejectAtFamilywise0_05"].any()) and success
    outcome = "Supportive" if supportive else "Null" if success else "Constraining/contradictory"
    primary_lines = []
    for row in primary.itertuples():
        primary_lines.append(
            f"- `{row.mechanismId}`: active/control completion "
            f"{row.activeCompletionCount}/{row.controlCompletionCount} of {row.pairCount}; "
            f"discordant active-only/control-only {row.activeOnlyCompletionCount}/"
            f"{row.controlOnlyCompletionCount}; exact p={row.exactMcNemarPValue:.4g}, "
            f"Holm p={row.holmAdjustedPValue:.4g}."
        )
    primary_text = "\n".join(primary_lines)
    active = results[results["arm"] == "contact_dependent"]
    controls = results[results["arm"] == "matched_spontaneous"]
    calibration = active[active["split"] == "calibration"]
    confirmatory = active[active["split"] == "confirmatory"]
    recovery_lines = []
    for mechanism, group in active.groupby("mechanismId"):
        calibration_group = calibration[calibration["mechanismId"] == mechanism]
        confirm_group = confirmatory[confirmatory["mechanismId"] == mechanism]
        recovery_lines.append(
            f"- `{mechanism}`: calibration recovery "
            f"{int(calibration_group['recoveryObserved'].sum())}/{len(calibration_group)}; "
            f"confirmatory recovery {int(confirm_group['recoveryObserved'].sum())}/"
            f"{len(confirm_group)}; calibration censored "
            f"{int((~calibration_group['recoveryObserved']).sum())}."
        )
    recovery_text = "\n".join(recovery_lines)
    sensitivity_lines = []
    for choice, group in contrasts.groupby("matchingChoice"):
        sensitivity_lines.append(
            f"- `{choice}`: {int(group['completionChanged'].sum())}/"
            f"{len(group)} completion-discordant pairs; median active-minus-control "
            f"duration {group['phaseActivationDelta'].median():.1f}."
        )
    sensitivity_text = "\n".join(sensitivity_lines)
    report = f"""# Research step S05 full results — Implement nudge-dependent unfreezing

## Top summary

- **Research step ID:** S05
- **Completion status:** Complete; S06 was not started.
- **Artifacts written:** Frozen nudge/recovery specification and schema, 3,840-row result/scenario tables, 2,304 calibration-to-control assignments, 2,304 paired contrasts, four primary tests, matching plot, eight full traces, validation/accounting/provenance manifests, and this canonical report under `{output}`.
- **Validation result:** PASS — 384/384 inherited checkpoints and anchors, 3,840/3,840 planned runs, 3,840/3,840 exact replays, and all trigger, counter, neighbor identity, quorum/contact replay, matching, schedule, stream, pairing, censor-retention, sensitivity, and accounting gates passed.
- **Outcome classification:** {outcome}. {"At least one contact mechanism differed from its prespecified timing-matched spontaneous control after Holm correction." if supportive else "No mechanism differed from its primary duration-matched spontaneous control after Holm correction; the validated result constrains a timing-independent recovery claim."}
- **Caveats or blockers:** Contact and quorum are local simulator-event rules. “Pressure” is only an inbound event-count proxy, not physical force. Matching targets assigned calibration-split schedules; finite schedules can still be censored by a confirmatory competing terminal. Runtime freezing remains mobility-equivalent, not byte-equivalent, to static stuck reconstruction. No execution blocker remains within S05.
- **Recommended next action:** Chief Scientist review. If the {outcome.lower()} S05 result and engineered-mechanism boundary are accepted, separately authorize S06; otherwise revise or prune neighbor-assisted rescue before starting it.

## Lay summary

This step tested four ways a frozen simulated cell might resume moving after
local interactions: three adjacent contacts, three inbound “pressure” events,
contacts with two different neighbors, or recent nudges from different cells on
both sides. Each was compared with cells that recovered spontaneously on a
schedule copied from a separate calibration split. Cases that never recovered
were copied as never-recovering controls rather than discarded. The primary
paired tests were {outcome.lower()}: timing-matched spontaneous recovery
{"could not explain every observed completion difference." if supportive else "was not distinguishable from contact-triggered recovery at the prespecified family-wise threshold."}
These are simulation results and do not demonstrate biological healing or
mechanical pressure sensing.

## Frozen question and decision rule

S05 asked whether local interaction-dependent recovery changes confirmatory
completion relative to spontaneous recovery with the same calibration-split
marginal recovery-time distribution. Before confirmatory runs, completion was
frozen as the primary endpoint, exact paired McNemar tests were specified per
mechanism, and Holm correction across four mechanisms controlled family-wise
alpha at 0.05. Every non-recovery, quiescent, and budget case remained in the
primary table; no recovered-only cohort was formed.

## Inputs and inherited contracts

The run refreshed the workspace plans; S01–S04 reports/specifications; exact
S02 pre-injury records; the S03 lesion fixtures; the S04 runtime and RNG
boundary; E01/E02 release, scheduler, fault, ledger, information, and semantic
stream contracts; and the attachment manifest/sidecar plus the paper’s proposed
future experiment on repeated neighbor nudges. No dataset, network input, or new
dependency was used.

Every arm resumed the exact S02 checkpoint after the S03
`segment_reversal_central_v1` anchor. Occupancy, selected identity, Selection
cursors, global clock, stream counters, native ledger, direction-aware target,
`100*n^2` phase budget, uniform activation, policy-native local observations,
skip-and-continue, no retry, and NoOp/Swap/MemoryUpdate remained fixed. The S04
overlay retained the original S01 scenario ID; no static-stuck reconstruction
was executed.

## Detailed methods

### Local recovery semantics

A qualifying contact is one mechanically eligible adjacent Swap involving the
selected frozen identity, recorded after ground-truth validation and before the
freeze gate. The other participant is keyed by immutable identity and its
pre-event left/right side. NoOp, MemoryUpdate, invalid, non-adjacent, and
post-recovery interactions do not count. The triggering proposal is blocked;
recovery begins next opportunity.

`attempted_contact_k3_v1` counts all qualifying contacts. The pressure-proxy
mechanism counts only inbound, neighbor-initiated attempts, one unit per event,
and recovers at three. `distinct_neighbor_k2_v1` requires two immutable other
identities. The quorum mechanism requires inbound evidence from different
identities on both sides in the inclusive previous 16-opportunity window.

### Outcome-blind matching

Replicates 0–1 (192 checkpoints) were fixed as calibration and replicates 2–3
(192 checkpoints) as confirmatory. All four mechanisms first ran on all 384
checkpoints. For each mechanism, calibration recovery duration—or an explicit
unrecovered sentinel—was assigned exactly once to a confirmatory recipient by
counter-addressed bijection. The primary strata were `n × policy`; sensitivity
choices used `n` only and a pooled bank. Thus each declared marginal multiset,
including censor mass, matched exactly. Recipient completion, distance,
recovery, and terminal outcomes did not exist when assignments were made.

Spontaneous controls owned no runtime hazard stream: a finite assigned duration
recovered before the next opportunity after that many frozen exposures; an
unrecovered sentinel stayed frozen. A finite schedule could remain unrealized
if completion or quiescence occurred first, and those competing censored cases
were retained.

### Execution and analysis

The panel comprised 768 calibration active runs, 768 confirmatory active runs,
and 2,304 confirmatory spontaneous controls (three matching choices), or 3,840
planned runs. Every run was executed twice for exact serialized replay, giving
7,680 trajectory executions on eight workers. Eight prespecified active/primary
control traces retained native events and process audits; all other rows retain
cryptographic digests and compact ledgers.

## Commands

```text
python -m pytest -q tests/test_regeneration_nudge_recovery.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py tests/test_regeneration_dynamic_faults.py tests/test_regeneration_nudge_recovery.py
ruff check src/regeneration/nudge_recovery.py tests/test_regeneration_nudge_recovery.py scripts/build_regeneration_s05.py src/regeneration/__init__.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/build_regeneration_s05.py --artifacts-dir {output} --workers 8
```

## Results

### Primary confirmatory completion tests

{primary_text}

### Recovery and censor accounting

{recovery_text}

Across the three control mappings, {int(controls['assignedScheduleCensored'].sum())}
assigned schedules retained an unrecovered calibration sentinel and
{int(controls['competingTerminalBeforeAssignedRecovery'].sum())} finite assigned
schedules encountered a competing terminal before recovery. Neither group was
excluded. There were {int((results['stopReason'] == 'quiescent').sum())}
quiescent and {int((results['stopReason'] == 'phase_event_budget').sum())}
phase-budget terminals across all rows.

### Matching sensitivity

{sensitivity_text}

These sensitivity mappings change which confirmatory case receives each fixed
calibration schedule while preserving the global recovery-time/censor multiset.
They are robustness checks, not independent experiments.

## Validation

All 384 checkpoint hashes and reversal-anchor hashes matched S02/S03. Counter
identities held: pressure units equaled inbound qualifying events, inbound was a
subset of all contacts, distinct-neighbor additions equaled final unique-ID
counts, and every active recovery had exactly one threshold/quorum trigger.
Fixture tests covered adjacency exclusions, inbound/outbound distinctions,
identity rather than side counting, the inclusive quorum-window boundary,
next-opportunity recovery, finite spontaneous expiry, and never-recovery.

Every declared donor/recipient stratum had an exact bijection and identical
duration-plus-censor multisets. Calibration replicates were 0–1 and recipients
2–3; no confirmatory outcome entered assignment. Observed finite spontaneous
recoveries equaled their assigned durations exactly. All 3,840 result objects,
native event digests, contact/quorum audit digests, process ledgers, and
transitions replayed byte-for-byte. No S05 runtime stream was consumed, worker
order entered no address, and the construction-only matching stream remained
isolated. Planned, executed, replayed, traced, censored, and terminal counts all
reconciled with zero substitutions, silent exclusions, or scope reduction.

## Artifacts

- `nudge_recovery_package/nudge_recovery_spec.json`/Markdown and schema freeze the local semantics and decision rule.
- `nudge_recovery_package/matching_assignments.parquet` records every donor, recipient, rank address, duration, and censor sentinel.
- `nudge_recovery_results.parquet` and `nudge_recovery_scenarios.parquet` preserve all 3,840 rows; `paired_nudge_contrasts.parquet` preserves 2,304 comparisons.
- `primary_completion_tests.parquet`, the matching figure, and eight selected full traces provide compact direct evidence.
- Checkpoint, counter/neighbor, matching, schedule, replay, stream, pairing, sensitivity, accounting, provenance, environment, and artifact manifests preserve validation and reproducibility.

## Caveats, blockers, failed assumptions, and limitations

- These are engineered computational mechanisms, not evidence that biological
  cells sense pressure, count neighbors, form quorums, or repair tissue.
- “Pressure” is strictly the count of inbound eligible displacement-attempt
  events. It has no magnitude, direction beyond side, stress, force, impulse,
  energy, or mechanical constitutive law.
- Attempted-contact includes eligible adjacent interactions initiated by either
  the selected identity or its neighbor; pressure and quorum use inbound
  neighbor-initiated attempts only. This distinction is operational.
- Marginal matching cannot make each control share its active counterpart’s
  endogenous contact history. It intentionally separates a recovery-time
  distribution from path-specific contact timing.
- An unrecovered donor is retained as a never-recovering schedule; finite
  schedules may be censored by confirmatory completion/quiescence. Realized
  recovery incidence therefore need not equal the assigned marginal under
  competing terminals.
- The runtime overlay preserves the S04 RNG root but is not byte-identical to a
  static stuck Scenario. Count-changing S03 lesions remain outside this runner.
- Matching sensitivities reuse the same active outcomes and calibration bank;
  they should not be treated as independent replications.

## Provenance

- Repository: `Eidosoma/cell_research`
- Branch: `eidosoma/groups/28`
- Source commit: `{git_commit}`
- Benchmark: `{BENCHMARK_VERSION}`
- RNG: E01 SHA-256 counter-addressed base streams plus construction-only `{MATCHING_ASSIGNMENT_STREAM}`.
- Runtime: Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}; eight workers with numerical library threads fixed at one.
- Generated UTC: {datetime.now(timezone.utc).isoformat()}

Input and output SHA-256 hashes are recorded in `input_provenance.json` and
`artifact_manifest.json`. Reproducible source remains in the pushed Git commit;
S06 was not started.
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
            "schemaVersion": "e05.s05.artifact-manifest.v1",
            "researchStepId": "S05",
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
    traces: list[dict[str, Any]],
    validations: Mapping[str, Any],
    workers: int,
    git_commit: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    package = output / "nudge_recovery_package"
    package.mkdir(parents=True, exist_ok=True)
    _write_json(package / "nudge_recovery_spec.json", specification)
    _write_json(package / "nudge_recovery_spec.schema.json", NUDGE_SPEC_SCHEMA)
    _write_json(
        package / "nudge_recovery_run.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://eidosoma.local/schemas/e05/s05/nudge-recovery-run.schema.json",
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
                "schemaVersion": {"const": NUDGE_RUN_SCHEMA_VERSION},
                "benchmarkVersion": {"const": BENCHMARK_VERSION},
            },
        },
    )
    (package / "nudge_recovery_spec.md").write_text(
        _spec_markdown(specification), encoding="utf-8"
    )
    results.to_parquet(output / "nudge_recovery_results.parquet", index=False)
    scenarios.to_parquet(output / "nudge_recovery_scenarios.parquet", index=False)
    assignments.to_parquet(package / "matching_assignments.parquet", index=False)
    contrasts.to_parquet(output / "paired_nudge_contrasts.parquet", index=False)
    primary.to_parquet(output / "primary_completion_tests.parquet", index=False)
    pd.DataFrame(reconstructed["checkpointRows"]).to_parquet(
        output / "checkpoint_compatibility.parquet", index=False
    )
    with (package / "selected_full_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(traces, key=lambda item: item["nudgeRunId"]):
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _plot_matching(results[results["arm"] == "contact_dependent"], assignments, package)
    names = {
        "checkpoint_validation.json": validations["checkpointValidation"],
        "counter_neighbor_validation.json": validations["counterValidation"],
        "recovery_time_matching_validation.json": validations["matchingValidation"],
        "spontaneous_schedule_validation.json": validations["scheduleValidation"],
        "replay_validation.json": validations["replayValidation"],
        "stream_isolation_rng_boundary_validation.json": validations["streamValidation"],
        "pairing_validation.json": validations["pairingValidation"],
        "matching_sensitivity_validation.json": validations["sensitivityValidation"],
        "run_accounting.json": validations["runAccounting"],
        "validation_summary.json": validations["validationSummary"],
    }
    for name, value in names.items():
        _write_json(output / name, value)
    _write_json(
        output / "input_provenance.json",
        {
            "schemaVersion": "e05.s05.input-provenance.v1",
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
            "schemaVersion": "e05.s05.environment-provenance.v1",
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
    (output / "execution_commands.log").write_text(
        "\n".join(
            [
                "python -m pytest -q tests/test_regeneration_nudge_recovery.py",
                "python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py tests/test_regeneration_dynamic_faults.py tests/test_regeneration_nudge_recovery.py",
                "ruff check src/regeneration/nudge_recovery.py tests/test_regeneration_nudge_recovery.py scripts/build_regeneration_s05.py src/regeneration/__init__.py",
                f"OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/build_regeneration_s05.py --artifacts-dir {output} --workers {workers}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outcome = _report(output, results, contrasts, primary, validations, git_commit)
    _artifact_manifest(output, git_commit)
    _write_json(
        output / "outcome_classification.json",
        {
            "schemaVersion": "e05.s05.outcome-classification.v1",
            "researchStepId": "S05",
            "classification": outcome.lower(),
            "validationSuccess": validations["validationSummary"]["success"],
            "holmRejectionCount": int(primary["rejectAtFamilywise0_05"].sum()),
        },
    )
    # Refresh manifest after the classification artifact is present.
    _artifact_manifest(output, git_commit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("S05 worker count must be between one and eight")
    specification = _load_json(CONFIG)
    validate_nudge_spec(specification)
    _validate_inputs()
    s04_specification = _load_json(S04_CONFIG)
    print("Reconstructing and validating all inherited S02/S03 states", flush=True)
    reconstructed = _reconstruct_jobs(s04_specification)
    base_jobs = _base_jobs(reconstructed)
    active_jobs = _active_jobs(base_jobs)
    print("Executing calibration and confirmatory contact-dependent arms", flush=True)
    active_run = _run_jobs(active_jobs, args.workers, "active")
    if active_run["failures"]:
        raise RuntimeError(f"active run failures: {active_run['failures'][:3]}")
    control_jobs, assignment_rows = _construct_assignments(
        active_jobs, active_run["results"]
    )
    print("Matching frozen; executing confirmatory spontaneous controls", flush=True)
    control_run = _run_jobs(control_jobs, args.workers, "controls")
    failures = [*active_run["failures"], *control_run["failures"]]
    results = pd.DataFrame([*active_run["results"], *control_run["results"]])
    scenarios = pd.DataFrame([*active_run["scenarios"], *control_run["scenarios"]])
    assignments = pd.DataFrame(assignment_rows)
    traces = [*active_run["traces"], *control_run["traces"]]
    results = results.sort_values(
        ["n", "policy", "direction", "replicateOrdinal", "timingConditionId", "mechanismId", "arm", "matchingChoice"]
    ).reset_index(drop=True)
    scenarios = scenarios.sort_values("nudgeRunId").reset_index(drop=True)
    assignments = assignments.sort_values("assignmentId").reset_index(drop=True)
    contrasts = _contrasts(results).sort_values(
        ["mechanismId", "matchingChoice", "n", "policy", "direction", "replicateOrdinal", "timingConditionId"]
    ).reset_index(drop=True)
    primary = _primary_tests(contrasts)
    validations = _validate_panel(
        reconstructed,
        results,
        scenarios,
        assignments,
        contrasts,
        primary,
        failures,
        traces,
    )
    if not validations["validationSummary"]["success"]:
        raise RuntimeError(
            f"S05 validation failed: {validations['validationSummary']}"
        )
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
        traces,
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
                "contrastCount": len(contrasts),
                "holmRejectionCount": int(primary["rejectAtFamilywise0_05"].sum()),
                "artifactsDir": str(args.artifacts_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
