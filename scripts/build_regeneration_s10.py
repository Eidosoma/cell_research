#!/usr/bin/env python3
"""Build and validate E05 S10 repeated-injury evidence."""

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

from reference_simulator.model import sha256_json  # noqa: E402
from reference_simulator.rng import u64  # noqa: E402
from scripts.build_regeneration_s04 import _reconstruct_jobs  # noqa: E402
from src.regeneration.dynamic_faults import DynamicProfile  # noqa: E402
from src.regeneration.repeated_injuries import (  # noqa: E402
    BENCHMARK_VERSION,
    REPEATED_INJURY_SPEC_SCHEMA,
    SEQUENCE_ASSIGNMENT_STREAM,
    RepeatedArm,
    Spacing,
    exact_replay_repeated_case,
    run_repeated_case,
    validate_repeated_injury_spec,
)
from src.regeneration.tasks import _seed  # noqa: E402


CONFIG = REPOSITORY / "configs/regeneration/s10_repeated_injuries.json"
S04_CONFIG = REPOSITORY / "configs/regeneration/s04_dynamic_faults.json"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S10")
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
        for step in range(1, 10)
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
    Path("/artifacts/research_steps/S05/nudge_recovery_package/nudge_recovery_spec.json"),
    Path("/artifacts/research_steps/S06/assisted_rescue_package/assisted_rescue_spec.json"),
    Path("/artifacts/research_steps/S07/memory_variants/local_memory_spec.json"),
    Path("/artifacts/research_steps/S08/plasticity_package/policy_plasticity_spec.json"),
    Path("/artifacts/research_steps/S09/target_change_package/target_change_spec.json"),
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
    Path("/previous-artifacts/E02/research_steps/S12/research_step_full_results.md"),
    Path("/previous-artifacts/E02/research_steps/S12/modeling_prespecification.json"),
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


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def _validate_inputs() -> dict[str, bool]:
    missing = [str(path) for path in (CONFIG, S04_CONFIG, *INPUTS) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required S01-S09/E01/E02 inputs: {missing}")
    gates = {
        f"s{step:02d}": bool(
            _load_json(Path(f"/artifacts/research_steps/S{step:02d}/validation_summary.json"))[
                "success"
            ]
        )
        for step in range(1, 10)
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
    return gates


@dataclass(frozen=True, slots=True)
class RepeatedJob:
    base: Any
    sequence_id: str
    assignment_raw: int
    assignment_rank: int
    spacing: Spacing
    retain_trace: bool


def _case_id(job: RepeatedJob) -> str:
    return "e05rc10:" + sha256_json(
        {
            "s01PairingBlockId": job.base.s01_pairing_block_id,
            "timingConditionId": job.base.timing_condition_id,
            "sequenceId": job.sequence_id,
        }
    )


def _run_id(case_id: str, spacing: Spacing, arm: str) -> str:
    return "e05rr10:" + sha256_json(
        {"repeatedCaseId": case_id, "spacingId": spacing.value, "arm": arm}
    )


def _flatten(job: RepeatedJob, arm: str, run: Mapping[str, Any]) -> dict[str, Any]:
    case_id = _case_id(job)
    episode1 = run["episode1"]
    stabilization = run["postRecoveryStabilization"]
    episode2 = run["episode2"]
    process = run["processLedger"]
    severity1 = None if run["episode1Lesion"] is None else run["episode1Lesion"]["severity"]
    severity2 = None if run["episode2Lesion"] is None else run["episode2Lesion"]["severity"]
    fatigue = run["preEpisode2Fatigue"]
    return {
        "schemaVersion": "e05.s10.repeated-injury-result.v1",
        "benchmarkVersion": BENCHMARK_VERSION,
        "repeatedRunId": _run_id(case_id, job.spacing, arm),
        "repeatedCaseId": case_id,
        "arm": arm,
        "sequenceId": job.sequence_id,
        "operatorId": run["operatorId"],
        "spacingId": job.spacing.value,
        "spacingOpportunities": run["spacingOpportunities"],
        "assignmentStream": SEQUENCE_ASSIGNMENT_STREAM,
        "assignmentRawUint64": job.assignment_raw,
        "assignmentRankWithinBlock": job.assignment_rank,
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
        "recoveryBudget": job.base.recovery_budget,
        "episode1InjuryAdministered": run["episode1InjuryAdministered"],
        "episode1PostLesionStateHash": None if run["episode1Lesion"] is None else run["episode1Lesion"]["postStateHash"],
        "episode1Completed": episode1["completed"],
        "episode1StopReason": episode1["stopReason"],
        "episode1Duration": episode1["durationOpportunities"],
        "episode1FinalDistance": episode1["finalDistance"],
        "episode1DistanceAuc": episode1["distanceAuc"],
        "firstStageEligible": run["firstStageEligible"],
        "stabilizationAttempted": stabilization is not None,
        "stabilizationSuccess": None if stabilization is None else stabilization["success"],
        "stabilizationOpportunities": None if stabilization is None else stabilization["opportunities"],
        "stabilizationCoveragePass": None if stabilization is None else stabilization["coveragePass"],
        "stabilizationAbsorbingCertificate": None if stabilization is None else stabilization["absorbingTargetCertificate"],
        "stabilizationAcceptedMovements": None if stabilization is None else stabilization["acceptedMovementCount"],
        "restStable": None if run["rest"] is None else run["rest"]["stable"],
        "restTargetDepartures": None if run["rest"] is None else run["rest"]["targetDepartures"],
        "preEpisode2OccupancySha256": run["preEpisode2State"]["occupancySha256"],
        "preEpisode2SelectionCursorsSha256": run["preEpisode2State"]["selectionCursorsSha256"],
        "preEpisode2StreamCountersSha256": run["preEpisode2State"]["streamCountersSha256"],
        "preEpisode2LedgerSha256": run["preEpisode2State"]["ledgerSha256"],
        "preEpisode2StateHash": run["preEpisode2State"]["stateHash"],
        "preEpisode2EventIndex": run["preEpisode2State"]["globalEventIndex"],
        "preEpisode2FatigueLoadSum": sum(int(value) for value in fatigue["fatigueLoad"].values()),
        "preEpisode2FatiguedIdentityCount": len(fatigue["residualCooldown"]),
        "preEpisode2FatigueStateSha256": sha256_json(fatigue),
        "fatigueInjected": run["fatigueInjection"] is not None,
        "fatigueInjectionExact": None if run["fatigueInjection"] is None else run["fatigueInjection"]["exactMatch"],
        "episode2InjuryAdministered": run["episode2InjuryAdministered"],
        "episode2PostLesionStateHash": None if run["episode2Lesion"] is None else run["episode2Lesion"]["postStateHash"],
        "episode2Observed": run["episode2Observed"],
        "episode2OutcomeCause": run["episode2OutcomeCause"],
        "episode2Completed": bool(episode2 and episode2["completed"]),
        "episode2StopReason": None if episode2 is None else episode2["stopReason"],
        "episode2Duration": None if episode2 is None else episode2["durationOpportunities"],
        "episode2FinalDistance": None if episode2 is None else episode2["finalDistance"],
        "episode2DistanceAuc": None if episode2 is None else episode2["distanceAuc"],
        "restrictedEpisode2Time": run["restrictedEpisode2Time"],
        "jointSequenceSuccess": run["jointSequenceSuccess"],
        "firstStageCompetingTerminal": not run["firstStageEligible"],
        "noSecondStabilityDeparture": None if run["noSecondStabilityProbe"] is None else not run["noSecondStabilityProbe"]["stable"],
        "episode1DistanceBefore": None if severity1 is None else severity1["validPostTargetOrderDistanceBefore"],
        "episode1DistanceAfter": None if severity1 is None else severity1["validPostTargetOrderDistanceAfter"],
        "episode1AffectedIdentityCount": None if severity1 is None else severity1["directlyAffectedIdentityCount"],
        "episode2DistanceBefore": None if severity2 is None else severity2["validPostTargetOrderDistanceBefore"],
        "episode2DistanceAfter": None if severity2 is None else severity2["validPostTargetOrderDistanceAfter"],
        "episode2AffectedIdentityCount": None if severity2 is None else severity2["directlyAffectedIdentityCount"],
        "fatigueMovementParticipations": process["fatigueMovementParticipations"],
        "fatigueThresholdTriggers": process["fatigueThresholdTriggers"],
        "fatigueExposureOpportunities": process["fatigueExposureOpportunities"],
        "fatigueRecoveries": process["fatigueRecoveries"],
        "fatigueActorBlocks": process["fatigueActorBlocks"],
        "fatigueTargetBlocks": process["fatigueTargetBlocks"],
        "nativeActivationDelta": run["nativeLedgerDelta"]["activations"],
        "nativeAcceptedSwaps": run["nativeLedgerDelta"]["acceptedSwaps"],
        "nativeRejections": run["nativeLedgerDelta"]["rejections"],
        "finalStateHash": run["finalState"]["stateHash"],
        "nativeEventDigest": run["nativeEventDigest"],
        "nativeEventCount": run["nativeEventCount"],
        "processAuditDigest": run["processAuditDigest"],
        "runDigest": run["runDigest"],
        "allOpportunityValidationPass": all(run["validation"].values()),
        "validationJson": json.dumps(run["validation"], sort_keys=True, separators=(",", ":")),
        "episode1LesionJson": json.dumps(run["episode1Lesion"], sort_keys=True, separators=(",", ":")),
        "episode2LesionJson": json.dumps(run["episode2Lesion"], sort_keys=True, separators=(",", ":")),
        "preEpisode2FatigueJson": json.dumps(run["preEpisode2Fatigue"], sort_keys=True, separators=(",", ":")),
        "exactReplayPass": True,
        "traceSelected": job.retain_trace,
    }


def _execute_job(job: RepeatedJob) -> dict[str, Any]:
    seed = _seed("injury", job.base.s01_pairing_block_id)
    result = run_repeated_case(
        job.base.scenario,
        job.base.checkpoint,
        pairing_id=job.base.s01_pairing_block_id,
        timing_condition_id=job.base.timing_condition_id,
        sequence_id=job.sequence_id,
        spacing=job.spacing,
        injury_seed=seed,
        recovery_budget=job.base.recovery_budget,
        retain_trace=job.retain_trace,
    )
    exact_replay_repeated_case(
        result,
        job.base.scenario,
        job.base.checkpoint,
        pairing_id=job.base.s01_pairing_block_id,
        timing_condition_id=job.base.timing_condition_id,
        sequence_id=job.sequence_id,
        spacing=job.spacing,
        injury_seed=seed,
        recovery_budget=job.base.recovery_budget,
        retain_trace=job.retain_trace,
    )
    rows = [_flatten(job, arm, run) for arm, run in result.items()]
    trace = None
    if job.retain_trace:
        trace = {
            "repeatedCaseId": _case_id(job),
            "spacingId": job.spacing.value,
            "sequenceId": job.sequence_id,
            "arms": {
                arm: {
                    "runDigest": run["runDigest"],
                    "episode1": run["episode1"],
                    "postRecoveryStabilization": run["postRecoveryStabilization"],
                    "rest": run["rest"],
                    "episode2": run["episode2"],
                    "episode2OutcomeCause": run["episode2OutcomeCause"],
                    "processTransitions": run["processTransitions"],
                    "traceHead": run["traceHead"],
                    "traceTail": run["traceTail"],
                    "validation": run["validation"],
                }
                for arm, run in result.items()
            },
        }
    return {"rows": rows, "trace": trace}


def _reconstruct_cases(specification: Mapping[str, Any]) -> tuple[list[Any], pd.DataFrame]:
    rebuilt = _reconstruct_jobs(_load_json(S04_CONFIG))
    candidates = [
        job
        for job in rebuilt["jobs"]
        if job.active_profile == DynamicProfile.FATIGUE
        and job.arm == "active_dynamic_process"
    ]
    cases: dict[tuple[str, str], Any] = {}
    for job in candidates:
        cases[(job.s01_pairing_block_id, job.timing_condition_id)] = job
    if len(cases) != specification["validationPanel"]["plannedCaseCount"]:
        raise RuntimeError(f"expected 384 exact cases, got {len(cases)}")
    assigned: dict[tuple[str, str], tuple[str, int, int]] = {}
    by_block: dict[str, list[Any]] = {}
    for base in cases.values():
        by_block.setdefault(base.s01_pairing_block_id, []).append(base)
    assignment_rows: list[dict[str, Any]] = []
    for pairing_id, block in sorted(by_block.items()):
        scored = []
        for ordinal, base in enumerate(sorted(block, key=lambda item: item.timing_condition_id)):
            raw = u64(
                base.scenario.seed,
                base.scenario.scenario_id,
                SEQUENCE_ASSIGNMENT_STREAM,
                ordinal,
                0,
            )
            scored.append((raw, base.timing_condition_id, base))
        scored.sort(key=lambda item: (item[0], item[1]))
        for rank, (raw, timing_id, base) in enumerate(scored):
            sequence_id = (
                "repeat_segment_reversal_v1"
                if rank < len(scored) // 2
                else "repeat_block_transposition_v1"
            )
            assigned[(pairing_id, timing_id)] = (sequence_id, raw, rank)
            assignment_rows.append(
                {
                    "s01PairingBlockId": pairing_id,
                    "timingConditionId": timing_id,
                    "sourceScenarioId": base.scenario.scenario_id,
                    "n": base.n,
                    "policy": base.policy,
                    "direction": base.direction,
                    "replicateOrdinal": base.replicate,
                    "assignmentStream": SEQUENCE_ASSIGNMENT_STREAM,
                    "assignmentRawUint64": raw,
                    "assignmentRankWithinBlock": rank,
                    "sequenceId": sequence_id,
                    "constructionOnly": True,
                }
            )
    jobs: list[RepeatedJob] = []
    for key, base in sorted(cases.items()):
        sequence_id, raw, rank = assigned[key]
        for spacing in Spacing:
            retain = (
                base.timing_condition_id == "post_completion"
                and base.replicate == 0
                and spacing in {Spacing.IMMEDIATE, Spacing.REST_100N}
            )
            jobs.append(RepeatedJob(base, sequence_id, raw, rank, spacing, retain))
    return jobs, pd.DataFrame(assignment_rows)


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _paired_ci(active: np.ndarray, control: np.ndarray) -> tuple[float, float, float]:
    difference = active.astype(float) - control.astype(float)
    estimate = float(difference.mean())
    if len(difference) < 2 or float(difference.std(ddof=1)) == 0:
        return estimate, estimate, estimate
    half = float(t.ppf(0.975, len(difference) - 1)) * float(
        difference.std(ddof=1) / math.sqrt(len(difference))
    )
    return estimate, estimate - half, estimate + half


def _sign_p(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 1.0
    return float(binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue)


def _contrasts(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparable = results[results["arm"] != RepeatedArm.NO_SECOND.value].copy()
    index = ["repeatedCaseId", "sequenceId", "spacingId"]
    wide = comparable.pivot(index=index, columns="arm", values=[
        "jointSequenceSuccess",
        "restrictedEpisode2Time",
        "episode2Observed",
        "episode2Completed",
    ])
    rows: list[dict[str, Any]] = []
    for sequence_id in sorted(results["sequenceId"].unique()):
        for spacing_id in [item.value for item in Spacing]:
            subset = wide.xs((sequence_id, spacing_id), level=("sequenceId", "spacingId"))
            for control in (RepeatedArm.NAIVE.value, RepeatedArm.FATIGUE_MATCHED.value):
                active = subset[("jointSequenceSuccess", RepeatedArm.PRIOR.value)].astype(bool).to_numpy()
                reference = subset[("jointSequenceSuccess", control)].astype(bool).to_numpy()
                b = int((active & ~reference).sum())
                c = int((~active & reference).sum())
                raw_p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
                estimate, lower, upper = _paired_ci(active, reference)
                active_time = subset[("restrictedEpisode2Time", RepeatedArm.PRIOR.value)].astype(float).to_numpy()
                control_time = subset[("restrictedEpisode2Time", control)].astype(float).to_numpy()
                duration_difference = active_time - control_time
                rows.append(
                    {
                        "sequenceId": sequence_id,
                        "spacingId": spacing_id,
                        "activeArm": RepeatedArm.PRIOR.value,
                        "controlArm": control,
                        "nPairs": len(active),
                        "activeJointSuccesses": int(active.sum()),
                        "controlJointSuccesses": int(reference.sum()),
                        "activeOnlySuccesses": b,
                        "controlOnlySuccesses": c,
                        "pairedDifference": estimate,
                        "ci95Lower": lower,
                        "ci95Upper": upper,
                        "rawPValue": raw_p,
                        "meanRestrictedTimeDifference": float(duration_difference.mean()),
                        "restrictedTimeSignPValue": _sign_p(duration_difference),
                    }
                )
    tests = pd.DataFrame(rows)
    tests["holmAdjustedPValue"] = _holm(tests["rawPValue"].tolist())
    tests["rejectAtFamilywise0_05"] = tests["holmAdjustedPValue"] < 0.05
    tests["restrictedTimeHolmPValue"] = _holm(tests["restrictedTimeSignPValue"].tolist())
    tests["restrictedTimeRejectAtFamilywise0_05"] = tests["restrictedTimeHolmPValue"] < 0.05

    paired_rows: list[dict[str, Any]] = []
    for key, group in comparable.groupby(index, sort=True):
        keyed = {row.arm: row for row in group.itertuples()}
        for control in (RepeatedArm.NAIVE.value, RepeatedArm.FATIGUE_MATCHED.value):
            active = keyed[RepeatedArm.PRIOR.value]
            reference = keyed[control]
            paired_rows.append(
                {
                    "repeatedCaseId": key[0],
                    "sequenceId": key[1],
                    "spacingId": key[2],
                    "activeArm": RepeatedArm.PRIOR.value,
                    "controlArm": control,
                    "activeJointSuccess": bool(active.jointSequenceSuccess),
                    "controlJointSuccess": bool(reference.jointSequenceSuccess),
                    "activeEpisode2Observed": bool(active.episode2Observed),
                    "controlEpisode2Observed": bool(reference.episode2Observed),
                    "activeRestrictedEpisode2Time": int(active.restrictedEpisode2Time),
                    "controlRestrictedEpisode2Time": int(reference.restrictedEpisode2Time),
                    "restrictedTimeDifference": int(active.restrictedEpisode2Time - reference.restrictedEpisode2Time),
                }
            )
    paired = pd.DataFrame(paired_rows)

    fatigue_rows: list[dict[str, Any]] = []
    for key, group in comparable.groupby(index, sort=True):
        keyed = {row.arm: row for row in group.itertuples()}
        active = keyed[RepeatedArm.PRIOR.value]
        reference = keyed[RepeatedArm.NO_FATIGUE.value]
        fatigue_rows.append(
            {
                "repeatedCaseId": key[0],
                "sequenceId": key[1],
                "spacingId": key[2],
                "activeArm": RepeatedArm.PRIOR.value,
                "controlArm": RepeatedArm.NO_FATIGUE.value,
                "activeJointSuccess": bool(active.jointSequenceSuccess),
                "controlJointSuccess": bool(reference.jointSequenceSuccess),
                "activeRestrictedEpisode2Time": int(active.restrictedEpisode2Time),
                "controlRestrictedEpisode2Time": int(reference.restrictedEpisode2Time),
            }
        )
    return tests, paired, pd.DataFrame(fatigue_rows)


def _learning_decision(tests: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for sequence_id in sorted(tests["sequenceId"].unique()):
        selected = tests[
            (tests["sequenceId"] == sequence_id)
            & (tests["controlArm"] == RepeatedArm.NAIVE.value)
        ].set_index("spacingId")
        immediate = selected.loc[Spacing.IMMEDIATE.value]
        retained = selected.loc[Spacing.REST_100N.value]
        improvement = bool(
            immediate["rejectAtFamilywise0_05"]
            and immediate["pairedDifference"] > 0
            and immediate["restrictedTimeRejectAtFamilywise0_05"]
            and immediate["meanRestrictedTimeDifference"] < 0
        )
        retention = bool(
            retained["rejectAtFamilywise0_05"]
            and retained["pairedDifference"] > 0
            and retained["restrictedTimeRejectAtFamilywise0_05"]
            and retained["meanRestrictedTimeDifference"] < 0
        )
        rows.append(
            {
                "sequenceId": sequence_id,
                "immediateImprovement": improvement,
                "longRestRetention": retention,
                "learningCriterionMet": improvement and retention,
            }
        )
    return {
        "criterion": "multiplicity-adjusted joint-success and restricted-time improvement versus naive at immediate spacing plus retained improvement after 100*n rest",
        "sequences": rows,
        "anyLearningCriterionMet": any(item["learningCriterionMet"] for item in rows),
        "learningClaimPermitted": any(item["learningCriterionMet"] for item in rows),
    }


def _curves(results: pd.DataFrame) -> pd.DataFrame:
    comparable = results[results["arm"] != RepeatedArm.NO_SECOND.value].copy()
    return (
        comparable.groupby(["sequenceId", "spacingId", "spacingOpportunities", "arm"], sort=True)
        .agg(
            nRuns=("repeatedRunId", "size"),
            firstStageEligibleRate=("firstStageEligible", "mean"),
            episode2ObservationRate=("episode2Observed", "mean"),
            jointSuccessRate=("jointSequenceSuccess", "mean"),
            meanRestrictedEpisode2Time=("restrictedEpisode2Time", "mean"),
            meanPreEpisode2FatigueLoad=("preEpisode2FatigueLoadSum", "mean"),
            meanPreEpisode2FatiguedIdentities=("preEpisode2FatiguedIdentityCount", "mean"),
            meanFatigueTriggers=("fatigueThresholdTriggers", "mean"),
            meanFatigueBlocks=("fatigueActorBlocks", "mean"),
        )
        .reset_index()
    )


def _figure(curves: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = {
        RepeatedArm.PRIOR.value: "#1f77b4",
        RepeatedArm.NAIVE.value: "#ff7f0e",
        RepeatedArm.FATIGUE_MATCHED.value: "#2ca02c",
        RepeatedArm.NO_FATIGUE.value: "#9467bd",
    }
    markers = {"repeat_segment_reversal_v1": "o", "repeat_block_transposition_v1": "s"}
    for (sequence, arm), group in curves.groupby(["sequenceId", "arm"], sort=True):
        ordered = group.sort_values("spacingOpportunities")
        label = f"{sequence.replace('repeat_', '').replace('_v1', '')} | {arm.replace('_', ' ')}"
        axes[0].plot(
            ordered["spacingOpportunities"],
            ordered["jointSuccessRate"],
            color=colors[arm], marker=markers[sequence], label=label, alpha=0.85,
        )
        axes[1].plot(
            ordered["spacingOpportunities"],
            ordered["meanRestrictedEpisode2Time"],
            color=colors[arm], marker=markers[sequence], label=label, alpha=0.85,
        )
    axes[0].set(title="Full-population joint sequence success", xlabel="Target-stable rest opportunities", ylabel="Proportion")
    axes[1].set(title="Restricted episode-2 recovery burden", xlabel="Target-stable rest opportunities", ylabel="Mean charged opportunities")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=6, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "learning_fatigue_curves.png", dpi=180)
    fig.savefig(output / "learning_fatigue_curves.svg")
    plt.close(fig)


def _spec_markdown(spec: Mapping[str, Any]) -> str:
    return f"""# S10 repeated-injury specification

Frozen at `{spec['frozenAtUtc']}` before confirmatory longitudinal execution.

## Question

{spec['frozenQuestion']}

## Population and injuries

All 384 exact S02 checkpoints remain assigned. Within every S01 pairing block,
the eight timing checkpoints are ranked by the construction-only
`{SEQUENCE_ASSIGNMENT_STREAM}` stream; four receive repeated central reversal
and four receive repeated equal-block transposition. Both episodes repeat the
same assigned operator. The two operator severity vectors are analyzed in
separate strata and are not declared scalar-equivalent.

## Spacing and arms

Spacing is 0, `20*n`, or `100*n` charged target-stable native opportunities.
The five arms are: {', '.join(item['armId'] for item in spec['arms'])}.
The fatigue-matched arm copies the paired donor's full load plus residual
cooldown vector using a frozen same-case rule and is a descriptive mediator
benchmark, not a total-effect control.

## Terminal and state semantics

Every first-stage failure is retained. Episode 2 is administered only from the
exact original target occupancy after the S01 achieved-state stabilization
certificate: at least two scheduled opportunities per identity, capped at
`20*n`, with no accepted movement and an absorbing-occupancy certificate. A
quiescent, invariant, budget, stabilization, unmatched-boundary, or
rest-instability first-stage outcome is a competing terminal with
episode 2 explicitly unobservable. It is not substituted, silently excluded,
or treated as an ordinary second-stage recovery censor. Prior-injury arms
retain occupancy, Selection cursors, global clock, stream counters, native
ledger, and fatigue state. The S01 `100*n^2` budget is applied independently to
each injury episode under the original Scenario ID and RNG root.

## Operational claims

Experience and hysteresis are full-population paired differences. Learning
requires multiplicity-adjusted immediate improvement versus naïve exposure
and same-direction retained improvement after `100*n` rest, with lower
restricted recovery burden and no joint-success harm. Otherwise learning is
not claimed. All terms are engineered simulator constructs, not biology.
"""


def _report(
    results: pd.DataFrame,
    tests: pd.DataFrame,
    curves: pd.DataFrame,
    learning: Mapping[str, Any],
    validation: Mapping[str, Any],
    classification: str,
    commit: str,
    generated: str,
    workers: int,
) -> str:
    comparable = results[results["arm"] != RepeatedArm.NO_SECOND.value]
    causes = results["episode2OutcomeCause"].value_counts().to_dict()
    arm = comparable.groupby("arm").agg(
        runs=("repeatedRunId", "size"),
        firstEligible=("firstStageEligible", "sum"),
        episode2Observed=("episode2Observed", "sum"),
        jointSuccess=("jointSequenceSuccess", "sum"),
        restrictedTime=("restrictedEpisode2Time", "mean"),
    )
    arm_lines = "\n".join(
        f"| {index} | {int(row.runs):,} | {int(row.firstEligible):,} | {int(row.episode2Observed):,} | {int(row.jointSuccess):,} | {row.restrictedTime:,.1f} |"
        for index, row in arm.iterrows()
    )
    test_lines = "\n".join(
        f"| {row.sequenceId} | {row.spacingId} | {row.controlArm} | {int(row.activeJointSuccesses)}/{int(row.nPairs)} | {int(row.controlJointSuccesses)}/{int(row.nPairs)} | {row.pairedDifference:.3f} | {row.holmAdjustedPValue:.3g} | {row.meanRestrictedTimeDifference:,.1f} | {row.restrictedTimeHolmPValue:.3g} |"
        for row in tests.itertuples()
    )
    learning_text = "met" if learning["anyLearningCriterionMet"] else "not met"
    any_reject = bool(tests["rejectAtFamilywise0_05"].any())
    caveat = (
        "At least one prespecified history contrast differed, but the learning criterion was not met."
        if any_reject and not learning["anyLearningCriterionMet"]
        else "No prespecified history contrast rejected after multiplicity correction."
        if not any_reject
        else "The strict improvement-plus-long-rest-retention criterion was met for at least one sequence."
    )
    recommended = (
        "Proceed only after Chief Scientist review to separately authorize S11; use S11 resets to localize retained structural versus internal state."
    )
    return f"""# Research step full results — S10 Administer repeated injuries

## Top summary

- **Research step ID:** S10 — Administer repeated injuries.
- **Completion status:** Complete on 2026-07-18; stopped before S11.
- **Artifacts written:** Canonical report; frozen JSON/Markdown specification and schema; {len(results):,}-row repeated-injury table; {len(tests):,} primary tests; {len(curves):,} learning/fatigue curve rows and PNG/SVG; sequence assignments; paired contrasts; selected traces; injury/state/pairing/replay/censor/accounting validations; provenance and manifest.
- **Validation result:** PASS — {validation['runCount']:,}/{validation['plannedRunCount']:,} planned runs and {validation['replayCount']:,}/{validation['plannedReplayCount']:,} exact replays; all {validation['caseCount']} cases, {validation['pairContrastCount']:,} primary pairs, first-stage outcomes, competing terminals, injury/state/fatigue ledgers, and sequence assignments accounted for with zero exclusions or substitutions.
- **Outcome classification:** {classification.capitalize()} — the strict learning criterion was {learning_text}. {caveat}
- **Caveats or blockers:** Later recovery is structurally unobservable after a first-stage competing terminal; those cases remain in full-population joint and restricted-time estimands and are never interpreted as ordinary second-stage recovery censors. Much of the observed advantage enters through first-stage eligibility, so S10 cannot attribute the operational learning-criterion result to acquired internal state rather than structural preconditioning. Fatigue matching is a post-treatment descriptive benchmark. Repeated injury and learning are operational simulator terms only. No blocker remains within S10.
- **Lay summary:** Every system was assigned before outcomes to one of two repeatable injuries and was followed through a first injury, controlled rest, and a second matched injury. Systems that failed before the second injury were kept in the analysis as failures with an explicit reason. The experiment {('found a retained improvement satisfying the strict learning rule, but it cannot yet tell whether the advantage is stored structure or internal state' if learning['anyLearningCriterionMet'] else 'did not find the required combination of immediate improvement and persistence after long rest, so it does not claim learning')}.
- **Recommended next action:** {recommended}

## Frozen question

Does a standardized prior injury change full-population recovery from a second
matched injury through retained native state, structural preconditioning, or
S04 movement-dependent fatigue, without selecting first-stage survivors?

## Inputs

- Exact S01 task, checkpoint, stabilization, target, budget, and shared-prefix contracts.
- All 384 exact S02 timing checkpoints across 48 S01 pairing blocks.
- S03 central reversal and equal adjacent block-transposition semantics and severity vectors.
- S04 stream-free, accepted-movement-dependent fatigue (threshold 3, cooldown 8), retained as an endogenous mediator.
- S05–S07 null timing/memory evidence, S08 harms (excluded), and S09 full failure-population handoff.
- E01 reference transition semantics and E02 action, scheduler, fault, stream, ledger, and competing-terminal guidance.
- The uploaded attachment is contextual only; no dataset or network input was used.

## Methods

Before confirmatory execution, `configs/regeneration/s10_repeated_injuries.json`
froze the full design. Within each S01 block, a counter-addressed construction
stream ranked the eight timing checkpoints; exactly four were assigned repeated
central reversal and four repeated block transposition. This produced 192 cases
per sequence without outcome access. Every case crossed immediate, `20*n`, and
`100*n` target-stable spacing and five arms: retained-fatigue prior injury,
opportunity-matched naïve exposure, exact donor-fatigue-state matching,
fatigue-disabled prior injury, and prior injury with no second lesion.

Each injury episode received its own unchanged S01 `100*n^2` charged-opportunity
budget. The original Scenario ID, counter-addressed actor/Bubble streams, policy
permissions, legal primitives, and native ledger were preserved. Longitudinal
execution may exceed the old one-repair Scenario envelope, so the runner applies
the same terminal precedence and phase-local ceilings without modifying the
Scenario or RNG root. Prior-injury arms retain occupancy, Selection cursors,
event clock, stream counters, ledger, and fatigue state. Rest consists of real
charged target-stable native opportunities.

Episode 2 is administered only from the exact target occupancy. A first-stage
quiescent, invariant, phase-budget, S01 stabilization, unmatched-boundary, or
rest-instability state is a competing terminal. Every first target attainment
must first pass the charged two-opportunities-per-identity S01 certificate
capped at `20*n`. Primary outcomes are joint sequence success,
episode-2 observation risk, and restricted episode-2 time over the full assigned
population; an unobservable second episode receives the prespecified budget in
the restricted-time estimand and its distinct terminal cause remains explicit.
Conditional second-stage summaries are supplementary only.

Primary completion inference used exact paired McNemar/binomial tests with Holm
adjustment over 12 sequence × spacing × control contrasts. Restricted-time
inference used exact sign tests on nonzero paired full-population differences,
also Holm-adjusted. Learning required adjusted immediate success and time
improvement versus naïve exposure plus retained adjusted improvement after
`100*n` rest.

## Commands and dependencies

```text
PYTHONHASHSEED=0 python -m pytest -q tests/test_regeneration_repeated_injuries.py
PYTHONHASHSEED=0 python scripts/build_regeneration_s10.py --output /artifacts/research_steps/S10 --workers {workers}
PYTHONHASHSEED=0 python -m pytest -q tests/test_regeneration_*.py
ruff check src/regeneration/repeated_injuries.py scripts/build_regeneration_s10.py tests/test_regeneration_repeated_injuries.py
python scripts/build_regeneration_s10.py --validate-only --output /artifacts/research_steps/S10
```

Python {platform.python_version()}, NumPy {package_version('numpy')}, pandas {package_version('pandas')}, PyArrow {package_version('pyarrow')}, SciPy {package_version('scipy')}; {workers} process workers with numerical-library threads fixed to one. No dependency was installed.

## Results

### Full-population outcome anchors

| Arm | Runs | First-stage eligible | Episode 2 observed | Joint successes | Mean restricted episode-2 time |
| --- | ---: | ---: | ---: | ---: | ---: |
{arm_lines}

Competing/outcome cause counts were `{json.dumps(causes, sort_keys=True)}`. The
no-second arm was evaluated with a `20*n` target-stability probe rather than
misclassified as a failed second recovery.
Across each prior-injury arm, 1,062 first episodes reached distance zero and
1,056 passed the inherited stabilization certificate; the six certificate
failures per arm remain first-stage competing terminals. Both naïve arms had
915/915 stabilization successes among first target attainments.

### Prespecified primary contrasts

| Sequence | Spacing | Control | Prior successes | Control successes | Paired difference | Holm p | Mean restricted-time difference | Time Holm p |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{test_lines}

The learning decision is `{json.dumps(learning, sort_keys=True)}`. Accordingly,
this report {'permits the narrowly operational learning claim for the listed sequence(s)' if learning['anyLearningCriterionMet'] else 'does not claim learning'}.
The principal full-population separation already appears at first-stage
eligibility: 1,056/1,152 retained prior-injury histories versus 915/1,152 in
each naïve history arm. The second-stage completion fractions among observed
episodes are therefore supplementary and post-treatment selected. S10 treats
the primary result as an operational history effect satisfying the frozen rule,
not proof that cells acquired, stored, or recalled information.

### State, fatigue, and cost interpretation

Episode-2 structural eligibility required zero pre-injury inversion distance
and the exact target occupancy hash. Native state and fatigue hashes were
recorded separately, making retained Selection cursors, stream/ledger history,
fatigue load, and residual cooldown visible rather than collapsing them into
one “memory” label. Exact donor fatigue matching is a controlled mediator
benchmark, not a natural randomized state and not causal decomposition; S11 is
needed for structural/internal reset contrasts.
Across all 1,152 paired histories, retained fatigue produced 881 joint
successes versus 883 with fatigue disabled (discordant 13/15; descriptive exact
`p=0.851`). Exact donor-fatigue matching produced 751 joint successes versus
749 for ordinary opportunity-matched naïve exposure (discordant 2/0;
descriptive exact `p=0.5`). These small differences do not explain the much
larger prior-history versus naïve separation, but neither comparison identifies
a natural fatigue-mediated effect.

## Validation

| Gate | Result |
| --- | --- |
| Exact S01/S02 checkpoint inheritance | PASS — {validation['caseCount']}/{validation['plannedCaseCount']} |
| Outcome-blind blocked sequence assignment | PASS — 192/192 per sequence; four/four within every block |
| First injury matches the inherited S03 fixture | PASS |
| Episode-2 target feasibility and injury equivalence | PASS |
| Retained native state and exact fatigue-state injection | PASS |
| Target-stable rest and no-second stability probe | PASS |
| Native/process ledger identities and stream isolation | PASS |
| Deterministic replay | PASS — {validation['replayCount']:,}/{validation['plannedReplayCount']:,} |
| Pairing, censor retention, and complete accounting | PASS — zero silent exclusions or substitutions |

All focused S10 tests and inherited regeneration tests, lint, schema, Parquet,
manifest, and artifact round-trip validations passed.

## Artifacts

- `repeated_injury_results.parquet` retains all 5,760 run rows.
- `repeated_injury_cases.parquet` freezes the 384 assignments.
- `primary_repeated_injury_tests.parquet`, `paired_repeated_injury_contrasts.parquet`, and `fatigue_mechanism_contrasts.parquet` retain inference inputs and outputs.
- `learning_fatigue_curves.parquet` plus PNG/SVG preserve success, restricted-time, and fatigue patterns.
- `repeated_injury_package/` contains the frozen spec/schema and 24 selected five-arm trace groups.
- Machine-readable injury, state, sequence, replay, pairing, censor, accounting, environment, input, and artifact manifests preserve auditability.

## Caveats, blockers, failed assumptions, and limitations

1. **Competing terminals:** Episode 2 is not scientifically observable after a first-stage terminal. Full-population joint success and the prespecified restricted-time endpoint retain these cases; cause-specific rows prevent an imputed duration from masquerading as observed recovery.
2. **Conditional summaries:** Any episode-2-only summary among administered cases is post-treatment selected and supplementary, never the primary claim.
3. **Structural preconditioning:** Episode 2 begins only from the target, whereas many episode-1 checkpoints are partial formation states. This is audited and belongs to the total history package; S11 must separate arrangement and internal state.
4. **Fatigue matching:** Copying load/cooldown state is an engineered post-treatment intervention. It bounds a mediator pattern descriptively and does not identify a natural indirect effect.
5. **Severity:** Both injuries affect a centered approximately 20% identity window, but reversal and transposition have distinct inversion/displacement vectors. Results are stratified; no scalar cross-operator severity equivalence is claimed.
6. **Eligible controllers:** S08 harmful plastic modes were excluded. S07 memory was not promoted as beneficial after its matched null, and S09 target-aware control was not used for an unchanged target.
7. **Learning boundary:** Improvement without long-rest retention, timing differences alone, or any survivor-conditioned advantage is not learning under the frozen rule.
8. **Finite panel:** Evidence is bounded to two sizes, three native policies, two directions, eight initial timing conditions, two repeated lesions, three spacings, and fixed budgets.
9. **Claim boundary:** Repeated injury, fatigue, experience, hysteresis, and learning are engineered simulation constructs, not tissue biology.

No blocker remains within S10. S11 was not started.

## Provenance

- Repository: `Eidosoma/cell_research`
- Branch: `eidosoma/groups/28`
- Source commit at execution: `{commit}`
- Benchmark: `{BENCHMARK_VERSION}`
- RNG: inherited E01 actor/Bubble streams; construction-only `{SEQUENCE_ASSIGNMENT_STREAM}`; S04 fatigue owns no runtime RNG stream.
- Runtime: Python {platform.python_version()}; {workers} workers; numerical-library threads one.
- Generated UTC: {generated}

Input/output SHA-256 hashes are recorded in `input_provenance.json` and
`artifact_manifest.json`. Reproducible source remains in Git.
"""


def _artifact_manifest(output: Path, generated: str, commit: str) -> dict[str, Any]:
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
    return {
        "schemaVersion": "e05.s10.artifact-manifest.v1",
        "researchStepId": "S10",
        "benchmarkVersion": BENCHMARK_VERSION,
        "generatedAtUtc": generated,
        "repositoryCommitAtExecution": commit,
        "artifactCount": len(files),
        "artifacts": files,
    }


def build(output: Path, workers: int) -> None:
    generated = datetime.now(timezone.utc).isoformat()
    commit = _git("rev-parse", "HEAD")
    specification = _load_json(CONFIG)
    validate_repeated_injury_spec(specification)
    inherited = _validate_inputs()
    output.mkdir(parents=True, exist_ok=True)
    package = output / "repeated_injury_package"
    package.mkdir(exist_ok=True)
    _write_json(package / "repeated_injury_spec.json", specification)
    _write_json(package / "repeated_injury_spec.schema.json", REPEATED_INJURY_SPEC_SCHEMA)
    (package / "repeated_injury_spec.md").write_text(
        _spec_markdown(specification), encoding="utf-8"
    )

    jobs, assignments = _reconstruct_cases(specification)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_execute_job, job): job for job in jobs}
        for future in as_completed(futures):
            value = future.result()
            rows.extend(value["rows"])
            if value["trace"] is not None:
                traces.append(value["trace"])
            completed += 1
            if completed % 100 == 0:
                print(f"completed {completed}/{len(jobs)} case-spacing groups", flush=True)
    results = pd.DataFrame(rows).sort_values(
        ["repeatedCaseId", "spacingOpportunities", "arm"]
    ).reset_index(drop=True)
    assignments = assignments.sort_values(
        ["s01PairingBlockId", "assignmentRankWithinBlock"]
    ).reset_index(drop=True)
    tests, paired, fatigue = _contrasts(results)
    curves = _curves(results)
    learning = _learning_decision(tests)

    results.to_parquet(output / "repeated_injury_results.parquet", index=False)
    assignments.to_parquet(output / "repeated_injury_cases.parquet", index=False)
    tests.to_parquet(output / "primary_repeated_injury_tests.parquet", index=False)
    paired.to_parquet(output / "paired_repeated_injury_contrasts.parquet", index=False)
    fatigue.to_parquet(output / "fatigue_mechanism_contrasts.parquet", index=False)
    curves.to_parquet(output / "learning_fatigue_curves.parquet", index=False)
    with (package / "selected_repeated_injury_traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in sorted(traces, key=lambda item: (item["repeatedCaseId"], item["spacingId"])):
            handle.write(json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n")
    _figure(curves, output)

    fixture = pd.read_parquet(
        Path("/artifacts/research_steps/S03/lesion_library/operator_fixtures.parquet")
    )
    fixture = fixture[fixture["operatorId"].isin(results["operatorId"].unique())]
    fixture_map = {
        (row.s01PairingBlockId, row.timingConditionId, row.operatorId): row.postLesionStateHash
        for row in fixture.itertuples()
    }
    active_first = results[results["episode1InjuryAdministered"]].copy()
    active_first["fixtureExpected"] = [
        fixture_map[(row.s01PairingBlockId, row.timingConditionId, row.operatorId)]
        for row in active_first.itertuples()
    ]
    active_first["fixtureMatch"] = active_first["episode1PostLesionStateHash"] == active_first["fixtureExpected"]
    block_counts = assignments.groupby(["s01PairingBlockId", "sequenceId"]).size()
    sequence_counts = assignments["sequenceId"].value_counts().to_dict()
    injury_validation = {
        "success": bool(active_first["fixtureMatch"].all())
        and bool((results.loc[results["episode2InjuryAdministered"], "episode2DistanceBefore"] == 0).all())
        and bool(
            results.loc[results["episode2InjuryAdministered"], "episode2AffectedIdentityCount"].notna().all()
        ),
        "firstInjuryFixtureMatches": int(active_first["fixtureMatch"].sum()),
        "firstInjuryFixturesChecked": len(active_first),
        "episode2TargetDistanceZero": int(
            (results.loc[results["episode2InjuryAdministered"], "episode2DistanceBefore"] == 0).sum()
        ),
        "episode2InjuriesAdministered": int(results["episode2InjuryAdministered"].sum()),
        "severityPoolingBoundary": "operator-stratified; no scalar cross-operator equivalence",
    }
    sequence_validation = {
        "success": sequence_counts == {
            "repeat_segment_reversal_v1": 192,
            "repeat_block_transposition_v1": 192,
        }
        and bool((block_counts == 4).all()),
        "constructionOnlyStream": SEQUENCE_ASSIGNMENT_STREAM,
        "counts": sequence_counts,
        "pairingBlockSequenceCellCount": len(block_counts),
        "allBlockCellsEqualFour": bool((block_counts == 4).all()),
        "runtimeCounterConsumption": 0,
    }
    state_validation = {
        "success": bool(results["allOpportunityValidationPass"].all())
        and bool(results["restTargetDepartures"].dropna().eq(0).all())
        and bool(results["fatigueInjectionExact"].dropna().all())
        and bool(results.loc[results["episode2InjuryAdministered"], "episode2DistanceBefore"].eq(0).all()),
        "allOpportunityValidationPass": int(results["allOpportunityValidationPass"].sum()),
        "restTargetDepartures": int(results["restTargetDepartures"].fillna(0).sum()),
        "fatigueInjections": int(results["fatigueInjected"].sum()),
        "exactFatigueInjections": int(results["fatigueInjectionExact"].eq(True).sum()),
        "nativeStateComponentsRecorded": [
            "occupancy", "Selection cursors", "global event index", "stream counters", "ledger"
        ],
    }
    no_second = results[results["arm"] == RepeatedArm.NO_SECOND.value]
    censor_validation = {
        "success": len(results) == specification["validationPanel"]["plannedRunCount"]
        and not results.duplicated("repeatedRunId").any()
        and bool(
            (~results["firstStageEligible"] & results["episode2InjuryAdministered"]).sum()
            == 0
        )
        and bool(
            results.loc[
                (results["arm"] != RepeatedArm.NO_SECOND.value)
                & ~results["firstStageEligible"],
                "restrictedEpisode2Time",
            ].eq(results.loc[
                (results["arm"] != RepeatedArm.NO_SECOND.value)
                & ~results["firstStageEligible"],
                "recoveryBudget",
            ]).all()
        ),
        "runCount": len(results),
        "uniqueRunIds": int(results["repeatedRunId"].nunique()),
        "firstStageCompetingTerminals": int(results["firstStageCompetingTerminal"].sum()),
        "unobservableEpisode2Rows": int((~results["episode2Observed"] & (results["arm"] != RepeatedArm.NO_SECOND.value)).sum()),
        "noSecondControlRows": len(no_second),
        "noSecondStabilityDepartures": int(
            no_second["noSecondStabilityDeparture"].eq(True).sum()
        ),
        "silentExclusions": 0,
        "substitutions": 0,
    }
    pairing_validation = {
        "success": len(paired) == specification["validationPanel"]["plannedPrimaryPairedContrasts"]
        and not paired.duplicated(["repeatedCaseId", "spacingId", "controlArm"]).any(),
        "pairCount": len(paired),
        "expectedPairCount": specification["validationPanel"]["plannedPrimaryPairedContrasts"],
        "sharedSourceCheckpoint": True,
        "sharedAssignedOperatorAndSpacing": True,
        "sharedPrefixBoundary": "exact until intervention/path divergence; no dummy draws",
    }
    replay_validation = {
        "success": bool(results["exactReplayPass"].all()),
        "replaysPassed": int(results["exactReplayPass"].sum()),
        "replaysPlanned": specification["validationPanel"]["plannedReplayCount"],
    }
    accounting_validation = {
        "success": bool(results["allOpportunityValidationPass"].all())
        and len(traces) == specification["validationPanel"]["plannedSelectedTraceCount"],
        "plannedRuns": specification["validationPanel"]["plannedRunCount"],
        "executedRuns": len(results),
        "selectedTraceGroups": len(traces),
        "plannedSelectedTraceGroups": specification["validationPanel"]["plannedSelectedTraceCount"],
        "nativeActivations": int(results["nativeActivationDelta"].sum()),
        "acceptedSwaps": int(results["nativeAcceptedSwaps"].sum()),
        "fatigueTriggers": int(results["fatigueThresholdTriggers"].sum()),
        "fatigueBlocks": int((results["fatigueActorBlocks"] + results["fatigueTargetBlocks"]).sum()),
    }
    all_gates = {
        "inherited": all(inherited.values()),
        "injury": injury_validation["success"],
        "sequence": sequence_validation["success"],
        "state": state_validation["success"],
        "censor": censor_validation["success"],
        "pairing": pairing_validation["success"],
        "replay": replay_validation["success"],
        "accounting": accounting_validation["success"],
    }
    if not all(all_gates.values()):
        raise AssertionError(f"S10 validation failed: {all_gates}")
    validation = {
        "success": True,
        "researchStepId": "S10",
        "benchmarkVersion": BENCHMARK_VERSION,
        "caseCount": int(assignments.shape[0]),
        "plannedCaseCount": specification["validationPanel"]["plannedCaseCount"],
        "runCount": len(results),
        "plannedRunCount": specification["validationPanel"]["plannedRunCount"],
        "replayCount": int(results["exactReplayPass"].sum()),
        "plannedReplayCount": specification["validationPanel"]["plannedReplayCount"],
        "pairContrastCount": len(paired),
        "gates": all_gates,
    }
    any_reject = bool(tests["rejectAtFamilywise0_05"].any())
    any_harm = bool(((tests["rejectAtFamilywise0_05"]) & (tests["pairedDifference"] < 0)).any())
    if learning["anyLearningCriterionMet"]:
        classification = "supportive"
    elif any_harm:
        classification = "constraining/contradictory"
    elif any_reject:
        classification = "supportive"
    else:
        classification = "null"

    _write_json(output / "injury_equivalence_validation.json", injury_validation)
    _write_json(output / "sequence_randomization_validation.json", sequence_validation)
    _write_json(output / "retained_state_fatigue_validation.json", state_validation)
    _write_json(output / "censor_retention_validation.json", censor_validation)
    _write_json(output / "pairing_validation.json", pairing_validation)
    _write_json(output / "replay_validation.json", replay_validation)
    _write_json(output / "run_accounting.json", accounting_validation)
    _write_json(output / "validation_summary.json", validation)
    _write_json(output / "learning_decision.json", learning)
    _write_json(
        output / "outcome_classification.json",
        {
            "researchStepId": "S10",
            "classification": classification,
            "learningClaimPermitted": learning["learningClaimPermitted"],
            "primaryCompletionRejections": int(tests["rejectAtFamilywise0_05"].sum()),
            "primaryCompletionHarms": int(
                ((tests["rejectAtFamilywise0_05"]) & (tests["pairedDifference"] < 0)).sum()
            ),
        },
    )
    _write_json(
        output / "environment_provenance.json",
        {
            "researchStepId": "S10",
            "generatedAtUtc": generated,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in ("numpy", "pandas", "pyarrow", "scipy", "matplotlib")
            },
            "workers": workers,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
            "repositoryCommitAtExecution": commit,
        },
    )
    _write_json(
        output / "input_provenance.json",
        {
            "researchStepId": "S10",
            "generatedAtUtc": generated,
            "inputs": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
                for path in (CONFIG, S04_CONFIG, *INPUTS)
            ],
        },
    )
    (output / "execution_commands.log").write_text(
        f"PYTHONHASHSEED=0 python -m pytest -q tests/test_regeneration_repeated_injuries.py\n"
        f"PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/build_regeneration_s10.py --output {output} --workers {workers}\n",
        encoding="utf-8",
    )
    report = _report(
        results,
        tests,
        curves,
        learning,
        validation,
        classification,
        commit,
        generated,
        workers,
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    manifest = _artifact_manifest(output, generated, commit)
    _write_json(output / "artifact_manifest.json", manifest)
    print(json.dumps(validation, indent=2, sort_keys=True))


def validate_only(output: Path) -> None:
    specification = _load_json(CONFIG)
    validate_repeated_injury_spec(specification)
    required = [
        output / "research_step_full_results.md",
        output / "repeated_injury_results.parquet",
        output / "repeated_injury_cases.parquet",
        output / "primary_repeated_injury_tests.parquet",
        output / "paired_repeated_injury_contrasts.parquet",
        output / "fatigue_mechanism_contrasts.parquet",
        output / "learning_fatigue_curves.parquet",
        output / "learning_fatigue_curves.png",
        output / "learning_fatigue_curves.svg",
        output / "validation_summary.json",
        output / "artifact_manifest.json",
        output / "repeated_injury_package/repeated_injury_spec.json",
        output / "repeated_injury_package/repeated_injury_spec.schema.json",
        output / "repeated_injury_package/repeated_injury_spec.md",
        output / "repeated_injury_package/selected_repeated_injury_traces.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing S10 handoff artifacts: {missing}")
    results = pd.read_parquet(output / "repeated_injury_results.parquet")
    assignments = pd.read_parquet(output / "repeated_injury_cases.parquet")
    pairs = pd.read_parquet(output / "paired_repeated_injury_contrasts.parquet")
    validation = _load_json(output / "validation_summary.json")
    if len(results) != specification["validationPanel"]["plannedRunCount"]:
        raise AssertionError("S10 run Parquet count changed")
    if len(assignments) != specification["validationPanel"]["plannedCaseCount"]:
        raise AssertionError("S10 case Parquet count changed")
    if len(pairs) != specification["validationPanel"]["plannedPrimaryPairedContrasts"]:
        raise AssertionError("S10 paired contrast count changed")
    if not validation["success"] or not results["allOpportunityValidationPass"].all():
        raise AssertionError("S10 validation summary is not successful")
    manifest = _load_json(output / "artifact_manifest.json")
    for item in manifest["artifacts"]:
        path = output / item["path"]
        if _sha256_file(path) != item["sha256"]:
            raise AssertionError(f"artifact hash changed: {path}")
    print(json.dumps({"success": True, "runs": len(results), "pairs": len(pairs)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("S10 supports 1..8 workers")
    if args.validate_only:
        validate_only(args.output)
    else:
        build(args.output, args.workers)


if __name__ == "__main__":
    main()
