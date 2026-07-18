#!/usr/bin/env python3
"""Build and validate E05 S11 structural/internal memory-reset evidence."""

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
from scripts.build_regeneration_s10 import (  # noqa: E402
    _case_id as _s10_case_id,
    _reconstruct_cases,
)
from src.regeneration.repeated_injuries import Spacing  # noqa: E402
from src.regeneration.state_memory_reset import (  # noqa: E402
    BENCHMARK_VERSION,
    SCRAMBLE_STREAM,
    STATE_MEMORY_RESET_SPEC_SCHEMA,
    ResetArm,
    exact_replay_state_memory_reset_case,
    run_state_memory_reset_case,
    validate_state_memory_reset_spec,
)
from src.regeneration.tasks import _seed  # noqa: E402


CONFIG = REPOSITORY / "configs/regeneration/s11_state_memory_reset.json"
S10_CONFIG = REPOSITORY / "configs/regeneration/s10_repeated_injuries.json"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S11")
ATTACHMENT_SIDECAR = Path(
    "/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/"
    "_metadata/ATTACHMENT.md"
)
INPUTS: tuple[Path, ...] = (
    Path("/workspace/AGENTS.md"),
    Path("/workspace/FULL_PLAN.md"),
    Path("/workspace/RESEARCH_PLAN.md"),
    Path("/workspace/PREVIOUS_ARTIFACTS.md"),
    Path("/workspace/PREVIOUS_ARTIFACTS.json"),
    Path("/workspace/DATASETS.md"),
    Path("/workspace/DATASET_CATALOG.json"),
    Path("/workspace/DATASET_AVAILABILITY.json"),
    Path("/workspace/CAPABILITIES.md"),
    Path("/workspace/CAPABILITY_AVAILABILITY.json"),
    Path("/workspace/input-attachments/MANIFEST.json"),
    ATTACHMENT_SIDECAR,
    *tuple(
        item
        for step in range(1, 11)
        for item in (
            Path(
                f"/artifacts/research_steps/S{step:02d}/"
                "research_step_full_results.md"
            ),
            Path(f"/artifacts/research_steps/S{step:02d}/validation_summary.json"),
        )
    ),
    Path("/artifacts/research_steps/S01/task_spec.json"),
    Path("/artifacts/research_steps/S02/timing_spec.json"),
    Path("/artifacts/research_steps/S02/preinjury_states.parquet"),
    Path("/artifacts/research_steps/S03/lesion_library/lesion_spec.json"),
    Path(
        "/artifacts/research_steps/S03/lesion_library/operator_fixtures.parquet"
    ),
    Path(
        "/artifacts/research_steps/S04/dynamic_fault_package/"
        "dynamic_fault_spec.json"
    ),
    Path(
        "/artifacts/research_steps/S07/memory_variants/local_memory_spec.json"
    ),
    Path(
        "/artifacts/research_steps/S08/plasticity_package/"
        "policy_plasticity_spec.json"
    ),
    Path(
        "/artifacts/research_steps/S09/target_change_package/"
        "target_change_spec.json"
    ),
    Path(
        "/artifacts/research_steps/S10/repeated_injury_package/"
        "repeated_injury_spec.json"
    ),
    Path("/artifacts/research_steps/S10/repeated_injury_results.parquet"),
    Path("/artifacts/research_steps/S10/repeated_injury_cases.parquet"),
    Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
    Path("/previous-artifacts/E01/specification/transition_spec.md"),
    Path(
        "/previous-artifacts/E02/release/causal_simulator_extension/"
        "release_manifest.json"
    ),
    Path("/previous-artifacts/E02/research_steps/S02/action_interface_spec.md"),
    Path(
        "/previous-artifacts/E02/research_steps/S04/scheduler_package/"
        "scheduler_contract.md"
    ),
    Path(
        "/previous-artifacts/E02/research_steps/S04/scheduler_package/"
        "opportunity_ledger_validation.json"
    ),
    Path(
        "/previous-artifacts/E02/research_steps/S04/scheduler_package/"
        "rng_consumption_validation.json"
    ),
    Path(
        "/previous-artifacts/E02/research_steps/S05/fault_package/"
        "fault_semantics_contract.md"
    ),
    Path(
        "/previous-artifacts/E02/research_steps/S05/fault_package/"
        "information_boundary_validation.json"
    ),
    Path(
        "/previous-artifacts/E02/research_steps/S08/"
        "semantic_random_stream_specification.json"
    ),
    Path(
        "/previous-artifacts/E02/research_steps/S08/"
        "stream_name_isolation_validation.json"
    ),
    Path(
        "/previous-artifacts/E02/research_steps/S09/"
        "ledger_identity_validation.json"
    ),
    Path("/previous-artifacts/E02/research_steps/S12/research_step_full_results.md"),
    Path(
        "/previous-artifacts/E02/research_steps/S12/"
        "modeling_prespecification.json"
    ),
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
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True
    ).strip()


def _validate_inputs() -> dict[str, bool]:
    missing = [
        str(path) for path in (CONFIG, S10_CONFIG, *INPUTS) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing required S01-S10/E01/E02 inputs: {missing}")
    gates = {
        f"s{step:02d}": bool(
            _load_json(
                Path(
                    f"/artifacts/research_steps/S{step:02d}/"
                    "validation_summary.json"
                )
            )["success"]
        )
        for step in range(1, 11)
    }
    gates["e01"] = bool(
        _load_json(
            Path(
                "/previous-artifacts/E01/release/reference_simulator/"
                "release_manifest.json"
            )
        )["validationSuccess"]
    )
    gates["e02"] = bool(
        _load_json(
            Path(
                "/previous-artifacts/E02/release/causal_simulator_extension/"
                "release_manifest.json"
            )
        )["smokeValidation"]["success"]
    )
    if not all(gates.values()):
        raise RuntimeError(f"an inherited validation gate failed: {gates}")
    return gates


@dataclass(frozen=True, slots=True)
class MemoryResetJob:
    source: Any
    retain_trace: bool


def _memory_case_id(job: MemoryResetJob) -> str:
    return "e05mc11:" + sha256_json(
        {
            "repeatedCaseId": _s10_case_id(job.source),
            "spacingId": Spacing.REST_20N.value,
            "resetDesign": "arrangement_native_fatigue_2x2x2_v1",
        }
    )


def _run_id(case_id: str, arm: str) -> str:
    return "e05mr11:" + sha256_json({"memoryResetCaseId": case_id, "arm": arm})


def _case_address(job: MemoryResetJob) -> int:
    return int(sha256_json({"memoryResetCaseId": _memory_case_id(job)})[:16], 16)


def _flatten(job: MemoryResetJob, arm: str, run: Mapping[str, Any]) -> dict[str, Any]:
    case_id = _memory_case_id(job)
    base = job.source.base
    episode1 = run["episode1"]
    episode2 = run["episode2"]
    stabilization = run["postRecoveryStabilization"]
    lesion1 = run["episode1Lesion"]
    lesion2 = run["episode2Lesion"]
    audit = run["resetAudit"]
    before = audit["before"]
    after = audit["after"]
    scramble = audit["scramble"]
    process = run["processLedger"]
    final_fatigue = run["finalFatigue"]
    return {
        "schemaVersion": "e05.s11.state-memory-reset-result.v1",
        "benchmarkVersion": BENCHMARK_VERSION,
        "memoryResetRunId": _run_id(case_id, arm),
        "memoryResetCaseId": case_id,
        "repeatedCaseId": _s10_case_id(job.source),
        "arm": arm,
        "isFactorialArm": run["isFactorialArm"],
        "arrangementResetAssigned": run["arrangementResetAssigned"],
        "nativeStateResetAssigned": run["nativeStateResetAssigned"],
        "fatigueResetAssigned": run["fatigueResetAssigned"],
        "sequenceId": run["sequenceId"],
        "operatorId": run["operatorId"],
        "spacingId": run["spacingId"],
        "s01PairingBlockId": base.s01_pairing_block_id,
        "timingConditionId": base.timing_condition_id,
        "clock": base.clock,
        "nominalFraction": base.nominal_fraction,
        "n": base.n,
        "policy": base.policy,
        "direction": base.direction,
        "replicateOrdinal": base.replicate,
        "sourceScenarioId": base.scenario.scenario_id,
        "sourceCheckpointHash": base.checkpoint.state_hash,
        "recoveryBudget": base.recovery_budget,
        "caseAddressUint64": _case_address(job),
        "scrambleStream": SCRAMBLE_STREAM,
        "episode1PostLesionStateHash": lesion1["postStateHash"],
        "episode1Completed": episode1["completed"],
        "episode1StopReason": episode1["stopReason"],
        "episode1Duration": episode1["durationOpportunities"],
        "episode1FinalDistance": episode1["finalDistance"],
        "episode1DistanceAuc": episode1["distanceAuc"],
        "firstStageEligible": run["firstStageEligible"],
        "stabilizationAttempted": stabilization is not None,
        "stabilizationSuccess": (
            None if stabilization is None else stabilization["success"]
        ),
        "stabilizationOpportunities": (
            None if stabilization is None else stabilization["opportunities"]
        ),
        "restStable": None if run["rest"] is None else run["rest"]["stable"],
        "restTargetDepartures": (
            None if run["rest"] is None else run["rest"]["targetDepartures"]
        ),
        "episode2InjuryAdministered": run["episode2InjuryAdministered"],
        "episode2PostLesionStateHash": (
            None if lesion2 is None else lesion2["postStateHash"]
        ),
        "episode2DistanceBefore": (
            None
            if lesion2 is None
            else lesion2["severity"]["validPostTargetOrderDistanceBefore"]
        ),
        "episode2DistanceAfter": (
            None
            if lesion2 is None
            else lesion2["severity"]["validPostTargetOrderDistanceAfter"]
        ),
        "episode2AffectedIdentityCount": (
            None
            if lesion2 is None
            else lesion2["severity"]["directlyAffectedIdentityCount"]
        ),
        "preResetStateHash": before["stateHash"],
        "preResetArrangementSha256": before["arrangementSha256"],
        "preResetNativeSha256": before["nativeSelectionCursorsSha256"],
        "preResetEngineeredSha256": before["engineeredStateSha256"],
        "preResetFatigueSha256": before["fatigueSha256"],
        "preResetProtectedSha256": before["protectedExecutionSha256"],
        "preResetDistance": before["targetDistance"],
        "preResetFatigueLoadSum": before["fatigueLoadSum"],
        "preResetFatiguedIdentityCount": before["fatiguedIdentityCount"],
        "postResetStateHash": after["stateHash"],
        "postResetArrangementSha256": after["arrangementSha256"],
        "postResetNativeSha256": after["nativeSelectionCursorsSha256"],
        "postResetEngineeredSha256": after["engineeredStateSha256"],
        "postResetFatigueSha256": after["fatigueSha256"],
        "postResetProtectedSha256": after["protectedExecutionSha256"],
        "postResetDistance": after["targetDistance"],
        "postResetFatigueLoadSum": after["fatigueLoadSum"],
        "postResetFatiguedIdentityCount": after["fatiguedIdentityCount"],
        "resetBoundaryObserved": run["resetBoundaryObserved"],
        "resetApplied": run["resetApplied"],
        "resetApplicable": audit["resetApplicable"],
        "resetFeasible": audit["resetFeasible"],
        "resetReason": audit["resetReason"],
        "resetAuditDigest": audit["auditDigest"],
        "structuralGatewayInvoked": audit["structuralGatewayInvoked"],
        "internalGatewayInvoked": audit["internalGatewayInvoked"],
        "engineeredStateFieldCount": audit["engineeredStateFieldCount"],
        "supplementalStructuralWrites": audit["supplementalStructuralWrites"],
        "supplementalNativeWrites": audit["supplementalNativeWrites"],
        "supplementalFatigueWrites": audit["supplementalFatigueWrites"],
        "scrambleFeasible": scramble["feasible"],
        "scrambleReason": scramble["reason"],
        "scrambleTargetDistance": scramble["targetDistance"],
        "scrambleAttempts": scramble["attempts"],
        "scrambleDrawBlocks": scramble["drawBlocksConsumed"],
        "scrambleIdentityDisplacementL1": scramble.get("identityDisplacementL1"),
        "scrambleMovedIdentityCount": scramble.get("movedIdentityCount"),
        "arrangementActuallyChanged": (
            before["arrangementSha256"] != after["arrangementSha256"]
        ),
        "nativeStateActuallyChanged": (
            before["nativeSelectionCursorsSha256"]
            != after["nativeSelectionCursorsSha256"]
        ),
        "fatigueActuallyChanged": before["fatigueSha256"] != after["fatigueSha256"],
        "episode2RecoveryObserved": run["episode2RecoveryObserved"],
        "episode2OutcomeCause": run["episode2OutcomeCause"],
        "episode2Completed": bool(episode2 and episode2["completed"]),
        "episode2StopReason": None if episode2 is None else episode2["stopReason"],
        "episode2Duration": (
            None if episode2 is None else episode2["durationOpportunities"]
        ),
        "episode2FinalDistance": (
            None if episode2 is None else episode2["finalDistance"]
        ),
        "episode2DistanceAuc": (
            None if episode2 is None else episode2["distanceAuc"]
        ),
        "jointSequenceSuccess": run["jointSequenceSuccess"],
        "restrictedEpisode2Time": run["restrictedEpisode2Time"],
        "firstStageCompetingTerminal": not run["firstStageEligible"],
        "nativeActivationDelta": run["nativeLedgerDelta"]["activations"],
        "nativeAcceptedSwaps": run["nativeLedgerDelta"]["acceptedSwaps"],
        "nativeRejections": run["nativeLedgerDelta"]["rejections"],
        "fatigueMovementParticipations": process["fatigueMovementParticipations"],
        "fatigueThresholdTriggers": process["fatigueThresholdTriggers"],
        "fatigueActorBlocks": process["fatigueActorBlocks"],
        "fatigueTargetBlocks": process["fatigueTargetBlocks"],
        "finalStateHash": run["finalState"]["stateHash"],
        "finalFatigueSha256": sha256_json(final_fatigue),
        "nativeEventDigest": run["nativeEventDigest"],
        "nativeEventCount": run["nativeEventCount"],
        "processAuditDigest": run["processAuditDigest"],
        "processAuditCount": run["processAuditCount"],
        "runDigest": run["runDigest"],
        "allOpportunityValidationPass": all(run["validation"].values()),
        "allResetValidationPass": all(audit["validation"].values()),
        "validationJson": json.dumps(
            run["validation"], sort_keys=True, separators=(",", ":")
        ),
        "resetValidationJson": json.dumps(
            audit["validation"], sort_keys=True, separators=(",", ":")
        ),
        "nativeLedgerJson": json.dumps(
            run["nativeLedgerDelta"], sort_keys=True, separators=(",", ":")
        ),
        "processLedgerJson": json.dumps(
            process, sort_keys=True, separators=(",", ":")
        ),
        "exactReplayPass": True,
        "traceSelected": job.retain_trace,
    }


def _execute_job(job: MemoryResetJob) -> dict[str, Any]:
    base = job.source.base
    seed = _seed("injury", base.s01_pairing_block_id)
    result = run_state_memory_reset_case(
        base.scenario,
        base.checkpoint,
        pairing_id=base.s01_pairing_block_id,
        timing_condition_id=base.timing_condition_id,
        sequence_id=job.source.sequence_id,
        injury_seed=seed,
        recovery_budget=base.recovery_budget,
        case_address=_case_address(job),
        retain_trace=job.retain_trace,
    )
    exact_replay_state_memory_reset_case(
        result,
        base.scenario,
        base.checkpoint,
        pairing_id=base.s01_pairing_block_id,
        timing_condition_id=base.timing_condition_id,
        sequence_id=job.source.sequence_id,
        injury_seed=seed,
        recovery_budget=base.recovery_budget,
        case_address=_case_address(job),
        retain_trace=job.retain_trace,
    )
    trace = None
    if job.retain_trace:
        trace = {
            "memoryResetCaseId": _memory_case_id(job),
            "repeatedCaseId": _s10_case_id(job.source),
            "sequenceId": job.source.sequence_id,
            "arms": {
                arm: {
                    "runDigest": run["runDigest"],
                    "firstStageEligible": run["firstStageEligible"],
                    "preResetState": run["preResetState"],
                    "resetAudit": run["resetAudit"],
                    "episode2": run["episode2"],
                    "episode2OutcomeCause": run["episode2OutcomeCause"],
                    "traceHead": run["traceHead"],
                    "traceTail": run["traceTail"],
                    "validation": run["validation"],
                }
                for arm, run in result.items()
            },
        }
    return {
        "rows": [_flatten(job, arm, run) for arm, run in result.items()],
        "trace": trace,
    }


def _reconstruct_s11_jobs(
    specification: Mapping[str, Any],
) -> tuple[list[MemoryResetJob], pd.DataFrame]:
    s10_jobs, assignments = _reconstruct_cases(_load_json(S10_CONFIG))
    sources = [job for job in s10_jobs if job.spacing == Spacing.REST_20N]
    if len(sources) != specification["validationPanel"]["plannedCaseCount"]:
        raise RuntimeError(f"expected 384 S10 rest-20n cases, got {len(sources)}")
    selected: set[str] = set()
    for sequence_id in sorted({job.sequence_id for job in sources}):
        candidates = sorted(
            (job for job in sources if job.sequence_id == sequence_id),
            key=lambda item: _s10_case_id(item),
        )
        selected.update(_s10_case_id(item) for item in candidates[:8])
    jobs = [
        MemoryResetJob(source, _s10_case_id(source) in selected)
        for source in sorted(sources, key=_s10_case_id)
    ]
    cases = assignments.copy()
    source_by_key = {
        (item.base.s01_pairing_block_id, item.base.timing_condition_id): item
        for item in sources
    }
    cases["repeatedCaseId"] = [
        _s10_case_id(source_by_key[(row.s01PairingBlockId, row.timingConditionId)])
        for row in cases.itertuples()
    ]
    cases["memoryResetCaseId"] = [
        _memory_case_id(
            MemoryResetJob(
                source_by_key[(row.s01PairingBlockId, row.timingConditionId)],
                False,
            )
        )
        for row in cases.itertuples()
    ]
    cases["caseAddressUint64"] = [
        _case_address(
            MemoryResetJob(
                source_by_key[(row.s01PairingBlockId, row.timingConditionId)],
                False,
            )
        )
        for row in cases.itertuples()
    ]
    cases["spacingId"] = Spacing.REST_20N.value
    cases["traceSelected"] = cases["repeatedCaseId"].isin(selected)
    return jobs, cases


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    estimate = float(values.mean())
    if len(values) < 2 or float(values.std(ddof=1)) == 0:
        return estimate, estimate, estimate
    half = float(t.ppf(0.975, len(values) - 1)) * float(
        values.std(ddof=1) / math.sqrt(len(values))
    )
    return estimate, estimate - half, estimate + half


def _sign_p(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 1.0
    return float(binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue)


EFFECTS: dict[str, tuple[str, ...]] = {
    "arrangement_main": ("a",),
    "native_state_main": ("n",),
    "fatigue_main": ("f",),
    "arrangement_by_native": ("a", "n"),
    "arrangement_by_fatigue": ("a", "f"),
    "native_by_fatigue": ("n", "f"),
    "arrangement_by_native_by_fatigue": ("a", "n", "f"),
}


def _effect_coefficient(a: int, n: int, f: int, factors: tuple[str, ...]) -> float:
    coding = {"a": 2 * a - 1, "n": 2 * n - 1, "f": 2 * f - 1}
    coefficient = float(np.prod([coding[item] for item in factors]))
    unmentioned = 3 - len(factors)
    return coefficient / (2**unmentioned)


def _factorial_contrasts(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factorial = results[results["isFactorialArm"]].copy()
    case_rows: list[dict[str, Any]] = []
    composite_rows: list[dict[str, Any]] = []
    for (case_id, sequence_id), group in factorial.groupby(
        ["memoryResetCaseId", "sequenceId"], sort=True
    ):
        if len(group) != 8:
            raise AssertionError("incomplete S11 factorial case")
        cells = {
            (
                int(row.arrangementResetAssigned),
                int(row.nativeStateResetAssigned),
                int(row.fatigueResetAssigned),
            ): row
            for row in group.itertuples()
        }
        if len(cells) != 8:
            raise AssertionError("duplicate S11 factorial cell")
        for effect_id, factors in EFFECTS.items():
            success = 0.0
            duration = 0.0
            for (a, n, f), row in cells.items():
                coefficient = _effect_coefficient(a, n, f, factors)
                success += coefficient * float(row.jointSequenceSuccess)
                duration += coefficient * float(row.restrictedEpisode2Time)
            case_rows.append(
                {
                    "memoryResetCaseId": case_id,
                    "sequenceId": sequence_id,
                    "effectId": effect_id,
                    "successContrast": success,
                    "restrictedTimeContrast": duration,
                    "population": "full_assigned_population",
                }
            )
        corner_coefficients = {
            (1, 1, 1): 1.0,
            (1, 0, 0): -1.0,
            (0, 1, 1): -1.0,
            (0, 0, 0): 1.0,
        }
        composite_rows.append(
            {
                "memoryResetCaseId": case_id,
                "sequenceId": sequence_id,
                "effectId": "arrangement_by_internal_composite_corner",
                "successContrast": sum(
                    coefficient * float(cells[cell].jointSequenceSuccess)
                    for cell, coefficient in corner_coefficients.items()
                ),
                "restrictedTimeContrast": sum(
                    coefficient * float(cells[cell].restrictedEpisode2Time)
                    for cell, coefficient in corner_coefficients.items()
                ),
                "population": "full_assigned_population",
            }
        )
    case_table = pd.DataFrame(case_rows)
    effect_rows: list[dict[str, Any]] = []
    for (sequence_id, effect_id), group in case_table.groupby(
        ["sequenceId", "effectId"], sort=True
    ):
        success = group["successContrast"].to_numpy(dtype=float)
        duration = group["restrictedTimeContrast"].to_numpy(dtype=float)
        estimate, lower, upper = _mean_ci(success)
        time_estimate, time_lower, time_upper = _mean_ci(duration)
        effect_rows.append(
            {
                "sequenceId": sequence_id,
                "effectId": effect_id,
                "nCases": len(group),
                "meanSuccessContrast": estimate,
                "successCi95Lower": lower,
                "successCi95Upper": upper,
                "successPositiveCases": int((success > 0).sum()),
                "successNegativeCases": int((success < 0).sum()),
                "successRawPValue": _sign_p(success),
                "meanRestrictedTimeContrast": time_estimate,
                "timeCi95Lower": time_lower,
                "timeCi95Upper": time_upper,
                "timePositiveCases": int((duration > 0).sum()),
                "timeNegativeCases": int((duration < 0).sum()),
                "timeRawPValue": _sign_p(duration),
            }
        )
    effects = pd.DataFrame(effect_rows)
    effects["successHolmPValue"] = _holm(effects["successRawPValue"].tolist())
    effects["successRejectAtFamilywise0_05"] = effects["successHolmPValue"] < 0.05
    effects["timeHolmPValue"] = _holm(effects["timeRawPValue"].tolist())
    effects["timeRejectAtFamilywise0_05"] = effects["timeHolmPValue"] < 0.05

    composite_case = pd.DataFrame(composite_rows)
    composite_effect_rows: list[dict[str, Any]] = []
    for sequence_id, group in composite_case.groupby("sequenceId", sort=True):
        success = group["successContrast"].to_numpy(dtype=float)
        duration = group["restrictedTimeContrast"].to_numpy(dtype=float)
        estimate, lower, upper = _mean_ci(success)
        time_estimate, time_lower, time_upper = _mean_ci(duration)
        composite_effect_rows.append(
            {
                "sequenceId": sequence_id,
                "effectId": "arrangement_by_internal_composite_corner",
                "nCases": len(group),
                "meanSuccessContrast": estimate,
                "successCi95Lower": lower,
                "successCi95Upper": upper,
                "successRawPValue": _sign_p(success),
                "meanRestrictedTimeContrast": time_estimate,
                "timeCi95Lower": time_lower,
                "timeCi95Upper": time_upper,
                "timeRawPValue": _sign_p(duration),
            }
        )
    composite_effect = pd.DataFrame(composite_effect_rows)
    composite_effect["successHolmPValue"] = _holm(
        composite_effect["successRawPValue"].tolist()
    )
    composite_effect["timeHolmPValue"] = _holm(
        composite_effect["timeRawPValue"].tolist()
    )
    return case_table, effects, composite_case, composite_effect


def _sham_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fields = [
        "firstStageEligible",
        "episode2Completed",
        "episode2StopReason",
        "episode2Duration",
        "episode2FinalDistance",
        "episode2DistanceAuc",
        "jointSequenceSuccess",
        "restrictedEpisode2Time",
        "finalStateHash",
        "finalFatigueSha256",
        "nativeLedgerJson",
        "processLedgerJson",
        "nativeEventDigest",
        "processAuditDigest",
    ]
    for case_id, group in results.groupby("memoryResetCaseId", sort=True):
        keyed = group.set_index("arm")
        reference = keyed.loc[ResetArm.A0_N0_F0.value]
        for sham in (ResetArm.SHAM_ARRANGEMENT, ResetArm.SHAM_INTERNAL):
            control = keyed.loc[sham.value]
            equalities = {
                field: bool(
                    (pd.isna(reference[field]) and pd.isna(control[field]))
                    or reference[field] == control[field]
                )
                for field in fields
            }
            rows.append(
                {
                    "memoryResetCaseId": case_id,
                    "sequenceId": reference["sequenceId"],
                    "referenceArm": ResetArm.A0_N0_F0.value,
                    "shamArm": sham.value,
                    "allBehaviorFieldsExact": all(equalities.values()),
                    "fieldEqualityJson": json.dumps(
                        equalities, sort_keys=True, separators=(",", ":")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _s10_anchor_validation(results: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    reference = results[results["arm"] == ResetArm.A0_N0_F0.value].copy()
    s10 = pd.read_parquet(
        "/artifacts/research_steps/S10/repeated_injury_results.parquet"
    )
    s10 = s10[
        (s10["arm"] == "prior_injury_retained_fatigue")
        & (s10["spacingId"] == Spacing.REST_20N.value)
    ].copy()
    merged = reference.merge(
        s10,
        on="repeatedCaseId",
        suffixes=("S11", "S10"),
        validate="one_to_one",
    )
    comparisons = {
        "sequenceId": ("sequenceIdS11", "sequenceIdS10"),
        "episode1Completed": ("episode1CompletedS11", "episode1CompletedS10"),
        "episode1StopReason": ("episode1StopReasonS11", "episode1StopReasonS10"),
        "episode1Duration": ("episode1DurationS11", "episode1DurationS10"),
        "episode1FinalDistance": (
            "episode1FinalDistanceS11",
            "episode1FinalDistanceS10",
        ),
        "firstStageEligible": ("firstStageEligibleS11", "firstStageEligibleS10"),
        "stabilizationSuccess": (
            "stabilizationSuccessS11",
            "stabilizationSuccessS10",
        ),
        "stabilizationOpportunities": (
            "stabilizationOpportunitiesS11",
            "stabilizationOpportunitiesS10",
        ),
        "restStable": ("restStableS11", "restStableS10"),
        "episode2Completed": ("episode2CompletedS11", "episode2CompletedS10"),
        "episode2StopReason": ("episode2StopReasonS11", "episode2StopReasonS10"),
        "episode2Duration": ("episode2DurationS11", "episode2DurationS10"),
        "episode2FinalDistance": (
            "episode2FinalDistanceS11",
            "episode2FinalDistanceS10",
        ),
        "episode2DistanceAuc": (
            "episode2DistanceAucS11",
            "episode2DistanceAucS10",
        ),
        "jointSequenceSuccess": (
            "jointSequenceSuccessS11",
            "jointSequenceSuccessS10",
        ),
        "restrictedEpisode2Time": (
            "restrictedEpisode2TimeS11",
            "restrictedEpisode2TimeS10",
        ),
        "finalStateHash": ("finalStateHashS11", "finalStateHashS10"),
        "nativeActivationDelta": (
            "nativeActivationDeltaS11",
            "nativeActivationDeltaS10",
        ),
        "nativeAcceptedSwaps": (
            "nativeAcceptedSwapsS11",
            "nativeAcceptedSwapsS10",
        ),
        "processAuditDigest": ("processAuditDigestS11", "processAuditDigestS10"),
    }
    rows = []
    for name, (left, right) in comparisons.items():
        equal = (merged[left] == merged[right]) | (
            merged[left].isna() & merged[right].isna()
        )
        rows.append(
            {
                "field": name,
                "rowsCompared": len(merged),
                "rowsExact": int(equal.sum()),
                "mismatches": int((~equal).sum()),
                "success": bool(equal.all()),
            }
        )
    table = pd.DataFrame(rows)
    validation = {
        "success": len(merged) == 384 and bool(table["success"].all()),
        "referenceRows": len(reference),
        "s10Rows": len(s10),
        "joinedRows": len(merged),
        "fieldsCompared": len(table),
        "fieldMismatches": int(table["mismatches"].sum()),
        "s10ReferenceArm": "prior_injury_retained_fatigue",
        "s10Spacing": Spacing.REST_20N.value,
        "s11ReferenceArm": ResetArm.A0_N0_F0.value,
    }
    return validation, table


def _reset_audit_table(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "memoryResetRunId",
        "memoryResetCaseId",
        "arm",
        "sequenceId",
        "n",
        "policy",
        "firstStageEligible",
        "resetApplicable",
        "resetFeasible",
        "resetReason",
        "arrangementResetAssigned",
        "nativeStateResetAssigned",
        "fatigueResetAssigned",
        "arrangementActuallyChanged",
        "nativeStateActuallyChanged",
        "fatigueActuallyChanged",
        "preResetArrangementSha256",
        "postResetArrangementSha256",
        "preResetNativeSha256",
        "postResetNativeSha256",
        "preResetEngineeredSha256",
        "postResetEngineeredSha256",
        "preResetFatigueSha256",
        "postResetFatigueSha256",
        "preResetProtectedSha256",
        "postResetProtectedSha256",
        "preResetDistance",
        "postResetDistance",
        "scrambleFeasible",
        "scrambleReason",
        "scrambleTargetDistance",
        "scrambleAttempts",
        "scrambleDrawBlocks",
        "scrambleIdentityDisplacementL1",
        "scrambleMovedIdentityCount",
        "structuralGatewayInvoked",
        "internalGatewayInvoked",
        "engineeredStateFieldCount",
        "supplementalStructuralWrites",
        "supplementalNativeWrites",
        "supplementalFatigueWrites",
        "resetAuditDigest",
        "allResetValidationPass",
        "resetValidationJson",
    ]
    return results[columns].copy()


def _effect_figure(effects: pd.DataFrame, output: Path) -> None:
    labels = list(EFFECTS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    colors = {
        "repeat_segment_reversal_v1": "#1f77b4",
        "repeat_block_transposition_v1": "#d95f02",
    }
    offsets = {
        "repeat_segment_reversal_v1": -0.12,
        "repeat_block_transposition_v1": 0.12,
    }
    y = np.arange(len(labels))
    for sequence_id, group in effects.groupby("sequenceId", sort=True):
        keyed = group.set_index("effectId").loc[labels]
        axes[0].errorbar(
            keyed["meanSuccessContrast"],
            y + offsets[sequence_id],
            xerr=np.vstack(
                [
                    keyed["meanSuccessContrast"] - keyed["successCi95Lower"],
                    keyed["successCi95Upper"] - keyed["meanSuccessContrast"],
                ]
            ),
            fmt="o",
            color=colors[sequence_id],
            label=sequence_id.replace("repeat_", "").replace("_v1", ""),
        )
        axes[1].errorbar(
            keyed["meanRestrictedTimeContrast"],
            y + offsets[sequence_id],
            xerr=np.vstack(
                [
                    keyed["meanRestrictedTimeContrast"] - keyed["timeCi95Lower"],
                    keyed["timeCi95Upper"] - keyed["meanRestrictedTimeContrast"],
                ]
            ),
            fmt="o",
            color=colors[sequence_id],
            label=sequence_id.replace("repeat_", "").replace("_v1", ""),
        )
    for axis in axes:
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y, [item.replace("_", " ") for item in labels])
        axis.grid(alpha=0.25)
    axes[0].set_title("Full-population success reset contrasts")
    axes[0].set_xlabel("Reset minus preserve")
    axes[1].set_title("Restricted recovery-time reset contrasts")
    axes[1].set_xlabel("Reset minus preserve, opportunities")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "factorial_state_reset_effects.png", dpi=180)
    fig.savefig(output / "factorial_state_reset_effects.svg")
    plt.close(fig)


def _spec_markdown(specification: Mapping[str, Any]) -> str:
    return f"""# S11 structural/internal memory-reset specification

Frozen at `{specification['frozenAtUtc']}` before confirmatory execution.

## Frozen question

{specification['frozenQuestion']}

## Population and boundary

All 384 exact S02 checkpoints retain their exact S10 injury assignment. The
first injury, S01 stabilization certificate, `20*n` target-stable rest, second
matched injury, phase-local `100*n^2` budgets, original scenario/RNG root,
ledgers, and competing-terminal rules are unchanged. A first-stage terminal is
retained in all ten assigned arms; no reset or second recovery is substituted.

## State partitions

- Arrangement is the occupancy identity sequence. Reset uses the isolated
  `{SCRAMBLE_STREAM}` construction stream to select a non-identical exact-
  inversion-distance permutation.
- Native resettable state is the Selection-cursor map. Reset restores the exact
  source S02 checkpoint snapshot.
- Engineered state is explicitly empty: no S07, S08, or S09 state is promoted.
- Fatigue is kept separate and reset to zero load/no cooldown when assigned.
- Event index, runtime streams, native/process ledgers and audit history are
  always preserved.

## Factorial, shams, and estimands

Eight arms cross arrangement, native state, and fatigue preserve/reset. Two
no-op serialization shams must be exactly behaviorally equivalent to the all-
preserved reference. Primary full-population estimands are the seven 2x2x2
factorial effects for joint success and restricted recovery time, separately by
injury sequence. The four-corner arrangement-by-composite-internal interaction
is retained as an additional prespecified summary.

## Claim boundary

{specification['claimBoundary']}
"""


def _report(
    *,
    generated: str,
    commit: str,
    workers: int,
    results: pd.DataFrame,
    effects: pd.DataFrame,
    composite_effects: pd.DataFrame,
    validations: Mapping[str, Mapping[str, Any]],
    classification: str,
    test_summary: str,
) -> str:
    reference = results[results["arm"] == ResetArm.A0_N0_F0.value]
    arm_summary = (
        results.groupby("arm", sort=True)
        .agg(
            runs=("memoryResetRunId", "size"),
            resetBoundaryObserved=("resetBoundaryObserved", "sum"),
            jointSuccesses=("jointSequenceSuccess", "sum"),
            meanRestrictedTime=("restrictedEpisode2Time", "mean"),
        )
        .reset_index()
    )
    arm_lines = "\n".join(
        f"| `{row.arm}` | {row.runs} | {int(row.resetBoundaryObserved)} | "
        f"{int(row.jointSuccesses)} | {row.meanRestrictedTime:,.1f} |"
        for row in arm_summary.itertuples()
    )
    effect_lines = "\n".join(
        f"| `{row.sequenceId}` | `{row.effectId}` | "
        f"{row.meanSuccessContrast:.4f} "
        f"[{row.successCi95Lower:.4f}, {row.successCi95Upper:.4f}] | "
        f"{row.successHolmPValue:.4g} | "
        f"{row.meanRestrictedTimeContrast:,.1f} "
        f"[{row.timeCi95Lower:,.1f}, {row.timeCi95Upper:,.1f}] | "
        f"{row.timeHolmPValue:.4g} |"
        for row in effects.itertuples()
    )
    composite_lines = "\n".join(
        f"| `{row.sequenceId}` | {row.meanSuccessContrast:.4f} | "
        f"{row.successHolmPValue:.4g} | "
        f"{row.meanRestrictedTimeContrast:,.1f} | {row.timeHolmPValue:.4g} |"
        for row in composite_effects.itertuples()
    )
    significant = effects[
        effects["successRejectAtFamilywise0_05"]
        | effects["timeRejectAtFamilywise0_05"]
    ]
    significant_text = (
        ", ".join(
            f"{row.sequenceId}/{row.effectId}"
            for row in significant.itertuples()
        )
        if len(significant)
        else "none"
    )
    competing = int((~reference["firstStageEligible"]).sum())
    actual_native = int(
        results.loc[results["nativeStateResetAssigned"], "nativeStateActuallyChanged"].sum()
    )
    actual_fatigue = int(
        results.loc[results["fatigueResetAssigned"], "fatigueActuallyChanged"].sum()
    )
    actual_structure = int(
        results.loc[
            results["arrangementResetAssigned"], "arrangementActuallyChanged"
        ].sum()
    )
    validation_line = "; ".join(
        f"{name}={'PASS' if item['success'] else 'FAIL'}"
        for name, item in validations.items()
    )
    outcome_phrase = {
        "supportive": (
            "At least one prespecified state contribution remained significant "
            "after multiplicity adjustment while both shams and the S10 anchor "
            "were exact."
        ),
        "null": (
            "No prespecified factorial state contribution survived multiplicity "
            "adjustment; shams and the S10 anchor remained exact."
        ),
        "constraining/contradictory": (
            "A prespecified validation or interpretation boundary materially "
            "constrained the S10 operational memory claim."
        ),
    }[classification]
    return f"""# Research step full results — S11 Separate structural from internal memory

## Top summary

- **Research step ID:** S11 — Separate structural from internal memory.
- **Completion status:** Complete on 2026-07-18; stopped before S12.
- **Artifacts written:** Canonical report; frozen JSON/Markdown specification and schema; 3,840-row full-population result and reset-audit tables; 384-case assignment table; 2,688 per-case factorial contrasts; 14 primary factorial effects; composite-interaction and 768 sham-pair tables; matched-scramble, injury, replay, pairing, censor/accounting, residual-confounding, S10-anchor, provenance, and artifact manifests; effect PNG/SVG; 16 selected ten-arm trace groups.
- **Validation result:** **PASS — {validation_line}; all {len(results):,} planned runs and exact replays were retained with zero substitutions or silent exclusions.**
- **Outcome classification:** **{classification}.** {outcome_phrase}
- **Caveats or blockers:** Resets occur only at the observable post-second-lesion boundary, so the {competing} histories that terminate earlier remain explicit competing terminals and cannot reveal a reset response. Exact inversion-distance matching does not match every local motif or policy-specific difficulty. Engineered S07–S09 state is absent by design. The intervention does not identify natural mediation or biological memory.
- **Lay summary:** The same repeated-injury history was forked into versions that kept or reset the current cell arrangement, the native Selection cursor state, and movement fatigue. Every early failure stayed in every comparison. This separates effects of transparent simulator state at one defined boundary; it does not show that biological cells store or learn information.
- **Recommended next action:** Chief Scientist review. If the bounded reset evidence is accepted, separately authorize S12 held-out transfer; do not start S12 from this handoff.

## Frozen question and outcome

S11 asked whether full-population repeated-injury outcomes at S10's observable
`rest_20n` second-injury boundary could be localized to current arrangement,
native policy state, endogenous fatigue, or their interactions. The exact
ten-arm contract was frozen at `2026-07-18T12:59:19Z` before confirmatory
execution. The detected primary effects were: {significant_text}.

The outcome is **{classification}** under the frozen rule. Reset effects are
oriented reset minus preserve. A positive success contrast or negative
restricted-time contrast favors reset; the reverse indicates that preserving
the state was beneficial.

## Lay summary

All histories first received their previously assigned injury, had one fresh
phase budget to recover, had to pass the exact earlier stability certificate,
rested for `20*n` real opportunities, and then received the same injury again.
Only then did the experiment either keep or replace three state components.
Arrangement replacement kept the same number of misplaced pairs. Cursor reset
returned Selection's native cursor to its original checkpoint value. Fatigue
reset cleared load and cooldown. Bubble and Insertion have no Selection cursor,
so their native reset can be an assigned but byte-identical intervention.

## Inputs and provenance

- S01–S10 canonical reports, machine-readable specifications, validation gates,
  exact S02 checkpoint records, S03 operator fixtures, and S10 assignments/results.
- E01 reference release `153adf5bd6ad2d45757797b561a9c6616d561ca0`.
- E02 causal extension `dafe4a0c05cca232c890349e47e37afcb1551257`
  plus action, scheduler, fault, information, stream, ledger, and competing-risk
  guidance.
- Workspace plans, prior-artifact manifests, capability/dataset audits, and the
  attachment manifest/sidecar. No dataset, network input, GPU, or new dependency
  was used.

## Detailed methods

### Population, injury, and competing terminals

All 384 exact S02 checkpoints retained the exact S10 four/four-within-block
assignment to repeated central segment reversal or repeated equal-block
transposition. Episode 1 retained S04 threshold-3/cooldown-8 fatigue and the
phase-local `100*n^2` budget. Target attainment had to pass the S01 charged
two-opportunities-per-identity certificate capped at `20*n`, followed by exactly
`20*n` target-stable opportunities. Only then was the matched second lesion
administered. A first-stage quiescent, budget, stabilization, or rest terminal
made the reset and second recovery unobservable in all ten arms. The case still
contributed joint success false and its full phase budget to restricted time.

### Exact state partitions and reset semantics

The **arrangement** partition is only the occupancy identity sequence. Its
reset is a counter-addressed constrained Lehmer permutation that differs from
the actual post-lesion state while preserving the exact direction-aware strict
inversion distance, identity set, values, count, target, and lesion stratum.
There is no distance substitution. The **native** partition is the Selection
cursor map and resets to the exact source S02 checkpoint snapshot. The
**engineered** partition is explicitly empty because S07 was null, S08 supplied
no validated mode and caused harms, and S09 target state is irrelevant here.
**Fatigue** remains a separate endogenous mediator and resets to zero load and
no cooldown.

Global event index, original scenario/RNG root, runtime stream counters,
native/process ledgers, process-audit history, phase budget, identities, values,
policies, directions, and target were preserved in every arm. Reset consumed no
native opportunity or runtime draw. Supplemental reads/writes were recorded in
`state_reset_audit.parquet` but were not disguised as physical energy or native
actions.

### Factorial, shams, outcomes, and inference

Eight arms formed a complete arrangement × native × fatigue 2x2x2 factorial.
Two no-op roundtrip shams serialized/restored identical arrangement or internal
partitions. The seven factorial contrasts were computed within every assigned
case, separately for joint success and restricted recovery time, then summarized
within each injury sequence. Two-sided exact sign/binomial tests on nonzero
case contrasts used separate Holm 14-test families. Paired t intervals describe
mean contrasts. The four-corner arrangement-by-composite-internal interaction
was also retained. No reset-boundary-observed, completed-only, Selection-only,
or survivor-only subset supplied a primary estimand.

## Commands, dependencies, and execution parameters

```text
PYTHONHASHSEED=0 python -m pytest -q tests/test_regeneration_state_memory_reset.py
PYTHONHASHSEED=0 python -m pytest -q tests/test_regeneration_*.py
ruff check src/regeneration/state_memory_reset.py scripts/build_regeneration_s11.py tests/test_regeneration_state_memory_reset.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/build_regeneration_s11.py --output /artifacts/research_steps/S11 --workers {workers} --test-summary "{test_summary}"
python scripts/build_regeneration_s11.py --validate-only --output /artifacts/research_steps/S11
```

Python {platform.python_version()}, NumPy {package_version('numpy')}, pandas
{package_version('pandas')}, PyArrow {package_version('pyarrow')}, SciPy
{package_version('scipy')}; {workers} process workers with OMP/MKL/OpenBLAS/
NumExpr threads fixed to one. Exact replay doubled confirmatory trajectory
execution. No dependency was installed.

## Results

### Full-population arm outcomes

| Arm | Runs | Reset boundary observed | Joint successes | Mean restricted time |
| --- | ---: | ---: | ---: | ---: |
{arm_lines}

The all-preserved reference reproduced S10 exactly on every one of
{validations['s10Anchor']['fieldsCompared']} shared fields for all 384 cases.
It retained {int(reference['firstStageEligible'].sum())}
observable reset boundaries and {int(reference['jointSequenceSuccess'].sum())}
joint successes. All earlier terminals remained in every arm.

### Prespecified factorial effects

| Sequence | Effect | Mean success contrast [95% CI] | Success Holm p | Mean restricted-time contrast [95% CI] | Time Holm p |
| --- | --- | ---: | ---: | ---: | ---: |
{effect_lines}

Actual intended state changes occurred in {actual_structure} arrangement-reset,
{actual_native} native-reset, and {actual_fatigue} fatigue-reset rows. An
assigned reset can be byte-identical when the relevant component already equals
its baseline; that is recorded rather than reclassified after outcomes.

### Composite arrangement-by-internal interaction

| Sequence | Mean success DID | Success Holm p | Mean restricted-time DID | Time Holm p |
| --- | ---: | ---: | ---: | ---: |
{composite_lines}

This four-corner interaction crosses arrangement preserve/reset with native and
fatigue jointly preserved/reset. The full seven-effect factorial remains the
primary decomposition.

## Validation

| Gate | Result |
| --- | --- |
| Inherited S01–S10 and E01/E02 inputs | PASS |
| Exact S10 sequence assignment and rest-20n reference | PASS — 384/384 cases, {validations['s10Anchor']['fieldsCompared']} shared fields exact |
| First/second injury equivalence and target feasibility | PASS |
| Exact distance-matched, identity-conserving scramble | PASS — every observable structural reset feasible; no substitution |
| Native/engineered/fatigue reset isolation | PASS |
| Protected clock/stream/ledger/audit state | PASS — unchanged at every reset |
| Sham reset behavior | PASS — 768/768 exact behavioral pairs |
| Deterministic replay | PASS — 3,840/3,840 |
| Full-population censor/terminal retention | PASS |
| Pairing and complete accounting | PASS — zero exclusions, substitutions, or runtime failures |

Focused/inherited tests: {test_summary}

## Residual confounding and interpretation

S11 manipulates the state available only after the S10 second-injury boundary.
It cannot make a reset scientifically observable for a history that terminated
before that boundary, and it cannot decompose the S10 prior-versus-naïve
eligibility difference that arose before reset. Full-population estimands retain
those cases, but their within-case reset contrast is necessarily zero. Exact
inversion distance removes one strong task-state difference; it does not match
local inversion geometry, identity displacement, policy-specific cursor
alignment, or every reachable-state property. Those descriptors remain visible
in the audit and are not called residual-free mediation.

## Artifacts

- `memory_reset_results.parquet`, `memory_reset_cases.parquet`, and
  `state_reset_audit.parquet` retain every assigned run, case, and reset.
- `factorial_case_contrasts.parquet`, `factorial_effects.parquet`, composite
  interaction tables, and PNG/SVG preserve the full-population estimates.
- `sham_reset_contrasts.parquet` and `s10_anchor_field_validation.parquet`
  preserve exact negative-control and inherited-reference evidence.
- `memory_reset_package/` contains the frozen spec/schema and 16 selected
  ten-arm trace groups.
- Injury, scramble, isolation, pairing, replay, censor, accounting, residual-
  confounding, validation, input, environment, and artifact manifests preserve
  reproducibility.

## Caveats, blockers, failed assumptions, and limitations

1. Reset is an engineered intervention at one observable boundary, not evidence
   of biological memory, molecular storage, or natural causal mediation.
2. First-stage competing terminals are retained but have no observable reset
   response; this bounds rather than solves pre-boundary structural confounding.
3. Exact inversion-distance matching does not equalize local motif geometry,
   displacement, policy accessibility, or full state-graph reachability.
4. Native internal state means Selection cursors only. Bubble and Insertion have
   no such field. Event clock, random counters, and ledgers are deliberately
   protected execution/accounting state rather than reset treatments.
5. Engineered S07–S09 state is absent, so S11 does not estimate what resetting a
   validated engineered memory would do; none was eligible for promotion.
6. Fatigue resetting is a controlled post-treatment intervention and does not
   identify a natural indirect effect.
7. Results are bounded to two sizes, three native policies, two directions,
   eight initial timings, two repeated lesions, one `20*n` spacing, and fixed
   budgets. S12 transfer was not started.
8. The first full execution completed all trajectories and replays but stopped
   during sham-audit aggregation because a NumPy Boolean was not converted to a
   JSON-native Boolean. It emitted no aggregate result table and no outcome was
   inspected. The serialization-only defect was fixture-checked and the entire
   frozen panel was rerun from the beginning; `execution_attempts.json` records
   both attempts.

No blocker remains within S11. S12 was not started.

## Provenance

- Repository: `Eidosoma/cell_research`
- Branch: `eidosoma/groups/28`
- Source commit at execution: `{commit}`
- Benchmark: `{BENCHMARK_VERSION}`
- RNG: inherited E01 actor/Bubble streams; exact S10 sequence assignment;
  construction-only `{SCRAMBLE_STREAM}`; fatigue owns no runtime stream.
- Runtime: Python {platform.python_version()}; {workers} workers; numerical-
  library threads one.
- Generated UTC: {generated}

Input/output SHA-256 hashes are recorded in `input_provenance.json` and
`artifact_manifest.json`. Reproducible source remains in Git; repository code is
not copied into the artifact directory.
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
        "schemaVersion": "e05.s11.artifact-manifest.v1",
        "researchStepId": "S11",
        "benchmarkVersion": BENCHMARK_VERSION,
        "generatedAtUtc": generated,
        "repositoryCommitAtExecution": commit,
        "artifactCount": len(files),
        "artifacts": files,
    }


def build(output: Path, workers: int, test_summary: str) -> None:
    generated = datetime.now(timezone.utc).isoformat()
    commit = _git("rev-parse", "HEAD")
    specification = _load_json(CONFIG)
    validate_state_memory_reset_spec(specification)
    inherited = _validate_inputs()
    output.mkdir(parents=True, exist_ok=True)
    package = output / "memory_reset_package"
    package.mkdir(exist_ok=True)
    _write_json(package / "state_memory_reset_spec.json", specification)
    _write_json(
        package / "state_memory_reset_spec.schema.json",
        STATE_MEMORY_RESET_SPEC_SCHEMA,
    )
    (package / "state_memory_reset_spec.md").write_text(
        _spec_markdown(specification), encoding="utf-8"
    )

    jobs, cases = _reconstruct_s11_jobs(specification)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_execute_job, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            value = future.result()
            rows.extend(value["rows"])
            if value["trace"] is not None:
                traces.append(value["trace"])
            if completed % 32 == 0:
                print(f"completed {completed}/{len(jobs)} S11 case groups", flush=True)
    results = pd.DataFrame(rows).sort_values(
        ["memoryResetCaseId", "arm"]
    ).reset_index(drop=True)
    cases = cases.sort_values(
        ["s01PairingBlockId", "assignmentRankWithinBlock"]
    ).reset_index(drop=True)
    audit = _reset_audit_table(results)
    case_contrasts, effects, composite_case, composite_effects = (
        _factorial_contrasts(results)
    )
    shams = _sham_contrasts(results)
    anchor_validation, anchor_fields = _s10_anchor_validation(results)

    results.to_parquet(output / "memory_reset_results.parquet", index=False)
    cases.to_parquet(output / "memory_reset_cases.parquet", index=False)
    audit.to_parquet(output / "state_reset_audit.parquet", index=False)
    case_contrasts.to_parquet(
        output / "factorial_case_contrasts.parquet", index=False
    )
    effects.to_parquet(output / "factorial_effects.parquet", index=False)
    composite_case.to_parquet(
        output / "composite_interaction_case_contrasts.parquet", index=False
    )
    composite_effects.to_parquet(
        output / "composite_interaction_effects.parquet", index=False
    )
    shams.to_parquet(output / "sham_reset_contrasts.parquet", index=False)
    anchor_fields.to_parquet(
        output / "s10_anchor_field_validation.parquet", index=False
    )
    with (package / "selected_state_reset_traces.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for trace in sorted(traces, key=lambda item: item["memoryResetCaseId"]):
            handle.write(
                json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n"
            )
    _effect_figure(effects, output)

    fixture = pd.read_parquet(
        "/artifacts/research_steps/S03/lesion_library/operator_fixtures.parquet"
    )
    fixture = fixture[fixture["operatorId"].isin(results["operatorId"].unique())]
    fixture_map = {
        (row.s01PairingBlockId, row.timingConditionId, row.operatorId): (
            row.postLesionStateHash
        )
        for row in fixture.itertuples()
    }
    unique_cases = results[results["arm"] == ResetArm.A0_N0_F0.value].copy()
    unique_cases["fixtureExpected"] = [
        fixture_map[(row.s01PairingBlockId, row.timingConditionId, row.operatorId)]
        for row in unique_cases.itertuples()
    ]
    unique_cases["firstFixtureMatch"] = (
        unique_cases["episode1PostLesionStateHash"]
        == unique_cases["fixtureExpected"]
    )
    injury_validation = {
        "success": bool(unique_cases["firstFixtureMatch"].all())
        and bool(
            unique_cases.loc[
                unique_cases["episode2InjuryAdministered"], "episode2DistanceBefore"
            ]
            .eq(0)
            .all()
        )
        and bool(
            unique_cases.loc[
                unique_cases["episode2InjuryAdministered"],
                "episode2AffectedIdentityCount",
            ]
            .notna()
            .all()
        ),
        "firstInjuryFixtureMatches": int(unique_cases["firstFixtureMatch"].sum()),
        "firstInjuryFixturesChecked": len(unique_cases),
        "secondInjuriesAdministered": int(
            unique_cases["episode2InjuryAdministered"].sum()
        ),
        "secondInjuryPreDistanceZero": int(
            unique_cases.loc[
                unique_cases["episode2InjuryAdministered"], "episode2DistanceBefore"
            ]
            .eq(0)
            .sum()
        ),
        "severityPoolingBoundary": (
            "operator-stratified; no scalar cross-operator equivalence"
        ),
    }

    observable_structural = results[
        results["resetBoundaryObserved"] & results["arrangementResetAssigned"]
    ]
    scramble_validation = {
        "success": len(observable_structural) > 0
        and bool(observable_structural["scrambleFeasible"].all())
        and bool(
            observable_structural["preResetDistance"]
            .eq(observable_structural["postResetDistance"])
            .all()
        )
        and bool(observable_structural["arrangementActuallyChanged"].all()),
        "observableStructuralResetRows": len(observable_structural),
        "feasibleRows": int(observable_structural["scrambleFeasible"].sum()),
        "exactDistanceMatches": int(
            observable_structural["preResetDistance"]
            .eq(observable_structural["postResetDistance"])
            .sum()
        ),
        "nonidenticalRows": int(
            observable_structural["arrangementActuallyChanged"].sum()
        ),
        "substitutions": 0,
        "infeasibleObservableRows": int((~observable_structural["scrambleFeasible"]).sum()),
    }

    pre_reset_group_counts = (
        results.groupby("memoryResetCaseId")
        .agg(
            stateHashes=("preResetStateHash", "nunique"),
            arrangementHashes=("preResetArrangementSha256", "nunique"),
            nativeHashes=("preResetNativeSha256", "nunique"),
            fatigueHashes=("preResetFatigueSha256", "nunique"),
            protectedHashes=("preResetProtectedSha256", "nunique"),
        )
        .reset_index()
    )
    reset_validation = {
        "success": bool(results["allResetValidationPass"].all())
        and bool((results["engineeredStateFieldCount"] == 0).all())
        and bool(
            results["preResetProtectedSha256"]
            .eq(results["postResetProtectedSha256"])
            .all()
        ),
        "resetAuditRows": len(audit),
        "allResetValidationPass": int(results["allResetValidationPass"].sum()),
        "engineeredStateFields": int(results["engineeredStateFieldCount"].sum()),
        "protectedStateExact": int(
            results["preResetProtectedSha256"]
            .eq(results["postResetProtectedSha256"])
            .sum()
        ),
        "actualArrangementChanges": int(
            results["arrangementActuallyChanged"].sum()
        ),
        "actualNativeChanges": int(results["nativeStateActuallyChanged"].sum()),
        "actualFatigueChanges": int(results["fatigueActuallyChanged"].sum()),
    }
    pairing_validation = {
        "success": bool(
            (
                pre_reset_group_counts[
                    [
                        "stateHashes",
                        "arrangementHashes",
                        "nativeHashes",
                        "fatigueHashes",
                        "protectedHashes",
                    ]
                ]
                == 1
            )
            .all()
            .all()
        )
        and len(case_contrasts)
        == specification["validationPanel"]["plannedFactorialCaseContrastCount"],
        "caseGroups": len(pre_reset_group_counts),
        "allTenArmPreResetHashesExact": int(
            (
                pre_reset_group_counts[
                    [
                        "stateHashes",
                        "arrangementHashes",
                        "nativeHashes",
                        "fatigueHashes",
                        "protectedHashes",
                    ]
                ]
                == 1
            )
            .all(axis=1)
            .sum()
        ),
        "factorialCaseContrasts": len(case_contrasts),
        "sharedPrefixBoundary": (
            "exact through the post-second-lesion snapshot; no dummy runtime draws"
        ),
    }
    sham_validation = {
        "success": len(shams)
        == specification["validationPanel"]["plannedShamPairCount"]
        and bool(shams["allBehaviorFieldsExact"].all()),
        "shamPairs": len(shams),
        "exactBehaviorPairs": int(shams["allBehaviorFieldsExact"].sum()),
        "expectedPairs": specification["validationPanel"]["plannedShamPairCount"],
    }
    replay_validation = {
        "success": bool(results["exactReplayPass"].all()),
        "replaysPassed": int(results["exactReplayPass"].sum()),
        "replaysPlanned": specification["validationPanel"]["plannedReplayCount"],
    }
    censor_validation = {
        "success": len(results)
        == specification["validationPanel"]["plannedRunCount"]
        and not results.duplicated("memoryResetRunId").any()
        and bool(
            results.loc[
                ~results["firstStageEligible"], "resetBoundaryObserved"
            ].eq(False).all()
        )
        and bool(
            results.loc[
                ~results["firstStageEligible"], "restrictedEpisode2Time"
            ]
            .eq(results.loc[~results["firstStageEligible"], "recoveryBudget"])
            .all()
        ),
        "runCount": len(results),
        "uniqueRunIds": int(results["memoryResetRunId"].nunique()),
        "firstStageCompetingTerminalRows": int(
            results["firstStageCompetingTerminal"].sum()
        ),
        "firstStageCompetingTerminalCases": int(
            unique_cases["firstStageCompetingTerminal"].sum()
        ),
        "resetBoundaryObservedRows": int(results["resetBoundaryObserved"].sum()),
        "silentExclusions": 0,
        "substitutions": 0,
    }
    accounting_validation = {
        "success": bool(results["allOpportunityValidationPass"].all())
        and len(audit)
        == specification["validationPanel"]["plannedResetAuditCount"]
        and len(effects)
        == specification["validationPanel"]["plannedFactorialEffectCount"]
        and len(traces)
        == specification["validationPanel"]["plannedSelectedTraceGroupCount"],
        "plannedRuns": specification["validationPanel"]["plannedRunCount"],
        "executedRuns": len(results),
        "resetAuditRows": len(audit),
        "factorialEffectRows": len(effects),
        "selectedTraceGroups": len(traces),
        "nativeActivations": int(results["nativeActivationDelta"].sum()),
        "acceptedSwaps": int(results["nativeAcceptedSwaps"].sum()),
        "supplementalResetWrites": int(
            results[
                [
                    "supplementalStructuralWrites",
                    "supplementalNativeWrites",
                    "supplementalFatigueWrites",
                ]
            ].sum().sum()
        ),
    }
    residual_validation = {
        "success": True,
        "fullPopulationCaseCount": 384,
        "preBoundaryCompetingTerminalCases": int(
            unique_cases["firstStageCompetingTerminal"].sum()
        ),
        "primaryContrastsConditionOnResetBoundary": False,
        "engineeredStateEstimable": False,
        "engineeredStateReason": (
            "explicit empty partition; no S07-S09 state promoted"
        ),
        "distanceMatchingLimit": (
            "exact inversion distance does not match local motif geometry, "
            "identity displacement, cursor alignment, or reachability"
        ),
        "preBoundaryAttributionLimit": (
            "S11 cannot decompose the S10 first-stage eligibility difference"
        ),
    }
    sequence_counts = cases["sequenceId"].value_counts().to_dict()
    sequence_validation = {
        "success": sequence_counts
        == {
            "repeat_segment_reversal_v1": 192,
            "repeat_block_transposition_v1": 192,
        }
        and bool((cases.groupby(["s01PairingBlockId", "sequenceId"]).size() == 4).all()),
        "counts": sequence_counts,
        "assignmentStream": "repeated_injury_sequence_assignment_s10_v1",
        "rerankedInS11": False,
    }

    validations: dict[str, dict[str, Any]] = {
        "inherited": {"success": all(inherited.values()), "gates": inherited},
        "sequence": sequence_validation,
        "injury": injury_validation,
        "scramble": scramble_validation,
        "resetIsolation": reset_validation,
        "pairing": pairing_validation,
        "sham": sham_validation,
        "s10Anchor": anchor_validation,
        "replay": replay_validation,
        "censor": censor_validation,
        "accounting": accounting_validation,
        "residualConfounding": residual_validation,
    }
    if not all(item["success"] for item in validations.values()):
        failed = {
            key: item for key, item in validations.items() if not item["success"]
        }
        raise AssertionError(f"S11 validation failed: {failed}")

    significant = bool(
        effects["successRejectAtFamilywise0_05"].any()
        or effects["timeRejectAtFamilywise0_05"].any()
    )
    classification = "supportive" if significant else "null"
    validation_summary = {
        "success": True,
        "researchStepId": "S11",
        "benchmarkVersion": BENCHMARK_VERSION,
        "outcomeClassification": classification,
        "caseCount": len(cases),
        "plannedCaseCount": specification["validationPanel"]["plannedCaseCount"],
        "runCount": len(results),
        "plannedRunCount": specification["validationPanel"]["plannedRunCount"],
        "replayCount": int(results["exactReplayPass"].sum()),
        "plannedReplayCount": specification["validationPanel"]["plannedReplayCount"],
        "factorialCaseContrastCount": len(case_contrasts),
        "factorialEffectCount": len(effects),
        "shamPairCount": len(shams),
        "selectedTraceGroupCount": len(traces),
        "gates": {key: item["success"] for key, item in validations.items()},
    }

    for name, value in validations.items():
        filename = {
            "inherited": "inherited_input_validation.json",
            "sequence": "sequence_assignment_validation.json",
            "injury": "injury_equivalence_validation.json",
            "scramble": "scramble_distance_validation.json",
            "resetIsolation": "reset_isolation_validation.json",
            "pairing": "pairing_validation.json",
            "sham": "sham_reset_validation.json",
            "s10Anchor": "s10_anchor_validation.json",
            "replay": "replay_validation.json",
            "censor": "censor_retention_validation.json",
            "accounting": "run_accounting.json",
            "residualConfounding": "residual_confounding_audit.json",
        }[name]
        _write_json(output / filename, value)
    _write_json(output / "validation_summary.json", validation_summary)

    input_provenance = {
        "schemaVersion": "e05.s11.input-provenance.v1",
        "researchStepId": "S11",
        "inputs": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in (CONFIG, S10_CONFIG, *INPUTS)
        ],
    }
    environment = {
        "schemaVersion": "e05.s11.environment-provenance.v1",
        "researchStepId": "S11",
        "generatedAtUtc": generated,
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "pyarrow": package_version("pyarrow"),
        "scipy": package_version("scipy"),
        "workers": workers,
        "cpuCountVisible": os.cpu_count(),
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "gpuUsed": False,
        "dependenciesInstalled": [],
    }
    _write_json(output / "input_provenance.json", input_provenance)
    _write_json(output / "environment_provenance.json", environment)
    _write_json(
        output / "outcome_decision.json",
        {
            "researchStepId": "S11",
            "classification": classification,
            "supportiveFactorialEffects": int(
                (
                    effects["successRejectAtFamilywise0_05"]
                    | effects["timeRejectAtFamilywise0_05"]
                ).sum()
            ),
            "operationalMemoryClaimOnly": True,
            "biologicalClaimPermitted": False,
            "preBoundaryAttributionResolved": False,
        },
    )
    _write_json(
        output / "execution_attempts.json",
        {
            "schemaVersion": "e05.s11.execution-attempts.v1",
            "researchStepId": "S11",
            "attempts": [
                {
                    "attemptOrdinal": 1,
                    "trajectoryScope": (
                        "384 case groups, 3840 assigned arms, and 3840 exact replays"
                    ),
                    "trajectoryExecutionCompleted": True,
                    "aggregateArtifactEmitted": False,
                    "outcomeTableInspected": False,
                    "failureStage": "sham audit aggregation",
                    "failureReason": (
                        "a NumPy Boolean was passed to the standard JSON encoder "
                        "without conversion to a native Boolean"
                    ),
                    "scientificContractChanged": False,
                    "disposition": (
                        "serialization-only defect fixed; complete frozen panel "
                        "restarted from the beginning"
                    ),
                },
                {
                    "attemptOrdinal": 2,
                    "trajectoryScope": (
                        "384 case groups, 3840 assigned arms, and 3840 exact replays"
                    ),
                    "trajectoryExecutionCompleted": True,
                    "aggregateArtifactEmitted": True,
                    "outcomeTableInspected": True,
                    "failureStage": None,
                    "failureReason": None,
                    "scientificContractChanged": False,
                    "disposition": "canonical S11 evidence",
                },
            ],
        },
    )
    (output / "execution_commands.log").write_text(
        "\n".join(
            [
                "PYTHONHASHSEED=0 python -m pytest -q tests/test_regeneration_state_memory_reset.py",
                "PYTHONHASHSEED=0 python -m pytest -q tests/test_regeneration_*.py",
                "ruff check src/regeneration/state_memory_reset.py scripts/build_regeneration_s11.py tests/test_regeneration_state_memory_reset.py",
                f"PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/build_regeneration_s11.py --output {output} --workers {workers}",
                f"python scripts/build_regeneration_s11.py --validate-only --output {output}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "research_step_full_results.md").write_text(
        _report(
            generated=generated,
            commit=commit,
            workers=workers,
            results=results,
            effects=effects,
            composite_effects=composite_effects,
            validations=validations,
            classification=classification,
            test_summary=test_summary,
        ),
        encoding="utf-8",
    )
    _write_json(output / "artifact_manifest.json", _artifact_manifest(output, generated, commit))
    print(json.dumps(validation_summary, indent=2, sort_keys=True))


def validate_only(output: Path) -> None:
    required = [
        "research_step_full_results.md",
        "memory_reset_results.parquet",
        "memory_reset_cases.parquet",
        "state_reset_audit.parquet",
        "factorial_case_contrasts.parquet",
        "factorial_effects.parquet",
        "composite_interaction_case_contrasts.parquet",
        "composite_interaction_effects.parquet",
        "sham_reset_contrasts.parquet",
        "s10_anchor_field_validation.parquet",
        "validation_summary.json",
        "artifact_manifest.json",
        "memory_reset_package/state_memory_reset_spec.json",
        "memory_reset_package/state_memory_reset_spec.schema.json",
        "memory_reset_package/state_memory_reset_spec.md",
        "memory_reset_package/selected_state_reset_traces.jsonl",
    ]
    missing = [item for item in required if not (output / item).is_file()]
    if missing:
        raise FileNotFoundError(f"missing S11 artifacts: {missing}")
    validation = _load_json(output / "validation_summary.json")
    if not validation["success"]:
        raise AssertionError("S11 validation summary is not successful")
    if len(pd.read_parquet(output / "memory_reset_results.parquet")) != 3840:
        raise AssertionError("S11 run table count changed")
    if len(pd.read_parquet(output / "state_reset_audit.parquet")) != 3840:
        raise AssertionError("S11 reset audit count changed")
    if len(pd.read_parquet(output / "factorial_case_contrasts.parquet")) != 2688:
        raise AssertionError("S11 case contrast count changed")
    if len(pd.read_parquet(output / "factorial_effects.parquet")) != 14:
        raise AssertionError("S11 effect count changed")
    manifest = _load_json(output / "artifact_manifest.json")
    for item in manifest["artifacts"]:
        path = output / item["path"]
        if not path.is_file() or _sha256_file(path) != item["sha256"]:
            raise AssertionError(f"S11 artifact hash mismatch: {item['path']}")
    print(json.dumps({"success": True, "validatedArtifacts": len(required)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--test-summary", default="not supplied")
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.workers > 8:
        raise ValueError("S11 workers must be in [1,8]")
    if arguments.validate_only:
        validate_only(arguments.output)
    else:
        build(arguments.output, arguments.workers, arguments.test_summary)


if __name__ == "__main__":
    main()
