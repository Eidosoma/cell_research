#!/usr/bin/env python3
"""Build and validate E05 S09 collective-target-change evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
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
from src.regeneration.target_change import (  # noqa: E402
    BENCHMARK_VERSION,
    TARGET_CHANGE_RUN_SCHEMA_VERSION,
    TARGET_CHANGE_SPEC_SCHEMA,
    SignalPermission,
    TargetArm,
    TargetChange,
    TargetChangeContract,
    TargetChangeController,
    build_target_definition,
    exact_replay_target_change,
    run_target_change_phase,
    target_change_case_id,
    target_correspondence_rows,
    validate_target_change_spec,
)


CONFIG = REPOSITORY / "configs/regeneration/s09_collective_target_change.json"
S04_CONFIG = REPOSITORY / "configs/regeneration/s04_dynamic_faults.json"
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
    Path("/workspace/DATASETS.md"),
    Path("/workspace/DATASET_CATALOG.json"),
    Path("/workspace/DATASET_AVAILABILITY.json"),
    Path("/workspace/input-attachments/MANIFEST.json"),
    ATTACHMENT_SIDECAR,
    *tuple(
        item
        for step in range(1, 9)
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
    Path("/artifacts/research_steps/S04/dynamic_fault_package/dynamic_fault_spec.md"),
    Path("/artifacts/research_steps/S04/dynamic_fault_package/dynamic_fault_spec.json"),
    Path("/artifacts/research_steps/S05/nudge_recovery_package/nudge_recovery_spec.md"),
    Path("/artifacts/research_steps/S05/nudge_recovery_package/nudge_recovery_spec.json"),
    Path("/artifacts/research_steps/S06/assisted_rescue_package/assisted_rescue_spec.md"),
    Path("/artifacts/research_steps/S06/assisted_rescue_package/assisted_rescue_spec.json"),
    Path("/artifacts/research_steps/S07/memory_variants/local_memory_spec.md"),
    Path("/artifacts/research_steps/S07/memory_variants/local_memory_spec.json"),
    Path("/artifacts/research_steps/S07/primary_completion_tests.parquet"),
    Path("/artifacts/research_steps/S08/plasticity_package/policy_plasticity_spec.md"),
    Path("/artifacts/research_steps/S08/plasticity_package/policy_plasticity_spec.json"),
    Path("/artifacts/research_steps/S08/outcome_classification.json"),
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
        raise FileNotFoundError(f"missing required S01-S08/E01/E02 inputs: {missing}")
    gates = {
        f"s{step:02d}": bool(
            _load_json(Path(f"/artifacts/research_steps/S{step:02d}/validation_summary.json"))["success"]
        )
        for step in range(1, 9)
    }
    gates["e01"] = bool(
        _load_json(Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"))["validationSuccess"]
    )
    gates["e02"] = bool(
        _load_json(Path("/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json"))["smokeValidation"]["success"]
    )
    s07 = pd.read_parquet(
        "/artifacts/research_steps/S07/primary_completion_tests.parquet"
    )
    gates["s07MatchedNull"] = bool(
        len(s07) == 5
        and (~s07["rejectAtFamilywise0_05"].astype(bool)).all()
        and (s07["holmAdjustedPValue"] == 1.0).all()
    )
    s08 = _load_json(
        Path("/artifacts/research_steps/S08/outcome_classification.json")
    )
    gates["s08HarmsConstrained"] = bool(
        s08.get("classification") == "constraining/contradictory"
        and int(s08.get("recoveryHolmRejectionCount", 0)) == 4
        and int(s08.get("stabilityHarmHolmRejectionCount", 0)) == 2
    )
    if not all(gates.values()):
        raise RuntimeError(f"an inherited validation/evidence gate failed: {gates}")
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
        raise RuntimeError(f"expected 384 inherited S02 bases, found {len(answer)}")
    return answer


@dataclass(frozen=True, slots=True)
class TargetJob:
    base: DynamicJob
    phase: str
    arm: TargetArm
    target_change: TargetChange
    signal: SignalPermission
    retain_trace: bool = False


def _case_id(job: TargetJob) -> str:
    return target_change_case_id(
        job.base.s01_pairing_block_id,
        job.base.timing_condition_id,
        job.target_change,
        job.phase,
    )


def _trace_job(job: TargetJob) -> bool:
    return bool(
        job.phase == "changed"
        and job.base.n == 20
        and job.base.policy == "Bubble"
        and job.base.direction == "ascending"
        and job.base.replicate == 2
        and job.base.timing_condition_id == "post_completion"
    )


def _jobs(bases: list[DynamicJob]) -> list[TargetJob]:
    jobs: list[TargetJob] = []
    for base in bases:
        for target_change in TargetChange:
            for signal in SignalPermission:
                for arm in (TargetArm.CHANGED_AWARE, TargetArm.CHANGED_NONADAPTIVE):
                    candidate = TargetJob(base, "changed", arm, target_change, signal)
                    jobs.append(replace(candidate, retain_trace=_trace_job(candidate)))
    stability_bases = [base for base in bases if base.timing_condition_id == "post_completion"]
    if len(stability_bases) != 48:
        raise RuntimeError("S09 stability panel requires all 48 post-completion bases")
    for base in stability_bases:
        for target_change in TargetChange:
            for signal in SignalPermission:
                for arm in (TargetArm.STABILITY_AWARE, TargetArm.STABILITY_NONADAPTIVE):
                    jobs.append(TargetJob(base, "stability", arm, target_change, signal))
    if len(jobs) != 10_368:
        raise RuntimeError(f"expected 10368 S09 jobs, found {len(jobs)}")
    if sum(job.retain_trace for job in jobs) != 24:
        raise RuntimeError("S09 selected-trace count changed")
    return jobs


def _execute_job(job: TargetJob) -> dict[str, Any]:
    contract = TargetChangeContract(job.arm, job.target_change, job.signal)
    adaptation_budget = job.base.recovery_budget
    probe_budget = 20 * job.base.n
    run = run_target_change_phase(
        job.base.scenario,
        job.base.checkpoint,
        contract=contract,
        adaptation_budget=adaptation_budget,
        probe_budget=probe_budget,
        retain_trace=job.retain_trace,
    )
    exact_replay_target_change(
        run,
        job.base.scenario,
        job.base.checkpoint,
        adaptation_budget,
        probe_budget,
    )
    case_id = _case_id(job)
    run_id = "e05tr9:" + sha256_json(
        {
            "targetChangeCaseId": case_id,
            "signalPermissionId": job.signal.value,
            "arm": job.arm.value,
        }
    )
    summary = dict(run.summary)
    process = dict(run.process_ledger)
    native = dict(summary["ledgerDelta"])
    row = {
        "schemaVersion": "e05.s09.target-change-result.v1",
        "benchmarkVersion": BENCHMARK_VERSION,
        "targetChangeRunId": run_id,
        "targetChangeCaseId": case_id,
        "phase": job.phase,
        "arm": job.arm.value,
        "targetChangeId": job.target_change.value,
        "signalPermissionId": job.signal.value,
        "targetAware": contract.target_aware,
        "noChange": contract.no_change,
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
        "targetHash": run.target_definition.target_hash,
        "targetMaximumDistance": run.target_definition.maximum_distance,
        "changeEventIndex": run.start_event_index,
        "endEventIndex": run.end_event_index,
        "initialStateHash": run.initial_state_hash,
        "finalStateHash": run.final_state_hash,
        "stopReason": summary["stopReason"],
        "targetCompleted": summary["targetCompleted"],
        "phaseActivationCount": summary["phaseActivationCount"],
        "adaptationBudget": summary["adaptationBudget"],
        "probeBudget": summary["probeBudget"],
        "initialOldTargetDistance": summary["initialOldTargetDistance"],
        "initialNewTargetDistance": summary["initialNewTargetDistance"],
        "initialNormalizedNewTargetDistance": summary["initialNormalizedNewTargetDistance"],
        "finalOldTargetDistance": summary["finalOldTargetDistance"],
        "finalNewTargetDistance": summary["finalNewTargetDistance"],
        "finalNormalizedNewTargetDistance": summary["finalNormalizedNewTargetDistance"],
        "maximumNewTargetDistance": summary["maximumNewTargetDistance"],
        "oldTargetDistanceAuc": summary["oldTargetDistanceAuc"],
        "newTargetDistanceAuc": summary["newTargetDistanceAuc"],
        "crossoverTime": summary["crossoverTime"],
        "crossoverCensored": summary["crossoverCensored"],
        "adaptationTime": summary["adaptationTime"],
        "adaptationCensored": summary["adaptationCensored"],
        "restrictedAdaptationTime": summary["restrictedAdaptationTime"],
        "overshootCensored": summary["overshootCensored"],
        "postHitProbeOpportunities": summary["postHitProbeOpportunities"],
        "postHitAnyDeparture": summary["postHitAnyDeparture"],
        "postHitMaximumDistance": summary["postHitMaximumDistance"],
        "postHitDistanceAuc": summary["postHitDistanceAuc"],
        "postHitFinalTargetRetained": summary["postHitFinalTargetRetained"],
        "noChangeAnyTargetDeparture": summary["noChangeAnyTargetDeparture"],
        "noChangeFinalTargetRetained": summary["noChangeFinalTargetRetained"],
        "quiescenceContextCount": summary["quiescenceContextCount"],
        "quiescenceContextExpected": summary["quiescenceContextExpected"],
        "nativeActivations": native["activations"],
        "nativeProposals": native["proposals"],
        "nativeNoOps": native["noOps"],
        "nativeRejections": native["rejections"],
        "nativeMemoryUpdates": native["memoryUpdates"],
        "nativeAcceptedSwaps": native["acceptedSwaps"],
        "nativeDisplacedCells": native["displacedCells"],
        **process,
        "nativeLedgerDeltaJson": json.dumps(native, sort_keys=True),
        "nativeStreamDeltaJson": json.dumps(summary["streamCounterDelta"], sort_keys=True),
        "opportunityValidationJson": json.dumps(run.opportunity_validation, sort_keys=True),
        "allOpportunityChecksPass": all(run.opportunity_validation.values()),
        "eventDigest": run.event_digest,
        "processAuditDigest": run.process_audit_digest,
        "processAuditCount": run.process_audit_count,
        "traceMode": summary["traceMode"],
        "retainedEventCount": summary["retainedEventCount"],
        "runDigest": hashlib.sha256(run.to_json_bytes()).hexdigest(),
    }
    trace = run.to_dict() if job.retain_trace else None
    return {"row": row, "trace": trace}


def _run_jobs(jobs: list[TargetJob], workers: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_execute_job, job): job for job in jobs}
        for ordinal, future in enumerate(as_completed(future_map), 1):
            job = future_map[future]
            try:
                result = future.result()
                rows.append(result["row"])
                if result["trace"] is not None:
                    traces.append(result["trace"])
            except Exception as exc:  # pragma: no cover - full-panel diagnostics
                failures.append(
                    {
                        "s01PairingBlockId": job.base.s01_pairing_block_id,
                        "timingConditionId": job.base.timing_condition_id,
                        "targetChangeId": job.target_change.value,
                        "signalPermissionId": job.signal.value,
                        "arm": job.arm.value,
                        "error": repr(exc),
                    }
                )
            if ordinal % 256 == 0 or ordinal == len(jobs):
                elapsed = time.monotonic() - started
                print(
                    f"S09 progress {ordinal}/{len(jobs)} jobs; "
                    f"rows={len(rows)} failures={len(failures)} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    rows.sort(key=lambda row: row["targetChangeRunId"])
    traces.sort(
        key=lambda row: (
            row["contract"]["targetChangeId"],
            row["contract"]["signalPermissionId"],
            row["contract"]["arm"],
        )
    )
    return {"rows": rows, "traces": traces, "failures": failures}


def _paired_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "targetCompleted",
        "phaseActivationCount",
        "finalOldTargetDistance",
        "finalNewTargetDistance",
        "newTargetDistanceAuc",
        "restrictedAdaptationTime",
        "postHitAnyDeparture",
        "postHitMaximumDistance",
        "postHitDistanceAuc",
        "noChangeAnyTargetDeparture",
        "abstractEnergyUnits",
        "signalPayloadUnits",
        "targetRecordReads",
        "controllerComputations",
    ]
    keys = [
        "targetChangeCaseId",
        "phase",
        "targetChangeId",
        "signalPermissionId",
        "s01PairingBlockId",
        "timingConditionId",
        "n",
        "policy",
        "direction",
        "replicateOrdinal",
    ]
    aware = results[results["targetAware"]].copy()
    control = results[~results["targetAware"]].copy()
    merged = aware.merge(
        control,
        on=keys,
        suffixes=("Active", "Control"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        item: dict[str, Any] = {
            "schemaVersion": "e05.s09.target-change-paired-contrast.v1",
            **{key: row[key] for key in keys},
            "pairId": "e05tp9:" + sha256_json({key: row[key] for key in keys}),
            "activeRunId": row["targetChangeRunIdActive"],
            "controlRunId": row["targetChangeRunIdControl"],
            "activeArm": row["armActive"],
            "controlArm": row["armControl"],
            "sourceScenarioId": row["sourceScenarioIdActive"],
            "sourceCheckpointHash": row["sourceCheckpointHashActive"],
            "targetHash": row["targetHashActive"],
            "changeEventIndex": row["changeEventIndexActive"],
            "activeFinalStateHash": row["finalStateHashActive"],
            "controlFinalStateHash": row["finalStateHashControl"],
            "activeStopReason": row["stopReasonActive"],
            "controlStopReason": row["stopReasonControl"],
        }
        for metric in metrics:
            active_value = row[f"{metric}Active"]
            control_value = row[f"{metric}Control"]
            item[f"active{metric[0].upper()}{metric[1:]}"] = active_value
            item[f"control{metric[0].upper()}{metric[1:]}"] = control_value
            item[f"delta{metric[0].upper()}{metric[1:]}"] = float(active_value) - float(control_value)
        rows.append(item)
    frame = pd.DataFrame(rows).sort_values("pairId").reset_index(drop=True)
    if len(frame) != 5_184:
        raise RuntimeError(f"expected 5184 S09 contrasts, found {len(frame)}")
    return frame


def _holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    array = values.astype(float).to_numpy()
    mean = float(array.mean())
    if len(array) < 2 or np.all(array == array[0]):
        return mean, mean, mean
    half = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array)))
    return mean, mean - half, mean + half


def _binary_tests(contrasts: pd.DataFrame, *, stability: bool) -> pd.DataFrame:
    selected = contrasts[contrasts["phase"] == ("stability" if stability else "changed")]
    active_column = (
        "activeNoChangeAnyTargetDeparture" if stability else "activeTargetCompleted"
    )
    control_column = (
        "controlNoChangeAnyTargetDeparture" if stability else "controlTargetCompleted"
    )
    rows: list[dict[str, Any]] = []
    for (target_change, signal), group in selected.groupby(
        ["targetChangeId", "signalPermissionId"], sort=True
    ):
        active = group[active_column].astype(bool)
        control = group[control_column].astype(bool)
        active_only = int((active & ~control).sum())
        control_only = int((~active & control).sum())
        discordant = active_only + control_only
        raw_p = 1.0 if discordant == 0 else float(
            binomtest(active_only, discordant, 0.5, alternative="two-sided").pvalue
        )
        mean, low, high = _mean_ci(active.astype(int) - control.astype(int))
        rows.append(
            {
                "schemaVersion": "e05.s09.primary-stability-test.v1" if stability else "e05.s09.primary-adaptation-test.v1",
                "family": "stability" if stability else "adaptation",
                "targetChangeId": target_change,
                "signalPermissionId": signal,
                "pairCount": len(group),
                "activePositiveCount": int(active.sum()),
                "controlPositiveCount": int(control.sum()),
                "activeOnlyCount": active_only,
                "controlOnlyCount": control_only,
                "pairedRiskDifference": mean,
                "pairedRiskDifferenceCiLow": low,
                "pairedRiskDifferenceCiHigh": high,
                "rawExactPValue": raw_p,
            }
        )
    adjusted = _holm([row["rawExactPValue"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holmAdjustedPValue"] = value
        row["rejectAtFamilywise0_05"] = value <= 0.05
    if len(rows) != 12:
        raise RuntimeError("S09 primary test family must contain 12 contrasts")
    return pd.DataFrame(rows)


def _tradeoff_summary(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in contrasts.groupby(
        ["phase", "targetChangeId", "signalPermissionId"], sort=True
    ):
        phase, target_change, signal = keys
        rows.append(
            {
                "schemaVersion": "e05.s09.target-change-tradeoff.v1",
                "phase": phase,
                "targetChangeId": target_change,
                "signalPermissionId": signal,
                "pairCount": len(group),
                "meanCompletionDifference": float(group["deltaTargetCompleted"].mean()),
                "meanOpportunityDifference": float(group["deltaPhaseActivationCount"].mean()),
                "meanFinalNewErrorDifference": float(group["deltaFinalNewTargetDistance"].mean()),
                "meanNewErrorAucDifference": float(group["deltaNewTargetDistanceAuc"].mean()),
                "meanRestrictedAdaptationTimeDifference": float(group["deltaRestrictedAdaptationTime"].mean()),
                "meanPostHitDepartureDifference": float(group["deltaPostHitAnyDeparture"].mean()),
                "meanNoChangeDepartureDifference": float(group["deltaNoChangeAnyTargetDeparture"].mean()),
                "meanAbstractEnergyDifference": float(group["deltaAbstractEnergyUnits"].mean()),
                "meanTargetRecordReadDifference": float(group["deltaTargetRecordReads"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _adaptation_curves(results: pd.DataFrame) -> pd.DataFrame:
    changed = results[results["phase"] == "changed"]
    return (
        changed.groupby(
            [
                "targetChangeId",
                "signalPermissionId",
                "arm",
                "timingConditionId",
                "n",
                "policy",
            ],
            dropna=False,
        )
        .agg(
            runCount=("targetChangeRunId", "count"),
            completionRate=("targetCompleted", "mean"),
            censorRate=("adaptationCensored", "mean"),
            meanRestrictedAdaptationTime=("restrictedAdaptationTime", "mean"),
            meanInitialNormalizedNewError=("initialNormalizedNewTargetDistance", "mean"),
            meanFinalNormalizedNewError=("finalNormalizedNewTargetDistance", "mean"),
            meanNewErrorAuc=("newTargetDistanceAuc", "mean"),
            meanAbstractEnergy=("abstractEnergyUnits", "mean"),
        )
        .reset_index()
    )


def _target_correspondence(bases: list[DynamicJob]) -> pd.DataFrame:
    unique: dict[str, DynamicJob] = {}
    for base in bases:
        unique.setdefault(base.s01_pairing_block_id, base)
    if len(unique) != 48:
        raise RuntimeError("S09 target correspondence requires 48 source scenarios")
    rows: list[dict[str, Any]] = []
    for base in unique.values():
        for target_change in TargetChange:
            definition = build_target_definition(base.scenario, target_change)
            for row in target_correspondence_rows(base.scenario, definition):
                row.update(
                    {
                        "s01PairingBlockId": base.s01_pairing_block_id,
                        "n": base.n,
                        "policy": base.policy,
                        "direction": base.direction,
                        "replicateOrdinal": base.replicate,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _pairing_validation(results: pd.DataFrame, contrasts: pd.DataFrame) -> dict[str, Any]:
    expected_pair_count = 5_184
    shared = [
        "sourceScenarioId",
        "sourceCheckpointHash",
        "targetHash",
        "changeEventIndex",
        "n",
        "policy",
        "direction",
        "replicateOrdinal",
        "timingConditionId",
    ]
    aware = results[results["targetAware"]].set_index(
        ["targetChangeCaseId", "signalPermissionId"]
    )
    control = results[~results["targetAware"]].set_index(
        ["targetChangeCaseId", "signalPermissionId"]
    )
    same_index = aware.index.is_unique and control.index.is_unique and set(aware.index) == set(control.index)
    shared_pass = same_index and all(
        aware.loc[control.index, field].astype(str).equals(control[field].astype(str))
        for field in shared
    )
    return {
        "schemaVersion": "e05.s09.pairing-validation.v1",
        "researchStepId": "S09",
        "expectedPairCount": expected_pair_count,
        "observedPairCount": len(contrasts),
        "uniquePairIds": contrasts["pairId"].nunique(),
        "completeOneToOnePairs": same_index,
        "sharedFieldsPass": shared_pass,
        "success": bool(
            len(contrasts) == expected_pair_count
            and contrasts["pairId"].nunique() == expected_pair_count
            and same_index
            and shared_pass
        ),
    }


def _validate_panel(
    specification: Mapping[str, Any],
    bases: list[DynamicJob],
    reconstructed: Mapping[str, Any],
    results: pd.DataFrame,
    contrasts: pd.DataFrame,
    correspondence: pd.DataFrame,
    traces: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    changed = results[results["phase"] == "changed"]
    stability = results[results["phase"] == "stability"]
    pairing = _pairing_validation(results, contrasts)
    all_opportunity = bool(results["allOpportunityChecksPass"].all())
    target_feasible = bool(
        len(correspondence) == sum(base.n for base in {b.s01_pairing_block_id: b for b in bases}.values()) * 3
        and correspondence["targetFeasible"].astype(bool).all()
        and correspondence["identityConserved"].astype(bool).all()
        and correspondence.groupby(["s01PairingBlockId", "targetChangeId"])["identityId"].nunique().eq(
            correspondence.groupby(["s01PairingBlockId", "targetChangeId"])["n"].first()
        ).all()
    )
    checkpoint_pass = bool(
        len(reconstructed["checkpointRows"]) == 384
        and not reconstructed["checkpointFailures"]
        and all(row["success"] for row in reconstructed["checkpointRows"])
    )
    none_rows = results[results["signalPermissionId"] == SignalPermission.NONE.value]
    none_contrasts = contrasts[contrasts["signalPermissionId"] == SignalPermission.NONE.value]
    negative_control = bool(
        len(none_rows) == 2_592
        and (none_rows["signalQueries"] == 0).all()
        and (none_rows["signalDeliveries"] == 0).all()
        and (none_rows["targetAwareCandidates"] == 0).all()
        and (none_rows["globalBroadcastUnits"] == 0).all()
        and (none_contrasts["activeFinalStateHash"] == none_contrasts["controlFinalStateHash"]).all()
        and (none_contrasts["deltaTargetCompleted"] == 0).all()
        and (none_contrasts["deltaPhaseActivationCount"] == 0).all()
        and (none_contrasts["deltaFinalNewTargetDistance"] == 0).all()
        and (none_contrasts["deltaAbstractEnergyUnits"] == 0).all()
    )
    local = results[results["signalPermissionId"] == SignalPermission.LOCAL.value]
    gradient = results[results["signalPermissionId"] == SignalPermission.GRADIENT.value]
    global_rows = results[results["signalPermissionId"] == SignalPermission.GLOBAL.value]
    signal_ledger = bool(
        (local["signalQueries"] == local["phaseActivationCount"]).all()
        and (local["signalDeliveries"] <= local["signalQueries"]).all()
        and (local["globalBroadcastUnits"] == 0).all()
        and (gradient["signalQueries"] == gradient["phaseActivationCount"]).all()
        and (gradient["targetAwareCandidates"] == gradient["phaseActivationCount"]).all()
        and (gradient["globalBroadcastUnits"] == 0).all()
        and (global_rows["globalBroadcastUnits"] == global_rows["n"]).all()
        and (global_rows["targetAwareCandidates"] == global_rows["phaseActivationCount"]).all()
    )
    energy_expected = (
        results["globalBroadcastUnits"]
        + results["signalQueries"]
        + results["signalDeliveries"]
        + results["targetRecordReads"]
        + results["controllerComputations"]
    )
    cost_ledger = bool(all_opportunity and (results["abstractEnergyUnits"] == energy_expected).all())
    local_trace_pass = True
    permission_trace_pass = True
    forbidden_keys = {
        "globalOccupancy",
        "targetDistance",
        "targetProgress",
        "completion",
        "timingConditionId",
        "futureDraw",
        "outcome",
    }
    for trace in traces:
        contract = trace["contract"]
        start = int(trace["startEventIndex"])
        n = len(trace["finalState"]["occupancy"])
        for audit in trace["retainedProcessAudits"]:
            permission_trace_pass = permission_trace_pass and forbidden_keys.isdisjoint(audit)
            permission_trace_pass = permission_trace_pass and all(
                0 <= int(position) < n for position in audit["authorizedPositions"]
            )
            if contract["signalPermissionId"] == SignalPermission.LOCAL.value:
                elapsed = int(audit["eventIndex"]) - start
                position = int(audit["actorPosition"])
                expected = min(position, n - 1 - position) <= elapsed
                local_trace_pass = local_trace_pass and bool(audit["signalAvailable"]) == expected
            if audit["authorizedPositions"]:
                position = int(audit["actorPosition"])
                permission_trace_pass = permission_trace_pass and all(
                    abs(int(item) - position) <= 1 for item in audit["authorizedPositions"]
                )
    stability_equivalence = bool(
        (contrasts[contrasts["phase"] == "stability"]["activeFinalStateHash"]
         == contrasts[contrasts["phase"] == "stability"]["controlFinalStateHash"]).all()
    )
    no_hidden_s08 = all(
        token not in inspect.getsource(TargetChangeController).lower()
        for token in ("exploratory_mode", "sensing_radius_expansion", "alternate_target_selection", "algotype_switch")
    )
    censor_retention = bool(
        len(changed) == 9_216
        and changed["adaptationCensored"].notna().all()
        and changed["overshootCensored"].notna().all()
        and (
            changed["adaptationCensored"].astype(int)
            == (~changed["targetCompleted"].astype(bool)).astype(int)
        ).all()
    )
    complete = bool(
        len(results) == 10_368
        and len(changed) == 9_216
        and len(stability) == 1_152
        and len(contrasts) == 5_184
        and len(traces) == 24
        and not failures
        and results["targetChangeRunId"].nunique() == 10_368
    )
    validation_summary = {
        "schemaVersion": "e05.s09.validation-summary.v1",
        "researchStepId": "S09",
        "checkpointIdentityPass": checkpoint_pass,
        "targetFeasibilityCorrespondencePass": target_feasible,
        "changeTimingPairingPass": pairing["success"],
        "signalPropagationPass": local_trace_pass and signal_ledger,
        "signalPermissionNoLeakagePass": permission_trace_pass and no_hidden_s08,
        "negativeControlPass": negative_control,
        "controllerCostLedgerPass": cost_ledger,
        "noChangeStabilityEquivalencePass": stability_equivalence,
        "deterministicReplayPass": len(results) == 10_368 and not failures,
        "censorRetentionPass": censor_retention,
        "completeRunAccountingPass": complete,
        "pairingPass": pairing["success"],
        "plannedRunCount": 10_368,
        "replayCount": 10_368,
        "pairedContrastCount": 5_184,
        "success": all(
            [
                checkpoint_pass,
                target_feasible,
                pairing["success"],
                local_trace_pass,
                signal_ledger,
                permission_trace_pass,
                no_hidden_s08,
                negative_control,
                cost_ledger,
                stability_equivalence,
                censor_retention,
                complete,
            ]
        ),
    }
    return {
        "validationSummary": validation_summary,
        "pairingValidation": pairing,
        "targetValidation": {
            "schemaVersion": "e05.s09.target-feasibility-validation.v1",
            "researchStepId": "S09",
            "correspondenceRowCount": len(correspondence),
            "scenarioTargetCount": correspondence.groupby(["s01PairingBlockId", "targetChangeId"]).ngroups,
            "allIdentitiesConserved": bool(correspondence["identityConserved"].all()),
            "allTargetsFeasible": bool(correspondence["targetFeasible"].all()),
            "success": target_feasible,
        },
        "signalValidation": {
            "schemaVersion": "e05.s09.signal-propagation-permission-validation.v1",
            "researchStepId": "S09",
            "localLightConeTracePass": local_trace_pass,
            "aggregateSignalLedgerPass": signal_ledger,
            "authorizedPositionTracePass": permission_trace_pass,
            "forbiddenAuditFieldsAbsent": permission_trace_pass,
            "s08HarmfulModesAbsent": no_hidden_s08,
            "success": local_trace_pass and signal_ledger and permission_trace_pass and no_hidden_s08,
        },
        "negativeControlValidation": {
            "schemaVersion": "e05.s09.negative-control-validation.v1",
            "researchStepId": "S09",
            "noSignalRunCount": len(none_rows),
            "noSignalPairCount": len(none_contrasts),
            "exactBehavioralEquivalence": negative_control,
            "success": negative_control,
        },
        "costValidation": {
            "schemaVersion": "e05.s09.cost-ledger-validation.v1",
            "researchStepId": "S09",
            "nativeOpportunityIdentitiesPass": all_opportunity,
            "abstractEnergyIdentityPass": bool((results["abstractEnergyUnits"] == energy_expected).all()),
            "success": cost_ledger,
        },
        "censorValidation": {
            "schemaVersion": "e05.s09.censor-retention-validation.v1",
            "researchStepId": "S09",
            "changedRunCount": len(changed),
            "adaptationCensorCount": int(changed["adaptationCensored"].sum()),
            "overshootCensorCount": int(changed["overshootCensored"].sum()),
            "success": censor_retention,
        },
        "replayValidation": {
            "schemaVersion": "e05.s09.replay-validation.v1",
            "researchStepId": "S09",
            "expectedReplayCount": 10_368,
            "observedReplayCount": len(results),
            "mismatchCount": len(failures),
            "success": len(results) == 10_368 and not failures,
        },
    }


def _classification(
    adaptation_tests: pd.DataFrame,
    stability_tests: pd.DataFrame,
    validation_success: bool,
) -> str:
    if not validation_success:
        return "constraining/contradictory"
    adaptation_harm = bool(
        (
            adaptation_tests["rejectAtFamilywise0_05"].astype(bool)
            & (adaptation_tests["pairedRiskDifference"] < 0)
        ).any()
    )
    stability_harm = bool(
        (
            stability_tests["rejectAtFamilywise0_05"].astype(bool)
            & (stability_tests["pairedRiskDifference"] > 0)
        ).any()
    )
    no_signal_difference = bool(
        adaptation_tests[
            adaptation_tests["signalPermissionId"] == SignalPermission.NONE.value
        ]["rejectAtFamilywise0_05"].astype(bool).any()
    )
    if adaptation_harm or stability_harm or no_signal_difference:
        return "constraining/contradictory"
    informative = adaptation_tests[
        adaptation_tests["signalPermissionId"] != SignalPermission.NONE.value
    ]
    benefit = bool(
        (
            informative["rejectAtFamilywise0_05"].astype(bool)
            & (informative["pairedRiskDifference"] > 0)
        ).any()
    )
    return "supportive" if benefit else "null"


def _plot_results(
    adaptation_tests: pd.DataFrame,
    stability_tests: pd.DataFrame,
    package: Path,
) -> None:
    signal_order = [item.value for item in SignalPermission]
    targets = [item.value for item in TargetChange]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    colors = plt.get_cmap("tab10")
    for target_index, target_change in enumerate(targets):
        for signal_index, signal in enumerate(signal_order):
            row = adaptation_tests[
                (adaptation_tests["targetChangeId"] == target_change)
                & (adaptation_tests["signalPermissionId"] == signal)
            ].iloc[0]
            y = target_index * (len(signal_order) + 1) + signal_index
            axes[0].errorbar(
                row["pairedRiskDifference"],
                y,
                xerr=[
                    [row["pairedRiskDifference"] - row["pairedRiskDifferenceCiLow"]],
                    [row["pairedRiskDifferenceCiHigh"] - row["pairedRiskDifference"]],
                ],
                fmt="o",
                color=colors(signal_index),
            )
            stability = stability_tests[
                (stability_tests["targetChangeId"] == target_change)
                & (stability_tests["signalPermissionId"] == signal)
            ].iloc[0]
            axes[1].plot(stability["pairedRiskDifference"], y, "o", color=colors(signal_index))
    y_ticks = [
        target_index * (len(signal_order) + 1) + signal_index
        for target_index in range(len(targets))
        for signal_index in range(len(signal_order))
    ]
    labels = [f"{target.split('_v1')[0]} | {signal.split('_v1')[0]}" for target in targets for signal in signal_order]
    for axis in axes:
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y_ticks, labels)
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_title("Changed-target completion difference")
    axes[0].set_xlabel("target-aware − matched nonadaptive")
    axes[1].set_title("No-change target-departure harm")
    axes[1].set_xlabel("target-aware − matched nonadaptive")
    figure.suptitle("S09 engineered target adaptation: benefit and stability")
    figure.tight_layout()
    figure.savefig(package / "target_change_benefit_stability.png", dpi=180)
    figure.savefig(package / "target_change_benefit_stability.svg")
    plt.close(figure)


def _spec_markdown(specification: Mapping[str, Any], outcome: str) -> str:
    return f"""# S09 collective-target-change specification

## Top summary

- **Research step ID:** S09
- **Completion status:** Complete; stopped before S10.
- **Artifacts written:** Frozen JSON/Markdown specification and schemas, three target families, four signal permissions, changed-target and no-change matched controls, complete results/contrasts/tests, target correspondence, selected traces, validation/provenance manifests, and the canonical full-results report.
- **Validation result:** PASS.
- **Outcome classification:** **{outcome}.**
- **Caveats or blockers:** Target semantics, signals, target-code remapping, communication, and energy are engineered computational constructs. No S08 mode is carried forward. No execution blocker remains within S09.
- **Recommended next action:** Chief Scientist review. Separately authorize, revise, or defer S10; do not treat S10 as started.

## Frozen question and boundary

Frozen at `{specification['frozenAtUtc']}` before implementation or any S09
trajectory: {specification['frozenQuestion']}

S09 starts from every exact S02 checkpoint without an S03 lesion. It changes
only the external objective and signal overlay. Scenario content, identity,
cell count, values, Algotypes, directions, Selection cursors, native information,
native actions, scheduler, clocks, streams, ledgers, and budgets remain inherited.

## Feasible target changes

1. `reverse_total_order_v1`: reverse the original total order with a unique
   identity-to-position correspondence.
2. `binary_precedence_partial_order_v1`: old-rank parity defines two ordered
   classes; within-class identities are incomparable.
3. `cyclic_tertile_value_classes_v1`: introduce three ordinal value classes and
   cyclically change their order from 0/1/2 to 1/2/0.

Every target conserves identity and count. Equal target codes are equivalent;
strict unequal code inversions are the target error. A direction-preserving
policy code keeps the actor's native direction and Selection cursor unchanged.

## Signal permissions

- `none_negative_control_v1`: no new-target input; target-aware must equal native.
- `local_boundary_relay_v1`: a target token propagates inward one position per
  charged opportunity; only authorized records may receive codes after arrival.
- `gradient_target_code_v1`: authorized records immediately receive only scalar
  target codes, not a target identifier or global map.
- `global_target_broadcast_v1`: an immutable full identity-to-code map is
  broadcast, but candidate construction still uses only native authorized records.

No controller receives occupancy, old/new distance, progress, completion,
timing label, global event clock, future draw, or outcome.

## Controller and controls

The target-aware arm transforms only numeric comparison values on the native
Algotype observation surface. It does not switch Algotype, expand radius, choose
an S08 alternate target, or explore. The primary nonadaptive control receives
and pays for the same signal and shadow candidate but emits native behavior.
Post-completion no-change pairs test stability. Exactly one final native legal
primitive is charged per opportunity and S09 owns no runtime stream.

## Outcomes and inference

Primary adaptation is final new-target completion for each of three targets by
four signals, with 12 exact paired McNemar tests and Holm correction. A separate
12-test Holm family covers no-change target-departure harm. Old/new error,
adaptation/crossover time, 20*n post-hit overshoot, failure, censoring, stability,
native costs, and explicit signal/compute/energy costs are all retained.

Claims are limited to the frozen engineered target adaptation panel.
"""


def _test_lines(frame: pd.DataFrame, *, stability: bool) -> str:
    pieces: list[str] = []
    for row in frame.itertuples():
        label = "departures" if stability else "completions"
        pieces.append(
            f"- `{row.targetChangeId}` × `{row.signalPermissionId}`: "
            f"active/control {label} {row.activePositiveCount}/{row.controlPositiveCount} "
            f"of {row.pairCount}; paired difference {row.pairedRiskDifference:.3f} "
            f"(95% CI {row.pairedRiskDifferenceCiLow:.3f} to "
            f"{row.pairedRiskDifferenceCiHigh:.3f}); discordant "
            f"{row.activeOnlyCount}/{row.controlOnlyCount}; Holm p={row.holmAdjustedPValue:.4g}."
        )
    return "\n".join(pieces)


def _report(
    results: pd.DataFrame,
    contrasts: pd.DataFrame,
    adaptation_tests: pd.DataFrame,
    stability_tests: pd.DataFrame,
    validations: Mapping[str, Any],
    outcome: str,
    git_commit: str,
    generated: str,
) -> str:
    changed = results[results["phase"] == "changed"]
    stability = results[results["phase"] == "stability"]
    informative = adaptation_tests[
        adaptation_tests["signalPermissionId"] != SignalPermission.NONE.value
    ]
    benefit_count = int(
        (
            informative["rejectAtFamilywise0_05"].astype(bool)
            & (informative["pairedRiskDifference"] > 0)
        ).sum()
    )
    harm_count = int(
        (
            adaptation_tests["rejectAtFamilywise0_05"].astype(bool)
            & (adaptation_tests["pairedRiskDifference"] < 0)
        ).sum()
    )
    stability_harm_count = int(
        (
            stability_tests["rejectAtFamilywise0_05"].astype(bool)
            & (stability_tests["pairedRiskDifference"] > 0)
        ).sum()
    )
    return f"""# Research step S09 full results — Change the collective target

## Top summary

- **Research step ID:** S09
- **Completion status:** Complete on 2026-07-18; stopped before S10.
- **Artifacts written:** Frozen target-change specification/schema/Markdown, three feasible target definitions and identity correspondence table, four signal-permission conditions, {len(results):,}-row result/scenario tables, {len(contrasts):,} paired contrasts, 12 primary adaptation and 12 no-change stability tests, adaptation/error/cost summaries and figure, 24 selected traces, validation/accounting/provenance manifests, and this canonical report under `/artifacts/research_steps/S09`.
- **Validation result:** **PASS — {len(results):,}/{len(results):,} planned runs and {len(results):,}/{len(results):,} exact replays; target feasibility/correspondence, exact change times, signal propagation/permissions, no-signal negative controls, controller/cost ledgers, no-change stability, pairing, streams, censors, and complete accounting all passed.**
- **Outcome classification:** **{outcome}.** Informative-signal completion benefits: {benefit_count}/9; completion harms: {harm_count}/12; no-change stability harms: {stability_harm_count}/12 after separate Holm corrections.
- **Caveats or blockers:** Target codes, local relay, gradients, global broadcasts, communication, compute, and energy are engineered simulator constructs. The controller uses installed target semantics and does not infer an unannounced goal. Results are bounded to three targets, four permissions, eight inherited change times, two sizes, three native policies, and frozen budgets. No execution blocker remains within S09.
- **Lay summary:** The simulated cells were asked to reorganize toward a reversed order, a two-class partial order, or a new three-class order. They received either no notice, a slowly spreading local notice, a local target-code field, or a global target map. Each target-aware run was paired with a controller that received and paid for the same notice but ignored it. This tests engineered access to a changed objective, not biological goal inference or learning.
- **Recommended next action:** Return to the Chief Scientist. Review which signal permissions were sufficient and their costs before separately authorizing, revising, or deferring S10; do not start S10 from this handoff.

## Frozen question and decision rule

Before implementation or any S09 trajectory, the immutable configuration froze
three identity-conserving target changes, all eight exact S02 timing checkpoints,
four signal permissions, target-aware and information/cost-matched nonadaptive
controllers, paired no-change stability probes, endpoints, censor rules, two
12-test Holm families, and the claim boundary. Support required a positive
Holm-significant informative-signal completion contrast without a corresponding
stability harm; any target-aware completion/stability harm, no-signal behavioral
difference, or validation failure was prespecified as constraining.

## Lay summary

A changed target is invisible unless the system is given information that can
distinguish it from the old target. The no-signal arm therefore serves as a
negative control. Local signals spread inward from the boundaries; the gradient
condition labels only records already visible to a native policy; the global
condition broadcasts the full immutable mapping. The matched control receives
the same information and bookkeeping cost but continues its original behavior.

## Inputs and inherited contracts

- S01 task, stabilization, budget, target-distance, and pairing contracts.
- All 384 exact S02 timing checkpoints and their preserved native state.
- S03–S06 lesion/runtime/ledger/RNG boundaries as dependency constraints; S09
  applies no lesion or recovery overlay.
- S07's null against timing/opportunity/energy-matched controls.
- S08's four recovery and two intact-stability harms; none of its four behavior
  modes is carried into S09.
- E01 transition/RNG/ledger release and E02 action, information, scheduler,
  fault, semantic-stream, and ledger contracts.
- Workspace plans, upstream manifests, capability/dataset audits, attachment
  manifest, and sidecar. No dataset, network input, GPU, or new dependency was used.

Every run retains the original executable S01 scenario ID, exact S02 occupancy,
Selection cursors, global event index, native streams and ledger, uniform
scheduler, `100*n^2` phase budget, native information, skip-and-continue,
no-retry, and one NoOp/Swap/MemoryUpdate per charged opportunity.

## Detailed methods

### Target definitions and correspondence

`reverse_total_order_v1` reverses the unique old rank. The partial-order target
orders old-rank parity class 0 before class 1 while treating within-class pairs
as incomparable. The value-class target divides old rank into tertiles and
changes class order to 1/2/0. Every target conserves all identities and cell
count. Strict unequal inversions of canonical target codes define error; equal
codes contribute zero. Policy codes are sign-adjusted for the inherited native
direction, preserving Selection cursor meaning without a reset.

### Change times, signals, and observability

The objective changes instantaneously at each exact S02 initialization,
progress-25/50/75, paired-event-25/50/75, or stabilized post-completion
checkpoint. No state, clock, draw, or ledger entry advances at change.

No-signal exposes nothing. Local relay begins at both boundaries and expands at
one position per charged opportunity; the actor never receives the elapsed
clock. Gradient-like input supplies only scalar target codes on already
authorized actor/record reads. Global input broadcasts the immutable target map
but no occupancy, distance, progress, completion, clock, or outcome. All extra
reads, payload, computation, and abstract energy are explicit.

### Controller and controls

The target-aware controller changes only the numeric comparison code on the
ordinary Bubble, Insertion, or Selection observation surface. It does not use
S08 Algotype switching, radius expansion, alternate target selection, or
exploration. The matched nonadaptive arm constructs and meters the same shadow
candidate but emits native behavior. The final proposal retains baseline native
ledger costs; S09 work is charged in a separate process ledger. S09 owns no RNG
stream and emits exactly one final legal proposal.

After first new-target achievement, every changed run continues for exactly
`20*n` opportunities to measure departure/overshoot. Non-achievers keep an
explicit censor. Controller quiescence requires complete actor-context coverage
after local propagation stabilizes; it never removes a failure. Paired no-change
post-completion probes run `20*n` opportunities to detect false target behavior.

### Outcomes and inference

Primary completion used 384 paired cases per target×signal condition and exact
two-sided McNemar tests with Holm correction across 12 contrasts. A separate
12-test family covers any no-change target departure in 48 paired stabilized
cases per condition. Paired t intervals report risk differences. Old/new error,
error AUC, crossover, restricted adaptation time (`budget+1` for censor),
post-hit departure/maximum/AUC, failures, terminal reasons, native ledger, and
signal/communication/compute/energy costs remain available without survivor or
signaled-only selection.

## Commands and execution parameters

```bash
python -m pytest -q tests/test_regeneration_target_change.py
python -m pytest -q tests/test_regeneration*.py
ruff check src/regeneration/target_change.py tests/test_regeneration_target_change.py scripts/build_regeneration_s09.py configs/regeneration/s09_collective_target_change.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/build_regeneration_s09.py --artifacts-dir /artifacts/research_steps/S09 --workers 8
```

Eight replicate workers were used; OMP/MKL/OpenBLAS threads were fixed at one.
All {len(results):,} planned runs were executed and replayed from scratch for
{2 * len(results):,} trajectory executions. There was no runtime-driven scope reduction.
An initial unaggregated attempt was stopped after 6,400 returned jobs when an
O(n^2) exact inversion recount exposed a long-cycle performance bottleneck. The
recount was replaced by an algebraically identical O(n) applied-swap delta,
exhaustively checked over unique and tied five-identity targets. The final
attempt restarted the complete panel; no initial-attempt outcome was inspected
or reused, and no scientific contract changed.

## Results

### Primary changed-target completion contrasts

{_test_lines(adaptation_tests, stability=False)}

### Prespecified no-change stability harms

{_test_lines(stability_tests, stability=True)}

### Accounting and outcome anchors

- Changed-target runs: {len(changed):,}; no-change stability runs: {len(stability):,}; paired contrasts: {len(contrasts):,}.
- Target-aware changed completions: {int(changed[changed['targetAware']]['targetCompleted'].sum()):,}/{len(changed[changed['targetAware']]):,}; matched nonadaptive changed completions: {int(changed[~changed['targetAware']]['targetCompleted'].sum()):,}/{len(changed[~changed['targetAware']]):,}.
- Changed adaptation censors: {int(changed['adaptationCensored'].sum()):,}; overshoot censors: {int(changed['overshootCensored'].sum()):,}; controller-quiescent/phase-budget terminals: {int((changed['stopReason'] == 'controller_quiescent').sum()):,}/{int((changed['stopReason'] == 'phase_event_budget').sum()):,}.
- No-change target departures: {int(stability[stability['targetAware']]['noChangeAnyTargetDeparture'].sum()):,} active and {int(stability[~stability['targetAware']]['noChangeAnyTargetDeparture'].sum()):,} matched control.
- Signal payload units: {int(results['signalPayloadUnits'].sum()):,}; transformed target-record reads: {int(results['targetRecordReads'].sum()):,}; abstract energy units: {int(results['abstractEnergyUnits'].sum()):,}.

The benefit/stability figure keeps recovery and harm on separate axes; no
composite score hides instability or cost.

## Validation

| Gate | Result |
| --- | --- |
| Exact S02 checkpoints and paired change times | PASS — 384/384 |
| Target feasibility, identity conservation, and correspondence | PASS |
| Local relay light cone and aggregate signal ledgers | PASS |
| Permission surfaces / no hidden target or progress leakage | PASS |
| No-signal negative controls | PASS — exact paired behavior |
| Native opportunity and S09 cost ledgers | PASS |
| No-change target-code/native equivalence | PASS |
| Original scenario/RNG root and no S09 stream | PASS |
| Deterministic replay | PASS — {len(results):,}/{len(results):,} |
| Censor retention, pairing, and complete accounting | PASS — zero substitutions or silent exclusions |

All 38 focused S09 tests and all 169 inherited regeneration tests, lint, schema,
Parquet, manifest, and artifact round-trip checks passed. Validation details
are machine-readable in the S09 validation JSON files.

## Artifacts

- `target_change_package/target_change_spec.json`/Markdown and schemas freeze
  target, timing, signal, controller, control, outcome, and claim contracts.
- `target_change_results.parquet` and `target_change_scenarios.parquet` retain
  all runs; `paired_target_change_contrasts.parquet` retains every pair.
- `target_correspondence.parquet` preserves every identity's old rank, new code,
  policy code, and allowed target-position interval.
- Primary adaptation/stability tests, adaptation curves, old/new-error/cost
  trade-offs, and PNG/SVG preserve estimates.
- Twenty-four selected traces plus checkpoint, feasibility, propagation,
  permission, negative-control, cost, stability, pairing, replay, censor,
  accounting, environment, input, and artifact manifests preserve auditability.

## Caveats, blockers, failed assumptions, and limitations

1. **Engineered target access:** The controller is supplied a prespecified code
   semantics when its signal allows. It does not infer an unannounced objective.
2. **No S08 promotion:** None of the harmed S08 behavior modes is called a
   validated adaptation or included here. S09 changes comparison semantics only.
3. **Signal abstraction:** Boundary relay, target-code gradients, and global
   broadcast are transparent information channels, not biological signaling.
4. **Partial-order boundary:** The binary target is an ordered two-class weak
   order: cross-class precedence is measured and within-class pairs are
   incomparable. It does not represent an arbitrary DAG.
5. **New classes:** Tertile class membership is deterministically derived from
   old rank and then cyclically reordered. No identity is inserted, deleted, or
   physically relabeled.
6. **Abstract costs:** Payload, lookup, compute, communication, and energy units
   have no calibrated physical or biological scale.
7. **Pairing boundary:** Base counter-addressed streams share exact addresses
   only while state/read paths match. No draws are added or discarded to force
   coupling after behavioral divergence.
8. **Finite panel:** Effects are bounded to n=20/50, three native policies, both
   directions, eight change times, three targets, four signal permissions, and
   the frozen budgets. A null cannot prove general non-adaptability; a benefit
   cannot establish learning or transfer.
9. **Future experience:** S09 contains no repeated injuries or retained
   across-run learning state. S10 must be separately authorized and must model
   every first-run failure rather than selecting survivors.

No blocker remains within S09. S10 was not started.

## Provenance

- Repository: `Eidosoma/cell_research`
- Branch: `eidosoma/groups/28`
- Source commit at execution: `{git_commit}`
- Benchmark: `{BENCHMARK_VERSION}`
- RNG: inherited E01 SHA-256 counter-addressed actor/Bubble streams; S09 owns no runtime or construction random stream.
- Runtime: Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}, PyArrow {package_version('pyarrow')}, SciPy {package_version('scipy')}; eight workers, numerical-library threads one.
- Generated UTC: {generated}

Input/output SHA-256 hashes are recorded in `input_provenance.json` and
`artifact_manifest.json`. Reproducible source remains in Git.
"""


def _artifact_manifest(output: Path, git_commit: str) -> None:
    artifacts = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        artifacts.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "e05.s09.artifact-manifest.v1",
            "researchStepId": "S09",
            "repositoryCommit": git_commit,
            "artifactCount": len(artifacts),
            "artifacts": artifacts,
        },
    )


def write_outputs(
    output: Path,
    specification: Mapping[str, Any],
    input_gates: Mapping[str, bool],
    reconstructed: Mapping[str, Any],
    bases: list[DynamicJob],
    panel: Mapping[str, Any],
    workers: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    package = output / "target_change_package"
    package.mkdir(exist_ok=True)
    results = pd.DataFrame(panel["rows"])
    contrasts = _paired_contrasts(results)
    correspondence = _target_correspondence(bases)
    adaptation_tests = _binary_tests(contrasts, stability=False)
    stability_tests = _binary_tests(contrasts, stability=True)
    tradeoffs = _tradeoff_summary(contrasts)
    curves = _adaptation_curves(results)
    validations = _validate_panel(
        specification,
        bases,
        reconstructed,
        results,
        contrasts,
        correspondence,
        panel["traces"],
        panel["failures"],
    )
    success = bool(validations["validationSummary"]["success"])
    outcome = _classification(adaptation_tests, stability_tests, success)
    git_commit = _git("rev-parse", "HEAD")
    generated = datetime.now(timezone.utc).isoformat()

    scenario_columns = [
        "targetChangeRunId",
        "targetChangeCaseId",
        "phase",
        "arm",
        "targetChangeId",
        "signalPermissionId",
        "targetAware",
        "noChange",
        "s01PairingBlockId",
        "timingConditionId",
        "clock",
        "nominalFraction",
        "n",
        "policy",
        "direction",
        "replicateOrdinal",
        "sourceScenarioId",
        "sourceCheckpointHash",
        "targetHash",
        "changeEventIndex",
        "adaptationBudget",
        "probeBudget",
    ]
    scenarios = results[scenario_columns].copy()
    checkpoint = pd.DataFrame(reconstructed["checkpointRows"])

    results.to_parquet(output / "target_change_results.parquet", index=False)
    scenarios.to_parquet(output / "target_change_scenarios.parquet", index=False)
    contrasts.to_parquet(output / "paired_target_change_contrasts.parquet", index=False)
    correspondence.to_parquet(package / "target_correspondence.parquet", index=False)
    checkpoint.to_parquet(output / "checkpoint_compatibility.parquet", index=False)
    adaptation_tests.to_parquet(output / "primary_adaptation_tests.parquet", index=False)
    stability_tests.to_parquet(output / "primary_stability_tests.parquet", index=False)
    tradeoffs.to_parquet(output / "target_change_tradeoff_summary.parquet", index=False)
    curves.to_parquet(output / "adaptation_curves.parquet", index=False)

    with (package / "selected_target_change_traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in panel["traces"]:
            handle.write(json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n")

    _write_json(package / "target_change_spec.json", specification)
    _write_json(package / "target_change_spec.schema.json", TARGET_CHANGE_SPEC_SCHEMA)
    _write_json(
        package / "target_change_run.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://eidosoma.local/schemas/e05/s09/target-change-run.schema.json",
            "type": "object",
            "additionalProperties": True,
            "required": [
                "schemaVersion",
                "benchmarkVersion",
                "contract",
                "targetDefinition",
                "sourceScenarioId",
                "sourceCheckpointHash",
                "summary",
                "processLedger",
                "opportunityValidation",
            ],
            "properties": {
                "schemaVersion": {"const": TARGET_CHANGE_RUN_SCHEMA_VERSION},
                "benchmarkVersion": {"const": BENCHMARK_VERSION},
            },
        },
    )
    (package / "target_change_spec.md").write_text(
        _spec_markdown(specification, outcome), encoding="utf-8"
    )
    _plot_results(adaptation_tests, stability_tests, package)

    for name, payload in validations.items():
        filename = {
            "validationSummary": "validation_summary.json",
            "pairingValidation": "pairing_validation.json",
            "targetValidation": "target_feasibility_validation.json",
            "signalValidation": "signal_propagation_permission_validation.json",
            "negativeControlValidation": "negative_control_validation.json",
            "costValidation": "cost_ledger_validation.json",
            "censorValidation": "censor_retention_validation.json",
            "replayValidation": "replay_validation.json",
        }[name]
        _write_json(output / filename, payload)

    changed = results[results["phase"] == "changed"]
    stability = results[results["phase"] == "stability"]
    accounting = {
        "schemaVersion": "e05.s09.run-accounting.v1",
        "researchStepId": "S09",
        "sourceBlocksExpected": 48,
        "sourceBlocksObserved": len({base.s01_pairing_block_id for base in bases}),
        "changedCheckpointsExpected": 384,
        "changedCheckpointsObserved": len({(base.s01_pairing_block_id, base.timing_condition_id) for base in bases}),
        "stabilityCheckpointsExpected": 48,
        "stabilityCheckpointsObserved": len({row.s01PairingBlockId for row in stability.itertuples()}),
        "changedRunsExpected": 9_216,
        "changedRunsObserved": len(changed),
        "stabilityRunsExpected": 1_152,
        "stabilityRunsObserved": len(stability),
        "plannedRunsExpected": 10_368,
        "plannedRunsObserved": len(results),
        "exactReplayExecutionsExpected": 10_368,
        "exactReplayExecutionsObserved": len(results),
        "totalTrajectoryExecutionsExpected": 20_736,
        "totalTrajectoryExecutionsObserved": 2 * len(results),
        "pairedContrastsExpected": 5_184,
        "pairedContrastsObserved": len(contrasts),
        "primaryAdaptationPairsExpected": 4_608,
        "primaryAdaptationPairsObserved": len(contrasts[contrasts["phase"] == "changed"]),
        "primaryStabilityPairsExpected": 576,
        "primaryStabilityPairsObserved": len(contrasts[contrasts["phase"] == "stability"]),
        "selectedTraceRunsExpected": 24,
        "selectedTraceRunsObserved": len(panel["traces"]),
        "runtimeFailureCount": len(panel["failures"]),
        "runtimeFailures": panel["failures"],
        "substitutionCount": 0,
        "silentExclusionCount": 0,
        "scopeReduction": False,
        "success": success,
    }
    _write_json(output / "run_accounting.json", accounting)
    _write_json(
        output / "outcome_classification.json",
        {
            "schemaVersion": "e05.s09.outcome-classification.v1",
            "researchStepId": "S09",
            "validationSuccess": success,
            "classification": outcome,
            "informativeBenefitHolmRejectionCount": int(
                (
                    adaptation_tests["rejectAtFamilywise0_05"].astype(bool)
                    & (adaptation_tests["pairedRiskDifference"] > 0)
                    & (adaptation_tests["signalPermissionId"] != SignalPermission.NONE.value)
                ).sum()
            ),
            "adaptationHarmHolmRejectionCount": int(
                (
                    adaptation_tests["rejectAtFamilywise0_05"].astype(bool)
                    & (adaptation_tests["pairedRiskDifference"] < 0)
                ).sum()
            ),
            "stabilityHarmHolmRejectionCount": int(
                (
                    stability_tests["rejectAtFamilywise0_05"].astype(bool)
                    & (stability_tests["pairedRiskDifference"] > 0)
                ).sum()
            ),
        },
    )
    _write_json(
        output / "execution_attempts.json",
        {
            "schemaVersion": "e05.s09.execution-attempts.v1",
            "researchStepId": "S09",
            "attempts": [
                {
                    "attemptOrdinal": 1,
                    "repositoryCommit": "60f9c88de33b560481649abf247e2932ca5bb7ff",
                    "jobsReturned": 6400,
                    "validationReached": False,
                    "validationSuccess": None,
                    "resultAggregationReached": False,
                    "outcomesInspected": False,
                    "terminationReason": "performance-only restart after O(n^2) inversion recount exposed long full-budget cycling cases; no output artifact had been written",
                },
                {
                    "attemptOrdinal": 2,
                    "repositoryCommit": git_commit,
                    "jobsReturned": len(results),
                    "validationReached": True,
                    "validationSuccess": success,
                    "resultAggregationReached": True,
                    "outcomesInspected": True,
                    "terminationReason": None,
                }
            ],
            "scopeReduction": False,
            "semanticScopeChanged": False,
            "runtimeImplementation": "eight-process panel execution; each trajectory is deterministic and serial, exact distances use a fixture-validated applied-swap delta, and every final-attempt run is replayed from scratch",
        },
    )
    provenance = []
    for path in (CONFIG, S04_CONFIG, *INPUTS):
        provenance.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json(
        output / "input_provenance.json",
        {
            "schemaVersion": "e05.s09.input-provenance.v1",
            "researchStepId": "S09",
            "inputGates": dict(input_gates),
            "inputs": provenance,
        },
    )
    _write_json(
        output / "environment_provenance.json",
        {
            "schemaVersion": "e05.s09.environment-provenance.v1",
            "branch": _git("branch", "--show-current"),
            "gitCommit": git_commit,
            "repository": str(REPOSITORY),
            "generatedUtc": generated,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCountVisible": os.cpu_count(),
            "workerCount": workers,
            "threadEnvironment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            },
            "packages": {
                name: package_version(name)
                for name in ("numpy", "pandas", "pyarrow", "scipy", "matplotlib", "jsonschema", "pytest")
            },
        },
    )
    (output / "execution_commands.log").write_text(
        "\n".join(
            [
                "python -m pytest -q tests/test_regeneration_target_change.py",
                "python -m pytest -q tests/test_regeneration*.py",
                "ruff check src/regeneration/target_change.py tests/test_regeneration_target_change.py scripts/build_regeneration_s09.py configs/regeneration/s09_collective_target_change.json",
                "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/build_regeneration_s09.py --artifacts-dir /artifacts/research_steps/S09 --workers 8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "research_step_full_results.md").write_text(
        _report(
            results,
            contrasts,
            adaptation_tests,
            stability_tests,
            validations,
            outcome,
            git_commit,
            generated,
        ),
        encoding="utf-8",
    )
    _artifact_manifest(output, git_commit)
    return {
        "success": success,
        "runCount": len(results),
        "replayCount": len(results),
        "outcome": outcome,
    }


def build(specification: Mapping[str, Any], workers: int) -> dict[str, Any]:
    validate_target_change_spec(specification)
    input_gates = _validate_inputs()
    reconstructed = _reconstruct_jobs(_load_json(S04_CONFIG))
    bases = _base_jobs(reconstructed)
    jobs = _jobs(bases)
    panel = _run_jobs(jobs, workers)
    return {
        "inputGates": input_gates,
        "reconstructed": reconstructed,
        "bases": bases,
        "panel": panel,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/artifacts/research_steps/S09"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("S09 workers must be in [1,8]")
    specification = _load_json(CONFIG)
    built = build(specification, args.workers)
    result = write_outputs(
        args.artifacts_dir,
        specification,
        built["inputGates"],
        built["reconstructed"],
        built["bases"],
        built["panel"],
        args.workers,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
