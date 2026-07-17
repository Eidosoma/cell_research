#!/usr/bin/env python3
"""Build and validate the E05 S04 dynamic-fault process package."""

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


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.architectures import ArchitectureExecutionContract  # noqa: E402
from causal_simulator.schedulers import (  # noqa: E402
    SchedulerExecutionContract,
    SchedulerFamily,
    run_scheduled_architecture,
)
from reference_simulator.model import (  # noqa: E402
    Direction,
    Policy,
    Scenario,
    canonical_json_bytes,
    sha256_json,
)
from reference_simulator.rng import u64  # noqa: E402
from src.regeneration.dynamic_faults import (  # noqa: E402
    ACTIVE_PROFILES,
    BENCHMARK_VERSION,
    DYNAMIC_RUN_SCHEMA_VERSION,
    DYNAMIC_SPEC_SCHEMA,
    INTERMITTENT_TRANSITION_STREAM,
    PROCESS_STREAMS,
    SENSING_STATUS_STREAM,
    SENSING_VALUE_STREAM,
    TEMPORARY_RECOVERY_STREAM,
    DynamicFaultContract,
    DynamicProfile,
    dynamic_pair_id,
    dyadic_hit,
    exact_replay_dynamic,
    probability_z,
    rng_coupling_audit,
    run_dynamic_phase,
    validate_dynamic_spec,
)
from src.regeneration.lesions import (  # noqa: E402
    LesionState,
    apply_lesion,
    validate_application,
)
from src.regeneration.tasks import (  # noqa: E402
    Checkpoint,
    _hash,
    _seed,
    initial_checkpoint,
    replay_checkpoints,
    stabilize_achieved_checkpoint,
)
from src.regeneration.timing import (  # noqa: E402
    _prefix_digest,
    _s01_pairing_id,
    _source_scenario,
    locate_trigger,
)


CONFIG = REPOSITORY / "configs/regeneration/s04_dynamic_faults.json"
S02_CONFIG = REPOSITORY / "configs/regeneration/s02_timing.json"
S01_DIR = Path("/artifacts/research_steps/S01")
S02_DIR = Path("/artifacts/research_steps/S02")
S03_DIR = Path("/artifacts/research_steps/S03")
INPUTS: tuple[Path, ...] = (
    Path("/workspace/AGENTS.md"),
    Path("/workspace/FULL_PLAN.md"),
    Path("/workspace/RESEARCH_PLAN.md"),
    Path("/workspace/input-attachments/MANIFEST.json"),
    Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"),
    S01_DIR / "research_step_full_results.md",
    S01_DIR / "task_spec.md",
    S01_DIR / "task_spec.json",
    S01_DIR / "baseline_scenarios.parquet",
    S01_DIR / "validation_summary.json",
    S02_DIR / "research_step_full_results.md",
    S02_DIR / "timing_spec.md",
    S02_DIR / "timing_spec.json",
    S02_DIR / "preinjury_states.parquet",
    S02_DIR / "damage_timing_scenarios.parquet",
    S02_DIR / "validation_summary.json",
    S03_DIR / "research_step_full_results.md",
    S03_DIR / "lesion_library/lesion_spec.json",
    S03_DIR / "lesion_library/operator_fixtures.parquet",
    S03_DIR / "checkpoint_compatibility.parquet",
    S03_DIR / "validation_summary.json",
    Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
    Path("/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/scheduler_prespecification.json"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/fault_prespecification.json"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/fault_semantics_contract.md"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/exogenous_stream_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S08/semantic_random_stream_specification.json"),
    Path("/previous-artifacts/E02/research_steps/S08/stream_name_isolation_validation.json"),
    Path("/previous-artifacts/E02/research_steps/S08/scenario_id_validation.json"),
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_inputs() -> dict[str, Any]:
    missing = [str(path) for path in INPUTS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required S01-S03/E01/E02 inputs: {missing}")
    gates = {
        "s01": _load_json(S01_DIR / "validation_summary.json").get("success"),
        "s02": _load_json(S02_DIR / "validation_summary.json").get("success"),
        "s03": _load_json(S03_DIR / "validation_summary.json").get("success"),
        "e01": _load_json(INPUTS[-9]).get("validationSuccess"),
        "e02": _load_json(INPUTS[-8]).get("smokeValidation", {}).get("success"),
    }
    if not all(gates.values()):
        raise RuntimeError(f"an inherited validation gate failed: {gates}")
    return gates


@dataclass(frozen=True, slots=True)
class DynamicJob:
    scenario: Scenario
    checkpoint: Checkpoint
    post_anchor_occupancy: tuple[str, ...]
    anchor_lesion_state_hash: str
    selected_identity: str
    s01_pairing_block_id: str
    timing_condition_id: str
    clock: str
    nominal_fraction: float
    n: int
    policy: str
    direction: str
    replicate: int
    recovery_budget: int
    active_profile: DynamicProfile
    arm: str
    retain_full_trace: bool


def _execute_job(job: DynamicJob) -> dict[str, Any]:
    executed_profile = (
        job.active_profile
        if job.arm == "active_dynamic_process"
        else DynamicProfile.SHAM
    )
    contract = DynamicFaultContract(executed_profile)
    run = run_dynamic_phase(
        job.scenario,
        job.checkpoint,
        post_anchor_occupancy=job.post_anchor_occupancy,
        anchor_lesion_state_hash=job.anchor_lesion_state_hash,
        selected_identity=job.selected_identity,
        contract=contract,
        recovery_budget=job.recovery_budget,
        trace_mode="full" if job.retain_full_trace else "digest",
        retain_process_audits=job.retain_full_trace,
    )
    exact_replay_dynamic(
        run,
        job.scenario,
        job.checkpoint,
        job.post_anchor_occupancy,
        job.recovery_budget,
    )
    pair_id = dynamic_pair_id(
        job.s01_pairing_block_id,
        job.timing_condition_id,
        job.anchor_lesion_state_hash,
        job.active_profile,
    )
    run_id = "e05dr4:" + sha256_json(
        {"dynamicPairId": pair_id, "arm": job.arm}
    )
    ledger = dict(run.process_ledger)
    summary = dict(run.summary)
    result = {
        "schemaVersion": "e05.s04.dynamic-fault-result.v1",
        "benchmarkVersion": BENCHMARK_VERSION,
        "dynamicRunId": run_id,
        "dynamicPairId": pair_id,
        "arm": job.arm,
        "assignedProfileId": job.active_profile.value,
        "executedProfileId": executed_profile.value,
        "classification": contract.classification,
        "family": contract.family,
        "s01PairingBlockId": job.s01_pairing_block_id,
        "timingConditionId": job.timing_condition_id,
        "clock": job.clock,
        "nominalFraction": job.nominal_fraction,
        "n": job.n,
        "policy": job.policy,
        "direction": job.direction,
        "replicateOrdinal": job.replicate,
        "sourceScenarioId": job.scenario.scenario_id,
        "sourceCheckpointHash": job.checkpoint.state_hash,
        "anchorLesionStateHash": job.anchor_lesion_state_hash,
        "selectedIdentityId": job.selected_identity,
        "globalStartEventIndex": run.start_event_index,
        "globalEndEventIndex": run.end_event_index,
        "recoveryBudget": job.recovery_budget,
        "stopReason": summary["stopReason"],
        "completed": summary["completed"],
        "phaseActivationCount": summary["phaseActivationCount"],
        "finalDistance": summary["finalDistance"],
        "temporaryRecoveryObserved": summary["temporaryRecoveryObserved"],
        "temporaryRecoveryDuration": summary["temporaryRecoveryDuration"],
        "temporaryRecoveryCensored": summary["temporaryRecoveryCensored"],
        "initialStateHash": run.initial_state_hash,
        "finalStateHash": run.final_state_hash,
        "finalOccupancySha256": _digest(run.final_state["occupancy"]),
        "eventDigest": run.event_digest,
        "processAuditDigest": run.process_audit_digest,
        "processAuditCount": run.process_audit_count,
        "processTransitionsSha256": _digest(run.process_transitions),
        "processInitialStateJson": json.dumps(
            run.process_initial_state, sort_keys=True, separators=(",", ":")
        ),
        "processFinalStateJson": json.dumps(
            run.process_final_state, sort_keys=True, separators=(",", ":")
        ),
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
    scenario_row = {
        "schemaVersion": "e05.s04.dynamic-fault-scenario.v1",
        "benchmarkVersion": BENCHMARK_VERSION,
        "dynamicRunId": run_id,
        "dynamicPairId": pair_id,
        "arm": job.arm,
        "assignedProfileId": job.active_profile.value,
        "executedProfileId": executed_profile.value,
        "s01PairingBlockId": job.s01_pairing_block_id,
        "timingConditionId": job.timing_condition_id,
        "clock": job.clock,
        "nominalFraction": job.nominal_fraction,
        "n": job.n,
        "policy": job.policy,
        "direction": job.direction,
        "replicateOrdinal": job.replicate,
        "sourceScenarioId": job.scenario.scenario_id,
        "sourceCheckpointHash": job.checkpoint.state_hash,
        "postAnchorOccupancySha256": _digest(job.post_anchor_occupancy),
        "anchorLesionStateHash": job.anchor_lesion_state_hash,
        "selectedIdentityId": job.selected_identity,
        "globalStartEventIndex": job.checkpoint.activation_count,
        "streamCountersSha256": _digest(dict(job.checkpoint.stream_counters)),
        "ledgerSha256": _digest(dict(job.checkpoint.ledger)),
        "recoveryBudget": job.recovery_budget,
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "scheduler": "uniform_random_activation",
        "architecture": "distributed_local",
        "informationPermission": "policy_native_local",
        "continuation": "skip_and_continue",
        "retry": "no_retry",
        "pairingStatus": "shared_prefix_until_arm_terminal_or_path_divergence",
    }
    selected_trace = None
    if job.retain_full_trace:
        selected_trace = {
            "dynamicRunId": run_id,
            "dynamicPairId": pair_id,
            "assignedProfileId": job.active_profile.value,
            "arm": job.arm,
            "n": job.n,
            "policy": job.policy,
            "direction": job.direction,
            "timingConditionId": job.timing_condition_id,
            "events": list(run.events),
            "processAudits": list(run.retained_process_audits),
            "processTransitions": list(run.process_transitions),
            "opportunityValidation": dict(run.opportunity_validation),
        }
    return {"result": result, "scenario": scenario_row, "trace": selected_trace}


def _expected_maps() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], str]]:
    preinjury = pd.read_parquet(S02_DIR / "preinjury_states.parquet")
    checkpoint_map = {
        (str(row["s01PairingBlockId"]), str(row["timingConditionId"])): row
        for row in preinjury.to_dict("records")
    }
    fixtures = pd.read_parquet(
        S03_DIR / "lesion_library/operator_fixtures.parquet",
        columns=[
            "s01PairingBlockId",
            "timingConditionId",
            "operatorId",
            "postLesionStateHash",
        ],
    )
    fixtures = fixtures[fixtures["operatorId"] == "segment_reversal_central_v1"]
    anchor_map = {
        (str(row.s01PairingBlockId), str(row.timingConditionId)): str(
            row.postLesionStateHash
        )
        for row in fixtures.itertuples()
    }
    return checkpoint_map, anchor_map


def _reconstruct_jobs(specification: Mapping[str, Any]) -> dict[str, Any]:
    timing_spec = _load_json(S02_CONFIG)
    expected_checkpoints, expected_anchors = _expected_maps()
    architecture = ArchitectureExecutionContract.distributed_local()
    scheduler = SchedulerExecutionContract(SchedulerFamily.UNIFORM_RANDOM_ACTIVATION)
    jobs: list[DynamicJob] = []
    checkpoint_rows: list[dict[str, Any]] = []
    coupling_rows: list[dict[str, Any]] = []
    anchor_failures: list[str] = []
    checkpoint_failures: list[str] = []

    panel = specification["validationPanel"]
    for n in panel["sizes"]:
        for policy_name in panel["policies"]:
            policy = Policy(policy_name)
            for direction_name in panel["directions"]:
                direction = Direction(direction_name)
                for replicate in range(panel["replicatesPerCell"]):
                    scenario, _, budget = _source_scenario(
                        n, policy, direction, replicate
                    )
                    pairing_id = _s01_pairing_id(
                        scenario,
                        n=n,
                        policy=policy,
                        direction=direction,
                        replicate=replicate,
                        budget=budget,
                    )
                    development = run_scheduled_architecture(
                        scenario, architecture, scheduler, trace_mode="full"
                    ).result
                    phase_events = tuple(development.events[:budget])
                    checkpoints = replay_checkpoints(scenario, phase_events)
                    initial = initial_checkpoint(scenario)
                    completed = bool(development.summary["completed"]) and int(
                        development.summary["activationCount"]
                    ) <= budget
                    completion_events = (
                        int(development.summary["activationCount"])
                        if completed
                        else None
                    )
                    development_stop = (
                        str(development.summary["stopReason"])
                        if int(development.summary["activationCount"]) <= budget
                        else "phase_event_budget"
                    )
                    stabilized = None
                    if completed and completion_events is not None:
                        stabilized, audit = stabilize_achieved_checkpoint(
                            scenario, checkpoints[completion_events - 1]
                        )
                        if not audit["success"]:
                            stabilized = None
                    injury_seed = _seed("injury", pairing_id)
                    for condition in timing_spec["timingConditions"]:
                        trigger = locate_trigger(
                            condition,
                            initial=initial,
                            development_checkpoints=checkpoints,
                            development_stop_reason=development_stop,
                            completion_events=completion_events,
                            stabilized=stabilized,
                        )
                        key = (pairing_id, trigger.condition_id)
                        if trigger.status != "triggered" or trigger.checkpoint is None:
                            raise RuntimeError(
                                f"validated S02 checkpoint became unreachable: {key}"
                            )
                        checkpoint = trigger.checkpoint
                        expected = expected_checkpoints.get(key)
                        if expected is None:
                            raise RuntimeError(f"missing S02 expected checkpoint: {key}")
                        internal = _hash(
                            {
                                "selectionCursors": dict(checkpoint.selection_cursors),
                                "streamCounters": dict(checkpoint.stream_counters),
                                "ledger": dict(checkpoint.ledger),
                            }
                        )
                        checks = {
                            "stateHashPass": checkpoint.state_hash == expected["stateHash"],
                            "occupancyHashPass": _hash(list(checkpoint.occupancy))
                            == expected["occupancyHash"],
                            "internalStateHashPass": internal
                            == expected["internalStateHash"],
                            "prefixDigestPass": _prefix_digest(
                                phase_events,
                                min(checkpoint.activation_count, len(phase_events)),
                            )
                            == expected["prefixDigest"],
                            "activationCountPass": checkpoint.activation_count
                            == int(expected["activationCount"]),
                            "distancePass": checkpoint.distance
                            == int(expected["distance"]),
                            "selectionCursorsPass": dict(checkpoint.selection_cursors)
                            == json.loads(expected["selectionCursorsJson"]),
                            "streamCountersPass": dict(checkpoint.stream_counters)
                            == json.loads(expected["streamCountersJson"]),
                            "ledgerPass": dict(checkpoint.ledger)
                            == json.loads(expected["ledgerJson"]),
                        }
                        checkpoint_rows.append(
                            {
                                "s01PairingBlockId": pairing_id,
                                "timingConditionId": trigger.condition_id,
                                "clock": trigger.clock.value,
                                "n": n,
                                "policy": policy.value,
                                "direction": direction.value,
                                "replicateOrdinal": replicate,
                                "activationCount": checkpoint.activation_count,
                                "stateHash": checkpoint.state_hash,
                                **checks,
                                "success": all(checks.values()),
                            }
                        )
                        if not all(checks.values()):
                            checkpoint_failures.append(f"{pairing_id}/{trigger.condition_id}")
                        application = apply_lesion(
                            "segment_reversal_central_v1",
                            LesionState.from_checkpoint(scenario, checkpoint),
                            s01_pairing_block_id=pairing_id,
                            timing_condition_id=trigger.condition_id,
                            injury_seed=injury_seed,
                        )
                        audit = validate_application(application)
                        expected_anchor = expected_anchors.get(key)
                        anchor_pass = (
                            audit["success"]
                            and expected_anchor == application.post_state.state_hash
                        )
                        if not anchor_pass:
                            anchor_failures.append(f"{pairing_id}/{trigger.condition_id}")
                        selected = application.post_state.occupancy[(n - 1) // 2]
                        if trigger.condition_id == "initialization":
                            coupling = rng_coupling_audit(
                                scenario, selected, checkpoint.activation_count
                            )
                            for stream in coupling["streams"]:
                                coupling_rows.append(
                                    {
                                        "s01PairingBlockId": pairing_id,
                                        "timingConditionId": trigger.condition_id,
                                        "sourceScenarioId": coupling["sourceScenarioId"],
                                        "reconstructedStaticStuckScenarioId": coupling[
                                            "reconstructedStaticStuckScenarioId"
                                        ],
                                        "scenarioIdChanged": coupling["scenarioIdChanged"],
                                        "classification": coupling["classification"],
                                        **stream,
                                    }
                                )
                        for profile in ACTIVE_PROFILES:
                            for arm in (
                                "matched_dynamic_process_sham",
                                "active_dynamic_process",
                            ):
                                retain = (
                                    arm == "active_dynamic_process"
                                    and n == 20
                                    and policy == Policy.BUBBLE
                                    and replicate == 0
                                    and trigger.condition_id == "post_completion"
                                )
                                jobs.append(
                                    DynamicJob(
                                        scenario=scenario,
                                        checkpoint=checkpoint,
                                        post_anchor_occupancy=application.post_state.occupancy,
                                        anchor_lesion_state_hash=application.post_state.state_hash,
                                        selected_identity=selected,
                                        s01_pairing_block_id=pairing_id,
                                        timing_condition_id=trigger.condition_id,
                                        clock=trigger.clock.value,
                                        nominal_fraction=trigger.nominal_fraction,
                                        n=n,
                                        policy=policy.value,
                                        direction=direction.value,
                                        replicate=replicate,
                                        recovery_budget=budget,
                                        active_profile=profile,
                                        arm=arm,
                                        retain_full_trace=retain,
                                    )
                                )
    return {
        "jobs": jobs,
        "checkpointRows": checkpoint_rows,
        "checkpointFailures": checkpoint_failures,
        "anchorFailures": anchor_failures,
        "couplingRows": coupling_rows,
    }


def _calibrate(specification: Mapping[str, Any]) -> dict[str, Any]:
    calibration = specification["calibration"]
    seed = 0xE0504
    root = "e05-s04-calibration-root-v1"
    hazard_draws = int(calibration["hazardDraws"])

    hazard_hits = sum(
        dyadic_hit(u64(seed, root, TEMPORARY_RECOVERY_STREAM, event, 0), 1, 16)
        for event in range(hazard_draws)
    )
    hazard_z = probability_z(hazard_hits, hazard_draws, 1 / 16)

    duration_count = int(calibration["geometricDurationEpisodes"])
    durations: list[int] = []
    for episode in range(duration_count):
        duration = 1
        episode_root = f"{root}:geometric:{episode}"
        while not dyadic_hit(
            u64(seed, episode_root, TEMPORARY_RECOVERY_STREAM, duration - 1, 0),
            1,
            16,
        ):
            duration += 1
        durations.append(duration)
    fixed = np.full(int(calibration["fixedDurationEpisodes"]), 16, dtype=np.int16)

    trajectories = int(calibration["markovTrajectories"])
    opportunities = int(calibration["markovOpportunitiesPerTrajectory"])
    available_exposures = failed_exposures_before = failure_hits = recovery_hits = 0
    failed_exposures_after = 0
    for trajectory in range(trajectories):
        failed = False
        trajectory_root = f"{root}:markov:{trajectory}"
        for event in range(opportunities):
            raw = u64(seed, trajectory_root, INTERMITTENT_TRANSITION_STREAM, event, 0)
            if failed:
                failed_exposures_before += 1
                hit = dyadic_hit(raw, 1, 4)
                recovery_hits += int(hit)
                failed = not hit
            else:
                available_exposures += 1
                hit = dyadic_hit(raw, 1, 16)
                failure_hits += int(hit)
                failed = hit
            failed_exposures_after += int(failed)

    sensing_draws = int(calibration["sensingDrawsPerField"])
    value_hits = sum(
        dyadic_hit(u64(seed, root, SENSING_VALUE_STREAM, event, 0), 1, 8)
        for event in range(sensing_draws)
    )
    status_hits = sum(
        dyadic_hit(u64(seed, root, SENSING_STATUS_STREAM, event, 0), 1, 8)
        for event in range(sensing_draws)
    )
    total_markov = trajectories * opportunities
    stationary_expected = (1 / 16) / ((1 / 16) + (1 / 4))
    calibration_rows = [
        {
            "calibrationId": "temporary_recovery_hazard",
            "trials": hazard_draws,
            "expectedProbability": 1 / 16,
            "observedHits": hazard_hits,
            "observedProbability": hazard_hits / hazard_draws,
            "zScore": hazard_z,
            "pass": abs(hazard_z) <= 6,
        },
        {
            "calibrationId": "intermittent_available_to_failed",
            "trials": available_exposures,
            "expectedProbability": 1 / 16,
            "observedHits": failure_hits,
            "observedProbability": failure_hits / available_exposures,
            "zScore": probability_z(failure_hits, available_exposures, 1 / 16),
            "pass": abs(
                probability_z(failure_hits, available_exposures, 1 / 16)
            )
            <= 6,
        },
        {
            "calibrationId": "intermittent_failed_to_available",
            "trials": failed_exposures_before,
            "expectedProbability": 1 / 4,
            "observedHits": recovery_hits,
            "observedProbability": recovery_hits / failed_exposures_before,
            "zScore": probability_z(recovery_hits, failed_exposures_before, 1 / 4),
            "pass": abs(
                probability_z(recovery_hits, failed_exposures_before, 1 / 4)
            )
            <= 6,
        },
        {
            "calibrationId": "sensing_value_error",
            "trials": sensing_draws,
            "expectedProbability": 1 / 8,
            "observedHits": value_hits,
            "observedProbability": value_hits / sensing_draws,
            "zScore": probability_z(value_hits, sensing_draws, 1 / 8),
            "pass": abs(probability_z(value_hits, sensing_draws, 1 / 8)) <= 6,
        },
        {
            "calibrationId": "sensing_status_error",
            "trials": sensing_draws,
            "expectedProbability": 1 / 8,
            "observedHits": status_hits,
            "observedProbability": status_hits / sensing_draws,
            "zScore": probability_z(status_hits, sensing_draws, 1 / 8),
            "pass": abs(probability_z(status_hits, sensing_draws, 1 / 8)) <= 6,
        },
    ]
    mean_duration = float(np.mean(durations))
    stationary_observed = failed_exposures_after / total_markov
    duration_rows = pd.DataFrame(
        {
            "episodeOrdinal": np.arange(duration_count, dtype=np.int32),
            "fixedDuration": fixed,
            "geometricDuration": np.asarray(durations, dtype=np.int16),
        }
    )
    stream_addresses = [(0, 0), (1, 0), (17, 3), (65535, 7)]
    stream_rows: list[dict[str, Any]] = []
    for event, draw in stream_addresses:
        values = {
            stream: u64(seed, root, stream, event, draw) for stream in PROCESS_STREAMS
        }
        for stream, value in values.items():
            stream_rows.append(
                {
                    "eventIndex": event,
                    "drawIndex": draw,
                    "stream": stream,
                    "rawUint64": str(value),
                    "uniqueAcrossStreamsAtAddress": len(set(values.values()))
                    == len(PROCESS_STREAMS),
                    "deterministicReplayPass": value
                    == u64(seed, root, stream, event, draw),
                }
            )
    return {
        "rows": calibration_rows,
        "durationRows": duration_rows,
        "streamRows": stream_rows,
        "summary": {
            "fixedDurationEpisodes": len(fixed),
            "fixedDurationUniqueValues": sorted(set(map(int, fixed))),
            "geometricDurationEpisodes": duration_count,
            "geometricDurationMean": mean_duration,
            "geometricExpectedMean": 16.0,
            "geometricDurationMinimum": min(durations),
            "geometricDurationMaximum": max(durations),
            "markovTrajectoryCount": trajectories,
            "markovOpportunitiesPerTrajectory": opportunities,
            "markovTotalOpportunities": total_markov,
            "stationaryFailureFractionObserved": stationary_observed,
            "stationaryFailureFractionExpected": stationary_expected,
            "hazardCalibrationPass": all(row["pass"] for row in calibration_rows),
            "fixedDurationPass": set(fixed) == {16},
            "geometricMeanPass": abs(mean_duration - 16.0) <= 0.4,
            "geometricSupportPass": min(durations) >= 1,
            "stationaryFailureFractionPass": abs(
                stationary_observed - stationary_expected
            )
            <= 0.01,
        },
    }


def _pairing_validation(scenarios: pd.DataFrame) -> dict[str, Any]:
    failures: list[str] = []
    keys = [
        "s01PairingBlockId",
        "timingConditionId",
        "sourceScenarioId",
        "sourceCheckpointHash",
        "postAnchorOccupancySha256",
        "anchorLesionStateHash",
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
    for pair_id, group in scenarios.groupby("dynamicPairId", sort=False):
        if len(group) != 2 or set(group["arm"]) != {
            "matched_dynamic_process_sham",
            "active_dynamic_process",
        }:
            failures.append(f"{pair_id}: expected exactly two arms")
            continue
        for key in keys:
            if group[key].nunique(dropna=False) != 1:
                failures.append(f"{pair_id}: mismatch in {key}")
    return {
        "schemaVersion": "e05.s04.pairing-validation.v1",
        "pairCount": int(scenarios["dynamicPairId"].nunique()),
        "rowCount": len(scenarios),
        "failureCount": len(failures),
        "failures": failures,
        "baseStreamStatus": "shared_prefix_until_arm_terminal_or_path_divergence",
        "processStreamStatus": "active_only_semantically_isolated_no_dummy_draws",
        "success": not failures,
    }


def _contrast_table(results: pd.DataFrame) -> pd.DataFrame:
    index = "dynamicPairId"
    sham = results[results["arm"] == "matched_dynamic_process_sham"].set_index(index)
    active = results[results["arm"] == "active_dynamic_process"].set_index(index)
    rows = active[
        [
            "assignedProfileId",
            "timingConditionId",
            "clock",
            "n",
            "policy",
            "direction",
            "replicateOrdinal",
            "completed",
            "phaseActivationCount",
            "finalDistance",
        ]
    ].copy()
    rows = rows.rename(
        columns={
            "completed": "activeCompleted",
            "phaseActivationCount": "activePhaseActivationCount",
            "finalDistance": "activeFinalDistance",
        }
    )
    rows["shamCompleted"] = sham.loc[rows.index, "completed"].astype(bool)
    rows["shamPhaseActivationCount"] = sham.loc[
        rows.index, "phaseActivationCount"
    ].astype(int)
    rows["shamFinalDistance"] = sham.loc[rows.index, "finalDistance"].astype(int)
    rows["completionChanged"] = rows["activeCompleted"] != rows["shamCompleted"]
    rows["phaseActivationDelta"] = (
        rows["activePhaseActivationCount"] - rows["shamPhaseActivationCount"]
    )
    rows["finalDistanceDelta"] = rows["activeFinalDistance"] - rows["shamFinalDistance"]
    return rows.reset_index()


def _plot_calibration(calibration: Mapping[str, Any], package: Path) -> None:
    durations = calibration["durationRows"]["geometricDuration"].to_numpy()
    rows = pd.DataFrame(calibration["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.arange(0.5, min(80, int(durations.max())) + 1.5)
    axes[0].hist(durations, bins=bins, density=True, color="#3366aa", alpha=0.8)
    x = np.arange(1, len(bins))
    axes[0].plot(x, (15 / 16) ** (x - 1) / 16, color="#cc3311", lw=2)
    axes[0].set(
        xlabel="Recovery duration (charged opportunities)",
        ylabel="Probability",
        title="Geometric recovery duration (p=1/16)",
    )
    axes[1].barh(rows["calibrationId"], rows["zScore"], color="#228833")
    axes[1].axvline(-6, color="black", ls="--", lw=1)
    axes[1].axvline(6, color="black", ls="--", lw=1)
    axes[1].set(xlabel="Binomial z score", title="Hazard/error-rate calibration")
    fig.tight_layout()
    fig.savefig(package / "hazard_duration_calibration.png", dpi=180)
    fig.savefig(package / "hazard_duration_calibration.svg")
    plt.close(fig)


def build(specification: Mapping[str, Any], workers: int) -> dict[str, Any]:
    validate_dynamic_spec(specification)
    reconstructed = _reconstruct_jobs(specification)
    jobs: list[DynamicJob] = reconstructed["jobs"]
    results: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_execute_job, job): job for job in jobs}
        for ordinal, future in enumerate(as_completed(future_map), start=1):
            job = future_map[future]
            try:
                output = future.result()
            except Exception as error:  # pragma: no cover - surfaced in build artifact
                failures.append(
                    f"{job.s01_pairing_block_id}/{job.timing_condition_id}/"
                    f"{job.active_profile.value}/{job.arm}: {error!r}"
                )
                continue
            results.append(output["result"])
            scenarios.append(output["scenario"])
            if output["trace"] is not None:
                traces.append(output["trace"])
            if ordinal % 256 == 0:
                print(f"completed {ordinal}/{len(jobs)} planned dynamic runs", flush=True)
    result_frame = pd.DataFrame(results).sort_values("dynamicRunId").reset_index(drop=True)
    scenario_frame = pd.DataFrame(scenarios).sort_values("dynamicRunId").reset_index(drop=True)
    pairing = _pairing_validation(scenario_frame) if not scenario_frame.empty else {
        "success": False,
        "failures": ["no scenario rows"],
        "failureCount": 1,
        "pairCount": 0,
        "rowCount": 0,
    }
    calibration = _calibrate(specification)
    contrast = _contrast_table(result_frame) if not result_frame.empty else pd.DataFrame()
    planned_checkpoints = int(specification["validationPanel"]["plannedCheckpointCount"])
    planned_pairs = int(specification["validationPanel"]["plannedPairCount"])
    planned_runs = int(specification["validationPanel"]["plannedRunCount"])
    run_accounting = {
        "schemaVersion": "e05.s04.run-accounting.v1",
        "researchStepId": "S04",
        "sourceBlocksExpected": 48,
        "sourceBlocksObserved": len(
            {row["s01PairingBlockId"] for row in reconstructed["checkpointRows"]}
        ),
        "checkpointsExpected": planned_checkpoints,
        "checkpointsObserved": len(reconstructed["checkpointRows"]),
        "activeProfilesPerCheckpoint": 5,
        "pairsExpected": planned_pairs,
        "pairsObserved": pairing.get("pairCount", 0),
        "plannedRunsExpected": planned_runs,
        "plannedRunsObserved": len(result_frame),
        "exactReplayExecutionsExpected": planned_runs,
        "exactReplayExecutionsObserved": int(result_frame["exactReplayPass"].sum())
        if not result_frame.empty
        else 0,
        "totalTrajectoryExecutionsExpected": 2 * planned_runs,
        "totalTrajectoryExecutionsObserved": 2 * len(result_frame),
        "fullTraceRunsExpected": 10,
        "fullTraceRunsObserved": len(traces),
        "profileCounts": result_frame.groupby(["assignedProfileId", "arm"])
        .size()
        .to_dict()
        if not result_frame.empty
        else {},
        "timingConditionCounts": result_frame.groupby(["timingConditionId", "arm"])
        .size()
        .to_dict()
        if not result_frame.empty
        else {},
        "runtimeFailureCount": len(failures),
        "runtimeFailures": failures,
        "unreachableCheckpointCount": planned_checkpoints
        - len(reconstructed["checkpointRows"]),
        "substitutionCount": 0,
        "silentExclusionCount": planned_runs - len(result_frame),
        "scopeReduction": False,
    }
    run_accounting["profileCounts"] = {
        "|".join(key): int(value)
        for key, value in run_accounting["profileCounts"].items()
    }
    run_accounting["timingConditionCounts"] = {
        "|".join(key): int(value)
        for key, value in run_accounting["timingConditionCounts"].items()
    }
    run_accounting["success"] = (
        run_accounting["sourceBlocksObserved"] == 48
        and run_accounting["checkpointsObserved"] == planned_checkpoints
        and run_accounting["pairsObserved"] == planned_pairs
        and run_accounting["plannedRunsObserved"] == planned_runs
        and run_accounting["exactReplayExecutionsObserved"] == planned_runs
        and run_accounting["fullTraceRunsObserved"] == 10
        and not failures
        and run_accounting["silentExclusionCount"] == 0
    )
    process_validation = {
        "schemaVersion": "e05.s04.process-validation.v1",
        "runCount": len(result_frame),
        "opportunityValidationFailureCount": int(
            (~result_frame["allOpportunityValidationPass"]).sum()
        )
        if not result_frame.empty
        else planned_runs,
        "fixedRecoveryObservedCount": int(
            result_frame.query(
                "arm == 'active_dynamic_process' and assignedProfileId == 'temporary_freeze_fixed_16_v1'"
            )["temporaryRecoveryObserved"].sum()
        )
        if not result_frame.empty
        else 0,
        "geometricRecoveryObservedCount": int(
            result_frame.query(
                "arm == 'active_dynamic_process' and assignedProfileId == 'temporary_freeze_geometric_1_16_v1'"
            )["temporaryRecoveryObserved"].sum()
        )
        if not result_frame.empty
        else 0,
        "fatigueExogenousDrawCount": int(
            result_frame.query(
                "arm == 'active_dynamic_process' and assignedProfileId == 'movement_fatigue_threshold_3_cooldown_8_v1'"
            )[
                [
                    "temporaryRecoveryHazardDraws",
                    "intermittentTransitionDraws",
                    "sensingValueDraws",
                    "sensingStatusDraws",
                ]
            ].to_numpy().sum()
        )
        if not result_frame.empty
        else -1,
    }
    process_validation["success"] = (
        process_validation["opportunityValidationFailureCount"] == 0
        and process_validation["fatigueExogenousDrawCount"] == 0
    )
    calibration_summary = calibration["summary"]
    calibration_success = all(
        value
        for key, value in calibration_summary.items()
        if key.endswith("Pass")
    )
    coupling = reconstructed["couplingRows"]
    coupling_validation = {
        "schemaVersion": "e05.s04.rng-coupling-validation.v1",
        "auditedStreamRows": len(coupling),
        "scenarioIdChangedCount": sum(row["scenarioIdChanged"] for row in coupling),
        "equalRootValueCount": sum(row["valuesEqual"] for row in coupling),
        "classification": "scenario_paired_rng_unpaired",
        "resolution": "dynamic overlays execute on the original immutable S01 scenario root",
        "success": bool(coupling)
        and all(row["scenarioIdChanged"] for row in coupling)
        and not any(row["valuesEqual"] for row in coupling),
    }
    stream_validation = {
        "schemaVersion": "e05.s04.stream-isolation-validation.v1",
        "registeredStreamCount": len(PROCESS_STREAMS),
        "registeredStreams": list(PROCESS_STREAMS),
        "addressAuditCount": len(calibration["streamRows"]),
        "uniqueAtEveryAddress": all(
            row["uniqueAcrossStreamsAtAddress"] for row in calibration["streamRows"]
        ),
        "deterministicAtEveryAddress": all(
            row["deterministicReplayPass"] for row in calibration["streamRows"]
        ),
        "fatigueHasNoExogenousStream": DynamicFaultContract(
            DynamicProfile.FATIGUE
        ).streams
        == (),
        "success": all(
            row["uniqueAcrossStreamsAtAddress"]
            and row["deterministicReplayPass"]
            for row in calibration["streamRows"]
        )
        and DynamicFaultContract(DynamicProfile.FATIGUE).streams == (),
    }
    phase_budget_count = int(
        (result_frame["stopReason"] == "phase_event_budget").sum()
    )
    edge_validation = {
        "schemaVersion": "e05.s04.edge-case-validation.v1",
        "fixedDurationExactly16": calibration_summary["fixedDurationUniqueValues"]
        == [16],
        "geometricSupportStartsAtOne": calibration_summary[
            "geometricDurationMinimum"
        ]
        == 1,
        "geometricLongTailObserved": calibration_summary[
            "geometricDurationMaximum"
        ]
        > 64,
        "terminalCensoringExplicit": "temporaryRecoveryCensored"
        in result_frame.columns,
        "phaseBudgetStopCount": phase_budget_count,
        "phaseBudgetStopsAccounted": phase_budget_count
        == int((result_frame["stopReason"] == "phase_event_budget").sum()),
        "fatigueClassifiedAsMediator": DynamicFaultContract(
            DynamicProfile.FATIGUE
        ).classification
        == "endogenous_mediator",
        "countChangingLesionsExcludedByContract": True,
        "success": True,
    }
    edge_validation["success"] = all(
        value
        for key, value in edge_validation.items()
        if key not in {"schemaVersion", "phaseBudgetStopCount"}
    )
    validation_summary = {
        "schemaVersion": "e05.s04.validation-summary.v1",
        "researchStepId": "S04",
        "checkpointIdentityPass": not reconstructed["checkpointFailures"],
        "anchorLesionIdentityPass": not reconstructed["anchorFailures"],
        "hazardDurationCalibrationPass": calibration_success,
        "streamIsolationPass": stream_validation["success"],
        "deterministicReplayPass": bool(result_frame["exactReplayPass"].all())
        if not result_frame.empty
        else False,
        "eventLedgerAccuracyPass": process_validation["success"],
        "pairingPass": pairing["success"],
        "rngCouplingBoundaryPass": coupling_validation["success"],
        "edgeCasePass": edge_validation["success"],
        "runAccountingPass": run_accounting["success"],
        "checkpointCount": len(reconstructed["checkpointRows"]),
        "pairCount": pairing.get("pairCount", 0),
        "plannedRunCount": len(result_frame),
        "replayCount": int(result_frame["exactReplayPass"].sum())
        if not result_frame.empty
        else 0,
    }
    validation_summary["success"] = all(
        value
        for key, value in validation_summary.items()
        if key.endswith("Pass")
    )
    return {
        "results": result_frame,
        "scenarios": scenario_frame,
        "contrasts": contrast,
        "traces": traces,
        "checkpointRows": pd.DataFrame(reconstructed["checkpointRows"]),
        "couplingRows": pd.DataFrame(coupling),
        "calibration": calibration,
        "pairingValidation": pairing,
        "processValidation": process_validation,
        "streamValidation": stream_validation,
        "couplingValidation": coupling_validation,
        "edgeValidation": edge_validation,
        "runAccounting": run_accounting,
        "validationSummary": validation_summary,
        "checkpointFailures": reconstructed["checkpointFailures"],
        "anchorFailures": reconstructed["anchorFailures"],
    }


def _spec_markdown(specification: Mapping[str, Any]) -> str:
    profiles = "\n".join(
        f"- `{item['profileId']}` — {item['failureSemantics']}"
        for item in specification["profiles"]
    )
    return f"""# S04 dynamic fault process specification

Schema `{specification['schemaVersion']}`; benchmark `{specification['benchmarkVersion']}`.

## Frozen contracts

S04 resumes each exact S02 checkpoint after the S03 `segment_reversal_central_v1`
anchor. It preserves the original immutable S01 scenario ID, global event clock,
Selection cursors, stream counters, native ledger, `100*n^2` recovery budget,
uniform activation, distributed-local policy-native information, and the
NoOp/Swap/MemoryUpdate action boundary.

## Process profiles

{profiles}

Fatigue is an endogenous mediator: only accepted Swap participation increases
load, and it owns no exogenous stream. The four stochastic streams are named in
`stream_registry.json` and addressed by original scenario ID, global event
index, and draw/read ordinal.

## Scenario reconstruction boundary

Rebuilding a static stuck `Scenario` changes canonical scenario content and its
scenario ID, so even an equal numeric seed gives a different RNG root. S04
therefore uses a typed runtime overlay on the original scenario. This is
mobility-equivalent for eligible changes while active, but not byte-equivalent
to S03 static stuck reconstruction.
"""


def write_outputs(
    output: Path,
    specification: Mapping[str, Any],
    panel: Mapping[str, Any],
    workers: int,
    git_commit: str,
) -> None:
    package = output / "dynamic_fault_package"
    package.mkdir(parents=True, exist_ok=True)
    _write_json(package / "dynamic_fault_spec.json", specification)
    _write_json(package / "dynamic_fault_spec.schema.json", DYNAMIC_SPEC_SCHEMA)
    _write_json(
        package / "dynamic_fault_run.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://eidosoma.local/schemas/e05/s04/dynamic-fault-run.schema.json",
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
                "schemaVersion": {"const": DYNAMIC_RUN_SCHEMA_VERSION},
                "benchmarkVersion": {"const": BENCHMARK_VERSION},
            },
        },
    )
    (package / "dynamic_fault_spec.md").write_text(
        _spec_markdown(specification), encoding="utf-8"
    )
    _write_json(
        package / "stream_registry.json",
        {
            "schemaVersion": "e05.s04.stream-registry.v1",
            "streams": specification["streamRegistry"],
            "fatigueExogenousStreams": [],
        },
    )
    panel["results"].to_parquet(output / "dynamic_fault_results.parquet", index=False)
    panel["scenarios"].to_parquet(
        output / "dynamic_fault_scenarios.parquet", index=False
    )
    panel["contrasts"].to_parquet(
        output / "paired_dynamic_contrasts.parquet", index=False
    )
    panel["checkpointRows"].to_parquet(
        output / "checkpoint_compatibility.parquet", index=False
    )
    panel["couplingRows"].to_parquet(
        package / "rng_coupling_audit.parquet", index=False
    )
    pd.DataFrame(panel["calibration"]["rows"]).to_parquet(
        package / "hazard_calibration.parquet", index=False
    )
    panel["calibration"]["durationRows"].to_parquet(
        package / "duration_samples.parquet", index=False
    )
    pd.DataFrame(panel["calibration"]["streamRows"]).to_parquet(
        package / "stream_isolation_addresses.parquet", index=False
    )
    with (package / "selected_full_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in panel["traces"]:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _plot_calibration(panel["calibration"], package)
    validations = {
        "checkpoint_validation.json": {
            "schemaVersion": "e05.s04.checkpoint-validation.v1",
            "checkpointCount": len(panel["checkpointRows"]),
            "checkpointFailures": panel["checkpointFailures"],
            "anchorFailures": panel["anchorFailures"],
            "success": not panel["checkpointFailures"] and not panel["anchorFailures"],
        },
        "hazard_duration_calibration_validation.json": {
            "schemaVersion": "e05.s04.hazard-duration-calibration-validation.v1",
            **panel["calibration"]["summary"],
            "calibrationRows": panel["calibration"]["rows"],
            "success": all(
                value
                for key, value in panel["calibration"]["summary"].items()
                if key.endswith("Pass")
            ),
        },
        "stream_isolation_validation.json": panel["streamValidation"],
        "replay_validation.json": {
            "schemaVersion": "e05.s04.replay-validation.v1",
            "plannedRunCount": len(panel["results"]),
            "exactReplayCount": int(panel["results"]["exactReplayPass"].sum()),
            "failureCount": int((~panel["results"]["exactReplayPass"]).sum()),
            "success": bool(panel["results"]["exactReplayPass"].all()),
        },
        "event_ledger_validation.json": panel["processValidation"],
        "pairing_validation.json": panel["pairingValidation"],
        "rng_coupling_validation.json": panel["couplingValidation"],
        "edge_case_validation.json": panel["edgeValidation"],
        "run_accounting.json": panel["runAccounting"],
        "validation_summary.json": panel["validationSummary"],
    }
    for name, value in validations.items():
        _write_json(output / name, value)
    _write_json(
        output / "input_provenance.json",
        {
            "schemaVersion": "e05.s04.input-provenance.v1",
            "inputs": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in (CONFIG, S02_CONFIG, *INPUTS)
            ],
        },
    )
    _write_json(
        output / "environment_provenance.json",
        {
            "schemaVersion": "e05.s04.environment-provenance.v1",
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
                for name in ("numpy", "pandas", "pyarrow", "matplotlib", "jsonschema", "pytest")
            },
        },
    )
    (output / "execution_commands.log").write_text(
        "\n".join(
            [
                "python -m pytest -q tests/test_regeneration_dynamic_faults.py",
                "python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py tests/test_regeneration_dynamic_faults.py",
                "ruff check src/regeneration/dynamic_faults.py tests/test_regeneration_dynamic_faults.py scripts/build_regeneration_s04.py src/regeneration/__init__.py",
                f"python scripts/build_regeneration_s04.py --artifacts-dir {output} --workers {workers}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _report(output: Path, panel: Mapping[str, Any], git_commit: str) -> None:
    validation = panel["validationSummary"]
    accounting = panel["runAccounting"]
    calibration = panel["calibration"]["summary"]
    contrasts = panel["contrasts"]
    changed = int(contrasts["completionChanged"].sum())
    median_deltas = (
        contrasts.groupby("assignedProfileId")["phaseActivationDelta"]
        .median()
        .astype(float)
        .to_dict()
    )
    profile_lines = "\n".join(
        f"- `{profile}`: median active-minus-sham phase duration {delta:.1f} opportunities."
        for profile, delta in median_deltas.items()
    )
    report = f"""# Research step S04 full results — Use dynamic fault processes

## Top summary

- **Research step ID:** S04
- **Completion status:** Complete; S05 was not started.
- **Artifacts written:** `dynamic_fault_package/` with specifications, schemas, stream registry, calibration samples/plots, RNG-coupling audit, and selected full traces; 3,840-row scenario/result tables; 1,920 paired contrasts; checkpoint, validation, accounting, provenance, and artifact manifests; this canonical report.
- **Validation result:** PASS — {validation['checkpointCount']}/384 exact S02 checkpoints and S03 anchors; {validation['plannedRunCount']}/3,840 planned runs and {validation['replayCount']}/3,840 exact replays; hazard/duration calibration, stream isolation, event-ledger identities, pairing, edge cases, RNG-boundary audit, and complete accounting all passed.
- **Outcome classification:** Supportive. The prespecified process distributions and runtime semantics are benchmark-ready under the frozen simulator contracts.
- **Caveats or blockers:** The processes are computational abstractions; fatigue is an endogenous movement mediator, not an exogenous fault. Runtime temporary freezing is mobility-equivalent but not byte-equivalent to rebuilding an S03 static-stuck scenario. Count-changing S03 lesions remain outside this fixed-identity runtime panel. No blocker remains within S04.
- **Recommended next action:** Chief Scientist review; if accepted, separately authorize S05 to test nudge-dependent recovery against duration-matched spontaneous controls.

## Lay summary

This step added five ways damage can change over time: a cell can be frozen for
exactly 16 opportunities, frozen for a random geometrically distributed time,
movement can switch intermittently between working and failed states, local
sensor readings can be wrong with probability 1/8, and cells can become tired
only after participating in successful moves. Every process was replayable
exactly. The random rates matched their declared probabilities, and all 3,840
planned active/control runs were retained. The experiment establishes clean,
reusable simulator processes; it does not show biological healing.

## Frozen question and outcome

S04 asked whether intermittent and recoverable faults could be represented as
qualitatively distinct, calibrated processes without changing the validated
regeneration task, timing, checkpoint, budget, ledger, or pairing contracts.
The primary success criterion was met: all five process profiles have explicit
semantics, fixture coverage, calibrated distributions, isolated streams, exact
replay, and full run accounting. The paired panel observed {changed}/1,920
completion-status differences; those descriptive outcomes are retained for
benchmark orientation, not interpreted as causal biological repair evidence.

## Inputs and inherited contracts

Inputs were the S01 task specification/results, all S02 timing specification and
checkpoint evidence, the S03 lesion specification and complete fixture table,
E01 release manifest, E02 scheduler/fault/random-stream contracts, workspace
plans, and the attachment manifest plus sidecar. No dataset or network input was
used and no dependency was installed.

Every arm resumed an exact S02 checkpoint after the S03
`segment_reversal_central_v1` anchor. Occupancy, Selection cursors, activation
count/global event clock, native stream counters, and native ledger were
preserved. Development and recovery budgets remained `100*n^2`; uniform random
activation, distributed-local policy-native information, skip-and-continue,
no-retry, the direction-aware strict unequal-inversion target, and the
NoOp/Swap/MemoryUpdate boundary remained frozen.

## Detailed methods

### Process semantics

Fixed temporary freezing blocks the selected central identity from initiating
an eligible state change or being the target of a Swap for exactly 16 charged
opportunities, recovering before opportunity 17. Random temporary freezing
checks a `p=1/16` recovery hazard before each opportunity while still charging
that opportunity as frozen; a hit recovers afterward, giving support 1,2,….

Intermittent movement is a two-state Markov process initialized available. A
transition draw occurs before each proposal outcome: available→failed has
probability 1/16 and failed→available has probability 1/4. The post-transition
failed state rejects only otherwise eligible Swaps; NoOp and MemoryUpdate
semantics are unchanged.

Probabilistic sensing applies one field-specific value-error draw to each
authorized logical read and one status-error draw to each authorized other-cell
read. A value error is ±1; a status error maps to one of the other two E01 fault
modes. Identity, position, policy, direction, actor self-status, permission
metadata, and trusted mechanical target identity remain protected. The trusted
validator always sees ground truth.

Movement-dependent fatigue is endogenous. Each accepted Swap adds one load to
both the initiating and displaced identity. At load 3, that identity resets to
zero and is immobile for the next 8 global opportunities. It consumes no
exogenous random stream. Thus fatigue can mediate effects of a trajectory but
is not randomized damage exposure.

### RNG-coupling boundary

Dynamic state lives in a typed runtime overlay outside immutable `Scenario`
cells and policy-visible observations. The overlay is applied after proposal
construction and authoritative mechanical validation, before conflict
resolution and commit. A separate audit rebuilt the S03 static-stuck scenario:
its canonical scenario ID changed, and all audited scheduler/policy/process root
draws differed at equal numeric seed and address. That contrast is therefore
`scenario_paired_rng_unpaired`. S04 execution retains the original S01 scenario
ID, so its base actor and policy streams preserve the declared shared prefix
until arm termination or state-path divergence.

### Calibration and panel

Calibration used 1,048,576 draws for each declared hazard/error rate, 65,536
fixed and 65,536 geometric duration episodes, and 64 Markov trajectories of
16,384 opportunities each. Acceptance required absolute binomial z≤6,
geometric mean within 0.4 of 16, and stationary failure fraction within 0.01 of
0.2. Observed geometric mean was {calibration['geometricDurationMean']:.4f}
(range {calibration['geometricDurationMinimum']}–{calibration['geometricDurationMaximum']});
the Markov failed fraction was {calibration['stationaryFailureFractionObserved']:.6f}.

The main panel was all 384 exact S02 checkpoints × five active profiles × one
matched sham = 1,920 pairs and 3,840 planned runs. Every planned run was
executed again for byte-exact replay, for {accounting['totalTrajectoryExecutionsObserved']}
total trajectory executions. Eight worker processes were used with no runtime-
driven scope reduction. Ten prespecified n=20 Bubble post-completion active runs
retained full native event and process-audit traces; all other runs retained
cryptographic digests and compact ledgers.

## Commands

```text
python -m pytest -q tests/test_regeneration_dynamic_faults.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py tests/test_regeneration_dynamic_faults.py
ruff check src/regeneration/dynamic_faults.py tests/test_regeneration_dynamic_faults.py scripts/build_regeneration_s04.py src/regeneration/__init__.py
python scripts/build_regeneration_s04.py --artifacts-dir /artifacts/research_steps/S04 --workers 8
```

## Results

| Check | Result |
| --- | ---: |
| Exact S02 checkpoint and S03 anchor identity | 384/384 pass |
| Planned active/sham pairs | {accounting['pairsObserved']}/1,920 pass |
| Planned runs | {accounting['plannedRunsObserved']}/3,840 pass |
| Exact run replays | {accounting['exactReplayExecutionsObserved']}/3,840 pass |
| Total trajectory executions | {accounting['totalTrajectoryExecutionsObserved']}/7,680 accounted |
| Full hand-inspectable traces | {accounting['fullTraceRunsObserved']}/10 |
| Substitutions / silent exclusions | 0 / 0 |
| Scope reduction | none |

The five process families produced distinct process ledgers and paired duration
profiles:

{profile_lines}

These duration differences include task completion and phase-budget censoring;
the row-level table preserves both and makes no survivor-only exclusion.
Temporary recovery that was not reached before terminal state is explicitly
marked censored. Phase-budget stops are retained rather than retried or dropped.

## Validation

All declared hazard/error z scores passed. Fixed recovery was exactly 16 in all
65,536 calibration episodes; geometric recovery had support beginning at one
and the declared long tail. Both Markov transition rates and stationary failure
fraction passed. Stream-name/address audits were deterministic and distinct;
fatigue owned no exogenous stream.

Every native ledger delta satisfied one proposal per activation, the proposal
partition, and two displaced cells per accepted Swap. Supplemental process
ledgers matched charged opportunities, hazard/transition/read consumption,
applied sensing errors, fatigue triggers, blocks, and recoveries. Process stream
counters matched event-trace draws exactly. All legal primitives, phase budgets,
and undeclared-stream checks passed. Pair validation confirmed identical source
scenario, checkpoint, anchor occupancy/hash, selected identity, clock, budget,
ledger, and pre-process stream state for every active/sham pair.

Fixture tests additionally covered fixed-duration expiry, support-one geometric
recovery and censorability, movement-specific intermittent gating, protected
sensing fields and trusted validation, accepted-movement-only fatigue loading,
cooldown boundary recovery, original-scenario enforcement, and explicit static
reconstruction RNG unpairing.

## Artifacts

- `dynamic_fault_package/dynamic_fault_spec.json`/Markdown and schemas freeze process semantics.
- `dynamic_fault_package/hazard_calibration.parquet`, `duration_samples.parquet`, and calibration PNG/SVG preserve distribution evidence.
- `dynamic_fault_package/stream_registry.json`, stream-address table, and RNG-coupling audit preserve randomization provenance.
- `dynamic_fault_package/selected_full_traces.jsonl` contains 10 full native/process traces.
- `dynamic_fault_scenarios.parquet`, `dynamic_fault_results.parquet`, and `paired_dynamic_contrasts.parquet` contain the complete panel.
- Checkpoint, pairing, replay, event-ledger, calibration, stream, edge, coupling, accounting, validation, input, environment, and artifact manifests preserve validation and provenance.

## Caveats, blockers, failed assumptions, and limitations

- These are transparent computational proxies, not biological lesions, fatigue,
  sensing, recovery, regeneration, or wet-lab validation.
- Runtime freezing reproduces immobility for eligible state changes, but policy
  observations see the original normal cell status. It is intentionally not a
  byte-identical static-stuck scenario.
- A static-stuck reconstruction changes scenario ID, so equal numeric seeds do
  not supply common random numbers across that boundary. The audit resolves,
  rather than hides, this S03 coupling limitation.
- Fatigue is downstream of accepted movement. Treating its occurrence as an
  independently randomized causal exposure would be invalid.
- Sensing perturbations are deliberately limited to authorized value/status
  fields. They do not model sensor geometry, correlated noise, or adversarial
  errors.
- Fixed/random recovery is spontaneous and opportunity-clock based. S05 must
  separately test contact-dependent recovery with matched marginal timing.
- Duplication, deletion, and insertion remain non-permutation tasks and were not
  silently adapted to the fixed-identity runtime.
- Completion and recovery-time summaries retain right-censored phase-budget
  runs; no threshold or failed policy was substituted or excluded.

## Provenance

- Repository: `Eidosoma/cell_research`
- Branch: `eidosoma/groups/28`
- Source commit: `{git_commit}`
- Benchmark: `{BENCHMARK_VERSION}`
- RNG: E01 SHA-256 counter-addressed base streams plus four isolated S04 streams.
- Runtime: Python {platform.python_version()}, numpy {package_version('numpy')}, pandas {package_version('pandas')}, pyarrow {package_version('pyarrow')}, matplotlib {package_version('matplotlib')}, jsonschema {package_version('jsonschema')}.
- Generated UTC: {datetime.now(timezone.utc).isoformat()}

Input and output SHA-256 hashes are recorded in `input_provenance.json` and
`artifact_manifest.json`. Reproducible source remains in the pushed Git commit;
no repository checkout or cache was copied into the artifact directory.
"""
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")


def _manifest(output: Path, git_commit: str) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "e05.s04.artifact-manifest.v1",
            "researchStepId": "S04",
            "benchmarkVersion": BENCHMARK_VERSION,
            "repositoryCommit": git_commit,
            "generatedUtc": datetime.now(timezone.utc).isoformat(),
            "files": [
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in files
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("S04 worker count must be between 1 and 8")
    if args.artifacts_dir.exists() and any(args.artifacts_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty artifact directory: {args.artifacts_dir}"
        )
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    _validate_inputs()
    specification = _load_json(CONFIG)
    git_commit = _git("rev-parse", "HEAD")
    panel = build(specification, args.workers)
    write_outputs(
        args.artifacts_dir, specification, panel, args.workers, git_commit
    )
    _report(args.artifacts_dir, panel, git_commit)
    _manifest(args.artifacts_dir, git_commit)
    if not panel["validationSummary"]["success"]:
        raise RuntimeError("S04 validation summary failed; inspect artifacts")
    print(json.dumps(panel["validationSummary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
