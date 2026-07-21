"""Outcome-blind materializers and native baseline runners for E07 S02.

Each function constructs only an E07 development/validation fixture from
public parameters.  Protected predecessor outcomes are neither imported nor
addressable from this module.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

from analysis.aggregation_metrics import aggregation_metrics
from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.costing import build_complete_cost_ledger
from causal_simulator.faults import (
    FaultExecutionContract,
    MobilityProfile,
    exact_replay_fault,
    run_faulted_architecture,
)
from causal_simulator.schedulers import SchedulerExecutionContract, SchedulerFamily
from reference_simulator import (
    Direction,
    FaultMode,
    Policy,
    create_scenario,
    exact_replay,
    run_scenario,
)
from reference_simulator.model import Scenario
from src.detours.distances import distance_profile
from src.morph2d import load_engine_context, run_cpu_episode
from src.morph2d.engine import canonical_episode_result_bytes
from src.regeneration.benchmark import run_fresh_core_smoke
from src.regeneration.chimeric import build_composition_scenario
from src.regeneration.target_change import (
    SignalPermission,
    TargetArm,
    TargetChange,
    TargetChangeContract,
    exact_replay_target_change,
    run_target_change_phase,
)
from src.regeneration.tasks import initial_checkpoint

from .contracts import (
    EvaluationAction,
    NativeEpisodeResult,
    ScenarioRecord,
    SuiteValidationError,
    canonical_sha256,
)


REPOSITORY = Path(__file__).resolve().parents[2]
Runner = Callable[[ScenarioRecord, EvaluationAction], NativeEpisodeResult]


def baseline_policy_hash(policy_id: str) -> str:
    return canonical_sha256(
        "E07/S02/native-baseline-policy/v1", {"policyId": policy_id}
    )


def _validate_action(record: ScenarioRecord, action: EvaluationAction) -> None:
    expected = str(record.public_parameters["baselinePolicyId"])
    if action.policy_id != expected:
        raise SuiteValidationError(
            f"scenario baseline policy is {expected!r}, got {action.policy_id!r}"
        )
    if action.policy_sha256 != baseline_policy_hash(expected):
        raise SuiteValidationError("native baseline policy hash mismatch")


def _line_scenario(record: ScenarioRecord, *, faults: bool = False) -> Scenario:
    values = tuple(int(value) for value in record.public_parameters["values"])
    policy = Policy(str(record.public_parameters["nativePolicy"]))
    direction = Direction(str(record.public_parameters.get("direction", "ascending")))
    fault_map = None
    if faults:
        index = int(record.public_parameters["faultIndex"])
        fault_map = {index: FaultMode(str(record.public_parameters["faultMode"]))}
    return create_scenario(
        values,
        policy=policy,
        direction=direction,
        seed=int(record.public_parameters["seed"]),
        max_activations=int(record.public_parameters["eventBudget"]),
        generation_key=f"E07/S02/{record.scenario_id}",
        permute=bool(record.public_parameters.get("permute", True)),
        faults=fault_map,
    )


def run_sorting(
    record: ScenarioRecord, action: EvaluationAction
) -> NativeEpisodeResult:
    _validate_action(record, action)
    scenario = _line_scenario(record)
    result = run_scenario(scenario, trace_mode="digest")
    exact_replay(result)
    summary = dict(result.summary)
    return NativeEpisodeResult(
        stop_reason=str(summary["stopReason"]),
        censored=summary["stopReason"] == "event_budget",
        failed=summary["stopReason"] == "invariant_error",
        native_costs={"e01ReferenceLedger": dict(summary["ledger"])},
        native_event={
            "schemaVersion": "E01.reference-event-pre-S06.v1",
            "eventDigest": result.event_digest,
            "activationCount": int(summary["activationCount"]),
            "traceMode": "digest",
        },
        native_outcome={
            "completed": bool(summary["completed"]),
            "finalValues": list(summary["finalValues"]),
            "finalStateHash": result.final_state_hash,
        },
        replay_pass=True,
        validation={"exactReplay": True, "nativeLedgerIdentity": True},
        provenance={
            "predecessor": "E01",
            "semanticsVersion": str(summary["semanticsVersion"]),
        },
    )


def run_faults(record: ScenarioRecord, action: EvaluationAction) -> NativeEpisodeResult:
    _validate_action(record, action)
    scenario = _line_scenario(record, faults=True)
    architecture = ArchitectureExecutionContract.distributed_local()
    scheduler = SchedulerExecutionContract(SchedulerFamily.UNIFORM_RANDOM_ACTIVATION)
    fault = FaultExecutionContract(
        mobility=MobilityProfile(str(record.public_parameters["faultMode"]))
    )
    run = run_faulted_architecture(
        scenario, architecture, scheduler, fault, trace_mode="digest"
    )
    exact_replay_fault(run)
    ledger = build_complete_cost_ledger(run, wall_time_seconds=0.0)
    summary = dict(run.result.summary)
    deterministic_costs = {
        key: value for key, value in ledger.costs.items() if key != "wallTimeSeconds"
    }
    return NativeEpisodeResult(
        stop_reason=str(summary["stopReason"]),
        censored=summary["stopReason"] == "event_budget",
        failed=summary["stopReason"] in {"invariant_error", "blocking_failure"},
        native_costs={
            "e01ReferenceLedger": dict(summary["ledger"]),
            "e02MechanismLedger": deterministic_costs,
        },
        native_event={
            "schemaVersion": "E02.fault-run.v1",
            "eventDigest": run.result.event_digest,
            "activationCount": int(summary["activationCount"]),
            "faultAuditCount": len(run.sensing_audit)
            + len(run.action_failure_audit)
            + len(run.retry_audit)
            + len(run.blocking_failure_audit),
        },
        native_outcome={
            "completed": bool(summary["completed"]),
            "finalValues": list(summary["finalValues"]),
            "faultProfile": fault.to_dict(),
            "comparability": "matched_policy_native_local",
        },
        replay_pass=True,
        validation={
            "exactReplay": True,
            **{key: bool(value) for key, value in run.opportunity_validation().items()},
            **{key: bool(value) for key, value in ledger.validation.items()},
        },
        provenance={
            "predecessor": "E02",
            "faultInterfaceVersion": "E02-matched-faults-v1",
        },
    )


def run_detour(record: ScenarioRecord, action: EvaluationAction) -> NativeEpisodeResult:
    _validate_action(record, action)
    scenario = _line_scenario(record)
    initial_values = [
        scenario.cell_map[item].value for item in scenario.initial_occupancy
    ]
    result = run_scenario(scenario, trace_mode="digest")
    exact_replay(result)
    summary = dict(result.summary)
    initial = distance_profile(initial_values, direction=scenario.cells[0].direction)
    final = distance_profile(
        summary["finalValues"], direction=scenario.cells[0].direction
    )
    return NativeEpisodeResult(
        stop_reason=str(summary["stopReason"]),
        censored=summary["stopReason"] == "event_budget",
        failed=summary["stopReason"] == "invariant_error",
        native_costs={"e01ReferenceLedger": dict(summary["ledger"])},
        native_event={
            "schemaVersion": "E03.empirical-distance-overlay.v1",
            "eventDigest": result.event_digest,
            "activationCount": int(summary["activationCount"]),
            "onlineMetricExposure": False,
        },
        native_outcome={
            "completed": bool(summary["completed"]),
            "initialDistanceProfile": initial.to_dict(),
            "finalDistanceProfile": final.to_dict(),
            "metricAggregationPermitted": False,
            "necessaryDetourClaim": False,
        },
        replay_pass=True,
        validation={"exactReplay": True, "metricsOfflineOnly": True},
        provenance={
            "predecessor": "E03",
            "distanceSpecification": "e03.s01.distance-spec.v1",
        },
    )


def run_chimera(
    record: ScenarioRecord, action: EvaluationAction
) -> NativeEpisodeResult:
    _validate_action(record, action)
    n = int(record.public_parameters["n"])
    scenario, metadata = build_composition_scenario(
        n=n,
        portfolio_id=str(record.public_parameters["portfolioId"]),
        direction=str(record.public_parameters["direction"]),
        replicate=int(record.public_parameters["replicate"]),
        placement_map=int(record.public_parameters["placementMap"]),
        max_activations=int(record.public_parameters["eventBudget"]),
    )
    initial_labels = [
        scenario.cell_map[item].policy.value for item in scenario.initial_occupancy
    ]
    initial_values = [
        scenario.cell_map[item].value for item in scenario.initial_occupancy
    ]
    result = run_scenario(scenario, trace_mode="digest")
    exact_replay(result)
    final_occupancy = tuple(result.final_state["occupancy"])
    final_labels = [scenario.cell_map[item].policy.value for item in final_occupancy]
    final_values = [scenario.cell_map[item].value for item in final_occupancy]
    initial_metrics = aggregation_metrics(initial_labels, initial_values)
    final_metrics = aggregation_metrics(final_labels, final_values)
    summary = dict(result.summary)
    return NativeEpisodeResult(
        stop_reason=str(summary["stopReason"]),
        censored=summary["stopReason"] == "event_budget",
        failed=summary["stopReason"] == "invariant_error",
        native_costs={"e01ReferenceLedger": dict(summary["ledger"])},
        native_event={
            "schemaVersion": "E04.composition-corrected-overlay.v1",
            "eventDigest": result.event_digest,
            "activationCount": int(summary["activationCount"]),
            "policyLabelsVisibleToPolicy": False,
        },
        native_outcome={
            "completed": bool(summary["completed"]),
            "compositionCounts": dict(Counter(initial_labels)),
            "initialAggregationMetrics": initial_metrics,
            "finalAggregationMetrics": final_metrics,
            "aggregationMetricsPooled": False,
            "causalAggregationClaim": False,
        },
        replay_pass=True,
        validation={
            "exactReplay": True,
            "compositionConserved": Counter(initial_labels) == Counter(final_labels),
            "labelsOfflineOnly": True,
        },
        provenance={
            "predecessor": "E04",
            "metricContract": "e04.s04.aggregation_metrics.v1",
            "policyAssignmentSha256": str(metadata["policyAssignmentSha256"]),
        },
    )


def run_regeneration(
    record: ScenarioRecord, action: EvaluationAction
) -> NativeEpisodeResult:
    _validate_action(record, action)
    specification = json.loads(
        (REPOSITORY / "configs/regeneration/s14_regeneration_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    specification["smokeAndReproduction"]["freshScenario"]["replicateOrdinal"] = int(
        record.public_parameters["replicateOrdinal"]
    )
    smoke = run_fresh_core_smoke(specification)
    validations = {
        "developmentReplay": bool(smoke["developmentReplayPass"]),
        "stabilizationReplay": bool(smoke["stabilizationReplayPass"]),
        "recoveryReplay": bool(smoke["recoveryReplayPass"]),
        "runtimeValidation": bool(smoke["allRuntimeValidationPass"]),
    }
    return NativeEpisodeResult(
        stop_reason=str(smoke["stopReason"]),
        censored=not bool(smoke["success"]),
        failed=not all(validations.values()),
        native_costs={
            "phaseOpportunityLedger": {
                "developmentOpportunities": int(smoke["developmentActivationCount"]),
                "stabilizationOpportunities": int(smoke["stabilizationOpportunities"]),
                "recoveryOpportunities": int(smoke["phaseActivationCount"]),
            }
        },
        native_event={
            "schemaVersion": str(smoke["schemaVersion"]),
            "developmentEventDigest": str(smoke["developmentEventDigest"]),
            "resultDigest": str(smoke["resultDigest"]),
            "phaseBoundariesRetained": True,
        },
        native_outcome={
            key: smoke[key]
            for key in (
                "success",
                "restrictedTime",
                "initialDistance",
                "finalDistance",
                "lesionWindowLength",
                "lesionPostDistance",
            )
        },
        replay_pass=all(
            validations[key]
            for key in ("developmentReplay", "stabilizationReplay", "recoveryReplay")
        ),
        validation=validations,
        provenance={
            "predecessor": "E05",
            "benchmarkVersion": "E05-regeneration-benchmark-v1",
            "sourceScenarioId": str(smoke["sourceScenarioId"]),
        },
    )


def run_target_change(
    record: ScenarioRecord, action: EvaluationAction
) -> NativeEpisodeResult:
    _validate_action(record, action)
    n = int(record.public_parameters["n"])
    scenario = create_scenario(
        tuple(range(n)),
        policy=Policy.BUBBLE,
        direction=Direction.ASCENDING,
        seed=int(record.public_parameters["seed"]),
        max_activations=3 * int(record.public_parameters["adaptationBudget"]),
        generation_key=f"E07/S02/{record.scenario_id}",
        permute=False,
    )
    checkpoint = initial_checkpoint(scenario)
    contract = TargetChangeContract(
        TargetArm.CHANGED_AWARE,
        TargetChange(str(record.public_parameters["targetChangeId"])),
        SignalPermission(str(record.public_parameters["signalPermissionId"])),
    )
    adaptation_budget = int(record.public_parameters["adaptationBudget"])
    probe_budget = int(record.public_parameters["probeBudget"])
    run = run_target_change_phase(
        scenario,
        checkpoint,
        contract=contract,
        adaptation_budget=adaptation_budget,
        probe_budget=probe_budget,
        retain_trace=False,
    )
    exact_replay_target_change(
        run, scenario, checkpoint, adaptation_budget, probe_budget
    )
    summary = dict(run.summary)
    return NativeEpisodeResult(
        stop_reason=str(summary["stopReason"]),
        censored=bool(summary["adaptationCensored"]),
        failed=not all(run.opportunity_validation.values()),
        native_costs={
            "e01ReferenceLedgerDelta": dict(summary["ledgerDelta"]),
            "e05TargetSignalLedger": dict(run.process_ledger),
        },
        native_event={
            "schemaVersion": "e05.s09.target-change-run.v1",
            "eventDigest": run.event_digest,
            "startEventIndex": int(run.start_event_index),
            "endEventIndex": int(run.end_event_index),
            "targetChangeInstantaneousBeforeNextOpportunity": True,
        },
        native_outcome={
            key: summary[key]
            for key in (
                "targetCompleted",
                "phaseActivationCount",
                "initialOldTargetDistance",
                "initialNewTargetDistance",
                "finalOldTargetDistance",
                "finalNewTargetDistance",
                "adaptationTime",
                "adaptationCensored",
                "restrictedAdaptationTime",
                "overshootCensored",
                "postHitAnyDeparture",
            )
        },
        replay_pass=True,
        validation={
            "exactReplay": True,
            **{key: bool(value) for key, value in run.opportunity_validation.items()},
        },
        provenance={
            "predecessor": "E05",
            "benchmarkVersion": "E05-collective-target-change-v1",
            "targetHash": run.target_definition.target_hash,
        },
    )


def _morph_context():
    return load_engine_context(
        REPOSITORY / "configs/morphologies/engine_catalog.yaml",
        environment_catalog=REPOSITORY
        / "configs/morphologies/environment_catalog.yaml",
        policy_catalog=REPOSITORY / "configs/morphologies/policy_catalog.yaml",
        grammar_catalog=REPOSITORY / "configs/morphologies/grammar_catalog.yaml",
        channel_catalog=REPOSITORY
        / "configs/morphologies/control_channel_catalog.yaml",
    )


def run_spatial(
    record: ScenarioRecord, action: EvaluationAction
) -> NativeEpisodeResult:
    _validate_action(record, action)
    context = _morph_context()
    scenario_id = str(record.public_parameters["nativeScenarioId"])
    definition = next(
        item for item in context.episodes if item.scenario_id == scenario_id
    )
    first = run_cpu_episode(context, definition, include_selected_traces=False)
    second = run_cpu_episode(context, definition, include_selected_traces=False)
    replay_pass = canonical_episode_result_bytes(
        first
    ) == canonical_episode_result_bytes(second)
    validation = {
        "exactReplay": replay_pass,
        "permissionAudit": all(
            not value for value in first["permissionAudit"].values()
        ),
        "fixedTransitionCount": len(first["transitionSummaries"])
        == int(first["transitionBudget"]),
        "configurationChargedOnce": bool(
            first["configurationAccounting"]["chargedOnceInFull"]
        )
        and not bool(first["configurationAccounting"]["amortizedOrDivided"]),
    }
    return NativeEpisodeResult(
        stop_reason=str(first["stopReason"]),
        censored=False,
        failed=not all(validation.values()),
        native_costs={
            "e06MovementLedger": dict(first["movementLedger"]),
            "e06ObservationLedger": dict(first["observationLedger"]),
            "e06ChannelLedger": dict(first["channelLedger"]),
        },
        native_event={
            "schemaVersion": str(first["schemaVersion"]),
            "episodeSha256": str(first["episodeSha256"]),
            "transitionCount": len(first["transitionSummaries"]),
            "actorBatchSize": int(first["actorBatchSize"]),
            "onlineGlobalCompletionComputed": False,
        },
        native_outcome={
            "fixedBudgetCompleted": True,
            "finalStateSha256": str(first["finalState"]["stateSha256"]),
            "evaluationSeparation": dict(first["evaluationSeparation"]),
            "formationOrRepairClaim": False,
        },
        replay_pass=replay_pass,
        validation=validation,
        provenance={
            "predecessor": "E06",
            "engineVersion": "e06-morph2d-hybrid-engine-v1",
            "nativeScenarioId": scenario_id,
        },
    )


RUNNERS: Mapping[str, Runner] = {
    "e01_sorting": run_sorting,
    "e02_faults": run_faults,
    "e03_detour_metrics": run_detour,
    "e04_chimera": run_chimera,
    "e05_regeneration": run_regeneration,
    "e05_target_change": run_target_change,
    "e06_spatial": run_spatial,
}


def runner_source_hashes() -> dict[str, str]:
    path = Path(__file__)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {runner_id: digest for runner_id in sorted(RUNNERS)}
