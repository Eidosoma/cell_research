"""Outcome-blind materializers and native baseline runners for E07 S02.

Each function constructs only an E07 development/validation fixture from
public parameters.  Protected predecessor outcomes are neither imported nor
addressable from this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

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
from .dsl_adapters import (
    LineDslRuntime,
    bind_homogeneous_line_scenario,
    compiled_policies,
    line_replay_bytes,
    run_e05_regeneration_dsl,
    run_e05_target_change_dsl,
    run_line_dsl_episode,
    run_spatial_dsl_episode,
)

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
    if action.mode == "dsl_episode":
        if record.split.value != "train":
            raise SuiteValidationError(
                "S04A DSL adapters are authorized for the frozen training split only"
            )
        return
    expected = str(record.public_parameters["baselinePolicyId"])
    if action.policy_id != expected:
        raise SuiteValidationError(
            f"scenario baseline policy is {expected!r}, got {action.policy_id!r}"
        )
    if action.policy_sha256 != baseline_policy_hash(expected):
        raise SuiteValidationError("native baseline policy hash mismatch")


def _normalized_kendall(values: list[int | float], direction: Direction) -> float:
    profile = distance_profile(values, direction=direction)
    return float(profile.normalized_kendall_distance)


def _line_common(
    result, runtime, *, direction: Direction
) -> tuple[dict[str, Mapping[str, int | float]], dict[str, Any], dict[str, Any]]:
    summary = dict(result.summary)
    adapter = runtime.adapter_ledgers()
    opportunities = int(runtime.ledger["nativeEligibleActionOpportunities"])
    committed = int(runtime.ledger["committedNativeMovementActions"])
    displacement = int(runtime.ledger["committedNativeDisplacement"])
    maximum = max(1, len(summary["finalValues"]) - 1)
    descriptors = {
        "acceptedNativeActionFraction": committed / opportunities
        if opportunities
        else 0.0,
        "committedDisplacementFraction": (
            displacement / (committed * maximum) if committed else 0.0
        ),
    }
    costs = {
        "e01ReferenceLedger": dict(summary["ledger"]),
        **adapter,
    }
    event = {
        "schemaVersion": "e07.s04a.line-dsl-event.v1",
        "eventDigest": result.event_digest,
        "activationCount": int(summary["activationCount"]),
        "policyControlledEpisode": True,
        "decisionCounts": dict(sorted(runtime.decision_counts.items())),
        "descriptors": descriptors,
    }
    outcome = {
        "completed": bool(summary["completed"]),
        "finalValues": list(summary["finalValues"]),
        "finalStateHash": result.final_state_hash,
        "derived": {
            "finalNormalizedKendallDistance": _normalized_kendall(
                list(summary["finalValues"]), direction
            )
        },
        "descriptors": descriptors,
    }
    return costs, event, outcome


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
    if action.mode == "dsl_episode":
        policies = compiled_policies(action)
        if len(policies) != 1:
            raise SuiteValidationError("sorting accepts one DSL policy")
        scenario = bind_homogeneous_line_scenario(
            scenario, next(iter(policies.values()))
        )
        result, runtime = run_line_dsl_episode(scenario, action)
        replay, replay_runtime = run_line_dsl_episode(scenario, action)
        replay_pass = line_replay_bytes(result, runtime) == line_replay_bytes(
            replay, replay_runtime
        )
        costs, event, outcome = _line_common(
            result, runtime, direction=scenario.cells[0].direction
        )
        return NativeEpisodeResult(
            stop_reason=str(result.summary["stopReason"]),
            censored=result.summary["stopReason"] == "event_budget",
            failed=(
                result.summary["stopReason"] == "invariant_error" or not replay_pass
            ),
            native_costs=costs,
            native_event=event,
            native_outcome=outcome,
            replay_pass=replay_pass,
            validation={
                "exactReplay": replay_pass,
                "policyControlledEpisode": True,
                "nativeLedgerIdentity": int(result.summary["ledger"]["activations"])
                == int(result.summary["ledger"]["proposals"]),
            },
            provenance={
                "predecessor": "E01",
                "semanticsVersion": str(result.summary["semanticsVersion"]),
                "adapterVersion": "e07.s04a.dsl-native-adapters.v1",
            },
        )
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
    if action.mode == "dsl_episode":
        policies = compiled_policies(action)
        if len(policies) != 1:
            raise SuiteValidationError("fault task accepts one DSL policy")
        policy = next(iter(policies.values()))
        scenario = bind_homogeneous_line_scenario(scenario, policy)

        def execute(active_scenario, active_fault):
            active_runtime = LineDslRuntime({active_scenario.cells[0].policy: policy})
            active_run = run_faulted_architecture(
                active_scenario,
                architecture,
                scheduler,
                active_fault,
                trace_mode="full",
                action_interface=active_runtime,
                after_batch_observer=active_runtime,
            )
            return active_run, active_runtime

        run, runtime = execute(scenario, fault)
        replay, replay_runtime = execute(scenario, fault)
        fault_replay_pass = (
            run.to_json_bytes() == replay.to_json_bytes()
            and runtime.adapter_ledgers() == replay_runtime.adapter_ledgers()
        )
        clean_scenario = Scenario.create(
            tuple(replace(cell, fault=FaultMode.NORMAL) for cell in scenario.cells),
            initial_occupancy=scenario.initial_occupancy,
            initial_selection_cursors=dict(scenario.initial_selection_cursors),
            seed=scenario.seed,
            max_activations=scenario.max_activations,
            architecture=scenario.architecture,
            batch_width=scenario.batch_width,
            generation_key=f"{scenario.generation_key}/matched-clean",
            fault_placement="explicit",
            requested_fault_count=0,
        )
        clean_fault = FaultExecutionContract()
        clean, clean_runtime = execute(clean_scenario, clean_fault)
        clean_replay, clean_replay_runtime = execute(clean_scenario, clean_fault)
        clean_replay_pass = (
            clean.to_json_bytes() == clean_replay.to_json_bytes()
            and clean_runtime.adapter_ledgers()
            == clean_replay_runtime.adapter_ledgers()
        )
        replay_pass = fault_replay_pass and clean_replay_pass
        ledger = build_complete_cost_ledger(run, wall_time_seconds=0.0)
        clean_ledger = build_complete_cost_ledger(clean, wall_time_seconds=0.0)
        summary = dict(run.result.summary)
        clean_summary = dict(clean.result.summary)
        costs, event, outcome = _line_common(
            run.result, runtime, direction=scenario.cells[0].direction
        )
        costs["e02MechanismLedger"] = {
            key: value
            for key, value in ledger.costs.items()
            if key != "wallTimeSeconds"
        }
        costs["e01MatchedCleanReferenceLedger"] = dict(clean_summary["ledger"])
        costs["e02MatchedCleanMechanismLedger"] = {
            key: value
            for key, value in clean_ledger.costs.items()
            if key != "wallTimeSeconds"
        }
        fault_residual = float(outcome["derived"]["finalNormalizedKendallDistance"])
        clean_residual = _normalized_kendall(
            list(clean_summary["finalValues"]),
            clean_scenario.cells[0].direction,
        )
        outcome.update(
            {
                "faultProfile": fault.to_dict(),
                "comparability": "matched_policy_native_local_dsl",
                "paired": {
                    "completedFaultMinusMatchedClean": int(bool(summary["completed"]))
                    - int(bool(clean_summary["completed"])),
                    "residualFaultMinusMatchedClean": fault_residual - clean_residual,
                    "matchedCleanCompleted": bool(clean_summary["completed"]),
                    "matchedCleanResidual": clean_residual,
                    "couplingClass": "scenario_paired_rng_unpaired",
                },
            }
        )
        event.update(
            {
                "schemaVersion": "e07.s04a.e02-dsl-fault-event.v1",
                "faultAuditCount": len(run.sensing_audit)
                + len(run.action_failure_audit)
                + len(run.retry_audit)
                + len(run.blocking_failure_audit),
            }
        )
        validations = {
            "exactReplay": replay_pass,
            "matchedCleanPairPresent": True,
            "matchedCleanInitialValuesAndOccupancy": scenario.initial_occupancy
            == clean_scenario.initial_occupancy
            and [cell.value for cell in scenario.cells]
            == [cell.value for cell in clean_scenario.cells],
            "rngCouplingDeclared": outcome["paired"]["couplingClass"]
            == "scenario_paired_rng_unpaired",
            **{key: bool(value) for key, value in run.opportunity_validation().items()},
            **{key: bool(value) for key, value in ledger.validation.items()},
            **{
                f"matchedClean_{key}": bool(value)
                for key, value in clean.opportunity_validation().items()
            },
            **{
                f"matchedCleanCost_{key}": bool(value)
                for key, value in clean_ledger.validation.items()
            },
        }
        return NativeEpisodeResult(
            stop_reason=str(summary["stopReason"]),
            censored=summary["stopReason"] == "event_budget",
            failed=summary["stopReason"] in {"invariant_error", "blocking_failure"}
            or not all(validations.values()),
            native_costs=costs,
            native_event=event,
            native_outcome=outcome,
            replay_pass=replay_pass,
            validation=validations,
            provenance={
                "predecessor": "E02",
                "faultInterfaceVersion": "E02-matched-faults-v1",
                "adapterVersion": "e07.s04a.dsl-native-adapters.v1",
            },
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
    if action.mode == "dsl_episode":
        policies = compiled_policies(action)
        if len(policies) != 1:
            raise SuiteValidationError("detour task accepts one DSL policy")
        scenario = bind_homogeneous_line_scenario(
            scenario, next(iter(policies.values()))
        )
        initial_values = [
            scenario.cell_map[item].value for item in scenario.initial_occupancy
        ]
        result, runtime = run_line_dsl_episode(scenario, action)
        replay, replay_runtime = run_line_dsl_episode(scenario, action)
        replay_pass = line_replay_bytes(result, runtime) == line_replay_bytes(
            replay, replay_runtime
        )
        occupancy = list(scenario.initial_occupancy)
        profiles = [
            distance_profile(initial_values, direction=scenario.cells[0].direction)
        ]
        for event in result.events:
            proposal = event["proposal"]
            if event["decision"] == "accepted" and proposal["kind"] == "Swap":
                left = int(proposal["actorPos"])
                right = int(proposal["targetPos"])
                occupancy[left], occupancy[right] = occupancy[right], occupancy[left]
            values_now = [scenario.cell_map[item].value for item in occupancy]
            profiles.append(
                distance_profile(values_now, direction=scenario.cells[0].direction)
            )
        metric_fields = {
            "adjacent_descents": "normalized_adjacent_descents",
            "inversion_count": "normalized_kendall_distance",
            "spearman_footrule": "normalized_spearman_footrule",
            "maximum_rank_error": "normalized_maximum_rank_error",
            "duplicate_aware_earth_movers_distance": "normalized_duplicate_aware_earth_movers_distance",
        }
        excursions = {}
        for metric_id, field in metric_fields.items():
            values = [float(getattr(item, field)) for item in profiles]
            start = values[0]
            excursions[metric_id] = {
                "maximumPositiveExcursionFraction": min(
                    1.0, max(0.0, max(values) - start)
                ),
                "worseningEventFraction": sum(
                    right > left for left, right in zip(values, values[1:])
                )
                / max(1, len(values) - 1),
            }
        costs, event, outcome = _line_common(
            result, runtime, direction=scenario.cells[0].direction
        )
        outcome.update(
            {
                "initialDistanceProfile": profiles[0].to_dict(),
                "finalDistanceProfile": profiles[-1].to_dict(),
                "metricAggregationPermitted": False,
                "necessaryDetourClaim": False,
                "metricSpecificExcursions": excursions,
            }
        )
        event.update(
            {
                "schemaVersion": "e07.s04a.e03-dsl-distance-event.v1",
                "onlineMetricExposure": False,
                "distanceProfileCount": len(profiles),
            }
        )
        return NativeEpisodeResult(
            stop_reason=str(result.summary["stopReason"]),
            censored=result.summary["stopReason"] == "event_budget",
            failed=result.summary["stopReason"] == "invariant_error" or not replay_pass,
            native_costs=costs,
            native_event=event,
            native_outcome=outcome,
            replay_pass=replay_pass,
            validation={
                "exactReplay": replay_pass,
                "metricsOfflineOnly": True,
                "metricFamiliesSeparated": set(excursions) == set(metric_fields),
            },
            provenance={
                "predecessor": "E03",
                "distanceSpecification": "e03.s01.distance-spec.v1",
                "adapterVersion": "e07.s04a.dsl-native-adapters.v1",
            },
        )
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
    if action.mode == "dsl_episode":
        expected_bindings = {item.policy.value for item in scenario.cells}
        if set(action.native_policy_bindings) != expected_bindings:
            raise SuiteValidationError(
                "E04 DSL portfolio must bind every native Algotype explicitly"
            )
        initial_labels = [
            scenario.cell_map[item].policy.value for item in scenario.initial_occupancy
        ]
        initial_values = [
            scenario.cell_map[item].value for item in scenario.initial_occupancy
        ]
        result, runtime = run_line_dsl_episode(scenario, action)
        replay, replay_runtime = run_line_dsl_episode(scenario, action)
        replay_pass = line_replay_bytes(result, runtime) == line_replay_bytes(
            replay, replay_runtime
        )
        counts = Counter(initial_labels)
        if len(counts) != 2:
            raise SuiteValidationError("S04A exact null fixture expects two labels")
        label_names = sorted(counts)
        first_count = counts[label_names[0]]
        null_values = []
        for selected in combinations(range(len(initial_labels)), first_count):
            chosen = set(selected)
            labels = [
                label_names[0] if index in chosen else label_names[1]
                for index in range(len(initial_labels))
            ]
            null_values.append(
                aggregation_metrics(labels, list(range(len(labels))))[
                    "corrected_edge_aggregation"
                ]
            )
        null_mean = sum(null_values) / len(null_values)
        occupancy = list(scenario.initial_occupancy)
        corrected_trace = []
        for event in result.events:
            proposal = event["proposal"]
            if event["decision"] == "accepted" and proposal["kind"] == "Swap":
                left = int(proposal["actorPos"])
                right = int(proposal["targetPos"])
                occupancy[left], occupancy[right] = occupancy[right], occupancy[left]
            labels = [scenario.cell_map[item].policy.value for item in occupancy]
            values = [scenario.cell_map[item].value for item in occupancy]
            corrected_trace.append(
                aggregation_metrics(labels, values)["corrected_edge_aggregation"]
                - null_mean
            )
        if not corrected_trace:
            corrected_trace = [
                aggregation_metrics(initial_labels, initial_values)[
                    "corrected_edge_aggregation"
                ]
                - null_mean
            ]
        final_labels = [scenario.cell_map[item].policy.value for item in occupancy]
        final_values = [scenario.cell_map[item].value for item in occupancy]
        initial_metrics = aggregation_metrics(initial_labels, initial_values)
        final_metrics = aggregation_metrics(final_labels, final_values)
        costs, event, outcome = _line_common(
            result, runtime, direction=scenario.cells[0].direction
        )
        outcome.update(
            {
                "compositionCounts": dict(counts),
                "initialAggregationMetrics": initial_metrics,
                "finalAggregationMetrics": final_metrics,
                "dynamicNull": {
                    "nullFamily": "exact_fixed_composition_position_permutation_train_v1",
                    "nullSupportSize": len(null_values),
                    "compositionCorrectedPeakExcess": max(corrected_trace),
                    "fixedHorizonPositiveAreaMean": sum(
                        max(0.0, value) for value in corrected_trace
                    )
                    / len(corrected_trace),
                },
                "aggregationMetricsPooled": False,
                "causalAggregationClaim": False,
            }
        )
        event.update(
            {
                "schemaVersion": "e07.s04a.e04-dsl-composition-event.v1",
                "policyLabelsVisibleToPolicy": False,
                "fullTraceMetricCount": len(corrected_trace),
            }
        )
        validations = {
            "exactReplay": replay_pass,
            "compositionConserved": Counter(initial_labels) == Counter(final_labels),
            "labelsOfflineOnly": True,
            "exactCompositionNull": len(null_values)
            == math.comb(len(initial_labels), first_count),
        }
        return NativeEpisodeResult(
            stop_reason=str(result.summary["stopReason"]),
            censored=result.summary["stopReason"] == "event_budget",
            failed=result.summary["stopReason"] == "invariant_error"
            or not all(validations.values()),
            native_costs=costs,
            native_event=event,
            native_outcome=outcome,
            replay_pass=replay_pass,
            validation=validations,
            provenance={
                "predecessor": "E04",
                "metricContract": "e04.s04.aggregation_metrics.v1",
                "policyAssignmentSha256": str(metadata["policyAssignmentSha256"]),
                "adapterVersion": "e07.s04a.dsl-native-adapters.v1",
            },
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
    if action.mode == "dsl_episode":
        first = run_e05_regeneration_dsl(
            action,
            replicate_ordinal=int(record.public_parameters["replicateOrdinal"]),
        )
        second = run_e05_regeneration_dsl(
            action,
            replicate_ordinal=int(record.public_parameters["replicateOrdinal"]),
        )
        replay_pass = canonical_sha256(
            "E07/S04A/E05-replay/v1", first
        ) == canonical_sha256("E07/S04A/E05-replay/v1", second)
        checks = {key: bool(value) for key, value in first["validation"].items()}
        checks["exactReplay"] = replay_pass
        axes = first["competencyAxes"]
        repair = axes.get("repair", {})
        native_ledgers = {
            f"e05_{phase}": ledger for phase, ledger in first["nativeLedgers"].items()
        }
        for name, ledger in first.get("extensionLedgers", {}).items():
            native_ledgers[f"e05Extension_{name}"] = ledger
        return NativeEpisodeResult(
            stop_reason=(
                "source_terminal"
                if first["sourceTerminal"]
                else str(first["phaseRows"][-1]["stopReason"])
            ),
            censored=bool(repair.get("recoveryCensored", first["sourceTerminal"])),
            failed=not all(checks.values()),
            native_costs=native_ledgers,
            native_event={
                "schemaVersion": str(first["schemaVersion"]),
                "panelSha256": str(first.get("panelSha256", "source-terminal")),
                "phaseRows": first["phaseRows"],
                "stoppedRows": first["stoppedRows"],
                "fiveAxesNonaggregated": True,
                "descriptorsByAxis": first.get("descriptorsByAxis", {}),
                "nativeMovementDescriptorsByPhase": first.get(
                    "nativeMovementDescriptorsByPhase", {}
                ),
            },
            native_outcome={
                "success": bool(repair.get("recoverySuccess", False)),
                "restrictedTime": int(
                    repair.get("restrictedRecoveryTime", 100 * 32 * 32 + 1)
                ),
                "initialDistance": int(repair.get("initialDistance", 0)),
                "finalDistance": int(repair.get("finalDistance", 0)),
                "e05CompetencyAxes": axes,
                "sourceTerminal": bool(first["sourceTerminal"]),
                "aggregateCompetencyScore": None,
                "descriptorsByAxis": first.get("descriptorsByAxis", {}),
                "nativeMovementDescriptorsByPhase": first.get(
                    "nativeMovementDescriptorsByPhase", {}
                ),
            },
            replay_pass=replay_pass,
            validation=checks,
            provenance={
                "predecessor": "E05",
                "benchmarkVersion": "E05-regeneration-benchmark-v1",
                "adapterVersion": "e07.s04a.dsl-native-adapters.v1",
                "sourceScenarioId": str(
                    first.get("sourceScenarioId", "source-terminal")
                ),
            },
        )
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
    if action.mode == "dsl_episode":
        first = run_e05_target_change_dsl(
            scenario,
            action,
            target_change_id=str(record.public_parameters["targetChangeId"]),
            signal_permission_id=str(record.public_parameters["signalPermissionId"]),
            adaptation_budget=int(record.public_parameters["adaptationBudget"]),
            probe_budget=int(record.public_parameters["probeBudget"]),
        )
        second = run_e05_target_change_dsl(
            scenario,
            action,
            target_change_id=str(record.public_parameters["targetChangeId"]),
            signal_permission_id=str(record.public_parameters["signalPermissionId"]),
            adaptation_budget=int(record.public_parameters["adaptationBudget"]),
            probe_budget=int(record.public_parameters["probeBudget"]),
        )
        replay_pass = first == second
        checks = {key: bool(value) for key, value in first["validation"].items()}
        checks["exactReplay"] = replay_pass
        return NativeEpisodeResult(
            stop_reason=str(first["stopReason"]),
            censored=bool(first["adaptationCensored"]),
            failed=not all(checks.values()),
            native_costs={
                "e01ReferenceLedgerDelta": first["nativeLedger"],
                "e05TargetSignalLedger": first["targetSignalLedger"],
                "dslRuntimeLedger": first["dslLedgers"]["dslRuntimeLedger"],
                "dslCommunicationLedger": first["dslLedgers"]["dslCommunicationLedger"],
            },
            native_event={
                "schemaVersion": str(first["schemaVersion"]),
                "resultSha256": str(first["resultSha256"]),
                "targetChangeInstantaneousBeforeNextOpportunity": True,
                "targetSignalAuthority": first["targetSignalAuthority"],
                "descriptorsByAxis": first["descriptorsByAxis"],
                "nativeMovementDescriptors": first["nativeMovementDescriptors"],
            },
            native_outcome={
                key: first[key]
                for key in (
                    "targetCompleted",
                    "phaseActivationCount",
                    "adaptationTime",
                    "adaptationCensored",
                    "restrictedAdaptationTime",
                    "overshootCensored",
                    "postHitAnyDeparture",
                    "finalNewTargetDistance",
                )
            }
            | {
                "descriptorsByAxis": first["descriptorsByAxis"],
                "nativeMovementDescriptors": first["nativeMovementDescriptors"],
            },
            replay_pass=replay_pass,
            validation=checks,
            provenance={
                "predecessor": "E05",
                "benchmarkVersion": "E05-collective-target-change-v1",
                "targetHash": str(first["targetDefinition"]["targetHash"]),
                "adapterVersion": "e07.s04a.dsl-native-adapters.v1",
            },
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
    if action.mode == "dsl_episode":
        counter_schedule_key = record.public_parameters.get("counterScheduleKey")
        if counter_schedule_key is not None:
            if record.split.value != "train" or record.protected:
                raise SuiteValidationError(
                    "spatial counter-schedule resampling is train-only"
                )
            # E06 counter-based randomness is keyed by scenario_id.  S05 keeps
            # the frozen native challenge and varies only that public counter
            # address; no outcome or protected row is consulted.
            definition = replace(
                definition,
                scenario_id=str(counter_schedule_key),
            )
        first = run_spatial_dsl_episode(
            context,
            definition,
            action,
            target_id=str(record.public_parameters["targetId"]),
        )
        second = run_spatial_dsl_episode(
            context,
            definition,
            action,
            target_id=str(record.public_parameters["targetId"]),
        )
        replay_pass = first == second
        authority = {key: bool(value) for key, value in first["authorityAudit"].items()}
        validation = {
            "exactReplay": replay_pass,
            **authority,
            "fixedTransitionCount": len(first["transitionSummaries"])
            == int(first["transitionBudget"]),
            "configurationChargedOnce": first["e06ChannelLedger"]["configurationBits"]
            == 0,
        }
        metrics = first["offlineEvaluation"]
        return NativeEpisodeResult(
            stop_reason=str(first["stopReason"]),
            censored=not bool(metrics["conjunctiveCompletionByBudget"]),
            failed=not all(validation.values()),
            native_costs={
                "e06MovementLedger": first["movementLedger"],
                "e06ObservationLedger": first["observationLedger"],
                "e06ChannelLedger": first["e06ChannelLedger"],
                "dslRuntimeLedger": first["dslRuntimeLedger"],
                "dslCommunicationLedger": first["dslCommunicationLedger"],
            },
            native_event={
                "schemaVersion": str(first["schemaVersion"]),
                "episodeSha256": str(first["episodeSha256"]),
                "transitionCount": len(first["transitionSummaries"]),
                "actorBatchSize": int(first["actorBatchSize"]),
                "onlineGlobalCompletionComputed": False,
                "policyControlledEpisode": True,
                "descriptors": first["descriptors"],
            },
            native_outcome={
                "fixedBudgetCompleted": True,
                "finalStateSha256": str(first["finalStateSha256"]),
                "s01GlobalCompletionAudit": {
                    "success": bool(metrics["terminalS01GlobalSuccess"]),
                    "normalizedMismatch": float(metrics["terminalS01MismatchFraction"]),
                },
                "s02LocalGrammar": {
                    "accepted": bool(metrics["terminalS02GrammarAccepted"]),
                    "relationalScore": float(metrics["terminalS02RelationalScore"]),
                },
                "conjunctiveCompletion": bool(metrics["conjunctiveCompletionByBudget"]),
                "formationCompletion": None,
                "formationCompletionApplicability": "not_applicable_one_swap_repair_fixture",
                "repairCompletion": bool(metrics["conjunctiveCompletionByBudget"]),
                "descriptors": first["descriptors"],
                "evaluationSeparation": {
                    "offlineOnly": True,
                    "fixedBudgetIsNotCompletion": True,
                },
                "formationOrRepairClaim": bool(
                    metrics["conjunctiveCompletionByBudget"]
                ),
            },
            replay_pass=replay_pass,
            validation=validation,
            provenance={
                "predecessor": "E06",
                "engineVersion": "e06-morph2d-hybrid-engine-v1",
                "nativeScenarioId": definition.scenario_id,
                "baseNativeScenarioId": scenario_id,
                "adapterVersion": "e07.s04a.dsl-native-adapters.v1",
            },
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
