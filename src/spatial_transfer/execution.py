"""Execution helpers for the byte-frozen S12R replacement transfer study.

This module does not define a new transfer estimand.  It binds the already
frozen S12R logical reservations to the qualified E06/S08 execution paths and
projects task-local endpoints without constructing a universal score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import json
from typing import Any

from src.environment_suite.contracts import (
    SuiteValidationError,
    canonical_sha256,
)
from src.environment_suite.dsl_adapters import (
    dsl_action,
    run_spatial_dsl_episode,
)
from src.environment_suite.portfolio_adapters import portfolio_action
from src.morph2d.baseline import (
    TargetMetricTracker,
    load_baseline_assets,
    state_grid,
)
from src.morph2d.engine import EpisodeDefinition, run_cpu_episode
from src.morph2d.elapsed_clock import (
    validate_elapsed_clock_summary,
    validate_first_completion_projection,
)
from src.morph2d.grammar import score_grid
from src.morph2d.movements import MovementState
from src.morph2d.targets import evaluate_success
from src.spatial_transfer.qualification import (
    DiagnosticSpatialTracker,
    build_panel_fixture,
)
from src.spatial_transfer.replacement import (
    FAULT_FAMILY_ID,
    SCHEDULER_FAMILY_ID,
    apply_spurious_swap_fault,
    identity_round_robin_schedule,
    scheduler_cost_ledger,
    validate_spurious_swap_fault_audit,
)


NATIVE_SCHEDULER_ID = "E06_state_blind_identity_hash_batch4"
NATIVE_BASELINE_POLICY = {
    "e07_s02_spatial2d_local": "greedy_neighbor_satisfaction_v1",
    "e07_s02_spatial2d_memory": "memory_based_recovery_v1",
}
NEW_REPLACEMENT_COMPARATOR_IDS = frozenset(
    {
        "055941575546d1e08ff91252a1be31e9b7c5005257917a621da63b344e9e4ca0",
        "756c599e70fe8464def363c3536514cfc166a9bc07340f28e4f4fccdba2f9f1c",
    }
)


def canonical_json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def zero_ledger(fields: Sequence[str]) -> dict[str, int]:
    return {str(field): 0 for field in fields}


class CalibratedTransferTracker:
    """Add maintenance/departure state to the native S01/S02 tracker."""

    def __init__(self, environment, target, grammar) -> None:
        self.environment = environment
        self.target = target
        self.grammar = grammar
        self.inner = TargetMetricTracker(environment, target, grammar)
        self.completion_sequence: list[bool] = []

    def _observe_completion(self, state: MovementState) -> None:
        grid = state_grid(self.environment, state)
        global_audit = evaluate_success(
            grid,
            self.target,
            equivalence_orbit=self.inner.orbit_grids,
        )
        local = score_grid(grid, self.grammar)
        self.completion_sequence.append(
            bool(global_audit["success"] and local["accepted"])
        )

    def observe(
        self,
        transition_index: int,
        state: MovementState,
        summary: Mapping[str, Any],
    ) -> None:
        self.inner.observe(transition_index, state, summary)
        self._observe_completion(state)

    def observe_authenticated(
        self,
        clock_record: Mapping[str, Any],
        state: MovementState,
        summary: Mapping[str, Any],
    ) -> None:
        self.inner.observe_authenticated(clock_record, state, summary)
        self._observe_completion(state)

    def finalize(self) -> dict[str, Any]:
        result = self.inner.finalize()
        if not self.completion_sequence:
            raise SuiteValidationError("calibrated tracker observed no state")
        initial = self.completion_sequence[0]
        result.update(
            {
                "schemaVersion": "e07.s12s.calibrated-transfer-endpoint.v1",
                "initialConjunctiveCompletion": initial,
                "terminalConjunctiveCompletion": self.completion_sequence[-1],
                "departureAfterInitiallyComplete": bool(
                    initial and not all(self.completion_sequence)
                ),
                "completionObservationCount": len(self.completion_sequence),
                "completionSequenceSha256": canonical_sha256(
                    "E07/S12S/completion-sequence/v1",
                    self.completion_sequence,
                ),
            }
        )
        return result


@lru_cache(maxsize=16)
def cached_panel_fixture(task_id: str, panel_id: str) -> dict[str, Any]:
    return build_panel_fixture(task_id, panel_id)


def _runtime_action(
    configuration: Mapping[str, Any],
    documents_by_hash: Mapping[str, Mapping[str, Any]],
):
    documents = [
        deepcopy(documents_by_hash[str(member["policySha256"])])
        for member in configuration["members"]
    ]
    if configuration["mode"] == "single_policy":
        if len(documents) != 1:
            raise SuiteValidationError("single-policy runtime is not singleton")
        return dsl_action(documents)
    return portfolio_action(documents, configuration)


def _scheduler(
    scheduler_family_id: str,
):
    if scheduler_family_id == NATIVE_SCHEDULER_ID:
        return None
    if scheduler_family_id != SCHEDULER_FAMILY_ID:
        raise SuiteValidationError("unknown frozen scheduler family")
    return identity_round_robin_schedule


def _tracker(fixture: Mapping[str, Any]):
    if fixture["endpointMode"] == "diagnostic_only":
        return DiagnosticSpatialTracker(
            fixture["environment"],
            fixture["reference"],
        )
    return CalibratedTransferTracker(
        fixture["environment"],
        fixture["target"],
        fixture["grammar"],
    )


def _initial_state_and_fault(
    fixture: Mapping[str, Any],
    logical: Mapping[str, Any],
) -> tuple[MovementState, dict[str, Any], dict[str, int]]:
    state = fixture["initialState"]
    fault_fields = (
        "faultEventsScheduled",
        "faultEventsTriggered",
        "faultEligibilityEvaluations",
        "faultCounterDraws",
        "faultCommittedSpuriousSwaps",
        "faultDisplacedEntities",
        "faultDisplacedCells",
        "faultDisplacedVacancies",
        "faultGraphDisplacement",
        "faultPolicyOpportunityUnits",
        "faultNativeProposalUnits",
    )
    if logical["faultFamilyId"] == "none":
        return state, {}, zero_ledger(fault_fields)
    if logical["faultFamilyId"] != FAULT_FAMILY_ID:
        raise SuiteValidationError("unknown frozen spatial fault family")
    if fixture["endpointMode"] == "diagnostic_only":
        raise SuiteValidationError("target-breaking fault entered diagnostic panel")
    post, audit = apply_spurious_swap_fault(
        fixture["environment"],
        state,
        fixture["target"],
        fixture["grammar"],
        scenario_family_id=str(logical["scenarioFamilyId"]),
    )
    validate_spurious_swap_fault_audit(
        fixture["environment"],
        state,
        fixture["target"],
        fixture["grammar"],
        audit,
    )
    return post, audit, dict(audit["faultLedger"])


def _native_baseline_episode(
    logical: Mapping[str, Any],
    fixture: Mapping[str, Any],
    initial_state: MovementState,
    tracker: Any,
) -> dict[str, Any]:
    context, _targets, _grammars, _environments = load_baseline_assets()
    context = replace(
        context,
        environments={
            **context.environments,
            fixture["environment"].environment_id: fixture["environment"],
        },
    )
    policy_id = NATIVE_BASELINE_POLICY[str(logical["taskId"])]
    parameters: dict[str, Any] = {}
    if policy_id == "memory_based_recovery_v1":
        parameters["memoryInitialBestLocalUtility"] = 12
    definition = EpisodeDefinition(
        scenario_id=str(logical["scenarioFamilyId"]),
        environment_id=fixture["environment"].environment_id,
        policy_id=policy_id,
        relation_grammar_id=fixture["grammar"].grammar_id,
        channel_mode="none",
        transitions=32,
        actor_batch_size=4,
        parameters=parameters,
    )
    authenticated_audit = getattr(tracker, "observe_authenticated", None)
    result = run_cpu_episode(
        context,
        definition,
        include_selected_traces=False,
        initial_state_override=initial_state,
        state_audit=(tracker.observe if authenticated_audit is None else None),
        authenticated_state_audit=authenticated_audit,
        authenticated_elapsed_clock=True,
        actor_schedule=_scheduler(str(logical["schedulerFamilyId"])),
        actor_schedule_id=str(logical["schedulerFamilyId"]),
    )
    return {
        "engineKind": "native_e06_baseline",
        "nativeResultSha256": str(result["episodeSha256"]),
        "initialStateSha256": str(result["initialStateSha256"]),
        "finalStateSha256": str(result["finalState"]["stateSha256"]),
        "movementLedger": dict(result["movementLedger"]),
        "observationLedger": dict(result["observationLedger"]),
        "channelLedger": dict(result["channelLedger"]),
        "dslRuntimeLedger": {},
        "licensedCapabilityLedger": {},
        "portfolioCoordinationLedger": {},
        "transitionSummaries": list(result["transitionSummaries"]),
        "authenticatedElapsedClockAudit": dict(
            result["authenticatedElapsedClockAudit"]
        ),
        "stopReason": str(result["stopReason"]),
        "authorityAudit": {
            **dict(result["permissionAudit"]),
            "nativeLegalityPreserved": (
                int(result["movementLedger"]["invalidProposals"]) == 0
            ),
            "fixedClockPreserved": len(result["transitionSummaries"]) == 32,
        },
    }


def _dsl_episode(
    logical: Mapping[str, Any],
    fixture: Mapping[str, Any],
    initial_state: MovementState,
    tracker: Any,
    configuration: Mapping[str, Any],
    documents_by_hash: Mapping[str, Mapping[str, Any]],
    runtime_action: Any | None = None,
) -> dict[str, Any]:
    context, _targets, _grammars, _environments = load_baseline_assets()
    context = replace(
        context,
        environments={
            **context.environments,
            fixture["environment"].environment_id: fixture["environment"],
        },
    )
    action = runtime_action or _runtime_action(configuration, documents_by_hash)
    definition = EpisodeDefinition(
        scenario_id=str(logical["scenarioFamilyId"]),
        environment_id=fixture["environment"].environment_id,
        policy_id=action.policy_id,
        relation_grammar_id=fixture["grammar"].grammar_id,
        channel_mode="none",
        transitions=32,
        actor_batch_size=4,
        parameters={},
    )
    result = run_spatial_dsl_episode(
        context,
        definition,
        action,
        target_id=str(fixture["targetId"]),
        initial_state_override=initial_state,
        offline_tracker=tracker,
        authenticated_elapsed_clock=True,
        actor_schedule=_scheduler(str(logical["schedulerFamilyId"])),
    )
    if not all(bool(value) for value in result["authorityAudit"].values()):
        raise SuiteValidationError("DSL episode authority audit failed")
    return {
        "engineKind": "qualified_dsl_portfolio_or_single",
        "nativeResultSha256": str(result["episodeSha256"]),
        "initialStateSha256": str(result["initialStateSha256"]),
        "finalStateSha256": str(result["finalStateSha256"]),
        "movementLedger": dict(result["movementLedger"]),
        "observationLedger": dict(result["observationLedger"]),
        "channelLedger": dict(result["e06ChannelLedger"]),
        "dslRuntimeLedger": dict(result["dslRuntimeLedger"]),
        "licensedCapabilityLedger": dict(result["licensedCapabilityLedger"]),
        "portfolioCoordinationLedger": dict(
            result.get("portfolioCoordinationLedger", {})
        ),
        "transitionSummaries": list(result["transitionSummaries"]),
        "authenticatedElapsedClockAudit": dict(
            result["authenticatedElapsedClockAudit"]
        ),
        "stopReason": str(result["stopReason"]),
        "authorityAudit": dict(result["authorityAudit"]),
    }


def execute_physical(
    logical: Mapping[str, Any],
    *,
    replay_ordinal: int,
    physical_execution_id: str,
    configuration: Mapping[str, Any] | None,
    resolved_runtime_id: str,
    adaptation_variant_id: str | None,
    documents_by_hash: Mapping[str, Mapping[str, Any]],
    runtime_action: Any | None = None,
) -> dict[str, Any]:
    """Execute and project one frozen physical commitment."""

    fixture = cached_panel_fixture(str(logical["taskId"]), str(logical["panelId"]))
    tracker = _tracker(fixture)
    initial_state, fault_audit, fault_ledger = _initial_state_and_fault(
        fixture, logical
    )
    if logical["conditionRole"] == "native_baseline":
        native = _native_baseline_episode(logical, fixture, initial_state, tracker)
        structural_ledger: dict[str, int] = {}
    else:
        if configuration is None:
            raise SuiteValidationError("non-native reservation has no runtime binding")
        native = _dsl_episode(
            logical,
            fixture,
            initial_state,
            tracker,
            configuration,
            documents_by_hash,
            runtime_action,
        )
        structural_ledger = {
            str(key): int(value)
            for key, value in configuration["portfolioStructuralCosts"].items()
        }
    endpoint = tracker.finalize()
    clock_summary = validate_elapsed_clock_summary(
        native["authenticatedElapsedClockAudit"],
        expected_scenario_id=str(logical["scenarioFamilyId"]),
        expected_horizon=32,
    )
    actor_count = sum(
        occupant.kind == "cell" for _, occupant in initial_state.occupancy
    )
    if logical["schedulerFamilyId"] == SCHEDULER_FAMILY_ID:
        scheduler_ledger = scheduler_cost_ledger(
            actor_count=actor_count,
            transition_count=32,
            batch_size=4,
        )
    else:
        scheduler_ledger = zero_ledger(
            (
                "schedulerActorPopulationReads",
                "schedulerStartOffsetCounterDraws",
                "schedulerCursorArithmeticOperations",
                "schedulerSelectedActorSlots",
                "schedulerStateReads",
                "schedulerOutcomeReads",
                "schedulerReplacementWithinSweep",
            )
        )
    diagnostic = fixture["endpointMode"] == "diagnostic_only"
    faulted = logical["faultFamilyId"] == FAULT_FAMILY_ID
    if diagnostic:
        status = "endpoint_unavailable_diagnostic_retained"
        censored = False
        repair = None
    elif faulted:
        repair = bool(endpoint["conjunctiveCompletionByBudget"])
        status = "repair_observed" if repair else "right_censored_at_transition_32"
        censored = not repair
    else:
        repair = None
        status = "calibrated_endpoint_retained"
        censored = False
    comparator_ledger = {
        "comparatorReservation": int(
            logical["conditionRole"]
            in {"component_single", "matched_random", "native_baseline"}
        ),
        "newReplacementComparatorReservation": int(
            resolved_runtime_id in NEW_REPLACEMENT_COMPARATOR_IDS
        ),
    }
    adaptation_ledger = {
        "adaptationVariantApplied": int(adaptation_variant_id is not None),
        "adaptationMemberSetChanges": 0,
        "adaptationRuleChanges": 0,
        "adaptationMemoryChanges": 0,
        "adaptationObservationChanges": 0,
        "adaptationSignalOrCommunicationChanges": 0,
    }
    if diagnostic:
        first_completion = None
        completion_projection = None
    else:
        endpoint_clock_summary = validate_elapsed_clock_summary(
            endpoint["authenticatedElapsedClockSummary"],
            expected_scenario_id=str(logical["scenarioFamilyId"]),
            expected_horizon=32,
        )
        if (
            endpoint_clock_summary["summaryCommitmentSha256"]
            != clock_summary["summaryCommitmentSha256"]
        ):
            raise SuiteValidationError(
                "native episode and endpoint elapsed-clock commitments differ"
            )
        completion_projection = validate_first_completion_projection(
            endpoint["authenticatedFirstCompletionProjection"],
            expected_scenario_id=str(logical["scenarioFamilyId"]),
            expected_horizon=32,
            clock_summary=clock_summary,
        )
        if (
            completion_projection["clockSummaryCommitmentSha256"]
            != clock_summary["summaryCommitmentSha256"]
        ):
            raise SuiteValidationError(
                "completion projection is detached from native elapsed clock"
            )
        first_completion = completion_projection["firstCompletionTransition"]
    result_projection = {
        "schemaVersion": "e07.s12z.physical-result-projection.v2",
        "logicalReservationId": str(logical["logicalReservationId"]),
        "scenarioFamilyId": str(logical["scenarioFamilyId"]),
        "partition": str(logical["partition"]),
        "taskCellId": str(logical["taskCellId"]),
        "couplingCommitmentSha256": str(logical["couplingCommitmentSha256"]),
        "taskId": str(logical["taskId"]),
        "panelId": str(logical["panelId"]),
        "conditionId": str(logical["conditionId"]),
        "conditionRole": str(logical["conditionRole"]),
        "lineageId": str(logical["lineageId"]),
        "baseConfigurationId": str(logical["baseConfigurationId"]),
        "resolvedRuntimeConfigurationId": resolved_runtime_id,
        "adaptationVariantId": adaptation_variant_id,
        "schedulerFamilyId": str(logical["schedulerFamilyId"]),
        "faultFamilyId": str(logical["faultFamilyId"]),
        "engineKind": native["engineKind"],
        "nativeResultSha256": native["nativeResultSha256"],
        "initialStateSha256": native["initialStateSha256"],
        "finalStateSha256": native["finalStateSha256"],
        "status": status,
        "failed": False,
        "censored": censored,
        "endpointAvailable": not diagnostic,
        "repairByTransition32": repair,
        "terminalConjunctiveCompletion": (
            None
            if diagnostic
            else bool(endpoint["terminalConjunctiveCompletion"])
        ),
        "completionByBudget": (
            None if diagnostic else bool(endpoint["conjunctiveCompletionByBudget"])
        ),
        "firstCompletionTransition": first_completion,
        "authenticatedElapsedClockSummaryJson": canonical_json_text(clock_summary),
        "authenticatedFirstCompletionProjectionJson": (
            None
            if completion_projection is None
            else canonical_json_text(completion_projection)
        ),
        "rawObservationLabelsUsedAsScientificTime": False,
        "minimumMismatchFraction": (
            None if diagnostic else float(endpoint["minimumS01MismatchFraction"])
        ),
        "terminalMismatchFraction": (
            None if diagnostic else float(endpoint["terminalS01MismatchFraction"])
        ),
        "departureAfterInitiallyComplete": (
            None if diagnostic else bool(endpoint["departureAfterInitiallyComplete"])
        ),
        "diagnosticEndpointJson": (
            canonical_json_text(endpoint) if diagnostic else None
        ),
        "movementLedgerJson": canonical_json_text(native["movementLedger"]),
        "observationLedgerJson": canonical_json_text(native["observationLedger"]),
        "channelLedgerJson": canonical_json_text(native["channelLedger"]),
        "dslRuntimeLedgerJson": canonical_json_text(native["dslRuntimeLedger"]),
        "portfolioStructuralLedgerJson": canonical_json_text(structural_ledger),
        "portfolioCoordinationLedgerJson": canonical_json_text(
            native["portfolioCoordinationLedger"]
        ),
        "licensedCapabilityLedgerJson": canonical_json_text(
            native["licensedCapabilityLedger"]
        ),
        "adaptationLedgerJson": canonical_json_text(adaptation_ledger),
        "comparatorLedgerJson": canonical_json_text(comparator_ledger),
        "faultLedgerJson": canonical_json_text(fault_ledger),
        "schedulerLedgerJson": canonical_json_text(scheduler_ledger),
        "faultAuditSha256": fault_audit.get("faultAuditSha256"),
        "transitionSequenceSha256": canonical_sha256(
            "E07/S12S/transition-sequence/v1",
            native["transitionSummaries"],
        ),
        "stopReason": native["stopReason"],
        "nativeLegalityPreserved": bool(
            native["authorityAudit"]["nativeLegalityPreserved"]
        ),
        "fixedClockPreserved": len(native["transitionSummaries"]) == 32,
        "diagnosticPromotionEligible": False if diagnostic else None,
    }
    if (
        not result_projection["nativeLegalityPreserved"]
        or not result_projection["fixedClockPreserved"]
    ):
        raise SuiteValidationError("native legality or fixed clock failed")
    result_sha256 = canonical_sha256(
        "E07/S12S/physical-result-projection/v1",
        result_projection,
    )
    return {
        **result_projection,
        "replayOrdinal": int(replay_ordinal),
        "physicalExecutionId": str(physical_execution_id),
        "resultProjectionSha256": result_sha256,
    }


def logical_from_replays(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    if first["logicalReservationId"] != second["logicalReservationId"]:
        raise SuiteValidationError("replay logical identities differ")
    if first["resultProjectionSha256"] != second["resultProjectionSha256"]:
        raise SuiteValidationError("exact physical replay mismatch")
    excluded = {"replayOrdinal", "physicalExecutionId"}
    projected = {key: value for key, value in first.items() if key not in excluded}
    return {
        **projected,
        "physicalExecutionId0": str(first["physicalExecutionId"]),
        "physicalExecutionId1": str(second["physicalExecutionId"]),
        "exactReplay": True,
    }


def adaptation_component_vector(row: Mapping[str, Any]) -> dict[str, float]:
    """Return direction-aligned task-local components without aggregation."""

    prefix = f"{row['taskId']}::{row['panelId']}::{row['conditionId']}"
    if not bool(row["endpointAvailable"]):
        return {}
    components = {
        f"{prefix}::failure_avoidance": -float(bool(row["failed"])),
        f"{prefix}::minimum_mismatch_benefit": -float(
            row["minimumMismatchFraction"]
        ),
        f"{prefix}::terminal_completion": float(
            bool(row["terminalConjunctiveCompletion"])
        ),
    }
    if row["faultFamilyId"] == FAULT_FAMILY_ID:
        components[f"{prefix}::repair_by_32"] = float(
            bool(row["repairByTransition32"])
        )
    else:
        components[f"{prefix}::departure_avoidance"] = -float(
            bool(row["departureAfterInitiallyComplete"])
        )
    return components


def choose_adaptation_winners(
    development_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Lock one nondominated adaptation per base, then use the frozen hash tie."""

    by_base: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in development_rows:
        by_base.setdefault(str(row["baseConfigurationId"]), {}).setdefault(
            str(row["adaptationVariantId"]), []
        ).append(row)
    locks = []
    for base_id, by_variant in sorted(by_base.items()):
        vectors: dict[str, dict[str, float]] = {}
        for variant_id, rows in sorted(by_variant.items()):
            accum: dict[str, list[float]] = {}
            for row in rows:
                for key, value in adaptation_component_vector(row).items():
                    accum.setdefault(key, []).append(float(value))
            vectors[variant_id] = {
                key: sum(values) / len(values)
                for key, values in sorted(accum.items())
            }
        keys = sorted({key for vector in vectors.values() for key in vector})
        if not keys or any(set(vector) != set(keys) for vector in vectors.values()):
            raise SuiteValidationError(
                "adaptation task-local component support is incomplete"
            )
        nondominated = []
        for candidate, vector in sorted(vectors.items()):
            dominated = False
            for other, other_vector in vectors.items():
                if other == candidate:
                    continue
                if all(other_vector[key] >= vector[key] for key in keys) and any(
                    other_vector[key] > vector[key] for key in keys
                ):
                    dominated = True
                    break
            if not dominated:
                nondominated.append(candidate)
        winner = min(nondominated)
        locks.append(
            {
                "baseConfigurationId": base_id,
                "winnerAdaptationVariantId": winner,
                "nondominatedAdaptationVariantIds": sorted(nondominated),
                "componentDirections": "every persisted component is higher-is-better",
                "taskLocalComponentMeans": vectors,
                "universalScoreConstructed": False,
                "tieBreak": "lowest_stable_S12R_variant_hash",
            }
        )
    body = {
        "schemaVersion": "e07.s12s.adaptation-lock.v1",
        "researchStepId": "S12S",
        "selectionPopulation": "S12R_development_only",
        "selectionMethod": (
            "task-local calibrated endpoint Pareto dominance; diagnostics "
            "excluded; lowest stable S12R variant hash over the nondominated set"
        ),
        "configurationCount": len(locks),
        "locks": locks,
        "universalAdaptationScore": None,
        "postLockOutcomesUsed": False,
        "protectedOutcomesUsed": False,
    }
    body["adaptationLockSha256"] = canonical_sha256(
        "E07/S12S/adaptation-lock/v1", body
    )
    return body
