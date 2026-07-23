"""Outcome-independent S12A native spatial-transfer qualification.

This module never selects a policy from outcomes and never opens a protected
split.  It binds exact frozen spatial policy bytes to E06 authority, compiles
named native target fixtures, and makes endpoint availability explicit.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.environment_suite.contracts import SuiteValidationError, canonical_sha256
from src.environment_suite.portfolio_adapters import (
    ALLOWED_SELECTOR_SIGNALS,
    PortfolioDispatcher,
    portfolio_action,
)
from src.morph2d.baseline import build_target_environment, load_baseline_assets
from src.morph2d.environments import (
    Environment,
    EnvironmentValidationError,
    boundary_observation,
    evaluate_conjunctive,
)
from src.morph2d.minimal_control import (
    build_transfer_fixture,
    graph_diagnostics,
)
from src.morph2d.movements import (
    MovementState,
    initial_movement_state,
    make_proposal,
    movement_state_sha256,
    parse_movement_state,
    resolve_batch,
)
from src.policy_dsl import compile_policy


SPATIAL_TASKS = (
    "e07_s02_spatial2d_local",
    "e07_s02_spatial2d_memory",
)
PANEL_IDS = (
    "native_reference_control",
    "unseen_target_bilateral",
    "unseen_target_single_hole",
    "unseen_target_two_holes",
    "unseen_size_layers_15x15",
    "unseen_topology_irregular_layers_9x9",
    "displacement_native_target",
    "combined_size_displacement",
)
CALIBRATED_TARGETS = (
    "stripes_alternating_three_band",
    "layers_three_ordered_tissues",
    "bilateral_lobes_with_midline",
    "tissue_single_hole",
    "tissue_two_holes",
)
DIAGNOSTIC_FIXTURES = (
    "larger_square_layers_15x15",
    "irregular_pruned_layers_9x9",
)
TARGET_BY_PANEL = {
    "unseen_target_bilateral": "bilateral_lobes_with_midline",
    "unseen_target_single_hole": "tissue_single_hole",
    "unseen_target_two_holes": "tissue_two_holes",
}
NATIVE_TARGET_BY_TASK = {
    "e07_s02_spatial2d_local": "stripes_alternating_three_band",
    "e07_s02_spatial2d_memory": "layers_three_ordered_tissues",
}
DIAGNOSTIC_FIXTURE_BY_PANEL = {
    "unseen_size_layers_15x15": "larger_square_layers_15x15",
    "unseen_topology_irregular_layers_9x9": "irregular_pruned_layers_9x9",
    "combined_size_displacement": "larger_square_layers_15x15",
}
NATIVE_MOVEMENT_KINDS = frozenset(
    {"adjacent_swap", "vacancy_move", "short_exchange", "rotation"}
)
AUTHORITY_BEARING_PERMISSIONS = frozenset(
    {
        "natural.boundary_signal",
        "candidate.natural_boundary_delta",
        "gradient.current_u8",
        "candidate.gradient_delta",
    }
)
DIAGNOSTIC_UNAVAILABLE_REASON = "TOPOLOGY_SPECIFIC_COMPLETION_NOT_CALIBRATED"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + _json_bytes(value)
    ).hexdigest()


def _grammar_for_target(context, target_id: str):
    return next(
        grammar
        for grammar in context.grammars.values()
        if grammar.target_id == target_id
    )


def _state_assignment(state: MovementState) -> dict[str, str]:
    return {
        site_id: occupant.token
        for site_id, occupant in sorted(state.occupancy, key=lambda item: item[0])
    }


def _one_swap_state(
    environment: Environment,
    *,
    target_validation: Mapping[str, Any] | None = None,
) -> tuple[MovementState, int]:
    state = initial_movement_state(environment)
    if target_validation and target_validation.get("acceptedBoundarySwap"):
        first, second = target_validation["acceptedBoundarySwap"]
        first_site = f"r{int(first[0])}_c{int(first[1])}"
        second_site = f"r{int(second[0])}_c{int(second[1])}"
        occupants = state.occupant_map
        occupants[first_site], occupants[second_site] = (
            occupants[second_site],
            occupants[first_site],
        )
        # S01's acceptedBoundarySwap is a target-set perturbation witness, not
        # an S04 adjacent-movement proposal (the declared sites can be
        # diagonal). Preserve that distinction and never call it a native
        # spatial fault or an executed policy action.
        return (
            MovementState(
                environment_id=state.environment_id,
                environment_sha256=state.environment_sha256,
                transition_index=state.transition_index,
                occupancy=tuple(sorted(occupants.items())),
            ),
            2,
        )
    edge = next(
        candidate
        for candidate in environment.edges
        if state.occupant_map[candidate[0]].token
        != state.occupant_map[candidate[1]].token
    )
    proposal = make_proposal(state, "adjacent_swap", edge)
    batch = resolve_batch(
        environment,
        state,
        (proposal,),
        batch_nonce=f"{environment.environment_id}:s12a-one-swap",
    )
    if int(batch["costLedger"]["acceptedMovements"]) != 1:
        raise SuiteValidationError("qualification swap was not accepted natively")
    return (
        parse_movement_state(batch["postState"]),
        int(batch["costLedger"]["totalGraphDisplacement"]),
    )


def build_panel_fixture(
    task_id: str,
    panel_id: str,
) -> dict[str, Any]:
    """Build one authoritative target/topology fixture without an S12 identity."""

    if task_id not in SPATIAL_TASKS:
        raise SuiteValidationError("S12A supports only the two native spatial tasks")
    if panel_id not in PANEL_IDS:
        raise SuiteValidationError("unknown frozen S12P panel")
    context, targets, _grammars, environments = load_baseline_assets()
    challenge_id = "exact_maintenance"
    endpoint_mode = "calibrated_task_native"
    if panel_id in DIAGNOSTIC_FIXTURE_BY_PANEL:
        fixture_id = DIAGNOSTIC_FIXTURE_BY_PANEL[panel_id]
        environment, reference = build_transfer_fixture(fixture_id)
        if panel_id == "combined_size_displacement":
            challenge_id = "one_swap_displacement_diagnostic"
            state, displacement = _one_swap_state(environment)
        else:
            state = initial_movement_state(environment)
            displacement = 0
        return {
            "taskId": task_id,
            "panelId": panel_id,
            "targetId": "layers_three_ordered_tissues",
            "fixtureId": fixture_id,
            "challengeId": challenge_id,
            "endpointMode": "diagnostic_only",
            "environment": environment,
            "reference": dict(reference),
            "initialState": state,
            "externalDisplacement": displacement,
            "grammar": _grammar_for_target(context, "layers_three_ordered_tissues"),
        }

    target_id = TARGET_BY_PANEL.get(panel_id, NATIVE_TARGET_BY_TASK[task_id])
    target = targets[target_id]
    grammar = _grammar_for_target(context, target_id)
    environment = environments.get(target_id) or build_target_environment(
        target, grammar
    )
    state = initial_movement_state(environment)
    displacement = 0
    if panel_id == "displacement_native_target":
        challenge_id = "one_swap_displacement_diagnostic"
        endpoint_mode = "calibrated_task_native_maintenance_not_repair"
        state, displacement = _one_swap_state(
            environment,
            target_validation=target.validation,
        )
    return {
        "taskId": task_id,
        "panelId": panel_id,
        "targetId": target_id,
        "fixtureId": "canonical_target_domain",
        "challengeId": challenge_id,
        "endpointMode": endpoint_mode,
        "environment": environment,
        "reference": dict(environment.initial_state),
        "initialState": state,
        "externalDisplacement": displacement,
        "target": target,
        "grammar": grammar,
    }


def endpoint_contract(task_id: str, panel_id: str) -> dict[str, Any]:
    fixture = build_panel_fixture(task_id, panel_id)
    diagnostic = fixture["endpointMode"] == "diagnostic_only"
    displaced_calibrated = panel_id == "displacement_native_target"
    return {
        "schemaVersion": "e07.s12a.endpoint-contract.v1",
        "taskId": task_id,
        "panelId": panel_id,
        "targetId": fixture["targetId"],
        "fixtureId": fixture["fixtureId"],
        "challengeId": fixture["challengeId"],
        "calibratedCompletionAvailable": not diagnostic,
        "repairRiskSetEligible": False,
        "repairUnavailableReason": (
            DIAGNOSTIC_UNAVAILABLE_REASON
            if diagnostic
            else (
                "POST_SWAP_STATE_REMAINS_INSIDE_CALIBRATED_TARGET"
                if displaced_calibrated
                else "EXACT_MAINTENANCE_IS_NOT_A_REPAIR_RISK_SET"
            )
        ),
        "endpointTimeAvailable": not diagnostic,
        "endpointUnavailableReason": (
            DIAGNOSTIC_UNAVAILABLE_REASON if diagnostic else None
        ),
        "completionRule": (
            None
            if diagnostic
            else "S02_local_accepted_AND_independent_S01_global_success"
        ),
        "diagnosticFields": (
            [
                "referenceMismatchCount",
                "referenceMismatchFraction",
                "graphComponentError",
                "normalizedBoundaryError",
                "compositionCorrectedHomotypicEdgeExcess",
            ]
            if diagnostic
            else []
        ),
        "diagnosticPromotionEligible": False,
        "oneSwapIsFault": False,
        "crossTaskNormalization": "forbidden",
    }


def _environment_with_assignment(
    environment: Environment,
    assignment: Mapping[str, str],
) -> Environment:
    if set(assignment) != set(environment.initial_state):
        raise EnvironmentValidationError("assignment does not cover environment")
    return replace(environment, initial_state=dict(sorted(assignment.items())))


def qualify_endpoint(task_id: str, panel_id: str) -> dict[str, Any]:
    """Qualify endpoint availability from native algebra and fixed fixtures."""

    fixture = build_panel_fixture(task_id, panel_id)
    contract = endpoint_contract(task_id, panel_id)
    environment = fixture["environment"]
    replay = _sha256(
        "E07/S12A/environment-structural-replay/v1",
        {
            "environmentId": environment.environment_id,
            "sites": [
                {
                    "siteId": site.site_id,
                    "coordinate": list(site.coordinate),
                    "role": site.role,
                    "boundaryTags": list(site.boundary_tags),
                }
                for site in environment.sites
            ],
            "edges": list(environment.edges),
            "initialState": dict(environment.initial_state),
        },
    )
    if contract["calibratedCompletionAvailable"]:
        target = fixture["target"]
        grammar = fixture["grammar"]
        source = evaluate_conjunctive(environment, target, grammar)
        evaluated_environment = _environment_with_assignment(
            environment, _state_assignment(fixture["initialState"])
        )
        challenge = evaluate_conjunctive(evaluated_environment, target, grammar)
        boundary_required = bool(target.equivalence["boundaryCueRequired"])
        no_signal_denied = False
        if boundary_required:
            no_signal = replace(
                environment,
                boundary_signal={
                    "mode": "none",
                    "observable": False,
                    "includeObstacleContact": False,
                    "includeFixedRole": False,
                },
            )
            try:
                evaluate_conjunctive(no_signal, target, grammar)
            except EnvironmentValidationError:
                no_signal_denied = True
        observations = [
            boundary_observation(environment, site.site_id)
            for site in environment.occupiable_sites
        ]
        signaled = sum(bool(item["boundaryTags"]) for item in observations)
        passed = bool(
            source["completion"]
            and challenge["completion"]
            and (
                not boundary_required
                or (
                    environment.boundary_signal["observable"]
                    and signaled > 0
                    and no_signal_denied
                )
            )
            and contract["repairRiskSetEligible"] is False
        )
        return {
            "schemaVersion": "e07.s12a.endpoint-qualification.v1",
            "taskId": task_id,
            "panelId": panel_id,
            "targetId": fixture["targetId"],
            "fixtureId": fixture["fixtureId"],
            "endpointMode": fixture["endpointMode"],
            "sourceConjunctionPass": bool(source["completion"]),
            "challengeConjunctionPass": bool(challenge["completion"]),
            "boundaryCueRequired": boundary_required,
            "boundarySignalObservable": bool(environment.boundary_signal["observable"]),
            "boundarySignaledSiteCount": signaled,
            "missingBoundarySignalDenied": no_signal_denied,
            "repairRiskSetEligible": False,
            "repairUnavailableReason": contract["repairUnavailableReason"],
            "environmentReplaySha256": replay,
            "passed": passed,
            "qualificationOnly": True,
            "efficacyEvidence": False,
        }

    reference = fixture["reference"]
    diagnostics = graph_diagnostics(environment, reference, reference)
    passed = bool(
        environment.target_binding is None
        and diagnostics["referenceMismatchCount"] == 0
        and diagnostics["graphComponentError"] == 0
        and contract["calibratedCompletionAvailable"] is False
        and contract["repairRiskSetEligible"] is False
        and contract["endpointUnavailableReason"] == DIAGNOSTIC_UNAVAILABLE_REASON
    )
    return {
        "schemaVersion": "e07.s12a.endpoint-qualification.v1",
        "taskId": task_id,
        "panelId": panel_id,
        "targetId": fixture["targetId"],
        "fixtureId": fixture["fixtureId"],
        "endpointMode": "diagnostic_only",
        "sourceConjunctionPass": None,
        "challengeConjunctionPass": None,
        "boundaryCueRequired": False,
        "boundarySignalObservable": bool(environment.boundary_signal["observable"]),
        "boundarySignaledSiteCount": sum(
            bool(site.boundary_tags) for site in environment.occupiable_sites
        ),
        "missingBoundarySignalDenied": False,
        "repairRiskSetEligible": False,
        "repairUnavailableReason": DIAGNOSTIC_UNAVAILABLE_REASON,
        "completionAvailable": False,
        "endpointTimeAvailable": False,
        "diagnosticReferenceIdentityPass": True,
        "environmentReplaySha256": replay,
        "passed": passed,
        "qualificationOnly": True,
        "efficacyEvidence": False,
    }


def _movement_kinds(document: Mapping[str, Any]) -> set[str]:
    kinds: set[str] = set()
    action_groups = [rule["actions"] for rule in document["rules"]] + [
        document["default"]["actions"]
    ]
    for actions in action_groups:
        for action in actions:
            if action["kind"] == "move_candidate":
                kinds.update(map(str, action["allowedMovementKinds"]))
    return kinds or {"adjacent_swap"}


def apply_adaptation_variant(
    configuration: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply one exact S12P adaptation edit and verify its frozen commitment."""

    if str(configuration["configurationId"]) != str(variant["baseConfigurationId"]):
        raise SuiteValidationError("adaptation variant/base configuration mismatch")
    result = deepcopy(dict(configuration))
    edit = dict(variant["edit"])
    kind = str(edit["kind"])
    if kind == "assignment_rotation":
        result["assignmentRotation"] = int(edit["assignmentRotation"])
    elif kind in {"selector_original", "selector_branch_swap"}:
        result["selector"] = deepcopy(edit["selector"])
    else:
        raise SuiteValidationError("unknown frozen S12P adaptation edit")
    commitment = canonical_sha256("E07/S12P/adapted-runtime-configuration/v1", result)
    if commitment != variant["runtimeConfigurationCommitmentSha256"]:
        raise SuiteValidationError("adaptation runtime commitment mismatch")
    return result


def qualify_configuration_binding(
    configuration: Mapping[str, Any],
    documents_by_hash: Mapping[str, Mapping[str, Any]],
    *,
    execution_task_id: str,
    panel_id: str,
    selection_ref: str,
) -> dict[str, Any]:
    """Statically bind exact configuration bytes without an episode outcome."""

    if execution_task_id not in SPATIAL_TASKS or panel_id not in PANEL_IDS:
        raise SuiteValidationError("binding left the frozen spatial task/panel scope")
    source_task = str(configuration["taskId"])
    if source_task not in SPATIAL_TASKS:
        raise SuiteValidationError("configuration source task is not native spatial")
    members = list(configuration["members"])
    if not members or any(item["nativeCarrier"] != "spatial" for item in members):
        raise SuiteValidationError("configuration must retain its spatial carrier")
    compiled = {}
    policy_documents = []
    authority_permissions: set[str] = set()
    movement_kinds: set[str] = set()
    for member in members:
        policy_hash = str(member["policySha256"])
        if policy_hash not in documents_by_hash:
            raise SuiteValidationError("frozen member document is unavailable")
        document = deepcopy(dict(documents_by_hash[policy_hash]))
        policy = compile_policy(document)
        if policy.policy_sha256 != policy_hash or policy.environment != "spatial2d.v1":
            raise SuiteValidationError("frozen member bytes or spatial type changed")
        compiled[policy.policy_id] = policy
        policy_documents.append(document)
        authority_permissions.update(policy.permissions & AUTHORITY_BEARING_PERMISSIONS)
        movement_kinds.update(_movement_kinds(document))
    if movement_kinds - NATIVE_MOVEMENT_KINDS:
        raise SuiteValidationError("member action is outside E06 native movement kinds")
    if authority_permissions:
        raise SuiteValidationError(
            "frozen S12P member requests an unbound E06 authority channel"
        )
    selector = configuration.get("selector")
    selector_signal = None
    if isinstance(selector, Mapping):
        selector_signal = str(selector["signal"])
        if selector_signal not in ALLOWED_SELECTOR_SIGNALS[execution_task_id]:
            raise SuiteValidationError("selector signal is not legal on target task")
    action = portfolio_action(policy_documents, configuration)
    dispatcher = PortfolioDispatcher(configuration, compiled)
    actors = tuple(f"s12a-cell-{index:03d}" for index in range(7))
    dispatcher.register_identities({"spatial": actors})
    choices = []
    for actor_index, actor in enumerate(actors[:3]):
        for selector_value in (False, True):
            choice = dispatcher.select(
                "spatial",
                actor,
                actor_index,
                selector_value=selector_value,
                mutate=False,
            )
            choices.append(choice.policy_sha256)
    panel = endpoint_contract(execution_task_id, panel_id)
    body = {
        "schemaVersion": "e07.s12a.binding-qualification.v1",
        "selectionRef": selection_ref,
        "baseConfigurationId": str(configuration["configurationId"]),
        "sourceTaskId": source_task,
        "executionTaskId": execution_task_id,
        "panelId": panel_id,
        "actionSha256": action.policy_sha256,
        "memberPolicySha256": [str(item["policySha256"]) for item in members],
        "memberCount": len(members),
        "mode": str(configuration["mode"]),
        "selectorSignal": selector_signal,
        "selectorTargetTaskAuthorized": selector_signal is None
        or selector_signal in ALLOWED_SELECTOR_SIGNALS[execution_task_id],
        "nativeCarrierPreserved": True,
        "dslEnvironmentPreserved": True,
        "nativeMovementKinds": sorted(movement_kinds),
        "authorityChannelPermissions": sorted(authority_permissions),
        "dispatchProbeSha256": _sha256("E07/S12A/dispatch-probe/v1", choices),
        "calibratedCompletionAvailable": panel["calibratedCompletionAvailable"],
        "diagnosticPromotionEligible": False,
        "nativeClock": "synchronous_graph_transition_with_four_actor_slots",
        "transitionBudget": 32,
        "actorSlotsPerTransition": 4,
        "actionRebound": False,
        "costSemanticsRebound": False,
        "outcomeFieldsLoaded": False,
        "passed": True,
    }
    body["bindingCommitmentSha256"] = canonical_sha256(
        "E07/S12A/binding-commitment/v1", body
    )
    return body


def canonical_binding_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        "E07/S12A/binding-set/v1",
        sorted(
            (dict(row) for row in rows),
            key=lambda row: str(row["bindingCommitmentSha256"]),
        ),
    )


class DiagnosticSpatialTracker:
    """Track native diagnostic fields while making completion unavailable."""

    def __init__(
        self,
        environment: Environment,
        reference: Mapping[str, str],
    ) -> None:
        if set(reference) != {site.site_id for site in environment.occupiable_sites}:
            raise SuiteValidationError("diagnostic reference site set mismatch")
        self.environment = environment
        self.reference = dict(reference)
        self.initial: Mapping[str, Any] | None = None
        self.terminal: Mapping[str, Any] | None = None
        self.minimum_mismatch = len(reference)
        self.observation_count = 0
        self.state_hashes: list[str] = []

    def observe(
        self,
        _transition_index: int,
        state: MovementState,
        _summary: Mapping[str, Any],
    ) -> None:
        assignment = _state_assignment(state)
        diagnostics = graph_diagnostics(self.environment, assignment, self.reference)
        if self.initial is None:
            self.initial = diagnostics
        self.terminal = diagnostics
        self.minimum_mismatch = min(
            self.minimum_mismatch,
            int(diagnostics["referenceMismatchCount"]),
        )
        self.observation_count += 1
        self.state_hashes.append(movement_state_sha256(state))

    def finalize(self) -> dict[str, Any]:
        if self.initial is None or self.terminal is None:
            raise SuiteValidationError("diagnostic tracker received no states")
        return {
            "schemaVersion": "e07.s12a.diagnostic-endpoint.v1",
            "evaluationMode": "diagnostic_only",
            "topologyCompletionCalibrated": False,
            "terminalS01GlobalSuccess": None,
            "terminalS01MismatchFraction": None,
            "terminalS02GrammarAccepted": None,
            "terminalS02RelationalScore": None,
            "conjunctiveCompletionByBudget": None,
            "formationCompletion": None,
            "repairCompletion": None,
            "endpointTime": None,
            "endpointUnavailableReason": DIAGNOSTIC_UNAVAILABLE_REASON,
            "initialDiagnostics": dict(self.initial),
            "terminalDiagnostics": dict(self.terminal),
            "minimumReferenceMismatchCount": self.minimum_mismatch,
            "observationCount": self.observation_count,
            "stateSequenceSha256": _sha256(
                "E07/S12A/diagnostic-state-sequence/v1", self.state_hashes
            ),
            "diagnosticPromotionEligible": False,
        }


def run_fail_atomic_fixture_batch(
    item_ids: Sequence[str],
    evaluator: Callable[[str], Mapping[str, Any]],
    *,
    failure_position: int | None,
) -> dict[str, Any]:
    """Execute synthetic qualification items with complete dispositions.

    Result publication is all-or-zero.  Completed values are represented only
    by commitments because qualification fixtures cannot become efficacy rows.
    """

    if len(item_ids) != len(set(item_ids)):
        raise SuiteValidationError("qualification item IDs must be unique")
    dispositions = [
        {
            "position": position,
            "itemId": item_id,
            "state": "precommitted_not_attempted",
            "resultCommitmentSha256": None,
            "errorType": None,
        }
        for position, item_id in enumerate(item_ids)
    ]
    success = True
    for position, item_id in enumerate(item_ids):
        try:
            if failure_position == position:
                raise RuntimeError("injected qualification failure")
            result = dict(evaluator(item_id))
            dispositions[position].update(
                {
                    "state": "qualification_succeeded",
                    "resultCommitmentSha256": _sha256(
                        "E07/S12A/qualification-result/v1", result
                    ),
                }
            )
        except Exception as error:  # exact forensic disposition is the contract
            success = False
            dispositions[position].update(
                {
                    "state": "qualification_failed",
                    "errorType": type(error).__name__,
                }
            )
            for later in range(position + 1, len(dispositions)):
                dispositions[later]["state"] = "not_attempted_after_failure"
            break
    return {
        "schemaVersion": "e07.s12a.fail-atomic-fixture-batch.v1",
        "precommitted": len(item_ids),
        "attempted": sum(
            row["state"] in {"qualification_succeeded", "qualification_failed"}
            for row in dispositions
        ),
        "succeeded": sum(
            row["state"] == "qualification_succeeded" for row in dispositions
        ),
        "failed": sum(row["state"] == "qualification_failed" for row in dispositions),
        "notAttempted": sum(
            row["state"] == "not_attempted_after_failure" for row in dispositions
        ),
        "publicationRows": len(item_ids) if success else 0,
        "allOrZeroPublication": True,
        "success": success,
        "failurePosition": failure_position,
        "dispositions": dispositions,
    }
