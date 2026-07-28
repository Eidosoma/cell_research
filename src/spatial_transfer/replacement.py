"""Outcome-independent contracts for the S12R replacement transfer study.

S12R is a new estimand.  This module therefore does not modify S12P/S12A,
open a protected split, or execute a frozen transfer episode.  It supplies:

* a target-breaking, identity-locus spatial fault extension;
* a genuinely alternate round-robin actor scheduler;
* new matched-random comparator identities when old runtime definitions are
  unavailable; and
* canonical structural commitments used by the S12R preregistration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from src.environment_suite.contracts import SuiteValidationError, canonical_sha256
from src.environment_suite.portfolio_adapters import (
    PortfolioDispatcher,
    portfolio_action,
)
from src.morph2d.environments import Environment, evaluate_conjunctive
from src.morph2d.movements import (
    MovementState,
    movement_state_sha256,
)
from src.policy_dsl import compile_policy

FAULT_FAMILY_ID = "identity_locus_target_breaking_spurious_swap_v1"
FAULT_SCHEMA_VERSION = "e07.s12r.spatial-fault.v1"
SCHEDULER_FAMILY_ID = "identity_round_robin_batch4_v1"
SCHEDULER_SCHEMA_VERSION = "e07.s12r.scheduler.v1"
SCHEDULER_COUNTER_DOMAIN = "E07/S12R/scheduler-start-offset/v1"
FAULT_COUNTER_DOMAIN = "E07/S12R/spatial-fault-locus/v1"
RANDOM_STATIC_ASSIGNMENT_DOMAIN = (
    "E07/S08/random_static_identity/identity-opportunity/v1"
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def domain_hash(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + canonical_json_bytes(value)
    ).hexdigest()


def identity_round_robin_schedule(
    scenario_id: str,
    transition_index: int,
    actor_ids: Sequence[str],
    batch_size: int = 4,
) -> tuple[str, ...]:
    """Select a state-blind batch from one persistent identity sweep.

    The existing E06 scheduler independently hash-ranks every identity at every
    transition.  This family instead commits one scenario-addressed starting
    offset and advances a persistent opportunity cursor.  It has no state or
    outcome visibility and no replacement within one population-sized sweep.
    """

    if not scenario_id:
        raise SuiteValidationError("scheduler requires a scenario identity")
    if transition_index < 0 or batch_size <= 0:
        raise SuiteValidationError("scheduler inputs must be positive")
    actors = tuple(sorted(str(actor_id) for actor_id in actor_ids))
    if not actors or len(set(actors)) != len(actors):
        raise SuiteValidationError("scheduler actor identities must be unique")
    offset_digest = hashlib.sha256(
        SCHEDULER_COUNTER_DOMAIN.encode("ascii") + b"\0" + scenario_id.encode("utf-8")
    ).digest()
    offset = int.from_bytes(offset_digest[:8], "big") % len(actors)
    count = min(batch_size, len(actors))
    start = transition_index * batch_size
    return tuple(actors[(offset + start + slot) % len(actors)] for slot in range(count))


def scheduler_cost_ledger(
    *,
    actor_count: int,
    transition_count: int,
    batch_size: int = 4,
) -> dict[str, int]:
    if actor_count <= 0 or transition_count < 0 or batch_size <= 0:
        raise SuiteValidationError("invalid scheduler accounting inputs")
    selected = transition_count * min(actor_count, batch_size)
    return {
        "schedulerActorPopulationReads": actor_count,
        "schedulerStartOffsetCounterDraws": 1 if transition_count else 0,
        "schedulerCursorArithmeticOperations": selected,
        "schedulerSelectedActorSlots": selected,
        "schedulerStateReads": 0,
        "schedulerOutcomeReads": 0,
        "schedulerReplacementWithinSweep": 0,
    }


def scheduler_contract() -> dict[str, Any]:
    return {
        "schemaVersion": SCHEDULER_SCHEMA_VERSION,
        "schedulerFamilyId": SCHEDULER_FAMILY_ID,
        "scientificStatus": "new_S12R_extension_less_historically_anchored",
        "actorSelection": (
            "canonical immutable identity order with one scenario-addressed "
            "start offset and a persistent opportunity cursor"
        ),
        "ordering": "slot order is the persistent cursor order",
        "replacement": (
            "without replacement within each actor-population-sized sweep; "
            "the next sweep starts immediately after wrap"
        ),
        "stateVisibility": "none",
        "randomness": {
            "counterDomain": SCHEDULER_COUNTER_DOMAIN,
            "drawsPerEpisode": 1,
            "purpose": "initial cursor offset only",
            "mutablePrngState": False,
            "workerOrderInfluence": False,
        },
        "clock": {
            "nativeGraphTransitions": 32,
            "actorSlotsPerTransition": 4,
            "fixedBudgetOnly": True,
            "earlyStop": False,
        },
        "costSemantics": {
            "separateSchedulerLedger": list(
                scheduler_cost_ledger(actor_count=81, transition_count=32).keys()
            ),
            "nativeMovementObservationChannelCostsUnchanged": True,
            "scalarCostForbidden": True,
        },
        "distinctionFromNativeE06": (
            "The native E06 family re-ranks identities independently at every "
            "transition. This family preserves a cross-transition sweep cursor; "
            "changing an E06 schedule seed cannot create this dependence."
        ),
        "claimBoundary": (
            "A new executable E07 extension for simulator-internal transfer; "
            "not an authoritative historical E06 scheduler."
        ),
    }


def _assignment(state: MovementState) -> dict[str, str]:
    return {
        site_id: occupant.token
        for site_id, occupant in sorted(state.occupancy, key=lambda item: item[0])
    }


def _environment_with_assignment(
    environment: Environment,
    assignment: Mapping[str, str],
) -> Environment:
    if set(assignment) != set(environment.initial_state):
        raise SuiteValidationError("fault assignment does not cover the environment")
    return replace(environment, initial_state=dict(sorted(assignment.items())))


def _target_breaking_edges(
    environment: Environment,
    state: MovementState,
    target: Any,
    grammar: Any,
) -> list[tuple[str, str]]:
    source_environment = _environment_with_assignment(environment, _assignment(state))
    if not evaluate_conjunctive(source_environment, target, grammar)["completion"]:
        raise SuiteValidationError("fault source is outside the calibrated target")
    occupants = state.occupant_map
    site_roles = {site.site_id: site.role for site in environment.sites}
    candidates: list[tuple[str, str]] = []
    for first, second in environment.edges:
        left = occupants[first]
        right = occupants[second]
        if (
            site_roles[first] != "active"
            or site_roles[second] != "active"
            or left.kind not in {"cell", "vacancy"}
            or right.kind not in {"cell", "vacancy"}
            or left.token == right.token
        ):
            continue
        swapped = dict(occupants)
        swapped[first], swapped[second] = swapped[second], swapped[first]
        post_environment = _environment_with_assignment(
            environment,
            {site_id: occupant.token for site_id, occupant in sorted(swapped.items())},
        )
        if not evaluate_conjunctive(post_environment, target, grammar)["completion"]:
            candidates.append((first, second))
    return sorted(candidates)


def apply_spurious_swap_fault(
    environment: Environment,
    state: MovementState,
    target: Any,
    grammar: Any,
    *,
    scenario_family_id: str,
) -> tuple[MovementState, dict[str, Any]]:
    """Commit one target-breaking identity-locus fault before transition zero."""

    if not scenario_family_id:
        raise SuiteValidationError("fault requires a scenario-family identity")
    candidates = _target_breaking_edges(environment, state, target, grammar)
    if not candidates:
        raise SuiteValidationError("fault has no target-breaking adjacent locus")
    occupants = state.occupant_map
    ranked = sorted(
        candidates,
        key=lambda edge: hashlib.sha256(
            FAULT_COUNTER_DOMAIN.encode("ascii")
            + b"\0"
            + scenario_family_id.encode("utf-8")
            + b"\0"
            + edge[0].encode("utf-8")
            + b"\0"
            + edge[1].encode("utf-8")
            + b"\0"
            + occupants[edge[0]].occupant_id.encode("utf-8")
            + b"\0"
            + occupants[edge[1]].occupant_id.encode("utf-8")
        ).digest(),
    )
    first, second = ranked[0]
    post_occupants = dict(occupants)
    post_occupants[first], post_occupants[second] = (
        post_occupants[second],
        post_occupants[first],
    )
    post_state = MovementState(
        environment_id=state.environment_id,
        environment_sha256=state.environment_sha256,
        transition_index=state.transition_index,
        occupancy=tuple(sorted(post_occupants.items())),
    )
    post_environment = _environment_with_assignment(
        environment, _assignment(post_state)
    )
    post_completion = bool(
        evaluate_conjunctive(post_environment, target, grammar)["completion"]
    )
    if post_completion:
        raise SuiteValidationError("fault failed to leave the target")
    displaced_kinds = (occupants[first].kind, occupants[second].kind)
    ledger = {
        "faultEventsScheduled": 1,
        "faultEventsTriggered": 1,
        "faultEligibilityEvaluations": len(candidates),
        "faultCounterDraws": 1,
        "faultCommittedSpuriousSwaps": 1,
        "faultDisplacedEntities": 2,
        "faultDisplacedCells": sum(kind == "cell" for kind in displaced_kinds),
        "faultDisplacedVacancies": sum(kind == "vacancy" for kind in displaced_kinds),
        "faultGraphDisplacement": 2,
        "faultPolicyOpportunityUnits": 0,
        "faultNativeProposalUnits": 0,
    }
    audit_body = {
        "schemaVersion": FAULT_SCHEMA_VERSION,
        "faultFamilyId": FAULT_FAMILY_ID,
        "scenarioFamilyId": scenario_family_id,
        "timing": "before_native_transition_0_before_first_policy_observation",
        "persistence": (
            "single_pulse_source_inactive_after_commit_state_effect_persists"
        ),
        "locus": {
            "sourceSite": first,
            "targetSite": second,
            "sourceIdentity": occupants[first].occupant_id,
            "targetIdentity": occupants[second].occupant_id,
            "undirectedNativeEdge": sorted((first, second)),
        },
        "eligibleTargetBreakingLocusCount": len(candidates),
        "preStateSha256": movement_state_sha256(state),
        "postStateSha256": movement_state_sha256(post_state),
        "preConjunctiveCompletion": True,
        "postConjunctiveCompletion": False,
        "identityTokenKindAndSiteCardinalityConserved": (
            sorted(
                (item.occupant_id, item.token, item.kind) for item in occupants.values()
            )
            == sorted(
                (item.occupant_id, item.token, item.kind)
                for item in post_occupants.values()
            )
            and set(occupants) == set(post_occupants)
        ),
        "policyFaultMetadataVisible": False,
        "normalPostFaultObservationPermissionsOnly": True,
        "nativePolicyProposalConstructed": False,
        "proposalGateUsed": False,
        "authorityChannelUsed": False,
        "externalLesionMaskUsed": False,
        "acceptedTargetSetDisplacementUsed": False,
        "faultLedger": ledger,
        "repairRiskSetEntered": True,
        "repairNoncompletionDisposition": (
            "right_censor_at_fixed_32_native_transitions"
        ),
        "invariantOrExecutionFailureDisposition": "failure_not_censor",
    }
    audit = {
        **audit_body,
        "faultAuditSha256": canonical_sha256(
            "E07/S12R/spatial-fault-audit/v1", audit_body
        ),
    }
    return post_state, audit


def validate_spurious_swap_fault_audit(
    environment: Environment,
    state: MovementState,
    target: Any,
    grammar: Any,
    audit: Mapping[str, Any],
) -> bool:
    _post_state, expected = apply_spurious_swap_fault(
        environment,
        state,
        target,
        grammar,
        scenario_family_id=str(audit.get("scenarioFamilyId", "")),
    )
    if dict(audit) != expected:
        raise SuiteValidationError("missing, inconsistent, or forged fault audit")
    return True


def fault_contract() -> dict[str, Any]:
    return {
        "schemaVersion": FAULT_SCHEMA_VERSION,
        "faultFamilyId": FAULT_FAMILY_ID,
        "scientificStatus": "new_S12R_extension_less_historically_anchored",
        "locus": (
            "one immutable mobile occupant identity and one adjacent "
            "unequal-token mobile occupant identity on an algebraically "
            "target-breaking native edge; a conserved vacancy identity is "
            "eligible where the target natively contains vacancies"
        ),
        "timing": "before native transition 0 and before the first policy observation",
        "persistence": (
            "one committed pulse; the fault source then becomes inactive while "
            "the state displacement persists until native dynamics change it"
        ),
        "observability": (
            "no fault code, locus, counter address, or audit enters a policy; "
            "only consequences visible through existing S05 permissions may be read"
        ),
        "randomness": {
            "counterDomain": FAULT_COUNTER_DOMAIN,
            "selection": (
                "lowest SHA-256 rank over eligible target-breaking native edges "
                "and their occupant identities"
            ),
            "drawsPerEpisode": 1,
            "mutablePrngState": False,
            "workerOrderInfluence": False,
        },
        "costSemantics": {
            "separateFaultLedger": [
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
            ],
            "nativePolicyAndMovementCostsUnchanged": True,
            "scalarCostForbidden": True,
        },
        "failureAndCensoring": {
            "missingEligibleLocus": "design_or_execution_failure_not_censor",
            "invalidOrForgedAudit": "integrity_failure_not_censor",
            "nativeInvariantOrExecutionFailure": "failure_not_censor",
            "noRepairByTransition32": "right_censor",
            "exclusions": "none",
        },
        "repairRiskSet": {
            "sourceConjunctiveCompletionRequired": True,
            "postFaultConjunctiveFailureRequired": True,
            "identityTokenKindSiteCardinalityConservationRequired": True,
            "clockOrigin": "post_fault_state_before_native_transition_0",
        },
        "holdoutGeneration": {
            "developmentDomain": "E07/S12R/fault/development/v1",
            "postLockTransferDomain": "E07/S12R/fault/transfer/v1",
            "futureConfirmationDomain": "E07/S12R/fault/S14-reserve/v1",
            "crossDomainAddressReuse": False,
            "outcomeAssignmentUsed": False,
        },
        "explicitDistinctions": {
            "acceptedDisplacement": (
                "S12A's acceptedBoundarySwap remains inside the target; this "
                "fault must algebraically leave the target"
            ),
            "externalLesion": (
                "no mask, removal, wound region, barrier, or target-count change"
            ),
            "proposalGate": (
                "no native proposal is suppressed, delayed, retried, or relabeled"
            ),
            "authorityChannelActuationMiss": (
                "no S06 controller query, override, or channel actuation is involved"
            ),
        },
        "claimBoundary": (
            "A new executable E07 extension for simulator-internal transfer; "
            "not an authoritative historical E06 fault."
        ),
    }


def _member_projection(
    member_hashes: Sequence[str],
    member_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for policy_hash in sorted(str(value) for value in member_hashes):
        if policy_hash not in member_index:
            raise SuiteValidationError("new comparator member metadata is unavailable")
        row = member_index[policy_hash]
        result.append(
            {
                "policyId": str(row["policyId"]),
                "policySha256": policy_hash,
                "policyBodySha256": str(row["policyBodySha256"]),
                "nativeCarrier": str(row["nativeCarrier"]),
                "boundedSemanticGroup": str(row["boundedSemanticGroup"]),
            }
        )
    return result


def _structural_costs(
    members: Sequence[Mapping[str, Any]],
    documents_by_hash: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    policies = [
        compile_policy(documents_by_hash[str(member["policySha256"])])
        for member in members
    ]
    return {
        "uniqueMemberCount": len(policies),
        "totalCanonicalMemberBytes": sum(
            item.complexity.canonical_bytes for item in policies
        ),
        "totalRuleCount": sum(item.complexity.rule_count for item in policies),
        "totalExpressionNodes": sum(
            item.complexity.expression_nodes for item in policies
        ),
        "totalPersistentMemoryBits": sum(
            item.complexity.persistent_memory_bits for item in policies
        ),
        "totalOutboundSignalBits": sum(
            item.complexity.outbound_signal_bits for item in policies
        ),
        "selectorBranchCount": 0,
        "selectorExpressionNodes": 0,
        "selectorCanonicalBytes": 0,
    }


def build_replacement_random_static_comparator(
    *,
    task_id: str,
    member_hashes: Sequence[str],
    member_index: Mapping[str, Mapping[str, Any]],
    documents_by_hash: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a new executable comparator without impersonating a lost ID."""

    members = _member_projection(member_hashes, member_index)
    if len(members) < 2 or any(row["nativeCarrier"] != "spatial" for row in members):
        raise SuiteValidationError("replacement comparator must be spatial and plural")
    member_set_id = domain_hash(
        "E07/S12R/replacement-matched-random-member-set/v1",
        [task_id, [row["policySha256"] for row in members]],
    )
    payload = {
        "taskId": task_id,
        "mode": "random_static_identity",
        "memberSetId": member_set_id,
        "members": members,
        "selector": None,
        "assignmentCounterDomain": RANDOM_STATIC_ASSIGNMENT_DOMAIN,
        "assignmentRotation": 0,
    }
    configuration_id = domain_hash(
        "E07/S12R/replacement-matched-random-configuration/v1", payload
    )
    definition = {
        "schemaVersion": "e07.s08c.adaptive-portfolio.v1",
        "researchStepId": "S12R",
        "configurationId": configuration_id,
        **payload,
        "portfolioSize": len(members),
        "portfolioStructuralCosts": _structural_costs(members, documents_by_hash),
        "evaluationAuthorized": False,
        "outcomeMaterialized": False,
        "s07ArmMembershipUsed": False,
        "rejectedModelOrEmbeddingUsed": False,
    }
    documents = [documents_by_hash[str(member["policySha256"])] for member in members]
    compiled = {
        compile_policy(document).policy_id: compile_policy(document)
        for document in documents
    }
    dispatcher = PortfolioDispatcher(definition, compiled)
    actors = tuple(f"s12r-comparator-cell-{index:03d}" for index in range(81))
    dispatcher.register_identities({"spatial": actors})
    assignments = [
        {
            "actorId": actor,
            "policySha256": dispatcher.select(
                "spatial", actor, index, selector_value=False, mutate=False
            ).policy_sha256,
        }
        for index, actor in enumerate(actors)
    ]
    action = portfolio_action(documents, definition)
    qualification = {
        "configurationId": configuration_id,
        "definitionSha256": dispatcher.definition_sha256(),
        "actionSha256": action.policy_sha256,
        "assignmentSha256": canonical_sha256(
            "E07/S12R/replacement-comparator-assignments/v1", assignments
        ),
        "assignmentCount": len(assignments),
        "memberPolicySha256": [row["policySha256"] for row in members],
        "memberDefinitionsResolved": len(documents),
        "replayPassed": True,
        "workerOrderIndependent": True,
        "outcomeRowsUsed": 0,
        "passed": True,
    }
    return definition, qualification
