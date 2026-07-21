"""Frozen S12 bounded hybrid-control factorial.

This module composes the validated S04--S10 contracts. It does not introduce
a full-state planner: the only central authority is S06's one-lagged-epoch,
state-blind-recipient, one-query, one-action adapter. Whole-target S01 and
whole-grid S02 evaluation remains an offline callback.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from .baseline import (
    TargetMetricTracker,
    load_baseline_assets,
    make_initial_state,
    state_grid,
)
from .engine import EpisodeDefinition, canonical_episode_result_bytes, run_cpu_episode
from .grammar import score_grid
from .movements import MovementState, initial_movement_state, movement_state_sha256
from .perturbations import LesionApplication, apply_lesion
from .targets import evaluate_success


ROOT = Path(__file__).resolve().parents[2]
HYBRID_CATALOG_VERSION = "e06.s12.hybrid-control-catalog.v1"
HYBRID_RUN_VERSION = "e06.s12.hybrid-control-run.v1"
MASTER_SEED_HEX = "0xe0612000000000000000000000000001"
DIRECT_MODES = {"central_only", "sparse_direct", "combined"}
NONEXACT_CHALLENGES = {
    "partially_correct_formation",
    "compact_wound_mild",
    "formed_displacement_mild",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_value(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_bytes(value)
    ).hexdigest()


@lru_cache(maxsize=1)
def load_hybrid_catalog() -> Mapping[str, Any]:
    path = ROOT / "configs/morphologies/hybrid_control_catalog.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_hybrid_catalog(raw)
    return raw


@lru_cache(maxsize=1)
def load_upstream_designs() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    baseline = yaml.safe_load(
        (ROOT / "configs/morphologies/baseline_catalog.yaml").read_text(
            encoding="utf-8"
        )
    )
    perturbation = yaml.safe_load(
        (ROOT / "configs/morphologies/perturbation_catalog.yaml").read_text(
            encoding="utf-8"
        )
    )
    return baseline, perturbation


def validate_hybrid_catalog(catalog: Mapping[str, Any]) -> None:
    required = {
        "schemaVersion",
        "researchStepId",
        "title",
        "frozenQuestion",
        "claimBoundary",
        "canonicalContracts",
        "seedContract",
        "topologyAndFeasibility",
        "controlArms",
        "scenarioFactorial",
        "simulation",
        "budgetEnvelope",
        "outcomes",
        "factorialInteractions",
        "censoring",
        "promotion",
        "confirmation",
        "pareto",
        "traceAndReplay",
        "validationRequirements",
    }
    if set(catalog) != required:
        raise ValueError("S12 catalog top-level schema mismatch")
    if (
        catalog["schemaVersion"] != HYBRID_CATALOG_VERSION
        or catalog["researchStepId"] != "S12"
    ):
        raise ValueError("S12 catalog version mismatch")
    arms = catalog["controlArms"]
    if not isinstance(arms, list) or {item["armId"] for item in arms} != {
        "local_only",
        "central_only",
        "gradient_only",
        "sparse_direct",
        "combined",
    }:
        raise ValueError("S12 requires exactly the five frozen primary arms")
    if not all(bool(item["primary"]) for item in arms):
        raise ValueError("all S12 control arms must be primary")
    factorial = catalog["scenarioFactorial"]["factors"]
    expected_conditions = (
        len(factorial["target"])
        * len(factorial["challenge"])
        * len(factorial["controlArm"])
    )
    simulation = catalog["simulation"]
    if expected_conditions != int(simulation["exploratoryConditionCount"]):
        raise ValueError("S12 condition-count mismatch")
    if expected_conditions * int(
        simulation["exploratoryReplicatesPerCondition"]
    ) != int(simulation["exploratoryRunCount"]):
        raise ValueError("S12 exploratory run-count mismatch")
    if int(simulation["allocatedActorSlotsPerTransition"]) != 4:
        raise ValueError("S12 actor-opportunity envelope changed")
    if catalog["topologyAndFeasibility"]["countChangingConditions"] != "forbidden":
        raise ValueError("S12 cannot enable count-changing conditions")


def arm_catalog(catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["armId"]): item for item in catalog["controlArms"]}


def scenario_identity(
    split: str,
    target_id: str,
    challenge_id: str,
    replicate: int,
    arm_id: str,
) -> dict[str, Any]:
    address = {
        "split": split,
        "targetId": target_id,
        "challengeId": challenge_id,
        "replicate": int(replicate),
    }
    seeded = {"masterSeedHex": MASTER_SEED_HEX, "address": address}
    pairing_digest = sha256_value("E06/S12/pairing-block/v1", seeded)
    seed_digest = sha256_value("E06/S12/seed/v1", seeded)
    return {
        "scenarioId": f"s12-{split[:4]}-{pairing_digest[:24]}",
        "pairingBlockId": "pb1:" + pairing_digest,
        "runId": "run1:" + sha256_value("E06/S12/run/v1", {**seeded, "armId": arm_id}),
        "seedHex": "0x" + seed_digest[:32],
        "seedDecimal": str(int(seed_digest[:32], 16)),
    }


def _zero_external_ledger() -> dict[str, int]:
    return {
        "externalDisplacedEntities": 0,
        "externalDisplacedCells": 0,
        "externalDisplacedVacancies": 0,
        "externalGraphDisplacement": 0,
    }


def make_challenge_state(
    target_id: str,
    challenge_id: str,
    pairing_block_id: str,
) -> tuple[MovementState, Mapping[str, Any], Mapping[str, int], bool, bool]:
    """Return state, mask, external ledger, formed source, and feasibility."""

    _, targets, _, environments = load_baseline_assets()
    target = targets[target_id]
    environment = environments[target_id]
    baseline_catalog, perturbation_catalog = load_upstream_designs()
    exact = initial_movement_state(environment)
    if challenge_id == "exact_maintenance":
        mask = {
            "schemaVersion": "e06.s12.challenge-mask.v1",
            "challengeId": challenge_id,
            "lesionMaskSha256": None,
        }
        return exact, mask, _zero_external_ledger(), True, True
    if challenge_id == "partially_correct_formation":
        state = make_initial_state(
            environment,
            target,
            "partially_correct",
            pairing_block_id,
            baseline_catalog,
        )
        mask = {
            "schemaVersion": "e06.s12.challenge-mask.v1",
            "challengeId": challenge_id,
            "initialStateSha256": movement_state_sha256(state),
            "lesionMaskSha256": None,
        }
        return state, mask, _zero_external_ledger(), False, True
    lesion_id = {
        "compact_wound_mild": "compact_wound",
        "formed_displacement_mild": "formed_structure_displacement",
    }.get(challenge_id)
    if lesion_id is None:
        raise ValueError("unknown S12 challenge")
    application: LesionApplication = apply_lesion(
        environment,
        target,
        exact,
        lesion_id,
        "mild",
        pairing_block_id,
        perturbation_catalog,
    )
    return (
        application.state,
        {**application.mask, "challengeId": challenge_id},
        application.external_ledger,
        True,
        bool(application.target_feasible),
    )


def _episode_definition(
    catalog: Mapping[str, Any],
    arm_id: str,
    target_record: Mapping[str, Any],
    environment_id: str,
    scenario_id: str,
    transitions: int,
) -> EpisodeDefinition:
    arm = arm_catalog(catalog)[arm_id]
    policy_id = str(arm["nativePolicyId"])
    relation_grammar_id = (
        str(target_record["grammarId"])
        if policy_id == "greedy_neighbor_satisfaction_v1"
        else None
    )
    parameters: dict[str, Any] = {}
    if arm_id == "gradient_only":
        parameters = {
            "gradientAxis": str(target_record["gradientAxis"]),
            "gradientDirection": str(target_record["gradientDirection"]),
        }
    if arm_id in DIRECT_MODES:
        parameters["summaryRecipients"] = 1
        if target_record.get("directCandidateLimit") is not None:
            parameters["directCandidateLimit"] = int(
                target_record["directCandidateLimit"]
            )
    return EpisodeDefinition(
        scenario_id=scenario_id,
        environment_id=environment_id,
        policy_id=policy_id,
        relation_grammar_id=relation_grammar_id,
        channel_mode=str(arm["channelMode"]),
        transitions=int(transitions),
        actor_batch_size=int(
            arm.get(
                "actorBatchSize",
                catalog["simulation"]["allocatedActorSlotsPerTransition"],
            )
        ),
        parameters=parameters,
    )


def _policy_source_reads(observation: Mapping[str, int]) -> int:
    return sum(
        int(observation[field])
        for field in (
            "actorTokenReads",
            "localTokenReads",
            "candidateDescriptorReads",
            "boundarySignalReads",
            "gradientSignalReads",
            "laggedConflictReads",
            "memoryReads",
            "counterRandomDraws",
            "affordanceTopologyReads",
            "affordanceSiteRoleReads",
        )
    )


def _budget_audit(
    catalog: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    envelope = catalog["budgetEnvelope"]
    transitions = int(result["transitionBudget"])
    epochs = math.ceil(
        transitions / int(catalog["simulation"]["epochLengthTransitions"])
    )
    direct_epochs = max(0, epochs - int(envelope["firstDirectEpoch"]))
    channel = result["channelLedger"]
    checks = {
        "configurationBits": int(channel["configurationBits"])
        <= int(envelope["perScenarioConfigurationBits"]),
        "controllerInputBits": int(channel["controllerInputBits"])
        <= direct_epochs * int(envelope["perEligibleDirectEpochControllerInputBits"]),
        "policyDeliveryBits": int(channel["policyDeliveryBits"])
        <= epochs * int(envelope["perEpochPolicyDeliveryBits"]),
        "addressBits": int(channel["addressBits"])
        <= direct_epochs * int(envelope["perEligibleDirectEpochAddressBits"]),
        "actuationAttempts": int(channel["actuationAttempts"])
        <= direct_epochs * int(envelope["perEligibleDirectEpochActuationAttempts"]),
        "overrideActionUnits": int(channel["overrideActionUnits"])
        <= direct_epochs * int(envelope["perEligibleDirectEpochOverrideActionUnits"]),
        "movementGraphDisplacement": int(channel["movementGraphDisplacement"])
        <= direct_epochs
        * int(envelope["perEligibleDirectEpochMovementGraphDisplacement"]),
        "informationIdentity": int(channel["totalInformationBits"])
        == int(channel["configurationBits"])
        + int(channel["controllerInputBits"])
        + int(channel["policyDeliveryBits"])
        + int(channel["addressBits"]),
        "configurationChargedOnce": bool(
            result["configurationAccounting"]["chargedOnceInFull"]
        )
        and not bool(result["configurationAccounting"]["amortizedOrDivided"]),
    }
    return {
        "epochs": epochs,
        "eligibleDirectEpochs": direct_epochs,
        "checks": checks,
        "success": all(checks.values()),
    }


def run_hybrid_once(
    specification: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any] | None, Mapping[str, Any]]:
    catalog = specification.get("catalog") or load_hybrid_catalog()
    validate_hybrid_catalog(catalog)
    context, targets, grammars, environments = load_baseline_assets()
    target_id = str(specification["targetId"])
    challenge_id = str(specification["challengeId"])
    arm_id = str(specification["armId"])
    target = targets[target_id]
    target_record = dict(
        next(
            item
            for item in catalog["topologyAndFeasibility"]["targets"]
            if item["targetId"] == target_id
        )
    )
    if specification.get("directCandidateLimit") is not None:
        target_record["directCandidateLimit"] = int(
            specification["directCandidateLimit"]
        )
    grammar = grammars[str(target_record["grammarId"])]
    environment = environments[target_id]
    identity = dict(
        specification.get("identityOverride")
        or scenario_identity(
            str(specification["split"]),
            target_id,
            challenge_id,
            int(specification["replicate"]),
            arm_id,
        )
    )
    if set(identity) != {
        "scenarioId",
        "pairingBlockId",
        "runId",
        "seedHex",
        "seedDecimal",
    }:
        raise ValueError("hybrid identity override schema mismatch")
    initial_state, mask, external, formed_source, feasible = make_challenge_state(
        target_id, challenge_id, identity["pairingBlockId"]
    )
    initial_grid = state_grid(environment, initial_state)
    initial_global = evaluate_success(initial_grid, target)
    initial_local = score_grid(initial_grid, grammar)
    initial_conjunctive = bool(initial_global["success"] and initial_local["accepted"])
    if challenge_id == "exact_maintenance" and not initial_conjunctive:
        raise ValueError("S12 maintenance source is not conjunctively exact")
    if challenge_id in NONEXACT_CHALLENGES and initial_conjunctive:
        raise ValueError("S12 non-exact challenge must begin outside conjunction")
    if not feasible:
        raise ValueError("S12 launched an infeasible challenge")
    tracker = TargetMetricTracker(environment, target, grammar)
    definition = _episode_definition(
        catalog,
        arm_id,
        target_record,
        environment.environment_id,
        identity["scenarioId"],
        int(specification["eventBudget"]),
    )
    started = time.perf_counter()
    direct_epochs = (
        None
        if specification.get("directEpochs") is None
        else frozenset(int(value) for value in specification["directEpochs"])
    )
    action_relay_enabled = bool(specification.get("directActionRelayEnabled", True))
    result = run_cpu_episode(
        context,
        definition,
        include_selected_traces=bool(specification.get("retainTrace", False)),
        initial_state_override=initial_state,
        state_audit=tracker.observe,
        native_batch_gate=(lambda _index: False) if arm_id == "central_only" else None,
        direct_epoch_gate=(
            None if direct_epochs is None else lambda epoch: epoch in direct_epochs
        ),
        direct_action_gate=(None if action_relay_enabled else lambda _epoch: False),
    )
    elapsed = time.perf_counter() - started
    metrics = tracker.finalize()
    final_state = tracker.last_state
    if final_state is None:
        raise ValueError("S12 tracker has no final state")
    movement = result["movementLedger"]
    observation = result["observationLedger"]
    channel = result["channelLedger"]
    budget_audit = _budget_audit(catalog, result)
    allocated_slots = int(
        catalog["simulation"]["allocatedActorSlotsPerTransition"]
    ) * int(specification["eventBudget"])
    used_native_slots = sum(
        int(item["scheduledActorCount"])
        for item in result["transitionSummaries"]
        if not bool(item["directIntervention"])
    )
    direct_query_slots = sum(
        int(item["scheduledActorCount"])
        for item in result["transitionSummaries"]
        if bool(item["directIntervention"])
    )
    foregone_slots = allocated_slots - used_native_slots
    policy_reads = _policy_source_reads(observation)
    source_work = (
        policy_reads
        + int(channel["sourceScalarReads"])
        + int(channel["controllerStateReads"])
    )
    computation = (
        int(observation["utilityEvaluations"])
        + int(observation["comparisonOperations"])
        + int(channel["controllerComputeUnits"])
    )
    opportunity = (
        foregone_slots
        + int(channel["suppressedNativeActions"])
        + int(channel["opportunityCostUnits"])
    )
    episode_bytes = canonical_episode_result_bytes(result)
    initial_ids = {item.occupant_id for _, item in initial_state.occupancy}
    final_ids = {item.occupant_id for _, item in final_state.occupancy}
    initial_tokens = Counter(item.token for _, item in initial_state.occupancy)
    final_tokens = Counter(item.token for _, item in final_state.occupancy)
    initial_kinds = Counter(item.kind for _, item in initial_state.occupancy)
    final_kinds = Counter(item.kind for _, item in final_state.occupancy)
    absolute_family = str(
        catalog["scenarioFactorial"]["challengeContracts"][challenge_id][
            "absoluteEndpointFamily"
        ]
    )
    nonexact = challenge_id in NONEXACT_CHALLENGES
    repair_eligible = challenge_id in {
        "compact_wound_mild",
        "formed_displacement_mild",
    }
    intervention = {
        "schemaVersion": "e06.s12.intervention-summary.v1",
        "pairingBlockId": identity["pairingBlockId"],
        "challengeId": challenge_id,
        "lesionMaskSha256": mask.get("lesionMaskSha256"),
        "formedSource": formed_source,
        "targetFeasible": feasible,
        **{key: int(value) for key, value in external.items()},
    }
    intervention["interventionSummarySha256"] = sha256_value(
        "E06/S12/intervention-summary/v1", intervention
    )
    row: dict[str, Any] = {
        "schemaVersion": HYBRID_RUN_VERSION,
        "phase": str(specification["phase"]),
        "split": str(specification["split"]),
        **identity,
        "targetId": target_id,
        "grammarId": grammar.grammar_id,
        "environmentId": environment.environment_id,
        "challengeId": challenge_id,
        "absoluteEndpointFamily": absolute_family,
        "armId": arm_id,
        "replicate": int(specification["replicate"]),
        "policyId": definition.policy_id,
        "channelMode": definition.channel_mode,
        "nativeActuationEnabled": bool(
            arm_catalog(catalog)[arm_id]["nativeActuation"] == "enabled"
        ),
        "backend": "canonical_s07_cpu_oracle_fallback",
        "eventBudgetTransitions": int(specification["eventBudget"]),
        "allocatedActorSlotsPerTransition": int(
            catalog["simulation"]["allocatedActorSlotsPerTransition"]
        ),
        "runStatus": "completed",
        "stopReason": "fixed_event_budget",
        "failed": False,
        "errorType": None,
        "errorMessage": None,
        "initialConjunctiveCompletion": initial_conjunctive,
        "formedSource": formed_source,
        "targetFeasible": feasible,
        "formationEligible": challenge_id == "partially_correct_formation",
        "repairEligible": repair_eligible,
        "absoluteTerminalSuccess": bool(metrics["terminalConjunctiveCompletion"]),
        "censored": bool(nonexact and not metrics["conjunctiveCompletionByBudget"]),
        "initialStateSha256": movement_state_sha256(initial_state),
        "finalStateSha256": result["finalState"]["stateSha256"],
        "episodeSha256": result["episodeSha256"],
        "episodeCanonicalBytesSha256": hashlib.sha256(episode_bytes).hexdigest(),
        "initialGridRowsJson": json.dumps(
            ["".join(item) for item in initial_grid], separators=(",", ":")
        ),
        **metrics,
        **{key: int(value) for key, value in external.items()},
        "lesionMaskSha256": mask.get("lesionMaskSha256"),
        "interventionSummarySha256": intervention["interventionSummarySha256"],
        "acceptedMovements": int(movement["acceptedMovements"]),
        "submittedProposals": int(movement["submittedProposals"]),
        "conflictLosses": int(movement["conflictLosses"]),
        "invalidProposals": int(movement["invalidProposals"]),
        "totalGraphDisplacement": int(movement["totalGraphDisplacement"]),
        "observationCommunicatedBitsUpperBound": int(
            observation["communicatedBitsUpperBound"]
        ),
        "policyLogicalSourceReads": policy_reads,
        "policyUtilityEvaluations": int(observation["utilityEvaluations"]),
        "policyComparisonOperations": int(observation["comparisonOperations"]),
        "configurationBits": int(channel["configurationBits"]),
        "controllerInputBits": int(channel["controllerInputBits"]),
        "policyDeliveryBits": int(channel["policyDeliveryBits"]),
        "addressBits": int(channel["addressBits"]),
        "channelTotalInformationBits": int(channel["totalInformationBits"]),
        "totalInformationBitsIncludingObservation": int(
            observation["communicatedBitsUpperBound"]
        )
        + int(channel["totalInformationBits"]),
        "sourceScalarReads": int(channel["sourceScalarReads"]),
        "controllerStateReads": int(channel["controllerStateReads"]),
        "controllerComputeUnits": int(channel["controllerComputeUnits"]),
        "actuationAttempts": int(channel["actuationAttempts"]),
        "actuationSuccesses": int(channel["actuationSuccesses"]),
        "overrideActionUnits": int(channel["overrideActionUnits"]),
        "directMovementGraphDisplacement": int(channel["movementGraphDisplacement"]),
        "suppressedNativeActions": int(channel["suppressedNativeActions"]),
        "channelOpportunityCostUnits": int(channel["opportunityCostUnits"]),
        "shamSuppressedControllerRecommendations": sum(
            int(
                event.get("actionRelayEnabled") is False
                and event.get("controllerRecommendation", {}).get("action")
                == "select_candidate"
            )
            for event in result["channelEvents"]
        ),
        "allocatedNativeActorSlots": allocated_slots,
        "usedNativeActorSlots": used_native_slots,
        "directRecipientQuerySlots": direct_query_slots,
        "foregoneNativeActorSlots": foregone_slots,
        "externalInterventionGraphDisplacement": int(
            external["externalGraphDisplacement"]
        ),
        "totalSourceWorkUnits": source_work,
        "totalComputationUnits": computation,
        "totalOpportunityCostUnits": opportunity,
        "budgetEnvelopeSuccess": bool(budget_audit["success"]),
        "budgetAuditJson": json.dumps(
            budget_audit, sort_keys=True, separators=(",", ":")
        ),
        "invariantSuccess": bool(
            initial_ids == final_ids
            and initial_tokens == final_tokens
            and initial_kinds == final_kinds
            and movement_state_sha256(final_state)
            == result["finalState"]["stateSha256"]
        ),
        "permissionAuditSuccess": all(
            value is False for value in result["permissionAudit"].values()
        ),
        "zeroTerminalClaimBoundaryAcknowledged": True,
        "wallSeconds": elapsed,
        "traceSelected": bool(specification.get("retainTrace", False)),
    }
    row["metricSummarySha256"] = sha256_value(
        "E06/S12/metric-summary/v1",
        {
            key: row[key]
            for key in sorted(row)
            if key.startswith(
                ("initialS01", "minimumS01", "terminalS01", "terminalS02")
            )
            or key
            in {
                "conjunctiveCompletionByBudget",
                "firstCompletionTransition",
                "terminalConjunctiveCompletion",
                "localGlobalDiscordance",
                "absoluteTerminalSuccess",
            }
        },
    )
    trace = result if bool(specification.get("retainTrace", False)) else None
    mask_record = {
        "schemaVersion": "e06.s12.scenario-mask.v1",
        "pairingBlockId": identity["pairingBlockId"],
        "targetId": target_id,
        "challengeId": challenge_id,
        "replicate": int(specification["replicate"]),
        "initialStateSha256": movement_state_sha256(initial_state),
        "targetFeasible": feasible,
        **mask,
        **external,
    }
    return row, trace, mask_record


def run_condition_task(task: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    masks: dict[str, Mapping[str, Any]] = {}
    trace_ids = set(task["traceRunIds"])
    for replicate in task["replicates"]:
        specification = {**task, "replicate": int(replicate)}
        specification.pop("replicates", None)
        identity = scenario_identity(
            str(specification["split"]),
            str(specification["targetId"]),
            str(specification["challengeId"]),
            int(replicate),
            str(specification["armId"]),
        )
        specification["retainTrace"] = identity["runId"] in trace_ids
        try:
            row, trace, mask = run_hybrid_once(specification)
            rows.append(row)
            masks[str(mask["pairingBlockId"])] = mask
            if trace is not None:
                traces.append(
                    {
                        "runId": row["runId"],
                        "reason": "preregistered_one_percent_condition_sample",
                        "episode": trace,
                    }
                )
        except Exception as error:
            rows.append(
                {
                    "schemaVersion": HYBRID_RUN_VERSION,
                    "phase": str(specification["phase"]),
                    "split": str(specification["split"]),
                    **identity,
                    "targetId": str(specification["targetId"]),
                    "challengeId": str(specification["challengeId"]),
                    "armId": str(specification["armId"]),
                    "replicate": int(replicate),
                    "runStatus": "failed",
                    "failed": True,
                    "censored": False,
                    "errorType": type(error).__name__,
                    "errorMessage": str(error),
                    "traceSelected": True,
                }
            )
            traces.append(
                {
                    "runId": identity["runId"],
                    "reason": "execution_failure",
                    "errorType": type(error).__name__,
                    "errorMessage": str(error),
                }
            )
    return {
        "conditionId": str(task["conditionId"]),
        "rows": rows,
        "traces": traces,
        "masks": list(masks.values()),
    }
