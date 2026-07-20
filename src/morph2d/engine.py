"""Deterministic S07 CPU episode oracle for the E06 morphology benchmark.

The oracle composes the frozen S04 movement, S05 policy-observation, and S06
channel contracts.  It deliberately keeps policy/controller payloads separate
from engine-private state, authentication, routes, and evaluation fields.  The
GPU data plane in :mod:`src.morph2d.gpu_engine` consumes only masked numeric
observations and validated proposal tensors produced by this exact control
plane.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .channels import (
    CHANNEL_LEDGER_FIELDS,
    ChannelDefinition,
    GlobalSummarySource,
    build_boundary_delivery,
    build_direct_controller_view,
    build_global_summary_delivery,
    build_sparse_instruction_delivery,
    compile_boundary_levels,
    compile_static_gradient,
    decide_direct_controller,
    execute_direct_intervention,
    load_channel_catalog,
    realize_noisy_gradient,
)
from .environments import Environment, load_environment_catalog
from .grammar import RelationalGrammar, load_grammar_catalog
from .movements import (
    LEDGER_FIELDS,
    MovementProposal,
    MovementState,
    initial_movement_state,
    make_proposal,
    movement_state_sha256,
    parse_movement_state,
    parse_proposal,
    resolve_batch,
    validate_state_against_environment,
)
from .policies import (
    OBSERVATION_LEDGER_FIELDS,
    ObservationBuild,
    PolicyDefinition,
    PolicyDecision,
    PolicyMemory,
    RelationProfile,
    build_policy_observation,
    compile_relation_profile,
    decide_policy,
    load_policy_catalog,
    materialize_decision,
    update_policy_memory,
)


ENGINE_CATALOG_VERSION = "e06.s07.engine-catalog.v1"
EPISODE_RESULT_VERSION = "e06.s07.cpu-episode-result.v1"
EPISODE_SUMMARY_VERSION = "e06.s07.episode-transition-summary.v1"
EPOCH_LENGTH_TRANSITIONS = 16


class EpisodeValidationError(ValueError):
    """Raised when an S07 episode configuration or result is invalid."""


@dataclass(frozen=True)
class EpisodeDefinition:
    scenario_id: str
    environment_id: str
    policy_id: str
    relation_grammar_id: str | None
    channel_mode: str
    transitions: int
    actor_batch_size: int
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class EngineContext:
    metadata: Mapping[str, Any]
    environments: Mapping[str, Environment]
    policies: Mapping[str, PolicyDefinition]
    grammars: Mapping[str, RelationalGrammar]
    channels: Mapping[str, ChannelDefinition]
    episodes: tuple[EpisodeDefinition, ...]


PolicyBatchAudit = Callable[
    [
        int,
        PolicyDefinition,
        Sequence[ObservationBuild],
        Sequence[Mapping[str, Any]],
        Sequence[PolicyDecision],
    ],
    None,
]
TransitionAudit = Callable[
    [
        int,
        str,
        Environment,
        MovementState,
        Sequence[MovementProposal],
        str,
        Mapping[str, Any],
    ],
    None,
]
StateAudit = Callable[[int, MovementState, Mapping[str, Any]], None]
ProposalGate = Callable[
    [int, Environment, MovementState, Sequence[MovementProposal]],
    Sequence[MovementProposal],
]

ActorPolicyAssignments = Mapping[str, str]
ActorRelationProfiles = Mapping[str, RelationProfile | None]
ActorAssignmentSwitches = Mapping[
    int, tuple[ActorPolicyAssignments, ActorRelationProfiles]
]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_payload(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + _canonical_json_bytes(value)
    ).hexdigest()


def _empty_ledger(fields: Sequence[str]) -> dict[str, int]:
    return {field: 0 for field in fields}


def _add_ledger(
    aggregate: dict[str, int],
    delta: Mapping[str, int],
    fields: Sequence[str],
    *,
    include_configuration: bool = True,
) -> None:
    if set(delta) != set(fields):
        raise EpisodeValidationError("ledger schema mismatch")
    for field in fields:
        if field == "configurationBits" and not include_configuration:
            continue
        value = int(delta[field])
        if value < 0:
            raise EpisodeValidationError("ledger values must be nonnegative")
        aggregate[field] += value
    if fields == CHANNEL_LEDGER_FIELDS:
        aggregate["totalInformationBits"] = (
            aggregate["configurationBits"]
            + aggregate["controllerInputBits"]
            + aggregate["policyDeliveryBits"]
            + aggregate["addressBits"]
        )


def _state_blind_actor_schedule(
    scenario_id: str,
    transition_index: int,
    actor_ids: Sequence[str],
    batch_size: int,
) -> tuple[str, ...]:
    """Rank immutable identities by a counter address, never by state."""

    if transition_index < 0 or batch_size <= 0:
        raise EpisodeValidationError("scheduler inputs must be positive")
    ranked = sorted(
        actor_ids,
        key=lambda actor_id: hashlib.sha256(
            b"E06/S07/actor-schedule/v1\x00"
            + scenario_id.encode("utf-8")
            + b"\x00"
            + transition_index.to_bytes(8, "big")
            + b"\x00"
            + actor_id.encode("utf-8")
        ).digest(),
    )
    return tuple(ranked[: min(batch_size, len(ranked))])


def _episode_body(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "episodeSha256"}


def canonical_episode_result_bytes(result: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(dict(result))


def _parse_episode(
    raw: Mapping[str, Any], defaults: Mapping[str, Any]
) -> EpisodeDefinition:
    required = {
        "scenarioId",
        "environmentId",
        "policyId",
        "relationGrammarId",
        "channelMode",
    }
    if not required <= set(raw):
        raise EpisodeValidationError("episode declaration missing required fields")
    known = required | {
        "initialSwap",
        "gradientAxis",
        "gradientDirection",
        "boundaryDirection",
        "boundaryTokens",
        "memoryInitialBestLocalUtility",
        "instruction",
        "regionAlias",
        "regionSize",
        "regionCount",
        "summaryRecipients",
    }
    if set(raw) - known:
        raise EpisodeValidationError("episode declaration has unknown fields")
    return EpisodeDefinition(
        scenario_id=str(raw["scenarioId"]),
        environment_id=str(raw["environmentId"]),
        policy_id=str(raw["policyId"]),
        relation_grammar_id=(
            None if raw["relationGrammarId"] is None else str(raw["relationGrammarId"])
        ),
        channel_mode=str(raw["channelMode"]),
        transitions=int(defaults["transitionBudget"]),
        actor_batch_size=int(defaults["actorBatchSize"]),
        parameters={key: raw[key] for key in raw if key not in required},
    )


def load_engine_context(
    engine_catalog: str | Path,
    *,
    environment_catalog: str | Path,
    policy_catalog: str | Path,
    grammar_catalog: str | Path,
    channel_catalog: str | Path,
) -> EngineContext:
    raw = yaml.safe_load(Path(engine_catalog).read_text(encoding="utf-8"))
    required = {
        "schemaVersion",
        "researchStepId",
        "engineVersion",
        "scope",
        "episodeContract",
        "gpuTensorContract",
        "benchmarkContract",
        "scenarios",
    }
    if set(raw) != required or raw["schemaVersion"] != ENGINE_CATALOG_VERSION:
        raise EpisodeValidationError("engine catalog schema mismatch")
    if (
        int(raw["episodeContract"]["epochLengthTransitions"])
        != EPOCH_LENGTH_TRANSITIONS
    ):
        raise EpisodeValidationError("S06 epoch length changed")
    _, environments = load_environment_catalog(environment_catalog)
    _, policies = load_policy_catalog(policy_catalog)
    _, grammars = load_grammar_catalog(grammar_catalog)
    _, channels = load_channel_catalog(channel_catalog)
    context = EngineContext(
        metadata={key: raw[key] for key in required - {"scenarios"}},
        environments={item.environment_id: item for item in environments},
        policies={item.policy_id: item for item in policies},
        grammars={item.grammar_id: item for item in grammars},
        channels={item.channel_id: item for item in channels},
        episodes=tuple(
            _parse_episode(item, raw["episodeContract"]) for item in raw["scenarios"]
        ),
    )
    validate_engine_context(context)
    return context


def validate_engine_context(context: EngineContext) -> None:
    if len(context.episodes) != 9:
        raise EpisodeValidationError("engine catalog must contain nine S07 episodes")
    modes = {item.channel_mode for item in context.episodes}
    if modes != {
        "none",
        "static_gradient",
        "boundary_signal",
        "sparse_instruction",
        "global_summary",
        "direct_intervention",
    }:
        raise EpisodeValidationError("channel-mode coverage mismatch")
    identifiers = [item.scenario_id for item in context.episodes]
    if len(identifiers) != len(set(identifiers)):
        raise EpisodeValidationError("duplicate episode scenario ID")
    for episode in context.episodes:
        if episode.environment_id not in context.environments:
            raise EpisodeValidationError("unknown episode environment")
        if episode.policy_id not in context.policies:
            raise EpisodeValidationError("unknown episode policy")
        policy = context.policies[episode.policy_id]
        if policy.relation_profile_required != (
            episode.relation_grammar_id is not None
        ):
            raise EpisodeValidationError("episode relation-profile mismatch")
        if episode.relation_grammar_id not in {None, *context.grammars}:
            raise EpisodeValidationError("unknown episode grammar")
        if episode.transitions <= 0 or episode.actor_batch_size <= 0:
            raise EpisodeValidationError("episode budget invalid")


def _channel_id(mode: str) -> str:
    return {
        "static_gradient": "static_gradient_v1",
        "boundary_signal": "natural_boundary_signal_v1",
        "sparse_instruction": "sparse_instruction_v1",
        "global_summary": "lagged_global_summary_v1",
        "direct_intervention": "sparse_direct_intervention_v1",
    }[mode]


def _cell_actor_ids(state: MovementState) -> tuple[str, ...]:
    return tuple(
        sorted(
            occupant.occupant_id
            for _, occupant in state.occupancy
            if occupant.kind == "cell"
        )
    )


def _initial_state(
    environment: Environment, definition: EpisodeDefinition
) -> tuple[MovementState, dict[str, Any] | None]:
    state = initial_movement_state(environment)
    initial_swap = definition.parameters.get("initialSwap")
    if initial_swap is None:
        return state, None
    proposal = make_proposal(state, "adjacent_swap", tuple(map(str, initial_swap)))
    damage = resolve_batch(
        environment,
        state,
        (proposal,),
        batch_nonce=f"{definition.scenario_id}:initial-condition",
    )
    if damage["acceptedProposalIds"] != [proposal.proposal_id]:
        raise EpisodeValidationError("initial-condition swap was not accepted")
    return parse_movement_state(damage["postState"]), damage


def _trace_selected(transition_index: int, transition_budget: int) -> bool:
    return (
        transition_index == 0
        or (transition_index + 1) % EPOCH_LENGTH_TRANSITIONS == 0
        or transition_index == transition_budget - 1
    )


def _configuration_bits_for_instruction(scheduled_count: int, region_count: int) -> int:
    address_bits = max(1, math.ceil(math.log2(region_count)))
    return 3 * scheduled_count + address_bits * region_count


def _summary_source(
    epoch_index: int,
    dissatisfaction_bits: Sequence[int],
    submitted: int,
    conflict_losses: int,
) -> GlobalSummarySource:
    bits = tuple(int(value) for value in dissatisfaction_bits)
    if not bits:
        bits = (0,)
    return GlobalSummarySource(
        source_epoch=epoch_index,
        local_dissatisfaction_bits=bits,
        submitted_proposals=max(1, int(submitted)),
        conflict_losses=min(max(0, int(conflict_losses)), max(1, int(submitted))),
        active_count=len(bits),
    )


def run_cpu_episode(
    context: EngineContext,
    definition: EpisodeDefinition,
    *,
    include_selected_traces: bool = True,
    initial_state_override: MovementState | None = None,
    policy_batch_audit: PolicyBatchAudit | None = None,
    transition_audit: TransitionAudit | None = None,
    state_audit: StateAudit | None = None,
    proposal_gate: ProposalGate | None = None,
    actor_policy_assignments: ActorPolicyAssignments | None = None,
    actor_relation_profiles: ActorRelationProfiles | None = None,
    actor_assignment_switches: ActorAssignmentSwitches | None = None,
) -> dict[str, Any]:
    """Execute one exact fixed-budget reference episode.

    Optional S08 audit callbacks receive immutable CPU records and cannot
    replace a decision, proposal, transition, ledger, or state. They are a
    differential-observation boundary, not an alternative engine authority.
    The optional proposal gate is an engine-owned intervention boundary for
    later perturbation studies. It may suppress already-authenticated native
    proposals, but cannot create or edit one; its separate intervention cost is
    intentionally owned by its caller rather than the frozen S04 ledger.

    S11 may additionally supply complete identity-keyed policy and actor-local
    relation-profile assignments, with complete replacement maps at declared
    transition indices.  This does not add a policy observation: assignment
    selection remains engine-private, analysis labels are absent, and normal
    S05 observation construction plus S04 proposal authentication remain the
    only decision/actuation path.  Heterogeneous assignments are intentionally
    restricted to the no-channel, memory-free S11 scope so that no S05/S06
    state-transfer semantics are invented.
    """

    environment = context.environments[definition.environment_id]
    policy = context.policies[definition.policy_id]
    relation_profile = (
        None
        if definition.relation_grammar_id is None
        else compile_relation_profile(context.grammars[definition.relation_grammar_id])
    )
    if initial_state_override is None:
        state, initial_transform = _initial_state(environment, definition)
    else:
        if definition.parameters.get("initialSwap") is not None:
            raise EpisodeValidationError(
                "initial-state override cannot be combined with initialSwap"
            )
        validate_state_against_environment(environment, initial_state_override)
        state = initial_state_override
        initial_transform = None
    if transition_audit is not None and initial_transform is not None:
        transition_audit(
            -1,
            "initial_condition",
            environment,
            parse_movement_state(initial_transform["preState"]),
            tuple(parse_proposal(item) for item in initial_transform["proposals"]),
            str(initial_transform["batchNonce"]),
            initial_transform,
        )
    initial_state_sha256 = movement_state_sha256(state)
    if state_audit is not None:
        state_audit(-1, state, {"transitionKind": "initial_state"})
    actor_ids = _cell_actor_ids(state)
    heterogeneous = any(
        item is not None
        for item in (
            actor_policy_assignments,
            actor_relation_profiles,
            actor_assignment_switches,
        )
    )
    if heterogeneous and definition.channel_mode != "none":
        raise EpisodeValidationError(
            "heterogeneous assignments are restricted to no-channel episodes"
        )

    def validated_actor_contracts(
        policy_ids: ActorPolicyAssignments,
        profiles: ActorRelationProfiles,
    ) -> tuple[dict[str, PolicyDefinition], dict[str, RelationProfile | None]]:
        expected = set(actor_ids)
        if set(policy_ids) != expected or set(profiles) != expected:
            raise EpisodeValidationError(
                "heterogeneous assignments must cover every cell identity exactly"
            )
        policies_by_actor: dict[str, PolicyDefinition] = {}
        profiles_by_actor: dict[str, RelationProfile | None] = {}
        for actor_id in actor_ids:
            policy_id = str(policy_ids[actor_id])
            if policy_id not in context.policies:
                raise EpisodeValidationError("unknown heterogeneous policy")
            actor_policy = context.policies[policy_id]
            actor_profile = profiles[actor_id]
            if actor_policy.strategy == "memory_based_recovery":
                raise EpisodeValidationError(
                    "heterogeneous memory transfer is deferred, not inferred"
                )
            if actor_policy.relation_profile_required != (actor_profile is not None):
                raise EpisodeValidationError(
                    "heterogeneous policy/relation-profile mismatch"
                )
            policies_by_actor[actor_id] = actor_policy
            profiles_by_actor[actor_id] = actor_profile
        return policies_by_actor, profiles_by_actor

    if heterogeneous:
        if actor_policy_assignments is None or actor_relation_profiles is None:
            raise EpisodeValidationError(
                "heterogeneous policy and relation assignments are both required"
            )
        policies_by_actor, profiles_by_actor = validated_actor_contracts(
            actor_policy_assignments, actor_relation_profiles
        )
        switches = dict(actor_assignment_switches or {})
        if any(index < 0 or index >= definition.transitions for index in switches):
            raise EpisodeValidationError("heterogeneous switch index outside episode")
        validated_switches = {
            int(index): validated_actor_contracts(*replacement)
            for index, replacement in switches.items()
        }
    else:
        policies_by_actor = {actor_id: policy for actor_id in actor_ids}
        profiles_by_actor = {actor_id: relation_profile for actor_id in actor_ids}
        validated_switches = {}
    memory = {
        actor_id: PolicyMemory(
            best_local_utility=int(
                definition.parameters.get("memoryInitialBestLocalUtility", 0)
            ),
            frustration=0,
        )
        for actor_id in actor_ids
    }
    lagged_conflicts = {site.site_id: 0 for site in environment.occupiable_sites}
    movement_ledger = _empty_ledger(LEDGER_FIELDS)
    observation_ledger = _empty_ledger(OBSERVATION_LEDGER_FIELDS)
    channel_ledger = _empty_ledger(CHANNEL_LEDGER_FIELDS)
    transition_summaries: list[dict[str, Any]] = []
    sampled_traces: list[dict[str, Any]] = []
    channel_events: list[dict[str, Any]] = []
    epoch_bits: list[int] = []
    epoch_submitted = 0
    epoch_conflict_losses = 0
    previous_epoch: tuple[list[int], int, int] | None = None
    latest_summary = {
        "dissatisfactionBin": 0,
        "laggedConflictBin": 0,
        "summaryAgeEpochs": 1,
    }

    gradient_levels: Mapping[str, int] | None = None
    boundary_levels: Mapping[str, int] | None = None
    mode = definition.channel_mode
    scheduled_epochs = math.ceil(definition.transitions / EPOCH_LENGTH_TRANSITIONS)
    if mode == "static_gradient":
        channel = context.channels[_channel_id(mode)]
        field, config = compile_static_gradient(
            environment,
            channel,
            axis=str(definition.parameters["gradientAxis"]),
            direction=str(definition.parameters["gradientDirection"]),
        )
        gradient_levels, noise_draws = realize_noisy_gradient(
            channel, field, scenario_key=definition.scenario_id
        )
        _add_ledger(channel_ledger, config, CHANNEL_LEDGER_FIELDS)
        channel_ledger["noiseDraws"] += noise_draws
    elif mode == "boundary_signal":
        channel = context.channels[_channel_id(mode)]
        boundary_levels, config = compile_boundary_levels(environment, channel)
        _add_ledger(channel_ledger, config, CHANNEL_LEDGER_FIELDS)
    elif mode == "sparse_instruction":
        channel_ledger["configurationBits"] = _configuration_bits_for_instruction(
            scheduled_epochs, int(definition.parameters["regionCount"])
        )
    elif mode == "global_summary":
        channel_ledger["configurationBits"] = 8
    elif mode == "direct_intervention":
        # The direct controller consumes a lagged S06 summary.  Charge both
        # declarations in full once rather than hiding the auxiliary monitor.
        channel_ledger["configurationBits"] = 64 + 8
    channel_ledger["totalInformationBits"] = channel_ledger["configurationBits"]

    for transition_index in range(definition.transitions):
        if transition_index in validated_switches:
            policies_by_actor, profiles_by_actor = validated_switches[transition_index]
        epoch_index = transition_index // EPOCH_LENGTH_TRANSITIONS
        epoch_start = transition_index % EPOCH_LENGTH_TRANSITIONS == 0
        if epoch_start and transition_index > 0:
            previous_epoch = (epoch_bits, epoch_submitted, epoch_conflict_losses)
            epoch_bits = []
            epoch_submitted = 0
            epoch_conflict_losses = 0

        transition_channel_events: list[dict[str, Any]] = []
        if mode == "sparse_instruction" and epoch_start:
            delivery = build_sparse_instruction_delivery(
                context.channels[_channel_id(mode)],
                instruction=str(definition.parameters["instruction"]),
                region_alias=str(definition.parameters["regionAlias"]),
                region_size=int(definition.parameters["regionSize"]),
                region_count=int(definition.parameters["regionCount"]),
                scenario_key=definition.scenario_id,
                epoch_index=epoch_index,
                scheduled_instruction_count=scheduled_epochs,
            )
            _add_ledger(
                channel_ledger,
                delivery.ledger,
                CHANNEL_LEDGER_FIELDS,
                include_configuration=False,
            )
            transition_channel_events.append(
                {
                    "channelId": delivery.channel_id,
                    "epochIndex": delivery.epoch_index,
                    "payload": delivery.payload,
                    "ledger": delivery.ledger,
                    "deliverySha256": delivery.delivery_sha256,
                }
            )

        if (
            mode in {"global_summary", "direct_intervention"}
            and epoch_start
            and previous_epoch
        ):
            prior_bits, prior_submitted, prior_losses = previous_epoch
            summary = build_global_summary_delivery(
                context.channels["lagged_global_summary_v1"],
                _summary_source(
                    epoch_index - 1, prior_bits, prior_submitted, prior_losses
                ),
                scenario_key=definition.scenario_id,
                epoch_index=epoch_index,
                recipient_count=(
                    int(definition.parameters.get("summaryRecipients", 1))
                    if mode == "global_summary"
                    else 1
                ),
            )
            latest_summary = dict(summary.payload)
            _add_ledger(
                channel_ledger,
                summary.ledger,
                CHANNEL_LEDGER_FIELDS,
                include_configuration=False,
            )
            transition_channel_events.append(
                {
                    "channelId": summary.channel_id,
                    "epochIndex": summary.epoch_index,
                    "payload": summary.payload,
                    "ledger": summary.ledger,
                    "deliverySha256": summary.delivery_sha256,
                }
            )

        # Direct intervention owns this transition and uses exactly one
        # state-blind recipient/query/action.  It never co-schedules a native
        # proposal or retries.
        if mode == "direct_intervention" and epoch_start and previous_epoch:
            recipient = actor_ids[epoch_index % len(actor_ids)]
            build = build_policy_observation(
                environment,
                state,
                recipient,
                policy,
                relation_profile=relation_profile,
                lagged_conflicts=lagged_conflicts,
                decision_key=definition.scenario_id,
                activation_index=transition_index * definition.actor_batch_size,
            )
            _add_ledger(
                observation_ledger,
                build.observation.budget,
                OBSERVATION_LEDGER_FIELDS,
            )
            view = build_direct_controller_view(
                context.channels[_channel_id(mode)],
                build,
                epoch_index=epoch_index,
                population_size=len(actor_ids),
                lagged_summary=latest_summary,
                remaining_information_budget=352,
                remaining_action_budget=1,
            )
            decision = decide_direct_controller(view.payload)
            direct_event = execute_direct_intervention(
                context.channels[_channel_id(mode)],
                environment,
                state,
                build,
                view,
                decision,
                scenario_key=definition.scenario_id,
                batch_nonce=f"{definition.scenario_id}:direct:{transition_index}",
            )
            batch = direct_event["movementBatchResult"]
            if transition_audit is not None:
                transition_audit(
                    transition_index,
                    "direct_intervention",
                    environment,
                    state,
                    tuple(parse_proposal(item) for item in batch["proposals"]),
                    str(batch["batchNonce"]),
                    batch,
                )
            state = parse_movement_state(batch["postState"])
            _add_ledger(movement_ledger, batch["costLedger"], LEDGER_FIELDS)
            _add_ledger(
                channel_ledger,
                direct_event["ledger"],
                CHANNEL_LEDGER_FIELDS,
                include_configuration=False,
            )
            epoch_bits.append(int(decision.action != "noop"))
            epoch_submitted += batch["costLedger"]["submittedProposals"]
            epoch_conflict_losses += batch["costLedger"]["conflictLosses"]
            summary_record = {
                "schemaVersion": EPISODE_SUMMARY_VERSION,
                "transitionIndex": transition_index,
                "epochIndex": epoch_index,
                "scheduledActorCount": 1,
                "proposalCount": batch["costLedger"]["submittedProposals"],
                "acceptedCount": batch["costLedger"]["acceptedMovements"],
                "conflictLosses": batch["costLedger"]["conflictLosses"],
                "invalidProposals": batch["costLedger"]["invalidProposals"],
                "preStateSha256": batch["preState"]["stateSha256"],
                "postStateSha256": batch["postState"]["stateSha256"],
                "transitionSha256": batch["transitionSha256"],
                "channelEventCount": len(transition_channel_events) + 1,
                "directIntervention": True,
            }
            transition_summaries.append(summary_record)
            if state_audit is not None:
                state_audit(transition_index, state, summary_record)
            transition_channel_events.append(
                {
                    "channelId": direct_event["channelId"],
                    "epochIndex": direct_event["epochIndex"],
                    "outcome": direct_event["outcome"],
                    "ledger": direct_event["ledger"],
                    "eventSha256": direct_event["eventSha256"],
                }
            )
            channel_events.extend(transition_channel_events)
            if include_selected_traces and _trace_selected(
                transition_index, definition.transitions
            ):
                sampled_traces.append(
                    {
                        "transitionIndex": transition_index,
                        "reason": "preregistered_transition_sample",
                        "movementBatchResult": batch,
                        "channelEvents": transition_channel_events,
                    }
                )
            continue

        scheduled_actors = _state_blind_actor_schedule(
            definition.scenario_id,
            transition_index,
            actor_ids,
            definition.actor_batch_size,
        )
        builds: list[ObservationBuild] = []
        decision_payloads: list[Mapping[str, Any]] = []
        decisions: list[PolicyDecision] = []
        proposals: list[MovementProposal] = []
        proposal_to_build: dict[
            str, tuple[str, ObservationBuild, Any, PolicyDefinition]
        ] = {}
        policies_for_builds: list[PolicyDefinition] = []
        for slot, actor_id in enumerate(scheduled_actors):
            actor_policy = policies_by_actor[actor_id]
            actor_profile = profiles_by_actor[actor_id]
            build = build_policy_observation(
                environment,
                state,
                actor_id,
                actor_policy,
                relation_profile=actor_profile,
                boundary_direction=definition.parameters.get("boundaryDirection"),
                boundary_tokens=definition.parameters.get("boundaryTokens"),
                gradient_levels=gradient_levels,
                gradient_direction=definition.parameters.get("gradientDirection"),
                lagged_conflicts=lagged_conflicts,
                memory=memory[actor_id]
                if actor_policy.strategy == "memory_based_recovery"
                else None,
                decision_key=definition.scenario_id,
                activation_index=transition_index * definition.actor_batch_size + slot,
            )
            payload: Mapping[str, Any] = build.observation.payload
            if mode == "boundary_signal":
                if boundary_levels is None:
                    raise EpisodeValidationError("missing compiled boundary field")
                source_site = next(
                    site_id
                    for site_id, occupant in state.occupancy
                    if occupant.occupant_id == actor_id
                )
                candidate_levels = {
                    key: boundary_levels[item.target_site]
                    for key, item in build.candidate_map.items()
                }
                delivery = build_boundary_delivery(
                    context.channels[_channel_id(mode)],
                    current_level=boundary_levels[source_site],
                    candidate_levels=candidate_levels,
                    direction=str(definition.parameters["boundaryDirection"]),
                    directive_applies=(
                        state.occupant_map[source_site].token
                        in set(map(str, definition.parameters["boundaryTokens"]))
                    ),
                    configuration_bits=channel_ledger["configurationBits"],
                    scenario_key=definition.scenario_id,
                    epoch_index=epoch_index,
                )
                payload = delivery.payload
                _add_ledger(
                    channel_ledger,
                    delivery.ledger,
                    CHANNEL_LEDGER_FIELDS,
                    include_configuration=False,
                )
            elif mode == "static_gradient":
                channel_ledger["sourceScalarReads"] += build.observation.budget[
                    "gradientSignalReads"
                ]
                channel_ledger["recipientDeliveries"] += 1
                channel_ledger["policyDeliveryBits"] += build.observation.budget[
                    "communicatedBitsUpperBound"
                ]
                channel_ledger["totalInformationBits"] = (
                    channel_ledger["configurationBits"]
                    + channel_ledger["controllerInputBits"]
                    + channel_ledger["policyDeliveryBits"]
                    + channel_ledger["addressBits"]
                )
            decision = decide_policy(actor_policy, payload)
            proposal = materialize_decision(build, decision)
            builds.append(build)
            decision_payloads.append(payload)
            decisions.append(decision)
            policies_for_builds.append(actor_policy)
            _add_ledger(
                observation_ledger,
                build.observation.budget,
                OBSERVATION_LEDGER_FIELDS,
            )
            epoch_bits.append(int(decision.action == "proposal"))
            if proposal is not None:
                proposals.append(proposal)
                proposal_to_build[proposal.proposal_id] = (
                    actor_id,
                    build,
                    decision,
                    actor_policy,
                )

        if policy_batch_audit is not None:
            if heterogeneous:
                raise EpisodeValidationError(
                    "S08 policy-batch audit is unavailable for heterogeneous episodes"
                )
            else:
                policy_batch_audit(
                    transition_index,
                    policy,
                    tuple(builds),
                    tuple(decision_payloads),
                    tuple(decisions),
                )

        if proposal_gate is not None:
            proposed_by_id = {item.proposal_id: item for item in proposals}
            gated = tuple(
                proposal_gate(
                    transition_index,
                    environment,
                    state,
                    tuple(proposals),
                )
            )
            gated_ids = [item.proposal_id for item in gated]
            if len(gated_ids) != len(set(gated_ids)):
                raise EpisodeValidationError("proposal gate returned duplicates")
            if any(
                proposal_id not in proposed_by_id
                or proposed_by_id[proposal_id] != proposal
                for proposal_id, proposal in zip(gated_ids, gated, strict=True)
            ):
                raise EpisodeValidationError(
                    "proposal gate may only retain authenticated proposals"
                )
            proposals = list(gated)

        batch_nonce = f"{definition.scenario_id}:transition:{transition_index}"
        batch = resolve_batch(
            environment,
            state,
            tuple(proposals),
            batch_nonce=batch_nonce,
        )
        if transition_audit is not None:
            transition_audit(
                transition_index,
                "native_policy_batch",
                environment,
                state,
                tuple(proposals),
                batch_nonce,
                batch,
            )
        state = parse_movement_state(batch["postState"])
        _add_ledger(movement_ledger, batch["costLedger"], LEDGER_FIELDS)
        epoch_submitted += batch["costLedger"]["submittedProposals"]
        epoch_conflict_losses += batch["costLedger"]["conflictLosses"]
        accepted = set(batch["acceptedProposalIds"])
        outcome_by_actor = {actor_id: "noop" for actor_id in scheduled_actors}
        for proposal_id, (
            actor_id,
            build,
            decision,
            actor_policy,
        ) in proposal_to_build.items():
            outcome_by_actor[actor_id] = (
                "accepted" if proposal_id in accepted else "rejected"
            )
            if actor_policy.strategy == "memory_based_recovery":
                memory[actor_id] = update_policy_memory(
                    memory[actor_id],
                    build.observation,
                    decision,
                    outcome_by_actor[actor_id],
                )
        for actor_id, build, decision, actor_policy in zip(
            scheduled_actors,
            builds,
            decisions,
            policies_for_builds,
            strict=True,
        ):
            if (
                actor_policy.strategy == "memory_based_recovery"
                and decision.action == "noop"
            ):
                memory[actor_id] = update_policy_memory(
                    memory[actor_id], build.observation, decision, "noop"
                )

        next_lagged = {site.site_id: 0 for site in environment.occupiable_sites}
        proposal_by_id = {item.proposal_id: item for item in proposals}
        for record in batch["decisions"]:
            if record["outcome"] != "conflict_lost":
                continue
            proposal = proposal_by_id[record["proposalId"]]
            if proposal.target_site is not None:
                next_lagged[proposal.target_site] = min(
                    3, next_lagged[proposal.target_site] + 1
                )
        lagged_conflicts = next_lagged
        summary_record = {
            "schemaVersion": EPISODE_SUMMARY_VERSION,
            "transitionIndex": transition_index,
            "epochIndex": epoch_index,
            "scheduledActorCount": len(scheduled_actors),
            "proposalCount": len(proposals),
            "acceptedCount": batch["costLedger"]["acceptedMovements"],
            "conflictLosses": batch["costLedger"]["conflictLosses"],
            "invalidProposals": batch["costLedger"]["invalidProposals"],
            "preStateSha256": batch["preState"]["stateSha256"],
            "postStateSha256": batch["postState"]["stateSha256"],
            "transitionSha256": batch["transitionSha256"],
            "channelEventCount": len(transition_channel_events),
            "directIntervention": False,
        }
        transition_summaries.append(summary_record)
        if state_audit is not None:
            state_audit(transition_index, state, summary_record)
        channel_events.extend(transition_channel_events)
        if include_selected_traces and _trace_selected(
            transition_index, definition.transitions
        ):
            sampled_traces.append(
                {
                    "transitionIndex": transition_index,
                    "reason": "preregistered_transition_sample",
                    "movementBatchResult": batch,
                    "channelEvents": transition_channel_events,
                }
            )

    result: dict[str, Any] = {
        "schemaVersion": EPISODE_RESULT_VERSION,
        "scenarioId": definition.scenario_id,
        "environmentId": definition.environment_id,
        "policyId": definition.policy_id,
        "channelMode": definition.channel_mode,
        "transitionBudget": definition.transitions,
        "epochLengthTransitions": EPOCH_LENGTH_TRANSITIONS,
        "actorBatchSize": definition.actor_batch_size,
        "scheduler": "state_blind_identity_hash_rank_without_replacement_per_transition",
        "initialStateSha256": initial_state_sha256,
        "initialTransformSha256": (
            None if initial_transform is None else initial_transform["transitionSha256"]
        ),
        "finalState": {
            "stateSha256": movement_state_sha256(state),
            "transitionIndex": state.transition_index,
            "occupantProjection": [
                [site_id, occupant.occupant_id, occupant.token, occupant.kind]
                for site_id, occupant in state.occupancy
            ],
        },
        "movementLedger": movement_ledger,
        "observationLedger": observation_ledger,
        "channelLedger": channel_ledger,
        "configurationAccounting": {
            "chargedOnceInFull": True,
            "amortizedOrDivided": False,
            "configurationBits": channel_ledger["configurationBits"],
        },
        "transitionSummaries": transition_summaries,
        "channelEvents": channel_events,
        "sampledTraces": sampled_traces,
        "stopReason": "transition_budget",
        "permissionAudit": {
            "s05PayloadWidened": False,
            "controllerAuthorityWidened": False,
            "globalEvaluationUsedByPolicyOrController": False,
            "analysisLabelsUsed": False,
            "futureStateUsed": False,
        },
        "evaluationSeparation": {
            "s02LocalFeedback": {
                "pricedObservationField": "localRelationDelta",
                "activeForEpisodePolicy": any(
                    item is not None for item in profiles_by_actor.values()
                ),
            },
            "s01GlobalCompletion": {
                "fields": [
                    "canonicalEquivalence",
                    "componentAudit",
                    "topologyAudit",
                ],
                "computedOnline": False,
                "policyOrControllerAccessible": False,
            },
            "conjunctiveCompletionEvaluatedOnline": False,
        },
    }
    result["episodeSha256"] = _sha256_payload(
        "E06/S07/cpu-episode/v1", _episode_body(result)
    )
    validate_episode_result(environment, result)
    return result


def validate_episode_result(
    environment: Environment, result: Mapping[str, Any]
) -> None:
    required = {
        "schemaVersion",
        "scenarioId",
        "environmentId",
        "policyId",
        "channelMode",
        "transitionBudget",
        "epochLengthTransitions",
        "actorBatchSize",
        "scheduler",
        "initialStateSha256",
        "initialTransformSha256",
        "finalState",
        "movementLedger",
        "observationLedger",
        "channelLedger",
        "configurationAccounting",
        "transitionSummaries",
        "channelEvents",
        "sampledTraces",
        "stopReason",
        "permissionAudit",
        "evaluationSeparation",
        "episodeSha256",
    }
    if set(result) != required or result["schemaVersion"] != EPISODE_RESULT_VERSION:
        raise EpisodeValidationError("episode result schema mismatch")
    if result["environmentId"] != environment.environment_id:
        raise EpisodeValidationError("episode environment mismatch")
    if len(result["transitionSummaries"]) != int(result["transitionBudget"]):
        raise EpisodeValidationError("episode transition count mismatch")
    if set(result["movementLedger"]) != set(LEDGER_FIELDS):
        raise EpisodeValidationError("episode movement ledger mismatch")
    if set(result["observationLedger"]) != set(OBSERVATION_LEDGER_FIELDS):
        raise EpisodeValidationError("episode observation ledger mismatch")
    if set(result["channelLedger"]) != set(CHANNEL_LEDGER_FIELDS):
        raise EpisodeValidationError("episode channel ledger mismatch")
    if result["channelLedger"]["totalInformationBits"] != (
        result["channelLedger"]["configurationBits"]
        + result["channelLedger"]["controllerInputBits"]
        + result["channelLedger"]["policyDeliveryBits"]
        + result["channelLedger"]["addressBits"]
    ):
        raise EpisodeValidationError("episode information ledger does not reconcile")
    if not result["configurationAccounting"]["chargedOnceInFull"]:
        raise EpisodeValidationError("configuration cost was not charged")
    if result["configurationAccounting"]["amortizedOrDivided"]:
        raise EpisodeValidationError("configuration cost was silently amortized")
    projection = result["finalState"]["occupantProjection"]
    expected_sites = sorted(site.site_id for site in environment.occupiable_sites)
    if sorted(item[0] for item in projection) != expected_sites:
        raise EpisodeValidationError("final state site coverage failed")
    occupant_ids = [item[1] for item in projection]
    if len(occupant_ids) != len(set(occupant_ids)):
        raise EpisodeValidationError("final state identity conservation failed")
    if not all(not bool(value) for value in result["permissionAudit"].values()):
        raise EpisodeValidationError("episode permission audit failed")
    evaluation = result["evaluationSeparation"]
    if (
        evaluation["s02LocalFeedback"]["pricedObservationField"] != "localRelationDelta"
        or evaluation["s01GlobalCompletion"]["fields"]
        != ["canonicalEquivalence", "componentAudit", "topologyAudit"]
        or evaluation["s01GlobalCompletion"]["computedOnline"]
        or evaluation["s01GlobalCompletion"]["policyOrControllerAccessible"]
        or evaluation["conjunctiveCompletionEvaluatedOnline"]
    ):
        raise EpisodeValidationError("S02/S01 evaluation separation failed")
    expected_hash = _sha256_payload("E06/S07/cpu-episode/v1", _episode_body(result))
    if expected_hash != result["episodeSha256"]:
        raise EpisodeValidationError("episode result hash mismatch")


def replay_cpu_episode(
    context: EngineContext,
    definition: EpisodeDefinition,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    replayed = run_cpu_episode(context, definition)
    if canonical_episode_result_bytes(replayed) != canonical_episode_result_bytes(
        expected
    ):
        raise EpisodeValidationError("CPU episode replay mismatch")
    return replayed
