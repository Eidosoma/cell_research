"""Compact transition-equivalent S10 screening executor.

The executor is specialized to the 14 frozen S08 treatment signatures used in
S10: serial uniform or random-permutation scheduling, exact sensing, no
exogenous action failure, and one policy proposal per opportunity.  It keeps
aggregate audits instead of one Python audit object per opportunity.  The
authoritative S05/S09 executor remains the validation authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping

from causal_simulator.scenario_extensions import scenario_from_record
from reference_simulator.engine import initial_state, is_complete
from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    LEDGER_FIELDS,
    Policy,
    Proposal,
    ProposalKind,
    RunState,
    Scenario,
    canonical_json_bytes,
    sha256_json,
)
from reference_simulator.policies import has_admissible_change
from reference_simulator.rng import bounded, u64
from reference_simulator.transition_primitives import cost_delta, validate_proposal

from .costing import COST_SCHEMA_SHA256, COST_SCHEMA_VERSION, S01_ADDITIVE_FIELDS
from .schedulers import PERMUTATION_STREAM


COMPACT_EXECUTOR_VERSION = "e02.s10.compact_executor.v1"
COMPACT_DIGEST_DOMAIN = hashlib.sha256(b"E02/S10/compact-transition/v1").digest()


def materialize_screening_scenario(
    scenario_record: Mapping[str, Any],
    pairing_record: Mapping[str, Any],
    settings: Mapping[str, str],
) -> Scenario:
    """Materialize one executable scenario without changing S07/S08 fields."""

    source = scenario_from_record(scenario_record)
    policy = Policy(str(pairing_record["policyProfile"]))
    direction = Direction(str(pairing_record["direction"]))
    mobility = str(settings["mobility"])
    fault_ids = (
        set(map(str, pairing_record["faultIdentityIds"]))
        if mobility != "normal"
        else set()
    )
    mode = FaultMode(mobility)
    cells = tuple(
        Cell(
            f"cell-{index:04d}",
            value,
            policy,
            direction,
            mode if f"cell-{index:04d}" in fault_ids else FaultMode.NORMAL,
        )
        for index, value in enumerate(source.values_by_identity)
    )
    variant_field = {
        "normal": "normalScenarioVariantId",
        "passive": "passiveScenarioVariantId",
        "stuck": "stuckScenarioVariantId",
    }[mobility]
    return Scenario.create(
        cells,
        initial_occupancy=tuple(
            f"cell-{index:04d}" for index in source.initial_occupancy_indices
        ),
        seed=source.scenario_seed,
        max_activations=int(pairing_record["eventBudgetOpportunities"]),
        architecture=Architecture.CELL_VIEW,
        generation_key=f"S08/{pairing_record[variant_field]}",
        # The reference Scenario type accepts only its three implementation
        # placement labels. S08's richer structural class remains in pairing
        # metadata; identities are supplied explicitly here.
        fault_placement="explicit",
        requested_fault_count=(0 if mobility == "normal" else int(pairing_record["faultCount"])),
    )


def _noop(actor_id: str, position: int, reason: str, reads: int, comparisons: int = 0) -> Proposal:
    return Proposal(
        ProposalKind.NO_OP,
        actor_id,
        position,
        reason=reason,
        observation_reads=reads,
        value_comparisons=comparisons,
    )


def compact_policy_proposal(
    scenario: Scenario,
    state: RunState,
    positions: Mapping[str, int],
    actor_id: str,
    *,
    side: str | None,
    insertion_prefix_cache: tuple[int | None, tuple[int, ...]] | None = None,
) -> Proposal:
    """Exact S02 proposal projection using only the policy's authorized records."""

    actor = scenario.cell_map[actor_id]
    position = positions[actor_id]
    if actor.fault != FaultMode.NORMAL:
        return _noop(actor_id, position, "actor_fault", 1)
    occupancy = state.occupancy
    cells = scenario.cell_map

    if actor.policy == Policy.BUBBLE:
        if side not in {"left", "right"}:
            raise ValueError("Bubble activation requires one frozen side draw")
        target_position = position + (-1 if side == "left" else 1)
        if not 0 <= target_position < len(occupancy):
            return _noop(actor_id, position, "boundary", 1)
        target_id = occupancy[target_position]
        target = cells[target_id]
        if actor.direction == Direction.ASCENDING:
            inversion = actor.value < target.value if side == "left" else actor.value > target.value
        else:
            inversion = actor.value > target.value if side == "left" else actor.value < target.value
        if not inversion:
            return _noop(actor_id, position, "ordered_or_equal", 2, 1)
        return Proposal(
            ProposalKind.SWAP,
            actor_id,
            position,
            target_pos=target_position,
            reason="strict_adjacent_inversion",
            observation_reads=2,
            value_comparisons=1,
            observed_target_id=target_id,
        )

    if actor.policy == Policy.INSERTION:
        if position == 0:
            return _noop(actor_id, position, "boundary", 1)
        if insertion_prefix_cache is not None:
            first_invalid, comparison_prefix = insertion_prefix_cache
            if first_invalid is not None and first_invalid < position:
                return _noop(
                    actor_id,
                    position,
                    "prefix_not_ordered",
                    first_invalid + 2,
                    comparison_prefix[first_invalid],
                )
            target_position = position - 1
            target_id = occupancy[target_position]
            target = cells[target_id]
            reads = position + 2
            comparisons = comparison_prefix[position - 1] + 1
            inversion = (
                actor.value < target.value
                if actor.direction == Direction.ASCENDING
                else actor.value > target.value
            )
            if not inversion:
                return _noop(actor_id, position, "ordered_or_equal", reads, comparisons)
            return Proposal(
                ProposalKind.SWAP,
                actor_id,
                position,
                target_pos=target_position,
                reason="insertion_into_ordered_prefix",
                observation_reads=reads,
                value_comparisons=comparisons,
                observed_target_id=target_id,
            )
        reads = 1
        comparisons = 0
        prior_value: int | float | None = None
        prior_normal = False
        for prefix_position in range(position):
            item = cells[occupancy[prefix_position]]
            reads += 1
            if item.fault != FaultMode.NORMAL:
                prior_normal = False
                prior_value = None
                continue
            if prior_normal:
                comparisons += 1
                ordered = (
                    prior_value <= item.value
                    if actor.direction == Direction.ASCENDING
                    else prior_value >= item.value
                )
                if not ordered:
                    return _noop(actor_id, position, "prefix_not_ordered", reads, comparisons)
            prior_normal = True
            prior_value = item.value
        target_position = position - 1
        target_id = occupancy[target_position]
        target = cells[target_id]
        reads += 1
        comparisons += 1
        inversion = (
            actor.value < target.value
            if actor.direction == Direction.ASCENDING
            else actor.value > target.value
        )
        if not inversion:
            return _noop(actor_id, position, "ordered_or_equal", reads, comparisons)
        return Proposal(
            ProposalKind.SWAP,
            actor_id,
            position,
            target_pos=target_position,
            reason="insertion_into_ordered_prefix",
            observation_reads=reads,
            value_comparisons=comparisons,
            observed_target_id=target_id,
        )

    if actor.policy == Policy.SELECTION:
        cursor = state.selection_cursors[actor_id]
        if not 0 <= cursor < len(occupancy):
            return _noop(actor_id, position, "cursor_exhausted", 1)
        if cursor == position:
            return _noop(actor_id, position, "at_cursor", 1)
        target_id = occupancy[cursor]
        target = cells[target_id]
        if target.fault == FaultMode.STUCK:
            return Proposal(
                ProposalKind.MEMORY_UPDATE,
                actor_id,
                position,
                new_cursor=cursor + (1 if actor.direction == Direction.ASCENDING else -1),
                reason="skip_stuck_target",
                observation_reads=2,
            )
        if target.value <= actor.value:
            return Proposal(
                ProposalKind.MEMORY_UPDATE,
                actor_id,
                position,
                new_cursor=cursor + (1 if actor.direction == Direction.ASCENDING else -1),
                reason="target_already_extreme",
                observation_reads=2,
                value_comparisons=1,
            )
        return Proposal(
            ProposalKind.SWAP,
            actor_id,
            position,
            target_pos=cursor,
            reason="selection_target",
            observation_reads=2,
            value_comparisons=1,
            observed_target_id=target_id,
        )
    raise AssertionError(actor.policy)


def build_insertion_prefix_cache(
    scenario: Scenario,
    state: RunState,
) -> tuple[int | None, tuple[int, ...]]:
    """Summarize the exact authorized strict-prefix walk for one state revision."""

    cells = scenario.cell_map
    occupancy = state.occupancy
    direction = scenario.cells[0].direction
    cumulative = [0] * len(occupancy)
    first_invalid: int | None = None
    for right in range(1, len(occupancy)):
        left_cell = cells[occupancy[right - 1]]
        right_cell = cells[occupancy[right]]
        compared = left_cell.fault == FaultMode.NORMAL and right_cell.fault == FaultMode.NORMAL
        cumulative[right] = cumulative[right - 1] + int(compared)
        if compared and first_invalid is None:
            ordered = (
                left_cell.value <= right_cell.value
                if direction == Direction.ASCENDING
                else left_cell.value >= right_cell.value
            )
            if not ordered:
                first_invalid = right
    return first_invalid, tuple(cumulative)


class CompactScheduler:
    __slots__ = (
        "family",
        "seed",
        "scenario_id",
        "actors",
        "cached_sweep",
        "cached_permutation",
        "cached_blocks",
        "rng_blocks",
    )

    def __init__(self, family: str, scenario: Scenario) -> None:
        if family not in {"uniform_random_activation", "random_permutation_sweep"}:
            raise ValueError("S10 compact executor accepts only frozen S08 schedulers")
        self.family = family
        self.seed = scenario.seed
        self.scenario_id = scenario.scenario_id
        self.actors = tuple(cell.cell_id for cell in scenario.cells)
        self.cached_sweep = -1
        self.cached_permutation: tuple[str, ...] = ()
        self.cached_blocks = 0
        self.rng_blocks = 0

    def actor(self, event_index: int) -> str:
        n = len(self.actors)
        if self.family == "uniform_random_activation":
            selected, consumed = bounded(
                self.seed, self.scenario_id, "actor_activation", event_index, n
            )
            self.rng_blocks += consumed
            return self.actors[selected]
        sweep = event_index // n
        offset = event_index % n
        if sweep != self.cached_sweep:
            if sweep != self.cached_sweep + 1:
                raise AssertionError("permutation sweep address discontinuity")
            result = list(self.actors)
            cursor = 0
            for index in range(n - 1, 0, -1):
                selected, consumed = bounded(
                    self.seed,
                    self.scenario_id,
                    PERMUTATION_STREAM,
                    sweep,
                    index + 1,
                    cursor,
                )
                cursor += consumed
                result[index], result[selected] = result[selected], result[index]
            self.cached_sweep = sweep
            self.cached_permutation = tuple(result)
            self.cached_blocks = cursor
            self.rng_blocks += cursor
        return self.cached_permutation[offset]


@dataclass(slots=True)
class CompactAudit:
    coordinator_eligible: int = 0
    coordinator_interventions: int = 0
    blocking_failures: int = 0
    continuation_stops: int = 0


def _base_terminal(
    scenario: Scenario,
    state: RunState,
) -> str | None:
    if is_complete(scenario, state):
        return "complete"
    if not has_admissible_change(scenario, state):
        return "quiescent"
    return None


def _blocking_disposition(scenario: Scenario, proposal: Proposal, decision: str) -> bool:
    if decision in {"rejected_actor_stuck", "rejected_target_stuck"}:
        return True
    return (
        proposal.kind == ProposalKind.NO_OP
        and proposal.reason == "actor_fault"
        and scenario.cell_map[proposal.actor_id].fault != FaultMode.NORMAL
    )


def _costs(
    scenario: Scenario,
    state: RunState,
    settings: Mapping[str, str],
    scheduler: CompactScheduler,
    audit: CompactAudit,
    wall_time_seconds: float,
) -> tuple[dict[str, int | float], dict[str, int], dict[str, float | None], dict[str, bool]]:
    native = state.ledger
    active_weak = settings["coordinatorProfile"] == "weak_frozen_budget"
    costs: dict[str, int | float] = {
        "activations": native["activations"],
        "observationRecordReads": native["observationReads"],
        "valueReads": native["observationReads"],
        "statusReads": native["observationReads"],
        "valueComparisons": native["valueComparisons"],
        "targetCalculations": native["proposals"],
        "policyCandidateConstructions": native["proposals"],
        "deferredRetryEnvelopeReuses": 0,
        "proposals": native["proposals"],
        "failedProposals": native["rejections"] + native["conflictLosses"],
        "noOps": native["noOps"],
        "rejections": native["rejections"],
        "memoryUpdates": native["memoryUpdates"],
        "acceptedSwaps": native["acceptedSwaps"],
        "displacedCells": native["displacedCells"],
        "conflictLosses": native["conflictLosses"],
        "coordinatorMessages": 2 * audit.coordinator_eligible,
        "coordinatorMessageBits": 2 * audit.coordinator_eligible,
        "coordinatorClockChecks": native["activations"] if active_weak else 0,
        "coordinatorCandidateEvaluations": audit.coordinator_eligible,
        "coordinatorEligibleDecisions": audit.coordinator_eligible,
        "coordinatorInterventions": audit.coordinator_interventions,
        "sensingValueDraws": 0,
        "sensingStatusDraws": 0,
        "sensingValueErrors": 0,
        "sensingStatusErrors": 0,
        "sensingErrorsApplied": 0,
        "sensingErrorHandlingOperations": 0,
        "actionFailureDraws": 0,
        "actionFailureExposures": 0,
        "actionFailures": 0,
        "blockingFailures": audit.blocking_failures,
        "continuationStops": audit.continuation_stops,
        "retryQueued": 0,
        "retryAttempts": 0,
        "retryExhausted": 0,
        "retryQueueCollisions": 0,
        "retryPendingAtStop": 0,
        "faultRngDraws": 0,
        "schedulerActorSelectionRngBlocks": scheduler.rng_blocks,
        "schedulerIdentitiesScored": native["activations"],
        "schedulerCandidateInspections": 0,
        "wallTimeSeconds": wall_time_seconds,
    }
    s01 = sum(int(costs[name]) for name in S01_ADDITIVE_FIELDS)
    controller = s01 + int(costs["coordinatorClockChecks"]) + int(costs["coordinatorCandidateEvaluations"])
    projections = {
        "s01UnitWeightFullCost": s01,
        "controllerExpandedSensitivity": controller,
        "mechanismExpandedSensitivity": controller,
        "zeroCostControl": 0,
    }
    activations = int(costs["activations"])
    normalizations = {
        "s01UnitWeightPerOpportunity": s01 / activations if activations else None,
        "s01UnitWeightPerCell": s01 / len(scenario.cells),
        "s01UnitWeightPerN2": s01 / (len(scenario.cells) ** 2),
        "failedProposalsPerOpportunity": int(costs["failedProposals"]) / activations if activations else None,
        "displacedCellsPerOpportunity": int(costs["displacedCells"]) / activations if activations else None,
        "coordinatorMessagesPerOpportunity": int(costs["coordinatorMessages"]) / activations if activations else None,
    }
    validations = {
        "oneProposalPerActivation": native["activations"] == native["proposals"],
        "proposalPartition": native["proposals"]
        == native["noOps"] + native["rejections"] + native["memoryUpdates"] + native["acceptedSwaps"] + native["conflictLosses"],
        "twoDisplacementsPerSwap": native["displacedCells"] == 2 * native["acceptedSwaps"],
        "readFieldAliasIdentity": costs["valueReads"] == costs["statusReads"] == costs["observationRecordReads"],
        "candidateConstructionIdentity": costs["targetCalculations"] == costs["policyCandidateConstructions"] == costs["proposals"],
        "failedProposalIdentity": costs["failedProposals"] == costs["rejections"] + costs["conflictLosses"],
        "coordinatorMessageIdentity": costs["coordinatorMessages"] == 2 * costs["coordinatorEligibleDecisions"],
        "coordinatorCandidateIdentity": costs["coordinatorCandidateEvaluations"] == costs["coordinatorEligibleDecisions"],
        "coordinatorClockIdentity": costs["coordinatorClockChecks"] == (costs["activations"] if active_weak else 0),
        "schedulerCandidateBoundary": costs["schedulerCandidateInspections"] == 0,
        "s01ProjectionIdentity": projections["s01UnitWeightFullCost"] == sum(int(costs[name]) for name in S01_ADDITIVE_FIELDS),
        "zeroCostProjectionIdentity": projections["zeroCostControl"] == 0,
        "legalPrimitivesPreserved": settings["legalPrimitives"] == "NoOp|Swap|MemoryUpdate",
        "informationPermissionPreserved": settings["informationPermission"] == "policy_native_local",
        "opportunityContractPreserved": settings["proposalCandidatesPerOpportunity"] == "1",
        "sensingExact": settings["sensing"] == "exact",
        "actionFailureNone": settings["actionFailure"] == "none",
    }
    return costs, projections, normalizations, validations


def run_compact_screening(
    scenario: Scenario,
    settings: Mapping[str, str],
    *,
    retain_step_projection: bool = False,
) -> dict[str, Any]:
    """Run one frozen S10 arm with aggregate audits and deterministic replay bytes."""

    start = perf_counter_ns()
    state = initial_state(scenario)
    positions = {cell_id: index for index, cell_id in enumerate(state.occupancy)}
    scheduler = CompactScheduler(settings["scheduler"], scenario)
    audit = CompactAudit()
    revision = 0
    cached_revision = -1
    cached_terminal: str | None = None
    insertion_cache_revision = -1
    insertion_prefix_cache: tuple[int | None, tuple[int, ...]] | None = None
    digest = COMPACT_DIGEST_DOMAIN
    step_projection: list[dict[str, Any]] = []

    while True:
        if cached_revision != revision:
            cached_terminal = _base_terminal(scenario, state)
            cached_revision = revision
        terminal = cached_terminal
        if terminal is None and state.activation_count >= scenario.max_activations:
            terminal = "event_budget"
        if terminal is not None:
            state.terminal = terminal
            break

        event_index = state.activation_count
        actor_id = scheduler.actor(event_index)
        actor = scenario.cell_map[actor_id]
        side: str | None = None
        if actor.policy == Policy.BUBBLE and actor.fault == FaultMode.NORMAL:
            side = "left" if u64(scenario.seed, scenario.scenario_id, "bubble_side", event_index, 0) < (1 << 63) else "right"
            state.stream_counters["bubble_side"] = state.stream_counters.get("bubble_side", 0) + 1
        if settings["scheduler"] == "uniform_random_activation":
            # Rejection-block accounting is held by CompactScheduler; the run-state
            # stream count must match the authoritative counter.
            state.stream_counters["actor_activation"] = scheduler.rng_blocks
        elif event_index % len(scenario.cells) == 0:
            state.stream_counters[PERMUTATION_STREAM] = scheduler.rng_blocks

        if actor.policy == Policy.INSERTION and insertion_cache_revision != revision:
            insertion_prefix_cache = build_insertion_prefix_cache(scenario, state)
            insertion_cache_revision = revision
        proposal = compact_policy_proposal(
            scenario,
            state,
            positions,
            actor_id,
            side=side,
            insertion_prefix_cache=insertion_prefix_cache,
        )
        if settings["coordinatorProfile"] == "weak_frozen_budget" and event_index % 8 == 7:
            audit.coordinator_eligible += 1
            if (
                proposal.kind == ProposalKind.SWAP
                and proposal.target_pos is not None
                and abs(proposal.target_pos - proposal.actor_pos) > 1
            ):
                proposal = Proposal(
                    ProposalKind.NO_OP,
                    proposal.actor_id,
                    proposal.actor_pos,
                    reason="coordinator_veto_nonlocal_swap",
                    observation_reads=proposal.observation_reads,
                    value_comparisons=proposal.value_comparisons,
                )
                audit.coordinator_interventions += 1
        validation = validate_proposal(scenario, state, proposal)
        decision = "accepted" if validation.eligible_for_commit else validation.decision
        delta = cost_delta(state.ledger, proposal, decision)
        changed = False
        if decision == "accepted" and proposal.kind == ProposalKind.MEMORY_UPDATE:
            assert proposal.new_cursor is not None
            state.selection_cursors[proposal.actor_id] = proposal.new_cursor
            changed = True
        elif decision == "accepted" and proposal.kind == ProposalKind.SWAP:
            assert proposal.target_pos is not None
            actor_pos, target_pos = proposal.actor_pos, proposal.target_pos
            actor_at = state.occupancy[actor_pos]
            target_at = state.occupancy[target_pos]
            state.occupancy[actor_pos], state.occupancy[target_pos] = target_at, actor_at
            positions[actor_at], positions[target_at] = target_pos, actor_pos
            changed = True
        for key, value in delta.items():
            state.ledger[key] += value
        state.activation_count += 1
        if changed:
            revision += 1

        if cached_revision != revision:
            cached_terminal = _base_terminal(scenario, state)
            cached_revision = revision
        after_terminal = cached_terminal
        if after_terminal is None and state.activation_count >= scenario.max_activations:
            after_terminal = "event_budget"
        blocking = _blocking_disposition(scenario, proposal, decision)
        if blocking:
            audit.blocking_failures += 1
            if settings["continuation"] == "stop_on_first_blocking_failure" and after_terminal is None:
                after_terminal = "blocking_failure"
                audit.continuation_stops += 1
        if after_terminal is not None:
            state.terminal = after_terminal
            cached_terminal = after_terminal

        if retain_step_projection:
            event_projection = {
                "eventIndex": event_index,
                "actorId": actor_id,
                "side": side,
                "proposal": {
                    "kind": proposal.kind.value,
                    "actorPos": proposal.actor_pos,
                    "targetPos": proposal.target_pos,
                    "newCursor": proposal.new_cursor,
                    "reason": proposal.reason,
                    "observationReads": proposal.observation_reads,
                    "valueComparisons": proposal.value_comparisons,
                    "observedTargetId": proposal.observed_target_id,
                },
                "decision": decision,
                "ledgerDelta": delta,
                "stopReasonIfAny": after_terminal,
            }
            encoded = canonical_json_bytes(event_projection)
            digest = hashlib.sha256(digest + encoded).digest()
            step_projection.append(event_projection)
        if state.terminal is not None:
            break

    wall_time = (perf_counter_ns() - start) / 1_000_000_000
    final_values = [scenario.cell_map[cell_id].value for cell_id in state.occupancy]
    direction = scenario.cells[0].direction
    residual_count = sum(
        (left > right if direction == Direction.ASCENDING else left < right)
        for left, right in zip(final_values, final_values[1:])
    )
    costs, projections, normalizations, validations = _costs(
        scenario, state, settings, scheduler, audit, wall_time
    )
    deterministic = {
        "executorVersion": COMPACT_EXECUTOR_VERSION,
        "scenarioId": scenario.scenario_id,
        "settings": dict(settings),
        "stopReason": state.terminal,
        "activationCount": state.activation_count,
        "finalOccupancy": state.occupancy,
        "selectionCursors": dict(sorted(state.selection_cursors.items())),
        "streamCounters": dict(sorted(state.stream_counters.items())),
        "nativeLedger": dict(state.ledger),
        "compactTransitionDigest": digest.hex() if retain_step_projection else None,
        "costs": {key: value for key, value in costs.items() if key != "wallTimeSeconds"},
        "projections": projections,
        "normalizations": normalizations,
        "validation": validations,
    }
    replay_projection = {
        "scenarioId": scenario.scenario_id,
        "settings": dict(settings),
        "stopReason": state.terminal,
        "activationCount": state.activation_count,
        "finalOccupancy": state.occupancy,
        "selectionCursors": dict(sorted(state.selection_cursors.items())),
        "streamCounters": dict(sorted(state.stream_counters.items())),
        "nativeLedger": dict(state.ledger),
        "costs": {key: value for key, value in costs.items() if key != "wallTimeSeconds"},
        "projections": projections,
    }
    return {
        "schemaVersion": "e02.s10.screening_run.v1",
        **deterministic,
        "deterministicResultSha256": sha256_json(deterministic),
        "compactReplayDigest": sha256_json(replay_projection),
        "finalValues": final_values,
        "normalizedResidualError": residual_count / max(len(final_values) - 1, 1),
        "strictAdjacentResidualCount": residual_count,
        "successByBudget": int(state.terminal == "complete"),
        "completionOpportunity": state.activation_count,
        "completionObserved": int(state.terminal == "complete"),
        "quiescentCompetingFailure": int(state.terminal == "quiescent"),
        "wallTimeSeconds": wall_time,
        "costSchemaVersion": COST_SCHEMA_VERSION,
        "costSchemaSha256": COST_SCHEMA_SHA256,
        **{f"cost_{key}": value for key, value in costs.items()},
        **{f"projection_{key}": value for key, value in projections.items()},
        **{f"normalization_{key}": value for key, value in normalizations.items()},
        "contractValidationPass": all(validations.values()),
        "contractValidationJson": canonical_json_bytes(validations).decode("utf-8"),
        "stepProjection": step_projection if retain_step_projection else None,
    }


def flat_screening_row(
    run_design: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach immutable design identifiers to a compact result row."""

    excluded = {"stepProjection", "settings", "finalOccupancy", "selectionCursors", "streamCounters", "nativeLedger", "costs", "projections", "normalizations", "validation"}
    row = {key: value for key, value in run_design.items() if key not in {"settingsJson", "linkedEstimandArmsJson"}}
    row.update({key: value for key, value in result.items() if key not in excluded})
    row["nativeLedgerJson"] = canonical_json_bytes(result["nativeLedger"]).decode("utf-8")
    row["finalOccupancySha256"] = sha256_json(result["finalOccupancy"])
    row["selectionCursorsSha256"] = sha256_json(result["selectionCursors"])
    row["streamCountersJson"] = canonical_json_bytes(result["streamCounters"]).decode("utf-8")
    return row


def repository_code_sha256() -> str:
    return sha256_json(
        {
            "screening": Path(__file__).read_text(),
            "costSchemaSha256": COST_SCHEMA_SHA256,
        }
    )
