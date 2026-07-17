"""E05 S01 task contracts for development and regeneration benchmarks.

This module deliberately implements task definitions and validation fixtures,
not dynamic lesion processes.  The single adjacent-swap perturbation is an S01
validation fixture and is not the S03 lesion library.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import jsonschema

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.schedulers import (
    SchedulerExecutionContract,
    SchedulerFamily,
    run_scheduled_architecture,
)
from reference_simulator.api import create_scenario
from reference_simulator.engine import EMPTY_DIGEST, execute_batch
from reference_simulator.model import (
    Direction,
    LEDGER_FIELDS,
    Policy,
    ProposalKind,
    RunState,
    Scenario,
    canonical_json_bytes,
    sha256_json,
)
from reference_simulator.policies import cell_view_proposal
from reference_simulator.rng import u64
from reference_simulator.scheduler import ScheduledOpportunity, scheduled_actor
from reference_simulator.transition_primitives import validate_proposal


BENCHMARK_VERSION = "E05-regeneration-task-v1"
TASK_SCHEMA_VERSION = "e05.s01.task-spec.v1"
BASELINE_SCHEMA_VERSION = "e05.s01.baseline-scenario.v1"
VALIDATION_SCHEMA_VERSION = "e05.s01.validation-result.v1"
EVENT_BUDGET_PROFILE = "profile_scaled_frozen_s07_v1"
PAIRING_PROFILE = "e05.s01.preinjury-shared-prefix-v1"
MASTER_SEED = int("e0501000000000000000000000000001", 16)


class TaskFamily(str, Enum):
    DEVELOPMENT_RANDOM = "development_random"
    REPAIR_ACHIEVED = "repair_achieved"
    REPAIR_PARTIAL = "repair_partial"


class Arm(str, Enum):
    DEVELOPMENT_BASELINE = "development_baseline"
    MATCHED_NO_INJURY = "matched_no_injury"
    VALIDATION_INJURY = "validation_injury"


TASK_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s01/task-spec.schema.json",
    "title": "E05 S01 regeneration task specification",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenQuestion",
        "taskFamilies",
        "executionDefaults",
        "eventBudget",
        "stabilization",
        "controls",
        "pairing",
        "claimBoundary",
        "validationPanel",
    ],
    "properties": {
        "schemaVersion": {"const": TASK_SCHEMA_VERSION},
        "researchStepId": {"const": "S01"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "frozenQuestion": {"type": "string", "minLength": 20},
        "taskFamilies": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "taskFamily",
                    "startState",
                    "preInjuryRequirement",
                    "injuryOnset",
                    "primaryTarget",
                    "secondaryTarget",
                    "successRule",
                ],
                "properties": {
                    "taskFamily": {"enum": [item.value for item in TaskFamily]},
                    "startState": {"type": "string", "minLength": 3},
                    "preInjuryRequirement": {"type": ["string", "null"]},
                    "injuryOnset": {"type": ["string", "null"]},
                    "primaryTarget": {"type": "string", "minLength": 3},
                    "secondaryTarget": {"type": ["string", "null"]},
                    "successRule": {"type": "string", "minLength": 3},
                },
            },
        },
        "executionDefaults": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "architecture",
                "scheduler",
                "schedulerSensitivity",
                "mobility",
                "continuation",
                "retry",
                "actionFailure",
                "sensing",
                "informationPermission",
            ],
            "properties": {
                "architecture": {"const": "distributed_local"},
                "scheduler": {"const": "uniform_random_activation"},
                "schedulerSensitivity": {"const": "random_permutation_sweep"},
                "mobility": {"const": "normal"},
                "continuation": {"const": "skip_and_continue"},
                "retry": {"const": "no_retry"},
                "actionFailure": {"const": "none"},
                "sensing": {"const": "exact"},
                "informationPermission": {"const": "policy_native_local"},
            },
        },
        "eventBudget": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "profile",
                "developmentOpportunities",
                "recoveryOpportunities",
                "terminalHandling",
                "ledgerRule",
            ],
            "properties": {
                "profile": {"const": EVENT_BUDGET_PROFILE},
                "developmentOpportunities": {"const": "100*n^2"},
                "recoveryOpportunities": {"const": "100*n^2 after injury"},
                "terminalHandling": {"type": "string", "minLength": 10},
                "ledgerRule": {"type": "string", "minLength": 10},
            },
        },
        "stabilization": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "achievedRule",
                "uniformCoverageRule",
                "uniformCoverageCap",
                "permutationSensitivityRule",
                "partialRule",
            ],
            "properties": {
                "achievedRule": {"type": "string", "minLength": 10},
                "uniformCoverageRule": {
                    "const": "at least two opportunities per identity"
                },
                "uniformCoverageCap": {"const": "20*n opportunities"},
                "permutationSensitivityRule": {
                    "const": "two complete sweeps (2*n opportunities)"
                },
                "partialRule": {"type": "string", "minLength": 10},
            },
        },
        "controls": {
            "type": "array",
            "minItems": 2,
            "items": {"type": "string", "minLength": 5},
        },
        "pairing": {
            "type": "object",
            "additionalProperties": False,
            "required": ["profile", "sharedFields", "rngStatus"],
            "properties": {
                "profile": {"const": PAIRING_PROFILE},
                "sharedFields": {
                    "type": "array",
                    "minItems": 5,
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "rngStatus": {"const": "shared_prefix_until_arm_terminal"},
            },
        },
        "claimBoundary": {"type": "string", "minLength": 30},
        "validationPanel": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "sizes",
                "policies",
                "directions",
                "replicatesPerCell",
                "inputProfile",
                "injuryFixture",
            ],
            "properties": {
                "sizes": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "integer", "minimum": 8},
                    "uniqueItems": True,
                },
                "policies": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"enum": [item.value for item in Policy]},
                    "uniqueItems": True,
                },
                "directions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"enum": [item.value for item in Direction]},
                    "uniqueItems": True,
                },
                "replicatesPerCell": {"type": "integer", "minimum": 1},
                "inputProfile": {
                    "const": "counter_addressed_random_permutation_unique_values"
                },
                "injuryFixture": {"const": "s01_validation_adjacent_swap_v1"},
            },
        },
    },
}


BASELINE_SCENARIO_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s01/baseline-scenario.schema.json",
    "title": "E05 S01 baseline scenario row",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schemaVersion",
        "benchmarkVersion",
        "baselineScenarioId",
        "pairingBlockId",
        "taskFamily",
        "arm",
        "n",
        "policy",
        "direction",
        "replicateOrdinal",
        "runtimeSeed",
        "injurySeed",
        "scheduler",
        "continuation",
        "mobility",
        "eventBudgetProfile",
        "developmentBudget",
        "recoveryBudget",
        "stabilizationOpportunityCap",
        "initialDistance",
        "preInjuryDistance",
        "postInjuryDistance",
        "primaryRecoveryThreshold",
        "preInjuryEventIndex",
        "initialOccupancySha256",
        "preInjuryStateHash",
        "postInjuryStateHash",
        "targetValuesSha256",
        "injuryOperator",
        "injuryOnsetRule",
        "stabilizationRule",
        "rngPairingStatus",
        "executableScenarioId",
    ],
    "properties": {
        "schemaVersion": {"const": BASELINE_SCHEMA_VERSION},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "baselineScenarioId": {"type": "string", "pattern": "^e05s01:[0-9a-f]{64}$"},
        "pairingBlockId": {"type": "string", "pattern": "^e05pb1:[0-9a-f]{64}$"},
        "taskFamily": {"enum": [item.value for item in TaskFamily]},
        "arm": {"enum": [item.value for item in Arm]},
        "n": {"type": "integer", "minimum": 2},
        "policy": {"enum": [item.value for item in Policy]},
        "direction": {"enum": [item.value for item in Direction]},
        "replicateOrdinal": {"type": "integer", "minimum": 0},
        "runtimeSeed": {"type": "string", "pattern": "^[0-9]+$"},
        "injurySeed": {"type": "string", "pattern": "^[0-9]+$"},
        "scheduler": {"const": "uniform_random_activation"},
        "continuation": {"const": "skip_and_continue"},
        "mobility": {"const": "normal"},
        "eventBudgetProfile": {"const": EVENT_BUDGET_PROFILE},
        "developmentBudget": {"type": "integer", "minimum": 400},
        "recoveryBudget": {"type": "integer", "minimum": 400},
        "stabilizationOpportunityCap": {"type": "integer", "minimum": 40},
        "initialDistance": {"type": "integer", "minimum": 1},
        "preInjuryDistance": {"type": ["integer", "null"], "minimum": 0},
        "postInjuryDistance": {"type": "integer", "minimum": 0},
        "primaryRecoveryThreshold": {"type": "integer", "minimum": 0},
        "preInjuryEventIndex": {"type": ["integer", "null"], "minimum": 0},
        "initialOccupancySha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "preInjuryStateHash": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
        "postInjuryStateHash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "targetValuesSha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "injuryOperator": {"enum": ["none", "s01_validation_adjacent_swap_v1"]},
        "injuryOnsetRule": {"type": ["string", "null"]},
        "stabilizationRule": {"type": ["string", "null"]},
        "rngPairingStatus": {
            "enum": ["not_applicable", "shared_prefix_until_arm_terminal"]
        },
        "executableScenarioId": {"type": "string", "pattern": "^r1:[0-9a-f]{64}$"},
    },
    "allOf": [
        {
            "if": {"properties": {"taskFamily": {"const": "development_random"}}},
            "then": {
                "properties": {
                    "arm": {"const": "development_baseline"},
                    "preInjuryDistance": {"type": "null"},
                    "preInjuryEventIndex": {"type": "null"},
                    "preInjuryStateHash": {"type": "null"},
                    "injuryOperator": {"const": "none"},
                    "injuryOnsetRule": {"type": "null"},
                    "stabilizationRule": {"type": "null"},
                    "rngPairingStatus": {"const": "not_applicable"},
                }
            },
        },
        {
            "if": {"properties": {"taskFamily": {"pattern": "^repair_"}}},
            "then": {
                "properties": {
                    "arm": {"enum": ["matched_no_injury", "validation_injury"]},
                    "preInjuryDistance": {"type": "integer"},
                    "preInjuryEventIndex": {"type": "integer"},
                    "preInjuryStateHash": {"type": "string"},
                    "injuryOnsetRule": {"type": "string"},
                    "rngPairingStatus": {"const": "shared_prefix_until_arm_terminal"},
                }
            },
        },
    ],
}


@dataclass(frozen=True, slots=True)
class Checkpoint:
    occupancy: tuple[str, ...]
    selection_cursors: tuple[tuple[str, int], ...]
    activation_count: int
    stream_counters: tuple[tuple[str, int], ...]
    ledger: tuple[tuple[str, int], ...]
    distance: int
    state_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "occupancy": list(self.occupancy),
            "selectionCursors": dict(self.selection_cursors),
            "activationCount": self.activation_count,
            "streamCounters": dict(self.stream_counters),
            "ledger": dict(self.ledger),
            "distance": self.distance,
            "stateHash": self.state_hash,
        }

    def to_run_state(self, *, occupancy: Sequence[str] | None = None) -> RunState:
        return RunState(
            occupancy=list(self.occupancy if occupancy is None else occupancy),
            selection_cursors=dict(self.selection_cursors),
            activation_count=self.activation_count,
            stream_counters=dict(self.stream_counters),
            ledger=dict(self.ledger),
            terminal=None,
        )


@dataclass(frozen=True, slots=True)
class PhaseResult:
    summary: Mapping[str, Any]
    final_state: Mapping[str, Any]
    event_digest: str
    events: tuple[Mapping[str, Any], ...]


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_unequal_inversions(
    values: Sequence[int | float], direction: Direction | str
) -> int:
    """Count direction-aware discordant unequal pairs; equal values never count."""

    selected = Direction(direction)
    if selected == Direction.ASCENDING:
        return sum(
            left > right
            for index, left in enumerate(values)
            for right in values[index + 1 :]
        )
    return sum(
        left < right
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )


def target_values(
    values: Sequence[int | float], direction: Direction | str
) -> tuple[int | float, ...]:
    return tuple(sorted(values, reverse=Direction(direction) == Direction.DESCENDING))


def occupancy_values(
    scenario: Scenario, occupancy: Sequence[str]
) -> tuple[int | float, ...]:
    return tuple(scenario.cell_map[cell_id].value for cell_id in occupancy)


def _checkpoint_hash(
    scenario_id: str,
    occupancy: Sequence[str],
    selection_cursors: Mapping[str, int],
    activation_count: int,
    stream_counters: Mapping[str, int],
    ledger: Mapping[str, int],
) -> str:
    return _hash(
        {
            "scenarioId": scenario_id,
            "occupancy": list(occupancy),
            "selectionCursors": dict(sorted(selection_cursors.items())),
            "activationCount": activation_count,
            "streamCounters": dict(sorted(stream_counters.items())),
            "ledger": {field: int(ledger[field]) for field in LEDGER_FIELDS},
        }
    )


def initial_checkpoint(scenario: Scenario) -> Checkpoint:
    cursors = dict(scenario.initial_selection_cursors)
    ledger = {field: 0 for field in LEDGER_FIELDS}
    distance = strict_unequal_inversions(
        occupancy_values(scenario, scenario.initial_occupancy),
        scenario.cells[0].direction,
    )
    return Checkpoint(
        tuple(scenario.initial_occupancy),
        tuple(sorted(cursors.items())),
        0,
        (),
        tuple(ledger.items()),
        distance,
        _checkpoint_hash(
            scenario.scenario_id,
            scenario.initial_occupancy,
            cursors,
            0,
            {},
            ledger,
        ),
    )


def replay_checkpoints(
    scenario: Scenario, events: Iterable[Mapping[str, Any]]
) -> list[Checkpoint]:
    """Reconstruct compact occupancy/cursor checkpoints from authoritative events."""

    occupancy = list(scenario.initial_occupancy)
    cursors = dict(scenario.initial_selection_cursors)
    stream_counters: dict[str, int] = {}
    ledger = {field: 0 for field in LEDGER_FIELDS}
    checkpoints: list[Checkpoint] = []
    for expected_index, event in enumerate(events):
        if int(event["eventIndex"]) != expected_index:
            raise ValueError("event indices are not contiguous")
        proposal = event["proposal"]
        if event["decision"] == "accepted":
            if proposal["kind"] == "Swap":
                left = int(proposal["actorPos"])
                right = int(proposal["targetPos"])
                occupancy[left], occupancy[right] = occupancy[right], occupancy[left]
            elif proposal["kind"] == "MemoryUpdate":
                cursors[str(proposal["actorId"])] = int(proposal["newCursor"])
        for draw in event["randomAddressesAndDraws"]:
            stream = str(draw["stream"])
            stream_counters[stream] = stream_counters.get(stream, 0) + 1
        for field, value in event["ledgerDelta"].items():
            ledger[str(field)] += int(value)
        distance = strict_unequal_inversions(
            occupancy_values(scenario, occupancy), scenario.cells[0].direction
        )
        checkpoints.append(
            Checkpoint(
                tuple(occupancy),
                tuple(sorted(cursors.items())),
                expected_index + 1,
                tuple(sorted(stream_counters.items())),
                tuple((field, ledger[field]) for field in LEDGER_FIELDS),
                distance,
                _checkpoint_hash(
                    scenario.scenario_id,
                    occupancy,
                    cursors,
                    expected_index + 1,
                    stream_counters,
                    ledger,
                ),
            )
        )
    return checkpoints


def first_progress_checkpoint(
    scenario: Scenario,
    events: Iterable[Mapping[str, Any]],
    *,
    progress_fraction: float = 0.5,
) -> Checkpoint:
    if not 0.0 < progress_fraction < 1.0:
        raise ValueError("partial progress fraction must be inside (0, 1)")
    start = initial_checkpoint(scenario)
    if start.distance < 2:
        raise ValueError("partial checkpoint requires at least two initial inversions")
    maximum_distance = int(start.distance * (1.0 - progress_fraction))
    maximum_distance = max(1, maximum_distance)
    for checkpoint in replay_checkpoints(scenario, events):
        if 0 < checkpoint.distance <= maximum_distance:
            return checkpoint
    raise ValueError("run did not expose a nonterminal partial-progress checkpoint")


def apply_validation_injury(
    occupancy: Sequence[str],
    values_by_id: Mapping[str, int | float],
    direction: Direction | str,
) -> tuple[tuple[str, ...], int]:
    """Swap one central, correctly ordered unequal adjacent pair.

    The fixture increases strict unequal inversion distance by exactly one.  It
    is only an S01 feasibility probe and must not be treated as the S03 lesion
    operator library.
    """

    selected = Direction(direction)
    candidates: list[int] = []
    for index in range(len(occupancy) - 1):
        left = values_by_id[occupancy[index]]
        right = values_by_id[occupancy[index + 1]]
        ordered = left < right if selected == Direction.ASCENDING else left > right
        if ordered:
            candidates.append(index)
    if not candidates:
        raise ValueError("no correctly ordered unequal adjacent pair is available")
    center = (len(occupancy) - 2) / 2
    index = min(candidates, key=lambda item: (abs(item - center), item))
    damaged = list(occupancy)
    damaged[index], damaged[index + 1] = damaged[index + 1], damaged[index]
    before = strict_unequal_inversions(
        [values_by_id[item] for item in occupancy], selected
    )
    after = strict_unequal_inversions(
        [values_by_id[item] for item in damaged], selected
    )
    if after != before + 1:
        raise AssertionError("S01 adjacent-swap fixture must add exactly one inversion")
    return tuple(damaged), index


def _checkpoint_from_state(scenario: Scenario, state: RunState) -> Checkpoint:
    distance = strict_unequal_inversions(
        occupancy_values(scenario, state.occupancy), scenario.cells[0].direction
    )
    return Checkpoint(
        tuple(state.occupancy),
        tuple(sorted(state.selection_cursors.items())),
        state.activation_count,
        tuple(sorted(state.stream_counters.items())),
        tuple((field, state.ledger[field]) for field in LEDGER_FIELDS),
        distance,
        _checkpoint_hash(
            scenario.scenario_id,
            state.occupancy,
            state.selection_cursors,
            state.activation_count,
            state.stream_counters,
            state.ledger,
        ),
    )


def absorbing_occupancy_certificate(
    scenario: Scenario, checkpoint: Checkpoint
) -> dict[str, Any]:
    """Exhaustively verify that no identity/side can commit an occupancy swap."""

    state = checkpoint.to_run_state()
    eligible_swaps: list[dict[str, Any]] = []
    proposal_count = 0
    for cell in scenario.cells:
        sides: tuple[str | None, ...] = (
            ("left", "right") if cell.policy == Policy.BUBBLE else (None,)
        )
        for side in sides:
            proposal_count += 1
            proposal = cell_view_proposal(
                scenario,
                state,
                cell.cell_id,
                side=side,  # type: ignore[arg-type]
            )
            decision = validate_proposal(scenario, state, proposal)
            if proposal.kind == ProposalKind.SWAP and decision.eligible_for_commit:
                eligible_swaps.append(
                    {
                        "actorId": cell.cell_id,
                        "side": side,
                        "actorPos": proposal.actor_pos,
                        "targetPos": proposal.target_pos,
                    }
                )
    return {
        "proposalCount": proposal_count,
        "eligibleOccupancySwapCount": len(eligible_swaps),
        "eligibleOccupancySwaps": eligible_swaps,
        "success": checkpoint.distance == 0 and not eligible_swaps,
    }


def _uniform_schedule(scenario: Scenario):
    def schedule(
        event_index: int, remaining_opportunities: int
    ) -> tuple[ScheduledOpportunity, ...]:
        if remaining_opportunities < 1:
            return ()
        actor_id, draws, consumed = scheduled_actor(
            scenario, event_index, include_draws=True
        )
        return (
            ScheduledOpportunity(
                actor_id,
                draws,
                (("actor_activation", consumed),),
            ),
        )

    return schedule


def stabilize_achieved_checkpoint(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    minimum_per_actor: int = 2,
    cap_multiplier: int = 20,
) -> tuple[Checkpoint, dict[str, Any]]:
    """Run the charged post-target stability probe while suppressing target stop."""

    if minimum_per_actor < 1 or cap_multiplier < minimum_per_actor:
        raise ValueError("invalid stabilization coverage parameters")
    certificate = absorbing_occupancy_certificate(scenario, checkpoint)
    state = checkpoint.to_run_state()
    actor_ids = tuple(cell.cell_id for cell in scenario.cells)
    counts: Counter[str] = Counter()
    cap = cap_multiplier * len(actor_ids)
    accepted_movement_count = 0
    schedule = _uniform_schedule(scenario)
    for offset in range(cap):
        state.terminal = None
        events, _ = execute_batch(
            scenario,
            state,
            retain_events=True,
            proposal_factory=cell_view_proposal,
            schedule_factory=schedule,
        )
        if len(events) != 1:
            raise AssertionError("serial uniform stabilization must emit one event")
        event = events[0]
        counts[str(event["actorId"])] += 1
        accepted_movement_count += int(
            event["decision"] == "accepted"
            and event["proposal"]["kind"] == ProposalKind.SWAP.value
        )
        covered = (
            min(counts.get(actor_id, 0) for actor_id in actor_ids) >= minimum_per_actor
        )
        if accepted_movement_count or state.terminal not in {None, "complete"}:
            break
        if covered:
            stabilized = _checkpoint_from_state(scenario, state)
            audit = {
                "coveragePass": True,
                "opportunities": offset + 1,
                "cap": cap,
                "minimumPerActor": minimum_per_actor,
                "minimumObserved": min(counts.values()),
                "acceptedMovementCount": accepted_movement_count,
                "absorbingTargetCertificate": certificate["success"],
                "absorbingProposalCount": certificate["proposalCount"],
                "eligibleOccupancySwapCount": certificate["eligibleOccupancySwapCount"],
                "startEventIndex": checkpoint.activation_count,
                "endEventIndex": stabilized.activation_count,
                "success": certificate["success"] and stabilized.distance == 0,
            }
            return stabilized, audit
    failed = _checkpoint_from_state(scenario, state)
    audit = {
        "coveragePass": False,
        "opportunities": failed.activation_count - checkpoint.activation_count,
        "cap": cap,
        "minimumPerActor": minimum_per_actor,
        "minimumObserved": min(counts.get(actor_id, 0) for actor_id in actor_ids),
        "acceptedMovementCount": accepted_movement_count,
        "absorbingTargetCertificate": certificate["success"],
        "absorbingProposalCount": certificate["proposalCount"],
        "eligibleOccupancySwapCount": certificate["eligibleOccupancySwapCount"],
        "startEventIndex": checkpoint.activation_count,
        "endEventIndex": failed.activation_count,
        "success": False,
    }
    return failed, audit


def run_recovery_phase(
    scenario: Scenario,
    checkpoint: Checkpoint,
    *,
    postinjury_occupancy: Sequence[str],
    primary_threshold: int,
    recovery_budget: int,
) -> PhaseResult:
    """Resume the exact state and global RNG clock after an occupancy lesion."""

    state = checkpoint.to_run_state(occupancy=postinjury_occupancy)
    start_event = state.activation_count
    initial_distance = strict_unequal_inversions(
        occupancy_values(scenario, state.occupancy), scenario.cells[0].direction
    )
    primary_time: int | None = 0 if initial_distance <= primary_threshold else None
    retained: list[Mapping[str, Any]] = []
    digest = bytes.fromhex(EMPTY_DIGEST)
    schedule = _uniform_schedule(scenario)
    stop_reason: str | None = "complete" if initial_distance == 0 else None
    while (
        stop_reason is None and state.activation_count - start_event < recovery_budget
    ):
        state.terminal = None
        events, encoded_events = execute_batch(
            scenario,
            state,
            retain_events=True,
            proposal_factory=cell_view_proposal,
            schedule_factory=schedule,
        )
        for encoded in encoded_events:
            digest = hashlib.sha256(digest + encoded).digest()
        retained.extend(events)
        distance = strict_unequal_inversions(
            occupancy_values(scenario, state.occupancy), scenario.cells[0].direction
        )
        if primary_time is None and distance <= primary_threshold:
            primary_time = state.activation_count - start_event
        if state.terminal is not None:
            stop_reason = state.terminal
    if stop_reason is None:
        stop_reason = "phase_event_budget"
        state.terminal = stop_reason
    final_values = occupancy_values(scenario, state.occupancy)
    phase_activations = state.activation_count - start_event
    summary = {
        "stopReason": stop_reason,
        "completed": strict_unequal_inversions(
            final_values, scenario.cells[0].direction
        )
        == 0,
        "activationCount": phase_activations,
        "globalStartEventIndex": start_event,
        "globalEndEventIndex": state.activation_count,
        "primaryRecoveryTime": primary_time,
        "finalOccupancy": list(state.occupancy),
        "finalValues": list(final_values),
        "ledger": dict(state.ledger),
        "traceMode": "full",
        "retainedEventCount": len(retained),
    }
    return PhaseResult(
        summary=summary,
        final_state=state.to_dict(),
        event_digest=digest.hex(),
        events=tuple(retained),
    )


def validate_task_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(TASK_SPEC_SCHEMA).validate(specification)
    families = [item["taskFamily"] for item in specification["taskFamilies"]]
    if sorted(families) != sorted(item.value for item in TaskFamily):
        raise ValueError("task specification must define each task family exactly once")
    by_family = {item["taskFamily"]: item for item in specification["taskFamilies"]}
    development = by_family[TaskFamily.DEVELOPMENT_RANDOM.value]
    if (
        development["preInjuryRequirement"] is not None
        or development["injuryOnset"] is not None
    ):
        raise ValueError(
            "development cannot contain a pre-injury state or injury onset"
        )
    achieved = by_family[TaskFamily.REPAIR_ACHIEVED.value]
    partial = by_family[TaskFamily.REPAIR_PARTIAL.value]
    if achieved["preInjuryRequirement"] == partial["preInjuryRequirement"]:
        raise ValueError(
            "achieved and partial repair require distinct pre-injury classes"
        )
    if achieved["primaryTarget"] == partial["primaryTarget"]:
        raise ValueError("achieved and partial repair require distinct primary targets")


def _seed(*parts: str | int) -> int:
    address = "/".join(str(item) for item in parts)
    draw_index = int(_hash(address)[:8], 16)
    return u64(
        MASTER_SEED,
        "E05/S01",
        "scenario_construction_s01_v1",
        0,
        draw_index,
    )


def _pairing_id(content: Mapping[str, Any]) -> str:
    return "e05pb1:" + sha256_json(content)


def _baseline_id(content: Mapping[str, Any]) -> str:
    return "e05s01:" + sha256_json(content)


def _row(
    *,
    pairing_id: str,
    family: TaskFamily,
    arm: Arm,
    source: Scenario,
    executable: Scenario,
    replicate: int,
    injury_seed: int,
    initial_distance: int,
    preinjury: Checkpoint | None,
    postinjury_occupancy: Sequence[str],
    primary_threshold: int,
    target_sha256: str,
    injury_operator: str,
    injury_onset: str | None,
    stabilization_rule: str | None,
) -> dict[str, Any]:
    post_distance = strict_unequal_inversions(
        occupancy_values(source, postinjury_occupancy), source.cells[0].direction
    )
    content = {
        "pairingBlockId": pairing_id,
        "taskFamily": family.value,
        "arm": arm.value,
        "postInjuryStateHash": _hash(
            {
                "preInjuryStateHash": None
                if preinjury is None
                else preinjury.state_hash,
                "occupancy": list(postinjury_occupancy),
            }
        ),
        "injuryOperator": injury_operator,
        "executableScenarioId": executable.scenario_id,
    }
    return {
        "schemaVersion": BASELINE_SCHEMA_VERSION,
        "benchmarkVersion": BENCHMARK_VERSION,
        "baselineScenarioId": _baseline_id(content),
        "pairingBlockId": pairing_id,
        "taskFamily": family.value,
        "arm": arm.value,
        "n": len(source.cells),
        "policy": source.cells[0].policy.value,
        "direction": source.cells[0].direction.value,
        "replicateOrdinal": replicate,
        "runtimeSeed": str(source.seed),
        "injurySeed": str(injury_seed),
        "scheduler": "uniform_random_activation",
        "continuation": "skip_and_continue",
        "mobility": "normal",
        "eventBudgetProfile": EVENT_BUDGET_PROFILE,
        "developmentBudget": 100 * len(source.cells) ** 2,
        "recoveryBudget": 100 * len(source.cells) ** 2,
        "stabilizationOpportunityCap": 20 * len(source.cells),
        "initialDistance": initial_distance,
        "preInjuryDistance": None if preinjury is None else preinjury.distance,
        "postInjuryDistance": post_distance,
        "primaryRecoveryThreshold": primary_threshold,
        "preInjuryEventIndex": None
        if preinjury is None
        else preinjury.activation_count,
        "initialOccupancySha256": _hash(list(source.initial_occupancy)),
        "preInjuryStateHash": None if preinjury is None else preinjury.state_hash,
        "postInjuryStateHash": _hash(
            {
                "preInjuryStateHash": None
                if preinjury is None
                else preinjury.state_hash,
                "occupancy": list(postinjury_occupancy),
            }
        ),
        "targetValuesSha256": target_sha256,
        "injuryOperator": injury_operator,
        "injuryOnsetRule": injury_onset,
        "stabilizationRule": stabilization_rule,
        "rngPairingStatus": (
            "not_applicable"
            if family == TaskFamily.DEVELOPMENT_RANDOM
            else "shared_prefix_until_arm_terminal"
        ),
        "executableScenarioId": executable.scenario_id,
    }


def validate_pairing_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["pairingBlockId"]), str(row["taskFamily"]))
        grouped.setdefault(key, []).append(row)
    failures: list[str] = []
    checked_pairs = 0
    for (pairing_id, family), items in sorted(grouped.items()):
        if family == TaskFamily.DEVELOPMENT_RANDOM.value:
            if len(items) != 1 or items[0]["arm"] != Arm.DEVELOPMENT_BASELINE.value:
                failures.append(f"{pairing_id}/{family}: invalid development singleton")
            continue
        checked_pairs += 1
        by_arm = {str(item["arm"]): item for item in items}
        if set(by_arm) != {Arm.MATCHED_NO_INJURY.value, Arm.VALIDATION_INJURY.value}:
            failures.append(f"{pairing_id}/{family}: missing paired arms")
            continue
        left, right = by_arm.values()
        shared = (
            "n",
            "policy",
            "direction",
            "replicateOrdinal",
            "runtimeSeed",
            "injurySeed",
            "preInjuryDistance",
            "preInjuryEventIndex",
            "preInjuryStateHash",
            "executableScenarioId",
            "targetValuesSha256",
            "developmentBudget",
            "recoveryBudget",
            "scheduler",
            "continuation",
            "mobility",
            "rngPairingStatus",
        )
        for field in shared:
            if left[field] != right[field]:
                failures.append(f"{pairing_id}/{family}: shared field differs: {field}")
    identifiers = [str(row["baselineScenarioId"]) for row in rows]
    return {
        "pairingProfile": PAIRING_PROFILE,
        "rowCount": len(rows),
        "repairPairCount": checked_pairs,
        "uniqueScenarioIds": len(identifiers) == len(set(identifiers)),
        "failureCount": len(failures),
        "failures": failures,
        "success": not failures and len(identifiers) == len(set(identifiers)),
    }


def build_baseline_panel(specification: Mapping[str, Any]) -> dict[str, Any]:
    """Build and execute the deterministic S01 validation panel."""

    validate_task_spec(specification)
    panel = specification["validationPanel"]
    architecture = ArchitectureExecutionContract.distributed_local()
    scheduler = SchedulerExecutionContract(SchedulerFamily.UNIFORM_RANDOM_ACTIVATION)
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    stabilization: list[dict[str, Any]] = []

    for n in panel["sizes"]:
        for policy_name in panel["policies"]:
            policy = Policy(policy_name)
            for direction_name in panel["directions"]:
                direction = Direction(direction_name)
                for replicate in range(panel["replicatesPerCell"]):
                    seed = _seed(n, policy.value, direction.value, replicate)
                    budget = 100 * n * n
                    total_task_envelope = 2 * budget + 20 * n
                    generation_key = f"E05/S01/base/n{n}/{policy.value}/{direction.value}/r{replicate}"
                    source = create_scenario(
                        list(range(n)),
                        policy=policy,
                        direction=direction,
                        seed=seed,
                        max_activations=total_task_envelope,
                        generation_key=generation_key,
                    )
                    development = run_scheduled_architecture(
                        source, architecture, scheduler, trace_mode="full"
                    ).result
                    initial = initial_checkpoint(source)
                    if not development.summary["completed"]:
                        raise RuntimeError(
                            f"development baseline failed: {source.scenario_id} "
                            f"({development.summary['stopReason']})"
                        )
                    achieved_occupancy = tuple(development.summary["finalOccupancy"])
                    achieved_cursors = dict(development.final_state["selectionCursors"])
                    achieved_checkpoint = Checkpoint(
                        achieved_occupancy,
                        tuple(sorted(achieved_cursors.items())),
                        int(development.summary["activationCount"]),
                        tuple(
                            sorted(
                                (
                                    str(key),
                                    int(value),
                                )
                                for key, value in development.final_state[
                                    "streamCounters"
                                ].items()
                            )
                        ),
                        tuple(
                            (field, int(development.final_state["ledger"][field]))
                            for field in LEDGER_FIELDS
                        ),
                        0,
                        _checkpoint_hash(
                            source.scenario_id,
                            achieved_occupancy,
                            achieved_cursors,
                            int(development.summary["activationCount"]),
                            development.final_state["streamCounters"],
                            development.final_state["ledger"],
                        ),
                    )
                    partial = first_progress_checkpoint(source, development.events)
                    values_by_id = {cell.cell_id: cell.value for cell in source.cells}
                    target_sha256 = _hash(
                        list(
                            target_values(
                                [cell.value for cell in source.cells], direction
                            )
                        )
                    )
                    pairing_content = {
                        "schemaVersion": PAIRING_PROFILE,
                        "sourceScenarioId": source.scenario_id,
                        "n": n,
                        "policy": policy.value,
                        "direction": direction.value,
                        "replicateOrdinal": replicate,
                        "runtimeSeed": str(seed),
                        "initialOccupancySha256": _hash(list(source.initial_occupancy)),
                        "targetValuesSha256": target_sha256,
                        "eventBudget": budget,
                    }
                    pairing_id = _pairing_id(pairing_content)
                    injury_seed = _seed("injury", pairing_id)

                    development_row = _row(
                        pairing_id=pairing_id,
                        family=TaskFamily.DEVELOPMENT_RANDOM,
                        arm=Arm.DEVELOPMENT_BASELINE,
                        source=source,
                        executable=source,
                        replicate=replicate,
                        injury_seed=injury_seed,
                        initial_distance=initial.distance,
                        preinjury=None,
                        postinjury_occupancy=source.initial_occupancy,
                        primary_threshold=0,
                        target_sha256=target_sha256,
                        injury_operator="none",
                        injury_onset=None,
                        stabilization_rule=None,
                    )
                    rows.append(development_row)
                    results.append(
                        _validation_result(
                            development_row,
                            development,
                            primary_recovery_time=int(
                                development.summary["activationCount"]
                            ),
                            expected_initial_class="random_disorder",
                            final_distance=0,
                        )
                    )

                    stabilized_achieved, coverage = stabilize_achieved_checkpoint(
                        source,
                        achieved_checkpoint,
                    )
                    stabilization.append(
                        {
                            "pairingBlockId": pairing_id,
                            "sourceScenarioId": source.scenario_id,
                            **coverage,
                        }
                    )

                    for family, checkpoint in (
                        (TaskFamily.REPAIR_ACHIEVED, stabilized_achieved),
                        (TaskFamily.REPAIR_PARTIAL, partial),
                    ):
                        damaged, injury_index = apply_validation_injury(
                            checkpoint.occupancy, values_by_id, direction
                        )
                        for arm, postoccupancy, operator in (
                            (Arm.MATCHED_NO_INJURY, checkpoint.occupancy, "none"),
                            (
                                Arm.VALIDATION_INJURY,
                                damaged,
                                "s01_validation_adjacent_swap_v1",
                            ),
                        ):
                            threshold = (
                                0
                                if family == TaskFamily.REPAIR_ACHIEVED
                                else checkpoint.distance
                            )
                            run_result = run_recovery_phase(
                                source,
                                checkpoint,
                                postinjury_occupancy=postoccupancy,
                                primary_threshold=threshold,
                                recovery_budget=budget,
                            )
                            row = _row(
                                pairing_id=pairing_id,
                                family=family,
                                arm=arm,
                                source=source,
                                executable=source,
                                replicate=replicate,
                                injury_seed=injury_seed,
                                initial_distance=initial.distance,
                                preinjury=checkpoint,
                                postinjury_occupancy=postoccupancy,
                                primary_threshold=threshold,
                                target_sha256=target_sha256,
                                injury_operator=operator,
                                injury_onset=(
                                    "after_achieved_target_and_stabilization"
                                    if family == TaskFamily.REPAIR_ACHIEVED
                                    else "first_crossing_50_percent_strict_inversion_progress"
                                ),
                                stabilization_rule=(
                                    "target_absorbing_certificate_plus_uniform_two_per_identity_cap_20n"
                                    if family == TaskFamily.REPAIR_ACHIEVED
                                    else "not_applicable_atomic_active_checkpoint"
                                ),
                            )
                            rows.append(row)
                            final_distance = strict_unequal_inversions(
                                run_result.summary["finalValues"], direction
                            )
                            results.append(
                                _validation_result(
                                    row,
                                    run_result,
                                    primary_recovery_time=run_result.summary[
                                        "primaryRecoveryTime"
                                    ],
                                    expected_initial_class=family.value,
                                    final_distance=final_distance,
                                )
                            )
                        snapshots.append(
                            {
                                "schemaVersion": "e05.s01.checkpoint-snapshot.v1",
                                "pairingBlockId": pairing_id,
                                "taskFamily": family.value,
                                "checkpoint": checkpoint.to_dict(),
                                "injuryFixture": {
                                    "operator": "s01_validation_adjacent_swap_v1",
                                    "adjacentLeftIndex": injury_index,
                                    "preDistance": checkpoint.distance,
                                    "postDistance": checkpoint.distance + 1,
                                },
                            }
                        )

    for row in rows:
        jsonschema.Draft202012Validator(BASELINE_SCENARIO_SCHEMA).validate(row)
    pairing_validation = validate_pairing_rows(rows)
    validation_summary = _summarize_validation(
        rows, results, stabilization, pairing_validation
    )
    return {
        "taskSpec": dict(specification),
        "baselineScenarios": rows,
        "validationResults": results,
        "checkpointSnapshots": snapshots,
        "stabilizationValidation": stabilization,
        "pairingValidation": pairing_validation,
        "validationSummary": validation_summary,
    }


def _validation_result(
    row: Mapping[str, Any],
    run_result: Any,
    *,
    primary_recovery_time: int | None,
    expected_initial_class: str,
    final_distance: int,
) -> dict[str, Any]:
    within_budget = int(run_result.summary["activationCount"]) <= int(
        row["recoveryBudget"]
    )
    primary_success = primary_recovery_time is not None
    target_feasible = len(run_result.summary["finalValues"]) == int(row["n"]) and len(
        run_result.summary["finalOccupancy"]
    ) == int(row["n"])
    return {
        "schemaVersion": VALIDATION_SCHEMA_VERSION,
        "baselineScenarioId": row["baselineScenarioId"],
        "pairingBlockId": row["pairingBlockId"],
        "taskFamily": row["taskFamily"],
        "arm": row["arm"],
        "n": row["n"],
        "policy": row["policy"],
        "direction": row["direction"],
        "expectedInitialClass": expected_initial_class,
        "stopReason": run_result.summary["stopReason"],
        "activationCount": int(run_result.summary["activationCount"]),
        "completedFinalTarget": bool(run_result.summary["completed"]),
        "primaryRecoverySuccess": primary_success,
        "primaryRecoveryTime": primary_recovery_time,
        "finalDistance": final_distance,
        "withinEventBudget": within_budget,
        "targetFeasible": target_feasible,
        "taskDefinitionValid": primary_success
        and within_budget
        and final_distance == 0,
        "eventDigest": run_result.event_digest,
    }


def _summarize_validation(
    rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    stabilization: Sequence[Mapping[str, Any]],
    pairing_validation: Mapping[str, Any],
) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    for family in (item.value for item in TaskFamily):
        selected = [item for item in results if item["taskFamily"] == family]
        by_family[family] = {
            "runCount": len(selected),
            "completedFinalTargetCount": sum(
                item["completedFinalTarget"] for item in selected
            ),
            "primaryRecoverySuccessCount": sum(
                item["primaryRecoverySuccess"] for item in selected
            ),
            "withinEventBudgetCount": sum(
                item["withinEventBudget"] for item in selected
            ),
            "maximumActivationCount": max(item["activationCount"] for item in selected),
        }
    development_rows = [
        item
        for item in rows
        if item["taskFamily"] == TaskFamily.DEVELOPMENT_RANDOM.value
    ]
    achieved_rows = [
        item for item in rows if item["taskFamily"] == TaskFamily.REPAIR_ACHIEVED.value
    ]
    partial_rows = [
        item for item in rows if item["taskFamily"] == TaskFamily.REPAIR_PARTIAL.value
    ]
    active_rows = [item for item in rows if item["arm"] == Arm.VALIDATION_INJURY.value]
    sham_rows = [item for item in rows if item["arm"] == Arm.MATCHED_NO_INJURY.value]
    operationally_distinct = (
        all(
            item["preInjuryDistance"] is None
            and item["initialDistance"] > 0
            and item["injuryOnsetRule"] is None
            for item in development_rows
        )
        and all(item["preInjuryDistance"] == 0 for item in achieved_rows)
        and all(
            0 < item["preInjuryDistance"] < item["initialDistance"]
            and item["primaryRecoveryThreshold"] == item["preInjuryDistance"]
            for item in partial_rows
        )
    )
    injury_semantics_valid = all(
        item["postInjuryDistance"] == item["preInjuryDistance"] + 1
        for item in active_rows
    ) and all(
        item["postInjuryDistance"] == item["preInjuryDistance"] for item in sham_rows
    )
    active_results = [
        item for item in results if item["arm"] == Arm.VALIDATION_INJURY.value
    ]
    active_outcomes = {
        "runCount": len(active_results),
        "completedFinalTargetCount": sum(
            item["completedFinalTarget"] for item in active_results
        ),
        "primaryRecoverySuccessCount": sum(
            item["primaryRecoverySuccess"] for item in active_results
        ),
        "failureCount": sum(not item["taskDefinitionValid"] for item in active_results),
        "stopReasonCounts": dict(
            sorted(Counter(str(item["stopReason"]) for item in active_results).items())
        ),
    }
    checks = {
        "taskSchemaValid": True,
        "baselineScenarioSchemaValid": True,
        "allTargetsFeasible": all(item["targetFeasible"] for item in results),
        "allRunsWithinEventBudget": all(item["withinEventBudget"] for item in results),
        "allDevelopmentRunsComplete": all(
            item["completedFinalTarget"]
            for item in results
            if item["taskFamily"] == TaskFamily.DEVELOPMENT_RANDOM.value
        ),
        "allNoInjuryControlsValid": all(
            item["taskDefinitionValid"]
            for item in results
            if item["arm"] == Arm.MATCHED_NO_INJURY.value
        ),
        "allValidationOutcomesRecorded": len(results) == len(rows)
        and len(active_results) == pairing_validation["repairPairCount"],
        "allStabilizationCoveragePass": all(
            item["coveragePass"] and item["success"] for item in stabilization
        ),
        "allInjuryFixtureSemanticsValid": injury_semantics_valid,
        "pairingValidationPass": bool(pairing_validation["success"]),
        "taskFamiliesOperationallyDistinct": operationally_distinct,
    }
    return {
        "schemaVersion": "e05.s01.validation-summary.v1",
        "researchStepId": "S01",
        "benchmarkVersion": BENCHMARK_VERSION,
        "baselineScenarioCount": len(rows),
        "executedRunCount": len(results),
        "pairingBlockCount": len({item["pairingBlockId"] for item in rows}),
        "repairPairCount": pairing_validation["repairPairCount"],
        "stabilizationAuditCount": len(stabilization),
        "familyResults": by_family,
        "validationInjuryOutcomes": active_outcomes,
        "checks": checks,
        "success": all(checks.values()),
    }
