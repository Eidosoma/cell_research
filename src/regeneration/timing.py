"""E05 S02 temporal intervention design over the exact S01 task state.

The S01 adjacent swap is retained only as a calibrated +1-inversion timing
fixture.  Lesion diversity remains owned by S03.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import inspect
import math
from typing import Any, Mapping, Sequence

import jsonschema

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.schedulers import (
    SchedulerExecutionContract,
    SchedulerFamily,
    run_scheduled_architecture,
)
from reference_simulator.api import create_scenario
from reference_simulator.model import (
    Direction,
    Policy,
    Scenario,
    canonical_json_bytes,
    sha256_json,
)
from reference_simulator.policies import cell_view_proposal

from .tasks import (
    EVENT_BUDGET_PROFILE,
    PAIRING_PROFILE,
    Checkpoint,
    _hash,
    _seed,
    apply_validation_injury,
    initial_checkpoint,
    occupancy_values,
    replay_checkpoints,
    run_recovery_phase,
    stabilize_achieved_checkpoint,
    strict_unequal_inversions,
    target_values,
)


BENCHMARK_VERSION = "E05-damage-timing-v1"
TIMING_SPEC_SCHEMA_VERSION = "e05.s02.timing-spec.v1"
TIMING_SCENARIO_SCHEMA_VERSION = "e05.s02.damage-timing-scenario.v1"
TIMING_RESULT_SCHEMA_VERSION = "e05.s02.timing-result.v1"
MASTER_PANEL_PAIRING_PROFILE = PAIRING_PROFILE


class TimingClock(str, Enum):
    INITIALIZATION = "initialization"
    PROGRESS = "progress_first_crossing"
    EVENT = "paired_uninjured_completion_fraction"
    POST_COMPLETION = "post_completion_stabilized"


class TimingArm(str, Enum):
    SHAM = "matched_no_injury"
    INJURY = "validation_injury"


@dataclass(frozen=True, slots=True)
class Trigger:
    condition_id: str
    clock: TimingClock
    nominal_fraction: float
    status: str
    checkpoint: Checkpoint | None
    planned_event_index: int | None
    prior_event_index: int | None
    prior_distance: int | None
    prior_progress: float | None
    actual_progress: float | None
    reference_completion_events: int | None
    reason: str | None = None


TIMING_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s02/timing-spec.schema.json",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenQuestion",
        "inherits",
        "timingConditions",
        "unreachableTriggerRule",
        "eventClockReference",
        "lesionCalibration",
        "validationPanel",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": TIMING_SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S02"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "frozenQuestion": {"type": "string", "minLength": 20},
        "inherits": {"type": "object"},
        "timingConditions": {
            "type": "array",
            "minItems": 8,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "timingConditionId",
                    "clock",
                    "nominalFraction",
                    "triggerRule",
                ],
                "properties": {
                    "timingConditionId": {"type": "string"},
                    "clock": {"enum": [item.value for item in TimingClock]},
                    "nominalFraction": {"type": "number", "minimum": 0, "maximum": 1},
                    "triggerRule": {"type": "string", "minLength": 10},
                },
            },
        },
        "unreachableTriggerRule": {"type": "object"},
        "eventClockReference": {"type": "object"},
        "lesionCalibration": {"type": "object"},
        "validationPanel": {"type": "object"},
        "claimBoundary": {"type": "string", "minLength": 30},
    },
}


TIMING_SCENARIO_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s02/damage-timing-scenario.schema.json",
    "type": "object",
    "required": [
        "schemaVersion",
        "benchmarkVersion",
        "timingScenarioId",
        "timingPairId",
        "s01PairingBlockId",
        "executableScenarioId",
        "timingConditionId",
        "clock",
        "nominalFraction",
        "arm",
        "triggerStatus",
        "executionStatus",
        "n",
        "policy",
        "direction",
        "replicateOrdinal",
        "runtimeSeed",
        "injurySeed",
        "developmentBudget",
        "recoveryBudget",
        "eventBudgetProfile",
        "scheduler",
        "continuation",
        "mobility",
        "rngPairingStatus",
        "initialDistance",
        "preInjuryEventIndex",
        "preInjuryDistance",
        "preInjuryStateHash",
        "preInjuryOccupancyHash",
        "preInjuryInternalStateHash",
        "preInjuryPrefixDigest",
        "postInjuryDistance",
        "lesionSeverityDelta",
        "primaryRecoveryThreshold",
        "targetValuesSha256",
        "injuryOperator",
    ],
    "properties": {
        "schemaVersion": {"const": TIMING_SCENARIO_SCHEMA_VERSION},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "timingScenarioId": {"type": "string", "pattern": "^e05s02:[0-9a-f]{64}$"},
        "timingPairId": {"type": "string", "pattern": "^e05tp2:[0-9a-f]{64}$"},
        "s01PairingBlockId": {"type": "string", "pattern": "^e05pb1:[0-9a-f]{64}$"},
        "executableScenarioId": {"type": "string", "pattern": "^r1:[0-9a-f]{64}$"},
        "timingConditionId": {"type": "string"},
        "clock": {"enum": [item.value for item in TimingClock]},
        "nominalFraction": {"type": "number"},
        "arm": {"enum": [item.value for item in TimingArm]},
        "triggerStatus": {"type": "string"},
        "executionStatus": {"type": "string"},
        "n": {"type": "integer", "minimum": 2},
        "policy": {"enum": [item.value for item in Policy]},
        "direction": {"enum": [item.value for item in Direction]},
        "replicateOrdinal": {"type": "integer", "minimum": 0},
        "runtimeSeed": {"type": "string"},
        "injurySeed": {"type": "string"},
        "developmentBudget": {"type": "integer", "minimum": 1},
        "recoveryBudget": {"type": "integer", "minimum": 1},
        "eventBudgetProfile": {"const": EVENT_BUDGET_PROFILE},
        "scheduler": {"const": "uniform_random_activation"},
        "continuation": {"const": "skip_and_continue"},
        "mobility": {"const": "normal"},
        "rngPairingStatus": {"const": "shared_prefix_until_arm_terminal"},
        "initialDistance": {"type": "integer", "minimum": 1},
        "preInjuryEventIndex": {"type": ["integer", "null"], "minimum": 0},
        "preInjuryDistance": {"type": ["integer", "null"], "minimum": 0},
        "preInjuryStateHash": {"type": ["string", "null"]},
        "preInjuryOccupancyHash": {"type": ["string", "null"]},
        "preInjuryInternalStateHash": {"type": ["string", "null"]},
        "preInjuryPrefixDigest": {"type": ["string", "null"]},
        "postInjuryDistance": {"type": ["integer", "null"], "minimum": 0},
        "lesionSeverityDelta": {"type": ["integer", "null"]},
        "primaryRecoveryThreshold": {"type": ["integer", "null"], "minimum": 0},
        "targetValuesSha256": {"type": "string"},
        "injuryOperator": {"enum": ["none", "s01_validation_adjacent_swap_v1"]},
    },
}


def validate_timing_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(TIMING_SPEC_SCHEMA).validate(specification)
    conditions = specification["timingConditions"]
    identifiers = [item["timingConditionId"] for item in conditions]
    if len(set(identifiers)) != 8:
        raise ValueError("timing condition IDs must be unique")
    clocks = Counter(item["clock"] for item in conditions)
    expected = {
        TimingClock.INITIALIZATION.value: 1,
        TimingClock.PROGRESS.value: 3,
        TimingClock.EVENT.value: 3,
        TimingClock.POST_COMPLETION.value: 1,
    }
    if dict(clocks) != expected:
        raise ValueError(
            "timing specification must contain the frozen 1+3+3+1 clock panel"
        )
    for clock in (TimingClock.PROGRESS.value, TimingClock.EVENT.value):
        fractions = sorted(
            float(item["nominalFraction"])
            for item in conditions
            if item["clock"] == clock
        )
        if fractions != [0.25, 0.5, 0.75]:
            raise ValueError(f"{clock} must use 25%, 50%, and 75%")
    inherited = specification["inherits"]
    required_inherited = {
        "eventBudgetProfile": EVENT_BUDGET_PROFILE,
        "pairingProfile": PAIRING_PROFILE,
        "rngStatus": "shared_prefix_until_arm_terminal",
    }
    for key, expected_value in required_inherited.items():
        if inherited.get(key) != expected_value:
            raise ValueError(f"S01 inherited contract changed: {key}")


def _progress(initial_distance: int, distance: int) -> float:
    if initial_distance <= 0:
        raise ValueError("progress clock requires positive initial distance")
    return (initial_distance - distance) / initial_distance


def locate_trigger(
    condition: Mapping[str, Any],
    *,
    initial: Checkpoint,
    development_checkpoints: Sequence[Checkpoint],
    development_stop_reason: str,
    completion_events: int | None,
    stabilized: Checkpoint | None,
) -> Trigger:
    """Locate one trigger without fallback or threshold substitution."""

    condition_id = str(condition["timingConditionId"])
    clock = TimingClock(condition["clock"])
    fraction = float(condition["nominalFraction"])
    if clock == TimingClock.INITIALIZATION:
        return Trigger(
            condition_id,
            clock,
            fraction,
            "triggered",
            initial,
            0,
            None,
            None,
            None,
            0.0,
            completion_events,
        )
    if clock == TimingClock.PROGRESS:
        previous = initial
        for checkpoint in development_checkpoints:
            actual = _progress(initial.distance, checkpoint.distance)
            if actual >= fraction:
                return Trigger(
                    condition_id,
                    clock,
                    fraction,
                    "triggered",
                    checkpoint,
                    None,
                    previous.activation_count,
                    previous.distance,
                    _progress(initial.distance, previous.distance),
                    actual,
                    completion_events,
                )
            previous = checkpoint
        return Trigger(
            condition_id,
            clock,
            fraction,
            "unreached_terminal_before_progress_threshold",
            None,
            None,
            previous.activation_count,
            previous.distance,
            _progress(initial.distance, previous.distance),
            None,
            completion_events,
            f"development terminated as {development_stop_reason} before first crossing",
        )
    if clock == TimingClock.EVENT:
        if completion_events is None:
            return Trigger(
                condition_id,
                clock,
                fraction,
                "unreached_no_paired_completion_reference",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "paired uninjured development did not complete within its phase budget",
            )
        planned = int(math.ceil(fraction * completion_events))
        if planned < 1 or planned > len(development_checkpoints):
            raise AssertionError(
                "event trigger falls outside paired development checkpoints"
            )
        checkpoint = development_checkpoints[planned - 1]
        previous = initial if planned == 1 else development_checkpoints[planned - 2]
        return Trigger(
            condition_id,
            clock,
            fraction,
            "triggered",
            checkpoint,
            planned,
            previous.activation_count,
            previous.distance,
            _progress(initial.distance, previous.distance),
            _progress(initial.distance, checkpoint.distance),
            completion_events,
        )
    if stabilized is None:
        return Trigger(
            condition_id,
            clock,
            fraction,
            "unreached_no_stabilized_completion",
            None,
            None,
            None,
            None,
            None,
            None,
            completion_events,
            "paired development did not produce a valid stabilized completion",
        )
    return Trigger(
        condition_id,
        clock,
        fraction,
        "triggered",
        stabilized,
        stabilized.activation_count,
        completion_events,
        0,
        1.0,
        1.0,
        completion_events,
    )


def _prefix_digest(events: Sequence[Mapping[str, Any]], event_count: int) -> str:
    digest = hashlib.sha256()
    for event in events[:event_count]:
        digest.update(canonical_json_bytes(event))
    return digest.hexdigest()


def _source_scenario(
    n: int, policy: Policy, direction: Direction, replicate: int
) -> tuple[Scenario, int, int]:
    """Recreate the exact S01 source scenario and task envelope."""

    seed = _seed(n, policy.value, direction.value, replicate)
    budget = 100 * n * n
    scenario = create_scenario(
        list(range(n)),
        policy=policy,
        direction=direction,
        seed=seed,
        max_activations=2 * budget + 20 * n,
        generation_key=f"E05/S01/base/n{n}/{policy.value}/{direction.value}/r{replicate}",
    )
    return scenario, seed, budget


def _s01_pairing_id(
    scenario: Scenario,
    *,
    n: int,
    policy: Policy,
    direction: Direction,
    replicate: int,
    budget: int,
) -> str:
    target_hash = _hash(
        list(target_values([cell.value for cell in scenario.cells], direction))
    )
    return "e05pb1:" + sha256_json(
        {
            "schemaVersion": PAIRING_PROFILE,
            "sourceScenarioId": scenario.scenario_id,
            "n": n,
            "policy": policy.value,
            "direction": direction.value,
            "replicateOrdinal": replicate,
            "runtimeSeed": str(scenario.seed),
            "initialOccupancySha256": _hash(list(scenario.initial_occupancy)),
            "targetValuesSha256": target_hash,
            "eventBudget": budget,
        }
    )


def _run_signature(result: Any) -> bytes:
    return canonical_json_bytes(
        {
            "summary": result.summary,
            "finalState": result.final_state,
            "eventDigest": result.event_digest,
            "events": list(result.events),
        }
    )


def _recovery_distance_audit(
    scenario: Scenario,
    start_occupancy: Sequence[str],
    events: Sequence[Mapping[str, Any]],
) -> tuple[int, float]:
    occupancy = list(start_occupancy)
    start_distance = strict_unequal_inversions(
        occupancy_values(scenario, occupancy), scenario.cells[0].direction
    )
    minimum = start_distance
    for event in events:
        proposal = event["proposal"]
        if event["decision"] == "accepted" and proposal["kind"] == "Swap":
            left = int(proposal["actorPos"])
            right = int(proposal["targetPos"])
            occupancy[left], occupancy[right] = occupancy[right], occupancy[left]
        minimum = min(
            minimum,
            strict_unequal_inversions(
                occupancy_values(scenario, occupancy), scenario.cells[0].direction
            ),
        )
    maximum_progress = (
        1.0 if start_distance == 0 else (start_distance - minimum) / start_distance
    )
    return minimum, maximum_progress


def _scenario_row(
    *,
    scenario: Scenario,
    pairing_id: str,
    trigger: Trigger,
    condition: Mapping[str, Any],
    arm: TimingArm,
    replicate: int,
    injury_seed: int,
    budget: int,
    target_hash: str,
    development_events: Sequence[Mapping[str, Any]],
    postinjury: Sequence[str] | None,
    lesion_delta: int | None,
    primary_threshold: int | None,
    injury_operator: str,
    execution_status: str,
) -> dict[str, Any]:
    checkpoint = trigger.checkpoint
    timing_pair_id = "e05tp2:" + sha256_json(
        {
            "s01PairingBlockId": pairing_id,
            "timingConditionId": trigger.condition_id,
            "preInjuryStateHash": None if checkpoint is None else checkpoint.state_hash,
        }
    )
    post_distance = None
    if checkpoint is not None and postinjury is not None:
        post_distance = strict_unequal_inversions(
            occupancy_values(scenario, postinjury), scenario.cells[0].direction
        )
    base = {
        "timingPairId": timing_pair_id,
        "timingConditionId": trigger.condition_id,
        "arm": arm.value,
        "postInjuryDistance": post_distance,
        "injuryOperator": injury_operator,
    }
    checkpoint_event = None if checkpoint is None else checkpoint.activation_count
    row = {
        "schemaVersion": TIMING_SCENARIO_SCHEMA_VERSION,
        "benchmarkVersion": BENCHMARK_VERSION,
        "timingScenarioId": "e05s02:" + sha256_json(base),
        "timingPairId": timing_pair_id,
        "s01PairingBlockId": pairing_id,
        "executableScenarioId": scenario.scenario_id,
        "timingConditionId": trigger.condition_id,
        "clock": trigger.clock.value,
        "nominalFraction": trigger.nominal_fraction,
        "arm": arm.value,
        "triggerRule": condition["triggerRule"],
        "triggerStatus": trigger.status,
        "triggerReason": trigger.reason,
        "executionStatus": execution_status,
        "n": len(scenario.cells),
        "policy": scenario.cells[0].policy.value,
        "direction": scenario.cells[0].direction.value,
        "replicateOrdinal": replicate,
        "runtimeSeed": str(scenario.seed),
        "injurySeed": str(injury_seed),
        "developmentBudget": budget,
        "recoveryBudget": budget,
        "stabilizationOpportunityCap": 20 * len(scenario.cells),
        "eventBudgetProfile": EVENT_BUDGET_PROFILE,
        "scheduler": "uniform_random_activation",
        "continuation": "skip_and_continue",
        "mobility": "normal",
        "rngPairingStatus": "shared_prefix_until_arm_terminal",
        "initialDistance": initial_checkpoint(scenario).distance,
        "referenceCompletionEvents": trigger.reference_completion_events,
        "plannedTriggerEventIndex": trigger.planned_event_index,
        "preInjuryEventIndex": checkpoint_event,
        "priorEventIndex": trigger.prior_event_index,
        "priorDistance": trigger.prior_distance,
        "priorProgress": trigger.prior_progress,
        "actualProgress": trigger.actual_progress,
        "preInjuryDistance": None if checkpoint is None else checkpoint.distance,
        "preInjuryStateHash": None if checkpoint is None else checkpoint.state_hash,
        "preInjuryOccupancyHash": None
        if checkpoint is None
        else _hash(list(checkpoint.occupancy)),
        "preInjuryInternalStateHash": None
        if checkpoint is None
        else _hash(
            {
                "selectionCursors": dict(checkpoint.selection_cursors),
                "streamCounters": dict(checkpoint.stream_counters),
                "ledger": dict(checkpoint.ledger),
            }
        ),
        "preInjuryPrefixDigest": None
        if checkpoint is None
        else _prefix_digest(
            development_events, min(checkpoint_event or 0, len(development_events))
        ),
        "postInjuryDistance": post_distance,
        "lesionSeverityDelta": lesion_delta,
        "primaryRecoveryThreshold": primary_threshold,
        "targetValuesSha256": target_hash,
        "injuryOperator": injury_operator,
        "injuryAdjacentLeftIndex": None,
    }
    return row


def validate_timing_pairing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["timingPairId"]), []).append(row)
    failures: list[str] = []
    shared = (
        "s01PairingBlockId",
        "executableScenarioId",
        "timingConditionId",
        "clock",
        "nominalFraction",
        "triggerStatus",
        "n",
        "policy",
        "direction",
        "replicateOrdinal",
        "runtimeSeed",
        "injurySeed",
        "developmentBudget",
        "recoveryBudget",
        "eventBudgetProfile",
        "preInjuryEventIndex",
        "preInjuryDistance",
        "preInjuryStateHash",
        "preInjuryOccupancyHash",
        "preInjuryInternalStateHash",
        "preInjuryPrefixDigest",
        "targetValuesSha256",
        "rngPairingStatus",
    )
    for pair_id, items in sorted(grouped.items()):
        by_arm = {str(item["arm"]): item for item in items}
        if set(by_arm) != {item.value for item in TimingArm} or len(items) != 2:
            failures.append(f"{pair_id}: missing or duplicate paired arms")
            continue
        left = by_arm[TimingArm.SHAM.value]
        right = by_arm[TimingArm.INJURY.value]
        for field in shared:
            if left[field] != right[field]:
                failures.append(f"{pair_id}: shared field differs: {field}")
    identifiers = [str(row["timingScenarioId"]) for row in rows]
    return {
        "schemaVersion": "e05.s02.pairing-validation.v1",
        "pairingProfile": MASTER_PANEL_PAIRING_PROFILE,
        "rngPairingStatus": "shared_prefix_until_arm_terminal",
        "rowCount": len(rows),
        "pairCount": len(grouped),
        "uniqueScenarioRows": len(identifiers) == len(set(identifiers)),
        "failureCount": len(failures),
        "failures": failures,
        "success": not failures and len(identifiers) == len(set(identifiers)),
    }


def build_timing_panel(
    specification: Mapping[str, Any],
    *,
    exact_replay: bool = False,
    expected_s01_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the S02 timing panel with complete planned-row accounting."""

    validate_timing_spec(specification)
    panel = specification["validationPanel"]
    architecture = ArchitectureExecutionContract.distributed_local()
    scheduler = SchedulerExecutionContract(SchedulerFamily.UNIFORM_RANDOM_ACTIVATION)
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    trigger_audits: list[dict[str, Any]] = []
    preinjury_records: list[dict[str, Any]] = []
    stabilization_audits: list[dict[str, Any]] = []
    formation_replays = 0
    recovery_replays = 0
    formation_replay_failures: list[str] = []
    recovery_replay_failures: list[str] = []
    source_ids: list[dict[str, Any]] = []

    for n in panel["sizes"]:
        for policy_name in panel["policies"]:
            policy = Policy(policy_name)
            for direction_name in panel["directions"]:
                direction = Direction(direction_name)
                for replicate in range(panel["replicatesPerCell"]):
                    scenario, seed, budget = _source_scenario(
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
                    source_ids.append(
                        {
                            "s01PairingBlockId": pairing_id,
                            "executableScenarioId": scenario.scenario_id,
                        }
                    )
                    development = run_scheduled_architecture(
                        scenario, architecture, scheduler, trace_mode="full"
                    ).result
                    if exact_replay:
                        replayed = run_scheduled_architecture(
                            scenario, architecture, scheduler, trace_mode="full"
                        ).result
                        formation_replays += 1
                        if development.to_json_bytes() != replayed.to_json_bytes():
                            formation_replay_failures.append(scenario.scenario_id)
                    phase_events = tuple(development.events[:budget])
                    checkpoints = replay_checkpoints(scenario, phase_events)
                    initial = initial_checkpoint(scenario)
                    completed_within_budget = (
                        bool(development.summary["completed"])
                        and int(development.summary["activationCount"]) <= budget
                    )
                    completion_events = (
                        int(development.summary["activationCount"])
                        if completed_within_budget
                        else None
                    )
                    development_stop = (
                        str(development.summary["stopReason"])
                        if int(development.summary["activationCount"]) <= budget
                        else "phase_event_budget"
                    )
                    stabilized: Checkpoint | None = None
                    stabilization_audit: Mapping[str, Any] | None = None
                    if completed_within_budget:
                        achieved = checkpoints[completion_events - 1]
                        stabilized, stabilization_audit = stabilize_achieved_checkpoint(
                            scenario, achieved
                        )
                        stabilization_audits.append(
                            {
                                "s01PairingBlockId": pairing_id,
                                "executableScenarioId": scenario.scenario_id,
                                **stabilization_audit,
                            }
                        )
                        if not stabilization_audit["success"]:
                            stabilized = None
                    target_hash = _hash(
                        list(
                            target_values(
                                [cell.value for cell in scenario.cells], direction
                            )
                        )
                    )
                    injury_seed = _seed("injury", pairing_id)
                    values_by_id = {cell.cell_id: cell.value for cell in scenario.cells}

                    for condition in specification["timingConditions"]:
                        trigger = locate_trigger(
                            condition,
                            initial=initial,
                            development_checkpoints=checkpoints,
                            development_stop_reason=development_stop,
                            completion_events=completion_events,
                            stabilized=stabilized,
                        )
                        accuracy = False
                        if (
                            trigger.status == "triggered"
                            and trigger.checkpoint is not None
                        ):
                            if trigger.clock == TimingClock.INITIALIZATION:
                                accuracy = trigger.checkpoint.activation_count == 0
                            elif trigger.clock == TimingClock.PROGRESS:
                                accuracy = bool(
                                    trigger.prior_progress is not None
                                    and trigger.prior_progress
                                    < trigger.nominal_fraction
                                    and trigger.actual_progress is not None
                                    and trigger.actual_progress
                                    >= trigger.nominal_fraction
                                )
                            elif trigger.clock == TimingClock.EVENT:
                                accuracy = bool(
                                    trigger.planned_event_index
                                    == trigger.checkpoint.activation_count
                                    == math.ceil(
                                        trigger.nominal_fraction
                                        * (trigger.reference_completion_events or 0)
                                    )
                                )
                            else:
                                accuracy = bool(
                                    stabilization_audit
                                    and stabilization_audit["success"]
                                    and trigger.checkpoint.distance == 0
                                    and trigger.checkpoint.activation_count
                                    == stabilization_audit["endEventIndex"]
                                )
                        else:
                            accuracy = trigger.status.startswith("unreached_")
                        trigger_audits.append(
                            {
                                "s01PairingBlockId": pairing_id,
                                "executableScenarioId": scenario.scenario_id,
                                "timingConditionId": trigger.condition_id,
                                "clock": trigger.clock.value,
                                "nominalFraction": trigger.nominal_fraction,
                                "triggerStatus": trigger.status,
                                "plannedEventIndex": trigger.planned_event_index,
                                "actualEventIndex": None
                                if trigger.checkpoint is None
                                else trigger.checkpoint.activation_count,
                                "priorProgress": trigger.prior_progress,
                                "actualProgress": trigger.actual_progress,
                                "referenceCompletionEvents": completion_events,
                                "triggerAccuracyPass": accuracy,
                                "unreachableRetained": trigger.status != "triggered",
                                "substitutionUsed": False,
                            }
                        )
                        if trigger.checkpoint is None:
                            for arm in TimingArm:
                                row = _scenario_row(
                                    scenario=scenario,
                                    pairing_id=pairing_id,
                                    trigger=trigger,
                                    condition=condition,
                                    arm=arm,
                                    replicate=replicate,
                                    injury_seed=injury_seed,
                                    budget=budget,
                                    target_hash=target_hash,
                                    development_events=phase_events,
                                    postinjury=None,
                                    lesion_delta=None,
                                    primary_threshold=None,
                                    injury_operator="none"
                                    if arm == TimingArm.SHAM
                                    else "s01_validation_adjacent_swap_v1",
                                    execution_status="not_run_trigger_unreached",
                                )
                                rows.append(row)
                                results.append(
                                    {
                                        "schemaVersion": TIMING_RESULT_SCHEMA_VERSION,
                                        "timingScenarioId": row["timingScenarioId"],
                                        "timingPairId": row["timingPairId"],
                                        "timingConditionId": row["timingConditionId"],
                                        "clock": row["clock"],
                                        "nominalFraction": row["nominalFraction"],
                                        "arm": row["arm"],
                                        "n": n,
                                        "policy": policy.value,
                                        "direction": direction.value,
                                        "triggerStatus": trigger.status,
                                        "executionStatus": "not_run_trigger_unreached",
                                        "stopReason": None,
                                        "activationCount": None,
                                        "completedFinalTarget": None,
                                        "primaryRecoverySuccess": None,
                                        "primaryRecoveryTime": None,
                                        "finalDistance": None,
                                        "minimumRecoveryDistance": None,
                                        "maximumRecoveryProgress": None,
                                        "eventDigest": None,
                                    }
                                )
                            continue

                        checkpoint = trigger.checkpoint
                        try:
                            damaged, injury_index = apply_validation_injury(
                                checkpoint.occupancy, values_by_id, direction
                            )
                        except ValueError as exc:
                            raise RuntimeError(
                                f"calibration lesion unavailable at {pairing_id}/{trigger.condition_id}"
                            ) from exc
                        primary_threshold = (
                            0
                            if trigger.clock
                            in {TimingClock.INITIALIZATION, TimingClock.POST_COMPLETION}
                            else checkpoint.distance
                        )
                        preinjury_records.append(
                            {
                                "schemaVersion": "e05.s02.preinjury-state.v1",
                                "s01PairingBlockId": pairing_id,
                                "timingConditionId": trigger.condition_id,
                                "clock": trigger.clock.value,
                                "nominalFraction": trigger.nominal_fraction,
                                "activationCount": checkpoint.activation_count,
                                "distance": checkpoint.distance,
                                "stateHash": checkpoint.state_hash,
                                "occupancyHash": _hash(list(checkpoint.occupancy)),
                                "internalStateHash": _hash(
                                    {
                                        "selectionCursors": dict(
                                            checkpoint.selection_cursors
                                        ),
                                        "streamCounters": dict(
                                            checkpoint.stream_counters
                                        ),
                                        "ledger": dict(checkpoint.ledger),
                                    }
                                ),
                                "prefixDigest": _prefix_digest(
                                    phase_events,
                                    min(checkpoint.activation_count, len(phase_events)),
                                ),
                                "selectionCursors": dict(checkpoint.selection_cursors),
                                "streamCounters": dict(checkpoint.stream_counters),
                                "ledger": dict(checkpoint.ledger),
                            }
                        )
                        for arm, occupancy, operator, lesion_delta in (
                            (TimingArm.SHAM, checkpoint.occupancy, "none", 0),
                            (
                                TimingArm.INJURY,
                                damaged,
                                "s01_validation_adjacent_swap_v1",
                                1,
                            ),
                        ):
                            run_result = run_recovery_phase(
                                scenario,
                                checkpoint,
                                postinjury_occupancy=occupancy,
                                primary_threshold=primary_threshold,
                                recovery_budget=budget,
                            )
                            if exact_replay:
                                replayed_recovery = run_recovery_phase(
                                    scenario,
                                    checkpoint,
                                    postinjury_occupancy=occupancy,
                                    primary_threshold=primary_threshold,
                                    recovery_budget=budget,
                                )
                                recovery_replays += 1
                                if _run_signature(run_result) != _run_signature(
                                    replayed_recovery
                                ):
                                    recovery_replay_failures.append(
                                        f"{pairing_id}/{trigger.condition_id}/{arm.value}"
                                    )
                            row = _scenario_row(
                                scenario=scenario,
                                pairing_id=pairing_id,
                                trigger=trigger,
                                condition=condition,
                                arm=arm,
                                replicate=replicate,
                                injury_seed=injury_seed,
                                budget=budget,
                                target_hash=target_hash,
                                development_events=phase_events,
                                postinjury=occupancy,
                                lesion_delta=lesion_delta,
                                primary_threshold=primary_threshold,
                                injury_operator=operator,
                                execution_status="executed",
                            )
                            row["injuryAdjacentLeftIndex"] = (
                                None if arm == TimingArm.SHAM else injury_index
                            )
                            rows.append(row)
                            minimum_distance, maximum_progress = (
                                _recovery_distance_audit(
                                    scenario, occupancy, run_result.events
                                )
                            )
                            final_distance = strict_unequal_inversions(
                                run_result.summary["finalValues"], direction
                            )
                            results.append(
                                {
                                    "schemaVersion": TIMING_RESULT_SCHEMA_VERSION,
                                    "timingScenarioId": row["timingScenarioId"],
                                    "timingPairId": row["timingPairId"],
                                    "timingConditionId": row["timingConditionId"],
                                    "clock": row["clock"],
                                    "nominalFraction": row["nominalFraction"],
                                    "arm": row["arm"],
                                    "n": n,
                                    "policy": policy.value,
                                    "direction": direction.value,
                                    "triggerStatus": trigger.status,
                                    "executionStatus": "executed",
                                    "stopReason": run_result.summary["stopReason"],
                                    "activationCount": int(
                                        run_result.summary["activationCount"]
                                    ),
                                    "completedFinalTarget": bool(
                                        run_result.summary["completed"]
                                    ),
                                    "primaryRecoverySuccess": run_result.summary[
                                        "primaryRecoveryTime"
                                    ]
                                    is not None,
                                    "primaryRecoveryTime": run_result.summary[
                                        "primaryRecoveryTime"
                                    ],
                                    "finalDistance": final_distance,
                                    "minimumRecoveryDistance": minimum_distance,
                                    "maximumRecoveryProgress": maximum_progress,
                                    "globalStartEventIndex": run_result.summary[
                                        "globalStartEventIndex"
                                    ],
                                    "globalEndEventIndex": run_result.summary[
                                        "globalEndEventIndex"
                                    ],
                                    "eventDigest": run_result.event_digest,
                                }
                            )

    for row in rows:
        jsonschema.Draft202012Validator(TIMING_SCENARIO_SCHEMA).validate(row)
    pairing_validation = validate_timing_pairing(rows)
    source_validation = _validate_s01_source_ids(
        source_ids,
        None if expected_s01_contract is None else expected_s01_contract.get("sources"),
    )
    checkpoint_identity_validation = _validate_s01_checkpoint_identity(
        preinjury_records,
        None
        if expected_s01_contract is None
        else expected_s01_contract.get("checkpoints"),
    )
    trigger_validation = {
        "schemaVersion": "e05.s02.trigger-validation.v1",
        "auditCount": len(trigger_audits),
        "triggeredCount": sum(
            item["triggerStatus"] == "triggered" for item in trigger_audits
        ),
        "unreachableCount": sum(
            item["triggerStatus"] != "triggered" for item in trigger_audits
        ),
        "accuracyPassCount": sum(
            item["triggerAccuracyPass"] for item in trigger_audits
        ),
        "substitutionCount": sum(item["substitutionUsed"] for item in trigger_audits),
        "success": all(item["triggerAccuracyPass"] for item in trigger_audits)
        and not any(item["substitutionUsed"] for item in trigger_audits),
    }
    severity_validation = _validate_severity(rows)
    leakage_validation = _validate_leakage(rows)
    replay_validation = {
        "schemaVersion": "e05.s02.replay-validation.v1",
        "exactReplayRequested": exact_replay,
        "coverageStatus": "full" if exact_replay else "skipped_for_smoke",
        "formationReplayCount": formation_replays,
        "recoveryReplayCount": recovery_replays,
        "formationFailures": formation_replay_failures,
        "recoveryFailures": recovery_replay_failures,
        "success": exact_replay
        and not formation_replay_failures
        and not recovery_replay_failures,
    }
    accounting = _run_accounting(rows, results, source_ids)
    unreachable_validation = _unreachable_policy_validation(rows, results)
    checks = {
        "timingSpecValid": True,
        "scenarioSchemaValid": True,
        "s01SourceIdentityPreserved": source_validation["success"],
        "s01CheckpointStatePreserved": checkpoint_identity_validation["success"],
        "triggerAccuracyPass": trigger_validation["success"],
        "preInjuryStateComplete": _validate_preinjury_records(rows, preinjury_records),
        "timingLeakagePass": leakage_validation["success"],
        "lesionSeverityComparable": severity_validation["success"],
        "pairingPass": pairing_validation["success"],
        "replayPass": replay_validation["success"],
        "completeRunAccounting": accounting["success"],
        "unreachablePolicyHandlingPass": unreachable_validation["success"],
        "stabilizationPass": all(
            item["success"] and item["coveragePass"] for item in stabilization_audits
        ),
    }
    return {
        "timingSpec": dict(specification),
        "scenarioRows": rows,
        "resultRows": results,
        "triggerAudits": trigger_audits,
        "preinjuryRecords": preinjury_records,
        "stabilizationAudits": stabilization_audits,
        "pairingValidation": pairing_validation,
        "sourceIdentityValidation": source_validation,
        "checkpointIdentityValidation": checkpoint_identity_validation,
        "triggerValidation": trigger_validation,
        "severityValidation": severity_validation,
        "leakageValidation": leakage_validation,
        "replayValidation": replay_validation,
        "runAccounting": accounting,
        "unreachablePolicyValidation": unreachable_validation,
        "validationSummary": {
            "schemaVersion": "e05.s02.validation-summary.v1",
            "researchStepId": "S02",
            "benchmarkVersion": BENCHMARK_VERSION,
            "checks": checks,
            "success": all(checks.values()),
        },
    }


def _validate_s01_source_ids(
    source_ids: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    observed_set = {
        (item["s01PairingBlockId"], item["executableScenarioId"]) for item in source_ids
    }
    expected_set = (
        None
        if expected is None
        else {
            (item["s01PairingBlockId"], item["executableScenarioId"])
            for item in expected
        }
    )
    exact_match = expected_set is None or observed_set == expected_set
    return {
        "schemaVersion": "e05.s02.s01-source-identity-validation.v1",
        "sourceCount": len(source_ids),
        "uniquePairingBlocks": len({item["s01PairingBlockId"] for item in source_ids}),
        "uniqueExecutableScenarios": len(
            {item["executableScenarioId"] for item in source_ids}
        ),
        "artifactComparisonPerformed": expected_set is not None,
        "expectedSourceCount": None if expected_set is None else len(expected_set),
        "exactArtifactIdentityMatch": exact_match,
        "success": len(source_ids)
        == len({item["s01PairingBlockId"] for item in source_ids})
        == len({item["executableScenarioId"] for item in source_ids})
        and exact_match,
    }


def _validate_s01_checkpoint_identity(
    records: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    observed = {
        (
            item["s01PairingBlockId"],
            "repair_partial"
            if item["timingConditionId"] == "progress_50"
            else "repair_achieved",
        ): item["stateHash"]
        for item in records
        if item["timingConditionId"] in {"progress_50", "post_completion"}
    }
    expected_map = (
        None
        if expected is None
        else {
            (item["s01PairingBlockId"], item["taskFamily"]): item["stateHash"]
            for item in expected
        }
    )
    exact_match = expected_map is None or observed == expected_map
    return {
        "schemaVersion": "e05.s02.s01-checkpoint-identity-validation.v1",
        "observedCheckpointCount": len(observed),
        "artifactComparisonPerformed": expected_map is not None,
        "expectedCheckpointCount": None if expected_map is None else len(expected_map),
        "exactStateHashMatch": exact_match,
        "success": exact_match and len(observed) == len({key for key in observed}),
    }


def _validate_severity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    triggered = [item for item in rows if item["triggerStatus"] == "triggered"]
    active = [item for item in triggered if item["arm"] == TimingArm.INJURY.value]
    sham = [item for item in triggered if item["arm"] == TimingArm.SHAM.value]
    active_pass = all(
        item["lesionSeverityDelta"] == 1
        and item["postInjuryDistance"] == item["preInjuryDistance"] + 1
        for item in active
    )
    sham_pass = all(
        item["lesionSeverityDelta"] == 0
        and item["postInjuryDistance"] == item["preInjuryDistance"]
        for item in sham
    )
    return {
        "schemaVersion": "e05.s02.severity-validation.v1",
        "activeTriggeredCount": len(active),
        "shamTriggeredCount": len(sham),
        "activeDeltaOneCount": sum(item["lesionSeverityDelta"] == 1 for item in active),
        "shamDeltaZeroCount": sum(item["lesionSeverityDelta"] == 0 for item in sham),
        "activePass": active_pass,
        "shamPass": sham_pass,
        "success": active_pass and sham_pass and len(active) == len(sham),
    }


def _validate_leakage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairing = validate_timing_pairing(rows)
    signature = inspect.signature(cell_view_proposal)
    policy_parameters = list(signature.parameters)
    forbidden = {
        "timingConditionId",
        "clock",
        "nominalFraction",
        "plannedTriggerEventIndex",
        "futureInjuryTime",
    }
    source = inspect.getsource(cell_view_proposal)
    forbidden_source_hits = sorted(item for item in forbidden if item in source)
    return {
        "schemaVersion": "e05.s02.timing-leakage-validation.v1",
        "policyParameters": policy_parameters,
        "forbiddenTimingParametersPresent": sorted(forbidden & set(policy_parameters)),
        "forbiddenTimingSourceHits": forbidden_source_hits,
        "pairedPreInjuryHashesAndPrefixesExact": pairing["success"],
        "eventClockPolicyVisibility": "not passed to scenario, scheduler, state, or policy proposal",
        "success": pairing["success"]
        and not (forbidden & set(policy_parameters))
        and not forbidden_source_hits,
    }


def _validate_preinjury_records(
    rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> bool:
    triggered_pairs = {
        (item["s01PairingBlockId"], item["timingConditionId"])
        for item in rows
        if item["triggerStatus"] == "triggered"
    }
    record_pairs = {
        (item["s01PairingBlockId"], item["timingConditionId"]) for item in records
    }
    required = (
        "stateHash",
        "occupancyHash",
        "internalStateHash",
        "prefixDigest",
        "selectionCursors",
        "streamCounters",
        "ledger",
    )
    return triggered_pairs == record_pairs and all(
        all(item.get(field) is not None for field in required) for item in records
    )


def _run_accounting(
    rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    source_ids: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    planned = len(source_ids) * 8 * 2
    executed = sum(item["executionStatus"] == "executed" for item in rows)
    not_run = sum(item["executionStatus"] != "executed" for item in rows)
    result_executed = sum(item["executionStatus"] == "executed" for item in results)
    return {
        "schemaVersion": "e05.s02.run-accounting.v1",
        "formationRunCount": len(source_ids),
        "plannedTimingArmRows": planned,
        "recordedTimingArmRows": len(rows),
        "resultRows": len(results),
        "executedTimingRuns": executed,
        "notRunTriggerUnreachedRows": not_run,
        "resultExecutedRows": result_executed,
        "silentExclusionCount": planned - len(rows),
        "duplicateScenarioRowCount": len(rows)
        - len({item["timingScenarioId"] for item in rows}),
        "success": planned == len(rows) == len(results)
        and executed == result_executed
        and planned - len(rows) == 0,
    }


def _unreachable_policy_validation(
    rows: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Audit real preserved-state Selection failures plus planned onset rows."""

    by_id = {item["timingScenarioId"]: item for item in rows}
    preserved_selection_failures = []
    for result in results:
        row = by_id[result["timingScenarioId"]]
        if (
            result["executionStatus"] == "executed"
            and result["arm"] == TimingArm.INJURY.value
            and result["policy"] == Policy.SELECTION.value
            and result["timingConditionId"] == "post_completion"
            and not result["completedFinalTarget"]
        ):
            reached = [
                fraction
                for fraction in (0.25, 0.5, 0.75)
                if float(result["maximumRecoveryProgress"] or 0.0) >= fraction
            ]
            preserved_selection_failures.append(
                {
                    "timingScenarioId": result["timingScenarioId"],
                    "stopReason": result["stopReason"],
                    "preInjuryStateHash": row["preInjuryStateHash"],
                    "maximumRecoveryProgress": result["maximumRecoveryProgress"],
                    "requestedThresholds": [0.25, 0.5, 0.75],
                    "reachedThresholds": reached,
                    "unreachedThresholds": [
                        fraction
                        for fraction in (0.25, 0.5, 0.75)
                        if fraction not in reached
                    ],
                    "handling": "retained_terminal_outcome_no_substitution",
                }
            )
    planned_unreachable = [
        item for item in rows if item["triggerStatus"] != "triggered"
    ]
    planned_pair_complete = len(planned_unreachable) % 2 == 0 and all(
        sum(
            other["timingPairId"] == item["timingPairId"]
            for other in planned_unreachable
        )
        == 2
        for item in planned_unreachable
    )
    # The real Selection failures prove the terminal/unreached path against an
    # executable preserved-state policy. Main onset rows may all be reachable.
    return {
        "schemaVersion": "e05.s02.unreachable-policy-validation.v1",
        "plannedUnreachableOnsetRowCount": len(planned_unreachable),
        "plannedUnreachablePairsComplete": planned_pair_complete,
        "preservedStateSelectionFailureCount": len(preserved_selection_failures),
        "preservedStateSelectionFailures": preserved_selection_failures,
        "substitutionCount": 0,
        "silentExclusionCount": 0,
        "success": planned_pair_complete
        and bool(preserved_selection_failures)
        and all(
            item["handling"] == "retained_terminal_outcome_no_substitution"
            for item in preserved_selection_failures
        ),
    }
