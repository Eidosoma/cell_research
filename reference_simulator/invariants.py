"""Independent release-critical audit for E01 reference runs.

The audit intentionally re-expresses the S03 transition contract instead of
calling the engine's policy, scheduler, conflict, terminal, or hash helpers.
It accepts only full S05 traces and raises a coded :class:`InvariantViolation`
at the first unexplained divergence.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .model import (
    Architecture,
    Direction,
    FaultMode,
    LEDGER_FIELDS,
    Policy,
    RunResult,
    Scenario,
)


AUDIT_VERSION = "E01-reference-invariant-audit-v1"
TRACE_SCHEMA = "E01.reference-event-pre-S06.v1"
EMPTY_DIGEST = hashlib.sha256(b"E01/reference/trace/v1").digest()
UINT64_SPACE = 1 << 64


class InvariantViolation(AssertionError):
    """A release-critical invariant failed with a stable diagnostic code."""

    def __init__(self, code: str, message: str, *, event_index: int | None = None):
        self.code = code
        self.event_index = event_index
        suffix = f" at event {event_index}" if event_index is not None else ""
        super().__init__(f"{code}{suffix}: {message}")


@dataclass(frozen=True, slots=True)
class OracleProposal:
    kind: str
    actor_id: str
    actor_pos: int
    target_pos: int | None
    new_cursor: int | None
    reason: str
    observation_reads: int
    value_comparisons: int
    ordinal: int
    priority: int
    random_draws: tuple[tuple[str, int, int, int], ...]

    def source_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "actorId": self.actor_id,
            "actorPos": self.actor_pos,
            "targetPos": self.target_pos,
            "newCursor": self.new_cursor,
            "reason": self.reason,
            "observationReads": self.observation_reads,
            "valueComparisons": self.value_comparisons,
            "ordinal": self.ordinal,
            "priority": self.priority,
            "randomDraws": [
                {"stream": stream, "eventIndex": event, "drawIndex": draw, "value": value}
                for stream, event, draw, value in self.random_draws
            ],
        }


def _fail(code: str, message: str, event_index: int | None = None) -> None:
    raise InvariantViolation(code, message, event_index=event_index)


def _require(condition: bool, code: str, message: str, event_index: int | None = None) -> None:
    if not condition:
        _fail(code, message, event_index)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "occupancy": list(state["occupancy"]),
        "selectionCursors": dict(sorted(state["selectionCursors"].items())),
        "activationCount": int(state["activationCount"]),
        "streamCounters": dict(sorted(state["streamCounters"].items())),
        "ledger": {key: int(state["ledger"][key]) for key in LEDGER_FIELDS},
        "terminal": state["terminal"],
    }


def _state_hash(scenario_id: str, state: Mapping[str, Any]) -> str:
    return _hash_json({"scenarioId": scenario_id, "state": _state_dict(state)})


def _rng_u64(seed: int, scenario_id: str, stream: str, event: int, draw: int) -> int:
    root = hashlib.sha256(
        b"E01/RNG/v1\x00"
        + seed.to_bytes(16, "big")
        + b"\x00"
        + scenario_id.encode("utf-8")
    ).digest()
    block = hashlib.sha256(
        root
        + b"\x00"
        + stream.encode("ascii")
        + b"\x00"
        + event.to_bytes(8, "big")
        + draw.to_bytes(4, "big")
    ).digest()
    return int.from_bytes(block[:8], "big")


def _bounded(seed: int, scenario_id: str, stream: str, event: int, upper: int) -> tuple[int, int]:
    limit = UINT64_SPACE - (UINT64_SPACE % upper)
    draw = 0
    while True:
        value = _rng_u64(seed, scenario_id, stream, event, draw)
        draw += 1
        if value < limit:
            return value % upper, draw


def _ordered(left: int | float, right: int | float, direction: Direction) -> bool:
    return left <= right if direction == Direction.ASCENDING else left >= right


def _noop(actor: str, position: int, reason: str, reads: int = 0, comparisons: int = 0) -> OracleProposal:
    return OracleProposal("NoOp", actor, position, None, None, reason, reads, comparisons, 0, 0, ())


def _prefix_status(
    scenario: Scenario, occupancy: Sequence[str], stop: int, direction: Direction
) -> tuple[bool, int, int]:
    reads = 0
    comparisons = 0
    previous: int | float | None = None
    previous_normal = False
    for cell_id in occupancy[:stop]:
        cell = scenario.cell_map[cell_id]
        reads += 1
        if cell.fault != FaultMode.NORMAL:
            previous = None
            previous_normal = False
            continue
        if previous_normal:
            comparisons += 1
            if not _ordered(previous, cell.value, direction):  # type: ignore[arg-type]
                return False, reads, comparisons
        previous = cell.value
        previous_normal = True
    return True, reads, comparisons


def _cell_proposal(
    scenario: Scenario,
    state: Mapping[str, Any],
    actor_id: str,
    *,
    side: str | None,
) -> OracleProposal:
    occupancy = state["occupancy"]
    positions = {cell_id: index for index, cell_id in enumerate(occupancy)}
    actor = scenario.cell_map[actor_id]
    position = positions[actor_id]
    if actor.fault != FaultMode.NORMAL:
        return _noop(actor_id, position, "actor_fault", 1, 0)
    if actor.policy == Policy.BUBBLE:
        _require(side in {"left", "right"}, "INV-RNG-SIDE", "normal Bubble actor lacks a side")
        target_position = position + (-1 if side == "left" else 1)
        if not 0 <= target_position < len(occupancy):
            return _noop(actor_id, position, "boundary", 1, 0)
        target = scenario.cell_map[occupancy[target_position]]
        if actor.direction == Direction.ASCENDING:
            inversion = actor.value < target.value if side == "left" else actor.value > target.value
        else:
            inversion = actor.value > target.value if side == "left" else actor.value < target.value
        if not inversion:
            return _noop(actor_id, position, "ordered_or_equal", 2, 1)
        return OracleProposal(
            "Swap", actor_id, position, target_position, None,
            "strict_adjacent_inversion", 2, 1, 0, 0, (),
        )
    if actor.policy == Policy.INSERTION:
        if position == 0:
            return _noop(actor_id, position, "boundary", 1, 0)
        valid, reads, comparisons = _prefix_status(
            scenario, occupancy, position, actor.direction
        )
        if not valid:
            return _noop(actor_id, position, "prefix_not_ordered", reads + 1, comparisons)
        target_position = position - 1
        target = scenario.cell_map[occupancy[target_position]]
        inversion = (
            actor.value < target.value
            if actor.direction == Direction.ASCENDING
            else actor.value > target.value
        )
        if not inversion:
            return _noop(actor_id, position, "ordered_or_equal", reads + 2, comparisons + 1)
        return OracleProposal(
            "Swap", actor_id, position, target_position, None,
            "insertion_into_ordered_prefix", reads + 2, comparisons + 1, 0, 0, (),
        )
    cursor = state["selectionCursors"][actor_id]
    if not 0 <= cursor < len(occupancy):
        return _noop(actor_id, position, "cursor_exhausted", 1, 0)
    if cursor == position:
        return _noop(actor_id, position, "at_cursor", 1, 0)
    target = scenario.cell_map[occupancy[cursor]]
    delta = 1 if actor.direction == Direction.ASCENDING else -1
    if target.fault == FaultMode.STUCK:
        return OracleProposal(
            "MemoryUpdate", actor_id, position, None, cursor + delta,
            "skip_stuck_target", 2, 0, 0, 0, (),
        )
    if target.value <= actor.value:
        return OracleProposal(
            "MemoryUpdate", actor_id, position, None, cursor + delta,
            "target_already_extreme", 2, 1, 0, 0, (),
        )
    return OracleProposal(
        "Swap", actor_id, position, cursor, None,
        "selection_target", 2, 1, 0, 0, (),
    )


def _traditional_proposal(scenario: Scenario, state: Mapping[str, Any]) -> OracleProposal:
    occupancy = state["occupancy"]
    cells = scenario.cell_map
    direction = scenario.cells[0].direction
    reads = comparisons = 0
    if scenario.traditional_policy == Policy.BUBBLE:
        for left in range(len(occupancy) - 1):
            a, b = cells[occupancy[left]], cells[occupancy[left + 1]]
            reads += 2
            comparisons += 1
            if not _ordered(a.value, b.value, direction):
                return OracleProposal(
                    "Swap", a.cell_id, left, left + 1, None,
                    "traditional_global_bubble_first_inversion", reads, comparisons, 0, 0, (),
                )
        return _noop("__controller__", -1, "traditional_no_inversion", reads, comparisons)
    if scenario.traditional_policy == Policy.INSERTION:
        if len(occupancy) < 2:
            return _noop("__controller__", -1, "traditional_singleton")
        for right in range(1, len(occupancy)):
            a, b = cells[occupancy[right - 1]], cells[occupancy[right]]
            reads += 2
            comparisons += 1
            if not _ordered(a.value, b.value, direction):
                return OracleProposal(
                    "Swap", b.cell_id, right, right - 1, None,
                    "traditional_global_insertion_adjacent_action", reads, comparisons, 0, 0, (),
                )
        return _noop("__controller__", -1, "traditional_no_inversion", reads, comparisons)
    if len(occupancy) < 2:
        return _noop("__controller__", -1, "traditional_singleton")
    for boundary in range(len(occupancy) - 1):
        extreme = boundary
        for index in range(boundary + 1, len(occupancy)):
            reads += 2
            comparisons += 1
            candidate = cells[occupancy[index]].value
            current = cells[occupancy[extreme]].value
            if (
                direction == Direction.ASCENDING and candidate < current
            ) or (
                direction == Direction.DESCENDING and candidate > current
            ):
                extreme = index
        if extreme != boundary:
            return OracleProposal(
                "Swap", occupancy[extreme], extreme, boundary, None,
                "traditional_global_selection_extreme", reads, comparisons, 0, 0, (),
            )
    return _noop("__controller__", -1, "traditional_no_inversion", reads, comparisons)


def _proposal_for_slot(
    scenario: Scenario, state: Mapping[str, Any], event_index: int, ordinal: int
) -> OracleProposal:
    if scenario.architecture == Architecture.TRADITIONAL:
        base = _traditional_proposal(scenario, state)
        return OracleProposal(
            base.kind, base.actor_id, base.actor_pos, base.target_pos,
            base.new_cursor, base.reason, base.observation_reads,
            base.value_comparisons, ordinal, base.priority, base.random_draws,
        )
    actor_index, consumed = _bounded(
        scenario.seed, scenario.scenario_id, "actor_activation", event_index, len(scenario.cells)
    )
    actor_id = scenario.cells[actor_index].cell_id
    draws = [
        (
            "actor_activation", event_index, draw,
            _rng_u64(scenario.seed, scenario.scenario_id, "actor_activation", event_index, draw),
        )
        for draw in range(consumed)
    ]
    actor = scenario.cell_map[actor_id]
    side = None
    if actor.policy == Policy.BUBBLE and actor.fault == FaultMode.NORMAL:
        side_value = _rng_u64(
            scenario.seed, scenario.scenario_id, "bubble_side", event_index, 0
        )
        side = "left" if side_value < (1 << 63) else "right"
        draws.append(("bubble_side", event_index, 0, side_value))
    base = _cell_proposal(scenario, state, actor_id, side=side)
    priority = 0
    if scenario.batch_width > 1:
        priority = _rng_u64(
            scenario.seed, scenario.scenario_id, "conflict_priority", event_index, 0
        )
        draws.append(("conflict_priority", event_index, 0, priority))
    return OracleProposal(
        base.kind, base.actor_id, base.actor_pos, base.target_pos, base.new_cursor,
        base.reason, base.observation_reads, base.value_comparisons,
        ordinal, priority, tuple(draws),
    )


def _resources(proposal: OracleProposal) -> set[tuple[str, str | int]]:
    result: set[tuple[str, str | int]] = {("actor", proposal.actor_id)}
    if proposal.kind == "Swap":
        result.add(("position", proposal.actor_pos))
        result.add(("position", proposal.target_pos))  # type: ignore[arg-type]
    return result


def _decisions(
    scenario: Scenario, state: Mapping[str, Any], proposals: Sequence[OracleProposal]
) -> dict[int, str]:
    candidates: list[OracleProposal] = []
    decisions: dict[int, str] = {}
    for proposal in proposals:
        if proposal.kind == "NoOp":
            decisions[proposal.ordinal] = "no_op"
        elif proposal.kind == "Swap":
            target = proposal.target_pos
            if target is None or not 0 <= target < len(state["occupancy"]):
                decisions[proposal.ordinal] = "rejected_invalid_target"
            elif not 0 <= proposal.actor_pos < len(state["occupancy"]):
                decisions[proposal.ordinal] = "rejected_invalid_actor_position"
            elif state["occupancy"][proposal.actor_pos] != proposal.actor_id:
                decisions[proposal.ordinal] = "rejected_stale_actor"
            elif scenario.cell_map[proposal.actor_id].fault == FaultMode.STUCK:
                decisions[proposal.ordinal] = "rejected_actor_stuck"
            elif scenario.cell_map[state["occupancy"][target]].fault == FaultMode.STUCK:
                decisions[proposal.ordinal] = "rejected_target_stuck"
            else:
                decisions[proposal.ordinal] = "valid"
                candidates.append(proposal)
        elif proposal.actor_id not in state["selectionCursors"] or proposal.new_cursor is None:
            decisions[proposal.ordinal] = "rejected_invalid_memory_update"
        else:
            decisions[proposal.ordinal] = "valid"
            candidates.append(proposal)
    if scenario.batch_width == 1:
        accepted = {item.ordinal for item in candidates}
        lost: set[int] = set()
    else:
        accepted = set()
        lost = set()
        reserved: set[tuple[str, str | int]] = set()
        for proposal in sorted(
            candidates,
            key=lambda item: (
                item.priority, item.actor_id,
                item.target_pos if item.target_pos is not None else -1,
                item.ordinal,
            ),
        ):
            resources = _resources(proposal)
            if resources & reserved:
                lost.add(proposal.ordinal)
            else:
                accepted.add(proposal.ordinal)
                reserved.update(resources)
    for ordinal in lost:
        decisions[ordinal] = "conflict_loss"
    for ordinal in accepted:
        decisions[ordinal] = "accepted"
    return decisions


def _state_invariants(scenario: Scenario, state: Mapping[str, Any], *, event: int | None) -> None:
    expected = {cell.cell_id for cell in scenario.cells}
    occupancy = list(state["occupancy"])
    _require(
        len(occupancy) == len(expected) and set(occupancy) == expected,
        "INV-OCCUPANCY-BIJECTION", "occupancy is not a unique permutation of cell IDs", event,
    )
    cursor_owners = {
        cell.cell_id for cell in scenario.cells if cell.policy == Policy.SELECTION
    }
    _require(
        set(state["selectionCursors"]) == cursor_owners,
        "INV-SELECTION-MEMORY-OWNERS", "Selection cursor owners do not match identities", event,
    )
    _require(
        all(isinstance(value, int) for value in state["selectionCursors"].values()),
        "INV-SELECTION-MEMORY-TYPE", "Selection cursor is not an integer", event,
    )
    _require(state["activationCount"] >= 0, "INV-COUNTER-NONNEGATIVE", "negative activation count", event)
    _require(
        set(state["ledger"]) == set(LEDGER_FIELDS)
        and all(isinstance(value, int) and value >= 0 for value in state["ledger"].values()),
        "INV-LEDGER-NONNEGATIVE", "ledger fields are missing or negative", event,
    )
    _require(
        all(isinstance(value, int) and value >= 0 for value in state["streamCounters"].values()),
        "INV-STREAM-NONNEGATIVE", "stream counter is negative", event,
    )


def _is_complete(scenario: Scenario, state: Mapping[str, Any]) -> bool:
    directions = {cell.direction for cell in scenario.cells}
    if len(directions) != 1:
        return False
    direction = next(iter(directions))
    values = [scenario.cell_map[cell_id].value for cell_id in state["occupancy"]]
    return all(_ordered(left, right, direction) for left, right in zip(values, values[1:]))


def _has_change(scenario: Scenario, state: Mapping[str, Any]) -> bool:
    if scenario.architecture == Architecture.TRADITIONAL:
        proposal = _traditional_proposal(scenario, state)
        if proposal.kind != "Swap" or proposal.target_pos is None:
            return proposal.kind == "MemoryUpdate"
        return (
            scenario.cell_map[state["occupancy"][proposal.actor_pos]].fault != FaultMode.STUCK
            and scenario.cell_map[state["occupancy"][proposal.target_pos]].fault != FaultMode.STUCK
        )
    positions = {cell_id: index for index, cell_id in enumerate(state["occupancy"])}
    for actor in scenario.cells:
        if actor.fault != FaultMode.NORMAL:
            continue
        if actor.policy == Policy.BUBBLE:
            for side in ("left", "right"):
                proposal = _cell_proposal(scenario, state, actor.cell_id, side=side)
                if proposal.kind == "Swap":
                    target = scenario.cell_map[state["occupancy"][proposal.target_pos]]  # type: ignore[index]
                    if target.fault != FaultMode.STUCK:
                        return True
        elif actor.policy == Policy.INSERTION:
            proposal = _cell_proposal(scenario, state, actor.cell_id, side=None)
            if proposal.kind == "Swap":
                target = scenario.cell_map[state["occupancy"][proposal.target_pos]]  # type: ignore[index]
                if target.fault != FaultMode.STUCK:
                    return True
        else:
            cursor = state["selectionCursors"][actor.cell_id]
            if 0 <= cursor < len(state["occupancy"]) and cursor != positions[actor.cell_id]:
                return True
    return False


def _terminal(scenario: Scenario, state: Mapping[str, Any]) -> str | None:
    try:
        _state_invariants(scenario, state, event=None)
    except InvariantViolation:
        return "invariant_error"
    if _is_complete(scenario, state):
        return "complete"
    if not _has_change(scenario, state):
        return "quiescent"
    if state["activationCount"] >= scenario.max_activations:
        return "event_budget"
    return None


def audit_reference_result(result: RunResult | Mapping[str, Any]) -> dict[str, Any]:
    """Independently replay and audit one full reference result."""
    raw = result.to_dict() if isinstance(result, RunResult) else deepcopy(dict(result))
    _require(raw.get("schema") == "E01.reference-result.v1", "INV-RESULT-SCHEMA", "unsupported result schema")
    scenario = Scenario.from_dict(raw["scenario"])
    _require(raw["summary"].get("traceMode") == "full", "INV-TRACE-MODE", "full trace is required")
    events = list(raw["events"])
    _require(
        len(events) == raw["summary"].get("retainedEventCount"),
        "INV-EVENT-COUNT", "retained event count mismatch",
    )
    state: dict[str, Any] = {
        "occupancy": list(scenario.initial_occupancy),
        "selectionCursors": dict(scenario.initial_selection_cursors),
        "activationCount": 0,
        "streamCounters": {},
        "ledger": {key: 0 for key in LEDGER_FIELDS},
        "terminal": None,
    }
    state["terminal"] = _terminal(scenario, state)
    _state_invariants(scenario, state, event=None)
    _require(
        raw["initialStateHash"] == _state_hash(scenario.scenario_id, state),
        "INV-INITIAL-HASH", "initial state hash mismatch",
    )
    initial_metadata = {
        cell.cell_id: (cell.value, cell.policy, cell.direction, cell.fault, cell.analysis_label)
        for cell in scenario.cells
    }
    initial_values = Counter(cell.value for cell in scenario.cells)
    stuck_positions = {
        cell.cell_id: state["occupancy"].index(cell.cell_id)
        for cell in scenario.cells if cell.fault == FaultMode.STUCK
    }
    coverage = Counter(
        scenarios=1,
        events=0,
        acceptedSwaps=0,
        memoryUpdates=0,
        noOps=0,
        rejections=0,
        conflictLosses=0,
        boundaryNoOps=0,
        passiveActorNoOps=0,
        passiveTargetDisplacements=0,
        stuckTargetRejections=0,
        duplicateScenarios=int(len(initial_values) < len(scenario.cells)),
        mixedDirectionScenarios=int(len({cell.direction for cell in scenario.cells}) > 1),
        batchScenarios=int(scenario.batch_width > 1),
        traditionalScenarios=int(scenario.architecture == Architecture.TRADITIONAL),
    )
    cursor = 0
    digest = EMPTY_DIGEST
    while cursor < len(events):
        _require(state["terminal"] is None, "INV-EVENT-AFTER-TERMINAL", "event occurs after terminal state")
        first = events[cursor]
        _require(first.get("schemaVersion") == TRACE_SCHEMA, "INV-EVENT-SCHEMA", "wrong event schema", first.get("eventIndex"))
        width = min(scenario.batch_width, scenario.max_activations - state["activationCount"])
        _require(width > 0, "INV-BUDGET-TRUNCATION", "batch exists with no remaining budget")
        group = events[cursor : cursor + width]
        _require(len(group) == width, "INV-BATCH-WIDTH", "truncated event group", first.get("eventIndex"))
        expected_indices = list(range(state["activationCount"], state["activationCount"] + width))
        _require([event["eventIndex"] for event in group] == expected_indices, "INV-EVENT-INDEX", "event indices are not contiguous")
        _require([event["batchOrdinal"] for event in group] == list(range(width)), "INV-BATCH-ORDINAL", "batch ordinals are invalid")
        _require(all(event["batchWidth"] == width for event in group), "INV-BATCH-WIDTH", "declared batch width mismatch")

        snapshot = deepcopy(state)
        pre_hash = _state_hash(scenario.scenario_id, snapshot)
        _require(all(event["preStateHash"] == pre_hash for event in group), "INV-PRE-HASH", "pre-state hash mismatch", first["eventIndex"])
        proposals = [
            _proposal_for_slot(scenario, snapshot, event_index, ordinal)
            for ordinal, event_index in enumerate(expected_indices)
        ]
        decisions = _decisions(scenario, snapshot, proposals)
        for event, proposal in zip(group, proposals):
            index = event["eventIndex"]
            _require(event["scenarioId"] == scenario.scenario_id, "INV-SCENARIO-ID", "event scenario ID mismatch", index)
            _require(event["proposal"] == proposal.source_dict(), "INV-PROPOSAL-ORACLE", "proposal differs from independent policy/RNG oracle", index)
            _require(event["randomAddressesAndDraws"] == proposal.source_dict()["randomDraws"], "INV-RNG-ADDRESS", "random draw/address mismatch", index)
            actor_cell = scenario.cell_map.get(proposal.actor_id)
            expected_algotype = (
                actor_cell.policy.value
                if actor_cell is not None
                else f"Traditional:{scenario.traditional_policy.value}"
            )
            expected_direction = actor_cell.direction.value if actor_cell is not None else "controller"
            _require(event["actorId"] == proposal.actor_id, "INV-ACTOR-ID", "actor identity mismatch", index)
            _require(event["actorAlgotype"] == expected_algotype, "INV-ACTOR-ALGOTYPE", "actor Algotype mismatch", index)
            _require(event["actorDirection"] == expected_direction, "INV-ACTOR-DIRECTION", "actor direction mismatch", index)
            _require(event["decision"] == decisions[proposal.ordinal], "INV-DECISION", "decision differs from validation/conflict oracle", index)
            _require(
                event["observation"] == {
                    "reads": proposal.observation_reads,
                    "valueComparisons": proposal.value_comparisons,
                },
                "INV-OBSERVATION-COST", "observation counts differ from oracle", index,
            )

        expected_deltas: list[dict[str, int]] = []
        accepted_resources: set[tuple[str, str | int]] = set()
        for event, proposal in zip(group, proposals):
            index = event["eventIndex"]
            decision = decisions[proposal.ordinal]
            delta = {key: 0 for key in LEDGER_FIELDS}
            delta["activations"] = 1
            delta["observationReads"] = proposal.observation_reads
            delta["valueComparisons"] = proposal.value_comparisons
            delta["proposals"] = 1
            if proposal.kind == "NoOp":
                delta["noOps"] = 1
                coverage["noOps"] += 1
                coverage["boundaryNoOps"] += int(proposal.reason == "boundary")
                if scenario.cell_map.get(proposal.actor_id) is not None:
                    coverage["passiveActorNoOps"] += int(
                        scenario.cell_map[proposal.actor_id].fault == FaultMode.PASSIVE
                    )
            elif decision.startswith("rejected"):
                delta["rejections"] = 1
                coverage["rejections"] += 1
                coverage["stuckTargetRejections"] += int(decision == "rejected_target_stuck")
            elif decision == "conflict_loss":
                delta["conflictLosses"] = 1
                coverage["conflictLosses"] += 1
            elif proposal.kind == "MemoryUpdate" and decision == "accepted":
                delta["memoryUpdates"] = 1
                state["selectionCursors"][proposal.actor_id] = proposal.new_cursor
                coverage["memoryUpdates"] += 1
            elif proposal.kind == "Swap" and decision == "accepted":
                _require(proposal.target_pos is not None, "INV-TARGET-POSITION", "accepted swap lacks target", index)
                resources = _resources(proposal)
                _require(not resources & accepted_resources, "INV-CONFLICT-DISJOINT", "accepted batch changes overlap", index)
                accepted_resources.update(resources)
                target_id = snapshot["occupancy"][proposal.target_pos]
                _require(proposal.actor_pos != proposal.target_pos, "INV-SWAP-DISTINCT", "swap endpoints are equal", index)
                _require(snapshot["occupancy"][proposal.actor_pos] == proposal.actor_id, "INV-ACTOR-POSITION", "actor position does not resolve to actor", index)
                _require(scenario.cell_map[target_id].fault != FaultMode.STUCK, "INV-STUCK-IMMOBILE", "accepted swap displaces a stuck target", index)
                state["occupancy"][proposal.actor_pos] = snapshot["occupancy"][proposal.target_pos]
                state["occupancy"][proposal.target_pos] = snapshot["occupancy"][proposal.actor_pos]
                delta["acceptedSwaps"] = 1
                delta["displacedCells"] = 2
                coverage["acceptedSwaps"] += 1
                coverage["passiveTargetDisplacements"] += int(
                    scenario.cell_map[target_id].fault == FaultMode.PASSIVE
                )
            expected_deltas.append(delta)
            _require(event["ledgerDelta"] == delta, "INV-LEDGER-DELTA", "ledger delta mismatch", index)
            for key in LEDGER_FIELDS:
                state["ledger"][key] += delta[key]

        for proposal in proposals:
            for stream, _event, _draw, _value in proposal.random_draws:
                state["streamCounters"][stream] = state["streamCounters"].get(stream, 0) + 1
        state["activationCount"] += width
        state["terminal"] = _terminal(scenario, state)
        _state_invariants(scenario, state, event=group[-1]["eventIndex"])
        for stuck_id, position in stuck_positions.items():
            _require(
                state["occupancy"].index(stuck_id) == position,
                "INV-STUCK-IMMOBILE", f"stuck identity {stuck_id} moved", group[-1]["eventIndex"],
            )
        _require(
            Counter(scenario.cell_map[cell_id].value for cell_id in state["occupancy"]) == initial_values,
            "INV-VALUE-CONSERVATION", "value multiset changed", group[-1]["eventIndex"],
        )
        post_hash = _state_hash(scenario.scenario_id, state)
        _require(all(event["postStateHash"] == post_hash for event in group), "INV-POST-HASH", "post-state hash mismatch", group[-1]["eventIndex"])
        _require(all(event["stopReasonIfAny"] == state["terminal"] for event in group), "INV-STOP-METADATA", "event stop reason mismatch", group[-1]["eventIndex"])
        for event in group:
            digest = hashlib.sha256(digest + _canonical_bytes(event)).digest()
        coverage["events"] += width
        cursor += width

    _require(state["terminal"] is not None, "INV-NONTERMINAL-RESULT", "run ended without terminal state")
    _require(_state_dict(raw["finalState"]) == _state_dict(state), "INV-FINAL-STATE", "final state differs from replay")
    _require(raw["finalStateHash"] == _state_hash(scenario.scenario_id, state), "INV-FINAL-HASH", "final state hash mismatch")
    _require(raw["eventDigest"] == digest.hex(), "INV-EVENT-DIGEST", "event digest mismatch")
    final_occupancy = list(state["occupancy"])
    final_values = [scenario.cell_map[cell_id].value for cell_id in final_occupancy]
    _require(raw["summary"]["finalOccupancy"] == final_occupancy, "INV-SUMMARY-OCCUPANCY", "summary occupancy mismatch")
    _require(raw["summary"]["finalValues"] == final_values, "INV-SUMMARY-VALUES", "summary values mismatch")
    _require(raw["summary"]["ledger"] == state["ledger"], "INV-SUMMARY-LEDGER", "summary ledger mismatch")
    _require(raw["summary"]["activationCount"] == state["activationCount"], "INV-SUMMARY-ACTIVATIONS", "summary activation count mismatch")
    _require(raw["summary"]["stopReason"] == state["terminal"], "INV-SUMMARY-STOP", "summary stop reason mismatch")
    _require(raw["summary"]["completed"] == (state["terminal"] == "complete"), "INV-SUMMARY-COMPLETION", "summary completion flag mismatch")
    _require(state["ledger"]["activations"] == state["activationCount"] == len(events), "INV-ACTIVATION-ACCOUNTING", "activation/event/ledger counts differ")
    _require(state["ledger"]["proposals"] == len(events), "INV-PROPOSAL-ACCOUNTING", "proposal count differs from events")
    _require(state["ledger"]["displacedCells"] == 2 * state["ledger"]["acceptedSwaps"], "INV-DISPLACEMENT-ACCOUNTING", "displacement is not two per swap")
    _require(
        {cell.cell_id: (cell.value, cell.policy, cell.direction, cell.fault, cell.analysis_label) for cell in scenario.cells}
        == initial_metadata,
        "INV-METADATA-CONSERVATION", "immutable cell metadata changed",
    )
    if state["terminal"] == "complete":
        _require(_is_complete(scenario, state), "INV-COMPLETE-PREDICATE", "complete state is not consensus non-strict sorted")
    elif state["terminal"] == "quiescent":
        _require(not _is_complete(scenario, state) and not _has_change(scenario, state), "INV-QUIESCENCE", "quiescent state has an admissible change or is complete")
    elif state["terminal"] == "event_budget":
        _require(
            state["activationCount"] >= scenario.max_activations
            and not _is_complete(scenario, state)
            and _has_change(scenario, state),
            "INV-EVENT-BUDGET", "budget terminal violates precedence",
        )
    else:
        _fail("INV-TERMINAL-REASON", f"unexpected terminal {state['terminal']!r}")
    coverage[f"stop.{state['terminal']}"] += 1
    coverage[f"architecture.{scenario.architecture.value}"] += 1
    for policy in {cell.policy.value for cell in scenario.cells}:
        coverage[f"policy.{policy}"] += 1
    for direction in {cell.direction.value for cell in scenario.cells}:
        coverage[f"direction.{direction}"] += 1
    return {
        "auditVersion": AUDIT_VERSION,
        "scenarioId": scenario.scenario_id,
        "success": True,
        "coverage": dict(sorted(coverage.items())),
    }
