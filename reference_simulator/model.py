"""Typed immutable scenario and serializable run-state structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping


SEMANTICS_VERSION = "E01-reference-v1"


class Policy(str, Enum):
    BUBBLE = "Bubble"
    INSERTION = "Insertion"
    SELECTION = "Selection"


class Direction(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


class FaultMode(str, Enum):
    NORMAL = "normal"
    PASSIVE = "passive"
    STUCK = "stuck"


class Architecture(str, Enum):
    CELL_VIEW = "cell_view"
    TRADITIONAL = "traditional"


class ProposalKind(str, Enum):
    NO_OP = "NoOp"
    SWAP = "Swap"
    MEMORY_UPDATE = "MemoryUpdate"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class Cell:
    cell_id: str
    value: int | float
    policy: Policy
    direction: Direction = Direction.ASCENDING
    fault: FaultMode = FaultMode.NORMAL
    analysis_label: str | None = None

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("cell_id must be nonempty")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("cell value must be numeric")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("cell value must be finite")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "cellId": self.cell_id,
            "value": self.value,
            "policy": self.policy.value,
            "direction": self.direction.value,
            "fault": self.fault.value,
        }
        if self.analysis_label is not None:
            result["analysisLabel"] = self.analysis_label
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Cell":
        return cls(
            cell_id=str(value["cellId"]),
            value=value["value"],
            policy=Policy(value["policy"]),
            direction=Direction(value.get("direction", "ascending")),
            fault=FaultMode(value.get("fault", "normal")),
            analysis_label=value.get("analysisLabel"),
        )


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    cells: tuple[Cell, ...]
    initial_occupancy: tuple[str, ...]
    initial_selection_cursors: tuple[tuple[str, int], ...]
    seed: int
    max_activations: int
    architecture: Architecture = Architecture.CELL_VIEW
    scheduler: str = "serial_counter_addressed"
    batch_width: int = 1
    traditional_policy: Policy | None = None
    generation_key: str = "caller-specified"
    fault_placement: str = "explicit"
    requested_fault_count: int = 0
    realized_fault_count: int = 0
    rng_profile: str = "sha256_counter_E01_v1"
    goal_profile: str = "homogeneous_direction_nonstrict_v1"
    metric_profile: str = "reference_ledger_v1"
    semantics_version: str = SEMANTICS_VERSION
    _cell_map_cache: dict[str, Cell] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Scenario cells are immutable.  Caching this identity lookup avoids
        # rebuilding the same n-entry dictionary on every activation without
        # changing scenario serialization or transition semantics.
        object.__setattr__(
            self,
            "_cell_map_cache",
            {cell.cell_id: cell for cell in self.cells},
        )

    @staticmethod
    def _content(
        cells: Iterable[Cell],
        initial_occupancy: Iterable[str],
        initial_selection_cursors: Iterable[tuple[str, int]],
        seed: int,
        max_activations: int,
        architecture: Architecture,
        scheduler: str,
        batch_width: int,
        traditional_policy: Policy | None,
        generation_key: str,
        fault_placement: str,
        requested_fault_count: int,
        realized_fault_count: int,
        rng_profile: str,
        goal_profile: str,
        metric_profile: str,
        semantics_version: str,
    ) -> dict[str, Any]:
        ordered_cells = sorted(cells, key=lambda item: item.cell_id)
        return {
            "semanticsVersion": semantics_version,
            "cells": [cell.to_dict() for cell in ordered_cells],
            "initialOccupancy": list(initial_occupancy),
            "initialSelectionCursors": {
                key: value for key, value in sorted(initial_selection_cursors)
            },
            "seed": str(seed),
            "maxActivations": max_activations,
            "architecture": architecture.value,
            "scheduler": scheduler,
            "batchWidth": batch_width,
            "traditionalPolicy": traditional_policy.value if traditional_policy else None,
            "generationKey": generation_key,
            "faultPlacement": fault_placement,
            "requestedFaultCount": requested_fault_count,
            "realizedFaultCount": realized_fault_count,
            "rngProfile": rng_profile,
            "goalProfile": goal_profile,
            "metricProfile": metric_profile,
        }

    @classmethod
    def create(
        cls,
        cells: Iterable[Cell],
        *,
        initial_occupancy: Iterable[str] | None = None,
        initial_selection_cursors: Mapping[str, int] | None = None,
        seed: int = 0,
        max_activations: int = 100_000,
        architecture: Architecture = Architecture.CELL_VIEW,
        scheduler: str | None = None,
        batch_width: int = 1,
        traditional_policy: Policy | None = None,
        generation_key: str = "caller-specified",
        fault_placement: str = "explicit",
        requested_fault_count: int | None = None,
        rng_profile: str = "sha256_counter_E01_v1",
        goal_profile: str = "homogeneous_direction_nonstrict_v1",
        metric_profile: str = "reference_ledger_v1",
    ) -> "Scenario":
        frozen_cells = tuple(sorted(cells, key=lambda item: item.cell_id))
        if not frozen_cells:
            raise ValueError("scenario must have at least one cell")
        if len({cell.cell_id for cell in frozen_cells}) != len(frozen_cells):
            raise ValueError("cell IDs must be unique")
        occupancy = tuple(initial_occupancy or (cell.cell_id for cell in frozen_cells))
        default_cursors = {
            cell.cell_id: (0 if cell.direction == Direction.ASCENDING else len(frozen_cells) - 1)
            for cell in frozen_cells
            if cell.policy == Policy.SELECTION
        }
        if initial_selection_cursors:
            default_cursors.update(initial_selection_cursors)
        cursors = tuple(sorted(default_cursors.items()))
        effective_scheduler = scheduler or (
            "traditional_controller"
            if architecture == Architecture.TRADITIONAL
            else ("batch_counter_addressed" if batch_width > 1 else "serial_counter_addressed")
        )
        if max_activations < 0:
            raise ValueError("max_activations must be nonnegative")
        if not 1 <= batch_width <= 8:
            raise ValueError("batch_width must be in [1, 8]")
        if architecture == Architecture.TRADITIONAL and traditional_policy is None:
            raise ValueError("traditional_policy is required for traditional architecture")
        realized_fault_count = sum(cell.fault != FaultMode.NORMAL for cell in frozen_cells)
        requested = realized_fault_count if requested_fault_count is None else requested_fault_count
        if requested < 0:
            raise ValueError("requested_fault_count must be nonnegative")
        content = cls._content(
            frozen_cells, occupancy, cursors, seed, max_activations, architecture,
            effective_scheduler, batch_width, traditional_policy, generation_key,
            fault_placement, requested, realized_fault_count, rng_profile,
            goal_profile, metric_profile, SEMANTICS_VERSION,
        )
        scenario_id = "r1:" + sha256_json(content)
        scenario = cls(
            scenario_id=scenario_id,
            cells=frozen_cells,
            initial_occupancy=occupancy,
            initial_selection_cursors=cursors,
            seed=seed,
            max_activations=max_activations,
            architecture=architecture,
            scheduler=effective_scheduler,
            batch_width=batch_width,
            traditional_policy=traditional_policy,
            generation_key=generation_key,
            fault_placement=fault_placement,
            requested_fault_count=requested,
            realized_fault_count=realized_fault_count,
            rng_profile=rng_profile,
            goal_profile=goal_profile,
            metric_profile=metric_profile,
        )
        scenario.validate()
        return scenario

    def content_dict(self) -> dict[str, Any]:
        return self._content(
            self.cells, self.initial_occupancy, self.initial_selection_cursors,
            self.seed, self.max_activations, self.architecture, self.scheduler,
            self.batch_width, self.traditional_policy, self.generation_key,
            self.fault_placement, self.requested_fault_count,
            self.realized_fault_count, self.rng_profile, self.goal_profile,
            self.metric_profile, self.semantics_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"scenarioId": self.scenario_id, **self.content_dict()}

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def validate(self) -> None:
        ids = {cell.cell_id for cell in self.cells}
        if len(self.initial_occupancy) != len(self.cells) or set(self.initial_occupancy) != ids:
            raise ValueError("initial occupancy must be a permutation of cell IDs")
        cell_map = self.cell_map
        if not 0 <= self.seed < (1 << 128):
            raise ValueError("seed must be an unsigned 128-bit integer")
        if self.max_activations < 0:
            raise ValueError("maxActivations must be nonnegative")
        if not 1 <= self.batch_width <= 8:
            raise ValueError("batchWidth must be in [1, 8]")
        realized = sum(cell.fault != FaultMode.NORMAL for cell in self.cells)
        if self.realized_fault_count != realized or self.requested_fault_count < 0:
            raise ValueError("fault count fields do not match scenario cells")
        if self.fault_placement == "reference_without_replacement" and self.requested_fault_count != realized:
            raise ValueError("reference fault placement must realize exactly the requested count")
        if self.fault_placement == "explicit" and self.requested_fault_count != realized:
            raise ValueError("explicit fault placement count must match scenario cells")
        if self.fault_placement == "legacy_with_replacement" and realized > self.requested_fault_count:
            raise ValueError("legacy placement cannot realize more distinct faults than requested draws")
        if self.fault_placement not in {
            "explicit", "reference_without_replacement", "legacy_with_replacement"
        }:
            raise ValueError("unsupported reference fault placement profile")
        if self.architecture == Architecture.TRADITIONAL:
            if self.traditional_policy is None or self.scheduler != "traditional_controller" or self.batch_width != 1:
                raise ValueError("traditional scenarios require a policy, controller scheduler, and batch width 1")
            if len({cell.direction for cell in self.cells}) != 1:
                raise ValueError("traditional clean-room controllers require a homogeneous direction")
            if any(cell.policy != self.traditional_policy for cell in self.cells):
                raise ValueError("traditional scenario cell policy labels must match its controller")
        elif self.traditional_policy is not None:
            raise ValueError("cell-view scenario cannot declare a traditional policy")
        else:
            expected_scheduler = "batch_counter_addressed" if self.batch_width > 1 else "serial_counter_addressed"
            if self.scheduler != expected_scheduler:
                raise ValueError("cell-view scheduler and batch width are inconsistent")
        if self.rng_profile != "sha256_counter_E01_v1":
            raise ValueError("unsupported RNG profile")
        if self.goal_profile != "homogeneous_direction_nonstrict_v1":
            raise ValueError("unsupported goal profile")
        if self.metric_profile != "reference_ledger_v1":
            raise ValueError("unsupported metric profile")
        if self.semantics_version != SEMANTICS_VERSION:
            raise ValueError("unsupported reference semantics version")
        if not self.generation_key:
            raise ValueError("generationKey must be nonempty")
        for cell_id, cursor in self.initial_selection_cursors:
            if cell_id not in ids or cell_map[cell_id].policy != Policy.SELECTION:
                raise ValueError("cursor owner must be a Selection cell")
            if not isinstance(cursor, int):
                raise TypeError("cursor must be an integer")
            expected_cursor = 0 if cell_map[cell_id].direction == Direction.ASCENDING else len(self.cells) - 1
            if cursor != expected_cursor:
                raise ValueError("initial Selection cursor must be the direction-specific boundary")
        expected_id = "r1:" + sha256_json(self.content_dict())
        if self.scenario_id != expected_id:
            raise ValueError("scenario ID does not match canonical content")

    @property
    def cell_map(self) -> Mapping[str, Cell]:
        return MappingProxyType(self._cell_map_cache)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Scenario":
        cursors = tuple(sorted((str(k), int(v)) for k, v in value["initialSelectionCursors"].items()))
        scenario = cls(
            scenario_id=str(value["scenarioId"]),
            cells=tuple(sorted((Cell.from_dict(item) for item in value["cells"]), key=lambda x: x.cell_id)),
            initial_occupancy=tuple(value["initialOccupancy"]),
            initial_selection_cursors=cursors,
            seed=int(value["seed"]),
            max_activations=int(value["maxActivations"]),
            architecture=Architecture(value["architecture"]),
            scheduler=str(value["scheduler"]),
            batch_width=int(value["batchWidth"]),
            traditional_policy=Policy(value["traditionalPolicy"]) if value.get("traditionalPolicy") else None,
            generation_key=str(value["generationKey"]),
            fault_placement=str(value["faultPlacement"]),
            requested_fault_count=int(value["requestedFaultCount"]),
            realized_fault_count=int(value["realizedFaultCount"]),
            rng_profile=str(value["rngProfile"]),
            goal_profile=str(value["goalProfile"]),
            metric_profile=str(value["metricProfile"]),
            semantics_version=str(value["semanticsVersion"]),
        )
        scenario.validate()
        return scenario

    @classmethod
    def from_json_bytes(cls, value: bytes) -> "Scenario":
        return cls.from_dict(json.loads(value))


LEDGER_FIELDS = (
    "activations",
    "observationReads",
    "valueComparisons",
    "proposals",
    "noOps",
    "rejections",
    "memoryUpdates",
    "acceptedSwaps",
    "displacedCells",
    "conflictLosses",
)


@dataclass(slots=True)
class RunState:
    occupancy: list[str]
    selection_cursors: dict[str, int]
    activation_count: int = 0
    stream_counters: dict[str, int] = field(default_factory=dict)
    ledger: dict[str, int] = field(default_factory=lambda: {name: 0 for name in LEDGER_FIELDS})
    terminal: str | None = None

    def clone(self) -> "RunState":
        return RunState(
            occupancy=list(self.occupancy),
            selection_cursors=dict(self.selection_cursors),
            activation_count=self.activation_count,
            stream_counters=dict(self.stream_counters),
            ledger=dict(self.ledger),
            terminal=self.terminal,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "occupancy": list(self.occupancy),
            "selectionCursors": dict(sorted(self.selection_cursors.items())),
            "activationCount": self.activation_count,
            "streamCounters": dict(sorted(self.stream_counters.items())),
            "ledger": {key: self.ledger[key] for key in LEDGER_FIELDS},
            "terminal": self.terminal,
        }


@dataclass(frozen=True, slots=True)
class Proposal:
    kind: ProposalKind
    actor_id: str
    actor_pos: int
    target_pos: int | None = None
    new_cursor: int | None = None
    reason: str = ""
    observation_reads: int = 0
    value_comparisons: int = 0
    ordinal: int = 0
    priority: int = 0
    random_draws: tuple[tuple[str, int, int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
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
                {"stream": s, "eventIndex": e, "drawIndex": d, "value": v}
                for s, e, d, v in self.random_draws
            ],
        }


@dataclass(frozen=True, slots=True)
class RunResult:
    scenario: Scenario
    final_state: Mapping[str, Any]
    initial_state_hash: str
    final_state_hash: str
    event_digest: str
    events: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "E01.reference-result.v1",
            "scenario": self.scenario.to_dict(),
            "initialStateHash": self.initial_state_hash,
            "finalStateHash": self.final_state_hash,
            "eventDigest": self.event_digest,
            "events": list(self.events),
            "finalState": dict(self.final_state),
            "summary": dict(self.summary),
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_json_bytes(cls, value: bytes) -> "RunResult":
        decoded = json.loads(value)
        return cls(
            scenario=Scenario.from_dict(decoded["scenario"]),
            final_state=decoded["finalState"],
            initial_state_hash=decoded["initialStateHash"],
            final_state_hash=decoded["finalStateHash"],
            event_digest=decoded["eventDigest"],
            events=tuple(decoded["events"]),
            summary=decoded["summary"],
        )


def state_hash(scenario_id: str, state: RunState) -> str:
    return sha256_json({"scenarioId": scenario_id, "state": state.to_dict()})
