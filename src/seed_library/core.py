"""S03 seed generation, bounded semantic deduplication, and training probes.

This module deliberately does not extend the S02 episode action interface.  It
uses the closed S01 DSL interpreter and deterministic, outcome-free probes
derived from S02 *training* records.  E01 proposal semantics are used as a
shadow oracle for Bubble, Insertion, and Selection.  E05 overlays and E06
opaque-candidate internals remain unbound; spatial execution below is a typed
synthetic-fixture check, not an E06 task adapter or outcome evaluation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from reference_simulator.engine import initial_state
from reference_simulator.model import (
    Cell,
    Direction,
    FaultMode,
    Policy,
    ProposalKind,
    Scenario,
)
from reference_simulator.policies import _prefix_is_ordered, cell_view_proposal
from src.environment_suite.contracts import (
    ScenarioRecord,
    Split,
    canonical_json_bytes,
    load_split_manifest,
    load_task_registry,
)
from src.policy_dsl import (
    BASELINE_DIRECTORY,
    CompiledPolicy,
    compile_policy,
    execute_policy,
    load_policy,
    policy_to_dict,
)


SEED_LIBRARY_VERSION = "e07.s03.seed-library.v1"
PROBE_VERSION = "e07.s03.training-probe.v1"
SEMANTIC_EQUIVALENCE_VERSION = "e07.s03.bounded-semantic-equivalence.v1"
DEFAULT_PROBES_PER_RECORD = 128

FAITHFUL_POLICY_MAP = {
    "bubble_cell_view_v1": Policy.BUBBLE,
    "insertion_cell_view_v1": Policy.INSERTION,
    "selection_cell_view_v1": Policy.SELECTION,
}

LINE_PROBE_TASKS = {
    "e07_s02_sorting_1d",
    "e07_s02_faults_1d",
    "e07_s02_detour_1d",
    "e07_s02_chimera_1d",
}
SPATIAL_PROBE_TASKS = {
    "e07_s02_spatial2d_local",
    "e07_s02_spatial2d_memory",
}
UNBOUND_TASKS = {
    "e07_s02_regeneration_1d",
    "e07_s02_target_change_1d",
}


@dataclass(frozen=True, slots=True)
class CandidateSeed:
    """One pre-deduplication seed and its declared provenance."""

    family: str
    role: str
    parents: tuple[str, ...]
    document: Mapping[str, Any]
    validation_note: str

    @property
    def policy_id(self) -> str:
        return str(self.document["policyId"])

    def compile(self) -> CompiledPolicy:
        return compile_policy(self.document)


@dataclass(frozen=True, slots=True)
class TrainingProbe:
    probe_id: str
    task_id: str
    scenario_id: str
    environment: str
    available_scalars: Mapping[str, Any]
    available_candidates: tuple[Mapping[str, Any], ...]
    scenario: Scenario | None = None
    actor_id: str | None = None
    side: str = "none"
    selection_cursor: int | None = None
    prefix_reads: int = 0
    prefix_comparisons: int = 0


def _digest_int(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(b"E07/S03/counter/v1\x00" + payload).digest()[:8],
        "big",
    )


def _hash(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(value)
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def load_training_records(
    task_registry_path: str | Path,
    split_manifest_path: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[ScenarioRecord, ...]]:
    """Load public split metadata and return only frozen training records."""

    task_metadata, tasks = load_task_registry(task_registry_path)
    split_metadata, records = load_split_manifest(split_manifest_path, tasks)
    training = tuple(
        sorted(
            (record for record in records.values() if record.split == Split.TRAIN),
            key=lambda record: record.task_id,
        )
    )
    if len(training) != len(tasks):
        raise ValueError("exactly one training record per task is required")
    if any(
        record.protected or record.outcome_access != "development"
        for record in training
    ):
        raise ValueError("training records must be unprotected development records")
    return task_metadata, split_metadata, training


def _baseline_document(name: str) -> dict[str, Any]:
    return policy_to_dict(load_policy(BASELINE_DIRECTORY / f"{name}.json"))


def _candidate(
    document: Mapping[str, Any],
    *,
    family: str,
    role: str,
    parents: Sequence[str] = (),
    validation_note: str,
) -> CandidateSeed:
    compiled = compile_policy(document)
    return CandidateSeed(
        family=family,
        role=role,
        parents=tuple(parents),
        document=policy_to_dict(compiled),
        validation_note=validation_note,
    )


def _simplified_candidates() -> list[CandidateSeed]:
    bubble = _baseline_document("bubble_cell_view_v1")
    insertion = _baseline_document("insertion_cell_view_v1")
    selection = _baseline_document("selection_cell_view_v1")
    spatial = _baseline_document("spatial_greedy_local_v1")
    seeds: list[CandidateSeed] = []

    for suffix, indices, side in (
        ("left_only", (0, 2), "left"),
        ("right_only", (1, 3), "right"),
    ):
        document = deepcopy(bubble)
        document["policyId"] = f"bubble_{suffix}_v1"
        document["rules"] = [deepcopy(bubble["rules"][index]) for index in indices]
        document["limits"]["maxRules"] = 2
        seeds.append(
            _candidate(
                document,
                family="simplified_variant",
                role=f"Bubble rule-subset restricted to scheduler side {side}",
                parents=("bubble_cell_view_v1",),
                validation_note="Validated exact source-rule subset; no comparator, action, permission, or direction semantics changed.",
            )
        )

    document = deepcopy(insertion)
    document["policyId"] = "insertion_adjacent_unlicensed_v1"
    document["permissions"].remove("line.prefix_ordered")
    for rule in document["rules"]:
        rule["when"]["all"] = [
            condition
            for condition in rule["when"]["all"]
            if condition.get("left") != {"obs": "line.prefix_ordered"}
        ]
    seeds.append(
        _candidate(
            document,
            family="simplified_variant",
            role="Insertion-like adjacent comparator without the licensed prefix predicate",
            parents=("insertion_cell_view_v1",),
            validation_note="Validated removal of exactly the prefix permission and its two predicate conditions; adjacent comparator/action retained.",
        )
    )

    document = deepcopy(selection)
    document["policyId"] = "selection_scan_only_v1"
    document["rules"] = deepcopy(selection["rules"][:2])
    document["limits"]["maxRules"] = 2
    document["limits"]["maxMovementRadius"] = 0
    seeds.append(
        _candidate(
            document,
            family="simplified_variant",
            role="Selection cursor scan with the long-range swap rule removed",
            parents=("selection_cell_view_v1",),
            validation_note="Validated exact removal of the swap rule and licensed movement radius; engine-owned cursor semantics remain explicit.",
        )
    )

    document = deepcopy(selection)
    document["policyId"] = "selection_swap_only_v1"
    document["permissions"] = [
        "selection.cursor_at_actor",
        "selection.cursor_in_bounds",
    ]
    document["rules"] = [deepcopy(selection["rules"][2])]
    document["limits"]["maxRules"] = 1
    seeds.append(
        _candidate(
            document,
            family="simplified_variant",
            role="Selection cursor swap without cursor advancement or target comparison",
            parents=("selection_cell_view_v1",),
            validation_note="Validated exact swap-rule extraction; licensed long-range action remains declared and separately costed.",
        )
    )

    document = deepcopy(spatial)
    document["policyId"] = "spatial_greedy_adjacent_only_v1"
    document["rules"][0]["actions"][0]["allowedMovementKinds"] = ["adjacent_swap"]
    document["limits"]["maxMovementRadius"] = 1
    seeds.append(
        _candidate(
            document,
            family="simplified_variant",
            role="Spatial local greedy rule restricted to adjacent swaps",
            parents=("spatial_greedy_local_v1",),
            validation_note="Validated movement-allowlist and radius restriction; candidate selector and positive threshold retained.",
        )
    )
    return seeds


def _line_repair_document() -> dict[str, Any]:
    return {
        "schemaVersion": "e07.policy-dsl.v1",
        "policyId": "line_rejection_backoff_v1",
        "environment": "line1d.v1",
        "permissions": ["last_action.rejected"],
        "memory": [{"name": "frustration", "bits": 2, "initial": 0}],
        "signals": {"channels": 0, "bitsPerChannel": 0},
        "limits": {
            "maxRules": 2,
            "maxExpressionNodes": 4,
            "maxActionsPerActivation": 2,
            "maxOperationsPerActivation": 16,
            "maxMovementRadius": 0,
            "maxCandidates": 0,
        },
        "rules": [
            {
                "when": {
                    "op": "eq",
                    "left": {"obs": "last_action.rejected"},
                    "right": {"const": True},
                },
                "actions": [
                    {
                        "kind": "set_memory",
                        "register": "frustration",
                        "mode": "increment_saturating",
                        "value": None,
                    },
                    {"kind": "noop"},
                ],
            },
            {
                "when": {
                    "op": "eq",
                    "left": {"obs": "last_action.rejected"},
                    "right": {"const": False},
                },
                "actions": [
                    {
                        "kind": "set_memory",
                        "register": "frustration",
                        "mode": "decrement_saturating",
                        "value": None,
                    },
                    {"kind": "noop"},
                ],
            },
        ],
        "default": {"actions": [{"kind": "noop"}]},
    }


def _spatial_conflict_repair_document() -> dict[str, Any]:
    return {
        "schemaVersion": "e07.policy-dsl.v1",
        "policyId": "spatial_conflict_avoid_repair_v1",
        "environment": "spatial2d.v1",
        "permissions": [
            "candidate.kind",
            "candidate.lagged_conflict_count",
            "candidate.local_relation_delta",
            "last_action.rejected",
        ],
        "memory": [{"name": "frustration", "bits": 2, "initial": 0}],
        "signals": {"channels": 0, "bitsPerChannel": 0},
        "limits": {
            "maxRules": 2,
            "maxExpressionNodes": 4,
            "maxActionsPerActivation": 2,
            "maxOperationsPerActivation": 32,
            "maxMovementRadius": 2,
            "maxCandidates": 16,
        },
        "rules": [
            {
                "when": {
                    "op": "eq",
                    "left": {"obs": "last_action.rejected"},
                    "right": {"const": True},
                },
                "actions": [
                    {
                        "kind": "set_memory",
                        "register": "frustration",
                        "mode": "increment_saturating",
                        "value": None,
                    },
                    {
                        "kind": "move_candidate",
                        "selector": {
                            "mode": "argmin",
                            "field": "candidate.lagged_conflict_count",
                            "requirePositive": False,
                            "indexObservation": None,
                        },
                        "allowedMovementKinds": [
                            "adjacent_swap",
                            "short_exchange",
                            "vacancy_move",
                        ],
                    },
                ],
            },
            {
                "when": {"const": True},
                "actions": [
                    {
                        "kind": "set_memory",
                        "register": "frustration",
                        "mode": "decrement_saturating",
                        "value": None,
                    },
                    {
                        "kind": "move_candidate",
                        "selector": {
                            "mode": "argmax",
                            "field": "candidate.local_relation_delta",
                            "requirePositive": True,
                            "indexObservation": None,
                        },
                        "allowedMovementKinds": [
                            "adjacent_swap",
                            "short_exchange",
                            "vacancy_move",
                        ],
                    },
                ],
            },
        ],
        "default": {"actions": [{"kind": "noop"}]},
    }


def _random_line_document(index: int) -> dict[str, Any]:
    thresholds = [
        24 + (_digest_int("line-threshold", index, phase, side) % 208)
        for phase in range(2)
        for side in ("left", "right")
    ]
    rules = []
    cursor = 0
    for phase in range(2):
        for side, offset in (("left", -1), ("right", 1)):
            rules.append(
                {
                    "when": {
                        "all": [
                            {
                                "op": "eq",
                                "left": {"mem": "state"},
                                "right": {"const": phase},
                            },
                            {
                                "op": "eq",
                                "left": {"obs": "activation.side"},
                                "right": {"const": side},
                            },
                            {
                                "op": "lt",
                                "left": {"obs": "counter.choice_u8"},
                                "right": {"const": thresholds[cursor]},
                            },
                        ]
                    },
                    "actions": [
                        {
                            "kind": "set_memory",
                            "register": "state",
                            "mode": "toggle",
                            "value": None,
                        },
                        {"kind": "swap_relative", "offset": offset},
                    ],
                }
            )
            cursor += 1
    return {
        "schemaVersion": "e07.policy-dsl.v1",
        "policyId": f"random_line_fsm_{index:02d}_v1",
        "environment": "line1d.v1",
        "permissions": ["activation.side", "counter.choice_u8"],
        "memory": [{"name": "state", "bits": 1, "initial": index % 2}],
        "signals": {"channels": 0, "bitsPerChannel": 0},
        "limits": {
            "maxRules": 4,
            "maxExpressionNodes": 20,
            "maxActionsPerActivation": 2,
            "maxOperationsPerActivation": 32,
            "maxMovementRadius": 1,
            "maxCandidates": 0,
        },
        "rules": rules,
        "default": {
            "actions": [
                {
                    "kind": "set_memory",
                    "register": "state",
                    "mode": "toggle",
                    "value": None,
                },
                {"kind": "noop"},
            ]
        },
    }


def _random_spatial_document(index: int) -> dict[str, Any]:
    thresholds = [
        32 + (_digest_int("spatial-threshold", index, phase) % 192)
        for phase in range(2)
    ]
    movement_sets = (
        ["adjacent_swap"],
        ["vacancy_move"],
        ["adjacent_swap", "vacancy_move"],
        ["adjacent_swap", "short_exchange", "vacancy_move"],
        ["rotation"],
    )
    allowed = movement_sets[index % len(movement_sets)]
    required_radius = 2 if "short_exchange" in allowed else 1
    rules = []
    for phase in range(2):
        rules.append(
            {
                "when": {
                    "all": [
                        {
                            "op": "eq",
                            "left": {"mem": "state"},
                            "right": {"const": phase},
                        },
                        {
                            "op": "lt",
                            "left": {"obs": "counter.choice_u8"},
                            "right": {"const": thresholds[phase]},
                        },
                    ]
                },
                "actions": [
                    {
                        "kind": "set_memory",
                        "register": "state",
                        "mode": "toggle",
                        "value": None,
                    },
                    {
                        "kind": "move_candidate",
                        "selector": {
                            "mode": "index",
                            "field": None,
                            "requirePositive": False,
                            "indexObservation": "counter.choice_u8",
                        },
                        "allowedMovementKinds": allowed,
                    },
                ],
            }
        )
    return {
        "schemaVersion": "e07.policy-dsl.v1",
        "policyId": f"random_spatial_fsm_{index:02d}_v1",
        "environment": "spatial2d.v1",
        "permissions": ["candidate.kind", "counter.choice_u8"],
        "memory": [{"name": "state", "bits": 1, "initial": index % 2}],
        "signals": {"channels": 0, "bitsPerChannel": 0},
        "limits": {
            "maxRules": 2,
            "maxExpressionNodes": 8,
            "maxActionsPerActivation": 2,
            "maxOperationsPerActivation": 32,
            "maxMovementRadius": required_radius,
            "maxCandidates": 16,
        },
        "rules": rules,
        "default": {
            "actions": [
                {
                    "kind": "set_memory",
                    "register": "state",
                    "mode": "toggle",
                    "value": None,
                },
                {"kind": "noop"},
            ]
        },
    }


def _semantic_duplicate_noop_document(
    policy_id: str, *, variant: str
) -> dict[str, Any]:
    condition = (
        {
            "op": "lt",
            "left": {"obs": "counter.choice_u8"},
            "right": {"const": 0},
        }
        if variant == "lt_zero"
        else {
            "op": "gt",
            "left": {"obs": "counter.choice_u8"},
            "right": {"const": 255},
        }
    )
    return {
        "schemaVersion": "e07.policy-dsl.v1",
        "policyId": policy_id,
        "environment": "line1d.v1",
        "permissions": ["counter.choice_u8"],
        "memory": [],
        "signals": {"channels": 0, "bitsPerChannel": 0},
        "limits": {
            "maxRules": 1,
            "maxExpressionNodes": 2,
            "maxActionsPerActivation": 1,
            "maxOperationsPerActivation": 8,
            "maxMovementRadius": 0,
            "maxCandidates": 0,
        },
        "rules": [{"when": condition, "actions": [{"kind": "noop"}]}],
        "default": {"actions": [{"kind": "noop"}]},
    }


def build_candidate_seeds() -> tuple[CandidateSeed, ...]:
    """Construct the deterministic pre-deduplication candidate set."""

    faithful = []
    for policy_id in FAITHFUL_POLICY_MAP:
        faithful.append(
            _candidate(
                _baseline_document(policy_id),
                family="faithful",
                role=f"Faithful proposal-level {FAITHFUL_POLICY_MAP[policy_id].value} seed",
                validation_note="Exact S01 baseline source; S03 revalidates canonical hash and native proposal behavior on training-derived probes.",
            )
        )

    repair = [
        _candidate(
            _baseline_document("nudge_signal_repair_v1"),
            family="repair",
            role="Bounded rejection/nudge signal repair seed",
            validation_note="Exact S01 repair encoding; behavior is evaluated, efficacy is not claimed.",
        ),
        _candidate(
            _baseline_document("spatial_memory_repair_v1"),
            family="repair",
            role="Bounded actor-local spatial memory repair seed",
            validation_note="Exact S01 repair encoding; typed synthetic candidate fixtures only, with no E06 task binding inferred.",
        ),
        _candidate(
            _line_repair_document(),
            family="repair",
            role="Hand-designed rejection backoff memory seed",
            validation_note="Hand-designed finite-state repair seed; no movement and no task-efficacy claim.",
        ),
        _candidate(
            _spatial_conflict_repair_document(),
            family="repair",
            role="Hand-designed lagged-conflict avoidance repair seed",
            validation_note="Hand-designed finite-state typed candidate seed; no E06 binding or repair-efficacy claim.",
        ),
    ]

    random_seeds = [
        _candidate(
            _random_line_document(index),
            family="random_finite_state",
            role="Deterministically generated random 1D finite-state controller",
            validation_note="Generation is counter-addressed and reproducible; runtime counter input is bounded uint8.",
        )
        for index in range(8)
    ] + [
        _candidate(
            _random_spatial_document(index),
            family="random_finite_state",
            role="Deterministically generated random 2D finite-state controller",
            validation_note="Generation is counter-addressed and reproducible; candidates remain opaque typed fixtures.",
        )
        for index in range(8)
    ]

    duplicate_probes = [
        _candidate(
            _semantic_duplicate_noop_document(
                "random_line_noop_guard_a_v1", variant="lt_zero"
            ),
            family="random_finite_state",
            role="Deliberate bounded-semantic duplicate detector fixture",
            validation_note="Counter uint8 is never less than zero; used to test finite-corpus semantic deduplication.",
        ),
        _candidate(
            _semantic_duplicate_noop_document(
                "random_line_noop_guard_b_v1", variant="gt_255"
            ),
            family="random_finite_state",
            role="Deliberate bounded-semantic duplicate detector fixture",
            validation_note="Counter uint8 is never greater than 255; expected to deduplicate with guard A.",
        ),
    ]
    candidates = tuple(
        faithful + _simplified_candidates() + random_seeds + repair + duplicate_probes
    )
    if len({candidate.policy_id for candidate in candidates}) != len(candidates):
        raise AssertionError("candidate policy IDs must be unique")
    return candidates


def _permutation(length: int, *parts: object) -> list[int]:
    return sorted(range(length), key=lambda index: (_digest_int(*parts, index), index))


def _line_values(record: ScenarioRecord) -> tuple[int, ...]:
    if "values" in record.public_parameters:
        return tuple(int(value) for value in record.public_parameters["values"])
    return tuple(range(int(record.public_parameters["n"])))


def _line_probe(record: ScenarioRecord, index: int) -> TrainingProbe:
    values = _line_values(record)
    n = len(values)
    direction = Direction(str(record.public_parameters.get("direction", "ascending")))
    faults = [FaultMode.NORMAL] * n
    if record.task_id == "e07_s02_faults_1d":
        faults[int(record.public_parameters["faultIndex"])] = FaultMode(
            str(record.public_parameters["faultMode"])
        )
    native_policies = [Policy.BUBBLE] * n
    if record.task_id == "e07_s02_chimera_1d":
        half = n // 2
        native_policies = [Policy.BUBBLE] * half + [Policy.INSERTION] * (n - half)
    cells = [
        Cell(
            f"c{cell_index}",
            value,
            native_policies[cell_index],
            direction,
            faults[cell_index],
        )
        for cell_index, value in enumerate(values)
    ]
    occupancy = tuple(
        f"c{cell_index}"
        for cell_index in _permutation(n, record.scenario_id, "occupancy", index)
    )
    normal_ids = [
        cell_id
        for cell_id in occupancy
        if cells[int(cell_id[1:])].fault == FaultMode.NORMAL
    ]
    actor_id = normal_ids[
        _digest_int(record.scenario_id, "actor", index) % len(normal_ids)
    ]
    scenario = Scenario.create(
        cells,
        initial_occupancy=occupancy,
        generation_key=f"E07/S03/{record.scenario_id}/{index}",
    )
    state = initial_state(scenario)
    position = state.occupancy.index(actor_id)
    cursor_code = _digest_int(record.scenario_id, "cursor", index) % (n + 4)
    cursor = cursor_code - 2
    state.selection_cursors[actor_id] = cursor
    side = ("left", "right")[_digest_int(record.scenario_id, "side", index) % 2]
    cells_by_id = scenario.cell_map
    actor = cells_by_id[actor_id]
    left = cells_by_id[state.occupancy[position - 1]] if position > 0 else None
    right = cells_by_id[state.occupancy[position + 1]] if position + 1 < n else None
    prefix, prefix_reads, prefix_comparisons = _prefix_is_ordered(
        scenario, state, position, direction
    )
    cursor_in_bounds = 0 <= cursor < n
    target = cells_by_id[state.occupancy[cursor]] if cursor_in_bounds else None
    available = {
        "activation.side": side,
        "own.value": int(actor.value),
        "own.position": position,
        "own.direction": direction.value,
        "neighbor.left.exists": left is not None,
        "neighbor.left.value": int(left.value) if left is not None else 0,
        "neighbor.left.movable": left is not None and left.fault != FaultMode.STUCK,
        "neighbor.right.exists": right is not None,
        "neighbor.right.value": int(right.value) if right is not None else 0,
        "neighbor.right.movable": right is not None and right.fault != FaultMode.STUCK,
        "line.prefix_ordered": prefix,
        "selection.cursor_in_bounds": cursor_in_bounds,
        "selection.cursor_at_actor": cursor == position,
        "selection.target.value": int(target.value) if target is not None else 0,
        "selection.target.stuck": target is not None
        and target.fault == FaultMode.STUCK,
        "last_action.rejected": bool(
            _digest_int(record.scenario_id, "rejected", index) % 3 == 0
        ),
        "repair.nudge_count": int(_digest_int(record.scenario_id, "nudge", index) % 4),
        "signal.neighbor_sum_u8": int(
            _digest_int(record.scenario_id, "signal", index) % 16
        ),
        "counter.choice_u8": int(
            _digest_int(record.scenario_id, "choice", index) % 256
        ),
    }
    return TrainingProbe(
        probe_id=f"{record.scenario_id}:probe:{index:04d}",
        task_id=record.task_id,
        scenario_id=record.scenario_id,
        environment="line1d.v1",
        available_scalars=MappingProxyType(available),
        available_candidates=(),
        scenario=scenario,
        actor_id=actor_id,
        side=side,
        selection_cursor=cursor,
        prefix_reads=prefix_reads,
        prefix_comparisons=prefix_comparisons,
    )


def _spatial_probe(record: ScenarioRecord, index: int) -> TrainingProbe:
    count = int(_digest_int(record.scenario_id, "candidate-count", index) % 17)
    kinds = ("adjacent_swap", "vacancy_move", "short_exchange", "rotation")
    candidates = []
    for candidate_index in range(count):
        candidates.append(
            MappingProxyType(
                {
                    "key": f"opaque-{index:04d}-{candidate_index:02d}",
                    "candidate.kind": kinds[
                        _digest_int(record.scenario_id, index, candidate_index, "kind")
                        % len(kinds)
                    ],
                    "candidate.local_relation_delta": int(
                        _digest_int(
                            record.scenario_id, index, candidate_index, "relation"
                        )
                        % 17
                    )
                    - 8,
                    "candidate.natural_boundary_delta": int(
                        _digest_int(
                            record.scenario_id, index, candidate_index, "boundary"
                        )
                        % 9
                    )
                    - 4,
                    "candidate.gradient_delta": int(
                        _digest_int(
                            record.scenario_id, index, candidate_index, "gradient"
                        )
                        % 17
                    )
                    - 8,
                    "candidate.lagged_conflict_count": int(
                        _digest_int(
                            record.scenario_id, index, candidate_index, "conflict"
                        )
                        % 4
                    ),
                    "candidate.movement_cost": int(
                        1
                        + _digest_int(
                            record.scenario_id, index, candidate_index, "cost"
                        )
                        % 8
                    ),
                }
            )
        )
    available = {
        "own.token": ("A", "B", "C")[
            _digest_int(record.scenario_id, "token", index) % 3
        ],
        "own.local_relation_utility": int(
            _digest_int(record.scenario_id, "utility", index) % 25
        ),
        "natural.boundary_signal": int(
            _digest_int(record.scenario_id, "boundary-signal", index) % 7
        ),
        "gradient.current_u8": int(
            _digest_int(record.scenario_id, "gradient-current", index) % 256
        ),
        "candidate.count": count,
        "last_action.rejected": bool(
            _digest_int(record.scenario_id, "rejected", index) % 3 == 0
        ),
        "signal.neighbor_sum_u8": int(
            _digest_int(record.scenario_id, "signal", index) % 16
        ),
        "counter.choice_u8": int(
            _digest_int(record.scenario_id, "choice", index) % 256
        ),
    }
    return TrainingProbe(
        probe_id=f"{record.scenario_id}:typed-probe:{index:04d}",
        task_id=record.task_id,
        scenario_id=record.scenario_id,
        environment="spatial2d.v1",
        available_scalars=MappingProxyType(available),
        available_candidates=tuple(candidates),
    )


def build_training_probes(
    training_records: Sequence[ScenarioRecord],
    probes_per_record: int = DEFAULT_PROBES_PER_RECORD,
) -> tuple[TrainingProbe, ...]:
    probes: list[TrainingProbe] = []
    for record in training_records:
        if record.split != Split.TRAIN:
            raise ValueError("non-training record passed to S03 evaluator")
        if record.task_id in LINE_PROBE_TASKS:
            probes.extend(
                _line_probe(record, index) for index in range(probes_per_record)
            )
        elif record.task_id in SPATIAL_PROBE_TASKS:
            probes.extend(
                _spatial_probe(record, index) for index in range(probes_per_record)
            )
        elif record.task_id not in UNBOUND_TASKS:
            raise ValueError(f"unclassified S03 task binding: {record.task_id}")
    return tuple(probes)


def _observation(policy: CompiledPolicy, probe: TrainingProbe) -> dict[str, Any]:
    scalar_permissions = sorted(
        permission
        for permission in policy.permissions
        if not permission.startswith("candidate.")
    )
    candidate_permissions = sorted(
        permission
        for permission in policy.permissions
        if permission.startswith("candidate.")
    )
    candidates = []
    if candidate_permissions:
        for item in probe.available_candidates:
            candidates.append(
                {
                    "key": item["key"],
                    **{
                        permission: item[permission]
                        for permission in candidate_permissions
                    },
                }
            )
    return {
        "scalars": {
            permission: probe.available_scalars[permission]
            for permission in scalar_permissions
        },
        "candidates": candidates,
    }


def _shadow_reference(
    probe: TrainingProbe, reference_policy: Policy
) -> tuple[Scenario, Any]:
    assert probe.scenario is not None and probe.actor_id is not None
    cells = [
        replace(
            cell,
            policy=(
                reference_policy if cell.cell_id == probe.actor_id else cell.policy
            ),
        )
        for cell in probe.scenario.cells
    ]
    scenario = Scenario.create(
        cells,
        initial_occupancy=probe.scenario.initial_occupancy,
        generation_key=f"{probe.scenario.generation_key}/shadow/{reference_policy.value}",
    )
    state = initial_state(scenario)
    assert probe.selection_cursor is not None
    state.selection_cursors[probe.actor_id] = probe.selection_cursor
    return scenario, state


def _faithful_match(
    policy: CompiledPolicy, probe: TrainingProbe, actions: Sequence[Mapping[str, Any]]
) -> bool:
    reference_policy = FAITHFUL_POLICY_MAP[policy.policy_id]
    scenario, state = _shadow_reference(probe, reference_policy)
    reference = cell_view_proposal(
        scenario,
        state,
        probe.actor_id,  # type: ignore[arg-type]
        side=probe.side if reference_policy == Policy.BUBBLE else None,
    )
    terminal = [
        action
        for action in actions
        if action["kind"]
        in {"noop", "swap_relative", "swap_cursor", "advance_cursor", "move_candidate"}
    ]
    if len(terminal) != 1:
        return False
    action = terminal[0]
    if reference.kind == ProposalKind.NO_OP:
        return action["kind"] == "noop"
    if reference.kind == ProposalKind.SWAP:
        if action["kind"] == "swap_relative":
            return (
                state.occupancy.index(probe.actor_id) + int(action["offset"])
                == reference.target_pos
            )
        return (
            action["kind"] == "swap_cursor"
            and state.selection_cursors[probe.actor_id] == reference.target_pos
        )
    return (
        reference.kind == ProposalKind.MEMORY_UPDATE
        and action["kind"] == "advance_cursor"
        and state.selection_cursors[probe.actor_id] + int(action["delta"])
        == reference.new_cursor
    )


def _licensed_capabilities(policy: CompiledPolicy) -> dict[str, Any]:
    prefix = "line.prefix_ordered" in policy.permissions
    cursor = any(
        permission.startswith("selection.") for permission in policy.permissions
    )
    long_range = any(
        action["kind"] == "swap_cursor"
        for _, actions in policy.rules
        for action in actions
    ) or any(action["kind"] == "swap_cursor" for action in policy.default_actions)
    return {
        "licensedPrefixPredicate": prefix,
        "engineOwnedSelectionCursor": cursor,
        "licensedLongRangeCursorAction": long_range,
        "trustedOpaqueSpatialCandidates": any(
            permission.startswith("candidate.") for permission in policy.permissions
        ),
        "nativeCostsReplaced": False,
        "scalarCapabilityCost": None,
    }


def _evaluate_candidate(
    candidate: CandidateSeed, probes: Sequence[TrainingProbe]
) -> dict[str, Any]:
    policy = candidate.compile()
    selected = [probe for probe in probes if probe.environment == policy.environment]
    state: Mapping[str, int] | None = None
    trace = []
    action_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    operations: list[int] = []
    faithful_matches = 0
    faithful_mismatches = 0
    ledger = {
        "evaluatedActivations": 0,
        "declaredScalarProjectionFields": 0,
        "dslInterpreterOperations": 0,
        "licensedPrefixPredicateEvaluations": 0,
        "licensedPrefixValueReads": 0,
        "licensedPrefixValueComparisons": 0,
        "engineCursorStateReads": 0,
        "engineCursorTargetProjectionReads": 0,
        "engineCursorAdvanceActions": 0,
        "engineCursorSwapActions": 0,
        "licensedLongRangeRequestedDistance": 0,
        "licensedLongRangeMaximumRequestedDistance": 0,
        "opaqueCandidateRecordsProjected": 0,
    }
    for probe in selected:
        observation = _observation(policy, probe)
        result = execute_policy(policy, observation, state)
        state = result.memory
        actions = [_jsonable(action) for action in result.actions]
        terminal_kind = next(
            action["kind"]
            for action in actions
            if action["kind"]
            in {
                "noop",
                "swap_relative",
                "swap_cursor",
                "advance_cursor",
                "move_candidate",
            }
        )
        action_counts[terminal_kind] = action_counts.get(terminal_kind, 0) + 1
        task_counts[probe.task_id] = task_counts.get(probe.task_id, 0) + 1
        operations.append(result.operation_count)
        ledger["evaluatedActivations"] += 1
        ledger["declaredScalarProjectionFields"] += len(observation["scalars"])
        ledger["dslInterpreterOperations"] += result.operation_count
        ledger["opaqueCandidateRecordsProjected"] += len(observation["candidates"])
        if "line.prefix_ordered" in policy.permissions:
            ledger["licensedPrefixPredicateEvaluations"] += 1
            ledger["licensedPrefixValueReads"] += probe.prefix_reads
            ledger["licensedPrefixValueComparisons"] += probe.prefix_comparisons
        if any(
            permission.startswith("selection.") for permission in policy.permissions
        ):
            ledger["engineCursorStateReads"] += 1
            if bool(probe.available_scalars.get("selection.cursor_in_bounds", False)):
                ledger["engineCursorTargetProjectionReads"] += 1
        for action in actions:
            if action["kind"] == "advance_cursor":
                ledger["engineCursorAdvanceActions"] += 1
            elif action["kind"] == "swap_cursor":
                ledger["engineCursorSwapActions"] += 1
                assert probe.selection_cursor is not None and probe.actor_id is not None
                assert probe.scenario is not None
                distance = abs(
                    probe.selection_cursor
                    - probe.scenario.initial_occupancy.index(probe.actor_id)
                )
                ledger["licensedLongRangeRequestedDistance"] += distance
                ledger["licensedLongRangeMaximumRequestedDistance"] = max(
                    ledger["licensedLongRangeMaximumRequestedDistance"], distance
                )
        if policy.policy_id in FAITHFUL_POLICY_MAP and probe.environment == "line1d.v1":
            if _faithful_match(policy, probe, result.actions):
                faithful_matches += 1
            else:
                faithful_mismatches += 1
        trace.append(
            {
                "probeId": probe.probe_id,
                "actions": actions,
                "memory": _jsonable(result.memory),
                "signals": _jsonable(result.emitted_signals),
                "operations": result.operation_count,
            }
        )
    capabilities = _licensed_capabilities(policy)
    interface = {
        "environment": policy.environment,
        "permissions": sorted(policy.permissions),
        "memory": [
            {"name": item.name, "bits": item.bits, "initial": item.initial}
            for item in policy.memory
        ],
        "signals": {
            "channels": policy.signal_channels,
            "bitsPerChannel": policy.signal_bits_per_channel,
        },
        "maxMovementRadius": policy.max_movement_radius,
        "maxCandidates": policy.max_candidates,
        "licensedCapabilities": capabilities,
    }
    behavior_trace = [
        {
            "actions": item["actions"],
            "memory": item["memory"],
            "signals": item["signals"],
            "operations": item["operations"],
        }
        for item in trace
    ]
    action_trace = [item["actions"] for item in trace]
    semantic_signature = _hash(
        "E07/S03/bounded-semantic-signature/v1",
        {"interface": interface, "trace": behavior_trace},
    )
    return {
        "schemaVersion": "e07.s03.seed-evaluation.v1",
        "researchStepId": "S03",
        "policyId": policy.policy_id,
        "policySha256": policy.policy_sha256,
        "family": candidate.family,
        "environment": policy.environment,
        "probeContract": PROBE_VERSION,
        "probeCount": len(selected),
        "taskProbeCounts": dict(sorted(task_counts.items())),
        "actionCounts": dict(sorted(action_counts.items())),
        "operationCount": {
            "minimum": min(operations),
            "maximum": max(operations),
            "sum": sum(operations),
            "mean": sum(operations) / len(operations),
        },
        "faithfulReference": {
            "applicable": policy.policy_id in FAITHFUL_POLICY_MAP,
            "matches": faithful_matches,
            "mismatches": faithful_mismatches,
            "scope": "E01 proposal kind, target, and cursor update on training-derived valid-state shadow probes",
        },
        "adapterProjectionLedger": ledger,
        "licensedCapabilities": capabilities,
        "behaviorSha256": _hash("E07/S03/behavior-trace/v1", behavior_trace),
        "actionSha256": _hash("E07/S03/action-trace/v1", action_trace),
        "boundedSemanticSha256": semantic_signature,
        "traceRecordCount": len(trace),
        "traceRetained": False,
        "deterministicReplay": True,
    }


def _task_binding_matrix(training_records: Sequence[ScenarioRecord]) -> dict[str, Any]:
    rows = []
    for record in training_records:
        if record.task_id in LINE_PROBE_TASKS:
            status = "training_derived_typed_probe"
            native_binding = "E01 proposal-shadow parity for faithful policies only; no seed episode outcome"
        elif record.task_id in SPATIAL_PROBE_TASKS:
            status = "synthetic_typed_fixture_only"
            native_binding = "No E06 adapter inferred; opaque candidates are deterministic DSL fixtures only"
        else:
            status = "unbound_preserved"
            native_binding = "No E05 DSL binding inferred from matching fields; native training baseline guard only"
        rows.append(
            {
                "taskId": record.task_id,
                "scenarioId": record.scenario_id,
                "split": record.split.value,
                "bindingStatus": status,
                "nativeBindingBoundary": native_binding,
                "outcomeUsedForSeedEvaluation": False,
                "nativeLegalityActionEventStoppingCensoringContractsChanged": False,
            }
        )
    return {
        "schemaVersion": "e07.s03.task-binding-matrix.v1",
        "researchStepId": "S03",
        "rows": rows,
        "arbitraryE05BindingsInferred": 0,
        "arbitraryE06BindingsInferred": 0,
        "seedEpisodeOutcomesRead": 0,
    }


def _coverage(final_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, int] = {}
    environments: dict[str, int] = {}
    features = {
        "memory": 0,
        "signals": 0,
        "counterInput": 0,
        "licensedPrefix": 0,
        "engineCursor": 0,
        "licensedLongRange": 0,
        "opaqueCandidates": 0,
        "radiusOne": 0,
        "radiusTwoOrMore": 0,
        "nonmoving": 0,
    }
    for record in final_records:
        families[record["family"]] = families.get(record["family"], 0) + 1
        environments[record["environment"]] = (
            environments.get(record["environment"], 0) + 1
        )
        complexity = record["complexity"]
        capabilities = record["licensedCapabilities"]
        features["memory"] += complexity["persistentMemoryBits"] > 0
        features["signals"] += complexity["outboundSignalBits"] > 0
        features["counterInput"] += "counter.choice_u8" in record["permissions"]
        features["licensedPrefix"] += capabilities["licensedPrefixPredicate"]
        features["engineCursor"] += capabilities["engineOwnedSelectionCursor"]
        features["licensedLongRange"] += capabilities["licensedLongRangeCursorAction"]
        features["opaqueCandidates"] += capabilities["trustedOpaqueSpatialCandidates"]
        radius = complexity["maxMovementRadius"]
        features["nonmoving"] += radius == 0
        features["radiusOne"] += radius == 1
        features["radiusTwoOrMore"] += radius >= 2
    required = {
        "faithfulBubbleInsertionSelection": all(
            policy_id in {record["policyId"] for record in final_records}
            for policy_id in FAITHFUL_POLICY_MAP
        ),
        "bothTypedEnvironments": set(environments) == {"line1d.v1", "spatial2d.v1"},
        "allFourSeedFamilies": set(families)
        == {"faithful", "simplified_variant", "random_finite_state", "repair"},
        "memoryAndMemoryless": features["memory"] > 0
        and features["memory"] < len(final_records),
        "communicationPresent": features["signals"] > 0,
        "counterDrivenPresent": features["counterInput"] > 0,
        "licensedPrefixPresent": features["licensedPrefix"] > 0,
        "engineCursorPresent": features["engineCursor"] > 0,
        "licensedLongRangePresent": features["licensedLongRange"] > 0,
        "opaqueCandidatePresent": features["opaqueCandidates"] > 0,
        "multipleMovementRadii": all(
            features[key] > 0 for key in ("nonmoving", "radiusOne", "radiusTwoOrMore")
        ),
    }
    return {
        "schemaVersion": "e07.s03.seed-coverage.v1",
        "researchStepId": "S03",
        "scope": "structural seed coverage only; these are not frozen S04 objectives or archive descriptors",
        "seedCount": len(final_records),
        "familyCounts": dict(sorted(families.items())),
        "environmentCounts": dict(sorted(environments.items())),
        "structuralFeatureCounts": features,
        "coverageChecks": required,
        "baselineStructuralCoveragePassed": all(required.values()),
    }


def build_s03_evidence(
    task_registry_path: str | Path,
    split_manifest_path: str | Path,
    *,
    probes_per_record: int = DEFAULT_PROBES_PER_RECORD,
) -> dict[str, Any]:
    """Build all in-memory deterministic S03 seed evidence."""

    task_metadata, split_metadata, training_records = load_training_records(
        task_registry_path, split_manifest_path
    )
    probes = build_training_probes(training_records, probes_per_record)
    candidates = build_candidate_seeds()
    first_evaluations = [
        _evaluate_candidate(candidate, probes) for candidate in candidates
    ]
    second_evaluations = [
        _evaluate_candidate(candidate, probes) for candidate in candidates
    ]
    if canonical_json_bytes(first_evaluations) != canonical_json_bytes(
        second_evaluations
    ):
        raise AssertionError("seed evaluation replay diverged")

    groups: dict[str, list[int]] = {}
    for index, evaluation in enumerate(first_evaluations):
        groups.setdefault(evaluation["boundedSemanticSha256"], []).append(index)
    retained_indices: list[int] = []
    equivalence_groups = []
    for signature, indices in sorted(groups.items()):
        retained = indices[0]
        retained_indices.append(retained)
        equivalence_groups.append(
            {
                "boundedSemanticSha256": signature,
                "memberPolicyIds": [candidates[index].policy_id for index in indices],
                "memberPolicySha256": [
                    first_evaluations[index]["policySha256"] for index in indices
                ],
                "retainedPolicyId": candidates[retained].policy_id,
                "discardedPolicyIds": [
                    candidates[index].policy_id for index in indices[1:]
                ],
                "duplicateDetected": len(indices) > 1,
            }
        )
    retained_indices.sort()
    records = []
    evaluations = []
    complexity_records = []
    for index in retained_indices:
        candidate = candidates[index]
        policy = candidate.compile()
        evaluation = first_evaluations[index]
        policy_document = policy_to_dict(policy)
        record = {
            "schemaVersion": SEED_LIBRARY_VERSION,
            "researchStepId": "S03",
            "policyId": policy.policy_id,
            "policySha256": policy.policy_sha256,
            "canonicalPolicyBytesSha256": hashlib.sha256(
                policy.canonical_json
            ).hexdigest(),
            "family": candidate.family,
            "role": candidate.role,
            "parents": list(candidate.parents),
            "environment": policy.environment,
            "permissions": sorted(policy.permissions),
            "complexity": policy.complexity.to_dict(),
            "licensedCapabilities": evaluation["licensedCapabilities"],
            "boundedSemanticSha256": evaluation["boundedSemanticSha256"],
            "variantValidation": candidate.validation_note,
            "evaluationScope": (
                "training-derived E01 proposal shadows"
                if policy.environment == "line1d.v1"
                else "training-derived typed synthetic spatial fixtures; not an E06 binding"
            ),
            "canonicalPolicy": policy_document,
        }
        records.append(record)
        evaluations.append(evaluation)
        complexity_records.append(
            {
                "schemaVersion": "e07.s03.complexity-accounting.v1",
                "researchStepId": "S03",
                "policyId": policy.policy_id,
                "policySha256": policy.policy_sha256,
                "structuralComplexity": policy.complexity.to_dict(),
                "licensedCapabilities": evaluation["licensedCapabilities"],
                "trainingAdapterProjectionLedger": evaluation[
                    "adapterProjectionLedger"
                ],
                "scalarTotalDefined": False,
                "nativePredecessorCostLedgerReplaced": False,
            }
        )

    canonical_roundtrip = all(
        compile_policy(record["canonicalPolicy"]).policy_sha256
        == record["policySha256"]
        for record in records
    )
    faithful = [
        evaluation
        for evaluation in evaluations
        if evaluation["faithfulReference"]["applicable"]
    ]
    coverage = _coverage(records)
    task_bindings = _task_binding_matrix(training_records)
    duplicate_groups = [
        group for group in equivalence_groups if group["duplicateDetected"]
    ]
    validation = {
        "schemaVersion": "e07.s03.validation-summary.v1",
        "researchStepId": "S03",
        "checks": {
            "candidateCountExpected": len(candidates) == 31,
            "deduplicatedSeedCountExpected": len(records) == 30,
            "allCanonicalHashesRoundTrip": canonical_roundtrip,
            "allPolicyHashesUniqueAfterDedup": len(
                {record["policySha256"] for record in records}
            )
            == len(records),
            "faithfulBehaviorExact": all(
                evaluation["faithfulReference"]["mismatches"] == 0
                and evaluation["faithfulReference"]["matches"] > 0
                for evaluation in faithful
            ),
            "complexityWithinDslBounds": all(
                record["complexity"]["worstCaseOperations"]
                <= record["canonicalPolicy"]["limits"]["maxOperationsPerActivation"]
                for record in records
            ),
            "baselineStructuralCoverage": coverage["baselineStructuralCoveragePassed"],
            "semanticDuplicateDetected": len(duplicate_groups) >= 1,
            "deliberateDuplicateCollapsed": any(
                set(group["memberPolicyIds"])
                == {
                    "random_line_noop_guard_a_v1",
                    "random_line_noop_guard_b_v1",
                }
                for group in duplicate_groups
            ),
            "deterministicReplay": canonical_json_bytes(first_evaluations)
            == canonical_json_bytes(second_evaluations),
            "trainingRecordsOnly": all(
                record.split == Split.TRAIN for record in training_records
            ),
            "noValidationOutcomeEvaluations": True,
            "noConfirmationOutcomeEvaluations": True,
            "noArbitraryE05BindingsInferred": True,
            "noArbitraryE06BindingsInferred": True,
            "nativeCostFamiliesRemainUnscalarized": True,
        },
        "counts": {
            "candidateSeeds": len(candidates),
            "retainedSeeds": len(records),
            "discardedSemanticDuplicates": len(candidates) - len(records),
            "trainingRecords": len(training_records),
            "lineTrainingProbes": sum(
                probe.environment == "line1d.v1" for probe in probes
            ),
            "spatialTypedTrainingProbes": sum(
                probe.environment == "spatial2d.v1" for probe in probes
            ),
            "faithfulProposalComparisons": sum(
                evaluation["faithfulReference"]["matches"]
                + evaluation["faithfulReference"]["mismatches"]
                for evaluation in faithful
            ),
            "faithfulProposalMismatches": sum(
                evaluation["faithfulReference"]["mismatches"] for evaluation in faithful
            ),
            "boundedSemanticGroups": len(equivalence_groups),
            "duplicateGroups": len(duplicate_groups),
        },
    }
    validation["success"] = all(validation["checks"].values())
    return {
        "taskMetadata": _jsonable(task_metadata),
        "splitMetadata": _jsonable(split_metadata),
        "trainingRecords": [record.public_dict() for record in training_records],
        "probeSummary": {
            "schemaVersion": PROBE_VERSION,
            "probesPerEligibleTrainingRecord": probes_per_record,
            "totalProbes": len(probes),
            "lineProbeCount": sum(probe.environment == "line1d.v1" for probe in probes),
            "spatialTypedProbeCount": sum(
                probe.environment == "spatial2d.v1" for probe in probes
            ),
            "unboundTrainingRecords": sorted(
                record.scenario_id
                for record in training_records
                if record.task_id in UNBOUND_TASKS
            ),
            "outcomesRead": 0,
        },
        "seedRecords": records,
        "evaluations": evaluations,
        "complexityRecords": complexity_records,
        "equivalence": {
            "schemaVersion": SEMANTIC_EQUIVALENCE_VERSION,
            "researchStepId": "S03",
            "criteria": {
                "canonicalDuplicate": "identical S01 canonical policy SHA-256",
                "boundedSemanticDuplicate": "same typed environment, permissions, memory/signal contract, licensed capabilities, movement/candidate bounds, and identical action/memory/signal/operation trace on every deterministic S03 training probe",
                "actionOnlyNearDuplicate": "same action trace but different permission, memory, licensed-resource, or operation contract; reported but not collapsed",
                "proofBoundary": "finite deterministic corpus equivalence, not universal program equivalence",
            },
            "groups": equivalence_groups,
            "duplicateGroups": duplicate_groups,
            "discardedPolicyIds": [
                policy_id
                for group in duplicate_groups
                for policy_id in group["discardedPolicyIds"]
            ],
        },
        "coverage": coverage,
        "taskBindings": task_bindings,
        "validation": validation,
    }
