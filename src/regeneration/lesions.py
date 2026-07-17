"""E05 S03 explicit lesion operators over exact S02 checkpoints.

The operator layer is intentionally independent of the E01 fixed-identity run
engine.  This lets S03 represent count-changing tasks without pretending they
are permutations or silently coercing them back into the original scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import jsonschema

from reference_simulator.model import (
    Cell,
    Direction,
    FaultMode,
    LEDGER_FIELDS,
    Policy,
    Scenario,
    canonical_json_bytes,
    sha256_json,
)
from reference_simulator.rng import bounded

from .tasks import Checkpoint, strict_unequal_inversions, target_values


BENCHMARK_VERSION = "E05-lesion-library-v1"
LESION_SPEC_SCHEMA_VERSION = "e05.s03.lesion-spec.v1"
LESION_FIXTURE_SCHEMA_VERSION = "e05.s03.lesion-fixture.v1"
LESION_STATE_SCHEMA_VERSION = "e05.s03.lesion-state.v1"
SCRAMBLE_STREAM = "lesion_local_scramble_s03_v1"

OPERATOR_IDS = (
    "freeze_stuck_one_central_v1",
    "segment_reversal_central_v1",
    "local_scramble_sattolo_v1",
    "block_transposition_adjacent_equal_v1",
    "duplication_tandem_v1",
    "deletion_central_v1",
    "insertion_out_of_place_max_v1",
)
PROHIBITED_OPERATOR_IDS = ("s01_validation_adjacent_swap_v1",)


LESION_SPEC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s03/lesion-spec.schema.json",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "schemaVersion",
        "researchStepId",
        "benchmarkVersion",
        "frozenQuestion",
        "inherits",
        "prohibitedOperators",
        "operators",
        "identityContract",
        "targetContract",
        "pairingContract",
        "validationPanel",
        "severityContract",
        "claimBoundary",
    ],
    "properties": {
        "schemaVersion": {"const": LESION_SPEC_SCHEMA_VERSION},
        "researchStepId": {"const": "S03"},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "frozenQuestion": {"type": "string", "minLength": 20},
        "inherits": {"type": "object"},
        "prohibitedOperators": {
            "type": "array",
            "contains": {"const": "s01_validation_adjacent_swap_v1"},
        },
        "operators": {
            "type": "array",
            "minItems": 7,
            "maxItems": 7,
            "items": {
                "type": "object",
                "required": [
                    "operatorId",
                    "family",
                    "countSemantics",
                    "targetProfile",
                    "metricProfile",
                    "reversibility",
                    "parameterRule",
                ],
            },
        },
        "identityContract": {"type": "object"},
        "targetContract": {"type": "object"},
        "pairingContract": {"type": "object"},
        "validationPanel": {"type": "object"},
        "severityContract": {"type": "object"},
        "claimBoundary": {"type": "string", "minLength": 30},
    },
}


LESION_FIXTURE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://eidosoma.local/schemas/e05/s03/lesion-fixture.schema.json",
    "type": "object",
    "additionalProperties": True,
    "required": [
        "schemaVersion",
        "benchmarkVersion",
        "lesionCaseId",
        "lesionPairId",
        "operatorId",
        "family",
        "sourceScenarioId",
        "s01PairingBlockId",
        "timingConditionId",
        "preInjuryStateHash",
        "preLesionStateHash",
        "postLesionStateHash",
        "identityContract",
        "targetCorrespondence",
        "severity",
        "reversibility",
        "runtimeCompatibility",
        "parameters",
    ],
    "properties": {
        "schemaVersion": {"const": LESION_FIXTURE_SCHEMA_VERSION},
        "benchmarkVersion": {"const": BENCHMARK_VERSION},
        "lesionCaseId": {"type": "string", "pattern": "^e05s03:[0-9a-f]{64}$"},
        "lesionPairId": {"type": "string", "pattern": "^e05lp3:[0-9a-f]{64}$"},
        "operatorId": {"enum": list(OPERATOR_IDS)},
        "family": {"type": "string"},
        "sourceScenarioId": {"type": "string", "pattern": "^r1:[0-9a-f]{64}$"},
        "s01PairingBlockId": {"type": "string", "pattern": "^e05pb1:[0-9a-f]{64}$"},
        "timingConditionId": {"type": "string"},
        "preInjuryStateHash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "preLesionStateHash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "postLesionStateHash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "identityContract": {"type": "object"},
        "targetCorrespondence": {"type": "object"},
        "severity": {"type": "object"},
        "reversibility": {"type": "object"},
        "runtimeCompatibility": {"type": "object"},
        "parameters": {"type": "object"},
    },
}


@dataclass(frozen=True, slots=True)
class LesionState:
    """Exact checkpoint state plus an explicit, possibly changed, cell set."""

    source_scenario_id: str
    source_checkpoint_hash: str
    cells: tuple[Cell, ...]
    occupancy: tuple[str, ...]
    selection_cursors: tuple[tuple[str, int], ...]
    activation_count: int
    stream_counters: tuple[tuple[str, int], ...]
    ledger: tuple[tuple[str, int], ...]

    @classmethod
    def from_checkpoint(
        cls, scenario: Scenario, checkpoint: Checkpoint
    ) -> "LesionState":
        state = cls(
            source_scenario_id=scenario.scenario_id,
            source_checkpoint_hash=checkpoint.state_hash,
            cells=tuple(sorted(scenario.cells, key=lambda item: item.cell_id)),
            occupancy=tuple(checkpoint.occupancy),
            selection_cursors=tuple(checkpoint.selection_cursors),
            activation_count=checkpoint.activation_count,
            stream_counters=tuple(checkpoint.stream_counters),
            ledger=tuple(checkpoint.ledger),
        )
        state.validate()
        return state

    @property
    def cell_map(self) -> dict[str, Cell]:
        return {cell.cell_id: cell for cell in self.cells}

    @property
    def direction(self) -> Direction:
        directions = {cell.direction for cell in self.cells}
        if len(directions) != 1:
            raise ValueError("S03 fixtures require one homogeneous target direction")
        return next(iter(directions))

    @property
    def values(self) -> tuple[int | float, ...]:
        cells = self.cell_map
        return tuple(cells[cell_id].value for cell_id in self.occupancy)

    @property
    def state_hash(self) -> str:
        return sha256_json(self.content_dict())

    def content_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": LESION_STATE_SCHEMA_VERSION,
            "sourceScenarioId": self.source_scenario_id,
            "sourceCheckpointHash": self.source_checkpoint_hash,
            "cells": [cell.to_dict() for cell in self.cells],
            "occupancy": list(self.occupancy),
            "selectionCursors": dict(self.selection_cursors),
            "activationCount": self.activation_count,
            "streamCounters": dict(self.stream_counters),
            "ledger": dict(self.ledger),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"stateHash": self.state_hash, **self.content_dict()}

    def validate(self) -> None:
        identifiers = [cell.cell_id for cell in self.cells]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("lesion cells must contain unique identities")
        if len(self.occupancy) != len(identifiers) or set(self.occupancy) != set(
            identifiers
        ):
            raise ValueError("lesion occupancy must contain every cell exactly once")
        cell_map = self.cell_map
        cursor_map = dict(self.selection_cursors)
        if len(cursor_map) != len(self.selection_cursors):
            raise ValueError("selection cursor owners must be unique")
        for cell_id, cursor in cursor_map.items():
            if cell_id not in cell_map or cell_map[cell_id].policy != Policy.SELECTION:
                raise ValueError("cursor owner must be a surviving Selection identity")
            if not isinstance(cursor, int):
                raise TypeError("Selection cursor must remain an integer")
        expected_ledger = set(LEDGER_FIELDS)
        if set(dict(self.ledger)) != expected_ledger:
            raise ValueError("lesion state must preserve the complete E01 ledger")
        if self.activation_count < 0 or any(
            value < 0 for _, value in self.stream_counters
        ):
            raise ValueError("event clock and stream counters must be nonnegative")
        _ = self.direction


@dataclass(frozen=True, slots=True)
class LesionApplication:
    operator_id: str
    family: str
    lesion_case_id: str
    lesion_pair_id: str
    s01_pairing_block_id: str
    timing_condition_id: str
    pre_state: LesionState
    post_state: LesionState
    parameters: Mapping[str, Any]
    identity_contract: Mapping[str, Any]
    target_correspondence: Mapping[str, Any]
    severity: Mapping[str, Any]
    reversibility: Mapping[str, Any]
    runtime_compatibility: Mapping[str, Any]
    tombstone: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schemaVersion": LESION_FIXTURE_SCHEMA_VERSION,
            "benchmarkVersion": BENCHMARK_VERSION,
            "lesionCaseId": self.lesion_case_id,
            "lesionPairId": self.lesion_pair_id,
            "operatorId": self.operator_id,
            "family": self.family,
            "sourceScenarioId": self.pre_state.source_scenario_id,
            "s01PairingBlockId": self.s01_pairing_block_id,
            "timingConditionId": self.timing_condition_id,
            "preInjuryStateHash": self.pre_state.source_checkpoint_hash,
            "preLesionStateHash": self.pre_state.state_hash,
            "postLesionStateHash": self.post_state.state_hash,
            "identityContract": dict(self.identity_contract),
            "targetCorrespondence": dict(self.target_correspondence),
            "severity": dict(self.severity),
            "reversibility": dict(self.reversibility),
            "runtimeCompatibility": dict(self.runtime_compatibility),
            "parameters": dict(self.parameters),
        }
        if self.tombstone is not None:
            result["tombstone"] = dict(self.tombstone)
        return result


def validate_lesion_spec(specification: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(LESION_SPEC_SCHEMA).validate(specification)
    identifiers = tuple(item["operatorId"] for item in specification["operators"])
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(
        OPERATOR_IDS
    ):
        raise ValueError("S03 must define each frozen operator exactly once")
    if any(item in identifiers for item in PROHIBITED_OPERATOR_IDS):
        raise ValueError("the S01/S02 +1-swap fixture cannot enter the lesion library")
    inherited = specification["inherits"]
    expected = {
        "taskSpecSchemaVersion": "e05.s01.task-spec.v1",
        "timingSpecSchemaVersion": "e05.s02.timing-spec.v1",
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
    }
    for key, value in expected.items():
        if inherited.get(key) != value:
            raise ValueError(f"inherited S01/S02 contract changed: {key}")
    panel = specification["validationPanel"]
    if panel.get("timingConditionsPerBlock") != 8:
        raise ValueError("S03 must inherit all eight S02 timing conditions")
    if panel.get("operatorsPerCheckpoint") != 7:
        raise ValueError("S03 must apply all seven operators per checkpoint")


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _central_window(n: int, length: int) -> tuple[int, int]:
    if not 1 <= length <= n:
        raise ValueError("central window must be nonempty and fit the state")
    start = (n - length) // 2
    return start, start + length


def _generated_id(
    operator_id: str, pairing_id: str, timing_condition_id: str, pre_hash: str
) -> str:
    return "e05cell3:" + sha256_json(
        {
            "operatorId": operator_id,
            "s01PairingBlockId": pairing_id,
            "timingConditionId": timing_condition_id,
            "preLesionStateHash": pre_hash,
        }
    )


def _replace_state(
    state: LesionState,
    *,
    cells: Sequence[Cell] | None = None,
    occupancy: Sequence[str] | None = None,
    selection_cursors: Mapping[str, int] | None = None,
) -> LesionState:
    result = LesionState(
        source_scenario_id=state.source_scenario_id,
        source_checkpoint_hash=state.source_checkpoint_hash,
        cells=tuple(sorted(state.cells if cells is None else cells, key=lambda x: x.cell_id)),
        occupancy=tuple(state.occupancy if occupancy is None else occupancy),
        selection_cursors=tuple(
            sorted(
                dict(state.selection_cursors).items()
                if selection_cursors is None
                else selection_cursors.items()
            )
        ),
        activation_count=state.activation_count,
        stream_counters=state.stream_counters,
        ledger=state.ledger,
    )
    result.validate()
    return result


def _cell_clone(cell: Cell, *, cell_id: str, value: int | float | None = None) -> Cell:
    return Cell(
        cell_id=cell_id,
        value=cell.value if value is None else value,
        policy=cell.policy,
        direction=cell.direction,
        fault=cell.fault,
        analysis_label=cell.analysis_label,
    )


def _fault_clone(cell: Cell, fault: FaultMode) -> Cell:
    return Cell(
        cell_id=cell.cell_id,
        value=cell.value,
        policy=cell.policy,
        direction=cell.direction,
        fault=fault,
        analysis_label=cell.analysis_label,
    )


def _position_map(occupancy: Sequence[str]) -> dict[str, int]:
    return {cell_id: index for index, cell_id in enumerate(occupancy)}


def _severity(
    pre: LesionState,
    post: LesionState,
    *,
    direct_ids: Sequence[str],
    identity_additions: int,
    identity_removals: int,
) -> dict[str, Any]:
    pre_distance = strict_unequal_inversions(pre.values, pre.direction)
    post_distance = strict_unequal_inversions(post.values, post.direction)
    pre_denominator = len(pre.occupancy) * (len(pre.occupancy) - 1) // 2
    post_denominator = len(post.occupancy) * (len(post.occupancy) - 1) // 2
    pre_normalized = pre_distance / pre_denominator if pre_denominator else 0.0
    post_normalized = post_distance / post_denominator if post_denominator else 0.0
    pre_positions = _position_map(pre.occupancy)
    post_positions = _position_map(post.occupancy)
    surviving = sorted(set(pre_positions) & set(post_positions))
    displacement_l1 = sum(
        abs(pre_positions[cell_id] - post_positions[cell_id])
        for cell_id in surviving
    )
    displacement_denominator = max(1, len(pre.occupancy) * len(post.occupancy))
    moved = sum(pre_positions[cell_id] != post_positions[cell_id] for cell_id in surviving)
    frozen = sum(cell.fault == FaultMode.STUCK for cell in post.cells) - sum(
        cell.fault == FaultMode.STUCK for cell in pre.cells
    )
    return {
        "validPostTargetOrderDistanceBefore": pre_distance,
        "validPostTargetOrderDistanceAfter": post_distance,
        "validPostTargetOrderDistanceDelta": post_distance - pre_distance,
        "normalizedOrderDistanceBefore": pre_normalized,
        "normalizedOrderDistanceAfter": post_normalized,
        "normalizedOrderDistanceDelta": post_normalized - pre_normalized,
        "directlyAffectedIdentityCount": len(set(direct_ids)),
        "directlyAffectedIdentityFraction": len(set(direct_ids))
        / len(pre.occupancy),
        "movedExistingIdentityCount": moved,
        "existingDisplacementL1": displacement_l1,
        "normalizedExistingDisplacementL1": displacement_l1
        / displacement_denominator,
        "identityAdditionCount": identity_additions,
        "identityRemovalCount": identity_removals,
        "identityCardinalityEditDistance": identity_additions + identity_removals,
        "frozenIdentityCount": frozen,
        "preIdentityCount": len(pre.occupancy),
        "postIdentityCount": len(post.occupancy),
    }


def _target_contract(
    pre: LesionState,
    post: LesionState,
    *,
    target_profile: str,
    metric_profile: str,
    original_target_feasible: bool,
    runtime_reachability: str,
) -> dict[str, Any]:
    post_target = target_values(post.values, post.direction)
    return {
        "targetProfile": target_profile,
        "metricProfile": metric_profile,
        "originalTargetValuesSha256": hashlib.sha256(
            canonical_json_bytes(list(target_values(pre.values, pre.direction)))
        ).hexdigest(),
        "postLesionTargetValuesSha256": hashlib.sha256(
            canonical_json_bytes(list(post_target))
        ).hexdigest(),
        "postLesionTargetIdentityCount": len(post.occupancy),
        "postLesionTargetSequenceFeasible": True,
        "originalIdentityTargetFeasible": original_target_feasible,
        "runtimeReachabilityStatus": runtime_reachability,
    }


def _identity_contract(
    pre: LesionState,
    post: LesionState,
    *,
    count_semantics: str,
    additions: Sequence[str] = (),
    removals: Sequence[str] = (),
    derived_from: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    pre_ids = set(pre.occupancy)
    post_ids = set(post.occupancy)
    survivor_cells_preserved = all(
        pre.cell_map[cell_id].to_dict() == post.cell_map[cell_id].to_dict()
        for cell_id in pre_ids & post_ids
    )
    pre_cursors = dict(pre.selection_cursors)
    post_cursors = dict(post.selection_cursors)
    survivor_cursors_preserved = all(
        pre_cursors[cell_id] == post_cursors[cell_id]
        for cell_id in set(pre_cursors) & set(post_cursors)
    )
    return {
        "countSemantics": count_semantics,
        "identityConserving": pre_ids == post_ids,
        "preIdentityCount": len(pre_ids),
        "postIdentityCount": len(post_ids),
        "addedIdentityIds": list(additions),
        "removedIdentityIds": list(removals),
        "derivedFrom": dict(derived_from or {}),
        "survivingCellRecordsPreserved": survivor_cells_preserved,
        "survivingSelectionCursorsPreserved": survivor_cursors_preserved,
        "activationCountPreserved": pre.activation_count == post.activation_count,
        "streamCountersPreserved": pre.stream_counters == post.stream_counters,
        "ledgerPreserved": pre.ledger == post.ledger,
    }


def _pair_ids(
    operator_id: str,
    pairing_id: str,
    timing_condition_id: str,
    pre_state_hash: str,
) -> tuple[str, str]:
    content = {
        "s01PairingBlockId": pairing_id,
        "timingConditionId": timing_condition_id,
        "preLesionStateHash": pre_state_hash,
        "operatorId": operator_id,
    }
    pair_id = "e05lp3:" + sha256_json(content)
    return "e05s03:" + sha256_json({**content, "arm": "active_lesion"}), pair_id


def _runtime(operator_id: str) -> dict[str, Any]:
    if operator_id in {
        "segment_reversal_central_v1",
        "local_scramble_sattolo_v1",
        "block_transposition_adjacent_equal_v1",
    }:
        return {
            "e01ExecutionCompatibility": "compatible_occupancy_only",
            "postInjuryPairingStatus": "shared_prefix_until_arm_terminal",
            "reason": "scenario content and fixed identity set are unchanged",
        }
    if operator_id == "freeze_stuck_one_central_v1":
        return {
            "e01ExecutionCompatibility": "compatible_after_faulted_scenario_rebuild",
            "postInjuryPairingStatus": "scenario_paired_rng_unpaired",
            "reason": "fault mode changes canonical scenario content and scenario ID",
        }
    return {
        "e01ExecutionCompatibility": "not_executable_in_e01_fixed_identity_engine",
        "postInjuryPairingStatus": "no_postinjury_coupling_claim",
        "reason": "the lesion changes the fixed scenario identity set",
    }


def _make_application(
    *,
    operator_id: str,
    family: str,
    pairing_id: str,
    timing_condition_id: str,
    pre: LesionState,
    post: LesionState,
    parameters: Mapping[str, Any],
    identity_contract: Mapping[str, Any],
    target_correspondence: Mapping[str, Any],
    severity: Mapping[str, Any],
    reversibility_class: str,
    tombstone: Mapping[str, Any] | None = None,
) -> LesionApplication:
    case_id, lesion_pair_id = _pair_ids(
        operator_id, pairing_id, timing_condition_id, pre.state_hash
    )
    application = LesionApplication(
        operator_id=operator_id,
        family=family,
        lesion_case_id=case_id,
        lesion_pair_id=lesion_pair_id,
        s01_pairing_block_id=pairing_id,
        timing_condition_id=timing_condition_id,
        pre_state=pre,
        post_state=post,
        parameters=parameters,
        identity_contract=identity_contract,
        target_correspondence=target_correspondence,
        severity=severity,
        reversibility={
            "class": reversibility_class,
            "inverseFixtureAvailable": True,
            "exactStateRestorationRequiresTombstone": operator_id
            in {"freeze_stuck_one_central_v1", "deletion_central_v1"},
        },
        runtime_compatibility=_runtime(operator_id),
        tombstone=tombstone,
    )
    jsonschema.Draft202012Validator(LESION_FIXTURE_SCHEMA).validate(
        application.to_dict()
    )
    return application


def apply_lesion(
    operator_id: str,
    state: LesionState,
    *,
    s01_pairing_block_id: str,
    timing_condition_id: str,
    injury_seed: int,
) -> LesionApplication:
    """Apply one frozen S03 operator deterministically at a checkpoint."""

    if operator_id in PROHIBITED_OPERATOR_IDS:
        raise ValueError("S01/S02 adjacent-swap fixture is prohibited in S03")
    if operator_id not in OPERATOR_IDS:
        raise ValueError(f"unknown S03 lesion operator: {operator_id}")
    state.validate()
    n = len(state.occupancy)
    if n < 5:
        raise ValueError("S03 lesion fixtures require at least five identities")
    cells = state.cell_map
    center = (n - 1) // 2
    additions: tuple[str, ...] = ()
    removals: tuple[str, ...] = ()
    derived: dict[str, str] = {}
    tombstone: dict[str, Any] | None = None

    if operator_id == "freeze_stuck_one_central_v1":
        family = "freezing"
        selected_id = state.occupancy[center]
        original = cells[selected_id]
        if original.fault != FaultMode.NORMAL:
            raise ValueError("central freezing fixture requires a normal source identity")
        post_cells = [
            _fault_clone(cell, FaultMode.STUCK) if cell.cell_id == selected_id else cell
            for cell in state.cells
        ]
        post = _replace_state(state, cells=post_cells)
        direct = (selected_id,)
        parameters = {"selectedIndex": center, "selectedIdentityId": selected_id}
        tombstone = {"cell": original.to_dict()}
        count_semantics = "identity_conserving"
        target_profile = "original_permutation_target_v1"
        metric_profile = "permutation_inversion_distance_v1"
        reversibility = "exact_with_fault_tombstone"
        original_feasible = True
        reachability = "requires_faulted_scenario_execution"
    elif operator_id == "segment_reversal_central_v1":
        family = "segment_reversal"
        length = min(n, max(4, _round_half_up(0.2 * n)))
        start, end = _central_window(n, length)
        occupancy = list(state.occupancy)
        occupancy[start:end] = reversed(occupancy[start:end])
        post = _replace_state(state, occupancy=occupancy)
        direct = state.occupancy[start:end]
        parameters = {"startIndex": start, "endIndexExclusive": end, "length": length}
        count_semantics = "identity_conserving"
        target_profile = "original_permutation_target_v1"
        metric_profile = "permutation_inversion_distance_v1"
        reversibility = "self_inverse"
        original_feasible = True
        reachability = "executable_in_fixed_identity_engine"
    elif operator_id == "local_scramble_sattolo_v1":
        family = "local_scrambling"
        length = min(n, max(5, _round_half_up(0.2 * n)))
        start, end = _central_window(n, length)
        before = list(state.occupancy[start:end])
        window = list(before)
        draw_index = 0
        draws: list[dict[str, int]] = []
        for index in range(length - 1, 0, -1):
            selected, consumed = bounded(
                injury_seed,
                state.source_scenario_id,
                SCRAMBLE_STREAM,
                state.activation_count,
                index,
                draw_index,
            )
            draws.append(
                {
                    "drawIndex": draw_index,
                    "upperExclusive": index,
                    "selectedIndex": selected,
                    "blocksConsumed": consumed,
                }
            )
            draw_index += consumed
            window[index], window[selected] = window[selected], window[index]
        if any(left == right for left, right in zip(before, window, strict=True)):
            raise AssertionError("Sattolo window must be a fixed-point-free derangement")
        occupancy = list(state.occupancy)
        occupancy[start:end] = window
        post = _replace_state(state, occupancy=occupancy)
        direct = tuple(before)
        parameters = {
            "startIndex": start,
            "endIndexExclusive": end,
            "length": length,
            "stream": SCRAMBLE_STREAM,
            "eventIndex": state.activation_count,
            "drawCount": len(draws),
            "rngBlocksConsumed": draw_index,
            "drawDigest": sha256_json(draws),
            "preWindow": before,
            "postWindow": window,
        }
        count_semantics = "identity_conserving"
        target_profile = "original_permutation_target_v1"
        metric_profile = "permutation_inversion_distance_v1"
        reversibility = "exact_recorded_inverse_permutation"
        original_feasible = True
        reachability = "executable_in_fixed_identity_engine"
    elif operator_id == "block_transposition_adjacent_equal_v1":
        family = "block_transposition"
        block_length = max(2, _round_half_up(0.1 * n))
        if 2 * block_length > n:
            block_length = n // 2
        start, end = _central_window(n, 2 * block_length)
        occupancy = list(state.occupancy)
        left = occupancy[start : start + block_length]
        right = occupancy[start + block_length : end]
        occupancy[start:end] = right + left
        post = _replace_state(state, occupancy=occupancy)
        direct = state.occupancy[start:end]
        parameters = {
            "startIndex": start,
            "endIndexExclusive": end,
            "blockLength": block_length,
        }
        count_semantics = "identity_conserving"
        target_profile = "original_permutation_target_v1"
        metric_profile = "permutation_inversion_distance_v1"
        reversibility = "self_inverse"
        original_feasible = True
        reachability = "executable_in_fixed_identity_engine"
    elif operator_id == "duplication_tandem_v1":
        family = "duplication"
        source_id = state.occupancy[center]
        new_id = _generated_id(
            operator_id,
            s01_pairing_block_id,
            timing_condition_id,
            state.state_hash,
        )
        if new_id in cells:
            raise AssertionError("generated duplication identity collided")
        copied = _cell_clone(cells[source_id], cell_id=new_id)
        occupancy = list(state.occupancy)
        occupancy.insert(center + 1, new_id)
        cursors = dict(state.selection_cursors)
        if copied.policy == Policy.SELECTION:
            cursors[new_id] = (
                0 if copied.direction == Direction.ASCENDING else len(occupancy) - 1
            )
        post = _replace_state(
            state,
            cells=(*state.cells, copied),
            occupancy=occupancy,
            selection_cursors=cursors,
        )
        additions = (new_id,)
        derived = {new_id: source_id}
        direct = (source_id, new_id)
        parameters = {
            "sourceIndex": center,
            "sourceIdentityId": source_id,
            "insertedIndex": center + 1,
            "generatedIdentityId": new_id,
        }
        count_semantics = "derived_identity_addition"
        target_profile = "augmented_multiset_target_v1"
        metric_profile = (
            "augmented_multiset_order_distance_v1+"
            "identity_cardinality_edit_distance_v1"
        )
        reversibility = "exact_by_removing_generated_identity"
        original_feasible = False
        reachability = "target_defined_but_not_e01_runtime_executable"
    elif operator_id == "deletion_central_v1":
        family = "deletion"
        removed_id = state.occupancy[center]
        removed_cell = cells[removed_id]
        occupancy = list(state.occupancy)
        occupancy.pop(center)
        cursors = dict(state.selection_cursors)
        removed_cursor = cursors.pop(removed_id, None)
        post = _replace_state(
            state,
            cells=[cell for cell in state.cells if cell.cell_id != removed_id],
            occupancy=occupancy,
            selection_cursors=cursors,
        )
        removals = (removed_id,)
        direct = (removed_id,)
        parameters = {"removedIndex": center, "removedIdentityId": removed_id}
        tombstone = {
            "cell": removed_cell.to_dict(),
            "occupancyIndex": center,
            "selectionCursor": removed_cursor,
        }
        count_semantics = "identity_removal"
        target_profile = "sorted_survivor_target_v1"
        metric_profile = (
            "survivor_order_distance_v1+identity_cardinality_edit_distance_v1"
        )
        reversibility = "conditional_exact_with_tombstone"
        original_feasible = False
        reachability = "target_defined_but_not_e01_runtime_executable"
    else:
        family = "out_of_place_insertion"
        reference = cells[state.occupancy[center]]
        new_id = _generated_id(
            operator_id,
            s01_pairing_block_id,
            timing_condition_id,
            state.state_hash,
        )
        if new_id in cells:
            raise AssertionError("generated inserted identity collided")
        new_value = max(cell.value for cell in state.cells) + 1
        inserted = _cell_clone(reference, cell_id=new_id, value=new_value)
        insertion_index = 0 if state.direction == Direction.ASCENDING else n
        occupancy = list(state.occupancy)
        occupancy.insert(insertion_index, new_id)
        cursors = dict(state.selection_cursors)
        if inserted.policy == Policy.SELECTION:
            cursors[new_id] = (
                0 if inserted.direction == Direction.ASCENDING else len(occupancy) - 1
            )
        post = _replace_state(
            state,
            cells=(*state.cells, inserted),
            occupancy=occupancy,
            selection_cursors=cursors,
        )
        additions = (new_id,)
        direct = (new_id,)
        parameters = {
            "insertedIndex": insertion_index,
            "generatedIdentityId": new_id,
            "insertedValue": new_value,
            "placementRule": "direction_wrong_boundary",
        }
        count_semantics = "novel_identity_addition"
        target_profile = "augmented_sorted_target_v1"
        metric_profile = (
            "augmented_order_distance_v1+identity_cardinality_edit_distance_v1"
        )
        reversibility = "exact_by_removing_generated_identity"
        original_feasible = False
        reachability = "target_defined_but_not_e01_runtime_executable"

    identity = _identity_contract(
        state,
        post,
        count_semantics=count_semantics,
        additions=additions,
        removals=removals,
        derived_from=derived,
    )
    target = _target_contract(
        state,
        post,
        target_profile=target_profile,
        metric_profile=metric_profile,
        original_target_feasible=original_feasible,
        runtime_reachability=reachability,
    )
    severity = _severity(
        state,
        post,
        direct_ids=direct,
        identity_additions=len(additions),
        identity_removals=len(removals),
    )
    return _make_application(
        operator_id=operator_id,
        family=family,
        pairing_id=s01_pairing_block_id,
        timing_condition_id=timing_condition_id,
        pre=state,
        post=post,
        parameters=parameters,
        identity_contract=identity,
        target_correspondence=target,
        severity=severity,
        reversibility_class=reversibility,
        tombstone=tombstone,
    )


def inverse_lesion(application: LesionApplication) -> LesionState:
    """Apply the declared inverse using fixture metadata/tombstones."""

    operator = application.operator_id
    post = application.post_state
    parameters = application.parameters
    if operator == "freeze_stuck_one_central_v1":
        if application.tombstone is None:
            raise ValueError("freezing inversion requires the original cell tombstone")
        original = Cell.from_dict(application.tombstone["cell"])
        restored_cells = [
            original if cell.cell_id == original.cell_id else cell for cell in post.cells
        ]
        return _replace_state(post, cells=restored_cells)
    if operator == "segment_reversal_central_v1":
        start = int(parameters["startIndex"])
        end = int(parameters["endIndexExclusive"])
        occupancy = list(post.occupancy)
        occupancy[start:end] = reversed(occupancy[start:end])
        return _replace_state(post, occupancy=occupancy)
    if operator == "local_scramble_sattolo_v1":
        start = int(parameters["startIndex"])
        end = int(parameters["endIndexExclusive"])
        pre_window = list(parameters["preWindow"])
        if set(pre_window) != set(post.occupancy[start:end]):
            raise ValueError("scramble inverse window metadata does not correspond")
        occupancy = list(post.occupancy)
        occupancy[start:end] = pre_window
        return _replace_state(post, occupancy=occupancy)
    if operator == "block_transposition_adjacent_equal_v1":
        start = int(parameters["startIndex"])
        end = int(parameters["endIndexExclusive"])
        block = int(parameters["blockLength"])
        occupancy = list(post.occupancy)
        left = occupancy[start : start + block]
        right = occupancy[start + block : end]
        occupancy[start:end] = right + left
        return _replace_state(post, occupancy=occupancy)
    if operator in {"duplication_tandem_v1", "insertion_out_of_place_max_v1"}:
        generated = str(parameters["generatedIdentityId"])
        occupancy = [cell_id for cell_id in post.occupancy if cell_id != generated]
        cursors = dict(post.selection_cursors)
        cursors.pop(generated, None)
        cells = [cell for cell in post.cells if cell.cell_id != generated]
        return _replace_state(
            post, cells=cells, occupancy=occupancy, selection_cursors=cursors
        )
    if application.tombstone is None:
        raise ValueError("deletion inversion requires a complete tombstone")
    tombstone = application.tombstone
    restored = Cell.from_dict(tombstone["cell"])
    index = int(tombstone["occupancyIndex"])
    occupancy = list(post.occupancy)
    occupancy.insert(index, restored.cell_id)
    cursors = dict(post.selection_cursors)
    if tombstone.get("selectionCursor") is not None:
        cursors[restored.cell_id] = int(tombstone["selectionCursor"])
    return _replace_state(
        post,
        cells=(*post.cells, restored),
        occupancy=occupancy,
        selection_cursors=cursors,
    )


def validate_application(application: LesionApplication) -> dict[str, Any]:
    """Validate invariants, clock/state preservation, target, and exact inverse."""

    pre = application.pre_state
    post = application.post_state
    pre.validate()
    post.validate()
    identity = application.identity_contract
    expected_added = set(post.occupancy) - set(pre.occupancy)
    expected_removed = set(pre.occupancy) - set(post.occupancy)
    inverse = inverse_lesion(application)
    checks = {
        "fixtureSchemaPass": True,
        "operatorAllowed": application.operator_id in OPERATOR_IDS
        and application.operator_id not in PROHIBITED_OPERATOR_IDS,
        "preStateValid": True,
        "postStateValid": True,
        "activationCountPreserved": pre.activation_count == post.activation_count,
        "streamCountersPreserved": pre.stream_counters == post.stream_counters,
        "ledgerPreserved": pre.ledger == post.ledger,
        "sourceCheckpointPreserved": pre.source_checkpoint_hash
        == post.source_checkpoint_hash,
        "addedIdentityDeclarationExact": expected_added
        == set(identity["addedIdentityIds"]),
        "removedIdentityDeclarationExact": expected_removed
        == set(identity["removedIdentityIds"]),
        "targetSequenceFeasible": bool(
            application.target_correspondence["postLesionTargetSequenceFeasible"]
        ),
        "inverseExact": inverse.state_hash == pre.state_hash,
        "activeStateChanged": post.state_hash != pre.state_hash,
    }
    if application.operator_id == "local_scramble_sattolo_v1":
        start = int(application.parameters["startIndex"])
        end = int(application.parameters["endIndexExclusive"])
        checks["scrambleDerangement"] = all(
            left != right
            for left, right in zip(
                pre.occupancy[start:end], post.occupancy[start:end], strict=True
            )
        )
    success = all(checks.values())
    return {"success": success, "checks": checks}


def validate_pairing_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate one sham/active pair for every lesion case."""

    failures: list[str] = []
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["lesionPairId"]), []).append(row)
    shared = (
        "s01PairingBlockId",
        "timingConditionId",
        "preInjuryStateHash",
        "preLesionStateHash",
        "activationCount",
        "streamCountersSha256",
        "ledgerSha256",
        "developmentBudget",
        "recoveryBudget",
        "eventBudgetProfile",
        "sourceScenarioId",
    )
    for pair_id, group in groups.items():
        if len(group) != 2 or {item["arm"] for item in group} != {
            "matched_sham",
            "active_lesion",
        }:
            failures.append(f"{pair_id}: expected exact sham/active pair")
            continue
        for field in shared:
            if len({str(item[field]) for item in group}) != 1:
                failures.append(f"{pair_id}: mismatch in {field}")
    return {
        "success": not failures,
        "pairCount": len(groups),
        "rowCount": len(rows),
        "failureCount": len(failures),
        "failures": failures,
    }
