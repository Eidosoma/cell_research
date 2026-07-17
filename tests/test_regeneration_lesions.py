from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from reference_simulator.api import create_scenario
from reference_simulator.model import Direction, FaultMode, Policy
from src.regeneration.lesions import (
    LESION_FIXTURE_SCHEMA,
    OPERATOR_IDS,
    LesionState,
    apply_lesion,
    inverse_lesion,
    validate_application,
    validate_lesion_spec,
    validate_pairing_rows,
)
from src.regeneration.tasks import initial_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/regeneration/s03_lesions.json"


def _spec() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _state(
    *,
    n: int = 10,
    policy: Policy = Policy.BUBBLE,
    direction: Direction = Direction.ASCENDING,
) -> LesionState:
    values = list(range(n))
    if direction == Direction.DESCENDING:
        values.reverse()
    scenario = create_scenario(
        values,
        policy=policy,
        direction=direction,
        seed=17,
        max_activations=100 * n * n,
        generation_key=f"test/s03/{n}/{policy.value}/{direction.value}",
        permute=False,
    )
    return LesionState.from_checkpoint(scenario, initial_checkpoint(scenario))


def _apply(operator_id: str, state: LesionState):
    return apply_lesion(
        operator_id,
        state,
        s01_pairing_block_id="e05pb1:" + "a" * 64,
        timing_condition_id="post_completion",
        injury_seed=123456789,
    )


def test_spec_freezes_exact_seven_operators_and_prohibits_s01_swap() -> None:
    specification = _spec()
    validate_lesion_spec(specification)
    assert {item["operatorId"] for item in specification["operators"]} == set(
        OPERATOR_IDS
    )
    assert "s01_validation_adjacent_swap_v1" in specification["prohibitedOperators"]
    assert (
        LESION_FIXTURE_SCHEMA["properties"]["schemaVersion"]["const"]
        == "e05.s03.lesion-fixture.v1"
    )


def test_s01_swap_is_rejected_by_operator_dispatch() -> None:
    with pytest.raises(ValueError, match="prohibited"):
        _apply("s01_validation_adjacent_swap_v1", _state())


@pytest.mark.parametrize("operator_id", OPERATOR_IDS)
@pytest.mark.parametrize("direction", [Direction.ASCENDING, Direction.DESCENDING])
def test_all_operator_invariants_replay_and_exact_inverse(
    operator_id: str, direction: Direction
) -> None:
    state = _state(policy=Policy.SELECTION, direction=direction)
    first = _apply(operator_id, state)
    second = _apply(operator_id, state)
    audit = validate_application(first)
    assert audit["success"], audit
    assert first.to_dict() == second.to_dict()
    assert inverse_lesion(first).state_hash == state.state_hash
    assert first.post_state.activation_count == state.activation_count
    assert first.post_state.stream_counters == state.stream_counters
    assert first.post_state.ledger == state.ledger


def test_identity_and_count_semantics_are_operator_specific() -> None:
    state = _state(policy=Policy.SELECTION)
    results = {operator: _apply(operator, state) for operator in OPERATOR_IDS}
    conserving = OPERATOR_IDS[:4]
    assert all(
        results[operator].identity_contract["identityConserving"]
        for operator in conserving
    )
    assert (
        results["freeze_stuck_one_central_v1"].post_state.cell_map[
            results["freeze_stuck_one_central_v1"].parameters["selectedIdentityId"]
        ].fault
        == FaultMode.STUCK
    )
    duplication = results["duplication_tandem_v1"]
    deletion = results["deletion_central_v1"]
    insertion = results["insertion_out_of_place_max_v1"]
    assert len(duplication.post_state.occupancy) == len(state.occupancy) + 1
    assert len(deletion.post_state.occupancy) == len(state.occupancy) - 1
    assert len(insertion.post_state.occupancy) == len(state.occupancy) + 1
    assert duplication.identity_contract["derivedFrom"]
    assert deletion.tombstone is not None
    assert insertion.parameters["insertedIndex"] == 0


@pytest.mark.parametrize("direction", [Direction.ASCENDING, Direction.DESCENDING])
def test_sorted_fixture_severity_has_analytically_expected_anchors(
    direction: Direction,
) -> None:
    state = _state(direction=direction)
    applications = {operator: _apply(operator, state) for operator in OPERATOR_IDS}
    deltas = {
        operator: item.severity["validPostTargetOrderDistanceDelta"]
        for operator, item in applications.items()
    }
    assert deltas["freeze_stuck_one_central_v1"] == 0
    assert deltas["segment_reversal_central_v1"] == 6  # C(4, 2)
    assert deltas["block_transposition_adjacent_equal_v1"] == 4  # 2*2 pairs
    assert deltas["duplication_tandem_v1"] == 0
    assert deltas["deletion_central_v1"] == 0
    assert deltas["insertion_out_of_place_max_v1"] == len(state.occupancy)
    assert deltas["local_scramble_sattolo_v1"] > 0


def test_count_changing_tasks_use_nonpermutation_metrics_and_targets() -> None:
    state = _state()
    duplication = _apply("duplication_tandem_v1", state)
    deletion = _apply("deletion_central_v1", state)
    insertion = _apply("insertion_out_of_place_max_v1", state)
    for application in (duplication, deletion, insertion):
        target = application.target_correspondence
        assert target["postLesionTargetSequenceFeasible"]
        assert not target["originalIdentityTargetFeasible"]
        assert "identity_cardinality_edit_distance_v1" in target["metricProfile"]
        assert "permutation_inversion_distance_v1" not in target["metricProfile"]
        assert (
            application.runtime_compatibility["e01ExecutionCompatibility"]
            == "not_executable_in_e01_fixed_identity_engine"
        )


def test_new_selection_cursor_uses_new_boundary_without_remapping_survivors() -> None:
    descending = _state(policy=Policy.SELECTION, direction=Direction.DESCENDING)
    old_cursors = dict(descending.selection_cursors)
    for operator in ("duplication_tandem_v1", "insertion_out_of_place_max_v1"):
        application = _apply(operator, descending)
        cursors = dict(application.post_state.selection_cursors)
        generated = application.parameters["generatedIdentityId"]
        assert cursors[generated] == len(application.post_state.occupancy) - 1
        assert all(cursors[cell_id] == value for cell_id, value in old_cursors.items())


def test_pairing_validator_rejects_checkpoint_or_ledger_mismatch() -> None:
    application = _apply("segment_reversal_central_v1", _state())
    common = {
        "lesionPairId": application.lesion_pair_id,
        "s01PairingBlockId": application.s01_pairing_block_id,
        "timingConditionId": application.timing_condition_id,
        "preInjuryStateHash": application.pre_state.source_checkpoint_hash,
        "preLesionStateHash": application.pre_state.state_hash,
        "activationCount": application.pre_state.activation_count,
        "streamCountersSha256": "1" * 64,
        "ledgerSha256": "2" * 64,
        "developmentBudget": 10_000,
        "recoveryBudget": 10_000,
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "sourceScenarioId": application.pre_state.source_scenario_id,
    }
    sham = {**common, "arm": "matched_sham"}
    active = {**common, "arm": "active_lesion"}
    assert validate_pairing_rows([sham, active])["success"]
    broken = deepcopy(active)
    broken["ledgerSha256"] = "3" * 64
    audit = validate_pairing_rows([sham, broken])
    assert not audit["success"]
    assert any("ledgerSha256" in failure for failure in audit["failures"])
