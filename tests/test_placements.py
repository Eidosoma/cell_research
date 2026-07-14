from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.faults import (
    FaultExecutionContract,
    MobilityProfile,
    exact_replay_fault,
    run_faulted_architecture,
)
from causal_simulator.placements import (
    AnalysisRole,
    ConfirmatoryLeakageError,
    OutcomeAccess,
    PlacementClass,
    PlacementContext,
    boundary_band_positions,
    generate_structural_bank,
    generate_structural_placement,
    materialize_fault_scenario,
    median_rank_band_positions,
    search_exploratory_placements,
    select_confirmatory_placements,
    structural_lock_record,
)
from causal_simulator.schedulers import SchedulerExecutionContract, SchedulerFamily
from reference_simulator.model import FaultMode, Policy


FIXTURE = Path(__file__).parent / "fixtures" / "s06_placement_fixtures.json"


def _context(n: int = 24) -> PlacementContext:
    return PlacementContext(
        f"test_n{n}",
        tuple(f"cell-{index:04d}" for index in range(n)),
        tuple(range(n, 0, -1)),
    )


def test_portable_rule_fixture_and_preoutcome_api() -> None:
    fixture = json.loads(FIXTURE.read_text())
    raw = fixture["context"]
    context = PlacementContext(
        raw["contextId"], tuple(raw["identityIds"]), tuple(raw["values"])
    )
    for class_name, expected in fixture["expected"].items():
        placement = generate_structural_placement(
            context,
            PlacementClass(class_name),
            fixture["faultCount"],
            seed=fixture["seed"],
            replicate_ordinal=fixture["replicateOrdinal"],
        )
        assert list(placement.positions) == expected["positions"]
        assert placement.placement_id == expected["placementId"]
        assert placement.outcome_access == OutcomeAccess.NONE
        assert placement.confirmatory_eligible
    parameters = set(inspect.signature(generate_structural_placement).parameters)
    assert not parameters & {"scorer", "outcomes", "run_result", "confirmatory"}


def test_structural_bank_exact_count_cross_class_uniqueness_and_classes() -> None:
    context = _context()
    bank = generate_structural_bank(
        [context], [2, 4, 6], seed=20260714, maps_per_class=12
    )
    assert len(bank) == 144
    assert len({item.placement_id for item in bank}) == len(bank)
    for fault_count in (2, 4, 6):
        cell = [item for item in bank if item.fault_count == fault_count]
        assert len({item.positions for item in cell}) == 48
        for placement_class in tuple(PlacementClass)[:4]:
            items = [
                item for item in cell if item.placement_class == placement_class
            ]
            assert len(items) == 12
            assert all(len(item.positions) == len(set(item.positions)) == fault_count for item in items)
            assert all(len(set(item.identity_ids)) == fault_count for item in items)
        clustered = [
            item
            for item in cell
            if item.placement_class == PlacementClass.CLUSTERED_EXACT
        ]
        assert all(item.descriptor_dict()["longestContiguousRun"] == fault_count for item in clustered)
        boundary_band = set(boundary_band_positions(context, fault_count))
        boundary = [
            item
            for item in cell
            if item.placement_class == PlacementClass.BOUNDARY_EXACT
        ]
        assert all(set(item.positions) <= boundary_band for item in boundary)
        median_band = set(median_rank_band_positions(context, fault_count))
        median = [
            item
            for item in cell
            if item.placement_class == PlacementClass.MEDIAN_RANK_EXACT
        ]
        assert all(set(item.positions) <= median_band for item in median)
    lock = structural_lock_record(bank)
    assert lock["placementCount"] == 144
    assert not lock["outcomesAccessedBeforeLock"]
    assert len(lock["lockedPlacementSetSha256"]) == 64
    assert select_confirmatory_placements(bank) == bank


def test_search_is_disjoint_exploratory_deterministic_and_rejected() -> None:
    context = _context(10)
    structural = generate_structural_bank(
        [context], [3], seed=73, maps_per_class=2
    )
    forbidden = {item.positions for item in structural}
    calls: list[tuple[int, ...]] = []

    def scorer(positions: tuple[int, ...]) -> tuple[int, int, int]:
        assert positions not in forbidden
        calls.append(positions)
        local = sum((ordinal + 1) * position for ordinal, position in enumerate(positions))
        score = positions[-1] - positions[0] - 5
        return score, local, local + score

    placements, history = search_exploratory_placements(
        context,
        3,
        seed=918022,
        forbidden_structural=forbidden,
        scorer=scorer,
    )
    assert len(placements) == 6 and len(history) == len(set(calls))
    assert not {item.positions for item in placements} & forbidden
    assert all(
        item.outcome_access == OutcomeAccess.EXPLORATORY_TRAINING_ONLY
        and item.analysis_role == AnalysisRole.EXPLORATORY_SEARCH_DERIVED
        and not item.confirmatory_eligible
        for item in placements
    )
    with pytest.raises(ConfirmatoryLeakageError):
        select_confirmatory_placements((*structural, *placements))

    replayed, replay_history = search_exploratory_placements(
        context,
        3,
        seed=918022,
        forbidden_structural=forbidden,
        scorer=lambda positions: (
            positions[-1] - positions[0] - 5,
            sum((ordinal + 1) * position for ordinal, position in enumerate(positions)),
            sum((ordinal + 1) * position for ordinal, position in enumerate(positions))
            + positions[-1]
            - positions[0]
            - 5,
        ),
    )
    assert [item.to_json_bytes() for item in replayed] == [
        item.to_json_bytes() for item in placements
    ]
    assert replay_history == history


@pytest.mark.parametrize("mode", (FaultMode.PASSIVE, FaultMode.STUCK))
def test_materialization_preserves_s05_semantics_ledger_and_replay(mode: FaultMode) -> None:
    context = PlacementContext(
        "s05_overlay_n8",
        tuple(f"cell-{index:04d}" for index in range(8)),
        (8, 1, 7, 2, 6, 3, 5, 4),
    )
    placement = generate_structural_placement(
        context,
        PlacementClass.BOUNDARY_EXACT,
        2,
        seed=20260714,
        replicate_ordinal=0,
    )
    scenario = materialize_fault_scenario(
        context,
        placement,
        fault_mode=mode,
        policy=Policy.BUBBLE,
        seed=20260714,
        max_activations=128,
        generation_key=f"S06/test/{mode.value}",
    )
    faulty_ids = {cell.cell_id for cell in scenario.cells if cell.fault == mode}
    assert faulty_ids == set(placement.identity_ids)
    assert scenario.requested_fault_count == scenario.realized_fault_count == 2
    run = run_faulted_architecture(
        scenario,
        ArchitectureExecutionContract.distributed_local(),
        SchedulerExecutionContract(SchedulerFamily.DETERMINISTIC_SCAN),
        FaultExecutionContract(mobility=MobilityProfile(mode.value)),
        trace_mode="full",
    )
    assert all(run.opportunity_validation().values())
    assert exact_replay_fault(run).to_json_bytes() == run.to_json_bytes()
