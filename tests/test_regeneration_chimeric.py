from __future__ import annotations

import json
from pathlib import Path

from reference_simulator.model import LEDGER_FIELDS
from reference_simulator.engine import evaluate_terminal, execute_batch
from causal_simulator.architectures import (
    ArchitectureExecutionContract,
    ArchitectureProposalRouter,
)
from src.regeneration.chimeric import (
    PORTFOLIOS,
    assign_policies,
    build_composition_scenario,
    clone_with_stuck_identities,
    composition_counts,
    exact_replay_native,
    fault_count,
    fault_identity_ranking,
    run_native_recovery,
    should_stop_threshold,
    UniformSchedule,
    validate_chimeric_spec,
)
from src.regeneration.tasks import Checkpoint, _checkpoint_hash
from src.regeneration.transfer import apply_transfer_lesion


REPOSITORY = Path(__file__).resolve().parents[1]


def _sorted_checkpoint(scenario) -> Checkpoint:
    reverse = scenario.cells[0].direction.value == "descending"
    occupancy = tuple(
        cell.cell_id for cell in sorted(scenario.cells, key=lambda cell: cell.value, reverse=reverse)
    )
    cursors = dict(scenario.initial_selection_cursors)
    streams: dict[str, int] = {}
    ledger = {field: 0 for field in LEDGER_FIELDS}
    return Checkpoint(
        occupancy,
        tuple(sorted(cursors.items())),
        0,
        (),
        tuple((field, 0) for field in LEDGER_FIELDS),
        0,
        _checkpoint_hash(
            scenario.scenario_id,
            occupancy,
            cursors,
            0,
            streams,
            ledger,
        ),
    )


def test_s13_specification_and_count_equations() -> None:
    specification = json.loads(
        (REPOSITORY / "configs/regeneration/s13_chimeric_recovery.json").read_text()
    )
    validate_chimeric_spec(specification)
    assert specification["panel"]["plannedSourceCount"] == 352
    assert specification["panel"]["plannedMainRuns"] == 2816


def test_exact_compositions_and_outcome_blind_placement_maps() -> None:
    for n in (32, 80):
        for portfolio, policies in PORTFOLIOS.items():
            for placement_map in (0, 1):
                counts = composition_counts(
                    n, portfolio, replicate=3, placement_map=placement_map
                )
                assert sum(counts.values()) == n
                assert set(counts) == set(policies)
                if len(policies) == 2:
                    assert sorted(counts.values()) == [n // 2, n // 2]
                if len(policies) == 3:
                    assert max(counts.values()) - min(counts.values()) <= 1
    ids = tuple(f"cell-{index:04d}" for index in range(32))
    first = assign_policies(
        ids,
        "chimera_bubble_selection",
        n=32,
        direction="ascending",
        replicate=0,
        placement_map=0,
    )
    replay = assign_policies(
        ids,
        "chimera_bubble_selection",
        n=32,
        direction="ascending",
        replicate=0,
        placement_map=0,
    )
    sensitivity = assign_policies(
        ids,
        "chimera_bubble_selection",
        n=32,
        direction="ascending",
        replicate=0,
        placement_map=1,
    )
    assert first == replay
    assert first != sensitivity
    assert sorted(policy.value for policy in first.values()) == sorted(
        policy.value for policy in sensitivity.values()
    )


def test_fault_ranks_are_nested_exact_and_portfolio_independent() -> None:
    ids = tuple(f"cell-{index:04d}" for index in range(32))
    ranking = fault_identity_ranking(ids, n=32, direction="ascending", replicate=2)
    assert set(ranking) == set(ids)
    assert len(ranking[: fault_count(32, 0.2)]) == 6
    assert set(ranking[: fault_count(32, 0.2)]) <= set(
        ranking[: fault_count(32, 0.5)]
    )


def test_fault_clone_changes_only_fault_flags_and_rebinds_checkpoint() -> None:
    scenario, _ = build_composition_scenario(
        n=8,
        portfolio_id="chimera_three_way",
        direction="ascending",
        replicate=0,
        placement_map=0,
        max_activations=20_000,
    )
    checkpoint = _sorted_checkpoint(scenario)
    selected = tuple(sorted(scenario.cell_map)[:2])
    faulted, rebound, metadata = clone_with_stuck_identities(
        scenario, checkpoint, selected, fraction=0.25
    )
    assert metadata["onlyFaultFlagsChanged"]
    assert faulted.realized_fault_count == 2
    assert faulted.scenario_id != scenario.scenario_id
    assert rebound.state_hash != checkpoint.state_hash
    assert rebound.occupancy == checkpoint.occupancy
    assert rebound.selection_cursors == checkpoint.selection_cursors


def test_native_mixed_recovery_ledger_and_exact_replay() -> None:
    scenario, _ = build_composition_scenario(
        n=8,
        portfolio_id="chimera_bubble_insertion",
        direction="ascending",
        replicate=1,
        placement_map=0,
        max_activations=20_000,
    )
    checkpoint = _sorted_checkpoint(scenario)
    lesion = apply_transfer_lesion(
        scenario,
        checkpoint,
        lesion_type="segment_reversal_central_v1",
        location="central",
    )
    result = run_native_recovery(
        scenario,
        checkpoint,
        postinjury_occupancy=lesion["postOccupancy"],
        lesion_state_hash=lesion["lesionStateHash"],
        recovery_budget=6_400,
    )
    exact_replay_native(
        scenario,
        checkpoint,
        postinjury_occupancy=lesion["postOccupancy"],
        lesion_state_hash=lesion["lesionStateHash"],
        recovery_budget=6_400,
        expected=result,
    )
    assert all(result["validation"].values())
    assert result["nativeLedgerDelta"]["activations"] == result["phaseActivationCount"]
    assert result["nativeLedgerDelta"]["proposals"] == result["phaseActivationCount"]
    # The S13 summary projection must end in the same authoritative state as
    # the ordinary event engine; this is independent of replaying the fast path.
    state = checkpoint.to_run_state(occupancy=lesion["postOccupancy"])
    start = state.activation_count
    router = ArchitectureProposalRouter(ArchitectureExecutionContract.distributed_local())
    schedule = UniformSchedule(scenario)
    state.terminal = evaluate_terminal(scenario, state)
    while state.terminal is None and state.activation_count - start < 6_400:
        execute_batch(
            scenario,
            state,
            retain_events=False,
            emit_event_records=False,
            proposal_factory=router.proposal_for,
            schedule_factory=schedule,
        )
    if state.terminal is None:
        state.terminal = "phase_event_budget"
    assert state.terminal == result["stopReason"]
    assert dict(state.ledger) == {
        field: result["nativeLedgerDelta"][field] for field in LEDGER_FIELDS
    }
    assert state.occupancy == result["finalOccupancy"]


def test_global_two_level_threshold_stop_rule() -> None:
    portfolios = tuple(PORTFOLIOS)
    low = {portfolio: 0.49 for portfolio in portfolios}
    high_one = dict(low)
    high_one[portfolios[0]] = 0.51
    assert not should_stop_threshold([low], portfolios)
    assert not should_stop_threshold([low, high_one], portfolios)
    assert should_stop_threshold([low, low], portfolios)
