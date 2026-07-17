from __future__ import annotations

import itertools
import json

import numpy as np

from analysis.s11r_corrective import (
    CONTRACT_PATH,
    bank_inversion_counts,
    build_value_continuation,
    exhaustive_small_validation,
    positional_inversion_count,
    select_pair_at_exact_target,
    select_value_pair,
)
from reference_simulator.engine import initial_state
from reference_simulator.model import Cell, Direction, FaultMode, Policy, Scenario


def _scenario() -> Scenario:
    cells = tuple(
        Cell(
            cell_id=f"c{index:02d}",
            value=index,
            policy=Policy.SELECTION if index % 2 else Policy.BUBBLE,
            direction=Direction.ASCENDING,
            fault=FaultMode.NORMAL,
            analysis_label=f"label-{index % 2}",
        )
        for index in range(10)
    )
    return Scenario.create(
        cells,
        initial_occupancy=tuple(cell.cell_id for cell in cells),
        seed=17,
        max_activations=1_000_000,
        generation_key="E04/S11R/test",
    )


def test_contract_freezes_correction_and_no_s12() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["researchStepId"] == "S11R"
    assert contract["scope"]["stopsBefore"] == "S12"
    assert contract["scope"]["preservesCompletedS11"]
    assert not contract["scope"]["valueReassignmentIsPureMediation"]
    assert contract["matcher"]["primaryTarget"].endswith("intended distance is zero.")
    assert contract["calibration"]["perConditionArmMinimumComplete"] == 24


def test_positional_inversions_match_brute_force_on_small_arrangements() -> None:
    for n in range(1, 7):
        for values in itertools.permutations(range(n)):
            brute = sum(
                values[left] > values[right]
                for left in range(n)
                for right in range(left + 1, n)
            )
            assert positional_inversion_count(values) == brute
    assert positional_inversion_count([2, 1, 3, 1]) == 3
    assert positional_inversion_count([1, 1, 1]) == 0


def test_corrected_matcher_attains_exact_target_and_objective() -> None:
    bank = np.asarray(list(itertools.permutations(range(5))), dtype=np.int16)
    original = np.arange(5, dtype=np.int16)
    inversions = bank_inversion_counts(bank)
    selected = select_pair_at_exact_target(bank, original, inversions, 5)
    assert selected["inversionDistance"] == 0
    assert selected["primaryInversions"] == selected["matchedInversions"] == 5
    eligible = np.flatnonzero(inversions == 5)
    brute = []
    for left_offset in range(len(eligible)):
        for right_offset in range(left_offset + 1, len(eligible)):
            left, right = int(eligible[left_offset]), int(eligible[right_offset])
            brute.append(
                (
                    abs(
                        int(np.abs(bank[left] - original).sum())
                        - int(np.abs(bank[right] - original).sum())
                    ),
                    -int(np.sum(bank[left] != bank[right])),
                )
            )
    assert (selected["l1Difference"], -selected["assignmentHamming"]) == min(brute)


def test_production_pair_support_and_exact_target() -> None:
    values = list(range(100))
    result = select_value_pair(values, "M02_MIN", 2, "minimum", "fixture-unique")
    assert result["targetAttained"]
    assert result["inversionDistance"] == 0
    assert result["supportQualified"]
    assert result["targetSupport"] >= 16
    assert result["primaryChangedCount"] == 2
    assert result["matchedChangedCount"] == 2
    assert result["primaryValueMultisetPreserved"]
    assert result["matchedValueMultisetPreserved"]


def test_value_continuation_preserves_state_scheduler_and_selection_cursors() -> None:
    source = _scenario()
    checkpoint = initial_state(source)
    for index, cell_id in enumerate(sorted(checkpoint.selection_cursors)):
        checkpoint.selection_cursors[cell_id] = index + 2
    values = [source.cell_map[cell_id].value for cell_id in checkpoint.occupancy]
    changed = list(values)
    changed[1], changed[8] = changed[8], changed[1]
    scenario, state, audit = build_value_continuation(
        source, checkpoint, changed, "value_primary"
    )
    assert audit["passed"]
    assert audit["value_changed_count"] == 2
    assert audit["global_value_multiset_preserved"]
    assert audit["policy_direction_fault_label_preserved"]
    assert state.selection_cursors == checkpoint.selection_cursors
    assert scenario.max_activations == source.max_activations == 1_000_000
    assert scenario.scenario_id == source.scenario_id


def test_exhaustive_validation_summary_passes() -> None:
    result = exhaustive_small_validation()
    assert result["passed"]
    assert result["uniqueArrangementsChecked"] > 5_000
    assert result["repeatedArrangementsChecked"] > 1_000
    assert result["matcherCandidatePairsCompared"] > 1_000
