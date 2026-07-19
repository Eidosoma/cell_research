from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.morph2d.targets import (
    REQUIRED_FAMILIES,
    adjacent_swap_witness,
    apply_swap_witness,
    deterministic_scramble,
    evaluate_success,
    exact_equivalence_orbit,
    grid_counts,
    load_target_catalog,
    vacancy_hole_count,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/morphologies/target_catalog.yaml"


@pytest.fixture(scope="module")
def targets():
    _, loaded = load_target_catalog(CATALOG)
    return loaded


def _swap(grid, first, second):
    mutable = [list(row) for row in grid]
    mutable[first[0]][first[1]], mutable[second[0]][second[1]] = (
        mutable[second[0]][second[1]],
        mutable[first[0]][first[1]],
    )
    return tuple(tuple(row) for row in mutable)


def test_catalog_covers_every_required_family_and_two_hole_scales(targets) -> None:
    assert {target.family for target in targets} == REQUIRED_FAMILIES
    hole_targets = [target for target in targets if target.family == "holes"]
    assert len(hole_targets) == 2
    assert {tuple(target.success["vacancyHoles"]) for target in hole_targets} == {
        (1, 1),
        (2, 2),
    }
    assert all(target.local_grammar_candidate for target in targets)


def test_exact_equivalence_orbits_are_nonempty_count_preserving_and_successful(
    targets,
) -> None:
    for target in targets:
        orbit = exact_equivalence_orbit(target)
        assert orbit
        assert all(
            grid_counts(member) == Counter(target.cell_type_counts) for member in orbit
        )
        assert all(evaluate_success(member, target)["success"] for member in orbit)
    assert len(exact_equivalence_orbit(targets[0])) == 2
    ring = next(target for target in targets if target.family == "rings")
    assert len(exact_equivalence_orbit(ring)) == 25


def test_every_fixture_has_a_replayable_legal_adjacent_swap_witness(targets) -> None:
    for target in targets:
        scramble = deterministic_scramble(target.grid)
        assert grid_counts(scramble) == grid_counts(target.grid)
        assert scramble != target.grid
        witness = adjacent_swap_witness(scramble, target.grid)
        assert witness
        assert all(
            abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
            for first, second in witness
        )
        assert apply_swap_witness(scramble, witness) == target.grid


def test_declared_boundary_variants_and_interior_controls_are_consistent(
    targets,
) -> None:
    exercised = 0
    for target in targets:
        accepted = target.validation.get("acceptedBoundarySwap")
        rejected = target.validation.get("rejectedInteriorSwap")
        if accepted:
            exercised += 1
            accepted_grid = _swap(target.grid, *[tuple(item) for item in accepted])
            accepted_result = evaluate_success(accepted_grid, target)
            assert accepted_result["success"], target.target_id
            assert accepted_result["mismatchCount"] == 2
            assert accepted_result["boundaryViolationCount"] == 0
        if rejected:
            rejected_grid = _swap(target.grid, *[tuple(item) for item in rejected])
            rejected_result = evaluate_success(rejected_grid, target)
            assert not rejected_result["success"], target.target_id
    assert exercised == 3


def test_exact_counts_are_a_hard_success_gate(targets) -> None:
    for target in targets:
        mutable = [list(row) for row in target.grid]
        replacement = next(
            token for token in target.cell_type_counts if token != mutable[0][0]
        )
        mutable[0][0] = replacement
        result = evaluate_success(mutable, target)
        assert not result["success"]
        assert not result["countMatch"]


def test_vacancy_topology_distinguishes_zero_one_and_two_holes(targets) -> None:
    observed = {
        target.target_id: vacancy_hole_count(target.grid, target.vacancy_label)
        for target in targets
    }
    assert observed["ring_core_shell_translatable"] == 0
    assert observed["tissue_single_hole"] == 1
    assert observed["tissue_two_holes"] == 2


def test_witness_rejects_count_infeasible_goal(targets) -> None:
    target = targets[0]
    invalid = [list(row) for row in target.grid]
    invalid[0][0] = "B"
    with pytest.raises(ValueError, match="counts differ"):
        adjacent_swap_witness(target.grid, tuple(tuple(row) for row in invalid))
