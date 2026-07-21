from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.morph2d.grammar import load_grammar_catalog
from src.morph2d.morphometrics import (
    ALL_COST_SOURCE_COLUMNS,
    block_upscale,
    categorical_transport_distance,
    component_counts,
    heterotypic_boundary_edges,
    load_morphometric_catalog,
    occupied_region_purity,
    score_morphology,
    standardize_cost_components,
    standardize_endpoint_time,
    symmetry_mismatch_fraction,
)
from src.morph2d.targets import (
    exact_equivalence_orbit,
    load_target_catalog,
    vacancy_hole_count,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET_CATALOG = ROOT / "configs/morphologies/target_catalog.yaml"
GRAMMAR_CATALOG = ROOT / "configs/morphologies/grammar_catalog.yaml"
METRIC_CATALOG = ROOT / "configs/morphologies/morphometric_catalog.yaml"
ADVERSARIAL = ROOT / "configs/morphologies/grammar_adversarial_fixtures.yaml"


@pytest.fixture(scope="module")
def assets():
    _, targets = load_target_catalog(TARGET_CATALOG)
    _, grammars = load_grammar_catalog(GRAMMAR_CATALOG)
    return (
        {target.target_id: target for target in targets},
        {grammar.target_id: grammar for grammar in grammars},
    )


def swap(grid, first, second):
    mutable = [list(row) for row in grid]
    mutable[first[0]][first[1]], mutable[second[0]][second[1]] = (
        mutable[second[0]][second[1]],
        mutable[first[0]][first[1]],
    )
    return tuple(tuple(row) for row in mutable)


def test_frozen_catalog_has_complete_panel_and_no_redefinition() -> None:
    catalog = load_morphometric_catalog(METRIC_CATALOG)
    assert len(catalog["metricDefinitions"]) == 10
    assert catalog["decisionRules"]["completionRedefinition"] == "forbidden"
    assert catalog["metricDefinitions"]["componentWiseCost"]["scalarCollapse"] == "forbidden"


def test_all_exact_equivalence_states_have_zero_geometric_and_transport_distance(assets) -> None:
    targets, grammars = assets
    observed = 0
    for target_id, target in targets.items():
        orbit = exact_equivalence_orbit(target)
        for grid in orbit:
            result = score_morphology(
                grid, target, grammars[target_id], equivalence_orbit=orbit
            )
            observed += 1
            assert result["s01GlobalSuccess"]
            assert result["s02GrammarAccepted"]
            assert result["conjunctiveCompletion"]
            assert result["imageDistanceModuloEquivalence"] == 0
            assert result["transportDistance"] == 0
            assert result["transportRawCost"] == 0
            assert result["componentExcess"] == 0
            assert result["holeCountError"] == 0
            assert result["targetBoundaryRelativeError"] == 0
            assert result["occupiedRegionPurity"] == 1
            if result["symmetryTransformCount"]:
                assert result["symmetryMismatchFraction"] == 0
    assert observed == 148


def test_all_three_s01_tolerated_variants_retain_local_and_global_acceptance(assets) -> None:
    targets, grammars = assets
    observed = 0
    for target_id, target in targets.items():
        declared = target.validation.get("acceptedBoundarySwap")
        if not declared:
            continue
        observed += 1
        grid = swap(target.grid, tuple(declared[0]), tuple(declared[1]))
        result = score_morphology(grid, target, grammars[target_id])
        assert result["s01GlobalSuccess"]
        assert result["s02GrammarAccepted"]
        assert result["s01MismatchCount"] == 2
        displacement = abs(declared[0][0] - declared[1][0]) + abs(
            declared[0][1] - declared[1][1]
        )
        assert result["transportRawCost"] == 2 * displacement
    assert observed == 3


def test_each_frozen_s02_false_positive_remains_detectable_by_global_panel(assets) -> None:
    targets, grammars = assets
    raw = yaml.safe_load(ADVERSARIAL.read_text(encoding="utf-8"))
    assert len(raw["fixtures"]) == 7
    for fixture in raw["fixtures"]:
        target_id = fixture["targetId"]
        result = score_morphology(
            fixture["rows"], targets[target_id], grammars[target_id]
        )
        assert result["s02GrammarAccepted"]
        assert not result["s01GlobalSuccess"]
        assert not result["conjunctiveCompletion"]
        assert (
            not result["s01GeometryMatch"]
            or result["componentExcess"] > 0
            or result["holeCountError"] > 0
            or result["s01BoundaryViolationCount"] > 0
        )


@pytest.mark.parametrize(
    ("rows", "expected_components", "expected_holes", "expected_edges"),
    [
        (["TTTTT"] * 5, {"T": 1, ".": 0}, 0, 0),
        (["TTTTT", "T...T", "T...T", "T...T", "TTTTT"], {"T": 1, ".": 1}, 1, 12),
        (["TT.TT", "TT.TT", "TT.TT", "TTTTT", "TTTTT"], {"T": 1, ".": 1}, 0, 7),
        (["TTTTT", "T.TTT", "TTTTT", "TTT.T", "TTTTT"], {"T": 1, ".": 2}, 2, 8),
    ],
)
def test_hand_labeled_component_hole_and_boundary_fixtures(
    rows, expected_components, expected_holes, expected_edges
) -> None:
    assert component_counts(rows, ["T", "."]) == expected_components
    assert vacancy_hole_count(tuple(tuple(row) for row in rows), ".") == expected_holes
    assert heterotypic_boundary_edges(rows) == expected_edges


def test_transport_is_exact_for_an_adjacent_cross_token_exchange() -> None:
    reference = (tuple("AAB"), tuple("AAB"), tuple("BBB"))
    candidate = swap(reference, (0, 1), (0, 2))
    distance, raw_cost, valid = categorical_transport_distance(candidate, reference)
    assert valid
    assert raw_cost == 2
    assert distance == pytest.approx(2 / (9 * 4))
    invalid = [list(row) for row in candidate]
    invalid[0][0] = "B"
    assert categorical_transport_distance(invalid, reference) == (None, None, False)


def test_translation_equivalence_gives_zero_transport(assets) -> None:
    targets, grammars = assets
    target = targets["ring_core_shell_translatable"]
    orbit = exact_equivalence_orbit(target)
    translated = orbit[-1]
    result = score_morphology(translated, target, grammars[target.target_id], equivalence_orbit=orbit)
    assert result["s01ReferenceKey"] == "\n".join("".join(row) for row in translated)
    assert result["transportDistance"] == 0


def test_purity_and_symmetry_are_distinct_diagnostics() -> None:
    reference = ["AAAAA", "ABBBA", "ABBBA", "ABBBA", "AAAAA"]
    asymmetric = ["AAAAA", "ABBBA", "ABBBA", "ABBAA", "AAAAB"]
    purity = occupied_region_purity(asymmetric, reference, ".")
    symmetry, count = symmetry_mismatch_fraction(
        asymmetric, reference, vacancy_label=".", translation_mode="none"
    )
    assert purity is not None and purity > 0.8
    assert count >= 1
    assert symmetry is not None and symmetry > 0


def test_block_resolution_preserves_declared_size_robust_metrics() -> None:
    reference = tuple(tuple(row) for row in ["TTTTT", "T...T", "T...T", "T...T", "TTTTT"])
    candidate = swap(reference, (0, 2), (1, 2))
    baseline = {}
    for scale in (1, 2, 3):
        scaled_reference = block_upscale(reference, scale)
        scaled_candidate = block_upscale(candidate, scale)
        site_count = len(scaled_candidate) * len(scaled_candidate[0])
        mismatch = sum(
            scaled_candidate[row][col] != scaled_reference[row][col]
            for row in range(len(scaled_candidate))
            for col in range(len(scaled_candidate[0]))
        ) / site_count
        purity = occupied_region_purity(scaled_candidate, scaled_reference, ".")
        boundary = heterotypic_boundary_edges(scaled_candidate) / (site_count**0.5)
        symmetry, _ = symmetry_mismatch_fraction(
            scaled_candidate,
            scaled_reference,
            vacancy_label=".",
            translation_mode="none",
        )
        transport, _, _ = categorical_transport_distance(scaled_candidate, scaled_reference)
        current = {
            "mismatch": mismatch,
            "purity": purity,
            "boundary": boundary,
            "symmetry": symmetry,
            "components": component_counts(scaled_candidate, ["T", "."]),
            "holes": vacancy_hole_count(scaled_candidate, "."),
            "transport": transport,
        }
        if scale == 1:
            baseline = current
        else:
            assert current["mismatch"] == baseline["mismatch"]
            assert current["purity"] == baseline["purity"]
            assert current["boundary"] == pytest.approx(baseline["boundary"], abs=1e-12)
            assert current["symmetry"] == pytest.approx(baseline["symmetry"], abs=1e-12)
            assert current["components"] == baseline["components"]
            assert current["holes"] == baseline["holes"]
            assert abs(current["transport"] - baseline["transport"]) <= 0.03


def test_endpoint_time_does_not_turn_exact_terminal_restoration_into_maintenance() -> None:
    central_exact = standardize_endpoint_time(
        "S12",
        {
            "formationEligible": False,
            "repairEligible": False,
            "conjunctiveCompletionByBudget": True,
            "firstCompletionTransition": 0,
            "eventBudgetTransitions": 128,
        },
    )
    assert central_exact == {
        "timeEndpointType": "not_applicable",
        "timeEligible": False,
        "timeEventObserved": False,
        "timeToEndpoint": None,
        "timeRightCensored": False,
        "timeAnalysisTransition": None,
    }
    repair = standardize_endpoint_time(
        "S10",
        {
            "repairEligible": True,
            "lesionRepairByBudget": False,
            "firstRepairTransition": None,
            "eventBudgetTransitions": 64,
        },
    )
    assert repair["timeEndpointType"] == "repair"
    assert repair["timeRightCensored"]
    assert repair["timeAnalysisTransition"] == 64


def test_cost_components_keep_structural_missingness() -> None:
    row = {
        "acceptedMovements": 4,
        "totalGraphDisplacement": 8,
        "channelTotalInformationBits": 12,
    }
    costs = standardize_cost_components("S09", row)
    assert len(costs) == len(ALL_COST_SOURCE_COLUMNS)
    assert costs["cost_acceptedMovements"] == 4
    assert costs["cost_totalGraphDisplacement"] == 8
    assert costs["cost_controllerComputeUnits"] is None
