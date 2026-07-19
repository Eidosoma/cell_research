from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.morph2d.grammar import (
    GrammarValidationError,
    catalog_to_dict,
    detect_contradictions,
    load_grammar_catalog,
    parse_grammar,
    parse_grammar_catalog,
    score_grid,
)
from src.morph2d.targets import (
    evaluate_success,
    exact_equivalence_orbit,
    load_target_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
GRAMMARS = ROOT / "configs/morphologies/grammar_catalog.yaml"
HAND_FIXTURES = ROOT / "configs/morphologies/grammar_hand_fixtures.yaml"
ADVERSARIAL_FIXTURES = ROOT / "configs/morphologies/grammar_adversarial_fixtures.yaml"
TARGETS = ROOT / "configs/morphologies/target_catalog.yaml"


@pytest.fixture(scope="module")
def grammar_context():
    metadata, grammars = load_grammar_catalog(GRAMMARS)
    _, targets = load_target_catalog(TARGETS)
    return metadata, grammars, targets


def _swap(grid, first, second):
    mutable = [list(row) for row in grid]
    mutable[first[0]][first[1]], mutable[second[0]][second[1]] = (
        mutable[second[0]][second[1]],
        mutable[first[0]][first[1]],
    )
    return tuple(tuple(row) for row in mutable)


def test_parser_semantic_round_trip_is_exact(grammar_context) -> None:
    metadata, grammars, _ = grammar_context
    reparsed_metadata, reparsed_grammars = parse_grammar_catalog(
        catalog_to_dict(metadata, grammars)
    )
    assert reparsed_metadata == metadata
    assert reparsed_grammars == grammars
    assert len(grammars) == 7


def test_hand_scored_fixtures_match_every_typed_constraint() -> None:
    raw = yaml.safe_load(HAND_FIXTURES.read_text(encoding="utf-8"))
    grammar = parse_grammar(raw["grammar"])
    for case in raw["cases"]:
        result = score_grid(case["rows"], grammar)
        assert result["accepted"] is case["expectedAccepted"]
        assert result["hardViolationCount"] == case["expectedHardViolationCount"]
        assert result["softScore"] == pytest.approx(case["expectedSoftScore"])
        observed = {
            item["constraintId"]: item["observed"] for item in result["constraints"]
        }
        assert observed == case["expectedObservations"]


def test_every_exact_target_equivalence_scores_one_and_is_accepted(
    grammar_context,
) -> None:
    _, grammars, targets = grammar_context
    grammar_by_target = {grammar.target_id: grammar for grammar in grammars}
    observed_states = 0
    for target in targets:
        grammar = grammar_by_target[target.target_id]
        for member in exact_equivalence_orbit(target):
            observed_states += 1
            result = score_grid(member, grammar)
            assert result["accepted"], (target.target_id, result)
            assert result["softScore"] == pytest.approx(1.0)
            assert result["hardViolationCount"] == 0
    assert observed_states == 148


def test_declared_s01_boundary_variants_remain_grammar_accepted(
    grammar_context,
) -> None:
    _, grammars, targets = grammar_context
    grammar_by_target = {grammar.target_id: grammar for grammar in grammars}
    exercised = 0
    for target in targets:
        swap = target.validation.get("acceptedBoundarySwap")
        if not swap:
            continue
        exercised += 1
        candidate = _swap(target.grid, tuple(swap[0]), tuple(swap[1]))
        assert evaluate_success(candidate, target)["success"]
        assert score_grid(candidate, grammar_by_target[target.target_id])["accepted"]
    assert exercised == 3


def test_predeclared_structural_challenges_falsify_every_target_grammar(
    grammar_context,
) -> None:
    _, grammars, targets = grammar_context
    grammar_by_target = {grammar.target_id: grammar for grammar in grammars}
    target_by_id = {target.target_id: target for target in targets}
    fixtures = yaml.safe_load(ADVERSARIAL_FIXTURES.read_text(encoding="utf-8"))[
        "fixtures"
    ]
    assert len(fixtures) == len(targets) == 7
    for fixture in fixtures:
        grammar_result = score_grid(
            fixture["rows"], grammar_by_target[fixture["targetId"]]
        )
        target_result = evaluate_success(
            fixture["rows"], target_by_id[fixture["targetId"]]
        )
        assert grammar_result["accepted"] is fixture["expectedGrammarAccepted"]
        assert target_result["success"] is fixture["expectedS01Accepted"]
        assert target_result["mismatchCount"] >= fixture["minimumMismatch"]


def test_hard_constraints_override_high_soft_score(grammar_context) -> None:
    _, grammars, targets = grammar_context
    grammar = next(
        item for item in grammars if item.target_id == "layers_three_ordered_tissues"
    )
    target = next(item for item in targets if item.target_id == grammar.target_id)
    candidate = _swap(target.grid, (0, 0), (6, 0))
    result = score_grid(candidate, grammar)
    assert result["softScore"] >= grammar.acceptance_threshold
    assert result["hardViolationCount"] >= 1
    assert not result["accepted"]


def test_contradiction_detection_rejects_empty_hard_interval_intersection() -> None:
    catalog = yaml.safe_load(GRAMMARS.read_text(encoding="utf-8"))
    raw = deepcopy(catalog["grammars"][0])
    conflicting = deepcopy(raw["constraints"][1])
    conflicting["constraintId"] = "conflicting_A_B_interval"
    conflicting["priority"] = "hard"
    conflicting["parameters"]["min"] = 30
    conflicting["parameters"]["max"] = 31
    raw["constraints"][1]["priority"] = "hard"
    raw["constraints"].append(conflicting)
    grammar = parse_grammar(raw, reject_contradictions=False)
    contradictions = detect_contradictions(grammar)
    assert any("empty intersection" in item for item in contradictions)
    with pytest.raises(GrammarValidationError, match="empty intersection"):
        parse_grammar(raw)


def test_soft_interval_disagreement_is_resolved_by_weighting() -> None:
    catalog = yaml.safe_load(GRAMMARS.read_text(encoding="utf-8"))
    raw = deepcopy(catalog["grammars"][0])
    conflicting = deepcopy(raw["constraints"][1])
    conflicting["constraintId"] = "soft_A_B_preference_elsewhere"
    conflicting["parameters"]["min"] = 30
    conflicting["parameters"]["max"] = 31
    raw["constraints"].append(conflicting)
    grammar = parse_grammar(raw)
    assert not any(
        "empty intersection" in item for item in detect_contradictions(grammar)
    )


def test_contradiction_detection_rejects_impossible_neighbor_capacity() -> None:
    catalog = yaml.safe_load(GRAMMARS.read_text(encoding="utf-8"))
    raw = deepcopy(catalog["grammars"][0])
    neighbor = next(
        item for item in raw["constraints"] if item["kind"] == "neighbor_count"
    )
    neighbor["parameters"]["min"] = 5
    neighbor["parameters"]["max"] = 5
    grammar = parse_grammar(raw, reject_contradictions=False)
    assert any(
        "neighborhood capacity" in item for item in detect_contradictions(grammar)
    )
    with pytest.raises(GrammarValidationError, match="neighborhood capacity"):
        parse_grammar(raw)
