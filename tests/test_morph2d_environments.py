from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.morph2d.environments import (
    EnvironmentValidationError,
    boundary_observation,
    canonical_environment_bytes,
    connected_components,
    evaluate_conjunctive,
    load_environment_catalog,
    neighbor_map,
    parse_environment_spec,
    parse_serialized_environment,
    replay_environment,
    topology_summary,
)
from src.morph2d.grammar import load_grammar_catalog
from src.morph2d.targets import load_target_catalog


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/morphologies/environment_catalog.yaml"
TARGETS = ROOT / "configs/morphologies/target_catalog.yaml"
GRAMMARS = ROOT / "configs/morphologies/grammar_catalog.yaml"


@pytest.fixture(scope="module")
def context():
    metadata, environments = load_environment_catalog(CATALOG)
    _, targets = load_target_catalog(TARGETS)
    _, grammars = load_grammar_catalog(GRAMMARS)
    return metadata, environments, targets, grammars


def _environment(context, environment_id):
    return next(item for item in context[1] if item.environment_id == environment_id)


def test_catalog_covers_planned_environment_axes(context) -> None:
    metadata, environments, _, _ = context
    assert len(environments) == 9
    assert {item.geometry for item in environments} == {
        "square",
        "hexagonal",
        "irregular",
    }
    assert {item.boundary_mode for item in environments} == {"bounded", "periodic"}
    assert {item.occupancy_mode for item in environments} == {
        "fully_occupied",
        "vacancy_enabled",
    }
    assert any(
        any(site.role == "obstacle" for site in item.sites) for item in environments
    )
    assert any(
        any(site.role == "fixed_boundary" for site in item.sites)
        for item in environments
    )
    assert len(metadata["unsupportedCombinations"]) == 8


def test_all_graphs_are_undirected_symmetric_and_connected(context) -> None:
    for environment in context[1]:
        neighbors = neighbor_map(environment)
        for site_id, others in neighbors.items():
            for other in others:
                assert site_id in neighbors[other]
        assert len(connected_components(environment)) == 1
        assert topology_summary(environment)["componentCount"] == 1


def test_periodic_square_and_hex_wrap_without_boundary_signal(context) -> None:
    square = _environment(context, "square_periodic_vacancy")
    assert neighbor_map(square)["r0_c0"] == (
        "r0_c1",
        "r0_c4",
        "r1_c0",
        "r3_c0",
    )
    assert boundary_observation(square, "r0_c0") == {
        "observable": False,
        "boundaryTags": [],
        "obstacleContacts": [],
        "isFixedBoundary": False,
        "fixedBoundaryNeighbors": [],
    }
    hexagonal = _environment(context, "hexagonal_periodic_vacancy")
    assert all(len(items) == 6 for items in neighbor_map(hexagonal).values())
    assert topology_summary(hexagonal)["naturalBoundarySiteCount"] == 0


def test_bounded_regular_and_explicit_irregular_boundary_semantics(context) -> None:
    square = _environment(context, "square_bounded_occupied_stripes")
    assert boundary_observation(square, "r0_c0")["boundaryTags"] == [
        "north",
        "west",
    ]
    assert boundary_observation(square, "r4_c4")["boundaryTags"] == []

    irregular = _environment(context, "irregular_bounded_occupied")
    assert boundary_observation(irregular, "a")["boundaryTags"] == ["north", "west"]
    assert boundary_observation(irregular, "e")["boundaryTags"] == []
    # Boundary is explicit rather than inferred from degree: c has degree two and tags,
    # while topology authority remains the declared edge list.
    assert len(neighbor_map(irregular)["c"]) == 2


def test_obstacle_and_fixed_boundary_are_distinct_site_roles(context) -> None:
    obstacle = _environment(context, "square_bounded_vacancy_obstacle")
    assert obstacle.initial_state["r2_c2"] == "#"
    assert "r2_c2" not in neighbor_map(obstacle)
    assert boundary_observation(obstacle, "r1_c2")["obstacleContacts"] == ["r2_c2"]
    assert topology_summary(obstacle)["vacancyCount"] == 1

    fixed = _environment(context, "square_bounded_fixed_boundary")
    fixed_sites = [site for site in fixed.sites if site.role == "fixed_boundary"]
    assert len(fixed_sites) == 16
    assert all(fixed.initial_state[site.site_id] == "F" for site in fixed_sites)
    center_signal = boundary_observation(fixed, "r1_c1")
    assert center_signal["fixedBoundaryNeighbors"] == ["r0_c1", "r1_c0"]
    assert not center_signal["isFixedBoundary"]


def test_canonical_serialization_and_replay_are_byte_deterministic(context) -> None:
    for environment in context[1]:
        encoded = canonical_environment_bytes(environment)
        reparsed = parse_serialized_environment(json.loads(encoded))
        assert canonical_environment_bytes(reparsed) == encoded
        first = replay_environment(environment)
        second = replay_environment(reparsed)
        assert first == second
        assert first["replaySha256"] == replay_environment(environment)["replaySha256"]


def test_conjunctive_contract_keeps_local_and_global_results_separate(context) -> None:
    _, environments, targets, grammars = context
    target_by_id = {item.target_id: item for item in targets}
    grammar_by_id = {item.grammar_id: item for item in grammars}
    records = {}
    for environment in environments:
        if environment.target_binding is None:
            continue
        record = evaluate_conjunctive(
            environment,
            target_by_id[environment.target_binding["targetId"]],
            grammar_by_id[environment.target_binding["grammarId"]],
        )
        records[environment.environment_id] = record
    assert len(records) == 3
    assert records["square_bounded_occupied_stripes"]["completion"]
    assert records["square_bounded_vacancy_single_hole"]["completion"]
    counterexample = records["square_bounded_occupied_s02_counterexample"]
    assert counterexample["localGrammar"]["accepted"]
    assert not counterexample["globalAudit"]["equivalenceOrGeometryMatch"]
    assert not counterexample["globalAudit"]["success"]
    assert not counterexample["completion"]


def test_hole_target_rejects_hidden_boundary_signal(context) -> None:
    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    fixture = deepcopy(raw["environments"][1])
    fixture["boundarySignal"] = {
        "mode": "none",
        "observable": False,
        "includeObstacleContact": False,
        "includeFixedRole": False,
    }
    fixture["expected"] = {}
    environment = parse_environment_spec(fixture)
    target = next(item for item in context[2] if item.target_id == "tissue_single_hole")
    grammar = next(
        item
        for item in context[3]
        if item.grammar_id == "single_hole_boundary_relations_v1"
    )
    with pytest.raises(EnvironmentValidationError, match="boundary signal"):
        evaluate_conjunctive(environment, target, grammar)


@pytest.mark.parametrize(
    ("fixture_index", "mutation", "message"),
    [
        (
            3,
            lambda item: item.update(
                boundarySignal={
                    "mode": "domain_edge_flags",
                    "observable": True,
                    "includeObstacleContact": False,
                    "includeFixedRole": False,
                }
            ),
            "no natural exterior",
        ),
        (
            3,
            lambda item: item.update(
                fixedBoundary={"mode": "sites", "sites": ["r0_c0"], "token": "F"}
            ),
            "fixed_boundary",
        ),
        (
            8,
            lambda item: item.update(
                boundarySignal={
                    "mode": "domain_edge_flags",
                    "observable": True,
                    "includeObstacleContact": False,
                    "includeFixedRole": False,
                }
            ),
            "explicit_site_tags",
        ),
        (
            0,
            lambda item: item["initialState"].update(rows=[".AAAAAAAA"] * 9),
            "fully occupied",
        ),
        (
            7,
            lambda item: item.update(
                targetBinding={
                    "targetId": "stripes_alternating_three_band",
                    "grammarId": "stripes_axis_relations_v1",
                    "evaluationProfile": "s01_s02_conjunctive_square_v1",
                }
            ),
            "only unobstructed bounded square",
        ),
    ],
)
def test_unsupported_combinations_are_rejected(
    fixture_index, mutation, message
) -> None:
    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    fixture = deepcopy(raw["environments"][fixture_index])
    mutation(fixture)
    with pytest.raises(EnvironmentValidationError, match=message):
        parse_environment_spec(fixture)


def test_disconnected_required_irregular_fixture_is_rejected() -> None:
    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    fixture = deepcopy(raw["environments"][8])
    fixture["generator"]["edges"] = [["a", "b"]]
    fixture["expected"] = {}
    with pytest.raises(EnvironmentValidationError, match="disconnected"):
        parse_environment_spec(fixture)
