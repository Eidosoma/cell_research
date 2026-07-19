from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from src.morph2d.environments import evaluate_conjunctive, load_environment_catalog
from src.morph2d.grammar import load_grammar_catalog
from src.morph2d.movements import (
    DEFERRED_KINDS,
    ENABLED_KINDS,
    LEDGER_FIELDS,
    MovementValidationError,
    canonical_batch_result_bytes,
    canonical_movement_state_bytes,
    canonical_proposal_bytes,
    conflict_order_key,
    initial_movement_state,
    inverse_proposal,
    make_proposal,
    movement_state_sha256,
    parse_movement_state,
    parse_proposal,
    replay_batch_result,
    resolve_batch,
    state_tokens,
    validate_proposal,
)
from src.morph2d.targets import load_target_catalog


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ROOT / "configs/morphologies/environment_catalog.yaml"
MOVEMENTS = ROOT / "configs/morphologies/movement_catalog.yaml"
TARGETS = ROOT / "configs/morphologies/target_catalog.yaml"
GRAMMARS = ROOT / "configs/morphologies/grammar_catalog.yaml"


@pytest.fixture(scope="module")
def context():
    movement_catalog = yaml.safe_load(MOVEMENTS.read_text(encoding="utf-8"))
    _, environments = load_environment_catalog(ENVIRONMENTS)
    _, targets = load_target_catalog(TARGETS)
    _, grammars = load_grammar_catalog(GRAMMARS)
    return (
        movement_catalog,
        {item.environment_id: item for item in environments},
        {item.target_id: item for item in targets},
        {item.grammar_id: item for item in grammars},
    )


def _proposals(state, raw_proposals):
    return [
        make_proposal(
            state,
            item["kind"],
            item["route"],
            rotation_direction=item["rotationDirection"],
        )
        for item in raw_proposals
    ]


def _fixture(context, fixture_id):
    return next(
        item for item in context[0]["fixtures"] if item["fixtureId"] == fixture_id
    )


def test_catalog_freezes_enabled_deferred_and_signal_contract(context) -> None:
    catalog = context[0]
    assert catalog["schemaVersion"] == "e06.s04.movement-catalog.v1"
    assert {item["kind"] for item in catalog["enabledMovements"]} == ENABLED_KINDS
    assert {item["kind"] for item in catalog["deferredMovements"]} == DEFERRED_KINDS
    assert catalog["signalBudget"]["boundarySignalReadsForMovementLegality"] == 0
    assert catalog["conflictContract"]["proposalOrderInvariant"] is True
    assert catalog["evaluationContract"]["excludedInputs"] == [
        "S02_local_grammar",
        "S01_global_completion",
        "analysis_labels",
        "future_state",
    ]


def test_enabled_fixture_proposals_validate_on_square_hex_irregular_and_periodic(
    context,
) -> None:
    seen = set()
    geometries = set()
    boundaries = set()
    for raw in context[0]["fixtures"]:
        environment = context[1][raw["environmentId"]]
        state = initial_movement_state(environment)
        for proposal in _proposals(state, raw["proposals"]):
            validation = validate_proposal(environment, state, proposal)
            if raw["fixtureId"] == "fixed_boundary_mixed_validation" and (
                "r0_c1" in proposal.route
            ):
                assert not validation.valid
                assert validation.reason == "fixed_boundary_site"
                continue
            assert validation.valid, (raw["fixtureId"], validation.reason)
            seen.add(proposal.kind)
        geometries.add(environment.geometry)
        boundaries.add(environment.boundary_mode)
    assert seen == ENABLED_KINDS
    assert geometries == {"square", "hexagonal", "irregular"}
    assert boundaries == {"bounded", "periodic"}


def test_every_fixture_commit_preserves_identity_composition_roles_and_signal_budget(
    context,
) -> None:
    for raw in context[0]["fixtures"]:
        environment = context[1][raw["environmentId"]]
        state = initial_movement_state(environment)
        result = resolve_batch(
            environment,
            state,
            _proposals(state, raw["proposals"]),
            batch_nonce=raw["batchNonce"],
        )
        assert result["invariants"]["success"]
        assert all(result["invariants"]["checks"].values())
        assert result["costLedger"]["boundarySignalReads"] == 0
        assert result["boundarySignalBudget"]["preserved"]
        assert set(result["costLedger"]) == set(LEDGER_FIELDS)
        assert all(value >= 0 for value in result["costLedger"].values())


def test_periodic_vacancy_move_uses_declared_wrap_edge_without_boundary_read(
    context,
) -> None:
    raw = _fixture(context, "vacancy_move_periodic_seam")
    environment = context[1][raw["environmentId"]]
    state = initial_movement_state(environment)
    proposal = _proposals(state, raw["proposals"])[0]
    result = resolve_batch(
        environment, state, [proposal], batch_nonce=raw["batchNonce"]
    )
    post = parse_movement_state(result["postState"])
    assert post.occupant_map["r0_c0"].kind == "cell"
    assert post.occupant_map["r0_c4"].kind == "vacancy"
    assert result["costLedger"]["displacedCells"] == 1
    assert result["costLedger"]["displacedVacancies"] == 1
    assert result["costLedger"]["totalGraphDisplacement"] == 2
    assert result["costLedger"]["boundarySignalReads"] == 0


def test_short_exchange_reserves_middle_site_and_charges_endpoint_distance(
    context,
) -> None:
    raw = _fixture(context, "conflicting_and_disjoint_square_batch")
    environment = context[1][raw["environmentId"]]
    state = initial_movement_state(environment)
    proposals = _proposals(state, raw["proposals"])
    result = resolve_batch(environment, state, proposals, batch_nonce=raw["batchNonce"])
    decisions = {item["proposalId"]: item for item in result["decisions"]}
    short = next(item for item in proposals if item.kind == "short_exchange")
    overlapping = next(
        item
        for item in proposals
        if item.kind == "adjacent_swap" and item.route == ("r2_c3", "r3_c3")
    )
    assert decisions[short.proposal_id]["reservedSites"] == [
        "r2_c2",
        "r2_c3",
        "r3_c3",
    ]
    assert {
        decisions[short.proposal_id]["outcome"],
        decisions[overlapping.proposal_id]["outcome"],
    } == {"accepted", "conflict_lost"}

    single = resolve_batch(environment, state, [short], batch_nonce="short-cost-v1")
    ledger = single["costLedger"]
    assert ledger["reservedSiteClaims"] == 3
    assert ledger["displacedEntities"] == 2
    assert ledger["cellGraphDisplacement"] == 4
    assert ledger["totalGraphDisplacement"] == 4


def test_conflict_resolution_is_invariant_to_all_120_input_permutations(
    context,
) -> None:
    raw = _fixture(context, "conflicting_and_disjoint_square_batch")
    environment = context[1][raw["environmentId"]]
    state = initial_movement_state(environment)
    proposals = _proposals(state, raw["proposals"])
    canonical = None
    for permutation in itertools.permutations(proposals):
        result = resolve_batch(
            environment, state, permutation, batch_nonce=raw["batchNonce"]
        )
        encoded = canonical_batch_result_bytes(result)
        if canonical is None:
            canonical = encoded
        assert encoded == canonical


def test_equal_priority_tie_uses_lexicographic_proposal_id(context) -> None:
    raw = _fixture(context, "conflicting_and_disjoint_square_batch")
    environment = context[1][raw["environmentId"]]
    state = initial_movement_state(environment)
    proposals = _proposals(state, raw["proposals"])
    first, second = sorted(proposals, key=lambda item: item.proposal_id)[:2]
    batch_id = "0" * 64
    assert conflict_order_key(
        batch_id, first, priority_override=7
    ) < conflict_order_key(batch_id, second, priority_override=7)


def test_deferred_illegal_stale_obstacle_and_fixed_probes_reject_exactly(
    context,
) -> None:
    for raw in context[0]["validationProbes"]:
        environment = context[1][raw["environmentId"]]
        state = initial_movement_state(environment)
        kwargs = {}
        if raw.get("mutation") == "staleStateHash":
            kwargs["observed_state_sha256"] = "0" * 64
        elif raw.get("mutation") == "staleResourceIdentity":
            kwargs["expected_occupants"] = {
                site_id: (
                    "stale:identity"
                    if index == 0
                    else state.occupant_map[site_id].occupant_id
                )
                for index, site_id in enumerate(raw["route"])
                if site_id in state.occupant_map
            }
        elif raw.get("mutation") == "actorMismatch":
            kwargs["actor_id"] = "wrong:actor"
        proposal = make_proposal(
            state,
            raw["kind"],
            raw["route"],
            rotation_direction=raw["rotationDirection"],
            **kwargs,
        )
        validation = validate_proposal(environment, state, proposal)
        assert not validation.valid, raw["probeId"]
        assert validation.reason == raw["expectedReason"], raw["probeId"]


@pytest.mark.parametrize(
    "fixture_id",
    [
        "adjacent_swap_square_bounded",
        "vacancy_move_periodic_seam",
        "short_exchange_square_distance_two",
        "rotation_square_cycle",
        "rotation_hexagonal_six_cycle",
        "short_exchange_irregular_explicit_path",
        "vacancy_move_obstacle_environment",
    ],
)
def test_each_enabled_kind_has_identity_exact_reversibility_fixture(
    context, fixture_id
) -> None:
    raw = _fixture(context, fixture_id)
    environment = context[1][raw["environmentId"]]
    initial = initial_movement_state(environment)
    proposal = _proposals(initial, raw["proposals"])[0]
    forward = resolve_batch(
        environment, initial, [proposal], batch_nonce=raw["batchNonce"]
    )
    post = parse_movement_state(forward["postState"])
    inverse = inverse_proposal(post, proposal)
    backward = resolve_batch(
        environment, post, [inverse], batch_nonce=f"{raw['batchNonce']}-inverse"
    )
    final = parse_movement_state(backward["postState"])
    assert final.occupancy == initial.occupancy
    assert final.transition_index == initial.transition_index + 2


def test_state_proposal_batch_serialization_and_exact_replay(context) -> None:
    for raw in context[0]["fixtures"]:
        environment = context[1][raw["environmentId"]]
        state = initial_movement_state(environment)
        reparsed_state = parse_movement_state(
            json.loads(canonical_movement_state_bytes(state))
        )
        assert canonical_movement_state_bytes(
            reparsed_state
        ) == canonical_movement_state_bytes(state)
        proposals = _proposals(state, raw["proposals"])
        for proposal in proposals:
            assert canonical_proposal_bytes(
                parse_proposal(json.loads(canonical_proposal_bytes(proposal)))
            ) == canonical_proposal_bytes(proposal)
        result = resolve_batch(
            environment, state, proposals, batch_nonce=raw["batchNonce"]
        )
        replayed = replay_batch_result(
            environment, json.loads(canonical_batch_result_bytes(result))
        )
        assert canonical_batch_result_bytes(replayed) == canonical_batch_result_bytes(
            result
        )


def test_local_grammar_and_global_completion_fields_remain_separate(context) -> None:
    environment = context[1]["square_bounded_occupied_stripes"]
    target = context[2][environment.target_binding["targetId"]]
    grammar = context[3][environment.target_binding["grammarId"]]
    cases = {
        "adjacent_swap": (["r2_c0", "r3_c0"], 0),
        "short_exchange": (["r2_c4", "r2_c5", "r3_c5"], 0),
        "rotation": (["r2_c0", "r2_c1", "r3_c1", "r3_c0"], 1),
    }
    records = {}
    for kind, (route, direction) in cases.items():
        initial = initial_movement_state(environment)
        proposal = make_proposal(initial, kind, route, rotation_direction=direction)
        result = resolve_batch(
            environment, initial, [proposal], batch_nonce=f"evaluation-{kind}"
        )
        post = parse_movement_state(result["postState"])
        moved_environment = replace(
            environment,
            initial_state={**environment.initial_state, **state_tokens(post)},
        )
        records[kind] = evaluate_conjunctive(moved_environment, target, grammar)
    assert records["adjacent_swap"]["localGrammar"]["accepted"]
    assert not records["adjacent_swap"]["globalAudit"]["success"]
    assert records["short_exchange"]["localGrammar"]["accepted"]
    assert records["short_exchange"]["globalAudit"]["success"]
    assert not records["rotation"]["localGrammar"]["accepted"]
    assert records["rotation"]["globalAudit"]["success"]


def test_batch_rejects_duplicate_ids_and_proposal_hash_tampering(context) -> None:
    environment = context[1]["square_bounded_occupied_stripes"]
    state = initial_movement_state(environment)
    proposal = make_proposal(state, "adjacent_swap", ["r2_c0", "r3_c0"])
    with pytest.raises(MovementValidationError, match="unique"):
        resolve_batch(environment, state, [proposal, proposal], batch_nonce="duplicate")
    raw = json.loads(canonical_proposal_bytes(proposal))
    raw["actorId"] = "tampered"
    with pytest.raises(MovementValidationError, match="ID mismatch"):
        parse_proposal(raw)


def test_rotation_with_vacancy_is_not_silently_enabled(context) -> None:
    environment = context[1]["square_periodic_vacancy"]
    state = initial_movement_state(environment)
    proposal = make_proposal(
        state,
        "rotation",
        ["r0_c0", "r0_c1", "r1_c1", "r1_c0"],
        rotation_direction=1,
    )
    validation = validate_proposal(environment, state, proposal)
    assert not validation.valid
    assert validation.reason == "rotation_requires_fully_occupied_cycle"


def test_state_hash_is_engine_state_not_evaluation_output(context) -> None:
    environment = context[1]["square_bounded_occupied_stripes"]
    state = initial_movement_state(environment)
    payload = json.loads(canonical_movement_state_bytes(state))
    assert payload["stateSha256"] == movement_state_sha256(state)
    encoded = json.dumps(payload, sort_keys=True)
    assert "localGrammar" not in encoded
    assert "globalAudit" not in encoded
    assert "completion" not in encoded
