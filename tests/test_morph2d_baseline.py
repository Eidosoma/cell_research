from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from src.morph2d.baseline import (
    TargetMetricTracker,
    eligible_policies,
    load_baseline_assets,
    make_initial_state,
    run_baseline_once,
    scenario_identity,
    state_grid,
)
from src.morph2d.engine import EpisodeValidationError, run_cpu_episode
from src.morph2d.grammar import score_grid
from src.morph2d.movements import (
    initial_movement_state,
    make_proposal,
    parse_movement_state,
    resolve_batch,
    validate_proposal,
)
from src.morph2d.policies import _preview_state
from src.morph2d.targets import evaluate_success


ROOT = Path(__file__).resolve().parents[1]
CATALOG = yaml.safe_load(
    (ROOT / "configs/morphologies/baseline_catalog.yaml").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def assets():
    return load_baseline_assets()


def test_frozen_catalog_expands_to_exact_screen_and_forbids_runtime_selection(
    assets,
) -> None:
    _context, targets, _grammars, _environments = assets
    condition_count = sum(
        3 * len(eligible_policies(target_id, CATALOG)) for target_id in targets
    )
    simulation = CATALOG["simulation"]
    assert condition_count == simulation["exploratoryConditionCount"] == 96
    assert (
        condition_count * simulation["exploratoryReplicatesPerCondition"]
        == simulation["exploratoryRunCount"]
        == 24_000
    )
    assert simulation["stopping"] == "fixed_event_budget_only"
    assert simulation["earlyStopping"] == "forbidden"
    assert CATALOG["promotion"]["runtimeAndWallClockForbiddenFromSelection"]
    assert CATALOG["backend"]["production"] == "canonical_s07_cpu_oracle_fallback"


def test_target_derived_environments_are_calibrated_and_exact_targets_complete(
    assets,
) -> None:
    _context, targets, grammars, environments = assets
    assert set(environments) == set(targets)
    grammar_by_target = {item.target_id: item for item in grammars.values()}
    for target_id, target in targets.items():
        environment = environments[target_id]
        assert environment.geometry == "square"
        assert environment.boundary_mode == "bounded"
        assert all(site.role != "obstacle" for site in environment.sites)
        assert target.shape == (
            int(environment.generator["rows"]),
            int(environment.generator["columns"]),
        )
        global_audit = evaluate_success(target.grid, target)
        local_audit = score_grid(target.grid, grammar_by_target[target_id])
        assert global_audit["success"]
        assert local_audit["accepted"]


def test_all_initial_families_are_deterministic_conservative_and_incomplete(
    assets,
) -> None:
    _context, targets, _grammars, environments = assets
    for target_id, target in targets.items():
        environment = environments[target_id]
        reference = initial_movement_state(environment)
        reference_ids = {item.occupant_id for _, item in reference.occupancy}
        reference_tokens = Counter(item.token for _, item in reference.occupancy)
        for family in CATALOG["initialStateFamilies"]:
            identity = scenario_identity(
                "exploratory", target_id, family, 7, "exploration_v1"
            )
            first = make_initial_state(
                environment, target, family, identity["pairingBlockId"], CATALOG
            )
            second = make_initial_state(
                environment, target, family, identity["pairingBlockId"], CATALOG
            )
            assert first == second
            assert {item.occupant_id for _, item in first.occupancy} == reference_ids
            assert (
                Counter(item.token for _, item in first.occupancy) == reference_tokens
            )
            assert not evaluate_success(state_grid(environment, first), target)[
                "success"
            ]


def test_fast_singleton_previews_are_exactly_canonical_for_all_movement_kinds(
    assets,
) -> None:
    context, _targets, _grammars, _environments = assets
    occupied = context.environments["square_bounded_occupied_stripes"]
    occupied_state = initial_movement_state(occupied)
    periodic = context.environments["square_periodic_vacancy"]
    periodic_state = initial_movement_state(periodic)
    cases = (
        (occupied, occupied_state, "adjacent_swap", ("r0_c0", "r0_c1"), 0),
        (
            occupied,
            occupied_state,
            "short_exchange",
            ("r0_c0", "r0_c1", "r0_c2"),
            0,
        ),
        (
            occupied,
            occupied_state,
            "rotation",
            ("r0_c0", "r0_c1", "r1_c1", "r1_c0"),
            1,
        ),
        (periodic, periodic_state, "vacancy_move", ("r0_c1", "r0_c0"), 0),
    )
    for environment, state, kind, route, direction in cases:
        proposal = make_proposal(state, kind, route, rotation_direction=direction)
        assert validate_proposal(environment, state, proposal).valid
        preview = _preview_state(environment, state, proposal)
        canonical = resolve_batch(
            environment, state, (proposal,), batch_nonce=f"s09-{kind}"
        )
        assert canonical["acceptedProposalIds"] == [proposal.proposal_id]
        assert preview == parse_movement_state(canonical["postState"])


def test_initial_override_and_read_only_state_audit_preserve_episode_contract(
    assets,
) -> None:
    context, targets, grammars, environments = assets
    target = targets["stripes_alternating_three_band"]
    environment = environments[target.target_id]
    identity = scenario_identity(
        "exploratory", target.target_id, "partially_correct", 0, "exploration_v1"
    )
    state = make_initial_state(
        environment,
        target,
        "partially_correct",
        identity["pairingBlockId"],
        CATALOG,
    )
    grammar = next(
        item for item in grammars.values() if item.target_id == target.target_id
    )
    base = next(item for item in context.episodes if item.policy_id == "exploration_v1")
    definition = replace(
        base,
        scenario_id=identity["scenarioId"],
        environment_id=environment.environment_id,
        relation_grammar_id=None,
        channel_mode="none",
        transitions=3,
        actor_batch_size=4,
        parameters={},
    )
    tracker = TargetMetricTracker(environment, target, grammar)
    result = run_cpu_episode(
        context,
        definition,
        initial_state_override=state,
        state_audit=tracker.observe,
        include_selected_traces=False,
    )
    assert result["initialStateSha256"]
    assert tracker.observation_count == 4
    assert tracker.finalize()["initialS01MismatchCount"] > 0
    with pytest.raises(EpisodeValidationError, match="cannot be combined"):
        run_cpu_episode(
            context,
            replace(definition, parameters={"initialSwap": ["r0_c0", "r0_c1"]}),
            initial_state_override=state,
        )


def test_one_run_keeps_local_global_fields_and_censoring_separate() -> None:
    specification = {
        "catalog": CATALOG,
        "phase": "exploratory",
        "split": "exploratory",
        "targetId": "stripes_alternating_three_band",
        "grammarId": "stripes_axis_relations_v1",
        "startFamily": "random",
        "replicate": 3,
        "policyId": "exploration_v1",
        "eventBudget": 2,
        "retainTrace": False,
    }
    first, _ = run_baseline_once(specification)
    second, _ = run_baseline_once(specification)
    for key in ("wallSeconds",):
        first.pop(key)
        second.pop(key)
    assert first == second
    assert first["runStatus"] == "completed"
    assert first["stopReason"] == "fixed_event_budget"
    assert first["censored"] is (not first["conjunctiveCompletionByBudget"])
    assert isinstance(first["terminalS01GlobalSuccess"], bool)
    assert isinstance(first["terminalS02GrammarAccepted"], bool)
    assert first["metricObservationCount"] == 3
    assert first["invariantSuccess"] and first["permissionAuditSuccess"]
    assert json.loads(first["initialGridRowsJson"])


def test_policy_pairs_share_scenario_but_splits_and_runs_are_disjoint() -> None:
    exploration = scenario_identity(
        "exploratory", "tissue_single_hole", "random", 11, "exploration_v1"
    )
    boundary = scenario_identity(
        "exploratory", "tissue_single_hole", "random", 11, "boundary_seeking_v1"
    )
    confirmation = scenario_identity(
        "confirmation_holdout", "tissue_single_hole", "random", 11, "exploration_v1"
    )
    assert exploration["scenarioId"] == boundary["scenarioId"]
    assert exploration["pairingBlockId"] == boundary["pairingBlockId"]
    assert exploration["seedHex"] == boundary["seedHex"]
    assert exploration["runId"] != boundary["runId"]
    assert confirmation["scenarioId"] != exploration["scenarioId"]
    assert confirmation["seedHex"] != exploration["seedHex"]
