from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from src.morph2d.environments import evaluate_conjunctive, load_environment_catalog
from src.morph2d.grammar import load_grammar_catalog
from src.morph2d.movements import (
    ENABLED_KINDS,
    initial_movement_state,
    make_proposal,
    parse_movement_state,
    resolve_batch,
    state_tokens,
    validate_proposal,
)
from src.morph2d.policies import (
    FEATURES_BY_STRATEGY,
    OBSERVATION_LEDGER_FIELDS,
    PolicyMemory,
    PolicyValidationError,
    build_policy_observation,
    canonical_decision_bytes,
    canonical_observation_bytes,
    canonical_policy_event_bytes,
    compile_relation_profile,
    decide_policy,
    decision_source_forbidden_accesses,
    execute_policy_activation,
    load_policy_catalog,
    materialize_decision,
    parse_policy_catalog,
    policy_complexity,
    policy_to_dict,
    relation_profile_to_dict,
    update_policy_memory,
    validate_policy_memory,
    validate_policy_payload,
)
from src.morph2d.targets import load_target_catalog


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "configs/morphologies/policy_catalog.yaml"
ENVIRONMENTS = ROOT / "configs/morphologies/environment_catalog.yaml"
GRAMMARS = ROOT / "configs/morphologies/grammar_catalog.yaml"
TARGETS = ROOT / "configs/morphologies/target_catalog.yaml"


@pytest.fixture(scope="module")
def context():
    raw = yaml.safe_load(POLICIES.read_text(encoding="utf-8"))
    metadata, policies = load_policy_catalog(POLICIES)
    _, environments = load_environment_catalog(ENVIRONMENTS)
    _, grammars = load_grammar_catalog(GRAMMARS)
    _, targets = load_target_catalog(TARGETS)
    return {
        "raw": raw,
        "metadata": metadata,
        "policies": {item.policy_id: item for item in policies},
        "environments": {item.environment_id: item for item in environments},
        "grammars": {item.grammar_id: item for item in grammars},
        "targets": {item.target_id: item for item in targets},
    }


def _actor_at(state, site_id: str) -> str:
    return state.occupant_map[site_id].occupant_id


def _perturbed_stripes(context):
    environment = context["environments"]["square_bounded_occupied_stripes"]
    state = initial_movement_state(environment)
    proposal = make_proposal(state, "adjacent_swap", ("r2_c0", "r3_c0"))
    result = resolve_batch(environment, state, (proposal,), batch_nonce="s05-damage")
    return environment, parse_movement_state(result["postState"])


def _zero_lagged(environment) -> dict[str, int]:
    return {site.site_id: 0 for site in environment.sites if site.role != "obstacle"}


def _field_by_column(environment) -> dict[str, int]:
    return {
        site.site_id: int(site.site_id.split("_c")[1])
        for site in environment.sites
        if site.role != "obstacle"
    }


def _relation_fixture(context):
    environment, state = _perturbed_stripes(context)
    profile = compile_relation_profile(context["grammars"]["stripes_axis_relations_v1"])
    actor_id = _actor_at(state, "r2_c0")
    return environment, state, profile, actor_id


def test_catalog_defines_exactly_six_planned_strategies_and_round_trips(
    context,
) -> None:
    policies = tuple(context["policies"].values())
    assert len(policies) == 6
    assert {item.strategy for item in policies} == set(FEATURES_BY_STRATEGY)
    assert all(
        set(item.observation_features) == FEATURES_BY_STRATEGY[item.strategy]
        for item in policies
    )
    reloaded = yaml.safe_load(
        yaml.safe_dump(context["raw"], sort_keys=False, allow_unicode=False)
    )
    _, parsed = parse_policy_catalog(reloaded)
    assert [policy_to_dict(item) for item in parsed] == [
        policy_to_dict(item) for item in policies
    ]


def test_catalog_rejects_unpriced_or_strategy_incompatible_features(context) -> None:
    raw = json.loads(json.dumps(context["raw"]))
    raw["policies"][0]["observationFeatures"].append("site_roles")
    with pytest.raises(PolicyValidationError, match="feature set mismatch"):
        parse_policy_catalog(raw)


def test_all_s02_grammars_compile_to_bounded_actor_local_profiles(context) -> None:
    projected = 0
    excluded = 0
    for grammar in context["grammars"].values():
        profile = compile_relation_profile(grammar)
        record = relation_profile_to_dict(profile)
        assert record["grammarId"] == grammar.grammar_id
        assert record["targetId"] == grammar.target_id
        assert "whole-grid" in record["claimBoundary"]
        assert not (
            set(profile.projected_constraint_ids) & set(profile.excluded_constraint_ids)
        )
        constraint_ids = {item.constraint_id for item in grammar.constraints}
        assert (
            set(profile.projected_constraint_ids) | set(profile.excluded_constraint_ids)
            == constraint_ids
        )
        projected += len(profile.projected_constraint_ids)
        excluded += len(profile.excluded_constraint_ids)
    assert projected > 0
    assert excluded > 0


def test_policy_payload_permission_gate_rejects_engine_only_fields() -> None:
    for key in (
        "actorId",
        "siteId",
        "route",
        "expectedOccupants",
        "stateSha256",
        "boundaryTags",
        "fixedBoundaryNeighbors",
        "conflictPriority",
        "analysisLabel",
        "s01GlobalCompletionAudit",
    ):
        with pytest.raises(PolicyValidationError, match="forbidden"):
            validate_policy_payload({"candidates": [], key: "leak"})
    assert decision_source_forbidden_accesses() == []


def test_greedy_observation_is_priced_local_projection_and_deterministic(
    context,
) -> None:
    environment, state, profile, actor_id = _relation_fixture(context)
    definition = context["policies"]["greedy_neighbor_satisfaction_v1"]
    first = build_policy_observation(
        environment, state, actor_id, definition, relation_profile=profile
    )
    second = build_policy_observation(
        environment, state, actor_id, definition, relation_profile=profile
    )
    assert canonical_observation_bytes(
        first.observation
    ) == canonical_observation_bytes(second.observation)
    decision = decide_policy(definition, first.observation.payload)
    assert canonical_decision_bytes(decision) == canonical_decision_bytes(
        decide_policy(definition, second.observation.payload)
    )
    assert decision.action == "proposal"
    assert decision.decision_score > 0
    payload_json = json.dumps(first.observation.payload, sort_keys=True)
    assert all(site.site_id not in payload_json for site in environment.sites)
    assert "grammar" not in payload_json.lower()
    assert "global" not in payload_json.lower()
    assert set(first.observation.budget) == set(OBSERVATION_LEDGER_FIELDS)
    assert (
        first.observation.budget["communicatedBitsUpperBound"]
        <= definition.information_budget_max_bits
    )


def test_boundary_policy_prices_only_natural_exterior_and_periodic_has_no_signal(
    context,
) -> None:
    definition = context["policies"]["boundary_seeking_v1"]
    bounded = context["environments"]["square_bounded_occupied_stripes"]
    bounded_state = initial_movement_state(bounded)
    build = build_policy_observation(
        bounded,
        bounded_state,
        _actor_at(bounded_state, "r1_c1"),
        definition,
        boundary_direction="seek",
        boundary_tokens=("A",),
    )
    decision = decide_policy(definition, build.observation.payload)
    assert decision.action == "proposal"
    assert decision.decision_score == 1
    assert build.observation.budget["boundarySignalReads"] == 1 + len(
        build.candidate_map
    )
    payload = json.dumps(build.observation.payload).lower()
    assert "boundarytags" not in payload
    assert "obstacle" not in payload
    assert "fixed" not in payload

    periodic = context["environments"]["square_periodic_vacancy"]
    periodic_state = initial_movement_state(periodic)
    periodic_build = build_policy_observation(
        periodic,
        periodic_state,
        _actor_at(periodic_state, "r2_c2"),
        definition,
        boundary_direction="seek",
        boundary_tokens=(periodic_state.occupant_map["r2_c2"].token,),
    )
    assert (
        decide_policy(definition, periodic_build.observation.payload).action == "noop"
    )
    assert all(
        item["naturalBoundaryDelta"] == 0
        for item in periodic_build.observation.payload["candidates"]
    )


def test_gradient_policy_uses_supplied_uint8_projection_and_requires_coverage(
    context,
) -> None:
    definition = context["policies"]["gradient_following_v1"]
    environment = context["environments"]["square_periodic_vacancy"]
    state = initial_movement_state(environment)
    actor_id = _actor_at(state, "r2_c2")
    field = _field_by_column(environment)
    build = build_policy_observation(
        environment,
        state,
        actor_id,
        definition,
        gradient_levels=field,
        gradient_direction="up",
    )
    decision = decide_policy(definition, build.observation.payload)
    assert decision.action == "proposal"
    assert decision.decision_score == 1
    assert build.observation.budget["gradientSignalReads"] == 1 + len(
        build.candidate_map
    )
    assert all(
        "gradientDelta" in item for item in build.observation.payload["candidates"]
    )
    with pytest.raises(PolicyValidationError, match="cover"):
        build_policy_observation(
            environment,
            state,
            actor_id,
            definition,
            gradient_levels={"r2_c2": 2},
            gradient_direction="up",
        )


def test_exploration_is_counter_addressed_replayable_and_covers_all_s04_kinds(
    context,
) -> None:
    definition = context["policies"]["exploration_v1"]
    observed_kinds = set()
    for environment_id, actor_site in (
        ("hexagonal_bounded_occupied", "q0_r0"),
        ("square_periodic_vacancy", "r0_c1"),
    ):
        environment = context["environments"][environment_id]
        state = initial_movement_state(environment)
        actor_id = _actor_at(state, actor_site)
        first = build_policy_observation(
            environment,
            state,
            actor_id,
            definition,
            decision_key="counter-replay-v1",
            activation_index=11,
        )
        second = build_policy_observation(
            environment,
            state,
            actor_id,
            definition,
            decision_key="counter-replay-v1",
            activation_index=11,
        )
        assert first.observation.payload == second.observation.payload
        assert first.observation.budget["counterRandomDraws"] == 1
        observed_kinds |= {
            affordance.proposal.kind for affordance in first.candidate_map.values()
        }
        for index in range(96):
            build = build_policy_observation(
                environment,
                state,
                actor_id,
                definition,
                decision_key=f"movement-coverage-{index}",
                activation_index=index,
            )
            decision = decide_policy(definition, build.observation.payload)
            proposal = materialize_decision(build, decision)
            assert proposal is not None
            assert validate_proposal(environment, state, proposal).valid
            single = resolve_batch(
                environment,
                state,
                (proposal,),
                batch_nonce=f"cost-reconciliation-{environment_id}-{index}",
            )
            selected = build.candidate_map[decision.selected_candidate_key]
            assert (
                selected.movement_cost == single["costLedger"]["totalGraphDisplacement"]
            )
            observed_kinds.add(proposal.kind)
    assert observed_kinds == ENABLED_KINDS


def test_memory_policy_uses_ten_identity_owned_bits_and_recovers_only_below_reference(
    context,
) -> None:
    environment, state, profile, actor_id = _relation_fixture(context)
    definition = context["policies"]["memory_based_recovery_v1"]
    memory = PolicyMemory(best_local_utility=12, frustration=1)
    build = build_policy_observation(
        environment,
        state,
        actor_id,
        definition,
        relation_profile=profile,
        memory=memory,
    )
    decision = decide_policy(definition, build.observation.payload)
    assert build.observation.payload["ownMemory"] == {
        "bestLocalRelationUtility": 12,
        "frustration": 1,
    }
    assert build.observation.budget["memoryReads"] == 2
    assert build.observation.budget["persistentMemoryBits"] == 10
    assert decision.action == "proposal"
    accepted = update_policy_memory(memory, build.observation, decision, "accepted")
    assert accepted.frustration == 0
    met_memory = PolicyMemory(
        best_local_utility=build.observation.payload["currentLocalRelationUtility"],
        frustration=0,
    )
    met = build_policy_observation(
        environment,
        state,
        actor_id,
        definition,
        relation_profile=profile,
        memory=met_memory,
    )
    assert decide_policy(definition, met.observation.payload).reason == (
        "remembered_reference_met"
    )
    with pytest.raises(PolicyValidationError):
        validate_policy_memory(PolicyMemory(128, 0))
    with pytest.raises(PolicyValidationError):
        validate_policy_memory(PolicyMemory(0, 4))


def test_conflict_avoidance_uses_only_saturated_lagged_counts_and_s04_cost(
    context,
) -> None:
    environment, state, profile, actor_id = _relation_fixture(context)
    definition = context["policies"]["conflict_avoidance_v1"]
    zero = _zero_lagged(environment)
    baseline = build_policy_observation(
        environment,
        state,
        actor_id,
        definition,
        relation_profile=profile,
        lagged_conflicts=zero,
    )
    baseline_decision = decide_policy(definition, baseline.observation.payload)
    assert baseline_decision.action == "proposal"
    selected = baseline.candidate_map[baseline_decision.selected_candidate_key]
    penalized = dict(zero)
    penalized[selected.target_site] = 3
    changed = build_policy_observation(
        environment,
        state,
        actor_id,
        definition,
        relation_profile=profile,
        lagged_conflicts=penalized,
    )
    changed_decision = decide_policy(definition, changed.observation.payload)
    assert changed.observation.budget["laggedConflictReads"] == len(
        changed.candidate_map
    )
    assert all(
        {"candidateKey", "localRelationDelta", "laggedConflictCount", "movementCost"}
        == set(item)
        for item in changed.observation.payload["candidates"]
    )
    assert changed_decision.action == "proposal"
    assert (
        changed_decision.selected_candidate_key
        != baseline_decision.selected_candidate_key
    )
    with pytest.raises(PolicyValidationError, match="saturated"):
        invalid = dict(zero)
        invalid[selected.target_site] = 4
        build_policy_observation(
            environment,
            state,
            actor_id,
            definition,
            relation_profile=profile,
            lagged_conflicts=invalid,
        )


def test_every_policy_materializes_only_legal_s04_intents_and_replays(context) -> None:
    environment, state, profile, actor_id = _relation_fixture(context)
    cases = {
        "greedy_neighbor_satisfaction_v1": {"relation_profile": profile},
        "memory_based_recovery_v1": {
            "relation_profile": profile,
            "memory": PolicyMemory(12, 1),
        },
        "conflict_avoidance_v1": {
            "relation_profile": profile,
            "lagged_conflicts": _zero_lagged(environment),
        },
    }
    for policy_id, kwargs in cases.items():
        definition = context["policies"][policy_id]
        first = execute_policy_activation(
            environment,
            state,
            actor_id,
            definition,
            batch_nonce=f"replay:{policy_id}",
            **kwargs,
        )
        second = execute_policy_activation(
            environment,
            state,
            actor_id,
            definition,
            batch_nonce=f"replay:{policy_id}",
            **kwargs,
        )
        assert canonical_policy_event_bytes(first) == canonical_policy_event_bytes(
            second
        )
        assert first["movementBatchResult"]["invariants"]["success"]
        assert first["movementBatchResult"]["costLedger"]["boundarySignalReads"] == 0
        assert first["materializationAudit"]["policySawProposalEnvelope"] is False
        assert first["materializationAudit"]["authenticationAddedAfterPolicyDecision"]


def test_s02_local_feedback_and_s01_global_evaluation_are_separate(context) -> None:
    environment, state, profile, actor_id = _relation_fixture(context)
    definition = context["policies"]["greedy_neighbor_satisfaction_v1"]
    event = execute_policy_activation(
        environment,
        state,
        actor_id,
        definition,
        relation_profile=profile,
        batch_nonce="evaluation-separation-v1",
    )
    grammar = context["grammars"][profile.grammar_id]
    target = context["targets"][grammar.target_id]
    post_state = parse_movement_state(event["movementBatchResult"]["postState"])
    post_environment = replace(
        environment,
        initial_state={**environment.initial_state, **state_tokens(post_state)},
    )
    offline = evaluate_conjunctive(post_environment, target, grammar)
    assert event["s02LocalFeedback"]["profile"] == (
        "s02_actor_local_contact_projection_v1"
    )
    assert event["s02LocalFeedback"]["wholeGridGrammarScoreDisclosed"] is False
    assert event["s01GlobalEvaluation"] == {
        "availability": "offline_only_not_computed_in_policy_activation",
        "disclosedToPolicy": False,
    }
    assert "globalAudit" in offline
    assert "localGrammar" in offline
    assert "completion" in offline
    assert "global" not in json.dumps(event["observation"]["payload"]).lower()


def test_information_and_complexity_budgets_are_explicit_for_every_policy(
    context,
) -> None:
    complexities = {
        policy_id: policy_complexity(definition)
        for policy_id, definition in context["policies"].items()
    }
    assert complexities["memory_based_recovery_v1"]["persistentMemoryBits"] == 10
    assert complexities["memory_based_recovery_v1"]["complexityScore"] == 11
    assert complexities["exploration_v1"]["targetSpecificity"] == "none"
    assert all(item["complexityScore"] > 0 for item in complexities.values())
    assert all(item["informationBudgetMaxBits"] > 0 for item in complexities.values())
