from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest

from src.portfolio_preregistration.core import (
    TASK_IDS,
    build_budget_slots,
    build_candidate_eligibility_registry,
    build_portfolio_seed_registry,
    candidate_commitment,
    checked_protocol,
    plan_digest,
    read_jsonl,
    validate_portfolio_registry,
    verify_frozen_inputs,
)


@pytest.fixture(scope="module")
def design():
    protocol = checked_protocol()
    sources = read_jsonl(protocol["frozenInputs"]["s07rCandidateRegistry"]["path"])
    candidates = build_candidate_eligibility_registry(protocol, source_order=sources)
    member_sets, configurations = build_portfolio_seed_registry(protocol, candidates)
    slots = build_budget_slots(protocol, configurations)
    return protocol, sources, candidates, member_sets, configurations, slots


def test_protocol_is_design_only_and_inputs_are_frozen(design):
    protocol = design[0]
    assert protocol["designOnly"] is True
    assert protocol["authorizationBoundary"]["episodeEvaluations"] == 0
    assert (
        protocol["authorizationBoundary"]["substantivePortfolioEvaluationsAuthorized"]
        is False
    )
    assert len(verify_frozen_inputs(protocol)) == len(protocol["frozenInputs"])


def test_candidate_population_recompiles_and_is_outcome_free(design):
    candidates = design[2]
    assert len(candidates) == 512
    assert Counter(row["taskId"] for row in candidates) == Counter(
        {task: 64 for task in TASK_IDS}
    )
    assert len(candidate_commitment(candidates)) == 64
    forbidden = {"armId", "armMembership", "selected", "nativeOutcome", "nativeCosts"}
    assert all(not (forbidden & set(row)) for row in candidates)
    assert all(row["eligible"] and not row["promotionEvidence"] for row in candidates)


def test_seed_registry_exact_counts_and_legality(design):
    protocol, _, candidates, member_sets, configurations, _ = design
    audit = validate_portfolio_registry(
        protocol, candidates, member_sets, configurations
    )
    assert audit["success"] is True
    assert len(member_sets) == 192
    assert len(configurations) == 1216
    for task in TASK_IDS:
        counts = Counter(row["mode"] for row in configurations if row["taskId"] == task)
        assert counts == Counter(
            {
                "single_policy": 64,
                "fixed_balanced_identity": 24,
                "random_static_identity": 24,
                "random_dynamic_opportunity": 24,
                "environment_conditioned": 16,
            }
        )


def test_budget_and_smoke_are_complete_and_unevaluated(design):
    slots = design[5]
    assert len(slots) == 11008
    assert Counter(row["stage"] for row in slots) == Counter(
        {"initial": 4864, "adaptive": 6144}
    )
    assert sum(row["smoke"] for row in slots) == 40
    assert all(row["split"] == "train" for row in slots)
    assert all(not row["evaluationAuthorized"] for row in slots)
    assert all(not row["outcomeMaterialized"] for row in slots)


def test_plan_is_independent_of_candidate_order(design):
    protocol, sources, candidates, member_sets, configurations, slots = design
    baseline = plan_digest(candidates, member_sets, configurations, slots)
    reversed_candidates = build_candidate_eligibility_registry(
        protocol, source_order=list(reversed(sources))
    )
    reversed_sets, reversed_configs = build_portfolio_seed_registry(
        protocol, reversed_candidates
    )
    reversed_slots = build_budget_slots(protocol, reversed_configs)
    assert (
        plan_digest(
            reversed_candidates, reversed_sets, reversed_configs, reversed_slots
        )
        == baseline
    )


def test_duplicate_member_configuration_is_rejected(design):
    protocol, _, candidates, member_sets, configurations, _ = design
    invalid = deepcopy(configurations)
    target = next(row for row in invalid if row["portfolioSize"] >= 2)
    target["members"][1] = deepcopy(target["members"][0])
    with pytest.raises(RuntimeError, match="duplicate member"):
        validate_portfolio_registry(protocol, candidates, member_sets, invalid)


def test_selector_registry_is_narrow_and_no_universal_score(design):
    protocol = design[0]
    assert protocol["selectorSignals"]["globalRules"]["outcomeFieldsForbidden"]
    assert protocol["selectorSignals"]["globalRules"][
        "s07ArmOrSelectionFieldsForbidden"
    ]
    assert protocol["estimands"]["universalNormalizedOutcome"] == "forbidden"
    assert (
        protocol["portfolioCosts"]["aggregation"]
        == "separate_vectors_never_universal_total"
    )


def test_execution_prerequisite_is_fail_closed(design):
    protocol = design[0]
    assert protocol["executionPrerequisite"]["currentState"] == "blocked"
    assert protocol["selectionBoundaries"]["confirmation"]["S08Budget"] == 0
    assert (
        protocol["promotionAndS09Gate"]["trainingShortlist"][
            "allocationArmMembershipTieBreak"
        ]
        == "forbidden"
    )
