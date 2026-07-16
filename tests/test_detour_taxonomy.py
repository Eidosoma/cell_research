import json
from pathlib import Path

from src.detours.taxonomy import (
    CATEGORY_ORDER,
    UNSUPPORTED,
    EvidenceFlags,
    anthropomorphic_assertion_hits,
    strongest_supported_category,
    supported_categories,
    validate_claim_assignment,
)


def test_category_order_matches_frozen_contract() -> None:
    contract = json.loads(
        (Path(__file__).parents[1] / "analysis" / "e03_s14_detour_taxonomy_contract.json").read_text()
    )
    assert [row["category"] for row in contract["categories"]] == list(CATEGORY_ORDER)
    assert len(contract["majorClaimRegistry"]) == 22


def test_empty_flags_are_explicitly_unsupported() -> None:
    assert strongest_supported_category(EvidenceFlags()) == UNSUPPORTED
    assert supported_categories(EvidenceFlags()) == ()


def test_local_category_requires_local_worsening() -> None:
    flags = EvidenceFlags(replayable_local_worsening=True)
    assert strongest_supported_category(flags) == "local_observable_backtracking"


def test_global_category_requires_metric_and_goal() -> None:
    assert (
        strongest_supported_category(EvidenceFlags(named_global_worsening=True))
        == UNSUPPORTED
    )
    flags = EvidenceFlags(
        named_global_worsening=True, explicit_goal_projection=True
    )
    assert strongest_supported_category(flags) == "global_regression"


def test_barrier_category_requires_all_isolation_gates() -> None:
    partial = EvidenceFlags(
        isolated_barrier_contrast=True, matched_pre_state=True
    )
    assert strongest_supported_category(partial) == UNSUPPORTED
    complete = EvidenceFlags(
        isolated_barrier_contrast=True,
        matched_pre_state=True,
        valid_stream_scope=True,
    )
    assert strongest_supported_category(complete) == "barrier_correlated_detour"


def test_necessary_category_requires_reachability_and_positive_minimax_excursion() -> None:
    assert (
        strongest_supported_category(
            EvidenceFlags(exact_minimum_excursion_positive=True)
        )
        == UNSUPPORTED
    )
    complete = EvidenceFlags(
        exact_goal_reachable=True, exact_minimum_excursion_positive=True
    )
    assert strongest_supported_category(complete) == "necessary_detour"


def full_adaptive_flags(**overrides: bool) -> EvidenceFlags:
    values = {
        "named_global_worsening": True,
        "explicit_goal_projection": True,
        "observed_successful_recovered_global_excursion": True,
        "intervention_relative_utility": True,
        "no_created_impossibility_as_benefit": True,
        "no_censoring_as_efficiency": True,
        "matched_null_exceedance": True,
        "same_support_metric_goal": True,
        "replay_and_provenance_pass": True,
    }
    values.update(overrides)
    return EvidenceFlags(**values)


def test_adaptive_category_requires_every_gate() -> None:
    assert strongest_supported_category(full_adaptive_flags()) == "adaptive_detour"
    for gate in [
        "named_global_worsening",
        "explicit_goal_projection",
        "observed_successful_recovered_global_excursion",
        "intervention_relative_utility",
        "no_created_impossibility_as_benefit",
        "no_censoring_as_efficiency",
        "matched_null_exceedance",
        "same_support_metric_goal",
        "replay_and_provenance_pass",
    ]:
        assert strongest_supported_category(full_adaptive_flags(**{gate: False})) != "adaptive_detour"


def test_strongest_category_wins_when_lower_flags_are_also_present() -> None:
    flags = EvidenceFlags(
        replayable_local_worsening=True,
        named_global_worsening=True,
        explicit_goal_projection=True,
        exact_goal_reachable=True,
        exact_minimum_excursion_positive=True,
    )
    assert supported_categories(flags) == (
        "local_observable_backtracking",
        "global_regression",
        "necessary_detour",
    )
    assert strongest_supported_category(flags) == "necessary_detour"


def test_assignment_validator_detects_upgrade() -> None:
    flags = EvidenceFlags(replayable_local_worsening=True)
    assignment = {
        "strongest_supported_category": "adaptive_detour",
        "supported_categories": ["local_observable_backtracking"],
    }
    errors = validate_claim_assignment(assignment, flags)
    assert errors


def test_language_scanner_rejects_mental_state_assertions() -> None:
    assert not anthropomorphic_assertion_hits(
        ["The exact graph requires a positive inversion excursion."]
    )
    hits = anthropomorphic_assertion_hits(
        ["The controller anticipates a later gain and wants the goal state."]
    )
    assert {row["pattern"] for row in hits} == {"anticipation", "wanting"}
