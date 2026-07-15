from __future__ import annotations

from causal_simulator.counterexample_search import (
    OBJECTIVE_IDS,
    Candidate,
    candidate_distance,
    confirmation_neighbor,
    derive_seed,
    evaluate_candidate,
    initial_candidate,
    mutate_candidate,
    paired_bootstrap_interval,
    wilson_interval,
)


def test_candidate_generation_and_mutation_preserve_exact_contract() -> None:
    parent = initial_candidate(20, "balanced_duplicate", 4, 0)
    parent.validate()
    children = [mutate_candidate(parent, 1, slot, 0) for slot in range(4)]
    assert len({child.candidate_id for child in children}) == 4
    for child in children:
        child.validate()
        assert sorted(child.occupancy) == list(range(20))
        assert len(child.fault_identities) == 4
        assert child.stratum_id == parent.stratum_id


def test_confirmation_neighbor_uses_new_map_and_preserves_marginals() -> None:
    source = initial_candidate(50, "unique", 2, 4)
    first = confirmation_neighbor(source, 7)
    replay = confirmation_neighbor(source, 7)
    assert first == replay
    assert first.candidate_id != source.candidate_id
    assert first.fault_identities != source.fault_identities
    assert sorted(first.occupancy) == sorted(source.occupancy)
    assert len(first.fault_identities) == source.fault_count
    assert candidate_distance(source, first) > 0


def test_distance_is_symmetric_and_stratum_separated() -> None:
    left = initial_candidate(20, "unique", 2, 1)
    right = initial_candidate(20, "unique", 2, 2)
    other = initial_candidate(20, "unique", 4, 1)
    assert candidate_distance(left, left) == 0
    assert candidate_distance(left, right) == candidate_distance(right, left)
    assert 0 < candidate_distance(left, right) <= 1
    assert candidate_distance(left, other) == 1


def test_five_frozen_signatures_replay_and_validate() -> None:
    candidate = initial_candidate(20, "unique", 2, 3)
    seed = derive_seed(123, "focused-test")
    first = evaluate_candidate(
        candidate.to_record(),
        stage="training",
        scheduler="uniform_random_activation",
        seed=seed,
        instance_id="focused-test-instance",
    )
    second = evaluate_candidate(
        candidate.to_record(),
        stage="training",
        scheduler="uniform_random_activation",
        seed=seed,
        instance_id="focused-test-instance",
    )
    assert len(first["runRows"]) == 5
    assert set(first["contrasts"]) == set(OBJECTIVE_IDS)
    assert all(row["contractValidationPass"] for row in first["runRows"])
    assert [row["deterministicResultSha256"] for row in first["runRows"]] == [
        row["deterministicResultSha256"] for row in second["runRows"]
    ]
    assert first["contrasts"] == second["contrasts"]


def test_frozen_interval_helpers_are_deterministic() -> None:
    values = [-0.1, 0.0, 0.1, 0.2] * 32
    first = paired_bootstrap_interval(
        values, objective_id="E06_active_advantage", panel="exact_instance", replicates=100
    )
    second = paired_bootstrap_interval(
        values, objective_id="E06_active_advantage", panel="exact_instance", replicates=100
    )
    assert first == second
    low, high = wilson_interval(16, 128)
    assert 0 < low < 0.125 < high < 1
