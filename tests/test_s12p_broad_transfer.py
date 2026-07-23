from __future__ import annotations

from copy import deepcopy

from scripts.preregister_broad_transfer_s12p import (
    TASKS,
    build_adaptation_variants,
    build_logical_roster,
    build_scenario_population,
    canonical_sha256,
    roster_digest,
)


PANELS = [
    {
        "panelId": f"panel_{ordinal}",
        "targetId": "target",
        "fixtureId": "fixture",
        "challengeId": "challenge",
        "endpointMode": "diagnostic_only",
    }
    for ordinal in range(8)
]


def _candidate(
    configuration_id: str,
    lineage_id: str,
    *,
    mode: str,
    size: int,
) -> dict[str, object]:
    definition: dict[str, object] = {
        "configurationId": configuration_id,
        "taskId": TASKS[0],
        "mode": mode,
        "portfolioSize": size,
        "members": [{"policySha256": f"member-{i}"} for i in range(size)],
        "assignmentRotation": 0,
        "selector": None,
    }
    if mode == "environment_conditioned":
        definition["selector"] = {
            "profile": "bounded_memoryless_two_branch_v1",
            "operator": "eq",
            "signal": "last_action.rejected",
            "threshold": True,
            "trueBranch": "next_compatible_member",
            "falseBranch": "current_or_first_compatible_member",
        }
    return {
        "configurationId": configuration_id,
        "lineageId": lineage_id,
        "candidateRole": "s08m_parent",
        "structuralConfiguration": definition,
    }


def _population() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates = []
    lineages = []
    for ordinal in range(7):
        lineage_id = f"lineage-{ordinal}"
        parent_id = f"parent-{ordinal}"
        compressed_id = f"compressed-{ordinal}"
        parent_mode = (
            "fixed_balanced_identity"
            if ordinal in (0, 1)
            else "environment_conditioned"
        )
        candidates.append(
            _candidate(
                parent_id,
                lineage_id,
                mode=parent_mode,
                size=3 if parent_mode == "fixed_balanced_identity" else 2,
            )
        )
        compressed_mode = (
            "fixed_balanced_identity"
            if ordinal in (0, 1, 2)
            else "environment_conditioned"
        )
        candidates.append(
            _candidate(
                compressed_id,
                lineage_id,
                mode=compressed_mode,
                size=2,
            )
        )
        lineages.append(
            {
                "lineageId": lineage_id,
                "parentConfigurationId": parent_id,
                "compressedConfigurationId": compressed_id,
                "componentSingleConfigurationIds": [
                    f"single-{ordinal}-{member}"
                    for member in range(2 if ordinal == 0 else 3)
                ],
                "matchedRandomConfigurationId": f"random-{ordinal}",
            }
        )
    return candidates, lineages


def test_adaptation_grid_is_bounded_and_outcome_independent() -> None:
    candidates, _ = _population()
    variants = build_adaptation_variants(candidates)
    assert len(variants) == 30
    assert len({row["adaptationVariantId"] for row in variants}) == 30
    assert all(row["outcomeFieldsLoaded"] is False for row in variants)
    assert all(row["memberSetChanged"] is False for row in variants)
    assert {row["edit"]["kind"] for row in variants} == {
        "assignment_rotation",
        "selector_original",
        "selector_branch_swap",
    }


def test_scenario_partitions_are_disjoint_and_exact() -> None:
    scenarios = build_scenario_population(PANELS)
    assert len(scenarios) == 1280
    assert len({row["scenarioFamilyId"] for row in scenarios}) == 1280
    development = {
        row["scenarioFamilyId"]
        for row in scenarios
        if row["partition"] == "adaptation_development"
    }
    evaluation = {
        row["scenarioFamilyId"]
        for row in scenarios
        if row["partition"] == "transfer_evaluation"
    }
    assert len(development) == 256
    assert len(evaluation) == 1024
    assert not development & evaluation
    assert all(row["outcomeMaterialized"] is False for row in scenarios)


def test_roster_accounting_and_worker_order_independence() -> None:
    candidates, lineages = _population()
    variants = build_adaptation_variants(candidates)
    scenarios = build_scenario_population(PANELS)
    roster = build_logical_roster(candidates, lineages, variants, scenarios)
    reverse = build_logical_roster(
        list(reversed(candidates)),
        list(reversed(lineages)),
        list(reversed(variants)),
        list(reversed(scenarios)),
    )
    assert len(roster) == 65024
    assert len({row["logicalReservationId"] for row in roster}) == 65024
    assert roster_digest(roster) == roster_digest(reverse)
    assert all(row["outcomeMaterialized"] is False for row in roster)
    assert all(row["protectedOutcomeAccess"] is False for row in roster)
    assert all(row["s10OrS11SignalUsed"] is False for row in roster)


def test_canonical_hash_rejects_nonfinite_and_is_order_invariant() -> None:
    left = {"a": 1, "b": [2, 3]}
    right = {"b": [2, 3], "a": 1}
    assert canonical_sha256("test", left) == canonical_sha256("test", right)
    forged = deepcopy(left)
    forged["b"] = [2, 4]
    assert canonical_sha256("test", left) != canonical_sha256("test", forged)
