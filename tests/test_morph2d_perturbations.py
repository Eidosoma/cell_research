from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from morph2d.baseline import load_baseline_assets, state_grid
from morph2d.engine import EpisodeValidationError, run_cpu_episode
from morph2d.movements import MovementProposal, initial_movement_state
from morph2d.perturbations import (
    FEASIBILITY_ONLY_LESIONS,
    SIMULATED_LESIONS,
    PerturbationProposalGate,
    apply_lesion,
    count_changing_feasibility_records,
    run_perturbation_once,
    scenario_identity,
    simulated_target_lesion_pairs,
)
from morph2d.targets import evaluate_success


ROOT = Path(__file__).resolve().parents[1]
CATALOG = yaml.safe_load(
    (ROOT / "configs/morphologies/perturbation_catalog.yaml").read_text()
)


def _spec(*, arm: str, event_budget: int = 4) -> dict:
    return {
        "catalog": CATALOG,
        "phase": "unit",
        "split": "unit",
        "targetId": "separated_paired_regions",
        "grammarId": "separated_region_relations_v1",
        "policyId": "memory_based_recovery_v1",
        "lesionId": "compact_wound",
        "severity": "mild",
        "arm": arm,
        "replicate": 0,
        "eventBudget": event_budget,
        "retainTrace": False,
    }


def test_frozen_matrix_expands_to_declared_counts() -> None:
    pairs = simulated_target_lesion_pairs(CATALOG)
    assert len(pairs) == 33
    assert {item["lesionId"] for item in pairs} == SIMULATED_LESIONS
    arms = len(CATALOG["simulation"]["arms"])
    severities = len(CATALOG["severities"])
    assert len(pairs) * arms * severities == 132
    assert 132 * CATALOG["simulation"]["exploratoryReplicatesPerConditionArm"] == 33000


def test_count_changing_lesions_are_explicitly_infeasible() -> None:
    records = count_changing_feasibility_records(CATALOG)
    assert len(records) == 7 * 2 * 2
    assert {item["lesionId"] for item in records} == FEASIBILITY_ONLY_LESIONS
    assert all(not item["targetFeasible"] for item in records)
    assert all(not item["lesionArmLaunched"] for item in records)
    assert all(
        item["classification"] == "infeasible_no_adjusted_contract" for item in records
    )


@pytest.mark.parametrize("severity", ["mild", "severe"])
def test_every_simulated_lesion_preserves_contract_and_breaks_target(
    severity: str,
) -> None:
    _context, targets, _grammars, environments = load_baseline_assets()
    for item in simulated_target_lesion_pairs(CATALOG):
        environment = environments[item["targetId"]]
        exact = initial_movement_state(environment)
        application = apply_lesion(
            environment,
            targets[item["targetId"]],
            exact,
            item["lesionId"],
            severity,
            f"test:{item['targetId']}:{item['lesionId']}:{severity}",
            CATALOG,
        )
        assert application.target_feasible
        assert not evaluate_success(
            state_grid(environment, application.state), targets[item["targetId"]]
        )["success"]
        assert {value.occupant_id for _, value in exact.occupancy} == {
            value.occupant_id for _, value in application.state.occupancy
        }
        assert Counter(value.token for _, value in exact.occupancy) == Counter(
            value.token for _, value in application.state.occupancy
        )
        assert Counter(value.kind for _, value in exact.occupancy) == Counter(
            value.kind for _, value in application.state.occupancy
        )
        assert len(application.mask["lesionMaskSha256"]) == 64


def test_lesion_masks_are_deterministic_and_severity_changes_them() -> None:
    _context, targets, _grammars, environments = load_baseline_assets()
    environment = environments["bilateral_lobes_with_midline"]
    exact = initial_movement_state(environment)
    mild_first = apply_lesion(
        environment,
        targets["bilateral_lobes_with_midline"],
        exact,
        "compact_wound",
        "mild",
        "mask-address",
        CATALOG,
    )
    mild_second = apply_lesion(
        environment,
        targets["bilateral_lobes_with_midline"],
        exact,
        "compact_wound",
        "mild",
        "mask-address",
        CATALOG,
    )
    severe = apply_lesion(
        environment,
        targets["bilateral_lobes_with_midline"],
        exact,
        "compact_wound",
        "severe",
        "mask-address",
        CATALOG,
    )
    assert mild_first == mild_second
    assert mild_first.mask["lesionMaskSha256"] != severe.mask["lesionMaskSha256"]
    assert (
        severe.external_ledger["externalDisplacedEntities"]
        > mild_first.external_ledger["externalDisplacedEntities"]
    )


def test_pair_identity_shares_scenario_but_not_run_across_arms() -> None:
    lesion = scenario_identity("exploratory", "target", "wound", "mild", 3, "lesion")
    control = scenario_identity(
        "exploratory", "target", "wound", "mild", 3, "no_damage"
    )
    assert lesion["scenarioId"] == control["scenarioId"]
    assert lesion["pairingBlockId"] == control["pairingBlockId"]
    assert lesion["seedHex"] == control["seedHex"]
    assert lesion["runId"] != control["runId"]


def test_barrier_and_immobility_gate_only_retain_original_proposals() -> None:
    _context, targets, _grammars, environments = load_baseline_assets()
    environment = environments["stripes_alternating_three_band"]
    exact = initial_movement_state(environment)
    barrier = apply_lesion(
        environment,
        targets["stripes_alternating_three_band"],
        exact,
        "temporary_barrier",
        "mild",
        "gate",
        CATALOG,
    )
    edge = sorted(barrier.barrier_edges)[0]
    proposal = MovementProposal(
        proposal_id="proposal:test",
        kind="adjacent_swap",
        actor_id=barrier.state.occupant_map[edge[0]].occupant_id,
        source_site=edge[0],
        target_site=edge[1],
        route=edge,
        rotation_direction=0,
        observed_state_sha256="state",
        expected_occupants=tuple(
            (site, barrier.state.occupant_map[site].occupant_id) for site in edge
        ),
    )
    gate = PerturbationProposalGate(barrier)
    assert gate(0, environment, barrier.state, (proposal,)) == ()
    assert gate.ledger()["suppressedProposals"] == 1
    assert gate(
        barrier.gate_duration_transitions, environment, barrier.state, (proposal,)
    ) == (proposal,)


def test_engine_rejects_proposal_gate_forgery() -> None:
    context, _targets, _grammars, _environments = load_baseline_assets()
    definition = context.episodes[0]

    def forge(_index, _environment, _state, proposals):
        if not proposals:
            return proposals
        item = proposals[0]
        return (MovementProposal(**{**item.__dict__, "proposal_id": "forged"}),)

    with pytest.raises(EpisodeValidationError, match="only retain"):
        run_cpu_episode(context, definition, proposal_gate=forge)


def test_paired_smoke_run_separates_repair_and_maintenance() -> None:
    lesion, _, mask = run_perturbation_once(_spec(arm="lesion"))
    control, _, control_mask = run_perturbation_once(_spec(arm="no_damage"))
    assert lesion["pairingBlockId"] == control["pairingBlockId"]
    assert lesion["preDamageStateSha256"] == control["preDamageStateSha256"]
    assert lesion["lesionMaskSha256"] == control["lesionMaskSha256"]
    assert mask["lesionStateSha256"] == control_mask["lesionStateSha256"]
    assert lesion["repairEligible"]
    assert not control["repairEligible"]
    assert not lesion["immediatePostDamageConjunctive"]
    assert control["immediatePostDamageConjunctive"]
    assert not control["censored"]
    assert lesion["invariantSuccess"] and control["invariantSuccess"]
    assert lesion["permissionAuditSuccess"] and control["permissionAuditSuccess"]


def test_perturbation_run_replays_exactly() -> None:
    first, _, first_mask = run_perturbation_once(_spec(arm="lesion", event_budget=8))
    second, _, second_mask = run_perturbation_once(_spec(arm="lesion", event_budget=8))
    assert first["episodeCanonicalBytesSha256"] == second["episodeCanonicalBytesSha256"]
    assert first["metricSummarySha256"] == second["metricSummarySha256"]
    assert first["interventionSummarySha256"] == second["interventionSummarySha256"]
    assert first_mask == second_mask


def test_gate_none_preserves_canonical_episode_bytes() -> None:
    context, _targets, _grammars, _environments = load_baseline_assets()
    definition = context.episodes[0]
    ordinary = run_cpu_episode(context, definition)
    gated = run_cpu_episode(
        context,
        definition,
        proposal_gate=lambda _index, _environment, _state, proposals: proposals,
    )
    assert ordinary == gated
