from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import pandas as pd
import yaml

from scripts.build_morph2d_s11 import eligible_contrast_count

from src.morph2d.chimeras import (
    CHIMERA_CATALOG_VERSION,
    _same_edge_count,
    _swap_same_edge_delta,
    assign_executable_groups,
    assign_ghost_labels,
    decluster_assignment,
    graph_label_metrics,
    group_assignment_audit,
    load_chimera_assets,
    relation_profile_overlap_audit,
    run_chimera_pair_once,
    transform_relation_profile,
)
from src.morph2d.engine import (
    EpisodeDefinition,
    EpisodeValidationError,
    run_cpu_episode,
)
from src.morph2d.movements import initial_movement_state
from src.morph2d.policies import compile_relation_profile


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def assets():
    return load_chimera_assets()


def test_catalog_freezes_exact_matrix_and_s10_constraint(assets) -> None:
    _context, target, grammar, environment, catalogs = assets
    catalog = catalogs["chimera"]
    assert catalog["schemaVersion"] == CHIMERA_CATALOG_VERSION
    assert catalog["researchStepId"] == "S11"
    assert catalog["acceptedS10Constraint"] == {
        "repairOutcome": "null",
        "semanticsNotWidened": [
            "repair",
            "removal",
            "insertion",
            "division",
            "token_conversion",
            "count_adjusted_targets",
            "vacancy_adjusted_targets",
        ],
        "s11IsNotRepair": True,
    }
    assert len(catalog["mixtureArchetypes"]) == 6
    assert 6 * 2 * 2 == catalog["simulation"]["baseConditionCount"] == 24
    assert catalog["simulation"]["exploratoryArmCount"] == 48
    assert catalog["simulation"]["exploratoryRunCount"] == 12_000
    assert target.target_id == "layers_three_ordered_tissues"
    assert grammar.grammar_id == "layers_ordered_contacts_v1"
    assert environment.geometry == "square"
    assert environment.boundary_mode == "bounded"
    assert len(environment.occupiable_sites) == 81
    assert len(environment.edges) == 144


def test_promotion_eligibility_counts_any_frozen_metric(assets) -> None:
    _context, _target, _grammar, _environment, catalogs = assets
    catalog = catalogs["chimera"]
    contrasts = pd.DataFrame(
        [
            {
                "peakDifference": 0.0,
                "positiveAreaDifference": 0.0,
                "terminalMismatchDifference": 0.031,
                "completionRiskDifference": 0.0,
                "recoveryBeyondSham": 0.0,
            },
            {
                "peakDifference": 0.0,
                "positiveAreaDifference": 0.0,
                "terminalMismatchDifference": 0.0,
                "completionRiskDifference": 0.0,
                "recoveryBeyondSham": 0.0,
            },
        ]
    )
    assert eligible_contrast_count(contrasts, catalog) == 1


def test_relation_profiles_are_aligned_overlapping_and_exactly_opposed(assets) -> None:
    _context, _target, grammar, _environment, _catalogs = assets
    base = compile_relation_profile(grammar)
    audit = {
        item["relationClass"]: item for item in relation_profile_overlap_audit(base)
    }
    assert audit["aligned"]["exactWeightCommonPairs"] == 7
    assert 0 < audit["overlapping"]["sameSignCommonPairs"] < 7
    assert audit["overlapping"]["oppositeSignCommonPairs"] > 0
    assert audit["contradictory"]["allBasePairsSignOpposed"]
    opposed = transform_relation_profile(base, "contradictory")
    assert {key: -value for key, value in base.weight_map.items()} == opposed.weight_map


@pytest.mark.parametrize(
    ("composition_id", "counts"),
    [
        ("near_balanced_41_40", Counter({0: 41, 1: 40})),
        ("minority_27_54", Counter({1: 54, 0: 27})),
    ],
)
def test_group_and_ghost_assignments_preserve_exact_composition(
    assets, composition_id, counts
) -> None:
    _context, _target, _grammar, environment, _catalogs = assets
    state = initial_movement_state(environment)
    groups = assign_executable_groups(state, composition_id, "assignment-test", 0)
    ghost = assign_ghost_labels(state, groups, "assignment-test")
    audit = group_assignment_audit(state, groups, ghost)
    assert Counter(groups.values()) == counts
    assert Counter(ghost.values()) == counts
    assert audit["success"]
    if composition_id == "minority_27_54":
        assert audit["runtimeByToken"] == {
            "A:0": 9,
            "A:1": 18,
            "B:0": 9,
            "B:1": 18,
            "C:0": 9,
            "C:1": 18,
        }


def test_2d_composition_correction_and_collapsed_label_control(assets) -> None:
    _context, _target, _grammar, environment, _catalogs = assets
    state = initial_movement_state(environment)
    groups = assign_executable_groups(state, "near_balanced_41_40", "metric-test", 2)
    metrics = graph_label_metrics(environment, state, groups)
    expected = (41 * 40 + 40 * 39) / (81 * 80)
    assert metrics["exactCompositionExpectation"] == pytest.approx(expected)
    collapsed = graph_label_metrics(
        environment, state, {identity: 0 for identity in groups}
    )
    assert collapsed["rawHomotypicEdgeFraction"] == 1.0
    assert collapsed["exactCompositionExpectation"] == 1.0
    assert collapsed["compositionCorrectedHomotypy"] == 0.0
    assert collapsed["categoricalAssortativity"] is None


def test_decluster_preserves_state_token_strata_and_never_increases_edges(
    assets,
) -> None:
    _context, _target, _grammar, environment, _catalogs = assets
    state = initial_movement_state(environment)
    groups = assign_executable_groups(state, "near_balanced_41_40", "decluster-test", 1)
    changed, audit = decluster_assignment(environment, state, groups, "decluster-test")
    tokens = {occupant.occupant_id: occupant.token for _, occupant in state.occupancy}
    assert Counter(changed.values()) == Counter(groups.values())
    assert Counter((tokens[key], value) for key, value in changed.items()) == Counter(
        (tokens[key], value) for key, value in groups.items()
    )
    assert audit["physicalStateSha256Before"] == audit["physicalStateSha256After"]
    assert audit["statePreserved"]
    assert audit["compositionPreserved"]
    assert audit["sameGroupEdgesNonincreasing"]
    assert audit["postSameGroupEdges"] <= audit["preSameGroupEdges"]


def test_local_swap_delta_is_exactly_brute_force_equivalent(assets) -> None:
    _context, _target, _grammar, environment, _catalogs = assets
    state = initial_movement_state(environment)
    groups = assign_executable_groups(state, "near_balanced_41_40", "delta-test", 3)
    labels = {
        site_id: groups[occupant.occupant_id] for site_id, occupant in state.occupancy
    }
    neighbors = {site.site_id: [] for site in environment.sites}
    for first, second in environment.edges:
        neighbors[first].append(second)
        neighbors[second].append(first)
    before = _same_edge_count(environment, labels)
    zeros = [site_id for site_id, label in labels.items() if label == 0]
    ones = [site_id for site_id, label in labels.items() if label == 1]
    for first in zeros:
        for second in ones:
            changed = dict(labels)
            changed[first], changed[second] = changed[second], changed[first]
            brute_force = _same_edge_count(environment, changed) - before
            assert (
                _swap_same_edge_delta(neighbors, labels, first, second) == brute_force
            )


def test_engine_heterogeneous_dispatch_is_no_channel_and_memory_free(assets) -> None:
    context, _target, grammar, environment, _catalogs = assets
    state = initial_movement_state(environment)
    identities = [occupant.occupant_id for _, occupant in state.occupancy]
    profile = compile_relation_profile(grammar)
    policies = {identity: "greedy_neighbor_satisfaction_v1" for identity in identities}
    profiles = {identity: profile for identity in identities}
    definition = EpisodeDefinition(
        scenario_id="s11-engine-boundary-test",
        environment_id=environment.environment_id,
        policy_id="greedy_neighbor_satisfaction_v1",
        relation_grammar_id=grammar.grammar_id,
        channel_mode="none",
        transitions=2,
        actor_batch_size=4,
        parameters={},
    )
    result = run_cpu_episode(
        context,
        definition,
        initial_state_override=state,
        actor_policy_assignments=policies,
        actor_relation_profiles=profiles,
    )
    assert result["permissionAudit"]["analysisLabelsUsed"] is False
    with pytest.raises(EpisodeValidationError, match="no-channel"):
        run_cpu_episode(
            context,
            replace(definition, channel_mode="global_summary"),
            initial_state_override=state,
            actor_policy_assignments=policies,
            actor_relation_profiles=profiles,
        )
    memory_policies = {identity: "memory_based_recovery_v1" for identity in identities}
    with pytest.raises(EpisodeValidationError, match="memory transfer"):
        run_cpu_episode(
            context,
            definition,
            initial_state_override=state,
            actor_policy_assignments=memory_policies,
            actor_relation_profiles=profiles,
        )


def test_identical_policy_labels_are_transition_blind_and_pair_replays(assets) -> None:
    catalog = assets[-1]["chimera"]
    specification = {
        "catalog": catalog,
        "phase": "test",
        "split": "test",
        "mixtureId": "aligned_identical_policy",
        "compositionId": "near_balanced_41_40",
        "startFamily": "exact_formed",
        "replicate": 0,
        "eventBudget": 8,
        "interventionTransition": 4,
        "retainTrace": False,
        "retainNullTrajectory": True,
    }
    first, _traces, audit = run_chimera_pair_once(specification)
    second, _traces2, audit2 = run_chimera_pair_once(specification)
    by_arm = {item["interventionArm"]: item for item in first}
    replay_by_arm = {item["interventionArm"]: item for item in second}
    assert (
        by_arm["native"]["episodeSha256"]
        == by_arm["executable_decluster"]["episodeSha256"]
    )
    assert (
        by_arm["native"]["episodeCanonicalBytesSha256"]
        == by_arm["executable_decluster"]["episodeCanonicalBytesSha256"]
    )
    for arm in by_arm:
        assert by_arm[arm]["episodeSha256"] == replay_by_arm[arm]["episodeSha256"]
        assert (
            by_arm[arm]["metricSummarySha256"]
            == replay_by_arm[arm]["metricSummarySha256"]
        )
        assert by_arm[arm]["permissionAuditSuccess"]
        assert by_arm[arm]["channelTotalInformationBits"] == 0
    assert audit["intervention"] == audit2["intervention"]
    assert audit["intervention"]["statePreserved"]
    assert audit["intervention"]["labelShamRuntimeBytesEqualByConstruction"]


def test_catalog_yaml_round_trip_has_no_unpriced_channels() -> None:
    catalog = yaml.safe_load(
        (ROOT / "configs/morphologies/chimera_catalog.yaml").read_text(encoding="utf-8")
    )
    assert catalog["backend"]["gpuUsedForS11"] is False
    assert catalog["interventions"]["costs"]["stateReadBitsPerIdentity"] == 10
    assert catalog["outcomes"]["repairClaimsForbidden"] is True
    assert catalog["analysisLabels"]["policyVisibility"] == "forbidden"
