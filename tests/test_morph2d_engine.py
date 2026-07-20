from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.morph2d.channels import compile_static_gradient, realize_noisy_gradient
from src.morph2d.engine import (
    CHANNEL_LEDGER_FIELDS,
    EngineContext,
    canonical_episode_result_bytes,
    load_engine_context,
    replay_cpu_episode,
    run_cpu_episode,
)
from src.morph2d.gpu_engine import (
    MAX_CANDIDATES,
    compile_validated_proposal_batch,
    decode_post_state_occupant_ids,
    decide_masked_observations,
    encode_masked_observations,
    expected_post_state_occupant_ids,
    masked_observation_tensor_names,
    resolve_compiled_proposals,
    synthetic_compiled_workload,
    tensor_permutation_invariants,
    validate_masked_decision_parity,
)
from src.morph2d.movements import (
    initial_movement_state,
    make_proposal,
    parse_movement_state,
    resolve_batch,
)
from src.morph2d.policies import (
    PolicyMemory,
    build_policy_observation,
    compile_relation_profile,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def context() -> EngineContext:
    return load_engine_context(
        ROOT / "configs/morphologies/engine_catalog.yaml",
        environment_catalog=ROOT / "configs/morphologies/environment_catalog.yaml",
        policy_catalog=ROOT / "configs/morphologies/policy_catalog.yaml",
        grammar_catalog=ROOT / "configs/morphologies/grammar_catalog.yaml",
        channel_catalog=ROOT / "configs/morphologies/control_channel_catalog.yaml",
    )


def _episode(context: EngineContext, scenario_id: str, transitions: int):
    definition = next(
        item for item in context.episodes if item.scenario_id == scenario_id
    )
    return replace(definition, transitions=transitions)


def _actor_at(state, site_id: str) -> str:
    return state.occupant_map[site_id].occupant_id


def _damaged_stripes(context: EngineContext):
    environment = context.environments["square_bounded_occupied_stripes"]
    state = initial_movement_state(environment)
    proposal = make_proposal(state, "adjacent_swap", ("r2_c0", "r3_c0"))
    result = resolve_batch(
        environment, state, (proposal,), batch_nonce="s07-engine-test-damage"
    )
    return environment, parse_movement_state(result["postState"])


def _six_policy_observations(context: EngineContext):
    stripes, damaged = _damaged_stripes(context)
    profile = compile_relation_profile(context.grammars["stripes_axis_relations_v1"])
    greedy = context.policies["greedy_neighbor_satisfaction_v1"]
    memory = context.policies["memory_based_recovery_v1"]
    conflict = context.policies["conflict_avoidance_v1"]
    greedy_build = build_policy_observation(
        stripes,
        damaged,
        _actor_at(damaged, "r2_c0"),
        greedy,
        relation_profile=profile,
    )
    memory_build = build_policy_observation(
        stripes,
        damaged,
        _actor_at(damaged, "r2_c0"),
        memory,
        relation_profile=profile,
        memory=PolicyMemory(12, 1),
    )
    conflict_build = build_policy_observation(
        stripes,
        damaged,
        _actor_at(damaged, "r2_c0"),
        conflict,
        relation_profile=profile,
        lagged_conflicts={site.site_id: 0 for site in stripes.occupiable_sites},
    )

    boundary = context.policies["boundary_seeking_v1"]
    boundary_state = initial_movement_state(stripes)
    boundary_build = build_policy_observation(
        stripes,
        boundary_state,
        _actor_at(boundary_state, "r1_c1"),
        boundary,
        boundary_direction="seek",
        boundary_tokens=("A",),
    )

    periodic = context.environments["square_periodic_vacancy"]
    periodic_state = initial_movement_state(periodic)
    gradient = context.policies["gradient_following_v1"]
    field, _ = compile_static_gradient(
        periodic,
        context.channels["static_gradient_v1"],
        axis="second",
        direction="up",
    )
    noisy, _ = realize_noisy_gradient(
        context.channels["static_gradient_v1"],
        field,
        scenario_key="s07-masked-observation-test",
    )
    gradient_build = build_policy_observation(
        periodic,
        periodic_state,
        _actor_at(periodic_state, "r2_c2"),
        gradient,
        gradient_levels=noisy,
        gradient_direction="up",
    )

    hexagonal = context.environments["hexagonal_bounded_occupied"]
    hex_state = initial_movement_state(hexagonal)
    exploration = context.policies["exploration_v1"]
    exploration_build = build_policy_observation(
        hexagonal,
        hex_state,
        _actor_at(hex_state, "q0_r0"),
        exploration,
        decision_key="s07-masked-observation-test",
        activation_index=7,
    )
    return (
        [
            greedy_build,
            boundary_build,
            gradient_build,
            exploration_build,
            memory_build,
            conflict_build,
        ],
        [greedy, boundary, gradient, exploration, memory, conflict],
    )


def test_engine_catalog_freezes_cpu_gpu_scope_and_all_channels(context) -> None:
    assert len(context.episodes) == 9
    assert {item.channel_mode for item in context.episodes} == {
        "none",
        "static_gradient",
        "boundary_signal",
        "sparse_instruction",
        "global_summary",
        "direct_intervention",
    }
    scope = context.metadata["scope"]
    assert scope["gpuControlPlane"].startswith("cpu_exact")
    assert "configuration_cost_amortization" in scope["forbiddenChanges"]
    assert context.metadata["episodeContract"]["epochLengthTransitions"] == 16


def test_cpu_episode_replays_exactly_and_preserves_invariants(context) -> None:
    definition = _episode(context, "s07-greedy-local", 4)
    first = run_cpu_episode(context, definition)
    second = replay_cpu_episode(context, definition, first)
    assert canonical_episode_result_bytes(first) == canonical_episode_result_bytes(
        second
    )
    assert len(first["transitionSummaries"]) == 4
    assert first["movementLedger"]["acceptedMovements"] > 0
    assert first["permissionAudit"] == {
        "s05PayloadWidened": False,
        "controllerAuthorityWidened": False,
        "globalEvaluationUsedByPolicyOrController": False,
        "analysisLabelsUsed": False,
        "futureStateUsed": False,
    }
    assert first["evaluationSeparation"]["s02LocalFeedback"] == {
        "pricedObservationField": "localRelationDelta",
        "activeForEpisodePolicy": True,
    }
    assert first["evaluationSeparation"]["s01GlobalCompletion"] == {
        "fields": ["canonicalEquivalence", "componentAudit", "topologyAudit"],
        "computedOnline": False,
        "policyOrControllerAccessible": False,
    }
    occupants = [item[1] for item in first["finalState"]["occupantProjection"]]
    assert len(occupants) == len(set(occupants)) == 81


@pytest.mark.parametrize(
    ("scenario_id", "transitions", "configuration_bits"),
    [
        ("s07-gradient-static", 2, 161),
        ("s07-boundary-natural", 2, 245),
        ("s07-instruction-advisory", 17, 21),
        ("s07-summary-advisory", 17, 8),
        ("s07-direct-bounded", 17, 72),
    ],
)
def test_channel_timing_and_configuration_are_not_amortized(
    context, scenario_id: str, transitions: int, configuration_bits: int
) -> None:
    result = run_cpu_episode(
        context,
        _episode(context, scenario_id, transitions),
        include_selected_traces=False,
    )
    assert set(result["channelLedger"]) == set(CHANNEL_LEDGER_FIELDS)
    assert result["channelLedger"]["configurationBits"] == configuration_bits
    assert result["configurationAccounting"] == {
        "chargedOnceInFull": True,
        "amortizedOrDivided": False,
        "configurationBits": configuration_bits,
    }
    if scenario_id == "s07-instruction-advisory":
        assert [item["epochIndex"] for item in result["channelEvents"]] == [0, 1]
    if scenario_id == "s07-summary-advisory":
        assert [item["epochIndex"] for item in result["channelEvents"]] == [1]
    if scenario_id == "s07-direct-bounded":
        assert (
            sum(item["directIntervention"] for item in result["transitionSummaries"])
            == 1
        )
        assert {item["channelId"] for item in result["channelEvents"]} == {
            "lagged_global_summary_v1",
            "sparse_direct_intervention_v1",
        }


def test_masked_policy_kernel_matches_all_six_cpu_decisions(context) -> None:
    builds, definitions = _six_policy_observations(context)
    encoded = encode_masked_observations([builds], [definitions], device="cuda")
    rows = validate_masked_decision_parity(encoded, [builds], [definitions])
    assert len(rows) == 6
    assert all(item["match"] for item in rows)
    assert decide_masked_observations(encoded).selected_candidate_index.shape == (
        1,
        6,
    )
    forbidden = {
        "actor_identity",
        "site_ids",
        "routes",
        "occupant_identities",
        "analysis_labels",
        "global_completion",
        "future_state",
    }
    assert not (set(masked_observation_tensor_names()) & forbidden)


def _conflict_proposals(context: EngineContext):
    environment = context.environments["square_bounded_occupied_stripes"]
    state = initial_movement_state(environment)
    proposals = [
        make_proposal(state, "adjacent_swap", ("r2_c0", "r3_c0")),
        make_proposal(state, "adjacent_swap", ("r3_c0", "r4_c0")),
        make_proposal(state, "adjacent_swap", ("r2_c3", "r3_c3")),
        make_proposal(state, "short_exchange", ("r2_c2", "r2_c3", "r3_c3")),
        make_proposal(state, "adjacent_swap", ("r2_c8", "r3_c8")),
    ]
    return environment, state, proposals


def test_gpu_transition_matches_exact_s04_cpu_and_proposal_order(context) -> None:
    environment, state, proposals = _conflict_proposals(context)
    compiled, expected = compile_validated_proposal_batch(
        environment,
        [state, state],
        [proposals, list(reversed(proposals))],
        ["s07-order", "s07-order"],
        device="cuda",
    )
    result = resolve_compiled_proposals(compiled)
    assert decode_post_state_occupant_ids(
        compiled, result
    ) == expected_post_state_occupant_ids(compiled, expected)
    assert result.accepted_movements.tolist() == [3, 3]
    assert result.total_graph_displacement.tolist() == [
        item["costLedger"]["totalGraphDisplacement"] for item in expected
    ]
    assert bool(tensor_permutation_invariants(compiled.state, result.post_state).all())
    accepted_ids = []
    for row in range(2):
        accepted_ids.append(
            {
                proposal_id
                for proposal_id, accepted in zip(
                    compiled.proposal_ids[row],
                    result.accepted_mask[row].tolist(),
                    strict=True,
                )
                if accepted
            }
        )
    assert accepted_ids[0] == accepted_ids[1] == set(expected[0]["acceptedProposalIds"])


def test_gpu_batch_order_invariance_and_exact_device_replay() -> None:
    compiled = synthetic_compiled_workload(32, device="cuda")
    first = resolve_compiled_proposals(compiled)
    second = resolve_compiled_proposals(compiled)
    assert torch.equal(first.post_state, second.post_state)
    assert torch.equal(first.accepted_mask, second.accepted_mask)
    permutation = torch.tensor(
        list(reversed(range(32))), dtype=torch.int64, device="cuda"
    )
    permuted = replace(
        compiled,
        state=compiled.state[permutation],
        route=compiled.route[permutation],
        route_length=compiled.route_length[permutation],
        kind_code=compiled.kind_code[permutation],
        rotation_direction=compiled.rotation_direction[permutation],
        priority_rank=compiled.priority_rank[permutation],
        proposal_mask=compiled.proposal_mask[permutation],
        proposal_displacement=compiled.proposal_displacement[permutation],
    )
    permuted_result = resolve_compiled_proposals(permuted)
    inverse = torch.argsort(permutation)
    assert torch.equal(first.post_state, permuted_result.post_state[inverse])
    assert torch.equal(first.accepted_mask, permuted_result.accepted_mask[inverse])


def test_engine_results_do_not_smuggle_evaluation_into_summary(context) -> None:
    result = run_cpu_episode(
        context,
        _episode(context, "s07-summary-advisory", 17),
        include_selected_traces=False,
    )
    serialized = json.dumps(result["transitionSummaries"], sort_keys=True).lower()
    assert "s01" not in serialized
    assert "wholegrid" not in serialized
    assert "analysislabel" not in serialized
    assert result["channelLedger"]["controllerInputBits"] == 0
    assert MAX_CANDIDATES == 16
