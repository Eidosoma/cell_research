from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.morph2d.channels import (
    CHANNEL_LEDGER_FIELDS,
    INSTRUCTION_ALPHABET,
    ChannelValidationError,
    DirectControllerDecision,
    GlobalSummarySource,
    build_boundary_delivery,
    build_direct_controller_view,
    build_global_summary_delivery,
    build_gradient_delivery_from_s05,
    build_sparse_instruction_delivery,
    calibrate_noise,
    canonical_delivery_bytes,
    canonical_direct_event_bytes,
    canonical_direct_view_bytes,
    channel_comparison_vector,
    channel_to_dict,
    compile_boundary_levels,
    compile_static_gradient,
    controller_source_forbidden_accesses,
    decide_direct_controller,
    execute_direct_intervention,
    load_channel_catalog,
    parse_channel_catalog,
    parse_delivery,
    realize_noisy_gradient,
    state_blind_recipient_alias,
    validate_channel_payload,
    validate_controller_view_payload,
    validate_ledger_against_budget,
    validate_source_projection,
)
from src.morph2d.environments import load_environment_catalog
from src.morph2d.grammar import load_grammar_catalog
from src.morph2d.movements import (
    initial_movement_state,
    make_proposal,
    parse_movement_state,
    resolve_batch,
)
from src.morph2d.policies import (
    build_policy_observation,
    compile_relation_profile,
    load_policy_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ROOT / "configs/morphologies/control_channel_catalog.yaml"
ENVIRONMENTS = ROOT / "configs/morphologies/environment_catalog.yaml"
POLICIES = ROOT / "configs/morphologies/policy_catalog.yaml"
GRAMMARS = ROOT / "configs/morphologies/grammar_catalog.yaml"


@pytest.fixture(scope="module")
def context():
    raw = yaml.safe_load(CHANNELS.read_text(encoding="utf-8"))
    metadata, channels = load_channel_catalog(CHANNELS)
    _, environments = load_environment_catalog(ENVIRONMENTS)
    _, policies = load_policy_catalog(POLICIES)
    _, grammars = load_grammar_catalog(GRAMMARS)
    return {
        "raw": raw,
        "metadata": metadata,
        "channels": {item.channel_id: item for item in channels},
        "environments": {item.environment_id: item for item in environments},
        "policies": {item.policy_id: item for item in policies},
        "grammars": {item.grammar_id: item for item in grammars},
    }


def _actor_at(state, site_id: str) -> str:
    return state.occupant_map[site_id].occupant_id


def _perturbed_stripes(context):
    environment = context["environments"]["square_bounded_occupied_stripes"]
    state = initial_movement_state(environment)
    proposal = make_proposal(state, "adjacent_swap", ("r2_c0", "r3_c0"))
    result = resolve_batch(
        environment, state, (proposal,), batch_nonce="s06-direct-damage"
    )
    return environment, parse_movement_state(result["postState"])


def _greedy_build(context):
    environment, state = _perturbed_stripes(context)
    policy = context["policies"]["greedy_neighbor_satisfaction_v1"]
    profile = compile_relation_profile(context["grammars"]["stripes_axis_relations_v1"])
    build = build_policy_observation(
        environment,
        state,
        _actor_at(state, "r2_c0"),
        policy,
        relation_profile=profile,
    )
    return environment, state, build


def _gradient_delivery(context):
    definition = context["channels"]["static_gradient_v1"]
    environment = context["environments"]["square_periodic_vacancy"]
    policy = context["policies"]["gradient_following_v1"]
    state = initial_movement_state(environment)
    field, configuration = compile_static_gradient(
        environment, definition, axis="second", direction="up"
    )
    noisy, noise_draws = realize_noisy_gradient(
        definition, field, scenario_key="s06-gradient-test"
    )
    build = build_policy_observation(
        environment,
        state,
        _actor_at(state, "r2_c2"),
        policy,
        gradient_levels=noisy,
        gradient_direction="up",
    )
    delivery = build_gradient_delivery_from_s05(
        definition,
        build,
        compiled_field=field,
        direction="up",
        configuration_ledger=configuration,
        noise_draws=noise_draws,
        epoch_index=0,
    )
    return definition, environment, state, field, noisy, build, delivery


def _boundary_delivery(context):
    definition = context["channels"]["natural_boundary_signal_v1"]
    environment = context["environments"]["square_bounded_occupied_stripes"]
    policy = context["policies"]["boundary_seeking_v1"]
    state = initial_movement_state(environment)
    build = build_policy_observation(
        environment,
        state,
        _actor_at(state, "r1_c1"),
        policy,
        boundary_direction="seek",
        boundary_tokens=("A",),
    )
    levels, configuration = compile_boundary_levels(environment, definition)
    candidates = {
        key: levels[affordance.target_site]
        for key, affordance in build.candidate_map.items()
    }
    source_site = next(iter(build.candidate_map.values())).proposal.source_site
    delivery = build_boundary_delivery(
        definition,
        current_level=levels[source_site],
        candidate_levels=candidates,
        direction="seek",
        directive_applies=True,
        configuration_bits=configuration["configurationBits"],
        scenario_key="s06-boundary-test",
        epoch_index=0,
    )
    return definition, environment, levels, delivery


def _summary_delivery(context):
    definition = context["channels"]["lagged_global_summary_v1"]
    source = GlobalSummarySource(
        source_epoch=3,
        local_dissatisfaction_bits=(1, 0, 1, 0, 1, 0, 1, 1),
        submitted_proposals=8,
        conflict_losses=2,
        active_count=8,
    )
    delivery = build_global_summary_delivery(
        definition,
        source,
        scenario_key="s06-summary-test",
        epoch_index=4,
        recipient_count=8,
    )
    return definition, source, delivery


def test_catalog_defines_five_channels_and_round_trips(context) -> None:
    channels = tuple(context["channels"].values())
    assert len(channels) == 5
    assert {item.channel_type for item in channels} == {
        "static_gradient",
        "boundary_signal",
        "sparse_instruction",
        "global_summary",
        "direct_intervention",
    }
    assert tuple(context["raw"]["instructionAlphabet"]) == INSTRUCTION_ALPHABET
    reloaded = yaml.safe_load(yaml.safe_dump(context["raw"], sort_keys=False))
    _, parsed = parse_channel_catalog(reloaded)
    assert [channel_to_dict(item) for item in parsed] == [
        channel_to_dict(item) for item in channels
    ]
    assert context["raw"]["commonBudgetContract"]["ledgerFields"] == list(
        CHANNEL_LEDGER_FIELDS
    )


def test_catalog_freezes_every_required_semantic_dimension(context) -> None:
    for definition in context["channels"].values():
        assert definition.source["owner"]
        assert definition.source["implementation"]
        assert definition.source["allowedInputs"]
        assert definition.semantic_content
        assert definition.spatial_resolution
        assert definition.update_schedule
        assert definition.noise_model["kind"]
        assert definition.bandwidth["configurationBitsFormula"]
        assert definition.action_cost["kind"]
        assert "allowed" in definition.controller_state_permissions
        assert definition.target_specificity
        assert definition.unavoidable_asymmetries


def test_catalog_rejects_hidden_state_in_direct_source_or_controller_memory(
    context,
) -> None:
    raw = json.loads(json.dumps(context["raw"]))
    direct = next(
        item for item in raw["channels"] if item["channelType"] == "direct_intervention"
    )
    direct["source"]["allowedInputs"].append("occupancy")
    with pytest.raises(ChannelValidationError, match="hidden state"):
        parse_channel_catalog(raw)

    raw = json.loads(json.dumps(context["raw"]))
    summary = next(
        item for item in raw["channels"] if item["channelType"] == "global_summary"
    )
    summary["controllerStatePermissions"]["allowed"].append("future_state")
    with pytest.raises(ChannelValidationError, match="hidden state"):
        parse_channel_catalog(raw)


def test_controller_and_channel_payloads_reject_hidden_engine_state() -> None:
    hidden = (
        "fullState",
        "occupancy",
        "siteTokens",
        "siteIds",
        "route",
        "occupantIdentities",
        "actorId",
        "siteRoles",
        "rawBoundaryTags",
        "currentBatchProposals",
        "conflictPriority",
        "analysisLabel",
        "s02WholeGridScore",
        "s01GlobalCompletionAudit",
        "targetMembership",
        "futureState",
    )
    for key in hidden:
        with pytest.raises(ChannelValidationError, match="forbidden"):
            validate_controller_view_payload({key: "leak"})
        with pytest.raises(ChannelValidationError, match="forbidden"):
            validate_channel_payload({key: "leak"})
    assert controller_source_forbidden_accesses() == []


def test_source_projection_uses_exact_allowlist(context) -> None:
    definition = context["channels"]["static_gradient_v1"]
    validate_source_projection(
        definition,
        {
            "environment_geometry": "square",
            "display_coordinates": "engine_only",
            "frozen_axis": "second",
            "frozen_direction": "up",
        },
    )
    with pytest.raises(ChannelValidationError, match="not allowlisted"):
        validate_source_projection(definition, {"occupancy": ["A", "B"]})


def test_static_gradient_is_state_blind_deterministic_and_s05_compatible(
    context,
) -> None:
    definition, environment, _, field, noisy, build, delivery = _gradient_delivery(
        context
    )
    repeated_field, repeated_configuration = compile_static_gradient(
        environment, definition, axis="second", direction="up"
    )
    repeated_noisy, repeated_draws = realize_noisy_gradient(
        definition, field, scenario_key="s06-gradient-test"
    )
    assert field == repeated_field
    assert noisy == repeated_noisy
    assert repeated_draws == len(field)
    assert min(field.values()) == 0
    assert max(field.values()) == 255
    assert build.observation.payload == delivery.payload
    assert repeated_configuration["configurationBits"] == 8 * len(field) + 1
    assert (
        delivery.ledger["policyDeliveryBits"]
        == build.observation.budget["communicatedBitsUpperBound"]
    )
    assert validate_ledger_against_budget(definition, delivery.ledger) == []


def test_boundary_projection_hides_tags_and_periodic_exterior_is_zero(context) -> None:
    definition, environment, levels, delivery = _boundary_delivery(context)
    assert max(levels.values()) > 0
    assert "boundarytags" not in json.dumps(delivery.payload).lower()
    assert "site" not in json.dumps(delivery.payload).lower()
    assert delivery.ledger["noiseDraws"] == 1 + len(delivery.payload["candidates"])
    assert validate_ledger_against_budget(definition, delivery.ledger) == []

    periodic = context["environments"]["square_periodic_vacancy"]
    periodic_levels, _ = compile_boundary_levels(periodic, definition)
    assert set(periodic_levels.values()) == {0}


def test_sparse_instruction_is_coarse_state_blind_priced_and_replayable(
    context,
) -> None:
    definition = context["channels"]["sparse_instruction_v1"]
    first = build_sparse_instruction_delivery(
        definition,
        instruction="prefer_exploration",
        region_alias="zone_1",
        region_size=4,
        region_count=5,
        scenario_key="s06-instruction-test",
        epoch_index=2,
    )
    second = build_sparse_instruction_delivery(
        definition,
        instruction="prefer_exploration",
        region_alias="zone_1",
        region_size=4,
        region_count=5,
        scenario_key="s06-instruction-test",
        epoch_index=2,
    )
    assert canonical_delivery_bytes(first) == canonical_delivery_bytes(second)
    assert first.ledger["recipientDeliveries"] == 4
    assert first.ledger["policyDeliveryBits"] == 16
    assert first.ledger["actuationAttempts"] == 0
    assert validate_ledger_against_budget(definition, first.ledger) == []
    with pytest.raises(ChannelValidationError, match="too fine"):
        build_sparse_instruction_delivery(
            definition,
            instruction="prefer_exploration",
            region_alias="cell_1",
            region_size=1,
            region_count=5,
            scenario_key="bad",
            epoch_index=0,
        )


def test_global_summary_is_binned_delayed_broadcast_and_not_completion(context) -> None:
    definition, source, delivery = _summary_delivery(context)
    assert set(delivery.payload) == {
        "dissatisfactionBin",
        "laggedConflictBin",
        "summaryAgeEpochs",
    }
    assert delivery.payload["summaryAgeEpochs"] == 1
    assert delivery.ledger["recipientDeliveries"] == 8
    assert delivery.ledger["policyDeliveryBits"] == 48
    assert "completion" not in json.dumps(delivery.payload).lower()
    assert validate_ledger_against_budget(definition, delivery.ledger) == []
    with pytest.raises(ChannelValidationError, match="delayed"):
        build_global_summary_delivery(
            definition,
            source,
            scenario_key="bad-delay",
            epoch_index=5,
            recipient_count=8,
        )


def test_direct_controller_gets_one_state_blind_alias_and_one_s05_payload(
    context,
) -> None:
    definition = context["channels"]["sparse_direct_intervention_v1"]
    _, _, build = _greedy_build(context)
    view = build_direct_controller_view(
        definition,
        build,
        epoch_index=3,
        population_size=81,
        lagged_summary={
            "dissatisfactionBin": 5,
            "laggedConflictBin": 1,
            "summaryAgeEpochs": 1,
        },
        remaining_information_budget=352,
        remaining_action_budget=1,
    )
    assert view.recipient_alias == state_blind_recipient_alias(3, 81) == "u0003"
    assert set(view.payload) == {
        "epochIndex",
        "recipientAlias",
        "laggedGlobalSummary",
        "localPolicyPayload",
        "remainingInformationBudget",
        "remainingActionBudget",
    }
    assert json.dumps(build.candidate_map, default=str) not in json.dumps(view.payload)
    assert view.ledger["recipientDeliveries"] == 1
    assert view.ledger["addressBits"] == 7
    assert validate_ledger_against_budget(definition, view.ledger) == []


def test_direct_intervention_materializes_engine_side_and_replays(context) -> None:
    definition = context["channels"]["sparse_direct_intervention_v1"]
    environment, state, build = _greedy_build(context)
    view = build_direct_controller_view(
        definition,
        build,
        epoch_index=3,
        population_size=81,
        lagged_summary={
            "dissatisfactionBin": 5,
            "laggedConflictBin": 1,
            "summaryAgeEpochs": 1,
        },
        remaining_information_budget=352,
        remaining_action_budget=1,
    )
    decision = decide_direct_controller(view.payload)
    assert decision.action == "select_candidate"
    first = execute_direct_intervention(
        definition,
        environment,
        state,
        build,
        view,
        decision,
        scenario_key="s06-direct-success-2",
        batch_nonce="s06-direct-test",
    )
    second = execute_direct_intervention(
        definition,
        environment,
        state,
        build,
        view,
        decision,
        scenario_key="s06-direct-success-2",
        batch_nonce="s06-direct-test",
    )
    assert canonical_direct_event_bytes(first) == canonical_direct_event_bytes(second)
    assert first["materializationAudit"]["controllerSawProposalEnvelope"] is False
    assert first["movementBatchResult"]["invariants"]["success"]
    assert first["ledger"]["actuationAttempts"] == 1
    assert first["ledger"]["suppressedNativeActions"] == 0
    assert (
        first["ledger"]["movementGraphDisplacement"]
        == first["movementBatchResult"]["costLedger"]["totalGraphDisplacement"]
    )
    assert validate_ledger_against_budget(definition, first["ledger"]) == []
    with pytest.raises(ChannelValidationError, match="unsupported"):
        execute_direct_intervention(
            definition,
            environment,
            state,
            build,
            view,
            DirectControllerDecision("suppress_native", None, "probe", None),
            scenario_key="s06-direct-suppression-probe",
            batch_nonce="s06-direct-suppression-probe",
        )


def test_direct_controller_respects_zero_action_budget(context) -> None:
    definition = context["channels"]["sparse_direct_intervention_v1"]
    _, _, build = _greedy_build(context)
    view = build_direct_controller_view(
        definition,
        build,
        epoch_index=0,
        population_size=81,
        lagged_summary={
            "dissatisfactionBin": 0,
            "laggedConflictBin": 0,
            "summaryAgeEpochs": 1,
        },
        remaining_information_budget=352,
        remaining_action_budget=0,
    )
    decision = decide_direct_controller(view.payload)
    assert decision.action == "noop"
    assert decision.reason == "action_budget_exhausted"


def test_channel_serialization_roundtrip_and_tamper_detection(context) -> None:
    _, _, _, _, _, _, delivery = _gradient_delivery(context)
    encoded = canonical_delivery_bytes(delivery)
    parsed = parse_delivery(json.loads(encoded))
    assert canonical_delivery_bytes(parsed) == encoded
    tampered = json.loads(encoded)
    tampered["payload"]["currentGradientLevel"] += 1
    with pytest.raises(ChannelValidationError, match="mismatch"):
        parse_delivery(tampered)


def test_direct_view_serialization_is_deterministic(context) -> None:
    definition = context["channels"]["sparse_direct_intervention_v1"]
    _, _, build = _greedy_build(context)
    kwargs = {
        "epoch_index": 3,
        "population_size": 81,
        "lagged_summary": {
            "dissatisfactionBin": 5,
            "laggedConflictBin": 1,
            "summaryAgeEpochs": 1,
        },
        "remaining_information_budget": 352,
        "remaining_action_budget": 1,
    }
    first = build_direct_controller_view(definition, build, **kwargs)
    second = build_direct_controller_view(definition, build, **kwargs)
    assert canonical_direct_view_bytes(first) == canonical_direct_view_bytes(second)


def test_noise_calibration_passes_frozen_100000_draw_tolerance(context) -> None:
    for definition in context["channels"].values():
        calibration = calibrate_noise(definition)
        assert calibration["sampleCount"] == 100000
        assert calibration["maximumAbsoluteError"] <= 0.01
        assert calibration["success"]


def test_comparison_vectors_keep_information_compute_and_action_separate(
    context,
) -> None:
    gradient_definition, *_, gradient = _gradient_delivery(context)
    summary_definition, _, summary = _summary_delivery(context)
    vectors = [
        channel_comparison_vector(gradient_definition, gradient.ledger),
        channel_comparison_vector(summary_definition, summary.ledger),
    ]
    assert all(item["scalarCollapseAllowed"] is False for item in vectors)
    assert vectors[0]["configurationBits"] != vectors[1]["configurationBits"]
    assert all(item["semanticContent"] for item in vectors)


def test_s05_observation_is_not_mutated_by_channel_attachment(context) -> None:
    _, _, build = _greedy_build(context)
    before = json.dumps(build.observation.payload, sort_keys=True)
    definition = context["channels"]["sparse_direct_intervention_v1"]
    _ = build_direct_controller_view(
        definition,
        build,
        epoch_index=2,
        population_size=81,
        lagged_summary={
            "dissatisfactionBin": 4,
            "laggedConflictBin": 1,
            "summaryAgeEpochs": 1,
        },
        remaining_information_budget=352,
        remaining_action_budget=1,
    )
    assert json.dumps(build.observation.payload, sort_keys=True) == before


def test_every_fixture_ledger_has_exact_schema_nonnegative_values_and_budget(
    context,
) -> None:
    gradient_definition, *_, gradient = _gradient_delivery(context)
    boundary_definition, _, _, boundary = _boundary_delivery(context)
    instruction_definition = context["channels"]["sparse_instruction_v1"]
    instruction = build_sparse_instruction_delivery(
        instruction_definition,
        instruction="prefer_exploration",
        region_alias="zone_1",
        region_size=4,
        region_count=5,
        scenario_key="s06-ledger-instruction",
        epoch_index=2,
    )
    summary_definition, _, summary = _summary_delivery(context)
    records = (
        (gradient_definition, gradient.ledger),
        (boundary_definition, boundary.ledger),
        (instruction_definition, instruction.ledger),
        (summary_definition, summary.ledger),
    )
    for definition, ledger in records:
        assert set(ledger) == set(CHANNEL_LEDGER_FIELDS)
        assert all(value >= 0 for value in ledger.values())
        assert validate_ledger_against_budget(definition, ledger) == []
