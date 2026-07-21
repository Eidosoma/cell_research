from __future__ import annotations

from src.morph2d.minimal_control import (
    build_transfer_fixture,
    graph_diagnostics,
    load_minimal_control_catalog,
    policy_catalog,
    run_minimal_once,
    run_transfer_once,
    scenario_identity,
    schedule_configuration_bits,
)
from scripts.build_morph2d_s14 import (
    holdout_tasks,
    training_tasks,
    validation_tasks,
)


def _minimal_spec(policy_id: str, transitions: int = 33):
    return {
        "phase": "unit",
        "split": "unit",
        "targetId": "layers_three_ordered_tissues",
        "challengeId": "exact_maintenance",
        "controlPolicyId": policy_id,
        "replicate": 2,
        "eventBudget": transitions,
        "retainTrace": False,
    }


def test_catalog_freezes_endpoint_splits_costs_and_transfer_boundary() -> None:
    catalog = load_minimal_control_catalog()
    assert catalog["researchStepId"] == "S14"
    assert len(catalog["interventionPolicies"]["searchCandidates"]) == 13
    assert catalog["budgets"]["scalarCostForbidden"]
    assert catalog["splits"]["heldout"]["e07ProtectedConfirmation"]
    assert (
        catalog["topologySuccessCriteria"]["transferDiagnostics"][0][
            "completionCalibrated"
        ]
        is False
    )
    assert catalog["endpointRules"]["s14HypothesisSupportRule"].startswith(
        "heldoutRetentionGate"
    )


def test_pairing_is_policy_blind_and_run_identity_is_policy_specific() -> None:
    first = scenario_identity("train", "layers", "exact", 4, "central_q0_freeze")
    second = scenario_identity("train", "layers", "exact", 4, "central_q7_full")
    assert first["scenarioId"] == second["scenarioId"]
    assert first["pairingBlockId"] == second["pairingBlockId"]
    assert first["seedHex"] == second["seedHex"]
    assert first["runId"] != second["runId"]


def test_schedule_bits_are_charged_once_and_not_scalarized() -> None:
    policies = policy_catalog(load_minimal_control_catalog())
    assert schedule_configuration_bits(policies["local_only_control"]) == 0
    assert schedule_configuration_bits(policies["central_q0_freeze"]) == 8
    assert schedule_configuration_bits(policies["central_q7_full"]) == 29


def test_zero_query_central_freeze_replays_and_prices_foregone_opportunity() -> None:
    first, _, _ = run_minimal_once(_minimal_spec("central_q0_freeze"))
    second, _, _ = run_minimal_once(_minimal_spec("central_q0_freeze"))
    excluded = {"wallSeconds"}
    assert {k: v for k, v in first.items() if k not in excluded} == {
        k: v for k, v in second.items() if k not in excluded
    }
    assert first["directRecipientQuerySlots"] == 0
    assert first["acceptedMovements"] == 0
    assert first["foregoneNativeActorSlots"] == 132
    assert first["terminalConjunctiveCompletion"]
    assert first["s14BudgetSuccess"]
    assert first["permissionAuditSuccess"]


def test_narrow_query_reduces_controller_input_without_reducing_source_work() -> None:
    narrow, _, _ = run_minimal_once(_minimal_spec("central_q4_low1"))
    full, _, _ = run_minimal_once(_minimal_spec("central_q4_spread_full"))
    assert narrow["directRecipientQuerySlots"] == full["directRecipientQuerySlots"] == 2
    assert narrow["controllerInputBits"] < full["controllerInputBits"]
    assert narrow["policyLogicalSourceReads"] == full["policyLogicalSourceReads"]
    assert narrow["policyUtilityEvaluations"] == full["policyUtilityEvaluations"]


def test_action_disconnected_sham_keeps_query_cost_and_blocks_actuation() -> None:
    sham, _, _ = run_minimal_once(_minimal_spec("central_monitor_sham_control"))
    active, _, _ = run_minimal_once(_minimal_spec("bounded_central_full_control"))
    assert sham["directRecipientQuerySlots"] == active["directRecipientQuerySlots"] == 2
    assert sham["controllerInputBits"] == active["controllerInputBits"]
    assert sham["controllerComputeUnits"] == active["controllerComputeUnits"]
    assert sham["actuationAttempts"] == 0
    assert sham["shamSuppressedControllerRecommendations"] >= 1


def test_transfer_fixtures_are_connected_unbound_diagnostics() -> None:
    for fixture_id, expected_sites in (
        ("larger_square_layers_15x15", 225),
        ("irregular_pruned_layers_9x9", 81),
    ):
        environment, reference = build_transfer_fixture(fixture_id)
        assert len(environment.occupiable_sites) == expected_sites
        assert len(reference) == expected_sites
        diagnostics = graph_diagnostics(environment, reference, reference)
        assert diagnostics["referenceMismatchCount"] == 0
        assert diagnostics["graphComponentError"] == 0


def test_transfer_run_has_null_completion_fields_and_exact_replay() -> None:
    specification = {
        "phase": "unit_transfer",
        "split": "unit_transfer",
        "fixtureId": "irregular_pruned_layers_9x9",
        "challengeId": "one_swap_displacement_diagnostic",
        "controlPolicyId": "central_q0_freeze",
        "replicate": 1,
        "eventBudget": 17,
        "retainTrace": False,
    }
    first, _ = run_transfer_once(specification)
    second, _ = run_transfer_once(specification)
    assert first == second
    assert first["topologyCompletionCalibrated"] is False
    assert first["terminalConjunctiveCompletion"] is None
    assert first["terminalS01GlobalSuccess"] is None
    assert first["terminalS02GrammarAccepted"] is None
    assert first["invariantSuccess"]
    assert first["s14BudgetSuccess"]


def test_builder_freezes_search_and_seals_holdout_task_construction() -> None:
    catalog = load_minimal_control_catalog()
    training = training_tasks(catalog)
    assert len(training) == 13
    assert sum(len(item["replicates"]) for item in training) == 3_250
    validation = validation_tasks(catalog, ["central_q0_freeze"])
    assert len(validation) == 4
    assert sum(len(item["replicates"]) for item in validation) == 4_000
    heldout, negative, transfer = holdout_tasks(catalog, ["central_q0_freeze"])
    assert len(heldout) == 8
    assert len(negative) == 8
    assert len(transfer) == 16
    assert sum(len(item["replicates"]) for item in heldout) == 8_000
    assert sum(len(item["replicates"]) for item in negative) == 8_000
    assert sum(len(item["replicates"]) for item in transfer) == 4_000
