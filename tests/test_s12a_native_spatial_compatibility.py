from __future__ import annotations

from copy import deepcopy

import pytest

from src.environment_suite.contracts import canonical_sha256
from src.morph2d.minimal_control import build_transfer_fixture
from src.morph2d.movements import initial_movement_state
from src.policy_dsl import compile_policy
from src.spatial_transfer.qualification import (
    PANEL_IDS,
    SPATIAL_TASKS,
    DiagnosticSpatialTracker,
    apply_adaptation_variant,
    build_panel_fixture,
    canonical_binding_digest,
    endpoint_contract,
    qualify_configuration_binding,
    qualify_endpoint,
    run_fail_atomic_fixture_batch,
)


def _noop_document(policy_id: str) -> dict[str, object]:
    return {
        "schemaVersion": "e07.policy-dsl.v1",
        "policyId": policy_id,
        "environment": "spatial2d.v1",
        "permissions": [],
        "memory": [],
        "signals": {"channels": 0, "bitsPerChannel": 0},
        "limits": {
            "maxRules": 1,
            "maxExpressionNodes": 1,
            "maxActionsPerActivation": 1,
            "maxOperationsPerActivation": 8,
            "maxCandidates": 4,
            "maxMovementRadius": 0,
        },
        "rules": [
            {
                "when": {"const": False},
                "actions": [{"kind": "noop"}],
            }
        ],
        "default": {"actions": [{"kind": "noop"}]},
    }


def _configuration() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    documents = [_noop_document("s12a_test_noop_a"), _noop_document("s12a_test_noop_b")]
    compiled = [compile_policy(item) for item in documents]
    members = [
        {
            "policyId": policy.policy_id,
            "policySha256": policy.policy_sha256,
            "nativeCarrier": "spatial",
        }
        for policy in compiled
    ]
    configuration: dict[str, object] = {
        "schemaVersion": "e07.s08a.qualification-portfolio.v1",
        "configurationId": "1" * 64,
        "taskId": SPATIAL_TASKS[0],
        "mode": "fixed_balanced_identity",
        "portfolioSize": 2,
        "members": members,
        "assignmentRotation": 0,
        "assignmentCounterDomain": (
            "E07/S08/fixed_balanced_identity/identity-opportunity/v1"
        ),
        "selector": None,
        "s07ArmMembershipUsed": False,
        "rejectedModelOrEmbeddingUsed": False,
    }
    return configuration, {
        policy.policy_sha256: document
        for policy, document in zip(compiled, documents, strict=True)
    }


def test_every_frozen_panel_has_an_explicit_task_local_endpoint_contract() -> None:
    records = [
        qualify_endpoint(task_id, panel_id)
        for task_id in SPATIAL_TASKS
        for panel_id in PANEL_IDS
    ]
    assert len(records) == 16
    assert all(item["passed"] for item in records)
    diagnostics = [
        item for item in records if item["endpointMode"] == "diagnostic_only"
    ]
    assert len(diagnostics) == 6
    assert all(item["completionAvailable"] is False for item in diagnostics)
    assert all(item["repairRiskSetEligible"] is False for item in records)


def test_hole_targets_require_and_receive_an_observable_exterior() -> None:
    for task_id in SPATIAL_TASKS:
        for panel_id in ("unseen_target_single_hole", "unseen_target_two_holes"):
            result = qualify_endpoint(task_id, panel_id)
            assert result["boundaryCueRequired"] is True
            assert result["boundarySignalObservable"] is True
            assert result["boundarySignaledSiteCount"] > 0
            assert result["missingBoundarySignalDenied"] is True


def test_one_swap_native_panel_is_maintenance_not_repair() -> None:
    for task_id in SPATIAL_TASKS:
        fixture = build_panel_fixture(task_id, "displacement_native_target")
        contract = endpoint_contract(task_id, "displacement_native_target")
        result = qualify_endpoint(task_id, "displacement_native_target")
        assert fixture["externalDisplacement"] == 2
        assert result["challengeConjunctionPass"] is True
        assert contract["repairRiskSetEligible"] is False
        assert (
            contract["repairUnavailableReason"]
            == "POST_SWAP_STATE_REMAINS_INSIDE_CALIBRATED_TARGET"
        )
        assert contract["oneSwapIsFault"] is False


def test_diagnostic_tracker_never_synthesizes_completion_or_repair() -> None:
    environment, reference = build_transfer_fixture("irregular_pruned_layers_9x9")
    tracker = DiagnosticSpatialTracker(environment, reference)
    tracker.observe(-1, initial_movement_state(environment), {})
    result = tracker.finalize()
    assert result["topologyCompletionCalibrated"] is False
    assert result["conjunctiveCompletionByBudget"] is None
    assert result["formationCompletion"] is None
    assert result["repairCompletion"] is None
    assert result["endpointTime"] is None
    assert result["diagnosticPromotionEligible"] is False


def test_exact_configuration_binding_preserves_bytes_and_is_order_stable() -> None:
    configuration, documents = _configuration()
    rows = [
        qualify_configuration_binding(
            configuration,
            documents,
            execution_task_id=task_id,
            panel_id=panel_id,
            selection_ref=configuration["configurationId"],
        )
        for task_id in SPATIAL_TASKS
        for panel_id in PANEL_IDS
    ]
    assert len(rows) == 16
    assert all(item["passed"] for item in rows)
    assert all(item["actionRebound"] is False for item in rows)
    assert all(item["costSemanticsRebound"] is False for item in rows)
    assert canonical_binding_digest(rows) == canonical_binding_digest(reversed(rows))


def test_adaptation_variant_revalidates_frozen_runtime_commitment() -> None:
    configuration, _documents = _configuration()
    adapted = deepcopy(configuration)
    adapted["assignmentRotation"] = 1
    variant = {
        "baseConfigurationId": configuration["configurationId"],
        "edit": {"kind": "assignment_rotation", "assignmentRotation": 1},
        "runtimeConfigurationCommitmentSha256": canonical_sha256(
            "E07/S12P/adapted-runtime-configuration/v1", adapted
        ),
    }
    assert apply_adaptation_variant(configuration, variant) == adapted
    forged = deepcopy(variant)
    forged["runtimeConfigurationCommitmentSha256"] = "0" * 64
    with pytest.raises(Exception, match="commitment"):
        apply_adaptation_variant(configuration, forged)


@pytest.mark.parametrize("failure_position", [0, 5, 10])
def test_fail_atomic_fixture_dispositions_are_complete(
    failure_position: int,
) -> None:
    item_ids = [f"fixture-{index:02d}" for index in range(11)]
    result = run_fail_atomic_fixture_batch(
        item_ids,
        lambda item_id: {"itemId": item_id, "passed": True},
        failure_position=failure_position,
    )
    assert result["precommitted"] == 11
    assert result["publicationRows"] == 0
    assert result["failed"] == 1
    assert result["attempted"] + result["notAttempted"] == 11
    assert len(result["dispositions"]) == 11


def test_fail_atomic_fixture_success_publishes_complete_set() -> None:
    item_ids = [f"fixture-{index:02d}" for index in range(11)]
    result = run_fail_atomic_fixture_batch(
        item_ids,
        lambda item_id: {"itemId": item_id, "passed": True},
        failure_position=None,
    )
    assert result["success"] is True
    assert result["publicationRows"] == 11
    assert result["failed"] == 0
