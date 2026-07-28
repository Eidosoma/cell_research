from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.environment_suite.contracts import SuiteValidationError
from src.morph2d.baseline import build_target_environment, load_baseline_assets
from src.morph2d.engine import _state_blind_actor_schedule
from src.morph2d.movements import initial_movement_state
from src.policy_dsl import compile_policy
from src.spatial_transfer.qualification import _grammar_for_target
from src.spatial_transfer.replacement import (
    FAULT_FAMILY_ID,
    SCHEDULER_FAMILY_ID,
    apply_spurious_swap_fault,
    build_replacement_random_static_comparator,
    fault_contract,
    identity_round_robin_schedule,
    scheduler_contract,
    scheduler_cost_ledger,
    validate_spurious_swap_fault_audit,
)

REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY / "configs/transfer/s12r_replacement_spatial_transfer_protocol.yaml"
)
CALIBRATED_TARGETS = (
    "stripes_alternating_three_band",
    "layers_three_ordered_tissues",
    "bilateral_lobes_with_midline",
    "tissue_single_hole",
    "tissue_two_holes",
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


def test_s12r_protocol_is_new_design_only_estimand() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["researchStepId"] == "S12R"
    assert protocol["designOnly"] is True
    assert protocol["newEstimand"] is True
    relationship = protocol["historicalRelationship"]
    assert relationship["amendsS12P"] is False
    assert relationship["amendsS12A"] is False
    assert relationship["executionOfS12PFrozenEstimand"] is False
    assert protocol["authorizationBoundary"]["transferEpisodes"] == 0
    assert protocol["executionGate"]["substantiveTransferAuthorized"] is False


def test_round_robin_scheduler_is_state_blind_sweep_and_distinct() -> None:
    actors = tuple(f"cell-{index:03d}" for index in range(81))
    scenario = "s12r-synthetic-scheduler-fixture"
    forward = [
        identity_round_robin_schedule(scenario, transition, actors)
        for transition in range(32)
    ]
    reverse = {
        transition: identity_round_robin_schedule(
            scenario, transition, tuple(reversed(actors))
        )
        for transition in reversed(range(32))
    }
    assert all(forward[index] == reverse[index] for index in range(32))
    opportunity_stream = [actor for batch in forward for actor in batch]
    assert len(set(opportunity_stream[:81])) == 81
    assert opportunity_stream[81:] == opportunity_stream[:47]
    native = [
        _state_blind_actor_schedule(scenario, transition, actors, 4)
        for transition in range(32)
    ]
    assert forward != native
    assert scheduler_contract()["schedulerFamilyId"] == SCHEDULER_FAMILY_ID
    ledger = scheduler_cost_ledger(actor_count=len(actors), transition_count=32)
    assert ledger["schedulerSelectedActorSlots"] == 128
    assert ledger["schedulerStateReads"] == 0
    assert ledger["schedulerOutcomeReads"] == 0


@pytest.mark.parametrize("target_id", CALIBRATED_TARGETS)
def test_fault_is_conserving_target_breaking_and_replayable(target_id: str) -> None:
    context, targets, _grammars, environments = load_baseline_assets()
    target = targets[target_id]
    grammar = _grammar_for_target(context, target_id)
    environment = environments.get(target_id) or build_target_environment(
        target, grammar
    )
    source = initial_movement_state(environment)
    first_state, first = apply_spurious_swap_fault(
        environment,
        source,
        target,
        grammar,
        scenario_family_id=f"s12r-fault-fixture/{target_id}",
    )
    second_state, second = apply_spurious_swap_fault(
        environment,
        source,
        target,
        grammar,
        scenario_family_id=f"s12r-fault-fixture/{target_id}",
    )
    assert first_state == second_state
    assert first == second
    assert first["faultFamilyId"] == FAULT_FAMILY_ID
    assert first["preConjunctiveCompletion"] is True
    assert first["postConjunctiveCompletion"] is False
    assert first["repairRiskSetEntered"] is True
    assert first["identityTokenKindAndSiteCardinalityConserved"] is True
    assert first["proposalGateUsed"] is False
    assert first["authorityChannelUsed"] is False
    assert first["externalLesionMaskUsed"] is False
    assert first["acceptedTargetSetDisplacementUsed"] is False
    assert validate_spurious_swap_fault_audit(
        environment, source, target, grammar, first
    )


def test_fault_audit_fails_closed_when_forged() -> None:
    context, targets, _grammars, environments = load_baseline_assets()
    target = targets["layers_three_ordered_tissues"]
    grammar = _grammar_for_target(context, target.target_id)
    environment = environments[target.target_id]
    source = initial_movement_state(environment)
    _state, audit = apply_spurious_swap_fault(
        environment,
        source,
        target,
        grammar,
        scenario_family_id="s12r-fault-forgery-fixture",
    )
    forged = deepcopy(audit)
    forged["faultLedger"]["faultGraphDisplacement"] = 1
    with pytest.raises(SuiteValidationError, match="forged"):
        validate_spurious_swap_fault_audit(environment, source, target, grammar, forged)
    contract = fault_contract()
    assert contract["explicitDistinctions"]["acceptedDisplacement"]
    assert contract["claimBoundary"].startswith("A new executable E07 extension")


def test_replacement_comparator_gets_new_executable_identity() -> None:
    documents = [_noop_document("s12r_noop_a"), _noop_document("s12r_noop_b")]
    compiled = [compile_policy(document) for document in documents]
    documents_by_hash = {
        policy.policy_sha256: document
        for policy, document in zip(compiled, documents, strict=True)
    }
    member_index = {
        policy.policy_sha256: {
            "policyId": policy.policy_id,
            "policyBodySha256": f"{index + 1:064x}",
            "nativeCarrier": "spatial",
            "boundedSemanticGroup": f"{index + 11:064x}",
        }
        for index, policy in enumerate(compiled)
    }
    definition, evidence = build_replacement_random_static_comparator(
        task_id="e07_s02_spatial2d_memory",
        member_hashes=sorted(documents_by_hash),
        member_index=member_index,
        documents_by_hash=documents_by_hash,
    )
    old_ids = {
        "d3537d6d43fede6a531120babaff28675d56caf6d5a299ab0ff99c72cb2c5074",
        "fc3a833fb46284e83db19a1943a998ab9ef97eb2bd5e2835b96eb8993d66775a",
    }
    assert definition["configurationId"] not in old_ids
    assert definition["mode"] == "random_static_identity"
    assert evidence["memberDefinitionsResolved"] == 2
    assert evidence["assignmentCount"] == 81
    assert evidence["passed"] is True
