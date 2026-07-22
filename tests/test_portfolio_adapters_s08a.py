from __future__ import annotations

import copy
import hashlib

import pytest

from reference_simulator import Direction, Policy, create_scenario
from src.environment_suite import (
    AccessDeniedError,
    AccessGrant,
    AccessPhase,
    EnvironmentSuite,
    SuiteValidationError,
    portfolio_action,
)
from src.environment_suite.dsl_adapters import (
    bind_portfolio_line_scenario,
    line_replay_bytes,
    run_line_dsl_episode,
)
from src.environment_suite.portfolio_adapters import PortfolioDispatcher
from src.policy_dsl import compile_policy


def _line_member(policy_id: str, *, aggressive: bool) -> dict:
    terminal = (
        {"kind": "swap_relative", "offset": -1} if aggressive else {"kind": "noop"}
    )
    return {
        "schemaVersion": "e07.policy-dsl.v1",
        "policyId": policy_id,
        "environment": "line1d.v1",
        "permissions": [
            "activation.side",
            "last_action.rejected",
            "signal.neighbor_sum_u8",
        ],
        "memory": [{"name": "state", "bits": 1, "initial": 0}],
        "signals": {"channels": 1, "bitsPerChannel": 2},
        "rules": [
            {
                "when": {
                    "left": {"obs": "activation.side"},
                    "op": "eq",
                    "right": {"const": "left"},
                },
                "actions": [
                    {
                        "kind": "set_memory",
                        "register": "state",
                        "mode": "toggle",
                        "value": None,
                    },
                    {"kind": "emit_signal", "channel": 0, "value": {"const": 1}},
                    terminal,
                ],
            }
        ],
        "default": {
            "actions": [
                {
                    "kind": "set_memory",
                    "register": "state",
                    "mode": "toggle",
                    "value": None,
                },
                {"kind": "emit_signal", "channel": 0, "value": {"const": 2}},
                {"kind": "noop"},
            ]
        },
        "limits": {
            "maxRules": 1,
            "maxExpressionNodes": 3,
            "maxActionsPerActivation": 3,
            "maxOperationsPerActivation": 32,
            "maxMovementRadius": 1 if aggressive else 0,
            "maxCandidates": 0,
        },
    }


def _definition(documents, *, task_id="e07_s02_sorting_1d"):
    policies = [compile_policy(item) for item in documents]
    selector = {
        "profile": "bounded_memoryless_two_branch_v1",
        "signal": "last_action.rejected",
        "operator": "eq",
        "threshold": True,
        "falseBranch": "current_or_first_compatible_member",
        "trueBranch": "next_compatible_member",
        "persistentMemoryBits": 0,
    }
    base = {
        "schemaVersion": "e07.s08a.qualification-portfolio.v1",
        "taskId": task_id,
        "configurationId": hashlib.sha256(task_id.encode()).hexdigest(),
        "mode": "environment_conditioned",
        "assignmentCounterDomain": "E07/S08/environment_conditioned/identity-opportunity/v1",
        "selector": selector,
        "members": [
            {
                "policyId": item.policy_id,
                "policySha256": item.policy_sha256,
                "nativeCarrier": "Bubble",
            }
            for item in policies
        ],
        "portfolioSize": len(policies),
        "s07ArmMembershipUsed": False,
        "rejectedModelOrEmbeddingUsed": False,
    }
    dispatcher = PortfolioDispatcher(base, {item.policy_id: item for item in policies})
    base["portfolioStructuralCosts"] = dispatcher.structural
    return base


def test_identity_switching_memory_communication_and_replay_are_auditable():
    documents = [
        _line_member("s08a-test-a", aggressive=True),
        _line_member("s08a-test-b", aggressive=False),
    ]
    action = portfolio_action(documents, _definition(documents))
    source = create_scenario(
        (0, 1, 2, 3),
        policy=Policy.BUBBLE,
        direction=Direction.ASCENDING,
        seed=4451,
        max_activations=256,
        generation_key="S08A/test/line-portfolio",
        permute=True,
    )
    scenario = bind_portfolio_line_scenario(source, action)
    first, first_runtime = run_line_dsl_episode(scenario, action)
    second, second_runtime = run_line_dsl_episode(scenario, action)
    assert line_replay_bytes(first, first_runtime) == line_replay_bytes(
        second, second_runtime
    )
    ledgers = first_runtime.adapter_ledgers()
    assert ledgers["portfolioCoordinationLedger"]["memberAssignments"] > 0
    assert ledgers["portfolioCoordinationLedger"]["memberSwitches"] > 0
    assert ledgers["portfolioCoordinationLedger"]["memberNamespaceInitializations"] > 0
    assert ledgers["portfolioCoordinationLedger"]["emittedSignalWrites"] > 0
    assert ledgers["portfolioCoordinationLedger"]["recipientDeliveries"] > 0
    assert set(ledgers) >= {
        "portfolioStructuralLedger",
        "portfolioCoordinationLedger",
        "licensedCapabilityLedger",
        "dslRuntimeLedger",
        "dslCommunicationLedger",
    }
    audit = first_runtime.portfolio_assignment_audit()
    assert (
        audit["memberMemoryNamespaceCount"]
        == ledgers["portfolioCoordinationLedger"]["memberNamespaceInitializations"]
    )


def test_selector_overreach_and_rejected_dependencies_fail_closed():
    documents = [
        _line_member("s08a-test-c", aggressive=True),
        _line_member("s08a-test-d", aggressive=False),
    ]
    definition = _definition(documents)
    compiled = {item.policy_id: item for item in map(compile_policy, documents)}
    bad_signal = copy.deepcopy(definition)
    bad_signal["selector"]["signal"] = "scenario.id"
    with pytest.raises(SuiteValidationError):
        PortfolioDispatcher(bad_signal, compiled)
    bad_dependency = copy.deepcopy(definition)
    bad_dependency["rejectedModelOrEmbeddingUsed"] = True
    with pytest.raises(SuiteValidationError):
        PortfolioDispatcher(bad_dependency, compiled)
    bad_arm = copy.deepcopy(definition)
    bad_arm["s07ArmMembershipUsed"] = True
    with pytest.raises(SuiteValidationError):
        PortfolioDispatcher(bad_arm, compiled)


def test_portfolio_action_is_denied_on_nontraining_split_before_materialization():
    documents = [
        _line_member("s08a-test-e", aggressive=True),
        _line_member("s08a-test-f", aggressive=False),
    ]
    action = portfolio_action(documents, _definition(documents))
    suite = EnvironmentSuite(
        "configs/environment_suite/task_registry.yaml",
        "configs/environment_suite/split_manifest.json",
    )
    before = suite.broker.audit.to_dict()
    with pytest.raises(AccessDeniedError):
        suite.open_for_action(
            "e07_s02_sorting_1d",
            "e07s02:sorting:validation:000",
            AccessGrant(AccessPhase.VALIDATION),
            action,
        )
    with pytest.raises(AccessDeniedError):
        suite.open_for_action(
            "e07_s02_sorting_1d",
            "e07s02:sorting:confirmation:opaque",
            AccessGrant(AccessPhase.CONFIRMATION, "0" * 64),
            action,
        )
    after = suite.broker.audit.to_dict()
    assert after["scenarioDenials"] == before["scenarioDenials"] + 2
    assert after["materializerInvocations"] == before["materializerInvocations"]
