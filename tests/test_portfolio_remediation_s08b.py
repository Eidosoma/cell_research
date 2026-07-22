from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.environment_suite.contracts import SuiteValidationError
from src.environment_suite.portfolio_adapters import PortfolioDispatcher
from src.policy_dsl import compile_policy
from src.portfolio_search.preflight import S05, S08P, read_jsonl


def _affected_definition(mode: str) -> tuple[dict, dict]:
    blocked = json.loads(
        Path(
            "/artifacts/research_steps/S08/configuration_binding_validation.json"
        ).read_text()
    )
    affected_ids = {row["configurationId"] for row in blocked["errors"]}
    definitions = read_jsonl(S08P / "portfolio_seed_registry.jsonl")
    definition = next(
        row
        for row in definitions
        if row["configurationId"] in affected_ids and row["mode"] == mode
    )
    catalog = {
        row["policySha256"]: row["document"]
        for row in read_jsonl(S05 / "policy_catalog.jsonl")
    }
    policies = {
        member["policyId"]: compile_policy(catalog[member["policySha256"]])
        for member in definition["members"]
    }
    return definition, policies


@pytest.mark.parametrize(
    "mode",
    [
        "fixed_balanced_identity",
        "random_static_identity",
        "random_dynamic_opportunity",
        "environment_conditioned",
    ],
)
def test_singleton_carrier_dispatch_is_deterministic_and_native(mode: str):
    definition, policies = _affected_definition(mode)
    dispatcher = PortfolioDispatcher(definition, policies)
    singleton_carrier, policy_hash = next(
        iter(dispatcher.singleton_members_by_carrier.items())
    )
    dispatcher.register_identities(
        {carrier: (f"{carrier}-0",) for carrier in dispatcher.members_by_carrier}
    )
    actor = f"{singleton_carrier}-0"
    selected = {
        dispatcher.select(
            singleton_carrier,
            actor,
            opportunity,
            selector_value=opportunity % 2 == 1,
            mutate=True,
        ).policy_sha256
        for opportunity in range(6)
    }
    assert selected == {policy_hash}
    ledger = dispatcher.ledger_snapshot()
    assert ledger["singletonCarrierPolicySha256"][singleton_carrier] == policy_hash
    assert ledger["portfolioCoordinationLedger"]["memberAssignments"] == 6
    assert ledger["portfolioCoordinationLedger"]["memberSwitches"] == 0
    if mode in {"random_static_identity", "random_dynamic_opportunity"}:
        assert ledger["portfolioCoordinationLedger"]["selectorCounterReads"] == 0
    if mode == "environment_conditioned":
        assert ledger["portfolioCoordinationLedger"]["selectorReads"] == 6
        assert ledger["portfolioCoordinationLedger"]["selectorDecisions"] == 6
    with pytest.raises(SuiteValidationError, match="lacks the native carrier"):
        dispatcher.select("not-a-native-carrier", actor, 7, mutate=False)


def test_singleton_remediation_does_not_permit_one_member_portfolio():
    definition, policies = _affected_definition("fixed_balanced_identity")
    singleton_carrier = next(
        carrier
        for carrier in {member["nativeCarrier"] for member in definition["members"]}
        if sum(member["nativeCarrier"] == carrier for member in definition["members"])
        == 1
    )
    member = next(
        member
        for member in definition["members"]
        if member["nativeCarrier"] == singleton_carrier
    )
    narrowed = deepcopy(definition)
    narrowed["members"] = [member]
    narrowed["portfolioSize"] = 1
    with pytest.raises(
        SuiteValidationError, match="members must exactly match action policies"
    ):
        PortfolioDispatcher(
            narrowed, {member["policyId"]: policies[member["policyId"]]}
        )


def test_conditioned_singleton_still_enforces_frozen_signal_contract():
    definition, policies = _affected_definition("environment_conditioned")
    invalid = deepcopy(definition)
    invalid["selector"]["signal"] = "native.outcome"
    with pytest.raises(SuiteValidationError, match="not authorized"):
        PortfolioDispatcher(invalid, policies)
