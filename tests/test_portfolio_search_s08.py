from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.portfolio_search.preflight import (
    S08P,
    read_jsonl,
    validate_executable_bindings,
)


def _binding_result():
    configurations = read_jsonl(S08P / "portfolio_seed_registry.jsonl")
    budget = pd.read_parquet(S08P / "budget_slot_ledger.parquet")
    smoke_ids = set(budget.loc[budget["smoke"], "configurationId"].astype(str))
    return validate_executable_bindings(configurations, smoke_ids)


def test_full_frozen_registry_exposes_unqualified_chimera_carrier_cardinality():
    result = _binding_result()
    assert not result["success"]
    assert result["configurationRowsChecked"] == 1216
    assert result["configurationRowsBound"] == 1160
    assert result["configurationRowsBlocked"] == 56
    assert result["blockedByTask"] == {"e07_s02_chimera_1d": 56}
    assert result["blockedByMode"] == {
        "environment_conditioned": 8,
        "fixed_balanced_identity": 16,
        "random_dynamic_opportunity": 16,
        "random_static_identity": 16,
    }
    assert {row["error"] for row in result["errors"]} == {
        "qualification portfolios require at least two members per carrier"
    }


def test_three_frozen_smoke_configurations_are_blocked_before_evaluation():
    result = _binding_result()
    assert result["frozenSmokeConfigurationsBlocked"] == 3
    assert result["frozenSmokeConfigurationIdsBlocked"] == [
        "0056c4f5bfca13099a5a58acbae6de46330c062e7d2cd366244c10f9bee454de",
        "0369864632113c4f231a7b675e073a0fcab15ef5262aa3e82e2cc7dafe45e653",
        "0de09b1a76e05b5d5f2dd5a2ec1a380093626666f6ef60f0dc751c034925d368",
    ]
    assert result["scenarioMaterializations"] == 0
    assert result["episodeEvaluations"] == 0


def test_s08a_gate_historically_passed_but_did_not_cover_frozen_cardinalities():
    gate = json.loads(
        Path(
            "/artifacts/research_steps/S08A/s08_execution_eligibility_gate.json"
        ).read_text()
    )
    assert gate["technicalEligibilityPass"]
    assert all(row["status"] == "pass" for row in gate["rows"])
    fixtures = read_jsonl(
        "/artifacts/research_steps/S08A/qualification_fixture_registry.jsonl"
    )
    chimera = next(row for row in fixtures if row["taskId"] == "e07_s02_chimera_1d")
    per_carrier = {}
    for member in chimera["members"]:
        per_carrier[member["nativeCarrier"]] = (
            per_carrier.get(member["nativeCarrier"], 0) + 1
        )
    assert per_carrier == {"Bubble": 2, "Insertion": 2}
