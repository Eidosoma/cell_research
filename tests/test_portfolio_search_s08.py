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


def test_full_frozen_registry_binds_after_s08b_singleton_remediation():
    result = _binding_result()
    assert result["success"]
    assert result["configurationRowsChecked"] == 1216
    assert result["configurationRowsBound"] == 1216
    assert result["configurationRowsBlocked"] == 0
    assert result["blockedByTask"] == {}
    assert result["blockedByMode"] == {}
    assert result["errors"] == []


def test_all_frozen_smoke_configurations_bind_without_evaluation():
    result = _binding_result()
    assert result["frozenSmokeConfigurationsBlocked"] == 0
    assert result["frozenSmokeConfigurationIdsBlocked"] == []
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

    blocked = json.loads(
        Path(
            "/artifacts/research_steps/S08/configuration_binding_validation.json"
        ).read_text()
    )
    assert blocked["configurationRowsBlocked"] == 56
    assert blocked["frozenSmokeConfigurationsBlocked"] == 3
