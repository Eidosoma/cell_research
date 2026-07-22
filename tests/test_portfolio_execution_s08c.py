from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from src.environment_suite import Split, SuiteValidationError
from src.environment_suite.portfolio_adapters import PortfolioDispatcher
from src.environment_suite.runners import _validate_action, locked_dsl_validation_scope
from src.policy_dsl import compile_policy
from src.portfolio_preregistration.core import read_jsonl
from src.portfolio_search.execution import (
    S08P,
    TASK_IDS,
    _base_records,
    _catalog,
    build_action,
    derive_record,
    scenario_commitment,
)


def _initial_definitions():
    return read_jsonl(S08P / "portfolio_seed_registry.jsonl")


def test_frozen_budget_has_exact_smoke_generation_and_task_accounting():
    budget = pd.read_parquet(S08P / "budget_slot_ledger.parquet")
    assert len(budget) == 11008
    assert int(budget["smoke"].sum()) == 40
    assert budget.groupby("generation").size().to_dict() == {
        0: 4864,
        1: 1024,
        2: 1024,
        3: 1024,
        4: 1024,
        5: 1024,
        6: 1024,
    }
    assert set(budget.groupby("taskId").size()) == {1376}


def test_s08c_record_derivation_never_crosses_split_and_discloses_e05_reuse():
    _, train, validation = _base_records()
    for task_id in TASK_IDS:
        train_records = [
            derive_record(train[task_id], family, split=Split.TRAIN)
            for family in (300, 301, 302, 303)
        ]
        validation_records = [
            derive_record(validation[task_id], family, split=Split.VALIDATION)
            for family in range(400, 408)
        ]
        assert all(
            row.split is Split.TRAIN and not row.protected for row in train_records
        )
        assert all(
            row.split is Split.VALIDATION and not row.protected
            for row in validation_records
        )
        assert len({row.scenario_id for row in train_records}) == 4
        assert len({row.scenario_id for row in validation_records}) == 8
        assert all(len(scenario_commitment(row)) == 64 for row in validation_records)
    regeneration = [
        derive_record(
            validation["e07_s02_regeneration_1d"], family, split=Split.VALIDATION
        )
        for family in range(400, 408)
    ]
    assert [row.public_parameters["replicateOrdinal"] for row in regeneration] == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]


def test_validation_dsl_action_requires_narrow_locked_scope():
    definition = next(
        row
        for row in _initial_definitions()
        if row["taskId"] == "e07_s02_sorting_1d" and row["mode"] == "single_policy"
    )
    action = build_action(definition, *_catalog())
    _, _, validation = _base_records()
    record = derive_record(
        validation["e07_s02_sorting_1d"], 400, split=Split.VALIDATION
    )
    with pytest.raises(SuiteValidationError):
        _validate_action(record, action)
    with locked_dsl_validation_scope(record, action, "0" * 64):
        _validate_action(record, action)
    with pytest.raises(SuiteValidationError):
        _validate_action(record, action)


def test_adaptive_dispatch_accepts_only_frozen_rotation_and_branch_flip():
    definition = next(
        copy.deepcopy(row)
        for row in _initial_definitions()
        if row["taskId"] == "e07_s02_sorting_1d"
        and row["mode"] == "environment_conditioned"
    )
    definition["schemaVersion"] = "e07.s08c.adaptive-portfolio.v1"
    definition["researchStepId"] = "S08C"
    definition["assignmentRotation"] = 1
    definition["selector"]["trueBranch"], definition["selector"]["falseBranch"] = (
        definition["selector"]["falseBranch"],
        definition["selector"]["trueBranch"],
    )
    by_hash, _ = _catalog()
    policies = {
        compile_policy(by_hash[row["policySha256"]]).policy_id: compile_policy(
            by_hash[row["policySha256"]]
        )
        for row in definition["members"]
    }
    dispatcher = PortfolioDispatcher(definition, policies)
    assert dispatcher.assignment_rotation == 1
    bad = copy.deepcopy(definition)
    bad["selector"]["trueBranch"] = "arbitrary_member"
    with pytest.raises(SuiteValidationError):
        PortfolioDispatcher(bad, policies)


def test_historical_artifact_steps_do_not_contain_s08c_outputs():
    for step in ("S08P", "S08A", "S08", "S08B"):
        root = Path("/artifacts/research_steps") / step
        assert root.is_dir()
        assert not any("s08c" in path.name.lower() for path in root.iterdir())
