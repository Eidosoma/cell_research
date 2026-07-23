from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pandas as pd
import pytest

from reference_simulator import Direction, Policy, create_scenario
from src.environment_suite import SuiteValidationError
from src.environment_suite.dsl_adapters import (
    bind_homogeneous_line_scenario,
    bind_portfolio_line_scenario,
    finalize_e05_result,
    run_line_dsl_episode,
    validate_e05_portfolio_result_contract,
)
from src.portfolio_preregistration.core import read_jsonl
from src.policy_dsl import compile_policy
from src.portfolio_search.execution import (
    FailAtomicBatchError,
    S08P,
    _catalog,
    build_action,
    execute_fail_atomic_batch,
    execute_work,
    publish_parquet_fail_atomic,
)


E05 = "e07_s02_regeneration_1d"
BRANCHES = (
    "development_source_terminal",
    "stabilization_source_terminal",
    "completed_panel",
)


def _e05_definition(mode: str) -> dict:
    return next(
        row
        for row in read_jsonl(S08P / "portfolio_seed_registry.jsonl")
        if row["taskId"] == E05 and row["mode"] == mode
    )


@pytest.mark.parametrize(
    "mode",
    [
        "single_policy",
        "fixed_balanced_identity",
        "random_static_identity",
        "random_dynamic_opportunity",
        "environment_conditioned",
    ],
)
def test_every_e05_result_branch_has_total_assignment_contract(mode: str):
    definition = _e05_definition(mode)
    action = build_action(definition, *_catalog())
    carrier = Policy(str(definition["members"][0]["nativeCarrier"]))
    source = create_scenario(
        tuple(range(8)),
        policy=carrier,
        direction=Direction.ASCENDING,
        seed=708_008,
        max_activations=24,
        generation_key=f"E07/S08D/qualification/{mode}",
        permute=True,
    )
    if action.portfolio_definition:
        scenario = bind_portfolio_line_scenario(source, action)
    else:
        scenario = bind_homogeneous_line_scenario(
            source, compile_policy(next(iter(action.policy_documents)))
        )
    _, runtime = run_line_dsl_episode(scenario, action, trace_mode="digest")
    for branch in BRANCHES:
        result = finalize_e05_result({"qualificationBranch": branch}, runtime)
        validation = validate_e05_portfolio_result_contract(result, action)
        assert validation["complete"]
        if action.portfolio_definition:
            assert (
                result["portfolioAssignmentAudit"]["configurationId"]
                == definition["configurationId"]
            )
        else:
            assert result["portfolioAssignmentAudit"] is None


def test_e05_portfolio_contract_rejects_missing_incomplete_and_mismatched_audit():
    definition = _e05_definition("random_dynamic_opportunity")
    action = build_action(definition, *_catalog())
    with pytest.raises(SuiteValidationError, match="omitted"):
        validate_e05_portfolio_result_contract({}, action)
    with pytest.raises(SuiteValidationError, match="incomplete"):
        validate_e05_portfolio_result_contract({"portfolioAssignmentAudit": {}}, action)


def _qualification_worker(item: dict) -> dict:
    if item.get("fail"):
        raise RuntimeError(f"injected-position-{item['position']}")
    return {"position": item["position"], "payload": item["position"] ** 2}


def _execution_success_worker(item: dict) -> dict:
    return {
        "stableEvaluationSha256": f"{item['position'] + 1:064x}",
        "position": item["position"],
    }


def _identity_qualified_work_item(position: int, *, fail: bool = False) -> dict:
    key = f"{position + 100:064x}"
    configuration_id = f"{position + 200:064x}"
    return {
        "key": key,
        "position": position,
        "fail": fail,
        "stage": "qualification",
        "generation": 0,
        "taskId": "qualification_task",
        "split": "train",
        "logicalSlotId": f"logical-{position}",
        "logicalSlotOrdinal": position,
        "reservedConfigurationSlotId": f"reservation-{position}",
        "configurationRole": "qualification",
        "pairedSlotId": None,
        "scenarioFamilyOrdinal": position,
        "smoke": False,
        "configuration": {
            "taskId": "qualification_task",
            "configurationId": configuration_id,
            "mode": "single_policy",
            "memberSetId": f"{position + 300:064x}",
            "members": [{"policySha256": f"{position + 400:064x}"}],
        },
    }


@pytest.mark.parametrize("failure_position", [0, 2, 4])
def test_fail_atomic_batch_accounts_all_positions_and_publishes_zero(
    failure_position: int,
):
    items = [
        {"position": position, "fail": position == failure_position}
        for position in range(5)
    ]
    keys = [f"key-{position}" for position in range(5)]
    with pytest.raises(FailAtomicBatchError) as caught:
        execute_fail_atomic_batch(
            keys,
            items,
            worker=_qualification_worker,
            workers=3,
            executor_factory=ThreadPoolExecutor,
        )
    accounting = caught.value.accounting
    assert accounting["newResultRowsPublished"] == 0
    assert accounting["newCacheRowsPublished"] == 0
    assert accounting["publicationAuthorized"] is False
    assert accounting["failedPhysicalRows"] == 1
    assert (
        accounting["attemptedPhysicalRows"] + accounting["cancelledPhysicalRows"] == 5
    )
    assert {row["position"] for row in accounting["positions"]} == set(range(5))
    assert all(row["status"] != "submitted" for row in accounting["positions"])


def test_fail_atomic_batch_is_worker_order_independent():
    items = [{"position": position, "fail": False} for position in range(12)]
    keys = [f"key-{position:02d}" for position in range(12)]
    forward, forward_accounting = execute_fail_atomic_batch(
        keys,
        items,
        worker=_qualification_worker,
        workers=4,
        executor_factory=ThreadPoolExecutor,
    )
    reverse, reverse_accounting = execute_fail_atomic_batch(
        list(reversed(keys)),
        list(reversed(items)),
        worker=_qualification_worker,
        workers=2,
        executor_factory=ThreadPoolExecutor,
    )
    assert forward == reverse
    assert (
        forward_accounting["resultCommitmentSha256"]
        == reverse_accounting["resultCommitmentSha256"]
    )


def test_execute_work_writes_forensics_but_no_cache_rows_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from src.portfolio_search import execution

    monkeypatch.setattr(execution, "_physical_key", lambda item: item["key"])
    monkeypatch.setattr(execution, "evaluate_work_item", _qualification_worker)
    work = [
        _identity_qualified_work_item(position, fail=position == 2)
        for position in range(5)
    ]
    cache = tmp_path / "cache"
    forensics = tmp_path / "forensics.json"
    with pytest.raises(FailAtomicBatchError):
        execute_work(
            work,
            workers=3,
            cache_dir=cache,
            failure_accounting_path=forensics,
            executor_factory=ThreadPoolExecutor,
            atomic_new_cache=True,
        )
    assert forensics.is_file()
    accounting = json.loads(forensics.read_text())
    assert accounting["newResultRowsPublished"] == 0
    assert accounting["newCacheRowsPublished"] == 0
    assert not cache.exists()


def test_execute_work_commits_complete_cache_directory_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from src.portfolio_search import execution

    monkeypatch.setattr(execution, "_physical_key", lambda item: item["key"])
    monkeypatch.setattr(execution, "evaluate_work_item", _execution_success_worker)
    work = [_identity_qualified_work_item(position) for position in range(5)]
    cache = tmp_path / "committed-cache"
    rows, accounting = execute_work(
        work,
        workers=3,
        cache_dir=cache,
        executor_factory=ThreadPoolExecutor,
        atomic_new_cache=True,
    )
    assert len(rows) == 5
    assert accounting["atomicNewCache"] is True
    assert accounting["batchAccounting"]["success"] is True
    assert len(list(cache.glob("*.json"))) == 5
    assert not list(tmp_path.glob(".*.s08d-staging-*"))


def test_parquet_publication_is_atomic_and_never_replaces_existing(tmp_path: Path):
    target = tmp_path / "smoke.parquet"
    publish_parquet_fail_atomic(pd.DataFrame({"row": [1, 2, 3]}), target)
    assert pd.read_parquet(target)["row"].tolist() == [1, 2, 3]
    with pytest.raises(FileExistsError):
        publish_parquet_fail_atomic(pd.DataFrame({"row": [4]}), target)
    assert pd.read_parquet(target)["row"].tolist() == [1, 2, 3]
    assert not list(tmp_path.glob("*.staging"))
