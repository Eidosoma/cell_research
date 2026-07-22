from __future__ import annotations

from pathlib import Path

import yaml

from scripts.execute_portfolio_search_s08g import (
    s08f_contract_revalidation,
    s08h_contract_revalidation,
    s08j_contract_revalidation,
)
from src.portfolio_search.execution import run_preflight


CONTROL = Path("configs/portfolio/s08k_execution_continuation.yaml")


def test_s08k_control_is_fresh_exact_and_quarantine_excluding() -> None:
    control = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))

    assert control["researchStepId"] == "S08K"
    assert control["prospectivePreregistration"] is True
    assert control["historicalDesignMayChange"] is False
    assert control["freshCacheRoot"] == "/cache/e07-s08k"
    assert control["quarantinedCacheRoots"] == [
        "/cache/e07-s08c",
        "/cache/e07-s08e",
        "/cache/e07-s08g",
        "/cache/e07-s08i",
    ]
    assert control["reuseS08cOutcomes"] is False
    assert control["reuseS08eOutcomes"] is False
    assert control["reuseS08gOutcomes"] is False
    assert control["reuseS08iOutcomes"] is False
    assert control["frozenExecution"]["smokeLogicalRows"] == 40
    assert control["frozenExecution"]["initialUniquePhysicalRows"] == 4864
    assert control["frozenExecution"]["trainingLogicalRows"] == 11008
    assert control["frozenExecution"]["adaptiveGenerations"] == 6
    assert control["validationBoundary"]["confirmationLogicalRows"] == 0
    assert control["s08jContracts"]["persistedPlane"] == (
        "nativeEvent.targetChangeAuditProjection"
    )


def test_s08k_preflight_clears_all_frozen_gates() -> None:
    result = run_preflight(CONTROL)

    assert result["researchStepId"] == "S08K"
    assert result["success"] is True
    assert result["blockedGateIds"] == []
    assert [row["gateId"] for row in result["gateRows"]] == [
        "G01",
        "G02",
        "G03",
        "G04",
        "G05",
        "G06",
    ]
    assert result["bindingValidation"]["configurationRowsBound"] == 1216
    assert result["bindingValidation"]["frozenSmokeConfigurationsBlocked"] == 0
    assert len(result["accessRows"]) == 16
    assert all(row["deniedBeforeMaterialization"] for row in result["accessRows"])
    assert result["loadedProhibitedModules"] == []
    assert result["noMutation"] is True


def test_s08k_revalidates_s08f_s08h_and_s08j_contracts() -> None:
    s08f = s08f_contract_revalidation()
    s08h = s08h_contract_revalidation()
    s08j = s08j_contract_revalidation()

    assert s08f["success"] is True
    assert s08h["success"] is True
    assert s08j["success"] is True
    assert all(s08j["checks"].values())
    assert s08j["persistedPlane"] == "nativeEvent.targetChangeAuditProjection"


def test_projection_integrity_precedes_smoke_publication_and_archives() -> None:
    source = Path("scripts/execute_portfolio_search_s08g.py").read_text(
        encoding="utf-8"
    )
    smoke_audit = source.index(
        "smoke_runtime_integrity = s08h_runtime_integrity_audit(smoke_rows)"
    )
    smoke_publication = source.index(
        "publish_parquet_fail_atomic(\n"
        '        ledger_frame(smoke_rows), output / "frozen_smoke_results.parquet"'
    )
    initial_audit = source.index(
        "initial_runtime_integrity = s08h_runtime_integrity_audit(training_rows)"
    )
    first_aggregate = source.index(
        "initial_aggregates = aggregate_panel(definitions, training_rows)"
    )
    assert smoke_audit < smoke_publication
    assert initial_audit < first_aggregate
