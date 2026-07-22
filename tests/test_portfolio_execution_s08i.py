from __future__ import annotations

from pathlib import Path

import yaml

from scripts.execute_portfolio_search_s08g import (
    s08h_contract_revalidation,
    s08h_runtime_integrity_audit,
)
from src.portfolio_search.execution import run_preflight


CONTROL = Path("configs/portfolio/s08i_execution_continuation.yaml")


def test_s08i_control_is_fresh_exact_and_quarantine_excluding() -> None:
    control = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))

    assert control["researchStepId"] == "S08I"
    assert control["prospectivePreregistration"] is True
    assert control["historicalDesignMayChange"] is False
    assert control["freshCacheRoot"] == "/cache/e07-s08i"
    assert control["quarantinedCacheRoots"] == [
        "/cache/e07-s08c",
        "/cache/e07-s08e",
        "/cache/e07-s08g",
    ]
    assert control["frozenExecution"]["smokeLogicalRows"] == 40
    assert control["frozenExecution"]["initialUniquePhysicalRows"] == 4864
    assert control["frozenExecution"]["trainingLogicalRows"] == 11008
    assert control["frozenExecution"]["adaptiveGenerations"] == 6
    assert control["s08hContracts"]["rawFiniteDomain"] == [-495, 1]
    assert control["s08hContracts"]["zeroInitialDistance"] == (
        "explicitly_unavailable"
    )
    assert control["s08hContracts"]["hitRequiresExactPostHitProbe"] is True
    assert control["validationBoundary"]["confirmationLogicalRows"] == 0


def test_s08i_preflight_clears_all_frozen_gates() -> None:
    result = run_preflight(CONTROL)

    assert result["researchStepId"] == "S08I"
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


def test_s08h_freeze_and_runtime_state_machine_are_enforced() -> None:
    frozen = s08h_contract_revalidation()
    assert frozen["success"] is True
    assert all(frozen["checks"].values())
    assert frozen["repairRawFiniteDomain"] == [-495, 1]

    valid = {
        "taskId": "e07_s02_target_change_1d",
        "logicalSlotId": "slot:valid",
        "configurationId": "configuration:valid",
        "scenarioFamilyOrdinal": 300,
        "stopReason": "phase_event_budget",
        "outcome": {
            "targetChangeSemanticsVersion": (
                "e07.s08h.e05-target-deadline-probe.v1"
            ),
            "targetCompleted": False,
            "adaptationTime": None,
            "postHitProbeRetained": True,
            "postHitProbeOpportunities": 0,
            "targetChangeSemanticAudit": {
                "validNativeContract": True,
                "adapterFailure": False,
            },
        },
    }
    assert s08h_runtime_integrity_audit([valid])["success"] is True

    invalid = {
        **valid,
        "logicalSlotId": "slot:invalid",
        "outcome": {
            **valid["outcome"],
            "targetCompleted": True,
            "adaptationTime": 6400,
            "postHitProbeRetained": False,
            "targetChangeSemanticAudit": {
                "validNativeContract": False,
                "adapterFailure": True,
            },
        },
    }
    audit = s08h_runtime_integrity_audit([invalid])
    assert audit["success"] is False
    assert "hit_without_exact_post_hit_probe" in audit["errorRows"][0]["errors"]


def test_s08i_integrity_checks_precede_publication_and_archive() -> None:
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
