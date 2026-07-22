from __future__ import annotations

from pathlib import Path

import yaml

from scripts.execute_portfolio_search_s08g import (
    s08h_contract_revalidation,
    s08h_runtime_integrity_audit,
)
from src.environment_suite.e05_semantics import (
    TARGET_CHANGE_SEMANTICS_VERSION,
    build_target_change_audit_projection,
    validate_target_change_result_semantics,
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
    assert control["s08hContracts"]["zeroInitialDistance"] == ("explicitly_unavailable")
    assert control["s08hContracts"]["hitRequiresExactPostHitProbe"] is True
    assert control["validationBoundary"]["confirmationLogicalRows"] == 0


def test_s08i_historical_preflight_fails_closed_after_s08j_code_change() -> None:
    result = run_preflight(CONTROL)

    assert result["researchStepId"] == "S08I"
    assert result["success"] is False
    assert result["blockedGateIds"] == ["G01"]
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

    source = {
        "targetChangeSemanticsVersion": TARGET_CHANGE_SEMANTICS_VERSION,
        "stopReason": "phase_event_budget",
        "targetCompleted": False,
        "phaseActivationCount": 6400,
        "adaptationTime": None,
        "adaptationCensored": True,
        "overshootCensored": True,
        "postHitProbeOpportunities": 0,
        "postHitProbeRetained": True,
        "postHitProbeApplicable": False,
    }
    source["targetChangeSemanticAudit"] = validate_target_change_result_semantics(
        source, adaptation_budget=6400, probe_budget=160
    )
    projection = build_target_change_audit_projection(
        source,
        adaptation_budget=6400,
        probe_budget=160,
        source_result_sha256="a" * 64,
    )
    valid = {
        "taskId": "e07_s02_target_change_1d",
        "logicalSlotId": "slot:valid",
        "configurationId": "configuration:valid",
        "scenarioFamilyOrdinal": 300,
        "stopReason": "phase_event_budget",
        "outcome": {
            key: source[key]
            for key in (
                "targetCompleted",
                "phaseActivationCount",
                "adaptationTime",
                "adaptationCensored",
                "overshootCensored",
            )
        },
        "nativeEvent": {
            "resultSha256": "a" * 64,
            "targetChangeAuditProjection": projection,
        },
        "validation": {
            "targetChangeSemanticContract": True,
            "postHitProbeRetained": True,
        },
    }
    assert s08h_runtime_integrity_audit([valid])["success"] is True

    invalid = {
        **valid,
        "logicalSlotId": "slot:invalid",
        "outcome": {**valid["outcome"], "targetCompleted": True},
    }
    audit = s08h_runtime_integrity_audit([invalid])
    assert audit["success"] is False
    assert (
        "persisted_endpoint_mismatch:targetCompleted" in audit["errorRows"][0]["errors"]
    )


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
