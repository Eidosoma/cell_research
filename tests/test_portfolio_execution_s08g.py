from __future__ import annotations

from pathlib import Path

import yaml

from scripts.execute_portfolio_search_s08g import (
    IMMUTABLE_STEPS,
    S08G_CACHE_ROOT,
    S08G_CONTROL_PATH,
    s08f_contract_revalidation,
)
from src.portfolio_search.execution import run_preflight


def test_s08g_control_preserves_frozen_design_and_uses_fresh_namespace() -> None:
    control = yaml.safe_load(S08G_CONTROL_PATH.read_text(encoding="utf-8"))

    assert control["researchStepId"] == "S08G"
    assert control["historicalDesignMayChange"] is False
    assert control["freshCacheRoot"] == "/cache/e07-s08g"
    assert control["reuseS08cOutcomes"] is False
    assert control["reuseS08eOutcomes"] is False
    assert S08G_CACHE_ROOT == Path("/cache/e07-s08g")
    assert tuple(control["immutableSteps"]) == IMMUTABLE_STEPS
    assert control["frozenExecution"]["trainingLogicalRows"] == 11008
    assert control["frozenExecution"]["adaptiveGenerations"] == 6
    assert control["frozenExecution"]["initialUniquePhysicalRows"] == 4864
    assert control["analysis"]["availabilityAsNovelty"] == "forbidden"
    assert control["smokePublication"]["onFailurePublishedSmokeRows"] == 0
    assert control["validationBoundary"]["confirmationLogicalRows"] == 0


def test_completed_s08g_preflight_rejects_postexecution_plan_update_only() -> None:
    result = run_preflight(S08G_CONTROL_PATH)

    assert result["researchStepId"] == "S08G"
    assert result["success"] is False
    assert result["status"] == "blocked_before_smoke"
    assert result["blockedGateIds"] == ["G01"]
    assert [row["gateId"] for row in result["gateRows"]] == [
        "G01",
        "G02",
        "G03",
        "G04",
        "G05",
        "G06",
    ]
    assert result["gateRows"][0]["status"] == "blocked"
    assert all(row["status"] == "pass" for row in result["gateRows"][1:])
    failed_files = [row for row in result["fileChecks"] if not row["pass"]]
    assert len(failed_files) == 1
    assert failed_files[0]["path"] == "/workspace/RESEARCH_PLAN.md"
    assert (
        failed_files[0]["expectedSha256"]
        == "defda3af4cb8c88ca345e8068be486bac929ce704a44290af1eee1914dbf7c5c"
    )
    assert result["qualificationGatePath"].endswith(
        "/S08F/s08_execution_eligibility_gate.json"
    )
    assert result["qualificationGatePass"] is True
    assert result["failAtomicEvidencePass"] is True
    assert result["bindingValidation"]["configurationRowsBound"] == 1216
    assert result["bindingValidation"]["frozenSmokeConfigurationsBlocked"] == 0
    assert len(result["accessRows"]) == 16
    assert all(row["deniedBeforeMaterialization"] for row in result["accessRows"])
    assert result["loadedProhibitedModules"] == []
    assert result["noMutation"] is True


def test_s08g_revalidates_s08f_support_equivalence_and_identity_contracts() -> None:
    result = s08f_contract_revalidation()

    assert result["success"] is True
    assert all(result["checks"].values())
    assert (
        result["descriptorAvailabilityVersion"]
        == "e07.s08f.e05-descriptor-availability.v1"
    )


def test_s08g_smoke_publishes_only_after_integrity_validation() -> None:
    source = Path("scripts/execute_portfolio_search_s08g.py").read_text(
        encoding="utf-8"
    )

    validation = source.index("smoke_validation = validate_rows(smoke_rows, 40)")
    rejection = source.index('if not smoke_validation["success"]:')
    publication = source.index(
        "publish_parquet_fail_atomic(\n"
        '        ledger_frame(smoke_rows), output / "frozen_smoke_results.parquet"'
    )
    assert validation < rejection < publication
    assert 'cache_dir=S08G_CACHE_ROOT / "evaluations"' in source
    assert "atomic_new_cache=True" in source
    assert "partial_smoke_results.parquet" not in source
    assert "/cache/e07-s08c" not in source
    assert "/cache/e07-s08e" not in source
