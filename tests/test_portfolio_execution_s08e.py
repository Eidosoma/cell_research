from __future__ import annotations

from pathlib import Path

import yaml

from scripts.execute_portfolio_search_s08e import (
    IMMUTABLE_STEPS,
    S08E_CACHE_ROOT,
    S08E_CONTROL_PATH,
)
from src.portfolio_search.execution import run_preflight


def test_s08e_control_preserves_frozen_design_and_uses_fresh_namespace() -> None:
    control = yaml.safe_load(S08E_CONTROL_PATH.read_text(encoding="utf-8"))

    assert control["researchStepId"] == "S08E"
    assert control["historicalDesignMayChange"] is False
    assert control["freshCacheRoot"] == "/cache/e07-s08e"
    assert control["reuseS08cOutcomes"] is False
    assert S08E_CACHE_ROOT == Path("/cache/e07-s08e")
    assert tuple(control["immutableSteps"]) == IMMUTABLE_STEPS
    assert control["frozenExecution"]["trainingLogicalRows"] == 11008
    assert control["frozenExecution"]["adaptiveGenerations"] == 6
    assert control["smokePublication"]["onFailurePublishedSmokeRows"] == 0
    assert control["validationBoundary"]["confirmationLogicalRows"] == 0


def test_historical_s08e_preflight_detects_post_s08e_s08f_source_change() -> None:
    result = run_preflight(S08E_CONTROL_PATH)

    assert result["researchStepId"] == "S08E"
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
    assert result["gateRows"][0]["status"] == "blocked"
    assert all(row["status"] == "pass" for row in result["gateRows"][1:])
    assert result["qualificationGatePath"].endswith(
        "/S08D/s08_execution_eligibility_gate.json"
    )
    assert result["qualificationGatePass"] is True
    assert result["failAtomicEvidencePass"] is True
    assert result["bindingValidation"]["configurationRowsBound"] == 1216
    assert result["bindingValidation"]["frozenSmokeConfigurationsBlocked"] == 0
    assert len(result["accessRows"]) == 16
    assert all(row["deniedBeforeMaterialization"] for row in result["accessRows"])
    assert result["loadedProhibitedModules"] == []
    assert result["noMutation"] is True


def test_s08e_smoke_publishes_only_after_integrity_validation() -> None:
    source = Path("scripts/execute_portfolio_search_s08e.py").read_text(
        encoding="utf-8"
    )

    validation = source.index("smoke_validation = validate_rows(smoke_rows, 40)")
    rejection = source.index('if not smoke_validation["success"]:')
    publication = source.index(
        "publish_parquet_fail_atomic(\n"
        '        ledger_frame(smoke_rows), output / "frozen_smoke_results.parquet"'
    )
    assert validation < rejection < publication
    assert 'cache_dir=S08E_CACHE_ROOT / "evaluations"' in source
    assert "atomic_new_cache=True" in source
    assert "partial_smoke_results.parquet" not in source
    assert "/cache/e07-s08c" not in source
