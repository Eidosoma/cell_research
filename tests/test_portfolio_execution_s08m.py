from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from scripts.execute_portfolio_search_s08g import (
    reservation_roster_audit,
    s08f_contract_revalidation,
    s08h_contract_revalidation,
    s08j_contract_revalidation,
    s08l_contract_revalidation,
)
from src.portfolio_search.execution import run_preflight


CONTROL = Path("configs/portfolio/s08m_execution_continuation.yaml")
BUDGET = Path("/artifacts/research_steps/S08P/budget_slot_ledger.parquet")


def test_s08m_control_is_fresh_exact_and_identity_gated() -> None:
    control = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))

    assert control["researchStepId"] == "S08M"
    assert control["prospectivePreregistration"] is True
    assert control["historicalDesignMayChange"] is False
    assert control["freshCacheRoot"] == "/cache/e07-s08m"
    assert control["quarantinedCacheRoots"] == [
        "/cache/e07-s08c",
        "/cache/e07-s08e",
        "/cache/e07-s08g",
        "/cache/e07-s08i",
        "/cache/e07-s08k",
    ]
    assert control["frozenExecution"]["smokeLogicalRows"] == 40
    assert control["frozenExecution"]["trainingLogicalRows"] == 11008
    assert control["frozenExecution"]["adaptiveGenerations"] == 6
    assert control["identityPersistence"]["reservationRosterRows"] == 11008
    assert control["identityPersistence"]["initialRosterCommittedBeforeSmoke"] is True
    assert (
        control["identityPersistence"][
            "returnedPlaneValidatedAfterSmokeAndEveryGeneration"
        ]
        is True
    )
    assert control["validationBoundary"]["confirmationLogicalRows"] == 0


def test_s08m_preflight_clears_g01_to_g06() -> None:
    result = run_preflight(CONTROL)

    assert result["researchStepId"] == "S08M"
    assert result["success"] is True
    assert result["blockedGateIds"] == []
    assert result["bindingValidation"]["configurationRowsBound"] == 1216
    assert result["bindingValidation"]["frozenSmokeConfigurationsBlocked"] == 0
    assert len(result["accessRows"]) == 16
    assert all(row["deniedBeforeMaterialization"] for row in result["accessRows"])
    assert result["loadedProhibitedModules"] == []
    assert result["noMutation"] is True


def test_s08m_revalidates_all_bounded_contracts_and_reservations() -> None:
    assert s08f_contract_revalidation()["success"]
    assert s08h_contract_revalidation()["success"]
    assert s08j_contract_revalidation()["success"]
    assert s08l_contract_revalidation()["success"]

    audit = reservation_roster_audit(pd.read_parquet(BUDGET))
    assert audit["success"]
    assert audit["logicalReservations"] == 11008
    assert audit["uniqueLogicalSlots"] == 11008
    assert audit["uniqueReservationSlotIdentities"] == 11008
    assert audit["byGeneration"] == {
        0: 4864,
        1: 1024,
        2: 1024,
        3: 1024,
        4: 1024,
        5: 1024,
        6: 1024,
    }


def test_identity_checks_precede_publication_archive_and_validation_outcomes() -> None:
    source = Path("scripts/execute_portfolio_search_s08g.py").read_text(
        encoding="utf-8"
    )
    initial_precommit = source.index(
        '        "initial_roster_precommitment",\n        initial_work'
    )
    smoke_execution = source.index("smoke_rows, smoke_physical = execute_work(")
    smoke_identity = source.index('        "smoke_returned",\n        smoke_rows')
    smoke_publication = source.index(
        "publish_parquet_fail_atomic(\n"
        '        ledger_frame(smoke_rows), output / "frozen_smoke_results.parquet"'
    )
    generation_identity = source.index(
        '            f"generation_{generation}_returned",\n            rows'
    )
    generation_aggregate = source.index(
        "generation_aggregates = aggregate_panel(definitions, rows)"
    )
    validation_precommit = source.index(
        '        "validation_roster_precommitment",\n        validation_work'
    )
    validation_execution = source.index(
        "validation_rows, validation_physical = execute_work("
    )

    assert initial_precommit < smoke_execution
    assert smoke_identity < smoke_publication
    assert generation_identity < generation_aggregate
    assert validation_precommit < validation_execution
