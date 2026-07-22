from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest

from src.allocation_redesign.core import (
    PROTOCOL_PATH,
    build_allocation_roster,
    build_candidate_selection_frame,
    build_candidate_registry,
    build_scenario_registry,
    candidate_population_commitment,
    checked_protocol,
    derive_rare_status_registry,
    feasibility_summary,
    synthetic_worker_order_validation,
)


def test_protocol_is_design_only_and_gate_closed() -> None:
    protocol = checked_protocol()
    assert protocol["designOnly"] is True
    assert protocol["S07ExecutionApprovalGate"]["approved"] is False
    assert protocol["S07ExecutionApprovalGate"]["episodesEvaluated"] == 0
    assert protocol["S07ExecutionApprovalGate"]["allocationRosterWritten"] is False


def test_candidate_registry_is_exact_authorized_projection() -> None:
    protocol = checked_protocol()
    rows = build_candidate_registry(protocol)
    assert len(rows) == 512
    assert len(candidate_population_commitment(rows)) == 64
    assert {row["taskId"] for row in rows} == set(
        feasibility_summary(rows, protocol)["tasks"]
    )
    for task_id in {row["taskId"] for row in rows}:
        task_rows = [row for row in rows if row["taskId"] == task_id]
        assert len(task_rows) == 64
        assert sorted(row["complexityQuartile"] for row in task_rows) == [
            quartile for quartile in range(4) for _ in range(16)
        ]
    forbidden = {
        "outcome",
        "objectives",
        "nativeLedgerFamilies",
        "elapsedSeconds",
        "nativeEvent",
    }
    assert all(forbidden.isdisjoint(row) for row in rows)
    assert all(row["allocationEfficacyFieldsIncluded"] is False for row in rows)


def test_fixed_quotas_and_scenario_population_are_feasible() -> None:
    protocol = checked_protocol()
    candidates = build_candidate_registry(protocol)
    feasibility = feasibility_summary(candidates, protocol)
    assert feasibility["success"] is True
    assert all(row["descriptorQuotaFeasible"] for row in feasibility["tasks"].values())
    scenarios = build_scenario_registry(protocol)
    assert len(scenarios) == 32
    assert all(row["split"] == "train" and not row["materialized"] for row in scenarios)


def test_rare_registry_uses_only_frozen_status_signatures() -> None:
    rows = build_candidate_registry(checked_protocol())
    registry = derive_rare_status_registry(rows)
    assert registry["prevalenceThreshold"] == 0.10
    assert registry["unseenFutureSignaturesAreRare"] is True
    assert set(registry["tasks"]) == {row["taskId"] for row in rows}
    assert all(
        sum(signature["count"] for signature in task["signatures"])
        == task["evaluationRows"]
        for task in registry["tasks"].values()
    )


def test_real_roster_fails_closed_without_future_approval() -> None:
    protocol = checked_protocol()
    candidates = build_candidate_registry(protocol)
    with pytest.raises(PermissionError, match="separate execution approval"):
        build_allocation_roster(candidates, protocol)
    with pytest.raises(PermissionError, match="separate execution approval"):
        build_candidate_selection_frame(candidates, protocol)


def test_synthetic_selection_is_worker_order_independent() -> None:
    result = synthetic_worker_order_validation(checked_protocol())
    assert result["success"] is True
    assert result["fixtureOnly"] is True
    assert result["realCandidateAllocationPerformed"] is False
    assert result["syntheticCandidateSelectionFrameRows"] == 1536
    assert result["syntheticLogicalRosterRows"] == 3072
    assert result["completeAccountingPass"] is True
    assert len(set(result["allDigests"])) == 1


def test_allocation_module_does_not_import_rejected_model_code() -> None:
    before = set(sys.modules)
    importlib.import_module("src.allocation_redesign.core")
    newly_loaded = set(sys.modules) - before
    assert not any(
        name.startswith(("src.surrogate_models", "src.surrogate_remediation"))
        for name in newly_loaded
    )
    source = (Path(__file__).parents[1] / "src/allocation_redesign/core.py").read_text()
    assert "from src.surrogate" not in source
    assert "import src.surrogate" not in source


def test_protocol_source_is_frozen_repository_file() -> None:
    assert (
        PROTOCOL_PATH
        == Path(__file__).parents[1]
        / "configs/allocation/s07r_non_surrogate_protocol.yaml"
    )
