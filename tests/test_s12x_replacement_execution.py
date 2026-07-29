from __future__ import annotations

import inspect
import json
from pathlib import Path

import yaml

from scripts import execute_replacement_spatial_transfer_s12x as s12x
from src.environment_suite.dsl_adapters import run_spatial_dsl_episode
from src.morph2d.engine import run_cpu_episode
from src.spatial_transfer.dependency_preflight import (
    DependencyRegistry,
    StaticImportAnalyzer,
    sha256_file,
)
from src.spatial_transfer.execution import choose_adaptation_winners
from src.spatial_transfer.execution_output_guard_s12x import (
    OutputAccessLedger,
    OutputAccessRegistry,
    validate_output_ledger,
)
from src.spatial_transfer.s12x_estimands import (
    FAULT_COST_FIELDS,
    SCHEDULER_COST_FIELDS,
    build_fixed_slot_registry,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONTROL = REPOSITORY / "configs/transfer/s12x_fresh_s12r_execution.yaml"


def _row(variant: str, completion: bool, mismatch: float) -> dict[str, object]:
    return {
        "baseConfigurationId": "a" * 64,
        "adaptationVariantId": variant,
        "taskId": "e07_s02_spatial2d_local",
        "panelId": "native_reference_control",
        "conditionId": "native_scheduler_no_fault",
        "endpointAvailable": True,
        "failed": False,
        "minimumMismatchFraction": mismatch,
        "terminalConjunctiveCompletion": completion,
        "faultFamilyId": "none",
        "repairByTransition32": None,
        "departureAfterInitiallyComplete": False,
    }


def test_s12x_control_is_exact_s12r_continuation() -> None:
    control = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    relationship = control["historicalRelationship"]
    assert control["researchStepId"] == "S12X"
    assert relationship["executesS12RReplacementEstimand"] is True
    assert relationship["executesS12P"] is False
    assert relationship["executesS12A"] is False
    assert relationship["executesOriginalS12Estimand"] is False
    assert control["freshness"]["cacheNamespace"] == "/cache/e07-s12x"
    assert control["frozenS12R"]["logicalReservationCount"] == 195_072
    assert control["frozenS12R"]["physicalReplayCommitmentCount"] == 390_144
    assert control["frozenS12R"]["taskConditionCellCount"] == 48
    assert control["frozenS12R"]["scenarioFamilyCount"] == 3_840
    assert control["frozenS12R"]["adaptationVariantCount"] == 30
    assert control["frozenS12R"]["replacementComparatorCount"] == 2
    assert control["frozenS12T"]["installOnEveryWorkerDispatch"] is True
    assert control["frozenS12W"]["exactQualifiedImplementationRequired"] is True
    assert control["frozenS12W"]["fixedFamilyConservationRequired"] is True
    assert control["frozenS12W"]["S12UCachePermanentlyProhibited"] is True
    assert control["execution"]["workers"] == 8
    assert control["execution"]["universalOutcomeScore"] is None
    assert control["execution"]["scalarCrossFamilyCost"] is None


def test_exact_frozen_s12t_dependency_control_is_bound() -> None:
    control = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    assert (
        sha256_file(s12x.S12T_REGISTRY_FILE) == control["frozenS12T"]["registrySha256"]
    )
    assert (
        sha256_file(s12x.S12T_IMPLEMENTATION)
        == control["frozenS12T"]["implementationSha256"]
    )
    assert (
        sha256_file(s12x.S12T_PROTOCOL_FILE) == control["frozenS12T"]["protocolSha256"]
    )
    registry = DependencyRegistry.load(s12x.S12T_REGISTRY_FILE)
    analysis = StaticImportAnalyzer(
        registry,
        search_roots=[REPOSITORY],
    ).analyze(
        [
            Path(s12x.__file__),
            REPOSITORY / "src/spatial_transfer/execution.py",
            REPOSITORY / "src/spatial_transfer/execution_output_guard_s12x.py",
            REPOSITORY / "src/spatial_transfer/estimator_feasibility.py",
            REPOSITORY / "src/spatial_transfer/s12x_estimands.py",
        ]
    )
    assert analysis["passed"], analysis["violations"]


def test_s12t_preflight_qualifies_real_s12x_dispatch_sources() -> None:
    result = s12x._s12t_dependency_preflight(s12x.control_record())
    assert result["passed"] is True
    assert result["runtimeDispatchAuditRequired"] is True
    assert result["cacheReadEvents"] == 0
    assert result["S07SignalReads"] == 0
    assert result["staticImportGraph"]["passed"] is True


def test_s12x_output_registry_deny_first_and_authenticated() -> None:
    registry = OutputAccessRegistry.load(s12x.OUTPUT_REGISTRY_FILE)
    assert registry.path_disposition("/cache/e07-s12x/part.parquet")["allowed"]
    assert registry.path_disposition("/artifacts/research_steps/S12X/status.json")[
        "allowed"
    ]
    assert not registry.path_disposition("/cache/e07-s12s/part.parquet")["allowed"]
    assert not registry.path_disposition("/cache/e07-s12u/part.parquet")["allowed"]
    assert not registry.path_disposition("/artifacts/research_steps/S14/outcome.json")[
        "allowed"
    ]
    ledger = OutputAccessLedger(registry, session_commitment="a" * 64)
    ledger.authorize_path(
        "/cache/e07-s12x/part.parquet",
        operation="write",
        artifact_class="fixture",
    )
    raw = ledger.finalize()
    validation = validate_output_ledger(
        raw,
        registry,
        expected_session_commitment="a" * 64,
    )
    assert validation["passed"]
    forged = json.loads(json.dumps(raw))
    forged["events"][0]["resolvedPath"] = "/cache/e07-s12s/forged.parquet"
    assert not validate_output_ledger(
        forged,
        registry,
        expected_session_commitment="a" * 64,
    )["passed"]


def test_native_and_dsl_runners_expose_engine_owned_schedule_callback() -> None:
    native = inspect.signature(run_cpu_episode)
    dsl = inspect.signature(run_spatial_dsl_episode)
    assert native.parameters["actor_schedule"].default is None
    assert native.parameters["actor_schedule_id"].default is None
    assert dsl.parameters["actor_schedule"].default is None


def test_s12x_structural_fixed_slot_registry_is_outcome_independent() -> None:
    structural = s12x._regenerate_structural_commitments()
    first = build_fixed_slot_registry(
        structural["roster"],
        structural["lineages"],
    )
    second = build_fixed_slot_registry(
        list(reversed(structural["roster"])),
        structural["lineages"],
    )
    control = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    assert first == second
    assert first["outcomeFieldsUsed"] == 0
    assert first["slotCount"] == 7_986
    assert set(first["familySlotCounts"]) == set(
        control["frozenS12W"]["fixedFamilies"]
    )
    assert all(value > 0 for value in first["familySlotCounts"].values())
    assert len({row["testId"] for row in first["slots"]}) == first["slotCount"]


def test_s12x_cost_components_match_frozen_registry() -> None:
    cost = s12x.read_json(s12x.S12R / "cost_registry.json")
    assert tuple(cost["faultLedger"]["separateFaultLedger"]) == FAULT_COST_FIELDS
    assert (
        tuple(cost["schedulerLedger"]["separateSchedulerLedger"])
        == SCHEDULER_COST_FIELDS
    )


def test_adaptation_lock_uses_pareto_then_lowest_hash_without_score() -> None:
    low = "1" * 64
    high = "2" * 64
    lock = choose_adaptation_winners(
        [
            _row(low, True, 0.25),
            _row(high, True, 0.25),
        ]
    )
    assert lock["configurationCount"] == 1
    assert lock["universalAdaptationScore"] is None
    assert lock["locks"][0]["winnerAdaptationVariantId"] == low
    assert lock["locks"][0]["nondominatedAdaptationVariantIds"] == [low, high]


def test_adaptation_lock_prefers_strict_task_local_pareto_dominance() -> None:
    dominated = "1" * 64
    winner = "2" * 64
    lock = choose_adaptation_winners(
        [
            _row(dominated, False, 0.50),
            _row(winner, True, 0.25),
        ]
    )
    assert lock["locks"][0]["winnerAdaptationVariantId"] == winner
    assert lock["locks"][0]["nondominatedAdaptationVariantIds"] == [winner]
