from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from src.environment_suite.dsl_adapters import run_spatial_dsl_episode
from src.morph2d.engine import run_cpu_episode
from src.spatial_transfer.execution import choose_adaptation_winners


REPOSITORY = Path(__file__).resolve().parents[1]
CONTROL = REPOSITORY / "configs/transfer/s12s_fresh_s12r_execution.yaml"


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


def test_s12s_control_is_exact_s12r_continuation() -> None:
    control = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    relationship = control["historicalRelationship"]
    assert control["researchStepId"] == "S12S"
    assert relationship["executesS12RReplacementEstimand"] is True
    assert relationship["executesS12P"] is False
    assert relationship["executesS12A"] is False
    assert relationship["executesOriginalS12Estimand"] is False
    assert control["freshness"]["cacheNamespace"] == "/cache/e07-s12s"
    assert control["frozenS12R"]["logicalReservationCount"] == 195_072
    assert control["frozenS12R"]["physicalReplayCommitmentCount"] == 390_144


def test_native_and_dsl_runners_expose_engine_owned_schedule_callback() -> None:
    native = inspect.signature(run_cpu_episode)
    dsl = inspect.signature(run_spatial_dsl_episode)
    assert native.parameters["actor_schedule"].default is None
    assert native.parameters["actor_schedule_id"].default is None
    assert dsl.parameters["actor_schedule"].default is None


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
