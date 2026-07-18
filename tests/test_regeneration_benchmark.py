from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_regeneration_s14 import core_comparator_figure_data
from src.regeneration.benchmark import (
    CORE_S13_PORTFOLIOS,
    run_fresh_core_smoke,
    s13_partition,
    split_is_outcome_invariant,
    validate_benchmark_spec,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SPECIFICATION = REPOSITORY / "configs/regeneration/s14_regeneration_benchmark.json"


def load_specification() -> dict:
    return json.loads(SPECIFICATION.read_text(encoding="utf-8"))


def test_s14_specification_freezes_required_core_and_no_aggregate_score() -> None:
    specification = load_specification()
    validate_benchmark_spec(specification)
    assert set(specification["coreExtendedSplit"]["core"]["s13Portfolios"]) == CORE_S13_PORTFOLIOS
    assert specification["suiteAndCompetencyContract"]["aggregateScorePermitted"] is False
    assert specification["e07UsageSplit"]["outcomeInputsProhibited"] is True


def test_s13_partition_is_outcome_blind() -> None:
    core = {
        "portfolioId": "pure_bubble",
        "placementMap": 0,
        "success": True,
        "stopReason": "complete",
    }
    extended = {
        "portfolioId": "chimera_bubble_selection",
        "placementMap": 0,
        "success": False,
        "stopReason": "quiescent",
    }
    sensitivity = {
        "portfolioId": "chimera_bubble_insertion",
        "placementMap": 1,
        "success": True,
        "stopReason": "complete",
    }
    assert s13_partition(core) == "core"
    assert s13_partition(extended) == "extended"
    assert s13_partition(sensitivity) == "extended"
    assert all(split_is_outcome_invariant(row) for row in [core, extended, sensitivity])


def test_frozen_s13_main_partition_counts_and_union() -> None:
    frame = pd.read_parquet("/artifacts/research_steps/S13/chimeric_recovery.parquet")
    main = frame[frame["thresholdCaseId"].isna()].copy()
    core = main[
        main["portfolioId"].isin(CORE_S13_PORTFOLIOS)
        & (main["placementMap"] == 0)
    ]
    extended = main.drop(core.index)
    assert len(main) == 2816
    assert len(core) == 768
    assert len(extended) == 2048
    assert set(core["mainCaseId"]).isdisjoint(set(extended["mainCaseId"]))
    assert set(core["mainCaseId"]) | set(extended["mainCaseId"]) == set(main["mainCaseId"])


def test_fresh_core_smoke_replays_exactly() -> None:
    result = run_fresh_core_smoke(load_specification())
    assert result["success"] is True
    assert result["developmentReplayPass"] is True
    assert result["stabilizationReplayPass"] is True
    assert result["recoveryReplayPass"] is True
    assert result["allRuntimeValidationPass"] is True


def test_core_comparator_figure_uses_explicit_validated_arms() -> None:
    portfolios = ["pure_bubble", "pure_insertion", "chimera_bubble_insertion"]
    tasks = [
        ("injury", "segment_reversal_central_v1", "native_recovery", 1.0),
        ("injury", "local_scramble_sattolo_v1", "native_recovery", 1.0),
        ("target_change", "adjacent_pair_swap_total_order_v1", "changed_target_aware", 1.0),
        ("target_change", "adjacent_pair_swap_total_order_v1", "changed_nonadaptive_matched", 0.0),
        ("target_change", "quartile_rotation_2_0_3_1_v1", "changed_target_aware", 1.0),
        ("target_change", "quartile_rotation_2_0_3_1_v1", "changed_nonadaptive_matched", 0.0),
    ]
    rows = [
        {
            "portfolioId": portfolio,
            "taskKind": task_kind,
            "taskId": task_id,
            "arm": arm,
            "successRate": success_rate,
        }
        for portfolio in portfolios
        for task_kind, task_id, arm, success_rate in tasks
    ]
    # A legacy alias must not silently stand in for the validated target-aware arm.
    rows.append(
        {
            "portfolioId": "pure_bubble",
            "taskKind": "target_change",
            "taskId": "adjacent_pair_swap_total_order_v1",
            "arm": "changed_aware",
            "successRate": 0.5,
        }
    )
    selected = core_comparator_figure_data(pd.DataFrame(rows))
    assert len(selected) == 18
    assert set(selected["arm"]) == {
        "native_recovery",
        "changed_target_aware",
        "changed_nonadaptive_matched",
    }
    target = selected[selected["taskKind"] == "target_change"]
    assert target[target["arm"] == "changed_target_aware"]["successRate"].eq(1.0).all()
    assert target[target["arm"] == "changed_nonadaptive_matched"]["successRate"].eq(0.0).all()
