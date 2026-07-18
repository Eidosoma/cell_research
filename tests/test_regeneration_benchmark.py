from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
