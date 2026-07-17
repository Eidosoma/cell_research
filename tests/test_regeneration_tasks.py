from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from reference_simulator.model import Direction
from src.regeneration.tasks import (
    BASELINE_SCENARIO_SCHEMA,
    TASK_SPEC_SCHEMA,
    apply_validation_injury,
    build_baseline_panel,
    strict_unequal_inversions,
    validate_pairing_rows,
    validate_task_spec,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/regeneration/s01_benchmark.json"


def _spec() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_task_spec_is_schema_valid_and_operationally_distinct() -> None:
    specification = _spec()
    validate_task_spec(specification)
    assert (
        TASK_SPEC_SCHEMA["properties"]["schemaVersion"]["const"]
        == specification["schemaVersion"]
    )
    families = {item["taskFamily"]: item for item in specification["taskFamilies"]}
    assert families["development_random"]["injuryOnset"] is None
    assert (
        families["repair_achieved"]["primaryTarget"]
        != families["repair_partial"]["primaryTarget"]
    )


def test_task_spec_rejects_overlap_between_achieved_and_partial_repair() -> None:
    specification = _spec()
    by_family = {item["taskFamily"]: item for item in specification["taskFamilies"]}
    by_family["repair_partial"]["primaryTarget"] = by_family["repair_achieved"][
        "primaryTarget"
    ]
    with pytest.raises(ValueError, match="distinct primary targets"):
        validate_task_spec(specification)


@pytest.mark.parametrize(
    ("values", "direction", "expected"),
    [
        ([3, 1, 2, 2], Direction.ASCENDING, 3),
        ([3, 1, 2, 2], Direction.DESCENDING, 2),
        ([1, 1, 1], Direction.ASCENDING, 0),
    ],
)
def test_strict_unequal_inversions_are_direction_and_duplicate_aware(
    values, direction, expected
) -> None:
    assert strict_unequal_inversions(values, direction) == expected


@pytest.mark.parametrize("direction", [Direction.ASCENDING, Direction.DESCENDING])
def test_validation_injury_adds_exactly_one_inversion(direction: Direction) -> None:
    occupancy = tuple(f"c{i}" for i in range(8))
    raw_values = list(range(8))
    if direction == Direction.DESCENDING:
        raw_values.reverse()
    values = {cell_id: value for cell_id, value in zip(occupancy, raw_values)}
    damaged, index = apply_validation_injury(occupancy, values, direction)
    before = strict_unequal_inversions([values[item] for item in occupancy], direction)
    after = strict_unequal_inversions([values[item] for item in damaged], direction)
    assert 0 <= index < len(occupancy) - 1
    assert after == before + 1


def test_pairing_validator_rejects_mismatched_preinjury_hash() -> None:
    base = {
        "pairingBlockId": "e05pb1:" + "a" * 64,
        "taskFamily": "repair_achieved",
        "arm": "matched_no_injury",
        "baselineScenarioId": "e05s01:" + "b" * 64,
        "n": 20,
        "policy": "Bubble",
        "direction": "ascending",
        "replicateOrdinal": 0,
        "runtimeSeed": "1",
        "injurySeed": "2",
        "preInjuryDistance": 0,
        "preInjuryEventIndex": 10,
        "preInjuryStateHash": "c" * 64,
        "executableScenarioId": "r1:" + "1" * 64,
        "targetValuesSha256": "d" * 64,
        "developmentBudget": 40000,
        "recoveryBudget": 40000,
        "scheduler": "uniform_random_activation",
        "continuation": "skip_and_continue",
        "mobility": "normal",
        "rngPairingStatus": "shared_prefix_until_arm_terminal",
    }
    active = deepcopy(base)
    active["arm"] = "validation_injury"
    active["baselineScenarioId"] = "e05s01:" + "e" * 64
    active["preInjuryStateHash"] = "f" * 64
    validation = validate_pairing_rows([base, active])
    assert not validation["success"]
    assert any("preInjuryStateHash" in item for item in validation["failures"])


def test_small_end_to_end_panel_validates_schemas_targets_budgets_and_pairing() -> None:
    specification = _spec()
    specification["validationPanel"] = {
        "sizes": [8],
        "policies": ["Bubble"],
        "directions": ["ascending", "descending"],
        "replicatesPerCell": 1,
        "inputProfile": "counter_addressed_random_permutation_unique_values",
        "injuryFixture": "s01_validation_adjacent_swap_v1",
    }
    panel = build_baseline_panel(specification)
    assert len(panel["baselineScenarios"]) == 10
    assert panel["validationSummary"]["success"]
    assert panel["pairingValidation"]["repairPairCount"] == 4
    repair_rows = [
        row
        for row in panel["baselineScenarios"]
        if row["taskFamily"] == "repair_achieved"
    ]
    assert len({row["executableScenarioId"] for row in repair_rows}) == 2
    assert {row["rngPairingStatus"] for row in repair_rows} == {
        "shared_prefix_until_arm_terminal"
    }
    assert all(
        audit["endEventIndex"] > audit["startEventIndex"]
        and audit["absorbingTargetCertificate"]
        and audit["acceptedMovementCount"] == 0
        for audit in panel["stabilizationValidation"]
    )
    assert all(
        snapshot["checkpoint"]["ledger"]["activations"]
        == snapshot["checkpoint"]["activationCount"]
        for snapshot in panel["checkpointSnapshots"]
    )
    assert (
        BASELINE_SCENARIO_SCHEMA["properties"]["schemaVersion"]["const"]
        == "e05.s01.baseline-scenario.v1"
    )
