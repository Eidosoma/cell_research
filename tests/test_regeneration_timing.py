from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from reference_simulator.model import Direction, Policy

from src.regeneration.timing import (
    TIMING_SCENARIO_SCHEMA,
    TIMING_SPEC_SCHEMA,
    TimingClock,
    _source_scenario,
    build_timing_panel,
    locate_trigger,
    validate_timing_pairing,
    validate_timing_spec,
)
from src.regeneration.tasks import Checkpoint, initial_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/regeneration/s02_timing.json"


def _spec() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _checkpoint(initial: Checkpoint, event: int, distance: int) -> Checkpoint:
    return Checkpoint(
        initial.occupancy,
        initial.selection_cursors,
        event,
        initial.stream_counters,
        initial.ledger,
        distance,
        f"{event:064x}",
    )


def test_timing_spec_freezes_exact_eight_condition_panel_and_s01_contracts() -> None:
    specification = _spec()
    validate_timing_spec(specification)
    assert TIMING_SPEC_SCHEMA["properties"]["researchStepId"]["const"] == "S02"
    assert len(specification["timingConditions"]) == 8
    assert (
        specification["inherits"]["pairingProfile"]
        == "e05.s01.preinjury-shared-prefix-v1"
    )


def test_progress_first_crossing_and_event_clock_are_distinct() -> None:
    scenario, _, _ = _source_scenario(8, Policy.BUBBLE, Direction.ASCENDING, 0)
    initial = initial_checkpoint(scenario)
    initial = _checkpoint(initial, 0, 12)
    checkpoints = [
        _checkpoint(initial, 1, 11),
        _checkpoint(initial, 2, 9),
        _checkpoint(initial, 3, 8),
        _checkpoint(initial, 4, 6),
        _checkpoint(initial, 5, 0),
    ]
    progress = locate_trigger(
        {
            "timingConditionId": "progress_25",
            "clock": TimingClock.PROGRESS.value,
            "nominalFraction": 0.25,
        },
        initial=initial,
        development_checkpoints=checkpoints,
        development_stop_reason="complete",
        completion_events=5,
        stabilized=None,
    )
    event = locate_trigger(
        {
            "timingConditionId": "event_25",
            "clock": TimingClock.EVENT.value,
            "nominalFraction": 0.25,
        },
        initial=initial,
        development_checkpoints=checkpoints,
        development_stop_reason="complete",
        completion_events=5,
        stabilized=None,
    )
    assert progress.checkpoint is not None and progress.checkpoint.activation_count == 2
    assert progress.prior_progress < 0.25 <= progress.actual_progress
    assert event.checkpoint is not None and event.checkpoint.activation_count == 2
    assert event.planned_event_index == 2


def test_unreached_progress_and_event_references_are_retained_without_fallback() -> (
    None
):
    scenario, _, _ = _source_scenario(8, Policy.SELECTION, Direction.ASCENDING, 0)
    initial = initial_checkpoint(scenario)
    checkpoints = [_checkpoint(initial, 1, initial.distance)]
    progress = locate_trigger(
        {
            "timingConditionId": "progress_75",
            "clock": TimingClock.PROGRESS.value,
            "nominalFraction": 0.75,
        },
        initial=initial,
        development_checkpoints=checkpoints,
        development_stop_reason="quiescent",
        completion_events=None,
        stabilized=None,
    )
    event = locate_trigger(
        {
            "timingConditionId": "event_75",
            "clock": TimingClock.EVENT.value,
            "nominalFraction": 0.75,
        },
        initial=initial,
        development_checkpoints=checkpoints,
        development_stop_reason="quiescent",
        completion_events=None,
        stabilized=None,
    )
    assert progress.checkpoint is None
    assert progress.status == "unreached_terminal_before_progress_threshold"
    assert event.checkpoint is None
    assert event.status == "unreached_no_paired_completion_reference"


def test_pairing_rejects_preinjury_state_or_prefix_mismatch() -> None:
    base = {
        "timingPairId": "e05tp2:" + "a" * 64,
        "timingScenarioId": "e05s02:" + "b" * 64,
        "arm": "matched_no_injury",
        "s01PairingBlockId": "e05pb1:" + "c" * 64,
        "executableScenarioId": "r1:" + "d" * 64,
        "timingConditionId": "progress_25",
        "clock": "progress_first_crossing",
        "nominalFraction": 0.25,
        "triggerStatus": "triggered",
        "n": 20,
        "policy": "Bubble",
        "direction": "ascending",
        "replicateOrdinal": 0,
        "runtimeSeed": "1",
        "injurySeed": "2",
        "developmentBudget": 40000,
        "recoveryBudget": 40000,
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "preInjuryEventIndex": 5,
        "preInjuryDistance": 10,
        "preInjuryStateHash": "e" * 64,
        "preInjuryOccupancyHash": "f" * 64,
        "preInjuryInternalStateHash": "1" * 64,
        "preInjuryPrefixDigest": "2" * 64,
        "targetValuesSha256": "3" * 64,
        "rngPairingStatus": "shared_prefix_until_arm_terminal",
    }
    active = deepcopy(base)
    active["arm"] = "validation_injury"
    active["timingScenarioId"] = "e05s02:" + "4" * 64
    active["preInjuryPrefixDigest"] = "5" * 64
    audit = validate_timing_pairing([base, active])
    assert not audit["success"]
    assert any("preInjuryPrefixDigest" in item for item in audit["failures"])


def test_small_end_to_end_panel_validates_all_contracts_and_exact_replay() -> None:
    specification = _spec()
    specification["validationPanel"] = {
        "sizes": [8],
        "policies": ["Bubble", "Selection"],
        "directions": ["ascending", "descending"],
        "replicatesPerCell": 1,
        "sourceConstruction": "exact S01 counter-addressed construction keys and seeds",
        "arms": ["matched_no_injury", "validation_injury"],
    }
    panel = build_timing_panel(specification, exact_replay=True)
    assert len(panel["scenarioRows"]) == 4 * 8 * 2
    assert panel["validationSummary"]["success"]
    assert panel["runAccounting"]["silentExclusionCount"] == 0
    assert panel["severityValidation"]["success"]
    assert panel["replayValidation"]["recoveryReplayCount"] == 64
    assert (
        TIMING_SCENARIO_SCHEMA["properties"]["schemaVersion"]["const"]
        == "e05.s02.damage-timing-scenario.v1"
    )
