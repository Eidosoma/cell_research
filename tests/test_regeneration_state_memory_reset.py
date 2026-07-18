from __future__ import annotations

import json
from pathlib import Path

from reference_simulator.api import create_scenario
from reference_simulator.model import Direction, Policy
from src.regeneration.state_memory_reset import (
    BENCHMARK_VERSION,
    ENGINEERED_EMPTY_STATE,
    SCRAMBLE_STREAM,
    STATE_MEMORY_RESET_RUN_SCHEMA_VERSION,
    STATE_MEMORY_RESET_SPEC_SCHEMA_VERSION,
    ResetArm,
    distance_matched_scramble,
    exact_replay_state_memory_reset_case,
    run_state_memory_reset_case,
    validate_state_memory_reset_spec,
)
from src.regeneration.tasks import (
    initial_checkpoint,
    occupancy_values,
    strict_unequal_inversions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/regeneration/s11_state_memory_reset.json"
PAIRING_ID = "e05pb1:" + "2" * 64


def _spec() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _scenario(*, policy: Policy = Policy.BUBBLE):
    return create_scenario(
        [7, 0, 6, 1, 5, 2, 4, 3],
        policy=policy,
        direction=Direction.ASCENDING,
        seed=0xE0511,
        max_activations=50_000,
        generation_key=f"test/s11/{policy.value}",
        permute=False,
    )


def test_spec_freezes_complete_factorial_shams_and_full_population_counts() -> None:
    specification = _spec()
    validate_state_memory_reset_spec(specification)
    assert specification["schemaVersion"] == STATE_MEMORY_RESET_SPEC_SCHEMA_VERSION
    assert specification["benchmarkVersion"] == BENCHMARK_VERSION
    assert {item["armId"] for item in specification["arms"]} == {
        item.value for item in ResetArm
    }
    assert specification["population"]["survivorOnlyAnalysis"] == "prohibited"
    assert specification["validationPanel"]["plannedRunCount"] == 3840
    assert specification["statePartitions"]["engineeredState"]["fields"] == []
    assert ENGINEERED_EMPTY_STATE == {}


def test_distance_matched_scramble_is_deterministic_nonidentical_and_exact() -> None:
    scenario = _scenario()
    occupancy = [cell.cell_id for cell in scenario.cells]
    first = distance_matched_scramble(scenario, occupancy, case_address=17)
    second = distance_matched_scramble(scenario, occupancy, case_address=17)
    assert first == second
    assert first["feasible"]
    assert first["stream"] == SCRAMBLE_STREAM
    assert first["candidateOccupancy"] != occupancy
    assert set(first["candidateOccupancy"]) == set(occupancy)
    before_distance = strict_unequal_inversions(
        occupancy_values(scenario, occupancy), Direction.ASCENDING
    )
    after_distance = strict_unequal_inversions(
        occupancy_values(scenario, first["candidateOccupancy"]),
        Direction.ASCENDING,
    )
    assert after_distance == before_distance


def test_unique_distance_classes_are_explicitly_infeasible_without_substitution() -> None:
    scenario = _scenario()
    target = [
        cell.cell_id for cell in sorted(scenario.cells, key=lambda item: item.value)
    ]
    low = distance_matched_scramble(scenario, target, case_address=19)
    high = distance_matched_scramble(scenario, list(reversed(target)), case_address=19)
    assert not low["feasible"] and low["reason"] == "unique_distance_class"
    assert not high["feasible"] and high["reason"] == "unique_distance_class"
    assert low["candidateOccupancy"] is None
    assert high["candidateOccupancy"] is None


def test_all_ten_arms_replay_reset_isolation_and_sham_equivalence() -> None:
    scenario = _scenario(policy=Policy.SELECTION)
    checkpoint = initial_checkpoint(scenario)
    result = run_state_memory_reset_case(
        scenario,
        checkpoint,
        pairing_id=PAIRING_ID,
        timing_condition_id="initialization",
        sequence_id="repeat_segment_reversal_v1",
        injury_seed=43,
        recovery_budget=20_000,
        case_address=23,
        retain_trace=True,
    )
    assert set(result) == {item.value for item in ResetArm}
    assert all(
        item["schemaVersion"] == STATE_MEMORY_RESET_RUN_SCHEMA_VERSION
        for item in result.values()
    )
    assert all(all(item["validation"].values()) for item in result.values())
    reference = result[ResetArm.A0_N0_F0.value]
    assert reference["firstStageEligible"]
    assert reference["resetBoundaryObserved"]
    pre_hashes = {item["preResetState"]["stateHash"] for item in result.values()}
    assert len(pre_hashes) == 1
    for sham in (ResetArm.SHAM_ARRANGEMENT, ResetArm.SHAM_INTERNAL):
        item = result[sham.value]
        assert item["episode2"] == reference["episode2"]
        assert item["finalState"] == reference["finalState"]
        assert item["finalFatigue"] == reference["finalFatigue"]
        assert item["nativeLedgerDelta"] == reference["nativeLedgerDelta"]
        assert item["processLedger"] == reference["processLedger"]
        assert item["nativeEventDigest"] == reference["nativeEventDigest"]
    exact_replay_state_memory_reset_case(
        result,
        scenario,
        checkpoint,
        pairing_id=PAIRING_ID,
        timing_condition_id="initialization",
        sequence_id="repeat_segment_reversal_v1",
        injury_seed=43,
        recovery_budget=20_000,
        case_address=23,
        retain_trace=True,
    )


def test_first_stage_competing_terminal_remains_in_every_factorial_and_sham_arm() -> None:
    scenario = _scenario(policy=Policy.SELECTION)
    checkpoint = initial_checkpoint(scenario)
    result = run_state_memory_reset_case(
        scenario,
        checkpoint,
        pairing_id=PAIRING_ID,
        timing_condition_id="initialization",
        sequence_id="repeat_block_transposition_v1",
        injury_seed=47,
        recovery_budget=1,
        case_address=29,
    )
    assert len(result) == 10
    for item in result.values():
        assert not item["firstStageEligible"]
        assert not item["episode2InjuryAdministered"]
        assert not item["resetBoundaryObserved"]
        assert not item["resetApplied"]
        assert item["episode2OutcomeCause"] == "episode1_competing_terminal"
        assert item["jointSequenceSuccess"] is False
        assert item["restrictedEpisode2Time"] == 1
        assert item["resetAudit"]["validation"]["unobservableResetRetained"]
