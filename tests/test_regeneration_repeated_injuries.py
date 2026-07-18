from __future__ import annotations

import json
from pathlib import Path

from reference_simulator.api import create_scenario
from reference_simulator.model import Direction, Policy
from src.regeneration.repeated_injuries import (
    BENCHMARK_VERSION,
    REPEATED_INJURY_RUN_SCHEMA_VERSION,
    REPEATED_INJURY_SPEC_SCHEMA_VERSION,
    SEQUENCE_ASSIGNMENT_STREAM,
    RepeatedArm,
    Spacing,
    exact_replay_repeated_case,
    inject_fatigue_snapshot,
    run_repeated_case,
    validate_repeated_injury_spec,
)
from src.regeneration.dynamic_faults import (
    DynamicFaultContract,
    DynamicProcessController,
    DynamicProfile,
)
from src.regeneration.tasks import initial_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/regeneration/s10_repeated_injuries.json"
PAIRING_ID = "e05pb1:" + "1" * 64


def _spec() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _scenario(*, policy: Policy = Policy.BUBBLE, max_activations: int = 50_000):
    return create_scenario(
        [7, 0, 6, 1, 5, 2, 4, 3],
        policy=policy,
        direction=Direction.ASCENDING,
        seed=0xE0510,
        max_activations=max_activations,
        generation_key=f"test/s10/{policy.value}/{max_activations}",
        permute=False,
    )


def test_spec_freezes_full_population_sequences_spacings_arms_and_counts() -> None:
    specification = _spec()
    validate_repeated_injury_spec(specification)
    assert specification["schemaVersion"] == REPEATED_INJURY_SPEC_SCHEMA_VERSION
    assert specification["benchmarkVersion"] == BENCHMARK_VERSION
    assert specification["sequenceAssignment"]["stream"] == SEQUENCE_ASSIGNMENT_STREAM
    assert {item["armId"] for item in specification["arms"]} == {
        item.value for item in RepeatedArm
    }
    assert {item["spacingId"] for item in specification["spacing"]} == {
        item.value for item in Spacing
    }
    assert specification["population"]["survivorOnlyAnalysis"] == "prohibited"
    assert specification["validationPanel"]["plannedRunCount"] == 5760


def test_fatigue_snapshot_injection_matches_load_and_residual_cooldown_exactly() -> None:
    scenario = _scenario()
    controller = DynamicProcessController(
        DynamicFaultContract(DynamicProfile.FATIGUE),
        scenario,
        scenario.cells[0].cell_id,
        0,
    )
    identities = sorted(controller.fatigue_load)
    donor = {
        "fatigueLoad": {identity: index % 3 for index, identity in enumerate(identities)},
        "residualCooldown": {identities[0]: 3, identities[-1]: 8},
    }
    audit = inject_fatigue_snapshot(controller, donor, 17)
    assert audit["exactMatch"]
    assert controller.fatigue_until == {identities[0]: 20, identities[-1]: 25}


def test_all_five_arms_replay_and_second_injuries_start_from_exact_target() -> None:
    scenario = _scenario()
    checkpoint = initial_checkpoint(scenario)
    result = run_repeated_case(
        scenario,
        checkpoint,
        pairing_id=PAIRING_ID,
        timing_condition_id="initialization",
        sequence_id="repeat_segment_reversal_v1",
        spacing=Spacing.REST_20N,
        injury_seed=31,
        recovery_budget=20_000,
        retain_trace=True,
    )
    assert set(result) == {item.value for item in RepeatedArm}
    assert all(item["schemaVersion"] == REPEATED_INJURY_RUN_SCHEMA_VERSION for item in result.values())
    assert all(all(item["validation"].values()) for item in result.values())
    for arm, item in result.items():
        if arm != RepeatedArm.NO_SECOND.value and item["episode2InjuryAdministered"]:
            assert item["episode2Lesion"]["severity"]["validPostTargetOrderDistanceBefore"] == 0
    exact_replay_repeated_case(
        result,
        scenario,
        checkpoint,
        pairing_id=PAIRING_ID,
        timing_condition_id="initialization",
        sequence_id="repeat_segment_reversal_v1",
        spacing=Spacing.REST_20N,
        injury_seed=31,
        recovery_budget=20_000,
        retain_trace=True,
    )


def test_first_stage_failures_remain_and_second_episode_is_not_substituted() -> None:
    scenario = _scenario(policy=Policy.SELECTION)
    checkpoint = initial_checkpoint(scenario)
    result = run_repeated_case(
        scenario,
        checkpoint,
        pairing_id=PAIRING_ID,
        timing_condition_id="initialization",
        sequence_id="repeat_block_transposition_v1",
        spacing=Spacing.IMMEDIATE,
        injury_seed=37,
        recovery_budget=1,
    )
    prior = result[RepeatedArm.PRIOR.value]
    assert not prior["firstStageEligible"]
    assert not prior["episode2InjuryAdministered"]
    assert prior["episode2OutcomeCause"] == "episode1_competing_terminal"
    assert prior["restrictedEpisode2Time"] == 1
    assert prior["jointSequenceSuccess"] is False


def test_phase_local_longitudinal_budget_can_cross_old_scenario_envelope() -> None:
    scenario = _scenario(max_activations=2)
    checkpoint = initial_checkpoint(scenario)
    result = run_repeated_case(
        scenario,
        checkpoint,
        pairing_id=PAIRING_ID,
        timing_condition_id="initialization",
        sequence_id="repeat_segment_reversal_v1",
        spacing=Spacing.IMMEDIATE,
        injury_seed=41,
        recovery_budget=200,
    )
    assert all(item["validation"]["phaseLocalEnvelopeUsed"] for item in result.values())
    assert all(item["finalState"]["globalEventIndex"] >= 0 for item in result.values())
