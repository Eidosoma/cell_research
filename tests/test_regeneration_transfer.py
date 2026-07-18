from __future__ import annotations

import json
from pathlib import Path

from causal_simulator.schedulers import SchedulerFamily
from reference_simulator.api import create_scenario
from reference_simulator.model import Direction, Policy
from src.regeneration.target_change import TargetChange
from src.regeneration.tasks import initial_checkpoint
from src.regeneration.transfer import (
    TransferSchedule,
    TransferTargetChange,
    apply_transfer_lesion,
    build_transfer_target_definition,
    exact_replay_result,
    lesion_window,
    run_goal_transfer,
    transfer_target_correspondence_rows,
    validate_transfer_spec,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _sorted_fixture(n: int = 12, policy: Policy = Policy.BUBBLE):
    scenario = create_scenario(
        list(range(n)),
        policy=policy,
        direction=Direction.ASCENDING,
        seed=17012,
        max_activations=100 * n * n + 20 * n,
        generation_key=f"E05/S12/test/n{n}/{policy.value}",
        permute=False,
    )
    return scenario, initial_checkpoint(scenario)


def test_frozen_spec_and_disjoint_split() -> None:
    specification = json.loads(
        (REPOSITORY / "configs/regeneration/s12_transfer.json").read_text()
    )
    validate_transfer_spec(specification)


def test_location_reversal_has_equal_severity_and_preserves_state() -> None:
    scenario, checkpoint = _sorted_fixture(20)
    outcomes = []
    for location in ("central", "left_off_center", "right_off_center"):
        lesion = apply_transfer_lesion(
            scenario,
            checkpoint,
            lesion_type="segment_reversal_central_v1",
            location=location,
        )
        outcomes.append(lesion)
        assert lesion["identityConserved"]
        assert lesion["activationCountPreserved"]
        assert lesion["streamCountersPreserved"]
        assert lesion["ledgerPreserved"]
        start, end = lesion_window(20, location)
        assert lesion["postDistance"] == (end - start) * (end - start - 1) // 2
    assert len({item["postDistance"] for item in outcomes}) == 1


def test_sattolo_window_is_deranged_and_deterministic() -> None:
    scenario, checkpoint = _sorted_fixture(20)
    first = apply_transfer_lesion(
        scenario,
        checkpoint,
        lesion_type="local_scramble_sattolo_v1",
        location="left_off_center",
    )
    second = apply_transfer_lesion(
        scenario,
        checkpoint,
        lesion_type="local_scramble_sattolo_v1",
        location="left_off_center",
    )
    assert first == second
    start, end = first["windowStart"], first["windowEndExclusive"]
    before = first["preOccupancy"][start:end]
    after = first["postOccupancy"][start:end]
    assert all(left != right for left, right in zip(before, after))


def test_scheduler_pairing_and_stream_isolation() -> None:
    scenario, checkpoint = _sorted_fixture(12)
    for family in (
        SchedulerFamily.UNIFORM_RANDOM_ACTIVATION,
        SchedulerFamily.DETERMINISTIC_SCAN,
        SchedulerFamily.RANDOM_PERMUTATION_SWEEP,
    ):
        left = TransferSchedule(scenario, family, checkpoint.activation_count)
        right = TransferSchedule(scenario, family, checkpoint.activation_count)
        left_items = [left(index, 1)[0] for index in range(24)]
        right_items = [right(index, 1)[0] for index in range(24)]
        assert left_items == right_items
        streams = {
            stream
            for item in left_items
            for stream, _ in item.stream_consumption
        }
        if family == SchedulerFamily.UNIFORM_RANDOM_ACTIVATION:
            assert streams == {"actor_activation"}
        elif family == SchedulerFamily.DETERMINISTIC_SCAN:
            assert not streams
        else:
            assert streams == {"scheduler_permutation_s04_v1"}


def test_new_target_correspondence_feasible() -> None:
    scenario, _ = _sorted_fixture(20)
    for target in TransferTargetChange:
        definition = build_transfer_target_definition(scenario, target)
        rows = transfer_target_correspondence_rows(scenario, definition)
        assert len(rows) == 20
        assert all(row["targetFeasible"] and row["identityConserved"] for row in rows)
        assert definition.maximum_distance > 0
        assert definition.distance(
            tuple(
                identity
                for identity, _ in sorted(
                    definition.target_code_map.items(), key=lambda item: (item[1], item[0])
                )
            )
        ) == 0


def test_goal_gateway_exact_replay_and_no_change_stability() -> None:
    scenario, checkpoint = _sorted_fixture(12)

    def factory():
        return run_goal_transfer(
            scenario,
            checkpoint,
            target_change=TargetChange.PARTIAL,
            scheduler=SchedulerFamily.RANDOM_PERMUTATION_SWEEP,
            process_id="none",
            target_aware=True,
            adaptation_budget=100 * 12 * 12,
            probe_budget=20 * 12,
            stability=True,
        )

    result = factory()
    exact_replay_result(factory, result)
    assert result["success"]
    assert not result["noChangeAnyTargetDeparture"]
    assert result["phaseActivationCount"] == 240
    assert all(result["validation"].values())


def test_dynamic_target_quiescence_uses_preprocess_eligibility() -> None:
    scenario, checkpoint = _sorted_fixture(12, Policy.SELECTION)
    result = run_goal_transfer(
        scenario,
        checkpoint,
        target_change=TransferTargetChange.QUARTILE_ROTATION,
        scheduler=SchedulerFamily.DETERMINISTIC_SCAN,
        process_id="intermittent_movement_markov_1_16_1_4_v1",
        target_aware=True,
        adaptation_budget=100 * 12 * 12,
        probe_budget=20 * 12,
    )
    assert result["stopReason"] in {
        "target_hit_probe_complete",
        "target_quiescent",
    }
    assert result["phaseActivationCount"] < 100 * 12 * 12 + 20 * 12
    assert all(result["validation"].values())
