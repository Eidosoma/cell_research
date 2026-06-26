from __future__ import annotations

import unittest

from scripts.e02_s12_frozen_behavior_variants import (
    ALGORITHMS,
    PLACEMENT_CATEGORIES,
    BehaviorVariantSimulator,
    behavior_conditions,
    behavior_specs,
    compare_baseline,
    required_behavior_ids,
    run_vanilla_baseline,
)


def spec_by_id(behavior_id: str):
    return {spec.behavior_id: spec for spec in behavior_specs()}[behavior_id]


class TestE02FrozenBehaviorVariants(unittest.TestCase):
    def test_required_behavior_variants_are_present(self) -> None:
        expected = {
            "baseline_passive",
            "baseline_stuck",
            "probabilistic_stuck_p50",
            "time_varying_stuck_periodic",
            "fatigue_recovery",
            "directional_sticky_from_left",
            "directional_sticky_from_right",
        }
        self.assertEqual(required_behavior_ids(), expected)
        self.assertEqual(len(behavior_conditions()), len(ALGORITHMS) * len(PLACEMENT_CATEGORIES) * len(expected))

    def test_directional_sticky_blocks_only_configured_side(self) -> None:
        values = [3, 2, 1]
        left_block = BehaviorVariantSimulator(
            values,
            "bubble",
            behavior=spec_by_id("directional_sticky_from_left"),
            behavior_seed=1,
            frozen_positions=[1],
            scheduler_seed=2,
            tie_breaker_seed=3,
            condition_id="unit_left_block",
        )
        left_outcome = left_block.step(forced_cell_id=0, forced_direction=1)
        self.assertFalse(left_outcome.swapped)
        self.assertEqual(left_block.behavior_blocked_attempts, 1)

        right_block = BehaviorVariantSimulator(
            values,
            "bubble",
            behavior=spec_by_id("directional_sticky_from_right"),
            behavior_seed=1,
            frozen_positions=[1],
            scheduler_seed=2,
            tie_breaker_seed=3,
            condition_id="unit_right_block",
        )
        right_outcome = right_block.step(forced_cell_id=0, forced_direction=1)
        self.assertTrue(right_outcome.swapped)
        self.assertEqual(right_block.behavior_allowed_frozen_target_swaps, 1)

    def test_fatigue_recovery_logs_transition_after_repeated_nudges(self) -> None:
        sim = BehaviorVariantSimulator(
            [3, 2, 1],
            "bubble",
            behavior=spec_by_id("fatigue_recovery"),
            behavior_seed=1,
            frozen_positions=[1],
            scheduler_seed=2,
            tie_breaker_seed=3,
            condition_id="unit_fatigue",
        )
        sim.activation_count = 1
        self.assertTrue(sim._can_swap(0, 1))
        sim.activation_count = 2
        self.assertTrue(sim._can_swap(0, 1))
        transition_reasons = [row["reason"] for row in sim.behavior_event_log if row["eventType"] == "transition"]
        self.assertIn("fatigue_threshold_reached", transition_reasons)
        sim.activation_count = 20
        sim._update_fatigue_recovery(1)
        transition_reasons = [row["reason"] for row in sim.behavior_event_log if row["eventType"] == "transition"]
        self.assertIn("recovery_window_elapsed", transition_reasons)

    def test_probabilistic_variant_is_seed_replayable(self) -> None:
        draws = []
        for _ in range(2):
            sim = BehaviorVariantSimulator(
                [3, 2, 1],
                "bubble",
                behavior=spec_by_id("probabilistic_stuck_p50"),
                behavior_seed=123,
                frozen_positions=[1],
                scheduler_seed=2,
                tie_breaker_seed=3,
                condition_id="unit_probability",
            )
            sim.activation_count = 1
            sim._can_swap(0, 1)
            draws.append([row["randomDraw"] for row in sim.behavior_event_log if row["eventType"] == "decision"])
        self.assertEqual(draws[0], draws[1])

    def test_passive_and_stuck_baselines_match_vanilla_simulator(self) -> None:
        values = [4, 1, 3, 2]
        for behavior_id in ("baseline_passive", "baseline_stuck"):
            with self.subTest(behavior_id=behavior_id):
                behavior = spec_by_id(behavior_id)
                custom = BehaviorVariantSimulator(
                    values,
                    "bubble",
                    behavior=behavior,
                    behavior_seed=1,
                    frozen_positions=[1],
                    scheduler_seed=5,
                    tie_breaker_seed=6,
                    condition_id=f"unit_{behavior_id}",
                )
                custom_result = custom.run(max_activations=200, max_swaps=100, max_comparisons=400)
                condition = type(
                    "Condition",
                    (),
                    {
                        "algorithm": "bubble",
                        "placement_category": "unit",
                        "behavior": behavior,
                    },
                )()
                vanilla_result = run_vanilla_baseline(
                    initial_values=values,
                    condition=condition,
                    frozen_positions=[1],
                    scheduler_seed=5,
                    tie_seed=6,
                    max_activations=200,
                    max_swaps=100,
                    max_comparisons=400,
                )
                comparison = compare_baseline(custom_result, vanilla_result, f"unit_{behavior_id}", behavior_id)
                self.assertTrue(comparison["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
