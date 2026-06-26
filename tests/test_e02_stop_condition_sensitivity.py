from __future__ import annotations

import unittest

from scripts.e02_s13_stop_condition_sensitivity import (
    S13Condition,
    balanced_algotypes,
    condition_seeds,
    convergence_reached,
    frozen_positions_for_condition,
    goal_sortedness_by_algotype,
    labels_for_algotypes,
    policy_config_hash,
    reverse_flags_for_algotypes,
    run_stop_policy,
    s13_conditions,
    stop_policies,
)


class TestE02StopConditionSensitivity(unittest.TestCase):
    def test_stop_policy_families_cover_s13_requirements(self) -> None:
        families = {policy.policy_family for policy in stop_policies(30)}
        self.assertTrue(
            {"canonical", "no_movement_window", "event_cap", "convergence_criteria", "stable_state_definition"}.issubset(
                families
            )
        )
        stable_definitions = {policy.stable_state_definition for policy in stop_policies(30)}
        self.assertTrue({"legal_action_absent", "no_swap_window", "state_hash_plateau"}.issubset(stable_definitions))

    def test_condition_classes_include_baseline_frozen_and_conflict(self) -> None:
        classes = {condition.condition_class for condition in s13_conditions()}
        self.assertEqual(classes, {"baseline", "frozen_cell", "conflict_chimera"})

    def test_policy_hash_is_constant_across_stop_policies(self) -> None:
        condition = next(item for item in s13_conditions() if item.condition_template_id == "conflict_bubble_down_selection_up")
        seeds = condition_seeds(condition, 0)
        algotypes = balanced_algotypes(condition.algotypes, 30, seeds["algotypeSeed"])
        reverse = reverse_flags_for_algotypes(algotypes, condition.goal_directions)
        frozen_positions = frozen_positions_for_condition(condition, 30)
        hashes = {
            policy_config_hash(algotypes, reverse, frozen_positions, condition.frozen_variant, condition.goal_directions)
            for _policy in stop_policies(30)
        }
        self.assertEqual(len(hashes), 1)
        self.assertIn(True, reverse)
        self.assertIn(False, reverse)

    def test_tight_activation_cap_stops_unsorted_reverse_bubble(self) -> None:
        condition = S13Condition(
            "unit_reverse_bubble",
            "baseline",
            "bubble",
            ("bubble",),
            {"bubble": "increasing"},
        )
        policy = next(policy for policy in stop_policies(5) if policy.stop_policy_id == "tight_activation_cap")
        result = run_stop_policy(
            initial_values=[5, 4, 3, 2, 1],
            initial_algotypes=["bubble"] * 5,
            reverse_directions=[False] * 5,
            frozen_positions=[],
            labels=labels_for_algotypes(["bubble"] * 5),
            condition=condition,
            policy=policy,
            replicate_index=0,
            scheduler_seed=1,
            tie_seed=2,
        )
        self.assertIn(result.stop_reason, {"max_activation_cap", "sorted"})
        if result.stop_reason == "max_activation_cap":
            self.assertFalse(result.global_increasing_sorted)

    def test_goal_aware_convergence_for_conflict_subsequences(self) -> None:
        condition = S13Condition(
            "unit_conflict",
            "conflict_chimera",
            "bubble+selection",
            ("bubble", "selection"),
            {"bubble": "decreasing", "selection": "increasing"},
        )
        metrics = goal_sortedness_by_algotype(
            [4, 1, 3, 2],
            ["bubble", "selection", "bubble", "selection"],
            condition.goal_directions,
        )
        self.assertTrue(metrics["bubble"]["complete"])
        self.assertTrue(metrics["selection"]["complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
