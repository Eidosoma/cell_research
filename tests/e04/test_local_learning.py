"""E04 S06 local learning pilot tests."""

from __future__ import annotations

import unittest

from src.e04.local_learning import (
    LEARNING_POLICY_MODES,
    LocalLearningConfig,
    default_local_learning_configs,
    run_local_learning_matrix,
    simulate_local_learning_pilot,
)


class LocalLearningTests(unittest.TestCase):
    def test_learning_updates_logged_and_no_oracle(self) -> None:
        result = simulate_local_learning_pilot(
            LocalLearningConfig(
                task_name="swap_shocks",
                interface_mode="full",
                reliability_mode="no_fatigue_control",
                policy_mode="local_learning",
                activation_seed=41,
                policy_seed=1041,
                schedule_seed=2041,
                learning_seed=3041,
                max_events=40,
            )
        )
        row = result.to_row()

        self.assertGreater(row["learning_update_count"], 0)
        self.assertEqual(row["learning_update_count"], row["learning_decision_count"])
        self.assertFalse(row["uses_global_oracle"])
        self.assertIn("local_order_delta", row["learning_audit_json"])

    def test_nonlearning_control_makes_decisions_without_updates(self) -> None:
        result = simulate_local_learning_pilot(
            LocalLearningConfig(
                task_name="swap_shocks",
                interface_mode="full",
                reliability_mode="no_fatigue_control",
                policy_mode="nonlearning_control",
                activation_seed=42,
                policy_seed=1042,
                schedule_seed=2042,
                learning_seed=3042,
                max_events=40,
            )
        )
        row = result.to_row()

        self.assertGreater(row["learning_decision_count"], 0)
        self.assertEqual(row["learning_update_count"], 0)
        self.assertFalse(row["uses_global_oracle"])

    def test_update_access_indices_are_local_windows(self) -> None:
        result = simulate_local_learning_pilot(
            LocalLearningConfig(
                task_name="mixed_perturbations",
                interface_mode="full",
                reliability_mode="fatigue_recovery",
                policy_mode="local_learning",
                activation_seed=43,
                policy_seed=1043,
                schedule_seed=2043,
                learning_seed=3043,
                max_events=60,
            )
        )

        for event in result.event_log:
            indices = tuple(event["local_accessed_indices"])
            if not indices:
                continue
            actor = int(event["activated_position"])
            self.assertLessEqual(max(abs(idx - actor) for idx in indices), 2)

    def test_default_matrix_has_matched_learning_controls(self) -> None:
        configs = default_local_learning_configs(
            task_names=("swap_shocks", "frozen_damage"),
            interface_modes=("full", "no_signal"),
            reliability_modes=("no_fatigue_control",),
            seeds=(44,),
            max_events=10,
        )
        results = run_local_learning_matrix(configs)
        modes = {result.config.policy_mode for result in results}

        self.assertEqual(modes, set(LEARNING_POLICY_MODES))
        self.assertEqual(len(results), 8)
        groups = {}
        for result in results:
            key = (
                result.config.task_name,
                result.config.interface_mode,
                result.config.reliability_mode,
                result.config.activation_seed,
                result.config.schedule_seed,
            )
            groups.setdefault(key, set()).add(result.config.policy_mode)
        self.assertTrue(all(modes == set(LEARNING_POLICY_MODES) for modes in groups.values()))

    def test_fixed_seed_replay_is_deterministic(self) -> None:
        config = LocalLearningConfig(
            task_name="turnover_replacement",
            interface_mode="no_signal",
            reliability_mode="cumulative_damage",
            policy_mode="local_learning",
            activation_seed=45,
            policy_seed=1045,
            schedule_seed=2045,
            learning_seed=3045,
            max_events=70,
            damage_threshold=1,
        )

        first = simulate_local_learning_pilot(config)
        second = simulate_local_learning_pilot(config)

        self.assertEqual(first.to_row(), second.to_row())


if __name__ == "__main__":
    unittest.main()
