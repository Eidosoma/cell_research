"""E04 S05 homeostatic benchmark tests."""

from __future__ import annotations

import unittest

from src.e04.homeostasis import (
    HOMEOSTATIC_TASKS,
    HomeostaticBenchmarkConfig,
    build_homeostatic_schedule,
    default_homeostatic_configs,
    run_homeostatic_matrix,
    simulate_homeostatic_benchmark,
)


class HomeostaticBenchmarkTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_covers_perturbation_types(self) -> None:
        configs = [
            HomeostaticBenchmarkConfig(task_name=task, schedule_seed=31, max_events=120)
            for task in HOMEOSTATIC_TASKS
        ]
        schedules = [build_homeostatic_schedule(config) for config in configs]
        repeated = [build_homeostatic_schedule(config) for config in configs]

        self.assertEqual(schedules, repeated)
        kinds = {spec.perturbation_type for schedule in schedules for spec in schedule}
        self.assertIn("swap", kinds)
        self.assertIn("delete_insert", kinds)
        self.assertIn("freeze", kinds)

    def test_turnover_replacement_preserves_length_and_changes_values(self) -> None:
        result = simulate_homeostatic_benchmark(
            HomeostaticBenchmarkConfig(
                algorithm="bubble",
                task_name="turnover_replacement",
                repair_rule="nudge_repair",
                interface_mode="full",
                reliability_mode="no_fatigue_control",
                activation_seed=32,
                policy_seed=1032,
                schedule_seed=2032,
                max_events=70,
            )
        )
        turnover = [item for item in result.perturbation_log if item["perturbation_type"] == "delete_insert"]

        self.assertGreaterEqual(len(turnover), 1)
        self.assertEqual(len(result.final_values), len(result.config.values))
        self.assertTrue(any(item["values_changed"] for item in turnover))
        self.assertIn("fixed_size_delete_insert", turnover[0]["safe_representation"])

    def test_frozen_events_and_nudge_repair_log_unfreezing(self) -> None:
        result = simulate_homeostatic_benchmark(
            HomeostaticBenchmarkConfig(
                algorithm="bubble",
                task_name="frozen_damage",
                repair_rule="nudge_repair",
                interface_mode="full",
                reliability_mode="no_fatigue_control",
                activation_seed=4,
                policy_seed=1004,
                schedule_seed=2004,
                max_events=120,
                nudge_threshold=1,
            )
        )
        row = result.to_row()

        self.assertGreater(row["freeze_perturbation_count"], 0)
        self.assertGreaterEqual(row["unfreeze_count"], 1)
        self.assertFalse(row["uses_global_oracle"])

    def test_homeostatic_metrics_are_bounded_and_fixed_horizon(self) -> None:
        result = simulate_homeostatic_benchmark(
            HomeostaticBenchmarkConfig(
                algorithm="bubble",
                task_name="mixed_perturbations",
                repair_rule="nudge_repair",
                interface_mode="full",
                reliability_mode="fatigue_recovery",
                activation_seed=34,
                policy_seed=1034,
                schedule_seed=2034,
                max_events=80,
                fatigue_threshold=1,
                recovery_events=2,
            )
        )
        row = result.to_row()

        self.assertEqual(row["stop_reason"], "fixed_horizon_complete")
        self.assertEqual(row["event_count"], 80)
        self.assertGreaterEqual(row["time_in_target_fraction"], 0.0)
        self.assertLessEqual(row["time_in_target_fraction"], 1.0)
        self.assertGreaterEqual(row["perturbation_count"], 1)
        self.assertGreaterEqual(row["fatigue_transition_count"], 1)

    def test_default_matrix_contains_controls(self) -> None:
        configs = default_homeostatic_configs(algorithms=("bubble",), seeds=(35,), max_events=10)
        results = run_homeostatic_matrix(configs)
        task_names = {result.config.task_name for result in results}
        repair_rules = {result.config.repair_rule for result in results}
        interface_modes = {result.config.interface_mode for result in results}
        reliability_modes = {result.config.reliability_mode for result in results}

        self.assertEqual(len(results), 108)
        self.assertEqual(task_names, set(HOMEOSTATIC_TASKS))
        self.assertIn("passive_control", repair_rules)
        self.assertIn("stuck_control", repair_rules)
        self.assertIn("nudge_repair", repair_rules)
        self.assertIn("no_memory", interface_modes)
        self.assertIn("no_signal", interface_modes)
        self.assertIn("no_fatigue_control", reliability_modes)
        self.assertIn("cumulative_damage", reliability_modes)

    def test_fixed_seed_replay_is_deterministic(self) -> None:
        config = HomeostaticBenchmarkConfig(
            algorithm="bubble",
            task_name="mixed_perturbations",
            repair_rule="nudge_repair",
            interface_mode="full",
            reliability_mode="cumulative_damage",
            activation_seed=36,
            policy_seed=1036,
            schedule_seed=2036,
            max_events=80,
            damage_threshold=1,
        )

        first = simulate_homeostatic_benchmark(config)
        second = simulate_homeostatic_benchmark(config)

        self.assertEqual(first.to_row(), second.to_row())


if __name__ == "__main__":
    unittest.main()
