"""E04 S03 repairable Frozen Cell benchmark tests."""

from __future__ import annotations

import unittest

from src.e04.repairable_frozen import (
    RepairBenchmarkConfig,
    default_benchmark_configs,
    run_benchmark_matrix,
    simulate_repair_benchmark,
)


class RepairableFrozenTests(unittest.TestCase):
    def test_nudge_repair_unfreezes_and_logs_trigger(self) -> None:
        result = simulate_repair_benchmark(
            RepairBenchmarkConfig(
                algorithm="bubble",
                repair_rule="nudge_repair",
                interface_mode="full",
                activation_seed=11,
                policy_seed=1011,
                max_events=40,
                nudge_threshold=1,
            )
        )

        self.assertTrue(result.repair_success)
        self.assertEqual(len(result.unfreeze_log), 1)
        self.assertEqual(result.unfreeze_log[0]["trigger"], "nudge_threshold")
        self.assertFalse(result.signal_audit["usesGlobalOracle"])

    def test_stuck_and_passive_controls_remain_controls(self) -> None:
        stuck = simulate_repair_benchmark(
            RepairBenchmarkConfig(
                algorithm="bubble",
                repair_rule="stuck_control",
                interface_mode="full",
                activation_seed=12,
                policy_seed=1012,
                max_events=30,
            )
        )
        passive = simulate_repair_benchmark(
            RepairBenchmarkConfig(
                algorithm="bubble",
                repair_rule="passive_control",
                interface_mode="full",
                activation_seed=12,
                policy_seed=1012,
                max_events=30,
            )
        )

        self.assertFalse(stuck.repair_success)
        self.assertFalse(passive.repair_success)
        self.assertEqual(len(stuck.unfreeze_log), 0)
        self.assertEqual(len(passive.unfreeze_log), 0)
        self.assertGreaterEqual(stuck.frozen_attempt_count, 1)

    def test_signal_threshold_repair_requires_signals_for_this_condition(self) -> None:
        common = dict(
            algorithm="bubble",
            repair_rule="signal_threshold_repair",
            activation_seed=13,
            policy_seed=1013,
            max_events=80,
            signal_threshold=0.2,
        )
        full = simulate_repair_benchmark(RepairBenchmarkConfig(**common, interface_mode="full"))
        no_signal = simulate_repair_benchmark(RepairBenchmarkConfig(**common, interface_mode="no_signal"))

        self.assertTrue(full.repair_success)
        self.assertEqual(full.unfreeze_log[0]["trigger"], "signal_threshold")
        self.assertFalse(no_signal.repair_success)
        self.assertEqual(len(no_signal.unfreeze_log), 0)

    def test_time_and_directional_triggers_log_conditions(self) -> None:
        time_result = simulate_repair_benchmark(
            RepairBenchmarkConfig(
                algorithm="bubble",
                repair_rule="time_repair",
                interface_mode="no_signal",
                activation_seed=14,
                policy_seed=1014,
                max_events=20,
                time_threshold=3,
            )
        )
        direction_result = simulate_repair_benchmark(
            RepairBenchmarkConfig(
                algorithm="bubble",
                repair_rule="directional_repair",
                interface_mode="full",
                activation_seed=15,
                policy_seed=1015,
                max_events=20,
                directional_required_from="left",
            )
        )

        self.assertTrue(time_result.repair_success)
        self.assertEqual(time_result.unfreeze_log[0]["trigger"], "time_threshold")
        self.assertTrue(direction_result.repair_success)
        self.assertEqual(direction_result.unfreeze_log[0]["trigger"], "directional_contact")
        self.assertEqual(direction_result.unfreeze_log[0]["direction_from_actor"], "left")

    def test_fixed_seed_replay_is_deterministic(self) -> None:
        config = RepairBenchmarkConfig(
            algorithm="bubble",
            repair_rule="signal_threshold_repair",
            interface_mode="full",
            activation_seed=16,
            policy_seed=1016,
            max_events=50,
            signal_threshold=0.2,
        )
        first = simulate_repair_benchmark(config)
        second = simulate_repair_benchmark(config)

        self.assertEqual(first.to_row(), second.to_row())

    def test_default_matrix_contains_controls_and_ablation_modes(self) -> None:
        configs = default_benchmark_configs(algorithms=("bubble",), seeds=(17,), max_events=10)
        results = run_benchmark_matrix(configs)
        rules = {result.config.repair_rule for result in results}
        modes = {result.config.interface_mode for result in results}

        self.assertIn("passive_control", rules)
        self.assertIn("stuck_control", rules)
        self.assertIn("no_memory", modes)
        self.assertIn("no_signal", modes)
        self.assertEqual(len(results), 18)


if __name__ == "__main__":
    unittest.main()
