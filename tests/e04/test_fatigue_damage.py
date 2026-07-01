"""E04 S04 fatigue and damage benchmark tests."""

from __future__ import annotations

import unittest

from src.e04.fatigue_damage import (
    FatigueDamageBenchmarkConfig,
    default_fatigue_damage_configs,
    run_fatigue_damage_matrix,
    simulate_fatigue_damage_benchmark,
)
from src.e04.repairable_frozen import simulate_repair_benchmark


class FatigueDamageTests(unittest.TestCase):
    def test_fatigue_recovery_logs_transition_recovery_and_impairment(self) -> None:
        result = simulate_fatigue_damage_benchmark(
            FatigueDamageBenchmarkConfig(
                algorithm="bubble",
                repair_rule="nudge_repair",
                interface_mode="full",
                reliability_mode="fatigue_recovery",
                activation_seed=21,
                policy_seed=1021,
                max_events=60,
                nudge_threshold=1,
                fatigue_threshold=1,
                recovery_events=2,
            )
        )

        states = {(entry["from_state"], entry["to_state"], entry["reason"]) for entry in result.fatigue_transition_log}
        self.assertIn(("healthy", "fatigued", "fatigue_threshold_reached"), states)
        self.assertIn(("fatigued", "healthy", "recovery_after_rest"), states)
        self.assertGreater(len(result.impairment_log), 0)
        self.assertFalse(result.signal_audit["usesGlobalOracle"])

    def test_cumulative_damage_logs_damage_and_persistent_impairment(self) -> None:
        result = simulate_fatigue_damage_benchmark(
            FatigueDamageBenchmarkConfig(
                algorithm="bubble",
                repair_rule="nudge_repair",
                interface_mode="full",
                reliability_mode="cumulative_damage",
                activation_seed=22,
                policy_seed=1022,
                max_events=60,
                nudge_threshold=1,
                damage_threshold=1,
            )
        )

        self.assertTrue(any(entry["to_state"] == "damaged" for entry in result.fatigue_transition_log))
        self.assertTrue(all(entry["state"] == "damaged" for entry in result.impairment_log))
        self.assertGreater(len(result.impairment_log), 0)

    def test_no_fatigue_control_matches_s03_repair_baseline(self) -> None:
        config = FatigueDamageBenchmarkConfig(
            algorithm="bubble",
            repair_rule="nudge_repair",
            interface_mode="full",
            reliability_mode="no_fatigue_control",
            activation_seed=23,
            policy_seed=1023,
            max_events=50,
            nudge_threshold=1,
        )

        s04_result = simulate_fatigue_damage_benchmark(config)
        s03_result = simulate_repair_benchmark(config.to_repair_config())

        self.assertTrue(s04_result.baseline_match)
        self.assertEqual(s04_result.final_values, s03_result.final_values)
        self.assertEqual(s04_result.final_frozen_positions, s03_result.final_frozen_positions)
        self.assertEqual(s04_result.swap_count, s03_result.swap_count)

    def test_default_matrix_preserves_s03_and_s04_controls(self) -> None:
        configs = default_fatigue_damage_configs(algorithms=("bubble",), seeds=(24,), max_events=10)
        results = run_fatigue_damage_matrix(configs)
        rules = {result.config.repair_rule for result in results}
        modes = {result.config.interface_mode for result in results}
        reliability_modes = {result.config.reliability_mode for result in results}

        self.assertEqual(len(results), 27)
        self.assertIn("passive_control", rules)
        self.assertIn("stuck_control", rules)
        self.assertIn("no_memory", modes)
        self.assertIn("no_signal", modes)
        self.assertIn("no_fatigue_control", reliability_modes)
        self.assertIn("fatigue_recovery", reliability_modes)
        self.assertIn("cumulative_damage", reliability_modes)

    def test_fixed_seed_replay_is_deterministic(self) -> None:
        config = FatigueDamageBenchmarkConfig(
            algorithm="bubble",
            repair_rule="nudge_repair",
            interface_mode="full",
            reliability_mode="fatigue_recovery",
            activation_seed=25,
            policy_seed=1025,
            max_events=50,
            nudge_threshold=1,
            fatigue_threshold=1,
            recovery_events=2,
        )

        first = simulate_fatigue_damage_benchmark(config)
        second = simulate_fatigue_damage_benchmark(config)

        self.assertEqual(first.to_row(), second.to_row())

    def test_signal_audit_reports_no_global_oracle(self) -> None:
        result = simulate_fatigue_damage_benchmark(
            FatigueDamageBenchmarkConfig(
                algorithm="bubble",
                repair_rule="nudge_repair",
                interface_mode="full",
                reliability_mode="fatigue_recovery",
                activation_seed=26,
                policy_seed=1026,
                max_events=20,
            )
        )

        self.assertFalse(result.to_row()["uses_global_oracle"])


if __name__ == "__main__":
    unittest.main()
