from __future__ import annotations

import unittest

from memory_repair import (
    apply_perturbation,
    build_homeostatic_benchmark_config,
    evaluate_trivial_controller,
    generate_seeded_schedule,
    normalized_sortedness_record,
    stable_hash,
)


class TestE04HomeostasisTasks(unittest.TestCase):
    def test_seeded_schedule_replay_is_deterministic(self) -> None:
        templates = [
            {"tick": 1, "type": "swap", "source": "seeded_random_adjacent"},
            {"tick": 2, "type": "insert", "source": "seeded_random_insert"},
            {"tick": 3, "type": "delete", "source": "seeded_random_delete"},
        ]
        first = generate_seeded_schedule(initial_values=[1, 2, 3, 4], schedule_seed=42, event_templates=templates)
        second = generate_seeded_schedule(initial_values=[1, 2, 3, 4], schedule_seed=42, event_templates=templates)
        different = generate_seeded_schedule(initial_values=[1, 2, 3, 4], schedule_seed=43, event_templates=templates)
        self.assertEqual(first, second)
        self.assertEqual(stable_hash(first), stable_hash(second))
        self.assertNotEqual(stable_hash(first), stable_hash(different))

    def test_insertion_uses_current_length_denominator(self) -> None:
        values, _, _ = apply_perturbation([1, 2, 3, 4], {"type": "insert", "position": 0, "value": 99})
        record = normalized_sortedness_record(values, initial_length=4)
        self.assertEqual(record["values"], [99, 1, 2, 3, 4])
        self.assertEqual(record["currentAdjacentPairCount"], 4)
        self.assertEqual(record["currentPairDenominator"], 4)
        self.assertEqual(record["initialPairDenominator"], 3)
        self.assertEqual(record["sortedPairCount"], 3)
        self.assertAlmostEqual(record["sortednessPercent"], 75.0)

    def test_deletion_recomputes_current_length_denominator(self) -> None:
        values, _, _ = apply_perturbation([99, 1, 2, 3, 4], {"type": "delete", "position": 0})
        record = normalized_sortedness_record(values, initial_length=4)
        self.assertEqual(record["values"], [1, 2, 3, 4])
        self.assertEqual(record["currentAdjacentPairCount"], 3)
        self.assertEqual(record["currentPairDenominator"], 3)
        self.assertEqual(record["sortednessPercent"], 100.0)

    def test_noop_controller_stays_perfect_without_perturbations(self) -> None:
        task = next(
            task for task in build_homeostatic_benchmark_config()["tasks"] if task["taskId"] == "steady_sorted_no_perturbation"
        )
        result = evaluate_trivial_controller(task, "noop")
        self.assertEqual(result["timeInRangeFraction"], 1.0)
        self.assertEqual(result["failureDurationTicks"], 0)
        self.assertEqual(result["energyProxy"], 0)

    def test_noop_controller_fails_seeded_swap_task(self) -> None:
        task = next(task for task in build_homeostatic_benchmark_config()["tasks"] if task["taskId"] == "adjacent_swap_recovery")
        result = evaluate_trivial_controller(task, "noop")
        self.assertGreater(result["failureDurationTicks"], 0)
        self.assertGreater(result["unrecoveredPerturbationCount"], 0)

    def test_oracle_sort_controller_recovers_seeded_swap_task(self) -> None:
        task = next(task for task in build_homeostatic_benchmark_config()["tasks"] if task["taskId"] == "adjacent_swap_recovery")
        result = evaluate_trivial_controller(task, "oracle_sort")
        self.assertEqual(result["timeInRangeFraction"], 1.0)
        self.assertEqual(result["failureDurationTicks"], 0)
        self.assertEqual(result["maxRecoveryTimeTicks"], 0)
        self.assertGreaterEqual(result["energyProxy"], 1)

    def test_config_contains_required_tasks_and_normalization_contract(self) -> None:
        config = build_homeostatic_benchmark_config()
        task_ids = {task["taskId"] for task in config["tasks"]}
        self.assertIn("insertion_deletion_normalization", task_ids)
        self.assertIn("frozen_damage_mixed_events", task_ids)
        self.assertIn("sortednessDenominator", config["normalization"])
        for task in config["tasks"]:
            self.assertIn("scheduleHash", task)
            self.assertIn("perturbationSchedule", task)


if __name__ == "__main__":
    unittest.main(verbosity=2)
