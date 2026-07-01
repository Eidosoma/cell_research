"""Focused tests for E02 S05 dummy-Algotype controls."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.e02_s05_dummy_algotypes import (
    DUMMY_BEHAVIOR,
    DUMMY_LABELS,
    DummyTask,
    run_assignment,
    run_dummy_task,
)
from src.e02.deterministic_simulator import DEFAULT_LABEL_TO_BEHAVIOR


class DummyAlgotypeControlTests(unittest.TestCase):
    def make_task(self) -> DummyTask:
        return DummyTask(
            condition={
                "condition_id": "E01C047",
                "run_family": "same_goal_chimera",
                "mode": "cell_view",
                "algorithm": "mixed",
                "algotype_mix": "same_algorithm_bubble_label_control",
                "value_bank_id": "test_values",
                "algotype_assignment_bank_id": "bubble_bubble_label_control_50_50",
                "frozen_semantics": "none",
                "frozen_count": 0,
                "frozen_index_bank_id": "frozen_0",
            },
            repeat_index=0,
            values=(4, 1, 3, 2),
            initial_array_seed=1,
            assignments=("bubble_label_a", "bubble_label_b", "bubble_label_a", "bubble_label_b"),
            assignment_seed=2,
            frozen_indices=(),
            frozen_index_seed=None,
            scheduler_seed=123,
            repo_dir=str(Path(__file__).resolve().parents[2]),
            max_sweeps=100,
            max_successful_swaps=1000,
            stall_sweeps=2,
        )

    def test_dummy_labels_map_to_same_behavior(self) -> None:
        for label in DUMMY_LABELS:
            self.assertEqual(DEFAULT_LABEL_TO_BEHAVIOR[label], DUMMY_BEHAVIOR)

    def test_dummy_dispatch_and_pure_reference_match_value_trace(self) -> None:
        task = self.make_task()
        dummy = run_assignment(task, task.assignments, "dummy")
        pure = run_assignment(task, tuple("bubble" for _ in task.assignments), "pure")
        self.assertTrue(dummy["dispatch"]["dispatchProven"])
        self.assertEqual(dummy["value_trace_sha256"], pure["value_trace_sha256"])
        self.assertEqual(dummy["sortedness_trace_sha256"], pure["sortedness_trace_sha256"])
        self.assertEqual(dummy["final_values"], pure["final_values"])

    def test_run_dummy_task_records_balanced_counts_and_sorted_output(self) -> None:
        result = run_dummy_task(self.make_task())["row"]
        self.assertTrue(result["dispatch_proven"])
        self.assertTrue(result["dummy_labels_balanced_initial"])
        self.assertTrue(result["dummy_labels_balanced_final"])
        self.assertEqual(result["activation_label_balance_abs_difference"], 0)
        self.assertTrue(result["value_trace_matches_pure_reference"])
        self.assertEqual(result["stop_reason"], "sorted")
        self.assertEqual(result["final_sortedness_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
