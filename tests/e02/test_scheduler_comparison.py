"""Focused tests for the E02 S02 scheduler comparison helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.e02_s02_scheduler_comparison import (
    SCHEDULER_REGIMES,
    SchedulerTask,
    dg_from_sortedness,
    run_scheduler_task,
)


class SchedulerComparisonTests(unittest.TestCase):
    def make_task(self, scheduler: str) -> SchedulerTask:
        return SchedulerTask(
            condition={
                "condition_id": "TEST",
                "run_family": "unperturbed_baseline",
                "mode": "cell_view",
                "algorithm": "bubble",
                "algotype_mix": "bubble",
                "value_bank_id": "test_values",
                "algotype_assignment_bank_id": "not_applicable",
                "frozen_semantics": "none",
                "frozen_count": 0,
                "frozen_index_bank_id": "frozen_0",
            },
            repeat_index=0,
            values=(4, 1, 3, 2),
            initial_array_seed=1,
            assignments=("bubble", "bubble", "bubble", "bubble"),
            assignment_seed=None,
            frozen_indices=(),
            frozen_index_seed=None,
            scheduler_regime=scheduler,
            scheduler_seed=101,
            repo_dir=str(Path(__file__).resolve().parents[2]),
            max_sweeps=500,
            max_successful_swaps=1000,
            stall_sweeps=2,
        )

    def test_dg_toy_cases_match_e01_formula(self) -> None:
        self.assertEqual(dg_from_sortedness([50, 60, 70])["dg_primary"], 0.0)
        self.assertEqual(dg_from_sortedness([50, 40, 70])["dg_primary"], 2.0)
        self.assertEqual(dg_from_sortedness([50, 60, 40, 70])["dg_primary"], 0.5)

    def test_all_scheduler_regimes_sort_small_bubble_fixture(self) -> None:
        for scheduler in SCHEDULER_REGIMES:
            with self.subTest(scheduler=scheduler):
                row = run_scheduler_task(self.make_task(scheduler))
                self.assertEqual(row["stop_reason"], "sorted")
                self.assertEqual(row["final_sortedness_percent"], 100.0)
                self.assertEqual(row["final_monotonicity_error_count"], 0)
                self.assertTrue(row["activation_log_sha256"])


if __name__ == "__main__":
    unittest.main()
