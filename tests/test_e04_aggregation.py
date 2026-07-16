from __future__ import annotations

import unittest

from analysis.aggregation import (
    EXPECTED_CONDITIONS,
    EXPECTED_RUNS,
    _linear_curve,
    _metric_fixture,
    load_tasks,
)


class E04AggregationTests(unittest.TestCase):
    def test_population_is_frozen_and_unprotected(self) -> None:
        tasks = load_tasks()
        self.assertEqual(len(tasks), EXPECTED_RUNS)
        counts: dict[str, int] = {}
        for task in tasks:
            row = task["scenario_row"]
            counts[row["conditionId"]] = counts.get(row["conditionId"], 0) + 1
            self.assertEqual(row["split"], "paper_scale")
            self.assertFalse(row["protected"])
        self.assertEqual(len(counts), EXPECTED_CONDITIONS)
        self.assertEqual(set(counts.values()), {100})
        self.assertEqual(sum(bool(task["retain_raw_trace"]) for task in tasks), EXPECTED_CONDITIONS)

    def test_repeated_three_way_extension_is_explicit(self) -> None:
        tasks = load_tasks()
        extension = [task for task in tasks if not task["upstream_scenario"]]
        self.assertEqual(len(extension), 200)
        self.assertEqual(
            {task["scenario_row"]["conditionId"] for task in extension},
            {
                "C-REP-CHIM-BUB-INS-SEL-EXACT-ASC",
                "C-REP-CHIM-BUB-INS-SEL-RANDOM-ASC",
            },
        )

    def test_linear_progress_fixture(self) -> None:
        points = [
            {"swap_index": 0, "paper_sortedness_percent": 25.0, "publication_aggregation": 0.25},
            {"swap_index": 2, "paper_sortedness_percent": 75.0, "publication_aggregation": 0.75},
            {"swap_index": 4, "paper_sortedness_percent": 100.0, "publication_aggregation": 0.5},
        ]
        curve = _linear_curve(points)
        self.assertEqual(len(curve), 101)
        self.assertAlmostEqual(curve[25]["publication_aggregation"], 0.5)
        self.assertAlmostEqual(curve[50]["publication_aggregation"], 0.75)
        self.assertAlmostEqual(curve[-1]["paper_sortedness_percent"], 100.0)

    def test_publication_metric_fixture(self) -> None:
        fixture = _metric_fixture()
        self.assertTrue(fixture["passed"], fixture)


if __name__ == "__main__":
    unittest.main()
