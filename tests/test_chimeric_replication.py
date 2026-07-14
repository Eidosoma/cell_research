from __future__ import annotations

import json
from pathlib import Path
import unittest

from analysis.chimeric_replication import (
    AdmissibilityTracker,
    GRID,
    MetricTracker,
    _execute_s13_activation,
    _composition_null,
    _floor_grid,
    load_tasks,
)
from reference_simulator.engine import (
    evaluate_terminal,
    execute_serial_summary_activation,
    initial_state,
    is_complete,
)
from reference_simulator.model import Cell, Policy, Scenario
from reference_simulator.policies import has_admissible_change
from scenario_bank.core import materialize_scenario
from analysis.no_fault_sorting import condition_from_dict


class ChimericReplicationTests(unittest.TestCase):
    def test_frozen_population_and_trace_selection(self) -> None:
        tasks = load_tasks()
        self.assertEqual(len(tasks), 2900)
        self.assertEqual(sum(bool(task["retain_raw_trace"]) for task in tasks), 29)
        self.assertEqual({task["scenario_row"]["split"] for task in tasks}, {"paper_scale"})
        self.assertFalse(any(task["scenario_row"]["protected"] for task in tasks))

    def test_publication_and_reference_nulls(self) -> None:
        publication, reference = _composition_null({"A": 50, "B": 50}, 100)
        self.assertAlmostEqual(publication, 0.49)
        self.assertAlmostEqual(reference, 49 / 99)
        publication3, reference3 = _composition_null({"A": 34, "B": 33, "C": 33}, 100)
        self.assertAlmostEqual(publication3, (34 * 33 + 2 * 33 * 32) / 10000)
        self.assertAlmostEqual(publication3, reference3 * 0.99)

    def test_incremental_metrics_after_nonadjacent_swap(self) -> None:
        scenario = Scenario.create(
            (
                Cell("a", 3, Policy.BUBBLE, analysis_label="x"),
                Cell("b", 1, Policy.BUBBLE, analysis_label="y"),
                Cell("c", 1, Policy.BUBBLE, analysis_label="y"),
                Cell("d", 2, Policy.BUBBLE, analysis_label="x"),
            ),
            initial_occupancy=("a", "b", "c", "d"),
            generation_key="S13/test/nonadjacent",
        )
        tracker = MetricTracker(scenario, control=True)
        state = initial_state(scenario)
        state.occupancy[0], state.occupancy[3] = state.occupancy[3], state.occupancy[0]
        tracker.after_swap(state, 0, 3)
        metrics = tracker.metrics()
        self.assertAlmostEqual(metrics["paper_sortedness_percent"], 50.0)
        self.assertAlmostEqual(metrics["reference_sortedness_percent"], 75.0)
        self.assertAlmostEqual(metrics["publication_aggregation"], 0.25)
        self.assertAlmostEqual(metrics["reference_aggregation"], 1 / 3)
        self.assertAlmostEqual(metrics["duplicate_same_algotype_edge_rate"], 1.0)

    def test_floor_grid_uses_no_interpolation(self) -> None:
        points = [{"swap_index": index, "value": index} for index in range(4)]
        grid = _floor_grid(points)
        self.assertEqual(len(grid), len(GRID))
        self.assertEqual(grid[33]["value"], 0)
        self.assertEqual(grid[34]["value"], 1)
        self.assertEqual(grid[-1]["value"], 3)

    def test_incremental_admissibility_matches_authoritative_predicates(self) -> None:
        task = next(
            item
            for item in load_tasks()
            if item["scenario_row"]["conditionId"] == "C-UNQ-CHIM-BUB-INS-EXACT-OPP"
            and item["scenario_row"]["replicateOrdinal"] == 1
        )
        scenario, _ = materialize_scenario(condition_from_dict(task["condition"]), task["base"])
        state = initial_state(scenario)
        state.terminal = evaluate_terminal(scenario, state)
        tracker = AdmissibilityTracker(scenario, state)
        checked = 0
        while state.terminal is None:
            swapped = False

            def callback(current, first, second):
                nonlocal swapped
                swapped = True
                tracker.after_swap(current, first, second)

            changed = execute_serial_summary_activation(scenario, state, on_accepted_swap=callback)
            if changed:
                if not swapped:
                    tracker.after_memory_update(state)
                self.assertEqual(tracker.completion(), is_complete(scenario, state))
                self.assertEqual(tracker.has_change(state), has_admissible_change(scenario, state))
                checked += 1
                state.terminal = evaluate_terminal(scenario, state)
        self.assertGreater(checked, 100)

    def test_specialized_activation_matches_authoritative_projection(self) -> None:
        tasks = load_tasks()
        for condition_id in (
            "C-UNQ-CHIM-BUB-INS-EXACT-OPP",
            "C-UNQ-CHIM-INS-SEL-EXACT-OPP",
        ):
            task = next(
                item
                for item in tasks
                if item["scenario_row"]["conditionId"] == condition_id
                and item["scenario_row"]["replicateOrdinal"] == 1
            )
            scenario, _ = materialize_scenario(condition_from_dict(task["condition"]), task["base"])
            fast = initial_state(scenario)
            authoritative = initial_state(scenario)
            fast.terminal = evaluate_terminal(scenario, fast)
            authoritative.terminal = evaluate_terminal(scenario, authoritative)
            tracker = AdmissibilityTracker(scenario, fast)
            while fast.terminal is None:
                swapped = False

                def callback(current, first, second):
                    nonlocal swapped
                    swapped = True
                    tracker.after_swap(current, first, second)

                outcome = _execute_s13_activation(scenario, fast, tracker, on_swap=callback)
                changed = execute_serial_summary_activation(scenario, authoritative)
                self.assertEqual(outcome != "noop", changed)
                if outcome != "noop":
                    if not swapped:
                        tracker.after_memory_update(fast)
                    fast.terminal = tracker.terminal_after_change(fast)
                    authoritative.terminal = evaluate_terminal(scenario, authoritative)
                self.assertEqual(fast.occupancy, authoritative.occupancy)
                self.assertEqual(fast.selection_cursors, authoritative.selection_cursors)
                self.assertEqual(fast.stream_counters, authoritative.stream_counters)
                self.assertEqual(fast.ledger, authoritative.ledger)
                self.assertEqual(fast.terminal, authoritative.terminal)

    def test_preregistration_is_valid_and_frozen_to_s13(self) -> None:
        path = Path(__file__).resolve().parents[1] / "analysis" / "s13_chimera_preregistration.json"
        value = json.loads(path.read_text())
        self.assertEqual(value["researchStepId"], "S13")
        self.assertEqual(value["analysisPopulation"]["executedReferenceRuns"], 2900)
        self.assertEqual(value["analysisPopulation"]["totalResultRows"], 3200)
        self.assertIn("/n", value["metrics"]["publicationAggregation"])


if __name__ == "__main__":
    unittest.main()
