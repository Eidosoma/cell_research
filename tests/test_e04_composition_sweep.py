from __future__ import annotations

import json
from pathlib import Path
import unittest

from analysis.chimeric_replication import policy_edge_summary
from analysis.composition_sweep import (
    EXPECTED_CONDITIONS,
    EXPECTED_RUNS,
    SweepCondition,
    build_conditions,
    load_base_draws,
    materialize_sweep_scenario,
)
from reference_simulator.model import Cell, Policy, Scenario


class CompositionSweepTests(unittest.TestCase):
    def test_frozen_factorial_accounting(self) -> None:
        conditions = build_conditions()
        self.assertEqual(len(conditions), EXPECTED_CONDITIONS)
        self.assertEqual(EXPECTED_RUNS, 46_500)
        pairwise = [item for item in conditions if item.composition_class == "pairwise"]
        three_way = [
            item for item in conditions if item.composition_class == "three_way"
        ]
        self.assertEqual(len(pairwise), 162)
        self.assertEqual(len(three_way), 24)
        self.assertEqual(
            {item.first_policy_count for item in pairwise}, set(range(10, 100, 10))
        )

    def test_three_way_exact_count_rules(self) -> None:
        balanced = SweepCondition(
            "fixture",
            "unique_1_100",
            ("Bubble", "Insertion", "Selection"),
            "balanced_rotating",
            "absent",
        )
        self.assertEqual(balanced.counts(0), (34, 33, 33))
        self.assertEqual(balanced.counts(1), (33, 34, 33))
        self.assertEqual(balanced.counts(2), (33, 33, 34))
        rare = SweepCondition(
            "fixture-rare",
            "unique_1_100",
            ("Bubble", "Insertion", "Selection"),
            "rare_insertion_10",
            "positive",
            rare_policy="Insertion",
        )
        self.assertEqual(rare.counts(17), (45, 10, 45))

    def test_correlation_assignment_is_exact_and_deterministic(self) -> None:
        base = load_base_draws()[("unique_1_100", 0)]
        correlations = {}
        for profile in ("positive", "negative", "absent"):
            condition = SweepCondition(
                f"fixture-{profile}",
                "unique_1_100",
                ("Bubble", "Selection"),
                "p10_90",
                profile,
                first_policy_count=10,
            )
            first, metadata = materialize_sweep_scenario(condition, base)
            second, repeated = materialize_sweep_scenario(condition, base)
            self.assertEqual(first.to_json_bytes(), second.to_json_bytes())
            self.assertEqual(metadata, repeated)
            self.assertEqual(metadata["policyCounts"], {"Bubble": 10, "Selection": 90})
            correlations[profile] = metadata["achievedSpearmanRho"]
        self.assertGreater(correlations["positive"], 0)
        self.assertLess(correlations["negative"], 0)

    def test_policy_edge_summary_has_expected_lattice(self) -> None:
        scenario = Scenario.create(
            (
                Cell("a", 1, Policy.BUBBLE),
                Cell("b", 2, Policy.BUBBLE),
                Cell("c", 3, Policy.SELECTION),
                Cell("d", 4, Policy.SELECTION),
            ),
            initial_occupancy=("a", "b", "c", "d"),
            generation_key="E04/S03/test/policy-edges",
        )
        edges, rates = policy_edge_summary(scenario, scenario.initial_occupancy)
        self.assertEqual(edges, {"Bubble": 1, "Selection": 1})
        self.assertEqual(rates, {"Bubble": 0.5, "Selection": 0.5})

    def test_repository_contract_is_frozen_to_s03(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "analysis/s03_composition_factorial.json"
        )
        value = json.loads(path.read_text())
        self.assertEqual(value["researchStepId"], "S03")
        self.assertTrue(value["frozenBeforeSimulation"])
        self.assertEqual(value["factorial"]["conditionCount"], 186)
        self.assertEqual(value["factorial"]["intendedRunCount"], 46_500)


if __name__ == "__main__":
    unittest.main()
