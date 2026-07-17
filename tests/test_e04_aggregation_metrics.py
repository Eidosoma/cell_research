from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

from analysis.aggregation_metrics import (
    EXPECTED_GRID_ROWS,
    aggregation_metrics,
    hand_built_validation,
)


class AggregationMetricTests(unittest.TestCase):
    def test_original_denominators_and_run_identity(self) -> None:
        metrics = aggregation_metrics(tuple("AAABBB"), range(6))
        self.assertEqual(metrics["same_label_edge_count"], 4)
        self.assertAlmostEqual(metrics["publication_aggregation"], 4 / 6)
        self.assertAlmostEqual(metrics["edge_aggregation"], 4 / 5)
        self.assertEqual(metrics["run_count"], 2)
        self.assertEqual(
            metrics["run_count"], 6 - metrics["same_label_edge_count"]
        )

    def test_alternation_signs_mutual_information(self) -> None:
        metrics = aggregation_metrics(tuple("ABABAB"), range(6))
        self.assertAlmostEqual(metrics["categorical_assortativity"], -1)
        self.assertAlmostEqual(metrics["neighbor_mi_bits"], 1)
        self.assertAlmostEqual(metrics["neighbor_nmi"], 1)
        self.assertAlmostEqual(metrics["signed_neighbor_nmi"], -1)

    def test_homogeneous_undefined_convention(self) -> None:
        metrics = aggregation_metrics(tuple("AAAAA"), range(5))
        self.assertTrue(math.isnan(metrics["categorical_assortativity"]))
        self.assertTrue(math.isnan(metrics["neighbor_nmi"]))
        self.assertTrue(math.isnan(metrics["signed_neighbor_nmi"]))

    def test_unique_values_make_conditional_metric_undefined(self) -> None:
        metrics = aggregation_metrics(tuple("AABB"), (1, 2, 3, 4))
        self.assertEqual(metrics["equal_value_edge_count"], 0)
        self.assertTrue(math.isnan(metrics["equal_value_same_label_rate"]))
        self.assertTrue(math.isnan(metrics["equal_value_corrected_excess"]))

    def test_duplicate_conditioned_exact_baseline(self) -> None:
        metrics = aggregation_metrics(tuple("AABAAB"), (1, 1, 1, 2, 2, 2))
        self.assertEqual(metrics["equal_value_edge_count"], 4)
        self.assertEqual(metrics["equal_value_same_label_edge_count"], 2)
        self.assertAlmostEqual(metrics["equal_value_same_label_rate"], 1 / 2)
        self.assertAlmostEqual(metrics["equal_value_conditioned_baseline"], 1 / 3)
        self.assertAlmostEqual(metrics["equal_value_corrected_excess"], 1 / 6)

    def test_endpoint_degree_assortativity_is_explicit(self) -> None:
        metrics = aggregation_metrics(tuple("ABAABB"), range(6))
        # Stub marginals differ from global 3/3 composition on this open path.
        self.assertAlmostEqual(metrics["edge_aggregation"], 2 / 5)
        self.assertAlmostEqual(metrics["categorical_assortativity"], -0.2)

    def test_frozen_fixtures_pass(self) -> None:
        self.assertEqual(len(hand_built_validation()), 6)

    def test_contract_is_frozen_to_s04(self) -> None:
        path = Path(__file__).resolve().parents[1] / "analysis/s04_metric_contract.json"
        contract = json.loads(path.read_text())
        self.assertEqual(contract["researchStepId"], "S04")
        self.assertTrue(contract["frozenBeforePopulationAnalysis"])
        self.assertEqual(contract["population"]["scenarioCount"], 46_500)
        self.assertEqual(EXPECTED_GRID_ROWS, 4_696_500)


if __name__ == "__main__":
    unittest.main()
