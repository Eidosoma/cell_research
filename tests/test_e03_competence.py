from __future__ import annotations

import math
import unittest

import pandas as pd

from e02_deterministic_simulator.metrics import monotonicity_error, sortedness_percent
from morphospace import (
    PolicyEventSimulator,
    competence_metric_specs,
    competence_schema,
    compute_competence_vector,
    delayed_gratification_from_sortedness,
    run_metric_unit_cases,
    state_distance_metrics,
    summarize_competence_vectors,
    toy_competence_examples,
    vector_from_summary_record,
)
from scripts.e02_s08_dg_nulls import delayed_gratification_from_sortedness as e02_dg


class TestE03CompetenceVector(unittest.TestCase):
    def test_schema_has_required_metric_contracts(self) -> None:
        schema = competence_schema()
        specs = {spec["metricId"]: spec for spec in schema["metricSpecs"]}
        required = {
            "completion_success",
            "final_sortedness_percent",
            "final_monotonicity_error",
            "swap_count",
            "comparison_count",
            "delayed_gratification",
            "aggregation_peak",
            "dominance_score",
            "transfer_score",
            "failure_or_oscillation",
        }
        self.assertTrue(required.issubset(specs))
        self.assertIn("missing", schema["missingValuePolicy"].lower())
        for spec in competence_metric_specs():
            with self.subTest(metric=spec.metric_id):
                self.assertIn(spec.direction, {"higher", "lower", "neutral", "contextual"})
                self.assertTrue(spec.normalization)
                self.assertTrue(spec.uncertainty_columns)

    def test_state_distance_metrics_match_e02_conventions(self) -> None:
        values = [2, 1, 2, 1]
        metrics = state_distance_metrics(values)
        self.assertAlmostEqual(metrics["sortednessPercent"], sortedness_percent(values))
        self.assertEqual(metrics["monotonicityError"], monotonicity_error(values))
        self.assertEqual(metrics["inversionCount"], 3)
        self.assertEqual(metrics["comparablePairCount"], 4)
        self.assertAlmostEqual(metrics["kendallTauDistanceNormalized"], 0.75)
        self.assertEqual(metrics["spearmanFootruleDistance"], 6)

    def test_dg_formula_matches_e02_port(self) -> None:
        trajectory = [50.0, 60.0, 55.0, 70.0, 65.0, 80.0]
        observed = delayed_gratification_from_sortedness(trajectory)
        expected = e02_dg(trajectory)
        self.assertAlmostEqual(observed["delayedGratification"], expected["delayedGratification"])
        self.assertEqual(observed["dgEventCount"], expected["dgEventCount"])
        self.assertEqual(observed["dgSignedSegmentsJson"], expected["dgSignedSegmentsJson"])

    def test_compute_vector_matches_simulation_result_fields(self) -> None:
        result = PolicyEventSimulator([3, 1, 2], "bubble", scheduler_seed=1, tie_breaker_seed=2).run()
        vector = compute_competence_vector(
            result,
            policy_id="classic_bubble",
            task_id="unit_bubble",
            task_panel="unit",
            input_profile="manual_unique",
        )
        self.assertEqual(vector["finalSortednessPercent"], result.final_sortedness_percent)
        self.assertEqual(vector["finalMonotonicityError"], result.final_monotonicity_error)
        self.assertEqual(vector["swapCount"], result.swap_count)
        self.assertEqual(vector["comparisonCount"], result.comparison_count)
        self.assertEqual(vector["completionSuccess"], 1.0)
        self.assertIn("robustnessScore", vector["missingMetricReasons"])

    def test_vector_from_summary_record_preserves_available_values(self) -> None:
        source = {
            "algorithm": "bubble",
            "condition_id": "summary_case",
            "n": 4,
            "completed": True,
            "stop_reason": "sorted",
            "swap_count": 6,
            "comparison_count": 7,
            "activation_count": 8,
            "final_sortedness_percent": 100.0,
            "final_monotonicity_error": 0,
            "final_aggregation": 1.0,
        }
        vector = vector_from_summary_record(source)
        self.assertEqual(vector["swapCount"], 6)
        self.assertEqual(vector["comparisonCount"], 7)
        self.assertEqual(vector["activationCount"], 8)
        self.assertEqual(vector["finalSortednessScore"], 1.0)
        self.assertEqual(vector["aggregationFinal"], 1.0)

    def test_toy_examples_and_uncertainty_summary(self) -> None:
        examples = toy_competence_examples()
        self.assertEqual(set(examples["taskPanel"]), {"toy_examples"})
        self.assertEqual(len(examples), 3)
        self.assertTrue(examples["vectorId"].is_unique)
        summary = summarize_competence_vectors(
            pd.concat([examples, examples], ignore_index=True),
            group_columns=("policyId", "taskPanel"),
            metric_columns=("finalSortednessScore", "completionSuccess"),
        )
        self.assertFalse(summary.empty)
        self.assertTrue((summary["uncertaintyN"] >= 2).any())
        self.assertTrue(summary["uncertaintyMethod"].str.contains("normal_approximation").all())

    def test_metric_unit_cases_pass(self) -> None:
        success, cases = run_metric_unit_cases()
        self.assertTrue(success, cases.to_string(index=False))
        self.assertTrue(cases["passed"].all())
        self.assertGreaterEqual(len(cases), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)

