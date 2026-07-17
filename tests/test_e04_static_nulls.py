from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import numpy as np

from analysis.aggregation_metrics import aggregation_metrics
from analysis.static_nulls import (
    AUDIT_GRID,
    AUDIT_REPLICATES,
    GLOBAL_PROFILES,
    _bh_adjust,
    _support_size,
    _upper_tail_values,
    batch_metrics,
    sample_global_labels,
    sample_stratified_labels,
)


class StaticNullTests(unittest.TestCase):
    def test_global_sampler_preserves_counts(self) -> None:
        rng = np.random.Generator(np.random.PCG64DXSM(123))
        sampled = sample_global_labels((3, 2, 2), 2000, rng)
        for label, count in enumerate((3, 2, 2)):
            np.testing.assert_array_equal((sampled == label).sum(axis=1), count)

    def test_stratified_sampler_preserves_each_stratum(self) -> None:
        labels = np.asarray((0, 0, 1, 0, 1, 1), dtype=np.int8)
        strata = np.asarray((0, 0, 0, 1, 1, 1), dtype=np.int8)
        rng = np.random.Generator(np.random.PCG64DXSM(456))
        sampled = sample_stratified_labels(labels, strata, 2000, rng)
        for stratum in (0, 1):
            positions = strata == stratum
            expected = np.bincount(labels[positions], minlength=2)
            for label in (0, 1):
                np.testing.assert_array_equal(
                    (sampled[:, positions] == label).sum(axis=1), expected[label]
                )

    def test_exact_value_singletons_are_degenerate(self) -> None:
        labels = np.asarray((0, 1, 0, 1), dtype=np.int8)
        strata = np.arange(4)
        rng = np.random.Generator(np.random.PCG64DXSM(789))
        sampled = sample_stratified_labels(labels, strata, 100, rng)
        np.testing.assert_array_equal(sampled, np.broadcast_to(labels, sampled.shape))

    def test_batch_metrics_match_s04_scalar_metrics(self) -> None:
        arrangements = np.asarray(
            [[0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1]],
            dtype=np.int8,
        )
        values = (1, 1, 1, 2, 2, 2)
        batch = batch_metrics(arrangements, values)
        for index, row in enumerate(arrangements):
            scalar = aggregation_metrics(tuple(map(str, row)), values)
            for metric, observed in batch.items():
                expected = scalar[metric]
                if math.isnan(expected):
                    self.assertTrue(math.isnan(float(observed[index])))
                else:
                    self.assertAlmostEqual(float(observed[index]), expected, places=12)

    def test_contract_freezes_population_and_sampling(self) -> None:
        path = Path(__file__).resolve().parents[1] / "analysis/s05_static_null_contract.json"
        contract = json.loads(path.read_text())
        self.assertEqual(contract["researchStepId"], "S05")
        self.assertTrue(contract["frozenBeforePopulationAnalysis"])
        self.assertEqual(contract["sourcePopulation"]["conditionalAuditSnapshotCount"], 23_250)
        self.assertEqual(len(AUDIT_REPLICATES), 25)
        self.assertEqual(AUDIT_GRID, (0, 25, 50, 75, 100))
        self.assertEqual(len(GLOBAL_PROFILES), 7)

    def test_upper_tail_and_bh_are_monotone(self) -> None:
        reference = np.asarray((0.0, 0.0, 1.0, 2.0))
        conservative, randomized = _upper_tail_values(
            reference,
            np.asarray((0.0, 1.0, 2.0)),
            randomized_u=np.asarray((0.5, 0.5, 0.5)),
        )
        self.assertIsNotNone(randomized)
        np.testing.assert_allclose(conservative, (1.0, 0.6, 0.4))
        self.assertTrue(np.all(np.diff(conservative) <= 0))
        adjusted = _bh_adjust((0.03, 0.01, 0.20, np.nan))
        np.testing.assert_allclose(adjusted[:3], (0.045, 0.03, 0.2))
        self.assertTrue(math.isnan(adjusted[3]))

    def test_conditional_support_is_product_of_multinomials(self) -> None:
        labels = np.asarray((0, 0, 1, 0, 1, 1), dtype=np.int8)
        strata = np.asarray((0, 0, 0, 1, 1, 1), dtype=np.int8)
        self.assertEqual(_support_size(labels, strata), 9)


if __name__ == "__main__":
    unittest.main()
