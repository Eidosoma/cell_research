from __future__ import annotations

import unittest

from analysis.aggregation_baselines import (
    S01_MANIFEST_SHA256,
    aggregation_numerators,
    exact_pair_probability,
    exhaustive_validation,
    expected_cyclic_aggregation,
    expected_edge_aggregation,
    expected_paper_aggregation,
    paper_with_replacement_approximation,
    verify_s01_immutable,
    with_replacement_pair_probability,
)


class CompositionBaselineTests(unittest.TestCase):
    def test_balanced_pair_and_three_way_publication_expectations(self) -> None:
        self.assertAlmostEqual(expected_paper_aggregation((50, 50)), 0.49)
        self.assertAlmostEqual(expected_edge_aggregation((50, 50)), 49 / 99)
        self.assertAlmostEqual(expected_paper_aggregation((34, 33, 33)), 0.3234)
        self.assertAlmostEqual(
            expected_edge_aggregation((34, 33, 33)),
            (34 * 33 + 2 * 33 * 32) / (100 * 99),
        )

    def test_finite_n_with_replacement_gaps(self) -> None:
        counts = (5, 3, 2)
        n = sum(counts)
        exact = exact_pair_probability(counts)
        replacement = with_replacement_pair_probability(counts)
        self.assertAlmostEqual(replacement - exact, (1 - replacement) / (n - 1))
        self.assertAlmostEqual(
            paper_with_replacement_approximation(counts)
            - expected_paper_aggregation(counts),
            (1 - replacement) / n,
        )

    def test_paper_edge_and_cyclic_denominators(self) -> None:
        counts = (2, 1)
        self.assertAlmostEqual(expected_paper_aggregation(counts), 2 / 9)
        self.assertAlmostEqual(expected_edge_aggregation(counts), 1 / 3)
        self.assertAlmostEqual(expected_cyclic_aggregation(counts), 1 / 3)
        self.assertEqual(aggregation_numerators((0, 0, 1)), (1, 1))
        self.assertEqual(aggregation_numerators((0, 1, 0)), (0, 1))

    def test_exhaustive_small_fixture(self) -> None:
        frame, accounting = exhaustive_validation(max_n=5, max_k=3)
        self.assertTrue(accounting["allMultinomialCountsMatch"])
        self.assertLessEqual(frame.paper_abs_error.max(), 1e-15)
        self.assertLessEqual(frame.edge_abs_error.max(), 1e-15)
        self.assertLessEqual(frame.cyclic_abs_error.max(), 1e-15)

    def test_s01_is_immutable(self) -> None:
        audit = verify_s01_immutable()
        self.assertEqual(audit["expectedManifestSha256"], S01_MANIFEST_SHA256)
        self.assertTrue(audit["allPassed"], audit)


if __name__ == "__main__":
    unittest.main()
