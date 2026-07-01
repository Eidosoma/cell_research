"""Focused tests for E02 S08 matched DG nulls."""

from __future__ import annotations

import unittest

from scripts.e02_s08_dg_nulls import (
    bridge_counts,
    build_matched_bridge,
    dg_from_monotonicity_error,
    dg_from_sortedness,
    total_step_drop,
    total_step_recovery,
    validate_toy_cases,
)


class DgNullTests(unittest.TestCase):
    def test_toy_dg_cases_pass(self) -> None:
        self.assertTrue(validate_toy_cases()["toyCaseValidationPassed"])

    def test_sortedness_and_error_dg_agree(self) -> None:
        sortedness = [50.0, 45.0, 60.0, 55.0, 80.0]
        error = [100.0 - value for value in sortedness]
        dg = dg_from_sortedness(sortedness)
        error_dg = dg_from_monotonicity_error(error)
        self.assertAlmostEqual(dg["dg_primary"], error_dg["dg_primary_from_monotonicity_error"])
        self.assertEqual(dg["dg_event_count"], error_dg["dg_event_count_from_monotonicity_error"])

    def test_bridge_counts_match_net_and_drop(self) -> None:
        down, up, zero = bridge_counts(start=50.0, end=80.0, swap_count=100, total_drop=20.0)
        self.assertEqual(down, 20)
        self.assertEqual(up, 50)
        self.assertEqual(zero, 30)

    def test_matched_bridge_exactly_preserves_constraints(self) -> None:
        base = [50, 49, 50, 51, 50, 51, 52, 53, 54, 55, 56]
        bridge, attempts = build_matched_bridge(
            start=50.0,
            end=56.0,
            swap_count=10,
            total_drop=2.0,
            base_sortedness_values=base,
            seed=123,
        )
        self.assertGreaterEqual(attempts, 1)
        self.assertEqual(len(bridge), 11)
        self.assertEqual(bridge[0], 50.0)
        self.assertEqual(bridge[-1], 56.0)
        self.assertEqual(total_step_drop(bridge), 2.0)
        self.assertEqual(total_step_recovery(bridge), 8.0)
        self.assertGreaterEqual(min(bridge), 0.0)
        self.assertLessEqual(max(bridge), 100.0)


if __name__ == "__main__":
    unittest.main()
