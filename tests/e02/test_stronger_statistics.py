"""Focused tests for the E02 S14 stronger-statistics runner."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.e02_s14_stronger_statistics import (
    add_delta_rows,
    apply_fdr_and_interpretation,
    bh_adjust,
    blocked_group_range_permutation,
    paired_metric_delta,
    sign_flip_p_value,
)


class StrongerStatisticsTests(unittest.TestCase):
    def test_bh_adjust_is_monotone_and_handles_missing_values(self) -> None:
        adjusted = bh_adjust([0.01, None, 0.04, 0.20])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertIsNone(adjusted[1])
        self.assertGreaterEqual(adjusted[2], adjusted[0])
        self.assertAlmostEqual(adjusted[3], 0.20)

    def test_sign_flip_uses_exact_small_sample_and_detects_nonzero_mean(self) -> None:
        p_value, method, reps = sign_flip_p_value([1, 2, 3, 4], seed=1, reps=1000)
        self.assertEqual(method, "exact_sign_flip")
        self.assertEqual(reps, 16)
        self.assertLess(p_value, 0.20)

    def test_paired_metric_delta_aligns_by_repeat(self) -> None:
        frame = pd.DataFrame(
            {
                "condition_id": ["c1", "c1", "c1", "c1"],
                "repeat_index": [0, 0, 1, 1],
                "regime": ["ref", "alt", "ref", "alt"],
                "metric": [10.0, 13.0, 20.0, 15.0],
            }
        )
        deltas = paired_metric_delta(
            frame,
            id_cols=("condition_id", "repeat_index"),
            treatment_col="regime",
            reference="ref",
            treatment="alt",
            metric="metric",
        )
        self.assertCountEqual(np.round(deltas, 6).tolist(), [3.0, -5.0])

    def test_blocked_group_range_permutation_respects_blocks(self) -> None:
        frame = pd.DataFrame(
            {
                "block": ["a", "a", "b", "b", "c", "c"],
                "group": ["x", "y", "x", "y", "x", "y"],
                "metric": [1.0, 5.0, 2.0, 7.0, 3.0, 9.0],
            }
        )
        effect, ci_low, ci_high, p_value, method, reps, n_units = blocked_group_range_permutation(
            frame,
            block_col="block",
            group_col="group",
            value_col="metric",
            seed=2,
            reps=200,
        )
        self.assertEqual(method, "blocked_label_permutation")
        self.assertEqual(reps, 200)
        self.assertEqual(n_units, 3)
        self.assertGreater(effect, 0.0)
        self.assertLessEqual(ci_low, ci_high)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_delta_rows_receive_fdr_interpretation(self) -> None:
        rows = add_delta_rows(
            values=[5.0] * 10,
            source_step_id="S99",
            fdr_family="unit_family",
            claim_family="robustness",
            claim_id="unit_claim",
            metric="delta_metric",
            comparison="alt_minus_ref",
            prior_classification="stop_sensitive_final_state",
            context={"condition_id": "unit"},
            base_seed=3,
            reps=100,
            caveat="unit",
        )
        frame = apply_fdr_and_interpretation(pd.DataFrame(rows))
        self.assertEqual(frame.loc[0, "corrected_result"], "detected_after_family_fdr")
        self.assertEqual(frame.loc[0, "statistical_interpretation"], "corrected_constraining_signal")


if __name__ == "__main__":
    unittest.main()
