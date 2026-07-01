"""Focused tests for the E02 S04 label-shuffle null helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.e02_s04_label_shuffle_nulls import (
    REQUIRED_TRACE_COLUMNS,
    aggregation_left_neighbor_percent_from_code,
    code_counts,
    composition_key_from_counts,
    max_distribution_stats,
    simulate_row_null_distribution,
    validate_trace_columns,
)


class LabelShuffleNullTests(unittest.TestCase):
    def test_aggregation_from_code_uses_e01_denominator(self) -> None:
        self.assertEqual(aggregation_left_neighbor_percent_from_code("AABB"), 50.0)
        self.assertEqual(aggregation_left_neighbor_percent_from_code("ABAB"), 0.0)
        self.assertEqual(aggregation_left_neighbor_percent_from_code("AAAA"), 75.0)

    def test_composition_key_ignores_label_names(self) -> None:
        self.assertEqual(composition_key_from_counts(code_counts("AABB")), (2, 2))
        self.assertEqual(composition_key_from_counts(code_counts("BBAA")), (2, 2))
        self.assertEqual(composition_key_from_counts(code_counts("AAABC")), (3, 1, 1))

    def test_simulated_null_distribution_shape_and_bounds(self) -> None:
        values = simulate_row_null_distribution((2, 2), 128, 123)
        self.assertEqual(len(values), 128)
        self.assertGreaterEqual(values.min(), 0.0)
        self.assertLessEqual(values.max(), 75.0)

    def test_max_distribution_detects_large_peak(self) -> None:
        row_values = simulate_row_null_distribution((2, 2), 2048, 456)
        stats = max_distribution_stats(row_values, row_count=10, observed_peak=75.0)
        self.assertIn("peak_empirical_p_upper", stats)
        self.assertGreaterEqual(stats["peak_empirical_p_upper"], 0.0)
        self.assertLessEqual(stats["peak_empirical_p_upper"], 1.0)

    def test_validate_trace_columns_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.parquet"
            pd.DataFrame({"condition_id": ["x"]}).to_parquet(path, index=False)
            ok, missing, available = validate_trace_columns(path)
            self.assertFalse(ok)
            self.assertIn("algotype_positions_code", missing)
            self.assertEqual(available, ["condition_id"])

    def test_validate_trace_columns_accepts_required_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.parquet"
            pd.DataFrame({col: [None] for col in REQUIRED_TRACE_COLUMNS}).to_parquet(path, index=False)
            ok, missing, _ = validate_trace_columns(path)
            self.assertTrue(ok)
            self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
