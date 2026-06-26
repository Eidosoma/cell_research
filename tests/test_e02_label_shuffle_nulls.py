from __future__ import annotations

import json
import unittest

import numpy as np

from e02_deterministic_simulator import state_hash
from scripts.e02_s04_label_shuffle_nulls import (
    aggregation_curve_for_assignment,
    empirical_p_high,
    normalized_auc,
    validate_fixed_trajectory,
)


class TestE02LabelShuffleNulls(unittest.TestCase):
    def test_assignment_curve_follows_fixed_cell_history(self) -> None:
        cell_ids_by_event = np.asarray(
            [
                [0, 1, 2, 3],
                [1, 0, 2, 3],
                [1, 2, 0, 3],
            ],
            dtype=np.int16,
        )
        labels_by_cell_id = np.asarray([0, 0, 1, 1], dtype=np.int16)
        curve = aggregation_curve_for_assignment(cell_ids_by_event, labels_by_cell_id)
        self.assertTrue(np.allclose(curve, [2 / 3, 2 / 3, 0.0]))

    def test_empirical_high_tail_p_value_is_plus_one_corrected(self) -> None:
        self.assertAlmostEqual(empirical_p_high(0.75, [0.1, 0.5, 0.8]), 0.5)
        self.assertAlmostEqual(empirical_p_high(0.9, [0.1, 0.5, 0.8]), 0.25)

    def test_normalized_auc_handles_short_curves(self) -> None:
        self.assertAlmostEqual(normalized_auc(np.asarray([0.25])), 0.25)
        self.assertAlmostEqual(normalized_auc(np.asarray([0.0, 1.0, 0.0])), 0.5)

    def test_fixed_trajectory_validation_accepts_identity_consistent_rows(self) -> None:
        initial_values = [4, 1, 3, 2]
        initial_algotypes = ["bubble", "insertion", "bubble", "insertion"]
        rows = [
            {
                "event_index": 0,
                "values_json": json.dumps([4, 1, 3, 2], separators=(",", ":")),
                "cell_ids_json": json.dumps([0, 1, 2, 3], separators=(",", ":")),
                "algotypes_json": json.dumps(["bubble", "insertion", "bubble", "insertion"], separators=(",", ":")),
                "state_hash": state_hash([4, 1, 3, 2]),
            },
            {
                "event_index": 1,
                "values_json": json.dumps([1, 4, 3, 2], separators=(",", ":")),
                "cell_ids_json": json.dumps([1, 0, 2, 3], separators=(",", ":")),
                "algotypes_json": json.dumps(["insertion", "bubble", "bubble", "insertion"], separators=(",", ":")),
                "state_hash": state_hash([1, 4, 3, 2]),
            },
        ]
        self.assertEqual(
            validate_fixed_trajectory(rows, initial_values=initial_values, initial_algotypes=initial_algotypes),
            [],
        )

    def test_fixed_trajectory_validation_rejects_changed_value_history(self) -> None:
        initial_values = [4, 1, 3, 2]
        initial_algotypes = ["bubble", "insertion", "bubble", "insertion"]
        rows = [
            {
                "event_index": 0,
                "values_json": json.dumps([4, 1, 3, 2], separators=(",", ":")),
                "cell_ids_json": json.dumps([0, 1, 2, 3], separators=(",", ":")),
                "algotypes_json": json.dumps(["bubble", "insertion", "bubble", "insertion"], separators=(",", ":")),
                "state_hash": state_hash([4, 1, 3, 2]),
            },
            {
                "event_index": 1,
                "values_json": json.dumps([1, 4, 2, 3], separators=(",", ":")),
                "cell_ids_json": json.dumps([1, 0, 2, 3], separators=(",", ":")),
                "algotypes_json": json.dumps(["insertion", "bubble", "bubble", "insertion"], separators=(",", ":")),
                "state_hash": state_hash([1, 4, 2, 3]),
            },
        ]
        failures = validate_fixed_trajectory(rows, initial_values=initial_values, initial_algotypes=initial_algotypes)
        self.assertTrue(any("values do not match" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main(verbosity=2)
