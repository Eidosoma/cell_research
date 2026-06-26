from __future__ import annotations

import json
import unittest

import numpy as np

from e02_deterministic_simulator import DeterministicEventSimulator
from scripts.e02_s05_dummy_algotypes import (
    aggregation_curve_for_labels,
    behavior_history_hash,
    dummy_labels_from_seed,
    fixed_count_expected_aggregation,
    label_codes,
    snapshot_trace_row,
    validate_trace_identity,
)


class TestE02DummyAlgotypes(unittest.TestCase):
    def test_dummy_labels_are_seeded_and_balanced(self) -> None:
        first = dummy_labels_from_seed("two_label_balanced", 11, 123)
        second = dummy_labels_from_seed("two_label_balanced", 11, 123)
        self.assertEqual(first, second)
        self.assertEqual(first.count("dummy_A"), 5)
        self.assertEqual(first.count("dummy_B"), 6)
        three = dummy_labels_from_seed("three_label_balanced", 11, 456)
        self.assertEqual(three.count("dummy_A"), 4)
        self.assertEqual(three.count("dummy_B"), 4)
        self.assertEqual(three.count("dummy_C"), 3)

    def test_fixed_count_expected_aggregation(self) -> None:
        self.assertAlmostEqual(fixed_count_expected_aggregation(["a", "a", "b", "b"]), 1 / 3)
        self.assertAlmostEqual(fixed_count_expected_aggregation(["a", "a", "a"]), 1.0)

    def test_aggregation_curve_uses_labels_by_cell_identity(self) -> None:
        cell_ids_by_event = np.asarray(
            [
                [0, 1, 2, 3],
                [1, 0, 2, 3],
                [1, 2, 0, 3],
            ],
            dtype=np.int16,
        )
        labels_by_cell_id = np.asarray([0, 0, 1, 1], dtype=np.int16)
        curve = aggregation_curve_for_labels(cell_ids_by_event, labels_by_cell_id)
        self.assertTrue(np.allclose(curve, [2 / 3, 2 / 3, 0.0]))

    def test_trace_identity_validation_rejects_label_drift(self) -> None:
        initial_values = [4, 1, 3, 2]
        labels = ["dummy_A", "dummy_B", "dummy_A", "dummy_B"]
        row = {
            "event_index": 0,
            "values_json": json.dumps([4, 1, 3, 2], separators=(",", ":")),
            "cell_ids_json": json.dumps([0, 1, 2, 3], separators=(",", ":")),
            "policy_algotypes_json": json.dumps(["bubble"] * 4, separators=(",", ":")),
            "dummy_labels_json": json.dumps(["dummy_A", "dummy_A", "dummy_B", "dummy_B"], separators=(",", ":")),
            "state_hash": "not-used-for-this-failure",
        }
        failures = validate_trace_identity([row], initial_values=initial_values, base_policy="bubble", dummy_labels_by_cell_id=labels)
        self.assertTrue(any("dummy labels do not follow cell identities" in failure for failure in failures))

    def test_behavior_hash_ignores_dummy_labels_when_policy_is_same(self) -> None:
        initial_values = [4, 1, 3, 2]
        first_labels = ["dummy_A", "dummy_B", "dummy_A", "dummy_B"]
        second_labels = ["dummy_B", "dummy_A", "dummy_B", "dummy_A"]
        rows = []
        for labels in [first_labels, second_labels]:
            sim = DeterministicEventSimulator(
                initial_values,
                ["bubble"] * 4,
                labels=label_codes(labels),
                scheduler_seed=7,
                tie_breaker_seed=11,
                research_step_id="S05",
            )
            rows.append(
                [
                    snapshot_trace_row(
                        sim,
                        sim.trace_rows[-1],
                        base_policy="bubble",
                        label_scheme_id="two_label_balanced",
                        dummy_labels_by_cell_id=labels,
                    )
                ]
            )
        self.assertEqual(behavior_history_hash(rows[0]), behavior_history_hash(rows[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
