from __future__ import annotations

import unittest

import numpy as np

from scripts.e02_s06_speed_matched_algotypes import (
    PureCalibration,
    aggregation_curve_for_assignment,
    movement_share_tolerance,
    run_speed_profile,
    speed_match_weights,
    target_activation_shares,
    target_movement_shares,
)


class TestE02SpeedMatchedAlgotypes(unittest.TestCase):
    def test_speed_match_weights_boost_slow_policy(self) -> None:
        calibrations = {
            "bubble": PureCalibration("bubble", True, "sorted", 100, 400, 400, 100.0, 0.25, 1.0, 0.0),
            "insertion": PureCalibration("insertion", True, "sorted", 50, 1000, 1000, 100.0, 0.05, 1.0, 0.0),
        }
        weights = speed_match_weights(calibrations, ["bubble", "insertion"] * 3)
        self.assertAlmostEqual(weights["bubble"], 1.0)
        self.assertGreater(weights["insertion"], weights["bubble"])
        self.assertAlmostEqual(weights["insertion"], 5.0)

    def test_target_activation_shares_use_counts_and_weights(self) -> None:
        shares = target_activation_shares(["bubble", "bubble", "insertion"], {"bubble": 1.0, "insertion": 4.0})
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertAlmostEqual(shares["bubble"], 2 / 6)
        self.assertAlmostEqual(shares["insertion"], 4 / 6)

    def test_target_movement_shares_follow_cell_counts(self) -> None:
        shares = target_movement_shares(["bubble", "bubble", "selection"])
        self.assertAlmostEqual(shares["bubble"], 2 / 3)
        self.assertAlmostEqual(shares["selection"], 1 / 3)

    def test_aggregation_curve_for_assignment_uses_cell_identity(self) -> None:
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

    def test_movement_share_tolerance_has_binomial_floor(self) -> None:
        self.assertAlmostEqual(movement_share_tolerance(10000), 0.15)
        self.assertGreater(movement_share_tolerance(4), 0.15)

    def test_small_speed_profile_sorts_and_tracks_movement(self) -> None:
        condition = {"conditionId": "unit", "mixtureId": "bubble_insertion", "inputProfile": "unique_random"}
        seed_row = {
            "replicateIndex": 0,
            "replicateNumber": 1,
            "inputPermutationSeed": 1,
            "algotypeAssignmentSeed": 2,
            "schedulerSeed": 3,
            "tieBreakerSeed": 4,
        }
        initial_values = [5, 1, 4, 2, 3, 0]
        initial_algotypes = ["bubble", "insertion", "bubble", "insertion", "bubble", "insertion"]
        calibrations = {
            "bubble": PureCalibration("bubble", True, "sorted", 8, 40, 40, 100.0, 0.2, 1.0, 0.0),
            "insertion": PureCalibration("insertion", True, "sorted", 6, 120, 120, 100.0, 0.05, 1.0, 0.0),
        }
        result = run_speed_profile(
            condition=condition,
            seed_row=seed_row,
            n=6,
            mixture_id="bubble_insertion",
            initial_values=initial_values,
            initial_algotypes=initial_algotypes,
            calibrations=calibrations,
            speed_profile_id="unit_speed",
            speed_profile_family="speed_matched",
            speed_weights_by_algotype={"bubble": 1.0, "insertion": 4.0},
            max_activations=10000,
            max_swaps=1000,
            max_comparisons=10000,
        )
        self.assertTrue(result.completed)
        self.assertEqual(result.stop_reason, "sorted")
        self.assertEqual(sum(result.cell_move_counts_by_algotype.values()), result.swap_count * 2)
        self.assertLessEqual(result.max_activation_share_abs_error, result.activation_share_tolerance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
