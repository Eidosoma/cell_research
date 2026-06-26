from __future__ import annotations

import unittest

import numpy as np

from scripts.e01_s08_delayed_gratification import (
    delayed_gratification_from_sortedness as e01_delayed_gratification_from_sortedness,
)
from scripts.e02_s08_dg_nulls import (
    delayed_gratification_from_sortedness,
    delta_total_variation,
    raw_path_in_bounds,
    shuffled_delta_bridge,
    sortedness_percent_from_raw,
)


class TestE02DgNulls(unittest.TestCase):
    def test_dg_function_matches_e01_cases(self) -> None:
        trajectories = [
            [],
            [50.0],
            [40.0, 50.0, 60.0],
            [50.0, 60.0, 55.0, 70.0],
            [50.0, 60.0, 55.0, 58.0],
            [50.0, 50.0, 60.0, 60.0, 55.0, 55.0, 70.0],
            [60.0, 55.0, 70.0],
        ]
        for trajectory in trajectories:
            with self.subTest(trajectory=trajectory):
                observed = delayed_gratification_from_sortedness(trajectory)
                expected = e01_delayed_gratification_from_sortedness(trajectory)
                self.assertEqual(observed["dgEventCount"], expected["dgEventCount"])
                self.assertAlmostEqual(observed["delayedGratification"], expected["delayedGratification"])
                self.assertEqual(observed["dgSignedSegmentsJson"], expected["dgSignedSegmentsJson"])

    def test_dg_function_matches_e01_random_trajectory(self) -> None:
        rng = np.random.default_rng(123)
        raw = np.clip(np.cumsum(rng.integers(-2, 4, size=100)) + 15, 0, 29)
        trajectory = sortedness_percent_from_raw(raw.astype(np.int16), 30)
        observed = delayed_gratification_from_sortedness(trajectory)
        expected = e01_delayed_gratification_from_sortedness(trajectory)
        self.assertEqual(observed["dgEventCount"], expected["dgEventCount"])
        self.assertAlmostEqual(observed["delayedGratification"], expected["delayedGratification"])
        self.assertEqual(observed["dgSignedSegmentsJson"], expected["dgSignedSegmentsJson"])

    def test_shuffled_delta_bridge_preserves_matching_invariants(self) -> None:
        observed = np.asarray([10, 12, 11, 13, 13, 15, 14, 16], dtype=np.int16)
        rng = np.random.default_rng(42)
        null_path, info = shuffled_delta_bridge(observed, n=30, rng=rng, mix_multiplier=20)
        self.assertEqual(len(null_path), len(observed))
        self.assertEqual(int(null_path[0]), int(observed[0]))
        self.assertEqual(int(null_path[-1]), int(observed[-1]))
        self.assertEqual(delta_total_variation(np.diff(null_path)), delta_total_variation(np.diff(observed)))
        self.assertEqual(sorted(np.diff(null_path).tolist()), sorted(np.diff(observed).tolist()))
        self.assertTrue(raw_path_in_bounds(null_path, 30))
        self.assertGreaterEqual(info["mixAttemptCount"], info["mixAcceptedCount"])

    def test_shuffled_delta_bridge_handles_single_point(self) -> None:
        observed = np.asarray([7], dtype=np.int16)
        null_path, info = shuffled_delta_bridge(observed, n=30, rng=np.random.default_rng(1), mix_multiplier=5)
        self.assertTrue(np.array_equal(null_path, observed))
        self.assertTrue(info["nullEqualsObserved"])
        self.assertEqual(info["mixAttemptCount"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
