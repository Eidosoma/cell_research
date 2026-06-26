from __future__ import annotations

import unittest

from scripts.e02_s03_activation_rates import (
    algotypes_from_seed,
    rate_profiles_for_mixture,
    run_rate_profile,
    target_activation_shares,
)


class TestE02ActivationRates(unittest.TestCase):
    def test_rate_profile_targets_for_balanced_pair(self) -> None:
        algotypes = ["bubble"] * 5 + ["insertion"] * 5
        equalized = target_activation_shares(algotypes, {"bubble": 1.0, "insertion": 1.0})
        fast = target_activation_shares(algotypes, {"bubble": 3.0, "insertion": 1.0})
        throttled = target_activation_shares(algotypes, {"bubble": 1.0 / 3.0, "insertion": 1.0})
        self.assertAlmostEqual(equalized["bubble"], 0.5)
        self.assertAlmostEqual(fast["bubble"], 0.75)
        self.assertAlmostEqual(throttled["bubble"], 0.25)

    def test_profiles_include_equalized_fast_and_throttle(self) -> None:
        profiles = {profile["rateProfileId"]: profile for profile in rate_profiles_for_mixture("bubble_insertion")}
        self.assertIn("equalized_all_1x", profiles)
        self.assertIn("bubble_fast_3x", profiles)
        self.assertIn("insertion_fast_3x", profiles)
        self.assertIn("bubble_throttle_0p33x", profiles)
        self.assertIn("insertion_throttle_0p33x", profiles)

    def test_algotype_assignment_is_seeded(self) -> None:
        first = algotypes_from_seed("bubble_insertion_selection", 11, 77)
        second = algotypes_from_seed("bubble_insertion_selection", 11, 77)
        self.assertEqual(first, second)
        self.assertEqual(first.count("bubble"), 4)
        self.assertEqual(first.count("insertion"), 4)
        self.assertEqual(first.count("selection"), 3)

    def test_weighted_rate_profile_sorts_small_pair(self) -> None:
        result = run_rate_profile(
            initial_values=[4, 1, 3, 2],
            initial_algotypes=["bubble", "insertion", "bubble", "insertion"],
            condition_id="test_rate_profile",
            rate_profile_id="bubble_fast_3x",
            rate_weights={"bubble": 3.0, "insertion": 1.0},
            scheduler_seed=123,
            tie_breaker_seed=456,
            max_activations=10000,
            max_swaps=1000,
            max_comparisons=20000,
        )
        self.assertTrue(result.completed)
        self.assertEqual(result.final_values, [1, 2, 3, 4])
        self.assertGreater(result.activation_shares_by_algotype["bubble"], result.activation_shares_by_algotype["insertion"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
