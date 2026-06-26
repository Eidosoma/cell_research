from __future__ import annotations

import unittest

from scripts.e02_s09_alternative_metrics import state_distance_metrics
from scripts.e02_s10_input_distributions import (
    PROFILE_IDS,
    balanced_algotypes,
    input_values_for_profile,
    stable_seed,
    stress_conditions,
    validate_generators,
)


class TestE02InputDistributions(unittest.TestCase):
    def test_all_profiles_are_seed_replayable(self) -> None:
        n = 30
        for profile in PROFILE_IDS:
            seed = stable_seed("unit", profile)
            with self.subTest(profile=profile):
                self.assertEqual(input_values_for_profile(profile, n, seed), input_values_for_profile(profile, n, seed))
                self.assertEqual(len(input_values_for_profile(profile, n, seed)), n)

    def test_profile_specific_invariants(self) -> None:
        n = 30
        random_unique = input_values_for_profile("random_unique", n, 1)
        self.assertEqual(sorted(random_unique), list(range(1, n + 1)))
        self.assertEqual(len(set(random_unique)), n)

        reverse_unique = input_values_for_profile("reverse_unique", n, 1)
        self.assertEqual(reverse_unique, list(range(n, 0, -1)))
        self.assertEqual(state_distance_metrics(reverse_unique)["sortednessPercent"], 0.0)

        nearly_sorted = input_values_for_profile("nearly_sorted_unique", n, 1)
        self.assertGreaterEqual(state_distance_metrics(nearly_sorted)["sortednessPercent"], 75.0)

        duplicate_heavy = input_values_for_profile("duplicate_heavy", n, 1)
        self.assertLess(len(set(duplicate_heavy)), n)
        sorted_duplicate_metrics = state_distance_metrics(sorted(duplicate_heavy))
        self.assertEqual(sorted_duplicate_metrics["kendallTauDistanceNormalized"], 0.0)
        self.assertEqual(sorted_duplicate_metrics["earthMoverPositionDistanceNormalized"], 0.0)

        heavy_tailed = input_values_for_profile("heavy_tailed", n, 1)
        self.assertGreaterEqual(max(heavy_tailed.count(value) for value in set(heavy_tailed)), n // 3)

        motif = input_values_for_profile("local_repeated_motifs", n, 1)
        self.assertLessEqual(len(set(motif)), 4)

    def test_balanced_chimera_assignment_preserves_counts(self) -> None:
        algotypes = balanced_algotypes(("bubble", "insertion", "selection"), 30, 123)
        counts = {name: algotypes.count(name) for name in set(algotypes)}
        self.assertEqual(counts, {"bubble": 10, "insertion": 10, "selection": 10})
        self.assertEqual(balanced_algotypes(("bubble",), 5, 123), ["bubble"] * 5)

    def test_condition_matrix_shape(self) -> None:
        conditions = stress_conditions()
        self.assertEqual(len(conditions), 10)
        self.assertEqual({condition.condition_class for condition in conditions}, {"pure_no_frozen", "same_goal_chimera", "frozen_cell"})
        self.assertEqual(len(PROFILE_IDS) * len(conditions) * 3, 210)

    def test_generator_validation_table_passes(self) -> None:
        profile_df, checks_df = validate_generators(30)
        self.assertEqual(len(profile_df), len(PROFILE_IDS))
        self.assertTrue(checks_df["passed"].all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
