"""E05 S08 regeneration perturbation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.regeneration import (
    PERTURBATION_TYPES,
    STUCK_STATUS,
    apply_regeneration_perturbation,
    identity_multiset_matches_target,
    run_regeneration_benchmark,
    select_patch_mask,
    simulate_regeneration_run,
)
from src.e05.targets import default_target_gallery, gradient_target, organ_like_target, stripes_target


class RegenerationTests(unittest.TestCase):
    def test_patch_mask_is_deterministic_and_nonempty(self) -> None:
        target = gradient_target(6, 4)
        first = select_patch_mask(target, "missing_patch", 2101)
        second = select_patch_mask(target, "missing_patch", 2101)

        self.assertEqual(first, second)
        self.assertGreater(len(first.site_ids), 0)
        self.assertEqual(len(first.site_ids), len(first.coordinates))

    def test_missing_patch_records_population_deficit_and_blocker(self) -> None:
        target = gradient_target(6, 4)
        perturbed = apply_regeneration_perturbation(target, "missing_patch", 2101)

        self.assertGreater(perturbed.perturbed_target_error, perturbed.baseline_target_error)
        self.assertEqual(perturbed.empty_site_count, len(perturbed.mask.site_ids))
        self.assertLess(perturbed.population_perturbed, perturbed.population_baseline)
        self.assertIn("birth", perturbed.semantic_blocker)
        self.assertFalse(perturbed.identity_multiset_matches_target)

    def test_frozen_patch_uses_stuck_cells_and_logs_unfreeze_blocker(self) -> None:
        target = stripes_target(6, 4)
        perturbed = apply_regeneration_perturbation(target, "frozen_patch", 2102)

        self.assertGreater(perturbed.perturbed_target_error, perturbed.baseline_target_error)
        self.assertEqual(perturbed.stuck_site_count, len(perturbed.mask.site_ids))
        self.assertTrue(all(perturbed.state.cell_at(site_id).status == STUCK_STATUS for site_id in perturbed.mask.site_ids))
        self.assertIn("unfreeze", perturbed.semantic_blocker)
        self.assertTrue(perturbed.identity_multiset_matches_target)

    def test_rotated_patch_is_population_conserving_and_swap_repairable(self) -> None:
        target = organ_like_target(8, 5)
        perturbed = apply_regeneration_perturbation(target, "rotated_patch", 2103)

        self.assertGreater(perturbed.perturbed_target_error, 0.0)
        self.assertEqual(perturbed.population_perturbed, perturbed.population_baseline)
        self.assertEqual(perturbed.semantic_blocker, "")
        self.assertTrue(identity_multiset_matches_target(target, perturbed.state))

    def test_duplicate_and_foreign_patches_break_identity_multiset(self) -> None:
        target = gradient_target(6, 4)
        donors = default_target_gallery()
        duplicate = apply_regeneration_perturbation(target, "duplicated_patch", 2101, donor_targets=donors)
        foreign = apply_regeneration_perturbation(target, "foreign_patch", 2101, donor_targets=donors)

        self.assertGreater(duplicate.perturbed_target_error, 0.0)
        self.assertFalse(duplicate.identity_multiset_matches_target)
        self.assertIn("duplicate", duplicate.semantic_blocker)
        self.assertGreater(foreign.perturbed_target_error, 0.0)
        self.assertFalse(foreign.identity_multiset_matches_target)
        self.assertTrue(foreign.donor_target_id)
        self.assertIn("foreign", foreign.semantic_blocker)

    def test_regeneration_run_logs_pre_post_errors_and_population(self) -> None:
        target = gradient_target(6, 4)
        result = simulate_regeneration_run(
            target=target,
            perturbation_type="rotated_patch",
            policy_id="s07_local_target_neighbor_descent",
            seed=2104,
            donor_targets=(target,),
            event_multiplier=10,
            records_per_run=5,
        )

        self.assertEqual(result.summary_row["research_step_id"], "S08")
        self.assertGreater(result.summary_row["pre_repair_target_error"], 0.0)
        self.assertIn("post_repair_target_error", result.summary_row)
        self.assertEqual(result.summary_row["population_delta_from_baseline"], 0)
        self.assertTrue(result.trace_rows)
        self.assertEqual(len(result.metric_rows), 16)

    def test_compact_benchmark_covers_all_perturbation_types(self) -> None:
        target = gradient_target(6, 4)
        summaries, traces, metrics, masks, _runs = run_regeneration_benchmark(
            targets=(target,),
            perturbation_types=PERTURBATION_TYPES,
            policy_ids=("s07_local_target_neighbor_descent",),
            seeds=(2101,),
            event_multiplier=5,
            records_per_run=3,
        )

        self.assertEqual(len(summaries), len(PERTURBATION_TYPES))
        self.assertEqual({row["perturbation_type"] for row in summaries}, set(PERTURBATION_TYPES))
        self.assertEqual(len(masks), len(PERTURBATION_TYPES))
        self.assertTrue(traces)
        self.assertEqual(len(metrics), len(PERTURBATION_TYPES) * 2 * 8)


if __name__ == "__main__":
    unittest.main()
