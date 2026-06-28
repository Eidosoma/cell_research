from __future__ import annotations

import unittest

from morphospace2d import (
    REGENERATION_SCHEMA_VERSION,
    RegenerationPolicySpec,
    apply_perturbation,
    audit_regeneration_policy_payload,
    build_organ_like_target,
    build_sorted_row_target,
    evaluate_morphospace_metrics,
    run_regeneration_benchmark,
    standard_perturbation_specs,
    standard_regeneration_policy_specs,
)


class TestE05Regeneration(unittest.TestCase):
    def test_standard_perturbations_cover_required_families(self) -> None:
        families = {spec.family for spec in standard_perturbation_specs()}
        required = {
            "contiguous_chunk_removal",
            "freeze_patch",
            "rotate_graft",
            "duplicate_region",
            "insert_foreign_patch",
        }
        self.assertTrue(required.issubset(families))

    def test_perturbation_masks_and_transforms_are_recorded(self) -> None:
        target = build_organ_like_target()
        for spec in standard_perturbation_specs():
            with self.subTest(perturbation=spec.perturbation_id):
                perturbed = apply_perturbation(target, spec, seed=808)
                record = perturbed.to_record(target)
                self.assertEqual(record["target_id"], target.target_id)
                if spec.family in {"contiguous_chunk_removal", "freeze_patch", "rotate_graft"}:
                    self.assertGreater(len(perturbed.mask_positions), 0)
                if spec.family in {"rotate_graft", "duplicate_region", "insert_foreign_patch"}:
                    self.assertNotEqual(perturbed.graft_transform["transform"], "none")

    def test_metrics_handle_missing_and_extra_cell_count_changes(self) -> None:
        target = build_organ_like_target()
        by_family = {spec.family: spec for spec in standard_perturbation_specs()}
        removed = apply_perturbation(target, by_family["contiguous_chunk_removal"], seed=1)
        inserted = apply_perturbation(target, by_family["insert_foreign_patch"], seed=1)

        removed_result = evaluate_morphospace_metrics(target, removed.observed_by_position)
        inserted_result = evaluate_morphospace_metrics(target, inserted.observed_by_position)
        removed_energy = next(metric for metric in removed_result["metrics"] if metric["metricId"] == "target_energy")
        inserted_energy = next(metric for metric in inserted_result["metrics"] if metric["metricId"] == "target_energy")

        self.assertGreater(removed_energy["detail"]["missingPenalty"], 0.0)
        self.assertGreater(inserted_energy["detail"]["extraPenalty"], 0.0)
        self.assertGreater(removed_result["compositeError"], 0.0)
        self.assertGreater(inserted_result["compositeError"], 0.0)

    def test_regeneration_policy_audit_rejects_full_masks_and_target_maps(self) -> None:
        for policy in standard_regeneration_policy_specs():
            with self.subTest(policy=policy.policy_id):
                audit = audit_regeneration_policy_payload(policy)
                self.assertTrue(audit["success"], audit)

        leaking = RegenerationPolicySpec(
            policy_id="bad_repair_oracle",
            family="invalid",
            description="invalid full-mask policy",
            rank_weight=0.0,
            parameters={"full_perturbation_mask": [(0, 0)], "target_map": {"0": "leak"}},
        )
        audit = audit_regeneration_policy_payload(leaking)
        self.assertFalse(audit["success"])
        self.assertGreaterEqual(audit["errorCount"], 1)

    def test_repair_run_logs_nonconservative_actions_and_valid_metrics(self) -> None:
        target = build_sorted_row_target()
        perturbation = next(spec for spec in standard_perturbation_specs() if spec.family == "contiguous_chunk_removal")
        policy = next(policy for policy in standard_regeneration_policy_specs() if policy.policy_id == "classic_growth_prune_rank_repair")
        row, trace_rows, perturbed = run_regeneration_benchmark(
            target,
            perturbation,
            policy,
            seed=909,
            max_steps=300,
            snapshot_interval=50,
        )
        self.assertEqual(row["schema_version"], REGENERATION_SCHEMA_VERSION)
        self.assertTrue(row["metric_handling_finite"])
        self.assertGreater(row["initial_missing_position_count"], 0)
        self.assertGreaterEqual(row["divide_count"], 1)
        self.assertGreater(len(trace_rows), 1)
        self.assertLessEqual(row["final_missing_position_count"], row["initial_missing_position_count"])
        self.assertGreater(len(perturbed.mask_positions), 0)

    def test_frozen_patch_does_not_move_frozen_cells(self) -> None:
        target = build_sorted_row_target()
        perturbation = next(spec for spec in standard_perturbation_specs() if spec.family == "freeze_patch")
        policy = next(policy for policy in standard_regeneration_policy_specs() if policy.policy_id == "random_local_repair_null")
        row, _, _ = run_regeneration_benchmark(target, perturbation, policy, seed=7, max_steps=100, snapshot_interval=50)
        self.assertEqual(row["frozen_violation_count"], 0)
        self.assertEqual(row["collision_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
