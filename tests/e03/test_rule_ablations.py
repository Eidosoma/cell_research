"""S12 targeted rule-ablation tests."""

from __future__ import annotations

import unittest

import pandas as pd

from src.e03.coarse_sweep import SweepConfig
from src.e03.rule_ablations import (
    aggregate_ablation_pairs,
    apply_ablation,
    build_ablation_variants,
    claim_frame,
    component_profile,
    matched_seed_validation,
    policy_record_for_variant,
    run_cpu_policy_config_matched,
    verify_ablation_change,
)
from src.e03.rule_dsl import DSLPolicy, parse_policy


RICH_POLICY = """policy rich_ablation_fixture v1
state ideal_position=left_boundary
rule if random_lt(0.5) and prefix_sorted and target_exists(right) and self_gt(right) then remember(flag, yes), signal(ready), compare(right), choose(0.5, swap(right), wait)
rule if target_exists(ideal) and not_at_ideal and self_lt(ideal) then compare(ideal), swap(ideal)
rule if always then estimate_target_position(next)
rule else wait
end
"""


class RuleAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = parse_policy(RICH_POLICY)

    def test_component_profile_counts_targeted_primitives(self) -> None:
        profile = component_profile(self.policy)
        self.assertEqual(profile["random_guard_count"], 1)
        self.assertEqual(profile["prefix_guard_count"], 1)
        self.assertEqual(profile["probabilistic_action_count"], 1)
        self.assertGreaterEqual(profile["state_action_count"], 1)
        self.assertGreaterEqual(profile["memory_signal_action_count"], 2)
        self.assertGreaterEqual(profile["swap_action_count"], 2)

    def test_each_declared_ablation_verifies_intended_source_change(self) -> None:
        for ablation_id in (
            "drop_random_guards",
            "determinize_choose_first",
            "drop_prefix_sorted_guards",
            "disable_target_position_state",
            "drop_memory_signal_actions",
            "drop_compare_actions",
            "drop_swap_actions",
        ):
            with self.subTest(ablation_id=ablation_id):
                variant = apply_ablation(self.policy, ablation_id)
                self.assertIsInstance(variant, DSLPolicy)
                verification = verify_ablation_change(self.policy, variant, ablation_id)
                self.assertTrue(verification["success"], verification)

    def test_build_ablation_variants_includes_verified_metadata(self) -> None:
        variants = build_ablation_variants(
            policy=self.policy,
            original_policy_id="dsl:unit",
            original_policy_name="rich",
            class_id="UC99",
            cautious_label="unit",
            exemplar_roles="unit_fixture",
            selection_reason="unit",
            code_source="unit",
        )
        self.assertGreaterEqual(len(variants), 6)
        self.assertTrue(all(variant.source_verification["success"] for variant in variants))

    def test_matched_runner_uses_same_config_seed_set_for_ablation_pairs(self) -> None:
        variant = apply_ablation(self.policy, "drop_compare_actions")
        config = SweepConfig("unit_s12_n5", "screen", "heldout", 5, 123, 8)
        original_record = policy_record_for_variant(self.policy, policy_id="dsl:unit", policy_name="rich", source_kind="unit")
        variant_record = policy_record_for_variant(variant, source_kind="unit_ablation")
        rows = [
            run_cpu_policy_config_matched(
                original_record,
                config,
                original_policy_id="dsl:unit",
                variant_kind="original",
                ablation_id="original",
                ablation_component="none",
            ),
            run_cpu_policy_config_matched(
                variant_record,
                config,
                original_policy_id="dsl:unit",
                variant_kind="ablation",
                ablation_id="drop_compare_actions",
                ablation_component="neighbor_compare",
            ),
        ]
        run_df = pd.DataFrame(rows)
        self.assertTrue(matched_seed_validation(run_df))
        self.assertEqual(run_df["matched_rng_seed"].nunique(), 1)

    def test_aggregate_and_claim_frame_classifies_large_drop(self) -> None:
        run_df = pd.DataFrame(
            [
                {
                    "original_policy_id": "p1",
                    "variant_policy_id": "p1",
                    "variant_kind": "original",
                    "ablation_id": "original",
                    "ablation_component": "none",
                    "config_id": "c1",
                    "seed": 1,
                    "array_size": 8,
                    "split": "heldout",
                    "heldout": True,
                    "matched_rng_seed": 7,
                    "final_inversion_sortedness": 0.9,
                    "inversion_sortedness_delta": 0.5,
                    "final_is_sorted": True,
                    "work_count": 10,
                    "compare_count": 5,
                    "swap_count": 5,
                    "update_count": 0,
                    "wait_count": 0,
                    "invalid": False,
                    "timed_out": False,
                },
                {
                    "original_policy_id": "p1",
                    "variant_policy_id": "p1_ab",
                    "variant_kind": "ablation",
                    "ablation_id": "drop_swap_actions",
                    "ablation_component": "neighbor_exchange",
                    "config_id": "c1",
                    "seed": 1,
                    "array_size": 8,
                    "split": "heldout",
                    "heldout": True,
                    "matched_rng_seed": 7,
                    "final_inversion_sortedness": 0.65,
                    "inversion_sortedness_delta": 0.25,
                    "final_is_sorted": False,
                    "work_count": 5,
                    "compare_count": 5,
                    "swap_count": 0,
                    "update_count": 0,
                    "wait_count": 0,
                    "invalid": False,
                    "timed_out": True,
                },
            ]
        )
        meta = pd.DataFrame(
            [
                {
                    "original_policy_id": "p1",
                    "variant_policy_id": "p1_ab",
                    "original_policy_name": "orig",
                    "variant_policy_name": "ablated",
                    "class_id": "UC01",
                    "cautious_label": "unit",
                    "exemplar_roles": "unit",
                    "selection_reason": "unit",
                    "ablation_description": "unit",
                    "source_verification_success": True,
                    "source_verification_notes": "",
                    "changed_key_count": 1,
                }
            ]
        )
        aggregate = aggregate_ablation_pairs(run_df, meta)
        claims = claim_frame(aggregate)
        self.assertEqual(claims["claim_type"].iloc[0], "local_necessary_candidate")


if __name__ == "__main__":
    unittest.main()
