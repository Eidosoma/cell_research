"""S07 coarse sweep tests."""

from __future__ import annotations

import unittest

from src.e03.coarse_sweep import (
    PolicyRecord,
    SweepConfig,
    actor_schedule,
    policy_summary_frame,
    run_configs,
    run_cpu_policy_config,
    sortedness_metrics,
    validation_frame,
)
from src.e03.gpu_batch_simulator import compile_policy
from src.e03.rule_dsl import SIMPLE_SWAP_LEFT_POLICY, SIMPLE_SWAP_RIGHT_POLICY, parse_policy


def policy_record(source: str, *, route: str, source_kind: str = "unit") -> PolicyRecord:
    policy = parse_policy(source)
    compiled = compile_policy(policy)
    return PolicyRecord(
        policy=policy,
        policy_id=policy.policy_id,
        policy_name=policy.name,
        source_kind=source_kind,
        semantic_hash=policy.sha256,
        dsl_sha256=policy.sha256,
        features={},
        compiles_for_batch=True,
        requires_cpu_fallback=route == "cpu_fallback",
        fallback_reasons=("unit fallback",) if route == "cpu_fallback" else (),
        route=route,
    )


class CoarseSweepTests(unittest.TestCase):
    def test_sortedness_metrics_rank_sorted_and_reverse_arrays(self) -> None:
        sorted_metrics = sortedness_metrics((1, 2, 3, 4))
        reverse_metrics = sortedness_metrics((4, 3, 2, 1))
        self.assertTrue(sorted_metrics["is_sorted"])
        self.assertEqual(sorted_metrics["inversion_count"], 0)
        self.assertLess(reverse_metrics["inversion_sortedness"], sorted_metrics["inversion_sortedness"])
        self.assertEqual(reverse_metrics["adjacent_error_count"], 3)

    def test_actor_schedule_is_deterministic_scan(self) -> None:
        self.assertEqual(actor_schedule(4, 6, 1), (1, 2, 3, 0, 1, 2))
        self.assertEqual(actor_schedule(4, 6, 1), actor_schedule(4, 6, 1))

    def test_cpu_fallback_run_classifies_timeout_or_sorted(self) -> None:
        record = policy_record(SIMPLE_SWAP_LEFT_POLICY, route="cpu_fallback")
        config = SweepConfig("unit_n4", "screen", "heldout", 4, 11, 8)
        row = run_cpu_policy_config(record, config)
        self.assertEqual(row["route"], "cpu_fallback")
        self.assertIn(row["run_status"], {"ok_sorted", "timeout_event_cap"})
        self.assertFalse(row["invalid"])

    def test_mixed_route_tiny_sweep_outputs_validation_rows(self) -> None:
        records = [
            policy_record(SIMPLE_SWAP_LEFT_POLICY, route="jax_batch", source_kind="classic_dsl_seed"),
            policy_record(SIMPLE_SWAP_RIGHT_POLICY, route="cpu_fallback"),
        ]
        config = SweepConfig("unit_n5", "screen", "heldout", 5, 17, 6)
        run_df = run_configs(records, [config])
        summary = policy_summary_frame(run_df)
        validation = validation_frame(run_df, summary, policy_count=2, require_scale=False)
        self.assertEqual(len(run_df), 2)
        self.assertEqual(set(run_df["route"]), {"jax_batch", "cpu_fallback"})
        self.assertTrue(validation["success"].all(), validation.to_string(index=False))


if __name__ == "__main__":
    unittest.main()
