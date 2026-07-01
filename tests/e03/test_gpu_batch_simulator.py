"""S06 batched simulator tests."""

from __future__ import annotations

import unittest

from src.e03.gpu_batch_simulator import (
    compare_batch_to_cpu,
    compile_policy,
    deterministic_validation_policies,
    jax_backend_summary,
    validation_fixtures,
)
from src.e03.rule_dsl import parse_policy


class GPUBatchSimulatorTests(unittest.TestCase):
    def test_jax_backend_is_available(self) -> None:
        summary = jax_backend_summary()
        self.assertTrue(summary["available"], summary)
        self.assertGreaterEqual(len(summary["devices"]), 1)

    def test_deterministic_batch_matches_cpu_reference(self) -> None:
        policies = deterministic_validation_policies()[:6]
        fixtures = validation_fixtures(len(policies))
        comparison = compare_batch_to_cpu(policies, fixtures, steps=3, seed=123)
        self.assertTrue(comparison["within_tolerance"].all(), comparison.to_string(index=False))
        self.assertTrue(comparison["final_state_match"].all())
        self.assertTrue(comparison["metrics_match"].all())

    def test_selection_target_position_is_batched_state(self) -> None:
        policy = parse_policy(
            """policy target_update_unit v1
state ideal_position=left_boundary
rule if target_exists(ideal) and not_at_ideal and target_movable(ideal) and self_ge(ideal) then compare(ideal), set_ideal(next)
rule else wait
end
"""
        )
        comparison = compare_batch_to_cpu([policy], validation_fixtures(1), steps=2, seed=5)
        self.assertTrue(bool(comparison.loc[0, "within_tolerance"]), comparison.to_string(index=False))
        self.assertEqual(int(comparison.loc[0, "cpu_ideal_position"]), int(comparison.loc[0, "gpu_ideal_position"]))

    def test_jax_cpu_backend_fallback_matches_cpu_reference(self) -> None:
        policy = deterministic_validation_policies()[0]
        fixture = validation_fixtures(1)[0]
        comparison = compare_batch_to_cpu([policy], [fixture], steps=2, seed=17, backend="cpu")
        self.assertTrue(bool(comparison.loc[0, "within_tolerance"]), comparison.to_string(index=False))
        self.assertEqual(comparison.loc[0, "backend"], "cpu")

    def test_fallback_reasons_cover_stochastic_and_metadata_primitives(self) -> None:
        stochastic = parse_policy(
            """policy stochastic_unit v1
state ideal_position=none
rule if random_lt(0.5) then choose(0.5, wait, wait)
end
"""
        )
        metadata = parse_policy(
            """policy metadata_unit v1
state ideal_position=none
rule if always then remember(flag, true), signal(cluster), wait
end
"""
        )
        deterministic = deterministic_validation_policies()[0]
        self.assertFalse(compile_policy(deterministic).requires_cpu_fallback)
        self.assertTrue(compile_policy(stochastic).requires_cpu_fallback)
        self.assertTrue(compile_policy(metadata).requires_cpu_fallback)


if __name__ == "__main__":
    unittest.main()
