from __future__ import annotations

import unittest

from morphospace2d import (
    RECOVERY_SCHEMA_VERSION,
    RecoveryPolicySpec,
    audit_recovery_policy_payload,
    build_gradient_target,
    build_sorted_row_target,
    evaluate_morphospace_metrics,
    run_recovery_benchmark,
    scramble_target_state,
    standard_recovery_policy_specs,
)


class TestE05ScrambledRecovery(unittest.TestCase):
    def test_scramble_preserves_cells_and_logs_severity(self) -> None:
        target = build_gradient_target()
        state, severity = scramble_target_state(target, seed=17)
        self.assertEqual(set(state), set(target.substrate.nodes))
        self.assertEqual(
            sorted(identity.cell_id for identity in state.values()),
            sorted(identity.cell_id for identity in target.target_state().values()),
        )
        self.assertGreater(severity["movedFraction"], 0.9)
        self.assertGreater(severity["initialCompositeError"], 0.0)
        self.assertIn(severity["severityClass"], {"moderate", "severe"})

    def test_target_state_remains_zero_error_control(self) -> None:
        target = build_sorted_row_target()
        result = evaluate_morphospace_metrics(target, target.target_state())
        self.assertAlmostEqual(result["compositeError"], 0.0)
        for metric in result["metrics"]:
            self.assertAlmostEqual(metric["normalizedValue"], 0.0)

    def test_standard_policy_audits_reject_whole_target_payloads(self) -> None:
        for policy in standard_recovery_policy_specs():
            with self.subTest(policy=policy.policy_id):
                audit = audit_recovery_policy_payload(policy)
                self.assertTrue(audit["success"], audit)

        leaking = RecoveryPolicySpec(
            policy_id="bad_oracle",
            family="invalid",
            description="invalid leaking payload",
            rank_weight=0.0,
            parameters={"target_map": {"0": "leak"}},
        )
        audit = audit_recovery_policy_payload(leaking)
        self.assertFalse(audit["success"])
        self.assertGreater(audit["errorCount"], 0)

    def test_recovery_run_keeps_occupancy_and_cell_ids_conserved(self) -> None:
        target = build_sorted_row_target()
        policy = next(policy for policy in standard_recovery_policy_specs() if policy.policy_id == "classic_scalar_rank_swap")
        row, trace, severity = run_recovery_benchmark(
            target,
            policy,
            seed=23,
            max_steps=500,
            snapshot_interval=50,
        )
        self.assertEqual(row["schema_version"], RECOVERY_SCHEMA_VERSION)
        self.assertTrue(row["occupancy_preserved"])
        self.assertTrue(row["cell_ids_preserved"])
        self.assertEqual(row["collision_count"], 0)
        self.assertEqual(row["birth_count"], 0)
        self.assertEqual(row["death_count"], 0)
        self.assertGreaterEqual(row["relative_error_reduction"], 0.0)
        self.assertGreater(len(trace), 1)
        self.assertGreater(severity["movedFraction"], 0.9)

    def test_random_null_is_marked_as_nonleaking_conservative_baseline(self) -> None:
        target = build_sorted_row_target()
        policy = next(policy for policy in standard_recovery_policy_specs() if policy.policy_id == "random_local_swap_null")
        row, _, _ = run_recovery_benchmark(target, policy, seed=31, max_steps=20, snapshot_interval=10)
        self.assertFalse(row["uses_whole_target_leakage"])
        self.assertEqual(row["conservation_mode"], "conservative_adjacent_swaps_only")
        self.assertTrue(row["occupancy_preserved"])
        self.assertTrue(row["cell_ids_preserved"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
