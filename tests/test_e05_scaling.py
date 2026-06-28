from __future__ import annotations

import unittest

from morphospace2d import (
    RecoveryPolicySpec,
    SCALING_SCHEMA_VERSION,
    audit_scaling_policy_payload,
    build_gradient_target,
    build_scaled_target,
    build_scaling_target_panel,
    metric_normalization_rows,
    run_scaling_recovery_benchmark,
    standard_recovery_policy_specs,
    standard_scaling_target_specs,
    target_scaling_rule_rows,
)


class TestE05Scaling(unittest.TestCase):
    def test_scaled_target_specs_isolate_training_and_heldout_sizes(self) -> None:
        specs = standard_scaling_target_specs()
        motifs = {spec.motif for spec in specs}
        self.assertEqual(motifs, {"gradient", "stripes", "ring", "boundary", "organ_like"})
        for motif in motifs:
            motif_specs = [spec for spec in specs if spec.motif == motif]
            self.assertEqual({spec.size_class for spec in motif_specs}, {"training_small", "heldout_medium", "heldout_large"})
            self.assertFalse(next(spec for spec in motif_specs if spec.size_class == "training_small").is_heldout_size)
            self.assertTrue(all(spec.is_heldout_size for spec in motif_specs if spec.size_class.startswith("heldout_")))
            self.assertTrue(all(spec.scale_rule for spec in motif_specs))

    def test_scaled_target_builders_encode_actual_size(self) -> None:
        default = build_gradient_target()
        larger = build_gradient_target(width=8, height=6)
        self.assertEqual(default.target_id, "gradient_x_5x4")
        self.assertEqual(larger.target_id, "gradient_x_8x6")
        self.assertNotEqual(default.target_id, larger.target_id)

    def test_scaling_rule_rows_document_each_target(self) -> None:
        panel = build_scaling_target_panel()
        rows = target_scaling_rule_rows(panel)
        self.assertEqual(len(rows), len(panel))
        self.assertTrue(all(row["scaleRule"] for row in rows))
        self.assertTrue(all(row["target_record_hash"] for row in rows))

    def test_policy_audit_rejects_hidden_size_and_target_map(self) -> None:
        for policy in standard_recovery_policy_specs():
            audit = audit_scaling_policy_payload(policy)
            self.assertTrue(audit["success"], audit)

        leaking = RecoveryPolicySpec(
            policy_id="bad_scaled_oracle",
            family="invalid",
            description="invalid hidden-size policy",
            rank_weight=0.0,
            parameters={"global_grid_size": [12, 10], "target_map": {"0": "oracle"}},
        )
        audit = audit_scaling_policy_payload(leaking)
        self.assertFalse(audit["success"])
        self.assertGreaterEqual(audit["errorCount"], 1)

    def test_metric_normalization_rows_pass_for_standard_panel(self) -> None:
        panel = build_scaling_target_panel()
        rows = metric_normalization_rows(panel, seed=9090)
        self.assertGreater(len(rows), len(panel))
        self.assertTrue(all(row["success"] for row in rows), [row for row in rows if not row["success"]][:3])

    def test_scaling_recovery_wrapper_marks_s09_and_heldout(self) -> None:
        spec = next(spec for spec in standard_scaling_target_specs() if spec.motif == "gradient" and spec.size_class == "heldout_medium")
        target = build_scaled_target(spec)
        policy = next(policy for policy in standard_recovery_policy_specs() if policy.policy_id == "classic_scalar_rank_swap")
        row, trace_rows, severity = run_scaling_recovery_benchmark(
            spec,
            target,
            policy,
            seed=9191,
            max_steps=200,
            snapshot_interval=100,
        )
        self.assertEqual(row["schema_version"], SCALING_SCHEMA_VERSION)
        self.assertEqual(row["research_step_id"], "S09")
        self.assertTrue(row["is_heldout_size"])
        self.assertEqual(row["training_size_class"], "training_small")
        self.assertFalse(row["uses_hidden_global_size"])
        self.assertFalse(row["uses_whole_target_leakage"])
        self.assertTrue(all(trace["research_step_id"] == "S09" for trace in trace_rows))
        self.assertEqual(severity["researchStepId"], "S09")


if __name__ == "__main__":
    unittest.main(verbosity=2)
