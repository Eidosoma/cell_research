from __future__ import annotations

import unittest

import torch

from morphospace2d import (
    GPU_BATCH_SCHEMA_VERSION,
    GPUPolicySpec,
    audit_gpu_policy_payload,
    build_gpu_target_spec,
    cpu_gpu_validation_rows,
    gpu_policy_catalog_rows,
    gpu_sweep_summary_rows,
    gpu_target_catalog_rows,
    make_scrambled_label_batch,
    run_batched_gpu_sweep,
    seed_reproducibility_rows,
    standard_gpu_policy_specs,
    standard_gpu_sweep_targets,
    standard_gpu_validation_targets,
    torch_device_summary,
)


class TestE05GPUBatches(unittest.TestCase):
    def test_standard_gpu_targets_cover_core_motifs(self) -> None:
        targets = standard_gpu_sweep_targets()
        self.assertEqual({target.motif for target in targets}, {"gradient", "stripes", "ring", "boundary", "appendage"})
        rows = gpu_target_catalog_rows(targets)
        self.assertEqual(len(rows), len(targets))
        self.assertTrue(all(row["schema_version"] == GPU_BATCH_SCHEMA_VERSION for row in rows))
        self.assertTrue(all(row["target_hash"] for row in rows))

    def test_policy_audit_flags_target_map_baseline_and_rejects_local_leakage(self) -> None:
        policies = standard_gpu_policy_specs()
        rows = gpu_policy_catalog_rows(policies)
        self.assertTrue(all(row["audit_success"] for row in rows), rows)
        baselines = [row for row in rows if row["is_global_information_baseline"]]
        local = [row for row in rows if row["is_local_only_policy"]]
        self.assertGreaterEqual(len(local), 2)
        self.assertGreaterEqual(len(baselines), 1)
        self.assertTrue(any(row["uses_target_map"] for row in baselines))

        leaking = GPUPolicySpec(
            policy_id="bad_gpu_oracle",
            family="invalid",
            description="invalid local target-map access",
            information_scope="local_only",
            update_rule="neighbor_majority",
            is_local_only=True,
            parameters={"target_map": {"0": "oracle"}, "target_labels": [0, 1]},
        )
        audit = audit_gpu_policy_payload(leaking)
        self.assertFalse(audit["success"])
        self.assertGreaterEqual(audit["errorCount"], 1)

    def test_scrambled_batch_is_reproducible_on_cpu(self) -> None:
        target = build_gpu_target_spec("gradient", width=7, height=5)
        left = make_scrambled_label_batch(target, [11, 12, 13], device="cpu")
        right = make_scrambled_label_batch(target, [11, 12, 13], device="cpu")
        self.assertTrue(torch.equal(left, right))
        self.assertEqual(tuple(left.shape), (3, target.height, target.width))

    def test_batched_sweep_rows_and_summary_on_cpu(self) -> None:
        target = build_gpu_target_spec("ring", width=9, height=9)
        policy = next(policy for policy in standard_gpu_policy_specs() if policy.policy_id == "explicit_target_relaxation_gpu")
        rows, trace, finals = run_batched_gpu_sweep(
            target,
            policy,
            [101, 102],
            steps=8,
            device="cpu",
            trace_interval=4,
        )
        self.assertEqual(len(rows), 2)
        self.assertGreater(len(trace), len(rows))
        self.assertEqual(len(finals), 2)
        self.assertTrue(all(row["research_step_id"] == "S11" for row in rows))
        self.assertTrue(all(row["is_global_information_baseline"] for row in rows))
        summary = gpu_sweep_summary_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertGreaterEqual(summary[0]["relative_error_reduction_mean"], 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_gpu_validation_rows_pass_on_small_cases(self) -> None:
        target = standard_gpu_validation_targets()[0]
        policies = (
            next(policy for policy in standard_gpu_policy_specs() if policy.policy_id == "local_neighbor_majority_gpu"),
            next(policy for policy in standard_gpu_policy_specs() if policy.policy_id == "explicit_target_relaxation_gpu"),
        )
        rows = cpu_gpu_validation_rows([target], policies, [201, 202], steps=6, trace_interval=3, gpu_device="cuda")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["success"] for row in rows), rows)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_gpu_seed_reproducibility_rows_pass(self) -> None:
        target = standard_gpu_validation_targets()[0]
        policy = next(policy for policy in standard_gpu_policy_specs() if policy.policy_id == "local_inertia_smoothing_gpu")
        rows = seed_reproducibility_rows(target, policy, [301, 302], steps=6, trace_interval=3, device="cuda")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["success"] for row in rows), rows)
        summary = torch_device_summary("cuda")
        self.assertEqual(summary["selectedDevice"], "cuda")
        self.assertTrue(summary["cudaAvailable"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
