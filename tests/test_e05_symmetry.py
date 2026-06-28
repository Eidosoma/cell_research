from __future__ import annotations

import unittest

from morphospace2d import (
    SYMMETRY_SCHEMA_VERSION,
    SymmetryPolicySpec,
    audit_symmetry_policy_payload,
    axis_consistency_rows,
    make_symmetric_initial_state,
    run_symmetry_breaking_benchmark,
    standard_symmetry_policy_specs,
    standard_symmetry_task_specs,
    symmetry_policy_catalog_rows,
    symmetry_task_catalog_rows,
)


class TestE05SymmetryBreaking(unittest.TestCase):
    def test_standard_tasks_cover_axis_ring_and_asymmetry(self) -> None:
        tasks = standard_symmetry_task_specs()
        self.assertEqual({task.target_kind for task in tasks}, {"axis", "ring", "asymmetric_appendage"})
        rows = symmetry_task_catalog_rows(tasks)
        self.assertEqual(len(rows), len(tasks))
        self.assertTrue(all(row["task_record_json"] for row in rows))

    def test_initial_state_is_d4_symmetric_and_seed_logged(self) -> None:
        task = next(task for task in standard_symmetry_task_specs() if task.target_kind == "axis")
        initial = make_symmetric_initial_state(task, seed=1010)
        self.assertEqual(initial["seed"], 1010)
        self.assertTrue(initial["audit"]["success"], initial["audit"])
        self.assertEqual(initial["audit"]["transformCount"], 8)
        self.assertEqual(initial["audit"]["labelCounts"], {"neutral": task.grid_size * task.grid_size})

    def test_policy_audit_rejects_local_axis_or_target_map_leakage(self) -> None:
        policies = standard_symmetry_policy_specs()
        for policy in policies:
            with self.subTest(policy=policy.policy_id):
                audit = audit_symmetry_policy_payload(policy)
                self.assertTrue(audit["success"], audit)

        leaking = SymmetryPolicySpec(
            policy_id="bad_axis_oracle",
            family="invalid",
            description="invalid hidden-axis local policy",
            information_scope="local_only",
            is_local_only=True,
            parameters={"global_axis": "x", "target_map": {"0": "oracle"}},
        )
        audit = audit_symmetry_policy_payload(leaking)
        self.assertFalse(audit["success"])
        self.assertGreaterEqual(audit["errorCount"], 1)

    def test_policy_catalog_flags_global_baselines(self) -> None:
        rows = symmetry_policy_catalog_rows(standard_symmetry_policy_specs())
        baseline_rows = [row for row in rows if row["is_global_information_baseline"]]
        local_rows = [row for row in rows if row["is_local_only_policy"]]
        self.assertGreaterEqual(len(baseline_rows), 2)
        self.assertGreaterEqual(len(local_rows), 3)
        self.assertTrue(any(row["uses_organizer"] for row in baseline_rows))
        self.assertTrue(any(row["uses_global_gradient"] for row in baseline_rows))
        self.assertTrue(all(row["audit_success"] for row in rows))

    def test_symmetry_benchmark_logs_s10_fields(self) -> None:
        task = next(task for task in standard_symmetry_task_specs() if task.target_kind == "axis")
        policy = next(policy for policy in standard_symmetry_policy_specs() if policy.policy_id == "global_gradient_baseline")
        row, trace_rows, initial_audit = run_symmetry_breaking_benchmark(
            task,
            policy,
            seed=2020,
            max_steps=120,
            snapshot_interval=40,
        )
        self.assertEqual(row["schema_version"], SYMMETRY_SCHEMA_VERSION)
        self.assertEqual(row["research_step_id"], "S10")
        self.assertEqual(row["seed"], 2020)
        self.assertTrue(row["initial_symmetry_success"])
        self.assertTrue(initial_audit["success"])
        self.assertTrue(row["is_global_information_baseline"])
        self.assertFalse(row["uses_hidden_axis_leakage"])
        self.assertFalse(row["uses_target_map_leakage"])
        self.assertGreater(len(trace_rows), 1)
        self.assertTrue(all(trace["research_step_id"] == "S10" for trace in trace_rows))

    def test_axis_consistency_rows_summarize_seed_variance(self) -> None:
        task = next(task for task in standard_symmetry_task_specs() if task.target_kind == "axis")
        policy = next(policy for policy in standard_symmetry_policy_specs() if policy.policy_id == "local_polarity_noise_alignment")
        run_rows = [
            run_symmetry_breaking_benchmark(task, policy, seed=seed, max_steps=80, snapshot_interval=40)[0]
            for seed in (1, 2, 3)
        ]
        rows = axis_consistency_rows(run_rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_count"], 3)
        self.assertIn(rows[0]["dominant_selected_axis"], {"x", "y", "none"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
