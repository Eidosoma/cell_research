from __future__ import annotations

import unittest

import pandas as pd

from memory_repair import (
    CENTRALIZED_COMPARISON_VERSION,
    CENTRALIZED_ORACLE_CONTROLLER_ID,
    CENTRALIZED_PROXY_SCOPE_NOTE,
    CentralizedOracleRepairSimulator,
    build_s14_local_policies,
    build_s14_tasks,
    centralized_oracle_controller_spec,
    centralized_replay_fingerprint,
    fairness_assumptions,
    global_condition_rows_for_s14,
    global_oracle_baseline_protocol,
    local_condition_rows_for_s14,
    run_s14_global_condition,
    run_s14_local_condition,
    summarize_centralized_groups,
    summarize_local_global_deltas,
    validate_centralized_outputs,
)


class TestE04CentralizedComparison(unittest.TestCase):
    def test_global_oracle_controller_is_explicitly_labeled(self) -> None:
        spec = centralized_oracle_controller_spec()
        self.assertEqual(spec["policyId"], CENTRALIZED_ORACLE_CONTROLLER_ID)
        self.assertEqual(spec["controllerKind"], "global_oracle")
        self.assertTrue(spec["oracleAllowed"])
        self.assertTrue(spec["oracleBaseline"])
        self.assertTrue(spec["usesGlobalController"])
        self.assertTrue(spec["usesGlobalSortednessSignal"])
        self.assertTrue(spec["usesWholeArrayValues"])
        self.assertTrue(spec["usesWholeArrayRanks"])
        self.assertTrue(spec["usesWholeArrayTargetSignal"])
        self.assertTrue(spec["usesTargetPositionOracle"])
        protocol = global_oracle_baseline_protocol().to_dict()
        self.assertEqual(spec["trainingProtocol"]["mode"], protocol["mode"])
        self.assertEqual(spec["claimBoundary"], CENTRALIZED_PROXY_SCOPE_NOTE)

    def test_condition_rows_pair_local_and_global_by_task_seed(self) -> None:
        tasks = build_s14_tasks()
        policies = build_s14_local_policies()[:2]
        local_rows = local_condition_rows_for_s14(policies, tasks, seed_count=2, seed_base=18100)
        global_rows = global_condition_rows_for_s14(tasks, seed_count=2, seed_base=18100)
        df = pd.DataFrame(local_rows + global_rows)
        self.assertEqual(set(df["taskFamily"]), {"sorting", "repairable", "fatigue", "homeostasis"})
        self.assertFalse(df.loc[df["controllerKind"] == "local", "oracleAllowed"].any())
        self.assertFalse(df.loc[df["controllerKind"] == "local", "usesGlobalController"].any())
        self.assertTrue(df.loc[df["controllerKind"] == "local", "policyAuditSuccess"].all())
        global_df = df[df["controllerKind"] == "global_oracle"]
        self.assertTrue(global_df["oracleAllowed"].all())
        self.assertTrue(global_df["oracleBaseline"].all())
        self.assertTrue(global_df["usesGlobalController"].all())
        for _, group in df.groupby(["pairKey", "taskId", "seedIndex"], dropna=False):
            self.assertEqual(group["taskConfigHash"].nunique(dropna=False), 1)
            self.assertEqual(group["schedulerSeed"].nunique(dropna=False), 1)
            self.assertIn("global_oracle", set(group["controllerKind"]))
            self.assertIn("local", set(group["controllerKind"]))

    def test_global_oracle_replay_is_deterministic_and_repairs(self) -> None:
        task = build_s14_tasks()[1].to_dict()
        first = CentralizedOracleRepairSimulator(task, scheduler_seed=18400).run()[0]
        second = CentralizedOracleRepairSimulator(task, scheduler_seed=18400).run()[0]
        self.assertEqual(centralized_replay_fingerprint([first]), centralized_replay_fingerprint([second]))
        self.assertTrue(first["completed"], first)
        self.assertEqual(first["remainingFrozenCellCount"], 0)
        self.assertGreater(first["repairInterventionCount"], 0)
        self.assertTrue(first["oracleAllowed"])
        self.assertTrue(first["usesGlobalController"])
        self.assertEqual(first["centralizedComparisonVersion"], CENTRALIZED_COMPARISON_VERSION)

    def test_validation_accepts_complete_small_panel(self) -> None:
        tasks = build_s14_tasks()
        policies = build_s14_local_policies()[:1]
        local_conditions = local_condition_rows_for_s14(policies, tasks, seed_count=1, seed_base=18500)
        global_conditions = global_condition_rows_for_s14(tasks, seed_count=1, seed_base=18500)
        policy_by_id = {policy.policy_id: policy for policy in policies}
        results = []
        for condition in local_conditions:
            result, _ = run_s14_local_condition(policy_by_id[str(condition["policyId"])], condition)
            results.append(result)
        for condition in global_conditions:
            result, _ = run_s14_global_condition(condition)
            results.append(result)
        condition_df = pd.DataFrame(local_conditions + global_conditions)
        result_df = pd.DataFrame(results)
        delta_df = summarize_local_global_deltas(result_df)
        group_df = summarize_centralized_groups(delta_df)
        fairness_df = pd.DataFrame(fairness_assumptions())
        validation_df = validate_centralized_outputs(condition_df, result_df, delta_df, group_df, fairness_df)
        self.assertTrue(validation_df["success"].all(), validation_df.to_string(index=False))
        self.assertEqual(len(delta_df), len(local_conditions))
        self.assertTrue(delta_df["globalOracleExplicitlyLabeled"].all())
        self.assertTrue(delta_df["pairedByTaskAndSeed"].all())
        self.assertTrue(result_df["claimBoundary"].eq(CENTRALIZED_PROXY_SCOPE_NOTE).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
