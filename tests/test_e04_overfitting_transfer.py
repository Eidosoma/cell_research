from __future__ import annotations

import unittest

import pandas as pd

from memory_repair import (
    OVERFITTING_PROXY_SCOPE_NOTE,
    OVERFITTING_TRANSFER_VERSION,
    build_s13_policies,
    build_s13_tasks,
    condition_rows_for_transfer_policy,
    overfitting_replay_fingerprint,
    run_transfer_condition,
    select_policies_from_selection_results,
    summarize_transfer_gaps,
    transfer_group_summary,
    validate_overfitting_outputs,
)


class TestE04OverfittingTransfer(unittest.TestCase):
    def test_tasks_have_strict_selection_holdout_separation(self) -> None:
        tasks = [task.to_dict() for task in build_s13_tasks()]
        task_df = pd.DataFrame(tasks)
        self.assertEqual(set(task_df["splitRole"]), {"selection", "holdout"})
        selection = task_df[task_df["splitRole"] == "selection"]
        holdout = task_df[task_df["splitRole"] == "holdout"]
        for column in (
            "initialLength",
            "perturbationSignature",
            "frozenPlacementSignature",
            "recoveryThresholdSignature",
            "noiseRegime",
        ):
            with self.subTest(column=column):
                self.assertTrue(set(selection[column].map(str)).isdisjoint(set(holdout[column].map(str))))
        self.assertGreater(int(holdout["initialLength"].min()), int(selection["initialLength"].max()))
        self.assertTrue(task_df["claimBoundary"].str.contains("Direct computational generalization proxy only", regex=False).all())

    def test_condition_rows_mark_selection_and_holdout_without_oracle(self) -> None:
        policy = build_s13_policies()[0]
        rows = condition_rows_for_transfer_policy(policy, build_s13_tasks()[:4], seed_count=2, seed_base=16100)
        df = pd.DataFrame(rows)
        self.assertEqual(set(df["splitRole"]), {"selection", "holdout"})
        self.assertTrue(df.loc[df["splitRole"] == "selection", "selectionUsed"].all())
        self.assertFalse(df.loc[df["splitRole"] == "holdout", "selectionUsed"].any())
        self.assertTrue(df.loc[df["splitRole"] == "holdout", "heldOut"].all())
        self.assertFalse(df["oracleAccessAllowed"].any())
        self.assertTrue(df["policyAuditSuccess"].all())
        self.assertTrue(set(df.loc[df["splitRole"] == "selection", "seedIndex"]).isdisjoint(set(df.loc[df["splitRole"] == "holdout", "seedIndex"])))
        self.assertTrue(df["claimBoundary"].eq(OVERFITTING_PROXY_SCOPE_NOTE).all())

    def test_small_transfer_replay_is_deterministic_and_bounded(self) -> None:
        policy = build_s13_policies()[1]
        tasks = (build_s13_tasks()[0], build_s13_tasks()[3])
        rows = condition_rows_for_transfer_policy(policy, tasks, seed_count=1, seed_base=16200)
        first = [run_transfer_condition(policy, row)[0] for row in rows]
        second = [run_transfer_condition(policy, row)[0] for row in rows]
        self.assertEqual(overfitting_replay_fingerprint(first), overfitting_replay_fingerprint(second))
        for result in first:
            self.assertEqual(result["overfittingTransferVersion"], OVERFITTING_TRANSFER_VERSION)
            self.assertGreaterEqual(result["conditionTransferScore"], 0.0)
            self.assertLessEqual(result["conditionTransferScore"], 1.0)
            self.assertEqual(result["claimBoundary"], OVERFITTING_PROXY_SCOPE_NOTE)

    def test_selection_gap_summary_and_validation_accept_complete_panel(self) -> None:
        policies = list(build_s13_policies()[:2])
        tasks = build_s13_tasks()
        conditions = []
        results = []
        for index, policy in enumerate(policies):
            rows = condition_rows_for_transfer_policy(policy, tasks, seed_count=1, seed_base=16300 + index * 1000)
            conditions.extend(rows)
            for row in rows:
                result, _ = run_transfer_condition(policy, row)
                results.append(result)
        task_df = pd.DataFrame([task.to_dict() for task in tasks])
        condition_df = pd.DataFrame(conditions)
        result_df = pd.DataFrame(results)
        decision_df = select_policies_from_selection_results(result_df, enhanced_per_group=1)
        gap_df = summarize_transfer_gaps(result_df, decision_df)
        group_df = transfer_group_summary(gap_df)
        validation_df = validate_overfitting_outputs(task_df, condition_df, result_df, decision_df, gap_df)
        self.assertTrue(validation_df["success"].all(), validation_df.to_string(index=False))
        self.assertFalse(decision_df.empty)
        self.assertFalse(gap_df.empty)
        self.assertFalse(group_df.empty)
        self.assertTrue((~decision_df["holdoutResultsUsedForSelection"]).all())
        self.assertTrue(gap_df["claimBoundary"].eq(OVERFITTING_PROXY_SCOPE_NOTE).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
