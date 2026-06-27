from __future__ import annotations

import unittest

import pandas as pd

from memory_repair import (
    COMPETENCE_AXES,
    COMPETENCE_PROXY_VERSION,
    PROXY_SCOPE_NOTE,
    baseline_competence_policies,
    build_s11_tasks,
    competence_group_comparison,
    competence_metric_definitions,
    competence_metric_table,
    competence_replay_fingerprint,
    condition_rows_for_competence_policy,
    local_memory_signal_competence_policy,
    run_competence_condition,
    summarize_competence_profiles,
    validate_competence_outputs,
)


class TestE04CompetenceProxies(unittest.TestCase):
    def test_metric_definitions_are_proxy_scoped(self) -> None:
        definitions = competence_metric_definitions()
        metric_ids = {row["metricId"] for row in definitions}
        self.assertTrue(set(COMPETENCE_AXES).issubset(metric_ids))
        self.assertIn("overall_competence_proxy", metric_ids)
        for row in definitions:
            with self.subTest(metric=row["metricId"]):
                self.assertIn("proxy", row["metricId"])
                self.assertIn("Direct computational proxy only", row["claimBoundary"])
                self.assertGreater(len(row["directComputationalMetric"]), 20)

    def test_condition_rows_are_held_out_and_no_oracle(self) -> None:
        policy = local_memory_signal_competence_policy()
        tasks = build_s11_tasks()[:2]
        rows = condition_rows_for_competence_policy(policy, tasks, seed_count=2, seed_base=12100)
        df = pd.DataFrame(rows)
        self.assertEqual(len(df), 4)
        self.assertTrue(df["heldOut"].all())
        self.assertFalse(df["selectionUsed"].any())
        self.assertFalse(df["oracleAccessAllowed"].any())
        self.assertTrue(df["policyAuditSuccess"].all())
        self.assertTrue((df["initialLength"] > 6).all())
        self.assertTrue(df["claimBoundary"].str.contains("Direct computational proxy only", regex=False).all())

    def test_small_replay_is_deterministic_and_bounded(self) -> None:
        policy = local_memory_signal_competence_policy()
        route_task = build_s11_tasks()[0]
        rows = condition_rows_for_competence_policy(policy, [route_task], seed_count=1, seed_base=12200)
        first = [run_competence_condition(policy, row)[0] for row in rows]
        second = [run_competence_condition(policy, row)[0] for row in rows]
        self.assertEqual(competence_replay_fingerprint(first), competence_replay_fingerprint(second))
        result = first[0]
        self.assertEqual(result["competenceProxyVersion"], COMPETENCE_PROXY_VERSION)
        self.assertGreaterEqual(result["goalAttainmentProxy"], 0.0)
        self.assertLessEqual(result["goalAttainmentProxy"], 1.0)
        self.assertGreaterEqual(result["energyEfficiencyProxy"], 0.0)
        self.assertLessEqual(result["energyEfficiencyProxy"], 1.0)
        self.assertTrue(result["routeTransitionHash"])

    def test_validation_accepts_small_complete_panel_when_evolved_requirement_relaxed(self) -> None:
        policies = list(baseline_competence_policies()[:1]) + [local_memory_signal_competence_policy()]
        tasks = build_s11_tasks()
        conditions = []
        results = []
        for index, policy in enumerate(policies):
            rows = condition_rows_for_competence_policy(policy, tasks, seed_count=1, seed_base=12300 + index * 1000)
            conditions.extend(rows)
            for row in rows:
                result, _ = run_competence_condition(policy, row)
                results.append(result)
        condition_df = pd.DataFrame(conditions)
        result_df = pd.DataFrame(results)
        profile_df = summarize_competence_profiles(result_df)
        metric_df = competence_metric_table(result_df, profile_df)
        definition_df = pd.DataFrame(competence_metric_definitions())
        validation_df = validate_competence_outputs(
            condition_df,
            result_df,
            metric_df,
            profile_df,
            definition_df,
            expected_evolved_min=0,
        )
        self.assertTrue(validation_df["success"].all(), validation_df.to_string(index=False))
        comparison_df = competence_group_comparison(profile_df)
        self.assertFalse(comparison_df.empty)
        self.assertTrue(metric_df["metricValue"].dropna().between(0.0, 1.0).all())
        self.assertTrue(metric_df["claimBoundary"].eq(PROXY_SCOPE_NOTE).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
