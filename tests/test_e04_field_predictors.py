from __future__ import annotations

import unittest

import pandas as pd

from memory_repair import (
    FIELD_PROXY_SCOPE_NOTE,
    FIELD_TARGETS,
    FIELD_VARIANTS,
    build_s12_policies,
    build_s12_tasks,
    condition_rows_for_field_policy,
    evaluate_field_predictors,
    field_feature_columns,
    field_feature_definitions,
    field_replay_fingerprint,
    run_field_condition,
    summarize_field_predictors,
    validate_field_predictor_outputs,
)


class TestE04FieldPredictors(unittest.TestCase):
    def test_condition_rows_have_seed_separated_splits_and_proxy_scope(self) -> None:
        policy = build_s12_policies()[1]
        rows = condition_rows_for_field_policy(policy, build_s12_tasks()[:1], seed_count=5, train_seed_count=3, seed_base=14100)
        df = pd.DataFrame(rows)
        self.assertEqual(set(df["split"]), {"train", "test"})
        self.assertTrue(set(df.loc[df["split"] == "train", "seedIndex"]).isdisjoint(set(df.loc[df["split"] == "test", "seedIndex"])))
        self.assertFalse(df["selectionUsed"].any())
        self.assertFalse(df["oracleAccessAllowed"].any())
        self.assertTrue(df["policyAuditSuccess"].all())
        self.assertTrue(df["claimBoundary"].str.contains("Simulated aggregate field proxy only", regex=False).all())

    def test_small_field_replay_is_deterministic_and_extracts_features(self) -> None:
        policy = build_s12_policies()[1]
        task = build_s12_tasks()[0]
        row = condition_rows_for_field_policy(policy, [task], seed_count=1, seed_base=14200)[0]
        first_result, first_trace = run_field_condition(policy, row, lookahead_rows=6)
        second_result, second_trace = run_field_condition(policy, row, lookahead_rows=6)
        self.assertEqual(field_replay_fingerprint(first_trace), field_replay_fingerprint(second_trace))
        self.assertEqual(first_result["featureRowCount"], len(first_trace))
        trace_df = pd.DataFrame(first_trace)
        feature_columns = field_feature_columns(trace_df)
        self.assertGreaterEqual(len(feature_columns), 12)
        for target in FIELD_TARGETS:
            self.assertIn(target, trace_df.columns)
        self.assertTrue(trace_df["claimBoundary"].eq(FIELD_PROXY_SCOPE_NOTE).all())

    def test_model_evaluation_and_validation_accept_small_complete_panel(self) -> None:
        policies = build_s12_policies()
        tasks = build_s12_tasks()
        conditions = []
        results = []
        traces = []
        for policy_index, policy in enumerate(policies):
            rows = condition_rows_for_field_policy(
                policy,
                tasks,
                seed_count=5,
                train_seed_count=3,
                seed_base=14300 + policy_index * 1000,
            )
            conditions.extend(rows)
            for row in rows:
                result, trace = run_field_condition(policy, row, lookahead_rows=8)
                results.append(result)
                traces.extend(trace)
        condition_df = pd.DataFrame(conditions)
        trace_df = pd.DataFrame(traces)
        model_df = evaluate_field_predictors(trace_df, null_seed=15100)
        summary_df = summarize_field_predictors(model_df)
        feature_defs = pd.DataFrame(field_feature_definitions(field_feature_columns(trace_df)))
        validation_df = validate_field_predictor_outputs(condition_df, trace_df, model_df, summary_df, feature_defs)
        self.assertTrue(validation_df["success"].all(), validation_df.to_string(index=False))
        self.assertEqual(set(model_df["fieldVariant"]), set(FIELD_VARIANTS))
        self.assertEqual(set(summary_df["targetId"]), set(FIELD_TARGETS))
        self.assertTrue(model_df["claimBoundary"].eq(FIELD_PROXY_SCOPE_NOTE).all())
        self.assertTrue(summary_df["claimBoundary"].eq(FIELD_PROXY_SCOPE_NOTE).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
