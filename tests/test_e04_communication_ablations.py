from __future__ import annotations

import unittest

import pandas as pd

from memory_repair import (
    COMMUNICATION_ABLATION_VERSION,
    SIGNAL_ABLATION_VARIANTS,
    build_s10_tasks,
    classic_policy_families,
    communication_ablation_replay_fingerprint,
    communication_transfer_gaps,
    compute_communication_deltas,
    condition_rows_for_communication_family,
    learned_policy_families,
    no_signal_reproduction_rows,
    run_communication_condition,
    summarize_communication_effects,
    validate_communication_outputs,
)


class TestE04CommunicationAblations(unittest.TestCase):
    def test_condition_rows_are_paired_and_signal_parameters_logged(self) -> None:
        family = classic_policy_families()[0]
        task = build_s10_tasks()[0]
        rows = condition_rows_for_communication_family(family, [task], seed_count=1, seed_base=10100)
        variants = {row["ablationVariant"] for row in rows}
        self.assertEqual(variants, set(SIGNAL_ABLATION_VARIANTS))
        grouped = pd.DataFrame(rows).groupby("controlledGroupId")
        self.assertTrue((grouped["schedulerSeed"].nunique() == 1).all())
        self.assertTrue((grouped["tieBreakerSeed"].nunique() == 1).all())
        self.assertTrue((grouped["taskSpecJson"].nunique() == 1).all())
        self.assertTrue((grouped["controlledConfigHash"].nunique() == 1).all())
        for row in rows:
            self.assertIn("signalRange", row)
            self.assertIn("noiseStd", row)
            self.assertIn("signalRandomSeed", row)

    def test_small_replay_is_deterministic_and_quantifies_deltas(self) -> None:
        family = learned_policy_families()[0]
        task = build_s10_tasks()[0]
        rows = condition_rows_for_communication_family(family, [task], seed_count=1, seed_base=10200)
        first_results = [run_communication_condition(family, row)[0] for row in rows]
        second_results = [run_communication_condition(family, row)[0] for row in rows]
        self.assertEqual(communication_ablation_replay_fingerprint(first_results), communication_ablation_replay_fingerprint(second_results))
        result_df = pd.DataFrame(first_results)
        delta_df = compute_communication_deltas(result_df)
        self.assertGreater(len(delta_df), 0)
        summary_df = summarize_communication_effects(delta_df)
        self.assertFalse(summary_df.empty)
        self.assertEqual(summary_df["communicationAblationVersion"].iloc[0], COMMUNICATION_ABLATION_VERSION)

    def test_no_signal_reproduction_and_validation_accept_small_panel(self) -> None:
        families = list(classic_policy_families()[:1]) + list(learned_policy_families())
        tasks = (build_s10_tasks()[0], build_s10_tasks()[2], build_s10_tasks()[4], build_s10_tasks()[5])
        conditions = []
        results = []
        family_by_id = {family.family_id: family for family in families}
        for family in families:
            rows = condition_rows_for_communication_family(family, tasks, seed_count=1, seed_base=10300)
            conditions.extend(rows)
            for row in rows:
                result, _ = run_communication_condition(family, row)
                results.append(result)
        condition_df = pd.DataFrame(conditions)
        result_df = pd.DataFrame(results)
        delta_df = compute_communication_deltas(result_df)
        no_signal_df = no_signal_reproduction_rows(condition_df, result_df, family_by_id)
        self.assertTrue(no_signal_df["success"].all(), no_signal_df.to_string(index=False))
        validation_df = validate_communication_outputs(
            condition_df,
            result_df,
            delta_df,
            no_signal_df,
            expected_evolved_min=0,
            required_family_kinds={"classic", "learned"},
        )
        self.assertTrue(validation_df["success"].all(), validation_df.to_string(index=False))
        gaps_df = communication_transfer_gaps(delta_df)
        self.assertFalse(gaps_df.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
