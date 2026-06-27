from __future__ import annotations

import unittest

import pandas as pd

from memory_repair import (
    CELL_MEMORY_VARIANTS,
    FIELD_MEMORY_VARIANTS,
    MEMORY_ABLATION_VERSION,
    build_s09_tasks,
    classic_policy_families,
    compute_ablation_deltas,
    condition_rows_for_family,
    learned_policy_families,
    memory_ablation_replay_fingerprint,
    run_ablation_condition,
    summarize_ablation_effects,
    validate_ablation_outputs,
)


class TestE04MemoryAblations(unittest.TestCase):
    def test_condition_rows_are_paired_and_controlled(self) -> None:
        family = classic_policy_families()[0]
        task = build_s09_tasks()[0]
        rows = condition_rows_for_family(family, [task], seed_count=1, seed_base=9100)
        variants = {row["ablationVariant"] for row in rows}
        self.assertTrue(set(CELL_MEMORY_VARIANTS).issubset(variants))
        self.assertTrue(set(FIELD_MEMORY_VARIANTS).issubset(variants))
        grouped = pd.DataFrame(rows).groupby("controlledGroupId")
        self.assertTrue((grouped["schedulerSeed"].nunique() == 1).all())
        self.assertTrue((grouped["tieBreakerSeed"].nunique() == 1).all())
        self.assertTrue((grouped["controlledConfigHash"].nunique() == 1).all())

    def test_small_replay_is_deterministic_and_quantifies_deltas(self) -> None:
        family = learned_policy_families()[0]
        task = build_s09_tasks()[0]
        rows = condition_rows_for_family(family, [task], seed_count=1, seed_base=9200)
        first_results = [run_ablation_condition(family, row)[0] for row in rows]
        second_results = [run_ablation_condition(family, row)[0] for row in rows]
        self.assertEqual(memory_ablation_replay_fingerprint(first_results), memory_ablation_replay_fingerprint(second_results))
        result_df = pd.DataFrame(first_results)
        delta_df = compute_ablation_deltas(result_df)
        self.assertGreater(len(delta_df), 0)
        summary_df = summarize_ablation_effects(delta_df)
        self.assertFalse(summary_df.empty)
        self.assertEqual(summary_df["memoryAblationVersion"].iloc[0], MEMORY_ABLATION_VERSION)

    def test_validation_accepts_complete_small_panel_when_requirements_relaxed(self) -> None:
        families = list(classic_policy_families()[:1]) + list(learned_policy_families())
        tasks = build_s09_tasks()[:2]
        conditions = []
        results = []
        for family in families:
            rows = condition_rows_for_family(family, tasks, seed_count=1, seed_base=9300)
            conditions.extend(rows)
            for row in rows:
                result, _ = run_ablation_condition(family, row)
                results.append(result)
        condition_df = pd.DataFrame(conditions)
        result_df = pd.DataFrame(results)
        delta_df = compute_ablation_deltas(result_df)
        validation_df = validate_ablation_outputs(condition_df, result_df, delta_df, expected_evolved_min=0)
        required = validation_df[~validation_df["checkId"].isin({"required_policy_families_present"})]
        self.assertTrue(required["success"].all(), validation_df.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
