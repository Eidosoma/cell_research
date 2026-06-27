from __future__ import annotations

import unittest

import pandas as pd

from morphospace import (
    SweepTask,
    local_inversion_program,
    make_missing_record,
    parse_rule_program,
    run_batch_simulator,
    run_record_validation,
    run_records_to_competence_vectors,
    select_stratified_policy_ids,
    task_coverage_table,
    value_counts_conserved,
)
from morphospace.coarse_sweep import batch_result_to_run_records


class TestE03CoarseSweep(unittest.TestCase):
    def test_value_counts_conserved_accepts_duplicates(self) -> None:
        self.assertTrue(value_counts_conserved("[2,1,2,3,1]", "[1,1,2,2,3]"))
        self.assertFalse(value_counts_conserved("[2,1,2,3,1]", "[1,2,2,3,3]"))

    def test_batch_rows_preserve_task_identity_in_competence_vectors(self) -> None:
        program = local_inversion_program()
        result = run_batch_simulator(
            [program],
            [[3, 1, 2]],
            [101],
            max_activations=64,
            max_swaps=64,
            max_comparisons=128,
        )
        task = SweepTask(
            task_id="unit_task",
            task_panel="unit_panel",
            task_family="sorting",
            input_profile="unit_unique",
            initial_values=(3, 1, 2),
            scheduler_seed_base=101,
            replicate_index=0,
            backend="jax_batch",
        )
        metadata = {program.policy_id: {"family": "seed", "generationMethod": "unit", "complexityScore": 4}}
        records = batch_result_to_run_records(result, task, metadata)
        vectors = run_records_to_competence_vectors(records)
        self.assertEqual(vectors.iloc[0]["taskId"], "unit_task")
        self.assertEqual(vectors.iloc[0]["taskFamily"], "sorting")
        self.assertEqual(vectors.iloc[0]["taskPanel"], "unit_panel")
        self.assertEqual(vectors.iloc[0]["backend"], "jax_batch")
        self.assertEqual(vectors.iloc[0]["finalSortednessScore"], 1.0)

    def test_stratified_selection_is_deterministic_and_spans_strata(self) -> None:
        corpus = pd.DataFrame(
            [
                {"policyId": "a0", "family": "seed", "generationMethod": "x", "complexityScore": 1},
                {"policyId": "a1", "family": "seed", "generationMethod": "x", "complexityScore": 2},
                {"policyId": "b0", "family": "mutated", "generationMethod": "y", "complexityScore": 1},
                {"policyId": "b1", "family": "mutated", "generationMethod": "y", "complexityScore": 2},
            ]
        )
        selected = select_stratified_policy_ids(corpus, ["a0", "a1", "b0", "b1"], sample_size=3)
        self.assertEqual(selected, ["b0", "a0", "b1"])

    def test_missing_records_and_coverage_table(self) -> None:
        policy_row = {
            "policyId": "p0",
            "family": "generated",
            "generationMethod": "mutation",
            "complexityScore": 3,
        }
        missing = make_missing_record(
            policy_row=policy_row,
            task_id="frozen_full",
            task_panel="cpu_frozen_full_not_run",
            task_family="frozen",
            backend="missing_reason",
            missing_reason="cpu_reference_budgeted_spot_sample_only",
            support_reasons_json='["state_updates_not_supported"]',
        )
        coverage = task_coverage_table([], [missing])
        self.assertEqual(missing["recordStatus"], "missing")
        self.assertEqual(coverage.iloc[0]["recordStatus"], "missing")
        self.assertEqual(coverage.iloc[0]["policyCount"], 1)

    def test_run_record_validation(self) -> None:
        task = SweepTask(
            task_id="unit_task",
            task_panel="unit_panel",
            task_family="sorting",
            input_profile="unit_unique",
            initial_values=(3, 1, 2),
            scheduler_seed_base=101,
            replicate_index=0,
            backend="jax_batch",
        )
        program = parse_rule_program(local_inversion_program())
        result = run_batch_simulator([program], [[3, 1, 2]], [101], max_activations=64, max_swaps=64, max_comparisons=128)
        records = batch_result_to_run_records(result, task, {program.policy_id: {}})
        validation = run_record_validation(records)
        self.assertTrue(validation["valueCountsConserved"])
        self.assertTrue(validation["metricRangesValid"])
        self.assertEqual(validation["rowCount"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
