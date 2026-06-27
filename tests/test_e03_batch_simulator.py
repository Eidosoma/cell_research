import unittest

from morphospace import (
    available_jax_devices,
    batch_support_report,
    bubble_template,
    compare_batch_to_cpu,
    generate_policy_corpus,
    local_inversion_program,
    null_program,
    run_batch_simulator,
    run_cpu_reference_case,
    stochastic_right_program,
    summarize_support,
)


class BatchSimulatorTests(unittest.TestCase):
    def test_support_report_accepts_stateless_adjacent_subset(self):
        reports = [
            batch_support_report(null_program()),
            batch_support_report(local_inversion_program()),
            batch_support_report(bubble_template(action_probability=1.0)),
        ]
        self.assertTrue(all(report.supported for report in reports))
        summary = summarize_support(reports)
        self.assertEqual(summary["supportedPolicyCount"], 3)
        self.assertEqual(summary["unsupportedPolicyCount"], 0)

    def test_support_report_rejects_stochastic_and_stateful_programs(self):
        stochastic = batch_support_report(stochastic_right_program())
        self.assertFalse(stochastic.supported)
        self.assertIn("stochastic_action_not_supported", stochastic.reasons)
        stateful = batch_support_report(
            {
                "policy_id": "stateful_memory_demo",
                "initial_state": {"memory": 0},
                "rules": [
                    {
                        "name": "remember",
                        "when": [{"op": "always"}],
                        "then": {"action": "remember", "updates": [{"op": "increment", "key": "memory"}]},
                    }
                ],
                "default": {"action": "wait"},
            }
        )
        self.assertFalse(stateful.supported)
        self.assertIn("initial_state_not_supported", stateful.reasons)
        self.assertIn("state_updates_not_supported", stateful.reasons)

    def test_batch_simulator_matches_cpu_reference_cases(self):
        self.assertTrue(available_jax_devices())
        programs = [null_program(), local_inversion_program(), bubble_template(action_probability=1.0)]
        initial_values = [[3, 1, 2], [3, 1, 2], [2, 3, 1]]
        seeds = [1, 2, 3]
        cpu_rows = [
            run_cpu_reference_case(program, values, seed, max_activations=128, max_swaps=128, max_comparisons=512)
            for program, values, seed in zip(programs, initial_values, seeds)
        ]
        batch_rows = run_batch_simulator(
            programs,
            initial_values,
            seeds,
            max_activations=128,
            max_swaps=128,
            max_comparisons=512,
        ).to_rows()
        comparisons = compare_batch_to_cpu(cpu_rows, batch_rows)
        self.assertTrue(all(row["agreement"] for row in comparisons), comparisons)

    def test_s05_subset_policies_match_cpu_reference(self):
        records = generate_policy_corpus()
        subset = [record for record in records if batch_support_report(record.program).supported]
        self.assertGreaterEqual(len(subset), 50)
        selected = subset[:12]
        initial_values = [[3, 1, 2] for _ in selected]
        seeds = [100 + index for index, _ in enumerate(selected)]
        cpu_rows = [
            run_cpu_reference_case(record.program, values, seed, max_activations=128, max_swaps=128, max_comparisons=512)
            for record, values, seed in zip(selected, initial_values, seeds)
        ]
        batch_rows = run_batch_simulator(
            [record.program for record in selected],
            initial_values,
            seeds,
            max_activations=128,
            max_swaps=128,
            max_comparisons=512,
        ).to_rows()
        comparisons = compare_batch_to_cpu(cpu_rows, batch_rows)
        self.assertTrue(all(row["agreement"] for row in comparisons), comparisons)


if __name__ == "__main__":
    unittest.main()
