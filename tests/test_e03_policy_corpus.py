import unittest
from collections import Counter

from morphospace import (
    DSLPolicy,
    POLICY_CORPUS_VERSION,
    PolicyEventSimulator,
    generate_policy_corpus,
    policy_corpus_table,
    policy_from_spec,
    policy_lineage_table,
    validate_corpus_records,
)
from morphospace.rule_dsl import parse_rule_program


class PolicyCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate_policy_corpus()

    def test_default_corpus_has_thousands_and_expected_families(self):
        self.assertGreaterEqual(len(self.records), 2000)
        self.assertEqual({record.corpus_version for record in self.records}, {POLICY_CORPUS_VERSION})
        families = Counter(record.family for record in self.records)
        for family in ["seed", "hand_designed", "mutation", "recombined", "random_generated"]:
            self.assertGreater(families[family], 0, family)

    def test_policy_ids_and_structure_hashes_are_unique(self):
        validation = validate_corpus_records(self.records)
        self.assertTrue(validation["success"], validation)
        self.assertEqual(validation["uniquePolicyIds"], len(self.records))
        self.assertEqual(validation["uniqueStructureHashes"], len(self.records))
        self.assertEqual(validation["duplicatePolicyIds"], [])
        self.assertEqual(validation["duplicateStructureHashes"], [])

    def test_rows_include_lineage_complexity_and_dsl_payloads(self):
        rows = policy_corpus_table(self.records[:10])
        lineage = policy_lineage_table(self.records[:10])
        self.assertEqual(len(rows), 10)
        self.assertEqual(len(lineage), 10)
        for row in rows:
            self.assertIn("policyId", row)
            self.assertIn("dslProgramJson", row)
            self.assertIn("complexityScore", row)
            self.assertIn("observationRequirementsJson", row)
            self.assertGreaterEqual(row["ruleCount"], 1)
            restored = parse_rule_program(row["dslProgramJson"])
            self.assertEqual(restored.policy_id, row["policyId"])
        for row in lineage:
            self.assertIn("lineageId", row)
            self.assertIn("parentPolicyIdsJson", row)

    def test_policy_specs_restore_through_s01_boundary(self):
        for record in self.records[:50]:
            policy = DSLPolicy(record.program)
            restored = policy_from_spec(policy.to_spec())
            self.assertEqual(restored.to_spec().policy_id, record.policy_id)

    def test_sample_execution_on_tiny_array_conserves_values(self):
        initial = [3, 1, 2]
        expected_counts = Counter(initial)
        for index, record in enumerate(self.records[:80]):
            result = PolicyEventSimulator(
                initial,
                DSLPolicy(record.program),
                scheduler_seed=10 + index,
                tie_breaker_seed=20 + index,
                condition_id=f"unit_tiny_{record.policy_id}",
                implementation="e03_s05_policy_corpus_unit",
                research_step_id="S05",
            ).run(max_activations=128, max_swaps=128, max_comparisons=512)
            self.assertEqual(Counter(result.final_values), expected_counts, record.policy_id)


if __name__ == "__main__":
    unittest.main()
