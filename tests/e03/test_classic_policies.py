"""S03 classic-policy mapping tests."""

from __future__ import annotations

import unittest

from src.e03.classic_policies import CLASSIC_DSL_SOURCES, classic_policy_library, classic_policy_mappings, validation_cases
from src.e03.rule_dsl import parse_policy


class ClassicPolicyMappingTests(unittest.TestCase):
    def test_policy_library_has_unique_stable_ids(self) -> None:
        library = classic_policy_library()
        policy_ids = [entry["policyId"] for entry in library["policies"]]
        self.assertEqual(library["policyCount"], len(policy_ids))
        self.assertEqual(len(policy_ids), len(set(policy_ids)))
        self.assertTrue({"bubble", "insertion", "selection"}.issubset({entry["algorithm"] for entry in library["policies"]}))

    def test_all_dsl_sources_parse_and_hash(self) -> None:
        for name, source in CLASSIC_DSL_SOURCES.items():
            with self.subTest(name=name):
                policy = parse_policy(source)
                self.assertTrue(policy.policy_id.startswith("dsl:"))
                self.assertEqual(parse_policy(policy.to_source()).sha256, policy.sha256)

    def test_library_contains_exact_interface_mapping_for_each_classic(self) -> None:
        exact_interface = {
            mapping.algorithm
            for mapping in classic_policy_mappings()
            if mapping.representation_type == "interface_wrapper" and mapping.exactness == "exact_public_method"
        }
        self.assertEqual(exact_interface, {"bubble", "insertion", "selection"})

    def test_validation_cases_pass_or_document_expected_deviation(self) -> None:
        df = validation_cases()
        self.assertEqual(int(df["success"].sum()), len(df))
        deviation_rows = df[df["expected_match"] == False]
        self.assertGreaterEqual(len(deviation_rows), 1)
        self.assertTrue((deviation_rows["observed_match"] == False).all())


if __name__ == "__main__":
    unittest.main()
