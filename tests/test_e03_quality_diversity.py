from __future__ import annotations

import unittest

import pandas as pd

from morphospace import (
    aggregate_s07_policy_metrics,
    build_map_elites_archive,
    qd_config_dict,
    run_qd_smoke_check,
    validate_qd_archive,
)


def competence_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policyId": "p0",
                "vectorId": "v0",
                "taskId": "t0",
                "taskPanel": "jax_unique",
                "taskFamily": "sorting",
                "completionSuccess": 1.0,
                "finalSortednessScore": 1.0,
                "finalMonotonicityScore": 1.0,
                "energyScore": 0.8,
                "swapEfficiencyScore": 0.7,
                "comparisonEfficiencyScore": 0.7,
                "activationEfficiencyScore": 0.7,
                "robustnessScore": None,
                "aggregationFinal": None,
                "failureOscillationScore": 1.0,
            },
            {
                "policyId": "p0",
                "vectorId": "v1",
                "taskId": "t1",
                "taskPanel": "frozen",
                "taskFamily": "frozen",
                "completionSuccess": 0.0,
                "finalSortednessScore": 0.5,
                "finalMonotonicityScore": 0.5,
                "energyScore": 0.9,
                "swapEfficiencyScore": 1.0,
                "comparisonEfficiencyScore": 1.0,
                "activationEfficiencyScore": 0.6,
                "robustnessScore": 0.0,
                "aggregationFinal": None,
                "failureOscillationScore": 0.0,
            },
            {
                "policyId": "p1",
                "vectorId": "v2",
                "taskId": "t0",
                "taskPanel": "jax_unique",
                "taskFamily": "sorting",
                "completionSuccess": 0.0,
                "finalSortednessScore": 0.4,
                "finalMonotonicityScore": 0.4,
                "energyScore": 1.0,
                "swapEfficiencyScore": 1.0,
                "comparisonEfficiencyScore": 1.0,
                "activationEfficiencyScore": 0.8,
                "robustnessScore": None,
                "aggregationFinal": None,
                "failureOscillationScore": 0.0,
            },
            {
                "policyId": "p2",
                "vectorId": "v3",
                "taskId": "t2",
                "taskPanel": "duplicate",
                "taskFamily": "sorting_duplicate_values",
                "completionSuccess": 1.0,
                "finalSortednessScore": 1.0,
                "finalMonotonicityScore": 1.0,
                "energyScore": 0.4,
                "swapEfficiencyScore": 0.3,
                "comparisonEfficiencyScore": 0.3,
                "activationEfficiencyScore": 0.3,
                "robustnessScore": None,
                "aggregationFinal": None,
                "failureOscillationScore": 1.0,
            },
        ]
    )


def corpus_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policyId": "p0",
                "family": "seed",
                "generationMethod": "seed",
                "lineageId": "l0",
                "lineageDepth": 0,
                "parentPolicyIdsJson": "[]",
                "mutationOperatorsJson": "[]",
                "recombinationParentsJson": "[]",
                "structureHash": "h0",
                "generationIndex": 0,
                "description": "policy 0",
                "tagsJson": "[]",
                "observationRequirementsJson": "[]",
                "dslProgramJson": "{}",
                "dslRelativePath": "p0.json",
                "ruleCount": 1,
                "predicateCount": 1,
                "updateCount": 0,
                "stateKeyCount": 0,
                "stochasticActionCount": 0,
                "targetUpdateCount": 0,
                "signalCount": 0,
                "complexityScore": 2,
                "actionCountsJson": "{}",
            },
            {
                "policyId": "p1",
                "family": "mutated",
                "generationMethod": "mutation",
                "lineageId": "l1",
                "lineageDepth": 1,
                "parentPolicyIdsJson": '["p0"]',
                "mutationOperatorsJson": '["flip_operator"]',
                "recombinationParentsJson": "[]",
                "structureHash": "h1",
                "generationIndex": 1,
                "description": "policy 1",
                "tagsJson": "[]",
                "observationRequirementsJson": "[]",
                "dslProgramJson": "{}",
                "dslRelativePath": "p1.json",
                "ruleCount": 2,
                "predicateCount": 2,
                "updateCount": 0,
                "stateKeyCount": 0,
                "stochasticActionCount": 0,
                "targetUpdateCount": 0,
                "signalCount": 0,
                "complexityScore": 4,
                "actionCountsJson": "{}",
            },
            {
                "policyId": "p2",
                "family": "mutated",
                "generationMethod": "mutation",
                "lineageId": "l2",
                "lineageDepth": 1,
                "parentPolicyIdsJson": '["p0"]',
                "mutationOperatorsJson": '["swap_action"]',
                "recombinationParentsJson": "[]",
                "structureHash": "h2",
                "generationIndex": 2,
                "description": "policy 2",
                "tagsJson": "[]",
                "observationRequirementsJson": "[]",
                "dslProgramJson": "{}",
                "dslRelativePath": "p2.json",
                "ruleCount": 4,
                "predicateCount": 4,
                "updateCount": 0,
                "stateKeyCount": 0,
                "stochasticActionCount": 0,
                "targetUpdateCount": 0,
                "signalCount": 0,
                "complexityScore": 8,
                "actionCountsJson": "{}",
            },
        ]
    )


class TestE03QualityDiversity(unittest.TestCase):
    def test_aggregate_assigns_descriptor_bins_and_scores(self) -> None:
        missing = pd.DataFrame(
            [
                {
                    "policyId": "p1",
                    "taskId": "missing_jax",
                    "missingReason": "policy_outside_s06_jax_subset",
                }
            ]
        )
        summary = aggregate_s07_policy_metrics(competence_rows(), pd.DataFrame(), corpus_rows(), missing)
        self.assertEqual(set(summary["policyId"]), {"p0", "p1", "p2"})
        self.assertTrue(summary["qualityScore"].between(0.0, 1.0).all())
        self.assertTrue(summary["noveltyScore"].between(0.0, 1.0).all())
        p1 = summary[summary["policyId"].eq("p1")].iloc[0]
        self.assertEqual(p1["supportBin"], "cpu_or_future_kernel")

    def test_archive_selects_one_elite_per_cell(self) -> None:
        summary = aggregate_s07_policy_metrics(competence_rows(), pd.DataFrame(), corpus_rows(), pd.DataFrame())
        archive, elites = build_map_elites_archive(summary, max_elites=2)
        self.assertTrue(archive["descriptorCellId"].is_unique)
        self.assertLessEqual(len(elites), 2)
        self.assertTrue(elites["policyId"].is_unique)

    def test_smoke_and_archive_validation_pass(self) -> None:
        summary = aggregate_s07_policy_metrics(competence_rows(), pd.DataFrame(), corpus_rows(), pd.DataFrame())
        smoke = run_qd_smoke_check(summary, sample_size=2, max_elites=2)
        self.assertTrue(smoke["success"].all())
        archive, elites = build_map_elites_archive(summary, max_elites=2)
        checks = validate_qd_archive(summary, archive, elites)
        self.assertTrue(checks["success"].all())

    def test_config_records_bounds(self) -> None:
        config = qd_config_dict(max_elites=32, smoke_sample_size=16, smoke_max_elites=5)
        self.assertEqual(config["maxElites"], 32)
        self.assertIn("nichingColumns", config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
