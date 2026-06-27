from __future__ import annotations

import unittest

import pandas as pd

from morphospace import (
    FRONTIER_CANDIDATE_VERSION,
    build_frontier_candidate_pool,
    select_frontier_candidates,
    summarize_frontier_validation,
    summarize_pairwise_frontier_evidence,
    summarize_phase_boundary_roles,
    summarize_s12_ablation_sources,
    validate_frontier_outputs,
)


def policy_fixture() -> pd.DataFrame:
    rows = []
    for index in range(10):
        rows.append(
            {
                "policyId": f"disc_{index:02d}",
                "family": "generated",
                "generationMethod": "mutation" if index % 2 else "single_rule_sweep",
                "lineageId": f"lineage_{index:02d}",
                "description": "fixture discovered policy",
                "comparisonRole": "discovered",
                "selectedForS13": True,
                "primaryRole": "elite" if index in {0, 3, 6} else "generated",
                "isClassicPolicy": False,
                "isNullPolicy": False,
                "isS08Elite": index in {0, 3, 6},
                "isPhaseBoundaryPolicy": index in {1, 4, 7},
                "isPathologicalPolicy": index == 9,
                "classicFamily": "none",
                "className": f"UC{index % 4:02d}",
                "classLabel": "fixture",
                "distanceToCentroid": 0.1 + index * 0.05,
                "phaseBoundaryInvolvementCount": float(index % 3),
                "s08Elite_qualityScore": 0.4 + index * 0.02 if index in {0, 3, 6} else None,
                "completionScore": 0.45 + index * 0.04,
                "sortednessScore": 0.50 + index * 0.035,
                "energyScore": 0.65 - index * 0.02,
                "robustnessScore": 0.15 + (index % 4) * 0.08,
                "delayedGratificationScore": 0.10 + index * 0.05,
                "aggregationScore": 0.25 + (9 - index) * 0.035,
                "oscillationStabilityScore": 0.55 + index * 0.025,
                "s13CompositeScore": 0.45 + index * 0.03,
                "s13_completionRateTaskFamily_transfer": 0.4 + (index % 5) * 0.1,
                "s13_completionRateTaskFamily_chimera": 0.3 + (index % 4) * 0.12,
            }
        )
    rows.append(
        {
            "policyId": "classic_bubble",
            "family": "hand_designed",
            "generationMethod": "classic_parameter_sweep",
            "lineageId": "bubble_fixture",
            "comparisonRole": "classic",
            "selectedForS13": True,
            "primaryRole": "classic",
            "isClassicPolicy": True,
            "isNullPolicy": False,
            "classicFamily": "bubble",
            "s13CompositeScore": 0.8,
        }
    )
    rows.append(
        {
            "policyId": "null_policy",
            "family": "seed",
            "generationMethod": "seed_import",
            "lineageId": "null_fixture",
            "comparisonRole": "null_context",
            "selectedForS13": False,
            "primaryRole": "null",
            "isClassicPolicy": False,
            "isNullPolicy": True,
            "classicFamily": "none",
            "s13CompositeScore": 0.0,
        }
    )
    return pd.DataFrame(rows)


def corpus_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policyId": f"disc_{index:02d}",
                "structureHash": f"hash_{index:02d}",
                "dslProgramJson": f'{{"name":"disc_{index:02d}","rules":[],"default":{{"action":"wait"}}}}',
                "dslRelativePath": f"generated/disc_{index:02d}.json",
                "parentPolicyIdsJson": "[]",
                "mutationOperatorsJson": "[]",
                "recombinationParentsJson": "[]",
                "tagsJson": "[]",
                "observationRequirementsJson": "[]",
                "actionCountsJson": '{"wait": 1}',
            }
            for index in range(10)
        ]
    )


def pairwise_fixture() -> pd.DataFrame:
    rows = []
    for index in range(10):
        for family in ["bubble", "insertion", "selection"]:
            rows.append(
                {
                    "classicPolicyId": f"classic_{family}",
                    "discoveredPolicyId": f"disc_{index:02d}",
                    "classicFamily": family,
                    "discoveredDominatesClassicBroad": index >= 6,
                    "classicDominatesDiscoveredBroad": index < 2,
                    "discoveredDominatesClassicSameSeed": index >= 5,
                    "classicDominatesDiscoveredSameSeed": index == 0,
                    "sameSeedNetAdvantageScore": -0.05 + index * 0.03,
                    "completionScoreDelta": -0.02 + index * 0.01,
                    "sortednessScoreDelta": -0.01 + index * 0.01,
                    "energyScoreDelta": 0.04 - index * 0.005,
                    "robustnessScoreDelta": 0.01 * index,
                    "delayedGratificationScoreDelta": 0.015 * index,
                    "aggregationScoreDelta": 0.02 * (9 - index),
                    "oscillationStabilityScoreDelta": 0.02,
                    "completionSuccessDeltaMean": -0.02 + index * 0.01,
                    "finalSortednessScoreDeltaMean": -0.01 + index * 0.01,
                    "energyScoreDeltaMean": 0.03 - index * 0.004,
                    "robustnessScoreDeltaMean": 0.01 * index,
                    "delayedGratificationScoreDeltaMean": 0.015 * index,
                    "aggregationAucScoreDeltaMean": 0.02 * (9 - index),
                    "oscillationScoreDeltaMean": 0.02,
                    "normalizedFeatureDistance": 1.0 + index,
                    "pcaSeed0Distance": 0.5 + index,
                }
            )
    return pd.DataFrame(rows)


def pareto_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policyId": f"disc_{index:02d}",
                "paretoEligible": True,
                "paretoFrontRank": 1 if index in {7, 8, 9} else 2 + index,
                "isParetoOptimal": index in {7, 8, 9},
                "s13CompositeScore": 0.45 + index * 0.03,
            }
            for index in range(10)
        ]
    )


def qd_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policyId": f"disc_{index:02d}",
                "eliteRank": index + 1,
                "archiveRank": index + 1,
                "descriptorCellId": f"cell_{index}",
                "descriptorCellLabel": "fixture",
                "qualityScore": 0.6 + index * 0.01,
                "noveltyScore": 0.2 + index * 0.03,
                "coverageScore": 0.9,
                "completionRate": 1.0,
                "meanSortednessScore": 0.8,
                "meanEnergyScore": 0.7,
                "duplicateCompletionRate": 1.0,
                "robustnessMean": 0.4,
                "aggregationMean": 0.5,
            }
            for index in [0, 3, 6]
        ]
    )


def boundary_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "boundaryId": f"pb_{index}",
                "leftPolicyId": f"disc_{index:02d}",
                "rightPolicyId": f"disc_{(index + 1) % 10:02d}",
                "boundaryScore": 0.4 + index * 0.1,
                "sortingSuccessTransition": index % 2 == 0,
                "aggregationTransition": index % 2 == 1,
            }
            for index in range(5)
        ]
    )


def class_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policyId": f"disc_{index:02d}",
                "universalityClassId": f"UC{index % 4:02d}",
                "className": f"UC{index % 4:02d}",
                "classLabel": "fixture",
                "classInterpretation": "fixture class",
                "empiricalCaveat": "fixture only",
                "distanceToCentroid": 0.1 + index * 0.05,
            }
            for index in range(10)
        ]
    )


def ablation_fixture() -> pd.DataFrame:
    rows = []
    for index in range(10):
        rows.append(
            {
                "sourcePolicyId": f"disc_{index:02d}",
                "ablationPolicyId": f"ab_{index:02d}",
                "ablationType": "remove_memory_updates",
                "taskFamily": "sorting",
                "completionSuccessDelta": -0.01 * index,
                "finalSortednessScoreDelta": -0.01 * index,
                "energyScoreDelta": -0.005 * index,
                "robustnessScoreDelta": -0.02 * index,
                "delayedGratificationScoreDelta": -0.01 * index,
                "aggregationAucScoreDelta": -0.01 * index,
                "oscillationScoreDelta": -0.005 * index,
                "completionSuccessOriginal": 0.8,
                "finalSortednessScoreOriginal": 0.8,
                "energyScoreOriginal": 0.7,
                "robustnessScoreOriginal": 0.4,
                "delayedGratificationScoreOriginal": 0.5,
                "aggregationAucScoreOriginal": 0.6,
                "oscillationScoreOriginal": 0.9,
            }
        )
    return pd.DataFrame(rows)


def validation_rows(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows = []
    vector_rows = []
    families = ["sorting", "sorting_duplicate_values", "frozen", "transfer", "chimera"]
    for _, row in selected.iterrows():
        for task_index, family in enumerate(families):
            for seed_index in [0]:
                base = {
                    "policyId": row["policyId"],
                    "taskId": f"task_{family}",
                    "taskFamily": family,
                    "taskPanel": "fixture",
                    "inputProfile": "fixture",
                    "replicateIndex": seed_index,
                    "schedulerSeed": 81000 + task_index,
                    "tieBreakerSeed": 91000 + task_index,
                    "initialValuesJson": "[8,1,7,2,6,3,5,4]",
                    "valueCountsConserved": True,
                    "completedNumeric": 1.0,
                    "completed": True,
                    "stopReason": "sorted",
                    "swapCount": 4,
                    "comparisonCount": 12,
                    "activationCount": 16,
                    "runtimeSeconds": 0.001,
                }
                run_rows.append(base)
                vector_rows.append(
                    {
                        **base,
                        "completionSuccess": 1.0,
                        "finalSortednessScore": 1.0,
                        "energyScore": 0.8,
                        "robustnessScore": 0.6 if family == "frozen" else None,
                        "delayedGratificationScore": 0.5,
                        "aggregationAucScore": 0.7,
                        "compatibilityScore": 0.7 if family == "chimera" else None,
                        "transferScore": 1.0 if family == "transfer" else None,
                        "oscillationScore": 0.9,
                    }
                )
    run_df = pd.DataFrame(run_rows)
    vector_df = pd.DataFrame(vector_rows)
    summary_df = summarize_frontier_validation(vector_df, run_df, selected)
    return run_df, vector_df, summary_df


class TestE03FrontierCandidates(unittest.TestCase):
    def test_pool_selection_and_validation(self) -> None:
        pair_summary = summarize_pairwise_frontier_evidence(pairwise_fixture())
        self.assertEqual(pair_summary["policyId"].nunique(), 10)

        boundary_summary = summarize_phase_boundary_roles(boundary_fixture())
        self.assertFalse(boundary_summary.empty)

        s12_summary = summarize_s12_ablation_sources(ablation_fixture())
        self.assertEqual(s12_summary["policyId"].nunique(), 10)

        pool = build_frontier_candidate_pool(
            policy_df=policy_fixture(),
            pairwise_df=pairwise_fixture(),
            pareto_df=pareto_fixture(),
            corpus_df=corpus_fixture(),
            qd_elites_df=qd_fixture(),
            phase_boundaries_df=boundary_fixture(),
            class_assignments_df=class_fixture(),
            feature_ablations_df=ablation_fixture(),
        )
        self.assertEqual(pool["frontierCandidateVersion"].iloc[0], FRONTIER_CANDIDATE_VERSION)
        self.assertEqual(len(pool), 10)
        self.assertFalse(pool["isClassicPolicy"].any())
        self.assertFalse(pool["isNullPolicy"].any())

        selected = select_frontier_candidates(pool, min_count=10, max_count=10, target_count=10, max_per_class=4)
        self.assertEqual(len(selected), 10)
        self.assertTrue(selected["policyId"].is_unique)
        self.assertTrue(selected["structureHash"].is_unique)
        self.assertTrue(selected["selectedForS14"].all())
        represented_tags = {tag for value in selected["frontierObjectiveTagsJson"] for tag in __import__("json").loads(value)}
        self.assertGreaterEqual(len(represented_tags), 6)

        run_df, vector_df, validation_summary = validation_rows(selected)
        dsl_index = pd.DataFrame(
            {
                "policyId": selected["policyId"],
                "roundtripSuccess": True,
                "dslPath": [f"/tmp/{policy_id}.json" for policy_id in selected["policyId"]],
                "dslSha256": ["fixture"] * len(selected),
            }
        )
        checks = validate_frontier_outputs(
            candidate_df=selected,
            pool_df=pool,
            validation_run_df=run_df,
            validation_vector_df=vector_df,
            validation_summary_df=validation_summary,
            dsl_index_df=dsl_index,
            upstream_statuses={step: {"success": True, "status": "completed"} for step in ["S08", "S09", "S10", "S11", "S12", "S13"]},
            repo_test_payload={"success": True, "returnCode": 0, "command": ["unit"]},
            report_exists=True,
            figure_paths=["a.png", "b.png"],
            s15_dir_exists=False,
            expected_run_rows=len(run_df),
        )
        self.assertTrue(checks["success"].all(), checks.to_string(index=False))


if __name__ == "__main__":
    unittest.main()
