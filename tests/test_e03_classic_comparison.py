from __future__ import annotations

import unittest

import pandas as pd

from morphospace import (
    build_pairwise_comparison_table,
    build_policy_comparison_universe,
    classic_family_from_lineage,
    family_summary_table,
    missing_comparison_records,
    pareto_front_table,
    same_seed_task_delta_table,
    summarize_same_seed_pairs,
    validate_classic_comparison_outputs,
)


def feature_fixture() -> pd.DataFrame:
    rows = [
        {
            "policyId": "classic_bubble",
            "family": "hand_designed",
            "generationMethod": "classic_parameter_sweep",
            "lineageId": "bubble_increasing_right_first_1",
            "primaryRole": "classic",
            "isClassicPolicy": True,
            "isNullPolicy": False,
            "isS08Elite": False,
            "isPhaseBoundaryPolicy": False,
            "isPathologicalPolicy": False,
            "classicFamily": "bubble",
            "s09_completionSuccessMean": 0.8,
            "s09_finalSortednessScoreMean": 0.8,
            "s09_energyScoreMean": 0.7,
            "s09_robustnessScoreMean": 0.6,
            "s09_delayedGratificationScoreMean": 0.2,
            "s09_aggregationAucScoreMean": 0.3,
            "s09_oscillationScoreMean": 0.8,
        },
        {
            "policyId": "classic_insertion",
            "family": "hand_designed",
            "generationMethod": "classic_parameter_sweep",
            "lineageId": "insertion_increasing_1",
            "primaryRole": "classic",
            "isClassicPolicy": True,
            "isNullPolicy": False,
            "isS08Elite": False,
            "isPhaseBoundaryPolicy": False,
            "isPathologicalPolicy": False,
            "classicFamily": "insertion",
            "s09_completionSuccessMean": 0.7,
            "s09_finalSortednessScoreMean": 0.7,
            "s09_energyScoreMean": 0.8,
            "s09_robustnessScoreMean": 0.4,
            "s09_delayedGratificationScoreMean": 0.1,
            "s09_aggregationAucScoreMean": 0.2,
            "s09_oscillationScoreMean": 0.7,
        },
        {
            "policyId": "classic_selection",
            "family": "hand_designed",
            "generationMethod": "classic_parameter_sweep",
            "lineageId": "selection_increasing_1",
            "primaryRole": "classic",
            "isClassicPolicy": True,
            "isNullPolicy": False,
            "isS08Elite": False,
            "isPhaseBoundaryPolicy": False,
            "isPathologicalPolicy": False,
            "classicFamily": "selection",
            "s09_completionSuccessMean": 0.5,
            "s09_finalSortednessScoreMean": 0.6,
            "s09_energyScoreMean": 0.6,
            "s09_robustnessScoreMean": 0.5,
            "s09_delayedGratificationScoreMean": 0.6,
            "s09_aggregationAucScoreMean": 0.4,
            "s09_oscillationScoreMean": 0.6,
        },
        {
            "policyId": "disc_elite",
            "family": "generated",
            "generationMethod": "single_rule_sweep",
            "lineageId": "disc_elite_lineage",
            "primaryRole": "elite",
            "isClassicPolicy": False,
            "isNullPolicy": False,
            "isS08Elite": True,
            "isPhaseBoundaryPolicy": False,
            "isPathologicalPolicy": False,
            "classicFamily": "none",
            "s09_completionSuccessMean": 0.9,
            "s09_finalSortednessScoreMean": 0.9,
            "s09_energyScoreMean": 0.9,
            "s09_robustnessScoreMean": 0.7,
            "s09_delayedGratificationScoreMean": 0.3,
            "s09_aggregationAucScoreMean": 0.5,
            "s09_oscillationScoreMean": 0.9,
        },
        {
            "policyId": "disc_boundary",
            "family": "generated",
            "generationMethod": "memory_mutation",
            "lineageId": "disc_boundary_lineage",
            "primaryRole": "phase_boundary",
            "isClassicPolicy": False,
            "isNullPolicy": False,
            "isS08Elite": False,
            "isPhaseBoundaryPolicy": True,
            "isPathologicalPolicy": False,
            "classicFamily": "none",
            "s09_completionSuccessMean": 0.4,
            "s09_finalSortednessScoreMean": 0.5,
            "s09_energyScoreMean": 0.9,
            "s09_robustnessScoreMean": 0.6,
            "s09_delayedGratificationScoreMean": 0.8,
            "s09_aggregationAucScoreMean": 0.7,
            "s09_oscillationScoreMean": 0.8,
        },
        {
            "policyId": "unmeasured_generated",
            "family": "generated",
            "generationMethod": "seeded_random_grammar",
            "lineageId": "unmeasured",
            "primaryRole": "generated",
            "isClassicPolicy": False,
            "isNullPolicy": False,
            "isS08Elite": False,
            "isPhaseBoundaryPolicy": False,
            "isPathologicalPolicy": False,
            "classicFamily": "none",
        },
    ]
    return pd.DataFrame(rows)


def assignments_fixture() -> pd.DataFrame:
    rows = []
    for row in feature_fixture().to_dict(orient="records"):
        rows.append(
            {
                "policyId": row["policyId"],
                "className": "UC_good" if row["policyId"] in {"classic_bubble", "disc_elite"} else "UC_other",
                "classLabel": "fixture",
                "primaryRole": row["primaryRole"],
                "isClassicPolicy": row["isClassicPolicy"],
                "isNullPolicy": row["isNullPolicy"],
                "isS08Elite": row["isS08Elite"],
                "isPhaseBoundaryPolicy": row["isPhaseBoundaryPolicy"],
                "isPathologicalPolicy": row["isPathologicalPolicy"],
                "distanceToCentroid": 0.1,
                "phaseBoundaryInvolvementCount": 1 if row["policyId"] == "disc_boundary" else 0,
            }
        )
    return pd.DataFrame(rows)


def corpus_fixture() -> pd.DataFrame:
    return feature_fixture()[["policyId", "family", "generationMethod", "lineageId"]].copy()


def s12_specs_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sourcePolicyId": "disc_elite", "ablationType": "original"},
            {"sourcePolicyId": "classic_bubble", "ablationType": "original"},
        ]
    )


def vector_fixture() -> pd.DataFrame:
    rows = []
    scores = {
        "classic_bubble": 0.75,
        "classic_insertion": 0.65,
        "classic_selection": 0.55,
        "disc_elite": 0.95,
        "disc_boundary": 0.60,
    }
    for policy_id, score in scores.items():
        for task_id, task_family in [("task_sort", "sorting"), ("task_frozen", "frozen")]:
            for seed in [0, 1, 2]:
                rows.append(
                    {
                        "policyId": policy_id,
                        "taskId": task_id,
                        "taskFamily": task_family,
                        "taskPanel": "fixture",
                        "inputProfile": "fixture",
                        "replicateIndex": seed,
                        "schedulerSeed": 100 + seed,
                        "tieBreakerSeed": 200 + seed,
                        "completionSuccess": score,
                        "finalSortednessScore": score,
                        "energyScore": score,
                        "robustnessScore": score if task_family == "frozen" else None,
                        "delayedGratificationScore": score / 2,
                        "aggregationAucScore": score / 3,
                        "oscillationScore": score,
                    }
                )
    return pd.DataFrame(rows)


class TestE03ClassicComparison(unittest.TestCase):
    def test_classic_family_from_lineage(self) -> None:
        self.assertEqual(classic_family_from_lineage("seed_bubble_template"), "bubble")
        self.assertEqual(classic_family_from_lineage("insertion_increasing_1"), "insertion")
        self.assertEqual(classic_family_from_lineage("selection_decreasing_1"), "selection")
        self.assertEqual(classic_family_from_lineage("random"), "none")

    def test_universe_pareto_and_pairwise_validation(self) -> None:
        universe = build_policy_comparison_universe(
            features_df=feature_fixture(),
            assignments_df=assignments_fixture(),
            corpus_df=corpus_fixture(),
            s12_policy_specs_df=s12_specs_fixture(),
        )
        selected = universe[universe["selectedForS13"]]
        self.assertEqual(set(selected[selected["comparisonRole"].eq("classic")]["classicFamily"]), {"bubble", "insertion", "selection"})
        self.assertIn("disc_elite", set(selected[selected["comparisonRole"].eq("discovered")]["policyId"]))
        missing = missing_comparison_records(universe)
        self.assertIn("unmeasured_generated", set(missing["policyId"]))

        pareto = pareto_front_table(universe)
        self.assertTrue(pareto["isParetoOptimal"].any())

        task_deltas = same_seed_task_delta_table(vector_fixture(), universe)
        self.assertFalse(task_deltas.empty)
        pair_summary = summarize_same_seed_pairs(task_deltas)
        self.assertFalse(pair_summary.empty)

        normalized = pd.DataFrame(
            {
                "policyId": universe["policyId"],
                "f1": range(len(universe)),
                "f2": [value * 2 for value in range(len(universe))],
            }
        )
        embedding_rows = []
        for method in ["pca", "spectral", "mds"]:
            for _, row in universe.iterrows():
                embedding_rows.append(
                    {
                        "policyId": row["policyId"],
                        "embeddingMethod": method,
                        "embeddingSeed": 0,
                        "embeddingDim1": float(len(str(row["policyId"]))),
                        "embeddingDim2": float(len(str(row["policyId"])) / 2),
                        "embeddingDim3": 0.0,
                        "embeddingDim4": 0.0,
                        "embeddingDim5": 0.0,
                    }
                )
        neighbors = pd.DataFrame(
            [{"policyId": "classic_bubble", "neighborPolicyId": "disc_elite", "neighborRank": 1}]
        )
        boundaries = pd.DataFrame(
            [
                {
                    "boundaryId": "b1",
                    "leftPolicyId": "classic_bubble",
                    "rightPolicyId": "disc_boundary",
                    "sortingSuccessTransition": True,
                }
            ]
        )
        pairwise = build_pairwise_comparison_table(
            policy_df=universe,
            same_seed_summary_df=pair_summary,
            pareto_df=pareto,
            normalized_matrix_df=normalized,
            embeddings_df=pd.DataFrame(embedding_rows),
            neighbors_df=neighbors,
            boundaries_df=boundaries,
        )
        self.assertEqual(len(pairwise), 3 * 2)
        self.assertIn("normalizedFeatureDistance", pairwise.columns)
        self.assertTrue(pairwise["directPhaseBoundaryCount"].max() >= 1)

        family = family_summary_table(pairwise, universe, pareto)
        checks = validate_classic_comparison_outputs(
            policy_df=universe,
            pairwise_df=pairwise,
            task_delta_df=task_deltas,
            pareto_df=pareto,
            family_summary_df=family,
            upstream_statuses={step: {"success": True, "status": "completed"} for step in ["S08", "S09", "S10", "S11", "S12"]},
            repo_test_payload={"success": True, "returnCode": 0, "command": ["unit"]},
            report_exists=True,
            figure_paths=["a.png", "b.png", "c.png"],
            s14_dir_exists=False,
        )
        self.assertTrue(checks["success"].all(), checks.to_string(index=False))


if __name__ == "__main__":
    unittest.main()
