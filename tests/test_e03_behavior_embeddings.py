from __future__ import annotations

import unittest

import pandas as pd

from morphospace import (
    build_policy_feature_table,
    compute_embeddings,
    embedding_stability_table,
    nearest_neighbors_table,
    normalize_feature_table,
)


def corpus_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policyId": "pc_null",
                "family": "seed",
                "generationMethod": "seed_import",
                "lineageId": "seed_null_wait",
                "description": "null",
                "lineageDepth": 0,
                "generationIndex": 0,
                "ruleCount": 1,
                "predicateCount": 1,
                "updateCount": 0,
                "stateKeyCount": 0,
                "stochasticActionCount": 0,
                "targetUpdateCount": 0,
                "signalCount": 0,
                "complexityScore": 2,
                "actionCountsJson": '{"wait":1}',
            },
            {
                "policyId": "pc_bubble",
                "family": "hand",
                "generationMethod": "classic_parameter_sweep",
                "lineageId": "bubble_increasing_right_first_1",
                "description": "classic bubble",
                "lineageDepth": 0,
                "generationIndex": 1,
                "ruleCount": 2,
                "predicateCount": 2,
                "updateCount": 0,
                "stateKeyCount": 0,
                "stochasticActionCount": 0,
                "targetUpdateCount": 0,
                "signalCount": 0,
                "complexityScore": 4,
                "actionCountsJson": '{"swap_left":1,"swap_right":1}',
            },
            {
                "policyId": "pc_elite",
                "family": "generated",
                "generationMethod": "single_rule_sweep",
                "lineageId": "single_compare_right_gt_swap_right_1",
                "description": "elite",
                "lineageDepth": 0,
                "generationIndex": 2,
                "ruleCount": 1,
                "predicateCount": 1,
                "updateCount": 0,
                "stateKeyCount": 0,
                "stochasticActionCount": 0,
                "targetUpdateCount": 0,
                "signalCount": 0,
                "complexityScore": 2,
                "actionCountsJson": '{"swap_right":1}',
            },
            {
                "policyId": "pc_path",
                "family": "generated",
                "generationMethod": "single_rule_sweep",
                "lineageId": "single_compare_left_lt_swap_right_1",
                "description": "pathological",
                "lineageDepth": 0,
                "generationIndex": 3,
                "ruleCount": 1,
                "predicateCount": 1,
                "updateCount": 0,
                "stateKeyCount": 0,
                "stochasticActionCount": 0,
                "targetUpdateCount": 0,
                "signalCount": 0,
                "complexityScore": 2,
                "actionCountsJson": '{"swap_right":1}',
            },
        ]
    )


def competence_rows() -> pd.DataFrame:
    rows = []
    for policy_id, completion, sortedness, oscillation in [
        ("pc_null", 0.0, 0.4, 0.0),
        ("pc_bubble", 1.0, 1.0, 0.0),
        ("pc_elite", 1.0, 0.9, 0.1),
        ("pc_path", 0.0, 0.3, 0.8),
    ]:
        rows.append(
            {
                "policyId": policy_id,
                "taskId": "task",
                "taskFamily": "sorting",
                "taskPanel": "panel",
                "completed": bool(completion),
                "stopReason": "sorted" if completion else "cap",
                "completionSuccess": completion,
                "finalSortednessScore": sortedness,
                "finalMonotonicityScore": sortedness,
                "energyScore": 0.5,
                "oscillationProxy": oscillation,
                "oscillationScore": 1.0 - oscillation,
                "failureOscillationScore": min(completion, 1.0 - oscillation),
                "swapCount": 1,
                "comparisonCount": 2,
                "activationCount": 3,
                "eventCount": 2,
            }
        )
    return pd.DataFrame(rows)


def run_rows() -> pd.DataFrame:
    rows = []
    for policy_id, completion, sortedness, oscillation in [
        ("pc_null", 0.0, 0.4, 0.0),
        ("pc_bubble", 1.0, 1.0, 0.0),
        ("pc_elite", 1.0, 0.9, 0.1),
        ("pc_path", 0.0, 0.3, 0.8),
    ]:
        rows.append(
            {
                "policyId": policy_id,
                "taskId": "run_task",
                "taskFamily": "sorting",
                "completed": bool(completion),
                "completedNumeric": completion,
                "stopReason": "sorted" if completion else "cap",
                "finalSortednessScore": sortedness,
                "finalSortednessPercent": sortedness * 100,
                "oscillationProxy": oscillation,
                "finalAggregation": 1.0,
                "traceStateCount": 2,
                "swapCount": 1,
                "comparisonCount": 2,
                "activationCount": 3,
                "eventCount": 2,
                "runtimeSeconds": 0.01,
            }
        )
    return pd.DataFrame(rows)


class TestE03BehaviorEmbeddings(unittest.TestCase):
    def test_feature_table_roles_and_normalization(self) -> None:
        elites = pd.DataFrame([{"policyId": "pc_elite", "eliteRank": 1, "qualityScore": 0.9, "noveltyScore": 0.5}])
        boundaries = pd.DataFrame(
            [
                {
                    "leftPolicyId": "pc_path",
                    "rightPolicyId": "pc_elite",
                    "boundaryScore": 2.0,
                    "transitionKindCount": 2,
                    "sortingSuccessTransition": True,
                    "sortednessTransition": True,
                    "delayedGratificationTransition": False,
                    "aggregationTransition": False,
                    "oscillationTransition": True,
                    "robustnessTransition": False,
                }
            ]
        )
        variance = pd.DataFrame(
            [
                {
                    "policyId": "pc_path",
                    "completionRateSd": 0.1,
                    "finalSortednessScoreSd": 0.2,
                    "oscillationProxySd": 0.3,
                }
            ]
        )
        features = build_policy_feature_table(
            corpus_df=corpus_rows(),
            s07_competence_df=competence_rows(),
            s07_run_df=run_rows(),
            s08_elites_df=elites,
            s08_validation_df=run_rows().iloc[:0],
            s09_competence_df=competence_rows(),
            s09_run_df=run_rows(),
            s09_boundaries_df=boundaries,
            s09_variance_df=variance,
        )
        self.assertEqual(len(features), 4)
        self.assertTrue(features.loc[features["policyId"].eq("pc_null"), "isNullPolicy"].iloc[0])
        self.assertTrue(features.loc[features["policyId"].eq("pc_bubble"), "isClassicPolicy"].iloc[0])
        self.assertTrue(features.loc[features["policyId"].eq("pc_elite"), "isS08Elite"].iloc[0])
        self.assertTrue(features.loc[features["policyId"].eq("pc_path"), "isPathologicalPolicy"].iloc[0])

        matrix, spec = normalize_feature_table(features)
        self.assertEqual(len(matrix), 4)
        self.assertGreater(matrix.shape[1], 10)
        self.assertEqual(len(spec), matrix.shape[1] - 1)
        self.assertFalse(matrix.drop(columns=["policyId"]).isna().any().any())

    def test_embeddings_stability_and_neighbors(self) -> None:
        features = build_policy_feature_table(
            corpus_df=corpus_rows(),
            s07_competence_df=competence_rows(),
            s07_run_df=run_rows(),
            s08_elites_df=pd.DataFrame([{"policyId": "pc_elite", "eliteRank": 1, "qualityScore": 0.9}]),
            s08_validation_df=run_rows().iloc[:0],
            s09_competence_df=competence_rows(),
            s09_run_df=run_rows(),
            s09_boundaries_df=pd.DataFrame(
                [
                    {
                        "leftPolicyId": "pc_path",
                        "rightPolicyId": "pc_elite",
                        "boundaryScore": 2.0,
                        "transitionKindCount": 2,
                        "sortingSuccessTransition": True,
                        "sortednessTransition": True,
                        "delayedGratificationTransition": False,
                        "aggregationTransition": False,
                        "oscillationTransition": True,
                        "robustnessTransition": False,
                    }
                ]
            ),
            s09_variance_df=pd.DataFrame(),
        )
        matrix, _ = normalize_feature_table(features)
        embeddings = compute_embeddings(matrix, methods=("pca", "mds"), seeds=(0, 1))
        self.assertEqual(set(embeddings["embeddingMethod"]), {"pca", "mds"})
        self.assertEqual(len(embeddings), len(matrix) * 2 * 2)
        self.assertFalse(embeddings[["embeddingDim1", "embeddingDim2"]].isna().any().any())
        stability = embedding_stability_table(embeddings, neighbor_k=2)
        self.assertGreaterEqual(len(stability), 3)
        neighbors = nearest_neighbors_table(matrix, neighbor_k=2)
        self.assertEqual(len(neighbors), len(matrix) * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
