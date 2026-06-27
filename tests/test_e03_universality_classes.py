from __future__ import annotations

import unittest

import pandas as pd

from morphospace import (
    identify_universality_classes,
    validate_universality_outputs,
)


def toy_feature_table() -> pd.DataFrame:
    rows = []
    groups = [
        ("null", 0.0, 0.45, 0.02, 1.0, 0.0, False, True, False, False, False),
        ("classic", 0.95, 0.96, 0.03, 0.0, 1.0, True, False, False, False, False),
        ("elite", 0.90, 0.92, 0.08, 0.0, 1.0, False, False, True, False, False),
        ("boundary", 0.35, 0.68, 0.18, 0.0, 1.0, False, False, False, True, False),
        ("pathological", 0.05, 0.42, 0.52, 0.0, 1.0, False, False, False, True, True),
    ]
    for group_index, (
        group,
        completion,
        sortedness,
        oscillation,
        wait_count,
        swap_count,
        is_classic,
        is_null,
        is_elite,
        is_phase,
        is_pathological,
    ) in enumerate(groups):
        for offset in range(4):
            policy_id = f"toy_{group}_{offset}"
            rows.append(
                {
                    "policyId": policy_id,
                    "primaryRole": "elite" if is_elite else group,
                    "family": "toy",
                    "generationMethod": "toy_method",
                    "lineageId": f"{group}_lineage_{offset}",
                    "classicFamily": "bubble" if is_classic else "none",
                    "description": f"{group} policy",
                    "isNullPolicy": is_null,
                    "isClassicPolicy": is_classic,
                    "isS08Elite": is_elite and offset < 2,
                    "isPhaseBoundaryPolicy": is_phase,
                    "isPathologicalPolicy": is_pathological,
                    "actionTotalCount": wait_count + swap_count,
                    "swapActionCount": swap_count,
                    "waitActionCount": wait_count,
                    "swapTargetActionCount": 1.0 if is_phase else 0.0,
                    "stochasticActionCount": 1.0 if is_phase else 0.0,
                    "targetUpdateCount": 1.0 if is_phase else 0.0,
                    "stateKeyCount": 1.0 if is_phase else 0.0,
                    "signalCount": 0.0,
                    "complexityScore": 2.0 + group_index,
                    "s09_completionSuccessMean": completion + offset * 0.01,
                    "s09_finalSortednessScoreMean": sortedness + offset * 0.005,
                    "s09Trace_oscillationProxyMean": oscillation + offset * 0.01,
                    "s09_aggregationFinalMean": 0.75 + group_index * 0.05,
                    "s09_robustnessScoreMean": 0.5 + group_index * 0.05,
                    "s09_delayedGratificationMean": 0.1 * group_index,
                    "s09_completionRateTaskFamily_chimera": 0.4 if is_phase else 0.0,
                    "s09_completionRateTaskFamily_frozen": completion,
                    "s09_completionRateTaskFamily_sorting_duplicate_values": completion,
                    "phaseBoundaryInvolvementCount": 2.0 if is_phase else 0.0,
                    "nearBoundaryVarianceRecordCount": 1.0 if is_phase else 0.0,
                    "s08Elite_qualityScore": 0.8 if is_elite and offset < 2 else None,
                    "evidenceLayerCount": 3,
                }
            )
    return pd.DataFrame(rows)


def toy_normalized_matrix(features: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "isNullPolicy",
        "isClassicPolicy",
        "isS08Elite",
        "isPhaseBoundaryPolicy",
        "isPathologicalPolicy",
        "waitActionCount",
        "swapActionCount",
        "swapTargetActionCount",
        "stochasticActionCount",
        "targetUpdateCount",
        "stateKeyCount",
        "complexityScore",
        "s09_completionSuccessMean",
        "s09_finalSortednessScoreMean",
        "s09Trace_oscillationProxyMean",
        "s09_aggregationFinalMean",
        "s09_completionRateTaskFamily_chimera",
        "phaseBoundaryInvolvementCount",
    ]
    matrix = features[["policyId", *numeric_columns]].copy()
    for column in numeric_columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").fillna(0.0).astype(float)
        std = matrix[column].std(ddof=0)
        if std > 0:
            matrix[column] = (matrix[column] - matrix[column].mean()) / std
    return matrix


def toy_embeddings(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    centers = {
        "null": (-3.0, 0.0),
        "classic": (0.0, 3.0),
        "elite": (0.0, 3.0),
        "boundary": (3.0, -1.5),
        "pathological": (4.0, 1.5),
    }
    for method in ["pca", "spectral", "mds"]:
        for _, row in features.iterrows():
            x, y = centers[str(row["primaryRole"])]
            offset = int(str(row["policyId"]).split("_")[-1]) * 0.05
            rows.append(
                {
                    "policyId": row["policyId"],
                    "embeddingMethod": method,
                    "embeddingSeed": 0,
                    "embeddingDim1": x + offset,
                    "embeddingDim2": y + offset,
                    "embeddingDim3": offset,
                    "embeddingDim4": 0.0,
                    "embeddingDim5": 0.0,
                }
            )
    return pd.DataFrame(rows)


def toy_neighbors(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ["null", "classic", "elite", "boundary", "pathological"]:
        ids = [f"toy_{group}_{idx}" for idx in range(4)]
        for policy_id in ids:
            rank = 1
            for neighbor_id in ids:
                if neighbor_id == policy_id:
                    continue
                rows.append(
                    {
                        "policyId": policy_id,
                        "neighborPolicyId": neighbor_id,
                        "neighborRank": rank,
                        "featureDistance": float(rank) * 0.1,
                    }
                )
                rank += 1
    return pd.DataFrame(rows)


class TestE03UniversalityClasses(unittest.TestCase):
    def test_identify_universality_classes_tables(self) -> None:
        features = toy_feature_table()
        matrix = toy_normalized_matrix(features)
        embeddings = toy_embeddings(features)
        neighbors = toy_neighbors(features)

        result = identify_universality_classes(
            feature_df=features,
            normalized_matrix_df=matrix,
            embeddings_df=embeddings,
            neighbors_df=neighbors,
            candidate_k=(5,),
            n_components=5,
            exemplars_per_class=3,
        )

        assignments = result["class_assignments"]
        self.assertEqual(set(assignments["policyId"]), set(features["policyId"]))
        self.assertGreaterEqual(result["class_summary"]["universalityClassId"].nunique(), 5)
        self.assertIn("seed_repeat", set(result["cluster_robustness"]["comparisonType"]))
        self.assertIn("feature_subset", set(result["cluster_robustness"]["comparisonType"]))
        self.assertIn("embedding_coordinates", set(result["cluster_robustness"]["comparisonType"]))
        self.assertGreaterEqual(
            len(result["class_exemplars"]),
            result["class_summary"]["universalityClassId"].nunique() * 2,
        )
        placement_roles = ",".join(result["classic_null_elite_placements"]["placementRole"].astype(str))
        self.assertIn("classic", placement_roles)
        self.assertIn("null", placement_roles)
        self.assertIn("s08_elite", placement_roles)
        all_alignment = result["neighbor_alignment"][result["neighbor_alignment"]["universalityClassId"].eq("ALL")]
        self.assertFalse(all_alignment.empty)
        self.assertGreater(float(all_alignment["meanSameClassNeighborRate"].iloc[0]), 0.5)

    def test_validate_universality_outputs(self) -> None:
        features = toy_feature_table()
        matrix = toy_normalized_matrix(features)
        embeddings = toy_embeddings(features)
        neighbors = toy_neighbors(features)
        result = identify_universality_classes(
            feature_df=features,
            normalized_matrix_df=matrix,
            embeddings_df=embeddings,
            neighbors_df=neighbors,
            candidate_k=(5,),
            n_components=5,
            exemplars_per_class=3,
        )

        validation = validate_universality_outputs(
            feature_df=features,
            normalized_matrix_df=matrix,
            embeddings_df=embeddings,
            neighbors_df=neighbors,
            class_assignments_df=result["class_assignments"],
            class_summary_df=result["class_summary"],
            robustness_df=result["cluster_robustness"],
            exemplars_df=result["class_exemplars"],
            placements_df=result["classic_null_elite_placements"],
            neighbor_alignment_df=result["neighbor_alignment"],
            upstream_statuses={step: {"success": True, "status": "completed"} for step in ["S07", "S08", "S09", "S10"]},
            repo_test_payload={"success": True, "returnCode": 0, "command": ["unittest"]},
            taxonomy_report_exists=True,
            figure_paths=["figure_a.png", "figure_b.png"],
            s12_dir_exists=False,
        )
        self.assertTrue(validation["success"].all(), validation.to_string())


if __name__ == "__main__":
    unittest.main(verbosity=2)
