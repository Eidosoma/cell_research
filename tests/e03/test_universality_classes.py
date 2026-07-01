"""S11 universality-class clustering tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.e03.rule_dsl import SIMPLE_SWAP_LEFT_POLICY, SIMPLE_SWAP_RIGHT_POLICY
from src.e03.universality_classes import (
    build_assignments,
    build_code_feature_frame,
    candidate_k_frame,
    cluster_summary_frame,
    exemplar_frame,
    feature_columns,
    feature_matrix,
    fit_kmeans,
    stability_frame,
    validation_frame,
)


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(123)
    rows = []
    norm_rows = []
    code_rows = []
    centers = [
        (0.9, 0.9, 10.0, 0.1),
        (0.6, 0.6, 20.0, 0.2),
        (0.3, 0.3, 30.0, -0.2),
        (0.05, 0.05, 40.0, -0.4),
    ]
    for cluster_index, (score, sortedness, work, xcenter) in enumerate(centers):
        for item in range(8):
            policy_id = f"p{cluster_index}_{item}"
            jitter = float(rng.normal(0, 0.01))
            classic_family = ""
            classic = False
            if item == 0 and cluster_index < 3:
                classic = True
                classic_family = ["Bubble", "Insertion", "Selection"][cluster_index]
            rows.append(
                {
                    "policy_id": policy_id,
                    "policy_name": policy_id,
                    "source_kind": "classic_dsl_seed" if classic else "unit",
                    "route": "jax_batch",
                    "screen_score": score + jitter,
                    "screen_heldout_final_sortedness_mean": sortedness + jitter,
                    "screen_heldout_work_mean": work,
                    "timeout_run_count": 0,
                    "quality_score": score + jitter,
                    "s08_archive_winner": item == 1,
                    "qd_candidate": item >= 4,
                    "classic_landmark": classic,
                    "classic_family": classic_family,
                    "s09_diagnostic_available": item == 2,
                    "embedding_x": xcenter + jitter,
                    "embedding_y": float(cluster_index),
                    "embedding_z": jitter,
                }
            )
            norm_rows.append(
                {
                    "policy_id": policy_id,
                    "policy_name": policy_id,
                    "screen_score": xcenter + jitter,
                    "screen_heldout_final_sortedness_mean": xcenter + jitter,
                    "screen_heldout_work_mean": float(cluster_index) + jitter,
                    "timeout_run_count": float(cluster_index),
                    "s09_mean_final_sortedness": xcenter + jitter,
                    "s09_boundary_count": float(cluster_index),
                }
            )
            code_rows.append(
                {
                    "policy_id": policy_id,
                    "dsl_source": SIMPLE_SWAP_RIGHT_POLICY if cluster_index == 0 else SIMPLE_SWAP_LEFT_POLICY,
                    "code_source": "unit",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(norm_rows), pd.DataFrame(code_rows)


class UniversalityClassTests(unittest.TestCase):
    def test_fit_kmeans_and_candidate_k_are_finite(self) -> None:
        _, normalized, _ = synthetic_inputs()
        cols = feature_columns(normalized)
        matrix = feature_matrix(normalized, cols)
        result = fit_kmeans(matrix, cluster_count=4, seed=55)
        self.assertEqual(len(result.labels), len(normalized))
        self.assertGreater(result.quality["silhouette_score"], 0.5)
        candidates = candidate_k_frame(matrix, k_values=(3, 4), seed=55)
        self.assertEqual(set(candidates["candidate_k"]), {3, 4})

    def test_code_features_assignments_and_exemplars(self) -> None:
        embeddings, normalized, code_table = synthetic_inputs()
        code_features = build_code_feature_frame(embeddings["policy_id"].tolist(), code_table)
        self.assertTrue(code_features["code_parse_success"].all())
        result = fit_kmeans(feature_matrix(normalized), cluster_count=4, seed=55)
        assignments = build_assignments(embeddings, normalized, code_features, result)
        summary = cluster_summary_frame(assignments)
        assignments = assignments.merge(summary[["class_id", "cautious_label", "label_basis", "label_confidence"]], on="class_id", how="left")
        exemplars = exemplar_frame(assignments, normalized)
        self.assertEqual(summary["class_id"].nunique(), 4)
        self.assertTrue(summary["cautious_label"].notna().all())
        self.assertEqual(exemplars["class_id"].nunique(), 4)
        self.assertIn("dsl_excerpt", exemplars.columns)

    def test_stability_and_validation_frame(self) -> None:
        embeddings, normalized, code_table = synthetic_inputs()
        code_features = build_code_feature_frame(embeddings["policy_id"].tolist(), code_table)
        result = fit_kmeans(feature_matrix(normalized), cluster_count=4, seed=55)
        assignments = build_assignments(embeddings, normalized, code_features, result)
        summary = cluster_summary_frame(assignments)
        assignments = assignments.merge(summary[["class_id", "cautious_label", "label_basis", "label_confidence"]], on="class_id", how="left")
        stability = stability_frame(normalized, embeddings, result.labels, cluster_count=4, seeds=(55, 56))
        candidates = candidate_k_frame(feature_matrix(normalized), k_values=(4,), seed=55)
        exemplars = exemplar_frame(assignments, normalized)
        validation = validation_frame(
            assignments=assignments,
            summary=summary,
            stability=stability,
            candidate_k=candidates,
            exemplars=exemplars,
            figure_exists=True,
            unit_success=True,
            expected_rows=len(assignments),
            cluster_count=4,
        )
        self.assertIn("metric_subset_stability", set(validation["validation_case"]))
        self.assertTrue(validation.loc[validation["validation_case"] == "input_rows_preserved", "success"].iloc[0])


if __name__ == "__main__":
    unittest.main()
