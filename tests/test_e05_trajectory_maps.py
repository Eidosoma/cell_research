from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from morphospace2d import (
    TRAJECTORY_FEATURE_SCHEMA_VERSION,
    TRAJECTORY_MAP_SCHEMA_VERSION,
    TraceSourceSpec,
    assign_route_and_failure_clusters,
    embedding_rows,
    embedding_stability_checks,
    fit_embedding_coordinates,
    harmonize_trace_table,
    metadata_confounding_checks,
    prepare_embedding_matrix,
    route_diversity_summary,
    summarize_trajectories,
    trajectory_validation_rows,
)


class TestE05TrajectoryMaps(unittest.TestCase):
    def test_harmonize_preserves_native_run_and_source_ids(self) -> None:
        spec = TraceSourceSpec("S07", "/tmp/s07_trace.parquet", "/tmp/s07_runs.parquet", trace_kind="scrambled_recovery")
        trace = pd.DataFrame(
            [
                {
                    "research_step_id": "S07",
                    "run_id": "S07::gradient::policy_a::seed1",
                    "target_id": "gradient",
                    "motif": "gradient",
                    "policy_id": "policy_a",
                    "policy_family": "classic_derived",
                    "seed": 1,
                    "step": 0,
                    "state_hash": "aaa",
                    "composite_error": 0.6,
                },
                {
                    "research_step_id": "S07",
                    "run_id": "S07::gradient::policy_a::seed1",
                    "target_id": "gradient",
                    "motif": "gradient",
                    "policy_id": "policy_a",
                    "policy_family": "classic_derived",
                    "seed": 1,
                    "step": 5,
                    "state_hash": "bbb",
                    "composite_error": 0.2,
                },
            ]
        )
        runs = pd.DataFrame([{"run_id": "S07::gradient::policy_a::seed1", "node_count": 6}])
        harmonized = harmonize_trace_table(spec, trace, runs)
        self.assertEqual(set(harmonized["source_step_id"]), {"S07"})
        self.assertEqual(set(harmonized["trajectory_id"]), {"S07::gradient::policy_a::seed1"})
        self.assertTrue((harmonized["schema_version"] == TRAJECTORY_FEATURE_SCHEMA_VERSION).all())
        self.assertAlmostEqual(float(harmonized["progress_fraction"].iloc[-1]), 2.0 / 3.0)
        self.assertEqual(str(harmonized["source_trace_path"].iloc[0]), "/tmp/s07_trace.parquet")

    def test_harmonize_reconstructs_s11_batch_run_ids(self) -> None:
        spec = TraceSourceSpec(
            "S11",
            "/tmp/s11_trace.parquet",
            "/tmp/s11_runs.parquet",
            trace_kind="gpu_label_dynamics",
            run_id_strategy="batch_index_seed_order",
        )
        trace = pd.DataFrame(
            [
                {
                    "target_id": "gpu_gradient",
                    "motif": "gradient",
                    "policy_id": "policy_gpu",
                    "policy_family": "local_only",
                    "batch_index": 0,
                    "step": 0,
                    "hamming_error": 0.7,
                },
                {
                    "target_id": "gpu_gradient",
                    "motif": "gradient",
                    "policy_id": "policy_gpu",
                    "policy_family": "local_only",
                    "batch_index": 1,
                    "step": 0,
                    "hamming_error": 0.5,
                },
            ]
        )
        runs = pd.DataFrame(
            [
                {"run_id": "S11::gpu_gradient::policy_gpu::seed10", "target_id": "gpu_gradient", "policy_id": "policy_gpu", "seed": 10},
                {"run_id": "S11::gpu_gradient::policy_gpu::seed11", "target_id": "gpu_gradient", "policy_id": "policy_gpu", "seed": 11},
            ]
        )
        harmonized = harmonize_trace_table(spec, trace, runs)
        self.assertEqual(harmonized.loc[harmonized["seed"].eq(10), "trajectory_id"].iloc[0], "S11::gpu_gradient::policy_gpu::seed10")
        self.assertEqual(harmonized.loc[harmonized["seed"].eq(11), "trajectory_id"].iloc[0], "S11::gpu_gradient::policy_gpu::seed11")

    def synthetic_feature_table(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for source_idx, source in enumerate(["S07", "S08", "S10", "S11"]):
            for policy_idx, policy_family in enumerate(["classic_derived", "random_null"]):
                for seed in range(2):
                    start_error = 0.75 - 0.04 * source_idx + 0.03 * seed
                    final_error = 0.12 + 0.12 * policy_idx + 0.02 * seed
                    mid_error = max(final_error, start_error - 0.35 + 0.04 * policy_idx)
                    values = [start_error, mid_error, final_error]
                    for snap, error in enumerate(values):
                        rows.append(
                            {
                                "schema_version": TRAJECTORY_FEATURE_SCHEMA_VERSION,
                                "research_step_id": "S12",
                                "source_step_id": source,
                                "source_trace_kind": "synthetic",
                                "source_trace_path": f"/tmp/{source}.parquet",
                                "source_run_path": f"/tmp/{source}_runs.parquet",
                                "trajectory_id": f"{source}::motif_{source_idx}::{policy_family}::seed{seed}",
                                "run_id": f"{source}::motif_{source_idx}::{policy_family}::seed{seed}",
                                "target_id": f"target_{source_idx}",
                                "task_id": None,
                                "motif": "gradient" if source_idx % 2 == 0 else "ring",
                                "policy_id": f"policy_{policy_family}",
                                "policy_family": policy_family,
                                "seed": seed,
                                "step": snap * 10,
                                "step_fraction": snap / 2,
                                "snapshot_index": snap,
                                "snapshot_reason": "interval",
                                "state_hash": f"hash-{source}-{policy_idx}-{seed}-{snap}",
                                "error_proxy": error,
                                "score_proxy": 1.0 - error,
                                "progress_fraction": (start_error - error) / start_error,
                                "edge_disagreement_feature": 0.2 + 0.1 * policy_idx,
                                "axis_strength_feature": 0.1 * source_idx,
                                "target_energy_feature": error * 2,
                                "cell_count_feature": 10.0,
                                "action_activity_feature": snap + policy_idx,
                                "is_global_information_baseline": False,
                                "is_local_only_policy": True,
                            }
                        )
        return pd.DataFrame(rows)

    def test_summary_embeddings_clusters_and_audits(self) -> None:
        features = self.synthetic_feature_table()
        summary = summarize_trajectories(features)
        self.assertEqual(len(summary), features["trajectory_id"].nunique())
        self.assertTrue((summary["path_curvature_proxy"] >= 1.0).all())
        self.assertTrue((summary["temporary_worsening_count"] >= 0).all())

        matrix, imputed, columns = prepare_embedding_matrix(summary)
        self.assertEqual(matrix.shape[0], len(summary))
        self.assertEqual(set(columns), set(imputed.columns))

        clustered, cluster_summary = assign_route_and_failure_clusters(summary, matrix)
        self.assertIn("route_cluster_id", clustered.columns)
        self.assertGreater(len(cluster_summary), 0)

        coords, method_info = fit_embedding_coordinates(matrix, random_state=12012)
        self.assertEqual(set(coords), {"pca", "diffusion_map_knn"})
        self.assertTrue((method_info["schema_version"] == TRAJECTORY_MAP_SCHEMA_VERSION).all())

        embedding = embedding_rows(clustered, coords)
        self.assertEqual(set(embedding["embedding_method"]), {"pca", "diffusion_map_knn"})
        self.assertEqual(embedding.groupby("embedding_method")["trajectory_id"].nunique().min(), len(summary))

        stability = embedding_stability_checks(clustered, matrix, random_states=(12012, 12013), max_items=12)
        self.assertTrue(stability["passed"].all(), stability.to_string(index=False))

        confounding = metadata_confounding_checks(embedding, metadata_fields=("source_step_id", "policy_family"), random_state=12012)
        computed = confounding[confounding["computed"]]
        self.assertGreaterEqual(len(computed), 2)
        self.assertTrue(np.isfinite(computed["cv_accuracy"]).all())

        diversity = route_diversity_summary(clustered, matrix)
        self.assertGreater(len(diversity), 0)

        source_catalog = pd.DataFrame(
            [
                {"source_step_id": source, "trace_available": True, "run_summary_available": True}
                for source in ["S07", "S08", "S10", "S11"]
            ]
        )
        validation = trajectory_validation_rows(
            source_catalog=source_catalog,
            feature_df=features,
            summary_df=clustered,
            embedding_df=embedding,
            method_info_df=method_info,
            stability_df=stability,
            confounding_df=confounding,
            route_diversity_df=diversity,
            cluster_df=cluster_summary,
            expected_sources=("S07", "S08", "S10", "S11"),
        )
        self.assertTrue(validation["success"].all(), validation.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
