"""E05 S12 morphospace trajectory embedding tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.morphospace_trajectories import (
    FEATURE_COLUMNS,
    build_trajectory_point_table,
    compute_route_metrics,
    embed_points,
    metric_scaling_stability,
    normalize_s11_trace,
    normalize_s07_trace,
    seed_split_stability,
    target_reference_rows,
)


class MorphospaceTrajectoryTests(unittest.TestCase):
    def test_s07_trace_normalization_labels_initial_and_final(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "research_step_id": "S07",
                    "target_id": "toy",
                    "target_kind": "gradient",
                    "policy_id": "p",
                    "policy_family": "pf",
                    "simulation_seed": 2,
                    "event_step": 0,
                    "accepted_swaps": 0,
                    "attempted_swaps": 0,
                    "rejected_actions": 0,
                    "energy_cost": 0.0,
                    "target_error": 0.5,
                    "aggregate_morphospace_error": 0.4,
                },
                {
                    "research_step_id": "S07",
                    "target_id": "toy",
                    "target_kind": "gradient",
                    "policy_id": "p",
                    "policy_family": "pf",
                    "simulation_seed": 2,
                    "event_step": 10,
                    "accepted_swaps": 1,
                    "attempted_swaps": 1,
                    "rejected_actions": 0,
                    "energy_cost": 1.0,
                    "target_error": 0.2,
                    "aggregate_morphospace_error": 0.1,
                },
            ]
        )
        normalized = normalize_s07_trace(raw)
        embedded, _variance = embed_points(pd.concat([normalized, target_reference_rows(normalized)], ignore_index=True))

        self.assertEqual(set(embedded["state_role"]), {"initial", "final", "target_reference"})
        self.assertTrue(all(column in embedded.columns for column in FEATURE_COLUMNS))
        self.assertEqual(float(embedded.loc[embedded["state_role"] == "target_reference", "target_error"].iloc[0]), 0.0)

    def test_s11_trace_normalization_preserves_s11_as_artifact_source(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "research_step_id": "S11",
                    "run_uid": "gpu-run",
                    "task_id": "s07_scrambled_ap_gradient",
                    "source_research_step_id": "S07",
                    "source_task_type": "scrambled_embryo",
                    "target_id": "ap_gradient",
                    "target_kind": "gradient",
                    "policy_id": "s07_local_target_neighbor_descent",
                    "simulation_seed": 2,
                    "event_step": 0,
                    "event_cap": 2,
                    "accepted_swaps": 0,
                    "attempted_swaps": 0,
                    "rejected_actions": 0,
                    "wait_actions": 0,
                    "energy_cost": 0.0,
                    "target_error": 0.2,
                    "displacement_fraction": 0.1,
                }
            ]
        )
        normalized = normalize_s11_trace(raw)

        self.assertEqual(set(normalized["source_research_step_id"]), {"S11"})
        self.assertEqual(set(normalized["source_task_type"]), {"scrambled_embryo"})

    def test_route_metrics_detect_curvature_and_temporary_away(self) -> None:
        rows = []
        for step, target_error, x, y in ((0, 0.5, 0.0, 0.0), (1, 0.6, 1.0, 1.0), (2, 0.2, 2.0, 0.0)):
            rows.append(
                {
                    "research_step_id": "S12",
                    "run_uid": "run",
                    "source_research_step_id": "S07",
                    "source_task_type": "scrambled",
                    "task_id": "task",
                    "target_id": "target",
                    "target_kind": "gradient",
                    "policy_id": "policy",
                    "policy_family": "family",
                    "simulation_seed": 1,
                    "endpoint_only": False,
                    "event_step": step,
                    "event_cap": 2,
                    "event_progress": step / 2,
                    "target_error": target_error,
                    "morph_x": x,
                    "morph_y": y,
                    "is_target_reference": False,
                }
            )
        metrics, summary = compute_route_metrics(pd.DataFrame(rows))

        self.assertEqual(len(metrics), 1)
        self.assertGreater(metrics["path_curvature"].iloc[0], 1.0)
        self.assertEqual(metrics["temporary_away_steps"].iloc[0], 1)
        self.assertEqual(len(summary), 1)

    def test_build_point_table_uses_endpoint_s09_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            (root / "traces").mkdir()
            s07 = pd.DataFrame(
                [
                    {
                        "research_step_id": "S07",
                        "target_id": "toy",
                        "target_kind": "gradient",
                        "policy_id": "p",
                        "policy_family": "pf",
                        "simulation_seed": 1,
                        "event_step": 0,
                        "accepted_swaps": 0,
                        "attempted_swaps": 0,
                        "rejected_actions": 0,
                        "energy_cost": 0.0,
                        "target_error": 0.2,
                        "aggregate_morphospace_error": 0.2,
                    },
                    {
                        "research_step_id": "S07",
                        "target_id": "toy",
                        "target_kind": "gradient",
                        "policy_id": "p",
                        "policy_family": "pf",
                        "simulation_seed": 1,
                        "event_step": 5,
                        "accepted_swaps": 1,
                        "attempted_swaps": 1,
                        "rejected_actions": 0,
                        "energy_cost": 1.0,
                        "target_error": 0.1,
                        "aggregate_morphospace_error": 0.1,
                    },
                ]
            )
            s07.to_parquet(root / "traces" / "e05_scrambled_embryo_trace_table.parquet")
            s09_summary = pd.DataFrame(
                [
                    {
                        "target_id": "scaled",
                        "target_kind": "gradient",
                        "task_type": "scrambled_embryo",
                        "config_id": "scaled:small",
                        "scale_label": "small",
                        "policy_id": "p",
                        "policy_family": "pf",
                        "simulation_seed": 3,
                        "event_cap": 10,
                        "initial_target_error": 0.4,
                        "final_target_error": 0.2,
                        "initial_aggregate_morphospace_error": 0.3,
                        "final_aggregate_morphospace_error": 0.15,
                        "initial_displacement_fraction": 0.8,
                        "final_displacement_fraction": 0.4,
                        "accepted_swaps": 2,
                        "attempted_swaps": 2,
                        "rejected_actions": 0,
                        "wait_actions": 8,
                        "total_energy_cost": 2.0,
                        "population_initial": 4,
                        "site_count": 4,
                    }
                ]
            )
            s09_metric = pd.DataFrame(
                [
                    {
                        "target_id": "scaled",
                        "task_type": "scrambled_embryo",
                        "config_id": "scaled:small",
                        "scale_label": "small",
                        "policy_id": "p",
                        "simulation_seed": 3,
                        "benchmark_state_label": "initial",
                        "metric_id": "target_identity_error",
                        "value": 0.4,
                    },
                    {
                        "target_id": "scaled",
                        "task_type": "scrambled_embryo",
                        "config_id": "scaled:small",
                        "scale_label": "small",
                        "policy_id": "p",
                        "simulation_seed": 3,
                        "benchmark_state_label": "final",
                        "metric_id": "target_identity_error",
                        "value": 0.2,
                    },
                ]
            )
            s09_summary.to_parquet(root / "results" / "e05_scaling_tests.parquet")
            s09_metric.to_parquet(root / "results" / "e05_scaling_metric_rows.parquet")
            pd.DataFrame().to_parquet(root / "results" / "e05_metric_catalog.parquet")

            points, sources, _metrics = build_trajectory_point_table(root)

        self.assertIn("S09", set(points["source_research_step_id"]))
        self.assertTrue(points["is_target_reference"].any())
        self.assertTrue(sources.loc[sources["source_research_step_id"] == "S09", "used_for_embedding"].iloc[0])

    def test_stability_checks_return_success_on_real_artifact_subset_when_available(self) -> None:
        artifacts = Path("/artifacts")
        if not (artifacts / "traces" / "e05_scrambled_embryo_trace_table.parquet").exists():
            self.skipTest("S07 artifacts are not available")
        points, _sources, _metrics = build_trajectory_point_table(artifacts)
        subset = pd.concat(
            [
                points[points["is_target_reference"]],
                points[(~points["is_target_reference"]) & (points["source_research_step_id"].isin(["S07", "S10"]))].head(500),
            ],
            ignore_index=True,
        )
        seed_result = seed_split_stability(subset)
        scaling_result = metric_scaling_stability(subset)

        self.assertIn("stability_score", seed_result)
        self.assertIn("stability_score", scaling_result)
        self.assertGreaterEqual(seed_result["stability_score"], 0.0)
        self.assertGreaterEqual(scaling_result["stability_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
