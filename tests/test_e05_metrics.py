from __future__ import annotations

import unittest

from morphospace2d import (
    METRIC_SCHEMA_VERSION,
    add_extra_cell,
    build_boundary_target,
    build_organ_like_target,
    build_standard_target_library,
    evaluate_morphospace_metrics,
    metric_catalog,
    remove_positions,
    rotate_state_180,
    scrambled_state,
    trajectory_curvature,
)


class TestE05MorphospaceMetrics(unittest.TestCase):
    def test_metric_catalog_covers_required_metrics(self) -> None:
        metric_ids = {row["metric_id"] for row in metric_catalog()}
        required = {
            "target_energy",
            "target_neighborhood_error",
            "graph_edit_approx",
            "boundary_error",
            "topology_error",
            "shape_moment_error",
            "hausdorff_distance",
            "earth_mover_distance",
            "trajectory_curvature",
        }
        self.assertTrue(required.issubset(metric_ids))

    def test_exact_target_states_have_zero_error(self) -> None:
        for target in build_standard_target_library():
            with self.subTest(target=target.target_id):
                result = evaluate_morphospace_metrics(target, target.target_state())
                self.assertEqual(result["schemaVersion"], METRIC_SCHEMA_VERSION)
                self.assertAlmostEqual(result["compositeError"], 0.0)
                for metric in result["metrics"]:
                    self.assertAlmostEqual(metric["value"], 0.0)
                    self.assertAlmostEqual(metric["normalizedValue"], 0.0)

    def test_scrambled_identity_state_is_detected_by_identity_sensitive_metrics(self) -> None:
        target = build_organ_like_target()
        result = evaluate_morphospace_metrics(target, scrambled_state(target))
        by_metric = {metric["metricId"]: metric for metric in result["metrics"]}
        self.assertGreater(by_metric["target_energy"]["value"], 0.0)
        self.assertGreater(by_metric["target_neighborhood_error"]["value"], 0.0)
        self.assertGreater(by_metric["graph_edit_approx"]["value"], 0.0)
        self.assertGreater(by_metric["earth_mover_distance"]["value"], 0.0)

    def test_hole_state_has_shape_boundary_topology_and_hausdorff_error(self) -> None:
        target = build_boundary_target()
        hole_state = remove_positions(target.target_state(), [(2, 2)])
        result = evaluate_morphospace_metrics(target, hole_state)
        by_metric = {metric["metricId"]: metric for metric in result["metrics"]}
        self.assertGreater(by_metric["boundary_error"]["value"], 0.0)
        self.assertGreater(by_metric["topology_error"]["value"], 0.0)
        self.assertGreater(by_metric["shape_moment_error"]["value"], 0.0)
        self.assertGreater(by_metric["hausdorff_distance"]["value"], 0.0)

    def test_extra_cell_state_is_handled_without_invalid_metrics(self) -> None:
        target = build_boundary_target()
        extra_identity = next(iter(target.target_state().values()))
        extra_state = add_extra_cell(target.target_state(), (99, 99), extra_identity)
        result = evaluate_morphospace_metrics(target, extra_state)
        by_metric = {metric["metricId"]: metric for metric in result["metrics"]}
        self.assertGreater(by_metric["graph_edit_approx"]["value"], 0.0)
        self.assertGreater(by_metric["hausdorff_distance"]["value"], 0.0)
        self.assertGreater(by_metric["earth_mover_distance"]["value"], 0.0)

    def test_rotated_state_is_detected_by_assignment_and_boundary_metrics(self) -> None:
        target = build_organ_like_target()
        rotated = rotate_state_180(target)
        result = evaluate_morphospace_metrics(target, rotated)
        by_metric = {metric["metricId"]: metric for metric in result["metrics"]}
        self.assertGreater(by_metric["target_energy"]["value"], 0.0)
        self.assertGreater(by_metric["boundary_error"]["value"], 0.0)
        self.assertGreater(by_metric["earth_mover_distance"]["value"], 0.0)

    def test_trajectory_curvature_tie_cases(self) -> None:
        straight = trajectory_curvature([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
        kinked = trajectory_curvature([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])
        stationary = trajectory_curvature([(1.0, 1.0), (1.0, 1.0)])
        self.assertEqual(straight["value"], 0.0)
        self.assertEqual(stationary["value"], 0.0)
        self.assertGreater(kinked["value"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
