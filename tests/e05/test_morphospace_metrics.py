"""E05 S05 morphospace metric tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.morphospace_metrics import (
    aggregate_morphospace_error,
    boundary_error,
    default_metric_specs,
    earth_mover_distance,
    evaluate_morphology_metrics,
    graph_edit_distance_proxy,
    hausdorff_distance,
    metric_spec_rows,
    reordered_target_state,
    swapped_target_state,
    target_neighborhood_error,
    topology_component_error,
)
from src.e05.targets import boundary_target, default_target_gallery, sorted_row_target


class MorphospaceMetricTests(unittest.TestCase):
    def test_metric_catalog_includes_required_families_and_labels_approximations(self) -> None:
        rows = metric_spec_rows(default_metric_specs())
        families = {row["metric_family"] for row in rows}
        self.assertTrue(
            {
                "spatial_identity",
                "target_neighborhood",
                "boundary",
                "topology",
                "shape_moments",
                "hausdorff",
                "earth_mover",
                "graph_edit",
            }.issubset(families)
        )
        approximations = [row for row in rows if row["exactness"] == "approximation"]
        self.assertTrue(approximations)
        self.assertTrue(all(row["approximation_label"] for row in approximations))

    def test_constructed_target_states_have_zero_all_metrics(self) -> None:
        for target in default_target_gallery():
            with self.subTest(target=target.target_id):
                results = evaluate_morphology_metrics(target, target.constructed_substrate(), state_label="target")
                self.assertTrue(all(result.value == 0.0 for result in results))

    def test_boundary_swap_produces_positive_boundary_and_graph_errors(self) -> None:
        target = boundary_target(4, 3)
        perturbed = swapped_target_state(target, 1, 5)

        boundary_value, _boundary_details = boundary_error(target, perturbed)
        graph_value, graph_details = graph_edit_distance_proxy(target, perturbed)
        neighborhood_value, _neighborhood_details = target_neighborhood_error(target, perturbed)

        self.assertGreater(boundary_value, 0.0)
        self.assertGreater(graph_value, 0.0)
        self.assertGreater(graph_details["node_mismatch"], 0.0)
        self.assertGreater(neighborhood_value, 0.0)

    def test_topology_hausdorff_and_emd_detect_label_displacement(self) -> None:
        target = boundary_target(4, 3)
        perturbed = swapped_target_state(target, 1, 5)

        topology_value, topology_details = topology_component_error(target, perturbed)
        hausdorff_value, _hausdorff_details = hausdorff_distance(target, perturbed)
        emd_value, _emd_details = earth_mover_distance(target, perturbed)

        self.assertGreaterEqual(topology_value, 0.0)
        self.assertTrue(topology_details["component_counts"])
        self.assertGreater(hausdorff_value, 0.0)
        self.assertGreater(emd_value, 0.0)

    def test_aggregate_metric_is_monotonic_for_ordered_row_sanity_case(self) -> None:
        target = sorted_row_target((5, 1, 4, 2, 3))
        exact = target.constructed_substrate()
        adjacent_swap = swapped_target_state(target, 1, 2)
        reversed_state = reordered_target_state(target, tuple(reversed(target.substrate.site_ids)))

        exact_error = aggregate_morphospace_error(evaluate_morphology_metrics(target, exact, state_label="exact"))
        adjacent_error = aggregate_morphospace_error(evaluate_morphology_metrics(target, adjacent_swap, state_label="adjacent_swap"))
        reversed_error = aggregate_morphospace_error(evaluate_morphology_metrics(target, reversed_state, state_label="reversed"))

        self.assertEqual(exact_error, 0.0)
        self.assertGreater(adjacent_error, exact_error)
        self.assertGreater(reversed_error, adjacent_error)

    def test_metric_result_rows_are_machine_readable(self) -> None:
        target = boundary_target(4, 3)
        results = evaluate_morphology_metrics(target, target.constructed_substrate())
        rows = [result.compact_dict() for result in results]
        self.assertEqual(len(rows), len(default_metric_specs()))
        self.assertTrue(all("metric_id" in row for row in rows))
        self.assertTrue(all(row["value"] == 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
