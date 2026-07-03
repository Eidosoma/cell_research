"""Tests for E07 S13 substrate-transfer helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.e07.substrate_transfer_schema import (
    transfer_mapping_id,
    transfer_score,
    validate_substrate_transfer_artifacts,
    validation_summary,
)


class SubstrateTransferSchemaTests(unittest.TestCase):
    def test_mapping_ids_are_stable(self) -> None:
        left = {"source_substrate_kind": "array_1d", "target_substrate_kind": "embedded_square_grid_2d", "algorithm": "bubble"}
        right = {"algorithm": "bubble", "target_substrate_kind": "embedded_square_grid_2d", "source_substrate_kind": "array_1d"}
        self.assertEqual(transfer_mapping_id(left), transfer_mapping_id(right))

    def test_transfer_score_bounds_and_priority(self) -> None:
        self.assertEqual(transfer_score(blocked=True, exact_trajectory_match=True), 0.0)
        self.assertEqual(transfer_score(exact_trajectory_match=True, target_final_sortedness_percent=25.0), 1.0)
        self.assertAlmostEqual(transfer_score(target_final_sortedness_percent=82.5), 0.825)
        self.assertEqual(transfer_score(target_recovery_fraction=1.5), 1.0)
        self.assertEqual(transfer_score(target_recovery_fraction=-0.2), 0.0)

    def test_validate_substrate_transfer_artifacts(self) -> None:
        base_mapping = {
            "source_experiment_id": "E03",
            "target_experiment_id": "E05",
            "source_substrate_kind": "array_1d",
            "target_substrate_kind": "embedded_square_grid_2d",
            "execution_path": "src.e05.embedded_1d.run_adjacent_sort",
            "adapter_built_in_s13": False,
            "mapping_status": "executable_existing_path",
            "mapping_kind": "e03_to_e05_embedded_row",
        }
        mapping_id = transfer_mapping_id(base_mapping)
        mappings = pd.DataFrame([{**base_mapping, "mapping_id": mapping_id}])
        results = pd.DataFrame(
            [
                {
                    "mapping_id": mapping_id,
                    "mapping_kind": "e03_to_e05_embedded_row",
                    "target_experiment_id": "E05",
                    "evaluation_kind": "fresh_existing_path_evaluation",
                    "control_family": "cross_substrate_transfer",
                    "exact_trajectory_match": True,
                    "transfer_score": 1.0,
                },
                {
                    "mapping_id": mapping_id,
                    "mapping_kind": "within_1d_baseline",
                    "target_experiment_id": "E03",
                    "evaluation_kind": "fresh_existing_path_evaluation",
                    "control_family": "within_1d_baseline",
                    "exact_trajectory_match": True,
                    "transfer_score": 1.0,
                },
                {
                    "mapping_id": "context",
                    "mapping_kind": "e05_reference_context",
                    "target_experiment_id": "E05",
                    "evaluation_kind": "existing_e05_reference_context",
                    "control_family": "e05_context_control",
                    "transfer_score": 0.5,
                },
                {
                    "mapping_id": "blocker",
                    "mapping_kind": "s12_dsl_to_e05_embedded_row",
                    "target_experiment_id": "E05",
                    "evaluation_kind": "transfer_blocker",
                    "control_family": "blocked_transfer",
                    "transfer_score": 0.0,
                },
            ]
        )
        manifest = {
            "transferMappingArtifactSha256": "abc",
            "mappingsFrozenAtUtc": "2026-07-03T00:00:00+00:00",
            "evaluationStartedAtUtc": "2026-07-03T00:01:00+00:00",
            "noNewE05OrE06AdaptersBuilt": True,
        }
        checks = validate_substrate_transfer_artifacts(mappings, results, manifest)
        self.assertTrue(bool(checks["success"].all()), checks.to_string())
        self.assertTrue(validation_summary(checks)["allPassed"])


if __name__ == "__main__":
    unittest.main()
