"""Tests for E07 S12 inverse-design helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.e07.inverse_design_schema import (
    design_record_id,
    e03_proxy_target_score,
    target_profile_id,
    timestamp_order_ok,
    validate_inverse_design_artifacts,
)


class InverseDesignSchemaTests(unittest.TestCase):
    def test_hash_ids_are_stable(self) -> None:
        self.assertEqual(target_profile_id({"b": 2, "a": 1}), target_profile_id({"a": 1, "b": 2}))
        self.assertEqual(design_record_id({"dsl": "x", "rank": 1}), design_record_id({"rank": 1, "dsl": "x"}))

    def test_proxy_target_score_rewards_target_match(self) -> None:
        strong = e03_proxy_target_score(final_sortedness=0.95, work_per_item=2.0, dg_recovery_proxy=0.03, invalid=False)
        weak = e03_proxy_target_score(final_sortedness=0.4, work_per_item=8.0, dg_recovery_proxy=0.3, invalid=False)
        invalid = e03_proxy_target_score(final_sortedness=0.95, work_per_item=2.0, dg_recovery_proxy=0.03, invalid=True)
        self.assertGreater(strong, weak)
        self.assertGreater(strong, invalid)
        self.assertGreaterEqual(strong, 0.0)
        self.assertLessEqual(strong, 1.0)

    def test_validate_inverse_design_artifacts(self) -> None:
        targets = pd.DataFrame(
            {
                "target_axis_id": [
                    "order_quality_proxy",
                    "energy_efficiency_proxy",
                    "moderate_dg_proxy",
                    "aggregation_blocker",
                    "repair_robustness_blocker",
                ]
            }
        )
        designs = pd.DataFrame(
            [
                {
                    "design_id": "d1",
                    "design_scope": "e03_dsl_array_world",
                    "source_experiment_id": "E03",
                    "representation_type": "dsl",
                    "parser_validation_status": "parsed",
                    "executable_in_s12": True,
                }
            ]
        )
        validations = pd.DataFrame(
            [
                {
                    "design_id": "d1",
                    "validation_kind": "heldout_e03_simulation",
                    "validation_split": "heldout",
                    "source_experiment_id": "E03",
                    "world_id": "w1",
                    "control_family": "source_control",
                },
                {
                    "design_id": "d1",
                    "validation_kind": "heldout_e03_simulation",
                    "validation_split": "heldout",
                    "source_experiment_id": "E03",
                    "world_id": "w2",
                    "control_family": "metric_only_control",
                },
                {
                    "design_id": "d1",
                    "validation_kind": "heldout_e03_simulation",
                    "validation_split": "heldout",
                    "source_experiment_id": "E03",
                    "world_id": "w3",
                    "control_family": "class_status_control",
                },
                {
                    "design_id": "",
                    "validation_kind": "simulator_blocker",
                    "validation_split": "not_executable",
                    "source_experiment_id": "E06",
                    "world_id": "",
                    "control_family": "missingness_control",
                },
            ]
        )
        target_manifest = {
            "targetProfileArtifactSha256": "abc",
            "targetProfilesFrozenAtUtc": "2026-07-03T00:00:00+00:00",
        }
        design_manifest = {
            "candidateDesignArtifactSha256": "def",
            "candidateDesignsFrozenAtUtc": "2026-07-03T00:01:00+00:00",
            "validationStartedAtUtc": "2026-07-03T00:02:00+00:00",
        }
        checks = validate_inverse_design_artifacts(targets, designs, validations, target_manifest, design_manifest)
        self.assertTrue(timestamp_order_ok(target_manifest["targetProfilesFrozenAtUtc"], design_manifest["candidateDesignsFrozenAtUtc"]))
        self.assertEqual(int(checks["success"].sum()), len(checks))


if __name__ == "__main__":
    unittest.main()
