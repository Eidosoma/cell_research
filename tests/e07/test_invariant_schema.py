"""Tests for E07 S09 invariant-discovery helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.e07.invariant_schema import (
    extract_policy_invariants,
    extract_world_invariants,
    parse_json_field,
    residualize_by_group_median,
    validate_invariant_artifacts,
    weighted_correlation,
)


class InvariantSchemaTests(unittest.TestCase):
    def test_parse_json_field_handles_bad_input(self) -> None:
        self.assertEqual(parse_json_field('{"a": 1}')["a"], 1)
        self.assertEqual(parse_json_field("not-json", default=[]), [])

    def test_policy_invariants_detect_memory_signal_and_target_knowledge(self) -> None:
        features = extract_policy_invariants(
            {
                "representation_payload_json": (
                    '{"dslSource":"policy p v1\\nstate ideal_position=none\\nrule if random_lt(0.5) then signal(right), '
                    'remember(seen,true), set_ideal(left_boundary)\\nend","ruleCount":1,'
                    '"conditionVocabulary":["random_lt"]}'
                ),
                "action_vocabulary_json": '["signal","remember","set_ideal","wait"]',
                "state_variables_json": '["ideal_position","target_seen"]',
                "parameter_keys_json": "[]",
                "requires_memory": True,
                "requires_signaling": True,
                "uses_global_oracle": False,
                "information_access": "local_neighbors",
                "interpretable": True,
            }
        )
        self.assertEqual(features["policy_requires_memory"], 1.0)
        self.assertEqual(features["policy_requires_signaling"], 1.0)
        self.assertEqual(features["policy_target_position_knowledge"], 1.0)
        self.assertEqual(features["policy_stochastic_choice"], 1.0)
        self.assertEqual(features["policy_rule_count"], 1.0)

    def test_world_invariants_extract_dimensionality_and_perturbation(self) -> None:
        features = extract_world_invariants(
            {
                "world_family": "target_morphology",
                "substrate_kind": "square_grid_2d",
                "state_space": "2D grid with cells",
                "local_observations": "local neighbor observation",
                "action_set": "swap; wait; signal",
                "transition_rules": "repair damaged sites",
                "goal_predicate": "shape boundary match",
                "perturbation_model": "frozen_count=3; foreign patch damage",
                "measurement_functions": "boundary_error; topology_error",
                "scheduler": "cyclic_scan_seed_offset",
                "metadata_json": "{}",
            }
        )
        self.assertEqual(features["world_has_2d_or_graph_substrate"], 1.0)
        self.assertEqual(features["world_frozen_count"], 3.0)
        self.assertEqual(features["world_has_damage_or_repair"], 1.0)
        self.assertEqual(features["world_has_target_morphology"], 1.0)
        self.assertEqual(features["world_scheduler_cyclic"], 1.0)

    def test_residualization_and_weighted_correlation(self) -> None:
        frame = pd.DataFrame({"group": ["a", "a", "b", "b"], "target": [1.0, 3.0, 10.0, 14.0]})
        residual = residualize_by_group_median(frame, target_column="target", group_columns=["group"])
        self.assertEqual(list(residual), [-1.0, 1.0, -2.0, 2.0])
        corr = weighted_correlation(pd.Series([0.0, 1.0, 0.0, 1.0]), residual, pd.Series([1.0, 1.0, 1.0, 1.0]))
        self.assertGreater(corr, 0.9)

    def test_validation_accepts_minimal_valid_artifacts(self) -> None:
        candidates = pd.DataFrame(
            {
                "candidate_feature": ["policy_requires_memory"],
                "feature_group": ["policy_invariant"],
                "effect_full": [0.2],
                "effect_heldout_policy": [0.1],
                "effect_heldout_world": [0.1],
                "source_dominance_rate": [0.5],
                "candidate_status": ["stable_supported_candidate"],
            }
        )
        metrics = pd.DataFrame(
            {
                "split_name": ["heldout_policy", "heldout_world"],
                "model_name": ["source_metric_median", "source_missingness_control_linear_sgd"],
                "mae": [1.0, 1.1],
                "rmse": [1.2, 1.3],
                "r2": [0.1, 0.0],
            }
        )
        counterexamples = pd.DataFrame(
            {
                "candidate_feature": ["policy_requires_memory"],
                "counterexample_type": ["active_low_residual"],
                "residual_vs_source_metric": [-1.0],
            }
        )
        s08 = pd.DataFrame({"s08_caveat": ["source_dominance_failure", "goal_conflict_separation_failure"]})
        checks = validate_invariant_artifacts(candidates, metrics, counterexamples, s08)
        self.assertTrue(bool(checks["success"].all()))


if __name__ == "__main__":
    unittest.main()
