"""E04 S12 tissue-field predictor tests."""

from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.e04.tissue_fields import (
    FEATURE_SETS,
    PREDICTION_TARGETS,
    build_tissue_field_feature_rows,
    diffuse_field,
    evaluate_field_predictors,
    infer_field_predictor_outcome,
    predictor_feature_audit,
    split_audit,
)


class TissueFieldTests(unittest.TestCase):
    def test_diffuse_field_matches_s02_scalar_update(self) -> None:
        field = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])
        updated = diffuse_field(field, rate=0.25, decay=0.0)

        self.assertEqual(updated.tolist(), [0.0, 0.25, 0.5, 0.25, 0.0])

    def test_predictor_feature_audit_excludes_forbidden_signal_fields(self) -> None:
        audit = predictor_feature_audit()

        self.assertTrue(audit["success"])
        serialized_features = json.dumps(audit["featureColumns"])
        self.assertNotIn("target_seeking", serialized_features)
        self.assertNotIn("morphogen", serialized_features)
        self.assertNotIn("sorted", serialized_features)

    def test_build_features_from_s11_traces_and_split_groups(self) -> None:
        outcomes = pd.DataFrame(
            [
                {
                    "run_id": "train_run",
                    "activation_seed": 41001,
                    "schedule_seed": 61001,
                    "schedule_sha256": "sched_train",
                    "competency_axis": "heldout_perturbations",
                    "scenario_name": "s11_unit_train",
                    "policy_mode": "diffusive_signaling",
                    "policy_mode_order": 4,
                    "s08_source_policy_id": "policy_a",
                    "array_size": 4,
                    "damage_level": 0,
                    "max_events": 5,
                    "uses_global_oracle": False,
                },
                {
                    "run_id": "test_run",
                    "activation_seed": 41002,
                    "schedule_seed": 61002,
                    "schedule_sha256": "sched_test",
                    "competency_axis": "heldout_perturbations",
                    "scenario_name": "s11_unit_test",
                    "policy_mode": "diffusive_signaling",
                    "policy_mode_order": 4,
                    "s08_source_policy_id": "policy_a",
                    "array_size": 4,
                    "damage_level": 0,
                    "max_events": 5,
                    "uses_global_oracle": False,
                },
            ]
        )
        traces = []
        for run_id in ("train_run", "test_run"):
            for event_step, values, action, blocked in [
                (0, [1, 2, 3, 4], "initial_state", 0),
                (1, [1, 3, 2, 4], "blocked_swap_attempt", 1),
                (2, [1, 2, 3, 4], "swap", 0),
            ]:
                traces.append(
                    {
                        "run_id": run_id,
                        "event_step": event_step,
                        "activated_position": None if event_step == 0 else 1,
                        "current_values_json": json.dumps(values),
                        "current_frozen_positions_json": "[]",
                        "frozen_attempt_delta": blocked,
                        "applied_action": action,
                        "swap_delta": 1 if action == "swap" else 0,
                        "fatigue_transition_delta": 0,
                        "perturbation_count_at_event": 0,
                        "target_met": values == [1, 2, 3, 4],
                        "feature_audit_uses_oracle": False,
                        "feature_protocol_id": "s07_local_only_adjacent_update_v1",
                        "sortedness_percent": 100.0 if values == [1, 2, 3, 4] else 75.0,
                    }
                )
        feature_df = build_tissue_field_feature_rows(outcomes, pd.DataFrame(traces), decay=0.0)
        audit = split_audit(feature_df)

        self.assertEqual(set(feature_df["run_id"]), {"train_run", "test_run"})
        self.assertGreater(feature_df["blocked_field_mean"].max(), 0.0)
        self.assertGreater(feature_df["frustrated_field_mean"].max(), 0.0)
        self.assertTrue(audit["success"])
        self.assertEqual(audit["overlapSeeds"], [])
        self.assertEqual(audit["overlapPerturbationGroups"], [])

    def test_evaluate_predictors_returns_complete_matrix(self) -> None:
        rows = []
        for split, seed_offset in (("train", 1), ("test", 2)):
            for index in range(30):
                signal = float(index % 6) / 5.0
                label = float(signal > 0.45)
                row = {
                    "split": split,
                    "event_fraction": index / 30.0,
                    "array_size_norm": 0.5,
                    "damage_level_norm": 0.0,
                    "current_frozen_fraction": 0.0,
                    "perturbation_event_flag": float(index % 7 == 0),
                    "policy_mode_order_norm": 0.5,
                    "order_pair_fraction": 0.4 + 0.1 * (index % 2),
                    "order_error_fraction": 0.6 - 0.1 * (index % 2),
                    "local_actor_order_fraction": 0.5,
                    "current_frozen_neighbor_fraction": 0.0,
                    "blocked_field_mean": signal,
                    "blocked_field_max": signal,
                    "blocked_field_std": signal / 4,
                    "blocked_field_local": signal,
                    "blocked_field_gradient_mean": signal / 5,
                    "frustrated_field_mean": signal,
                    "frustrated_field_max": signal,
                    "frustrated_field_std": signal / 4,
                    "frustrated_field_local": signal,
                    "frustrated_field_gradient_mean": signal / 5,
                    "activation_seed": 41000 + seed_offset,
                    "split_perturbation_group": f"sched_{split}",
                }
                for target in PREDICTION_TARGETS:
                    row[target] = label
                rows.append(row)
        feature_df = pd.DataFrame(rows)
        results_df, importance_df = evaluate_field_predictors(feature_df)
        outcome = infer_field_predictor_outcome(results_df)

        self.assertEqual(len(results_df), len(FEATURE_SETS) * len(PREDICTION_TARGETS))
        self.assertTrue((results_df["model_status"] == "ok").all())
        self.assertFalse(importance_df.empty)
        self.assertIn(outcome["outcome"], {"supportive", "null", "constraining/contradictory"})


if __name__ == "__main__":
    unittest.main()
