"""S10 behavior embedding tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.e03.behavior_embedding import (
    aggregate_phase_diagnostics,
    behavior_feature_columns,
    build_policy_feature_frame,
    compute_embedding,
    embedding_digest,
    embedding_stability_frame,
    landmark_frame,
    normalize_behavior_features,
    validation_frame,
    attach_embedding_columns,
)


def competence_row(policy_id: str, name: str, source_kind: str, score: float, *, classic: bool = False) -> dict[str, object]:
    return {
        "schema": "unit",
        "experiment_id": "E03",
        "research_step_id": "S07",
        "policy_id": policy_id,
        "policy_name": name,
        "source_kind": source_kind,
        "route": "jax_batch",
        "requires_cpu_fallback": False,
        "screen_run_count": 8,
        "heldout_run_count": 3,
        "scale_run_count": 1,
        "invalid_run_count": 0,
        "timeout_run_count": int(score < 0.4),
        "screen_train_final_sortedness_mean": score + 0.03,
        "screen_heldout_final_sortedness_mean": score,
        "screen_heldout_sorted_run_fraction": float(score > 0.9),
        "screen_heldout_improvement_mean": score - 0.4,
        "screen_heldout_work_mean": 10.0 + score,
        "screen_score": score,
        "n100_final_sortedness_mean": score - 0.02,
        "n1000_final_sortedness_mean": np.nan,
        "best_final_sortedness": score + 0.08,
        "classic_dsl_seed": classic,
    }


def phase_row(policy_id: str, axis: str, final: float, *, failure: str = "timeout_partial_progress") -> dict[str, object]:
    return {
        "base_policy_id": policy_id,
        "base_policy_name": policy_id,
        "axis_name": axis,
        "axis_numeric": 16.0,
        "event_cap": 32,
        "array_size": 8,
        "final_inversion_sortedness": final,
        "inversion_sortedness_delta": final - 0.5,
        "max_inversion_sortedness": final,
        "min_inversion_sortedness": 0.5,
        "dg_drop": 0.0,
        "dg_present": False,
        "competent_at_threshold": final >= 0.9,
        "final_is_sorted": final >= 0.99,
        "timed_out": final < 0.99,
        "failure_mode": failure,
        "cycle_detected": False,
        "oscillation_score": 0.0,
        "no_change_event_fraction": 0.2,
        "distinct_state_fraction": 0.7,
        "work_count": 12,
    }


class BehaviorEmbeddingTests(unittest.TestCase):
    def test_phase_diagnostics_are_policy_level(self) -> None:
        sweeps = pd.DataFrame(
            [
                phase_row("p1", "event_cap", 0.5),
                phase_row("p1", "event_cap", 0.9, failure="sorted"),
                phase_row("p1", "array_size", 0.7),
            ]
        )
        boundaries = pd.DataFrame(
            [
                {
                    "base_policy_id": "p1",
                    "transition_strength": 0.4,
                    "replication_available": True,
                    "replication_direction_match": True,
                    "lower_success_fraction": 0.0,
                    "upper_success_fraction": 1.0,
                    "lower_mean_final_sortedness": 0.5,
                    "upper_mean_final_sortedness": 0.9,
                    "boundary_kind": "competence_transition",
                }
            ]
        )
        agg = aggregate_phase_diagnostics(sweeps, boundaries)
        self.assertEqual(len(agg), 1)
        self.assertEqual(int(agg.iloc[0]["s09_run_count"]), 3)
        self.assertIn("s09_axis_event_cap_mean_final_sortedness", agg.columns)
        self.assertEqual(int(agg.iloc[0]["s09_boundary_count"]), 1)

    def test_build_policy_feature_frame_keeps_classic_landmarks(self) -> None:
        s07 = pd.DataFrame(
            [
                competence_row("p_bubble", "classic_bubble_shadow", "classic_dsl_seed", 0.7, classic=True),
                competence_row("p_insert", "classic_insertion_active", "classic_dsl_seed", 0.8, classic=True),
                competence_row("p_select", "classic_selection_target", "classic_dsl_seed", 0.9, classic=True),
                competence_row("p_random", "random_policy", "random_expansion", 0.3),
            ]
        )
        s08 = pd.DataFrame([competence_row("p_qd", "qd_policy", "qd_mutation", 0.6)])
        s08["qd_candidate"] = True
        archive = pd.DataFrame([{"policy_id": "p_qd"}])
        sweeps = pd.DataFrame([phase_row("p_qd", "event_cap", 0.85)])
        frame = build_policy_feature_frame(s07, s08, archive, sweeps)
        self.assertEqual(len(frame), 5)
        self.assertTrue(frame.loc[frame["policy_id"] == "p_qd", "s08_archive_winner"].iloc[0])
        self.assertEqual(set(frame.loc[frame["classic_landmark"], "classic_family"]), {"Bubble", "Insertion", "Selection"})
        self.assertTrue(frame.loc[frame["policy_id"] == "p_qd", "s09_diagnostic_available"].iloc[0])

    def test_embedding_and_stability_are_deterministic(self) -> None:
        s07 = pd.DataFrame(
            [
                competence_row("p_bubble", "classic_bubble_shadow", "classic_dsl_seed", 0.7, classic=True),
                competence_row("p_insert", "classic_insertion_active", "classic_dsl_seed", 0.8, classic=True),
                competence_row("p_select", "classic_selection_target", "classic_dsl_seed", 0.9, classic=True),
                competence_row("p_random1", "random_policy_1", "random_expansion", 0.3),
                competence_row("p_random2", "random_policy_2", "random_expansion", 0.4),
                competence_row("p_random3", "random_policy_3", "random_expansion", 0.5),
            ]
        )
        s08 = pd.DataFrame(
            [
                competence_row("p_qd1", "qd_policy_1", "qd_mutation", 0.62),
                competence_row("p_qd2", "qd_policy_2", "qd_recombination", 0.77),
            ]
        )
        s08["qd_candidate"] = True
        archive = pd.DataFrame([{"policy_id": "p_qd1"}])
        sweeps = pd.DataFrame(
            [
                phase_row("p_bubble", "event_cap", 0.72),
                phase_row("p_qd1", "event_cap", 0.92, failure="sorted"),
                phase_row("p_qd2", "array_size", 0.82),
            ]
        )
        frame = build_policy_feature_frame(s07, s08, archive, sweeps)
        features = behavior_feature_columns(frame)
        normalized = normalize_behavior_features(frame, features)
        self.assertGreaterEqual(normalized.matrix.shape[1], 2)
        result = compute_embedding(frame, features, seed=123)
        embedded = attach_embedding_columns(frame, result)
        stability = embedding_stability_frame(frame, features, result.embedding, seeds=(123, 124), distance_sample_size=8)
        landmarks = landmark_frame(embedded)
        validation = validation_frame(
            policy_frame=frame,
            embedding_frame=embedded,
            feature_columns=features,
            stability=stability,
            landmarks=landmarks,
            figure_exists=True,
            unit_success=True,
        )
        self.assertTrue(np.isfinite(result.embedding).all())
        self.assertEqual(embedding_digest(embedded), embedding_digest(embedded.copy()))
        self.assertFalse(stability.empty)
        self.assertIn("classic_landmarks_highlighted", set(validation["validation_case"]))


if __name__ == "__main__":
    unittest.main()
