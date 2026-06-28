from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from morphospace2d import (
    DG_SCHEMA_VERSION,
    attach_control_metadata,
    detect_dg_events,
    dg_validation_rows,
    empirical_local_move_null_matches,
    hand_constructed_validation_examples,
    normalization_artifact_audit,
    null_comparison_summary,
    source_policy_dg_summary,
    synthetic_start_end_null_summary,
    trajectory_dg_summary,
)


def _state_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        (
            "S08::gradient::local_repair::seed1",
            "local_repair",
            "classic_derived",
            [0.8, 0.4, 0.5, 0.2],
            [1.6, 0.8, 1.0, 0.4],
        ),
        (
            "S08::gradient::proxy_artifact::seed2",
            "proxy_artifact",
            "classic_derived",
            [0.8, 0.4, 0.5, 0.2],
            [1.6, 1.2, 0.8, 0.4],
        ),
        (
            "S08::gradient::random_local_swap_null::seed3",
            "random_local_swap_null",
            "random_null",
            [0.8, 0.7, 0.75, 0.6],
            [1.6, 1.4, 1.5, 1.2],
        ),
    ]
    for run_id, policy_id, policy_family, errors, energies in specs:
        for snapshot_index, (error, energy) in enumerate(zip(errors, energies, strict=True)):
            rows.append(
                {
                    "schema_version": "test",
                    "research_step_id": "S12",
                    "source_step_id": "S08",
                    "source_trace_kind": "regeneration",
                    "source_trace_path": "/tmp/trace.parquet",
                    "source_run_path": "/tmp/run.parquet",
                    "trajectory_id": run_id,
                    "run_id": run_id,
                    "target_id": "gradient_x_5x4",
                    "task_id": None,
                    "motif": "gradient",
                    "policy_id": policy_id,
                    "policy_family": policy_family,
                    "seed": snapshot_index + 1,
                    "step": float(snapshot_index),
                    "snapshot_index": snapshot_index,
                    "snapshot_reason": "interval",
                    "state_hash": f"hash::{run_id}::{snapshot_index}",
                    "error_proxy": error,
                    "score_proxy": 1.0 - error,
                    "progress_fraction": (errors[0] - error) / errors[0],
                    "target_energy_feature": energy,
                    "neighborhood_error_feature": np.nan,
                    "is_local_only_policy": True,
                    "is_global_information_baseline": False,
                }
            )
    return pd.DataFrame(rows)


class E05DelayedGratificationTest(unittest.TestCase):
    def test_detect_dg_event_details(self) -> None:
        events = detect_dg_events([1.0, 0.7, 0.8, 0.4], steps=[0, 1, 2, 3])

        self.assertEqual(len(events), 1)
        event = events.iloc[0]
        self.assertAlmostEqual(float(event["worsening_magnitude"]), 0.1)
        self.assertAlmostEqual(float(event["recovery_magnitude"]), 0.4)
        self.assertTrue(bool(event["productive_event"]))
        self.assertAlmostEqual(float(event["productive_dg_index"]), 3.0)

    def test_hand_constructed_validation_examples(self) -> None:
        validation = hand_constructed_validation_examples()

        self.assertEqual(set(validation["schema_version"]), {DG_SCHEMA_VERSION})
        self.assertTrue(validation["validation_success"].all(), validation.to_string(index=False))

    def test_summary_nulls_and_normalization_audit(self) -> None:
        state = _state_rows()
        control = pd.DataFrame(
            [
                {
                    "source_step_id": "S08",
                    "run_id": run_id,
                    "control_class": "local_only",
                    "information_scope": "local_target_map_free",
                    "is_local_only_policy": True,
                    "is_global_information_baseline": False,
                }
                for run_id in state["run_id"].unique()
            ]
        )
        state = attach_control_metadata(state, control)
        summary, events = trajectory_dg_summary(state)
        policy_summary = source_policy_dg_summary(summary)
        synthetic = synthetic_start_end_null_summary(summary, replicates=4, random_state=14)
        empirical = empirical_local_move_null_matches(summary)
        audit = normalization_artifact_audit(state, summary)
        comparison = null_comparison_summary(summary, synthetic, empirical)
        validation = dg_validation_rows(
            state_df=state,
            summary_df=summary,
            event_df=events,
            hand_validation_df=hand_constructed_validation_examples(),
            synthetic_null_df=synthetic,
            empirical_match_df=empirical,
            normalization_audit_df=audit,
            expected_sources=("S08",),
        )

        self.assertEqual(len(summary), 3)
        self.assertFalse(events.empty)
        self.assertTrue(events[["trajectory_id", "run_id", "source_step_id"]].notna().all().all())
        self.assertFalse(policy_summary.empty)
        self.assertEqual(len(synthetic), len(summary) * 4)
        self.assertLessEqual(float(synthetic["start_end_match_abs_error"].max()), 1e-12)
        self.assertGreater(int(empirical["matched"].sum()), 0)
        artifact_row = audit[audit["policy_id"].eq("proxy_artifact")].iloc[0]
        supported_row = audit[audit["policy_id"].eq("local_repair")].iloc[0]
        self.assertTrue(bool(artifact_row["normalization_artifact_flag"]))
        self.assertFalse(bool(supported_row["normalization_artifact_flag"]))
        self.assertFalse(comparison.empty)
        self.assertTrue(validation["success"].all(), validation.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
