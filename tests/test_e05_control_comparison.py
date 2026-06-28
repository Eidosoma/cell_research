import unittest

import pandas as pd

from morphospace2d import (
    CONTROL_COMPARISON_SCHEMA_VERSION,
    ControlSourceSpec,
    control_validation_rows,
    harmonize_control_runs,
    information_access_summary,
    leakage_audit_summary,
    target_control_gap_summary,
)


def _audit(policy_ids, source_step_id="S13"):
    return pd.DataFrame(
        [
            {
                "research_step_id": source_step_id,
                "policy_id": policy_id,
                "audit_success": True,
                "audit_errors_json": "[]",
                "audit_payload_hash": f"hash-{policy_id}",
            }
            for policy_id in policy_ids
        ]
    )


def _trajectory_rows(rows):
    return pd.DataFrame(
        [
            {
                "source_step_id": source,
                "run_id": run_id,
                "trajectory_id": f"traj::{source}::{run_id}",
                "path_curvature_proxy": 1.2,
                "temporary_worsening_count": 0,
                "monotonicity_error_proxy": 0.0,
                "convergence_basin": "near_target" if near_target else "off_target",
                "route_cluster_id": 1,
                "failure_cluster_id": 0,
                "exact_or_near_target": near_target,
                "initial_error_proxy": initial,
                "final_error_proxy": final,
                "relative_error_reduction_proxy": (initial - final) / initial if initial else 0.0,
            }
            for source, run_id, initial, final, near_target in rows
        ]
    )


class E05ControlComparisonTest(unittest.TestCase):
    def test_harmonize_local_and_global_rows(self):
        spec = ControlSourceSpec("S10", "/unused", source_kind="symmetry_breaking")
        run_df = pd.DataFrame(
            [
                {
                    "run_id": "s10-local",
                    "target_id": "ring",
                    "task_id": "sym-ring",
                    "motif": "ring",
                    "policy_id": "local_axis_alignment",
                    "policy_family": "local_only_no_gradient",
                    "seed": 1,
                    "node_count": 16,
                    "pattern_score": 0.75,
                    "success": False,
                    "is_local_only_policy": True,
                    "is_global_information_baseline": False,
                    "uses_global_gradient": False,
                    "uses_organizer": False,
                },
                {
                    "run_id": "s10-global",
                    "target_id": "ring",
                    "task_id": "sym-ring",
                    "motif": "ring",
                    "policy_id": "global_gradient_axis_baseline",
                    "policy_family": "global_gradient_baseline",
                    "seed": 2,
                    "node_count": 16,
                    "pattern_score": 1.0,
                    "success": True,
                    "is_local_only_policy": False,
                    "is_global_information_baseline": True,
                    "uses_global_gradient": True,
                    "uses_organizer": False,
                },
            ]
        )
        trajectories = _trajectory_rows(
            [
                ("S10", "s10-local", 0.8, 0.25, False),
                ("S10", "s10-global", 0.8, 0.0, True),
            ]
        )

        out = harmonize_control_runs(spec, run_df, _audit(["local_axis_alignment", "global_gradient_axis_baseline"]), trajectories)

        self.assertEqual(set(out["schema_version"]), {CONTROL_COMPARISON_SCHEMA_VERSION})
        self.assertTrue(out["s12_route_joined"].all())
        local = out[out["run_id"].eq("s10-local")].iloc[0]
        global_row = out[out["run_id"].eq("s10-global")].iloc[0]
        self.assertTrue(local["is_local_only_policy"])
        self.assertFalse(local["is_global_information_baseline"])
        self.assertEqual(local["control_class"], "local_only")
        self.assertEqual(local["target_error_metric"], "pattern_score_complement")
        self.assertAlmostEqual(local["initial_target_error"], 0.8)
        self.assertAlmostEqual(local["final_target_error"], 0.25)
        self.assertFalse(local["local_only_leakage_violation"])
        self.assertEqual(global_row["control_class"], "global_gradient_baseline")
        self.assertEqual(global_row["information_scope"], "explicit_global_gradient_baseline")
        self.assertTrue(global_row["global_baseline_explicitly_flagged"])
        self.assertAlmostEqual(global_row["information_access_score"], 4.0)

    def test_target_map_baseline_and_hidden_leakage(self):
        spec = ControlSourceSpec("S11", "/unused", source_kind="gpu_label_dynamics")
        run_df = pd.DataFrame(
            [
                {
                    "run_id": "s11-local-clean",
                    "target_id": "gradient",
                    "motif": "gradient",
                    "policy_id": "local_neighbor_majority_gpu",
                    "policy_family": "local_only_label_diffusion",
                    "seed": 3,
                    "node_count": 36,
                    "initial_hamming_error": 0.7,
                    "final_hamming_error": 0.5,
                    "exact_match": False,
                    "is_local_only_policy": True,
                    "is_global_information_baseline": False,
                    "uses_target_map": False,
                },
                {
                    "run_id": "s11-target-map",
                    "target_id": "gradient",
                    "motif": "gradient",
                    "policy_id": "explicit_target_relaxation_gpu",
                    "policy_family": "global_target_map_baseline",
                    "seed": 4,
                    "node_count": 36,
                    "initial_hamming_error": 0.7,
                    "final_hamming_error": 0.0,
                    "exact_match": True,
                    "is_local_only_policy": False,
                    "is_global_information_baseline": True,
                    "uses_target_map": True,
                },
                {
                    "run_id": "s11-local-leaky",
                    "target_id": "gradient",
                    "motif": "gradient",
                    "policy_id": "local_leaky_gpu",
                    "policy_family": "local_only_label_diffusion",
                    "seed": 5,
                    "node_count": 36,
                    "initial_hamming_error": 0.7,
                    "final_hamming_error": 0.1,
                    "exact_match": False,
                    "is_local_only_policy": True,
                    "is_global_information_baseline": False,
                    "uses_hidden_target_map_leakage": True,
                },
            ]
        )
        trajectories = _trajectory_rows(
            [
                ("S11", "s11-local-clean", 0.7, 0.5, False),
                ("S11", "s11-target-map", 0.7, 0.0, True),
                ("S11", "s11-local-leaky", 0.7, 0.1, False),
            ]
        )

        out = harmonize_control_runs(
            spec,
            run_df,
            _audit(["local_neighbor_majority_gpu", "explicit_target_relaxation_gpu", "local_leaky_gpu"]),
            trajectories,
        )
        target_map = out[out["run_id"].eq("s11-target-map")].iloc[0]
        leaky = out[out["run_id"].eq("s11-local-leaky")].iloc[0]

        self.assertEqual(target_map["control_class"], "target_map_baseline")
        self.assertEqual(target_map["information_scope"], "explicit_target_map_baseline")
        self.assertTrue(target_map["exact_success"])
        self.assertTrue(leaky["hidden_global_leakage_flag"])
        self.assertTrue(leaky["local_only_leakage_violation"])

    def test_summaries_and_validation_checks(self):
        sources = []
        audits = []
        trajectories = []
        for source_step_id, source_kind, run_rows in [
            (
                "S07",
                "scrambled_recovery",
                [
                    {
                        "run_id": "s07-local",
                        "target_id": "sorted_row",
                        "motif": "sorted_row",
                        "policy_id": "classic_scalar_rank_swap",
                        "policy_family": "classic_derived",
                        "seed": 10,
                        "node_count": 8,
                        "initial_composite_error": 0.8,
                        "final_composite_error": 0.2,
                        "best_composite_error": 0.2,
                        "success": False,
                    }
                ],
            ),
            (
                "S10",
                "symmetry_breaking",
                [
                    {
                        "run_id": "s10-local",
                        "target_id": "ring",
                        "motif": "ring",
                        "policy_id": "local_axis_alignment",
                        "policy_family": "local_only_no_gradient",
                        "seed": 11,
                        "node_count": 16,
                        "pattern_score": 0.8,
                        "success": False,
                        "is_local_only_policy": True,
                    },
                    {
                        "run_id": "s10-global",
                        "target_id": "ring",
                        "motif": "ring",
                        "policy_id": "organizer_axis_baseline",
                        "policy_family": "organizer_baseline",
                        "seed": 12,
                        "node_count": 16,
                        "pattern_score": 1.0,
                        "success": True,
                        "is_local_only_policy": False,
                        "is_global_information_baseline": True,
                        "uses_organizer": True,
                    },
                ],
            ),
            (
                "S11",
                "gpu_label_dynamics",
                [
                    {
                        "run_id": "s11-local",
                        "target_id": "gradient",
                        "motif": "gradient",
                        "policy_id": "local_neighbor_majority_gpu",
                        "policy_family": "local_only_label_diffusion",
                        "seed": 13,
                        "node_count": 36,
                        "initial_hamming_error": 0.7,
                        "final_hamming_error": 0.5,
                        "exact_match": False,
                    },
                    {
                        "run_id": "s11-target-map",
                        "target_id": "gradient",
                        "motif": "gradient",
                        "policy_id": "explicit_target_relaxation_gpu",
                        "policy_family": "global_target_map_baseline",
                        "seed": 14,
                        "node_count": 36,
                        "initial_hamming_error": 0.7,
                        "final_hamming_error": 0.0,
                        "exact_match": True,
                        "is_local_only_policy": False,
                        "is_global_information_baseline": True,
                        "uses_target_map": True,
                    },
                ],
            ),
        ]:
            spec = ControlSourceSpec(source_step_id, "/unused", source_kind=source_kind)
            run_df = pd.DataFrame(run_rows)
            policy_ids = run_df["policy_id"].tolist()
            audit_df = _audit(policy_ids, source_step_id=source_step_id)
            audits.append(audit_df.assign(source_step_id=source_step_id, source_kind=source_kind))
            trajectories.extend((source_step_id, row["run_id"], 0.8, 0.0 if row.get("success", False) or row.get("exact_match", False) else 0.3, bool(row.get("success", False) or row.get("exact_match", False))) for row in run_rows)
            sources.append((spec, run_df, audit_df))

        trajectory_df = _trajectory_rows(trajectories)
        control = pd.concat(
            [harmonize_control_runs(spec, run_df, audit_df, trajectory_df) for spec, run_df, audit_df in sources],
            ignore_index=True,
        )
        audit = pd.concat(audits, ignore_index=True)
        access = information_access_summary(control)
        gaps = target_control_gap_summary(control)
        leakage = leakage_audit_summary(control, audit)
        catalog = pd.DataFrame(
            [
                {
                    "source_step_id": source,
                    "source_kind": kind,
                    "run_available": True,
                    "leakage_audit_available": True,
                }
                for source, kind in [("S07", "scrambled_recovery"), ("S10", "symmetry_breaking"), ("S11", "gpu_label_dynamics")]
            ]
        )
        validation = control_validation_rows(
            source_catalog=catalog,
            control_df=control,
            access_summary_df=access,
            gap_summary_df=gaps,
            leakage_summary_df=leakage,
            expected_sources=("S07", "S10", "S11"),
        )

        self.assertFalse(access.empty)
        self.assertTrue(gaps["has_local_and_global"].any())
        self.assertTrue(leakage["policy_audit_record_present"].all())
        self.assertTrue(validation["success"].all(), validation.to_string())


if __name__ == "__main__":
    unittest.main()
