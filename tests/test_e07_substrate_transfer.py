from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from platonic_space.substrate_transfer import (
    SOURCE_TARGET_ID,
    TransferPolicyPanel,
    annotate_s08_transfer_distances,
    build_transfer_policy_panel,
    build_transfer_target_panel,
    distance_prediction_correlations,
    run_transfer_panel,
    summarize_transfer_runs,
    validation_checks,
)


class TestE07SubstrateTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        design_path = Path("/artifacts/research_steps/S12/designed_policies.parquet")
        nearest_path = Path("/artifacts/research_steps/S12/nearest_existing_policy_comparison.parquet")
        distance_path = Path("/artifacts/research_steps/S08/platonic_distances.parquet")
        if not design_path.exists() or not nearest_path.exists() or not distance_path.exists():
            raise unittest.SkipTest("S12 design artifacts and S08 distances are required for S13 substrate-transfer tests")
        cls.designed = pd.read_parquet(design_path)
        cls.nearest = pd.read_parquet(nearest_path)
        cls.s08_distances = pd.read_parquet(distance_path)

    def test_policy_mapping_covers_s12_designs_and_controls(self) -> None:
        panel = build_transfer_policy_panel(self.designed, self.nearest)
        mapping = panel.mapping_catalog

        self.assertEqual(
            set(mapping[mapping["mappingKind"].eq("s12_designed_policy_transfer")]["sourcePolicyId"].astype(str)),
            set(self.designed["policyId"].astype(str)),
        )
        self.assertIn("baseline_transfer_control", set(mapping["mappingKind"]))
        self.assertIn("random_transfer_control", set(mapping["mappingKind"]))
        self.assertIn("randomized_transfer_control", set(mapping["mappingKind"]))
        self.assertIn("no_transfer_control", set(mapping["mappingKind"]))
        self.assertGreaterEqual(int(mapping["mappingKind"].eq("unsupported_documented_mapping").sum()), 2)
        self.assertTrue(mapping["mappingCaveat"].astype(str).str.contains("Bubble", case=False).any())

    def test_transfer_targets_validate_and_cover_required_substrates(self) -> None:
        target_panel = build_transfer_target_panel()
        classes = set(target_panel.target_catalog["substrateClass"].astype(str))

        self.assertIn("row_1d", classes)
        self.assertIn("square_grid_2d", classes)
        self.assertIn("irregular_graph", classes)
        for target in target_panel.targets:
            with self.subTest(target=target.target_id):
                self.assertEqual(target.validate(), [])

    def test_mini_transfer_run_summarizes_conservative_replay(self) -> None:
        mini_designs = self.designed.head(1).copy()
        panel = build_transfer_policy_panel(mini_designs, self.nearest, include_randomized_controls=True)
        randomized_id = next(policy.policy_id for policy in panel.policies if policy.policy_id.endswith("__s13_randomized_transfer"))
        selected_ids = {
            panel.policies[0].policy_id,
            randomized_id,
            "classic_scalar_rank_swap",
            "random_local_swap_null",
            "s13_wait_no_transfer_control",
        }
        selected_policies = tuple(policy for policy in panel.policies if policy.policy_id in selected_ids)
        selected_mapping = panel.mapping_catalog[
            panel.mapping_catalog["transferPolicyId"].isin(selected_ids)
            | panel.mapping_catalog["mappingKind"].eq("unsupported_documented_mapping")
        ].copy()
        mini_panel = TransferPolicyPanel(policies=selected_policies, mapping_catalog=selected_mapping)
        target_panel = build_transfer_target_panel()
        selected_targets = tuple(target for target in target_panel.targets if target.target_id in {SOURCE_TARGET_ID, "gradient_x_5x4", "branch_graph_12"})
        selected_target_catalog = target_panel.target_catalog[target_panel.target_catalog["targetId"].isin([target.target_id for target in selected_targets])].copy()
        target_panel = type(target_panel)(targets=selected_targets, target_catalog=selected_target_catalog)

        runs, _traces, _severity = run_transfer_panel(
            mini_panel.policies,
            target_panel.targets,
            seeds=(91001,),
            max_steps_by_target={target.target_id: 120 for target in target_panel.targets},
        )
        summary = summarize_transfer_runs(runs, mini_panel.mapping_catalog, target_panel.target_catalog)
        distance_summary = annotate_s08_transfer_distances(summary, self.s08_distances)
        correlations = distance_prediction_correlations(distance_summary)
        checks = validation_checks(
            designed_policies=mini_designs,
            policy_panel=mini_panel,
            target_panel=target_panel,
            run_rows=runs,
            transfer_summary=distance_summary,
            distance_correlations=correlations,
            seeds=(91001,),
            upstream_statuses=[{"success": True}],
        )
        hard_failures = checks[checks["severity"].eq("error") & ~checks["success"]]

        self.assertEqual(len(runs), len(mini_panel.policies) * len(target_panel.targets))
        self.assertTrue(runs["occupancy_preserved"].astype(bool).all())
        self.assertTrue(runs["cell_ids_preserved"].astype(bool).all())
        self.assertFalse(runs["uses_whole_target_leakage"].astype(bool).any())
        self.assertFalse(summary.empty)
        self.assertIn("failureMode", summary.columns)
        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))

    def test_s08_distance_annotation_marks_missing_graph_distance(self) -> None:
        mini_designs = self.designed.head(1).copy()
        panel = build_transfer_policy_panel(mini_designs, self.nearest, include_randomized_controls=False)
        selected_policies = (panel.policies[0],)
        selected_mapping = panel.mapping_catalog[panel.mapping_catalog["transferPolicyId"].eq(selected_policies[0].policy_id)].copy()
        mini_panel = TransferPolicyPanel(policies=selected_policies, mapping_catalog=selected_mapping)
        target_panel = build_transfer_target_panel()
        selected_targets = tuple(target for target in target_panel.targets if target.target_id in {SOURCE_TARGET_ID, "branch_graph_12"})
        selected_target_catalog = target_panel.target_catalog[target_panel.target_catalog["targetId"].isin([target.target_id for target in selected_targets])].copy()
        target_panel = type(target_panel)(targets=selected_targets, target_catalog=selected_target_catalog)
        runs, _traces, _severity = run_transfer_panel(
            mini_panel.policies,
            target_panel.targets,
            seeds=(91002,),
            max_steps_by_target={target.target_id: 80 for target in target_panel.targets},
        )
        summary = summarize_transfer_runs(runs, mini_panel.mapping_catalog, target_panel.target_catalog)
        annotated = annotate_s08_transfer_distances(summary, self.s08_distances)

        source_row = annotated[annotated["target_id"].eq(SOURCE_TARGET_ID)].iloc[0]
        graph_row = annotated[annotated["target_id"].eq("branch_graph_12")].iloc[0]
        self.assertAlmostEqual(float(source_row["s08WorldDistance"]), 0.0)
        self.assertAlmostEqual(float(source_row["s08GoalDistance"]), 0.0)
        self.assertIn("world", str(graph_row["s08DistanceMissingComponents"]))
        self.assertIn("goal", str(graph_row["s08DistanceMissingComponents"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
