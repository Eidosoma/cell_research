"""E05 S13 local-versus-global control tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.local_global_control import (  # noqa: E402
    add_failure_and_scaling_fields,
    baseline_specs,
    compute_gap_summary,
    evaluate_baseline_on_state,
    exact_identity_assignment_plan,
)
from src.e05.regeneration import apply_regeneration_perturbation  # noqa: E402
from src.e05.scrambled_embryo import scramble_target_state  # noqa: E402
from src.e05.targets import gradient_target  # noqa: E402


class LocalGlobalControlTests(unittest.TestCase):
    def test_baseline_specs_label_added_global_information(self) -> None:
        specs = baseline_specs()
        self.assertEqual({spec.control_class for spec in specs}, {"top_down", "organizer", "global_controller"})
        self.assertTrue(all(spec.global_state_access for spec in specs))
        self.assertTrue(all(spec.global_target_access for spec in specs))
        self.assertTrue(any(spec.organizer_cue_added for spec in specs))
        self.assertTrue(all(not spec.local_policy_information_access_changed for spec in specs))

    def test_exact_assignment_solves_scrambled_state_without_identity_edits(self) -> None:
        target = gradient_target(4, 3, target_id="s13_test_gradient")
        state = scramble_target_state(target, 17)

        plan = exact_identity_assignment_plan(target, state, allow_unfreeze=False)

        self.assertTrue(plan.feasible)
        self.assertIsNotNone(plan.final_state)
        self.assertEqual(float(target.target_error(plan.final_state)), 0.0)
        self.assertGreaterEqual(plan.nonlocal_relocation_count, 1)

    def test_assignment_blocks_missing_patch_but_global_rebuild_solves_it(self) -> None:
        target = gradient_target(4, 3, target_id="s13_missing_gradient")
        perturbed = apply_regeneration_perturbation(target, "missing_patch", 21, donor_targets=(target,))
        top_down, _organizer, global_rebuild = baseline_specs()

        blocked = evaluate_baseline_on_state(
            spec=top_down,
            target=target,
            initial_state=perturbed.state,
            metadata={
                "source_research_step_id": "S08",
                "benchmark_task": "regeneration",
                "task_id": "toy",
                "simulation_seed": 21,
                "perturbation_type": "missing_patch",
            },
        )
        rebuilt = evaluate_baseline_on_state(
            spec=global_rebuild,
            target=target,
            initial_state=perturbed.state,
            metadata={
                "source_research_step_id": "S08",
                "benchmark_task": "regeneration",
                "task_id": "toy",
                "simulation_seed": 21,
                "perturbation_type": "missing_patch",
            },
        )

        self.assertTrue(blocked["baseline_blocked"])
        self.assertGreater(blocked["final_target_error"], 0.0)
        self.assertFalse(rebuilt["baseline_blocked"])
        self.assertEqual(rebuilt["final_target_error"], 0.0)
        self.assertGreater(rebuilt["birth_count"], 0)

    def test_gap_summary_quantifies_global_minus_local_recovery_and_energy(self) -> None:
        rows = [
            {
                "benchmark_task": "scrambled_embryo",
                "target_kind": "gradient",
                "policy_id": "s07_local_target_neighbor_descent",
                "control_class": "local",
                "scale_label": "",
                "task_id": "t",
                "initial_target_error": 0.5,
                "final_target_error": 0.2,
                "target_recovery_fraction": 0.6,
                "final_aggregate_morphospace_error": 0.2,
                "aggregate_recovery_fraction": 0.6,
                "energy_per_site": 0.3,
                "low_recovery_failure": False,
                "failure_to_improve": False,
                "baseline_blocked": False,
            },
            {
                "benchmark_task": "scrambled_embryo",
                "target_kind": "gradient",
                "policy_id": "s13_global_rebuild_controller",
                "control_class": "global_controller",
                "scale_label": "",
                "task_id": "t",
                "initial_target_error": 0.5,
                "final_target_error": 0.0,
                "target_recovery_fraction": 1.0,
                "final_aggregate_morphospace_error": 0.0,
                "aggregate_recovery_fraction": 1.0,
                "energy_per_site": 1.2,
                "low_recovery_failure": False,
                "failure_to_improve": False,
                "baseline_blocked": False,
            },
        ]
        frame = add_failure_and_scaling_fields(pd.DataFrame(rows))
        gap = compute_gap_summary(frame)
        rebuilt = gap[gap["policy_id"].eq("s13_global_rebuild_controller")].iloc[0]

        self.assertAlmostEqual(float(rebuilt["target_recovery_gap_vs_local_target"]), 0.4)
        self.assertAlmostEqual(float(rebuilt["final_target_error_gap_vs_local_target"]), -0.2)
        self.assertAlmostEqual(float(rebuilt["energy_per_site_gap_vs_local_target"]), 0.9)


if __name__ == "__main__":
    unittest.main()
