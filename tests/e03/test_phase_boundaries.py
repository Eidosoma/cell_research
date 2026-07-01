"""S09 phase-boundary tests."""

from __future__ import annotations

import unittest

import pandas as pd

from src.e03.phase_boundaries import (
    PhaseConfig,
    PhasePolicy,
    detect_boundaries,
    initial_values,
    policy_has_probability,
    run_phase_config,
    set_policy_probability,
)
from src.e03.rule_dsl import parse_policy


STOCHASTIC_POLICY = """policy stochastic_swap v1
state ideal_position=none
rule if random_lt(0.25) and target_exists(right) and target_movable(right) and self_gt(right) then compare(right), choose(0.75, swap(right), wait)
rule else wait
end
"""


def phase_policy(source: str) -> PhasePolicy:
    policy = parse_policy(source)
    return PhasePolicy(
        base_policy_id=policy.policy_id,
        policy_name=policy.name,
        source_kind="unit",
        dsl_source=source,
        route="cpu_fallback",
        quality_score=0.0,
        archive_winner=False,
        nonclassic_cell=True,
        selection_reason="unit",
    )


class PhaseBoundaryTests(unittest.TestCase):
    def test_probability_variant_rewrites_random_guards_and_choose_actions(self) -> None:
        policy = parse_policy(STOCHASTIC_POLICY)
        self.assertTrue(policy_has_probability(policy))
        variant = set_policy_probability(policy, 0.9)
        rendered = variant.to_source()
        self.assertIn("random_lt(0.9)", rendered)
        self.assertIn("choose(0.9, swap(right), wait)", rendered)
        self.assertNotEqual(policy.policy_id, variant.policy_id)

    def test_initial_value_profiles_are_deterministic(self) -> None:
        self.assertEqual(initial_values(5, 1, "reverse"), (5, 4, 3, 2, 1))
        self.assertEqual(initial_values(8, 11, "nearly_sorted_1swap"), initial_values(8, 11, "nearly_sorted_1swap"))
        self.assertNotEqual(initial_values(8, 11, "random_permutation"), initial_values(8, 12, "random_permutation"))

    def test_phase_config_run_records_failure_mode_and_trace_metrics(self) -> None:
        policy = phase_policy(STOCHASTIC_POLICY)
        config = PhaseConfig(
            config_id="unit_prob_1",
            axis_name="probability",
            axis_value="1",
            axis_numeric=1.0,
            array_size=6,
            event_cap=24,
            seed=101,
            split="primary",
            probability_override=1.0,
        )
        row = run_phase_config(policy, config)
        self.assertEqual(row["axis_name"], "probability")
        self.assertEqual(row["execution_status"], "ok")
        self.assertIn(row["failure_mode"], {"sorted", "timeout_stalled", "timeout_partial_progress", "timeout_degraded", "timeout_no_clear_progress", "timeout_oscillation_candidate"})
        self.assertIn("oscillation_score", row)
        self.assertGreaterEqual(row["events_executed"], 0)

    def test_boundary_detection_finds_competence_crossing(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "base_policy_id": "p1",
                    "base_policy_name": "policy",
                    "axis_name": "event_cap",
                    "axis_value": "16",
                    "axis_numeric": 16.0,
                    "run_count": 3,
                    "mean_final_sortedness": 0.5,
                    "success_fraction": 0.0,
                    "dg_fraction": 0.0,
                    "oscillation_fraction": 0.0,
                },
                {
                    "base_policy_id": "p1",
                    "base_policy_name": "policy",
                    "axis_name": "event_cap",
                    "axis_value": "32",
                    "axis_numeric": 32.0,
                    "run_count": 3,
                    "mean_final_sortedness": 0.95,
                    "success_fraction": 1.0,
                    "dg_fraction": 0.0,
                    "oscillation_fraction": 0.0,
                },
            ]
        )
        boundaries = detect_boundaries(summary)
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries.iloc[0]["boundary_kind"], "competence_transition")


if __name__ == "__main__":
    unittest.main()
