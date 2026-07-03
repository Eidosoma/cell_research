"""E06 S09 governance-mechanism intervention tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e06.algotype_library import builtin_control_records
from src.e06.governance_mechanisms import (
    DEFAULT_GOVERNANCE_MECHANISMS,
    DEFAULT_INTERFACE_RULES,
    S09Config,
    compare_to_no_governance_baseline,
    governance_decision,
    governance_rankings,
    information_access_strata,
    run_s09_sweep,
    select_s09_conflict_cases,
    summarize_s09_runs,
    validate_s09_outputs,
)


def _json(values: list) -> str:
    return json.dumps(values, separators=(",", ":"))


def _sample_s04(records: list[dict]) -> pd.DataFrame:
    left = records[2]
    right = records[3]
    rows = []
    specs = [
        ("opposite_direction", "opposite_goal", 0.55, 0.82, 0.26, "polarized_goal_conflict"),
        ("partially_compatible_halves", "partially_compatible", 0.42, 0.88, 0.44, "mixed_goal_compromise"),
        ("unrelated_parity_center", "unrelated_goal", 0.33, 0.70, 0.50, "mixed_goal_tension"),
    ]
    for condition_idx, (profile, compatibility, gap, assigned, reference, state_class) in enumerate(specs):
        for seed in (101, 102):
            rows.append(
                {
                    "condition_id": f"s04_unit_{condition_idx}",
                    "panel": "unit",
                    "candidate_reason": "unit_conflict",
                    "arrangement": "alternating" if condition_idx == 0 else "random_permutation",
                    "arrangement_role": "unit",
                    "goal_profile_id": profile,
                    "goal_compatibility_class": compatibility,
                    "policy_ids_json": _json([left["algotypeId"], right["algotypeId"]]),
                    "display_names_json": _json([left["displayName"], right["displayName"]]),
                    "source_categories_json": _json([left["sourceCategory"], right["sourceCategory"]]),
                    "ratio_targets_json": _json([0.5, 0.5]),
                    "assigned_policy_mean_sortedness": assigned,
                    "reference_increasing_score": reference,
                    "goal_alignment_gap": gap,
                    "aggregation_delta_percent": 5.0 + condition_idx,
                    "goal_state_class": state_class,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def _sample_s08_strata(config: S09Config) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "interface_rule_id": rule.rule_id,
                "interface_mechanism_family": rule.mechanism_family,
                "explicit_recognition_used": rule.explicit_recognition_used,
            }
            for rule in config.interface_rules
        ]
    )


class GovernanceMechanismTests(unittest.TestCase):
    def test_governance_metadata_documents_radius_and_access(self) -> None:
        self.assertTrue(all(item.influence_radius_label for item in DEFAULT_GOVERNANCE_MECHANISMS))
        self.assertTrue(all(item.information_access_class for item in DEFAULT_GOVERNANCE_MECHANISMS))
        global_items = [item for item in DEFAULT_GOVERNANCE_MECHANISMS if item.global_controller_like]
        self.assertEqual([item.mechanism_id for item in global_items], ["global_reference_controller"])
        local_items = [item for item in DEFAULT_GOVERNANCE_MECHANISMS if not item.global_controller_like]
        self.assertTrue(all("global" not in item.information_access_class for item in local_items))

    def test_select_s09_conflict_cases_prioritizes_conflict_classes(self) -> None:
        records = builtin_control_records()
        selected = select_s09_conflict_cases(_sample_s04(records), S09Config(max_base_conditions=2))
        self.assertEqual(len(selected), 2)
        self.assertTrue(set(selected["goal_compatibility_class"]).issubset({"opposite_goal", "partially_compatible", "unrelated_goal"}))
        self.assertTrue(selected["selection_reason"].astype(str).str.contains("s04_conflict_case").all())

    def test_global_controller_decision_is_labeled_comparator(self) -> None:
        mechanism = [item for item in DEFAULT_GOVERNANCE_MECHANISMS if item.mechanism_id == "global_reference_controller"][0]
        decision = governance_decision(
            mechanism,
            values=[2, 1, 3, 4],
            labels=["a", "b", "a", "b"],
            actor_index=0,
            target_index=1,
            goal_assignments={"a": "increasing", "b": "decreasing"},
            policy_ids=("a", "b"),
            array_size=4,
            rng=__import__("random").Random(7),
        )
        self.assertTrue(decision["invoked"])
        self.assertTrue(decision["accepted"])
        self.assertGreater(decision["global_reference_after"], decision["global_reference_before"])
        self.assertGreater(decision["cost_units"], 1)

    def test_small_s09_sweep_runs_and_validates_access_strata(self) -> None:
        records = builtin_control_records()
        config = S09Config(
            array_size=20,
            event_cap=35,
            seeds=(11, 12),
            max_base_conditions=1,
            interface_rules=(DEFAULT_INTERFACE_RULES[0], DEFAULT_INTERFACE_RULES[1]),
            governance_mechanisms=(DEFAULT_GOVERNANCE_MECHANISMS[0], DEFAULT_GOVERNANCE_MECHANISMS[1], DEFAULT_GOVERNANCE_MECHANISMS[-1]),
        )
        run_df, condition_df, selected = run_s09_sweep(records, _sample_s04(records), _sample_s08_strata(config), config)
        summary = summarize_s09_runs(run_df)
        comparison = compare_to_no_governance_baseline(summary)
        rankings = governance_rankings(comparison, config)
        info = information_access_strata(comparison)
        validation = validate_s09_outputs(
            run_df,
            condition_df,
            selected,
            summary,
            comparison,
            rankings,
            info,
            _sample_s08_strata(config),
            config,
            figure_written=True,
            unit_tests_success=True,
        )
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertEqual(len(run_df), 12)
        self.assertTrue(cases["no_governance_baseline_for_each_base_interface"])
        self.assertTrue(cases["information_access_documented"])
        self.assertTrue(cases["global_controller_access_labeled_and_stratified"])
        self.assertTrue(cases["matched_seed_sets_by_base_interface_governance"])


if __name__ == "__main__":
    unittest.main()
