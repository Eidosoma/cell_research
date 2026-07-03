"""E06 S08 interface-rule intervention tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e06.algotype_library import builtin_control_records
from src.e06.interface_rules import (
    DEFAULT_INTERFACE_RULES,
    S08Config,
    compare_to_behavior_baseline,
    interface_decision,
    mechanism_strata,
    run_s08_sweep,
    select_s08_base_conditions,
    summarize_s08_runs,
    validate_s08_outputs,
)


def _json(values: list) -> str:
    return json.dumps(values, separators=(",", ":"))


def _sample_mosaic(records: list[dict]) -> pd.DataFrame:
    left = records[2]
    right = records[3]
    rows = []
    for idx, label in enumerate(("patchy_mosaic", "polarized_goal_conflict", "low_progress_diffuse")):
        rows.append(
            {
                "run_id": f"s07_unit_{idx}",
                "source_research_step_id": "S02",
                "source_condition_id": f"s02_unit_{idx}",
                "panel": "unit",
                "candidate_reason": "unit",
                "policy_ids_json": _json([left["algotypeId"], right["algotypeId"]]),
                "display_names_json": _json([left["displayName"], right["displayName"]]),
                "source_categories_json": _json([left["sourceCategory"], right["sourceCategory"]]),
                "ratio_targets_json": _json([0.5, 0.5]),
                "arrangement": "random_permutation_labels" if idx != 1 else "alternating",
                "goal_profile_id": "same_increasing",
                "goal_compatibility_class": "same_goal",
                "value_profile": "",
                "perturbation_profile": "",
                "s07_label": label,
                "s07_classification_mode": "metric_continuum_with_exemplars",
                "final_target_quality": 0.4 + idx * 0.2,
                "goal_conflict_index": 0.1 + idx * 0.3,
                "aggregation_delta_percent": 5.0 + idx * 15.0,
                "largest_block_fraction": 0.2 + idx * 0.2,
                "position_bias_abs": 0.05 + idx * 0.2,
                "mosaic_continuum_score": 0.2 + idx * 0.3,
                "conflict_continuum_score": 0.1 + idx * 0.4,
            }
        )
    return pd.DataFrame(rows)


class InterfaceRuleTests(unittest.TestCase):
    def test_select_s08_base_conditions_uses_exemplar_and_extreme_reasons(self) -> None:
        records = builtin_control_records()
        mosaic = _sample_mosaic(records)
        selected = select_s08_base_conditions(mosaic, mosaic.head(1), S08Config(max_base_conditions=2))
        reasons = ",".join(selected["selection_reason"].astype(str))
        self.assertIn("s07_exemplar", reasons)
        self.assertIn("metric_extreme", reasons)
        self.assertTrue((selected["arrangement"] == "random_permutation").any())

    def test_behavior_only_scope_is_no_recognition(self) -> None:
        records = builtin_control_records()
        mosaic = _sample_mosaic(records)
        config = S08Config(array_size=20, event_cap=30, seeds=(101,), max_base_conditions=1)
        run_df, condition_df, selected = run_s08_sweep(records, mosaic, mosaic.head(1), config)
        baseline = run_df[run_df["interface_rule_id"] == "behavior_only"]
        explicit = run_df[run_df["interface_rule_id"] != "behavior_only"]
        self.assertFalse(baseline["explicit_recognition_used"].any())
        self.assertEqual(set(baseline["recognition_access_class"]), {"none"})
        self.assertEqual(int(baseline["interface_rule_invocation_count"].sum()), 0)
        self.assertTrue(explicit["explicit_recognition_used"].all())
        summary = summarize_s08_runs(run_df)
        comparison = compare_to_behavior_baseline(summary)
        strata = mechanism_strata(comparison, config)
        validation = validate_s08_outputs(
            run_df,
            condition_df,
            selected,
            summary,
            comparison,
            strata,
            config,
            figure_written=True,
            unit_tests_success=True,
        )
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertTrue(cases["explicit_recognition_rules_separated_from_original_claims"])
        self.assertTrue(cases["behavior_baseline_for_each_base_condition"])

    def test_permeability_zero_blocks_cross_label_swap(self) -> None:
        rule = [rule for rule in DEFAULT_INTERFACE_RULES if rule.rule_id == "permeability_barrier"][0]
        strict_rule = type(rule)(
            rule_id=rule.rule_id,
            mechanism_family=rule.mechanism_family,
            explicit_recognition_used=rule.explicit_recognition_used,
            recognition_access_class=rule.recognition_access_class,
            permeability=0.0,
            description=rule.description,
        )
        decision = interface_decision(
            strict_rule,
            values=[2, 1, 3, 4],
            labels=["a", "b", "b", "a"],
            actor_index=0,
            target_index=1,
            goal_assignments={"a": "increasing", "b": "increasing"},
            array_size=4,
            rng=__import__("random").Random(17),
        )
        self.assertTrue(decision["explicit_invoked"])
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["block_reason"], "permeability_cross_label_barrier")


if __name__ == "__main__":
    unittest.main()
