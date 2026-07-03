"""E06 S10 staged graft experiment tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e06.algotype_library import builtin_control_records
from src.e06.governance_mechanisms import DEFAULT_GOVERNANCE_MECHANISMS, DEFAULT_INTERFACE_RULES
from src.e06.graft_experiments import (
    GraftSpec,
    S10Config,
    build_s10_conditions,
    compare_to_behavior_baseline,
    graft_layout_labels,
    graft_strata_summary,
    run_s10_sweep,
    select_s10_base_contexts,
    summarize_s10_runs,
    validate_s10_outputs,
)


def _json(values: list) -> str:
    return json.dumps(values, separators=(",", ":"))


def _sample_s07_mosaic(records: list[dict]) -> pd.DataFrame:
    left = records[2]
    right = records[3]
    rows = []
    specs = [
        ("s07_unit_exemplar", [0.75, 0.25], "stable_segregated_patch", 0.72, 0.12, 14.0, 0.62, 0.26, 0.18),
        ("s07_unit_conflict", [0.50, 0.50], "partial_mosaic", 0.44, 0.68, 31.0, 0.54, 0.72, 0.91),
        ("s07_unit_low_quality", [0.25, 0.75], "absorbed_mixed", 0.33, 0.54, 8.0, 0.21, 0.83, 0.63),
    ]
    for idx, (run_id, ratios, label, quality, conflict, aggregation, block, mosaic, conflict_score) in enumerate(specs):
        rows.append(
            {
                "run_id": run_id,
                "source_research_step_id": "S07",
                "source_condition_id": f"s07_unit_condition_{idx}",
                "policy_ids_json": _json([left["algotypeId"], right["algotypeId"]]),
                "display_names_json": _json([left["displayName"], right["displayName"]]),
                "source_categories_json": _json([left["sourceCategory"], right["sourceCategory"]]),
                "ratio_targets_json": _json(ratios),
                "arrangement": "graft_like_insertions" if idx == 0 else "alternating",
                "goal_profile_id": "opposite_direction" if idx else "same_increasing",
                "goal_compatibility_class": "opposite_goal" if idx else "same_goal",
                "s07_label": label,
                "s07_classification_mode": "metric_continuum",
                "final_target_quality": quality,
                "goal_conflict_index": conflict,
                "aggregation_delta_percent": aggregation,
                "largest_block_fraction": block,
                "mosaic_continuum_score": mosaic,
                "conflict_continuum_score": conflict_score,
                "panel": "unit",
                "candidate_reason": "unit_fixture",
                "value_profile": "random_permutation",
                "perturbation_profile": "none",
            }
        )
    return pd.DataFrame(rows)


def _sample_s07_exemplars(mosaic: pd.DataFrame) -> pd.DataFrame:
    return mosaic.head(1).copy()


def _sample_s09_rankings(config: S10Config) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "governance_mechanism_id": mechanism.mechanism_id,
                "global_controller_like": mechanism.global_controller_like,
                "mean_rescue_effect_score": 0.0,
            }
            for mechanism in config.governance_mechanisms
        ]
    )


class GraftExperimentTests(unittest.TestCase):
    def test_graft_layout_uses_s03_arrangement_families(self) -> None:
        spec = GraftSpec("unit_center", 5, 4, "center", "graft_like_insertions", "unit")
        labels = graft_layout_labels("host", "graft", spec, array_size=12, seed=7, condition_id="unit")
        self.assertEqual(labels.count("graft"), 4)
        self.assertEqual(labels[4:8], ("graft", "graft", "graft", "graft"))

        edge_spec = GraftSpec("unit_left", 5, 3, "left_edge", "contiguous_patch", "unit")
        edge_labels = graft_layout_labels("host", "graft", edge_spec, array_size=10, seed=7, condition_id="unit")
        self.assertEqual(edge_labels[:3], ("graft", "graft", "graft"))
        self.assertEqual(edge_labels.count("host"), 7)

    def test_select_s10_base_contexts_records_host_graft_identities(self) -> None:
        records = builtin_control_records()
        config = S10Config(max_base_contexts=2)
        selected = select_s10_base_contexts(_sample_s07_mosaic(records), _sample_s07_exemplars(_sample_s07_mosaic(records)), config)
        self.assertEqual(len(selected), 2)
        self.assertTrue(selected["selection_reason"].astype(str).str.contains("s07_exemplar").any())
        self.assertTrue(selected["selection_reason"].astype(str).str.contains("metric_extreme").any())
        self.assertTrue((selected["host_policy_id"] != selected["graft_policy_id"]).all())
        self.assertTrue(selected["host_display_name"].astype(str).str.len().gt(0).all())
        self.assertTrue(selected["graft_display_name"].astype(str).str.len().gt(0).all())

    def test_small_s10_sweep_runs_and_validates_strata(self) -> None:
        records = builtin_control_records()
        config = S10Config(
            array_size=20,
            event_cap=40,
            seeds=(11, 12),
            max_base_contexts=2,
            graft_specs=(GraftSpec("unit_center", 10, 4, "center", "graft_like_insertions", "unit"),),
            interface_rules=(DEFAULT_INTERFACE_RULES[0], DEFAULT_INTERFACE_RULES[1]),
            governance_mechanisms=(DEFAULT_GOVERNANCE_MECHANISMS[0], DEFAULT_GOVERNANCE_MECHANISMS[1], DEFAULT_GOVERNANCE_MECHANISMS[-1]),
        )
        mosaic = _sample_s07_mosaic(records)
        exemplars = _sample_s07_exemplars(mosaic)
        run_df, condition_df, selected, pre_state_df = run_s10_sweep(records, mosaic, exemplars, _sample_s09_rankings(config), config)
        summary = summarize_s10_runs(run_df)
        comparison = compare_to_behavior_baseline(summary)
        strata = graft_strata_summary(comparison, config)
        validation = validate_s10_outputs(
            run_df,
            condition_df,
            selected,
            pre_state_df,
            summary,
            comparison,
            strata,
            config,
            figure_written=True,
            unit_tests_success=True,
        )
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertEqual(len(run_df), 24)
        self.assertEqual(pre_state_df["pre_graft_state_id"].nunique(), 4)
        self.assertTrue(cases["selected_contexts_use_s07_exemplars_and_continua"])
        self.assertTrue(cases["host_pre_graft_state_saved"])
        self.assertTrue(cases["behavior_explicit_governance_global_strata_separated"])
        self.assertTrue(cases["matched_seed_sets_by_condition"])
        self.assertTrue(cases["post_graft_counts_preserved"])


if __name__ == "__main__":
    unittest.main()
