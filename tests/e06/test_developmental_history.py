"""E06 S12 developmental-history experiment tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e04.evolutionary_search import PARAMETER_NAMES
from src.e06.developmental_history import (
    DEFAULT_HISTORY_SPECS,
    HistorySpec,
    S12Config,
    compare_to_simultaneous_baseline,
    history_strata_summary,
    history_spec_table,
    run_s12_sweep,
    select_s12_base_contexts,
    summarize_s12_runs,
    validate_s12_outputs,
)
from src.e06.governance_mechanisms import DEFAULT_GOVERNANCE_MECHANISMS, DEFAULT_INTERFACE_RULES


def _json(values: list) -> str:
    return json.dumps(values, separators=(",", ":"))


def _records() -> list[dict]:
    params = {name: 0.0 for name in PARAMETER_NAMES}
    params.update({"disorder_weight": 1.0, "illegal_penalty": 6.0, "swap_right_bias": 0.1})
    return [
        {
            "algotypeId": "orig_bubble",
            "sourceCategory": "original",
            "displayName": "original_bubble",
            "executionBackend": "e02_public_cell_simulator",
            "algorithm": "bubble",
        },
        {
            "algotypeId": "orig_selection",
            "sourceCategory": "original",
            "displayName": "original_selection",
            "executionBackend": "e02_public_cell_simulator",
            "algorithm": "selection",
        },
        {
            "algotypeId": "mem_repair",
            "sourceCategory": "memory_repair",
            "displayName": "e04_memory_repair_01",
            "executionBackend": "e04_local_training_policy",
            "algorithm": "memory_repair_adjacent",
            "sourcePolicyId": "e04_unit_memory",
            "parameters": params,
        },
    ]


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"algotype_id": "orig_bubble", "display_name": "original_bubble", "source_category": "original"},
            {"algotype_id": "orig_selection", "display_name": "original_selection", "source_category": "original"},
            {"algotype_id": "mem_repair", "display_name": "e04_memory_repair_01", "source_category": "memory_repair"},
        ]
    )


def _s10_selected() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "base_context_id": "s10_base_unit",
                "selection_rank": 1,
                "selection_reason": "s07_exemplar",
                "source_s07_run_id": "s07_unit",
                "source_research_step_id": "S07",
                "source_condition_id": "s07_condition_unit",
                "s07_label": "diffuse_intermediate",
                "s07_classification_mode": "metric_continuum_with_exemplars",
                "panel": "unit",
                "candidate_reason": "unit_fixture",
                "source_arrangement": "random_permutation_labels",
                "goal_profile_id": "same_increasing",
                "goal_compatibility_class": "same_goal",
                "policy_ids_json": _json(["orig_bubble", "orig_selection"]),
                "display_names_json": _json(["original_bubble", "original_selection"]),
                "source_categories_json": _json(["original", "original"]),
                "ratio_targets_json": _json([0.5, 0.5]),
                "host_policy_id": "orig_bubble",
                "graft_policy_id": "orig_selection",
                "host_display_name": "original_bubble",
                "graft_display_name": "original_selection",
                "host_source_category": "original",
                "graft_source_category": "original",
                "value_profile": "random_permutation",
                "perturbation_profile": "none",
                "source_final_target_quality": 0.7,
                "source_goal_conflict_index": 0.0,
                "source_aggregation_delta_percent": 4.0,
                "source_largest_block_fraction": 0.25,
                "source_mosaic_continuum_score": 0.2,
                "source_conflict_continuum_score": 0.1,
                "s10_graft_sensitivity_score": 0.5,
            }
        ]
    )


def _s11_runs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"base_context_id": "s10_base_unit", "takeover_success": False},
            {"base_context_id": "s10_base_unit", "takeover_success": True},
        ]
    )


def _s09_rankings(config: S12Config) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "governance_mechanism_id": mechanism.mechanism_id,
                "global_controller_like": mechanism.global_controller_like,
            }
            for mechanism in config.governance_mechanisms
        ]
    )


class DevelopmentalHistoryTests(unittest.TestCase):
    def test_history_specs_cover_required_families(self) -> None:
        table = history_spec_table()
        self.assertIn("simultaneous", set(table["history_family"]))
        self.assertIn("staged_introduction", set(table["history_family"]))
        self.assertIn("transient_perturbation", set(table["history_family"]))
        self.assertIn("prior_exposure_memory", set(table["history_family"]))
        self.assertEqual(len(table), len(DEFAULT_HISTORY_SPECS))

    def test_select_s12_contexts_adds_e04_memory_input(self) -> None:
        config = S12Config(max_s10_contexts=1, max_memory_contexts=1)
        selected = select_s12_base_contexts(_s10_selected(), _s11_runs(), _metadata(), _records(), config)
        self.assertEqual(set(selected["source_context_family"]), {"s10_graft_sensitive", "e04_memory_policy_input"})
        self.assertTrue(selected["memory_policy_present"].astype(bool).any())
        self.assertIn("random_permutation", set(selected["source_arrangement"]))

    def test_small_s12_sweep_runs_and_validates_histories(self) -> None:
        histories = (
            HistorySpec("simultaneous_mixed_start", "simultaneous", "matched_mixture_from_event0", "unit"),
            HistorySpec(
                "staged_unit_intro",
                "staged_introduction",
                "host_predevelopment_then_patch",
                "unit",
                introduction_event=10,
                staged_arrangement_reference="graft_like_insertions",
            ),
            HistorySpec(
                "early_transient_unit",
                "transient_perturbation",
                "matched_mixture_from_event0",
                "unit",
                transient_start_event=5,
                transient_end_event=15,
                transient_profile="center_block_10",
            ),
            HistorySpec(
                "prior_unit_memory",
                "prior_exposure_memory",
                "prior_mixed_exposure_then_reset",
                "unit",
                prior_exposure_events=8,
                reset_after_exposure=True,
                preserves_memory_across_reset=True,
            ),
        )
        config = S12Config(
            array_size=20,
            event_cap=60,
            seeds=(11, 12),
            max_s10_contexts=1,
            max_memory_contexts=1,
            history_specs=histories,
            interface_rules=(DEFAULT_INTERFACE_RULES[0], DEFAULT_INTERFACE_RULES[1]),
            governance_mechanisms=(DEFAULT_GOVERNANCE_MECHANISMS[0], DEFAULT_GOVERNANCE_MECHANISMS[1], DEFAULT_GOVERNANCE_MECHANISMS[-1]),
        )
        run_df, condition_df, selected, history_specs = run_s12_sweep(
            _records(),
            _metadata(),
            _s10_selected(),
            _s11_runs(),
            _s09_rankings(config),
            config,
        )
        summary = summarize_s12_runs(run_df)
        comparison = compare_to_simultaneous_baseline(summary)
        strata = history_strata_summary(comparison, config)
        validation = validate_s12_outputs(
            run_df,
            condition_df,
            selected,
            history_specs,
            summary,
            comparison,
            strata,
            config,
            figure_written=True,
            unit_tests_success=True,
        )
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertEqual(len(run_df), 96)
        self.assertTrue(cases["histories_encoded_in_config"])
        self.assertTrue(cases["s10_and_e04_memory_contexts_selected"])
        self.assertTrue(cases["matched_final_composition_where_possible"])
        self.assertTrue(cases["behavior_explicit_finite_global_strata_preserved"])
        self.assertTrue(cases["staged_introduction_logged"])
        self.assertTrue(cases["transient_perturbation_windows_encoded"])
        self.assertTrue(cases["prior_exposure_preserves_memory_across_reset"])
        self.assertTrue(cases["matched_seed_sets_by_condition"])
        self.assertTrue(cases["no_invalid_actions"])


if __name__ == "__main__":
    unittest.main()
