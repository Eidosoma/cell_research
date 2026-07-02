"""E04 S11 competency-metric tests."""

from __future__ import annotations

import json
import unittest

from src.e04.competency_metrics import (
    REQUIRED_S11_POLICY_MODES,
    S11_COMPETENCY_AXES,
    S11_DEGRADATION_LEVELS,
    S11CompetencyConfig,
    assert_s11_design,
    build_s11_competency_schedule,
    default_s11_eval_configs,
    default_s11_policy_modes,
    offline_metrics_from_trace,
    run_s11_competency_matrix,
)
from src.e04.evolutionary_search import EvolutionGenome, heuristic_parameter_vector, params_from_vector, policy_id_for


class CompetencyMetricsTests(unittest.TestCase):
    def _genome(self) -> EvolutionGenome:
        params = params_from_vector(heuristic_parameter_vector())
        return EvolutionGenome(
            policy_id=policy_id_for(0, 0, params, ()),
            generation=0,
            population_index=0,
            parent_ids=(),
            mutation_seed=1,
            mutation_scale=0.0,
            parameters=params,
            lineage_note="unit_test_heuristic",
        )

    def test_default_modes_and_configs_cover_required_design(self) -> None:
        modes = default_s11_policy_modes()
        configs = default_s11_eval_configs(seeds=(41001,), max_events=40, transfer_max_events=44)
        audit = assert_s11_design(configs, modes)

        self.assertEqual(tuple(mode.name for mode in modes), REQUIRED_S11_POLICY_MODES)
        self.assertTrue(audit["success"])
        self.assertEqual(set(audit["competencyAxisCounts"]), set(S11_COMPETENCY_AXES))
        self.assertEqual(audit["degradationLevels"], list(S11_DEGRADATION_LEVELS))
        self.assertIn(12, audit["arraySizes"])

    def test_s11_schedules_are_custom_and_do_not_read_state(self) -> None:
        configs = default_s11_eval_configs(seeds=(41001,), max_events=180, transfer_max_events=220)
        by_axis = {config.competency_axis: config for config in configs if config.competency_axis != "graceful_degradation"}
        barrier = build_s11_competency_schedule(by_axis["alternate_route_barrier"])
        transfer = build_s11_competency_schedule(by_axis["transfer_larger_array"])
        graceful = [
            build_s11_competency_schedule(config)
            for config in configs
            if config.competency_axis == "graceful_degradation"
        ]

        self.assertTrue(all(item.safe_representation.startswith("s11_") for item in barrier))
        self.assertGreaterEqual(sum(1 for item in barrier if item.perturbation_type == "freeze"), 2)
        self.assertEqual(len(transfer), 5)
        self.assertEqual([sum(1 for item in schedule if item.perturbation_type == "freeze") for schedule in graceful], [0, 1, 2, 3])

    def test_offline_metrics_use_trace_rows(self) -> None:
        config = S11CompetencyConfig(
            competency_axis="recovery_profile",
            scenario_name="s11_unit_recovery",
            values=(1, 2, 3, 4),
            reliability_mode="no_fatigue_control",
            activation_seed=41001,
            policy_seed=51001,
            schedule_seed=61001,
            max_events=4,
        )
        trace_rows = [
            {"event_step": 0, "sortedness_percent": 100.0, "swap_side": None, "swap_delta": 0},
            {"event_step": 1, "sortedness_percent": 75.0, "swap_side": "left", "swap_delta": 1},
            {"event_step": 2, "sortedness_percent": 100.0, "swap_side": "right", "swap_delta": 1},
            {"event_step": 3, "sortedness_percent": 100.0, "swap_side": None, "swap_delta": 0},
        ]
        metrics = offline_metrics_from_trace(
            config=config,
            trace_rows=trace_rows,
            perturbation_log=[{"event_step": 1, "perturbation_type": "swap"}],
            final_values=(1, 2, 3, 4),
            fatigue_transition_log=(),
            impairment_log=(),
        )

        self.assertTrue(metrics["competency_metrics_offline_from_trace"])
        self.assertEqual(metrics["recovered_perturbation_count"], 1)
        self.assertEqual(metrics["route_left_successful_swaps"], 1)
        self.assertEqual(metrics["route_right_successful_swaps"], 1)
        self.assertAlmostEqual(metrics["route_diversity_score"], 1.0)

    def test_small_s11_runs_have_no_oracle_hits_and_trace_metrics(self) -> None:
        config = S11CompetencyConfig(
            competency_axis="heldout_perturbations",
            scenario_name="s11_unit_mixed_shocks",
            values=(1, 2, 3, 4),
            reliability_mode="fatigue_recovery",
            activation_seed=41011,
            policy_seed=51011,
            schedule_seed=61011,
            max_events=24,
        )
        modes = [
            mode
            for mode in default_s11_policy_modes()
            if mode.name in {"s08_discovered_no_memory", "neighbor_memory", "noisy_diffusive_signaling"}
        ]
        rows, traces = run_s11_competency_matrix(genomes=(self._genome(),), configs=(config,), modes=modes)

        self.assertEqual(len(rows), 3)
        self.assertGreater(len(traces), 3)
        self.assertEqual({row["uses_global_oracle"] for row in rows}, {False})
        self.assertTrue(all(row["competency_metrics_offline_from_trace"] for row in rows))
        self.assertEqual({row["run_id"] for row in rows}, {row["run_id"] for row in traces})
        for row in rows:
            feature_audit = json.loads(row["feature_audit_summary_json"])
            signal_audit = json.loads(row["signal_observed_json"])
            self.assertFalse(feature_audit["usesGlobalOracle"])
            self.assertTrue(signal_audit["targetDerivedSignalFieldsZeroed"])
        noisy = next(row for row in rows if row["policy_mode"] == "noisy_diffusive_signaling")
        self.assertGreater(json.loads(noisy["signal_observed_json"])["noiseAppliedCount"], 0)


if __name__ == "__main__":
    unittest.main()
