"""E05 S09 scaling benchmark tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.actions import ActionExecutor, MorphogenesisActionRequest
from src.e05.scaling import (
    DEFAULT_EVENT_MULTIPLIER,
    SCALING_TASK_TYPES,
    default_scaling_target_configs,
    execute_swap_wait_fast,
    policy_parameter_signature,
    run_scaling_benchmark,
    scaling_summary_rows,
    select_scaled_patch_mask,
    simulate_scaling_run,
    target_config_rows,
    validate_scaling_target_configs,
)


class ScalingTests(unittest.TestCase):
    def test_scaled_target_configs_pair_small_and_large_targets(self) -> None:
        configs = default_scaling_target_configs()
        rows = target_config_rows(configs)
        by_source = {}
        for config in configs:
            by_source.setdefault(config.source_target_id, set()).add(config.scale_label)

        self.assertEqual(len(configs), 12)
        self.assertTrue(all(labels == {"small", "large"} for labels in by_source.values()))
        self.assertTrue(all(row["constructed_target_error"] == 0.0 for row in rows))
        self.assertTrue(all(row["retuning_allowed"] is False for row in rows))

    def test_scaled_target_validation_cases_pass(self) -> None:
        rows = validate_scaling_target_configs(default_scaling_target_configs())

        self.assertTrue(rows)
        self.assertTrue(all(row["success"] for row in rows))

    def test_large_grid_policy_signature_does_not_retune(self) -> None:
        configs = default_scaling_target_configs()
        signatures = {
            (
                config.source_target_id,
                config.scale_label,
                policy_parameter_signature("s07_local_target_neighbor_descent", DEFAULT_EVENT_MULTIPLIER),
            )
            for config in configs
        }
        by_target = {}
        for source_target_id, scale_label, signature in signatures:
            by_target.setdefault(source_target_id, {})[scale_label] = signature

        self.assertTrue(all(pair["small"] == pair["large"] for pair in by_target.values()))

    def test_scaled_rotated_patch_mask_area_increases_on_large_target(self) -> None:
        configs = default_scaling_target_configs()
        gradient_small = next(config for config in configs if config.source_target_id == "ap_gradient" and config.scale_label == "small")
        gradient_large = next(config for config in configs if config.source_target_id == "ap_gradient" and config.scale_label == "large")
        small_mask = select_scaled_patch_mask(gradient_small.make_target(), config=gradient_small, seed=3101)
        large_mask = select_scaled_patch_mask(gradient_large.make_target(), config=gradient_large, seed=3101)

        self.assertGreater(len(large_mask.site_ids), len(small_mask.site_ids))
        self.assertGreaterEqual(large_mask.width, small_mask.width)
        self.assertGreaterEqual(large_mask.height, small_mask.height)

    def test_fast_swap_wait_path_matches_s04_executor_for_basic_actions(self) -> None:
        config = next(config for config in default_scaling_target_configs() if config.source_target_id == "ap_gradient" and config.scale_label == "small")
        target = config.make_target()
        state_for_s04 = target.constructed_substrate()
        state_for_fast = target.constructed_substrate()
        swap_request = MorphogenesisActionRequest("swap", 0, 1)
        s04_result = ActionExecutor().execute(state_for_s04, swap_request)
        fast_result = execute_swap_wait_fast(state_for_fast, swap_request, population_current=len(target.substrate.site_ids))

        self.assertEqual(s04_result.allowed, fast_result.allowed)
        self.assertEqual(s04_result.reason, fast_result.reason)
        self.assertEqual(s04_result.state_changed, fast_result.state_changed)
        self.assertEqual(s04_result.energy_cost_charged, fast_result.energy_cost_charged)
        self.assertEqual(state_for_s04.signature(), state_for_fast.signature())

        wait_request = MorphogenesisActionRequest("wait", 0)
        s04_wait = ActionExecutor().execute(state_for_s04, wait_request)
        fast_wait = execute_swap_wait_fast(state_for_fast, wait_request, population_current=len(target.substrate.site_ids))
        self.assertEqual(s04_wait.allowed, fast_wait.allowed)
        self.assertEqual(s04_wait.energy_cost_charged, fast_wait.energy_cost_charged)

    def test_simulate_scaling_run_records_transfer_metrics(self) -> None:
        config = next(config for config in default_scaling_target_configs() if config.source_target_id == "ap_gradient" and config.scale_label == "small")
        result = simulate_scaling_run(
            config=config,
            target=config.make_target(),
            task_type="scrambled_embryo",
            policy_id="s07_local_target_neighbor_descent",
            seed=3101,
            event_multiplier=5,
            records_per_run=3,
        )

        self.assertEqual(result.summary_row["research_step_id"], "S09")
        self.assertGreater(result.summary_row["initial_target_error"], 0.0)
        self.assertIn("energy_per_site", result.summary_row)
        self.assertFalse(result.summary_row["retuning_allowed"])
        self.assertFalse(result.summary_row["large_grid_retuned"])
        self.assertEqual(len(result.metric_rows), 16)

    def test_compact_scaling_benchmark_matrix_and_summary(self) -> None:
        configs = tuple(config for config in default_scaling_target_configs() if config.source_target_id == "ap_gradient")
        summaries, metrics, _runs = run_scaling_benchmark(
            configs=configs,
            task_types=SCALING_TASK_TYPES,
            policy_ids=("s07_local_target_neighbor_descent", "s07_random_adjacent_swap_control"),
            seeds=(3101,),
            event_multiplier=3,
            records_per_run=2,
        )
        scaling_rows = scaling_summary_rows(summaries)

        self.assertEqual(len(summaries), 2 * len(SCALING_TASK_TYPES) * 2)
        self.assertEqual(len(metrics), len(summaries) * 2 * 8)
        self.assertTrue(scaling_rows)
        self.assertTrue(all("large_to_small_recovery_ratio" in row for row in scaling_rows))
        self.assertTrue(all(row["population_delta_total"] == 0 for row in summaries))


if __name__ == "__main__":
    unittest.main()
