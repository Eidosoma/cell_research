"""S01 deterministic event-simulator tests."""

from __future__ import annotations

import unittest

from src.e02.deterministic_simulator import (
    EventTracingStatusProbe,
    SimulatorConfig,
    _build_cells,
    is_sorted,
    has_public_legal_action,
    result_summary,
    simulate,
    stable_json_sha256,
)


class DeterministicSimulatorTests(unittest.TestCase):
    def slow_public_legal_action(self, config: SimulatorConfig) -> bool:
        probe = EventTracingStatusProbe()
        cells, cell_status = _build_cells(config, probe)
        behavior_map = config.label_to_behavior
        for cell in cells:
            if cell.status != cell_status.ACTIVE:
                continue
            label = str(cell.label)
            behavior = behavior_map.get(label, label if label in {"bubble", "insertion", "selection"} else str(cell.cell_type).lower())
            current_idx = int(cell.current_position[0])
            reverse = bool(cell.reverse_direction)
            if behavior == "bubble":
                left_idx = current_idx - 1
                right_idx = current_idx + 1
                if left_idx >= 0 and (cells[left_idx].status == cell_status.ACTIVE or (config.frozen_semantics == "passive" and cells[left_idx].status == cell_status.FREEZE)):
                    if (reverse and cell.value > cells[left_idx].value) or ((not reverse) and cell.value < cells[left_idx].value):
                        return True
                if right_idx < len(cells) and (cells[right_idx].status == cell_status.ACTIVE or (config.frozen_semantics == "passive" and cells[right_idx].status == cell_status.FREEZE)):
                    if (reverse and cell.value < cells[right_idx].value) or ((not reverse) and cell.value > cells[right_idx].value):
                        return True
            elif behavior == "insertion":
                left_idx = current_idx - 1
                if left_idx < 0:
                    continue
                if not (cells[left_idx].status == cell_status.ACTIVE or (config.frozen_semantics == "passive" and cells[left_idx].status == cell_status.FREEZE)):
                    continue
                if not cell.is_enable_to_move():
                    continue
                if (reverse and cell.value > cells[left_idx].value) or ((not reverse) and cell.value < cells[left_idx].value):
                    return True
            elif behavior == "selection":
                if cell.current_position == cell.ideal_position:
                    continue
                target_idx = int(cell.ideal_position[0])
                if 0 <= target_idx < len(cells) and (cells[target_idx].status == cell_status.ACTIVE or (config.frozen_semantics == "passive" and cells[target_idx].status == cell_status.FREEZE)):
                    return True
            else:
                raise ValueError(behavior)
        return False

    def test_fast_legal_action_scan_matches_public_insertion_scan(self) -> None:
        configs = [
            SimulatorConfig(
                values=(6, 4, 5, 1, 3, 2),
                algorithm="insertion",
                frozen_indices=(2,),
                frozen_semantics="passive",
            ),
            SimulatorConfig(
                values=(6, 4, 5, 1, 3, 2),
                algorithm="insertion",
                frozen_indices=(2,),
                frozen_semantics="stuck",
            ),
            SimulatorConfig(
                values=(1, 3, 2, 5, 4, 6),
                algotypes=("bubble", "insertion", "selection", "insertion", "bubble", "insertion"),
                reverse_directions=(False, True, False, False, False, True),
                frozen_indices=(3,),
                frozen_semantics="passive",
            ),
        ]
        for config in configs:
            with self.subTest(config=config):
                probe = EventTracingStatusProbe()
                cells, cell_status = _build_cells(config, probe)
                fast = has_public_legal_action(cells, cell_status, config.label_to_behavior, config.frozen_semantics)
                self.assertEqual(fast, self.slow_public_legal_action(config))

    def test_fixed_seed_repeats_identical_trace_and_activation_log(self) -> None:
        config = SimulatorConfig(
            values=(4, 1, 3, 2),
            algorithm="bubble",
            activation_seed=4101,
            policy_seed=5101,
            max_events=20_000,
        )
        first = simulate(config)
        second = simulate(config)
        self.assertEqual(first.final_values, second.final_values)
        self.assertEqual(first.records, second.records)
        self.assertEqual(first.activation_log, second.activation_log)
        self.assertEqual(result_summary(first)["trace_sha256"], result_summary(second)["trace_sha256"])

    def test_pure_algorithms_sort_small_fixture(self) -> None:
        seeds = {"bubble": 4201, "insertion": 4202, "selection": 4203}
        for algorithm, seed in seeds.items():
            with self.subTest(algorithm=algorithm):
                result = simulate(
                    SimulatorConfig(
                        values=(4, 1, 3, 2),
                        algorithm=algorithm,
                        activation_seed=seed,
                        policy_seed=seed + 1000,
                        max_events=30_000,
                    )
                )
                self.assertEqual(result.stop_reason, "sorted")
                self.assertTrue(is_sorted(result.final_values))
                self.assertEqual(result.final_values, (1, 2, 3, 4))
                self.assertGreater(result.event_count, 0)
                self.assertEqual(len(result.activation_log), result.event_count)

    def test_mixed_same_goal_fixture_sorts_and_preserves_labels(self) -> None:
        config = SimulatorConfig(
            values=(5, 1, 4, 2, 3),
            algotypes=("bubble", "insertion", "bubble", "insertion", "selection"),
            activation_seed=4301,
            policy_seed=5301,
            max_events=50_000,
        )
        result = simulate(config)
        self.assertEqual(result.stop_reason, "sorted")
        self.assertTrue(is_sorted(result.final_values))
        self.assertCountEqual(result.final_labels, config.algotypes)

    def test_stuck_frozen_cell_is_reproducible_and_keeps_frozen_identity_fixed(self) -> None:
        config = SimulatorConfig(
            values=(3, 1, 2, 4),
            algorithm="bubble",
            frozen_indices=(1,),
            frozen_semantics="stuck",
            activation_seed=4401,
            policy_seed=5401,
            max_events=10_000,
            stall_events=100,
        )
        first = simulate(config)
        second = simulate(config)
        self.assertEqual(first.records, second.records)
        self.assertEqual(first.activation_log, second.activation_log)
        self.assertEqual(first.final_frozen_values, (1,))
        self.assertEqual(first.final_frozen_positions, (1,))
        self.assertGreaterEqual(first.frozen_attempt_count, 1)

    def test_dynamic_frozen_behavior_recovers_passive_and_stuck_limits(self) -> None:
        common = dict(
            values=(3, 1, 2, 4),
            algorithm="bubble",
            frozen_indices=(1,),
            activation_seed=4401,
            policy_seed=5401,
            max_events=10_000,
            stall_events=100,
        )
        passive = simulate(SimulatorConfig(**common, frozen_semantics="passive"))
        passive_dynamic = simulate(
            SimulatorConfig(
                **common,
                frozen_semantics="dynamic",
                frozen_behavior={"type": "passive_limit", "label": "unit_passive_limit", "seed": 5401},
            )
        )
        stuck = simulate(SimulatorConfig(**common, frozen_semantics="stuck"))
        stuck_dynamic = simulate(
            SimulatorConfig(
                **common,
                frozen_semantics="dynamic",
                frozen_behavior={"type": "stuck_limit", "label": "unit_stuck_limit", "seed": 5401},
            )
        )
        self.assertEqual(passive_dynamic.final_values, passive.final_values)
        self.assertEqual(passive_dynamic.records, passive.records)
        self.assertEqual(passive_dynamic.activation_log, passive.activation_log)
        self.assertEqual(stuck_dynamic.final_values, stuck.final_values)
        self.assertEqual(stuck_dynamic.records, stuck.records)
        self.assertEqual(stuck_dynamic.activation_log, stuck.activation_log)
        self.assertGreaterEqual(passive_dynamic.frozen_behavior_summary["transition_count"], 1)
        self.assertGreaterEqual(stuck_dynamic.frozen_behavior_summary["blocked_attempt_count"], 1)

    def test_dynamic_frozen_behavior_logs_state_and_attempts(self) -> None:
        config = SimulatorConfig(
            values=(4, 1, 3, 2),
            algorithm="bubble",
            frozen_indices=(1,),
            frozen_semantics="dynamic",
            frozen_behavior={
                "type": "time_varying",
                "label": "unit_time_varying",
                "window_events": 1,
                "seed": 5501,
            },
            activation_seed=4501,
            policy_seed=5501,
            max_events=200,
            stall_events=20,
            stop_when_sorted=False,
        )
        result = simulate(config)
        summary = result.frozen_behavior_summary
        self.assertGreaterEqual(summary["transition_count"], 2)
        self.assertGreaterEqual(summary["attempt_count"], 1)
        self.assertTrue(result.frozen_behavior_transition_log)
        self.assertTrue(result.frozen_behavior_attempt_log)

    def test_sortedness_threshold_can_stop_before_exact_sort(self) -> None:
        result = simulate(
            SimulatorConfig(
                values=(2, 1, 3),
                algorithm="bubble",
                activation_seed=4601,
                policy_seed=5601,
                stop_when_sorted=False,
                stop_sortedness_threshold=60.0,
                max_events=100,
            )
        )
        self.assertEqual(result.stop_reason, "sortedness_threshold_reached")
        self.assertEqual(result.event_count, 0)
        self.assertEqual(result_summary(result)["stop_sortedness_threshold"], 60.0)

    def test_convergence_criteria_stop_with_distinct_reasons(self) -> None:
        cases = [
            ("no_state_change", "no_state_change_window"),
            ("no_swap", "no_swap_window"),
            ("sortedness_plateau", "sortedness_plateau_window"),
        ]
        for criterion, reason in cases:
            with self.subTest(criterion=criterion):
                result = simulate(
                    SimulatorConfig(
                        values=(1, 2, 3),
                        algorithm="bubble",
                        activation_seed=4701,
                        policy_seed=5701,
                        stop_when_sorted=False,
                        convergence_criterion=criterion,
                        stall_events=4,
                        max_events=100,
                    )
                )
                self.assertEqual(result.stop_reason, reason)
                self.assertEqual(result.event_count, 4)

    def test_convergence_none_runs_until_event_cap_when_sorted_stop_disabled(self) -> None:
        result = simulate(
            SimulatorConfig(
                values=(1, 2, 3),
                algorithm="bubble",
                activation_seed=4801,
                policy_seed=5801,
                stop_when_sorted=False,
                convergence_criterion="none",
                max_events=5,
            )
        )
        self.assertEqual(result.stop_reason, "max_events_exceeded")
        self.assertTrue(result.max_guard_hit)
        self.assertEqual(result.event_count, 5)

    def test_max_successful_swap_cap_is_encoded_and_reported(self) -> None:
        result = simulate(
            SimulatorConfig(
                values=(3, 2, 1),
                algorithm="bubble",
                activation_seed=4901,
                policy_seed=5901,
                max_successful_swaps=1,
                max_events=10_000,
                stop_when_sorted=False,
            )
        )
        self.assertEqual(result.stop_reason, "max_successful_swaps_exceeded")
        self.assertTrue(result.max_guard_hit)
        self.assertGreaterEqual(result.swap_count, 1)
        self.assertEqual(result_summary(result)["max_successful_swaps"], 1)

    def test_trace_hash_changes_when_activation_seed_changes(self) -> None:
        base = SimulatorConfig(values=(4, 1, 3, 2), algorithm="bubble", activation_seed=4501, policy_seed=5501)
        changed = SimulatorConfig(values=(4, 1, 3, 2), algorithm="bubble", activation_seed=4502, policy_seed=5501)
        base_hash = stable_json_sha256(simulate(base).activation_log)
        changed_hash = stable_json_sha256(simulate(changed).activation_log)
        self.assertNotEqual(base_hash, changed_hash)


if __name__ == "__main__":
    unittest.main()
