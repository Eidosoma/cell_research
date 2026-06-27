from __future__ import annotations

import unittest

from morphospace import (
    DSLPolicy,
    PolicyEventSimulator,
    RuleDslError,
    local_inversion_program,
    null_program,
    parse_rule_program,
    policy_from_spec,
    stochastic_right_program,
)


class TestE03RuleDsl(unittest.TestCase):
    def test_round_trip_and_pretty_print(self) -> None:
        program = local_inversion_program()
        parsed = parse_rule_program(program.to_json())
        self.assertEqual(parsed.to_dict(), program.to_dict())
        self.assertIn("move_left_if_smaller", parsed.pretty())
        self.assertIn("swap_left", parsed.pretty())

    def test_policy_spec_round_trip(self) -> None:
        policy = DSLPolicy(local_inversion_program())
        restored = policy_from_spec(policy.to_spec())
        self.assertEqual(restored.to_spec().to_dict(), policy.to_spec().to_dict())

    def test_invalid_predicate_and_probability_raise(self) -> None:
        with self.assertRaises(RuleDslError):
            parse_rule_program(
                {
                    "policy_id": "bad_predicate",
                    "rules": [{"when": [{"op": "teleport"}], "then": {"action": "wait"}}],
                }
            )
        with self.assertRaises(RuleDslError):
            parse_rule_program(
                {
                    "policy_id": "bad_probability",
                    "rules": [{"when": [{"op": "always"}], "then": {"action": "swap_right", "probability": 1.5}}],
                }
            )

    def test_null_policy_executes_and_stops_without_swaps(self) -> None:
        result = PolicyEventSimulator([3, 1, 2], DSLPolicy(null_program()), scheduler_seed=1, tie_breaker_seed=2).run()
        self.assertFalse(result.completed)
        self.assertEqual(result.stop_reason, "no_cell_can_move_after_two_checks")
        self.assertEqual(result.swap_count, 0)

    def test_local_inversion_rule_sorts_small_array(self) -> None:
        result = PolicyEventSimulator(
            [5, 1, 4, 2, 3],
            DSLPolicy(local_inversion_program()),
            scheduler_seed=10,
            tie_breaker_seed=20,
        ).run(max_activations=100000)
        self.assertTrue(result.completed)
        self.assertEqual(result.final_values, [1, 2, 3, 4, 5])

    def test_stochastic_rule_is_seed_replayable(self) -> None:
        kwargs = {
            "initial_values": [4, 1, 3, 2],
            "policies": DSLPolicy(stochastic_right_program()),
            "scheduler_seed": 42,
            "tie_breaker_seed": 99,
        }
        first = PolicyEventSimulator(**kwargs).run(max_activations=50)
        second = PolicyEventSimulator(**kwargs).run(max_activations=50)
        self.assertEqual(first.final_values, second.final_values)
        self.assertEqual(first.swap_count, second.swap_count)
        self.assertEqual([row["state_hash"] for row in first.trace_rows], [row["state_hash"] for row in second.trace_rows])

    def test_memory_target_estimate_and_signal_updates(self) -> None:
        program = parse_rule_program(
            {
                "policy_id": "memory_signal_demo",
                "initial_state": {"counter": 0},
                "rules": [
                    {
                        "name": "remember",
                        "when": [{"op": "always"}],
                        "then": {
                            "action": "remember",
                            "updates": [
                                {"op": "estimate_target_position", "key": "target_position", "direction": "increasing"},
                                {"op": "increment", "key": "counter", "amount": 1, "min": 0, "max": 2},
                                {"op": "set", "key": "seen_value", "value": {"expr": "actor_value"}},
                                {"op": "signal", "channel": "marker", "value": "seen"},
                            ],
                        },
                    }
                ],
                "default": {"action": "wait"},
            }
        )
        sim = PolicyEventSimulator([3, 1, 2], DSLPolicy(program), scheduler_seed=1, tie_breaker_seed=2)
        sim.step(forced_cell_id=0)
        state = sim.cells[sim.positions_by_id[0]].state
        self.assertEqual(state["target_position"], 2)
        self.assertEqual(state["counter"], 1)
        self.assertEqual(state["seen_value"], 3)
        self.assertEqual(state["signal_marker"], "seen")

    def test_world_constraints_block_stuck_frozen_dsl_swap(self) -> None:
        sim = PolicyEventSimulator(
            [2, 1],
            DSLPolicy(local_inversion_program()),
            frozen_positions=[1],
            frozen_variant="stuck",
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        outcome = sim.step(forced_cell_id=0)
        self.assertFalse(outcome.swapped)
        self.assertTrue(outcome.blocked_move_attempt)
        self.assertEqual(sim.current_values(), [2, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
