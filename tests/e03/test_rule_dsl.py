"""S02 rule-DSL tests."""

from __future__ import annotations

import random
import unittest

from src.e03.rule_dsl import (
    DSLArrayState,
    DSLInterpreter,
    DSLValidationError,
    SELECTION_TARGET_POLICY,
    SIMPLE_SWAP_LEFT_POLICY,
    SIMPLE_SWAP_RIGHT_POLICY,
    parse_and_render_round_trip,
    parse_policy,
)


class RuleDSLTests(unittest.TestCase):
    def test_parse_render_json_round_trip_and_hash_are_stable(self) -> None:
        policy, rendered, reparsed = parse_and_render_round_trip(SIMPLE_SWAP_LEFT_POLICY)
        self.assertEqual(policy.to_dict(), reparsed.to_dict())
        self.assertEqual(rendered, policy.to_source())
        self.assertEqual(policy.sha256, reparsed.sha256)
        from_json = policy.from_json(policy.to_json())
        self.assertEqual(policy.to_dict(), from_json.to_dict())
        self.assertEqual(policy.policy_id, from_json.policy_id)

    def test_simple_swap_left_executes_compare_and_swap(self) -> None:
        policy = parse_policy(SIMPLE_SWAP_LEFT_POLICY)
        result = DSLInterpreter(policy).step_state(DSLArrayState(values=(2, 1, 3), actor_index=1))
        self.assertEqual(result.action.action_type, "swap")
        self.assertEqual(result.action.target_index, 0)
        self.assertTrue(result.action.compare_counted)
        self.assertEqual(result.state_after.values, (1, 2, 3))
        self.assertEqual(result.state_after.actor_index, 0)

    def test_simple_swap_right_executes_compare_and_swap(self) -> None:
        policy = parse_policy(SIMPLE_SWAP_RIGHT_POLICY)
        result = DSLInterpreter(policy).step_state(DSLArrayState(values=(2, 1), actor_index=0))
        self.assertEqual(result.action.action_type, "swap")
        self.assertEqual(result.action.target_index, 1)
        self.assertTrue(result.action.compare_counted)
        self.assertEqual(result.state_after.values, (1, 2))
        self.assertEqual(result.state_after.actor_index, 1)

    def test_rule_else_waits_when_guard_is_false(self) -> None:
        policy = parse_policy(SIMPLE_SWAP_LEFT_POLICY)
        result = DSLInterpreter(policy).step_state(DSLArrayState(values=(1, 2, 3), actor_index=1))
        self.assertEqual(result.action.action_type, "wait")
        self.assertFalse(result.action.compare_counted)
        self.assertEqual(result.state_after.values, (1, 2, 3))

    def test_selection_target_state_is_first_class(self) -> None:
        policy = parse_policy(SELECTION_TARGET_POLICY)
        update = DSLInterpreter(policy).step_state(DSLArrayState(values=(1, 2, 3), actor_index=1))
        self.assertEqual(update.state_before.ideal_position, 0)
        self.assertEqual(update.action.action_type, "update_state")
        self.assertEqual(update.action.state_update["ideal_position"], 1)
        self.assertEqual(update.state_after.ideal_position, 1)

        swap = DSLInterpreter(policy).step_state(DSLArrayState(values=(2, 1, 3), actor_index=1))
        self.assertEqual(swap.action.action_type, "swap")
        self.assertEqual(swap.state_after.values, (1, 2, 3))

    def test_invalid_rules_are_rejected(self) -> None:
        invalid_sources = [
            "policy bad v1\nrule else wait\n",
            "policy bad v1\nrule if target_exists(up) then wait\nend\n",
            "policy bad v1\nrule if always then teleport(left)\nend\n",
            "policy bad v1\nrule else wait\nrule if always then wait\nend\n",
            "policy bad v1\nrule if random_lt(1.2) then wait\nend\n",
        ]
        for source in invalid_sources:
            with self.subTest(source=source):
                with self.assertRaises(DSLValidationError):
                    parse_policy(source)

    def test_probabilistic_choose_is_deterministic_for_fixed_seed(self) -> None:
        source = """policy choose_swap v1
state ideal_position=none
rule if always then choose(0.75, swap(right), wait)
end
"""
        policy = parse_policy(source)
        first = DSLInterpreter(policy).step_state(DSLArrayState(values=(2, 1), actor_index=0), random.Random(7))
        second = DSLInterpreter(policy).step_state(DSLArrayState(values=(2, 1), actor_index=0), random.Random(7))
        self.assertEqual(first.action.action_type, second.action.action_type)
        self.assertEqual(first.state_after.values, second.state_after.values)
        self.assertEqual(first.action.metadata["choice_roll"], second.action.metadata["choice_roll"])


if __name__ == "__main__":
    unittest.main()
