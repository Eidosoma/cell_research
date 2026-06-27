from __future__ import annotations

import unittest

from e02_deterministic_simulator import DeterministicEventSimulator
from morphospace import (
    BubblePolicy,
    InsertionPolicy,
    NullPolicy,
    PolicyEventSimulator,
    RandomWalkPolicy,
    SelectionPolicy,
    policy_from_json,
    policy_to_json,
)


class TestE03PolicyInterface(unittest.TestCase):
    def assert_matches_e02(self, values, algotypes, **kwargs) -> None:
        policy_result = PolicyEventSimulator(values, algotypes, **kwargs).run(max_activations=200000)
        e02_result = DeterministicEventSimulator(values, algotypes, **kwargs).run(max_activations=200000)
        self.assertEqual(policy_result.completed, e02_result.completed)
        self.assertEqual(policy_result.stop_reason, e02_result.stop_reason)
        self.assertEqual(policy_result.final_values, e02_result.final_values)
        self.assertEqual(policy_result.final_algotypes, e02_result.final_algotypes)
        self.assertEqual(policy_result.final_frozen_positions, e02_result.final_frozen_positions)
        self.assertEqual(policy_result.swap_count, e02_result.swap_count)
        self.assertEqual(policy_result.comparison_count, e02_result.comparison_count)
        self.assertEqual(policy_result.archived_compare_and_swap_count, e02_result.archived_compare_and_swap_count)
        self.assertEqual(policy_result.blocked_move_attempts, e02_result.blocked_move_attempts)
        self.assertEqual([row["state_hash"] for row in policy_result.trace_rows], [row["state_hash"] for row in e02_result.trace_rows])

    def test_policy_specs_round_trip(self) -> None:
        for policy in [BubblePolicy(), InsertionPolicy(), SelectionPolicy(), NullPolicy(), RandomWalkPolicy(0.25)]:
            with self.subTest(policy=policy.policy_id):
                round_tripped = policy_from_json(policy_to_json(policy))
                self.assertEqual(round_tripped.to_spec().to_dict(), policy.to_spec().to_dict())

    def test_forced_frozen_steps_match_e02(self) -> None:
        for frozen_variant, expected_swapped in [("passive", True), ("stuck", False)]:
            with self.subTest(frozen_variant=frozen_variant):
                policy_sim = PolicyEventSimulator(
                    [2, 1],
                    "bubble",
                    frozen_positions=[1],
                    frozen_variant=frozen_variant,
                    scheduler_seed=1,
                    tie_breaker_seed=1,
                )
                e02_sim = DeterministicEventSimulator(
                    [2, 1],
                    "bubble",
                    frozen_positions=[1],
                    frozen_variant=frozen_variant,
                    scheduler_seed=1,
                    tie_breaker_seed=1,
                )
                policy_outcome = policy_sim.step(forced_cell_id=0, forced_direction=1)
                e02_outcome = e02_sim.step(forced_cell_id=0, forced_direction=1)
                self.assertEqual(policy_outcome, e02_outcome)
                self.assertEqual(policy_outcome.swapped, expected_swapped)
                self.assertEqual(policy_sim.current_values(), e02_sim.current_values())
                self.assertEqual(policy_sim.current_frozen_positions(), e02_sim.current_frozen_positions())

    def test_classic_algorithms_match_e02_no_frozen(self) -> None:
        values = [6, 2, 4, 1, 5, 3]
        for algorithm in ["bubble", "insertion", "selection"]:
            with self.subTest(algorithm=algorithm):
                self.assert_matches_e02(
                    values,
                    algorithm,
                    scheduler_seed=123,
                    tie_breaker_seed=456,
                    condition_id=f"policy_{algorithm}",
                )

    def test_classic_algorithms_match_e02_frozen(self) -> None:
        values = [8, 4, 7, 2, 6, 1, 5, 3]
        for algorithm in ["bubble", "insertion", "selection"]:
            for frozen_variant in ["passive", "stuck"]:
                with self.subTest(algorithm=algorithm, frozen_variant=frozen_variant):
                    self.assert_matches_e02(
                        values,
                        algorithm,
                        frozen_positions=[2],
                        frozen_variant=frozen_variant,
                        scheduler_seed=1234,
                        tie_breaker_seed=5678,
                        condition_id=f"policy_{algorithm}_{frozen_variant}",
                    )

    def test_chimera_match_e02(self) -> None:
        self.assert_matches_e02(
            [9, 3, 8, 2, 7, 1, 6, 4, 5],
            ["bubble", "insertion", "selection", "bubble", "insertion", "selection", "bubble", "insertion", "selection"],
            scheduler_seed=222,
            tie_breaker_seed=333,
            condition_id="policy_chimera",
        )

    def test_null_and_random_policies(self) -> None:
        null_result = PolicyEventSimulator([3, 1, 2], NullPolicy(), scheduler_seed=1, tie_breaker_seed=2).run()
        self.assertFalse(null_result.completed)
        self.assertEqual(null_result.stop_reason, "no_cell_can_move_after_two_checks")
        self.assertEqual(null_result.swap_count, 0)

        kwargs = {
            "initial_values": [4, 1, 3, 2],
            "policies": RandomWalkPolicy(0.5),
            "scheduler_seed": 42,
            "tie_breaker_seed": 99,
            "max_activations": 50,
        }
        first = PolicyEventSimulator(
            kwargs["initial_values"],
            kwargs["policies"],
            scheduler_seed=kwargs["scheduler_seed"],
            tie_breaker_seed=kwargs["tie_breaker_seed"],
        ).run(max_activations=kwargs["max_activations"])
        second = PolicyEventSimulator(
            kwargs["initial_values"],
            kwargs["policies"],
            scheduler_seed=kwargs["scheduler_seed"],
            tie_breaker_seed=kwargs["tie_breaker_seed"],
        ).run(max_activations=kwargs["max_activations"])
        self.assertEqual(first.final_values, second.final_values)
        self.assertEqual(first.swap_count, second.swap_count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
