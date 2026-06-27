from __future__ import annotations

import json
import unittest

from e02_deterministic_simulator.simulator import StepOutcome
from memory_repair import (
    LEARNING_PENDING_ACTION_KEY,
    LEARNING_STATE_KEY,
    LOCAL_LEARNING_ACTIONS,
    LOCAL_REWARD_ALLOWED_INPUTS,
    LOCAL_REWARD_FORBIDDEN_INPUTS,
    LearningEventSimulator,
    LocalLearningConfig,
    LocalLearningPolicyWrapper,
    RepairRuleConfig,
    apply_homeostatic_event_to_simulator,
    compute_local_learning_reward,
    learning_information_boundary,
    learning_policy_from_json,
    learning_policy_to_json,
    learning_state_for_trace,
    normalize_action_probabilities,
)


class TestE04LocalLearning(unittest.TestCase):
    def test_probability_normalization_is_bounded(self) -> None:
        probabilities = normalize_action_probabilities(
            {"swap_left": 1000.0, "swap_right": 1.0, "wait": 0.0},
            min_probability=0.05,
            max_probability=0.9,
        )
        self.assertEqual(set(probabilities), set(LOCAL_LEARNING_ACTIONS))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreaterEqual(min(probabilities.values()), 0.05)
        self.assertLessEqual(max(probabilities.values()), 0.9)

    def test_reward_record_uses_only_allowed_local_inputs(self) -> None:
        config = LocalLearningConfig("local_adaptive")
        policy = LocalLearningPolicyWrapper("bubble", config)
        sim = LearningEventSimulator([2, 1], policy, scheduler_seed=1, tie_breaker_seed=1)
        actor = sim.cells[0]
        observation = actor.policy.base_policy.observe(sim.cells, 0, actor.state, sim.frozen_variant)
        action = actor.policy.base_policy.propose_action(observation, actor.state, sim.tie_rng, forced_direction=1)
        action = actor.policy.base_policy.constrain_action(observation, actor.state, action)
        pending = dict(actor.state)
        pending[LEARNING_PENDING_ACTION_KEY] = "swap_right"
        outcome = StepOutcome(True, actor.cell_id, 0, 1, 1, True, 1, 1, False, "learning_local_swap")
        record = compute_local_learning_reward(observation, pending, action, outcome, config)
        self.assertEqual(set(record["inputFieldsUsed"]), set(LOCAL_REWARD_ALLOWED_INPUTS))
        self.assertEqual(record["forbiddenFieldsUsed"], [])
        self.assertFalse(record["usesGlobalSortednessSignal"])
        self.assertFalse(record["usesWholeArrayTargetSignal"])
        for forbidden in LOCAL_REWARD_FORBIDDEN_INPUTS:
            self.assertNotIn(forbidden, record["localInputs"])

    def test_local_adaptive_update_increases_rewarded_action(self) -> None:
        config = LocalLearningConfig("local_adaptive", learning_rate=0.6)
        policy = LocalLearningPolicyWrapper("bubble", config)
        sim = LearningEventSimulator(
            [2, 1],
            policy,
            frozen_positions=[1],
            frozen_variant="stuck",
            repair_config=RepairRuleConfig("nudge_count", nudge_threshold=2),
            scheduler_seed=1,
            tie_breaker_seed=1,
        )
        before = learning_state_for_trace(sim.cells[0].state)["actionProbabilities"]["swap_right"]
        sim.step(forced_cell_id=0, forced_direction=1)
        after_first_nudge = learning_state_for_trace(sim.cells[0].state)["actionProbabilities"]["swap_right"]
        sim.step(forced_cell_id=0, forced_direction=1)
        sim.step(forced_cell_id=0, forced_direction=1)
        after_swap = learning_state_for_trace(sim.cells[1].state)["actionProbabilities"]["swap_right"]
        self.assertGreater(after_first_nudge, before)
        self.assertGreater(after_swap, after_first_nudge)
        self.assertEqual(sim.current_values(), [1, 2])

    def test_fixed_update_keeps_probabilities_constant(self) -> None:
        config = LocalLearningConfig("fixed")
        policy = LocalLearningPolicyWrapper("bubble", config)
        sim = LearningEventSimulator([2, 1], policy, scheduler_seed=1, tie_breaker_seed=1)
        before = learning_state_for_trace(sim.cells[0].state)["actionProbabilities"]
        sim.step(forced_cell_id=0, forced_direction=1)
        after = learning_state_for_trace(sim.cells[1].state)["actionProbabilities"]
        self.assertEqual(before, after)

    def test_random_adaptation_is_seed_replayable(self) -> None:
        def run(seed: int) -> list[float]:
            policy = LocalLearningPolicyWrapper("bubble", LocalLearningConfig("random_adaptation", random_seed=seed))
            sim = LearningEventSimulator([3, 2, 1], policy, scheduler_seed=2, tie_breaker_seed=3)
            for _ in range(6):
                sim.step()
            states = [learning_state_for_trace(cell.state)["actionProbabilities"]["swap_right"] for cell in sim.cells]
            return [round(value, 12) for value in states]

        self.assertEqual(run(10), run(10))
        self.assertNotEqual(run(10), run(11))

    def test_trace_marks_no_global_or_target_leakage(self) -> None:
        policy = LocalLearningPolicyWrapper("bubble", LocalLearningConfig("local_adaptive"))
        sim = LearningEventSimulator([2, 1], policy, scheduler_seed=1, tie_breaker_seed=1)
        sim.step(forced_cell_id=0, forced_direction=1)
        trace_state = json.loads(sim.trace_rows[-1]["actor_learning_state_json"])["learning_state"]
        reward_record = trace_state["lastRewardRecord"]
        self.assertFalse(reward_record["usesGlobalSortednessSignal"])
        self.assertFalse(reward_record["usesWholeArrayTargetSignal"])
        self.assertEqual(reward_record["forbiddenFieldsUsed"], [])
        boundary = learning_information_boundary()
        self.assertFalse(boundary["usesGlobalSortednessSignal"])
        self.assertFalse(boundary["usesWholeArrayTargetSignal"])

    def test_homeostatic_insert_delete_updates_learning_simulator(self) -> None:
        policy = LocalLearningPolicyWrapper("bubble", LocalLearningConfig("local_adaptive"))
        sim = LearningEventSimulator([1, 2, 3, 4], policy, scheduler_seed=1, tie_breaker_seed=1)
        apply_homeostatic_event_to_simulator(sim, {"type": "insert", "position": 0, "value": 99})
        self.assertEqual(sim.current_values(), [99, 1, 2, 3, 4])
        self.assertEqual(len(sim.signal_fields["blocked"]), 5)
        self.assertIn(LEARNING_STATE_KEY, sim.cells[0].state)
        apply_homeostatic_event_to_simulator(sim, {"type": "delete", "position": 0})
        self.assertEqual(sim.current_values(), [1, 2, 3, 4])
        self.assertEqual(len(sim.signal_fields["blocked"]), 4)
        self.assertEqual(sim.eligible_cell_ids(), [0, 1, 2, 3])

    def test_policy_round_trip(self) -> None:
        policy = LocalLearningPolicyWrapper("bubble", LocalLearningConfig("local_adaptive", learning_rate=0.25))
        parsed = learning_policy_from_json(learning_policy_to_json(policy))
        self.assertEqual(parsed.to_spec().to_dict(), policy.to_spec().to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
