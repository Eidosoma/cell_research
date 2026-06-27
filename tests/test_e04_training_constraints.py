from __future__ import annotations

import unittest

from e02_deterministic_simulator.simulator import StepOutcome
from memory_repair import (
    LEARNING_PENDING_ACTION_KEY,
    LocalLearningConfig,
    LocalLearningPolicyWrapper,
    TrainingProtocolConfig,
    audit_observation_log,
    audit_policy_spec_for_oracle_access,
    audit_reward_record,
    build_homeostatic_benchmark_config,
    compute_local_learning_reward,
    evaluate_global_oracle_homeostatic_task,
    global_oracle_baseline_protocol,
    global_oracle_baseline_record,
    local_only_training_protocol,
    training_protocol_bundle,
)
from memory_repair.learning import LearningEventSimulator
from morphospace import SelectionPolicy


class TestE04TrainingConstraints(unittest.TestCase):
    def test_local_protocol_rejects_oracle_flags(self) -> None:
        with self.assertRaises(ValueError):
            TrainingProtocolConfig(mode="local_only", oracle_allowed=True)
        protocol = local_only_training_protocol()
        self.assertFalse(protocol.oracle_allowed)
        self.assertEqual(protocol.mode, "local_only")
        bundle = training_protocol_bundle()
        self.assertFalse(bundle["localOnlyProtocol"]["oracleAllowed"])
        self.assertTrue(bundle["globalOracleBaselineProtocol"]["oracleAllowed"])

    def test_s06_reward_record_passes_local_only_audit(self) -> None:
        config = LocalLearningConfig("local_adaptive")
        policy = LocalLearningPolicyWrapper("bubble", config)
        sim = LearningEventSimulator([2, 1], policy, scheduler_seed=1, tie_breaker_seed=1)
        actor = sim.cells[0]
        observation = actor.policy.observe(sim.cells, 0, actor.state, sim.frozen_variant)
        action = actor.policy.propose_action(observation, actor.state, sim.tie_rng, forced_direction=1)
        pending = dict(actor.state)
        pending[LEARNING_PENDING_ACTION_KEY] = "swap_right"
        outcome = StepOutcome(True, actor.cell_id, 0, 1, 1, True, 1, 1, False, "learning_local_swap")
        record = compute_local_learning_reward(observation, pending, action, outcome, config)
        audit = audit_reward_record(record, local_only_training_protocol())
        self.assertTrue(audit["success"], audit)

    def test_reward_leakage_is_rejected_locally(self) -> None:
        leaky = {
            "inputFieldsUsed": ["actor_value", "global_sortedness"],
            "localInputs": {"actor_value": 2, "whole_array_values": [2, 1]},
            "usesGlobalSortednessSignal": True,
        }
        audit = audit_reward_record(leaky, local_only_training_protocol())
        self.assertFalse(audit["success"])
        self.assertIn("global_sortedness", audit["disallowedInputs"])
        self.assertIn("whole_array_values", audit["disallowedLocalInputKeys"])

    def test_observation_log_leakage_is_rejected(self) -> None:
        local = {"fields": ["actor_value", "left_value", "right_value", "actor_local_memory"]}
        self.assertTrue(audit_observation_log(local, local_only_training_protocol())["success"])
        leaky = {"fields": ["actor_value", "future_target_array"], "future_target_array": [1, 2, 3]}
        audit = audit_observation_log(leaky, local_only_training_protocol())
        self.assertFalse(audit["success"])
        self.assertIn("future_target_array", audit["disallowedObservationFields"])

    def test_policy_spec_audit_classifies_target_oracles(self) -> None:
        local_policy = LocalLearningPolicyWrapper("bubble", LocalLearningConfig("local_adaptive"))
        self.assertTrue(audit_policy_spec_for_oracle_access(local_policy, local_only_training_protocol())["success"])

        selection_audit = audit_policy_spec_for_oracle_access(SelectionPolicy(), local_only_training_protocol())
        self.assertFalse(selection_audit["success"])
        self.assertIn("selection_target_position_semantics", {item["field"] for item in selection_audit["violations"]})
        self.assertTrue(audit_policy_spec_for_oracle_access(SelectionPolicy(), global_oracle_baseline_protocol())["success"])

        target_dsl = {
            "policy_id": "dsl_target_leak",
            "family": "dsl",
            "algotype": "dsl",
            "parameters": {
                "program": {
                    "rules": [
                        {
                            "name": "target_rule",
                            "when": [{"op": "target_exists"}],
                            "then": {"action": "swap_target", "target_key": "target_position"},
                        }
                    ]
                }
            },
        }
        dsl_audit = audit_policy_spec_for_oracle_access(target_dsl, local_only_training_protocol())
        self.assertFalse(dsl_audit["success"])
        fields = {item["field"] for item in dsl_audit["violations"]}
        self.assertIn("target_exists", fields)
        self.assertIn("swap_target", fields)

    def test_global_oracle_baseline_is_labeled_and_separate(self) -> None:
        record = global_oracle_baseline_record([2, 1])
        self.assertFalse(audit_reward_record(record, local_only_training_protocol())["success"])
        self.assertTrue(audit_reward_record(record, global_oracle_baseline_protocol())["success"])

        task = next(task for task in build_homeostatic_benchmark_config()["tasks"] if task["taskId"] == "adjacent_swap_recovery")
        result = evaluate_global_oracle_homeostatic_task(task)
        self.assertTrue(result["oracleAllowed"])
        self.assertTrue(result["oracleBaseline"])
        self.assertEqual(result["timeInRangeFraction"], 1.0)
        self.assertEqual(result["failureDurationTicks"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
