"""E04 S08 local-only evolutionary-search tests."""

from __future__ import annotations

import unittest

from src.e04.evolutionary_search import (
    ACTION_NAMES,
    S08EvaluationConfig,
    S08LocalEvolutionPolicy,
    assert_split_separation,
    default_s08_eval_configs,
    heuristic_parameter_vector,
    initial_population,
    next_generation,
    params_from_vector,
    policy_id_for,
    simulate_evolved_policy,
)
from src.e04.evolutionary_search import EvolutionGenome
from src.e04.no_oracle_protocol import LOCAL_ONLY_PROTOCOL_ID, LocalTrainingObservation


class EvolutionarySearchTests(unittest.TestCase):
    def test_default_splits_have_disjoint_seeds_and_holdout_size(self) -> None:
        configs = default_s08_eval_configs(max_events=20, heldout_max_events=24)
        audit = assert_split_separation(configs)
        self.assertTrue(audit["success"])
        self.assertEqual(audit["trainArraySizes"], [6])
        self.assertEqual(audit["validationArraySizes"], [6])
        self.assertEqual(audit["heldoutArraySizes"], [8])

    def test_policy_consumes_projected_local_training_observation(self) -> None:
        params = params_from_vector(heuristic_parameter_vector())
        genome = EvolutionGenome(
            policy_id=policy_id_for(0, 0, params, ()),
            generation=0,
            population_index=0,
            parent_ids=(),
            mutation_seed=1,
            mutation_scale=0.0,
            parameters=params,
        )
        policy = S08LocalEvolutionPolicy(genome)
        observation = LocalTrainingObservation(
            protocol_id=LOCAL_ONLY_PROTOCOL_ID,
            behavior="bubble",
            actor_value=3,
            actor_status="ACTIVE",
            reverse_direction=False,
            left_present=True,
            left_value=4,
            left_status="ACTIVE",
            right_present=True,
            right_value=5,
            right_status="ACTIVE",
            at_left_boundary=False,
            at_right_boundary=False,
            last_move_success=False,
            last_action_type="blocked_swap_attempt",
            time_since_movement=3,
            local_frustration=2,
            failed_swap_count=1,
            neighbor_identity_count=2,
            signal_blocked=0.5,
            signal_frustrated=0.25,
            signal_scope="diffusive",
        )
        decision = policy.select_action(observation)
        self.assertIn(decision.action, ACTION_NAMES)
        self.assertFalse(decision.feature_audit["usesGlobalOracle"])
        self.assertEqual(decision.feature_audit["excludedSignalHits"], [])

    def test_generation_lineage_records_parent_ids(self) -> None:
        population, _backend = initial_population(4, 123)
        children, _backend = next_generation(
            elites=population[:2],
            generation=1,
            population_size=4,
            seed=123,
            mutation_scale=0.2,
        )
        self.assertEqual(children[0].parent_ids, (population[0].policy_id,))
        self.assertEqual(children[1].parent_ids, (population[1].policy_id,))
        self.assertGreater(children[2].mutation_scale, 0.0)

    def test_small_simulation_has_no_projected_feature_oracle_hits(self) -> None:
        population, _backend = initial_population(2, 321)
        config = S08EvaluationConfig(
            split="train",
            values=(1, 2, 3, 4),
            task_name="swap_shocks",
            reliability_mode="no_fatigue_control",
            activation_seed=31,
            policy_seed=32,
            schedule_seed=33,
            max_events=12,
        )
        result = simulate_evolved_policy(config, population[0])
        row = result.to_row()
        self.assertFalse(row["uses_global_oracle"])
        self.assertEqual(row["protocol_id"].split(":")[0], LOCAL_ONLY_PROTOCOL_ID)
        self.assertEqual(row["stop_reason"], "fixed_horizon_complete")


if __name__ == "__main__":
    unittest.main()
