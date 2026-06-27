from __future__ import annotations

import json
import unittest

import numpy as np

from morphospace import BubblePolicy, DSLPolicy, PolicyEventSimulator, parse_rule_program
from memory_repair import (
    SIGNAL_STATE_KEY,
    SignalConfig,
    SignalEventSimulator,
    SignalPolicyWrapper,
    diffuse_signal_fields,
    empty_signal_fields,
    signal_policy_from_json,
    signal_policy_to_json,
    signal_state_for_trace,
)


class TestE04LocalSignals(unittest.TestCase):
    def test_no_signal_wrapper_matches_e03(self) -> None:
        values = [6, 2, 4, 1, 5, 3]
        base = PolicyEventSimulator(values, "bubble", scheduler_seed=123, tie_breaker_seed=456).run(
            max_activations=200000
        )
        wrapped = SignalEventSimulator(
            values,
            SignalPolicyWrapper(BubblePolicy(), "no_signal"),
            signal_config="no_signal",
            scheduler_seed=123,
            tie_breaker_seed=456,
            trace_signal_activations=False,
            trace_memory_activations=False,
            auto_wrap_policies=False,
        ).run(max_activations=200000)
        self.assertEqual(wrapped.completed, base.completed)
        self.assertEqual(wrapped.stop_reason, base.stop_reason)
        self.assertEqual(wrapped.final_values, base.final_values)
        self.assertEqual(wrapped.swap_count, base.swap_count)
        self.assertEqual(wrapped.comparison_count, base.comparison_count)
        self.assertEqual(
            [row["state_hash"] for row in wrapped.trace_rows],
            [row["state_hash"] for row in base.trace_rows],
        )

    def test_nearest_neighbor_blocked_signal_is_local(self) -> None:
        config = SignalConfig("nearest_neighbor", signal_range=1, decay=0.0)
        sim = SignalEventSimulator(
            [2, 1, 3, 4],
            SignalPolicyWrapper(BubblePolicy(), config),
            signal_config=config,
            frozen_positions=[1],
            frozen_variant="stuck",
            scheduler_seed=1,
            tie_breaker_seed=1,
            auto_wrap_policies=False,
        )
        sim.step(forced_cell_id=0, forced_direction=1)
        near = sim.sense_at_position(1)
        far = sim.sense_at_position(3)
        self.assertGreater(near["channels"]["blocked"]["local_sum"], 0.0)
        self.assertEqual(far["channels"]["blocked"]["local_sum"], 0.0)

    def test_diffusion_kernel_expected_values(self) -> None:
        fields = empty_signal_fields(5, ["blocked"])
        fields["blocked"][2] = 1.0
        out = diffuse_signal_fields(fields, SignalConfig("diffusive", diffusion_rate=0.25, decay=0.0))
        np.testing.assert_allclose(out["blocked"], np.asarray([0.0, 0.25, 0.5, 0.25, 0.0]))

    def test_decay_after_diffusion(self) -> None:
        fields = empty_signal_fields(5, ["blocked"])
        fields["blocked"][2] = 1.0
        out = diffuse_signal_fields(fields, SignalConfig("diffusive", diffusion_rate=0.25, decay=0.2))
        np.testing.assert_allclose(out["blocked"], np.asarray([0.0, 0.2, 0.4, 0.2, 0.0]))

    def test_signal_state_is_visible_to_dsl_policy(self) -> None:
        program = parse_rule_program(
            {
                "policy_id": "signal_sensing_demo",
                "name": "signal sensing demo",
                "rules": [
                    {
                        "name": "swap_right_when_blocked_signal_seen",
                        "when": [{"op": "state_compare", "key": "signal_blocked_local_sum", "operator": ">", "value": 0}],
                        "then": {"action": "swap_right"},
                    }
                ],
                "default": {"action": "wait"},
            }
        )
        config = SignalConfig("nearest_neighbor", signal_range=1)
        sim = SignalEventSimulator(
            [3, 1, 2],
            SignalPolicyWrapper(DSLPolicy(program), config),
            signal_config=config,
            scheduler_seed=1,
            tie_breaker_seed=1,
            auto_wrap_policies=False,
        )
        sim.signal_fields["blocked"][1] = 1.0
        outcome = sim.step(forced_cell_id=1)
        self.assertTrue(outcome.swapped)
        self.assertEqual(sim.current_values(), [3, 2, 1])
        actor_state = sim.cells[sim.positions_by_id[1]].state
        self.assertIn(SIGNAL_STATE_KEY, actor_state)
        self.assertGreater(signal_state_for_trace(actor_state)["channels"]["blocked"]["local_sum"], 0.0)

    def test_randomized_control_is_seed_replayable(self) -> None:
        config = SignalConfig("randomized", signal_range=1, random_seed=77)
        first = SignalEventSimulator([3, 1, 2], BubblePolicy(), signal_config=config, scheduler_seed=4, tie_breaker_seed=5)
        second = SignalEventSimulator([3, 1, 2], BubblePolicy(), signal_config=config, scheduler_seed=4, tie_breaker_seed=5)
        first.step(forced_cell_id=1)
        second.step(forced_cell_id=1)
        self.assertEqual(first.trace_rows[-1]["actor_signal_state_json"], second.trace_rows[-1]["actor_signal_state_json"])

    def test_signal_policy_round_trip(self) -> None:
        policy = SignalPolicyWrapper(BubblePolicy(), SignalConfig("diffusive", signal_range=1, decay=0.1))
        restored = signal_policy_from_json(signal_policy_to_json(policy))
        self.assertEqual(restored.to_spec().to_dict(), policy.to_spec().to_dict())

    def test_signal_trace_fields_are_json(self) -> None:
        sim = SignalEventSimulator(
            [3, 1, 2],
            BubblePolicy(),
            signal_config=SignalConfig("diffusive", signal_range=1),
            scheduler_seed=4,
            tie_breaker_seed=5,
        )
        sim.run(max_activations=10)
        self.assertTrue(sim.trace_rows)
        json.loads(sim.trace_rows[-1]["signal_fields_json"])
        json.loads(sim.trace_rows[-1]["signal_config_json"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
