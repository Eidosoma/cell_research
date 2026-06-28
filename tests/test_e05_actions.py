from __future__ import annotations

import json
import unittest

from morphospace2d import (
    ACTION_SCHEMA_VERSION,
    ActionProposal,
    build_s04_validation_world,
    action_semantics_spec,
    audit_local_action_payload,
    run_standard_action_sequence,
    validate_action_trace_schema,
)


class TestE05ActionSemantics(unittest.TestCase):
    def test_action_semantics_spec_covers_required_actions(self) -> None:
        spec = action_semantics_spec()
        self.assertEqual(spec["schemaVersion"], ACTION_SCHEMA_VERSION)
        required = {"swap", "crawl", "rotate", "divide", "die", "adhere", "detach", "exchange_signal"}
        self.assertTrue(required.issubset(set(spec["supportedActions"])))
        self.assertGreater(spec["energyCosts"]["divide"], spec["energyCosts"]["swap"])
        self.assertEqual(spec["conservationModes"]["divide"], "non_conservative_birth")

    def test_swap_and_crawl_preserve_cell_count_and_handle_collisions(self) -> None:
        world = build_s04_validation_world()
        initial_cell_count = len(world.cells)
        swap = world.apply_action(ActionProposal.swap(0, (1, 0), "swap_test"))
        collision = world.apply_action(ActionProposal.crawl(0, (0, 0), "collision_test"))
        crawl = world.apply_action(ActionProposal.crawl(0, (1, 1), "crawl_test"))

        self.assertTrue(swap.accepted)
        self.assertEqual(swap.cell_delta, 0)
        self.assertFalse(collision.accepted)
        self.assertFalse(collision.legal)
        self.assertTrue(collision.collision)
        self.assertEqual(collision.reason, "crawl_target_occupied")
        self.assertTrue(crawl.accepted)
        self.assertEqual(len(world.cells), initial_cell_count)
        self.assertEqual(world.validate_state(), [])

    def test_rotate_divide_adhere_signal_detach_and_die_are_trace_consistent(self) -> None:
        world, outcomes, validation_rows = run_standard_action_sequence()
        by_action = {outcome.action: outcome for outcome in outcomes if outcome.accepted}

        self.assertEqual(world.validate_state(), [])
        self.assertEqual(validate_action_trace_schema(world.trace_rows), [])
        self.assertTrue(all(row["state_valid_after"] for row in validation_rows))
        self.assertEqual(by_action["rotate"].energy_cost, 0.25)
        self.assertEqual(by_action["divide"].cell_delta, 1)
        self.assertIsNotNone(by_action["divide"].born_cell_id)
        self.assertEqual(by_action["adhere"].adhesion_bond_delta, 1)
        self.assertEqual(by_action["detach"].detached_delta, 1)
        self.assertEqual(by_action["detach"].occupied_delta, -1)
        self.assertEqual(by_action["die"].cell_delta, -1)
        self.assertEqual(by_action["die"].detached_delta, -1)
        self.assertGreater(world.energy_spent, 0.0)

    def test_rotate_updates_only_local_actor_polarity(self) -> None:
        world = build_s04_validation_world()
        before = dict(world.cells[0].identity)
        outcome = world.apply_action(ActionProposal.rotate(0, quarter_turns=1))
        after = dict(world.cells[0].identity)
        self.assertTrue(outcome.accepted)
        self.assertEqual(before["polarity"], [1.0, 0.0])
        self.assertEqual(after["polarity"], [-0.0, 1.0])
        self.assertEqual(after["scalar_value"], before["scalar_value"])

    def test_signal_exchange_mutates_only_adjacent_receiver_signal_state(self) -> None:
        world = build_s04_validation_world()
        world.apply_action(ActionProposal.swap(0, (1, 0)))
        world.apply_action(ActionProposal.crawl(0, (1, 1)))
        division = world.apply_action(ActionProposal.divide(0, (2, 1), child_value=5))
        self.assertEqual(division.born_cell_id, 4)
        signal = world.apply_action(ActionProposal.exchange_signal(0, (2, 1), channel="morphogen", amount=0.75))
        child_identity = world.cells[4].identity
        actor_identity = world.cells[0].identity
        self.assertTrue(signal.accepted)
        self.assertEqual(child_identity["local_signals"]["morphogen"]["received"], 0.75)
        self.assertEqual(actor_identity["local_signals_emitted"]["morphogen"], 0.75)

    def test_local_action_payload_does_not_expose_global_state(self) -> None:
        world = build_s04_validation_world()
        payload = world.local_observation(0).to_policy_payload()
        self.assertEqual(audit_local_action_payload(payload), [])
        self.assertNotIn("occupancy", json.dumps(payload))
        self.assertNotIn("trace_rows", json.dumps(payload))
        self.assertIn("neighbors", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
