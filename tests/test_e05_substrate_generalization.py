from __future__ import annotations

import json
import unittest

from morphospace2d import (
    CellState,
    ClockwiseCrawlPolicy,
    GreedyLowerValueSwapPolicy,
    MorphologyWorld,
    MoveProposal,
    Substrate,
    run_local_dynamics,
    validate_trace_schema,
)


class TestE05SubstrateGeneralization(unittest.TestCase):
    def assert_valid_substrate(self, substrate: Substrate) -> None:
        self.assertEqual(substrate.validate(), [])
        for left, right in substrate.edges():
            self.assertIn(right, substrate.neighbors(left))
            self.assertIn(left, substrate.neighbors(right))
            self.assertNotEqual(left, right)

    def test_required_substrates_have_consistent_undirected_adjacency(self) -> None:
        substrates = [
            Substrate.row(4),
            Substrate.square_grid(3, 2),
            Substrate.hex_grid(3, 2),
            Substrate.irregular_graph([((0,), (1,)), ((1,), (2,)), ((2,), (3,))]),
        ]
        for substrate in substrates:
            with self.subTest(substrate=substrate.substrate_type):
                self.assert_valid_substrate(substrate)
                self.assertGreaterEqual(len(substrate.nodes), 4)
                self.assertGreater(len(substrate.edges()), 0)

    def test_row_preserves_adjacent_swap_semantics(self) -> None:
        substrate = Substrate.row(3)
        world = MorphologyWorld.from_position_values(substrate, {(0,): 2, (1,): 1, (2,): 3})
        outcome = world.apply_move(MoveProposal.swap(0, (1,), "test_swap"))
        self.assertTrue(outcome.accepted)
        self.assertTrue(outcome.legal)
        self.assertEqual(world.position_values(), {(0,): 1, (1,): 2, (2,): 3})
        self.assertEqual(world.value_counter()[1], 1)
        self.assertEqual(world.value_counter()[2], 1)

    def test_reversible_toy_swap_conserves_occupancy(self) -> None:
        substrate = Substrate.square_grid(2, 1)
        world = MorphologyWorld.from_position_values(substrate, {(0, 0): "A", (1, 0): "B"})
        initial_positions = dict(world.occupancy)
        forward = world.apply_move(MoveProposal.swap(0, (1, 0), "forward"))
        backward = world.apply_move(MoveProposal.swap(0, (0, 0), "backward"))
        self.assertTrue(forward.accepted)
        self.assertTrue(backward.accepted)
        self.assertEqual(world.occupancy, initial_positions)
        self.assertEqual(world.validate_occupancy(), [])

    def test_crawl_requires_adjacent_empty_target(self) -> None:
        substrate = Substrate.irregular_graph(
            [((0,), (1,)), ((1,), (2,))],
            nodes=[(0,), (1,), (2,)],
        )
        world = MorphologyWorld(
            substrate,
            {0: CellState(0, "A"), 1: CellState(1, "B")},
            {(0,): 0, (1,): 1},
        )
        blocked = world.apply_move(MoveProposal.crawl(0, (1,), "occupied"))
        accepted = world.apply_move(MoveProposal.crawl(1, (2,), "empty"))
        self.assertFalse(blocked.accepted)
        self.assertFalse(blocked.legal)
        self.assertTrue(accepted.accepted)
        self.assertEqual(world.cell_position(1), (2,))
        self.assertEqual(world.occupied_count(), 2)

    def test_occupancy_validation_rejects_detached_cells(self) -> None:
        substrate = Substrate.row(2)
        with self.assertRaises(ValueError):
            MorphologyWorld(
                substrate,
                {0: CellState(0, "A"), 1: CellState(1, "B")},
                {(0,): 0},
            )

    def test_local_observation_exposes_neighbors_without_whole_world_state(self) -> None:
        substrate = Substrate.hex_grid(3, 2)
        world = MorphologyWorld.from_position_values(
            substrate,
            {(0, 0): 4, (1, 0): 1, (2, 0): 3, (0, 1): 2},
        )
        observation = world.local_observation(0)
        self.assertEqual(observation.actor.value, 4)
        self.assertGreater(len(observation.neighbors), 0)
        self.assertLessEqual(len(observation.neighbors), 6)
        self.assertFalse(hasattr(observation, "occupancy"))
        self.assertFalse(hasattr(observation, "target_morphology"))

    def test_policy_runs_on_row_square_hex_and_irregular_graph(self) -> None:
        cases = [
            MorphologyWorld.from_position_values(Substrate.row(4), {(0,): 4, (1,): 1, (2,): 3, (3,): 2}),
            MorphologyWorld.from_position_values(
                Substrate.square_grid(2, 2),
                {(0, 0): 4, (1, 0): 1, (0, 1): 3, (1, 1): 2},
            ),
            MorphologyWorld.from_position_values(
                Substrate.hex_grid(2, 2),
                {(0, 0): 4, (1, 0): 1, (0, 1): 3, (1, 1): 2},
            ),
            MorphologyWorld.from_position_values(
                Substrate.irregular_graph([((0,), (1,)), ((1,), (2,)), ((2,), (3,))]),
                {(0,): 4, (1,): 1, (2,): 3, (3,): 2},
            ),
        ]
        for world in cases:
            with self.subTest(substrate=world.substrate.substrate_type):
                initial_counter = world.cell_id_counter()
                outcomes = run_local_dynamics(world, GreedyLowerValueSwapPolicy(), steps=6)
                self.assertTrue(all(outcome.legal for outcome in outcomes))
                self.assertEqual(world.cell_id_counter(), initial_counter)
                self.assertEqual(world.validate_occupancy(), [])
                self.assertGreater(sum(outcome.accepted and outcome.action == "swap" for outcome in outcomes), 0)

    def test_crawl_policy_uses_empty_irregular_node(self) -> None:
        substrate = Substrate.irregular_graph(
            [((0,), (1,)), ((1,), (2,)), ((2,), (3,))],
            nodes=[(0,), (1,), (2,), (3,)],
        )
        world = MorphologyWorld(
            substrate,
            {0: CellState(0, "A"), 1: CellState(1, "B"), 2: CellState(2, "C")},
            {(0,): 0, (1,): 1, (2,): 2},
        )
        outcomes = run_local_dynamics(world, ClockwiseCrawlPolicy(), steps=3, actor_order=[2])
        self.assertTrue(any(outcome.accepted and outcome.action == "crawl" for outcome in outcomes))
        self.assertEqual(world.occupied_count(), 3)
        self.assertEqual(world.validate_occupancy(), [])

    def test_trace_schema_is_json_serializable(self) -> None:
        substrate = Substrate.row(2)
        world = MorphologyWorld.from_position_values(substrate, {(0,): 2, (1,): 1}, condition_id="trace_test")
        world.apply_move(MoveProposal.swap(0, (1,), "trace_swap"))
        self.assertEqual(validate_trace_schema(world.trace_rows), [])
        json.dumps(world.trace_rows, sort_keys=True)
        self.assertEqual(world.trace_rows[0]["event_kind"], "initial")
        self.assertEqual(world.trace_rows[-1]["action"], "swap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
