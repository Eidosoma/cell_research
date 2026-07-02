"""E05 S01 substrate generalization tests."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e02.deterministic_simulator import EventTracingStatusProbe, SimulatorConfig, _build_cells
from src.e03.policy_interface import OriginalCellPolicyWrapper
from src.e05.substrates import (
    SubstrateCell,
    array_substrate,
    graph_substrate,
    hex_grid_substrate,
    lattice3d_substrate,
    make_cells,
    square_grid_substrate,
)


def build_reference_cells(config: SimulatorConfig):
    probe = EventTracingStatusProbe()
    cells, _cell_status = _build_cells(config, probe)
    return cells, probe


class SubstrateTests(unittest.TestCase):
    def test_array_neighbors_boundaries_and_swap(self) -> None:
        substrate = array_substrate(4)
        substrate.fill_sites(make_cells((2, 1, 3, 4), label="bubble"))

        self.assertEqual(substrate.neighbors(0), (1,))
        self.assertEqual(substrate.neighbors(1), (0, 2))
        self.assertTrue(substrate.site(0).boundary)
        self.assertFalse(substrate.site(1).boundary)

        observation = substrate.local_observation(0)
        self.assertEqual(observation.actor_cell.value, 2)
        self.assertEqual(observation.neighbor_site_ids, (1,))
        self.assertEqual(observation.neighbors[0].direction, "right")
        self.assertEqual(observation.neighbors[0].value, 1)

        result = substrate.apply_action("swap", 0, 1)
        self.assertTrue(result.allowed)
        self.assertTrue(result.state_changed)
        self.assertEqual(substrate.values_in_site_order(), (1, 2, 3, 4))

    def test_move_requires_adjacent_empty_target(self) -> None:
        substrate = array_substrate(3)
        substrate.place_cell(0, SubstrateCell("a", 1))
        substrate.place_cell(1, SubstrateCell("b", 2))

        occupied_result = substrate.apply_action("move", 0, 1)
        self.assertFalse(occupied_result.allowed)
        self.assertEqual(occupied_result.reason, "target_not_empty")

        nonadjacent_result = substrate.apply_action("move", 0, 2)
        self.assertFalse(nonadjacent_result.allowed)
        self.assertEqual(nonadjacent_result.reason, "target_not_adjacent")

        allowed_result = substrate.apply_action("move", 1, 2)
        self.assertTrue(allowed_result.allowed)
        self.assertEqual(substrate.values_in_site_order(), (1, None, 2))
        self.assertEqual(substrate.site_of_cell("b"), 2)

    def test_status_constraints_match_local_action_semantics(self) -> None:
        passive = array_substrate(2)
        passive.place_cell(0, SubstrateCell("actor", 2, status="ACTIVE"))
        passive.place_cell(1, SubstrateCell("target", 1, status="FREEZE"))
        self.assertTrue(passive.apply_action("swap", 0, 1).allowed)

        stuck = array_substrate(2)
        stuck.place_cell(0, SubstrateCell("actor", 2, status="ACTIVE"))
        stuck.place_cell(1, SubstrateCell("target", 1, status="stuck"))
        result = stuck.apply_action("swap", 0, 1)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "target_not_movable_or_missing")

        frozen_actor = array_substrate(2)
        frozen_actor.place_cell(0, SubstrateCell("actor", 2, status="FREEZE"))
        frozen_actor.place_cell(1, SubstrateCell("target", 1, status="ACTIVE"))
        result = frozen_actor.apply_action("swap", 0, 1)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "actor_not_active_or_missing")

    def test_square_grid_neighbor_counts_and_illegal_diagonal(self) -> None:
        substrate = square_grid_substrate(3, 3)
        self.assertEqual(substrate.neighbor_count(4), 4)
        self.assertEqual(substrate.neighbor_count(0), 2)
        self.assertEqual(substrate.neighbors(4), (1, 5, 7, 3))
        self.assertTrue(substrate.site(0).boundary)
        self.assertFalse(substrate.site(4).boundary)

        substrate.place_cell(0, SubstrateCell("a", 1))
        substrate.place_cell(4, SubstrateCell("b", 2))
        result = substrate.apply_action("swap", 0, 4)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "target_not_adjacent")

    def test_hex_grid_uses_six_neighbor_axial_center(self) -> None:
        substrate = hex_grid_substrate(3, 3)
        self.assertEqual(substrate.neighbor_count(4), 6)
        self.assertEqual(substrate.neighbor_count(0), 2)
        self.assertEqual(
            tuple(substrate.direction_between(4, target) for target in substrate.neighbors(4)),
            ("east", "north_east", "north_west", "west", "south_west", "south_east"),
        )

    def test_irregular_graph_validation_and_neighbors(self) -> None:
        substrate = graph_substrate({0: [1], 1: [0, 2, 3], 2: [1], 3: [1]})
        self.assertEqual(substrate.neighbors(1), (0, 2, 3))
        self.assertEqual(substrate.neighbor_count(0), 1)
        self.assertTrue(substrate.site(0).boundary)

        with self.assertRaises(ValueError):
            graph_substrate({0: [1], 1: []})

    def test_lattice3d_optional_substrate_neighbor_counts(self) -> None:
        substrate = lattice3d_substrate(3, 3, 3)
        self.assertEqual(substrate.neighbor_count(13), 6)
        self.assertEqual(substrate.neighbor_count(0), 3)
        self.assertEqual(substrate.direction_between(13, 22), "up")

    def test_legal_actions_are_dry_runs(self) -> None:
        substrate = array_substrate(3)
        substrate.place_cell(0, SubstrateCell("a", 2))
        substrate.place_cell(1, SubstrateCell("b", 1))
        before = substrate.signature()

        actions = substrate.legal_actions(0)

        self.assertEqual(substrate.signature(), before)
        self.assertEqual([action.action_type for action in actions], ["wait", "swap", "move"])
        self.assertTrue(actions[1].allowed)
        self.assertFalse(actions[2].allowed)

    def test_1d_substrate_applies_e03_bubble_reference_action(self) -> None:
        config = SimulatorConfig(values=(2, 1), algorithm="bubble")
        reference_cells, _reference_probe = build_reference_cells(config)
        policy_cells, _policy_probe = build_reference_cells(config)
        substrate = array_substrate(2)
        substrate.fill_sites(
            tuple(
                SubstrateCell(str(cell.threadID), cell.value, label=cell.label, status=str(cell.status))
                for cell in policy_cells
            )
        )

        wrapper = OriginalCellPolicyWrapper.from_cell(policy_cells[0])
        observation = wrapper.observe(policy_cells[0])
        local_observation = substrate.local_observation(0)
        self.assertEqual(local_observation.actor_cell.value, observation.actor_value)
        self.assertEqual(local_observation.neighbor_site_ids, (1,))
        self.assertEqual(local_observation.neighbors[0].value, observation.values[1])

        random.seed(1)
        action = wrapper.propose_action(observation)
        result = substrate.apply_indexed_policy_action(0, action)
        self.assertTrue(result.allowed)

        random.seed(1)
        reference_cells[0].move()
        self.assertEqual(substrate.values_in_site_order(), tuple(cell.value for cell in reference_cells))


if __name__ == "__main__":
    unittest.main()
