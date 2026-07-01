"""E04 S01 finite-memory tests."""

from __future__ import annotations

import random
import unittest

from src.e02.deterministic_simulator import EventTracingStatusProbe, SimulatorConfig, _build_cells
from src.e03.policy_interface import OriginalCellPolicyWrapper, cells_signature
from src.e04.memory_policies import (
    BoundedCellMemoryBank,
    CellMemoryConfig,
    memory_policy_for_cell,
    memory_signature,
)


def build_cells(config: SimulatorConfig):
    probe = EventTracingStatusProbe()
    cells, _cell_status = _build_cells(config, probe)
    return cells, probe


def probe_counts(probe: EventTracingStatusProbe) -> tuple[int, int, int]:
    return (
        int(probe.compare_and_swap_count),
        int(probe.swap_count),
        int(probe.frozen_swap_attempts),
    )


class MemoryStateTests(unittest.TestCase):
    def test_successful_swap_updates_last_success_and_neighbor_memory(self) -> None:
        cells, _probe = build_cells(SimulatorConfig(values=(2, 1), algorithm="bubble"))
        bank = BoundedCellMemoryBank(CellMemoryConfig(failed_swap_capacity=3, neighbor_capacity=3))
        actor = cells[0]

        random.seed(1)
        result = memory_policy_for_cell(actor, bank).step(actor, event_step=1)
        state = bank.get(actor.threadID)

        self.assertEqual(result.applied_action.action_type, "swap")
        self.assertTrue(state.last_move_success)
        self.assertEqual(state.time_since_movement, 0)
        self.assertEqual(state.local_frustration, 0)
        self.assertEqual(len(state.recent_failed_swaps), 0)
        self.assertEqual(len(state.recent_neighbor_identities), 1)
        self.assertEqual(state.recent_neighbor_identities[0].side, "right")
        self.assertEqual(state.recent_neighbor_identities[0].thread_id, 2)

    def test_failed_swaps_and_neighbor_identities_are_bounded(self) -> None:
        cells, _probe = build_cells(
            SimulatorConfig(values=(2, 1), algorithm="bubble", frozen_indices=(1,), frozen_semantics="stuck")
        )
        bank = BoundedCellMemoryBank(
            CellMemoryConfig(
                failed_swap_capacity=2,
                neighbor_capacity=3,
                max_time_since_movement=3,
                max_frustration=2,
            )
        )
        actor = cells[0]

        for event_step in range(1, 6):
            random.seed(1)
            memory_policy_for_cell(actor, bank).step(actor, event_step=event_step)

        state = bank.get(actor.threadID)
        self.assertFalse(state.last_move_success)
        self.assertEqual(state.time_since_movement, 3)
        self.assertEqual(state.local_frustration, 2)
        self.assertEqual(len(state.recent_failed_swaps), 2)
        self.assertEqual([entry.event_step for entry in state.recent_failed_swaps], [4, 5])
        self.assertEqual(len(state.recent_neighbor_identities), 3)
        self.assertEqual([entry.event_step for entry in state.recent_neighbor_identities], [3, 4, 5])

    def test_memory_follows_thread_identity_across_position_swaps(self) -> None:
        cells, _probe = build_cells(SimulatorConfig(values=(2, 1, 3), algorithm="bubble"))
        bank = BoundedCellMemoryBank(CellMemoryConfig(neighbor_capacity=4))
        actor = cells[0]
        thread_id = int(actor.threadID)

        random.seed(1)
        memory_policy_for_cell(actor, bank).step(actor, event_step=1)

        self.assertEqual(cells[1].threadID, thread_id)
        self.assertTrue(bank.get(thread_id).last_move_success)
        self.assertEqual(bank.get(thread_id).last_action_type, "swap")

    def test_disabled_memory_leaves_public_step_signature_and_counts_unchanged(self) -> None:
        config = SimulatorConfig(values=(4, 1, 3, 2), algorithm="bubble")
        direct_cells, direct_probe = build_cells(config)
        memory_cells, memory_probe = build_cells(config)

        random.seed(7)
        OriginalCellPolicyWrapper.from_cell(direct_cells[0]).step(direct_cells[0])

        disabled_bank = BoundedCellMemoryBank(CellMemoryConfig(enabled=False))
        random.seed(7)
        memory_policy_for_cell(memory_cells[0], disabled_bank).step(memory_cells[0], event_step=1)

        self.assertEqual(cells_signature(memory_cells), cells_signature(direct_cells))
        self.assertEqual(probe_counts(memory_probe), probe_counts(direct_probe))
        self.assertEqual(disabled_bank.to_dict()["states"], {})

    def test_fixed_seed_sequence_repeats_identical_public_and_memory_signatures(self) -> None:
        def run_sequence() -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[str, object], ...]]:
            cells, _probe = build_cells(SimulatorConfig(values=(4, 1, 3, 2), algorithm="bubble"))
            bank = BoundedCellMemoryBank(CellMemoryConfig(failed_swap_capacity=3, neighbor_capacity=4))
            for event_step, position in enumerate((0, 2, 1, 0, 2, 1), start=1):
                random.seed(100 + event_step)
                actor = cells[position]
                memory_policy_for_cell(actor, bank).step(actor, event_step=event_step)
            return cells_signature(cells), memory_signature(bank)

        first = run_sequence()
        second = run_sequence()
        self.assertEqual(first, second)

    def test_invalid_memory_capacity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CellMemoryConfig(failed_swap_capacity=-1)
        with self.assertRaises(ValueError):
            CellMemoryConfig(neighbor_capacity=-1)


if __name__ == "__main__":
    unittest.main()
