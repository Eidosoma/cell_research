"""E04 S02 local and diffusive signaling tests."""

from __future__ import annotations

import random
import unittest

from src.e02.deterministic_simulator import EventTracingStatusProbe, SimulatorConfig, _build_cells
from src.e03.policy_interface import OriginalCellPolicyWrapper, cells_signature
from src.e04.memory_policies import BoundedCellMemoryBank, CellMemoryConfig, memory_signature
from src.e04.signaling import (
    LocalSignalValues,
    SignalBank,
    SignalConfig,
    signal_policy_for_cell,
    signal_signature,
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


class SignalingTests(unittest.TestCase):
    def test_blocked_signal_emission_is_local_and_audited(self) -> None:
        cells, _probe = build_cells(
            SimulatorConfig(values=(2, 1, 3), algorithm="bubble", frozen_indices=(1,), frozen_semantics="stuck")
        )
        memory_bank = BoundedCellMemoryBank(CellMemoryConfig(max_frustration=3))
        signal_bank = SignalBank(len(cells), SignalConfig(local_radius=1, diffusion_enabled=True, decay=0.0))

        random.seed(1)
        result = signal_policy_for_cell(cells[0], signal_bank, memory_bank).step(cells[0], event_step=1)
        emission = result.emission_after

        self.assertIsNotNone(emission)
        self.assertEqual(emission.values.blocked, 1.0)
        self.assertGreater(emission.values.frustrated, 0.0)
        self.assertEqual(set(emission.accessed_indices), {0, 1})
        audit = signal_bank.audit_no_global_oracle()
        self.assertFalse(audit["usesGlobalOracle"])
        self.assertTrue(audit["diffusiveFieldsExplicitlyLabeled"])

    def test_diffusion_spreads_scalar_field_deterministically(self) -> None:
        bank = SignalBank(5, SignalConfig(diffusion_rate=0.25, decay=0.0))
        bank.deposit_field(2, LocalSignalValues(blocked=1.0))
        bank.diffuse_once()

        self.assertEqual(bank.fields["blocked"], [0.0, 0.25, 0.5, 0.25, 0.0])
        self.assertEqual(bank.diffusion_step_count, 1)

    def test_noisy_sensing_is_reproducible_for_fixed_seed(self) -> None:
        def sense_digest(seed: int):
            bank = SignalBank(3, SignalConfig(noise_std=0.05, noise_seed=seed))
            bank.deposit_field(1, LocalSignalValues(morphogen=1.0))
            cells, _probe = build_cells(SimulatorConfig(values=(2, 1, 3), algorithm="bubble"))
            observation = OriginalCellPolicyWrapper.from_cell(cells[1]).observe(cells[1])
            return bank.sense(observation, event_step=1).to_dict()

        first = sense_digest(42)
        second = sense_digest(42)
        different = sense_digest(43)

        self.assertEqual(first, second)
        self.assertNotEqual(first["diffusive_fields"], different["diffusive_fields"])
        self.assertTrue(first["noise_applied"])

    def test_disabled_signals_match_memory_wrapper_public_and_memory_state(self) -> None:
        config = SimulatorConfig(values=(4, 1, 3, 2), algorithm="bubble")
        memory_cells, memory_probe = build_cells(config)
        signal_cells, signal_probe = build_cells(config)
        memory_bank = BoundedCellMemoryBank(CellMemoryConfig())
        signal_memory_bank = BoundedCellMemoryBank(CellMemoryConfig())
        disabled_signal_bank = SignalBank(len(signal_cells), SignalConfig(enabled=False))

        random.seed(7)
        from src.e04.memory_policies import memory_policy_for_cell

        memory_policy_for_cell(memory_cells[0], memory_bank).step(memory_cells[0], event_step=1)
        random.seed(7)
        signal_policy_for_cell(signal_cells[0], disabled_signal_bank, signal_memory_bank).step(signal_cells[0], event_step=1)

        self.assertEqual(cells_signature(signal_cells), cells_signature(memory_cells))
        self.assertEqual(probe_counts(signal_probe), probe_counts(memory_probe))
        self.assertEqual(memory_signature(signal_memory_bank), memory_signature(memory_bank))
        self.assertEqual(disabled_signal_bank.emission_log, [])
        self.assertEqual(disabled_signal_bank.sensation_log, [])

    def test_sensing_respects_local_radius(self) -> None:
        bank = SignalBank(5, SignalConfig(local_radius=1, diffusion_enabled=False))
        for position in range(5):
            bank.set_local_signal(position, LocalSignalValues(blocked=float(position)))
        cells, _probe = build_cells(SimulatorConfig(values=(5, 4, 3, 2, 1), algorithm="bubble"))
        observation = OriginalCellPolicyWrapper.from_cell(cells[2]).observe(cells[2])
        sensation = bank.sense(observation, event_step=3)

        self.assertEqual([position for position, _values in sensation.local_signals], [1, 2, 3])
        self.assertEqual(set(sensation.accessed_indices), {1, 2, 3})
        self.assertEqual(sensation.access_scope, "local_window")

    def test_fixed_seed_sequence_repeats_signal_signature(self) -> None:
        def run_sequence():
            cells, _probe = build_cells(SimulatorConfig(values=(4, 1, 3, 2), algorithm="bubble"))
            memory_bank = BoundedCellMemoryBank(CellMemoryConfig())
            signal_bank = SignalBank(len(cells), SignalConfig(noise_seed=11))
            for event_step, position in enumerate((0, 2, 1, 0), start=1):
                random.seed(200 + event_step)
                signal_policy_for_cell(cells[position], signal_bank, memory_bank).step(cells[position], event_step=event_step)
            return cells_signature(cells), memory_signature(memory_bank), signal_signature(signal_bank)

        self.assertEqual(run_sequence(), run_sequence())

    def test_invalid_signal_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SignalConfig(local_radius=-1)
        with self.assertRaises(ValueError):
            SignalConfig(diffusion_rate=0.75)
        with self.assertRaises(ValueError):
            SignalConfig(decay=1.5)
        with self.assertRaises(ValueError):
            SignalConfig(noise_std=-0.1)


if __name__ == "__main__":
    unittest.main()
