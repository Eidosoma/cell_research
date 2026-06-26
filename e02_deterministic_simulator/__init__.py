"""Deterministic event simulator for E02 scheduler and null-model audits."""

from .metrics import (
    aggregation,
    initial_values_from_seed,
    monotonicity_error,
    sortedness_percent,
    sortedness_raw,
    state_hash,
)
from .simulator import (
    Cell,
    DeterministicEventSimulator,
    SimulationResult,
    StepOutcome,
)

__all__ = [
    "Cell",
    "DeterministicEventSimulator",
    "SimulationResult",
    "StepOutcome",
    "aggregation",
    "initial_values_from_seed",
    "monotonicity_error",
    "sortedness_percent",
    "sortedness_raw",
    "state_hash",
]
