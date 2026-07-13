"""Deterministic clean-room reference simulator for Experiment E01."""

from .api import (
    REFERENCE_SEMANTICS_VERSION,
    create_scenario,
    exact_replay,
    run_many,
    run_scenario,
)
from .model import Architecture, Cell, Direction, FaultMode, Policy, Scenario

__all__ = [
    "Architecture",
    "Cell",
    "Direction",
    "FaultMode",
    "Policy",
    "REFERENCE_SEMANTICS_VERSION",
    "Scenario",
    "create_scenario",
    "exact_replay",
    "run_many",
    "run_scenario",
]
