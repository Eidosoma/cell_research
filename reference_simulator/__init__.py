"""Deterministic clean-room reference simulator for Experiment E01."""

from .api import (
    REFERENCE_SEMANTICS_VERSION,
    create_scenario,
    exact_replay,
    run_many,
    run_scenario,
)
from .invariants import AUDIT_VERSION, InvariantViolation, audit_reference_result
from .model import Architecture, Cell, Direction, FaultMode, Policy, Scenario

__all__ = [
    "Architecture",
    "AUDIT_VERSION",
    "Cell",
    "Direction",
    "FaultMode",
    "InvariantViolation",
    "Policy",
    "REFERENCE_SEMANTICS_VERSION",
    "Scenario",
    "audit_reference_result",
    "create_scenario",
    "exact_replay",
    "run_many",
    "run_scenario",
]
