"""Design-only contracts for the E07 S07R non-surrogate allocation protocol."""

from .core import (
    TASK_IDS,
    build_allocation_roster,
    build_candidate_selection_frame,
    build_candidate_registry,
    build_scenario_registry,
    candidate_population_commitment,
    checked_protocol,
    derive_rare_status_registry,
    feasibility_summary,
    synthetic_worker_order_validation,
)

__all__ = [
    "TASK_IDS",
    "build_allocation_roster",
    "build_candidate_selection_frame",
    "build_candidate_registry",
    "build_scenario_registry",
    "candidate_population_commitment",
    "checked_protocol",
    "derive_rare_status_registry",
    "feasibility_summary",
    "synthetic_worker_order_validation",
]
