"""Two-dimensional relational morphology benchmark specifications."""

from .targets import (
    CATALOG_VERSION,
    TargetDefinition,
    adjacent_swap_witness,
    apply_swap_witness,
    deterministic_scramble,
    evaluate_success,
    exact_equivalence_orbit,
    load_target_catalog,
    validate_target_catalog,
)

__all__ = [
    "CATALOG_VERSION",
    "TargetDefinition",
    "adjacent_swap_witness",
    "apply_swap_witness",
    "deterministic_scramble",
    "evaluate_success",
    "exact_equivalence_orbit",
    "load_target_catalog",
    "validate_target_catalog",
]
