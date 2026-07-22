"""Train-only surrogate and event-summary representation tools for E07 S06."""

from .core import (
    TASK_IDS,
    assign_grouped_splits,
    build_s05_freeze,
    lineage_component,
    verify_frozen_inputs,
)

__all__ = [
    "TASK_IDS",
    "assign_grouped_splits",
    "build_s05_freeze",
    "lineage_component",
    "verify_frozen_inputs",
]
