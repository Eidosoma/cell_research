"""Frozen S04 objective, descriptor, novelty, and eligibility contracts."""

from .core import (
    S04_SCHEMA_VERSION,
    bin_index,
    build_s04_evidence,
    load_yaml,
    require_s05_eligible,
    validate_registries,
)

__all__ = [
    "S04_SCHEMA_VERSION",
    "bin_index",
    "build_s04_evidence",
    "load_yaml",
    "require_s05_eligible",
    "validate_registries",
]
