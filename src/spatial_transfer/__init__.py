"""Outcome-independent spatial-transfer compatibility contracts."""

from .qualification import (
    CALIBRATED_TARGETS,
    DIAGNOSTIC_FIXTURES,
    PANEL_IDS,
    SPATIAL_TASKS,
    DiagnosticSpatialTracker,
    apply_adaptation_variant,
    build_panel_fixture,
    canonical_binding_digest,
    endpoint_contract,
    qualify_configuration_binding,
    qualify_endpoint,
    run_fail_atomic_fixture_batch,
)

__all__ = [
    "CALIBRATED_TARGETS",
    "DIAGNOSTIC_FIXTURES",
    "PANEL_IDS",
    "SPATIAL_TASKS",
    "DiagnosticSpatialTracker",
    "apply_adaptation_variant",
    "build_panel_fixture",
    "canonical_binding_digest",
    "endpoint_contract",
    "qualify_configuration_binding",
    "qualify_endpoint",
    "run_fail_atomic_fixture_batch",
]
