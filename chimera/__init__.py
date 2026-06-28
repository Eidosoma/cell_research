"""Chimeric-policy panel helpers for E06."""

from .panel import (
    PANEL_SCHEMA_VERSION,
    STEP_ID,
    build_algotype_panel,
    instantiate_panel_policy,
    run_s01_algotype_panel,
    run_validation_panel,
)
from .mixtures import (
    MIXTURE_SWEEP_VERSION,
    build_condition_matrix,
    run_s02_mixture_ratios,
    select_priority_policy_ids,
)
from .arrangements import (
    ARRANGEMENT_SWEEP_VERSION,
    ARRANGEMENT_TYPES,
    build_arrangement_condition_matrix,
    generate_arrangement,
    run_s03_initial_arrangements,
    select_s03_base_groups,
)
from .goals import (
    GOAL_MODES,
    GOAL_SWEEP_VERSION,
    build_goal_condition_matrix,
    policy_goal_specs,
    run_s04_goal_compatibility,
    select_s04_base_conditions,
)
from .compatibility import (
    COMPATIBILITY_VECTOR_VERSION,
    compute_compatibility_metrics,
    metric_definitions,
    run_s05_compatibility_metrics,
)
from .dominance import (
    DOMINANCE_SWEEP_VERSION,
    WINNER_CRITERIA_VERSION,
    build_dominance_condition_matrix,
    run_s06_dominance_hierarchy,
    score_dominance_results,
    select_s06_pairs,
)

__all__ = [
    "ARRANGEMENT_SWEEP_VERSION",
    "ARRANGEMENT_TYPES",
    "GOAL_MODES",
    "GOAL_SWEEP_VERSION",
    "COMPATIBILITY_VECTOR_VERSION",
    "DOMINANCE_SWEEP_VERSION",
    "MIXTURE_SWEEP_VERSION",
    "PANEL_SCHEMA_VERSION",
    "STEP_ID",
    "WINNER_CRITERIA_VERSION",
    "build_algotype_panel",
    "build_arrangement_condition_matrix",
    "build_condition_matrix",
    "build_dominance_condition_matrix",
    "build_goal_condition_matrix",
    "compute_compatibility_metrics",
    "generate_arrangement",
    "instantiate_panel_policy",
    "metric_definitions",
    "policy_goal_specs",
    "run_s01_algotype_panel",
    "run_s02_mixture_ratios",
    "run_s03_initial_arrangements",
    "run_s04_goal_compatibility",
    "run_s05_compatibility_metrics",
    "run_s06_dominance_hierarchy",
    "run_validation_panel",
    "score_dominance_results",
    "select_priority_policy_ids",
    "select_s03_base_groups",
    "select_s04_base_conditions",
    "select_s06_pairs",
]
