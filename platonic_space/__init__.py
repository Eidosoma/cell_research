"""Utilities for E07 Platonic-space corpus construction."""

from .behavior_corpus import build_behavior_corpus, validate_behavior_corpus
from .behavior_predictor import build_modeling_frame, target_coverage, train_evaluate_predictor
from .counterfactual_tests import (
    score_heldout_predictions,
    select_counterfactual_candidates,
    validate_candidates,
)
from .goal_embeddings import build_goal_profiles, fit_goal_embedding
from .goal_catalog import build_goal_catalog, goal_schema_document, validate_goal_catalog
from .invariant_search import build_invariant_feature_frame, cross_world_holdout_search
from .inverse_design import (
    INVERSE_DESIGN_CLAIM_BOUNDARY,
    INVERSE_DESIGN_MODEL_VERSION,
    INVERSE_DESIGN_SCHEMA_VERSION,
    build_inverse_design_target_profiles,
    generate_inverse_design_pool,
    holdout_panel,
    nearest_existing_policy_comparison,
    policy_outcome_classification,
    reference_policy_table,
    run_design_panel,
    score_candidate_pool,
    select_designed_policies,
    s11_support_table,
    summarize_profile_validation,
    training_panel,
    validation_checks as validate_inverse_design_outputs,
    write_designed_policy_files,
)
from .platonic_distances import build_world_profiles, fit_world_embedding, pairwise_distance_tables
from .policy_embeddings import build_policy_profiles, fit_behavior_embedding, target_error_weights
from .policy_catalog import build_policy_catalog, validate_policy_catalog
from .universality_classes import build_policy_taxonomy_features, build_taxonomy_tables
from .world_schema import (
    SCHEMA_VERSION,
    build_world_catalog,
    catalog_to_dataframe,
    validate_world_catalog,
    world_schema_document,
)

__all__ = [
    "SCHEMA_VERSION",
    "build_behavior_corpus",
    "build_goal_profiles",
    "build_goal_catalog",
    "build_invariant_feature_frame",
    "build_inverse_design_target_profiles",
    "build_modeling_frame",
    "build_policy_profiles",
    "build_policy_catalog",
    "build_policy_taxonomy_features",
    "build_taxonomy_tables",
    "build_world_profiles",
    "build_world_catalog",
    "catalog_to_dataframe",
    "cross_world_holdout_search",
    "generate_inverse_design_pool",
    "fit_goal_embedding",
    "fit_behavior_embedding",
    "fit_world_embedding",
    "pairwise_distance_tables",
    "score_heldout_predictions",
    "select_counterfactual_candidates",
    "goal_schema_document",
    "holdout_panel",
    "nearest_existing_policy_comparison",
    "policy_outcome_classification",
    "reference_policy_table",
    "run_design_panel",
    "score_candidate_pool",
    "select_designed_policies",
    "s11_support_table",
    "summarize_profile_validation",
    "target_coverage",
    "target_error_weights",
    "train_evaluate_predictor",
    "training_panel",
    "validate_candidates",
    "validate_inverse_design_outputs",
    "validate_behavior_corpus",
    "validate_goal_catalog",
    "validate_policy_catalog",
    "validate_world_catalog",
    "world_schema_document",
    "write_designed_policy_files",
    "INVERSE_DESIGN_CLAIM_BOUNDARY",
    "INVERSE_DESIGN_MODEL_VERSION",
    "INVERSE_DESIGN_SCHEMA_VERSION",
]
