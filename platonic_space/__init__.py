"""Utilities for E07 Platonic-space corpus construction."""

from .behavior_corpus import build_behavior_corpus, validate_behavior_corpus
from .behavior_predictor import build_modeling_frame, target_coverage, train_evaluate_predictor
from .goal_embeddings import build_goal_profiles, fit_goal_embedding
from .goal_catalog import build_goal_catalog, goal_schema_document, validate_goal_catalog
from .platonic_distances import build_world_profiles, fit_world_embedding, pairwise_distance_tables
from .policy_embeddings import build_policy_profiles, fit_behavior_embedding, target_error_weights
from .policy_catalog import build_policy_catalog, validate_policy_catalog
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
    "build_modeling_frame",
    "build_policy_profiles",
    "build_policy_catalog",
    "build_world_profiles",
    "build_world_catalog",
    "catalog_to_dataframe",
    "fit_goal_embedding",
    "fit_behavior_embedding",
    "fit_world_embedding",
    "pairwise_distance_tables",
    "goal_schema_document",
    "target_coverage",
    "target_error_weights",
    "train_evaluate_predictor",
    "validate_behavior_corpus",
    "validate_goal_catalog",
    "validate_policy_catalog",
    "validate_world_catalog",
    "world_schema_document",
]
