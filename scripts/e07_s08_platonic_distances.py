#!/usr/bin/env python3
"""Define and validate E07 S08 empirical Platonic distances."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True
for thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_var, "1")

import joblib
import numpy as np
import pandas as pd
import sklearn

from platonic_space.behavior_predictor import DEFAULT_TARGET_COLUMNS
from platonic_space.platonic_distances import (
    PLATONIC_DISTANCE_CLAIM_BOUNDARY,
    PLATONIC_DISTANCE_MODEL_VERSION,
    PLATONIC_DISTANCE_SCHEMA_VERSION,
    SPARSE_COVERAGE_PENALTY_WEIGHT,
    build_world_profiles,
    distance_baseline_comparison,
    entity_table,
    fit_world_embedding,
    goal_sparse_penalty,
    nearest_neighbor_sanity,
    nearest_neighbor_table,
    pairwise_distance_tables,
    policy_sparse_penalty,
    prior_neighbor_overlap,
    symmetry_triangle_diagnostics,
    target_error_weights,
    validate_platonic_distances,
    world_sparse_penalty,
)
from platonic_space.world_schema import sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S08"
STEP_NUMBER = 8
STEP_TITLE = "Define Platonic distance"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_FRAME_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "modeling_frame.parquet"
DEFAULT_WORLD_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S01" / "normalized_world_catalog.parquet"
DEFAULT_ERROR_LIMITS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S05" / "error_limits.parquet"
DEFAULT_S05_MODEL_CARD_PATH = DEFAULT_ARTIFACTS_DIR / "models" / "e07_behavior_predictor" / "model_card.json"
DEFAULT_POLICY_EMBEDDINGS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S06" / "policy_embeddings.parquet"
DEFAULT_POLICY_COVERAGE_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S06" / "policy_embedding_coverage.parquet"
DEFAULT_POLICY_NEIGHBORS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S06" / "nearest_neighbors.parquet"
DEFAULT_GOAL_EMBEDDINGS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S07" / "goal_embeddings.parquet"
DEFAULT_GOAL_UNCERTAINTY_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S07" / "goal_uncertainty.parquet"
DEFAULT_GOAL_NEIGHBORS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S07" / "nearest_neighbors.parquet"
FOCUSED_TESTS = ["tests.test_e07_platonic_distances"]


POLICY_LABEL_COLUMNS = ("policyLabel", "sourceExperimentId", "policyFamily", "policyKind", "representationType", "stochasticity")
GOAL_LABEL_COLUMNS = ("goalLabel", "goalFamily", "goalKind", "representationType", "targetDimensionality", "sparseUncertaintyLevel")
WORLD_LABEL_COLUMNS = ("title", "experimentId", "worldFamily", "substrateClass", "replayability", "metadataCompleteness")
BASELINE_LABEL_COLUMNS = {
    "policy": ("sourceExperimentId", "policyFamily", "policyKind", "representationType"),
    "goal": ("goalFamily", "goalKind", "representationType", "targetDimensionality"),
    "world": ("experimentId", "worldFamily", "substrateClass", "replayability"),
}


def run_command(command: Sequence[str], cwd: Path = REPO_ROOT) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(thread_var, "1")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    return {
        "command": list(command),
        "cwd": str(cwd),
        "returncode": int(completed.returncode),
        "success": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsedSeconds": round(time.perf_counter() - started, 6),
    }


def git_value(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def write_dataframe(df: pd.DataFrame, stem: Path, csv: bool = True) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if csv:
        csv_path = stem.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        paths.append(csv_path)
    parquet_path = stem.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)
    paths.append(parquet_path)
    return paths


def collect_artifacts(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(out, key=lambda row: row["path"])


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("\n", " ").replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def dataframe_markdown(df: pd.DataFrame, limit: int = 60) -> str:
    if df.empty:
        return "No rows."
    display = df.head(limit).copy()
    return markdown_table(display.astype(str).to_dict(orient="records"), [str(column) for column in display.columns])


def target_columns_from_model_card(path: Path) -> list[str]:
    if not path.exists():
        return list(DEFAULT_TARGET_COLUMNS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = payload.get("targetColumns") or list(DEFAULT_TARGET_COLUMNS)
    return [str(column) for column in columns]


def performance_summary(
    entities: pd.DataFrame,
    distances: pd.DataFrame,
    diagnostics: pd.DataFrame,
    sanity: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
) -> dict[str, Any]:
    counts = entities["entityType"].value_counts().to_dict() if not entities.empty else {}
    triangle_violations = int(diagnostics["triangleViolationCount"].sum()) if not diagnostics.empty else 0
    max_triangle = float(diagnostics["maxTriangleViolation"].max()) if not diagnostics.empty else 0.0
    max_symmetry = float(diagnostics["maxSymmetryAbsError"].max()) if not diagnostics.empty else 0.0
    overlap = sanity[sanity["validationTask"].eq("prior_embedding_neighbor_overlap")] if not sanity.empty else pd.DataFrame()

    def overlap_value(entity_type: str, k: int) -> float:
        row = overlap[(overlap["entityType"].eq(entity_type)) & (overlap["k"].eq(k))]
        if row.empty or "meanJaccard" not in row.columns:
            return float("nan")
        return float(row["meanJaccard"].iloc[0])

    closer = baseline_comparison[pd.to_numeric(baseline_comparison.get("sameLabelCloserDelta", pd.Series(dtype=float)), errors="coerce") > 0]
    high_uncertainty = entities[entities["distanceUncertaintyLevel"].eq("high")] if "distanceUncertaintyLevel" in entities.columns else pd.DataFrame()
    return {
        "policyEntityCount": int(counts.get("policy", 0)),
        "goalEntityCount": int(counts.get("goal", 0)),
        "worldEntityCount": int(counts.get("world", 0)),
        "distancePairCount": int(len(distances)),
        "triangleViolationCount": triangle_violations,
        "maxTriangleViolation": max_triangle,
        "maxSymmetryAbsError": max_symmetry,
        "policyPriorMeanJaccardK10": overlap_value("policy", 10),
        "goalPriorMeanJaccardK10": overlap_value("goal", 10),
        "metadataLabelsWithSameLabelCloser": int(len(closer)),
        "metadataLabelComparisonCount": int(len(baseline_comparison)),
        "highUncertaintyEntityCount": int(len(high_uncertainty)),
        "sparseCoveragePenaltyWeight": float(SPARSE_COVERAGE_PENALTY_WEIGHT),
    }


def definition_report_markdown(
    path: Path,
    status: Mapping[str, Any],
    perf: Mapping[str, Any],
    diagnostics: pd.DataFrame,
    sanity: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
) -> None:
    label_sanity = sanity[sanity["validationTask"].eq("nearest_neighbor_label_sanity")] if not sanity.empty else pd.DataFrame()
    overlap = sanity[sanity["validationTask"].eq("prior_embedding_neighbor_overlap")] if not sanity.empty else pd.DataFrame()
    lines = [
        f"# {STEP_ID} Platonic Distance Definition Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status['artifactsWritten'])} files, including the distance table, entity table, nearest-neighbor table, world behavior profile embedding, diagnostics, baseline comparisons, validation report, summary, status, model card, and manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Claim Boundary",
        "",
        PLATONIC_DISTANCE_CLAIM_BOUNDARY,
        "",
        "## Definition",
        "",
        "For each entity type, S08 defines the primary empirical Platonic distance between two embedded entities as:",
        "",
        "`D(a,b) = normalized_euclidean(E_a,E_b) + 0.10 * mean(sparse_coverage_penalty_a, sparse_coverage_penalty_b)`.",
        "",
        "The Euclidean term is computed in the S06 policy embedding, S07 goal embedding, or S08 world behavior-profile embedding space, then divided by the maximum observed pairwise Euclidean distance within that entity type. The sparse-coverage term carries S05 error-limit and S07 sparse-goal caveats into downstream distance use. Cosine distance is reported as a diagnostic component, but it is not part of the primary metric because cosine distance can violate triangle inequalities.",
        "",
        "## Scope",
        "",
        "- Policy distances use S06 S05-error-weighted policy embeddings and S06 policy coverage.",
        "- Goal distances use S07 goal embeddings and S07 sparse-goal uncertainty.",
        "- World distances use S05/S04 behavior profiles, S05 target reliability weights, and mean S06 policy plus S07 goal coordinates observed within each world.",
        "",
        "## Performance Anchor",
        "",
        f"S08 produced {perf['distancePairCount']} pairwise distances across {perf['policyEntityCount']} policies, {perf['goalEntityCount']} goals, and {perf['worldEntityCount']} worlds. Maximum symmetry error was {perf['maxSymmetryAbsError']:.6g}; sampled or exhaustive triangle violations: {perf['triangleViolationCount']} with maximum excess {perf['maxTriangleViolation']:.6g}.",
        "",
        "## Symmetry And Triangle Diagnostics",
        "",
        dataframe_markdown(diagnostics.sort_values("entityType"), limit=20),
        "",
        "## Prior Embedding Neighbor Overlap",
        "",
        dataframe_markdown(overlap.sort_values(["entityType", "k"]), limit=20),
        "",
        "## Nearest-Neighbor Label Sanity",
        "",
        dataframe_markdown(label_sanity.sort_values(["entityType", "label", "k"]), limit=80),
        "",
        "## Metadata/Substrate Baseline Comparison",
        "",
        dataframe_markdown(baseline_comparison.sort_values(["entityType", "metadataLabel"]), limit=80),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def validation_markdown(path: Path, validation: pd.DataFrame, diagnostics: pd.DataFrame, sanity: pd.DataFrame, status: Mapping[str, Any]) -> None:
    failed = validation[~validation["success"]]
    lines = [
        f"# {STEP_ID} Platonic Distance Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        "- Artifacts written: distance/entity tables, world behavior profiles and embeddings, nearest neighbors, symmetry/triangle diagnostics, nearest-neighbor sanity checks, metadata/substrate baseline comparisons, validation summary, model card, status, and artifact manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Validation Checks",
        "",
        dataframe_markdown(validation, limit=80),
        "",
        "## Symmetry And Triangle Diagnostics",
        "",
        dataframe_markdown(diagnostics, limit=20),
        "",
        "## Nearest-Neighbor Sanity Checks",
        "",
        dataframe_markdown(sanity, limit=100),
        "",
        "## Failed Or Warning Checks",
        "",
        dataframe_markdown(failed, limit=80) if not failed.empty else "No failed validation checks.",
        "",
        PLATONIC_DISTANCE_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_markdown(path: Path, status: Mapping[str, Any], perf: Mapping[str, Any]) -> None:
    lines = [
        f"# {STEP_ID} Status Summary",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status['artifactsWritten'])} files; primary outputs are the empirical Platonic distance table, distance entity table, nearest-neighbor table, world behavior-profile embedding, symmetry/triangle diagnostics, metadata/substrate baseline comparison, validation report, model card, status, and manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Outcome classification: {status['outcomeClassification']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        "- Lay summary: S08 turns the learned policy and goal maps, plus a behavior-profile map of worlds, into a single empirical distance table. Nearby entries mean similar computational behavior under the available simulated evidence, with sparse-data penalties carried into the score.",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        f"Anchor result: {perf['distancePairCount']} pairwise distances across {perf['policyEntityCount']} policies, {perf['goalEntityCount']} goals, and {perf['worldEntityCount']} worlds; maximum symmetry error {perf['maxSymmetryAbsError']:.6g}; triangle violations {perf['triangleViolationCount']}.",
        "",
        PLATONIC_DISTANCE_CLAIM_BOUNDARY,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--modeling-frame-path", type=Path, default=DEFAULT_FRAME_PATH)
    parser.add_argument("--world-catalog-path", type=Path, default=DEFAULT_WORLD_CATALOG_PATH)
    parser.add_argument("--error-limits-path", type=Path, default=DEFAULT_ERROR_LIMITS_PATH)
    parser.add_argument("--s05-model-card-path", type=Path, default=DEFAULT_S05_MODEL_CARD_PATH)
    parser.add_argument("--policy-embeddings-path", type=Path, default=DEFAULT_POLICY_EMBEDDINGS_PATH)
    parser.add_argument("--policy-coverage-path", type=Path, default=DEFAULT_POLICY_COVERAGE_PATH)
    parser.add_argument("--policy-neighbors-path", type=Path, default=DEFAULT_POLICY_NEIGHBORS_PATH)
    parser.add_argument("--goal-embeddings-path", type=Path, default=DEFAULT_GOAL_EMBEDDINGS_PATH)
    parser.add_argument("--goal-uncertainty-path", type=Path, default=DEFAULT_GOAL_UNCERTAINTY_PATH)
    parser.add_argument("--goal-neighbors-path", type=Path, default=DEFAULT_GOAL_NEIGHBORS_PATH)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    result_dir = args.artifacts_dir / "results"
    model_dir = args.artifacts_dir / "models" / "e07_platonic_distances"
    provenance_dir = args.artifacts_dir / "provenance"
    for directory in (step_dir, result_dir, model_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.modeling_frame_path)
    world_catalog = pd.read_parquet(args.world_catalog_path)
    error_limits = pd.read_parquet(args.error_limits_path)
    policy_embeddings = pd.read_parquet(args.policy_embeddings_path)
    policy_coverage = pd.read_parquet(args.policy_coverage_path)
    policy_neighbors = pd.read_parquet(args.policy_neighbors_path)
    goal_embeddings = pd.read_parquet(args.goal_embeddings_path)
    goal_uncertainty = pd.read_parquet(args.goal_uncertainty_path)
    goal_neighbors = pd.read_parquet(args.goal_neighbors_path)
    target_columns = target_columns_from_model_card(args.s05_model_card_path)
    target_weights = target_error_weights(frame, error_limits, target_columns)

    world_profiles, world_coverage = build_world_profiles(
        frame,
        world_catalog,
        policy_embeddings,
        goal_embeddings,
        target_weights,
        target_columns,
    )
    world_embeddings, world_model, world_matrix = fit_world_embedding(world_profiles)
    policy_penalty = policy_sparse_penalty(policy_embeddings, policy_coverage)
    goal_penalty = goal_sparse_penalty(goal_embeddings, goal_uncertainty)
    world_penalty = world_sparse_penalty(world_embeddings, world_coverage)

    policy_distances, policy_entities, policy_matrix = pairwise_distance_tables(
        policy_embeddings, "policy", "abstractPolicyId", POLICY_LABEL_COLUMNS, policy_penalty
    )
    goal_distances, goal_entities, goal_matrix = pairwise_distance_tables(goal_embeddings, "goal", "abstractGoalId", GOAL_LABEL_COLUMNS, goal_penalty)
    world_distances, world_entities, world_distance_matrix = pairwise_distance_tables(
        world_embeddings, "world", "worldId", WORLD_LABEL_COLUMNS, world_penalty
    )
    distances = pd.concat([policy_distances, goal_distances, world_distances], ignore_index=True)
    entities = pd.concat([policy_entities, goal_entities, world_entities], ignore_index=True)

    policy_nearest = nearest_neighbor_table(policy_embeddings, "policy", "abstractPolicyId", POLICY_LABEL_COLUMNS, policy_matrix, k=10)
    goal_nearest = nearest_neighbor_table(goal_embeddings, "goal", "abstractGoalId", GOAL_LABEL_COLUMNS, goal_matrix, k=10)
    world_nearest = nearest_neighbor_table(world_embeddings, "world", "worldId", WORLD_LABEL_COLUMNS, world_distance_matrix, k=10)
    nearest = pd.concat([policy_nearest, goal_nearest, world_nearest], ignore_index=True)

    diagnostics = pd.concat(
        [
            symmetry_triangle_diagnostics(
                "policy", policy_embeddings.sort_values("abstractPolicyId")["abstractPolicyId"].astype(str).tolist(), policy_matrix
            ),
            symmetry_triangle_diagnostics("goal", goal_embeddings.sort_values("abstractGoalId")["abstractGoalId"].astype(str).tolist(), goal_matrix),
            symmetry_triangle_diagnostics("world", world_embeddings.sort_values("worldId")["worldId"].astype(str).tolist(), world_distance_matrix),
        ],
        ignore_index=True,
    )
    sanity = pd.concat(
        [
            nearest_neighbor_sanity(nearest, "policy", BASELINE_LABEL_COLUMNS["policy"]),
            nearest_neighbor_sanity(nearest, "goal", BASELINE_LABEL_COLUMNS["goal"]),
            nearest_neighbor_sanity(nearest, "world", BASELINE_LABEL_COLUMNS["world"]),
            prior_neighbor_overlap(nearest, policy_neighbors, "policy"),
            prior_neighbor_overlap(nearest, goal_neighbors, "goal"),
        ],
        ignore_index=True,
    )
    baseline_comparison = distance_baseline_comparison(distances, BASELINE_LABEL_COLUMNS)
    validation = validate_platonic_distances(distances, entities, diagnostics, sanity, baseline_comparison, world_embeddings)
    hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]

    model_bundle = {
        "schemaVersion": PLATONIC_DISTANCE_SCHEMA_VERSION,
        "modelVersion": PLATONIC_DISTANCE_MODEL_VERSION,
        "worldEmbeddingModel": world_model,
        "targetWeights": target_weights.to_dict(orient="records"),
        "distanceDefinition": {
            "primary": "normalized_euclidean_plus_sparse_coverage_penalty",
            "formula": "D(a,b)=euclidean(Ea,Eb)/max_pairwise_euclidean + 0.10*mean(sparseCoveragePenaltyA,sparseCoveragePenaltyB)",
            "sparseCoveragePenaltyWeight": SPARSE_COVERAGE_PENALTY_WEIGHT,
            "cosineUse": "reported as diagnostic component only",
        },
        "claimBoundary": PLATONIC_DISTANCE_CLAIM_BOUNDARY,
    }
    model_path = model_dir / "distance_model_bundle.joblib"
    joblib.dump(model_bundle, model_path, compress=3)

    profile_paths = write_dataframe(world_profiles, step_dir / "world_behavior_profiles", csv=False)
    world_coverage_paths = write_dataframe(world_coverage, step_dir / "world_embedding_coverage")
    world_embedding_paths = write_dataframe(world_embeddings, step_dir / "world_embeddings")
    target_weight_paths = write_dataframe(target_weights, step_dir / "target_error_weights")
    entity_paths = write_dataframe(entities, step_dir / "distance_entities")
    distance_paths = write_dataframe(distances, step_dir / "platonic_distances")
    result_distance_path = result_dir / "e07_platonic_distances.parquet"
    distances.to_parquet(result_distance_path, index=False)
    result_embeddings_distance_path = result_dir / "e07_embeddings_distances.parquet"
    distances.to_parquet(result_embeddings_distance_path, index=False)
    nearest_paths = write_dataframe(nearest, step_dir / "nearest_neighbors")
    diagnostic_paths = write_dataframe(diagnostics, step_dir / "distance_diagnostics")
    sanity_paths = write_dataframe(sanity, step_dir / "nearest_neighbor_sanity")
    baseline_paths = write_dataframe(baseline_comparison, step_dir / "distance_baseline_comparison")
    validation_paths = write_dataframe(validation, step_dir / "distance_validation")

    test_command = None
    if not args.skip_tests:
        test_command = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS], cwd=REPO_ROOT)

    tests_ok = True if test_command is None else bool(test_command["success"])
    validation_ok = hard_failures.empty
    success = bool(validation_ok and tests_ok)
    validation_result = (
        f"{'passed' if success else 'failed'}: {len(distances)} pairwise distances, "
        f"{len(entities)} entities, {len(diagnostics)} symmetry/triangle diagnostic rows, "
        f"{len(hard_failures)} hard validation failures; focused tests "
        f"{'skipped' if test_command is None else 'passed' if tests_ok else 'failed'}"
    )
    perf = performance_summary(entities, distances, diagnostics, sanity, baseline_comparison)
    outcome = "supportive" if success and perf["triangleViolationCount"] == 0 else ("null" if success else "constraining/contradictory")
    caveats = (
        "Platonic distance is an empirical computational proxy over completed S04-S07 artifacts, not a metaphysical "
        "or biological distance. Policy distances exclude catalog-only S02 policies without S06 learned coordinates; "
        "goal distances exclude unobserved S03 goals without S07 learned coordinates; world distances are S08 behavior "
        "profile embeddings derived from available S04/S05 rows and mean S06/S07 context, so sparse worlds, rare "
        "morphology or chimeric regimes, and high-uncertainty goals should be treated cautiously downstream."
    )
    recommended = "Stop for Chief Scientist review before S09; if accepted, use S08 empirical distances and sparse-coverage caveats to search invariants in S09."
    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "title": STEP_TITLE,
        "artifactsWritten": [],
        "validationResult": validation_result,
        "outcomeClassification": outcome,
        "caveatsOrBlockers": caveats if success else caveats + " Review failed validation/test checks before S09.",
        "recommendedNextAction": recommended,
        "performanceSummary": perf,
        "focusedTestCommand": test_command,
        "claimBoundary": PLATONIC_DISTANCE_CLAIM_BOUNDARY,
    }

    validation_summary_path = step_dir / "distance_validation_summary.json"
    write_json(
        validation_summary_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": success,
            "hardValidationFailureCount": int(len(hard_failures)),
            "warningFailureCount": int(len(validation[(validation["severity"].eq("warning")) & (~validation["success"])])),
            "performanceSummary": perf,
            "validationResult": validation_result,
            "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
            "recommendedNextAction": recommended,
        },
    )

    model_card = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": PLATONIC_DISTANCE_SCHEMA_VERSION,
        "modelVersion": PLATONIC_DISTANCE_MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "modelingFramePath": str(args.modeling_frame_path),
        "modelingFrameSha256": sha256_path(args.modeling_frame_path),
        "worldCatalogPath": str(args.world_catalog_path),
        "worldCatalogSha256": sha256_path(args.world_catalog_path),
        "s05ErrorLimitsPath": str(args.error_limits_path),
        "s05ErrorLimitsSha256": sha256_path(args.error_limits_path),
        "s06PolicyEmbeddingsPath": str(args.policy_embeddings_path),
        "s06PolicyEmbeddingsSha256": sha256_path(args.policy_embeddings_path),
        "s07GoalEmbeddingsPath": str(args.goal_embeddings_path),
        "s07GoalEmbeddingsSha256": sha256_path(args.goal_embeddings_path),
        "distanceDefinition": model_bundle["distanceDefinition"],
        "targetColumns": target_columns,
        "targetWeights": target_weights.to_dict(orient="records"),
        "performanceSummary": perf,
        "claimBoundary": PLATONIC_DISTANCE_CLAIM_BOUNDARY,
        "sklearnVersion": sklearn.__version__,
    }
    model_card_path = model_dir / "model_card.json"
    write_json(model_card_path, model_card)

    definition_report_path = step_dir / "distance_definition_report.md"
    validation_report_path = step_dir / "validation_report.md"
    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    artifact_paths = [
        *profile_paths,
        *world_coverage_paths,
        *world_embedding_paths,
        *target_weight_paths,
        *entity_paths,
        *distance_paths,
        result_distance_path,
        result_embeddings_distance_path,
        *nearest_paths,
        *diagnostic_paths,
        *sanity_paths,
        *baseline_paths,
        *validation_paths,
        model_path,
        model_card_path,
        validation_summary_path,
        definition_report_path,
        validation_report_path,
        summary_path,
        status_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    status_payload["artifactsWritten"] = collect_artifacts(
        [
            path
            for path in artifact_paths
            if path not in {definition_report_path, validation_report_path, summary_path, status_path, artifact_manifest_path, run_manifest_path}
        ]
    )
    definition_report_markdown(definition_report_path, status_payload, perf, diagnostics, sanity, baseline_comparison)
    validation_markdown(validation_report_path, validation, diagnostics, sanity, status_payload)
    summary_markdown(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": PLATONIC_DISTANCE_SCHEMA_VERSION,
        "modelVersion": PLATONIC_DISTANCE_MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "status": status_payload["status"],
        "artifactsWritten": collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path]),
        "validationResult": status_payload["validationResult"],
        "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
        "recommendedNextAction": status_payload["recommendedNextAction"],
        "claimBoundary": PLATONIC_DISTANCE_CLAIM_BOUNDARY,
    }
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    definition_report_markdown(definition_report_path, status_payload, perf, diagnostics, sanity, baseline_comparison)
    validation_markdown(validation_report_path, validation, diagnostics, sanity, status_payload)
    summary_markdown(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    run_manifest = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "startedAtApprox": datetime.now(UTC).isoformat(),
        "elapsedSeconds": round(time.perf_counter() - started, 6),
        "command": [sys.executable, *sys.argv],
        "repo": {
            "root": str(REPO_ROOT),
            "branch": git_value(["branch", "--show-current"]),
            "commit": git_value(["rev-parse", "HEAD"]),
            "dirtyStatus": git_value(["status", "--short"]),
        },
        "inputs": {
            "modelingFramePath": str(args.modeling_frame_path),
            "worldCatalogPath": str(args.world_catalog_path),
            "errorLimitsPath": str(args.error_limits_path),
            "s05ModelCardPath": str(args.s05_model_card_path),
            "policyEmbeddingsPath": str(args.policy_embeddings_path),
            "policyCoveragePath": str(args.policy_coverage_path),
            "policyNeighborsPath": str(args.policy_neighbors_path),
            "goalEmbeddingsPath": str(args.goal_embeddings_path),
            "goalUncertaintyPath": str(args.goal_uncertainty_path),
            "goalNeighborsPath": str(args.goal_neighbors_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threading": "serial numpy/sklearn; no GPU used",
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "researchSteps": {STEP_ID: status_payload},
    }
    write_json(run_manifest_path, run_manifest)
    manifest_payload["artifactsWritten"] = collect_artifacts([path for path in artifact_paths if path != artifact_manifest_path])
    write_json(artifact_manifest_path, manifest_payload)
    status_payload["artifactsWritten"] = manifest_payload["artifactsWritten"]
    definition_report_markdown(definition_report_path, status_payload, perf, diagnostics, sanity, baseline_comparison)
    validation_markdown(validation_report_path, validation, diagnostics, sanity, status_payload)
    summary_markdown(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
