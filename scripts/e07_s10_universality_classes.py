#!/usr/bin/env python3
"""Find E07 S10 substrate-spanning empirical universality classes."""

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

import numpy as np
import pandas as pd
import sklearn

from platonic_space.universality_classes import (
    S09_CONSTRAINING_CAVEAT,
    UNIVERSALITY_CLASS_CLAIM_BOUNDARY,
    UNIVERSALITY_CLASS_MODEL_VERSION,
    UNIVERSALITY_CLASS_SCHEMA_VERSION,
    build_policy_taxonomy_features,
    build_taxonomy_tables,
    build_upstream_label_table,
    policy_distance_matrix,
    select_cluster_count,
    stability_tables,
    upstream_label_agreement,
    validation_table,
)
from platonic_space.world_schema import sha256_path, write_json


EXPERIMENT_ID = "E07"
STEP_ID = "S10"
STEP_NUMBER = 10
STEP_TITLE = "Find universality classes"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_POLICY_EMBEDDINGS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S06" / "policy_embeddings.parquet"
DEFAULT_POLICY_PROFILES_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S06" / "policy_behavior_profiles.parquet"
DEFAULT_GOAL_EMBEDDINGS_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S07" / "goal_embeddings.parquet"
DEFAULT_S08_DISTANCES_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S08" / "platonic_distances.parquet"
DEFAULT_S08_ENTITIES_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S08" / "distance_entities.parquet"
DEFAULT_S09_FEATURE_FRAME_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S09" / "invariant_feature_frame.parquet"
DEFAULT_S09_CANDIDATES_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S09" / "invariant_candidates.parquet"
DEFAULT_S09_MODEL_COMPARISON_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S09" / "model_comparison.parquet"
DEFAULT_POLICY_CATALOG_PATH = DEFAULT_ARTIFACTS_DIR / "research_steps" / "S02" / "policy_abstract_catalog.parquet"
DEFAULT_E03_ASSIGNMENTS_PATH = Path("/previous-artifacts/E03/research_steps/S11/policy_class_assignments.parquet")
DEFAULT_E06_PANEL_PATH = Path("/previous-artifacts/E06/research_steps/S01/algotype_panel.parquet")
DEFAULT_E06_DOMINANCE_PATH = Path("/previous-artifacts/E06/results/e06_dominance_hierarchy.parquet")
DEFAULT_E06_MOSAIC_PATH = Path("/previous-artifacts/E06/results/e06_mosaic_classifications.parquet")
FOCUSED_TESTS = ["tests.test_e07_universality_classes"]


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


def read_optional_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


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


def dataframe_markdown(df: pd.DataFrame, limit: int = 40) -> str:
    if df.empty:
        return "No rows."
    display = df.head(limit).copy()
    return markdown_table(display.astype(str).to_dict(orient="records"), [str(column) for column in display.columns])


def missingness_table(assignments: pd.DataFrame, label_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in ("e03ClassLabel", "e06RoleLabel", "e06MosaicModalClass", "goalContextTopGoalFamily", "distanceUncertaintyLevel"):
        if column not in assignments.columns:
            continue
        present = assignments[column].astype(str).str.strip().replace({"unknown": "", "nan": ""}).ne("")
        rows.append(
            {
                "field": column,
                "rowCount": int(len(assignments)),
                "presentCount": int(present.sum()),
                "missingCount": int((~present).sum()),
                "presentFraction": float(present.mean()),
                "caveat": "Expected sparse upstream labels outside the source experiment; not all S10 policies have E03 or E06 labels.",
            }
        )
    for row in label_audit.to_dict(orient="records"):
        rows.append(
            {
                "field": row.get("usableLabelColumn", "unknown"),
                "rowCount": int(row.get("sourceRows", 0)),
                "presentCount": int(row.get("embeddedPolicyMatches", 0)),
                "missingCount": None,
                "presentFraction": None,
                "caveat": row.get("caveat", ""),
            }
        )
    return pd.DataFrame(rows)


def performance_summary(
    assignments: pd.DataFrame,
    class_summary: pd.DataFrame,
    stability: pd.DataFrame,
    agreement: pd.DataFrame,
    validation: pd.DataFrame,
    selected_cluster_count: int,
) -> dict[str, Any]:
    def metric(task: str, column: str = "value") -> float:
        rows = stability[stability["validationTask"].eq(task)]
        return float(rows[column].iloc[0]) if not rows.empty and pd.notna(rows[column].iloc[0]) else float("nan")

    e03 = agreement[(agreement["comparisonScope"].eq("E03")) & (agreement["referenceLabelColumn"].eq("e03ClassLabel"))]
    e06 = agreement[(agreement["comparisonScope"].eq("E06")) & (agreement["referenceLabelColumn"].eq("e06RoleLabel"))]
    hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]
    warnings = validation[(validation["severity"].eq("warning")) & (~validation["success"])]
    return {
        "policyAssignmentCount": int(len(assignments)),
        "classCount": int(class_summary["universalityClassId"].nunique()) if not class_summary.empty else 0,
        "selectedClusterCount": int(selected_cluster_count),
        "minClassSize": int(class_summary["policyCount"].min()) if not class_summary.empty else 0,
        "maxClassSize": int(class_summary["policyCount"].max()) if not class_summary.empty else 0,
        "s08SameClassNeighborFractionAt10": metric("s08_nearest_neighbor_class_coherence"),
        "s08RandomExpectedSameClassFraction": metric("s08_nearest_neighbor_class_coherence", "baselineValue"),
        "subsampleAdjustedRandMean": metric("subsample_cluster_stability"),
        "seedAdjustedRandMean": metric("seed_stability"),
        "s08AverageLinkageAdjustedRand": metric("s08_distance_tree_modality_agreement"),
        "e03LabeledPolicyCount": int(e03["policyCount"].iloc[0]) if not e03.empty else 0,
        "e03NormalizedMutualInformation": float(e03["normalizedMutualInformation"].iloc[0]) if not e03.empty else float("nan"),
        "e06LabeledPolicyCount": int(e06["policyCount"].iloc[0]) if not e06.empty else 0,
        "e06NormalizedMutualInformation": float(e06["normalizedMutualInformation"].iloc[0]) if not e06.empty else float("nan"),
        "hardValidationFailureCount": int(len(hard_failures)),
        "warningValidationFailureCount": int(len(warnings)),
    }


def class_report_markdown(path: Path, status: Mapping[str, Any], class_summary: pd.DataFrame, exemplars: pd.DataFrame, agreement: pd.DataFrame) -> None:
    display_cols = [
        "universalityClassId",
        "classLabel",
        "policyCount",
        "exemplarPolicyLabel",
        "topPolicyFamiliesJson",
        "topGoalFamiliesJson",
        "topE03ClassLabelsJson",
        "topE06RoleLabelsJson",
        "s08MeanWithinClassDistance",
        "s08NearestOutsideClassDistance",
    ]
    exemplar_cols = [
        "universalityClassId",
        "exemplarRank",
        "policyLabel",
        "sourceExperimentId",
        "policyFamily",
        "distanceToClassMedoidS08",
        "classAssignmentConfidence",
    ]
    lines = [
        f"# {STEP_ID} Class Exemplar Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        "- Artifacts written: universality class assignment table, class summary, class exemplars, cluster selection diagnostics, stability metrics, upstream label agreement, label-source audit, validation report, summary, status, manifest, model card, and results export.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Claim Boundary",
        "",
        UNIVERSALITY_CLASS_CLAIM_BOUNDARY,
        "",
        "## S09 Constraint Carried Forward",
        "",
        S09_CONSTRAINING_CAVEAT,
        "",
        "## Class Summary",
        "",
        dataframe_markdown(class_summary[[column for column in display_cols if column in class_summary.columns]], limit=30),
        "",
        "## Exemplars",
        "",
        dataframe_markdown(exemplars[[column for column in exemplar_cols if column in exemplars.columns]], limit=80),
        "",
        "## Upstream Label Agreement",
        "",
        dataframe_markdown(agreement, limit=20),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def stability_report_markdown(
    path: Path,
    status: Mapping[str, Any],
    cluster_selection: pd.DataFrame,
    stability: pd.DataFrame,
    validation: pd.DataFrame,
    missingness: pd.DataFrame,
    label_audit: pd.DataFrame,
) -> None:
    lines = [
        f"# {STEP_ID} Stability And Validation Report",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        "- Artifacts written: cluster selection table, stability metrics, upstream label agreement, taxonomy validation, missingness report, class report, summary, status, manifest, and results export.",
        f"- Validation result: {status['validationResult']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        "## Cluster Selection",
        "",
        dataframe_markdown(cluster_selection, limit=20),
        "",
        "## Stability Metrics",
        "",
        dataframe_markdown(stability, limit=20),
        "",
        "## Validation Checks",
        "",
        dataframe_markdown(validation, limit=40),
        "",
        "## Missingness",
        "",
        dataframe_markdown(missingness, limit=40),
        "",
        "## Label Source Audit",
        "",
        dataframe_markdown(label_audit, limit=20),
        "",
        "## Interpretation Caveat",
        "",
        S09_CONSTRAINING_CAVEAT,
        "",
        UNIVERSALITY_CLASS_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_markdown(path: Path, status: Mapping[str, Any], perf: Mapping[str, Any]) -> None:
    lines = [
        f"# {STEP_ID} Status Summary",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {status['status']}",
        f"- Artifacts written: {len(status.get('artifactsWritten', []))} files; primary outputs are policy class assignments, class summary, class exemplars, stability/validation reports, upstream label agreement, model card, results export, status, and manifest.",
        f"- Validation result: {status['validationResult']}",
        f"- Outcome classification: {status['outcomeClassification']}",
        f"- Caveats or blockers: {status['caveatsOrBlockers']}",
        "- Lay summary: S10 groups the behavior-embedded policies into empirical behavioral neighborhoods and checks whether those neighborhoods are stable, coherent under S08 nearest-neighbor distances, and compatible with upstream E03/E06 labels.",
        f"- Recommended next action: {status['recommendedNextAction']}",
        "",
        f"Anchor result: {perf['policyAssignmentCount']} policies were assigned to {perf['classCount']} classes; S08 same-class neighbor fraction@10 was {perf['s08SameClassNeighborFractionAt10']:.4f} versus a random-label expectation of {perf['s08RandomExpectedSameClassFraction']:.4f}; subsample ARI mean was {perf['subsampleAdjustedRandMean']:.4f}.",
        "",
        S09_CONSTRAINING_CAVEAT,
        "",
        UNIVERSALITY_CLASS_CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--policy-embeddings-path", type=Path, default=DEFAULT_POLICY_EMBEDDINGS_PATH)
    parser.add_argument("--policy-profiles-path", type=Path, default=DEFAULT_POLICY_PROFILES_PATH)
    parser.add_argument("--goal-embeddings-path", type=Path, default=DEFAULT_GOAL_EMBEDDINGS_PATH)
    parser.add_argument("--s08-distances-path", type=Path, default=DEFAULT_S08_DISTANCES_PATH)
    parser.add_argument("--s08-entities-path", type=Path, default=DEFAULT_S08_ENTITIES_PATH)
    parser.add_argument("--s09-feature-frame-path", type=Path, default=DEFAULT_S09_FEATURE_FRAME_PATH)
    parser.add_argument("--s09-candidates-path", type=Path, default=DEFAULT_S09_CANDIDATES_PATH)
    parser.add_argument("--s09-model-comparison-path", type=Path, default=DEFAULT_S09_MODEL_COMPARISON_PATH)
    parser.add_argument("--policy-catalog-path", type=Path, default=DEFAULT_POLICY_CATALOG_PATH)
    parser.add_argument("--e03-assignments-path", type=Path, default=DEFAULT_E03_ASSIGNMENTS_PATH)
    parser.add_argument("--e06-panel-path", type=Path, default=DEFAULT_E06_PANEL_PATH)
    parser.add_argument("--e06-dominance-path", type=Path, default=DEFAULT_E06_DOMINANCE_PATH)
    parser.add_argument("--e06-mosaic-path", type=Path, default=DEFAULT_E06_MOSAIC_PATH)
    parser.add_argument("--stability-resamples", type=int, default=24)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    result_dir = args.artifacts_dir / "results"
    model_dir = args.artifacts_dir / "models" / "e07_universality_classes"
    provenance_dir = args.artifacts_dir / "provenance"
    for directory in (step_dir, result_dir, model_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    policy_embeddings = pd.read_parquet(args.policy_embeddings_path)
    policy_profiles = read_optional_parquet(args.policy_profiles_path)
    goal_embeddings = pd.read_parquet(args.goal_embeddings_path)
    s08_distances = pd.read_parquet(args.s08_distances_path)
    s08_entities = pd.read_parquet(args.s08_entities_path)
    s09_feature_frame = pd.read_parquet(args.s09_feature_frame_path)
    s09_candidates = pd.read_parquet(args.s09_candidates_path)
    s09_model_comparison = read_optional_parquet(args.s09_model_comparison_path)
    policy_catalog = pd.read_parquet(args.policy_catalog_path)
    e03_assignments = read_optional_parquet(args.e03_assignments_path)
    e06_panel = read_optional_parquet(args.e06_panel_path)
    e06_dominance = read_optional_parquet(args.e06_dominance_path)
    e06_mosaic = read_optional_parquet(args.e06_mosaic_path)

    embedded_policy_ids = policy_embeddings["abstractPolicyId"].astype(str).tolist()
    upstream_labels, label_audit = build_upstream_label_table(
        policy_catalog,
        embedded_policy_ids,
        e03_assignments=e03_assignments,
        e06_panel=e06_panel,
        e06_dominance=e06_dominance,
        e06_mosaic=e06_mosaic,
    )
    policies, feature_matrix, feature_sources = build_policy_taxonomy_features(
        policy_embeddings,
        goal_embeddings,
        s09_feature_frame,
        policy_catalog,
        s08_entities,
        upstream_labels,
        policy_behavior_profiles=policy_profiles,
    )
    policy_ids = policies["abstractPolicyId"].astype(str).tolist()
    s08_matrix = policy_distance_matrix(s08_distances, policy_ids)
    selected_k, labels, cluster_selection = select_cluster_count(feature_matrix, s08_matrix)
    assignments, class_summary, exemplars = build_taxonomy_tables(policies, feature_matrix, s08_matrix, labels, selected_k)
    stability = stability_tables(feature_matrix, s08_matrix, labels, selected_k, resamples=args.stability_resamples)
    agreement = upstream_label_agreement(assignments)
    validation = validation_table(assignments, class_summary, stability, agreement, cluster_selection, s09_candidates)
    missingness = missingness_table(assignments, label_audit)
    validation_for_write = validation.copy()
    for column in ("value", "threshold"):
        if column in validation_for_write.columns:
            validation_for_write[column] = validation_for_write[column].map(lambda value: "" if value is None else str(value))

    assignment_paths = write_dataframe(assignments, step_dir / "universality_classes")
    class_paths = write_dataframe(class_summary, step_dir / "class_summary")
    exemplar_paths = write_dataframe(exemplars, step_dir / "class_exemplars")
    selection_paths = write_dataframe(cluster_selection, step_dir / "cluster_selection")
    stability_paths = write_dataframe(stability, step_dir / "class_stability")
    agreement_paths = write_dataframe(agreement, step_dir / "upstream_label_agreement")
    label_audit_paths = write_dataframe(label_audit, step_dir / "label_source_audit")
    validation_paths = write_dataframe(validation_for_write, step_dir / "taxonomy_validation")
    missingness_paths = write_dataframe(missingness, step_dir / "taxonomy_missingness")
    result_path = result_dir / "e07_universality_classes.parquet"
    assignments.to_parquet(result_path, index=False)
    result_summary_path = result_dir / "e07_universality_class_summary.parquet"
    class_summary.to_parquet(result_summary_path, index=False)

    test_command = None
    if not args.skip_tests:
        test_command = run_command([sys.executable, "-m", "unittest", *FOCUSED_TESTS], cwd=REPO_ROOT)
    tests_ok = True if test_command is None else bool(test_command["success"])
    hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]
    success = bool(hard_failures.empty and tests_ok)
    perf = performance_summary(assignments, class_summary, stability, agreement, validation, selected_k)
    validation_result = (
        f"{'passed' if success else 'failed'}: {perf['policyAssignmentCount']} policy assignments, "
        f"{perf['classCount']} classes, selected k={selected_k}, S08 same-neighbor@10="
        f"{perf['s08SameClassNeighborFractionAt10']:.4f} versus random {perf['s08RandomExpectedSameClassFraction']:.4f}, "
        f"{perf['hardValidationFailureCount']} hard validation failures; focused tests "
        f"{'skipped' if test_command is None else 'passed' if tests_ok else 'failed'}"
    )
    if not success:
        outcome = "constraining/contradictory"
    elif perf["s08SameClassNeighborFractionAt10"] > perf["s08RandomExpectedSameClassFraction"] and perf["subsampleAdjustedRandMean"] >= 0.40:
        outcome = "supportive"
    else:
        outcome = "null"
    caveats = (
        "Classes are empirical computational neighborhoods, not mathematical universality proofs. "
        f"{S09_CONSTRAINING_CAVEAT} Upstream E03 labels are partly inherited through base pc_* policy IDs for ablation rows, "
        "E06 role labels are sparse and context-dependent, and S08 direct average-linkage clustering is retained as a modality diagnostic rather than the class-definition algorithm."
    )
    recommended = "Stop for Chief Scientist review before S11; if accepted, use S10 classes as strata for S11 counterfactual prediction tests without treating them as causal invariants."
    status_payload: dict[str, Any] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "title": STEP_TITLE,
        "artifactsWritten": [],
        "validationResult": validation_result,
        "outcomeClassification": outcome,
        "caveatsOrBlockers": caveats if success else caveats + " Review failed validation/test checks before S11.",
        "recommendedNextAction": recommended,
        "performanceSummary": perf,
        "focusedTestCommand": test_command,
        "claimBoundary": UNIVERSALITY_CLASS_CLAIM_BOUNDARY,
    }

    validation_summary_path = step_dir / "taxonomy_validation_summary.json"
    model_card_path = model_dir / "model_card.json"
    class_report_path = step_dir / "class_exemplar_report.md"
    stability_report_path = step_dir / "stability_validation_report.md"
    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

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
        "schemaVersion": UNIVERSALITY_CLASS_SCHEMA_VERSION,
        "modelVersion": UNIVERSALITY_CLASS_MODEL_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "policyEmbeddingsPath": str(args.policy_embeddings_path),
        "policyEmbeddingsSha256": sha256_path(args.policy_embeddings_path),
        "goalEmbeddingsPath": str(args.goal_embeddings_path),
        "goalEmbeddingsSha256": sha256_path(args.goal_embeddings_path),
        "s08DistancesPath": str(args.s08_distances_path),
        "s08DistancesSha256": sha256_path(args.s08_distances_path),
        "s09FeatureFramePath": str(args.s09_feature_frame_path),
        "s09FeatureFrameSha256": sha256_path(args.s09_feature_frame_path),
        "s09CandidatesPath": str(args.s09_candidates_path),
        "s09CandidatesSha256": sha256_path(args.s09_candidates_path),
        "policyCatalogPath": str(args.policy_catalog_path),
        "policyCatalogSha256": sha256_path(args.policy_catalog_path),
        "upstreamLabelSources": label_audit.to_dict(orient="records"),
        "featureSources": feature_sources,
        "selectedClusterCount": int(selected_k),
        "performanceSummary": perf,
        "s09Constraint": S09_CONSTRAINING_CAVEAT,
        "claimBoundary": UNIVERSALITY_CLASS_CLAIM_BOUNDARY,
        "sklearnVersion": sklearn.__version__,
    }
    write_json(model_card_path, model_card)

    class_report_markdown(class_report_path, status_payload, class_summary, exemplars, agreement)
    stability_report_markdown(stability_report_path, status_payload, cluster_selection, stability, validation, missingness, label_audit)

    artifact_paths = [
        *assignment_paths,
        *class_paths,
        *exemplar_paths,
        *selection_paths,
        *stability_paths,
        *agreement_paths,
        *label_audit_paths,
        *validation_paths,
        *missingness_paths,
        result_path,
        result_summary_path,
        validation_summary_path,
        model_card_path,
        class_report_path,
        stability_report_path,
        summary_path,
        status_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    status_payload["artifactsWritten"] = collect_artifacts(artifact_paths)
    summary_markdown(summary_path, status_payload, perf)
    write_json(status_path, status_payload)

    final_artifacts = collect_artifacts(artifact_paths)
    status_payload["artifactsWritten"] = final_artifacts
    write_json(status_path, status_payload)
    summary_markdown(summary_path, status_payload, perf)

    input_paths = [
        args.policy_embeddings_path,
        args.policy_profiles_path,
        args.goal_embeddings_path,
        args.s08_distances_path,
        args.s08_entities_path,
        args.s09_feature_frame_path,
        args.s09_candidates_path,
        args.s09_model_comparison_path,
        args.policy_catalog_path,
        args.e03_assignments_path,
        args.e06_panel_path,
        args.e06_dominance_path,
        args.e06_mosaic_path,
    ]
    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": UNIVERSALITY_CLASS_SCHEMA_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "status": status_payload["status"],
        "success": success,
        "artifactsWritten": final_artifacts,
        "inputArtifacts": [
            {"path": str(path), "exists": path.exists(), "sha256": sha256_path(path) if path.exists() and path.is_file() else None}
            for path in input_paths
        ],
        "validationResult": validation_result,
        "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
        "recommendedNextAction": recommended,
        "claimBoundary": UNIVERSALITY_CLASS_CLAIM_BOUNDARY,
    }
    write_json(artifact_manifest_path, manifest_payload)

    run_manifest = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "argv": sys.argv[1:],
        "createdAt": datetime.now(UTC).isoformat(),
        "elapsedSeconds": round(time.perf_counter() - started, 6),
        "status": status_payload["status"],
        "success": success,
        "gitCommit": git_value(["rev-parse", "HEAD"]),
        "gitBranch": git_value(["branch", "--show-current"]),
        "gitStatusShort": git_value(["status", "--short"]),
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "packageVersions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "performanceSummary": perf,
        "artifactsWritten": collect_artifacts([*artifact_paths, artifact_manifest_path, status_path, summary_path]),
        "claimBoundary": UNIVERSALITY_CLASS_CLAIM_BOUNDARY,
    }
    write_json(run_manifest_path, run_manifest)

    final_artifacts = collect_artifacts([*artifact_paths, artifact_manifest_path, status_path, summary_path, run_manifest_path])
    status_payload["artifactsWritten"] = final_artifacts
    write_json(status_path, status_payload)
    manifest_payload["artifactsWritten"] = final_artifacts
    write_json(artifact_manifest_path, manifest_payload)
    summary_markdown(summary_path, status_payload, perf)
    class_report_markdown(class_report_path, status_payload, class_summary, exemplars, agreement)
    stability_report_markdown(stability_report_path, status_payload, cluster_selection, stability, validation, missingness, label_audit)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
