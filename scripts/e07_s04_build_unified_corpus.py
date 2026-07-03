#!/usr/bin/env python3
"""Build the E07 S04 unified behavior corpus."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e07.corpus_schema import (  # noqa: E402
    CORPUS_SCHEMA_VERSION,
    MISSINGNESS_COLUMNS,
    SOURCE_MANIFEST_COLUMNS,
    dataframe_from_records,
    sha256_file,
    stable_hash,
    stable_json,
    validate_unified_corpus,
)


STEP_ID = "S04"
STEP_NUMBER = 4
EXPERIMENT_ID = "E07"

SOURCE_SPECS = (
    ("E01", "e01_core_results", "/previous-artifacts/E01/tables/e01_core_results.csv", "claim_summary", None),
    ("E01", "e01_dg_numeric_table", "/previous-artifacts/E01/tables/e01_dg_numeric_table.csv", "condition_summary", None),
    ("E01", "e01_aggregation_peak_table", "/previous-artifacts/E01/tables/e01_aggregation_peak_table.csv", "condition_summary", None),
    ("E01", "e01_conflict_equilibria_summary", "/previous-artifacts/E01/tables/e01_conflict_equilibria_summary.csv", "condition_summary", None),
    ("E02", "e02_activation_rate_artifacts", "/previous-artifacts/E02/results/e02_activation_rate_artifacts.parquet", "run", None),
    ("E02", "e02_alternative_metrics", "/previous-artifacts/E02/results/e02_alternative_metrics.parquet", "run_or_summary", None),
    ("E02", "e02_dg_null_benchmarks", "/previous-artifacts/E02/results/e02_dg_null_benchmarks.parquet", "null_run", None),
    ("E02", "e02_dummy_algotype_controls", "/previous-artifacts/E02/results/e02_dummy_algotype_controls.parquet", "run", None),
    ("E02", "e02_frozen_behavior_variants", "/previous-artifacts/E02/results/e02_frozen_behavior_variants.parquet", "run", None),
    ("E02", "e02_frozen_placement", "/previous-artifacts/E02/results/e02_frozen_placement.parquet", "run", None),
    ("E02", "e02_input_distribution_matrix", "/previous-artifacts/E02/results/e02_input_distribution_matrix.parquet", "run", None),
    ("E02", "e02_label_shuffle_aggregation_nulls", "/previous-artifacts/E02/results/e02_label_shuffle_aggregation_nulls.parquet", "null_summary", None),
    ("E02", "e02_local_move_nulls", "/previous-artifacts/E02/results/e02_local_move_nulls.parquet", "null_run", None),
    ("E02", "e02_scheduler_comparison", "/previous-artifacts/E02/results/e02_scheduler_comparison.parquet", "run", None),
    ("E02", "e02_simulator_validation", "/previous-artifacts/E02/results/e02_simulator_validation.parquet", "validation_case", None),
    ("E02", "e02_speed_matched_chimeras", "/previous-artifacts/E02/results/e02_speed_matched_chimeras.parquet", "run", None),
    ("E02", "e02_stop_condition_sensitivity", "/previous-artifacts/E02/results/e02_stop_condition_sensitivity.parquet", "run", None),
    ("E02", "e02_alternative_explanation_matrix", "/previous-artifacts/E02/tables/e02_alternative_explanation_matrix.csv", "audit_summary", None),
    ("E02", "e02_claims_that_survive", "/previous-artifacts/E02/tables/e02_claims_that_survive.csv", "audit_summary", None),
    ("E03", "e03_policy_competence", "/previous-artifacts/E03/results/e03_policy_competence.parquet", "policy_summary", None),
    ("E03", "e03_policy_embeddings", "/previous-artifacts/E03/results/e03_policy_embeddings.parquet", "policy_summary", None),
    ("E03", "e03_policy_cluster_summary", "/previous-artifacts/E03/results/e03_policy_cluster_summary.parquet", "cluster_summary", None),
    ("E03", "e03_frontier_candidate_evaluations", "/previous-artifacts/E03/results/e03_frontier_candidate_evaluations.parquet", "run", None),
    ("E03", "e03_rule_ablation_results", "/previous-artifacts/E03/results/e03_rule_ablation_results.parquet", "ablation_summary", None),
    ("E04", "e04_homeostatic_baselines", "/previous-artifacts/E04/results/e04_homeostatic_baselines.parquet", "run", "S05"),
    ("E04", "e04_intelligence_like_competencies", "/previous-artifacts/E04/results/e04_intelligence_like_competencies.parquet", "run_or_policy_summary", "S11"),
    ("E04", "e04_repairable_frozen_cells", "/previous-artifacts/E04/results/e04_repairable_frozen_cells.parquet", "run", "S03"),
    ("E04", "e04_memory_ablations", "/previous-artifacts/E04/results/e04_memory_ablations.parquet", "ablation_run", "S09"),
    ("E04", "e04_communication_ablations", "/previous-artifacts/E04/results/e04_communication_ablations.parquet", "ablation_run", "S10"),
    ("E05", "e05_benchmark_reference_results", "/previous-artifacts/E05/results/e05_benchmark_reference_results.parquet", "run", None),
    ("E05", "e05_benchmark_task_catalog", "/previous-artifacts/E05/results/e05_benchmark_task_catalog.parquet", "task_summary", None),
    ("E05", "e05_morphospace_route_metrics", "/previous-artifacts/E05/results/e05_morphospace_route_metrics.parquet", "route_summary", None),
    ("E05", "e05_scrambled_embryo_results", "/previous-artifacts/E05/results/e05_scrambled_embryo_results.parquet", "run", None),
    ("E05", "e05_regeneration_results", "/previous-artifacts/E05/results/e05_regeneration_results.parquet", "run", None),
    ("E05", "e05_shape_dg", "/previous-artifacts/E05/results/e05_shape_dg.parquet", "trajectory_summary", None),
    ("E05", "e05_metric_target_state_checks", "/previous-artifacts/E05/results/e05_metric_target_state_checks.parquet", "metric_target_check", None),
    ("E06", "e06_compatibility_scores", "/previous-artifacts/E06/results/e06_compatibility_scores.parquet", "condition_summary", None),
    ("E06", "e06_goal_compatibility", "/previous-artifacts/E06/results/e06_goal_compatibility.parquet", "run", None),
    ("E06", "e06_dominance_contests", "/previous-artifacts/E06/results/e06_dominance_contests.parquet", "run", None),
    ("E06", "e06_governance_interventions", "/previous-artifacts/E06/results/e06_governance_interventions.parquet", "intervention_run", None),
    ("E06", "e06_intervention_search", "/previous-artifacts/E06/results/e06_intervention_search.parquet", "intervention_summary", None),
    ("E06", "e06_final_state_prediction", "/previous-artifacts/E06/results/e06_final_state_prediction.parquet", "prediction", "S13"),
)

TRACE_ARTIFACTS = {
    "E01": (
        "/previous-artifacts/E01/traces/e01_figure3_trajectories.parquet",
        "/previous-artifacts/E01/traces/e01_frozen_cell_trajectories.parquet",
        "/previous-artifacts/E01/traces/e01_same_goal_chimeras.parquet",
        "/previous-artifacts/E01/traces/e01_duplicate_value_chimeras.parquet",
        "/previous-artifacts/E01/traces/e01_opposite_direction_chimeras.parquet",
    ),
    "E05": (
        "/previous-artifacts/E05/results/e05_morphospace_route_metrics.parquet",
        "/previous-artifacts/E05/results/e05_shape_dg.parquet",
    ),
    "E06": (
        "/previous-artifacts/E06/results/e06_goal_compatibility.parquet",
        "/previous-artifacts/E06/results/e06_dominance_contests.parquet",
        "/previous-artifacts/E06/results/e06_governance_interventions.parquet",
    ),
}

EXCLUDE_NUMERIC_PATTERNS = (
    "seed",
    "row_index",
    "source_row",
    "rank",
    "index",
    "array_size",
    "site_count",
    "event_cap",
    "record_every",
    "split_index",
    "generation",
    "population_index",
)

INCLUDE_METRIC_PATTERNS = (
    "sortedness",
    "aggregation",
    "dg_",
    "target_error",
    "aggregate_morphospace_error",
    "recovery_fraction",
    "recovered_perturbation",
    "quality",
    "score",
    "fitness_score",
    "competency_score",
    "synergy",
    "efficiency",
    "conflict",
    "dominance_abs_margin",
    "integration_score",
    "interface_stability_score",
    "largest_block_fraction",
    "position_bias_margin",
    "goal_alignment_gap",
    "observed_numeric",
    "predicted_numeric",
    "swap_count",
    "compare_count",
    "comparison_count",
    "work_count",
    "energy",
    "invalid_action_count",
)

TEXT_COUNT_COLUMNS = ("support_level", "evidence_level", "classification", "interpretation", "claim_family")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--s01-world-inventory", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e07_world_inventory.csv")
    parser.add_argument("--s02-policy-table", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e07_policy_representations.parquet")
    parser.add_argument("--s03-goal-table", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e07_goal_representations.parquet")
    parser.add_argument("--s03-world-goal-links", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e07_world_goal_links.parquet")
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.json_normalize(payload)
    raise ValueError(f"Unsupported table format: {path}")


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = __import__("hashlib").sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(elapsed),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_path(path),
        "sizeBytes": path.stat().st_size if path.is_file() else sum(child.stat().st_size for child in path.rglob("*") if child.is_file()),
        "artifactType": "directory" if path.is_dir() else "file",
    }


def self_referential_artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "artifactType": "file",
        "note": "Checksum omitted because this report or manifest contains the artifact list.",
    }


def parse_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def as_list(value: Any) -> list[str]:
    parsed = parse_json_value(value)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    if isinstance(parsed, dict):
        return [str(item) for item in parsed.keys()]
    return [str(parsed)] if str(parsed) else []


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def choose_first(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    for column in columns:
        value = clean_text(row.get(column))
        if value:
            return value
    return ""


def infer_metric_unit(metric_name: str) -> str:
    name = metric_name.lower()
    if "percent" in name:
        return "percent"
    if any(token in name for token in ("fraction", "ratio", "score", "quality", "sortedness", "entropy", "margin", "p_upper", "p_value", "curvature")):
        return "unitless_ratio_or_score"
    if any(token in name for token in ("count", "steps", "events", "swaps", "compare", "comparison", "work", "wait", "update", "attempt")):
        return "count"
    if "energy" in name or "cost" in name:
        return "cost"
    if "seconds" in name:
        return "seconds"
    if "error" in name or "distance" in name:
        return "proxy_distance_or_error"
    return "numeric"


def infer_metric_direction(metric_name: str) -> str:
    name = metric_name.lower()
    if any(token in name for token in ("error", "cost", "invalid", "rejected", "timeout", "unrecovered", "conflict_index", "interference", "impairment", "damage", "fatigue", "monotonicity_error")):
        return "minimize"
    if any(token in name for token in ("sortedness", "quality", "score", "recovery", "aggregation", "integration", "competency", "fitness", "success", "efficiency", "cohesion")):
        return "maximize"
    return "descriptive"


def infer_metric_family(metric_name: str) -> str:
    name = metric_name.lower()
    if "sortedness" in name or "monotonicity" in name:
        return "sorting_order"
    if "aggregation" in name or "neighbor" in name:
        return "aggregation"
    if name.startswith("dg") or "_dg" in name or "delayed" in name:
        return "delayed_gratification"
    if "recovery" in name or "repair" in name or "regeneration" in name:
        return "repair_recovery"
    if "target" in name or "error" in name or "distance" in name:
        return "target_match"
    if "energy" in name or "work" in name or "swap" in name or "compare" in name or "wait" in name or "update" in name:
        return "work_efficiency"
    if "compatibility" in name or "conflict" in name or "dominance" in name or "integration" in name or "interface" in name:
        return "chimeric_governance"
    if "observed_numeric" in name or "predicted_numeric" in name or "prediction" in name:
        return "prediction"
    if "count" in name:
        return "count"
    return "general_metric"


def is_metric_column(column: str, df: pd.DataFrame) -> bool:
    if column not in df.select_dtypes(include="number").columns:
        return False
    name = column.lower()
    if any(pattern in name for pattern in EXCLUDE_NUMERIC_PATTERNS):
        return False
    return any(pattern in name for pattern in INCLUDE_METRIC_PATTERNS)


def build_world_indexes(world_inventory: pd.DataFrame) -> dict[str, Any]:
    task_to_world: dict[str, str] = {}
    artifact_to_world: dict[str, list[str]] = defaultdict(list)
    step_to_world: dict[tuple[str, str], str] = {}
    for row in world_inventory.to_dict(orient="records"):
        world_id = str(row["world_id"])
        task_id = clean_text(row.get("task_id"))
        experiment_id = clean_text(row.get("experiment_id"))
        source_step = clean_text(row.get("source_step_id"))
        if task_id:
            task_to_world[task_id] = world_id
            task_to_world[normalize_key(task_id)] = world_id
        for step in [part.strip() for part in source_step.split(",") if part.strip()]:
            step_to_world[(experiment_id, step)] = world_id
        for artifact in as_list(row.get("source_artifacts_json")):
            artifact_to_world[artifact].append(world_id)
    return {"task": task_to_world, "artifact": artifact_to_world, "step": step_to_world}


def resolve_world(row: Mapping[str, Any], source_experiment_id: str, source_path: Path, default_step: str | None, indexes: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    task_to_world: Mapping[str, str] = indexes["task"]
    artifact_to_world: Mapping[str, list[str]] = indexes["artifact"]
    step_to_world: Mapping[tuple[str, str], str] = indexes["step"]

    path_worlds = artifact_to_world.get(str(source_path), [])
    if source_experiment_id == "E02" and len(path_worlds) == 1:
        return path_worlds[0], "resolved_by_source_artifact", {"source_artifact_path": str(source_path)}

    task_columns = ("benchmark_task_id", "target_id", "task_id")
    if source_experiment_id == "E01":
        task_columns = ("condition_id", "source_condition_id", *task_columns)
    elif source_experiment_id == "E05":
        task_columns = ("benchmark_task_id", "target_id", "task_id")

    for column in task_columns:
        value = clean_text(row.get(column))
        if value in task_to_world:
            return task_to_world[value], f"resolved_by_{column}", {column: value}
        normalized = normalize_key(value)
        if normalized in task_to_world:
            return task_to_world[normalized], f"resolved_by_normalized_{column}", {column: value}

    if len(path_worlds) == 1:
        return path_worlds[0], "resolved_by_source_artifact", {"source_artifact_path": str(source_path)}
    if len(path_worlds) > 1:
        return "", "unkeyed_missing_world", {"reason": "source_artifact_matches_multiple_worlds", "candidate_world_ids": path_worlds}

    for column in ("research_step_id", "source_research_step_id"):
        step = clean_text(row.get(column))
        if (source_experiment_id, step) in step_to_world:
            return step_to_world[(source_experiment_id, step)], f"resolved_by_{column}", {column: step}

    if default_step and (source_experiment_id, default_step) in step_to_world:
        return step_to_world[(source_experiment_id, default_step)], "resolved_by_table_default_step", {"default_step": default_step}

    return "", "unkeyed_missing_world", {"reason": "no_s01_world_key_found"}


def build_policy_indexes(policy_table: pd.DataFrame) -> dict[str, Any]:
    by_source_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_algorithm: dict[str, dict[str, Any]] = {}
    for row in policy_table.sort_values(["source_experiment_id", "source_step_id", "source_record_id"]).to_dict(orient="records"):
        source_policy_id = clean_text(row.get("source_policy_id"))
        if source_policy_id:
            by_source_policy[source_policy_id].append(row)
        algorithm = clean_text(row.get("algorithm"))
        if row.get("source_experiment_id") == "original_repo" and algorithm:
            by_algorithm[algorithm] = row
    return {"source_policy": by_source_policy, "algorithm": by_algorithm}


def choose_policy_candidate(candidates: Sequence[Mapping[str, Any]], source_experiment_id: str, source_step_id: str) -> Mapping[str, Any] | None:
    if not candidates:
        return None
    for candidate in candidates:
        if clean_text(candidate.get("source_experiment_id")) == source_experiment_id and clean_text(candidate.get("source_step_id")) == source_step_id:
            return candidate
    for candidate in candidates:
        if clean_text(candidate.get("source_experiment_id")) == source_experiment_id:
            return candidate
    return candidates[0]


def resolve_policy(row: Mapping[str, Any], source_experiment_id: str, indexes: Mapping[str, Any]) -> tuple[str, str, str, str, dict[str, Any]]:
    by_source_policy: Mapping[str, list[dict[str, Any]]] = indexes["source_policy"]
    by_algorithm: Mapping[str, dict[str, Any]] = indexes["algorithm"]
    source_step = clean_text(row.get("research_step_id")) or clean_text(row.get("source_research_step_id"))

    raw_policy_values: list[str] = []
    for column in ("policy_id", "original_policy_id", "variant_policy_id", "s08_source_policy_id", "leftmost_policy_id", "source_policy_id"):
        value = clean_text(row.get(column))
        if value:
            raw_policy_values.append(value)
    for column in ("policy_ids_json", "parent_ids_json"):
        raw_policy_values.extend(as_list(row.get(column)))

    for source_policy_id in raw_policy_values:
        candidate = choose_policy_candidate(by_source_policy.get(source_policy_id, []), source_experiment_id, source_step)
        if candidate:
            status = "resolved_first_of_multiple" if len(raw_policy_values) > 1 else "resolved"
            return (
                source_policy_id,
                clean_text(candidate.get("policy_uid")),
                clean_text(candidate.get("canonical_policy_id")),
                status,
                {"candidate_policy_ids": raw_policy_values},
            )

    algorithm = clean_text(row.get("algorithm"))
    if algorithm in by_algorithm:
        candidate = by_algorithm[algorithm]
        return (
            clean_text(candidate.get("source_policy_id")),
            clean_text(candidate.get("policy_uid")),
            clean_text(candidate.get("canonical_policy_id")),
            "resolved_by_original_algorithm",
            {"algorithm": algorithm},
        )

    return "", "", "", "unkeyed_missing_policy", {"candidate_policy_ids": raw_policy_values, "algorithm": algorithm, "reason": "no_s02_policy_key_found"}


def build_goal_indexes(goal_table: pd.DataFrame, world_goal_links: pd.DataFrame) -> dict[str, Any]:
    goal_by_uid = {str(row["goal_uid"]): row for row in goal_table.to_dict(orient="records")}
    links_by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in world_goal_links.to_dict(orient="records"):
        links_by_world[str(row["world_id"])].append(row)
    for rows in links_by_world.values():
        rows.sort(key=lambda item: (0 if clean_text(item.get("link_role")) == "primary_or_secondary" else 1, clean_text(item.get("source_goal_id"))))
    return {"goal_by_uid": goal_by_uid, "links_by_world": links_by_world}


def metric_matches_goal(metric_name: str, primary_metric_id: str) -> bool:
    metric_norm = normalize_key(metric_name)
    for part in re.split(r"[;,]", primary_metric_id):
        part_norm = normalize_key(part)
        if part_norm and (part_norm in metric_norm or metric_norm in part_norm):
            return True
    return False


def resolve_goal(world_id: str, metric_name: str, indexes: Mapping[str, Any]) -> tuple[str, str, str, str, dict[str, Any]]:
    if not world_id:
        return "", "", "", "world_unkeyed", {"reason": "goal link skipped because world_id is blank"}
    links = indexes["links_by_world"].get(world_id, [])
    if not links:
        return "", "", "", "unkeyed_missing_goal", {"reason": "no_s03_goal_link_for_world"}
    for link in links:
        if metric_matches_goal(metric_name, clean_text(link.get("primary_metric_id"))):
            return (
                clean_text(link.get("source_goal_id")),
                clean_text(link.get("goal_uid")),
                clean_text(link.get("canonical_goal_id")),
                "resolved_by_metric_contract",
                {"primary_metric_id": clean_text(link.get("primary_metric_id"))},
            )
    link = links[0]
    return (
        clean_text(link.get("source_goal_id")),
        clean_text(link.get("goal_uid")),
        clean_text(link.get("canonical_goal_id")),
        "resolved_by_world_primary_goal_proxy",
        {"candidate_goal_count": len(links), "primary_metric_id": clean_text(link.get("primary_metric_id"))},
    )


def source_record_id(row: Mapping[str, Any], source_table: str, row_index: int) -> str:
    value = choose_first(
        row,
        (
            "result_id",
            "score_id",
            "condition_id",
            "source_condition_id",
            "benchmark_task_id",
            "target_id",
            "run_id",
            "run_uid",
            "prediction_row_id",
            "policy_id",
            "original_policy_id",
            "variant_policy_id",
            "research_step_id",
        ),
    )
    return f"{source_table}:{value or f'row{row_index}'}"


def seed_value(row: Mapping[str, Any]) -> str:
    values = {}
    for column in ("simulation_seed", "seed", "scheduler_seed", "activation_seed", "policy_seed", "schedule_seed", "null_seed", "initial_array_seed"):
        value = clean_text(row.get(column))
        if value:
            values[column] = value
    if not values:
        return ""
    if len(values) == 1:
        return next(iter(values.values()))
    return stable_json(values)


def repeat_value(row: Mapping[str, Any]) -> str:
    return choose_first(row, ("repeat_index", "replicate_id", "null_replicate", "split_index"))


def perturbation_type(row: Mapping[str, Any]) -> str:
    return choose_first(
        row,
        (
            "perturbation_type",
            "repair_rule",
            "control_type",
            "condition_kind",
            "candidate_reason",
            "task_type",
            "placement_policy",
            "null_policy",
            "intervention_spec_id",
        ),
    )


def source_columns_payload(row: Mapping[str, Any], metric_name: str) -> dict[str, Any]:
    keys = {
        metric_name,
        "experiment_id",
        "research_step_id",
        "source_research_step_id",
        "condition_id",
        "source_condition_id",
        "benchmark_task_id",
        "target_id",
        "task_id",
        "policy_id",
        "policy_ids_json",
        "algorithm",
        "mode",
        "mode_or_mix",
        "goal_profile_id",
        "goal_compatibility_class",
        "metric_name",
        "metric_category",
        "classification",
        "support_level",
        "evidence_level",
    }
    return {key: row.get(key) for key in sorted(keys) if key in row}


def emit_numeric_records(
    df: pd.DataFrame,
    *,
    source_experiment_id: str,
    source_table: str,
    source_path: Path,
    source_sha: str,
    row_granularity: str,
    default_step: str | None,
    world_indexes: Mapping[str, Any],
    policy_indexes: Mapping[str, Any],
    goal_indexes: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    metric_columns: list[str] = []
    long_metric = {"metric", "value"}.issubset(df.columns) or {"metric_id", "value"}.issubset(df.columns)
    metric_label_column = "metric" if "metric" in df.columns else "metric_id"
    if long_metric:
        metric_columns = ["value"]
    else:
        metric_columns = [column for column in df.columns if is_metric_column(column, df)]

    for row_index, row in enumerate(df.to_dict(orient="records")):
        world_id, world_status, world_detail = resolve_world(row, source_experiment_id, source_path, default_step, world_indexes)
        source_policy_id, policy_uid, canonical_policy_id, policy_status, policy_detail = resolve_policy(row, source_experiment_id, policy_indexes)
        items: list[tuple[str, Any, str]] = []
        if long_metric:
            items.append((clean_text(row.get(metric_label_column)) or "value", row.get("value"), clean_text(row.get("unit"))))
        else:
            items = [(column, row.get(column), "") for column in metric_columns]
        for metric_name, metric_value, explicit_unit in items:
            numeric_value = coerce_float(metric_value)
            if numeric_value is None:
                continue
            source_goal_id, goal_uid, canonical_goal_id, goal_status, goal_detail = resolve_goal(world_id, metric_name, goal_indexes)
            trace_refs = [path for path in TRACE_ARTIFACTS.get(source_experiment_id, ()) if Path(path).exists()]
            missingness_payload = {}
            if not world_id:
                missingness_payload["world"] = world_detail
            if not policy_uid:
                missingness_payload["policy"] = policy_detail
            if not goal_uid:
                missingness_payload["goal"] = goal_detail
            records.append(
                {
                    "schema_version": CORPUS_SCHEMA_VERSION,
                    "source_experiment_id": source_experiment_id,
                    "source_artifact_path": str(source_path),
                    "source_artifact_sha256": source_sha,
                    "source_table": source_table,
                    "source_row_index": row_index,
                    "source_record_id": source_record_id(row, source_table, row_index),
                    "source_metric_name": metric_name,
                    "metric_value": numeric_value,
                    "metric_unit": explicit_unit or infer_metric_unit(metric_name),
                    "metric_direction": infer_metric_direction(metric_name),
                    "metric_family": infer_metric_family(metric_name),
                    "evidence_kind": "metric_observation",
                    "row_granularity": row_granularity,
                    "world_id": world_id,
                    "world_link_status": world_status,
                    "source_policy_id": source_policy_id,
                    "policy_uid": policy_uid,
                    "canonical_policy_id": canonical_policy_id,
                    "policy_link_status": policy_status,
                    "source_goal_id": source_goal_id,
                    "goal_uid": goal_uid,
                    "canonical_goal_id": canonical_goal_id,
                    "goal_link_status": goal_status,
                    "perturbation_type": perturbation_type(row),
                    "seed": seed_value(row),
                    "repeat_index": repeat_value(row),
                    "trace_artifacts_json": trace_refs,
                    "missingness_json": missingness_payload,
                    "source_columns_json": source_columns_payload(row, metric_name),
                }
            )
    return records, metric_columns


def emit_text_count_records(
    df: pd.DataFrame,
    *,
    source_experiment_id: str,
    source_table: str,
    source_path: Path,
    source_sha: str,
    world_indexes: Mapping[str, Any],
    goal_indexes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected_columns = [column for column in TEXT_COUNT_COLUMNS if column in df.columns]
    if not selected_columns:
        return records
    world_id, world_status, world_detail = resolve_world({}, source_experiment_id, source_path, None, world_indexes)
    for column in selected_columns:
        counts = Counter(clean_text(value) or "blank" for value in df[column].tolist())
        for index, (label, count) in enumerate(sorted(counts.items())):
            metric_name = f"{column}_count:{normalize_key(label) or 'blank'}"
            source_goal_id, goal_uid, canonical_goal_id, goal_status, goal_detail = resolve_goal(world_id, metric_name, goal_indexes)
            missingness_payload = {}
            if not world_id:
                missingness_payload["world"] = world_detail
            if not goal_uid:
                missingness_payload["goal"] = goal_detail
            records.append(
                {
                    "schema_version": CORPUS_SCHEMA_VERSION,
                    "source_experiment_id": source_experiment_id,
                    "source_artifact_path": str(source_path),
                    "source_artifact_sha256": source_sha,
                    "source_table": source_table,
                    "source_row_index": index,
                    "source_record_id": f"{source_table}:{column}:{normalize_key(label) or 'blank'}",
                    "source_metric_name": metric_name,
                    "metric_value": float(count),
                    "metric_unit": "row count",
                    "metric_direction": "descriptive",
                    "metric_family": "audit_category_count",
                    "evidence_kind": "metric_observation",
                    "row_granularity": "audit_category_count",
                    "world_id": world_id,
                    "world_link_status": world_status,
                    "source_policy_id": "",
                    "policy_uid": "",
                    "canonical_policy_id": "",
                    "policy_link_status": "not_applicable",
                    "source_goal_id": source_goal_id,
                    "goal_uid": goal_uid,
                    "canonical_goal_id": canonical_goal_id,
                    "goal_link_status": goal_status,
                    "perturbation_type": column,
                    "seed": "",
                    "repeat_index": "",
                    "trace_artifacts_json": [],
                    "missingness_json": missingness_payload,
                    "source_columns_json": {column: label, "count": count},
                }
            )
    return records


def missingness_id(source_table: str, missingness_type: str, detail: str) -> str:
    return f"miss:{stable_hash({'sourceTable': source_table, 'type': missingness_type, 'detail': detail})[:20]}"


def source_missingness_rows(source_manifest: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_row in source_manifest.to_dict(orient="records"):
        source_table = clean_text(manifest_row.get("source_table"))
        source_path = clean_text(manifest_row.get("source_artifact_path"))
        source_exp = clean_text(manifest_row.get("source_experiment_id"))
        subset = corpus[corpus["source_table"] == source_table] if not corpus.empty else corpus
        if clean_text(manifest_row.get("ingest_status")) == "missing":
            detail = "Expected source artifact was unavailable."
            rows.append(
                {
                    "missingness_id": missingness_id(source_table, "missing_source_artifact", detail),
                    "source_experiment_id": source_exp,
                    "source_artifact_path": source_path,
                    "source_table": source_table,
                    "missingness_type": "missing_source_artifact",
                    "affected_rows": 0,
                    "affected_metric_rows": 0,
                    "detail": detail,
                    "documented_limitation": "Source file absent from previous-artifacts mount.",
                }
            )
            continue
        if subset.empty:
            detail = "Available source had no numeric or count metrics selected for corpus rows."
            rows.append(
                {
                    "missingness_id": missingness_id(source_table, "no_selected_metric_rows", detail),
                    "source_experiment_id": source_exp,
                    "source_artifact_path": source_path,
                    "source_table": source_table,
                    "missingness_type": "no_selected_metric_rows",
                    "affected_rows": int(manifest_row.get("row_count") or 0),
                    "affected_metric_rows": 0,
                    "detail": detail,
                    "documented_limitation": "Source is metadata/text-only or all candidate metric values were nonnumeric.",
                }
            )
            continue
        for key, status_column, id_column in (
            ("unkeyed_world", "world_link_status", "world_id"),
            ("unkeyed_policy", "policy_link_status", "policy_uid"),
            ("unkeyed_goal", "goal_link_status", "goal_uid"),
        ):
            affected = subset[subset[id_column].astype(str) == ""]
            if affected.empty:
                continue
            detail = f"{len(affected)} metric rows have blank {id_column} with statuses {dict(Counter(affected[status_column].astype(str)))}."
            rows.append(
                {
                    "missingness_id": missingness_id(source_table, key, detail),
                    "source_experiment_id": source_exp,
                    "source_artifact_path": source_path,
                    "source_table": source_table,
                    "missingness_type": key,
                    "affected_rows": int(affected["source_row_index"].nunique()),
                    "affected_metric_rows": int(len(affected)),
                    "detail": detail,
                    "documented_limitation": "Upstream row did not expose a key that could be linked to the S01/S02/S03 tables without over-inference.",
                }
            )
    return pd.DataFrame(rows, columns=list(MISSINGNESS_COLUMNS))


def build_corpus(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    world_inventory = pd.read_csv(args.s01_world_inventory)
    policy_table = pd.read_parquet(args.s02_policy_table)
    goal_table = pd.read_parquet(args.s03_goal_table)
    world_goal_links = pd.read_parquet(args.s03_world_goal_links)

    world_indexes = build_world_indexes(world_inventory)
    policy_indexes = build_policy_indexes(policy_table)
    goal_indexes = build_goal_indexes(goal_table, world_goal_links)

    all_records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for source_experiment_id, source_table, source_path_text, row_granularity, default_step in SOURCE_SPECS:
        print(f"[S04] ingesting {source_table}", file=sys.stderr, flush=True)
        source_path = Path(source_path_text)
        if not source_path.exists():
            manifest_rows.append(
                {
                    "source_experiment_id": source_experiment_id,
                    "source_table": source_table,
                    "source_artifact_path": str(source_path),
                    "source_artifact_sha256": "",
                    "row_count": 0,
                    "column_count": 0,
                    "selected_metric_column_count": 0,
                    "metric_observation_rows": 0,
                    "world_keyed_rows": 0,
                    "world_unkeyed_rows": 0,
                    "policy_keyed_rows": 0,
                    "policy_unkeyed_rows": 0,
                    "goal_keyed_rows": 0,
                    "goal_unkeyed_rows": 0,
                    "ingest_status": "missing",
                }
            )
            continue
        source_sha = sha256_file(source_path)
        df = read_table(source_path)
        records, metric_columns = emit_numeric_records(
            df,
            source_experiment_id=source_experiment_id,
            source_table=source_table,
            source_path=source_path,
            source_sha=source_sha,
            row_granularity=row_granularity,
            default_step=default_step,
            world_indexes=world_indexes,
            policy_indexes=policy_indexes,
            goal_indexes=goal_indexes,
        )
        if not records:
            records = emit_text_count_records(
                df,
                source_experiment_id=source_experiment_id,
                source_table=source_table,
                source_path=source_path,
                source_sha=source_sha,
                world_indexes=world_indexes,
                goal_indexes=goal_indexes,
            )
        all_records.extend(records)
        world_keyed = sum(1 for record in records if clean_text(record.get("world_id")))
        policy_keyed = sum(1 for record in records if clean_text(record.get("policy_uid")))
        goal_keyed = sum(1 for record in records if clean_text(record.get("goal_uid")))
        manifest_rows.append(
            {
                "source_experiment_id": source_experiment_id,
                "source_table": source_table,
                "source_artifact_path": str(source_path),
                "source_artifact_sha256": source_sha,
                "row_count": len(df),
                "column_count": len(df.columns),
                "selected_metric_column_count": len(metric_columns) if metric_columns else len({record["source_metric_name"] for record in records}),
                "metric_observation_rows": len(records),
                "world_keyed_rows": int(world_keyed),
                "world_unkeyed_rows": int(len(records) - world_keyed),
                "policy_keyed_rows": int(policy_keyed),
                "policy_unkeyed_rows": int(len(records) - policy_keyed),
                "goal_keyed_rows": int(goal_keyed),
                "goal_unkeyed_rows": int(len(records) - goal_keyed),
                "ingest_status": "ingested",
            }
        )
        print(f"[S04] {source_table}: {len(records)} metric rows", file=sys.stderr, flush=True)

    print(f"[S04] finalizing {len(all_records)} corpus records", file=sys.stderr, flush=True)
    corpus = dataframe_from_records(all_records)
    source_manifest = pd.DataFrame(manifest_rows, columns=list(SOURCE_MANIFEST_COLUMNS))
    missingness = source_missingness_rows(source_manifest, corpus)
    validation = validate_unified_corpus(
        corpus,
        missingness,
        source_manifest,
        s01_world_ids=world_inventory["world_id"].astype(str).tolist(),
        known_policy_ids=policy_table["policy_uid"].astype(str).tolist(),
        known_goal_ids=goal_table["goal_uid"].astype(str).tolist(),
    )
    return corpus, missingness, source_manifest, validation


def coverage_summary(corpus: pd.DataFrame, source_manifest: pd.DataFrame, world_inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_experiment_id, subset in corpus.groupby("source_experiment_id", dropna=False):
        manifest_subset = source_manifest[source_manifest["source_experiment_id"] == source_experiment_id]
        rows.append(
            {
                "source_experiment_id": source_experiment_id,
                "metric_observation_rows": int(len(subset)),
                "source_tables": int(manifest_subset["source_table"].nunique()),
                "source_files": int(manifest_subset["source_artifact_path"].nunique()),
                "s01_worlds_available": int((world_inventory["experiment_id"] == source_experiment_id).sum()),
                "worlds_with_metric_rows": int(subset.loc[subset["world_id"].astype(str) != "", "world_id"].nunique()),
                "world_keyed_metric_rows": int((subset["world_id"].astype(str) != "").sum()),
                "world_unkeyed_metric_rows": int((subset["world_id"].astype(str) == "").sum()),
                "policy_keyed_metric_rows": int((subset["policy_uid"].astype(str) != "").sum()),
                "policy_unkeyed_metric_rows": int((subset["policy_uid"].astype(str) == "").sum()),
                "goal_keyed_metric_rows": int((subset["goal_uid"].astype(str) != "").sum()),
                "goal_unkeyed_metric_rows": int((subset["goal_uid"].astype(str) == "").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("source_experiment_id").reset_index(drop=True)


def world_coverage(corpus: pd.DataFrame, world_inventory: pd.DataFrame) -> pd.DataFrame:
    counts = corpus[corpus["world_id"].astype(str) != ""].groupby("world_id").size().rename("metric_observation_rows").reset_index()
    result = world_inventory[["world_id", "experiment_id", "source_step_id", "task_id", "world_family"]].merge(counts, on="world_id", how="left")
    result["metric_observation_rows"] = result["metric_observation_rows"].fillna(0).astype(int)
    result["has_metric_rows"] = result["metric_observation_rows"] > 0
    return result


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    frame = df.head(max_rows).copy()
    columns = [str(column) for column in frame.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for record in frame.to_dict(orient="records"):
        values = [clean_text(record.get(column)).replace("|", "\\|").replace("\n", " ") for column in frame.columns]
        rows.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        rows.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(rows)


def build_coverage_report(
    *,
    path: Path,
    artifacts_written: Sequence[str],
    coverage: pd.DataFrame,
    world_cov: pd.DataFrame,
    missingness: pd.DataFrame,
    source_manifest: pd.DataFrame,
    validation: pd.DataFrame,
    corpus: pd.DataFrame,
) -> None:
    validation_success = bool(validation["success"].all())
    outcome = "supportive" if validation_success else "constraining/contradictory"
    caveats = (
        "Policy links are often absent for audit/summary rows that do not name a policy; E03/E04 step-level world links are coarser than per-config worlds; trace artifacts are referenced by path and hash rather than copied."
    )
    text = f"""# E07 S04 Corpus Coverage Report

## Top Summary

- Research step ID: S04
- Completion status: complete
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: {"pass" if validation_success else "fail"} ({int(validation['success'].sum())}/{len(validation)} checks passed)
- Outcome classification: {outcome}
- Caveats or blockers: {caveats}
- Recommended next action: Chief review, then proceed to S05 only if the unified corpus shape and documented missingness are accepted.

## Coverage by Source Experiment

{markdown_table(coverage)}

## Source Manifest Summary

{markdown_table(source_manifest)}

## Missingness Summary

{markdown_table(missingness.groupby(['source_experiment_id', 'missingness_type'], dropna=False).agg(affected_metric_rows=('affected_metric_rows', 'sum'), records=('missingness_id', 'count')).reset_index() if not missingness.empty else missingness)}

## World Coverage

Worlds with metric rows: {int(world_cov['has_metric_rows'].sum())}/{len(world_cov)}.

{markdown_table(world_cov.groupby(['experiment_id', 'has_metric_rows'], dropna=False).size().rename('world_count').reset_index())}

## Validation

{markdown_table(validation)}

## Corpus Shape

- Metric observation rows: {len(corpus)}
- Source tables represented: {corpus['source_table'].nunique() if not corpus.empty else 0}
- Metric names represented: {corpus['source_metric_name'].nunique() if not corpus.empty else 0}
- Non-empty world links: {int((corpus['world_id'].astype(str) != '').sum()) if not corpus.empty else 0}
- Non-empty policy links: {int((corpus['policy_uid'].astype(str) != '').sum()) if not corpus.empty else 0}
- Non-empty goal links: {int((corpus['goal_uid'].astype(str) != '').sum()) if not corpus.empty else 0}
"""
    write_text(path, text)


def build_full_report(
    *,
    path: Path,
    artifacts_written: Sequence[str],
    coverage: pd.DataFrame,
    world_cov: pd.DataFrame,
    missingness: pd.DataFrame,
    source_manifest: pd.DataFrame,
    validation: pd.DataFrame,
    corpus: pd.DataFrame,
    unit_test_result: Mapping[str, Any] | None,
    repo_dir: Path,
    artifacts_dir: Path,
    source_hashes: Mapping[str, str],
) -> None:
    validation_success = bool(validation["success"].all())
    outcome = "supportive" if validation_success else "constraining/contradictory"
    caveats = (
        "Some upstream rows lack direct policy keys; E03/E04/E06 world links are step-level where S01 modeled those experiments by step/config rather than by every run condition; large trace files were not copied into S04 artifacts."
    )
    test_line = "not run"
    if unit_test_result is not None:
        test_line = f"{'pass' if unit_test_result['success'] else 'fail'}: `{unit_test_result['command']}` return code {unit_test_result['returnCode']}"
    text = f"""# E07 S04 Full Results: Unified Behavior Corpus

## Top Summary

- Research step ID: S04
- Completion status: complete
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: {"pass" if validation_success else "fail"} ({int(validation['success'].sum())}/{len(validation)} checks passed)
- Outcome classification: {outcome}
- Caveats or blockers: {caveats}
- Lay summary: S04 assembled a single machine-readable table that puts upstream behavior, audit, benchmark, and governance metrics from E01-E06 into one row-per-metric format, using S01 world identifiers, S02 policy identifiers, and S03 goal links where the upstream rows expose enough keys.
- Recommended next action: Chief review of coverage and missingness, then proceed to S05 only after acceptance.

## Frozen Question

Can available E01-E06 behavior metrics and trace summaries be merged into a unified E07 corpus keyed to the S01 world inventory, S02 policy representations, and S03 goal representations, while explicitly documenting rows that cannot be keyed?

## Inputs

- S01 world inventory: `/artifacts/tables/e07_world_inventory.csv`
- S02 policy table: `/artifacts/tables/e07_policy_representations.parquet`
- S03 goal table: `/artifacts/tables/e07_goal_representations.parquet`
- S03 world-goal links: `/artifacts/tables/e07_world_goal_links.parquet`
- Upstream previous artifacts: `/previous-artifacts/E01` through `/previous-artifacts/E06`

## Methods

The builder read selected CSV and Parquet metric artifacts from E01-E06 and emitted one corpus row per numeric metric observation. Long metric/value tables were preserved directly; wide result tables were normalized across metric-like numeric columns; text-only E02 audit tables were summarized as categorical count metrics. Each corpus row stores the upstream artifact path and SHA-256 hash, source table name, source row index, metric name/value/unit/direction, and compact JSON provenance for relevant source columns.

World links were resolved by exact S01 task IDs, exact S01 source-artifact paths, or source step IDs where S01 modeled the experiment at step granularity. Policy links were resolved from explicit policy IDs or policy ID lists where present, with E01 public algorithms mapped to original-repository policies. Goal links were resolved by S03 world-goal links and metric-contract matching when possible; otherwise the first world-level primary/proxy goal was recorded as a documented proxy. Blank world, policy, or goal links were retained with explicit missingness statuses and companion missingness records.

Trace artifacts were not copied into S04 outputs. Relevant trace-like upstream files are referenced by path in corpus rows and in the source manifest so S04 remains a compact final evidence layer.

## Commands

- `python -m unittest tests.e07.test_corpus_schema`
- `python scripts/e07_s04_build_unified_corpus.py`

Unit-test result: {test_line}

## Dependencies and Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- Repository commit before S04 commit: `{git_output(repo_dir, ['rev-parse', 'HEAD'])}`
- Branch: `{git_output(repo_dir, ['branch', '--show-current'])}`
- Worker count: serial execution; no CPU parallelism was needed for table normalization.

## Results

Corpus shape:

- Metric observation rows: {len(corpus)}
- Source experiments represented: {corpus['source_experiment_id'].nunique() if not corpus.empty else 0}
- Source tables represented: {corpus['source_table'].nunique() if not corpus.empty else 0}
- Metric names represented: {corpus['source_metric_name'].nunique() if not corpus.empty else 0}
- S01 worlds with metric rows: {int(world_cov['has_metric_rows'].sum())}/{len(world_cov)}
- Rows with S01 world links: {int((corpus['world_id'].astype(str) != '').sum()) if not corpus.empty else 0}
- Rows with S02 policy links: {int((corpus['policy_uid'].astype(str) != '').sum()) if not corpus.empty else 0}
- Rows with S03 goal links: {int((corpus['goal_uid'].astype(str) != '').sum()) if not corpus.empty else 0}

### Coverage by Source

{markdown_table(coverage)}

### World Coverage

{markdown_table(world_cov.groupby(['experiment_id', 'has_metric_rows'], dropna=False).size().rename('world_count').reset_index())}

### Missingness

{markdown_table(missingness.groupby(['source_experiment_id', 'missingness_type'], dropna=False).agg(affected_metric_rows=('affected_metric_rows', 'sum'), records=('missingness_id', 'count')).reset_index() if not missingness.empty else missingness)}

## Validation

{markdown_table(validation)}

Validation outcome: {"all checks passed" if validation_success else "one or more checks failed; inspect validation table"}.

## Output Artifacts

{chr(10).join(f'- `{item}`' for item in artifacts_written)}

## Provenance

Upstream hashes were recorded in `/artifacts/tables/e07_corpus_source_manifest.csv` and in the S04 artifact manifest. Source files ingested: {len(source_hashes)}.

Selected source-hash sample:

{markdown_table(pd.DataFrame([{'source_artifact_path': path, 'sha256': sha} for path, sha in list(source_hashes.items())[:20]]))}

## Caveats, Blockers, and Limitations

- Policy coverage is intentionally sparse for aggregate audit tables, null summaries, and prediction tables that do not name a policy. These rows have blank `policy_uid` plus explicit missingness records rather than inferred policy links.
- Some world links are step-level proxies because S01 represented E03, E04, and E06 worlds at step/config granularity, while the upstream metric rows may represent many lower-level conditions.
- Goal links are metric-contract matches where possible and world-primary proxies otherwise; S05 should treat proxy goal links as analysis covariates, not as proof that every metric is a direct target variable.
- Trace files and large upstream result files remain in `/previous-artifacts`; S04 records their paths and hashes but does not duplicate them under `/artifacts`.
- No web or package installation was used.

## Recommended Next Action

Chief review should inspect the corpus shape, coverage by source, and missingness summaries. If accepted, proceed to S05; do not start S05 until explicitly instructed.
"""
    write_text(path, text)


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    reports_dir = artifacts_dir / "reports"
    tables_dir = artifacts_dir / "tables"
    for directory in (step_dir, results_dir, reports_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_test_result = None
    if args.run_unit_tests:
        unit_test_result = run_command([sys.executable, "-m", "unittest", "tests.e07.test_corpus_schema"], args.repo_dir)

    corpus, missingness, source_manifest, validation = build_corpus(args)
    world_inventory = pd.read_csv(args.s01_world_inventory)
    coverage = coverage_summary(corpus, source_manifest, world_inventory)
    world_cov = world_coverage(corpus, world_inventory)

    corpus_path = results_dir / "e07_unified_behavior_corpus.parquet"
    sample_path = tables_dir / "e07_unified_behavior_corpus_sample.csv"
    coverage_path = tables_dir / "e07_corpus_coverage_summary.csv"
    world_coverage_path = tables_dir / "e07_corpus_world_coverage.csv"
    missingness_path = tables_dir / "e07_corpus_missingness_records.csv"
    source_manifest_path = tables_dir / "e07_corpus_source_manifest.csv"
    validation_path = step_dir / "e07_s04_validation_checks.csv"
    coverage_report_path = reports_dir / "e07_corpus_coverage_report.md"
    full_report_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"

    corpus.to_parquet(corpus_path, index=False)
    corpus.head(500).to_csv(sample_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    world_cov.to_csv(world_coverage_path, index=False)
    missingness.to_csv(missingness_path, index=False)
    source_manifest.to_csv(source_manifest_path, index=False)
    validation.to_csv(validation_path, index=False)

    source_hashes = {
        clean_text(row["source_artifact_path"]): clean_text(row["source_artifact_sha256"])
        for row in source_manifest.to_dict(orient="records")
        if clean_text(row.get("source_artifact_sha256"))
    }
    artifacts_written = [
        str(corpus_path),
        str(sample_path),
        str(coverage_path),
        str(world_coverage_path),
        str(missingness_path),
        str(source_manifest_path),
        str(validation_path),
        str(coverage_report_path),
        str(full_report_path),
        str(artifact_manifest_path),
    ]

    build_coverage_report(
        path=coverage_report_path,
        artifacts_written=artifacts_written,
        coverage=coverage,
        world_cov=world_cov,
        missingness=missingness,
        source_manifest=source_manifest,
        validation=validation,
        corpus=corpus,
    )
    build_full_report(
        path=full_report_path,
        artifacts_written=artifacts_written,
        coverage=coverage,
        world_cov=world_cov,
        missingness=missingness,
        source_manifest=source_manifest,
        validation=validation,
        corpus=corpus,
        unit_test_result=unit_test_result,
        repo_dir=args.repo_dir,
        artifacts_dir=artifacts_dir,
        source_hashes=source_hashes,
    )

    manifest_payload = {
        "schemaVersion": "eidosoma.e07.s04.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "success": bool(validation["success"].all()) and (unit_test_result is None or bool(unit_test_result["success"])),
        "validationResult": {
            "passed": int(validation["success"].sum()),
            "total": int(len(validation)),
            "allPassed": bool(validation["success"].all()),
        },
        "unitTestResult": unit_test_result,
        "artifacts": [
            artifact_entry(corpus_path, artifacts_dir, "Unified E07 metric-observation corpus."),
            artifact_entry(sample_path, artifacts_dir, "Small CSV preview of the unified corpus."),
            artifact_entry(coverage_path, artifacts_dir, "Coverage summary by source experiment."),
            artifact_entry(world_coverage_path, artifacts_dir, "Coverage summary by S01 world."),
            artifact_entry(missingness_path, artifacts_dir, "Explicit missingness records for unkeyed/unavailable source data."),
            artifact_entry(source_manifest_path, artifacts_dir, "Source artifact manifest with SHA-256 hashes."),
            artifact_entry(validation_path, artifacts_dir, "S04 validation checks."),
            artifact_entry(coverage_report_path, artifacts_dir, "Markdown coverage report."),
            self_referential_artifact_entry(full_report_path, artifacts_dir, "S04 full-results report."),
            self_referential_artifact_entry(artifact_manifest_path, artifacts_dir, "S04 artifact manifest."),
        ],
        "sourceArtifacts": [
            {
                "sourceExperimentId": row["source_experiment_id"],
                "sourceTable": row["source_table"],
                "sourceArtifactPath": row["source_artifact_path"],
                "sha256": row["source_artifact_sha256"],
                "rowCount": int(row["row_count"]),
                "metricObservationRows": int(row["metric_observation_rows"]),
                "ingestStatus": row["ingest_status"],
            }
            for row in source_manifest.to_dict(orient="records")
        ],
    }
    write_json(artifact_manifest_path, manifest_payload)

    # Refresh the self-referential manifest entry sizes after writing it.
    manifest_payload["artifacts"][-1]["sizeBytes"] = artifact_manifest_path.stat().st_size
    write_json(artifact_manifest_path, manifest_payload)


if __name__ == "__main__":
    main()
