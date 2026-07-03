#!/usr/bin/env python3
"""Run E07 S12 scoped E03 DSL executable-proxy inverse design."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e03.coarse_sweep import actor_schedule, initial_values, sortedness_metrics  # noqa: E402
from src.e03.policy_generation import policy_feature_flags, semantic_hash  # noqa: E402
from src.e03.rule_dsl import DSLAction, DSLArrayState, DSLCondition, DSLInterpreter, DSLPolicy, DSLRule, parse_policy, validate_policy  # noqa: E402
from src.e07.corpus_schema import sha256_file  # noqa: E402
from src.e07.inverse_design_schema import (  # noqa: E402
    INVERSE_DESIGN_SCHEMA_VERSION,
    design_record_id,
    e03_proxy_target_score,
    target_profile_id,
    validate_inverse_design_artifacts,
    validation_summary,
)


STEP_ID = "S12"
STEP_NUMBER = 12
EXPERIMENT_ID = "E07"
RANDOM_SEED = 2026070312
TARGET_PROFILE_SLUG = "e03_array_proxy_balanced_order_energy_moderate_dg"
TARGET_SORTEDNESS = 0.90
TARGET_WORK_PER_ITEM = 3.50
TARGET_DG_RECOVERY_PROXY = 0.03
TARGET_DG_TOLERANCE = 0.08


@dataclass(frozen=True)
class S12Config:
    config_id: str
    world_id: str
    split: str
    array_size: int
    seed: int
    event_cap: int
    input_profile: str = "random_permutation"
    scheduler: str = "cyclic_scan_seed_offset"


def parse_args() -> argparse.Namespace:
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_dir)
    parser.add_argument("--e03-generated-policy-library", type=Path, default=Path("/previous-artifacts/E03/policies/e03_generated_policy_library.jsonl"))
    parser.add_argument("--e03-frontier-policy-library", type=Path, default=Path("/previous-artifacts/E03/policies/e03_frontier_candidate_policies.jsonl"))
    parser.add_argument("--e03-qd-policy-library", type=Path, default=Path("/previous-artifacts/E03/policies/e03_qd_discovered_policies.jsonl"))
    parser.add_argument("--e03-policy-competence", type=Path, default=Path("/previous-artifacts/E03/results/e03_policy_competence.parquet"))
    parser.add_argument("--e03-policy-clusters", type=Path, default=Path("/previous-artifacts/E03/results/e03_policy_clusters.parquet"))
    parser.add_argument("--e03-frontier-candidates", type=Path, default=Path("/previous-artifacts/E03/results/e03_frontier_candidates.parquet"))
    parser.add_argument("--s02-policy-table", type=Path, default=artifacts_dir / "tables" / "e07_policy_representations.parquet")
    parser.add_argument("--s05-modeling-dataset", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_modeling_dataset.parquet")
    parser.add_argument("--s05-linear-model", type=Path, default=artifacts_dir / "models" / "e07_behavior_predictor" / "hashed_linear_sgd_heldout_policy.pkl")
    parser.add_argument("--s10-classes", type=Path, default=artifacts_dir / "results" / "e07_universality_classes.parquet")
    parser.add_argument("--s10-assignments", type=Path, default=artifacts_dir / "results" / "e07_universality_assignments.parquet")
    parser.add_argument("--s11-predictions", type=Path, default=artifacts_dir / "results" / "e07_counterfactual_predictions.parquet")
    parser.add_argument("--s11-validations", type=Path, default=artifacts_dir / "results" / "e07_counterfactual_validations.parquet")
    parser.add_argument("--s11-freeze-manifest", type=Path, default=artifacts_dir / "research_steps" / "S11" / "candidate_freeze_manifest.json")
    parser.add_argument("--candidate-pool-size", type=int, default=72)
    parser.add_argument("--mutate-top-n", type=int, default=16)
    parser.add_argument("--designed-count", type=int, default=10)
    parser.add_argument("--control-count-per-family", type=int, default=3)
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


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False) + "\n")


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


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def maybe_sha256(path: Path) -> str:
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path) if path.is_file() else None,
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


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value).replace("\n", " ").replace("|", "\\|")


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        view[column] = view[column].map(format_cell)
    header = "| " + " | ".join(map(str, view.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    if len(frame) > max_rows:
        rows.append(f"| ... | {len(frame) - max_rows} more rows omitted |" + " |" * max(0, len(view.columns) - 2))
    return "\n".join([header, separator, *rows])


def target_profile_records(frozen_at: str) -> list[dict[str, Any]]:
    rows = [
        {
            "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "target_profile_slug": TARGET_PROFILE_SLUG,
            "target_axis_id": "order_quality_proxy",
            "target_axis_label": "High E03 array-order quality",
            "scope_status": "executable_proxy",
            "metric_name": "final_inversion_sortedness",
            "metric_direction": "maximize",
            "target_value": TARGET_SORTEDNESS,
            "target_tolerance": 0.10,
            "score_weight": 0.50,
            "limitation": "Single-policy E03 array sortedness proxy; not biological morphology.",
            "frozen_at_utc": frozen_at,
        },
        {
            "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "target_profile_slug": TARGET_PROFILE_SLUG,
            "target_axis_id": "energy_efficiency_proxy",
            "target_axis_label": "Low E03 DSL work per item",
            "scope_status": "executable_proxy",
            "metric_name": "work_count_per_item",
            "metric_direction": "minimize",
            "target_value": TARGET_WORK_PER_ITEM,
            "target_tolerance": 1.50,
            "score_weight": 0.25,
            "limitation": "Work count is compare plus swap plus state-update count, not biological energy.",
            "frozen_at_utc": frozen_at,
        },
        {
            "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "target_profile_slug": TARGET_PROFILE_SLUG,
            "target_axis_id": "moderate_dg_proxy",
            "target_axis_label": "Moderate E03 DG recovery proxy",
            "scope_status": "executable_proxy",
            "metric_name": "dg_recovery_proxy",
            "metric_direction": "target_band",
            "target_value": TARGET_DG_RECOVERY_PROXY,
            "target_tolerance": TARGET_DG_TOLERANCE,
            "score_weight": 0.25,
            "limitation": "DG proxy is sortedness recovery from within-trajectory trough, not evidence of planning or intention.",
            "frozen_at_utc": frozen_at,
        },
        {
            "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "target_profile_slug": TARGET_PROFILE_SLUG,
            "target_axis_id": "aggregation_blocker",
            "target_axis_label": "Low aggregation",
            "scope_status": "simulator_blocked_by_s12_scope",
            "metric_name": "aggregation_value",
            "metric_direction": "minimize",
            "target_value": None,
            "target_tolerance": None,
            "score_weight": 0.0,
            "limitation": "S12 is restricted to single-policy E03 array worlds and does not instantiate chimeric algotype aggregation.",
            "frozen_at_utc": frozen_at,
        },
        {
            "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "target_profile_slug": TARGET_PROFILE_SLUG,
            "target_axis_id": "repair_robustness_blocker",
            "target_axis_label": "High repair or damage robustness",
            "scope_status": "simulator_blocked_by_s12_scope",
            "metric_name": "repair_or_damage_recovery",
            "metric_direction": "maximize",
            "target_value": None,
            "target_tolerance": None,
            "score_weight": 0.0,
            "limitation": "S12 does not build new E05/E06 repair, regeneration, governance, or damage adapters; held-out E03 sizes/seeds are only an executable generalization proxy.",
            "frozen_at_utc": frozen_at,
        },
    ]
    profile_id = target_profile_id({"target_profile_slug": TARGET_PROFILE_SLUG, "rows": rows})
    for row in rows:
        row["target_profile_id"] = profile_id
    return rows


def training_configs() -> list[S12Config]:
    return [
        S12Config("s12_train_n16_seed6101", "world:e03:s12:train-array-n16-seed6101", "train", 16, 6101, 96),
        S12Config("s12_train_n16_seed6102", "world:e03:s12:train-array-n16-seed6102", "train", 16, 6102, 96),
        S12Config("s12_train_n32_seed6201", "world:e03:s12:train-array-n32-seed6201", "train", 32, 6201, 192),
    ]


def heldout_configs() -> list[S12Config]:
    return [
        S12Config("s12_heldout_n16_seed7301", "world:e03:s12:heldout-array-n16-seed7301", "heldout", 16, 7301, 96),
        S12Config("s12_heldout_n32_seed7302", "world:e03:s12:heldout-array-n32-seed7302", "heldout", 32, 7302, 192),
        S12Config("s12_heldout_n64_seed7303", "world:e03:s12:heldout-array-n64-seed7303", "heldout", 64, 7303, 384),
    ]


def trajectory_dg_recovery_proxy(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    initial = float(values[0])
    final = float(values[-1])
    trough = min(float(value) for value in values)
    if trough >= initial:
        return 0.0
    return max(0.0, final - trough)


def load_policy_jsonl(path: Path, source_label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            dsl_source = str(raw.get("dslSource") or raw.get("dsl_source") or "")
            if not dsl_source:
                continue
            try:
                policy = parse_policy(dsl_source)
            except Exception as exc:
                rows.append(
                    {
                        "source_file": str(path),
                        "source_label": source_label,
                        "source_line_number": line_number,
                        "parse_error": repr(exc)[:300],
                        "raw": raw,
                    }
                )
                continue
            rows.append(
                {
                    "policy": policy,
                    "policy_id": str(raw.get("policyId") or policy.policy_id),
                    "policy_name": str(raw.get("policyName") or policy.name),
                    "source_kind": str(raw.get("sourceKind") or source_label),
                    "semantic_hash": str(raw.get("semanticHash") or semantic_hash(policy)),
                    "dsl_sha256": str(raw.get("dslSha256") or policy.sha256),
                    "dsl_source": dsl_source,
                    "source_file": str(path),
                    "source_label": source_label,
                    "source_line_number": line_number,
                    "features": raw.get("features") or policy_feature_flags(policy),
                    "raw": raw,
                }
            )
    return rows


def load_policy_sources(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    priority = {"frontier": 0, "qd": 1, "generated": 2}
    loaded: list[dict[str, Any]] = []
    parse_issues: list[dict[str, Any]] = []
    for path, label in (
        (args.e03_frontier_policy_library, "frontier"),
        (args.e03_qd_policy_library, "qd"),
        (args.e03_generated_policy_library, "generated"),
    ):
        for record in load_policy_jsonl(path, label):
            if "parse_error" in record:
                parse_issues.append(record)
            else:
                loaded.append(record)
    by_policy: dict[str, dict[str, Any]] = {}
    for record in sorted(loaded, key=lambda item: priority.get(str(item["source_label"]), 9)):
        policy_id = str(record["policy_id"])
        if policy_id in by_policy:
            existing = by_policy[policy_id]
            sources = list(existing.get("source_records", []))
            sources.append({"source_file": record["source_file"], "source_label": record["source_label"], "source_line_number": record["source_line_number"]})
            existing["source_records"] = sources
            continue
        record = dict(record)
        record["source_records"] = [
            {"source_file": record["source_file"], "source_label": record["source_label"], "source_line_number": record["source_line_number"]}
        ]
        by_policy[policy_id] = record
    return by_policy, parse_issues


def read_parquet_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def s10_policy_status_by_source_policy(args: argparse.Namespace) -> pd.DataFrame:
    policy_table = read_parquet_or_empty(args.s02_policy_table)
    classes = read_parquet_or_empty(args.s10_classes)
    assignments = read_parquet_or_empty(args.s10_assignments)
    if policy_table.empty or classes.empty or assignments.empty:
        return pd.DataFrame(columns=["source_policy_id", "s10_policy_class_id", "s10_policy_class_status", "s10_policy_class_label"])
    primary = classes[(classes["entity_type"].astype(str) == "policy") & (classes["branch_role"].astype(str) == "primary")][
        ["class_id", "class_status", "cautious_label"]
    ].drop_duplicates("class_id")
    assigned = assignments[
        (assignments["entity_type"].astype(str) == "policy") & (assignments["branch_role"].astype(str) == "primary")
    ][["entity_id", "class_id"]].merge(primary, on="class_id", how="left")
    s02 = policy_table[
        (policy_table["source_experiment_id"].astype(str) == "E03")
        & (policy_table["representation_type"].astype(str) == "dsl")
        & (policy_table["source_policy_id"].astype(str).str.startswith("dsl:"))
    ][["canonical_policy_id", "source_policy_id"]].drop_duplicates()
    return (
        assigned.merge(s02, left_on="entity_id", right_on="canonical_policy_id", how="inner")
        .rename(
            columns={
                "class_id": "s10_policy_class_id",
                "class_status": "s10_policy_class_status",
                "cautious_label": "s10_policy_class_label",
            }
        )[["source_policy_id", "s10_policy_class_id", "s10_policy_class_status", "s10_policy_class_label"]]
        .drop_duplicates("source_policy_id")
    )


def build_policy_metadata(args: argparse.Namespace, sources: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for policy_id, record in sources.items():
        rows.append(
            {
                "policy_id": policy_id,
                "policy_name": record["policy_name"],
                "source_kind": record["source_kind"],
                "semantic_hash": record["semantic_hash"],
                "dsl_sha256": record["dsl_sha256"],
                "source_label": record["source_label"],
                "features": record["features"],
            }
        )
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        return metadata
    competence = read_parquet_or_empty(args.e03_policy_competence)
    clusters = read_parquet_or_empty(args.e03_policy_clusters)
    frontier = read_parquet_or_empty(args.e03_frontier_candidates)
    s10_status = s10_policy_status_by_source_policy(args)
    if not competence.empty:
        keep = [
            column
            for column in [
                "policy_id",
                "screen_score",
                "screen_heldout_final_sortedness_mean",
                "screen_heldout_improvement_mean",
                "screen_heldout_work_mean",
                "best_final_sortedness",
                "n100_final_sortedness_mean",
                "n1000_final_sortedness_mean",
                "classic_dsl_seed",
                "route",
            ]
            if column in competence.columns
        ]
        metadata = metadata.merge(competence[keep].drop_duplicates("policy_id"), on="policy_id", how="left")
    if not clusters.empty:
        keep = [
            column
            for column in ["policy_id", "class_id", "cautious_label", "screen_score", "screen_heldout_work_mean", "screen_heldout_final_sortedness_mean"]
            if column in clusters.columns
        ]
        cluster_meta = clusters[keep].drop_duplicates("policy_id").rename(
            columns={
                "class_id": "e03_class_id",
                "cautious_label": "e03_cautious_label",
                "screen_score": "cluster_screen_score",
                "screen_heldout_work_mean": "cluster_screen_heldout_work_mean",
                "screen_heldout_final_sortedness_mean": "cluster_screen_heldout_final_sortedness_mean",
            }
        )
        metadata = metadata.merge(cluster_meta, on="policy_id", how="left")
    if not frontier.empty:
        keep = [
            column
            for column in [
                "policy_id",
                "s14_candidate_status",
                "curated_rank",
                "s14_mean_final_sortedness",
                "s14_perturbation_mean_final_sortedness",
                "s14_n32plus_mean_final_sortedness",
                "s14_mean_work",
                "s14_retention_ratio_vs_s07",
            ]
            if column in frontier.columns
        ]
        metadata = metadata.merge(frontier[keep].drop_duplicates("policy_id"), on="policy_id", how="left")
    if not s10_status.empty:
        metadata = metadata.merge(s10_status, left_on="policy_id", right_on="source_policy_id", how="left").drop(columns=["source_policy_id"])
    for column in (
        "screen_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_improvement_mean",
        "screen_heldout_work_mean",
        "best_final_sortedness",
        "s14_mean_final_sortedness",
        "s14_mean_work",
    ):
        if column not in metadata.columns:
            metadata[column] = np.nan
    final_proxy = metadata[["screen_heldout_final_sortedness_mean", "best_final_sortedness", "s14_mean_final_sortedness"]].max(axis=1, skipna=True)
    work_proxy = pd.to_numeric(metadata["screen_heldout_work_mean"], errors="coerce").fillna(pd.to_numeric(metadata["s14_mean_work"], errors="coerce"))
    work_per_item_proxy = (work_proxy / 16.0).replace([np.inf, -np.inf], np.nan).fillna(99.0)
    dg_proxy = pd.to_numeric(metadata["screen_heldout_improvement_mean"], errors="coerce").clip(lower=0.0, upper=TARGET_DG_RECOVERY_PROXY + TARGET_DG_TOLERANCE).fillna(0.0)
    metadata["prior_final_sortedness_proxy"] = final_proxy.fillna(0.0)
    metadata["prior_work_per_item_proxy"] = work_per_item_proxy
    metadata["prior_dg_proxy"] = dg_proxy
    metadata["prior_target_score"] = [
        e03_proxy_target_score(
            final_sortedness=float(final),
            work_per_item=float(work),
            dg_recovery_proxy=float(dg),
            invalid=False,
        )
        for final, work, dg in zip(metadata["prior_final_sortedness_proxy"], metadata["prior_work_per_item_proxy"], metadata["prior_dg_proxy"], strict=False)
    ]
    return metadata


def adjusted_probability(text: str, delta: float) -> str:
    value = min(0.98, max(0.02, float(text) + delta))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def renamed_policy(policy: DSLPolicy, name: str) -> DSLPolicy:
    renamed = DSLPolicy(name=name, version=policy.version, state_init=policy.state_init, rules=policy.rules)
    validate_policy(renamed)
    return renamed


def mutate_policy(policy: DSLPolicy, parent_policy_id: str, *, max_mutations: int = 4) -> list[tuple[DSLPolicy, str]]:
    mutants: list[tuple[DSLPolicy, str]] = []
    for rule_idx, rule in enumerate(policy.rules):
        for condition_idx, condition in enumerate(rule.conditions):
            if condition.name != "random_lt":
                continue
            for delta in (-0.15, 0.15):
                conditions = list(rule.conditions)
                conditions[condition_idx] = DSLCondition("random_lt", (adjusted_probability(condition.args[0], delta),))
                rules = list(policy.rules)
                rules[rule_idx] = DSLRule(conditions=tuple(conditions), actions=rule.actions, is_else=rule.is_else)
                mutant = DSLPolicy(
                    name=f"s12_mut_{parent_policy_id.replace(':', '_')}_{len(mutants):02d}",
                    version=1,
                    state_init=policy.state_init,
                    rules=tuple(rules),
                )
                validate_policy(mutant)
                mutants.append((mutant, f"random_lt_probability_delta_{delta:+.2f}"))
                if len(mutants) >= max_mutations:
                    return mutants
        for action_idx, action in enumerate(rule.actions):
            if action.name == "choose":
                for delta in (-0.15, 0.15):
                    actions = list(rule.actions)
                    actions[action_idx] = DSLAction("choose", (adjusted_probability(action.args[0], delta), action.args[1], action.args[2]))
                    rules = list(policy.rules)
                    rules[rule_idx] = DSLRule(conditions=rule.conditions, actions=tuple(actions), is_else=rule.is_else)
                    mutant = DSLPolicy(
                        name=f"s12_mut_{parent_policy_id.replace(':', '_')}_{len(mutants):02d}",
                        version=1,
                        state_init=policy.state_init,
                        rules=tuple(rules),
                    )
                    validate_policy(mutant)
                    mutants.append((mutant, f"choose_probability_delta_{delta:+.2f}"))
                    if len(mutants) >= max_mutations:
                        return mutants
            if action.name == "swap":
                for probability in ("0.9", "0.75"):
                    actions = list(rule.actions)
                    actions[action_idx] = DSLAction("choose", (probability, action.to_source(), "wait"))
                    rules = list(policy.rules)
                    rules[rule_idx] = DSLRule(conditions=rule.conditions, actions=tuple(actions), is_else=rule.is_else)
                    mutant = DSLPolicy(
                        name=f"s12_mut_{parent_policy_id.replace(':', '_')}_{len(mutants):02d}",
                        version=1,
                        state_init=policy.state_init,
                        rules=tuple(rules),
                    )
                    validate_policy(mutant)
                    mutants.append((mutant, f"swap_probability_gate_{probability}"))
                    if len(mutants) >= max_mutations:
                        return mutants
    return mutants


def build_search_pool(args: argparse.Namespace, sources: dict[str, dict[str, Any]], metadata: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    rng = random.Random(RANDOM_SEED)
    frontier_ids = set(metadata[pd.to_numeric(metadata.get("curated_rank", pd.Series(dtype=float)), errors="coerce").notna()]["policy_id"].astype(str))
    classic = metadata[metadata.get("classic_dsl_seed", False).fillna(False).astype(bool)].copy()
    top_prior = metadata.sort_values(["prior_target_score", "prior_final_sortedness_proxy"], ascending=False).head(args.candidate_pool_size)
    top_frontier = metadata[metadata["policy_id"].astype(str).isin(frontier_ids)].copy()
    top_class_status = metadata[
        metadata.get("s10_policy_class_status", pd.Series("", index=metadata.index))
        .fillna("")
        .astype(str)
        .str.contains("source_dominated|missingness|constraint|uninterpretable", regex=True)
    ].sort_values("prior_target_score", ascending=False).head(max(args.control_count_per_family * 3, 8))
    selected_ids = list(dict.fromkeys([*top_prior["policy_id"].astype(str), *top_frontier["policy_id"].astype(str), *classic["policy_id"].astype(str), *top_class_status["policy_id"].astype(str)]))
    rng.shuffle(selected_ids)
    base_rows: list[dict[str, Any]] = []
    for policy_id in selected_ids:
        record = sources.get(policy_id)
        if record is None:
            continue
        meta_row = metadata[metadata["policy_id"].astype(str) == policy_id].head(1)
        source_selection = "frontier_candidate" if policy_id in frontier_ids else "prior_target_score_pool"
        if not meta_row.empty and bool(meta_row.iloc[0].get("classic_dsl_seed", False)):
            source_selection = "classic_source_control_pool"
        base_rows.append(
            {
                "policy": record["policy"],
                "policy_id": policy_id,
                "policy_name": record["policy_name"],
                "source_kind": record["source_kind"],
                "semantic_hash": record["semantic_hash"],
                "dsl_sha256": record["dsl_sha256"],
                "dsl_source": record["dsl_source"],
                "features": record["features"],
                "source_records": record["source_records"],
                "candidate_origin": source_selection,
                "parent_policy_ids": [],
                "mutation_operator": "",
                "prior_target_score": float(meta_row.iloc[0].get("prior_target_score", 0.0)) if not meta_row.empty else 0.0,
                "e03_class_id": str(meta_row.iloc[0].get("e03_class_id", "")) if not meta_row.empty else "",
                "e03_cautious_label": str(meta_row.iloc[0].get("e03_cautious_label", "")) if not meta_row.empty else "",
                "s10_policy_class_id": str(meta_row.iloc[0].get("s10_policy_class_id", "")) if not meta_row.empty else "",
                "s10_policy_class_status": str(meta_row.iloc[0].get("s10_policy_class_status", "")) if not meta_row.empty else "",
                "s10_policy_class_label": str(meta_row.iloc[0].get("s10_policy_class_label", "")) if not meta_row.empty else "",
            }
        )
    mutation_seed_rows = sorted(base_rows, key=lambda item: item["prior_target_score"], reverse=True)[: args.mutate_top_n]
    for base in mutation_seed_rows:
        for mutant, operator in mutate_policy(base["policy"], base["policy_id"]):
            source_hash = mutant.sha256
            base_rows.append(
                {
                    "policy": mutant,
                    "policy_id": mutant.policy_id,
                    "policy_name": mutant.name,
                    "source_kind": f"s12_{operator}",
                    "semantic_hash": semantic_hash(mutant),
                    "dsl_sha256": source_hash,
                    "dsl_source": mutant.to_source(),
                    "features": policy_feature_flags(mutant),
                    "source_records": [{"source_file": "generated_by_s12_inverse_design", "source_label": "s12_mutation", "source_line_number": 0}],
                    "candidate_origin": "s12_dsl_mutation",
                    "parent_policy_ids": [base["policy_id"]],
                    "mutation_operator": operator,
                    "prior_target_score": base["prior_target_score"],
                    "e03_class_id": base.get("e03_class_id", ""),
                    "e03_cautious_label": base.get("e03_cautious_label", ""),
                    "s10_policy_class_id": base.get("s10_policy_class_id", ""),
                    "s10_policy_class_status": base.get("s10_policy_class_status", ""),
                    "s10_policy_class_label": base.get("s10_policy_class_label", ""),
                }
            )
    deduped: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        deduped.setdefault(str(row["dsl_sha256"]), row)
    return list(deduped.values()), metadata


def simulate_design_on_config(design: Mapping[str, Any], config: S12Config, *, validation_kind: str, target_profile: str) -> dict[str, Any]:
    started = time.perf_counter()
    policy = design["policy"]
    values = tuple(initial_values(config.array_size, config.seed, config.input_profile))
    statuses = tuple("ACTIVE" for _ in values)
    schedule = actor_schedule(config.array_size, config.event_cap, config.seed)
    ideal_position: int | None = None
    current_values = values
    compare_count = 0
    swap_count = 0
    update_count = 0
    wait_count = 0
    trajectory = [float(sortedness_metrics(current_values)["inversion_sortedness"])]
    rng = random.Random((int(config.seed) * 1000003) ^ int(policy.sha256[:12], 16))
    base = {
        "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
        "research_step_id": STEP_ID,
        "design_id": design.get("design_id", ""),
        "target_profile_id": target_profile,
        "design_role": design.get("design_role", "candidate_pool"),
        "control_family": design.get("control_family", "inverse_design"),
        "design_scope": "e03_dsl_array_world",
        "source_experiment_id": "E03",
        "source_policy_id": design.get("source_policy_id", design.get("policy_id", "")),
        "policy_name": design.get("policy_name", policy.name),
        "dsl_sha256": design.get("dsl_sha256", policy.sha256),
        "validation_kind": validation_kind,
        "validation_split": config.split,
        "world_id": config.world_id,
        "config_id": config.config_id,
        "array_size": int(config.array_size),
        "seed": int(config.seed),
        "event_cap": int(config.event_cap),
        "input_profile": config.input_profile,
        "scheduler": config.scheduler,
        "initial_values_json": json.dumps(list(values), separators=(",", ":")),
        "simulator_blocker": "",
    }
    try:
        interpreter = DSLInterpreter(policy)
        for actor_index in schedule:
            state = DSLArrayState(values=current_values, statuses=statuses, actor_index=int(actor_index), ideal_position=ideal_position)
            result = interpreter.step_state(state, rng)
            current_values = result.state_after.values
            statuses = tuple(result.state_after.statuses or statuses)
            ideal_position = result.state_after.ideal_position
            compare_count += int(bool(result.action.compare_counted))
            swap_count += int(result.action.action_type == "swap")
            update_count += int(result.action.action_type == "update_state")
            wait_count += int(result.action.action_type == "wait")
            trajectory.append(float(sortedness_metrics(current_values)["inversion_sortedness"]))
        metrics = sortedness_metrics(current_values)
        work_count = compare_count + swap_count + update_count
        work_per_item = float(work_count / max(config.array_size, 1))
        dg_proxy = trajectory_dg_recovery_proxy(trajectory)
        score = e03_proxy_target_score(
            final_sortedness=float(metrics["inversion_sortedness"]),
            work_per_item=work_per_item,
            dg_recovery_proxy=dg_proxy,
            invalid=False,
        )
        validation_status = "ok_sorted" if bool(metrics["is_sorted"]) else "timeout_event_cap"
        return base | {
            "execution_status": "ok",
            "validation_status": validation_status,
            "invalid": False,
            "error_message": "",
            "events_executed": int(config.event_cap),
            "compare_count": int(compare_count),
            "swap_count": int(swap_count),
            "update_count": int(update_count),
            "wait_count": int(wait_count),
            "work_count": int(work_count),
            "work_per_item": work_per_item,
            "initial_inversion_sortedness": float(trajectory[0]),
            "final_inversion_sortedness": float(metrics["inversion_sortedness"]),
            "inversion_sortedness_delta": float(metrics["inversion_sortedness"] - trajectory[0]),
            "final_is_sorted": bool(metrics["is_sorted"]),
            "dg_recovery_proxy": float(dg_proxy),
            "target_match_score": float(score),
            "final_values_head_json": json.dumps(list(current_values)[:20], separators=(",", ":")),
            "final_values_tail_json": json.dumps(list(current_values)[-20:], separators=(",", ":")),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    except Exception as exc:
        return base | {
            "execution_status": "invalid",
            "validation_status": "invalid",
            "invalid": True,
            "error_message": repr(exc)[:300],
            "events_executed": 0,
            "compare_count": 0,
            "swap_count": 0,
            "update_count": 0,
            "wait_count": 0,
            "work_count": 0,
            "work_per_item": math.nan,
            "initial_inversion_sortedness": float(trajectory[0]),
            "final_inversion_sortedness": math.nan,
            "inversion_sortedness_delta": math.nan,
            "final_is_sorted": False,
            "dg_recovery_proxy": math.nan,
            "target_match_score": 0.0,
            "final_values_head_json": "[]",
            "final_values_tail_json": "[]",
            "elapsed_seconds": float(time.perf_counter() - started),
            "simulator_blocker": f"e03_dsl_simulation_failed:{repr(exc)[:200]}",
        }


def run_simulations(designs: Sequence[Mapping[str, Any]], configs: Sequence[S12Config], *, validation_kind: str, target_profile: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for design in designs:
        for config in configs:
            rows.append(simulate_design_on_config(design, config, validation_kind=validation_kind, target_profile=target_profile))
    return pd.DataFrame(rows)


def summarize_training(train_rows: pd.DataFrame) -> pd.DataFrame:
    if train_rows.empty:
        return pd.DataFrame()
    grouped = train_rows.groupby("design_id", dropna=False)
    return grouped.agg(
        train_config_count=("config_id", "nunique"),
        train_invalid_count=("invalid", "sum"),
        train_mean_final_sortedness=("final_inversion_sortedness", "mean"),
        train_mean_work_per_item=("work_per_item", "mean"),
        train_mean_dg_recovery_proxy=("dg_recovery_proxy", "mean"),
        train_mean_target_match_score=("target_match_score", "mean"),
        train_sorted_run_fraction=("final_is_sorted", "mean"),
    ).reset_index()


def candidate_pool_design_records(pool: Sequence[Mapping[str, Any]], target_profile: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in pool:
        base = {
            "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "step_number": STEP_NUMBER,
            "experiment_id": EXPERIMENT_ID,
            "target_profile_id": target_profile,
            "design_scope": "e03_dsl_array_world",
            "source_experiment_id": "E03",
            "source_policy_id": row["policy_id"],
            "policy_name": row["policy_name"],
            "source_kind": row["source_kind"],
            "representation_type": "dsl",
            "semantic_hash": row["semantic_hash"],
            "dsl_sha256": row["dsl_sha256"],
            "candidate_origin": row["candidate_origin"],
            "parent_policy_ids": list(row.get("parent_policy_ids", [])),
            "mutation_operator": row.get("mutation_operator", ""),
            "prior_target_score": float(row.get("prior_target_score", 0.0)),
            "e03_class_id": row.get("e03_class_id", ""),
            "e03_cautious_label": row.get("e03_cautious_label", ""),
            "s10_policy_class_id": row.get("s10_policy_class_id", ""),
            "s10_policy_class_status": row.get("s10_policy_class_status", ""),
            "s10_policy_class_label": row.get("s10_policy_class_label", ""),
            "parser_validation_status": "parsed",
            "executable_in_s12": True,
            "heldout_validation_redacted_for_freeze": True,
            "source_records_json": json.dumps(row.get("source_records", []), sort_keys=True, separators=(",", ":")),
            "features_json": json.dumps(row.get("features", {}), sort_keys=True, separators=(",", ":"), default=str),
            "dsl_source": row["dsl_source"],
        }
        base["design_id"] = design_record_id(base)
        records.append(base | {"policy": row["policy"]})
    return records


def select_final_designs(candidate_records: list[dict[str, Any]], train_summary: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    frame = pd.DataFrame([{key: value for key, value in record.items() if key != "policy"} for record in candidate_records])
    frame = frame.merge(train_summary, on="design_id", how="left")
    for column in ["train_mean_target_match_score", "train_mean_final_sortedness", "train_mean_work_per_item", "train_invalid_count"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    selected: list[str] = []

    eligible = frame[
        ~frame["source_kind"].astype(str).eq("classic_dsl_seed")
        & ~frame["s10_policy_class_status"].fillna("").astype(str).str.contains("source_dominated|missingness|constraint|uninterpretable", regex=True)
    ].sort_values(["train_mean_target_match_score", "train_mean_final_sortedness"], ascending=False)
    selected.extend(eligible["design_id"].head(args.designed_count).astype(str).tolist())
    if len(selected) < args.designed_count:
        fallback = frame[~frame["design_id"].astype(str).isin(selected)].sort_values(
            ["train_mean_target_match_score", "train_mean_final_sortedness"], ascending=False
        )
        selected.extend(fallback["design_id"].head(args.designed_count - len(selected)).astype(str).tolist())

    def add_control(mask: pd.Series, family: str, sort_columns: list[str]) -> None:
        nonlocal frame, selected
        pool = frame[mask & ~frame["design_id"].astype(str).isin(selected)].sort_values(sort_columns, ascending=False)
        for design_id in pool["design_id"].head(args.control_count_per_family).astype(str).tolist():
            selected.append(design_id)
            frame.loc[frame["design_id"].astype(str).eq(design_id), "control_family_override"] = family

    add_control(frame["source_kind"].astype(str).eq("classic_dsl_seed"), "source_control", ["train_mean_target_match_score", "train_mean_final_sortedness"])
    add_control(pd.Series(True, index=frame.index), "metric_only_control", ["train_mean_final_sortedness", "train_mean_target_match_score"])
    class_mask = frame["s10_policy_class_status"].fillna("").astype(str).str.contains("source_dominated|missingness|constraint|uninterpretable", regex=True)
    add_control(class_mask, "class_status_control", ["train_mean_target_match_score", "train_mean_final_sortedness"])

    final_ids = list(dict.fromkeys(selected))
    by_id = {record["design_id"]: record for record in candidate_records}
    final_records: list[dict[str, Any]] = []
    frame_by_id = frame.set_index("design_id")
    for rank, design_id in enumerate(final_ids, start=1):
        record = dict(by_id[design_id])
        row = frame_by_id.loc[design_id]
        control_override = str(row.get("control_family_override", "")) if pd.notna(row.get("control_family_override", "")) else ""
        if control_override:
            design_role = f"{control_override}_policy"
            control_family = control_override
            eligible_for_positive = False
        else:
            design_role = "designed_candidate"
            control_family = "inverse_design"
            eligible_for_positive = True
        record.update(
            {
                "selection_rank": int(rank),
                "design_role": design_role,
                "control_family": control_family,
                "eligible_for_positive_claim": bool(eligible_for_positive),
                "train_config_count": int(row.get("train_config_count", 0) or 0),
                "train_invalid_count": int(row.get("train_invalid_count", 0) or 0),
                "train_mean_final_sortedness": float(row.get("train_mean_final_sortedness", math.nan)),
                "train_mean_work_per_item": float(row.get("train_mean_work_per_item", math.nan)),
                "train_mean_dg_recovery_proxy": float(row.get("train_mean_dg_recovery_proxy", math.nan)),
                "train_mean_target_match_score": float(row.get("train_mean_target_match_score", math.nan)),
                "train_sorted_run_fraction": float(row.get("train_sorted_run_fraction", math.nan)),
            }
        )
        final_records.append(record)
    return final_records


def json_ready_designs(records: Sequence[Mapping[str, Any]], frozen_at: str) -> list[dict[str, Any]]:
    output = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "policy"}
        row["frozen_at_utc"] = frozen_at
        output.append(row)
    return output


def blocker_rows_from_s11(args: argparse.Namespace, target_profile: str) -> pd.DataFrame:
    if not args.s11_predictions.exists():
        return pd.DataFrame()
    predictions = pd.read_parquet(args.s11_predictions)
    rows: list[dict[str, Any]] = []
    missing_controls = predictions[
        predictions.get("has_policy_link", pd.Series(dtype=float)).fillna(0).astype(float).eq(0)
        | predictions.get("canonical_policy_id", pd.Series(dtype=str)).fillna("").astype(str).isin(["", "__missing__"])
    ].head(4)
    non_e03_or_blocked = predictions[
        ~predictions.get("source_experiment_id", pd.Series(dtype=str)).fillna("").astype(str).eq("E03")
        | predictions.get("capability_target", pd.Series(dtype=str)).fillna("").astype(str).isin(["high_aggregation", "high_repair_robustness"])
    ].head(8)
    for control_family, frame in (("missingness_control", missing_controls), ("scope_blocker", non_e03_or_blocked)):
        for _, record in frame.iterrows():
            rows.append(
                {
                    "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
                    "research_step_id": STEP_ID,
                    "design_id": "",
                    "target_profile_id": target_profile,
                    "design_role": f"{control_family}_s11_record",
                    "control_family": control_family,
                    "design_scope": "not_executable_in_s12",
                    "source_experiment_id": str(record.get("source_experiment_id", "")),
                    "source_policy_id": str(record.get("source_policy_id", "")),
                    "policy_name": str(record.get("policy_display_name", "")),
                    "dsl_sha256": "",
                    "validation_kind": "simulator_blocker",
                    "validation_split": "not_executable",
                    "world_id": str(record.get("world_id", "")),
                    "config_id": "",
                    "array_size": math.nan,
                    "seed": math.nan,
                    "event_cap": math.nan,
                    "input_profile": "",
                    "scheduler": "",
                    "initial_values_json": "[]",
                    "execution_status": "blocked",
                    "validation_status": "blocked",
                    "invalid": True,
                    "error_message": "",
                    "events_executed": 0,
                    "compare_count": 0,
                    "swap_count": 0,
                    "update_count": 0,
                    "wait_count": 0,
                    "work_count": 0,
                    "work_per_item": math.nan,
                    "initial_inversion_sortedness": math.nan,
                    "final_inversion_sortedness": math.nan,
                    "inversion_sortedness_delta": math.nan,
                    "final_is_sorted": False,
                    "dg_recovery_proxy": math.nan,
                    "target_match_score": math.nan,
                    "final_values_head_json": "[]",
                    "final_values_tail_json": "[]",
                    "elapsed_seconds": 0.0,
                    "simulator_blocker": "s12_scope_restricted_to_e03_dsl_array_world_no_e05_e06_adapter_built",
                    "s11_prediction_id": str(record.get("prediction_id", "")),
                    "s11_candidate_role": str(record.get("candidate_role", "")),
                    "s11_capability_target": str(record.get("capability_target", "")),
                }
            )
    target_blockers = [
        ("aggregation_blocker", "missingness_control", "S12 does not instantiate chimeric algotype aggregation in single-policy E03 DSL array worlds."),
        ("repair_robustness_blocker", "scope_blocker", "S12 does not build E05/E06 repair, regeneration, damage, or governance simulator adapters."),
    ]
    for axis, family, reason in target_blockers:
        rows.append(
            {
                "schema_version": INVERSE_DESIGN_SCHEMA_VERSION,
                "research_step_id": STEP_ID,
                "design_id": "",
                "target_profile_id": target_profile,
                "design_role": f"{axis}_target_axis",
                "control_family": family,
                "design_scope": "not_executable_in_s12",
                "source_experiment_id": "E07",
                "source_policy_id": "",
                "policy_name": axis,
                "dsl_sha256": "",
                "validation_kind": "simulator_blocker",
                "validation_split": "not_executable",
                "world_id": "",
                "config_id": "",
                "array_size": math.nan,
                "seed": math.nan,
                "event_cap": math.nan,
                "input_profile": "",
                "scheduler": "",
                "initial_values_json": "[]",
                "execution_status": "blocked",
                "validation_status": "blocked",
                "invalid": True,
                "error_message": "",
                "events_executed": 0,
                "compare_count": 0,
                "swap_count": 0,
                "update_count": 0,
                "wait_count": 0,
                "work_count": 0,
                "work_per_item": math.nan,
                "initial_inversion_sortedness": math.nan,
                "final_inversion_sortedness": math.nan,
                "inversion_sortedness_delta": math.nan,
                "final_is_sorted": False,
                "dg_recovery_proxy": math.nan,
                "target_match_score": math.nan,
                "final_values_head_json": "[]",
                "final_values_tail_json": "[]",
                "elapsed_seconds": 0.0,
                "simulator_blocker": reason,
            }
        )
    return pd.DataFrame(rows)


def summarize_validation(validations: pd.DataFrame) -> pd.DataFrame:
    heldout = validations[validations["validation_kind"].astype(str) == "heldout_e03_simulation"].copy()
    if heldout.empty:
        return pd.DataFrame()
    return heldout.groupby(["design_id", "design_role", "control_family", "source_policy_id", "policy_name"], dropna=False).agg(
        heldout_world_count=("world_id", "nunique"),
        heldout_invalid_count=("invalid", "sum"),
        heldout_mean_final_sortedness=("final_inversion_sortedness", "mean"),
        heldout_mean_work_per_item=("work_per_item", "mean"),
        heldout_mean_dg_recovery_proxy=("dg_recovery_proxy", "mean"),
        heldout_mean_target_match_score=("target_match_score", "mean"),
        heldout_sorted_run_fraction=("final_is_sorted", "mean"),
    ).reset_index().sort_values("heldout_mean_target_match_score", ascending=False)


def classify_outcome(validation_checks: pd.DataFrame, summary: pd.DataFrame) -> str:
    if validation_checks.empty or not bool(validation_checks["success"].all()):
        return "constraining/contradictory"
    designed = summary[summary["control_family"] == "inverse_design"]
    controls = summary[summary["control_family"].isin(["source_control", "metric_only_control", "class_status_control"])]
    if designed.empty:
        return "null"
    top_design = designed.iloc[0]
    best_control_score = float(controls["heldout_mean_target_match_score"].max()) if not controls.empty else -math.inf
    meets_proxy = (
        float(top_design["heldout_mean_final_sortedness"]) >= TARGET_SORTEDNESS
        and float(top_design["heldout_mean_work_per_item"]) <= TARGET_WORK_PER_ITEM
        and int(top_design["heldout_invalid_count"]) == 0
    )
    beats_controls = float(top_design["heldout_mean_target_match_score"]) > best_control_score + 0.01
    if meets_proxy and beats_controls:
        return "supportive"
    if not meets_proxy or not beats_controls:
        return "constraining/contradictory"
    return "null"


def plot_target_match(summary: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    colors = {
        "inverse_design": "#2f6fbb",
        "source_control": "#6b6b6b",
        "metric_only_control": "#b85c38",
        "class_status_control": "#8a6bb8",
    }
    for family, group in summary.groupby("control_family"):
        ax.scatter(
            group["heldout_mean_work_per_item"],
            group["heldout_mean_final_sortedness"],
            s=70 + 220 * group["heldout_mean_dg_recovery_proxy"].fillna(0.0).clip(lower=0.0, upper=0.15),
            alpha=0.82,
            color=colors.get(str(family), "#444444"),
            label=str(family),
            edgecolor="white",
            linewidth=0.7,
        )
    ax.axhline(TARGET_SORTEDNESS, color="#222222", linewidth=1.0, linestyle="--", label="target sortedness")
    ax.axvline(TARGET_WORK_PER_ITEM, color="#555555", linewidth=1.0, linestyle=":", label="target work/item")
    ax.set_xlabel("Heldout mean work count per item")
    ax.set_ylabel("Heldout mean final inversion sortedness")
    ax.set_title("S12 E03 executable-proxy target match")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)


def render_report(
    *,
    report_path: Path,
    artifacts_written: Sequence[str],
    target_profiles: pd.DataFrame,
    frozen_designs: pd.DataFrame,
    validations: pd.DataFrame,
    validation_summary_frame: pd.DataFrame,
    validation_checks: pd.DataFrame,
    target_manifest: Mapping[str, Any],
    design_manifest: Mapping[str, Any],
    outcome: str,
    unit_test_result: Mapping[str, Any] | None,
    input_hashes: Mapping[str, str],
    parse_issue_count: int,
    candidate_pool_size: int,
    args: argparse.Namespace,
) -> None:
    checks = validation_summary(validation_checks)
    blockers = validations[validations["validation_kind"].astype(str) == "simulator_blocker"]
    heldout = validations[validations["validation_kind"].astype(str) == "heldout_e03_simulation"]
    designed = validation_summary_frame[validation_summary_frame["control_family"] == "inverse_design"]
    controls = validation_summary_frame[validation_summary_frame["control_family"] != "inverse_design"]
    unit_test_line = "not run"
    if unit_test_result is not None:
        unit_test_line = f"{'pass' if unit_test_result['success'] else 'fail'}: `{unit_test_result['command']}` return code {unit_test_result['returnCode']}"
    caveats = (
        "S12 was explicitly scoped to executable E03 DSL array-world policies. It did not build E05/E06 simulator adapters, and aggregation or repair/robustness targets remain blocker records. "
        "S07-S11 caveats remain active: supported behavior coverage is narrow, goal-conflict separation failed, S08 all-policy distances were source dominated, S09 invariants were small-effect, S10 classes are provisional, and S11 found strict positive-candidate gaps."
    )
    best_design_text = "none"
    if not designed.empty:
        best = designed.iloc[0]
        best_design_text = (
            f"{best['policy_name']} score={best['heldout_mean_target_match_score']:.3f}, "
            f"sortedness={best['heldout_mean_final_sortedness']:.3f}, work/item={best['heldout_mean_work_per_item']:.3f}, "
            f"DG={best['heldout_mean_dg_recovery_proxy']:.3f}"
        )
    text = f"""# E07 S12 Full Results: Scoped Executable-Proxy Inverse Design

## Top Summary

- Research step ID: S12
- Completion status: complete
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: {'pass' if checks['allPassed'] else 'fail'} ({checks['passed']}/{checks['total']} checks passed)
- Outcome classification: {outcome}
- Caveats or blockers: {caveats}
- Lay summary: S12 froze an E03-only target profile, searched executable E03 DSL array-policy candidates and simple DSL mutations on training seeds, froze the selected candidate designs, and only then validated them on held-out E03 array sizes/seeds. The branch provides an executable proxy result, not broad inverse design across repair, aggregation, governance, or biological substrates.
- Recommended next action: Stop before S13 for Chief review. S13 should treat these policies as E03 executable-proxy designs only and should not infer substrate transfer without explicit simulator support and the blocker/caveat records carried forward.

## Frozen Question

Can a desired behavioral personality, such as high robustness, low aggregation, low energy, and moderate DG, be synthesized or evolved?

Scoped S12 interpretation: can an E03 DSL array-world proxy for high order quality, low work, and moderate DG be selected or mutated, frozen, and validated on held-out E03 seeds/worlds, while recording aggregation and repair/robustness as out-of-scope blockers?

## Inputs

- E03 generated policy library: `{args.e03_generated_policy_library}` (SHA-256 `{input_hashes['e03_generated_policy_library']}`)
- E03 frontier policy library: `{args.e03_frontier_policy_library}` (SHA-256 `{input_hashes['e03_frontier_policy_library']}`)
- E03 QD policy library: `{args.e03_qd_policy_library}` (SHA-256 `{input_hashes['e03_qd_policy_library']}`)
- E03 policy competence: `{args.e03_policy_competence}` (SHA-256 `{input_hashes['e03_policy_competence']}`)
- E03 policy clusters: `{args.e03_policy_clusters}` (SHA-256 `{input_hashes['e03_policy_clusters']}`)
- E03 frontier candidate table: `{args.e03_frontier_candidates}` (SHA-256 `{input_hashes['e03_frontier_candidates']}`)
- S02 policy table: `{args.s02_policy_table}` (SHA-256 `{input_hashes['s02_policy_table']}`)
- S05 modeling dataset: `{args.s05_modeling_dataset}` (SHA-256 `{input_hashes['s05_modeling_dataset']}`)
- S05 heldout-policy surrogate: `{args.s05_linear_model}` (SHA-256 `{input_hashes['s05_linear_model']}`)
- S10 classes and assignments: `{args.s10_classes}`, `{args.s10_assignments}` (SHA-256 `{input_hashes['s10_classes']}`, `{input_hashes['s10_assignments']}`)
- S11 frozen predictions, validations, and freeze manifest: `{args.s11_predictions}`, `{args.s11_validations}`, `{args.s11_freeze_manifest}` (SHA-256 `{input_hashes['s11_predictions']}`, `{input_hashes['s11_validations']}`, `{input_hashes['s11_freeze_manifest']}`)

## Methods

S12 used an executable-proxy branch chosen by the Chief/user instruction. The target profile was frozen before search and includes three executable proxy axes: final inversion sortedness, work count per item, and a sortedness-trajectory DG recovery proxy. Low aggregation and repair/robustness remain explicit blocker axes because this step was restricted to single-policy E03 DSL array worlds.

The search pool came from E03 generated, QD, and frontier DSL policy artifacts. Policies were parsed through the E03 DSL parser, merged with E03 competence/cluster metadata and S10 class-status metadata, then ranked by a pre-heldout target proxy. Deterministic mutation generated small variants by adjusting `random_lt` or `choose` probabilities and by gating swap actions with probabilistic wait choices. Candidate search used only training E03 array configs: `s12_train_n16_seed6101`, `s12_train_n16_seed6102`, and `s12_train_n32_seed6201`.

After training-score selection, S12 froze the candidate design JSONL. The held-out validation then used only `s12_heldout_n16_seed7301`, `s12_heldout_n32_seed7302`, and `s12_heldout_n64_seed7303`. Source controls were classic E03 DSL seeds, metric controls were selected by final sortedness rather than full target score, and class-status controls carried S10 constraining labels. Missingness and out-of-scope S11 records were preserved as simulator-blocker rows.

The S05 surrogate was hashed and retained as an input but not used to rank novel DSL mutations because S11 made source/missingness dominance and target-transform mismatch active caveats. S12 therefore used direct executable proxy simulations for optimization and validation.

## Commands

- `python -m unittest tests.e07.test_inverse_design_schema`
- `python scripts/e07_s12_inverse_design.py`

Unit-test result: {unit_test_line}

## Dependencies and Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- NumPy: {np.__version__}
- matplotlib: {matplotlib.__version__}
- New dependencies installed: none
- CPU/GPU use: serial CPU E03 DSL interpreter; no GPU work or E05/E06 adapters.
- Repository commit before S12 commit: `{git_output(args.repo_dir, ['rev-parse', 'HEAD'])}`
- Branch: `{git_output(args.repo_dir, ['branch', '--show-current'])}`

## Freeze Provenance

- Target profile frozen at UTC: `{target_manifest.get('targetProfilesFrozenAtUtc')}`
- Target profile SHA-256: `{target_manifest.get('targetProfileArtifactSha256')}`
- Candidate designs frozen at UTC: `{design_manifest.get('candidateDesignsFrozenAtUtc')}`
- Candidate design SHA-256: `{design_manifest.get('candidateDesignArtifactSha256')}`
- Validation started at UTC: `{design_manifest.get('validationStartedAtUtc')}`
- Candidate pool size before final freeze: {candidate_pool_size}
- Parser issue records in source libraries: {parse_issue_count}

## Results

Best inverse-designed heldout proxy result: {best_design_text}.

### Target Profile

{markdown_table(target_profiles[['target_axis_id', 'scope_status', 'metric_name', 'metric_direction', 'target_value', 'score_weight', 'limitation']], max_rows=20)}

### Frozen Design Counts

{markdown_table(frozen_designs.groupby(['design_role', 'control_family']).size().reset_index(name='row_count'))}

### Heldout Validation Summary

{markdown_table(validation_summary_frame[['design_role', 'control_family', 'policy_name', 'heldout_world_count', 'heldout_mean_target_match_score', 'heldout_mean_final_sortedness', 'heldout_mean_work_per_item', 'heldout_mean_dg_recovery_proxy', 'heldout_sorted_run_fraction']].head(30), max_rows=30)}

### Control Comparison

{markdown_table(controls[['design_role', 'control_family', 'policy_name', 'heldout_mean_target_match_score', 'heldout_mean_final_sortedness', 'heldout_mean_work_per_item']].head(30), max_rows=30)}

### Simulator Blockers

- Held-out E03 simulation rows: {len(heldout)}
- Simulator blocker rows: {len(blockers)}

{markdown_table(blockers[['design_role', 'control_family', 'source_experiment_id', 's11_capability_target', 'simulator_blocker']].head(40), max_rows=40)}

## Validation

{markdown_table(validation_checks)}

## Output Artifacts

{chr(10).join(f'- `{item}`' for item in artifacts_written)}

## Caveats, Blockers, and Limitations

- This is not broad inverse design. It is an executable E03 DSL array-world proxy branch.
- No E05/E06 simulator adapter was built, by instruction. Aggregation, E05 repair/regeneration, E06 governance, and biological substrate claims remain unvalidated.
- The optimization target is a hand-frozen proxy composite. It does not solve S07/S08 goal-conflict separation failures.
- S08 all-policy distances were source-dominated, S09 invariants were small-effect and baseline-limited, and S10 classes were provisional; S12 controls preserve those concerns rather than tuning them away.
- The DG metric here is a recovery-from-trough proxy over sortedness, not direct evidence of planning or cognition.
- Candidate DSL mutations are small local grammar edits; they are useful executable probes but not a complete policy search.

## Recommended Next Action

Stop before S13 for Chief review. If S13 proceeds, use the S12 designed policies only as E03 executable-proxy inputs and keep blocker rows for missing aggregation, repair, and governance simulators in the substrate-transfer interpretation.
"""
    write_text(report_path, text)


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    policies_dir = artifacts_dir / "policies"
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e07"
    for directory in (step_dir, policies_dir, results_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_test_result = None
    if args.run_unit_tests:
        unit_test_result = run_command([sys.executable, "-m", "unittest", "tests.e07.test_inverse_design_schema"], args.repo_dir)

    target_frozen_at = utc_now()
    target_records = target_profile_records(target_frozen_at)
    target_profile_path = step_dir / "target_profiles.json"
    target_profile_csv_path = tables_dir / "e07_inverse_design_target_profiles.csv"
    target_manifest_path = step_dir / "target_profile_freeze_manifest.json"
    target_profiles = pd.DataFrame(target_records)
    write_json(target_profile_path, target_records)
    target_profiles.to_csv(target_profile_csv_path, index=False)
    target_manifest = {
        "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "targetProfilesFrozenAtUtc": target_frozen_at,
        "targetProfileArtifactPath": str(target_profile_path),
        "targetProfileArtifactSha256": sha256_file(target_profile_path),
        "targetProfileCsvPath": str(target_profile_csv_path),
        "targetProfileCsvSha256": sha256_file(target_profile_csv_path),
        "freezeRule": "Target profile axes and proxy scoring thresholds are frozen before S12 search and heldout validation.",
        "scopeRestriction": "E03 DSL array-world executable proxy only; no E05/E06 adapters built.",
    }
    write_json(target_manifest_path, target_manifest)
    target_profile = str(target_records[0]["target_profile_id"])

    sources, parse_issues = load_policy_sources(args)
    if not sources:
        raise RuntimeError("No executable E03 DSL policy sources were available for S12")
    metadata = build_policy_metadata(args, sources)
    search_pool, metadata = build_search_pool(args, sources, metadata)
    candidate_records = candidate_pool_design_records(search_pool, target_profile)
    train_rows = run_simulations(candidate_records, training_configs(), validation_kind="train_design_simulation", target_profile=target_profile)
    train_summary = summarize_training(train_rows)
    final_records_internal = select_final_designs(candidate_records, train_summary, args)

    candidate_frozen_at = utc_now()
    frozen_json_records = json_ready_designs(final_records_internal, candidate_frozen_at)
    policy_path = policies_dir / "e07_inverse_designed_policies.jsonl"
    write_jsonl(policy_path, frozen_json_records)
    design_hash = sha256_file(policy_path)
    validation_started_at = utc_now()
    design_manifest_path = step_dir / "candidate_design_freeze_manifest.json"
    design_manifest = {
        "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "candidateDesignsFrozenAtUtc": candidate_frozen_at,
        "validationStartedAtUtc": validation_started_at,
        "candidateDesignArtifactPath": str(policy_path),
        "candidateDesignArtifactSha256": design_hash,
        "candidateDesignRowCount": int(len(frozen_json_records)),
        "targetProfileArtifactPath": str(target_profile_path),
        "targetProfileArtifactSha256": target_manifest["targetProfileArtifactSha256"],
        "targetProfilesFrozenAtUtc": target_manifest["targetProfilesFrozenAtUtc"],
        "freezeRule": "Candidate design JSONL includes training/search summaries but excludes heldout validation results.",
        "scopeRestriction": "Only E03 DSL array-world designs are executable in S12; missingness and E05/E06 scope blockers are recorded in validation rows.",
        "trainingConfigIds": [config.config_id for config in training_configs()],
        "heldoutConfigIds": [config.config_id for config in heldout_configs()],
    }
    write_json(design_manifest_path, design_manifest)

    heldout_rows = run_simulations(final_records_internal, heldout_configs(), validation_kind="heldout_e03_simulation", target_profile=target_profile)
    blockers = blocker_rows_from_s11(args, target_profile)
    validations = pd.concat([heldout_rows, blockers], ignore_index=True, sort=False)
    validation_summary_frame = summarize_validation(validations)
    validation_path = results_dir / "e07_inverse_design_validation.parquet"
    validation_csv_path = tables_dir / "e07_inverse_design_validation.csv"
    validation_summary_path = tables_dir / "e07_inverse_design_validation_summary.csv"
    validation_checks_path = step_dir / "e07_s12_validation_checks.csv"
    config_path = step_dir / "s12_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "artifact_manifest.json"
    figure_path = figures_dir / "inverse_design_target_match.png"

    validations.to_parquet(validation_path, index=False)
    validations.to_csv(validation_csv_path, index=False)
    validation_summary_frame.to_csv(validation_summary_path, index=False)
    frozen_designs = pd.DataFrame(frozen_json_records)
    validation_checks = validate_inverse_design_artifacts(target_profiles, frozen_designs, validations, target_manifest, design_manifest)
    if unit_test_result is not None:
        validation_checks = pd.concat(
            [
                validation_checks,
                pd.DataFrame(
                    [
                        {
                            "validation_case": "unit_tests_passed",
                            "success": bool(unit_test_result["success"]),
                            "detail": f"{unit_test_result['command']} rc={unit_test_result['returnCode']}",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    validation_checks.to_csv(validation_checks_path, index=False)
    plot_target_match(validation_summary_frame, figure_path)
    outcome = classify_outcome(validation_checks, validation_summary_frame)
    config_payload = {
        "schemaVersion": INVERSE_DESIGN_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "randomSeed": RANDOM_SEED,
        "scopeRestriction": "E03 DSL array-world executable proxy only",
        "targetSortedness": TARGET_SORTEDNESS,
        "targetWorkPerItem": TARGET_WORK_PER_ITEM,
        "targetDgRecoveryProxy": TARGET_DG_RECOVERY_PROXY,
        "targetDgTolerance": TARGET_DG_TOLERANCE,
        "candidatePoolSizeArg": int(args.candidate_pool_size),
        "candidatePoolSizeObserved": int(len(candidate_records)),
        "mutateTopN": int(args.mutate_top_n),
        "designedCount": int(args.designed_count),
        "controlCountPerFamily": int(args.control_count_per_family),
        "trainingConfigs": [config.__dict__ for config in training_configs()],
        "heldoutConfigs": [config.__dict__ for config in heldout_configs()],
        "noE05OrE06AdaptersBuilt": True,
    }
    write_json(config_path, config_payload)

    input_hashes = {
        "e03_generated_policy_library": maybe_sha256(args.e03_generated_policy_library),
        "e03_frontier_policy_library": maybe_sha256(args.e03_frontier_policy_library),
        "e03_qd_policy_library": maybe_sha256(args.e03_qd_policy_library),
        "e03_policy_competence": maybe_sha256(args.e03_policy_competence),
        "e03_policy_clusters": maybe_sha256(args.e03_policy_clusters),
        "e03_frontier_candidates": maybe_sha256(args.e03_frontier_candidates),
        "s02_policy_table": maybe_sha256(args.s02_policy_table),
        "s05_modeling_dataset": maybe_sha256(args.s05_modeling_dataset),
        "s05_linear_model": maybe_sha256(args.s05_linear_model),
        "s10_classes": maybe_sha256(args.s10_classes),
        "s10_assignments": maybe_sha256(args.s10_assignments),
        "s11_predictions": maybe_sha256(args.s11_predictions),
        "s11_validations": maybe_sha256(args.s11_validations),
        "s11_freeze_manifest": maybe_sha256(args.s11_freeze_manifest),
    }
    artifacts_written = [
        str(report_path),
        str(policy_path),
        str(validation_path),
        str(validation_csv_path),
        str(validation_summary_path),
        str(figure_path),
        str(target_profile_path),
        str(target_profile_csv_path),
        str(target_manifest_path),
        str(design_manifest_path),
        str(validation_checks_path),
        str(config_path),
        str(manifest_path),
    ]
    render_report(
        report_path=report_path,
        artifacts_written=artifacts_written,
        target_profiles=target_profiles,
        frozen_designs=frozen_designs,
        validations=validations,
        validation_summary_frame=validation_summary_frame,
        validation_checks=validation_checks,
        target_manifest=target_manifest,
        design_manifest=design_manifest,
        outcome=outcome,
        unit_test_result=unit_test_result,
        input_hashes=input_hashes,
        parse_issue_count=len(parse_issues),
        candidate_pool_size=len(candidate_records),
        args=args,
    )

    manifest = {
        "schemaVersion": "eidosoma.e07.s12.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "generatedAtUtc": utc_now(),
        "outcomeClassification": outcome,
        "validationResult": validation_summary(validation_checks),
        "artifactEntries": [
            self_referential_artifact_entry(report_path, artifacts_dir, "S12 full-results report."),
            artifact_entry(policy_path, artifacts_dir, "Frozen S12 inverse-designed E03 DSL policy/control records."),
            artifact_entry(validation_path, artifacts_dir, "S12 heldout validation and simulator-blocker rows."),
            artifact_entry(validation_csv_path, artifacts_dir, "CSV copy of S12 validation rows."),
            artifact_entry(validation_summary_path, artifacts_dir, "Per-design heldout validation summary."),
            artifact_entry(figure_path, artifacts_dir, "S12 target-match figure."),
            artifact_entry(target_profile_path, artifacts_dir, "Frozen S12 target profile JSON."),
            artifact_entry(target_profile_csv_path, artifacts_dir, "Frozen S12 target profile table."),
            artifact_entry(target_manifest_path, artifacts_dir, "S12 target-profile freeze manifest."),
            artifact_entry(design_manifest_path, artifacts_dir, "S12 candidate-design freeze manifest."),
            artifact_entry(validation_checks_path, artifacts_dir, "S12 validation checks."),
            artifact_entry(config_path, artifacts_dir, "S12 run configuration."),
            self_referential_artifact_entry(manifest_path, artifacts_dir, "S12 artifact manifest."),
        ],
        "inputHashes": input_hashes,
        "git": {
            "commit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
    }
    write_json(manifest_path, manifest)
    print(f"[S12] outcome={outcome}")
    print(f"[S12] validation checks: {validation_summary(validation_checks)}")
    print(f"[S12] wrote {report_path}")


if __name__ == "__main__":
    main()
