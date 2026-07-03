#!/usr/bin/env python3
"""Run E07 S11 frozen counterfactual prediction tests."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
import time
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
from src.e03.rule_dsl import DSLArrayState, DSLInterpreter, parse_policy  # noqa: E402
from src.e07.corpus_schema import sha256_file  # noqa: E402
from src.e07.counterfactual_schema import (  # noqa: E402
    CAPABILITY_TARGETS,
    COUNTERFACTUAL_SCHEMA_VERSION,
    POSITIVE_CLASS_STATUS,
    capability_target_for_record,
    feature_dicts,
    grouped_median_predictions,
    has_constraining_status,
    stable_record_hash,
    validate_counterfactual_artifacts,
    validation_summary,
)
from src.e07.predictor_schema import assign_group_holdout  # noqa: E402


STEP_ID = "S11"
STEP_NUMBER = 11
EXPERIMENT_ID = "E07"
RANDOM_SEED = 20260703
SIMULATION_SEEDS = (7201, 7202, 7203, 7204)


def parse_args() -> argparse.Namespace:
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_dir)
    parser.add_argument("--s05-modeling-dataset", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_modeling_dataset.parquet")
    parser.add_argument("--s05-linear-model", type=Path, default=artifacts_dir / "models" / "e07_behavior_predictor" / "hashed_linear_sgd_heldout_policy.pkl")
    parser.add_argument("--s05-feature-manifest", type=Path, default=artifacts_dir / "results" / "e07_behavior_predictor_feature_manifest.json")
    parser.add_argument("--s10-classes", type=Path, default=artifacts_dir / "results" / "e07_universality_classes.parquet")
    parser.add_argument("--s10-assignments", type=Path, default=artifacts_dir / "results" / "e07_universality_assignments.parquet")
    parser.add_argument("--s02-policy-table", type=Path, default=artifacts_dir / "tables" / "e07_policy_representations.parquet")
    parser.add_argument("--e03-generated-policy-library", type=Path, default=Path("/previous-artifacts/E03/policies/e03_generated_policy_library.jsonl"))
    parser.add_argument("--positive-per-capability", type=int, default=8)
    parser.add_argument("--caution-per-capability", type=int, default=4)
    parser.add_argument("--executable-caution-count", type=int, default=8)
    parser.add_argument("--simulate-array-size", type=int, default=32)
    parser.add_argument("--simulate-event-multiplier", type=int, default=6)
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


def coalesce(frame: pd.DataFrame, columns: Sequence[str], default: str = "") -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype("object")
        mask = out.isna() | out.astype(str).eq("")
        out.loc[mask] = values.loc[mask]
    return out.fillna(default)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def attach_primary_classes(modeling: pd.DataFrame, classes: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    primary = classes[classes["branch_role"].astype(str) == "primary"][
        ["entity_type", "class_id", "class_status", "cautious_label"]
    ].drop_duplicates(["entity_type", "class_id"])
    assigned = assignments.merge(primary, on=["entity_type", "class_id"], how="left")

    def entity_frame(entity_type: str, id_column: str, prefix: str) -> pd.DataFrame:
        frame = assigned[(assigned["entity_type"].astype(str) == entity_type) & (assigned["branch_role"].astype(str) == "primary")][
            ["entity_id", "class_id", "class_status", "cautious_label"]
        ].drop_duplicates("entity_id")
        return frame.rename(
            columns={
                "entity_id": id_column,
                "class_id": f"{prefix}_class_id",
                "class_status": f"{prefix}_class_status",
                "cautious_label": f"{prefix}_class_label",
            }
        )

    frame = (
        modeling.merge(entity_frame("policy", "canonical_policy_id", "policy"), on="canonical_policy_id", how="left")
        .merge(entity_frame("world", "world_id", "world"), on="world_id", how="left")
        .merge(entity_frame("goal", "canonical_goal_id", "goal"), on="canonical_goal_id", how="left")
    )
    for prefix in ("policy", "world", "goal"):
        frame[f"{prefix}_class_status"] = frame[f"{prefix}_class_status"].fillna("unclassified")
        frame[f"{prefix}_class_id"] = frame[f"{prefix}_class_id"].fillna("")
        frame[f"{prefix}_class_label"] = frame[f"{prefix}_class_label"].fillna("")
    frame["class_status_tuple"] = (
        frame["policy_class_status"].astype(str)
        + "|"
        + frame["world_class_status"].astype(str)
        + "|"
        + frame["goal_class_status"].astype(str)
    )
    frame["has_constraining_class_status"] = frame[["policy_class_status", "world_class_status", "goal_class_status"]].apply(
        lambda row: has_constraining_status(row), axis=1
    )
    frame["has_positive_class_status"] = frame[["policy_class_status", "world_class_status", "goal_class_status"]].eq(POSITIVE_CLASS_STATUS).any(axis=1)
    frame["capability_target"] = [capability_target_for_record(row) for row in frame.to_dict(orient="records")]
    return frame


def load_model_prediction(frame: pd.DataFrame, model_path: Path) -> np.ndarray:
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["model"].predict(payload["hasher"].transform(feature_dicts(frame)))


def outcome_metric_mask(frame: pd.DataFrame) -> pd.Series:
    """Prefer outcome-like rows and exclude setup or model-prediction fields."""

    metric = frame["source_metric_name"].fillna("").astype(str).str.lower()
    source_table = frame["source_table"].fillna("").astype(str).str.lower()
    reject = (
        metric.str.startswith("initial_")
        | metric.str.startswith("pre_")
        | metric.str.contains("pre_history", regex=False)
        | metric.str.startswith("predicted_")
        | metric.eq("predicted_numeric")
        | source_table.str.contains("prediction", regex=False)
    )
    accept_tokens = (
        metric.str.contains("final", regex=False)
        | metric.str.contains("peak", regex=False)
        | metric.str.contains("delta", regex=False)
        | metric.str.contains("recovery", regex=False)
        | metric.str.contains("score", regex=False)
        | metric.str.contains("dg_", regex=False)
        | metric.str.contains("sortedness", regex=False)
        | metric.str.contains("aggregation", regex=False)
    )
    return ~reject & accept_tokens


def score_candidates(frame: pd.DataFrame, train: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    scored = frame.copy()
    scored["surrogate_prediction"] = load_model_prediction(scored, args.s05_linear_model)
    scored["source_metric_median_prediction"] = grouped_median_predictions(
        train, scored, ("source_experiment_id", "source_metric_name", "metric_direction")
    )
    scored["metric_family_median_prediction"] = grouped_median_predictions(train, scored, ("metric_family", "metric_direction"))
    scored["missingness_pattern_median_prediction"] = grouped_median_predictions(
        train,
        scored,
        ("has_world_link", "has_policy_link", "has_goal_link", "policy_metadata_only", "goal_partial"),
    )
    scored["class_status_median_prediction"] = grouped_median_predictions(
        train,
        scored,
        ("policy_class_status", "world_class_status", "goal_class_status", "capability_target"),
    )
    baseline_cols = [
        "source_metric_median_prediction",
        "metric_family_median_prediction",
        "missingness_pattern_median_prediction",
        "class_status_median_prediction",
    ]
    scored["max_baseline_prediction"] = scored[baseline_cols].max(axis=1)
    scored["surrogate_lift_vs_source_metric"] = scored["surrogate_prediction"] - scored["source_metric_median_prediction"]
    scored["surrogate_lift_vs_max_baseline"] = scored["surrogate_prediction"] - scored["max_baseline_prediction"]
    return scored


def deduplicate_for_selection(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["capability_target", "canonical_policy_id", "world_id", "canonical_goal_id", "source_metric_name"]
    return frame.sort_values(["surrogate_lift_vs_source_metric", "surrogate_prediction"], ascending=False).drop_duplicates(keys)


def select_candidate_rows(scored: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    target_mask = scored["capability_target"].isin(CAPABILITY_TARGETS)
    has_policy = scored["canonical_policy_id"].fillna("").astype(str).ne("__missing__") & scored["canonical_policy_id"].fillna("").astype(str).ne("")
    outcome_like = outcome_metric_mask(scored)
    positive_pool = scored[
        target_mask
        & outcome_like
        & has_policy
        & scored["policy_class_status"].astype(str).eq(POSITIVE_CLASS_STATUS)
        & ~scored["has_constraining_class_status"].astype(bool)
    ].copy()
    positive_pool = deduplicate_for_selection(positive_pool)

    selected_parts: list[pd.DataFrame] = []
    selection_gaps: list[dict[str, Any]] = []
    for target in CAPABILITY_TARGETS:
        sub = positive_pool[positive_pool["capability_target"] == target].head(args.positive_per_capability).copy()
        if sub.empty:
            selection_gaps.append(
                {
                    "capabilityTarget": target,
                    "candidateRole": "positive_candidate",
                    "reason": "no heldout-policy outcome-like rows satisfied positive policy class and non-constraining world/goal class filters",
                }
            )
        else:
            sub["candidate_role"] = "positive_candidate"
            sub["selection_reason"] = "top_surrogate_lift_with_positive_policy_class_and_no_constraining_s10_status"
            selected_parts.append(sub)

    caution_pool = scored[target_mask & outcome_like & scored["has_constraining_class_status"].astype(bool)].copy()
    caution_pool = deduplicate_for_selection(caution_pool)
    for target in CAPABILITY_TARGETS:
        sub = caution_pool[caution_pool["capability_target"] == target].head(args.caution_per_capability).copy()
        if not sub.empty:
            sub["candidate_role"] = "caution_control"
            sub["selection_reason"] = "top_surrogate_prediction_with_source_missingness_or_other_constraining_s10_status"
            selected_parts.append(sub)

    executable_pool = caution_pool[
        caution_pool["source_policy_id"].fillna("").astype(str).str.startswith("dsl:")
        & caution_pool["capability_target"].isin(["high_order_quality", "high_dg"])
    ].copy()
    if not executable_pool.empty:
        executable = executable_pool.head(args.executable_caution_count).copy()
        executable["candidate_role"] = "caution_executable_simulation_control"
        executable["selection_reason"] = "executable_dsl_caution_control_for_fresh_s11_simulator_probe"
        selected_parts.append(executable)

    if not selected_parts:
        return pd.DataFrame(), selection_gaps

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(["candidate_role", "capability_target", "surrogate_lift_vs_source_metric"], ascending=[True, True, False])
    selected = selected.drop_duplicates("corpus_row_id").reset_index(drop=True)
    selected["selection_rank"] = selected.groupby(["candidate_role", "capability_target"]).cumcount() + 1
    return selected, selection_gaps


def recursive_find_dsl_source(value: Any) -> str:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) == "dslSource" and isinstance(item, str) and item.strip():
                return item
            found = recursive_find_dsl_source(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = recursive_find_dsl_source(item)
            if found:
                return found
    return ""


def load_dsl_sources(policy_table: pd.DataFrame, e03_library_path: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    if e03_library_path.exists():
        with e03_library_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("policyId") and record.get("dslSource"):
                    sources[str(record["policyId"])] = str(record["dslSource"])
    for row in policy_table.to_dict(orient="records"):
        payload_text = row.get("representation_payload_json")
        dsl_source = ""
        if isinstance(payload_text, str) and payload_text.strip():
            try:
                dsl_source = recursive_find_dsl_source(json.loads(payload_text))
            except json.JSONDecodeError:
                dsl_source = ""
        for key in (row.get("source_policy_id"), row.get("canonical_policy_id")):
            if key and dsl_source:
                sources[str(key)] = dsl_source
    return sources


def trajectory_dg_recovery_proxy(values: Sequence[float]) -> float:
    """A bounded trajectory proxy: final recovery from the worst post-start dip."""

    if not values:
        return math.nan
    initial = float(values[0])
    final = float(values[-1])
    trough = min(float(value) for value in values)
    if trough >= initial:
        return 0.0
    return max(0.0, final - trough)


def simulate_dsl_policy(dsl_source: str, *, array_size: int, event_cap: int, seeds: Sequence[int]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        policy = parse_policy(dsl_source)
        interpreter = DSLInterpreter(policy)
    except Exception as exc:
        return {
            "simulation_status": "blocked",
            "simulation_blocker": f"dsl_parse_failed:{repr(exc)[:200]}",
            "simulated_seed_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        values = tuple(initial_values(array_size, int(seed), "random_permutation"))
        statuses = tuple("ACTIVE" for _ in values)
        schedule = actor_schedule(array_size, event_cap, int(seed))
        ideal_position: int | None = None
        current_values = values
        compare_count = 0
        swap_count = 0
        update_count = 0
        wait_count = 0
        trajectory = [float(sortedness_metrics(current_values)["inversion_sortedness"])]
        rng = random.Random((int(seed) * 1000003) ^ int(policy.sha256[:12], 16))
        try:
            for actor_index in schedule:
                state = DSLArrayState(
                    values=current_values,
                    statuses=statuses,
                    actor_index=int(actor_index),
                    ideal_position=ideal_position,
                )
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
            rows.append(
                {
                    "seed": int(seed),
                    "final_inversion_sortedness": float(metrics["inversion_sortedness"]),
                    "final_is_sorted": bool(metrics["is_sorted"]),
                    "inversion_sortedness_delta": float(metrics["inversion_sortedness"] - trajectory[0]),
                    "dg_recovery_proxy": trajectory_dg_recovery_proxy(trajectory),
                    "compare_count": int(compare_count),
                    "swap_count": int(swap_count),
                    "update_count": int(update_count),
                    "wait_count": int(wait_count),
                    "work_count": int(compare_count + swap_count + update_count),
                    "invalid": False,
                    "error_message": "",
                }
            )
        except Exception as exc:  # pragma: no cover - recorded as validation blocker
            rows.append(
                {
                    "seed": int(seed),
                    "final_inversion_sortedness": math.nan,
                    "final_is_sorted": False,
                    "inversion_sortedness_delta": math.nan,
                    "dg_recovery_proxy": math.nan,
                    "compare_count": 0,
                    "swap_count": 0,
                    "update_count": 0,
                    "wait_count": 0,
                    "work_count": 0,
                    "invalid": True,
                    "error_message": repr(exc)[:300],
                }
            )
    frame = pd.DataFrame(rows)
    ok = frame[~frame["invalid"].astype(bool)]
    if ok.empty:
        return {
            "simulation_status": "blocked",
            "simulation_blocker": "all_fresh_dsl_simulation_replicates_failed",
            "simulated_seed_count": int(len(frame)),
            "elapsed_seconds": time.perf_counter() - started,
        }
    return {
        "simulation_status": "simulated_proxy",
        "simulation_blocker": "",
        "simulated_seed_count": int(len(ok)),
        "simulation_array_size": int(array_size),
        "simulation_event_cap": int(event_cap),
        "fresh_final_sortedness_mean": float(ok["final_inversion_sortedness"].mean()),
        "fresh_sorted_run_fraction": float(ok["final_is_sorted"].mean()),
        "fresh_sortedness_delta_mean": float(ok["inversion_sortedness_delta"].mean()),
        "fresh_dg_recovery_proxy_mean": float(ok["dg_recovery_proxy"].mean()),
        "fresh_work_count_mean": float(ok["work_count"].mean()),
        "fresh_invalid_replicates": int(frame["invalid"].sum()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def simulator_validation_rows(
    predictions: pd.DataFrame,
    policy_table: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    dsl_sources = load_dsl_sources(policy_table, args.e03_generated_policy_library)
    event_cap = int(args.simulate_array_size * args.simulate_event_multiplier)
    rows: list[dict[str, Any]] = []
    for record in predictions.to_dict(orient="records"):
        source_policy_id = str(record.get("source_policy_id", ""))
        canonical_policy_id = str(record.get("canonical_policy_id", ""))
        dsl_source = dsl_sources.get(source_policy_id) or dsl_sources.get(canonical_policy_id)
        capability = str(record.get("capability_target", ""))
        base = {
            "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "prediction_id": record.get("prediction_id"),
            "candidate_role": record.get("candidate_role"),
            "capability_target": capability,
            "validation_kind": "fresh_simulator_blocker",
            "fresh_simulation_comparable_to_prediction": False,
            "observed_target_transformed": math.nan,
            "surrogate_prediction": record.get("surrogate_prediction"),
            "source_metric_median_prediction": record.get("source_metric_median_prediction"),
            "metric_family_median_prediction": record.get("metric_family_median_prediction"),
            "missingness_pattern_median_prediction": record.get("missingness_pattern_median_prediction"),
            "class_status_median_prediction": record.get("class_status_median_prediction"),
            "surrogate_abs_error": math.nan,
            "source_metric_abs_error": math.nan,
            "metric_family_abs_error": math.nan,
            "missingness_pattern_abs_error": math.nan,
            "class_status_abs_error": math.nan,
            "best_baseline_abs_error": math.nan,
            "surrogate_beats_source_metric": False,
            "surrogate_beats_best_baseline": False,
        }
        if not dsl_source:
            rows.append(base | {"validation_status": "blocked", "simulator_blocker": "no_executable_dsl_source_for_policy"})
            continue
        if capability == "high_aggregation":
            rows.append(
                base
                | {
                    "validation_status": "blocked",
                    "simulator_blocker": "existing_fresh_dsl_simulator_is_single_policy_array_sorting_and_does_not_instantiate_chimeric_aggregation",
                }
            )
            continue
        if capability == "high_repair_robustness":
            rows.append(
                base
                | {
                    "validation_status": "blocked",
                    "simulator_blocker": "existing_fresh_dsl_simulator_lacks_validated_repair_damage_or_frozen_semantics_for_this_cross_world_candidate",
                }
            )
            continue
        if capability not in {"high_order_quality", "high_dg"}:
            rows.append(base | {"validation_status": "blocked", "simulator_blocker": f"unsupported_capability_for_fresh_dsl_proxy:{capability}"})
            continue
        sim = simulate_dsl_policy(
            dsl_source,
            array_size=args.simulate_array_size,
            event_cap=event_cap,
            seeds=SIMULATION_SEEDS,
        )
        if sim.get("simulation_status") == "simulated_proxy":
            rows.append(
                base
                | sim
                | {
                    "validation_kind": "fresh_simulator_proxy",
                    "validation_status": "proxy_simulated_not_metric_matched",
                    "fresh_simulation_comparable_to_prediction": False,
                    "simulator_blocker": "fresh simulation ran only an E03 single-policy array-sorting proxy; target metric/world may not match frozen prediction row",
                }
            )
        else:
            rows.append(base | sim | {"validation_status": "blocked", "simulator_blocker": sim.get("simulation_blocker", "unknown_simulator_blocker")})
    return pd.DataFrame(rows)


def retrospective_validation_rows(predictions: pd.DataFrame, selected_internal: pd.DataFrame) -> pd.DataFrame:
    observed = selected_internal[["prediction_id", "target_transformed", "metric_value"]].copy()
    frame = predictions.merge(observed, on="prediction_id", how="left")
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        observed_target = float(record.get("target_transformed")) if pd.notna(record.get("target_transformed")) else math.nan
        surrogate = float(record.get("surrogate_prediction")) if pd.notna(record.get("surrogate_prediction")) else math.nan
        baselines = {
            "source_metric_abs_error": abs(observed_target - float(record.get("source_metric_median_prediction"))),
            "metric_family_abs_error": abs(observed_target - float(record.get("metric_family_median_prediction"))),
            "missingness_pattern_abs_error": abs(observed_target - float(record.get("missingness_pattern_median_prediction"))),
            "class_status_abs_error": abs(observed_target - float(record.get("class_status_median_prediction"))),
        }
        best_baseline = min(baselines.values())
        rows.append(
            {
                "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
                "research_step_id": STEP_ID,
                "prediction_id": record.get("prediction_id"),
                "candidate_role": record.get("candidate_role"),
                "capability_target": record.get("capability_target"),
                "validation_kind": "heldout_retrospective",
                "validation_status": "observed_in_s05_heldout_policy_rows",
                "fresh_simulation_comparable_to_prediction": False,
                "observed_metric_value": record.get("metric_value"),
                "observed_target_transformed": observed_target,
                "surrogate_prediction": surrogate,
                "source_metric_median_prediction": record.get("source_metric_median_prediction"),
                "metric_family_median_prediction": record.get("metric_family_median_prediction"),
                "missingness_pattern_median_prediction": record.get("missingness_pattern_median_prediction"),
                "class_status_median_prediction": record.get("class_status_median_prediction"),
                "surrogate_abs_error": abs(observed_target - surrogate),
                **baselines,
                "best_baseline_abs_error": best_baseline,
                "surrogate_beats_source_metric": abs(observed_target - surrogate) < baselines["source_metric_abs_error"],
                "surrogate_beats_best_baseline": abs(observed_target - surrogate) < best_baseline,
                "simulator_blocker": "not_a_fresh_simulation; observed value came from frozen heldout-policy S05/S04 row",
            }
        )
    return pd.DataFrame(rows)


def prediction_output_frame(selected: pd.DataFrame, policy_table: pd.DataFrame) -> pd.DataFrame:
    metadata = policy_table[
        [
            "canonical_policy_id",
            "source_policy_id",
            "display_name",
            "source_experiment_id",
            "representation_type",
            "execution_backend",
            "parser_validation_status",
        ]
    ].drop_duplicates("canonical_policy_id")
    metadata = metadata.rename(
        columns={
            "display_name": "policy_display_name",
            "source_experiment_id": "policy_table_source_experiment_id",
            "representation_type": "policy_representation_type_s02",
            "execution_backend": "policy_execution_backend_s02",
        }
    )
    frame = selected.merge(metadata, on="canonical_policy_id", how="left", suffixes=("", "_s02"))
    frame["source_policy_id"] = coalesce(frame, ["source_policy_id_s02", "source_policy_id"], "")
    frame["policy_display_name"] = coalesce(frame, ["policy_display_name", "display_name"], "")
    output_columns = [
        "schema_version",
        "research_step_id",
        "prediction_id",
        "candidate_role",
        "selection_rank",
        "selection_reason",
        "capability_target",
        "corpus_row_id",
        "source_experiment_id",
        "source_table",
        "source_metric_name",
        "metric_unit",
        "metric_direction",
        "metric_family",
        "world_id",
        "canonical_policy_id",
        "source_policy_id",
        "policy_display_name",
        "canonical_goal_id",
        "perturbation_type",
        "policy_class_id",
        "policy_class_status",
        "policy_class_label",
        "world_class_id",
        "world_class_status",
        "world_class_label",
        "goal_class_id",
        "goal_class_status",
        "goal_class_label",
        "has_world_link",
        "has_policy_link",
        "has_goal_link",
        "policy_metadata_only",
        "goal_partial",
        "surrogate_model_name",
        "surrogate_model_split",
        "surrogate_prediction",
        "source_metric_median_prediction",
        "metric_family_median_prediction",
        "missingness_pattern_median_prediction",
        "class_status_median_prediction",
        "max_baseline_prediction",
        "surrogate_lift_vs_source_metric",
        "surrogate_lift_vs_max_baseline",
        "observed_target_redacted_for_freeze",
    ]
    frame["schema_version"] = COUNTERFACTUAL_SCHEMA_VERSION
    frame["research_step_id"] = STEP_ID
    frame["surrogate_model_name"] = "hashed_linear_sgd"
    frame["surrogate_model_split"] = "heldout_policy"
    frame["observed_target_redacted_for_freeze"] = True
    for column in output_columns:
        if column not in frame.columns:
            frame[column] = ""
    prediction_ids = []
    for record in frame[output_columns].drop(columns=["prediction_id"]).to_dict(orient="records"):
        prediction_ids.append(stable_record_hash(record))
    frame["prediction_id"] = prediction_ids
    return frame[output_columns].copy()


def summarize_accuracy(validations: pd.DataFrame) -> pd.DataFrame:
    retro = validations[validations["validation_kind"] == "heldout_retrospective"].copy()
    if retro.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in retro.groupby(["candidate_role", "capability_target"], dropna=False):
        role, target = keys
        rows.append(
            {
                "candidate_role": role,
                "capability_target": target,
                "row_count": int(len(group)),
                "surrogate_mae": float(group["surrogate_abs_error"].mean()),
                "source_metric_median_mae": float(group["source_metric_abs_error"].mean()),
                "metric_family_median_mae": float(group["metric_family_abs_error"].mean()),
                "missingness_pattern_median_mae": float(group["missingness_pattern_abs_error"].mean()),
                "class_status_median_mae": float(group["class_status_abs_error"].mean()),
                "surrogate_beats_source_metric_fraction": float(group["surrogate_beats_source_metric"].mean()),
                "surrogate_beats_best_baseline_fraction": float(group["surrogate_beats_best_baseline"].mean()),
            }
        )
    return pd.DataFrame(rows)


def classify_outcome(accuracy: pd.DataFrame, validation_checks: pd.DataFrame, selection_gaps: Sequence[Mapping[str, Any]], simulator_rows: pd.DataFrame) -> str:
    if not bool(validation_checks["success"].all()):
        return "constraining/contradictory"
    positive = accuracy[accuracy["candidate_role"] == "positive_candidate"]
    positive_supportive = (
        not positive.empty
        and (positive["surrogate_mae"] < positive["source_metric_median_mae"]).all()
        and (positive["surrogate_beats_best_baseline_fraction"] >= 0.5).all()
    )
    simulator_success = bool((simulator_rows.get("validation_kind", pd.Series(dtype=str)) == "fresh_simulator_proxy").any())
    if selection_gaps or not positive_supportive:
        return "constraining/contradictory"
    if simulator_success:
        return "supportive"
    return "null"


def plot_validation(validations: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    retro = validations[validations["validation_kind"] == "heldout_retrospective"].copy()
    sim_counts = validations[validations["validation_kind"].astype(str).str.startswith("fresh_simulator")]["validation_status"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {
        "positive_candidate": "#2f6fbb",
        "caution_control": "#b85c38",
        "caution_executable_simulation_control": "#6b6b6b",
    }
    for role, group in retro.groupby("candidate_role"):
        axes[0].scatter(
            group["surrogate_prediction"],
            group["observed_target_transformed"],
            s=38,
            alpha=0.8,
            label=role,
            color=colors.get(str(role), "#333333"),
        )
    if not retro.empty:
        low = float(np.nanmin([retro["surrogate_prediction"].min(), retro["observed_target_transformed"].min()]))
        high = float(np.nanmax([retro["surrogate_prediction"].max(), retro["observed_target_transformed"].max()]))
        axes[0].plot([low, high], [low, high], color="#222222", linewidth=1, linestyle="--")
    axes[0].set_title("Frozen predictions vs heldout observed targets")
    axes[0].set_xlabel("Frozen surrogate prediction")
    axes[0].set_ylabel("Observed oriented signed-log target")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    if not sim_counts.empty:
        axes[1].bar(sim_counts.index.astype(str), sim_counts.values, color="#6a8f5a")
        axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_title("Fresh simulator validation status")
    axes[1].set_ylabel("Prediction rows")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)


def render_report(
    *,
    report_path: Path,
    artifacts_written: Sequence[str],
    predictions: pd.DataFrame,
    validations: pd.DataFrame,
    accuracy: pd.DataFrame,
    validation_checks: pd.DataFrame,
    freeze_manifest: Mapping[str, Any],
    selection_gaps: Sequence[Mapping[str, Any]],
    outcome: str,
    unit_test_result: Mapping[str, Any] | None,
    input_hashes: Mapping[str, str],
    args: argparse.Namespace,
) -> None:
    validation = validation_summary(validation_checks)
    positive = predictions[predictions["candidate_role"] == "positive_candidate"]
    simulator = validations[validations["validation_kind"].astype(str).str.startswith("fresh_simulator")]
    blocked = simulator[simulator["validation_kind"] == "fresh_simulator_blocker"]
    fresh_proxy = simulator[simulator["validation_kind"] == "fresh_simulator_proxy"]
    gap_text = "; ".join(f"{item['capabilityTarget']}: {item['reason']}" for item in selection_gaps) or "none"
    caveats = (
        "Candidate predictions are frozen and leakage-aware with respect to S05 heldout-policy models, but observed validation values are retrospective S04/S05 rows. "
        "Fresh simulations are only an E03 single-policy array-sorting proxy where executable DSL is available; positive S10 candidates were mostly E06 governance rows and were not directly executable."
    )
    test_line = "not run"
    if unit_test_result is not None:
        test_line = f"{'pass' if unit_test_result['success'] else 'fail'}: `{unit_test_result['command']}` return code {unit_test_result['returnCode']}"
    positive_accuracy = accuracy[accuracy["candidate_role"] == "positive_candidate"]
    text = f"""# E07 S11 Full Results: Run Counterfactual Prediction Tests

## Top Summary

- Research step ID: S11
- Completion status: complete
- Artifacts written: {', '.join(artifacts_written)}
- Validation result: {'pass' if validation['allPassed'] else 'fail'} ({validation['passed']}/{validation['total']} checks passed)
- Outcome classification: {outcome}
- Caveats or blockers: {caveats} Selection gaps: {gap_text}.
- Lay summary: S11 froze surrogate-ranked candidate predictions before validation, then compared the frozen predictions with heldout observed corpus rows and audited which candidates could be run through available simulators. The result constrains the counterfactual program because strict S10-supported positive candidates covered aggregation/order only, did not cover high-DG or repair/robustness, and were not directly executable in the available fresh simulator.
- Recommended next action: Chief review before S12. If S12 proceeds, first add executable simulator adapters for S10-positive E06 governance and E05 repair worlds or restrict inverse design to the E03 DSL array-sorting proxy with explicit scope limits.

## Frozen Question

Can the surrogate predict surprising high-DG, high-robustness, or high-aggregation policies before real simulations validate them?

## Inputs

- S05 modeling dataset: `{args.s05_modeling_dataset}` (SHA-256 `{input_hashes['s05_modeling_dataset']}`)
- S05 heldout-policy hashed linear surrogate: `{args.s05_linear_model}` (SHA-256 `{input_hashes['s05_linear_model']}`)
- S05 feature manifest: `{args.s05_feature_manifest}` (SHA-256 `{input_hashes['s05_feature_manifest']}`)
- S10 class summary: `{args.s10_classes}` (SHA-256 `{input_hashes['s10_classes']}`)
- S10 assignments: `{args.s10_assignments}` (SHA-256 `{input_hashes['s10_assignments']}`)
- S02 policy table: `{args.s02_policy_table}` (SHA-256 `{input_hashes['s02_policy_table']}`)
- E03 generated policy library: `{args.e03_generated_policy_library}` ({'present' if args.e03_generated_policy_library.exists() else 'missing'})

## Methods

S11 used the S05 `hashed_linear_sgd_heldout_policy` surrogate so every selected candidate policy belonged to the deterministic heldout-policy split and had no policy-ID leakage into that model's training rows. Candidate rows were limited to S05/S04 metric observations, scored without using observed targets, and ranked by surrogate lift over the source-metric median baseline.

Positive candidate selection used only S10 primary supported-only `bounded_interpretable_class` policy assignments and rejected rows where the world or goal class status was source-dominated, missingness-driven, small/uninterpretable, or otherwise constraining. Source-dominated, missingness-driven, or constraining classes were used only as caution controls.

The frozen prediction artifact omits `metric_value` and `target_transformed`; those observed values appear only in the validation artifact after the prediction file was written, hashed, and timestamped. The freeze manifest records `frozenAtUtc`, `validationStartedAtUtc`, and the prediction artifact SHA-256.

Fresh simulator validation was attempted only after freezing. It was intentionally conservative: executable DSL policies could be run through an E03 single-policy array-sorting simulator for order/DG proxy checks. Aggregation, repair/robustness, E06 governance, E05 morphology, metadata-only, public-wrapper, and local-training-weight candidates were recorded as simulator blockers unless an executable DSL source and matching proxy were available.

## Commands

- `python -m unittest tests.e07.test_counterfactual_schema`
- `python scripts/e07_s11_counterfactual_tests.py`

Unit-test result: {test_line}

## Dependencies and Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- NumPy: {np.__version__}
- scikit-learn model loaded from S05 pickle; no new package installs.
- CPU worker policy: serial S11 selection and small fresh-simulator probes; no GPU work.
- Repository commit before S11 commit: `{git_output(args.repo_dir, ['rev-parse', 'HEAD'])}`
- Branch: `{git_output(args.repo_dir, ['branch', '--show-current'])}`

## Freeze Provenance

- Frozen prediction rows: {len(predictions)}
- Frozen prediction SHA-256: `{freeze_manifest.get('predictionArtifactSha256')}`
- Frozen at UTC: `{freeze_manifest.get('frozenAtUtc')}`
- Validation started at UTC: `{freeze_manifest.get('validationStartedAtUtc')}`
- Selection gaps: {gap_text}

## Results

### Candidate Counts

{markdown_table(predictions.groupby(['candidate_role', 'capability_target']).size().reset_index(name='row_count'))}

### Retrospective Heldout Accuracy

{markdown_table(accuracy)}

Positive-candidate accuracy rows:

{markdown_table(positive_accuracy)}

### Fresh Simulator Audit

- Fresh proxy simulation rows: {len(fresh_proxy)}
- Simulator blocker rows: {len(blocked)}

{markdown_table(simulator[['prediction_id', 'candidate_role', 'capability_target', 'validation_kind', 'validation_status', 'simulator_blocker']].head(60), max_rows=60)}

### Selection Gaps

{markdown_table(pd.DataFrame(selection_gaps))}

## Validation

{markdown_table(validation_checks)}

## Output Artifacts

{chr(10).join(f'- `{item}`' for item in artifacts_written)}

## Caveats, Blockers, and Limitations

- S11 does not provide prospective biological validation or causal evidence.
- The strongest positive filter found no eligible high-DG or repair/robustness candidates without using constraining S10 classes. Those targets are therefore selection gaps, not tuned-away failures.
- Positive candidates were concentrated in E06 governance aggregation/order rows, and available fresh simulators could not directly replay those E06 governance worlds from a frozen E07 row.
- Retrospective validation uses heldout-policy rows from existing S04/S05 artifacts. It checks model generalization under S05's split, but it is not a new independent simulation.
- Fresh DSL simulations are proxy-only and single-policy array-sorting; they do not validate chimeric aggregation, E05 repair/regeneration, or E06 governance interventions.
- S10 caveats remain active: class labels are provisional and many nearby classes are source-dominated, missingness-driven, or small/uninterpretable.
- The S05 target is a heterogeneous oriented signed-log metric, so errors and lifts are proxy comparisons, not unit-level physical effects.

## Recommended Next Action

Stop before S12 for Chief review. Before inverse design, decide whether to add simulator adapters for E06/E05 positive classes or scope S12 to executable E03 DSL array-world candidates only.
"""
    write_text(report_path, text)


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e07"
    for directory in (step_dir, results_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_test_result = None
    if args.run_unit_tests:
        unit_test_result = run_command([sys.executable, "-m", "unittest", "tests.e07.test_counterfactual_schema"], args.repo_dir)

    modeling = pd.read_parquet(args.s05_modeling_dataset)
    classes = pd.read_parquet(args.s10_classes)
    assignments = pd.read_parquet(args.s10_assignments)
    policy_table = pd.read_parquet(args.s02_policy_table)
    enriched = attach_primary_classes(modeling, classes, assignments)
    policy_lookup = policy_table[
        [
            "canonical_policy_id",
            "source_policy_id",
            "display_name",
            "representation_type",
            "execution_backend",
            "parser_validation_status",
        ]
    ].drop_duplicates("canonical_policy_id")
    enriched = enriched.merge(policy_lookup, on="canonical_policy_id", how="left", suffixes=("", "_s02"))
    holdout_mask = assign_group_holdout(enriched, group_column="canonical_policy_id", split_name="heldout_policy", test_fraction=0.2)
    holdout = enriched[holdout_mask].copy()
    train = enriched[~holdout_mask].copy()
    scored = score_candidates(holdout, train, args)
    selected_internal, selection_gaps = select_candidate_rows(scored, args)
    if selected_internal.empty:
        raise RuntimeError("S11 candidate selection produced no rows; cannot freeze an empty prediction artifact")

    predictions = prediction_output_frame(selected_internal, policy_table)
    selected_internal = selected_internal.merge(predictions[["corpus_row_id", "prediction_id"]], on="corpus_row_id", how="left")

    prediction_path = results_dir / "e07_counterfactual_predictions.parquet"
    prediction_csv_path = tables_dir / "e07_counterfactual_predictions.csv"
    freeze_manifest_path = step_dir / "candidate_freeze_manifest.json"
    predictions.to_parquet(prediction_path, index=False)
    predictions.to_csv(prediction_csv_path, index=False)
    frozen_at = utc_now()
    prediction_hash = sha256_file(prediction_path)
    validation_started_at = utc_now()
    freeze_manifest = {
        "schemaVersion": COUNTERFACTUAL_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "frozenAtUtc": frozen_at,
        "validationStartedAtUtc": validation_started_at,
        "predictionArtifactPath": str(prediction_path),
        "predictionArtifactSha256": prediction_hash,
        "predictionCsvPath": str(prediction_csv_path),
        "predictionCsvSha256": sha256_file(prediction_csv_path),
        "predictionRowCount": int(len(predictions)),
        "selectionGaps": selection_gaps,
        "freezeRule": "Observed metric_value and target_transformed columns are omitted from the frozen prediction artifact.",
        "surrogateModel": str(args.s05_linear_model),
        "surrogateModelSha256": sha256_file(args.s05_linear_model),
        "candidateSelection": {
            "positiveClassStatus": POSITIVE_CLASS_STATUS,
            "positivePerCapability": int(args.positive_per_capability),
            "cautionPerCapability": int(args.caution_per_capability),
            "executableCautionCount": int(args.executable_caution_count),
            "heldoutSplit": "heldout_policy",
        },
    }
    write_json(freeze_manifest_path, freeze_manifest)

    frozen_predictions = pd.read_parquet(prediction_path)
    retrospective = retrospective_validation_rows(frozen_predictions, selected_internal)
    simulator = simulator_validation_rows(frozen_predictions, policy_table, args)
    validations = pd.concat([retrospective, simulator], ignore_index=True, sort=False)
    accuracy = summarize_accuracy(validations)
    validation_checks = validate_counterfactual_artifacts(frozen_predictions, validations, freeze_manifest)
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

    validation_path = results_dir / "e07_counterfactual_validations.parquet"
    validation_csv_path = tables_dir / "e07_counterfactual_validations.csv"
    accuracy_path = tables_dir / "e07_counterfactual_accuracy_summary.csv"
    validation_checks_path = step_dir / "e07_s11_validation_checks.csv"
    config_path = step_dir / "s11_config.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "artifact_manifest.json"
    figure_path = figures_dir / "counterfactual_validation.png"

    validations.to_parquet(validation_path, index=False)
    validations.to_csv(validation_csv_path, index=False)
    accuracy.to_csv(accuracy_path, index=False)
    validation_checks.to_csv(validation_checks_path, index=False)
    plot_validation(validations, figure_path)
    outcome = classify_outcome(accuracy, validation_checks, selection_gaps, simulator)

    input_hashes = {
        "s05_modeling_dataset": sha256_file(args.s05_modeling_dataset),
        "s05_linear_model": sha256_file(args.s05_linear_model),
        "s05_feature_manifest": sha256_file(args.s05_feature_manifest),
        "s10_classes": sha256_file(args.s10_classes),
        "s10_assignments": sha256_file(args.s10_assignments),
        "s02_policy_table": sha256_file(args.s02_policy_table),
    }
    write_json(
        config_path,
        {
            "schemaVersion": COUNTERFACTUAL_SCHEMA_VERSION,
            "researchStepId": STEP_ID,
            "randomSeed": RANDOM_SEED,
            "simulationSeeds": list(SIMULATION_SEEDS),
            "simulateArraySize": int(args.simulate_array_size),
            "simulateEventMultiplier": int(args.simulate_event_multiplier),
            "positivePerCapability": int(args.positive_per_capability),
            "cautionPerCapability": int(args.caution_per_capability),
            "executableCautionCount": int(args.executable_caution_count),
            "inputHashes": input_hashes,
            "outcomeClassification": outcome,
        },
    )

    artifacts_written = [
        str(report_path),
        str(prediction_path),
        str(prediction_csv_path),
        str(validation_path),
        str(validation_csv_path),
        str(accuracy_path),
        str(figure_path),
        str(freeze_manifest_path),
        str(validation_checks_path),
        str(config_path),
        str(manifest_path),
    ]
    render_report(
        report_path=report_path,
        artifacts_written=artifacts_written,
        predictions=frozen_predictions,
        validations=validations,
        accuracy=accuracy,
        validation_checks=validation_checks,
        freeze_manifest=freeze_manifest,
        selection_gaps=selection_gaps,
        outcome=outcome,
        unit_test_result=unit_test_result,
        input_hashes=input_hashes,
        args=args,
    )

    manifest_payload = {
        "schemaVersion": "eidosoma.e07.s11.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "success": bool(validation_checks["success"].all()),
        "outcomeClassification": outcome,
        "validationResult": validation_summary(validation_checks),
        "candidateFreezeManifest": str(freeze_manifest_path),
        "predictionArtifactSha256": prediction_hash,
        "selectionGaps": selection_gaps,
        "artifacts": [
            self_referential_artifact_entry(report_path, artifacts_dir, "S11 full-results report."),
            artifact_entry(prediction_path, artifacts_dir, "Frozen S11 counterfactual predictions in Parquet format."),
            artifact_entry(prediction_csv_path, artifacts_dir, "Frozen S11 counterfactual predictions in CSV format."),
            artifact_entry(validation_path, artifacts_dir, "S11 counterfactual validation rows in Parquet format."),
            artifact_entry(validation_csv_path, artifacts_dir, "S11 counterfactual validation rows in CSV format."),
            artifact_entry(accuracy_path, artifacts_dir, "S11 accuracy summary by candidate role and capability."),
            artifact_entry(figure_path, artifacts_dir, "S11 counterfactual validation figure."),
            artifact_entry(freeze_manifest_path, artifacts_dir, "Candidate freeze hash/timestamp manifest."),
            artifact_entry(validation_checks_path, artifacts_dir, "S11 validation checks."),
            artifact_entry(config_path, artifacts_dir, "S11 configuration and input hashes."),
            self_referential_artifact_entry(manifest_path, artifacts_dir, "S11 artifact manifest."),
        ],
    }
    write_json(manifest_path, manifest_payload)
    manifest_payload["artifacts"][-1]["sizeBytes"] = manifest_path.stat().st_size
    write_json(manifest_path, manifest_payload)
    print(f"[S11] wrote {report_path}")
    print(f"[S11] frozen predictions {len(predictions)} hash={prediction_hash}")
    print(f"[S11] validation {int(validation_checks['success'].sum())}/{len(validation_checks)} outcome={outcome}")


if __name__ == "__main__":
    main()
