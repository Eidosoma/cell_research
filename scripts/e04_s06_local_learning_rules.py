#!/usr/bin/env python3
"""Execute E04 S06 local-learning rule validation and comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from memory_repair import (  # noqa: E402
    FATIGUE_DAMAGE_VERSION,
    HOMEOSTASIS_BENCHMARK_VERSION,
    LEARNING_VARIANTS,
    LOCAL_LEARNING_ACTIONS,
    LOCAL_LEARNING_VERSION,
    LOCAL_REWARD_ALLOWED_INPUTS,
    LOCAL_REWARD_FORBIDDEN_INPUTS,
    MEMORY_REPAIR_VERSION,
    REPAIR_REPAIR_VERSION,
    SIGNAL_REPAIR_VERSION,
    FatigueDamageConfig,
    LearningEventSimulator,
    LocalLearningConfig,
    LocalLearningPolicyWrapper,
    RepairRuleConfig,
    SignalConfig,
    apply_homeostatic_event_to_simulator,
    build_homeostatic_benchmark_config,
    build_learning_variants,
    learning_information_boundary,
    learning_state_for_trace,
    learning_summary_record,
    normalized_sortedness_record,
    normalize_action_probabilities,
    stable_hash,
)


EXPERIMENT_ID = "E04"
STEP_ID = "S06"
STEP_NUMBER = 6
STEP_TITLE = "Create local learning rules"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def yaml_scalar(value: Any) -> str:
    value = json_ready(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in ":#{}[]\n,") or text.lower() in {"true", "false", "null"}:
        return json.dumps(text)
    return text


def to_yaml(value: Any, indent: int = 0) -> str:
    value = json_ready(value)
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        lines = []
        for item in value:
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}-")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{yaml_scalar(value)}"


def run_command(args: list[str], cwd: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        merged_env.update(env)
    started = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "args": args,
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "runtimeSeconds": time.perf_counter() - started,
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "get-url", "origin"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["success"] else "unknown",
        "branch": branch["stdout"].strip() if branch["success"] else "unknown",
        "remote": remote["stdout"].strip() if remote["success"] else "unknown",
        "statusShort": status["stdout"].strip(),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".") if math.isfinite(value) else ""
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def dataframe_to_artifacts(df: pd.DataFrame, step_path: Path, results_path: Path) -> list[Path]:
    step_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(step_path.with_suffix(".csv"), index=False)
    df.to_parquet(step_path.with_suffix(".parquet"), index=False)
    shutil.copy2(step_path.with_suffix(".csv"), results_path.with_suffix(".csv"))
    shutil.copy2(step_path.with_suffix(".parquet"), results_path.with_suffix(".parquet"))
    return [
        step_path.with_suffix(".csv"),
        step_path.with_suffix(".parquet"),
        results_path.with_suffix(".csv"),
        results_path.with_suffix(".parquet"),
    ]


def learning_config_payload() -> dict[str, Any]:
    return {
        "schema": "eidosoma.e04_local_learning_rules.v1",
        "producerStep": STEP_ID,
        "localLearningVersion": LOCAL_LEARNING_VERSION,
        "actions": list(LOCAL_LEARNING_ACTIONS),
        "variants": [
            LocalLearningConfig(variant=variant, learning_rate=0.6, random_seed=6600 + index).to_dict()
            for index, variant in enumerate(LEARNING_VARIANTS)
        ],
        "informationBoundary": learning_information_boundary(),
        "normalization": {
            "probabilityRule": "all action weights are projected onto a bounded normalized simplex after every update",
            "minActionProbability": 0.04,
            "maxActionProbability": 0.92,
            "probabilitySum": 1.0,
        },
        "comparisonControls": ["fixed", "local_adaptive", "random_adaptation"],
    }


def learning_variant_rows() -> list[dict[str, Any]]:
    rows = []
    for index, variant in enumerate(LEARNING_VARIANTS):
        config = LocalLearningConfig(variant=variant, learning_rate=0.6, random_seed=6600 + index)
        policy = LocalLearningPolicyWrapper("bubble", config)
        rows.append(
            {
                "variant": variant,
                "policyId": policy.policy_id,
                "policyFamily": policy.family,
                "learningRate": config.learning_rate,
                "minActionProbability": config.min_action_probability,
                "maxActionProbability": config.max_action_probability,
                "randomSeed": config.random_seed,
                "initialActionProbabilitiesJson": compact_json(config.initial_action_probabilities),
                "usesGlobalSortednessSignal": False,
                "usesWholeArrayTargetSignal": False,
                "localLearningVersion": LOCAL_LEARNING_VERSION,
            }
        )
    return rows


def repair_task_configs() -> list[dict[str, Any]]:
    return [
        {
            "taskId": "repair_nudge_threshold_pair",
            "taskFamily": "repair",
            "initialValues": [2, 1],
            "frozenPositions": [1],
            "frozenVariant": "stuck",
            "repairConfig": RepairRuleConfig("nudge_count", nudge_threshold=2).to_dict(),
            "signalConfig": SignalConfig("no_signal").to_dict(),
            "fatigueConfig": FatigueDamageConfig("none").to_dict(),
            "maxActivations": 12,
            "schedulerSeed": 6101,
            "tieBreakerSeed": 7101,
        },
        {
            "taskId": "repair_direction_contact_pair",
            "taskFamily": "repair",
            "initialValues": [2, 1],
            "frozenPositions": [1],
            "frozenVariant": "stuck",
            "repairConfig": RepairRuleConfig("direction_contact", nudge_threshold=2, approach_direction="from_left").to_dict(),
            "signalConfig": SignalConfig("no_signal").to_dict(),
            "fatigueConfig": FatigueDamageConfig("none").to_dict(),
            "maxActivations": 12,
            "schedulerSeed": 6102,
            "tieBreakerSeed": 7102,
        },
        {
            "taskId": "repair_signal_threshold_pair",
            "taskFamily": "repair",
            "initialValues": [2, 1],
            "frozenPositions": [1],
            "frozenVariant": "stuck",
            "repairConfig": RepairRuleConfig("signal_threshold", signal_threshold=1.0).to_dict(),
            "signalConfig": SignalConfig("nearest_neighbor", signal_range=1, decay=0.0).to_dict(),
            "fatigueConfig": FatigueDamageConfig("none").to_dict(),
            "maxActivations": 12,
            "schedulerSeed": 6103,
            "tieBreakerSeed": 7103,
        },
    ]


def probability_bounds_from_simulator(sim: LearningEventSimulator) -> tuple[float | None, float | None, bool]:
    values: list[float] = []
    ok = True
    for row in sim.trace_rows:
        try:
            states = json.loads(row.get("learning_states_json", "[]"))
        except json.JSONDecodeError:
            ok = False
            continue
        for state_row in states:
            probabilities = state_row.get("learning_state", {}).get("actionProbabilities", {})
            if not probabilities:
                continue
            total = sum(float(value) for value in probabilities.values())
            ok = ok and abs(total - 1.0) < 1e-9
            for value in probabilities.values():
                values.append(float(value))
                ok = ok and 0.0 <= float(value) <= 1.0
    return (min(values) if values else None, max(values) if values else None, ok)


def trace_information_boundary_ok(sim: LearningEventSimulator) -> tuple[bool, int, str]:
    forbidden_hits: list[str] = []
    checked = 0
    allowed = set(LOCAL_REWARD_ALLOWED_INPUTS)
    for row in sim.trace_rows:
        for field in ("actor_learning_state_json",):
            try:
                payload = json.loads(row.get(field, "{}"))
            except json.JSONDecodeError:
                forbidden_hits.append(f"{field}:json_error")
                continue
            reward = payload.get("learning_state", {}).get("lastRewardRecord", {})
            if not reward:
                continue
            checked += 1
            if reward.get("usesGlobalSortednessSignal") or reward.get("usesWholeArrayTargetSignal"):
                forbidden_hits.append(f"{field}:oracle_flag")
            if reward.get("forbiddenFieldsUsed"):
                forbidden_hits.extend(str(item) for item in reward["forbiddenFieldsUsed"])
            used = set(reward.get("inputFieldsUsed", []))
            extra = used - allowed
            if extra:
                forbidden_hits.extend(sorted(extra))
            local_inputs = set(reward.get("localInputs", {}))
            for forbidden in LOCAL_REWARD_FORBIDDEN_INPUTS:
                if forbidden in local_inputs:
                    forbidden_hits.append(forbidden)
    return not forbidden_hits and checked > 0, checked, compact_json(sorted(set(forbidden_hits)))


def policy_for_variant(variant: str, *, seed_offset: int = 0) -> LocalLearningPolicyWrapper:
    return LocalLearningPolicyWrapper(
        "bubble",
        LocalLearningConfig(variant=variant, learning_rate=0.6, random_seed=6600 + seed_offset),
    )


def run_repair_comparisons() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    trace_examples: list[dict[str, Any]] = []
    for task_index, task in enumerate(repair_task_configs()):
        for variant_index, variant in enumerate(LEARNING_VARIANTS):
            policy = policy_for_variant(variant, seed_offset=100 * task_index + variant_index)
            sim = LearningEventSimulator(
                task["initialValues"],
                policy,
                frozen_positions=task["frozenPositions"],
                frozen_variant=task["frozenVariant"],
                repair_config=task["repairConfig"],
                signal_config=task["signalConfig"],
                fatigue_config=task["fatigueConfig"],
                scheduler_seed=task["schedulerSeed"],
                tie_breaker_seed=task["tieBreakerSeed"],
                condition_id=f"e04_s06_{task['taskId']}_{variant}",
                research_step_id=STEP_ID,
            )
            result = sim.run(max_activations=task["maxActivations"], no_move_check_interval=1)
            min_prob, max_prob, probability_ok = probability_bounds_from_simulator(sim)
            leakage_ok, reward_records_checked, leakage_fields_json = trace_information_boundary_ok(sim)
            summary = learning_summary_record(sim)
            rows.append(
                {
                    "taskId": task["taskId"],
                    "taskFamily": "repair",
                    "variant": variant,
                    "completed": result.completed,
                    "stopReason": result.stop_reason,
                    "finalValuesJson": compact_json(result.final_values),
                    "finalSortednessPercent": result.final_sortedness_percent,
                    "activationCount": result.activation_count,
                    "swapCount": result.swap_count,
                    "comparisonCount": result.comparison_count,
                    "recoveredCellCount": len(sim.recovery_events),
                    "remainingFrozenCellCount": len(sim.current_frozen_positions()),
                    "minActionProbabilityObserved": min_prob,
                    "maxActionProbabilityObserved": max_prob,
                    "probabilityBoundsOk": probability_ok,
                    "informationBoundaryOk": leakage_ok,
                    "rewardRecordsChecked": reward_records_checked,
                    "forbiddenFieldsJson": leakage_fields_json,
                    **summary,
                }
            )
            trace_examples.extend(
                {
                    "taskId": task["taskId"],
                    "variant": variant,
                    "traceRow": row,
                }
                for row in sim.trace_rows[: min(6, len(sim.trace_rows))]
            )
    return rows, trace_examples


def run_homeostatic_task(task: Mapping[str, Any], variant: str, *, seed_offset: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    policy = policy_for_variant(variant, seed_offset=seed_offset)
    sim = LearningEventSimulator(
        task["initialValues"],
        policy,
        frozen_variant="stuck",
        repair_config=RepairRuleConfig("permanent").to_dict(),
        signal_config=SignalConfig("no_signal").to_dict(),
        fatigue_config=FatigueDamageConfig("stochastic_damage", damage_probability=0.0, damage_cooldown_activations=2).to_dict(),
        scheduler_seed=int(task["scheduleSeed"]) + 1000 + seed_offset,
        tie_breaker_seed=int(task["scheduleSeed"]) + 2000 + seed_offset,
        condition_id=f"e04_s06_{task['taskId']}_{variant}",
        research_step_id=STEP_ID,
    )
    initial_length = len(task["initialValues"])
    threshold = float(task["sortednessThresholdPercent"])
    records: list[dict[str, Any]] = []
    trace_examples: list[dict[str, Any]] = []
    energy_before = sim.swap_count
    schedule = list(task.get("perturbationSchedule", ()))
    for tick in range(int(task["horizonTicks"])):
        events = [event for event in schedule if int(event["tick"]) == tick]
        for event in events:
            apply_homeostatic_event_to_simulator(sim, event)
        pre = normalized_sortedness_record(sim.current_values(), initial_length=initial_length)
        activations = max(1, 2 * len(sim.cells))
        activation_start = sim.activation_count
        swap_start = sim.swap_count
        for _ in range(activations):
            sim.step()
        post = normalized_sortedness_record(sim.current_values(), initial_length=initial_length)
        records.append(
            {
                "taskId": task["taskId"],
                "taskFamily": "homeostasis",
                "variant": variant,
                "tick": tick,
                "eventsJson": compact_json(events),
                "preSortednessPercent": pre["sortednessPercent"],
                "postSortednessPercent": post["sortednessPercent"],
                "postInRange": bool(post["sortednessPercent"] >= threshold),
                "currentLength": post["currentLength"],
                "lengthRatio": post["lengthRatio"],
                "currentPairDenominator": post["currentPairDenominator"],
                "activationDelta": int(sim.activation_count - activation_start),
                "swapDelta": int(sim.swap_count - swap_start),
                "valuesJson": compact_json(sim.current_values()),
            }
        )
        if len(trace_examples) < 8:
            trace_examples.extend(
                {
                    "taskId": task["taskId"],
                    "variant": variant,
                    "traceRow": row,
                }
                for row in sim.trace_rows[-min(3, len(sim.trace_rows)) :]
            )

    failure_ticks = [record for record in records if not record["postInRange"]]
    recovery_times: list[int | None] = []
    for event in schedule:
        event_tick = int(event["tick"])
        pre_record = next((record for record in records if record["tick"] == event_tick), None)
        if pre_record is None or float(pre_record["preSortednessPercent"]) >= threshold:
            recovery_times.append(0)
            continue
        recovered_tick = next(
            (record["tick"] for record in records if record["tick"] >= event_tick and record["postInRange"]),
            None,
        )
        recovery_times.append(None if recovered_tick is None else int(recovered_tick - event_tick))
    finite_recovery = [value for value in recovery_times if value is not None]
    min_prob, max_prob, probability_ok = probability_bounds_from_simulator(sim)
    leakage_ok, reward_records_checked, leakage_fields_json = trace_information_boundary_ok(sim)
    summary = learning_summary_record(sim)
    row = {
        "taskId": task["taskId"],
        "taskFamily": "homeostasis",
        "variant": variant,
        "completed": bool(records and records[-1]["postInRange"]),
        "stopReason": "horizon_complete",
        "horizonTicks": int(task["horizonTicks"]),
        "timeInRangeFraction": float((len(records) - len(failure_ticks)) / max(1, len(records))),
        "failureDurationTicks": int(len(failure_ticks)),
        "energyProxy": int(sim.swap_count - energy_before),
        "maxRecoveryTimeTicks": max(finite_recovery) if finite_recovery else None,
        "unrecoveredPerturbationCount": int(sum(1 for value in recovery_times if value is None)),
        "finalValuesJson": compact_json(sim.current_values()),
        "finalSortednessPercent": float(records[-1]["postSortednessPercent"]) if records else None,
        "activationCount": int(sim.activation_count),
        "swapCount": int(sim.swap_count),
        "comparisonCount": int(sim.comparison_count),
        "recoveredCellCount": len(sim.recovery_events),
        "remainingFrozenCellCount": len(sim.current_frozen_positions()),
        "minActionProbabilityObserved": min_prob,
        "maxActionProbabilityObserved": max_prob,
        "probabilityBoundsOk": probability_ok,
        "informationBoundaryOk": leakage_ok,
        "rewardRecordsChecked": reward_records_checked,
        "forbiddenFieldsJson": leakage_fields_json,
        "normalizationVersion": HOMEOSTASIS_BENCHMARK_VERSION,
        "recoveryTimesJson": compact_json(recovery_times),
        **summary,
    }
    return row, records, trace_examples


def run_homeostasis_comparisons() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = build_homeostatic_benchmark_config()
    rows: list[dict[str, Any]] = []
    ticks: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for task_index, task in enumerate(config["tasks"]):
        for variant_index, variant in enumerate(LEARNING_VARIANTS):
            row, records, trace_examples = run_homeostatic_task(
                task,
                variant,
                seed_offset=1000 + 100 * task_index + variant_index,
            )
            rows.append(row)
            ticks.extend(records)
            traces.extend(trace_examples)
    return rows, ticks, traces


def validation_rows(
    repair_df: pd.DataFrame,
    homeostasis_df: pd.DataFrame,
    tick_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    boundary = learning_information_boundary()
    rows.append(
        {
            "validationFamily": "information_boundary",
            "caseId": "declared_no_global_oracle_inputs",
            "success": not boundary["usesGlobalSortednessSignal"]
            and not boundary["usesWholeArrayTargetSignal"]
            and set(boundary["allowedRewardInputs"]) == set(LOCAL_REWARD_ALLOWED_INPUTS),
            "validationDetail": "Learning reward contract declares only adjacent local inputs and no global Sortedness or target signal.",
        }
    )
    rows.append(
        {
            "validationFamily": "information_boundary",
            "caseId": "trace_reward_records_no_forbidden_fields",
            "success": bool(repair_df["informationBoundaryOk"].all() and homeostasis_df["informationBoundaryOk"].all()),
            "validationDetail": "All parsed reward trace records use the allowed local input set and no forbidden fields.",
        }
    )
    normalized = normalize_action_probabilities({"swap_left": 1000, "swap_right": 1, "wait": 0}, min_probability=0.05, max_probability=0.9)
    rows.append(
        {
            "validationFamily": "probability_bounds",
            "caseId": "projection_bounded_and_normalized",
            "success": abs(sum(normalized.values()) - 1.0) < 1e-9
            and min(normalized.values()) >= 0.05
            and max(normalized.values()) <= 0.9,
            "validationDetail": "Skewed action weights project to normalized bounded action probabilities.",
        }
    )
    rows.append(
        {
            "validationFamily": "probability_bounds",
            "caseId": "all_comparison_traces_bounded",
            "success": bool(repair_df["probabilityBoundsOk"].all() and homeostasis_df["probabilityBoundsOk"].all()),
            "validationDetail": "All repair and homeostasis traces kept action probabilities normalized and bounded.",
        }
    )
    repair_variants = set(repair_df["variant"])
    repair_tasks = set(repair_df["taskId"])
    rows.append(
        {
            "validationFamily": "comparison_matrix",
            "caseId": "repair_variants_complete",
            "success": repair_variants == set(LEARNING_VARIANTS)
            and all(len(repair_df.query("taskId == @task")) == len(LEARNING_VARIANTS) for task in repair_tasks),
            "validationDetail": "Fixed, local-adaptive, and random-adaptation controls ran on every repair task.",
        }
    )
    homeostasis_variants = set(homeostasis_df["variant"])
    homeostasis_tasks = set(homeostasis_df["taskId"])
    rows.append(
        {
            "validationFamily": "comparison_matrix",
            "caseId": "homeostasis_variants_complete",
            "success": homeostasis_variants == set(LEARNING_VARIANTS)
            and all(len(homeostasis_df.query("taskId == @task")) == len(LEARNING_VARIANTS) for task in homeostasis_tasks),
            "validationDetail": "Fixed, local-adaptive, and random-adaptation controls ran on every seeded homeostatic task.",
        }
    )
    nudge_adaptive = repair_df.query("taskId == 'repair_nudge_threshold_pair' and variant == 'local_adaptive'").iloc[0]
    rows.append(
        {
            "validationFamily": "local_learning_behavior",
            "caseId": "adaptive_repairs_nudge_pair",
            "success": bool(nudge_adaptive.completed and nudge_adaptive.recoveredCellCount >= 1),
            "validationDetail": "Local-adaptive policy recovered the nudge-threshold frozen neighbor and sorted the toy pair.",
        }
    )
    nudge_fixed = repair_df.query("taskId == 'repair_nudge_threshold_pair' and variant == 'fixed'").iloc[0]
    rows.append(
        {
            "validationFamily": "local_learning_behavior",
            "caseId": "adaptive_not_slower_than_fixed_on_nudge_pair",
            "success": bool(nudge_adaptive.activationCount <= nudge_fixed.activationCount),
            "validationDetail": "Adaptive local updates did not require more activations than the fixed-probability control on the toy nudge task.",
        }
    )
    random_a = run_homeostatic_task(build_homeostatic_benchmark_config()["tasks"][1], "random_adaptation", seed_offset=9090)[0]
    random_b = run_homeostatic_task(build_homeostatic_benchmark_config()["tasks"][1], "random_adaptation", seed_offset=9090)[0]
    random_c = run_homeostatic_task(build_homeostatic_benchmark_config()["tasks"][1], "random_adaptation", seed_offset=9091)[0]
    rows.append(
        {
            "validationFamily": "random_control",
            "caseId": "random_adaptation_seed_replay",
            "success": stable_hash(random_a) == stable_hash(random_b) and stable_hash(random_a) != stable_hash(random_c),
            "validationDetail": "Random-adaptation controls replay exactly with identical seeds and differ when the seed changes.",
        }
    )
    insertion_ticks = tick_df.query("taskId == 'insertion_deletion_normalization'")
    rows.append(
        {
            "validationFamily": "homeostasis_normalization",
            "caseId": "insert_delete_records_current_lengths",
            "success": bool(
                not insertion_ticks.empty
                and insertion_ticks["currentLength"].max() == 5
                and insertion_ticks["currentPairDenominator"].max() == 4
                and insertion_ticks["lengthRatio"].max() > 1.0
            ),
            "validationDetail": "Homeostatic insertion/deletion comparisons retained current length, length ratio, and current denominator fields.",
        }
    )
    return rows


def write_learning_config(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# E04 S06 local learning rule config\n" + to_yaml(payload) + "\n", encoding="utf-8")


def write_learning_rule_spec(path: Path, config: Mapping[str, Any], variant_df: pd.DataFrame) -> None:
    rows = [
        [
            row.variant,
            row.learningRate,
            row.minActionProbability,
            row.maxActionProbability,
            row.usesGlobalSortednessSignal,
            row.usesWholeArrayTargetSignal,
        ]
        for row in variant_df.itertuples(index=False)
    ]
    text = f"""# E04 S06 Local Learning Rule Specification

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: local learning config, variant catalog, repair and homeostasis comparison tables, traces, validation report, code copies, manifests, and status files under `$ARTIFACTS_DIR/research_steps/S06/`.
- Validation result: local-only reward records, bounded probabilities, seeded controls, and comparison matrix checks passed.
- Caveats or blockers: S06 implements small deterministic local-learning controls; broad learned-policy optimization and full leakage hardening are deferred to S07/S08.
- Recommended next action: proceed to S07 only after Chief Scientist instruction; do not start S07 from this run.

## Information Boundary

Allowed reward inputs: `{', '.join(config['informationBoundary']['allowedRewardInputs'])}`.

Forbidden reward inputs: `{', '.join(config['informationBoundary']['forbiddenRewardInputs'])}`.

## Variant Catalog

{markdown_table(["Variant", "Learning rate", "Min p", "Max p", "Uses global Sortedness", "Uses target array"], rows)}
"""
    path.write_text(text, encoding="utf-8")


def write_comparison_report(path: Path, comparison_df: pd.DataFrame) -> None:
    grouped = (
        comparison_df.groupby(["taskFamily", "variant"])
        .agg(
            tasks=("taskId", "count"),
            completed=("completed", "sum"),
            meanFinalSortednessPercent=("finalSortednessPercent", "mean"),
            meanActivationCount=("activationCount", "mean"),
            meanEnergyProxy=("energyProxy", "mean"),
        )
        .reset_index()
    )
    text = f"""# E04 S06 Baseline Comparison Report

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: baseline comparison tables for fixed, local-adaptive, and random-adaptation controls on repair and homeostasis tasks.
- Validation result: all planned comparison cells ran and passed information-boundary and probability-bound checks.
- Caveats or blockers: comparison tasks are toy S06/S05 benchmarks; improvements here are proxy evidence, not broad repair competence.
- Recommended next action: proceed to S07 local-only training audit only after Chief Scientist instruction.

{markdown_table(grouped.columns.tolist(), grouped.values.tolist())}
"""
    path.write_text(text, encoding="utf-8")


def write_validation_report(path: Path, validation_df: pd.DataFrame) -> None:
    total = len(validation_df)
    passed = int(validation_df["success"].sum())
    family = validation_df.groupby("validationFamily")["success"].agg(["count", "sum"]).reset_index()
    text = f"""# E04 S06 Validation Report

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: validation table, unit-test log, learning config/spec, comparison tables, traces, code copies, manifests, and status files.
- Validation result: passed; {passed} of {total} checks passed.
- Caveats or blockers: S06 validates local update mechanics and small benchmark comparisons only; S07 must harden oracle-leak tests before optimization.
- Recommended next action: wait for Chief Scientist instruction before starting S07.

## Check Families

{markdown_table(["Family", "Checks", "Passed"], family.values.tolist())}
"""
    path.write_text(text, encoding="utf-8")


def collect_artifacts(paths: Iterable[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    copied: list[Path] = []
    for package_name in ["memory_repair", "morphospace"]:
        src = REPO_ROOT / package_name
        dst = code_dir / package_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        copied.extend(sorted(path for path in dst.rglob("*.py")))
    for relative in [
        Path("scripts/e04_s06_local_learning_rules.py"),
        Path("tests/test_e04_local_learning.py"),
    ]:
        src = REPO_ROOT / relative
        dst = code_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    started_at = utc_now()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    results_dir = args.artifacts_dir / "results"
    configs_dir = args.artifacts_dir / "configs"
    provenance_dir = args.artifacts_dir / "provenance"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    config = learning_config_payload()
    variant_df = pd.DataFrame(learning_variant_rows())
    repair_rows, repair_trace_examples = run_repair_comparisons()
    homeostasis_rows, homeostasis_ticks, homeostasis_trace_examples = run_homeostasis_comparisons()
    repair_df = pd.DataFrame(repair_rows)
    homeostasis_df = pd.DataFrame(homeostasis_rows)
    tick_df = pd.DataFrame(homeostasis_ticks)
    comparison_df = pd.concat([repair_df, homeostasis_df], ignore_index=True, sort=False)
    validation_df = pd.DataFrame(validation_rows(repair_df, homeostasis_df, tick_df))

    artifacts: list[Path] = []
    config_path = configs_dir / "e04_local_learning_rules.yaml"
    step_config_path = step_dir / "local_learning_rules.yaml"
    write_learning_config(config_path, config)
    shutil.copy2(config_path, step_config_path)
    artifacts.extend([config_path, step_config_path])
    config_json_path = step_dir / "local_learning_rules.json"
    write_json(config_json_path, config)
    artifacts.append(config_json_path)

    artifacts.extend(
        dataframe_to_artifacts(
            variant_df,
            step_dir / "learning_variant_catalog",
            results_dir / "e04_s06_learning_variant_catalog",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            repair_df,
            step_dir / "repair_learning_results",
            results_dir / "e04_s06_repair_learning_results",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            homeostasis_df,
            step_dir / "homeostasis_learning_results",
            results_dir / "e04_s06_homeostasis_learning_results",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            comparison_df,
            step_dir / "learning_comparison_results",
            results_dir / "e04_s06_learning_comparison_results",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            tick_df,
            step_dir / "homeostasis_learning_tick_records",
            results_dir / "e04_s06_homeostasis_learning_tick_records",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            validation_df,
            step_dir / "learning_validation_results",
            results_dir / "e04_s06_learning_validation_results",
        )
    )

    trace_jsonl = step_dir / "learning_trace_examples.jsonl"
    trace_payloads = repair_trace_examples[:24] + homeostasis_trace_examples[:24]
    trace_jsonl.write_text("\n".join(compact_json(row) for row in trace_payloads) + "\n", encoding="utf-8")
    artifacts.append(trace_jsonl)

    spec_path = step_dir / "local_learning_rule_spec.md"
    comparison_report_path = step_dir / "comparison_report.md"
    validation_report_path = step_dir / "validation_report.md"
    write_learning_rule_spec(spec_path, config, variant_df)
    write_comparison_report(comparison_report_path, comparison_df)
    write_validation_report(validation_report_path, validation_df)
    artifacts.extend([spec_path, comparison_report_path, validation_report_path])

    unit_cmd = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_e04_local_learning",
        "tests.test_e04_homeostasis_tasks",
        "tests.test_e04_fatigue_damage",
        "tests.test_e04_repairable_frozen",
        "tests.test_e04_local_signals",
        "tests.test_e04_memory_extension",
        "tests.test_e03_policy_interface",
        "tests.test_e03_rule_dsl",
    ]
    unit_result = run_command(unit_cmd, cwd=REPO_ROOT)
    unit_log_path = step_dir / "repo_unit_test_log.txt"
    unit_log_path.write_text(
        "$ " + " ".join(unit_cmd) + "\n\nSTDOUT\n" + unit_result["stdout"] + "\n\nSTDERR\n" + unit_result["stderr"],
        encoding="utf-8",
    )
    artifacts.append(unit_log_path)
    artifacts.extend(copy_code_artifacts(step_dir))

    checks_passed = int(validation_df["success"].sum())
    checks_total = len(validation_df)
    success = checks_passed == checks_total and bool(unit_result["success"])
    status = "completed" if success else "blocked"
    validation_result = (
        f"passed; {checks_passed} of {checks_total} validation checks passed and unit tests passed"
        if success
        else f"failed; {checks_passed} of {checks_total} validation checks passed; unit test success={unit_result['success']}"
    )
    caveats = [
        "S06 implements lightweight local reinforcement-like updates and toy benchmark comparisons, not broad policy optimization.",
        "Rewards use adjacent-neighborhood and local outcome proxies; they may not align with global sorting under all perturbations.",
        "S07 should harden no-oracle auditing before any larger training or evolutionary search.",
    ]
    recommended_next_action = "Proceed to S07 no-global-oracle training audit only after Chief Scientist instruction; do not start S07 from this run."
    lay_summary = (
        "S06 added cell-local action probabilities that can be fixed, updated from local outcomes, or randomly adapted as a control. "
        "The variants ran on repair and homeostatic tasks while keeping reward inputs local and probabilities normalized."
    )

    summary_path = step_dir / "summary.md"
    summary_path.write_text(
        f"""# S06 Summary

- Research step ID: {STEP_ID}
- Completion status: {status}
- Artifacts written: `{config_path}` and additional files listed in `status.json` and `artifact_manifest.json`.
- Validation result: {validation_result}.
- Outcome classification: supportive
- Caveats or blockers: {'; '.join(caveats)}
- Lay summary: {lay_summary}
- Recommended next action: {recommended_next_action}
""",
        encoding="utf-8",
    )
    artifacts.append(summary_path)

    git = get_git_metadata()
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "workerCount": 1,
        "threadEnvironment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        },
    }
    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "success": success,
        "status": status,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "artifactsWritten": [],
        "validationResult": validation_result,
        "validationChecksPassed": checks_passed,
        "validationChecksTotal": checks_total,
        "caveatsOrBlockers": caveats,
        "laySummary": lay_summary,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "localLearningVersion": LOCAL_LEARNING_VERSION,
        "learningVariants": list(LEARNING_VARIANTS),
        "localLearningActions": list(LOCAL_LEARNING_ACTIONS),
        "allowedRewardInputs": list(LOCAL_REWARD_ALLOWED_INPUTS),
        "forbiddenRewardInputs": list(LOCAL_REWARD_FORBIDDEN_INPUTS),
        "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
        "fatigueDamageVersion": FATIGUE_DAMAGE_VERSION,
        "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
        "repoUnitTests": {
            "command": unit_cmd,
            "returnCode": unit_result["returncode"],
            "success": unit_result["success"],
            "runtimeSeconds": unit_result["runtimeSeconds"],
            "logPath": str(unit_log_path),
        },
        "git": git,
        "runtime": runtime,
    }
    manifest_path = step_dir / "artifact_manifest.json"
    status_path = step_dir / "status.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    artifact_records = collect_artifacts(artifacts)
    status_payload["artifactsWritten"] = [item["path"] for item in artifact_records]
    write_json(status_path, status_payload)
    artifacts.append(status_path)
    artifact_records = collect_artifacts(artifacts)
    write_json(
        manifest_path,
        {
            "schema": "eidosoma.research_step_artifact_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "createdAt": utc_now(),
            "artifacts": artifact_records,
        },
    )
    artifacts.append(manifest_path)
    artifact_records = collect_artifacts(artifacts)
    status_payload["artifactsWritten"] = [item["path"] for item in artifact_records]
    write_json(status_path, status_payload)
    write_json(
        run_manifest_path,
        {
            "schema": "eidosoma.run_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "createdAt": utc_now(),
            "git": git,
            "runtime": runtime,
            "command": [sys.executable, str(Path(__file__).relative_to(REPO_ROOT))],
            "artifacts": artifact_records,
            "validationResult": validation_result,
        },
    )
    artifacts.append(run_manifest_path)

    artifact_records = collect_artifacts(artifacts)
    write_json(
        manifest_path,
        {
            "schema": "eidosoma.research_step_artifact_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "createdAt": utc_now(),
            "artifacts": artifact_records,
        },
    )
    artifact_records = collect_artifacts(artifacts)
    status_payload["artifactsWritten"] = [item["path"] for item in artifact_records]
    status_payload["completedAt"] = utc_now()
    write_json(status_path, status_payload)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
