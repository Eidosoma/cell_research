"""Centralized global-state repair baseline for E04 S14.

This module intentionally implements a non-local controller.  It sees the full
array state, frozen positions, and target sorted order, so every output row is
marked as centralized/global-access and excluded from local-only claims.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evolutionary_search import fitness_from_row
from .homeostasis import PerturbationSpec, RepairController, _apply_perturbation, _reindex_cells
from .fatigue_damage import FatigueDamageController, delayed_gratification_from_sortedness
from .no_oracle_protocol import CENTRALIZED_BASELINE_TYPE
from .overfitting_transfer import S13TransferConfig
from .signaling import SignalBank, SignalConfig
from ..e02.deterministic_simulator import (
    EventTracingStatusProbe,
    SimulatorConfig,
    _build_cells,
    cell_labels,
    cell_state_signature,
    cell_values,
    frozen_positions,
    monotonicity_error_count,
    stable_json_sha256,
    sortedness_percent,
)


S14_POLICY_ID = "centralized_global_state_repair_v1"
S14_PROTOCOL_ID = "e04_s14_centralized_global_repair_baseline_v1"
LOCAL_COMPARATOR_GROUP = "best_local_s13_neighbor_memory"
CENTRALIZED_COMPARATOR_GROUP = "centralized_global_state_repair"
GLOBAL_ACCESS_FIELDS = (
    "full_array_values",
    "all_cell_statuses",
    "global_frozen_positions",
    "target_sorted_order",
    "full_perturbation_schedule",
)


def load_s13_configs(config_path: str | Path) -> list[dict[str, Any]]:
    """Load the predefined S13 config records used for matched S14 runs."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    configs = payload.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError(f"No S13 configs found in {config_path}")
    return [dict(record) for record in configs]


def s13_config_from_record(record: Mapping[str, Any]) -> S13TransferConfig:
    """Reconstruct an S13 transfer config from its frozen JSON record."""

    required = (
        "split",
        "benchmark_family",
        "generalization_axis",
        "scenario_name",
        "values",
        "array_size",
        "task_name",
        "repair_rule",
        "reliability_mode",
        "max_events",
        "activation_seed",
        "schedule_seed",
        "policy_seed",
        "schedule_variant",
        "target_sortedness_percent",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"S13 config record is missing fields: {missing}")
    return S13TransferConfig(
        split=str(record["split"]),
        benchmark_family=str(record["benchmark_family"]),
        generalization_axis=str(record["generalization_axis"]),
        scenario_name=str(record["scenario_name"]),
        values=tuple(int(value) for value in record["values"]),
        algorithm=str(record.get("algorithm", "bubble")),
        task_name=str(record["task_name"]),
        repair_rule=str(record["repair_rule"]),
        reliability_mode=str(record["reliability_mode"]),
        max_events=int(record["max_events"]),
        activation_seed=int(record["activation_seed"]),
        schedule_seed=int(record["schedule_seed"]),
        policy_seed=int(record["policy_seed"]),
        schedule_variant=str(record["schedule_variant"]),
        target_sortedness_percent=float(record["target_sortedness_percent"]),
        nudge_threshold=int(record.get("nudge_threshold", 2)),
        fatigue_threshold=int(record.get("fatigue_threshold", 3)),
        recovery_events=int(record.get("recovery_events", 4)),
        damage_threshold=int(record.get("damage_threshold", 4)),
        sort_direction=str(record.get("sort_direction", "increasing")),
        activation_distribution=str(record.get("activation_distribution", "uniform_active")),
    )


def schedule_from_record(record: Mapping[str, Any]) -> tuple[PerturbationSpec, ...]:
    """Rehydrate S13's predefined perturbation schedule."""

    schedule_json = record.get("schedule_json")
    if not isinstance(schedule_json, str):
        raise ValueError("S13 config record has no schedule_json string")
    raw = json.loads(schedule_json)
    return tuple(PerturbationSpec(**dict(item)) for item in raw)


def inversion_count(values: Sequence[int], *, direction: str = "increasing") -> int:
    """Count pairwise inversions in a short array."""

    count = 0
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            if (direction == "increasing" and left > right) or (
                direction == "decreasing" and left < right
            ):
                count += 1
    return count


def _global_reorder_to_target(cells: list[Any], *, direction: str = "increasing") -> int:
    """Globally reorder cells to target order and return adjacent-swap-equivalent work."""

    values = cell_values(cells)
    adjacent_equivalent_swaps = inversion_count(values, direction=direction)
    if adjacent_equivalent_swaps <= 0:
        return 0
    cells.sort(key=lambda cell: int(cell.value), reverse=direction == "decreasing")
    _reindex_cells(cells)
    return adjacent_equivalent_swaps


def _globally_unfreeze_all(
    cells: Sequence[Any],
    repair_controller: RepairController,
    cell_status: Any,
) -> list[int]:
    """Use global state access to repair all currently frozen cells."""

    repaired: list[int] = []
    for cell in cells:
        if cell.status != cell_status.FREEZE:
            continue
        repaired.append(int(cell.threadID))
        repair_controller._unfreeze(  # noqa: SLF001 - intentionally uses repair instrumentation.
            cell,
            trigger="centralized_global_state_repair",
            trigger_value="full_array_status_scan",
            actor=None,
            direction_from_actor=None,
        )
    return repaired


def _offline_trace_metrics(
    *,
    event_log: Sequence[Mapping[str, Any]],
    config: S13TransferConfig,
    perturbation_log: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute S14 metrics offline from the centralized trace."""

    sortedness_values = [float(row["sortedness_percent"]) for row in event_log]
    target_flags = [value >= config.target_sortedness_percent for value in sortedness_values]
    recovery_times: list[int] = []
    unrecovered = 0
    for perturbation in perturbation_log:
        event_step = int(perturbation["event_step"])
        if event_step >= len(target_flags):
            unrecovered += 1
            continue
        recovered_at = None
        for idx in range(event_step, len(target_flags)):
            if target_flags[idx]:
                recovered_at = idx
                break
        if recovered_at is None:
            unrecovered += 1
        else:
            recovery_times.append(recovered_at - event_step)

    delayed = delayed_gratification_from_sortedness(sortedness_values)
    max_recovery_streak = 0
    current_streak = 0
    for flag in target_flags:
        if flag:
            current_streak += 1
            max_recovery_streak = max(max_recovery_streak, current_streak)
        else:
            current_streak = 0
    final_values = list(event_log[-1]["current_values"]) if event_log else []
    final_sortedness = float(sortedness_values[-1]) if sortedness_values else math.nan
    final_error = monotonicity_error_count(final_values) if final_values else math.nan
    mean_recovery = float(np.mean(recovery_times)) if recovery_times else math.nan
    return {
        "final_values": final_values,
        "final_sortedness_percent": final_sortedness,
        "final_monotonicity_error_count": int(final_error) if not math.isnan(final_error) else math.nan,
        "time_in_target_fraction": float(np.mean(target_flags)) if target_flags else math.nan,
        "recovered_perturbations": int(len(recovery_times)),
        "recovered_perturbation_count": int(len(recovery_times)),
        "unrecovered_perturbations": int(unrecovered),
        "unrecovered_perturbation_count": int(unrecovered),
        "mean_recovery_events": mean_recovery,
        "min_sortedness_percent": float(np.min(sortedness_values)) if sortedness_values else math.nan,
        "mean_sortedness_percent": float(np.mean(sortedness_values)) if sortedness_values else math.nan,
        "sortedness_auc_percent": float(np.mean(sortedness_values)) if sortedness_values else math.nan,
        "delayed_gratification_auc": float(delayed["dg_primary"]),
        "late_recovery_auc": float(delayed["dg_total_recovery"]),
        "max_recovery_streak": int(max_recovery_streak),
        "dg_metrics_json": json.dumps(delayed, sort_keys=True),
        "final_state_signature": cell_state_signature_from_values(final_values),
    }


def cell_state_signature_from_values(values: Sequence[int]) -> str:
    """Create a compact value-only final-state signature for derived rows."""

    return stable_json_sha256({"values": list(values)})[:16]


def repair_quality_score_from_row(row: Mapping[str, Any]) -> float:
    """Score repaired perturbations with explicit emphasis on freeze repair."""

    freeze_count = float(row.get("freeze_perturbation_count", 0) or 0)
    unfreeze_count = float(row.get("unfreeze_count", 0) or 0)
    if freeze_count > 0:
        return float(min(1.0, unfreeze_count / freeze_count))
    perturbations = float(row.get("perturbation_count", 0) or 0)
    recovered = float(
        row.get("recovered_perturbation_count", row.get("recovered_perturbations", 0)) or 0
    )
    if perturbations <= 0:
        return 1.0
    return float(min(1.0, recovered / perturbations))


def recovery_speed_score_from_row(row: Mapping[str, Any]) -> float:
    """Normalize mean recovery delay so higher is faster."""

    mean_recovery = row.get("mean_recovery_events")
    max_events = float(row.get("max_events", 0) or 0)
    unrecovered = float(
        row.get("unrecovered_perturbation_count", row.get("unrecovered_perturbations", 0)) or 0
    )
    if max_events <= 0:
        return math.nan
    if mean_recovery is None or pd.isna(mean_recovery):
        return 1.0 if unrecovered == 0 else 0.0
    return float(max(0.0, 1.0 - float(mean_recovery) / max_events))


def robustness_score_from_row(row: Mapping[str, Any]) -> float:
    """Combine average and worst observed sortedness as a bounded robustness proxy."""

    mean_sortedness = float(row.get("mean_sortedness_percent", row.get("sortedness_auc_percent", 0)) or 0)
    min_sortedness = float(row.get("min_sortedness_percent", 0) or 0)
    return float(np.clip(0.5 * mean_sortedness / 100.0 + 0.5 * min_sortedness / 100.0, 0.0, 1.0))


def comparison_repair_score_from_row(row: Mapping[str, Any]) -> float:
    """Compute S08-style repair score without the oracle-access penalty."""

    comparable = dict(row)
    comparable["uses_global_oracle"] = False
    return fitness_from_row(comparable)


def simulate_centralized_repair(
    config: S13TransferConfig,
    schedule: Sequence[PerturbationSpec],
    *,
    s13_config_id: str | None = None,
) -> dict[str, Any]:
    """Run the centralized/global-state baseline on one frozen S13 config."""

    base_config = SimulatorConfig(
        values=tuple(config.values),
        algorithm=config.algorithm,
        frozen_indices=(),
        frozen_semantics="passive",
        activation_seed=config.activation_seed,
        policy_seed=config.policy_seed,
        activation_distribution=config.activation_distribution,
        max_events=config.max_events,
        stop_when_sorted=False,
        convergence_criterion="none",
        sort_direction=config.sort_direction,
    )
    probe = EventTracingStatusProbe()
    cells, local_cell_status = _build_cells(base_config, probe)
    signal_bank = SignalBank(len(cells), SignalConfig(enabled=False))
    repair_controller = RepairController(cells, local_cell_status, config.to_repair_config(), signal_bank)
    reliability = FatigueDamageController(config.to_fatigue_config())
    reliability.initialize(cells)
    schedule_by_step: dict[int, list[PerturbationSpec]] = {}
    for spec in schedule:
        schedule_by_step.setdefault(int(spec.event_step), []).append(spec)
    next_thread_id = max(int(cell.threadID) for cell in cells) + 1
    target_values = tuple(sorted(config.values))

    event_log: list[dict[str, Any]] = []
    perturbation_log: list[dict[str, Any]] = []
    action_counts = {
        "centralized_global_reorder": 0,
        "centralized_adjacent_equivalent_swaps": 0,
        "centralized_global_unfreeze": 0,
        "centralized_idle": 0,
        "global_candidate_evaluations": 0,
    }
    swap_count = 0
    comparison_count = 0

    def append_event(event_step: int, action: str, repaired_thread_ids: Sequence[int]) -> None:
        values = cell_values(cells)
        event_log.append(
            {
                "event_step": int(event_step),
                "action": action,
                "current_values": values,
                "labels": cell_labels(cells),
                "frozen_positions": frozen_positions(cells, local_cell_status),
                "repaired_thread_ids": list(repaired_thread_ids),
                "sortedness_percent": sortedness_percent(values),
                "monotonicity_error_count": monotonicity_error_count(values),
                "state_signature": cell_state_signature(cells),
            }
        )

    append_event(0, "initial", [])
    for event_step in range(1, config.max_events + 1):
        for spec in schedule_by_step.get(event_step, []):
            perturbation_record, next_thread_id = _apply_perturbation(
                spec=spec,
                cells=cells,
                cell_status=local_cell_status,
                repair_controller=repair_controller,
                reliability=reliability,
                next_thread_id=next_thread_id,
            )
            perturbation_log.append(perturbation_record)
        repair_controller.before_event(event_step)
        reliability.before_event(event_step, cells)

        repaired = _globally_unfreeze_all(cells, repair_controller, local_cell_status)
        if repaired:
            action_counts["centralized_global_unfreeze"] += len(repaired)

        candidate_evaluations = max(0, len(cells) * (len(cells) - 1) // 2)
        action_counts["global_candidate_evaluations"] += candidate_evaluations
        comparison_count += candidate_evaluations
        action = "centralized_idle"
        reorder_swaps = _global_reorder_to_target(cells, direction=config.sort_direction)
        if reorder_swaps > 0:
            swap_count += int(reorder_swaps)
            action_counts["centralized_global_reorder"] += 1
            action_counts["centralized_adjacent_equivalent_swaps"] += int(reorder_swaps)
            action = "centralized_global_reorder"
        else:
            action_counts["centralized_idle"] += 1
        append_event(event_step, action, repaired)

    metrics = _offline_trace_metrics(
        event_log=event_log,
        config=config,
        perturbation_log=perturbation_log,
    )
    perturbation_types = [str(item.get("type", "")) for item in perturbation_log]
    row: dict[str, Any] = {
        **config.to_dict(),
        "s13_config_id": s13_config_id or stable_json_sha256(config.to_dict())[:16],
        "policy_id": S14_POLICY_ID,
        "policy_family": "centralized_global_state_repair",
        "candidate_type": "centralized_global_oracle_baseline",
        "comparison_group": CENTRALIZED_COMPARATOR_GROUP,
        "protocol_id": S14_PROTOCOL_ID,
        "s07_projection_version": "not_applicable_global_access_baseline",
        "baseline_type": CENTRALIZED_BASELINE_TYPE,
        "centralized_baseline": True,
        "uses_global_oracle": True,
        "global_state_access": True,
        "eligible_for_local_only_claims": False,
        "local_only_policy": False,
        "local_only_claim_group": False,
        "target_derived_feature_excluded": False,
        "s07_local_only_inputs": False,
        "forbidden_field_audit_passed": False,
        "global_access_fields_json": json.dumps(list(GLOBAL_ACCESS_FIELDS), sort_keys=True),
        "global_access_audit_json": json.dumps(
            {
                "uses_global_oracle": True,
                "eligible_for_local_only_claims": False,
                "accessed_fields": list(GLOBAL_ACCESS_FIELDS),
                "rationale": "Centralized ceiling baseline for S14; not a local policy.",
            },
            sort_keys=True,
        ),
        "feature_audit_summary_json": json.dumps(
            {
                "protocol_id": S14_PROTOCOL_ID,
                "uses_global_oracle": True,
                "eligible_for_local_only_claims": False,
                "target_derived_fields": ["target_sorted_order"],
                "forbidden_for_local_policy": True,
            },
            sort_keys=True,
        ),
        "signal_audit_json": json.dumps(
            {
                "uses_signal_channels": False,
                "targetDerivedHitCount": 1,
                "oracleHintHitCount": 1,
                "note": "Centralized/global baseline intentionally violates local-only policy view.",
            },
            sort_keys=True,
        ),
        "target_values_json": json.dumps(list(target_values)),
        "schedule_json": json.dumps([spec.to_dict() for spec in schedule], sort_keys=True),
        "schedule_sha256": stable_json_sha256([spec.to_dict() for spec in schedule]),
        "perturbation_log_json": json.dumps(perturbation_log, sort_keys=True),
        "event_trace_json": json.dumps(event_log, sort_keys=True),
        "action_counts_json": json.dumps(action_counts, sort_keys=True),
        "global_repair_action_count": int(action_counts["centralized_global_unfreeze"]),
        "global_reorder_action_count": int(action_counts["centralized_global_reorder"]),
        "adjacent_equivalent_swap_count": int(action_counts["centralized_adjacent_equivalent_swaps"]),
        "unfreeze_count": int(action_counts["centralized_global_unfreeze"]),
        "swap_count": int(swap_count),
        "comparison_count": int(comparison_count),
        "energy_total": float(swap_count),
        "perturbation_count": int(len(perturbation_log)),
        "freeze_perturbation_count": int(sum(item == "freeze" for item in perturbation_types)),
        "insert_perturbation_count": int(sum(item == "insert" for item in perturbation_types)),
        "swap_perturbation_count": int(sum(item == "swap" for item in perturbation_types)),
        "repair_summary_json": json.dumps(repair_controller.summary(), sort_keys=True),
        "reliability_summary_json": json.dumps(reliability.summary(), sort_keys=True),
        "state_trace_length": int(len(event_log)),
        "event_count": int(config.max_events),
        "cumulative_impairment_count": int(len(reliability.impairment_log)),
    }
    row.update(metrics)
    row["repair_quality_score"] = repair_quality_score_from_row(row)
    row["recovery_speed_score"] = recovery_speed_score_from_row(row)
    row["robustness_score"] = robustness_score_from_row(row)
    row["fitness_score"] = fitness_from_row(row)
    row["comparison_repair_score"] = comparison_repair_score_from_row(row)
    row["fitness_score_note"] = "S08 oracle penalty retained; use comparison_repair_score for S14 central/local repair comparison."
    row["s14_row_id"] = stable_json_sha256(
        {
            "s13_config_id": row["s13_config_id"],
            "policy_id": S14_POLICY_ID,
            "schedule_sha256": row["schedule_sha256"],
        }
    )[:16]
    return row


def select_best_local_s13_rows(s13_results: pd.DataFrame) -> pd.DataFrame:
    """Select one matched best local S13/S09-supported policy per S13 config."""

    df = s13_results.copy()
    if "fitness_score" not in df.columns:
        df["fitness_score"] = df.apply(fitness_from_row, axis=1)
    mask = (
        (df.get("memory_ablation") == "neighbor_memory")
        & (df.get("eligible_for_local_only_claims") == True)  # noqa: E712
        & (df.get("uses_global_oracle") == False)  # noqa: E712
        & (df.get("centralized_baseline") == False)  # noqa: E712
    )
    candidates = df.loc[mask].copy()
    if candidates.empty:
        raise ValueError("No eligible S13 neighbor-memory local comparator rows found")
    idx = candidates.groupby("s13_config_id")["fitness_score"].idxmax()
    selected = candidates.loc[idx].copy().reset_index(drop=True)
    selected["comparison_group"] = LOCAL_COMPARATOR_GROUP
    selected["global_state_access"] = False
    selected["local_only_policy"] = True
    selected["local_only_claim_group"] = True
    selected["global_access_fields_json"] = json.dumps([])
    selected["global_access_audit_json"] = selected.apply(
        lambda _row: json.dumps(
            {
                "uses_global_oracle": False,
                "eligible_for_local_only_claims": True,
                "accessed_fields": [],
                "rationale": "Matched S13 best local neighbor-memory comparator.",
            },
            sort_keys=True,
        ),
        axis=1,
    )
    selected["repair_quality_score"] = selected.apply(repair_quality_score_from_row, axis=1)
    selected["recovery_speed_score"] = selected.apply(recovery_speed_score_from_row, axis=1)
    selected["robustness_score"] = selected.apply(robustness_score_from_row, axis=1)
    selected["comparison_repair_score"] = selected.apply(comparison_repair_score_from_row, axis=1)
    selected["fitness_score_note"] = "Local row; fitness_score and comparison_repair_score are identical unless prior penalties are present."
    selected["s14_row_id"] = selected.apply(
        lambda row: stable_json_sha256(
            {
                "s13_config_id": row["s13_config_id"],
                "policy_id": row["policy_id"],
                "schedule_sha256": row["schedule_sha256"],
                "comparison_group": LOCAL_COMPARATOR_GROUP,
            }
        )[:16],
        axis=1,
    )
    return selected


def run_s14_comparison(
    *,
    s13_results_path: str | Path,
    s13_configs_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run S14 and return full rows, paired deltas, and grouped summaries."""

    config_records = load_s13_configs(s13_configs_path)
    s13_results = pd.read_parquet(s13_results_path)
    local_rows = select_best_local_s13_rows(s13_results)

    central_rows = [
        simulate_centralized_repair(
            s13_config_from_record(record),
            schedule_from_record(record),
            s13_config_id=str(record.get("s13_config_id", record.get("config_id"))),
        )
        for record in config_records
    ]
    central_df = pd.DataFrame(central_rows)

    expected_ids = {str(record.get("s13_config_id", record.get("config_id"))) for record in config_records}
    missing_local = sorted(expected_ids - set(local_rows["s13_config_id"].astype(str)))
    if missing_local:
        raise ValueError(f"Missing matched local comparator rows for S13 configs: {missing_local}")
    local_rows = local_rows[local_rows["s13_config_id"].astype(str).isin(expected_ids)].copy()

    full = pd.concat([local_rows, central_df], ignore_index=True, sort=False)
    paired = paired_comparison_deltas(full)
    summary = grouped_s14_summary(full, paired)
    return full, paired, summary


def paired_comparison_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Create matched central-minus-local deltas by frozen S13 config."""

    metrics = [
        "comparison_repair_score",
        "fitness_score",
        "final_sortedness_percent",
        "time_in_target_fraction",
        "repair_quality_score",
        "energy_total",
        "mean_recovery_events",
        "recovery_speed_score",
        "robustness_score",
        "unrecovered_perturbations",
    ]
    local = df[df["comparison_group"] == LOCAL_COMPARATOR_GROUP].copy()
    central = df[df["comparison_group"] == CENTRALIZED_COMPARATOR_GROUP].copy()
    key_cols = [
        "s13_config_id",
        "split",
        "benchmark_family",
        "generalization_axis",
        "scenario_name",
        "schedule_sha256",
    ]
    merged = central[key_cols + metrics].merge(
        local[key_cols + metrics + ["policy_id"]],
        on=key_cols,
        suffixes=("_centralized", "_local"),
        how="inner",
    )
    for metric in metrics:
        merged[f"delta_{metric}_centralized_minus_local"] = (
            merged[f"{metric}_centralized"] - merged[f"{metric}_local"]
        )
    merged["local_policy_id"] = merged.pop("policy_id")
    merged["fitness_retention_local_vs_centralized"] = (
        merged["fitness_score_local"] / merged["fitness_score_centralized"].replace(0, np.nan)
    )
    merged["comparison_repair_score_retention_local_vs_centralized"] = (
        merged["comparison_repair_score_local"]
        / merged["comparison_repair_score_centralized"].replace(0, np.nan)
    )
    return merged


def grouped_s14_summary(df: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """Summarize central/local absolute metrics and matched deltas."""

    absolute = (
        df.groupby(["comparison_group", "split", "generalization_axis"], dropna=False)
        .agg(
            n=("s13_config_id", "count"),
            mean_comparison_repair_score=("comparison_repair_score", "mean"),
            mean_fitness=("fitness_score", "mean"),
            mean_final_sortedness=("final_sortedness_percent", "mean"),
            mean_time_in_target=("time_in_target_fraction", "mean"),
            mean_repair_quality=("repair_quality_score", "mean"),
            mean_energy=("energy_total", "mean"),
            mean_recovery_events=("mean_recovery_events", "mean"),
            mean_recovery_speed=("recovery_speed_score", "mean"),
            mean_robustness=("robustness_score", "mean"),
        )
        .reset_index()
    )
    delta = (
        paired.groupby(["split", "generalization_axis"], dropna=False)
        .agg(
            n=("s13_config_id", "count"),
            mean_delta_fitness=("delta_fitness_score_centralized_minus_local", "mean"),
            mean_delta_comparison_repair_score=(
                "delta_comparison_repair_score_centralized_minus_local",
                "mean",
            ),
            mean_delta_final_sortedness=(
                "delta_final_sortedness_percent_centralized_minus_local",
                "mean",
            ),
            mean_delta_repair_quality=("delta_repair_quality_score_centralized_minus_local", "mean"),
            mean_delta_energy=("delta_energy_total_centralized_minus_local", "mean"),
            mean_delta_recovery_events=(
                "delta_mean_recovery_events_centralized_minus_local",
                "mean",
            ),
            mean_fitness_retention=("fitness_retention_local_vs_centralized", "mean"),
            mean_comparison_repair_score_retention=(
                "comparison_repair_score_retention_local_vs_centralized",
                "mean",
            ),
        )
        .reset_index()
    )
    delta.insert(0, "comparison_group", "matched_delta_centralized_minus_local")
    return pd.concat([absolute, delta], ignore_index=True, sort=False)


def validate_s14_outputs(df: pd.DataFrame, paired: pd.DataFrame) -> dict[str, Any]:
    """Run claim-boundary and matched-seed validation checks."""

    checks: dict[str, Any] = {}
    central = df[df["comparison_group"] == CENTRALIZED_COMPARATOR_GROUP]
    local = df[df["comparison_group"] == LOCAL_COMPARATOR_GROUP]
    checks["has_centralized_and_local_rows"] = {
        "passed": bool(len(central) > 0 and len(local) > 0),
        "centralized_rows": int(len(central)),
        "local_rows": int(len(local)),
    }
    checks["centralized_rows_labeled_global_access"] = {
        "passed": bool(
            (central["centralized_baseline"] == True).all()  # noqa: E712
            and (central["uses_global_oracle"] == True).all()  # noqa: E712
            and (central["global_state_access"] == True).all()  # noqa: E712
            and (central["eligible_for_local_only_claims"] == False).all()  # noqa: E712
            and (central["baseline_type"] == CENTRALIZED_BASELINE_TYPE).all()
        ),
    }
    checks["local_rows_remain_local_only"] = {
        "passed": bool(
            (local["centralized_baseline"] == False).all()  # noqa: E712
            and (local["uses_global_oracle"] == False).all()  # noqa: E712
            and (local["global_state_access"] == False).all()  # noqa: E712
            and (local["eligible_for_local_only_claims"] == True).all()  # noqa: E712
        ),
    }
    checks["global_rows_excluded_from_local_claims"] = {
        "passed": bool(
            df.loc[df["uses_global_oracle"] == True, "eligible_for_local_only_claims"].eq(False).all()
        ),
    }
    per_config = df.groupby("s13_config_id")["comparison_group"].nunique()
    checks["matched_config_pairs"] = {
        "passed": bool((per_config == 2).all() and len(paired) == per_config.shape[0]),
        "paired_configs": int(len(paired)),
    }
    schedule_counts = df.groupby("s13_config_id")["schedule_sha256"].nunique()
    checks["matched_schedule_hashes"] = {
        "passed": bool((schedule_counts == 1).all()),
        "unique_schedule_mismatches": int((schedule_counts != 1).sum()),
    }
    metric_cols = [
        "fitness_score",
        "comparison_repair_score",
        "repair_quality_score",
        "energy_total",
        "recovery_speed_score",
        "robustness_score",
    ]
    checks["required_metric_columns_complete"] = {
        "passed": bool(df[metric_cols].notna().all().all()),
        "metrics": metric_cols,
    }
    checks["all_passed"] = all(bool(item.get("passed")) for item in checks.values() if isinstance(item, dict))
    return checks


def write_s14_figure(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Write the S14 central-vs-local comparison figure."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    absolute = summary[
        summary["comparison_group"].isin([LOCAL_COMPARATOR_GROUP, CENTRALIZED_COMPARATOR_GROUP])
    ].copy()
    metric_map = {
        "mean_comparison_repair_score": "Repair score",
        "mean_repair_quality": "Repair quality",
        "mean_recovery_speed": "Recovery speed",
        "mean_robustness": "Robustness",
    }
    plot_rows = absolute.melt(
        id_vars=["comparison_group", "split"],
        value_vars=list(metric_map),
        var_name="metric",
        value_name="value",
    )
    plot_rows["metric_label"] = plot_rows["metric"].map(metric_map)
    plot_rows["policy"] = plot_rows["comparison_group"].map(
        {
            LOCAL_COMPARATOR_GROUP: "Best local S13/S09",
            CENTRALIZED_COMPARATOR_GROUP: "Centralized global",
        }
    )
    grouped = (
        plot_rows.groupby(["policy", "metric_label"], dropna=False)["value"].mean().reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    metrics = list(metric_map.values())
    policies = ["Best local S13/S09", "Centralized global"]
    x = np.arange(len(metrics))
    width = 0.34
    colors = {"Best local S13/S09": "#2f6b5f", "Centralized global": "#9b3f38"}
    for offset, policy in zip([-width / 2, width / 2], policies, strict=True):
        values = [
            float(
                grouped.loc[
                    (grouped["policy"] == policy) & (grouped["metric_label"] == metric),
                    "value",
                ].mean()
            )
            for metric in metrics
        ]
        ax.bar(x + offset, values, width, label=policy, color=colors[policy])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("E04 S14 centralized/global repair baseline vs matched local policies")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def classify_s14_outcome(paired: pd.DataFrame) -> tuple[str, str]:
    """Classify whether local policies approach the centralized ceiling."""

    if paired.empty:
        return "null", "No matched central/local pairs were produced."
    retention = float(paired["comparison_repair_score_retention_local_vs_centralized"].mean())
    repair_gap = float(paired["delta_repair_quality_score_centralized_minus_local"].mean())
    repair_score_gap = float(paired["delta_comparison_repair_score_centralized_minus_local"].mean())
    if retention >= 0.85 and repair_gap <= 0.20:
        return (
            "supportive",
            (
                "Best local S13/S09 policies retained at least 85% of the centralized "
                "comparison repair ceiling with limited repair-quality gap."
            ),
        )
    if retention < 0.75 or repair_score_gap > 0.15:
        return (
            "constraining/contradictory",
            (
                "The centralized/global-access baseline substantially exceeded the best "
                "local policies, narrowing local-only claims to a partial mechanism."
            ),
        )
    return (
        "null",
        "The central-local gap was intermediate and does not cleanly support or reject local sufficiency.",
    )
