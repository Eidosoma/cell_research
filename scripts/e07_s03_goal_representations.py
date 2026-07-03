#!/usr/bin/env python3
"""Build E07 S03 abstract goal representations."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e07.goal_schema import (  # noqa: E402
    GoalRepresentation,
    dataframe_from_goal_records,
    parse_json_list,
    sha256_file,
    stable_hash,
    validate_goal_table,
)


STEP_ID = "S03"
STEP_NUMBER = 3
EXPERIMENT_ID = "E07"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-artifacts-dir", type=Path, default=Path("/previous-artifacts"))
    parser.add_argument("--s01-world-inventory", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e07_world_inventory.csv")
    parser.add_argument("--s02-policy-table", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e07_policy_representations.parquet")
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def source_artifact(path: Path) -> str:
    return str(path)


def sorted_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))


def world_ids(world_inventory: pd.DataFrame, *, experiment: str | None = None, families: Sequence[str] = (), task_ids: Sequence[str] = ()) -> tuple[str, ...]:
    frame = world_inventory
    if experiment:
        frame = frame[frame["experiment_id"] == experiment]
    if families:
        family_set = set(families)
        frame = frame[frame["world_family"].isin(family_set)]
    if task_ids:
        task_set = set(task_ids)
        frame = frame[frame["task_id"].isin(task_set)]
    return sorted_unique(frame["world_id"].astype(str).tolist())


def world_ids_matching(world_inventory: pd.DataFrame, experiment: str, predicate) -> tuple[str, ...]:
    frame = world_inventory[world_inventory["experiment_id"] == experiment]
    matched = [row["world_id"] for row in frame.to_dict(orient="records") if predicate(row)]
    return sorted_unique(matched)


def e05_target_worlds(world_inventory: pd.DataFrame, target_id: str) -> tuple[str, ...]:
    return world_ids(world_inventory, experiment="E05", task_ids=(target_id,))


def e05_task_worlds(world_inventory: pd.DataFrame, benchmark_task_id: str) -> tuple[str, ...]:
    return world_ids(world_inventory, experiment="E05", task_ids=(benchmark_task_id,))


def e06_step_worlds(world_inventory: pd.DataFrame, step_ids: Sequence[str]) -> tuple[str, ...]:
    step_set = set(step_ids)
    return sorted_unique(
        [
            row["world_id"]
            for row in world_inventory[world_inventory["experiment_id"] == "E06"].to_dict(orient="records")
            if str(row["source_step_id"]) in step_set
        ]
    )


def status_for(partial: bool, target_direction: str, source_validation: str = "source_artifacts_present") -> dict[str, str]:
    return {
        "direction_validation_status": "partial_direction_declared" if partial or target_direction == "partial_unknown" else "explicit_direction_declared",
        "unit_validation_status": "partial_unit_declared" if partial else "explicit_unit_declared",
        "source_validation_status": source_validation,
    }


def goal_record(
    *,
    source_experiment_id: str,
    source_step_id: str,
    source_goal_id: str,
    display_name: str,
    goal_family: str,
    goal_kind: str,
    abstraction_kind: str,
    target_structure: str,
    target_direction: str,
    unit: str,
    primary_metric_id: str,
    metric_family: str,
    target_value: str,
    linked_world_ids: Sequence[str],
    source_artifacts: Sequence[str],
    partial: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    zero_point_definition: str = "",
    energy_function: str = "",
    constraint_predicate: str = "",
    conflict_group_id: str = "",
    conflicts_with_goal_ids: Sequence[str] = (),
    compatible_with_goal_ids: Sequence[str] = (),
    linked_policy_ids: Sequence[str] = (),
    limitations: Sequence[str] = (),
    metric_contract: Mapping[str, Any] | None = None,
    representation_payload: Mapping[str, Any] | None = None,
    target_hash: str | None = None,
    source_record_id: str | None = None,
) -> GoalRepresentation:
    statuses = status_for(partial, target_direction)
    conflict_status = "encoded" if conflicts_with_goal_ids else ("partial_no_conflict_source" if partial else "no_known_conflict")
    return GoalRepresentation(
        source_experiment_id=source_experiment_id,
        source_step_id=source_step_id,
        source_record_id=source_record_id or f"{source_experiment_id}:{source_goal_id}",
        source_goal_id=source_goal_id,
        display_name=display_name,
        goal_family=goal_family,
        goal_kind=goal_kind,
        abstraction_kind=abstraction_kind,
        target_structure=target_structure,
        representation_status="partial_explicit" if partial else "complete",
        partial=partial,
        target_direction=target_direction,
        unit=unit,
        primary_metric_id=primary_metric_id,
        metric_family=metric_family,
        target_value=target_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        zero_point_definition=zero_point_definition,
        energy_function=energy_function,
        constraint_predicate=constraint_predicate,
        conflict_group_id=conflict_group_id,
        conflicts_with_goal_ids=tuple(conflicts_with_goal_ids),
        compatible_with_goal_ids=tuple(compatible_with_goal_ids),
        linked_world_ids=tuple(linked_world_ids),
        linked_policy_ids=tuple(linked_policy_ids),
        source_artifacts=tuple(source_artifacts),
        limitations=tuple(limitations),
        metric_contract=dict(metric_contract or {}),
        representation_payload=dict(representation_payload or {}),
        target_hash=target_hash,
        conflict_validation_status=conflict_status,
        **statuses,
    )


def collect_e01_goals(previous: Path, world_inventory: pd.DataFrame) -> list[GoalRepresentation]:
    table_dir = previous / "E01" / "tables"
    report_dir = previous / "E01" / "reports"
    all_e01 = world_ids(world_inventory, experiment="E01")
    all_increasing = world_ids_matching(world_inventory, "E01", lambda row: not str(row["world_family"]).startswith("opposite_direction"))
    opposite = world_ids_matching(world_inventory, "E01", lambda row: str(row["world_family"]).startswith("opposite_direction"))
    chimera = world_ids_matching(world_inventory, "E01", lambda row: "chimera" in str(row["world_family"]))
    baseline_or_frozen = world_ids_matching(world_inventory, "E01", lambda row: str(row["world_family"]) in {"unperturbed_baseline", "frozen_robustness"})
    condition_matrix = table_dir / "e01_condition_matrix.csv"
    records = [
        goal_record(
            source_experiment_id="E01",
            source_step_id="S04-S12",
            source_goal_id="e01:increasing_order",
            display_name="Increasing sortedness target",
            goal_family="monotonic_order",
            goal_kind="order_constraint",
            abstraction_kind="scalar_metric_and_constraint",
            target_structure="1D array values monotonically nondecreasing from left to right",
            target_direction="maximize",
            unit="percent sortedness",
            primary_metric_id="sortedness_percent",
            metric_family="order_quality",
            target_value="100",
            lower_bound=0.0,
            upper_bound=100.0,
            zero_point_definition="0 monotonicity error means every adjacent pair satisfies increasing order.",
            energy_function="100 - sortedness_percent; monotonicity_error_count as count-space proxy.",
            constraint_predicate="for all adjacent positions i < j=i+1, value[i] <= value[j]",
            conflict_group_id="array_direction_order",
            conflicts_with_goal_ids=("e01:decreasing_order", "e06:decreasing"),
            compatible_with_goal_ids=("e06:increasing", "e06:low_focus_increasing", "e06:high_focus_increasing"),
            linked_world_ids=all_increasing + opposite,
            source_artifacts=(source_artifact(condition_matrix), source_artifact(report_dir / "e01_report_bundle_handoff.md")),
            metric_contract={"paper_metric": "Sortedness", "target_direction": "higher_is_better", "unit": "percent"},
        ),
        goal_record(
            source_experiment_id="E01",
            source_step_id="S12",
            source_goal_id="e01:decreasing_order",
            display_name="Decreasing sortedness target",
            goal_family="monotonic_order",
            goal_kind="opposite_order_constraint",
            abstraction_kind="scalar_metric_and_constraint",
            target_structure="1D array values monotonically nonincreasing from left to right",
            target_direction="maximize",
            unit="percent decreasing sortedness",
            primary_metric_id="sortedness_decreasing_percent",
            metric_family="order_quality",
            target_value="100",
            lower_bound=0.0,
            upper_bound=100.0,
            zero_point_definition="0 decreasing monotonicity error means every adjacent pair satisfies decreasing order.",
            energy_function="100 - sortedness_decreasing_percent",
            constraint_predicate="for all adjacent positions i < j=i+1, value[i] >= value[j]",
            conflict_group_id="array_direction_order",
            conflicts_with_goal_ids=("e01:increasing_order", "e06:increasing"),
            compatible_with_goal_ids=("e06:decreasing",),
            linked_world_ids=opposite,
            source_artifacts=(source_artifact(table_dir / "e01_conflict_equilibria_summary.csv"), source_artifact(condition_matrix)),
            metric_contract={"paper_metric": "opposite-direction sortedness", "target_direction": "higher_is_better", "unit": "percent"},
        ),
        goal_record(
            source_experiment_id="E01",
            source_step_id="S09-S12",
            source_goal_id="e01:algotype_aggregation",
            display_name="Same-Algotype aggregation proxy",
            goal_family="aggregation",
            goal_kind="emergent_metric_proxy",
            abstraction_kind="spatial_neighbor_metric",
            target_structure="Adjacent same-label contacts in 1D chimeric arrays",
            target_direction="maximize",
            unit="percent same-label left-neighbor contacts",
            primary_metric_id="aggregation_left_neighbor_primary",
            metric_family="spatial_label_clustering",
            target_value="higher than random-label expectation",
            lower_bound=0.0,
            upper_bound=100.0,
            zero_point_definition="Random-label expectation is computed from configured Algotype counts, not literal zero.",
            energy_function="negative aggregation_left_neighbor_primary for maximization contexts",
            constraint_predicate="same-label adjacent contacts exceed matched random-label baseline",
            conflict_group_id="aggregation_integration",
            conflicts_with_goal_ids=("e06:integration_low_aggregation",),
            linked_world_ids=chimera,
            source_artifacts=(source_artifact(table_dir / "e01_aggregation_peak_table.csv"), source_artifact(table_dir / "e01_duplicate_aggregation_summary.csv")),
            partial=True,
            limitations=("Aggregation is an observed spatial proxy, not a declared objective inside the original local sorting policies.",),
            metric_contract={"target_direction": "higher_is_better", "unit": "percent", "baseline": "random-label expectation"},
        ),
        goal_record(
            source_experiment_id="E01",
            source_step_id="S05-S08",
            source_goal_id="e01:minimize_sorting_work",
            display_name="Sorting work minimization proxy",
            goal_family="efficiency",
            goal_kind="secondary_metric_proxy",
            abstraction_kind="trajectory_cost_metric",
            target_structure="Swap and comparison work during sorting trajectories",
            target_direction="minimize",
            unit="event count",
            primary_metric_id="swap_steps;compare_plus_swap_steps",
            metric_family="work_cost",
            target_value="as low as possible subject to target sortedness",
            lower_bound=0.0,
            zero_point_definition="0 work would mean no swaps/comparisons were needed.",
            energy_function="swap_steps + comparison_steps",
            constraint_predicate="minimize work after maintaining final sortedness target",
            linked_world_ids=all_e01,
            source_artifacts=(source_artifact(table_dir / "e01_efficiency_numeric_table.csv"), source_artifact(condition_matrix)),
            partial=True,
            limitations=("Work count is an evaluation metric and not necessarily optimized by each local policy.",),
        ),
        goal_record(
            source_experiment_id="E01",
            source_step_id="S08",
            source_goal_id="e01:delayed_gratification_proxy",
            display_name="Delayed Gratification trajectory proxy",
            goal_family="trajectory_competency",
            goal_kind="post_hoc_metric_proxy",
            abstraction_kind="trajectory_area_metric",
            target_structure="Temporary sortedness loss followed by recovery above the local baseline",
            target_direction="maximize",
            unit="normalized sortedness-drop/recovery ratio",
            primary_metric_id="dg_primary",
            metric_family="delayed_gratification",
            target_value="higher positive DG ratio",
            lower_bound=0.0,
            zero_point_definition="0 means no measured transient sacrifice and recovery under the DG definition.",
            energy_function="post-hoc integral of recoveries after sortedness drops",
            constraint_predicate="trajectory contains recoverable local decreases in sortedness",
            linked_world_ids=baseline_or_frozen,
            source_artifacts=(source_artifact(table_dir / "e01_dg_numeric_table.csv"), source_artifact(condition_matrix)),
            partial=True,
            limitations=("DG is a trajectory competency proxy, not a literal encoded reward function or evidence of planning.",),
        ),
    ]
    return records


def collect_e04_goals(previous: Path, world_inventory: pd.DataFrame) -> list[GoalRepresentation]:
    e04_worlds = world_ids(world_inventory, experiment="E04")
    base = previous / "E04"
    config = base / "configs" / "e04_s05_homeostatic_tasks.json"
    report = base / "reports" / "e04_homeostatic_task_spec.md"
    table_dir = base / "tables"
    config_data = read_json(config)
    tasks = tuple(config_data.get("tasks", []))
    return [
        goal_record(
            source_experiment_id="E04",
            source_step_id="S05",
            source_goal_id="e04:homeostatic_sortedness_maintenance",
            display_name="Homeostatic sortedness maintenance",
            goal_family="homeostasis",
            goal_kind="maintenance_target",
            abstraction_kind="threshold_occupancy_metric",
            target_structure="Maintain or regain increasing sortedness during scheduled perturbations",
            target_direction="maintain_at_or_above",
            unit="fraction of logged event states",
            primary_metric_id="time_in_target_fraction",
            metric_family="homeostatic_time_in_target",
            target_value="sortedness_percent >= target_sortedness_percent",
            lower_bound=0.0,
            upper_bound=1.0,
            zero_point_definition="0 means no logged event states were at or above the target sortedness threshold.",
            energy_function="1 - time_in_target_fraction with final_sortedness_percent as endpoint check",
            constraint_predicate="sortedness_percent >= configured target_sortedness_percent after perturbations",
            compatible_with_goal_ids=("e01:increasing_order", "e06:increasing"),
            linked_world_ids=e04_worlds,
            source_artifacts=(source_artifact(config), source_artifact(report), source_artifact(table_dir / "e04_homeostatic_baselines.csv")),
            metric_contract={"target_sortedness_percent": 100.0, "tasks": tasks, "unit": "fraction"},
        ),
        goal_record(
            source_experiment_id="E04",
            source_step_id="S05",
            source_goal_id="e04:perturbation_recovery",
            display_name="Perturbation recovery",
            goal_family="repair",
            goal_kind="recovery_target",
            abstraction_kind="event_delay_and_recovery_metric",
            target_structure="Recover from swap, turnover, frozen, and mixed perturbations",
            target_direction="mixed_profile",
            unit="recovered perturbation count and event delay",
            primary_metric_id="recovered_perturbation_count;mean_recovery_events",
            metric_family="repair_recovery",
            target_value="maximize recovered perturbations and minimize recovery delay",
            lower_bound=0.0,
            zero_point_definition="0 recovered perturbations indicates no logged recovery under the benchmark definition.",
            energy_function="-recovered_perturbation_count + mean_recovery_events",
            constraint_predicate="return to in-target sortedness after each perturbation when possible",
            linked_world_ids=e04_worlds,
            source_artifacts=(source_artifact(config), source_artifact(report), source_artifact(table_dir / "e04_homeostatic_baselines_summary.csv")),
            metric_contract={"component_directions": {"recovered_perturbation_count": "maximize", "mean_recovery_events": "minimize"}},
        ),
        goal_record(
            source_experiment_id="E04",
            source_step_id="S04-S05",
            source_goal_id="e04:minimize_damage_impairment",
            display_name="Minimize fatigue and damage impairment",
            goal_family="repair",
            goal_kind="health_proxy",
            abstraction_kind="damage_count_metric",
            target_structure="Reliability-state transitions and impaired activations under fatigue or damage",
            target_direction="minimize",
            unit="transition or impaired-activation count",
            primary_metric_id="cumulative_damage_transition_count;cumulative_impairment_count",
            metric_family="damage_burden",
            target_value="0",
            lower_bound=0.0,
            zero_point_definition="0 means no fatigue/damage transition or impaired activation events.",
            energy_function="cumulative_damage_transition_count + cumulative_impairment_count",
            constraint_predicate="avoid entering damaged or impaired local reliability states",
            linked_world_ids=e04_worlds,
            source_artifacts=(source_artifact(config), source_artifact(table_dir / "e04_fatigue_damage.csv"), source_artifact(report)),
            partial=True,
            limitations=("Damage minimization is an offline health proxy layered onto sorting policies, not always a separately optimized objective.",),
        ),
        goal_record(
            source_experiment_id="E04",
            source_step_id="S11-S14",
            source_goal_id="e04:minimize_local_energy",
            display_name="Local policy energy minimization",
            goal_family="efficiency",
            goal_kind="secondary_metric_proxy",
            abstraction_kind="action_cost_metric",
            target_structure="Swap/action counts across memory, repair, and local-only policies",
            target_direction="minimize",
            unit="action count",
            primary_metric_id="energy_total",
            metric_family="work_cost",
            target_value="as low as possible subject to recovery and sortedness",
            lower_bound=0.0,
            zero_point_definition="0 means no policy swap actions were used.",
            energy_function="energy_total",
            constraint_predicate="reduce local action cost without losing recovery target",
            linked_world_ids=e04_worlds,
            source_artifacts=(source_artifact(table_dir / "e04_intelligence_like_competencies_summary.csv"), source_artifact(table_dir / "e04_minimal_mechanism_table.csv")),
            partial=True,
            limitations=("Energy is a competency/cost axis used for comparison, not the sole optimized reward in E04.",),
        ),
    ]


def e05_target_direction(primary_metric: str) -> tuple[str, str, str]:
    metric = str(primary_metric)
    if any(token in metric for token in ("error", "distance")):
        return "minimize", "proxy distance/error", "0 means constructed target state matches the declared target under this metric."
    if "match" in metric or "label" in metric:
        return "match_target", "target-state match predicate", "Constructed target has zero target_error and exact intended identity labels."
    return "match_target", "target-state predicate", "Constructed target satisfies the target-state predicate."


def collect_e05_goals(previous: Path, world_inventory: pd.DataFrame, known_policy_ids: set[str]) -> list[GoalRepresentation]:
    base = previous / "E05"
    tables = base / "tables"
    reports = base / "reports"
    target_index = pd.read_csv(tables / "e05_target_morphology_index.csv")
    benchmark_tasks = pd.read_csv(tables / "e05_benchmark_task_catalog.csv")
    benchmark_metrics = pd.read_csv(tables / "e05_benchmark_metric_catalog.csv")
    metric_direction = {
        str(row["metric_id"]): ("minimize" if bool(row["lower_is_better"]) else "maximize")
        for row in benchmark_metrics.to_dict(orient="records")
    }
    metric_units = {
        str(row["metric_id"]): ("proxy error" if bool(row["lower_is_better"]) else "percent or normalized score")
        for row in benchmark_metrics.to_dict(orient="records")
    }

    records: list[GoalRepresentation] = []
    target_source = (source_artifact(tables / "e05_target_morphology_index.csv"), source_artifact(base / "results" / "e05_target_gallery_specs.json"), source_artifact(reports / "e05_target_morphology_spec.md"))
    for row in target_index.to_dict(orient="records"):
        target_id = str(row["target_id"])
        direction, unit, zero_definition = e05_target_direction(str(row["primary_metric"]))
        linked = list(e05_target_worlds(world_inventory, target_id))
        linked.extend(world_ids(world_inventory, experiment="E05", task_ids=benchmark_tasks[benchmark_tasks["target_id"].astype(str) == target_id]["benchmark_task_id"].astype(str).tolist()))
        records.append(
            goal_record(
                source_experiment_id="E05",
                source_step_id="S03",
                source_goal_id=f"e05:target:{target_id}",
                display_name=str(row["title"]),
                goal_family=f"target_morphology_{row['target_kind']}",
                goal_kind="target_state",
                abstraction_kind="site_identity_target",
                target_structure=f"{row['target_kind']} target over {row['substrate_kind']} with {row['site_count']} sites",
                target_direction=direction,
                unit=unit,
                primary_metric_id=str(row["primary_metric"]),
                metric_family=str(row["target_kind"]),
                target_value="exact declared target morphology",
                lower_bound=0.0 if direction == "minimize" else None,
                zero_point_definition=zero_definition,
                energy_function=f"target_error for {target_id}; S05 metrics provide additional distances",
                constraint_predicate="site identities match target specification and compatible components",
                linked_world_ids=sorted_unique(linked),
                source_artifacts=target_source,
                metric_contract={
                    "target_id": target_id,
                    "target_hash": row["target_hash"],
                    "primary_metric": row["primary_metric"],
                    "compatible_components_json": row["compatible_components_json"],
                    "constructed_target_error": row["constructed_target_error"],
                },
                representation_payload={"schema_id": row["schema_id"], "render_mode": row["render_mode"]},
                target_hash=str(row["target_hash"]),
            )
        )

    task_source = (source_artifact(tables / "e05_benchmark_task_catalog.csv"), source_artifact(tables / "e05_benchmark_metric_catalog.csv"), source_artifact(reports / "e05_morphology_benchmark_suite.md"))
    for row in benchmark_tasks.to_dict(orient="records"):
        metric_ids = [str(item) for item in parse_json_list(row["metric_ids_json"])]
        policy_ids = [policy_id for policy_id in parse_json_list(row["policy_set_json"]) if str(policy_id) in known_policy_ids]
        primary_metric = str(row["primary_metric"])
        direction = metric_direction.get(primary_metric, "minimize" if "error" in primary_metric else "maximize")
        unit = metric_units.get(primary_metric, "proxy task metric")
        partial = bool(row.get("known_blocker")) and str(row.get("known_blocker")) != "nan"
        limitations = []
        if partial:
            limitations.append(str(row["known_blocker"]))
        limitations.append(str(row["claim_boundary"]))
        records.append(
            goal_record(
                source_experiment_id="E05",
                source_step_id="S15",
                source_goal_id=f"e05:benchmark:{row['benchmark_task_id']}",
                display_name=str(row["title"]),
                goal_family=f"benchmark_{row['benchmark_family']}",
                goal_kind="benchmark_task_target",
                abstraction_kind="target_metric_bundle",
                target_structure=f"{row['task_type']} on {row['substrate_kind']} target {row['target_id']} after {row['perturbation_type']}",
                target_direction=direction,
                unit=unit,
                primary_metric_id=primary_metric,
                metric_family=str(row["benchmark_family"]),
                target_value="best attainable reference target recovery under declared controls",
                lower_bound=0.0 if direction == "minimize" else None,
                upper_bound=100.0 if primary_metric == "final_sortedness_percent" else None,
                zero_point_definition="Lower-bound zero for error metrics; 100 percent for sortedness continuity tasks.",
                energy_function=f"{primary_metric} plus metric bundle {metric_ids}",
                constraint_predicate=f"task target_id={row['target_id']} with perturbation_type={row['perturbation_type']}",
                linked_world_ids=e05_task_worlds(world_inventory, str(row["benchmark_task_id"])),
                linked_policy_ids=policy_ids,
                source_artifacts=task_source,
                partial=partial,
                limitations=tuple(limitations),
                metric_contract={"metric_ids": metric_ids, "primary_metric": primary_metric, "smoke_test": row["smoke_test"]},
                representation_payload={"target_hash": row["target_hash"], "task_type": row["task_type"], "reference_rows": row["reference_rows"]},
                target_hash=str(row["target_hash"]),
            )
        )

    substrate_worlds = world_ids(world_inventory, experiment="E05", families=("higher_dimensional_substrate",))
    records.append(
        goal_record(
            source_experiment_id="E05",
            source_step_id="S01-S04",
            source_goal_id="e05:substrate_mechanics_validation",
            display_name="Higher-dimensional substrate mechanics validation",
            goal_family="substrate_affordance",
            goal_kind="validation_constraint",
            abstraction_kind="partial_validation_goal",
            target_structure="Neighbor, occupancy, and action invariants before assigning a biological-style target",
            target_direction="constraint_set",
            unit="boolean validation pass",
            primary_metric_id="substrate_action_invariant_checks",
            metric_family="substrate_validation",
            target_value="all declared invariant checks pass",
            zero_point_definition="Failure count is zero when substrate/action invariants pass.",
            constraint_predicate="substrate graph and local action invariants are valid",
            linked_world_ids=substrate_worlds,
            source_artifacts=(source_artifact(reports / "e05_substrate_spec.md"), source_artifact(reports / "e05_action_set_spec.md"), source_artifact(tables / "e05_substrate_validation.csv")),
            partial=True,
            limitations=("These E05 S01-S04 rows validate substrate mechanics and actions; target morphologies are supplied by S03/S15 rows, not by the substrate rows themselves.",),
        )
    )
    return records


def collect_e06_goals(previous: Path, world_inventory: pd.DataFrame, known_policy_ids: set[str]) -> list[GoalRepresentation]:
    base = previous / "E06"
    config_path = base / "configs" / "e06_s04_goal_compatibility_config.json"
    condition_path = base / "research_steps" / "S04" / "e06_s04_condition_matrix.csv"
    selfish_path = base / "research_steps" / "S11" / "e06_s11_selfish_objectives.csv"
    metric_spec = base / "reports" / "e06_compatibility_metric_spec.md"
    config = read_json(config_path)
    condition_df = pd.read_csv(condition_path)
    all_e06 = world_ids(world_inventory, experiment="E06")
    s04_worlds = e06_step_worlds(world_inventory, ("S04",))
    policy_ids = sorted_unique(
        [
            policy_id
            for row in condition_df.to_dict(orient="records")
            for policy_id in parse_json_list(row["policy_ids_json"])
            if str(policy_id) in known_policy_ids
        ]
    )

    descriptions: dict[str, list[str]] = {}
    for profile in config.get("goalProfiles", []):
        for name in profile.get("relevantGoalNames", []):
            descriptions.setdefault(str(name), []).append(str(profile.get("description", "")))

    goal_specs = {
        "increasing": ("maximize", "normalized sortedness score", "monotonic_order", "1D increasing order", False),
        "decreasing": ("maximize", "normalized decreasing-sortedness score", "monotonic_order", "1D decreasing order", False),
        "high_half_first": ("maximize", "normalized assigned-goal satisfaction", "partial_order_profile", "High-value half prioritized before lower values", True),
        "low_focus_increasing": ("maximize", "normalized assigned-goal satisfaction", "local_focus_profile", "Increasing order with low-value-region emphasis", True),
        "high_focus_increasing": ("maximize", "normalized assigned-goal satisfaction", "local_focus_profile", "Increasing order with high-value-region emphasis", True),
        "parity_even_first": ("maximize", "normalized assigned-goal satisfaction", "permutation_profile", "Even values before odd values", True),
        "center_out": ("maximize", "normalized assigned-goal satisfaction", "permutation_profile", "Center-out ordering profile", True),
    }
    records: list[GoalRepresentation] = []
    for name, (direction, unit, family, target_structure, partial) in goal_specs.items():
        conflicts: tuple[str, ...] = ()
        compatibles: tuple[str, ...] = ()
        conflict_group = "e06_order_profiles"
        if name == "increasing":
            conflicts = ("e01:decreasing_order", "e06:decreasing")
            compatibles = ("e01:increasing_order", "e06:low_focus_increasing", "e06:high_focus_increasing")
        elif name == "decreasing":
            conflicts = ("e01:increasing_order", "e06:increasing")
        elif name in {"low_focus_increasing", "high_focus_increasing"}:
            compatibles = ("e06:increasing",)
        elif name in {"parity_even_first", "center_out"}:
            conflicts = ("e06:increasing",)
        limitations = ()
        if partial:
            limitations = (f"E06 names `{name}` in compatibility profiles, but no standalone formal energy function was mounted; represented from profile labels and observed assigned-goal scores.",)
        records.append(
            goal_record(
                source_experiment_id="E06",
                source_step_id="S04",
                source_goal_id=f"e06:{name}",
                display_name=f"E06 {name.replace('_', ' ')} goal",
                goal_family=family,
                goal_kind="assigned_policy_goal",
                abstraction_kind="array_order_profile",
                target_structure=target_structure,
                target_direction=direction,
                unit=unit,
                primary_metric_id="mean_assigned_policy_sortedness",
                metric_family="assigned_goal_satisfaction",
                target_value="1.0",
                lower_bound=0.0,
                upper_bound=1.0,
                zero_point_definition="0 means no assigned-goal satisfaction under E06 proxy scoring.",
                energy_function="1 - mean_assigned_policy_sortedness",
                constraint_predicate="policy-assigned local order profile is satisfied",
                conflict_group_id=conflict_group,
                conflicts_with_goal_ids=conflicts,
                compatible_with_goal_ids=compatibles,
                linked_world_ids=s04_worlds,
                linked_policy_ids=policy_ids,
                source_artifacts=(source_artifact(config_path), source_artifact(condition_path), source_artifact(base / "tables" / "e06_goal_compatibility_summary.csv")),
                partial=partial,
                limitations=limitations,
                metric_contract={"profile_descriptions": descriptions.get(name, []), "array_size": config.get("arraySize"), "event_cap": config.get("eventCap")},
            )
        )

    records.extend(
        [
            goal_record(
                source_experiment_id="E06",
                source_step_id="S05-S15",
                source_goal_id="e06:goal_conflict_index",
                display_name="Goal conflict index minimization",
                goal_family="compatibility",
                goal_kind="conflict_metric",
                abstraction_kind="compatibility_dimension",
                target_structure="Gap between relevant assigned goals in mixed Algotype arrays",
                target_direction="minimize",
                unit="normalized goal-alignment gap",
                primary_metric_id="goal_conflict_index",
                metric_family="goal_compatibility",
                target_value="0",
                lower_bound=0.0,
                upper_bound=1.0,
                zero_point_definition="0 means no final-state disagreement among relevant goals for that row.",
                energy_function="goal_conflict_index",
                constraint_predicate="mixed policies do not create relevant-goal disagreement",
                conflict_group_id="e06_goal_profile_conflict",
                conflicts_with_goal_ids=("e06:decreasing", "e06:parity_even_first", "e06:center_out"),
                linked_world_ids=all_e06,
                source_artifacts=(source_artifact(metric_spec), source_artifact(base / "tables" / "e06_compatibility_dimension_summary.csv")),
                metric_contract={"formula_source": "e06_compatibility_metric_spec.md", "direction": "lower_is_better"},
            ),
            goal_record(
                source_experiment_id="E06",
                source_step_id="S05-S15",
                source_goal_id="e06:integration_low_aggregation",
                display_name="Low extra aggregation / integration",
                goal_family="anti_aggregation",
                goal_kind="compatibility_metric",
                abstraction_kind="label_mixing_metric",
                target_structure="Chimeric arrays avoid excess same-label aggregation beyond random expectation",
                target_direction="maximize",
                unit="normalized integration score",
                primary_metric_id="integration_score",
                metric_family="label_integration",
                target_value="1.0",
                lower_bound=0.0,
                upper_bound=1.0,
                zero_point_definition="0 indicates maximal positive aggregation delta after clamping.",
                energy_function="1 - integration_score",
                constraint_predicate="aggregation_delta_percent is not positive after random-expectation correction",
                conflict_group_id="aggregation_integration",
                conflicts_with_goal_ids=("e01:algotype_aggregation", "e06:clone_cohesion"),
                linked_world_ids=all_e06,
                source_artifacts=(source_artifact(metric_spec), source_artifact(base / "tables" / "e06_compatibility_dimension_summary.csv")),
                metric_contract={"formula": "1 - clamp(max(aggregation delta percent, 0) / 100, 0, 1)"},
            ),
            goal_record(
                source_experiment_id="E06",
                source_step_id="S06-S15",
                source_goal_id="e06:dominance_margin_proxy",
                display_name="Dominance margin proxy",
                goal_family="dominance",
                goal_kind="analysis_metric_proxy",
                abstraction_kind="spatial_bias_metric",
                target_structure="Absolute positional dominance margin by one Algotype",
                target_direction="mixed_profile",
                unit="absolute normalized margin",
                primary_metric_id="dominance_abs_margin",
                metric_family="dominance",
                target_value="context-dependent: minimize for integration, maximize for dominance-seeker analysis",
                lower_bound=0.0,
                upper_bound=1.0,
                zero_point_definition="0 means no measured positional dominance margin.",
                energy_function="dominance_abs_margin or -dominance_abs_margin depending on analysis question",
                constraint_predicate="dominance direction is analysis-dependent",
                linked_world_ids=all_e06,
                source_artifacts=(source_artifact(metric_spec), source_artifact(base / "tables" / "e06_dominance_hierarchy.csv")),
                partial=True,
                limitations=("Dominance is a compatibility-analysis dimension, not a single universal target direction across all E06 contexts.",),
            ),
        ]
    )

    selfish_df = pd.read_csv(selfish_path)
    s11_worlds = e06_step_worlds(world_inventory, ("S11",))
    for row in selfish_df.to_dict(orient="records"):
        records.append(
            goal_record(
                source_experiment_id="E06",
                source_step_id="S11",
                source_goal_id=f"e06:{row['objective_id']}",
                display_name=str(row["objective_id"]).replace("_", " ").title(),
                goal_family=str(row["objective_family"]),
                goal_kind="selfish_clone_objective",
                abstraction_kind="mutant_actor_score",
                target_structure=str(row["definition"]),
                target_direction="maximize",
                unit=f"normalized {row['score_name']}",
                primary_metric_id=str(row["score_name"]),
                metric_family=str(row["objective_family"]),
                target_value="1.0",
                lower_bound=0.0,
                upper_bound=1.0,
                zero_point_definition="0 means no selfish-objective score under the declared E06 proxy.",
                energy_function=f"1 - {row['score_name']}",
                constraint_predicate=str(row["action_rule"]),
                conflict_group_id="host_selfish_conflict",
                conflicts_with_goal_ids=("e06:increasing", "e04:homeostatic_sortedness_maintenance"),
                linked_world_ids=s11_worlds,
                source_artifacts=(source_artifact(selfish_path), source_artifact(base / "configs" / "e06_s11_mutant_clone_config.json"), source_artifact(base / "reports" / "e06_s11_selfish_objective_spec.md")),
                limitations=(str(row["biological_analogy_caveat"]),),
                metric_contract={"allows_replication": bool(row["allows_replication"]), "gate_goal_proxy": row["gate_goal_proxy"]},
            )
        )

    return records


def collect_s01_derived_goals(world_inventory: pd.DataFrame, artifacts_dir: Path) -> list[GoalRepresentation]:
    s01_report = artifacts_dir / "research_steps" / "S01" / "research_step_full_results.md"
    s01_inventory = artifacts_dir / "tables" / "e07_world_inventory.csv"
    return [
        goal_record(
            source_experiment_id="S01_derived",
            source_step_id="S01",
            source_goal_id="s01derived:e02_audit_goal_bundle",
            display_name="E02 audit goal bundle",
            goal_family="audit_objective_bundle",
            goal_kind="partial_goal_bundle",
            abstraction_kind="world_inventory_partial",
            target_structure="E02 audits standard increasing sorting plus null, scheduler, robustness, aggregation, and metric-substitution objectives",
            target_direction="mixed_profile",
            unit="heterogeneous audit metric bundle",
            primary_metric_id="varies_by_e02_audit_step",
            metric_family="replication_audit",
            target_value="audit-specific pass/fail or effect-size target",
            zero_point_definition="Defined separately by each E02 audit table; not mounted as one goal contract for S03.",
            energy_function="not a single energy function",
            constraint_predicate="standard increasing target plus audit-specific control condition",
            linked_world_ids=world_ids(world_inventory, experiment="E02"),
            source_artifacts=(source_artifact(s01_inventory), source_artifact(s01_report)),
            partial=True,
            limitations=("S02/E02 audit worlds in S01 lack a single upstream goal contract; this row keeps them linked for S04 without inventing an energy function.",),
        ),
        goal_record(
            source_experiment_id="S01_derived",
            source_step_id="S01",
            source_goal_id="s01derived:e03_competence_vector_bundle",
            display_name="E03 morphospace competence-vector bundle",
            goal_family="competence_vector",
            goal_kind="partial_goal_bundle",
            abstraction_kind="world_inventory_partial",
            target_structure="E03 policy morphospace combines increasing sortedness, robustness, aggregation, DG, frontier, and embedding objectives",
            target_direction="mixed_profile",
            unit="heterogeneous competence-vector metrics",
            primary_metric_id="e03_competence_vector",
            metric_family="policy_morphospace_competence",
            target_value="multi-axis quality and diversity",
            zero_point_definition="Defined per E03 competence metric, not as one scalar target.",
            energy_function="not a single energy function",
            constraint_predicate="policy quality/diversity over E03 morphospace axes",
            linked_world_ids=world_ids(world_inventory, experiment="E03"),
            source_artifacts=(source_artifact(s01_inventory), source_artifact(s01_report)),
            partial=True,
            limitations=("E03 is represented from S01 world metadata in this S03 pass; E03-specific competence-vector internals are deferred to S04 corpus integration.",),
        ),
    ]


def collect_goal_records(args: argparse.Namespace, world_inventory: pd.DataFrame, policy_table: pd.DataFrame) -> list[GoalRepresentation]:
    known_policy_ids = set(policy_table["source_policy_id"].astype(str)) | set(policy_table["policy_uid"].astype(str)) | set(policy_table["canonical_policy_id"].astype(str))
    previous = args.previous_artifacts_dir
    records: list[GoalRepresentation] = []
    records.extend(collect_e01_goals(previous, world_inventory))
    records.extend(collect_e04_goals(previous, world_inventory))
    records.extend(collect_e05_goals(previous, world_inventory, known_policy_ids))
    records.extend(collect_e06_goals(previous, world_inventory, known_policy_ids))
    records.extend(collect_s01_derived_goals(world_inventory, args.artifacts_dir))
    return records


def build_world_goal_links(goal_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in goal_table.to_dict(orient="records"):
        for world_id in parse_json_list(row["linked_world_ids_json"]):
            rows.append(
                {
                    "schema_version": "eidosoma.e07.world_goal_link.v1",
                    "world_id": str(world_id),
                    "goal_uid": row["goal_uid"],
                    "canonical_goal_id": row["canonical_goal_id"],
                    "source_goal_id": row["source_goal_id"],
                    "source_experiment_id": row["source_experiment_id"],
                    "goal_family": row["goal_family"],
                    "link_role": "partial_context" if bool(row["partial"]) else "primary_or_secondary",
                    "target_direction": row["target_direction"],
                    "primary_metric_id": row["primary_metric_id"],
                    "record_hash": stable_hash({"world_id": str(world_id), "source_goal_id": row["source_goal_id"]}),
                }
            )
    return pd.DataFrame(rows).sort_values(["world_id", "source_goal_id"], kind="mergesort").reset_index(drop=True)


def build_goal_conflicts(goal_table: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(row["source_goal_id"]): row for row in goal_table.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for row in goal_table.to_dict(orient="records"):
        for conflict_goal_id in parse_json_list(row["conflicts_with_goal_ids_json"]):
            other = lookup.get(str(conflict_goal_id), {})
            conflict_type = "direct_or_proxy_tension"
            if row["conflict_group_id"] == "array_direction_order":
                conflict_type = "direct_order_opposition"
            elif row["conflict_group_id"] == "aggregation_integration":
                conflict_type = "aggregation_integration_tension"
            elif row["conflict_group_id"] == "host_selfish_conflict":
                conflict_type = "selfish_host_goal_conflict"
            elif row["conflict_group_id"] == "e06_goal_profile_conflict":
                conflict_type = "multi_goal_profile_tension"
            rows.append(
                {
                    "schema_version": "eidosoma.e07.goal_conflict.v1",
                    "conflict_group_id": row["conflict_group_id"],
                    "source_goal_id_a": row["source_goal_id"],
                    "source_goal_id_b": str(conflict_goal_id),
                    "canonical_goal_id_a": row["canonical_goal_id"],
                    "canonical_goal_id_b": other.get("canonical_goal_id", ""),
                    "conflict_type": conflict_type,
                    "source_experiment_id": row["source_experiment_id"],
                    "validation_basis": row["conflict_validation_status"],
                    "record_hash": stable_hash({"a": row["source_goal_id"], "b": str(conflict_goal_id), "group": row["conflict_group_id"]}),
                }
            )
    return pd.DataFrame(rows).sort_values(["conflict_group_id", "source_goal_id_a", "source_goal_id_b"], kind="mergesort").reset_index(drop=True)


def markdown_table(df: pd.DataFrame, columns: list[str], *, max_rows: int = 24) -> str:
    if df.empty:
        return "_No rows._"
    shown = df[columns].head(max_rows).copy()
    for column in shown.columns:
        shown[column] = shown[column].map(lambda value: str(value).replace("|", "\\|"))
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(record[column]) for column in columns) + " |" for record in shown.to_dict(orient="records")]
    if len(df) > max_rows:
        omitted = [f"... {len(df) - max_rows} more rows omitted", *("" for _ in columns[1:])]
        rows.append("| " + " | ".join(omitted) + " |")
    return "\n".join([header, separator, *rows])


def top_summary_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
    lay_summary: str,
) -> str:
    artifact_lines = "\n".join(f"- `{entry['path']}`" for entry in artifacts if entry.get("path"))
    return f"""## Top Summary

- Research step ID: {STEP_ID}
- Completion status: Completed
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {outcome_classification}
- Caveats or blockers: {caveats_or_blockers}
- Lay summary: {lay_summary}
- Recommended next action: {recommended_next_action}

"""


def spec_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome: str,
    caveats: str,
    recommended_next_action: str,
    lay_summary: str,
    goal_table: pd.DataFrame,
    conflict_table: pd.DataFrame,
) -> str:
    source_counts = goal_table.groupby(["source_experiment_id"], dropna=False).size().reset_index(name="record_count")
    family_counts = goal_table.groupby(["goal_family"], dropna=False).size().reset_index(name="record_count").sort_values("record_count", ascending=False)
    return (
        top_summary_markdown(artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary)
        + f"""
# E07 S03 Goal Representation Specification

## Frozen Question

Can goals such as monotonic order, target morphology, aggregation, anti-aggregation, regeneration, symmetry, and homeostasis be encoded as constraints or energy functions?

## Representation Contract

Each row records one target predicate, metric objective, multi-goal profile, or explicit partial proxy. The required fields include:

- `goal_uid`: source-row-stable ID.
- `canonical_goal_id`: hash-based ID for equivalent abstract target contracts.
- Source provenance: upstream experiment, step, source goal ID, and source artifact paths.
- Target contract: family, kind, abstraction, target direction, unit, metric, target value, constraints, and energy/proxy expression.
- Crosswalks: S01 `world_id` links and optional S02 policy IDs.
- Conflict fields: `conflict_group_id`, `conflicts_with_goal_ids_json`, and the separate conflict table.

## Coverage By Source

{markdown_table(source_counts, ["source_experiment_id", "record_count"])}

## Coverage By Goal Family

{markdown_table(family_counts, ["goal_family", "record_count"], max_rows=30)}

## Conflict Encodings

Conflict table rows written: {len(conflict_table)}.

{markdown_table(conflict_table, ["conflict_group_id", "source_goal_id_a", "source_goal_id_b", "conflict_type"], max_rows=20)}

## Partial Record Rule

Rows marked `partial=true` are intentionally not treated as full energy functions. They are metadata-preserving goal or metric proxies for ambiguous E01 aggregation/DG axes, E04 secondary health or energy axes, E05 blocker tasks, E06 partial order profiles, and S01-derived E02/E03 coverage rows.
"""
    )


def full_results_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome: str,
    caveats: str,
    recommended_next_action: str,
    lay_summary: str,
    goal_table: pd.DataFrame,
    link_table: pd.DataFrame,
    conflict_table: pd.DataFrame,
    validation_df: pd.DataFrame,
    unit_test_result: dict[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    source_counts = goal_table.groupby(["source_experiment_id"], dropna=False).size().reset_index(name="record_count")
    direction_counts = goal_table.groupby(["target_direction"], dropna=False).size().reset_index(name="record_count")
    partial_counts = goal_table.groupby(["source_experiment_id", "partial"], dropna=False).size().reset_index(name="record_count")
    validation_cols = ["validation_case", "success", "detail"]
    return (
        top_summary_markdown(artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary)
        + f"""
# E07 S03 Full Results: Represent Goals Abstractly

## Lay Summary

{lay_summary}

## Frozen Question

Can goals such as monotonic order, target morphology, aggregation, anti-aggregation, regeneration, symmetry, and homeostasis be encoded as constraints or energy functions?

## Inputs

- S01 world inventory: `{manifest['inputs']['s01WorldInventory']}`.
- S02 policy table: `{manifest['inputs']['s02PolicyTable']}`.
- E01 condition and metric tables under `/previous-artifacts/E01/tables/`.
- E04 homeostatic task config/report/tables under `/previous-artifacts/E04/`.
- E05 target morphology, metric, and benchmark catalogs under `/previous-artifacts/E05/`.
- E06 goal compatibility, compatibility metric, and selfish-objective artifacts under `/previous-artifacts/E06/`.

## Methods

S03 normalized upstream goal and metric specifications into a row-stable goal table. Explicit target morphologies and benchmark tasks were represented as target predicates or metric bundles. Metrics with declared lower-is-better flags became `minimize`; sortedness and satisfaction scores became `maximize`; exact target-state catalogs became `match_target`; homeostatic thresholds became `maintain_at_or_above`; multi-axis or ambiguous proxies became `mixed_profile` or explicit partial rows. Every S01 world received at least one goal link. E05/E06 policy references were checked against the S02 policy table where policy IDs were available.

## Commands

- `python -m unittest discover -s tests/e07 -p 'test_*.py'`
- `python scripts/e07_s03_goal_representations.py --artifacts-dir /artifacts --previous-artifacts-dir /previous-artifacts --s01-world-inventory /artifacts/tables/e07_world_inventory.csv --s02-policy-table /artifacts/tables/e07_policy_representations.parquet --run-unit-tests`

Unit-test command result: return code {unit_test_result['returnCode']}, success={unit_test_result['success']}, elapsed={unit_test_result['elapsedSeconds']} seconds.

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- New dependencies installed: none.
- CPU/GPU use: serial metadata extraction and validation only; no GPU and no parallel workers.

## Results

Goal representation rows written: {len(goal_table)}.
World-goal links written: {len(link_table)}.
Goal conflict rows written: {len(conflict_table)}.

### Records By Source

{markdown_table(source_counts, ["source_experiment_id", "record_count"])}

### Records By Target Direction

{markdown_table(direction_counts, ["target_direction", "record_count"])}

### Partial Rows By Source

{markdown_table(partial_counts, ["source_experiment_id", "partial", "record_count"])}

## Validation

{markdown_table(validation_df, validation_cols, max_rows=20)}

Validation result: {validation_result}

## Artifacts

{markdown_table(pd.DataFrame(artifacts), ["relativePath", "description", "sha256"], max_rows=20)}

## Caveats, Blockers, And Limitations

{caveats}

- S03 is an abstract metadata and proxy contract, not a behavioral re-run or biological validation.
- E01 aggregation and DG rows are post-hoc metric proxies rather than policy-internal reward functions.
- E05 target records are toy computational targets; regeneration-style benchmark rows preserve declared local-control blockers.
- E06 partial ordering profiles are represented from goal-profile labels and compatibility scores because standalone energy functions were not mounted for all profile names.
- S01-derived E02/E03 records keep the world inventory covered but should be refined in S04 when the unified behavior corpus joins exact result tables.

## Provenance

- Git commit before/at run: `{manifest['git']['head']}`
- Git branch: `{manifest['git']['branch']}`
- Generated at UTC: `{manifest['generatedAtUtc']}`
- S01 world inventory SHA-256: `{manifest['inputs']['s01WorldInventorySha256']}`
- S02 policy table SHA-256: `{manifest['inputs']['s02PolicyTableSha256']}`

## Recommended Next Action

{recommended_next_action}
"""
    )


def build_manifest(
    *,
    args: argparse.Namespace,
    goal_table: pd.DataFrame,
    link_table: pd.DataFrame,
    conflict_table: pd.DataFrame,
    validation_df: pd.DataFrame,
    unit_test_result: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "eidosoma.e07.s03.artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "git": {
            "head": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
        "inputs": {
            "s01WorldInventory": str(args.s01_world_inventory),
            "s01WorldInventorySha256": sha256_file(args.s01_world_inventory),
            "s02PolicyTable": str(args.s02_policy_table),
            "s02PolicyTableSha256": sha256_file(args.s02_policy_table),
            "previousArtifactsDir": str(args.previous_artifacts_dir),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "workerCount": 1,
            "gpuUsed": False,
        },
        "results": {
            "goalRecordCount": int(len(goal_table)),
            "worldGoalLinkCount": int(len(link_table)),
            "goalConflictCount": int(len(conflict_table)),
            "partialRecordCount": int(goal_table["partial"].astype(bool).sum()),
            "sourceCounts": goal_table.groupby("source_experiment_id").size().to_dict(),
            "targetDirectionCounts": goal_table.groupby("target_direction").size().to_dict(),
        },
        "validation": {
            "success": bool(validation_df["success"].all()) and bool(unit_test_result["success"]),
            "checks": validation_df.to_dict(orient="records"),
            "unitTests": unit_test_result,
        },
        "artifacts": artifacts,
    }


def main() -> int:
    args = parse_args()
    artifacts_dir: Path = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    tables_dir = artifacts_dir / "tables"
    reports_dir = artifacts_dir / "reports"
    step_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    world_inventory = pd.read_csv(args.s01_world_inventory)
    policy_table = pd.read_parquet(args.s02_policy_table)
    records = collect_goal_records(args, world_inventory, policy_table)
    goal_table = dataframe_from_goal_records(records)
    link_table = build_world_goal_links(goal_table)
    conflict_table = build_goal_conflicts(goal_table)

    known_policy_ids = set(policy_table["source_policy_id"].astype(str)) | set(policy_table["policy_uid"].astype(str)) | set(policy_table["canonical_policy_id"].astype(str))
    validation_df = validate_goal_table(
        goal_table,
        s01_world_ids=world_inventory["world_id"].astype(str).tolist(),
        known_policy_ids=known_policy_ids,
    )

    unit_test_result = {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    if args.run_unit_tests:
        unit_test_result = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e07", "-p", "test_*.py"], args.repo_dir)

    goal_parquet = tables_dir / "e07_goal_representations.parquet"
    goal_csv = tables_dir / "e07_goal_representations.csv"
    link_parquet = tables_dir / "e07_world_goal_links.parquet"
    link_csv = tables_dir / "e07_world_goal_links.csv"
    conflict_parquet = tables_dir / "e07_goal_conflicts.parquet"
    conflict_csv = tables_dir / "e07_goal_conflicts.csv"
    validation_path = step_dir / "e07_s03_validation_checks.csv"
    spec_path = reports_dir / "e07_goal_representation_spec.md"
    manifest_path = step_dir / "artifact_manifest.json"
    full_results_path = step_dir / "research_step_full_results.md"

    goal_table.to_parquet(goal_parquet, index=False)
    goal_table.to_csv(goal_csv, index=False)
    link_table.to_parquet(link_parquet, index=False)
    link_table.to_csv(link_csv, index=False)
    conflict_table.to_parquet(conflict_parquet, index=False)
    conflict_table.to_csv(conflict_csv, index=False)
    validation_df.to_csv(validation_path, index=False)

    validation_success = bool(validation_df["success"].all()) and bool(unit_test_result["success"])
    validation_result = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} goal validation checks passed; "
        f"unit tests {'passed' if unit_test_result['success'] else 'failed'}; "
        f"{len(goal_table)} goal records, {len(link_table)} world-goal links, {len(conflict_table)} conflict rows"
    )
    outcome = "supportive" if validation_success and len(goal_table) > 0 else "null"
    caveats = (
        "S03 is a representation and provenance audit, not behavioral validation. Ambiguous E01/E04/E06 metric axes and S01-only E02/E03 coverage rows are explicit partial records. "
        "E05 regeneration-style tasks retain declared blockers where local swap/wait policies cannot synthesize missing cells."
    )
    lay_summary = (
        "S03 converted the available upstream goals, targets, metric directions, and conflict profiles into one goal table. "
        "Every S01 world now has at least one goal link, and ambiguous goals remain visible as partial records rather than being silently converted into unsupported energy functions."
    )
    recommended_next_action = "Stop before S04; after Chief Scientist review, proceed to S04 unified behavior corpus construction using `canonical_goal_id`, S01 `world_id`, and S02 `canonical_policy_id` links."

    artifacts = [
        artifact_entry(goal_parquet, artifacts_dir, "Required S03 goal representation table"),
        artifact_entry(goal_csv, artifacts_dir, "CSV inspection copy of S03 goal representation table"),
        artifact_entry(link_parquet, artifacts_dir, "S03 S01-world to goal crosswalk"),
        artifact_entry(link_csv, artifacts_dir, "CSV inspection copy of S03 world-goal crosswalk"),
        artifact_entry(conflict_parquet, artifacts_dir, "S03 goal conflict encoding table"),
        artifact_entry(conflict_csv, artifacts_dir, "CSV inspection copy of S03 goal conflict table"),
        artifact_entry(validation_path, artifacts_dir, "S03 validation check table"),
        self_referential_artifact_entry(spec_path, artifacts_dir, "Required S03 goal representation specification report"),
        self_referential_artifact_entry(manifest_path, artifacts_dir, "S03 artifact and provenance manifest"),
        self_referential_artifact_entry(full_results_path, artifacts_dir, "Required S03 full-results handoff report"),
    ]

    write_text(spec_path, spec_markdown(artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary, goal_table, conflict_table))
    manifest = build_manifest(
        args=args,
        goal_table=goal_table,
        link_table=link_table,
        conflict_table=conflict_table,
        validation_df=validation_df,
        unit_test_result=unit_test_result,
        artifacts=artifacts,
    )
    manifest["validation"]["success"] = validation_success
    write_json(manifest_path, manifest)
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts,
            validation_result,
            outcome,
            caveats,
            recommended_next_action,
            lay_summary,
            goal_table,
            link_table,
            conflict_table,
            validation_df,
            unit_test_result,
            manifest,
        ),
    )

    final_artifacts = [
        artifact_entry(goal_parquet, artifacts_dir, "Required S03 goal representation table"),
        artifact_entry(goal_csv, artifacts_dir, "CSV inspection copy of S03 goal representation table"),
        artifact_entry(link_parquet, artifacts_dir, "S03 S01-world to goal crosswalk"),
        artifact_entry(link_csv, artifacts_dir, "CSV inspection copy of S03 world-goal crosswalk"),
        artifact_entry(conflict_parquet, artifacts_dir, "S03 goal conflict encoding table"),
        artifact_entry(conflict_csv, artifacts_dir, "CSV inspection copy of S03 goal conflict table"),
        artifact_entry(validation_path, artifacts_dir, "S03 validation check table"),
        self_referential_artifact_entry(spec_path, artifacts_dir, "Required S03 goal representation specification report"),
        self_referential_artifact_entry(manifest_path, artifacts_dir, "S03 artifact and provenance manifest"),
        self_referential_artifact_entry(full_results_path, artifacts_dir, "Required S03 full-results handoff report"),
    ]
    manifest["artifacts"] = final_artifacts
    write_json(manifest_path, manifest)
    write_text(spec_path, spec_markdown(final_artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary, goal_table, conflict_table))
    write_text(
        full_results_path,
        full_results_markdown(
            final_artifacts,
            validation_result,
            outcome,
            caveats,
            recommended_next_action,
            lay_summary,
            goal_table,
            link_table,
            conflict_table,
            validation_df,
            unit_test_result,
            manifest,
        ),
    )

    print(json.dumps({"success": validation_success, "recordCount": len(goal_table), "validationResult": validation_result}, indent=2))
    return 0 if validation_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
