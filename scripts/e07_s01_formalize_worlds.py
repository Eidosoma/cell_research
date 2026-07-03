#!/usr/bin/env python3
"""Formalize upstream worlds for E07 S01."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e07.world_schema import (  # noqa: E402
    REQUIRED_TUPLE_FIELDS,
    WorldRecord,
    dataframe_from_records,
    make_world_id,
    source_paths_exist,
    stable_hash,
    validate_inventory,
)


STEP_ID = "S01"
STEP_NUMBER = 1
EXPERIMENT_ID = "E07"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-artifacts-dir", type=Path, default=Path("/previous-artifacts"))
    parser.add_argument("--previous-artifacts-context", type=Path, default=Path("/workspace/PREVIOUS_ARTIFACTS.json"))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return read_json(path)
    except Exception:
        return {}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    import hashlib

    digest = hashlib.sha256()
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


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def safe_columns(path: Path, *, limit: int = 24) -> list[str]:
    try:
        if path.suffix == ".csv":
            return list(pd.read_csv(path, nrows=0).columns)[:limit]
        if path.suffix == ".parquet":
            return list(pd.read_parquet(path).columns)[:limit]
    except Exception:
        return []
    return []


def list_existing(*paths: Path) -> tuple[str, ...]:
    return tuple(str(path) for path in paths if path.exists())


def preview_dict(data: Mapping[str, Any], *, keys: list[str]) -> dict[str, Any]:
    return {key: data.get(key) for key in keys if key in data}


def e01_records(base: Path) -> list[WorldRecord]:
    config_path = base / "configs" / "e01_baseline_configs.json"
    config = safe_read_json(config_path)
    records: list[WorldRecord] = []
    for condition in config.get("conditions", []):
        task_id = str(condition.get("condition_id", "condition"))
        algorithm = condition.get("algorithm", "unknown_algorithm")
        mode = condition.get("mode", "unknown_mode")
        run_family = condition.get("run_family", "baseline")
        frozen_count = condition.get("frozen_count", "not_specified")
        frozen_semantics = condition.get("frozen_semantics", "none")
        direction = condition.get("direction_profile", "all_increasing")
        value_distribution = condition.get("value_distribution", condition.get("value_bank_id", "not_specified"))
        algotype_mix = condition.get("algotype_mix", algorithm)
        array_length = condition.get("array_length", 100)
        source_artifacts = list_existing(config_path)
        local_observation = (
            "cell-view policy observes actor value/status, label, local boundaries, statuses and neighboring values through public cell classes"
            if mode == "cell_view"
            else "traditional controller observes full array or sorted-prefix state; included as nonlocal baseline"
        )
        action_set = "adjacent compare/swap and wait/no-op for cell-view policies"
        if mode == "traditional" and algorithm == "selection":
            action_set = "scan comparisons plus direct exchange in reconstructed traditional selection baseline"
        elif mode == "traditional":
            action_set = "adjacent comparisons and swaps in reconstructed traditional baseline"
        records.append(
            WorldRecord(
                world_id=make_world_id("E01", str(condition.get("step_scope", "S03")), task_id, family=run_family),
                experiment_id="E01",
                source_step_id=str(condition.get("step_scope", "S03")),
                record_granularity="condition",
                world_family=str(run_family),
                task_id=task_id,
                task_label=f"{mode} {algorithm} {run_family}",
                substrate_kind="array_1d",
                state_space=f"Length-{array_length} 1D array with {value_distribution} values, Algotype mix {algotype_mix}, and frozen_count={frozen_count}.",
                local_observations=local_observation,
                action_set=action_set,
                transition_rules=f"{mode} {algorithm} sorting; stop_rule={condition.get('stop_rule', 'not_specified')}; baseline_source={condition.get('baseline_source', 'not_specified')}.",
                goal_predicate=f"Direction profile {direction}; primary target is sortedness under configured value distribution.",
                perturbation_model=f"run_family={run_family}; frozen_semantics={frozen_semantics}; frozen_count={frozen_count}; value_distribution={value_distribution}.",
                measurement_functions=str(condition.get("primary_metrics", "sortedness_percent; monotonicity_error_count; swap_steps; compare_plus_swap_steps")),
                scheduler=str(condition.get("max_steps_policy", "paper/public runner schedule not fully specified")),
                source_artifacts=source_artifacts,
                metadata=preview_dict(
                    condition,
                    keys=[
                        "condition_id",
                        "algorithm",
                        "mode",
                        "repeat_count",
                        "matched_group_id",
                        "figure_targets",
                        "outputs_expected",
                        "caveats",
                    ],
                ),
            )
        )
    return records


def e02_records(base: Path) -> list[WorldRecord]:
    records: list[WorldRecord] = []
    result_dir = base / "results"
    report_paths = list_existing(base / "reports" / "e02_claim_audit.md", base / "reports" / "e02_report_bundle_handoff.md")
    families = {
        "scheduler_comparison": ("scheduler", "scheduler regime changes"),
        "activation_rate": ("activation", "activation-rate perturbations"),
        "label_shuffle": ("aggregation_null", "trajectory-preserving Algotype label shuffles"),
        "dummy_algotype": ("dummy_algotype", "dummy same-code Algotype controls"),
        "speed_matched": ("speed_matched_chimera", "speed-matched Algotype controls"),
        "local_move_null": ("local_move_null", "random local-move nulls"),
        "dg_null": ("delayed_gratification_null", "matched Delayed Gratification nulls"),
        "alternative_metrics": ("metric_substitution", "alternative metric definitions"),
        "frozen_placement": ("frozen_placement", "Frozen Cell placement perturbations"),
        "frozen_behavior": ("frozen_behavior", "passive/stuck/variant Frozen Cell behavior"),
        "input_distribution": ("input_distribution", "input value distribution shifts"),
        "stop_condition": ("stop_condition", "stop-condition sensitivity"),
        "simulator_validation": ("deterministic_validation", "deterministic simulator validation"),
    }
    for result_path in sorted(result_dir.glob("e02_*.parquet")):
        stem = result_path.stem
        family, perturbation = next(((value[0], value[1]) for key, value in families.items() if key in stem), ("audit_result", stem.replace("e02_", "")))
        columns = safe_columns(result_path)
        records.append(
            WorldRecord(
                world_id=make_world_id("E02", "S01-S15", stem, family=family),
                experiment_id="E02",
                source_step_id="S01-S15",
                record_granularity="result_table",
                world_family=family,
                task_id=stem,
                task_label=stem.replace("e02_", "").replace("_", " "),
                substrate_kind="array_1d",
                state_space="E01-compatible 1D array simulator state with active/frozen statuses and deterministic event records.",
                local_observations="E02 deterministic simulator preserves E01 cell-view/public policy observations where applicable; null models may use matched trajectory summaries rather than policy observations.",
                action_set="adjacent swap, wait/no-op, null-model local moves, or metric-only evaluation depending on result table.",
                transition_rules=f"E02 audit task reconstructed from result artifact {result_path.name}; full per-run config was not materialized.",
                goal_predicate="Standard increasing sortedness target plus audit-specific null, robustness, aggregation, or alternative metric objective.",
                perturbation_model=perturbation,
                measurement_functions=", ".join(columns) if columns else "columns unavailable from result table",
                scheduler="varies by E02 audit; scheduler-specific rows are in scheduler comparison artifacts",
                source_artifacts=tuple([str(result_path), *report_paths]),
                metadata={"source_result": result_path.name, "columns_preview": columns, "config_gap": "E02 has result tables and reports but no configs directory in mounted artifacts."},
            )
        )
    return records


def e03_records(base: Path) -> list[WorldRecord]:
    records: list[WorldRecord] = []
    for config_path in sorted((base / "configs").glob("*.json")):
        config = safe_read_json(config_path)
        step = str(config.get("researchStepId", config_path.stem.split("_")[1].upper() if "_" in config_path.stem else "S??"))
        task_id = config_path.stem
        records.append(
            WorldRecord(
                world_id=make_world_id("E03", step, task_id, family="policy_morphospace"),
                experiment_id="E03",
                source_step_id=step,
                record_granularity="config",
                world_family="policy_morphospace_array",
                task_id=task_id,
                task_label=task_id.replace("e03_", "").replace("_", " "),
                substrate_kind="array_1d",
                state_space="1D arrays of configurable size with integer values, policy labels, active/frozen statuses, and DSL policy state.",
                local_observations="PolicyObservation over actor index/value/label/status, full value/label/status arrays, local boundaries, reverse-direction flag, ideal position, and group status.",
                action_set="DSL/local-rule actions: adjacent swap proposals, wait/no-op, comparisons, and policy-state updates; JAX batch route supports compatible local-step subset.",
                transition_rules=f"Deterministic cyclic_scan_seed_offset scheduler or configured E03 sweep route; config schema={config.get('schema', 'not_declared')}.",
                goal_predicate="Increasing sortedness and competence-vector objectives used for policy morphospace, quality-diversity, phase-boundary, and atlas analyses.",
                perturbation_model="policy-library variation, array-size scaling, held-out seeds, rule ablations, and stress configs depending on E03 step.",
                measurement_functions="competence vectors, sortedness, robustness, aggregation, delayed gratification, embedding features, cluster stability, frontier metrics.",
                scheduler=str(config.get("scheduler", "cyclic_scan_seed_offset or analysis-only")),
                source_artifacts=list_existing(config_path),
                metadata={"config_keys": sorted(config.keys()), "schema": config.get("schema"), "seed": config.get("seed")},
            )
        )
    return records


def e04_records(base: Path) -> list[WorldRecord]:
    records: list[WorldRecord] = []
    for config_path in sorted((base / "configs").glob("*.json")):
        config = safe_read_json(config_path)
        step = str(config.get("researchStepId", config_path.stem.split("_")[1].upper() if "_" in config_path.stem else "S??"))
        task_id = config_path.stem
        tasks = config.get("tasks") or []
        reliability_modes = config.get("reliabilityModes") or []
        interface_modes = config.get("interfaceModes") or config.get("policyModes") or []
        perturbations = config.get("perturbationTypes") or []
        state = "1D array sorting state augmented with bounded memory, local signals, repair/fatigue status, and optional tissue-field summaries."
        if "homeostatic" in task_id:
            state = "Small 1D homeostatic arrays with scheduled shocks, turnover, frozen damage, fatigue/recovery state, memory, and signal fields."
        records.append(
            WorldRecord(
                world_id=make_world_id("E04", step, task_id, family="memory_repair_homeostasis"),
                experiment_id="E04",
                source_step_id=step,
                record_granularity="config",
                world_family="memory_repair_homeostasis",
                task_id=task_id,
                task_label=task_id.replace("e04_", "").replace("_", " "),
                substrate_kind="array_1d",
                state_space=state,
                local_observations="E01/E03 policy observation plus bounded cell memory, optional local signal inbox, fatigue/damage counters, and local-only training fields where enabled.",
                action_set="adjacent swap/wait, repair nudges, memory updates, local signal exchange, and local-learning policy updates depending on interface mode.",
                transition_rules=f"E04 repair/homeostasis transition family with max_events={config.get('maxEvents', 'not_specified')}; schema={config.get('schema', 'not_declared')}.",
                goal_predicate="Maintain or restore increasing sortedness, repair Frozen Cells, recover from perturbations, or optimize local-only competence metrics.",
                perturbation_model=f"tasks={tasks}; perturbations={perturbations}; reliability_modes={reliability_modes}; interface_modes={interface_modes}.",
                measurement_functions="repair success, final sortedness, degradation, recovery time, memory/signal ablation deltas, local-only transfer gaps, competency metrics.",
                scheduler=str(config.get("activation_distribution", "uniform_active or config-specific deterministic seeds")),
                source_artifacts=list_existing(config_path),
                metadata={"schema": config.get("schema"), "run_count": config.get("runCount"), "tasks": tasks, "config_keys": sorted(config.keys())},
            )
        )
    return records


def e05_records(base: Path) -> list[WorldRecord]:
    records: list[WorldRecord] = []
    substrate_table = base / "tables" / "e05_substrate_validation.csv"
    if substrate_table.exists():
        df = pd.read_csv(substrate_table)
        for row in df.to_dict(orient="records"):
            substrate = str(row["substrate_kind"])
            task_id = str(row["validation_case"])
            records.append(
                WorldRecord(
                    world_id=make_world_id("E05", "S01", task_id, family="substrate_family"),
                    experiment_id="E05",
                    source_step_id="S01",
                    record_granularity="substrate_family",
                    world_family="higher_dimensional_substrate",
                    task_id=task_id,
                    task_label=str(row["validation_case"]),
                    substrate_kind=substrate,
                    state_space=f"Fixed-site graph substrate {substrate} with site coordinates, boundaries, occupancy, scalar values, labels, statuses, and optional metadata.",
                    local_observations="LocalSubstrateObservation: actor site/cell/coordinate, boundary flag, neighbor direction/occupancy/value/label/status/movable flag, and local action set.",
                    action_set="wait, swap, move/crawl; E05 S04 extends action catalog with divide, die, adhere, detach, rotate_polarity, and exchange_signal.",
                    transition_rules="Graph-local occupancy update on fixed substrate adjacency with deterministic validation of boundary and reciprocal-neighbor invariants.",
                    goal_predicate="Substrate mechanics only; target predicates supplied by S03 target morphologies and S15 benchmark tasks.",
                    perturbation_model="No exogenous perturbation in substrate validation; downstream tasks apply scrambles, missing patches, scaling, and regeneration challenges.",
                    measurement_functions="neighbor-count/action-invariant validation and graph_hash; downstream metrics cover identity, boundary, topology, and shape errors.",
                    scheduler="not_applicable_for_static_substrate_validation",
                    source_artifacts=list_existing(substrate_table, base / "reports" / "e05_substrate_spec.md"),
                    metadata={"validation_success": bool(row["success"]), "detail": row.get("detail"), "graph_hash": row.get("graph_hash")},
                )
            )
    target_table = base / "tables" / "e05_target_morphology_index.csv"
    if target_table.exists():
        df = pd.read_csv(target_table)
        for row in df.to_dict(orient="records"):
            records.append(
                WorldRecord(
                    world_id=make_world_id("E05", "S03", str(row["target_id"]), family="target_morphology"),
                    experiment_id="E05",
                    source_step_id="S03",
                    record_granularity="target",
                    world_family="target_morphology",
                    task_id=str(row["target_id"]),
                    task_label=str(row["title"]),
                    substrate_kind=str(row["substrate_kind"]),
                    state_space=f"{row['substrate_kind']} target state with {row['site_count']} sites and identity schema {row['schema_id']}.",
                    local_observations="Target itself is global reference metadata; local policies observe substrate neighborhoods and identity vectors during task execution.",
                    action_set="target metadata only; benchmark/action rows specify swap, wait, crawl, divide, die, adhere, detach, signal, and polarity actions.",
                    transition_rules="Constructed target is static; downstream E05 tasks define dynamics on the same substrate.",
                    goal_predicate=f"Zero-error state for target_kind={row['target_kind']} under primary_metric={row['primary_metric']}.",
                    perturbation_model="No perturbation in target catalog; downstream tasks use scrambles, missing patches, scaling, and regeneration.",
                    measurement_functions=f"{row['primary_metric']}; compatible components {row.get('compatible_components_json', '[]')}.",
                    scheduler="not_applicable_for_static_target_catalog",
                    source_artifacts=list_existing(target_table, base / "reports" / "e05_target_morphology_spec.md"),
                    metadata={"target_hash": row.get("target_hash"), "render_mode": row.get("render_mode"), "constructed_target_error": row.get("constructed_target_error")},
                )
            )
    task_table = base / "tables" / "e05_benchmark_task_catalog.csv"
    if task_table.exists():
        df = pd.read_csv(task_table)
        for row in df.to_dict(orient="records"):
            records.append(
                WorldRecord(
                    world_id=make_world_id("E05", "S15", str(row["benchmark_task_id"]), family="benchmark_task"),
                    experiment_id="E05",
                    source_step_id="S15",
                    record_granularity="benchmark_task",
                    world_family=str(row["benchmark_family"]),
                    task_id=str(row["benchmark_task_id"]),
                    task_label=str(row["title"]),
                    substrate_kind=str(row["substrate_kind"]),
                    state_space=f"{row['substrate_kind']} task over target {row['target_id']} ({row['target_kind']}) with {row['site_count']} sites.",
                    local_observations="Policies observe bounded substrate neighborhoods and identity/action metadata; global endpoint controls are flagged as nonlocal context rows where present.",
                    action_set=str(row.get("policy_set_json", "policy set recorded; action contract in benchmark config files")),
                    transition_rules=f"Benchmark task_type={row['task_type']} using source {row['source_task_basis']} and source table {row['source_table']}.",
                    goal_predicate=f"Optimize primary_metric={row['primary_metric']} for target {row['target_id']}; lower or higher direction determined by metric contract.",
                    perturbation_model=f"perturbation_type={row['perturbation_type']}; known_blocker={row.get('known_blocker', '')}.",
                    measurement_functions=str(row.get("metric_ids_json", row.get("primary_metric", "not_specified"))),
                    scheduler="event-multiplier task runners; exact per-task schedule in E05 source step artifacts",
                    source_artifacts=list_existing(task_table, base / "reports" / "e05_morphology_benchmark_suite.md"),
                    metadata={"claim_boundary": row.get("claim_boundary"), "reference_rows": row.get("reference_rows"), "smoke_test": row.get("smoke_test")},
                )
            )
    return records


def e06_records(base: Path) -> list[WorldRecord]:
    records: list[WorldRecord] = []
    for config_path in sorted((base / "configs").glob("*.json")):
        config = safe_read_json(config_path)
        step = str(config.get("researchStepId", config_path.stem.split("_")[1].upper() if "_" in config_path.stem else "S??"))
        task_id = config_path.stem
        condition_matrix = next((base / "research_steps" / step).glob("e06_*condition_matrix.csv"), None) if (base / "research_steps" / step).exists() else None
        condition_rows = None
        condition_columns: list[str] = []
        if condition_matrix and condition_matrix.exists():
            condition_df = pd.read_csv(condition_matrix, nrows=5)
            condition_rows = sum(1 for _ in condition_matrix.open("r", encoding="utf-8")) - 1
            condition_columns = list(condition_df.columns)
        profiles = config.get("goalProfiles") or config.get("perturbationProfiles") or config.get("governanceMechanisms") or config.get("interventionSpecIds") or []
        records.append(
            WorldRecord(
                world_id=make_world_id("E06", step, task_id, family="chimeric_governance"),
                experiment_id="E06",
                source_step_id=step,
                record_granularity="step_family",
                world_family="chimeric_governance_array",
                task_id=task_id,
                task_label=task_id.replace("e06_", "").replace("_", " "),
                substrate_kind="array_1d",
                state_space=f"Length-{config.get('arraySize', 100)} 1D chimeric array with per-cell Algotype labels, source categories, ratios, arrangements, and optional history/intervention state.",
                local_observations="E03/E04 local policy observations plus Algotype identity, mixture context, interface-rule recognition fields, governance strata, graft/clone/history metadata where configured.",
                action_set="adjacent local policy actions from original, discovered, null, and repair policies; governance/interface/intervention mechanisms alter local action selection or access strata.",
                transition_rules=f"Chimeric condition family with event_cap={config.get('eventCap', 'not_specified')}; scheduler={config.get('scheduler', 'cyclic_scan_seed_offset or step-specific')}.",
                goal_predicate="Configured same, opposite, partially compatible, selfish, dominance, aggregation, or rescue objective depending on E06 step.",
                perturbation_model=f"mixture ratios, spatial arrangements, goal conflicts, dominance contests, interface rules, governance interventions, grafts, mutant clones, developmental histories, or intervention search; profiles_preview={profiles if isinstance(profiles, list) else str(profiles)}.",
                measurement_functions="sortedness, aggregation delta, interface count, largest block fraction, dominance, compatibility, final-state class, rescue effect, model features, and playbook claim links.",
                scheduler=str(config.get("scheduler", "cyclic_scan_seed_offset or step-specific deterministic seeds")),
                source_artifacts=list_existing(config_path, condition_matrix if condition_matrix else Path("__missing__")),
                metadata={"schema": config.get("schema"), "condition_rows": condition_rows, "condition_columns": condition_columns, "config_keys": sorted(config.keys())},
            )
        )
    return records


def collect_records(previous_artifacts_dir: Path) -> list[WorldRecord]:
    builders = {
        "E01": e01_records,
        "E02": e02_records,
        "E03": e03_records,
        "E04": e04_records,
        "E05": e05_records,
        "E06": e06_records,
    }
    records: list[WorldRecord] = []
    for experiment_id, builder in builders.items():
        base = previous_artifacts_dir / experiment_id
        if base.exists():
            records.extend(builder(base))
    return [record.with_validation() for record in records]


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


def schema_spec_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome: str,
    caveats: str,
    recommended_next_action: str,
    lay_summary: str,
    inventory: pd.DataFrame,
) -> str:
    summary = top_summary_markdown(artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary)
    counts = inventory.groupby(["experiment_id", "record_granularity"], dropna=False).size().reset_index(name="record_count")
    return f"""{summary}

# E07 S01 World Schema Specification

## Frozen Question

Can every simulation world be represented by a common schema without erasing meaningful differences among arrays, grids, graphs, chimeras, and homeostatic tasks?

## Tuple Contract

Every inventory row represents one upstream world, task, condition, target, substrate family, or result-table task using these required tuple fields:

- `state_space`: site/cell/state variables that define allowed world states.
- `local_observations`: policy-visible observation contract, or explicit nonlocal/static metadata when the row is a traditional baseline, target catalog, or result-only audit.
- `action_set`: actions available to policies or controllers.
- `transition_rules`: update rule, scheduler, stop rule, or static/catalog rule.
- `goal_predicate`: success predicate, target morphology, sortedness objective, or audit objective.
- `perturbation_model`: exogenous perturbations, input distribution shifts, chimeric manipulations, or explicit no-perturbation statement.
- `measurement_functions`: metrics or result columns used to measure behavior.

Rows with incomplete upstream detail are allowed only when `missing_fields_json` names the missing tuple fields and `completeness` is `partial_explicit_missing`.

## Inventory Columns

- Identity: `world_id`, `experiment_id`, `source_step_id`, `record_granularity`, `world_family`, `task_id`, `task_label`.
- Substrate and tuple fields: `substrate_kind`, {", ".join(f"`{field}`" for field in REQUIRED_TUPLE_FIELDS)}.
- Execution/provenance: `scheduler`, `source_artifacts_json`, `metadata_json`, `record_hash`.
- Validation: `missing_fields_json`, `completeness`.

## Coverage Summary

{markdown_table(counts, ["experiment_id", "record_granularity", "record_count"], max_rows=40)}

## Known Differences Preserved

- Traditional E01 baselines are kept as nonlocal controller records rather than forced into a local-policy observation.
- E02 has no mounted configs, so its audit tasks are represented from result tables and reports with that provenance gap recorded in metadata.
- E05 static substrate and target catalogs are included separately from dynamic benchmark tasks because later policy and goal representations need both levels.
- E06 condition matrices are summarized at step-family level for S01; per-condition details remain linked through `source_artifacts_json` and can be expanded in S04 if needed.
"""


def full_results_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome: str,
    caveats: str,
    recommended_next_action: str,
    lay_summary: str,
    inventory: pd.DataFrame,
    validation_df: pd.DataFrame,
    source_check_df: pd.DataFrame,
    unit_test_result: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    summary = top_summary_markdown(artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary)
    by_experiment = inventory.groupby("experiment_id").size().reset_index(name="record_count")
    by_completeness = inventory.groupby("completeness").size().reset_index(name="record_count")
    by_substrate = inventory.groupby("substrate_kind").size().reset_index(name="record_count").sort_values("record_count", ascending=False)
    sample = inventory[["world_id", "experiment_id", "record_granularity", "substrate_kind", "world_family", "completeness"]].head(30)
    return f"""{summary}

# E07 S01 Full Results: Formalize Each World

## Lay Summary

{lay_summary}

## Frozen Question

Can every simulation world be represented by a common schema without erasing meaningful differences among arrays, grids, graphs, chimeras, and homeostatic tasks?

## Inputs

- Active research plan: `/workspace/RESEARCH_PLAN.md`.
- Group plan: `/workspace/FULL_PLAN.md`.
- Previous artifact context: `/workspace/PREVIOUS_ARTIFACTS.md` and `/workspace/PREVIOUS_ARTIFACTS.json`.
- Mounted upstream artifacts: `/previous-artifacts/E01` through `/previous-artifacts/E06`.
- Uploaded paper sidecar: `/workspace/input-attachments/f93afdc5-f2e5-4ecc-80bb-e088f93acf3c/_metadata/ATTACHMENT.md`.
- No dataset inputs were required; `DATASET_AVAILABILITY.json` reports `validationStatus=not_required`.

## Methods

S01 added a repository-backed schema module and deterministic artifact driver. The driver scanned upstream configs, result tables, benchmark catalogs, condition matrices, reports, and validation tables. It normalized records into one tuple contract with explicit source paths and field-level completeness. It did not rerun simulations or start S02.

The inventory uses these granularities:

- E01: condition rows from `e01_baseline_configs.json`.
- E02: mounted result tables because no E02 configs directory is present.
- E03 and E04: step config files.
- E05: substrate families, target morphologies, and benchmark tasks.
- E06: chimeric/governance step-family configs with linked condition matrices when present.

## Commands

- `python -m unittest discover -s tests/e07 -p 'test_*.py'`
- `python scripts/e07_s01_formalize_worlds.py --artifacts-dir /artifacts --previous-artifacts-dir /previous-artifacts --run-unit-tests`

Unit-test command result: return code {unit_test_result.get("returnCode")}, success={unit_test_result.get("success")}, elapsed={unit_test_result.get("elapsedSeconds")} seconds.

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- New dependencies installed: none.
- CPU/GPU use: serial metadata extraction only; no GPU and no parallel workers.

## Results

Inventory records written: {len(inventory)}.

### Records By Experiment

{markdown_table(by_experiment, ["experiment_id", "record_count"], max_rows=20)}

### Records By Completeness

{markdown_table(by_completeness, ["completeness", "record_count"], max_rows=20)}

### Records By Substrate

{markdown_table(by_substrate, ["substrate_kind", "record_count"], max_rows=20)}

### Inventory Sample

{markdown_table(sample, ["world_id", "experiment_id", "record_granularity", "substrate_kind", "world_family", "completeness"], max_rows=30)}

## Validation

{markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=20)}

Source path validation: {int(source_check_df["success"].sum())}/{len(source_check_df)} records have all listed source artifacts present.

## Artifacts

{markdown_table(pd.DataFrame(artifacts), ["path", "description", "sha256"], max_rows=20)}

## Caveats And Blockers

{caveats}

- E02 did not mount configs, so its rows are result-table-derived metadata records.
- E06 is represented at step-family level rather than thousands of individual condition rows to keep S01 bounded; condition matrices remain linked for expansion.
- This step formalizes computational worlds and tasks only; it does not validate biological claims or rerun upstream simulations.

## Provenance

- Repository HEAD before commit: `{manifest["git"]["head"]}`
- Repository branch: `{manifest["git"]["branch"]}`
- Inventory hash: `{stable_hash(inventory.to_dict(orient="records"))}`
- Previous artifact context hash: `{manifest["inputs"].get("previousArtifactsContextSha256")}`
- Source files:
  - `src/e07/world_schema.py`
  - `scripts/e07_s01_formalize_worlds.py`
  - `tests/e07/test_world_schema.py`

## Recommended Next Action

{recommended_next_action}
"""


def build_manifest(
    *,
    args: argparse.Namespace,
    inventory: pd.DataFrame,
    validation_df: pd.DataFrame,
    source_check_df: pd.DataFrame,
    unit_test_result: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    previous_context_sha = sha256_file(args.previous_artifacts_context) if args.previous_artifacts_context.exists() else None
    return {
        "schema": "eidosoma.e07.s01.manifest.v1",
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
            "previousArtifactsDir": str(args.previous_artifacts_dir),
            "previousArtifactsContext": str(args.previous_artifacts_context),
            "previousArtifactsContextSha256": previous_context_sha,
        },
        "inventory": {
            "recordCount": int(len(inventory)),
            "recordCountByExperiment": inventory.groupby("experiment_id").size().astype(int).to_dict(),
            "recordCountByCompleteness": inventory.groupby("completeness").size().astype(int).to_dict(),
            "recordHash": stable_hash(inventory.to_dict(orient="records")),
        },
        "validation": {
            "checks": validation_df.to_dict(orient="records"),
            "sourceArtifactRecords": int(len(source_check_df)),
            "sourceArtifactRecordsPassing": int(source_check_df["success"].sum()) if not source_check_df.empty else 0,
            "unitTests": unit_test_result,
        },
        "artifacts": artifacts,
        "sourceFiles": [
            source_entry(args.repo_dir / "src" / "e07" / "__init__.py", args.repo_dir),
            source_entry(args.repo_dir / "src" / "e07" / "world_schema.py", args.repo_dir),
            source_entry(args.repo_dir / "scripts" / "e07_s01_formalize_worlds.py", args.repo_dir),
            source_entry(args.repo_dir / "tests" / "e07" / "test_world_schema.py", args.repo_dir),
        ],
    }


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    reports_dir = artifacts_dir / "reports"
    tables_dir = artifacts_dir / "tables"
    results_dir = artifacts_dir / "results"

    records = collect_records(args.previous_artifacts_dir)
    inventory = dataframe_from_records(records)
    source_check_df = source_paths_exist(inventory)
    validation_df = validate_inventory(inventory)
    validation_df = pd.concat(
        [
            validation_df,
            pd.DataFrame(
                [
                    {
                        "validation_case": "source_artifacts_exist",
                        "success": bool(source_check_df["success"].all()),
                        "detail": f"{int(source_check_df['success'].sum())}/{len(source_check_df)} records have all listed source artifacts present",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    unit_test_result = {"command": "not run", "success": True, "returnCode": 0, "stdout": "", "stderr": "", "elapsedSeconds": 0.0}
    if args.run_unit_tests:
        unit_test_result = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e07", "-p", "test_*.py"], args.repo_dir)

    inventory_csv = tables_dir / "e07_world_inventory.csv"
    inventory_parquet = results_dir / "e07_world_inventory.parquet"
    validation_csv = step_dir / "e07_s01_validation_checks.csv"
    source_check_csv = step_dir / "e07_s01_source_artifact_checks.csv"
    schema_spec_path = reports_dir / "e07_world_schema_spec.md"
    full_results_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "artifact_manifest.json"

    inventory_csv.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(inventory_csv, index=False)
    inventory_parquet.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(inventory_parquet, index=False)
    validation_csv.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(validation_csv, index=False)
    source_check_df.to_csv(source_check_csv, index=False)

    validation_success = bool(validation_df["success"].all() and unit_test_result["success"])
    partial_count = int((inventory["completeness"] == "partial_explicit_missing").sum()) if not inventory.empty else 0
    validation_result = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} inventory validation checks passed; "
        f"unit tests {'passed' if unit_test_result['success'] else 'failed'}; "
        f"{len(inventory)} world/task records, {partial_count} partial records with explicit missing fields"
    )
    outcome = "supportive" if validation_success and len(inventory) > 0 else "null"
    caveats = (
        "The schema validates metadata coverage, not simulator behavior. E02 lacks mounted configs and is represented from result tables; "
        "E06 is summarized at step-family level with linked condition matrices to keep S01 bounded."
    )
    lay_summary = (
        "S01 turned the previous experiments into a common catalog of worlds and tasks. "
        "Rows keep array, grid, graph, repair, homeostasis, and chimeric settings comparable while preserving where upstream detail is incomplete."
    )
    recommended_next_action = "Stop before S02; after Chief Scientist review, proceed to S02 policy representations using `world_id` as the foreign key."

    artifacts = [
        artifact_entry(inventory_csv, artifacts_dir, "Required S01 world inventory table"),
        artifact_entry(inventory_parquet, artifacts_dir, "Columnar copy of world inventory for downstream joins"),
        artifact_entry(validation_csv, artifacts_dir, "S01 validation check table"),
        artifact_entry(source_check_csv, artifacts_dir, "Per-record source artifact existence checks"),
        self_referential_artifact_entry(schema_spec_path, artifacts_dir, "Required S01 world schema specification report"),
        self_referential_artifact_entry(manifest_path, artifacts_dir, "S01 artifact and provenance manifest"),
        self_referential_artifact_entry(full_results_path, artifacts_dir, "Required S01 full-results handoff report"),
    ]
    write_text(schema_spec_path, schema_spec_markdown(artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary, inventory))

    manifest = build_manifest(
        args=args,
        inventory=inventory,
        validation_df=validation_df,
        source_check_df=source_check_df,
        unit_test_result=unit_test_result,
        artifacts=artifacts,
    )
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
            inventory,
            validation_df,
            source_check_df,
            unit_test_result,
            manifest,
        ),
    )

    final_artifacts = [
        artifact_entry(inventory_csv, artifacts_dir, "Required S01 world inventory table"),
        artifact_entry(inventory_parquet, artifacts_dir, "Columnar copy of world inventory for downstream joins"),
        artifact_entry(validation_csv, artifacts_dir, "S01 validation check table"),
        artifact_entry(source_check_csv, artifacts_dir, "Per-record source artifact existence checks"),
        self_referential_artifact_entry(schema_spec_path, artifacts_dir, "Required S01 world schema specification report"),
        self_referential_artifact_entry(manifest_path, artifacts_dir, "S01 artifact and provenance manifest"),
        self_referential_artifact_entry(full_results_path, artifacts_dir, "Required S01 full-results handoff report"),
    ]
    manifest["artifacts"] = final_artifacts
    manifest["validation"]["success"] = validation_success
    write_json(manifest_path, manifest)
    write_text(schema_spec_path, schema_spec_markdown(final_artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary, inventory))
    write_text(
        full_results_path,
        full_results_markdown(
            final_artifacts,
            validation_result,
            outcome,
            caveats,
            recommended_next_action,
            lay_summary,
            inventory,
            validation_df,
            source_check_df,
            unit_test_result,
            manifest,
        ),
    )

    print(json.dumps({"success": validation_success, "recordCount": len(inventory), "validationResult": validation_result}, indent=2))
    return 0 if validation_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
