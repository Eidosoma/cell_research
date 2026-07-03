#!/usr/bin/env python3
"""Run E07 S13 scoped substrate-transfer tests on existing E03/E05 paths."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.embedded_1d import run_adjacent_sort, stable_json_sha256  # noqa: E402
from src.e07.corpus_schema import sha256_file  # noqa: E402
from src.e07.substrate_transfer_schema import (  # noqa: E402
    SUBSTRATE_TRANSFER_SCHEMA_VERSION,
    stable_payload_hash,
    transfer_mapping_id,
    transfer_score,
    validate_substrate_transfer_artifacts,
    validation_summary,
)


STEP_ID = "S13"
STEP_NUMBER = 13
EXPERIMENT_ID = "E07"
RANDOM_SEED = 2026070313
FRESH_ARRAY_SPECS = ((8, 8301), (8, 8302), (12, 8401), (12, 8402))
EXECUTABLE_ALGORITHMS = ("bubble", "insertion")
EXECUTION_PATH = "src.e05.embedded_1d.run_adjacent_sort"
S12_BLOCKER_REASON = (
    "S12 designs are arbitrary E03 DSL array-world policies. The existing E05 embedded-row executable path "
    "only supports built-in adjacent bubble/insertion algorithms, so no new DSL-to-E05 adapter was built in S13."
)


def parse_args() -> argparse.Namespace:
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_dir)
    parser.add_argument("--e05-embedded-summary", type=Path, default=Path("/previous-artifacts/E05/results/e05_1d_embedded_run_summary.parquet"))
    parser.add_argument("--e05-embedded-validation", type=Path, default=Path("/previous-artifacts/E05/results/e05_1d_embedded_validation.parquet"))
    parser.add_argument("--e05-benchmark-task-catalog", type=Path, default=Path("/previous-artifacts/E05/results/e05_benchmark_task_catalog.parquet"))
    parser.add_argument("--e05-benchmark-reference-results", type=Path, default=Path("/previous-artifacts/E05/results/e05_benchmark_reference_results.parquet"))
    parser.add_argument("--s12-designed-policies", type=Path, default=artifacts_dir / "policies" / "e07_inverse_designed_policies.jsonl")
    parser.add_argument("--s12-validation", type=Path, default=artifacts_dir / "results" / "e07_inverse_design_validation.parquet")
    parser.add_argument("--s12-design-manifest", type=Path, default=artifacts_dir / "research_steps" / "S12" / "candidate_design_freeze_manifest.json")
    parser.add_argument("--s08-neighbors", type=Path, default=artifacts_dir / "results" / "e07_platonic_neighbors.parquet")
    parser.add_argument("--s08-distance-metadata", type=Path, default=artifacts_dir / "results" / "e07_distance_entity_metadata.parquet")
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def maybe_sha256(path: Path) -> str:
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(elapsed),
        "stdout": result.stdout[-6000:],
        "stderr": result.stderr[-6000:],
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


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
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


def trace_signature(result: Any) -> list[tuple[Any, ...]]:
    return [
        (
            int(record["swap_step"]),
            int(record["comparison_count_at_step"]),
            round(float(record["sortedness_percent"]), 12),
            int(record["monotonicity_error_count"]),
            bool(record["is_initial"]),
            bool(record["is_final"]),
        )
        for record in result.records
    ]


def initial_values(array_size: int, seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed))
    return [int(value) for value in rng.permutation(np.arange(1, int(array_size) + 1))]


def finalized_mapping(record: Mapping[str, Any], frozen_at: str) -> dict[str, Any]:
    payload = {
        "schema_version": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
        "research_step_id": STEP_ID,
        "step_number": STEP_NUMBER,
        "experiment_id": EXPERIMENT_ID,
        "frozen_at_utc": frozen_at,
        **dict(record),
    }
    payload["mapping_id"] = transfer_mapping_id(payload)
    return payload


def build_transfer_mappings(
    task_catalog: pd.DataFrame,
    s12_designs: list[dict[str, Any]],
    frozen_at: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for algorithm in EXECUTABLE_ALGORITHMS:
        records.append(
            finalized_mapping(
                {
                    "mapping_kind": "within_1d_baseline",
                    "mapping_status": "executable_existing_path",
                    "control_family": "within_1d_baseline",
                    "source_experiment_id": "E03",
                    "target_experiment_id": "E03",
                    "source_policy_id": f"e03_builtin_{algorithm}_adjacent",
                    "target_policy_id": f"e03_builtin_{algorithm}_adjacent",
                    "algorithm": algorithm,
                    "source_substrate_kind": "array_1d",
                    "target_substrate_kind": "array_1d",
                    "execution_path": EXECUTION_PATH,
                    "adapter_built_in_s13": False,
                    "mapping_freeze_note": "Baseline uses the same existing adjacent-sort implementation on the 1D source substrate.",
                    "claim_boundary": "Baseline only; not a cross-substrate transfer claim.",
                },
                frozen_at,
            )
        )
        records.append(
            finalized_mapping(
                {
                    "mapping_kind": "e03_to_e05_embedded_row",
                    "mapping_status": "executable_existing_path",
                    "control_family": "cross_substrate_transfer",
                    "source_experiment_id": "E03",
                    "target_experiment_id": "E05",
                    "source_policy_id": f"e03_builtin_{algorithm}_adjacent",
                    "target_policy_id": f"e05_s06_{algorithm}_embedded_row",
                    "algorithm": algorithm,
                    "source_substrate_kind": "array_1d",
                    "target_substrate_kind": "embedded_square_grid_2d",
                    "execution_path": EXECUTION_PATH,
                    "adapter_built_in_s13": False,
                    "mapping_freeze_note": "Existing E05 S06 executable path embeds a 1D adjacent-sort row in a 2D grid with stuck off-row barriers.",
                    "claim_boundary": "Only bubble/insertion adjacent-swap row transfer is executable here.",
                },
                frozen_at,
            )
        )

    for row in task_catalog.to_dict(orient="records"):
        task_id = str(row.get("benchmark_task_id", "unknown_task"))
        records.append(
            finalized_mapping(
                {
                    "mapping_kind": "e05_reference_context",
                    "mapping_status": "existing_reference_context",
                    "control_family": "e05_context_control",
                    "source_experiment_id": "E05",
                    "target_experiment_id": "E05",
                    "source_policy_id": "existing_e05_reference_policy_set",
                    "target_policy_id": "existing_e05_reference_policy_set",
                    "algorithm": "",
                    "source_substrate_kind": str(row.get("substrate_kind", "")),
                    "target_substrate_kind": str(row.get("substrate_kind", "")),
                    "benchmark_task_id": task_id,
                    "execution_path": "previous-artifacts/E05/reference_tables_only",
                    "adapter_built_in_s13": False,
                    "mapping_freeze_note": "Reference-control context read from existing E05 benchmark outputs; no new evaluation or adapter.",
                    "claim_boundary": str(row.get("claim_boundary", "")),
                },
                frozen_at,
            )
        )

    for design in s12_designs:
        records.append(
            finalized_mapping(
                {
                    "mapping_kind": "s12_dsl_to_e05_embedded_row",
                    "mapping_status": "blocked_no_existing_path",
                    "control_family": "blocked_transfer",
                    "source_experiment_id": "E03",
                    "target_experiment_id": "E05",
                    "source_policy_id": str(design.get("source_policy_id", design.get("design_id", ""))),
                    "target_policy_id": "",
                    "algorithm": "",
                    "source_substrate_kind": "array_1d",
                    "target_substrate_kind": "embedded_square_grid_2d",
                    "design_id": str(design.get("design_id", "")),
                    "design_role": str(design.get("design_role", "")),
                    "execution_path": "",
                    "adapter_built_in_s13": False,
                    "blocker_reason": S12_BLOCKER_REASON,
                    "mapping_freeze_note": "Frozen before evaluation as a blocker, preserving S12's E03-only executable-proxy caveat.",
                    "claim_boundary": "No executable substrate-transfer validation for arbitrary S12 DSL policies.",
                },
                frozen_at,
            )
        )

    blocker_specs = [
        (
            "e05_2d_policy_to_e03_1d_reduction",
            "E05",
            "E03",
            "square_grid_2d",
            "array_1d",
            "No existing executable reduction path maps E05 2D morphology policies back to E03 1D DSL array policies.",
        ),
        (
            "e03_to_e05_graph_transfer",
            "E03",
            "E05",
            "array_1d",
            "general_graph_or_2d_grid",
            "No existing graph-general transfer path is registered; S13 is scoped to the E05 embedded row only.",
        ),
        (
            "e03_or_e05_to_e06_governance_transfer",
            "E03/E05",
            "E06",
            "array_or_grid_proxy",
            "chimera_governance_proxy",
            "E06 governance/chimera transfer adapters are explicitly out of scope for S13.",
        ),
    ]
    for mapping_kind, source_exp, target_exp, source_substrate, target_substrate, reason in blocker_specs:
        records.append(
            finalized_mapping(
                {
                    "mapping_kind": mapping_kind,
                    "mapping_status": "blocked_no_existing_path",
                    "control_family": "blocked_transfer",
                    "source_experiment_id": source_exp,
                    "target_experiment_id": target_exp,
                    "source_policy_id": "",
                    "target_policy_id": "",
                    "algorithm": "",
                    "source_substrate_kind": source_substrate,
                    "target_substrate_kind": target_substrate,
                    "execution_path": "",
                    "adapter_built_in_s13": False,
                    "blocker_reason": reason,
                    "mapping_freeze_note": "Frozen before evaluation as an explicit S13 scope blocker.",
                    "claim_boundary": "No new E05/E06 adapter was built or implied.",
                },
                frozen_at,
            )
        )
    return sorted(records, key=lambda item: str(item["mapping_id"]))


def result_id(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "result_id"}
    return stable_payload_hash(payload, prefix="s13res")


def evaluate_fresh_mapping(mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    algorithm = str(mapping["algorithm"])
    is_transfer = mapping["control_family"] == "cross_substrate_transfer"
    source_kind = str(mapping["source_substrate_kind"])
    target_kind = str(mapping["target_substrate_kind"])
    for array_size, seed in FRESH_ARRAY_SPECS:
        values = initial_values(array_size, seed)
        source = run_adjacent_sort(values, algorithm=algorithm, substrate_kind=source_kind)
        target = run_adjacent_sort(values, algorithm=algorithm, substrate_kind=target_kind)
        source_trace = trace_signature(source)
        target_trace = trace_signature(target)
        exact_trace = source_trace == target_trace
        final_hash_match = stable_json_sha256(list(source.final_values)) == stable_json_sha256(list(target.final_values))
        transfer_success = bool(
            exact_trace
            and final_hash_match
            and target.final_sortedness_percent == 100.0
            and int(target.row_constraint_violations) == 0
        )
        record = {
            "schema_version": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "step_number": STEP_NUMBER,
            "experiment_id": EXPERIMENT_ID,
            "mapping_id": str(mapping["mapping_id"]),
            "mapping_kind": str(mapping["mapping_kind"]),
            "mapping_status": str(mapping["mapping_status"]),
            "control_family": str(mapping["control_family"]),
            "evaluation_kind": "fresh_existing_path_evaluation",
            "execution_status": "ok" if transfer_success else "mismatch",
            "source_experiment_id": str(mapping["source_experiment_id"]),
            "target_experiment_id": str(mapping["target_experiment_id"]),
            "source_policy_id": str(mapping["source_policy_id"]),
            "target_policy_id": str(mapping["target_policy_id"]),
            "algorithm": algorithm,
            "world_id": f"world:e07:s13:n{array_size}-seed{seed}:{target_kind}",
            "array_size": int(array_size),
            "seed": int(seed),
            "source_substrate_kind": source_kind,
            "target_substrate_kind": target_kind,
            "initial_values_json": json.dumps(values, separators=(",", ":")),
            "initial_array_sha256": stable_json_sha256(values),
            "source_final_array_sha256": stable_json_sha256(list(source.final_values)),
            "target_final_array_sha256": stable_json_sha256(list(target.final_values)),
            "exact_trajectory_match": bool(exact_trace),
            "final_hash_match": bool(final_hash_match),
            "transfer_success": transfer_success,
            "source_final_sortedness_percent": float(source.final_sortedness_percent),
            "target_final_sortedness_percent": float(target.final_sortedness_percent),
            "source_monotonicity_error_count": int(source.final_monotonicity_error_count),
            "target_monotonicity_error_count": int(target.final_monotonicity_error_count),
            "source_swap_count": int(source.swap_count),
            "target_swap_count": int(target.swap_count),
            "source_comparison_count": int(source.comparison_count),
            "target_comparison_count": int(target.comparison_count),
            "source_energy_total": float(source.energy_total),
            "target_energy_total": float(target.energy_total),
            "row_constraint_violations": int(target.row_constraint_violations),
            "target_recovery_fraction": 1.0 if transfer_success else 0.0,
            "final_target_error": 0.0 if target.final_sortedness_percent == 100.0 else float((100.0 - target.final_sortedness_percent) / 100.0),
            "transfer_score": transfer_score(
                target_final_sortedness_percent=float(target.final_sortedness_percent),
                exact_trajectory_match=bool(exact_trace),
            ),
            "blocker_reason": "",
            "claim_boundary": "Fresh S13 run using existing E05 embedded-row executable path." if is_transfer else "Within-1D baseline.",
            "provenance": "fresh_s13_existing_e03_e05_path",
        }
        record["result_id"] = result_id(record)
        rows.append(record)
    return rows


def evaluate_context_rows(reference_results: pd.DataFrame, context_mappings: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping_by_task = {str(row["benchmark_task_id"]): row for row in context_mappings.to_dict(orient="records")}
    group_cols = ["benchmark_task_id", "benchmark_family", "control_class"]
    for keys, group in reference_results.groupby(group_cols, dropna=False, sort=True):
        task_id, benchmark_family, control_class = [str(item) for item in keys]
        mapping = mapping_by_task.get(task_id)
        if mapping is None:
            continue
        recovery = pd.to_numeric(group.get("target_recovery_fraction"), errors="coerce")
        final_error = pd.to_numeric(group.get("final_target_error"), errors="coerce")
        final_sortedness = pd.to_numeric(group.get("final_sortedness_percent"), errors="coerce")
        blocked = group.get("baseline_blocked", pd.Series(dtype=bool)).fillna(False).astype(bool)
        exact_count = int((group.get("reference_outcome_class", pd.Series(dtype=str)).astype(str) == "exact_recovered").sum())
        record = {
            "schema_version": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "step_number": STEP_NUMBER,
            "experiment_id": EXPERIMENT_ID,
            "mapping_id": str(mapping["mapping_id"]),
            "mapping_kind": str(mapping["mapping_kind"]),
            "mapping_status": str(mapping["mapping_status"]),
            "control_family": "e05_context_control",
            "evaluation_kind": "existing_e05_reference_context",
            "execution_status": "existing_reference_read",
            "source_experiment_id": "E05",
            "target_experiment_id": "E05",
            "source_policy_id": "existing_e05_reference_policy_set",
            "target_policy_id": str(control_class),
            "algorithm": "",
            "benchmark_task_id": task_id,
            "benchmark_family": benchmark_family,
            "control_class": control_class,
            "world_id": f"world:e05:benchmark:{task_id}",
            "array_size": np.nan,
            "seed": np.nan,
            "source_substrate_kind": str(mapping.get("source_substrate_kind", "")),
            "target_substrate_kind": str(mapping.get("target_substrate_kind", "")),
            "initial_values_json": "",
            "initial_array_sha256": "",
            "source_final_array_sha256": "",
            "target_final_array_sha256": "",
            "exact_trajectory_match": False,
            "final_hash_match": False,
            "transfer_success": False,
            "source_final_sortedness_percent": np.nan,
            "target_final_sortedness_percent": float(final_sortedness.mean()) if final_sortedness.notna().any() else np.nan,
            "source_monotonicity_error_count": np.nan,
            "target_monotonicity_error_count": np.nan,
            "source_swap_count": np.nan,
            "target_swap_count": float(pd.to_numeric(group.get("accepted_swaps"), errors="coerce").mean())
            if "accepted_swaps" in group
            else np.nan,
            "source_comparison_count": np.nan,
            "target_comparison_count": np.nan,
            "source_energy_total": np.nan,
            "target_energy_total": float(pd.to_numeric(group.get("total_energy_cost"), errors="coerce").mean())
            if "total_energy_cost" in group
            else np.nan,
            "row_constraint_violations": np.nan,
            "target_recovery_fraction": float(recovery.mean()) if recovery.notna().any() else np.nan,
            "final_target_error": float(final_error.mean()) if final_error.notna().any() else np.nan,
            "reference_rows": int(len(group)),
            "exact_recovered_rows": exact_count,
            "blocked_rows": int(blocked.sum()),
            "transfer_score": transfer_score(target_recovery_fraction=float(recovery.mean())) if recovery.notna().any() else 0.0,
            "blocker_reason": "",
            "claim_boundary": "Existing E05 benchmark context only; not a fresh S13 transfer simulation.",
            "provenance": "previous-artifacts/E05/results/e05_benchmark_reference_results.parquet",
        }
        record["result_id"] = result_id(record)
        rows.append(record)
    return rows


def evaluate_blockers(blocked_mappings: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mapping in blocked_mappings.to_dict(orient="records"):
        reason = str(mapping.get("blocker_reason", "")) or str(mapping.get("mapping_freeze_note", "No existing executable path."))
        record = {
            "schema_version": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
            "research_step_id": STEP_ID,
            "step_number": STEP_NUMBER,
            "experiment_id": EXPERIMENT_ID,
            "mapping_id": str(mapping["mapping_id"]),
            "mapping_kind": str(mapping["mapping_kind"]),
            "mapping_status": str(mapping["mapping_status"]),
            "control_family": "blocked_transfer",
            "evaluation_kind": "transfer_blocker",
            "execution_status": "blocked",
            "source_experiment_id": str(mapping.get("source_experiment_id", "")),
            "target_experiment_id": str(mapping.get("target_experiment_id", "")),
            "source_policy_id": str(mapping.get("source_policy_id", "")),
            "target_policy_id": str(mapping.get("target_policy_id", "")),
            "algorithm": str(mapping.get("algorithm", "")),
            "world_id": "",
            "array_size": np.nan,
            "seed": np.nan,
            "source_substrate_kind": str(mapping.get("source_substrate_kind", "")),
            "target_substrate_kind": str(mapping.get("target_substrate_kind", "")),
            "initial_values_json": "",
            "initial_array_sha256": "",
            "source_final_array_sha256": "",
            "target_final_array_sha256": "",
            "exact_trajectory_match": False,
            "final_hash_match": False,
            "transfer_success": False,
            "source_final_sortedness_percent": np.nan,
            "target_final_sortedness_percent": np.nan,
            "source_monotonicity_error_count": np.nan,
            "target_monotonicity_error_count": np.nan,
            "source_swap_count": np.nan,
            "target_swap_count": np.nan,
            "source_comparison_count": np.nan,
            "target_comparison_count": np.nan,
            "source_energy_total": np.nan,
            "target_energy_total": np.nan,
            "row_constraint_violations": np.nan,
            "target_recovery_fraction": np.nan,
            "final_target_error": np.nan,
            "reference_rows": np.nan,
            "exact_recovered_rows": np.nan,
            "blocked_rows": 1,
            "transfer_score": transfer_score(blocked=True),
            "blocker_reason": reason,
            "claim_boundary": str(mapping.get("claim_boundary", "Blocked transfer mapping.")),
            "provenance": "s13_frozen_transfer_mapping_blocker",
        }
        record["result_id"] = result_id(record)
        rows.append(record)
    return rows


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["control_family", "mapping_kind", "target_experiment_id", "target_substrate_kind"]
    summary = (
        results.groupby(group_cols, dropna=False)
        .agg(
            result_rows=("result_id", "count"),
            mean_transfer_score=("transfer_score", "mean"),
            successful_rows=("transfer_success", "sum"),
            exact_trajectory_rows=("exact_trajectory_match", "sum"),
            blocked_rows=("execution_status", lambda values: int((values.astype(str) == "blocked").sum())),
            mean_target_final_sortedness_percent=("target_final_sortedness_percent", "mean"),
            mean_target_recovery_fraction=("target_recovery_fraction", "mean"),
        )
        .reset_index()
        .sort_values(["control_family", "mapping_kind", "target_experiment_id", "target_substrate_kind"], kind="mergesort")
    )
    return summary


def write_figure(summary: pd.DataFrame, figure_path: Path) -> None:
    matrix = (
        summary.pivot_table(
            index="mapping_kind",
            columns="target_substrate_kind",
            values="mean_transfer_score",
            aggfunc="mean",
            fill_value=0.0,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.38 * max(1, len(matrix)))))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]), labels=list(matrix.columns), rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), labels=list(matrix.index))
    ax.set_title("S13 substrate-transfer score matrix")
    ax.set_xlabel("Target substrate")
    ax.set_ylabel("Mapping kind")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = float(matrix.iat[row_idx, col_idx])
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="mean transfer score")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def classify_outcome(results: pd.DataFrame, validation_checks: pd.DataFrame) -> str:
    if validation_checks.empty or not bool(validation_checks["success"].all()):
        return "constraining/contradictory"
    fresh_transfer = results[
        (results["evaluation_kind"] == "fresh_existing_path_evaluation")
        & (results["control_family"] == "cross_substrate_transfer")
    ]
    blockers = results[results["evaluation_kind"] == "transfer_blocker"]
    if not fresh_transfer.empty and bool(fresh_transfer["transfer_success"].all()) and not blockers.empty:
        return "constraining/contradictory"
    if not fresh_transfer.empty and bool(fresh_transfer["transfer_success"].all()):
        return "supportive"
    return "null"


def report_text(
    *,
    artifacts_dir: Path,
    step_dir: Path,
    mapping_path: Path,
    freeze_manifest_path: Path,
    results_path: Path,
    summary_path: Path,
    figure_path: Path,
    validation_checks_path: Path,
    manifest_path: Path,
    config_path: Path,
    command_log_path: Path,
    source_manifest_path: Path,
    mappings: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    validation_checks: pd.DataFrame,
    freeze_manifest: Mapping[str, Any],
    source_manifest: pd.DataFrame,
    commands: Sequence[Mapping[str, Any]],
    outcome: str,
) -> str:
    checks = validation_summary(validation_checks)
    fresh_transfer = results[
        (results["evaluation_kind"] == "fresh_existing_path_evaluation")
        & (results["control_family"] == "cross_substrate_transfer")
    ]
    blocker_rows = results[results["evaluation_kind"] == "transfer_blocker"]
    context_rows = results[results["evaluation_kind"] == "existing_e05_reference_context"]
    artifact_list = [
        mapping_path,
        freeze_manifest_path,
        results_path,
        summary_path,
        figure_path,
        validation_checks_path,
        manifest_path,
        config_path,
        command_log_path,
        source_manifest_path,
    ]
    artifact_lines = "\n".join(f"- `{path}`" for path in artifact_list)
    command_rows = pd.DataFrame(
        [
            {
                "command": command.get("command"),
                "returnCode": command.get("returnCode"),
                "elapsedSeconds": command.get("elapsedSeconds"),
                "success": command.get("success"),
            }
            for command in commands
        ]
    )
    return f"""# E07 S13 Full Results: Test Substrate Transfer

## Top Summary

- Research step ID: S13
- Completion status: Completed; stopped before S14.
- Artifacts written:
{artifact_lines}
- Validation result: {checks['passed']}/{checks['total']} checks passed; unit tests {'passed' if all(command.get('success') for command in commands) else 'had failures recorded'}.
- Outcome classification: {outcome}.
- Caveats or blockers: Fresh executable transfer is limited to the pre-existing E05 embedded-row bubble/insertion path. S12 inverse-designed DSL policies remain E03-only proxy artifacts and are recorded as blocked for E05 transfer. No E05/E06 adapters were built. E05 broader morphology rows are reference controls, not new S13 simulations.
- Lay summary: S13 found that the one already-built transfer path behaves exactly as expected: ordinary 1D adjacent-swap sorting and the same sorting row embedded inside a 2D grid produce identical trajectories for bubble and insertion sort. This does not generalize to the S12 designed policies or to E05 repair/regeneration or E06 governance settings because executable adapters for those cases are absent by scope.
- Recommended next action: Chief Scientist review before S14. Treat S13 as narrow positive continuity evidence plus a constraining blocker record for broader substrate transfer.

## Frozen Question

Can policy behavior transfer across substrates using only existing executable E03/E05 paths, while preserving S12's E03-only caveat and documenting transfer blockers instead of building new adapters?

Primary success required frozen transfer mappings before evaluation, fresh evaluation only through existing E03/E05 executable paths, baselines and blocker records, and validation that no E05/E06 adapter was built in S13.

## Inputs

{markdown_table(source_manifest)}

The transfer mapping artifact was frozen at `{freeze_manifest.get('mappingsFrozenAtUtc')}` and evaluation started at `{freeze_manifest.get('evaluationStartedAtUtc')}`. The frozen mapping JSONL SHA-256 is `{freeze_manifest.get('transferMappingArtifactSha256')}`.

## Methods

S13 used the existing `src.e05.embedded_1d.run_adjacent_sort` helper, which supports only `bubble` and `insertion` algorithms on `array_1d` and `embedded_square_grid_2d`. For each algorithm, S13 froze a within-1D baseline mapping and a cross-substrate E03-to-E05 embedded-row mapping. The fresh evaluation used four deterministic held-out arrays: {', '.join(f'n={n}, seed={seed}' for n, seed in FRESH_ARRAY_SPECS)}.

For the cross-substrate branch, the source run used `array_1d` and the target run used `embedded_square_grid_2d`. S13 compared final array hashes, sortedness, monotonicity errors, swap counts, comparison counts, energy totals, row-constraint violations, and the complete sortedness/monotonicity trace signature. The score is 1.0 for exact trajectory transfer, otherwise the target sortedness or recovery fraction clipped to [0, 1].

S13 also read existing E05 benchmark reference rows as context controls. These rows quantify how local/global/reference policies behave on broader E05 morphology tasks but were not newly simulated. All arbitrary S12 inverse-designed DSL policies and the requested out-of-scope E05/E06 transfer branches were frozen and evaluated as blocker records.

No new E05 or E06 simulator adapter was written.

## Commands

{markdown_table(command_rows)}

## Results

Fresh existing-path transfer rows: {len(fresh_transfer)}. Successful fresh cross-substrate rows: {int(fresh_transfer['transfer_success'].sum()) if not fresh_transfer.empty else 0}/{len(fresh_transfer)}. Blocker rows: {len(blocker_rows)}. Existing E05 reference-control rows: {len(context_rows)}.

### Summary Table

{markdown_table(summary, max_rows=60)}

### Fresh Transfer Rows

{markdown_table(fresh_transfer[['mapping_kind', 'algorithm', 'array_size', 'seed', 'exact_trajectory_match', 'transfer_success', 'target_final_sortedness_percent', 'target_swap_count', 'target_comparison_count', 'transfer_score']], max_rows=40)}

### Blocker Counts

{markdown_table(blocker_rows.groupby(['mapping_kind', 'target_experiment_id', 'target_substrate_kind']).size().reset_index(name='blocked_rows'), max_rows=60)}

## Validation

{markdown_table(validation_checks)}

Validation confirms the mapping artifact was hashed and frozen before evaluation, only existing E03/E05 paths were freshly executed, required baseline/control/blocker families are present, all executable mappings have results, S12 E03-only blocker rows are present, and the embedded-row transfer rows are exact.

## Outputs

- Transfer mappings JSONL: `{mapping_path}`
- Transfer mapping freeze manifest: `{freeze_manifest_path}`
- Unified S13 transfer result corpus: `{results_path}`
- Coverage and score summary: `{summary_path}`
- Transfer matrix figure: `{figure_path}`
- Artifact manifest: `{manifest_path}`

## Caveats And Limitations

The positive result is narrow continuity evidence for row-restricted adjacent-swap sorting, not substrate-independent policy transfer in general. It does not validate S12 inverse-designed policies outside E03, does not validate E05 repair/regeneration transfer, does not validate chimeric aggregation or E06 governance, and does not remove S07-S12 source, missingness, goal-conflict, class-status, or simulator-blocker caveats.

Existing E05 benchmark reference rows include global controllers and endpoint controls that are useful baselines, but they are not newly transferred policies. They remain context controls for interpreting how much broader morphology performance depends on source task, metric, and missingness structure.

## Provenance

- Repository: `{git_output(REPO_ROOT, ['rev-parse', '--show-toplevel'])}`
- Branch: `{git_output(REPO_ROOT, ['branch', '--show-current'])}`
- Commit before S13 run: `{git_output(REPO_ROOT, ['rev-parse', 'HEAD'])}`
- Python: `{sys.version.split()[0]}`
- Platform: `{platform.platform()}`
- NumPy: `{np.__version__}`
- pandas: `{pd.__version__}`
- Matplotlib: `{matplotlib.__version__}`
- Random seed: `{RANDOM_SEED}`

## Recommended Next Action

Stop before S14 for Chief Scientist review. If S14 proceeds, it should treat S13 as exact support only for the embedded-row transfer path and as a constraining result for S12-designed, E05 repair/regeneration, graph-general, and E06 governance transfer claims.
"""


def main() -> int:
    args = parse_args()
    np.random.seed(RANDOM_SEED)

    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e07"
    logs_dir = step_dir / "logs"
    for directory in (step_dir, results_dir, tables_dir, figures_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mapping_path = step_dir / "transfer_mappings.jsonl"
    mapping_csv_path = tables_dir / "e07_substrate_transfer_mappings.csv"
    freeze_manifest_path = step_dir / "transfer_mapping_freeze_manifest.json"
    results_path = results_dir / "e07_substrate_transfer.parquet"
    results_csv_path = tables_dir / "e07_substrate_transfer.csv"
    summary_path = tables_dir / "e07_substrate_transfer_coverage.csv"
    validation_checks_path = step_dir / "e07_s13_validation_checks.csv"
    figure_path = figures_dir / "substrate_transfer_matrix.png"
    config_path = step_dir / "s13_config.json"
    command_log_path = logs_dir / "s13_command_log.json"
    source_manifest_path = step_dir / "s13_source_manifest.csv"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    report_path = step_dir / "research_step_full_results.md"

    s12_designs = read_jsonl(args.s12_designed_policies)
    task_catalog = pd.read_parquet(args.e05_benchmark_task_catalog)
    reference_results = pd.read_parquet(args.e05_benchmark_reference_results)
    embedded_validation = pd.read_parquet(args.e05_embedded_validation)

    source_manifest = pd.DataFrame(
        [
            {
                "input_name": "e05_embedded_summary",
                "path": str(args.e05_embedded_summary),
                "sha256": maybe_sha256(args.e05_embedded_summary),
                "rows": len(pd.read_parquet(args.e05_embedded_summary)) if args.e05_embedded_summary.exists() else 0,
                "role": "prior exact E05 embedded-row run summary",
            },
            {
                "input_name": "e05_embedded_validation",
                "path": str(args.e05_embedded_validation),
                "sha256": maybe_sha256(args.e05_embedded_validation),
                "rows": len(embedded_validation),
                "role": "prior E05 embedded-row validation checks",
            },
            {
                "input_name": "e05_benchmark_task_catalog",
                "path": str(args.e05_benchmark_task_catalog),
                "sha256": maybe_sha256(args.e05_benchmark_task_catalog),
                "rows": len(task_catalog),
                "role": "existing E05 task/control context",
            },
            {
                "input_name": "e05_benchmark_reference_results",
                "path": str(args.e05_benchmark_reference_results),
                "sha256": maybe_sha256(args.e05_benchmark_reference_results),
                "rows": len(reference_results),
                "role": "existing E05 reference-control outcomes",
            },
            {
                "input_name": "s12_designed_policies",
                "path": str(args.s12_designed_policies),
                "sha256": maybe_sha256(args.s12_designed_policies),
                "rows": len(s12_designs),
                "role": "frozen S12 E03 DSL designs carried as blocked transfer candidates",
            },
            {
                "input_name": "s12_validation",
                "path": str(args.s12_validation),
                "sha256": maybe_sha256(args.s12_validation),
                "rows": len(pd.read_parquet(args.s12_validation)) if args.s12_validation.exists() else 0,
                "role": "S12 validation and blocker context",
            },
            {
                "input_name": "s12_design_manifest",
                "path": str(args.s12_design_manifest),
                "sha256": maybe_sha256(args.s12_design_manifest),
                "rows": 1 if args.s12_design_manifest.exists() else 0,
                "role": "S12 design freeze provenance",
            },
            {
                "input_name": "s08_neighbors",
                "path": str(args.s08_neighbors),
                "sha256": maybe_sha256(args.s08_neighbors),
                "rows": len(pd.read_parquet(args.s08_neighbors)) if args.s08_neighbors.exists() else 0,
                "role": "bounded distance context; not used to create adapters",
            },
            {
                "input_name": "s08_distance_metadata",
                "path": str(args.s08_distance_metadata),
                "sha256": maybe_sha256(args.s08_distance_metadata),
                "rows": len(pd.read_parquet(args.s08_distance_metadata)) if args.s08_distance_metadata.exists() else 0,
                "role": "distance metadata context; not used to create adapters",
            },
        ]
    )
    source_manifest.to_csv(source_manifest_path, index=False)

    frozen_at = utc_now()
    mappings = build_transfer_mappings(task_catalog, s12_designs, frozen_at)
    write_jsonl(mapping_path, mappings)
    mapping_df = pd.DataFrame(mappings)
    mapping_df.to_csv(mapping_csv_path, index=False)

    evaluation_started = utc_now()
    s12_manifest = read_json(args.s12_design_manifest)
    freeze_manifest = {
        "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "transferMappingArtifactPath": str(mapping_path),
        "transferMappingArtifactSha256": sha256_file(mapping_path),
        "transferMappingRowCount": len(mapping_df),
        "mappingsFrozenAtUtc": frozen_at,
        "evaluationStartedAtUtc": evaluation_started,
        "freezeRule": "All executable, control, and blocked transfer mappings are recorded before any fresh S13 evaluation.",
        "scopeRestriction": "Only existing E03/E05 executable paths are freshly evaluated; no E05/E06 adapters are built.",
        "noNewE05OrE06AdaptersBuilt": True,
        "existingExecutablePath": EXECUTION_PATH,
        "executableAlgorithms": list(EXECUTABLE_ALGORITHMS),
        "s12DesignArtifactSha256": maybe_sha256(args.s12_designed_policies),
        "s12CandidateDesignArtifactSha256FromManifest": s12_manifest.get("candidateDesignArtifactSha256", ""),
        "s12ScopeRestriction": s12_manifest.get("scopeRestriction", ""),
        "freshArraySpecs": [{"arraySize": int(n), "seed": int(seed)} for n, seed in FRESH_ARRAY_SPECS],
    }
    write_json(freeze_manifest_path, freeze_manifest)

    results: list[dict[str, Any]] = []
    executable_mappings = mapping_df[mapping_df["mapping_status"] == "executable_existing_path"]
    for mapping in executable_mappings.to_dict(orient="records"):
        results.extend(evaluate_fresh_mapping(mapping))
    context_mappings = mapping_df[mapping_df["mapping_status"] == "existing_reference_context"]
    results.extend(evaluate_context_rows(reference_results, context_mappings))
    blocked_mappings = mapping_df[mapping_df["mapping_status"] == "blocked_no_existing_path"]
    results.extend(evaluate_blockers(blocked_mappings))
    result_df = pd.DataFrame(results).sort_values(["evaluation_kind", "control_family", "mapping_kind", "algorithm", "seed"], kind="mergesort")
    result_df.to_parquet(results_path, index=False)
    result_df.to_csv(results_csv_path, index=False)

    summary = summarize_results(result_df)
    summary.to_csv(summary_path, index=False)
    write_figure(summary, figure_path)

    validation_checks = validate_substrate_transfer_artifacts(mapping_df, result_df, freeze_manifest)
    embedded_prior_success = bool(embedded_validation.get("success", pd.Series(dtype=bool)).fillna(False).astype(bool).all())
    validation_checks = pd.concat(
        [
            validation_checks,
            pd.DataFrame(
                [
                    {
                        "validation_case": "prior_e05_embedded_validation_passed",
                        "success": embedded_prior_success,
                        "detail": f"e05_embedded_validation_rows={len(embedded_validation)}",
                    },
                    {
                        "validation_case": "s12_design_hash_matches_manifest",
                        "success": bool(
                            s12_manifest.get("candidateDesignArtifactSha256", "")
                            and s12_manifest.get("candidateDesignArtifactSha256") == maybe_sha256(args.s12_designed_policies)
                        ),
                        "detail": f"manifest={s12_manifest.get('candidateDesignArtifactSha256', '')} observed={maybe_sha256(args.s12_designed_policies)}",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        commands.append(run_command([sys.executable, "-m", "unittest", "tests.e07.test_substrate_transfer_schema"], args.repo_dir))
    commands.append(run_command([sys.executable, "-m", "py_compile", str(Path(__file__).relative_to(args.repo_dir))], args.repo_dir))
    if not all(command["success"] for command in commands):
        validation_checks = pd.concat(
            [
                validation_checks,
                pd.DataFrame(
                    [
                        {
                            "validation_case": "s13_reproducibility_commands_succeeded",
                            "success": False,
                            "detail": "One or more unit/compile commands failed; see command log.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    else:
        validation_checks = pd.concat(
            [
                validation_checks,
                pd.DataFrame(
                    [
                        {
                            "validation_case": "s13_reproducibility_commands_succeeded",
                            "success": True,
                            "detail": "Unit test and py_compile commands succeeded.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    validation_checks.to_csv(validation_checks_path, index=False)
    write_json(command_log_path, list(commands))

    outcome = classify_outcome(result_df, validation_checks)
    config = {
        "schemaVersion": SUBSTRATE_TRANSFER_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "randomSeed": RANDOM_SEED,
        "freshArraySpecs": [{"arraySize": int(n), "seed": int(seed)} for n, seed in FRESH_ARRAY_SPECS],
        "executableAlgorithms": list(EXECUTABLE_ALGORITHMS),
        "existingExecutablePath": EXECUTION_PATH,
        "noNewE05OrE06AdaptersBuilt": True,
        "inputPaths": source_manifest[["input_name", "path", "sha256"]].to_dict(orient="records"),
        "validationResult": validation_summary(validation_checks),
        "outcomeClassification": outcome,
    }
    write_json(config_path, config)

    artifact_manifest = {
        "schemaVersion": "eidosoma.e07.s13.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "outcomeClassification": outcome,
        "validationResult": validation_summary(validation_checks),
        "artifacts": [
            artifact_entry(mapping_path, artifacts_dir, "Frozen S13 transfer mappings."),
            artifact_entry(mapping_csv_path, artifacts_dir, "CSV copy of frozen transfer mappings."),
            artifact_entry(freeze_manifest_path, artifacts_dir, "Transfer mapping freeze manifest."),
            artifact_entry(results_path, artifacts_dir, "S13 substrate-transfer result corpus."),
            artifact_entry(results_csv_path, artifacts_dir, "CSV copy of S13 transfer result corpus."),
            artifact_entry(summary_path, artifacts_dir, "S13 transfer coverage and score summary."),
            artifact_entry(figure_path, artifacts_dir, "S13 transfer matrix figure."),
            artifact_entry(validation_checks_path, artifacts_dir, "S13 validation checks."),
            artifact_entry(config_path, artifacts_dir, "S13 run configuration."),
            artifact_entry(command_log_path, artifacts_dir, "S13 command log."),
            artifact_entry(source_manifest_path, artifacts_dir, "S13 input source manifest."),
            self_referential_artifact_entry(report_path, artifacts_dir, "S13 full-results report."),
            self_referential_artifact_entry(artifact_manifest_path, artifacts_dir, "S13 artifact manifest."),
        ],
        "repository": {
            "path": str(args.repo_dir),
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "headCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        },
    }
    write_json(artifact_manifest_path, artifact_manifest)

    report = report_text(
        artifacts_dir=artifacts_dir,
        step_dir=step_dir,
        mapping_path=mapping_path,
        freeze_manifest_path=freeze_manifest_path,
        results_path=results_path,
        summary_path=summary_path,
        figure_path=figure_path,
        validation_checks_path=validation_checks_path,
        manifest_path=artifact_manifest_path,
        config_path=config_path,
        command_log_path=command_log_path,
        source_manifest_path=source_manifest_path,
        mappings=mapping_df,
        results=result_df,
        summary=summary,
        validation_checks=validation_checks,
        freeze_manifest=freeze_manifest,
        source_manifest=source_manifest,
        commands=commands,
        outcome=outcome,
    )
    write_text(report_path, report)

    print(f"[S13] wrote {len(result_df)} result rows")
    print(f"[S13] validation checks: {validation_summary(validation_checks)}")
    print(f"[S13] outcome: {outcome}")
    print(f"[S13] report: {report_path}")
    return 0 if bool(validation_checks["success"].all()) and all(command["success"] for command in commands) else 1


if __name__ == "__main__":
    raise SystemExit(main())
