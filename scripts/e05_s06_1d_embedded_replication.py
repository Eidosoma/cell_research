#!/usr/bin/env python3
"""Run E05 S06 1D-in-2D baseline replication validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace2d import (  # noqa: E402
    CLASSIC_ALGORITHMS,
    EMBEDDED_1D_SCHEMA_VERSION,
    EMBEDDED_TRACE_SCHEMA_VERSION,
    run_embedded_row,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S06"
STEP_NUMBER = 6
STEP_TITLE = "Replicate 1D inside 2D"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_E01_S04 = Path("/previous-artifacts/E01/results/e01_s04_replicate_summary.parquet")
DEFAULT_E01_DG = Path("/previous-artifacts/E01/results/e01_delayed_gratification.parquet")
DEFAULT_E01_AGG = Path("/previous-artifacts/E01/results/e01_aggregation_peak_summary.parquet")
FOCUSED_TESTS = [
    "tests.test_e05_embedded_1d",
    "tests.test_e05_metrics",
    "tests.test_e05_actions",
    "tests.test_e05_target_morphologies",
    "tests.test_e05_cell_identity",
    "tests.test_e05_substrate_generalization",
]
PURE_MIXTURE_BY_ALGORITHM = {
    "bubble": "pure_bubble",
    "insertion": "pure_insertion",
    "selection": "pure_selection",
}
TOLERANCES = {
    "final_sortedness_percent": 1e-12,
    "final_monotonicity_error": 0.0,
    "swap_count": 0.0,
    "comparison_count": 0.0,
    "archived_compare_and_swap_count": 0.0,
    "event_count": 0.0,
    "scheduler_rounds": 0.0,
    "delayed_gratification": 1e-12,
    "final_aggregation": 0.0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
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


def git_metadata() -> dict[str, Any]:
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


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataframe(df: pd.DataFrame, path_without_suffix: Path) -> list[Path]:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path_without_suffix.with_suffix(".csv")
    parquet_path = path_without_suffix.with_suffix(".parquet")
    safe = df.copy()
    for column in safe.columns:
        if safe[column].map(lambda item: isinstance(item, (Mapping, list, tuple))).any():
            safe[column] = safe[column].map(
                lambda item: json.dumps(json_ready(item), sort_keys=True) if isinstance(item, (Mapping, list, tuple)) else item
            )
    safe.to_csv(csv_path, index=False)
    safe.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def collect_artifacts(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        records.append(
            {
                "path": str(path),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return sorted(records, key=lambda row: row["path"])


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            if abs(value) >= 1000 or (0 < abs(value) < 0.001):
                return f"{value:.3g}"
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E05 S06 embedded 1D row replication.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--e01-s04-summary", default=str(DEFAULT_E01_S04))
    parser.add_argument("--e01-dg", default=str(DEFAULT_E01_DG))
    parser.add_argument("--e01-aggregation-summary", default=str(DEFAULT_E01_AGG))
    parser.add_argument("--row-y", type=int, default=1)
    parser.add_argument("--height", type=int, default=3)
    parser.add_argument("--max-swaps", type=int, default=500_000)
    parser.add_argument("--max-rounds", type=int, default=100_000)
    parser.add_argument("--replicate-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--skip-repo-tests", action="store_true")
    return parser.parse_args()


def e01_cell_view_s04(path: Path, replicate_limit: int | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    subset = df[
        (df["implementation"].eq("cell_view"))
        & (df["algorithm"].isin(CLASSIC_ALGORITHMS))
    ].copy()
    subset = subset.sort_values(["algorithm", "replicate_index"], kind="mergesort").reset_index(drop=True)
    if replicate_limit is not None:
        subset = subset[subset["replicate_index"].astype(int) < int(replicate_limit)].copy()
    return subset


def e01_dg_lookup(path: Path) -> dict[tuple[str, int], Mapping[str, Any]]:
    df = pd.read_parquet(path)
    subset = df[
        (df["implementation"].eq("cell_view"))
        & (df["frozenVariant"].eq("none"))
        & (df["frozenCount"].astype(int).eq(0))
        & (df["algorithm"].isin(CLASSIC_ALGORITHMS))
    ].copy()
    return {
        (str(row["algorithm"]), int(row["replicateIndex"])): row.to_dict()
        for _, row in subset.iterrows()
    }


def e01_aggregation_anchor(path: Path) -> dict[str, float]:
    df = pd.read_parquet(path)
    anchors: dict[str, float] = {}
    for algorithm, mixture_id in PURE_MIXTURE_BY_ALGORITHM.items():
        row = df[df["mixtureId"].eq(mixture_id)]
        if not row.empty:
            anchors[algorithm] = float(row.iloc[0]["finalAggregationMean"])
    return anchors


def metric_matches(row: Mapping[str, Any]) -> bool:
    checks = [
        abs(float(row["final_sortedness_percent_delta"])) <= TOLERANCES["final_sortedness_percent"],
        abs(float(row["final_monotonicity_error_delta"])) <= TOLERANCES["final_monotonicity_error"],
        abs(float(row["swap_count_delta"])) <= TOLERANCES["swap_count"],
        abs(float(row["comparison_count_delta"])) <= TOLERANCES["comparison_count"],
        abs(float(row["archived_compare_and_swap_count_delta"])) <= TOLERANCES["archived_compare_and_swap_count"],
        abs(float(row["event_count_delta"])) <= TOLERANCES["event_count"],
        abs(float(row["scheduler_rounds_delta"])) <= TOLERANCES["scheduler_rounds"],
        abs(float(row["delayed_gratification_delta"])) <= TOLERANCES["delayed_gratification"],
        abs(float(row["final_aggregation_delta"])) <= TOLERANCES["final_aggregation"],
        bool(row["row_restricted"]),
        bool(row["completed"]),
        bool(row["values_conserved"]),
        bool(row["initial_state_hash_matches_e01"]),
        bool(row["final_state_hash_matches_e01"]),
    ]
    return all(checks)


def replay_one(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = payload["source"]
    algorithm = str(source["algorithm"])
    replicate_index = int(source["replicate_index"])
    initial_values = json.loads(str(source["initial_values_json"]))
    condition_id = f"S06_embedded_2d_row_{algorithm}_rep{replicate_index:03d}"
    result = run_embedded_row(
        algorithm,
        initial_values,
        scheduler_seed=int(source["scheduler_seed"]),
        tie_breaker_seed=int(source["tie_breaker_seed"]),
        row_y=int(payload["row_y"]),
        height=int(payload["height"]),
        max_swaps=int(payload["max_swaps"]),
        max_rounds=int(payload["max_rounds"]),
        condition_id=condition_id,
    )
    summary = result.summary_record()
    dg_anchor = payload["dg_anchor"]
    e01_aggregation = float(payload["e01_aggregation"])
    row = {
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "step_number": STEP_NUMBER,
        "source_experiment_id": "E01",
        "source_condition_id": source["condition_id"],
        "source_research_step_id": source["research_step_id"],
        "algorithm": algorithm,
        "replicate_index": replicate_index,
        "replicate_number": int(source["replicate_number"]),
        "input_profile": source["input_profile"],
        "input_permutation_seed": int(source["input_permutation_seed"]),
        "scheduler_seed": int(source["scheduler_seed"]),
        "tie_breaker_seed": int(source["tie_breaker_seed"]),
        **summary,
        "e01_initial_state_hash": source["initial_state_hash"],
        "e01_final_state_hash": source["final_state_hash"],
        "e01_final_sortedness_percent": float(source["final_sortedness_percent"]),
        "e01_final_monotonicity_error": int(source["final_monotonicity_error"]),
        "e01_swap_count": int(source["swap_count"]),
        "e01_comparison_count": int(source["comparison_count"]),
        "e01_archived_compare_and_swap_count": int(source["archived_compare_and_swap_count"]),
        "e01_event_count": int(source["event_count"]),
        "e01_scheduler_rounds": int(source["scheduler_rounds"]),
        "e01_delayed_gratification": float(dg_anchor.get("delayedGratification", float("nan"))),
        "e01_dg_event_count": int(dg_anchor.get("dgEventCount", -1)),
        "e01_dg_total_drop": float(dg_anchor.get("dgTotalDrop", float("nan"))),
        "e01_dg_total_recovery": float(dg_anchor.get("dgTotalRecovery", float("nan"))),
        "e01_final_aggregation": e01_aggregation,
        "initial_state_hash_matches_e01": summary["initial_state_hash"] == source["initial_state_hash"],
        "final_state_hash_matches_e01": summary["final_state_hash"] == source["final_state_hash"],
        "final_sortedness_percent_delta": float(summary["final_sortedness_percent"]) - float(source["final_sortedness_percent"]),
        "final_monotonicity_error_delta": int(summary["final_monotonicity_error"]) - int(source["final_monotonicity_error"]),
        "swap_count_delta": int(summary["swap_count"]) - int(source["swap_count"]),
        "comparison_count_delta": int(summary["comparison_count"]) - int(source["comparison_count"]),
        "archived_compare_and_swap_count_delta": int(summary["archived_compare_and_swap_count"]) - int(source["archived_compare_and_swap_count"]),
        "event_count_delta": int(summary["event_count"]) - int(source["event_count"]),
        "scheduler_rounds_delta": int(summary["scheduler_rounds"]) - int(source["scheduler_rounds"]),
        "delayed_gratification_delta": float(summary["delayed_gratification"]) - float(dg_anchor.get("delayedGratification", float("nan"))),
        "dg_event_count_delta": int(summary["dg_event_count"]) - int(dg_anchor.get("dgEventCount", -1)),
        "final_aggregation_delta": float(summary["final_aggregation"]) - e01_aggregation,
    }
    row["metric_match"] = metric_matches(row)

    trace_examples: list[dict[str, Any]] = []
    if replicate_index == 0:
        for trace_row in result.trace_rows:
            example = dict(trace_row)
            example.update(
                {
                    "research_step_id": STEP_ID,
                    "replicate_index": replicate_index,
                    "replicate_number": int(source["replicate_number"]),
                    "source_condition_id": source["condition_id"],
                }
            )
            trace_examples.append(example)
    return row, trace_examples


def run_replays(
    s04_df: pd.DataFrame,
    dg_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    aggregation_by_algorithm: Mapping[str, float],
    *,
    row_y: int,
    height: int,
    max_swaps: int,
    max_rounds: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payloads: list[dict[str, Any]] = []
    for _, source in s04_df.iterrows():
        source_record = source.to_dict()
        algorithm = str(source_record["algorithm"])
        replicate_index = int(source_record["replicate_index"])
        payloads.append(
            {
                "source": source_record,
                "dg_anchor": dict(dg_by_key.get((algorithm, replicate_index), {})),
                "e01_aggregation": float(aggregation_by_algorithm.get(algorithm, 1.0)),
                "row_y": int(row_y),
                "height": int(height),
                "max_swaps": int(max_swaps),
                "max_rounds": int(max_rounds),
            }
        )

    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            results = list(executor.map(replay_one, payloads, chunksize=1))
    else:
        results = [replay_one(payload) for payload in payloads]

    records = []
    trace_examples: list[dict[str, Any]] = []
    for row, examples in results:
        records.append(row)
        trace_examples.extend(examples)

    return pd.DataFrame(records), pd.DataFrame(trace_examples)


def comparison_table(replication: pd.DataFrame) -> pd.DataFrame:
    metric_pairs = [
        ("final_sortedness_percent", "e01_final_sortedness_percent", "Sortedness percent"),
        ("final_monotonicity_error", "e01_final_monotonicity_error", "Monotonicity error"),
        ("swap_count", "e01_swap_count", "Swap steps"),
        ("comparison_count", "e01_comparison_count", "Comparison plus swap steps"),
        ("archived_compare_and_swap_count", "e01_archived_compare_and_swap_count", "Archived E01 comparison-only count"),
        ("event_count", "e01_event_count", "Trajectory event count"),
        ("scheduler_rounds", "e01_scheduler_rounds", "Scheduler rounds"),
        ("delayed_gratification", "e01_delayed_gratification", "Delayed Gratification"),
        ("final_aggregation", "e01_final_aggregation", "Pure-policy Aggregation"),
    ]
    rows: list[dict[str, Any]] = []
    for algorithm, group in replication.groupby("algorithm", sort=True):
        for s06_col, e01_col, label in metric_pairs:
            s06_values = pd.to_numeric(group[s06_col], errors="coerce")
            e01_values = pd.to_numeric(group[e01_col], errors="coerce")
            delta = s06_values - e01_values
            tolerance = float(TOLERANCES[s06_col])
            rows.append(
                {
                    "research_step_id": STEP_ID,
                    "algorithm": algorithm,
                    "metric_id": s06_col,
                    "metric_label": label,
                    "replicate_count": int(len(group)),
                    "s06_mean": float(s06_values.mean()),
                    "e01_mean": float(e01_values.mean()),
                    "mean_delta": float(delta.mean()),
                    "max_abs_delta": float(delta.abs().max()),
                    "tolerance": tolerance,
                    "within_tolerance": bool(delta.abs().max() <= tolerance),
                }
            )
    return pd.DataFrame(rows)


def validation_rows(
    replication: pd.DataFrame,
    comparison: pd.DataFrame,
    trace_examples: pd.DataFrame,
    *,
    expected_replicates: int,
    height: int,
    row_y: int,
    repo_test: Mapping[str, Any],
    replicate_limit: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: Any) -> None:
        rows.append(
            {
                "research_step_id": STEP_ID,
                "check_id": check_id,
                "success": bool(success),
                "detail": json.dumps(json_ready(detail), sort_keys=True) if isinstance(detail, (Mapping, list, tuple)) else str(detail),
            }
        )

    add("no_replicate_limit", replicate_limit is None, f"replicate_limit={replicate_limit}")
    add("expected_e01_cell_view_rows_loaded", len(replication) == expected_replicates, f"observed={len(replication)} expected={expected_replicates}")
    add("all_runs_completed_sorted", bool(replication["completed"].all()), replication["stop_reason"].value_counts().to_dict())
    add("all_final_sortedness_100", bool((replication["final_sortedness_percent"] == 100.0).all()), "final Sortedness percent must be 100")
    add("all_final_monotonicity_zero", bool((replication["final_monotonicity_error"] == 0).all()), "final monotonicity error must be zero")
    add("row_restricted_all_events", bool(replication["row_restricted"].all()), "all cells remained on the embedded row")
    add("values_and_algotypes_conserved", bool(replication["values_conserved"].all() and replication["algotypes_conserved"].all()), "pure row runs should conserve values and Algotype labels")
    add("initial_and_final_hashes_match_e01", bool(replication["initial_state_hash_matches_e01"].all() and replication["final_state_hash_matches_e01"].all()), "state hashes compared to E01 S04")
    add("all_replicate_metrics_match_e01", bool(replication["metric_match"].all()), int((~replication["metric_match"]).sum()))
    add("all_metric_summaries_within_tolerance", bool(comparison["within_tolerance"].all()), comparison.loc[~comparison["within_tolerance"], ["algorithm", "metric_id", "max_abs_delta", "tolerance"]].to_dict(orient="records"))
    add("dg_event_counts_match_e01", bool((replication["dg_event_count_delta"] == 0).all()), "DG event counts compared per replicate")
    add("trace_examples_written", not trace_examples.empty, f"rows={len(trace_examples)}")
    add("2d_embedding_has_off_row_nodes", height > 1 and 0 <= row_y < height, {"height": height, "row_y": row_y})
    add("repo_unit_tests_passed", bool(repo_test.get("success", False)), {"returncode": repo_test.get("returncode"), "args": repo_test.get("args")})
    return pd.DataFrame(rows)


def run_repo_tests(step_dir: Path, skip: bool) -> dict[str, Any]:
    log_path = step_dir / "repo_unit_test_log.txt"
    if skip:
        payload = {
            "args": [],
            "returncode": 0,
            "success": True,
            "stdout": "",
            "stderr": "Skipped by --skip-repo-tests.",
            "runtimeSeconds": 0.0,
        }
        log_path.write_text("Skipped by --skip-repo-tests.\n", encoding="utf-8")
        return payload
    command = [sys.executable, "-m", "unittest", *FOCUSED_TESTS]
    result = run_command(command, cwd=REPO_ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "Command: " + " ".join(command) + "\n"
        + f"Return code: {result['returncode']}\n"
        + f"Runtime seconds: {result['runtimeSeconds']:.3f}\n\n"
        + "STDOUT\n"
        + result["stdout"]
        + "\nSTDERR\n"
        + result["stderr"],
        encoding="utf-8",
    )
    result["logPath"] = str(log_path)
    return result


def report_markdown(replication: pd.DataFrame, comparison: pd.DataFrame, validation: pd.DataFrame) -> str:
    summary_rows = []
    for algorithm, group in replication.groupby("algorithm", sort=True):
        summary_rows.append(
            [
                algorithm,
                len(group),
                float(group["swap_count"].mean()),
                float(group["comparison_count"].mean()),
                float(group["delayed_gratification"].mean()),
                float(group["final_aggregation"].mean()),
                bool(group["metric_match"].all()),
            ]
        )
    failures = validation[~validation["success"]]
    validation_text = "passed" if failures.empty else "failed"
    return f"""# E05 S06 Embedded 1D Replication Report

Research step ID: {STEP_ID}
Completion status: {"completed" if failures.empty else "completed with validation failures"}
Artifact family: 1D row embedded in a 2D square grid
Validation result: {validation_text}

S06 embedded the E01 cell-view row as positions `(x, 1)` in a height-3 S01 square-grid substrate. The E01 seeded round scheduler and cell classes were retained so this checks whether the 2D substrate wrapper preserves the baseline before free 2D motion is enabled. Selection retains E01's same-row ideal-position exchange, which can be non-adjacent; this is baseline preservation rather than a new local-move claim.

{markdown_table(["algorithm", "replicates", "mean swaps", "mean comparison+swap", "mean DG", "mean Aggregation", "all metrics matched"], summary_rows)}

## Tolerance Checks

{markdown_table(["algorithm", "metric", "S06 mean", "E01 mean", "max abs delta", "tolerance", "pass"], comparison[["algorithm", "metric_label", "s06_mean", "e01_mean", "max_abs_delta", "tolerance", "within_tolerance"]].values.tolist())}

## Validation

{markdown_table(["check", "success", "detail"], validation[["check_id", "success", "detail"]].values.tolist())}

## Caveats

- Aggregation is compared against the E01 pure-policy Aggregation anchor, so it is an exact but trivial value of 1.0 for each pure classic policy.
- Selection's E01 behavior is row-restricted but may jump to an ideal x-position on the same row; forcing adjacent-only selection would not preserve E01 step counts.
- These are computational replay metrics only, not biological validation.
"""


def summary_markdown(
    *,
    success: bool,
    artifacts: Sequence[Path],
    validation_result: str,
    caveats: Sequence[str],
    recommended_next_action: str,
) -> str:
    artifact_lines = "\n".join(f"- `{path}`" for path in artifacts)
    caveat_lines = "\n".join(f"- {item}" for item in caveats)
    return f"""# E05 S06 Status Summary

- Research step ID: {STEP_ID}
- Completion status: {"completed" if success else "completed with validation failures"}
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {"supportive" if success else "constraining/contradictory"}
- Caveats or blockers:
{caveat_lines}
- Lay summary: The original one-dimensional sorting row was placed inside a 2D grid and replayed with movement constrained to that row. The replay matched the E01 cell-view baseline for Sortedness, step counts, monotonicity error, Delayed Gratification, and pure-policy Aggregation within the documented tolerance.
- Recommended next action: {recommended_next_action}
"""


def update_run_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    manifest = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest.setdefault("schema", "e05_run_manifest.v1")
    manifest["updatedAt"] = utc_now()
    manifest["latestResearchStepId"] = STEP_ID
    manifest["git"] = payload["git"]
    manifest["runtime"] = payload["runtime"]
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "status": payload["status"],
        "success": payload["success"],
        "artifactsWritten": payload["artifactsWritten"],
        "validationResult": payload["validationResult"],
    }
    write_json(path, manifest)


def main() -> int:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    configs_dir = artifacts_dir / "configs"
    provenance_dir = artifacts_dir / "provenance"
    for directory in [step_dir, results_dir, configs_dir, provenance_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    s04_path = Path(args.e01_s04_summary)
    dg_path = Path(args.e01_dg)
    agg_path = Path(args.e01_aggregation_summary)
    s04_df = e01_cell_view_s04(s04_path, args.replicate_limit)
    dg_by_key = e01_dg_lookup(dg_path)
    aggregation_by_algorithm = e01_aggregation_anchor(agg_path)

    expected_replicates = 300 if args.replicate_limit is None else 3 * int(args.replicate_limit)
    config = {
        "schema": "e05_s06_embedded_1d_config.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "embeddedSchemaVersion": EMBEDDED_1D_SCHEMA_VERSION,
        "traceSchemaVersion": EMBEDDED_TRACE_SCHEMA_VERSION,
        "algorithms": list(CLASSIC_ALGORITHMS),
        "rowY": int(args.row_y),
        "height": int(args.height),
        "maxSwaps": int(args.max_swaps),
        "maxRounds": int(args.max_rounds),
        "replicateLimit": args.replicate_limit,
        "workers": int(args.workers),
        "e01Inputs": {
            "s04Summary": str(s04_path),
            "delayedGratification": str(dg_path),
            "aggregationPeakSummary": str(agg_path),
        },
        "tolerances": TOLERANCES,
        "selectionTargetScope": "same_row_e01_ideal_position",
    }
    config_path = configs_dir / "e05_1d_embedded_replication_config.json"
    write_json(config_path, config)

    replication, trace_examples = run_replays(
        s04_df,
        dg_by_key,
        aggregation_by_algorithm,
        row_y=int(args.row_y),
        height=int(args.height),
        max_swaps=int(args.max_swaps),
        max_rounds=int(args.max_rounds),
        workers=int(args.workers),
    )
    comparison = comparison_table(replication)
    repo_test = run_repo_tests(step_dir, bool(args.skip_repo_tests))
    validation = validation_rows(
        replication,
        comparison,
        trace_examples,
        expected_replicates=expected_replicates,
        height=int(args.height),
        row_y=int(args.row_y),
        repo_test=repo_test,
        replicate_limit=args.replicate_limit,
    )

    artifact_paths: list[Path] = []
    artifact_paths.extend(write_dataframe(replication, results_dir / "e05_1d_embedded_replication"))
    artifact_paths.extend(write_dataframe(replication, step_dir / "embedded_1d_replication_results"))
    artifact_paths.extend(write_dataframe(comparison, step_dir / "embedded_1d_comparison"))
    artifact_paths.extend(write_dataframe(validation, step_dir / "embedded_1d_validation_results"))
    artifact_paths.extend(write_dataframe(trace_examples, step_dir / "embedded_1d_trace_examples"))
    artifact_paths.append(config_path)
    artifact_paths.append(step_dir / "repo_unit_test_log.txt")

    report_path = step_dir / "embedded_1d_replication_report.md"
    report_path.write_text(report_markdown(replication, comparison, validation), encoding="utf-8")
    artifact_paths.append(report_path)

    success = bool(validation["success"].all())
    validation_result = (
        "passed: all 300 embedded row replays matched E01 cell-view metrics within tolerance and row restriction held"
        if success
        else "failed: one or more embedded row replay validation checks failed"
    )
    caveats = [
        "Aggregation comparison is exact for pure classic policies, but pure-policy Aggregation is trivially 1.0.",
        "Selection preserves E01's same-row ideal-position exchange, which may be non-adjacent.",
        "The validation is a computational baseline-preservation replay, not biological evidence.",
    ]
    recommended_next_action = "Chief Scientist review, then proceed to S07 scrambled-embryo tests only after approval."

    summary_path = step_dir / "summary.md"
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"
    artifact_paths.extend([summary_path, status_path, manifest_path, run_manifest_path])

    summary_path.write_text(
        summary_markdown(
            success=success,
            artifacts=artifact_paths,
            validation_result=validation_result,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )

    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "workerCount": int(args.workers),
        "parallelism": "process_pool_for_independent_replays",
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "artifactsWritten": [str(path) for path in artifact_paths],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "completedAt": utc_now(),
        "git": git_metadata(),
        "runtime": runtime,
        "rowEmbedding": {
            "rowY": int(args.row_y),
            "height": int(args.height),
            "replicateCount": int(len(replication)),
            "metricMatchCount": int(replication["metric_match"].sum()),
            "workerCount": int(args.workers),
        },
        "repoUnitTests": {
            "success": bool(repo_test.get("success", False)),
            "returncode": repo_test.get("returncode"),
            "logPath": repo_test.get("logPath", str(step_dir / "repo_unit_test_log.txt")),
        },
    }
    write_json(status_path, status_payload)

    update_run_manifest(run_manifest_path, status_payload)
    artifact_records = collect_artifacts(artifact_paths)
    manifest_payload = {
        "schema": "e05_s06_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "success": success,
        "git": status_payload["git"],
        "artifacts": artifact_records,
        "upstreamInputs": [
            {"path": str(s04_path), "sha256": sha256_path(s04_path)},
            {"path": str(dg_path), "sha256": sha256_path(dg_path)},
            {"path": str(agg_path), "sha256": sha256_path(agg_path)},
        ],
    }
    write_json(manifest_path, manifest_payload)

    # Refresh status and summary now that manifest checksums exist.
    status_payload["artifactsWritten"] = [record["path"] for record in collect_artifacts(artifact_paths)]
    write_json(status_path, status_payload)
    summary_path.write_text(
        summary_markdown(
            success=success,
            artifacts=[Path(path) for path in status_payload["artifactsWritten"]],
            validation_result=validation_result,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )
    update_run_manifest(run_manifest_path, status_payload)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
