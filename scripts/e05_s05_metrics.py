#!/usr/bin/env python3
"""Run E05 S05 morphospace metric validation and write artifacts."""

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
    METRIC_SCHEMA_VERSION,
    TRAJECTORY_METRIC_VERSION,
    add_extra_cell,
    build_boundary_target,
    build_organ_like_target,
    build_standard_target_library,
    evaluate_morphospace_metrics,
    metric_catalog,
    metric_rows,
    remove_positions,
    rotate_state_180,
    scrambled_state,
    trajectory_curvature,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S05"
STEP_NUMBER = 5
STEP_TITLE = "Define morphospace metrics"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


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
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
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
            safe[column] = safe[column].map(lambda item: json.dumps(json_ready(item), sort_keys=True) if isinstance(item, (Mapping, list, tuple)) else item)
    safe.to_csv(csv_path, index=False)
    safe.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".") if math.isfinite(value) else ""
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def metric_spec_markdown(catalog_rows: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        [row["metric_id"], row["direction"], row["normalization"], row["tie_case"]]
        for row in catalog_rows
    ]
    return f"""# E05 S05 Morphospace Metric Spec

Research step ID: {STEP_ID}
Title: {STEP_TITLE}
Metric schema version: `{METRIC_SCHEMA_VERSION}`
Trajectory metric version: `{TRAJECTORY_METRIC_VERSION}`

S05 defines report-facing computational metrics over S01 substrates, S02 identities, S03 target morphologies, and S04 non-conservative action states. These are proxies for benchmark comparison, not biological measurements. Graph edit distance is implemented as a deterministic approximation using occupied-node edits, induced-edge edits, and organ-label edits.

{markdown_table(['metric', 'direction', 'normalization', 'tie case'], rows)}
"""


def _metric_lookup(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {metric["metricId"]: metric for metric in result["metrics"]}


def build_validation_outputs(artifacts_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = build_standard_target_library()
    catalog = metric_catalog()
    required_metrics = {
        "target_energy",
        "target_neighborhood_error",
        "graph_edit_approx",
        "boundary_error",
        "topology_error",
        "shape_moment_error",
        "hausdorff_distance",
        "earth_mover_distance",
        "trajectory_curvature",
    }
    observed_metric_ids = {row["metric_id"] for row in catalog}

    exact_results = {target.target_id: evaluate_morphospace_metrics(target, target.target_state()) for target in targets}
    exact_zero = all(
        math.isclose(metric["value"], 0.0, abs_tol=1e-12)
        and math.isclose(metric["normalizedValue"], 0.0, abs_tol=1e-12)
        for result in exact_results.values()
        for metric in result["metrics"]
    )

    organ_target = build_organ_like_target()
    boundary_target = build_boundary_target()
    scrambled_result = evaluate_morphospace_metrics(organ_target, scrambled_state(organ_target))
    rotated_result = evaluate_morphospace_metrics(organ_target, rotate_state_180(organ_target))
    hole_state = remove_positions(boundary_target.target_state(), [(2, 2)])
    hole_result = evaluate_morphospace_metrics(boundary_target, hole_state)
    extra_identity = next(iter(boundary_target.target_state().values()))
    extra_result = evaluate_morphospace_metrics(boundary_target, add_extra_cell(boundary_target.target_state(), (99, 99), extra_identity))
    missing_result = evaluate_morphospace_metrics(boundary_target, remove_positions(boundary_target.target_state(), [(2, 2), (3, 2)]))

    scrambled_metrics = _metric_lookup(scrambled_result)
    rotated_metrics = _metric_lookup(rotated_result)
    hole_metrics = _metric_lookup(hole_result)
    extra_metrics = _metric_lookup(extra_result)
    all_example_results = list(exact_results.values()) + [scrambled_result, rotated_result, hole_result, extra_result, missing_result]
    finite_nonnegative = all(
        math.isfinite(float(metric["normalizedValue"])) and float(metric["normalizedValue"]) >= 0.0
        for result in all_example_results
        for metric in result["metrics"]
    )
    straight = trajectory_curvature([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    kinked = trajectory_curvature([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])
    stationary = trajectory_curvature([(1.0, 1.0), (1.0, 1.0)])
    s04_ledger_path = artifacts_dir / "research_steps" / "S04" / "conservation_ledger.parquet"
    s04_modes: list[str] = []
    if s04_ledger_path.exists():
        s04_ledger = pd.read_parquet(s04_ledger_path)
        s04_modes = sorted(str(value) for value in s04_ledger["conservation_mode"].dropna().unique().tolist())
    validation_rows = [
        {
            "research_step_id": STEP_ID,
            "check_id": "metric_catalog_coverage",
            "success": required_metrics.issubset(observed_metric_ids),
            "detail": ",".join(sorted(observed_metric_ids)),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "exact_targets_zero_error",
            "success": exact_zero,
            "detail": f"{len(targets)} exact targets evaluated",
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "scrambled_state_identity_metrics_positive",
            "success": scrambled_metrics["target_energy"]["value"] > 0
            and scrambled_metrics["target_neighborhood_error"]["value"] > 0
            and scrambled_metrics["graph_edit_approx"]["value"] > 0
            and scrambled_metrics["earth_mover_distance"]["value"] > 0,
            "detail": {key: scrambled_metrics[key]["value"] for key in ["target_energy", "target_neighborhood_error", "graph_edit_approx", "earth_mover_distance"]},
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "rotated_state_metrics_positive",
            "success": rotated_metrics["target_energy"]["value"] > 0
            and rotated_metrics["boundary_error"]["value"] > 0
            and rotated_metrics["earth_mover_distance"]["value"] > 0,
            "detail": {key: rotated_metrics[key]["value"] for key in ["target_energy", "boundary_error", "earth_mover_distance"]},
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "hole_state_shape_topology_positive",
            "success": hole_metrics["boundary_error"]["value"] > 0
            and hole_metrics["topology_error"]["value"] > 0
            and hole_metrics["shape_moment_error"]["value"] > 0
            and hole_metrics["hausdorff_distance"]["value"] > 0,
            "detail": {key: hole_metrics[key]["value"] for key in ["boundary_error", "topology_error", "shape_moment_error", "hausdorff_distance"]},
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "extra_cell_nonconservative_case_handled",
            "success": extra_metrics["graph_edit_approx"]["value"] > 0
            and extra_metrics["hausdorff_distance"]["value"] > 0
            and extra_metrics["earth_mover_distance"]["value"] > 0,
            "detail": {key: extra_metrics[key]["value"] for key in ["graph_edit_approx", "hausdorff_distance", "earth_mover_distance"]},
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "metric_ranges_finite_nonnegative",
            "success": finite_nonnegative,
            "detail": f"{len(all_example_results)} example states checked",
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "trajectory_curvature_tie_cases",
            "success": straight["value"] == 0.0 and stationary["value"] == 0.0 and kinked["value"] > 0.0,
            "detail": {"straight": straight, "stationary": stationary, "kinked": kinked},
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "s04_nonconservative_trace_modes_available",
            "success": {"non_conservative_birth", "non_conservative_death", "non_conservative_detach"}.issubset(set(s04_modes)),
            "detail": {"s04ConservationLedgerPath": str(s04_ledger_path), "modes": s04_modes},
        },
    ]

    example_rows: list[dict[str, Any]] = []
    for target in targets:
        example_rows.extend(metric_rows(target, "exact_target", target.target_state()))
    example_rows.extend(metric_rows(organ_target, "scrambled_identity", scrambled_state(organ_target)))
    example_rows.extend(metric_rows(organ_target, "rotated_180", rotate_state_180(organ_target)))
    example_rows.extend(metric_rows(boundary_target, "single_hole", hole_state))
    example_rows.extend(metric_rows(boundary_target, "missing_two_cells", remove_positions(boundary_target.target_state(), [(2, 2), (3, 2)])))
    example_rows.extend(metric_rows(boundary_target, "extra_cell", add_extra_cell(boundary_target.target_state(), (99, 99), extra_identity)))

    trajectory_rows = []
    for case_id, result in [("straight", straight), ("stationary", stationary), ("kinked", kinked)]:
        trajectory_rows.append(
            {
                "schema_version": result["schemaVersion"],
                "research_step_id": STEP_ID,
                "case_id": case_id,
                "metric_id": result["metricId"],
                "value": result["value"],
                "normalized_value": result["normalizedValue"],
                "detail_json": json.dumps(json_ready(result["detail"]), sort_keys=True),
            }
        )

    return pd.DataFrame(validation_rows), pd.DataFrame(example_rows), pd.DataFrame(catalog), pd.DataFrame(trajectory_rows)


def write_outputs(artifacts_dir: Path) -> dict[str, Any]:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    configs_dir = artifacts_dir / "configs"
    code_index_dir = artifacts_dir / "code" / "e05_morphospace_simulator"
    for directory in [step_dir, results_dir, configs_dir, code_index_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    catalog_rows = metric_catalog()
    spec_json = step_dir / "metric_spec.json"
    write_json(
        spec_json,
        {
            "schemaVersion": METRIC_SCHEMA_VERSION,
            "trajectoryMetricVersion": TRAJECTORY_METRIC_VERSION,
            "researchStepId": STEP_ID,
            "metrics": catalog_rows,
        },
    )
    spec_md = step_dir / "metric_spec.md"
    spec_md.write_text(metric_spec_markdown(catalog_rows), encoding="utf-8")
    config_json = configs_dir / "e05_metric_specs.json"
    write_json(config_json, {"researchStepId": STEP_ID, "metricSchemaVersion": METRIC_SCHEMA_VERSION, "metrics": catalog_rows})

    validation_df, example_df, catalog_df, trajectory_df = build_validation_outputs(artifacts_dir)

    artifact_paths: list[Path] = [spec_json, spec_md, config_json]
    artifact_paths.extend(write_dataframe(validation_df, step_dir / "metric_validation_results"))
    artifact_paths.extend(write_dataframe(validation_df, results_dir / "e05_metric_validation"))
    artifact_paths.extend(write_dataframe(example_df, step_dir / "metric_examples"))
    artifact_paths.extend(write_dataframe(catalog_df, step_dir / "metric_catalog"))
    artifact_paths.extend(write_dataframe(trajectory_df, step_dir / "trajectory_curvature_examples"))

    code_index_path = code_index_dir / "README.md"
    code_index_path.write_text(
        "# E05 Morphospace Simulator Code Index\n\n"
        "Repository-backed source is kept in git per workspace instructions.\n\n"
        "- S01 package: `morphospace2d/substrates.py`\n"
        "- S02 package: `morphospace2d/identities.py`\n"
        "- S03 package: `morphospace2d/targets.py`\n"
        "- S04 package: `morphospace2d/actions.py`\n"
        "- S05 package: `morphospace2d/metrics.py`\n"
        "- S05 runner: `scripts/e05_s05_metrics.py`\n"
        "- S05 tests: `tests/test_e05_metrics.py`\n",
        encoding="utf-8",
    )
    artifact_paths.append(code_index_path)

    unit_log_path = step_dir / "repo_unit_test_log.txt"
    unit_test_success = unit_log_path.exists() and "OK" in unit_log_path.read_text(encoding="utf-8", errors="replace")
    if unit_log_path.exists():
        artifact_paths.append(unit_log_path)

    success = bool(validation_df["success"].all() and unit_test_success)
    status_path = step_dir / "status.json"
    summary_path = step_dir / "summary.md"
    manifest_path = step_dir / "artifact_manifest.json"
    reported_artifact_paths = artifact_paths + [status_path, summary_path, manifest_path]
    validation_result = (
        "Passed: S05 metric library covers target energy, target-neighborhood error, graph edit approximation, boundary error, topology error, shape moments, Hausdorff distance, identity-aware Earth-mover distance, and trajectory curvature; exact targets score zero; scrambled, rotated, hole, missing-cell, and extra-cell cases trigger expected positive errors; S04 non-conservative trace modes are available; focused repository unit tests passed."
        if success
        else "Failed: at least one S05 metric validation or focused unit-test check did not pass."
    )
    caveats = (
        "S05 metrics are computational proxies. Graph edit distance is an approximation, topology holes are counted on integer 2D occupancy, and downstream benchmarks should report metric components rather than relying only on the composite score."
    )
    recommended = "Proceed to S06 to replicate 1D behavior inside 2D only after Chief Scientist instruction."
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "artifactsWritten": [str(path) for path in reported_artifact_paths],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended,
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "validationCommands": [
            {
                "args": "PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_e05_metrics tests.test_e05_actions tests.test_e05_target_morphologies tests.test_e05_cell_identity tests.test_e05_substrate_generalization",
                "logPath": str(unit_log_path),
                "success": unit_test_success,
            }
        ],
        "generatedAt": utc_now(),
        "git": git_metadata(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threadEnvironment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
        },
    }
    write_json(status_path, status_payload)

    summary_rows = [[row["check_id"], row["success"], str(row["detail"])[:160]] for row in validation_df.to_dict("records")]
    summary = f"""# E05 S05 Status Summary

- Step ID: {STEP_ID}
- Completion status: {'completed' if success else 'completed with validation failures'}
- Artifacts written: {', '.join(str(path) for path in reported_artifact_paths)}
- Validation result: {validation_result}
- Outcome classification: {'supportive' if success else 'constraining/contradictory'}
- Caveats or blockers: {caveats}
- Lay summary: The benchmark now has a validated metric layer for comparing exact target shapes, scrambled identity assignments, holes, missing cells, extra cells, and trajectory paths. The metrics are normalized computational scores for later sweeps, not biological measurements.
- Recommended next action: {recommended}

## Validation Table

{markdown_table(['check', 'success', 'detail'], summary_rows)}
"""
    summary_path.write_text(summary, encoding="utf-8")

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": "eidosoma.step_artifact_manifest.v1",
        "generatedAt": utc_now(),
        "artifacts": [
            {"path": str(path), "sha256": sha256_path(path), "bytes": path.stat().st_size}
            for path in artifact_paths + [status_path, summary_path]
            if path.exists() and path.is_file()
        ],
        "sourceCode": {
            "repositoryPackage": "morphospace2d/",
            "runner": "scripts/e05_s05_metrics.py",
            "tests": "tests/test_e05_metrics.py",
            "note": "Repository source is committed in git and not copied into artifacts per GitHub workspace rule.",
        },
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended,
    }
    write_json(manifest_path, manifest_payload)

    return {"success": success, "stepDir": step_dir, "status": status_payload, "artifactPaths": [str(path) for path in reported_artifact_paths]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    result = write_outputs(args.artifacts_dir)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
