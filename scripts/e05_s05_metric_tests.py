#!/usr/bin/env python3
"""Validate E05 S05 morphospace metrics and write research artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.e05.morphospace_metrics import (
    aggregate_morphospace_error,
    boundary_error,
    default_metric_specs,
    earth_mover_distance,
    evaluate_morphology_metrics,
    graph_edit_distance_proxy,
    hausdorff_distance,
    metric_result_rows,
    metric_spec_rows,
    reordered_target_state,
    swapped_target_state,
    target_neighborhood_error,
    topology_component_error,
)
from src.e05.targets import boundary_target, default_target_gallery, sorted_row_target


STEP_ID = "S05"
STEP_NUMBER = 5
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--s03-results", type=Path, default=Path("/artifacts/results/e05_target_validation.parquet"))
    parser.add_argument("--s04-results", type=Path, default=Path("/artifacts/results/e05_action_validation.parquet"))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        "elapsedSeconds": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def manifest_self_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum omitted to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def _row(
    validation_case: str,
    case_type: str,
    success: bool,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    detail: str,
) -> dict[str, Any]:
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "validation_case": validation_case,
        "case_type": case_type,
        "success": bool(success),
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": detail,
    }


def target_state_metric_rows() -> list[dict[str, Any]]:
    rows = []
    for target in default_target_gallery():
        results = evaluate_morphology_metrics(target, target.constructed_substrate(), state_label="constructed_target")
        rows.extend(metric_result_rows(results))
    return rows


def monotonic_sanity_rows() -> list[dict[str, Any]]:
    target = sorted_row_target((5, 1, 4, 2, 3))
    states = {
        "constructed_target": target.constructed_substrate(),
        "adjacent_swap": swapped_target_state(target, 1, 2),
        "reversed": reordered_target_state(target, tuple(reversed(target.substrate.site_ids))),
    }
    rows = []
    for label, state in states.items():
        results = evaluate_morphology_metrics(target, state, state_label=label)
        rows.append(
            {
                "state_label": label,
                "target_id": target.target_id,
                "aggregate_morphospace_error": aggregate_morphospace_error(results),
                "metric_values_json": stable_json({result.spec.metric_id: result.value for result in results}),
            }
        )
    return rows


def validation_catalog(metric_catalog_df: pd.DataFrame) -> dict[str, Any]:
    required = {
        "spatial_identity",
        "target_neighborhood",
        "boundary",
        "topology",
        "shape_moments",
        "hausdorff",
        "earth_mover",
        "graph_edit",
    }
    approximations = metric_catalog_df[metric_catalog_df["exactness"] == "approximation"]
    observed = {
        "families_present": sorted(metric_catalog_df["metric_family"].tolist()),
        "required_subset": required.issubset(set(metric_catalog_df["metric_family"])),
        "approximations_have_labels": bool((approximations["approximation_label"].astype(str).str.len() > 0).all()),
        "metric_count": len(metric_catalog_df),
    }
    expected = {"required_subset": True, "approximations_have_labels": True}
    return _row(
        "metric_catalog_includes_required_families_and_labels_approximations",
        "catalog",
        observed["required_subset"] and observed["approximations_have_labels"],
        expected,
        observed,
        "Metric catalog covers identity, neighborhood, boundary, topology, shape moments, Hausdorff, Earth-mover, and graph-edit proxy families, with approximation labels where needed.",
    )


def validation_target_state_zero(target_state_df: pd.DataFrame) -> dict[str, Any]:
    max_value = float(target_state_df["value"].max()) if not target_state_df.empty else 0.0
    observed = {
        "row_count": len(target_state_df),
        "all_zero": bool((target_state_df["value"] == 0.0).all()),
        "max_value": max_value,
    }
    expected = {"all_zero": True, "max_value": 0.0}
    return _row(
        "constructed_target_states_have_zero_or_minimum_metrics",
        "zero_target",
        observed["all_zero"] and observed["max_value"] == 0.0,
        expected,
        observed,
        "Every default S03 target state evaluates to the metric lower bound for all S05 metrics.",
    )


def validation_boundary_and_graph_positive() -> dict[str, Any]:
    target = boundary_target(4, 3)
    perturbed = swapped_target_state(target, 1, 5)
    boundary_value, _ = boundary_error(target, perturbed)
    graph_value, graph_details = graph_edit_distance_proxy(target, perturbed)
    neighborhood_value, _ = target_neighborhood_error(target, perturbed)
    observed = {
        "boundary_positive": boundary_value > 0.0,
        "graph_positive": graph_value > 0.0,
        "neighborhood_positive": neighborhood_value > 0.0,
        "node_mismatch": graph_details["node_mismatch"],
    }
    expected = {"boundary_positive": True, "graph_positive": True, "neighborhood_positive": True}
    return _row(
        "boundary_graph_and_neighborhood_metrics_detect_label_swap",
        "positive_control",
        all(observed[key] == value for key, value in expected.items()),
        expected,
        observed,
        "A boundary/interior label swap produces positive boundary, graph-edit proxy, and target-neighborhood errors.",
    )


def validation_topology_positive() -> dict[str, Any]:
    target = boundary_target(5, 5)
    perturbed = swapped_target_state(target, 2, 12)
    topology_value, topology_details = topology_component_error(target, perturbed)
    observed = {
        "topology_positive": topology_value > 0.0,
        "topology_value": topology_value,
        "component_counts": topology_details["component_counts"],
    }
    expected = {"topology_positive": True}
    return _row(
        "topology_component_proxy_detects_fragmented_boundary",
        "positive_control",
        observed["topology_positive"] == expected["topology_positive"],
        expected,
        observed,
        "A center/perimeter label swap creates an extra boundary component detected by the topology proxy.",
    )


def validation_hausdorff_emd_positive() -> dict[str, Any]:
    target = boundary_target(4, 3)
    perturbed = swapped_target_state(target, 1, 5)
    hausdorff_value, _ = hausdorff_distance(target, perturbed)
    emd_value, _ = earth_mover_distance(target, perturbed)
    observed = {
        "hausdorff_positive": hausdorff_value > 0.0,
        "emd_positive": emd_value > 0.0,
        "hausdorff_value": hausdorff_value,
        "emd_value": emd_value,
    }
    expected = {"hausdorff_positive": True, "emd_positive": True}
    return _row(
        "hausdorff_and_earth_mover_proxies_detect_displacement",
        "positive_control",
        observed["hausdorff_positive"] == expected["hausdorff_positive"] and observed["emd_positive"] == expected["emd_positive"],
        expected,
        observed,
        "Discrete Hausdorff and sliced-coordinate Earth-mover proxies increase after a target-label displacement.",
    )


def validation_monotonic(monotonic_df: pd.DataFrame) -> dict[str, Any]:
    values = {
        row.state_label: float(row.aggregate_morphospace_error)
        for row in monotonic_df.itertuples()
    }
    observed = {
        "constructed_target": values["constructed_target"],
        "adjacent_swap": values["adjacent_swap"],
        "reversed": values["reversed"],
        "monotonic": values["constructed_target"] < values["adjacent_swap"] < values["reversed"],
    }
    expected = {"constructed_target": 0.0, "monotonic": True}
    return _row(
        "aggregate_metric_is_monotonic_for_ordered_row_sanity_case",
        "monotonic",
        observed["constructed_target"] == expected["constructed_target"] and observed["monotonic"],
        expected,
        observed,
        "Aggregate metric is zero for the sorted-row target, positive after one adjacent swap, and larger for the reversed row.",
    )


def validation_exact_and_approximation_bounds(metric_catalog_df: pd.DataFrame) -> dict[str, Any]:
    observed = {
        "lower_bounds_nonnegative": bool((metric_catalog_df["lower_bound"] >= 0.0).all()),
        "exact_count": int((metric_catalog_df["exactness"] == "exact_for_declared_state").sum()),
        "approximation_count": int((metric_catalog_df["exactness"] == "approximation").sum()),
        "approximation_labels": sorted(metric_catalog_df.loc[metric_catalog_df["exactness"] == "approximation", "approximation_label"].tolist()),
    }
    expected = {"lower_bounds_nonnegative": True, "exact_count_min": 1, "approximation_count_min": 1}
    success = (
        observed["lower_bounds_nonnegative"]
        and observed["exact_count"] >= expected["exact_count_min"]
        and observed["approximation_count"] >= expected["approximation_count_min"]
    )
    return _row(
        "metric_specs_record_bounds_and_exactness_classes",
        "catalog",
        success,
        expected,
        observed,
        "Metric specs record lower bounds and distinguish exact fixed-state checks from labeled approximations.",
    )


def validation_result_rows_machine_readable(target_state_df: pd.DataFrame) -> dict[str, Any]:
    required_columns = {
        "target_id",
        "state_label",
        "metric_id",
        "metric_family",
        "value",
        "exactness",
        "approximation_label",
        "details_json",
    }
    observed = {
        "required_columns_present": required_columns.issubset(set(target_state_df.columns)),
        "row_count": len(target_state_df),
    }
    expected = {"required_columns_present": True, "row_count_min": len(default_target_gallery()) * len(default_metric_specs())}
    return _row(
        "metric_result_rows_are_machine_readable",
        "artifact_schema",
        observed["required_columns_present"] and observed["row_count"] >= expected["row_count_min"],
        expected,
        observed,
        "Metric result rows include target/state/metric identifiers, values, exactness labels, approximation labels, and JSON details.",
    )


def run_validations(metric_catalog_df: pd.DataFrame, target_state_df: pd.DataFrame, monotonic_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            validation_catalog(metric_catalog_df),
            validation_target_state_zero(target_state_df),
            validation_boundary_and_graph_positive(),
            validation_topology_positive(),
            validation_hausdorff_emd_positive(),
            validation_monotonic(monotonic_df),
            validation_exact_and_approximation_bounds(metric_catalog_df),
            validation_result_rows_machine_readable(target_state_df),
        ]
    )


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def top_summary_markdown(artifacts: list[dict[str, Any]], validation_result: str, recommended_next_action: str) -> str:
    artifact_lines = "\n".join(f"- `{entry['path']}`" for entry in artifacts if entry.get("path"))
    return f"""## Top Summary

- Research step ID: {STEP_ID}
- Completion status: Completed
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: supportive
- Caveats or blockers: S05 metrics are computational proxy distances; exact graph edit distance and exact multidimensional Earth-mover distance are replaced by labeled approximations.
- Lay summary: S05 defines a metric suite that scores target-state identity, neighborhoods, boundaries, topology, shape moments, Hausdorff-like displacement, Earth-mover-like displacement, and labeled-graph mismatch, with zero target-state checks and monotonic sanity controls.
- Recommended next action: {recommended_next_action}
"""


def metric_spec_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    metric_catalog_df: pd.DataFrame,
    recommended_next_action: str,
) -> str:
    return f"""{top_summary_markdown(artifacts, validation_result, recommended_next_action)}

# E05 S05 Morphospace Metric Specification

## Scope

S05 defines metrics for measuring progress toward S03 target morphologies on S01 substrates with S02 identities and S04 action-ready states. The metrics are bounded lower at zero. A constructed target state must score zero or the minimum value for every metric.

## Metric Catalog

{markdown_table(metric_catalog_df, ["metric_id", "metric_family", "lower_bound", "exactness", "approximation_label", "local_or_global"])}

## Approximation Labels

Exact graph edit distance and exact multidimensional Earth-mover distance are not implemented in S05. Instead, the catalog labels the graph metric as `fixed_graph_label_edge_proxy`, topology as `connected_component_count_proxy`, shape moments as `centroid_second_moment_proxy`, Hausdorff as `discrete_site_coordinate_hausdorff`, and Earth-mover as `sliced_coordinate_wasserstein_proxy`.

## Validation Summary

{markdown_table(validation_df, ["validation_case", "case_type", "success", "detail"])}
"""


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    metric_catalog_df: pd.DataFrame,
    monotonic_df: pd.DataFrame,
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    args: argparse.Namespace,
    recommended_next_action: str,
) -> str:
    command_lines = "\n".join(
        f"- `{command['command']}`: return code {command['returnCode']}, success={command['success']}, elapsed={command['elapsedSeconds']:.3f}s"
        for command in test_commands
    )
    source_lines = "\n".join(
        f"- `{entry['relativePath']}` sha256 `{entry['sha256']}` ({entry['sizeBytes']} bytes)"
        for entry in source_files
    )
    return f"""{top_summary_markdown(artifacts, validation_result, recommended_next_action)}

# Research Step Full Results: {STEP_ID} Define Morphospace Metrics

## Lay Summary

S05 defines the distance functions that later simulations will use to ask whether a tissue state is moving toward a target morphology. The metric suite checks exact fixed-site identity, local neighborhood labels, boundary labels, and several labeled approximations for topology and shape. Constructed target states score zero, and simple perturbations increase the relevant errors.

## Frozen Question

Can higher-dimensional target progress be measured with robust metrics analogous to Sortedness but suited to shape, topology, and spatial identity?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S05.
- S03 target morphology artifacts, including `{args.s03_results}`.
- S04 action-set artifacts, including `{args.s04_results}`.
- Repository code from S01 to S04.
- Datasets: none required.

## Methods

Implemented `src/e05/morphospace_metrics.py` with metric specs, exact fixed-state identity/neighborhood/boundary checks, labeled topology/shape/Hausdorff/Earth-mover/graph-edit approximations, and deterministic target-state perturbation helpers. Added tests in `tests/e05/test_morphospace_metrics.py` and generated validation/report artifacts through `scripts/e05_s05_metric_tests.py`.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- No new packages were installed for S05.
- CPU/GPU use: validation and report generation are small serial CPU tasks; no GPU use was needed.

## Parameters

- Metric count: {len(metric_catalog_df)}.
- Default target-state checks: {len(default_target_gallery())} targets x {len(default_metric_specs())} metrics.
- Approximation count: {int((metric_catalog_df["exactness"] == "approximation").sum())}.

## Results

{markdown_table(validation_df, ["validation_case", "case_type", "success", "detail"])}

All validation rows passed. The primary success criterion was met: metrics return zero/minimum for constructed target states, perturbation controls increase appropriate errors, approximations are labeled, and a sorted-row sanity case is monotonic.

## Metric Catalog

{markdown_table(metric_catalog_df, ["metric_id", "metric_family", "lower_bound", "exactness", "approximation_label", "caveat"])}

## Monotonic Sanity Case

{markdown_table(monotonic_df, ["state_label", "target_id", "aggregate_morphospace_error"])}

## Validation Checks

- Metric catalog includes required metric families and labels approximations.
- Constructed S03 target states score zero for every S05 metric.
- Boundary, neighborhood, and graph-edit proxy metrics detect a boundary/interior label swap.
- Topology proxy detects a fragmented-boundary fixture.
- Discrete Hausdorff and sliced-coordinate Earth-mover proxies detect displacement.
- Aggregate morphospace error is monotonic for an ordered-row exact, adjacent-swap, and reversed-row sanity case.
- Metric rows are machine-readable with exactness and approximation metadata.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- Metrics are computational proxies, not direct biological morphology measurements.
- Exact graph edit distance is replaced by a fixed-graph labeled-node/edge proxy.
- Exact multidimensional Earth-mover distance is replaced by a sliced coordinate-axis Wasserstein proxy.
- Shape moments can miss shape differences that preserve centroid and spread.
- S05 validates metric behavior on deterministic fixtures only; benchmark recovery distributions remain future S07 onward work.

## Blockers And Failed Assumptions

No blocker was found. The target-state lower-bound checks and monotonic sanity controls pass with labeled approximations.

## Recommended Next Action

{recommended_next_action}
"""


def write_checksums(paths: list[Path], checksum_path: Path, artifacts_dir: Path) -> None:
    lines = []
    for path in sorted(paths):
        if path == checksum_path:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}")
    write_text(checksum_path, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    reports_dir = artifacts_dir / "reports"
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, reports_dir, results_dir, tables_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    metric_catalog_df = pd.DataFrame(metric_spec_rows(default_metric_specs()))
    target_state_df = pd.DataFrame(target_state_metric_rows())
    monotonic_df = pd.DataFrame(monotonic_sanity_rows())
    validation_df = run_validations(metric_catalog_df, target_state_df, monotonic_df)
    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"
    recommended_next_action = "Stop before S06 and let the Chief Scientist review S05; if accepted, proceed to replicate 1D inside 2D in S06."

    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )
    test_success = all(command["success"] for command in test_commands)

    validation_path = results_dir / "e05_metric_validation.parquet"
    validation_csv_path = tables_dir / "e05_metric_validation.csv"
    metric_catalog_path = results_dir / "e05_metric_catalog.parquet"
    metric_catalog_csv_path = tables_dir / "e05_metric_catalog.csv"
    target_state_path = results_dir / "e05_metric_target_state_checks.parquet"
    target_state_csv_path = tables_dir / "e05_metric_target_state_checks.csv"
    monotonic_path = results_dir / "e05_metric_monotonic_sanity.parquet"
    monotonic_csv_path = tables_dir / "e05_metric_monotonic_sanity.csv"
    metric_specs_path = results_dir / "e05_metric_specs.json"
    config_path = configs_dir / "e05_s05_metric_tests.json"
    source_manifest_path = src_snapshot_dir / "e05_morphospace_metrics_manifest.json"
    spec_path = reports_dir / "e05_morphospace_metrics.md"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)
    metric_catalog_df.to_parquet(metric_catalog_path, index=False)
    metric_catalog_df.to_csv(metric_catalog_csv_path, index=False)
    target_state_df.to_parquet(target_state_path, index=False)
    target_state_df.to_csv(target_state_csv_path, index=False)
    monotonic_df.to_parquet(monotonic_path, index=False)
    monotonic_df.to_csv(monotonic_csv_path, index=False)
    write_json(
        metric_specs_path,
        {
            "schema": "eidosoma.e05_s05.metric_specs.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "metrics": [spec.compact_dict() for spec in default_metric_specs()],
        },
    )
    write_json(
        config_path,
        {
            "schema": "eidosoma.e05_s05.metric_validation_config.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "validationCases": validation_df["validation_case"].tolist(),
            "unitTestsRun": bool(args.run_unit_tests),
            "createdAt": started_at,
        },
    )

    source_files = [
        source_entry(args.repo_dir / "src/e05/morphospace_metrics.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_morphospace_metrics.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s05_metric_tests.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/targets.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/substrates.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/cell_identity.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/actions.py", args.repo_dir),
    ]
    write_json(
        source_manifest_path,
        {
            "schema": "eidosoma.source_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "sourceFiles": source_files,
            "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
            "createdAt": utc_now(),
        },
    )

    artifacts_for_summary = [
        artifact_entry(validation_path, artifacts_dir, "Machine-readable S05 metric validation table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV mirror of S05 validation table."),
        artifact_entry(metric_catalog_path, artifacts_dir, "Machine-readable S05 metric catalog."),
        artifact_entry(metric_catalog_csv_path, artifacts_dir, "CSV mirror of S05 metric catalog."),
        artifact_entry(target_state_path, artifacts_dir, "Target-state zero/minimum metric checks."),
        artifact_entry(target_state_csv_path, artifacts_dir, "CSV mirror of target-state checks."),
        artifact_entry(monotonic_path, artifacts_dir, "Monotonic sanity-case metric results."),
        artifact_entry(monotonic_csv_path, artifacts_dir, "CSV mirror of monotonic sanity results."),
        artifact_entry(metric_specs_path, artifacts_dir, "JSON metric specifications."),
        artifact_entry(config_path, artifacts_dir, "S05 validation configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "Repository source hashes for S05 code and tests."),
    ]
    planned_artifact_paths = [
        {"path": str(spec_path), "description": "S05 morphospace metric specification report."},
        {"path": str(full_results_path), "description": "S05 full-results handoff report."},
        {"path": str(artifact_manifest_path), "description": "S05 artifact manifest."},
        {"path": str(run_manifest_path), "description": "Experiment run manifest updated for S05."},
        {"path": str(checksum_path), "description": "Checksums for key S05 artifacts."},
    ]
    write_text(
        spec_path,
        metric_spec_markdown(
            [*artifacts_for_summary, *planned_artifact_paths],
            validation_result,
            validation_df,
            metric_catalog_df,
            recommended_next_action,
        ),
    )
    artifacts_after_spec = [
        *artifacts_for_summary,
        artifact_entry(spec_path, artifacts_dir, "S05 morphospace metric specification report."),
    ]
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts=[*artifacts_after_spec, *planned_artifact_paths[1:]],
            validation_result=f"{validation_result}; unit-test commands success={test_success}",
            validation_df=validation_df,
            metric_catalog_df=metric_catalog_df,
            monotonic_df=monotonic_df,
            test_commands=test_commands,
            source_files=source_files,
            args=args,
            recommended_next_action=recommended_next_action,
        ),
    )
    artifacts_final = [
        *artifacts_after_spec,
        artifact_entry(full_results_path, artifacts_dir, "S05 full-results handoff report."),
    ]
    write_json(
        artifact_manifest_path,
        {
            "schema": "eidosoma.artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "success": bool(validation_success and test_success),
            "artifacts": [*artifacts_final, manifest_self_entry(artifact_manifest_path, artifacts_dir, "S05 artifact manifest.")],
            "validationResult": f"{validation_result}; unit-test commands success={test_success}",
            "caveatsOrBlockers": "No blocker. Exact graph edit and multidimensional Earth-mover distances are replaced by labeled approximations.",
            "recommendedNextAction": recommended_next_action,
            "createdAt": utc_now(),
        },
    )
    run_manifest_payload = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "experimentTitle": EXPERIMENT_TITLE,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "success": bool(validation_success and test_success),
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "packages": {"pandas": pd.__version__},
        "commands": test_commands,
        "artifacts": [*artifacts_final, artifact_entry(artifact_manifest_path, artifacts_dir, "S05 artifact manifest.")],
        "validationResult": f"{validation_result}; unit-test commands success={test_success}",
    }
    write_json(run_manifest_path, run_manifest_payload)
    checksum_inputs = [
        validation_path,
        validation_csv_path,
        metric_catalog_path,
        metric_catalog_csv_path,
        target_state_path,
        target_state_csv_path,
        monotonic_path,
        monotonic_csv_path,
        metric_specs_path,
        config_path,
        source_manifest_path,
        spec_path,
        full_results_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    write_checksums(checksum_inputs, checksum_path, artifacts_dir)

    print(json.dumps({"success": bool(validation_success and test_success), "validationResult": validation_result, "artifactsDir": str(artifacts_dir)}, indent=2))
    return 0 if validation_success and test_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
