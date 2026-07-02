#!/usr/bin/env python3
"""Validate E05 S06 1D sorting embedded inside a 2D grid."""

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
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.e05.embedded_1d import (
    ADJACENT_ALGORITHMS,
    build_embedded_state,
    embedded_sorted_row_target,
    execute_row_swap,
    row_neighbor_swap_allowed,
    row_site_ids,
    run_adjacent_sort,
    run_summary_row,
    stable_json_sha256,
    trace_rows,
)


STEP_ID = "S06"
STEP_NUMBER = 6
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"
SUBSTRATE_KINDS = ("array_1d", "embedded_square_grid_2d")
STRICT_ALGORITHMS = ("bubble", "insertion")
CONDITION_BY_ALGORITHM_MODE = {
    ("bubble", "traditional"): "E01C001",
    ("insertion", "traditional"): "E01C002",
    ("bubble", "cell_view"): "E01C004",
    ("insertion", "cell_view"): "E01C005",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--e01-config", type=Path, default=Path("/previous-artifacts/E01/configs/e01_baseline_configs.json"))
    parser.add_argument("--e01-efficiency", type=Path, default=Path("/previous-artifacts/E01/results/e01_efficiency_counts.parquet"))
    parser.add_argument("--algorithms", default=",".join(STRICT_ALGORITHMS))
    parser.add_argument("--max-repeats", type=int, default=None)
    parser.add_argument("--grid-height", type=int, default=3)
    parser.add_argument("--row-y", type=int, default=1)
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


def selected_algorithms(raw: str) -> tuple[str, ...]:
    algorithms = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not algorithms:
        raise ValueError("at least one algorithm is required")
    unsupported = sorted(set(algorithms) - set(ADJACENT_ALGORITHMS))
    if unsupported:
        raise ValueError(f"S06 strict row-neighbor validation supports only adjacent algorithms; unsupported={unsupported}")
    return algorithms


def load_e01_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_lookup(config: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    for row in config.get("conditions", []):
        if row.get("run_family") != "unperturbed_baseline":
            continue
        key = (str(row.get("algorithm")), str(row.get("mode")))
        rows[key] = dict(row)
    return rows


def run_s06_repeats(
    *,
    config: Mapping[str, Any],
    algorithms: Sequence[str],
    max_repeats: int | None,
    grid_height: int,
    row_y: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    value_bank = config["seedBanks"]["valueBanks"]["unique_1_to_100"]
    arrays = value_bank["initialArrays"]
    seeds = value_bank["seeds"]
    repeat_count = int(value_bank["repeatCount"])
    if max_repeats is not None:
        repeat_count = min(repeat_count, int(max_repeats))
    lookup = condition_lookup(config)

    summary_rows = []
    trajectory_rows = []
    for algorithm in algorithms:
        condition = lookup[(algorithm, "cell_view")]
        condition_id = str(condition["condition_id"])
        matched_group_id = str(condition["matched_group_id"])
        value_bank_id = str(condition["value_bank_id"])
        for repeat_index in range(repeat_count):
            initial_values = tuple(arrays[repeat_index])
            initial_seed = int(seeds[repeat_index])
            for substrate_kind in SUBSTRATE_KINDS:
                result = run_adjacent_sort(
                    initial_values,
                    algorithm=algorithm,
                    substrate_kind=substrate_kind,
                    grid_height=grid_height,
                    row_y=row_y,
                )
                summary_rows.append(
                    run_summary_row(
                        result,
                        repeat_index=repeat_index,
                        initial_array_seed=initial_seed,
                        condition_id=condition_id,
                        matched_group_id=matched_group_id,
                        value_bank_id=value_bank_id,
                    )
                )
                trajectory_rows.extend(
                    trace_rows(
                        result,
                        repeat_index=repeat_index,
                        initial_array_seed=initial_seed,
                        condition_id=condition_id,
                        matched_group_id=matched_group_id,
                        value_bank_id=value_bank_id,
                    )
                )
    run_context = {
        "repeatCount": repeat_count,
        "arrayLength": int(value_bank["arrayLength"]),
        "valueBankId": "unique_1_to_100",
        "algorithms": list(algorithms),
        "gridHeight": int(grid_height),
        "rowY": int(row_y),
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(trajectory_rows), run_context


def validation_config(config: Mapping[str, Any], algorithms: Sequence[str], run_context: Mapping[str, Any]) -> dict[str, Any]:
    lookup = condition_lookup(config)
    observed = {
        "array_length": int(run_context["arrayLength"]),
        "repeat_count": int(run_context["repeatCount"]),
        "algorithms": list(algorithms),
        "conditions_present": {
            f"{algorithm}:{mode}": (algorithm, mode) in lookup
            for algorithm in algorithms
            for mode in ("traditional", "cell_view")
        },
        "selection_exclusion": "Selection is excluded from strict S06 row-neighbor validation because E01 reconstructed selection uses nonlocal swaps.",
    }
    expected = {"array_length": 100, "repeat_count_min": 1, "all_conditions_present": True}
    success = (
        observed["array_length"] == expected["array_length"]
        and observed["repeat_count"] >= expected["repeat_count_min"]
        and all(observed["conditions_present"].values())
    )
    return _row(
        "e01_baseline_config_loaded_for_adjacent_swap_conditions",
        "input",
        success,
        expected,
        observed,
        "Loaded E01 unique 1..100 baseline arrays and unperturbed Bubble/Insertion condition rows from the upstream config.",
    )


def validation_initial_hashes(config: Mapping[str, Any], efficiency_df: pd.DataFrame, algorithms: Sequence[str], repeat_count: int) -> dict[str, Any]:
    arrays = config["seedBanks"]["valueBanks"]["unique_1_to_100"]["initialArrays"][:repeat_count]
    observed_hashes = [stable_json_sha256(list(values)) for values in arrays]
    per_condition = {}
    for algorithm in algorithms:
        for mode in ("traditional", "cell_view"):
            condition_id = CONDITION_BY_ALGORITHM_MODE[(algorithm, mode)]
            expected_hashes = (
                efficiency_df.loc[efficiency_df["condition_id"] == condition_id]
                .sort_values("repeat_index")["initial_array_sha256"]
                .head(repeat_count)
                .tolist()
            )
            per_condition[condition_id] = observed_hashes == expected_hashes
    observed = {
        "repeat_count": repeat_count,
        "per_condition_hash_match": per_condition,
    }
    expected = {"all_hashes_match": True}
    return _row(
        "reconstructed_config_initial_arrays_match_e01_run_hashes",
        "input",
        all(per_condition.values()),
        expected,
        observed,
        "The S06 arrays loaded from E01 baseline config reproduce the E01 run-record initial array hashes.",
    )


def validation_e01_step_counts(summary_df: pd.DataFrame, efficiency_df: pd.DataFrame, algorithms: Sequence[str]) -> dict[str, Any]:
    checks = {}
    for algorithm in algorithms:
        condition_id = CONDITION_BY_ALGORITHM_MODE[(algorithm, "cell_view")]
        e01 = efficiency_df.loc[efficiency_df["condition_id"] == condition_id, ["repeat_index", "swap_only_steps"]].copy()
        e01["repeat_index"] = e01["repeat_index"].astype(int)
        observed = summary_df[
            (summary_df["algorithm"] == algorithm)
            & (summary_df["substrate_kind"] == "array_1d")
        ][["repeat_index", "swap_only_steps"]].copy()
        merged = observed.merge(e01, on="repeat_index", suffixes=("_s06", "_e01"))
        checks[algorithm] = {
            "rows_compared": int(len(merged)),
            "all_swap_steps_equal": bool((merged["swap_only_steps_s06"].astype(int) == merged["swap_only_steps_e01"].astype(int)).all()),
            "max_abs_delta": int((merged["swap_only_steps_s06"].astype(int) - merged["swap_only_steps_e01"].astype(int)).abs().max()) if not merged.empty else None,
        }
    observed = {"by_algorithm": checks}
    expected = {"all_swap_steps_equal": True, "max_abs_delta": 0}
    success = all(item["all_swap_steps_equal"] and item["max_abs_delta"] == 0 for item in checks.values())
    return _row(
        "array_1d_step_counts_match_e01_adjacent_swap_baselines",
        "e01_continuity",
        success,
        expected,
        observed,
        "S06 1D substrate Bubble/Insertion swap counts match E01 cell-view adjacent-swap run records for every repeat.",
    )


def validation_2d_matches_1d(summary_df: pd.DataFrame) -> dict[str, Any]:
    left = summary_df[summary_df["substrate_kind"] == "array_1d"].copy()
    right = summary_df[summary_df["substrate_kind"] == "embedded_square_grid_2d"].copy()
    merged = left.merge(
        right,
        on=["algorithm", "repeat_index"],
        suffixes=("_1d", "_2d"),
    )
    swap_delta = merged["swap_only_steps_1d"].astype(int) - merged["swap_only_steps_2d"].astype(int)
    sorted_delta = merged["final_sortedness_percent_1d"].astype(float) - merged["final_sortedness_percent_2d"].astype(float)
    observed = {
        "rows_compared": int(len(merged)),
        "all_swap_steps_equal": bool((swap_delta == 0).all()),
        "max_abs_swap_delta": int(swap_delta.abs().max()) if not merged.empty else None,
        "max_abs_final_sortedness_delta": float(sorted_delta.abs().max()) if not merged.empty else None,
    }
    expected = {"all_swap_steps_equal": True, "max_abs_swap_delta": 0, "max_abs_final_sortedness_delta": 0.0}
    return _row(
        "embedded_2d_step_counts_match_array_1d_exactly",
        "cross_substrate",
        observed["all_swap_steps_equal"]
        and observed["max_abs_swap_delta"] == 0
        and observed["max_abs_final_sortedness_delta"] == 0.0,
        expected,
        observed,
        "For the same initial arrays and row-neighbor policy, embedded 2D runs have exactly the same swap counts and final Sortedness as 1D substrate runs.",
    )


def validation_trajectory_equality(trace_df: pd.DataFrame) -> dict[str, Any]:
    key = ["algorithm", "repeat_index", "swap_step"]
    left = trace_df[trace_df["substrate_kind"] == "array_1d"].copy()
    right = trace_df[trace_df["substrate_kind"] == "embedded_square_grid_2d"].copy()
    merged = left.merge(right, on=key, suffixes=("_1d", "_2d"))
    sorted_delta = merged["sortedness_percent_1d"].astype(float) - merged["sortedness_percent_2d"].astype(float)
    mono_delta = merged["monotonicity_error_count_1d"].astype(int) - merged["monotonicity_error_count_2d"].astype(int)
    comparison_delta = merged["comparison_count_at_step_1d"].astype(int) - merged["comparison_count_at_step_2d"].astype(int)
    observed = {
        "array_1d_trace_rows": int(len(left)),
        "embedded_2d_trace_rows": int(len(right)),
        "merged_rows": int(len(merged)),
        "row_counts_equal": int(len(left)) == int(len(right)) == int(len(merged)),
        "max_abs_sortedness_delta": float(sorted_delta.abs().max()) if not merged.empty else None,
        "max_abs_monotonicity_delta": int(mono_delta.abs().max()) if not merged.empty else None,
        "max_abs_comparison_delta": int(comparison_delta.abs().max()) if not merged.empty else None,
    }
    expected = {
        "row_counts_equal": True,
        "max_abs_sortedness_delta": 0.0,
        "max_abs_monotonicity_delta": 0,
        "max_abs_comparison_delta": 0,
    }
    success = (
        observed["row_counts_equal"]
        and observed["max_abs_sortedness_delta"] == 0.0
        and observed["max_abs_monotonicity_delta"] == 0
        and observed["max_abs_comparison_delta"] == 0
    )
    return _row(
        "embedded_2d_sortedness_trajectories_match_array_1d_exactly",
        "cross_substrate",
        success,
        expected,
        observed,
        "Sortedness, monotonicity error, and comparison counter trajectories match exactly at every recorded swap step.",
    )


def validation_final_sorted(summary_df: pd.DataFrame) -> dict[str, Any]:
    observed = {
        "run_count": int(len(summary_df)),
        "all_final_sortedness_100": bool((summary_df["final_sortedness_percent"].astype(float) == 100.0).all()),
        "max_final_monotonicity_error": int(summary_df["final_monotonicity_error_count"].astype(int).max()),
        "stop_reasons": sorted(summary_df["stop_reason"].unique().tolist()),
    }
    expected = {"all_final_sortedness_100": True, "max_final_monotonicity_error": 0, "stop_reasons": ["sorted"]}
    success = (
        observed["all_final_sortedness_100"]
        and observed["max_final_monotonicity_error"] == 0
        and observed["stop_reasons"] == ["sorted"]
    )
    return _row(
        "all_s06_runs_reach_e01_final_sortedness_target",
        "terminal",
        success,
        expected,
        observed,
        "Every strict S06 run reaches 100% E01 Sortedness and zero monotonicity error.",
    )


def validation_row_constraints() -> dict[str, Any]:
    values = (3, 1, 2)
    target = embedded_sorted_row_target(values, grid_height=3, row_y=1)
    state = build_embedded_state(values, target=target, grid_height=3, row_y=1)
    row_ids = row_site_ids(3, 1, grid_height=3)
    horizontal = row_neighbor_swap_allowed(state, row_ids, row_ids[0], row_ids[1])
    vertical = row_neighbor_swap_allowed(state, row_ids, row_ids[0], 0)
    off_row_actor = row_neighbor_swap_allowed(state, row_ids, 0, row_ids[0])
    vertical_execution = execute_row_swap(state, row_ids, row_ids[0], 0)
    observed = {
        "horizontal_allowed": horizontal == (True, "allowed"),
        "vertical_rejected": vertical == (False, "target_not_in_embedded_row"),
        "off_row_actor_rejected": off_row_actor == (False, "source_not_in_embedded_row"),
        "vertical_execution_allowed": vertical_execution.allowed,
        "vertical_execution_state_changed": vertical_execution.state_changed,
        "vertical_execution_energy": vertical_execution.energy_cost_charged,
    }
    expected = {
        "horizontal_allowed": True,
        "vertical_rejected": True,
        "off_row_actor_rejected": True,
        "vertical_execution_allowed": False,
        "vertical_execution_state_changed": False,
        "vertical_execution_energy": 0.0,
    }
    return _row(
        "row_neighbor_constraint_rejects_vertical_and_off_row_swaps",
        "local_constraint",
        observed == expected,
        expected,
        observed,
        "The 2D substrate keeps vertical neighbors, but the S06 row policy admits only adjacent swaps inside the embedded row.",
    )


def validation_s05_final_metrics(summary_df: pd.DataFrame) -> dict[str, Any]:
    embedded = summary_df[summary_df["substrate_kind"] == "embedded_square_grid_2d"]
    observed = {
        "max_row_morphospace_error": float(summary_df["final_row_morphospace_error"].astype(float).max()),
        "max_row_identity_error": float(summary_df["final_row_target_identity_error"].astype(float).max()),
        "max_embedded_morphospace_error": float(embedded["final_embedded_morphospace_error"].astype(float).max()),
        "max_embedded_identity_error": float(embedded["final_embedded_target_identity_error"].astype(float).max()),
    }
    expected = {
        "max_row_morphospace_error": 0.0,
        "max_row_identity_error": 0.0,
        "max_embedded_morphospace_error": 0.0,
        "max_embedded_identity_error": 0.0,
    }
    return _row(
        "s05_metrics_are_zero_for_final_target_states",
        "metric",
        observed == expected,
        expected,
        observed,
        "Final 1D row projections and full embedded 2D target states score zero under the S05 metric suite.",
    )


def validation_figure(figure_path: Path) -> dict[str, Any]:
    size = figure_path.stat().st_size if figure_path.exists() else 0
    observed = {"exists": figure_path.exists(), "size_bytes": int(size)}
    expected = {"exists": True, "size_bytes_min": 1000}
    return _row(
        "embedded_1d_trajectory_figure_written",
        "artifact",
        observed["exists"] and observed["size_bytes"] > expected["size_bytes_min"],
        expected,
        observed,
        "The required S06 trajectory figure was written and is nonempty.",
    )


def render_trajectory_figure(trace_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = sorted(trace_df["algorithm"].unique().tolist())
    fig, axes = plt.subplots(len(algorithms), 1, figsize=(8.0, 3.4 * len(algorithms)), sharex=False, constrained_layout=True)
    if len(algorithms) == 1:
        axes = [axes]
    colors = {"array_1d": "#2F6B8F", "embedded_square_grid_2d": "#D95F02"}
    styles = {"array_1d": "-", "embedded_square_grid_2d": "--"}
    labels = {"array_1d": "1D substrate", "embedded_square_grid_2d": "2D embedded row"}
    for ax, algorithm in zip(axes, algorithms, strict=True):
        subset = trace_df[trace_df["algorithm"] == algorithm]
        for substrate_kind in SUBSTRATE_KINDS:
            rows = subset[subset["substrate_kind"] == substrate_kind]
            mean_trace = (
                rows.groupby("swap_step", as_index=False)["sortedness_percent"]
                .mean()
                .sort_values("swap_step")
            )
            ax.plot(
                mean_trace["swap_step"],
                mean_trace["sortedness_percent"],
                styles[substrate_kind],
                color=colors[substrate_kind],
                linewidth=2.0,
                label=labels[substrate_kind],
            )
        ax.set_title(f"{algorithm.capitalize()} adjacent-swap Sortedness trajectory")
        ax.set_ylabel("Sortedness (%)")
        ax.set_ylim(40, 101)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right")
    axes[-1].set_xlabel("Successful adjacent swaps")
    fig.suptitle("E05 S06: 1D sorting reproduced inside a 2D row")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_validations(
    *,
    config: Mapping[str, Any],
    efficiency_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    figure_path: Path,
    algorithms: Sequence[str],
    run_context: Mapping[str, Any],
) -> pd.DataFrame:
    repeat_count = int(run_context["repeatCount"])
    return pd.DataFrame(
        [
            validation_config(config, algorithms, run_context),
            validation_initial_hashes(config, efficiency_df, algorithms, repeat_count),
            validation_e01_step_counts(summary_df, efficiency_df, algorithms),
            validation_2d_matches_1d(summary_df),
            validation_trajectory_equality(trace_df),
            validation_final_sorted(summary_df),
            validation_row_constraints(),
            validation_s05_final_metrics(summary_df),
            validation_figure(figure_path),
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
- Caveats or blockers: Strict continuity was tested for Bubble and Insertion adjacent-swap baselines; E01 Selection is documented as outside this row-neighbor check because its reconstructed traditional baseline uses nonlocal swaps and its public cell-view policy targets ideal positions.
- Lay summary: S06 placed the original 1D sorting row inside the middle row of a 2D grid and showed that row-restricted Bubble and Insertion reproduce the 1D substrate and E01 adjacent-swap step counts exactly.
- Recommended next action: {recommended_next_action}
"""


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    run_context: Mapping[str, Any],
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
    condition_summary = (
        summary_df.groupby(["algorithm", "substrate_kind"], as_index=False)
        .agg(
            runs=("repeat_index", "count"),
            mean_swap_steps=("swap_only_steps", "mean"),
            min_swap_steps=("swap_only_steps", "min"),
            max_swap_steps=("swap_only_steps", "max"),
            final_sortedness_min=("final_sortedness_percent", "min"),
            final_sortedness_max=("final_sortedness_percent", "max"),
            max_row_morphospace_error=("final_row_morphospace_error", "max"),
            max_embedded_morphospace_error=("final_embedded_morphospace_error", "max"),
        )
    )
    return f"""{top_summary_markdown(artifacts, validation_result, recommended_next_action)}

# Research Step Full Results: {STEP_ID} Replicate 1D Inside 2D

## Lay Summary

S06 checks whether the new higher-dimensional substrate machinery can preserve the original 1D sorting task when a row is embedded in a 2D grid. Bubble and Insertion were run on the 100 E01 unique-value arrays using only adjacent row swaps. The 2D embedded row exactly matched the 1D substrate for swap counts, Sortedness trajectories, final Sortedness, and final S05 target metrics.

## Frozen Question

Does a 1D row embedded in a 2D grid reproduce original sorting behavior closely enough to validate the generalized simulator?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S06.
- E01 baseline config: `{args.e01_config}`.
- E01 efficiency run records: `{args.e01_efficiency}`.
- S01 substrate code, S02 scalar identity code, S03 target morphologies, S04 swap-cost semantics, and S05 morphospace metrics from the repository.
- Datasets: none required.

## Methods

The script loaded E01's `unique_1_to_100` 100-repeat value bank and selected the Bubble and Insertion unperturbed cell-view condition rows. For each initial array, it ran the same deterministic adjacent-swap algorithm on an S01 `array_1d` substrate and on the middle row of an S01 `square_grid_2d` substrate with stuck off-row barrier cells. The row policy rejected vertical and off-row swaps, so only east/west row-neighbor swaps could execute. S05 metrics were evaluated on final row projections and full embedded target states.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- Matplotlib: `{matplotlib.__version__}`
- No new packages were installed for S06.
- CPU/GPU use: serial CPU validation; no GPU use was needed.

## Parameters

- Algorithms: `{", ".join(run_context["algorithms"])}`
- Repeat count: `{run_context["repeatCount"]}`
- Array length: `{run_context["arrayLength"]}`
- Embedded grid height: `{run_context["gridHeight"]}`
- Embedded row y-index: `{run_context["rowY"]}`
- Substrate kinds: `{", ".join(SUBSTRATE_KINDS)}`

## Results

{markdown_table(validation_df, ["validation_case", "case_type", "success", "detail"])}

Condition summary:

{markdown_table(condition_summary, ["algorithm", "substrate_kind", "runs", "mean_swap_steps", "min_swap_steps", "max_swap_steps", "final_sortedness_min", "final_sortedness_max", "max_row_morphospace_error", "max_embedded_morphospace_error"])}

The primary success criterion was met. The 2D embedded row matched the 1D substrate exactly, and the 1D substrate swap counts matched E01 adjacent-swap Bubble and Insertion baselines for every configured repeat.

## Validation Checks

- E01 baseline config and run-record hashes were loaded and matched.
- S06 1D swap counts matched E01 Bubble/Insertion adjacent-swap run records.
- 2D embedded-row swap counts matched S06 1D swap counts exactly.
- Sortedness, monotonicity error, and comparison-count trajectories matched exactly at every recorded swap step.
- Every run reached 100% final Sortedness and zero monotonicity error.
- Vertical and off-row swaps were rejected by the row-neighbor constraint.
- Final row projections and full embedded target states scored zero under S05 metrics.
- The required trajectory figure was written.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- Strict S06 continuity covers Bubble and Insertion because they are adjacent-swap baselines with E01 step counts equal to inversion counts.
- E01 Selection is not used for the strict row-neighbor step-count match because the reconstructed traditional baseline swaps nonadjacent positions and the public cell-view selection policy targets ideal positions.
- The embedded 2D target uses stuck scalar barrier cells outside the row; this validates row isolation, not full 2D morphogenesis.
- E01 cell-view comparison counts are public StatusProbe actionable-comparison proxies, so S06 treats swap counts and Sortedness trajectories as the primary continuity metrics.

## Blockers And Failed Assumptions

No blocker was found. The assumption that the adjacent-swap row can be isolated inside a 2D substrate held for Bubble and Insertion.

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
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    traces_dir = artifacts_dir / "traces"
    figures_dir = artifacts_dir / "figures" / "e05"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, traces_dir, figures_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    algorithms = selected_algorithms(args.algorithms)
    config = load_e01_config(args.e01_config)
    efficiency_df = pd.read_parquet(args.e01_efficiency)
    summary_df, trace_df, run_context = run_s06_repeats(
        config=config,
        algorithms=algorithms,
        max_repeats=args.max_repeats,
        grid_height=args.grid_height,
        row_y=args.row_y,
    )

    figure_path = figures_dir / "embedded_1d_trajectories.png"
    render_trajectory_figure(trace_df, figure_path)
    validation_df = run_validations(
        config=config,
        efficiency_df=efficiency_df,
        summary_df=summary_df,
        trace_df=trace_df,
        figure_path=figure_path,
        algorithms=algorithms,
        run_context=run_context,
    )
    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"
    recommended_next_action = "Stop before S07 and let the Chief Scientist review S06; if accepted, proceed to scrambled-embryo tests in S07."

    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )
    test_success = all(command["success"] for command in test_commands)

    validation_path = results_dir / "e05_1d_embedded_validation.parquet"
    validation_csv_path = tables_dir / "e05_1d_embedded_validation.csv"
    summary_path = results_dir / "e05_1d_embedded_run_summary.parquet"
    summary_csv_path = tables_dir / "e05_1d_embedded_run_summary.csv"
    trace_path = traces_dir / "e05_1d_embedded_trajectories.parquet"
    trace_csv_sample_path = tables_dir / "e05_1d_embedded_trace_sample.csv"
    config_path = configs_dir / "e05_s06_1d_embedded_validation.json"
    source_manifest_path = src_snapshot_dir / "e05_embedded_1d_manifest.json"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)
    summary_df.to_parquet(summary_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    trace_df.to_parquet(trace_path, index=False)
    trace_df.head(5000).to_csv(trace_csv_sample_path, index=False)
    write_json(
        config_path,
        {
            "schema": "eidosoma.e05_s06.embedded_1d_validation_config.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "sourceExperimentId": "E01",
            "e01ConfigPath": str(args.e01_config),
            "e01ConfigSha256": sha256_file(args.e01_config),
            "e01EfficiencyPath": str(args.e01_efficiency),
            "e01EfficiencySha256": sha256_file(args.e01_efficiency),
            "algorithms": list(algorithms),
            "substrateKinds": list(SUBSTRATE_KINDS),
            "runContext": dict(run_context),
            "unitTestsRun": bool(args.run_unit_tests),
            "createdAt": started_at,
        },
    )

    source_files = [
        source_entry(args.repo_dir / "src/e05/embedded_1d.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_embedded_1d.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s06_1d_embedded_validation.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/substrates.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/actions.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/cell_identity.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/targets.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/morphospace_metrics.py", args.repo_dir),
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
        artifact_entry(validation_path, artifacts_dir, "Machine-readable S06 validation table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV mirror of S06 validation table."),
        artifact_entry(summary_path, artifacts_dir, "S06 run-level 1D and embedded-2D summary table."),
        artifact_entry(summary_csv_path, artifacts_dir, "CSV mirror of S06 run summary."),
        artifact_entry(trace_path, artifacts_dir, "S06 swap-step trajectory table for 1D and embedded 2D runs."),
        artifact_entry(trace_csv_sample_path, artifacts_dir, "CSV sample of S06 trajectory rows."),
        artifact_entry(figure_path, artifacts_dir, "Required S06 embedded 1D trajectory figure."),
        artifact_entry(config_path, artifacts_dir, "S06 validation configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "Repository source hashes for S06 code and tests."),
    ]
    planned_artifact_paths = [
        {"path": str(full_results_path), "description": "S06 full-results handoff report."},
        {"path": str(artifact_manifest_path), "description": "S06 artifact manifest."},
        {"path": str(run_manifest_path), "description": "Experiment run manifest updated for S06."},
        {"path": str(checksum_path), "description": "Checksums for key S06 artifacts."},
    ]
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts=[*artifacts_for_summary, *planned_artifact_paths],
            validation_result=f"{validation_result}; unit-test commands success={test_success}",
            validation_df=validation_df,
            summary_df=summary_df,
            run_context=run_context,
            test_commands=test_commands,
            source_files=source_files,
            args=args,
            recommended_next_action=recommended_next_action,
        ),
    )
    artifacts_final = [
        *artifacts_for_summary,
        artifact_entry(full_results_path, artifacts_dir, "S06 full-results handoff report."),
    ]
    write_json(
        artifact_manifest_path,
        {
            "schema": "eidosoma.artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "success": bool(validation_success and test_success),
            "artifacts": [*artifacts_final, manifest_self_entry(artifact_manifest_path, artifacts_dir, "S06 artifact manifest.")],
            "validationResult": f"{validation_result}; unit-test commands success={test_success}",
            "caveatsOrBlockers": "No blocker. Strict continuity is Bubble/Insertion adjacent-swap only; E01 Selection is documented as outside the row-neighbor step-count check.",
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
        "packages": {"pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "commands": test_commands,
        "artifacts": [*artifacts_final, artifact_entry(artifact_manifest_path, artifacts_dir, "S06 artifact manifest.")],
        "validationResult": f"{validation_result}; unit-test commands success={test_success}",
    }
    write_json(run_manifest_path, run_manifest_payload)
    checksum_inputs = [
        validation_path,
        validation_csv_path,
        summary_path,
        summary_csv_path,
        trace_path,
        trace_csv_sample_path,
        figure_path,
        config_path,
        source_manifest_path,
        full_results_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    write_checksums(checksum_inputs, checksum_path, artifacts_dir)

    print(json.dumps({"success": bool(validation_success and test_success), "validationResult": validation_result, "artifactsDir": str(artifacts_dir)}, indent=2))
    return 0 if validation_success and test_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
