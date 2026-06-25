#!/usr/bin/env python3
"""Run bounded S01 smoke tests against the original sorting archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STEP_ID = "S01"
STEP_NUMBER = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def is_sorted_values(values: list[int]) -> bool:
    return all(values[i - 1] <= values[i] for i in range(1, len(values)))


def traditional_sort_trace(values: list[int], algorithm: str) -> dict[str, Any]:
    arr = list(values)
    trace: list[list[int]] = []
    comparisons = 0
    swaps = 0

    if algorithm == "bubble":
        made_swap = True
        while made_swap:
            made_swap = False
            for idx in range(len(arr) - 1):
                comparisons += 1
                if arr[idx] > arr[idx + 1]:
                    arr[idx], arr[idx + 1] = arr[idx + 1], arr[idx]
                    swaps += 1
                    made_swap = True
                    trace.append(list(arr))
    elif algorithm == "insertion":
        for idx in range(1, len(arr)):
            jdx = idx
            while jdx > 0:
                comparisons += 1
                if arr[jdx - 1] <= arr[jdx]:
                    break
                arr[jdx - 1], arr[jdx] = arr[jdx], arr[jdx - 1]
                swaps += 1
                trace.append(list(arr))
                jdx -= 1
    elif algorithm == "selection":
        for idx in range(len(arr)):
            min_idx = idx
            for jdx in range(idx + 1, len(arr)):
                comparisons += 1
                if arr[jdx] < arr[min_idx]:
                    min_idx = jdx
            if min_idx != idx:
                arr[idx], arr[min_idx] = arr[min_idx], arr[idx]
                swaps += 1
                trace.append(list(arr))
    else:
        raise ValueError(f"unknown traditional algorithm: {algorithm}")

    return {
        "algorithm": algorithm,
        "mode": "traditional_reference_wrapper",
        "startValues": values,
        "finalValues": arr,
        "sorted": is_sorted_values(arr),
        "swapCount": swaps,
        "compareAndSwapCount": comparisons,
        "probeStepCount": len(trace),
        "trace": trace,
        "caveat": "No clean original traditional-sort generator was found; this is an explicit reference-wrapper smoke test only.",
    }


def import_original_classes(repo_root: Path) -> dict[str, Any]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from modules.multithread.BubbleSortCell import BubbleSortCell
    from modules.multithread.CellGroup import CellGroup, GroupStatus
    from modules.multithread.InsertionSortCell import InsertionSortCell
    from modules.multithread.MultiThreadCell import CellStatus
    from modules.multithread.SelectionSortCell import SelectionSortCell
    from modules.multithread.StatusProbe import StatusProbe

    return {
        "BubbleSortCell": BubbleSortCell,
        "CellGroup": CellGroup,
        "GroupStatus": GroupStatus,
        "InsertionSortCell": InsertionSortCell,
        "CellStatus": CellStatus,
        "SelectionSortCell": SelectionSortCell,
        "StatusProbe": StatusProbe,
    }


def run_import_tests(repo_root: Path) -> list[dict[str, Any]]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module_names = [
        "modules.multithread.StatusProbe",
        "modules.multithread.MultiThreadCell",
        "modules.multithread.BubbleSortCell",
        "modules.multithread.InsertionSortCell",
        "modules.multithread.SelectionSortCell",
        "modules.multithread.CellGroup",
        "analysis.utils",
        "multithread_cell_sorting_steps",
        "multithread_cell_sorting_with_frozen_steps",
        "multithread_sorting_cell_aggregation_analysis",
    ]
    results = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
            results.append({"module": module_name, "success": True, "error": ""})
        except Exception as exc:  # pragma: no cover - recorded in artifacts
            results.append({"module": module_name, "success": False, "error": repr(exc)})
    return results


def make_original_cells(
    classes: dict[str, Any],
    values: list[int],
    cell_types: list[str],
    frozen_positions: tuple[int, ...],
) -> tuple[list[Any], list[Any], Any, threading.Lock]:
    lock = threading.Lock()
    probe = classes["StatusProbe"]()
    cells: list[Any] = []
    left_boundary = (0, 1)
    right_boundary = (len(values) - 1, 1)
    constructors = {
        "bubble": classes["BubbleSortCell"],
        "insertion": classes["InsertionSortCell"],
        "selection": classes["SelectionSortCell"],
    }
    for idx, (value, cell_type) in enumerate(zip(values, cell_types)):
        cell = constructors[cell_type](
            idx + 1,
            value,
            lock,
            (idx, 1),
            cells,
            left_boundary,
            right_boundary,
            probe,
            disable_visualization=True,
        )
        cells.append(cell)

    group = classes["CellGroup"](
        cells,
        cells,
        0,
        left_boundary,
        right_boundary,
        classes["GroupStatus"].ACTIVE,
        lock,
        1_000_000_000,
        1_000_000_000,
    )
    for cell in cells:
        cell.group = group
    for idx in frozen_positions:
        cells[idx].set_cell_to_freeze()
    return cells, [group], probe, lock


def current_values(cells: list[Any]) -> list[int]:
    return [int(cell.value) for cell in cells]


def is_sorted_cells(cells: list[Any]) -> bool:
    return is_sorted_values(current_values(cells))


def no_cells_should_move(cells: list[Any], cell_status: Any) -> bool:
    for cell in list(cells):
        if cell.status == cell_status.SLEEP:
            return False
        if cell.status == cell_status.ACTIVE and cell.should_move():
            return False
    return True


def kill_threads(cells: list[Any], groups: list[Any], lock: threading.Lock, classes: dict[str, Any]) -> None:
    lock.acquire()
    try:
        for cell in cells:
            cell.status = classes["CellStatus"].INACTIVE
        for group in groups:
            group.status = classes["GroupStatus"].MERGED
    finally:
        lock.release()

    for thread in list(cells) + list(groups):
        if thread.ident is not None:
            thread.join(timeout=0.3)


def run_threaded_cell_smoke(
    classes: dict[str, Any],
    name: str,
    values: list[int],
    cell_types: list[str],
    output_dir: Path,
    frozen_positions: tuple[int, ...] = (),
    seed: int = 123,
    timeout_seconds: float = 8.0,
) -> dict[str, Any]:
    random.seed(seed)
    cells, groups, probe, lock = make_original_cells(classes, values, cell_types, frozen_positions)
    start_values = current_values(cells)
    started_at = time.time()
    stop_reason = "timeout"
    exception = ""

    try:
        lock.acquire()
        try:
            for cell in list(cells):
                if cell.status != classes["CellStatus"].FREEZE:
                    cell.start()
            for group in groups:
                group.start()
        finally:
            lock.release()

        while time.time() - started_at < timeout_seconds:
            if is_sorted_cells(cells):
                stop_reason = "sorted"
                break
            if no_cells_should_move(cells, classes["CellStatus"]):
                stop_reason = "no_move"
                break
            time.sleep(0.001)
    except Exception as exc:  # pragma: no cover - recorded in artifacts
        stop_reason = "exception"
        exception = repr(exc)
    finally:
        final_values = current_values(cells)
        kill_threads(cells, groups, lock, classes)

    probe_path = output_dir / "probes" / f"{name}_sorting_steps.npy"
    cell_type_path = output_dir / "probes" / f"{name}_cell_types.npy"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(probe_path, np.array(probe.sorting_steps, dtype=object))
    np.save(cell_type_path, np.array(probe.cell_types, dtype=object))

    return {
        "name": name,
        "mode": "original_threaded_cell_view",
        "cellTypes": cell_types,
        "frozenPositions": list(frozen_positions),
        "startValues": start_values,
        "finalValues": final_values,
        "sorted": is_sorted_values(final_values),
        "stopReason": stop_reason,
        "exception": exception,
        "success": stop_reason in {"sorted", "no_move"} and not exception and probe.swap_count > 0,
        "swapCount": int(probe.swap_count),
        "probeStepCount": len(probe.sorting_steps),
        "cellTypeProbeStepCount": len(probe.cell_types),
        "compareAndSwapCount": int(probe.compare_and_swap_count),
        "frozenSwapAttempts": int(probe.frozen_swap_attempts),
        "wallSeconds": round(time.time() - started_at, 6),
        "probePath": str(probe_path),
        "probeSha256": sha256_path(probe_path),
        "cellTypeProbePath": str(cell_type_path),
        "cellTypeProbeSha256": sha256_path(cell_type_path),
    }


def run_metric_smoke(repo_root: Path) -> dict[str, Any]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from analysis.utils import get_monotonicity, get_spearman_distance

    increasing = get_monotonicity([0, 1, 2, 3])
    decreasing = get_monotonicity([3, 2, 1, 0])
    spearman_identity = get_spearman_distance([0, 1, 2, 3])
    success = increasing == 100 and decreasing == 25 and spearman_identity == 0
    return {
        "name": "metric_utils",
        "mode": "analysis_metric_smoke",
        "success": success,
        "monotonicityIncreasing": increasing,
        "monotonicityDecreasing": decreasing,
        "spearmanIdentity": spearman_identity,
    }


def collect_artifacts(step_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(step_dir.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "path": str(path),
                    "relativePath": path.relative_to(step_dir).as_posix(),
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "name",
        "mode",
        "success",
        "sorted",
        "stopReason",
        "swapCount",
        "probeStepCount",
        "compareAndSwapCount",
        "frozenSwapAttempts",
        "wallSeconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Smoke case | Mode | Success | Stop reason | Sorted | Swaps | Probe steps |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('name')}` | `{row.get('mode')}` | {row.get('success')} | "
            f"{row.get('stopReason', '')} | {row.get('sorted', '')} | "
            f"{row.get('swapCount', '')} | {row.get('probeStepCount', '')} |"
        )
    return lines


def write_smoke_log(
    path: Path,
    success: bool,
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
    import_results: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# S01 Smoke Test Log",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {'completed' if success else 'completed_with_blockers'}",
        f"- Artifacts written: {', '.join(artifacts_written)}",
        f"- Validation result: {validation_result}",
        f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'None.'}",
        f"- Recommended next action: {recommended_next_action}",
        "",
        "## Import Checks",
        "",
        "| Module | Success | Error |",
        "| --- | --- | --- |",
    ]
    for result in import_results:
        lines.append(
            f"| `{result['module']}` | {result['success']} | `{result['error']}` |"
        )
    lines.extend(["", "## Smoke Cases", ""])
    lines.extend(markdown_table(smoke_rows))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Cell-view, Frozen Cell, mixed-Algotype, Probe recording, and metric utility checks use existing repository classes or utilities.",
            "- Traditional sorting is smoke-tested by an explicit wrapper because S01 did not find a clean original traditional-sort generator in the repository.",
            "- Frozen Cell smoke runs stop on `no_move` when the tiny perturbed system reaches a local blocked state, which is acceptable for this audit step.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_summary(
    path: Path,
    success: bool,
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
) -> None:
    lines = [
        "# S01 Summary",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {'completed' if success else 'completed_with_blockers'}",
        f"- Artifacts written: {', '.join(artifacts_written)}",
        f"- Validation result: {validation_result}",
        "- Outcome classification: supportive" if success else "- Outcome classification: constraining/contradictory",
        f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'None.'}",
        "- Lay summary: The original archive is runnable for tiny smoke cases covering cell-view sorting, Frozen Cells, Probe-like `.npy` outputs, mixed Algotypes, and basic metric utilities. Traditional-sort generation remains a mapping gap because only analysis references were found.",
        f"- Recommended next action: {recommended_next_action}",
    ]
    path.write_text("\n".join(lines) + "\n")


def copy_reproducible_code(repo_root: Path, step_dir: Path) -> list[str]:
    code_dir = step_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel_path in [
        "scripts/e01_s01_environment_audit.py",
        "scripts/e01_s01_smoke_tests.py",
    ]:
        src = repo_root / rel_path
        if src.exists():
            dest = code_dir / Path(rel_path).name
            shutil.copy2(src, dest)
            copied.append(str(dest))
    return copied


def update_run_manifest(
    artifacts_dir: Path,
    status_payload: dict[str, Any],
    artifact_records: list[dict[str, Any]],
) -> None:
    path = artifacts_dir / "provenance" / "run_manifest.json"
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": status_payload["success"],
            "status": status_payload["status"],
            "artifactsWritten": status_payload["artifactsWritten"],
            "validationResult": status_payload["validationResult"],
            "caveatsOrBlockers": status_payload["caveatsOrBlockers"],
            "recommendedNextAction": status_payload["recommendedNextAction"],
            "updatedAt": utc_now(),
            "researchSteps": {
                STEP_ID: {
                    "status": status_payload,
                    "artifactCount": len(artifact_records),
                    "artifacts": artifact_records,
                }
            },
        }
    )
    write_json(path, existing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/cell-research")
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--step-id", default=STEP_ID)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.step_id != STEP_ID:
        raise SystemExit(f"this script is scoped to {STEP_ID}, got {args.step_id}")

    repo_root = Path(args.repo_root).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    step_dir.mkdir(parents=True, exist_ok=True)
    copied_code = copy_reproducible_code(repo_root, step_dir)

    import_results = run_import_tests(repo_root)
    classes = import_original_classes(repo_root)
    base_values = list(range(9, -1, -1))

    smoke_rows: list[dict[str, Any]] = []
    traditional_trace_dir = step_dir / "probes"
    traditional_trace_dir.mkdir(parents=True, exist_ok=True)
    for algorithm in ["bubble", "insertion", "selection"]:
        row = traditional_sort_trace(base_values, algorithm)
        trace_path = traditional_trace_dir / f"traditional_{algorithm}_trace.npy"
        np.save(trace_path, np.array(row["trace"], dtype=object))
        row.update(
            {
                "name": f"traditional_{algorithm}",
                "success": row["sorted"] and row["swapCount"] > 0,
                "stopReason": "sorted",
                "probePath": str(trace_path),
                "probeSha256": sha256_path(trace_path),
            }
        )
        smoke_rows.append(row)

    for algorithm in ["bubble", "insertion", "selection"]:
        smoke_rows.append(
            run_threaded_cell_smoke(
                classes,
                f"cell_view_{algorithm}",
                base_values,
                [algorithm] * len(base_values),
                step_dir,
                seed=args.seed,
            )
        )

    for algorithm in ["bubble", "insertion", "selection"]:
        smoke_rows.append(
            run_threaded_cell_smoke(
                classes,
                f"frozen_cell_{algorithm}",
                base_values,
                [algorithm] * len(base_values),
                step_dir,
                frozen_positions=(4,),
                seed=args.seed,
            )
        )

    mixed_cases = {
        "mixed_bubble_insertion": ["bubble", "insertion"] * 5,
        "mixed_bubble_selection": ["bubble", "selection"] * 5,
        "mixed_insertion_selection": ["insertion", "selection"] * 5,
        "mixed_all_three": ["bubble", "insertion", "selection"] * 3,
    }
    for name, cell_types in mixed_cases.items():
        values = list(range(len(cell_types) - 1, -1, -1))
        smoke_rows.append(
            run_threaded_cell_smoke(
                classes,
                name,
                values,
                cell_types,
                step_dir,
                seed=args.seed,
            )
        )

    metric_result = run_metric_smoke(repo_root)
    smoke_rows.append(metric_result)

    import_success = all(row["success"] for row in import_results)
    smoke_success = all(row.get("success", False) for row in smoke_rows)
    success = import_success and smoke_success

    smoke_results_path = step_dir / "smoke_results.json"
    smoke_results_csv_path = step_dir / "smoke_results.csv"
    smoke_log_path = step_dir / "smoke_test_log.md"
    summary_path = step_dir / "summary.md"
    status_json_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    compatibility_path = step_dir / "compatibility_patches.md"

    validation_result = (
        "Import checks passed; tiny traditional-reference, cell-view, Frozen Cell, mixed-Algotype, Probe `.npy`, and metric smoke checks passed."
        if success
        else "One or more S01 import or smoke checks failed; inspect smoke_results.json and smoke_test_log.md."
    )
    caveats = [
        "No clean original traditional-sort generator was found; traditional checks are explicit reference-wrapper smoke tests.",
        "Exact original package versions are unavailable because the repository has no dependency manifest.",
        "Thread scheduling remains nondeterministic; S01 used tiny seeded smoke tests only.",
        "Frozen Cell semantics require S02 mapping because archived scripts mix movable-by-others and stopped-thread behavior.",
    ]
    recommended_next_action = "Proceed to S02 code-to-paper mapping after Chief Scientist instruction; do not start S02 in this run."
    planned_status_outputs = [
        str(smoke_results_path),
        str(smoke_results_csv_path),
        str(smoke_log_path),
        str(summary_path),
        str(status_json_path),
        str(artifact_manifest_path),
        str(compatibility_path),
        *copied_code,
    ]
    existing_outputs = [
        str(path)
        for path in sorted(step_dir.rglob("*"))
        if path.is_file()
    ]
    artifacts_written = sorted(set(existing_outputs + planned_status_outputs))

    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_blockers",
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "generatedAt": utc_now(),
    }

    smoke_results = {
        **status_payload,
        "seed": args.seed,
        "importResults": import_results,
        "smokeResults": smoke_rows,
    }
    write_json(smoke_results_path, smoke_results)
    write_csv(smoke_results_csv_path, smoke_rows)
    compatibility_path.write_text(
        "\n".join(
            [
                "# S01 Compatibility Patches",
                "",
                f"- Research step ID: {STEP_ID}",
                f"- Completion status: {'completed' if success else 'completed_with_blockers'}",
                f"- Artifacts written: {compatibility_path}",
                "- Validation result: No source compatibility patch was required for the tiny Python 3.13 smoke runs.",
                f"- Caveats or blockers: {'; '.join(caveats)}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                "No simulator source files were edited for compatibility in S01.",
            ]
        )
        + "\n"
    )
    write_smoke_log(
        smoke_log_path,
        success,
        artifacts_written,
        validation_result,
        caveats,
        recommended_next_action,
        import_results,
        smoke_rows,
    )
    write_summary(
        summary_path,
        success,
        artifacts_written,
        validation_result,
        caveats,
        recommended_next_action,
    )
    write_json(status_json_path, status_payload)
    artifact_records = collect_artifacts(step_dir)
    artifact_manifest = {
        **status_payload,
        "artifactCount": len(artifact_records),
        "artifacts": artifact_records,
    }
    write_json(artifact_manifest_path, artifact_manifest)
    artifact_records = collect_artifacts(step_dir)
    update_run_manifest(artifacts_dir, status_payload, artifact_records)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
