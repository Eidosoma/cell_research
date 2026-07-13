#!/usr/bin/env python3
"""Source-external runner for the frozen public thread-per-cell backend.

This module never rewrites the historical tree.  It verifies every frozen file
against the S02 checksum ledger before importing the original policy classes,
then supplies only the small amount of driver glue needed for bounded smoke
execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import stat
import sys
import threading
import time
from typing import Any, Iterable, Sequence


FROZEN_PUBLIC_COMMIT = "1fd2bd5921c1f6b423a71f691d5189106a8a1020"
FROZEN_PUBLIC_TREE = "42c1ec7fba45f2f236dbdde76438f59a0c1a4920"
GROUP_PERIOD = 100_000_000
POLL_INTERVAL_SECONDS = 0.0001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksum_manifest(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            expected, relative_name = raw_line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line {line_number}: {raw_line!r}") from exc
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe checksum path at line {line_number}: {relative_name!r}")
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(f"invalid SHA-256 at line {line_number}: {expected!r}")
        entries.append((expected, relative_name))
    if not entries:
        raise ValueError(f"checksum manifest is empty: {path}")
    return entries


def verify_source_tree(source_root: Path, checksum_manifest: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    checksum_manifest = checksum_manifest.resolve()
    entries = read_checksum_manifest(checksum_manifest)
    missing: list[str] = []
    mismatches: list[dict[str, str]] = []
    writable_files: list[str] = []
    mode_writable_files: list[str] = []

    for expected, relative_name in entries:
        candidate = source_root / relative_name
        if not candidate.is_file():
            missing.append(relative_name)
            continue
        actual = sha256_file(candidate)
        if actual != expected:
            mismatches.append(
                {"path": relative_name, "expectedSha256": expected, "actualSha256": actual}
            )
        if os.access(candidate, os.W_OK):
            writable_files.append(relative_name)
        if candidate.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            mode_writable_files.append(relative_name)

    result = {
        "sourceRoot": str(source_root),
        "checksumManifest": str(checksum_manifest),
        "expectedCommit": FROZEN_PUBLIC_COMMIT,
        "expectedTree": FROZEN_PUBLIC_TREE,
        "filesChecked": len(entries),
        "missing": missing,
        "mismatches": mismatches,
        "writableFilesAsCurrentUser": writable_files,
        "modeWritableFiles": mode_writable_files,
        "contentValid": not missing and not mismatches,
    }
    return result


def _import_historical_classes(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    from modules.multithread.BubbleSortCell import BubbleSortCell
    from modules.multithread.CellGroup import CellGroup, GroupStatus
    from modules.multithread.InsertionSortCell import InsertionSortCell
    from modules.multithread.MultiThreadCell import CellStatus
    from modules.multithread.SelectionSortCell import SelectionSortCell
    from modules.multithread.StatusProbe import StatusProbe

    imported = {
        "bubble": BubbleSortCell,
        "insertion": InsertionSortCell,
        "selection": SelectionSortCell,
        "CellGroup": CellGroup,
        "GroupStatus": GroupStatus,
        "CellStatus": CellStatus,
        "StatusProbe": StatusProbe,
    }
    for klass in imported.values():
        module = sys.modules[klass.__module__]
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(source_root):
            raise RuntimeError(f"historical import escaped frozen tree: {module_path}")
    return imported


def _is_sorted(cells: Sequence[Any]) -> bool:
    """Exact non-strict no-fault stopping predicate from the frozen driver."""
    previous = cells[0]
    for cell in cells:
        if cell.value < previous.value:
            return False
        previous = cell
    return True


def _no_cells_should_move(cells: Sequence[Any], cell_status: Any) -> bool:
    """Exact passive-fault stopping predicate from the frozen driver."""
    for cell in cells:
        if cell.status == cell_status.SLEEP:
            return False
        if cell.status == cell_status.ACTIVE and cell.should_move():
            return False
    return True


def _create_cells(
    values: Sequence[int],
    policy: str,
    lock: threading.Lock,
    imported: dict[str, Any],
    frozen_indices: Iterable[int],
) -> tuple[list[Any], list[Any], Any]:
    if not values:
        raise ValueError("the historical driver does not define an empty-array run")
    left_boundary = (0, 1)
    right_boundary = (len(values) - 1, 1)
    cells: list[Any] = []
    probe = imported["StatusProbe"]()
    cell_class = imported[policy]

    # Constructor calls intentionally retain the historical default shared
    # swapping_count/export_steps objects; passing replacements would alter C.
    for index, value in enumerate(values):
        cell = cell_class(
            index + 1,
            value,
            lock,
            (index, 1),
            cells,
            left_boundary,
            right_boundary,
            probe,
            disable_visualization=True,
        )
        cells.append(cell)
        if policy == "insertion":
            cells[0].enable_to_move = True

    group = imported["CellGroup"](
        cells,
        cells,
        0,
        left_boundary,
        right_boundary,
        imported["GroupStatus"].ACTIVE,
        lock,
        GROUP_PERIOD,
        GROUP_PERIOD,
    )
    for cell in cells:
        cell.group = group
    for index in frozen_indices:
        if index < 0 or index >= len(cells):
            raise ValueError(f"frozen index {index} is outside [0, {len(cells)})")
        cells[index].set_cell_to_freeze()
    return cells, [group], probe


def _kill(cells: Sequence[Any], groups: Sequence[Any], imported: dict[str, Any]) -> None:
    for cell in cells:
        cell.status = imported["CellStatus"].INACTIVE
    for group in groups:
        group.status = imported["GroupStatus"].MERGED


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_historical_toy(
    *,
    source_root: Path,
    checksum_manifest: Path,
    values: Sequence[int],
    policy: str,
    stop_mode: str = "is_sorted",
    timeout_seconds: float = 10.0,
    seed: int | None = None,
    frozen_indices: Sequence[int] = (),
    frozen_count_with_replacement: int = 0,
) -> dict[str, Any]:
    """Run original classes and return adapter metadata plus original probes."""
    precheck = verify_source_tree(source_root, checksum_manifest)
    if not precheck["contentValid"]:
        raise RuntimeError(f"frozen source verification failed: {precheck}")
    imported = _import_historical_classes(source_root)

    if policy not in {"bubble", "insertion", "selection"}:
        raise ValueError(f"unsupported policy: {policy}")
    if stop_mode not in {"is_sorted", "no_cells_should_move", "no_move_or_15000_swaps"}:
        raise ValueError(f"unsupported stop mode: {stop_mode}")
    if seed is not None:
        random.seed(seed)

    selected_frozen = list(frozen_indices)
    for _ in range(frozen_count_with_replacement):
        selected_frozen.append(random.randint(0, len(values) - 1))

    lock = threading.Lock()
    cells, groups, probe = _create_cells(values, policy, lock, imported, selected_frozen)
    started_cells: list[Any] = []
    start = time.monotonic()
    stop_reason = "adapter_error"
    timed_out = False
    thread_shutdown_clean = False
    try:
        # This start ordering and the main-thread lock match the frozen drivers.
        lock.acquire()
        try:
            for cell in cells:
                if cell.status != imported["CellStatus"].FREEZE:
                    cell.start()
                    started_cells.append(cell)
            for group in groups:
                group.start()
        finally:
            lock.release()

        while True:
            if stop_mode == "is_sorted" and _is_sorted(cells):
                stop_reason = "historical_is_sorted"
                break
            if stop_mode in {"no_cells_should_move", "no_move_or_15000_swaps"}:
                if _no_cells_should_move(cells, imported["CellStatus"]):
                    stop_reason = "historical_no_cells_should_move"
                    break
            if stop_mode == "no_move_or_15000_swaps" and len(probe.sorting_steps) >= 15_000:
                stop_reason = "historical_15000_recorded_swaps"
                break
            if time.monotonic() - start > timeout_seconds:
                stop_reason = "adapter_safety_timeout"
                timed_out = True
                break
            time.sleep(POLL_INTERVAL_SECONDS)
    finally:
        with lock:
            _kill(cells, groups, imported)
        for thread in [*started_cells, *groups]:
            if thread.ident is not None:
                thread.join(timeout=2.0)
        thread_shutdown_clean = all(not thread.is_alive() for thread in [*started_cells, *groups])

    elapsed = time.monotonic() - start
    final_values = [cell.value for cell in cells]
    steps = [list(step) for step in probe.sorting_steps]
    cell_types = probe.cell_types
    return {
        "backendProfile": "C-frozen-public-commit",
        "publicCommit": FROZEN_PUBLIC_COMMIT,
        "publicTree": FROZEN_PUBLIC_TREE,
        "publicationSnapshotClaimed": False,
        "policy": policy,
        "inputValues": list(values),
        "finalValues": final_values,
        "frozenIndicesRequested": list(frozen_indices),
        "frozenIndicesAfterWithReplacementDraws": selected_frozen,
        "stopMode": stop_mode,
        "stopReason": stop_reason,
        "adapterSafetyTimeoutSeconds": timeout_seconds,
        "timedOut": timed_out,
        "seedOverride": seed,
        "metrics": {
            "swapCount": probe.swap_count,
            "compareAndSwapCount": probe.compare_and_swap_count,
            "frozenSwapAttempts": probe.frozen_swap_attempts,
            "recordedSortingSteps": len(steps),
            "recordedCellTypeSteps": len(cell_types),
        },
        "sortingSteps": steps,
        "cellTypes": cell_types,
        "traceSha256": _json_hash(steps),
        "valueMultisetConserved": sorted(values) == sorted(final_values),
        "nonStrictlySortedAtShutdown": all(
            final_values[index] <= final_values[index + 1]
            for index in range(len(final_values) - 1)
        ),
        "threadShutdownClean": thread_shutdown_clean,
        "elapsedSeconds": elapsed,
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "pythonDontWriteBytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        },
        "sourcePrecheck": precheck,
    }


def write_run_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write adapter JSON and raw arrays without claiming they are historical data."""
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sorting_path = output_dir / "sorting_steps.npy"
    cell_types_path = output_dir / "cell_types.npy"
    np.save(sorting_path, np.asarray(result["sortingSteps"], dtype=np.int64), allow_pickle=False)
    np.save(cell_types_path, np.asarray(result["cellTypes"], dtype=np.int64), allow_pickle=False)
    return {
        "runJson": str(summary_path),
        "sortingStepsNpy": str(sorting_path),
        "cellTypesNpy": str(cell_types_path),
        "sha256": {
            summary_path.name: sha256_file(summary_path),
            sorting_path.name: sha256_file(sorting_path),
            cell_types_path.name: sha256_file(cell_types_path),
        },
    }


def _parse_int_list(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(item.strip()) for item in raw.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="verify frozen source bytes")
    verify_parser.add_argument("--output", type=Path)

    run_parser = subparsers.add_parser("run", help="execute one bounded historical toy")
    run_parser.add_argument("--policy", choices=["bubble", "insertion", "selection"], required=True)
    run_parser.add_argument("--values", required=True, help="comma-separated integer values")
    run_parser.add_argument(
        "--stop-mode",
        choices=["is_sorted", "no_cells_should_move", "no_move_or_15000_swaps"],
        default="is_sorted",
    )
    run_parser.add_argument("--timeout-seconds", type=float, default=10.0)
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--frozen-indices", default="")
    run_parser.add_argument("--frozen-count-with-replacement", type=int, default=0)
    run_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        result = verify_source_tree(args.source, args.checksums)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        return 0 if result["contentValid"] else 1

    result = run_historical_toy(
        source_root=args.source,
        checksum_manifest=args.checksums,
        values=_parse_int_list(args.values),
        policy=args.policy,
        stop_mode=args.stop_mode,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
        frozen_indices=_parse_int_list(args.frozen_indices),
        frozen_count_with_replacement=args.frozen_count_with_replacement,
    )
    if args.output:
        outputs = write_run_outputs(result, args.output)
        sys.stdout.write(json.dumps({"result": result, "outputs": outputs}, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if not result["timedOut"] and result["threadShutdownClean"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
