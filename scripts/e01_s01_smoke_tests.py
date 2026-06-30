#!/usr/bin/env python3
"""Run bounded S01 smoke tests against the cloned paper repository."""

from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import os
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_ID = "S01"
EXPERIMENT_ID = "E01"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nondecreasing(values: list[int]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def traditional_bubble(values: list[int]) -> dict[str, Any]:
    arr = list(values)
    swaps = 0
    comparisons = 0
    trace = [list(arr)]
    for end in range(len(arr) - 1, 0, -1):
        for i in range(end):
            comparisons += 1
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swaps += 1
                trace.append(list(arr))
    return {"mode": "traditional_reference", "algorithm": "bubble", "finalValues": arr, "swaps": swaps, "comparisons": comparisons, "trace": trace}


def traditional_insertion(values: list[int]) -> dict[str, Any]:
    arr = list(values)
    swaps = 0
    comparisons = 0
    trace = [list(arr)]
    for i in range(1, len(arr)):
        j = i
        while j > 0:
            comparisons += 1
            if arr[j - 1] <= arr[j]:
                break
            arr[j - 1], arr[j] = arr[j], arr[j - 1]
            swaps += 1
            trace.append(list(arr))
            j -= 1
    return {"mode": "traditional_reference", "algorithm": "insertion", "finalValues": arr, "swaps": swaps, "comparisons": comparisons, "trace": trace}


def traditional_selection(values: list[int]) -> dict[str, Any]:
    arr = list(values)
    swaps = 0
    comparisons = 0
    trace = [list(arr)]
    for i in range(len(arr)):
        min_index = i
        for j in range(i + 1, len(arr)):
            comparisons += 1
            if arr[j] < arr[min_index]:
                min_index = j
        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]
            swaps += 1
            trace.append(list(arr))
    return {"mode": "traditional_reference", "algorithm": "selection", "finalValues": arr, "swaps": swaps, "comparisons": comparisons, "trace": trace}


def import_core_modules(repo_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo_dir))
    modules: dict[str, Any] = {}
    imports = {
        "StatusProbe": "modules.multithread.StatusProbe",
        "BubbleSortCell": "modules.multithread.BubbleSortCell",
        "InsertionSortCell": "modules.multithread.InsertionSortCell",
        "SelectionSortCell": "modules.multithread.SelectionSortCell",
        "CellGroup": "modules.multithread.CellGroup",
        "MultiThreadCell": "modules.multithread.MultiThreadCell",
        "analysis.utils": "analysis.utils",
    }
    results: list[dict[str, Any]] = []
    for label, module_name in imports.items():
        try:
            module = __import__(module_name, fromlist=["*"])
            modules[label] = module
            results.append({"label": label, "module": module_name, "success": True})
        except Exception as exc:
            results.append({"label": label, "module": module_name, "success": False, "error": repr(exc)})
    return {"success": all(item["success"] for item in results), "results": results, "modules": modules}


def run_cell_view(repo_dir: Path, algorithm: str, values: list[int], timeout_seconds: float, seed: int) -> dict[str, Any]:
    import_result = import_core_modules(repo_dir)
    if not import_result["success"]:
        return {"mode": "cell_view_original", "algorithm": algorithm, "success": False, "error": "core import failed", "imports": import_result["results"]}

    modules = import_result["modules"]
    status_probe = modules["StatusProbe"].StatusProbe()
    BubbleSortCell = modules["BubbleSortCell"].BubbleSortCell
    InsertionSortCell = modules["InsertionSortCell"].InsertionSortCell
    SelectionSortCell = modules["SelectionSortCell"].SelectionSortCell
    CellGroup = modules["CellGroup"].CellGroup
    GroupStatus = modules["CellGroup"].GroupStatus
    CellStatus = modules["MultiThreadCell"].CellStatus

    cls_by_algorithm = {
        "bubble": BubbleSortCell,
        "insertion": InsertionSortCell,
        "selection": SelectionSortCell,
    }
    cls = cls_by_algorithm[algorithm]

    random.seed(seed)
    lock = threading.Lock()
    left_boundary = (0, 1)
    right_boundary = (len(values) - 1, 1)
    cells = []
    for idx, value in enumerate(values):
        cell = cls(
            idx + 1,
            value,
            lock,
            (idx, 1),
            cells,
            left_boundary,
            right_boundary,
            status_probe,
            disable_visualization=True,
        )
        cells.append(cell)
    group = CellGroup(cells, cells, 0, left_boundary, right_boundary, GroupStatus.ACTIVE, lock, 1_000_000, 1_000_000)
    for cell in cells:
        cell.group = group

    started = time.monotonic()
    lock.acquire()
    try:
        for cell in cells:
            cell.start()
        group.start()
    finally:
        lock.release()

    timed_out = False
    while not nondecreasing([cell.value for cell in cells]):
        if time.monotonic() - started > timeout_seconds:
            timed_out = True
            break
        time.sleep(0.005)

    acquired = lock.acquire(timeout=2)
    if acquired:
        try:
            for cell in cells:
                cell.status = CellStatus.INACTIVE
            group.status = GroupStatus.MERGED
        finally:
            lock.release()
    else:
        for cell in cells:
            cell.status = CellStatus.INACTIVE
        group.status = GroupStatus.MERGED

    for cell in cells:
        cell.join(timeout=1)
    group.join(timeout=1)

    final_values = [cell.value for cell in cells]
    alive_threads = [cell.threadID for cell in cells if cell.is_alive()]
    if group.is_alive():
        alive_threads.append("group")
    return {
        "mode": "cell_view_original",
        "algorithm": algorithm,
        "success": nondecreasing(final_values) and not timed_out and not alive_threads,
        "timedOut": timed_out,
        "timeoutSeconds": timeout_seconds,
        "elapsedSeconds": time.monotonic() - started,
        "initialValues": values,
        "finalValues": final_values,
        "sorted": nondecreasing(final_values),
        "swapCount": status_probe.swap_count,
        "compareAndSwapCount": status_probe.compare_and_swap_count,
        "probeSortingStepCount": len(status_probe.sorting_steps),
        "probeCellTypeStepCount": len(status_probe.cell_types),
        "aliveThreadsAfterCleanup": alive_threads,
        "probeSortingSteps": status_probe.sorting_steps,
        "probeCellTypes": status_probe.cell_types,
    }


def scan_script_inventory(repo_dir: Path) -> dict[str, Any]:
    python_files = sorted(path.relative_to(repo_dir).as_posix() for path in repo_dir.rglob("*.py"))
    entrypoints: list[str] = []
    hardcoded_absolute_paths: list[dict[str, Any]] = []
    for rel in python_files:
        path = repo_dir / rel
        text = path.read_text(encoding="utf-8", errors="replace")
        if 'if __name__ == "__main__"' in text or "if __name__ == '__main__'" in text:
            entrypoints.append(rel)
        if "/Users/" in text:
            hardcoded_absolute_paths.append({"path": rel, "containsUsersPath": True})
    return {
        "pythonFileCount": len(python_files),
        "pythonFiles": python_files,
        "entrypointFiles": entrypoints,
        "hardcodedAbsolutePathFiles": hardcoded_absolute_paths,
    }


def static_network_import_scan(repo_dir: Path) -> list[dict[str, Any]]:
    network_markers = ("requests", "urllib", "httpx", "socket", "ftplib")
    hits: list[dict[str, Any]] = []
    for path in sorted(repo_dir.rglob("*.py")):
        rel = path.relative_to(repo_dir).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and any(marker in stripped for marker in network_markers):
                hits.append({"path": rel, "line": lineno, "text": stripped})
    return hits


def compile_repo(repo_dir: Path) -> dict[str, Any]:
    ok = compileall.compile_dir(str(repo_dir), quiet=1, force=True, workers=1)
    return {"success": bool(ok), "method": "compileall.compile_dir", "workers": 1}


def update_run_manifest(artifacts_dir: Path, additions: dict[str, Any], checksums: dict[str, str]) -> None:
    path = artifacts_dir / "run_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema": "eidosoma.e01.run_manifest.v1", "experimentId": EXPERIMENT_ID, "researchStepId": STEP_ID}
    manifest["updatedAtUtc"] = utc_now()
    manifest["status"] = "s01_smoke_tests_complete" if additions.get("success") else "s01_smoke_tests_failed"
    manifest["smokeTests"] = additions
    manifest.setdefault("checksums", {}).update(checksums)
    write_json(path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default="/cache/e01_s01/patched_repo")
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20240630)
    args = parser.parse_args()

    started = time.time()
    repo_dir = Path(args.repo_dir)
    artifacts_dir = Path(args.artifacts_dir)
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    logs_dir = artifacts_dir / "logs"
    step_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "e01_s01_smoke_tests.log"

    if not repo_dir.exists():
        raise FileNotFoundError(f"Repository directory not found: {repo_dir}")

    values = [5, 2, 4, 1, 3, 0]
    inventory = scan_script_inventory(repo_dir)
    compile_result = compile_repo(repo_dir)
    import_result = import_core_modules(repo_dir)
    traditional_results = [
        traditional_bubble(values),
        traditional_insertion(values),
        traditional_selection(values),
    ]
    for item in traditional_results:
        item["success"] = nondecreasing(item["finalValues"])
        item["traceLength"] = len(item.pop("trace"))

    cell_view_results = [
        run_cell_view(repo_dir, "bubble", values, args.timeout_seconds, args.seed + 1),
        run_cell_view(repo_dir, "insertion", values, args.timeout_seconds, args.seed + 2),
        run_cell_view(repo_dir, "selection", values, args.timeout_seconds, args.seed + 3),
    ]

    import numpy as np

    probe_source = next((item for item in cell_view_results if item["algorithm"] == "bubble"), None)
    probe_path = step_dir / "probe_output_smoke.npy"
    probe_read_result: dict[str, Any]
    if probe_source and probe_source.get("probeSortingSteps"):
        np.save(probe_path, np.array(probe_source["probeSortingSteps"], dtype=object))
        loaded = np.load(probe_path, allow_pickle=True)
        probe_read_result = {
            "success": loaded.shape[0] == len(probe_source["probeSortingSteps"]),
            "path": str(probe_path),
            "loadedShape": list(loaded.shape),
        }
    else:
        np.save(probe_path, np.array([], dtype=object))
        loaded = np.load(probe_path, allow_pickle=True)
        probe_read_result = {
            "success": loaded.shape[0] == 0,
            "path": str(probe_path),
            "loadedShape": list(loaded.shape),
            "note": "Bubble run emitted no swap snapshots; empty Probe file still round-tripped.",
        }

    for item in cell_view_results:
        item.pop("probeSortingSteps", None)
        item.pop("probeCellTypes", None)

    network_scan = static_network_import_scan(repo_dir)
    hidden_network_result = {
        "success": len(network_scan) == 0,
        "networkImportHits": network_scan,
        "note": "No network APIs were imported by repository Python files during static scan." if len(network_scan) == 0 else "Review listed imports before disconnected execution.",
    }

    results: dict[str, Any] = {
        "schema": "eidosoma.e01_s01.smoke_results.v1",
        "researchStepId": STEP_ID,
        "experimentId": EXPERIMENT_ID,
        "createdAtUtc": utc_now(),
        "repoDir": str(repo_dir),
        "seed": args.seed,
        "timeoutSeconds": args.timeout_seconds,
        "scriptInventory": inventory,
        "compileCheck": compile_result,
        "importCheck": {"success": import_result["success"], "results": import_result["results"]},
        "traditionalReferenceSmokeTests": traditional_results,
        "cellViewOriginalSmokeTests": cell_view_results,
        "probeReadCheck": probe_read_result,
        "hiddenNetworkDependencyCheck": hidden_network_result,
        "notes": [
            "Traditional smoke tests are bounded reference implementations in this harness because no standalone traditional-generation runner was identifiable in the public repository during S01.",
            "Cell-view smoke tests instantiate original repository cell classes directly and terminate threads explicitly after sorted or timeout.",
            "Full paper-scale scripts were not executed in S01 because several are configured as long 50/100-repeat sweeps with hard-coded output paths; mapping and wrapping are deferred to S02/S03 as planned.",
        ],
        "wallTimeSeconds": time.time() - started,
    }
    results["success"] = bool(
        compile_result["success"]
        and import_result["success"]
        and all(item["success"] for item in traditional_results)
        and all(item["success"] for item in cell_view_results)
        and probe_read_result["success"]
        and hidden_network_result["success"]
    )

    smoke_results_path = step_dir / "smoke_results.json"
    write_json(smoke_results_path, results)

    log_lines = [
        f"S01 smoke tests completed at {utc_now()}",
        f"Repository: {repo_dir}",
        f"Overall success: {results['success']}",
        f"Compile check: {compile_result['success']}",
        f"Import check: {import_result['success']}",
        f"Traditional reference checks: {[item['success'] for item in traditional_results]}",
        f"Cell-view original checks: {[item['algorithm'] + '=' + str(item['success']) for item in cell_view_results]}",
        f"Probe read check: {probe_read_result}",
        f"Hidden network dependency check: {hidden_network_result['success']}",
        f"Python files: {inventory['pythonFileCount']}",
        f"Entrypoint files: {inventory['entrypointFiles']}",
        f"Hard-coded /Users path files: {[item['path'] for item in inventory['hardcodedAbsolutePathFiles']]}",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    checksums = {
        str(smoke_results_path): sha256_file(smoke_results_path),
        str(probe_path): sha256_file(probe_path),
        str(log_path): sha256_file(log_path),
    }
    update_run_manifest(artifacts_dir, results, checksums)
    return 0 if results["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
