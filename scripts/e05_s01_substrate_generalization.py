#!/usr/bin/env python3
"""Run E05 S01 substrate-generalization validation and write artifacts."""

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
    CellState,
    ClockwiseCrawlPolicy,
    GreedyLowerValueSwapPolicy,
    MorphologyWorld,
    MoveProposal,
    Substrate,
    SUBSTRATE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    run_local_dynamics,
    validate_trace_schema,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S01"
STEP_NUMBER = 1
STEP_TITLE = "Generalize the substrate"
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


def build_substrate_worlds() -> list[tuple[str, Substrate, MorphologyWorld, str]]:
    row = Substrate.row(5)
    row_world = MorphologyWorld.from_position_values(
        row,
        {(0,): 4, (1,): 1, (2,): 3, (3,): 2, (4,): 5},
        condition_id="s01_row_swap",
    )

    square = Substrate.square_grid(3, 2)
    square_world = MorphologyWorld.from_position_values(
        square,
        {(0, 0): 6, (1, 0): 1, (2, 0): 5, (0, 1): 2, (1, 1): 4, (2, 1): 3},
        condition_id="s01_square_swap",
    )

    hex_grid = Substrate.hex_grid(3, 2)
    hex_world = MorphologyWorld.from_position_values(
        hex_grid,
        {(0, 0): 4, (1, 0): 1, (2, 0): 6, (0, 1): 2, (1, 1): 5, (2, 1): 3},
        condition_id="s01_hex_swap",
    )

    irregular = Substrate.irregular_graph(
        [((0,), (1,)), ((1,), (2,)), ((1,), (3,)), ((3,), (4,)), ((2,), (4,))],
        nodes=[(0,), (1,), (2,), (3,), (4,), (5,)],
    )
    irregular_cells = {
        0: CellState(0, 5),
        1: CellState(1, 1),
        2: CellState(2, 4),
        3: CellState(3, 2),
        4: CellState(4, 3),
    }
    irregular_world = MorphologyWorld(
        irregular,
        irregular_cells,
        {(0,): 0, (1,): 1, (2,): 2, (3,): 3, (4,): 4},
        condition_id="s01_irregular_crawl_and_swap",
    )

    lattice = Substrate.lattice3d(2, 2, 2)
    lattice_world = MorphologyWorld.from_position_values(
        lattice,
        {
            (0, 0, 0): 8,
            (1, 0, 0): 1,
            (0, 1, 0): 7,
            (1, 1, 0): 2,
            (0, 0, 1): 6,
            (1, 0, 1): 3,
            (0, 1, 1): 5,
            (1, 1, 1): 4,
        },
        condition_id="s01_lattice3d_optional_swap",
    )

    return [
        ("row_1d", row, row_world, "swap"),
        ("square_grid_2d", square, square_world, "swap"),
        ("hex_grid_2d", hex_grid, hex_world, "swap"),
        ("irregular_graph", irregular, irregular_world, "crawl_then_swap"),
        ("lattice_3d_optional", lattice, lattice_world, "swap"),
    ]


def validate_world(label: str, substrate: Substrate, world: MorphologyWorld, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    initial_counter = world.cell_id_counter()
    initial_value_counter = world.value_counter()
    initial_occupied = world.occupied_count()
    if mode == "crawl_then_swap":
        crawl_outcomes = run_local_dynamics(world, ClockwiseCrawlPolicy(), steps=3, seed=11, actor_order=[0, 1, 2])
        swap_outcomes = run_local_dynamics(world, GreedyLowerValueSwapPolicy(), steps=6, seed=12)
        outcomes = crawl_outcomes + swap_outcomes
    else:
        outcomes = run_local_dynamics(world, GreedyLowerValueSwapPolicy(), steps=8, seed=13)
    trace_errors = validate_trace_schema(world.trace_rows)
    occupancy_errors = world.validate_occupancy()
    validation = {
        "research_step_id": STEP_ID,
        "substrate_label": label,
        "substrate_type": substrate.substrate_type,
        "node_count": len(substrate.nodes),
        "edge_count": len(substrate.edges()),
        "boundary": substrate.boundary,
        "adjacency_consistent": len(substrate.validate()) == 0,
        "initial_occupied_count": initial_occupied,
        "final_occupied_count": world.occupied_count(),
        "cell_id_conserved": initial_counter == world.cell_id_counter(),
        "identity_value_conserved": initial_value_counter == world.value_counter(),
        "legal_moves_only": all(outcome.legal for outcome in outcomes),
        "accepted_non_wait_moves": sum(1 for outcome in outcomes if outcome.accepted and outcome.action in {"swap", "crawl"}),
        "trace_rows": len(world.trace_rows),
        "trace_schema_valid": len(trace_errors) == 0,
        "occupancy_valid": len(occupancy_errors) == 0,
        "validation_errors": "; ".join(trace_errors + occupancy_errors),
    }
    return validation, world.trace_rows


def substrate_spec_markdown() -> str:
    return f"""# E05 S01 Substrate Abstraction Spec

Research step ID: {STEP_ID}
Title: {STEP_TITLE}
Substrate schema version: `{SUBSTRATE_SCHEMA_VERSION}`
Trace schema version: `{TRACE_SCHEMA_VERSION}`

## Contract

- A substrate is an undirected graph with stable node positions, adjacency, boundary label, optional geometry metadata, and a schema version.
- A one-dimensional row is represented as a graph over `(x,)` positions, so the E01/E02 adjacent-swap world is a degenerate substrate.
- Square grids use open four-neighbor von Neumann adjacency over `(x, y)` positions.
- Hex grids use open axial coordinates `(q, r)` and six local neighbor directions.
- Irregular graphs use explicit undirected edges and may include empty nodes for crawl moves.
- The optional 3D lattice uses six-neighbor cubic adjacency over `(x, y, z)` positions.

## Local Policy Boundary

Policies receive a `LocalObservation` containing the actor cell, actor position, boundary label, and neighbor views. Each neighbor view includes only adjacent position, relative offset, occupancy, and the neighbor cell payload when occupied. S01 does not expose target morphology, global gradients, or whole-world state through this policy boundary.

## Move Semantics

- `wait`: always legal and preserves state.
- `swap`: legal only when the target is adjacent and occupied; the two cell IDs exchange positions.
- `crawl`: legal only when the target is adjacent and empty; the actor cell moves into the target and leaves its source empty.

All accepted S01 actions preserve the set of cell IDs and identity payloads. Future E05 steps may add non-conservative actions such as division or death, but those are intentionally outside S01.

## Trace Schema

Each trace row records `condition_id`, `step_index`, `event_kind`, `substrate_type`, node and edge counts, occupancy and cell counts, action, actor and target fields, legality and acceptance flags, reason, occupancy hash, and cell-identity hash.
"""


def write_outputs(artifacts_dir: Path) -> dict[str, Any]:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    code_index_dir = artifacts_dir / "code" / "e05_morphospace_simulator"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    code_index_dir.mkdir(parents=True, exist_ok=True)

    validation_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    substrate_rows: list[dict[str, Any]] = []
    for label, substrate, world, mode in build_substrate_worlds():
        validation, traces = validate_world(label, substrate, world, mode)
        validation_rows.append(validation)
        trace_rows.extend(traces)
        substrate_record = substrate.to_record()
        substrate_record["research_step_id"] = STEP_ID
        substrate_record["substrate_label"] = label
        substrate_rows.append(substrate_record)

    validation_df = pd.DataFrame(validation_rows)
    traces_df = pd.DataFrame(trace_rows)
    substrate_df = pd.DataFrame(substrate_rows)

    artifact_paths: list[Path] = []
    artifact_paths.extend(write_dataframe(validation_df, step_dir / "substrate_validation_results"))
    artifact_paths.extend(write_dataframe(validation_df, results_dir / "e05_s01_substrate_validation"))
    artifact_paths.extend(write_dataframe(traces_df, step_dir / "toy_trace_events"))
    artifact_paths.extend(write_dataframe(substrate_df, step_dir / "substrate_specs"))

    spec_path = step_dir / "substrate_abstraction_spec.md"
    spec_path.write_text(substrate_spec_markdown(), encoding="utf-8")
    artifact_paths.append(spec_path)

    code_index_path = code_index_dir / "README.md"
    code_index_path.write_text(
        "# E05 Morphospace Simulator Code Index\n\n"
        "Repository-backed source for S01 is kept in git per workspace instructions.\n\n"
        "- Package: `morphospace2d/`\n"
        "- Runner: `scripts/e05_s01_substrate_generalization.py`\n"
        "- Tests: `tests/test_e05_substrate_generalization.py`\n",
        encoding="utf-8",
    )
    artifact_paths.append(code_index_path)

    success = bool(
        validation_df["adjacency_consistent"].all()
        and validation_df["cell_id_conserved"].all()
        and validation_df["identity_value_conserved"].all()
        and validation_df["legal_moves_only"].all()
        and validation_df["trace_schema_valid"].all()
        and validation_df["occupancy_valid"].all()
        and (validation_df["accepted_non_wait_moves"] > 0).all()
    )
    unit_log_path = step_dir / "repo_unit_test_log.txt"
    unit_test_success = unit_log_path.exists() and "OK" in unit_log_path.read_text(encoding="utf-8", errors="replace")
    if unit_log_path.exists():
        artifact_paths.append(unit_log_path)
    status_path = step_dir / "status.json"
    summary_path = step_dir / "summary.md"
    manifest_path = step_dir / "artifact_manifest.json"
    reported_artifact_paths = artifact_paths + [status_path, summary_path, manifest_path]
    validation_result = (
        "Passed: 1D row, square grid, hex grid, irregular graph, and optional 3D lattice toy simulations "
        "had symmetric adjacency, legal local moves, occupancy conservation, identity conservation, and valid trace schema"
        + ("; focused repository unit tests passed." if unit_test_success else ".")
        if success
        else "Failed: at least one substrate validation invariant did not pass."
    )
    caveats = (
        "S01 validates substrate mechanics only. Identity vectors, target morphologies, higher-dimensional metrics, "
        "repair tasks, and biological interpretation remain deferred to S02 and later steps."
    )
    recommended = "Proceed to S02 to generalize scalar cell value into higher-dimensional cell identity vectors."

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
                "args": "PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_e05_substrate_generalization",
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

    summary_rows = [
        [
            row["substrate_label"],
            row["node_count"],
            row["edge_count"],
            row["accepted_non_wait_moves"],
            row["cell_id_conserved"],
            row["legal_moves_only"],
            row["trace_schema_valid"],
        ]
        for row in validation_rows
    ]
    summary = f"""# E05 S01 Status Summary

- Step ID: {STEP_ID}
- Completion status: {'completed' if success else 'completed with validation failures'}
- Artifacts written: {', '.join(str(path) for path in reported_artifact_paths)}
- Validation result: {validation_result}
- Outcome classification: {'supportive' if success else 'constraining/contradictory'}
- Caveats or blockers: {caveats}
- Lay summary: The simulator now treats rows, grids, hex grids, irregular graphs, and a small 3D lattice as the same kind of local-neighborhood world. Toy cells can swap or crawl only to adjacent locations, and the validation checks confirm that no cells are lost, duplicated, or moved through illegal edges.
- Recommended next action: {recommended}

## Validation Table

{markdown_table(['substrate', 'nodes', 'edges', 'accepted moves', 'cell IDs conserved', 'legal moves only', 'trace schema valid'], summary_rows)}
"""
    summary_path.write_text(summary, encoding="utf-8")

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": "eidosoma.step_artifact_manifest.v1",
        "generatedAt": utc_now(),
        "artifacts": [
            {
                "path": str(path),
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths + [status_path, summary_path]
            if path.exists() and path.is_file()
        ],
        "sourceCode": {
            "repositoryPackage": "morphospace2d/",
            "runner": "scripts/e05_s01_substrate_generalization.py",
            "tests": "tests/test_e05_substrate_generalization.py",
            "note": "Repository source is committed in git and not copied into artifacts per GitHub workspace rule.",
        },
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended,
    }
    write_json(manifest_path, manifest_payload)

    return {
        "success": success,
        "stepDir": step_dir,
        "status": status_payload,
        "artifactPaths": [str(path) for path in reported_artifact_paths],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    result = write_outputs(args.artifacts_dir)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
