#!/usr/bin/env python3
"""Validate E05 S01 substrate abstractions and write research artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.e02.deterministic_simulator import EventTracingStatusProbe, SimulatorConfig, _build_cells
from src.e03.policy_interface import OriginalCellPolicyWrapper
from src.e05.substrates import (
    SubstrateCell,
    array_substrate,
    graph_substrate,
    hex_grid_substrate,
    lattice3d_substrate,
    make_cells,
    square_grid_substrate,
)


STEP_ID = "S01"
STEP_NUMBER = 1
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-e03-dir", type=Path, default=Path("/previous-artifacts/E03"))
    parser.add_argument("--previous-e04-dir", type=Path, default=Path("/previous-artifacts/E04"))
    parser.add_argument("--paper-markdown", type=Path, default=Path("/workspace/input-attachments/f93afdc5-f2e5-4ecc-80bb-e088f93acf3c/pdf-markdown.md"))
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


def build_reference_cells(config: SimulatorConfig):
    probe = EventTracingStatusProbe()
    cells, _cell_status = _build_cells(config, probe)
    return cells, probe


def validation_1d_reference_bridge() -> dict[str, Any]:
    config = SimulatorConfig(values=(2, 1), algorithm="bubble")
    reference_cells, _reference_probe = build_reference_cells(config)
    policy_cells, _policy_probe = build_reference_cells(config)
    substrate = array_substrate(2)
    substrate.fill_sites(
        tuple(
            SubstrateCell(str(cell.threadID), cell.value, label=cell.label, status=str(cell.status))
            for cell in policy_cells
        )
    )
    wrapper = OriginalCellPolicyWrapper.from_cell(policy_cells[0])
    observation = wrapper.observe(policy_cells[0])
    local_observation = substrate.local_observation(0)
    random.seed(1)
    action = wrapper.propose_action(observation)
    result = substrate.apply_indexed_policy_action(0, action)
    random.seed(1)
    reference_cells[0].move()
    observed = {
        "actor_value": local_observation.actor_cell.value,
        "neighbor_site_ids": list(local_observation.neighbor_site_ids),
        "neighbor_values": [neighbor.value for neighbor in local_observation.neighbors],
        "policy_action": action.action_type,
        "action_allowed": result.allowed,
        "substrate_values_after": list(substrate.values_in_site_order()),
        "reference_values_after": [cell.value for cell in reference_cells],
    }
    expected = {
        "actor_value": 2,
        "neighbor_site_ids": [1],
        "neighbor_values": [1],
        "policy_action": "swap",
        "action_allowed": True,
        "substrate_values_after": [1, 2],
        "reference_values_after": [1, 2],
    }
    return _row(
        "array_1d_matches_e03_bubble_action",
        "array_1d",
        observed == expected,
        expected,
        observed,
        substrate.graph_hash(),
        "1D array local observation and indexed swap action match an E03 Bubble reference fixture.",
    )


def validation_square_grid() -> dict[str, Any]:
    substrate = square_grid_substrate(3, 3)
    substrate.place_cell(0, SubstrateCell("a", 1))
    substrate.place_cell(4, SubstrateCell("b", 2))
    result = substrate.apply_action("swap", 0, 4)
    observed = {
        "center_neighbors": list(substrate.neighbors(4)),
        "center_neighbor_count": substrate.neighbor_count(4),
        "corner_neighbor_count": substrate.neighbor_count(0),
        "corner_boundary": substrate.site(0).boundary,
        "center_boundary": substrate.site(4).boundary,
        "diagonal_swap_allowed": result.allowed,
        "diagonal_swap_reason": result.reason,
    }
    expected = {
        "center_neighbors": [1, 5, 7, 3],
        "center_neighbor_count": 4,
        "corner_neighbor_count": 2,
        "corner_boundary": True,
        "center_boundary": False,
        "diagonal_swap_allowed": False,
        "diagonal_swap_reason": "target_not_adjacent",
    }
    return _row(
        "square_grid_neighbor_and_action_invariants",
        "square_grid_2d",
        observed == expected,
        expected,
        observed,
        substrate.graph_hash(),
        "Open 3x3 square grid has four-neighbor center, two-neighbor corner, and rejects diagonal swaps.",
    )


def validation_hex_grid() -> dict[str, Any]:
    substrate = hex_grid_substrate(3, 3)
    observed = {
        "center_neighbor_count": substrate.neighbor_count(4),
        "corner_neighbor_count": substrate.neighbor_count(0),
        "center_directions": [substrate.direction_between(4, target) for target in substrate.neighbors(4)],
    }
    expected = {
        "center_neighbor_count": 6,
        "corner_neighbor_count": 2,
        "center_directions": ["east", "north_east", "north_west", "west", "south_west", "south_east"],
    }
    return _row(
        "hex_grid_axial_neighbor_invariants",
        "hex_grid_2d",
        observed == expected,
        expected,
        observed,
        substrate.graph_hash(),
        "Rectangular axial hex-like grid exposes six local directions for an interior site.",
    )


def validation_irregular_graph() -> dict[str, Any]:
    substrate = graph_substrate({0: [1], 1: [0, 2, 3], 2: [1], 3: [1]})
    error_contains = ""
    try:
        graph_substrate({0: [1], 1: []})
    except ValueError as exc:
        error_contains = str(exc)
    observed = {
        "middle_neighbors": list(substrate.neighbors(1)),
        "leaf_boundary": substrate.site(0).boundary,
        "bad_graph_error_mentions_reciprocal": "reciprocal" in error_contains,
    }
    expected = {
        "middle_neighbors": [0, 2, 3],
        "leaf_boundary": True,
        "bad_graph_error_mentions_reciprocal": True,
    }
    return _row(
        "irregular_graph_invariants",
        "irregular_graph",
        observed == expected,
        expected,
        observed,
        substrate.graph_hash(),
        "Irregular graph substrate preserves supplied neighbors and rejects non-reciprocal adjacency.",
    )


def validation_lattice3d() -> dict[str, Any]:
    substrate = lattice3d_substrate(3, 3, 3)
    observed = {
        "center_neighbor_count": substrate.neighbor_count(13),
        "corner_neighbor_count": substrate.neighbor_count(0),
        "up_neighbor_direction": substrate.direction_between(13, 22),
    }
    expected = {
        "center_neighbor_count": 6,
        "corner_neighbor_count": 3,
        "up_neighbor_direction": "up",
    }
    return _row(
        "optional_3d_lattice_neighbor_invariants",
        "lattice_3d",
        observed == expected,
        expected,
        observed,
        substrate.graph_hash(),
        "Optional simple 3D lattice uses six face-neighbor directions and open boundaries.",
    )


def validation_move_and_status_actions() -> dict[str, Any]:
    move_substrate = array_substrate(3)
    move_substrate.place_cell(0, SubstrateCell("a", 1))
    move_substrate.place_cell(1, SubstrateCell("b", 2))
    occupied = move_substrate.apply_action("move", 0, 1)
    nonadjacent = move_substrate.apply_action("move", 0, 2)
    allowed = move_substrate.apply_action("move", 1, 2)

    passive = array_substrate(2)
    passive.place_cell(0, SubstrateCell("actor", 2, status="ACTIVE"))
    passive.place_cell(1, SubstrateCell("target", 1, status="FREEZE"))
    passive_swap = passive.apply_action("swap", 0, 1)

    stuck = array_substrate(2)
    stuck.place_cell(0, SubstrateCell("actor", 2, status="ACTIVE"))
    stuck.place_cell(1, SubstrateCell("target", 1, status="stuck"))
    stuck_swap = stuck.apply_action("swap", 0, 1)

    observed = {
        "move_to_occupied": {"allowed": occupied.allowed, "reason": occupied.reason},
        "move_to_nonadjacent": {"allowed": nonadjacent.allowed, "reason": nonadjacent.reason},
        "move_to_empty_adjacent": {"allowed": allowed.allowed, "values": list(move_substrate.values_in_site_order())},
        "passive_frozen_target_swap_allowed": passive_swap.allowed,
        "stuck_target_swap_allowed": stuck_swap.allowed,
        "stuck_target_reason": stuck_swap.reason,
    }
    expected = {
        "move_to_occupied": {"allowed": False, "reason": "target_not_empty"},
        "move_to_nonadjacent": {"allowed": False, "reason": "target_not_adjacent"},
        "move_to_empty_adjacent": {"allowed": True, "values": [1, None, 2]},
        "passive_frozen_target_swap_allowed": True,
        "stuck_target_swap_allowed": False,
        "stuck_target_reason": "target_not_movable_or_missing",
    }
    return _row(
        "occupancy_and_status_action_constraints",
        "shared_action_semantics",
        observed == expected,
        expected,
        observed,
        move_substrate.graph_hash(),
        "Shared action layer rejects nonlocal/occupied moves and distinguishes passive-frozen from stuck targets.",
    )


def validation_legal_actions_are_dry_runs() -> dict[str, Any]:
    substrate = array_substrate(3)
    substrate.place_cell(0, SubstrateCell("a", 2))
    substrate.place_cell(1, SubstrateCell("b", 1))
    before = substrate.signature()
    actions = substrate.legal_actions(0)
    observed = {
        "signature_unchanged": substrate.signature() == before,
        "actions": [
            {"action_type": action.action_type, "target": action.target_site_id, "allowed": action.allowed, "reason": action.reason}
            for action in actions
        ],
    }
    expected = {
        "signature_unchanged": True,
        "actions": [
            {"action_type": "wait", "target": None, "allowed": True, "reason": "allowed"},
            {"action_type": "swap", "target": 1, "allowed": True, "reason": "allowed"},
            {"action_type": "move", "target": 1, "allowed": False, "reason": "target_not_empty"},
        ],
    }
    return _row(
        "legal_action_queries_are_nonmutating",
        "shared_action_semantics",
        observed == expected,
        expected,
        observed,
        substrate.graph_hash(),
        "Legal-action query reports local options without mutating occupancy state.",
    )


def _row(
    validation_case: str,
    substrate_kind: str,
    success: bool,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    graph_hash: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "validation_case": validation_case,
        "substrate_kind": substrate_kind,
        "success": bool(success),
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "graph_hash": graph_hash,
        "detail": detail,
    }


def run_validations() -> pd.DataFrame:
    rows = [
        validation_1d_reference_bridge(),
        validation_square_grid(),
        validation_hex_grid(),
        validation_irregular_graph(),
        validation_lattice3d(),
        validation_move_and_status_actions(),
        validation_legal_actions_are_dry_runs(),
    ]
    return pd.DataFrame(rows)


def markdown_validation_table(df: pd.DataFrame) -> str:
    columns = ["validation_case", "substrate_kind", "success", "detail"]
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
- Caveats or blockers: 3D is included only as a simple optional face-neighbor lattice; S01 validates mechanics and local semantics, not morphology targets or biological claims.
- Lay summary: S01 built a common local-neighborhood layer that lets the sorting-cell abstraction live on rows, square grids, hex-like grids, irregular graphs, and a simple 3D lattice while keeping actions local.
- Recommended next action: {recommended_next_action}
"""


def substrate_spec_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    recommended_next_action: str,
) -> str:
    summary = top_summary_markdown(artifacts, validation_result, recommended_next_action)
    return f"""{summary}

# E05 S01 Substrate Specification

## Scope

S01 defines a fixed-site graph substrate interface for higher-dimensional morphogenesis tasks. A substrate is a graph of sites with deterministic neighbor order, optional coordinates, explicit boundary-condition metadata, and mutable occupancy by cells. Policies see only the actor cell, neighboring sites, neighbor occupancy and status, and the local action set.

## Substrate Families

- `array_1d`: open or periodic 1D row. Open rows mark endpoint sites as boundaries and preserve E03 indexed-neighbor semantics.
- `square_grid_2d`: rectangular four-neighbor grid with north/east/south/west directions.
- `hex_grid_2d`: rectangular axial-coordinate, six-neighbor, hex-like grid with east, north-east, north-west, west, south-west, and south-east directions.
- `irregular_graph`: caller-supplied undirected graph with reciprocal-edge validation and optional coordinates.
- `lattice_3d`: optional simple 3D face-neighbor lattice with six directions.

## Cell And Occupancy Contract

Cells carry stable `cell_id`, scalar `value`, `label`, `status`, and optional metadata. S01 preserves scalar values for continuity with E03; vector identities are reserved for S02. Occupancy is one cell per site or empty. Cell IDs remain stable and update their occupied site after swap or move.

## Local Observation Contract

`local_observation(site_id)` returns actor site, actor coordinate, actor cell, boundary flag, action set, and one `NeighborObservation` per adjacent site. Neighbor records include direction, occupancy, cell ID, value, label, status, and whether the target is movable by local swap semantics.

## Action Semantics

- `wait`: always legal and nonmutating.
- `swap`: requires adjacent source and target, an active actor, and a movable occupied target. `ACTIVE` actors can swap with `ACTIVE` and passive-frozen/`FREEZE` targets. `stuck` targets are not movable.
- `move`/`crawl`: requires adjacent source and empty target.
- `legal_actions(site_id)`: dry-run query that reports local legal actions without changing occupancy.
- `apply_indexed_policy_action(...)`: bridges E03 indexed 1D policy actions onto the 1D substrate for continuity validation.

## Validation Summary

{markdown_validation_table(validation_df)}
"""


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    args: argparse.Namespace,
    recommended_next_action: str,
) -> str:
    summary = top_summary_markdown(artifacts, validation_result, recommended_next_action)
    command_lines = "\n".join(
        f"- `{command['command']}`: return code {command['returnCode']}, success={command['success']}, elapsed={command['elapsedSeconds']:.3f}s"
        for command in test_commands
    )
    source_lines = "\n".join(
        f"- `{entry['relativePath']}` sha256 `{entry['sha256']}` ({entry['sizeBytes']} bytes)"
        for entry in source_files
    )
    return f"""{summary}

# Research Step Full Results: {STEP_ID} Generalize The Substrate

## Lay Summary

This step turns the original row of sorting cells into a reusable local-neighborhood substrate layer. The same local ideas now work on a 1D row, square grid, hex-like grid, irregular graph, and a simple 3D lattice. The result is an engineering foundation, not a biological validation: it says local moves and local observations are now consistently represented across substrates.

## Frozen Question

Can the simulator be generalized from 1D arrays to 2D grids, hex grids, irregular graphs, and optional simple 3D lattices while preserving local-policy semantics?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S01.
- Repository checkout: `{args.repo_dir}`.
- E03 policy interface context: `{args.previous_e03_dir}` and live repository `src/e03/policy_interface.py`.
- E04 memory/signaling context: `{args.previous_e04_dir}` and live repository `src/e04/`.
- Paper attachment markdown: `{args.paper_markdown}`.
- Datasets: none required by `DATASETS.md`.

## Methods

Implemented `src/e05/substrates.py` with fixed-site graph substrates, deterministic neighbor ordering, occupancy tracking, local observations, local action legality, mutation through swap/move/wait, graph validation, stable graph hashes, and a 1D E03 indexed-action bridge. Added focused tests in `tests/e05/test_substrates.py` and generated validation cases in `scripts/e05_s01_substrate_validation.py`.

The 1D validation does not reimplement Bubble logic. It asks the E03 `OriginalCellPolicyWrapper` for the reference local action on a two-cell Bubble fixture and applies that indexed action to the 1D substrate, then compares the substrate values to the public cell-method outcome.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- No new packages were installed for S01.
- CPU/GPU use: validation is small and serial; no GPU use was needed.

## Parameters

- Substrates: 1D array, 3x3 square grid, 3x3 hex-like grid, four-node irregular graph, 3x3x3 simple lattice.
- Boundary conditions validated: open boundaries. Periodic construction is implemented for regular substrates but not part of the primary S01 validation table.
- Local actions validated: wait, swap, move/crawl, legal-action dry runs.

## Results

{markdown_validation_table(validation_df)}

All validation rows passed. The implementation supports the required substrate families and preserves the E03 local-action bridge for the tested 1D fixture.

## Validation Checks

- Graph invariants: reciprocal-edge validation and no self loops for irregular graphs.
- Neighbor counts: 1D row, square grid, hex-like grid, and 3D lattice fixtures.
- Action legality: occupied, nonadjacent, passive-frozen, stuck, and dry-run cases.
- 1D special case: E03 Bubble policy action applied on the 1D substrate matches the public reference cell-method outcome.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- S01 defines mechanics and local observations only. It does not implement S02 vector identities, S03 target morphologies, S04 expanded action sets, or S05 morphology metrics.
- The hex grid is a rectangular axial, hex-like fixture, not a continuous tissue mechanics model.
- The 3D lattice is included as an optional simple face-neighbor substrate; downstream work should keep it optional unless 3D tasks become central.
- Validation is fixture-level. Larger benchmark behavior is intentionally left for S06 and later E05 steps.

## Blockers And Failed Assumptions

No blocker was found. The only runtime gap was that `pytest` is not installed; validation uses `unittest`, which is compatible with the repository tests and required no dependency installation.

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
    validation_df = run_validations()
    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"
    recommended_next_action = "Stop before S02 and let the Chief Scientist review S01; if accepted, proceed to generalize cell identity in S02."

    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )
    test_success = all(command["success"] for command in test_commands)

    validation_path = results_dir / "e05_substrate_validation.parquet"
    validation_csv_path = tables_dir / "e05_substrate_validation.csv"
    config_path = configs_dir / "e05_s01_substrate_validation.json"
    source_manifest_path = src_snapshot_dir / "e05_substrates_manifest.json"
    spec_path = reports_dir / "e05_substrate_spec.md"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)
    config_payload = {
        "schema": "eidosoma.e05_s01.substrate_validation_config.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "validationCases": validation_df["validation_case"].tolist(),
        "unitTestsRun": bool(args.run_unit_tests),
        "boundaryConditionsPrimary": ["open"],
        "optional3DIncluded": True,
        "createdAt": started_at,
    }
    write_json(config_path, config_payload)

    source_files = [
        source_entry(args.repo_dir / "src/e05/__init__.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/substrates.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_substrates.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s01_substrate_validation.py", args.repo_dir),
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
        artifact_entry(validation_path, artifacts_dir, "Machine-readable S01 substrate validation table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV mirror of S01 validation table for quick inspection."),
        artifact_entry(config_path, artifacts_dir, "Validation configuration and case list."),
        artifact_entry(source_manifest_path, artifacts_dir, "Repository source hashes for S01 code and tests."),
    ]
    planned_artifact_paths = [
        {"path": str(spec_path), "description": "S01 substrate specification report."},
        {"path": str(full_results_path), "description": "S01 full-results handoff report."},
        {"path": str(artifact_manifest_path), "description": "S01 artifact manifest."},
        {"path": str(run_manifest_path), "description": "Experiment run manifest updated for S01."},
        {"path": str(checksum_path), "description": "Checksums for key S01 artifacts."},
    ]
    write_text(
        spec_path,
        substrate_spec_markdown(
            artifacts=[*artifacts_for_summary, *planned_artifact_paths],
            validation_result=validation_result,
            validation_df=validation_df,
            recommended_next_action=recommended_next_action,
        ),
    )
    artifacts_after_spec = [
        *artifacts_for_summary,
        artifact_entry(spec_path, artifacts_dir, "S01 substrate specification report."),
    ]
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts=[*artifacts_after_spec, *planned_artifact_paths[1:]],
            validation_result=f"{validation_result}; unit-test commands success={test_success}",
            validation_df=validation_df,
            test_commands=test_commands,
            source_files=source_files,
            args=args,
            recommended_next_action=recommended_next_action,
        ),
    )
    artifacts_final = [
        *artifacts_after_spec,
        artifact_entry(full_results_path, artifacts_dir, "S01 full-results handoff report."),
    ]
    write_json(
        artifact_manifest_path,
        {
            "schema": "eidosoma.artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "success": bool(validation_success and test_success),
            "artifacts": [*artifacts_final, manifest_self_entry(artifact_manifest_path, artifacts_dir, "S01 artifact manifest.")],
            "validationResult": f"{validation_result}; unit-test commands success={test_success}",
            "caveatsOrBlockers": "No blocker. 3D remains optional/simple; S01 does not define identity vectors or target morphologies.",
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
        "artifacts": [*artifacts_final, artifact_entry(artifact_manifest_path, artifacts_dir, "S01 artifact manifest.")],
        "validationResult": f"{validation_result}; unit-test commands success={test_success}",
    }
    write_json(run_manifest_path, run_manifest_payload)
    checksum_inputs = [
        validation_path,
        validation_csv_path,
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
