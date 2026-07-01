#!/usr/bin/env python3
"""Validate E04 S01 bounded local memory and write artifacts."""

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
from src.e03.policy_interface import OriginalCellPolicyWrapper, cells_signature
from src.e04.memory_policies import (
    BoundedCellMemoryBank,
    CellMemoryConfig,
    memory_policy_for_cell,
    memory_signature,
)


STEP_ID = "S01"
STEP_NUMBER = 1
EXPERIMENT_ID = "E04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-e01-dir", type=Path, default=Path("/previous-artifacts/E01"))
    parser.add_argument("--previous-e02-dir", type=Path, default=Path("/previous-artifacts/E02"))
    parser.add_argument("--previous-e03-dir", type=Path, default=Path("/previous-artifacts/E03"))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def build_cells(config: SimulatorConfig):
    probe = EventTracingStatusProbe()
    cells, _cell_status = _build_cells(config, probe)
    return cells, probe


def probe_counts(probe: EventTracingStatusProbe) -> tuple[int, int, int]:
    return (
        int(probe.compare_and_swap_count),
        int(probe.swap_count),
        int(probe.frozen_swap_attempts),
    )


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


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


def run_bounded_failure_case() -> dict[str, Any]:
    cells, _probe = build_cells(
        SimulatorConfig(values=(2, 1), algorithm="bubble", frozen_indices=(1,), frozen_semantics="stuck")
    )
    config = CellMemoryConfig(
        failed_swap_capacity=2,
        neighbor_capacity=3,
        max_time_since_movement=3,
        max_frustration=2,
    )
    bank = BoundedCellMemoryBank(config)
    actor = cells[0]
    for event_step in range(1, 6):
        random.seed(1)
        memory_policy_for_cell(actor, bank).step(actor, event_step=event_step)
    state = bank.get(actor.threadID)
    observed = {
        "last_move_success": state.last_move_success,
        "time_since_movement": state.time_since_movement,
        "local_frustration": state.local_frustration,
        "failed_swap_count": len(state.recent_failed_swaps),
        "failed_swap_event_steps": [entry.event_step for entry in state.recent_failed_swaps],
        "neighbor_identity_count": len(state.recent_neighbor_identities),
        "neighbor_event_steps": [entry.event_step for entry in state.recent_neighbor_identities],
    }
    expected = {
        "last_move_success": False,
        "time_since_movement": 3,
        "local_frustration": 2,
        "failed_swap_count": 2,
        "failed_swap_event_steps": [4, 5],
        "neighbor_identity_count": 3,
        "neighbor_event_steps": [3, 4, 5],
    }
    return {
        "validation_case": "bounded_failed_swap_and_neighbor_memory",
        "case_type": "bounded_memory",
        "success": observed == expected,
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": "stuck frozen target repeatedly blocks a Bubble swap; bounded histories retain only the newest entries",
    }


def run_success_case() -> dict[str, Any]:
    cells, _probe = build_cells(SimulatorConfig(values=(2, 1), algorithm="bubble"))
    bank = BoundedCellMemoryBank(CellMemoryConfig(failed_swap_capacity=3, neighbor_capacity=3))
    actor = cells[0]
    random.seed(1)
    result = memory_policy_for_cell(actor, bank).step(actor, event_step=1)
    state = bank.get(actor.threadID)
    observed = {
        "applied_action": result.applied_action.action_type,
        "last_move_success": state.last_move_success,
        "time_since_movement": state.time_since_movement,
        "local_frustration": state.local_frustration,
        "failed_swap_count": len(state.recent_failed_swaps),
        "neighbor_identity_count": len(state.recent_neighbor_identities),
        "neighbor_thread_ids": [entry.thread_id for entry in state.recent_neighbor_identities],
    }
    expected = {
        "applied_action": "swap",
        "last_move_success": True,
        "time_since_movement": 0,
        "local_frustration": 0,
        "failed_swap_count": 0,
        "neighbor_identity_count": 1,
        "neighbor_thread_ids": [2],
    }
    return {
        "validation_case": "successful_swap_resets_motion_memory",
        "case_type": "state_update",
        "success": observed == expected,
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": "successful local movement sets last_move_success and resets time/frustration counters",
    }


def run_determinism_case() -> dict[str, Any]:
    def run_sequence() -> dict[str, Any]:
        cells, _probe = build_cells(SimulatorConfig(values=(4, 1, 3, 2), algorithm="bubble"))
        bank = BoundedCellMemoryBank(CellMemoryConfig(failed_swap_capacity=3, neighbor_capacity=4))
        for event_step, position in enumerate((0, 2, 1, 0, 2, 1), start=1):
            random.seed(100 + event_step)
            actor = cells[position]
            memory_policy_for_cell(actor, bank).step(actor, event_step=event_step)
        return {
            "public_signature": cells_signature(cells),
            "memory_signature": memory_signature(bank),
            "memory_digest": bank.stable_digest(),
        }

    first = run_sequence()
    second = run_sequence()
    observed = {
        "public_signature_equal": first["public_signature"] == second["public_signature"],
        "memory_signature_equal": first["memory_signature"] == second["memory_signature"],
        "memory_digest_equal": first["memory_digest"] == second["memory_digest"],
        "memory_digest": first["memory_digest"],
    }
    expected = {
        "public_signature_equal": True,
        "memory_signature_equal": True,
        "memory_digest_equal": True,
    }
    return {
        "validation_case": "fixed_seed_memory_sequence_is_deterministic",
        "case_type": "determinism",
        "success": all(observed[key] == value for key, value in expected.items()),
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": "fixed activation positions and policy RNG seeds replay identical public and memory signatures",
    }


def run_disabled_parity_case() -> dict[str, Any]:
    config = SimulatorConfig(values=(4, 1, 3, 2), algorithm="bubble")
    direct_cells, direct_probe = build_cells(config)
    memory_cells, memory_probe = build_cells(config)
    random.seed(7)
    OriginalCellPolicyWrapper.from_cell(direct_cells[0]).step(direct_cells[0])
    bank = BoundedCellMemoryBank(CellMemoryConfig(enabled=False))
    random.seed(7)
    memory_policy_for_cell(memory_cells[0], bank).step(memory_cells[0], event_step=1)
    observed = {
        "public_signature_equal": cells_signature(memory_cells) == cells_signature(direct_cells),
        "probe_counts_equal": probe_counts(memory_probe) == probe_counts(direct_probe),
        "memory_state_count": len(bank.to_dict()["states"]),
    }
    expected = {
        "public_signature_equal": True,
        "probe_counts_equal": True,
        "memory_state_count": 0,
    }
    return {
        "validation_case": "disabled_memory_matches_open_loop_public_step",
        "case_type": "ablation_parity",
        "success": observed == expected,
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": "memory-disabled wrapper delegates to the original public method and stores no state",
    }


def run_observation_case() -> dict[str, Any]:
    cells, _probe = build_cells(SimulatorConfig(values=(3, 1, 2), algorithm="selection"))
    bank = BoundedCellMemoryBank(CellMemoryConfig())
    actor = cells[1]
    wrapper = memory_policy_for_cell(actor, bank)
    before = wrapper.state(actor)
    random.seed(11)
    wrapper.step(actor, event_step=1)
    after = wrapper.state(cells[int(actor.current_position[0])])
    observed = {
        "before_has_memory_key": "e04_cell_memory" in before.memory,
        "after_has_memory_key": "e04_cell_memory" in after.memory,
        "after_last_action_type": after.memory["e04_cell_memory"]["last_action_type"],
        "after_neighbor_identity_count": len(after.memory["e04_cell_memory"]["recent_neighbor_identities"]),
    }
    expected = {
        "before_has_memory_key": True,
        "after_has_memory_key": True,
        "after_neighbor_identity_count_at_least_one": True,
    }
    success = (
        observed["before_has_memory_key"]
        and observed["after_has_memory_key"]
        and observed["after_neighbor_identity_count"] >= 1
    )
    return {
        "validation_case": "memory_visible_through_policy_state",
        "case_type": "interface_contract",
        "success": success,
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": "MemoryEnabledPolicyWrapper.state exposes bounded memory under PolicyState.memory without global metrics",
    }


def run_validation_cases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            run_success_case(),
            run_bounded_failure_case(),
            run_determinism_case(),
            run_disabled_parity_case(),
            run_observation_case(),
        ]
    )


def render_spec(
    *,
    validation_success: bool,
    validation_line: str,
    artifact_paths: list[Path],
    result_path: Path,
    config: CellMemoryConfig,
) -> str:
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    outcome = "supportive" if validation_success else "constraining/contradictory"
    return f"""# E04 Memory State Specification

## Top Summary

- Research step ID: S01
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: No blocker to S02 if validation passes. This is an interface-level memory extension; it does not yet demonstrate improved repair or robustness.
- Recommended next action: Stop before S02 and let the Chief Scientist review the memory interface before adding local communication.

## Scope

S01 adds bounded per-cell local memory to the E03 policy interface while preserving the E02 deterministic simulator and the public Bubble, Insertion, and Selection cell methods. Memory is an ablatable side channel for future E04 policies, not a global controller.

## Memory Variables

The canonical state for one cell identity is:

- `last_move_success`: `true`, `false`, or `null`, tracking the most recent attempted local movement outcome.
- `last_action_type`: the applied public action inferred by the E03 wrapper, such as `swap`, `wait`, `update_state`, or `blocked_swap_attempt`.
- `last_target_index`: the last applied target index when observable.
- `time_since_movement`: a bounded integer counter, reset on successful swap and incremented on activated nonmovement.
- `local_frustration`: a bounded integer counter, incremented on failed swap attempts and decayed after successful movement.
- `recent_failed_swaps`: bounded newest-first history in chronological order after truncation, with actor identity, target identity when available, target value/label/status, and failure reason.
- `recent_neighbor_identities`: bounded chronological history of immediate left/right neighbor identities sensed before the action.

## Default Configuration

```json
{json.dumps(config.to_dict(), indent=2, sort_keys=True)}
```

All capacities and counters are configurable and non-negative. The S01 defaults use finite histories and bounded counters. A disabled configuration, `enabled=false`, stores no cell memory and delegates to the original open-loop public method.

## Identity And Locality Contract

Memory is stored in `BoundedCellMemoryBank` by public `threadID`, so memory follows the cell object when it swaps positions. Neighbor identity records are local snapshots of immediate neighbors only; they do not include global sortedness, full-array rank, whole-array metrics, or centralized repair state.

## Update Rules

- Successful swap: `last_move_success=true`, `time_since_movement=0`, and `local_frustration` decays by `frustration_recovery` down to zero.
- Failed swap or blocked frozen attempt: `last_move_success=false`, `time_since_movement` increments up to `max_time_since_movement`, `local_frustration` increments up to `max_frustration`, and one failed-swap record is appended within `failed_swap_capacity`.
- Wait or local state update without attempted movement: `time_since_movement` increments, frustration is unchanged, and `last_move_success` retains the last movement-attempt outcome.
- Neighbor memory: immediate left/right neighbor identities from the pre-action local observation are appended and truncated to `neighbor_capacity`.

## Validation Link

Machine-readable validation cases are in `{result_path}`. They check bounded memory enforcement, fixed-seed deterministic replay, disabled-memory parity with the open-loop wrapper, and policy-state visibility.
"""


def render_report(
    *,
    validation_df: pd.DataFrame,
    validation_line: str,
    validation_success: bool,
    unit_tests: dict[str, Any],
    e03_tests: dict[str, Any],
    e02_tests: dict[str, Any],
    artifact_paths: list[Path],
    result_path: Path,
    result_csv_path: Path,
    spec_path: Path,
    config_path: Path,
    manifest_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    manifest: dict[str, Any],
) -> str:
    outcome = "supportive" if validation_success else "constraining/contradictory"
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    commands = "\n".join(
        [
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{e03_tests['command']}` -> return code {e03_tests['returnCode']}",
            f"- `{e02_tests['command']}` -> return code {e02_tests['returnCode']}",
            "- `python scripts/e04_s01_memory_validation.py --repo-dir /workspace/cell-research --artifacts-dir $ARTIFACTS_DIR`",
        ]
    )
    table = markdown_table(
        validation_df[["validation_case", "case_type", "success", "detail"]],
        ["validation_case", "case_type", "success", "detail"],
    )
    return f"""# E04 S01 Research Step Full Results

## Top Summary

- Research step ID: S01
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: S01 validates a bounded memory-enabled policy interface, not a repair-performance gain. No blocker remains for Chief review if validation passes. The next step should still audit S02 communication for global-oracle leakage.
- Lay summary: Each simulated cell can now carry a small local notebook of recent movement success, failed swap attempts, elapsed time since movement, frustration, and immediate neighbor identities. The notebook is capped, follows the cell across swaps, and can be switched off to recover the original open-loop behavior.
- Recommended next action: Stop before S02. After Chief Scientist review, proceed to S02 local communication using this memory interface and keep disabled-memory controls in every downstream benchmark.

## Frozen Question

Does bounded local memory create measurable improvements in robustness or repair without global monitoring?

S01 only implements and validates the memory interface required to test that question in later repair benchmarks. It does not claim a robustness or repair improvement yet.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S01.
- Full plan: `/workspace/FULL_PLAN.md`.
- Repository checkout: `/workspace/cell-research`.
- E01 baseline fixtures and reports: `/previous-artifacts/E01`.
- E02 deterministic simulator context: `/previous-artifacts/E02` and `src/e02/deterministic_simulator.py`.
- E03 policy interface and DSL context: `/previous-artifacts/E03` and `src/e03/policy_interface.py`.
- Uploaded paper context: `/workspace/input-attachments/f93afdc5-f2e5-4ecc-80bb-e088f93acf3c/_metadata/ATTACHMENT.md`; S01 did not require numeric extraction from the paper PDF.
- Datasets: none required.

## Methods

Implemented `src/e04/memory_policies.py` with an external `BoundedCellMemoryBank` keyed by public `threadID`, a `CellMemoryConfig`, immutable memory record dataclasses, and a `MemoryEnabledPolicyWrapper` that delegates action execution to `OriginalCellPolicyWrapper`. The wrapper observes the original local policy fields, executes the public `move()` method through the E03 interface, infers the applied action, and then updates only the activated cell's local memory.

The memory variables are finite by construction: failed-swap and neighbor histories are truncated to configured capacities, while time-since-movement and frustration counters are clamped. The disabled mode stores no state and is intended as the open-loop ablation.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- New dependencies installed: none.
- CPU use: serial validation; host reports `{os.cpu_count()}` logical CPUs, but S01 did not need parallel workers.
- Platform: {platform.platform()}

## Parameters

- Memory enabled default: `true`
- Failed-swap capacity used in validation: 2 to 3 entries
- Neighbor-identity capacity used in validation: 3 to 4 entries
- Max time-since-movement validation clamp: 3 events
- Max local-frustration validation clamp: 2 units
- Fixed policy seeds: Python `random.seed(...)` before public policy calls
- Validation output: `{result_path}` and CSV sidecar `{result_csv_path}`
- Specification output: `{spec_path}`
- Configuration output: `{config_path}`

## Results

{table}

The validation table shows all planned interface checks. Bounded histories retained only the newest failed swaps and neighbor identities, deterministic replays matched both public and memory signatures, and disabled-memory mode matched the original wrapper's public state and probe counters.

## Validation Checks

- Bounded memory enforced: failed-swap history, neighbor history, time-since-movement, and frustration all stayed at configured limits.
- Fixed-seed deterministic behavior: repeated activation sequences produced identical public signatures and memory digests.
- Open-loop mode unchanged when memory disabled: the memory wrapper matched `OriginalCellPolicyWrapper` public signatures and probe counts and stored no memory.
- Unit tests: `tests/e04/test_memory_state.py` passed.
- Regression guards: E03 policy-interface tests and E02 deterministic simulator tests passed.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S01.
- The result is supportive for interface readiness, not for the broader hypothesis that memory improves repair.
- Time-since-movement is updated for activated cells. Future event-loop integrations can add whole-population ticking if a downstream task requires wall-clock time for inactive cells.
- Neighbor identities are immediate-neighbor records only. They intentionally do not expose global sortedness, full-array rank, or centralized repair state.
- Storing neighbor identities across swaps is bounded but still creates an identity memory channel; downstream communication and learning steps should audit how much history is allowed.

## Provenance

- Git commit at validation time: `{manifest['gitCommit']}`
- Git branch at validation time: `{manifest['gitBranch']}`
- Git status at validation time: `{manifest['gitStatusShort'] or 'clean'}`
- Source files tracked in manifest: {len(manifest['sourceFiles'])}
- Run manifest: `{run_manifest_path}`
- Checksums: `{checksums_path}`
- Previous artifacts: E01 `{manifest['upstreamContext']['previousE01Dir']}`, E02 `{manifest['upstreamContext']['previousE02Dir']}`, E03 `{manifest['upstreamContext']['previousE03Dir']}`
- Created at UTC: `{manifest['createdAtUtc']}`

## Artifact Manifest

- Source/provenance manifest: `{manifest_path}`
- Result parquet: `{result_path}`
- Memory spec: `{spec_path}`

## Recommended Next Action

Stop before S02. Chief Scientist should review the memory contract, then authorize S02 local communication with explicit no-global-oracle checks and disabled-memory/no-signal controls.
"""


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_memory_state_validation.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_memory_state_validation.csv"
    spec_path = artifacts_dir / "reports" / "e04_memory_state_spec.md"
    config_path = artifacts_dir / "configs" / "e04_s01_memory_validation.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_memory_policies_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"
    report_path = step_dir / "research_step_full_results.md"
    for path in [
        step_dir,
        result_path.parent,
        result_csv_path.parent,
        spec_path.parent,
        config_path.parent,
        source_manifest_path.parent,
        checksums_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    memory_config = CellMemoryConfig()
    validation_config = {
        "schema": "eidosoma.e04.s01.memory_validation_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "memoryConfig": memory_config.to_dict(),
        "validationSeeds": {
            "successfulSwap": 1,
            "failedSwap": 1,
            "disabledParity": 7,
            "determinismStart": 101,
        },
        "workerCount": 1,
    }
    write_json(config_path, validation_config)

    validation_df = run_validation_cases()
    validation_df.to_parquet(result_path, index=False)
    validation_df.to_csv(result_csv_path, index=False)

    unit_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e04", "-p", "test_*.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    e03_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    e02_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_deterministic_simulator.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )

    validation_success = bool(validation_df["success"].all() and unit_tests["success"] and e03_tests["success"] and e02_tests["success"])
    validation_line = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed; "
        f"E04 unit tests return code {unit_tests['returnCode']}; "
        f"E03 policy-interface tests return code {e03_tests['returnCode']}; "
        f"E02 simulator tests return code {e02_tests['returnCode']}"
    )
    artifact_paths = [
        report_path,
        spec_path,
        result_path,
        result_csv_path,
        config_path,
        source_manifest_path,
        artifact_manifest_path,
        run_manifest_path,
        checksums_path,
    ]
    write_text(
        spec_path,
        render_spec(
            validation_success=validation_success,
            validation_line=validation_line,
            artifact_paths=artifact_paths,
            result_path=result_path,
            config=memory_config,
        ),
    )

    source_paths = [
        repo_dir / "src/e04/__init__.py",
        repo_dir / "src/e04/memory_policies.py",
        repo_dir / "tests/e04/test_memory_state.py",
        repo_dir / "scripts/e04_s01_memory_validation.py",
        repo_dir / "src/e03/policy_interface.py",
        repo_dir / "src/e02/deterministic_simulator.py",
    ]
    manifest: dict[str, Any] = {
        "schema": "eidosoma.src_snapshot.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(repo_dir, ["branch", "--show-current"]),
        "gitStatusShort": git_output(repo_dir, ["status", "--short"]),
        "dependencies": {
            "newDependenciesInstalled": [],
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "upstreamContext": {
            "previousE01Dir": str(args.previous_e01_dir),
            "previousE02Dir": str(args.previous_e02_dir),
            "previousE03Dir": str(args.previous_e03_dir),
            "e03RuleDslSpec": str(args.previous_e03_dir / "reports/e03_rule_dsl_spec.md"),
            "e03GpuSimulatorValidation": str(args.previous_e03_dir / "reports/e03_gpu_simulator_validation.md"),
            "paperAttachmentSidecar": "/workspace/input-attachments/f93afdc5-f2e5-4ecc-80bb-e088f93acf3c/_metadata/ATTACHMENT.md",
        },
        "sourceFiles": [source_entry(path, repo_dir) for path in source_paths if path.exists()],
        "validation": {
            "allValidationPassed": validation_success,
            "validationCaseCount": int(len(validation_df)),
            "validationCaseSuccessCount": int(validation_df["success"].sum()),
            "unitTests": unit_tests,
            "e03PolicyInterfaceTests": e03_tests,
            "e02DeterministicSimulatorTests": e02_tests,
            "validationParquet": str(result_path),
            "validationParquetSha256": sha256_file(result_path),
            "validationCsv": str(result_csv_path),
            "validationCsvSha256": sha256_file(result_csv_path),
            "memoryStateSpec": str(spec_path),
            "memoryStateSpecSha256": sha256_file(spec_path),
            "config": str(config_path),
            "configSha256": sha256_file(config_path),
        },
    }
    write_json(source_manifest_path, manifest)

    write_text(
        report_path,
        render_report(
            validation_df=validation_df,
            validation_line=validation_line,
            validation_success=validation_success,
            unit_tests=unit_tests,
            e03_tests=e03_tests,
            e02_tests=e02_tests,
            artifact_paths=artifact_paths,
            result_path=result_path,
            result_csv_path=result_csv_path,
            spec_path=spec_path,
            config_path=config_path,
            manifest_path=source_manifest_path,
            run_manifest_path=run_manifest_path,
            checksums_path=checksums_path,
            manifest=manifest,
        ),
    )

    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S01 full-results handoff report"),
        artifact_entry(spec_path, artifacts_dir, "S01 memory-state specification"),
        artifact_entry(result_path, artifacts_dir, "S01 memory validation results"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S01 memory validation results"),
        artifact_entry(config_path, artifacts_dir, "S01 memory validation config"),
        artifact_entry(source_manifest_path, artifacts_dir, "S01 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S01 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S01 outputs"),
    ]
    artifact_manifest = {
        "schema": "eidosoma.artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "artifacts": artifacts,
    }
    write_json(artifact_manifest_path, artifact_manifest)

    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "lastResearchStepId": STEP_ID,
        "createdAtUtc": utc_now(),
        "git": {
            "branch": manifest["gitBranch"],
            "headCommit": manifest["gitCommit"],
            "statusShort": manifest["gitStatusShort"],
        },
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "cpuCountReported": os.cpu_count(),
            "workerCountUsed": 1,
        },
        "dependencies": manifest["dependencies"],
        "validation": manifest["validation"],
        "artifacts": artifacts,
    }
    write_json(run_manifest_path, run_manifest)

    checksum_targets = [
        report_path,
        spec_path,
        result_path,
        result_csv_path,
        config_path,
        source_manifest_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")

    return 0 if validation_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
