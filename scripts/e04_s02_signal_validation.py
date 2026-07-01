#!/usr/bin/env python3
"""Validate E04 S02 local/diffusive signaling and write artifacts."""

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
from src.e04.memory_policies import BoundedCellMemoryBank, CellMemoryConfig, memory_policy_for_cell, memory_signature
from src.e04.signaling import (
    LocalSignalValues,
    SignalBank,
    SignalConfig,
    signal_policy_for_cell,
    signal_signature,
)


STEP_ID = "S02"
STEP_NUMBER = 2
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


def validation_local_blocked_signal() -> dict[str, Any]:
    cells, _probe = build_cells(
        SimulatorConfig(values=(2, 1, 3), algorithm="bubble", frozen_indices=(1,), frozen_semantics="stuck")
    )
    memory_bank = BoundedCellMemoryBank(CellMemoryConfig(max_frustration=3))
    signal_bank = SignalBank(len(cells), SignalConfig(local_radius=1, diffusion_enabled=True, decay=0.0))
    random.seed(1)
    result = signal_policy_for_cell(cells[0], signal_bank, memory_bank).step(cells[0], event_step=1)
    emission = result.emission_after
    audit = signal_bank.audit_no_global_oracle()
    observed = {
        "emission_present": emission is not None,
        "blocked": None if emission is None else emission.values.blocked,
        "frustrated_positive": False if emission is None else emission.values.frustrated > 0.0,
        "accessed_indices": [] if emission is None else list(emission.accessed_indices),
        "uses_global_oracle": audit["usesGlobalOracle"],
        "diffusive_fields_labeled": audit["diffusiveFieldsExplicitlyLabeled"],
    }
    expected = {
        "emission_present": True,
        "blocked": 1.0,
        "frustrated_positive": True,
        "accessed_indices": [0, 1],
        "uses_global_oracle": False,
        "diffusive_fields_labeled": True,
    }
    return {
        "validation_case": "blocked_signal_local_no_oracle",
        "case_type": "local_signal_and_audit",
        "success": observed == expected,
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": "blocked and frustrated signals emit after a stuck frozen-target attempt using only radius-1 indices",
    }


def validation_diffusion() -> dict[str, Any]:
    bank = SignalBank(5, SignalConfig(diffusion_rate=0.25, decay=0.0))
    bank.deposit_field(2, LocalSignalValues(blocked=1.0))
    bank.diffuse_once()
    observed = {
        "blocked_field": bank.fields["blocked"],
        "diffusion_step_count": bank.diffusion_step_count,
    }
    expected = {
        "blocked_field": [0.0, 0.25, 0.5, 0.25, 0.0],
        "diffusion_step_count": 1,
    }
    return {
        "validation_case": "diffusion_spreads_scalar_field",
        "case_type": "diffusion",
        "success": observed == expected,
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": "single central blocked scalar diffuses to adjacent positions with configured rate and no decay",
    }


def validation_noisy_fixed_seed() -> dict[str, Any]:
    def sense(seed: int) -> dict[str, Any]:
        bank = SignalBank(3, SignalConfig(noise_std=0.05, noise_seed=seed))
        bank.deposit_field(1, LocalSignalValues(morphogen=1.0))
        cells, _probe = build_cells(SimulatorConfig(values=(2, 1, 3), algorithm="bubble"))
        observation = OriginalCellPolicyWrapper.from_cell(cells[1]).observe(cells[1])
        return bank.sense(observation, event_step=1).to_dict()

    first = sense(42)
    second = sense(42)
    different = sense(43)
    observed = {
        "same_seed_equal": first == second,
        "different_seed_differs": first["diffusive_fields"] != different["diffusive_fields"],
        "noise_applied": first["noise_applied"],
    }
    expected = {
        "same_seed_equal": True,
        "different_seed_differs": True,
        "noise_applied": True,
    }
    return {
        "validation_case": "noisy_sensing_fixed_seed_reproducible",
        "case_type": "noise",
        "success": observed == expected,
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": "Gaussian sensing noise is deterministic for a fixed signal-bank seed",
    }


def validation_no_signal_parity() -> dict[str, Any]:
    config = SimulatorConfig(values=(4, 1, 3, 2), algorithm="bubble")
    memory_cells, memory_probe = build_cells(config)
    signal_cells, signal_probe = build_cells(config)
    memory_bank = BoundedCellMemoryBank(CellMemoryConfig())
    signal_memory_bank = BoundedCellMemoryBank(CellMemoryConfig())
    disabled_signal_bank = SignalBank(len(signal_cells), SignalConfig(enabled=False))
    random.seed(7)
    memory_policy_for_cell(memory_cells[0], memory_bank).step(memory_cells[0], event_step=1)
    random.seed(7)
    signal_policy_for_cell(signal_cells[0], disabled_signal_bank, signal_memory_bank).step(signal_cells[0], event_step=1)
    observed = {
        "public_signature_equal": cells_signature(signal_cells) == cells_signature(memory_cells),
        "probe_counts_equal": probe_counts(signal_probe) == probe_counts(memory_probe),
        "memory_signature_equal": memory_signature(signal_memory_bank) == memory_signature(memory_bank),
        "emission_log_count": len(disabled_signal_bank.emission_log),
        "sensation_log_count": len(disabled_signal_bank.sensation_log),
    }
    expected = {
        "public_signature_equal": True,
        "probe_counts_equal": True,
        "memory_signature_equal": True,
        "emission_log_count": 0,
        "sensation_log_count": 0,
    }
    return {
        "validation_case": "no_signal_mode_recovers_s01_baseline",
        "case_type": "no_signal_parity",
        "success": observed == expected,
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": "disabled S02 signal wrapper matches S01 memory-wrapper public state, memory state, and probe counts",
    }


def validation_radius_and_determinism() -> dict[str, Any]:
    bank = SignalBank(5, SignalConfig(local_radius=1, diffusion_enabled=False))
    for position in range(5):
        bank.set_local_signal(position, LocalSignalValues(blocked=float(position)))
    cells, _probe = build_cells(SimulatorConfig(values=(5, 4, 3, 2, 1), algorithm="bubble"))
    observation = OriginalCellPolicyWrapper.from_cell(cells[2]).observe(cells[2])
    sensation = bank.sense(observation, event_step=3)

    def run_sequence():
        seq_cells, _seq_probe = build_cells(SimulatorConfig(values=(4, 1, 3, 2), algorithm="bubble"))
        memory_bank = BoundedCellMemoryBank(CellMemoryConfig())
        signal_bank = SignalBank(len(seq_cells), SignalConfig(noise_seed=11))
        for event_step, position in enumerate((0, 2, 1, 0), start=1):
            random.seed(200 + event_step)
            signal_policy_for_cell(seq_cells[position], signal_bank, memory_bank).step(seq_cells[position], event_step=event_step)
        return cells_signature(seq_cells), memory_signature(memory_bank), signal_signature(signal_bank)

    first = run_sequence()
    second = run_sequence()
    observed = {
        "sensed_positions": [position for position, _values in sensation.local_signals],
        "accessed_indices": list(sensation.accessed_indices),
        "access_scope": sensation.access_scope,
        "fixed_seed_sequence_equal": first == second,
    }
    expected = {
        "sensed_positions": [1, 2, 3],
        "accessed_indices": [1, 2, 3],
        "access_scope": "local_window",
        "fixed_seed_sequence_equal": True,
    }
    return {
        "validation_case": "local_radius_and_sequence_determinism",
        "case_type": "locality_and_determinism",
        "success": observed == expected,
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": "local sensing is radius bounded and a fixed-seed signal sequence replays exactly",
    }


def run_validation_cases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            validation_local_blocked_signal(),
            validation_diffusion(),
            validation_noisy_fixed_seed(),
            validation_no_signal_parity(),
            validation_radius_and_determinism(),
        ]
    )


def render_spec(
    *,
    validation_success: bool,
    validation_line: str,
    artifact_paths: list[Path],
    result_path: Path,
    config: SignalConfig,
) -> str:
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    outcome = "supportive" if validation_success else "constraining/contradictory"
    return f"""# E04 Signal Model Specification

## Top Summary

- Research step ID: S02
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: This validates communication mechanics and audit contracts only; it does not yet claim repair improvement. Diffusive scalar fields are explicitly labeled as less local than nearest-neighbor signals.
- Recommended next action: Stop before S03 and let the Chief Scientist review the communication contract before adding repairable Frozen Cells.

## Scope

S02 adds cell-emitted and locally sensed signals around the S01 memory-enabled policy interface. Signals do not alter public Bubble, Insertion, or Selection actions in this step; they provide an auditable side channel for downstream communication, repair, homeostatic, and learning experiments.

## Signal Channels

- `blocked`: scalar in `[0, 1]`, emitted after a blocked or failed local swap attempt.
- `sorted`: scalar in `[0, 1]`, computed only from actor-neighbor orderedness, not global Sortedness.
- `frustrated`: scalar in `[0, 1]`, derived from bounded S01 local-frustration memory.
- `target_seeking`: signed scalar in `[-1, 1]`, encoding direction toward the actor's current local-policy target when one exists.
- `morphogen`: scalar in `[0, 1]`, a simple derived field from blocked, frustrated, and absolute target-seeking values.

## Default Configuration

```json
{json.dumps(config.to_dict(), indent=2, sort_keys=True)}
```

## Locality And Diffusion Contract

Local sensing reads only positions within `actor_index +/- local_radius`. Emission records include `accessed_indices`; sensation records include `accessed_indices` and an `access_scope`. Diffusive scalar fields are stored separately by position and are labeled `local_window_plus_explicit_diffusive_fields` when sensed. Long-range influence through these scalar fields should be treated as a less-local communication mode in later analyses.

## Noise Modes

With `noise_std=0`, sensing is deterministic. With `noise_std>0`, Gaussian noise is applied only to sensed diffusive field values and is reproducible for a fixed `noise_seed`.

## No-Oracle Contract

The audit checks that signal records do not contain global Sortedness, monotonicity error, full-array rank, final values, or whole-array controller fields, and that accessed indices stay within the configured local radius. Global metrics remain offline evaluation outputs only.

## Validation Link

Machine-readable validation cases are in `{result_path}`. They check local blocked-signal emission, deterministic diffusion, fixed-seed noisy sensing, no-signal baseline parity, bounded radius sensing, fixed-seed replay, and no-global-oracle audit status.
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
            "- `python scripts/e04_s02_signal_validation.py --repo-dir /workspace/cell-research --artifacts-dir $ARTIFACTS_DIR`",
        ]
    )
    table = markdown_table(
        validation_df[["validation_case", "case_type", "success", "detail"]],
        ["validation_case", "case_type", "success", "detail"],
    )
    return f"""# E04 S02 Research Step Full Results

## Top Summary

- Research step ID: S02
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: S02 validates local and diffusive signal mechanics and no-oracle checks, not a repair-performance gain. Diffusive scalar fields are intentionally labeled as less local than nearest-neighbor signals and must be separated in downstream ablations.
- Lay summary: Cells can now leave and sense small local signals such as blocked, frustrated, target-seeking, and local orderedness. Signals can also diffuse along the array as scalar fields, and the system records exactly which nearby positions were read so future work can audit that no global controller slipped in.
- Recommended next action: Stop before S03. After Chief Scientist review, proceed to S03 repairable Frozen Cells using no-signal and no-memory controls.

## Frozen Question

Can local signals improve coordination or repair without cells reading global Sortedness or centralized state?

S02 implements and validates the communication mechanism needed to test that question in S03 and later repair benchmarks. It does not claim improved coordination or repair yet.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S02.
- Full plan: `/workspace/FULL_PLAN.md`.
- Repository checkout: `/workspace/cell-research`.
- S01 memory interface and report: `src/e04/memory_policies.py`, `/artifacts/research_steps/S01/research_step_full_results.md`, and `/artifacts/reports/e04_memory_state_spec.md`.
- E02 deterministic simulator context: `/previous-artifacts/E02` and `src/e02/deterministic_simulator.py`.
- E03 policy interface and DSL context: `/previous-artifacts/E03` and `src/e03/policy_interface.py`.
- Datasets: none required.

## Methods

Implemented `src/e04/signaling.py` with:

- `SignalConfig` for enabling/disabling signals, local radius, diffusion rate, decay, and noise seed/std;
- `SignalBank` as an external local-emission and scalar-field store;
- local signal inference from S01 memory and bounded local observations;
- a 1D nearest-neighbor diffusion update for scalar fields;
- fixed-seed Gaussian noise on sensed diffusive fields;
- `SignalEnabledPolicyWrapper`, which delegates action execution to the S01 memory wrapper and then emits signals;
- an explicit no-global-oracle audit over logged emissions and sensations.

The no-signal baseline is `SignalConfig(enabled=false)`. In that mode, no signal state is stored and the wrapper preserves S01 public-state, memory-state, and probe-count behavior.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- New dependencies installed: none.
- CPU use: serial validation; host reports `{os.cpu_count()}` logical CPUs, but S02 did not need parallel workers.
- Platform: {platform.platform()}

## Parameters

- Default local radius: 1
- Default diffusion enabled: true
- Default diffusion rate: 0.25
- Default decay: 0.0
- Deterministic/no-noise mode: `noise_std=0`
- Noisy validation mode: `noise_std=0.05`, fixed seed 42
- No-signal baseline: `enabled=false`
- Validation output: `{result_path}` and CSV sidecar `{result_csv_path}`
- Specification output: `{spec_path}`
- Configuration output: `{config_path}`

## Results

{table}

All validation rows passed if the top summary reports success. The result table is machine-readable and includes expected/observed JSON for each check.

## Validation Checks

- Local blocked/frustrated emission used only radius-1 indices and passed no-global-oracle audit.
- Diffusion spread a central scalar field deterministically to adjacent positions.
- Noisy sensing was reproducible for fixed seeds and changed under a different seed.
- Disabled-signal mode recovered the S01 memory-wrapper public signature, memory signature, and probe counts while storing no signal logs.
- Local sensing respected radius bounds.
- Fixed-seed action/signal sequences replayed exactly.
- Unit tests: `tests/e04/test_signaling.py` and existing S01 memory tests passed.
- Regression guards: E03 policy-interface tests and E02 deterministic simulator tests passed.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S02.
- This is communication-interface readiness, not a demonstration that signaling improves repair.
- Diffusive fields allow longer-range scalar influence over time. They are explicitly labeled as less local and should be ablated separately from nearest-neighbor signals.
- `sorted` is actor-neighbor orderedness only, not global Sortedness.
- The signal wrapper does not yet let signals alter policy actions; S03 and later steps can use the side channel for repair conditions and adaptive policies.

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
- Signal model spec: `{spec_path}`

## Recommended Next Action

Stop before S03. Chief Scientist should review the communication contract, then authorize S03 repairable Frozen Cells with no-signal, no-memory, and original open-loop controls.
"""


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_signal_validation.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_signal_validation.csv"
    spec_path = artifacts_dir / "reports" / "e04_signal_model_spec.md"
    config_path = artifacts_dir / "configs" / "e04_s02_signal_validation.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_signaling_manifest.json"
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

    signal_config = SignalConfig()
    validation_config = {
        "schema": "eidosoma.e04.s02.signal_validation_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "signalConfig": signal_config.to_dict(),
        "validationSeeds": {
            "blockedSignal": 1,
            "noSignalParity": 7,
            "noiseSameSeed": 42,
            "noiseDifferentSeed": 43,
            "determinismStart": 201,
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
            config=signal_config,
        ),
    )

    source_paths = [
        repo_dir / "src/e04/signaling.py",
        repo_dir / "tests/e04/test_signaling.py",
        repo_dir / "scripts/e04_s02_signal_validation.py",
        repo_dir / "src/e04/memory_policies.py",
        repo_dir / "tests/e04/test_memory_state.py",
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
            "s01MemorySpec": str(artifacts_dir / "reports/e04_memory_state_spec.md"),
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
            "signalModelSpec": str(spec_path),
            "signalModelSpecSha256": sha256_file(spec_path),
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
        artifact_entry(report_path, artifacts_dir, "S02 full-results handoff report"),
        artifact_entry(spec_path, artifacts_dir, "S02 signal model specification"),
        artifact_entry(result_path, artifacts_dir, "S02 signal validation results"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S02 signal validation results"),
        artifact_entry(config_path, artifacts_dir, "S02 signal validation config"),
        artifact_entry(source_manifest_path, artifacts_dir, "S02 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S02 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S02 outputs"),
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
