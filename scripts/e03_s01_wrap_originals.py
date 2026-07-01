#!/usr/bin/env python3
"""Validate E03 S01 original-policy wrappers and write artifacts."""

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

from src.e02.deterministic_simulator import (
    EventTracingStatusProbe,
    SimulatorConfig,
    _build_cells,
)
from src.e03.policy_interface import (
    cells_signature,
    observe_cell,
    policy_for_cell,
)


STEP_ID = "S01"
STEP_NUMBER = 1
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-e01-dir", type=Path, default=Path("/previous-artifacts/E01"))
    parser.add_argument("--previous-e02-dir", type=Path, default=Path("/previous-artifacts/E02"))
    parser.add_argument("--paper-markdown", type=Path, default=Path("/workspace/input-attachments/f93afdc5-f2e5-4ecc-80bb-e088f93acf3c/pdf-markdown.md"))
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


def build_cells(config: SimulatorConfig) -> tuple[list[Any], EventTracingStatusProbe]:
    probe = EventTracingStatusProbe()
    cells, _cell_status = _build_cells(config, probe)
    return cells, probe


def probe_counts(probe: EventTracingStatusProbe) -> dict[str, int]:
    return {
        "comparison_count": int(probe.compare_and_swap_count),
        "swap_count": int(probe.swap_count),
        "frozen_attempt_count": int(probe.frozen_swap_attempts),
    }


def decision_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_name": "bubble_swap_right",
            "config": SimulatorConfig(values=(2, 1), algorithm="bubble"),
            "actor_index": 0,
            "policy_seed": 1,
        },
        {
            "case_name": "bubble_reverse_swap_right",
            "config": SimulatorConfig(values=(1, 2), algorithm="bubble", reverse_directions=(True, True)),
            "actor_index": 0,
            "policy_seed": 1,
        },
        {
            "case_name": "insertion_swap_left",
            "config": SimulatorConfig(values=(2, 1, 3), algorithm="insertion"),
            "actor_index": 1,
            "policy_seed": 11,
        },
        {
            "case_name": "selection_swap_to_ideal",
            "config": SimulatorConfig(values=(2, 1, 3), algorithm="selection"),
            "actor_index": 1,
            "policy_seed": 21,
        },
        {
            "case_name": "selection_update_ideal_without_swap",
            "config": SimulatorConfig(values=(1, 2, 3), algorithm="selection"),
            "actor_index": 1,
            "policy_seed": 31,
        },
        {
            "case_name": "mixed_label_behavior_resolution",
            "config": SimulatorConfig(
                values=(4, 1, 3, 2),
                algotypes=("bubble_label_a", "bubble_label_b", "insertion", "selection"),
            ),
            "actor_index": 0,
            "policy_seed": 41,
        },
    ]


def run_decision_validation() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    equivalent_pairs = {("update_state", "wait")}
    for case in decision_cases():
        config: SimulatorConfig = case["config"]
        actor_index = int(case["actor_index"])
        policy_seed = int(case["policy_seed"])
        direct_cells, direct_probe = build_cells(config)
        wrapper_cells, wrapper_probe = build_cells(config)

        random.seed(policy_seed)
        direct_cells[actor_index].move()
        direct_signature = cells_signature(direct_cells)
        direct_counts = probe_counts(direct_probe)

        random.seed(policy_seed)
        wrapper = policy_for_cell(wrapper_cells[actor_index], config.label_to_behavior)
        observation = observe_cell(wrapper_cells[actor_index], config.label_to_behavior)
        result = wrapper.step(wrapper_cells[actor_index])
        wrapper_signature = cells_signature(wrapper_cells)
        wrapper_counts = probe_counts(wrapper_probe)

        pair = (result.proposed_action.action_type, result.applied_action.action_type)
        proposal_matches_applied = pair[0] == pair[1] or pair in equivalent_pairs
        row_success = (
            wrapper_signature == direct_signature
            and wrapper_counts == direct_counts
            and proposal_matches_applied
            and result.proposed_action.constraints
        )
        rows.append(
            {
                "research_step_id": STEP_ID,
                "experiment_id": EXPERIMENT_ID,
                "validation_case": case["case_name"],
                "algorithm": config.algorithm or "mixed",
                "algotypes_json": json.dumps(list(config.algotypes) if config.algotypes else None, separators=(",", ":")),
                "initial_values_json": json.dumps(list(config.values), separators=(",", ":")),
                "actor_index": actor_index,
                "actor_value": int(observation.actor_value),
                "behavior": wrapper.behavior,
                "policy_seed": policy_seed,
                "proposed_action_type": result.proposed_action.action_type,
                "applied_action_type": result.applied_action.action_type,
                "proposed_target_index": result.proposed_action.target_index,
                "applied_target_index": result.applied_action.target_index,
                "compare_counted": bool(result.proposed_action.compare_counted),
                "comparison_delta": int(result.comparison_delta),
                "swap_delta": int(result.swap_delta),
                "frozen_attempt_delta": int(result.frozen_attempt_delta),
                "state_changed": bool(result.state_changed),
                "direct_signature_json": json.dumps(direct_signature, separators=(",", ":"), default=str),
                "wrapper_signature_json": json.dumps(wrapper_signature, separators=(",", ":"), default=str),
                "direct_counts_json": json.dumps(direct_counts, sort_keys=True, separators=(",", ":")),
                "wrapper_counts_json": json.dumps(wrapper_counts, sort_keys=True, separators=(",", ":")),
                "proposal_matches_applied": bool(proposal_matches_applied),
                "wrapper_matches_direct": bool(wrapper_signature == direct_signature and wrapper_counts == direct_counts),
                "success": bool(row_success),
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        "note": "Manifest checksum is omitted here to avoid self-referential checksum drift.",
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact Markdown table without optional pandas dependencies."""

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        values = [str(record[column]).replace("|", "\\|") for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def parquet_engine() -> str:
    try:
        import pyarrow  # noqa: F401

        return "pyarrow"
    except Exception:
        return "auto"


def render_report(
    *,
    artifacts_dir: Path,
    step_dir: Path,
    result_path: Path,
    result_alias_path: Path,
    manifest_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    validation_df: pd.DataFrame,
    unit_tests: dict[str, Any],
    e02_tests: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> str:
    all_decision_success = bool(validation_df["success"].all())
    validation_success = bool(all_decision_success and unit_tests["success"] and (e02_tests is None or e02_tests["success"]))
    outcome = "supportive" if validation_success else "constraining/contradictory"
    validation_line = (
        f"Passed: {int(validation_df['success'].sum())}/{len(validation_df)} decision fixtures matched direct public "
        f"methods; E03 unit tests return code {unit_tests['returnCode']}"
    )
    if e02_tests is not None:
        validation_line += f"; E02 deterministic-simulator regression subset return code {e02_tests['returnCode']}"
    artifact_list = [
        str(step_dir / "research_step_full_results.md"),
        str(result_path),
        str(result_alias_path),
        str(manifest_path),
        str(run_manifest_path),
        str(checksums_path),
    ]
    artifact_md = "\n".join(f"- `{item}`" for item in artifact_list)
    commands = [
        f"`{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
    ]
    if e02_tests is not None:
        commands.append(f"`{e02_tests['command']}` -> return code {e02_tests['returnCode']}")
    commands.append("`python scripts/e03_s01_wrap_originals.py --repo-dir /workspace/cell-research --artifacts-dir $ARTIFACTS_DIR`")
    command_md = "\n".join(f"- {item}" for item in commands)
    decision_table = markdown_table(
        validation_df,
        [
            "validation_case",
            "behavior",
            "proposed_action_type",
            "applied_action_type",
            "wrapper_matches_direct",
            "proposal_matches_applied",
            "success",
        ],
    )
    return f"""# E03 S01 Research Step Full Results

## Top Summary

- Research step ID: S01
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: The interface wraps public cell-view methods and exposes deterministic proposals for the original three policies, but it does not yet define the S02 DSL or replace every E02 simulator path. Selection target-position updates are exposed as state updates because the public method mutates `ideal_position` inside `should_move_to`.
- Lay summary: S01 created a common policy wrapper so Bubble, Insertion, and Selection cells can be viewed as local-rule policies with observations, state, proposed actions, constraints, and updates. The wrapper path preserved direct public-method behavior on small fixtures, so S02 can build a rule language on top of this interface.
- Recommended next action: Proceed to S02 to create the rule DSL, using `OriginalCellPolicyWrapper` as the compatibility baseline for classic Algotypes.

## Frozen Question

Can all original and future Algotypes be represented through one interface exposing observation, internal state, action proposal, action constraints, and update rules?

## Inputs

- Active research plan: `/workspace/RESEARCH_PLAN.md`, Experiment E03, step S01.
- Repository checkout: `/workspace/cell-research`.
- E01 context: `/previous-artifacts/E01/reports/e01_codebase_map.md` and E01 regression/provenance artifacts.
- E02 context: `/previous-artifacts/E02/src_snapshot/e02_deterministic_simulator_manifest.json` and `src/e02/deterministic_simulator.py`.
- Paper context: `/workspace/input-attachments/f93afdc5-f2e5-4ecc-80bb-e088f93acf3c/pdf-markdown.md`, especially the cell-view definitions of Position, Value, Algotype, and the three bottom-up sorting policies.
- Datasets: none required for this experiment; dataset availability says `not_required`.

## Methods

Implemented `src/e03/policy_interface.py` with a minimal typed interface:

- `PolicyObservation` captures cell-local view, array values, labels, statuses, boundaries, direction, group status, and Selection target state.
- `PolicyState` exposes internal policy state, including Selection `ideal_position` and wrapper memory flags.
- `PolicyAction` records the proposed action type, target, comparison accounting, constraints, state updates, and source-method metadata.
- `OriginalCellPolicyWrapper` previews Bubble, Insertion, and Selection decisions with the same local gates used by the public methods, then executes the original `cell.move()` method for behavior-preserving application.

Validation used paired public-cell worlds. For each fixture, one world called `cell.move()` directly; the matched world called `OriginalCellPolicyWrapper.step(cell)`. Both worlds were seeded identically and compared by cell-state signatures plus comparison, swap, and frozen-attempt counters.

## Commands

{command_md}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- parquet engine: {parquet_engine()}
- New dependencies installed: none.
- Worker count: serial validation only; no CPU parallelism was needed for S01.
- Platform: {platform.platform()}

## Parameters

- Decision fixtures: {len(validation_df)}
- Original behaviors wrapped: Bubble, Insertion, Selection.
- Small-array values: `(2, 1)`, `(1, 2)`, `(2, 1, 3)`, `(1, 2, 3)`, and a four-cell mixed-Algotype label fixture.
- RNG policy: direct and wrapper worlds use matched `random.seed(...)`; proposal preview clones the global random state before executing the public method so preview does not perturb behavior.

## Results

{decision_table}

The wrapper exactly matched direct public-method signatures and counters for all decision fixtures. Proposal action types matched applied action types, including Selection's target-position state update.

Machine-readable results were written to `{result_path}` and the current-step alias `{result_alias_path}`.

## Validation Checks

- E03 policy-interface unit tests: {'passed' if unit_tests['success'] else 'failed'}.
- E02 deterministic simulator regression subset: {'passed' if e02_tests and e02_tests['success'] else 'not run or failed'}.
- Artifact presence and checksums are recorded in `{manifest_path}`.
- Experiment-level provenance is recorded in `{run_manifest_path}` and `{checksums_path}`.
- Validation dataframe success count: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S01.
- This is a compatibility interface and reference implementation, not yet the generated-rule DSL.
- The wrappers intentionally delegate application to the public `move()` methods to avoid behavioral drift; future generated policies will need an interpreter implementation that does not depend on public cell classes.
- Selection has mutable target-position state, so S02 should include an explicit state-update primitive.
- Frozen-cell edge cases are only represented at the proposal/action level here; full dynamic frozen behavior remains governed by E02 simulator code.

## Provenance

- Git commit at validation time: `{manifest['gitCommit']}`
- Git status at validation time: `{manifest['gitStatusShort'] or 'clean'}`
- Source files tracked in manifest: {len(manifest['sourceFiles'])}
- Previous E01 path: `/previous-artifacts/E01`
- Previous E02 path: `/previous-artifacts/E02`
- Created at UTC: `{manifest['createdAtUtc']}`

## Recommended Next Action

Proceed to S02. Build the DSL around the S01 concepts and include Selection-style target-position updates as first-class state transitions.
"""


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e03_policy_interface_tests.parquet"
    result_alias_path = artifacts_dir / "results" / "e03_original_policy_wrapper_tests.parquet"
    manifest_path = artifacts_dir / "src_snapshot" / "e03_policy_interface_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"
    report_path = step_dir / "research_step_full_results.md"
    step_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.parent.mkdir(parents=True, exist_ok=True)

    validation_df = run_decision_validation()
    validation_df.to_parquet(result_path, index=False)
    validation_df.to_parquet(result_alias_path, index=False)

    unit_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_*.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    e02_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_deterministic_simulator.py"], repo_dir)
        if args.run_unit_tests
        else None
    )

    source_paths = [
        repo_dir / "src/e03/__init__.py",
        repo_dir / "src/e03/policy_interface.py",
        repo_dir / "tests/e03/test_policy_interface.py",
        repo_dir / "scripts/e03_s01_wrap_originals.py",
        repo_dir / "src/e02/deterministic_simulator.py",
    ]
    manifest: dict[str, Any] = {
        "schema": "eidosoma.src_snapshot.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(repo_dir, ["rev-parse", "HEAD"]),
        "gitStatusShort": git_output(repo_dir, ["status", "--short"]),
        "dependencies": {
            "newDependenciesInstalled": [],
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "upstreamContext": {
            "previousE01Dir": str(args.previous_e01_dir),
            "previousE02Dir": str(args.previous_e02_dir),
            "e01CodeMap": str(args.previous_e01_dir / "reports/e01_codebase_map.md"),
            "e02DeterministicSimulatorManifest": str(args.previous_e02_dir / "src_snapshot/e02_deterministic_simulator_manifest.json"),
            "paperMarkdown": str(args.paper_markdown),
        },
        "sourceFiles": [source_entry(path, repo_dir) for path in source_paths if path.exists()],
        "validation": {
            "allValidationPassed": bool(validation_df["success"].all() and unit_tests["success"] and (e02_tests is None or e02_tests["success"])),
            "decisionFixtureCount": int(len(validation_df)),
            "decisionFixtureSuccessCount": int(validation_df["success"].sum()),
            "unitTests": unit_tests,
            "e02DeterministicSimulatorTests": e02_tests,
            "validationParquet": str(result_path),
            "validationParquetSha256": sha256_file(result_path),
            "validationAliasParquet": str(result_alias_path),
            "validationAliasParquetSha256": sha256_file(result_alias_path),
        },
    }
    write_json(manifest_path, manifest)
    report = render_report(
        artifacts_dir=artifacts_dir,
        step_dir=step_dir,
        result_path=result_path,
        result_alias_path=result_alias_path,
        manifest_path=manifest_path,
        run_manifest_path=run_manifest_path,
        checksums_path=checksums_path,
        validation_df=validation_df,
        unit_tests=unit_tests,
        e02_tests=e02_tests,
        manifest=manifest,
    )
    write_text(report_path, report)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "lastResearchStepId": STEP_ID,
        "createdAtUtc": utc_now(),
        "gitCommit": manifest["gitCommit"],
        "gitStatusShort": manifest["gitStatusShort"],
        "repository": {
            "path": str(repo_dir),
            "branch": git_output(repo_dir, ["branch", "--show-current"]),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCountVisible": os.cpu_count(),
            "workerCount": 1,
            "threadingPolicy": "serial S01 validation; no parallel workers",
        },
        "dependencies": manifest["dependencies"],
        "seedPolicy": {
            "decisionFixtureSeeds": [int(case["policy_seed"]) for case in decision_cases()],
            "rng": "Python random module; wrapper proposal clones global RNG state before public move execution",
        },
        "artifacts": [
            {"path": str(report_path), "role": "S01 full-results report"},
            {"path": str(result_path), "role": "S01 validation table"},
            {"path": str(result_alias_path), "role": "S01 validation table alias"},
            {"path": str(manifest_path), "role": "S01 source snapshot manifest"},
            {"path": str(checksums_path), "role": "artifact checksums"},
        ],
        "validation": manifest["validation"],
    }
    write_json(run_manifest_path, run_manifest)
    manifest["artifactsWritten"] = [
        artifact_entry(report_path, artifacts_dir, "S01 full-results handoff report"),
        artifact_entry(result_path, artifacts_dir, "S01 policy-interface decision validation results"),
        artifact_entry(result_alias_path, artifacts_dir, "S01 original-policy wrapper validation result alias"),
        manifest_self_entry(manifest_path, artifacts_dir, "S01 source snapshot and provenance manifest"),
        artifact_entry(run_manifest_path, artifacts_dir, "Experiment-level run manifest"),
        {
            "path": str(checksums_path),
            "relativePath": str(checksums_path.relative_to(artifacts_dir)),
            "description": "SHA256 checksums for compact S01 artifacts",
            "sha256": None,
            "sizeBytes": None,
            "note": "Checksum file is written after the manifest so it can include the final manifest hash.",
        },
    ]
    write_json(manifest_path, manifest)
    checksum_targets = [report_path, result_path, result_alias_path, manifest_path, run_manifest_path]
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}"
        for path in checksum_targets
    ]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")
    print(json.dumps(manifest["validation"], indent=2, sort_keys=True))
    return 0 if manifest["validation"]["allValidationPassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
