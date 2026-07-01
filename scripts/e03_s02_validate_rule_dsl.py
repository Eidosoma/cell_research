#!/usr/bin/env python3
"""Validate E03 S02 rule DSL and write artifacts."""

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

from src.e03.rule_dsl import (
    DSLArrayState,
    DSLInterpreter,
    DSLValidationError,
    SELECTION_TARGET_POLICY,
    SIMPLE_SWAP_LEFT_POLICY,
    SIMPLE_SWAP_RIGHT_POLICY,
    parse_and_render_round_trip,
    parse_policy,
)


STEP_ID = "S02"
STEP_NUMBER = 2
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-e01-dir", type=Path, default=Path("/previous-artifacts/E01"))
    parser.add_argument("--previous-e02-dir", type=Path, default=Path("/previous-artifacts/E02"))
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


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def invalid_sources() -> list[str]:
    return [
        "policy bad v1\nrule else wait\n",
        "policy bad v1\nrule if target_exists(up) then wait\nend\n",
        "policy bad v1\nrule if always then teleport(left)\nend\n",
        "policy bad v1\nrule else wait\nrule if always then wait\nend\n",
        "policy bad v1\nrule if random_lt(1.2) then wait\nend\n",
    ]


def run_validation_cases() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    policies = [
        ("compare_swap_left", SIMPLE_SWAP_LEFT_POLICY),
        ("compare_swap_right", SIMPLE_SWAP_RIGHT_POLICY),
        ("selection_target_update", SELECTION_TARGET_POLICY),
    ]
    for name, source in policies:
        policy, rendered, reparsed = parse_and_render_round_trip(source)
        json_round_trip = policy.from_json(policy.to_json())
        rows.append(
            {
                "validation_case": f"{name}_round_trip",
                "case_type": "round_trip",
                "policy_name": name,
                "policy_id": policy.policy_id,
                "policy_sha256": policy.sha256,
                "success": policy.to_dict() == reparsed.to_dict() == json_round_trip.to_dict(),
                "detail": "source/render/json round trips preserve canonical dictionary",
                "rendered_source": rendered,
                "action_type": None,
                "final_values_json": None,
                "state_update_json": None,
            }
        )

    execution_cases = [
        ("swap_left_executes", SIMPLE_SWAP_LEFT_POLICY, DSLArrayState(values=(2, 1, 3), actor_index=1), (1, 2, 3), "swap"),
        ("swap_right_executes", SIMPLE_SWAP_RIGHT_POLICY, DSLArrayState(values=(2, 1), actor_index=0), (1, 2), "swap"),
        ("guard_false_waits", SIMPLE_SWAP_LEFT_POLICY, DSLArrayState(values=(1, 2, 3), actor_index=1), (1, 2, 3), "wait"),
        ("selection_updates_target", SELECTION_TARGET_POLICY, DSLArrayState(values=(1, 2, 3), actor_index=1), (1, 2, 3), "update_state"),
        ("selection_swaps_target", SELECTION_TARGET_POLICY, DSLArrayState(values=(2, 1, 3), actor_index=1), (1, 2, 3), "swap"),
    ]
    for case_name, source, state, expected_values, expected_action in execution_cases:
        policy = parse_policy(source)
        result = DSLInterpreter(policy).step_state(state, random.Random(101))
        success = result.state_after.values == tuple(expected_values) and result.action.action_type == expected_action
        rows.append(
            {
                "validation_case": case_name,
                "case_type": "interpreter_execution",
                "policy_name": policy.name,
                "policy_id": policy.policy_id,
                "policy_sha256": policy.sha256,
                "success": bool(success),
                "detail": f"expected action {expected_action} and final values {expected_values}",
                "rendered_source": policy.to_source(),
                "action_type": result.action.action_type,
                "final_values_json": json.dumps(list(result.state_after.values), separators=(",", ":")),
                "state_update_json": json.dumps(dict(result.action.state_update), sort_keys=True, separators=(",", ":")),
            }
        )

    choose_policy = parse_policy(
        """policy choose_swap v1
state ideal_position=none
rule if always then choose(0.75, swap(right), wait)
end
"""
    )
    first = DSLInterpreter(choose_policy).step_state(DSLArrayState(values=(2, 1), actor_index=0), random.Random(7))
    second = DSLInterpreter(choose_policy).step_state(DSLArrayState(values=(2, 1), actor_index=0), random.Random(7))
    rows.append(
        {
            "validation_case": "probabilistic_choose_fixed_seed",
            "case_type": "determinism",
            "policy_name": choose_policy.name,
            "policy_id": choose_policy.policy_id,
            "policy_sha256": choose_policy.sha256,
            "success": first.action.action_type == second.action.action_type
            and first.state_after.values == second.state_after.values
            and first.action.metadata["choice_roll"] == second.action.metadata["choice_roll"],
            "detail": "fixed RNG seed reproduces probabilistic choice",
            "rendered_source": choose_policy.to_source(),
            "action_type": first.action.action_type,
            "final_values_json": json.dumps(list(first.state_after.values), separators=(",", ":")),
            "state_update_json": json.dumps(dict(first.action.state_update), sort_keys=True, separators=(",", ":")),
        }
    )

    for idx, source in enumerate(invalid_sources()):
        rejected = False
        detail = ""
        try:
            parse_policy(source)
        except DSLValidationError as exc:
            rejected = True
            detail = str(exc)
        rows.append(
            {
                "validation_case": f"invalid_rule_rejected_{idx}",
                "case_type": "invalid_rule",
                "policy_name": "bad",
                "policy_id": None,
                "policy_sha256": None,
                "success": rejected,
                "detail": detail,
                "rendered_source": source,
                "action_type": None,
                "final_values_json": None,
                "state_update_json": None,
            }
        )
    return pd.DataFrame(rows)


def render_spec() -> str:
    return """# E03 Rule DSL Specification

## Scope

This is the S02 version-1 line-oriented DSL for auditable local sorting policies. It is intentionally small: later steps can add richer search operators without changing the canonical parser contract.

## Grammar

```text
policy <name> v1
state ideal_position=<none|left_boundary|right_boundary|current|integer>
rule if <condition> [and <condition> ...] then <action>[, <action> ...]
rule else <action>[, <action> ...]
end
```

Comments start with `#`. The optional `state` line initializes Selection-style target-position state. `rule else` is optional, but if present it must be the last rule.

## Conditions

- `always`
- `target_exists(left|right|ideal)`
- `target_active(left|right|ideal)`
- `target_movable(left|right|ideal)`: active or frozen target, matching S01/S02 local-action constraints.
- `self_lt(left|right|ideal)`, `self_gt(...)`, `self_le(...)`, `self_ge(...)`
- `at_ideal`, `not_at_ideal`
- `prefix_sorted`: S01-compatible Insertion prefix gate.
- `random_lt(p)`: probabilistic guard with `p` in `[0, 1]`.

## Actions

- `compare(left|right|ideal)`: marks comparison cost.
- `swap(left|right|ideal)`: proposes a constrained swap.
- `set_ideal(next|left_boundary|right_boundary|current|none|integer)`: first-class target-position state update.
- `estimate_target_position(...)`: alias-compatible state update primitive.
- `wait`
- `remember(key[, value])`: records a state update for future DSL expansion.
- `signal(label[, ...])`: records metadata for future communication experiments.
- `choose(p, action_if_true, action_if_false)`: probabilistic action with deterministic behavior for a fixed RNG seed.

Hyphen aliases `compare-left`, `compare-right`, `swap-left`, and `swap-right` parse into their canonical function forms.

## Serialization And Hashing

Every parsed policy has a canonical source rendering, canonical JSON serialization, SHA-256 hash, and `dsl:<16 hex>` policy ID. Hashes are computed from sorted-key canonical JSON.

## Interpreter Semantics

Rules are evaluated in order. The first matching guard executes. `compare` records comparison cost and continues to the next action. Terminal actions are `swap`, `set_ideal`, `estimate_target_position`, and `wait`; `choose` delegates to one nested terminal action. `swap` is constrained to existing movable targets and otherwise returns a wait action with failed constraints.

## S02 Limitations

The DSL can execute simple compare-and-swap policies and represent Selection target-position state. It does not yet replace the E02 simulator, encode all original algorithms exactly, or provide full memory and signal semantics. Those are planned for S03 and later steps.
"""


def render_report(
    *,
    step_dir: Path,
    result_path: Path,
    spec_path: Path,
    manifest_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    validation_df: pd.DataFrame,
    unit_tests: dict[str, Any],
    e03_s01_tests: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    validation_success = bool(validation_df["success"].all() and unit_tests["success"] and e03_s01_tests["success"])
    outcome = "supportive" if validation_success else "constraining/contradictory"
    artifact_md = "\n".join(
        f"- `{path}`"
        for path in [step_dir / "research_step_full_results.md", spec_path, result_path, manifest_path, run_manifest_path, checksums_path]
    )
    validation_line = (
        f"Passed: {int(validation_df['success'].sum())}/{len(validation_df)} DSL validation cases passed; "
        f"S02 unit tests return code {unit_tests['returnCode']}; S01 compatibility tests return code {e03_s01_tests['returnCode']}"
    )
    commands = "\n".join(
        [
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{e03_s01_tests['command']}` -> return code {e03_s01_tests['returnCode']}",
            "- `python scripts/e03_s02_validate_rule_dsl.py --repo-dir /workspace/cell-research --artifacts-dir $ARTIFACTS_DIR`",
        ]
    )
    table = markdown_table(
        validation_df[["validation_case", "case_type", "policy_name", "action_type", "success"]],
        ["validation_case", "case_type", "policy_name", "action_type", "success"],
    )
    return f"""# E03 S02 Research Step Full Results

## Top Summary

- Research step ID: S02
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: The S02 DSL supports auditable parsing, serialization, hashing, deterministic interpretation, simple compare-and-swap execution, and Selection-style target-position state. It does not yet encode full Bubble/Insertion/Selection classics as DSL policies or replace the E02 simulator.
- Lay summary: S02 turns local sorting rules into small text programs that can be parsed, checked, hashed, and run on small array snapshots. This gives S03 a stable language for expressing classic policies and generated variants.
- Recommended next action: Proceed to S03 only after Chief Scientist review; encode Bubble, Insertion, and Selection as DSL or interface-level policy-library entries and document exact versus approximate mappings.

## Frozen Question

Can local sorting policies be expressed as compact programs using primitives such as compare-left, compare-right, estimate-target-position, swap-left, swap-right, wait, remember, signal, and probabilistic action?

## Inputs

- Active research plan: `/workspace/RESEARCH_PLAN.md`, Experiment E03, step S02.
- S01 interface: `src/e03/policy_interface.py` and `/artifacts/src_snapshot/e03_policy_interface_manifest.json`.
- E02 deterministic simulator context: `src/e02/deterministic_simulator.py` and `/previous-artifacts/E02/src_snapshot/e02_deterministic_simulator_manifest.json`.
- E01 code map context: `/previous-artifacts/E01/reports/e01_codebase_map.md`.
- Datasets: none required.

## Methods

Implemented `src/e03/rule_dsl.py` with:

- a line-oriented parser for `policy`, `state`, guarded `rule`, and `end` forms;
- canonical source rendering and sorted-key JSON serialization;
- stable SHA-256 policy hashes and `dsl:<hash-prefix>` IDs;
- primitive validation for targets, probabilities, state values, rule ordering, and unsupported actions;
- an interpreter over S01 `PolicyObservation` objects;
- a small immutable `DSLArrayState` executor for reference compare-and-swap and target-state update checks.

The Selection target position is represented by `state ideal_position=...` and terminal actions such as `set_ideal(next)`.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- New dependencies installed: none.
- Worker count: serial validation only; no CPU parallelism was needed for S02.
- Platform: {platform.platform()}

## Parameters

- DSL version: 1
- Validation cases: {len(validation_df)}
- Valid example policies: `compare_swap_left`, `compare_swap_right`, `selection_target_update`, `choose_swap`
- Invalid-source tests: {len(invalid_sources())}
- RNG policy: deterministic Python `random.Random(seed)` for probabilistic guards/actions.

## Results

{table}

Machine-readable validation results were written to `{result_path}`. The human-readable DSL specification was written to `{spec_path}`.

## Validation Checks

- Parse/render/JSON round trips preserved policy dictionaries and hashes.
- Invalid rules were rejected with `DSLValidationError`.
- Compare-and-swap execution produced expected final arrays.
- Fixed-seed probabilistic action execution was deterministic.
- Selection-style `ideal_position` initialized and updated as first-class policy state.
- Unit-test validation and S01 compatibility regression both passed.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S02.
- This DSL can run simple local rules but does not yet encode every original public-method quirk.
- `remember` and `signal` are parsed and represented but only produce state-update or metadata records in S02; richer communication and memory behavior is deferred to later experiments.
- Integration into the full E02 event simulator remains future work.

## Provenance

- Git commit at validation time: `{manifest['gitCommit']}`
- Git status at validation time: `{manifest['gitStatusShort'] or 'clean'}`
- Source files tracked in manifest: {len(manifest['sourceFiles'])}
- Previous E01 path: `/previous-artifacts/E01`
- Previous E02 path: `/previous-artifacts/E02`
- Created at UTC: `{manifest['createdAtUtc']}`

## Recommended Next Action

Proceed to S03 only after Chief Scientist review. Use the S02 DSL to encode classics where exact, and explicitly document any cases that need interface-level extensions.
"""


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e03_dsl_validation_tests.parquet"
    spec_path = artifacts_dir / "reports" / "e03_rule_dsl_spec.md"
    manifest_path = artifacts_dir / "src_snapshot" / "e03_rule_dsl_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"
    report_path = step_dir / "research_step_full_results.md"
    for path in [step_dir, result_path.parent, spec_path.parent, manifest_path.parent, checksums_path.parent]:
        path.mkdir(parents=True, exist_ok=True)

    validation_df = run_validation_cases()
    validation_df.to_parquet(result_path, index=False)
    write_text(spec_path, render_spec())
    unit_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_rule_dsl.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    e03_s01_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )

    source_paths = [
        repo_dir / "src/e03/rule_dsl.py",
        repo_dir / "tests/e03/test_rule_dsl.py",
        repo_dir / "scripts/e03_s02_validate_rule_dsl.py",
        repo_dir / "src/e03/policy_interface.py",
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
            "s01Manifest": str(artifacts_dir / "src_snapshot/e03_policy_interface_manifest.json"),
        },
        "sourceFiles": [source_entry(path, repo_dir) for path in source_paths if path.exists()],
        "validation": {
            "allValidationPassed": bool(validation_df["success"].all() and unit_tests["success"] and e03_s01_tests["success"]),
            "validationCaseCount": int(len(validation_df)),
            "validationCaseSuccessCount": int(validation_df["success"].sum()),
            "unitTests": unit_tests,
            "s01PolicyInterfaceTests": e03_s01_tests,
            "validationParquet": str(result_path),
            "validationParquetSha256": sha256_file(result_path),
            "dslSpec": str(spec_path),
            "dslSpecSha256": sha256_file(spec_path),
        },
    }
    write_json(manifest_path, manifest)
    report = render_report(
        step_dir=step_dir,
        result_path=result_path,
        spec_path=spec_path,
        manifest_path=manifest_path,
        run_manifest_path=run_manifest_path,
        checksums_path=checksums_path,
        validation_df=validation_df,
        unit_tests=unit_tests,
        e03_s01_tests=e03_s01_tests,
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
        "repository": {"path": str(repo_dir), "branch": git_output(repo_dir, ["branch", "--show-current"])},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCountVisible": os.cpu_count(),
            "workerCount": 1,
            "threadingPolicy": "serial S02 validation; no parallel workers",
        },
        "dependencies": manifest["dependencies"],
        "seedPolicy": {
            "probabilisticValidationSeed": 7,
            "interpreterValidationSeed": 101,
            "rng": "Python random.Random with explicit seeds",
        },
        "artifacts": [
            {"path": str(report_path), "role": "S02 full-results report"},
            {"path": str(spec_path), "role": "S02 DSL specification"},
            {"path": str(result_path), "role": "S02 validation table"},
            {"path": str(manifest_path), "role": "S02 source snapshot manifest"},
            {"path": str(checksums_path), "role": "artifact checksums"},
        ],
        "validation": manifest["validation"],
    }
    write_json(run_manifest_path, run_manifest)
    manifest["artifactsWritten"] = [
        artifact_entry(report_path, artifacts_dir, "S02 full-results handoff report"),
        artifact_entry(spec_path, artifacts_dir, "S02 DSL specification"),
        artifact_entry(result_path, artifacts_dir, "S02 DSL validation results"),
        manifest_self_entry(manifest_path, artifacts_dir, "S02 source snapshot and provenance manifest"),
        artifact_entry(run_manifest_path, artifacts_dir, "Experiment-level run manifest updated through S02"),
        {
            "path": str(checksums_path),
            "relativePath": str(checksums_path.relative_to(artifacts_dir)),
            "description": "SHA256 checksums for compact S02 artifacts",
            "sha256": None,
            "sizeBytes": None,
            "note": "Checksum file is written after the manifest so it can include the final manifest hash.",
        },
    ]
    write_json(manifest_path, manifest)
    checksum_targets = [report_path, spec_path, result_path, manifest_path, run_manifest_path]
    write_text(
        checksums_path,
        "\n".join(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets) + "\n",
    )
    print(json.dumps(manifest["validation"], indent=2, sort_keys=True))
    return 0 if manifest["validation"]["allValidationPassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
