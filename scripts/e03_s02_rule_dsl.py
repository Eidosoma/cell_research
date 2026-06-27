#!/usr/bin/env python3
"""Execute E03 S02 rule DSL validation and artifact packaging.

S02 creates a compact JSON rule DSL on top of the S01 morphospace policy
interface, validates parser/serialization/interpreter behavior, writes S02
artifacts, updates provenance, and stops before S03.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace import (  # noqa: E402
    DSLPolicy,
    DSL_VERSION,
    PolicyEventSimulator,
    RuleDslError,
    local_inversion_program,
    null_program,
    parse_rule_program,
    policy_from_spec,
    stochastic_right_program,
)


EXPERIMENT_ID = "E03"
STEP_ID = "S02"
STEP_NUMBER = 2
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "args": args,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {
            "args": args,
            "returncode": None,
            "ok": False,
            "stdout": "",
            "stderr": repr(exc),
        }


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["ok"] else "unknown",
        "branch": branch["stdout"].strip() if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"].strip(),
        "remote": remote["stdout"].strip(),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def run_repo_unit_tests(step_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    result = run_command(cmd, cwd=REPO_ROOT)
    log_path = step_dir / "repo_unit_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + result["stdout"] + "\n\nSTDERR\n" + result["stderr"],
        encoding="utf-8",
    )
    row = {
        "validationFamily": "repo_unit_tests",
        "conditionId": "repo_unit_tests",
        "policyId": "all",
        "success": result["ok"],
        "validationDetail": f"Repository unittest discovery return code {result['returncode']}.",
        "logPath": str(log_path),
    }
    payload = {
        "command": cmd,
        "returnCode": result["returncode"],
        "success": result["ok"],
        "logPath": str(log_path),
    }
    return row, payload


def example_programs() -> dict[str, Any]:
    memory_program = parse_rule_program(
        {
            "version": DSL_VERSION,
            "policy_id": "dsl_memory_signal_demo",
            "name": "memory and signal placeholder demo",
            "initial_state": {"counter": 0},
            "rules": [
                {
                    "name": "remember_actor",
                    "when": [{"op": "always"}],
                    "then": {
                        "action": "remember",
                        "updates": [
                            {"op": "estimate_target_position", "key": "target_position", "direction": "increasing"},
                            {"op": "increment", "key": "counter", "amount": 1, "min": 0, "max": 2},
                            {"op": "set", "key": "seen_value", "value": {"expr": "actor_value"}},
                            {"op": "signal", "channel": "marker", "value": "seen"},
                        ],
                    },
                }
            ],
            "default": {"action": "wait"},
        }
    )
    target_program = parse_rule_program(
        {
            "version": DSL_VERSION,
            "policy_id": "dsl_target_estimate_swap",
            "name": "target estimate swap toy",
            "rules": [
                {
                    "name": "estimate_then_swap",
                    "when": [{"op": "always"}],
                    "then": {
                        "action": "swap_target",
                        "updates": [{"op": "estimate_target_position", "key": "target_position", "direction": "increasing"}],
                    },
                }
            ],
            "default": {"action": "wait"},
        }
    )
    programs = {
        "null_wait": null_program(),
        "local_inversion_cleaner": local_inversion_program(),
        "stochastic_right_swap": stochastic_right_program(),
        "memory_signal_demo": memory_program,
        "target_estimate_swap": target_program,
    }
    return {name: program.to_dict() for name, program in programs.items()}


def run_rule_validations(s01_status_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    examples = example_programs()

    for name, payload in examples.items():
        program = parse_rule_program(payload)
        round_tripped = parse_rule_program(program.to_json())
        rows.append(
            {
                "validationFamily": "parse_round_trip",
                "conditionId": f"round_trip_{name}",
                "policyId": program.policy_id,
                "success": round_tripped.to_dict() == program.to_dict(),
                "validationDetail": "DSL JSON parse and canonical serialization round-trip preserved the program.",
            }
        )
        rows.append(
            {
                "validationFamily": "pretty_print",
                "conditionId": f"pretty_{name}",
                "policyId": program.policy_id,
                "success": program.policy_id in program.pretty() and "when" in program.pretty(),
                "validationDetail": "Pretty printer emits policy ID and readable rule clauses.",
            }
        )

    invalid_cases = [
        (
            "invalid_predicate",
            {"policy_id": "invalid_predicate", "rules": [{"when": [{"op": "teleport"}], "then": {"action": "wait"}}]},
        ),
        (
            "invalid_probability",
            {
                "policy_id": "invalid_probability",
                "rules": [{"when": [{"op": "always"}], "then": {"action": "swap_right", "probability": 1.5}}],
            },
        ),
        (
            "invalid_update_key",
            {
                "policy_id": "invalid_update_key",
                "rules": [{"when": [{"op": "always"}], "then": {"action": "remember", "updates": [{"op": "set"}]}}],
            },
        ),
    ]
    for condition_id, payload in invalid_cases:
        try:
            parse_rule_program(payload)
            success = False
            detail = "Invalid DSL program parsed unexpectedly."
        except RuleDslError as exc:
            success = True
            detail = f"Invalid DSL program rejected: {exc}"
        rows.append(
            {
                "validationFamily": "invalid_programs",
                "conditionId": condition_id,
                "policyId": payload["policy_id"],
                "success": success,
                "validationDetail": detail,
            }
        )

    local_result = PolicyEventSimulator(
        [5, 1, 4, 2, 3],
        DSLPolicy(local_inversion_program()),
        scheduler_seed=10,
        tie_breaker_seed=20,
    ).run(max_activations=100000)
    rows.append(
        {
            "validationFamily": "toy_execution",
            "conditionId": "local_inversion_sorts_small_array",
            "policyId": "dsl_local_inversion_cleaner",
            "success": local_result.completed and local_result.final_values == [1, 2, 3, 4, 5],
            "validationDetail": "Local inversion DSL policy sorted a five-cell toy array.",
            "completed": local_result.completed,
            "stopReason": local_result.stop_reason,
            "swapCount": local_result.swap_count,
            "finalValuesJson": json.dumps(local_result.final_values, separators=(",", ":")),
        }
    )

    null_result = PolicyEventSimulator([3, 1, 2], DSLPolicy(null_program()), scheduler_seed=1, tie_breaker_seed=2).run()
    rows.append(
        {
            "validationFamily": "toy_execution",
            "conditionId": "null_policy_waits",
            "policyId": "dsl_null_wait",
            "success": (not null_result.completed) and null_result.stop_reason == "no_cell_can_move_after_two_checks" and null_result.swap_count == 0,
            "validationDetail": "Null DSL policy stayed executable and terminated by no-move checks without swaps.",
            "completed": null_result.completed,
            "stopReason": null_result.stop_reason,
            "swapCount": null_result.swap_count,
            "finalValuesJson": json.dumps(null_result.final_values, separators=(",", ":")),
        }
    )

    stochastic_kwargs = {
        "initial_values": [4, 1, 3, 2],
        "policies": DSLPolicy(stochastic_right_program()),
        "scheduler_seed": 42,
        "tie_breaker_seed": 99,
    }
    first = PolicyEventSimulator(**stochastic_kwargs).run(max_activations=50)
    second = PolicyEventSimulator(**stochastic_kwargs).run(max_activations=50)
    rows.append(
        {
            "validationFamily": "stochastic_replay",
            "conditionId": "stochastic_right_seed_replay",
            "policyId": "dsl_stochastic_right_swap",
            "success": first.final_values == second.final_values
            and first.swap_count == second.swap_count
            and [row["state_hash"] for row in first.trace_rows] == [row["state_hash"] for row in second.trace_rows],
            "validationDetail": "Stochastic DSL policy replayed exactly with fixed scheduler and tie-breaker seeds.",
            "swapCount": first.swap_count,
            "finalValuesJson": json.dumps(first.final_values, separators=(",", ":")),
        }
    )

    policy = DSLPolicy(parse_rule_program(examples["memory_signal_demo"]))
    sim = PolicyEventSimulator([3, 1, 2], policy, scheduler_seed=1, tie_breaker_seed=2)
    sim.step(forced_cell_id=0)
    state = sim.cells[sim.positions_by_id[0]].state
    rows.append(
        {
            "validationFamily": "state_updates",
            "conditionId": "memory_target_signal_update",
            "policyId": "dsl_memory_signal_demo",
            "success": state.get("target_position") == 2
            and state.get("counter") == 1
            and state.get("seen_value") == 3
            and state.get("signal_marker") == "seen",
            "validationDetail": "Bounded memory, target estimate, actor-value remember, and signal placeholder updates executed.",
            "stateJson": json.dumps(state, sort_keys=True, separators=(",", ":")),
        }
    )

    frozen_sim = PolicyEventSimulator(
        [2, 1],
        DSLPolicy(local_inversion_program()),
        frozen_positions=[1],
        frozen_variant="stuck",
        scheduler_seed=1,
        tie_breaker_seed=1,
    )
    frozen_outcome = frozen_sim.step(forced_cell_id=0)
    rows.append(
        {
            "validationFamily": "world_constraints",
            "conditionId": "stuck_frozen_blocks_dsl_swap",
            "policyId": "dsl_local_inversion_cleaner",
            "success": (not frozen_outcome.swapped) and frozen_outcome.blocked_move_attempt and frozen_sim.current_values() == [2, 1],
            "validationDetail": "S01 world constraints blocked a DSL swap into a stuck Frozen Cell.",
        }
    )

    restored = policy_from_spec(DSLPolicy(local_inversion_program()).to_spec())
    rows.append(
        {
            "validationFamily": "s01_boundary",
            "conditionId": "dsl_policy_spec_restores_via_s01_factory",
            "policyId": "dsl_local_inversion_cleaner",
            "success": restored.to_spec().to_dict() == DSLPolicy(local_inversion_program()).to_spec().to_dict(),
            "validationDetail": "S01 policy factory can restore DSL policies from PolicySpec parameters.",
        }
    )

    s01_status = read_json(s01_status_path) if s01_status_path.exists() else {}
    rows.append(
        {
            "validationFamily": "s01_boundary",
            "conditionId": "s01_completed_anchor",
            "policyId": "s01",
            "success": bool(s01_status.get("success")),
            "validationDetail": "S01 status confirms the morphospace policy interface boundary completed successfully.",
            "logPath": str(s01_status_path),
        }
    )
    return pd.DataFrame(rows), examples


def write_rule_dsl_spec(path: Path, examples: dict[str, Any], artifacts_written: list[str], validation_result: str, caveats: list[str]) -> None:
    primitive_rows = [
        ["Predicates", "always, left_exists, right_exists, target_exists", "Control flow and neighbor availability"],
        ["Predicates", "compare_left, compare_right, compare_target", "Actor value compared to local or stored target cell"],
        ["Predicates", "state_equals, state_compare, position_compare", "Bounded internal state and actor-position checks"],
        ["Actions", "wait, swap_left, swap_right, swap_target", "No-op or world-constrained move proposals"],
        ["Actions", "remember, signal", "State-only updates and no-op signal placeholders"],
        ["Updates", "set, increment, decrement", "Bounded internal memory updates"],
        ["Updates", "estimate_target_position", "Value-based target-position estimate for later Selection-like templates"],
        ["Updates", "signal", "Stores a local signal placeholder as `signal_<channel>` in state"],
        ["Stochasticity", "action.probability", "Seeded action gating through the S01 tie-breaker RNG"],
    ]
    example_rows = [[name, payload["policy_id"], len(payload["rules"])] for name, payload in examples.items()]
    artifact_preview = "\n".join(f"- `{path}`" for path in artifacts_written[:18])
    if len(artifacts_written) > 18:
        artifact_preview += f"\n- ... {len(artifacts_written) - 18} additional artifact path(s) in status.json"
    text = f"""# E03 S02 Rule DSL Specification

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written:
{artifact_preview}
- Validation result: {validation_result}
- Caveats or blockers: {"; ".join(caveats)}
- Recommended next action: stop before S03; after Chief Scientist review, encode known algorithms in S03 using this DSL or document required extensions.

## Frozen Question

A compact grammar can express useful local sorting policies and support generation, mutation, recombination, and analysis.

## Program Shape

Programs are canonical JSON objects with:

- `version`: `{DSL_VERSION}`
- `policy_id`: stable policy identifier
- `name`: human-readable name
- `initial_state`: optional bounded per-cell state defaults
- `rules`: ordered first-match rules, each with `name`, `when`, and `then`
- `default`: fallback action when no rule matches

Each program compiles to `DSLPolicy`, a `LocalRulePolicy` implementation from the S01 `morphospace` interface. World constraints, Frozen Cell rules, scheduler choice, swap execution, and traces remain owned by the S01 `PolicyEventSimulator`.

## Primitive Set

{markdown_table(["Family", "Primitive", "Purpose"], primitive_rows)}

## Example Programs

{markdown_table(["Example", "Policy ID", "Rule count"], example_rows)}

## Grammar Limits

- Rules use ordered first-match semantics; no loops or recursion are allowed.
- All actions are local or state-driven. `swap_target` can only use a target stored in policy state.
- Signals are placeholders stored in local state only; no inter-cell broadcast exists in S02.
- `estimate_target_position` uses a simple value-based estimate and is not an exact Selection encoding claim.
- Stochasticity is limited to per-action probability, driven by the S01 tie-breaker RNG for replayability.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def validation_counts(validation_df: pd.DataFrame) -> pd.DataFrame:
    return (
        validation_df.groupby("validationFamily", dropna=False)["success"]
        .agg(total="count", passed="sum")
        .reset_index()
        .assign(failed=lambda df: df["total"] - df["passed"])
    )


def render_validation_report(validation_df: pd.DataFrame, artifacts_written: list[str], caveats: list[str]) -> str:
    validation_passed = bool(validation_df["success"].all())
    validation_result = "passed" if validation_passed else "failed"
    failed = validation_df[~validation_df["success"].astype(bool)]
    artifact_preview = "\n".join(f"- `{path}`" for path in artifacts_written[:18])
    if len(artifacts_written) > 18:
        artifact_preview += f"\n- ... {len(artifacts_written) - 18} additional artifact path(s) in status.json"
    failure_text = "None"
    if not failed.empty:
        failure_text = markdown_table(
            ["Family", "Condition", "Policy", "Detail"],
            failed[["validationFamily", "conditionId", "policyId", "validationDetail"]].head(20).values.tolist(),
        )
    rows = validation_counts(validation_df)[["validationFamily", "passed", "total", "failed"]].values.tolist()
    return f"""# S02 Validation Report

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written:
{artifact_preview}
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Caveats or blockers: {"; ".join(caveats)}
- Recommended next action: stop before S03 for Chief Scientist review; S03 should encode known algorithms only after approval.

## Validation Counts

{markdown_table(["Validation family", "Passed", "Total", "Failed"], rows)}

## Failed Checks

{failure_text}
"""


def render_summary(validation_df: pd.DataFrame, artifacts_written: list[str], caveats: list[str]) -> str:
    validation_passed = bool(validation_df["success"].all())
    validation_result = "passed" if validation_passed else "failed"
    outcome = OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory"
    return f"""# S02 Summary

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written: `{artifacts_written[0]}` and {len(artifacts_written) - 1} additional files listed in `status.json` and `artifact_manifest.json`.
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Outcome classification: {outcome}
- Caveats or blockers: {"; ".join(caveats)}
- Lay summary: S02 adds a compact JSON rule DSL that compiles into S01 `LocalRulePolicy` objects. The DSL supports ordered predicates, local swap/wait actions, bounded memory updates, target-position estimates, signal placeholders, and seeded stochastic actions. Toy validation shows the null policy waits, the local inversion cleaner sorts a small array, stochastic rules replay exactly, and stuck Frozen Cell constraints still block invalid swaps.
- Recommended next action: stop before S03 for Chief Scientist review. If approved, S03 should encode Bubble, Insertion, and Selection as DSL points or document the minimal extension needed.
"""


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    copied: list[Path] = []
    package_dst = code_dir / "morphospace"
    if package_dst.exists():
        shutil.rmtree(package_dst)
    shutil.copytree(REPO_ROOT / "morphospace", package_dst, ignore=shutil.ignore_patterns("__pycache__"))
    copied.extend(sorted(path for path in package_dst.rglob("*.py")))

    script_dst = code_dir / "scripts" / "e03_s02_rule_dsl.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "e03_s02_rule_dsl.py", script_dst)
    copied.append(script_dst)

    test_dst = code_dir / "tests" / "test_e03_rule_dsl.py"
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "tests" / "test_e03_rule_dsl.py", test_dst)
    copied.append(test_dst)
    return copied


def write_run_manifest(provenance_dir: Path, artifacts_written: list[str], validation_passed: bool) -> Path:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    path = provenance_dir / "run_manifest.json"
    manifest = read_json(path) if path.exists() else {"schema": "eidosoma.run_manifest.v1", "experimentId": EXPERIMENT_ID}
    manifest.update(
        {
            "experimentId": EXPERIMENT_ID,
            "lastResearchStepId": STEP_ID,
            "lastStepNumber": STEP_NUMBER,
            "updatedAt": utc_now(),
            "git": get_git_metadata(),
            "platform": platform.platform(),
            "python": sys.version,
        }
    )
    manifest.setdefault("researchSteps", {})
    manifest["researchSteps"][STEP_ID] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "status": STATUS,
        "success": validation_passed,
        "artifactsWritten": artifacts_written,
        "validationResult": "passed" if validation_passed else "failed",
        "completedAt": utc_now(),
        "outcomeClassification": OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory",
    }
    write_json(path, manifest)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    provenance_dir = artifacts_dir / "provenance"
    s01_status_path = artifacts_dir / "research_steps" / "S01" / "status.json"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    validation_df, examples = run_rule_validations(s01_status_path)
    repo_row, repo_command = run_repo_unit_tests(step_dir)
    validation_df = pd.concat([validation_df, pd.DataFrame([repo_row])], ignore_index=True)

    validation_csv = step_dir / "rule_dsl_validation.csv"
    validation_parquet = step_dir / "rule_dsl_validation.parquet"
    validation_results_csv = results_dir / "e03_s02_rule_dsl_validation.csv"
    validation_results_parquet = results_dir / "e03_s02_rule_dsl_validation.parquet"
    validation_df.to_csv(validation_csv, index=False)
    validation_df.to_parquet(validation_parquet, index=False)
    validation_df.to_csv(validation_results_csv, index=False)
    validation_df.to_parquet(validation_results_parquet, index=False)

    examples_json = step_dir / "rule_examples.json"
    examples_pretty = step_dir / "rule_examples_pretty.txt"
    write_json(examples_json, {"schema": "eidosoma.e03.s02.rule_examples.v1", "programs": examples})
    examples_pretty.write_text(
        "\n\n".join(parse_rule_program(payload).pretty() for payload in examples.values()) + "\n",
        encoding="utf-8",
    )

    copied_code = copy_code_artifacts(step_dir)
    rule_spec_path = step_dir / "rule_dsl_spec.md"
    validation_report_path = step_dir / "validation_report.md"
    summary_path = step_dir / "summary.md"
    run_manifest_path = provenance_dir / "run_manifest.json"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"

    artifacts_written_paths = [
        rule_spec_path,
        validation_csv,
        validation_parquet,
        validation_results_csv,
        validation_results_parquet,
        examples_json,
        examples_pretty,
        step_dir / "repo_unit_test_log.txt",
        *copied_code,
        validation_report_path,
        summary_path,
        run_manifest_path,
        status_path,
        artifact_manifest_path,
    ]
    artifacts_written = [str(path) for path in artifacts_written_paths]
    validation_passed = bool(validation_df["success"].all())
    validation_result = (
        "passed: DSL parse/round-trip, invalid-rule rejection, deterministic and stochastic toy execution, state updates, S01 boundary restoration, and repository tests passed"
        if validation_passed
        else "failed: one or more S02 rule DSL validations failed"
    )
    caveats = [
        "S02 validates a compact JSON DSL and toy policies, not full Bubble/Insertion/Selection encodings; those are deferred to S03.",
        "Signals are local state placeholders only and do not implement inter-cell communication.",
        "The value-based target-position estimate is a bounded proxy primitive, not a claim that Selection is fully encoded.",
        "No generated policy corpus, competence vector, GPU simulator, or S03 artifacts were started.",
    ]
    recommended_next_action = "Stop before S03 for Chief Scientist review; if approved, encode known algorithms in S03."

    write_rule_dsl_spec(rule_spec_path, examples, artifacts_written, validation_result, caveats)
    validation_report_path.write_text(render_validation_report(validation_df, artifacts_written, caveats), encoding="utf-8")
    summary_path.write_text(render_summary(validation_df, artifacts_written, caveats), encoding="utf-8")
    write_run_manifest(provenance_dir, artifacts_written, validation_passed)

    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_passed,
        "status": STATUS,
        "artifactsWritten": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory",
        "startedAt": started_at,
        "completedAt": utc_now(),
        "workerCount": 1,
        "threadEnvironment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "repoUnitTestCommand": repo_command,
        "s01StatusPath": str(s01_status_path),
        "git": get_git_metadata(),
    }
    write_json(status_path, status)

    artifact_manifest = {
        "schema": "eidosoma.e03.s02.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "artifacts": collect_artifacts([Path(path) for path in artifacts_written]),
    }
    write_json(artifact_manifest_path, artifact_manifest)
    artifact_manifest["artifacts"] = collect_artifacts([Path(path) for path in artifacts_written])
    write_json(artifact_manifest_path, artifact_manifest)

    print(json.dumps({"success": validation_passed, "statusPath": str(status_path)}, indent=2))
    return 0 if validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
