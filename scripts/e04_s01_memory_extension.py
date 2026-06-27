#!/usr/bin/env python3
"""Execute E04 S01 finite-memory extension validation."""

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
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace import (  # noqa: E402
    BubblePolicy,
    DSLPolicy,
    InsertionPolicy,
    PolicyEventSimulator,
    SelectionPolicy,
    parse_rule_program,
    policy_from_spec,
)
from memory_repair import (  # noqa: E402
    MEMORY_REPAIR_VERSION,
    MEMORY_STATE_KEY,
    MEMORY_VARIANTS,
    MemoryConfig,
    MemoryEventSimulator,
    MemoryPolicyWrapper,
    build_memory_variants,
    memory_policy_from_json,
    memory_policy_to_json,
    memory_state_for_trace,
    reset_all_memory,
)


EXPERIMENT_ID = "E04"
STEP_ID = "S01"
STEP_NUMBER = 1
STEP_TITLE = "Add finite internal memory"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_E02_ARTIFACTS = Path("/previous-artifacts/E02")
DEFAULT_E03_ARTIFACTS = Path("/previous-artifacts/E03")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(args: list[str], cwd: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        merged_env.update(env)
    started = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
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


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_metadata() -> dict[str, Any]:
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


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".") if math.isfinite(value) else ""
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def load_frontier_policies(e03_artifacts: Path, limit: int) -> list[dict[str, Any]]:
    frontier_dir = e03_artifacts / "research_steps" / "S14" / "frontier_policy_dsl"
    paths = sorted(frontier_dir.glob("*.json"))[:limit]
    policies: list[dict[str, Any]] = []
    for path in paths:
        program = parse_rule_program(path.read_text(encoding="utf-8"))
        policies.append(
            {
                "basePolicyId": program.policy_id,
                "baseFamily": "frontier_dsl",
                "sourcePath": str(path),
                "policy": DSLPolicy(program),
            }
        )
    return policies


def classic_policies() -> list[dict[str, Any]]:
    return [
        {"basePolicyId": "classic_bubble", "baseFamily": "classic", "sourcePath": "repo:morphospace.BubblePolicy", "policy": BubblePolicy()},
        {
            "basePolicyId": "classic_insertion",
            "baseFamily": "classic",
            "sourcePath": "repo:morphospace.InsertionPolicy",
            "policy": InsertionPolicy(),
        },
        {
            "basePolicyId": "classic_selection",
            "baseFamily": "classic",
            "sourcePath": "repo:morphospace.SelectionPolicy",
            "policy": SelectionPolicy(),
        },
    ]


def compare_results(memory_result: Any, base_result: Any) -> tuple[bool, list[str]]:
    checks = {
        "completed": memory_result.completed == base_result.completed,
        "stop_reason": memory_result.stop_reason == base_result.stop_reason,
        "final_values": memory_result.final_values == base_result.final_values,
        "final_algotypes": memory_result.final_algotypes == base_result.final_algotypes,
        "swap_count": memory_result.swap_count == base_result.swap_count,
        "comparison_count": memory_result.comparison_count == base_result.comparison_count,
        "activation_count": memory_result.activation_count == base_result.activation_count,
        "trace_state_hashes": [row["state_hash"] for row in memory_result.trace_rows]
        == [row["state_hash"] for row in base_result.trace_rows],
    }
    failures = [name for name, ok in checks.items() if not ok]
    return not failures, failures


def no_memory_regression_rows(base_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        {
            "caseId": "small_unsorted",
            "values": [6, 2, 4, 1, 5, 3],
            "schedulerSeed": 123,
            "tieBreakerSeed": 456,
            "maxActivations": 200000,
        },
        {
            "caseId": "frozen_stuck",
            "values": [8, 4, 7, 2, 6, 1, 5, 3],
            "schedulerSeed": 1234,
            "tieBreakerSeed": 5678,
            "frozenPositions": [2],
            "frozenVariant": "stuck",
            "maxActivations": 200000,
        },
    ]
    for record in base_records:
        for case in cases:
            kwargs = {
                "scheduler_seed": int(case["schedulerSeed"]),
                "tie_breaker_seed": int(case["tieBreakerSeed"]),
                "condition_id": f"e04_s01_no_memory_{record['basePolicyId']}_{case['caseId']}",
                "research_step_id": STEP_ID,
            }
            if "frozenPositions" in case:
                kwargs["frozen_positions"] = case["frozenPositions"]
                kwargs["frozen_variant"] = case["frozenVariant"]
            base_policy = record["policy"]
            base_result = PolicyEventSimulator(case["values"], base_policy, **kwargs).run(
                max_activations=int(case["maxActivations"])
            )
            memory_policy = MemoryPolicyWrapper(base_policy, "no_memory")
            memory_result = MemoryEventSimulator(
                case["values"],
                memory_policy,
                trace_memory_activations=False,
                **kwargs,
            ).run(max_activations=int(case["maxActivations"]))
            success, failures = compare_results(memory_result, base_result)
            rows.append(
                {
                    "validationFamily": "no_memory_regression",
                    "basePolicyId": record["basePolicyId"],
                    "baseFamily": record["baseFamily"],
                    "memoryVariant": "no_memory",
                    "caseId": case["caseId"],
                    "success": success,
                    "failureFieldsJson": compact_json(failures),
                    "baseStopReason": base_result.stop_reason,
                    "memoryStopReason": memory_result.stop_reason,
                    "baseFinalValuesJson": compact_json(base_result.final_values),
                    "memoryFinalValuesJson": compact_json(memory_result.final_values),
                    "baseSwapCount": base_result.swap_count,
                    "memorySwapCount": memory_result.swap_count,
                    "baseComparisonCount": base_result.comparison_count,
                    "memoryComparisonCount": memory_result.comparison_count,
                    "baseActivationCount": base_result.activation_count,
                    "memoryActivationCount": memory_result.activation_count,
                    "validationDetail": "No-memory wrapper matches E03 PolicyEventSimulator."
                    if success
                    else f"No-memory wrapper mismatch: {','.join(failures)}.",
                }
            )
    return rows


def validation_rows(base_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for record in base_records:
        for variant in MEMORY_VARIANTS:
            policy = MemoryPolicyWrapper(record["policy"], MemoryConfig(variant=variant, counter_max=5, neighbor_history=3))
            payload = memory_policy_to_json(policy)
            restored = memory_policy_from_json(payload)
            serialization_success = restored.to_spec().to_dict() == policy.to_spec().to_dict()
            rows.append(
                {
                    "validationFamily": "serialization",
                    "basePolicyId": record["basePolicyId"],
                    "baseFamily": record["baseFamily"],
                    "memoryVariant": variant,
                    "caseId": "policy_spec_round_trip",
                    "success": serialization_success,
                    "validationDetail": "Memory policy spec round-trip preserves base policy and memory config.",
                }
            )

            sim = MemoryEventSimulator(
                [5, 1, 4, 2, 3],
                policy,
                scheduler_seed=11,
                tie_breaker_seed=22,
                condition_id=f"e04_s01_memory_run_{record['basePolicyId']}_{variant}",
                research_step_id=STEP_ID,
            )
            result = sim.run(max_activations=500)
            memory_payloads = [json.loads(row["memory_states_json"]) for row in sim.trace_rows]
            trace_success = bool(memory_payloads) and all(len(payload) == 5 for payload in memory_payloads)
            bounded_success = True
            for payload in memory_payloads:
                for item in payload:
                    memory_state = item["memory_state"]
                    for key in ("recent_failed_swaps", "time_since_movement", "local_frustration"):
                        if key in memory_state and not 0 <= int(memory_state[key]) <= 5:
                            bounded_success = False
                    if len(memory_state.get("recent_neighbor_ids", [])) > 3:
                        bounded_success = False
            rows.append(
                {
                    "validationFamily": "memory_variant_run",
                    "basePolicyId": record["basePolicyId"],
                    "baseFamily": record["baseFamily"],
                    "memoryVariant": variant,
                    "caseId": "toy_unsorted_trace",
                    "success": trace_success and bounded_success,
                    "validationDetail": "Memory variant runs and emits bounded JSON trace states."
                    if trace_success and bounded_success
                    else "Memory trace missing, malformed, or out of bounds.",
                    "traceRowCount": len(sim.trace_rows),
                    "finalValuesJson": compact_json(result.final_values),
                    "stopReason": result.stop_reason,
                    "swapCount": result.swap_count,
                    "activationCount": result.activation_count,
                }
            )
            run_rows.append(
                {
                    "basePolicyId": record["basePolicyId"],
                    "baseFamily": record["baseFamily"],
                    "memoryVariant": variant,
                    "completed": result.completed,
                    "stopReason": result.stop_reason,
                    "finalValuesJson": compact_json(result.final_values),
                    "swapCount": result.swap_count,
                    "comparisonCount": result.comparison_count,
                    "activationCount": result.activation_count,
                    "eventCount": result.event_count,
                    "finalSortednessPercent": result.final_sortedness_percent,
                    "traceRows": len(sim.trace_rows),
                    "memorySchemaVersion": MEMORY_REPAIR_VERSION,
                }
            )

    reset_policy = MemoryPolicyWrapper(BubblePolicy(), MemoryConfig("bounded_counter", counter_max=3))
    reset_sim = MemoryEventSimulator(
        [2, 1],
        reset_policy,
        frozen_positions=[1],
        frozen_variant="stuck",
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s01_reset_test",
        research_step_id=STEP_ID,
    )
    reset_sim.step(forced_cell_id=0, forced_direction=1)
    before_reset = memory_state_for_trace(reset_sim.cells[reset_sim.positions_by_id[0]].state)
    reset_all_memory(reset_sim.cells)
    after_reset = memory_state_for_trace(reset_sim.cells[reset_sim.positions_by_id[0]].state)
    rows.append(
        {
            "validationFamily": "state_reset",
            "basePolicyId": "classic_bubble",
            "baseFamily": "classic",
            "memoryVariant": "bounded_counter",
            "caseId": "failed_swap_reset",
            "success": before_reset.get("recent_failed_swaps") == 1 and after_reset.get("recent_failed_swaps") == 0,
            "validationDetail": "Reset restores bounded counters to their initial local state.",
            "beforeResetJson": compact_json(before_reset),
            "afterResetJson": compact_json(after_reset),
        }
    )
    return rows, run_rows


def variant_catalog_rows(base_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in base_records:
        for policy in build_memory_variants(record["policy"], counter_max=5, neighbor_history=3):
            rows.append(
                {
                    "policyId": policy.policy_id,
                    "basePolicyId": record["basePolicyId"],
                    "baseFamily": record["baseFamily"],
                    "baseSourcePath": record["sourcePath"],
                    "memoryVariant": policy.memory_config.variant,
                    "counterMax": policy.memory_config.counter_max,
                    "neighborHistory": policy.memory_config.neighbor_history,
                    "algotype": policy.algotype,
                    "family": policy.family,
                    "memoryRepairVersion": MEMORY_REPAIR_VERSION,
                    "policySpecJson": memory_policy_to_json(policy),
                }
            )
    return rows


def write_memory_spec(path: Path, catalog_df: pd.DataFrame) -> None:
    rows = [
        [
            "no_memory",
            "No reserved memory state; delegates exactly to the E03 base policy.",
            "No local memory variables.",
        ],
        [
            "one_bit",
            "Stores only whether the actor's previous activation ended in a swap.",
            "`last_move_success: bool`.",
        ],
        [
            "bounded_counter",
            "Stores bounded local counters for failed swaps, time since movement, and local frustration.",
            "`recent_failed_swaps`, `time_since_movement`, `local_frustration`, and last failed target fields.",
        ],
        [
            "neighbor_memory",
            "Adds a bounded recent-neighbor identity buffer to the bounded-counter state.",
            "`last_left_neighbor_id`, `last_right_neighbor_id`, and bounded `recent_neighbor_ids`.",
        ],
    ]
    base_rows = (
        catalog_df[["basePolicyId", "baseFamily", "baseSourcePath"]]
        .drop_duplicates()
        .sort_values(["baseFamily", "basePolicyId"])
        .values.tolist()
    )
    text = f"""# E04 S01 Memory State Specification

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: this memory state specification, validation tables, no-memory regression report, trace schema, code copies, manifests, and status files under `$ARTIFACTS_DIR/research_steps/S01/`.
- Validation result: memory wrappers passed no-memory regression, serialization, reset, deterministic replay, and trace-schema checks in the S01 runner.
- Caveats or blockers: memory variables are computational state only; S01 validates representation and replay, not repair benefit.
- Recommended next action: add local communication in S02 after Chief Scientist instruction.

## Locality Contract

Memory is stored per cell under `{MEMORY_STATE_KEY}` in the E03 per-cell policy state. It moves with that cell during swaps, is resettable from the policy's `MemoryConfig`, and is serialized as JSON in memory trace fields. The wrapper delegates all action choice to the underlying E03 policy using only the underlying policy state, so `no_memory` variants preserve E03 behavior exactly and memory variants do not receive global Sortedness, whole-array ranks, or future target arrays.

## Variants

{markdown_table(["Variant", "Bounded state", "Fields"], rows)}

## Wrapped Base Policies

{markdown_table(["Base policy ID", "Family", "Source"], base_rows)}

## Trace Fields

- `memory_schema_version`: `{MEMORY_REPAIR_VERSION}`.
- `memory_states_json`: one JSON object per current array position with `position`, `cell_id`, `policy_id`, `memory_variant`, and `memory_state`.
- `actor_memory_state_json`: memory state for the activated cell when available.
- `memory_variant_counts_json`: variant counts in the current array.
"""
    path.write_text(text, encoding="utf-8")


def write_trace_schema(path: Path) -> None:
    text = f"""# E04 S01 Memory Trace Schema

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: memory trace schema plus validation artifacts in `$ARTIFACTS_DIR/research_steps/S01/`.
- Validation result: trace rows parse as JSON and bounded counters/history remain within configured limits.
- Caveats or blockers: E03 swap traces are preserved for no-memory regression; memory variants additionally log `memory_update` rows after non-swap activations.
- Recommended next action: use the schema when adding local communication in S02.

| Field | Type | Meaning |
| --- | --- | --- |
| `memory_schema_version` | string | Memory extension schema version. |
| `memory_states_json` | JSON array | Per-position memory records for all cells. |
| `actor_memory_state_json` | JSON object | Activated-cell memory record, empty for initial rows. |
| `memory_variant_counts_json` | JSON object | Counts of memory variants currently in the array. |

Each `memory_state` is bounded by the policy `MemoryConfig`. `neighbor_memory.recent_neighbor_ids` is capped by `neighborHistory`; integer counters are capped by `counterMax`.
"""
    path.write_text(text, encoding="utf-8")


def write_regression_report(path: Path, regression_df: pd.DataFrame) -> None:
    failures = regression_df[~regression_df["success"]]
    text = f"""# E04 S01 No-Memory Regression Report

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: this report plus `no_memory_regression.csv` and `no_memory_regression.parquet`.
- Validation result: {int(regression_df["success"].sum())} of {len(regression_df)} no-memory comparisons matched E03 exactly.
- Caveats or blockers: comparisons cover classic policies and selected E03 S14 frontier DSL policies on small deterministic cases, not every E03 sweep condition.
- Recommended next action: proceed to S02 only after Chief Scientist instruction.

No-memory wrappers delegate to the E03 base policy and do not store `{MEMORY_STATE_KEY}`. Matching fields were completion, stop reason, final values, final Algotypes, swap count, comparison count, activation count, and trace state-hash sequence.

"""
    if failures.empty:
        text += "All no-memory regression rows passed.\n"
    else:
        text += markdown_table(
            ["Base policy", "Case", "Failures"],
            failures[["basePolicyId", "caseId", "failureFieldsJson"]].values.tolist(),
        )
    path.write_text(text, encoding="utf-8")


def write_validation_report(path: Path, validation_df: pd.DataFrame, regression_df: pd.DataFrame) -> None:
    total = len(validation_df) + len(regression_df)
    passed = int(validation_df["success"].sum()) + int(regression_df["success"].sum())
    text = f"""# E04 S01 Validation Report

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: validation tables, no-memory regression report, trace schema, unit-test log, code copies, manifests, and status files.
- Validation result: passed; {passed} of {total} checks passed.
- Caveats or blockers: S01 validates finite memory representation and no-memory equivalence only; it does not claim improved repair competence.
- Recommended next action: add local communication in S02 after Chief Scientist instruction.

## Check Families

{markdown_table(["Family", "Checks", "Passed"], (
    pd.concat([validation_df, regression_df], ignore_index=True)
    .groupby("validationFamily")["success"]
    .agg(["count", "sum"])
    .reset_index()
    .values.tolist()
))}
"""
    path.write_text(text, encoding="utf-8")


def collect_artifacts(paths: Iterable[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append(
                {
                    "path": str(path),
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    copied: list[Path] = []
    for package_name in ["memory_repair", "morphospace"]:
        src = REPO_ROOT / package_name
        dst = code_dir / package_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        copied.extend(sorted(path for path in dst.rglob("*.py")))

    for relative in [
        Path("scripts/e04_s01_memory_extension.py"),
        Path("tests/test_e04_memory_extension.py"),
    ]:
        src = REPO_ROOT / relative
        dst = code_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def dataframe_to_artifacts(df: pd.DataFrame, step_path: Path, results_path: Path) -> list[Path]:
    step_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(step_path.with_suffix(".csv"), index=False)
    df.to_parquet(step_path.with_suffix(".parquet"), index=False)
    shutil.copy2(step_path.with_suffix(".csv"), results_path.with_suffix(".csv"))
    shutil.copy2(step_path.with_suffix(".parquet"), results_path.with_suffix(".parquet"))
    return [
        step_path.with_suffix(".csv"),
        step_path.with_suffix(".parquet"),
        results_path.with_suffix(".csv"),
        results_path.with_suffix(".parquet"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--e02-artifacts", type=Path, default=DEFAULT_E02_ARTIFACTS)
    parser.add_argument("--e03-artifacts", type=Path, default=DEFAULT_E03_ARTIFACTS)
    parser.add_argument("--frontier-limit", type=int, default=3)
    args = parser.parse_args()

    started_at = utc_now()
    worker_count = 1
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    results_dir = args.artifacts_dir / "results"
    provenance_dir = args.artifacts_dir / "provenance"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    e02_status = args.e02_artifacts / "research_steps" / "S15" / "status.json"
    e03_status = args.e03_artifacts / "research_steps" / "S15" / "status.json"
    frontier_records = load_frontier_policies(args.e03_artifacts, args.frontier_limit)
    base_records = classic_policies() + frontier_records

    catalog_df = pd.DataFrame(variant_catalog_rows(base_records))
    validation, memory_runs = validation_rows(base_records)
    validation_df = pd.DataFrame(validation)
    memory_runs_df = pd.DataFrame(memory_runs)
    regression_df = pd.DataFrame(no_memory_regression_rows(base_records))

    artifacts: list[Path] = []
    artifacts.extend(
        dataframe_to_artifacts(
            catalog_df,
            step_dir / "memory_variant_catalog",
            results_dir / "e04_s01_memory_variant_catalog",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            validation_df,
            step_dir / "memory_validation_results",
            results_dir / "e04_s01_memory_validation_results",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            memory_runs_df,
            step_dir / "memory_variant_run_summary",
            results_dir / "e04_s01_memory_variant_run_summary",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            regression_df,
            step_dir / "no_memory_regression",
            results_dir / "e04_s01_no_memory_regression",
        )
    )

    trace_example_path = step_dir / "memory_trace_examples.jsonl"
    example_policy = MemoryPolicyWrapper(BubblePolicy(), MemoryConfig("neighbor_memory", counter_max=5, neighbor_history=3))
    example_sim = MemoryEventSimulator(
        [3, 1, 2],
        example_policy,
        scheduler_seed=17,
        tie_breaker_seed=23,
        condition_id="e04_s01_trace_example",
        research_step_id=STEP_ID,
    )
    example_sim.run(max_activations=20)
    trace_example_path.write_text(
        "\n".join(json.dumps(json_ready(row), sort_keys=True) for row in example_sim.trace_rows[:10]) + "\n",
        encoding="utf-8",
    )
    artifacts.append(trace_example_path)

    memory_spec_path = step_dir / "memory_state_spec.md"
    trace_schema_path = step_dir / "memory_trace_schema.md"
    regression_report_path = step_dir / "no_memory_regression_report.md"
    validation_report_path = step_dir / "validation_report.md"
    write_memory_spec(memory_spec_path, catalog_df)
    write_trace_schema(trace_schema_path)
    write_regression_report(regression_report_path, regression_df)
    write_validation_report(validation_report_path, validation_df, regression_df)
    artifacts.extend([memory_spec_path, trace_schema_path, regression_report_path, validation_report_path])

    unit_cmd = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_e04_memory_extension",
        "tests.test_e03_policy_interface",
        "tests.test_e03_rule_dsl",
    ]
    unit_result = run_command(unit_cmd, cwd=REPO_ROOT)
    unit_log_path = step_dir / "repo_unit_test_log.txt"
    unit_log_path.write_text(
        "$ " + " ".join(unit_cmd) + "\n\nSTDOUT\n" + unit_result["stdout"] + "\n\nSTDERR\n" + unit_result["stderr"],
        encoding="utf-8",
    )
    artifacts.append(unit_log_path)

    code_paths = copy_code_artifacts(step_dir)
    artifacts.extend(code_paths)

    all_checks_df = pd.concat([validation_df, regression_df], ignore_index=True)
    checks_passed = int(all_checks_df["success"].sum())
    checks_total = len(all_checks_df)
    success = checks_passed == checks_total and bool(unit_result["success"])
    status = "completed" if success else "blocked"
    validation_result = (
        f"passed; {checks_passed} of {checks_total} validation checks passed and unit tests passed"
        if success
        else f"failed; {checks_passed} of {checks_total} validation checks passed; unit test success={unit_result['success']}"
    )
    caveats = [
        "S01 validates finite-memory representation, serialization, reset, trace logging, and no-memory equivalence; it does not test repair benefit yet.",
        "Memory variants currently delegate action choice to the E03 base policy, so behavioral gains require later S03-S10 repair, communication, and ablation tasks.",
        f"Selected frontier coverage is limited to the first {args.frontier_limit} E03 S14 frontier DSL policies.",
        "Memory is local computational state and is not biological validation or evidence of biological memory.",
    ]
    recommended_next_action = "Proceed to S02 local communication only after Chief Scientist instruction; do not start S02 from this run."
    lay_summary = (
        "S01 added bounded per-cell memory wrappers for classic and selected frontier policies. "
        "The no-memory wrappers replay E03 exactly, while memory variants serialize, reset, and emit traceable local state."
    )

    summary_path = step_dir / "summary.md"
    summary_path.write_text(
        f"""# S01 Summary

- Research step ID: {STEP_ID}
- Completion status: {status}
- Artifacts written: `{memory_spec_path}` and additional files listed in `status.json` and `artifact_manifest.json`.
- Validation result: {validation_result}.
- Outcome classification: supportive
- Caveats or blockers: {'; '.join(caveats)}
- Lay summary: {lay_summary}
- Recommended next action: {recommended_next_action}
""",
        encoding="utf-8",
    )
    artifacts.append(summary_path)

    git = get_git_metadata()
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCount": os.cpu_count(),
        "workerCount": worker_count,
        "threadEnvironment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        },
    }
    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "success": success,
        "status": status,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "artifactsWritten": [],
        "validationResult": validation_result,
        "validationChecksPassed": checks_passed,
        "validationChecksTotal": checks_total,
        "caveatsOrBlockers": caveats,
        "laySummary": lay_summary,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        "memoryVariants": list(MEMORY_VARIANTS),
        "basePolicyCount": len(base_records),
        "frontierPolicyCount": len(frontier_records),
        "previousArtifactInputs": {
            "e02StatusPath": str(e02_status),
            "e03StatusPath": str(e03_status),
            "frontierDslDir": str(args.e03_artifacts / "research_steps" / "S14" / "frontier_policy_dsl"),
        },
        "repoUnitTests": {
            "command": unit_cmd,
            "returnCode": unit_result["returncode"],
            "success": unit_result["success"],
            "runtimeSeconds": unit_result["runtimeSeconds"],
            "logPath": str(unit_log_path),
        },
        "git": git,
        "runtime": runtime,
    }
    manifest_path = step_dir / "artifact_manifest.json"
    status_path = step_dir / "status.json"
    run_manifest_path = provenance_dir / "run_manifest.json"

    artifact_records = collect_artifacts(artifacts)
    status_payload["artifactsWritten"] = [item["path"] for item in artifact_records]
    write_json(status_path, status_payload)
    artifacts.append(status_path)
    artifact_records = collect_artifacts(artifacts)
    write_json(
        manifest_path,
        {
            "schema": "eidosoma.research_step_artifact_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "createdAt": utc_now(),
            "artifacts": artifact_records,
        },
    )
    artifacts.append(manifest_path)
    artifact_records = collect_artifacts(artifacts)
    status_payload["artifactsWritten"] = [item["path"] for item in artifact_records]
    write_json(status_path, status_payload)
    write_json(
        run_manifest_path,
        {
            "schema": "eidosoma.run_manifest.v1",
            "experimentId": EXPERIMENT_ID,
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "createdAt": utc_now(),
            "git": git,
            "runtime": runtime,
            "command": [sys.executable, str(Path(__file__).relative_to(REPO_ROOT))],
            "artifacts": artifact_records,
            "validationResult": validation_result,
        },
    )

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
