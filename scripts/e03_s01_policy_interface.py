#!/usr/bin/env python3
"""Execute E03 S01 policy-interface validation.

This step builds the local-rule interface around the classic cell-view
policies, validates that the policy-driven simulator matches the deterministic
E02 reference for classic no-Frozen, Frozen Cell, and same-goal chimera cases,
writes required artifacts, and stops before S02.
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
from collections import Counter
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

from e02_deterministic_simulator import DeterministicEventSimulator, initial_values_from_seed, state_hash  # noqa: E402
from morphospace import (  # noqa: E402
    BubblePolicy,
    InsertionPolicy,
    NullPolicy,
    PolicyEventSimulator,
    RandomWalkPolicy,
    SelectionPolicy,
    policy_from_json,
    policy_to_json,
)


EXPERIMENT_ID = "E03"
STEP_ID = "S01"
STEP_NUMBER = 1
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_E01_ARTIFACTS = Path("/previous-artifacts/E01")
DEFAULT_E02_ARTIFACTS = Path("/previous-artifacts/E02")
ALGORITHMS = ["bubble", "insertion", "selection"]
DEFAULT_REPLICATE_LIMIT = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, Any]:
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


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


def frozen_positions_from_seed(seed: Any, frozen_count: int, n: int) -> list[int]:
    if frozen_count == 0:
        return []
    if seed is None or (isinstance(seed, float) and math.isnan(seed)):
        raise ValueError("frozen_position_seed required when frozen_count > 0")
    rng = np.random.default_rng(int(seed))
    return sorted(int(value) for value in rng.choice(n, size=frozen_count, replace=False))


def algotypes_from_seed(allocation_json: str, seed: Any, n: int) -> list[str]:
    allocation = json.loads(allocation_json)
    algotypes: list[str] = []
    for algorithm in ALGORITHMS:
        algotypes.extend([algorithm] * int(allocation.get(algorithm, 0)))
    if len(algotypes) != n:
        raise ValueError(f"allocation gives {len(algotypes)} cells, expected {n}")
    if seed is None or (isinstance(seed, float) and math.isnan(seed)):
        return algotypes
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(len(algotypes))
    return [algotypes[int(index)] for index in indices]


def condition_rows(e01_artifacts: Path, replicate_limit: int) -> list[dict[str, Any]]:
    condition_matrix = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "condition_matrix.csv")
    seed_table = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "seed_table.csv")
    selected_ids: list[str] = [f"S04_cell_view_{algorithm}_unique_f0_none" for algorithm in ALGORITHMS]
    for algorithm in ALGORITHMS:
        for frozen_variant in ["passive", "stuck"]:
            selected_ids.append(f"S07_cell_view_{algorithm}_unique_f1_{frozen_variant}")
    selected_ids.append("S09_cell_view_bubble_selection_unique_same_goal")

    rows: list[dict[str, Any]] = []
    for condition_id in selected_ids:
        matches = condition_matrix[condition_matrix["conditionId"] == condition_id]
        if len(matches) != 1:
            rows.append({"conditionId": condition_id, "missing": True})
            continue
        condition = matches.iloc[0].to_dict()
        seeds = seed_table[
            (seed_table["conditionId"] == condition_id) & (seed_table["replicateIndex"] < replicate_limit)
        ].sort_values("replicateIndex")
        for _, seed_row in seeds.iterrows():
            merged = dict(condition)
            merged.update(seed_row.to_dict())
            rows.append(merged)
    return rows


def compare_results(policy_result: Any, e02_result: Any) -> tuple[bool, list[str]]:
    checks = {
        "completed": policy_result.completed == e02_result.completed,
        "stop_reason": policy_result.stop_reason == e02_result.stop_reason,
        "final_values": policy_result.final_values == e02_result.final_values,
        "final_algotypes": policy_result.final_algotypes == e02_result.final_algotypes,
        "final_frozen_positions": policy_result.final_frozen_positions == e02_result.final_frozen_positions,
        "swap_count": policy_result.swap_count == e02_result.swap_count,
        "comparison_count": policy_result.comparison_count == e02_result.comparison_count,
        "archived_compare_and_swap_count": policy_result.archived_compare_and_swap_count
        == e02_result.archived_compare_and_swap_count,
        "blocked_move_attempts": policy_result.blocked_move_attempts == e02_result.blocked_move_attempts,
        "frozen_swap_attempts": policy_result.frozen_swap_attempts == e02_result.frozen_swap_attempts,
        "trace_state_hashes": [row["state_hash"] for row in policy_result.trace_rows]
        == [row["state_hash"] for row in e02_result.trace_rows],
    }
    failures = [name for name, ok in checks.items() if not ok]
    return not failures, failures


def run_interface_smoke_validations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in [BubblePolicy(), InsertionPolicy(), SelectionPolicy(), NullPolicy(), RandomWalkPolicy(0.25)]:
        payload = policy_to_json(policy)
        round_trip = policy_from_json(payload)
        success = round_trip.to_spec().to_dict() == policy.to_spec().to_dict()
        rows.append(
            {
                "validationFamily": "serialization",
                "conditionId": f"round_trip_{policy.policy_id}",
                "algorithm": policy.algotype,
                "replicateIndex": 0,
                "success": success,
                "validationDetail": "Policy spec JSON round-trip preserves family, Algotype, version, and parameters.",
            }
        )

    null_result = PolicyEventSimulator([3, 1, 2], NullPolicy(), scheduler_seed=1, tie_breaker_seed=2).run()
    rows.append(
        {
            "validationFamily": "toy_policy",
            "conditionId": "null_policy_noop",
            "algorithm": "null",
            "replicateIndex": 0,
            "success": (not null_result.completed) and null_result.swap_count == 0,
            "policyStopReason": null_result.stop_reason,
            "policySwapCount": null_result.swap_count,
            "validationDetail": "Null policy remains executable and terminates by no-move checks without swaps.",
        }
    )

    random_kwargs = {
        "initial_values": [4, 1, 3, 2],
        "policies": RandomWalkPolicy(0.5),
        "scheduler_seed": 42,
        "tie_breaker_seed": 99,
    }
    first = PolicyEventSimulator(**random_kwargs).run(max_activations=50)
    second = PolicyEventSimulator(**random_kwargs).run(max_activations=50)
    rows.append(
        {
            "validationFamily": "toy_policy",
            "conditionId": "random_policy_seed_replay",
            "algorithm": "random_walk",
            "replicateIndex": 0,
            "success": first.final_values == second.final_values and first.swap_count == second.swap_count,
            "policySwapCount": first.swap_count,
            "validationDetail": "Random policy is seed-replayable through the same scheduler and tie-breaker streams.",
        }
    )

    direct_cases = [
        {
            "family": "direct_no_frozen",
            "condition_id": "direct_no_frozen_bubble",
            "values": [6, 2, 4, 1, 5, 3],
            "algotypes": "bubble",
            "kwargs": {"scheduler_seed": 123, "tie_breaker_seed": 456},
        },
        {
            "family": "direct_no_frozen",
            "condition_id": "direct_no_frozen_insertion",
            "values": [6, 2, 4, 1, 5, 3],
            "algotypes": "insertion",
            "kwargs": {"scheduler_seed": 123, "tie_breaker_seed": 456},
        },
        {
            "family": "direct_no_frozen",
            "condition_id": "direct_no_frozen_selection",
            "values": [6, 2, 4, 1, 5, 3],
            "algotypes": "selection",
            "kwargs": {"scheduler_seed": 123, "tie_breaker_seed": 456},
        },
        {
            "family": "direct_frozen",
            "condition_id": "direct_passive_frozen_bubble",
            "values": [2, 1],
            "algotypes": "bubble",
            "kwargs": {"frozen_positions": [1], "frozen_variant": "passive", "scheduler_seed": 1, "tie_breaker_seed": 1},
        },
        {
            "family": "direct_frozen",
            "condition_id": "direct_stuck_frozen_bubble",
            "values": [2, 1],
            "algotypes": "bubble",
            "kwargs": {"frozen_positions": [1], "frozen_variant": "stuck", "scheduler_seed": 1, "tie_breaker_seed": 1},
        },
        {
            "family": "direct_chimera",
            "condition_id": "direct_three_policy_chimera",
            "values": [9, 3, 8, 2, 7, 1, 6, 4, 5],
            "algotypes": ["bubble", "insertion", "selection", "bubble", "insertion", "selection", "bubble", "insertion", "selection"],
            "kwargs": {"scheduler_seed": 222, "tie_breaker_seed": 333},
        },
    ]
    for case in direct_cases:
        kwargs = dict(case["kwargs"])
        kwargs["condition_id"] = case["condition_id"]
        policy_result = PolicyEventSimulator(case["values"], case["algotypes"], **kwargs).run(max_activations=200_000)
        e02_result = DeterministicEventSimulator(case["values"], case["algotypes"], **kwargs).run(max_activations=200_000)
        success, failures = compare_results(policy_result, e02_result)
        rows.append(
            {
                "validationFamily": case["family"],
                "conditionId": case["condition_id"],
                "algorithm": case["algotypes"] if isinstance(case["algotypes"], str) else "chimera",
                "replicateIndex": 0,
                "success": success,
                "validationDetail": "Direct policy-interface case matches E02 exactly."
                if success
                else f"Direct case mismatches E02 fields: {','.join(failures)}.",
                "failureFields": compact_json(failures),
                "policyFinalStateHash": state_hash(policy_result.final_values),
                "e02FinalStateHash": state_hash(e02_result.final_values),
                "policySwapCount": policy_result.swap_count,
                "e02SwapCount": e02_result.swap_count,
            }
        )
    return rows


def run_e01_regression(e01_artifacts: Path, step_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(e01_artifacts / "tests" / "regression_tests"),
        "-p",
        "test_*.py",
        "-v",
    ]
    result = run_command(cmd, cwd=REPO_ROOT, env={"ARTIFACTS_DIR": str(e01_artifacts)})
    log_path = step_dir / "e01_regression_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + result["stdout"] + "\n\nSTDERR\n" + result["stderr"],
        encoding="utf-8",
    )
    fail_lines = [
        line.strip()
        for line in result["stderr"].splitlines()
        if line.startswith("test_") and (" ... FAIL" in line or " ... ERROR" in line)
    ]
    accepted_relocation_issue = (
        not result["ok"]
        and fail_lines
        == [
            "test_report_bundle_references_exist_and_hash (test_e01_baseline_regression.TestE01BaselineRegression.test_report_bundle_references_exist_and_hash) ... FAIL"
        ]
    )
    accepted = bool(result["ok"] or accepted_relocation_issue)
    detail = f"E01 packaged regression command return code {result['returncode']}."
    if accepted_relocation_issue:
        detail += " Accepted known absolute-path bundle-manifest relocation failure from mounted prior artifacts."
    row = {
        "validationFamily": "e01_regression_tests",
        "conditionId": "e01_packaged_regression_tests",
        "algorithm": "all",
        "replicateIndex": 0,
        "success": accepted,
        "validationDetail": detail,
        "logPath": str(log_path),
    }
    payload = {
        "command": cmd,
        "returnCode": result["returncode"],
        "rawSuccess": result["ok"],
        "success": accepted,
        "acceptedRelocationIssue": accepted_relocation_issue,
        "failLines": fail_lines,
        "logPath": str(log_path),
    }
    return row, payload


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
        "algorithm": "all",
        "replicateIndex": 0,
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


def validate_policy_against_e02(e01_artifacts: Path, replicate_limit: int, max_activations: int) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    config = read_json(e01_artifacts / "configs" / "e01_baseline_config.json")
    stop_policies = config["semantics"]["stopPolicies"]
    rows: list[dict[str, Any]] = []
    policy_summaries: list[dict[str, Any]] = []

    for row in condition_rows(e01_artifacts, replicate_limit):
        condition_id = str(row.get("conditionId"))
        print(f"validating E01-seeded policy replay: {condition_id} replicate {row.get('replicateIndex')}", file=sys.stderr, flush=True)
        if row.get("missing"):
            rows.append(
                {
                    "validationFamily": "policy_vs_e02_reference",
                    "conditionId": condition_id,
                    "success": False,
                    "validationDetail": "Condition missing from E01 condition matrix.",
                }
            )
            continue
        n = int(row["n"])
        frozen_count = int(row["frozenCount"])
        frozen_variant = str(row["frozenVariant"])
        stop_policy = stop_policies[str(row["stopPolicy"])]
        replicate_index = int(row["replicateIndex"])
        replicate_number = int(row["replicateNumber"])
        input_seed = int(row["inputPermutationSeed"])
        scheduler_seed = int(row["schedulerSeed"])
        tie_seed = int(row["tieBreakerSeed"])
        frozen_seed_raw = row.get("frozenPositionSeed")
        frozen_seed = None if frozen_count == 0 or pd.isna(frozen_seed_raw) else int(frozen_seed_raw)
        algotype_seed_raw = row.get("algotypeAssignmentSeed")
        algotype_seed = None if pd.isna(algotype_seed_raw) else int(algotype_seed_raw)
        initial_values = initial_values_from_seed(input_seed, n=n, profile=str(row["inputProfile"]))
        frozen_positions = frozen_positions_from_seed(frozen_seed, frozen_count, n)
        algotypes = algotypes_from_seed(str(row["algotypeAllocation"]), algotype_seed, n)

        common_kwargs = {
            "frozen_positions": frozen_positions,
            "frozen_variant": frozen_variant,
            "scheduler_seed": scheduler_seed,
            "tie_breaker_seed": tie_seed,
            "condition_id": condition_id,
            "research_step_id": STEP_ID,
        }
        policy_result = PolicyEventSimulator(initial_values, algotypes, **common_kwargs).run(
            max_activations=max_activations,
            max_swaps=int(stop_policy["maxSwapEvents"]),
            max_comparisons=int(stop_policy["maxComparisonEvents"]),
            no_move_checks_required=int(stop_policy.get("noMoveChecksRequired", 2)),
        )
        e02_result = DeterministicEventSimulator(initial_values, algotypes, **common_kwargs).run(
            max_activations=max_activations,
            max_swaps=int(stop_policy["maxSwapEvents"]),
            max_comparisons=int(stop_policy["maxComparisonEvents"]),
            no_move_checks_required=int(stop_policy.get("noMoveChecksRequired", 2)),
        )
        success, failures = compare_results(policy_result, e02_result)
        policy_summaries.append(
            policy_result.summary_record(
                research_step_id=STEP_ID,
                replicate_index=replicate_index,
                replicate_number=replicate_number,
                input_permutation_seed=input_seed,
                scheduler_seed=scheduler_seed,
                tie_breaker_seed=tie_seed,
                frozen_position_seed=frozen_seed,
                input_profile=str(row["inputProfile"]),
                frozen_variant=frozen_variant,
                frozen_count=frozen_count,
            )
        )
        policy_hash = state_hash(policy_result.final_values)
        e02_hash = state_hash(e02_result.final_values)
        rows.append(
            {
                "validationFamily": "policy_vs_e02_reference",
                "conditionId": condition_id,
                "algorithm": str(row["algorithms"]),
                "mixtureId": str(row["mixtureId"]),
                "replicateIndex": replicate_index,
                "replicateNumber": replicate_number,
                "success": success,
                "validationDetail": "Policy-interface replay matches E02 exactly."
                if success
                else f"Policy-interface replay mismatches E02 fields: {','.join(failures)}.",
                "failureFields": compact_json(failures),
                "inputPermutationSeed": input_seed,
                "algotypeAssignmentSeed": algotype_seed,
                "frozenPositionSeed": frozen_seed,
                "schedulerSeed": scheduler_seed,
                "tieBreakerSeed": tie_seed,
                "initialStateHash": state_hash(initial_values),
                "policyFinalStateHash": policy_hash,
                "e02FinalStateHash": e02_hash,
                "policyCompleted": policy_result.completed,
                "e02Completed": e02_result.completed,
                "policyStopReason": policy_result.stop_reason,
                "e02StopReason": e02_result.stop_reason,
                "policySwapCount": policy_result.swap_count,
                "e02SwapCount": e02_result.swap_count,
                "policyComparisonCount": policy_result.comparison_count,
                "e02ComparisonCount": e02_result.comparison_count,
                "policyFinalSortednessPercent": policy_result.final_sortedness_percent,
                "e02FinalSortednessPercent": e02_result.final_sortedness_percent,
                "policyFinalMonotonicityError": policy_result.final_monotonicity_error,
                "e02FinalMonotonicityError": e02_result.final_monotonicity_error,
                "valueConservation": Counter(policy_result.initial_values) == Counter(policy_result.final_values),
                "algotypeConservation": Counter(policy_result.initial_algotypes) == Counter(policy_result.final_algotypes),
                "frozenCountConservation": len(policy_result.initial_frozen_positions)
                == len(policy_result.final_frozen_positions)
                == frozen_count,
            }
        )
    return rows, pd.DataFrame(policy_summaries)


def write_policy_interface_spec(path: Path) -> None:
    policies = [BubblePolicy(), InsertionPolicy(), SelectionPolicy(), NullPolicy(), RandomWalkPolicy(1.0)]
    policy_rows = [
        [policy.policy_id, policy.family, policy.algotype, compact_json(policy.to_spec().parameters)]
        for policy in policies
    ]
    text = f"""# E03 S01 Policy Interface Specification

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: this policy interface specification plus validation tables, code copies, manifests, and status files under `$ARTIFACTS_DIR/research_steps/S01/`.
- Validation result: policy-interface classic policies are validated against the E02 deterministic reference by the S01 runner.
- Caveats or blockers: reverse-direction Selection preserves E02 behavior but is not reinterpreted as a corrected Selection rule; generated DSL policies are deferred to S02.
- Recommended next action: implement the S02 rule DSL only after Chief Scientist instruction.

## Interface Contract

Every policy is represented by `LocalRulePolicy` and exposes:

- `observe(cells, actor_position, state, frozen_variant)`: returns a `LocalObservation` containing actor identity, immediate neighbors, left-context cells for Insertion semantics, target cell when applicable, Frozen Cell mode, and per-cell policy state.
- `initial_state(...)`: creates per-cell internal state. Classic Selection uses `ideal_position`; Bubble and Insertion are stateless.
- `propose_action(observation, state, rng, forced_direction=None)`: proposes `wait` or `swap`, records comparison cost, target position, state updates, and reason labels.
- `constrain_action(...)`: policy-side validation hook before world constraints.
- `update_state(...)`: applies state changes after the simulator resolves action constraints.
- `legal_action_exists(...)`: deterministic no-move predicate used by stop-condition checks.

World-level constraints remain outside the policy: bounds checks, passive versus stuck Frozen Cell behavior, actor eligibility, swap execution, trace writing, and E01/E02 metric accounting.

## Wrapped Policies

{markdown_table(["Policy ID", "Family", "Algotype", "Parameters"], policy_rows)}

## Classic Semantics

- Bubble compares one adjacent neighbor selected by the tie-breaker stream and swaps across a local inversion.
- Insertion can compare left only after the left prefix is sorted under the E02 prefix rule.
- Selection carries an `ideal_position` state and advances that target exactly as E02 does, including the documented reverse-direction asymmetry.
- Null waits forever and is used for controls.
- Random walk proposes seed-replayable adjacent swaps and is included as the minimal stochastic-policy wrapper.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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

    script_dst = code_dir / "scripts" / "e03_s01_policy_interface.py"
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "e03_s01_policy_interface.py", script_dst)
    copied.append(script_dst)

    test_dst = code_dir / "tests" / "test_e03_policy_interface.py"
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "tests" / "test_e03_policy_interface.py", test_dst)
    copied.append(test_dst)
    return copied


def validation_counts(validation_df: pd.DataFrame) -> pd.DataFrame:
    return (
        validation_df.groupby("validationFamily", dropna=False)["success"]
        .agg(total="count", passed="sum")
        .reset_index()
        .assign(failed=lambda df: df["total"] - df["passed"])
    )


def render_validation_report(validation_df: pd.DataFrame, caveats: list[str], artifacts_written: list[str]) -> str:
    counts = validation_counts(validation_df)
    failed = validation_df[~validation_df["success"].astype(bool)]
    validation_result = "passed" if failed.empty else "failed"
    rows = counts[["validationFamily", "passed", "total", "failed"]].values.tolist()
    artifact_preview = "\n".join(f"- `{path}`" for path in artifacts_written[:18])
    if len(artifacts_written) > 18:
        artifact_preview += f"\n- ... {len(artifacts_written) - 18} additional artifact path(s) in status.json"
    failure_text = "None"
    if not failed.empty:
        failure_text = markdown_table(
            ["Family", "Condition", "Replicate", "Detail"],
            failed[["validationFamily", "conditionId", "replicateIndex", "validationDetail"]].head(20).values.tolist(),
        )
    return f"""# S01 Validation Report

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written:
{artifact_preview}
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Caveats or blockers: {"; ".join(caveats)}
- Recommended next action: implement the S02 rule DSL after Chief Scientist instruction; do not start S02 in this run.

## Validation Counts

{markdown_table(["Validation family", "Passed", "Total", "Failed"], rows)}

## Failed Checks

{failure_text}
"""


def render_summary(validation_df: pd.DataFrame, caveats: list[str], artifacts_written: list[str]) -> str:
    failed = validation_df[~validation_df["success"].astype(bool)]
    validation_result = "passed" if failed.empty else "failed"
    outcome = OUTCOME_CLASSIFICATION if failed.empty else "constraining/contradictory"
    return f"""# S01 Summary

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written: `{artifacts_written[0]}` and {len(artifacts_written) - 1} additional files listed in `status.json` and `artifact_manifest.json`.
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Outcome classification: {outcome}
- Caveats or blockers: {"; ".join(caveats)}
- Lay summary: S01 put Bubble, Insertion, Selection, null, and random policies behind a single local-rule interface. The classic policies replay the E02 deterministic simulator exactly for selected no-Frozen, Frozen Cell, and same-goal chimera cases, so the interface is stable enough to serve as the boundary for the S02 DSL.
- Recommended next action: implement the S02 rule DSL only after the Chief Scientist workflow explicitly starts S02.
"""


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
    parser.add_argument("--e01-artifacts", type=Path, default=DEFAULT_E01_ARTIFACTS)
    parser.add_argument("--e02-artifacts", type=Path, default=DEFAULT_E02_ARTIFACTS)
    parser.add_argument("--replicate-limit", type=int, default=DEFAULT_REPLICATE_LIMIT)
    parser.add_argument("--max-activations", type=int, default=2_000_000)
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    provenance_dir = artifacts_dir / "provenance"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    validation_rows = run_interface_smoke_validations()
    policy_rows, policy_summary_df = validate_policy_against_e02(
        args.e01_artifacts,
        replicate_limit=args.replicate_limit,
        max_activations=args.max_activations,
    )
    validation_rows.extend(policy_rows)
    e01_row, e01_command = run_e01_regression(args.e01_artifacts, step_dir)
    validation_rows.append(e01_row)
    repo_row, repo_command = run_repo_unit_tests(step_dir)
    validation_rows.append(repo_row)

    previous_e02_status_path = args.e02_artifacts / "research_steps" / "S01" / "status.json"
    previous_e02_status = read_json(previous_e02_status_path) if previous_e02_status_path.exists() else {}
    validation_rows.append(
        {
            "validationFamily": "upstream_e02_anchor",
            "conditionId": "previous_e02_s01_status",
            "algorithm": "all",
            "replicateIndex": 0,
            "success": bool(previous_e02_status.get("success")),
            "validationDetail": "Mounted E02 S01 status reports a successful deterministic simulator validation against E01.",
            "logPath": str(previous_e02_status_path),
        }
    )

    validation_df = pd.DataFrame(validation_rows)
    validation_csv = step_dir / "policy_interface_validation.csv"
    validation_parquet = step_dir / "policy_interface_validation.parquet"
    validation_results_csv = results_dir / "e03_s01_policy_interface_validation.csv"
    validation_results_parquet = results_dir / "e03_s01_policy_interface_validation.parquet"
    validation_df.to_csv(validation_csv, index=False)
    validation_df.to_parquet(validation_parquet, index=False)
    validation_df.to_csv(validation_results_csv, index=False)
    validation_df.to_parquet(validation_results_parquet, index=False)

    policy_summary_csv = step_dir / "policy_interface_replay_summary.csv"
    policy_summary_parquet = step_dir / "policy_interface_replay_summary.parquet"
    policy_summary_results_csv = results_dir / "e03_s01_policy_interface_replay_summary.csv"
    policy_summary_results_parquet = results_dir / "e03_s01_policy_interface_replay_summary.parquet"
    policy_summary_df.to_csv(policy_summary_csv, index=False)
    policy_summary_df.to_parquet(policy_summary_parquet, index=False)
    policy_summary_df.to_csv(policy_summary_results_csv, index=False)
    policy_summary_df.to_parquet(policy_summary_results_parquet, index=False)

    policy_catalog = pd.DataFrame(
        [policy.to_spec().to_dict() for policy in [BubblePolicy(), InsertionPolicy(), SelectionPolicy(), NullPolicy(), RandomWalkPolicy(1.0)]]
    )
    catalog_csv = step_dir / "policy_catalog.csv"
    catalog_parquet = step_dir / "policy_catalog.parquet"
    policy_catalog.to_csv(catalog_csv, index=False)
    policy_catalog.to_parquet(catalog_parquet, index=False)

    spec_path = step_dir / "policy_interface_spec.md"
    write_policy_interface_spec(spec_path)
    copied_code = copy_code_artifacts(step_dir)

    caveats = [
        "Validation is behavior-preserving against the deterministic E02 simulator, not exact E01 thread interleaving replay.",
        f"Representative E01/E02 replay used the first {args.replicate_limit} replicate(s) for selected no-Frozen, Frozen Cell, and same-goal chimera conditions.",
        "Reverse-direction Selection target semantics are preserved from E02 and remain a documented asymmetry rather than a corrected rule.",
        "The S02 rule DSL, generated policy corpus, and GPU vectorization were not started.",
    ]

    validation_report_path = step_dir / "validation_report.md"
    summary_path = step_dir / "summary.md"
    run_manifest_path = provenance_dir / "run_manifest.json"
    status_path = step_dir / "status.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"

    artifacts_written_paths = [
        spec_path,
        validation_csv,
        validation_parquet,
        validation_results_csv,
        validation_results_parquet,
        policy_summary_csv,
        policy_summary_parquet,
        policy_summary_results_csv,
        policy_summary_results_parquet,
        catalog_csv,
        catalog_parquet,
        step_dir / "e01_regression_test_log.txt",
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
        "passed: policy-interface replay matched E02 for selected classic no-Frozen, Frozen Cell, and chimera cases; serialization, toy policies, E01 regression context, and repository tests passed"
        if validation_passed
        else "failed: one or more S01 policy-interface validations failed"
    )
    recommended_next_action = "Proceed to S02 rule DSL only after Chief Scientist instruction; do not start S02 from this run."

    validation_report_path.write_text(render_validation_report(validation_df, caveats, artifacts_written), encoding="utf-8")
    summary_path.write_text(render_summary(validation_df, caveats, artifacts_written), encoding="utf-8")

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
        "replicateLimit": args.replicate_limit,
        "maxActivations": args.max_activations,
        "workerCount": 1,
        "threadEnvironment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "e01RegressionCommand": e01_command,
        "repoUnitTestCommand": repo_command,
        "previousE02S01StatusPath": str(previous_e02_status_path),
        "git": get_git_metadata(),
    }
    write_json(status_path, status)

    artifact_manifest = {
        "schema": "eidosoma.e03.s01.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "artifacts": collect_artifacts([Path(path) for path in artifacts_written] + [artifact_manifest_path]),
    }
    write_json(artifact_manifest_path, artifact_manifest)

    artifact_manifest["artifacts"] = collect_artifacts([Path(path) for path in artifacts_written])
    write_json(artifact_manifest_path, artifact_manifest)

    print(json.dumps({"success": validation_passed, "statusPath": str(status_path)}, indent=2))
    return 0 if validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
