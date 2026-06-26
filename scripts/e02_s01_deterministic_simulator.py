#!/usr/bin/env python3
"""Execute E02 S01 deterministic event simulator validation.

This step implements a single-event asynchronous simulator, validates it on
toy cases and selected E01 baseline seeds, writes S01 artifacts, and stops
before the S02 scheduler comparison.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from e02_deterministic_simulator import (  # noqa: E402
    DeterministicEventSimulator,
    initial_values_from_seed,
    monotonicity_error,
    sortedness_percent,
    sortedness_raw,
    state_hash,
)


EXPERIMENT_ID = "E02"
STEP_ID = "S01"
STEP_NUMBER = 1
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
DEFAULT_E01_ARTIFACTS = Path("/previous-artifacts/E01")


@dataclass
class CommandResult:
    args: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    ok: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return CommandResult(args, proc.returncode, proc.stdout, proc.stderr, proc.returncode == 0)
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return CommandResult(args, None, "", repr(exc), False)


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit.stdout.strip() if commit.ok else "unknown",
        "branch": branch.stdout.strip() if branch.ok else "unknown",
        "dirtyStatus": status.stdout.strip(),
        "remote": remote.stdout.strip(),
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


def e01_bundle_relocation_audit(e01_artifacts: Path) -> dict[str, Any]:
    """Validate E01 bundle-manifest paths after mounting at /previous-artifacts."""
    manifest_path = e01_artifacts / "report_bundle_inputs" / "bundle_manifest.json"
    if not manifest_path.exists():
        return {"success": False, "checkedCount": 0, "failures": [f"missing {manifest_path}"]}
    manifest = read_json(manifest_path)
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for section in ["figures", "tables", "reports"]:
        for item in manifest.get(section, []):
            for path_key, hash_key in [("path", "sha256"), ("pdfPath", "pdfSha256")]:
                raw_path = item.get(path_key)
                if not raw_path:
                    continue
                path = Path(raw_path)
                if path.is_absolute() and str(path).startswith("/artifacts/"):
                    relocated = e01_artifacts / path.relative_to("/artifacts")
                else:
                    relocated = path
                expected_hash = item.get(hash_key)
                exists = relocated.exists()
                hash_ok = True
                actual_hash = None
                if exists and expected_hash:
                    actual_hash = sha256_path(relocated)
                    hash_ok = actual_hash == expected_hash
                checks.append(
                    {
                        "section": section,
                        "manifestPath": raw_path,
                        "relocatedPath": str(relocated),
                        "exists": exists,
                        "expectedHash": expected_hash,
                        "actualHash": actual_hash,
                        "hashOk": hash_ok,
                    }
                )
                if not exists:
                    failures.append(f"missing relocated path {relocated}")
                elif not hash_ok:
                    failures.append(f"hash mismatch for {relocated}")
    return {
        "success": not failures,
        "checkedCount": len(checks),
        "failures": failures,
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            if abs(value) >= 1000 or (0 < abs(value) < 0.001):
                return f"{value:.3g}"
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
    import numpy as np

    rng = np.random.default_rng(int(seed))
    return sorted(int(value) for value in rng.choice(n, size=frozen_count, replace=False))


def normalize_condition_row(row: pd.Series) -> dict[str, Any]:
    return {str(key): row[key] for key in row.index}


def toy_validation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    metric_success = (
        sortedness_raw([1, 2, 3, 4]) == 3
        and sortedness_percent([1, 3, 2, 4]) == 100.0 * 2 / 3
        and monotonicity_error([1, 3, 2, 4]) == 1
    )
    rows.append(
        {
            "validation_family": "toy_metric",
            "condition_id": "toy_metric_contract",
            "algorithm": "metric_helpers",
            "replicate_index": 0,
            "success": metric_success,
            "validation_detail": "E01 Sortedness and monotonicity-error formulas match toy arrays.",
        }
    )

    passive = DeterministicEventSimulator(
        [2, 1],
        "bubble",
        frozen_positions=[1],
        frozen_variant="passive",
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="toy_passive_frozen",
    )
    passive_outcome = passive.step(forced_cell_id=0, forced_direction=1)
    rows.append(
        {
            "validation_family": "toy_frozen",
            "condition_id": "toy_passive_frozen",
            "algorithm": "bubble",
            "replicate_index": 0,
            "success": passive_outcome.swapped and passive.current_values() == [1, 2] and passive.current_frozen_positions() == [0],
            "e02_final_sortedness_percent": sortedness_percent(passive.current_values()),
            "validation_detail": "Passive Frozen Cell cannot initiate but can be moved by a non-frozen actor.",
        }
    )

    stuck = DeterministicEventSimulator(
        [2, 1],
        "bubble",
        frozen_positions=[1],
        frozen_variant="stuck",
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="toy_stuck_frozen",
    )
    stuck_outcome = stuck.step(forced_cell_id=0, forced_direction=1)
    rows.append(
        {
            "validation_family": "toy_frozen",
            "condition_id": "toy_stuck_frozen",
            "algorithm": "bubble",
            "replicate_index": 0,
            "success": (not stuck_outcome.swapped) and stuck.current_values() == [2, 1] and stuck.current_frozen_positions() == [1],
            "e02_final_sortedness_percent": sortedness_percent(stuck.current_values()),
            "validation_detail": "Stuck Frozen Cell blocks swaps involving the frozen target.",
        }
    )

    for algorithm in ["bubble", "insertion", "selection"]:
        result = DeterministicEventSimulator(
            [5, 1, 4, 2, 3],
            algorithm,
            scheduler_seed=10,
            tie_breaker_seed=20,
            condition_id=f"toy_sort_{algorithm}",
        ).run(max_activations=100_000)
        rows.append(
            {
                "validation_family": "toy_sort",
                "condition_id": f"toy_sort_{algorithm}",
                "algorithm": algorithm,
                "replicate_index": 0,
                "success": result.completed and result.final_values == [1, 2, 3, 4, 5],
                "e02_completed": result.completed,
                "e02_stop_reason": result.stop_reason,
                "e02_swap_count": result.swap_count,
                "e02_activation_count": result.activation_count,
                "e02_final_sortedness_percent": result.final_sortedness_percent,
                "validation_detail": "Small no-Frozen hand case reaches sorted order.",
            }
        )

    replay_kwargs = {
        "initial_values": [6, 2, 4, 1, 5, 3],
        "algotypes": "bubble",
        "scheduler_seed": 123,
        "tie_breaker_seed": 456,
        "condition_id": "toy_seed_replay",
    }
    first = DeterministicEventSimulator(**replay_kwargs).run(max_activations=100_000)
    second = DeterministicEventSimulator(**replay_kwargs).run(max_activations=100_000)
    rows.append(
        {
            "validation_family": "seed_replay",
            "condition_id": "toy_seed_replay",
            "algorithm": "bubble",
            "replicate_index": 0,
            "success": first.final_values == second.final_values
            and first.swap_count == second.swap_count
            and first.activation_count == second.activation_count
            and [row["state_hash"] for row in first.trace_rows] == [row["state_hash"] for row in second.trace_rows],
            "e02_completed": first.completed,
            "e02_swap_count": first.swap_count,
            "e02_activation_count": first.activation_count,
            "validation_detail": "Exact same seeds replay identical final state, counts, and trace hashes.",
        }
    )
    return rows


def run_e01_regression(e01_artifacts: Path, step_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    env = os.environ.copy()
    env["ARTIFACTS_DIR"] = str(e01_artifacts)
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
    result = run_command(cmd, cwd=REPO_ROOT, env=env)
    log_path = step_dir / "e01_regression_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + result.stdout + "\n\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    fail_lines = [
        line.strip()
        for line in result.stderr.splitlines()
        if line.startswith("test_") and (" ... FAIL" in line or " ... ERROR" in line)
    ]
    relocation_audit = e01_bundle_relocation_audit(e01_artifacts)
    relocation_only = (
        not result.ok
        and fail_lines == [
            "test_report_bundle_references_exist_and_hash (test_e01_baseline_regression.TestE01BaselineRegression.test_report_bundle_references_exist_and_hash) ... FAIL"
        ]
        and relocation_audit["success"]
    )
    accepted = result.ok or relocation_only
    detail = f"Packaged E01 regression tests return code {result.returncode}."
    if relocation_only:
        detail += " The only failure is the known absolute-path bundle-manifest relocation issue; relocated paths and hashes pass audit."
    validation_row = {
        "validation_family": "e01_regression_tests",
        "condition_id": "e01_packaged_regression_tests",
        "algorithm": "all",
        "replicate_index": 0,
        "success": accepted,
        "validation_detail": detail,
        "log_path": str(log_path),
    }
    command_payload = {
        "command": cmd,
        "returnCode": result.returncode,
        "rawSuccess": result.ok,
        "success": accepted,
        "logPath": str(log_path),
        "failLines": fail_lines,
        "relocationOnlyFailureAccepted": relocation_only,
        "relocationAudit": relocation_audit,
    }
    return validation_row, command_payload


def run_repo_unit_tests(step_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    result = run_command(cmd, cwd=REPO_ROOT)
    log_path = step_dir / "repo_unit_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + result.stdout + "\n\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    validation_row = {
        "validation_family": "repo_unit_tests",
        "condition_id": "e02_s01_repo_unit_tests",
        "algorithm": "all",
        "replicate_index": 0,
        "success": result.ok,
        "validation_detail": f"E02 S01 repository unit tests return code {result.returncode}.",
        "log_path": str(log_path),
    }
    command_payload = {
        "command": cmd,
        "returnCode": result.returncode,
        "success": result.ok,
        "logPath": str(log_path),
    }
    return validation_row, command_payload


def build_e01_condition_lookup(condition_matrix: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row.conditionId): row for row in condition_matrix.itertuples(index=False)}


def validate_against_e01(
    e01_artifacts: Path,
    replicate_limit: int,
    max_activations: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    config = read_json(e01_artifacts / "configs" / "e01_baseline_config.json")
    condition_matrix = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "condition_matrix.csv")
    seed_table = pd.read_csv(e01_artifacts / "research_steps" / "S03" / "seed_table.csv")
    e01_s04 = pd.read_parquet(e01_artifacts / "results" / "e01_s04_replicate_summary.parquet")
    e01_s07 = pd.read_parquet(e01_artifacts / "results" / "e01_frozen_cell_robustness.parquet")

    rows: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    stop_policies = config["semantics"]["stopPolicies"]

    selected_condition_ids: list[str] = []
    for algorithm in ["bubble", "insertion", "selection"]:
        selected_condition_ids.append(f"S04_cell_view_{algorithm}_unique_f0_none")
    for algorithm in ["bubble", "insertion", "selection"]:
        for frozen_variant in ["none", "passive", "stuck"]:
            frozen_counts = [0] if frozen_variant == "none" else [1]
            for frozen_count in frozen_counts:
                selected_condition_ids.append(f"S07_cell_view_{algorithm}_unique_f{frozen_count}_{frozen_variant}")

    for condition_id in selected_condition_ids:
        condition_match = condition_matrix[condition_matrix["conditionId"] == condition_id]
        if len(condition_match) != 1:
            rows.append(
                {
                    "validation_family": "e01_baseline_comparison",
                    "condition_id": condition_id,
                    "success": False,
                    "validation_detail": "Condition missing from E01 condition matrix.",
                }
            )
            continue
        condition = normalize_condition_row(condition_match.iloc[0])
        seeds = seed_table[
            (seed_table["conditionId"] == condition_id) & (seed_table["replicateIndex"] < replicate_limit)
        ].sort_values("replicateIndex")
        if seeds.empty:
            rows.append(
                {
                    "validation_family": "e01_baseline_comparison",
                    "condition_id": condition_id,
                    "success": False,
                    "validation_detail": "Seed rows missing from E01 seed table.",
                }
            )
            continue
        algorithm = str(condition["algorithms"])
        stop_policy = stop_policies[str(condition["stopPolicy"])]
        frozen_count = int(condition["frozenCount"])
        frozen_variant = str(condition["frozenVariant"])
        n = int(condition["n"])

        for seed_row_tuple in seeds.itertuples(index=False):
            seed_row = seed_row_tuple._asdict()
            replicate_index = int(seed_row["replicateIndex"])
            input_seed = int(seed_row["inputPermutationSeed"])
            scheduler_seed = int(seed_row["schedulerSeed"])
            tie_seed = int(seed_row["TieBreakerSeed"] if "TieBreakerSeed" in seed_row else seed_row["tieBreakerSeed"])
            frozen_seed_raw = seed_row.get("frozenPositionSeed")
            frozen_seed = None if frozen_count == 0 or pd.isna(frozen_seed_raw) else int(frozen_seed_raw)
            initial_values = initial_values_from_seed(input_seed, n=n, profile=str(condition["inputProfile"]))
            frozen_positions = frozen_positions_from_seed(frozen_seed, frozen_count, n)

            simulator = DeterministicEventSimulator(
                initial_values,
                algorithm,
                frozen_positions=frozen_positions,
                frozen_variant=frozen_variant,
                scheduler_seed=scheduler_seed,
                tie_breaker_seed=tie_seed,
                condition_id=condition_id,
                research_step_id=STEP_ID,
            )
            result = simulator.run(
                max_activations=max_activations,
                max_swaps=int(stop_policy["maxSwapEvents"]),
                max_comparisons=int(stop_policy["maxComparisonEvents"]),
                no_move_checks_required=int(stop_policy.get("noMoveChecksRequired", 2)),
            )
            summary_records.append(
                result.summary_record(
                    research_step_id=STEP_ID,
                    replicate_index=replicate_index,
                    replicate_number=int(seed_row["replicateNumber"]),
                    input_permutation_seed=input_seed,
                    scheduler_seed=scheduler_seed,
                    tie_breaker_seed=tie_seed,
                    frozen_position_seed=frozen_seed,
                    input_profile=str(condition["inputProfile"]),
                    frozen_variant=frozen_variant,
                    frozen_count=frozen_count,
                )
            )
            for trace_row in result.trace_rows:
                trace_row = dict(trace_row)
                trace_row["replicate_index"] = replicate_index
                trace_row["replicate_number"] = int(seed_row["replicateNumber"])
                trace_row["input_permutation_seed"] = input_seed
                trace_row["frozen_position_seed"] = frozen_seed
                trace_rows.append(trace_row)

            if condition_id.startswith("S04_"):
                e01_match = e01_s04[
                    (e01_s04["condition_id"] == condition_id) & (e01_s04["replicate_index"] == replicate_index)
                ]
                e01_prefix = "e01_"
                e01_completed = bool(e01_match.iloc[0]["completed"]) if len(e01_match) == 1 else None
                e01_sortedness = float(e01_match.iloc[0]["final_sortedness_percent"]) if len(e01_match) == 1 else None
                e01_error = int(e01_match.iloc[0]["final_monotonicity_error"]) if len(e01_match) == 1 else None
                e01_swap_count = int(e01_match.iloc[0]["swap_count"]) if len(e01_match) == 1 else None
                e01_initial_hash = str(e01_match.iloc[0]["initial_state_hash"]) if len(e01_match) == 1 else None
            else:
                e01_match = e01_s07[
                    (e01_s07["conditionId"] == condition_id) & (e01_s07["replicateIndex"] == replicate_index)
                ]
                e01_prefix = "e01_"
                e01_completed = bool(e01_match.iloc[0]["completed"]) if len(e01_match) == 1 else None
                e01_sortedness = float(e01_match.iloc[0]["finalSortednessPercent"]) if len(e01_match) == 1 else None
                e01_error = int(e01_match.iloc[0]["finalMonotonicityError"]) if len(e01_match) == 1 else None
                e01_swap_count = int(e01_match.iloc[0]["swapCount"]) if len(e01_match) == 1 else None
                e01_initial_hash = str(e01_match.iloc[0]["initialValuesHash"]) if len(e01_match) == 1 else None

            initial_hash_ok = e01_initial_hash == state_hash(initial_values)
            conservation_ok = (
                CounterSafe(result.initial_values) == CounterSafe(result.final_values)
                and CounterSafe(result.initial_algotypes) == CounterSafe(result.final_algotypes)
                and len(result.initial_frozen_positions) == frozen_count
                and len(result.final_frozen_positions) == frozen_count
            )
            no_frozen_success = frozen_count == 0 and result.completed and result.final_sortedness_percent == 100.0
            frozen_rule_success = frozen_count > 0 and conservation_ok and result.stop_reason in {
                "sorted",
                "no_cell_can_move_after_two_checks",
                "max_activation_cap",
                "max_step_cap",
                "max_comparison_cap",
            }
            success = bool(initial_hash_ok and conservation_ok and (no_frozen_success or frozen_rule_success))
            rows.append(
                {
                    "validation_family": "e01_baseline_comparison",
                    "condition_id": condition_id,
                    "algorithm": algorithm,
                    "replicate_index": replicate_index,
                    "success": success,
                    "initial_hash_matches_e01": initial_hash_ok,
                    "value_conservation": CounterSafe(result.initial_values) == CounterSafe(result.final_values),
                    "algotype_conservation": CounterSafe(result.initial_algotypes) == CounterSafe(result.final_algotypes),
                    "frozen_count_conservation": len(result.initial_frozen_positions) == len(result.final_frozen_positions) == frozen_count,
                    f"{e01_prefix}completed": e01_completed,
                    f"{e01_prefix}final_sortedness_percent": e01_sortedness,
                    f"{e01_prefix}final_monotonicity_error": e01_error,
                    f"{e01_prefix}swap_count": e01_swap_count,
                    "e02_completed": result.completed,
                    "e02_stop_reason": result.stop_reason,
                    "e02_final_sortedness_percent": result.final_sortedness_percent,
                    "e02_final_monotonicity_error": result.final_monotonicity_error,
                    "e02_swap_count": result.swap_count,
                    "e02_comparison_count": result.comparison_count,
                    "e02_archived_compare_and_swap_count": result.archived_compare_and_swap_count,
                    "e02_activation_count": result.activation_count,
                    "e02_blocked_move_attempts": result.blocked_move_attempts,
                    "swap_count_delta_e02_minus_e01": None
                    if e01_swap_count is None
                    else int(result.swap_count - e01_swap_count),
                    "validation_detail": "Counts may differ because S01 uses single-event random sequential activation; final metrics and invariants are the S01 acceptance target.",
                }
            )

    return rows, pd.DataFrame(summary_records), pd.DataFrame(trace_rows)


def CounterSafe(values: list[Any]) -> dict[Any, int]:
    from collections import Counter

    return dict(Counter(values))


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(artifacts_dir: Path, step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    package_dst = code_dir / "e02_deterministic_simulator"
    script_dst = code_dir / "scripts" / "e02_s01_deterministic_simulator.py"
    tests_dst = code_dir / "tests" / "test_e02_deterministic_simulator.py"
    canonical_code_dir = artifacts_dir / "code" / "e02_deterministic_simulator"
    if package_dst.exists():
        shutil.rmtree(package_dst)
    if canonical_code_dir.exists():
        shutil.rmtree(canonical_code_dir)
    shutil.copytree(REPO_ROOT / "e02_deterministic_simulator", package_dst, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO_ROOT / "e02_deterministic_simulator", canonical_code_dir, ignore=shutil.ignore_patterns("__pycache__"))
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "e02_s01_deterministic_simulator.py", script_dst)
    tests_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "tests" / "test_e02_deterministic_simulator.py", tests_dst)
    return [
        script_dst,
        tests_dst,
        *(path for path in package_dst.rglob("*.py")),
        *(path for path in canonical_code_dir.rglob("*.py")),
    ]


def write_trace_schema(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# S01 Trace Schema",
                "",
                "- `research_step_id`: producer step, always `S01` for this run.",
                "- `condition_id`: E01-compatible condition identifier used for validation.",
                "- `implementation`: `cell_view` for S01 simulator records.",
                "- `algorithm`: pure Algotype or `+`-joined mixed Algotype label.",
                "- `replicate_index`, `replicate_number`: E01 seed table replicate coordinates when applicable.",
                "- `event_index`: trace row index, with row 0 as `initial` and later rows as successful swaps.",
                "- `event_kind`: `initial` or `swap`.",
                "- `activation_index`: single-cell activation count at the time of the trace row.",
                "- `actor_cell_id`, `actor_algotype`, `target_position`: deterministic scheduler diagnostics for swap rows.",
                "- `swap_count`, `comparison_count`, `archived_compare_and_swap_count`: cumulative movement and comparison counters.",
                "- `sortedness_raw_count`, `sortedness_percent`, `monotonicity_error`: E01 metric fields.",
                "- `state_hash`, `initial_state_hash`: SHA-256 hashes using the E01 int16 value-array convention.",
                "- `frozen_positions_json`, `algotypes_json`: compact JSON state diagnostics.",
                "- `input_permutation_seed`, `frozen_position_seed`, `scheduler_seed`, `tie_breaker_seed`: explicit RNG provenance.",
                "",
                "The core E01 fields are preserved; S01 adds activation and actor fields because threaded interleavings are not replayable from E01.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_reports(
    *,
    artifacts_dir: Path,
    step_dir: Path,
    validation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    unit_payload: dict[str, Any],
    command_payload: dict[str, Any],
    artifact_paths: list[Path],
    started_at: str,
    replicate_limit: int,
    max_activations: int,
) -> tuple[Path, Path, Path, Path]:
    success = bool(validation_df["success"].fillna(False).all())
    validation_result = (
        "passed: toy cases, seed replay, E01 regression tests, and selected E01 baseline validations succeeded"
        if success
        else "failed: at least one S01 validation row did not pass"
    )
    caveats = [
        "Exact E01 thread interleavings are not replayable; S01 validates final metrics, conservation, and deterministic replay rather than exact event order.",
        "Comparison and activation counts are scheduler-dependent and should not be interpreted as exact E01 count reproduction.",
        f"Representative E01 baseline comparison used the first {replicate_limit} replicate(s) for selected S04/S07 cell-view conditions.",
    ]
    if command_payload.get("relocationOnlyFailureAccepted"):
        caveats.append(
            "The packaged E01 regression command returns nonzero only because its bundle manifest stores upstream absolute /artifacts paths; S01 verified the relocated /previous-artifacts/E01 paths and hashes."
        )
    recommended_next_action = "Proceed to S02 scheduler-regime comparison only after Chief Scientist instruction; use the S01 simulator as the reference engine."

    validation_counts = validation_df.groupby("validation_family")["success"].agg(["sum", "count"]).reset_index()
    validation_table = markdown_table(
        ["Validation family", "Passed", "Total"],
        [
            [row["validation_family"], int(row["sum"]), int(row["count"])]
            for row in validation_counts.to_dict("records")
        ],
    )
    comparison_preview = summary_df.head(12)
    preview_rows = []
    for row in comparison_preview.itertuples(index=False):
        preview_rows.append(
            [
                row.condition_id,
                row.replicate_index,
                row.algorithm,
                row.frozen_variant,
                row.frozen_count,
                row.completed,
                row.stop_reason,
                row.final_sortedness_percent,
                row.swap_count,
                row.activation_count,
            ]
        )
    preview_table = markdown_table(
        [
            "Condition",
            "Rep",
            "Alg",
            "Frozen",
            "f",
            "Done",
            "Stop",
            "Final Sortedness",
            "Swaps",
            "Activations",
        ],
        preview_rows,
    )

    artifacts_written = collect_artifacts(artifact_paths)
    artifacts_text = "\n".join(f"- `{item['path']}`" for item in artifacts_written)

    summary_path = step_dir / "summary.md"
    summary_path.write_text(
        f"""# E02 S01 Status Summary

- Research step ID: {STEP_ID}
- Step number: {STEP_NUMBER}
- Completion status: {'completed' if success else 'completed with validation failures'}
- Outcome classification: {OUTCOME_CLASSIFICATION if success else 'constraining'}
- Artifacts written:
{artifacts_text}
- Validation result: {validation_result}
- Caveats or blockers: {' '.join(caveats)}
- Lay summary: S01 replaced implicit Python thread scheduling with an explicit seeded event simulator that activates one cell at a time. The simulator preserves the local Bubble, Insertion, Selection, and Frozen Cell rules needed for E02 controls.
- Recommended next action: {recommended_next_action}

## Validation Counts

{validation_table}

## Representative S01 Runs

{preview_table}
""",
        encoding="utf-8",
    )

    validation_report_path = step_dir / "validation_report.md"
    validation_report_path.write_text(
        f"""# S01 Validation Report

- Research step ID: {STEP_ID}
- Completion status: {'completed' if success else 'completed with validation failures'}
- Artifacts written: `e01_vs_e02_validation.parquet`, `e02_s01_replicate_summary.parquet`, `e02_s01_trace_events.parquet`, code copies, manifests, and status files under `{step_dir}`.
- Validation result: {validation_result}
- Caveats or blockers: {' '.join(caveats)}
- Recommended next action: {recommended_next_action}

## What Was Validated

S01 validates a deterministic single-event simulator, not exact E01 thread interleavings. The acceptance checks are toy hand cases, value and Algotype conservation, Frozen Cell passive/stuck behavior, exact seed replay, packaged E01 regression tests, and representative n=100 E01 seed comparisons.

## Validation Counts

{validation_table}

## Notes

- Worker count: serial execution, one process; no GPU used.
- Max activations per S01 validation run: {max_activations}.
- E01 regression command accepted: {command_payload['success']} with raw return code {command_payload['returnCode']}.
- Repository unit tests: {unit_payload['success']} with return code {unit_payload['returnCode']}.
""",
        encoding="utf-8",
    )

    status_path = step_dir / "status.json"
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "artifactsWritten": [item["path"] for item in artifacts_written],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "outcomeClassification": OUTCOME_CLASSIFICATION if success else "constraining",
        "startedAt": started_at,
        "completedAt": utc_now(),
        "replicateLimit": replicate_limit,
        "maxActivations": max_activations,
        "e01RegressionCommand": command_payload,
        "repoUnitTestCommand": unit_payload,
    }
    write_json(status_path, status_payload)

    manifest_path = step_dir / "artifact_manifest.json"
    manifest_payload = {
        "schema": "eidosoma.e02.s01.artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAt": utc_now(),
        "git": get_git_metadata(),
        "artifacts": artifacts_written,
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
    }
    write_json(manifest_path, manifest_payload)
    return summary_path, validation_report_path, status_path, manifest_path


def write_run_manifest(
    artifacts_dir: Path,
    step_dir: Path,
    status_path: Path,
    artifact_paths: list[Path],
    started_at: str,
) -> Path:
    import numpy as np

    run_manifest_path = artifacts_dir / "provenance" / "run_manifest.json"
    payload = {
        "schema": "eidosoma.e02.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "latestResearchStepId": STEP_ID,
        "generatedAt": utc_now(),
        "startedAt": started_at,
        "statusPath": str(status_path),
        "git": get_git_metadata(),
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "gpuUsed": False,
        },
        "packageVersions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "artifacts": collect_artifacts(artifact_paths),
    }
    write_json(run_manifest_path, payload)
    return run_manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--e01-artifacts-dir", type=Path, default=DEFAULT_E01_ARTIFACTS)
    parser.add_argument("--replicate-limit", type=int, default=3)
    parser.add_argument("--max-activations", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    artifacts_dir: Path = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    traces_dir = artifacts_dir / "traces" / "e02" / STEP_ID
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    validation_rows = toy_validation_rows()
    unit_row, unit_payload = run_repo_unit_tests(step_dir)
    validation_rows.append(unit_row)
    e01_regression_row, command_payload = run_e01_regression(args.e01_artifacts_dir, step_dir)
    validation_rows.append(e01_regression_row)
    e01_rows, summary_df, trace_df = validate_against_e01(
        args.e01_artifacts_dir,
        replicate_limit=int(args.replicate_limit),
        max_activations=int(args.max_activations),
    )
    validation_rows.extend(e01_rows)
    validation_df = pd.DataFrame(validation_rows)

    validation_parquet = step_dir / "e01_vs_e02_validation.parquet"
    validation_csv = step_dir / "e01_vs_e02_validation.csv"
    summary_parquet = results_dir / "e02_s01_replicate_summary.parquet"
    summary_csv = results_dir / "e02_s01_replicate_summary.csv"
    trace_parquet = traces_dir / "e02_s01_trace_events.parquet"
    trace_csv_gz = traces_dir / "e02_s01_trace_events.csv.gz"
    validation_df.to_parquet(validation_parquet, index=False)
    validation_df.to_csv(validation_csv, index=False)
    summary_df.to_parquet(summary_parquet, index=False)
    summary_df.to_csv(summary_csv, index=False)
    trace_df.to_parquet(trace_parquet, index=False)
    trace_df.to_csv(trace_csv_gz, index=False, compression="gzip")

    trace_schema_path = step_dir / "trace_schema.md"
    write_trace_schema(trace_schema_path)
    code_paths = copy_code_artifacts(artifacts_dir, step_dir)
    artifact_paths = [
        validation_parquet,
        validation_csv,
        summary_parquet,
        summary_csv,
        trace_parquet,
        trace_csv_gz,
        trace_schema_path,
        step_dir / "repo_unit_test_log.txt",
        step_dir / "e01_regression_test_log.txt",
        *code_paths,
    ]
    summary_path, validation_report_path, status_path, manifest_path = write_reports(
        artifacts_dir=artifacts_dir,
        step_dir=step_dir,
        validation_df=validation_df,
        summary_df=summary_df,
        unit_payload=unit_payload,
        command_payload=command_payload,
        artifact_paths=artifact_paths,
        started_at=started_at,
        replicate_limit=int(args.replicate_limit),
        max_activations=int(args.max_activations),
    )
    artifact_paths.extend([summary_path, validation_report_path, status_path, manifest_path])
    run_manifest_path = write_run_manifest(artifacts_dir, step_dir, status_path, artifact_paths, started_at)

    # Refresh manifests after adding report/status paths and the run manifest itself.
    final_artifact_paths = artifact_paths + [run_manifest_path]
    manifest_payload = read_json(manifest_path)
    manifest_payload["artifacts"] = collect_artifacts(final_artifact_paths)
    write_json(manifest_path, manifest_payload)
    run_payload = read_json(run_manifest_path)
    run_payload["artifacts"] = collect_artifacts(final_artifact_paths)
    write_json(run_manifest_path, run_payload)

    success = bool(validation_df["success"].fillna(False).all())
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
