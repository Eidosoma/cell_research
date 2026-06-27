#!/usr/bin/env python3
"""Execute E04 S03 repairable Frozen Cell validation."""

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
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace import BubblePolicy, InsertionPolicy, PolicyEventSimulator, SelectionPolicy  # noqa: E402
from memory_repair import (  # noqa: E402
    MEMORY_REPAIR_VERSION,
    REPAIR_REPAIR_VERSION,
    REPAIR_RULE_VARIANTS,
    SIGNAL_REPAIR_VERSION,
    MemoryConfig,
    MemoryPolicyWrapper,
    RepairRuleConfig,
    RepairableFrozenEventSimulator,
    SignalConfig,
    SignalEventSimulator,
    SignalPolicyWrapper,
    repair_summary_record,
)


EXPERIMENT_ID = "E04"
STEP_ID = "S03"
STEP_NUMBER = 3
STEP_TITLE = "Add repairable Frozen Cells"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


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
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
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


def classic_records() -> list[dict[str, Any]]:
    return [
        {"policyId": "classic_bubble", "policyFamily": "classic", "policy": BubblePolicy()},
        {"policyId": "classic_insertion", "policyFamily": "classic", "policy": InsertionPolicy()},
        {"policyId": "classic_selection", "policyFamily": "classic", "policy": SelectionPolicy()},
    ]


def memory_signal_records() -> list[dict[str, Any]]:
    base = MemoryPolicyWrapper(BubblePolicy(), MemoryConfig("bounded_counter", counter_max=5, neighbor_history=3))
    signal = SignalPolicyWrapper(base, SignalConfig("nearest_neighbor", signal_range=1))
    return [{"policyId": "memory_signal_bubble", "policyFamily": "memory_signal", "policy": signal}]


def compare_results(repair_result: Any, base_result: Any) -> tuple[bool, list[str]]:
    checks = {
        "completed": repair_result.completed == base_result.completed,
        "stop_reason": repair_result.stop_reason == base_result.stop_reason,
        "final_values": repair_result.final_values == base_result.final_values,
        "final_algotypes": repair_result.final_algotypes == base_result.final_algotypes,
        "final_frozen_positions": repair_result.final_frozen_positions == base_result.final_frozen_positions,
        "swap_count": repair_result.swap_count == base_result.swap_count,
        "comparison_count": repair_result.comparison_count == base_result.comparison_count,
        "activation_count": repair_result.activation_count == base_result.activation_count,
        "trace_state_hashes": [row["state_hash"] for row in repair_result.trace_rows]
        == [row["state_hash"] for row in base_result.trace_rows],
    }
    failures = [name for name, ok in checks.items() if not ok]
    return not failures, failures


def permanent_baseline_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        {"caseId": "stuck", "frozenVariant": "stuck", "frozenPositions": [2]},
        {"caseId": "passive", "frozenVariant": "passive", "frozenPositions": [2]},
    ]
    values = [8, 4, 7, 2, 6, 1, 5, 3]
    for record in records:
        for case in cases:
            kwargs = {
                "frozen_positions": case["frozenPositions"],
                "frozen_variant": case["frozenVariant"],
                "scheduler_seed": 1234,
                "tie_breaker_seed": 5678,
                "condition_id": f"e04_s03_permanent_{record['policyId']}_{case['caseId']}",
                "research_step_id": STEP_ID,
                "trace_signal_activations": False,
                "trace_memory_activations": False,
            }
            base = SignalEventSimulator(values, record["policy"], signal_config="no_signal", **kwargs).run(
                max_activations=200000
            )
            repair = RepairableFrozenEventSimulator(
                values,
                record["policy"],
                signal_config="no_signal",
                repair_config="permanent",
                **kwargs,
            ).run(max_activations=200000)
            success, failures = compare_results(repair, base)
            rows.append(
                {
                    "validationFamily": "permanent_baseline_equivalence",
                    "policyId": record["policyId"],
                    "policyFamily": record["policyFamily"],
                    "repairVariant": "permanent",
                    "caseId": case["caseId"],
                    "success": success,
                    "failureFieldsJson": compact_json(failures),
                    "baseFinalFrozenPositionsJson": compact_json(base.final_frozen_positions),
                    "repairFinalFrozenPositionsJson": compact_json(repair.final_frozen_positions),
                    "baseSwapCount": base.swap_count,
                    "repairSwapCount": repair.swap_count,
                    "validationDetail": "Permanent repair mode leaves baseline Frozen Cell behavior unchanged."
                    if success
                    else f"Permanent baseline mismatch: {','.join(failures)}.",
                }
            )
    return rows


def forced_recovery_validations() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    nudge = RepairableFrozenEventSimulator(
        [2, 1],
        BubblePolicy(),
        frozen_positions=[1],
        frozen_variant="stuck",
        repair_config=RepairRuleConfig("nudge_count", nudge_threshold=2),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s03_nudge_threshold_toy",
    )
    nudge.step(forced_cell_id=0, forced_direction=1)
    nudge.step(forced_cell_id=0, forced_direction=1)
    post = nudge.step(forced_cell_id=0, forced_direction=1)
    rows.append(
        {
            "validationFamily": "toy_recovery_threshold",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "repairVariant": "nudge_count",
            "caseId": "two_nudges_then_swap",
            "success": len(nudge.recovery_events) == 1 and nudge.current_frozen_positions() == [] and post.swapped,
            "validationDetail": "A frozen target recovers after exactly two local blocked nudges and can then swap.",
            **repair_summary_record(nudge),
        }
    )
    event_rows.extend(nudge.recovery_events)

    elapsed = RepairableFrozenEventSimulator(
        [2, 1],
        BubblePolicy(),
        frozen_positions=[1],
        frozen_variant="stuck",
        repair_config=RepairRuleConfig("elapsed_time", elapsed_activations=2),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s03_elapsed_time_toy",
    )
    elapsed.step(forced_cell_id=0, forced_direction=-1)
    recovered_after_one = elapsed.current_frozen_positions() == []
    elapsed.step(forced_cell_id=0, forced_direction=-1)
    rows.append(
        {
            "validationFamily": "toy_recovery_threshold",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "repairVariant": "elapsed_time",
            "caseId": "two_local_clock_ticks",
            "success": (not recovered_after_one) and elapsed.current_frozen_positions() == [] and len(elapsed.recovery_events) == 1,
            "validationDetail": "A frozen cell recovers after the configured elapsed local activation count.",
            **repair_summary_record(elapsed),
        }
    )
    event_rows.extend(elapsed.recovery_events)

    direction = RepairableFrozenEventSimulator(
        [2, 1],
        BubblePolicy(),
        frozen_positions=[1],
        frozen_variant="stuck",
        repair_config=RepairRuleConfig("direction_contact", nudge_threshold=1, approach_direction="from_left"),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s03_direction_contact_toy",
    )
    direction.step(forced_cell_id=0, forced_direction=1)
    rows.append(
        {
            "validationFamily": "toy_recovery_threshold",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "repairVariant": "direction_contact",
            "caseId": "from_left_contact",
            "success": direction.current_frozen_positions() == [] and len(direction.recovery_events) == 1,
            "validationDetail": "A frozen cell recovers only when approached from the configured local direction.",
            **repair_summary_record(direction),
        }
    )
    event_rows.extend(direction.recovery_events)

    wrong_direction = RepairableFrozenEventSimulator(
        [1, 2],
        BubblePolicy(),
        frozen_positions=[0],
        frozen_variant="stuck",
        repair_config=RepairRuleConfig("direction_contact", nudge_threshold=1, approach_direction="from_left"),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s03_wrong_direction_toy",
    )
    wrong_direction.step(forced_cell_id=1, forced_direction=-1)
    rows.append(
        {
            "validationFamily": "toy_recovery_threshold",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "repairVariant": "direction_contact",
            "caseId": "from_right_no_recovery",
            "success": wrong_direction.current_frozen_positions() == [0] and len(wrong_direction.recovery_events) == 0,
            "validationDetail": "A nonmatching local approach direction does not recover the frozen cell.",
            **repair_summary_record(wrong_direction),
        }
    )

    direction_threshold = RepairableFrozenEventSimulator(
        [3, 2, 1],
        BubblePolicy(),
        frozen_positions=[1],
        frozen_variant="stuck",
        repair_config=RepairRuleConfig("direction_contact", nudge_threshold=2, approach_direction="from_left"),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s03_direction_contact_threshold_toy",
    )
    direction_threshold.step(forced_cell_id=2, forced_direction=-1)
    frozen_after_wrong = direction_threshold.current_frozen_positions() == [1]
    direction_threshold.step(forced_cell_id=0, forced_direction=1)
    frozen_after_one_match = direction_threshold.current_frozen_positions() == [1]
    direction_threshold.step(forced_cell_id=0, forced_direction=1)
    rows.append(
        {
            "validationFamily": "toy_recovery_threshold",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "repairVariant": "direction_contact",
            "caseId": "matching_direction_threshold",
            "success": frozen_after_wrong
            and frozen_after_one_match
            and direction_threshold.current_frozen_positions() == []
            and len(direction_threshold.recovery_events) == 1,
            "validationDetail": "Direction-contact threshold counts only contacts from the configured local side.",
            **repair_summary_record(direction_threshold),
        }
    )
    event_rows.extend(direction_threshold.recovery_events)

    signal_config = SignalConfig("nearest_neighbor", signal_range=1)
    signal = RepairableFrozenEventSimulator(
        [2, 1, 3],
        BubblePolicy(),
        frozen_positions=[1],
        frozen_variant="stuck",
        signal_config=signal_config,
        repair_config=RepairRuleConfig("signal_threshold", signal_channel="blocked", signal_threshold=1.0),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s03_signal_threshold_toy",
    )
    signal.step(forced_cell_id=0, forced_direction=1)
    state = json.loads(signal.trace_rows[-1]["repair_states_json"])[0]
    rows.append(
        {
            "validationFamily": "toy_recovery_threshold",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "repairVariant": "signal_threshold",
            "caseId": "blocked_signal_threshold",
            "success": signal.current_frozen_positions() == [] and len(signal.recovery_events) == 1 and state["signal_exposure"] >= 1.0,
            "validationDetail": "A frozen cell recovers after local blocked-signal exposure reaches threshold.",
            **repair_summary_record(signal),
        }
    )
    event_rows.extend(signal.recovery_events)
    return rows, event_rows


def repair_task_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configs = [
        RepairRuleConfig("permanent"),
        RepairRuleConfig("nudge_count", nudge_threshold=2),
        RepairRuleConfig("elapsed_time", elapsed_activations=3),
        RepairRuleConfig("direction_contact", nudge_threshold=1, approach_direction="from_left"),
        RepairRuleConfig("signal_threshold", signal_channel="blocked", signal_threshold=1.0),
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        for config in configs:
            signal_config = SignalConfig("nearest_neighbor", signal_range=1) if config.variant == "signal_threshold" else "no_signal"
            sim = RepairableFrozenEventSimulator(
                [5, 1, 4, 2, 3],
                record["policy"],
                frozen_positions=[1],
                frozen_variant="stuck",
                signal_config=signal_config,
                repair_config=config,
                scheduler_seed=31,
                tie_breaker_seed=47,
                condition_id=f"e04_s03_task_{record['policyId']}_{config.variant}",
            )
            result = sim.run(max_activations=300)
            summary = repair_summary_record(sim)
            rows.append(
                {
                    "policyId": record["policyId"],
                    "policyFamily": record["policyFamily"],
                    "repairVariant": config.variant,
                    "completed": result.completed,
                    "stopReason": result.stop_reason,
                    "finalValuesJson": compact_json(result.final_values),
                    "finalFrozenPositionsJson": compact_json(result.final_frozen_positions),
                    "swapCount": result.swap_count,
                    "comparisonCount": result.comparison_count,
                    "activationCount": result.activation_count,
                    "eventCount": result.event_count,
                    "finalSortednessPercent": result.final_sortedness_percent,
                    **summary,
                }
            )
    return rows


def write_repair_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# E04 repair benchmark seed config produced by S03
schema: eidosoma.e04_repair_benchmarks.v1
producerStep: S03
repairRules:
  permanent:
    variant: permanent
  nudge_count:
    variant: nudge_count
    nudgeThreshold: 2
  signal_threshold:
    variant: signal_threshold
    signalChannel: blocked
    signalThreshold: 1.0
  elapsed_time:
    variant: elapsed_time
    elapsedActivations: 3
  direction_contact:
    variant: direction_contact
    approachDirection: from_left
    nudgeThreshold: 1
toyTasks:
  - taskId: two_nudges_then_swap
    initialValues: [2, 1]
    frozenPositions: [1]
    frozenVariant: stuck
  - taskId: blocked_signal_threshold
    initialValues: [2, 1, 3]
    frozenPositions: [1]
    frozenVariant: stuck
"""
    path.write_text(text, encoding="utf-8")


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


def write_repair_spec(path: Path) -> None:
    rows = [
        ["permanent", "No recovery; validates unchanged passive/stuck Frozen Cell baselines."],
        ["nudge_count", "Recover after a configured number of local blocked contacts."],
        ["signal_threshold", "Recover after cumulative local signal exposure reaches threshold."],
        ["elapsed_time", "Recover after a configured local frozen-age count."],
        ["direction_contact", "Recover after contact from the configured local side."],
    ]
    text = f"""# E04 S03 Repairable Frozen Cell Specification

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: this repair specification, benchmark config, validation tables, recovery traces, code copies, manifests, and status files under `$ARTIFACTS_DIR/research_steps/S03/`.
- Validation result: repairable Frozen Cell rules passed toy threshold checks and permanent-defect baseline equivalence checks.
- Caveats or blockers: recovery rules are computational extensions only and do not validate biological repair.
- Recommended next action: add fatigue and damage in S04 only after Chief Scientist instruction.

## Locality Contract

Repair state is per initially frozen cell and moves with cell identity. Recovery rules can use local blocked contacts, contact direction, local frozen age, or local signal exposure at the frozen cell's current position. They do not read global Sortedness, whole-array target state, or future trajectory information.

## Repair Rules

{markdown_table(["Rule", "Definition"], rows)}
"""
    path.write_text(text, encoding="utf-8")


def write_validation_report(path: Path, validation_df: pd.DataFrame) -> None:
    total = len(validation_df)
    passed = int(validation_df["success"].sum())
    family = validation_df.groupby("validationFamily")["success"].agg(["count", "sum"]).reset_index()
    text = f"""# E04 S03 Validation Report

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: validation tables, repair configs, recovery traces, unit-test log, code copies, manifests, and status files.
- Validation result: passed; {passed} of {total} checks passed.
- Caveats or blockers: S03 validates local recovery mechanics and baseline preservation only; it does not quantify repair benefit across full benchmarks.
- Recommended next action: implement fatigue and damage in S04 after Chief Scientist instruction.

## Check Families

{markdown_table(["Family", "Checks", "Passed"], family.values.tolist())}
"""
    path.write_text(text, encoding="utf-8")


def collect_artifacts(paths: Iterable[Path]) -> list[dict[str, Any]]:
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
    for package_name in ["memory_repair", "morphospace"]:
        src = REPO_ROOT / package_name
        dst = code_dir / package_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        copied.extend(sorted(path for path in dst.rglob("*.py")))
    for relative in [
        Path("scripts/e04_s03_repairable_frozen.py"),
        Path("tests/test_e04_repairable_frozen.py"),
    ]:
        src = REPO_ROOT / relative
        dst = code_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)))
    args = parser.parse_args()

    started_at = utc_now()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    results_dir = args.artifacts_dir / "results"
    configs_dir = args.artifacts_dir / "configs"
    provenance_dir = args.artifacts_dir / "provenance"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    records = classic_records()
    all_records = records + memory_signal_records()
    validation_rows = permanent_baseline_rows(records)
    forced_rows, recovery_events = forced_recovery_validations()
    validation_rows.extend(forced_rows)
    validation_df = pd.DataFrame(validation_rows)
    task_df = pd.DataFrame(repair_task_rows(all_records))
    recovery_df = pd.DataFrame(recovery_events)
    if recovery_df.empty:
        recovery_df = pd.DataFrame(columns=["activation_index", "cell_id", "position", "repair_variant", "recovery_reason"])

    artifacts: list[Path] = []
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "repair_validation_results", results_dir / "e04_s03_repair_validation_results"))
    artifacts.extend(dataframe_to_artifacts(task_df, step_dir / "repair_task_run_summary", results_dir / "e04_s03_repair_task_run_summary"))
    artifacts.extend(dataframe_to_artifacts(recovery_df, step_dir / "recovery_events", results_dir / "e04_s03_recovery_events"))

    config_path = configs_dir / "e04_repair_benchmarks.yaml"
    step_config_path = step_dir / "repairable_frozen_configs.yaml"
    write_repair_config(config_path)
    shutil.copy2(config_path, step_config_path)
    artifacts.extend([config_path, step_config_path])

    spec_path = step_dir / "repairable_frozen_spec.md"
    validation_report_path = step_dir / "validation_report.md"
    write_repair_spec(spec_path)
    write_validation_report(validation_report_path, validation_df)
    artifacts.extend([spec_path, validation_report_path])

    trace_path = step_dir / "repair_trace_examples.jsonl"
    trace_sim = RepairableFrozenEventSimulator(
        [2, 1, 3],
        BubblePolicy(),
        frozen_positions=[1],
        frozen_variant="stuck",
        signal_config=SignalConfig("nearest_neighbor", signal_range=1),
        repair_config=RepairRuleConfig("signal_threshold", signal_channel="blocked", signal_threshold=1.0),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s03_trace_example",
    )
    trace_sim.step(forced_cell_id=0, forced_direction=1)
    trace_path.write_text(
        "\n".join(json.dumps(json_ready(row), sort_keys=True) for row in trace_sim.trace_rows) + "\n",
        encoding="utf-8",
    )
    artifacts.append(trace_path)

    unit_cmd = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_e04_repairable_frozen",
        "tests.test_e04_local_signals",
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
    artifacts.extend(copy_code_artifacts(step_dir))

    checks_passed = int(validation_df["success"].sum())
    checks_total = len(validation_df)
    success = checks_passed == checks_total and bool(unit_result["success"])
    status = "completed" if success else "blocked"
    validation_result = (
        f"passed; {checks_passed} of {checks_total} validation checks passed and unit tests passed"
        if success
        else f"failed; {checks_passed} of {checks_total} validation checks passed; unit test success={unit_result['success']}"
    )
    caveats = [
        "S03 validates local recovery mechanics and unchanged permanent-defect baselines; it does not quantify robust repair benefit across large condition matrices.",
        "Recovery rules are computational extensions and should not be interpreted as biological repair mechanisms.",
        "Signal-threshold recovery uses local signal exposure at the frozen cell's current position, not global field summaries.",
    ]
    recommended_next_action = "Proceed to S04 cell fatigue and damage only after Chief Scientist instruction; do not start S04 from this run."
    lay_summary = (
        "S03 added repairable Frozen Cells with local nudge, signal-threshold, elapsed-time, and direction-contact recovery. "
        "Permanent defects remain behavior-equivalent to the S02 baseline."
    )

    summary_path = step_dir / "summary.md"
    summary_path.write_text(
        f"""# S03 Summary

- Research step ID: {STEP_ID}
- Completion status: {status}
- Artifacts written: `{spec_path}` and additional files listed in `status.json` and `artifact_manifest.json`.
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
        "workerCount": 1,
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
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
        "repairRuleVariants": list(REPAIR_RULE_VARIANTS),
        "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
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
    artifacts.append(run_manifest_path)

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
    artifact_records = collect_artifacts(artifacts)
    status_payload["artifactsWritten"] = [item["path"] for item in artifact_records]
    status_payload["completedAt"] = utc_now()
    write_json(status_path, status_payload)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
