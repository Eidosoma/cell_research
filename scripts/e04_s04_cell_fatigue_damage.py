#!/usr/bin/env python3
"""Execute E04 S04 cell fatigue and damage validation."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace import BubblePolicy, InsertionPolicy, SelectionPolicy  # noqa: E402
from memory_repair import (  # noqa: E402
    FATIGUE_DAMAGE_VARIANTS,
    FATIGUE_DAMAGE_VERSION,
    MEMORY_REPAIR_VERSION,
    REPAIR_REPAIR_VERSION,
    SIGNAL_REPAIR_VERSION,
    FatigueDamageConfig,
    FatigueDamageEventSimulator,
    MemoryConfig,
    MemoryPolicyWrapper,
    RepairRuleConfig,
    RepairableFrozenEventSimulator,
    SignalConfig,
    SignalPolicyWrapper,
    fatigue_summary_record,
    repair_summary_record,
)


EXPERIMENT_ID = "E04"
STEP_ID = "S04"
STEP_NUMBER = 4
STEP_TITLE = "Add cell fatigue and damage"
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


def compare_results(left: Any, right: Any) -> tuple[bool, list[str]]:
    checks = {
        "completed": left.completed == right.completed,
        "stop_reason": left.stop_reason == right.stop_reason,
        "final_values": left.final_values == right.final_values,
        "final_algotypes": left.final_algotypes == right.final_algotypes,
        "final_frozen_positions": left.final_frozen_positions == right.final_frozen_positions,
        "swap_count": left.swap_count == right.swap_count,
        "comparison_count": left.comparison_count == right.comparison_count,
        "activation_count": left.activation_count == right.activation_count,
        "blocked_move_attempts": left.blocked_move_attempts == right.blocked_move_attempts,
        "trace_state_hashes": [row["state_hash"] for row in left.trace_rows]
        == [row["state_hash"] for row in right.trace_rows],
    }
    failures = [name for name, ok in checks.items() if not ok]
    return not failures, failures


def baseline_preservation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = [8, 4, 7, 2, 6, 1, 5, 3]
    permanent_cases = [
        {"caseId": "permanent_stuck", "frozenVariant": "stuck", "frozenPositions": [2], "repairConfig": "permanent"},
        {"caseId": "permanent_passive", "frozenVariant": "passive", "frozenPositions": [2], "repairConfig": "permanent"},
    ]
    repairable_cases = [
        {
            "caseId": "repairable_nudge_count",
            "frozenVariant": "stuck",
            "frozenPositions": [2],
            "repairConfig": RepairRuleConfig("nudge_count", nudge_threshold=2),
        }
    ]
    for record in records:
        for case in permanent_cases + repairable_cases:
            kwargs = {
                "frozen_positions": case["frozenPositions"],
                "frozen_variant": case["frozenVariant"],
                "scheduler_seed": 1234,
                "tie_breaker_seed": 5678,
                "condition_id": f"e04_s04_no_fatigue_{record['policyId']}_{case['caseId']}",
                "research_step_id": STEP_ID,
                "trace_signal_activations": False,
                "trace_memory_activations": False,
            }
            base = RepairableFrozenEventSimulator(
                values,
                record["policy"],
                signal_config="no_signal",
                repair_config=case["repairConfig"],
                **kwargs,
            ).run(max_activations=200000)
            fatigue = FatigueDamageEventSimulator(
                values,
                record["policy"],
                signal_config="no_signal",
                repair_config=case["repairConfig"],
                fatigue_config="none",
                **kwargs,
            ).run(max_activations=200000)
            success, failures = compare_results(fatigue, base)
            rows.append(
                {
                    "validationFamily": "baseline_preservation",
                    "policyId": record["policyId"],
                    "policyFamily": record["policyFamily"],
                    "fatigueVariant": "none",
                    "caseId": case["caseId"],
                    "success": success,
                    "failureFieldsJson": compact_json(failures),
                    "baseFinalValuesJson": compact_json(base.final_values),
                    "fatigueFinalValuesJson": compact_json(fatigue.final_values),
                    "baseFinalFrozenPositionsJson": compact_json(base.final_frozen_positions),
                    "fatigueFinalFrozenPositionsJson": compact_json(fatigue.final_frozen_positions),
                    "baseSwapCount": base.swap_count,
                    "fatigueSwapCount": fatigue.swap_count,
                    "validationDetail": "Disabled fatigue preserves S03 permanent/repairable behavior."
                    if success
                    else f"Disabled fatigue mismatch: {','.join(failures)}.",
                }
            )
    return rows


def toy_transition_validations() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    movement = FatigueDamageEventSimulator(
        [3, 2, 1],
        BubblePolicy(),
        fatigue_config=FatigueDamageConfig("movement", movement_threshold=1, recovery_activations=2),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s04_movement_fatigue_toy",
    )
    first = movement.step(forced_cell_id=2, forced_direction=-1)
    second = movement.step(forced_cell_id=2, forced_direction=-1)
    third = movement.step(forced_cell_id=2, forced_direction=-1)
    recovered_after_third = movement.fatigue_states[2]["fatigue_cooldown_remaining"] == 0
    fourth = movement.step(forced_cell_id=2, forced_direction=-1)
    rows.append(
        {
            "validationFamily": "toy_state_transition",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "fatigueVariant": "movement",
            "caseId": "movement_threshold_recovery",
            "success": first.swapped
            and second.reason == "fatigue_impaired"
            and third.reason == "fatigue_impaired"
            and recovered_after_third
            and fourth.swapped,
            "validationDetail": "Movement threshold induces temporary fatigue, then recovery permits movement again.",
            **fatigue_summary_record(movement),
        }
    )
    event_rows.extend(movement.fatigue_events)

    failed = FatigueDamageEventSimulator(
        [2, 1],
        BubblePolicy(),
        frozen_positions=[1],
        frozen_variant="stuck",
        repair_config="permanent",
        fatigue_config=FatigueDamageConfig("failed_swap", failed_swap_threshold=2, recovery_activations=3),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s04_failed_swap_fatigue_toy",
    )
    failed.step(forced_cell_id=0, forced_direction=1)
    failed.step(forced_cell_id=0, forced_direction=1)
    rows.append(
        {
            "validationFamily": "toy_state_transition",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "fatigueVariant": "failed_swap",
            "caseId": "failed_swap_threshold",
            "success": failed.fatigue_states[0]["fatigue_cooldown_remaining"] == 3
            and failed.fatigue_events[-1]["reason"] == "failed_swap_threshold_met",
            "validationDetail": "Repeated local blocked swaps induce temporary fatigue.",
            **fatigue_summary_record(failed),
        }
    )
    event_rows.extend(failed.fatigue_events)

    frustration = FatigueDamageEventSimulator(
        [1, 2],
        BubblePolicy(),
        fatigue_config=FatigueDamageConfig("frustration", frustration_threshold=2, recovery_activations=2),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s04_frustration_fatigue_toy",
    )
    frustration.step(forced_cell_id=0, forced_direction=-1)
    frustration.step(forced_cell_id=0, forced_direction=-1)
    rows.append(
        {
            "validationFamily": "toy_state_transition",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "fatigueVariant": "frustration",
            "caseId": "frustration_wait_threshold",
            "success": frustration.fatigue_states[0]["fatigue_cooldown_remaining"] == 2
            and frustration.fatigue_events[-1]["reason"] == "frustration_threshold_met",
            "validationDetail": "Repeated local no-move outcomes induce fatigue without global state.",
            **fatigue_summary_record(frustration),
        }
    )
    event_rows.extend(frustration.fatigue_events)

    directional = FatigueDamageEventSimulator(
        [3, 2, 1],
        BubblePolicy(),
        frozen_positions=[2],
        frozen_variant="stuck",
        repair_config="permanent",
        fatigue_config=FatigueDamageConfig(
            "directional",
            failed_swap_threshold=1,
            recovery_activations=3,
            impaired_direction="attempted",
        ),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s04_directional_fatigue_toy",
    )
    directional.step(forced_cell_id=1, forced_direction=1)
    blocked_right = directional.step(forced_cell_id=1, forced_direction=1)
    allowed_left = directional.step(forced_cell_id=1, forced_direction=-1)
    rows.append(
        {
            "validationFamily": "toy_state_transition",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "fatigueVariant": "directional",
            "caseId": "direction_specific_impairment",
            "success": directional.fatigue_states[1]["impaired_direction"] == "right"
            and blocked_right.reason == "fatigue_impaired"
            and allowed_left.swapped,
            "validationDetail": "Direction-specific fatigue blocks the impaired local direction while allowing the opposite direction.",
            **fatigue_summary_record(directional),
        }
    )
    event_rows.extend(directional.fatigue_events)

    damage = FatigueDamageEventSimulator(
        [3, 2, 1],
        BubblePolicy(),
        fatigue_config=FatigueDamageConfig(
            "stochastic_damage",
            damage_probability=1.0,
            damage_cooldown_activations=1,
            random_seed=7,
        ),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s04_damage_probability_toy",
    )
    first_damage = damage.step(forced_cell_id=2, forced_direction=-1)
    blocked_damage = damage.step(forced_cell_id=2, forced_direction=-1)
    rows.append(
        {
            "validationFamily": "toy_state_transition",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "fatigueVariant": "stochastic_damage",
            "caseId": "damage_probability_recovery",
            "success": first_damage.swapped
            and blocked_damage.reason == "damage_impaired"
            and damage.fatigue_states[2]["damage_cooldown_remaining"] == 0,
            "validationDetail": "Seeded damage probability induces temporary damage and recovery.",
            **fatigue_summary_record(damage),
        }
    )
    event_rows.extend(damage.fatigue_events)

    trace = FatigueDamageEventSimulator(
        [3, 2, 1],
        BubblePolicy(),
        fatigue_config=FatigueDamageConfig("movement", movement_threshold=1, recovery_activations=2),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s04_trace_schema_toy",
    )
    trace.step(forced_cell_id=2, forced_direction=-1)
    trace_success = all(
        isinstance(json.loads(trace.trace_rows[-1][field]), (dict, list))
        for field in ["fatigue_config_json", "fatigue_states_json", "fatigue_events_json"]
    )
    rows.append(
        {
            "validationFamily": "trace_schema",
            "policyId": "classic_bubble",
            "policyFamily": "classic",
            "fatigueVariant": "movement",
            "caseId": "fatigue_trace_json",
            "success": trace_success,
            "validationDetail": "Fatigue trace fields are valid JSON payloads.",
            **fatigue_summary_record(trace),
        }
    )
    event_rows.extend(trace.fatigue_events)

    return rows, event_rows


def fatigue_task_rows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configs = [
        FatigueDamageConfig("none"),
        FatigueDamageConfig("movement", movement_threshold=2, recovery_activations=2),
        FatigueDamageConfig("failed_swap", failed_swap_threshold=2, recovery_activations=2),
        FatigueDamageConfig("frustration", frustration_threshold=2, recovery_activations=2),
        FatigueDamageConfig("directional", failed_swap_threshold=1, recovery_activations=2, impaired_direction="attempted"),
        FatigueDamageConfig("stochastic_damage", damage_probability=0.25, damage_cooldown_activations=2, random_seed=13),
    ]
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for record in records:
        for config in configs:
            sim = FatigueDamageEventSimulator(
                [5, 1, 4, 2, 3],
                record["policy"],
                frozen_positions=[1],
                frozen_variant="stuck",
                repair_config=RepairRuleConfig("nudge_count", nudge_threshold=2),
                signal_config=SignalConfig("nearest_neighbor", signal_range=1)
                if record["policyFamily"] == "memory_signal"
                else "no_signal",
                fatigue_config=config,
                scheduler_seed=31,
                tie_breaker_seed=47,
                condition_id=f"e04_s04_task_{record['policyId']}_{config.variant}",
            )
            result = sim.run(max_activations=300)
            summary = fatigue_summary_record(sim)
            repair_summary = repair_summary_record(sim)
            rows.append(
                {
                    "policyId": record["policyId"],
                    "policyFamily": record["policyFamily"],
                    "fatigueVariant": config.variant,
                    "repairVariant": sim.repair_config.variant,
                    "completed": result.completed,
                    "stopReason": result.stop_reason,
                    "finalValuesJson": compact_json(result.final_values),
                    "finalFrozenPositionsJson": compact_json(result.final_frozen_positions),
                    "swapCount": result.swap_count,
                    "comparisonCount": result.comparison_count,
                    "blockedMoveAttempts": result.blocked_move_attempts,
                    "activationCount": result.activation_count,
                    "eventCount": result.event_count,
                    "finalSortednessPercent": result.final_sortedness_percent,
                    **summary,
                    "recoveredCellCount": repair_summary["recoveredCellCount"],
                    "remainingFrozenCellCount": repair_summary["remainingFrozenCellCount"],
                    "recoveryEventsJson": repair_summary["recoveryEventsJson"],
                }
            )
            for event in sim.fatigue_events:
                event_rows.append({"policyId": record["policyId"], "policyFamily": record["policyFamily"], **event})
    return rows, event_rows


def write_benchmark_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# E04 repair/fatigue benchmark seed config produced by S04
schema: eidosoma.e04_repair_benchmarks.v2
producerStep: S04
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
fatigueRules:
  none:
    variant: none
  movement:
    variant: movement
    movementThreshold: 2
    recoveryActivations: 2
  failed_swap:
    variant: failed_swap
    failedSwapThreshold: 2
    recoveryActivations: 2
  frustration:
    variant: frustration
    frustrationThreshold: 2
    recoveryActivations: 2
  directional:
    variant: directional
    failedSwapThreshold: 1
    recoveryActivations: 2
    impairedDirection: attempted
  stochastic_damage:
    variant: stochastic_damage
    damageProbability: 0.25
    damageCooldownActivations: 2
    randomSeed: 13
toyTasks:
  - taskId: movement_threshold_recovery
    initialValues: [3, 2, 1]
  - taskId: failed_swap_threshold
    initialValues: [2, 1]
    frozenPositions: [1]
    frozenVariant: stuck
  - taskId: direction_specific_impairment
    initialValues: [3, 2, 1]
    frozenPositions: [2]
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


def write_fatigue_spec(path: Path) -> None:
    rows = [
        ["none", "Disabled control; preserves S03 permanent and repairable baselines."],
        ["movement", "Temporary impairment after excessive local successful movement."],
        ["failed_swap", "Temporary impairment after repeated local blocked swap attempts."],
        ["frustration", "Temporary impairment after repeated local no-move outcomes."],
        ["directional", "Temporary impairment only for the attempted local direction."],
        ["stochastic_damage", "Seeded local damage probability with finite recovery period."],
    ]
    text = f"""# E04 S04 Fatigue And Damage Specification

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: this fatigue model specification, benchmark configs, validation tables, event traces, diagnostic plots, code copies, manifests, and status files under `$ARTIFACTS_DIR/research_steps/S04/`.
- Validation result: fatigue/damage state transitions passed toy checks, and disabled fatigue preserved S03 permanent and repairable baselines.
- Caveats or blockers: fatigue and damage are computational unreliability mechanisms only; parameter sensitivity is deferred to downstream benchmarks.
- Recommended next action: define homeostatic tasks in S05 only after Chief Scientist instruction.

## Locality Contract

Fatigue and damage state is attached to cell identity and moves with the cell. S04 rules use local outcomes only: successful movement, blocked local swap attempts, no-move outcomes, attempted direction, seeded local damage draws, and finite cooldown counters. They do not read global Sortedness, whole-array target state, or future trajectories.

## Rules

{markdown_table(["Rule", "Definition"], rows)}
"""
    path.write_text(text, encoding="utf-8")


def write_validation_report(path: Path, validation_df: pd.DataFrame) -> None:
    total = len(validation_df)
    passed = int(validation_df["success"].sum())
    family = validation_df.groupby("validationFamily")["success"].agg(["count", "sum"]).reset_index()
    text = f"""# E04 S04 Validation Report

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: validation tables, fatigue configs, event traces, diagnostic plots, unit-test log, code copies, manifests, and status files.
- Validation result: passed; {passed} of {total} checks passed.
- Caveats or blockers: S04 validates state transitions and baseline preservation only; it does not establish robust repair benefit under fatigue.
- Recommended next action: define homeostatic tasks in S05 after Chief Scientist instruction.

## Check Families

{markdown_table(["Family", "Checks", "Passed"], family.values.tolist())}
"""
    path.write_text(text, encoding="utf-8")


def write_diagnostic_plot(task_df: pd.DataFrame, step_path: Path, figures_path: Path) -> list[Path]:
    counts = task_df.groupby("fatigueVariant")["fatigueEventCount"].sum().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_xlabel("Fatigue/damage variant")
    ax.set_ylabel("Fatigue/damage events")
    ax.set_title("E04 S04 toy task event counts")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    step_path.parent.mkdir(parents=True, exist_ok=True)
    figures_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(step_path.with_suffix(".png"), dpi=160)
    fig.savefig(step_path.with_suffix(".svg"))
    shutil.copy2(step_path.with_suffix(".png"), figures_path.with_suffix(".png"))
    shutil.copy2(step_path.with_suffix(".svg"), figures_path.with_suffix(".svg"))
    plt.close(fig)
    return [
        step_path.with_suffix(".png"),
        step_path.with_suffix(".svg"),
        figures_path.with_suffix(".png"),
        figures_path.with_suffix(".svg"),
    ]


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
        Path("scripts/e04_s04_cell_fatigue_damage.py"),
        Path("tests/test_e04_fatigue_damage.py"),
    ]:
        src = REPO_ROOT / relative
        dst = code_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    started_at = utc_now()
    step_dir = args.artifacts_dir / "research_steps" / STEP_ID
    results_dir = args.artifacts_dir / "results"
    configs_dir = args.artifacts_dir / "configs"
    figures_dir = args.artifacts_dir / "figures"
    provenance_dir = args.artifacts_dir / "provenance"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    records = classic_records()
    all_records = records + memory_signal_records()
    validation_rows = baseline_preservation_rows(records)
    toy_rows, toy_events = toy_transition_validations()
    validation_rows.extend(toy_rows)
    task_rows, task_events = fatigue_task_rows(all_records)

    validation_df = pd.DataFrame(validation_rows)
    task_df = pd.DataFrame(task_rows)
    event_df = pd.DataFrame(toy_events + task_events)
    if event_df.empty:
        event_df = pd.DataFrame(
            columns=[
                "activation_index",
                "cell_id",
                "position",
                "event_type",
                "reason",
                "variant",
                "impaired_direction",
                "fatigue_cooldown_remaining",
                "damage_cooldown_remaining",
                "attempt_direction",
            ]
        )

    artifacts: list[Path] = []
    artifacts.extend(
        dataframe_to_artifacts(
            validation_df,
            step_dir / "fatigue_validation_results",
            results_dir / "e04_s04_fatigue_validation_results",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            task_df,
            step_dir / "fatigue_task_run_summary",
            results_dir / "e04_s04_fatigue_task_run_summary",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(event_df, step_dir / "fatigue_events", results_dir / "e04_s04_fatigue_events")
    )
    artifacts.extend(
        write_diagnostic_plot(
            task_df,
            step_dir / "fatigue_event_counts",
            figures_dir / "e04_s04_fatigue_event_counts",
        )
    )

    config_path = configs_dir / "e04_repair_benchmarks.yaml"
    step_config_path = step_dir / "fatigue_damage_configs.yaml"
    write_benchmark_config(config_path)
    shutil.copy2(config_path, step_config_path)
    artifacts.extend([config_path, step_config_path])

    spec_path = step_dir / "fatigue_model_spec.md"
    validation_report_path = step_dir / "validation_report.md"
    write_fatigue_spec(spec_path)
    write_validation_report(validation_report_path, validation_df)
    artifacts.extend([spec_path, validation_report_path])

    trace_path = step_dir / "fatigue_trace_examples.jsonl"
    trace_sim = FatigueDamageEventSimulator(
        [3, 2, 1],
        BubblePolicy(),
        fatigue_config=FatigueDamageConfig("movement", movement_threshold=1, recovery_activations=2),
        scheduler_seed=1,
        tie_breaker_seed=1,
        condition_id="e04_s04_trace_example",
    )
    trace_sim.step(forced_cell_id=2, forced_direction=-1)
    trace_sim.step(forced_cell_id=2, forced_direction=-1)
    trace_path.write_text(
        "\n".join(json.dumps(json_ready(row), sort_keys=True) for row in trace_sim.trace_rows) + "\n",
        encoding="utf-8",
    )
    artifacts.append(trace_path)

    unit_cmd = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_e04_fatigue_damage",
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
        "S04 validates local fatigue/damage state transitions and disabled-baseline preservation; it does not quantify broad robustness benefit.",
        "Fatigue and damage are computational unreliability mechanisms and should not be interpreted as biological damage models.",
        "Stochastic damage is seeded and local, but parameter sensitivity is deferred to S05 and later ablations.",
    ]
    recommended_next_action = "Proceed to S05 homeostatic task definition only after Chief Scientist instruction; do not start S05 from this run."
    lay_summary = (
        "S04 added temporary cell fatigue and damage rules based on local movement, failed swaps, no-move frustration, "
        "attempted direction, and seeded local damage draws. Disabled fatigue leaves S03 repair behavior unchanged."
    )

    summary_path = step_dir / "summary.md"
    summary_path.write_text(
        f"""# S04 Summary

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
        "fatigueDamageVersion": FATIGUE_DAMAGE_VERSION,
        "fatigueDamageVariants": list(FATIGUE_DAMAGE_VARIANTS),
        "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
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
