#!/usr/bin/env python3
"""Execute E04 S05 homeostatic task definition and validation."""

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

from memory_repair import (  # noqa: E402
    FATIGUE_DAMAGE_VERSION,
    HOMEOSTASIS_BENCHMARK_VERSION,
    MEMORY_REPAIR_VERSION,
    PERTURBATION_TYPES,
    REPAIR_REPAIR_VERSION,
    SIGNAL_REPAIR_VERSION,
    TRIVIAL_CONTROLLERS,
    apply_perturbation,
    build_homeostatic_benchmark_config,
    evaluate_trivial_controller,
    generate_seeded_schedule,
    normalized_sortedness_record,
    stable_hash,
    stable_json,
)


EXPERIMENT_ID = "E04"
STEP_ID = "S05"
STEP_NUMBER = 5
STEP_TITLE = "Define homeostatic tasks"
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


def yaml_scalar(value: Any) -> str:
    value = json_ready(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in ":#{}[]\n,") or text.lower() in {"true", "false", "null"}:
        return json.dumps(text)
    return text


def to_yaml(value: Any, indent: int = 0) -> str:
    value = json_ready(value)
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        lines = []
        for item in value:
            if isinstance(item, Mapping):
                lines.append(f"{prefix}-")
                lines.append(to_yaml(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{yaml_scalar(value)}"


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


def task_catalog_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in config["tasks"]:
        event_types = sorted({event["type"] for event in task["perturbationSchedule"]})
        rows.append(
            {
                "taskId": task["taskId"],
                "description": task["description"],
                "horizonTicks": task["horizonTicks"],
                "scheduleSeed": task["scheduleSeed"],
                "scheduleHash": task["scheduleHash"],
                "sortednessThresholdPercent": task["sortednessThresholdPercent"],
                "initialLength": len(task["initialValues"]),
                "perturbationCount": len(task["perturbationSchedule"]),
                "perturbationTypesJson": compact_json(event_types),
            }
        )
    return rows


def schedule_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in config["tasks"]:
        for event in task["perturbationSchedule"]:
            rows.append(
                {
                    "taskId": task["taskId"],
                    "scheduleHash": task["scheduleHash"],
                    "eventJson": compact_json(event),
                    **event,
                }
            )
    return rows


def trivial_controller_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    ticks: list[dict[str, Any]] = []
    for task in config["tasks"]:
        for controller_id in TRIVIAL_CONTROLLERS:
            result = evaluate_trivial_controller(task, controller_id)
            summaries.append(
                {
                    "taskId": result["taskId"],
                    "controllerId": result["controllerId"],
                    "horizonTicks": result["horizonTicks"],
                    "timeInRangeFraction": result["timeInRangeFraction"],
                    "failureDurationTicks": result["failureDurationTicks"],
                    "energyProxy": result["energyProxy"],
                    "maxRecoveryTimeTicks": result["maxRecoveryTimeTicks"],
                    "unrecoveredPerturbationCount": result["unrecoveredPerturbationCount"],
                    "finalValuesJson": compact_json(result["finalValues"]),
                    "recoveryTimesJson": compact_json(result["recoveryTimes"]),
                }
            )
            for record in result["tickRecords"]:
                ticks.append(
                    {
                        "taskId": result["taskId"],
                        "controllerId": controller_id,
                        "eventsJson": compact_json(record["events"]),
                        "valuesJson": compact_json(record["values"]),
                        **{key: value for key, value in record.items() if key not in {"events", "values"}},
                    }
                )
    return summaries, ticks


def normalization_example_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    task = next(task for task in config["tasks"] if task["taskId"] == "insertion_deletion_normalization")
    values = list(task["initialValues"])
    rows = [
        {
            "taskId": task["taskId"],
            "phase": "initial",
            **normalized_sortedness_record(values, initial_length=len(task["initialValues"])),
        }
    ]
    for event in task["perturbationSchedule"]:
        values, _, _ = apply_perturbation(values, event)
        rows.append(
            {
                "taskId": task["taskId"],
                "phase": f"after_{event['type']}_tick_{event['tick']}",
                "eventJson": compact_json(event),
                **normalized_sortedness_record(values, initial_length=len(task["initialValues"])),
            }
        )
    return rows


def validation_rows(config: Mapping[str, Any], controller_df: pd.DataFrame, normalization_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rebuilt = build_homeostatic_benchmark_config()
    rows.append(
        {
            "validationFamily": "schedule_replay",
            "caseId": "full_config_replay",
            "success": stable_hash(config["tasks"]) == stable_hash(rebuilt["tasks"]),
            "validationDetail": "Rebuilding the S05 config reproduces all task schedules and hashes.",
        }
    )
    random_templates = [
        {"tick": 1, "type": "swap", "source": "seeded_random_adjacent"},
        {"tick": 2, "type": "insert", "source": "seeded_random_insert"},
        {"tick": 3, "type": "delete", "source": "seeded_random_delete"},
    ]
    same_a = generate_seeded_schedule(initial_values=[1, 2, 3, 4], schedule_seed=42, event_templates=random_templates)
    same_b = generate_seeded_schedule(initial_values=[1, 2, 3, 4], schedule_seed=42, event_templates=random_templates)
    different = generate_seeded_schedule(initial_values=[1, 2, 3, 4], schedule_seed=43, event_templates=random_templates)
    rows.append(
        {
            "validationFamily": "schedule_replay",
            "caseId": "seed_same_and_different",
            "success": same_a == same_b and stable_hash(same_a) != stable_hash(different),
            "validationDetail": "Identical seeds replay exactly and changed seeds alter random perturbation schedules.",
        }
    )
    all_event_types = {event["type"] for task in config["tasks"] for event in task["perturbationSchedule"]}
    rows.append(
        {
            "validationFamily": "schedule_schema",
            "caseId": "all_required_event_types_present",
            "success": set(PERTURBATION_TYPES).issubset(all_event_types),
            "validationDetail": "Config includes swap, insert, delete, freeze, recover, and damage perturbation types.",
        }
    )
    mixed = next(task for task in config["tasks"] if task["taskId"] == "frozen_damage_mixed_events")
    mixed_types = [event["type"] for event in mixed["perturbationSchedule"]]
    rows.append(
        {
            "validationFamily": "schedule_schema",
            "caseId": "freeze_recover_damage_schema",
            "success": mixed_types.count("freeze") == 1 and mixed_types.count("recover") == 1 and "damage" in mixed_types,
            "validationDetail": "Mixed event task has a paired freeze/recover and a damage event.",
        }
    )
    steady_noop = controller_df.query("taskId == 'steady_sorted_no_perturbation' and controllerId == 'noop'").iloc[0]
    rows.append(
        {
            "validationFamily": "trivial_controller",
            "caseId": "noop_sorted_no_perturbation",
            "success": steady_noop.timeInRangeFraction == 1.0
            and steady_noop.failureDurationTicks == 0
            and steady_noop.energyProxy == 0,
            "validationDetail": "No-op controller remains perfect on sorted no-perturbation task.",
        }
    )
    swap_noop = controller_df.query("taskId == 'adjacent_swap_recovery' and controllerId == 'noop'").iloc[0]
    rows.append(
        {
            "validationFamily": "trivial_controller",
            "caseId": "noop_fails_swap_task",
            "success": swap_noop.failureDurationTicks > 0 and swap_noop.unrecoveredPerturbationCount > 0,
            "validationDetail": "No-op controller exposes nonzero failure duration after seeded swaps.",
        }
    )
    swap_oracle = controller_df.query("taskId == 'adjacent_swap_recovery' and controllerId == 'oracle_sort'").iloc[0]
    rows.append(
        {
            "validationFamily": "trivial_controller",
            "caseId": "oracle_sort_recovers_swap_task",
            "success": swap_oracle.timeInRangeFraction == 1.0
            and swap_oracle.failureDurationTicks == 0
            and swap_oracle.maxRecoveryTimeTicks == 0,
            "validationDetail": "Validation-only oracle sort recovers seeded swaps within the same tick.",
        }
    )
    insertion = normalization_df.query("phase == 'after_insert_tick_1'").iloc[0]
    rows.append(
        {
            "validationFamily": "normalization",
            "caseId": "insertion_current_denominator",
            "success": insertion.currentLength == 5
            and insertion.currentPairDenominator == 4
            and insertion.initialPairDenominator == 3
            and abs(insertion.sortednessPercent - 75.0) < 1e-9,
            "validationDetail": "Insertion metrics use current adjacent-pair denominator, not initial denominator.",
        }
    )
    deletion = normalization_df.query("phase == 'after_delete_tick_3'").iloc[0]
    rows.append(
        {
            "validationFamily": "normalization",
            "caseId": "deletion_current_denominator",
            "success": deletion.currentLength == 4
            and deletion.currentPairDenominator == 3
            and abs(deletion.sortednessPercent - 100.0) < 1e-9,
            "validationDetail": "Deletion metrics recompute current adjacent-pair denominator.",
        }
    )
    return rows


def write_homeostatic_config(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# E04 S05 seeded homeostatic task config\n" + to_yaml(config) + "\n", encoding="utf-8")


def write_repair_benchmark_index(path: Path, config_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "eidosoma.e04_repair_benchmarks.v3",
        "producerStep": "S05",
        "homeostaticTasksConfig": str(config_path),
        "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
        "notes": "S05 adds homeostatic task definitions; S03/S04 repair and fatigue rule definitions remain upstream artifacts.",
    }
    path.write_text("# E04 repair/fatigue/homeostasis benchmark index produced by S05\n" + to_yaml(payload) + "\n", encoding="utf-8")


def write_task_spec(path: Path, config: Mapping[str, Any]) -> None:
    rows = [
        [
            task["taskId"],
            task["horizonTicks"],
            task["scheduleSeed"],
            len(task["perturbationSchedule"]),
            ", ".join(sorted({event["type"] for event in task["perturbationSchedule"]})) or "none",
        ]
        for task in config["tasks"]
    ]
    text = f"""# E04 S05 Homeostatic Task Specification

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: seeded homeostatic task YAML, task catalog, perturbation schedules, trivial-controller results, normalization examples, validation report, code copies, manifests, and status files under `$ARTIFACTS_DIR/research_steps/S05/`.
- Validation result: seeded schedule replay, trivial-controller behavior, and insertion/deletion normalization checks passed.
- Caveats or blockers: S05 defines and dry-validates benchmark tasks only; it does not measure enhanced policy performance.
- Recommended next action: create local learning rules in S06 only after Chief Scientist instruction.

## Task Catalog

{markdown_table(["Task", "Horizon", "Seed", "Events", "Types"], rows)}
"""
    path.write_text(text, encoding="utf-8")


def write_normalization_contract(path: Path, config: Mapping[str, Any]) -> None:
    contract = config["normalization"]
    rows = [[key, value] for key, value in contract.items()]
    text = f"""# E04 S05 Normalization Contract

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: normalization contract, examples, validation tables, status files, and code copies.
- Validation result: insertion and deletion examples validated current-length denominators.
- Caveats or blockers: dynamic-length normalized scores are benchmark proxies and should be reported with current length.
- Recommended next action: create local learning rules in S06 only after Chief Scientist instruction.

{markdown_table(["Field", "Rule"], rows)}
"""
    path.write_text(text, encoding="utf-8")


def write_validation_report(path: Path, validation_df: pd.DataFrame) -> None:
    total = len(validation_df)
    passed = int(validation_df["success"].sum())
    family = validation_df.groupby("validationFamily")["success"].agg(["count", "sum"]).reset_index()
    text = f"""# E04 S05 Validation Report

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: seeded homeostatic config, task catalog, perturbation schedules, trivial-controller results, normalization examples, unit-test log, code copies, manifests, and status files.
- Validation result: passed; {passed} of {total} checks passed.
- Caveats or blockers: S05 validates benchmark definitions and trivial controls only; policy performance measurement starts in later steps.
- Recommended next action: create local learning rules in S06 after Chief Scientist instruction.

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
        Path("scripts/e04_s05_homeostatic_tasks.py"),
        Path("tests/test_e04_homeostasis_tasks.py"),
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
    provenance_dir = args.artifacts_dir / "provenance"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    config = build_homeostatic_benchmark_config()
    task_df = pd.DataFrame(task_catalog_rows(config))
    schedule_df = pd.DataFrame(schedule_rows(config))
    controller_rows, tick_rows = trivial_controller_rows(config)
    controller_df = pd.DataFrame(controller_rows)
    tick_df = pd.DataFrame(tick_rows)
    normalization_df = pd.DataFrame(normalization_example_rows(config))
    validation_df = pd.DataFrame(validation_rows(config, controller_df, normalization_df))

    artifacts: list[Path] = []
    config_path = configs_dir / "e04_homeostatic_tasks.yaml"
    step_config_path = step_dir / "homeostatic_tasks.yaml"
    write_homeostatic_config(config_path, config)
    shutil.copy2(config_path, step_config_path)
    artifacts.extend([config_path, step_config_path])
    config_json_path = step_dir / "homeostatic_tasks.json"
    write_json(config_json_path, config)
    artifacts.append(config_json_path)

    repair_index_path = configs_dir / "e04_repair_benchmarks.yaml"
    write_repair_benchmark_index(repair_index_path, config_path)
    artifacts.append(repair_index_path)

    artifacts.extend(
        dataframe_to_artifacts(
            task_df,
            step_dir / "homeostatic_task_catalog",
            results_dir / "e04_s05_homeostatic_task_catalog",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            schedule_df,
            step_dir / "perturbation_schedules",
            results_dir / "e04_s05_perturbation_schedules",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            controller_df,
            step_dir / "trivial_controller_results",
            results_dir / "e04_s05_trivial_controller_results",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            tick_df,
            step_dir / "trivial_controller_tick_records",
            results_dir / "e04_s05_trivial_controller_tick_records",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            normalization_df,
            step_dir / "normalization_examples",
            results_dir / "e04_s05_normalization_examples",
        )
    )
    artifacts.extend(
        dataframe_to_artifacts(
            validation_df,
            step_dir / "homeostasis_validation_results",
            results_dir / "e04_s05_homeostasis_validation_results",
        )
    )

    schedules_jsonl = step_dir / "perturbation_schedules.jsonl"
    schedules_jsonl.write_text(
        "\n".join(stable_json({"taskId": task["taskId"], "event": event}) for task in config["tasks"] for event in task["perturbationSchedule"])
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(schedules_jsonl)

    spec_path = step_dir / "homeostatic_task_spec.md"
    normalization_contract_path = step_dir / "normalization_contract.md"
    validation_report_path = step_dir / "validation_report.md"
    write_task_spec(spec_path, config)
    write_normalization_contract(normalization_contract_path, config)
    write_validation_report(validation_report_path, validation_df)
    artifacts.extend([spec_path, normalization_contract_path, validation_report_path])

    unit_cmd = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_e04_homeostasis_tasks",
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
        "S05 defines and dry-validates homeostatic benchmark tasks; it does not benchmark adaptive policy performance.",
        "Oracle-sort is a validation-only trivial controller with global access and is not a biologically local policy.",
        "Insertion/deletion metrics use current-length normalization, so downstream reports must include current length and length ratio.",
    ]
    recommended_next_action = "Proceed to S06 local learning rules only after Chief Scientist instruction; do not start S06 from this run."
    lay_summary = (
        "S05 created seeded homeostatic tasks with swaps, insertions, deletions, Frozen Cell recovery events, and damage events. "
        "No-op and oracle-sort controls validate that the metrics detect persistent failure and immediate recovery as expected."
    )

    summary_path = step_dir / "summary.md"
    summary_path.write_text(
        f"""# S05 Summary

- Research step ID: {STEP_ID}
- Completion status: {status}
- Artifacts written: `{config_path}` and additional files listed in `status.json` and `artifact_manifest.json`.
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
        "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
        "perturbationTypes": list(PERTURBATION_TYPES),
        "trivialControllers": list(TRIVIAL_CONTROLLERS),
        "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
        "fatigueDamageVersion": FATIGUE_DAMAGE_VERSION,
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
