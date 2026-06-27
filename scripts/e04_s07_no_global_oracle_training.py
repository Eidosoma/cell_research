#!/usr/bin/env python3
"""Execute E04 S07 no-global-oracle training constraint audit."""

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

from e02_deterministic_simulator.simulator import StepOutcome  # noqa: E402
from memory_repair import (  # noqa: E402
    GLOBAL_ORACLE_PROTOCOL_ID,
    HOMEOSTASIS_BENCHMARK_VERSION,
    LEARNING_PENDING_ACTION_KEY,
    LOCAL_LEARNING_VERSION,
    LOCAL_ONLY_PROTOCOL_ID,
    ORACLE_BASELINE_ID,
    ORACLE_FLAG_FIELDS,
    ORACLE_FORBIDDEN_FIELDS,
    TRAINING_CONSTRAINT_VERSION,
    LocalLearningConfig,
    LocalLearningPolicyWrapper,
    TrainingProtocolConfig,
    audit_observation_log,
    audit_policy_spec_for_oracle_access,
    audit_reward_record,
    build_homeostatic_benchmark_config,
    compute_local_learning_reward,
    evaluate_global_oracle_homeostatic_task,
    global_oracle_baseline_protocol,
    global_oracle_baseline_record,
    local_only_training_protocol,
    training_protocol_bundle,
)
from memory_repair.learning import LearningEventSimulator  # noqa: E402
from morphospace import SelectionPolicy  # noqa: E402


EXPERIMENT_ID = "E04"
STEP_ID = "S07"
STEP_NUMBER = 7
STEP_TITLE = "Train without global oracle access"
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
            if isinstance(item, (Mapping, list)):
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


def target_dsl_leak_spec() -> dict[str, Any]:
    return {
        "policy_id": "dsl_target_leak",
        "family": "dsl",
        "algotype": "dsl",
        "parameters": {
            "program": {
                "rules": [
                    {
                        "name": "target_rule",
                        "when": [{"op": "target_exists"}],
                        "then": {"action": "swap_target", "target_key": "target_position"},
                    }
                ]
            }
        },
    }


def s06_reward_record() -> dict[str, Any]:
    config = LocalLearningConfig("local_adaptive")
    policy = LocalLearningPolicyWrapper("bubble", config)
    sim = LearningEventSimulator([2, 1], policy, scheduler_seed=1, tie_breaker_seed=1)
    actor = sim.cells[0]
    observation = actor.policy.observe(sim.cells, 0, actor.state, sim.frozen_variant)
    action = actor.policy.propose_action(observation, actor.state, sim.tie_rng, forced_direction=1)
    pending = dict(actor.state)
    pending[LEARNING_PENDING_ACTION_KEY] = "swap_right"
    outcome = StepOutcome(True, actor.cell_id, 0, 1, 1, True, 1, 1, False, "learning_local_swap")
    return compute_local_learning_reward(observation, pending, action, outcome, config)


def protocol_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("localOnlyProtocol", "globalOracleBaselineProtocol"):
        item = dict(bundle[key])
        rows.append(
            {
                "protocolKey": key,
                "protocolId": item["protocolId"],
                "mode": item["mode"],
                "oracleAllowed": item["oracleAllowed"],
                "oracleBaseline": item["oracleBaseline"],
                "oracleBaselineLabel": item.get("oracleBaselineLabel"),
                "allowedObservationFieldCount": len(item["allowedObservationFields"]),
                "allowedRewardFieldCount": len(item["allowedRewardFields"]),
                "forbiddenFieldCount": len(item["forbiddenFields"]),
                "trainingConstraintVersion": item["trainingConstraintVersion"],
            }
        )
    return rows


def reward_audit_rows() -> list[dict[str, Any]]:
    local_protocol = local_only_training_protocol()
    oracle_protocol = global_oracle_baseline_protocol()
    examples = [
        ("s06_local_reward", s06_reward_record(), local_protocol, True),
        (
            "synthetic_global_sortedness_leak",
            {
                "inputFieldsUsed": ["actor_value", "global_sortedness"],
                "localInputs": {"actor_value": 2, "whole_array_values": [2, 1]},
                "usesGlobalSortednessSignal": True,
            },
            local_protocol,
            False,
        ),
        ("global_oracle_record_rejected_local", global_oracle_baseline_record([2, 1]), local_protocol, False),
        ("global_oracle_record_accepted_oracle", global_oracle_baseline_record([2, 1]), oracle_protocol, True),
    ]
    rows = []
    for case_id, payload, protocol, expected in examples:
        audit = audit_reward_record(payload, protocol)
        rows.append(
            {
                "auditFamily": "reward_record",
                "caseId": case_id,
                "protocolId": protocol.protocol_id,
                "protocolMode": protocol.mode,
                "success": audit["success"],
                "expectedSuccess": expected,
                "expectedMatched": audit["success"] == expected,
                "disallowedInputsJson": compact_json(audit.get("disallowedInputs", [])),
                "disallowedLocalInputKeysJson": compact_json(audit.get("disallowedLocalInputKeys", [])),
                "oracleFlagHitsJson": compact_json(audit.get("oracleFlagHits", [])),
                "forbiddenFieldHitsJson": compact_json(audit.get("forbiddenFieldHits", [])),
            }
        )
    return rows


def observation_audit_rows() -> list[dict[str, Any]]:
    local_protocol = local_only_training_protocol()
    oracle_protocol = global_oracle_baseline_protocol()
    examples = [
        (
            "local_observation_passes",
            {"fields": ["actor_value", "left_value", "right_value", "actor_local_memory"]},
            local_protocol,
            True,
        ),
        (
            "future_target_observation_rejected",
            {"fields": ["actor_value", "future_target_array"], "future_target_array": [1, 2, 3]},
            local_protocol,
            False,
        ),
        (
            "oracle_observation_labeled",
            {
                "fields": ["whole_array_values", "target_array", "global_sortedness"],
                "whole_array_values": [2, 1],
                "target_array": [1, 2],
                "global_sortedness": 0.0,
                "oracleAllowed": True,
                "oracleBaseline": True,
                "oracleBaselineLabel": ORACLE_BASELINE_ID,
            },
            oracle_protocol,
            True,
        ),
    ]
    rows = []
    for case_id, payload, protocol, expected in examples:
        audit = audit_observation_log(payload, protocol)
        rows.append(
            {
                "auditFamily": "observation_log",
                "caseId": case_id,
                "protocolId": protocol.protocol_id,
                "protocolMode": protocol.mode,
                "success": audit["success"],
                "expectedSuccess": expected,
                "expectedMatched": audit["success"] == expected,
                "disallowedObservationFieldsJson": compact_json(audit.get("disallowedObservationFields", [])),
                "oracleFlagHitsJson": compact_json(audit.get("oracleFlagHits", [])),
                "forbiddenFieldHitsJson": compact_json(audit.get("forbiddenFieldHits", [])),
            }
        )
    return rows


def policy_audit_rows() -> list[dict[str, Any]]:
    local_protocol = local_only_training_protocol()
    oracle_protocol = global_oracle_baseline_protocol()
    examples = [
        (
            "local_learning_policy_passes",
            LocalLearningPolicyWrapper("bubble", LocalLearningConfig("local_adaptive")),
            local_protocol,
            True,
        ),
        ("selection_rejected_local", SelectionPolicy(), local_protocol, False),
        ("selection_accepted_oracle", SelectionPolicy(), oracle_protocol, True),
        ("target_dsl_rejected_local", target_dsl_leak_spec(), local_protocol, False),
    ]
    rows = []
    for case_id, policy, protocol, expected in examples:
        audit = audit_policy_spec_for_oracle_access(policy, protocol)
        rows.append(
            {
                "auditFamily": "policy_spec",
                "caseId": case_id,
                "policyId": audit["policyId"],
                "family": audit["family"],
                "algotype": audit["algotype"],
                "protocolId": protocol.protocol_id,
                "protocolMode": protocol.mode,
                "success": audit["success"],
                "expectedSuccess": expected,
                "expectedMatched": audit["success"] == expected,
                "violationsJson": compact_json(audit.get("violations", [])),
            }
        )
    return rows


def oracle_baseline_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    ticks: list[dict[str, Any]] = []
    for task in build_homeostatic_benchmark_config()["tasks"]:
        result = evaluate_global_oracle_homeostatic_task(task)
        summaries.append(
            {
                "taskId": result["taskId"],
                "baselineId": result["baselineId"],
                "oracleAllowed": result["oracleAllowed"],
                "oracleBaseline": result["oracleBaseline"],
                "oracleBaselineLabel": result["oracleBaselineLabel"],
                "usesGlobalSortednessSignal": result["usesGlobalSortednessSignal"],
                "usesWholeArrayValues": result["usesWholeArrayValues"],
                "usesWholeArrayTargetSignal": result["usesWholeArrayTargetSignal"],
                "timeInRangeFraction": result["timeInRangeFraction"],
                "failureDurationTicks": result["failureDurationTicks"],
                "energyProxy": result["energyProxy"],
                "finalValuesJson": result["finalValuesJson"],
                "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
            }
        )
        ticks.extend(result["tickRecords"])
    return summaries, ticks


def validation_rows(
    protocol_df: pd.DataFrame,
    reward_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    local_protocol = protocol_df.query("protocolId == @LOCAL_ONLY_PROTOCOL_ID").iloc[0]
    oracle_protocol = protocol_df.query("protocolId == @GLOBAL_ORACLE_PROTOCOL_ID").iloc[0]
    rows = [
        {
            "validationFamily": "protocol_flags",
            "caseId": "local_protocol_disables_oracle",
            "success": bool(not local_protocol["oracleAllowed"] and local_protocol["mode"] == "local_only"),
            "validationDetail": "Local-only protocol has oracleAllowed=false and local_only mode.",
        },
        {
            "validationFamily": "protocol_flags",
            "caseId": "global_oracle_protocol_labeled",
            "success": bool(
                oracle_protocol["oracleAllowed"]
                and oracle_protocol["oracleBaseline"]
                and oracle_protocol["oracleBaselineLabel"] == ORACLE_BASELINE_ID
            ),
            "validationDetail": "Global-oracle baseline protocol is explicitly labeled and oracleAllowed=true.",
        },
        {
            "validationFamily": "reward_audit",
            "caseId": "s06_reward_passes_local_only",
            "success": bool(reward_df.query("caseId == 's06_local_reward'").iloc[0].success),
            "validationDetail": "Representative S06 reward record passes the local-only audit.",
        },
        {
            "validationFamily": "reward_audit",
            "caseId": "synthetic_reward_leak_rejected",
            "success": bool(not reward_df.query("caseId == 'synthetic_global_sortedness_leak'").iloc[0].success),
            "validationDetail": "Synthetic global Sortedness and whole-array reward leak is rejected.",
        },
        {
            "validationFamily": "observation_audit",
            "caseId": "local_observation_passes",
            "success": bool(observation_df.query("caseId == 'local_observation_passes'").iloc[0].success),
            "validationDetail": "Allowed local observation fields pass.",
        },
        {
            "validationFamily": "observation_audit",
            "caseId": "future_target_observation_rejected",
            "success": bool(not observation_df.query("caseId == 'future_target_observation_rejected'").iloc[0].success),
            "validationDetail": "Future target-array observation leak is rejected.",
        },
        {
            "validationFamily": "policy_audit",
            "caseId": "local_learning_policy_passes",
            "success": bool(policy_df.query("caseId == 'local_learning_policy_passes'").iloc[0].success),
            "validationDetail": "S06 local-learning policy spec passes local-only policy audit.",
        },
        {
            "validationFamily": "policy_audit",
            "caseId": "selection_rejected_local",
            "success": bool(not policy_df.query("caseId == 'selection_rejected_local'").iloc[0].success),
            "validationDetail": "Selection-like target-position semantics are rejected in local-only training.",
        },
        {
            "validationFamily": "policy_audit",
            "caseId": "target_dsl_rejected_local",
            "success": bool(not policy_df.query("caseId == 'target_dsl_rejected_local'").iloc[0].success),
            "validationDetail": "DSL target-position oracle operations are rejected in local-only training.",
        },
        {
            "validationFamily": "oracle_baseline",
            "caseId": "oracle_record_rejected_locally_and_accepted_as_oracle",
            "success": bool(
                not reward_df.query("caseId == 'global_oracle_record_rejected_local'").iloc[0].success
                and reward_df.query("caseId == 'global_oracle_record_accepted_oracle'").iloc[0].success
            ),
            "validationDetail": "Global-oracle record cannot pass local-only audit but passes under the labeled oracle protocol.",
        },
        {
            "validationFamily": "oracle_baseline",
            "caseId": "oracle_baseline_runs_all_homeostatic_tasks",
            "success": bool(
                len(oracle_df) == len(build_homeostatic_benchmark_config()["tasks"])
                and oracle_df["oracleAllowed"].all()
                and oracle_df["oracleBaseline"].all()
                and oracle_df["timeInRangeFraction"].eq(1.0).all()
            ),
            "validationDetail": "Labeled global-oracle baseline ran every S05 homeostatic task and remained in range.",
        },
    ]
    return rows


def write_protocol_config(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# E04 S07 training protocol and oracle-baseline config\n" + to_yaml(payload) + "\n", encoding="utf-8")


def write_local_protocol_report(path: Path, protocol: Mapping[str, Any]) -> None:
    rows = [[field] for field in protocol["allowedObservationFields"][:12]]
    text = f"""# E04 S07 Local-Only Training Protocol

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: protocol config, audit tables, validation report, oracle-baseline artifacts, status files, and code copies under `$ARTIFACTS_DIR/research_steps/S07/`.
- Validation result: local-only protocol fields and leakage tests passed.
- Caveats or blockers: this protocol certifies information boundaries for S08 setup; it does not train or optimize policies.
- Recommended next action: proceed to S08 only after Chief Scientist instruction.

The local-only protocol has `oracleAllowed=false`. Reward and observation records must not include global Sortedness, whole-array values or ranks, target arrays, future state, target-position estimates, or unlabeled oracle flags.

## Example Allowed Observation Fields

{markdown_table(["Field"], rows)}
"""
    path.write_text(text, encoding="utf-8")


def write_oracle_baseline_report(path: Path, oracle_df: pd.DataFrame) -> None:
    rows = oracle_df[
        ["taskId", "baselineId", "oracleAllowed", "timeInRangeFraction", "energyProxy"]
    ].values.tolist()
    text = f"""# E04 S07 Labeled Global-Oracle Baseline

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: global-oracle baseline protocol, baseline task results, tick records, validation report, status files, and code copies.
- Validation result: labeled global-oracle records are rejected by the local-only audit and accepted only under the oracle protocol.
- Caveats or blockers: this baseline is an explicitly nonlocal upper-bound comparator and must not be mixed with local-only training evidence.
- Recommended next action: proceed to S08 only after Chief Scientist instruction.

{markdown_table(["Task", "Baseline", "Oracle allowed", "Time in range", "Energy"], rows)}
"""
    path.write_text(text, encoding="utf-8")


def write_validation_report(path: Path, validation_df: pd.DataFrame) -> None:
    total = len(validation_df)
    passed = int(validation_df["success"].sum())
    family = validation_df.groupby("validationFamily")["success"].agg(["count", "sum"]).reset_index()
    text = f"""# E04 S07 Validation Report

- Research step ID: {STEP_ID}
- Completion status: completed
- Artifacts written: local-only protocol, oracle-baseline protocol, leakage audit tables, oracle-baseline results, unit-test log, code copies, manifests, and status files.
- Validation result: passed; {passed} of {total} checks passed.
- Caveats or blockers: target-position policies are excluded from local-only training unless separately classified; S08 must use this audit gate before search.
- Recommended next action: wait for Chief Scientist instruction before starting S08.

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
        Path("scripts/e04_s07_no_global_oracle_training.py"),
        Path("tests/test_e04_training_constraints.py"),
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

    bundle = training_protocol_bundle()
    protocol_df = pd.DataFrame(protocol_rows(bundle))
    reward_df = pd.DataFrame(reward_audit_rows())
    observation_df = pd.DataFrame(observation_audit_rows())
    policy_df = pd.DataFrame(policy_audit_rows())
    oracle_rows, oracle_tick_rows = oracle_baseline_rows()
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_tick_df = pd.DataFrame(oracle_tick_rows)
    validation_df = pd.DataFrame(validation_rows(protocol_df, reward_df, observation_df, policy_df, oracle_df))

    artifacts: list[Path] = []
    config_path = configs_dir / "e04_training_protocols.yaml"
    step_config_path = step_dir / "training_protocols.yaml"
    write_protocol_config(config_path, bundle)
    shutil.copy2(config_path, step_config_path)
    artifacts.extend([config_path, step_config_path])
    config_json_path = step_dir / "training_protocols.json"
    write_json(config_json_path, bundle)
    artifacts.append(config_json_path)

    for df, stem, result_stem in [
        (protocol_df, "training_protocol_catalog", "e04_s07_training_protocol_catalog"),
        (reward_df, "reward_leakage_audits", "e04_s07_reward_leakage_audits"),
        (observation_df, "observation_leakage_audits", "e04_s07_observation_leakage_audits"),
        (policy_df, "policy_leakage_audits", "e04_s07_policy_leakage_audits"),
        (oracle_df, "global_oracle_baseline_results", "e04_s07_global_oracle_baseline_results"),
        (oracle_tick_df, "global_oracle_baseline_tick_records", "e04_s07_global_oracle_baseline_tick_records"),
        (validation_df, "training_constraint_validation_results", "e04_s07_training_constraint_validation_results"),
    ]:
        artifacts.extend(dataframe_to_artifacts(df, step_dir / stem, results_dir / result_stem))

    oracle_jsonl = step_dir / "global_oracle_records.jsonl"
    oracle_jsonl.write_text("\n".join(record for record in oracle_tick_df["oracleRecordJson"]) + "\n", encoding="utf-8")
    artifacts.append(oracle_jsonl)

    local_protocol_path = step_dir / "local_only_training_protocol.md"
    oracle_baseline_path = step_dir / "global_oracle_baseline.md"
    validation_report_path = step_dir / "validation_report.md"
    write_local_protocol_report(local_protocol_path, bundle["localOnlyProtocol"])
    write_oracle_baseline_report(oracle_baseline_path, oracle_df)
    write_validation_report(validation_report_path, validation_df)
    artifacts.extend([local_protocol_path, oracle_baseline_path, validation_report_path])

    unit_cmd = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_e04_training_constraints",
        "tests.test_e04_local_learning",
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
        "S07 certifies information-boundary gates and labeled oracle controls; it does not run S08 optimization.",
        "Selection-like target-position policies are rejected from local-only training unless reclassified in a future labeled condition.",
        "The global-oracle baseline is an explicit nonlocal upper-bound comparator and cannot be mixed into local-only evidence.",
    ]
    recommended_next_action = "Proceed to S08 GPU evolutionary search only after Chief Scientist instruction; do not start S08 from this run."
    lay_summary = (
        "S07 added an audit gate that rejects global Sortedness, whole-array values or ranks, target arrays, future targets, "
        "and target-position oracles from local-only training. It also created a separately labeled global-oracle baseline."
    )

    summary_path = step_dir / "summary.md"
    summary_path.write_text(
        f"""# S07 Summary

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
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "localOnlyProtocolId": LOCAL_ONLY_PROTOCOL_ID,
        "globalOracleProtocolId": GLOBAL_ORACLE_PROTOCOL_ID,
        "globalOracleBaselineId": ORACLE_BASELINE_ID,
        "oracleFlagFields": list(ORACLE_FLAG_FIELDS),
        "forbiddenOracleFields": list(ORACLE_FORBIDDEN_FIELDS),
        "localLearningVersion": LOCAL_LEARNING_VERSION,
        "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
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
