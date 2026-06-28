#!/usr/bin/env python3
"""Run E05 S04 generalized-action validation and write artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace2d import (  # noqa: E402
    ACTION_SCHEMA_VERSION,
    ACTION_TRACE_SCHEMA_VERSION,
    LOCAL_ACTION_OBSERVATION_VERSION,
    ActionProposal,
    action_semantics_spec,
    audit_local_action_payload,
    build_s04_validation_world,
    run_standard_action_sequence,
    validate_action_trace_schema,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S04"
STEP_NUMBER = 4
STEP_TITLE = "Generalize actions"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "args": args,
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "runtimeSeconds": time.perf_counter() - started,
    }


def git_metadata() -> dict[str, Any]:
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


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataframe(df: pd.DataFrame, path_without_suffix: Path) -> list[Path]:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path_without_suffix.with_suffix(".csv")
    parquet_path = path_without_suffix.with_suffix(".parquet")
    safe = df.copy()
    for column in safe.columns:
        if safe[column].map(lambda item: isinstance(item, (Mapping, list, tuple))).any():
            safe[column] = safe[column].map(lambda item: json.dumps(json_ready(item), sort_keys=True) if isinstance(item, (Mapping, list, tuple)) else item)
    safe.to_csv(csv_path, index=False)
    safe.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".") if math.isfinite(value) else ""
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def action_spec_markdown(spec: Mapping[str, Any]) -> str:
    rows = [
        [row["action"], row["energyCost"], row["conservationMode"], row["targetRule"], row["collisionHandling"]]
        for row in spec["actions"]
    ]
    return f"""# E05 S04 Action Semantics Spec

Research step ID: {STEP_ID}
Title: {STEP_TITLE}
Action schema version: `{ACTION_SCHEMA_VERSION}`
Trace schema version: `{ACTION_TRACE_SCHEMA_VERSION}`
Local observation version: `{LOCAL_ACTION_OBSERVATION_VERSION}`

S04 defines local action semantics over the S01 substrate abstraction and S02 identity-bearing cells. Swap and crawl are the conservative movement core. Rotate, divide, die, adhere, detach, and local signal exchange are implemented with explicit legality checks, energy costs, and conservation or non-conservation trace fields.

{markdown_table(['action', 'energy', 'conservation mode', 'target rule', 'collision handling'], rows)}

Local policy observations expose only actor-local state and local neighbors. Whole-world occupancy maps, all-cell maps, target maps, hidden fields, and trace rows are prohibited from the local action payload.
"""


def build_validation_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    world, outcomes, action_rows = run_standard_action_sequence()
    trace_errors = validate_action_trace_schema(world.trace_rows)
    action_df = pd.DataFrame(action_rows)
    trace_df = pd.DataFrame(world.trace_rows)
    outcome_df = pd.DataFrame(
        [
            {
                "research_step_id": STEP_ID,
                **outcome.to_record(),
            }
            for outcome in outcomes
        ]
    )
    action_names = {outcome.action for outcome in outcomes if outcome.accepted}
    by_reason = {outcome.reason: outcome for outcome in outcomes}
    by_action = {outcome.action: outcome for outcome in outcomes if outcome.accepted}
    local_payload_world = build_s04_validation_world()
    local_payload = local_payload_world.local_observation(0).to_policy_payload()
    local_payload_errors = audit_local_action_payload(local_payload)
    validation_rows = [
        {
            "research_step_id": STEP_ID,
            "check_id": "required_action_coverage",
            "success": {"swap", "crawl", "rotate", "divide", "die", "adhere", "detach", "exchange_signal"}.issubset(action_names),
            "detail": ",".join(sorted(action_names)),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "collision_rejected_without_state_acceptance",
            "success": (not by_reason["crawl_target_occupied"].accepted) and by_reason["crawl_target_occupied"].collision,
            "detail": by_reason["crawl_target_occupied"].to_record(),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "occupancy_valid_after_each_action",
            "success": bool(action_df["state_valid_after"].all()),
            "detail": f"{int(action_df['state_valid_after'].sum())}/{len(action_df)} valid",
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "trace_schema_and_delta_consistency",
            "success": len(trace_errors) == 0,
            "detail": "; ".join(trace_errors),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "birth_trace_consistency",
            "success": by_action["divide"].cell_delta == 1 and by_action["divide"].born_cell_id is not None,
            "detail": by_action["divide"].to_record(),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "detach_trace_consistency",
            "success": by_action["detach"].cell_delta == 0 and by_action["detach"].detached_delta == 1 and by_action["detach"].occupied_delta == -1,
            "detail": by_action["detach"].to_record(),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "death_trace_consistency",
            "success": by_action["die"].cell_delta == -1 and by_action["die"].died_cell_id is not None,
            "detail": by_action["die"].to_record(),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "energy_costs_logged",
            "success": math.isclose(float(trace_df["energy_cost"].sum()), world.energy_spent, rel_tol=1e-12, abs_tol=1e-12)
            and world.energy_spent > 0.0,
            "detail": {
                "traceEnergySum": float(trace_df["energy_cost"].sum()),
                "totalEnergySpent": world.energy_spent,
                "acceptedActions": int(sum(outcome.accepted for outcome in outcomes)),
            },
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "conservation_modes_logged",
            "success": trace_df.loc[trace_df["event_kind"] == "action", "conservation_mode"].notna().all(),
            "detail": sorted(trace_df.loc[trace_df["event_kind"] == "action", "conservation_mode"].unique().tolist()),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "local_observation_payload_audit",
            "success": len(local_payload_errors) == 0,
            "detail": "; ".join(local_payload_errors),
        },
    ]
    validation_df = pd.DataFrame(validation_rows)
    conservation_df = trace_df.loc[trace_df["event_kind"] == "action", [
        "step_index",
        "action",
        "accepted",
        "legal",
        "collision",
        "energy_cost",
        "energy_spent",
        "conservation_mode",
        "cell_count_before",
        "cell_count_after",
        "cell_delta",
        "occupied_count_before",
        "occupied_count_after",
        "occupied_delta",
        "detached_count_before",
        "detached_count_after",
        "detached_delta",
        "born_cell_id",
        "died_cell_id",
        "detached_cell_id",
        "adhesion_bond_delta",
        "signal_delta_json",
    ]].copy()
    conservation_df.insert(0, "research_step_id", STEP_ID)
    payload_df = pd.DataFrame(
        [
            {
                "research_step_id": STEP_ID,
                "payload_schema_version": local_payload.get("schemaVersion"),
                "actor_cell_id": local_payload["actor"]["cellId"],
                "local_degree": local_payload["localDegree"],
                "neighbor_count": len(local_payload["neighbors"]),
                "payload_keys_json": json.dumps(sorted(local_payload)),
                "leakage_audit_passed": len(local_payload_errors) == 0,
                "payload_json": json.dumps(json_ready(local_payload), sort_keys=True),
            }
        ]
    )
    return validation_df, outcome_df, trace_df, conservation_df, payload_df


def write_outputs(artifacts_dir: Path) -> dict[str, Any]:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    configs_dir = artifacts_dir / "configs"
    code_index_dir = artifacts_dir / "code" / "e05_morphospace_simulator"
    for directory in [step_dir, results_dir, configs_dir, code_index_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    spec = action_semantics_spec()
    spec_json = step_dir / "action_semantics_spec.json"
    write_json(spec_json, spec)
    spec_md = step_dir / "action_semantics_spec.md"
    spec_md.write_text(action_spec_markdown(spec), encoding="utf-8")
    config_json = configs_dir / "e05_action_semantics.json"
    write_json(config_json, {"researchStepId": STEP_ID, "actionSemantics": spec})

    validation_df, outcome_df, trace_df, conservation_df, payload_df = build_validation_outputs()

    artifact_paths: list[Path] = [spec_json, spec_md, config_json]
    artifact_paths.extend(write_dataframe(validation_df, step_dir / "action_validation_results"))
    artifact_paths.extend(write_dataframe(validation_df, results_dir / "e05_s04_action_validation"))
    artifact_paths.extend(write_dataframe(outcome_df, step_dir / "action_outcomes"))
    artifact_paths.extend(write_dataframe(trace_df, step_dir / "example_action_trace"))
    artifact_paths.extend(write_dataframe(conservation_df, step_dir / "conservation_ledger"))
    artifact_paths.extend(write_dataframe(payload_df, step_dir / "local_action_payload_audit"))

    code_index_path = code_index_dir / "README.md"
    code_index_path.write_text(
        "# E05 Morphospace Simulator Code Index\n\n"
        "Repository-backed source is kept in git per workspace instructions.\n\n"
        "- S01 package: `morphospace2d/substrates.py`\n"
        "- S02 package: `morphospace2d/identities.py`\n"
        "- S03 package: `morphospace2d/targets.py`\n"
        "- S04 package: `morphospace2d/actions.py`\n"
        "- S04 runner: `scripts/e05_s04_actions.py`\n"
        "- S04 tests: `tests/test_e05_actions.py`\n",
        encoding="utf-8",
    )
    artifact_paths.append(code_index_path)

    unit_log_path = step_dir / "repo_unit_test_log.txt"
    unit_test_success = unit_log_path.exists() and "OK" in unit_log_path.read_text(encoding="utf-8", errors="replace")
    if unit_log_path.exists():
        artifact_paths.append(unit_log_path)

    success = bool(validation_df["success"].all() and unit_test_success)
    status_path = step_dir / "status.json"
    summary_path = step_dir / "summary.md"
    manifest_path = step_dir / "artifact_manifest.json"
    reported_artifact_paths = artifact_paths + [status_path, summary_path, manifest_path]
    validation_result = (
        "Passed: S04 action semantics cover swap, crawl, rotate, divide, die, adhere, detach, and local signal exchange; "
        "occupancy legality and collision rejection were validated; birth, death, detach, adhesion, signal, energy, and conservation/non-conservation trace fields are consistent; "
        "local action payload audits exclude whole-world maps, target maps, traces, and hidden fields; focused repository unit tests passed."
        if success
        else "Failed: at least one S04 action validation or focused unit-test check did not pass."
    )
    caveats = (
        "S04 action semantics are computational proxy mechanics. Division and death intentionally change cell counts, detach preserves live off-substrate cells, and downstream metrics in S05 must handle these non-conservative cases explicitly."
    )
    recommended = "Proceed to S05 to define morphospace metrics only after Chief Scientist instruction."
    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "artifactsWritten": [str(path) for path in reported_artifact_paths],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended,
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "validationCommands": [
            {
                "args": "PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_e05_actions tests.test_e05_target_morphologies tests.test_e05_cell_identity tests.test_e05_substrate_generalization",
                "logPath": str(unit_log_path),
                "success": unit_test_success,
            }
        ],
        "generatedAt": utc_now(),
        "git": git_metadata(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threadEnvironment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
        },
    }
    write_json(status_path, status_payload)

    summary_rows = [[row["check_id"], row["success"], str(row["detail"])[:140]] for row in validation_df.to_dict("records")]
    summary = f"""# E05 S04 Status Summary

- Step ID: {STEP_ID}
- Completion status: {'completed' if success else 'completed with validation failures'}
- Artifacts written: {', '.join(str(path) for path in reported_artifact_paths)}
- Validation result: {validation_result}
- Outcome classification: {'supportive' if success else 'constraining/contradictory'}
- Caveats or blockers: {caveats}
- Lay summary: The simulator now has explicit local action rules for movement, polarity rotation, birth, death, adhesion, detachment, and local signal exchange. The example trace records which actions conserve cells and which change cell or occupancy counts.
- Recommended next action: {recommended}

## Validation Table

{markdown_table(['check', 'success', 'detail'], summary_rows)}
"""
    summary_path.write_text(summary, encoding="utf-8")

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": "eidosoma.step_artifact_manifest.v1",
        "generatedAt": utc_now(),
        "artifacts": [
            {"path": str(path), "sha256": sha256_path(path), "bytes": path.stat().st_size}
            for path in artifact_paths + [status_path, summary_path]
            if path.exists() and path.is_file()
        ],
        "sourceCode": {
            "repositoryPackage": "morphospace2d/",
            "runner": "scripts/e05_s04_actions.py",
            "tests": "tests/test_e05_actions.py",
            "note": "Repository source is committed in git and not copied into artifacts per GitHub workspace rule.",
        },
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended,
    }
    write_json(manifest_path, manifest_payload)

    return {"success": success, "stepDir": step_dir, "status": status_payload, "artifactPaths": [str(path) for path in reported_artifact_paths]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    result = write_outputs(args.artifacts_dir)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
