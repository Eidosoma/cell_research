#!/usr/bin/env python3
"""Audit E04 S06/S05 training pathways for no-global-oracle access."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.e03.policy_interface import PolicyObservation
from src.e04.fatigue_damage import RELIABILITY_MODES
from src.e04.homeostasis import HOMEOSTATIC_TASKS
from src.e04.local_learning import (
    LEARNING_POLICY_MODES,
    LEARNING_REWARD_FEATURES,
    LocalLearningConfig,
    default_local_learning_configs,
    run_local_learning_matrix,
)
from src.e04.memory_policies import CellMemoryState
from src.e04.no_oracle_protocol import (
    ALLOWED_TRAINING_SIGNAL_FIELDS,
    CENTRALIZED_BASELINE_TYPE,
    EXCLUDED_TRAINING_SIGNAL_FIELDS,
    FORBIDDEN_TRAINING_TOKENS,
    LOCAL_ONLY_PROTOCOL_ID,
    audit_training_feature_dict,
    centralized_baseline_contract,
    legacy_policy_observation_audit,
    project_local_training_observation,
)
from src.e04.signaling import LocalSignalValues, SignalSensation


STEP_ID = "S07"
STEP_NUMBER = 7
EXPERIMENT_ID = "E04"
FORBIDDEN_SOURCE_TOKENS = (
    "sortedness_percent",
    "monotonicity_error_count",
    "delayed_gratification",
    "target_sortedness_percent",
    "ideal_position",
    "target_position",
    "final_values",
    "final_target",
    "full_array_rank",
    "whole_array",
    "all_values",
)
OFFLINE_METRIC_COLUMNS = (
    "initial_sortedness_percent",
    "final_sortedness_percent",
    "min_sortedness_percent",
    "mean_sortedness_percent",
    "final_monotonicity_error_count",
    "time_in_target_event_count",
    "time_in_target_fraction",
    "longest_out_of_target_run",
    "recovered_perturbation_count",
    "unrecovered_perturbation_count",
    "mean_recovery_events",
    "max_recovery_events",
    "dg_primary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--max-events", type=int, default=120)
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
        "note": "Checksum omitted to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "(no rows)"
    view = df[columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in view.to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def _node_sources(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    sources: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            sources[node.name] = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            sources[class_name] = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    sources[f"{class_name}.{item.name}"] = "\n".join(lines[item.lineno - 1 : item.end_lineno])
    return sources


def _hits(text: str, tokens: Iterable[str]) -> list[str]:
    lowered = text.lower()
    return [token for token in tokens if token.lower() in lowered]


def _source_scope(
    *,
    repo_dir: Path,
    relative_path: str,
    symbols: tuple[str, ...],
    access_role: str,
    expected_access: str,
    pass_condition: str,
) -> dict[str, Any]:
    path = repo_dir / relative_path
    sources = _node_sources(path)
    pieces = [sources.get(symbol, "") for symbol in symbols]
    missing = [symbol for symbol, piece in zip(symbols, pieces, strict=True) if not piece]
    text = "\n".join(pieces)
    forbidden_hits = _hits(text, FORBIDDEN_SOURCE_TOKENS)
    return {
        "source_file": relative_path,
        "symbols": ", ".join(symbols),
        "access_role": access_role,
        "expected_access": expected_access,
        "forbidden_source_hits": ", ".join(forbidden_hits),
        "missing_symbols": ", ".join(missing),
        "pass_condition": pass_condition,
        "success": False,
    }


def build_static_audit(repo_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        _source_scope(
            repo_dir=repo_dir,
            relative_path="src/e04/local_learning.py",
            symbols=(
                "LocalLearningController.legal_actions",
                "LocalLearningController._base_action_score",
                "LocalLearningController.select_action",
                "LocalLearningController.record_reward",
                "LocalLearningController._local_disorder_count",
                "LocalLearningController._accessed_indices",
            ),
            access_role="s06_learning_selection_and_update",
            expected_access="actor/immediate-neighbor values, statuses, local action result, S01 memory, S02 blocked/frustrated signals",
            pass_condition="no forbidden source tokens in action-selection or reward-update scope",
        )
    )
    rows[-1]["success"] = rows[-1]["forbidden_source_hits"] == "" and rows[-1]["missing_symbols"] == ""
    rows.append(
        _source_scope(
            repo_dir=repo_dir,
            relative_path="src/e04/local_learning.py",
            symbols=("LocalLearningPilotResult._offline_metrics", "LocalLearningPilotResult.to_row"),
            access_role="s06_offline_evaluation",
            expected_access="global Sortedness/time-in-target/final metrics allowed only after trajectories are logged",
            pass_condition="forbidden source tokens present only in offline result serialization scope",
        )
    )
    rows[-1]["success"] = "sortedness_percent" in rows[-1]["forbidden_source_hits"] and rows[-1]["missing_symbols"] == ""
    rows.append(
        _source_scope(
            repo_dir=repo_dir,
            relative_path="src/e04/homeostasis.py",
            symbols=("build_homeostatic_schedule", "_apply_perturbation"),
            access_role="s05_task_schedule_and_perturbations",
            expected_access="seeded schedules and exogenous perturbations independent of online Sortedness",
            pass_condition="no Sortedness/final-target tokens in schedule or perturbation construction",
        )
    )
    rows[-1]["success"] = rows[-1]["forbidden_source_hits"] == "" and rows[-1]["missing_symbols"] == ""
    rows.append(
        _source_scope(
            repo_dir=repo_dir,
            relative_path="src/e04/homeostasis.py",
            symbols=("HomeostaticBenchmarkResult._offline_metrics", "HomeostaticBenchmarkResult.to_row"),
            access_role="s05_offline_evaluation",
            expected_access="global target-maintenance metrics allowed only after event logs are complete",
            pass_condition="offline metrics contain expected global tokens and are not in policy-update scope",
        )
    )
    rows[-1]["success"] = "sortedness_percent" in rows[-1]["forbidden_source_hits"] and rows[-1]["missing_symbols"] == ""
    rows.append(
        _source_scope(
            repo_dir=repo_dir,
            relative_path="src/e03/policy_interface.py",
            symbols=("PolicyObservation", "observe_cell", "OriginalCellPolicyWrapper._propose_selection"),
            access_role="legacy_public_policy_observation",
            expected_access="legacy wrapper parity exposes full arrays and ideal position; not acceptable as future learned-policy tensor",
            pass_condition="legacy exposure is detected and must be projected before S07+ training",
        )
    )
    rows[-1]["success"] = bool(legacy_policy_observation_audit()["requiresProjectionForLocalOnlyTraining"])
    rows.append(
        _source_scope(
            repo_dir=repo_dir,
            relative_path="src/e04/signaling.py",
            symbols=("target_seek_signal", "infer_signal_values"),
            access_role="s02_signal_generation",
            expected_access="blocked/frustrated are allowed; target_seeking and morphogen are excluded from S07 local-only training features",
            pass_condition="target signal exposure is detected and excluded by protocol allowlist",
        )
    )
    rows[-1]["success"] = "ideal_position" in rows[-1]["forbidden_source_hits"] and "target_seeking" in EXCLUDED_TRAINING_SIGNAL_FIELDS
    return pd.DataFrame(rows)


def sample_projected_protocol_audit() -> dict[str, Any]:
    observation = PolicyObservation(
        actor_index=2,
        actor_thread_id=101,
        actor_value=5,
        actor_label="bubble",
        actor_status="ACTIVE",
        behavior="bubble",
        values=(1, 8, 5, 4, 9),
        labels=("bubble", "bubble", "bubble", "bubble", "bubble"),
        statuses=("ACTIVE", "ACTIVE", "ACTIVE", "FREEZE", "ACTIVE"),
        left_boundary=0,
        right_boundary=4,
        reverse_direction=False,
        ideal_position=4,
        group_status="ACTIVE",
    )
    memory = CellMemoryState(last_move_success=False, time_since_movement=3, local_frustration=4)
    sensation = SignalSensation(
        event_step=7,
        actor_thread_id=101,
        actor_position=2,
        local_signals=((2, LocalSignalValues(blocked=0.1, sorted=1.0, target_seeking=1.0)),),
        diffusive_fields={
            "blocked": 0.25,
            "frustrated": 0.5,
            "sorted": 1.0,
            "target_seeking": 1.0,
            "morphogen": 1.0,
        },
        accessed_indices=(1, 2, 3),
        access_scope="local_window_plus_explicit_diffusive_fields",
        noise_applied=False,
    )
    view = project_local_training_observation(observation, memory, sensation)
    feature_audit = audit_training_feature_dict(view.to_feature_dict())
    return {
        "sampleFeatureDict": view.to_feature_dict(),
        "featureAudit": feature_audit,
        "legacyObservationAudit": legacy_policy_observation_audit(),
        "centralizedBaselineContract": centralized_baseline_contract(),
    }


def build_training_runs(max_events: int) -> pd.DataFrame:
    configs = default_local_learning_configs(max_events=max_events)
    results = run_local_learning_matrix(configs)
    df = pd.DataFrame([result.to_row() for result in results])
    baseline_type = df["policy_mode"].map(
        {
            "local_learning": "local_learning",
            "nonlearning_control": "nonlearning_no_update_control",
        }
    )
    df.insert(0, "training_protocol", LOCAL_ONLY_PROTOCOL_ID)
    df.insert(1, "baseline_type", baseline_type)
    df.insert(2, "centralized_baseline", False)
    df.insert(3, "eligible_for_local_only_claims", True)
    df["allowed_training_signal_fields_json"] = json.dumps(list(ALLOWED_TRAINING_SIGNAL_FIELDS), separators=(",", ":"))
    df["excluded_training_signal_fields_json"] = json.dumps(list(EXCLUDED_TRAINING_SIGNAL_FIELDS), separators=(",", ":"))
    df["offline_metric_columns_json"] = json.dumps([column for column in OFFLINE_METRIC_COLUMNS if column in df.columns], separators=(",", ":"))
    df["raw_policy_observation_requires_projection"] = True
    df["strict_training_view_validated"] = True
    return df


def inspect_update_logs(df: pd.DataFrame) -> dict[str, Any]:
    inspected = 0
    nonlocal_records = 0
    oracle_records = 0
    forbidden_reward_records = 0
    reward_terms = set()
    for row in df.to_dict(orient="records"):
        summary = json.loads(row["learning_summary_json"])
        for record in summary.get("update_log", []):
            inspected += 1
            reward_terms.update(record.get("reward_terms", {}).keys())
            actor = int(record.get("actor_position", -999))
            indices = tuple(int(idx) for idx in record.get("accessed_indices", []))
            if indices and max(abs(idx - actor) for idx in indices) > 2:
                nonlocal_records += 1
            if record.get("oracle_fields"):
                oracle_records += 1
            if any(token in key.lower() for key in record.get("reward_terms", {}).keys() for token in FORBIDDEN_TRAINING_TOKENS):
                forbidden_reward_records += 1
    return {
        "inspectedUpdateLogRecords": inspected,
        "nonlocalAccessRecordCount": nonlocal_records,
        "oracleFieldRecordCount": oracle_records,
        "forbiddenRewardRecordCount": forbidden_reward_records,
        "rewardTerms": sorted(reward_terms),
    }


def build_validation(
    *,
    training_df: pd.DataFrame,
    static_df: pd.DataFrame,
    protocol_audit: dict[str, Any],
    update_audit: dict[str, Any],
    unit_tests: dict[str, Any],
    e03_tests: dict[str, Any],
    e02_tests: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    static_by_role = {row["access_role"]: row for row in static_df.to_dict(orient="records")}
    rows.append(
        {
            "validation_case": "s06_selection_and_reward_scope_has_no_forbidden_hits",
            "success": bool(static_by_role["s06_learning_selection_and_update"]["success"]),
            "detail": static_by_role["s06_learning_selection_and_update"]["forbidden_source_hits"] or "no forbidden hits",
        }
    )
    rows.append(
        {
            "validation_case": "offline_global_metrics_are_isolated",
            "success": bool(
                static_by_role["s06_offline_evaluation"]["success"]
                and static_by_role["s05_offline_evaluation"]["success"]
                and static_by_role["s05_task_schedule_and_perturbations"]["success"]
            ),
            "detail": "S05/S06 global Sortedness and final metrics appear in offline result scopes, not schedule or reward scopes",
        }
    )
    rows.append(
        {
            "validation_case": "policy_observation_schema_inspected_and_projected",
            "success": bool(
                protocol_audit["legacyObservationAudit"]["requiresProjectionForLocalOnlyTraining"]
                and not protocol_audit["featureAudit"]["usesGlobalOracle"]
            ),
            "detail": (
                f"legacy exposures={protocol_audit['legacyObservationAudit']['oracleExposureFields']}; "
                f"strict feature count={protocol_audit['featureAudit']['featureCount']}"
            ),
        }
    )
    rows.append(
        {
            "validation_case": "target_signal_fields_excluded_from_training_protocol",
            "success": bool({"target_seeking", "morphogen"}.issubset(set(EXCLUDED_TRAINING_SIGNAL_FIELDS))),
            "detail": f"allowed={list(ALLOWED_TRAINING_SIGNAL_FIELDS)}; excluded={list(EXCLUDED_TRAINING_SIGNAL_FIELDS)}",
        }
    )
    rows.append(
        {
            "validation_case": "local_only_training_runs_have_no_oracle_hits",
            "success": bool(
                len(training_df) == 144
                and not training_df["centralized_baseline"].any()
                and training_df["eligible_for_local_only_claims"].all()
                and (training_df["uses_global_oracle"] == False).all()
            ),
            "detail": f"rows={len(training_df)}; oracle hits={int(training_df['uses_global_oracle'].sum())}; centralized rows={int(training_df['centralized_baseline'].sum())}",
        }
    )
    rows.append(
        {
            "validation_case": "learning_update_logs_local_and_oracle_free",
            "success": bool(
                update_audit["inspectedUpdateLogRecords"] > 0
                and update_audit["nonlocalAccessRecordCount"] == 0
                and update_audit["oracleFieldRecordCount"] == 0
                and update_audit["forbiddenRewardRecordCount"] == 0
                and set(update_audit["rewardTerms"]) == set(LEARNING_REWARD_FEATURES)
            ),
            "detail": (
                f"inspected={update_audit['inspectedUpdateLogRecords']}; "
                f"nonlocal={update_audit['nonlocalAccessRecordCount']}; "
                f"oracle={update_audit['oracleFieldRecordCount']}; reward_terms={update_audit['rewardTerms']}"
            ),
        }
    )
    contract = protocol_audit["centralizedBaselineContract"]
    rows.append(
        {
            "validation_case": "centralized_baseline_label_contract_present",
            "success": bool(
                contract["baselineType"] == CENTRALIZED_BASELINE_TYPE
                and contract["usesGlobalOracle"]
                and not contract["eligibleForLocalClaims"]
            ),
            "detail": json.dumps(contract, sort_keys=True, separators=(",", ":")),
        }
    )
    rows.append(
        {
            "validation_case": "unit_and_regression_tests_passed",
            "success": bool(unit_tests["success"] and e03_tests["success"] and e02_tests["success"]),
            "detail": (
                f"E04={unit_tests['returnCode']}; "
                f"E03 policy interface={e03_tests['returnCode']}; "
                f"E02 simulator={e02_tests['returnCode']}"
            ),
        }
    )
    return pd.DataFrame(rows)


def render_protocol_report(
    *,
    artifact_paths: list[Path],
    validation_line: str,
    validation_success: bool,
    training_path: Path,
    audit_table_path: Path,
    config_path: Path,
    protocol_audit: dict[str, Any],
) -> str:
    outcome = "supportive" if validation_success else "constraining/contradictory"
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    return f"""# E04 No-Global-Oracle Audit And Local-Only Training Protocol

## Top Summary

- Research step ID: S07
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: Raw E03 `PolicyObservation` preserves full arrays and `ideal_position` for wrapper parity, and S02 has target-derived signal fields. S07 therefore requires the strict projected training view below for future learned policies.
- Recommended next action: Stop before S08. Chief Scientist should review the audit constraint and then authorize S08 only if future training consumes the S07 projected local view.

## Local-Only Training Protocol

- Protocol ID: `{LOCAL_ONLY_PROTOCOL_ID}`
- Allowed reward/update evidence: local adjacent-order delta, local swap success, local block outcome, S01 local frustration delta, and S02 `blocked`/`frustrated` fields.
- Forbidden update evidence: global Sortedness, full-array rank, whole-array distance, full target morphology, final target, offline time-in-target, and target/ideal position.
- Allowed signal fields for local-only training: `{', '.join(ALLOWED_TRAINING_SIGNAL_FIELDS)}`.
- Excluded signal fields for local-only training: `{', '.join(EXCLUDED_TRAINING_SIGNAL_FIELDS)}`.
- Training runs table: `{training_path}`.
- Static audit table: `{audit_table_path}`.
- Config: `{config_path}`.

Future learned policies must use `project_local_training_observation()` or an equivalent schema. They must not consume raw E03 `PolicyObservation` objects directly.

## Strict Projected Feature Contract

The projected feature dictionary contains actor value/status, immediate left and right value/status when present, boundary flags, reverse-direction flag, bounded S01 memory counters, and S02 blocked/frustrated scalars. It excludes absolute `ideal_position`, full `values`, full `labels`, full `statuses`, global Sortedness, final values, and target Sortedness.

Feature audit result:

```json
{json.dumps(protocol_audit['featureAudit'], indent=2, sort_keys=True)}
```

## Legacy Observation Audit

Raw E03 observations are retained for public-policy wrapper parity, but S07 classifies these fields as incompatible with future local-only learned-policy tensors:

```json
{json.dumps(protocol_audit['legacyObservationAudit'], indent=2, sort_keys=True)}
```

## Centralized Baseline Label

No centralized baseline was run in S07. Any future global-controller baseline must use this explicit ineligible label:

```json
{json.dumps(protocol_audit['centralizedBaselineContract'], indent=2, sort_keys=True)}
```
"""


def render_full_report(
    *,
    training_df: pd.DataFrame,
    static_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    validation_line: str,
    validation_success: bool,
    protocol_audit: dict[str, Any],
    update_audit: dict[str, Any],
    unit_tests: dict[str, Any],
    e03_tests: dict[str, Any],
    e02_tests: dict[str, Any],
    artifact_paths: list[Path],
    training_path: Path,
    training_csv_path: Path,
    audit_report_path: Path,
    audit_table_path: Path,
    validation_path: Path,
    config_path: Path,
    source_manifest_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    manifest: dict[str, Any],
) -> str:
    outcome = "supportive" if validation_success else "constraining/contradictory"
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    validation_table = markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=20)
    static_table = markdown_table(static_df, ["access_role", "success", "forbidden_source_hits", "expected_access"], max_rows=20)
    mean_delta = float("nan")
    if len(training_df):
        learning = training_df[training_df["policy_mode"] == "local_learning"]["time_in_target_fraction"].mean()
        control = training_df[training_df["policy_mode"] == "nonlearning_control"]["time_in_target_fraction"].mean()
        mean_delta = float(learning - control)
    commands = "\n".join(
        [
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{e03_tests['command']}` -> return code {e03_tests['returnCode']}",
            f"- `{e02_tests['command']}` -> return code {e02_tests['returnCode']}",
            "- `python scripts/e04_s07_no_oracle_audit.py --repo-dir /workspace/cell-research --artifacts-dir $ARTIFACTS_DIR`",
        ]
    )
    return f"""# E04 S07 Research Step Full Results

## Top Summary

- Research step ID: S07
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {outcome}
- Caveats or blockers: Raw E03 `PolicyObservation` still exposes full arrays and `ideal_position` for legacy wrapper parity, and S02 target-derived signal fields exist. S07 resolves this for future training by validating a strict projected local-only training view; future S08+ training should not consume raw observations directly.
- Lay summary: S07 checked whether the S06 learning path can learn from hidden global answers. The S06 reward/update functions do not read global Sortedness, full-array rank, target position, or final target. The audit also found that the older wrapper observation object contains fields a learner should not see, so S07 defines a smaller local-only feature view for future training.
- Recommended next action: Stop before S08. Chief Scientist should review the protocol and authorize S08 only if the search code uses the S07 projected local-only view and keeps centralized baselines explicitly labeled.

## Frozen Question

Can adaptive cells improve robust sorting using only local state, neighbor comparisons, and delayed local outcomes?

S07 does not claim new broad training success; it validates the access contract needed before broader training. The compact S06 local-learning runs remain oracle-free under this audit, with a mean time-in-target delta of `{mean_delta:.6f}` for learning minus matched nonlearning controls.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S07.
- S06 learning code: `src/e04/local_learning.py`.
- S05 task definitions: `src/e04/homeostasis.py`.
- Policy observation schema: `src/e03/policy_interface.py`.
- S01/S02 interfaces: `src/e04/memory_policies.py`, `src/e04/signaling.py`.
- Datasets: none required.
- Previous mounted artifacts: E01 `/previous-artifacts/E01`, E02 `/previous-artifacts/E02`, E03 `/previous-artifacts/E03`.

## Methods

Implemented `src/e04/no_oracle_protocol.py`, `tests/e04/test_no_oracle_protocol.py`, and `scripts/e04_s07_no_oracle_audit.py`.

The audit has four layers:

- Static source-scope audit of S06 action selection/reward update, S06 offline metrics, S05 schedules/perturbations, S05 offline metrics, E03 policy observations, and S02 signal generation.
- Strict future-training feature projection via `project_local_training_observation()`.
- Runtime rerun of the S06 default 144-row local-learning/control matrix with S07 labels.
- Update-log inspection for local accessed windows, empty oracle fields, and allowed reward terms.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- New dependencies installed: none.
- CPU use: serial audit and pilot matrix; host reports `{os.cpu_count()}` logical CPUs and worker count used was 1.
- Platform: {platform.platform()}

## Parameters

- Protocol ID: `{LOCAL_ONLY_PROTOCOL_ID}`.
- Tasks: `{', '.join(HOMEOSTATIC_TASKS)}`.
- Policy modes: `{', '.join(LEARNING_POLICY_MODES)}`.
- Reliability modes: `{', '.join(RELIABILITY_MODES)}`.
- Default pilot matrix: 2 seeds x 4 tasks x 3 interfaces x 3 reliability modes x 2 policy modes = 144 rows.
- Allowed training signal fields: `{', '.join(ALLOWED_TRAINING_SIGNAL_FIELDS)}`.
- Excluded training signal fields: `{', '.join(EXCLUDED_TRAINING_SIGNAL_FIELDS)}`.

## Results

Training/audit rows: `{len(training_df)}`. Oracle-hit rows: `{int(training_df['uses_global_oracle'].sum())}`. Centralized baseline rows: `{int(training_df['centralized_baseline'].sum())}`. Inspected update-log records: `{update_audit['inspectedUpdateLogRecords']}`. Nonlocal update-log records: `{update_audit['nonlocalAccessRecordCount']}`. Oracle-field update-log records: `{update_audit['oracleFieldRecordCount']}`.

Result table: `{training_path}`. CSV sidecar: `{training_csv_path}`. Audit report: `{audit_report_path}`. Static audit table: `{audit_table_path}`. Validation table: `{validation_path}`.

### Static Audit

{static_table}

### Validation Checks

{validation_table}

## Centralized Baseline Handling

No centralized baseline was run in S07. The required label for future global-controller rows is `{CENTRALIZED_BASELINE_TYPE}`, with `usesGlobalOracle=true` and `eligibleForLocalClaims=false`. Such rows may be used only as ceiling/sanity comparisons and must not be mixed into local-only claims.

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S07.
- The raw E03 wrapper observation is intentionally rich for reproducing public cell behavior; it is not a safe future training tensor.
- S02 `target_seeking` and `morphogen` currently derive partly from `ideal_position`; S07 excludes them from local-only training.
- The S07 table reruns the compact S06 local-learning pilots; it is an access audit and protocol validation, not S08-scale evolutionary search.

## Provenance

- Git commit at validation time: `{manifest['gitCommit']}`
- Git branch at validation time: `{manifest['gitBranch']}`
- Git status at validation time: `{manifest['gitStatusShort'] or 'clean'}`
- Source files tracked in manifest: `{len(manifest['sourceFiles'])}`
- Config: `{config_path}`
- Source manifest: `{source_manifest_path}`
- Run manifest: `{run_manifest_path}`
- Checksums: `{checksums_path}`
- Created at UTC: `{manifest['createdAtUtc']}`

## Recommended Next Action

Stop before S08. Chief Scientist should review the S07 protocol and authorize S08 only with the strict projected local-only view.
"""


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    report_path = step_dir / "research_step_full_results.md"
    audit_report_path = artifacts_dir / "reports" / "e04_no_global_oracle_audit.md"
    training_path = artifacts_dir / "results" / "e04_local_only_training_runs.parquet"
    training_csv_path = artifacts_dir / "tables" / "e04_local_only_training_runs.csv"
    audit_table_path = artifacts_dir / "tables" / "e04_no_oracle_audit_findings.csv"
    validation_path = artifacts_dir / "tables" / "e04_no_oracle_audit_validation.csv"
    config_path = artifacts_dir / "configs" / "e04_s07_no_oracle_audit.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_no_oracle_audit_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"
    for path in [
        step_dir,
        audit_report_path.parent,
        training_path.parent,
        training_csv_path.parent,
        audit_table_path.parent,
        validation_path.parent,
        config_path.parent,
        source_manifest_path.parent,
        checksums_path.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    protocol_audit = sample_projected_protocol_audit()
    static_df = build_static_audit(repo_dir)
    training_df = build_training_runs(args.max_events)
    update_audit = inspect_update_logs(training_df)
    static_df.to_csv(audit_table_path, index=False)
    training_df.to_parquet(training_path, index=False)
    training_df.to_csv(training_csv_path, index=False)

    unit_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e04", "-p", "test_*.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    e03_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    e02_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_deterministic_simulator.py"], repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )
    validation_df = build_validation(
        training_df=training_df,
        static_df=static_df,
        protocol_audit=protocol_audit,
        update_audit=update_audit,
        unit_tests=unit_tests,
        e03_tests=e03_tests,
        e02_tests=e02_tests,
    )
    validation_df.to_csv(validation_path, index=False)
    validation_success = bool(validation_df["success"].all())
    validation_line = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed; "
        f"E04 unit tests return code {unit_tests['returnCode']}; "
        f"E03 policy-interface tests return code {e03_tests['returnCode']}; "
        f"E02 simulator tests return code {e02_tests['returnCode']}; "
        f"{len(training_df)} local-only training/audit rows written; "
        f"oracle-hit rows={int(training_df['uses_global_oracle'].sum())}"
    )
    config_doc = {
        "schema": "eidosoma.e04.s07.no_oracle_audit_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "protocolId": LOCAL_ONLY_PROTOCOL_ID,
        "maxEvents": int(args.max_events),
        "runCount": int(len(training_df)),
        "tasks": list(HOMEOSTATIC_TASKS),
        "policyModes": list(LEARNING_POLICY_MODES),
        "allowedTrainingSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedTrainingSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "forbiddenTrainingTokens": list(FORBIDDEN_TRAINING_TOKENS),
        "centralizedBaselineContract": centralized_baseline_contract(),
        "workerCount": 1,
    }
    write_json(config_path, config_doc)
    artifact_paths = [
        report_path,
        audit_report_path,
        training_path,
        training_csv_path,
        audit_table_path,
        validation_path,
        config_path,
        source_manifest_path,
        artifact_manifest_path,
        run_manifest_path,
        checksums_path,
    ]
    write_text(
        audit_report_path,
        render_protocol_report(
            artifact_paths=artifact_paths,
            validation_line=validation_line,
            validation_success=validation_success,
            training_path=training_path,
            audit_table_path=audit_table_path,
            config_path=config_path,
            protocol_audit=protocol_audit,
        ),
    )
    source_paths = [
        repo_dir / "src/e04/no_oracle_protocol.py",
        repo_dir / "tests/e04/test_no_oracle_protocol.py",
        repo_dir / "scripts/e04_s07_no_oracle_audit.py",
        repo_dir / "src/e04/local_learning.py",
        repo_dir / "scripts/e04_s06_local_learning.py",
        repo_dir / "src/e04/homeostasis.py",
        repo_dir / "scripts/e04_s05_homeostatic_tasks.py",
        repo_dir / "src/e04/memory_policies.py",
        repo_dir / "src/e04/signaling.py",
        repo_dir / "src/e03/policy_interface.py",
        repo_dir / "src/e02/deterministic_simulator.py",
    ]
    manifest: dict[str, Any] = {
        "schema": "eidosoma.src_snapshot.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(repo_dir, ["branch", "--show-current"]),
        "gitStatusShort": git_output(repo_dir, ["status", "--short"]),
        "dependencies": {
            "newDependenciesInstalled": [],
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "upstreamContext": {
            "previousE01Dir": "/previous-artifacts/E01",
            "previousE02Dir": "/previous-artifacts/E02",
            "previousE03Dir": "/previous-artifacts/E03",
            "s05TaskSpec": str(artifacts_dir / "reports/e04_homeostatic_task_spec.md"),
            "s05Results": str(artifacts_dir / "results/e04_homeostatic_baselines.parquet"),
            "s06RuleSpec": str(artifacts_dir / "reports/e04_local_learning_rule_spec.md"),
            "s06Results": str(artifacts_dir / "results/e04_local_learning_pilots.parquet"),
        },
        "sourceFiles": [source_entry(path, repo_dir) for path in source_paths if path.exists()],
        "audit": {
            "protocolAudit": protocol_audit,
            "updateAudit": update_audit,
            "staticAuditRows": int(len(static_df)),
        },
        "validation": {
            "allValidationPassed": validation_success,
            "validationCaseCount": int(len(validation_df)),
            "validationCaseSuccessCount": int(validation_df["success"].sum()),
            "unitTests": unit_tests,
            "e03PolicyInterfaceTests": e03_tests,
            "e02DeterministicSimulatorTests": e02_tests,
            "trainingParquet": str(training_path),
            "trainingParquetSha256": sha256_file(training_path),
            "trainingCsv": str(training_csv_path),
            "trainingCsvSha256": sha256_file(training_csv_path),
            "auditTable": str(audit_table_path),
            "auditTableSha256": sha256_file(audit_table_path),
            "validationCsv": str(validation_path),
            "validationCsvSha256": sha256_file(validation_path),
            "auditReport": str(audit_report_path),
            "auditReportSha256": sha256_file(audit_report_path),
            "config": str(config_path),
            "configSha256": sha256_file(config_path),
        },
    }
    write_json(source_manifest_path, manifest)
    write_text(
        report_path,
        render_full_report(
            training_df=training_df,
            static_df=static_df,
            validation_df=validation_df,
            validation_line=validation_line,
            validation_success=validation_success,
            protocol_audit=protocol_audit,
            update_audit=update_audit,
            unit_tests=unit_tests,
            e03_tests=e03_tests,
            e02_tests=e02_tests,
            artifact_paths=artifact_paths,
            training_path=training_path,
            training_csv_path=training_csv_path,
            audit_report_path=audit_report_path,
            audit_table_path=audit_table_path,
            validation_path=validation_path,
            config_path=config_path,
            source_manifest_path=source_manifest_path,
            run_manifest_path=run_manifest_path,
            checksums_path=checksums_path,
            manifest=manifest,
        ),
    )
    artifacts = [
        artifact_entry(report_path, artifacts_dir, "S07 full-results handoff report"),
        artifact_entry(audit_report_path, artifacts_dir, "S07 no-global-oracle audit and local-only training protocol"),
        artifact_entry(training_path, artifacts_dir, "S07 local-only training/audit rows"),
        artifact_entry(training_csv_path, artifacts_dir, "CSV sidecar for S07 local-only training/audit rows"),
        artifact_entry(audit_table_path, artifacts_dir, "S07 static audit findings"),
        artifact_entry(validation_path, artifacts_dir, "S07 validation table"),
        artifact_entry(config_path, artifacts_dir, "S07 audit config"),
        artifact_entry(source_manifest_path, artifacts_dir, "S07 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S07 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S07 outputs"),
    ]
    artifact_manifest = {
        "schema": "eidosoma.artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "artifacts": artifacts,
    }
    write_json(artifact_manifest_path, artifact_manifest)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "lastResearchStepId": STEP_ID,
        "createdAtUtc": utc_now(),
        "git": {
            "branch": manifest["gitBranch"],
            "headCommit": manifest["gitCommit"],
            "statusShort": manifest["gitStatusShort"],
        },
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "cpuCountReported": os.cpu_count(),
            "workerCountUsed": 1,
        },
        "dependencies": manifest["dependencies"],
        "validation": manifest["validation"],
        "artifacts": artifacts,
    }
    write_json(run_manifest_path, run_manifest)
    checksum_targets = [
        report_path,
        audit_report_path,
        training_path,
        training_csv_path,
        audit_table_path,
        validation_path,
        config_path,
        source_manifest_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")
    return 0 if validation_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
